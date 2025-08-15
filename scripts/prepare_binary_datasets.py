"""
Extract and organize datasets.
"""

from argparse import ArgumentParser
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
import hashlib
from itertools import batched
from itertools import islice
from itertools import repeat
from io import BytesIO
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional
import zipfile
import zlib

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
import lief
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.binanal import rearm_disarmed_binary


MAGIC = {
    "pe": (b"MZ",),
    "elf": (b"\x7fELF",),
    "zlib": (b"\x78\x01", b"\x78\x5E", b"\x78\x9C", b"\x78\xDA"),
}


TMPDIR = "./tmp"


StrPath = str | os.PathLike[str]


def get_sample_path(sha: str, root: Path, depth: int = 2) -> Path:
    path = root
    for i in range(depth):
        path /= sha[i]
    path /= sha
    return path


def create_sample_paths(root: Path, depth: int = 2) -> None:
    path = root
    path.mkdir(exist_ok=True)
    if depth == 0:
        return

    for d in "0123456789abcdef":
        p = path / d
        p.mkdir(exist_ok=True)
        create_sample_paths(p, depth - 1)


def download_file(url: str, outfile: StrPath, chunk_size: int = 4096) -> None:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(outfile, "wb") as fp:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    fp.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def concatenate_files_to_file(files: Sequence[StrPath], outfile: StrPath) -> None:
    with open(outfile, "a") as fp_w:
        for f in files:
            with open(f, "r") as fp_r:
                fp_w.write(fp_r.read())


class PrepareAssemblage:

    PUBLIC_URL_PE = "https://assemblage-lps.s3.us-west-1.amazonaws.com/public/winpe_licensed.zip"
    PUBLIC_URL_ELF = "https://assemblage-lps.s3.us-west-1.amazonaws.com/public/licensed_linux.zip"

    def __init__(
        self,
        root: Path,
        indexfile: Path,
        magic: Iterable[bytes],
        url: Optional[str],
        archive: Optional[Path],
        num_samples: Optional[int] = None,
        num_workers: int = 0,
        verbose: bool = False,
        progress: bool = True,
        quiet: bool = False,
    ) -> None:
        self.root = root
        self.indexfile = indexfile
        self.magic = magic
        self.url = url
        self.archive = archive
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.verbose = verbose
        self.progress = progress
        self.quiet = quiet
        self.disable_tqdm = not progress or quiet or (num_workers > 0)

    def __call__(self) -> None:
        create_sample_paths(self.root)
        if self.archive is not None:
            self.run(self.archive)
        elif self.url is not None:
            with tempfile.NamedTemporaryFile(dir=TMPDIR, delete=False) as tmp_file:
                archive = Path(tmp_file.name)
                download_file(self.url, archive)
                self.run(archive)

    def run(self, archive: Path) -> None:
        if self.num_workers == 0:
            self.extract(archive, self.indexfile, None, self.num_samples)
            return

        indexfiles = [tempfile.mkstemp(prefix=self.indexfile.stem, suffix=f"-{i}.txt")[1] for i in range(self.num_workers)]
        print(f"Using {self.num_workers} workers, index files: {indexfiles}")
        with zipfile.ZipFile(archive, "r") as zip_ref:
            names = zip_ref.namelist()
        batches = list(batched(names, math.ceil(len(names) / self.num_workers)))
        iterable = zip(
            repeat(archive, self.num_workers),
            indexfiles,
            batches,
            repeat(math.ceil(self.num_samples / self.num_workers) if self.num_samples else None, self.num_workers),
            strict=True
        )
        with mp.Pool(self.num_workers) as pool:
            pool.starmap(self.extract, iterable)
        concatenate_files_to_file(indexfiles, self.indexfile)

    def extract(self, archive: Path, indexfile: Path, names: Optional[Iterable[str]], num_samples: Optional[int] = None) -> None:
        count = 0
        with zipfile.ZipFile(archive, "r") as zip_ref, open(indexfile, "a") as index_fp:
            names = names if names is not None else zip_ref.namelist()
            for file in tqdm(names, desc="Extracting...", disable=self.disable_tqdm):
                b = zip_ref.read(file)

                if len(b) == 0 or file.endswith(".pdb"):
                    continue
                if not any(b.startswith(m) for m in self.magic):
                    if not self.quiet:
                        print(f"Skipping {file} (Unexpected magic {b[0:8].decode()})")
                    continue

                sha = hashlib.sha256(b).hexdigest()
                outfile = get_sample_path(sha, self.root)
                if outfile.exists():
                    if not self.quiet:
                        print(f"Skipping {file} (SHA already exists {sha})")
                    continue
                outfile.write_bytes(b)
                index_fp.write(f"{sha} {0}\n")
                if self.verbose:
                    print(f"Extracted {file}")

                count += 1
                if num_samples is not None and count == num_samples:
                    break


class PrepareSorel:

    SOREL_BUCKET = "sorel-20m"
    SOREL_PREFIX = "09-DEC-2020/binaries/"

    def __init__(
        self,
        root: Path,
        indexfile: Path,
        num_samples: Optional[int] = None,
        num_workers: int = 0,
        verbose: bool = False,
        progress: bool = True,
        quiet: bool = False,
    ) -> None:
        self.root = root
        self.indexfile = indexfile
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.verbose = verbose
        self.progress = progress
        self.quiet = quiet

    def __call__(self) -> None:
        lief.logging.disable()
        create_sample_paths(self.root)
        shas = list(islice(self.namelist(), self.num_samples))
        if len(set(shas)) != len(shas):
            raise RuntimeError("Duplicate SHAs found in Sorel namelist.")
        if self.num_workers == 0:
            self.download(shas, self.indexfile)
        elif self.num_workers > 0:
            indexfiles = [tempfile.mkstemp(prefix=self.indexfile.stem, suffix=f"-{i}.txt")[1] for i in range(self.num_workers)]
            print(f"Using {self.num_workers} workers, index files: {indexfiles}")
            batches = list(batched(shas, math.ceil(len(shas) / self.num_workers)))
            with mp.Pool(self.num_workers) as pool:
                pool.starmap(self.download, zip(batches, indexfiles))
            concatenate_files_to_file(indexfiles, self.indexfile)

    def download(self, shas: Sequence[str], indexfile: Path) -> None:
        if not isinstance(shas, Sequence):
            raise TypeError(f"Expected shas to be a Sequence, got {type(shas)}.")

        iterable = zip(shas, self.stream(shas), strict=True)
        iterable = tqdm(iterable, total=len(shas), desc="Downloading...", disable=not self.progress or self.num_workers > 0)

        with open(indexfile, "a") as index_fp:
            for s, b in iterable:
                if b is None:
                    continue

                outfile = get_sample_path(s, self.root)
                if outfile.exists():
                    if not self.quiet:
                        print(f"Skipping {s} (SHA already exists)")
                    continue

                outfile.write_bytes(b)
                index_fp.write(f"{s} {1}\n")
                if self.verbose:
                    print(f"Extracted {s}")

    def stream(self, shas: Iterable[str]) -> Generator[Optional[bytes], None, None]:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        for s in shas:
            buffer = BytesIO()
            try:
                s3.download_fileobj(PrepareSorel.SOREL_BUCKET, PrepareSorel.SOREL_PREFIX + s, buffer)
            except ClientError as err:
                if not self.quiet:
                    print(f"Skipping {s} ({err.__class__.__name__}: {err})")
                yield None
                continue

            buffer.seek(0)
            b = buffer.read()

            try:
                b = zlib.decompress(b)
            except zlib.error as err:
                if not self.quiet:
                    print(f"Skipping {s} ({err.__class__.__name__}: {err})")
                yield None
                continue

            if len(b) == 0:
                if not self.quiet:
                    print(f"Skipping {s} (Empty sample)")
                yield None
                continue

            if not any(b.startswith(m) for m in MAGIC["pe"]):
                if not self.quiet:
                    print(f"Skipping {s} (Unexpected magic {b[0:8].decode()})")
                yield None
                continue

            try:
                b = rearm_disarmed_binary(b, s)
            except RuntimeError as err:
                if not self.quiet:
                    print(f"Skipping {s} ({err.__class__.__name__}: {err})")
                yield None
                continue

            yield b

    def namelist(self) -> Generator[str, None, None]:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=PrepareSorel.SOREL_BUCKET, Prefix=PrepareSorel.SOREL_PREFIX)

        for page in page_iterator:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                sha = key.split("/")[-1]
                yield sha


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--assemblage", action="store_true")
    parser.add_argument("--sorel", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.assemblage:
        PrepareAssemblage(
            Path("./data/ass"),
            Path("./data/ass.txt"),
            MAGIC["pe"],
            None,
            Path("./tmp/WindowsBinaries.zip"),
            num_samples=100000,
            num_workers=8,
            verbose=args.verbose,
            progress=args.progress,
            quiet=args.quiet,
        )()

    if args.sorel:
        PrepareSorel(
            Path("./data/sor"),
            Path("./data/sor.txt"),
            num_samples=100000,
            num_workers=8,
            verbose=args.verbose,
            progress=args.progress,
            quiet=args.quiet,
        )()


if __name__ == "__main__":
    main()

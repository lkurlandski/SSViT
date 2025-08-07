"""
Extract and organize datasets.
"""

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Optional
import zipfile

import requests
from tqdm import tqdm


MAGIC = {
    "pe": b"MZ",
    "elf": b"\x7fELF",
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


class PrepareAssemblage:

    PUBLIC_URL_PE = "https://assemblage-lps.s3.us-west-1.amazonaws.com/public/winpe_licensed.zip"
    PUBLIC_URL_ELF = "https://assemblage-lps.s3.us-west-1.amazonaws.com/public/licensed_linux.zip"

    def __init__(self, root: Path, indexfile: Path, magic: bytes, url: Optional[str], archive: Optional[Path], verbose: bool = True, progress: bool = False) -> None:
        self.root = root
        self.indexfile = indexfile
        self.magic = magic
        self.url = url
        self.archive = archive
        self.verbose = verbose
        self.progress = progress

    def __call__(self) -> None:
        create_sample_paths(self.root)
        if self.url is not None:
            with tempfile.NamedTemporaryFile(dir=TMPDIR, delete=False) as tmp_file:
                file = Path(tmp_file.name)
                download_file(self.url, file)
            self.extract(file, MAGIC["pe"])
        elif self.archive is not None:
            self.extract(self.archive, self.magic)

    def extract(self, archive: Path, magic: bytes) -> None:
        with zipfile.ZipFile(archive, "r") as zip_ref, open(self.indexfile, "a") as index_fp:
            for file in tqdm(zip_ref.namelist(), desc="Extracting", disable=not self.progress):
                b = zip_ref.read(file)
                if len(b) == 0:
                    continue
                if not b.startswith(magic):
                    if self.verbose:
                        print(f"Skipping {file} (Unexpected magic {b[0:8].decode()}).")
                    continue
                sha = hashlib.sha256(b).hexdigest()
                outfile = get_sample_path(sha, self.root)
                if outfile.exists():
                    if self.verbose:
                        print(f"Skipping {file} (SHA already exists {sha}).")
                    continue
                outfile.write_bytes(b)
                index_fp.write(f"{sha} {0}\n")
                if self.verbose:
                    print(f"Extracted {file}.")


def main() -> None:

    # PrepareAssemblage(Path("./data/binaries"), MAGIC["pe"], PrepareAssemblage.PUBLIC_URL_PE, None, verbose=False, progress=True)()
    PrepareAssemblage(Path("./data/binaries"), Path("./data/index.txt"), MAGIC["pe"], None, Path("/home/lk3591/Documents/datasets/Assemblage/assemblage.zip"), verbose=False, progress=True)()


if __name__ == "__main__":
    main()

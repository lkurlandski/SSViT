"""
Extract and organize datasets.
"""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from argparse import ArgumentParser
from collections.abc import Generator
from collections.abc import Iterable
import hashlib
from itertools import batched
from itertools import chain
from itertools import islice
from io import BytesIO
import math
import multiprocessing as mp
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
import os
from pathlib import Path
import queue
import sys
from typing import Any
from typing import NamedTuple
from typing import Optional
import zipfile
import zlib

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import rearm_disarmed_binary
from src.simpledb import CreateSimpleDB
from src.simpledb import CreateSimpleDBSample


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


_SENTINEL = object()


def _producer_worker(streamer: DatasetStreamer, names: Iterable[str], out_q: mp.Queue[Sample | object], stop_event: Event, worker_idx: int) -> None:
    try:
        for sample in streamer.stream(names):
            if stop_event.is_set():
                break
            # Don't block forever if the consumer stops early (e.g., islice).
            while not stop_event.is_set():
                try:
                    out_q.put(sample, timeout=0.2)
                    break
                except queue.Full:
                    continue
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"_producer_worker [worker {worker_idx}] {e}")
    finally:
        # Best effort to notify completion without hanging if the queue is gone.
        try:
            out_q.put(_SENTINEL, timeout=0.2)
        except Exception:
            pass
        # Let the child drop its handle promptly to avoid resource_tracker warnings.
        try:
            out_q.close()
            out_q.cancel_join_thread()
        except Exception:
            pass


class Sample(NamedTuple):
    sha: str
    data: bytes
    malware: bool


class DatasetStreamer(ABC):
    """
    Fast streaming interface for datasets using multi-producer single-consumer pattern.
    """

    def __init__(self, num_workers: int = 0, verbose: bool = True, progress: bool = False, quiet: bool = False) -> None:
        self.num_workers = num_workers
        self.verbose = verbose
        self.progress = progress
        self.quiet = quiet
        self.disable_tqdm = not progress or num_workers > 0

    def __iter__(self) -> Generator[Sample, None, None]:
        if self.num_workers == 0:
            yield from self.stream(None)
            return

        # Get the list of all available sample names and handle the split across workers.
        names = list(self.namelist())
        batch_size = math.ceil(len(names) / max(1, self.num_workers))
        batches = list(batched(names, batch_size))

        # Create a spawn context for safety (works well across platforms).
        ctx = mp.get_context("spawn")

        # Bounded queue: small buffer for backpressure. We use timeouts to avoid deadlocks.
        out_q = ctx.Queue(max(2, self.num_workers * 2))

        # Stop flag to tell producers to stop early if the consumer exits (e.g., due to islice).
        stop_event = ctx.Event()

        # Start workers (non-daemon so they can flush/exit cleanly).
        procs: list[SpawnProcess] = []
        for i, chunk in enumerate(batches):
            p = ctx.Process(
                target=_producer_worker,
                args=(self, chunk, out_q, stop_event, i),
                daemon=False,
            )
            p.start()
            procs.append(p)

        finished = 0
        try:
            # Greedy single-consumer yields as items arrive and stops after all sentinels.
            while finished < len(procs):
                try:
                    item = out_q.get(timeout=0.2)  # short timeout lets us react to early shutdown
                except queue.Empty:
                    # Periodically loop to notice if workers exited or a stop was requested
                    continue

                if item is _SENTINEL:
                    finished += 1
                    continue

                # Normal path: yield a sample
                yield item

        except GeneratorExit:
            # The upstream consumer stopped (e.g., islice took N items).
            # Signal producers to stop generating new work.
            stop_event.set()
            raise
        finally:
            # Always signal stop (idempotent) and drain any remaining items to unblock puts.
            stop_event.set()
            try:
                while True:
                    out_q.get_nowait()
            except queue.Empty:
                pass
            except Exception:
                pass

            # Close and join the queue feeder thread cleanly
            try:
                out_q.close()
            except Exception:
                pass
            try:
                out_q.join_thread()
            except Exception:
                pass

            # Let workers exit on their own; avoid terminate() unless they hang.
            for p in procs:
                p.join(timeout=5)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                if p.is_alive():
                    p.join(timeout=2)

    @abstractmethod
    def stream(self, names: Optional[Iterable[str]]) -> Generator[Sample, None, None]:
        """Streams valid samples from the dataset. If names is None, streams all available samples."""
        ...

    @abstractmethod
    def namelist(self) -> Generator[str, None, None]:
        """Yields the names of all available samples."""
        ...


class AssemblageStreamer(DatasetStreamer):

    PUBLIC_URL_PE = "https://assemblage-lps.s3.us-west-1.amazonaws.com/public/winpe_licensed.zip"
    PUBLIC_URL_ELF = "https://assemblage-lps.s3.us-west-1.amazonaws.com/public/licensed_linux.zip"

    def __init__(self, magic: Iterable[bytes], url: Optional[str] = None, archive: Path = Path(TMPDIR) / "Assemblage.zip", **kwds: Any) -> None:
        super().__init__(**kwds)

        if not archive.exists():
            if url is None:
                raise ValueError("Either url or an existing archive must be provided.")
            download_file(url, archive)

        self.magic = magic
        self.url = url
        self.archive = archive

    def stream(self, names: Optional[Iterable[str]]) -> Generator[Sample, None, None]:
        """
        Args:
            names: Iterable of file names to stream. If None, streams all available samples.
        """
        names = self.namelist() if names is None else names
        with zipfile.ZipFile(self.archive, "r") as zip_ref:
            for name in tqdm(names, desc="Extracting...", disable=self.disable_tqdm):
                b = zip_ref.read(name)
                if not any(b.startswith(m) for m in self.magic):
                    if not self.quiet: print(f"Skipping {name} (Unexpected magic {b[0:8].decode()})")
                    continue
                sha = hashlib.sha256(b).hexdigest()
                yield Sample(sha, b, False)

    def namelist(self) -> Generator[str, None, None]:
        with zipfile.ZipFile(self.archive, "r") as zip_ref:
            for name in zip_ref.namelist():
                info = zip_ref.getinfo(name)
                if info.is_dir():
                    continue
                if name.endswith(".pdb"):
                    continue
                if info.file_size == 0:
                    continue
                yield name


class SorelStreamer(DatasetStreamer):

    SOREL_BUCKET = "sorel-20m"
    SOREL_PREFIX = "09-DEC-2020/binaries/"

    def stream(self, names: Optional[Iterable[str]]) -> Generator[Sample, None, None]:
        """
        Args:
            names: Iterable of SHA256 names to stream. If None, streams all available samples.
        """
        names = self.namelist() if names is None else names

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        for s in names:
            buffer = BytesIO()
            try:
                s3.download_fileobj(SorelStreamer.SOREL_BUCKET, SorelStreamer.SOREL_PREFIX + s, buffer)
            except ClientError as err:
                if not self.quiet: print(f"Skipping {s} ({err.__class__.__name__}: {err})")
                continue

            buffer.seek(0)
            b = buffer.read()

            try:
                b = zlib.decompress(b)
            except zlib.error as err:
                if not self.quiet: print(f"Skipping {s} ({err.__class__.__name__}: {err})")
                continue

            if len(b) == 0:
                if not self.quiet: print(f"Skipping {s} (Empty sample)")
                continue

            if not any(b.startswith(m) for m in MAGIC["pe"]):
                if not self.quiet: print(f"Skipping {s} (Unexpected magic {b[0:8].decode()})")
                continue

            try:
                b = rearm_disarmed_binary(b, s)
            except RuntimeError as err:
                if not self.quiet: print(f"Skipping {s} ({err.__class__.__name__}: {err})")
                continue

            yield Sample(s, b, True)

    def namelist(self) -> Generator[str, None, None]:
        if (cachefile := Path("./cache/sorel_namelist.txt")).exists():
            with open(cachefile, "r") as fp:
                for line in fp:
                    yield line.strip()
            return

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=SorelStreamer.SOREL_BUCKET, Prefix=SorelStreamer.SOREL_PREFIX)

        for page in page_iterator:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith("/"):
                    continue
                sha = key.split("/")[-1]
                yield sha


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["ass", "sor"], required=True, nargs="+")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--storage", type=str, choices=["flat", "hier", "pack"], required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=sys.maxsize)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--shardsize", type=int, default=2**30)
    parser.add_argument("--hierdepth", type=int, default=2)
    args = parser.parse_args()

    kwds = {
        "num_workers": args.num_workers,
        "verbose": args.verbose,
        "progress": False,
        "quiet": args.quiet,
    }

    streamer: DatasetStreamer
    stream: Iterable[Sample]
    fullstream: Iterable[Sample] = iter(())

    if "ass" in args.dataset:
        streamer = AssemblageStreamer(MAGIC["pe"], archive=Path("./tmp/WindowsBinaries.zip"), **kwds)
        stream = islice(streamer, args.num_samples)
        stream = tqdm(stream, desc="Processing Assemblage...", disable=not args.progress)
        fullstream = chain(fullstream, stream)

    if "sor" in args.dataset:
        streamer = SorelStreamer(**kwds)
        stream = islice(streamer, args.num_samples)
        stream = tqdm(stream, desc="Processing Sorel...", disable=not args.progress)
        fullstream = chain(fullstream, stream)

    if args.storage in ("flat", "hier"):
        indexfile = args.root / "index.txt"
        binaries: Path = args.root / "binaries"
        binaries.mkdir(parents=True, exist_ok=True)
        if args.storage == "hier":
            create_sample_paths(binaries, depth=args.hierdepth)
        with open(indexfile, "w") as fp:
            for sample in fullstream:
                outpath = binaries / sample.sha
                if args.storage == "hier":
                    outpath = get_sample_path(sample.sha, binaries, depth=args.hierdepth)
                outpath.write_bytes(sample.data)
                fp.write(f"{sample.sha} {'1' if sample.malware else '0'}\n")

    if args.storage == "pack":
        creator = CreateSimpleDB(args.root, shardsize=args.shardsize)
        cstream = (CreateSimpleDBSample(sample.sha, sample.data, sample.malware, -1) for sample in fullstream)
        creator(cstream)


if __name__ == "__main__":
    main()

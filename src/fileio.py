"""
Utility functions.
"""

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
from collections.abc import Iterable
from itertools import batched
import os
from typing import Optional


ASYNCH_CHUNK_SIZE = 500000
StrPath = str | os.PathLike[str]


def read_file(f: StrPath, max_length: Optional[int] = None) -> bytes:
    with open(f, "rb") as fp:
        b = fp.read(max_length)
    return b


async def _read_file_asynch(f: StrPath, max_length: Optional[int] = None) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, read_file, f, max_length)


def read_file_asynch(f: StrPath, max_length: Optional[int] = None) -> bytes:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_read_file_asynch(f, max_length))
    else:
        raise RuntimeError("read_file_asynch() cannot be called from an async context.")


def read_files(files: list[StrPath], max_length: Optional[int] = None) -> list[bytes]:
    data = []
    for f in files:
        data.append(read_file(f, max_length))
    return data


async def _read_files_asynch(
    files: list[StrPath], max_length: Optional[int] = None, asynch_chunk_size: int = ASYNCH_CHUNK_SIZE
) -> list[bytes]:
    file_chunks = batched(files, asynch_chunk_size)
    data = []
    for batch_files in file_chunks:
        tasks = [_read_file_asynch(f, max_length) for f in batch_files]
        data.extend(await asyncio.gather(*tasks))
    return data


def read_files_asynch(
    files: list[StrPath], max_length: Optional[int] = None, asynch_chunk_size: int = ASYNCH_CHUNK_SIZE
) -> list[bytes]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_read_files_asynch(files, max_length, asynch_chunk_size))
    else:
        raise RuntimeError("read_files_async() cannot be called from an async context.")


def read_files_lazy(files: Iterable[StrPath], max_length: Optional[int] = None) -> Generator[bytes, None, None]:
    for f in files:
        yield read_file(f, max_length)


async def _read_files_asynch_lazy(
    files: Iterable[StrPath],
    max_length: Optional[int] = None,
    asynch_chunk_size: int = ASYNCH_CHUNK_SIZE,
    greedy: bool = False,
) -> AsyncGenerator[bytes, None]:
    file_chunks = batched(files, asynch_chunk_size)
    for batch_files in file_chunks:
        tasks = [_read_file_asynch(f, max_length) for f in batch_files]
        if greedy:
            for coro in asyncio.as_completed(tasks):
                yield await coro
        else:
            data = await asyncio.gather(*tasks)
            for b in data:
                yield b


def read_files_asynch_lazy(
    files: Iterable[StrPath],
    max_length: Optional[int] = None,
    asynch_chunk_size: int = ASYNCH_CHUNK_SIZE,
    greedy: bool = False,
) -> Generator[bytes, None, None]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        raise RuntimeError("read_files_asynch_lazy() cannot be called from an async context.")

    agen = _read_files_asynch_lazy(files, max_length, asynch_chunk_size)
    try:
        while True:
            try:
                chunk = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            else:
                yield chunk
    finally:
        loop.run_until_complete(agen.aclose())
        loop.close()

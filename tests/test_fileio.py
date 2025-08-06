"""
Tests.
"""

from collections.abc import Iterable, Generator
from os import PathLike
from pathlib import Path
import pytest
import tempfile
from typing import Callable, Optional

from src.fileio import read_file
from src.fileio import read_file_asynch
from src.fileio import read_files
from src.fileio import read_files_asynch
from src.fileio import read_files_lazy
from src.fileio import read_files_asynch_lazy


class TestReadFile:
    def get_random_bytes(self, size: int) -> bytes:
        return bytes([i % 256 for i in range(size)])

    def _test_read_file(self, func: Callable[[bytes, Optional[int]], bytes]):
        with tempfile.NamedTemporaryFile(delete=False, delete_on_close=False) as fp:
            buffer = self.get_random_bytes(1024)
            Path(fp.name).write_bytes(buffer)
            assert func(fp.name) == buffer
            assert func(fp.name, max_length=512) == buffer[:512]

    def test_read_file(self):
        self._test_read_file(read_file)

    def test_read_file_asynch(self):
        self._test_read_file(read_file_asynch)

    def _test_read_files(self, func: Callable[[list[PathLike], Optional[int]], list[bytes]]):
        with tempfile.NamedTemporaryFile() as fp1, tempfile.NamedTemporaryFile() as fp2:
            buffer1 = self.get_random_bytes(1024)
            buffer2 = self.get_random_bytes(2048)
            Path(fp1.name).write_bytes(buffer1)
            Path(fp2.name).write_bytes(buffer2)

            files = [fp1.name, fp2.name]
            data = func(files)
            assert data[0] == buffer1
            assert data[1] == buffer2
            assert len(data) == 2

            data = func(files, max_length=512)
            assert data[0] == buffer1[:512]
            assert data[1] == buffer2[:512]
            assert len(data) == 2

    def test_read_files(self):
        self._test_read_files(read_files)

    def test_read_files_asynch(self):
        self._test_read_files(read_files_asynch)

    def _test_read_files_lazy(self, func: Callable[[Iterable[PathLike], Optional[int]], Generator[bytes, None, None]]):
        with tempfile.NamedTemporaryFile() as fp1, tempfile.NamedTemporaryFile() as fp2:
            buffer1 = self.get_random_bytes(1024)
            buffer2 = self.get_random_bytes(2048)
            Path(fp1.name).write_bytes(buffer1)
            Path(fp2.name).write_bytes(buffer2)

            files = iter([fp1.name, fp2.name])
            stream = func(files)
            assert isinstance(stream, Generator)
            data = list(stream)
            assert data[0] == buffer1
            assert data[1] == buffer2
            assert len(data) == 2

            files = iter([fp1.name, fp2.name])
            stream = func(files, max_length=512)
            assert isinstance(stream, Generator)
            data = list(stream)
            assert data[0] == buffer1[:512]
            assert data[1] == buffer2[:512]
            assert len(data) == 2

    def test_read_files_lazy(self):
        self._test_read_files_lazy(read_files_lazy)

    def test_read_files_asynch_lazy(self):
        self._test_read_files_lazy(read_files_asynch_lazy)

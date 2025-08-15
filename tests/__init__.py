"""
Tests.
"""

from itertools import chain
from pathlib import Path
import os


NUM_FILES = int(os.environ.get("SSVIT_TESTS_NUM_FILES", "8"))
FILES = sorted(
    chain(
        filter(lambda f: f.is_file(), Path("./data/ass").rglob("*")),
        filter(lambda f: f.is_file(), Path("./data/sor").rglob("*")),
    ),
    key=lambda f: f.name
)[0:NUM_FILES]

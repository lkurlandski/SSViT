"""
Tests.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pytest

from src.mpsc import run_mpsc
from src.mpsc import SampleMsg
from src.mpsc import ProducerFn
from src.mpsc import ConsumerFn


# ------------------------------ helpers ------------------------------

def make_payload(n: int) -> bytes:
    # deterministic bytes payload
    return bytes((i % 251 for i in range(n)))


def simple_producer(size_by_item: dict[int, int]) -> ProducerFn:
    def _fn(item: Any) -> Optional[tuple[str, bytes, dict[str, Any]]]:
        size = size_by_item[int(item)]
        payload = make_payload(size)
        name = f"id-{item:06d}"
        meta = {"idx": int(item)}
        return name, payload, meta
    return _fn


def failing_producer(fail_on: set[int]) -> ProducerFn:
    def _fn(item: Any) -> tuple[str, bytes, dict[str, Any]]:
        if int(item) in fail_on:
            raise RuntimeError(f"producer failed on {item}")
        return f"id-{int(item):06d}", make_payload(16), {"idx": int(item)}
    return _fn


def collecting_consumer(collected: list[tuple[str, int, str]]) -> ConsumerFn:
    def _fn(stream: Iterable[SampleMsg]) -> None:
        for msg in stream:
            collected.append((msg.name, msg.length, msg.transport))
    return _fn


def slow_collecting_consumer(collected: list[str], delay_sec: float) -> ConsumerFn:
    import time
    def _fn(stream: Iterable[SampleMsg]) -> None:
        for msg in stream:
            collected.append(msg.name)
            time.sleep(delay_sec)
    return _fn


# ------------------------------ tests ------------------------------


def test_bytes_transport_basic() -> None:
    items = list(range(20))
    sizes = {i: 128 for i in items}
    produced = simple_producer(sizes)

    with mp.Manager() as m:
        seen = m.list()

        run_mpsc(
            items,
            producer_fn=produced,
            consumer_fn=collecting_consumer(seen),  # type: ignore[arg-type]
            num_producers=4,
            data_queue_size=8,
            transport="bytes",
        )

        # Convert to normal set for assertions
        names = {name for name, _, _ in list(seen)}
        assert names == {f"id-{i:06d}" for i in items}
        for _, length, transport in list(seen):
            assert length == 128
            assert transport == "bytes"


def test_shm_transport_mixed_threshold() -> None:
    items = list(range(12))
    sizes = {i: (256 if i % 2 == 0 else 4096) for i in items}
    produced = simple_producer(sizes)

    with mp.Manager() as m:
        seen = m.list()  # shared list visible across processes

        def consumer(stream: Iterable[SampleMsg]) -> None:
            for msg in stream:
                # Store minimal info needed for assertions
                seen.append((msg.name, msg.transport, msg.length))

        run_mpsc(
            items,
            producer_fn=produced,
            consumer_fn=consumer,
            num_producers=3,
            data_queue_size=6,
            transport="shm",
            shm_threshold=1024,
        )

        by_name = {name: (transport, length) for name, transport, length in list(seen)}
        assert set(by_name) == {f"id-{i:06d}" for i in items}
        for i in items:
            transport, length = by_name[f"id-{i:06d}"]
            if sizes[i] <= 1024:
                assert transport == "bytes"
            else:
                assert transport == "shm"
            assert length == sizes[i]


def test_path_transport_and_cleanup(tmp_path: Path) -> None:
    items = list(range(10))
    sizes = {i: 2048 for i in items}  # all reasonably sized
    produced = simple_producer(sizes)

    observed_paths: list[str] = []

    def consumer(stream: Iterable[SampleMsg]) -> None:
        for msg in stream:
            assert msg.transport == "path"
            # Keep a copy of the path to confirm cleanup later
            assert isinstance(msg.data, str)
            observed_paths.append(msg.data)
            # Read a few bytes to ensure file exists while consuming
            with open(msg.data, "rb") as fp:
                head = fp.read(4)
                assert head == make_payload(2048)[:4]

    run_mpsc(
        items,
        producer_fn=produced,
        consumer_fn=consumer,
        num_producers=2,
        data_queue_size=4,
        transport="path",
        tempdir=tmp_path,
    )

    # Confirm temp files have been removed
    for p in observed_paths:
        assert not Path(p).exists()


def test_backpressure_with_slow_consumer() -> None:
    items = list(range(30))
    sizes = {i: 512 for i in items}
    produced = simple_producer(sizes)

    with mp.Manager() as m:
        seen = m.list()

        run_mpsc(
            items,
            producer_fn=produced,
            consumer_fn=slow_collecting_consumer(seen, delay_sec=0.01),  # type: ignore[arg-type]
            num_producers=4,
            data_queue_size=2,   # tiny queue to exercise backpressure
            transport="bytes",
        )

        assert set(list(seen)) == {f"id-{i:06d}" for i in items}


def test_producer_exception_bubbles() -> None:
    items = list(range(8))
    # Fail on a specific item
    produced = failing_producer(fail_on={5})

    with pytest.raises(RuntimeError, match="producer failed on 5"):
        run_mpsc(
            items,
            producer_fn=produced,
            consumer_fn=collecting_consumer([]),
            num_producers=2,
            data_queue_size=4,
            transport="bytes",
        )


def test_consumer_exception_bubbles() -> None:
    items = list(range(6))
    sizes = {i: 64 for i in items}
    produced = simple_producer(sizes)

    def bad_consumer(stream: Iterable[SampleMsg]) -> None:
        # Consume the first couple, then blow up
        it: Iterator[SampleMsg] = iter(stream)
        _ = next(it)
        _ = next(it)
        raise RuntimeError("consumer oops")

    with pytest.raises(RuntimeError, match="consumer oops"):
        run_mpsc(
            items,
            producer_fn=produced,
            consumer_fn=bad_consumer,
            num_producers=2,
            data_queue_size=4,
            transport="bytes",
        )


def test_zero_items_is_noop() -> None:
    run_mpsc(
        [],
        producer_fn=lambda _: None,            # never called
        consumer_fn=lambda stream: None,       # nothing to consume
        num_producers=3,
        data_queue_size=4,
        transport="bytes",
    )


def test_large_payloads_shm_end_to_end() -> None:
    items = [1]
    sizes = {1: 5 * 1024 * 1024}
    produced = simple_producer(sizes)

    with mp.Manager() as m:
        shm_names = m.list()  # shared list across processes

        def consumer(stream: Iterable[SampleMsg]) -> None:
            for msg in stream:
                assert msg.transport == "shm"
                # remember SHM name before cleanup kicks in
                assert isinstance(msg.data, str)
                shm_names.append(msg.data)
                with msg.open_shared_memory() as mv:
                    assert len(mv) == sizes[1]
                    ref = make_payload(sizes[1])
                    assert mv[0] == ref[0]
                    assert mv[-1] == ref[-1]

        run_mpsc(
            items,
            producer_fn=produced,
            consumer_fn=consumer,
            num_producers=1,
            data_queue_size=1,
            transport="shm",
            shm_threshold=1024,
        )

        # After run_mpsc returns, SHM should be unlinked
        name = list(shm_names)[0]
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)

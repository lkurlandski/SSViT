"""
Multi-producer / single-consumer streaming pipeline.

Design:
    - N producer processes pull work items, transform them into (name, bytes, meta)
      via a user-provided producer function, and push SampleMsg into a bounded queue.
    - One consumer process reads the queue and hands an iterable of SampleMsg to a
      user-provided consumer function.
    - Backpressure is provided by a bounded data queue; producers block when the
      consumer cannot keep up.
    - Large payloads can be passed via shared memory ('shm') or temp files ('path')
      to avoid pickling large byte blobs across processes.

Public API:
    - run_mpsc(...)     : orchestrates producers and consumer (entry point)
    - SampleMsg         : message type flowing from producers to consumer
    - ProducerFn, ConsumerFn : callable type aliases for static typing

Typical usage:
    def producer(item: str) -> Optional[tuple[str, bytes, dict[str, Any]]]:
        # fetch/parse bytes from item...
        return sha, payload, {"malware": True, "timestamp": 0}

    def consumer(stream: Iterable[SampleMsg]) -> None:
        for msg in stream:
            if msg.transport == "bytes":
                data = msg.data  # bytes
            elif msg.transport == "shm":
                with msg.open_shared_memory() as mv:
                    data = bytes(mv)  # or stream from memoryview
            elif msg.transport == "path":
                with open(msg.data, "rb") as fp:
                    data = fp.read()
            # ... process/write ...

    run_mpsc(work_items, producer_fn=producer, consumer_fn=consumer, num_producers=8, transport="shm")
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import get_context
from multiprocessing import resource_tracker
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import queue
import tempfile
import time
import traceback
from typing import Any
from typing import Final
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Union


# ------------------------------ Public API ------------------------------ #


Transport = Literal["bytes", "shm", "path"]


class ProducerFn(Protocol):

    def __call__(self, item: Any) -> Optional[tuple[str, bytes, dict[str, Any]]]:
        """
        Transform a work item into a (name, payload_bytes, meta) triple.

        Returns:
            (name, payload_bytes, meta) on success, or None to skip this item.
        """
        ...


class ConsumerFn(Protocol):

    def __call__(self, stream: Iterable["SampleMsg"]) -> None:
        """
        Consume a stream of SampleMsg. The iterator must be fully consumed
        (or the function should return promptly) to allow the pipeline to shut down.
        """
        ...


@dataclass(frozen=True)
class SampleMsg:
    """
    Message flowing from producers to the consumer.

    Attributes:
        name: Stable sample identifier (e.g., sha256).
        meta: Small JSON-serializable dictionary with auxiliary fields.
        transport: Transport type used to deliver the payload: 'bytes', 'shm', or 'path'.
        data: Payload carrier, depending on `transport`:
              - 'bytes': the actual bytes
              - 'shm'  : the shared-memory block name
              - 'path' : filesystem path to a temporary file containing the payload
        length: Payload length in bytes.
    """

    name: str
    meta: dict[str, Any]
    transport: Transport
    data: Union[bytes, str]
    length: int

    def open_shared_memory(self) -> SharedMemoryView:
        """
        Open the shared-memory block referenced by this message (transport='shm').

        Returns:
            A SharedMemoryView context manager exposing a memoryview over the bytes.

        Raises:
            RuntimeError: if transport is not 'shm'.
        """
        if self.transport != "shm" or not isinstance(self.data, str):
            raise RuntimeError("SampleMsg.open_shared_memory() requires transport='shm'.")
        return SharedMemoryView(self.data, self.length)


class SharedMemoryView:
    """
    Context manager that opens a SharedMemory block by name and exposes its buffer.

    Example:
        with msg.open_shared_memory() as mv:
            # mv is a memoryview of length `self._length`
            use(mv)
    """

    def __init__(self, name: str, length: int) -> None:
        self._name = name
        self._length = length
        self._shm: SharedMemory | None = None
        self._mv: memoryview | None = None

    def __enter__(self) -> memoryview:
        shm = SharedMemory(name=self._name)
        # This *reader* instance should not be tracked by the resource tracker,
        # otherwise we can end up double-unregistering the same name.
        try:
            # mypy: _name is private on CPython; ignore attr check
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass

        self._shm = shm
        # Keep a handle to the exported buffer so we can release it explicitly.
        self._mv = shm.buf[: self._length]
        return self._mv

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Release the exported buffer first, then close the SHM mapping.
        try:
            if self._mv is not None:
                try:
                    self._mv.release()
                finally:
                    self._mv = None
        finally:
            if self._shm is not None:
                try:
                    self._shm.close()
                finally:
                    self._shm = None


def run_mpsc(
    work_items: Iterable[Any],
    *,
    producer_fn: ProducerFn,
    consumer_fn: ConsumerFn,
    num_producers: int = 8,
    data_queue_size: int = 256,
    work_queue_size: int | None = None,
    errr_queue_size: int | None = None,
    transport: Transport = "shm",
    shm_threshold: int = 1 << 20,
    tempdir: Optional[Path] = None,
    start_method: Literal["fork", "spawn", "forkserver"] | None = None,
    graceful_timeout: float = 10.0,
) -> None:
    """
    Run a multi-producer / single-consumer pipeline with backpressure.

    Args:
        work_items:
            Iterable of work units. Items are enqueued and consumed by producer processes.
            The iterable is consumed in the parent process; it may be a generator.
        producer_fn:
            Function that converts a work item into (name, payload_bytes, meta),
            or returns None to skip.
        consumer_fn:
            Function that consumes an iterable of SampleMsg. It is invoked in the
            consumer process and should iterate the stream until exhaustion.
        num_producers:
            Number of producer processes.
        data_queue_size:
            Max number of in-flight SampleMsg entries (provides backpressure).
        work_queue_size:
            Optional bound on queued work items. Defaults to max(num_producers * 4, 32).
        errr_queue_size:
            Optional bound on error queue. Defaults to max(num_producers * 4, 32).
        transport:
            Payload transport to the consumer:
                - 'bytes': send raw bytes through the queue (simplest).
                - 'shm'  : large bytes moved via SharedMemory blocks (fast, zero-copy across processes).
                - 'path' : write to a temporary file and pass its path.
        shm_threshold:
            For 'shm' transport, payloads with length <= shm_threshold are sent as 'bytes' for
            efficiency; larger payloads go via shared memory.
        tempdir:
            For 'path' transport, directory for temporary files. Defaults to system temp dir.
        start_method:
            Multiprocessing start method. If None, Python picks a platform default
            ('fork' on Linux, 'spawn' on macOS/Windows).
        graceful_timeout:
            Seconds to wait for processes to terminate gracefully during shutdown.

    Raises:
        RuntimeError: if a child process crashes or a transport error occurs.
    """
    if work_queue_size is None:
        work_queue_size = max(num_producers * 4, 32)
    if errr_queue_size is None:
        errr_queue_size = max(num_producers * 4, 32)

    ctx = get_context(start_method) if start_method is not None else get_context()

    stop: Event = ctx.Event()
    work_q = ctx.Queue(maxsize=work_queue_size)   # items for producers
    data_q = ctx.Queue(maxsize=data_queue_size)   # SampleMsg for consumer
    errr_q = ctx.Queue(maxsize=errr_queue_size)   # errors from producers/consumer

    # Sentinel objects (type-checked across processes)
    WORK_DONE = _WorkDone()
    DATA_DONE = _DataDone()

    # Start consumer process
    consumer_p = ctx.Process(  # type: ignore[attr-defined]
        target=_consumer_proc,
        args=(data_q, errr_q, consumer_fn, DATA_DONE, stop),
        name="consumer",
        daemon=True,
    )
    consumer_p.start()

    # Start producer processes
    producers: list[mp.Process] = []
    for i in range(num_producers):
        p = ctx.Process(  # type: ignore[attr-defined]
            target=_producer_proc,
            args=(work_q, data_q, errr_q, producer_fn, transport, shm_threshold, tempdir, WORK_DONE, DATA_DONE, stop),
            name=f"producer-{i}",
            daemon=True,
        )
        p.start()
        producers.append(p)

    # Feed work items into work_q from the parent process with early-error polling
    try:
        for item in work_items:
            err = _poll_child_error(errr_q)
            if err is not None:
                where, etype, msg, tb = err
                stop.set()
                # stop producers and consumer promptly
                for _ in range(num_producers):
                    _q_put_blocking(work_q, WORK_DONE, stop, errr_q=errr_q)
                _q_put_blocking(data_q, DATA_DONE, stop, errr_q=errr_q)
                _join_all(producers, graceful_timeout)
                _join_all([consumer_p], graceful_timeout)
                raise RuntimeError(f"{where} error: {etype}: {msg}\n{tb}")
            _q_put_blocking(work_q, item, stop, errr_q=errr_q)

        # Normal end-of-work
        for _ in range(num_producers):
            _q_put_blocking(work_q, WORK_DONE, stop, errr_q=errr_q)

    except KeyboardInterrupt:
        stop.set()
    except Exception:
        stop.set()
        raise

    # Wait for producers with early-error polling
    deadline = time.time() + graceful_timeout
    while any(p.is_alive() for p in producers) and time.time() < deadline:
        err = _poll_child_error(errr_q)
        if err is not None:
            where, etype, msg, tb = err
            stop.set()
            _q_put_blocking(data_q, DATA_DONE, stop, errr_q=errr_q)
            _join_all(producers, graceful_timeout)
            _join_all([consumer_p], graceful_timeout)
            raise RuntimeError(f"{where} error: {etype}: {msg}\n{tb}")
        for p in producers:
            p.join(timeout=0.05)

    # Tell consumer that data stream is complete
    _q_put_blocking(data_q, DATA_DONE, stop, errr_q=errr_q)

    # Wait for consumer with early-error polling
    deadline = time.time() + graceful_timeout
    while consumer_p.is_alive() and time.time() < deadline:
        err = _poll_child_error(errr_q)
        if err is not None:
            where, etype, msg, tb = err
            stop.set()
            _join_all([consumer_p], graceful_timeout)
            raise RuntimeError(f"{where} error: {etype}: {msg}\n{tb}")
        consumer_p.join(timeout=0.05)

    # Late error (if any) after joins
    err = _poll_child_error(errr_q)
    if err is not None:
        where, etype, msg, tb = err
        stop.set()
        raise RuntimeError(f"{where} error: {etype}: {msg}\n{tb}")

    # If any process crashed, raise
    for p in producers + [consumer_p]:
        if p.exitcode not in (0, None):
            raise RuntimeError(f"Subprocess {p.name} exited with code {p.exitcode}.")


# ------------------------------ Internal helpers ------------------------------ #


class _WorkDone:
    """Sentinel indicating no more work items will be enqueued."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_WORK_DONE"


class _DataDone:
    """Sentinel indicating no more data messages will be enqueued."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_DATA_DONE"


def _producer_proc(
    work_q: mp.Queue[Any],
    data_q: mp.Queue[Any],
    errr_q: mp.Queue[Any],
    producer_fn: ProducerFn,
    transport: Transport,
    shm_threshold: int,
    tempdir: Optional[Path],
    WORK_DONE: object,
    DATA_DONE: object,
    stop: Event,
) -> None:
    def send_bytes(name: str, payload: bytes, meta: dict[str, Any]) -> None:
        msg = SampleMsg(name=name, meta=meta, transport="bytes", data=payload, length=len(payload))
        _q_put_blocking(data_q, msg, stop)

    def send_shm(name: str, payload: bytes, meta: dict[str, Any]) -> None:
        shm = SharedMemory(create=True, size=len(payload))
        try:
            # Producer process: don't let the resource tracker try to clean this up.
            try:
                # mypy: SharedMemory has private attr _name in CPython; ignore attr check
                resource_tracker.unregister(shm._name, 'shared_memory')  # type: ignore[attr-defined]
            except Exception:
                pass

            shm.buf[: len(payload)] = payload
            msg = SampleMsg(name=name, meta=meta, transport="shm", data=shm.name, length=len(payload))
            _q_put_blocking(data_q, msg, stop)
        finally:
            shm.close()  # consumer will open by name and unlink later

    def send_path(name: str, payload: bytes, meta: dict[str, Any]) -> None:
        fd, path = tempfile.mkstemp(prefix="mpsc_", dir=str(tempdir) if tempdir else None)
        try:
            with os.fdopen(fd, "wb", closefd=True) as fp:
                fp.write(payload)
            msg = SampleMsg(name=name, meta=meta, transport="path", data=path, length=len(payload))
            _q_put_blocking(data_q, msg, stop)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise

    try:
        while not stop.is_set():
            try:
                item = work_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, _WorkDone):
                break

            out = producer_fn(item)
            if out is None:
                continue

            name, payload, meta = out
            if transport == "bytes":
                send_bytes(name, payload, meta)
            elif transport == "shm":
                if len(payload) <= shm_threshold:
                    send_bytes(name, payload, meta)
                else:
                    send_shm(name, payload, meta)
            elif transport == "path":
                send_path(name, payload, meta)
            else:
                raise RuntimeError(f"Unknown transport: {transport!r}")

    except KeyboardInterrupt:
        stop.set()
    except Exception as e:
        stop.set()
        errr_q.put(("producer", e.__class__.__name__, str(e), traceback.format_exc()))
        raise

    finally:
        # Best-effort: if this is the last producer to finish and no data remains,
        # the parent will still send DATA_DONE after joining producers.
        pass


def _consumer_proc(
    data_q: mp.Queue[Any],
    errr_q: mp.Queue[Any],
    consumer_fn: ConsumerFn,
    DATA_DONE: object,
    stop: Event,
) -> None:
    """
    Read SampleMsg from data_q and yield them to consumer_fn(stream).
    Ensures transport-specific cleanup (unlink shared memory, remove temp files)
    after the consumer has observed each message.
    """

    def stream() -> Iterator[SampleMsg]:
        while not stop.is_set():
            try:
                obj = data_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(obj, _DataDone):
                break
            msg: SampleMsg = obj
            try:
                if not isinstance(msg, SampleMsg):
                    raise RuntimeError(f"Unexpected object in data_q: {type(msg).__name__}")
                yield msg
            finally:
                # Cleanup resources after consumer sees the message
                if msg.transport == "shm" and isinstance(msg.data, str):
                    _unlink_shared_memory(msg.data)
                elif msg.transport == "path" and isinstance(msg.data, str):
                    try:
                        os.unlink(msg.data)
                    except FileNotFoundError:
                        pass

    try:
        consumer_fn(stream())
    except KeyboardInterrupt:
        stop.set()
    except Exception as e:
        stop.set()
        errr_q.put(("consumer", e.__class__.__name__, str(e), traceback.format_exc()))
        raise


def _unlink_shared_memory(name: str) -> None:
    try:
        shm = SharedMemory(name=name)
    except FileNotFoundError:
        return
    try:
        # Do NOT unregister here; let this instance's finalizer handle it once.
        shm.unlink()
    finally:
        try:
            shm.close()
        except FileNotFoundError:
            pass


def _q_put_blocking(q: mp.Queue[Any], item: Any, stop: Event, errr_q: Optional[mp.Queue[Any]] = None) -> None:
    """
    Put an item into a multiprocessing.Queue with cooperative shutdown and
    optional early abort if a child error is reported.
    """
    while not stop.is_set():
        # Early abort if an error is already queued
        if errr_q is not None:
            err = _poll_child_error(errr_q)
            if err is not None:
                where, etype, msg, tb = err
                raise RuntimeError(f"{where} error: {etype}: {msg}\n{tb}")
        try:
            q.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _join_all(procs: Sequence[mp.Process], timeout: float) -> None:
    """
    Join all processes with a grace period, then terminate any stragglers.
    """
    deadline = time.time() + timeout
    alive: list[mp.Process] = list(procs)

    while alive and time.time() < deadline:
        next_alive: list[mp.Process] = []
        for p in alive:
            p.join(timeout=0.1)
            if p.is_alive():
                next_alive.append(p)
        alive = next_alive

    for p in alive:
        try:
            p.terminate()
        except Exception:
            pass
        p.join(timeout=1.0)


def _poll_child_error(errr_q: mp.Queue[Any]) -> Optional[tuple[str, str, str, str]]:
    """
    Try to fetch one error tuple from a child process without blocking.

    Returns:
        (where, etype, msg, tb) if available, else None.
    """
    try:
        return errr_q.get_nowait()  # type: ignore[no-any-return]
    except queue.Empty:
        return None

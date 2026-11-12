"""Real blocked-writer regression and bounded serve-only logging lifecycle."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from local_operator.logger import configure_console_logging
from local_operator.serve_logging import ServeLogHandler, configure_serve_logging


class BlockedSink(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.delivered = threading.Event()
        self.exception_delivered = threading.Event()

    def write(self, text: str) -> int:
        self.entered.set()
        self.release.wait()
        result = super().write(text)
        self.delivered.set()
        if "ValueError: test exception" in text:
            self.exception_delivered.set()
        return result


@pytest.fixture
def logging_state():
    names = ("", "uvicorn", "uvicorn.error", "uvicorn.access")
    saved = [
        (
            logging.getLogger(n),
            list(logging.getLogger(n).handlers),
            logging.getLogger(n).level,
            logging.getLogger(n).propagate,
        )
        for n in names
    ]
    yield
    for logger, handlers, level, propagate in saved:
        for handler in logger.handlers:
            if isinstance(handler, ServeLogHandler):
                handler.close()
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def record(message: str, level: int = logging.WARNING) -> logging.LogRecord:
    return logging.LogRecord("fixture", level, __file__, 1, message, (), None)


def instrumented_handler(stream: io.StringIO, **kwargs) -> ServeLogHandler:
    """Inject only the consumer write for deterministic queue/formatting tests.

    Production intentionally drops streams without a private fd. These synthetic
    sinks measure queue behavior, not destination selection; subprocess tests
    below separately exercise the real private-fd and unavailable-fd paths.
    """
    handler = ServeLogHandler(stream, **kwargs)
    instrument_sink(handler, stream)
    return handler


def instrument_sink(handler: ServeLogHandler, stream: io.StringIO) -> None:
    def write(text: str) -> None:
        stream.write(text)
        stream.flush()

    handler._write = write


def test_blocked_background_write_does_not_block_event_loop_or_close() -> None:
    sink = BlockedSink()
    handler = instrumented_handler(sink)
    handler.handle(record("background cache warning"))
    assert sink.entered.wait(2)
    finished = threading.Event()

    async def loop_callback() -> None:
        handler.handle(record("asyncio slow callback warning"))
        await asyncio.sleep(0)
        handler.close()
        finished.set()

    thread = threading.Thread(target=lambda: asyncio.run(loop_callback()), daemon=True)
    thread.start()
    try:
        assert finished.wait(2), "event loop/close waited on the blocked sink"
        assert handler.lock is not None
    finally:
        sink.release.set()
        thread.join(2)
        handler.close()
        handler._worker.join(2)
    assert not thread.is_alive()
    assert not handler._worker.is_alive()


def test_overflow_reserves_diagnostics_and_counts_every_drop() -> None:
    sink = BlockedSink()
    handler = instrumented_handler(sink, capacity=8)
    handler.handle(record("inflight"))
    assert sink.entered.wait(2)
    try:
        for _ in range(100):
            handler.handle(record("info", logging.INFO))
        assert handler.pending == 4
        assert handler.dropped == 96
        for _ in range(10):
            handler.handle(record("error", logging.ERROR))
        assert handler.pending == 8
        assert handler.dropped == 102
        assert not handler._normal
        assert all(r.levelno == logging.ERROR for r in handler._important)
        handler.close()
        assert handler.pending == 0
        assert handler.dropped == 110
        handler.handle(record("after stop"))
        assert handler.dropped == 111
    finally:
        sink.release.set()
        handler.close()
        handler._worker.join(2)


def test_exception_formatting_and_drop_summary_are_on_consumer() -> None:
    sink = BlockedSink()
    handler = instrumented_handler(sink, capacity=2)
    handler.handle(record("first"))
    assert sink.entered.wait(2)
    formatted_on = []
    formatted = threading.Event()

    class Lazy:
        def __str__(self) -> str:
            formatted_on.append(threading.current_thread().name)
            formatted.set()
            return "lazy value"

    try:
        handler.handle(record("displaced", logging.INFO))
        handler.handle(record("replacement", logging.INFO))
        try:
            raise ValueError("test exception")
        except ValueError:
            exc = logging.LogRecord(
                "fixture", logging.ERROR, __file__, 1, "failed %s", (Lazy(),), sys.exc_info()
            )
        handler.handle(exc)
        assert not formatted_on
        sink.release.set()
        assert formatted.wait(2)
        assert sink.exception_delivered.wait(2)
        assert "ValueError: test exception" in sink.getvalue()
        assert "serve logging: dropped 1 records" in sink.getvalue()
        assert formatted_on == ["serve-log-sink"]
    finally:
        sink.release.set()
        handler.close()
        handler._worker.join(2)


def test_configure_cycles_and_cli_unchanged(logging_state, monkeypatch) -> None:
    sink = BlockedSink()
    monkeypatch.setattr(sys, "stderr", sink)
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    first = configure_serve_logging()
    assert configure_serve_logging() is first
    assert logging.getLogger().level == logging.INFO
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        assert logging.getLogger(name).handlers == [first]
        assert not logging.getLogger(name).propagate
    first.close()
    second = configure_serve_logging()
    assert second is not first
    second.close()
    configure_console_logging(level=logging.ERROR)
    console = logging.getLogger().handlers[0]
    assert isinstance(console, logging.StreamHandler)
    assert not isinstance(console, ServeLogHandler)
    assert console.stream is sink
    assert logging.getLogger().level == logging.ERROR
    sink.release.set()
    first._worker.join(2)
    second._worker.join(2)


def test_normal_logs_delivered(logging_state, monkeypatch) -> None:
    sink = BlockedSink()
    sink.release.set()
    monkeypatch.setattr(sys, "stderr", sink)
    handler = configure_serve_logging()
    instrument_sink(handler, sink)
    try:
        logging.getLogger("fixture").warning("normal %s", "delivery")
        assert sink.delivered.wait(2)
        assert "[WARNING] fixture: normal delivery" in sink.getvalue()
    finally:
        handler.close()
        handler._worker.join(2)


@pytest.mark.parametrize("exhaust_fds", [False, True])
def test_shutdown_does_not_flush_blocked_stderr_wrapper(tmp_path: Path, exhaust_fds: bool) -> None:
    # Child fills its undrained stderr pipe. The private fd consumer must not
    # hold TextIOWrapper's lock, or Python's final stderr flush hangs at exit.
    code = """
import io, logging, os, resource, sys, threading
assert isinstance(sys.stderr.buffer, io.BufferedWriter)
from local_operator.serve_logging import configure_serve_logging
fillers = []
if sys.argv[1] == "True":
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        while True:
            fillers.append(os.open(os.devnull, os.O_RDONLY))
    except OSError as exc:
        import errno
        assert exc.errno == errno.EMFILE
h = configure_serve_logging()
assert (h._fd is None) == (sys.argv[1] == "True")
for fd in fillers:
    os.close(fd)
os.set_blocking(2, False)
try:
    while True:
        os.write(2, b"x" * 1024)
except BlockingIOError:
    pass
os.set_blocking(2, True)
entered = threading.Event()
original = os.write
def write(fd, data):
    entered.set()
    return original(fd, data)
os.write = write
attempted = threading.Event()
original_write = h._write
def sink_write(text):
    try:
        original_write(text)
    finally:
        attempted.set()
h._write = sink_write
logging.warning("blocked write")
if h._fd is None:
    assert attempted.wait(2)
else:
    assert entered.wait(2)
h.close()
if h._fd is None:
    h._worker.join(2)
    assert not h._worker.is_alive()
    assert h.sink_errors == h.dropped == 1
logging.shutdown()
print("shutdown completed", flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(exhaust_fds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # The harness exports PYTHONUNBUFFERED=1, which bypasses the buffered
        # wrapper lock and would hide the interpreter-finalization regression.
        env={k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"},
        start_new_session=True,
    )
    try:
        assert process.wait(timeout=10) == 0
        assert process.stdout is not None
        assert b"shutdown completed" in process.stdout.read()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def test_handler_preserves_filter_record_replacement() -> None:
    sink = BlockedSink()
    sink.release.set()
    handler = instrumented_handler(sink)
    handler.addFilter(
        lambda incoming: record("replacement") if incoming.msg == "original" else False
    )
    try:
        handler.handle(record("rejected"))
        handler.handle(record("original"))
        assert sink.delivered.wait(2)
        assert "replacement" in sink.getvalue()
        assert "original" not in sink.getvalue()
        assert "rejected" not in sink.getvalue()
    finally:
        handler.close()
        handler._worker.join(2)


def test_private_fd_delivers_without_using_stream_wrapper(tmp_path: Path, monkeypatch) -> None:
    delivered = threading.Event()
    original = os.write

    def write(fd, data):
        result = original(fd, data)
        delivered.set()
        return result

    monkeypatch.setattr(os, "write", write)
    path = tmp_path / "private-fd.log"
    with path.open("w") as stream:
        handler = ServeLogHandler(stream)
        assert handler._fd is not None
        try:
            handler.handle(record("private fd delivery"))
            assert delivered.wait(2)
        finally:
            handler.close()
            handler._worker.join(2)
        # The caller's stream is still open and usable after private-fd cleanup.
        assert not stream.closed
    assert "private fd delivery" in path.read_text()


def test_unavailable_fd_drops_without_touching_wrapper() -> None:
    class ForbiddenSink(io.StringIO):
        def write(self, text: str) -> int:
            pytest.fail("unavailable private fd must not fall back to wrapper I/O")

    sink = ForbiddenSink()
    handler = ServeLogHandler(sink)
    attempted = threading.Event()
    original = handler._write

    def write(text: str) -> None:
        try:
            original(text)
        finally:
            attempted.set()

    handler._write = write
    try:
        handler.handle(record("unavailable fd"))
        assert attempted.wait(2)
    finally:
        handler.close()
        handler._worker.join(2)
    assert not handler._worker.is_alive()
    assert handler.sink_errors == handler.dropped == 1
    assert not sink.getvalue()


def test_queue_retains_only_capacity_payload_graphs() -> None:
    import weakref

    class Payload:
        pass

    sink = BlockedSink()
    handler = instrumented_handler(sink, capacity=8)
    handler.handle(record("inflight"))
    assert sink.entered.wait(2)
    payloads = weakref.WeakSet()
    try:
        for _ in range(3 * handler.capacity):
            payload = Payload()
            payloads.add(payload)
            handler.handle(
                logging.LogRecord("fixture", logging.WARNING, __file__, 1, "%s", (payload,), None)
            )
        del payload
        assert handler.pending == handler.capacity
        assert len(payloads) == handler.capacity
        handler.close()
        assert not payloads, "close must release queued argument graphs, not just stop draining"
    finally:
        sink.release.set()
        handler.close()
        handler._worker.join(2)


def test_cli_installs_before_serve_and_closes_on_failure(logging_state, monkeypatch) -> None:
    from local_operator import cli

    captured = []

    def fail(*args, **kwargs):
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler, ServeLogHandler)
        assert not handler.stopped
        captured.append(handler)
        raise RuntimeError("synthetic startup failure")

    monkeypatch.setattr(cli, "_serve_command", fail)
    with pytest.raises(RuntimeError, match="synthetic startup failure"):
        cli.serve_command("127.0.0.1", 0, False)
    assert captured[0].stopped
    captured[0]._worker.join(2)


@pytest.mark.asyncio
async def test_embedded_lifespan_closes_on_startup_failure(logging_state, monkeypatch) -> None:
    import importlib
    from contextlib import asynccontextmanager

    module = importlib.import_module("local_operator.server.app")
    captured = []

    @asynccontextmanager
    async def fail(app):
        captured.append(logging.getLogger().handlers[0])
        raise RuntimeError("synthetic startup failure")
        yield

    monkeypatch.setattr(module, "_lifespan", fail)
    with pytest.raises(RuntimeError, match="synthetic startup failure"):
        async with module.lifespan(module.app):
            pass
    assert captured[0].stopped
    captured[0]._worker.join(2)


def test_uvicorn_config_does_not_close_or_replace_serve_sink(logging_state) -> None:
    import uvicorn

    handler = configure_serve_logging()
    try:
        uvicorn.Config("local_operator.server.app:app", log_config=None)
        assert not handler.stopped
        assert logging.getLogger().handlers == [handler]
        assert logging.getLogger("uvicorn.error").handlers == [handler]
    finally:
        handler.close()
        handler._worker.join(2)


def test_sink_failure_is_counted_and_consumer_recovers() -> None:
    class RecoveringSink(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True
            self.recovered = threading.Event()

        def write(self, text: str) -> int:
            if self.fail:
                self.fail = False
                raise OSError("synthetic sink failure")
            result = super().write(text)
            if "recovery" in text:
                self.recovered.set()
            return result

    sink = RecoveringSink()
    handler = instrumented_handler(sink)
    try:
        handler.handle(record("lost"))
        handler.handle(record("recovery"))
        assert sink.recovered.wait(2)
        assert handler.sink_errors == 1
        assert handler.dropped == 1
        assert "dropped 1 records" in sink.getvalue()
    finally:
        handler.close()
        handler._worker.join(2)

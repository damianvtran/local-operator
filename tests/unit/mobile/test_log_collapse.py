"""The out-of-files accept loop writes one traceback per interval, and loses no count.

At its descriptor limit the phone daemon's asyncio accept loop logged a full
traceback on every wakeup (~7 MB/s on a rig). The invariant under test: the
same message is written at most once per interval, and the number of
suppressed copies is carried onto the next copy written.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from local_operator.mobile import log_collapse


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def asyncio_logger():
    logger = logging.getLogger("asyncio")
    capture = _Capture()
    saved = (list(logger.filters), logger.level, logger.propagate)
    logger.addHandler(capture)
    logger.setLevel(logging.WARNING)  # not the loop's own "Using selector" debug line
    logger.propagate = False
    try:
        yield logger, capture
    finally:
        logger.removeHandler(capture)
        logger.filters[:] = saved[0]
        logger.setLevel(saved[1])
        logger.propagate = saved[2]


def _accept_failure(loop: asyncio.AbstractEventLoop) -> None:
    # The exact context asyncio's selector loop hands its exception handler when
    # accept() returns EMFILE (base_events default handler -> "asyncio" logger).
    loop.call_exception_handler(
        {
            "message": "socket.accept() out of system resource",
            "exception": OSError(24, "Too many open files"),
        }
    )


def test_a_storm_of_accept_failures_is_written_once_per_interval(asyncio_logger):
    logger, capture = asyncio_logger
    clock = _Clock()
    logger.addFilter(log_collapse.RepeatCollapse(interval=60, clock=clock))
    loop = asyncio.new_event_loop()
    try:
        for _ in range(5000):  # one burst: what the loop produced in ~1 s
            _accept_failure(loop)
        assert len(capture.records) == 1
        assert capture.records[0].exc_info is not None  # the first keeps its traceback
        clock.now += 59.9
        _accept_failure(loop)
        assert len(capture.records) == 1
        clock.now += 0.2
        _accept_failure(loop)
    finally:
        loop.close()
    assert len(capture.records) == 2
    # Every occurrence is accounted for: 1 + 5000 suppressed + 1 written.
    assert "[5000 similar suppressed in the last 60s]" in capture.records[1].getMessage()
    assert capture.records[1].getMessage().startswith("socket.accept() out of system resource")


def test_different_messages_are_never_collapsed_together(asyncio_logger):
    logger, capture = asyncio_logger
    logger.addFilter(log_collapse.RepeatCollapse(interval=60, clock=_Clock()))
    logger.error("socket.accept() out of system resource")
    logger.error("Task exception was never retrieved")
    logger.warning("socket.accept() out of system resource")  # another level
    assert len(capture.records) == 3


def test_a_formatted_record_keeps_its_arguments(asyncio_logger):
    logger, capture = asyncio_logger
    clock = _Clock()
    logger.addFilter(log_collapse.RepeatCollapse(interval=10, clock=clock))
    logger.error("poll failed on %s", "fd 6")
    logger.error("poll failed on %s", "fd 6")
    clock.now += 11
    logger.error("poll failed on %s", "fd 6")
    assert capture.records[-1].getMessage() == (
        "poll failed on fd 6 [1 similar suppressed in the last 11s]"
    )


def test_install_is_idempotent(asyncio_logger):
    logger, _capture = asyncio_logger
    first = log_collapse.install("asyncio")
    assert log_collapse.install("asyncio") is first
    assert sum(isinstance(f, log_collapse.RepeatCollapse) for f in logger.filters) == 1

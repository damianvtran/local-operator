"""The slow-apply diagnostic on ``FrontendStateStore`` fan-out.

Why this exists: the 2026-11-15 delta fan-out study (lop-performance
docs/2026-11-15-delta-fanout/report.md) could not ship a freeze-once fan-out
because the live subscriber count per store was never logged. The store now
emits one rate-limited warning when an apply crosses 50 ms. These tests pin the
three properties that make that line safe to leave on in production: it carries
names and counts only (never conversation content), it is rate-limited, and it
stays silent under the threshold.

The store's clock is replaced with a fake one advanced by a subscriber, so the
result never depends on how loaded the test host is.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest

from local_operator.session import frontend_state
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
)

_LOGGER = frontend_state.logger.name
_SECRET = "TOP-SECRET-CONVERSATION-CONTENT"


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    # Only the module's own ``time`` binding is swapped: the rest of the
    # process keeps the real clock, and the CPU clock stays real (it is only
    # reported, never compared against the threshold).
    monkeypatch.setattr(
        frontend_state,
        "time",
        SimpleNamespace(
            perf_counter=fake.perf_counter,
            clock_gettime=time.clock_gettime,
            CLOCK_PROCESS_CPUTIME_ID=time.CLOCK_PROCESS_CPUTIME_ID,
        ),
    )
    return fake


def _store() -> FrontendStateStore:
    return FrontendStateStore(FrontendSessionState(session_id="sess-slow", epoch="e1"))


def _slow_subscriber(clock: _Clock, seconds: float):  # noqa: ANN202
    def deliver(update: FrontendUpdate) -> None:
        clock.now += seconds

    return deliver


def _slow_lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == _LOGGER and "slow apply" in record.getMessage()
    ]


def test_mutate_over_threshold_logs_names_and_counts_only(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.080))
    store.subscribe(lambda update: None)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        store.mutate(goal=_SECRET, conversation_title=_SECRET)

    [record] = _slow_lines(caplog)
    message = record.getMessage()
    assert record.levelno == logging.WARNING
    assert "site=mutate" in message
    assert "session=sess-slow" in message
    assert "subscribers=2" in message
    assert "fields=conversation_title,goal " in message
    assert "job_deltas=0 appends=0" in message
    assert "fanout_ms=80.0" in message
    assert "reducer_ms=0.0" in message
    assert "cpu_ms=" in message
    assert "suppressed=0" in message
    # Values are conversation content; the line must never carry them.
    assert _SECRET not in message
    assert all(_SECRET not in str(arg) for arg in record.args or ())


def test_follower_apply_and_degraded_sites_log(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.060))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        store.apply_update(FrontendUpdate(epoch="e1", sequence=1, changes={"goal": _SECRET}))
        clock.now += 10  # past the rate-limit window
        store.apply_update(FrontendUpdate(epoch="e1", sequence=2, changes={}, degraded=True))

    first, second = (record.getMessage() for record in _slow_lines(caplog))
    assert "site=apply " in first
    assert "fields=goal " in first
    assert _SECRET not in first
    assert "site=apply-degraded" in second
    assert "fields=- " in second
    assert "degraded=True" in second


def test_replace_and_notify_site_logs(clock: _Clock, caplog: pytest.LogCaptureFixture) -> None:
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.051))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        store.replace_and_notify(
            FrontendSessionState(session_id="sess-slow", epoch="e1", sequence=5, goal=_SECRET)
        )

    [record] = _slow_lines(caplog)
    assert "site=replace" in record.getMessage()
    assert _SECRET not in record.getMessage()


def test_slow_lines_are_rate_limited_with_a_suppressed_count(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.100))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for index in range(4):
            store.mutate(goal=f"goal {index}")
        assert len(_slow_lines(caplog)) == 1
        clock.now += 5.0
        store.mutate(goal="after the window")

    lines = [record.getMessage() for record in _slow_lines(caplog)]
    assert len(lines) == 2
    assert "suppressed=0" in lines[0]
    assert "suppressed=3" in lines[1]


def test_rate_limit_is_per_store(clock: _Clock, caplog: pytest.LogCaptureFixture) -> None:
    first, second = _store(), _store()
    for store in (first, second):
        store.subscribe(_slow_subscriber(clock, 0.100))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        first.mutate(goal="a")
        second.mutate(goal="b")

    assert len(_slow_lines(caplog)) == 2


def test_under_threshold_stays_silent(clock: _Clock, caplog: pytest.LogCaptureFixture) -> None:
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.049))

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        store.mutate(goal="quick")
        store.apply_update(FrontendUpdate(epoch="e1", sequence=2, changes={"goal": "x"}))

    assert _slow_lines(caplog) == []


def test_windows_shaped_time_module_falls_back_to_process_time(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Review R1-B1: ``clock_gettime`` and its CPU clock id are Unix-only.

    Windows CPython has neither, so the module's ``time`` binding is swapped
    for one without them (as the fixture does for the fake wall clock) and the
    CPU reader is re-resolved against it, the way import resolves it there.
    Every apply must still succeed and the slow line must still carry cpu_ms.
    """
    windows_time = SimpleNamespace(perf_counter=clock.perf_counter, process_time=lambda: 7.0)
    monkeypatch.setattr(frontend_state, "time", windows_time)
    monkeypatch.setattr(
        frontend_state,
        "_process_cpu_s",
        frontend_state._resolve_process_cpu_clock(windows_time),
    )
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.060))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        store.mutate(goal="windows")
        clock.now += 10
        store.apply_update(FrontendUpdate(epoch="e1", sequence=2, changes={"goal": "y"}))

    assert store.state.goal == "y"
    lines = [record.getMessage() for record in _slow_lines(caplog)]
    assert len(lines) == 2
    assert all("cpu_ms=0.0" in line for line in lines)


def test_unix_time_module_resolves_the_cpu_clock() -> None:
    reader = frontend_state._resolve_process_cpu_clock(
        SimpleNamespace(
            clock_gettime=lambda clock_id: float(clock_id),
            CLOCK_PROCESS_CPUTIME_ID=12,
            process_time=lambda: -1.0,
        )
    )
    assert reader() == 12.0


def _job_deltas_value(message: str) -> int:
    return int(message.split("job_deltas=", 1)[1].split()[0])


def test_job_deltas_counts_roster_rows_not_dict_keys(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """Review R1-M1: ``job_deltas`` is a dict of arms; the line counts rows."""
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.060))
    roster = [
        JobState(id=f"child-{index:02d}", type="task", label=f"child {index}")
        for index in range(50)
    ]

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        seed = store.mutate(jobs=roster)
        clock.now += 10
        # 30 rows change, 20 leave.
        moved = store.mutate(
            jobs=[job.model_copy(update={"label": f"moved {job.id}"}) for job in roster[:30]]
        )

    # ``mutate`` returns None for a no-op; both calls here change the roster.
    assert seed is not None
    assert moved is not None
    assert len(seed.job_deltas["upserted"]) == 50
    assert (len(moved.job_deltas["upserted"]), len(moved.job_deltas["removed"])) == (30, 20)
    first, second = (record.getMessage() for record in _slow_lines(caplog))
    assert _job_deltas_value(first) == 50
    assert _job_deltas_value(second) == 50
    assert "child-0" not in first + second  # ids are counted, never listed


def test_undeclared_change_keys_are_counted_not_named(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """Review R1-m1: ``apply_update`` takes any key off the wire."""
    store = _store()
    store.subscribe(_slow_subscriber(clock, 0.060))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        store.apply_update(
            FrontendUpdate(
                epoch="e1", sequence=1, changes={"goal": "x", "user secret text": _SECRET}
            )
        )

    [record] = _slow_lines(caplog)
    message = record.getMessage()
    assert "user secret text" not in message
    assert _SECRET not in message
    assert "fields=goal fields_other=1 " in message

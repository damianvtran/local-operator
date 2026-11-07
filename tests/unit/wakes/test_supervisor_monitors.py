"""The wake supervisor's MONITORS pass: an armed monitor re-engages its session.

Before this pass the supervisor read only the wakes index, so a session whose
runtime had idle-exited ticked none of its monitors until the operator opened
it again — measured at 260 minutes for the reply a watch was armed to catch
(docs/design/monitor-tool.md §10.4, reversed). These tests pin the decisions:
which monitor rows are due (COUNTERS first, not the index row), which are never
due (held, disabled, expired, ghost), that there is no staleness bound, that a
monitor alone keeps the supervisor up and sizes its sleep, and that a live
runtime is left alone with nothing written to the wake ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.wakes import supervisor as sup
from local_operator.wakes.supervisor import (
    _due_sessions,
    _has_fireable_wakes,
    _load_and_reconcile_state,
    _next_wake_ms,
    fire_due_wakes,
)

NOW_MS = 1_800_000_000_000
DAY_MS = 86_400_000


def _make_session(config_dir: Path, session_id: str) -> None:
    from local_operator.resume import TRANSCRIPT_NAME

    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TRANSCRIPT_NAME).write_text("", encoding="utf-8")


def _row(
    monitor_id: str = "m1",
    *,
    due: int | None,
    disabled: bool = False,
    until_at: int | None = None,
) -> dict[str, Any]:
    return {
        "id": monitor_id,
        "name": "watch",
        "tool": "bash",
        "arguments": {"command": "date -u"},
        "every_ms": 60_000,
        "until_at": until_at,
        "created_at": 1,
        "next_due_at": due,
        "disabled": disabled,
    }


def _arm(
    config_dir: Path,
    session_id: str,
    *,
    index_due: int | None,
    counters_due: int | None | str = "absent",
    disabled: bool = False,
    until_at: int | None = None,
    held: bool = False,
    ghost: bool = False,
) -> None:
    """Lay down one monitor entry (and optionally its counters file)."""
    if not ghost:
        _make_session(config_dir, session_id)
    monitor_store.write_entry(
        config_dir,
        session_id,
        cwd=str(config_dir),
        monitors=[_row(due=index_due, disabled=disabled, until_at=until_at)],
        preserve={"stopped_at": NOW_MS - 1} if held else None,
    )
    if counters_due != "absent":
        counters = {"schema": 1, "monitor_id": "m1", "next_due_at": counters_due}
        monitor_state.write_counters(config_dir, session_id, "m1", counters)


def _state(config_dir: Path) -> tuple[Any, ...]:
    return _load_and_reconcile_state(config_dir)


def _due(config_dir: Path, now_ms: int = NOW_MS) -> list[tuple[str, str, int]]:
    index, deliveries, spooled, triggers, monitors = _state(config_dir)
    return _due_sessions(
        index,
        now_ms,
        config_dir=config_dir,
        deliveries=deliveries,
        spooled=spooled,
        triggers=triggers,
        monitors=monitors,
    )


def _fireable(config_dir: Path, now_ms: int = NOW_MS) -> bool:
    index, deliveries, spooled, triggers, monitors = _state(config_dir)
    return _has_fireable_wakes(
        index,
        config_dir=config_dir,
        now_ms=now_ms,
        deliveries=deliveries,
        spooled=spooled,
        triggers=triggers,
        monitors=monitors,
    )


@pytest.fixture(autouse=True)
def _fresh_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backoff is module state; each test starts with none.

    ``raising=False`` so the fail-before run against a tree without the pass
    fails on the behaviour under test rather than erroring in setup.
    """
    backoff_cls = getattr(sup, "_MonitorBackoff", None)
    if backoff_cls is not None:
        monkeypatch.setattr(sup, "_monitor_backoff", backoff_cls(), raising=False)


# (a) ---------------------------------------------------------------------


def test_a_monitor_only_session_due_by_its_counters_is_due(tmp_path: Path) -> None:
    _arm(tmp_path, "monitoronly1", index_due=NOW_MS + DAY_MS, counters_due=NOW_MS - 1_000)
    assert _due(tmp_path) == [("monitoronly1", str(tmp_path), NOW_MS - 1_000)]


def test_a_row_with_no_counters_file_falls_back_to_the_index(tmp_path: Path) -> None:
    _arm(tmp_path, "freshlyarmed", index_due=NOW_MS - 500)
    assert _due(tmp_path) == [("freshlyarmed", str(tmp_path), NOW_MS - 500)]


# (b) ---------------------------------------------------------------------


def test_index_due_in_the_past_but_counters_in_the_future_is_not_due(tmp_path: Path) -> None:
    """The index is not rewritten on quiet ticks; trusting it would re-engage
    a healthy session on every slice."""
    _arm(tmp_path, "quietticker", index_due=NOW_MS - DAY_MS, counters_due=NOW_MS + 60_000)
    assert _due(tmp_path) == []


# (c) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        {"held": True},
        {"disabled": True},
        {"until_at": NOW_MS - 1},
        {"ghost": True},
    ],
    ids=["held", "disabled", "expired", "ghost"],
)
def test_unfireable_monitors_are_neither_due_nor_holding_the_supervisor(
    tmp_path: Path, shape: dict[str, Any]
) -> None:
    _arm(tmp_path, "cannotfire01", index_due=NOW_MS - 1_000, counters_due=NOW_MS - 1_000, **shape)
    assert _due(tmp_path) == []
    assert _fireable(tmp_path) is False


def test_a_counters_disable_wins_over_an_enabled_index_row(tmp_path: Path) -> None:
    _arm(tmp_path, "disabledlate", index_due=NOW_MS - 1_000)
    monitor_state.write_counters(
        tmp_path,
        "disabledlate",
        "m1",
        {"schema": 1, "monitor_id": "m1", "next_due_at": None, "disabled": True},
    )
    assert _due(tmp_path) == []
    assert _fireable(tmp_path) is False


# (d) ---------------------------------------------------------------------


def test_a_monitor_overdue_by_more_than_seven_days_is_still_due(tmp_path: Path) -> None:
    """No staleness bound: a durable watch dark for days owes its check."""
    overdue = NOW_MS - 9 * DAY_MS
    _arm(tmp_path, "longdormant1", index_due=overdue, counters_due=overdue)
    assert _due(tmp_path) == [("longdormant1", str(tmp_path), overdue)]
    assert _fireable(tmp_path) is True


# (e) ---------------------------------------------------------------------


def test_an_armed_monitor_alone_keeps_the_supervisor_and_sizes_its_sleep(tmp_path: Path) -> None:
    due = NOW_MS + 30 * 60_000
    _arm(tmp_path, "onlymonitor1", index_due=NOW_MS - DAY_MS, counters_due=due)
    index, _deliveries, _spooled, _triggers, monitors = _state(tmp_path)
    assert index == {}
    assert _fireable(tmp_path) is True
    assert _next_wake_ms(index, monitors=monitors) == due


# (f) ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_live_runtime_is_not_engaged_and_nothing_is_ledgered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_engage(session_id, cwd, work, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(session_id)

    async def live(config_dir, session_id):  # noqa: ANN001, ANN202
        return True

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    monkeypatch.setattr(sup, "_has_live_runtime", live)
    # Overdue past LIVE_STALL_WARN_S: the wake path would write an owed-fire
    # record here; a monitor must not.
    overdue = NOW_MS - 2 * DAY_MS
    _arm(tmp_path, "livesession1", index_due=overdue, counters_due=overdue)
    # The row IS due (so the no-engage below is the live rule, not a miss)...
    assert [row[0] for row in _due(tmp_path)] == ["livesession1"]

    # ...and the live runtime owns it: no engage, no ledger record.
    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 0
    assert calls == []
    from local_operator.wakes.deliveries import read_deliveries

    assert read_deliveries(tmp_path) == {}


@pytest.mark.asyncio
async def test_a_cold_monitor_session_is_engaged_with_a_monitor_schedule_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    works: list[Any] = []

    async def fake_engage(session_id, cwd, work, **_kwargs):  # noqa: ANN001, ANN202
        works.append((session_id, work))

    async def cold(config_dir, session_id):  # noqa: ANN001, ANN202
        return False

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    monkeypatch.setattr(sup, "_has_live_runtime", cold)
    _arm(tmp_path, "coldmonitor1", index_due=NOW_MS - 1_000, counters_due=NOW_MS - 1_000)

    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 1
    assert [(sid, work.schedule_id) for sid, work in works] == [("coldmonitor1", "monitor:m1")]
    from local_operator.wakes.deliveries import read_deliveries

    assert read_deliveries(tmp_path) == {}


@pytest.mark.asyncio
async def test_a_failed_monitor_engage_backs_off_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[str] = []

    async def failing(session_id, cwd, work, **_kwargs):  # noqa: ANN001, ANN202
        attempts.append(session_id)
        raise RuntimeError(f"could not start a runtime for session {session_id}: boom")

    async def cold(config_dir, session_id):  # noqa: ANN001, ANN202
        return False

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", failing)
    monkeypatch.setattr(sup, "_has_live_runtime", cold)
    _arm(tmp_path, "failingmon01", index_due=NOW_MS - 1_000, counters_due=NOW_MS - 1_000)

    await fire_due_wakes(tmp_path, now_ms=NOW_MS)
    assert attempts == ["failingmon01"]
    # Held by the backoff on the next pass, and nothing written to the ledger.
    assert sup._monitor_backoff.holds("failingmon01", NOW_MS + 1_000)
    from local_operator.wakes.deliveries import read_deliveries

    assert read_deliveries(tmp_path) == {}


def test_the_monitor_backoff_doubles_to_a_fifteen_minute_cap() -> None:
    backoff = sup._MonitorBackoff()
    delays = []
    for _ in range(10):
        backoff.note("s", "failed", now_ms=0)
        delays.append(backoff._delay_s["s"])
    assert delays[0] == sup.SLICE_S
    assert delays[1] == 2 * sup.SLICE_S
    assert max(delays) == 900.0
    backoff.note("s", "started", now_ms=0)
    assert not backoff.holds("s", 0)


def test_a_session_due_both_ways_is_engaged_once(tmp_path: Path) -> None:
    from local_operator.wakes.store import write_entry as write_wake_entry

    _arm(tmp_path, "bothways0001", index_due=NOW_MS - 1_000, counters_due=NOW_MS - 1_000)
    write_wake_entry(
        tmp_path,
        "bothways0001",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "x", "next_due_at": NOW_MS - 2_000, "created_at": 1}],
    )
    assert [row[0] for row in _due(tmp_path)] == ["bothways0001"]

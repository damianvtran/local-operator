"""Session PARK: free a runtime, keep its wakes and monitors armed (2026-11-12).

Covers the control ladder's park variant against a real control socket, the
index stamps (``parked_at`` is not a hold), the open-time clear, the deferred
self-park helper, and the fleet guard's park-before-kill rung.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_operator.monitors import store as monitor_store
from local_operator.paths import config_dir
from local_operator.session.runtime import control, park_after_turn, park_pressure
from local_operator.session.runtime.types import SessionRecord
from local_operator.wakes import store as wake_store
from tests.unit.session.runtime.test_control import _record_for, _serve
from tests.unit.session.runtime.test_graceful_only_stop import _IdleStopHandle

SID = "park00000001"


def _rec(session_id: str = SID, **overrides: Any) -> SessionRecord:
    fields: dict[str, Any] = dict(
        pid=999999,
        kind="daemon",
        session_id=session_id,
        conversation_name="parkable",
        cwd="/tmp",
        model_label="m",
        control_port=0,
        control_key="k",
        detached=True,
    )
    fields.update(overrides)
    return SessionRecord(**fields)


def _arm(root: Path, session_id: str = SID) -> None:
    wake_store.write_entry(
        root,
        session_id,
        cwd="/tmp",
        schedules=[{"id": "w1", "message": "prod window", "next_due_at": 1}],
    )
    monitor_store.write_entry(
        root,
        session_id,
        cwd="/tmp",
        monitors=[{"id": "m1", "name": "n", "tool": "bash", "arguments": {}, "every_ms": 60000}],
    )


# -- stamps --------------------------------------------------------------------


def test_park_stamps_parked_at_which_is_not_a_hold(tmp_path: Path) -> None:
    _arm(tmp_path)
    assert control._mark_wakes_dormant(_rec(), tmp_path, control.PARK_MARKER) == 1
    assert control._mark_monitors_dormant(_rec(), tmp_path, control.PARK_MARKER) == 1
    wake = wake_store.read_entry(tmp_path, SID)
    mon = monitor_store.read_entry(tmp_path, SID)
    assert wake and wake.get("parked_at") and not wake.get("stopped_at")
    assert mon and mon.get("parked_at") and not mon.get("stopped_at")
    # The whole point: every "will this fire" reader keeps firing it.
    assert not wake_store.is_held(wake)
    assert not monitor_store.is_held(mon)


def test_stop_still_stamps_the_dormant_hold(tmp_path: Path) -> None:
    """Negative control: stop semantics are unchanged (stop still means dormant)."""
    _arm(tmp_path)
    control._mark_wakes_dormant(_rec(), tmp_path)
    entry = wake_store.read_entry(tmp_path, SID)
    assert entry and wake_store.is_held(entry) and not entry.get("parked_at")


def test_armed_counts_reports_what_a_stop_would_make_dormant(tmp_path: Path) -> None:
    _arm(tmp_path)
    assert control.armed_counts(tmp_path, SID) == (1, 1)
    control._mark_wakes_dormant(_rec(), tmp_path)
    control._mark_monitors_dormant(_rec(), tmp_path)
    # Already held: a stop changes nothing for it.
    assert control.armed_counts(tmp_path, SID) == (0, 0)


def test_parked_line_wording() -> None:
    assert control.parked_line("x", 2, 1) == (
        'parked "x" — 2 wakes and 1 monitor stay armed; the supervisor reopens it when one fires'
    )
    assert "no wakes or monitors" in control.parked_line("x", 0, 0)


# -- the ladder ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_park_session_over_a_real_socket(busy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle → parked (stamps parked_at, never signals); busy → the stop's refusal."""
    handle = _IdleStopHandle(busy)
    server, record = await _serve(handle)
    root = config_dir()
    _arm(root, record.session_id)
    monkeypatch.setattr(
        control, "_signal_and_confirm", AsyncMock(side_effect=AssertionError("never signals"))
    )
    monkeypatch.setattr(
        control, "_await_stopped", AsyncMock(side_effect=lambda *a: bool(handle.stops))
    )
    try:
        outcome = await control.park_session(_record_for(record), timeout_s=1, _root=root)
    finally:
        server.close()
    wake = wake_store.read_entry(root, record.session_id)
    assert wake is not None
    if busy:
        assert outcome.method == "refused" and not outcome.parked
        assert not wake.get("parked_at") and not wake.get("stopped_at")
    else:
        assert outcome.parked and outcome.method == "socket"
        assert (outcome.wakes_armed, outcome.monitors_armed) == (1, 1)
        assert (outcome.wakes_dormant, outcome.monitors_dormant) == (0, 0)
        assert wake.get("parked_at") and not wake_store.is_held(wake)
        assert outcome.line.startswith(f'parked "{record.conversation_name}"')


# -- deferred self-park -------------------------------------------------------


@pytest.mark.asyncio
async def test_self_park_helper_waits_out_the_turn_then_parks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _rec()
    monkeypatch.setattr(park_after_turn, "_find_record", lambda *_a: rec)
    answers = [
        control.StopOutcome(rec.pid, SID, "x", "refused", "busy", reason=control.REASON_BUSY),
        control.StopOutcome(rec.pid, SID, "x", "refused", "busy", reason=control.REASON_BUSY),
        control.StopOutcome(rec.pid, SID, "x", "socket", "parked", parked=True),
    ]
    park = AsyncMock(side_effect=answers)
    monkeypatch.setattr(control, "park_session", park)
    outcome = await park_after_turn.run(rec.pid, SID, root=tmp_path, poll_s=0)
    assert outcome == "parked"
    assert park.await_count == 3


@pytest.mark.asyncio
async def test_self_park_helper_gives_up_on_a_due_wake_and_on_a_gone_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _rec()
    monkeypatch.setattr(park_after_turn, "_find_record", lambda *_a: rec)
    monkeypatch.setattr(
        control,
        "park_session",
        AsyncMock(
            return_value=control.StopOutcome(
                rec.pid, SID, "x", "refused", "wake", reason=control.REASON_WAKE
            )
        ),
    )
    assert await park_after_turn.run(rec.pid, SID, root=tmp_path, poll_s=0) == "refused:wake"
    monkeypatch.setattr(park_after_turn, "_find_record", lambda *_a: None)
    assert await park_after_turn.run(rec.pid, SID, root=tmp_path, poll_s=0) == "gone"


@pytest.mark.asyncio
async def test_self_park_helper_deadline_leaves_a_busy_session_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _rec()
    monkeypatch.setattr(park_after_turn, "_find_record", lambda *_a: rec)
    monkeypatch.setattr(
        control,
        "park_session",
        AsyncMock(
            return_value=control.StopOutcome(
                rec.pid, SID, "x", "refused", "busy", reason=control.REASON_BUSY
            )
        ),
    )
    outcome = await park_after_turn.run(rec.pid, SID, root=tmp_path, poll_s=0, deadline_s=0)
    assert outcome == "deadline"


# -- fleet guard --------------------------------------------------------------


def _touch_transcript(root: Path, session_id: str, mtime: float) -> None:
    import os

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "transcript.jsonl"
    path.write_text("{}\n")
    os.utime(path, (mtime, mtime))


def test_candidates_are_oldest_idle_first_and_never_busy_cos_or_watched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.aida import state as aida_state

    _touch_transcript(tmp_path, "old000000001", 1_000)
    _touch_transcript(tmp_path, "new000000001", 9_000)
    _touch_transcript(tmp_path, "aida00000001", 10)
    monkeypatch.setattr(aida_state, "is_aida_session", lambda _r, sid: sid == "aida00000001")
    records = [
        _rec("new000000001", pid=11),
        _rec("busy00000001", pid=12, busy=True),
        _rec("aida00000001", pid=13),
        _rec("old000000001", pid=14),
        _rec("tui000000001", pid=15, kind="tui"),
        _rec("seen00000001", pid=16, detached=False),
        _rec("jobs00000001", pid=17, jobs_running=1),
    ]
    keep, skipped = park_pressure.park_candidates(tmp_path, records)
    assert [r.session_id for r in keep] == ["old000000001", "new000000001"]
    reasons = dict(skipped)
    assert reasons["busy00000001"] == "busy"
    assert reasons["jobs00000001"] == "busy"
    assert reasons["aida00000001"] == "chief of staff"
    assert reasons["tui000000001"] == "viewer attached"
    assert reasons["seen00000001"] == "viewer attached"


@pytest.mark.asyncio
async def test_park_idle_sessions_parks_in_order_up_to_the_limit(tmp_path: Path) -> None:
    _touch_transcript(tmp_path, "a00000000001", 1)
    _touch_transcript(tmp_path, "b00000000001", 2)
    _touch_transcript(tmp_path, "c00000000001", 3)
    records = [
        _rec("c00000000001", pid=3),
        _rec("a00000000001", pid=1),
        _rec("b00000000001", pid=2),
    ]
    calls: list[str] = []

    async def _park(record: Any, **_kw: Any) -> Any:
        calls.append(record.session_id)
        if record.session_id == "a00000000001":
            # The owner's own latch refuses (a turn arrived): never forced.
            return control.StopOutcome(
                record.pid, record.session_id, "x", "refused", "", reason="busy"
            )
        return control.StopOutcome(record.pid, record.session_id, "x", "socket", "", parked=True)

    report = await park_pressure.park_idle_sessions(tmp_path, records=records, park=_park, limit=1)
    assert calls == ["a00000000001", "b00000000001"]
    assert report.parked == ["b00000000001"]
    assert report.refused == [("a00000000001", "busy")]


def test_disk_pressure_reads_swap_sharing_the_data_volume(tmp_path: Path) -> None:
    gib = 1024**3
    # The 2026-11-12 shape: swap 13.5 GiB, shared container with 8.2 GiB free.
    shared = park_pressure.measure_disk_pressure(
        tmp_path,
        swap_probe=lambda: int(13.5 * gib),
        free_probe=lambda _p: int(8.2 * gib),
        swap_volume=str(tmp_path),
    )
    assert shared.under_pressure and shared.swap_shares_volume
    assert "swap" in shared.reason
    calm = park_pressure.measure_disk_pressure(
        tmp_path,
        swap_probe=lambda: 1 * gib,
        free_probe=lambda _p: 50 * gib,
        swap_volume=str(tmp_path),
    )
    assert not calm.under_pressure
    low = park_pressure.measure_disk_pressure(
        tmp_path, swap_probe=lambda: 0, free_probe=lambda _p: gib, swap_volume="/nonexistent"
    )
    assert low.under_pressure and "< 2 GiB" in low.reason
    unknown = park_pressure.measure_disk_pressure(
        tmp_path, swap_probe=lambda: 0, free_probe=lambda _p: None, swap_volume="/nonexistent"
    )
    assert not unknown.under_pressure  # unknown never acts

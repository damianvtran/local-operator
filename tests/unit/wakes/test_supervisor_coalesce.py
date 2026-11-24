"""Coalescing a session's wakes that come due close together into one cold boot.

WHY (wake boot economy, 2026-11-24): a sibling wake due a few minutes after the
first finds the runtime idle-exited and pays its own cold boot. The supervisor
now holds a due ONE-SHOT wake until the latest sibling due within
``WAKE_COALESCE_WINDOW_MS`` (15 min), so one engagement carries them all. The
runtime's load folds every overdue schedule into ONE turn in due order, each
with its own id and original due time, and fires each schedule exactly as it
would have fired alone.

Invariants pinned here: never held past the window; recurring, operator,
notify and internal rows are never held; an owed (ledgered) fire is never held;
a wake outside the window keeps its own engagement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.wake import (
    COLD_BOOT_DELIVERY_MS,
    WAKE_COALESCE_WINDOW_MS,
    WakeSchedule,
    WakeScheduler,
)
from local_operator.session.session import Session
from local_operator.wakes import supervisor as sup
from local_operator.wakes.store import write_entry

NOW_MS = 1_800_000_000_000
MIN = 60_000


def _row(wake_id: str, due: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": wake_id,
        "message": f"message of {wake_id}",
        "next_due_at": due,
        "created_at": 1,
        "origin": "tool",
        **extra,
    }


def _arm(root: Path, session_id: str, rows: list[dict[str, Any]]) -> None:
    from local_operator.resume import TRANSCRIPT_NAME

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TRANSCRIPT_NAME).write_text("", encoding="utf-8")
    write_entry(root, session_id, cwd=str(root), schedules=rows)


def _due(root: Path, now_ms: int) -> list[tuple[str, str, int]]:
    index, deliveries, spooled, triggers, monitors = sup._load_and_reconcile_state(root)
    return sup._due_sessions(
        index,
        now_ms,
        config_dir=root,
        deliveries=deliveries,
        spooled=spooled,
        triggers=triggers,
        monitors=monitors,
    )


@pytest.fixture
def engagements(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_engage(session_id, cwd, work, **_kwargs):  # noqa: ANN001, ANN202
        calls.append({"session_id": session_id, "occurrence_ms": work.occurrence_ms})

    async def cold(_config_dir, _session_id):  # noqa: ANN001, ANN202
        return False

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    monkeypatch.setattr(sup, "_has_live_runtime", cold)
    return calls


def test_the_window_is_the_measured_fifteen_minutes_and_the_label_bound_covers_it() -> None:
    assert WAKE_COALESCE_WINDOW_MS == 15 * MIN
    # The "cold-boot delivery, not a miss" bound is the hold plus the engage
    # deadline and one slice; pin it against the supervisor's own constants.
    assert COLD_BOOT_DELIVERY_MS == (
        WAKE_COALESCE_WINDOW_MS + int(sup.WAKE_DEADLINE_S * 1000) + int(sup.SLICE_S * 1000)
    )


@pytest.mark.asyncio
async def test_three_wakes_within_the_window_make_one_engagement_at_the_last_due(
    tmp_path: Path, engagements: list[dict[str, Any]]
) -> None:
    first, second, third = NOW_MS, NOW_MS + 2 * MIN, NOW_MS + 9 * MIN
    _arm(tmp_path, "coalesce0001", [_row("w1", first), _row("w2", second), _row("w3", third)])

    # Due, but held: the later siblings are inside the window.
    assert await sup.fire_due_wakes(tmp_path, now_ms=first + 1_000) == 0
    assert await sup.fire_due_wakes(tmp_path, now_ms=second + 1_000) == 0
    assert engagements == []
    # The last one comes due: ONE engagement, for the earliest occurrence.
    assert await sup.fire_due_wakes(tmp_path, now_ms=third + 1_000) == 1
    assert engagements == [{"session_id": "coalesce0001", "occurrence_ms": first}]


@pytest.mark.asyncio
async def test_wakes_outside_the_window_keep_separate_engagements(
    tmp_path: Path, engagements: list[dict[str, Any]]
) -> None:
    """Negative control: 16 min apart is outside the 15 min window."""
    _arm(tmp_path, "coalesce0002", [_row("w1", NOW_MS), _row("w2", NOW_MS + 16 * MIN)])
    assert await sup.fire_due_wakes(tmp_path, now_ms=NOW_MS + 1_000) == 1
    assert engagements == [{"session_id": "coalesce0002", "occurrence_ms": NOW_MS}]


def test_a_hold_never_exceeds_the_window_even_when_wakes_chain(tmp_path: Path) -> None:
    """Every 10 min for 30 min: the cluster is anchored on the first due and ends
    at first + window, so the first wake waits 10 min, never 20 or 30."""
    rows = [_row(f"w{i}", NOW_MS + i * 10 * MIN) for i in range(4)]
    _arm(tmp_path, "coalesce0003", rows)
    assert _due(tmp_path, NOW_MS + 10 * MIN - 1) == []
    due = _due(tmp_path, NOW_MS + 10 * MIN)
    assert [(sid, at) for sid, _cwd, at in due] == [("coalesce0003", NOW_MS)]
    assert (
        sup._coalesced_release_ms({"schedules": rows}, NOW_MS) - NOW_MS <= WAKE_COALESCE_WINDOW_MS
    )


@pytest.mark.parametrize(
    "held_row",
    [
        pytest.param({"every_ms": 15 * MIN}, id="recurring"),
        pytest.param({"origin": "cli"}, id="operator-authored"),
        pytest.param({"notify": True}, id="notify"),
        pytest.param({"kind": "patience", "hidden": True}, id="internal-timer"),
    ],
)
def test_rows_that_must_keep_their_time_are_never_held(
    tmp_path: Path, held_row: dict[str, Any]
) -> None:
    _arm(tmp_path, "coalesce0004", [_row("w1", NOW_MS, **held_row), _row("w2", NOW_MS + 5 * MIN)])
    assert [row[0] for row in _due(tmp_path, NOW_MS + 1_000)] == ["coalesce0004"]


def test_a_recurring_wake_may_be_the_one_a_one_shot_waits_for(tmp_path: Path) -> None:
    """Mixed: the one-shot is held; the recurring row fires at its own time."""
    _arm(
        tmp_path,
        "coalesce0005",
        [_row("w1", NOW_MS), _row("w2", NOW_MS + 4 * MIN, every_ms=30 * MIN)],
    )
    assert _due(tmp_path, NOW_MS + 1_000) == []
    assert [row[0] for row in _due(tmp_path, NOW_MS + 4 * MIN)] == ["coalesce0005"]


def test_an_owed_fire_with_a_ledger_record_is_never_held(tmp_path: Path) -> None:
    from local_operator.wakes import deliveries as ledger

    _arm(tmp_path, "coalesce0006", [_row("w1", NOW_MS), _row("w2", NOW_MS + 5 * MIN)])
    ledger.note_failure(tmp_path, "coalesce0006", NOW_MS, error="boom", now_ms=NOW_MS)
    index, deliveries, *_rest = sup._load_and_reconcile_state(tmp_path)
    record = deliveries.get("coalesce0006")
    assert record is not None
    # Past the record's backoff, the owed fire engages without waiting for w2.
    later = int(record.get("next_attempt_ms") or NOW_MS) + 1
    assert later < NOW_MS + 5 * MIN, "the test needs the retry inside the window"
    assert [row[0] for row in _due(tmp_path, later)] == ["coalesce0006"]


# --- the delivered turn and the ledger ----------------------------------------


def _schedules(rows: list[dict[str, Any]]) -> list[WakeSchedule]:
    return [WakeSchedule.model_validate(row) for row in rows]


async def _load_and_fire(rows: list[dict[str, Any]], now_ms: int) -> tuple[Any, ...]:
    """A cold-booted runtime: load the persisted rows at ``now_ms``, fire all due."""
    delivered: list[str] = []
    persisted: list[list[WakeSchedule]] = []
    clock = {"now": now_ms}
    scheduler = WakeScheduler(
        now=lambda: clock["now"],
        deliver=lambda due: delivered.append(due.schedule.id),
        persist=lambda schedules: persisted.append(list(schedules)),
    )
    try:
        scheduler.load(_schedules(rows))
        missed = scheduler.take_missed()
        clock["now"] = now_ms + 5_000  # past LOAD_GRACE_MS
        await scheduler.pump()
        return delivered, missed, list(scheduler.schedules)
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_one_boot_fires_each_schedule_as_separate_boots_would() -> None:
    """Ledger parity: the coalesced boot fires and retires each one-shot once
    and leaves the recurring row exactly where its own (unheld) boot would."""
    rows = [
        _row("w1", NOW_MS),
        _row("w2", NOW_MS + 2 * MIN),
        _row("w3", NOW_MS + 6 * MIN, every_ms=30 * MIN),
    ]
    release = sup._coalesced_release_ms({"schedules": rows}, NOW_MS)
    assert release == NOW_MS + 6 * MIN

    coalesced, missed, remaining = await _load_and_fire(rows, release + 12_000)
    assert sorted(coalesced) == ["w1", "w2", "w3"]

    separate: dict[str, Any] = {}
    for row in rows:
        fired, _missed, left = await _load_and_fire([row], row["next_due_at"] + 12_000)
        assert fired == [row["id"]]
        separate[row["id"]] = left
    # One-shots retire in both; the recurring row (never held: it IS the
    # release) ends with the same fired_count and next_due_at.
    assert separate["w1"] == [] and separate["w2"] == []
    assert [s.model_dump() for s in remaining] == [s.model_dump() for s in separate["w3"]]


def test_the_delivered_turn_lists_every_wake_in_due_order_with_its_due_time() -> None:
    """Rows created in reverse order still read in the order they came due."""
    rows = [
        WakeSchedule(id="w9", message="third", next_due_at=NOW_MS + 9 * MIN, created_at=1),
        WakeSchedule(id="w5", message="second", next_due_at=NOW_MS + 2 * MIN, created_at=2),
        WakeSchedule(id="w2", message="first", next_due_at=NOW_MS, created_at=3),
    ]
    missed: list[Any] = [{"schedule": s, "occurrences": 1, "due": 1} for s in rows]
    text = Session._format_missed_wake_catchup(missed, NOW_MS + 9 * MIN + 12_000)
    order = [text.index(f"- {wid} (due ") for wid in ("w2", "w5", "w9")]
    assert order == sorted(order)
    for wid, message in (("w2", "first"), ("w5", "second"), ("w9", "third")):
        assert f"Message: {message}" in text
    assert "delivered by the wake supervisor's start" in text
    # A punctual cold-boot delivery is not a "missed" wake (the 316765a40496 bug).
    assert "missed while the session was down" not in text
    assert "handle them as missed wakes" not in text


def test_a_wake_overdue_past_the_cold_boot_bound_is_still_reported_missed() -> None:
    """Negative control: real downtime keeps the missed-wake wording."""
    schedule = WakeSchedule(id="w1", message="m", next_due_at=NOW_MS, created_at=1)
    text = Session._format_missed_wake_catchup(
        [{"schedule": schedule, "occurrences": 1, "due": 1}],
        NOW_MS + COLD_BOOT_DELIVERY_MS + 1,
    )
    assert "missed while the session was down" in text
    assert "handle them as missed wakes" in text

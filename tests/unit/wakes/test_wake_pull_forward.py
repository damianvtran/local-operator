"""One cold boot delivers a session's wakes due soon after: pulled forward, never held.

WHY (wake boot economy, 2026-11-24): a sibling wake due a few minutes after the
one a cold boot was paid for found the runtime idle-exited (~147 s resident)
and paid its own boot (median 11.4 s). A runtime the wake supervisor started now
folds the session's wakes due within ``WAKE_COALESCE_WINDOW_MS`` into the same
catch-up turn. On the scout's fire timelines this removes 118 of 2,878 boots
(4.1%), where HOLDING one-shots for their siblings removed 20 (0.7%).

Pinned here (Aida's conditions): every original message is delivered, in due
order, each with its own id and original due time; each schedule fires once
and keeps the ledger an on-time fire would leave (one-shots retire, a recurring
row advances from its OWN due time, so its cadence does not drift); operator,
notify and internal rows are never pulled; a runtime the supervisor did not
start pulls nothing.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from local_operator.harness.wake import (
    COLD_BOOT_DELIVERY_MS,
    LOAD_GRACE_MS,
    WAKE_COALESCE_WINDOW_MS,
    WakeSchedule,
    WakeScheduler,
    take_supervisor_started,
)
from local_operator.session.session import Session
from local_operator.wakes import owned_runtimes
from local_operator.wakes import supervisor as sup

T = 1_800_000_000_000
MIN = 60_000


def _row(wake_id: str, due: int, **extra: Any) -> WakeSchedule:
    return WakeSchedule.model_validate(
        {
            "id": wake_id,
            "message": f"message of {wake_id}",
            "next_due_at": due,
            "created_at": 1,
            "origin": "tool",
            **extra,
        }
    )


class _Runtime:
    """A cold-booted runtime's wake scheduler on a controllable clock."""

    def __init__(self, now: int) -> None:
        self.now = now
        self.delivered: list[str] = []
        self.scheduler = WakeScheduler(
            now=lambda: self.now,
            deliver=lambda due: self.delivered.append(due.schedule.id),
            persist=lambda _schedules: None,
        )

    async def boot(self, rows: list[WakeSchedule], *, pull: bool) -> list[Any]:
        self.scheduler.load(rows, pull_forward_ms=WAKE_COALESCE_WINDOW_MS if pull else 0)
        missed = self.scheduler.take_missed()
        self.now += 5_000  # past LOAD_GRACE_MS: the folded rows fire now
        await self.scheduler.pump()
        return missed

    def rows(self) -> dict[str, WakeSchedule]:
        return {row.id: row for row in self.scheduler.schedules}

    def close(self) -> None:
        self.scheduler.dispose()


def test_the_window_and_the_cold_boot_bound_match_the_supervisor() -> None:
    assert WAKE_COALESCE_WINDOW_MS == 15 * MIN
    assert COLD_BOOT_DELIVERY_MS == int(sup.WAKE_DEADLINE_S * 1000) + int(sup.SLICE_S * 1000)


@pytest.mark.asyncio
async def test_a_supervisor_boot_delivers_wakes_due_within_the_window_in_one_turn() -> None:
    runtime = _Runtime(T + 12_000)  # booted for w1, 12 s after it came due
    try:
        missed = await runtime.boot(
            [_row("w1", T), _row("w2", T + 2 * MIN), _row("w3", T + 20 * MIN)], pull=True
        )
        assert [(m["schedule"].id, bool(m.get("early"))) for m in missed] == [
            ("w1", False),
            ("w2", True),
        ]
        # Both fire once with this boot; w3 (outside the window) keeps its own time.
        assert sorted(runtime.delivered) == ["w1", "w2"]
        assert list(runtime.rows()) == ["w3"]
        assert runtime.rows()["w3"].next_due_at == T + 20 * MIN
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_without_the_supervisor_stamp_nothing_is_pulled() -> None:
    """Negative control: an interactive resume delivers nothing early."""
    runtime = _Runtime(T + 12_000)
    try:
        missed = await runtime.boot([_row("w1", T), _row("w2", T + 2 * MIN)], pull=False)
        assert [m["schedule"].id for m in missed] == ["w1"]
        assert runtime.delivered == ["w1"]
        assert runtime.rows()["w2"].next_due_at == T + 2 * MIN
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_pulled_recurring_row_keeps_its_cadence_and_ledger() -> None:
    """Ledger parity: the pulled fire leaves exactly what its on-time fire would."""
    every = 15 * MIN
    recurring = _row("w6", T + 3 * MIN, every_ms=every, fired_count=4)

    pulled = _Runtime(T + 12_000)
    on_time = _Runtime(T)  # loaded before w6 is due, as a live runtime holds it
    try:
        await pulled.boot([_row("w2", T), recurring], pull=True)
        # The on-time reference: a live runtime firing w6 exactly at its due.
        on_time.scheduler.load([recurring])
        await on_time.scheduler.pump(T + 3 * MIN)
        assert pulled.delivered.count("w6") == on_time.delivered.count("w6") == 1
        assert pulled.rows()["w6"].model_dump() == on_time.rows()["w6"].model_dump()
        assert pulled.rows()["w6"].next_due_at == T + 3 * MIN + every
        assert pulled.rows()["w6"].fired_count == 5
    finally:
        pulled.close()
        on_time.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger_every", [None, 30 * MIN], ids=["one-shot", "recurring"])
async def test_a_recurring_trigger_opens_the_pull_forward_like_a_one_shot(
    trigger_every: int | None,
) -> None:
    """Round 1, Q1: the common agent wake (recurring, seconds overdue) must pull too.

    A recurring row overdue by less than one period skips nothing, so it never
    entered ``missed``, and keying the trigger on ``missed`` pulled nothing.
    Every row's ledger is pinned: one-shots retire, recurring rows (trigger
    included) advance from their OWN due time, the out-of-window row is untouched.
    """
    extra = {"every_ms": trigger_every} if trigger_every else {}
    runtime = _Runtime(T + 12_000)  # booted for w1, 12 s after it came due
    try:
        missed = await runtime.boot(
            [
                _row("w1", T, **extra),
                _row("w2", T + 2 * MIN, every_ms=30 * MIN),  # recurring sibling
                _row("w3", T + 10 * MIN),  # one-shot sibling
                _row("w4", T + 20 * MIN, every_ms=30 * MIN),  # outside the window
            ],
            pull=True,
        )
        assert sorted((m["schedule"].id, bool(m.get("early"))) for m in missed) == [
            ("w1", False),
            ("w2", True),
            ("w3", True),
        ]
        assert sorted(runtime.delivered) == ["w1", "w2", "w3"]
        # Every row's next due, relative to T.
        expected = {"w2": 32 * MIN, "w4": 20 * MIN}
        if trigger_every:
            expected["w1"] = 30 * MIN  # its own due + period, NOT load time + period
        assert {wid: r.next_due_at - T for wid, r in runtime.rows().items()} == expected
        # ONE turn: the catch-up names all three and none as missed.
        text = Session._format_missed_wake_catchup(missed, T + 12_000)
        assert all(f"- {wid} (due " in text for wid in ("w1", "w2", "w3"))
        assert "missed while the session was down" not in text
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_recurring_trigger_without_the_supervisor_stamp_pulls_nothing() -> None:
    """Negative control for Q1: an interactive resume keeps today's behaviour."""
    runtime = _Runtime(T + 12_000)
    try:
        missed = await runtime.boot(
            [_row("w1", T, every_ms=30 * MIN), _row("w2", T + 2 * MIN, every_ms=30 * MIN)],
            pull=False,
        )
        assert missed == []
        assert runtime.delivered == ["w1"]
        # Load-time phase on an ordinary resume (pre-existing, unchanged): the
        # re-armed due (load + LOAD_GRACE_MS) plus one period.
        assert {wid: r.next_due_at - T for wid, r in runtime.rows().items()} == {
            "w1": 12_000 + LOAD_GRACE_MS + 30 * MIN,
            "w2": 2 * MIN,
        }
    finally:
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [{"origin": "cli"}, {"origin": "desktop"}, {"notify": True}, {"hidden": True}],
)
async def test_operator_notify_and_hidden_rows_are_never_pulled(extra: dict[str, Any]) -> None:
    runtime = _Runtime(T + 12_000)
    try:
        missed = await runtime.boot([_row("w1", T), _row("w2", T + 2 * MIN, **extra)], pull=True)
        assert [m["schedule"].id for m in missed] == ["w1"]
        assert runtime.rows()["w2"].next_due_at == T + 2 * MIN
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_boot_with_no_visible_overdue_wake_pulls_nothing() -> None:
    """A monitor or spooled-turn boot has no catch-up turn to fold into."""
    runtime = _Runtime(T)
    try:
        missed = await runtime.boot([_row("w2", T + 2 * MIN)], pull=True)
        assert missed == []
        assert runtime.delivered == []
    finally:
        runtime.close()


def test_the_turn_lists_every_wake_in_due_order_with_its_original_due_time() -> None:
    now = T + 12_000
    rows = [
        WakeSchedule(id="w9", message="third", next_due_at=T + 9 * MIN, created_at=1),
        WakeSchedule(id="w5", message="second", next_due_at=T + 2 * MIN, created_at=2),
        WakeSchedule(id="w2", message="first", next_due_at=T, created_at=3),
    ]
    missed: list[Any] = [
        {"schedule": rows[0], "occurrences": 1, "due": 1, "early": True},
        {"schedule": rows[1], "occurrences": 1, "due": 1, "early": True},
        {"schedule": rows[2], "occurrences": 1, "due": 1},
    ]
    text = Session._format_missed_wake_catchup(missed, now)
    order = [text.index(f"- {wid} (due ") for wid in ("w2", "w5", "w9")]
    assert order == sorted(order)
    for message in ("first", "second", "third"):
        assert f"Message: {message}" in text
    assert text.count("not due yet; delivered early") == 2
    assert "delivered at this start" in text
    # A punctual cold-boot delivery is not a "missed" wake (the 316765a40496 bug).
    assert "missed while the session was down" not in text
    assert "handle them as missed wakes" not in text


def test_a_wake_overdue_past_the_cold_boot_bound_is_still_reported_missed() -> None:
    """Negative control: real downtime keeps the missed-wake wording."""
    schedule = WakeSchedule(id="w1", message="m", next_due_at=T, created_at=1)
    text = Session._format_missed_wake_catchup(
        [{"schedule": schedule, "occurrences": 1, "due": 1}], T + COLD_BOOT_DELIVERY_MS + 1
    )
    assert "missed while the session was down" in text
    assert "handle them as missed wakes" in text


def test_only_the_supervisor_stamps_the_started_flag_and_it_is_read_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env: dict[str, str] = {owned_runtimes.SUPERVISOR_STARTED_ENV: "1"}
    monkeypatch.setenv(owned_runtimes.OWNER_ENV, "token")
    monkeypatch.setenv(owned_runtimes.SUPERVISOR_PID_ENV, str(os.getpid() + 1))
    owned_runtimes.stamp_child_env(env)  # not the supervisor: stripped
    assert owned_runtimes.SUPERVISOR_STARTED_ENV not in env
    monkeypatch.setenv(owned_runtimes.SUPERVISOR_PID_ENV, str(os.getpid()))
    owned_runtimes.stamp_child_env(env)  # the supervisor itself: stamped
    assert env[owned_runtimes.SUPERVISOR_STARTED_ENV] == "1"

    monkeypatch.setenv(owned_runtimes.SUPERVISOR_STARTED_ENV, "1")
    assert take_supervisor_started() is True
    assert take_supervisor_started() is False  # consumed: no child inherits it
    assert owned_runtimes.SUPERVISOR_STARTED_ENV not in os.environ

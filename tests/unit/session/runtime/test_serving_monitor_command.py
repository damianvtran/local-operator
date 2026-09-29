"""The runtime's ``monitor`` command: one mutation, inside the owning session.

This is the ladder word the desktop's monitor write routes reach
(``route_shared_slash("monitor", …)``) when a live runtime holds the session,
and the reason it exists is that ``Session._persist_monitor_schedules`` is the
ONE writer of monitor state: a live session republishes its whole in-memory list
on its next persist, so an external append would be deleted by it while nothing
ticked the watch either (the wound ``monitors/arm._refuse_if_owned`` names).

What this file pins is the contract between that word and the route: the ops it
understands, that create/cancel go through the SAME ``tools/builtin`` helpers
the agent's tool runs — against a REAL scheduler, so validation, dedupe and the
never-reused id sequence are the live implementations — that the persisted list
is the full-list snapshot the transcript and index are written from, and that
every refusal carries a machine-readable code (the route maps the code to a
status; the prose is only for a human).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.monitors.scheduler import CheckOutcome, MonitorScheduler
from local_operator.monitors.settings import MonitorSettings
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.frontend_state import SlashResult
from local_operator.session.runtime.serving import ServingSessionHandle

NOW = 1_700_000_000_000


class _Session:
    """The two things the slash reads off a session: its scheduler and cwd."""

    def __init__(self, scheduler: MonitorScheduler, cwd: str = "/repo") -> None:
        self.monitor_scheduler = scheduler
        self._cwd = cwd


def _scheduler(
    tmp_path: Any, *, settings: MonitorSettings | None = None
) -> tuple[MonitorScheduler, list[list[MonitorSpec]]]:
    """A real scheduler whose persist records the full-list snapshots.

    ``_persist_monitor_schedules`` is what the session hands this callback
    slot; recording the argument here is how the tests see the exact list a
    transcript append and an index rewrite would be written from.
    """
    persisted: list[list[MonitorSpec]] = []

    async def persist(monitors: list[MonitorSpec]) -> None:
        persisted.append(list(monitors))

    async def run_check(monitor: MonitorSpec) -> CheckOutcome:
        return {"text": "same", "error": None}

    scheduler = MonitorScheduler(
        now=lambda: NOW,
        config_dir=tmp_path / "cfg",
        session_id="sess",
        settings=settings or MonitorSettings(),
        validate=lambda tool, arguments: None,
        run_check=run_check,
        deliver=lambda delivery: None,
        persist=persist,
    )
    return scheduler, persisted


def _handle() -> ServingSessionHandle:
    """The handler without its constructor. Reaching the ladder branch is the
    whole test; standing up a socket, a process and a session to reach one
    ``if`` would test the harness instead."""
    return ServingSessionHandle.__new__(ServingSessionHandle)


async def _run(session: Any, payload: dict[str, Any]) -> SlashResult:
    return await _handle()._monitor_slash(session, json.dumps(payload), SlashResult)


_CREATE = {
    "op": "create",
    "request": {"tool": "bash", "arguments": {"command": "date -u"}, "name": "date-watch"},
}


@pytest.mark.asyncio
async def test_create_lands_through_the_scheduler_and_persists_the_full_list(
    tmp_path: Any,
) -> None:
    scheduler, persisted = _scheduler(tmp_path)
    session = _Session(scheduler)

    result = await _run(session, _CREATE)

    assert result.kind == "notice"
    assert result.data["monitor_id"] == "m1"
    assert result.data["name"] == "date-watch"
    assert result.data["remaining"] == 1
    assert result.data["already_armed"] is False
    assert result.data["reactivated"] is False
    # ONE persist, holding the FULL list: the transcript append and the index
    # rewrite are both written from it, so disk and memory cannot disagree.
    assert [[row.id for row in rows] for rows in persisted] == [["m1"]]
    assert persisted[0][0].cwd == "/repo"
    runtime = scheduler.runtime("m1")
    assert runtime is not None
    assert result.data["next_due_at"] == runtime.counters.get("next_due_at")


@pytest.mark.asyncio
async def test_a_retried_arm_is_answered_by_the_dedupe_not_a_second_persist(
    tmp_path: Any,
) -> None:
    """The identity IS the idempotency key: the second identical request finds
    the spec in force and writes nothing at all — the same answer the file
    writer's cold path gives."""
    scheduler, persisted = _scheduler(tmp_path)
    session = _Session(scheduler)

    first = await _run(session, _CREATE)
    second = await _run(session, _CREATE)

    assert first.data["already_armed"] is False
    assert second.kind == "notice"
    assert second.data["monitor_id"] == "m1"
    assert second.data["already_armed"] is True
    assert second.data["reactivated"] is False
    assert second.data["remaining"] == 1
    # The dedupe answer persists nothing — no second snapshot was ever built.
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_cancel_drops_the_row_and_persists_the_shrunken_list(tmp_path: Any) -> None:
    scheduler, persisted = _scheduler(tmp_path)
    session = _Session(scheduler)
    await _run(session, _CREATE)

    result = await _run(session, {"op": "cancel", "monitor_id": "m1"})

    assert result.kind == "notice"
    assert result.data["monitor_id"] == "m1"
    assert result.data["name"] == "date-watch"
    assert result.data["remaining"] == 0
    assert result.data["next_due_at"] is None
    assert persisted[-1] == []
    assert scheduler.monitors == ()


@pytest.mark.asyncio
async def test_a_cancel_for_an_unknown_handle_is_monitor_not_found(tmp_path: Any) -> None:
    scheduler, persisted = _scheduler(tmp_path)
    session = _Session(scheduler)

    result = await _run(session, {"op": "cancel", "monitor_id": "m9"})

    assert result.kind == "error"
    assert result.data["code"] == "monitor_not_found"
    assert "m9" in result.text
    assert persisted == []


@pytest.mark.asyncio
async def test_the_cap_and_a_bad_duration_are_refused_by_kind(tmp_path: Any) -> None:
    """``monitor_refused`` (a conflict with the session's own state) and
    ``monitor_invalid`` (this request cannot be read) are different statuses on
    the route, so the code has to be the fault marker the shared helper sets —
    not a match on the sentence."""
    scheduler, persisted = _scheduler(tmp_path, settings=MonitorSettings(max_monitors=1))
    session = _Session(scheduler)
    await _run(session, _CREATE)

    over_cap = await _run(
        session,
        {
            "op": "create",
            "request": {"tool": "bash", "arguments": {"command": "ls new"}, "name": "second"},
        },
    )
    bad_duration = await _run(
        session,
        {
            "op": "create",
            "request": {"tool": "bash", "arguments": {"command": "ls"}, "every": "soon"},
        },
    )

    assert over_cap.kind == "error"
    assert over_cap.data["code"] == "monitor_refused"
    assert "monitor limit reached" in over_cap.text
    assert bad_duration.kind == "error"
    assert bad_duration.data["code"] == "monitor_invalid"
    # Neither refusal wrote: the write log stands still.
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_a_payload_this_handler_cannot_read_is_refused_not_raised(
    tmp_path: Any,
) -> None:
    """The caller is our own route, so a shape it cannot produce is a protocol
    bug — refused in the same typed envelope as everything else rather than
    raised as a traceback through the transport."""
    scheduler, persisted = _scheduler(tmp_path)
    session = _Session(scheduler)

    for payload in (
        {"op": "edit", "monitor_id": "m1"},
        {"op": "create", "request": [1]},
        {"op": "create"},
    ):
        result = await _run(session, payload)
        assert result.kind == "error", payload
        assert result.data["code"] in {"monitor_invalid", "monitor_refused"}, payload
    assert persisted == []


@pytest.mark.asyncio
async def test_a_session_with_no_scheduler_says_so(tmp_path: Any) -> None:
    class _Bare:
        monitor_scheduler = None

    result = await _handle()._monitor_slash(_Bare(), json.dumps({"op": "create"}), SlashResult)

    assert result.kind == "error"
    assert result.data["code"] == "monitor_unavailable"

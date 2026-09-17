"""The runtime's ``wake`` command: one mutation, inside the owning session.

This is the ladder word the desktop's write routes reach
(``route_shared_slash("wake", …)``) when a live runtime holds the session, and
the reason it exists is that ``Session._persist_wake_schedules`` is the ONE
writer of schedule state: a live session republishes its whole in-memory list on
its next persist, so an external append would be deleted by it while the
supervisor — which skips any session with a live record — never fired it either.

What this file pins is the contract between that word and the route: the ops it
understands, that create/cancel go through the SAME ``tools/builtin`` helpers
the agent's tool runs, and that every refusal carries a machine-readable code
(the route maps the code to a status; the prose is only for a human).
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from local_operator.harness.wake import MAX_WAKE_SCHEDULES, WakeSchedule
from local_operator.session.frontend_state import SlashResult
from local_operator.session.runtime.serving import ServingSessionHandle

MINUTE = 60_000


class _Scheduler:
    """The two things the builtin helpers use: the current rows, and ``update``."""

    def __init__(self, rows: list[WakeSchedule] | None = None) -> None:
        self.schedules = list(rows or [])
        self.updates: list[list[WakeSchedule]] = []

    async def update(self, rows: list[WakeSchedule]) -> None:
        self.updates.append(list(rows))
        self.schedules = list(rows)


class _Session:
    def __init__(self, rows: list[WakeSchedule] | None = None) -> None:
        self.wake_scheduler = _Scheduler(rows)

    @property
    def scheduler(self) -> _Scheduler:
        return self.wake_scheduler


def _row(wake_id: str, *, due_ms: int | None = None, **extra) -> WakeSchedule:
    return WakeSchedule(
        id=wake_id,
        message=f"message {wake_id}",
        next_due_at=due_ms if due_ms is not None else int(time.time() * 1000) + 30 * MINUTE,
        created_at=1_700_000_000_000,
        **extra,
    )


def _handle() -> ServingSessionHandle:
    """The handler without its constructor. Reaching the ladder branch is the
    whole test; standing up a socket, a process and a session to reach one
    ``if`` would test the harness instead."""
    return ServingSessionHandle.__new__(ServingSessionHandle)


async def _run(session: _Session, payload: dict[str, Any]) -> SlashResult:
    return await _handle()._wake_slash(session, json.dumps(payload), SlashResult)


@pytest.mark.asyncio
async def test_create_lands_through_the_tools_own_helper() -> None:
    session = _Session()

    result = await _run(
        session, {"op": "create", "request": {"message": "check the build", "in": "30m"}}
    )

    assert result.kind == "notice"
    assert result.data["wake_id"] == "w1"
    assert result.data["next_due_at"] == session.scheduler.schedules[0].next_due_at
    # ONE update, holding the full list: the scheduler persists the whole
    # snapshot, which is what makes the transcript and the index agree.
    assert len(session.scheduler.updates) == 1
    assert [row.id for row in session.scheduler.updates[0]] == ["w1"]


@pytest.mark.asyncio
async def test_the_cap_and_a_bad_duration_are_refused_by_kind() -> None:
    """``wake_refused`` (a conflict with the session's own state) and
    ``wake_invalid`` (this request cannot be read) are different statuses on the
    route, so the code has to be the fault marker the shared helpers set — not a
    match on the sentence."""
    full = _Session([_row(f"w{i}") for i in range(1, MAX_WAKE_SCHEDULES + 1)])
    session = _Session()

    over_cap = await _run(full, {"op": "create", "request": {"message": "x", "in": "30m"}})
    bad_duration = await _run(session, {"op": "create", "request": {"message": "x", "in": "soon"}})

    assert over_cap.kind == "error"
    assert over_cap.data["code"] == "wake_refused"
    assert str(MAX_WAKE_SCHEDULES) in over_cap.text
    assert full.scheduler.updates == []
    assert bad_duration.kind == "error"
    assert bad_duration.data["code"] == "wake_invalid"
    assert session.scheduler.updates == []


@pytest.mark.asyncio
async def test_cancel_drops_the_row_and_reports_an_unknown_handle() -> None:
    session = _Session([_row("w1"), _row("w2")])

    missing = await _run(session, {"op": "cancel", "wake_id": "w9"})
    cancelled = await _run(session, {"op": "cancel", "wake_id": "w1"})

    assert missing.kind == "error"
    assert missing.data["code"] == "wake_not_found"
    assert cancelled.kind == "notice"
    assert [row.id for row in session.scheduler.schedules] == ["w2"]


@pytest.mark.asyncio
async def test_edit_rewords_in_place_and_keeps_the_handle() -> None:
    """An edit is the SAME row: the id, its position in the list and its
    history survive, because a reword that reordered the page under the user
    would be a different feature."""
    due = int(time.time() * 1000) + 10 * MINUTE
    session = _Session([_row("w1", due_ms=due), _row("w2")])

    result = await _run(
        session,
        {"op": "edit", "wake_id": "w1", "request": {"message": "reworded", "every": "1h"}},
    )

    assert result.kind == "notice"
    assert result.data["wake_id"] == "w1"
    rows = session.scheduler.schedules
    assert [row.id for row in rows] == ["w1", "w2"]
    assert rows[0].message == "reworded"
    assert rows[0].every_ms == 3_600_000
    # A message-only edit does not move the row's anchor.
    assert rows[0].next_due_at == due


@pytest.mark.asyncio
async def test_a_payload_this_handler_cannot_read_is_refused_not_raised() -> None:
    """The caller is our own route, so a shape it cannot produce is a protocol
    bug — refused in the same typed envelope as everything else rather than
    raised as a traceback through the transport."""
    session = _Session()

    for payload in (
        {"op": "delete", "wake_id": "w1"},
        {"op": "create", "request": {"message": ""}},
        {"op": "create"},
    ):
        result = await _run(session, payload)
        assert result.kind == "error", payload
        assert result.data["code"] in {"wake_invalid", "wake_refused"}, payload
    assert session.scheduler.updates == []


@pytest.mark.asyncio
async def test_a_session_with_no_scheduler_says_so() -> None:
    class _Bare:
        wake_scheduler = None

    result = await _handle()._wake_slash(_Bare(), json.dumps({"op": "create"}), SlashResult)

    assert result.kind == "error"
    assert result.data["code"] == "wake_unavailable"

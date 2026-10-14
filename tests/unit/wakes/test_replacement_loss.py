"""Replacement loss: a NEW wake that reuses a draining schedule's handle.

THE DEFECT THIS PINS (reproduced natively, independent probe, 2026-10-12).
A runtime that commits to leaving (``ServingSessionHandle.begin_drain``) keeps
serving its in-flight turn to completion — that is deliberate, the drain must
not cut a turn off mid-flight. So a schedule can still be CREATED after the
draining scheduler fired the last occurrence of an existing one.

Two things then happen in the wrong order for the same handle ``w3``:

1. the final fire retires ``w3`` from the scheduler's list and queues a
   one-shot re-arm for a successor (``Session._queue_wake_rearm``), which is
   NOT part of the scheduler's own view;
2. a create in the drained session sees the freed slot, allocates ``w3``, and
   reports success to the caller.

Nothing in the create path can see the queued re-arm, so the authoritative
transcript and the derived index keep the OLD row for ``w3`` — the queued
one-shot silently supersedes the fresh schedule on the next persist — and the
NEW schedule is gone after a reload. The user is told their wake is scheduled
by a receipt that the durable state contradicts.

Every layer here is the REAL one: ``Session``, ``WakeScheduler``, the ``wake``
tool through ``ToolContext``, the transcript append, the derived index and
``pump``. ``ScriptedStream`` stands in only for the external model transport —
the same substitution the reproduction rig made, and the only one that must be
synthetic (a real provider is neither available nor the subject).

The assertions below are the PRODUCT invariant, not a rig exit code: the
handle the tool reported as scheduled must be the row the durable store holds.
On the unrepaired source the last assertion is the one that fails.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.harness.wake import LOAD_GRACE_MS
from local_operator.session import session as session_module
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
from tests.e2e.harness import ScriptedStream, build_session, text_turn

#: The per-session transcript entry the durable schedule list lives in.
WAKE_SCHEDULES_CUSTOM_TYPE = "wake_schedules"


class _Clock:
    """Controlled wall clock: only the value the wake layer reads is moved."""

    def __init__(self) -> None:
        self.ms = int(time.time() * 1000)

    def advance(self, ms: int) -> None:
        self.ms += ms


class _ClockView:
    """Stands in for the ``time`` module inside the patched modules.

    Delegates every attribute to the real module so nothing but wall time is
    affected; ``time.time`` is the one call the wake path makes.
    """

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.ms / 1000

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Patch the clock the SESSION and the TOOL read.

    ``WakeScheduler`` is handed ``lambda: int(time.time() * 1000)`` by the
    session, and the tool resolves ``now_ms`` itself, so both module globals
    have to move together or the two disagree about when a wake is due.
    """
    controlled = _Clock()
    view = _ClockView(controlled)
    monkeypatch.setattr(session_module, "time", view)
    monkeypatch.setattr(builtin, "time", view)
    return controlled


def _text(result: Any) -> str:
    return "\n".join(getattr(block, "text", "") for block in result.content)


async def _wake_tool(session: Any, op: str, **args: Any) -> Any:
    """Drive the REAL ``wake`` tool over the real scheduler."""
    context = ToolContext(cwd=str(session._transcript.directory), wake_scheduler=session._wake)
    result = await builtin.execute_wake(f"red-{op}", {"op": op, **args}, context=context)
    assert not result.is_error, f"the tool refused ({op}): {_text(result)}"
    return result


def _durable_rows(directory: pathlib.Path) -> list[dict[str, Any]]:
    """The schedules as a FRESH reader sees them: re-read from disk.

    A new ``Transcript`` rather than the session's own object, because the
    invariant under test is about what is durable, not what the writer holds.
    """
    entry = Transcript(directory).latest_custom_entry(WAKE_SCHEDULES_CUSTOM_TYPE)
    assert entry is not None, "the session persisted no wake_schedules entry at all"
    return list(entry.payload.get("details", {}).get("schedules", []))


def _durable_message_for(directory: pathlib.Path, wake_id: str) -> str | None:
    for row in _durable_rows(directory):
        if row.get("id") == wake_id:
            return str(row.get("message"))
    return None


async def _settle(session: Any) -> None:
    """Join the session's background tasks so nothing leaks into the next cell."""
    pending = [task for task in session._background_tasks if not task.done()]
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending), 10)
    assert not [task for task in session._background_tasks if not task.done()]


@pytest.mark.asyncio
async def test_a_create_during_a_drain_does_not_reuse_a_held_handle(
    tmp_path: pathlib.Path, clock: _Clock
) -> None:
    """The C3 shape: a final fire's held re-arm must not be overwritten by a
    create that allocated the same handle, and the create's receipt must match
    what the durable store keeps.

    Sequence, all through the product's own paths:

    * create two keepers and one short one-shot (``OLD``), which takes ``w3``;
    * ``begin_drain`` — the runtime commits to leaving but keeps serving;
    * advance past OLD's due instant and ``pump`` — OLD fires finally and its
      replacement one-shot is queued for the successor;
    * ``end_drain`` — the move is abandoned, the runtime keeps serving;
    * create ``NEW`` through the tool.

    ``NEW`` must be told a handle the durable store really holds as ``NEW``.
    """
    directory = tmp_path / "sessions" / "red-replacement-loss"
    stream = ScriptedStream([text_turn("synthetic wake reply") for _ in range(4)])
    session = build_session(directory, stream, cwd=directory)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))

    try:
        for message in ("KEEP1", "KEEP2"):
            await _wake_tool(session, "create", message=message, **{"in": "1h"})
            clock.advance(1)
        await _wake_tool(session, "create", message="OLD", **{"in": "1s"})
        clock.advance(1)

        # Precondition: OLD owns the third handle, and that is durable.
        assert _durable_message_for(directory, "w3") == "OLD", "precondition: OLD took w3"

        assert handle.begin_drain("runtime-retired"), "the drain latch must take"
        assert session._wake_deliver_hook == session._spool_wake_to_inbox, (
            "a draining runtime spools its fires for the successor"
        )

        # OLD's final fire: retire from the live list, queue the successor's
        # one-shot. This is the window the defect lives in.
        clock.advance(2000)
        assert await session._wake.pump() == 1, "OLD's final occurrence fires exactly once"

        assert handle.end_drain(), "the abandoned move releases the latch"
        assert session._wake_deliver_hook != session._spool_wake_to_inbox, (
            "a runtime that keeps serving must deliver again"
        )

        receipt = await _wake_tool(session, "create", message="NEW", **{"in": "75m"})
        details = receipt.details or {}
        new_id = str(details.get("wake_id"))
        assert new_id, f"the receipt must name the handle it scheduled: {_text(receipt)}"

        # THE PRODUCT INVARIANT, and the assertion that fails on the unrepaired
        # source: the handle the caller was just told about must be the row a
        # FRESH reader finds on disk. Today the durable row for that handle is
        # still OLD — the queued one-shot superseded NEW on persist — so the
        # receipt is a success that the durable state contradicts, and NEW is
        # gone the moment the conversation is reloaded.
        durable_here = _durable_message_for(directory, new_id)
        assert durable_here == "NEW", (
            f"the create reported handle {new_id!r} as scheduled, but the authoritative "
            f"transcript holds {durable_here!r} for it: the create reused a handle that a "
            "draining fire had already reserved for the successor, so the new schedule "
            "was silently discarded"
        )

        # And the reload view the user would actually get: NEW must survive it.
        await _settle(session)
        await session.dispose()
        clock.advance(1)

        reload_stream = ScriptedStream([text_turn("synthetic wake reply") for _ in range(4)])
        reloaded = build_session(directory, reload_stream, cwd=directory)
        try:
            reloaded_ids = {row.id: row.message for row in reloaded._wake.schedules}
            assert reloaded_ids.get(new_id) == "NEW", (
                f"handle {new_id!r} did not survive the reload as NEW: {reloaded_ids!r} — "
                "the schedule the user was told about is gone"
            )
        finally:
            await _settle(reloaded)
            await reloaded.dispose()
    finally:
        if not session._disposed:
            await session.dispose()


#: Referenced so the grace constant stays meaningful to this module's readers:
#: a reloaded overdue row fires at ``now + LOAD_GRACE_MS``, which is why the
#: reload assertion above deliberately checks the ADOPTED list rather than
#: waiting for a fire.
_ = LOAD_GRACE_MS

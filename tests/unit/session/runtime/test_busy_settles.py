"""The record's ``busy`` bit must settle when the TURN settles — every turn.

**This file exists because the published bit stuck at ``True`` after any turn
that did not run through the prompt queue.** ``_observe_prompt_drain`` was the
only publisher that fires AFTER the final ``AgentEndEvent``, because that event
is emitted while ``Session._is_streaming`` is still True (cleared in
``_run_turn``'s ``finally``); the per-event publish therefore carries ``True``
and nothing after it fires. A peer wake (``lop send``), a scheduled wake, a
background-job result delivery and a resume catch-up all open their turn with
``_spawn_background(self._prompt_messages(...))`` — none of them touch the
drain — so ``lop sessions`` showed a running marker on a session that had been
idle for five hours (design-runtime-autorefresh.md §1.2).

The fix is a turn-boundary hook, ``Session.on_turn_settled``, fired from
``_run_turn_pipeline``'s ``finally`` after ``_is_streaming`` is already False.
These tests drive the REAL ``Session`` behind the production handle and
server, so the seam under test is the one production runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentEndEvent
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from tests.e2e.harness import ScriptedStream, build_session, text_turn


async def _rig(directory: Path, replies: int = 2) -> tuple[Any, OwnedSessionHandle, RuntimeServer]:
    """A real Session under the production handle and a server that is NOT
    listening: the record's busy bit lives on ``server._busy`` regardless, and
    the projection subscription is what ``_serve`` installs first."""
    directory.mkdir(parents=True, exist_ok=True)
    stream = ScriptedStream([text_turn(f"reply {i}") for i in range(replies)])
    session = build_session(directory, stream)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    handle.subscribe(server._schedule_push)
    return session, handle, server


def _watch_turn_end(session: Any) -> asyncio.Event:
    """An event set when the session's final ``AgentEndEvent`` is delivered.

    Every opener except ``prompt(wait_complete=True)`` returns BEFORE its turn
    runs (the turn is a background task), so a test that checked the record
    straight away would read the idle it started with and prove nothing.
    Waiting on the end event is what makes the assertion about the turn.
    """
    ended = asyncio.Event()

    def on_event(event: Any) -> None:
        if isinstance(event, AgentEndEvent):
            ended.set()

    session.subscribe(on_event)
    return ended


async def _settle(
    ended: asyncio.Event, handle: OwnedSessionHandle, server: RuntimeServer, timeout: float = 5.0
) -> None:
    """Wait for the turn to END, then bounded-poll until the record agrees.

    Never a fixed sleep: the bit is expected to settle within a couple of loop
    iterations of the turn ending, so the poll is on the record itself and the
    deadline only bounds a genuine regression. On the unfixed tree the poll
    runs to the deadline and the caller's assertion names the bit.
    """
    await asyncio.wait_for(ended.wait(), timeout)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.5
    while loop.time() < deadline:
        if not handle.is_busy() and not server._busy:
            return
        await asyncio.sleep(0.01)
    assert not handle.is_busy(), "the handle itself never went idle"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["prompt", "mailbox-wake", "idle-steer", "background-prompt"])
async def test_the_record_busy_bit_settles_for_every_turn_shape(tmp_path: Path, path: str) -> None:
    """The four ways a turn opens; the record must read idle after each.

    Before the hook only the ``prompt`` cell passed (verified on b0db51d9c:
    ``mailbox-wake`` and ``idle-steer`` left ``server._busy`` True with
    ``handle.is_busy()`` False).
    """
    session, handle, server = await _rig(tmp_path / "sess")
    ended = _watch_turn_end(session)
    try:
        if path == "prompt":
            await handle.prompt("hello", wait_complete=True)
        elif path == "mailbox-wake":
            await handle.receive_peer_message("wake up", mode="mailbox", wake=True)
        elif path == "idle-steer":
            await handle.receive_peer_message("steer me", mode="steer")
        else:
            from local_operator.harness.types import Message

            session._spawn_background(session._prompt_messages([Message.user("bg")]))
        await _settle(ended, handle, server)
        assert handle.is_busy() is False
        assert server._busy is False, f"{path}: the record still says busy after the turn"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_on_turn_settled_fires_after_streaming_cleared_and_after_the_end_event(
    tmp_path: Path,
) -> None:
    """Ordering is the contract: the hook is useless one line earlier.

    Asserted INSIDE the hook — ``is_streaming`` must already be False and the
    final ``AgentEndEvent`` must already have reached subscribers — because
    that is the state the handle's ``is_busy()`` reads when the hook
    republishes. The end event is the HELD one on this path, flushed from the
    pipeline's ``finally`` after ``_run_turn`` cleared the flag but while
    ``_turn_lock`` is still held: so at delivery ``is_busy()`` is True (the
    lock), and the hook — last under the lock — is the earliest point after
    which a deferred publish reads idle. That is why the handle defers it
    one loop iteration, and why the third row below must read ``busy=False``.
    """
    session, handle, server = await _rig(tmp_path / "sess", replies=1)
    seen: list[str] = []
    ended = _watch_turn_end(session)

    def on_event(event: Any) -> None:
        if isinstance(event, AgentEndEvent):
            seen.append(f"end:streaming={session.is_streaming}:busy={handle.is_busy()}")

    session.subscribe(on_event)
    inner = session.on_turn_settled

    def settled() -> None:
        seen.append(f"settled:streaming={session.is_streaming}")
        if inner is not None:
            inner()
        asyncio.get_running_loop().call_soon(
            lambda: seen.append(f"deferred:busy={handle.is_busy()}")
        )

    session.on_turn_settled = settled
    try:
        await handle.receive_peer_message("wake up", mode="mailbox", wake=True)
        await _settle(ended, handle, server)
    finally:
        await session.dispose()
    assert seen == [
        "end:streaming=False:busy=True",
        "settled:streaming=False",
        "deferred:busy=False",
    ], seen


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_fail_the_turn(tmp_path: Path) -> None:
    """A publish failure is not a turn failure: the hook is guarded."""
    session, handle, server = await _rig(tmp_path / "sess", replies=1)

    def boom() -> None:
        raise RuntimeError("publisher exploded")

    session.on_turn_settled = boom
    try:
        detail = await handle.prompt("hello", wait_complete=True)
        assert "admitted" in detail
        assert session.history(), "the turn must have run to completion"
    finally:
        await session.dispose()

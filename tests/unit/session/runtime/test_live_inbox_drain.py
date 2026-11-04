"""The live inbox drain delivers a queued message AT MOST ONCE.

``process._watch_live_inbox`` is what makes "queue, never refuse" honest: a
sender that got no acknowledgement from a busy owner appends the message to the
owner's ``inbox.jsonl`` (the desktop route's busy arm, ``send`` to a ``wedged``
peer), and the RUNNING owner reads it on its next loop pass.

THE PROPERTY THAT MAKES IT SAFE: the ack that timed out may have been for a
socket write the owner DID take. The same message then exists twice — once in
the owner's admission path, once in the spool — and only one user turn may come
of it. These cells drive a real ``Session`` behind the production
``ServingSessionHandle`` and count the user rows on DISK.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.inbox import (
    SOURCE_PEER,
    SOURCE_USER,
    InboxLine,
    append_inbox,
    peek_inbox,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn
from tests.unit.session.runtime.test_prompt_admission_race import (
    _deliver_a_job_result_and_hold_it,
    _idle,
    _landed,
)


def _user_rows(directory: Path, text: str) -> int:
    count = 0
    for raw in (directory / "transcript.jsonl").read_text().splitlines():
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if text in raw and (row.get("role") == "user" or row.get("type") == "message"):
            count += 1
    return count


def _owner_row(text: str, command_id: str, *, mode: str = "mailbox") -> InboxLine:
    return InboxLine(
        text=text, sender={}, mode=mode, wake=True, source=SOURCE_USER, command_id=command_id
    )


@pytest.mark.asyncio
async def test_a_spooled_copy_of_an_admitted_prompt_is_not_run_twice(tmp_path: Path) -> None:
    """Socket write taken AND spool row present -> exactly one user turn."""
    stream = ScriptedStream([text_turn(f"reply {i}") for i in range(4)])
    session = build_session(tmp_path / "sess", stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        # The socket copy: the owner admitted it, but the sender's ack timed out.
        assert await asyncio.wait_for(handle.prompt("hello twice", command_id="dup-1"), 10)
        await asyncio.wait_for(_idle(session, handle), 10)
        # Side calls (titling and the like) also reach the stream, so the
        # baseline is measured rather than assumed to be 1.
        calls_after_first_turn = len(stream.requests)
        # The spooled copy the sender wrote after that timeout.
        append_inbox(tmp_path / "sess", _owner_row("hello twice", "dup-1"))

        await asyncio.wait_for(child_mod._drain_inbox_into(handle, live=True), 10)
        await asyncio.wait_for(_idle(session, handle), 10)

        assert _landed(tmp_path / "sess", "dup-1")
        assert peek_inbox(tmp_path / "sess") == [], "the spooled copy must be consumed"
        assert _user_rows(tmp_path / "sess", "hello twice") == 1
        assert len(stream.requests) == calls_after_first_turn, "a turn ran for the duplicate"
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_queued_prompt_waits_its_turn_instead_of_being_steered(tmp_path: Path) -> None:
    """LIVE drain: a queued PROMPT runs as its own turn after the busy one.

    The BOOT drain folds a mid-turn owner row into the running turn as a steer
    (it must not wait: the socket is not listening yet —
    ``test_the_boot_inbox_drain_still_steers_instead_of_waiting``). The live
    drain runs beside a listening socket, so the message the user sent as a
    prompt keeps its meaning: it waits for the turn in flight, then runs.
    """
    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    try:
        append_inbox(tmp_path / "sess", _owner_row("queued while busy", "q-1"))
        drain = asyncio.ensure_future(child_mod._drain_inbox_into(handle, live=True))
        for _ in range(20):
            await asyncio.sleep(0)
        assert session.queued_steering() == [], "a prompt was folded into the busy turn"
        stream.release.set()
        assert await asyncio.wait_for(drain, 10) == 1
        await asyncio.wait_for(_idle(session, handle), 10)
        assert _landed(tmp_path / "sess", "q-1")
    finally:
        stream.release.set()
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_queued_steer_joins_the_turn_in_flight(tmp_path: Path) -> None:
    """A row the user sent as a STEER joins the running turn, once."""
    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    try:
        append_inbox(tmp_path / "sess", _owner_row("steer me", "s-1", mode="steer"))
        assert await asyncio.wait_for(child_mod._drain_inbox_into(handle, live=True), 5) == 1
        assert [getattr(m, "id", None) for m in session.queued_steering()] == ["s-1"]
        # The same steer again (a retried sender): answered as already admitted.
        append_inbox(tmp_path / "sess", _owner_row("steer me", "s-1", mode="steer"))
        await asyncio.wait_for(child_mod._drain_inbox_into(handle, live=True), 5)
        assert [getattr(m, "id", None) for m in session.queued_steering()] == ["s-1"]
    finally:
        stream.release.set()
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_peer_row_with_a_known_message_id_is_not_delivered_twice(tmp_path: Path) -> None:
    """The peer half: the receiver dedupes a queued note by its ``message_id``."""
    stream = ScriptedStream([text_turn(f"reply {i}") for i in range(4)])
    session = build_session(tmp_path / "sess", stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    message_id = "peer-" + "a" * 32
    sender = {"session_id": "synthetic-sender", "pid": 1, "name": "synthetic"}
    try:
        # Seed one turn so the session has durable history (the boot drain
        # defers peer rows on a pristine session).
        await asyncio.wait_for(handle.prompt("seed", command_id="seed-1"), 10)
        await asyncio.wait_for(_idle(session, handle), 10)
        await handle.receive_peer_message(
            "note from a peer", mode="mailbox", wake=False, sender=sender, message_id=message_id
        )
        append_inbox(
            tmp_path / "sess",
            InboxLine(
                text="note from a peer",
                sender=sender,
                mode="mailbox",
                wake=False,
                source=SOURCE_PEER,
                message_id=message_id,
            ),
        )
        await asyncio.wait_for(child_mod._drain_inbox_into(handle, live=True), 10)
        raw = (tmp_path / "sess" / "transcript.jsonl").read_text()
        assert raw.count(message_id) == 1, raw.count(message_id)
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_the_watcher_delivers_a_row_written_while_it_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end through ``_watch_live_inbox`` itself, not a direct drain call."""
    monkeypatch.setattr(child_mod, "LIVE_INBOX_POLL_S", 0.05)
    stream = ScriptedStream([text_turn(f"reply {i}") for i in range(4)])
    session = build_session(tmp_path / "sess", stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    stop = asyncio.Event()
    watcher = asyncio.ensure_future(child_mod._watch_live_inbox(handle, stop))
    try:
        append_inbox(tmp_path / "sess", _owner_row("late hello", "w-1"))
        for _ in range(200):
            if _landed(tmp_path / "sess", "w-1"):
                break
            await asyncio.sleep(0.05)
        assert _landed(tmp_path / "sess", "w-1")
        await asyncio.wait_for(_idle(session, handle), 10)
        assert peek_inbox(tmp_path / "sess") == []
    finally:
        stop.set()
        await asyncio.wait_for(watcher, 5)
        await handle.dispose()

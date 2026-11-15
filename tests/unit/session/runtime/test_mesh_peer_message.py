"""A mesh ``send`` lands in a BUSY owner's mailbox, stamped with the link's device.

PR #3827, finding A1 (architect, from a live trace): a mesh ``send`` used to reach
the owner as a ``prompt`` frame, which a busy owner parks in its in-memory
``_prompt_queue`` and acknowledges only after the running turn ends — so the
sender could not be told "delivered", and a runtime disposed meanwhile lost the
message. The mesh ``send`` now carries a ``peer_message`` frame, the op a local
``lop send`` delivers. These cells drive the REAL ``Session`` behind the
production ``ServingSessionHandle`` and ``RuntimeServer._dispatch`` with a
REMOTE connection authority (what the owner's relay dials with, its ``peer``
block taken from the authenticated link), so the seam under test is the one
production runs:

* with a turn in flight, the message is acknowledged at mailbox admission (the
  turn is still blocked), nothing joins the prompt queue, and the card carries
  the AUTHENTICATED device — a device the frame claims is ignored;
* ``wake=False`` on an idle owner is a quiet drop: no turn starts;
* a LOCAL dial cannot dress a message up as a mesh peer's;
* the owner advertises ``MESH_PEER_MESSAGE_CAPABILITY``, the licence the
  requester checks before choosing this path over the old prompt path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from local_operator.harness.types import CustomMessage, StreamEndEvent, StreamTextDelta
from local_operator.session.runtime.server import RuntimeServer, _ConnectionAuthority
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import (
    MESH_PEER_MESSAGE_CAPABILITY,
    MESH_SENDER_DEVICE_ID_KEY,
    MESH_SENDER_DEVICE_NAME_KEY,
)
from tests.e2e.harness import build_session

#: What the owner's relay writes into the auth frame for a link from ``dev-laptop``
#: (``relay._peer_block``) — the only device identity the owner may stamp.
REMOTE = _ConnectionAuthority(
    "remote", frozenset({"prompt"}), {"device_id": "dev-laptop", "name": "laptop"}
)


class _GatedStream:
    """One model call per turn; the FIRST one blocks until ``release`` is set, so
    a turn is genuinely in flight while the peer message arrives."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls = 0

    def __call__(self, request: Any, signal: Any = None) -> AsyncIterator[Any]:
        self.calls += 1
        first = self.calls == 1

        async def gen() -> AsyncIterator[Any]:
            if first:
                await self.release.wait()
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


async def _rig(directory: Path) -> tuple[Any, ServingSessionHandle, RuntimeServer, _GatedStream]:
    directory.mkdir(parents=True, exist_ok=True)
    stream = _GatedStream()
    session = build_session(directory, stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="tui")
    # An engaged session: the receive side refuses an unengaged composer window.
    server._started = True
    return session, handle, server, stream


def _peer_cards(session: Any) -> list[CustomMessage]:
    """The DURABLE peer cards: a busy owner writes the row to the transcript at
    once and parks only the live-context append to the next safe boundary."""
    return [
        m
        for m in session._transcript.build_llm_history()
        if isinstance(m, CustomMessage) and m.custom_type == "peer_message"
    ]


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_busy_owner_takes_a_mesh_send_into_its_mailbox_with_the_link_device(
    tmp_path: Path,
) -> None:
    session, handle, server, stream = await _rig(tmp_path / "sess")
    try:
        assert await handle.prompt("long job") == "prompt admitted"
        await _wait_for(lambda: stream.calls == 1)  # the turn is in flight, blocked
        assert handle.is_busy()

        frame = {
            "op": "peer_message",
            "text": "status?",
            "mode": "mailbox",
            "wake": False,
            "message_id": "peer-" + "b" * 32,
            "sender": {
                "session_id": "s-sender",
                "conversation_name": "planner",
                # Forged by the frame: must never reach the card.
                MESH_SENDER_DEVICE_ID_KEY: "dev-forged",
                MESH_SENDER_DEVICE_NAME_KEY: "forged",
                "pid": 4242,
            },
        }
        # Answered while the turn is STILL blocked: mailbox admission, not the
        # end of the running turn (the prompt path's ack).
        ack = await asyncio.wait_for(server._dispatch("peer_message", frame, REMOTE), 5)
        assert not stream.release.is_set()
        assert ack is not None
        # Nothing parked behind the turn (the old path's failure mode): the queue
        # holds only the running prompt itself.
        assert [command.text for command in handle._prompt_queue] == ["long job"]

        cards = _peer_cards(session)
        assert len(cards) == 1, cards
        sender = cards[0].details["sender"]
        assert sender[MESH_SENDER_DEVICE_ID_KEY] == "dev-laptop"
        assert sender[MESH_SENDER_DEVICE_NAME_KEY] == "laptop"
        assert sender["session_id"] == "s-sender"
        assert "pid" not in sender and sender["remote_origin"] is True
        wrapped = cards[0].details["text"]
        assert "device='laptop'" in wrapped and "from_session='s-sender'" in wrapped
        assert "send peer='laptop' session='s-sender'" in wrapped
        assert "forged" not in wrapped
    finally:
        stream.release.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_quiet_mesh_send_to_an_idle_owner_starts_no_turn(tmp_path: Path) -> None:
    session, handle, server, stream = await _rig(tmp_path / "sess")
    stream.release.set()
    try:
        frame = {
            "op": "peer_message",
            "text": "fyi",
            "mode": "mailbox",
            "wake": False,
            "sender": {"session_id": "s-sender"},
        }
        await asyncio.wait_for(server._dispatch("peer_message", frame, REMOTE), 5)
        await asyncio.sleep(0.2)
        assert stream.calls == 0, "wake=False must not open a turn"
        assert not handle.is_busy()
        assert len(_peer_cards(session)) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_local_dial_cannot_claim_a_mesh_device(tmp_path: Path) -> None:
    session, _handle, server, stream = await _rig(tmp_path / "sess")
    stream.release.set()
    try:
        frame = {
            "op": "peer_message",
            "text": "hi",
            "mode": "mailbox",
            "wake": False,
            "sender": {"pid": 7, MESH_SENDER_DEVICE_ID_KEY: "dev-forged"},
        }
        await asyncio.wait_for(server._dispatch("peer_message", frame), 5)
        sender = _peer_cards(session)[0].details["sender"]
        assert MESH_SENDER_DEVICE_ID_KEY not in sender
        assert sender["pid"] == 7  # a local sender keeps its local identity
        assert "from_pid=7" in _peer_cards(session)[0].details["text"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_owner_advertises_the_mesh_mailbox(tmp_path: Path) -> None:
    session, _handle, server, stream = await _rig(tmp_path / "sess")
    stream.release.set()
    try:
        assert MESH_PEER_MESSAGE_CAPABILITY in server._record.capabilities
    finally:
        await session.dispose()

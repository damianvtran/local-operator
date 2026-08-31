"""The in-session ``send`` tool against a REAL in-process registrant.

Mirrors ``test_peer_client.py``'s loopback approach: the tool resolves a target
from the registry, dials the target's control socket, and returns the receive
side's own detail string. What lives HERE rather than in the client tests is the
tool's own behaviour — the wake/now -> mode mapping, the self-send guard, the
multi-match disambiguation, and the sender identity — because those decisions
are made by ``execute_send`` before and around the dial, not by the wire.

The registrant publishes its record under the TEST process's pid, which is
exactly the pid the tool's self-send guard refuses. The delivery tests therefore
publish an ALIAS record under a live pid that is not this process's (the parent
pid), pointing at the same socket, so there is a reachable target that passes the
guard. The self-send test targets the registrant's OWN record to trip the guard.
"""

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from local_operator.harness.types import ToolContext
from local_operator.mobile import registry
from local_operator.mobile.registrant import Registrant
from local_operator.tools.builtin import execute_send
from tests.unit.mobile.test_registrant import FakeHandle, _wait_record


class _DetailHandle(FakeHandle):
    """Returns the detail string the REAL receive side would for each mode.

    ``Session.receive_peer_message`` answers differently per delivery mode; the
    tool must surface whichever string comes back verbatim, so the fake answers
    the way the real session does and the assertions pin the pass-through.
    """

    async def receive_peer_message(  # noqa: ANN001, ANN202
        self, text, *, mode="mailbox", wake=False, sender=None
    ) -> str:
        self.calls.append(
            ("receive_peer_message", (text,), {"mode": mode, "wake": wake, "sender": sender})
        )
        if mode == "steer":
            return "delivered mid-turn (steered)"
        if wake:
            return "delivered and woke the session"
        return "delivered to the mailbox (will be read on the next turn)"


async def _start_peer(conversation_name: str = "peer-target"):
    """Start a real registrant and publish an ALIAS record under a live pid that
    is NOT this process's, so the send tool's self-send guard does not fire on
    the only reachable session. Returns ``(registrant, alias_record, handle)``."""
    handle = _DetailHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        own = await _wait_record()  # the registrant's record (pid = this process)
        alias = registry.SessionRecord(
            # The parent pid is alive and is not us: it passes both the
            # registry's liveness check and the tool's self-send guard while the
            # socket it points at is still the registrant's real one.
            pid=os.getppid(),
            kind="tui",
            session_id="alias-session",
            conversation_name=conversation_name,
            cwd="/tmp",
            model_label="test/model",
            control_port=own.control_port,
            control_key=own.control_key,
        )
        registry.publish(alias)
        return registrant, alias, handle
    except BaseException:
        registrant.close()
        raise


def _context() -> ToolContext:
    return ToolContext(cwd="/tmp", session_id="sender-session", session_name="sender session")


async def _last_peer_call(handle: _DetailHandle) -> dict[str, Any]:
    name, args, kwargs = handle.calls[-1]
    assert name == "receive_peer_message"
    return {"text": args[0], **kwargs}


@pytest.mark.asyncio
async def test_default_wake_true_sends_a_waking_mailbox_frame() -> None:
    registrant, alias, handle = await _start_peer()
    try:
        result = await execute_send(
            "t1",
            {"target": "peer-target", "message": "gates are green"},
            None,
            None,
            _context(),
        )
        assert result.is_error is False
        call = await _last_peer_call(handle)
        assert call["text"] == "gates are green"
        assert call["mode"] == "mailbox"
        # wake defaults ON: an idle peer is driven to respond right away.
        assert call["wake"] is True
        assert "delivered and woke the session" in result.text
        assert "peer-target" in result.text
        assert f"pid {alias.pid}" in result.text
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_wake_false_is_the_quiet_mailbox_drop() -> None:
    registrant, _alias, handle = await _start_peer()
    try:
        result = await execute_send(
            "t2",
            {"target": "peer-target", "message": "fold this in later", "wake": False},
            None,
            None,
            _context(),
        )
        assert result.is_error is False
        call = await _last_peer_call(handle)
        assert call["mode"] == "mailbox"
        assert call["wake"] is False
        assert "delivered to the mailbox" in result.text
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_now_true_steers_mid_turn() -> None:
    registrant, _alias, handle = await _start_peer()
    try:
        result = await execute_send(
            "t3",
            {"target": "peer-target", "message": "hold off, schema changed", "now": True},
            None,
            None,
            _context(),
        )
        assert result.is_error is False
        call = await _last_peer_call(handle)
        assert call["mode"] == "steer"
        assert "delivered mid-turn (steered)" in result.text
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_sender_identity_carries_this_sessions_registry_record() -> None:
    """The tool runs INSIDE the sender process, so its identity comes from the
    registry record under ``os.getpid()`` (the registrant's own record here) —
    NOT ``os.getppid()`` as the CLI child does. The peer's inbound indicator
    reads these fields to name the sender."""
    registrant, _alias, handle = await _start_peer()
    try:
        await execute_send(
            "t4",
            {"target": "peer-target", "message": "who are you"},
            None,
            None,
            _context(),
        )
        call = await _last_peer_call(handle)
        sender = cast("dict[str, Any]", call["sender"])
        assert sender["pid"] == os.getpid()
        # Copied from this process's registry record (the registrant's).
        assert sender["conversation_name"] == "fake"
        assert sender["session_id"] == "s1"
        assert sender["model_label"] == "test/model"
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_sender_identity_falls_back_to_the_tool_context(monkeypatch) -> None:
    """When no registry record names this process, the sender still carries the
    pid plus the ToolContext's session identity, so the peer can label the card."""
    registrant, _alias, handle = await _start_peer()
    try:
        # The identity lookup finds nothing under this process's pid, so the
        # tool must backfill from the ToolContext. Patched at the module level
        # the function-local import reads at call time; target resolution above
        # it still runs against the real registry.
        import local_operator.mobile.peer_send as peer_send_mod

        monkeypatch.setattr(peer_send_mod, "peer_sender_identity", lambda pid: {"pid": pid})
        result = await execute_send(
            "t5",
            {"target": "peer-target", "message": "identity check"},
            None,
            None,
            _context(),
        )
        assert result.is_error is False
        call = await _last_peer_call(handle)
        sender = cast("dict[str, Any]", call["sender"])
        assert sender["pid"] == os.getpid()
        assert sender["session_id"] == "sender-session"
        assert sender["conversation_name"] == "sender session"
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_self_send_is_refused_before_any_dial() -> None:
    """Targeting the registrant's OWN record (pid = this process) is a session
    messaging itself; the tool refuses before opening a connection."""
    registrant, _alias, handle = await _start_peer()
    try:
        # The registrant's record is named by its FakeHandle projection ("fake").
        result = await execute_send(
            "t6",
            {"target": "fake", "message": "note to self"},
            None,
            None,
            _context(),
        )
        assert result.is_error is True
        assert "this session" in result.text
        # No delivery was attempted.
        assert not any(name == "receive_peer_message" for name, _a, _k in handle.calls)
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_ambiguous_target_returns_candidates_and_asks_for_a_pid() -> None:
    # Two live records sharing a substring: resolution is ambiguous and the tool
    # must list the candidates rather than guess. No socket is needed — the error
    # precedes any dial.
    for index, pid in enumerate((os.getppid(), 1)):
        registry.publish(
            registry.SessionRecord(
                pid=pid,
                kind="tui",
                session_id=f"multi-{index}",
                conversation_name=f"multi session {index}",
                cwd="/tmp",
                model_label="test/model",
                control_port=9,  # never dialed: disambiguation fires first
                control_key="k",
            )
        )
    result = await execute_send(
        "t7",
        {"target": "multi session", "message": "which one?"},
        None,
        None,
        _context(),
    )
    assert result.is_error is True
    assert "2 sessions match" in result.text
    assert "disambiguate with pid" in result.text
    assert "multi session 0" in result.text
    assert "multi session 1" in result.text


@pytest.mark.asyncio
async def test_no_matching_target_is_a_clean_error() -> None:
    result = await execute_send(
        "t8",
        {"target": "no-such-session-anywhere", "message": "hello?"},
        None,
        None,
        _context(),
    )
    assert result.is_error is True
    assert "no live session matches" in result.text


@pytest.mark.asyncio
async def test_empty_message_is_refused() -> None:
    registrant, _alias, handle = await _start_peer()
    try:
        result = await execute_send(
            "t9",
            {"target": "peer-target", "message": "   "},
            None,
            None,
            _context(),
        )
        assert result.is_error is True
        assert "empty" in result.text
        assert not any(name == "receive_peer_message" for name, _a, _k in handle.calls)
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_addressing_by_pid_and_session_id() -> None:
    registrant, alias, handle = await _start_peer()
    try:
        by_pid = await execute_send(
            "t10",
            {"pid": alias.pid, "message": "by pid"},
            None,
            None,
            _context(),
        )
        assert by_pid.is_error is False
        by_session = await execute_send(
            "t11",
            {"session": alias.session_id, "message": "by session id"},
            None,
            None,
            _context(),
        )
        assert by_session.is_error is False
        assert len([c for c in handle.calls if c[0] == "receive_peer_message"]) == 2
    finally:
        registrant.close()

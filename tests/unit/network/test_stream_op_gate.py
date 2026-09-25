"""The stream carrier's per-frame gate: what it ADMITS and, above all, what it REFUSES.

``net_stream`` opens on ``view``, and every frame down it is then resolved through
``types.INNER_OP_CAPABILITY`` by ``RelayServer._accept_stream_frame``. Slice V
added the payload ops a real viewer sends (``slash_result``, ``history_page`` …)
to that table. A table that only shows what is allowed does not show that the gate
works, so these pin the refusals over the REAL relay path — two relays, a paired
link, a live runtime socket on the owner:

* an op with NO decision refuses (``unknown_op``), and ``operator_challenge`` is one;
* a member holding only ``list``/``view`` is refused ``slash_result``, ``prompt``,
  the stop family and the credential writes, each by the capability it lacks;
* a member holding ``slash`` but NOT ``delete`` is refused the delete-scoped verbs
  (``/archive``, ``/unarchive``, ``/delete``) and nothing is written on the owner;
* and the ops that ARE admitted for a read member are exactly the view rows.
* the mute-and-unmute PAIR is admitted, and the stream outlives both — a refused
  frame on this carrier CLOSES the stream, so a missing row here is a lost session.
"""

from __future__ import annotations

import ast
import pathlib
import uuid
from typing import Any

import pytest

import local_operator
from local_operator.network import types
from tests.unit.network.test_relay_e2e import _pair, devices  # noqa: F401 — fixtures
from tests.unit.network.test_session_plane import (
    SESSION,
    Devices,
    _dial_to,
    _seed,
    _serve,
    _serve_real_sessions,
    _stop_all,
    _stop_real,
    _StreamClient,
    _viewer,
    _warm,
)

#: The client whose sends ARE the stream's traffic from a real viewer: every op the
#: mesh carries down a stream is one this class asked for, so its source is the
#: authoritative list of ops that need a row.
_VIEWER_CLIENT = pathlib.Path(local_operator.__file__).parent / "mobile/attach_client.py"

#: Its senders. Each takes the op name as its FIRST positional argument.
_REQUEST_SENDERS = frozenset({"_request", "_request_frame", "_request_payload"})

#: The one op a viewer sends that MUST have no row: producing this material is
#: signing as the operator, and no capability in the transport's set grants that.
#: Pinned by NAME rather than by omitting it from a list, so a future row here
#: fails the test rather than being quietly blessed by it.
_DELIBERATELY_UNGRANTABLE = frozenset({"operator_challenge"})


def _literal_ops(node: ast.AST) -> list[str]:
    """The op names in one argument, INCLUDING both arms of a conditional.

    ``self._request("event_mute" if muted else "event_unmute")`` is one call and
    two ops, and it is the exact shape that slipped past a hand-written list.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _literal_ops(node.body) + _literal_ops(node.orelse)
    return []


def _ops_a_viewer_sends() -> set[str]:
    """Every op name :data:`_VIEWER_CLIENT` sends, read from its source.

    DERIVED, never listed. The round-1 fixture was the same hand-written list as
    the table it checked, so it could only ever confirm itself: it said "the 21
    runtime ops" over a 20-tuple while the 21st — ``event_unmute``, the sibling
    of a row RIGHT THERE in the list — had no row at all, and a viewer that sent
    it lost its whole stream to the refusal. A guard built from the same mental
    list as the thing it guards cannot see an omission in either. This walks the
    caller's own source the way ``test_reason_surfaces`` enumerates renderers.
    """
    tree = ast.parse(_VIEWER_CLIENT.read_text(encoding="utf-8"))
    ops: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _REQUEST_SENDERS:
            ops.update(_literal_ops(node.args[0]))
    return ops


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    pair: Devices = request.getfixturevalue("devices")
    return pair


def _reply_for(client: _StreamClient, frame: dict[str, Any]) -> dict[str, Any] | None:
    """Send one frame; return the reply carrying its ``req``, or ``None`` on silence.

    SILENCE IS A READING, not a timeout to be swallowed: a stream closed by a
    refusal answers nothing at all (``_forward_stream_frame`` writes its error
    and calls ``_close_stream``), and that is the failure mode this file exists
    to catch — see the mute/unmute test.
    """
    client.send(frame)
    for _ in range(60):
        reply = client.recv(timeout=10.0)
        if reply is None:
            return None
        if reply.get("req") == frame["req"]:
            return reply
    return None


def _refusal_for(client: _StreamClient, frame: dict[str, Any]) -> dict[str, Any] | None:
    """Send one frame; return the relay's error reply to it, or ``None`` if admitted."""
    reply = _reply_for(client, frame)
    return reply if reply is not None and reply.get("op") == "error" else None


def test_every_op_a_viewer_sends_has_a_decision_except_the_signing_one() -> None:
    """THE DERIVED ENUMERATION, so this test cannot miss a row by omission.

    Round 1's version of this test iterated a hand-written tuple that was the
    same list the table was built from, over text claiming 21 entries on a
    20-tuple — and the missing one was ``event_unmute``. A guard that shares its
    mental list with the thing it guards is a second copy, not a check.
    """
    sent = _ops_a_viewer_sends()
    # The derivation found a real vocabulary, not an empty set (a moved file, a
    # renamed sender or a changed call shape must fail HERE and not pass by
    # finding nothing to complain about).
    assert len(sent) > 25, sorted(sent)
    assert {"prompt", "slash_result", "event_mute", "event_unmute"} <= sent, sorted(sent)

    undecided = sorted(sent - set(types.INNER_OP_CAPABILITY))
    assert undecided == sorted(_DELIBERATELY_UNGRANTABLE), (
        "every op a real viewer sends needs a row in INNER_OP_CAPABILITY; a "
        "missing one is refused AND closes the viewer's stream"
    )
    # And the one with no row has none, by name — a row that granted it would
    # be a capability handing out the operator's signature.
    assert not (_DELIBERATELY_UNGRANTABLE & set(types.INNER_OP_CAPABILITY))
    # No row may grant more than the capability vocabulary has.
    assert set(types.INNER_OP_CAPABILITY.values()) <= set(types.CAPABILITIES)
    # NOTHING in the stream table needs the broker capability: none of these ops
    # hands token material out (see the comments on credential/mcp_credentials).
    assert "broker_credential" not in set(types.INNER_OP_CAPABILITY.values())


def test_a_read_member_is_refused_every_write_and_admitted_only_the_reads(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="read")
    _seed(server_a.root, SESSION)
    served = _serve(monkeypatch, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        _dial_to(server_b, record, host_a, port_a)

        denied = {
            "slash_result": "slash",
            "prompt": "prompt",
            "abort": "stop",
            "cancel_subagents": "stop",
            "retire_now": "stop",
            "refresh_if_idle": "stop",
            "credential": "prompt",
            "mcp_credentials": "prompt",
            "fork_snapshot": "prompt",
            "register_secret_redaction": "prompt",
        }
        for req, (op, capability) in enumerate(denied.items(), start=10):
            client = _StreamClient(server_b.root)
            try:
                assert client.open_stream(server_a.identity.device_id, SESSION)["op"] == "ack"
                assert client.recv() is not None  # the owner's welcome
                before = list(served[SESSION].handle.calls)
                refusal = _refusal_for(
                    client, {"op": op, "req": req, "command_id": str(uuid.uuid4())}
                )
                assert refusal is not None, f"a read member's {op!r} was carried"
                assert repr(capability) in str(refusal["message"]), (op, refusal)
                assert served[SESSION].handle.calls == before, f"{op!r} reached the runtime"
            finally:
                client.close()

        # AN OP WITH NO DECISION REFUSES by name, whoever sends it — the signing
        # op is the one that must stay here. One stream each: a refusal CLOSES the
        # stream (``_forward_stream_frame``), which is itself fail-closed.
        for req, op in enumerate(("operator_challenge", "teleport"), start=40):
            client = _StreamClient(server_b.root)
            try:
                assert client.open_stream(server_a.identity.device_id, SESSION)["op"] == "ack"
                assert client.recv() is not None
                refusal = _refusal_for(client, {"op": op, "req": req})
                assert refusal is not None, f"{op!r} was carried with no decision"
                assert "no capability decision" in str(refusal["message"]), refusal
            finally:
                client.close()

        # AND THE READS ARE ADMITTED: a read member's viewer can sync its state.
        client = _StreamClient(server_b.root)
        try:
            assert client.open_stream(server_a.identity.device_id, SESSION)["op"] == "ack"
            assert client.recv() is not None
            assert _refusal_for(client, {"op": "frontend_sync", "req": 60}) is None
        finally:
            client.close()
    finally:
        _stop_all(served)


def test_the_mute_and_its_unmute_are_both_admitted_and_the_stream_outlives_them(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V1: the pair is one decision, and a missing row costs the WHOLE stream.

    ``event_unmute`` had no row, and a row-less frame is not refused politely:
    ``_forward_stream_frame`` (relay.py) writes the error back to the viewer and
    then calls ``_close_stream``. So a parked-then-unparked viewer lost its
    session — no reply to the unmute, and none to anything sent afterwards —
    which is why this test asserts on the frames that come back rather than on
    the refusal text. Reachable from a real viewer: the runtime advertises
    ``EVENT_MUTE_CAPABILITY`` unconditionally and ``AttachedSession.set_event_mute``
    sends this op on every park/unpark of a re-leased source.
    """
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    _seed(server_a.root, SESSION)
    served = _serve(monkeypatch, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        _dial_to(server_b, record, host_a, port_a)

        client = _StreamClient(server_b.root)
        try:
            assert client.open_stream(server_a.identity.device_id, SESSION)["op"] == "ack"
            assert client.recv() is not None  # the owner's welcome

            muted = _reply_for(client, {"op": "event_mute", "req": 1})
            assert muted is not None and muted.get("op") == "ack", muted
            # THE ROW THAT WAS MISSING. An ``error`` here is the defect: it is
            # also what closed the stream, so the two assertions below are the
            # half that a refusal-shaped test would not have noticed.
            unmuted = _reply_for(client, {"op": "event_unmute", "req": 2})
            assert unmuted is not None and unmuted.get("op") == "ack", unmuted

            # AND THE STREAM IS STILL UP: a later frame — the third op of the
            # reviewer's probe, and a ping — is answered, which is the half a
            # refusal-shaped assertion would never have noticed.
            assert _reply_for(client, {"op": "event_mute", "req": 3}) is not None
            assert _reply_for(client, {"op": "ping", "req": 4}) is not None
        finally:
            client.close()
    finally:
        _stop_all(served)


class _Deleted:
    """The shape ``_delete_slash`` reads off ``delete_session``, and never used."""

    found = True
    refusal = ""

    def rehearsal(self) -> str:
        return "this would delete it"


def _settle(client: _StreamClient) -> None:
    """Read until the owner stops pushing, so THIS connection is past its bind.

    The runtime refuses every non-priority op while a connection's canonical sync
    is still pending — "this viewer is still connecting to the session; the
    request was not run — retry once the interface has connected" — and the flag
    clears in the BIND TASK's ``finally`` (``server.py``), i.e. when the OWNER
    finishes its own work. A raw viewer that sends a gate frame the instant the
    welcome lands is racing that task: this file did exactly that and passed by
    luck until it failed under load (measured), which is a flake wearing a security
    test. The real client does not race it either — its dial sends its own
    ``frontend_sync`` and waits for the answer — so waiting is the faithful shape
    rather than a workaround.

    Bounded, and it returns on the first quiet interval: the pushes stop when the
    bind is done, and an idle fake session has nothing else to say.
    """
    for _ in range(20):
        if client.recv(timeout=0.5) is None:
            return


def test_the_delete_gate_refuses_a_relayed_caller_with_no_capability_set_at_all() -> None:
    """R3-3: the CAPABILITY half of the delete gate, which no routed cell covered.

    ``capabilities=None`` means "not said" and must fail closed, exactly as
    ``locality=None`` does. Until this cell existed, the mutation
    ``may_run_delete_scoped_slash = lambda loc, caps: loc == "local" or caps is None
    or "delete" in caps`` passed the whole suite — every routed cell either carried
    a set or was already refused by its locality, so a relayed caller that had
    resolved nothing could archive, and round 3's review found the mutation alive.

    Both spellings of "nothing" are here on purpose: ``None`` is a caller that did
    not forward a set, ``frozenset()`` is one that forwarded an empty one. The
    allowed directions are pinned too, so a predicate that simply refused every
    relayed caller could not satisfy this.
    """
    from local_operator.network.types import delete_scope_refusal

    for command in ("archive", "unarchive", "delete"):
        assert delete_scope_refusal(command, "remote", None) is not None, command
        assert delete_scope_refusal(command, "remote", frozenset()) is not None, command
        assert delete_scope_refusal(command, None, None) is not None, command
        assert delete_scope_refusal(command, "remote", frozenset({"slash", "delete"})) is None
        assert delete_scope_refusal(command, "local", None) is None
    # And a verb outside the set is never this gate's business, whatever the facts.
    assert delete_scope_refusal("rename", "remote", None) is None


def test_a_drive_member_cannot_archive_or_delete_through_the_slash_seam(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V2: ``slash`` does not imply the verbs whose effect is ``delete``.

    ``slash_result`` is authorised on ``slash``; ``/archive``, ``/unarchive`` and
    ``/delete`` reach the owner's own dispatch down that one row and write THE
    OWNER's archive index and session files — the effect
    ``OP_CAPABILITY["net_session_lifecycle"] == "delete"`` reserves, with the
    words "archive or delete a session here". A ``drive`` role holds ``slash`` and
    no ``delete``, so before the gate a drive member archived a session on the
    owner and the owner's index recorded it; the only refusal that ever appeared
    was ``/delete``'s live-lease guard, which a session whose writer has exited
    does not have.

    A REAL session and a REAL serving handle on the owner, deliberately: the
    escalation was in the owner's dispatch, so a fake handle would answer "this
    owner cannot run typed slash results" and the test would pass without
    exercising the gate at all.
    """
    from local_operator.session.archived import archived_ids

    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    _seed(server_a.root, SESSION)
    served = _serve_real_sessions(monkeypatch, server_a.root)

    archive_writes: list[tuple[Any, ...]] = []
    deletes: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "local_operator.session.archived.archive_change",
        lambda *args, **kwargs: archive_writes.append((args, kwargs)) or (False, []),
    )
    monkeypatch.setattr(
        "local_operator.session.cleanup.delete_session",
        lambda *args, **kwargs: deletes.append((args, kwargs)) or _Deleted(),
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        _dial_to(server_b, record, host_a, port_a)

        client = _StreamClient(server_b.root)
        try:
            assert client.open_stream(server_a.identity.device_id, SESSION)["op"] == "ack"
            assert client.recv() is not None  # the owner's welcome
            _settle(client)

            replies: dict[str, dict[str, Any]] = {}
            for req, command in enumerate(("archive", "delete"), start=10):
                reply = _reply_for(
                    client,
                    {"op": "slash_result", "req": req, "command": command, "args": "yes"},
                )
                assert reply is not None, f"the owner never answered /{command}"
                replies[command] = reply

            # THE OTHER CARRIER OF THE SAME EFFECT, on the same stream. A routed
            # slash command travels either as ``slash_result`` (the typed seam) or
            # as ``slash`` (the receipt carrier), and both are authorised on
            # ``slash`` — so a gate that only read the first would leave the
            # escalation one door over. It was there: the ``slash`` op reached
            # THIS owner's dispatcher through ``slash_images``, which sent no
            # connection facts, so the dispatcher saw the default
            # ``locality="local"`` and archived the session (measured over two
            # real relays before the fix). The image is what opens that branch —
            # without one the runtime refuses ``/archive`` as terminal-only.
            imaged = _reply_for(
                client,
                {
                    "op": "slash",
                    "req": 30,
                    "command": "archive",
                    "args": "",
                    "images": [{"media_type": "image/png", "data": "aGk="}],
                },
            )
            assert imaged is not None, "the owner never answered the imaged /archive"
            # The UN-IMAGED form is refused by the runtime for its own reason (a
            # runtime has no terminal, so ``h.slash`` answers "terminal-only
            # here"). Pinned so the two shapes cannot be conflated, and so the
            # image-specific branch above stays the one under test.
            unimaged = _reply_for(
                client, {"op": "slash", "req": 31, "command": "archive", "args": ""}
            )
            assert unimaged is not None, "the owner never answered the un-imaged /archive"

            # THE EFFECT IS ASSERTED FIRST, because it is the whole finding: the
            # owner's own writers must never be entered, by EITHER carrier. A gate
            # mutation has to fail HERE — on what the owner did — rather than on
            # the shape of a reply, and an assertion that this device's own archive
            # file is absent would only say the arrow travelled the other way (the
            # proxy round 1 rejected).
            assert archive_writes == [], archive_writes
            assert deletes == [], deletes
            # And the owner's REAL state is untouched — its index does not list the
            # session, and the session's own files are still on disk.
            assert SESSION not in archived_ids(server_a.root)
            assert (server_a.root / "sessions" / SESSION / "transcript.jsonl").exists()

            for command, reply in replies.items():
                # A REFUSAL THE VIEWER CAN READ, not a dropped stream: the verb is
                # refused where it is chosen, so the seam that carried it stays up.
                assert reply.get("op") == "result", reply
                data = reply["data"]
                assert data.get("style") == "warning", data
                assert "delete" in data.get("text", ""), data
                assert command in data.get("text", ""), data

            # The receipt carrier answers in its own shape — an ``ack`` whose
            # detail is the sentence both hosts give — and that sentence is the
            # SHARED one, so a follower cannot tell which carrier it used except
            # by the sentence it already reads everywhere else.
            assert imaged.get("op") == "ack", imaged
            assert "delete" in str(imaged.get("detail", "")), imaged
            assert "archive" in str(imaged.get("detail", "")), imaged
            assert unimaged.get("op") == "error", unimaged

            # The seam itself is still usable afterwards: a routed command the
            # member IS entitled to still runs.
            allowed = _reply_for(
                client, {"op": "slash_result", "req": 20, "command": "rename", "args": ""}
            )
            assert allowed is not None and allowed.get("op") == "result", allowed
        finally:
            client.close()
    finally:
        _stop_real(served)

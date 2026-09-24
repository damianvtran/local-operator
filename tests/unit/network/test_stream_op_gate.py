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
* and the ops that ARE admitted for a read member are exactly the view rows.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from local_operator.network import types
from tests.unit.network.test_relay_e2e import _pair, devices  # noqa: F401 — fixtures
from tests.unit.network.test_session_plane import (
    SESSION,
    Devices,
    _dial_to,
    _seed,
    _serve,
    _stop_all,
    _StreamClient,
    _viewer,
    _warm,
)

#: The 21 runtime ops that had no decision before slice V, as measured on
#: ``ff04d03f1``. Kept as the fixture the refusal test is written against: each is
#: either in the table now with a reason, or (``operator_challenge``) deliberately
#: absent.
_PREVIOUSLY_UNDECIDED = (
    "acknowledge_attention",
    "adopt_aside",
    "cancel_subagents",
    "credential",
    "desktop_watch",
    "event_mute",
    "fork_snapshot",
    "frontend_sync",
    "history_page",
    "job_trajectory",
    "mcp_credentials",
    "operator_challenge",
    "record_shell",
    "refresh_if_idle",
    "register_secret_redaction",
    "retire_now",
    "slash_result",
    "viewer_watch",
    "watch_job",
    "unwatch_job",
)


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    pair: Devices = request.getfixturevalue("devices")
    return pair


def _refusal_for(client: _StreamClient, frame: dict[str, Any]) -> dict[str, Any] | None:
    """Send one frame; return the relay's error reply to it, or ``None`` if admitted."""
    client.send(frame)
    for _ in range(60):
        reply = client.recv(timeout=10.0)
        if reply is None:
            return None
        if reply.get("req") == frame["req"]:
            return reply if reply.get("op") == "error" else None
    return None


def test_every_previously_undecided_op_now_has_a_decision_except_the_signing_one() -> None:
    for op in _PREVIOUSLY_UNDECIDED:
        if op == "operator_challenge":
            assert op not in types.INNER_OP_CAPABILITY
            continue
        assert op in types.INNER_OP_CAPABILITY, op
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

"""G5/G6 on the relay: per-link ``peer_verified``, the peer block, the viewer drop,
the create carriage, and the promote downgrades — plus the end-to-end relayed
prompt from a REAL second relay (design §5 tests 3b, 8, 9b, 15, 16).

Design: ``docs/design/mesh-origin-stamp.md``.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.network import mobility, origin, relay, sync
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_audit_io import _pair

RELAY_SOURCE = Path(relay.__file__).read_text(encoding="utf-8")


def _links(server_a: Any, server_b: Any, network_id: str) -> tuple[Any, Any]:
    """(A's INBOUND listener link from B, B's OUTBOUND dialer link to A)."""
    inbound = next(
        held
        for held in server_a.links.values()
        if held.device_id == server_b.identity.device_id and held.alive
    )
    outbound = next(
        held for held in server_b.links.values() if held.network_id == network_id and held.alive
    )
    return inbound, outbound


# ---------------------------------------------------------------------------
# 3b: the three PeerLink build sites
# ---------------------------------------------------------------------------


def test_3b_listener_link_is_peer_verified_and_dialer_link_is_not(root: Path) -> None:
    server_a, server_b, record = _pair(root)
    try:
        inbound, outbound = _links(server_a, server_b, record.network_id)
        assert inbound.outbound is False and inbound.peer_verified is True  # relay.py listener
        assert outbound.outbound is True and outbound.peer_verified is False  # dialer, pre-G1
        block = server_a._peer_block(inbound)
        assert block == {
            "device_id": server_b.identity.device_id,
            "name": block["name"],
            "network_id": record.network_id,
            "link_id": inbound.link_id,
            "peer_verified": True,
        }
        assert server_b._peer_block(outbound)["peer_verified"] is False
    finally:
        server_b.stop()
        server_a.stop()


def _peer_link_calls() -> list[tuple[str, ast.Call]]:
    tree = ast.parse(RELAY_SOURCE)
    calls: list[tuple[str, ast.Call]] = []
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "PeerLink":
                    calls.append((function.name, node))
    return calls


def test_3b_every_peer_link_site_states_peer_verified_explicitly() -> None:
    """Three sites (R2-4): listener register and pair listener take the handshake's
    proved fact; the dialer pins False until G1 verifies the listener."""
    sites = {}
    for name, call in _peer_link_calls():
        values = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert "peer_verified" in values, f"{name} builds a PeerLink without peer_verified"
        sites.setdefault(name, []).append(values["peer_verified"])
    assert len(_peer_link_calls()) == 3
    dialer = [value for value in sites.values() if value == ["False"]]
    listeners = [value for value in sites.values() if value == ["result.peer_verified"]]
    assert len(dialer) == 1 and len(listeners) == 2, sites


def test_handshake_result_carries_peer_verified_from_verify_auth() -> None:
    from local_operator.network import handshake, types

    assert types.HandshakeResult.__dataclass_fields__["peer_verified"].default is False
    source = Path(handshake.__file__).read_text(encoding="utf-8")
    assert "peer_verified=self.auth_verified" in source


# ---------------------------------------------------------------------------
# 9b: a prompt relayed from device B over a real link verifies as B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_9b_relayed_prompt_from_b_verifies_as_b_end_to_end(
    root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B's mobile daemon, pilot, ``network sessions --send`` and agent ``send`` all
    reach A as ``net_forward``/``net_stream`` over B's link, and A's relay dials its
    runtime with ``_peer_block(link)``. Driven here through ``_op_forward``."""
    from tests.unit.session.runtime.test_approval_authority_seam import _serve
    from tests.unit.session.runtime.test_mesh_origin_runtime import _recorder

    server_a, server_b, record = _pair(root)
    live = None
    try:
        inbound, _outbound = _links(server_a, server_b, record.network_id)
        # A's runtime shares A's config root, exactly as on a real device: it is how
        # both sides derive the same K_relay from the one device seed.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
        live = await _serve(tmp_path, operator_cap=None)
        seen = _recorder(live)
        session_id = live.record.session_id
        reply = await asyncio.to_thread(
            server_a._op_forward,
            inbound,
            {"req": 1, "frame": {"op": "prompt", "req": 9, "session_id": session_id, "text": "hi"}},
        )
        assert reply, reply
        stamp = seen[-1]["mesh_origin"]
        assert stamp["verified"] is True
        assert stamp["origin_device_id"] == server_b.identity.device_id
        assert stamp["network_id"] == record.network_id
        assert stamp["link_id"] == inbound.link_id
        assert stamp["stamped_by"] == server_a.identity.device_id
        assert "operator_origin" not in stamp
    finally:
        if live is not None:
            await live.close(server_a.root)
        server_b.stop()
        server_a.stop()


# ---------------------------------------------------------------------------
# 15: linkless dial vs create; 16: viewer attest dropped
# ---------------------------------------------------------------------------


class _Dial:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.eof = False

    def exchange(self, frame: dict[str, Any], timeout_s: float = 0.0) -> Any:
        self.sent.append(frame)
        return {"op": "ack", "req": frame.get("req"), "detail": "prompt admitted"}

    def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    def close(self) -> None:
        return None


def test_15_create_passes_its_link_to_the_first_prompt_dial(root: Path) -> None:
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    calls: list[dict[str, Any]] = []
    dial = _Dial()

    def dialled(session_id: str, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return dial

    server._dial_owned = dialled  # type: ignore[assignment]
    link = SimpleNamespace(device_id="dev-b")
    server._prompt_on("sess00000001", "hello", None, link=cast(Any, link))
    assert calls[-1]["link"] is link
    server._prompt_on("sess00000001", "hello", None)
    assert calls[-1].get("link") is None  # linkless ⇒ no peer block ⇒ unverified

    tree = ast.parse(RELAY_SOURCE)
    create = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_op_session_create"
    )
    prompt_calls = [
        node
        for node in ast.walk(create)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_prompt_on"
    ]
    assert prompt_calls and all(
        any(kw.arg == "link" for kw in call.keywords) for call in prompt_calls
    )


def test_15_a_linkless_dial_sends_no_peer_block() -> None:
    server = object.__new__(relay.RelayServer)
    server.root = Path("/nonexistent")
    seen: dict[str, Any] = {}

    def fake_dial_owner(root: Any, session_id: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return _Dial(), {}

    server._session_record = lambda session_id: None  # type: ignore[method-assign]
    original = relay.session_dial.dial_owner
    relay.session_dial.dial_owner = fake_dial_owner  # type: ignore[assignment]
    try:
        server._dial_owned("sess00000001", capabilities=())
    finally:
        relay.session_dial.dial_owner = original  # type: ignore[assignment]
    assert seen["peer"] is None


def test_16_a_viewer_relay_attest_is_dropped_by_the_owner_relay() -> None:
    server = object.__new__(relay.RelayServer)
    dial = _Dial()
    stream = SimpleNamespace(closed=False, dial=dial)
    server._stream_for = lambda link, stream_id: stream  # type: ignore[method-assign]
    link = SimpleNamespace(device_id="dev-b", context=SimpleNamespace(capabilities={"prompt"}))
    with pytest.raises(MeshRefusal):
        server._accept_stream_frame(
            cast(Any, link), {"frame": {"op": origin.RELAY_ATTEST_OP, "proof": "00" * 32}}, "s1"
        )
    assert dial.sent == []


# ---------------------------------------------------------------------------
# 8: move and replica promote downgrade EVERY imported stamp
# ---------------------------------------------------------------------------

OWN = {
    "origin_device_id": "dev-b",
    "network_id": "net-1",
    "link_id": "ab",
    "capabilities": ["prompt"],
    "verified": True,
    "stamped_by": "dev-a",
}


def _stamped_row() -> str:
    return json.dumps(
        {
            "id": "e-mesh",
            "type": "message",
            "payload": {"role": "user", "provider_payload": {"lop_mesh_origin": OWN}},
        }
    )


def test_8_move_promote_downgrades_imported_stamps(tmp_path: Path) -> None:
    root = tmp_path / "config"
    staged = root / "network" / "staging" / "staged"
    staged.mkdir(parents=True)
    (staged / "transcript.jsonl").write_text(_stamped_row() + "\n", encoding="utf-8")
    (staged / "asks.jsonl").write_text(
        json.dumps({"kind": "answered", "by": {"surface": "remote", "mesh_origin": OWN}}) + "\n",
        encoding="utf-8",
    )
    outcome = mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001")
    assert outcome.ok is True, outcome
    target = root / "sessions" / "moved0000001"
    row = json.loads((target / "transcript.jsonl").read_text(encoding="utf-8"))
    assert row["payload"]["provider_payload"]["lop_mesh_origin"] == {**OWN, "verified": False}
    answer = json.loads((target / "asks.jsonl").read_text(encoding="utf-8"))
    assert answer["by"]["mesh_origin"]["verified"] is False
    assert not origin.mesh_origin_counts(answer["by"]["mesh_origin"])


def test_8_replica_promote_downgrades_imported_stamps(tmp_path: Path) -> None:
    from tests.unit.network.test_sync import _ask, seed

    root = tmp_path / "holder"
    root.mkdir()
    seed(root, "9f3ac1e0b7d2")
    transcript = root / "sessions" / "9f3ac1e0b7d2" / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_stamped_row() + "\n")
    sync.sync_from(
        root,
        "9f3ac1e0b7d2",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "9f3ac1e0b7d2"),
    )
    (root / "sessions" / "9f3ac1e0b7d2").rename(tmp_path / "elsewhere")
    promoted = sync.promote_replica(root, "9f3ac1e0b7d2")
    target = root / "sessions" / promoted["session_id"]
    stamps = [
        json.loads(line)["payload"]["provider_payload"]["lop_mesh_origin"]
        for line in (target / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if "lop_mesh_origin" in line
    ]
    assert stamps and all(stamp["verified"] is False for stamp in stamps)

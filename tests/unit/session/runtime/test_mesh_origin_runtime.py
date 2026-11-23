"""G5/G6 on the runtime: the relay proof over a REAL socket, and the stamp's carriage.

Design: ``docs/design/mesh-origin-stamp.md``. The wire cells start a REAL
``RuntimeServer`` over a real ``ServingSessionHandle`` (``_serve`` from the
authority-seam suite) and dial it the way the relay does — with the production
``network.dial.dial_owner`` where the relay's own behaviour is under test, and with
a raw socket where a FORGED relay is. The handle's ``prompt``/``steer`` are
replaced by recorders so each cell reads exactly the keywords the server passed.

The carriage cells then follow the stamp past the server: serving → spool (all
four ``SOURCE_USER`` sites) → the drain's two arms → ``Session.prompt``/``steer``
(merge, never overwrite) and the ask answer path → ``answered_by.mesh_origin``.
Numbers in the test names are the design's §5 ids.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.approval import is_wire_hex
from local_operator.network import dial as session_dial
from local_operator.network import identity, origin
from local_operator.paths import config_dir
from local_operator.session.runtime.inbox import (
    SOURCE_PEER,
    SOURCE_USER,
    InboxLine,
    peek_inbox,
)
from local_operator.session.runtime.server import (
    RuntimeServer,
    _ConnectionAuthority,
    _stamp_peer_sender,
)
from local_operator.session.runtime.types import MESH_SENDER_DEVICE_ID_KEY
from tests.unit.session.runtime.test_approval_authority_seam import _serve

LINK_ID = "cd" * 16

PEER = {
    "device_id": "dev-b",
    "name": "laptop",
    "network_id": "net-1",
    "link_id": LINK_ID,
    "peer_verified": True,
}

FORGED = {
    "origin_device_id": "dev-evil",
    "network_id": "net-evil",
    "link_id": "00",
    "capabilities": ["admin"],
    "verified": True,
    "stamped_by": "dev-evil",
    "operator_origin": True,
}


def _recorder(live: Any) -> list[dict[str, Any]]:
    """Replace the handle's prompt/steer with recorders of the server's keywords."""
    seen: list[dict[str, Any]] = []

    async def prompt(
        text: str,
        images: Any = None,
        command_id: Any = None,
        *,
        owner_typed: bool = True,
        mesh_origin: Any = None,
    ) -> str:
        seen.append({"op": "prompt", "owner_typed": owner_typed, "mesh_origin": mesh_origin})
        return "prompt admitted"

    async def steer(
        text: str,
        images: Any = None,
        command_id: Any = None,
        *,
        owner_typed: bool = True,
        mesh_origin: Any = None,
    ) -> str:
        seen.append({"op": "steer", "owner_typed": owner_typed, "mesh_origin": mesh_origin})
        return "steering queued"

    live.handle.prompt = prompt
    live.handle.steer = steer
    return seen


async def _raw(record: Any, **auth: Any) -> tuple[Any, Any, dict[str, Any]]:
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
    writer.write(json.dumps({"key": record.control_key, **auth}).encode() + b"\n")
    await writer.drain()
    while True:
        welcome = json.loads(await asyncio.wait_for(reader.readline(), timeout=10))
        if welcome.get("op") in ("projection", "welcome"):
            return reader, writer, welcome


async def _request(reader: Any, writer: Any, frame: dict[str, Any]) -> dict[str, Any]:
    writer.write(json.dumps(frame).encode() + b"\n")
    await writer.drain()
    for _ in range(60):
        reply = json.loads(await asyncio.wait_for(reader.readline(), timeout=10))
        if isinstance(reply, dict) and reply.get("req") == frame.get("req"):
            return reply
    raise AssertionError("no reply")


def _remote_auth(nonce: str | None, peer: dict[str, Any] | None = None) -> dict[str, Any]:
    auth: dict[str, Any] = {
        "client": "attach",
        "locality": "remote",
        "capabilities": ["prompt"],
        "peer": dict(PEER if peer is None else peer),
    }
    if nonce is not None:
        auth["relay_nonce"] = nonce
    return auth


def _expected(verified: bool, me: str, peer: dict[str, Any] | None = None) -> dict[str, Any]:
    peer = PEER if peer is None else peer
    return {
        "origin_device_id": peer["device_id"],
        "network_id": peer["network_id"],
        "link_id": peer["link_id"],
        "capabilities": ["prompt"],
        "verified": verified,
        "stamped_by": me,
    }


async def _prompt_stamp(reader: Any, writer: Any, seen: list[dict[str, Any]], req: int = 9) -> Any:
    reply = await _request(reader, writer, {"op": "prompt", "req": req, "text": "hi"})
    assert reply["op"] == "ack", reply
    return seen[-1]["mesh_origin"]


# ---------------------------------------------------------------------------
# The real relay dial (tests 2, 3, 9b, 15)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_verified", [True, False, None])
async def test_2_3_the_relays_dial_attests_and_stamps_verified_only_with_peer_verified(
    tmp_path: Path, peer_verified: Any
) -> None:
    me = identity.mint(config_dir(), name="owner")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    peer = dict(PEER)
    if peer_verified is None:
        peer.pop("peer_verified")  # a pre-field relay: absent reads False
    else:
        peer["peer_verified"] = peer_verified
    try:
        dial, welcome = await asyncio.to_thread(
            session_dial.dial_owner,
            config_dir(),
            live.record.session_id,
            capabilities=["prompt"],
            peer=peer,
            _record=live.record,
        )
        try:
            # The salt is consumed by the relay: it never reaches a viewer.
            assert welcome is not None and "relay_salt" not in welcome
            for op in ("prompt", "steer"):
                reply = await asyncio.to_thread(
                    dial.exchange, {"op": op, "req": op, "text": "hi"}, timeout_s=10
                )
                assert reply is not None and reply["op"] == "ack", reply
        finally:
            dial.close()
        expected = _expected(peer_verified is True, me.device_id, {**PEER, **peer})
        assert [entry["mesh_origin"] for entry in seen] == [expected, expected]
        assert all("operator_origin" not in entry["mesh_origin"] for entry in seen)
        assert all(entry["owner_typed"] is False for entry in seen)
    finally:
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_15_a_linkless_dial_sends_no_nonce_and_stamps_unverified(tmp_path: Path) -> None:
    me = identity.mint(config_dir(), name="owner")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        dial, _ = await asyncio.to_thread(
            session_dial.dial_owner,
            config_dir(),
            live.record.session_id,
            capabilities=["prompt"],
            _record=live.record,
        )
        try:
            reply = await asyncio.to_thread(
                dial.exchange, {"op": "prompt", "req": 1, "text": "hi"}, timeout_s=10
            )
            assert reply is not None and reply["op"] == "ack"
        finally:
            dial.close()
        stamp = seen[-1]["mesh_origin"]
        assert stamp["verified"] is False and stamp["origin_device_id"] == ""
        assert stamp["stamped_by"] == me.device_id
    finally:
        await live.close(config_dir())


# ---------------------------------------------------------------------------
# A forged relay on a raw socket (tests 1, 4, 14, 16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack", ["no_attest", "wrong_label", "wrong_salt", "other_connection", "other_block"]
)
async def test_1_4_record_key_alone_cannot_mint_a_verified_stamp(
    tmp_path: Path, attack: str
) -> None:
    """The record key + a forged ``peer``/``capabilities``/``locality`` is not enough."""
    me = identity.mint(config_dir(), name="owner")
    key, _ = origin.relay_key_for(config_dir()) or (b"", "")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        nonce = origin.relay_nonce()
        reader, writer, welcome = await _raw(live.record, **_remote_auth(nonce))
        salt = str(welcome.get("relay_salt") or "")
        assert is_wire_hex(salt)
        block = origin.canonical_block(PEER, ["prompt"])
        proof: Any = None
        if attack == "wrong_label":
            from local_operator.harness.approval import request_proof

            proof = request_proof(key, client_nonce=nonce, server_salt=salt)
        elif attack == "wrong_salt":
            proof = origin.relay_proof(
                key, relay_nonce=nonce, relay_salt=origin.relay_nonce(), block=block
            )
        elif attack == "other_connection":
            other = origin.relay_nonce()
            r2, w2, welcome2 = await _raw(live.record, **_remote_auth(other))
            proof = origin.relay_proof(
                key, relay_nonce=other, relay_salt=str(welcome2["relay_salt"]), block=block
            )
            w2.close()
        elif attack == "other_block":
            # A real proof, over a block that names another device: the auth frame's
            # ``peer`` is what the runtime recomputes from, so the MAC cannot match.
            proof = origin.relay_proof(
                key,
                relay_nonce=nonce,
                relay_salt=salt,
                block=origin.canonical_block({**PEER, "device_id": "dev-x"}, ["prompt"]),
            )
        if attack != "no_attest":
            reply = await _request(
                reader, writer, {"op": "relay_attest", "req": "attest", "proof": proof}
            )
            assert reply.get("verified") is False, reply
        stamp = await _prompt_stamp(reader, writer, seen)
        assert stamp == _expected(False, me.device_id)
        writer.close()
    finally:
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_4_proof_over_a_flipped_peer_verified_is_refused(tmp_path: Path) -> None:
    """A relay cannot raise ``peer_verified`` after MACing the honest block."""
    identity.mint(config_dir(), name="owner")
    key, _ = origin.relay_key_for(config_dir()) or (b"", "")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        nonce = origin.relay_nonce()
        claimed = {**PEER, "peer_verified": True}
        reader, writer, welcome = await _raw(live.record, **_remote_auth(nonce, claimed))
        honest = origin.canonical_block({**PEER, "peer_verified": False}, ["prompt"])
        proof = origin.relay_proof(
            key, relay_nonce=nonce, relay_salt=welcome["relay_salt"], block=honest
        )
        reply = await _request(reader, writer, {"op": "relay_attest", "req": "a", "proof": proof})
        assert reply.get("verified") is False
        assert (await _prompt_stamp(reader, writer, seen))["verified"] is False
        writer.close()
    finally:
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_16_a_late_attest_is_refused_without_changing_state(tmp_path: Path) -> None:
    identity.mint(config_dir(), name="owner")
    key, _ = origin.relay_key_for(config_dir()) or (b"", "")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        nonce = origin.relay_nonce()
        reader, writer, welcome = await _raw(live.record, **_remote_auth(nonce))
        assert (await _prompt_stamp(reader, writer, seen, req=1))["verified"] is False
        valid = origin.relay_proof(
            key,
            relay_nonce=nonce,
            relay_salt=welcome["relay_salt"],
            block=origin.canonical_block(PEER, ["prompt"]),
        )
        reply = await _request(
            reader, writer, {"op": "relay_attest", "req": "late", "proof": valid}
        )
        assert reply["op"] == "error", reply
        assert (await _prompt_stamp(reader, writer, seen, req=2))["verified"] is False
        writer.close()
    finally:
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_a_second_attest_after_a_valid_one_is_refused(tmp_path: Path) -> None:
    identity.mint(config_dir(), name="owner")
    key, _ = origin.relay_key_for(config_dir()) or (b"", "")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        nonce = origin.relay_nonce()
        reader, writer, welcome = await _raw(live.record, **_remote_auth(nonce))
        valid = origin.relay_proof(
            key,
            relay_nonce=nonce,
            relay_salt=welcome["relay_salt"],
            block=origin.canonical_block(PEER, ["prompt"]),
        )
        first = await _request(reader, writer, {"op": "relay_attest", "req": "a", "proof": valid})
        assert first["op"] == "ack" and first["verified"] is True
        again = await _request(reader, writer, {"op": "relay_attest", "req": "b", "proof": valid})
        assert again["op"] == "error"
        assert (await _prompt_stamp(reader, writer, seen))["verified"] is True
        writer.close()
    finally:
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_14_an_old_relay_without_a_nonce_is_admitted_unverified(tmp_path: Path) -> None:
    me = identity.mint(config_dir(), name="owner")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        reader, writer, welcome = await _raw(live.record, **_remote_auth(None))
        assert "relay_salt" not in welcome
        assert await _prompt_stamp(reader, writer, seen) == _expected(False, me.device_id)
        writer.close()
    finally:
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_14_an_old_runtime_without_a_salt_gets_no_attest_and_the_dial_succeeds(
    tmp_path: Path,
) -> None:
    identity.mint(config_dir(), name="relay")
    frames: list[dict[str, Any]] = []
    done = asyncio.Event()

    async def old_owner(reader: Any, writer: Any) -> None:
        try:
            frames.append(json.loads(await reader.readline()))
            writer.write(b'{"op":"welcome","data":{"session_id":"old-sid"}}\n')
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=1.0)
            if line.strip():
                frames.append(json.loads(line))
        except asyncio.TimeoutError:
            pass
        finally:
            writer.close()
            done.set()

    server = await asyncio.start_server(old_owner, "127.0.0.1", 0)
    record = SimpleNamespace(
        control_port=server.sockets[0].getsockname()[1], control_key="k", session_id="old-sid"
    )
    try:
        dial, welcome = await asyncio.to_thread(
            session_dial.dial_owner, config_dir(), "old-sid", peer=dict(PEER), _record=record
        )
        dial.close()
        await asyncio.wait_for(done.wait(), timeout=5)
        assert origin.relay_nonce and len(frames[0]["relay_nonce"]) == 64
        assert [frame.get("op") for frame in frames[1:]] == []  # no relay_attest sent
        assert welcome is not None and welcome["data"]["session_id"] == "old-sid"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_16_frames_sent_during_the_attest_reach_the_caller_and_its_reply_does_not(
    tmp_path: Path,
) -> None:
    """Regression: the attest's reply wait must not swallow the runtime's other frames.

    Found by ``test_session_plane``'s stream pass-through cell on the first head:
    the runtime sends ``frontend_sync`` right after the welcome, and an
    ``exchange`` that discarded non-matching frames lost it, so the viewer never
    synced. The frames come back in order from ``recv``; the attest ack never does.
    """
    identity.mint(config_dir(), name="relay")
    salt = origin.relay_nonce()
    done = asyncio.Event()

    async def owner(reader: Any, writer: Any) -> None:
        try:
            await reader.readline()
            welcome = {"op": "welcome", "data": {"session_id": "sid"}, "relay_salt": salt}
            writer.write(json.dumps(welcome).encode() + b"\n")
            writer.write(b'{"op":"frontend_sync","n":1}\n{"op":"event","n":2}\n')
            await writer.drain()
            attest = json.loads(await reader.readline())
            assert attest["op"] == "relay_attest"
            reply = {"op": "ack", "req": attest["req"], "verified": True}
            writer.write(json.dumps(reply).encode() + b"\n")
            writer.write(b'{"op":"event","n":3}\n')
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            done.set()

    server = await asyncio.start_server(owner, "127.0.0.1", 0)
    record = SimpleNamespace(
        control_port=server.sockets[0].getsockname()[1], control_key="k", session_id="sid"
    )
    try:
        dial, welcome = await asyncio.to_thread(
            session_dial.dial_owner, config_dir(), "sid", peer=dict(PEER), _record=record
        )
        try:
            assert welcome is not None and "relay_salt" not in welcome
            got = [await asyncio.to_thread(dial.recv, 5.0) for _ in range(3)]
            assert [(frame or {}).get("n") for frame in got] == [1, 2, 3]
            assert all((frame or {}).get("op") != "ack" for frame in got)
        finally:
            dial.close()
        await asyncio.wait_for(done.wait(), timeout=5)
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Never from a frame; local writes none; this device's phone never verifies (5, 9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5_frame_copies_are_ignored_and_local_dials_write_no_stamp(tmp_path: Path) -> None:
    me = identity.mint(config_dir(), name="owner")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        forged = {"lop_mesh_origin": FORGED, "mesh_origin": FORGED}
        reader, writer, _ = await _raw(live.record, client="attach")
        for op in ("prompt", "steer"):
            reply = await _request(reader, writer, {"op": op, "req": op, "text": "x", **forged})
            assert reply["op"] == "ack"
        assert [entry["mesh_origin"] for entry in seen] == [None, None]
        writer.close()
        reader, writer, _ = await _raw(live.record, **_remote_auth(None))
        for op in ("prompt", "steer"):
            await _request(reader, writer, {"op": op, "req": op, "text": "x", **forged})
        assert [entry["mesh_origin"] for entry in seen[2:]] == [_expected(False, me.device_id)] * 2
        writer.close()
    finally:
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_9_this_devices_phone_daemon_never_verifies(tmp_path: Path) -> None:
    identity.mint(config_dir(), name="owner")
    key, _ = origin.relay_key_for(config_dir()) or (b"", "")
    live = await _serve(tmp_path, operator_cap=None)
    seen = _recorder(live)
    try:
        nonce = origin.relay_nonce()
        auth = {**_remote_auth(nonce), "client": "daemon"}
        reader, writer, welcome = await _raw(live.record, **auth)
        # A daemon is never offered the relay proof, so it cannot present one.
        assert "relay_salt" not in welcome
        stamp = await _prompt_stamp(reader, writer, seen)
        assert stamp["verified"] is False
        writer.close()
    finally:
        await live.close(config_dir())


# ---------------------------------------------------------------------------
# Ask answers and peer messages through ``_dispatch`` (tests 5, 9, 12)
# ---------------------------------------------------------------------------


def _authority(*, verified: bool, locality: str = "remote", kind: str = "attach") -> Any:
    return _ConnectionAuthority(
        locality,  # type: ignore[arg-type]
        frozenset({"prompt"}),
        dict(PEER),
        kind=kind,
        mesh_verified=verified,
        peer_verified=verified,
        stamped_by="dev-a",
    )


@pytest.mark.asyncio
async def test_5_ask_answers_carry_the_receivers_stamp_never_the_frames() -> None:
    seen: list[tuple[str, Any]] = []

    async def ask_respond(ask_id: str, answers: Any, by: str = "", mesh_origin: Any = None) -> str:
        seen.append(("respond", mesh_origin))
        return "ok"

    async def ask_revise(ask_id: str, answers: Any, by: str = "", mesh_origin: Any = None) -> str:
        seen.append(("revise", mesh_origin))
        return "ok"

    async def ask_answer(
        request_id: str,
        value: Any,
        *,
        question_index: Any = None,
        by: str = "",
        mesh_origin: Any = None,
    ) -> str:
        seen.append(("answer", mesh_origin))
        return "ok"

    server = object.__new__(RuntimeServer)
    server._handle = SimpleNamespace(  # type: ignore[attr-defined]
        ask_respond=ask_respond, ask_revise=ask_revise, ask_answer=ask_answer
    )
    forged = {"mesh_origin": FORGED, "lop_mesh_origin": FORGED}
    for op in ("ask_respond", "ask_revise"):
        await server._dispatch(
            op,
            {"op": op, "ask_id": "a1", "answers": {"q": ["yes"]}, **forged},
            authority=_authority(verified=True),
        )
    await server._dispatch(
        "ask_answer",
        {"op": "ask_answer", "request_id": "r", "value": "yes", **forged},
        authority=_authority(verified=False),
    )
    await server._dispatch(
        "ask_respond",
        {"op": "ask_respond", "ask_id": "a1", "answers": {}, **forged},
        authority=_authority(verified=True, locality="local"),
    )
    verified = origin.build_stamp(PEER, ["prompt"], verified=True, stamped_by="dev-a")
    unverified = {**verified, "verified": False}
    assert seen == [
        ("respond", verified),
        ("revise", verified),
        ("answer", unverified),
        ("respond", None),
    ]


def test_5_peer_message_sender_drops_both_stamp_keys() -> None:
    raw = {"session_id": "s", "mesh_origin": FORGED, "lop_mesh_origin": FORGED}
    for authority in (_authority(verified=True), _authority(verified=True, locality="local")):
        sender = _stamp_peer_sender(raw, authority)
        assert "mesh_origin" not in sender and "lop_mesh_origin" not in sender
    assert _stamp_peer_sender(raw, _authority(verified=True))[MESH_SENDER_DEVICE_ID_KEY] == "dev-b"


# ---------------------------------------------------------------------------
# The spool: four SOURCE_USER sites, both drain arms (test 6)
# ---------------------------------------------------------------------------

STAMP = origin.build_stamp(PEER, ["prompt"], verified=True, stamped_by="dev-a")


def _spool_handle(tmp_path: Path) -> tuple[Any, Path]:
    from tests.unit.session.runtime.test_serving import make_handle

    directory = tmp_path / "session"
    directory.mkdir(parents=True)
    handle, _session = make_handle(directory=directory)
    return handle, directory


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["prompt_updating", "prompt_draining", "steer_updating", "steer"])
async def test_6_every_owner_spool_site_carries_the_stamp_byte_equal(
    tmp_path: Path, site: str
) -> None:
    handle, directory = _spool_handle(tmp_path)
    if site.endswith("updating"):
        handle._updating = True
    if site == "prompt_draining":
        # The leaving-runtime branch: a retire is under way and the drain latched.
        handle._retiring_cause = "update"
        handle._draining = True
    if site.startswith("prompt"):
        await handle.prompt("hi", command_id="c1", owner_typed=False, mesh_origin=STAMP)
    else:
        await handle.steer("hi", command_id="c1", owner_typed=False, mesh_origin=STAMP)
    [line] = [row for row in peek_inbox(directory) if row.command_id == "c1"]
    assert line.source == SOURCE_USER
    assert json.dumps(line.mesh_origin, sort_keys=True) == json.dumps(STAMP, sort_keys=True)
    assert "mesh_origin" not in line.sender and "lop_mesh_origin" not in line.sender
    assert InboxLine.from_json(line.to_json()).mesh_origin == STAMP


@pytest.mark.asyncio
async def test_6_a_peer_row_never_carries_a_stamp(tmp_path: Path) -> None:
    handle, directory = _spool_handle(tmp_path)
    await handle._spool_for_successor(
        "note", mode="mailbox", wake=False, sender={}, source=SOURCE_PEER, mesh_origin=STAMP
    )
    assert [row.mesh_origin for row in peek_inbox(directory)] == [None]
    forged = InboxLine.from_json(
        {"text": "x", "sender": {}, "source": SOURCE_PEER, "mesh_origin": STAMP}
    )
    assert forged.mesh_origin is None


class _ArmHandle:
    def __init__(self, busy: bool) -> None:
        self.busy = busy
        self.calls: list[tuple[str, Any]] = []

    async def prompt(
        self,
        text: str,
        *,
        command_id: Any = None,
        owner_typed: bool = True,
        mesh_origin: Any = None,
    ) -> str:
        if self.busy:
            from local_operator.session.errors import TurnInFlight

            raise TurnInFlight()
        self.calls.append(("prompt", mesh_origin))
        return "ok"

    async def steer(
        self,
        text: str,
        *,
        command_id: Any = None,
        owner_typed: bool = True,
        mesh_origin: Any = None,
    ) -> str:
        self.calls.append(("steer", mesh_origin))
        return "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("source", [SOURCE_USER, SOURCE_PEER])
async def test_6_both_drain_arms_replay_the_stamp_for_user_rows_only(
    busy: bool, source: str
) -> None:
    from local_operator.session.runtime import process as process_mod

    line = InboxLine(
        text="hi",
        sender={"remote_origin": True},
        source=SOURCE_USER,
        command_id="c9",
        mesh_origin=STAMP,
    )
    if source == SOURCE_PEER:
        # A row the drain should never route here; the arm still refuses to trust it.
        object.__setattr__(line, "source", SOURCE_PEER)
    handle = _ArmHandle(busy)
    await process_mod._run_owner_prompt(handle, line, seen=set())
    expected = STAMP if source == SOURCE_USER else None
    assert handle.calls == [("steer" if busy else "prompt", expected)]


# ---------------------------------------------------------------------------
# Session prompt/steer merge (test 7) and the real boot drain
# ---------------------------------------------------------------------------


def _user_payloads(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        payload = record.get("payload") or {}
        if record.get("type") == "message" and payload.get("role") == "user":
            rows.append(payload.get("provider_payload") or {})
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("injected", [False, True])
async def test_7_session_prompt_and_steer_merge_the_stamp(tmp_path: Path, injected: bool) -> None:
    from local_operator.session.session import REMOTE_PROMPT_KEY, RENDERED_INJECTION_KEY
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    directory = tmp_path / "s"
    session = build_session(directory, ScriptedStream([text_turn("ok") for _ in range(3)]))
    try:
        session.steer("queued", owner_typed=False, harness_injected=injected, mesh_origin=STAMP)
        queued = cast(Any, session.queued_steering()[-1]).provider_payload
        await session.prompt("hi", owner_typed=False, harness_injected=injected, mesh_origin=STAMP)
        payloads = _user_payloads(directory)
        for payload in (cast(dict[str, Any], queued), payloads[0]):
            assert payload[origin.MESH_ORIGIN_PAYLOAD_KEY] == STAMP
            assert payload[REMOTE_PROMPT_KEY] is True  # M-12: no longer overwritten
            assert bool(payload.get(RENDERED_INJECTION_KEY)) is injected
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_6_the_real_boot_drain_lands_the_stamp_on_the_durable_row(tmp_path: Path) -> None:
    from local_operator.session.runtime import process as process_mod
    from local_operator.session.runtime.inbox import append_inbox
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    directory = tmp_path / "s"
    directory.mkdir()
    append_inbox(
        directory,
        InboxLine(
            text="spooled",
            sender={"remote_origin": True},
            source=SOURCE_USER,
            command_id="c-boot",
            wake=True,
            mesh_origin=STAMP,
        ),
    )
    session = build_session(directory, ScriptedStream([text_turn("ok") for _ in range(4)]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    handle._maybe_name_conversation = lambda text: None  # type: ignore[method-assign]
    try:
        await process_mod._drain_inbox_into(handle)
        for _ in range(100):
            if _user_payloads(directory):
                break
            await asyncio.sleep(0.05)
        assert _user_payloads(directory)[0][origin.MESH_ORIGIN_PAYLOAD_KEY] == STAMP
    finally:
        await handle.dispose()


# ---------------------------------------------------------------------------
# The ask store: write and fold (tests 12, 12b)
# ---------------------------------------------------------------------------


def _queue(tmp_path: Path) -> Any:
    from tests.unit.asks.test_ask_wire import _live_session, _questions

    _session, queue = _live_session(tmp_path)
    ask_id = queue.enqueue(_questions(), 600)["details"]["ask_id"]
    return queue, ask_id


def test_12_queue_writes_and_folds_the_answer_stamp(tmp_path: Path) -> None:
    queue, ask_id = _queue(tmp_path)
    assert queue.respond(ask_id, {"q0": ["no"]}, by="remote", mesh_origin=STAMP)["ok"]
    assert queue.find(ask_id)["answered_by"]["mesh_origin"] == STAMP
    assert queue.revise(ask_id, {"q0": ["no"]}, by="remote", mesh_origin=STAMP)["ok"]
    assert queue.find(ask_id)["answered_by"]["mesh_origin"]["verified"] is True


@pytest.mark.parametrize(
    "revision",
    [
        None,  # 12b: an unstamped (local or unproven) write
        {**STAMP, "verified": False},  # 12b
        {**STAMP, "origin_device_id": "dev-c"},  # 12: a different device
        {**STAMP, "stamped_by": "dev-z"},  # 12: minted by another runtime
    ],
)
def test_12_12b_a_revision_that_is_not_the_same_verified_device_voids_it(
    tmp_path: Path, revision: Any
) -> None:
    queue, ask_id = _queue(tmp_path)
    assert queue.respond(ask_id, {"q0": ["no"]}, by="remote", mesh_origin=STAMP)["ok"]
    assert queue.revise(ask_id, {"q0": ["yes"]}, by="remote", mesh_origin=revision)["ok"]
    folded = queue.find(ask_id)["answered_by"]["mesh_origin"]
    assert folded["verified"] is False
    assert folded["origin_device_id"] == "dev-b"  # the first write's attribution is kept


def test_10_a_legacy_answer_without_a_stamp_publishes_none(tmp_path: Path) -> None:
    queue, ask_id = _queue(tmp_path)
    assert queue.respond(ask_id, {"q0": ["no"]}, by="remote")["ok"]
    by = queue.find(ask_id)["answered_by"]
    assert "mesh_origin" not in by
    assert not origin.mesh_origin_counts(by.get("mesh_origin"))


@pytest.mark.asyncio
async def test_serving_and_session_carry_the_answer_stamp_to_the_queue(tmp_path: Path) -> None:
    seen: list[tuple[str, Any]] = []
    handle, _ = _spool_handle(tmp_path)

    def respond_ask(ask_id: str, answers: Any, *, by: str = "", mesh_origin: Any = None) -> Any:
        seen.append(("respond", mesh_origin))
        return {"ok": True}

    def revise_ask(ask_id: str, answers: Any, *, by: str = "", mesh_origin: Any = None) -> Any:
        seen.append(("revise", mesh_origin))
        return {"ok": True}

    async def reconcile_asks() -> None:
        return None

    handle._session.respond_ask = respond_ask
    handle._session.revise_ask = revise_ask
    handle._session.reconcile_asks = reconcile_asks
    await handle.ask_respond("a", {"q": ["y"]}, by="remote", mesh_origin=STAMP)
    await handle.ask_revise("a", {"q": ["y"]}, by="remote", mesh_origin=STAMP)
    assert seen == [("respond", STAMP), ("revise", STAMP)]

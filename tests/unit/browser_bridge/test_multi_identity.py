"""Several extension identities paired at once (design note §9.1, rows U1-U16).

The operator's case: Chrome runs the Web Store build while a locally loaded
unpacked build gets a DIFFERENT, path-derived id — and the daemon used to close
the second identity with 4004. These tests pin the allow-list, the standby role
that the allow-list FORCES on us (without it the two installs evict each other
at ~1 Hz, design §1.4), the per-identity pending-code fix (§1.3), and the
compatibility contract that keeps a released extension and a released daemon
working (§7).

Every test uses its own ``root=tmp_path``; none touches port 4099,
``~/.local-operator`` or the operator's pairing file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from local_operator.browser_bridge import daemon as daemon_module
from local_operator.browser_bridge.daemon import (
    BridgeService,
    _identities,
    add_identity,
    create_app,
    pairing_status,
    reset_pairing,
    revoke_all,
    revoke_identity,
)
from local_operator.browser_bridge.protocol import (
    PROTO_VERSION,
    ErrorCode,
    Request,
)

#: The store build's id in the design note, and a path-derived unpacked one.
STORE_ID = "omibaecbjdhgbbcedbnnnmjpmopfheof"
UNPACKED_ID = "jbadjeaodkoboanppmpjiifpconegdcj"
THIRD_ID = "gdmnboeijgcnngijgdnilkhmdaanmneg"

STORE_TOKEN = "store-token-" + "s" * 20
UNPACKED_TOKEN = "unpacked-token-" + "u" * 20


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _pairing_file(tmp_path: Path) -> Path:
    return tmp_path / "browser" / "pairing.json"


def _legacy_record(tmp_path: Path, extension_id: str, token: str) -> None:
    """Write the pre-multi-identity file exactly as a released daemon would."""
    path = _pairing_file(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "extension_id": extension_id,
                "token_sha256": _digest(token),
                "paired_at": 1_700_000_000.0,
            }
        ),
        encoding="utf-8",
    )


def _schema_two(tmp_path: Path, *, driver: str = STORE_ID) -> None:
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome 0.1.13")
    add_identity(tmp_path, UNPACKED_ID, _digest(UNPACKED_TOKEN), label="Chrome 0.1.10")
    if driver != UNPACKED_ID:
        # `add_identity` names the identity it just added as the driver record;
        # re-write so the top-level trio describes the driver we want to test.
        from local_operator.browser_bridge.daemon import _write_pairing

        _write_pairing(tmp_path, _identities(tmp_path), driver_id=driver)


class _FakePeer:
    """A scripted /extension connection: feeds frames, records what it is sent."""

    def __init__(self, extension_id: str) -> None:
        self.headers = {"origin": f"chrome-extension://{extension_id}"}
        self.extension_id = extension_id
        self.accepted = False
        self.closed: list[int | None] = []
        self.sent: list[dict[str, Any]] = []
        self._frames: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> dict[str, Any]:
        frame = await self._frames.get()
        if frame is None:
            # A real peer close, which is the path `extension()`'s finally
            # handles: WebSocketDisconnect, NOT cancellation.
            raise WebSocketDisconnect(1000)
        return frame

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append(code)

    def push(self, frame: dict[str, Any] | None) -> None:
        self._frames.put_nowait(frame)

    def requests(self) -> list[dict[str, Any]]:
        return [frame for frame in self.sent if "method" in frame]

    def acks(self) -> list[dict[str, Any]]:
        return [frame for frame in self.sent if frame.get("event") == "hello_ack"]

    def roles(self) -> list[str]:
        return [
            str(frame.get("role")) for frame in self.sent if frame.get("event") == "role"
        ]


def _hello_frame(token: str, *, version: str = "0.1.13") -> dict[str, Any]:
    return {
        "event": "hello",
        "proto": PROTO_VERSION,
        "token": token,
        "extension_version": version,
        "browser": "Chrome/153",
    }


def _connect(service: BridgeService, peer: _FakePeer, token: str, **extra: Any):
    """Start a handshake on ``peer`` and return its task."""
    peer.push(_hello_frame(token, **extra))
    return asyncio.create_task(service.extension(peer))  # type: ignore[arg-type]


async def _settles(predicate: Callable[[], bool], seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


async def _shutdown(*tasks: asyncio.Task[Any]) -> None:
    for task in tasks:
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


# --- U1-U3: the file format and the downgrade contract -----------------------


def test_u1_legacy_record_is_a_one_entry_allow_list_and_is_not_rewritten(
    tmp_path: Path,
) -> None:
    """Zero migration in this direction: an already-paired operator keeps working.

    The record must be read AS-IS and never rewritten on a read path: a write
    here would be the `state.py` defect class (a read that fails as a write on a
    full disk), and the pairing file is read on every handshake.
    """
    _legacy_record(tmp_path, STORE_ID, STORE_TOKEN)
    before = _pairing_file(tmp_path).read_bytes()

    status = pairing_status(tmp_path)

    assert status["paired"] is True
    assert status["extension_id"] == STORE_ID
    assert [entry["extension_id"] for entry in status["identities"]] == [STORE_ID]
    assert _pairing_file(tmp_path).read_bytes() == before, "a read rewrote the file"

    service = BridgeService(root=tmp_path)
    assert service._valid_saved_token(STORE_ID, STORE_TOKEN) is True
    assert service._valid_saved_token(UNPACKED_ID, STORE_TOKEN) is False


def test_u2_schema_two_authorises_both_ids_with_one_token_each(tmp_path: Path) -> None:
    _schema_two(tmp_path)
    status = pairing_status(tmp_path)

    assert {entry["extension_id"] for entry in status["identities"]} == {
        STORE_ID,
        UNPACKED_ID,
    }
    service = BridgeService(root=tmp_path)
    assert service._valid_saved_token(STORE_ID, STORE_TOKEN) is True
    assert service._valid_saved_token(UNPACKED_ID, UNPACKED_TOKEN) is True
    # Per-ID tokens, so one install's secret is not another's credential.
    assert service._valid_saved_token(UNPACKED_ID, STORE_TOKEN) is False
    assert service._valid_saved_token(STORE_ID, UNPACKED_TOKEN) is False


def test_u3_the_downgrade_contract_holds_for_a_schema_two_file(tmp_path: Path) -> None:
    """The §3.1 transcript, as a test: an OLDER reader still works.

    The top-level trio is not redundancy — it is what an older daemon (or an
    older `lop`) reads as the whole pairing record, so a rollback degrades to
    single-identity instead of breaking pairing.
    """
    _schema_two(tmp_path, driver=STORE_ID)
    saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))

    assert saved["extension_id"] == STORE_ID
    assert saved["token_sha256"] == _digest(STORE_TOKEN)
    assert saved["schema"] == 2

    # A schema-1 reader's two predicates, expressed against the same file.
    legacy_paired = saved is not None
    legacy_matches_store = saved.get("extension_id") == STORE_ID
    assert legacy_paired is True and legacy_matches_store is True
    service = BridgeService(root=tmp_path)
    assert service._valid_saved_token(STORE_ID, STORE_TOKEN) is True
    # "Correctly ignores what it cannot parse": the unpacked id under the
    # top-level (store) hash is not a credential.
    assert service._valid_saved_token(UNPACKED_ID, STORE_TOKEN) is False

    # `--reset` still removes everything an older CLI expects to remove.
    reset_pairing(tmp_path)
    assert not _pairing_file(tmp_path).exists()
    assert pairing_status(tmp_path)["paired"] is False


# --- U4: the gate ------------------------------------------------------------


def test_u4_an_unlisted_identity_presenting_a_token_is_closed_4004_before_attach(
    tmp_path: Path,
) -> None:
    """A peer CLAIMING a pairing it does not have is refused, before any link.

    The token is what makes the claim: a revoked identity dialling back in with
    its old secret, or an install pointed at a daemon that never authorised it.
    This is the one case the 4004 gate still covers, and it must fire BEFORE
    `attach()` so a refused peer cannot install a link and then be severed — the
    unbounded-close rule #996's audit pinned.
    """
    _schema_two(tmp_path)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        before = dict(service.links)
        with pytest.raises(WebSocketDisconnect) as refusal:
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{THIRD_ID}"}
            ) as socket:
                socket.send_json(_hello_frame(STORE_TOKEN))
                socket.receive_json()
        assert refusal.value.code == 4004
        # Refused BEFORE any link state exists: an unknown identity must not be
        # able to install a link and then be severed, because the install is what
        # the 4004 rule exists to prevent.
        assert service.links == before
        assert service.link.websocket is None


def test_u4c_an_unlisted_identity_with_no_token_may_enter_the_pairing_flow(
    tmp_path: Path,
) -> None:
    """...and a peer asking to PAIR is admitted, or the second install is stuck.

    This is the operator's actual sequence: the store build is already paired,
    then a locally loaded build is installed and dials for the first time. It
    presents no token because it has never been paired. Refusing it here left the
    second install with no code to enter — a dial-refuse-redial loop — which is
    the defect this change exists to remove (reproduced on the real rig; see the
    PR evidence). It is admitted, told `paired: false`, and given its OWN code;
    authority is unchanged, because an unpaired link is refused every RPC.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN))
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
        ) as socket:
            socket.send_json(_hello_frame(""))
            ack = socket.receive_json()
            assert ack["paired"] is False, "an unlisted, unpaired peer must hold no authority"
            # The wheel is free here (the store build is paired in the FILE but
            # has no live socket), so this link drives as soon as it connects.
            # Driving is not authority: `paired` above is what gates RPCs.
            assert ack["role"] == "driver"
            codes = pairing_status(tmp_path)["pending"]
            assert [entry["extension_id"] for entry in codes] == [UNPACKED_ID]
            # And the code it was given is the one that authorises it.
            socket.send_json({"event": "pair", "code": codes[0]["code"]})
            result = socket.receive_json()
            assert result["ok"] is True, result
            saved = _identities(tmp_path)
            assert {entry["extension_id"] for entry in saved} == {STORE_ID, UNPACKED_ID}
            service = BridgeService(root=tmp_path)
            assert service._valid_saved_token(UNPACKED_ID, result["token"]) is True
            assert service._valid_saved_token(STORE_ID, result["token"]) is False


def test_u4b_a_web_store_build_still_connects_while_another_identity_is_paired(
    tmp_path: Path,
) -> None:
    """The pin must not have become "the first identity only"."""
    _legacy_record(tmp_path, UNPACKED_ID, UNPACKED_TOKEN)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
        ) as socket:
            socket.send_json(_hello_frame(UNPACKED_TOKEN))
            assert socket.receive_json()["paired"] is True


# --- U5-U6: revocation is per identity ---------------------------------------


@pytest.mark.asyncio
async def test_u5_revoking_one_identity_severs_only_that_link(tmp_path: Path) -> None:
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)  # idle + 2
    assert store.acks()[0]["role"] == "driver"
    assert unpacked.acks()[0]["role"] == "standby"

    await service._sever_identity(UNPACKED_ID)

    assert unpacked.closed == [4003]
    assert [link.extension_id for link in service.links.values()] == ["", STORE_ID]
    assert service.link.extension_id == STORE_ID, "the store build stopped driving"
    assert service._valid_saved_token(STORE_ID, STORE_TOKEN) is True
    assert service._valid_saved_token(UNPACKED_ID, UNPACKED_TOKEN) is False
    saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
    assert saved["extension_id"] == STORE_ID
    assert [entry["extension_id"] for entry in saved["identities"]] == [STORE_ID]
    await _shutdown(*tasks)


@pytest.mark.asyncio
async def test_u6_revoke_all_removes_every_identity(tmp_path: Path) -> None:
    _schema_two(tmp_path)
    add_identity(tmp_path, THIRD_ID, _digest("third"))
    revoke_all(tmp_path)

    assert not _pairing_file(tmp_path).exists()
    assert pairing_status(tmp_path)["paired"] is False
    assert _identities(tmp_path) == []


# --- U7-U9: standby, failover, within-identity later-wins --------------------


@pytest.mark.asyncio
async def test_u7_a_standby_receives_no_commands_and_the_driver_serves_them(
    tmp_path: Path,
) -> None:
    """The whole point of the allow-list: paired to both, exactly one driving."""
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: bool(store.requests()) or len(service.links) == 3)
    assert store.acks()[0]["role"] == "driver"
    assert unpacked.acks()[0]["role"] == "standby"

    dispatch = asyncio.create_task(
        service._dispatch_serialized(
            Request(id="r-1", method="read", params={"tab": "bridge:1:n"})
        )
    )
    assert await _settles(lambda: bool(store.requests())), "the driver was not asked"
    assert store.requests() == [{"id": "r-1", "method": "read", "params": {"tab": "bridge:1:n"}}]
    assert unpacked.requests() == [], "a standby was sent a command"

    store.push({"id": "r-1", "ok": True, "result": {"url": "https://example.com"}})
    response = await asyncio.wait_for(dispatch, timeout=5.0)
    assert response.status_code == 200
    await _shutdown(*tasks)


@pytest.mark.asyncio
async def test_u8_driver_disconnect_promotes_the_standby(tmp_path: Path) -> None:
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    store_task = _connect(service, store, STORE_TOKEN)
    unpacked_task = _connect(service, unpacked, UNPACKED_TOKEN)
    assert await _settles(lambda: unpacked.acks() and unpacked.acks()[0]["role"] == "standby")

    # The driver's socket ends the way a real one does: the peer closes.
    store.push(None)

    assert await _settles(lambda: service.link.extension_id == UNPACKED_ID)
    assert await _settles(lambda: unpacked.roles() == ["driver"]), unpacked.sent
    assert service.standby_links() == []
    state = service.state
    assert state.extension_id == UNPACKED_ID
    # The promoted link keeps serving: the wheel moved, the pairing did not.
    assert service._valid_saved_token(UNPACKED_ID, UNPACKED_TOKEN) is True
    await _shutdown(store_task, unpacked_task)


@pytest.mark.asyncio
async def test_u9_the_same_identity_reconnects_as_driver_not_standby(tmp_path: Path) -> None:
    """later-wins WITHIN an identity: reconnect-after-worker-death depends on it."""
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: unpacked.acks() and unpacked.acks()[0]["role"] == "standby")

    replacement = _FakePeer(STORE_ID)
    tasks.append(_connect(service, replacement, STORE_TOKEN))

    assert await _settles(lambda: replacement.acks() and store.closed == [4000])
    assert replacement.acks()[0]["role"] == "driver", "a same-identity dial was demoted"
    assert service.link.extension_id == STORE_ID
    assert unpacked.roles() == [], "the standby was disturbed by a same-identity reconnect"
    assert unpacked.acks()[0]["role"] == "standby"
    await _shutdown(*tasks)


# --- U10: the pending code is per identity -----------------------------------


def test_u10_two_waiting_identities_each_keep_their_own_live_code(tmp_path: Path) -> None:
    """The §1.3 transcript, asserted as no-longer-reproducible.

    Before this, the second identity's dial rotated the FIRST identity's code
    away, and the first then failed with "That code didn't match" for a code the
    user had read correctly — unreachable only because 4004 refused the second
    identity first. Removing that gate without this fix would have made it live.
    """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
        ) as store:
            store.send_json(_hello_frame(""))
            store.receive_json()
            store_code = pairing_status(tmp_path)["pending_code"]
            assert store_code

            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
            ) as unpacked:
                unpacked.send_json(_hello_frame(""))
                unpacked.receive_json()
                pending = pairing_status(tmp_path)
                codes = {entry["extension_id"]: entry["code"] for entry in pending["pending"]}

                assert codes[STORE_ID] == store_code, "the second dial rotated the first code"
                unpacked_code = codes[UNPACKED_ID]
                assert unpacked_code and unpacked_code != store_code
                assert store_code != unpacked_code
                # Each waiting identity is NAMED, so two popups can be told
                # apart from the terminal.
                labels = {entry["extension_id"]: entry["label"] for entry in pending["pending"]}
                assert labels[STORE_ID].startswith("Chrome")
                assert labels[UNPACKED_ID].startswith("Chrome")

                # The first identity's code still pairs ITS identity.
                store.send_json({"event": "pair", "code": store_code})
                result = store.receive_json()
                assert result["ok"] is True and result["token"]
                assert pairing_status(tmp_path)["paired"] is True
                # ... and the second identity's code is still waiting, untouched.
                remaining = {entry["extension_id"] for entry in pairing_status(tmp_path)["pending"]}
                assert remaining == {UNPACKED_ID}


# --- U11-U12: pair/unpair on a standby link ----------------------------------


@pytest.mark.asyncio
async def test_u11_pair_is_served_on_a_standby_link(tmp_path: Path) -> None:
    """Otherwise pairing the second install deadlocks on itself (§3.5.1).

    The popup submits its code over its OWN socket, and with an allow-list that
    socket is a standby whenever another identity is already driving.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN))
    add_identity(tmp_path, UNPACKED_ID, _digest(UNPACKED_TOKEN))
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    # Authorised but unpaired: its stored token is gone (wiped profile), which
    # is exactly the state a second install is in when it needs the code.
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, "")]
    assert await _settles(
        lambda: unpacked.acks()
        and unpacked.acks()[0]["role"] == "standby"
        and unpacked.acks()[0]["paired"] is False
    )

    code = pairing_status(tmp_path)["pending_code"]
    unpacked.push({"event": "pair", "code": code})

    assert await _settles(
        lambda: any(frame.get("event") == "pair_result" for frame in unpacked.sent)
    )
    result = [f for f in unpacked.sent if f.get("event") == "pair_result"][0]
    assert result["ok"] is True, result
    assert service.link.extension_id == STORE_ID, "pairing a standby moved the wheel"
    assert unpacked.acks()[0]["role"] == "standby"
    assert service._valid_saved_token(UNPACKED_ID, result["token"]) is True
    await _shutdown(*tasks)


@pytest.mark.asyncio
async def test_u12_unpair_from_one_install_removes_only_that_install(tmp_path: Path) -> None:
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)

    unpacked.push({"event": "unpair"})

    assert await _settles(lambda: unpacked.closed == [4003])
    assert [entry["extension_id"] for entry in _identities(tmp_path)] == [STORE_ID]
    assert service.link.extension_id == STORE_ID
    assert store.closed == []
    await _shutdown(*tasks)


# --- U13-U15: the protocol compatibility pins ---------------------------------


def test_u13_an_extra_field_on_hello_is_still_rejected(tmp_path: Path) -> None:
    """Pins rule 2 of §7.1 so nobody relaxes `extra="forbid"` on `Hello`.

    If `Hello` accepted unknown fields, a NEW extension could add one and every
    already-released daemon would stop refusing it — but the real risk is the
    reverse: relaxing it here is what lets a future field be added to `Hello`
    without anybody noticing the released daemons that will close it 4001.
    """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as refusal:
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
            ) as socket:
                socket.send_json({**_hello_frame(""), "role": "standby"})
                socket.receive_json()
        assert refusal.value.code == 4001


def test_u14_an_unknown_event_from_the_extension_is_ignored_not_fatal(
    tmp_path: Path,
) -> None:
    """Pins rule 3 of §7.1: a released daemon must ignore a new frame.

    The receive loop dispatches on the event NAME and falls through to
    `Response.model_validate`, which raises and `continue`s.
    """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
        ) as socket:
            socket.send_json(_hello_frame(""))
            assert socket.receive_json()["paired"] is False
            code = pairing_status(tmp_path)["pending_code"]
            # A frame this daemon has never heard of, then a frame it must act
            # on: the pair_result is the proof that the loop IGNORED the unknown
            # event and went on serving, rather than falling out of the receive
            # loop or dying on a validation error.
            socket.send_json({"event": "something_a_future_build_sends", "value": 1})
            socket.send_json({"event": "pair", "code": code})
            for _ in range(10):
                frame = socket.receive_json()
                if frame.get("event") == "pair_result":
                    assert frame["ok"] is True, frame
                    break
            else:
                raise AssertionError("the link stopped answering an unknown event")


def test_u15_proto_version_is_unchanged_and_a_store_hello_is_accepted(
    tmp_path: Path,
) -> None:
    """No bump: a bump would close the released store build with 4001.

    `worker.ts` renders 4001 as the popup's unfixable "update needed" card, so a
    PROTO_VERSION bump would break the operator's installed build on the next
    daemon restart and there would be nothing they could do about it.
    """
    assert PROTO_VERSION == 1
    _legacy_record(tmp_path, STORE_ID, STORE_TOKEN)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
        ) as socket:
            socket.send_json(_hello_frame(STORE_TOKEN, version="0.1.10"))
            ack = socket.receive_json()
            assert ack["proto"] == 1
            assert ack["paired"] is True
            # Additive fields a released extension ignores, and the role an
            # older DAEMON never sent (absent must read as driver).
            assert ack["role"] == "driver"
            assert ack["authorized_count"] == 1


# --- U16: the #996 fences still hold with several links ----------------------


@pytest.mark.asyncio
async def test_u16_a_superseded_socket_stamps_nothing_on_the_live_link(
    tmp_path: Path,
) -> None:
    """#996's fence, in its multi-link form: a superseded socket stamps NOTHING.

    A superseded socket must not stamp liveness, resolve a future, or publish a
    driven record into the connection that replaced it.

    The mechanism changed with this work and the reviewer should know how: the
    link object is now per SOCKET, so a superseded socket's frames are written to
    its own retired link by construction. That makes the hazard structurally
    unreachable rather than guarded, which is *stronger* than #996's
    `is_authoritative` check and is why the proof-of-failure row for this test
    mutates the receive loop's TARGET (`link` -> `self.link`) rather than
    `is_authoritative`. The generation/socket test still runs — it is what makes a
    retired link's own loop stand down — but it is now belt-and-braces for this
    particular invariant, not the sole mechanism, and a mutation of it alone is
    deliberately NOT shown as a row because it would not go red.
    """
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    store_task = _connect(service, store, STORE_TOKEN)
    assert await _settles(lambda: service.link.extension_id == STORE_ID)

    replacement = _FakePeer(STORE_ID)
    replacement_task = _connect(service, replacement, STORE_TOKEN)
    assert await _settles(lambda: service.link.websocket is replacement)
    live = service.link
    assert live is not None
    liveness = live.last_frame_at
    driven = dict(live.driven)

    pending: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    live.pending["r-stale"] = pending

    # The superseded socket now says all the things a live one may say.
    store.push({"id": "r-stale", "ok": True, "result": {"url": "https://stale.example"}})
    store.push({"event": "tab_update", "tab": "bridge:9:x", "url": "https://stale.example"})
    await asyncio.sleep(0.05)

    assert live.pending.get("r-stale") is pending and not pending.done()
    assert live.driven == driven, "a superseded socket published a driven record"
    assert live.last_frame_at <= liveness or live.last_frame_at == liveness
    await _shutdown(store_task, replacement_task)


@pytest.mark.asyncio
async def test_u16b_the_drop_latch_survives_the_retire_it_describes(
    tmp_path: Path,
) -> None:
    """A sever for silence must still read as `extension_unresponsive`.

    The link that owns the latch is retired in the same breath as it is latched,
    so a reader resolving `self.link` afterwards would see a fresh idle link and
    report "not attached" — the #996 wedge copy going false exactly while the
    daemon is acting on it.
    """
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    peer = _FakePeer(STORE_ID)
    task = _connect(service, peer, STORE_TOKEN)
    assert await _settles(lambda: service.link.extension_id == STORE_ID)

    service.link.last_frame_at = time.monotonic() - (daemon_module.LINK_SILENCE_TIMEOUT_S + 1)
    dropped = await service._drop_unproven_link("test")

    assert dropped is True
    assert service.link.websocket is None
    assert service.drop_latched() is True
    assert service.state.extension_unresponsive is True
    await _shutdown(task)


def test_u16c_an_authorized_prefix_of_the_file_is_not_a_credential(tmp_path: Path) -> None:
    """Unlisted ids are refused, however much of a listed id they share."""
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN))
    service = BridgeService(root=tmp_path)
    assert service._identity_listed(STORE_ID) is True
    assert service._identity_listed(STORE_ID[:16]) is False
    assert service._identity_listed(THIRD_ID) is False
    assert service._identity_listed("") is False


def test_pending_codes_are_never_written_to_the_pairing_file(tmp_path: Path) -> None:
    """The two files stay separate: codes are ephemeral, authority is not."""
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN))
    saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
    assert "code" not in json.dumps(saved)
    assert _pairing_file(tmp_path).stat().st_mode & 0o777 == 0o600


def test_revoking_an_unknown_identity_leaves_the_others_alone(tmp_path: Path) -> None:
    _schema_two(tmp_path)
    revoke_identity(tmp_path, THIRD_ID)
    assert {entry["extension_id"] for entry in _identities(tmp_path)} == {STORE_ID, UNPACKED_ID}
    assert ErrorCode.NOT_PAIRED.value == "not_paired"

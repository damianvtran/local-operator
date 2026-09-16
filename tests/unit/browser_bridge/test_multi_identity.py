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
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.sync.client import connect as sync_connect

from local_operator.browser_bridge import daemon as daemon_module
from local_operator.browser_bridge.daemon import (
    MAX_UNLISTED_LINKS,
    BridgeService,
    _identities,
    _pending_entries,
    _pending_path,
    _write_pending,
    add_identity,
    create_app,
    pairing_status,
    reset_pairing,
    revoke_all,
    revoke_identity,
)
from local_operator.browser_bridge.protocol import PROTO_VERSION, ErrorCode, Request

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
    """Two authorised identities, with the legacy trio naming ``driver``.

    Through the REAL writers, and with no repair step: keeping the trio on the
    driver is a property of `add_identity`/`_write_pairing` themselves (review
    round 1, M2/Q2), and a fixture that rewrote the file afterwards would hide
    exactly the defect these rows exist to catch.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome 0.1.13", driver_id=driver)
    add_identity(
        tmp_path, UNPACKED_ID, _digest(UNPACKED_TOKEN), label="Chrome 0.1.10", driver_id=driver
    )


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
        return [str(frame.get("role")) for frame in self.sent if frame.get("event") == "role"]


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


async def _settles(predicate: Callable[[], object], seconds: float = 2.0) -> bool:
    """Wait until ``predicate`` is truthy. Typed loosely on purpose: callers pass
    expressions like ``unpacked.acks() and ...``, and only truthiness is read."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return bool(predicate())


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


def test_u4_an_unlisted_identity_presenting_a_token_is_admitted_unpaired(
    tmp_path: Path,
) -> None:
    """A token that cannot match is not an authorisation, and refusing it was a dead end.

    This row asserted a pre-attach 4004 until round 3. That gate was a DEAD END
    rather than a defence (design D1 / UX U3): a revoked install keeps the token
    it was issued, so it re-dialled with it, was closed before ``attach()``, and
    therefore never reached ``_ensure_pending`` — no code was ever minted, ``lop
    browser pair`` answered "already paired … use --reset", and the pairing form
    its own popup showed could not be completed by any code on earth.

    What the gate is replaced with is the rule the rest of the daemon already
    follows: authority comes from the FILE's entry for THIS id, and an id with no
    entry has nothing that can match. So the dial is admitted as an ASKER —
    ``paired: false``, offered a code, refused every RPC — and the file is
    untouched until that code is entered. Presenting a token that belongs to
    ANOTHER identity must not change that: the token is not a bearer credential.
    """
    _schema_two(tmp_path)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{THIRD_ID}"}
        ) as socket:
            socket.send_json(_hello_frame(STORE_TOKEN))
            ack = socket.receive_json()
            # STORE_TOKEN is a REAL token in this file — for the store build. It
            # buys the third identity nothing.
            assert ack["paired"] is False, ack
            assert ack["authorized_count"] == 2
            assert service._valid_saved_token(THIRD_ID, STORE_TOKEN) is False
            assert {entry["extension_id"] for entry in _identities(tmp_path)} == {
                STORE_ID,
                UNPACKED_ID,
            }
            # ...and a code exists for it, which is the whole point: this is the
            # only route back for a revoked install.
            codes = pairing_status(tmp_path)["pending"]
            assert [entry["extension_id"] for entry in codes] == [THIRD_ID]
            # An unpaired link cannot serve: asking for the wheel for it is
            # refused exactly as it would be for a stranger.
            key = service.state.session_key
            refused = client.post(
                "/driver", headers={"X-Bridge-Key": key}, json={"target": THIRD_ID}
            )
            assert refused.status_code == 409, refused.json()
            assert refused.json()["error"] == "not_paired"
            # And entering ITS code is what authorises it — the file grows by one
            # identity, with its own token, and nothing else changes.
            socket.send_json({"event": "pair", "code": codes[0]["code"]})
            result = socket.receive_json()
            assert result["ok"] is True, result
            assert service._valid_saved_token(THIRD_ID, result["token"]) is True
            assert service._valid_saved_token(STORE_ID, result["token"]) is False
            assert service._valid_saved_token(STORE_ID, STORE_TOKEN) is True


def test_u4d_the_never_pairable_dead_end_needs_no_storage_wipe(
    tmp_path: Path,
) -> None:
    """A REVOKED install re-pairs with the token it still holds, no reset, no wipe.

    The reproduction UX U3 and design D1 both filed, reduced to its essentials:
    two identities are paired, one is revoked (`revoke_identity` — what `pair
    --revoke` calls), and that install dials back with the token it still has. It
    must get a fresh code rather than a close, because ``--reset`` (which revokes
    the WORKING install too) and Settings → unpair were the only ways out.
    """
    _schema_two(tmp_path)
    revoke_identity(tmp_path, UNPACKED_ID)
    assert {entry["extension_id"] for entry in _identities(tmp_path)} == {STORE_ID}
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
        ) as socket:
            socket.send_json(_hello_frame(UNPACKED_TOKEN))  # its now-dead token
            ack = socket.receive_json()
            assert ack["paired"] is False
            codes = pairing_status(tmp_path)["pending"]
            assert [entry["extension_id"] for entry in codes] == [UNPACKED_ID], codes
            socket.send_json({"event": "pair", "code": codes[0]["code"]})
            result = socket.receive_json()
            assert result["ok"] is True, result
            assert {entry["extension_id"] for entry in _identities(tmp_path)} == {
                STORE_ID,
                UNPACKED_ID,
            }


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
        service._dispatch_serialized(Request(id="r-1", method="read", params={"tab": "bridge:1:n"}))
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


@pytest.mark.asyncio
async def test_u8b_an_unpaired_standby_is_not_promoted(tmp_path: Path) -> None:
    """The wheel only goes to a link that can actually serve a command.

    Measured on the real rig before this guard: the driver died, the only
    survivor was a second install that had connected but not yet paired, and it
    was promoted — after which every session read `not_paired` for as long as it
    held the wheel, i.e. the daemon chose a link it knew could serve nothing.
    """
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    stranger = _FakePeer(THIRD_ID)
    store_task = _connect(service, store, STORE_TOKEN)
    # THIRD_ID is not in the allow-list, so this dial presents no token and is
    # admitted to pair: unpaired, and a standby while the store build drives.
    stranger_task = _connect(service, stranger, "")
    assert await _settles(lambda: stranger.acks() and stranger.acks()[0]["role"] == "standby")
    assert stranger.acks()[0]["paired"] is False

    store.push(None)  # the driver's socket ends

    assert await _settles(lambda: service.link.extension_id == "")
    assert stranger.roles() == [], "an unpaired standby was handed the wheel"
    assert service.link.websocket is None

    # And when it PAIRS, it takes the idle wheel at that moment rather than
    # sitting paired while every session is told no browser is attached.
    code = pairing_status(tmp_path)["pending_code"]
    stranger.push({"event": "pair", "code": code})
    assert await _settles(lambda: service.link.extension_id == THIRD_ID), service.links
    assert stranger.roles() == ["driver"]
    await _shutdown(store_task, stranger_task)


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


# --- U17-U19: `lop browser drive` (design §8.2) ------------------------------


def test_u17_drive_moves_the_wheel_and_demotes_the_incumbent(tmp_path: Path) -> None:
    """The escape hatch from the incumbency rule, exercised over its real route.

    Without it, the install already driving keeps the wheel and the other cannot
    take it — so the operator's only lever is quitting a browser. The demotion
    travels as a `role` frame rather than a socket close: the demoted install is
    running this same build, and the frame is what tells it to release its
    surfaces without losing its pairing.
    """
    _schema_two(tmp_path)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
        ) as store:
            store.send_json(_hello_frame(STORE_TOKEN))
            assert store.receive_json()["role"] == "driver"
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
            ) as unpacked:
                unpacked.send_json(_hello_frame(UNPACKED_TOKEN))
                assert unpacked.receive_json()["role"] == "standby"

                pinned = client.post(
                    "/driver", headers={"X-Bridge-Key": key}, json={"target": UNPACKED_ID[:12]}
                )
                assert pinned.status_code == 200, pinned.text
                assert pinned.json()["driver_extension_id"] == UNPACKED_ID

                # The incumbent is told; the replacement is told. Both frames
                # matter: the demoted install must know to hand back its tabs.
                assert unpacked.receive_json() == {"event": "role", "role": "driver"}
                assert store.receive_json() == {"event": "role", "role": "standby"}
                health = client.get("/health").json()
                assert health["driver_extension_id"] == UNPACKED_ID
                assert health["standby_extension_ids"] == [STORE_ID]

                # Idempotent: driving the driver again is a no-op, not a churn.
                again = client.post(
                    "/driver", headers={"X-Bridge-Key": key}, json={"target": UNPACKED_ID}
                )
                assert again.status_code == 200 and again.json()["ok"] is True


def test_u18_drive_refuses_an_unknown_target_and_an_unpaired_one(tmp_path: Path) -> None:
    _schema_two(tmp_path)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key
        unknown = client.post("/driver", headers={"X-Bridge-Key": key}, json={"target": "deadbeef"})
        assert unknown.status_code == 404
        assert unknown.json()["error"] == "unknown_extension"
        # Names what IS paired, so "no connected extension matches" is actionable.
        assert unknown.json()["authorized_extension_ids"] == sorted([STORE_ID, UNPACKED_ID])

        # An unpaired, attached install is refused: handing it the wheel would
        # answer `not_paired` to every session for as long as it drove.
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{THIRD_ID}"}
        ) as stranger:
            stranger.send_json(_hello_frame(""))
            assert stranger.receive_json()["paired"] is False
            refused = client.post(
                "/driver", headers={"X-Bridge-Key": key}, json={"target": THIRD_ID}
            )
            assert refused.status_code == 409
            assert refused.json()["error"] == "not_paired"


def test_u19_drive_requires_the_session_key(tmp_path: Path) -> None:
    """Same authority as /rpc, and the same reason: choosing which install drives
    the user's real browser is not something any local process may ask for."""
    _schema_two(tmp_path)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        assert client.post("/driver", json={"target": STORE_ID}).status_code == 401
        assert (
            client.post(
                "/driver", headers={"X-Bridge-Key": "nope"}, json={"target": STORE_ID}
            ).status_code
            == 401
        )


# --- U20-U25: remediation round 1 (M1, M2/Q2, m1, m2) ------------------------
#
# Every row below fails on the head the review examined. They are grouped here
# rather than folded into U1-U19 so the next reader can see which defects the
# round found, and because each one pins a claim the fix makes about the FILE or
# about the wheel — not a re-assertion of the rule it already covered.


def test_u20_pairing_a_second_install_leaves_the_drivers_record_alone(tmp_path: Path) -> None:
    """M2/Q2 — the legacy trio names the DRIVER, not the newest arrival.

    The second install pairs through its OWN socket while the first holds the
    wheel (design §3.5.1: `pair` must be served on a standby link, or pairing a
    second install deadlocks). What must not happen is the downgrade record
    moving to it: an older daemon reading that file would authorise the standby
    and refuse the install the operator is using — a forced re-pair, on the one
    path decision 1 promises never needs one.
    """
    _schema_two(tmp_path, driver=STORE_ID)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
        ) as store:
            store.send_json(_hello_frame(STORE_TOKEN))
            assert store.receive_json()["role"] == "driver"
            # A third, unlisted install: it has no token, so it is admitted and
            # given a code (that is what makes a second install addable).
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{THIRD_ID}"}
            ) as third:
                third.send_json(_hello_frame(""))
                ack = third.receive_json()
                assert ack["paired"] is False and ack["role"] == "standby"
                code = next(
                    entry["code"]
                    for entry in pairing_status(tmp_path)["pending"]
                    if entry["extension_id"] == THIRD_ID
                )
                third.send_json({"event": "pair", "code": code})
                result = third.receive_json()
                assert result["event"] == "pair_result" and result["ok"] is True

                # The DRIVER's record is untouched, and the new identity's token
                # is a real credential of its own (decision 2).
                saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
                assert (
                    saved["extension_id"] == STORE_ID
                ), "pairing a standby moved the downgrade record off the driver"
                assert saved["token_sha256"] == _digest(STORE_TOKEN)
                status = pairing_status(tmp_path)
                driving = {
                    entry["extension_id"]: entry["driving"] for entry in status["identities"]
                }
                assert driving == {STORE_ID: True, UNPACKED_ID: False, THIRD_ID: False}
                # ... and /health agrees with the file (QA round 1, Q3).
                health = client.get("/health").json()
                assert health["driver_extension_id"] == STORE_ID

                # Revoking the standby must not rename the record either.
                assert revoke_identity(tmp_path, THIRD_ID) is None
                saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
                assert saved["extension_id"] == STORE_ID


@pytest.mark.asyncio
async def test_u21_promotion_moves_the_record_onto_the_promoted_install(tmp_path: Path) -> None:
    """M2/Q2, the other direction: when the wheel REALLY moves, so does the file.

    Preserving the record while the driver is unchanged is only half the
    contract. If the driver's link ends and a standby is promoted, the trio must
    name the promoted install — otherwise the rollback path authorises an
    install that is not driving, which is how QA Q2 measured the old daemon
    refusing the store build with 4004.
    """
    _schema_two(tmp_path, driver=STORE_ID)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)
    before = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
    assert before["extension_id"] == STORE_ID

    # The driver dies WITHOUT a close frame, as a dying MV3 worker does.
    store.push(None)
    assert await _settles(lambda: service.link.extension_id == UNPACKED_ID)
    saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
    assert (
        saved["extension_id"] == UNPACKED_ID
    ), "the promoted install is not the one the downgrade record names"
    assert saved["token_sha256"] == _digest(UNPACKED_TOKEN)
    await _shutdown(*tasks)


def test_u22_a_paired_install_takes_the_wheel_from_an_unpaired_incumbent(
    tmp_path: Path,
) -> None:
    """M1 — the third place the wheel is handed out now applies its own rule.

    `_promote_standby` and `POST /driver` already refuse to give the wheel to a
    link that cannot serve a command. The handshake did not, and the new
    admission of token-less dials makes that reachable without anybody being
    hostile: load a fresh unpacked build against a restarting daemon, and it can
    complete `hello` first. The authorised, paired install that dials next was
    told `standby`, and every session answered `not_paired` while a perfectly
    good browser sat idle.
    """
    _schema_two(tmp_path, driver=STORE_ID)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        # The stranger gets there first and takes the cold-start wheel.
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{THIRD_ID}"}
        ) as stranger:
            stranger.send_json(_hello_frame(""))
            first = stranger.receive_json()
            assert first["role"] == "driver" and first["paired"] is False
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
            ) as store:
                store.send_json(_hello_frame(STORE_TOKEN))
                ack = store.receive_json()
                assert ack["paired"] is True
                assert (
                    ack["role"] == "driver"
                ), "a paired install was made a standby by an unpaired incumbent"
                # The demoted incumbent is told, not merely dropped: it holds
                # debugger attachments it must release.
                assert stranger.receive_json() == {"event": "role", "role": "standby"}
                health = client.get("/health").json()
                assert health["driver_extension_id"] == STORE_ID
                assert health["paired"] is True
                # The user-visible outcome: a session can be served at all.
                served = client.post(
                    "/rpc",
                    headers={"X-Bridge-Key": app.state.bridge.state.session_key},
                    json={"id": "r-1", "method": "status", "params": {}},
                )
                assert served.status_code == 200, served.text


def test_u23_an_unpaired_dial_is_not_reported_as_a_standby(tmp_path: Path) -> None:
    """m2 — "standby" is a role, and an admitted stranger does not hold it.

    The stranger is admitted so it can pair; it receives no commands and
    `_promote_standby` will never promote it. Listing it under
    `standby_extension_ids` told the operator — in the popup and in
    `lop browser status` — that a peer with no pairing at all "is standing by".
    """
    _schema_two(tmp_path, driver=STORE_ID)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
        ) as store:
            store.send_json(_hello_frame(STORE_TOKEN))
            assert store.receive_json()["role"] == "driver"
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{THIRD_ID}"}
            ) as stranger:
                stranger.send_json(_hello_frame(""))
                assert stranger.receive_json()["role"] == "standby"
                health = client.get("/health").json()
                assert health["standby_extension_ids"] == []
                # Counted rather than invisible, so the bound below is checkable
                # from outside and an operator can see a dial that is not paired.
                assert health["unlisted_extension_count"] == 1
            # The paired install's own standby still reports as one.
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
            ) as unpacked:
                unpacked.send_json(_hello_frame(UNPACKED_TOKEN))
                assert unpacked.receive_json()["role"] == "standby"
                assert client.get("/health").json()["standby_extension_ids"] == [UNPACKED_ID]


@pytest.mark.asyncio
async def test_u24_unpaired_dials_are_bounded_and_the_oldest_is_retired(
    tmp_path: Path,
) -> None:
    """m1 — the link map is bounded, and a legitimate install is still admitted.

    Every token-less dial is admitted (that is how a second install pairs) and
    each holds a socket, a link entry and a pending-code record — all grown by
    one unauthenticated frame. The OLDEST stranger gives way, so the bound can
    never refuse the operator's own install because somebody's stale socket got
    there first.
    """
    _schema_two(tmp_path, driver=STORE_ID)
    service = BridgeService(root=tmp_path)
    drive = _FakePeer(STORE_ID)
    tasks = [_connect(service, drive, STORE_TOKEN)]
    assert await _settles(lambda: len(service.links) == 2)

    strangers: list[_FakePeer] = []
    for index in range(MAX_UNLISTED_LINKS):
        # A valid Chrome extension id: 32 characters from a-p, which the
        # Origin check enforces before anything below it runs.
        peer = _FakePeer("a" * 31 + chr(ord("a") + index))
        strangers.append(peer)
        tasks.append(_connect(service, peer, ""))
        assert await _settles(
            lambda peer=peer: bool(peer.acks()), seconds=3.0
        ), "a stranger was not admitted to pair"
    assert len(service.unlisted_links()) == MAX_UNLISTED_LINKS

    overflow = _FakePeer("b" * 32)
    tasks.append(_connect(service, overflow, ""))
    assert await _settles(lambda: bool(overflow.acks()), seconds=3.0)
    assert await _settles(
        lambda: bool(strangers[0].closed), seconds=3.0
    ), "the oldest unpaired link was not retired to make room"
    assert strangers[0].closed[0] == 4004
    assert len(service.unlisted_links()) <= MAX_UNLISTED_LINKS
    assert service.link.extension_id == STORE_ID, "the bound disturbed the driver"
    await _shutdown(*tasks)


def test_u25_a_long_dead_pending_record_is_dropped(tmp_path: Path) -> None:
    """m1 — expired codes stop accumulating, without becoming a way in.

    The record is kept one TTL past expiry so the honest "that code expired"
    answer still has something to read, then dropped. The A1 lockout is the
    ROTATION, not the record, so forgetting a long-dead entry hands nobody
    authority: a fresh dial mints a fresh code the user must still read off the
    terminal.
    """
    now = time.time()
    pending = _pending_path(tmp_path)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(
        json.dumps(
            {
                "pending": {
                    STORE_ID: {"code": "111111", "expires_at": now + 60, "attempts": 0},
                    UNPACKED_ID: {"code": "222222", "expires_at": now - 10_000, "attempts": 5},
                }
            }
        ),
        encoding="utf-8",
    )
    entries = _pending_entries(tmp_path)
    assert set(entries) == {STORE_ID}, "a long-dead record was still live"
    # The sweep reaches the FILE the next time anything writes it.
    _write_pending(tmp_path, entries)
    on_disk = json.loads(pending.read_text(encoding="utf-8"))
    assert set(on_disk["pending"]) == {STORE_ID}
    # And an identity with no record simply asks again: no authority is inherited.
    service = BridgeService(root=tmp_path)
    assert service._valid_saved_token(UNPACKED_ID, UNPACKED_TOKEN) is False


# --- U26-U33: remediation round 2 — the durable driver-record invariant ------
#
# The property under test, stated once and asserted after every transition:
#
#   the legacy trio names a LISTED identity, and whenever a PAIRED driver is
#   attached it names THAT driver — i.e. the file agrees with
#   /health.driver_extension_id in every state an operator can be served from.
#
# Round 2 found the property falsified on two transitions the round-1 fix did
# not cover: a cold-start dial after a daemon restart (review R2-1) and a
# promotion during graceful shutdown (QA R2-1/the same major), the second of
# which writes the STANDBY into the downgrade record and makes a rollback refuse
# the install in use.


def _record(root: Path) -> str:
    """The identity the legacy trio names right now."""
    return str(json.loads(_pairing_file(root).read_text(encoding="utf-8")).get("extension_id", ""))


def _assert_invariant(service: BridgeService, root: Path, where: str) -> None:
    """The file names a listed identity, and the live paired driver when there is one."""
    named = _record(root)
    listed = {entry["extension_id"] for entry in pairing_status(root)["identities"]}
    assert named in listed, f"{where}: the trio names {named!r}, which is not authorised"
    driver = service.link
    if driver.websocket is not None and driver.paired:
        assert (
            named == driver.extension_id
        ), f"{where}: the file names {named!r} while {driver.extension_id!r} drives"


@pytest.mark.asyncio
async def test_u26_a_cold_start_after_a_restart_moves_the_record_to_the_driver(
    tmp_path: Path,
) -> None:
    """R2-1: the handshake moves the wheel, so it must move the record too.

    The state is the one QA's round-1 evidence calls load-bearing: after a
    restart whichever worker re-dials first takes the idle wheel. The file,
    meanwhile, still names whoever drove BEFORE the restart — so a rollback on it
    authorises an install that is not the one running, which is the harm M2/Q2
    was filed for.
    """
    # The file names the DEV build; the STORE build is the one that dials first.
    _schema_two(tmp_path, driver=UNPACKED_ID)
    assert _record(tmp_path) == UNPACKED_ID
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    task = _connect(service, store, STORE_TOKEN)
    assert await _settles(lambda: bool(store.acks()))
    assert store.acks()[0]["role"] == "driver"
    assert _record(tmp_path) == STORE_ID, "the cold-start dial did not move the durable record"
    _assert_invariant(service, tmp_path, "after a cold-start dial")
    await _shutdown(task)


@pytest.mark.asyncio
async def test_u27_a_paired_install_taking_the_wheel_records_itself(tmp_path: Path) -> None:
    """M1's demotion branch is a wheel move too, so it records as well.

    An unlisted, token-less dial holds the cold-start wheel (that is how a second
    install pairs). When the AUTHORISED, paired install dials next and takes the
    wheel, the record has to follow — otherwise a rollback authorises the
    stranger's predecessor rather than the install the operator is using.
    """
    _schema_two(tmp_path, driver=UNPACKED_ID)
    service = BridgeService(root=tmp_path)
    stranger = _FakePeer(THIRD_ID)
    stranger_task = _connect(service, stranger, "")
    assert await _settles(lambda: bool(stranger.acks()))
    assert stranger.acks()[0]["role"] == "driver" and stranger.acks()[0]["paired"] is False
    # A token-less dial cannot be recorded (no hash to write) and must not be:
    # the file keeps naming the last listed driver.
    assert _record(tmp_path) == UNPACKED_ID
    _assert_invariant(service, tmp_path, "with an unpaired incumbent driving")

    store = _FakePeer(STORE_ID)
    store_task = _connect(service, store, STORE_TOKEN)
    assert await _settles(lambda: bool(store.acks()))
    assert store.acks()[0]["role"] == "driver"
    assert _record(tmp_path) == STORE_ID, "the paired install took the wheel without recording it"
    _assert_invariant(service, tmp_path, "after a paired install took the wheel")
    await _shutdown(stranger_task, store_task)


@pytest.mark.asyncio
async def test_u28_drive_and_a_promoting_revoke_record_the_new_driver(tmp_path: Path) -> None:
    """The two explicit wheel moves QA exercised, asserted against the file."""
    _schema_two(tmp_path, driver=STORE_ID)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)
    _assert_invariant(service, tmp_path, "paired and driving")

    # The promote-on-revoke path, at socket level: revoking the DRIVER hands the
    # wheel to the standby, and the durable record has to follow it.
    await service._sever_identity(STORE_ID)  # type: ignore[attr-defined]
    assert await _settles(lambda: service.link.extension_id == UNPACKED_ID)
    assert _record(tmp_path) == UNPACKED_ID, "a promoting revoke left the record on the revoked id"
    _assert_invariant(service, tmp_path, "after revoking the driver")
    await _shutdown(*tasks)


def test_u28b_drive_pins_the_driver_and_the_record_follows(tmp_path: Path) -> None:
    """`POST /driver` is a wheel move, so it is a record move (QA exercised this)."""
    _schema_two(tmp_path, driver=STORE_ID)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
        ) as store:
            store.send_json(_hello_frame(STORE_TOKEN))
            assert store.receive_json()["role"] == "driver"
            with client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
            ) as unpacked:
                unpacked.send_json(_hello_frame(UNPACKED_TOKEN))
                assert unpacked.receive_json()["role"] == "standby"
                assert _record(tmp_path) == STORE_ID
                pinned = client.post(
                    "/driver", headers={"X-Bridge-Key": key}, json={"target": UNPACKED_ID[:12]}
                )
                assert pinned.status_code == 200, pinned.text
                assert (
                    _record(tmp_path) == UNPACKED_ID
                ), "POST /driver moved the wheel without moving the record"
                _assert_invariant(app.state.bridge, tmp_path, "after POST /driver")


@pytest.mark.asyncio
async def test_u29_shutdown_does_not_move_the_durable_record(tmp_path: Path) -> None:
    """QA R2-1: a graceful stop must not write the STANDBY into the downgrade record.

    `_retire_link` promotes on a driver loss, and during teardown the driver's
    socket ends like any other — so the round-1 fix dutifully PERSISTED a
    handover that serves nobody, and a rollback then refused the install the
    operator uses with 4004. Measured on the pre-fix head as flaky (2/6 SIGTERM
    runs), which is worse than deterministic.
    """
    for attempt in range(5):
        root = tmp_path / f"attempt-{attempt}"
        _schema_two(root, driver=STORE_ID)
        service = BridgeService(root=root)
        store = _FakePeer(STORE_ID)
        unpacked = _FakePeer(UNPACKED_ID)
        tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
        assert await _settles(lambda: len(service.links) == 3)
        assert _record(root) == STORE_ID

        service.begin_shutdown()  # what `shutdown()` and uvicorn's should_exit do
        store.push(None)  # the driver's socket ends during teardown
        assert await _settles(lambda: service.link.websocket is None)
        assert (
            _record(root) == STORE_ID
        ), f"attempt {attempt}: teardown promoted the standby into the durable record"
        # And the old daemon's own rule still authorises the store build's token.
        assert _valid_saved_token_for(root, STORE_ID, STORE_TOKEN) is True
        assert _valid_saved_token_for(root, UNPACKED_ID, UNPACKED_TOKEN) is False
        await _shutdown(*tasks)


def _valid_saved_token_for(root: Path, extension_id: str, token: str) -> bool:
    """The PRE-CHANGE daemon's rule, verbatim (`102195106:daemon.py:1018-1023`).

    The downgrade contract is exactly this predicate, so it is the honest oracle
    for "would a rollback work": the trio's id must match the dial and the
    trio's hash must match the token.
    """
    saved = json.loads(_pairing_file(root).read_text(encoding="utf-8"))
    if saved.get("extension_id") != extension_id or not token:
        return False
    return secrets.compare_digest(str(saved.get("token_sha256", "")), _digest(token))


@pytest.mark.asyncio
async def test_u30_a_server_going_down_is_not_a_driver_loss(tmp_path: Path) -> None:
    """The signal the fix keys on is explicit, and it is the earliest one available.

    uvicorn sets `Server.should_exit` when a stop signal arrives and only THEN
    closes listeners and connection sockets (uvicorn 0.52 `Server.shutdown`), so
    the service can tell teardown from a live driver loss without guessing from
    socket state. This row drives that path directly.
    """
    _schema_two(tmp_path, driver=STORE_ID)
    service = BridgeService(root=tmp_path)
    store = _FakePeer(STORE_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)

    class _Stopping:
        should_exit = True

    service.watch_server_exit(_Stopping())
    assert service._daemon_leaving() is True  # type: ignore[attr-defined]
    store.push(None)
    assert await _settles(lambda: service.link.websocket is None)
    assert _record(tmp_path) == STORE_ID
    assert unpacked.roles() == [], "a teardown handed the wheel over on the wire"
    _assert_invariant(service, tmp_path, "after a teardown-driven socket end")
    await _shutdown(*tasks)


def test_u31_startup_reconciles_a_record_that_names_nobody_authorised(tmp_path: Path) -> None:
    """Startup repairs a stale trio instead of assuming it.

    Reachable with no attacker: `lop browser pair --revoke <the identity the trio
    names>` runs in ANOTHER process while the daemon is down, so the file is left
    naming somebody who is no longer authorised — and a rollback would then
    authorise nobody at all. The repair uses the same order as every other write,
    and an ordinary file is left byte-identical.
    """
    _schema_two(tmp_path, driver=UNPACKED_ID)
    # The stale shape, written as FILE STATE rather than through the writers
    # (which by design repair the trio): an out-of-process revoke removed the
    # identity the trio names, while this daemon was down. Constructing it
    # directly is the only way to test the repair, and it is what an older CLI
    # leaves behind.
    saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
    saved["identities"] = [
        entry for entry in saved["identities"] if entry["extension_id"] != UNPACKED_ID
    ]
    _pairing_file(tmp_path).write_text(json.dumps(saved), encoding="utf-8")
    assert _record(tmp_path) == UNPACKED_ID, "precondition: the file names the revoked id"

    service = BridgeService(root=tmp_path)
    service._reconcile_driver_record()  # type: ignore[attr-defined]
    assert _record(tmp_path) == STORE_ID, "startup left the downgrade record pointing at nobody"

    # An already-coherent file is NOT rewritten: a restart must not churn it.
    before = _pairing_file(tmp_path).read_bytes()
    service._reconcile_driver_record()  # type: ignore[attr-defined]
    assert _pairing_file(tmp_path).read_bytes() == before


# --- Round 3: the runner path, the free-wheel rule, /driver's two refusals ---

#: The daemon as `create_app()` documents it being served — plain `uvicorn.run`,
#: no `watch_server_exit`, no injected shutdown. This is the runner QA round 3
#: reproduced 9/12 on (Q3-1): uvicorn closes connections BEFORE it fires the
#: lifespan shutdown, so a lifespan-only guard promoted the standby and wrote it
#: into the downgrade record on the way out.
_RUNNER_CHILD = """
import sys

import uvicorn

from local_operator.browser_bridge.daemon import create_app

port = int(sys.argv[1])
# `create_app(port)` sets the SERVICE's port; the socket is uvicorn's, so it has
# to be told separately. Nothing here wires `should_exit` for the service — that
# absence is the whole point of the row.
uvicorn.run(create_app(port), host="127.0.0.1", port=port, log_level="warning")
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _health(port: int) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as response:
        return json.loads(response.read().decode())


def _runner_shutdown_moved_the_record(root: Path, port: int) -> bool:
    """Run one real `uvicorn.run(create_app(...))` daemon, SIGTERM it, and report.

    Two identities are attached over the REAL wire (a driver and a standby), then
    the process is signalled exactly as launchd or a shell would. Returns whether
    the durable record ended up naming the standby — the harm itself, measured on
    the artifact a rollback reads, rather than inferred from the code path.
    """
    _schema_two(root, driver=STORE_ID)
    environment = {**os.environ, "LOCAL_OPERATOR_CONFIG_DIR": str(root)}
    child = subprocess.Popen(
        [sys.executable, "-c", _RUNNER_CHILD, str(port)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with suppress(Exception):
                if _health(port)["status"] == "ok":
                    break
            time.sleep(0.1)
        else:  # pragma: no cover - a daemon that never came up is a broken test, not a finding
            raise AssertionError("the child daemon never answered /health")

        with (
            sync_connect(
                f"ws://127.0.0.1:{port}/extension",
                additional_headers={"Origin": f"chrome-extension://{STORE_ID}"},
                open_timeout=5.0,
            ) as driver,
            sync_connect(
                f"ws://127.0.0.1:{port}/extension",
                additional_headers={"Origin": f"chrome-extension://{UNPACKED_ID}"},
                open_timeout=5.0,
            ) as standby,
        ):
            driver.send(json.dumps(_hello_frame(STORE_TOKEN)))
            assert json.loads(driver.recv())["role"] == "driver"
            standby.send(json.dumps(_hello_frame(UNPACKED_TOKEN)))
            assert json.loads(standby.recv())["role"] == "standby"
            assert _record(root) == STORE_ID, "precondition: the record names the driver"
            child.send_signal(signal.SIGTERM)
            # WAIT for the child to finish while BOTH sockets are still open. A
            # `with` that exits first closes both clients and the daemon then has
            # no proven standby to promote — which is how this row passed on the
            # broken tree the first time it was written, i.e. as a vacuous test.
            child.wait(timeout=30.0)
        child.wait(timeout=30.0)
    finally:
        if child.poll() is None:  # pragma: no cover - only on a hung child
            child.kill()
            child.wait(timeout=10.0)
    return _record(root) == UNPACKED_ID


def test_u32_a_plain_uvicorn_runner_does_not_move_the_record_on_sigterm(
    tmp_path: Path,
) -> None:
    """QA round 3, Q3-1: the guard must hold on the runner path, not only ours.

    `create_app()` is the public factory, and it is served in the wild the way its
    own docstring shows — `uvicorn.run(app)` — by embedders and by this repo's
    harnesses. Nothing on that path wires `should_exit` for the service, and
    uvicorn tears connections down BEFORE the lifespan shutdown event, so a
    lifespan-only guard promoted the standby on the way out and persisted it into
    the record a rollback reads (QA measured 9/12: 4/6 then 5/6).

    Deliberately the BLUNT instrument: a real subprocess running
    `uvicorn.run(create_app(...))`, two real websocket clients, a real SIGTERM,
    and the answer read out of the pairing file. The signal it relies on is the
    one the receive loop sees — uvicorn's `websocket.disconnect` carrying code
    1012, delivered before the lifespan — and it is repeated, because the harm was
    FLAKY and a single lucky pass is exactly what hid this gap before.
    """
    moved = [
        _runner_shutdown_moved_the_record(tmp_path / f"run-{index}", _free_port())
        for index in range(3)
    ]
    assert moved == [False, False, False], f"a runner-path shutdown moved the record: {moved}"


@pytest.mark.asyncio
async def test_u33_pairing_takes_the_wheel_from_an_unpaired_incumbent(tmp_path: Path) -> None:
    """Review R3-3: a PAIRED install outranks one that cannot serve a command.

    Reachable without an attacker, and the shape is routine: the wheel is free, an
    install that is loaded but NOT yet paired dials first and takes it (the
    handshake gives a free wheel to whoever completes `hello`, because a token-less
    dial has to be admitted so it CAN pair), and only then does the authorised
    install dial — as a tokenless standby, since the M1 rule ranks a *paired* dial
    above it and this one is not paired yet. Entering the code then makes it
    paired while an unpaired dial still holds the wheel.

    The handshake and the promotion path both refuse to leave the wheel with a
    link that cannot answer a command; `_take_free_wheel` did not, so the install
    that had just proved it could serve sat as a standby until the other one's
    next dial — up to the alarm floor, which is a user watching a dead agent.
    """
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    stranger = _FakePeer(THIRD_ID)
    unpacked = _FakePeer(UNPACKED_ID)
    tasks = [_connect(service, stranger, ""), _connect(service, unpacked, "")]
    assert await _settles(lambda: len(service.links) == 3)
    # Precondition, asserted rather than assumed: the unpaired dial holds the
    # wheel and the authorised-but-tokenless install is the standby.
    assert stranger.acks()[0]["role"] == "driver"
    assert unpacked.acks()[0]["role"] == "standby"

    codes = pairing_status(tmp_path)["pending"]
    # Both unpaired dials have a code of their own (per-identity pending, §1.3):
    # pick THIS install's rather than the stranger's.
    mine = [entry for entry in codes if entry["extension_id"] == UNPACKED_ID]
    assert len(mine) == 1, codes
    unpacked.push({"event": "pair", "code": mine[0]["code"]})
    assert await _settles(lambda: unpacked.roles() == ["driver"]), unpacked.roles()

    assert unpacked.roles() == ["driver"], "the install that just proved it can serve was left idle"
    assert stranger.roles() == ["standby"], "the unpaired incumbent was left holding the wheel"
    assert service.link.extension_id == UNPACKED_ID
    _assert_invariant(service, tmp_path, "after a pairing took the free wheel")
    await _shutdown(*tasks)


def test_u34_drive_names_an_authorised_install_that_is_not_connected(tmp_path: Path) -> None:
    """UX U4: `drive <id>` must not answer "nothing matched" about a listed install.

    After a handover, the demoted install reads "paired, not connected" in
    `status`, so the natural next command is `drive <its id>`. Resolution is
    against live LINKS, so it answered 404 "no single connected extension
    matches" with an empty candidate list — contradicting the listing the user
    had just read, and reading as a typo'd id.
    """
    _schema_two(tmp_path)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key

        absent = client.post("/driver", headers={"X-Bridge-Key": key}, json={"target": UNPACKED_ID})
        assert absent.status_code == 409, absent.json()
        assert absent.json()["error"] == "not_connected"
        assert "not connected right now" in absent.json()["message"]

        # An unknown target is still an unknown target.
        unknown = client.post("/driver", headers={"X-Bridge-Key": key}, json={"target": "deadbeef"})
        assert unknown.status_code == 404
        assert unknown.json()["error"] == "unknown_extension"


def test_u35_the_label_says_which_thing_the_version_belongs_to(tmp_path: Path) -> None:
    """Copy review C3: `Chrome 0.1.10` reads as an ancient BROWSER version.

    Chrome itself is at 153 in the reported case, and the string's job is to say
    which Local Operator build in which browser holds the wheel.
    """
    label = daemon_module._browser_label(
        "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/153.0.0.0 Safari/537.36", "0.1.13"
    )
    assert label == "Chrome extension 0.1.13", label
    # No version reported: the browser name alone, unchanged from before.
    assert daemon_module._browser_label("Mozilla/5.0 ... Chrome/153", "") == "Chrome"
    # No browser recognised: still say WHICH extension, never a bare number.
    assert daemon_module._browser_label("", "0.1.13") == "extension 0.1.13"


def test_u36_two_identical_labels_refuse_rather_than_guess(tmp_path: Path) -> None:
    """UX U2 / copy review C1: the daemon declines an ambiguous label, and says so.

    Two installs of the SAME build share a label byte for byte, which is the case
    a single label cannot address. The CLI words this as "no single connected
    extension matches" and lists candidates with their labels, so the refusal has
    to carry them.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    add_identity(tmp_path, UNPACKED_ID, _digest(UNPACKED_TOKEN), label="Chrome extension 0.1.13")
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        ambiguous = client.post(
            "/driver",
            headers={"X-Bridge-Key": app.state.bridge.state.session_key},
            json={"target": "Chrome extension 0.1.13"},
        )
        assert ambiguous.status_code == 404, ambiguous.json()
        assert ambiguous.json()["error"] == "unknown_extension"
        # Both candidates, so the caller can print `id  label` rows and the user
        # has something to paste that resolves (the id prefix).
        assert sorted(ambiguous.json()["authorized_extension_ids"]) == sorted(
            [STORE_ID, UNPACKED_ID]
        )

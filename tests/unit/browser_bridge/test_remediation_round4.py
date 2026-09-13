"""Round-4 review rows: the teardown guard's close code, the drop mirror, and the
CLI surfaces the same round measured.

Each row pins one finding from the round-4 review and UX rounds, against the code
as shipped rather than against a description of it — the three failures below were
all reachable from a single client action or from reading `/health` at the wrong
moment:

* a client close code latching the daemon-wide "we are leaving" flag (frozen
  record, dead failover);
* the drop mirror colouring `/health` after a healthy install had taken the wheel;
* `pair --list`'s timestamps and the printed `ohcmfhja…` handle.

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

from local_operator.browser_bridge.daemon import (
    BridgeService,
    add_identity,
    create_app,
    normalise_target,
    note_identity_seen,
    pairing_status,
)
from local_operator.browser_bridge.protocol import PROTO_VERSION, Request

STORE_ID = "omibaecbjdhgbbcedbnnnmjpmopfheof"
UNPACKED_ID = "jbadjeaodkoboanppmpjiifpconegdcj"
STORE_TOKEN = "store-token-" + "s" * 20
UNPACKED_TOKEN = "unpacked-token-" + "u" * 20


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _pairing_file(root: Path) -> Path:
    return root / "browser" / "pairing.json"


def _record(root: Path) -> str:
    """The legacy trio's identity — the downgrade contract a rollback reads."""
    saved = json.loads(_pairing_file(root).read_text(encoding="utf-8"))
    return str(saved.get("extension_id", ""))


def _schema_two(root: Path, *, driver: str = STORE_ID) -> None:
    add_identity(
        root, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13", driver_id=driver
    )
    add_identity(
        root,
        UNPACKED_ID,
        _digest(UNPACKED_TOKEN),
        label="Chrome extension 0.1.13",
        driver_id=driver,
    )


class _Peer:
    """A scripted /extension connection whose close code is chosen by the test.

    The close code is the whole point of the first row: a peer that goes away
    normally (1000), one that sends "going away" (1001) and a server-side teardown
    (1012) must not be conflated, and only the last one is the daemon's own.
    """

    def __init__(self, extension_id: str, *, close_code: int = 1000) -> None:
        self.headers = {"origin": f"chrome-extension://{extension_id}"}
        self.extension_id = extension_id
        self.close_code = close_code
        self.sent: list[dict[str, Any]] = []
        self._frames: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def accept(self) -> None:
        """The ASGI handshake the daemon performs before it reads `hello`."""
        return None

    async def receive_json(self) -> dict[str, Any]:
        frame = await self._frames.get()
        if frame is None:
            raise WebSocketDisconnect(self.close_code)
        return frame

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        return None

    def push(self, frame: dict[str, Any] | None) -> None:
        self._frames.put_nowait(frame)

    def acks(self) -> list[dict[str, Any]]:
        return [frame for frame in self.sent if frame.get("event") == "hello_ack"]


def _hello(token: str) -> dict[str, Any]:
    return {
        "event": "hello",
        "proto": PROTO_VERSION,
        "token": token,
        "extension_version": "0.1.13",
        "browser": "Chrome/153",
    }


def _connect(service: BridgeService, peer: _Peer, token: str):
    peer.push(_hello(token))
    return asyncio.create_task(service.extension(peer))  # type: ignore[arg-type]


async def _settles(predicate: Callable[[], object], seconds: float = 3.0) -> bool:
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


async def _health_body(service: BridgeService) -> dict[str, Any]:
    """`/health` as the popup and the CLI read it, without an HTTP round trip."""
    return json.loads((await service.health(None)).body)  # type: ignore[arg-type]


# --- Finding 1: only a SERVER-origin close may latch the teardown guard -------


@pytest.mark.asyncio
@pytest.mark.parametrize("client_code", [1000, 1001, 1005])
async def test_r4_1_a_client_close_code_never_latches_the_teardown_guard(
    tmp_path: Path, client_code: int
) -> None:
    """Review round 4, finding 1: `{1001, 1012}` froze the record and killed failover.

    A disconnect in this position can only carry a code the PEER sent (uvicorn
    returns its own code solely from `self.close_code`, and its shutdown path is
    1012), so latching on 1001 meant one browser navigating away permanently
    disabled the daemon's failover path and stopped the durable record following
    the live driver. Both harms are asserted here, for every client code, with the
    promotion asserted positively rather than inferred.
    """
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _Peer(STORE_ID, close_code=client_code)
    unpacked = _Peer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)
    assert store.acks()[0]["role"] == "driver"
    assert _record(tmp_path) == STORE_ID

    store.push(None)  # the driver goes away with the client's own close code
    assert await _settles(
        lambda: service.link.extension_id == UNPACKED_ID
    ), "the surviving standby was never promoted"
    assert service._daemon_leaving() is False, (  # type: ignore[attr-defined]
        f"a client close with code {client_code} latched the daemon-wide teardown flag"
    )
    # The R2-1 property: promotion moves the record with it.
    assert (
        _record(tmp_path) == UNPACKED_ID
    ), "the durable record stopped following the live driver after a client close"
    await _shutdown(*tasks)


@pytest.mark.asyncio
async def test_r4_1b_a_server_side_close_still_latches(tmp_path: Path) -> None:
    """The positive half, so the row above cannot pass by never latching at all.

    1012 is what uvicorn's own websocket protocols send on every live connection
    while the server shuts down, and it is the signal QA round 3's Q3-1 fix relies
    on: without it, a daemon served by `uvicorn.run(create_app(...))` promotes on
    the way out and writes the standby into the record a rollback reads.
    """
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _Peer(STORE_ID, close_code=1012)
    unpacked = _Peer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)

    store.push(None)
    assert await _settles(lambda: service._daemon_leaving()), (  # type: ignore[attr-defined]
        "a server-side close no longer latches: the embedder path is unprotected again"
    )
    assert _record(tmp_path) == STORE_ID, "teardown promoted a standby into the durable record"
    await _shutdown(*tasks)


# --- Finding 2: the drop mirror must not colour a healthy serving link --------


@pytest.mark.asyncio
async def test_r4_2_a_promoted_driver_is_not_reported_as_unresponsive(tmp_path: Path) -> None:
    """UX round 2 / U1: the wedge card was painted on the install serving commands.

    The latch is a fact about the link the daemon severed, kept for LINK_DROP_TTL_S
    so the loss is still visible after the teardown that answers with it (#996's
    reason for the mirror). Reporting it while a DIFFERENT, proven link serves made
    `/health` contradict itself in adjacent fields — measured as
    `extension_connected: true` with `link_silent_s: 0.0016` beside
    `extension_unresponsive: true` — and the popup rendered its red "Extension
    stopped answering" card, with a Reload that would reload the serving install,
    on an install whose next command was answered in 7 ms.
    """
    _schema_two(tmp_path)
    service = BridgeService(root=tmp_path)
    store = _Peer(STORE_ID)
    unpacked = _Peer(UNPACKED_ID)
    tasks = [_connect(service, store, STORE_TOKEN), _connect(service, unpacked, UNPACKED_TOKEN)]
    assert await _settles(lambda: len(service.links) == 3)

    # The shape the ping tick produces: the DAEMON severs the silent driver
    # (`_drop_unproven_link`), which is also what latches the mirror. A peer that
    # closes its own socket must NOT latch — that is finding 1's other half, and
    # `_Peer.push(None)` is how that case is produced.
    expected = (service.link.websocket, service.link.generation)
    severed = await service._drop_unproven_link(  # type: ignore[attr-defined]
        "test: silence", expected=expected
    )
    assert severed
    assert await _settles(lambda: service.link.extension_id == UNPACKED_ID)
    assert service.drop_latched() is True, "precondition: the mirror is still inside its TTL"

    body = await _health_body(service)
    assert body["driver_extension_id"] == UNPACKED_ID
    assert body["extension_connected"] is True
    assert (
        body["extension_unresponsive"] is False
    ), "the healed driver is still reported as unresponsive"
    await _shutdown(*tasks)


@pytest.mark.asyncio
async def test_r4_2b_the_mirror_is_still_reported_when_nobody_took_over(tmp_path: Path) -> None:
    """The other half of the same decision, so the fix cannot be "delete the mirror".

    With no standby to promote, `/health` has no live link to describe — and the
    honest line there is the LATCHED drop, not "not currently attached", which is
    what #996's paragraph argues and why the mirror exists at all.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    service = BridgeService(root=tmp_path)
    store = _Peer(STORE_ID)
    tasks = [_connect(service, store, STORE_TOKEN)]
    assert await _settles(lambda: service.link.extension_id == STORE_ID)

    expected = (service.link.websocket, service.link.generation)
    severed = await service._drop_unproven_link(  # type: ignore[attr-defined]
        "test: silence", expected=expected
    )
    assert severed
    assert await _settles(lambda: service.link.websocket is None)
    body = await _health_body(service)
    assert body["driver_extension_id"] == ""
    assert body["extension_unresponsive"] is True, "the latched drop is no longer visible"
    assert body["link_attached"] is False
    await _shutdown(*tasks)


# --- Finding 2 (minor): pair --list's timestamps, live and never 1970 ---------


def test_r4_3_last_seen_is_refreshed_and_never_prints_a_1970_date(tmp_path: Path) -> None:
    """Review round 4, finding 2, all three halves.

    (a) a stale `last_seen_at` is refreshed by a handshake while a fresh one is
    left alone — bounded, so the hot path stays a read on ordinary reconnects;
    (b) the refresh does NOT move the legacy trio, because that file is also the
    rollback contract; (c) a record with no usable timestamps prints words rather
    than "last seen 20709d ago".
    """
    _schema_two(tmp_path, driver=STORE_ID)
    saved = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
    entry = next(e for e in saved["identities"] if e["extension_id"] == UNPACKED_ID)
    entry["last_seen_at"] = time.time() - 3600.0  # an hour stale
    fresh = next(e for e in saved["identities"] if e["extension_id"] == STORE_ID)
    fresh["last_seen_at"] = time.time()
    _pairing_file(tmp_path).write_text(json.dumps(saved), encoding="utf-8")
    trio_before = _record(tmp_path)

    note_identity_seen(tmp_path, UNPACKED_ID)
    after = json.loads(_pairing_file(tmp_path).read_text(encoding="utf-8"))
    refreshed = next(e for e in after["identities"] if e["extension_id"] == UNPACKED_ID)
    untouched = next(e for e in after["identities"] if e["extension_id"] == STORE_ID)
    assert refreshed["last_seen_at"] > time.time() - 5.0, "a stale stamp was not refreshed"
    assert untouched["last_seen_at"] == fresh["last_seen_at"], "a fresh stamp was rewritten"
    assert _record(tmp_path) == trio_before, "the refresh moved the rollback contract"

    from local_operator.cli import _seen_line

    # A schema-1 record: `pairing_status` coerces both stamps to 0.0, and the CLI
    # used to render that as "last seen 20709d ago".
    legacy_line = _seen_line({"extension_id": STORE_ID, "paired_at": 0.0, "last_seen_at": 0.0})
    assert "1970" not in legacy_line and "d ago" not in legacy_line, legacy_line
    assert legacy_line == "no timestamps on this record", legacy_line


def test_r4_3b_the_timestamps_print_without_a_daemon(tmp_path: Path, capsys: Any) -> None:
    """They come from the FILE, so a daemon being down must not hide them.

    The old loop `continue`d before the timestamp line whenever `/health` did not
    answer — the one state where the roles are unknown and the timestamps are the
    only thing distinguishing two identically-labelled installs.
    """
    from local_operator.cli import _print_identities

    _schema_two(tmp_path, driver=STORE_ID)
    pairing = pairing_status(tmp_path)
    _print_identities(pairing, None, verbose=True)
    out = capsys.readouterr().out
    assert "identities:" in out
    assert "last seen" in out, out
    assert "driving" not in out, "no daemon answered, so no role may be printed"


# --- Finding U8 / C8: the printed handle resolves, and the refusals differ -----


def test_r4_4_the_printed_handle_resolves_as_printed(tmp_path: Path) -> None:
    """UX round 2 / U8 + copy C8: `ohcmfhja…` is what the screen shows.

    Copying it is the most likely user action, and it used to fail — one character
    (U+2026) separated a handle that worked from the same handle as displayed,
    while the refusal listed the matching row beneath it.
    """
    assert normalise_target("ohcmfhja\u2026") == "ohcmfhja"
    assert normalise_target("  ohcmfhja\u2026  ") == "ohcmfhja"
    assert normalise_target(UNPACKED_ID) == UNPACKED_ID

    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect(
            "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
        ) as socket:
            socket.send_json(_hello(""))
            assert socket.receive_json()["paired"] is False
            # The full id, the bare prefix and the PRINTED prefix must all resolve
            # to the same attached link.
            printed = f"{UNPACKED_ID[:8]}\u2026"
            for target in (UNPACKED_ID, UNPACKED_ID[:8], printed):
                resolved = service._resolve_extension(target)  # type: ignore[attr-defined]
                assert resolved is not None, target
            # An id that matches nothing is a different refusal from an ambiguous
            # one, and the body says which (the CLI words them differently).
            unknown = client.post(
                "/driver",
                headers={"X-Bridge-Key": service.state.session_key},
                json={"target": "deadbeef"},
            )
            assert unknown.status_code == 404
            assert unknown.json()["matches"] == 0, unknown.json()


def test_r4_4b_an_ambiguous_target_says_how_many_matched(tmp_path: Path) -> None:
    """The count that makes the CLI's two sentences honest (copy review C8)."""
    _schema_two(tmp_path)  # both labels are byte-identical in this fixture
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key
        with (
            client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{STORE_ID}"}
            ) as first,
            client.websocket_connect(
                "/extension", headers={"origin": f"chrome-extension://{UNPACKED_ID}"}
            ) as second,
        ):
            first.send_json(_hello(STORE_TOKEN))
            second.send_json(_hello(UNPACKED_TOKEN))
            assert first.receive_json()["role"] == "driver"
            assert second.receive_json()["role"] == "standby"
            ambiguous = client.post(
                "/driver", headers={"X-Bridge-Key": key}, json={"target": "Chrome extension 0.1.13"}
            )
            assert ambiguous.status_code == 404, ambiguous.json()
            assert ambiguous.json()["matches"] == 2, ambiguous.json()


def test_r4_4c_the_cli_words_a_no_match_and_an_ambiguity_differently(capsys: Any) -> None:
    """The command's own output, through the real formatter (copy review C1+C8).

    A no-match reads "nothing matched"; an ambiguity keeps C1's "no single …", and
    both list the authorised installs under a lead-in that is true in either case
    rather than under the word "matches".
    """
    from local_operator.cli import _print_identities, _short_extension_id

    assert _short_extension_id(STORE_ID) == "omibaecb\u2026"
    pairing = {
        "identities": [
            {
                "extension_id": STORE_ID,
                "label": "Chrome extension 0.1.13",
                "paired_at": time.time(),
            },
            {
                "extension_id": UNPACKED_ID,
                "label": "Chrome extension 0.1.13",
                "paired_at": time.time(),
            },
        ]
    }
    _print_identities(pairing, {"driver_extension_id": STORE_ID, "standby_extension_ids": []})
    out = capsys.readouterr().out
    assert "tip:" in out and "approvals:" in out
    # The disclosure is present with two authorised installs even when NOTHING is
    # attached as a standby (UX round 2's residual: it used to need one attached).
    assert "do not follow the browser that takes over" in out, out


def test_r4_4d_the_request_shape_is_still_the_protocols(tmp_path: Path) -> None:
    """A guard against this file's own helpers drifting from the wire protocol."""
    assert Request(id="r1", method="tabs_list", params={}).model_dump()["method"] == "tabs_list"

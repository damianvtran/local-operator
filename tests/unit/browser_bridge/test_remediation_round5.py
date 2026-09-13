"""Round-5 rows: the zero-migration property, the guarded refresh, the forged-close
narrowing, the takeover bound, and the CLI wording assertions.

Round 5 found one regression in round 4's own delta and three smaller gaps. Every
row here exists because something was green a round earlier and wrong a round later
(`test_r4_3` built a schema-2 fixture, which is exactly why the migration regression
escaped), so each one asserts the property at the level the failure was found:
the FILE, not the dict this module happens to hold.
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
from starlette.websockets import WebSocketDisconnect

from local_operator.browser_bridge import daemon as daemon_module
from local_operator.browser_bridge.daemon import (
    LINK_SILENCE_TIMEOUT_S,
    PING_INTERVAL_S,
    BridgeService,
    add_identity,
    note_identity_seen,
)
from local_operator.browser_bridge.protocol import PROTO_VERSION

STORE_ID = "omibaecbjdhgbbcedbnnnmjpmopfheof"
UNPACKED_ID = "jbadjeaodkoboanppmpjiifpconegdcj"
STORE_TOKEN = "store-token-" + "s" * 20
UNPACKED_TOKEN = "unpacked-token-" + "u" * 20


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _pairing_file(root: Path) -> Path:
    return root / "browser" / "pairing.json"


def _legacy_record(root: Path, *, paired_at: float = 1_700_000_000.0) -> Path:
    """Write exactly what a pre-this-branch `lop` wrote: schema 1, trio only.

    No ``identities`` list and no ``last_seen_at`` — the shape whose first handshake
    was rewritten into schema 2 (QA round 5, Q5-3), i.e. the shape the refresh must
    now leave untouched.
    """
    path = _pairing_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "extension_id": STORE_ID,
                "token_sha256": _digest(STORE_TOKEN),
                "paired_at": paired_at,
                "schema": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


class _Peer:
    """A scripted /extension connection whose close code the test chooses."""

    def __init__(self, extension_id: str, *, close_code: int = 1000) -> None:
        self.headers = {"origin": f"chrome-extension://{extension_id}"}
        self.extension_id = extension_id
        self.close_code = close_code
        self.sent: list[dict[str, Any]] = []
        self._frames: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def accept(self) -> None:
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


def _hello(token: str) -> dict[str, Any]:
    return {
        "event": "hello",
        "proto": PROTO_VERSION,
        "token": token,
        "extension_version": "0.1.13",
        "browser": "Chrome/153",
    }


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


# --- Q5-3: zero migration, asserted against the FILE -------------------------


@pytest.mark.asyncio
async def test_r5_1_a_legacy_record_is_not_migrated_by_a_handshake(tmp_path: Path) -> None:
    """QA round 5, Q5-3 — MAJOR regression, and the row that would have caught it.

    A schema-1 record has no ``last_seen_at`` at all, and round 4's staleness test
    read the missing stamp as ``0.0`` ("20709d ago"), so the first handshake of a
    legacy-paired install rewrote the operator's file: 163 B → 392 B, schema 1 → 2.
    Decision 1 promises zero migration in BOTH directions, so the file must be
    byte-identical after a connect. `cases.py 7`'s `no_eager_migration` is the same
    property from outside; this row keeps it inside the suite.
    """
    path = _legacy_record(tmp_path)
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    service = BridgeService(root=tmp_path)
    peer = _Peer(STORE_ID)
    peer.push(_hello(STORE_TOKEN))
    task = asyncio.create_task(service.extension(peer))  # type: ignore[arg-type]
    assert await _settles(lambda: bool(peer.sent))
    assert peer.sent[0]["paired"] is True, "the legacy token must still authenticate"

    assert path.read_bytes() == before, "the first handshake rewrote the pairing file"
    assert path.stat().st_mtime_ns == before_mtime, "the first handshake touched the file"
    assert json.loads(before).get("schema") == 1
    await _shutdown(task)


def test_r5_1b_absent_and_malformed_stamps_are_unknown_not_stale(tmp_path: Path) -> None:
    """The two shapes that must not write, one call each, no daemon involved."""
    _legacy_record(tmp_path)
    path = _pairing_file(tmp_path)
    before = path.read_bytes()
    note_identity_seen(tmp_path, STORE_ID)  # no `last_seen_at` at all
    note_identity_seen(tmp_path, UNPACKED_ID)  # no entry at all
    assert path.read_bytes() == before

    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["identities"][0]["last_seen_at"] = "not-a-number"
    path.write_text(json.dumps(malformed), encoding="utf-8")
    before = path.read_bytes()
    note_identity_seen(tmp_path, STORE_ID)
    assert path.read_bytes() == before, "a malformed stamp was treated as stale"


def test_r5_1c_a_stale_stamp_is_refreshed_and_a_fresh_one_is_not(tmp_path: Path) -> None:
    """The half that must still work: round 4's liveness signal, unchanged."""
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    path = _pairing_file(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["identities"][0]["last_seen_at"] = time.time() - 3600.0
    path.write_text(json.dumps(saved), encoding="utf-8")

    note_identity_seen(tmp_path, STORE_ID)
    refreshed = json.loads(path.read_text(encoding="utf-8"))["identities"][0]["last_seen_at"]
    assert refreshed > time.time() - 5.0

    path.write_text(
        json.dumps(
            {"schema": 2, "identities": [{"extension_id": STORE_ID, "last_seen_at": refreshed}]}
        )
    )
    before = path.read_bytes()
    note_identity_seen(tmp_path, STORE_ID)
    assert path.read_bytes() == before, "a fresh stamp was rewritten"


def test_r5_2_a_concurrent_revoke_is_not_clobbered(tmp_path: Path) -> None:
    """Review round 5, finding 2: the write must not resurrect a revoked identity.

    `lop browser pair --revoke` runs in a SEPARATE process and writes this same
    file. The refresh re-reads immediately before writing, so a revoke that landed
    while the refresh was deciding wins — simulated here by making the second read
    (and every later one) return the post-revoke list.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    path = _pairing_file(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["identities"][0]["last_seen_at"] = time.time() - 3600.0
    path.write_text(json.dumps(saved), encoding="utf-8")
    before = path.read_bytes()

    calls = {"n": 0}
    real = daemon_module._identities

    def revoking_read(root: Path | None) -> list[dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] == 1:
            return real(root)
        return []  # the out-of-process revoke landed in between

    daemon_module._identities = revoking_read  # type: ignore[assignment]
    try:
        note_identity_seen(tmp_path, STORE_ID)
    finally:
        daemon_module._identities = real  # type: ignore[assignment]
    assert path.read_bytes() == before, "the refresh restored a concurrently revoked identity"


def test_r5_3_a_failed_write_does_not_fail_the_handshake(tmp_path: Path) -> None:
    """A read-only or full config root must cost a timestamp, not the connection.

    The refresh runs on the handshake path, so an unhandled `OSError` there would
    break connecting on a machine that can still serve — the same hazard
    `publish_safely` exists for next door.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    path = _pairing_file(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["identities"][0]["last_seen_at"] = time.time() - 3600.0
    path.write_text(json.dumps(saved), encoding="utf-8")

    def exploding_write(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    real = daemon_module._write_pairing
    daemon_module._write_pairing = exploding_write  # type: ignore[assignment]
    try:
        note_identity_seen(tmp_path, STORE_ID)  # must not raise
    finally:
        daemon_module._write_pairing = real  # type: ignore[assignment]


# --- Review finding 1: a forged close code cannot latch for a stranger --------


@pytest.mark.asyncio
async def test_r5_4_an_unpaired_stranger_cannot_latch_the_teardown_guard(tmp_path: Path) -> None:
    """Review round 5, finding 1 — narrowed, with the residual stated in the code.

    A browser cannot send 1012, but a local process can forge the Origin AND the
    close code. Such a dial holds neither pairing authority nor the wheel, so its
    frame must be inert: previously it latched the daemon's permanent "we are
    leaving" flag from outside the pairing boundary, which froze the durable record
    and disabled failover.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    service = BridgeService(root=tmp_path)
    store = _Peer(STORE_ID)
    stranger = _Peer("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", close_code=1012)
    store.push(_hello(STORE_TOKEN))
    stranger.push(
        {
            "event": "hello",
            "proto": PROTO_VERSION,
            "token": "",
            "extension_version": "0.1.13",
            "browser": "Chrome/153",
        }
    )
    tasks = [
        asyncio.create_task(service.extension(store)),  # type: ignore[arg-type]
        asyncio.create_task(service.extension(stranger)),  # type: ignore[arg-type]
    ]
    assert await _settles(lambda: len(service.links) == 3)
    assert service.link.extension_id == STORE_ID, "precondition: the paired install drives"

    stranger.push(None)  # forged 1012 from a link that holds nothing
    assert await _settles(lambda: service.links.get(stranger_id(service, stranger)) is None)
    assert service._daemon_leaving() is False, (  # type: ignore[attr-defined]
        "a stranger's forged 1012 latched the daemon-wide teardown flag"
    )
    await _shutdown(*tasks)


def stranger_id(service: BridgeService, peer: _Peer) -> int:
    """The generation the daemon gave ``peer`` (its link map is keyed by that)."""
    for generation, link in service.links.items():
        if link.extension_id == peer.extension_id:
            return generation
    return -1


@pytest.mark.asyncio
async def test_r5_4b_an_unpaired_WHEEL_HOLDER_still_latches(tmp_path: Path) -> None:
    """The narrowing must not disarm the guard for the link that matters.

    An unpaired dial that took the free wheel can still move the durable record on
    its way out (a promotion on a driver loss), so its 1012 has to latch — this is
    the case the guard exists for, and the reason the predicate is "authority OR
    the wheel" rather than "authority" alone.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    service = BridgeService(root=tmp_path)
    stranger = _Peer("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", close_code=1012)
    stranger.push(
        {
            "event": "hello",
            "proto": PROTO_VERSION,
            "token": "",
            "extension_version": "0.1.13",
            "browser": "Chrome/153",
        }
    )
    task = asyncio.create_task(service.extension(stranger))  # type: ignore[arg-type]
    assert await _settles(lambda: service.link.extension_id == stranger.extension_id)
    assert service.link.paired is False

    stranger.push(None)
    assert await _settles(lambda: service._daemon_leaving()), (  # type: ignore[attr-defined]
        "the wheel-holder's close no longer latches: the teardown rule is disarmed"
    )
    await _shutdown(task)


# --- UX round 3, U9: which of the two cases is the user in -------------------


@pytest.mark.asyncio
async def test_r5_5_the_silent_wheel_exposes_a_takeover_bound(tmp_path: Path) -> None:
    """UX round 3, U9 — the fact copy needs, so no surface has to hardcode a minute.

    A peer close promotes immediately (0.07 s measured on a real rig); a silent
    wedge is only discoverable by silence and takes the deadline plus one ping tick
    (61.2 s measured). `/health` now says which case the reader is in and how long
    the daemon's own bound is, while it is pending — and says nothing (`null`) when
    there is nothing pending, so "takeover coming" is never inferred from a zero.
    """
    add_identity(tmp_path, STORE_ID, _digest(STORE_TOKEN), label="Chrome extension 0.1.13")
    service = BridgeService(root=tmp_path)
    store = _Peer(STORE_ID)
    store.push(_hello(STORE_TOKEN))
    task = asyncio.create_task(service.extension(store))  # type: ignore[arg-type]
    assert await _settles(lambda: service.link.extension_id == STORE_ID)

    healthy = json.loads((await service.health(None)).body)  # type: ignore[arg-type]
    assert healthy["takeover_within_s"] is None, "a healthy driver is not a pending takeover"

    # Age the link past the proof deadline without sending anything: exactly the
    # state the standby install sits in for a minute before the daemon acts.
    service.link.last_frame_at = time.monotonic() - (LINK_SILENCE_TIMEOUT_S + 1.0)
    pending = json.loads((await service.health(None)).body)  # type: ignore[arg-type]
    assert pending["extension_unresponsive"] is True
    assert pending["takeover_within_s"] is not None
    assert 0.0 < float(pending["takeover_within_s"]) <= LINK_SILENCE_TIMEOUT_S + PING_INTERVAL_S

    service.latch_drop(service.link, 51.0)
    store.push(None)
    assert await _settles(lambda: service.link.websocket is None)
    severed = json.loads((await service.health(None)).body)  # type: ignore[arg-type]
    assert severed["takeover_within_s"] is None, "a wheel nobody holds has no takeover pending"
    await _shutdown(task)

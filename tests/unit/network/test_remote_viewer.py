"""A viewer on THIS device drives a session whose runtime lives on a PEER.

Plan §0 finding 2: ``remote_owner_for`` and ``RemoteSessionClient`` existed and
were reached only from tests, so no surface could pilot a remote session. The
production seam is ``session.remote_open.open_remote_viewer``, which the TUI's
sidebar pick, ``/resume`` and the shared session factory all go through. These
tests drive that SAME facade over two real relays, with a REAL ``Session`` (mock
provider, ``hosting: test``) behind a real ``RuntimeServer`` on the owning device,
and assert what the OWNER's transcript and runtime recorded — not what the viewer
believes it sent.

Device A is the VIEWER here and B the OWNER, mirroring the create tests this rig
comes from: A creates on B through the product's own ``net_session_create``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.peer_rows import clear_cache
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures
from tests.unit.network.test_session_plane import (
    Devices,
    _create_named_session_on_a_real_peer,
    _NamedRemoteCreate,
)


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    pair: Devices = request.getfixturevalue("devices")
    return pair


@pytest.fixture(autouse=True)
def _no_cached_rows() -> Any:
    clear_cache()
    yield
    clear_cache()


async def _wait(predicate: Any, timeout_s: float = 30.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if await asyncio.to_thread(predicate):
            return True
        await asyncio.sleep(0.05)
    return bool(await asyncio.to_thread(predicate))


def _user_texts(created: _NamedRemoteCreate) -> list[str]:
    """Every user message the OWNER's transcript journalled, in order."""
    texts: list[str] = []
    for entry in created.owner.transcript_entries():
        payload = entry.get("payload") or {}
        if payload.get("role") != "user":
            continue
        blob = json.dumps(payload.get("content"))
        texts.append(blob)
    return texts


async def _open(created: _NamedRemoteCreate, monkeypatch: pytest.MonkeyPatch) -> Any:
    from local_operator.session.remote_open import open_remote_viewer

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))

    async def _never() -> Any:
        raise AssertionError("a remote viewer never takes over")

    viewer = await open_remote_viewer(
        created.session_id, config_dir=created.server_a.root, takeover=_never
    )
    assert viewer is not None, "the peer's session did not resolve as remote"
    return viewer


@pytest.mark.asyncio
async def test_a_remote_session_is_opened_and_piloted_on_the_peer(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt, rename, a routed slash, /model and /stop all land on the PEER."""
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer, peer_pair, monkeypatch, name="pilot", prompt=""
    )
    try:
        viewer = await _open(created, monkeypatch)
        try:
            assert viewer.runtime_locality == "another-machine"
            await viewer.bind_runtime()
            assert not viewer.is_cold, viewer.cold_reason
            # NOTHING IS WRITTEN ON THIS DEVICE for a session it does not own.
            assert not (created.server_a.root / "sessions" / created.session_id).exists()

            # PROMPT: the owner's transcript journals it and the mock answers.
            await viewer.prompt("hello from the other device")
            assert await _wait(
                lambda: any("hello from the other device" in t for t in _user_texts(created))
            ), _user_texts(created)
            await asyncio.to_thread(created.owner.wait_for_turn)

            # STEER: the frame crosses both relays to the OWNER's session. Between
            # turns it is QUEUED there (``Session._steering_queue``) for the next
            # turn rather than journalled, so the queue is what is read — on the
            # peer's session object, not the viewer's.
            viewer.steer("steer from the other device")

            def steered() -> bool:
                queue = getattr(created.owner.session, "_steering_queue", None) or ()
                pending = list(getattr(queue, "_queue", ()))  # asyncio.Queue's own deque
                return any("steer from the other device" in str(m) for m in pending) or any(
                    "steer from the other device" in t for t in _user_texts(created)
                )

            assert await _wait(steered), _user_texts(created)

            # RENAME, the way the TUI sends it (`/rename` is advertised
            # authoritative, so the app routes it through `route_shared_slash`):
            # the OWNER's naming state is what moves.
            renamed = await viewer.route_shared_slash("rename", "renamed from A")
            assert isinstance(renamed, dict) and renamed.get("style") != "error", renamed
            assert await _wait(
                lambda: created.owner.session.conversation_name == "renamed from A"
            ), created.owner.session.conversation_name

            # A ROUTED SLASH answers from the peer's runtime.
            receipt = await viewer.route_shared_slash("goal", "ship the mesh")
            assert isinstance(receipt, dict), receipt
            assert await _wait(
                lambda: "ship the mesh" in str(getattr(created.owner.session, "goal", "") or "")
                or "ship the mesh" in json.dumps(created.owner.transcript_entries())
            ), receipt

            # /model, routed the same way: the OWNER answers from its own registry.
            model_receipt = await viewer.route_shared_slash("model", "")
            assert isinstance(model_receipt, dict), model_receipt

            # /stop: the deliberate stop reaches the owner and it answers.
            answer = await viewer.request_stop()
            assert answer, "the peer did not answer the stop"
        finally:
            await viewer.dispose()
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_closing_the_remote_viewer_leaves_the_peer_runtime_running(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell 1.2: the viewer going away (a TUI quitting) never stops the peer."""
    from local_operator.session.runtime import registry

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer, peer_pair, monkeypatch, name="quit", prompt=""
    )
    try:
        viewer = await _open(created, monkeypatch)
        await viewer.bind_runtime()
        root_b = created.server_b.root

        def live() -> list[int]:
            return [
                rec.pid
                for rec, state in registry.scan(root_b)
                if state == "live" and rec.session_id == created.session_id
            ]

        before = await asyncio.to_thread(live)
        assert before, "the peer's runtime was not live to begin with"
        await viewer.dispose()
        await asyncio.sleep(0.5)
        assert await asyncio.to_thread(live) == before, "closing the viewer stopped the peer"
        assert created.owner.runtime is not None
    finally:
        await asyncio.to_thread(created.stop)


def test_a_session_this_device_holds_never_resolves_as_remote(tmp_path: Path) -> None:
    """R16 topology 0: no relay, no network — ``None``, with no dial made."""
    from local_operator.session.remote_open import remote_row_for

    (tmp_path / "sessions" / "0123456789ab").mkdir(parents=True)
    assert remote_row_for("0123456789ab", tmp_path) is None
    assert remote_row_for("feedfacecafe", tmp_path) is None


@pytest.mark.asyncio
async def test_the_desktop_bridge_reads_and_prompts_a_peers_session(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DESKTOP's own read path over a real pair: snapshot, history, a prompt.

    Slice V proved the viewer over real relays for the TUI. What this cell adds
    is the half the desktop reaches through a different door: ``DesktopSessions``
    resolves the peer's id, builds the SAME viewer through
    ``remote_open.open_remote_viewer``, and serves the transcript out of the
    WIRE — the read that has no local journal behind it. Three things are
    asserted, and each is a requirement rather than a detail:

    * the snapshot carries the peer's rows (so a click paints a conversation
      rather than an empty panel claiming the conversation starts here);
    * nothing is written on the VIEWING device — no session directory for an id
      it does not own (INV-1's two-writer case, from the desktop's side);
    * a prompt sent through the desktop's own control envelope lands in the
      PEER's transcript, read from the owner's journal rather than from what the
      viewer believes it sent.
    """
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    from local_operator.session.peer_rows import clear_cache

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="desktop-open",
        prompt="hello from the peer",
    )
    try:
        # The projection cache is 20 s and the pool reads it cache-first, so the
        # row has to come from a REAL fan-out for this test to mean anything: the
        # clear makes the pool pay that read, exactly as a machine whose sidebar
        # has not polled yet does.
        await asyncio.to_thread(clear_cache)
        pool = DesktopSessions(created.server_a.root)
        try:
            async with pool.session(created.session_id, read=True) as bridge:
                assert bridge.remote_row is not None, "the desktop did not resolve the peer's row"
                assert bridge.remote_row.owner_device == created.server_b.identity.device_id

                snapshot = await bridge.snapshot()
                entries = snapshot["payload"]["history"]["entries"]
                assert entries, snapshot["payload"]["history"]
                assert "hello from the peer" in json.dumps(entries), entries
                # NOTHING IS WRITTEN HERE: a conversation another device owns has no
                # directory on this disk, and the wire is the only source.
                assert not (created.server_a.root / "sessions" / created.session_id).exists()

                page = await bridge.history(limit=50)
                assert "hello from the peer" in json.dumps(page["entries"]), page

            # A PROMPT THROUGH THE DESKTOP'S CONTROL ENVELOPE EXECUTES ON THE PEER.
            async with pool.session(created.session_id) as bridge:
                assert bridge.remote is not None
                # A REAL command id: the owner validates it as the canonical form
                # of a UUID, because it is the durable reservation key for one
                # admitted turn rather than any opaque label.
                detail, _duplicate = await bridge.remote.admit_prompt(
                    "sent from the desktop", command_id=str(uuid.uuid4()), images=[]
                )
                assert detail, "the peer did not admit the desktop's prompt"
            assert await _wait(
                lambda: any("sent from the desktop" in text for text in _user_texts(created))
            ), _user_texts(created)
            await asyncio.to_thread(created.owner.wait_for_turn)
        finally:
            await pool.close()
    finally:
        await asyncio.to_thread(created.stop)


def _png_bytes(width: int = 240, height: int = 240) -> bytes:
    """A real, decodable PNG with enough entropy to clear the externalise floor.

    Noise rather than a flat colour on purpose: a constant image compresses to a
    few hundred bytes and would land UNDER ``transcript._ATTACHMENT_FLOOR_BYTES``,
    where the row keeps the payload inline and NOTHING about the store is
    exercised — a cell that would pass on the defect.
    """
    import random
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + tag + data + zlib.crc32(tag + data).to_bytes(4, "big")

    rng = random.Random(7)
    rows = [
        b"\x00"
        + bytes(
            value
            for _ in range(width)
            for value in (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        )
        for _ in range(height)
    ]
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00",
        )
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )


async def _desktop_route_client(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The desktop's own router, over a real store, as the app attaches it."""
    import os

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_sessions
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "mesh-image-token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.desktop_sessions = DesktopSessions(root)
    app.include_router(desktop_sessions.router)
    return app, AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer mesh-image-token"},
    )


@pytest.mark.asyncio
async def test_an_image_a_desktop_sends_to_a_peer_stays_readable_here(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A peer-bound prompt's image must resolve on the device that SENT it.

    THE DEFECT THIS PINS (measured on the unfixed tree). A turn's images are
    content-addressed: the OWNER's journal row references
    ``{"attachment": <digest>}`` and nothing else, and the only store this
    device's ``GET .../attachments/<digest>`` reads is its own. The owner's
    runtime is another process with its own config dir, so it externalises the
    bytes into ITS store — the desktop held nothing, answered ``409
    attachment_on_peer`` over the picture the user had just sent, and the prompt
    itself reported ``admitted``. So the desktop mirrors what it sends
    (``DesktopSessionBridge.stage_peer_images``), and the digest being the
    CONTENT key is what makes a local copy resolvable against a row written on
    another machine.

    THE AMBIENT CONFIG DIR MOVES FOR THE ADMISSION, and that is the cell's own
    rigour rather than decoration: an in-process rig shares one
    ``LOCAL_OPERATOR_CONFIG_DIR``, so without this the peer's externalise would
    write into THIS device's store by accident and the assertions below would
    pass on the unfixed tree. Pointing it at a third directory for the admission
    puts the peer's copy where production puts it (the peer's own store) and
    leaves this device's store holding exactly what a real desktop holds.
    """
    import base64
    import hashlib
    import shutil

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="desktop-image",
        prompt="warm up",
    )
    try:
        await asyncio.to_thread(clear_cache)
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))
        raw = _png_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:32]
        app, client = await _desktop_route_client(created.server_a.root, monkeypatch)
        async with client:
            # BIND BEFORE THE ENV MOVES, so the dial this device makes resolves
            # its own relay and nothing else.
            async with app.state.desktop_sessions.session(created.session_id, read=True) as b:
                await b.snapshot()
            peer_store = tmp_path / "peer-env"
            monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(peer_store))
            response = await client.post(
                f"/v1/desktop/sessions/{created.session_id}/messages",
                json={
                    "request_id": str(uuid.uuid4()),
                    "text": "here is a picture",
                    "images": [
                        {
                            "data_b64": base64.b64encode(raw).decode("ascii"),
                            "mime_type": "image/png",
                        }
                    ],
                },
            )
            monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))
            assert response.status_code == 200, response.text

            # THE OWNER GOT THE BYTES: its row references the digest of what was
            # sent, which is the only way the two ends can name the same image.
            await asyncio.to_thread(created.owner.wait_for_turn)
            rows = [
                entry.get("payload") or {}
                for entry in created.owner.transcript_entries()
                if (entry.get("payload") or {}).get("role") == "user"
            ]
            assert digest in json.dumps(rows), rows

            # THE PEER'S OWN COPY GOES WHERE PRODUCTION PUTS IT, so what remains
            # here is what this device really holds.
            peer_own = created.server_b.root / "attachments"
            peer_own.mkdir(parents=True, exist_ok=True)
            for suffix in (".bin", ".json"):
                source = peer_store / "attachments" / f"{digest}{suffix}"
                assert source.exists(), sorted(p.name for p in source.parent.glob("*"))
                shutil.move(str(source), str(peer_own / source.name))

            read = await client.get(
                f"/v1/desktop/sessions/{created.session_id}/attachments/{digest}"
            )
            assert read.status_code == 200, (
                "the desktop cannot show the image it just sent to its own peer "
                f"session ({read.status_code}): {read.text[:200]}"
            )
            assert read.content == raw, "the bytes served are not the bytes sent"
    finally:
        await asyncio.to_thread(created.stop)

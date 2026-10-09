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
    _call,
    _create_named_session_on_a_real_peer,
    _dial_to,
    _listen,
    _NamedRemoteCreate,
    _pair,
    _seed_journal,
)

#: One peer-owned id, shared with the session-plane cells this rig comes from.
SESSION = "9f3ac1e0b7d2"


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


def _user_payload_with(created: _NamedRemoteCreate, text: str) -> dict[str, Any] | None:
    """The journalled user ROW whose content carries ``text``, or ``None`` until it lands.

    ``_user_texts`` flattens rows to their content blobs; keeping the whole payload
    is what lets a test read the silent metadata the carriage writes beside the text
    (see ``Message.input_mode``).
    """
    for entry in created.owner.transcript_entries():
        payload = entry.get("payload") or {}
        if payload.get("role") != "user":
            continue
        if text in json.dumps(payload.get("content")):
            return payload
    return None


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


@pytest.mark.asyncio
async def test_a_cold_peer_read_serves_the_owners_stored_journal(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5-core, end to end: the defect was that this page was EMPTY.

    The operator's report ("at times I can't see the full conversation — it says
    it's the end of the conversation when it's not") traced to here: a peer
    session with no runtime answered ``{entries: [], has_more: false}``, the same
    envelope a conversation with no rows produces, while the rows sat on the
    owner's disk. Three facts are asserted, and the second is the one that keeps
    the fix from being a warm-on-read:

    * the page carries the OWNER's rows — real ids, real timestamps, in order;
    * the OWNER IS STILL COLD afterwards: no runtime was spawned, which is what
      makes this work at all for a session that was deliberately stopped;
    * the DTO is the local page's shape, so the renderer needs no second path.

    The rig is deliberately the real one -- two relays, a paired mesh, a real
    journal on the peer and NO runtime behind it -- because the code under test
    spans four processes' worth of boundaries (route -> relay -> link -> owner
    store) and a stub would pin the seam rather than the behaviour.
    """
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    server_a, server_b, _host_a, _port_a = peer_pair
    ids = _seed_journal(server_b.root, SESSION, ["first from the peer", "second"])
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    link = _dial_to(server_a, record, host_b, port_b)
    pool = DesktopSessions(server_a.root)
    try:
        clear_cache()
        async with pool.session(SESSION, read=True) as bridge:
            assert bridge.remote_row is not None, "the pool did not resolve the peer's row"
            page = await bridge.history(limit=50)
        assert [entry["id"] for entry in page["entries"]] == ids, page
        assert [entry["ts"] for entry in page["entries"]] == [1_700_000_000.0, 1_700_000_001.0]
        assert page["has_more"] is False
        assert page["cursor_missing"] is False
        # THE READ STARTED NOTHING. This is the requirement that rules out every
        # warm-on-read design: a stopped session can never be engaged, so a page
        # that costs a spawn is a page that does not exist for exactly the
        # conversations the operator was complaining about.
        states = [
            row.get("state")
            for row in server_b.local_session_rows()
            if row.get("session_id") == SESSION
        ]
        assert states == ["stored"], f"a cold desktop read left the owner at {states}"
    finally:
        await pool.close()
        link.close("test")


@pytest.mark.asyncio
async def test_a_stored_page_and_a_wire_window_merge_by_id(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race this change must not create: two sources, ONE id space.

    A cold read serves journal rows and a warm read serves wire rows for the same
    conversation, and a renderer merges them by ``id``. That is only safe because
    the wire's message id IS the entry id (``transcript.encode_message_payload``
    drops ``id`` from the payload precisely because the row already carries it) —
    so the stored page's rows and a later window's rows dedupe rather than
    double-painting the same message.

    Both sources are read for ONE real conversation (a real turn behind a real
    ``RuntimeServer``), and the property pinned is the id space, not the order
    the two are read in: the stored page's message rows are a SUBSET of the
    window's, because the journal legitimately carries rows the wire window does
    not (custom entries), while every row the window carries comes from that same
    journal. A stop-and-restart of the owner's runtime is deliberately NOT part
    of this cell: that exercises the spawn path, which the engage and create
    cells already cover, and it is the IDS that make the merge safe.
    """
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    created = _create_named_session_on_a_real_peer(
        peer_pair, monkeypatch, name="merge-by-id", prompt="hello from the peer"
    )
    pool = DesktopSessions(created.server_a.root)
    try:
        # BOTH SOURCES READ AFTER THE TURN SETTLES, so the comparison is between two
        # complete views of one conversation rather than between a page read
        # mid-turn and a window read after it.
        await asyncio.to_thread(created.owner.wait_for_turn)

        # THE STORED SOURCE, through the op the cold fallback itself calls.
        stored = _call(
            created.server_a.root,
            "peer_session_history",
            peer=created.peer_token,
            session_id=created.session_id,
            limit=50,
        )["detail"]
        stored_ids = [entry["id"] for entry in stored["entries"]]
        assert stored_ids, stored

        clear_cache()
        async with pool.session(created.session_id, read=True) as bridge:
            wire = await bridge.history(limit=50)
        wire_ids = [entry["id"] for entry in wire["entries"]]
        assert wire_ids, wire

        # ONE ID SPACE. The window's rows come out of the journal, so every row it
        # carries is a row the stored page also names under the SAME id -- which is
        # the whole reason a renderer can merge the two by id rather than painting
        # a message twice when a cold page is followed by a warm window. (The
        # reverse is not claimed, and must not be: the journal also holds custom
        # entries the message window does not carry.)
        assert set(wire_ids) <= set(stored_ids), (wire_ids, stored_ids)
        assert set(stored_ids) & set(wire_ids), "the two sources share no row at all"

        # AND THE MERGE A RENDERER PERFORMS IS LOSSLESS: keying both sources by id
        # paints every row once, which is the property a duplicated id would break.
        merged = {entry["id"]: entry for entry in (*stored["entries"], *wire["entries"])}
        assert set(merged) == set(stored_ids) | set(wire_ids)
    finally:
        await pool.close()
        await asyncio.to_thread(created.stop)


def _png_bytes(width: int, height: int) -> bytes:
    """A real, decodable PNG, with enough entropy to clear the externalise floor.

    Noise rather than a flat colour on purpose: a constant image compresses to a
    few hundred bytes and would land UNDER ``transcript._ATTACHMENT_FLOOR_BYTES``,
    where the row keeps the payload inline and NOTHING about the store is
    exercised — a cell that would pass on the defect.
    """
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + tag + data + zlib.crc32(tag + data).to_bytes(4, "big")

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        # Deterministic, structured and COMPRESSIBLE, and both halves are
        # load-bearing: a noise PNG at 1024 px is megabytes, and the desktop route
        # refuses the body on its own bounds (``Image.data_b64`` max 1,000,000
        # chars, ``Prompt``'s 900 KB canonical frame) long before the mesh sees it,
        # so a sweep built from noise measures nothing but the route's 422. Flat
        # tiles with a slow ramp give a real screenshot's byte budget: tens of KB
        # at 1024 px, a few hundred at 2400.
        tile = ((x // 32) * 6 + (y // 32) * 3) % 256
        return (tile, (tile + 40) % 256, (tile + 90) % 256)

    rows = [b"\x00" + bytes(v for x in range(width) for v in pixel(x, y)) for y in range(height)]
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00",
        )
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )


def _rotated_jpeg(width: int, height: int) -> bytes:
    """A JPEG whose EXIF ``Orientation`` says the pixels must be transposed.

    The shape the shipped composer can put on the wire: ``bound-image.ts`` returns
    an image VERBATIM when it is already within its own edge/byte bound, and a
    phone or download JPEG within that bound keeps its ``Orientation`` tag — which
    is exactly the tag the owner's ingest acts on (``imaging._needs_exif_rotation``
    → a transposing re-encode), so the bytes it journals are not the bytes sent.
    """
    from local_operator.imaging import _needs_exif_rotation, pillow_image_module

    image_module = pillow_image_module()
    assert image_module is not None, "this rig needs Pillow to build the rotated shape"
    image = image_module.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x % 256, y % 256, (x * y) % 256)
    import io

    buffer = io.BytesIO()
    # 6 = "rotate 90 CW", the tag a phone camera writes for a landscape sensor
    # held upright. Built through Pillow's own ``Exif`` rather than by hand:
    # a hand-rolled APP1 block is malformed, ``_needs_exif_rotation`` reads it as
    # "no rotation", and the row silently becomes a CONTROL row — which is what
    # happened here first and is why the assertion below exists.
    exif = image_module.Exif()
    exif[0x0112] = 6
    image.save(buffer, format="JPEG", quality=90, exif=exif)
    payload = buffer.getvalue()

    # THE ROW MUST REALLY BE ROTATED, checked with the owner's own predicate: a
    # sweep row that quietly stops exercising its trigger is worse than no row,
    # because it reads as coverage.
    reopened = image_module.open(io.BytesIO(payload))
    assert _needs_exif_rotation(reopened), "the rotated sweep row is not rotated"
    return payload


#: THE SWEEP, and every row is a shape whose bytes the owner's ingest may rewrite
#: (``imaging.IMAGE_INGEST_MAX_EDGE`` is 1024 px; ``IMAGE_MAX_BYTES`` is 1 MiB; the
#: ``not rotated`` conjunct is the third trigger). 1023/1024 are the control rows —
#: the owner keeps those verbatim — and the three above them are the rows the
#: round-1 cell missed by using a 240x240 control.
_IMAGE_SHAPES: tuple[tuple[str, str, int, int, str], ...] = (
    ("png_1023px", "png", 1023, 640, "image/png"),
    ("png_1024px", "png", 1024, 640, "image/png"),
    ("png_1025px", "png", 1025, 640, "image/png"),
    ("png_2400x1600", "png", 2400, 1600, "image/png"),
    ("jpeg_900x600_exif_rotated", "jpeg", 900, 600, "image/jpeg"),
)


def _shape_bytes(shape: tuple[str, str, int, int, str]) -> bytes:
    """The image for one sweep row, built HERE rather than patched in from outside.

    THE TRAP THIS AVOIDS, measured by round 1's reviewer: patching a module-level
    builder through ``pytest_configure`` lands on a SECOND import of this module
    and silently no-ops, so three E2E probes "passed" while still sending the stock
    240x240 image — only the store trace caught it. Nothing here is patched: the
    row's shape is a parametrize argument and the bytes are built from it inside
    the test, so a row cannot silently become another row.
    """
    _name, kind, width, height, _mime = shape
    if kind == "jpeg":
        return _rotated_jpeg(width, height)
    return _png_bytes(width, height)


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

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "[redacted]")
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.desktop_sessions = DesktopSessions(root)
    app.include_router(desktop_sessions.router)
    return app, AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer [redacted]"},
    )


def _journal_digests(created: _NamedRemoteCreate) -> list[str]:
    """Every attachment digest the OWNER's journal rows carry, in order."""
    import json

    digests: list[str] = []
    for entry in created.owner.transcript_entries():
        payload = entry.get("payload") or {}
        if payload.get("role") != "user":
            continue
        for block in payload.get("content") or ():
            if isinstance(block, dict) and block.get("attachment"):
                digests.append(str(block["attachment"]))
        if "attachment" in json.dumps(payload) and not digests:
            digests.append("")
    return digests


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _IMAGE_SHAPES, ids=[row[0] for row in _IMAGE_SHAPES])
async def test_an_image_a_desktop_sends_to_a_peer_resolves_whatever_its_shape(
    shape: tuple[str, str, int, int, str],
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A peer-bound prompt's image must resolve on the device that SENT it.

    THE DEFECT THIS PINS (measured on the unfixed tree). A turn's images are
    content-addressed: the OWNER's journal row references
    ``{"attachment": <digest>}`` and nothing else, and the only store this
    device's ``GET .../attachments/<digest>`` reads is its own. The owner's
    runtime is another process with its own config dir, so it externalises the
    bytes into ITS store — the desktop held nothing, answered ``409
    attachment_on_peer`` over the picture the user had just sent, and the prompt
    itself reported ``admitted``.

    AND THE HALF ROUND 1 FOUND: the owner does not keep what it is given. Its
    ingest (``image_blocks`` → ``bound_image_for_model``) resizes anything over
    ``IMAGE_INGEST_MAX_EDGE`` (1024 px), re-encodes anything over
    ``IMAGE_MAX_BYTES``, and bakes an EXIF rotation into the pixels — so a mirror
    of the RAW wire bytes names a blob the owner's row never references, and the
    read still 409s. That is why the sweep carries three rows the round-1 cell
    (a 240x240 control) could not reach, and why the fix bounds each image with
    the owner's OWN ingest and sends exactly what it staged: the owner then
    returns those bytes unchanged, so the two digests are the same name by
    construction rather than by the coincidence that the shipped composer happens
    to bound at the same edge.

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
        name="desktop-image-" + shape[0],
        prompt="warm up",
    )
    try:
        await asyncio.to_thread(clear_cache)
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))
        raw = _shape_bytes(shape)
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
                            "mime_type": shape[4],
                        }
                    ],
                },
            )
            monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))
            assert response.status_code == 200, response.text

            # WHAT THE OWNER ACTUALLY JOURNALLED — the digest the desktop will
            # repaint from, which is the ONLY name the read has to resolve.
            await asyncio.to_thread(created.owner.wait_for_turn)
            journalled = _journal_digests(created)
            assert journalled, created.owner.transcript_entries()

            # THE PEER'S OWN COPY GOES WHERE PRODUCTION PUTS IT, so what remains
            # here is what this device really holds.
            peer_own = created.server_b.root / "attachments"
            peer_own.mkdir(parents=True, exist_ok=True)
            for suffix in (".bin", ".json"):
                source = peer_store / "attachments" / f"{journalled[-1]}{suffix}"
                assert source.exists(), (
                    f"{shape[0]}: the peer externalised nothing for {journalled[-1]} "
                    f"({sorted(p.name for p in (peer_store / 'attachments').glob('*'))})"
                )
                shutil.move(str(source), str(peer_own / source.name))

            read = await client.get(
                f"/v1/desktop/sessions/{created.session_id}/attachments/{journalled[-1]}"
            )
            staged = created.server_a.root / "attachments" / f"{journalled[-1]}.bin"
            print(
                f"SWEEP {shape[0]}: posted={hashlib.sha256(raw).hexdigest()[:12]} "
                f"journalled={journalled[-1][:12]} staged={staged.exists()} "
                f"GET={read.status_code} bytes={len(read.content)}"
            )
            assert read.status_code == 200, (
                f"{shape[0]}: the desktop cannot show the image it just sent to its own "
                f"peer session ({read.status_code}): {read.text[:200]}"
            )
            # BYTE-IDENTICAL TO THE BLOB THE OWNER'S ROW NAMES, which is what the
            # local path serves too: a local session's runtime bounds the image
            # with this same ingest, so the conversation shows the bounded bytes.
            assert (
                read.content == staged.read_bytes()
            ), f"{shape[0]}: the bytes served are not the bytes this device staged"
            # NOTHING UNREFERENCED IS LEFT BEHIND, which is the invariant stated as
            # a file listing: the only blob this device holds is the name the
            # owner's own row carries. The round-1 defect was exactly the other
            # shape — a blob under a name no row referenced, and a row naming a
            # blob nobody held.
            held = sorted(
                path.name for path in (created.server_a.root / "attachments").glob("*.bin")
            )
            assert held == [f"{journalled[-1]}.bin"], (
                f"{shape[0]}: this device holds {held}, and the owner's row names "
                f"{journalled[-1]}"
            )
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_peer_bound_command_carries_its_image_to_the_owner(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M3, CONFIRMED: the /command door can put a composer image in the owner's row.

    Round 1's M3 asked whether ``route_shared_slash`` can admit an image at all —
    the body claimed "both admission doors" while this third call carried composer
    images to the owner and staged nothing. It can: the route decodes the body
    with ``decode_images`` (the owner's own ingest), sends those blocks, and either
    the owner's runtime completes the receipt for itself (``serving.slash_images``
    admits them through the normal path) or this host re-admits the request with
    the same bounded bytes at ``admit_receipt_request``. Measured here: the owner's
    journal row carries a digest, and the sender can read it back.

    So the door is COVERED rather than declared unreachable: the route stages the
    payloads it sends (``stage_ingested_images``), before the call, because the
    owner can journal that row while this request is still in flight.
    """
    import base64
    import shutil

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="desktop-command-image",
        prompt="warm up",
    )
    try:
        await asyncio.to_thread(clear_cache)
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))
        raw = _shape_bytes(("png_1025px", "png", 1025, 640, "image/png"))
        app, client = await _desktop_route_client(created.server_a.root, monkeypatch)
        async with client:
            async with app.state.desktop_sessions.session(created.session_id, read=True) as b:
                await b.snapshot()
            peer_store = tmp_path / "peer-env"
            monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(peer_store))
            response = await client.post(
                f"/v1/desktop/sessions/{created.session_id}/commands",
                json={
                    "request_id": str(uuid.uuid4()),
                    "command": "goal",
                    "args": "Preserve one identity",
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
            await asyncio.to_thread(created.owner.wait_for_turn)
            journalled = _journal_digests(created)
            assert journalled, (
                "the /command door's image never reached an owner row, so this door "
                f"needs no staging at all: {created.owner.transcript_entries()}"
            )
            peer_own = created.server_b.root / "attachments"
            peer_own.mkdir(parents=True, exist_ok=True)
            for suffix in (".bin", ".json"):
                source = peer_store / "attachments" / f"{journalled[-1]}{suffix}"
                assert source.exists(), sorted(
                    p.name for p in (peer_store / "attachments").glob("*")
                )
                shutil.move(str(source), str(peer_own / source.name))
            read = await client.get(
                f"/v1/desktop/sessions/{created.session_id}/attachments/{journalled[-1]}"
            )
            print(
                f"COMMAND DOOR: journalled={journalled[-1][:12]} GET={read.status_code} "
                f"bytes={len(read.content)}"
            )
            assert read.status_code == 200, (
                "the /command door's image is unreadable on the device that sent it: "
                f"{read.status_code}: {read.text[:200]}"
            )
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_the_remote_client_mirrors_the_input_mode_carriage_gate(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-1: the remote dial resolves ``input-mode-v1`` from the OWNER's record.

    WHY THIS PIN CAN FAIL. Before commit 5bf1518e8 the carriage builders read
    ``_input_mode_supported`` directly, so a remote dial that never set the flag
    failed LOUD — the AttributeError that reddened CI's mesh pilot run. The same
    commit moved every read onto ``getattr(..., False)``, which makes the mirror
    in ``RemoteSessionClient.connect`` the only thing standing between a
    carriage-capable owner and a SILENT strip of the pair for every remote
    session: drop that line and every existing test still passes. So this cell
    is the counterpart of the local pins in
    ``tests/unit/mobile/test_attach_client.py`` (agent review round 2, R2-1),
    and it pins BOTH directions over a real pair: a capable owner ⇒ the flag
    resolves True and the pair rides to the owner's durable row; a record
    carrying no token this client gates on ⇒ False, fail closed, and the pair
    never leaves this device.
    """
    from local_operator.network import projection as projection_mod
    from local_operator.session.runtime.types import INPUT_MODE_CAPABILITY

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="carriage-mirror",
        prompt="",
    )
    try:
        viewer = await _open(created, monkeypatch)
        try:
            await viewer.bind_runtime()
            client = viewer._client
            assert client is not None, "the remote viewer bound no client"
            # The rig's owner MUST advertise the token, or the pair could fail to
            # reach the row below for a reason this test is not about.
            assert INPUT_MODE_CAPABILITY in client._facts.capabilities, client._facts.capabilities
            assert client._input_mode_supported is True, (
                "the remote mirror missed the owner's advertisement: dropping it "
                "strips the pair for every remote session, silently"
            )
            await viewer.admit_prompt(
                "spoken on the other device",
                command_id=str(uuid.uuid4()),
                images=[],
                input_mode="dictated",
                input_path="provider_stt_radient",
            )
            assert await _wait(
                lambda: _user_payload_with(created, "spoken on the other device") is not None
            ), _user_texts(created)
            carried = _user_payload_with(created, "spoken on the other device")
            assert carried is not None
            assert carried.get("input_mode") == "dictated", carried
            assert carried.get("input_path") == "provider_stt_radient", carried
            await asyncio.to_thread(created.owner.wait_for_turn)
        finally:
            await viewer.dispose()

        # THE OTHER DIRECTION: the owner still advertises its real token, but this
        # client gates on one the record does not carry — the shape a non-carriage
        # owner presents to a remote dial — and the read must fail closed rather
        # than carry.
        monkeypatch.setattr(projection_mod, "INPUT_MODE_CAPABILITY", "input-mode-v1-not-advertised")
        stripped_viewer = await _open(created, monkeypatch)
        try:
            await stripped_viewer.bind_runtime()
            stripped_client = stripped_viewer._client
            assert stripped_client is not None
            assert INPUT_MODE_CAPABILITY in stripped_client._facts.capabilities
            assert stripped_client._input_mode_supported is False, (
                "an unadvertised token must fail closed: a mirror that carries it "
                "anyway sends the pair to an owner whose record never promised it"
            )
            await stripped_viewer.admit_prompt(
                "typed on the other device",
                command_id=str(uuid.uuid4()),
                images=[],
                input_mode="mixed",
                input_path="provider_stt_elevenlabs",
            )
            assert await _wait(
                lambda: _user_payload_with(created, "typed on the other device") is not None
            ), _user_texts(created)
            stripped = _user_payload_with(created, "typed on the other device")
            assert stripped is not None
            assert "input_mode" not in stripped, stripped
            assert "input_path" not in stripped, stripped
        finally:
            await stripped_viewer.dispose()
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_an_owner_lost_verdict_leaves_no_session_directory_on_the_viewing_device(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PINNED AT ITS SOURCE: the writer of the 2026-09-29 recall strand.

    A viewer on THIS device, watching a session that lives on the peer, runs its
    owner-lost verdict when the owner's runtime goes away — and the verdict's
    journalling constructed a local ``Transcript`` for the peer's id. That
    construction is not read-only: ``Transcript`` MATERIALISES the directory it
    is pointed at, so the verdict left ``sessions/<id>/created_at.json`` on the
    VIEWING device — the bare remnant a later recall of that id stranded on
    (``test_mobility_recall_recovery`` carries the recall half; mobility's
    ``_bare_remnant`` the collision). A conversation that is not on this machine
    is not journalled here: the outcome that matters is written where the
    conversation is, by the owner's own runtime.

    THE PRECONDITION IS ESTABLISHED EXPLICITLY, AND EVENT-DRIVEN. The
    operator's viewer held a turn in flight when the runtime died (their
    session was parked mid-turn), and the verdict only fires for a live turn,
    so ``_streaming`` — the facade's own statement that a turn is live — is set
    directly. The owner is then torn down TO COMPLETION (``aclose`` on its own
    loop, not the bounded ``close``), the drop is delivered by invoking the
    callback the transport itself calls (``_on_disconnected``), and the
    recovery loop is AWAITED AS ITS OWN TASK — the wait ends exactly when the
    loop does, with no clock in it. The first shape of this cell waited on the
    EOF to travel two relays under a 20 s bound; it filled red on CI shard 0
    ("the owner-lost verdict never ran") while passing on every local run,
    for the reviewer and for QA. Everything under test is the shipped path
    from the drop's own callback on: the recovery deadline, the go-cold
    verdict, and the journalling guard itself.
    """
    from local_operator.session import attached as attached_mod

    monkeypatch.setattr(attached_mod, "COLD_FALLBACK_S", 0.5)
    journalled: list[str] = []
    real_journal = attached_mod.AttachedSession._journal_witnessed_cut_off  # noqa: SLF001

    def recording(self: Any, *, cause: str) -> None:
        # Recorded BEFORE the real method runs, so the assertion below proves
        # the verdict path was exercised whichever way the guard decides.
        journalled.append(cause)
        return real_journal(self, cause=cause)

    monkeypatch.setattr(attached_mod.AttachedSession, "_journal_witnessed_cut_off", recording)

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="owner-lost-verdict",
        prompt="",
    )
    try:
        viewer = await _open(created, monkeypatch)
        try:
            await viewer.bind_runtime()
            assert not viewer.is_cold, viewer.cold_reason
            assert (
                viewer.runtime_locality == "another-machine"
            ), "this cell is about a viewer whose session lives on a peer"
            viewer._streaming = True  # noqa: SLF001 — see the docstring

            # THE OWNER DIES — to COMPLETION, awaited. `close()` is bounded
            # (a 2 s wait on teardown) and may return while the runtime is
            # still winding down; under CI contention the recovery loop
            # then found a still-dialable owner and chased it instead of
            # reaching the verdict. `aclose` runs on the owner's own loop
            # (the rig's `on_session_loop`) and joins the SAME cleanup
            # task, so this returns only once the owner is really gone —
            # record withdrawn, sockets closed. No `stopping` announcement
            # rides it (a signal rung, a crash — not a /stop).
            await asyncio.to_thread(created.owner.on_session_loop, created.owner.runtime.aclose())

            # THE DROP, DELIVERED ONCE — invoked, not raced.
            # `_on_disconnected` is the exact callback the attach client
            # invokes on an owner exit; waiting for the EOF to travel two
            # relays is the other half of the same CI race and is not what
            # this cell is about. It is genuinely once-only whichever order
            # the real EOF callback arrives in: the second arrival returns
            # on the `_recovering` guard. The recovery loop is then AWAITED
            # AS ITS OWN TASK — it ends exactly when the loop does.
            viewer._on_disconnected("owner exited")  # noqa: SLF001 — the drop's own callback
            task = viewer._recovery_task
            assert task is not None, "the drop must start the recovery loop"
            await task

            assert (
                "owner-lost" in journalled
            ), "the owner-lost verdict never ran, so this cell would prove nothing"

            local = created.server_a.root / "sessions" / created.session_id
            assert not local.exists(), (
                "the viewer's verdict created "
                f"{sorted(p.name for p in local.iterdir())} on the viewing device "
                "for a session it does not own"
            )
        finally:
            await viewer.dispose()
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_the_remote_client_mirrors_the_operator_signature_gate(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remote dial resolves ``operator-signature-v1`` from the OWNER's record.

    WHY THIS PIN CAN FAIL, the same way its input-mode sibling can: the
    flag is read only through ``getattr(..., False)`` at
    ``_present_operator_signature``, so dropping the mirror leaves every other
    test green while a mesh viewer silently answers authority-increasing frames
    — an allow, a loosening slash — UNSIGNED. That is worse than a viewer with
    no signature route at all: the frame goes out, the runtime refuses it for
    want of a signature, and the surface that could have asked for one reads as
    complete.

    Both directions over a real pair, because the flag gates both: an owner
    that advertises it lets the frame through to the signing path (reached
    here WITHOUT touching the operator's key — the signing call itself is
    stubbed, and what is pinned is which ``(action, request_id)`` the signing
    path is asked for); an owner whose record carries no token this build
    gates on must fail closed, leaving the frame byte-identical to an older
    build's.
    """
    from local_operator.network import projection as projection_mod
    from local_operator.session.runtime.types import OPERATOR_SIGNATURE_CAPABILITY

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="signature-mirror",
        prompt="",
    )
    try:
        viewer = await _open(created, monkeypatch)
        try:
            await viewer.bind_runtime()
            client = viewer._client
            assert client is not None, "the remote viewer bound no client"
            # The rig's owner MUST advertise the token, or the second direction
            # could pass for a reason this cell is not about.
            assert (
                OPERATOR_SIGNATURE_CAPABILITY in client._facts.capabilities
            ), client._facts.capabilities
            assert client._operator_signature_supported is True, (
                "the remote mirror missed the owner's advertisement: every "
                "authority-increasing frame from this viewer goes out unsigned"
            )

            calls: list[tuple[str, str]] = []

            async def spy(action: str, request_id: str) -> Any:
                calls.append((action, request_id))
                return None

            monkeypatch.setattr(client, "_operator_signature", spy)
            frame = {"op": "approval_answer", "approved": True, "request_id": "card-1", "req": 9}
            out = await client._present_operator_signature(dict(frame))
            assert calls == [("approve", "card-1")], calls
            assert out == frame, "a failed signing call must leave the frame unchanged"
        finally:
            await viewer.dispose()

        # THE OTHER DIRECTION — the owner's record carries no token THIS client
        # gates on (the shape of a viewer built before the capability existed):
        # False, and the consumer does not even ask.
        monkeypatch.setattr(
            projection_mod, "OPERATOR_SIGNATURE_CAPABILITY", "operator-signature-v1-not-advertised"
        )
        stripped_viewer = await _open(created, monkeypatch)
        try:
            await stripped_viewer.bind_runtime()
            stripped_client = stripped_viewer._client
            assert stripped_client is not None
            assert OPERATOR_SIGNATURE_CAPABILITY in stripped_client._facts.capabilities
            assert stripped_client._operator_signature_supported is False, (
                "an unadvertised token must fail closed: a mirror that signs anyway "
                "presents to an owner whose record never promised it checks one"
            )
            asked: list[tuple[str, str]] = []

            async def spy_stripped(action: str, request_id: str) -> Any:
                asked.append((action, request_id))
                return None

            monkeypatch.setattr(stripped_client, "_operator_signature", spy_stripped)
            frame = {"op": "approval_answer", "approved": True, "request_id": "card-2", "req": 10}
            assert await stripped_client._present_operator_signature(dict(frame)) == frame
            assert asked == [], "the signing path was reached without the owner's advertisement"
        finally:
            await stripped_viewer.dispose()
    finally:
        await asyncio.to_thread(created.stop)


def _viewer_pending(viewer: Any) -> Any:
    """The viewer's pending gate, or ``None`` before its state synchronizes."""
    try:
        return viewer.pending_gate
    except RuntimeError:
        return None


@pytest.mark.asyncio
async def test_a_gate_parked_on_the_owner_reaches_a_bound_remote_viewer(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-2 trace pin: a gate parked on the owner must reach the viewer's card state.

    The origin-e2e defect (netpass F-5 / e2e-final F-2) reported that an
    attached viewer's surface says "Its gate card has not reached this view" —
    the viewer's canonical state never carried the pending gate. This cell
    drives the whole delivery path below the TUI — the owner's REAL gate
    installer (nothing here fakes the card), the relay passthrough, the
    viewer's client pump and facade, the bridge, and the deny travelling back
    — for BOTH attach orders. It exists so no layer on that path can silently
    regress while the live-side localization in the PR's F-2 section is
    carried out; a card that stops arriving at any of these seams fails it.
    """
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="parked-card",
        prompt="",
    )
    parked: Any = None
    try:
        viewer = await _open(created, monkeypatch)
        try:
            await viewer.bind_runtime()
            assert (
                _viewer_pending(viewer) is None
            ), "a gate was already pending before this cell parked one"
            served = created.owner
            # Park on the owner's OWN loop; the schedule call returns a
            # concurrent future without waiting on the gate, and teardown
            # denies whatever is still parked (``dispose``) even if this
            # cell's cancel loses the race.
            parked = asyncio.run_coroutine_threadsafe(
                served.handle._approval_gate("bash", "rm -rf build/"),  # noqa: SLF001
                served._loop,  # noqa: SLF001
            )
            assert await _wait(
                lambda: served.handle._fold.projection.pending is not None, timeout_s=15.0
            ), "the fixture did not park a gate on the owner"
            assert await _wait(
                lambda: _viewer_pending(viewer) is not None, timeout_s=15.0
            ), "the owner parked a gate the long-lived viewer's state never carried"
            gate = _viewer_pending(viewer)
            assert gate is not None and gate.kind == "approval", gate
            assert gate.title == "bash", gate
        finally:
            await viewer.dispose()

        # THE FRESH RE-ATTACH (netpass F-5's second shape): a viewer that binds
        # for the first time while the gate is ALREADY parked must receive it
        # WITH its actionable state — the bridge reaches the host, whose deny
        # then travels back and resolves the owner's gate.
        fresh = await _open(created, monkeypatch)
        try:
            answered: list[str] = []

            async def handler(tool_name: str, description: str) -> bool:
                answered.append(tool_name)
                return False

            fresh.set_approval_handler(handler)
            await fresh.bind_runtime()
            assert await _wait(
                lambda: _viewer_pending(fresh) is not None, timeout_s=15.0
            ), "a viewer binding during a live park never received the gate"
            assert await _wait(
                lambda: bool(answered), timeout_s=15.0
            ), "the gate reached the facade but never its host bridge"
            assert parked is not None
            assert await _wait(
                lambda: parked.done(), timeout_s=15.0
            ), "the pane's deny never reached the owner's parked gate"
            assert parked.result() is False, "the deny did not resolve the gate as denied"
        finally:
            await fresh.dispose()
    finally:
        if parked is not None and not parked.done():
            parked.cancel()
        await asyncio.to_thread(created.stop)

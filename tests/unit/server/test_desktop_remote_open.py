"""Opening and reading a conversation ANOTHER DEVICE holds, from the desktop.

WHAT THIS FILE PINS, and each item is a requirement of mesh slice DB2 rather
than an implementation detail:

* a peer's row OPENS: the pool resolves the id to a bridge whose owner is the
  remote device (``remote_open.open_remote_viewer`` — the same seam the TUI's
  pick and ``/resume`` use), so the snapshot, ``/history`` and every control act
  ride the mesh instead of answering 404 about the user's own conversation;
* the transcript is read OFF THE WIRE, never out of ``<root>/sessions/<id>`` —
  the local read ``mesh-session-mobility.md`` §3.4 forbids by name, which for a
  peer's id is at best absent and at worst a different conversation wearing the
  same id;
* an UNREACHABLE peer still refuses, with the SAME sentence the TUI refuses the
  same state with (``remote_open.unreachable_peer_sentence`` — the shared
  composer exists so two surfaces cannot describe one situation two ways);
* an id nobody holds is still the shared 404, and a LOCAL id is untouched —
  including its cost: the peer projection is never consulted for a directory
  this device already holds;
* this device's own stores are not written or believed about somebody else's
  conversation: a peer's completion receipts come from the peer's sync, not from
  a local ``attention.db`` that is empty by construction.

The REAL relay pair is the network suite's job
(``tests/unit/network/test_remote_viewer.py`` runs the desktop bridge over two
real relays and a real runtime on the peer). What is exercised here is the
mapping either side of that transport, against a real store on a real
filesystem and through the real ``errors()`` ladder.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import Message
from local_operator.resume import SessionRow
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attached import AttachedSession
from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore
from local_operator.session.owner import SessionSeed
from local_operator.session.placement import SessionPlacement
from local_operator.session.remote_open import unreachable_peer_sentence

MINE = "c" * 12
OTHER = "e" * 12
PEER = "d_" + "7" * 32
NET_ONE = "n_" + "3" * 24
DESKTOP_TOKEN = "synthetic-desktop-token"


class StubRemoteOwner:
    """The owner seam, answered without a peer.

    ``AttachedSession`` asks its owner five things (``placement``, ``seed``,
    ``locate``, ``engage``, ``make_client``). This answers the two a READ needs
    and FAILS LOUDLY on the other two, because a read that engaged a runtime or
    dialled a socket is the defect this file exists to keep out: the desktop's
    read envelope must never start work on somebody else's machine.
    """

    def __init__(self) -> None:
        self.placement = SessionPlacement(mode="peer", network_id=NET_ONE, home_device=PEER)

    def seed(self) -> SessionSeed:
        return SessionSeed(name="build box chat", model_label="", cwd="", device_name="build-box")

    def locate(self) -> tuple[Any, Any]:
        """No runtime on the peer, as far as a read can tell: the cold answer."""
        return None, None

    async def engage(self, **_kwargs: Any) -> None:
        raise AssertionError("a desktop READ must not engage a runtime on the peer")

    def make_client(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a desktop READ must not dial the peer")


@pytest_asyncio.fixture
async def remote_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The desktop router, a real store, and a peer row the projection will answer."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    from local_operator.session.cleanup import mark_store

    mark_store(tmp_path / "sessions")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {DESKTOP_TOKEN}"},
    ) as client:
        yield client, tmp_path.resolve()


def _peer_row(**overrides: Any) -> SessionRow:
    base: dict[str, Any] = {
        "id": OTHER,
        "mtime": 1789400000.0,
        "name": "build box chat",
        "live_state": "idle",
        "locality": "remote",
        "owner_device": PEER,
        "owner_device_name": "build-box",
        "reachable": True,
        "unreachable_reason": "",
    }
    base.update(overrides)
    return SessionRow(**base)


def _answer_rows(monkeypatch: pytest.MonkeyPatch, row: SessionRow | None) -> list[str]:
    """Point the pool's ONE remote question at ``row``, recording every ask."""
    from local_operator.session import remote_open

    asked: list[str] = []

    def fake_row(session_id: str, root: Any = None) -> SessionRow | None:
        asked.append(session_id)
        return row if row is not None and row.id == session_id else None

    monkeypatch.setattr(remote_open, "remote_row_for", fake_row)
    return asked


def _remote_facade(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[dict[str, Any]]:
    """Answer the viewer seam with a REAL cold facade whose owner is a stub peer."""
    from local_operator.session import remote_open

    built: list[dict[str, Any]] = []

    async def fake_open(session_id: str, **kwargs: Any) -> Any:
        built.append({"session_id": session_id, **kwargs})

        async def refuse_takeover() -> None:
            raise AssertionError("a remote viewer never takes over")

        return await AttachedSession.cold(
            session_id,
            config_dir=kwargs.get("config_dir", root),
            cwd="",
            takeover_factory=refuse_takeover,
            surface=kwargs.get("surface", "desktop"),
            owner=StubRemoteOwner(),
            seed=StubRemoteOwner().seed(),
        )

    monkeypatch.setattr(remote_open, "open_remote_viewer", fake_open)
    return built


def _seed_local(root: Path, session_id: str) -> None:
    path = root / "sessions" / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "created_at.json").write_text("1700000000")
    (path / "conversation.json").write_text(json.dumps({"name": "local chat"}))


# ---------------------------------------------------------------------------
# The refusal that REMAINS, and the 404 that must not move
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unreachable_peer_refuses_in_the_tuis_own_words(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """409 ``session_is_remote``, carrying the ONE composer both surfaces use.

    The row is visible, so 404 would be a lie about the user's own conversation;
    a viewer built for a device that cannot answer could never bind, so the
    honest answer is the sentence that names the device, the reason in words and
    the command that diagnoses the link — the SAME words the TUI's pick refuses
    with, asserted here against the composer itself rather than against a copy.
    """
    client, _root = remote_api
    row = _peer_row(reachable=False, unreachable_reason="connect_failed:ConnectionRefusedError")
    _answer_rows(monkeypatch, row)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "session_is_remote"
    assert detail["message"] == unreachable_peer_sentence(OTHER, row)
    assert "build-box" in detail["message"]
    assert "/network doctor" in detail["message"]
    assert (
        "connect_failed:ConnectionRefusedError" not in detail["message"]
    ), "the raw transport token reached the user"


@pytest.mark.asyncio
async def test_an_unknown_id_is_still_the_shared_404(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing about the pool's remote arm may change this answer."""
    client, _root = remote_api
    asked = _answer_rows(monkeypatch, None)

    response = await client.get(f"/v1/desktop/sessions/{'f' * 12}")
    assert response.status_code == 404, response.text
    assert asked == ["f" * 12], "the id was not even asked about"


@pytest.mark.asyncio
async def test_a_local_session_never_asks_the_peer_projection(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The zero-cost half: a directory this device holds answers from the disk.

    A conversation that moved HOME still has a local directory while the peer's
    cached listing can name it for one TTL, so the order (directory FIRST) is
    also what keeps "where does this live" answered from the durable fact.
    """
    client, root = remote_api
    _seed_local(root, MINE)
    asked = _answer_rows(monkeypatch, _peer_row(id=MINE))

    response = await client.get(f"/v1/desktop/sessions/{MINE}")
    assert response.status_code == 200, response.text
    assert asked == [], "a local read paid a peer lookup"


# ---------------------------------------------------------------------------
# The open path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_peers_row_opens_through_the_one_viewer_seam(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route answers 200 and the bridge is built by the SHARED seam."""
    client, _root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    built = _remote_facade(monkeypatch, _root)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}")
    assert response.status_code == 200, response.text
    assert built, "the desktop built its own facade instead of using remote_open"
    assert built[0]["session_id"] == OTHER
    # THE DESKTOP SURFACE IS DECLARED, and it has to be: the owner's runtime
    # advertises the desktop watch capability against a surface that says
    # "desktop", and the renderer's presence lease rides that same negotiation.
    assert built[0]["surface"] == "desktop"
    # THE ROW THE POOL RESOLVED is handed to the seam, so the placement and the
    # seed come from ONE read rather than from a second lookup that could
    # disagree with it.
    assert built[0]["row"].id == OTHER
    # NOTHING IS WRITTEN ON THIS DEVICE for a conversation it does not hold.
    assert not (_root / "sessions" / OTHER).exists()


@pytest.mark.asyncio
async def test_the_transcript_is_read_off_the_wire_not_off_this_disk(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The peer's rows reach ``/history`` in the contract's entry shape.

    The wire carries MESSAGES, so the projection into an entry is asserted here:
    the id from the row, the ``kind`` the durable encoder writes, and the content
    itself — which is what makes a remote page indistinguishable to the renderer
    from a local one.
    """
    client, _root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    _remote_facade(monkeypatch, _root)

    rows = [Message.user("hello from the peer"), Message.assistant("hello back")]

    def fake_history(self: AttachedSession) -> list[Any]:
        return list(rows)

    monkeypatch.setattr(AttachedSession, "history", fake_history)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/history")
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert page["cursor_missing"] is False
    assert [entry["id"] for entry in page["entries"]] == [row.id for row in rows]
    assert [entry["payload"]["kind"] for entry in page["entries"]] == ["message", "message"]
    assert "hello from the peer" in json.dumps(page["entries"])
    # ONE STAMP PER PAGE, and it is THIS device's clock: the wire carries no entry
    # time (see the method), so a page is dated when it is served rather than
    # pretending to know when the user sent it.
    assert len({entry["ts"] for entry in page["entries"]}) == 1
    assert page["entries"][0]["ts"] > 0


@pytest.mark.asyncio
async def test_a_page_that_runs_dry_asks_the_peer_for_older_rows(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``has_more`` must not mean "the window ended" while the peer holds more."""
    client, _root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    _remote_facade(monkeypatch, _root)

    newest = Message.user("newest")
    older = Message.user("older")
    loaded = [newest]

    def fake_history(self: AttachedSession) -> list[Any]:
        # WHAT THE FACADE HOLDS, which ``load_older_display_page`` grows in place:
        # the same contract the real method has (it inserts at the front of the
        # loaded window and returns the page it added).
        return list(loaded)

    asked: list[str] = []

    async def fake_older(self: AttachedSession) -> list[Any]:
        asked.append("older")
        loaded[:0] = [older]
        return [older]

    monkeypatch.setattr(AttachedSession, "history", fake_history)
    monkeypatch.setattr(AttachedSession, "load_older_display_page", fake_older)
    # The facade reports a token while it still has rows to fetch; only then is
    # asking the peer the right move.
    monkeypatch.setattr(
        AttachedSession, "history_before_token", property(lambda self: "tok"), raising=True
    )

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/history")
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert asked == ["older"], "the peer was never asked for the rows above the window"
    assert [entry["id"] for entry in page["entries"]] == [older.id, newest.id]
    assert page["has_more"] is True, "a page with rows still behind it claimed to be the end"


@pytest.mark.asyncio
async def test_a_peer_completion_is_read_from_the_peer_not_from_a_local_store(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """This device's ``attention.db`` is empty-by-construction for an id it lacks.

    Publishing that empty read as the conversation's receipts claims "nothing
    unseen here" about a completion that arrived on the peer -- the exact claim a
    notification is built on -- so the owner's own state is what the snapshot
    reports, and the local row for the same id is not even consulted.
    """
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    _remote_facade(monkeypatch, root)

    import uuid as _uuid

    from local_operator.session.attention import AttentionStore

    local = AttentionStore(root / "attention.db")
    # A REAL token shape: the store validates it as the CANONICAL form of a
    # UUID, because the token is the runtime's own completion identity rather
    # than any opaque string.
    local_token = str(_uuid.uuid4())
    local.publish(f"session/{OTHER}", local_token, "entry-1", "complete")
    local_state = local.state(f"session/{OTHER}")
    assert local_state["unseen"] is True, f"the fixture's local row reads {local_state}"

    from local_operator.session.frontend_state import FrontendSessionState

    # THE WIRE'S ANSWER, deliberately different from the local row above: the
    # owner says the completion has been seen, and this is what must be published.
    wire_attention = {
        "conversation_id": f"session/{OTHER}",
        "completion_token": local_token,
        "anchor_id": "entry-1",
        "kind": "complete",
        "reason": "done",
        "cause": "",
        "unseen": False,
        "revision": [1, 1],
    }
    monkeypatch.setattr(
        AttachedSession,
        "frontend_state",
        property(
            lambda self: FrontendSessionState(
                session_id=OTHER, epoch="e1", attention=dict(wire_attention)
            )
        ),
        raising=True,
    )

    response = await client.get(f"/v1/desktop/sessions/{OTHER}")
    assert response.status_code == 200, response.text
    snapshot = response.json()["result"]
    published = snapshot["payload"]["frontend"]["snapshot"]["attention"]
    assert published["unseen"] is False, (
        "the desktop published this device's own receipt row about a peer's "
        f"conversation instead of the owner's answer: {published}"
    )
    assert published["conversation_id"] == f"session/{OTHER}"


# ---------------------------------------------------------------------------
# Attachments: a sentence, never a 500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_peers_attachment_names_the_device_instead_of_500ing(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes are on the peer's disk, and the answer says so."""
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    digest = "a" * 32

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/attachments/{digest}")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "attachment_on_peer"
    assert "build-box" in detail["message"], detail

    # AND THE ORDINARY CASES DO NOT MOVE: an unknown id is still the 404, and a
    # LOCAL conversation whose digest resolves nowhere is still the 404 the store's
    # own contract gives.
    _seed_local(root, MINE)
    missing = await client.get(f"/v1/desktop/sessions/{'f' * 12}/attachments/{digest}")
    assert missing.status_code == 404, missing.text
    local_miss = await client.get(f"/v1/desktop/sessions/{MINE}/attachments/{digest}")
    assert local_miss.status_code == 404, local_miss.text


@pytest.mark.asyncio
async def test_a_peer_attachment_this_device_holds_is_still_served(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation that moved here keeps its images: the digest IS the content.

    The store is content-addressed, so "this device has a copy" is answered by
    the store itself rather than by where the conversation currently lives.
    """
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    import base64

    payload = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 64).decode()
    ref = AttachmentStore(root / ATTACHMENTS_DIRNAME).put(payload, "image/png")
    assert ref is not None

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/attachments/{ref.digest}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")


# ---------------------------------------------------------------------------
# The bytes a peer-bound prompt sends must stay resolvable on THIS device
# ---------------------------------------------------------------------------


def _real_png(width: int, height: int) -> bytes:
    """A real, decodable PNG with compressible but structured content.

    REAL BYTES because the transform under test IS an image ingest: a
    hand-built ``b"\x89PNG..." + zeros`` payload is not decodable, so
    ``image_blocks`` drops it and a cell built on one would measure the drop
    rather than the bound. Compressible because the shapes that matter here are
    a few hundred pixels wide and must stay well under the store's floor and the
    route's own body bound.
    """
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + tag + data + zlib.crc32(tag + data).to_bytes(4, "big")

    rows = []
    for y in range(height):
        line = bytearray(b"\x00")
        for x in range(width):
            tile = ((x // 32) * 6 + (y // 32) * 3) % 256
            line += bytes((tile, (tile + 40) % 256, (tile + 90) % 256))
        rows.append(bytes(line))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00",
        )
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )


def _wire(raw: bytes, mime: str = "image/png") -> dict[str, str]:
    return {"data_b64": base64.b64encode(raw).decode("ascii"), "mime_type": mime}


@pytest.mark.asyncio
async def test_a_peer_bound_prompt_sends_and_stages_what_the_owner_will_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transform's contract, at the object the route calls — and its four gates.

    WHY IT MUST EXIST. The owner's journal row references
    ``{"attachment": <digest>}``, the owner's runtime is another process with its
    own config dir, and this device's only read resolves ``<root>/attachments``.
    Without a copy here the picture the user had just sent answered ``409
    attachment_on_peer`` while the prompt reported ``admitted`` — the drop the
    sweep over two real relays in ``tests/unit/network/test_remote_viewer.py``
    reproduces row by row.

    WHY IT RETURNS WHAT IT SENT. The owner journals the digest of what it
    RECEIVES, so a mirror of the raw wire bytes only resolves while the owner
    keeps those bytes verbatim — and it does not, above ``IMAGE_INGEST_MAX_EDGE``
    (1024 px), over ``IMAGE_MAX_BYTES``, or with an EXIF ``Orientation`` to bake
    in. Round 1 measured that end to end; this cell pins the shape that fixes
    it: the returned payload is what the owner's own ingest makes of the input,
    and the digest staged is the digest OF THAT.

    THE GATES, each because it is a way to pay for nothing: a LOCAL conversation
    comes back untouched and stages nothing (this device's runtime runs that very
    ingest and writes that very store); an image the owner would DISCARD is
    dropped rather than sent (the alternative is the same loss one layer down);
    a payload under the transcript's externalise floor stages nothing, because
    such a row keeps its bytes inline; and ``AttachmentStore.put``'s silence is
    respected — a failed write must never refuse a prompt the owner would admit.
    """
    from local_operator.imaging import sniff_image
    from local_operator.server.utils.desktop_sessions import (
        DesktopSessionBridge,
        DesktopSessions,
    )

    root = tmp_path.resolve()
    row = _peer_row(id=OTHER)
    remote = DesktopSessionBridge(root, OTHER, "", remote_row=row)
    local = DesktopSessionBridge(root, MINE, "")
    store = root / ATTACHMENTS_DIRNAME

    oversize = _wire(_real_png(1025, 640))
    original = oversize["data_b64"]
    assert sniff_image(base64.b64decode(original)).width == 1025  # type: ignore[union-attr]

    # A LOCAL CONVERSATION IS UNTOUCHED, byte for byte, and pays no write.
    assert await local.prepare_peer_images([oversize]) == [oversize]
    assert not store.exists(), "a LOCAL conversation staged an image it already owns"

    # AN IMAGE-LESS PROMPT WRITES NOTHING.
    assert await remote.prepare_peer_images([]) == []
    assert not store.exists(), "an image-less prompt wrote a store"

    # THE BOUND: what comes back is the OWNER'S ingest output, not the wire bytes.
    prepared = await remote.prepare_peer_images([oversize])
    assert len(prepared) == 1
    settled_bytes = base64.b64decode(prepared[0]["data_b64"])
    assert settled_bytes != base64.b64decode(original), (
        "an image over IMAGE_INGEST_MAX_EDGE came back verbatim, so the owner's ingest "
        "will rewrite it and the digest staged here cannot be the one it journals"
    )
    assert sniff_image(settled_bytes).width <= 1024  # type: ignore[union-attr]
    digest = hashlib.sha256(settled_bytes).hexdigest()[:32]

    # AND IT IS WHAT IS STAGED: exactly one blob, under the digest of the returned
    # bytes. The old shape of the defect is a blob under a name no row references.
    assert sorted(p.name for p in store.glob("*.bin")) == [f"{digest}.bin"]

    # THE READ THIS EXISTS FOR, through the pool's own door (the one the route
    # opens): the bytes served are the bytes the owner's row will name, and the
    # peer is never consulted for them.
    _answer_rows(monkeypatch, row)
    pool = DesktopSessions(root)
    served, served_mime = await pool.attachment(OTHER, digest)
    assert served == settled_bytes
    assert served_mime == prepared[0]["mime_type"]

    # AN IMAGE THE OWNER WOULD DISCARD IS NOT SENT, and stages nothing.
    junk = {
        "data_b64": base64.b64encode(b"not an image at all").decode("ascii"),
        "mime_type": "image/png",
    }
    before = sorted(p.name for p in store.glob("*.bin"))
    assert await remote.prepare_peer_images([junk]) == []
    assert sorted(p.name for p in store.glob("*.bin")) == before

    # AND THE FLOOR: a payload the owner would leave INLINE is returned (it must
    # still be sent) but not staged, because nothing would reference the blob.
    tiny = _wire(_real_png(16, 16))
    tiny_prepared = await remote.prepare_peer_images([tiny])
    assert tiny_prepared, "a sub-floor image was dropped instead of sent"
    tiny_digest = hashlib.sha256(base64.b64decode(tiny_prepared[0]["data_b64"])).hexdigest()[:32]
    assert not (store / f"{tiny_digest}.bin").exists()
    assert sorted(p.name for p in store.glob("*.bin")) == [f"{digest}.bin"]

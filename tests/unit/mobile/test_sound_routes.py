"""The phone's sound family on the wire: digest-keyed bytes and waveforms.

WHY THESE THREE ROUTES. A turn's generated sound is a supplement artifact on the
beat's own ``sounds[]`` (``{attachment, mime, title, duration_s}``), not a message
block, so ``api_session_attachment``'s ``(entry, kind, i)`` index cannot address it.
The digest IS the address, exactly as the component document route already says for
frames. These tests pin the facts a player AND the origin depend on:

* audio bytes answer a Range request (phone Safari refuses a full-body 200) with the
  sidecar's own ``Content-Type``, and every miss is a clean 404/400;
* ONLY ``audio/*`` is served (round-1 review B1): the store is shared with
  agent-authored ``text/html`` component blobs, so a component digest must never
  come back as a page at the relay's own origin;
* authorisation is THREE facts (round-1 review m1): the cookie, a durable local user
  conversation, AND a ``sounds[]`` entry in one of that conversation's
  ``supplement_v1`` rows naming the digest. A component reference is not a sound
  reference, and another session's sound is not this session's to serve;
* the Library waveform is the SAME service reader the desktop route uses, proxied
  through ``mobile/library.py``'s refusal mapping.

Every gate is asserted before login as well as after: a route that is reachable
unauthenticated is not a route this surface has.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.library import identity
from local_operator.library.index import Asset
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import Transcript

SESSION_ID = "soundsession"
OTHER_SESSION = "othersound"
LIB_SESSION = "4e92693767fa"
#: Enough real bytes to exercise the Range arithmetic; no decoder ever sees them
#: (the bytes route serves the store verbatim).
AUDIO = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(64))


def _client() -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _session_dir(session_id: str) -> Any:
    """A durable user conversation's directory (a transcript, no origin marker)."""
    from local_operator.harness.types import Message

    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True)
    asyncio.run(Transcript(directory).append_message(Message.assistant("a finished answer")))
    return directory


def _reference_sounds(directory: Any, *digests: str) -> None:
    """Append one settled ``supplement_v1`` row whose ``sounds[]`` names ``digests``.

    The row shape is the producer's own (``supplements/persistence.build_details``):
    ``{anchor, job, version, state, files, components, sounds}`` under ``details``.
    """
    sounds = [
        {"attachment": digest, "mime": "audio/mpeg", "title": "clip", "duration_s": 1.5}
        for digest in digests
    ]
    asyncio.run(
        Transcript(directory).append_custom(
            "supplement_v1",
            {
                "anchor": "b" * 32,
                "job": "3f9c1a7e5b20",
                "version": 1,
                "state": "done",
                "files": [],
                "components": [],
                "sounds": sounds,
            },
        )
    )


def _reference_component(directory: Any, digest: str) -> None:
    """A supplement row whose only attachment reference is a COMPONENT (not a sound)."""
    asyncio.run(
        Transcript(directory).append_custom(
            "supplement_v1",
            {
                "anchor": "c" * 32,
                "job": "9a1b2c3d4e5f",
                "version": 1,
                "state": "done",
                "files": [],
                "components": [{"attachment": digest, "title": "card", "source": "x"}],
                "sounds": [],
            },
        )
    )


@pytest.fixture()
def session() -> dict[str, Any]:
    """A durable user conversation plus a referenced, digest-addressable sound."""
    directory = _session_dir(SESSION_ID)
    store = AttachmentStore()
    ref = store.put_bytes(AUDIO, "audio/mpeg")
    assert ref is not None
    _reference_sounds(directory, ref.digest)
    return {"store": store, "digest": ref.digest, "directory": directory}


def test_sound_family_requires_authentication(session: dict[str, Any]) -> None:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")))
    for path in ("sound", "waveform"):
        denied = client.get(
            f"/api/sessions/{SESSION_ID}/{path}", params={"digest": session["digest"]}
        )
        assert denied.status_code == 401
        # The family policy rides refusals too (qa round 1, Q1): an error body
        # must not become a sniffing surface either.
        assert denied.headers["x-content-type-options"] == "nosniff"
        assert denied.headers["cross-origin-resource-policy"] == "same-origin"


def test_sound_bytes_full_and_ranged(session: dict[str, Any]) -> None:
    client = _client()
    url = f"/api/sessions/{SESSION_ID}/sound"
    params = {"digest": session["digest"]}
    size = len(AUDIO)

    full = client.get(url, params=params)
    assert full.status_code == 200
    assert full.content == AUDIO
    assert full.headers["content-type"] == "audio/mpeg"
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["cache-control"] == "public, max-age=31536000, immutable"
    # Round-1 review B1: audio bytes never sniff into another type, and another
    # origin's page cannot embed them.
    assert full.headers["x-content-type-options"] == "nosniff"
    assert full.headers["cross-origin-resource-policy"] == "same-origin"

    ranged = client.get(url, params=params, headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert ranged.content == AUDIO[:10]
    assert ranged.headers["content-range"] == f"bytes 0-9/{size}"
    assert ranged.headers["x-content-type-options"] == "nosniff"

    tail = client.get(url, params=params, headers={"Range": "bytes=-4"})
    assert tail.status_code == 206
    assert tail.content == AUDIO[-4:]

    open_end = client.get(url, params=params, headers={"Range": "bytes=4-"})
    assert open_end.status_code == 206
    assert open_end.content == AUDIO[4:]

    refused = client.get(url, params=params, headers={"Range": f"bytes={size}-"})
    assert refused.status_code == 416
    assert refused.headers["content-range"] == f"bytes */{size}"

    # A Range the server does not honour is ignored into a full 200 (the
    # attachment route's own contract, now shared through one helper).
    multi = client.get(url, params=params, headers={"Range": "bytes=0-1,5-6"})
    assert multi.status_code == 200
    assert multi.content == AUDIO


def test_sound_bytes_misses_and_bad_shapes(session: dict[str, Any]) -> None:
    client = _client()
    url = f"/api/sessions/{SESSION_ID}/sound"

    # Absent digest, unroutable digest shape, absent bytes, unknown session: each
    # refusal is distinguishable, so a player can tell "malformed" from "gone".
    for params, status in (
        ({"digest": "0" * 32}, 404),
        ({"digest": "NOTHEX"}, 400),
        ({}, 400),
    ):
        refused = client.get(url, params=params)
        assert refused.status_code == status
        # Every answer from the family carries the pair (qa round 1, Q1).
        assert refused.headers["x-content-type-options"] == "nosniff"
        assert refused.headers["cross-origin-resource-policy"] == "same-origin"
    unknown_session = client.get(
        "/api/sessions/not-a-session/sound", params={"digest": session["digest"]}
    )
    assert unknown_session.status_code == 404
    assert unknown_session.headers["x-content-type-options"] == "nosniff"


def test_sound_bytes_require_a_reference_in_this_session(session: dict[str, Any]) -> None:
    """The third authorisation fact (round-1 review m1): a digest alone is not enough."""
    client = _client()
    store = session["store"]
    unreferenced = store.put_bytes(AUDIO + b"unreferenced", "audio/mpeg")
    assert unreferenced is not None

    # Bytes exist, but no supplement row of THIS session names the digest.
    assert (
        client.get(
            f"/api/sessions/{SESSION_ID}/sound", params={"digest": unreferenced.digest}
        ).status_code
        == 404
    )

    # A reference from ANOTHER conversation does not travel.
    other = _session_dir(OTHER_SESSION)
    _reference_sounds(other, unreferenced.digest)
    assert (
        client.get(
            f"/api/sessions/{SESSION_ID}/sound", params={"digest": unreferenced.digest}
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/sessions/{OTHER_SESSION}/sound", params={"digest": unreferenced.digest}
        ).status_code
        == 200
    )


def test_component_reference_does_not_authorise_audio(session: dict[str, Any]) -> None:
    """A ``components[]`` attachment is not a ``sounds[]`` attachment.

    The precise reader exists because ``_journal_references_digest`` matches any
    attachment on a supplement row; that would let a component digest reach the
    audio family and, before the B1 mime gate, serve a document.
    """
    client = _client()
    store = session["store"]
    ref = store.put_bytes(AUDIO + b"component", "audio/mpeg")
    assert ref is not None
    _reference_component(session["directory"], ref.digest)

    assert (
        client.get(f"/api/sessions/{SESSION_ID}/sound", params={"digest": ref.digest}).status_code
        == 404
    )


def test_sound_bytes_refuse_non_audio_sidecars(session: dict[str, Any]) -> None:
    """B1: an agent-authored ``text/html`` blob must never come back as a page.

    The digest is deliberately REFERENCED as a sound so the refusal under test is
    the mime gate, not the reference gate.
    """
    client = _client()
    store = session["store"]
    page = b"<!doctype html><html><body><script>document.title='pwned'</script></body></html>"
    ref = store.put_bytes(page, "text/html")
    assert ref is not None
    _reference_sounds(session["directory"], ref.digest)

    response = client.get(f"/api/sessions/{SESSION_ID}/sound", params={"digest": ref.digest})
    assert response.status_code == 404
    assert "text/html" not in response.headers.get("content-type", "")
    assert b"<script>" not in response.content
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"


def test_sound_bytes_default_mime_when_sidecar_states_none(session: dict[str, Any]) -> None:
    """The sidecar's ``application/octet-stream`` is the sentinel for "states none".

    ``AttachmentStore.get_bytes`` defaults to that value, so the route maps it to
    ``audio/mpeg`` — the one default the brief allows, and only in that case.
    """
    client = _client()
    ref = session["store"].put_bytes(b"\x00\x01\x02\x03", "application/octet-stream")
    assert ref is not None
    _reference_sounds(session["directory"], ref.digest)

    response = client.get(f"/api/sessions/{SESSION_ID}/sound", params={"digest": ref.digest})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"


def test_session_waveform_sidecar_and_absence(session: dict[str, Any]) -> None:
    client = _client()
    url = f"/api/sessions/{SESSION_ID}/waveform"
    peaks = [index / 240 for index in range(240)]
    assert session["store"].put_peaks(session["digest"], peaks)

    response = client.get(url, params={"digest": session["digest"]})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"version": 1, "bars": 240, "peaks": peaks}

    # A digest with bytes but NO sidecar is a 404 — never a fabricated flat line.
    bare = session["store"].put_bytes(b"\x00\x01\x02\x03", "audio/mpeg")
    assert bare is not None
    _reference_sounds(session["directory"], bare.digest)
    absent = client.get(url, params={"digest": bare.digest})
    assert absent.status_code == 404
    assert absent.headers["cache-control"] == "no-store"
    assert absent.headers["x-content-type-options"] == "nosniff"

    assert client.get(url, params={"digest": "NOTHEX"}).status_code == 400
    assert (
        client.get(
            "/api/sessions/not-a-session/waveform", params={"digest": session["digest"]}
        ).status_code
        == 404
    )


def test_waveform_requires_the_same_reference(session: dict[str, Any]) -> None:
    """The peaks family shares the byte route's three facts, not just the cookie."""
    client = _client()
    store = session["store"]
    peaks = [0.25] * 240
    unreferenced = store.put_bytes(AUDIO + b"wave", "audio/mpeg")
    assert unreferenced is not None
    assert store.put_peaks(unreferenced.digest, peaks)

    assert (
        client.get(
            f"/api/sessions/{SESSION_ID}/waveform", params={"digest": unreferenced.digest}
        ).status_code
        == 404
    )
    _reference_sounds(session["directory"], unreferenced.digest)
    assert (
        client.get(
            f"/api/sessions/{SESSION_ID}/waveform", params={"digest": unreferenced.digest}
        ).status_code
        == 200
    )


@pytest.fixture()
def audio_assets(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Two digest-backed audio assets: one with a peaks sidecar, one without.

    Seeded through the DERIVED INDEX the service reads (``library/store.py``),
    exactly as ``tests/unit/mobile/test_library_routes.py`` does after the SQLite
    index landed (#2793): the scanner normally writes these rows, and a fixture
    that only monkeypatched the old JSON catalog is invisible to
    ``_catalog_asset`` now. A digest-backed asset's key is ``a:<digest>``, so the
    aggregate carries ``digest`` with no path.
    """
    from local_operator.library.metadata import MetadataStore
    from local_operator.library.store import LibraryStore

    store = AttachmentStore()
    with_peaks = store.put_bytes(AUDIO, "audio/mpeg")
    without_peaks = store.put_bytes(AUDIO + b"2", "audio/mpeg")
    assert with_peaks is not None and without_peaks is not None
    # A realistic 240-bar envelope: the sidecar validator rejects any other length.
    assert store.put_peaks(with_peaks.digest, [0.5] * 240)

    def asset(ref: Any, name: str) -> Asset:
        key = identity.digest_key(ref.digest)
        return Asset(
            key=key,
            id=identity.asset_id(key),
            digest=ref.digest,
            derived_name=name,
            kind="audio",
            category="audio",
            content_type="audio/mpeg",
            size_bytes=ref.bytes,
        )

    ready = asset(with_peaks, "chime.mp3")
    silent = asset(without_peaks, "mute.mp3")
    metadata = MetadataStore(config_dir())
    db = LibraryStore(config_dir())
    db.ensure_schema()
    rows: list[dict[str, Any]] = []
    for item in (ready, silent):
        metadata.ensure_asset(item.id, item.key, derived_name=item.derived_name)
        rows.append(
            {
                "asset_id": item.id,
                "key": item.key,
                "session_id": LIB_SESSION,
                "title": "library fixture",
                "origin": "generated",
                "tool": "generate_sound_effect",
                "first_ts": 1_700_000_000.0,
                "last_ts": 1_700_000_100.0,
                "first_seq": 1,
                "last_seq": 2,
                "touches": 1,
                "name": item.derived_name,
                "path": None,
                "kind": item.kind,
                "content_type": item.content_type,
                "size_bytes": item.size_bytes,
                "duration_s": 3.5,
            }
        )

    # A fixed-row HTTP contract: the background worker would reconcile this
    # synthetic session against a config with no journal for it and prune the
    # rows. Its own suite covers indexing.
    class _NoWorker:
        @staticmethod
        def enqueue_thumbnail(_asset_id: str, _width: int = 480) -> None:
            return None

    monkeypatch.setattr(
        "local_operator.library.service.ensure_started", lambda *a, **k: _NoWorker()
    )
    with db.write() as connection:
        db.replace_session_occurrences(connection, LIB_SESSION, rows)
        db.recompute_assets(connection, [item.id for item in (ready, silent)])
    return {"ready": ready, "silent": silent}


def test_library_waveform_proxies_the_service(audio_assets: dict[str, Any]) -> None:
    client = _client()
    response = client.get(f"/api/library/assets/{audio_assets['ready'].id}/waveform")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["version"] == 1
    assert payload["bars"] == 240
    assert len(payload["peaks"]) == 240


def test_library_waveform_refusals(audio_assets: dict[str, Any]) -> None:
    client = _client()

    # An id the catalog does not know is the module's ordinary 404.
    missing = client.get("/api/library/assets/nope/waveform")
    assert missing.status_code == 404
    assert missing.json()["code"] == "library_asset_not_found"
    assert missing.headers["cache-control"] == "no-store"

    # An audio asset whose sidecar is absent degrades to the service's own code —
    # still a 404, never an empty payload a player would draw as silence.
    silent = client.get(f"/api/library/assets/{audio_assets['silent'].id}/waveform")
    assert silent.status_code == 404
    assert silent.json()["code"] == "content_missing"


def test_library_waveform_unavailable_is_typed_503(
    audio_assets: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable derived index is the 503 every sibling payload maps.

    Convergence review: after #2793 ``_catalog_asset`` reads the SQLite catalog
    and can raise ``LibraryUnavailable``; this branch-only payload must map it
    like ``thumb_ref`` does instead of letting it escape as a 500.
    ``corrupt=False`` keeps the test on the transient branch — no recovery
    rebuild is triggered.
    """
    from local_operator.library.store import IndexUnavailable, LibraryStore

    def boom(self: object, *args: object, **kwargs: object) -> Any:
        raise IndexUnavailable("database is busy", corrupt=False)

    monkeypatch.setattr(LibraryStore, "asset", boom)
    busy = _client().get(f"/api/library/assets/{audio_assets['ready'].id}/waveform")
    assert busy.status_code == 503
    assert busy.json()["code"] == "library_store_busy"
    assert busy.headers["cache-control"] == "no-store"

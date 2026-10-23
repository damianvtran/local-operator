"""The phone's sound family on the wire: digest-keyed bytes and waveforms.

WHY THESE THREE ROUTES. A turn's generated sound is a supplement artifact on the
beat's own ``sounds[]`` (``{attachment, mime, title, duration_s}``), not a message
block, so ``api_session_attachment``'s ``(entry, kind, i)`` index cannot address it.
The digest IS the address, exactly as the component document route already says for
frames. These tests pin the three facts a player depends on:

* audio bytes answer a Range request (phone Safari refuses a full-body 200) with the
  sidecar's own ``Content-Type``, and every miss is a clean 404/400;
* the peaks sidecar travels verbatim — and an absent one is a 404, NEVER fabricated
  zeros, because the phone draws its honest flat scrubber from the absence;
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
from local_operator.library import index as library_index
from local_operator.library.index import Asset, Catalog
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import Transcript

SESSION_ID = "soundsession"
#: Enough real bytes to exercise the Range arithmetic; no decoder ever sees them
#: (the bytes route serves the store verbatim).
AUDIO = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(64))


def _client() -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


@pytest.fixture()
def session() -> dict[str, Any]:
    """A durable user conversation plus a digest-addressable audio artifact.

    The directory is written the way a real runtime writes one (a transcript via
    ``Transcript``), so ``_durable_user_session_dir`` — the route's session gate —
    resolves it without a stubbed predicate.
    """
    from local_operator.harness.types import Message

    directory = config_dir() / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    asyncio.run(Transcript(directory).append_message(Message.assistant("a finished answer")))

    store = AttachmentStore()
    ref = store.put_bytes(AUDIO, "audio/mpeg")
    assert ref is not None
    return {"store": store, "digest": ref.digest, "directory": directory}


def test_sound_family_requires_authentication(session: dict[str, Any]) -> None:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")))
    assert (
        client.get(f"/api/sessions/{SESSION_ID}/sound", params={"digest": session["digest"]})
        .status_code
        == 401
    )
    assert (
        client.get(f"/api/sessions/{SESSION_ID}/waveform", params={"digest": session["digest"]})
        .status_code
        == 401
    )


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

    ranged = client.get(url, params=params, headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert ranged.content == AUDIO[:10]
    assert ranged.headers["content-range"] == f"bytes 0-9/{size}"

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
    assert client.get(url, params={"digest": "0" * 32}).status_code == 404
    assert client.get(url, params={"digest": "NOTHEX"}).status_code == 400
    assert client.get(url, params={}).status_code == 400
    assert (
        client.get("/api/sessions/not-a-session/sound", params={"digest": session["digest"]})
        .status_code
        == 404
    )


def test_sound_bytes_default_mime_when_sidecar_states_none(session: dict[str, Any]) -> None:
    """The sidecar's ``application/octet-stream`` is the sentinel for "states none".

    ``AttachmentStore.get_bytes`` defaults to that value, so the route maps it to
    ``audio/mpeg`` — the one default the brief allows, and only in that case.
    """
    client = _client()
    ref = session["store"].put_bytes(b"\x00\x01\x02\x03", "application/octet-stream")
    assert ref is not None

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
    absent = client.get(url, params={"digest": bare.digest})
    assert absent.status_code == 404
    assert absent.headers["cache-control"] == "no-store"

    assert client.get(url, params={"digest": "NOTHEX"}).status_code == 400
    assert (
        client.get("/api/sessions/not-a-session/waveform", params={"digest": session["digest"]})
        .status_code
        == 404
    )


@pytest.fixture()
def audio_assets(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Two digest-backed audio assets: one with a peaks sidecar, one without.

    The catalog is injected at BOTH read entry points ``_catalog_asset`` uses
    (``catalog_for_read``, then the refresh), so the route's own resolution path is
    exercised rather than bypassed.
    """
    store = AttachmentStore()
    with_peaks = store.put_bytes(AUDIO, "audio/mpeg")
    without_peaks = store.put_bytes(AUDIO + b"2", "audio/mpeg")
    assert with_peaks is not None and without_peaks is not None
    # A realistic 240-bar envelope: the sidecar validator rejects any other length.
    assert store.put_peaks(with_peaks.digest, [0.5] * 240)

    def asset(ref: Any, name: str) -> Asset:
        key = f"artifact:{ref.digest}"
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
    catalog = Catalog(
        assets=[ready, silent],
        revision=1,
        state="ready",
        sessions_total=1,
        sessions_scanned=1,
        built_at=1_700_000_200.0,
    )
    monkeypatch.setattr(library_index, "catalog_for_read", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(library_index, "build_catalog", lambda *args, **kwargs: catalog)
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

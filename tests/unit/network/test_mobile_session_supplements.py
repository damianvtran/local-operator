"""Remote Highlights on mobile: supplement bytes through two real mesh peers.

Viewer relay A serves the phone API; owner relay B holds the conversation. Every
request below is a real HTTP call on A's mobile app, forwarded over the real
authenticated TCP link to B's ``net_session_history`` handler, which runs the SAME
resolver the owner's own phone route runs (``session/media.py``). Featured files
follow the operator's 2026-11-15 decision ("every file the Highlight names, with the
same rules as on that computer"): cwd/home/scratchpad, denylist, 10 MB cap, suffix
allowlist — with cwd taken only from the OWNER's own registry record.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.harness.types import Message
from local_operator.mobile import remote_sessions
from local_operator.mobile.remote_sessions import RemoteSessionError
from local_operator.network import relay, store
from local_operator.network.sync import SYNC_CHUNK_BYTES
from local_operator.resume import SessionRow
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import Transcript
from tests.unit.network.test_relay_e2e import _dialable_devices, devices  # noqa: F401

SESSION = "mobile-peer-highlights"
JOB = "3f9c1a7e5b20"
COMPONENT = '<p class="lead">Q3</p>\n<div id="chart"></div>\n'
AUDIO = b"ID3\x04" + bytes(range(256)) * 40
#: Larger than one mesh chunk, so the Range/streaming path crosses a chunk boundary.
PAD_BYTES = bytes(range(256)) * (SYNC_CHUNK_BYTES // 256 + 3)

#: ``files[]`` order of the seeded Highlight; the HTTP ``i=`` index of each case.
F_PAD, F_CWD, F_ENV, F_BIG, F_SVG, F_HTML, F_CWD_REL = range(7)


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("devices")


def _seed(owner: relay.RelayServer, work: Path, monkeypatch: Any) -> dict[str, Any]:
    """One owner conversation with a Highlight naming a component, files and a sound."""
    directory = owner.root / "sessions" / SESSION
    directory.mkdir(parents=True)
    attachments = AttachmentStore(owner.root / "attachments")
    component = attachments.put_bytes(COMPONENT.encode(), "text/html")
    sound = attachments.put_bytes(AUDIO, "audio/mpeg")
    stray = attachments.put_bytes(b"never referenced", "text/plain")
    assert component and sound and stray
    pad = directory / "scratchpad"
    pad.mkdir()
    (pad / "chart.csv").write_bytes(PAD_BYTES)
    (pad / "diagram.svg").write_text("<svg onload='x()'/>")
    (pad / "page.html").write_text("<script>x()</script>")
    work.mkdir()
    (work / "report.md").write_text("# owner cwd report\n")
    (work / ".env").write_text("TOKEN=synthetic-not-a-secret\n")
    (work / "huge.txt").write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    files = [
        {"path": str(pad / "chart.csv")},
        {"path": str(work / "report.md")},
        {"path": str(work / ".env")},
        {"path": str(work / "huge.txt")},
        {"path": str(pad / "diagram.svg")},
        {"path": str(pad / "page.html")},
        {"path": "report.md"},
    ]
    details = {
        "anchor": "a" * 32,
        "job": JOB,
        "version": 1,
        "state": "done",
        "files": files,
        "components": [{"attachment": component.digest, "title": "t", "source": "s"}],
        "sounds": [{"attachment": sound.digest, "mime": "audio/mpeg", "title": "clip"}],
    }

    async def write() -> None:
        transcript = Transcript(directory)
        try:
            await transcript.append_message(Message.assistant("a finished answer"))
            await transcript.append_custom("supplement_v1", details)
        finally:
            transcript.close()

    # A writer runs under its own owner's environment, as the owner process does.
    with monkeypatch.context() as owner_env:
        owner_env.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(owner.root))
        asyncio.run(write())
    return {
        "component": component.digest,
        "sound": sound.digest,
        "stray": stray.digest,
        "pad": pad,
        "work": work,
    }


def _owner_runtime(owner: relay.RelayServer, cwd: Path) -> int:
    """Publish a LIVE discovery record for the session in the OWNER's registry.

    The record's pid is this test process (alive), so ``registry.scan`` classifies
    it non-stale and the owner reads ``cwd`` from it — the only cwd source it has.
    """
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    record = SessionRecord(
        pid=os.getpid(),
        kind="tui",
        session_id=SESSION,
        conversation_name="synthetic",
        cwd=str(cwd),
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )
    registry.publish(record, owner.root)
    return record.pid


def _client(viewer: relay.RelayServer, monkeypatch: Any) -> Any:
    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(viewer.root))
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    return client


def _file(client: Any, index: Any, **kwargs: Any) -> Any:
    return client.get(f"/api/sessions/{SESSION}/supplements/{JOB}/file?i={index}", **kwargs)


def test_remote_highlights_over_real_mesh(peer_pair: Any, monkeypatch: Any, tmp_path: Path) -> None:
    from local_operator.supplements.document import assemble_stored_component

    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    seeded = _seed(server_b, tmp_path / "work", monkeypatch)
    client = _client(server_a, monkeypatch)

    caps = client.get("/api/sessions").json()["capabilities"]["remote_sessions"]
    assert caps["supplements"] is True
    assert caps["supplement_kinds"] == ["document", "file", "sound"]

    # Document: the VIEWER assembles the owner's raw component; JSON, never text/html.
    document = client.get(f"/api/sessions/{SESSION}/supplements/{seeded['component']}/document")
    assert document.status_code == 200, document.text
    assert document.headers["content-type"].startswith("application/json")
    assert document.headers["cache-control"] == "no-store"
    assert document.json()["html"] == assemble_stored_component(COMPONENT)

    # Scratchpad file: full 200 across chunks, then 206 windows.
    full = _file(client, F_PAD)
    assert full.status_code == 200 and full.content == PAD_BYTES
    assert full.headers["content-type"].startswith("text/csv")
    assert full.headers["cache-control"] == "no-store"
    assert full.headers["x-content-type-options"] == "nosniff"
    assert full.headers["content-disposition"] == "attachment"
    window = _file(client, F_PAD, headers={"Range": f"bytes={SYNC_CHUNK_BYTES - 3}-"})
    assert window.status_code == 206 and window.content == PAD_BYTES[SYNC_CHUNK_BYTES - 3 :]
    assert window.headers["content-range"] == (
        f"bytes {SYNC_CHUNK_BYTES - 3}-{len(PAD_BYTES) - 1}/{len(PAD_BYTES)}"
    )

    # Sound: audio bytes with Range.
    sound_route = f"/api/sessions/{SESSION}/sound"
    sound = client.get(sound_route, params={"digest": seeded["sound"]})
    assert sound.status_code == 200 and sound.content == AUDIO
    assert sound.headers["content-type"] == "audio/mpeg"
    assert sound.headers["cross-origin-resource-policy"] == "same-origin"
    part = client.get(
        sound_route, params={"digest": seeded["sound"]}, headers={"Range": "bytes=0-3"}
    )
    assert part.status_code == 206 and part.content == AUDIO[:4]

    # No live owner runtime: absolute paths still serve; a cwd-relative one is 409.
    no_cwd = _file(client, F_CWD_REL)
    assert no_cwd.status_code == 409
    assert no_cwd.json()["code"] == "remote_supplement_needs_running"
    assert no_cwd.headers["cache-control"] == "no-store"

    # With the owner's runtime (its OWN record carries the cwd): Option B serves cwd
    # files under the same rules as the owner's own phone.
    pid = _owner_runtime(server_b, seeded["work"])
    try:
        relative = _file(client, F_CWD_REL)
        assert relative.status_code == 200 and relative.content == b"# owner cwd report\n"
        cwd_file = _file(client, F_CWD)
        assert cwd_file.status_code == 200 and cwd_file.content == b"# owner cwd report\n"
        assert cwd_file.headers["content-type"].startswith("text/markdown")
        # Same denylist: a credential file in cwd is the honest 404.
        assert _file(client, F_ENV).status_code == 404
        too_big = _file(client, F_BIG)
        assert too_big.status_code == 413 and too_big.json()["code"] == "remote_too_large"
    finally:
        from local_operator.session.runtime import registry

        registry.unpublish(pid, server_b.root)

    # Refusals.
    assert _file(client, F_SVG).status_code == 404
    assert _file(client, F_HTML).status_code == 404
    assert _file(client, 99).status_code == 404
    assert _file(client, "x").status_code == 400
    assert _file(client, -1).status_code == 404
    assert client.get(f"/api/sessions/{SESSION}/supplements/..%2Fetc/file?i=0").status_code == 404
    assert (
        client.get(f"/api/sessions/{SESSION}/supplements/{seeded['stray']}/document").status_code
        == 404
    )
    assert client.get(sound_route, params={"digest": seeded["stray"]}).status_code == 404
    # A component digest never reaches the audio family, nor a sound the document one.
    assert client.get(sound_route, params={"digest": seeded["component"]}).status_code == 404
    assert client.get(sound_route, params={"digest": "../x"}).status_code == 400
    # No read engaged an owner runtime or materialised the session on the viewer.
    assert not (server_a.root / "sessions" / SESSION).exists()


def test_file_mutated_between_chunks_is_refused(
    peer_pair: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    seeded = _seed(server_b, tmp_path / "work", monkeypatch)
    row = SessionRow(id=SESSION, mtime=0.0, name="s", owner_device=server_b.identity.device_id)
    metadata, data = remote_sessions.supplement_chunk(
        server_a.root, row, kind="file", ref=JOB, index=F_PAD, offset=0, length=16
    )
    assert data == PAD_BYTES[:16] and metadata["total"] == len(PAD_BYTES)
    (seeded["pad"] / "chart.csv").write_bytes(b"rewritten" + PAD_BYTES)
    with pytest.raises(RemoteSessionError) as caught:
        remote_sessions.supplement_chunk(
            server_a.root,
            row,
            kind="file",
            ref=JOB,
            index=F_PAD,
            offset=16,
            length=16,
            expected_media_id=metadata["media_id"],
        )
    assert caught.value.code == "remote_media_changed" and caught.value.status == 409


def test_old_owner_is_explicit_unsupported(
    peer_pair: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    seeded = _seed(server_b, tmp_path / "work", monkeypatch)
    client = _client(server_a, monkeypatch)
    link = server_a._ensure_link(server_b.identity.device_id)  # noqa: SLF001
    assert link is not None
    # Control: the same request is served while the capability is negotiated.
    assert _file(client, F_PAD).status_code == 200
    monkeypatch.setattr(
        link,
        "capabilities",
        tuple(cap for cap in link.capabilities if cap != "mobile-session-supplements-v1"),
    )
    for response in (
        _file(client, F_PAD),
        client.get(f"/api/sessions/{SESSION}/supplements/{seeded['component']}/document"),
        client.get(f"/api/sessions/{SESSION}/sound", params={"digest": seeded["sound"]}),
    ):
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "remote_unsupported"
    # Ordinary remote reads still work on that link: only the new purpose is gated.
    assert client.get(f"/api/sessions/{SESSION}/history").status_code == 200


def test_owner_side_refuses_unnegotiated_supplement_read(
    peer_pair: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    """The OWNER's own check: a viewer that skipped its check still gets ``unsupported``."""
    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    _seed(server_b, tmp_path / "work", monkeypatch)
    # Handlers are bound at construction, so wrap B's own table entry: the OWNER
    # sees a link that never negotiated the feature while A's link still has it.
    real = server_b._handlers["net_session_history"]  # noqa: SLF001

    class _Link:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.capabilities = tuple(
                cap for cap in inner.capabilities if cap != "mobile-session-supplements-v1"
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    monkeypatch.setitem(
        server_b._handlers,  # noqa: SLF001
        "net_session_history",
        lambda link, frame: real(_Link(link), frame),
    )
    row = SessionRow(id=SESSION, mtime=0.0, name="s", owner_device=server_b.identity.device_id)
    with pytest.raises(RemoteSessionError) as caught:
        remote_sessions.supplement_chunk(
            server_a.root, row, kind="file", ref=JOB, index=F_PAD, offset=0, length=8
        )
    assert caught.value.code == "remote_unsupported" and caught.value.status == 409


def test_offline_owner_is_503(peer_pair: Any, monkeypatch: Any, tmp_path: Path) -> None:
    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    _seed(server_b, tmp_path / "work", monkeypatch)
    client = _client(server_a, monkeypatch)
    assert _file(client, F_PAD).status_code == 200
    server_b.stop()
    response = _file(client, F_PAD)
    assert response.status_code == 503, response.text
    assert response.json()["remote"] is True


def test_view_grant_revoked_between_chunks(
    peer_pair: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    """Every chunk is its own frame: a revoke stops the NEXT one (#4057's S1 rule)."""
    from tests.unit.network.test_relay_e2e import _pair_settled

    server_a, server_b, _host, _port = peer_pair
    server_b.bind()
    server_b.bind_control()
    server_b.start()
    record, _host, _port = _pair_settled(
        peer_pair,
        monkeypatch,
        role="read",
        settings=relay.NetworkSettings(port=server_b.settings.port, listen_address="127.0.0.1"),
    )
    _seed(server_a, tmp_path / "work", monkeypatch)
    row = SessionRow(id=SESSION, mtime=0.0, name="s", owner_device=server_a.identity.device_id)
    metadata, data = remote_sessions.supplement_chunk(
        server_b.root, row, kind="file", ref=JOB, index=F_PAD, offset=0, length=16
    )
    assert data == PAD_BYTES[:16]
    with store.mutate(record.network_id, server_a.root) as current:
        relay.set_member_capabilities(
            current, device_id=server_b.identity.device_id, revoke=("view",)
        )
        store.save(current, server_a.root)
    with pytest.raises(RemoteSessionError) as caught:
        remote_sessions.supplement_chunk(
            server_b.root,
            row,
            kind="file",
            ref=JOB,
            index=F_PAD,
            offset=16,
            length=16,
            expected_media_id=metadata["media_id"],
        )
    assert caught.value.code == "remote_forbidden" and caught.value.status == 403


# --- HTTP contract over a REAL socket: refusals between chunks are statuses -------


@contextlib.contextmanager
def _socket_client(viewer: relay.RelayServer, monkeypatch: Any) -> Iterator[httpx.Client]:
    """The viewer's phone app on a real uvicorn socket, logged in.

    A TestClient cannot show the round-1 defect (R1/QA-1): it is the HTTP framing —
    a ``200`` already on the wire before a later chunk is refused — that matters, so
    these cases read real bytes off a real listener, as the phone does.
    """
    import threading

    import uvicorn

    from local_operator.mobile.daemon import MobileDaemon, build_app

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(viewer.root))
    app = build_app(MobileDaemon(port=0, password="test-only"))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(2000):
            if server.started:
                break
            time.sleep(0.01)
        assert server.started, "uvicorn did not start"
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
            client.post("/login", data={"password": "test-only"})
            yield client
    finally:
        server.should_exit = True
        thread.join(timeout=30)


def _after_second_chunk(monkeypatch: Any, act: Any) -> list[dict[str, Any]]:
    """Run ``act`` once the viewer has fetched bytes beyond the first-byte probe."""
    real = remote_sessions.async_supplement_chunk
    calls: list[dict[str, Any]] = []

    async def wrapped(*args: Any, **fields: Any) -> Any:
        calls.append(fields)
        reply = await real(*args, **fields)
        if len(calls) == 2:
            await asyncio.to_thread(act)
        return reply

    monkeypatch.setattr(remote_sessions, "async_supplement_chunk", wrapped)
    return calls


def test_http_file_rewritten_between_chunks_is_409_not_a_short_200(
    peer_pair: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    seeded = _seed(server_b, tmp_path / "work", monkeypatch)
    with _socket_client(server_a, monkeypatch) as client:
        control = client.get(f"/api/sessions/{SESSION}/supplements/{JOB}/file?i={F_PAD}")
        assert control.status_code == 200 and control.content == PAD_BYTES
        calls = _after_second_chunk(
            monkeypatch,
            lambda: (seeded["pad"] / "chart.csv").write_bytes(b"rewritten" + PAD_BYTES),
        )
        response = client.get(f"/api/sessions/{SESSION}/supplements/{JOB}/file?i={F_PAD}")
    assert len(calls) == 3  # probe, first full chunk, the refused one
    assert response.status_code == 409, response.status_code
    assert response.json()["code"] == "remote_media_changed"
    assert PAD_BYTES[:64] not in response.content
    assert response.headers["cache-control"] == "no-store"


def test_http_grant_revoked_between_chunks_is_403_not_a_short_200(
    peer_pair: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    from tests.unit.network.test_relay_e2e import _pair_settled

    server_a, server_b, _host, _port = peer_pair
    server_b.bind()
    server_b.bind_control()
    server_b.start()
    record, _host, _port = _pair_settled(
        peer_pair,
        monkeypatch,
        role="read",
        settings=relay.NetworkSettings(port=server_b.settings.port, listen_address="127.0.0.1"),
    )
    _seed(server_a, tmp_path / "work", monkeypatch)

    def revoke() -> None:
        with store.mutate(record.network_id, server_a.root) as current:
            relay.set_member_capabilities(
                current, device_id=server_b.identity.device_id, revoke=("view",)
            )
            store.save(current, server_a.root)

    calls = _after_second_chunk(monkeypatch, revoke)
    with _socket_client(server_b, monkeypatch) as client:
        response = client.get(f"/api/sessions/{SESSION}/supplements/{JOB}/file?i={F_PAD}")
    assert len(calls) == 3
    assert response.status_code == 403, response.status_code
    assert response.json()["code"] == "remote_forbidden"
    assert PAD_BYTES[:64] not in response.content


# --- unit: both peers' validate_read, the viewer MIME re-check --------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "file", "ref": "../etc/passwd", "index": 0},
        {"kind": "file", "ref": "a" * 32, "index": 0},
        {"kind": "document", "ref": "a" * 12, "index": 0},
        {"kind": "sound", "ref": "a" * 32, "index": 1},
        {"kind": "image", "ref": "a" * 32, "index": 0},
        {"kind": "file", "ref": JOB, "index": -1},
        {"kind": "file", "ref": JOB, "index": 0, "path": "/etc/passwd"},
        {"kind": "file", "ref": JOB, "index": 0, "entry_id": "x"},
        {"kind": "file", "ref": JOB, "index": 0, "length": SYNC_CHUNK_BYTES + 1},
        {"kind": "file", "ref": JOB, "index": 0, "expected_media_id": "zz"},
    ],
)
def test_validate_read_refuses_bad_supplement_frames(fields: dict[str, Any]) -> None:
    from local_operator.mobile.remote_sessions import validate_read
    from local_operator.network.types import MeshRefusal

    frame = {
        "op": "net_session_history",
        "session_id": SESSION,
        "mobile_session_read": "supplement",
        "offset": 0,
        "length": 8,
        **fields,
    }
    with pytest.raises(MeshRefusal) as caught:
        validate_read(frame)
    assert caught.value.code == "protocol_error"


def test_validate_read_admits_good_supplement_frames() -> None:
    from local_operator.mobile.remote_sessions import validate_read

    for kind, ref, index in (("file", JOB, 3), ("document", "b" * 32, 0), ("sound", "c" * 32, 0)):
        validate_read(
            {
                "op": "net_session_history",
                "session_id": SESSION,
                "mobile_session_read": "supplement",
                "limit": 50,
                "kind": kind,
                "ref": ref,
                "index": index,
                "offset": 0,
                "length": SYNC_CHUNK_BYTES,
                "expected_media_id": "d" * 64,
            }
        )


def test_viewer_never_serves_a_peer_declared_html_type(monkeypatch: Any, tmp_path: Path) -> None:
    """A hostile owner declaring ``text/html`` gets octet-stream (file) or 404 (sound)."""
    import base64

    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    body = b"<script>alert(1)</script>"
    row = SessionRow(id=SESSION, mtime=0.0, name="s", owner_device="owner-device")

    async def hostile(_root: Any, _row: Any, purpose: str, **fields: Any) -> dict[str, Any]:
        assert purpose == "supplement"
        chunk = body[fields["offset"] : fields["offset"] + fields["length"]]
        return {
            "mobile_session_read": 1,
            "session_id": SESSION,
            "ref": fields["ref"],
            "index": fields["index"],
            "kind": fields["kind"],
            "media_id": "e" * 64,
            "mime": "text/html",
            "total": len(body),
            "offset": fields["offset"],
            "data_b64": base64.b64encode(chunk).decode(),
        }

    monkeypatch.setattr(remote_sessions, "resolve_owner", lambda *_a: row)
    monkeypatch.setattr(remote_sessions, "async_mesh_read", hostile)
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    served = _file(client, 0)
    assert served.status_code == 200 and served.content == body
    assert served.headers["content-type"] == "application/octet-stream"
    assert served.headers["content-disposition"] == "attachment"
    assert served.headers["x-content-type-options"] == "nosniff"
    sound = client.get(f"/api/sessions/{SESSION}/sound", params={"digest": "f" * 32})
    assert sound.status_code == 404


def test_supplement_refusal_codes_map_to_phone_statuses() -> None:
    from local_operator.mobile.remote_sessions import _read_detail

    for code, (public, status) in {
        "supplement_needs_running": ("remote_supplement_needs_running", 409),
        "too_large": ("remote_too_large", 413),
        "media_missing": ("remote_media_missing", 404),
        "media_changed": ("remote_media_changed", 409),
        "unsupported": ("remote_unsupported", 409),
    }.items():
        with pytest.raises(RemoteSessionError) as caught:
            _read_detail({"op": "error", "code": code}, session_id=SESSION)
        assert (caught.value.code, caught.value.status) == (public, status), code


def test_json_frame_shape_is_stable() -> None:
    """The capability literal and wire's tuple are one spelling."""
    from local_operator.mobile.remote_sessions import SUPPLEMENTS_CAPABILITY
    from local_operator.network import wire

    assert SUPPLEMENTS_CAPABILITY == wire.MOBILE_SESSION_SUPPLEMENTS_V1
    assert SUPPLEMENTS_CAPABILITY in wire.LINK_CAPABILITIES
    assert json.dumps(SUPPLEMENTS_CAPABILITY) == '"mobile-session-supplements-v1"'


def test_viewer_caps_a_hostile_declared_file_total(monkeypatch: Any, tmp_path: Path) -> None:
    """S-1: a peer declaring a featured file over 10 MB is 413 at the viewer, unread."""
    import base64

    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.session.media import SUPPLEMENT_FILE_MAX_BYTES

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    row = SessionRow(id=SESSION, mtime=0.0, name="s", owner_device="owner-device")
    offsets: list[int] = []

    async def hostile(_root: Any, _row: Any, purpose: str, **fields: Any) -> dict[str, Any]:
        offsets.append(fields["offset"])
        return {
            "mobile_session_read": 1,
            "session_id": SESSION,
            "ref": fields["ref"],
            "index": fields["index"],
            "kind": fields["kind"],
            "media_id": "e" * 64,
            "mime": "audio/mpeg" if fields["kind"] == "sound" else "text/plain",
            "total": SUPPLEMENT_FILE_MAX_BYTES + 1,
            "offset": fields["offset"],
            "data_b64": base64.b64encode(b"x" * fields["length"]).decode(),
        }

    monkeypatch.setattr(remote_sessions, "resolve_owner", lambda *_a: row)
    monkeypatch.setattr(remote_sessions, "async_mesh_read", hostile)
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    refused = _file(client, 0)
    assert refused.status_code == 413 and refused.json()["code"] == "remote_too_large"
    assert offsets == [0]  # only the metadata probe; no body was pulled
    # Sounds have no local cap, so the same declared total still serves a Range.
    sound = client.get(
        f"/api/sessions/{SESSION}/sound",
        params={"digest": "f" * 32},
        headers={"Range": "bytes=0-3"},
    )
    assert sound.status_code == 206 and sound.content == b"xxxx"

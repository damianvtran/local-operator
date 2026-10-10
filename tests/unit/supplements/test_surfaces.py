"""Capability advertisement, `/v1/capabilities`, and the 404 route stubs (lane C0)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_operator.harness.types import Message
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.server.features import feature_flags
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.supplements.contract import (
    SUPPLEMENTS_CAPABILITY,
    SUPPLEMENTS_READ_OP,
)
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn

DIGEST = "04c2d29b140086cc637ad19ef34012c5"


# --- GET /v1/capabilities ---------------------------------------------------------------


def test_the_static_feature_flag_is_published() -> None:
    assert feature_flags()["supplements"] == 1


# --- the owner record's capability list ----------------------------------------------------


class _WithoutReadOp(ServingSessionHandle):
    """The C0 case a real handle can no longer be: one that PREDATES the read op.

    Lane C1b implements ``supplements_for`` on the production handle, so "an owner whose
    handle has no read op" has to be built. A property that raises ``AttributeError`` is the
    one spelling that makes ``hasattr`` answer False -- the exact question the owner record
    asks -- without changing the class under test or patching the server.
    """

    @property
    def supplements_for(self):  # type: ignore[override]
        raise AttributeError(SUPPLEMENTS_READ_OP)


async def _server(tmp_path: Path, *, with_read_op: bool) -> RuntimeServer:
    directory = tmp_path / "sessions" / "supplements-capability"
    await seed_transcript(directory, [Message.user("row")])
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    holder = ServingSessionHandle if with_read_op else _WithoutReadOp
    handle = holder(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    if with_read_op:
        # what the engine lane (C1) adds for real; here only its PRESENCE is under test
        setattr(handle, SUPPLEMENTS_READ_OP, lambda anchors: {})
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    return server


@pytest.mark.asyncio
async def test_an_owner_whose_handle_has_no_read_op_does_not_advertise(tmp_path) -> None:
    """Advertising what the handle cannot honour is worse than omitting it."""
    server = await _server(tmp_path, with_read_op=False)
    try:
        assert SUPPLEMENTS_CAPABILITY not in server._record.capabilities
        assert "display-history-window-v1" in server._record.capabilities
    finally:
        server.close()


@pytest.mark.asyncio
async def test_an_owner_whose_handle_implements_the_read_op_advertises(tmp_path) -> None:
    server = await _server(tmp_path, with_read_op=True)
    try:
        assert SUPPLEMENTS_CAPABILITY in server._record.capabilities
    finally:
        server.close()


async def _attach(server: RuntimeServer, **declare: Any) -> dict[str, Any]:
    """Attach speaking the wire protocol directly (so an old viewer can be reproduced)."""
    record = server._record
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
    auth = {"key": record.control_key, "client": "attach", "locality": "local", **declare}
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    try:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            assert line, "the owner closed before the welcome"
            frame = json.loads(line)
            if frame.get("op") in ("projection", "frontend_sync"):
                break
    finally:
        writer.close()
    # the connection the server built for us is the only one
    return {"conn": next(iter(server._clients.values())) if hasattr(server, "_clients") else None}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_has_op", "viewer_declares", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        # agent review R9: only the JSON boolean declares; truthy strings fail CLOSED
        (True, "false", False),
        (True, "1", False),
        (True, 1, False),
    ],
)
async def test_the_attach_gate_is_both_halves(
    tmp_path, owner_has_op: bool, viewer_declares: object, expected: bool
) -> None:
    server = await _server(tmp_path, with_read_op=owner_has_op)
    try:
        record = server._record
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        auth: dict[str, Any] = {
            "key": record.control_key,
            "client": "attach",
            "locality": "local",
            "frontend_state": True,
            "display_window": True,
        }
        if viewer_declares is not False:
            auth["supplements"] = viewer_declares
        writer.write(json.dumps(auth).encode() + b"\n")
        await writer.drain()
        try:
            for _ in range(50):
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                if json.loads(line).get("op") == "frontend_sync":
                    break
            connections = [
                c for c in vars(server).values() if isinstance(c, dict) for c in c.values()
            ]
            flags = [getattr(c, "supplements") for c in connections if hasattr(c, "supplements")]
            assert flags == [expected], flags
        finally:
            writer.close()
    finally:
        server.close()


# --- the relay routes (memo §2.7) --------------------------------------------------------


def _relay(password: str = "pw") -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password=password)), follow_redirects=False)
    assert client.post("/login", data={"password": password}).status_code in (200, 303)
    return client


@pytest.mark.parametrize(
    "path",
    [
        f"/api/sessions/s1/supplements/{DIGEST}/document",
        "/api/sessions/s1/supplements/3f9c1a7e5b20/file?i=0",
    ],
)
def test_the_relay_routes_answer_404_for_an_unknown_session(path: str) -> None:
    assert _relay().get(path).status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        f"/api/sessions/s1/supplements/{DIGEST}/document",
        "/api/sessions/s1/supplements/3f9c1a7e5b20/file?i=0",
    ],
)
def test_the_relay_routes_are_behind_the_gate(path: str) -> None:
    """Unauthenticated callers get the 401, learning nothing about the route's existence."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw")), follow_redirects=False)
    response = client.get(path)
    assert response.status_code == 401
    assert response.json()["error"] == "authentication required"
    # The family's response policy rides the refusals too (#2134 R4, D5): nosniff + no-store.
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


# --- the relay routes, against real store bytes and a real journal ----------------------

RELAY_SESSION = "ab12cd34ef56"
OTHER_SESSION = "cd34ef56ab12"
RELAY_JOB = "3f9c1a7e5b20"
COMPONENT = '<p class="lead">Q3</p>\n<div id="chart"></div>\n'


def _relay_config(tmp_path, monkeypatch):
    """A redirected config root; the daemon routes resolve through it per call.

    The root is a SUBDIRECTORY of ``tmp_path`` so fixture work directories can sit
    outside it: the §4.3 denylist denies everything under the live config dir (it
    holds credentials), and a work dir placed inside the root would be refused by
    that rule rather than by the one under test.
    """
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    return cfg


def _blob(cfg, blob: str = COMPONENT) -> str:
    from local_operator.session.attachments import AttachmentStore

    ref = AttachmentStore(root=cfg / "attachments").put_bytes(blob.encode("utf-8"), "text/html")
    assert ref is not None
    return ref.digest


def _journal(cfg, session_id: str, rows: list[dict[str, Any]]) -> None:
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _row(
    *, job: str = RELAY_JOB, files: list[dict[str, Any]] | None = None, digest: str | None = None
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "anchor": "a" * 32,
        "job": job,
        "version": 1,
        "state": "done",
        "files": files if files is not None else [],
        "components": (
            [
                {
                    "attachment": digest,
                    "title": "t",
                    "source": "s",
                    "mime": "text/html",
                    "height_hint": 320,
                }
            ]
            if digest
            else []
        ),
        "at": 1791000000.0,
    }
    return {
        "id": "0" * 32,
        "ts": 1791000000.0,
        "type": "custom",
        "payload": {"custom_type": "supplement_v1", "details": details},
    }


def _live(daemon: MobileDaemon, session_id: str, cwd: str, pid: int = 4242) -> None:
    from local_operator.session.runtime.types import SessionRecord

    record = SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name=f"session {session_id}",
        cwd=cwd,
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )
    daemon.table.entries[pid] = SessionEntry(record)


def _daemon_relay(password: str = "pw") -> tuple[TestClient, MobileDaemon]:
    daemon = MobileDaemon(port=0, password=password)
    client = TestClient(build_app(daemon), follow_redirects=False)
    assert client.post("/login", data={"password": password}).status_code in (200, 303)
    return client, daemon


def test_the_relay_document_route_serves_a_referenced_digest_for_an_ended_session(
    tmp_path, monkeypatch
) -> None:
    """The happy path with NO live entry: the route does not 404 like ``_entry_for_session``,
    which is exactly what ended-session history needs. Bytes, headers and bytes-equality
    with the one assembler are all pinned."""
    from local_operator.supplements.document import assemble_stored_component

    cfg = _relay_config(tmp_path, monkeypatch)
    digest = _blob(cfg)
    _journal(cfg, RELAY_SESSION, [_row(digest=digest)])
    client, _daemon = _daemon_relay()
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{digest}/document")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["html"] == assemble_stored_component(COMPONENT)


def test_the_relay_document_route_refuses_an_unreferenced_digest(tmp_path, monkeypatch) -> None:
    """The digest must be REFERENCED by this session's journal (memo §2.7 (c)): a blob held
    locally but named by no row is not servable through this route."""
    cfg = _relay_config(tmp_path, monkeypatch)
    digest = _blob(cfg)
    _journal(cfg, RELAY_SESSION, [_row()])  # a row, but it names nothing
    client, _daemon = _daemon_relay()
    assert (
        client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{digest}/document").status_code
        == 404
    )


def test_the_relay_document_route_refuses_another_session_digest(tmp_path, monkeypatch) -> None:
    cfg = _relay_config(tmp_path, monkeypatch)
    digest = _blob(cfg)
    _journal(cfg, RELAY_SESSION, [_row(digest=digest)])
    _journal(cfg, OTHER_SESSION, [_row()])
    client, _daemon = _daemon_relay()
    assert (
        client.get(f"/api/sessions/{OTHER_SESSION}/supplements/{digest}/document").status_code
        == 404
    ), "a digest is authorised by ITS session's journal, never by store membership"


def test_the_relay_document_route_is_digest_only(tmp_path, monkeypatch) -> None:
    """No path parameter exists, a smuggled one changes nothing, and a non-digest
    segment resolves to nothing."""
    cfg = _relay_config(tmp_path, monkeypatch)
    digest = _blob(cfg)
    _journal(cfg, RELAY_SESSION, [_row(digest=digest)])
    client, _daemon = _daemon_relay()
    clean = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{digest}/document")
    smuggled = client.get(
        f"/api/sessions/{RELAY_SESSION}/supplements/{digest}/document?path=/etc/passwd"
    )
    assert clean.status_code == smuggled.status_code == 200
    assert clean.json()["html"] == smuggled.json()["html"]
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/not-a-digest/document")
    assert response.status_code == 404
    assert client.get(
        f"/api/sessions/{RELAY_SESSION}/supplements/{'../' + digest}/document"
    ).status_code in (404, 400)


def test_the_relay_document_route_authorises_by_supplement_rows_only(tmp_path, monkeypatch) -> None:
    """§2.7's letter (round-1 review R4): the reference must be in a ``supplement_v1`` row.

    Sync's byte grammar also matches ``"attachment"`` references on OTHER rows --
    message media carries the same key today -- and a future text-mime producer must not
    become servable through this door. A digest named only by a message row does not
    authorise, even though the blob resolves in this store.
    """
    cfg = _relay_config(tmp_path, monkeypatch)
    digest = _blob(cfg)
    _journal(
        cfg,
        RELAY_SESSION,
        [
            {
                "id": "m" * 32,
                "ts": 1791000000.0,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "blocks": [{"type": "image", "attachment": digest, "mime_type": "image/png"}],
                },
            }
        ],
    )
    client, _daemon = _daemon_relay()
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{digest}/document")
    assert response.status_code == 404, "a non-supplement reference must not authorise this door"


def test_the_relay_document_route_carries_the_policy_on_errors(tmp_path, monkeypatch) -> None:
    """D5's pair rides the refusals too (#2134 R4): the 404 carries nosniff + no-store."""
    cfg = _relay_config(tmp_path, monkeypatch)
    digest = _blob(cfg)
    _journal(cfg, RELAY_SESSION, [_row()])  # a row, but it names nothing
    client, _daemon = _daemon_relay()
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{digest}/document")
    assert response.status_code == 404
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def _file_row(path: str) -> dict[str, Any]:
    return _row(
        files=[
            {
                "path": path,
                "name": path.rsplit("/", 1)[-1],
                "kind": "markdown",
                "size_bytes": 10,
                "mtime": 1791000000.0,
                "why": "written by write",
            }
        ]
    )


def test_the_relay_file_route_serves_the_recorded_index(tmp_path, monkeypatch) -> None:
    cfg = _relay_config(tmp_path, monkeypatch)
    work = tmp_path / "Work"
    (work / "reports").mkdir(parents=True)
    (work / "reports" / "q3.md").write_text("# Q3\nthe report\n", encoding="utf-8")
    _journal(cfg, RELAY_SESSION, [_file_row("reports/q3.md")])
    client, daemon = _daemon_relay()
    _live(daemon, RELAY_SESSION, str(work))
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file?i=0")
    assert response.status_code == 200
    assert response.content == b"# Q3\nthe report\n"
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == "attachment"


def test_the_relay_file_route_refuses_everything_but_recorded_indices(
    tmp_path, monkeypatch
) -> None:
    cfg = _relay_config(tmp_path, monkeypatch)
    work = tmp_path / "Work"
    (work / "reports").mkdir(parents=True)
    (work / "reports" / "q3.md").write_text("x", encoding="utf-8")
    _journal(cfg, RELAY_SESSION, [_file_row("reports/q3.md")])
    client, daemon = _daemon_relay()
    _live(daemon, RELAY_SESSION, str(work))
    base = f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file"
    assert client.get(f"{base}?i=1").status_code == 404
    assert client.get(f"{base}?i=-1").status_code == 404
    assert client.get(f"{base}?i=abc").status_code == 400
    assert client.get(base).status_code == 400
    # A traversal-shaped job id never reaches a lookup.
    assert (
        client.get(f"/api/sessions/{RELAY_SESSION}/supplements/..%2Fetc/file?i=0").status_code
        == 404
    )


def test_the_relay_file_route_carries_the_policy_on_errors(tmp_path, monkeypatch) -> None:
    """D5's pair rides the 400/404 refusals too (#2134 R4)."""
    cfg = _relay_config(tmp_path, monkeypatch)
    work = tmp_path / "Work"
    (work / "reports").mkdir(parents=True)
    (work / "reports" / "q3.md").write_text("x", encoding="utf-8")
    _journal(cfg, RELAY_SESSION, [_file_row("reports/q3.md")])
    client, daemon = _daemon_relay()
    _live(daemon, RELAY_SESSION, str(work))
    base = f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file"
    for url, status in ((f"{base}?i=abc", 400), (f"{base}?i=7", 404)):
        response = client.get(url)
        assert response.status_code == status
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"


def test_the_relay_file_route_reapplies_the_denylist(tmp_path, monkeypatch) -> None:
    """A row is data, not authority: even a tampered row naming a credential file is
    refused by the denylist re-applied at read time (memo §4.3)."""
    cfg = _relay_config(tmp_path, monkeypatch)
    work = tmp_path / "Work"
    work.mkdir(parents=True)
    (work / ".env").write_text("SECRET=1\n", encoding="utf-8")
    _journal(cfg, RELAY_SESSION, [_file_row(".env")])
    client, daemon = _daemon_relay()
    _live(daemon, RELAY_SESSION, str(work))
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file?i=0")
    assert response.status_code == 404


def test_the_relay_file_route_refuses_a_relative_path_without_a_live_record(
    tmp_path, monkeypatch
) -> None:
    cfg = _relay_config(tmp_path, monkeypatch)
    _journal(cfg, RELAY_SESSION, [_file_row("reports/q3.md")])
    client, _daemon = _daemon_relay()
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file?i=0")
    assert response.status_code == 409, "no cwd to vouch for -> refuse, never guess"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_the_relay_file_route_serves_a_home_relative_path_without_a_live_record(
    tmp_path, monkeypatch
) -> None:
    cfg = _relay_config(tmp_path, monkeypatch)
    fake_home = tmp_path / "home"
    (fake_home / "docs").mkdir(parents=True)
    (fake_home / "docs" / "n.md").write_text("home notes", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    _journal(cfg, RELAY_SESSION, [_file_row("~/docs/n.md")])
    client, _daemon = _daemon_relay()
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file?i=0")
    assert response.status_code == 200 and response.content == b"home notes"


def test_the_relay_file_route_serves_html_bodies_as_text(tmp_path, monkeypatch) -> None:
    """S15: a ``.txt`` whose body opens with ``<script>`` is served as ``text/plain``
    under ``nosniff`` -- the suffix allowlist decides the type, never the bytes."""
    cfg = _relay_config(tmp_path, monkeypatch)
    work = tmp_path / "Work"
    work.mkdir(parents=True)
    (work / "evil.txt").write_text("<!doctype html><script>alert(1)</script>", encoding="utf-8")
    _journal(cfg, RELAY_SESSION, [_file_row("evil.txt")])
    client, daemon = _daemon_relay()
    _live(daemon, RELAY_SESSION, str(work))
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file?i=0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"] == "attachment"


def test_the_relay_file_route_refuses_scriptable_suffixes(tmp_path, monkeypatch) -> None:
    cfg = _relay_config(tmp_path, monkeypatch)
    work = tmp_path / "Work"
    work.mkdir(parents=True)
    for name in ("page.html", "pic.svg", "app.command"):
        (work / name).write_text("<svg/>", encoding="utf-8")
    client, daemon = _daemon_relay()
    _live(daemon, RELAY_SESSION, str(work))
    for index, name in enumerate(("page.html", "pic.svg", "app.command")):
        _journal(cfg, RELAY_SESSION, [_file_row(name)])
        response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file?i=0")
        assert response.status_code == 404, name


def test_the_relay_file_route_caps_the_file_size(tmp_path, monkeypatch) -> None:
    cfg = _relay_config(tmp_path, monkeypatch)
    work = tmp_path / "Work"
    work.mkdir(parents=True)
    with (work / "big.md").open("wb") as handle:
        handle.truncate(10 * 1024 * 1024 + 1)
    _journal(cfg, RELAY_SESSION, [_file_row("big.md")])
    client, daemon = _daemon_relay()
    _live(daemon, RELAY_SESSION, str(work))
    response = client.get(f"/api/sessions/{RELAY_SESSION}/supplements/{RELAY_JOB}/file?i=0")
    assert response.status_code == 413
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "op",
    ["supplement_cancel", "supplement_steer", "supplement_restart", "supplement_dismiss"],
)
def test_the_relay_passes_the_supplement_ops_through(tmp_path, monkeypatch, op: str) -> None:
    """The four control ops ride the generic command route: validated, then forwarded --
    an unknown session is the 409 "session not connected", not a 422 "unknown op" (the
    runtime's own dispatch is the op authority, memo §2.7's ops table)."""
    _relay_config(tmp_path, monkeypatch)
    client, _daemon = _daemon_relay()
    response = client.post(
        f"/api/sessions/{RELAY_SESSION}/command",
        json={"op": op, "anchor": "a" * 32, "job": "b" * 12},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "session not connected"

"""Attachment proxy routes through actual loopback HTTP, with an isolated hub fixture.

Like ``test_desktop_team_bridge_scheduled.py``: only the credential store and
the hub's domain are fixtures. Desktop bearer checks, routing, id/query
validation, the bounded upload read, OAuth resolution, HTTP transport, refusal
mapping and response headers all run their production code. This proves the
proxy, not the hub's Mongo store or the renderer.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from PIL import Image

from local_operator.server.dependencies import get_config_manager
from local_operator.server.routes import desktop_team_bridge as routes
from local_operator.team_bridge import client as bridge
from tests.unit.team_bridge.test_client import FakeStore

CID = "bcv_" + "b" * 32
MID = "bcmsg_" + "c" * 16
AID = "batt_0f8fad5b-d9cb-469f-a165-70867728950e"
BASE = f"/v1/desktop/team-bridge/conversations/{CID}"


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 9), (10, 200, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


@pytest.fixture
def proxy(tmp_path, monkeypatch):
    calls: list[tuple[str, str, Any, str | None, str | None]] = []
    state: dict[str, Any] = {"serve_type": "image/png", "serve": PNG, "error": None}

    class Hub(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass  # bytes and bearer never reach fixture logs either

        def _reply(self, code: int, payload: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def dispatch(self):
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            ctype = self.headers.get("content-type")
            body: Any = raw
            if ctype == "application/json" and raw:
                body = json.loads(raw)
            calls.append((self.command, self.path, body, self.headers.get("authorization"), ctype))
            if state["error"]:
                code, error = state["error"]
                out = json.dumps({"code": error, "error": "Fixture refusal"}).encode()
                return self._reply(code, out, "application/json")
            if self.command == "POST" and self.path.split("?")[0].endswith("/attachments"):
                result = {
                    "attachment": {
                        "id": AID,
                        "mime": "image/png",
                        "size": len(raw),
                        "width": 12,
                        "height": 9,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "state": "staged",
                        "expires_at": "2030-04-08T12:00:00Z",
                    }
                }
                out = json.dumps({"msg": "Attachment staged", "result": result}).encode()
                return self._reply(201, out, "application/json")
            if self.command == "DELETE":
                self.send_response(204)
                self.end_headers()
                return None
            if self.command == "GET" and "/attachments/" in self.path:
                return self._reply(200, state["serve"], state["serve_type"])
            out = json.dumps({"msg": "Message sent", "result": {"id": MID, "seq": 1}}).encode()
            return self._reply(200, out, "application/json")

        do_GET = do_POST = do_PATCH = do_DELETE = dispatch

    hub = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    hub_thread = threading.Thread(target=hub.serve_forever, daemon=True)
    hub_thread.start()
    store = FakeStore()
    real_client = bridge.TeamBridgeClient
    monkeypatch.setattr(bridge, "org_oauth_destination_allowed", lambda _base: True)
    monkeypatch.setattr(
        bridge,
        "TeamBridgeClient",
        lambda config_dir: real_client(
            config_dir, base_url=f"http://127.0.0.1:{hub.server_port}/v1", store=store
        ),
    )
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "desktop-token")
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_config_manager] = lambda: SimpleNamespace(config_dir=tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "loopback proxy did not start"
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"authorization": "Bearer desktop-token"},
        ) as client:
            yield client, calls, state
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        hub.shutdown()
        hub.server_close()
        hub_thread.join(10)
        assert not thread.is_alive()
        assert not hub_thread.is_alive()


def test_route_constants_match_the_client() -> None:
    assert routes._ATTACHMENT_MAX_BYTES == bridge.ATTACHMENT_MAX_BYTES
    assert routes._ATTACHMENT_SERVABLE == set(bridge.ATTACHMENT_MIME_TYPES)


def test_upload_streams_raw_bytes_and_returns_201_descriptor(proxy):
    client, calls, _ = proxy
    response = client.post(
        f"{BASE}/attachments?tenant_id=org-a",
        content=PNG,
        headers={"content-type": "image/svg+xml"},  # ignored: the hub sniffs
    )
    assert response.status_code == 201, response.text
    attachment = response.json()["result"]["attachment"]
    assert attachment["id"] == AID and attachment["state"] == "staged"
    method, path, body, auth, ctype = calls[-1]
    assert (method, path) == (
        "POST",
        f"/v1/messaging/conversations/{CID}/attachments?tenant_id=org-a",
    )
    assert body == PNG and ctype == "application/octet-stream" and auth == "Bearer token-1"


def test_upload_refusals_happen_before_any_hub_call(proxy):
    client, calls, _ = proxy
    too_big = client.post(f"{BASE}/attachments", content=b"\0" * (bridge.ATTACHMENT_MAX_BYTES + 1))
    assert too_big.status_code == 413
    assert too_big.json()["detail"]["code"] == "attachment_too_large"
    empty = client.post(f"{BASE}/attachments", content=b"")
    assert empty.status_code == 422
    scoped = client.post(f"{BASE}/attachments?scope=account", content=PNG)
    assert scoped.status_code == 422
    dup = client.post(f"{BASE}/attachments?tenant_id=a&tenant_id=b", content=PNG)
    assert dup.status_code == 422
    bad_cid = client.post(
        "/v1/desktop/team-bridge/conversations/bcv_..%2F/attachments", content=PNG
    )
    assert bad_cid.status_code in (404, 422)
    assert calls == []


def test_hub_refusals_pass_through_with_their_status(proxy):
    client, _calls, state = proxy
    state["error"] = (415, "attachment_type_unsupported")
    response = client.post(f"{BASE}/attachments", content=b"<svg/>")
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "attachment_type_unsupported"
    state["error"] = (404, "attachment_not_found")
    response = client.get(f"{BASE}/messages/{MID}/attachments/{AID}")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "attachment_not_found"


def test_delete_is_204_and_validates_the_id(proxy):
    client, calls, _ = proxy
    response = client.delete(f"{BASE}/attachments/{AID}?tenant_id=org-a")
    assert response.status_code == 204 and response.content == b""
    assert calls[-1][:2] == (
        "DELETE",
        f"/v1/messaging/conversations/{CID}/attachments/{AID}?tenant_id=org-a",
    )
    before = len(calls)
    assert client.delete(f"{BASE}/attachments/batt_x%2F..").status_code in (404, 422)
    assert client.delete(f"{BASE}/attachments/not_an_id").status_code == 422
    assert len(calls) == before


def test_get_serves_bytes_with_the_closed_type_and_safe_headers(proxy):
    client, calls, _ = proxy
    response = client.get(f"{BASE}/messages/{MID}/attachments/{AID}?tenant_id=org-a")
    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"
    # The human route never names an agent endpoint or purpose upstream.
    assert calls[-1][1] == (
        f"/v1/messaging/conversations/{CID}/messages/{MID}/attachments/{AID}?tenant_id=org-a"
    )


@pytest.mark.parametrize("served", ["image/svg+xml", "text/html", "image/gif", ""])
def test_get_refuses_any_served_type_outside_png_jpeg(proxy, served):
    client, _calls, state = proxy
    state["serve_type"] = served or "application/octet-stream"
    state["serve"] = b"<svg onload=alert(1)>"
    response = client.get(f"{BASE}/messages/{MID}/attachments/{AID}")
    assert response.status_code == 502
    assert "svg" not in response.text


def test_get_refuses_agent_query_names(proxy):
    client, calls, _ = proxy
    for query in ("endpoint_id=bep_1", "purpose=context", "scope=account"):
        assert client.get(f"{BASE}/messages/{MID}/attachments/{AID}?{query}").status_code == 422
    assert calls == []


def test_send_forwards_attachment_ids_and_allows_image_only(proxy):
    client, calls, _ = proxy
    response = client.post(
        f"{BASE}/messages", json={"client_ref": "r1", "body": "", "attachment_ids": [AID]}
    )
    assert response.status_code == 200, response.text
    assert calls[-1][2] == {"client_ref": "r1", "body": "", "attachment_ids": [AID]}
    text_only = client.post(f"{BASE}/messages", json={"client_ref": "r2", "body": "hi"})
    assert text_only.status_code == 200
    assert calls[-1][2] == {"client_ref": "r2", "body": "hi"}  # unchanged wire


@pytest.mark.parametrize(
    "payload",
    [
        {"client_ref": "r", "body": ""},
        {"client_ref": "r", "body": "x", "attachment_ids": []},
        {"client_ref": "r", "body": "x", "attachment_ids": [AID, AID]},
        {"client_ref": "r", "body": "x", "attachment_ids": ["../../status"]},
        {"client_ref": "r", "body": "x", "attachment_ids": [AID] * 5},
    ],
)
def test_send_refuses_bad_attachment_shapes_before_io(proxy, payload):
    client, calls, _ = proxy
    response = client.post(f"{BASE}/messages", json=payload)
    # Input masking of this 422 is the production app's handler (app.py,
    # ``desktop_validation_error``), which this bare router app does not mount.
    assert response.status_code == 422
    assert calls == []


def test_scheduled_create_and_patch_carry_attachment_ids(proxy):
    client, calls, _ = proxy
    create = {
        "tenant_id": "org-a",
        "conversation_id": CID,
        "client_ref": "s1",
        "body": "",
        "send_at": "2030-04-05T12:30:00Z",
        "attachment_ids": [AID],
    }
    assert client.post("/v1/desktop/team-bridge/scheduled", json=create).status_code == 200
    assert calls[-1][2] == create
    sid = "bsch_" + "a" * 32
    patch = {"attachment_ids": []}
    response = client.patch(f"/v1/desktop/team-bridge/scheduled/{sid}?tenant_id=org-a", json=patch)
    assert response.status_code == 200, response.text
    assert calls[-1][2] == {"attachment_ids": []}
    before = len(calls)
    no_content = {**create, "attachment_ids": None}
    assert client.post("/v1/desktop/team-bridge/scheduled", json=no_content).status_code == 422
    cleared = client.patch(
        f"/v1/desktop/team-bridge/scheduled/{sid}?tenant_id=org-a",
        json={"body": "", "attachment_ids": []},
    )
    assert cleared.status_code == 422
    assert len(calls) == before


def test_scheduled_create_never_forwards_an_explicit_null_attachment_ids(proxy):
    """R1-04: an old hub refuses the unknown key even with a null value."""

    client, calls, _ = proxy
    create = {
        "tenant_id": "org-a",
        "conversation_id": CID,
        "client_ref": "s-null",
        "body": "hi",
        "send_at": "2030-04-05T12:30:00Z",
        "attachment_ids": None,
    }
    assert client.post("/v1/desktop/team-bridge/scheduled", json=create).status_code == 200
    assert "attachment_ids" not in calls[-1][2]


# -- access-log redaction (QA #1314 Q2) ------------------------------------------


def _access_record(path: str, method: str = "GET") -> logging.LogRecord:
    # The exact shape uvicorn's access logger emits (h11_impl / httptools_impl).
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", method, path, "1.1", 200),
        None,
    )


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        (
            "GET",
            f"/v1/desktop/team-bridge/conversations/{CID}/messages/{MID}/attachments/{AID}"
            "?tenant_id=org-a",
            "/v1/desktop/team-bridge/conversations/:conversationid/messages/:messageid"
            "/attachments/:attachmentid?tenant_id=org-a",
        ),
        (
            "POST",
            f"/v1/desktop/team-bridge/conversations/{CID}/attachments",
            "/v1/desktop/team-bridge/conversations/:conversationid/attachments",
        ),
        (
            "DELETE",
            f"/v1/desktop/team-bridge/conversations/{CID}/attachments/{AID}?tenant_id=org-a",
            "/v1/desktop/team-bridge/conversations/:conversationid/attachments/:attachmentid"
            "?tenant_id=org-a",
        ),
    ],
)
def test_access_log_line_carries_no_attachment_message_or_conversation_ids(
    method, path, expected
) -> None:
    from local_operator.server.routes import desktop_team_bridge as routes

    access = logging.getLogger("uvicorn.access")
    assert routes._ACCESS_LOG_REDACTION in access.filters  # installed on import
    record = _access_record(path, method)
    # The redaction filter is the record rewriter under test; other filters a
    # test session may attach are not, and need not be ``logging.Filter``s.
    assert routes._ACCESS_LOG_REDACTION.filter(record)
    line = record.getMessage()
    assert expected in line and line.startswith(f'127.0.0.1:5000 - "{method} ')
    for prefix in ("batt_", "bcmsg_", "bcv_"):
        assert prefix not in line


def test_access_log_redaction_leaves_other_paths_and_odd_records_alone() -> None:
    from local_operator.server.routes import desktop_team_bridge as routes

    other = f"/v1/desktop/team-bridge/conversations/{CID}/messages"
    record = _access_record(other)
    assert routes._ACCESS_LOG_REDACTION.filter(record) and record.getMessage().count(other) == 1
    odd = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "plain", None, None)
    assert routes._ACCESS_LOG_REDACTION.filter(odd) and odd.getMessage() == "plain"


def test_e2e_access_log_of_real_upload_fetch_delete_has_no_ids_or_hashes(
    tmp_path, monkeypatch
) -> None:
    """NEGATIVE end to end (QA #1314 Q2): the PRODUCTION app, served by uvicorn
    with its DEFAULT logging config (the dictConfig ``lop serve`` gets), runs a
    real authenticated upload -> fetch -> delete against a loopback hub. The
    access log the runtime actually writes is captured from a handler on the
    ``uvicorn.access`` logger.

    Controls make it non-vacuous: every request returned its success status,
    each of the three appears in the captured log with its templated path, and
    the hub's reply really did carry an id and a sha256 the log could have
    leaked.
    """

    import re as _re

    RIG_DESKTOP_KEY = "e2e-log-desktop-key"  # noqa: N806 — a test literal, not a credential

    from local_operator.server.app import app as production_app

    hub_seen: list[str] = []
    sha_served: list[str] = []

    class Hub(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _reply(self, code: int, payload: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def dispatch(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            hub_seen.append(f"{self.command} {self.path.split('?')[0]}")
            if self.command == "POST":
                digest = hashlib.sha256(raw).hexdigest()
                sha_served.append(digest)
                attachment = {
                    "id": AID,
                    "mime": "image/png",
                    "size": len(raw),
                    "width": 12,
                    "height": 9,
                    "sha256": digest,
                    "state": "staged",
                    "expires_at": "2030-04-08T12:00:00Z",
                }
                body = json.dumps({"msg": "ok", "result": {"attachment": attachment}}).encode()
                return self._reply(201, body, "application/json")
            if self.command == "DELETE":
                self.send_response(204)
                self.end_headers()
                return None
            return self._reply(200, PNG, "image/png")

        do_GET = do_POST = do_DELETE = dispatch

    hub = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    hub_thread = threading.Thread(target=hub.serve_forever, daemon=True)
    hub_thread.start()
    store = FakeStore()
    real_client = bridge.TeamBridgeClient
    monkeypatch.setattr(bridge, "org_oauth_destination_allowed", lambda _base: True)
    monkeypatch.setattr(
        bridge,
        "TeamBridgeClient",
        lambda config_dir: real_client(
            config_dir, base_url=f"http://127.0.0.1:{hub.server_port}/v1", store=store
        ),
    )
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", RIG_DESKTOP_KEY)
    production_app.dependency_overrides[get_config_manager] = lambda: SimpleNamespace(
        config_dir=tmp_path
    )
    # Default log_config: uvicorn runs its own dictConfig here, exactly as the
    # launcher's ``uvicorn.Config(app, host=..., port=...)`` does.
    config = uvicorn.Config(production_app, lifespan="off")
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    access = logging.getLogger("uvicorn.access")
    access.addHandler(handler)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(config)
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    statuses: list[int] = []
    upload_reply = ""
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"authorization": f"Bearer {RIG_DESKTOP_KEY}"},
        ) as client:
            up = client.post(f"{BASE}/attachments?tenant_id=org-a", content=PNG)
            upload_reply = up.text
            statuses.append(up.status_code)
            got = client.get(f"{BASE}/messages/{MID}/attachments/{AID}?tenant_id=org-a")
            statuses.append(got.status_code)
            statuses.append(client.delete(f"{BASE}/attachments/{AID}?tenant_id=org-a").status_code)
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        hub.shutdown()
        hub.server_close()
        hub_thread.join(10)
        access.removeHandler(handler)
        production_app.dependency_overrides.pop(get_config_manager, None)
    log = captured.getvalue()

    # Controls: the real path ran, and there was something to leak.
    assert statuses == [201, 200, 204]
    assert len(hub_seen) == 3 and AID in upload_reply and sha_served[0] in upload_reply
    prefix = "/v1/desktop/team-bridge/conversations/:conversationid"
    for line in (
        f'"POST {prefix}/attachments?tenant_id=org-a HTTP/1.1" 201',
        f'"GET {prefix}/messages/:messageid/attachments/:attachmentid?tenant_id=org-a '
        'HTTP/1.1" 200',
        f'"DELETE {prefix}/attachments/:attachmentid?tenant_id=org-a HTTP/1.1" 204',
    ):
        assert line in log, f"request not logged as templated: {line}"

    # The negative: no ids, no sha256-shaped hex.
    for needle in ("batt_", "bcmsg_", "bcv_", AID, MID, CID):
        assert needle not in log
    assert _re.search(r"[0-9a-f]{64}", log) is None

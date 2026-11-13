"""Tunnel failure classification, status honesty and relayed-request logging.

Field report (macOS, lop 0.96.5; still present on 0.102.4): a Radient 503 was
reported as `authorization_refused` (blaming the login), `lop tunnel status`
said the gateway was "not answering … run lop tunnel install" while it was
listening, and no request log existed to diagnose "phone Not answering while
the Mac is healthy". Synthetic ids and an isolated config dir only.
"""

from __future__ import annotations

import pytest

from local_operator.tunnels import gateway
from local_operator.tunnels.errors import RequestFailed
from local_operator.tunnels.service import (
    authorization_failure_reason,
    classify_failure,
)


def _server_error(status: int) -> ValueError:
    return RequestFailed(f"Radient tunnel request failed (HTTP {status}).", status=status)


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_control_plane_5xx_is_a_transient_server_error_not_a_refusal(status):
    failure = _server_error(status)
    assert authorization_failure_reason(failure) == gateway.SERVER_UNAVAILABLE
    verdict = classify_failure(failure)
    assert verdict.kind == "transient"
    assert verdict.reason == gateway.SERVER_UNAVAILABLE
    # Chained the way `service.run` re-raises it still classifies the same.
    try:
        try:
            raise failure
        except ValueError as inner:
            raise ValueError("cold start failed") from inner
    except ValueError as outer:
        assert authorization_failure_reason(outer) == gateway.SERVER_UNAVAILABLE


@pytest.mark.parametrize("status", [401, 403, 409])
def test_a_4xx_still_reads_as_a_refusal(status):
    assert authorization_failure_reason(_server_error(status)) == gateway.REFUSED


def test_the_server_unavailable_reason_has_every_surface_entry():
    reason = gateway.SERVER_UNAVAILABLE
    for table in (
        gateway.RELAY_DETAIL,
        gateway.TERMINAL_DETAIL,
        gateway.TERMINAL_REMEDY,
        gateway.REASON_LABEL,
    ):
        assert reason in table
    # Not the login's fault: no sign-in remedy, and the copy says so.
    assert gateway.TERMINAL_REMEDY[reason] == "lop tunnel status"
    assert "login" not in gateway.REASON_LABEL[reason]
    assert "Sign in" not in gateway.RELAY_DETAIL[reason]


# --- item 3: `lop tunnel status` must not send a listening gateway to install --

import json  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

from local_operator.tunnels import config, report  # noqa: E402


class _GarbageHealth(BaseHTTPRequestHandler):
    """A gateway that accepts the connection but answers a non-JSON 200."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<html>upstream 503</html>")


@pytest.fixture
def listening_gateway():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarbageHealth)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


RETRY_LINE = (
    "2026-11-13T10:15:02-08:00 local_operator.tunnels.service: connector stopped "
    "reason=server_unavailable: Radient tunnel request failed (HTTP 503).\n"
)


@pytest.mark.asyncio
async def test_a_listening_gateway_that_errors_is_never_sent_to_install(
    tmp_path, monkeypatch, listening_gateway
):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    result = await report.probe({"gateway_port": listening_gateway})
    assert "lop tunnel install" not in result["detail"]
    assert "listening" in result["detail"]
    assert result["state"] != "connected"


@pytest.mark.asyncio
async def test_a_closed_port_with_a_recorded_server_error_names_that_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.directory().mkdir(parents=True, exist_ok=True)
    (config.directory() / "service.log").write_text(
        "2026-11-13T10:14:50-08:00 local_operator.tunnels.service: connector stopped "
        "reason=control_plane_unreachable: old\n" + RETRY_LINE
    )
    result = await report.probe({"gateway_port": _closed_port()})
    assert result["reason"] == gateway.SERVER_UNAVAILABLE
    assert "lop tunnel install" not in result["detail"]
    assert result["last_error"]["reason"] == gateway.SERVER_UNAVAILABLE
    assert result["last_error"]["at"] is not None


@pytest.mark.asyncio
async def test_a_closed_port_with_no_recorded_error_still_names_install(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    result = await report.probe({"gateway_port": _closed_port()})
    assert "lop tunnel install" in result["detail"]
    assert result["state"] == "stopped"


def test_the_log_tail_read_is_bounded_and_ignores_foreign_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.directory().mkdir(parents=True, exist_ok=True)
    noise = ("x" * 200 + "\n") * 2000  # ~400 KB of unrelated output after it
    (config.directory() / "service.log").write_text(RETRY_LINE + noise)
    # The retry line is outside the bounded tail, so it is not claimed.
    assert report.last_connector_failure() is None
    (config.directory() / "service.log").write_text(noise + RETRY_LINE)
    found = report.last_connector_failure()
    assert found is not None and found["reason"] == gateway.SERVER_UNAVAILABLE
    json.dumps(found)


# --- item 4: bounded, redacted relayed-request logging -------------------------

import httpx  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse, PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from local_operator.tunnels import request_log  # noqa: E402

SECRET = "tok_SYNTHETIC_SECRET_ab12"


def _app() -> Starlette:
    async def session(request):
        return PlainTextResponse("ok")

    async def refused(request):
        request_log.note_reason(request.scope, "authorization_refused")
        return JSONResponse({"error": "no"}, status_code=503)

    async def boom(request):
        raise RuntimeError("handler failed")

    async def healthz(request):
        return PlainTextResponse("ok")

    return Starlette(
        routes=[
            Route("/api/sessions/{session_id}", session),
            Route("/refused", refused),
            Route("/boom", boom),
            Route("/healthz", healthz),
        ]
    )


async def _client(log: request_log.RequestLog) -> httpx.AsyncClient:
    app = log.middleware(_app(), skip=frozenset({"/healthz"}))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")


@pytest.mark.asyncio
async def test_requests_are_logged_by_template_without_query_headers_or_cookies(tmp_path):
    path = tmp_path / "requests.log"
    log = request_log.RequestLog(path)
    async with await _client(log) as client:
        reply = await client.get(
            f"/api/sessions/session-{SECRET}?token={SECRET}&code={SECRET}",
            headers={"Authorization": f"Bearer {SECRET}", "Cookie": f"lop={SECRET}"},
        )
        refused = await client.get(f"/refused?pair={SECRET}")
        await client.get("/boom")
        await client.get("/healthz")
    log.close()
    assert reply.status_code == 200 and refused.status_code == 503
    written = path.read_text()
    assert SECRET not in written and SECRET not in json.dumps(list(log.ring))
    entries = [json.loads(line) for line in written.splitlines()]
    assert [e["path"] for e in entries] == ["/api/sessions/{session_id}", "/refused", "/boom"]
    assert [e["status"] for e in entries] == [200, 503, 500]
    assert entries[1]["reason"] == "authorization_refused"
    assert all(isinstance(e["ms"], float) and e["id"] for e in entries)
    # The id is echoed so a phone-side report can be joined to this line.
    assert reply.headers["x-request-id"] == entries[0]["id"]


@pytest.mark.asyncio
async def test_a_safe_caller_id_is_kept_and_an_unsafe_one_is_replaced(tmp_path):
    log = request_log.RequestLog(None)
    async with await _client(log) as client:
        kept = await client.get("/refused", headers={"X-Request-Id": "phone-1234.a_b"})
        replaced = await client.get(
            "/refused", headers={"X-Request-Id": 'x" injected\\n{"status":200}'}
        )
        probe = await client.get("/refused", headers={"X-Request-Id": "lop-probe-gate"})
    assert kept.headers["x-request-id"] == "phone-1234.a_b"
    assert replaced.headers["x-request-id"] != 'x" injected\\n{"status":200}'
    assert probe.status_code == 503
    assert [entry["id"] for entry in log.ring][0] == "phone-1234.a_b"
    assert len(log.ring) == 2, "this package's own probe is not recorded"


def test_unrouted_paths_mask_long_opaque_segments():
    scope = {"path": f"/share/{SECRET}/view", "type": "http"}
    assert request_log._template(scope) == "/share/:id/view"


@pytest.mark.asyncio
async def test_the_ring_and_the_file_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(request_log, "FILE_MAX_BYTES", 4096)
    path = tmp_path / "requests.log"
    log = request_log.RequestLog(path, ring_size=5)
    async with await _client(log) as client:
        for _ in range(120):
            await client.get("/refused")
    log.close()
    assert len(log.ring) == 5
    files = sorted(tmp_path.glob("requests.log*"))
    assert [f.name for f in files] == ["requests.log", "requests.log.1"]
    assert all(f.stat().st_size <= 4096 for f in files)
    assert path.stat().st_mode & 0o777 == 0o600


def test_the_tail_read_counts_status_classes_and_skips_foreign_lines(tmp_path):
    path = tmp_path / "requests.log"
    rows = [
        {"at": 1, "id": "a", "method": "GET", "path": "/x", "status": 200, "reason": "", "ms": 1.0},
        {
            "at": 2,
            "id": "b",
            "method": "GET",
            "path": "/x",
            "status": 503,
            "reason": "server_unavailable",
            "ms": 2.0,
        },
        {
            "at": 3,
            "id": "c",
            "method": "GET",
            "path": "/x",
            "status": None,
            "reason": "",
            "ms": 9.0,
        },
    ]
    path.write_text("garbage line\n" + "\n".join(json.dumps(r) for r in rows) + "\n")
    summary = request_log.read_tail(path, limit=2)
    assert summary["counts"] == {"2xx": 1, "5xx": 1, "none": 1}
    assert [e["id"] for e in summary["recent"]] == ["b", "c"]
    rendered = request_log.render(summary)
    assert rendered[0] == "Requests (last 3): 2xx=1 5xx=1 none=1"
    assert "503 server_unavailable" in rendered[1] and "no-response" in rendered[2]
    assert request_log.read_tail(tmp_path / "absent.log")["recent"] == []
    assert request_log.render(request_log.read_tail(tmp_path / "absent.log")) == []


@pytest.mark.asyncio
async def test_the_gateway_records_its_refusal_reason(tmp_path, monkeypatch):
    """The real gateway app, behind its logged entry point, on a lapsed lease."""
    from tests.unit.test_tunnels import make_connection, make_signing_key

    connection = make_connection(make_signing_key())
    async with httpx.AsyncClient() as upstream:
        relay = gateway.Gateway(connection, upstream)
        relay.authorized_until = 0
        relay.note_authorization_failure(gateway.SERVER_UNAVAILABLE)
        log = request_log.RequestLog(None)
        transport = httpx.ASGITransport(app=relay.logged_app(log))
        async with httpx.AsyncClient(transport=transport, base_url="http://x") as client:
            reply = await client.get(f"/api/sessions?token={SECRET}")
            port = connection["tunnel"]["gateway_port"]
            await client.get(f"http://127.0.0.1:{port}/_lop_tunnel/health")
    assert reply.status_code == 503 and reply.headers["x-request-id"]
    assert len(log.ring) == 1, "the loopback health probe is not recorded"
    entry = log.ring[0]
    assert entry["reason"] == gateway.SERVER_UNAVAILABLE and entry["status"] == 503
    assert SECRET not in json.dumps(entry)


def test_tunnel_status_text_shows_the_last_exit_and_recent_requests():
    from local_operator.tunnels.cli import _status_text

    payload = {
        "connector": {
            "state": "stopped",
            "reason": gateway.SERVER_UNAVAILABLE,
            "detail": gateway.TERMINAL_DETAIL[gateway.SERVER_UNAVAILABLE],
            "since": None,
            "remedy": None,
            "last_error": {"reason": gateway.SERVER_UNAVAILABLE, "at": 1_700_000_000},
        },
        "login": {"state": "ok"},
        "requests": {
            "recent": [
                {
                    "at": 1,
                    "id": "r1",
                    "method": "GET",
                    "path": "/api/sessions",
                    "status": 503,
                    "reason": gateway.SERVER_UNAVAILABLE,
                    "ms": 3.2,
                }
            ],
            "counts": {"5xx": 1},
            "window": 1,
        },
    }
    text = _status_text(payload, {"id": "synthetic", "status": "active"}, "cached")
    assert "Connector: stopped — Radient server error" in text
    assert "Last connector exit: Radient server error (at " in text
    assert "Requests (last 1): 5xx=1" in text
    assert "GET /api/sessions 503 server_unavailable 3.2ms id=r1" in text
    assert "lop tunnel install" not in text

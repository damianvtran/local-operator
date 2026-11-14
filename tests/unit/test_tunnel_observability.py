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


@pytest.mark.parametrize("status", [401, 409])
def test_a_4xx_still_reads_as_a_refusal(status):
    assert authorization_failure_reason(_server_error(status)) == gateway.REFUSED


@pytest.mark.parametrize(
    "status,reason", [(403, gateway.TUNNEL_INACTIVE), (402, gateway.BILLING_INACTIVE)]
)
def test_a_403_or_402_names_the_inactive_tunnel_not_the_login(status, reason):
    # `/connect` answers 403 for any non-active tunnel (402 is billing): before
    # this reason they read as `REFUSED`, whose remedy is a sign-in that cannot
    # clear billing or a console state.
    assert authorization_failure_reason(_server_error(status)) == reason
    assert classify_failure(_server_error(status)).reason == reason


def test_the_server_unavailable_reason_has_every_surface_entry():
    reason = gateway.SERVER_UNAVAILABLE
    # Keyed copy (R1-3): resolved through the `tunnel.status` catalogue, so the
    # accessors, not the legacy English tables, are the surface.
    label = gateway.reason_label(reason)
    terminal = gateway.terminal_detail(reason, "")
    relay = gateway.relay_detail(reason)
    assert label == "Radient server error"
    assert terminal and terminal != reason and "no local command is needed" in terminal
    assert relay and relay != reason
    # Not the login's fault: no sign-in remedy, and the copy says so.
    assert gateway.TERMINAL_REMEDY[reason] == "lop tunnel status"
    assert "login" not in label
    assert "Sign in" not in relay
    # The connector's own classification carries the same keyed sentence.
    assert classify_failure(_server_error(503)).detail == terminal


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


def _stamp(age_s: float = 0.0) -> str:
    # The service's own stamp shape (`service._announce`): local time, offset.
    import datetime

    when = datetime.datetime.now().astimezone() - datetime.timedelta(seconds=age_s)
    return when.isoformat(timespec="seconds")


def _retry_line(age_s: float = 0.0, reason: str = "server_unavailable") -> str:
    return (
        f"{_stamp(age_s)} local_operator.tunnels.service: connector stopped "
        f"reason={reason}: Radient tunnel request failed (HTTP 503).\n"
    )


RETRY_LINE = _retry_line()


def _write_service_log(text: str) -> None:
    config.directory().mkdir(parents=True, exist_ok=True)
    (config.directory() / "service.log").write_text(text)


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
    _write_service_log(_retry_line(20, "control_plane_unreachable") + _retry_line(5))
    result = await report.probe({"gateway_port": _closed_port()})
    assert result["reason"] == gateway.SERVER_UNAVAILABLE
    assert "lop tunnel install" not in result["detail"]
    assert result["last_error"]["reason"] == gateway.SERVER_UNAVAILABLE
    assert result["last_error"]["at"] is not None
    assert result["last_error"]["current"] is True


@pytest.mark.asyncio
async def test_an_old_failure_beside_a_closed_port_keeps_the_install_advice(tmp_path, monkeypatch):
    """R1-1 / Q1: service.log outlives the outage; an old exit is history."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write_service_log(
        "2026-09-01T10:00:00+10:00 local_operator.tunnels.service: connector stopped "
        "reason=server_unavailable: Radient tunnel request failed (HTTP 503).\n"
    )
    result = await report.probe({"gateway_port": _closed_port()})
    assert "lop tunnel install" in result["detail"]
    assert "no local command is needed" not in result["detail"]
    assert result["reason"] == "" and result["state"] == "stopped"
    # Still reported, as history, for the CLI's "Last connector exit" row.
    assert result["last_error"]["reason"] == gateway.SERVER_UNAVAILABLE
    assert result["last_error"]["current"] is False


@pytest.mark.asyncio
async def test_a_later_service_line_supersedes_a_recent_failure(tmp_path, monkeypatch):
    """R1-1: a line after the failure (the connector moved on) means not current."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write_service_log(
        _retry_line(10)
        + f"{_stamp(5)} local_operator.tunnels.service: relay offered and cloudflared "
        "unavailable; running relay-only\n"
    )
    found = report.last_connector_failure()
    assert found is not None and found["current"] is False
    result = await report.probe({"gateway_port": _closed_port()})
    assert "lop tunnel install" in result["detail"]


def test_failure_freshness_follows_the_retry_cadence(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write_service_log(_retry_line(0))
    found = report.last_connector_failure()
    assert found is not None and found["current"] is True
    at = found["at"]
    bound = report.CURRENT_FAILURE_S
    at_bound = report.last_connector_failure(now=at + bound)
    assert at_bound is not None and at_bound["current"] is True
    past_bound = report.last_connector_failure(now=at + bound + 1)
    assert past_bound is not None and past_bound["current"] is False
    # A stamp from the future (clock moved back) is no evidence either.
    future = report.last_connector_failure(now=at - bound - 1)
    assert future is not None and future["current"] is False


class _Http500Health(BaseHTTPRequestHandler):
    """A foreign listener on the gateway port answering a non-200."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self):
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.mark.asyncio
async def test_a_non_200_health_answer_names_the_status_and_the_current_failure(
    tmp_path, monkeypatch
):
    """Q2: not a bare "stopped"; the HTTP status is named, and a current exit kept."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write_service_log(_retry_line(3))
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Http500Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = await report.probe({"gateway_port": server.server_port})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert "HTTP 500" in result["detail"]
    assert "lop tunnel install" not in result["detail"]
    assert result["state"] == "not serving"
    assert result["reason"] == gateway.SERVER_UNAVAILABLE
    assert gateway.terminal_detail(gateway.SERVER_UNAVAILABLE, "") in result["detail"]
    assert result["last_error"]["current"] is True


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
    assert SECRET not in written
    entries = [json.loads(line) for line in written.splitlines()]
    assert [e["path"] for e in entries] == ["/api/sessions/{session_id}", "/refused", "/boom"]
    assert [e["status"] for e in entries] == [200, 503, 500]
    assert entries[1]["reason"] == "authorization_refused"
    assert all(isinstance(e["ms"], float) and e["id"] for e in entries)
    # The id is echoed so a phone-side report can be joined to this line.
    assert reply.headers["x-request-id"] == entries[0]["id"]


@pytest.mark.asyncio
async def test_a_safe_caller_id_is_kept_and_an_unsafe_one_is_replaced(tmp_path):
    path = tmp_path / "requests.log"
    log = request_log.RequestLog(path)
    async with await _client(log) as client:
        kept = await client.get("/refused", headers={"X-Request-Id": "phone-1234.a_b"})
        replaced = await client.get(
            "/refused", headers={"X-Request-Id": 'x" injected\\n{"status":200}'}
        )
        probe = await client.get("/refused", headers={"X-Request-Id": "lop-probe-gate"})
    assert kept.headers["x-request-id"] == "phone-1234.a_b"
    assert replaced.headers["x-request-id"] != 'x" injected\\n{"status":200}'
    assert probe.status_code == 503
    log.close()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[0]["id"] == "phone-1234.a_b"
    assert len(entries) == 2, "this package's own probe is not recorded"


def test_unrouted_paths_mask_long_opaque_segments():
    scope = {"path": f"/share/{SECRET}/view", "type": "http"}
    assert request_log._template(scope) == "/share/:id/view"


@pytest.mark.parametrize(
    ("raw", "logged"),
    [
        # R1-4: lop's 12-hex session ids sit under the long-token bound.
        ("/api/sessions/3b11f69df1d3/history", "/api/sessions/:id/history"),
        ("/api/projects/a1b2c3/tags/9f8e7d6c", "/api/projects/:id/tags/:id"),
        # Route words and short version segments stay readable.
        ("/api/v1/push-devices/events", "/api/v1/push-devices/events"),
        ("/", "/"),
    ],
)
def test_unrouted_paths_mask_short_identifiers(raw, logged):
    assert request_log._template({"path": raw, "type": "http"}) == logged


@pytest.mark.asyncio
async def test_the_file_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(request_log, "FILE_MAX_BYTES", 4096)
    path = tmp_path / "requests.log"
    log = request_log.RequestLog(path)
    async with await _client(log) as client:
        for _ in range(120):
            await client.get("/refused")
    log.close()
    # R1-5: no in-memory copy is kept; the file is the only record.
    assert not hasattr(log, "ring")
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
    assert "503 server_unavailable" in rendered[1] and "no response" in rendered[2]
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
        path = tmp_path / "requests.log"
        log = request_log.RequestLog(path)
        transport = httpx.ASGITransport(app=relay.logged_app(log))
        async with httpx.AsyncClient(transport=transport, base_url="http://x") as client:
            reply = await client.get(f"/api/sessions?token={SECRET}")
            port = connection["tunnel"]["gateway_port"]
            await client.get(f"http://127.0.0.1:{port}/_lop_tunnel/health")
    assert reply.status_code == 503 and reply.headers["x-request-id"]
    log.close()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(entries) == 1, "the loopback health probe is not recorded"
    entry = entries[0]
    assert entry["reason"] == gateway.SERVER_UNAVAILABLE and entry["status"] == 503
    assert SECRET not in json.dumps(entry)


def test_tunnel_status_text_shows_the_last_exit_and_recent_requests():
    from local_operator.tunnels.cli import _status_text

    payload = {
        "connector": {
            "state": "stopped",
            "reason": gateway.SERVER_UNAVAILABLE,
            "detail": gateway.terminal_detail(gateway.SERVER_UNAVAILABLE, ""),
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


def test_the_mobile_daemon_names_why_it_refused(tmp_path, monkeypatch):
    """Q3: the daemon hop's 401/403 entries carry a reason code, not "".

    The daemon wrapped exactly as `mobile/service.amain` wraps it.
    """
    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app

    cfg = tmp_path / "config"
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    daemon = MobileDaemon(port=0, password="pw123")
    path = tmp_path / "mobile-requests.log"
    log = request_log.RequestLog(path)
    app = log.middleware(build_app(daemon), skip=frozenset({"/healthz"}))
    with TestClient(app, follow_redirects=False) as client:
        anonymous = client.get("/api/sessions")
        wrong = client.post("/login", data={"password": "nope"})
        foreign = client.post(
            "/login", data={"password": "pw123"}, headers={"Origin": "https://evil.example"}
        )
    log.close()
    assert (anonymous.status_code, wrong.status_code, foreign.status_code) == (401, 401, 403)
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(e["path"], e["status"], e["reason"]) for e in entries] == [
        ("/api/sessions", 401, "unauthorized"),
        ("/login", 401, "login_failed"),
        ("/login", 403, "cross_origin"),
    ]

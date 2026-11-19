"""A real HTTP 426 keeps one safe, permanent meaning across client surfaces."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels import cli, gateway, install, report, service, setup, state, upgrade
from local_operator.tunnels.api import RadientTunnels
from local_operator.tunnels.errors import ClientUpgradeRequired, RequestFailed


@contextmanager
def refusing_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"message": "untrusted-body-canary: synthetic-secret"}).encode()
            self.send_response(426)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        assert not thread.is_alive()


class LoopbackTransport(httpx.AsyncBaseTransport):
    """Only the injected test transport may redirect the pinned API to loopback."""

    def __init__(self, port):
        self.port = port
        self.inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        assert request.url.host == "api.radienthq.com"
        request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=self.port)
        return await self.inner.handle_async_request(request)

    async def aclose(self):
        await self.inner.aclose()


@pytest.mark.asyncio
async def test_http_426_is_actionable_permanent_and_body_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "acct", "access": "synthetic"}
        )
    with refusing_server() as port:
        async with httpx.AsyncClient(transport=LoopbackTransport(port), trust_env=False) as client:
            with pytest.raises(ClientUpgradeRequired) as raised:
                await RadientTunnels(row.id, client).request("POST", "/tunnel-1/connect")
    failure = raised.value
    assert failure.status == 426
    assert "untrusted-body-canary" not in str(failure)
    assert "lop update" in str(failure)
    assert upgrade.OVERRIDE in str(failure)
    assert "lop tunnel connect" in str(failure)
    assert "will not fix" in str(failure)

    async def rejected(*_args, **_kwargs):
        raise failure

    # These are the real shell and supervisor entrypoints, not a text renderer
    # replacing them: they consume the exception earned by the actual HTTP call.
    monkeypatch.setattr(cli, "dispatch", rejected)
    assert await asyncio.to_thread(cli.main, argparse.Namespace(tunnel_command="connect")) == 1
    assert str(failure) in capsys.readouterr().err
    applied = await setup._apply(initial_plan={
        "step": "apply", "ready": True, "verified": False,
        "configured": False, "credential_id": row.id, "activation_required": False,
    })
    assert applied["step"] == "client_update" and applied["failure_stage"] == "reservation"
    assert not applied["ready"] and applied["receipts"] == []
    assert "untrusted-body-canary" not in json.dumps(applied)
    monkeypatch.setattr(service, "run", rejected)
    assert await asyncio.to_thread(service.main) == 0  # supervisor must not restart
    parked = state.parked()
    assert parked is not None and parked["reason"] == upgrade.REASON
    assert parked["remedy"]["command"] == "lop update"
    assert service.authorization_failure_reason(failure) == upgrade.REASON
    verdict = service.classify_failure(failure)
    assert verdict.kind == "terminal_config" and verdict.reason == upgrade.REASON
    step, remedy = setup._unreadable_step(failure)
    assert step == "client_update" and remedy == upgrade.detail()
    plan = {"step": step, "remedy": remedy, "receipts": [], "verified": False, "ready": False}
    assert "lop update" in setup.text(plan)
    tui = setup.text(plan, surface="tui")
    assert "/update" in tui and "/mobile enable" in tui
    assert "lop update" not in tui and "lop login" not in tui
    assert upgrade.OVERRIDE in setup.plain_remedy(plan)
    from local_operator.server.routes.desktop_tunnel_setup import _contract

    projected = _contract({**plan, "apply_command": "lop login radient"}, local=True)
    assert projected["blocked_on"] == "client_update"
    assert projected["remedy"]["kind"] == upgrade.KIND
    assert projected["remedy"]["command"] is None
    assert "lop " not in projected["remedy"]["text"]
    assert "untrusted-body-canary" not in json.dumps(projected)
    assert gateway.reason_label(upgrade.REASON) == upgrade.label()
    assert gateway.terminal_detail(upgrade.REASON, "untrusted-body-canary") == upgrade.detail()
    assert gateway.remedy_kind(upgrade.REASON) == upgrade.KIND
    result = report.remedy({}, {**parked, "status": "parked"}, {"state": "ok"})
    assert result and result["kind"] == upgrade.KIND
    assert "untrusted-body-canary" not in json.dumps(parked)
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: "launchd")
    assert install.rearm_if_parked(provider="radient", credential_id=row.id) == ""
    assert state.parked()["reason"] == upgrade.REASON
    monkeypatch.setattr(
        setup.config, "transport_override", lambda: setup.config.TRANSPORT_CLOUDFLARE
    )
    known = setup._client_update_plan({"reason": upgrade.REASON}, {"credential_id": row.id})
    assert known and known["step"] == "client_update" and not known["ready"]
    monkeypatch.setattr(setup.config, "transport_override", lambda: None)
    assert setup._client_update_plan({"reason": upgrade.REASON}, {}) is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_older_typed_426_wrappers_keep_permanent_classification(wrapped):
    failure = RequestFailed("untrusted-body-canary", status=426)
    if wrapped:
        outer = ValueError("wrapper")
        outer.__cause__ = failure
        failure = outer
    verdict = service.classify_failure(failure)
    assert verdict.kind == "terminal_config" and verdict.reason == upgrade.REASON
    assert "untrusted-body-canary" not in verdict.detail
    assert setup._unreadable_step(failure)[0] == "client_update"


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_other_statuses_are_not_protocol_prerequisites(status):
    failure = RequestFailed("synthetic", status=status)
    assert service.classify_failure(failure).reason != upgrade.REASON
    assert setup._unreadable_step(failure)[0] != "client_update"


@pytest.mark.asyncio
async def test_running_connector_propagates_permanent_renewal_refusal(tmp_path, monkeypatch):
    import time

    from tests.unit.test_tunnel_relay import FakeRelay, _relay_service

    async def refused_refresh(*_args):
        raise ClientUpgradeRequired()

    async with FakeRelay({}) as relay:
        module, served, _, children = _relay_service(tmp_path, monkeypatch, relay, binary=False)
        # A valid but near-expiry grant makes the actual relay task request a
        # fresh /connect. A 426 there must reach main, not its reconnect backoff.
        served["relay"]["expires_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30)
        )
        monkeypatch.setattr(module, "refresh_relay", refused_refresh)
        with pytest.raises(ClientUpgradeRequired):
            await asyncio.wait_for(module.run(), timeout=15)
        assert children == []


def test_upgrade_copy_tracks_every_manifest_locale_at_display_time():
    from pathlib import Path

    from local_operator.i18n import catalogues
    from local_operator.i18n.text import active_locale, set_active_locale

    root = Path(__file__).resolve().parents[2]
    locales = json.loads((root / "i18n/languages.json").read_text())["languages"]
    previous = active_locale()
    try:
        for entry in locales:
            locale = entry["code"]
            set_active_locale(locale)
            source = catalogues.load_catalogue(locale, "tunnel.upgrade")
            assert gateway.reason_label(upgrade.REASON) == source["tunnel.upgrade.client.label"]
            detail = str(ClientUpgradeRequired())
            assert "lop update" in detail and upgrade.OVERRIDE in detail
            assert "lop tunnel connect" in detail
            assert "{update_command}" not in detail
            assert upgrade.plain() == source["tunnel.upgrade.client.plain"]
    finally:
        set_active_locale(previous)

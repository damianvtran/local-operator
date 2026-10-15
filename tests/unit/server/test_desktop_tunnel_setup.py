"""Real desktop admission and shared setup, with only upstream/service boundaries fake."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from local_operator.config_writer import nonblocking_file_claim
from local_operator.server.routes import desktop_tunnel_setup as routes
from local_operator.tunnels import api, cli, config, install, report, setup

TOKEN = "synthetic-desktop-capability"
# The production prerequisite check, captured BEFORE the `upstream` fixture stubs it,
# so one test can prove the real thing refuses on a host with no cloudflared.
REAL_CLOUDFLARED_PREFLIGHT = setup._cloudflared_preflight
ORIGIN = "http://desktop.fixture.test"
PATH = "/v1/desktop/tunnel/setup"


@pytest_asyncio.fixture
async def desktop(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", ORIGIN)
    monkeypatch.setenv("LOP_TUNNEL_NO_REARM", "1")
    app = FastAPI()
    app.include_router(routes.router)
    # No Depends overrides: bearer/origin admission is the real production gate.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN},
    ) as client:
        yield client, app, tmp_path
    await routes.drain(app)


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    calls = []
    record = {}
    quote = {
        "eligible": True,
        "balance_usd": 1,
        "monthly_price_usd": "0.05",
        "monthly_cost_usd": "0.01",
        "amount_due_usd": "0",
    }
    health = {"ok": True, "connected": True}
    installs = []
    release = threading.Event()
    entered = threading.Event()
    controlled = {"block_install": False}

    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def list_credentials(self, provider):
            return [
                SimpleNamespace(
                    id=i,
                    provider="radient",
                    credential_type="oauth",
                    data={"email": f"account{i}@example.test"},
                )
                for i in (17, 23)
            ]

        def get_credential(self, selected):
            return next((r for r in self.list_credentials("radient") if r.id == selected), None)

        async def ensure_oauth_fresh_or_raise(self, selected):
            return {"access": f"SYNTHETIC_OWNER_{selected}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self):
            calls.append(("GET", self.path))
            if self.path == "/_lop_tunnel/health":
                response = health
            else:
                selected = self.headers.get("Authorization", "").rsplit("_", 1)[-1]
                result = (
                    {"account": {"id": f"account-{selected}"}}
                    if self.path == "/v1/me"
                    else quote if self.path.endswith("/billing") else record
                )
                response = {"result": result}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, self.headers.get("Idempotency-Key")))
            if self.path.endswith("/billing/activate"):
                quote["eligible"] = True
                result = quote
            else:
                record.update(
                    body,
                    id="fixture-tunnel",
                    status="active",
                    owner_account_id="account-17",
                    connector_token="PRIVATE_FIXTURE_TOKEN",
                )
                record["harnesses"][0]["hostname"] = "exact.fixture.example.test"
                result = record
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"result": result}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(api, "DEFAULT_API_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(api, "AuthStore", Store)
    monkeypatch.setattr(setup, "AuthStore", Store)
    monkeypatch.setattr(config, "DEFAULT_GATEWAY_PORT", server.server_port)
    monkeypatch.setattr(setup.supervisors, "supervisor", lambda: "launchctl")
    monkeypatch.setattr(setup, "_cloudflared_preflight", lambda path: "/synthetic/cloudflared")
    # The run path ALSO dispatches the real `tunnel install`, whose own lookup
    # (cli.cloudflared_binary -> shutil.which) is a second, independent check. A
    # developer host has cloudflared on PATH, so stubbing only the preflight
    # passes there while a CI runner without it fails the install and never
    # reaches `verified`. Both seams are synthetic here; the real check has its own
    # test below.
    monkeypatch.setattr(cli, "cloudflared_binary", lambda configured=None: "/synthetic/cloudflared")
    monkeypatch.setattr(setup.report, "login_verdict", AsyncMock(return_value={"state": "ok"}))
    monkeypatch.setattr(setup.mobile_install, "health", lambda port: {"ok": True})
    monkeypatch.setattr(setup.mobile_install, "gate_closed", lambda port: True)

    def simulated_install():
        installs.append("install")
        if controlled["block_install"]:
            entered.set()
            assert release.wait(3), "bounded synthetic worker was not released"
        health["connected"] = True

    monkeypatch.setattr(install, "install", simulated_install)
    fixture = SimpleNamespace(
        calls=calls,
        quote=quote,
        record=record,
        health=health,
        installs=installs,
        release=release,
        entered=entered,
        controlled=controlled,
    )
    try:
        yield fixture
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_local_poll_absent_store_does_not_create_or_refresh(desktop, monkeypatch):
    client, _app, root = desktop
    monkeypatch.setattr(
        report, "login_verdict", AsyncMock(side_effect=AssertionError("no refresh"))
    )
    monkeypatch.setattr(
        api.RadientTunnels, "request", AsyncMock(side_effect=AssertionError("no upstream"))
    )
    before = set(root.rglob("*"))
    reply = await client.get(PATH)
    assert reply.status_code == 200
    result = reply.json()["result"]
    assert result["source"] == "local" and result["quote"] is None and not result["verified"]
    assert result["blocked_on"] == "login"
    assert set(root.rglob("*")) == before


@pytest.mark.asyncio
async def test_local_expired_metadata_has_no_refresh_or_store_write(desktop, monkeypatch):
    from local_operator.providers.auth_store import AuthStore

    client, _app, root = desktop
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "SYNTHETIC_PRIVATE",
                "email": "fixture@example.test",
                "expires": 0,
            },
        )
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    async def forbidden(*args, **kwargs):
        raise AssertionError("Local GET cannot refresh auth")

    monkeypatch.setattr(AuthStore, "ensure_oauth_fresh_or_raise", forbidden)
    response = await client.get(PATH)
    result = response.json()["result"]
    assert result["blocked_on"] == "quote_unread" and result["quote"] is None
    assert result["credential_id"] == row.id and result["identity_label"] == "f***@example.test"
    assert not result["verified"] and result["url"] is None
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert "SYNTHETIC_PRIVATE" not in response.text


@pytest.mark.asyncio
async def test_local_poll_rejects_non_loopback_authority_in_corrupt_port(
    desktop, upstream, monkeypatch
):
    client, _app, _root = desktop
    config.save(
        {"credential_id": 17, "tunnel_id": "fixture", "gateway_port": "80@wrong.example.test"}
    )
    probe = AsyncMock(side_effect=AssertionError("Malformed port must not reach any HTTP probe"))
    monkeypatch.setattr(report, "connector_state", probe)
    response = await client.get(PATH)
    assert response.status_code == 503 and not upstream.calls
    probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_poll_selected_metadata_has_unread_quote(desktop, upstream):
    client, _app, _root = desktop
    reply = await client.get(PATH, params={"credential_id": 17})
    result = reply.json()["result"]
    assert (
        result["blocked_on"] == "quote_unread" and result["identity_label"] == "a***@example.test"
    )
    assert result["quote"] is None and result["url"] is None
    assert upstream.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Authorization": ""}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Origin": "http://wrong.fixture.test"}, 403),
        ({"Origin": "", "Sec-Fetch-Site": "cross-site"}, 403),
    ],
)
async def test_real_bearer_origin_gate(desktop, upstream, headers, status):
    client, _app, _root = desktop
    response = await client.post(PATH, headers=headers, json={"action": "run", "credential_id": 17})
    assert response.status_code == status
    assert not upstream.calls and not (config.directory() / "config.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"action": "run"},
        {"action": "run", "credential_id": True},
        {"action": "run", "credential_id": "17"},
        {"action": "run", "credential_id": 0},
        {"action": "run", "credential_id": 17, "unexpected": True},
        {"action": "run", "credential_id": 17, "accept_monthly_price": "NaN"},
        {"action": "plan", "credential_id": 17, "accept_monthly_price": "0.05"},
    ],
)
async def test_strict_explicit_owner_and_price_model(desktop, upstream, body):
    client, _app, _root = desktop
    assert (await client.post(PATH, json=body)).status_code == 422
    assert not upstream.calls and not (config.directory() / "config.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "17",
        "true",
        '"PRIVATE_SENTINEL"',
        "{}",
        '{"credential_id":17,"key":"PRIVATE_SENTINEL","payload":[]}',
    ],
)
@pytest.mark.parametrize("action", [None, "plan", "run"])
async def test_corrupt_intent_is_structured_without_reads_or_writes(desktop, upstream, raw, action):
    client, _app, root = desktop
    config.directory().mkdir(parents=True, exist_ok=True)
    intent = config.directory() / "create.json"
    intent.write_text(raw)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    response = (
        await client.get(PATH)
        if action is None
        else await client.post(PATH, json={"action": action, "credential_id": 17})
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "setup_intent_corrupt"
    assert "Leave create.json unchanged" in response.json()["detail"]["text"]
    assert "PRIVATE_SENTINEL" not in response.text
    assert not upstream.calls and not upstream.installs
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_cli_setup_corrupt_intent_fails_before_account_selection(desktop, upstream, capsys):
    import argparse

    from local_operator.tunnels.arguments import add_parser
    from local_operator.tunnels.cli import main

    _client, _app, root = desktop
    config.directory().mkdir(parents=True, exist_ok=True)
    (config.directory() / "create.json").write_text('"CLI_PRIVATE_SENTINEL"')
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    assert (
        main(parser.parse_args(["tunnel", "setup", "apply", "--credential-id", "17", "--json"]))
        == 1
    )
    output = capsys.readouterr()
    assert "Leave create.json unchanged" in output.err
    assert "CLI_PRIVATE_SENTINEL" not in output.out + output.err
    assert not upstream.calls and not upstream.installs
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.asyncio
async def test_healthy_config_ignores_unused_corrupt_intent(desktop, upstream):
    client, _app, root = desktop
    assert (await client.post(PATH, json={"action": "run", "credential_id": 17})).json()["result"][
        "verified"
    ]
    (config.directory() / "create.json").write_text('"UNUSED_PRIVATE_SENTINEL"')
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    posts = sum(c[0] == "POST" for c in upstream.calls)
    installs = len(upstream.installs)
    local = await client.get(PATH)
    assert local.status_code == 200 and local.json()["result"]["quote"] is None
    result = await client.post(PATH, json={"action": "run", "credential_id": 17})
    assert result.status_code == 200 and result.json()["result"]["verified"]
    assert "UNUSED_PRIVATE_SENTINEL" not in local.text + result.text
    assert posts == sum(c[0] == "POST" for c in upstream.calls) and installs == len(
        upstream.installs
    )
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.asyncio
async def test_fresh_plan_and_wrong_price_never_mutate(desktop, upstream):
    client, _app, _root = desktop
    upstream.quote["eligible"] = False
    plan = (await client.post(PATH, json={"action": "plan", "credential_id": 17})).json()["result"]
    assert plan["blocked_on"] == "quote" and plan["quote"]["monthly_price_usd"] == "0.05"
    result = (
        await client.post(
            PATH, json={"action": "run", "credential_id": 17, "accept_monthly_price": "0.04"}
        )
    ).json()["result"]
    assert result["step"] == "quote_changed" and result["blocked_on"] == "quote"
    assert not any(c[0] == "POST" for c in upstream.calls)
    assert not (config.directory() / "config.json").exists()


@pytest.mark.asyncio
async def test_real_cloudflared_preflight_refuses_before_any_mutation(
    desktop, upstream, monkeypatch
):
    """Q2 on a host WITHOUT cloudflared (what a Linux CI runner is): refuse, write nothing."""
    client, _app, _root = desktop
    monkeypatch.setattr(setup, "_cloudflared_preflight", REAL_CLOUDFLARED_PREFLIGHT)
    # No binary anywhere, whatever the host has installed.
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)
    result = (await client.post(PATH, json={"action": "run", "credential_id": 17})).json()["result"]
    assert result["step"] == "cloudflared" and result["blocked_on"] == "cloudflared"
    assert result["verified"] is False
    assert result["remedy"]["kind"] == "install_cloudflared"
    assert not any(c[0] == "POST" for c in upstream.calls)
    assert not upstream.installs
    assert not (config.directory() / "config.json").exists()
    assert not (config.directory() / "create.json").exists()


@pytest.mark.asyncio
async def test_run_uses_real_connected_producer_and_healthy_noop(desktop, upstream):
    client, _app, _root = desktop
    result = (await client.post(PATH, json={"action": "run", "credential_id": 17})).json()["result"]
    assert result["verified"] and result["url"] == "https://exact.fixture.example.test"
    assert result["done"] == ["reservation", "installation"]
    assert (await report.connector_state(config.load()))["state"] == "connected"
    before = (config.directory() / "config.json").read_bytes()
    posts = sum(c[0] == "POST" for c in upstream.calls)
    install_count = len(upstream.installs)
    assert (await client.post(PATH, json={"action": "run", "credential_id": 17})).json()["result"][
        "verified"
    ]
    assert before == (config.directory() / "config.json").read_bytes()
    assert posts == sum(c[0] == "POST" for c in upstream.calls) and install_count == len(
        upstream.installs
    )
    assert "PRIVATE_FIXTURE_TOKEN" not in json.dumps(result)
    assert (await client.post(PATH, json={"action": "run", "credential_id": 23})).json()["result"][
        "blocked_on"
    ] == "choose_account"


@pytest.mark.asyncio
async def test_cancelled_response_retains_all_owner_claims_until_drain(desktop, upstream):
    client, app, _root = desktop
    upstream.controlled["block_install"] = True
    upstream.health["connected"] = False
    pending = asyncio.create_task(client.post(PATH, json={"action": "run", "credential_id": 17}))
    assert await asyncio.to_thread(upstream.entered.wait, 2)
    pending.cancel()
    await asyncio.sleep(0.01)
    pending.cancel()
    await asyncio.sleep(0.01)
    assert not pending.done() and app.state.tunnel_setup_task is not None
    assert (
        await client.post(PATH, json={"action": "plan", "credential_id": 23})
    ).status_code == 409
    with pytest.raises(setup.SetupInProgress):
        await setup.apply(17)
    with nonblocking_file_claim(config.directory() / "setup.lock") as acquired:
        assert not acquired
    upstream.release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert app.state.tunnel_setup_task is None
    assert (await client.post(PATH, json={"action": "run", "credential_id": 17})).status_code == 200


@pytest.mark.asyncio
async def test_shutdown_drain_joins_owned_install(desktop, upstream):
    client, app, _root = desktop
    upstream.controlled["block_install"] = True
    upstream.health["connected"] = False
    pending = asyncio.create_task(client.post(PATH, json={"action": "run", "credential_id": 17}))
    assert await asyncio.to_thread(upstream.entered.wait, 2)
    shutdown = asyncio.create_task(routes.drain(app))
    await asyncio.sleep(0.01)
    assert not shutdown.done()
    with nonblocking_file_claim(config.directory() / "setup.lock") as acquired:
        assert not acquired
    upstream.release.set()
    assert (await pending).status_code == 200
    await shutdown
    assert app.state.tunnel_setup_task is None


@pytest.mark.asyncio
async def test_deadline_drains_before_504_and_retry(desktop, upstream, monkeypatch):
    client, app, _root = desktop
    monkeypatch.setattr(routes, "RUN_DEADLINE_S", 0.02)
    upstream.controlled["block_install"] = True
    upstream.health["connected"] = False
    pending = asyncio.create_task(client.post(PATH, json={"action": "run", "credential_id": 17}))
    assert await asyncio.to_thread(upstream.entered.wait, 2)
    await asyncio.sleep(0.04)
    assert not pending.done()
    assert (await client.post(PATH, json={"action": "run", "credential_id": 23})).status_code == 409
    upstream.release.set()
    response = await pending
    assert response.status_code == 504 and response.json()["detail"]["code"] == "setup_deadline"
    assert app.state.tunnel_setup_task is None
    assert len(upstream.installs) == 1
    assert (await client.post(PATH, json={"action": "run", "credential_id": 17})).status_code == 200

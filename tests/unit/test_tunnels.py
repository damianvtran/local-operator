"""Tunnel trust boundaries exercised with real RSA proofs and ASGI requests."""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import hashlib
import json
import socket
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from local_operator.mobile.auth import COOKIE_NAME, verify_cookie
from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels import config, state
from local_operator.tunnels.arguments import add_parser
from local_operator.tunnels.cli import (
    _billing_summary,
    _ensure_billing,
    _summary,
    dispatch,
)
from local_operator.tunnels.gateway import (
    CONSOLE_URL,
    LEASE_PENDING,
    LOGIN_REQUIRED,
    MAX_BODY_BYTES,
    NOT_AUTHORIZED,
    PROOF_HEADER,
    REFUSED,
    RELAY_DETAIL,
    TERMINAL_DETAIL,
    UNREACHABLE,
    Gateway,
)
from local_operator.tunnels.service import active

HOST = "abc123-lop.radienthq.com"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def connection(signing_key):
    public = json.loads(RSAAlgorithm.to_jwk(signing_key.public_key()))
    public.update(kid="origin-1", alg="RS256")
    return {
        "gateway_port": 4099,
        "cloudflared_token": "private-connector-test-token",
        "tunnel": {
            "id": "tunnel-1",
            "version": 2,
            "enabled": True,
            "status": "active",
            "gateway_port": 4099,
            "harnesses": [
                {"id": "local-operator", "enabled": True, "port": 4098, "hostname": HOST}
            ],
        },
        "origin_auth": {
            "issuer": config.ORIGIN_ISSUER,
            "owner_account_id": "owner-1",
            "tunnel_id": "tunnel-1",
            "version": 2,
            "jwks": {"keys": [public]},
        },
    }


def proof(signing_key, *, method="GET", target="/api/sessions", body=b"", **overrides):
    now = int(time.time())
    claims = {
        "iss": config.ORIGIN_ISSUER,
        "aud": HOST,
        "sub": "owner-1",
        "tunnel_id": "tunnel-1",
        "harness_id": "local-operator",
        "version": 2,
        "method": method,
        "target": target,
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "iat": now,
        "exp": now + 30,
        "jti": str(uuid.uuid4()),
    }
    claims.update(overrides)
    return jwt.encode(claims, signing_key, algorithm="RS256", headers={"kid": "origin-1"})


@pytest.mark.asyncio
async def test_real_proof_authenticates_relay_without_leaking_cloud_headers(
    connection, signing_key
):
    seen = []

    def origin(request):
        seen.append(request)
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b'{"sessions":[]}'),
            headers={"content-type": "application/json", "set-cookie": "local_secret=hidden"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="private-local-password")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            response = await client.get(
                "/api/sessions",
                headers={
                    PROOF_HEADER: proof(signing_key),
                    "cookie": "radient_cloud=secret",
                    "authorization": "Bearer cloud-secret",
                    "x-forwarded-host": "attacker.invalid",
                    "cf-access-jwt-assertion": "forged",
                    "x-opencode-ticket": "1",
                },
            )
    assert response.status_code == 200
    assert response.json() == {"sessions": []}
    assert "set-cookie" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    request = seen[0]
    assert request.url.host == "127.0.0.1" and request.url.port == 4098
    assert request.headers["host"] == HOST
    assert verify_cookie(
        request.headers["cookie"].removeprefix(COOKIE_NAME + "="), "private-local-password"
    )
    for name in (
        PROOF_HEADER,
        "authorization",
        "cf-access-jwt-assertion",
        "x-forwarded-host",
        "x-opencode-ticket",
    ):
        assert name not in request.headers


@pytest.mark.asyncio
async def test_verified_phone_can_start_and_steer_through_real_relay_gate(
    connection, signing_key, tmp_path
):
    from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
    from local_operator.mobile.types import SessionRecord

    daemon = MobileDaemon(password="origin-only-password", dial_registrants=False)
    daemon.spawn_session = AsyncMock(return_value=4242)
    daemon.request = AsyncMock(return_value={"op": "ack", "detail": "steer accepted"})
    record = SessionRecord(
        pid=4242,
        kind="tui",
        session_id="fixture-session",
        conversation_name="fixture",
        cwd=str(tmp_path),
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )
    daemon.table.entries[record.pid] = SessionEntry(record)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(daemon))) as relay:
        gateway = Gateway(connection, relay, mobile_password="origin-only-password")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as phone:
            # The phone has never submitted the relay's pairing password.
            for target, payload in (
                ("/api/sessions/start", {"cwd": str(tmp_path)}),
                (
                    "/api/sessions/fixture-session/command",
                    {"op": "steer", "command_id": str(uuid.uuid4()), "text": "Phone direction"},
                ),
            ):
                body = json.dumps(payload).encode()
                response = await phone.post(
                    target,
                    content=body,
                    headers={
                        "origin": "https://" + HOST,
                        "content-type": "application/json",
                        PROOF_HEADER: proof(signing_key, method="POST", target=target, body=body),
                    },
                )
                assert response.status_code == 200
                assert "set-cookie" not in response.headers
            assert not phone.cookies
    daemon.spawn_session.assert_awaited_once()
    assert daemon.request.call_args.args[:2] == (4242, "steer")
    assert daemon.request.call_args.kwargs["text"] == "Phone direction"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims",
    [
        {"sub": "other-owner"},
        {"aud": "other-lop.radienthq.com"},
        {"tunnel_id": "other"},
        {"harness_id": "opencode"},
        {"version": 1},
        {"version": True},
        {"method": "POST"},
        {"target": "/api/other"},
        {"body_sha256": "0" * 64},
        {"iss": "https://other.invalid"},
        {"exp": 1},
        {"exp": int(time.time()) + 3600},
    ],
)
async def test_invalid_proofs_never_reach_loopback(connection, signing_key, claims):
    origin = AsyncMock(return_value=httpx.Response(200))
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            response = await client.get(
                "/api/sessions", headers={PROOF_HEADER: proof(signing_key, **claims)}
            )
    assert response.status_code == 401
    origin.assert_not_called()


@pytest.mark.asyncio
async def test_oversize_body_and_expired_policy_never_reach_origin(connection, signing_key):
    origin = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            assert (
                await client.post(
                    "/api/sessions",
                    content=b"x" * (MAX_BODY_BYTES + 1),
                    headers={"origin": "https://" + HOST, PROOF_HEADER: proof(signing_key)},
                )
            ).status_code == 413
            gateway.authorized_until = 0
            assert (
                await client.get("/api/sessions", headers={PROOF_HEADER: proof(signing_key)})
            ).status_code == 503
    origin.assert_not_called()


@pytest.mark.asyncio
async def test_persisted_explicit_stop_does_not_publish_or_restart(tmp_path, monkeypatch):
    from local_operator.tunnels import service

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save({"stopped": True})
    binary = Mock(side_effect=AssertionError("stopped services must not launch a connector"))
    monkeypatch.setattr(service, "cloudflared_binary", binary)
    assert await service.run() == 0
    binary.assert_not_called()


@pytest.mark.asyncio
async def test_mutation_bound_body_and_replay_and_sibling_origin(connection, signing_key):
    calls = []

    def origin(request):
        calls.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    body = b'{"text":"hello"}'
    token = proof(signing_key, method="POST", body=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            headers = {PROOF_HEADER: token, "origin": "https://other-lop.radienthq.com"}
            assert (
                await client.post("/api/sessions", content=body, headers=headers)
            ).status_code == 403
            headers["origin"] = "https://" + HOST
            assert (
                await client.post("/api/sessions", content=b"altered", headers=headers)
            ).status_code == 401
            assert (
                await client.post("/api/sessions", content=body, headers=headers)
            ).status_code == 200
            assert (
                await client.post("/api/sessions", content=body, headers=headers)
            ).status_code == 401
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("withdraw", ["revoke", "expire"])
async def test_upload_cannot_outlive_its_gateway_authorization(connection, signing_key, withdraw):
    origin = AsyncMock(return_value=httpx.Response(200))
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")

        async def upload():
            yield b"{"
            # This executes only after handle() has admitted the request and
            # started consuming its body. No timing assumption or sleep races
            # the policy change against the proxy's final authorization check.
            if withdraw == "revoke":
                gateway.revoked = True
            else:
                gateway.authorized_until = 0
            yield b"}"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            response = await client.post(
                "/api/sessions",
                content=upload(),
                headers={
                    "origin": "https://" + HOST,
                    PROOF_HEADER: proof(signing_key, method="POST", body=b"{}"),
                },
            )
            assert response.status_code == 503
    origin.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["cross-site", "same-site"])
async def test_external_phone_navigation_is_allowed_but_subrequests_are_denied(
    connection, signing_key, site
):
    origin = AsyncMock(return_value=httpx.Response(200, stream=httpx.ByteStream(b"phone page")))
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            headers = {
                "sec-fetch-site": site,
                "sec-fetch-mode": "navigate",
                "accept": "text/html",
                PROOF_HEADER: proof(signing_key, target="/"),
            }
            assert (await client.get("/", headers=headers)).status_code == 200
            for mode in ("cors", "no-cors"):
                assert (
                    await client.get("/", headers={**headers, "sec-fetch-mode": mode})
                ).status_code == 403
            assert (
                await client.post("/", headers={**headers, "origin": "https://" + HOST})
            ).status_code == 403
            assert (
                await client.get("/", headers={**headers, "origin": "https://other.invalid"})
            ).status_code == 403
    assert origin.call_count == 1


@pytest.mark.asyncio
async def test_raw_encoded_target_and_unknown_host(connection, signing_key):
    seen = []

    def origin(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    target = "/api/a%2Fb?x=a%20b&x=%2f"
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            assert (
                await client.get(target, headers={PROOF_HEADER: proof(signing_key, target=target)})
            ).status_code == 200
            assert (
                await client.get(
                    "/api/sessions",
                    headers={"host": "127.0.0.1:4099", PROOF_HEADER: proof(signing_key)},
                )
            ).status_code == 404
            gateway.revoked = True
            assert (
                await client.get("/api/sessions", headers={PROOF_HEADER: proof(signing_key)})
            ).status_code == 503
    assert seen == [target.encode()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location", ["https://other.invalid/", "//other.invalid/", "/\\other.invalid/"]
)
async def test_harness_redirect_cannot_escape_the_checked_origin(connection, signing_key, location):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(302, headers={"location": location}))
    ) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            response = await client.get("/api/sessions", headers={PROOF_HEADER: proof(signing_key)})
            assert response.status_code == 502
            assert "location" not in response.headers


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["origin_auth"].update(issuer="https://evil.invalid"),
        lambda c: c["origin_auth"].update(tunnel_id="other"),
        lambda c: c["tunnel"]["harnesses"][0].update(port=4099),
        lambda c: c["tunnel"]["harnesses"][0].update(hostname="api.radienthq.com"),
        lambda c: c["tunnel"]["harnesses"][0].update(port=True),
        lambda c: c["origin_auth"]["jwks"]["keys"][0].update(d="private-material"),
    ],
)
def test_connection_rejects_unsafe_cloud_metadata(connection, mutate):
    mutate(connection)
    with pytest.raises(ValueError):
        config.validate_connection(connection)


@pytest.mark.asyncio
async def test_billing_requires_current_exact_price_and_no_silent_activation():
    quote = {
        "eligible": False,
        "monthly_price_usd": 3.25,
        "monthly_cost_usd": 0.65,
        "balance_usd": 10,
        "amount_due_usd": 3.25,
    }
    api = AsyncMock()
    api.request.return_value = quote
    for accepted in (None, "3", "NaN"):
        with pytest.raises(ValueError):
            await _ensure_billing(api, accepted)
        assert all(call.args[0] == "GET" for call in api.request.call_args_list)
    api.request.reset_mock()
    api.request.side_effect = [quote, {**quote, "eligible": True}]
    assert (await _ensure_billing(api, "3.25"))["eligible"] is True
    assert api.request.call_args.args == ("POST", "/billing/activate")
    assert api.request.call_args.kwargs["body"] == {"accepted_monthly_price_usd": 3.25}


def test_billing_and_suspension_receipts_link_to_console_management_route():
    quote = {
        "monthly_price_usd": 1,
        "monthly_cost_usd": 0.2,
        "balance_usd": -1,
        "amount_due_usd": 1,
    }
    for receipt in (
        _billing_summary(quote),
        _summary({"id": "fixture", "status": "suspended"}),
    ):
        assert "https://console.radienthq.com/dashboard/tunnels" in receipt
        assert "https://console.radienthq.com/tunnels" not in receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_gateway", [None, 4099])
async def test_create_retry_reuses_intent_and_does_not_persist_connector_token(
    tmp_path, monkeypatch, connection, explicit_gateway
):
    from local_operator.browser_bridge.daemon import DEFAULT_PORT as BROWSER_PORT
    from local_operator.mobile.daemon import DEFAULT_PORT as MOBILE_PORT
    from local_operator.tunnels import cli

    # Default creation must coexist with both already-installed local services.
    # Explicit ports from earlier configurations remain supported unchanged.
    assert config.DEFAULT_GATEWAY_PORT == 4100
    assert config.DEFAULT_GATEWAY_PORT not in {BROWSER_PORT, MOBILE_PORT}
    expected_gateway = explicit_gateway or config.DEFAULT_GATEWAY_PORT
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "credential_id", lambda value=None: 7)
    api = AsyncMock()
    attempts = []

    async def request(method, path="", **kwargs):
        if path == "/billing":
            # The first response is lost after the cloud creates the tunnel.
            # Spending the remaining credit must not strand its saved retry.
            return {"eligible": True, "balance_usd": 0 if attempts else 10}
        assert kwargs["body"]["gateway_port"] == expected_gateway
        attempts.append(kwargs["idempotency_key"])
        if len(attempts) == 1:
            raise httpx.ConnectError("lost create response")
        return {**copy.deepcopy(connection["tunnel"]), "gateway_port": expected_gateway}

    api.request.side_effect = request
    monkeypatch.setattr(cli, "RadientTunnels", lambda *args: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    options = ["--gateway-port", str(explicit_gateway)] if explicit_gateway else []
    args = parser.parse_args(["tunnel", "create", "--name", "device", *options])
    with pytest.raises(httpx.ConnectError):
        await dispatch(args)
    assert "https://" + HOST in await dispatch(args)
    assert attempts[0] == attempts[1]
    stored = config.directory() / "config.json"
    assert stored.stat().st_mode & 0o777 == 0o600
    assert "cloudflared_token" not in stored.read_text()
    assert config.load()["credential_id"] == 7
    assert config.load()["gateway_port"] == expected_gateway


@pytest.mark.asyncio
async def test_tunnel_auth_never_rotates_or_falls_back_to_environment(
    tmp_path, monkeypatch, capsys
):
    from local_operator import cli
    from local_operator.tunnels.api import RadientTunnels, credential_id

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("RADIENT_API_KEY", "must-not-be-used")
    with closing(AuthStore()) as store:
        first = store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "first", "access": "selected-oauth"}
        )
        store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "second", "access": "other-oauth"}
        )
    with pytest.raises(ValueError, match="multiple accounts") as error:
        credential_id()
    assert "lop login-status" in str(error.value)
    # Exercise the advertised command through the real CLI dispatcher against
    # only the fixture store. `login status` instead attempts provider login.
    monkeypatch.setattr(cli, "_build_auth_stack", lambda _: (AuthStore(), None))
    monkeypatch.setattr(cli.sys, "argv", ["lop", "login-status"])
    assert cli.main() == 0
    listing = capsys.readouterr().out
    assert f"[{first.id}] radient" in listing
    assert "selected-oauth" not in listing and "other-oauth" not in listing
    assert credential_id(first.id) == first.id
    seen = []

    def api(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"msg": "ok", "result": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
        assert await RadientTunnels(first.id, client).request("GET") == []
        with closing(AuthStore()) as store:
            store.delete_credentials_for_provider("radient", disabled_cause="logged-out")
        with pytest.raises(ValueError, match="unavailable"):
            await RadientTunnels(first.id, client).request("GET")
    assert seen == ["Bearer selected-oauth"]


@pytest.mark.parametrize("billing", [{"eligible": False}, {"eligible": None}, {}, None, "bad"])
def test_billing_suspension_closes_even_when_record_is_active(billing):
    assert not active({"enabled": True, "status": "active", "billing": billing})
    assert active({"enabled": True, "status": "active", "billing": {"eligible": True}})


def test_concurrent_create_intents_publish_one_complete_private_winner(tmp_path):
    path = tmp_path / "private" / "create.json"
    candidates = [json.dumps({"key": str(uuid.uuid4()), "payload": "x" * 10000}) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda value: config.private_write(path, value, exclusive=True), candidates
            )
        )
    assert sum(results) == 1
    assert path.read_text() == candidates[results.index(True)]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_empty_header_is_not_origin_auth_and_wrong_signing_key_fails(connection, signing_key):
    wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    origin = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            for token in ("", "present-but-unsigned", proof(wrong)):
                assert (
                    await client.get("/api/sessions", headers={PROOF_HEADER: token})
                ).status_code == 401
    origin.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("no_start", [False, True])
async def test_console_connect_prepares_relay_and_starts_service_once(
    tmp_path, monkeypatch, connection, no_start
):
    from local_operator.tunnels import cli, install

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_MOBILE_PASSWORD", raising=False)
    monkeypatch.setattr(cli, "credential_id", lambda value=None: 7)
    api = AsyncMock()
    api.request.side_effect = [
        {"eligible": True, "balance_usd": 10},
        copy.deepcopy(connection["tunnel"]),
    ]
    monkeypatch.setattr(cli, "RadientTunnels", lambda *args: api)
    prepare = Mock()
    installed = Mock()
    monkeypatch.setattr(cli, "_prepare_mobile", prepare)
    monkeypatch.setattr(cli, "cloudflared_binary", lambda _: "/trusted/cloudflared")
    monkeypatch.setattr(install, "install", installed)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = ["tunnel", "connect", "tunnel-1"] + (["--no-start"] if no_start else [])
    receipt = await dispatch(parser.parse_args(args))
    assert HOST in receipt
    assert prepare.call_count == installed.call_count == (0 if no_start else 1)
    assert config.load()["stopped"] is no_start
    if not no_start:
        assert config.load()["cloudflared_path"] == "/trusted/cloudflared"


@pytest.mark.asyncio
async def test_configuration_edit_preserves_private_origin_and_explicit_stop(
    tmp_path, monkeypatch, connection
):
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "credential_id", lambda value=None: 7)
    config.save(
        {
            "credential_id": 7,
            "tunnel_id": "tunnel-1",
            "gateway_port": 4099,
            "stopped": True,
            "cloudflared_path": "/trusted/cloudflared",
            "mobile_password": "private-origin",
            "record": connection["tunnel"],
        }
    )
    api = AsyncMock()
    api.request.return_value = copy.deepcopy(connection["tunnel"])
    monkeypatch.setattr(cli, "RadientTunnels", lambda *args: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipt = await dispatch(parser.parse_args(["tunnel", "configure", "--name", "Renamed"]))
    stored = config.load()
    assert stored["gateway_port"] == 4099
    assert stored["stopped"] is True
    assert stored["mobile_password"] == "private-origin"
    assert stored["cloudflared_path"] == "/trusted/cloudflared"
    assert "private-origin" not in receipt
    assert "mobile_password" not in api.request.call_args.kwargs["body"]
    api.request.reset_mock()
    with pytest.raises(ValueError, match="Gateway port is fixed"):
        await dispatch(parser.parse_args(["tunnel", "configure", "--gateway-port", "4100"]))
    assert all(call.args[0] == "GET" for call in api.request.call_args_list)


@pytest.mark.parametrize("billing_succeeds", [False, True])
def test_mobile_enable_reactivates_existing_tunnel_before_starting(
    tmp_path, monkeypatch, billing_succeeds
):
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save({"tunnel_id": "existing"})
    calls = []

    async def command(args):
        calls.append(args)
        if args.tunnel_command == "configure":
            assert args.remote_enabled is True
            assert args.accept_monthly_price == "0.05"
            if not billing_succeeds:
                raise ValueError("The accepted price differs from the current quote.")
            return "Reactivated"
        return "Service started"

    monkeypatch.setattr(cli, "dispatch", command)
    receipt = cli.mobile_action("enable", "0.05")
    assert [args.tunnel_command for args in calls] == (
        ["configure", "install"] if billing_succeeds else ["configure"]
    )
    assert ("Service started" in receipt) is billing_succeeds


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["billing", "activate", "list"])
async def test_account_wide_commands_accept_and_use_explicit_login(action, monkeypatch):
    from local_operator.tunnels import cli

    select = Mock(return_value=19)
    monkeypatch.setattr(cli, "credential_id", select)
    api = AsyncMock()
    api.request.return_value = (
        []
        if action == "list"
        else {
            "eligible": True,
            "monthly_cost_usd": 0.2,
            "monthly_price_usd": 1,
            "balance_usd": 10,
            "amount_due_usd": 0,
        }
    )
    factory = Mock(return_value=api)
    monkeypatch.setattr(cli, "RadientTunnels", factory)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    argv = ["tunnel", action, "--credential-id", "19"]
    if action == "activate":
        argv += ["--accept-monthly-price", "1"]
    await dispatch(parser.parse_args(argv))
    select.assert_called_once_with(19)
    assert factory.call_args.args[0] == 19


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "connect"])
@pytest.mark.parametrize("balance", [None, False, True, "bad", "NaN", "Infinity", 0, -0.5, -1])
async def test_initial_setup_requires_fresh_positive_credit_before_any_mutation(
    tmp_path, monkeypatch, action, balance
):
    """Even a zero-price eligible subscription does not prove enrollment credit."""
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "credential_id", lambda _: 7)
    api = AsyncMock()
    api.request.return_value = {"eligible": True, "balance_usd": balance}
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    with pytest.raises(ValueError, match="balance above USD 0"):
        await dispatch(parser.parse_args(["tunnel", action]))
    api.request.assert_awaited_once_with("GET", "/billing")
    assert not (config.directory() / "config.json").exists()
    assert not (config.directory() / "create.json").exists()


@pytest.mark.asyncio
async def test_existing_billing_eligibility_preserves_negative_credit_floor():
    api = AsyncMock()
    api.request.return_value = {"eligible": True, "balance_usd": -0.5}
    assert (await _ensure_billing(api, None))["eligible"]
    api.request.assert_awaited_once_with("GET", "/billing")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "balance,eligible,ready", [(5, True, True), (0, True, False), (5, False, False)]
)
async def test_agent_billing_json_is_fresh_owner_pinned_and_allowlisted(
    monkeypatch, balance, eligible, ready
):
    from local_operator.tunnels import cli

    select = Mock(return_value=19)
    monkeypatch.setattr(cli, "credential_id", select)
    api = AsyncMock()
    api.request.return_value = {
        "balance_usd": balance,
        "eligible": eligible,
        "monthly_price_usd": 0,
        "monthly_cost_usd": 0,
        "amount_due_usd": 0,
        "private_future_field": "never-print",
    }
    factory = Mock(return_value=api)
    monkeypatch.setattr(cli, "RadientTunnels", factory)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipt = await dispatch(
        parser.parse_args(["tunnel", "billing", "--credential-id", "19", "--json"])
    )
    data = json.loads(receipt)
    assert data["credential_id"] == 19 and data["account_valid"] is True
    assert data["setup_ready"] is ready
    assert data["positive_balance"] is (balance > 0)
    assert data["billing"]["monthly_price_usd"] == 0
    assert "never-print" not in receipt
    select.assert_called_once_with(19)
    assert factory.call_args.args[0] == 19
    api.request.assert_awaited_once_with("GET", "/billing")


def _stored(connection: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Local config as `lop tunnel connect` writes it: the whole cloud record."""
    value = {
        "tunnel_id": connection["tunnel"]["id"],
        "credential_id": 7,
        "gateway_port": connection["gateway_port"],
        "record": copy.deepcopy(connection["tunnel"]),
    }
    value.update(overrides)
    return value


def _service_fixture(tmp_path, monkeypatch, connection, console_port, pinned_port=4098):
    """Config, stubs, and the mock cloud for a service serving `console_port`.

    Returns `(service, served, api)`: `served` is the connection the stubbed
    /connect hands back (so a test knows the port the real listener binds) and
    `api` is the stub whose `request` a test may re-point at a failing control
    plane. `pinned_port` is what the stored record pins; the default is the
    mobile relay's, so a test that leaves a gateway actually serving must pass a
    synthetic port rather than aim it at the operator's live daemon.
    """
    from local_operator.tunnels import service

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    stored = _stored(connection, gateway_port=free)
    stored["record"]["harnesses"][0]["port"] = pinned_port
    config.save(stored)
    served = copy.deepcopy(connection)
    served["gateway_port"] = served["tunnel"]["gateway_port"] = free
    served["tunnel"]["harnesses"][0]["port"] = console_port
    api = AsyncMock()
    api.request.return_value = served
    monkeypatch.setattr(service, "RadientTunnels", lambda *args: api)
    monkeypatch.setattr(service, "cloudflared_binary", lambda *_: "/trusted/cloudflared")
    monkeypatch.setattr(service, "load_password", lambda: "private-local-password")
    return service, served, api


def _pinned_service(tmp_path, monkeypatch, connection, console_port):
    """Config and stubs for a service whose /connect serves `console_port`."""
    return _service_fixture(tmp_path, monkeypatch, connection, console_port)[0]


class _Connector:
    """A cloudflared stand-in that stays up until the supervisor withdraws it.

    `wait()` returning early would end `run()` immediately, and `terminate()` has
    to release it or `run()`'s cleanup waits forever: a real cloudflared is a
    long-lived child on both counts, so a stub that is not trips the very
    behaviour under test.
    """

    returncode = None

    def __init__(self):
        self._gone = asyncio.Event()

    async def wait(self):
        await self._gone.wait()

    def terminate(self):
        self.returncode = 0
        self._gone.set()

    def kill(self):
        self.terminate()


def _synthetic_port() -> int:
    """A free loopback port, for a harness no test is allowed to reach."""
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _refusal_from_an_unreachable_control_plane() -> ValueError:
    """The shape the tunnel client refuses with when its refresh cannot connect.

    A ValueError with the transport error reachable on its chain: the client's
    own chain has an `AuthStoreError` in between (AuthStore wraps the transport
    failure, `request` chains that), and the test that uses this only needs the
    shape, because the chain is the whole reason a surface can tell this apart
    from an expired login.
    """
    try:
        try:
            raise httpx.ConnectError("network is unreachable")
        except httpx.ConnectError as transport:
            raise ValueError("The tunnel's Radient login could not be refreshed.") from transport
    except ValueError as refusal:
        return refusal


def _refusal_from_a_dead_grant():
    """The token-endpoint answer that means the grant is gone, as an exception.

    Built from the incident's own prose body through the shared raiser, so this
    is the same `InvalidGrantError` a real refused refresh produces rather than a
    hand-made exception shaped like one (review round 1, M1's first defect was
    exactly this body being read as retryable).
    """
    from local_operator.providers.oauth.callback_server import raise_for_refresh_failure

    try:
        raise_for_refresh_failure("Radient", 401, RADIENT_PROSE_REFUSAL)
    except Exception as refusal:  # noqa: BLE001 — the type is the store's business
        return refusal
    raise AssertionError("a prose refusal must raise")


def test_harness_port_change_in_the_console_alone_cannot_repoint_the_tunnel(connection):
    """A3: the console may replace harness ports wholesale, and the gateway
    attaches this device's relay credential to whatever answers on them. Only
    a locally run command may move a harness port."""
    from local_operator.tunnels import service

    stored = _stored(connection)
    moved = copy.deepcopy(connection)
    # 4098 is the mobile relay; 4096 is any other loopback service the console
    # session could aim this harness at to be handed a signed lop_mobile cookie.
    moved["tunnel"]["harnesses"][0]["port"] = 4096
    with pytest.raises(ValueError) as failure:
        service.enforce_harness_ports(moved, stored)
    # Fail-closed refusals must name the remedy: the console PATCH bumps the
    # version, the poller restarts, and without this wording the operator has
    # a permanently dark tunnel and no way to self-diagnose it.
    assert "local-operator" in str(failure.value)
    assert "lop tunnel connect" in str(failure.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("console_port", [4096, 4098])
async def test_service_refuses_to_publish_a_connector_on_an_unpinned_harness_port(
    tmp_path, monkeypatch, connection, console_port
):
    """The pin must be wired into the real supervisor, not merely callable.

    service.run() is the only path a device takes to publish a connector, so
    the guard is only worth anything if a mismatched /connect stops it before
    cloudflared launches. 4098 is the matching (pinned) port and must still
    start; 4096 is the repointed one and must not.
    """
    service = _pinned_service(tmp_path, monkeypatch, connection, console_port)
    launched = AsyncMock(side_effect=AssertionError("connector launched on an unpinned port"))
    monkeypatch.setattr(service.asyncio, "create_subprocess_exec", launched)
    if console_port == 4098:
        # The matching case must get all the way to launching the connector,
        # which is where this stub deliberately stops it.
        with pytest.raises(AssertionError):
            await service.run()
        return
    with pytest.raises(ValueError, match="Run lop tunnel connect again"):
        await service.run()
    launched.assert_not_called()


def test_the_refusal_reaches_the_operator_through_the_supervised_entry_point(
    tmp_path, monkeypatch, connection, capsys
):
    """M1: run() raises the right words, but launchd executes main(), and
    StandardOutPath/StandardErrorPath are service.log — so whatever main()
    prints is the entirety of what the operator can read.

    Asserting on the exception object proves nothing here: the previous round
    did exactly that while main() swallowed the message and sent the operator
    to a Radient login that was fine. This drives the real entry point and
    reads the bytes, and it is the observable docs/tunnels.md promises.
    """
    service = _pinned_service(tmp_path, monkeypatch, connection, 4096)
    launched = AsyncMock(side_effect=AssertionError("connector launched on an unpinned port"))
    monkeypatch.setattr(service.asyncio, "create_subprocess_exec", launched)

    # 0, not 1: this refusal cannot be retried away — a console port change is
    # only repaired by `lop tunnel connect` on this device — so the connector
    # PARKS (exits successfully, which is the one idiom both supervisors read as
    # "do not restart me") and names the remedy once. It used to exit 1, which
    # launchd answered with a fresh attempt every 10 seconds forever.
    assert service.main() == 0

    logged = capsys.readouterr().out
    assert "Harness port for local-operator changed in the console." in logged
    assert "Run lop tunnel connect again." in logged
    # The old generic line sent the operator to two places that are both fine.
    assert "your Radient login" not in logged
    launched.assert_not_called()
    parked = json.loads((tmp_path / "tunnel" / "state.json").read_text())
    assert parked["state"] == "parked"
    assert parked["reason"] == "reenrolment_required"
    assert parked["remedy"]["command"] == "lop tunnel connect"


def test_main_still_withholds_error_text_it_did_not_author(tmp_path, monkeypatch, capsys):
    """The suppression exists so upstream bodies, request URLs and filesystem
    paths stay out of a log the operator may paste into a support thread.
    Surfacing this package's own refusals must not widen that."""
    from local_operator.tunnels import service

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    leaky = httpx.HTTPStatusError(
        "Server error '500' for url 'https://api.radienthq.com/v1/tunnels/t-1?k=SECRET'",
        request=httpx.Request("POST", "https://api.radienthq.com/v1/tunnels/t-1?k=SECRET"),
        response=httpx.Response(500),
    )

    async def fail():
        raise leaky

    monkeypatch.setattr(service, "run", fail)
    assert service.main() == 1
    logged = capsys.readouterr().out
    assert "SECRET" not in logged
    assert "check lop tunnel status and your Radient login" in logged


def test_matching_harness_ports_still_connect_after_pinning(connection):
    """The §3.3 outage guard: over-refusal here strands every legitimate user,
    including one whose harness the console merely disabled or renamed."""
    from local_operator.tunnels import service

    stored = _stored(connection)
    service.enforce_harness_ports(copy.deepcopy(connection), stored)
    # A harness the operator turned off is never dialed, so its port cannot
    # carry a credential and must not cost the whole tunnel its connection.
    disabled = copy.deepcopy(connection)
    disabled["tunnel"]["harnesses"][0].update(enabled=False, port=4096)
    service.enforce_harness_ports(disabled, stored)
    # An added harness the local record predates is still pinned, not admitted.
    added = copy.deepcopy(connection)
    added["tunnel"]["harnesses"].append(
        {"id": "opencode", "enabled": True, "port": 4096, "hostname": "abc123-oc.radienthq.com"}
    )
    with pytest.raises(ValueError):
        service.enforce_harness_ports(added, stored)


def test_a_console_added_harness_is_refused_as_unapproved_not_as_a_port_change(connection):
    """m2: refusing an added harness is right, but calling it a port change
    describes an event that never happened and sends the operator hunting for
    a port they never set. Both refusals name the same remedy."""
    from local_operator.tunnels import service

    stored = _stored(connection)
    added = copy.deepcopy(connection)
    added["tunnel"]["harnesses"].append(
        {"id": "opencode", "enabled": True, "port": 4096, "hostname": "abc123-oc.radienthq.com"}
    )
    with pytest.raises(ValueError) as failure:
        service.enforce_harness_ports(added, stored)
    assert "opencode is not approved on this device" in str(failure.value)
    assert "changed in the console" not in str(failure.value)
    assert "lop tunnel connect" in str(failure.value)


def test_a_harness_listed_twice_in_the_stored_record_disables_the_pin(connection, capsys):
    """m1: two entries claiming one id leave the approved port decided by
    serialization order rather than by the operator, exactly the last-wins
    ambiguity the JWKS loader refuses on a repeated kid. Falling back keeps a
    hand-edited local file from stranding a device the cloud still serves."""
    from local_operator.tunnels import service

    stored = _stored(connection)
    stored["record"]["harnesses"] = [
        {"id": "local-operator", "enabled": True, "port": 4096, "hostname": HOST},
        {"id": "local-operator", "enabled": True, "port": 4098, "hostname": HOST},
    ]
    assert service.pinned_harness_ports(stored)[0] is None
    # Without the guard the second entry wins and 4098 is silently approved.
    served = copy.deepcopy(connection)
    served["tunnel"]["harnesses"][0]["port"] = 4098
    service.enforce_harness_ports(served, stored)
    assert "listed twice" in capsys.readouterr().out


@pytest.mark.parametrize(
    "record,names",
    [
        (None, "predates"),
        ({}, "predates"),
        ({"harnesses": []}, "predates"),
        ({"harnesses": "not-a-list"}, "predates"),
        ({"harnesses": [{"id": "x"}]}, "harness x has an unusable port"),
        ({"harnesses": [{"port": 4098}]}, "entry 1 has no usable id"),
    ],
)
def test_absent_local_harness_record_warns_instead_of_stranding_the_device(
    connection, capsys, record, names
):
    """Upgrade safety: hard-failing on config written before this field existed
    would convert a security fix into a fleet outage nobody can fix remotely.

    n1: one bad entry disables the pin for the whole record, so the warning
    names it — otherwise the operator knows the pin is off but not which line
    of config.json to repair.
    """
    from local_operator.tunnels import service

    stored = _stored(connection)
    if record is None:
        stored.pop("record")
    else:
        stored["record"] = record
    moved = copy.deepcopy(connection)
    moved["tunnel"]["harnesses"][0]["port"] = 4096
    service.enforce_harness_ports(moved, stored)
    warning = capsys.readouterr().out
    assert "harness port pinning is inactive" in warning
    assert names in warning


@pytest.mark.asyncio
async def test_replayed_websocket_upgrade_cannot_open_a_second_channel(connection):
    """A4: a WS upgrade is signed as method="GET" but is not idempotent —
    replaying it yields an extra live bidirectional channel to the harness."""
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute
    from websockets.asyncio.client import connect as ws_connect
    from websockets.exceptions import WebSocketException
    from websockets.typing import Origin

    from tests.unit.test_tunnel_sockets import server

    accepted = []

    async def harness(socket):
        accepted.append(True)
        await socket.accept()
        await socket.send_text("harness open")
        await socket.close()

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    public.update(kid="origin-1", alg="RS256")
    connection = copy.deepcopy(connection)
    connection["origin_auth"]["jwks"]["keys"] = [public]

    async with server(Starlette(routes=[WebSocketRoute("/ws", harness)])) as harness_port:
        connection["tunnel"]["harnesses"][0]["port"] = harness_port
        now = int(time.time())
        # ONE assertion, presented twice inside its 30s window — exactly what a
        # captor of a single upgrade holds. A fresh reconnect carries its own
        # jti (the Worker mints crypto.randomUUID() per request) and is unaffected.
        token = jwt.encode(
            {
                "iss": config.ORIGIN_ISSUER,
                "aud": HOST,
                "sub": "owner-1",
                "tunnel_id": "tunnel-1",
                "harness_id": "local-operator",
                "version": 2,
                "method": "GET",
                "target": "/ws",
                "body_sha256": hashlib.sha256(b"").hexdigest(),
                "iat": now,
                "exp": now + 30,
                "jti": str(uuid.uuid4()),
            },
            key,
            algorithm="RS256",
            headers={"kid": "origin-1"},
        )
        async with httpx.AsyncClient(trust_env=False) as upstream:
            gateway = Gateway(connection, upstream, mobile_password="pw")
            async with server(gateway.app()) as gateway_port:

                async def dial():
                    async with ws_connect(
                        "ws://" + HOST + "/ws",
                        host="127.0.0.1",
                        port=gateway_port,
                        proxy=None,
                        origin=Origin("https://" + HOST),
                        additional_headers={PROOF_HEADER: token},
                    ) as socket:
                        return await socket.recv()

                assert await dial() == "harness open"
                with pytest.raises(WebSocketException):
                    await dial()
    assert accepted == [True]


def test_duplicate_key_identifier_in_the_pinned_jwks_is_rejected(connection, signing_key):
    """Two keys sharing a kid make verification order-dependent; a dict
    comprehension silently lets the last row win. Refuse the whole set."""
    from local_operator.tunnels.gateway import OriginVerifier

    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    shadow = json.loads(RSAAlgorithm.to_jwk(other.public_key()))
    shadow.update(kid="origin-1", alg="RS256")
    connection["origin_auth"]["jwks"]["keys"].append(shadow)
    # config.validate_connection admits it — duplicate kid is a trust-boundary
    # ambiguity the verifier owns, not a schema violation.
    config.validate_connection(connection)
    with pytest.raises(ValueError, match="Duplicate key identifier"):
        OriginVerifier(connection["origin_auth"], Mock())


@pytest.mark.asyncio
async def test_the_relay_names_a_lost_network_instead_of_one_flat_refusal(connection, signing_key):
    """The response a phone gets must name the cause, not only the state.

    Field report: a computer moved onto a network that could not reach the
    control plane, the 30-second lease lapsed, and every request answered the
    same flat "tunnel authorization unavailable" that a revoked tunnel or a
    lapsed plan produces. Nothing said "this computer has no network", so the
    only remedy was to guess — and the guess on offer was to re-enroll a tunnel
    that was healthy. So the network cause and the withdrawn cause are asserted
    separately, and each is asserted not to read as the other.
    """
    from local_operator.tunnels import service

    origin = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(origin)) as upstream:
        gateway = Gateway(connection, upstream, mobile_password="pw")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()), base_url="https://" + HOST
        ) as client:
            gateway.note_authorization_failure(
                service.authorization_failure_reason(httpx.ConnectError("unreachable"))
            )
            gateway.authorized_until = 0
            refused = await client.get("/api/sessions", headers={PROOF_HEADER: proof(signing_key)})
            assert refused.status_code == 503
            body = refused.json()
            # The machine-readable error is unchanged: this names the cause, it
            # does not replace a field anything already keys on.
            assert body["error"] == "tunnel authorization unavailable"
            assert body["reason"] == UNREACHABLE
            # The human sentence leads. A phone renders this body as JSON in a
            # browser with no viewer to fold it, so its reader must meet the cause
            # and the remedy before a generic string and a machine token.
            assert list(body)[0] == "detail"
            assert "network may be down" in body["detail"]

        # `lop tunnel status` reads this payload, so the cause must be on it and
        # not only on the phone's response.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app()),
            base_url=f"http://127.0.0.1:{connection['gateway_port']}",
        ) as probe:
            health = (await probe.get("/_lop_tunnel/health")).json()
        assert health["ok"] is False
        assert health["reason"] == UNREACHABLE
        # No `error` key on a 200: `error` is this gateway's failure-body field
        # everywhere else, so a present one would have a client read success as
        # failure.
        assert "error" not in health

        # A reason describes the lease in force, so a successful renewal clears it
        # rather than leaving a stale cause on a working tunnel.
        gateway.authorize()
        assert gateway.unavailable_body()["reason"] == LEASE_PENDING
        # The state a phone meets most often must not send its reader to a command
        # surface it does not have, and the terminal's line must not send the
        # operator to the command that is printing it.
        assert "lop tunnel" not in RELAY_DETAIL[LEASE_PENDING]
        assert "Run lop tunnel status again shortly" in TERMINAL_DETAIL[LEASE_PENDING]

        # A withdrawn authorization must never read as a network fault: that
        # misdirection is the defect.
        gateway.revoked = True
        withdrawn = gateway.unavailable_body()
        assert withdrawn["reason"] == NOT_AUTHORIZED
        assert "network" not in withdrawn["detail"]
        # And the remedy has to be reachable from the device that was shown it.
        assert CONSOLE_URL in withdrawn["detail"]
    origin.assert_not_called()


@pytest.mark.asyncio
async def test_the_supervisor_reports_a_lost_network_on_both_status_surfaces(
    tmp_path, monkeypatch, connection, signing_key
):
    """What the poller *does* with a failure, through the real supervisor.

    Classifying in a helper proves nothing on its own: poll() is the only place
    the control plane's failure is visible, and /_lop_tunnel/health is exactly
    what `lop tunnel status` prints. This runs the real listener, the real
    gateway and the real renewal loop against a control plane that answers
    /connect and then goes away — the field report's sequence — and reads both
    surfaces the operator and the phone actually have.
    """
    # A synthetic harness port, never the mobile relay's 4098: this is the one
    # test that leaves a gateway actually serving, and a regression in the refusal
    # path would otherwise send a gateway-credentialed request to the operator's
    # live daemon.
    harness_port = _synthetic_port()
    service, served, api = _service_fixture(
        tmp_path, monkeypatch, connection, harness_port, pinned_port=harness_port
    )
    monkeypatch.setattr(service, "POLL_SECONDS", 0.02)
    # The 30-second lease cliff is not what is under test; shorten it so the
    # refusal is observed rather than slept through.
    monkeypatch.setattr("local_operator.tunnels.gateway.AUTHORIZATION_LEASE_SECONDS", 0.05)

    async def control_plane(method, path, **kwargs):
        if method == "POST":
            return served
        raise httpx.ConnectError("network is unreachable")

    api.request.side_effect = control_plane

    monkeypatch.setattr(
        service.asyncio, "create_subprocess_exec", AsyncMock(return_value=_Connector())
    )
    task = asyncio.create_task(service.run())
    async with httpx.AsyncClient(trust_env=False) as client:
        health: dict[str, Any] = {}
        for _ in range(200):
            await asyncio.sleep(0.05)
            reply = await client.get(
                f"http://127.0.0.1:{served['gateway_port']}/_lop_tunnel/health"
            )
            health = reply.json()
            if health.get("reason"):
                break
        assert health.get("ok") is False, health
        assert health["reason"] == "control_plane_unreachable", health
        assert "network" in health["detail"]

        # Reached over the real listener the way the edge reaches it: loopback
        # TCP, public Host. The gateway refuses before proof verification, which
        # is the point — the phone never gets as far as its harness.
        refused = await client.get(
            f"http://127.0.0.1:{served['gateway_port']}/api/sessions",
            headers={"host": HOST, PROOF_HEADER: proof(signing_key)},
        )
        assert refused.status_code == 503
        assert refused.json()["reason"] == "control_plane_unreachable"

    # End it the way an operator's `lop tunnel stop` does, so the supervisor is
    # not left running inside the test session.
    stopped = config.load()
    stopped["stopped"] = True
    config.save(stopped)
    assert await asyncio.wait_for(task, timeout=15) == 0


@pytest.mark.asyncio
async def test_tunnel_status_prints_the_reason_the_relay_is_refusing(
    tmp_path, monkeypatch, connection
):
    """The operator's terminal gets the same cause the phone does.

    A bare state ("stopped") names no remedy, and the remedies differ per
    cause: the network case needs no local command at all, so printing nothing
    is what sends an operator to re-enroll a healthy tunnel.
    """
    from local_operator.tunnels import cli, report

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save(_stored(connection))
    api = AsyncMock()
    api.request.return_value = connection["tunnel"]
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)

    class FakeClient:
        def __init__(self, *args, **kwargs): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kwargs):
            return httpx.Response(
                200,
                json={
                    "ok": False,
                    "connected": True,
                    "reason": UNREACHABLE,
                    # The relay's own (phone) sentence, which this surface must not
                    # print verbatim: a terminal can run the command a phone cannot.
                    "detail": RELAY_DETAIL[UNREACHABLE],
                },
            )

    # The loopback probe moved into `tunnels/report.py`, which is where the
    # shared assembly lives now: patch the module that actually owns the client.
    fake = SimpleNamespace(AsyncClient=FakeClient, HTTPError=httpx.HTTPError)
    monkeypatch.setattr(cli, "httpx", fake)
    monkeypatch.setattr(report, "httpx", fake)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))
    # Not "stopped": the gateway answered and reported `connected: true`, so a
    # state word contradicting it would be the command arguing with its own input.
    assert "Connector: not serving" in receipt
    assert TERMINAL_DETAIL[UNREACHABLE] in receipt
    assert "no local command is needed" in receipt
    assert RELAY_DETAIL[UNREACHABLE] not in receipt


@pytest.mark.asyncio
async def test_tunnel_status_survives_a_foreign_listener_on_the_gateway_port(
    tmp_path, monkeypatch, connection
):
    """A 200 that is not a health payload must not raise out of status.

    Parsing the body before checking what it is made a stale or unrelated
    listener on the gateway port — answering 200 with `null` or a list — an
    AttributeError from a read-only status command.
    """
    from local_operator.tunnels import cli, report

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save(_stored(connection))
    api = AsyncMock()
    api.request.return_value = connection["tunnel"]
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)

    class FakeClient:
        def __init__(self, *args, **kwargs): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kwargs):
            return httpx.Response(200, json=["someone else's server"])

    # The loopback probe moved into `tunnels/report.py`, which is where the
    # shared assembly lives now: patch the module that actually owns the client.
    fake = SimpleNamespace(AsyncClient=FakeClient, HTTPError=httpx.HTTPError)
    monkeypatch.setattr(cli, "httpx", fake)
    monkeypatch.setattr(report, "httpx", fake)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))
    assert "Connector: not serving" in receipt


async def _status_receipt_with_a_stale_credential(
    tmp_path, monkeypatch, connection, refusal: BaseException
) -> str:
    """`lop tunnel status` for a machine whose cloud read fails AND whose login must be re-checked.

    A stale access token, so the verdict has to ASK the token endpoint — without
    one the stored token is served as-is and nothing is attempted, which is the
    case these tests are not about. The cloud read fails separately, so both
    halves of the output are populated.
    """
    from local_operator.providers import auth_store
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "qa",
                "access": "stored-access",
                "refresh": "refresh-token",
                "expires": 1,
            },
        )
    config.save(_stored(connection, credential_id=row.id))

    async def refresh(credentials):  # noqa: ANN001 — the store's own refresh fn
        raise refusal

    monkeypatch.setattr(auth_store.AuthStore, "_refresh_fn", lambda self, provider: refresh)
    api = AsyncMock()
    api.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    return await dispatch(parser.parse_args(["tunnel", "status"]))


@pytest.mark.asyncio
async def test_tunnel_status_separates_a_network_fault_from_an_unusable_login(
    tmp_path, monkeypatch, connection
):
    """One shared line sent both causes to /login radient — the LOGIN line tells them apart.

    They are different jobs for the operator — one is a connection, the other an
    interactive login — and the cloud read is unavailable for both, so the status
    command is the only surface that can. Since round 1 the distinction lives in
    ONE place rather than two: the CLOUD line states provenance and no cause at all
    (D7 — a refusal verdict printed by a command that had just said it could not
    read the cloud, in the vocabulary that had already answered on line 1), and
    this device's credential store answers the question the operator is asking.
    `cloud.reason` still carries the cause as data.
    """
    receipt = await _status_receipt_with_a_stale_credential(
        tmp_path, monkeypatch, connection, httpx.ConnectError("network is unreachable")
    )

    assert "Login: could not be checked (a refresh could not reach Radient)." in receipt
    assert "sign-in expired" not in receipt
    # A cached read must never read as a live one, and the line that explains it
    # states provenance rather than a cause it cannot know.
    assert "Status: active (cached — cloud read failed)" in receipt
    assert "Cloud status: unavailable — showing the record stored at the last connect." in receipt
    assert TERMINAL_DETAIL[REFUSED] not in receipt


@pytest.mark.asyncio
async def test_tunnel_status_names_a_refused_grant_as_a_dead_login(
    tmp_path, monkeypatch, connection
):
    """The other half of the same split: the grant itself was refused.

    The incident's own prose body (not an RFC 6749 code), through the shared
    raiser, so this is the answer Radient really gives for a revoked refresh
    token — and it has to arrive as "sign-in expired", which is the one reading
    that tells the operator to do something.
    """
    receipt = await _status_receipt_with_a_stale_credential(
        tmp_path, monkeypatch, connection, _refusal_from_a_dead_grant()
    )

    assert "Login: sign-in expired — run lop login radient" in receipt
    assert "could not be checked" not in receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale,reply,expected",
    [
        # Stale token, and the refresh cannot reach the token endpoint: the
        # computer's network.
        (True, None, UNREACHABLE),
        # A usable token and an answer from Radient that refuses the request.
        (False, httpx.Response(401, json={"error": "invalid_token"}), REFUSED),
    ],
)
async def test_the_real_connector_client_classifies_its_own_failures(
    tmp_path, monkeypatch, connection, stale, reply, expected
):
    """The split is only real if the real client raises what it assumes.

    `authorization_failure_reason` reads exception types, so handing it a
    hand-made httpx error proves nothing about the tunnel client — and the case
    that first shipped broken was the refresh one: AuthStore wraps a transport
    failure to the token endpoint in an `AuthStoreError`, so the cause has to be
    read off the chain. This drives a real `RadientTunnels.request` through the
    real AuthStore and classifies whatever actually comes out.
    """
    from local_operator.providers import auth_store
    from local_operator.tunnels import service
    from local_operator.tunnels.api import RadientTunnels

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(AuthStore()) as store:
        credentials: dict[str, Any] = {
            "type": "oauth",
            "account_id": "only",
            "access": "stored-access",
            "refresh": "refresh-token",
        }
        if stale:
            # A past expiry forces the refresh path an offline machine takes;
            # without one the stored access token is used as-is and no refresh is
            # attempted at all.
            credentials["expires"] = 1
        row = store.upsert_credential("radient", credentials)

    async def refresh(*args, **kwargs):
        raise httpx.ConnectError("network is unreachable")

    monkeypatch.setattr(auth_store.AuthStore, "_refresh_fn", lambda self, provider: refresh)

    reached: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        reached.append(request.url.path)
        assert reply is not None, "the tunnel API is unreachable when the refresh failed"
        return reply

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        with pytest.raises(ValueError) as failure:
            await RadientTunnels(row.id, client).request("GET")

    assert service.authorization_failure_reason(failure.value) == expected
    if stale:
        # Nothing reached the tunnel API: the failure was in reaching Radient at
        # all, which is the entire distinction. A message blaming the login is
        # what sent an offline computer to /login radient.
        assert reached == []
        assert "log in again" not in str(failure.value)


@pytest.mark.asyncio
async def test_a_dead_poller_restarts_the_unit_instead_of_serving_a_stale_lease(
    tmp_path, monkeypatch, connection
):
    """A poller that dies must not leave a gateway that keeps answering.

    The escape is pre-existing — an exception outside (ValueError, httpx) ends the
    poll task silently — but naming the refusal is what makes it dangerous: the
    gateway would report `authorization_lease_pending`, a state whose own copy
    promises it clears itself in seconds, for as long as the process lived. `run()`
    returning 1 is what tells the supervisor to restart the unit.
    """
    port = _synthetic_port()
    service, served, api = _service_fixture(
        tmp_path, monkeypatch, connection, port, pinned_port=port
    )

    async def control_plane(method, path, **kwargs):
        if method == "POST":
            return served
        raise sqlite3.OperationalError("database is locked")

    api.request.side_effect = control_plane
    monkeypatch.setattr(
        service.asyncio, "create_subprocess_exec", AsyncMock(return_value=_Connector())
    )
    assert await asyncio.wait_for(asyncio.create_task(service.run()), timeout=15) == 1


# ---------------------------------------------------------------------------
# A dead login PARKS the connector instead of restarting it forever
# ---------------------------------------------------------------------------

#: The body the operator's own token endpoint returned for a revoked refresh
#: token, verbatim from the incident. Prose in the `error` field rather than an
#: RFC 6749 code -- the shape every code-only rule reads as retryable.
RADIENT_PROSE_REFUSAL = '{"error": "Token refresh failed: refresh token is expired or revoked"}'


def _free_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _dead_grant_service(tmp_path, monkeypatch, connection, *, stopped: bool = False):
    """A connector whose Radient refresh is refused by the REAL provider flow.

    Returns `(service, credential_id)`. The refresh POSTs to a stub token
    endpoint that answers the incident's own 401, so `main()` classifies the
    failure the operator's machine actually produced rather than a hand-made
    exception shaped like it — the distinction the earlier round of this work
    (M1) had to be fixed for once already.
    """
    from local_operator.providers import auth_store
    from local_operator.providers.oauth.radient import refresh_radient_token
    from local_operator.tunnels import service

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(service, "cloudflared_binary", lambda *_: "/trusted/cloudflared")
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "qa",
                "access": "stale-access",
                "refresh": "revoked-refresh",
                # A past expiry is what makes the store refresh at all: without
                # one the stored token is served as-is and nothing is refused.
                "expires": 1,
            },
        )
    stored = _stored(connection, credential_id=row.id, gateway_port=_free_port())
    stored["stopped"] = stopped
    config.save(stored)

    async def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=RADIENT_PROSE_REFUSAL)

    async def refresh(credentials):
        async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
            return await refresh_radient_token(credentials, http_client=client)

    monkeypatch.setattr(auth_store.AuthStore, "_refresh_fn", lambda self, provider: refresh)
    return service, row.id


def test_classify_failure_places_every_failure_on_the_retry_or_stop_side() -> None:
    """The one decision this change makes, in one table.

    A retry can only help where the fault is remote and temporary; everything a
    person has to act on parks. The `transient` default is deliberate and
    asymmetric — parking WITHDRAWS remote access — so an unclear failure keeps
    today's retrying behaviour.
    """
    from local_operator.providers.auth_store import CredentialInvalidError
    from local_operator.tunnels import service
    from local_operator.tunnels.errors import (
        LocalPrerequisite,
        LoginRequired,
        ReenrolmentRequired,
    )

    cases = [
        ("terminal_login", LoginRequired("login unavailable"), LOGIN_REQUIRED),
        ("terminal_login", CredentialInvalidError("grant is dead"), LOGIN_REQUIRED),
        ("terminal_config", LocalPrerequisite("Install cloudflared"), "local_prerequisite"),
        (
            "terminal_remote",
            ReenrolmentRequired("Gateway port changed in the console."),
            "reenrolment_required",
        ),
        ("transient", httpx.ConnectTimeout("no answer"), UNREACHABLE),
        (
            "transient",
            ValueError("Tunnel disabled or billing suspended."),
            REFUSED,
        ),
        ("transient", sqlite3.OperationalError("database is locked"), REFUSED),
    ]
    for kind, failure, reason in cases:
        verdict = service.classify_failure(failure)
        assert verdict.kind == kind, failure
        assert verdict.reason == reason, failure
    # A dead grant is recognised even when a wrapper is the outermost type:
    # `tunnels/api.py` raises `LoginRequired` for it, and this is the chain.
    try:
        raise ValueError("The tunnel's Radient login could not be refreshed.") from (
            CredentialInvalidError("grant is dead")
        )
    except ValueError as wrapped:
        assert service.classify_failure(wrapped).kind == "terminal_login"
    # A transport fault outranks a dead grant: a refresh that could not REACH
    # the endpoint has not been told anything about the grant.
    try:
        try:
            raise httpx.ConnectError("network is unreachable")
        except httpx.ConnectError as transport:
            raise CredentialInvalidError("ambiguous") from transport
    except CredentialInvalidError as both:
        assert service.classify_failure(both).kind == "transient"


def test_a_real_radient_prose_refusal_parks_instead_of_looping(
    tmp_path, monkeypatch, connection, capsys
):
    """The incident, at the supervised entry point.

    Before: this failure printed the same sentence and returned 1, and launchd's
    `KeepAlive{SuccessfulExit:false}` restarted the connector every 10 seconds
    (measured: 870 identical lines, `runs = 633`). The assertions that matter
    are the exit code — 0 is the only value both supervisors honour as "do not
    retry me" — and the state file, because the phone and every terminal read
    the park from there.
    """
    service, credential = _dead_grant_service(tmp_path, monkeypatch, connection)

    # SYNCHRONOUS, and `main()` rather than `run()`: main() is what launchd
    # executes, and it owns the event loop whose exit code the supervisor reads.
    assert service.main() == 0

    logged = capsys.readouterr().out
    assert "connector parked reason=login_required" in logged
    # Timestamped and self-describing: the incident's 870 lines carried neither
    # a time nor which process authored them.
    assert datetime.date.today().isoformat() in logged[:32]
    assert "local_operator.tunnels.service" in logged
    # The remedy sentence comes from the vocabulary the phone's 503 uses, so one
    # cause cannot be worded two ways.
    assert TERMINAL_DETAIL[LOGIN_REQUIRED] in logged
    # Redaction: the provider's body never reaches the log, only its class.
    assert "refresh token is expired or revoked" not in logged

    parked = state.read()
    assert parked is not None
    assert parked["state"] == "parked"
    assert parked["reason"] == LOGIN_REQUIRED
    assert parked["credential_id"] == credential
    assert parked["attempts"] == 1
    assert parked["remedy"]["command"] == "lop login radient"
    assert parked["remedy"]["url"] == CONSOLE_URL


def test_a_transient_failure_keeps_the_retry_and_records_no_park(
    tmp_path, monkeypatch, connection, capsys
):
    """The other half of the split, and the reason it is not "park everything".

    A network fault clears itself at the supervisor's documented 10-second
    floor, so it must keep exiting 1 — and it must NOT write a park, because
    every surface reads a park as "a person is needed here".
    """
    from local_operator.tunnels import service

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(service, "cloudflared_binary", lambda *_: "/trusted/cloudflared")
    config.save(_stored(connection, gateway_port=_free_port()))
    api = AsyncMock()
    api.request.side_effect = _refusal_from_an_unreachable_control_plane()
    monkeypatch.setattr(service, "RadientTunnels", lambda *args: api)
    # An OLD park, from before the login was fixed. Exiting 1 says the connector
    # is retrying, so it is not parked any more: leaving this file behind would
    # have every surface nagging about a login for a machine that is already
    # trying on its own.
    state.mark_parked(reason=LOGIN_REQUIRED, detail="dead grant", credential_id=7)

    assert service.main() == 1
    assert state.read() is None, "a retrying connector is not a parked one"
    logged = capsys.readouterr().out
    assert "reason=control_plane_unreachable" in logged
    assert "park withdrawn (the connector is retrying)" in logged


def test_a_park_is_private_and_announced_once_per_transition(tmp_path, monkeypatch) -> None:
    """The rate limit lives in the state file, because the process is what a
    restart loop throws away — and the file it lands in is 0600 inside 0700."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    assert state.mark_parked(reason=LOGIN_REQUIRED, detail="first", now=1_000) is True
    # A repeat of the same cause is not news. This is what stops a supervisor
    # that does reload the unit from refilling the log: 870 lines is the defect.
    assert state.mark_parked(reason=LOGIN_REQUIRED, detail="first", now=1_001) is False
    # …until the heartbeat terms elapse, so a long-lived park still shows a
    # recent line rather than going silent for hours.
    assert state.mark_parked(reason=LOGIN_REQUIRED, detail="first", now=1_000 + 900) is True
    record = state.read()
    assert record is not None
    assert record["attempts"] == 3
    assert record["first_at"] == 1_000, "the park's own age survives re-stating it"
    assert record["logged_at"] == 1_900

    # The attempts term is independent of the clock.
    for step in range(2, 51):
        state.mark_parked(reason=LOGIN_REQUIRED, detail="first", now=1_900 + step)
    assert state.mark_parked(reason=LOGIN_REQUIRED, detail="first", now=1_952) is True

    # A DIFFERENT cause is a transition, whatever the clock says.
    assert state.mark_parked(reason="local_prerequisite", detail="x", now=1_953) is True
    record = state.read()
    assert record is not None and record["attempts"] == 1

    path = state.path()
    assert path.stat().st_mode & 0o077 == 0
    assert path.parent.stat().st_mode & 0o077 == 0
    state.clear()
    assert state.read() is None


def test_a_fixed_login_re_arms_the_parked_connector(tmp_path, monkeypatch, connection) -> None:
    """Park → sign in → the connector comes back by itself.

    The whole reason a park is safe: nothing else notices, because the connector
    exited successfully and its supervisor is, correctly, not retrying it. This
    drives the real hook — a credential write through the store — with the
    guards' two decisions stubbed (`service_path`, the real-home check) and
    `launchctl` recorded rather than run, because a test suite must never reach
    the operator's own launchd.

    SYNCHRONOUS, deliberately: with no event loop on the calling thread the hook
    runs inline, so every `kicks` assertion below is about the decision the
    guard made and nothing else. That branch is real (a script or a plain
    non-async caller takes it) and it is the branch these guards live on; the
    loop-taking branch, and the reason it exists (review round 1, m3), is
    covered by `test_the_rearm_never_blocks_the_event_loop` below.
    """
    from local_operator.providers.auth_store import AuthStore as Store
    from local_operator.tunnels import install

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_TUNNEL_NO_REARM", raising=False)
    plist = tmp_path / "com.local-operator.tunnel.plist"
    plist.write_text("plist")
    # The supervisor layer, not `sys.platform`: that is the one way to ask
    # which host supervises this service, and it is what `action()` uses.
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: install.supervisors.LAUNCHCTL)
    monkeypatch.setattr(install, "service_path", lambda: plist)
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: True)
    kicks: list[tuple[str, ...]] = []

    def launchctl(*args: str):
        kicks.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(install, "_launchctl", launchctl)
    config.save(_stored(connection, credential_id=7, gateway_port=_free_port()))
    state.mark_parked(reason=LOGIN_REQUIRED, detail="x", credential_id=7)

    # The login this tunnel owns, written again: the guard is the credential ID,
    # so the row has to BE row 7, and re-writing the tunnel's own identity is how
    # a re-login of the same account lands here.
    with closing(Store()) as store:
        for index in range(6):
            store.upsert_credential(
                "radient",
                {"type": "oauth", "account_id": f"filler-{index}", "access": "a", "refresh": "r"},
            )
        rows = {}
        for index, account in enumerate(("qa", "someone-else")):
            rows[account] = store.upsert_credential(
                "radient", {"type": "oauth", "account_id": account, "access": "a", "refresh": "r"}
            ).id
    assert rows == {"qa": 7, "someone-else": 8}, "the fixture's ids are load-bearing"
    assert kicks == [
        ("kickstart", "-k", f"gui/{install.os.getuid()}/com.local-operator.tunnel")
    ], "only the row the tunnel owns may start it"

    # An unrelated provider's login, and a DIFFERENT radient account, touch
    # nothing: the guard is the credential id AND the provider, not the fact that
    # some credential was written. That is what makes this hook safe on a path
    # every writer process runs.
    with closing(Store()) as store:
        store.upsert_credential("anthropic", {"type": "oauth", "access": "a", "refresh": "r"})
        store.upsert_credential(
            "radient",
            {"type": "oauth", "account_id": "someone-else", "access": "a", "refresh": "r"},
        )
    assert len(kicks) == 1

    # And a tunnel the operator deliberately stopped is not waiting on anyone.
    value = config.load()
    value["stopped"] = True
    config.save(value)
    with closing(Store()) as store:
        store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "qa", "access": "a", "refresh": "r"}
        )
    assert len(kicks) == 1

    # The explicit test opt-out leaves the same parked state alone.
    monkeypatch.setenv("LOP_TUNNEL_NO_REARM", "1")
    value["stopped"] = False
    config.save(value)
    with closing(Store()) as store:
        store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "qa", "access": "a", "refresh": "r"}
        )
    assert len(kicks) == 1

    # `lop tunnel status` is the same self-heal for a login fixed elsewhere.
    monkeypatch.delenv("LOP_TUNNEL_NO_REARM")
    assert "starting again" in install.rearm_if_parked(provider="radient", credential_id=7)
    assert len(kicks) == 2


@pytest.mark.asyncio
async def test_the_rearm_never_blocks_the_event_loop(tmp_path, monkeypatch, connection) -> None:
    """The hook's blocking supervisor call runs OFF the loop (review round 1, m3).

    Measured rather than asserted structurally. The stubbed hook blocks on an
    event and records the thread it ran on, and the credential write has to
    RETURN while that block is still in place — which is the finding itself: the
    real hook ends in a `launchctl kickstart` / `systemctl start` with a
    20-second timeout, and this write path is reached from the desktop login
    route, so a hung service manager used to stall every request the server was
    serving. The thread identity is asserted too, because "the write returned"
    would also be true of a call that merely awaited something.

    Written so a REGRESSION cannot hang the suite: the stub's own wait is
    bounded, so on the pre-fix code the write blocks for that bound and the
    second assertion is what fails — with its reason in the message.
    """
    import threading

    from local_operator.tunnels import install

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    entered = threading.Event()
    released = threading.Event()
    finished = threading.Event()
    threads: list[int] = []

    def blocking_rearm(*, provider: str, credential_id: int) -> str:
        threads.append(threading.get_ident())
        entered.set()
        released.wait(2)
        finished.set()
        return "The Radient tunnel connector is starting again."

    monkeypatch.setattr(install, "rearm_if_parked", blocking_rearm)
    with closing(AuthStore()) as store:
        store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "qa", "access": "a", "refresh": "r"}
        )
        assert entered.wait(2), "the hook never ran at all"
        assert not finished.is_set(), "the credential write waited for the hook"
        assert threads and threads[0] != threading.get_ident(), "the hook ran on the loop"
    # Released before the loop closes, so the executor thread is not left in a
    # wait the suite would then join on.
    released.set()
    assert finished.wait(2), "the hook never completed once released"


def test_rearm_declines_everything_that_is_not_this_tunnel(
    tmp_path, monkeypatch, connection
) -> None:
    """Each guard, on its own, with the launchd seam stubbed out.

    These are the conditions under which a credential write must NOT start a
    supervised service. The first guard is the cheap one (nothing is parked, so
    nothing is read), which is the case every ordinary login takes.
    """
    from local_operator.tunnels import install

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_TUNNEL_NO_REARM", raising=False)
    plist = tmp_path / "com.local-operator.tunnel.plist"
    plist.write_text("plist")
    # The supervisor layer, not `sys.platform`: that is the one way to ask
    # which host supervises this service, and it is what `action()` uses.
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: install.supervisors.LAUNCHCTL)
    monkeypatch.setattr(install, "service_path", lambda: plist)
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: True)
    kicks: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *args: kicks.append(args) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    config.save(_stored(connection, credential_id=7, gateway_port=4100))
    assert install.rearm_if_parked(provider="radient", credential_id=7) == ""
    assert kicks == [], "nothing is parked, so nothing may be started"

    # Parked, right credential — but for something other than the login.
    state.mark_parked(reason="local_prerequisite", detail="Install cloudflared")
    assert install.rearm_if_parked(provider="radient", credential_id=7) == ""
    assert kicks == []

    # Parked for the login, wrong credential: the login that was just written is
    # not the one this tunnel owns, so starting it would only park it again.
    state.mark_parked(reason=LOGIN_REQUIRED, detail="x", credential_id=7)
    assert install.rearm_if_parked(provider="anthropic", credential_id=7) == ""
    assert install.rearm_if_parked(provider="radient", credential_id=99) == ""
    assert kicks == []

    # A unit that is not installed cannot be started.
    plist.unlink()
    assert install.rearm_if_parked(provider="radient", credential_id=7) == ""
    assert kicks == []

    # Something answered but refused: reported, never raised, because a login
    # must not fail over a service that could not be started.
    plist.write_text("plist")
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *args: SimpleNamespace(returncode=1, stdout="", stderr="no such service"),
    )
    assert "could not be restarted" in install.rearm_if_parked(provider="radient", credential_id=7)


@pytest.mark.asyncio
async def test_status_describes_a_config_with_no_record_key(
    tmp_path, monkeypatch, connection
) -> None:
    """A store that parses but carries no `record` is still describable (n2).

    The status path indexed the key directly, so a hand-edited `config.json`
    raised a bare `KeyError: 'record'` rendered as a stack trace instead of
    describing the tunnel it could still see — and `_summary` already defaults
    every field it reads, so there was nothing for the hard index to protect.
    `report.local_payload` reads the same key off the same file the same way
    (`value.get("record")`, empty when absent or not a dict), which is what makes
    this a fix rather than a new opinion.

    The port is free rather than 4099: the status path probes the gateway when
    nothing is parked, and the fixture's own default is the port the operator's
    live connector listens on.
    """
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    stored = _stored(connection, gateway_port=free)
    del stored["record"]
    config.save(stored)
    api = AsyncMock()
    api.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())

    out = await dispatch(parser.parse_args(["tunnel", "status"]))
    assert "Tunnel: not created" in out
    assert "Status: configured" in out

    # The machine surface agrees, and answers the same shape the desktop route
    # serves for a store in this state.
    payload = json.loads(await dispatch(parser.parse_args(["tunnel", "status", "--json"])))
    assert payload["tunnel_id"] is None
    assert payload["cloud"]["status"] == "configured"


@pytest.mark.asyncio
async def test_status_json_reports_the_park_the_login_and_the_remedy(
    tmp_path, monkeypatch, connection
) -> None:
    """The machine-readable shape the desktop route also serves."""
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save(_stored(connection))
    state.mark_parked(
        reason=LOGIN_REQUIRED,
        detail=TERMINAL_DETAIL[LOGIN_REQUIRED],
        credential_id=7,
        now=1_800_000_000,
    )
    api = AsyncMock()
    api.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())

    payload = json.loads(await dispatch(parser.parse_args(["tunnel", "status", "--json"])))

    assert payload["tunnel_id"] == connection["tunnel"]["id"]
    assert payload["cloud"]["source"] == "cached"
    assert payload["cloud"]["reason"] == UNREACHABLE
    # The park outranks the probe: the gateway is not running because the
    # connector parked, and "stopped" would describe that as an accident.
    assert payload["connector"]["state"] == "parked"
    assert payload["connector"]["reason"] == LOGIN_REQUIRED
    assert payload["connector"]["since"] == 1_800_000_000
    # No credential row exists for this config's id, which is itself a reason to
    # sign in — and a local answer, which is the only answer available here.
    assert payload["login"]["state"] == "login_required"
    assert payload["remedy"]["command"] == "lop login radient"


@pytest.mark.asyncio
async def test_status_json_says_ok_for_a_usable_login_and_offers_no_remedy(
    tmp_path, monkeypatch, connection
) -> None:
    """The positive case is in the machine shape too, and the negative claims
    are absent from it: a consumer must be able to tell "checked, fine" from
    "could not check"."""
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "qa",
                "access": "live-access",
                "refresh": "live-refresh",
            },
        )
    config.save(_stored(connection, credential_id=row.id))
    api = AsyncMock()
    api.request.return_value = connection["tunnel"]
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())

    payload = json.loads(await dispatch(parser.parse_args(["tunnel", "status", "--json"])))

    assert payload["login"] == {"credential_id": row.id, "state": "ok"}
    assert payload["cloud"]["source"] == "live"
    assert payload["remedy"] is None
    human = await dispatch(parser.parse_args(["tunnel", "status"]))
    assert "Login:" not in human


@pytest.mark.asyncio
async def test_status_names_a_dead_login_even_while_radient_is_unreachable(
    tmp_path, monkeypatch, connection
):
    """The incident's exact combination, which no surface could describe.

    The access token was still unexpired while the grant behind it was revoked,
    so the cloud read failed AND the local check was the only thing that could
    say why. Two lines have to appear: the state, and the credential, which is the
    one fact here that is decided on this device. The network's own cause does NOT
    (review round 1, D7): the sentence that used to carry it was a relay verdict
    printed by a command that had just said it could not read the cloud, so it now
    travels as DATA — which this asserts too, because a cause dropped from both
    surfaces would be a defect rather than a fix.
    """
    from local_operator.providers import auth_store
    from local_operator.providers.oauth.callback_server import InvalidGrantError
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "qa",
                "access": "unexpired-but-revoked",
                "refresh": "revoked",
                "expires": 1,
            },
        )
    config.save(_stored(connection, credential_id=row.id))

    async def refresh(credentials):
        raise InvalidGrantError("Radient refresh failed: HTTP 401 " + RADIENT_PROSE_REFUSAL)

    monkeypatch.setattr(auth_store.AuthStore, "_refresh_fn", lambda self, provider: refresh)
    api = AsyncMock()
    api.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())

    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))

    assert "Login: sign-in expired — run lop login radient" in receipt
    assert TERMINAL_DETAIL[UNREACHABLE] not in receipt
    assert "Cloud status: unavailable — showing the record stored at the last connect." in receipt
    assert "Status: active (cached — cloud read failed)" in receipt

    payload = json.loads(await dispatch(parser.parse_args(["tunnel", "status", "--json"])))
    assert payload["cloud"] == {"status": "active", "source": "cached", "reason": UNREACHABLE}
    assert payload["login"] == {"credential_id": row.id, "state": "login_required"}


def test_the_shared_park_sentence_names_no_command() -> None:
    """D1/M2: the sentence travels to surfaces that cannot run a command.

    It is written into `state.json`, printed by `lop tunnel status`, forwarded to
    the desktop as `connector.detail` and rendered by the TUI card, so a command
    baked into it is a command at least one of those readers cannot run — which is
    exactly what shipped: `/login radient` on a shell surface and in a desktop
    payload. Each surface appends `TERMINAL_REMEDY` in its own spelling instead,
    and this is the assertion that keeps the shared copy command-free.
    """
    from local_operator.tunnels import gateway

    sentence = gateway.TERMINAL_DETAIL[gateway.LOGIN_REQUIRED]
    assert "signing in again starts it again on its own" in sentence.lower()
    assert gateway.CONSOLE_URL not in sentence, "the terminal prints a billing block of its own"
    # No surface-specific spelling in the two entries this finding covers: not the
    # composer's slash form, and not a shell command either. (The lease-pending
    # entry is excluded on purpose, and its own note above says why: its advice is
    # a forward pointer for the terminal printing it, never a park's sentence.)
    for code in (gateway.LOGIN_REQUIRED, gateway.REFUSED):
        detail = gateway.TERMINAL_DETAIL[code]
        assert "/login" not in detail, code
        assert "lop " not in detail, code
    assert gateway.TERMINAL_DETAIL[gateway.REFUSED].count("Signing in again") == 1
    # ...while the command itself still exists as a VALUE, which is what the CLI's
    # `Login:` line, the TUI card and the park file each render their own way.
    assert gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED] == "lop login radient"


@pytest.mark.asyncio
async def test_the_status_first_line_is_a_state_line_not_a_paragraph(
    tmp_path, monkeypatch, connection
):
    """D2: state → remedy → provenance, one fact per line, one command in all of it.

    Measured before: the first line was **317 cells** at 80 columns — five
    rendered rows whose actionable clause sat on the third — and the sign-in
    advice appeared again on the `Login:` line and a third time in the cloud
    block, with a billing URL welded into two of them (D2/D8). The numbers here
    are the whole finding, so they are the assertions.
    """
    from local_operator.tunnels import cli, gateway

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save(_stored(connection))
    state.mark_parked(
        reason=LOGIN_REQUIRED, detail=gateway.TERMINAL_DETAIL[LOGIN_REQUIRED], credential_id=7
    )
    api = AsyncMock()
    api.request.return_value = connection["tunnel"]
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())

    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))
    lines = receipt.splitlines()

    assert lines[0].startswith("Connector: parked — login required")
    assert len(lines[0]) <= 80, lines[0]
    assert lines[1].startswith("  "), "the park's own sentence is a continuation row"
    assert lines[2].startswith("Login: sign-in expired — run lop login radient")
    # The command appears ONCE, and the billing URL is not welded to the login
    # copy: the terminal has a billing block for the line that is about billing.
    assert receipt.count(gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED]) == 1
    assert receipt.count(gateway.CONSOLE_URL) <= 1


@pytest.mark.asyncio
async def test_a_stopped_tunnel_is_never_told_to_sign_in(tmp_path, monkeypatch, connection):
    """D5: the text surface honours `stopped`, as `--json` already did.

    A deliberately stopped tunnel is one the operator is not using, and the
    machine-readable surface says so (`remedy` is null). The human one used to
    print `Login: ... run lop login radient` and a cloud block that ended in the
    same instruction — a nag about remote access they switched off, which is the
    one thing `report.remedy()`'s own comment rules out.
    """
    from local_operator.tunnels import cli, gateway

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save(_stored(connection, stopped=True))
    api = AsyncMock()
    api.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())

    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))

    # The fact stays, with its reason attached…
    assert "Login: sign-in expired (not in use — tunnel stopped)" in receipt
    # …and nothing in the output asks for an action.
    assert gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED] not in receipt
    assert "Cloud status:" not in receipt

    payload = json.loads(await dispatch(parser.parse_args(["tunnel", "status", "--json"])))
    assert payload["remedy"] is None
    assert payload["connector"]["state"] == "stopped"


@pytest.mark.asyncio
async def test_a_successful_connect_withdraws_the_park_and_says_so(
    tmp_path, monkeypatch, connection, capsys
):
    """The other end of the round trip: the connector comes back and the park goes.

    Every surface reads the state file as the truth about a process none of them
    can see, so a park that outlived its condition would have the terminal, `lop
    tunnel status` and the desktop route all describing a connector that is
    serving. Driven through `run()` itself — a stubbed `/connect` is the only
    substitution, exactly as the other service tests do it.
    """
    port = _synthetic_port()
    service, _served, _api = _service_fixture(
        tmp_path, monkeypatch, connection, port, pinned_port=port
    )
    # A park left by an earlier attempt, whose reason is now fixed.
    state.mark_parked(reason=LOGIN_REQUIRED, detail="dead grant", credential_id=7)
    assert state.parked() is not None
    # `run()` serves until something stops it, so the test stops it the way a
    # real shutdown arrives: cloudflared's child exits.
    connector = _Connector()
    monkeypatch.setattr(
        service.asyncio, "create_subprocess_exec", AsyncMock(return_value=connector)
    )

    async def stop_soon() -> None:
        await asyncio.sleep(0.3)
        connector.terminate()

    stopper = asyncio.create_task(stop_soon())
    try:
        assert await asyncio.wait_for(asyncio.create_task(service.run()), timeout=15) == 1
    finally:
        await stopper

    assert state.read() is None, "a serving connector must not leave a park behind"
    assert "park withdrawn" in capsys.readouterr().out


def test_rearm_refuses_a_configuration_with_no_usable_credential_id(
    tmp_path, monkeypatch, connection
) -> None:
    """The ownership guard may not match on two `None`s.

    A hand-edited `config.json` that lost its `credential_id` would otherwise
    compare equal to a caller passing nothing, and the guard's whole job is the
    claim "this is the credential THIS tunnel owns".
    """
    from local_operator.tunnels import install

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_TUNNEL_NO_REARM", raising=False)
    plist = tmp_path / "com.local-operator.tunnel.plist"
    plist.write_text("plist")
    # The supervisor layer, not `sys.platform`: that is the one way to ask
    # which host supervises this service, and it is what `action()` uses.
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: install.supervisors.LAUNCHCTL)
    monkeypatch.setattr(install, "service_path", lambda: plist)
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: True)
    kicks: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *args: kicks.append(args) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    stored = _stored(connection, gateway_port=_free_port())
    del stored["credential_id"]
    config.save(stored)
    state.mark_parked(reason=LOGIN_REQUIRED, detail="x")

    assert install.rearm_if_parked(provider="radient", credential_id=0) == ""
    assert kicks == []


def test_rearm_starts_the_windows_task_too(tmp_path, monkeypatch, connection) -> None:
    """The same re-arm on the supervisor the product now supports on Windows.

    The base this branch was rebased onto taught the four daemons to run off
    macOS (a systemd user unit, a Task Scheduler task), so a park can now happen
    on a host this hook must be able to end. The call it makes is the one
    `action("start")` makes, through the shared layer rather than a second
    spelling of it.
    """
    from local_operator.tunnels import install

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_TUNNEL_NO_REARM", raising=False)
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: install.supervisors.SCHTASKS)
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: True)
    monkeypatch.setattr(install.supervisors, "task_scheduler_is_addressable", lambda _base: True)
    runs: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install.supervisors,
        "schtasks",
        lambda *args: runs.append(args) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    config.save(_stored(connection, credential_id=7, gateway_port=4100))
    state.mark_parked(reason=LOGIN_REQUIRED, detail="x", credential_id=7)

    assert install.rearm_if_parked(provider="radient", credential_id=7) != ""
    assert runs == [tuple(install.supervisors.task_run_args(install.TASK_NAME))]

    # …and a task Scheduler refuses is reported, never raised: a login must not
    # fail over a service that could not be started.
    runs.clear()
    monkeypatch.setattr(
        install.supervisors,
        "schtasks",
        lambda *args: SimpleNamespace(returncode=1, stdout="", stderr="Access is denied."),
    )
    assert "could not be restarted" in install.rearm_if_parked(provider="radient", credential_id=7)

    # A store that would not outlive this process reaches nothing, on Windows
    # exactly as on the others.
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: False)
    assert install.rearm_if_parked(provider="radient", credential_id=7) == ""

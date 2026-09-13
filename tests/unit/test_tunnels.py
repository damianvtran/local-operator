"""Tunnel trust boundaries exercised with real RSA proofs and ASGI requests."""

from __future__ import annotations

import argparse
import asyncio
import copy
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
from local_operator.tunnels import config
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
    """The failure the tunnel client raises when its refresh cannot connect.

    Built as the client actually produces it — the transport error on
    `__cause__` of a ValueError — because that chain is the whole reason a
    surface can tell this apart from an expired login.
    """
    try:
        try:
            raise httpx.ConnectError("network is unreachable")
        except httpx.ConnectError as transport:
            raise ValueError("The tunnel's Radient login could not be refreshed.") from transport
    except ValueError as refusal:
        return refusal


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

    assert service.main() == 1

    logged = capsys.readouterr().out
    assert "Harness port for local-operator changed in the console." in logged
    assert "Run lop tunnel connect again." in logged
    # The old generic line sent the operator to two places that are both fine.
    assert "your Radient login" not in logged
    launched.assert_not_called()


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
    from local_operator.tunnels import cli

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

    monkeypatch.setattr(
        cli, "httpx", SimpleNamespace(AsyncClient=FakeClient, HTTPError=httpx.HTTPError)
    )
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))
    # Not "stopped": the gateway answered and reported `connected: true`, so a
    # state word contradicting it would be the command arguing with its own input.
    assert "Local connector: not serving" in receipt
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
    from local_operator.tunnels import cli

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

    monkeypatch.setattr(
        cli, "httpx", SimpleNamespace(AsyncClient=FakeClient, HTTPError=httpx.HTTPError)
    )
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))
    assert "Local connector: not serving" in receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,expected,absent",
    [
        (
            httpx.ConnectError("network is unreachable"),
            "could not reach Radient to renew the relay authorization",
            "check /login radient",
        ),
        (
            ValueError("The tunnel's Radient login expired; log in again."),
            "check /login radient",
            "could not reach Radient to renew the relay authorization",
        ),
        # A refresh that could not reach Radient is a ValueError too, so the
        # exception class cannot carry this decision: only the chain can.
        (
            _refusal_from_an_unreachable_control_plane(),
            "could not reach Radient to renew the relay authorization",
            "check /login radient",
        ),
    ],
)
async def test_tunnel_status_separates_a_network_fault_from_an_unusable_login(
    tmp_path, monkeypatch, connection, failure, expected, absent
):
    """One shared line sent both causes to /login radient.

    They are different jobs for the operator — one is a connection, the other an
    interactive login — and the cloud read is unavailable for both, so the status
    command is the only place that can tell them apart.
    """
    from local_operator.tunnels import cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config.save(_stored(connection))
    api = AsyncMock()
    api.request.side_effect = failure
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: api)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipt = await dispatch(parser.parse_args(["tunnel", "status"]))
    assert expected in receipt
    assert absent not in receipt


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

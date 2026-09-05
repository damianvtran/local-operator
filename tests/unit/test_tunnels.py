"""Tunnel trust boundaries exercised with real RSA proofs and ASGI requests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
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
from local_operator.tunnels.gateway import MAX_BODY_BYTES, PROOF_HEADER, Gateway
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
            return {"eligible": True}
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
    api.request.side_effect = [{"eligible": True}, copy.deepcopy(connection["tunnel"])]
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

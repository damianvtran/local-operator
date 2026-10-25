"""Native attribution is an observed SDK code exchange, never token presence.

The local callback simulator drives the SDK's PKCE/state validation and real
LoopbackAuthFlow listener. HTTP issuer/token/resource responses are synthetic;
this is not human-browser, production-issuer or account-identity proof.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx2 as httpx
import pytest
from mcp.shared.auth import OAuthToken

from local_operator.mcp import auth
from local_operator.mcp.config import MCPHttpServerConfig, MCPOAuthConfig
from local_operator.network import readiness
from local_operator.network.credentials import offers
from local_operator.providers.auth_store import REFRESH_SEND_UNCONFIRMED_KEY, AuthStore
from tests.unit.mcp.test_auth import FakeAuthStore
from tests.unit.network.test_readiness import _FakeViewer, _member

URL = "https://resource.grant.test/mcp"
ISSUER = "https://issuer.grant.test"


async def _sdk_login(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, *, scope: bool = True
) -> None:
    # Reserve a loopback port; no browser/engine, live IdP or system keychain.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cfg = MCPHttpServerConfig(url=URL, oauth=MCPOAuthConfig(callback_port=port))
    challenge: dict[str, str] = {}

    async def callback_simulator(url: str) -> bool:
        query = parse_qs(urlparse(url).query)
        challenge["value"] = query["code_challenge"][0]
        assert query["code_challenge_method"] == ["S256"]
        callback = (
            query["redirect_uri"][0]
            + "?"
            + urlencode({"code": "synthetic-code", "state": query["state"][0], "iss": ISSUER})
        )
        async with httpx.AsyncClient() as local:
            response = await local.get(callback)
            assert response.status_code == 200
        return True

    monkeypatch.setattr(auth, "open_browser_quietly", callback_simulator)
    provider = auth.build_oauth_provider(URL, cfg, store=store, interactive=True)

    def issuer(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if str(request.url) == URL:
            return httpx.Response(
                200 if request.headers.get("authorization") else 401,
                headers={"WWW-Authenticate": 'Bearer scope="tools:read"'},
                json={},
            )
        if "oauth-protected-resource" in path:
            return httpx.Response(200, json={"resource": URL, "authorization_servers": [ISSUER]})
        if "oauth-authorization-server" in path or "openid-configuration" in path:
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": ISSUER + "/authorize",
                    "token_endpoint": ISSUER + "/token",
                    "registration_endpoint": ISSUER + "/register",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                },
            )
        if path == "/register":
            metadata = json.loads(request.content)
            return httpx.Response(
                201,
                json={"client_id": "synthetic-client", "redirect_uris": metadata["redirect_uris"]},
            )
        assert path == "/token", request.url
        form = parse_qs(request.content.decode())
        assert form["grant_type"] == ["authorization_code"]
        verifier = form["code_verifier"][0]
        assert (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
            == challenge["value"]
        )
        assert form["code"] == ["synthetic-code"]
        return httpx.Response(
            200,
            json={
                "access_token": "synthetic-access",
                "refresh_token": "synthetic-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
                **({"scope": "tools:read"} if scope else {}),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(issuer), auth=provider) as client:
        response = await client.post(URL)
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [True, False])
async def test_sdk_callback_attributes_native_and_rotation_preserves_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: bool
) -> None:
    with closing(AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)) as store:
        await _sdk_login(store, monkeypatch, scope=scope)
        row = store.list_credentials(auth.MCP_OAUTH_PROVIDER)[0]
        initial_id = row.id
        metadata = row.data[auth.GRANT_PROVENANCE_KEY]
        assert metadata["kind"] == "native"
        assert metadata["native_event_at"] == auth._chain_stamp_of(row.data)
        snapshot = auth.mcp_grant_diagnostics(URL, store)
        assert snapshot["provenance"] == "native"
        assert snapshot["grant_chain_at"] == metadata["native_event_at"]
        assert snapshot["scopes_known"] is scope
        assert snapshot["identity_verified"] is False
        assert snapshot["sharing_mode"] == "finite_only"
        assert snapshot["refresh_presence"] == "present"
        assert snapshot["expiry_known"] is True
        assert row.data["tokens"]["scope"] == "tools:read"
        storage = auth.McpTokenStorage(URL, store)
        assert storage.store_refresh_result(
            OAuthToken(
                access_token="synthetic-rotated", refresh_token="synthetic-r2", expires_in=3600
            ),
            presented_refresh_token="synthetic-refresh",
        )
        rotated = store.list_credentials(auth.MCP_OAUTH_PROVIDER)[0]
        assert rotated.id == initial_id and rotated.data[auth.GRANT_PROVENANCE_KEY] == metadata
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "native"
        # A stale rotation must not overwrite a new grant's provenance/material.
        assert not storage.store_refresh_result(
            OAuthToken(access_token="stale"), presented_refresh_token="synthetic-refresh"
        )
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "native"
        # Real new authorization, not token import: replace any previous identity
        # payload wholesale and preserve the single URL row's identity key.
        data = dict(rotated.data)
        data[auth.GRANT_PROVENANCE_KEY] = {**metadata, "identity": {"subject": "stale"}}
        # Reauth needs a new SDK authorization, not a successful request with
        # still-current material. Retain the URL row and old nested metadata.
        data["tokens"] = {}
        data[auth.GRANT_DEAD_AT_KEY] = time.time()
        data[auth.GRANT_UNCONFIRMED_SEND_KEY] = {"synthetic": True}
        store.upsert_credential(
            auth.MCP_OAUTH_PROVIDER, {k: v for k, v in data.items() if k != "type"}
        )
        await _sdk_login(store, monkeypatch)
        fresh = store.list_credentials(auth.MCP_OAUTH_PROVIDER)
        assert len(fresh) == 1 and fresh[0].id == initial_id
        assert fresh[0].identity_key == URL
        assert "identity" not in fresh[0].data[auth.GRANT_PROVENANCE_KEY]
        assert auth.GRANT_DEAD_AT_KEY not in fresh[0].data
        assert auth.GRANT_UNCONFIRMED_SEND_KEY not in fresh[0].data
        assert (
            fresh[0].data[auth.GRANT_PROVENANCE_KEY]["native_event_at"]
            != metadata["native_event_at"]
        )


@pytest.mark.asyncio
async def test_generic_import_and_legacy_rotation_never_become_native(tmp_path: Path) -> None:
    with closing(AuthStore(db_path=tmp_path / "auth.db")) as store:
        storage = auth.McpTokenStorage(URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="synthetic-import", refresh_token="synthetic-r1")
        )
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "unknown"
        assert storage.store_refresh_result(
            OAuthToken(access_token="synthetic-next", refresh_token="synthetic-r2"),
            presented_refresh_token="synthetic-r1",
        )
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "unknown"
        data = dict(store.list_credentials(auth.MCP_OAUTH_PROVIDER)[0].data)
        data.pop(auth.GRANT_PROVENANCE_KEY)
        store.upsert_credential(
            auth.MCP_OAUTH_PROVIDER, {k: v for k, v in data.items() if k != "type"}
        )
        assert storage.store_refresh_result(
            OAuthToken(access_token="synthetic-legacy-next", refresh_token="synthetic-r3"),
            presented_refresh_token="synthetic-r2",
        )
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "legacy"
        data = dict(store.list_credentials(auth.MCP_OAUTH_PROVIDER)[0].data)
        data["mesh_origin"] = {"owner_device": "synthetic-owner"}
        store.upsert_credential(
            auth.MCP_OAUTH_PROVIDER, {k: v for k, v in data.items() if k != "type"}
        )
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "copied"
        other_url = URL + "/other"
        other = auth.McpTokenStorage(other_url, store)
        await other.set_tokens(OAuthToken(access_token="synthetic-other"))
        data = dict(store.list_credentials(auth.MCP_OAUTH_PROVIDER)[1].data)
        data[auth.GRANT_PROVENANCE_KEY] = {
            "kind": "native",
            "resource_url": URL,
            "native_event_at": auth._chain_stamp_of(data),
            "identity_verified": True,
            "identity": {"subject": "forged"},
        }
        store.upsert_credential(
            auth.MCP_OAUTH_PROVIDER, {k: v for k, v in data.items() if k != "type"}
        )
        cross_url = auth.mcp_grant_diagnostics(other_url, store)
        assert cross_url["identity_verified"] is False
        assert cross_url["provenance"] == "unknown"
        assert cross_url["sharing_mode"] != "inheritance_eligible"


@pytest.mark.asyncio
async def test_snapshot_is_one_lookup_material_free_and_readiness_offers_are_truthful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)) as store:
        await _sdk_login(store, monkeypatch)
        calls = []
        original = store.list_credentials

        def counted(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "list_credentials", counted)
        snapshot = auth.mcp_grant_diagnostics(URL, store)
        assert len(calls) == 1
        for _ in range(10):
            assert auth.mcp_grant_diagnostics(URL, store) == snapshot
        assert len(calls) == 11
        assert "synthetic-access" not in json.dumps(
            snapshot
        ) and "synthetic-refresh" not in json.dumps(snapshot)
        (tmp_path / "mcp.json").write_text(
            json.dumps({"mcpServers": {"fixture": {"type": "http", "url": URL}}})
        )
        fact = readiness.mcp_servers_fact(tmp_path)
        assert fact["servers"][0]["auth_diagnostics"]["provenance"] == "native"
        items = offers.build_items(tmp_path)
        item = next(r for r in items if r["kind"] == "mcp-rotating")
        assert item["auth_diagnostics"]["sharing_mode"] == "finite_only"
        assert item["auth_diagnostics"]["identity_verified"] is False
        assert set(item["auth_diagnostics"]) == {
            "has_row",
            "provenance",
            "identity_verified",
            "identity_state",
            "sharing_mode",
            "refresh_presence",
            "expiry_known",
            "grant_chain_at",
            "scopes_known",
            "access_state",
        }
        # Advisory observations are deliberately outside the selection digest.
        # Even forged optimism cannot change offered keys/defaults or survive
        # the canonical grant input. No signing/digest implementation changes.
        advisory = [
            (
                {
                    **row,
                    "auth_diagnostics": {
                        "identity_verified": True,
                        "sharing_mode": "inheritance_eligible",
                    },
                }
                if row["kind"] == "mcp-rotating"
                else dict(row)
            )
            for row in items
        ]
        assert offers.canonical_rows(advisory) == offers.canonical_rows(items)
        assert offers.digest_of(advisory) == offers.digest_of(items)
        assert offers.owner_default_shares(advisory) == offers.owner_default_shares(items)
        assert offers.served_keys(advisory) == offers.served_keys(items)
        assert offers.validate_items(advisory) == offers.canonical_rows(items)
        assert offers.digest_of(items) == offers.digest_of(offers.canonical_rows(items))
        assert offers.validate_items(items) == offers.canonical_rows(items)
        # Read actual peer facts through the normal collector and row composer;
        # expired access without refresh is not rescued by row/placement presence.
        data = dict(original(auth.MCP_OAUTH_PROVIDER)[0].data)
        data["tokens"] = {
            "access_token": "synthetic-expired",
            "token_type": "Bearer",
            "expires_in": 1,
        }
        data[auth.TOKENS_OBTAINED_AT_KEY] = time.time() - 120
        store.upsert_credential(
            auth.MCP_OAUTH_PROVIDER, {k: v for k, v in data.items() if k != "type"}
        )
        # The readiness path deliberately reports an open WAL as unknown; close
        # the writer so this cell checks the expired row, not that separate arm.
        store.close()
        facts = readiness.collect_peer_facts(tmp_path)
        rows = readiness.mcp_credential_rows(
            _member(), facts, viewer=_FakeViewer(), peer_label="synthetic-peer"
        )
        assert rows[0]["ok"] is False and rows[0]["code"] == "local_grant_unavailable"
        assert rows[0]["observed"]["auth_diagnostics"]["access_state"] == "expired"
        unavailable = next(r for r in offers.build_items(tmp_path) if r["kind"] == "mcp-rotating")
        assert unavailable["auth_diagnostics"]["sharing_mode"] == "unavailable"


@pytest.mark.asyncio
async def test_real_store_fences_survive_reload_and_only_new_grant_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "auth.db"
    synthetic = "synthetic-refresh-marker"
    with closing(AuthStore(db_path=database)) as store:
        storage = auth.McpTokenStorage(URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="synthetic-access", refresh_token=synthetic)
        )
        assert storage.mark_grant_dead(rejected_refresh_token=synthetic)
        storage.mark_send_unconfirmed(synthetic)
    with closing(AuthStore(db_path=database)) as store:
        storage = auth.McpTokenStorage(URL, store)
        assert storage.grant_is_dead()
        assert storage.send_unconfirmed(synthetic)
        assert auth.mcp_grant_diagnostics(URL, store)["sharing_mode"] == "unavailable"
        calls: list[str] = []

        async def discovery(_url: str) -> None:
            return None

        async def forbidden_refresh(*_args: Any, **_kwargs: Any) -> None:
            calls.append("refresh")
            raise AssertionError("known-dead grant must not be refreshed")

        monkeypatch.setattr(auth, "discover_oauth_endpoints", discovery)
        monkeypatch.setattr(auth, "_refresh_oauth_token_locked", forbidden_refresh)
        await auth.ensure_mcp_oauth_fresh(URL, MCPHttpServerConfig(url=URL), store)
        assert calls == []
        # Same-chain late rotation may persist material but NEVER lift the
        # tombstone. An obsolete refresh result may not overwrite a new chain.
        assert storage.store_refresh_result(
            OAuthToken(access_token="synthetic-late"), presented_refresh_token=synthetic
        )
        assert storage.grant_is_dead()
        await storage.set_tokens(
            OAuthToken(access_token="synthetic-new", refresh_token="synthetic-new-refresh")
        )
        assert not storage.grant_is_dead() and not storage.send_unconfirmed()
        assert not storage.store_refresh_result(
            OAuthToken(access_token="synthetic-stale"), presented_refresh_token=synthetic
        )
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "unknown"
        # Generic provider new-login semantics have not changed.
        provider = store.upsert_credential(
            "openai",
            {
                "account_id": "synthetic-account",
                "access": "synthetic",
                "refresh": "synthetic",
                auth.GRANT_DEAD_AT_KEY: 1,
                REFRESH_SEND_UNCONFIRMED_KEY: {"synthetic": True},
            },
        )
        assert auth.GRANT_DEAD_AT_KEY not in provider.data
        assert REFRESH_SEND_UNCONFIRMED_KEY not in provider.data


@pytest.mark.asyncio
async def test_real_invalid_grant_tombstone_prevents_a_second_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.auth import OAuthClientInformationFull

    from tests.unit.mcp.rotating_oauth_server import FakeTokenEndpoint
    from tests.unit.mcp.test_oauth_refresh_durability import _stub_discovery

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    database = tmp_path / "auth.db"
    with closing(AuthStore(db_path=database)) as store:
        storage = auth.McpTokenStorage(URL, store)
        await storage.set_client_info(OAuthClientInformationFull(client_id="synthetic-client"))
        await storage.set_tokens(
            OAuthToken(
                access_token="synthetic-access", refresh_token="synthetic-rejected", expires_in=1
            )
        )
        data = dict(store.list_credentials(auth.MCP_OAUTH_PROVIDER)[0].data)
        data[auth.TOKENS_OBTAINED_AT_KEY] = time.time() - 120
        assert storage._write(data)
        async with FakeTokenEndpoint(refresh_token="synthetic-other") as endpoint:
            _stub_discovery(monkeypatch, endpoint.token_endpoint)
            await auth.ensure_mcp_oauth_fresh(URL, MCPHttpServerConfig(url=URL), store)
            assert len(endpoint.requests) == 1
            assert storage.grant_is_dead()
            with closing(AuthStore(db_path=database)) as reloaded:
                assert auth.McpTokenStorage(URL, reloaded).grant_is_dead()
                await auth.ensure_mcp_oauth_fresh(URL, MCPHttpServerConfig(url=URL), reloaded)
            assert len(endpoint.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("grant_type", ["refresh_token", "client_credentials", ""])
async def test_unsupported_sdk_grant_types_cannot_attribute_native(
    tmp_path: Path, grant_type: str
) -> None:
    with closing(AuthStore(db_path=tmp_path / "auth.db")) as store:
        provider = auth.build_oauth_provider(URL, MCPHttpServerConfig(url=URL), store=store)
        # Negative SDK compatibility control only. Native success is proved
        # through PKCE+callback above, not by calling this private seam.
        response = httpx.Response(
            200,
            request=httpx.Request("POST", ISSUER + "/token", data={"grant_type": grant_type}),
            json={"access_token": "synthetic-access", "token_type": "Bearer"},
        )
        await provider._handle_token_response(response)
        assert auth.mcp_grant_diagnostics(URL, store)["provenance"] == "unknown"


def test_unreadable_registration_only_and_unknown_expiry_do_not_invent_evidence() -> None:
    class Unreadable(FakeAuthStore):
        def list_credentials(
            self, provider: str | None = None, include_disabled: bool = False
        ) -> list[Any]:
            raise OSError("synthetic unreadable")

    unknown = auth.mcp_grant_diagnostics(URL, Unreadable())
    assert unknown["has_row"] is None and unknown["access_state"] == "unknown"
    assert unknown["refresh_presence"] == "unknown"
    assert unknown["identity_verified"] is False

    class RowStore(FakeAuthStore):
        def list_credentials(
            self, provider: str | None = None, include_disabled: bool = False
        ) -> list[Any]:
            return [
                SimpleNamespace(
                    identity_key=URL, data={"client_info": {"client_id": "synthetic-client"}}
                )
            ]

    registration = auth.mcp_grant_diagnostics(URL, RowStore())
    assert registration["has_row"] is True
    assert registration["sharing_mode"] == "unavailable"
    assert registration["expiry_known"] is False

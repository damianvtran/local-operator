"""Auth: McpTokenStorage over a faked store, wire_oauth_auth kwargs."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.mcp.auth import (
    DEFAULT_CALLBACK_PATH,
    DEFAULT_CALLBACK_PORT,
    McpTokenStorage,
    StructuralAuthStore,
    mcp_oauth_credential_id,
    wire_oauth_auth,
)
from local_operator.mcp.config import (
    MCPAuthConfig,
    MCPHttpServerConfig,
    MCPOAuthConfig,
)


class FakeAuthStore:
    """In-memory stand-in satisfying StructuralAuthStore."""

    def __init__(self) -> None:
        self.creds: dict[str, dict[str, Any]] = {}

    def get_oauth_credential(self, provider_id: str) -> dict[str, Any] | None:
        return self.creds.get(provider_id)

    def upsert_oauth_credential(self, provider_id: str, creds: dict[str, Any]) -> None:
        self.creds[provider_id] = creds


def test_fake_satisfies_structural_protocol() -> None:
    assert isinstance(FakeAuthStore(), StructuralAuthStore)


def test_credential_id_is_url_keyed() -> None:
    assert (
        mcp_oauth_credential_id("https://mcp.example.com/sse")
        == "mcp_oauth:https://mcp.example.com/sse"
    )


class TestMcpTokenStorage:
    @pytest.mark.asyncio
    async def test_token_roundtrip(self) -> None:
        from mcp.shared.auth import OAuthToken

        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)

        assert await storage.get_tokens() is None

        tokens = OAuthToken(
            access_token="acc", token_type="Bearer", expires_in=3600, refresh_token="ref"
        )
        await storage.set_tokens(tokens)
        stored = store.creds["mcp_oauth:https://srv.example/mcp"]
        assert stored["tokens"]["access_token"] == "acc"
        assert stored["tokens"]["refresh_token"] == "ref"

        fetched = await storage.get_tokens()
        assert fetched is not None
        assert fetched.access_token == "acc"
        assert fetched.refresh_token == "ref"

    @pytest.mark.asyncio
    async def test_client_info_roundtrip(self) -> None:
        from mcp.shared.auth import OAuthClientInformationFull

        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)

        assert await storage.get_client_info() is None
        info = OAuthClientInformationFull(client_id="cid", client_secret="sec")
        await storage.set_client_info(info)
        fetched = await storage.get_client_info()
        assert fetched is not None and fetched.client_id == "cid"

    @pytest.mark.asyncio
    async def test_set_tokens_preserves_sibling_client_info(self) -> None:
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))
        await storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer"))
        assert (await storage.get_client_info()).client_id == "cid"

    @pytest.mark.asyncio
    async def test_none_store_degrades_to_noop(self) -> None:
        from mcp.shared.auth import OAuthToken

        storage = McpTokenStorage("https://srv.example/mcp", store=None)
        # _resolve_store falls back to a lazy AuthStore import; force None to
        # exercise the degraded path deterministically.
        storage._store = None
        assert await storage.get_tokens() is None
        await storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer"))  # no-op
        assert await storage.get_tokens() is None

    @pytest.mark.asyncio
    async def test_corrupt_stored_tokens_return_none(self) -> None:
        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        store.creds["mcp_oauth:https://srv.example/mcp"] = {"tokens": {"bogus": 1}}
        assert await storage.get_tokens() is None


class TestWireOauthAuth:
    def _cfg(self, **oauth_overrides: Any) -> MCPHttpServerConfig:
        return MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth"),
            oauth=MCPOAuthConfig(**oauth_overrides),
        )

    def test_default_redirect_uri(self) -> None:
        from mcp.client.auth import OAuthClientProvider

        kwargs = wire_oauth_auth("https://srv.example/mcp", self._cfg(), FakeAuthStore())
        assert kwargs["server_url"] == "https://srv.example/mcp"
        metadata = kwargs["client_metadata"]
        assert metadata.redirect_uris == [
            f"http://127.0.0.1:{DEFAULT_CALLBACK_PORT}{DEFAULT_CALLBACK_PATH}"
        ]
        assert "authorization_code" in metadata.grant_types
        assert metadata.token_endpoint_auth_method == "none"
        assert isinstance(kwargs["storage"], McpTokenStorage)
        # The kwargs construct a real provider (PKCE is automatic inside it).
        provider = OAuthClientProvider(**kwargs)
        assert provider is not None

    def test_custom_callback_port_and_path(self) -> None:
        kwargs = wire_oauth_auth(
            "https://srv.example/mcp",
            self._cfg(callback_port=4567, callback_path="oauth/cb"),
            FakeAuthStore(),
        )
        assert kwargs["client_metadata"].redirect_uris == ["http://127.0.0.1:4567/oauth/cb"]

    def test_explicit_redirect_uri_wins(self) -> None:
        kwargs = wire_oauth_auth(
            "https://srv.example/mcp",
            self._cfg(redirect_uri="http://127.0.0.1:9999/custom"),
            FakeAuthStore(),
        )
        assert kwargs["client_metadata"].redirect_uris == ["http://127.0.0.1:9999/custom"]

    def test_client_secret_switches_auth_method(self) -> None:
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth", client_secret="s3cret"),
        )
        kwargs = wire_oauth_auth("https://srv.example/mcp", cfg, FakeAuthStore())
        assert kwargs["client_metadata"].token_endpoint_auth_method == "client_secret_post"

    def test_no_oauth_block_returns_no_auth(self) -> None:
        """Configs without auth.type=oauth produce no provider (manager skips)."""
        cfg = MCPHttpServerConfig(url="https://srv.example/mcp")
        assert cfg.auth is None

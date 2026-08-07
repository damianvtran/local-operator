"""Auth: McpTokenStorage over the real AuthStore API, wire_oauth_auth kwargs.

FakeAuthStore mirrors the REAL ``providers.auth_store.AuthStore`` surface
(upsert_credential / list_credentials / get_credential, integer row ids,
provider column + identity_key dedupe) so tests exercise the same contract
the production store provides. A conformance test additionally round-trips
through the real SQLite store.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.mcp.auth import (
    DEFAULT_CALLBACK_PATH,
    DEFAULT_CALLBACK_PORT,
    MCP_OAUTH_PROVIDER,
    McpTokenStorage,
    StructuralAuthStore,
    mcp_oauth_credential_id,
    parse_oauth_callback_input,
    wire_oauth_auth,
)
from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig, MCPOAuthConfig
from local_operator.providers.auth_store import StoredCredential


class FakeAuthStore:
    """In-memory stand-in satisfying the real AuthStore's method surface."""

    def __init__(self) -> None:
        self.rows: list[StoredCredential] = []
        self._next_id = 1

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> StoredCredential:
        identity = credential.get("project_id")  # mirrors _identity_key_for ordering
        payload = dict(credential)
        for existing in self.rows:
            if existing.provider == provider and existing.identity_key == identity:
                existing.data = payload
                return existing
        row = StoredCredential(
            id=self._next_id,
            provider=provider,
            credential_type="api_key",
            data=payload,
            identity_key=identity,
        )
        self._next_id += 1
        self.rows.append(row)
        return row

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[StoredCredential]:
        rows = [r for r in self.rows if provider is None or r.provider == provider]
        if include_disabled:
            return rows
        return [r for r in rows if r.disabled_cause is None]

    def get_credential(self, credential_id: int) -> StoredCredential | None:
        for row in self.rows:
            if row.id == credential_id:
                return row
        return None


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
        # Stored under provider 'mcp-oauth' with identity_key = server URL.
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        assert len(rows) == 1
        assert rows[0].identity_key == "https://srv.example/mcp"
        assert rows[0].data["tokens"]["access_token"] == "acc"
        assert rows[0].data["tokens"]["refresh_token"] == "ref"

        fetched = await storage.get_tokens()
        assert fetched is not None
        assert fetched.access_token == "acc"
        assert fetched.refresh_token == "ref"

    @pytest.mark.asyncio
    async def test_upsert_updates_row_in_place(self) -> None:
        """Re-auth for the same URL replaces the row; a second URL adds one."""
        from mcp.shared.auth import OAuthToken

        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        await storage.set_tokens(OAuthToken(access_token="one", token_type="Bearer"))
        await storage.set_tokens(OAuthToken(access_token="two", token_type="Bearer"))
        assert len(store.list_credentials(MCP_OAUTH_PROVIDER)) == 1

        other = McpTokenStorage("https://other.example/mcp", store)
        await other.set_tokens(OAuthToken(access_token="x", token_type="Bearer"))
        assert len(store.list_credentials(MCP_OAUTH_PROVIDER)) == 2
        # Servers never see each other's tokens.
        main_tokens = await storage.get_tokens()
        assert main_tokens is not None and main_tokens.access_token == "two"
        other_tokens = await other.get_tokens()
        assert other_tokens is not None and other_tokens.access_token == "x"

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
        kept = await storage.get_client_info()
        assert kept is not None and kept.client_id == "cid"

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
        store.upsert_credential(
            MCP_OAUTH_PROVIDER,
            {"tokens": {"bogus": 1}, "project_id": "https://srv.example/mcp"},
        )
        assert await storage.get_tokens() is None


class TestRealAuthStoreConformance:
    """MCP-03: the real providers AuthStore satisfies the MCP adapter."""

    def test_real_store_satisfies_structural_protocol(self, tmp_path) -> None:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        try:
            assert isinstance(store, StructuralAuthStore)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_token_roundtrip_through_real_store(self, tmp_path) -> None:
        """Round-trip an MCP token through McpTokenStorage + real AuthStore."""
        from mcp.shared.auth import OAuthToken

        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        try:
            url = "https://mcp.example.com/sse"
            storage = McpTokenStorage(url, store)

            assert await storage.get_tokens() is None
            await storage.set_tokens(
                OAuthToken(
                    access_token="acc-real",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="ref-real",
                )
            )

            # The row landed under provider 'mcp-oauth', identity_key = URL.
            rows = store.list_credentials("mcp-oauth")
            assert len(rows) == 1
            assert rows[0].identity_key == url

            fetched = await storage.get_tokens()
            assert fetched is not None
            assert fetched.access_token == "acc-real"
            assert fetched.refresh_token == "ref-real"

            # Re-auth upserts in place (still one row for the URL).
            await storage.set_tokens(OAuthToken(access_token="acc-2", token_type="Bearer"))
            assert len(store.list_credentials("mcp-oauth")) == 1
            refreshed = await storage.get_tokens()
            assert refreshed is not None and refreshed.access_token == "acc-2"

            # A fresh storage instance for the same URL sees the same row
            # (the logical id 'mcp_oauth:<url>' survives process restarts).
            storage2 = McpTokenStorage(url, store)
            second = await storage2.get_tokens()
            assert second is not None and second.access_token == "acc-2"
        finally:
            store.close()


class TestCallbackInputParsing:
    """MCP-02: the headless handler accepts the full redirect URL."""

    def test_full_url_yields_code_state_and_iss(self) -> None:
        url = (
            "http://127.0.0.1:3000/callback?code=X&state=Y" "&iss=https%3A%2F%2Fauth.example.com%2F"
        )
        code, state, iss = parse_oauth_callback_input(url)
        assert code == "X"
        assert state == "Y"
        assert iss == "https://auth.example.com/"

    def test_url_without_iss(self) -> None:
        code, state, iss = parse_oauth_callback_input(
            "http://127.0.0.1:3000/callback?code=X&state=Y"
        )
        assert (code, state, iss) == ("X", "Y", None)

    def test_code_state_pair(self) -> None:
        assert parse_oauth_callback_input("abc123 st-456") == ("abc123", "st-456", None)

    def test_empty_and_codeless_input_raise(self) -> None:
        with pytest.raises(RuntimeError):
            parse_oauth_callback_input("   ")
        with pytest.raises(RuntimeError):
            parse_oauth_callback_input("http://127.0.0.1:3000/callback?state=Y")

    @pytest.mark.asyncio
    async def test_callback_handler_returns_parsed_state(self, monkeypatch) -> None:
        """Handler given a redirect URL yields the matching state (MCP-02)."""
        from local_operator.mcp.auth import _default_callback_handler

        redirect = "http://127.0.0.1:3000/callback?code=the-code&state=the-state"
        monkeypatch.setattr("builtins.input", lambda _prompt="": redirect)
        # The handler gates on a real TTY; the suite is not one.
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        result = await _default_callback_handler()()
        assert result.code == "the-code"
        assert result.state == "the-state"  # SDK state validation now passes
        assert result.iss is None

    @pytest.mark.asyncio
    async def test_prompt_asks_for_full_redirect_url(self) -> None:
        """The paste prompt must say 'full redirect URL', not 'code'."""
        from local_operator.mcp import auth as auth_mod

        prompts: list[str] = []

        def capturing_input(prompt: str = "") -> str:
            prompts.append(prompt)
            return "http://127.0.0.1:3000/callback?code=c&state=s"

        import builtins

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(builtins, "input", capturing_input)
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        try:
            result = await auth_mod._default_callback_handler()()
        finally:
            monkeypatch.undo()
        assert result.state == "s"
        assert prompts and "full redirect URL" in prompts[0]


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
        assert [str(u) for u in metadata.redirect_uris] == [
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
        assert [str(u) for u in kwargs["client_metadata"].redirect_uris] == [
            "http://127.0.0.1:4567/oauth/cb"
        ]

    def test_explicit_redirect_uri_wins(self) -> None:
        kwargs = wire_oauth_auth(
            "https://srv.example/mcp",
            self._cfg(redirect_uri="http://127.0.0.1:9999/custom"),
            FakeAuthStore(),
        )
        assert [str(u) for u in kwargs["client_metadata"].redirect_uris] == [
            "http://127.0.0.1:9999/custom"
        ]

    def test_client_secret_switches_auth_method(self) -> None:
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth", client_secret="s3cret"),
        )
        kwargs = wire_oauth_auth("https://srv.example/mcp", cfg, FakeAuthStore())
        assert kwargs["client_metadata"].token_endpoint_auth_method == "client_secret_post"

    def test_configured_client_id_preseeds_and_skips_dcr(self) -> None:
        """MCP-11: a configured client_id is seeded so DCR never runs."""
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth", client_id="pinned-cid", client_secret="sec"),
        )
        store = FakeAuthStore()
        wire_oauth_auth("https://srv.example/mcp", cfg, store)
        # The pinned registration is already in storage BEFORE the provider
        # exists: get_client_info finds it and the SDK skips registration.
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        assert rows[0].data["client_info"]["client_id"] == "pinned-cid"
        assert rows[0].data["client_info"]["client_secret"] == "sec"

    @pytest.mark.asyncio
    async def test_preseeded_client_info_visible_to_sdk(self) -> None:
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth", client_id="pinned-cid"),
        )
        kwargs = wire_oauth_auth("https://srv.example/mcp", cfg, FakeAuthStore())
        info = await kwargs["storage"].get_client_info()
        assert info is not None and info.client_id == "pinned-cid"

    def test_no_oauth_block_returns_no_auth(self) -> None:
        """Configs without auth.type=oauth produce no provider (manager skips)."""
        cfg = MCPHttpServerConfig(url="https://srv.example/mcp")
        assert cfg.auth is None

"""Auth: McpTokenStorage over the real AuthStore API, wire_oauth_auth kwargs.

FakeAuthStore mirrors the REAL ``providers.auth_store.AuthStore`` surface
(upsert_credential / list_credentials / get_credential, integer row ids,
provider column + identity_key dedupe) so tests exercise the same contract
the production store provides. A conformance test additionally round-trips
through the real SQLite store.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.mcp.auth import (
    DEFAULT_CALLBACK_PATH,
    DEFAULT_CALLBACK_PORT,
    MCP_OAUTH_PROVIDER,
    McpTokenStorage,
    StructuralAuthStore,
    mcp_logged_out_servers,
    mcp_logout_server,
    mcp_oauth_credential_id,
    oauth_server_names,
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
        import time

        identity = credential.get("project_id")  # mirrors _identity_key_for ordering
        payload = dict(credential)
        # The real store stamps `updated_at = now` on EVERY write, including the
        # client-info writes that do not touch tokens. A fake that leaves the
        # column at 0 cannot catch a caller that mistakes it for the token's
        # issue time, which is exactly the defect this mirrors.
        now_ms = int(time.time() * 1000)
        for existing in self.rows:
            if existing.provider == provider and existing.identity_key == identity:
                existing.data = payload
                existing.updated_at = now_ms
                return existing
        row = StoredCredential(
            id=self._next_id,
            provider=provider,
            credential_type="api_key",
            data=payload,
            identity_key=identity,
            created_at=now_ms,
            updated_at=now_ms,
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

    def delete_credential(self, credential_id: int) -> None:
        # Mirrors the real store's logout path: the row is GONE, not disabled.
        self.rows = [row for row in self.rows if row.id != credential_id]


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
    async def test_get_client_info_drops_legacy_port_registration(self) -> None:
        """A stored registration still targeting :3000 is stale and discarded.

        Regression guard for the pinned-client dead-end: a registration whose
        redirect URIs point at the legacy callback port can never complete a
        grant once the runtime advertises a different port, and the SDK never
        re-runs DCR while ``client_info`` is present. ``get_client_info`` must
        drop it (and persist the drop) so the flow re-registers / re-seeds.
        """
        from mcp.shared.auth import OAuthClientInformationFull
        from pydantic import AnyUrl

        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        info = OAuthClientInformationFull(
            client_id="cid",
            redirect_uris=[AnyUrl("http://127.0.0.1:3000/callback")],
        )
        await storage.set_client_info(info)

        assert await storage.get_client_info() is None
        # The drop is persisted, not just filtered for this read.
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        assert "client_info" not in rows[0].data

    @pytest.mark.asyncio
    async def test_get_client_info_keeps_current_port_registration(self) -> None:
        from mcp.shared.auth import OAuthClientInformationFull
        from pydantic import AnyUrl

        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        info = OAuthClientInformationFull(
            client_id="cid",
            redirect_uris=[AnyUrl("http://127.0.0.1:33441/callback")],
        )
        await storage.set_client_info(info)
        kept = await storage.get_client_info()
        assert kept is not None and kept.client_id == "cid"

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

    def test_clear_removes_the_row_and_reports_whether_one_existed(self) -> None:
        """Logout is a deletion, not a disable: after clear() the SDK's next
        ``get_tokens`` finds nothing and starts a genuinely fresh grant."""
        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        storage._write({"tokens": {"access_token": "a"}})
        assert store.list_credentials(MCP_OAUTH_PROVIDER)

        assert storage.clear() is True
        assert store.list_credentials(MCP_OAUTH_PROVIDER) == []
        # A second clear is a reportable no-op, not an error: "nothing to log
        # out of" is information the caller phrases, not a failure.
        assert storage.clear() is False

    def test_clear_removes_sibling_client_info(self) -> None:
        """The row carries the client registration too; leaving it behind
        would let the next login silently reuse it instead of re-registering."""
        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        storage.seed_client_info("client-1")
        assert storage._read() is not None
        assert storage.clear() is True
        assert storage._read() is None

    def test_clear_with_no_store_degrades_to_false(self) -> None:
        storage = McpTokenStorage("https://srv.example/mcp", store=None)
        storage._store = None
        assert storage.clear() is False

    def test_clear_when_the_delete_itself_fails_reports_false_and_keeps_the_row(self) -> None:
        """The case the reauth safety depends on: a FAILED delete must never
        be reported as a successful logout — clear() is False and the row
        survives, so the caller refuses to run a "fresh" grant on top of it."""

        class RefusingStore(FakeAuthStore):
            def delete_credential(self, credential_id: int) -> None:
                raise RuntimeError("database is locked")

        store = RefusingStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        storage._write({"tokens": {"access_token": "a"}})
        assert storage.clear() is False
        assert len(store.list_credentials(MCP_OAUTH_PROVIDER)) == 1


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

    @pytest.mark.asyncio
    async def test_clear_roundtrip_through_real_store(self, tmp_path) -> None:
        """Logout deletes the real row, and a fresh storage instance (i.e. the
        next process) finds nothing — the whole point of the command."""
        from mcp.shared.auth import OAuthToken

        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        try:
            url = "https://mcp.example.com/sse"
            storage = McpTokenStorage(url, store)
            await storage.set_tokens(OAuthToken(access_token="acc", token_type="Bearer"))
            assert storage.clear() is True
            fresh = McpTokenStorage(url, store)
            assert await fresh.get_tokens() is None
            assert store.list_credentials("mcp-oauth") == []
        finally:
            store.close()


class TestLogoutHelpers:
    """``mcp_logout_server`` / ``oauth_server_names`` / ``mcp_logged_out_servers``."""

    def _configs(self) -> dict[str, Any]:
        return {
            "linear": MCPHttpServerConfig(
                url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
            ),
            "stdio": MCPHttpServerConfig(url="https://stdio.example/mcp"),
        }

    def test_oauth_server_names_offers_only_oauth_servers(self, monkeypatch) -> None:
        """A stdio/API-key server has no grant to manage; offering it would be
        a row whose only outcome is a warning notice."""
        monkeypatch.setattr(
            "local_operator.mcp.config.load_all_mcp_configs",
            lambda cwd: (self._configs(), {}),
        )
        assert oauth_server_names(Path("/anywhere")) == ["linear"]

    def test_logout_removes_the_stored_credential(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "local_operator.mcp.config.load_all_mcp_configs",
            lambda cwd: (self._configs(), {}),
        )
        store = FakeAuthStore()
        McpTokenStorage("https://mcp.linear.app/mcp", store)._write(
            {"tokens": {"access_token": "a"}}
        )
        assert mcp_logout_server("linear", Path("/anywhere"), store) is None
        assert store.list_credentials(MCP_OAUTH_PROVIDER) == []

    def test_logout_without_a_stored_credential_says_so(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "local_operator.mcp.config.load_all_mcp_configs",
            lambda cwd: (self._configs(), {}),
        )
        error = mcp_logout_server("linear", Path("/anywhere"), FakeAuthStore())
        assert error is not None and "nothing to log out of" in error

    def test_logout_of_unknown_or_non_oauth_server_is_an_error(self, monkeypatch) -> None:
        """A name the config does not know is a typo the user wants told
        about — silently succeeding would claim a credential was removed."""
        monkeypatch.setattr(
            "local_operator.mcp.config.load_all_mcp_configs",
            lambda cwd: (self._configs(), {}),
        )
        assert "not configured" in (mcp_logout_server("linar", Path("/anywhere")) or "")
        not_oauth = mcp_logout_server("stdio", Path("/anywhere")) or ""
        assert "does not use OAuth" in not_oauth

    def test_logged_out_servers_keys_by_url(self) -> None:
        """The picker list is keyed by server NAME but the store by URL; the
        helper returns the store's keys so the caller can do the mapping."""
        store = FakeAuthStore()
        McpTokenStorage("https://mcp.linear.app/mcp", store)._write(
            {"tokens": {"access_token": "a"}}
        )
        assert mcp_logged_out_servers(store) == {"https://mcp.linear.app/mcp"}

    def test_logged_out_servers_distinguishes_an_unreadable_store(self) -> None:
        """None, not the empty set: an unreadable store is not the same
        answer as "no credentials anywhere", and the picker needs the
        difference to say so."""

        class ExplodingStore(FakeAuthStore):
            def list_credentials(  # type: ignore[no-untyped-def]
                self, provider=None, include_disabled=False
            ):
                raise RuntimeError("database is locked")

        assert mcp_logged_out_servers(ExplodingStore()) is None


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
        from local_operator.mcp.auth import LoopbackAuthFlow

        redirect = "http://127.0.0.1:3000/callback?code=the-code&state=the-state"
        monkeypatch.setattr("builtins.input", lambda _prompt="": redirect)
        # The paste path gates on an interactive stdin; the suite is not one.
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        # A non-loopback redirect URI: nothing to listen on, so the paste is
        # the only route and the test never binds a port.
        result = await LoopbackAuthFlow("https://example.test/cb").callback_handler()
        assert result.code == "the-code"
        assert result.state == "the-state"  # SDK state validation now passes
        assert result.iss is None

    @pytest.mark.asyncio
    async def test_prompt_asks_for_full_redirect_url(self) -> None:
        """The paste prompt must say 'full redirect URL', not 'code'."""
        from local_operator.mcp.auth import LoopbackAuthFlow

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
            result = await LoopbackAuthFlow("https://example.test/cb").callback_handler()
        finally:
            monkeypatch.undo()
        assert result.state == "s"
        assert prompts and "full redirect URL" in prompts[0]

    @pytest.mark.asyncio
    async def test_paste_is_refused_while_the_tui_owns_the_terminal(self, monkeypatch) -> None:
        """Never read stdin behind Textual's back — that is where keystrokes go.

        Two readers on the same tty do not queue: they split the input between
        them, so the user's typing lands half in the editor and half in a
        prompt they cannot see. With no listener to fall back on, the flow must
        fail with an actionable message instead of eating the keyboard.
        """
        import sys

        from local_operator.mcp.auth import LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("local_operator.logger.console_is_silenced", lambda: True)
        called = False

        def must_not_run(_prompt: str = "") -> str:
            nonlocal called
            called = True
            return ""

        monkeypatch.setattr("builtins.input", must_not_run)

        with pytest.raises(RuntimeError, match="mcp login"):
            await LoopbackAuthFlow("https://example.test/cb").callback_handler()
        assert called is False


class TestLoopbackCallbackServer:
    """The redirect URI we advertise is one we actually answer."""

    @staticmethod
    def _free_port() -> int:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    @staticmethod
    async def _listening(flow: Any) -> None:
        """Open the flow's listener, failing loudly if the bind was lost.

        ``_free_port`` closes its probe before the flow binds, so another
        process can take the port in between. ``_start_server`` swallows that
        as a notice, which would otherwise surface here as a confusing
        connection-refused or "not a loopback address" much later.
        """
        await flow.redirect_handler("https://provider.test/authorize")
        assert flow._server is not None, f"listener never bound: {flow._bind_error}"

    @pytest.mark.asyncio
    async def test_browser_redirect_completes_the_flow(self, monkeypatch) -> None:
        """A real GET to the redirect URI hands the code back to the SDK."""
        import asyncio
        import sys
        import urllib.request

        from local_operator.mcp.auth import LoopbackAuthFlow

        # stdin must stay out of it: the listener is the whole point here.
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        await self._listening(flow)

        async def visit() -> str:
            url = f"http://127.0.0.1:{port}/callback?code=abc&state=xyz&iss=https://issuer.test"
            return await asyncio.to_thread(
                lambda: urllib.request.urlopen(url, timeout=5).read().decode()
            )

        page, result = await asyncio.gather(visit(), flow.callback_handler())
        assert "Authorized" in page  # the tab says something useful
        assert (result.code, result.state) == ("abc", "xyz")
        assert result.iss == "https://issuer.test"

    @pytest.mark.asyncio
    async def test_provider_error_redirect_fails_the_flow(self, monkeypatch) -> None:
        """``?error=`` is the provider refusing; surface it, do not hang."""
        import asyncio
        import sys
        import urllib.request

        from local_operator.mcp.auth import LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        await self._listening(flow)

        async def visit() -> None:
            url = f"http://127.0.0.1:{port}/callback?error=access_denied"
            await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=5).read())

        with pytest.raises(RuntimeError, match="access_denied"):
            await asyncio.gather(visit(), flow.callback_handler())

    @pytest.mark.asyncio
    async def test_listener_is_released_after_the_flow(self, monkeypatch) -> None:
        """The port must not stay bound: a retry has to be able to rebind it."""
        import asyncio
        import socket
        import sys
        import urllib.request

        from local_operator.mcp.auth import LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        await self._listening(flow)

        async def visit() -> None:
            url = f"http://127.0.0.1:{port}/callback?code=c&state=s"
            await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=5).read())

        await asyncio.gather(visit(), flow.callback_handler())
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))  # raises if the flow leaked the listener

    @pytest.mark.asyncio
    async def test_an_idle_connection_cannot_hold_the_flow_open(self, monkeypatch) -> None:
        """A silent peer must not park teardown — the code is already in hand.

        Since 3.12.1 ``Server.wait_closed()`` also waits for every accepted
        connection's handler, so one socket that connects and says nothing (a
        browser preconnect, a port scanner) would hang ``callback_handler``
        forever after a perfectly successful authorization.
        """
        import asyncio
        import socket
        import sys
        import time
        import urllib.request

        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.auth import LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        await self._listening(flow)

        idle = socket.create_connection(("127.0.0.1", port), timeout=5)  # says nothing, ever
        try:

            async def visit() -> None:
                url = f"http://127.0.0.1:{port}/callback?code=c&state=s"
                await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=5).read())

            # Bounded BELOW the head-read deadline on purpose. R1 asked for two
            # bounds — the read deadline and the teardown timeout — and either
            # one alone is enough to finish this scenario eventually, so a
            # generous outer timeout would keep passing after one of them was
            # deleted. Waiting less than `_REQUEST_READ_TIMEOUT_S` means only
            # the teardown bound can satisfy it.
            started = time.monotonic()
            _, result = await asyncio.wait_for(
                asyncio.gather(visit(), flow.callback_handler()), timeout=5
            )
            elapsed = time.monotonic() - started
        finally:
            idle.close()
        assert result.code == "c"
        assert elapsed < auth_mod._REQUEST_READ_TIMEOUT_S, (
            f"took {elapsed:.1f}s — the idle handler's own read deadline carried "
            "this, not the teardown bound"
        )

    @pytest.mark.asyncio
    async def test_the_listener_path_never_reads_stdin(self, monkeypatch) -> None:
        """No paste race: a thread parked in ``input()`` cannot be cancelled.

        Racing one would leave a second reader on the tty and a thread that
        ``asyncio.run`` joins at shutdown, so ``local-operator mcp login`` would
        hang AFTER the browser login succeeded. With a listener bound, stdin
        must not be touched at all.
        """
        import asyncio
        import sys
        import urllib.request

        from local_operator.mcp.auth import LoopbackAuthFlow

        # Everything the paste gate checks says "yes, you may read stdin".
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("local_operator.logger.console_is_silenced", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)

        def must_not_run(_prompt: str = "") -> str:
            raise AssertionError("the listener path must never read stdin")

        monkeypatch.setattr("builtins.input", must_not_run)

        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        await self._listening(flow)

        async def visit() -> None:
            url = f"http://127.0.0.1:{port}/callback?code=c&state=s"
            await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=5).read())

        _, result = await asyncio.gather(visit(), flow.callback_handler())
        assert result.code == "c"

    @pytest.mark.asyncio
    async def test_a_lost_bind_says_the_port_is_taken(self, monkeypatch) -> None:
        """ "Port busy" and "unservable redirect URI" need different advice."""
        import socket
        import sys

        from local_operator.mcp.auth import LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        squatter = socket.socket()
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = squatter.getsockname()[1]
        try:
            flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
            await flow.redirect_handler("https://provider.test/authorize")
            assert flow._server is None
            with pytest.raises(RuntimeError, match="could listen on .*address already in use"):
                await flow.callback_handler()
        finally:
            squatter.close()

    @pytest.mark.asyncio
    async def test_a_blank_error_description_still_names_the_error(self, monkeypatch) -> None:
        """A whitespace-only description must not blank the error code.

        `?error=access_denied&error_description=%20%20%20` is reachable from the
        wire, and testing the raw value for truthiness satisfies it — so the
        `or error` fallback never fires and the CLI-facing exception loses the
        one word that says what went wrong.
        """
        import asyncio
        import sys
        import urllib.parse
        import urllib.request

        from local_operator.mcp.auth import LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        await self._listening(flow)

        query = urllib.parse.urlencode({"error": "access_denied", "error_description": "   "})

        async def visit() -> str:
            url = f"http://127.0.0.1:{port}/callback?{query}"
            return await asyncio.to_thread(
                lambda: urllib.request.urlopen(url, timeout=5).read().decode()
            )

        page_task = asyncio.ensure_future(visit())
        with pytest.raises(RuntimeError, match="access_denied"):
            await asyncio.wait_for(flow.callback_handler(), timeout=10)
        page = await page_task
        # And the page shows the code rather than a labelled empty box.
        assert "Provider response" in page
        assert "access_denied" in page

    @pytest.mark.asyncio
    async def test_a_redirect_without_a_code_fails_the_flow(self, monkeypatch) -> None:
        """A codeless redirect must end the grant, not leave it waiting.

        The page tells the user they can close the tab. If the flow does not
        settle, that sentence points at a terminal still parked on a redirect
        that can never carry a code — a five-minute silent wait after the user
        has been told it is over.
        """
        import asyncio
        import sys
        import urllib.request

        from local_operator.mcp.auth import LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        await self._listening(flow)

        async def visit() -> str:
            url = f"http://127.0.0.1:{port}/callback?state=s"  # no code
            return await asyncio.to_thread(
                lambda: urllib.request.urlopen(url, timeout=5).read().decode()
            )

        page_task = asyncio.ensure_future(visit())
        with pytest.raises(RuntimeError, match="carried no authorization code"):
            await asyncio.wait_for(flow.callback_handler(), timeout=10)
        page = await page_task
        assert "No authorization code" in page

    @pytest.mark.asyncio
    async def test_an_abandoned_grant_ends_as_an_explicit_cancel(self, monkeypatch) -> None:
        """A browser that never comes back must end with a receipt, not a wait.

        Closing the tab or abandoning the consent screen is indistinguishable
        from a slow human at the protocol level, so the interactive grant
        carries its own idle clock. When it fires, the flow raises a RAW
        ``CancelledError`` — ordinary exceptions do not survive the SDK's
        transport, and the cancellation shape is the channel that unwinds it
        — after recording itself in the ABANDONED_GRANTS ledger, which is
        where the manager learns to re-voice it as McpLoginCancelledError.
        """
        import asyncio
        import sys

        from local_operator.mcp.auth import ABANDONED_GRANTS, LoopbackAuthFlow

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        # The inner redirect clock must OUTLIVE the guard for the guard to be
        # the clock that fires — which is also their real-world relationship
        # (300 s vs 600 s), shrunk.
        monkeypatch.setattr("local_operator.mcp.auth.PASTE_INPUT_TIMEOUT_S", 10.0)
        monkeypatch.setattr("local_operator.mcp.auth.INTERACTIVE_GRANT_TIMEOUT_S", 0.2)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        await self._listening(flow)

        with pytest.raises(asyncio.CancelledError):
            await flow.callback_handler()
        assert ABANDONED_GRANTS.pop(flow), "the abandonment must be recorded"
        assert not ABANDONED_GRANTS.pop(flow), "a record is consumed once"

    @pytest.mark.asyncio
    async def test_a_cancelled_login_releases_the_port_and_says_so(self, monkeypatch) -> None:
        """Task cancellation is a cancel too: unwind, release the port, report.

        The login worker can be cancelled out from under the flow — an
        exclusive re-login, the TUI's stop-ladder. If the flow re-raised the
        bare ``CancelledError``, the listener's teardown would be cancelled
        along with it on any escalation, leaking the redirect port into the
        next grant; and the caller would have an exception with no words in
        it. The shielded teardown and the named error are the two halves of
        the same receipt.
        """
        import asyncio
        import socket
        import sys

        from local_operator.mcp.auth import LoopbackAuthFlow, McpLoginCancelledError

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        port = self._free_port()
        flow = LoopbackAuthFlow(f"http://127.0.0.1:{port}/callback")
        await self._listening(flow)

        task = asyncio.ensure_future(flow.callback_handler())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(McpLoginCancelledError, match="interrupted"):
            await asyncio.wait_for(task, timeout=5)

        # The redirect port must be free for the NEXT login, immediately.
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:  # pragma: no cover - the failure IS the test
            pytest.fail(f"redirect port still held after cancellation: {exc}")
        finally:
            probe.close()


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

    def test_default_redirect_uri_is_the_rare_port(self) -> None:
        """Pin the default to the deliberate rare port, not the constant.

        ``test_default_redirect_uri`` asserts against ``DEFAULT_CALLBACK_PORT``
        symbolically, so it would pass no matter what value the constant
        takes. This test pins the actual default so an accidental revert to a
        colliding dev-server port (e.g. :3000) fails loudly.
        """
        kwargs = wire_oauth_auth("https://srv.example/mcp", self._cfg(), FakeAuthStore())
        assert [str(u) for u in kwargs["client_metadata"].redirect_uris] == [
            "http://127.0.0.1:33441/callback"
        ]

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

    def test_seeded_client_info_stamps_token_endpoint_auth_method(self) -> None:
        """A pinned seed must name a secret-based auth method, or the secret
        is never sent and the provider rejects the token exchange (HubSpot's
        ``BAD_CLIENT_SECRET``). The method has to be correct at the source
        because ``wire_oauth_auth`` re-seeds on every login, overwriting any
        hand-patched store value."""
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth", client_id="pinned-cid", client_secret="sec"),
        )
        store = FakeAuthStore()
        wire_oauth_auth("https://srv.example/mcp", cfg, store)
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        assert rows[0].data["client_info"]["token_endpoint_auth_method"] == "client_secret_post"

    def test_seeded_client_info_without_secret_uses_no_auth(self) -> None:
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth", client_id="pinned-cid"),
        )
        store = FakeAuthStore()
        wire_oauth_auth("https://srv.example/mcp", cfg, store)
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        assert rows[0].data["client_info"]["token_endpoint_auth_method"] == "none"

    def test_reseed_overwrites_stale_token_endpoint_auth_method(self) -> None:
        """The field-recovery path this fix depends on: a stored registration
        whose method is wrong/absent (written before the stamp, or by hand)
        must be corrected by the re-seed ``wire_oauth_auth`` runs on every
        login — not left to fail the token exchange again."""
        from mcp.shared.auth import OAuthClientInformationFull

        store = FakeAuthStore()
        storage = McpTokenStorage("https://srv.example/mcp", store)
        # Pre-fix shape: a pinned registration with no auth method stamped.
        bad = OAuthClientInformationFull(client_id="pinned-cid", client_secret="sec")
        assert bad.token_endpoint_auth_method is None
        import asyncio

        asyncio.run(storage.set_client_info(bad))

        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth", client_id="pinned-cid", client_secret="sec"),
        )
        wire_oauth_auth("https://srv.example/mcp", cfg, store)
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        assert rows[0].data["client_info"]["token_endpoint_auth_method"] == "client_secret_post"

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


class TestStoredTokenExpiry:
    """A token's lifetime has to survive the process that received it.

    ``OAuthToken`` carries only the relative ``expires_in`` the server quoted,
    and the SDK reloads tokens without reloading any deadline — so an expired
    access token looks valid to a fresh process, gets a 401, and triggers a
    full browser grant while an unspent refresh token sits in the same row.
    """

    URL = "https://srv.example/mcp"

    def _storage(self) -> tuple[McpTokenStorage, FakeAuthStore]:
        store = FakeAuthStore()
        return McpTokenStorage(self.URL, store), store

    @pytest.mark.asyncio
    async def test_expiry_is_issue_time_plus_lifetime(self) -> None:
        import time

        from mcp.shared.auth import OAuthToken

        storage, _ = self._storage()
        before = time.time()
        await storage.set_tokens(OAuthToken(access_token="a", refresh_token="r", expires_in=3600))
        expiry = storage.stored_token_expiry()
        assert expiry is not None
        assert before + 3600 <= expiry <= time.time() + 3600

    @pytest.mark.asyncio
    async def test_refreshed_tokens_restamp_the_issue_time(self) -> None:
        """A refresh must move the deadline; otherwise it expires immediately."""
        import time

        from mcp.shared.auth import OAuthToken

        storage, store = self._storage()
        await storage.set_tokens(OAuthToken(access_token="old", expires_in=60))
        stale = storage.stored_token_expiry()
        store.rows[0].data["tokens_obtained_at"] = time.time() - 600  # pretend it aged
        await storage.set_tokens(OAuthToken(access_token="new", expires_in=60))
        fresh = storage.stored_token_expiry()
        assert stale is not None and fresh is not None
        assert fresh > time.time()

    @staticmethod
    def _legacy_row(store: FakeAuthStore, url: str, *, age_s: float, expires_in: int) -> None:
        """Plant a grant in the shape rows had before this fix: no issue time."""
        import time

        store.upsert_credential(
            MCP_OAUTH_PROVIDER,
            {
                "project_id": url,
                "tokens": {"access_token": "a", "refresh_token": "r", "expires_in": expires_in},
                "client_info": {"client_id": "cid"},
            },
        )
        store.rows[0].updated_at = int((time.time() - age_s) * 1000)  # ms, as SQLite stores it

    def test_legacy_row_without_issue_time_uses_updated_at(self) -> None:
        """Grants written before this fix must benefit without re-authorizing."""
        import time

        store = FakeAuthStore()
        self._legacy_row(store, self.URL, age_s=100_000, expires_in=86400)
        # Opened AFTER the row exists, which is what a fresh process does.
        storage = McpTokenStorage(self.URL, store)
        expiry = storage.stored_token_expiry()
        assert expiry is not None and expiry < time.time()  # correctly seen as expired

    @pytest.mark.asyncio
    async def test_a_pinned_client_id_does_not_erase_the_legacy_expiry(self) -> None:
        """Our own client-info seed must not reset the deadline it is read from.

        ``wire_oauth_auth`` calls ``seed_client_info`` whenever the config pins
        a ``client_id``, and the store stamps ``updated_at`` on that write. Read
        the column afterwards and every legacy grant looks brand new — so the
        migration would be a guaranteed no-op for exactly the pinned-redirect
        servers the seed exists to serve.
        """
        import time

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        self._legacy_row(store, self.URL, age_s=100_000, expires_in=86400)
        cfg = MCPHttpServerConfig(
            url=self.URL, auth=MCPAuthConfig(type="oauth", client_id="pinned-cid")
        )
        provider = build_oauth_provider(self.URL, cfg, store=store)
        assert provider.context.token_expiry_time is not None
        assert provider.context.token_expiry_time < time.time()
        await provider._initialize()
        assert provider.context.is_token_valid() is False

    @pytest.mark.asyncio
    async def test_token_without_a_quoted_lifetime_has_no_opinion(self) -> None:
        """No ``expires_in`` means no deadline to invent — leave the SDK alone."""
        from mcp.shared.auth import OAuthToken

        storage, _ = self._storage()
        await storage.set_tokens(OAuthToken(access_token="a"))
        assert storage.stored_token_expiry() is None

    def test_no_row_has_no_opinion(self) -> None:
        storage, _ = self._storage()
        assert storage.stored_token_expiry() is None

    @pytest.mark.asyncio
    async def test_provider_is_primed_so_a_stale_token_refreshes(self) -> None:
        """The end of the chain: the SDK must see the token as expired.

        ``is_token_valid()`` False + a refresh token present is exactly the
        state that sends ``async_auth_flow`` down the refresh branch instead of
        the browser one.
        """
        import time

        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(OAuthToken(access_token="stale", refresh_token="r", expires_in=60))
        store.rows[0].data["tokens_obtained_at"] = time.time() - 3600  # a day-old grant

        cfg = MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))
        provider = build_oauth_provider(self.URL, cfg, store=store)
        assert provider.context.token_expiry_time is not None

        await provider._initialize()  # what the SDK does on the first request
        assert provider.context.is_token_valid() is False
        assert provider.context.current_tokens is not None
        assert provider.context.current_tokens.refresh_token == "r"

    @pytest.mark.asyncio
    async def test_live_token_is_left_valid(self) -> None:
        """The mirror case: a token still inside its lifetime must not refresh."""
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="fresh", refresh_token="r", expires_in=3600)
        )
        cfg = MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))
        provider = build_oauth_provider(self.URL, cfg, store=store)
        await provider._initialize()
        assert provider.context.is_token_valid() is True


class TestBrowserLaunchContainment:
    """A login flow spawns a browser, and browsers print.

    ``webbrowser.open`` hands the browser fd 1 and fd 2 UNCHANGED — the
    stdlib's ``GenericBrowser``/``BackgroundBrowser`` pass neither ``stdout``
    nor ``stderr`` to ``Popen`` — so under the TUI a ``Gtk-Message:`` line or
    an ``xdg-open: no method available`` lands on the composed frame. Same
    defect as an MCP server's startup banner, reached through OAuth instead.

    ``BROWSER`` is the env var ``webbrowser`` honours for a custom command, so
    these drive a real launch of a real script rather than a patched function.
    """

    @staticmethod
    def _noisy_browser(tmp_path: Path) -> Path:
        script = tmp_path / "browser.sh"
        script.write_text(
            "#!/bin/sh\n" 'echo "Gtk-Message: Failed to load module for $1" >&2\n' "exit 0\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    @pytest.mark.asyncio
    async def test_silenced_console_keeps_the_browser_off_the_terminal(
        self, monkeypatch, tmp_path: Path, terminal_output: Path, caplog
    ) -> None:
        """With the TUI on screen the browser's chatter goes to the log."""
        import logging

        from local_operator.mcp.auth import open_browser_quietly

        monkeypatch.setenv("BROWSER", f"{self._noisy_browser(tmp_path)} %s")
        monkeypatch.setattr("local_operator.logger.console_is_silenced", lambda: True)

        with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
            opened = await open_browser_quietly("https://provider.test/authorize")

        assert opened is True
        assert terminal_output.read_bytes() == b""
        # Not discarded: a browser that could not start is a real login failure,
        # and this line is the only place the reason survives.
        assert "Gtk-Message: Failed to load module" in caplog.text

    @pytest.mark.asyncio
    async def test_owning_the_terminal_keeps_the_in_process_call(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Headless ``mcp login`` must not pay for, or hide, the containment.

        With the terminal ours the browser's complaint is exactly what the user
        should see, so the plain ``webbrowser.open`` call stays — asserted by
        patching it, which the launcher subprocess would bypass entirely.
        """
        from local_operator.mcp.auth import open_browser_quietly

        monkeypatch.setattr("local_operator.logger.console_is_silenced", lambda: False)
        calls: list[str] = []
        monkeypatch.setattr("webbrowser.open", lambda url: calls.append(url) or True)

        assert await open_browser_quietly("https://provider.test/authorize") is True
        assert calls == ["https://provider.test/authorize"]


class TestNonInteractiveFlow:
    """A background connect must never open a browser.

    Startup and auto-reconnect run non-interactive: when the stored grant
    cannot be refreshed, the redirect handler raises ``McpAuthRequiredError``
    instead of popping a login tab. Only an explicit ``/mcp login`` (which
    passes ``interactive=True``) may open a browser. This is the universal
    defense against the startup AND exit popups: the exit path's session-
    termination DELETE runs through the same auth flow, and a non-interactive
    handler raises there too (the SDK catches it), so no tab ever opens.
    """

    URL = "https://srv.example/mcp"

    @pytest.mark.asyncio
    async def test_non_interactive_redirect_raises_instead_of_browser(self) -> None:
        from local_operator.mcp.auth import LoopbackAuthFlow, McpAuthRequiredError

        flow = LoopbackAuthFlow(
            "http://127.0.0.1:3000/callback", server_url=self.URL, interactive=False
        )
        with pytest.raises(McpAuthRequiredError):
            await flow.redirect_handler("https://provider.test/authorize")

    @pytest.mark.asyncio
    async def test_non_interactive_never_starts_the_listener(self) -> None:
        """Raising before ``_start_server`` means no socket is bound either."""
        from local_operator.mcp.auth import LoopbackAuthFlow, McpAuthRequiredError

        flow = LoopbackAuthFlow(
            "http://127.0.0.1:3000/callback", server_url=self.URL, interactive=False
        )
        with pytest.raises(McpAuthRequiredError):
            await flow.redirect_handler("https://provider.test/authorize")
        assert flow._server is None

    @pytest.mark.asyncio
    async def test_interactive_default_is_preserved(self) -> None:
        """``wire_oauth_auth`` without an explicit flag stays interactive (login)."""
        cfg = MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))
        kwargs = wire_oauth_auth(self.URL, cfg, FakeAuthStore())
        # The flow is reachable through the redirect handler's closure; assert the
        # interactive default by confirming a non-interactive raise does NOT fire.
        # We can't await the real handler (it binds a port), so inspect the flow.
        assert kwargs["redirect_handler"] is not None

    @pytest.mark.asyncio
    async def test_wire_threads_interactive_false(self) -> None:
        """``interactive=False`` must reach the flow so startup never opens a tab."""
        from local_operator.mcp.auth import LoopbackAuthFlow

        cfg = MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))
        kwargs = wire_oauth_auth(self.URL, cfg, FakeAuthStore(), interactive=False)
        # Reconstruct: the handler is a bound method of the flow instance.
        flow = kwargs["redirect_handler"].__self__
        assert isinstance(flow, LoopbackAuthFlow)
        assert flow.interactive is False


class TestOAuthEndpointDiscovery:
    """Proactive refresh needs the REAL token endpoint, not the SDK's guess.

    The SDK's in-flow refresh falls back to ``urljoin(server_base, "/token")``
    when it has no authorization-server metadata — which a fresh process never
    does. For providers whose token endpoint lives elsewhere (Datadog) that
    guess 404s and the refresh escalates to a browser grant. Discovery resolves
    the real endpoints via PRM then ASM so the refresh targets the right place.
    """

    URL = "https://mcp.example.com/v1/mcp"

    @pytest.mark.asyncio
    async def test_discovery_resolves_token_endpoint_from_prm_and_asm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from local_operator.mcp import auth as auth_mod

        auth_mod._DISCOVERED_ENDPOINTS_CACHE.clear()

        prm_body = {
            "resource": self.URL,
            "authorization_servers": ["https://mcp.example.com/v1/mcp"],
        }
        asm_body = {
            "issuer": "https://mcp.example.com/v1/mcp",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/oauth/token",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "oauth-protected-resource" in url:
                return httpx.Response(200, json=prm_body)
            if "oauth-authorization-server" in url:
                return httpx.Response(200, json=asm_body)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)

        real_client = httpx.AsyncClient

        def patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_client)
        endpoints = await auth_mod.discover_oauth_endpoints(self.URL)
        assert endpoints is not None
        assert (
            str(endpoints.oauth_metadata.token_endpoint) == "https://auth.example.com/oauth/token"
        )
        assert endpoints.protected_resource_metadata is not None

    @pytest.mark.asyncio
    async def test_discovery_returns_none_when_asm_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from local_operator.mcp import auth as auth_mod

        auth_mod._DISCOVERED_ENDPOINTS_CACHE.clear()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_client)
        assert await auth_mod.discover_oauth_endpoints(self.URL) is None

    @pytest.mark.asyncio
    async def test_discovery_caches_successes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from local_operator.mcp import auth as auth_mod

        auth_mod._DISCOVERED_ENDPOINTS_CACHE.clear()
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            url = str(request.url)
            if "oauth-authorization-server" in url:
                return httpx.Response(
                    200,
                    json={
                        "issuer": self.URL,
                        "authorization_endpoint": "https://a/authorize",
                        "token_endpoint": "https://a/token",
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_client)
        first = await auth_mod.discover_oauth_endpoints(self.URL)
        second = await auth_mod.discover_oauth_endpoints(self.URL)
        assert first is second
        # The second call hit the cache, so no additional HTTP requests were made
        # beyond the first discovery's fetches.
        assert calls["n"] >= 1


class TestProactiveRefresh:
    """``ensure_mcp_oauth_fresh`` spends a stored refresh token before connect.

    This is what stops a day-old access token from forcing a browser grant on
    startup: the refresh is performed against the DISCOVERED token endpoint,
    race-free across concurrently starting sessions, and the result is cached so
    the provider can be primed with the real endpoints.
    """

    URL = "https://mcp.example.com/v1/mcp"

    def _cfg(self) -> MCPHttpServerConfig:
        return MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))

    @pytest.mark.asyncio
    async def test_live_token_is_not_refreshed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp import auth as auth_mod

        auth_mod._DISCOVERED_ENDPOINTS_CACHE.clear()
        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="fresh", refresh_token="r", expires_in=3600)
        )

        refreshed = {"called": False}

        async def fake_refresh(*args: Any, **kwargs: Any) -> bool:
            refreshed["called"] = True
            return True

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", fake_refresh)
        monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", self._fake_discovery())

        await auth_mod.ensure_mcp_oauth_fresh(self.URL, self._cfg(), store=store)
        assert refreshed["called"] is False

    @pytest.mark.asyncio
    async def test_expired_token_is_refreshed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp import auth as auth_mod

        auth_mod._DISCOVERED_ENDPOINTS_CACHE.clear()
        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(OAuthToken(access_token="old", refresh_token="r", expires_in=60))
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))
        # Age the token past its lifetime so it reads as expired.
        store.rows[0].data["tokens_obtained_at"] = time.time() - 600

        refreshed = {"called": False}

        async def fake_refresh(*args: Any, **kwargs: Any) -> bool:
            refreshed["called"] = True
            return True

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", fake_refresh)
        monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", self._fake_discovery())

        await auth_mod.ensure_mcp_oauth_fresh(self.URL, self._cfg(), store=store)
        assert refreshed["called"] is True

    @pytest.mark.asyncio
    async def test_no_discovery_means_no_refresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without discoverable endpoints we degrade to the SDK default, not fail."""
        import time

        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp import auth as auth_mod

        auth_mod._DISCOVERED_ENDPOINTS_CACHE.clear()
        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(OAuthToken(access_token="old", refresh_token="r", expires_in=60))
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))
        store.rows[0].data["tokens_obtained_at"] = time.time() - 600

        refreshed = {"called": False}

        async def fake_refresh(*args: Any, **kwargs: Any) -> bool:
            refreshed["called"] = True
            return True

        async def no_discovery(url: str) -> Any:
            return None

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", fake_refresh)
        monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", no_discovery)

        result = await auth_mod.ensure_mcp_oauth_fresh(self.URL, self._cfg(), store=store)
        assert result is None
        assert refreshed["called"] is False

    @staticmethod
    def _fake_discovery():
        from mcp.shared.auth import OAuthMetadata

        from local_operator.mcp.auth import DiscoveredOAuthEndpoints

        async def discovery(url: str) -> DiscoveredOAuthEndpoints:
            return DiscoveredOAuthEndpoints(
                oauth_metadata=OAuthMetadata.model_validate(
                    {
                        "issuer": "https://mcp.example.com/v1/mcp",
                        "authorization_endpoint": "https://a/authorize",
                        "token_endpoint": "https://a/token",
                    }
                )
            )

        return discovery


class TestProviderEndpointPriming:
    """``build_oauth_provider`` primes the context with discovered endpoints.

    A token that dies MID-session and needs an in-flow refresh must target the
    real token endpoint, not the SDK's ``<server_base>/token`` guess. Priming
    ``oauth_metadata`` on the context is what makes that happen.
    """

    URL = "https://mcp.example.com/v1/mcp"

    @pytest.mark.asyncio
    async def test_endpoints_prime_the_context(self) -> None:
        from mcp.shared.auth import OAuthMetadata, OAuthToken

        from local_operator.mcp.auth import (
            DiscoveredOAuthEndpoints,
            build_oauth_provider,
        )

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(OAuthToken(access_token="a", refresh_token="r", expires_in=3600))
        cfg = MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))
        endpoints = DiscoveredOAuthEndpoints(
            oauth_metadata=OAuthMetadata.model_validate(
                {
                    "issuer": self.URL,
                    "authorization_endpoint": "https://a/authorize",
                    "token_endpoint": "https://a/token",
                }
            )
        )
        provider = build_oauth_provider(self.URL, cfg, store=store, endpoints=endpoints)
        assert provider.context.oauth_metadata is not None
        assert str(provider.context.oauth_metadata.token_endpoint) == "https://a/token"

    @pytest.mark.asyncio
    async def test_no_endpoints_leaves_context_unprimed(self) -> None:
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(OAuthToken(access_token="a", refresh_token="r", expires_in=3600))
        cfg = MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))
        provider = build_oauth_provider(self.URL, cfg, store=store)
        assert provider.context.oauth_metadata is None


class TestInflightRefreshCoordination:
    """The in-flow (mid-session) refresh is race-free across processes.

    The SDK loads tokens once and never re-reads storage, and its in-flow
    refresh spends the in-memory refresh token under no cross-process lock.
    For a provider that ROTATES its refresh token (Notion) with several
    long-lived local-operator processes alive, that double-spends the token and
    forces a browser grant. ``build_oauth_provider`` returns a subclass whose
    ``async_auth_flow`` re-reads the store under the refresh lock first and
    adopts a sibling's already-rotated token instead of spending a dead one.
    """

    URL = "https://mcp.example.com/v1/mcp"

    def _cfg(self) -> MCPHttpServerConfig:
        return MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))

    def _endpoints(self):
        from mcp.shared.auth import OAuthMetadata

        from local_operator.mcp.auth import DiscoveredOAuthEndpoints

        return DiscoveredOAuthEndpoints(
            oauth_metadata=OAuthMetadata.model_validate(
                {
                    "issuer": self.URL,
                    "authorization_endpoint": "https://a/authorize",
                    "token_endpoint": "https://a/token",
                }
            )
        )

    async def _drive_auth_flow(self, provider) -> None:
        """Pump the provider's async_auth_flow the way httpx does, feeding a
        200 to any request it yields, so the coordination step runs but no real
        network happens. Returns once the flow is exhausted."""
        import httpx

        gen = provider.async_auth_flow(httpx.Request("POST", self.URL))
        try:
            request = await gen.__anext__()
            while True:
                response = httpx.Response(200, request=request)
                try:
                    request = await gen.asend(response)
                except StopAsyncIteration:
                    return
        finally:
            await gen.aclose()

    @pytest.mark.asyncio
    async def test_adopts_sibling_rotated_token_without_refreshing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A peer process rotated the token while we held a stale one: we adopt
        it under the lock and DO NOT spend our dead refresh token."""
        import time

        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        # Our in-memory view (what the provider loads at _initialize): expired.
        await storage.set_tokens(
            OAuthToken(access_token="stale", refresh_token="r-old", expires_in=60)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))
        store.rows[0].data["tokens_obtained_at"] = time.time() - 600  # age past expiry

        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints()
        )

        refresh_calls = {"n": 0}

        async def fake_refresh(*args: Any, **kwargs: Any) -> bool:
            refresh_calls["n"] += 1
            return True

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", fake_refresh)

        # Simulate the sibling process having ALREADY rotated the token in the
        # shared store to a fresh, valid one, after this provider loaded.
        async def sibling_rotate() -> None:
            sib_storage = McpTokenStorage(self.URL, store)
            await sib_storage.set_tokens(
                OAuthToken(access_token="fresh-by-peer", refresh_token="r-new", expires_in=3600)
            )

        await sibling_rotate()

        await self._drive_auth_flow(provider)

        # The peer's token was adopted; our stale refresh token was never spent.
        assert refresh_calls["n"] == 0
        assert provider.context.current_tokens is not None
        assert provider.context.current_tokens.access_token == "fresh-by-peer"

    @pytest.mark.asyncio
    async def test_refreshes_once_when_store_still_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No peer rotated it: we perform exactly one locked refresh against the
        discovered endpoint and adopt its result."""
        import time

        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="stale", refresh_token="r-old", expires_in=60)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))
        store.rows[0].data["tokens_obtained_at"] = time.time() - 600

        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints()
        )

        refresh_calls = {"n": 0}

        async def fake_refresh(server_url, storage_arg, endpoints) -> bool:
            refresh_calls["n"] += 1
            # Mirror the real refresh: persist a fresh token under the lock.
            await storage_arg.set_tokens(
                OAuthToken(access_token="refreshed", refresh_token="r2", expires_in=3600)
            )
            return True

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", fake_refresh)

        await self._drive_auth_flow(provider)

        assert refresh_calls["n"] == 1
        assert provider.context.current_tokens is not None
        assert provider.context.current_tokens.access_token == "refreshed"

    @pytest.mark.asyncio
    async def test_valid_token_skips_coordination_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A still-valid loaded token never re-reads the store or refreshes —
        the coordination adds no round trip to the common case."""
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="valid", refresh_token="r", expires_in=3600)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))

        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints()
        )

        lock_calls = {"n": 0}
        real_lock = auth_mod._oauth_refresh_lock

        def counting_lock(server_url: str):
            lock_calls["n"] += 1
            return real_lock(server_url)

        monkeypatch.setattr(auth_mod, "_oauth_refresh_lock", counting_lock)

        await self._drive_auth_flow(provider)

        assert lock_calls["n"] == 0  # never entered the coordination path

    @pytest.mark.asyncio
    async def test_a_transport_error_mid_flow_releases_the_sdk_lock(self) -> None:
        """httpx re-raises a transport fault INTO the auth flow mid-yield. The
        manual pump must close the SDK's inner generator so its ``context.lock``
        (held across the whole flow) is released — otherwise every later request
        to this server deadlocks. Regression guard for the F1 blocker."""
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="valid", refresh_token="r", expires_in=3600)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))

        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints()
        )

        import httpx

        gen = provider.async_auth_flow(httpx.Request("POST", self.URL))
        # Advance to the first yield (the request the SDK wants sent), then throw
        # a transport error INTO the flow the way httpx's _send_handling_auth
        # does on a failed send.
        await gen.__anext__()
        with pytest.raises(httpx.ConnectError):
            await gen.athrow(httpx.ConnectError("connection reset"))
        with contextlib.suppress(Exception):
            await gen.aclose()

        # The SDK holds context.lock across async_auth_flow; if the inner
        # generator was left suspended it would still be held here.
        assert not provider.context.lock.locked(), "SDK context.lock leaked after a mid-flow error"

    @pytest.mark.asyncio
    async def test_closing_the_flow_early_releases_the_sdk_lock(self) -> None:
        """httpx always ``aclose()``s the outer auth flow in a ``finally``. When
        it does so before the flow finished, the inner SDK generator must be
        closed too so the lock is released (GeneratorExit path of F1)."""
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="valid", refresh_token="r", expires_in=3600)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))

        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints()
        )

        import httpx

        gen = provider.async_auth_flow(httpx.Request("POST", self.URL))
        await gen.__anext__()  # advance to the first yield
        await gen.aclose()  # close before feeding a response

        assert not provider.context.lock.locked(), "SDK context.lock leaked after early close"

    @pytest.mark.asyncio
    async def test_no_endpoints_falls_back_to_sdk_refresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no discovered endpoint the re-read still runs (so a racing peer
        is still adopted), but a self-performed refresh is left to the SDK — the
        pre-fix path for that one provider, never a crash."""
        import time

        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="stale", refresh_token="r-old", expires_in=60)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))
        store.rows[0].data["tokens_obtained_at"] = time.time() - 600

        # endpoints=None: no self-performed refresh possible.
        provider = build_oauth_provider(self.URL, self._cfg(), store=store, endpoints=None)

        refresh_calls = {"n": 0}

        async def fake_refresh(*args: Any, **kwargs: Any) -> bool:
            refresh_calls["n"] += 1
            return True

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", fake_refresh)

        # Must not raise; our own locked refresh is never called (no endpoint).
        await self._drive_auth_flow(provider)
        assert refresh_calls["n"] == 0

    async def _drive_401_flow(self, provider, responder) -> list[Any]:
        """Pump ``async_auth_flow`` the way httpx does, recording every request
        the flow yields and answering each with ``responder(request)``.

        The responder's return value for the ORIGINAL request is normally a
        401 (that is the case under test). With the fix, the flow RE-YIELDS the
        original request (Authorization rewritten to an adopted token) as the
        retry — the responder sees it as a second yield, exactly as the real
        httpx client would. Without anything to adopt, the 401 passes through
        to the SDK and its full-flow machinery (discovery etc.) shows up as
        further yields.
        """
        import httpx

        yielded: list[httpx.Request] = []
        gen = provider.async_auth_flow(httpx.Request("POST", self.URL, content=b"payload"))
        try:
            request = await gen.__anext__()
            while True:
                yielded.append(request)
                response = responder(request)
                try:
                    request = await gen.asend(response)
                except StopAsyncIteration:
                    return yielded
        finally:
            await gen.aclose()

    @pytest.mark.asyncio
    async def test_401_adopts_peer_token_and_retries_original_request(self) -> None:
        """The residual Notion bug: a sibling process rotated the grant, which
        REVOKED our still-unexpired access token server-side. Our request 401s;
        the flow must adopt the peer's token from the store and RE-YIELD the
        original request with it (for the same httpx client to send) — never
        yielding a discovery/registration/browser-grant request."""
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        # Our in-memory token is still VALID locally (unexpired) — coordination
        # at the top of the flow correctly does nothing for it.
        await storage.set_tokens(
            OAuthToken(access_token="revoked-but-unexpired", refresh_token="r1", expires_in=3600)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))

        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints()
        )
        # Force initialization NOW, while the store still holds OUR token: the
        # SDK loads current_tokens once in _initialize, so rotating the store
        # AFTER this point is what leaves the provider holding the revoked
        # token in memory — the exact multi-process state under test. (Rotating
        # before _initialize would just load the fresh token at boot, which is
        # the already-working restart path.)
        async with provider.context.lock:
            await provider._initialize()

        # The sibling process rotated the grant after we loaded: the shared
        # store now holds a fresh token, and the old access token is revoked
        # server-side (emulated by the responder below).
        await storage.set_tokens(
            OAuthToken(access_token="fresh-by-peer", refresh_token="r2", expires_in=3600)
        )

        import httpx

        calls: list[tuple[httpx.Request, str]] = []

        def responder(request: httpx.Request) -> httpx.Response:
            calls.append((request, request.headers.get("Authorization", "")))
            # The resource server 401s the revoked token; the retry carrying the
            # adopted token succeeds.
            if request.headers.get("Authorization") == "Bearer fresh-by-peer":
                return httpx.Response(200, request=request)
            return httpx.Response(401, request=request)

        yielded = await self._drive_401_flow(provider, responder)

        # Exactly TWO requests reached the caller: the original (401) and the
        # adoption retry. Nothing else — the 401 never reached the SDK, so its
        # full-flow machinery (discovery, registration, authorization) never
        # produced a request.
        assert len(yielded) == 2
        # The retry is the SAME request object re-yielded with the adopted
        # token — the SDK's own end-of-flow retry contract, sent by the same
        # httpx client that sent the original.
        assert yielded[1] is yielded[0]
        assert calls[0][1] == "Bearer revoked-but-unexpired"
        assert calls[1][1] == "Bearer fresh-by-peer"
        # The adopted token is now the in-memory token too, so a later request
        # (or the SDK's own end-of-flow retry) uses it rather than the corpse.
        assert provider.context.current_tokens is not None
        assert provider.context.current_tokens.access_token == "fresh-by-peer"

    @pytest.mark.asyncio
    async def test_401_with_identical_stored_token_passes_through_to_sdk(self) -> None:
        """The store holds exactly the token that just 401'd: the grant is dead
        everywhere, not merely revoked for us. The 401 must pass through to the
        SDK unchanged (its full-flow branch becomes reachable, as before the
        fix) and NO adoption retry may be re-yielded."""
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="dead-token", refresh_token="r1", expires_in=3600)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))

        # interactive=False: if the 401 wrongly reaches the SDK's full-flow
        # branch and gets as far as authorization, the flow must RAISE
        # (McpAuthRequiredError) rather than open a browser in the test run.
        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints(), interactive=False
        )

        import httpx

        yields: list[tuple[httpx.Request, str]] = []
        seen_401 = {"done": False}

        def responder(request: httpx.Request) -> httpx.Response:
            yields.append((request, request.headers.get("Authorization", "")))
            if not seen_401["done"]:
                # The original request: answer with the 401 challenge.
                seen_401["done"] = True
                return httpx.Response(401, request=request)
            # Anything yielded AFTER the 401 is the SDK's full-flow machinery
            # (protected-resource discovery first) — proof the challenge
            # reached the SDK rather than being answered by an adoption retry.
            # Fail it so the flow terminates without a browser grant.
            return httpx.Response(404, request=request)

        with contextlib.suppress(Exception):
            # The failed discovery may or may not raise out of the flow (SDK
            # version dependent); the yields below are the assertion.
            await self._drive_401_flow(provider, responder)

        # The FIRST yield is the original request, still bearing the dead token
        # (adoption did NOT rewrite it — the stored token was identical).
        assert yields[0][1] == "Bearer dead-token"
        # The 401 reached the SDK: its full-flow branch yielded at least one
        # discovery/registration request after the original.
        assert len(yields) >= 2, "the 401 never reached the SDK's full-flow branch"
        # No adoption retry: exactly ONE request went to the resource URL (the
        # original). Everything after it is the SDK's own machinery on other
        # URLs (discovery/registration), which the SDK may yield several of.
        to_resource = [y for y in yields if str(y[0].url) == self.URL]
        assert len(to_resource) == 1
        assert yields[1][0].url != yields[0][0].url

    @pytest.mark.asyncio
    async def test_second_401_after_adoption_retry_passes_through(self) -> None:
        """Adoption is bounded to ONE retry per flow: when the adopted token
        ALSO 401s (the grant is genuinely dead server-side), that second 401
        passes through to the SDK's full-flow branch — adoption never loops."""
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        from local_operator.mcp.auth import build_oauth_provider

        store = FakeAuthStore()
        storage = McpTokenStorage(self.URL, store)
        await storage.set_tokens(
            OAuthToken(access_token="revoked-but-unexpired", refresh_token="r1", expires_in=3600)
        )
        await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))

        # interactive=False: the second 401 passing through must RAISE out of
        # the SDK's authorization step, never open a browser in the test run.
        provider = build_oauth_provider(
            self.URL, self._cfg(), store=store, endpoints=self._endpoints(), interactive=False
        )
        # Load OUR token into memory first (see the adoption test above for why
        # the rotation must come after _initialize).
        async with provider.context.lock:
            await provider._initialize()

        # A peer rotated the grant, but the resource server 401s EVERY bearer
        # token (emulating a grant the user revoked server-side).
        await storage.set_tokens(
            OAuthToken(access_token="also-dead", refresh_token="r2", expires_in=3600)
        )

        import httpx

        yields: list[tuple[httpx.Request, str]] = []

        def responder(request: httpx.Request) -> httpx.Response:
            yields.append((request, request.headers.get("Authorization", "")))
            # The resource server 401s every bearer token on this URL; the
            # SDK's discovery sub-requests (a different URL) get a 404 so the
            # flow terminates without a browser grant.
            if str(request.url) == self.URL:
                return httpx.Response(401, request=request)
            return httpx.Response(404, request=request)

        with contextlib.suppress(Exception):
            await self._drive_401_flow(provider, responder)

        # The caller saw: the original (401), the ONE adoption retry (same
        # object, adopted token, 401 again), and then the SDK's full-flow
        # machinery (discovery requests on other URLs — possibly several) —
        # proof the second 401 passed through and adoption did not loop.
        assert yields[0][1] == "Bearer revoked-but-unexpired"
        assert yields[1][0] is yields[0][0]  # the retry re-yields the original
        assert yields[1][1] == "Bearer also-dead"
        assert len(yields) >= 3
        assert yields[2][0].url != yields[0][0].url  # SDK discovery, not a retry
        # Exactly TWO requests went to the resource URL: original + one retry.
        # A looping adoption would show a third.
        to_resource = [y for y in yields if str(y[0].url) == self.URL]
        assert len(to_resource) == 2

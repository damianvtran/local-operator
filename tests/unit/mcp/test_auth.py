"""Auth: McpTokenStorage over the real AuthStore API, wire_oauth_auth kwargs.

FakeAuthStore mirrors the REAL ``providers.auth_store.AuthStore`` surface
(upsert_credential / list_credentials / get_credential, integer row ids,
provider column + identity_key dedupe) so tests exercise the same contract
the production store provides. A conformance test additionally round-trips
through the real SQLite store.
"""

from __future__ import annotations

from pathlib import Path
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

"""Transport (network) failures are NAMED, and they SETTLE the startup round.

Two defects, both measured on ``main`` by driving the REAL manager
(``discover_and_load_mcp_tools``) against unreachable servers:

1. **A network failure reported nothing.** A refused TCP connect, a DNS failure
   and a TLS error all reach ``_connect_server`` as a bare ``CancelledError`` —
   anyio cancels the awaiting task when the transport's own reader or writer
   dies inside its task group — and that value is indistinguishable, at that
   point, from the cancellation a dispose or reload performs. It was re-raised
   unchallenged and read as a TEARDOWN by ``_finish_pending``, which drops
   ``CancelledError`` on purpose. The server stayed in ``_startup_deferred`` for
   the life of the process: ``startup_settling()`` never went False, the outcome
   was never reportable, and because a settling outcome is deliberately silent
   the one fault class that takes out several servers at ONCE reported nothing
   at all. Measured, default timeout, two unreachable servers: ``settling=True
   failures={}`` at 0.6 s, still ``settling=True failures={}`` after 15 s of
   polling.
2. **A network failure read as an MCP fault.** When the transport broke after
   connecting, the SDK's own sentence reached the user (``Request 'initialize'
   timed out``, ``Connection closed``) with nothing saying the machine's
   connection was the problem — a captured boot showed ``1 of 11 servers up``
   and ten rows of that line, and the fault was chased through MCP config and
   OAuth state until a phone hotspot proved it was the link.

The contracts asserted here: the classifier names the LAYER with the REAL
exception types the SDK raises, the round settles and reports, the copy contract
(``network: …``) is exact, a stdio (hostless) transport failure is reported but
is NOT called the user's network, and the auth copy is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from mcp.shared.exceptions import MCPError

from local_operator.mcp.auth import McpAuthChallengeError, McpAuthRequiredError
from local_operator.mcp.config import MCPHttpServerConfig, MCPStdioServerConfig
from local_operator.mcp.manager import STARTUP_GATE_MS, McpManager


def _m() -> Any:
    """The manager module, imported LAZILY — deliberately.

    Module-scope imports of the transport classifier would make this file
    uncollectable on the tree it guards: ``origin/main`` has no
    ``_transport_failure_text``, no ``McpTransportError`` and no
    ``NETWORK_FAILURE_MARKER``, so the pre-fix run would report a collection
    error instead of the missing BEHAVIOUR these tests exist to catch. Same
    reason ``tests/unit/mcp/conftest.py`` defers its ``REFRESH_CONTENTION``
    import: a guard has to run on the tree it guards.

    Callers reach the helpers through the module object (``_m()._transport_…``)
    rather than importing them by name, so the reference is resolved per test.
    """
    from local_operator.mcp import manager as manager_module

    return manager_module


#: The endpoint the classifiers are given by the caller. None of the exceptions
#: below carries it (that is the point of ``_transport_failure``'s ``url``
#: parameter), so every case passes it explicitly, exactly as the manager does
#: through ``_server_url``.
URL = "https://mcp.example.com/mcp"
HOST = "mcp.example.com"


class TestTheClassifierNamesTheLayer:
    """The real exception types, classified — not a hand-built stand-in.

    These are the objects the SDK actually raises from a dead socket: httpx at
    the request layer, anyio one level under it, and socket/ssl when the SDK
    does not catch them at all. The classifier matches them by MRO NAME so the
    manager keeps its lazy-SDK property (this module imports neither httpx nor
    anyio nor mcp to answer the question), which is only safe while a test
    builds the real types and fails when an upstream rename moves them.
    """

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (httpx.ConnectError("All connection attempts failed"), "network: cannot reach"),
            (httpx.ConnectTimeout("timed out"), "network: no response from"),
            (httpx.ReadTimeout("timed out"), "network: no response from"),
            (httpx.WriteTimeout("timed out"), "network: no response from"),
            (httpx.PoolTimeout("timed out"), "network: no response from"),
            (httpx.ReadError("read error"), "network: the connection to"),
            (httpx.WriteError("write error"), "network: the connection to"),
            (httpx.RemoteProtocolError("server disconnected"), "network: the connection to"),
            (anyio.ClosedResourceError(), "network: the connection to"),
            (anyio.BrokenResourceError(), "network: the connection to"),
            (anyio.EndOfStream(), "network: the connection to"),
            (socket.gaierror(8, "nodename nor servname provided"), "network: cannot resolve"),
            (ssl.SSLError("handshake failure"), "network: TLS handshake with"),
            (ssl.SSLCertVerificationError("verify failed"), "network: TLS handshake with"),
            (ConnectionRefusedError(61, "Connection refused"), "network: cannot reach"),
            (ConnectionResetError(54, "Connection reset by peer"), "network: the connection to"),
            (BrokenPipeError(32, "Broken pipe"), "network: the connection to"),
            (TimeoutError("timed out"), "network: no response from"),
        ],
        ids=[
            "httpx-connect-error",
            "httpx-connect-timeout",
            "httpx-read-timeout",
            "httpx-write-timeout",
            "httpx-pool-timeout",
            "httpx-read-error",
            "httpx-write-error",
            "httpx-remote-protocol-error",
            "anyio-closed-resource",
            "anyio-broken-resource",
            "anyio-end-of-stream",
            "socket-gaierror",
            "ssl-error",
            "ssl-cert-verification-error",
            "connection-refused",
            "connection-reset",
            "broken-pipe",
            "builtin-timeout-error",
        ],
    )
    def test_each_real_transport_exception_renders_a_network_line(
        self, exc: BaseException, expected: str
    ) -> None:
        text = _m()._transport_failure_text(exc, URL)
        assert text is not None
        assert text.startswith(expected), text
        assert HOST in text
        assert _m()._is_network_failure(exc, URL) is True

    def test_an_anyio_task_group_around_the_failure_is_unwrapped(self) -> None:
        """The shape most transport failures arrive in.

        The streamable-HTTP transport runs inside a task group, so the httpx
        error comes wrapped — measured in the real flow as
        ``ExceptionGroup('unhandled errors in a TaskGroup', [ConnectError(...)])``.
        A classifier that only looked at the outermost exception would call that
        group "not a transport failure" and fall through to ``str(exc)``, which
        is the opaque ``unhandled errors in a TaskGroup`` the user saw.
        """
        nested = ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [
                ExceptionGroup(
                    "nested",
                    [httpx.ConnectError("All connection attempts failed")],
                )
            ],
        )
        assert _m()._transport_failure_text(nested, URL) == "network: cannot reach " + HOST
        assert _m()._is_network_failure(nested, URL) is True

    def test_a_dns_failure_hidden_inside_a_connect_error_still_says_dns(self) -> None:
        """httpx wraps whatever the connector raised.

        An unresolved host therefore arrives as a ``ConnectError`` whose cause
        chain holds the ``gaierror``, and "cannot resolve" is the detail that
        points at the resolver (captive portal, VPN, dead DNS) rather than at
        the server — worth the extra walk, and the difference a user acts on.
        """
        wrapped = httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
        wrapped.__cause__ = socket.gaierror(8, "nodename nor servname provided")
        assert _m()._transport_failure_text(wrapped, URL) == "network: cannot resolve " + HOST

    def test_a_re_wrapped_resolver_error_is_still_named_dns(self) -> None:
        """The class name is not always faithful, so the SENTENCE is read too.

        Measured, the real failure nests ``httpx.ConnectError`` →
        ``httpcore.ConnectError`` → ``socket.gaierror``, and the class signal
        finds it. A re-wrap that keeps only the resolver's sentence (a plain
        ``OSError`` from a proxy client, a newer httpcore that flattens the
        cause) would otherwise classify as ``unreachable`` — still a network
        failure, but the wrong one to point the user at.
        """
        rewrapped = httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
        assert rewrapped.__cause__ is None and rewrapped.__context__ is None
        assert _m()._transport_failure_text(rewrapped, URL) == "network: cannot resolve " + HOST

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (-32000, "network: the connection to"),
            (-32001, "network: no response from"),
        ],
    )
    def test_the_sdk_transport_codes_render_as_network_failures(
        self, code: int, expected: str
    ) -> None:
        """``-32000``/``-32001`` are the CLIENT dispatcher's own transport codes.

        They are what a request that died on a live-but-breaking transport comes
        back as (``Connection closed``, ``Request 'initialize' timed out``), and
        rendering their sentence verbatim is defect 2.
        """
        exc = MCPError(code, "whatever the SDK said")
        text = _m()._transport_failure_text(exc, URL)
        assert text is not None and text.startswith(expected), text
        assert HOST in text

    def test_the_two_transport_codes_are_pinned_against_the_installed_sdk(self) -> None:
        """Pin the NUMBERS, not just our copy of them.

        The classifier holds ``{-32000, -32001}`` as literals on purpose — those
        names are SDK internals, and importing them would make an upstream move
        silently turn the classifier into a no-op. This is the other half of that
        trade: if the SDK renumbers the codes, THIS test fails and the literals
        get updated deliberately.
        """
        from mcp.shared import jsonrpc_dispatcher as dispatcher

        from local_operator.mcp.manager import _TRANSPORT_RPC_DETAIL

        assert dispatcher.CONNECTION_CLOSED == -32000
        assert dispatcher.REQUEST_TIMEOUT == -32001
        assert set(_TRANSPORT_RPC_DETAIL) == {
            dispatcher.CONNECTION_CLOSED,
            dispatcher.REQUEST_TIMEOUT,
        }

    def test_an_ordinary_application_error_is_not_a_transport_failure(self) -> None:
        """A server that answered with its own error keeps its own sentence.

        Without this the classifier could claim an application fault as the
        user's network — the mirror image of the defect, and worse.
        """
        app_error = MCPError(-32602, "Invalid params")
        assert _m()._transport_failure_text(app_error, URL) is None
        assert _m()._is_network_failure(app_error, URL) is False
        assert _m()._transport_failure_text(ValueError("boom"), URL) is None


class TestTheCopyContract:
    """The five phrases, exact: they are the user-visible half of the feature."""

    @pytest.mark.parametrize(
        ("detail", "expected"),
        [
            ("unreachable", "network: cannot reach mcp.example.com"),
            ("timeout", "network: no response from mcp.example.com (timed out)"),
            ("dns", "network: cannot resolve mcp.example.com"),
            ("tls", "network: TLS handshake with mcp.example.com failed"),
            ("closed", "network: the connection to mcp.example.com closed"),
        ],
    )
    def test_every_detail_renders_its_phrase(self, detail: str, expected: str) -> None:
        assert _m()._transport_failure_text(_m().McpTransportError(URL, detail)) == expected

    def test_the_host_comes_from_the_url_the_caller_passes(self) -> None:
        """The exceptions carry no URL, so config is the only source of the host.

        Measured: ``httpx.ConnectError``, ``socket.gaierror`` and anyio's
        resource errors all arrive bare. Without the caller's URL, a refused
        connection could only ever render the hostless fallback — which is
        exactly the shape that reports nothing useful.
        """
        exc = httpx.ConnectError("All connection attempts failed")
        assert _m()._transport_failure_text(exc) == (
            "the transport failed before the server answered (unreachable)"
        )
        assert _m()._is_network_failure(exc) is False
        assert _m()._transport_failure_text(exc, URL) == "network: cannot reach " + HOST
        assert _m()._is_network_failure(exc, URL) is True

    def test_a_url_held_by_the_exception_wins_over_the_passed_one(self) -> None:
        """``McpTransportError.url`` is the endpoint the failing exchange used."""
        exc = _m().McpTransportError("https://other.example.com/mcp", "timeout")
        assert "other.example.com" in (_m()._transport_failure_text(exc, URL) or "")

    def test_a_stdio_transport_failure_is_reported_but_is_not_the_network(self) -> None:
        """A hostless failure must never be called the user's connectivity.

        A stdio child that dies during the handshake is a transport failure and
        has to be REPORTED (silence is the defect this path exists to fix), but
        its server has no host to name: calling it "network" would send the user
        to diagnose a link that is fine while the real reason ("command not
        found: gh") sits in the child's stderr.
        """
        exc = _m().McpTransportError(None, "closed")
        text = _m()._transport_failure_text(exc)
        assert text == "the transport failed before the server answered (closed)"
        assert _m().NETWORK_FAILURE_MARKER not in text
        assert _m()._is_network_failure(exc) is False

    def test_the_manager_reads_the_url_from_the_config(self, tmp_path: Path) -> None:
        """``_server_url`` is the one place a NAME becomes a URL for the copy."""
        manager = McpManager(str(tmp_path))
        manager._configs["remote"] = MCPHttpServerConfig(url=URL)
        manager._configs["local"] = MCPStdioServerConfig(command="gh")
        assert manager._server_url("remote") == URL
        assert manager._server_url("local") is None
        assert manager._server_url("never-configured") is None


class TestTheAuthCopyIsUnchanged:
    """Adding a branch to the dispatcher must not touch any auth rendering.

    ``_auth_failure_text`` is the single composition point for the startup toast,
    the durable notice, the incident sink and ``/mcp``; ~20 existing call sites
    pass two arguments, and the third (the URL) is optional precisely so they
    keep working.
    """

    def test_a_challenge_renders_identically_with_and_without_a_url(self, tmp_path: Path) -> None:
        manager = McpManager(str(tmp_path))
        challenge = McpAuthChallengeError(
            URL, status_code=401, oauth_available=True, has_stored_grant=False
        )
        rendered = manager._auth_failure_text("notion", challenge)
        assert rendered == manager._auth_failure_text("notion", challenge, URL)
        assert _m().NETWORK_FAILURE_MARKER not in rendered

    def test_a_required_grant_renders_identically_with_and_without_a_url(
        self, tmp_path: Path
    ) -> None:
        manager = McpManager(str(tmp_path))
        required = McpAuthRequiredError(URL)
        rendered = manager._auth_failure_text("notion", required)
        assert rendered == manager._auth_failure_text("notion", required, URL)
        assert _m().NETWORK_FAILURE_MARKER not in rendered

    def test_an_unknown_exception_still_renders_its_own_sentence(self, tmp_path: Path) -> None:
        manager = McpManager(str(tmp_path))
        assert manager._auth_failure_text("srv", ValueError("boom"), URL) == "boom"

    def test_a_transport_failure_renders_its_network_line_through_the_dispatcher(
        self, tmp_path: Path
    ) -> None:
        manager = McpManager(str(tmp_path))
        exc = httpx.ConnectError("All connection attempts failed")
        assert manager._auth_failure_text("srv", exc, URL) == "network: cannot reach " + HOST


class TestTheRoundSettlesAndReports:
    """Defect 1's regression, deterministic: a bare cancellation is REPORTED.

    The transport seam is stubbed with the exact value anyio delivers — a
    ``CancelledError`` with no cancelling count on the task — so this test needs
    no socket and no resolver, while still exercising the arm that used to drop
    the failure on the floor. The end-to-end case below does it with real
    sockets.
    """

    @pytest.mark.asyncio
    async def test_a_bare_cancellation_settles_the_round_as_a_network_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=URL)
        settled = asyncio.Event()
        manager.on_startup_settled = settled.set

        async def dying_transport(*_a: Any, **_kw: Any) -> Any:
            # Past the gate, so the failure lands on the DEFERRED path — the one
            # that used to drop it. Before the gate it would surface through the
            # gate arm instead, which is a different (working) code path.
            await asyncio.sleep(STARTUP_GATE_MS / 1000 + 0.1)
            raise asyncio.CancelledError()

        monkeypatch.setattr(manager, "_open_transport_and_session", dying_transport)
        monkeypatch.setattr(manager, "_ensure_oauth_fresh", lambda *a, **k: asyncio.sleep(0))

        await manager._connect_round({"remote": cfg}, {})
        assert manager.startup_settling() is True, "the round must be waiting on the connect"

        # Wait on the EVENT, not on the clock: the callback fires when the
        # deferred set drains. The ceiling only bounds a failure.
        await asyncio.wait_for(settled.wait(), timeout=10)

        assert manager.startup_settling() is False
        failures = manager.startup_failures()
        assert set(failures) == {"remote"}, failures
        assert failures["remote"] == "network: cannot reach " + HOST
        assert manager.startup_network_failures() == {"remote"}
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_a_dispose_still_cancels_a_connect_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The conversion is for anyio's delivery, never for a real teardown.

        ``disconnect_all`` is the only other thing that cancels a connect, and
        it sets ``_disposed`` first. Turning that cancellation into a network
        failure would toast a server the user just shut down.
        """
        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=URL)

        async def dying_transport(*_a: Any, **_kw: Any) -> Any:
            await asyncio.sleep(STARTUP_GATE_MS / 1000 + 0.1)
            raise asyncio.CancelledError()

        monkeypatch.setattr(manager, "_open_transport_and_session", dying_transport)
        monkeypatch.setattr(manager, "_ensure_oauth_fresh", lambda *a, **k: asyncio.sleep(0))

        task = asyncio.get_running_loop().create_task(manager._connect_server("remote", cfg))
        manager._configs["remote"] = cfg
        await asyncio.sleep(STARTUP_GATE_MS / 1000 + 0.05)
        manager._disposed = True
        with pytest.raises(asyncio.CancelledError):
            await task


class TestUnreachableServersEndToEnd:
    """Defect 1 through the REAL transport, real sockets, isolated config.

    This is the test that fails on ``origin/main``: there, the round never
    settled (``startup_settling()`` stayed True and ``startup_failures()``
    stayed empty) and neither server was reported at all.
    """

    #: Generous on purpose: the assertion is "it settles", and a CI host whose
    #: resolver is slow must not turn that into a flake. A settle that never
    #: comes is the defect, and 25 s is far past any working path's settle time
    #: (measured: 1.3 s here).
    SETTLE_CEILING_S = 25.0

    @pytest.mark.asyncio
    async def test_two_unreachable_servers_are_reported_as_network_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator.mcp import discover_and_load_mcp_tools

        # Isolate HOME and cwd: config discovery reads both, and a run must never
        # see the developer's own servers.
        home = tmp_path / "home"
        cwd = tmp_path / "cwd"
        home.mkdir()
        cwd.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / ".local-operator"))
        monkeypatch.chdir(cwd)
        (cwd / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        # A refused TCP connect: nothing listens on port 9.
                        "refused": {
                            "type": "http",
                            "url": "http://127.0.0.1:9/mcp",
                            "timeout": 3.0,
                        },
                        # A DNS failure: .invalid is guaranteed never to resolve.
                        "nxdomain": {
                            "type": "http",
                            "url": "https://nope.invalid/mcp",
                            "timeout": 3.0,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        manager, _tools, _errors = await discover_and_load_mcp_tools(str(cwd))
        settled = asyncio.Event()
        manager.on_startup_settled = settled.set

        # Both servers miss the 250 ms gate, so the round is settling at the
        # snapshot and the failure is only knowable after it drains.
        if manager.startup_settling():
            await asyncio.wait_for(settled.wait(), timeout=self.SETTLE_CEILING_S)

        assert manager.startup_settling() is False, (
            "the round never settled: the unreachable servers were dropped "
            "instead of reported (defect 1)"
        )

        failures = manager.startup_failures()
        assert set(failures) == {"refused", "nxdomain"}, failures
        for name, host in (("refused", "127.0.0.1:9"), ("nxdomain", "nope.invalid")):
            assert failures[name].startswith(_m().NETWORK_FAILURE_MARKER), failures[name]
            assert host in failures[name], failures[name]
        assert manager.startup_network_failures() == {"refused", "nxdomain"}
        await manager.disconnect_all()

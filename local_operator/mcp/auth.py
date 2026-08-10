"""MCP OAuth support on the official SDK's ``OAuthClientProvider``.

Flow (official SDK PKCE + RFC 7591 DCR under the hood):

- ``build_oauth_provider(server_url, cfg)`` is the entry point: it wires the
  provider AND primes it with the stored token's expiry, which is what makes a
  restart spend the refresh token instead of re-running a browser grant.
- ``wire_oauth_auth(server_url, cfg)`` returns the ``OAuthClientProvider``
  kwargs: client metadata with a loopback redirect URI, a token storage bound
  to the shared credential store, and a :class:`LoopbackAuthFlow` that
  actually LISTENS on that redirect URI (with a pasted-URL race for browsers
  that cannot reach this machine).
- ``McpTokenStorage`` is the SDK ``TokenStorage``: one row per server URL in
  the real ``providers.auth_store.AuthStore``, keyed ``mcp_oauth:<url>``, with
  the token's issue time recorded so its lifetime survives the process.

Credential mapping onto the REAL AuthStore API (MCP-03): the store is keyed
by integer row id + ``provider`` column + ``identity_key``, so the logical
credential id ``mcp_oauth:<server_url>`` maps to ``provider='mcp-oauth'`` +
``identity_key=<server_url>`` (carried through the payload's ``project_id``
field, which the store's dedupe logic picks up). Reads filter
``list_credentials('mcp-oauth')`` by ``identity_key``; writes go through
``upsert_credential``, which updates the row in place on re-auth.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from local_operator.mcp.callback_page import callback_response

from pydantic import AnyUrl

if TYPE_CHECKING:
    # The SDK is an optional extra: these names are needed for annotations
    # only, so importing them here keeps this module importable without it.
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthToken,
    )

    from local_operator.mcp.config import MCPServerConfig
    from local_operator.providers.auth_store import StoredCredential

logger = logging.getLogger(__name__)

# Logical credential id prefix for managed MCP OAuth credentials (URL-keyed).
MCP_OAUTH_CREDENTIAL_PREFIX = "mcp_oauth:"

# Provider column value in the shared auth_credentials table.
MCP_OAUTH_PROVIDER = "mcp-oauth"

DEFAULT_CALLBACK_PORT = 3000
DEFAULT_CALLBACK_PATH = "/callback"

#: Payload key holding the wall-clock time (epoch seconds) the stored access
#: token was issued. Not part of the SDK's ``OAuthToken`` — see
#: :meth:`McpTokenStorage.stored_token_expiry` for why we have to record it.
TOKENS_OBTAINED_AT_KEY = "tokens_obtained_at"


def mcp_oauth_credential_id(server_url: str) -> str:
    """Stable logical credential id for one MCP server's OAuth grant."""
    return f"{MCP_OAUTH_CREDENTIAL_PREFIX}{server_url}"


@runtime_checkable
class StructuralAuthStore(Protocol):
    """The slice of ``providers.auth_store.AuthStore`` this module consumes.

    Redefined to the REAL store's methods (MCP-03) so a test fake mirrors
    reality: integer-keyed rows, ``provider`` column, ``identity_key`` dedupe.
    """

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> StoredCredential:
        """Insert, or update the row for the same identity; returns the row."""
        ...

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[StoredCredential]:
        """Enabled credential rows (all providers or one), oldest first."""
        ...

    def get_credential(self, credential_id: int) -> StoredCredential | None:
        """Return one row by integer id, or ``None``."""
        ...


@runtime_checkable
class ManagedAuthStore(StructuralAuthStore, Protocol):
    """A store whose lifetime the MCP manager may own, and therefore close.

    ``McpManager`` constructs its own store when none is injected; that one
    has to be released on ``disconnect_all``, so the closing surface belongs
    in the type rather than being discovered at teardown.
    """

    def close(self) -> None:
        """Release the underlying database handle."""
        ...


def _resolve_store(store: StructuralAuthStore | None) -> StructuralAuthStore | None:
    """Return ``store`` or lazily construct the real ``AuthStore``.

    The providers import is deferred: the MCP package must stay importable in
    environments where the providers stream's dependencies are unavailable.
    """
    if store is not None:
        return store
    try:
        from local_operator.providers.auth_store import AuthStore

        return AuthStore()
    except Exception:  # pragma: no cover - environment dependent
        logger.debug("providers.auth_store unavailable; MCP OAuth storage disabled", exc_info=True)
        return None


class McpTokenStorage:
    """SDK ``TokenStorage`` over the shared credential store.

    One instance per server URL: the SDK calls ``get_tokens`` / ``set_tokens``
    (and the client-info pair for dynamic registration) against this object,
    and we round-trip the pydantic models through one credential row under
    provider ``mcp-oauth`` with ``identity_key = server_url``. All reads
    tolerate a missing store or missing row by returning ``None`` (the SDK
    then starts a fresh flow).
    """

    def __init__(self, server_url: str, store: StructuralAuthStore | None = None) -> None:
        self.server_url = server_url
        self.credential_id = mcp_oauth_credential_id(server_url)
        self._store = _resolve_store(store)
        # Snapshot ``updated_at`` NOW, before anything this process does can
        # move it. The store stamps that column on every write, including the
        # client-info writes that ``wire_oauth_auth`` makes moments after
        # constructing us — so reading it later would report this process's own
        # seed as the token's issue time. See :meth:`stored_token_expiry`.
        row = self._read_row()
        self._row_updated_at_at_open: int = row.updated_at if row is not None else 0

    def _read_row(self) -> StoredCredential | None:
        """The credential row for this server URL, or ``None`` (no store/no row)."""
        store = self._store
        if store is None:
            return None
        try:
            rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        except Exception:
            logger.debug("MCP token read failed for %s", self.credential_id, exc_info=True)
            return None
        for row in rows:
            if row.identity_key == self.server_url:
                return row
        return None

    def _read(self) -> dict[str, Any] | None:
        """Row payload for this server URL, or ``None`` (no store/no row)."""
        row = self._read_row()
        if row is None:
            return None
        data = row.data
        return dict(data) if isinstance(data, dict) else None

    def _write(self, creds: dict[str, Any]) -> None:
        store = self._store
        if store is None:
            return
        payload = dict(creds)
        # The store stamps ``type`` into the data it persists; carrying it
        # back on the next write would make _identity_key_for short-circuit
        # to None (api_key rows get no identity key) and INSERT a duplicate
        # row instead of updating in place.
        payload.pop("type", None)
        # The store dedupes by identity_key derived from the payload's
        # project_id (first non-empty of org_id/account_id/email/project_id);
        # pinning it to the server URL gives one row per server, upserted in
        # place on re-auth.
        payload["project_id"] = self.server_url
        try:
            store.upsert_credential(MCP_OAUTH_PROVIDER, payload)
        except Exception:
            logger.debug("MCP token write failed for %s", self.credential_id, exc_info=True)

    # --- SDK TokenStorage protocol ---------------------------------------

    async def get_tokens(self) -> OAuthToken | None:
        """Stored access/refresh tokens as an ``OAuthToken``, or ``None``."""
        creds = self._read()
        tokens = creds.get("tokens") if creds is not None else None
        if not isinstance(tokens, dict):
            return None
        try:
            from mcp.shared.auth import OAuthToken

            return OAuthToken.model_validate(tokens)
        except Exception:
            logger.debug("Stored MCP tokens invalid for %s", self.credential_id, exc_info=True)
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist fresh/refreshed tokens (access + refresh together).

        The issuing WALL-CLOCK time is written alongside them. ``OAuthToken``
        carries only the relative ``expires_in`` the server quoted, which is
        meaningless once the process that received it has exited — and the SDK
        reloads tokens without reloading any notion of when they die (see
        :func:`stored_token_expiry`). Recording the moment of issue is what
        turns that relative number back into an absolute deadline on the next
        launch.
        """
        creds = self._read() or {}
        creds["tokens"] = tokens.model_dump(mode="json")
        creds[TOKENS_OBTAINED_AT_KEY] = time.time()
        self._write(creds)

    def stored_token_expiry(self) -> float | None:
        """Epoch seconds at which the stored access token expires, if knowable.

        ``None`` means "no opinion" — no row, no token, or a token the server
        quoted no lifetime for — and callers must then leave the SDK's own
        default (treat as valid) alone: a provider that issues non-expiring
        tokens and no refresh token would otherwise be forced through a full
        re-authorization on every launch.

        Legacy rows written before :meth:`set_tokens` recorded the issue time
        fall back to the ``updated_at`` this instance snapshotted when it opened
        (milliseconds). That is an UPPER BOUND on the issue time, not the issue
        time: the store stamps the column on every write to the row, and
        client-info writes touch the same row without touching the tokens. So
        the fallback can read a genuinely expired token as live — never the
        reverse — and the cost of being wrong is the one browser grant that used
        to happen every launch. It is self-healing: that grant writes
        ``tokens_obtained_at``, and the row never takes the fallback again.

        Using the snapshot rather than a fresh read is what keeps THIS process's
        own ``seed_client_info`` from resetting the bound to "now" before we can
        read it, which would make the migration a guaranteed no-op for exactly
        the pinned-client servers that seed exists to serve.
        """
        row = self._read_row()
        if row is None:
            return None
        data = row.data if isinstance(row.data, dict) else {}
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            return None
        expires_in = tokens.get("expires_in")
        if not isinstance(expires_in, (int, float)) or expires_in <= 0:
            return None
        obtained_at = data.get(TOKENS_OBTAINED_AT_KEY)
        if not isinstance(obtained_at, (int, float)) or obtained_at <= 0:
            if self._row_updated_at_at_open <= 0:
                return None
            obtained_at = self._row_updated_at_at_open / 1000.0
        return float(obtained_at) + float(expires_in)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Stored client registration (DCR result or pinned config), or ``None``."""
        creds = self._read()
        info = creds.get("client_info") if creds is not None else None
        if not isinstance(info, dict):
            return None
        try:
            from mcp.shared.auth import OAuthClientInformationFull

            return OAuthClientInformationFull.model_validate(info)
        except Exception:
            logger.debug("Stored MCP client info invalid for %s", self.credential_id, exc_info=True)
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Persist a dynamic-client registration (RFC 7591)."""
        creds = self._read() or {}
        creds["client_info"] = client_info.model_dump(mode="json")
        self._write(creds)

    def seed_client_info(self, client_id: str, client_secret: str | None = None) -> None:
        """Synchronously pre-seed a pinned client registration (MCP-11).

        Same persistence path as :meth:`set_client_info` but callable from
        sync wiring code: when the config supplies a ``client_id`` the SDK
        finds it via ``get_client_info`` and skips dynamic client
        registration entirely — required for providers whose redirect URI was
        registered against a fixed loopback port (pinned-redirect providers).
        """
        from mcp.shared.auth import OAuthClientInformationFull

        info = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
        )
        creds = self._read() or {}
        creds["client_info"] = info.model_dump(mode="json")
        self._write(creds)


def parse_oauth_callback_input(raw: str) -> tuple[str, str | None, str | None]:
    """Parse the pasted callback input into ``(code, state, iss)`` (MCP-02).

    Accepts either the FULL redirect URL (``...?code=X&state=Y&iss=Z``) or a
    bare ``code state`` pair separated by whitespace. ``state`` (and ``iss``
    when present) MUST be handed back to the SDK: it validates ``state``
    against the value it generated (oauth2.py:421) and rejects the flow when
    the handler returns ``state=None``.
    """
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("No authorization input provided")
    if "://" in text or text.startswith("http"):
        query = parse_qs(urlparse(text).query)
        code = (query.get("code") or [""])[0]
        state = (query.get("state") or [None])[0]
        iss = (query.get("iss") or [None])[0]
        if not code:
            raise RuntimeError(f"No authorization code found in redirect URL: {text!r}")
        return code, state, iss
    parts = text.split()
    if len(parts) == 1:
        raise RuntimeError("Bare input needs 'code state' (paste the full redirect URL instead)")
    code, state = parts[0], parts[1]
    iss = parts[2] if len(parts) > 2 else None
    return code, state, iss


#: How long the whole grant may sit waiting for the human: the browser round
#: trip and, where it is offered, the paste. A connect that reaches this path
#: on an unattended host must fail eventually, not park the connect task
#: forever.
PASTE_INPUT_TIMEOUT_S = 300.0

#: Hosts a redirect URI can name that THIS process is able to answer.
#:
#: A redirect URI may legitimately point anywhere; only a loopback address is
#: something we can bind. Anything else — a hosted callback, a tunnel — has to
#: fall through to the paste path, because listening would not intercept it.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Bound on the request head we will read from the browser before giving up.
#: A callback is a GET with a short query string; anything larger is not the
#: browser we asked for, and an unbounded read on a public-ish port is a
#: memory-exhaustion invitation.
_MAX_REQUEST_HEAD_BYTES = 16 * 1024
_MAX_REQUEST_HEADERS = 64

#: How long one connection may take to send its request head.
#: A browser sends it in one packet; anything slower is a probe holding a
#: handler open, and a held handler is what makes ``wait_closed()`` hang.
_REQUEST_READ_TIMEOUT_S = 10.0

#: How long teardown may wait for in-flight handlers before abandoning them.
#: The authorization code is already in hand at that point; a lingering socket
#: is not worth stalling the connect for.
_SERVER_CLOSE_TIMEOUT_S = 1.0


class LoopbackAuthFlow:
    """The redirect/callback pair for one authorization, over a real listener.

    The SDK drives an authorization in two steps: it hands us the URL to open
    (:meth:`redirect_handler`), then blocks on us to produce the code the
    provider redirected back with (:meth:`callback_handler`). We advertise
    ``http://127.0.0.1:<port>/callback`` as the redirect URI, so the ONLY way
    that promise is kept is by listening on it — without a listener the user
    signs in, the provider redirects, and the browser lands on
    ``ERR_CONNECTION_REFUSED`` with the code stranded in the address bar.

    The listener is opened in :meth:`redirect_handler`, BEFORE the browser is
    launched, because the provider can redirect the instant the user's session
    is already authorized — binding afterwards is a race the fast path loses.

    Paste is a strict FALLBACK, never a race: it is offered only when there is
    no listener to wait on. A thread parked in ``input()`` cannot be cancelled,
    so racing it would leave a reader on the tty and a thread that
    ``asyncio.run`` joins at shutdown — the browser path would succeed and the
    process would hang anyway. Paste is additionally gated on the terminal
    being ours: under the TUI it belongs to Textual's input driver, and a
    second reader on the same file descriptor does not queue behind it — the
    two split the user's keystrokes, which reads as an app randomly ignoring
    what is typed. :func:`~local_operator.logger.console_is_silenced` is the
    signal for that, and it also decides whether our progress notices go to
    stderr or to the log file.
    """

    def __init__(self, redirect_uri: str, server_url: str | None = None) -> None:
        parsed = urlparse(redirect_uri)
        self.redirect_uri = redirect_uri
        #: Named on the callback page. Someone with several MCP servers
        #: configured has no other way to tell which tab belongs to which
        #: authorization, and "Authorized" without a subject is a page that
        #: could be about anything.
        self.server_url = server_url
        self._host = parsed.hostname or ""
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._path = parsed.path or "/"
        self._servable = parsed.scheme == "http" and self._host in LOOPBACK_HOSTS
        self._server: asyncio.AbstractServer | None = None
        self._result: asyncio.Future[tuple[str, str | None, str | None]] | None = None
        #: Why the bind failed, when it did. "Could not listen" and "could not
        #: have listened" need different advice, so the two are kept apart.
        self._bind_error: str | None = None

    # --- notices ---------------------------------------------------------

    def _notify(self, *lines: str) -> None:
        """Tell the user what is happening, without painting over a frame."""
        from local_operator.logger import console_is_silenced

        if console_is_silenced():
            for line in lines:
                logger.info("%s", line.strip())
            return
        for line in lines:
            print(line, file=sys.stderr)

    def _paste_allowed(self) -> bool:
        """Whether stdin is ours to read (see the class docstring)."""
        from local_operator.logger import console_is_silenced

        return sys.stdin.isatty() and not console_is_silenced()

    # --- SDK handlers ----------------------------------------------------

    async def redirect_handler(self, authorization_url: str) -> None:
        """Start listening, then send the user to the authorization URL.

        The URL is hard-wrapped in brackets so trailing OAuth params can never
        be silently lost on copy (a real production paste bug).
        """
        await self._start_server()
        lines = [
            "\nMCP OAuth authorization required. Open this URL in a browser:",
            f"  <{authorization_url}>",
        ]
        try:
            import webbrowser

            if webbrowser.open(authorization_url):
                lines.append("(opened in your default browser)")
        except Exception:
            pass  # headless: the listener or the paste fallback carries it
        if self._server is not None:
            lines.append(f"Waiting for the redirect to {self.redirect_uri} …")
        self._notify(*lines)

    async def callback_handler(self) -> AuthorizationCodeResult:
        """Wait for the provider's redirect (or a pasted URL) and return the code."""
        from mcp.shared.auth import AuthorizationCodeResult

        try:
            code, state, iss = await self._await_authorization()
        finally:
            await self._stop_server()
        return AuthorizationCodeResult(code=code, state=state, iss=iss)

    async def _await_authorization(self) -> tuple[str, str | None, str | None]:
        if self._result is None:
            # No listener: paste is the only route left, and reading stdin is
            # safe precisely because nothing else is going to.
            return await self._await_pasted()
        try:
            return await asyncio.wait_for(self._result, timeout=PASTE_INPUT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Timed out after {PASTE_INPUT_TIMEOUT_S:.0f}s waiting for the OAuth "
                f"redirect to {self.redirect_uri}. If you authorized in a browser on "
                "another machine it cannot reach this port — forward it "
                f"(ssh -L {self._port}:127.0.0.1:{self._port} …) and try again."
            ) from exc

    async def _await_pasted(self) -> tuple[str, str | None, str | None]:
        """Read the redirect URL from stdin. NEVER raced against the listener.

        A thread parked in ``input()`` cannot be cancelled: ``asyncio.to_thread``
        hands the call to the default executor, whose future refuses
        cancellation once running, and the thread then sits on stdin until a
        newline that — on the happy path, where the browser finished the grant —
        never comes. Two things break as a result: ``asyncio.run`` never returns
        (``Runner.close`` joins the default executor), so ``local-operator mcp
        login`` hangs AFTER succeeding; and the parked reader is a second
        consumer on the tty, which is the same keystroke-splitting bug this
        class exists to avoid, moved outside the TUI.

        So paste is a genuine FALLBACK, reached only when there is no listener
        to wait on, and the thread it starts is one the flow is committed to.
        """
        if not self._paste_allowed():
            raise RuntimeError(f"MCP OAuth cannot complete here: {self._no_route_reason()}")
        prompt = "Paste the full redirect URL (or 'code state' separated by a space): "
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(lambda: input(prompt).strip()), timeout=PASTE_INPUT_TIMEOUT_S
            )
        except asyncio.TimeoutError as exc:
            # TRANSLATE IT. Since 3.11 `asyncio.TimeoutError` IS `TimeoutError`
            # and `str(TimeoutError())` is the empty string, so letting it
            # propagate makes the CLI print "MCP login failed for 'x': " with no
            # reason at all — on exactly the paths (unservable redirect URI, a
            # bind lost to a squatter) that `_no_route_reason` exists to explain.
            raise RuntimeError(
                f"Timed out after {PASTE_INPUT_TIMEOUT_S:.0f}s waiting for the pasted "
                f"redirect URL. Reading from stdin because {self._no_listener_reason()}."
            ) from exc
        return parse_oauth_callback_input(raw)

    def _no_listener_reason(self) -> str:
        """Why the browser redirect is not being captured, in the user's terms.

        A bind failure and an unservable redirect URI are different problems
        with different fixes, and conflating them sends someone to audit their
        OAuth configuration when the real answer is that a dev server is
        squatting the port. Phrased as a clause so both callers — "no route at
        all" and "falling back to a paste" — can finish the sentence their own
        way.
        """
        if self._bind_error is not None:
            return (
                f"nothing could listen on {self.redirect_uri} ({self._bind_error}) "
                "— free that port, or set a different `oauth.callback_port` for "
                "this server"
            )
        if not self._servable:
            return (
                f"the redirect URI {self.redirect_uri} is not a loopback address "
                "this process can serve"
            )
        return "the callback listener is not running"

    def _no_route_reason(self) -> str:
        """Why NEITHER route is available: no listener, and no stdin either.

        The listener clause is ended with a full stop rather than spliced in on
        a comma: the bind branch carries an em-dash aside, and coordinating
        "and stdin is not available" onto it reads as a third item in that
        clause's remedy list rather than as a second problem.
        """
        return (
            f"{self._no_listener_reason()}. Stdin is not available for a paste "
            "either. Run `local-operator mcp login <server>` from a terminal, "
            "or configure the server with a token."
        )

    # --- the listener ----------------------------------------------------

    async def _start_server(self) -> None:
        """Bind the redirect URI, or leave ``_server`` None and record why."""
        if self._server is not None or not self._servable:
            return
        loop = asyncio.get_running_loop()
        self._result = loop.create_future()
        try:
            self._server = await asyncio.start_server(self._serve, self._host, self._port)
        except OSError as exc:
            # Almost always "address already in use": another local-operator, or
            # a dev server squatting :3000. Not fatal — the paste path still
            # completes the grant — but the user has to be told, because from
            # the browser's side this looks like the login simply not working.
            self._result = None
            self._bind_error = str(exc)
            self._notify(
                f"Could not listen on {self.redirect_uri} ({exc}); "
                "the browser redirect will not be captured automatically."
            )

    async def _stop_server(self) -> None:
        server, self._server = self._server, None
        self._result = None
        if server is None:
            return
        server.close()
        # BOUNDED. Since 3.12.1 ``wait_closed()`` also waits for every accepted
        # connection's handler to finish, so one peer that opened a socket and
        # sent nothing — a browser preconnect, a security scanner — would park
        # this forever. It runs in ``callback_handler``'s ``finally``, so an
        # unbounded wait here would swallow the authorization we already have
        # and turn every timeout into a hang.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.wait_closed(), timeout=_SERVER_CLOSE_TIMEOUT_S)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Answer one browser request; resolve the flow when it is the callback."""
        try:
            target = await self._read_request_target(reader)
            if target is None:
                writer.write(
                    callback_response(
                        "Bad request",
                        "That was not a request this page knows how to answer.",
                        closable=False,
                        status="400 Bad Request",
                    )
                )
                return
            parsed = urlparse(target)
            if parsed.path != self._path:
                # Browsers ask for /favicon.ico off their own bat; answering a
                # real 404 rather than resolving the flow keeps those from
                # being mistaken for the redirect.
                writer.write(
                    callback_response(
                        "Nothing here",
                        "This address only answers the authorization redirect.",
                        closable=False,
                        status="404 Not Found",
                    )
                )
                return
            query = parse_qs(parsed.query)
            error = (query.get("error") or [""])[0]
            if error:
                # The provider's own words go in their own labelled trough, not
                # spliced into a sentence spoken in our voice. `error_description`
                # is arbitrary text from a query string rendered inside a card
                # carrying our mark; escaping stops it being an injection, but
                # only a visible seam stops a hostile provider borrowing our
                # voice. It is also where a bare `access_denied` reads correctly
                # rather than being presented as English.
                detail = (query.get("error_description") or [""])[0] or error
                writer.write(
                    callback_response(
                        "Authorization failed",
                        "The provider did not grant this authorization, so nothing "
                        "was connected. You can start the connection again from "
                        "Local Operator.",
                        tone="danger",
                        server=self.server_url,
                        provider_message=detail,
                    )
                )
                self._settle_error(RuntimeError(f"OAuth authorization failed: {detail}"))
                return
            code = (query.get("code") or [""])[0]
            if not code:
                writer.write(
                    callback_response(
                        "No authorization code",
                        "The redirect arrived without an authorization code, so "
                        "there is nothing to hand back. You can start the "
                        "connection again from Local Operator.",
                        tone="danger",
                        server=self.server_url,
                    )
                )
                # SETTLE, do not just report. Without this the page says the tab
                # can be closed while the flow sits waiting out its full timeout
                # on a redirect that can never carry a code — the one call site
                # that made `closable` a lie.
                self._settle_error(RuntimeError("OAuth redirect carried no authorization code"))
                return
            writer.write(
                callback_response(
                    "Authorized",
                    "Local Operator has the authorization code and is finishing the connection.",
                    tone="success",
                    server=self.server_url,
                )
            )
            self._settle(
                code,
                (query.get("state") or [None])[0],
                (query.get("iss") or [None])[0],
            )
        except Exception:  # noqa: BLE001 — one bad request must not kill the flow
            logger.debug("MCP OAuth callback request failed", exc_info=True)
        finally:
            # The last unbounded awaits in a handler whose every other wait is
            # explicit. A peer that sends a valid GET and then stops reading
            # blocks `drain()` for as long as it likes, holding a task and an fd
            # for the life of the PROCESS — which under the TUI is the whole
            # session, not the flow. Today the page fits in a default send
            # buffer on macOS, but that is an accident of one platform's
            # defaults and a page size capped three functions away.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.drain(), timeout=_SERVER_CLOSE_TIMEOUT_S)
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), timeout=_SERVER_CLOSE_TIMEOUT_S)

    async def _read_request_target(self, reader: asyncio.StreamReader) -> str | None:
        """The request target of a GET, with the head read and discarded.

        Bounded in BOTH directions — bytes and time. A peer that connects and
        says nothing must not hold a handler open (see :meth:`_stop_server`),
        and a peer that talks forever must not be allowed to.
        """
        try:
            return await asyncio.wait_for(self._read_head(reader), timeout=_REQUEST_READ_TIMEOUT_S)
        except asyncio.TimeoutError:
            return None

    @staticmethod
    async def _read_head(reader: asyncio.StreamReader) -> str | None:
        line = await reader.readline()
        budget = _MAX_REQUEST_HEAD_BYTES - len(line)
        if not line or budget < 0:
            return None
        parts = line.decode("latin-1").split()
        if len(parts) < 2 or parts[0].upper() != "GET":
            return None
        # Drain the headers so the browser sees a well-formed exchange rather
        # than a reset mid-request (which some render as a failed navigation).
        # The byte budget spans the WHOLE head: bounding each `readline` alone
        # leaves 64 headers x StreamReader's own 64 KiB limit, which is 4 MB.
        for _ in range(_MAX_REQUEST_HEADERS):
            header = await reader.readline()
            if header in (b"\r\n", b"\n", b""):
                break
            budget -= len(header)
            if budget < 0:
                return None
        return parts[1]

    def _settle(self, code: str, state: str | None, iss: str | None) -> None:
        if self._result is not None and not self._result.done():
            self._result.set_result((code, state, iss))

    def _settle_error(self, exc: Exception) -> None:
        if self._result is not None and not self._result.done():
            self._result.set_exception(exc)


def wire_oauth_auth(
    server_url: str, cfg: MCPServerConfig, store: StructuralAuthStore | None = None
) -> dict[str, Any]:
    """Build ``OAuthClientProvider`` kwargs for one server.

    ``cfg`` is the server's :class:`~local_operator.mcp.config.MCPServerConfig`
    (its ``auth`` / ``oauth`` blocks supply client identity and callback
    knobs). Returns a dict suitable for ``OAuthClientProvider(**kwargs)``:

    - ``server_url``: the MCP server URL (resource indicator base);
    - ``client_metadata``: PKCE authorization-code client, redirect URI
      ``http://127.0.0.1:{callback_port or 3000}{callback_path or /callback}``
      (PKCE itself is automatic inside the SDK);
    - ``storage``: a :class:`McpTokenStorage` bound to ``store``; a config
      ``client_id`` pre-seeds the client registration so DCR is skipped
      (MCP-11);
    - ``redirect_handler`` / ``callback_handler``: the two halves of one
      :class:`LoopbackAuthFlow`, which listens on that redirect URI for the
      duration of the grant (see the module docstring).

    The returned dict is constructed eagerly but imports ``mcp`` lazily inside
    so config-only code paths never touch the SDK.
    """
    from mcp.shared.auth import OAuthClientMetadata

    auth = cfg.auth
    oauth = cfg.oauth

    callback_port = (oauth.callback_port if oauth is not None else None) or DEFAULT_CALLBACK_PORT
    callback_path = (oauth.callback_path if oauth is not None else None) or DEFAULT_CALLBACK_PATH
    if not callback_path.startswith("/"):
        callback_path = f"/{callback_path}"
    redirect_uri = (oauth.redirect_uri if oauth is not None else None) or (
        f"http://127.0.0.1:{callback_port}{callback_path}"
    )

    # Scopes: explicit `scope` on the auth block — an extra-allowed field, so
    # it lives in ``model_extra`` rather than being declared — else none (the
    # server advertises them via protected-resource metadata).
    scope: str | None = (auth.model_extra or {}).get("scope") if auth is not None else None

    client_secret = (auth.client_secret if auth is not None else None) or (
        oauth.client_secret if oauth is not None else None
    )

    client_metadata = OAuthClientMetadata(
        client_name="local-operator",
        redirect_uris=[AnyUrl(redirect_uri)],
        scope=scope,
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post" if client_secret else "none",
    )

    storage = McpTokenStorage(server_url, store)

    # A configured client_id is pinned: pre-seed it so the SDK skips dynamic
    # client registration (MCP-11). DCR would mint a fresh client whose
    # redirect URI need not match what the provider registered, which breaks
    # pinned-redirect providers outright.
    client_id = (auth.client_id if auth is not None else None) or (
        oauth.client_id if oauth is not None else None
    )
    if client_id:
        storage.seed_client_info(client_id, client_secret)

    flow = LoopbackAuthFlow(redirect_uri, server_url=server_url)
    return {
        "server_url": server_url,
        "client_metadata": client_metadata,
        "storage": storage,
        "redirect_handler": flow.redirect_handler,
        "callback_handler": flow.callback_handler,
    }


def build_oauth_provider(
    server_url: str, cfg: MCPServerConfig, store: StructuralAuthStore | None = None
) -> Any:
    """An ``OAuthClientProvider`` that knows when its stored token expires.

    The SDK reloads tokens on first use but NOT their deadline
    (``OAuthClientProvider._initialize`` sets ``current_tokens`` and leaves
    ``token_expiry_time`` at ``None``), and ``OAuthContext.is_token_valid``
    reads a missing deadline as "still good". So every fresh process presented
    a day-old access token, got a 401, and ran the FULL browser authorization
    — the refresh token sitting in the same row was never spent, because the
    refresh branch is only reached when the token is known to be expired.

    Priming the deadline from what we persisted is the whole fix: an expired
    token now takes the refresh grant, silently, with no browser. It is set
    after construction rather than passed in because the SDK offers no
    constructor argument for it, and ``_initialize`` does not clear it.
    """
    from mcp.client.auth import OAuthClientProvider

    kwargs = wire_oauth_auth(server_url, cfg, store=store)
    provider = OAuthClientProvider(**kwargs)
    storage = kwargs["storage"]
    try:
        expiry = storage.stored_token_expiry()
    except Exception:  # noqa: BLE001 — a metadata read must not block a connect
        logger.debug("MCP token expiry unreadable for %s", server_url, exc_info=True)
        return provider
    if expiry is not None:
        provider.context.token_expiry_time = expiry
    return provider

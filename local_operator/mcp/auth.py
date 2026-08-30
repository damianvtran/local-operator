"""MCP OAuth support on the official SDK's ``OAuthClientProvider``.

Flow (official SDK PKCE + RFC 7591 DCR under the hood):

- ``build_oauth_provider(server_url, cfg)`` is the entry point: it wires the
  provider AND primes it with the stored token's expiry, which is what makes a
  restart spend the refresh token instead of re-running a browser grant.
- ``ensure_mcp_oauth_fresh(server_url, cfg)`` refreshes an expired access
  token BEFORE connecting, against the token endpoint resolved from the
  server's OAuth metadata (PRM + ASM discovery). This is what stops a day-old
  token from forcing a browser grant on startup for providers whose token
  endpoint is not ``<server_base>/token`` (the SDK's fallback guess 404s for
  e.g. Datadog). The refresh is serialized across processes with a file lock
  and the token is re-read under it, so concurrently starting sessions cannot
  spend a rotating refresh token twice.
- ``wire_oauth_auth(server_url, cfg)`` returns the ``OAuthClientProvider``
  kwargs: client metadata with a loopback redirect URI, a token storage bound
  to the shared credential store, and a :class:`LoopbackAuthFlow` that
  actually LISTENS on that redirect URI (with a pasted-URL race for browsers
  that cannot reach this machine).
- ``McpTokenStorage`` is the SDK ``TokenStorage``: one row per server URL in
  the real ``providers.auth_store.AuthStore``, keyed ``mcp_oauth:<url>``, with
  the token's issue time recorded so its lifetime survives the process.

Non-interactive connects (ordinary startup and auto-reconnect) pass
``interactive=False``: when the stored grant cannot be refreshed the flow
raises :class:`McpAuthRequiredError` instead of opening a browser, and the
manager surfaces that as an actionable "run /mcp login <name>" failure. Only
an explicit login (``/mcp login`` / ``local-operator mcp login``) runs
interactive and may open a browser.

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
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from pydantic import AnyUrl

from local_operator.ansi import strip_control_sequences
from local_operator.callback_page import callback_response

if TYPE_CHECKING:
    # The SDK is an optional extra: these names are needed for annotations
    # only, so importing them here keeps this module importable without it.
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthMetadata,
        OAuthToken,
        ProtectedResourceMetadata,
    )

    from local_operator.mcp.config import MCPServerConfig
    from local_operator.providers.auth_store import StoredCredential

logger = logging.getLogger(__name__)

# Logical credential id prefix for managed MCP OAuth credentials (URL-keyed).
MCP_OAUTH_CREDENTIAL_PREFIX = "mcp_oauth:"

# Provider column value in the shared auth_credentials table.
MCP_OAUTH_PROVIDER = "mcp-oauth"

#: Loopback port for OAuth callbacks when a server does not pin its own.
#: 33441 is deliberately rare (sibling of Codex's 33418): :3000 collides with
#: local dev servers often enough that the listener bind routinely failed and
#: the grant fell back to the manual paste flow. Servers that registered a
#: redirect URI against the old default must pin ``callback_port: 3000`` in
#: their config oauth block to keep working.
DEFAULT_CALLBACK_PORT = 33441
DEFAULT_CALLBACK_PATH = "/callback"

#: Payload key holding the wall-clock time (epoch seconds) the stored access
#: token was issued. Not part of the SDK's ``OAuthToken`` — see
#: :meth:`McpTokenStorage.stored_token_expiry` for why we have to record it.
TOKENS_OBTAINED_AT_KEY = "tokens_obtained_at"

#: Refresh this far BEFORE the stored access token's deadline. A connect that
#: starts with a token dying in ten seconds would otherwise open with a 401 and
#: lean on the in-flow refresh at the worst possible moment; spending the
#: refresh grant proactively keeps the first request authenticated.
REFRESH_SKEW_S = 60.0

#: Bound on one proactive refresh's HTTP round trips (metadata discovery plus
#: the token POST). A slow authorization server must not park the connect —
#: the startup gate defers us, and the breaker bounds retries.
REFRESH_HTTP_TIMEOUT_S = 10.0

#: Bound on ACQUIRING the cross-process refresh lock, derived from the budget of
#: the critical section it guards: one token POST plus a couple of SQLite reads.
#: That derivation is only sound because the POST is capped in TOTAL wall time
#: (see the ``asyncio.timeout`` in :func:`_refresh_oauth_token_locked`) —
#: ``httpx.Timeout(REFRESH_HTTP_TIMEOUT_S)`` alone does NOT bound a request:
#: it is per operation, and its read timeout is per socket read, so a dribbling
#: server measured 140.7s inside a nominal 10s timeout. Keep the two in step: if
#: the POST's cap is ever raised or removed, this bound stops being honest and a
#: slow-but-working peer starts getting timed out by its own siblings.
#: Overrunning this bound therefore means the holder really is not working — a
#: leaked lock from a killed process, or a peer wedged on something that is not
#: our problem. Waiting longer than the work can possibly take buys nothing and
#: costs a hung connect, so we give up and degrade (see
#: :func:`_oauth_refresh_lock`, which yields False rather than raising).
LOCK_ACQUIRE_TIMEOUT_S = 15.0

#: FIRST gap between non-blocking lock attempts, and the cancellation
#: granularity: a cancelled acquire abandons the retry loop within one sleep,
#: so this also bounds how long an abandoned worker lingers. Small, because the
#: overwhelmingly common contended case is a peer finishing in a moment and
#: pickup latency is what the user feels.
_LOCK_RETRY_SLEEP_S = 0.05

#: Ceiling for the backoff below. A fixed 50 ms gap would cost ~300 wakeups per
#: contended acquire per server, and six OAuth servers connecting together would
#: run six worker threads ticking for up to the full bound against a default
#: executor of 18. Backing off to a quarter second cuts that by most while
#: leaving the fast case untouched, since a lock still free after a few seconds
#: is a leaked one we are going to abandon anyway.
_LOCK_RETRY_SLEEP_MAX_S = 0.25


class McpAuthRequiredError(RuntimeError):
    """An MCP server needs an interactive OAuth grant this run cannot open.

    Raised instead of opening a browser when the connect is NON-interactive
    (ordinary session startup and auto-reconnects): a background connect that
    pops a login tab is an interruption the user never asked for, and several
    sessions starting at once would each pop one. The connect fails with an
    actionable message instead; ``/mcp login <name>`` (or
    ``local-operator mcp login <name>``) runs the same grant deliberately.
    """

    def __init__(self, server_url: str) -> None:
        super().__init__(f"MCP OAuth authorization required for {server_url}")
        self.server_url = server_url


class McpAuthChallengeError(RuntimeError):
    """An HTTP MCP server refused the connect with 401/403.

    Distinct from :class:`McpAuthRequiredError`, which means "we HAVE an OAuth
    config and the grant needs a browser". This one means "the transport was
    rejected as unauthorized", and it exists because the SDK erases that fact:
    a 401 the provider cannot resolve surfaces from ``session.initialize()`` as
    a generic ``MCPError(-32603, 'Server returned an error response')`` (or
    ``-32001, 'unauthorized access'``) with **no status code anywhere on the
    exception** — verified against the live GitLab, LaunchDarkly, Datadog and
    Minerva QA endpoints. Those two strings are exactly what the user
    screenshotted, and neither says what to do.

    ``oauth_available`` records whether RFC 9728 / RFC 8414 discovery actually
    found an authorization server for this URL. It drives the WORDING and must
    never be guessed: a server can 401 with no ``WWW-Authenticate`` header at
    all (Datadog does), and promising ``/mcp login`` when we found no endpoint
    would send the user at a command that cannot work.

    ``has_stored_grant`` is what makes ``login`` vs ``reauth`` truthful: a
    server we have never held a credential for needs a first grant, while one
    whose stored grant just got rejected needs it replaced. The manager reads
    the credential store to set this rather than inferring it from the error.
    """

    def __init__(
        self,
        server_url: str,
        *,
        status_code: int,
        oauth_available: bool,
        has_stored_grant: bool,
    ) -> None:
        super().__init__(f"MCP server at {server_url} refused the connection ({status_code})")
        self.server_url = server_url
        self.status_code = status_code
        self.oauth_available = oauth_available
        self.has_stored_grant = has_stored_grant


class McpLoginCancelledError(RuntimeError):
    """An interactive MCP OAuth grant ended with no authorization arriving.

    Two routes land here: the human route (the browser tab was closed or the
    consent screen abandoned, surfaced once the idle guard fires) and the
    structural one (the login task was cancelled — an exclusive re-login, the
    TUI's stop-ladder, or a Ctrl+C at the CLI). The point of the dedicated
    type is the MESSAGE: the login flows catch the SDK's ``OAuthFlowError``
    and report ``str(exc)``, so the explanation has to live on the innermost
    raise or it is lost. A bare ``CancelledError`` would surface either as an
    empty ``MCP login failed for 'x':`` line or — worse inside the TUI — as
    silence, because a Textual worker cancelled by its exclusive group never
    runs the worker's exception handler at all.
    """


class AbandonedGrantError(Exception):
    """The browser round trip ended with no authorization — the human walked away.

    Distinct from :class:`McpLoginCancelledError` on purpose: THAT one is the
    user-facing receipt the login surfaces report; this one is the flow's
    internal record of WHY the grant died. The separation matters because of
    how the two endings have to travel. An abandoned grant is raised out of
    ``callback_handler`` as a raw ``asyncio.CancelledError``: the streamable-HTTP
    transport's SDK ``post_writer`` swallows any ordinary exception an auth
    handler raises (it logs and moves on), while a cancellation unwinds the
    transport's task group exactly the way a grant REQUIREMENT does — that is
    the channel the message cannot be eaten on. The flow records itself in
    :data:`ABANDONED_GRANTS` first, and the manager converts the arriving
    cancellation back into :class:`McpLoginCancelledError`.
    """


def mcp_oauth_credential_id(server_url: str) -> str:
    """Stable logical credential id for one MCP server's OAuth grant."""
    return f"{MCP_OAUTH_CREDENTIAL_PREFIX}{server_url}"


#: The callback port every local-operator shipped with before 33441. Stored
#: client registrations pinned to it are dropped once on sight (see
#: :meth:`McpTokenStorage.get_client_info`) so they re-register / re-seed
#: against the new default instead of dead-ending at the provider with
#: ``redirect_uri_mismatch``.
LEGACY_CALLBACK_PORT = 3000

#: How long to wait for the browser launcher before assuming the page opened.
#: The stdlib's ``GenericBrowser`` WAITS on a foreground browser, so a launcher
#: still running after this is the normal case for one, not a failure.
BROWSER_OPEN_TIMEOUT_S = 5.0

#: What the launcher child runs. ``webbrowser.open``'s own return value is the
#: exit status, so the parent still learns whether a browser was found.
_BROWSER_OPEN_SNIPPET = "import sys, webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1]) else 1)"


async def open_browser_quietly(url: str) -> bool:
    """Open ``url`` in a browser without letting it print over the frame.

    ``webbrowser.open`` spawns the browser with fd 1 and fd 2 INHERITED — the
    stdlib's ``GenericBrowser`` and ``BackgroundBrowser`` pass neither
    ``stdout`` nor ``stderr`` to ``Popen`` (verified in CPython 3.13's
    ``webbrowser``) — so ``xdg-open: no method available`` or a browser's
    ``Gtk-Message:`` chatter lands straight on the Textual frame. Same defect
    as an MCP server's startup banner, arriving through the login flow.

    Redirecting our OWN descriptors is not available as a fix: Textual is
    writing to fd 1 from this very process, so replacing it even briefly
    corrupts the display we are trying to protect. Instead the call is
    delegated to a short-lived Python child whose stdout and stderr are pipes
    this process owns and logs.

    Only while the console is silenced. With the terminal ours — ``local-
    operator mcp login`` — a browser's complaint on stderr is exactly what the
    user should see, and paying for an interpreter start to hide it would be
    backwards.
    """
    from local_operator.logger import console_is_silenced

    if not console_is_silenced():
        import webbrowser

        try:
            return webbrowser.open(url)
        except Exception:  # noqa: BLE001 — headless: the paste fallback carries it
            logger.debug("webbrowser.open failed", exc_info=True)
            return False

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _BROWSER_OPEN_SNIPPET,
            url,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            # Merged: the two streams are one diagnostic here, and a single
            # pipe cannot deadlock against itself the way two unread ones can.
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001 — no browser is a degraded login, not a crash
        logger.debug("browser launcher failed to start", exc_info=True)
        return False

    async def _drain(prefix: str) -> str:
        assert process.stdout is not None
        raw = await process.stdout.read()
        text = strip_control_sequences(raw.decode("utf-8", "replace")).strip()
        if text:
            logger.info("%s%s", prefix, text)
        return text

    drain = asyncio.ensure_future(_drain("browser launcher: "))
    try:
        await asyncio.wait_for(asyncio.shield(drain), timeout=BROWSER_OPEN_TIMEOUT_S)
    except asyncio.TimeoutError:
        # A foreground browser keeps the launcher alive for as long as its
        # window is open. Let the drain task run on so the pipe never fills and
        # blocks that browser; the page IS open, which is what the caller asks.
        return True
    await process.wait()
    return process.returncode == 0


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

    def delete_credential(self, credential_id: int) -> None:
        """Remove one row entirely (``/mcp logout`` / ``mcp logout``)."""
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
        logger.debug(
            "providers.auth_store unavailable; MCP OAuth storage disabled",
            exc_info=True,
        )
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

    def clear(self) -> bool:
        """Delete this server's credential row entirely (logout). Returns
        ``True`` when a row existed and was removed.

        The row carries BOTH the OAuth grant and any client registration
        (``seed_client_info`` pins it via ``project_id``, DCR writes it via
        ``set_client_info``), so removal is what makes the next login run a
        genuinely fresh grant — new consent, new registration — instead of
        silently reusing the stored client info. A pinned ``client_id`` from
        config is re-seeded by ``wire_oauth_auth`` on the next connect, so
        losing the row never strands a pinned-client server.
        """
        store = self._store
        if store is None:
            return False
        row = self._read_row()
        if row is None:
            return False
        try:
            store.delete_credential(row.id)
        except Exception:
            logger.debug("MCP credential delete failed for %s", self.credential_id, exc_info=True)
            return False
        return True

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
        """Stored client registration (DCR result or pinned config), or ``None``.

        A stored registration whose redirect URIs still point at the legacy
        :data:`LEGACY_CALLBACK_PORT` is stale by definition — the runtime now
        advertises a different loopback port, so that registration can never
        complete a grant (the provider rejects the authorization redirect with
        ``redirect_uri_mismatch``, and because ``client_info`` is present the
        SDK never re-runs DCR — a dead-end with no in-app recovery). Dropping
        it here, before the SDK reads it, lets the flow re-register (DCR
        servers) or re-seed (pinned ``client_id`` servers, which
        :meth:`seed_client_info` rewrites on the next login) against the new
        redirect URI.
        """
        creds = self._read()
        info = creds.get("client_info") if creds is not None else None
        if not isinstance(info, dict):
            return None
        if self._redirect_uris_use_legacy_port(info):
            logger.info(
                "Discarding MCP client registration for %s: its redirect URIs "
                "still target the legacy :%d callback, which can no longer "
                "complete a grant; it will re-register on this login.",
                self.credential_id,
                LEGACY_CALLBACK_PORT,
            )
            if creds is not None:
                creds.pop("client_info", None)
                self._write(creds)
            return None
        try:
            from mcp.shared.auth import OAuthClientInformationFull

            return OAuthClientInformationFull.model_validate(info)
        except Exception:
            logger.debug(
                "Stored MCP client info invalid for %s",
                self.credential_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _redirect_uris_use_legacy_port(info: dict[str, Any]) -> bool:
        """True when any stored redirect URI targets the legacy callback port."""
        uris = info.get("redirect_uris") or []
        if not isinstance(uris, list):
            return False
        for uri in uris:
            if not isinstance(uri, str):
                continue
            try:
                port = urlparse(uri).port
            except ValueError:
                continue
            if port == LEGACY_CALLBACK_PORT:
                return True
        return False

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

        ``token_endpoint_auth_method`` must be stamped here, not left at its
        ``None`` default: the SDK's ``prepare_token_auth`` only sends the
        ``client_secret`` when the method names a secret-based scheme, so a
        seed that omits it reaches the token endpoint with no secret at all
        and the provider rejects the exchange (HubSpot: ``BAD_CLIENT_SECRET``).
        Because :func:`wire_oauth_auth` re-seeds on EVERY login, a value
        hand-patched into the store is overwritten before it can be used —
        the method has to be correct at the source. ``client_secret_post``
        matches the providers that pin a client (and HubSpot's advertised
        ``token_endpoint_auth_methods_supported``); with no secret the method
        is ``none``.
        """
        from mcp.shared.auth import OAuthClientInformationFull

        info = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method="client_secret_post" if client_secret else "none",
        )
        creds = self._read() or {}
        creds["client_info"] = info.model_dump(mode="json")
        self._write(creds)


#: Server URLs OBSERVED to answer an MCP request with 401/403 during this
#: process's lifetime, mapped to whether OAuth metadata discovery then found an
#: authorization server for them.
#:
#: Why this exists: a config imported from a foreign tool (Codex's
#: ``config.toml``, issue #367) carries only a ``url`` and no ``auth`` block,
#: because that tool holds its OAuth grants elsewhere. Nothing in the static
#: config says the server needs OAuth, so the auth-capable paths
#: (``_build_oauth_auth``, ``_ensure_oauth_fresh``, ``/mcp login``) all declined
#: it and the connect went out unauthenticated. The server's own 401 challenge
#: is the authoritative signal, so we record it when we see it and let the
#: gates consult it. This is deliberately TRANSPORT-level and names no config
#: source: any config that omits an auth block benefits identically.
#:
#: Per-process and observation-only by design. It is not a cache that has to be
#: invalidated: the durable cross-process signal that a server uses OAuth is a
#: stored credential row (see :func:`server_has_stored_grant`), which is what
#: makes a RESTART re-authenticate rather than re-observe a 401 first.
OAUTH_CHALLENGES: dict[str, bool] = {}


def record_oauth_challenge(server_url: str, *, oauth_available: bool) -> None:
    """Remember that ``server_url`` answered with an authorization challenge.

    ``oauth_available`` is whether discovery found an authorization server.
    A True observation is never downgraded by a later False: discovery is a
    network call that can fail transiently, and forgetting that a server is
    OAuth-capable would put the user back on the dead-end message.
    """
    if oauth_available or server_url not in OAUTH_CHALLENGES:
        OAUTH_CHALLENGES[server_url] = oauth_available


def server_has_stored_grant(server_url: str, store: StructuralAuthStore | None = None) -> bool:
    """True when an OAuth credential row already exists for ``server_url``.

    This is the honest basis for choosing between ``/mcp login`` (we have
    never held a grant here) and ``/mcp reauth`` (we hold one and the server
    just rejected it), and it is also the DURABLE signal that a server without
    an explicit ``auth`` block authenticates over OAuth — the observed-challenge
    ledger above is per-process, so without this a restart would connect
    unauthenticated again and ignore a perfectly good stored token.

    A row is NOT enough: the same row also carries ``client_info``, the dynamic
    client registration the SDK writes when it merely DISCOVERS a server, well
    before any user authorizes. Counting that as a grant made a server nobody
    had ever logged into report "authorization expired — run /mcp reauth",
    sending the user to replace a credential that was never issued. Only an
    actual token payload counts.

    Never raises: an unreadable store degrades to "no stored grant", which
    costs a wording nuance rather than a connect.
    """
    try:
        payload = McpTokenStorage(server_url, store)._read()
    except Exception:  # noqa: BLE001 — the store is best-effort here
        logger.debug("stored-grant lookup failed for %s", server_url, exc_info=True)
        return False
    if not payload:
        return False
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return False
    return bool(tokens.get("access_token") or tokens.get("refresh_token"))


def server_is_oauth_capable(
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
    *,
    deliberate: bool = False,
) -> bool:
    """Whether this server should be connected through the OAuth provider.

    Three ways to qualify, in order of authority:

    1. an explicit ``auth.type == "oauth"`` block — the local-operator format's
       own signal, and the only one that existed before;
    2. a stored OAuth credential for its URL — durable proof across restarts
       that this server authenticates with a grant we already hold;
    3. an observed 401/403 challenge whose discovery found an authorization
       server — the live signal that rescues a config which simply does not
       carry an auth block.

    With ``deliberate=False`` (startup, auto-reconnect) a server matching none
    of the three stays unauthenticated, with no discovery and no added latency:
    that is what keeps a genuinely public MCP server
    (``developers.openai.com/mcp``) connecting exactly as it does today, and
    what stops us attaching an OAuth provider to every remote server on boot.

    ``deliberate=True`` is for an explicit ``/mcp login`` and accepts ANY remote
    server. Whether a server needs OAuth cannot be answered without a network
    round trip, so the strict test would otherwise refuse the very first login
    on a Codex-imported server — the dead end this change exists to remove.
    Being permissive here is safe rather than merely convenient: the SDK's
    provider only starts a grant in response to a 401, so attaching it to a
    server that needs no auth changes nothing and opens no browser. A stdio
    server is still refused, having no transport that can carry a bearer token.
    """
    auth = getattr(cfg, "auth", None)
    if auth is not None and getattr(auth, "type", None) == "oauth":
        return True
    url = getattr(cfg, "url", None)
    if not url:
        return False  # stdio: no transport-level challenge to observe
    if deliberate:
        return True
    if OAUTH_CHALLENGES.get(url):
        return True
    return server_has_stored_grant(url, store)


def oauth_server_names(cwd: str | os.PathLike[str]) -> list[str]:
    """Names of configured OAuth-enabled servers, in config order.

    The ``/mcp login|reauth|logout`` argument lists are filled from this, so
    they offer exactly the servers those commands can act on — a stdio server
    has no OAuth grant to log into or out of, and offering it would be a row
    whose only outcome is a warning notice.

    A server with no explicit ``auth`` block is offered once there is EVIDENCE
    it authenticates — a stored grant, or an observed OAuth challenge — which
    is how a foreign-tool import (Codex, issue #367) reaches this list: its
    connect fails first and records the challenge, so the picker then offers
    exactly the server the user just watched fail.

    Deliberately the STRICT test, unlike the gate that executes a login: a
    suggestion list is a claim that these servers have something to log into,
    and speculatively listing every remote server would fill the picker with
    rows whose only outcome is a warning notice. Typing an unlisted name still
    works (see the TUI's ``_resolve_mcp_server``), so nothing is unreachable.
    """
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _sources = load_all_mcp_configs(cwd)
    return [name for name, cfg in configs.items() if server_is_oauth_capable(cfg)]


def mcp_logout_server(
    name: str,
    cwd: str | os.PathLike[str],
    store: StructuralAuthStore | None = None,
) -> str | None:
    """Remove the stored OAuth credential for one configured server.

    Returns an error string on failure, ``None`` on success — the two callers
    (CLI and TUI) phrase their own output, so the helper reports outcomes,
    not prose. All three failure shapes — unknown name, non-OAuth config,
    nothing stored — are reported as errors, but they are DIFFERENT errors:
    a name the config does not know is a typo the user wants told about,
    while a known OAuth server holding no credential is a no-op worth
    distinguishing from a successful removal (the caller's message says
    which).

    The deletion goes through the REAL store (``_resolve_store(None)``), not
    the session manager's possibly-injected one: logout must remove the
    persisted row every future process will read, which is the shared
    ``auth.db`` regardless of what one session was handed.
    """
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _sources = load_all_mcp_configs(cwd)
    cfg = configs.get(name)
    if cfg is None:
        return f"MCP server {name!r} is not configured"
    # Strict, like the picker: logging out is only meaningful for a server we
    # actually hold — or have observed the need for — a grant on. A stored
    # grant satisfies this on its own, which is the case that matters here.
    if not server_is_oauth_capable(cfg, store):
        return f"MCP server {name!r} does not use OAuth login"
    # Only remote configs carry ``url``; a stdio config reaching here would
    # have already failed the OAuth check above, so the getattr is a type
    # narrowing rather than a guess.
    storage = McpTokenStorage(getattr(cfg, "url", ""), store)
    if not storage.clear():
        return f"no stored credential for MCP server {name!r} — nothing to log out of"
    return None


def mcp_logged_out_servers(store: StructuralAuthStore | None = None) -> set[str] | None:
    """Server URLs that still hold an ``mcp-oauth`` credential row, or
    ``None`` when the store could not be read.

    Read-only companion to :func:`mcp_logout_server` so the ``/mcp logout``
    picker can offer only servers that actually have something to remove
    (mirroring how ``/logout`` offers only providers holding a credential).
    ``None`` rather than the empty set on failure: an unreadable store is not
    the same answer as "no credentials anywhere", and the picker needs the
    difference to say so instead of rendering a bare empty list.
    """
    store = _resolve_store(store)
    if store is None:
        return None
    try:
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
    except Exception:
        logger.debug("MCP credential listing failed", exc_info=True)
        return None
    return {row.identity_key for row in rows if row.identity_key}


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

#: Idle bound on an INTERACTIVE grant: the longest a login waits for a browser
#: round trip that will never complete. The usual reason nothing arrives is
#: that the tab was closed or the consent screen abandoned — indistinguishable
#: from a slow human at the protocol level, so the flow has to give up on a
#: clock and say so. Sized to match the 10-minute budget both login callers
#: already allow the whole connect (``/mcp login`` and ``local-operator mcp
#: login``), so the grant now ends with an explicit "cancelled" receipt inside
#: that window instead of outliving it as a silent "logging in…" line.
INTERACTIVE_GRANT_TIMEOUT_S = 600.0

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


#: Flows whose grant the human abandoned (idle guard fired), keyed by the
#: flow object with the abandonment time as value. This is the side channel
#: that survives the transport: ``callback_handler`` raises a raw
#: ``CancelledError`` for an abandoned grant because the SDK's ``post_writer``
#: eats ordinary exceptions, and the manager consults this registry to tell
#: "the grant died of neglect" apart from "the login task was cancelled".
#: Entries are pruned on every write; a flow object whose grant never
#: abandoned never appears here, and one that did is dropped from the map
#: when its entry is consumed or when it ages past the prune horizon.
#: How long an abandonment record may sit unread. The manager consumes it
#: within seconds of the raise; the horizon only bounds a leak for flows
#: whose connect never got as far as consulting the registry.
_ABANDONED_GRANT_TTL_S = 120.0


class AbandonedGrantLedger:
    """Flows whose grant the human abandoned (idle guard fired).

    This is the side channel that survives the transport: ``callback_handler``
    raises a raw ``CancelledError`` for an abandoned grant because the SDK's
    ``post_writer`` eats ordinary exceptions, and the manager consults the
    ledger to tell "the grant died of neglect" apart from "the login task
    was cancelled". Records are consumed by the manager (``pop``) or age out
    on the next write; keys are weak so a flow nobody consults cannot be kept
    alive by its own record.
    """

    def __init__(self) -> None:
        self._records: weakref.WeakKeyDictionary[LoopbackAuthFlow, float] = (
            weakref.WeakKeyDictionary()
        )

    def record(self, flow: LoopbackAuthFlow) -> None:
        now = time.monotonic()
        for old, at in list(self._records.items()):
            if now - at > _ABANDONED_GRANT_TTL_S:
                self._records.pop(old, None)
        self._records[flow] = now

    def pop(self, flow: LoopbackAuthFlow) -> bool:
        """Consume one flow's abandonment record; ``True`` when one was there."""
        return self._records.pop(flow, None) is not None


ABANDONED_GRANTS = AbandonedGrantLedger()


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

    def __init__(
        self,
        redirect_uri: str,
        server_url: str | None = None,
        *,
        interactive: bool = True,
    ) -> None:
        parsed = urlparse(redirect_uri)
        self.redirect_uri = redirect_uri
        #: Named on the callback page. Someone with several MCP servers
        #: configured has no other way to tell which tab belongs to which
        #: authorization, and "Authorized" without a subject is a page that
        #: could be about anything.
        self.server_url = server_url
        #: Whether this flow may open a browser. Ordinary session startup and
        #: auto-reconnects pass ``False``: when the stored grant cannot be
        #: refreshed they must fail with :class:`McpAuthRequiredError` instead
        #: of popping a login tab the user never asked for. Only an explicit
        #: ``/mcp login`` / ``local-operator mcp login`` runs interactive.
        self.interactive = interactive
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

        Non-interactive flows RAISE here instead of opening a browser: the SDK
        only reaches this handler once the stored grant could not be refreshed,
        and a background connect must surface that as an actionable failure
        ("run /mcp login <name>"), never as a login tab popping up over the
        user's work. The exception propagates out of the connect cleanly — the
        SDK re-raises whatever the handler raises.
        """
        if not self.interactive:
            raise McpAuthRequiredError(self.server_url or self.redirect_uri)
        await self._start_server()
        lines = [
            "\nMCP OAuth authorization required. Open this URL in a browser:",
            f"  <{authorization_url}>",
        ]
        if await open_browser_quietly(authorization_url):
            lines.append("(opened in your default browser)")
        if self._server is not None:
            lines.append(f"Waiting for the redirect to {self.redirect_uri} …")
        self._notify(*lines)

    async def callback_handler(self) -> AuthorizationCodeResult:
        """Wait for the provider's redirect (or a pasted URL) and return the code.

        The transport cannot be trusted to deliver an exception raised here:
        the SDK's ``post_writer`` swallows ordinary auth-flow exceptions, so
        an ABANDONED grant (idle guard fired — the browser went away) is
        recorded in :data:`ABANDONED_GRANTS` and then raised as a RAW
        ``CancelledError``, the one exception shape that unwinds the
        transport's anyio task group intact. The manager recognises the
        pairing and re-voices it as :class:`McpLoginCancelledError`; raising
        the named error here directly would leave it stranded in a log line
        while the connect surfaced an unlabelled cancellation.
        """
        from mcp.shared.auth import AuthorizationCodeResult

        try:
            code, state, iss = await self._await_authorization()
        except AbandonedGrantError:
            ABANDONED_GRANTS.record(self)
            raise asyncio.CancelledError() from None
        finally:
            await self._stop_server()
        return AuthorizationCodeResult(code=code, state=state, iss=iss)

    async def _await_authorization(self) -> tuple[str, str | None, str | None]:
        if self._result is None:
            # No listener: paste is the only route left, and reading stdin is
            # safe precisely because nothing else is going to.
            return await self._await_pasted()
        try:
            # The inner clock is the 300 s redirect bound with its
            # port-forwarding advice for the genuinely slow case; the outer
            # coroutine adds the interactive idle guard AROUND it. They stay
            # separate because ``wait_for`` makes a timeout indistinguishable
            # from an outer one once nested — each clock must convert its own
            # expiry before the next layer can see it.
            # Captured once: a closure read of ``self._result`` types as
            # optional (``_stop_server`` resets it), and it cannot change
            # underneath the wait anyway — only ``_stop_server`` clears it,
            # and that runs after the wait resolves.
            result = self._result

            async def _within_idle_guard() -> tuple[str, str | None, str | None]:
                try:
                    return await asyncio.wait_for(result, timeout=PASTE_INPUT_TIMEOUT_S)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"Timed out after {PASTE_INPUT_TIMEOUT_S:.0f}s waiting for the "
                        f"OAuth redirect to {self.redirect_uri}. If you authorized in a "
                        "browser on another machine it cannot reach this port — "
                        f"forward it (ssh -L {self._port}:127.0.0.1:{self._port} …) "
                        "and try again."
                    ) from exc

            return await asyncio.wait_for(_within_idle_guard(), timeout=INTERACTIVE_GRANT_TIMEOUT_S)
        except asyncio.CancelledError:
            # The login task itself was cancelled (an exclusive re-login, the
            # TUI's stop-ladder, Ctrl+C at the CLI). The underlying result
            # future is cancelled by the wait_for on the way out, and
            # ``callback_handler``'s ``finally`` stops the listener — but only
            # if we CO-OPERATE: shielding the teardown lets one stop-ladder
            # escalation turn a cancelled login into a wedged listener holding
            # the redirect port into the next grant.
            # The interrupt that arrives while we are still WAITING for the
            # redirect is unambiguous: the browser never answered, so this is
            # the task being cancelled, never the abandonment channel (that
            # one only fires from the idle-guard arm). Report it directly.
            with contextlib.suppress(Exception):
                await asyncio.shield(self._stop_server())
            raise McpLoginCancelledError("interrupted before the browser completed it") from None
        except asyncio.TimeoutError as exc:
            # The idle guard, not the redirect clock: nothing arrived for the
            # whole interactive budget, which in practice means the browser
            # went away — tab closed, consent abandoned.
            raise AbandonedGrantError(
                f"no redirect arrived within {INTERACTIVE_GRANT_TIMEOUT_S / 60:.0f} "
                "minutes — the login was probably cancelled (browser tab closed, "
                "or the authorization left unfinished)"
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
                asyncio.to_thread(lambda: input(prompt).strip()),
                timeout=PASTE_INPUT_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            # Same receipt as the listener path: an interrupted CLI login must
            # not surface as a bare "MCP login failed for 'x':" with no reason.
            # The to_thread reader itself cannot be cancelled (that is why
            # paste is never raced), so no teardown is owed here.
            raise McpLoginCancelledError("interrupted before the redirect URL was pasted") from None
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
            # a dev server squatting the port. Not fatal — the paste path still
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
                # Stripped BEFORE the `or`, so a whitespace-only description
                # falls back to the code instead of satisfying the truthiness
                # test and blanking it. `?error=access_denied&error_description=
                # %20%20%20` otherwise raises "OAuth authorization failed:    ",
                # dropping the one word that says what went wrong.
                detail = (query.get("error_description") or [""])[0].strip() or error
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


# --- proactive OAuth refresh -------------------------------------------------
#
# Why this block exists: the SDK only refreshes a token INSIDE its 401 handler,
# and it derives the token endpoint from ``oauth_metadata`` — which a fresh
# process has not discovered yet, so the refresh falls back to
# ``urljoin(server_base, "/token")``. For providers whose token endpoint lives
# elsewhere (Datadog: ``https://us3.datadoghq.com/api/v2/oauth2/token``) that
# guess 404s, the refresh fails, and the SDK escalates to a FULL browser grant
# — the login tab popping up on every startup even though a valid refresh token
# sat in auth.db. The fix is to discover the real endpoints and spend the
# refresh token ourselves BEFORE the provider is built, race-free across the
# several sessions that start together.


@dataclass
class DiscoveredOAuthEndpoints:
    """What PRM/ASM discovery learned about one server's OAuth setup.

    ``oauth_metadata`` carries the real ``token_endpoint`` the refresh must
    target. ``protected_resource_metadata`` (when the server publishes it) is
    what makes the refresh include the RFC 8707 ``resource`` parameter, which
    some providers (Datadog) require. Both are also handed to the provider so a
    later in-flow refresh — e.g. a token that dies mid-session — targets the
    same endpoints instead of re-deriving the wrong guess.
    """

    oauth_metadata: "OAuthMetadata"
    protected_resource_metadata: "ProtectedResourceMetadata | None" = None
    auth_server_url: str | None = None


async def discover_oauth_endpoints(server_url: str) -> DiscoveredOAuthEndpoints | None:
    """Resolve a server's OAuth metadata via SEP-985 PRM then RFC 8414 ASM.

    Returns ``None`` when authorization-server metadata cannot be discovered;
    the caller then degrades to the SDK's own defaults (the pre-fix behavior)
    rather than failing the connect. Discovery is two unauthenticated GETs and
    only runs for OAuth servers, which are already the slow, deferred connects.

    Successful results are cached per process: reconnects rebuild the provider
    and would otherwise re-fetch stable metadata on every backoff rung. Only
    SUCCESSES are cached, so a transient discovery failure retries next time.
    """
    cached = _DISCOVERED_ENDPOINTS_CACHE.get(server_url)
    if cached is not None:
        return cached
    result = await _discover_oauth_endpoints_uncached(server_url)
    if result is not None:
        _DISCOVERED_ENDPOINTS_CACHE[server_url] = result
    return result


#: Per-process cache of successful endpoint discoveries, keyed by server URL.
_DISCOVERED_ENDPOINTS_CACHE: dict[str, DiscoveredOAuthEndpoints] = {}


async def _discover_oauth_endpoints_uncached(
    server_url: str,
) -> DiscoveredOAuthEndpoints | None:
    """The actual PRM/ASM fetch; :func:`discover_oauth_endpoints` caches it."""
    import httpx
    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        build_protected_resource_metadata_discovery_urls,
    )
    from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

    prm: ProtectedResourceMetadata | None = None
    auth_server_url: str | None = None
    timeout = httpx.Timeout(REFRESH_HTTP_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            for url in build_protected_resource_metadata_discovery_urls(None, server_url):
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    continue
                if response.status_code != 200:
                    continue
                try:
                    prm = ProtectedResourceMetadata.model_validate_json(response.content)
                except Exception:  # noqa: BLE001 — malformed metadata: try the next URL
                    continue
                if prm.authorization_servers:
                    auth_server_url = str(prm.authorization_servers[0])
                break

            asm: OAuthMetadata | None = None
            for url in build_oauth_authorization_server_metadata_discovery_urls(
                auth_server_url, server_url
            ):
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    continue
                # Mirror the SDK's fallback semantics: a 4xx means "try the next
                # discovery URL"; anything else non-200 means "stop looking".
                if 400 <= response.status_code < 500:
                    continue
                if response.status_code != 200:
                    break
                try:
                    asm = OAuthMetadata.model_validate_json(response.content)
                except Exception:  # noqa: BLE001 — treat as not-found here
                    asm = None
                break
            if asm is None:
                return None
            return DiscoveredOAuthEndpoints(
                oauth_metadata=asm,
                protected_resource_metadata=prm,
                auth_server_url=auth_server_url,
            )
    except Exception:  # noqa: BLE001 — discovery is best-effort; degrade, don't fail
        logger.debug("OAuth metadata discovery failed for %s", server_url, exc_info=True)
        return None


def _try_lock_exclusive(fd: int) -> bool:
    """ONE non-blocking attempt at the exclusive lock. True when taken.

    Deliberately non-blocking on BOTH platforms. A blocking acquire parks a
    worker thread inside the kernel on a descriptor the calling coroutine may
    be about to close, and on macOS/BSD ``os.close()`` of a descriptor with a
    sibling thread parked in ``flock()`` blocks until that ``flock()`` returns —
    which, with the lock held by another process, is never. Called from the
    event-loop thread's ``finally`` that is exactly the whole TUI freezing:
    no repaint, no input. See :func:`_oauth_refresh_lock` for the full story.
    Retrying a non-blocking attempt is strictly weaker than blocking and is the
    only shape that stays cancellable, so do not "simplify" this back into
    ``fcntl.flock(fd, fcntl.LOCK_EX)``.

    Contention is the expected outcome and returns False; anything else is a
    real fault (EBADF, EINVAL, an unsupported filesystem) that retrying cannot
    fix, so it is raised for the caller to degrade on.
    """
    if os.name == "nt":  # pragma: no cover - platform specific
        import errno as _errno
        import msvcrt

        try:
            # ``msvcrt.locking`` locks a byte RANGE, so the file needs at least
            # one byte to lock — an empty lock file would fail with EINVAL
            # forever. Mirrors ``session_lease``'s handling of the same API.
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as lock_err:
            if lock_err.errno in (_errno.EDEADLOCK, _errno.EACCES, _errno.EAGAIN):
                return False
            raise
    else:
        import errno as _errno
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as lock_err:
            if lock_err.errno in (_errno.EAGAIN, _errno.EACCES, _errno.EWOULDBLOCK):
                return False
            raise


def _acquire_locked_fd(path: str, cancelled: threading.Event) -> int | None:
    """Open ``path`` and take the exclusive lock on it, or give up. Worker-side.

    Runs entirely on a worker thread and OWNS the descriptor for its whole life:
    it opens the fd, retries the non-blocking acquire until the deadline, and on
    any outcome other than success closes the fd ITSELF, in this same thread,
    after the last lock syscall has returned. That ownership rule is the fix for
    the deadlock — the event loop never closes a descriptor another thread might
    still be inside a lock call on, because by construction no such moment
    exists. Returns the locked fd on success (ownership passes to the caller,
    which must unlock and close it) or ``None`` on timeout/cancellation.

    ``cancelled`` is set by the coroutine when its await is cancelled, so an
    abandoned acquire abandons the retry loop within one ``_LOCK_RETRY_SLEEP_S``
    tick instead of holding a thread for the full bound.
    """
    deadline = time.monotonic() + LOCK_ACQUIRE_TIMEOUT_S
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    sleep_s = _LOCK_RETRY_SLEEP_S
    try:
        while True:
            if _try_lock_exclusive(fd):
                return fd
            if cancelled.is_set() or time.monotonic() >= deadline:
                break
            time.sleep(sleep_s)
            # Gentle geometric backoff: keeps pickup fast for the common
            # released-in-a-moment case, then stops burning wakeups once the
            # holder is evidently not finishing soon.
            sleep_s = min(sleep_s * 1.5, _LOCK_RETRY_SLEEP_MAX_S)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None


def _unlock(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _oauth_refresh_lock_path(server_url: str) -> Path:
    """The lock file for ONE server's refresh exchange, in ``config_dir()``.

    Keyed per server, because the race this lock exists to prevent is per
    server: two processes spending the SAME server's rotating refresh token.
    A single global lock file also serialised servers that can never race each
    other, so one slow or unreachable provider parked every other server's
    connect behind it — with six OAuth servers and several concurrent sessions
    that is a queue nothing drains, and the observable symptom was two servers
    connecting and the rest sitting on "connecting" forever.

    The name is a SHA-256 digest of the URL rather than the URL itself: server
    URLs contain ``/``, ``:`` and query strings that are not filename-safe, and
    a digest is stable across runs and machines without any escaping scheme to
    keep in step. Truncated to 16 hex chars — this namespaces a handful of
    configured servers, not an adversarial keyspace.

    The pre-fix global ``mcp_oauth_refresh.lock`` is deliberately left in place
    and never cleaned up: another local-operator process running an older build
    may still be using it, and deleting a file that a live peer holds an flock
    on would silently drop that peer's mutual exclusion (its lock survives on
    the unlinked inode while a new process creates a fresh file and takes an
    uncontended lock). It is a zero-byte file; leaving it costs nothing.
    """
    import hashlib

    from local_operator.paths import config_dir

    lock_dir = config_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(server_url.encode("utf-8")).hexdigest()[:16]
    return lock_dir / f"mcp_oauth_refresh_{digest}.lock"


@contextlib.asynccontextmanager
async def _oauth_refresh_lock(server_url: str):
    """Serialize the refresh exchange across processes for one server.

    Rotating refresh tokens make concurrent refreshes destructive: whichever
    process spends the current token second gets an error — or invalidates the
    first process's brand-new token. Holding an exclusive file lock around the
    exchange, and RE-READING the stored token after acquiring it, guarantees
    exactly one process performs the refresh no matter how many sessions start
    at once. The lock file lives next to ``auth.db``, is per server (see
    :func:`_oauth_refresh_lock_path`), and carries no state: it is only ever
    flocked, never read or written for content (on Windows it holds a single
    padding byte, which ``msvcrt.locking`` requires to have a range to lock).

    Yields True when the lock was taken and False when the bounded acquire gave
    up. A False body must still be SAFE to run: the guarantee degrades from
    "exactly one process refreshes" to "we tried", which is the best-effort
    contract :func:`ensure_mcp_oauth_fresh` already documents. Blocking the
    connect instead would trade a rare double-spend for a guaranteed hang.

    Two rules this function exists to enforce, both learned from a freeze that
    took the whole TUI down:

    1. **The acquire is bounded and non-blocking.** A bare
       ``fcntl.flock(fd, LOCK_EX)`` waits forever, so a lock leaked by a killed
       process parks a connect eternally.
    2. **The event loop never closes a descriptor a thread may be parked on.**
       The acquiring thread owns its fd end to end (:func:`_acquire_locked_fd`
       closes it itself on every non-success path), and the fd only ever
       reaches this coroutine once no lock syscall is outstanding on it. This
       matters because cancellation is routine here — ``/resume`` disposes the
       manager, which cancels in-flight connect tasks mid-acquire — and on
       macOS/BSD ``os.close()`` of a descriptor with a sibling thread inside
       ``flock()`` blocks until that ``flock()`` returns. Called from a
       ``finally`` on the event-loop thread, that stops the loop dead: the
       screen freezes and never repaints. Do not reintroduce a blocking acquire
       or an event-loop-side close of a possibly-in-use fd.
    """
    lock_path = _oauth_refresh_lock_path(server_url)
    # Signals the worker to abandon its retry loop when our await is cancelled,
    # so a cancelled acquire never leaves a thread running to the full bound.
    cancelled = threading.Event()
    # Acquire off the event loop: a contended lock must not stall other
    # servers' connects. The lock is on the fd, so it survives the await.
    acquire = asyncio.create_task(asyncio.to_thread(_acquire_locked_fd, str(lock_path), cancelled))
    try:
        fd = await asyncio.shield(acquire)
    except asyncio.CancelledError:
        # Hand the fd's fate entirely to the worker: tell it to stop, and let
        # the (shielded, so still-running) task close whatever it opened once
        # its last lock syscall has returned. We return immediately — nothing
        # here touches the descriptor, which is what keeps the loop alive.
        cancelled.set()
        acquire.add_done_callback(_close_abandoned_lock_fd)
        raise
    if fd is None:
        logger.debug(
            "MCP OAuth refresh lock not acquired within %.0fs for %s; proceeding unlocked",
            LOCK_ACQUIRE_TIMEOUT_S,
            server_url,
        )
        yield False
        return
    try:
        yield True
    finally:
        # Safe to close on this thread: the worker returned the fd only after
        # its lock call completed, so no thread is parked on it.
        with contextlib.suppress(Exception):
            _unlock(fd)
        os.close(fd)


def _close_abandoned_lock_fd(task: asyncio.Task[int | None]) -> None:
    """Close a lock fd whose waiter was cancelled before it could take it.

    Runs on the event loop when the shielded acquire finally settles, which is
    AFTER the worker thread's last lock syscall — so this close can never block
    the loop. Without it a lock acquired just as its waiter was cancelled would
    leak the descriptor and, worse, keep the lock held for the process's life.
    """
    # On both of these paths the WORKER has already closed the fd itself, so
    # there is nothing here to clean up: a cancelled task never returns one,
    # and an exception unwinds through :func:`_acquire_locked_fd`'s
    # ``except BaseException``, which closes before re-raising. That coupling is
    # what makes the early returns safe — an edit that moves the ``os.open``
    # into the ``try``, or adds a ``return fd`` ahead of the loop, would leak
    # both the descriptor AND the lock here with no diagnostic.
    if task.cancelled():
        return
    if task.exception() is not None:
        return
    fd = task.result()
    if fd is None:
        return
    with contextlib.suppress(Exception):
        _unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


def _refresh_user_agent() -> str:
    """A stable ``local-operator/<version>`` identifier for the refresh POST.

    See the header comment in :func:`_refresh_oauth_token_locked`: Cloudflare
    blocks a no-UA refresh to mcp.notion.com. The version is looked up from the
    installed distribution metadata and degrades to a bare product token when
    running from a source checkout with no metadata, so this can never raise
    into a refresh.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return f"local-operator/{version('local-operator')}"
        except PackageNotFoundError:
            return "local-operator"
    except Exception:  # noqa: BLE001 — a UA lookup must never break a refresh
        return "local-operator"


def _is_invalid_grant(body: bytes) -> bool:
    """True when an OAuth error response body is an ``invalid_grant`` (RFC 6749).

    A revoked or reused refresh token comes back as HTTP 400 with a JSON body
    ``{\"error\": \"invalid_grant\", ...}``. Distinguishing it from a generic
    400 is what lets the caller log the actionable \"run /mcp login\" meaning
    (and never treat it as retriable). Any parse failure returns False so an
    unexpected body is handled as an ordinary rejection, never crashes.
    """
    import json

    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("error") == "invalid_grant"


def _fallback_endpoints_for(server_url: str) -> "DiscoveredOAuthEndpoints":
    """Synthesize the endpoint the SDK itself would refresh against.

    When metadata discovery failed at startup (``endpoints is None``), the SDK's
    ``_refresh_token`` falls back to ``urljoin(<scheme>://<netloc>, \"/token\")``
    — the authorization base URL with its path stripped (see
    ``OAuthContext.get_authorization_base_url``). Building a minimal
    :class:`DiscoveredOAuthEndpoints` that targets exactly that URL lets the
    coordinating provider perform the refresh UNDER THE LOCK with a fresh
    re-read even without discovery, instead of falling through to the SDK's own
    UNLOCKED refresh. That closes the residual reuse window: the SDK's unlocked
    path spends whatever refresh token ``_initialize`` loaded at boot, which a
    sibling may have already rotated away — presenting it a second time is the
    reuse-detection trigger that revokes the whole family.

    ``protected_resource_metadata`` is left None: with no discovery we have no
    PRM, so the refresh omits the RFC 8707 ``resource`` parameter — matching the
    SDK's own fallback path, whose ``should_include_resource_param`` is likewise
    False without PRM (barring a 2025-06-18 protocol header, which the proactive
    refresh does not carry).
    """
    from urllib.parse import urljoin, urlparse

    from mcp.shared.auth import OAuthMetadata

    parsed = urlparse(server_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    token_url = urljoin(base, "/token")
    return DiscoveredOAuthEndpoints(
        oauth_metadata=OAuthMetadata.model_validate(
            {
                "issuer": base,
                # authorization_endpoint is a required field on OAuthMetadata
                # but is unused by the refresh path (refresh only reads
                # token_endpoint); the SDK's own fallback derives it the same
                # way, so a synthesized value keeps the model valid without
                # affecting behaviour.
                "authorization_endpoint": urljoin(base, "/authorize"),
                "token_endpoint": token_url,
            }
        ),
        protected_resource_metadata=None,
        auth_server_url=base,
    )


async def _refresh_oauth_token_locked(
    server_url: str,
    storage: McpTokenStorage,
    endpoints: DiscoveredOAuthEndpoints,
) -> bool:
    """Spend the stored refresh token against the DISCOVERED token endpoint.

    Returns ``True`` when a fresh access token was persisted. The caller holds
    the cross-process refresh lock, so exactly one process performs this
    exchange even when several sessions start together. Mirrors the SDK's
    refresh request exactly (grant type, client auth methods, RFC 6749 §6
    carry-forward) so a provider cannot tell the two apart.
    """
    import base64
    from urllib.parse import quote

    import httpx
    from mcp.shared.auth import OAuthToken
    from mcp.shared.auth_utils import resource_url_from_server_url

    tokens = await storage.get_tokens()
    client_info = await storage.get_client_info()
    if tokens is None or not tokens.refresh_token or client_info is None:
        return False

    token_endpoint = str(endpoints.oauth_metadata.token_endpoint)
    data: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": tokens.refresh_token,
        "client_id": client_info.client_id,
    }
    # RFC 8707 resource indicator: included when the server publishes protected
    # resource metadata, matching the SDK's ``should_include_resource_param``.
    if endpoints.protected_resource_metadata is not None:
        data["resource"] = resource_url_from_server_url(server_url)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        # Explicit UA, not httpx's default. mcp.notion.com sits behind
        # Cloudflare, whose bot heuristics return HTTP 403 "error code 1010"
        # to a refresh POST carrying NO User-Agent (observed live this cycle:
        # httpx's built-in UA slips through, a missing one is blocked). Pinning
        # our own identifier means a future httpx default change or a stricter
        # Cloudflare rule cannot silently turn every refresh into a 403 and
        # force a browser grant on the whole fleet.
        "User-Agent": _refresh_user_agent(),
    }
    auth_method = client_info.token_endpoint_auth_method
    if auth_method == "client_secret_post" and client_info.client_secret:
        data["client_secret"] = client_info.client_secret
    elif auth_method == "client_secret_basic" and client_info.client_secret:
        cid = quote(client_info.client_id, safe="")
        csecret = quote(client_info.client_secret, safe="")
        encoded = base64.b64encode(f"{cid}:{csecret}".encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"

    try:
        # TOTAL wall-clock cap, not just httpx's per-operation timeouts.
        # ``httpx.Timeout(N)`` sets connect/read/write/pool to N EACH, and the
        # read timeout applies per socket read rather than to the whole
        # response — so a server that dribbles the body stays inside every
        # individual read and takes arbitrarily long. Measured against a server
        # sending one byte every 2s, this exact request returned HTTP 200 after
        # 140.7s with the 10s timeout set and never tripping it.
        #
        # That matters because this call runs UNDER the cross-process refresh
        # lock, and ``LOCK_ACQUIRE_TIMEOUT_S`` is derived from the claim that
        # the section is bounded by ``REFRESH_HTTP_TIMEOUT_S``. Without an
        # overall cap that claim is false, and a slow-but-honest provider would
        # systematically push every peer past its acquire bound onto the
        # unlocked degrade path — which for the in-flight coordinator means the
        # SDK's own unlocked refresh, i.e. the rotating-token double-spend this
        # subsystem exists to prevent, reached by timeout instead of by race.
        # Bounding the POST is what makes the documented budget true.
        async with asyncio.timeout(REFRESH_HTTP_TIMEOUT_S):
            async with httpx.AsyncClient(timeout=httpx.Timeout(REFRESH_HTTP_TIMEOUT_S)) as client:
                response = await client.post(token_endpoint, data=data, headers=headers)
    except (httpx.HTTPError, TimeoutError):
        # ``asyncio.timeout`` raises TimeoutError (which ``httpx.HTTPError``
        # does not cover); a refresh that overran its budget is simply a failed
        # refresh, handled exactly like a transport error.
        logger.debug("MCP token refresh request failed for %s", server_url, exc_info=True)
        return False
    if response.status_code != 200:
        # A revoked-grant rejection is qualitatively different from a transient
        # one and must be logged as such: for a rotating provider that runs
        # refresh-token REUSE DETECTION (Notion), presenting an already-rotated
        # refresh token returns HTTP 400 {"error":"invalid_grant"} and revokes
        # the ENTIRE token family, logging out every session at once. When that
        # happens the only recovery is an interactive login, so the log names
        # that action instead of implying a retry will heal it. We never
        # auto-retry here regardless: returning False lets the connect surface
        # McpAuthRequiredError, which the manager turns into a suspended
        # reconnect (see manager._reconnect's McpAuthRequiredError arm) rather
        # than hammering a dead grant.
        if response.status_code == 400 and _is_invalid_grant(response.content):
            logger.info(
                "MCP OAuth grant revoked for %s (invalid_grant); run /mcp login to restore it",
                server_url,
            )
            return False
        # Informational, not debug: a rejected refresh is the thing that turns
        # into a login prompt, so its cause belongs in the readable log.
        logger.info("MCP token refresh rejected for %s: HTTP %s", server_url, response.status_code)
        return False
    try:
        new_tokens = OAuthToken.model_validate_json(response.content)
    except Exception:  # noqa: BLE001 — an unparseable token is a failed refresh
        logger.debug(
            "MCP token refresh returned an invalid token for %s", server_url, exc_info=True
        )
        return False

    # RFC 6749 §6: a refresh response may omit ``scope`` (unchanged) and
    # ``refresh_token`` (not rotated). Carry both forward so the persisted row
    # stays self-describing and can refresh again next time.
    if new_tokens.scope is None and tokens.scope is not None:
        new_tokens.scope = tokens.scope
    if new_tokens.refresh_token is None:
        new_tokens.refresh_token = tokens.refresh_token
    await storage.set_tokens(new_tokens)
    return True


async def ensure_mcp_oauth_fresh(
    server_url: str,
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
) -> DiscoveredOAuthEndpoints | None:
    """Refresh a stored OAuth grant before connecting, race-free. Best-effort.

    Returns the discovered endpoints so the provider can be primed with them
    (``None`` when discovery failed and the SDK should fall back to its own
    defaults). This never opens a browser, and a failed REFRESH never raises:
    the stored token is simply left as-is, and the provider's non-interactive
    redirect handler is what converts the resulting grant attempt into an
    actionable :class:`McpAuthRequiredError` instead of a login tab. Lock
    ACQUISITION can still raise ``OSError`` (unwritable config dir, exhausted
    fds, the bounded Windows retry) — the manager's caller wraps this in a
    broad catch, and any new caller must do the same or accept the raise.

    ``cfg`` is accepted for signature stability (the manager passes it, and a
    future per-server knob — e.g. opting out of proactive refresh — will need
    it) but is not consulted today.

    The refresh is wrapped in a cross-process lock with a re-read after
    acquiring it, so only one of several concurrently STARTING sessions spends
    a rotating refresh token. Scope honestly stated: the SDK's own in-flow 401
    refresh (a token that dies mid-session) takes no such lock, so two
    already-running sessions can still race a rotation there; that path fails
    into the non-interactive handler (an actionable error, not a popup) and
    the next startup heals it here.
    """
    del cfg  # reserved — see docstring
    storage = McpTokenStorage(server_url, store)
    endpoints = await discover_oauth_endpoints(server_url)

    def _still_good(expiry: float | None, tokens: Any) -> bool:
        # ``expiry is None`` means "no lifetime recorded" — the SDK treats such
        # a token as valid, so we must not force a refresh on it.
        return bool(tokens is not None and tokens.access_token) and (
            expiry is None or time.time() < expiry - REFRESH_SKEW_S
        )

    if _still_good(storage.stored_token_expiry(), await storage.get_tokens()):
        return endpoints

    tokens = await storage.get_tokens()
    if (
        endpoints is None
        or await storage.get_client_info() is None
        or tokens is None
        or not tokens.refresh_token
    ):
        return endpoints

    async with _oauth_refresh_lock(server_url) as locked:
        if not locked:
            # The bounded acquire gave up, so we cannot claim exclusivity and
            # must not spend the rotating refresh token on a guess. Skip the
            # PROACTIVE refresh and connect anyway: the SDK's own in-flow
            # refresh still runs on the first 401, and the coordinator wrapper
            # re-reads the store before it does. A skipped optimisation costs a
            # round trip; blocking the connect costs the whole session.
            return endpoints
        # Re-read under the lock: another session may have refreshed while we
        # waited. Spending a rotated refresh token a second time is exactly the
        # race this lock exists to prevent.
        if not _still_good(storage.stored_token_expiry(), await storage.get_tokens()):
            await _refresh_oauth_token_locked(server_url, storage, endpoints)
    return endpoints


def _resolve_redirect_uri(cfg: MCPServerConfig) -> str:
    """The loopback redirect URI a config's ``oauth`` block resolves to.

    Shared by :func:`wire_oauth_auth` (which advertises it in the client
    metadata) and :func:`build_oauth_provider` (which binds the flow's
    listener to it): the two MUST agree, or the provider redirects the
    browser to an address nothing is serving.
    """
    oauth = cfg.oauth
    callback_port = (oauth.callback_port if oauth is not None else None) or DEFAULT_CALLBACK_PORT
    callback_path = (oauth.callback_path if oauth is not None else None) or DEFAULT_CALLBACK_PATH
    if not callback_path.startswith("/"):
        callback_path = f"/{callback_path}"
    return (oauth.redirect_uri if oauth is not None else None) or (
        f"http://127.0.0.1:{callback_port}{callback_path}"
    )


def wire_oauth_auth(
    server_url: str,
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
    *,
    interactive: bool = True,
    flow: LoopbackAuthFlow | None = None,
) -> dict[str, Any]:
    """Build ``OAuthClientProvider`` kwargs for one server.

    ``cfg`` is the server's :class:`~local_operator.mcp.config.MCPServerConfig`
    (its ``auth`` / ``oauth`` blocks supply client identity and callback
    knobs). Returns a dict suitable for ``OAuthClientProvider(**kwargs)``:

    - ``server_url``: the MCP server URL (resource indicator base);
    - ``client_metadata``: PKCE authorization-code client, redirect URI
      ``http://127.0.0.1:{callback_port or DEFAULT_CALLBACK_PORT}``
      ``{callback_path or /callback}`` (PKCE itself is automatic inside the
      SDK);
    - ``storage``: a :class:`McpTokenStorage` bound to ``store``; a config
      ``client_id`` pre-seeds the client registration so DCR is skipped
      (MCP-11);
    - ``redirect_handler`` / ``callback_handler``: the two halves of one
      :class:`LoopbackAuthFlow`, which listens on that redirect URI for the
      duration of the grant (see the module docstring). Callers that need
      the flow itself — :func:`build_oauth_provider` does, for the manager's
      abandoned-grant check — pass their own via ``flow``; the dict returned
      here stays exactly the SDK's kwargs so it can be splatted straight
      into ``OAuthClientProvider``.

    ``interactive`` controls whether the flow may open a browser. Ordinary
    session startup and auto-reconnects pass ``False`` so an unrefreshable
    grant surfaces as :class:`McpAuthRequiredError` instead of popping a login
    tab; only an explicit ``/mcp login`` runs interactive.

    The returned dict is constructed eagerly but imports ``mcp`` lazily inside
    so config-only code paths never touch the SDK.
    """
    from mcp.shared.auth import OAuthClientMetadata

    auth = cfg.auth
    oauth = cfg.oauth

    redirect_uri = _resolve_redirect_uri(cfg)

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

    if flow is None:
        flow = LoopbackAuthFlow(redirect_uri, server_url=server_url, interactive=interactive)
    return {
        "server_url": server_url,
        "client_metadata": client_metadata,
        "storage": storage,
        "redirect_handler": flow.redirect_handler,
        "callback_handler": flow.callback_handler,
    }


def _make_refresh_coordinating_provider(
    kwargs: dict[str, Any],
    *,
    server_url: str,
    storage: "McpTokenStorage",
    endpoints: DiscoveredOAuthEndpoints | None,
) -> Any:
    """An ``OAuthClientProvider`` whose in-flow refresh is race-free across processes.

    Why this subclass exists: the SDK loads the stored tokens ONCE in
    ``_initialize`` and never re-reads storage afterwards, and its in-flow
    refresh (``async_auth_flow`` -> ``_refresh_token``) spends whatever refresh
    token that first read put in memory, under NO cross-process lock. That is
    fine for a token that expires and is refreshed once. It is destructive for a
    provider that ROTATES its refresh token on every use (Notion: an 8-hour
    access token plus a rotating refresh token) the moment more than one
    local-operator process is alive: this harness runs one process per cmux
    workspace, so several long-lived sessions cross the same 8-hour boundary
    still holding the same in-memory refresh token from when they booted. They
    each POST it; the authorization server rotates; the first wins and every
    other session's brand-new token is already dead. The SDK then
    ``clear_tokens()`` and the next request opens a FULL browser grant — the
    "Notion logged out again" the user sees, several times a day.

    :func:`ensure_mcp_oauth_fresh` already closes this race for the STARTUP
    refresh with a cross-process file lock and a re-read under it; its own
    docstring notes that the SDK's in-flow refresh is the remaining unlocked
    path. This subclass extends the same guard to that path: before the SDK
    would refresh, it re-reads the persisted token under
    :func:`_oauth_refresh_lock` and, if a sibling process already rotated it,
    ADOPTS the fresh token into the context and skips the refresh entirely.
    Only when the store still holds an expired token does exactly one process
    perform the exchange — against the discovered endpoint, so a provider whose
    token endpoint is not ``<server_base>/token`` (Datadog) refreshes rather
    than 404-ing into a browser grant.

    The invariant, enforced on EVERY path: the SDK's own (unlocked)
    ``_refresh_token`` must never run with an in-memory refresh token older than
    what storage holds. With no discovered ``endpoints`` (metadata discovery
    failed at startup) the coordinator no longer falls through to that unlocked
    refresh — it synthesizes the SDK's own fallback token endpoint
    (``<scheme>://<netloc>/token``) and performs the exchange UNDER THE LOCK
    with a fresh re-read, so even without discovery a stale boot-time refresh
    token is never presented a second time. Presenting one to a reuse-detecting
    provider (Notion) is what returns ``invalid_grant`` and revokes the whole
    token family. If the coordinated refresh itself raises, a final under-lock
    re-read still adopts the freshest persisted token before the SDK runs, so
    the invariant survives the exception fall-through too.

    The subclass additionally intercepts the FIRST 401 the resource server
    returns for the original request. The coordination step above only fires
    when the loaded token is EXPIRED, but a rotating provider (Notion) revokes
    every previously issued access token when a sibling refreshes — so a
    session can hold a locally-valid, server-side-revoked token that sails
    past coordination and gets a 401. Before that 401 reaches the SDK's full
    browser-authorization branch, the flow re-reads the store under the lock
    and, if a peer's different token is already there, adopts it and re-sends
    the request once. See ``async_auth_flow`` for the bound and the pass-through
    rule for genuinely dead grants.
    """
    from mcp.client.auth import OAuthClientProvider

    class _RefreshCoordinatingOAuthProvider(OAuthClientProvider):
        # Bound on the re-read/refresh interception: a stuck lock acquisition or
        # a hung endpoint must not park an authenticated request forever. The
        # file lock is only ever held for one token POST plus the SQLite read,
        # and that POST is bounded in TOTAL wall time by the ``asyncio.timeout``
        # in ``_refresh_oauth_token_locked`` (httpx's own per-operation timeout
        # does not bound a dribbling response), so a peer that legitimately
        # holds it clears well inside this bound. Overrunning it means a leaked
        # lock, and degrading to the SDK's own refresh is strictly better than
        # blocking the request.
        _refresh_coord_server_url = server_url
        _refresh_coord_storage = storage
        _refresh_coord_endpoints = endpoints

        async def _coordinate_inflight_refresh(self) -> None:
            """Re-sync from the store (and refresh once, race-free) if the SDK is
            about to refresh. No-op unless the loaded token is invalid AND
            refreshable — exactly the SDK's own in-flow refresh trigger, so this
            never adds a round trip to a request that would have gone through
            unauthenticated-refresh-free."""
            ctx = self.context
            # Gate on the SAME predicate the SDK's async_auth_flow uses so we
            # intercept precisely when it would refresh, and never otherwise.
            if ctx.is_token_valid() or not ctx.can_refresh_token():
                return
            try:
                async with _oauth_refresh_lock(self._refresh_coord_server_url) as locked:
                    # Re-read under the lock: a sibling process may have rotated
                    # the token while we waited. Adopting its result is what
                    # turns a double-spend into a no-op.
                    await self._resync_from_store(ctx)
                    if not locked:
                        # No exclusivity, so we must not perform the exchange
                        # ourselves. The re-read above still upholds the
                        # invariant that matters most (the SDK never spends an
                        # in-memory refresh token older than storage's), and the
                        # SDK's own refresh then proceeds — the pre-coordination
                        # behaviour, which is the correct degrade.
                        return
                    if ctx.is_token_valid():
                        return  # a peer already refreshed; do not spend again
                    # The invariant this whole block exists to guarantee: the
                    # SDK's own ``_refresh_token`` must NEVER run with an
                    # in-memory refresh token older than the one in storage. The
                    # SDK loads the token once at ``_initialize`` and spends it
                    # unlocked; a sibling that rotated it in between leaves us
                    # holding a stale refresh token, and presenting that to a
                    # reuse-detecting provider (Notion) returns
                    # ``invalid_grant`` and revokes the ENTIRE token family —
                    # logging out every session at once. So we always perform
                    # the refresh ourselves, UNDER THE LOCK, with a fresh
                    # re-read; the SDK's unlocked path is never reached with a
                    # stale token.
                    endpoints = self._refresh_coord_endpoints
                    if endpoints is None:
                        # Discovery failed at startup, but we must still refresh
                        # under the lock rather than fall through to the SDK's
                        # UNLOCKED refresh (which would spend the possibly-stale
                        # boot-time token and risk the family revocation above).
                        # Synthesize the exact endpoint the SDK itself would
                        # fall back to (``<scheme>://<netloc>/token``) so the
                        # locked, re-reading refresh targets the same URL the
                        # unlocked path would have.
                        endpoints = _fallback_endpoints_for(self._refresh_coord_server_url)
                    refreshed = await _refresh_oauth_token_locked(
                        self._refresh_coord_server_url,
                        self._refresh_coord_storage,
                        endpoints,
                    )
                    if refreshed:
                        await self._resync_from_store(ctx)
            except Exception:  # noqa: BLE001 — coordination is best-effort
                # A failed re-read/refresh must never break the request. But we
                # must NOT let the SDK's unlocked refresh then spend a stale
                # boot-time token: re-read storage one final time under the lock
                # and overwrite the in-memory token with whatever is persisted,
                # so whatever the SDK spends next is at least the freshest
                # stored refresh token, never an older one a sibling already
                # rotated away. This upholds the same invariant on the exception
                # fall-through path (a raised locked-refresh, a transient store
                # error) as the success path does.
                logger.debug(
                    "MCP in-flight refresh coordination failed for %s",
                    self._refresh_coord_server_url,
                    exc_info=True,
                )
                await self._adopt_freshest_stored_token(ctx)

        async def _resync_from_store(self, ctx: Any) -> None:
            """Overwrite the in-memory token with the persisted one.

            Called under :func:`_oauth_refresh_lock`. Reads the store's current
            token through the same storage the provider persists through and,
            when present, adopts it into the context along with its recomputed
            expiry. Both the success path and the exception fall-through share
            this one definition of "sync from store" so the invariant (in-memory
            refresh token never older than storage) is enforced identically on
            every path.
            """
            stored = await ctx.storage.get_tokens()
            if stored is not None and stored.access_token:
                ctx.current_tokens = stored
                ctx.token_expiry_time = self._refresh_coord_storage.stored_token_expiry()

        async def _adopt_freshest_stored_token(self, ctx: Any) -> None:
            """Final under-lock re-read on the exception fall-through path.

            Guarantees the invariant even when the coordinated refresh raised:
            re-read storage under the refresh lock and adopt the freshest
            persisted token, so the SDK's subsequent unlocked refresh can never
            spend an in-memory refresh token older than what is on disk. Best
            effort — a failure here just leaves the pre-fix behaviour, never a
            raise into the request.
            """
            try:
                async with _oauth_refresh_lock(self._refresh_coord_server_url):
                    # Adopt regardless of whether the lock was taken: this is a
                    # READ, and reading the freshest persisted token is strictly
                    # better than keeping a staler in-memory one even unlocked.
                    await self._resync_from_store(ctx)
            except Exception:  # noqa: BLE001 — best-effort; never break the request
                logger.debug(
                    "MCP in-flight refresh final re-read failed for %s",
                    self._refresh_coord_server_url,
                    exc_info=True,
                )

        async def async_auth_flow(self, request):  # type: ignore[override]
            # Ensure tokens+client_info are loaded (so is_token_valid /
            # can_refresh_token below are meaningful), then coordinate the
            # refresh, then hand off to the SDK's flow. The SDK re-checks
            # is_token_valid under its own lock and will skip its refresh branch
            # because we have already made the token valid. context.lock is NOT
            # held across the coordination (it is not reentrant and the SDK
            # re-acquires it), matching how the SDK itself only holds it inside
            # the flow.
            async with self.context.lock:
                if not self._initialized:
                    await self._initialize()
            await self._coordinate_inflight_refresh()
            # Delegate to the SDK flow by hand, forwarding the RESPONSE the
            # caller sends back into each yield: httpx drives an auth flow with
            # ``gen.asend(response)``, and a plain ``async for ...: yield`` would
            # swallow those sent values (the sub-generator would see ``None`` for
            # every ``response = yield`` and crash on ``response.status_code``).
            # Manual pumping is the only correct way to delegate a receiving
            # async generator — there is no ``yield from`` for them.
            #
            # The whole pump is wrapped so the inner generator is ALWAYS closed
            # and any exception/cancellation is delivered INTO it, never dropped
            # on the floor. This matters because the SDK's ``async_auth_flow``
            # holds ``context.lock`` across its entire body: httpx re-raises a
            # transport error mid-flow (``_send_handling_auth`` does
            # ``raise exc`` after ``response.aclose()``), and if we let that
            # unwind past us without closing ``inner``, the SDK generator is
            # suspended forever at its ``yield`` still holding the lock — every
            # later request to this server then deadlocks on ``context.lock``.
            # ``athrow`` runs the SDK's own ``finally`` (which releases the lock
            # via the ``async with`` exit); ``aclose`` covers the GeneratorExit
            # path when httpx closes the OUTER flow (its ``finally:
            # await auth_flow.aclose()``).
            inner = super().async_auth_flow(request)
            # One 401-driven token adoption is allowed per flow invocation.
            # ``original_request`` is the caller's request object itself (NOT
            # the first yield: when the loaded token is expired the SDK yields
            # a refresh request first, and latching on the first yield would
            # point the identity guard at the wrong object). The SDK re-auth
            # machinery never yields this object again except its own end-of-flow
            # retry, so ``outgoing is original_request`` reliably identifies the
            # caller's request; ``adoption_attempted`` spends the one-retry
            # budget — a second 401 must pass through untouched.
            original_request: Any = request
            adoption_attempted = False
            try:
                try:
                    outgoing = await inner.__anext__()
                except StopAsyncIteration:
                    return
                while True:
                    try:
                        response = yield outgoing
                    except GeneratorExit:
                        # The outer flow is being closed (httpx's finally, or a
                        # cancellation): close the inner one so the SDK unwinds
                        # its lock, then let the close propagate.
                        raise
                    except BaseException as exc:  # noqa: BLE001
                        # An exception thrown INTO us (httpx ``athrow`` on a
                        # transport fault): deliver it into the SDK generator so
                        # its ``finally`` runs and the lock is released, then
                        # relay whatever it yields or re-raises.
                        try:
                            outgoing = await inner.athrow(exc)
                        except StopAsyncIteration:
                            return
                        continue
                    if (
                        response is not None
                        and response.status_code == 401
                        and not adoption_attempted
                        and outgoing is original_request
                    ):
                        # The coordination step above only fires when the loaded
                        # token is EXPIRED, but Notion revokes every previously
                        # issued access token the moment any sibling process
                        # rotates the grant. Every other live session then holds
                        # a locally-VALID, server-side-REVOKED token: it skips
                        # coordination, sends the corpse, and this 401 is what
                        # comes back. The SDK's answer to a 401 is the FULL
                        # browser authorization (non-interactive connects turn
                        # that into McpAuthRequiredError and suspend
                        # auto-reconnect), even though the shared store already
                        # holds the sibling's fresh token. So before the 401
                        # reaches the SDK, re-read the store under the refresh
                        # lock and adopt a DIFFERENT token if a peer wrote one.
                        #
                        # Adoption is race-free without spending anything: it
                        # copies a token a sibling already paid a refresh for,
                        # so it cannot invalidate anything and cannot double-
                        # spend the rotating refresh token. The lock only
                        # serializes the re-read against a concurrent rotation
                        # so the adopted value is not mid-write. It is bounded
                        # to ONE attempt per flow: if the adopted token ALSO
                        # 401s (a genuinely dead grant, e.g. the user revoked
                        # access server-side), the second 401 passes through to
                        # the SDK's own full-flow branch exactly as before —
                        # adoption never loops. A grant that is truly dead
                        # (stored token identical to ours, or none) also passes
                        # through unchanged on this first 401.
                        adoption_attempted = True
                        retry_request = await self._adopt_peer_token_once(original_request)
                        if retry_request is not None:
                            # Re-yield the retry request: the SAME httpx client
                            # that sent the original sends this one (same
                            # proxies, TLS, and event hooks), exactly matching
                            # the SDK's own end-of-flow 401 retry, which
                            # re-yields the request after ``_add_auth_header``.
                            # Then feed the retry's response into the SDK
                            # generator INSTEAD of the 401, so its full
                            # browser-authorization branch never sees a
                            # challenge.
                            try:
                                retry_response = yield retry_request
                            except GeneratorExit:
                                # The outer flow is closing: re-raise so the
                                # ``finally`` below closes the inner generator
                                # and the SDK unwinds its lock.
                                raise
                            except BaseException as exc:  # noqa: BLE001
                                # A transport fault on the retry: deliver it into
                                # the SDK generator so its ``finally`` runs and
                                # the lock is released, then relay what it yields.
                                try:
                                    outgoing = await inner.athrow(exc)
                                except StopAsyncIteration:
                                    return
                                continue
                            try:
                                outgoing = await inner.asend(retry_response)
                            except StopAsyncIteration:
                                return
                            continue
                        # No peer token to adopt: fall through and hand the 401
                        # to the SDK unchanged (existing dead-grant behaviour).
                    # ``response`` is whatever the caller sent into the flow;
                    # httpx always sends a real Response, so the None case is a
                    # type-narrowing artifact, not a reachable state.
                    assert response is not None  # noqa: S101 — httpx never sends None
                    try:
                        outgoing = await inner.asend(response)
                    except StopAsyncIteration:
                        return
            finally:
                # Unconditional: a normal return, a raise, or a GeneratorExit all
                # pass through here, so the SDK generator (and its held lock) is
                # never left suspended. Idempotent if already exhausted/closed.
                await inner.aclose()

        async def _adopt_peer_token_once(self, original_request: Any) -> Any:
            """Adopt a peer-rotated token and return the request to retry, if any.

            Returns the ORIGINAL request with its Authorization header rewritten
            to the adopted token when the store held an access token DIFFERENT
            from the one in memory — the caller re-yields it so the SAME httpx
            client that sent the original sends the retry (same proxies, TLS,
            and event hooks). Returns ``None`` when there is nothing to adopt
            (stored token identical to ours, or no row), in which case the
            caller passes the original 401 through to the SDK.

            Mutating the request object in place (rather than building a new
            one) matches the SDK's own end-of-flow 401 retry contract:
            ``_add_auth_header`` rewrites the same object's Authorization header
            and re-yields it. Header mutation never touches the request
            body/stream, so the re-send is body-safe.
            """
            try:
                async with _oauth_refresh_lock(self._refresh_coord_server_url):
                    # Adoption only READS the store and rewrites our own header;
                    # it never spends the refresh token, so an unlocked pass is
                    # safe and still beats handing a 401 straight to the SDK.
                    stored = await self.context.storage.get_tokens()
                    if stored is None or not stored.access_token:
                        return None
                    current = self.context.current_tokens
                    if current is not None and stored.access_token == current.access_token:
                        # The store still holds exactly the token that just 401'd:
                        # this grant is dead everywhere, not merely revoked for
                        # us. The SDK's full flow is the right answer.
                        return None
                    self.context.current_tokens = stored
                    self.context.token_expiry_time = (
                        self._refresh_coord_storage.stored_token_expiry()
                    )
                    original_request.headers["Authorization"] = f"Bearer {stored.access_token}"
            except Exception:  # noqa: BLE001 — adoption is best-effort
                # A failed re-read must not break the request: defer to the
                # SDK's own 401 handling rather than inventing a new failure.
                logger.debug(
                    "MCP 401 token adoption failed for %s",
                    self._refresh_coord_server_url,
                    exc_info=True,
                )
                return None
            logger.debug(
                "MCP 401 adoption retrying %s with a peer-rotated token",
                self._refresh_coord_server_url,
            )
            return original_request

    return _RefreshCoordinatingOAuthProvider(**kwargs)


def build_oauth_provider(
    server_url: str,
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
    *,
    interactive: bool = True,
    endpoints: DiscoveredOAuthEndpoints | None = None,
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

    ``interactive`` is forwarded to the flow (see :func:`wire_oauth_auth`).
    ``endpoints`` — the result of :func:`ensure_mcp_oauth_fresh` — primes the
    provider's authorization-server metadata, so a token that dies MID-session
    and needs an in-flow refresh targets the real token endpoint instead of the
    SDK's ``<server_base>/token`` guess (which 404s for providers like Datadog
    whose token endpoint lives on a different host).
    """
    # The flow is created HERE (not inside wire_oauth_auth) so it can be
    # attached to the provider: an abandoned grant arrives at the connect as
    # a raw CancelledError (see ``LoopbackAuthFlow.callback_handler``), and
    # the ABANDONED_GRANTS ledger keyed by this object is what identifies it.
    # Its redirect URI must match the one wire_oauth_auth computes, which a
    # config override can change — one helper keeps the two from drifting.
    flow = LoopbackAuthFlow(
        _resolve_redirect_uri(cfg), server_url=server_url, interactive=interactive
    )
    kwargs = wire_oauth_auth(server_url, cfg, store=store, interactive=interactive, flow=flow)
    storage = kwargs["storage"]
    # A refresh-coordinating provider (not the bare SDK one): its in-flow
    # refresh re-reads the store under the cross-process lock so several
    # long-lived sessions cannot double-spend a rotating refresh token
    # mid-session — see :func:`_make_refresh_coordinating_provider`.
    provider = _make_refresh_coordinating_provider(
        kwargs, server_url=server_url, storage=storage, endpoints=endpoints
    )
    provider._loopback_flow = flow  # type: ignore[attr-defined]
    if endpoints is not None:
        provider.context.oauth_metadata = endpoints.oauth_metadata
        provider.context.protected_resource_metadata = endpoints.protected_resource_metadata
        provider.context.auth_server_url = endpoints.auth_server_url
    try:
        expiry = storage.stored_token_expiry()
    except Exception:  # noqa: BLE001 — a metadata read must not block a connect
        logger.debug("MCP token expiry unreadable for %s", server_url, exc_info=True)
        return provider
    if expiry is not None:
        provider.context.token_expiry_time = expiry
    return provider

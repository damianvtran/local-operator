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
import time
import weakref
from dataclasses import dataclass
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


def oauth_server_names(cwd: str | os.PathLike[str]) -> list[str]:
    """Names of configured OAuth-enabled servers, in config order.

    The ``/mcp login|reauth|logout`` argument lists are filled from this, so
    they offer exactly the servers those commands can act on — a stdio or
    API-key server has no OAuth grant to log into or out of, and offering it
    would be a row whose only outcome is a warning notice.
    """
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _sources = load_all_mcp_configs(cwd)
    return [
        name
        for name, cfg in configs.items()
        if getattr(getattr(cfg, "auth", None), "type", None) == "oauth"
    ]


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
    auth = getattr(cfg, "auth", None)
    if auth is None or getattr(auth, "type", None) != "oauth":
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


def _lock_exclusive(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - platform specific
        import errno as _errno
        import msvcrt
        import time as _time

        # ``LK_LOCK`` blocks but gives up after ~10 s (it retries once per
        # second, ten times, then raises OSError) — while a peer legitimately
        # holds the lock for up to REFRESH_HTTP_TIMEOUT_S of network time.
        # Retry in a loop to match the POSIX flock's indefinite-block
        # semantics; the loop is bounded generously rather than forever so a
        # leaked lock (killed process) cannot park a connect eternally.
        # Contention surfaces as EDEADLOCK/EACCES; anything else (EBADF,
        # EINVAL) is a real fault that retrying cannot fix — raising it
        # immediately beats hot-spinning the worker thread for the bound.
        deadline = _time.monotonic() + 60.0
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                return
            except OSError as lock_err:
                contended = lock_err.errno in (_errno.EDEADLOCK, _errno.EACCES)
                if not contended or _time.monotonic() >= deadline:
                    raise
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.asynccontextmanager
async def _oauth_refresh_lock(server_url: str):
    """Serialize the refresh exchange across processes for one server.

    Rotating refresh tokens make concurrent refreshes destructive: whichever
    process spends the current token second gets an error — or invalidates the
    first process's brand-new token. Holding an exclusive file lock around the
    exchange, and RE-READING the stored token after acquiring it, guarantees
    exactly one process performs the refresh no matter how many sessions start
    at once. The lock file lives next to ``auth.db`` and is only ever flocked,
    never written.
    """
    from local_operator.paths import config_dir

    lock_dir = config_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "mcp_oauth_refresh.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        # Acquire off the event loop: a contended lock must not stall other
        # servers' connects. The lock is on the fd, so it survives the await.
        await asyncio.to_thread(_lock_exclusive, fd)
        yield
    finally:
        with contextlib.suppress(Exception):
            _unlock(fd)
        os.close(fd)


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

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth_method = client_info.token_endpoint_auth_method
    if auth_method == "client_secret_post" and client_info.client_secret:
        data["client_secret"] = client_info.client_secret
    elif auth_method == "client_secret_basic" and client_info.client_secret:
        cid = quote(client_info.client_id, safe="")
        csecret = quote(client_info.client_secret, safe="")
        encoded = base64.b64encode(f"{cid}:{csecret}".encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(REFRESH_HTTP_TIMEOUT_S)) as client:
            response = await client.post(token_endpoint, data=data, headers=headers)
    except httpx.HTTPError:
        logger.debug("MCP token refresh request failed for %s", server_url, exc_info=True)
        return False
    if response.status_code != 200:
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

    async with _oauth_refresh_lock(server_url):
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
    from mcp.client.auth import OAuthClientProvider

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
    provider = OAuthClientProvider(**kwargs)
    provider._loopback_flow = flow  # type: ignore[attr-defined]
    if endpoints is not None:
        provider.context.oauth_metadata = endpoints.oauth_metadata
        provider.context.protected_resource_metadata = endpoints.protected_resource_metadata
        provider.context.auth_server_url = endpoints.auth_server_url
    storage = kwargs["storage"]
    try:
        expiry = storage.stored_token_expiry()
    except Exception:  # noqa: BLE001 — a metadata read must not block a connect
        logger.debug("MCP token expiry unreadable for %s", server_url, exc_info=True)
        return provider
    if expiry is not None:
        provider.context.token_expiry_time = expiry
    return provider

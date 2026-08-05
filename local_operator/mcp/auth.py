"""MCP OAuth support on the official SDK's ``OAuthClientProvider``.

Headless-capable flow (official SDK PKCE + RFC 7591 DCR under the hood):

- ``wire_oauth_auth(server_url, cfg)`` returns ``OAuthClientProvider`` kwargs:
  client metadata with a loopback redirect URI, a token storage bound to the
  shared credential store, and redirect/callback handlers that print the
  authorization URL and accept a pasted redirect URL on stdin.
- ``McpTokenStorage`` is the SDK ``TokenStorage``: one row per server URL in
  the real ``providers.auth_store.AuthStore``, keyed ``mcp_oauth:<url>``.

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
import logging
import sys
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Logical credential id prefix for managed MCP OAuth credentials (URL-keyed).
MCP_OAUTH_CREDENTIAL_PREFIX = "mcp_oauth:"

# Provider column value in the shared auth_credentials table.
MCP_OAUTH_PROVIDER = "mcp-oauth"

DEFAULT_CALLBACK_PORT = 3000
DEFAULT_CALLBACK_PATH = "/callback"


def mcp_oauth_credential_id(server_url: str) -> str:
    """Stable logical credential id for one MCP server's OAuth grant."""
    return f"{MCP_OAUTH_CREDENTIAL_PREFIX}{server_url}"


@runtime_checkable
class StructuralAuthStore(Protocol):
    """The slice of ``providers.auth_store.AuthStore`` this module consumes.

    Redefined to the REAL store's methods (MCP-03) so a test fake mirrors
    reality: integer-keyed rows, ``provider`` column, ``identity_key`` dedupe.
    """

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> Any:
        """Insert, or update the row for the same identity; returns the row."""
        ...

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[Any]:
        """Enabled credential rows (all providers or one), oldest first."""
        ...

    def get_credential(self, credential_id: int) -> Any | None:
        """Return one row by integer id, or ``None``."""
        ...


def _resolve_store(store: Any) -> Any:
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

    def __init__(self, server_url: str, store: Any = None) -> None:
        self.server_url = server_url
        self.credential_id = mcp_oauth_credential_id(server_url)
        self._store = _resolve_store(store)

    def _read(self) -> dict[str, Any] | None:
        """Row payload for this server URL, or ``None`` (no store/no row)."""
        store = self._store
        if store is None:
            return None
        try:
            rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        except Exception:
            logger.debug("MCP token read failed for %s", self.credential_id, exc_info=True)
            return None
        for row in rows:
            if getattr(row, "identity_key", None) == self.server_url:
                data = getattr(row, "data", None)
                return dict(data) if isinstance(data, dict) else None
        return None

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

    async def get_tokens(self) -> Any:
        """Stored access/refresh tokens as an ``OAuthToken``, or ``None``."""
        creds = self._read()
        tokens = creds.get("tokens") if isinstance(creds, dict) else None
        if not isinstance(tokens, dict):
            return None
        try:
            from mcp.shared.auth import OAuthToken

            return OAuthToken.model_validate(tokens)
        except Exception:
            logger.debug("Stored MCP tokens invalid for %s", self.credential_id, exc_info=True)
            return None

    async def set_tokens(self, tokens: Any) -> None:
        """Persist fresh/refreshed tokens (access + refresh together)."""
        creds = self._read() or {}
        creds["tokens"] = tokens.model_dump(mode="json")
        self._write(creds)

    async def get_client_info(self) -> Any:
        """Stored client registration (DCR result or pinned config), or ``None``."""
        creds = self._read()
        info = creds.get("client_info") if isinstance(creds, dict) else None
        if not isinstance(info, dict):
            return None
        try:
            from mcp.shared.auth import OAuthClientInformationFull

            return OAuthClientInformationFull.model_validate(info)
        except Exception:
            logger.debug("Stored MCP client info invalid for %s", self.credential_id, exc_info=True)
            return None

    async def set_client_info(self, client_info: Any) -> None:
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
        self.seeded_client_id = client_id


def _default_redirect_handler() -> Any:
    """Build the redirect handler: print the URL, open a browser when possible.

    The URL is hard-wrapped in brackets so trailing OAuth params can never be
    silently lost on copy (omp issue #4418).
    """

    async def redirect_handler(authorization_url: str) -> None:
        print("\nMCP OAuth authorization required. Open this URL in a browser:", file=sys.stderr)
        print(f"  <{authorization_url}>", file=sys.stderr)
        try:
            import webbrowser

            if webbrowser.open(authorization_url):
                print("(opened in your default browser)", file=sys.stderr)
        except Exception:
            pass  # headless: the paste fallback below carries the flow

    return redirect_handler


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


def _default_callback_handler() -> Any:
    """Build the callback handler: accept the pasted FULL redirect URL.

    Headless fallback: the SDK hands us control between redirect and token
    exchange, so we read the input from stdin in a worker thread (never
    blocks the event loop). The user pastes the complete redirect URL (or a
    ``code state`` pair); state and iss are parsed out and returned so the
    SDK can complete its state validation (MCP-02).
    """

    async def callback_handler() -> Any:
        from mcp.shared.auth import AuthorizationCodeResult

        def _read_input() -> str:
            return input(
                "Paste the full redirect URL (or 'code state' separated by a space): "
            ).strip()

        raw = await asyncio.to_thread(_read_input)
        code, state, iss = parse_oauth_callback_input(raw)
        return AuthorizationCodeResult(code=code, state=state, iss=iss)

    return callback_handler


def wire_oauth_auth(server_url: str, cfg: Any, store: Any = None) -> dict[str, Any]:
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
    - ``redirect_handler`` / ``callback_handler``: headless-capable defaults
      (print URL + paste the full redirect URL; see module docstring).

    The returned dict is constructed eagerly but imports ``mcp`` lazily inside
    so config-only code paths never touch the SDK.
    """
    from mcp.shared.auth import OAuthClientMetadata

    auth = getattr(cfg, "auth", None)
    oauth = getattr(cfg, "oauth", None)

    callback_port = getattr(oauth, "callback_port", None) or DEFAULT_CALLBACK_PORT
    callback_path = getattr(oauth, "callback_path", None) or DEFAULT_CALLBACK_PATH
    if not callback_path.startswith("/"):
        callback_path = f"/{callback_path}"
    redirect_uri = (
        getattr(oauth, "redirect_uri", None) or f"http://127.0.0.1:{callback_port}{callback_path}"
    )

    # Scopes: explicit `scope` on the auth block (extra-allowed field), else
    # none (the server advertises them via protected-resource metadata).
    scope = getattr(auth, "scope", None) if auth is not None else None

    client_metadata = OAuthClientMetadata(
        client_name="local-operator",
        redirect_uris=[redirect_uri],
        scope=scope,
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method=(
            "client_secret_post"
            if getattr(auth, "client_secret", None) or getattr(oauth, "client_secret", None)
            else "none"
        ),
    )

    storage = McpTokenStorage(server_url, store)

    # A configured client_id is pinned: pre-seed it so the SDK skips dynamic
    # client registration (MCP-11). DCR would mint a fresh client whose
    # redirect URI need not match what the provider registered, which breaks
    # pinned-redirect providers outright.
    client_id = getattr(auth, "client_id", None) or getattr(oauth, "client_id", None)
    if client_id:
        client_secret = getattr(auth, "client_secret", None) or getattr(
            oauth, "client_secret", None
        )
        storage.seed_client_info(client_id, client_secret)

    return {
        "server_url": server_url,
        "client_metadata": client_metadata,
        "storage": storage,
        "redirect_handler": _default_redirect_handler(),
        "callback_handler": _default_callback_handler(),
    }

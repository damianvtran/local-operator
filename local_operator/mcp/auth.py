"""MCP OAuth support on the official SDK's ``OAuthClientProvider``.

The SDK ships the whole authorization-code + PKCE + discovery machinery
(RFC 9728 protected-resource metadata, RFC 8414 auth-server metadata, RFC
7591 dynamic client registration); this module supplies the two pieces it
delegates to us: a ``TokenStorage`` implementation over stream B's credential
store, and the construction kwargs that wire a server's auth config into an
``OAuthClientProvider``.

Credential ids follow omp's URL-keyed shape, without profiles (local-operator
has no profile concept yet): ``mcp_oauth:<server_url>``.

Headless fallback: the SDK delegates the browser step to a ``redirect_handler``
and the code capture to a ``callback_handler`` — it does NOT run a loopback
HTTP server itself. Our default handlers print the full authorization URL
(hard-wrapped so trailing OAuth params are never lost on copy) and read a
pasted authorization code from stdin, which works over SSH/WSL where a
loopback browser redirect cannot. When a browser IS available the redirect
handler also calls ``webbrowser.open``; hosting a loopback listener and
wiring it as the callback handler is the natural upgrade path (see
``providers/oauth/callback_server.py`` in stream B).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Credential id prefix for managed MCP OAuth credentials (URL-keyed, like omp).
MCP_OAUTH_CREDENTIAL_PREFIX = "mcp_oauth:"

DEFAULT_CALLBACK_PORT = 3000
DEFAULT_CALLBACK_PATH = "/callback"


def mcp_oauth_credential_id(server_url: str) -> str:
    """Stable credential id for one MCP server's OAuth grant."""
    return f"{MCP_OAUTH_CREDENTIAL_PREFIX}{server_url}"


@runtime_checkable
class StructuralAuthStore(Protocol):
    """The slice of ``providers.auth_store.AuthStore`` this module consumes.

    Declared here (not imported) so MCP tests can fake the store without the
    providers stream having landed; the real ``AuthStore`` satisfies it.
    """

    def get_oauth_credential(self, provider_id: str) -> dict[str, Any] | None:
        """Return the stored OAuth credential dict for ``provider_id`` or None."""
        ...

    def upsert_oauth_credential(self, provider_id: str, creds: dict[str, Any]) -> None:
        """Insert or replace the OAuth credential dict for ``provider_id``."""
        ...


def _resolve_store(store: Any) -> Any:
    """Return ``store`` or lazily construct the real ``AuthStore``.

    The providers import is deferred: the MCP package must stay importable in
    environments where stream B has not landed or its dependencies are
    unavailable.
    """
    if store is not None:
        return store
    try:
        from local_operator.providers.auth_store import AuthStore

        return AuthStore()
    except Exception:  # pragma: no cover - depends on stream B landing
        logger.debug("providers.auth_store unavailable; MCP OAuth storage disabled", exc_info=True)
        return None


class McpTokenStorage:
    """SDK ``TokenStorage`` over an injected credential store.

    One instance per server URL: the SDK calls ``get_tokens`` / ``set_tokens``
    (and the client-info pair for dynamic registration) against this object,
    and we round-trip the pydantic models through the store's credential dicts
    under ``mcp_oauth:<server_url>``. All reads tolerate a missing store or
    missing row by returning ``None`` (the SDK then starts a fresh flow).
    """

    def __init__(self, server_url: str, store: Any = None) -> None:
        self.server_url = server_url
        self.credential_id = mcp_oauth_credential_id(server_url)
        self._store = _resolve_store(store)

    def _read(self) -> dict[str, Any] | None:
        store = self._store
        if store is None:
            return None
        try:
            creds = store.get_oauth_credential(self.credential_id)
        except Exception:
            logger.debug("MCP token read failed for %s", self.credential_id, exc_info=True)
            return None
        return creds if isinstance(creds, dict) else None

    def _write(self, creds: dict[str, Any]) -> None:
        store = self._store
        if store is None:
            return
        try:
            store.upsert_oauth_credential(self.credential_id, creds)
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
        """Stored dynamic-client registration, or ``None``."""
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


def _default_callback_handler() -> Any:
    """Build the callback handler: accept a pasted authorization code.

    Headless fallback: the SDK hands us control between redirect and token
    exchange, so we read the code from stdin in a worker thread (never blocks
    the event loop). A loopback HTTP callback server can replace this handler
    where browser redirects work; the SDK makes no assumption either way.
    """

    async def callback_handler() -> Any:
        from mcp.shared.auth import AuthorizationCodeResult

        def _read_code() -> str:
            return input("Paste the authorization code: ").strip()

        code = await asyncio.to_thread(_read_code)
        if not code:
            raise RuntimeError("No authorization code provided")
        return AuthorizationCodeResult(code=code, state=None, iss=None)

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
    - ``storage``: a :class:`McpTokenStorage` bound to ``store``;
    - ``redirect_handler`` / ``callback_handler``: headless-capable defaults
      (print URL + paste code; see module docstring).

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
    redirect_uri = getattr(oauth, "redirect_uri", None) or f"http://127.0.0.1:{callback_port}{callback_path}"

    # Scopes: explicit `scope` on the auth block (extra-allowed field), else
    # none (the server advertises them via protected-resource metadata).
    scope = getattr(auth, "scope", None) if auth is not None else None

    client_metadata = OAuthClientMetadata(
        client_name="local-operator",
        redirect_uris=[redirect_uri],
        scope=scope,
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post"
        if getattr(auth, "client_secret", None) or getattr(oauth, "client_secret", None)
        else "none",
    )

    storage = McpTokenStorage(server_url, store)
    return {
        "server_url": server_url,
        "client_metadata": client_metadata,
        "storage": storage,
        "redirect_handler": _default_redirect_handler(),
        "callback_handler": _default_callback_handler(),
    }

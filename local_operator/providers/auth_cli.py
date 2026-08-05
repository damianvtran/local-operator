"""CLI handlers for ``local-operator login / logout / login status``.

Stream E lazy-imports these; they print plain text and return exit codes.
Interactive prompts use ``input()`` — the CLI entry points are only reached
from an interactive terminal (exec/headless mode never calls them).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from local_operator.providers.oauth.callback_server import LoginCallbacks, LoginError
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    get_provider_definition,
    list_login_providers,
)


def _callbacks_interactive(definition: Any) -> LoginCallbacks:
    """print-based callbacks for terminal logins.

    The paste-code prompt is attached ONLY for providers that declare
    ``paste_code_flow`` (omp trap: for the rest it races the loopback HTTP
    callback and leaves the terminal blocked). It runs in a thread via
    ``asyncio.to_thread(input, ...)`` so the callback server keeps serving
    the browser redirect while the prompt is pending.
    """

    def on_auth_url(url: str, instructions: str | None = None) -> None:
        print(f"\nOpen this URL to authorize:\n  {url}")
        if instructions:
            print(instructions)

    def on_progress(message: str) -> None:
        print(message)

    async def on_manual_code_input() -> str | None:
        try:
            value = await asyncio.to_thread(input, "Paste the code here (empty to cancel): ")
        except (EOFError, KeyboardInterrupt):
            return None
        return value.strip() or None

    manual_input = on_manual_code_input if getattr(definition, "paste_code_flow", False) else None
    return LoginCallbacks(
        on_auth_url=on_auth_url, on_progress=on_progress, on_manual_code_input=manual_input
    )


def run_login(provider_id: str | None, credential_manager: Any, auth_store: Any) -> int:
    """Log in to ``provider_id`` (or list options when ``None``). Exit code 0/1."""
    if provider_id is None:
        print("Available login providers:")
        for definition in list_login_providers():
            marker = "*" if definition.store_credentials_as else " "
            print(f"  {marker} {definition.id:<16} {definition.name}")
        print("\nUsage: local-operator login <provider>")
        return 0

    definition = get_provider_definition(provider_id)
    if definition is None:
        print(f"Unknown provider: {provider_id}")
        return 1
    if definition.login is None:
        print(f"Provider '{provider_id}' has no interactive login.")
        return 1

    async def _run() -> Any:
        return await definition.login(_callbacks_interactive(definition))  # type: ignore[misc]

    try:
        result = asyncio.run(_run())
    except LoginError as exc:
        print(f"Login failed: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Login cancelled.")
        return 1

    storage_provider = definition.store_credentials_as or definition.id
    if isinstance(result, str):
        # Paste-an-API-key login: store as api_key credential with source login.
        if result:
            auth_store.upsert_credential(
                storage_provider, {"key": result, "source": "login", "type": "api_key"}
            )
            print(f"Stored API key for '{storage_provider}'.")
        return 0

    # OAuth credentials dict; stamp authorized_at if missing.
    result.setdefault("authorized_at", int(time.time() * 1000))
    row = auth_store.upsert_credential(storage_provider, result)
    identity = result.get("email") or result.get("account_id") or result.get("org_name") or ""
    suffix = f" ({identity})" if identity else ""
    print(f"Logged in to '{storage_provider}'{suffix}.")
    if result.get("grant_note"):
        print(f"Note: {result['grant_note']}")
    _ = row
    return 0


def run_logout(provider_id: str, auth_store: Any) -> int:
    """Remove every stored credential (OAuth + pasted keys) for the provider."""
    definition = get_provider_definition(provider_id)
    if definition is None:
        print(f"Unknown provider: {provider_id}")
        return 1
    # Log out of both the alias (e.g. xai-oauth) and its storage id (xai).
    targets = {provider_id, definition.store_credentials_as or provider_id}
    removed = 0
    for target in sorted(targets):
        removed += auth_store.delete_credentials_for_provider(target, disabled_cause="logged-out")
    if removed == 0:
        print(f"No stored credentials for '{provider_id}'.")
        return 1
    print(f"Removed {removed} credential(s) for '{provider_id}'.")
    return 0


def list_logins(auth_store: Any, credential_manager: Any = None) -> int:
    """Print one line per active credential plus env/legacy keys in the cascade."""
    rows = auth_store.list_credentials()
    if rows:
        print("Stored credentials:")
        now_ms = int(time.time() * 1000)
        for row in rows:
            identity = row.identity_key or row.data.get("email") or row.data.get("account_id") or "-"
            if row.credential_type == "oauth":
                expires = row.data.get("expires")
                state = "expired" if expires is not None and int(expires) < now_ms else "active"
                detail = f"oauth, {state}"
            else:
                source = row.data.get("source") or "stored"
                detail = f"api_key ({source})"
            print(f"  [{row.id}] {row.provider:<14} {detail:<22} identity={identity}")
    else:
        print("No stored credentials.")

    from local_operator.providers.registry import env_key_name

    print("\nEnvironment keys visible to the cascade:")
    import os

    found = False
    for definition in PROVIDER_REGISTRY:
        name = env_key_name(definition.id)
        if name and os.environ.get(name):
            print(f"  {definition.id:<14} {name}=<set>")
            found = True
    if credential_manager is not None:
        try:
            for key, secret in credential_manager.get_credentials().items():
                if secret.get_secret_value():
                    print(f"  credentials.env  {key}=<set>")
                    found = True
        except Exception:
            pass
    if not found:
        print("  (none)")
    return 0

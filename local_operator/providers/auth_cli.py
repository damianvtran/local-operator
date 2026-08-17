"""CLI handlers for ``local-operator login / logout / login status``.

Stream E lazy-imports these; they print plain text and return exit codes.
Interactive prompts use ``input()`` — the CLI entry points are only reached
from an interactive terminal (exec/headless mode never calls them).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

from local_operator.providers.oauth.callback_server import LoginCallbacks, LoginError
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    ProviderDefinition,
    env_key_name,
    get_provider_definition,
    list_login_providers,
)

if TYPE_CHECKING:  # lazy at runtime: the CLI top level must not import these
    from local_operator.credentials import CredentialManager
    from local_operator.providers.auth_store import AuthStore


def _callbacks_interactive(definition: ProviderDefinition) -> LoginCallbacks:
    """print-based callbacks for terminal logins.

    The paste prompt is attached for providers that ACCEPT one, which is two
    distinct cases and used to be read as one:

    - ``requires_paste_prompt`` \u2014 pasting is the whole login (every
      "paste your API key" provider). Gating these on ``paste_code_flow``, as
      this did, attached no prompt and made the login fail every time with
      "requires an interactive code prompt" \u2014 for eight of the eleven
      providers that offer one.
    - ``paste_code_flow`` \u2014 Anthropic's optional fallback, raced against the
      loopback callback for the case where the browser is on another machine.

    A loopback-only provider still gets NO prompt: there the prompt races the
    HTTP callback and leaves the terminal blocked on a line nobody will type.

    The prompt runs in a thread via ``asyncio.to_thread(input, ...)`` so the
    callback server keeps serving the browser redirect while it is pending.
    """

    def on_auth_url(url: str, instructions: str | None = None) -> None:
        print(f"\nOpen this URL to authorize:\n  {url}")
        if instructions:
            print(instructions)

    def on_progress(message: str) -> None:
        print(message)

    # The prompt says what it wants. "Paste the code here" is wrong for the
    # providers this now serves \u2014 they want an API key off a dashboard, and a
    # user told to paste a "code" goes looking for an OAuth code that does not
    # exist for them.
    prompt = (
        "Paste your API key here (empty to cancel): "
        if definition.paste_prompt_required
        else "Paste the code here (empty to cancel): "
    )

    async def on_manual_code_input() -> str | None:
        try:
            value = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        return value.strip() or None

    manual_input = on_manual_code_input if definition.accepts_paste_prompt else None
    return LoginCallbacks(
        on_auth_url=on_auth_url, on_progress=on_progress, on_manual_code_input=manual_input
    )


def run_login(
    provider_id: str | None,
    _credential_manager: "CredentialManager | None",
    auth_store: "AuthStore",
) -> int:
    """Log in to ``provider_id`` (or list options when ``None``). Exit code 0/1.

    The legacy credential manager is accepted for call-shape symmetry with
    :func:`list_logins` but plays no part in a login: new credentials land in
    ``auth_store`` only.
    """
    if provider_id is None:
        print("Available login providers:")
        for candidate in list_login_providers():
            marker = "*" if candidate.store_credentials_as else " "
            print(f"  {marker} {candidate.id:<16} {candidate.name}")
        print("\nUsage: local-operator login <provider>")
        return 0

    definition = get_provider_definition(provider_id)
    if definition is None:
        print(f"Unknown provider: {provider_id}")
        return 1
    if definition.login is None:
        print(f"Provider '{provider_id}' has no interactive login.")
        return 1

    login = definition.login
    callbacks = _callbacks_interactive(definition)

    async def _run() -> str | dict[str, Any]:
        return await login(callbacks)

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


def run_logout(provider_id: str, auth_store: "AuthStore") -> int:
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
    # The cascade has two tiers below the stored rows: the process
    # environment and the legacy credentials.env. Deleting the rows does not
    # clear them, so the very next turn would authenticate again with no
    # indication the logout was partial. Name the variable, never its value.
    definition_env = definition.env_keys
    env_var: str | None = None
    if isinstance(definition_env, str) and os.environ.get(definition_env):
        env_var = definition_env
    elif callable(definition_env) and definition_env():
        env_var = "the provider's environment"
    if env_var:
        print(
            f"Warning: {provider_id} still authenticates from {env_var}. "
            "Unset it or remove the credentials.env entry to complete the "
            "logout."
        )
    return 0


def list_logins(
    auth_store: "AuthStore", credential_manager: "CredentialManager | None" = None
) -> int:
    """Print one line per active credential plus env/legacy keys in the cascade."""
    rows = auth_store.list_credentials()
    if rows:
        print("Stored credentials:")
        now_ms = int(time.time() * 1000)
        for row in rows:
            identity = (
                row.identity_key or row.data.get("email") or row.data.get("account_id") or "-"
            )
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

    print("\nEnvironment keys visible to the cascade:")

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

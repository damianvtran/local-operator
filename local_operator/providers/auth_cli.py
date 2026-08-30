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
    credential_provider_id,
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

    - ``requires_paste_prompt`` — pasting is the whole login (every
      "paste your API key" provider). Gating these on ``paste_code_flow``, as
      this did, attached no prompt and made the login fail every time with
      "requires an interactive code prompt" — for eight of the eleven
      providers that offer one.
    - ``paste_code_flow`` — Anthropic's optional fallback, raced against the
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
    # providers this now serves — they want an API key off a dashboard, and a
    # user told to paste a "code" goes looking for an OAuth code that does not
    # exist for them.
    wants_api_key = definition.paste_prompt_required
    prompt = (
        "Paste your API key here (empty to cancel): "
        if wants_api_key
        else "Paste the code here (empty to cancel): "
    )

    def read_line() -> str:
        """Read the pasted value, hiding it when it is a long-lived secret.

        ``getpass`` for an API KEY, which is the discipline ``CredentialManager``
        and the web-search CLI already apply to this same class of value: a
        provider key does not expire, and ``input()`` leaves it sitting in the
        scrollback of a terminal that is frequently being screen-shared while
        someone sets a tool up. Before this change no paste-a-key provider could
        reach this prompt at all, so making them work is also what makes the
        echo reachable. The TUI half masks it for the same reason.

        A plain ``input()`` for an OAuth CODE, the other branch: it is
        single-use, expires in minutes, and is spent the moment it is redeemed,
        while being a long opaque string the user genuinely needs to SEE to
        check their paste landed whole. Hiding it would cost real legibility to
        protect a value that is not worth protecting.
        """
        if wants_api_key:
            # Imported here rather than at module scope: this module is on the
            # CLI's startup path and getpass drags in termios/tty for a prompt
            # only an interactive login reaches.
            import getpass

            return getpass.getpass(prompt)
        return input(prompt)

    async def on_manual_code_input() -> str | None:
        try:
            value = await asyncio.to_thread(read_line)
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

    storage_provider = credential_provider_id(definition.id)
    if isinstance(result, str):
        # Paste-an-API-key login: store as api_key credential with source login.
        if result:
            auth_store.upsert_credential(
                storage_provider, {"key": result, "source": "login", "type": "api_key"}
            )
            print(f"Stored API key for '{storage_provider}'.")
            _apply_login_defaults(storage_provider)
        return 0

    # OAuth credentials dict; stamp authorized_at if missing.
    result.setdefault("authorized_at", int(time.time() * 1000))
    row = auth_store.upsert_credential(storage_provider, result)
    identity = result.get("email") or result.get("account_id") or result.get("org_name") or ""
    suffix = f" ({identity})" if identity else ""
    print(f"Logged in to '{storage_provider}'{suffix}.")
    if result.get("grant_note"):
        print(f"Note: {result['grant_note']}")
    _apply_login_defaults(storage_provider)
    _ = row
    return 0


def _apply_login_defaults(provider_id: str) -> None:
    """Make a fresh login usable: adopt it as hosting when none is set.

    Before this, ``login <provider>`` stored the credential but never touched
    the config, so the very recovery the missing-key error recommends
    (``local-operator login openai``) looped straight back to "Hosting platform
    is not configured" — the credential existed but nothing pointed the app at
    it. Now, when config hosting is empty, the just-logged-in provider becomes
    the hosting and its default model becomes ``model_name``, and we print what
    was set so the change is not silent.

    When hosting is ALREADY set we touch nothing and print nothing: a user
    logging into a second provider to switch models later has not asked to
    change their default, and silently repointing it would be a surprise.

    The exception is a hosting that is set but names a provider the registry
    does not own (a typo, a hand-edited config, an id dropped by an upgrade).
    That config cannot boot, and the error it produces RECOMMENDS this command
    as the remedy — so "already set, leave it alone" would send the user round
    the same loop the paragraph above describes, just one level further in. The
    stale ``model_name`` is replaced with it, because it belonged to the
    provider being repaired and would point a real provider at a model id that
    never existed.

    Imported lazily and guarded: this is a convenience on top of a login that
    already succeeded, so a config write failure (read-only dir) must not turn a
    successful login into a failure.
    """
    try:
        from local_operator.config import ConfigManager
        from local_operator.model.defaults import default_model_for
        from local_operator.paths import config_dir

        manager = ConfigManager(config_dir())
        configured = manager.get_config_value("hosting")
        # Asked of the registry, not of a hardcoded list, so this accepts
        # exactly what the engine accepts (legacy aliases included) and cannot
        # drift from the resolver's own validation.
        repairing = bool(configured) and get_provider_definition(str(configured)) is None
        if configured and not repairing:
            return
        manager.set_config_value("hosting", provider_id)
        model = default_model_for(provider_id) or ""
        message = (
            f"Replaced unusable hosting '{configured}' with '{provider_id}'"
            if repairing
            else f"Set default hosting to '{provider_id}'"
        )
        if model and (repairing or not manager.get_config_value("model_name")):
            manager.set_config_value("model_name", model)
            message += f" and model to '{model}'"
        print(f"{message}.")
    except Exception as exc:  # noqa: BLE001 — never fail a completed login
        print(f"Note: logged in, but could not set default hosting/model: {exc}")


def run_logout(provider_id: str, auth_store: "AuthStore") -> int:
    """Remove every stored credential (OAuth + pasted keys) for the provider."""
    definition = get_provider_definition(provider_id)
    if definition is None:
        print(f"Unknown provider: {provider_id}")
        return 1
    # Log out of both the alias (e.g. xai-oauth) and its storage id (xai).
    targets = {provider_id, credential_provider_id(provider_id)}
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

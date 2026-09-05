"""Compatibility readers for legacy Radient clients, backed by AuthStore.

Legacy clients accept SecretStr and perform synchronous HTTP; credential selection
must nevertheless share provider precedence and the sole OAuth refresh store.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from pydantic import SecretStr

from local_operator.credentials import CredentialManager
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.registry import get_provider_definition


def canonical_radient_destination(base_url: str) -> bool:
    definition = get_provider_definition("radient")
    if definition is None or not definition.base_url:
        return False
    try:
        requested = urlsplit(base_url)
        canonical = urlsplit(definition.base_url)
        return (
            not requested.username
            and not requested.password
            and not requested.query
            and not requested.fragment
            and (requested.scheme, requested.netloc) == (canonical.scheme, canonical.netloc)
            # Legacy marketplace CLI methods historically join their own /v1.
            and requested.path.rstrip("/") in {"", canonical.path.rstrip("/")}
        )
    except ValueError:
        return False


async def resolve_radient_credential(
    manager: CredentialManager, base_url: str, *, store: AuthStore | None = None
) -> SecretStr:
    if not canonical_radient_destination(base_url):
        # An explicit legacy gateway must not receive a centrally signed-in
        # account's bearer. Preserve its previous dedicated key lookup instead.
        return manager.get_credential("RADIENT_API_KEY")
    owns_store = store is None
    store = store or AuthStore(manager.config_dir / "auth.db", credential_manager=manager)
    try:
        # Read-only avoids moving inference account stickiness for catalogue,
        # upload and speech helpers; a required refresh still persists centrally.
        value = await store.get_api_key("radient", read_only=True)
        if value:
            return SecretStr(value)
        # Preserve the per-key extension seam of older/custom credential
        # managers after all canonical tiers, never ahead of a central login.
        return manager.get_credential("RADIENT_API_KEY") or SecretStr("")
    finally:
        if owns_store:
            store.close()


def resolve_radient_credential_sync(manager: CredentialManager, base_url: str) -> SecretStr:
    """CLI-only bridge; async hosts must await the shared resolver directly."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve_radient_credential(manager, base_url))
    raise RuntimeError("Await resolve_radient_credential inside an async host")

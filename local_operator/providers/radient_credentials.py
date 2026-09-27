"""Compatibility readers for legacy Radient clients, backed by AuthStore.

Legacy clients accept SecretStr and perform synchronous HTTP; credential selection
must nevertheless share provider precedence and the sole OAuth refresh store.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import SecretStr

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.registry import get_provider_definition

if TYPE_CHECKING:
    from local_operator.config import ConfigManager


def configured_radient_base_url(config_manager: "ConfigManager") -> str:
    """The hub API root a ``ConfigManager``'s configuration resolves to.

    THE one read of ``config.yml``'s ``radient_base_url`` for hub consumers:
    the NESTED ``values.radient_base_url`` the config store actually holds wins
    when set (a flat document-root key is dropped by the migration and would be
    silently ignored), and :func:`resolve_radient_api_base_url` supplies the
    ``RADIENT_API_BASE_URL``/canonical default otherwise, version segment
    included. The CLI's ``_radient_hub_base_url`` and the sync coordinator's
    ``agent_sync.hub_base_url`` both delegate here rather than each spelling
    the rule — one configuration resolving two destinations is the bug the
    single-place rule exists to prevent (agent review round 1, n2; the shape is
    asserted in ``tests/unit/test_radient_hub_base_resolution.py``).
    """

    from local_operator.env import resolve_radient_api_base_url

    return resolve_radient_api_base_url(config_manager.get_config_value("radient_base_url", None))


def _radient_api_key(config_dir: Path | None) -> SecretStr:
    """The `RADIENT_API_KEY` value: the provider store row, then the environment.

    Store-first, matching every other provider-key reader. The legacy
    ``credentials.env`` rung is GONE (PR2a): a name the store does not hold and
    the environment does not export resolves to an EMPTY ``SecretStr`` rather
    than to a file the consolidation no longer reads.
    """
    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value("RADIENT_API_KEY", base=config_dir)
    if stored:
        return SecretStr(stored)
    return SecretStr(os.environ.get("RADIENT_API_KEY", ""))


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
    config_dir: Path | None, base_url: str, *, store: AuthStore | None = None
) -> SecretStr:
    if not canonical_radient_destination(base_url):
        # An explicit legacy gateway must not receive a centrally signed-in
        # account's bearer. Preserve its previous dedicated key lookup instead —
        # store-first, so a RADIENT_API_KEY saved via the store is the only value
        # this route resolves.
        return _radient_api_key(config_dir)
    owns_store = store is None
    store = store or AuthStore(
        (config_dir / "auth.db") if config_dir is not None else None, config_dir=config_dir
    )
    try:
        # Read-only avoids moving inference account stickiness for catalogue,
        # upload and speech helpers; a required refresh still persists centrally.
        value = await store.get_api_key("radient", read_only=True)
        if value:
            return SecretStr(value)
        # Preserve the per-key extension seam of older/custom credential
        # managers after all canonical tiers, never ahead of a central login.
        return _radient_api_key(config_dir) or SecretStr("")
    finally:
        if owns_store:
            store.close()


def resolve_radient_credential_sync(config_dir: Path | None, base_url: str) -> SecretStr:
    """CLI-only bridge; async hosts must await the shared resolver directly."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve_radient_credential(config_dir, base_url))
    raise RuntimeError("Await resolve_radient_credential inside an async host")

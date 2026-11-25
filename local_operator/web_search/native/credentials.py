"""Credential reads for native-search legs, mirroring ``imagegen/availability.py``.

TWO QUESTIONS, TWO SHAPES, ONE CLASS RULE. ``modes()`` asks "which credential
classes are stored?" on every status render, so the sync probes here are bare
``list_credentials`` SELECTs on the process-level store (``shared_auth_store``,
the ``_deepseek_login_present`` precedent: no new connection per probe). The
transport asks "what do I send?" once per search, through the async resolvers.
Both halves filter by the SAME credential class, so a leg the status view shows
as a key leg can never spend an OAuth grant at call time, and the reverse (the
``imagegen`` reviewer-round-1 F1 rule: ``get_oauth_access`` can rotate to a
sibling ``api_key`` row, which must never reach a subscription endpoint).

``read_only=True`` on every resolve: a search must never decide model routing --
no OAuth rotation slot consumed, no session stickiness cleared (the DeepSeek
``_resolve_deepseek_key`` contract). Every function here swallows store errors
into "no credential": a probe must never take the search chain down, and the
chain reports the missing leg as a failure rather than crashing.

No value read here is ever logged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from local_operator.providers.auth_store import OAuthAccess

logger = logging.getLogger(__name__)


def _rows(namespace: str) -> Sequence[object]:
    try:
        from local_operator.providers.auth_store import shared_auth_store

        return shared_auth_store().list_credentials(namespace)
    except Exception:  # noqa: BLE001 - an unreadable store is "no credential"
        logger.debug("native search: %s credential probe failed", namespace, exc_info=True)
        return ()


def _row_key(row: object) -> str:
    data = getattr(row, "data", None)
    key = data.get("key") if isinstance(data, dict) else None
    return key if isinstance(key, str) else ""


def has_api_key_row(namespace: str) -> bool:
    """Whether an enabled ``api_key`` row with a secret is stored under ``namespace``."""
    return any(
        getattr(row, "credential_type", None) == "api_key" and _row_key(row)
        for row in _rows(namespace)
    )


def has_oauth_row(namespace: str) -> bool:
    """Whether an enabled OAuth grant is stored under ``namespace`` (never env)."""
    return any(getattr(row, "credential_type", None) == "oauth" for row in _rows(namespace))


def env_key(config_dir: Path | None, *names: str) -> str:
    """A key from the provider store row (``LOP_PROVIDER_<name>``), then the env.

    Store-first, the order ``providers._credential`` and every other key reader
    in the tree use. An exported model key genuinely runs the call, so it
    counts (the ``imagegen`` rule), unlike a Radient login which env never makes.
    """
    from local_operator.providers.registry import provider_secret_value

    for name in names:
        stored = provider_secret_value(name, base=config_dir)
        if stored:
            return stored
    for name in names:
        exported = os.environ.get(name, "").strip()
        if exported:
            return exported
    return ""


def has_api_key(namespace: str, config_dir: Path | None, *env_names: str) -> bool:
    """Sync availability of the KEY class: a stored ``api_key`` row, or a named key."""
    return has_api_key_row(namespace) or bool(env_key(config_dir, *env_names))


async def api_key(namespace: str, config_dir: Path | None, *env_names: str) -> str:
    """The pay-as-you-go key a key leg sends: persisted ``api_key`` row, then store/env.

    ``get_persisted_api_key(kinds={"api_key"})`` is the call-time twin of
    :func:`has_api_key_row`, so availability and the request agree on one class.
    """
    try:
        from local_operator.providers.auth_store import shared_auth_store

        key = await shared_auth_store().get_persisted_api_key(namespace, None, kinds={"api_key"})
    except Exception:  # noqa: BLE001 - resolution failure is "no key"
        logger.debug("native search: %s key resolve failed", namespace, exc_info=True)
        key = None
    return key or env_key(config_dir, *env_names)


async def oauth_access(namespace: str) -> "OAuthAccess | None":
    """The OAuth grant a plan/subscription leg sends, gated to ``kind == "oauth"``.

    ``None`` when no grant can mint a bearer, or when the store's cascade fell
    back to an ``api_key`` row (which must never reach a subscription route).
    """
    try:
        from local_operator.providers.auth_store import shared_auth_store

        access = await shared_auth_store().get_oauth_access(namespace, None, read_only=True)
    except Exception:  # noqa: BLE001 - resolution failure is "no grant"
        logger.debug("native search: %s grant resolve failed", namespace, exc_info=True)
        return None
    if access is None or access.kind != "oauth" or not access.access_token:
        return None
    return access

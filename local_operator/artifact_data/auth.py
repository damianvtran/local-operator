"""The bearer for the artifact data store: lop's own ``anthropic`` OAuth grant.

The db verbs accept only a user-scoped claude.ai OAuth token, so this resolves
through the chat path's own cascade (``get_oauth_access``) — refresh, rotation
and backoff behave exactly as a chat turn's credential would — and then GATES on
``kind == "oauth"``: when no grant can mint a bearer the cascade rotates to a
sibling ``api_key`` row, which is right for chat and useless here (the server
answers 403 to it), so an API-key-only machine is told to log in instead.

``read_only=True``: a side request must never move the conversation's sticky
credential or block a row on a failed refresh (see ``AuthStore._resolve``).
Claude Code's own keychain entry is deliberately never read.
"""

from __future__ import annotations

import logging
from pathlib import Path

from local_operator.artifact_data.client import Bearer

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"


def anthropic_grant_stored(config_dir: Path | None = None) -> bool:
    """Sync, socket-free probe: is an ``anthropic`` OAuth row stored? Never raises."""
    try:
        from local_operator.providers.auth_store import shared_auth_store

        db_path = (config_dir / "auth.db") if config_dir is not None else None
        rows = shared_auth_store(db_path, config_dir=config_dir).list_credentials(PROVIDER)
        return any(getattr(row, "credential_type", None) == "oauth" for row in rows)
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("anthropic grant probe failed; reporting none", exc_info=True)
        return False


async def anthropic_bearer(session_id: str | None = None) -> Bearer | None:
    """The grant's access token plus a force-refresh for one 401 retry, or None."""
    from local_operator.providers.auth_store import shared_auth_store

    store = shared_auth_store()

    async def resolve(force: bool) -> str | None:
        try:
            access = await store.get_oauth_access(
                PROVIDER, session_id, force_refresh=force, read_only=True
            )
        except Exception:  # noqa: BLE001 - an unreadable store is "no login"
            logger.warning("anthropic oauth access read failed; reporting none")
            return None
        if access is None or access.kind != "oauth" or not access.access_token:
            return None
        return access.access_token

    token = await resolve(False)
    if token is None:
        return None

    async def refresh() -> str | None:
        return await resolve(True)

    return Bearer(token=token, refresh=refresh)

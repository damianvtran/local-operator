"""Muse credential references, bound to the owning AuthStore's config root.

Existing OAuth rows live in owner-only SQLite, not the encrypted secret store.
Muse introduces no plaintext exception: only provenance and an opaque provider
secret reference belong in auth.db. A fresh secret per write avoids replacing a
working account's key before its metadata transaction succeeds.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from local_operator.providers.oauth.meta_muse import validate_muse_key

REF_FIELD = "access_secret_ref"
REF_PREFIX = "MUSE_CODE_ACCOUNT_"
REF_PATTERN = re.compile(r"MUSE_CODE_ACCOUNT_[a-f0-9]{32}")


def seal_credential(credential: dict[str, Any], base: Path | None) -> dict[str, Any]:
    """Seal the inference key; deliberately retain no Meta account access token."""
    from local_operator.providers.registry import store_provider_key

    key = validate_muse_key(credential.get("access") or credential.get("key") or "")
    ref = REF_PREFIX + uuid.uuid4().hex
    store_provider_key(ref, key, description="Muse Code subscription inference key", base=base)
    # An allowlist, not field removal: future protocol fields must not smuggle a
    # bearer into auth.db. Full token responses are never accepted as metadata.
    from local_operator.providers.meta_muse_usage import subscription_observation

    observation = subscription_observation(credential.get("subscription_observation"))
    return {
        **({"subscription_observation": observation} if observation is not None else {}),
        "type": "oauth",
        REF_FIELD: ref,
        **{
            name: credential[name]
            for name in ("account_id", "email", "source", "billing_provenance", "authorized_at")
            if name in credential
        },
    }


def read_key(credential: dict[str, Any], base: Path | None) -> str | None:
    """Resolve only our namespace, never arbitrary operator secret references."""
    from local_operator.providers.registry import provider_secret_value

    ref = credential.get(REF_FIELD)
    if not isinstance(ref, str) or not REF_PATTERN.fullmatch(ref):
        return None
    key = provider_secret_value(ref, base=base)
    return key if key and key.startswith("LLM|") else None


@contextmanager
def reversible_key_deletion(
    credential: dict[str, Any], base: Path | None, *, restore_if: Callable[[], bool]
) -> Iterator[None]:
    """Compensate a failed logout across the two stores, without disk plaintext.

    Auth SQLite rollback cannot undo encrypted-store deletion. Keep the key in
    process memory until its metadata commit succeeds; execution/commit failure
    restores the same opaque reference. This is exception compensation, not a
    promise of crash-atomic transactions across two independent WAL databases.
    """
    from local_operator.secrets.access import open_store, session_id
    from local_operator.secrets.errors import SecretNotFound, SecretStoreError
    from local_operator.secrets.store import provider_secret_name

    ref = credential.get(REF_FIELD)
    if not isinstance(ref, str) or not REF_PATTERN.fullmatch(ref):
        yield
        return
    store = open_store(base)
    name = provider_secret_name(ref)
    try:
        record = store.describe(name, role="provider")
        value = store.get(name, role="provider", session_id=session_id())
    except SecretNotFound:
        # A previously removed key has no value to restore; logout still
        # removes its unusable metadata, never an unrelated provider secret.
        yield
        return
    try:
        store.delete(name, role="provider", session_id=session_id())
        yield
    except BaseException:
        try:
            if restore_if():
                try:
                    store.describe(name, role="provider")
                except SecretNotFound:
                    store.set(
                        name,
                        value,
                        description=record.description,
                        kind=record.kind,
                        role="provider",
                        session_id=session_id(),
                        origin=record.origin,
                    )
        except Exception as recovery_error:
            # Both stores can fail independently. Never present failed restore
            # as a compensated logout, or copy the key to a plaintext fallback.
            raise SecretStoreError(
                "Muse logout compensation incomplete: sealed key ownership or "
                "restoration failed. Sign in again to repair the account."
            ) from recovery_error
        raise


def delete_key(credential: dict[str, Any], base: Path | None) -> None:
    """Remove the sealed key on logout or rollback; missing is already removed."""
    from local_operator.secrets.access import open_store, session_id
    from local_operator.secrets.errors import SecretNotFound
    from local_operator.secrets.store import provider_secret_name

    ref = credential.get(REF_FIELD)
    if not isinstance(ref, str) or not REF_PATTERN.fullmatch(ref):
        return
    try:
        open_store(base).delete(provider_secret_name(ref), role="provider", session_id=session_id())
    except SecretNotFound:
        pass

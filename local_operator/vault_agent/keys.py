"""The delegation key: an Ed25519 seed in the local encrypted store.

WHY THE ENCRYPTED STORE AND NOT THE OS KEYCHAIN. VAULT.md §4 (amendment
2026-11-07, gap 5) puts the seed in "the harness's local encrypted secret store,
in a namespace no agent-class read can reach, never in an OS keychain": the
background processes that resolve (a daemon, the connector) cannot rely on an
unlocked keychain, and a keychain prompt lands on the operator's screen. The
reserved name prefix (``secrets.store.DELEGATION_SECRET_PREFIX``) keeps the seed
away from the agent-class verbs (``lop secret get``, the ``secret`` tool) and
the listing surfaces. It is not isolation: the ``delegation`` role is a caller
claim inside the operator's own user boundary, so a same-uid process willing to
speak the store or broker as the operator can read it. A suspected compromise
is answered by revoking the grant in the console (Agent access).

ONE KEY PER ENDPOINT. A grant binds a public key to one Team Bridge endpoint, so
the row is keyed by the endpoint id (exact bytes; never normalised). A new
enrolment (a new endpoint) therefore gets a new key and needs a new grant, which
is the honest outcome: the old grant names an endpoint that no longer exists.

THE PUBLIC KEY IS DERIVED, NEVER STORED. Storing it separately would be a second
copy that could disagree with the seed.

Stored as 64 lowercase hex characters rather than raw bytes, so every store path
that treats a value as text (rotation, ``lop secret list``'s metadata, a damaged
row's repair) sees ASCII.
"""

from __future__ import annotations

import re
from pathlib import Path

from local_operator.vault_agent import wire

_SEED_HEX = re.compile(r"^[0-9a-f]{64}$")
_ROLE = "delegation"


def delegation_secret_name(endpoint_id: str) -> str:
    """The store name for ``endpoint_id``'s delegation seed."""

    from local_operator.secrets.store import DELEGATION_SECRET_PREFIX

    if not endpoint_id or endpoint_id != endpoint_id.strip():
        raise ValueError("endpoint_id must be a non-empty, unpadded id")
    return DELEGATION_SECRET_PREFIX + endpoint_id


def public_key_from_seed(seed: bytes) -> bytes:
    """The raw 32-byte Ed25519 public key for ``seed``."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.from_private_bytes(seed)
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def sign_proof(seed: bytes, message: bytes) -> str:
    """Pure RFC 8032 Ed25519 over ``message`` (no prehash, no context), b64url.

    The ``X-Vault-Agent-Proof`` wire form: unpadded URL-safe base64 of the 64-byte
    signature (VAULT.md §4.1 step 4).
    """

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return wire.b64url(Ed25519PrivateKey.from_private_bytes(seed).sign(message))


def fingerprint_for_seed(seed: bytes) -> str:
    return wire.key_fingerprint(public_key_from_seed(seed))


def _decode(raw: bytes) -> bytes:
    text = raw.decode("ascii", "strict").strip()
    if not _SEED_HEX.fullmatch(text):
        from local_operator.secrets.errors import SecretCorrupt

        raise SecretCorrupt("The Vault delegation key row is not a 32-byte seed.")
    return bytes.fromhex(text)


def load_seed(endpoint_id: str, base: Path | None = None) -> bytes | None:
    """The seed for ``endpoint_id``, or ``None`` when none exists. Never creates.

    The existence guard mirrors ``providers.registry.provider_secret_value``: a
    READ must not create the store directory or spawn a broker on a machine that
    never stored anything.
    """

    from local_operator.secrets.access import retrieve_secret
    from local_operator.secrets.errors import SecretNotFound
    from local_operator.secrets.keys import store_path

    if not store_path(base).exists():
        return None
    try:
        raw = retrieve_secret(delegation_secret_name(endpoint_id), base, role=_ROLE)
    except SecretNotFound:
        return None
    return _decode(raw)


def load_or_create_seed(endpoint_id: str, base: Path | None = None) -> tuple[bytes, bool]:
    """``(seed, created)``: the existing seed, else a fresh one stored now.

    A concurrent creator is resolved by the store's refuse-to-overwrite ``set``:
    the loser re-reads the winner's seed instead of replacing it (a replaced seed
    would silently orphan a grant the person just approved).
    """

    import os

    from local_operator.secrets.access import open_store
    from local_operator.secrets.errors import SecretExists

    existing = load_seed(endpoint_id, base)
    if existing is not None:
        return existing, False
    seed = os.urandom(32)
    store = open_store(base, create=True)
    try:
        store.set(
            delegation_secret_name(endpoint_id),
            seed.hex().encode("ascii"),
            description="Radient Vault agent delegation key (managed by `lop vault`).",
            role=_ROLE,
        )
    except SecretExists:
        again = load_seed(endpoint_id, base)
        if again is None:  # pragma: no cover - deleted between the two calls
            raise
        return again, False
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return seed, True


def delete_all_seeds(base: Path | None = None) -> int:
    """Delete EVERY delegation seed on this machine; return how many were removed.

    Not just the current endpoint's: a re-enrolment (a new ``bep_``) or a torn
    down Team Bridge leaves the earlier seeds behind, and ``lop secret rm`` is
    agent-class, so the store refuses it on this prefix by design. That makes
    ``lop vault disconnect`` the only cleanup, so it must remove all of them, or
    it would report success while a private key that can still sign remains.
    Names come from ``list`` (no value is read).
    """

    from local_operator.secrets.access import open_store
    from local_operator.secrets.errors import SecretNotFound
    from local_operator.secrets.keys import store_path
    from local_operator.secrets.store import DELEGATION_SECRET_PREFIX

    if not store_path(base).exists():
        return 0
    store = open_store(base)
    removed = 0
    try:
        for record in store.list():
            if not record.name.startswith(DELEGATION_SECRET_PREFIX):
                continue
            try:
                store.delete(record.name, role=_ROLE)
            except SecretNotFound:
                continue
            removed += 1
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return removed

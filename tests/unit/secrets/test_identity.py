"""The identity surface: identify a secret WITHOUT reading it.

PR A-prime added two things an operator or an agent can use to answer "which
secret is this?" and one it can use to ask "show me the bytes":

* :func:`local_operator.secrets.crypto.value_fingerprint` — a keyed, truncated
  digest of a VALUE. The crypto assertions live here because the property they
  defend is about the primitive (keyed, not a bare hash; truncated; stable), not
  about the CLI that prints it.
* :meth:`SecretStore.describe_fingerprint` — metadata, a byte length and that
  digest, never the value, with its own audit event.
* :meth:`SecretStore.reveal` / :meth:`SecretStore.note_reveal_refusal` — the
  audited, human-consent path to the bytes, recorded as ``reveal`` so it is not
  lost among the ``get`` rows scripts write.

The CLI half (the flags, the prompt, the refusal's exit code and empty stdout)
is exercised end to end on a real subprocess in ``test_cli.py``; this file drives
the library directly, where a failing assertion names the primitive rather than
the plumbing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from local_operator.secrets.crypto import (
    VALUE_FINGERPRINT_BYTES,
    derive_name_index_key,
    derive_value_fingerprint_key,
    value_fingerprint,
)
from local_operator.secrets.errors import InvalidSecretName
from local_operator.secrets.store import SecretStore

VALUE = b"alpha-synthetic-token-0001"


# --- the primitive -----------------------------------------------------------


def test_value_fingerprint_is_stable_keyed_and_truncated() -> None:
    """Deterministic within a key, unguessable without one, and bounded in size.

    The three claims the credentials guide makes about a fingerprint, asserted
    one at a time: two calls agree (so it can be compared across sessions), a
    DIFFERENT master key changes it (so it is not a bare hash of the value), and
    the tag is truncated to the documented width.
    """
    key = b"k" * 32
    other_key = b"j" * 32

    first = value_fingerprint(key, VALUE)
    assert first == value_fingerprint(key, VALUE), "not deterministic"
    assert first != value_fingerprint(key, VALUE + b"!"), "equal for different values"
    assert first != value_fingerprint(
        other_key, VALUE
    ), "two master keys produced the same fingerprint: the digest is not keyed"

    algorithm, _, hex_digest = first.partition(":")
    assert algorithm == "hmac-sha256"
    assert len(hex_digest) == VALUE_FINGERPRINT_BYTES * 2
    bytes.fromhex(hex_digest)  # a non-hex rendering would raise here


def test_value_fingerprint_is_not_a_hash_of_the_value() -> None:
    """A bare digest is a SEARCH; this must not be one.

    ``sha256(value)`` of a low-entropy secret is recoverable by hashing guesses,
    and recomputable by anyone holding a copy of the value from anywhere else.
    The fingerprint must contain neither that digest nor a prefix of it.
    """
    key = b"k" * 32
    fingerprint = value_fingerprint(key, VALUE)
    bare = hashlib.sha256(VALUE).hexdigest()
    assert bare not in fingerprint
    assert fingerprint != f"hmac-sha256:{bare}"


def test_the_fingerprint_key_is_domain_separated_from_the_other_subkeys() -> None:
    """HKDF with its own ``info`` label, not the master key and not the index key.

    If the fingerprint key were the master key (or the blind-index key), a
    fingerprint would become an oracle about material other parts of the store
    are built from, and the separation the rest of ``crypto`` maintains would
    quietly stop holding for this one surface.
    """
    master_key = b"m" * 32
    fingerprint_key = derive_value_fingerprint_key(master_key)
    assert fingerprint_key != master_key
    assert fingerprint_key != derive_name_index_key(master_key)
    assert len(fingerprint_key) == 32


# --- the store methods -------------------------------------------------------


def test_describe_fingerprint_returns_length_and_digest_never_the_value(
    store: SecretStore, master_key: bytes
) -> None:
    store.set("TOKEN", VALUE)
    record, length, fingerprint = store.describe_fingerprint("TOKEN")

    assert record.name == "TOKEN"
    assert length == len(VALUE)
    assert fingerprint == value_fingerprint(master_key, VALUE)
    # The identity is the digest, and it carries no bytes of the value: not the
    # value itself, not a hex or base64 rendering of it.
    assert VALUE not in fingerprint.encode()
    assert VALUE.hex() not in fingerprint


def test_equal_values_fingerprint_equal_and_equal_lengths_do_not(store: SecretStore) -> None:
    """What a fingerprint is FOR: telling two secrets apart by value identity.

    Two names holding the same bytes must agree (that is what makes a
    comparison meaningful), and two same-length values must not — a length alone
    cannot distinguish them, which is exactly why the digest exists.
    """
    store.set("TOKEN_A", VALUE)
    store.set("TOKEN_B", VALUE)
    store.set("TOKEN_C", b"alpha-synthetic-token-0002")

    _, _, first = store.describe_fingerprint("TOKEN_A")
    _, other_length, second = store.describe_fingerprint("TOKEN_B")
    _, same_length, third = store.describe_fingerprint("TOKEN_C")

    assert first == second
    assert same_length == other_length == len(VALUE)
    assert first != third


def test_describe_fingerprint_audits_itself_and_moves_no_use_timestamp(
    store: SecretStore,
) -> None:
    """A value-bearing read is recorded; an identity check is not a "use".

    ``last_used_at`` is the field an operator watches for unexpected access, and
    a fingerprint neither hands the value to anybody nor is a retrieval. It gets
    its own event instead, so the chain can distinguish "someone read this" from
    "someone identified this" rather than conflating them.
    """
    store.set("TOKEN", VALUE)
    before = store.describe("TOKEN").last_used_at

    store.describe_fingerprint("TOKEN")

    entries = store.audit_entries()
    assert [(entry.event, entry.outcome) for entry in entries][-1] == ("describe", "fingerprint")
    assert store.describe("TOKEN").last_used_at == before
    ok, _, _ = store.verify_audit()
    assert ok


def test_a_plain_describe_still_writes_no_audit_row(store: SecretStore) -> None:
    """The metadata-only path is untouched, which is what "unchanged" means.

    Existing verifiers and tests read this trail; a `describe` that started
    appending rows would change what a suite that counts events sees, for no
    gain — no value field is used on that path.
    """
    store.set("TOKEN", VALUE)
    store.describe("TOKEN")
    store.list()
    assert [entry.event for entry in store.audit_entries()] == ["set"]


def test_reveal_is_recorded_as_a_reveal_and_not_as_a_get(store: SecretStore) -> None:
    """The whole point of the audited opt-in: the row says a human saw the bytes."""
    store.set("TOKEN", VALUE)
    assert store.reveal("TOKEN") == VALUE
    assert [(entry.event, entry.outcome) for entry in store.audit_entries()] == [
        ("set", "ok"),
        ("reveal", "tty"),
    ]


def test_a_refused_reveal_is_recorded_without_touching_any_record(store: SecretStore) -> None:
    """A refusal is evidence too, and it names no record by design.

    Resolving the targeted name would be a decrypt of that record's payload —
    a value read in the one path whose purpose is that no value was read — so the
    row carries the pid and session and leaves ``secret_id`` NULL.
    """
    store.set("TOKEN", VALUE)
    store.note_reveal_refusal(outcome="refused")

    entry = store.audit_entries()[-1]
    assert (entry.event, entry.outcome) == ("reveal", "refused")
    assert entry.secret_id is None


def test_the_identity_surface_keeps_the_provider_namespace_boundary(config_root: Path) -> None:
    """Both new methods are agent surfaces, so both refuse a provider row.

    A fingerprint of a provider key would let an agent confirm a guessed key
    without being able to read it, and a reveal would simply hand it over; the
    prefix rule holds on the READ of a value and on the read of its identity, as
    it does on ``get`` and ``describe``.

    The row is written under its own base rather than the ``store`` fixture's:
    a provider row goes through the registry's disk-key path, and that fixture's
    key is generated in memory and never installed.
    """
    from local_operator.providers.registry import store_provider_key
    from local_operator.secrets.access import open_store

    base = config_root / "provider-base"
    store_provider_key("OPENROUTER_API_KEY", "sk-secret", base=base)
    provider_store = open_store(base)

    with pytest.raises(InvalidSecretName):
        provider_store.describe_fingerprint("LOP_PROVIDER_OPENROUTER_API_KEY")
    with pytest.raises(InvalidSecretName):
        provider_store.reveal("LOP_PROVIDER_OPENROUTER_API_KEY")

    # The provider-side reader still gets its identity, under its own role.
    _, length, fingerprint = provider_store.describe_fingerprint(
        "LOP_PROVIDER_OPENROUTER_API_KEY", role="provider"
    )
    assert length == len("sk-secret")
    assert fingerprint.startswith("hmac-sha256:")

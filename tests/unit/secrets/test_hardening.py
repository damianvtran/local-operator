"""The opt-in passphrase tier (design §2.3), and what it does NOT claim.

The default keyfile tier ships unchanged and unprompted — the operator
rejected admin-gated and per-access-prompting stores, so ``harden`` is an
upgrade they choose, not a default they must undo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from local_operator.secrets.crypto import generate_master_key
from local_operator.secrets.errors import SecretStoreError
from local_operator.secrets.keys import (
    assert_key_of_record_invariant,
    install_staged_wrapped_key,
    key_mode,
    key_of_record_inconsistency,
    key_of_record_is_plaintext,
    key_path,
    load_master_key,
    stage_wrapped_master_key,
    staged_key_paths,
    unwrap_master_key,
    unwrap_master_key_matching,
    wrap_master_key,
    wrapped_key_path,
)
from local_operator.secrets.store import SecretStore

PASSPHRASE = "correct horse battery staple"


def test_default_mode_is_keyfile_and_needs_no_prompt(config_root: Path) -> None:
    """The no-prompt default SHIPS; hardening is opt-in."""
    load_master_key(config_root, create=True)
    assert key_mode(config_root) == "keyfile"
    assert key_path(config_root).exists()


def test_harden_removes_the_plaintext_key_from_disk(config_root: Path) -> None:
    """The point of the tier: nothing on disk decrypts the store on its own."""
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)

    assert key_mode(config_root) == "passphrase"
    assert not key_path(config_root).exists(), "the plaintext key survived hardening"
    assert wrapped_key_path(config_root).exists()


def test_wrapped_key_is_private(config_root: Path) -> None:
    load_master_key(config_root, create=True)
    wrap_master_key(config_root, load_master_key(config_root), PASSPHRASE)
    if sys.platform != "win32":
        mode = wrapped_key_path(config_root).stat().st_mode & 0o777
        assert oct(mode) == "0o600"


def test_the_right_passphrase_recovers_the_exact_key(config_root: Path) -> None:
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    assert unwrap_master_key(config_root, PASSPHRASE) == original


def test_a_wrong_passphrase_is_refused(config_root: Path) -> None:
    """Fails closed with a sentence, not a traceback: this is a human prompt."""
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    with pytest.raises(SecretStoreError, match="Wrong passphrase"):
        unwrap_master_key(config_root, "not the passphrase")


def test_tampering_with_the_kdf_parameters_fails_closed(config_root: Path) -> None:
    """The header is bound as AAD, so its cost parameters cannot be edited down."""
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    path = wrapped_key_path(config_root)
    blob = bytearray(path.read_bytes())
    blob[len(b"lopsecwrap\x00") + 2] ^= 0xFF  # flip a byte of the salt
    path.write_bytes(bytes(blob))
    with pytest.raises(SecretStoreError):
        unwrap_master_key(config_root, PASSPHRASE)


def test_a_truncated_wrapped_key_is_refused(config_root: Path) -> None:
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    path = wrapped_key_path(config_root)
    path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(SecretStoreError, match="not a valid wrapped master key"):
        unwrap_master_key(config_root, PASSPHRASE)


def test_a_future_wrapped_format_is_refused_not_guessed(config_root: Path) -> None:
    """Version skew is routine; an older runtime must never guess at a format."""
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    path = wrapped_key_path(config_root)
    blob = bytearray(path.read_bytes())
    blob[len(b"lopsecwrap\x00")] = 99
    path.write_bytes(bytes(blob))
    with pytest.raises(SecretStoreError, match="format 99"):
        unwrap_master_key(config_root, PASSPHRASE)


def test_loading_a_hardened_store_from_disk_directs_the_operator_to_unlock(
    config_root: Path,
) -> None:
    """There is no unwrapped key on disk, so the message must say what to do."""
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    with pytest.raises(SecretStoreError, match="unlock"):
        load_master_key(config_root)


# --- the key of record must match the tier (QA Q10) ---------------------------


def test_a_plaintext_key_beside_a_wrapped_one_is_not_reported_as_hardened(
    config_root: Path,
) -> None:
    """The tier is decided by what is ABSENT from disk, not by what is present.

    `key_mode` answered on the presence of the wrapped file alone, so the state
    the pre-fix `rotate` produced — both files — reported `passphrase` while the
    live key sat unwrapped beside the database. A store advertising the one
    guarantee §2.3 sells the tier on while not providing it is the worst of the
    available failures, because nothing surfaces it.
    """
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    assert key_mode(config_root) == "passphrase"

    key_path(config_root).write_bytes(original)  # what the broken rotate did
    assert key_mode(config_root) == "keyfile"
    assert "plaintext master key" in (key_of_record_inconsistency(config_root) or "")


def test_the_authorization_predicate_is_not_fooled_by_a_planted_key(
    config_root: Path,
) -> None:
    """R4-1: the display answer and the authorization answer must differ here.

    `key_mode` answers on ABSENCE, which is honest and attacker-controlled. The
    broker's ancestry gate cannot consume that: a same-uid process writes any
    bytes it likes to `master.key`, and if presence alone decided the tier it
    would switch the hardened tier's lineage requirement off — the full bypass
    R4-1 measured. So the two are separate functions, and this pins the exact
    case where they must disagree.

    The plant is junk (the attacker has no real key, which is the whole point),
    so the fingerprint the DATABASE records is what tells them apart.
    """
    original = load_master_key(config_root, create=True)
    SecretStore(original, base=config_root).initialize()
    wrap_master_key(config_root, original, PASSPHRASE)
    assert not key_of_record_is_plaintext(config_root)

    key_path(config_root).write_bytes(generate_master_key())  # the attacker's plant
    # Honest for the operator: a plaintext key IS on disk, so `status` warns and
    # `harden` can repair it.
    assert key_mode(config_root) == "keyfile"
    assert "plaintext master key" in (key_of_record_inconsistency(config_root) or "")
    # Unforgeable for the gate: that key does not open this store, so the store
    # is still hardened and lineage is still required.
    assert not key_of_record_is_plaintext(config_root)


def test_the_authorization_predicate_accepts_a_genuine_keyfile_store(
    config_root: Path,
) -> None:
    """The other half: failing closed must not break the tier that ships.

    A predicate that answered "hardened" everywhere would be safe and useless —
    the keyfile tier's ticket-only registration is the default path, and §8 is
    explicit that it is sound there. So the real key of record, and a store with
    no fingerprint row to check against yet, both answer true.
    """
    original = load_master_key(config_root, create=True)
    assert key_of_record_is_plaintext(config_root), "a store with no database yet"

    SecretStore(original, base=config_root).initialize()
    assert key_of_record_is_plaintext(config_root), "the real key of record"

    # Q10 damage: the plaintext file really is the live key, beside a stale
    # wrapped one. The gate follows the key, not the wrapper.
    wrap_master_key(config_root, original, PASSPHRASE)
    key_path(config_root).write_bytes(original)
    assert key_of_record_is_plaintext(config_root)


def test_a_healthy_store_reports_no_inconsistency(config_root: Path) -> None:
    """The warning must not cry wolf on either tier in its normal state."""
    original = load_master_key(config_root, create=True)
    assert key_of_record_inconsistency(config_root) is None
    wrap_master_key(config_root, original, PASSPHRASE)
    assert key_of_record_inconsistency(config_root) is None


def test_the_invariant_refuses_a_plaintext_key_in_the_hardened_tier(
    config_root: Path,
) -> None:
    """The post-condition every key install is checked against.

    Asserted at the moment of the offending write rather than discovered later,
    because the failure it guards is silent by nature.
    """
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    assert_key_of_record_invariant(config_root, "passphrase")  # clean: no raise

    key_path(config_root).write_bytes(original)
    with pytest.raises(SecretStoreError, match="plaintext master key"):
        assert_key_of_record_invariant(config_root, "passphrase")


def test_the_invariant_can_excuse_damage_the_caller_inherited(config_root: Path) -> None:
    """R4-2: a keyfile writer must not be blamed for a stale wrapped file.

    `rotate` on a Q10-damaged store installs the right file for its tier and
    succeeds, but the wrapped file it never wrote is still there — so the
    post-condition fired and the command exited 2 with "Internal error" after
    completing. `stale_wrapped_ok` is scoped to exactly that clause: the two
    that describe THIS write stay unconditional, because those really would be
    a bug in the caller.
    """
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)
    key_path(config_root).write_bytes(original)

    with pytest.raises(SecretStoreError, match="still claims this store is hardened"):
        assert_key_of_record_invariant(config_root, "keyfile")
    assert_key_of_record_invariant(config_root, "keyfile", stale_wrapped_ok=True)

    # The clause about this write is NOT excusable by the same flag.
    key_path(config_root).unlink()
    with pytest.raises(SecretStoreError, match="no master key was installed"):
        assert_key_of_record_invariant(config_root, "keyfile", stale_wrapped_ok=True)


def test_rewrapping_replaces_the_key_of_record_without_a_plaintext_window(
    config_root: Path,
) -> None:
    """A hardened rotation stages WRAPPED, so no unwrapped key ever touches disk."""
    original = load_master_key(config_root, create=True)
    wrap_master_key(config_root, original, PASSPHRASE)

    incoming = generate_master_key()
    staged = stage_wrapped_master_key(config_root, incoming, PASSPHRASE)
    assert not key_path(config_root).exists(), "staging wrote a plaintext key"
    assert staged.read_bytes() != incoming, "the staged key was not wrapped"
    # The staging sets are disjoint: a wrapped blob must never be collected by
    # the glob that feeds `resolve_master_key`, which fingerprints raw keys.
    assert staged not in staged_key_paths(config_root)

    install_staged_wrapped_key(config_root, staged)
    assert unwrap_master_key(config_root, PASSPHRASE) == incoming
    assert not key_path(config_root).exists()
    assert key_mode(config_root) == "passphrase"


def test_unlock_finishes_a_hardened_rotation_that_died_before_installing(
    config_root: Path,
) -> None:
    """The crash window the keyfile tier has recovered from since round 1.

    A power cut between `rotate`'s COMMIT and its install leaves the wrapped key
    of record wrapping the SUPERSEDED key while the database has moved on. The
    keyfile tier repairs this unattended by fingerprinting staged plaintext
    keys; a wrapped staged key can only be tested when the passphrase exists,
    so the equivalent repair happens at `unlock`.
    """
    original = load_master_key(config_root, create=True)
    store = SecretStore(original, base=config_root)
    store.initialize()
    store.set("API_KEY", b"hunter2")
    wrap_master_key(config_root, original, PASSPHRASE)

    incoming = generate_master_key()
    stage_wrapped_master_key(config_root, incoming, PASSPHRASE)
    store.rotate(incoming)  # committed; the process dies before installing

    # The key of record no longer opens the store...
    assert unwrap_master_key(config_root, PASSPHRASE) == original
    # ...and the recovery-aware unwrap finds the staged one and installs it.
    recovered = unwrap_master_key_matching(config_root, PASSPHRASE)
    assert recovered == incoming
    assert SecretStore(recovered, base=config_root).get("API_KEY") == b"hunter2"
    assert unwrap_master_key(config_root, PASSPHRASE) == incoming
    assert not key_path(config_root).exists()

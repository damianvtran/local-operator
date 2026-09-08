"""The opt-in passphrase tier (design §2.3), and what it does NOT claim.

The default keyfile tier ships unchanged and unprompted — the operator
rejected admin-gated and per-access-prompting stores, so ``harden`` is an
upgrade they choose, not a default they must undo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from local_operator.secrets.errors import SecretStoreError
from local_operator.secrets.keys import (
    key_mode,
    key_path,
    load_master_key,
    unwrap_master_key,
    wrap_master_key,
    wrapped_key_path,
)

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

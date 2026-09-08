"""Round-trip fidelity: what goes in is exactly what comes out.

A store that corrupts a value is worse than no store, because the failure
surfaces as an authentication error from a remote service rather than as an
error here. These cases are the ones that actually break naive
implementations: text-mode encoding, newline translation, non-ASCII, embedded
NULs and a value large enough to cross buffer boundaries.
"""

from __future__ import annotations

import sqlite3

import pytest

from local_operator.secrets.crypto import NONCE_BYTES, generate_master_key
from local_operator.secrets.errors import (
    InvalidSecretName,
    SecretExists,
    SecretNotFound,
)
from local_operator.secrets.store import SecretStore

# 100 KB of non-repeating bytes: large enough to exceed any single read buffer
# and incompressible enough that a truncation shows up as a length mismatch.
_LARGE_VALUE = bytes(range(256)) * 400


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b"simple-token", id="ascii"),
        pytest.param(b"line one\nline two\nline three", id="newlines"),
        pytest.param(
            "passphrase-\u00e9\u00e0\u00fc-\u4f60\u597d-\U0001f510".encode(), id="unicode"
        ),
        pytest.param(b"trailing-newlines\n\n\n", id="trailing-newlines"),
        pytest.param(b"\x00\x01\x02binary\xff\xfe", id="binary-with-nul"),
        pytest.param(b"-----BEGIN KEY-----\r\nabc\r\n-----END KEY-----", id="crlf"),
        pytest.param(b"", id="empty"),
        pytest.param(_LARGE_VALUE, id="100kb"),
    ],
)
def test_round_trip_returns_exact_bytes(store: SecretStore, value: bytes) -> None:
    store.set("TOKEN", value)
    assert store.get("TOKEN") == value


def test_large_value_length_is_preserved(store: SecretStore) -> None:
    store.set("BIG", _LARGE_VALUE)
    assert len(store.get("BIG")) == 102_400


def test_update_returns_the_new_value(store: SecretStore) -> None:
    store.set("TOKEN", b"first")
    store.update("TOKEN", b"second\nwith newline")
    assert store.get("TOKEN") == b"second\nwith newline"


def test_every_update_uses_a_fresh_nonce(store: SecretStore) -> None:
    """A repeated nonce under one key is the catastrophic GCM misuse.

    Storing the SAME value repeatedly is the case a buggy implementation gets
    wrong: an implementation that derived the nonce from the plaintext, or
    reused the record's stored nonce on update, produces an identical
    (nonce, ciphertext) pair here and nothing else would notice.
    """
    store.set("TOKEN", b"identical-value")
    nonces: set[bytes] = set()
    ciphertexts: set[bytes] = set()
    for _ in range(20):
        store.update("TOKEN", b"identical-value")
        with sqlite3.connect(store.path) as connection:
            nonce, ciphertext = connection.execute(
                "SELECT nonce, ciphertext FROM secrets"
            ).fetchone()
        assert len(nonce) == NONCE_BYTES
        assert bytes(nonce) not in nonces, "nonce reused across updates"
        # The ciphertext must differ too. It only does because the nonce feeds
        # the keystream, so this catches the subtler bug the nonce check alone
        # misses: a fresh value written to the column but not actually used to
        # encrypt.
        assert bytes(ciphertext) not in ciphertexts, "ciphertext repeated despite a fresh nonce"
        nonces.add(bytes(nonce))
        ciphertexts.add(bytes(ciphertext))
    assert len(nonces) == 20
    assert len(ciphertexts) == 20


def test_unicode_name_normalisation_finds_the_same_record(store: SecretStore) -> None:
    """NFC and NFD spellings of one name must not become two records.

    Without NFKC normalisation the blind index differs and ``get`` misses a
    record the operator can plainly see in ``list`` — a confusing failure, and
    on macOS a routine one, since the filesystem hands back NFD.
    """
    store.set("CAF\u00c9_TOKEN", b"value")  # NFC
    assert store.get("CAFE\u0301_TOKEN") == b"value"  # NFD
    with pytest.raises(SecretExists):
        store.set("CAFE\u0301_TOKEN", b"other")


def test_missing_secret_raises_not_found(store: SecretStore) -> None:
    with pytest.raises(SecretNotFound, match="ABSENT"):
        store.get("ABSENT")


def test_set_refuses_to_overwrite(store: SecretStore) -> None:
    store.set("TOKEN", b"original")
    with pytest.raises(SecretExists, match="already exists"):
        store.set("TOKEN", b"replacement")
    assert store.get("TOKEN") == b"original"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace-only"),
        pytest.param("has\x00nul", id="nul"),
        pytest.param("has\nnewline", id="newline"),
        pytest.param("x" * 300, id="too-long"),
    ],
)
def test_invalid_names_are_refused(store: SecretStore, name: str) -> None:
    with pytest.raises(InvalidSecretName):
        store.set(name, b"value")


def test_description_with_newline_is_refused(store: SecretStore) -> None:
    with pytest.raises(InvalidSecretName, match="newlines"):
        store.set("TOKEN", b"value", description="first line\nsecond line")


def test_delete_removes_the_record(store: SecretStore) -> None:
    store.set("TOKEN", b"value")
    store.delete("TOKEN")
    with pytest.raises(SecretNotFound):
        store.get("TOKEN")


def test_a_different_master_key_cannot_read_the_store(store: SecretStore) -> None:
    """The store is not merely obfuscated — the key is what opens it."""
    from local_operator.secrets.errors import SecretNotFound as NotFound

    store.set("TOKEN", b"value")
    impostor = SecretStore(generate_master_key(), base=store.path.parent.parent)
    # The blind index is keyed too, so a wrong key does not even find the row.
    with pytest.raises(NotFound):
        impostor.get("TOKEN")


def test_rotate_re_seals_every_record_and_preserves_values(store: SecretStore) -> None:
    store.set("ALPHA", b"first-value", description="one")
    store.set("BETA", b"second\nvalue")
    before = store.key_generation()

    new_key = generate_master_key()
    assert store.rotate(new_key) == 2

    assert store.key_generation() == before + 1
    assert store.get("ALPHA") == b"first-value"
    assert store.get("BETA") == b"second\nvalue"
    assert store.describe("ALPHA").description == "one"


def test_list_never_exposes_a_value(store: SecretStore) -> None:
    """Structural, not textual: there is no field on the record to leak."""
    store.set("TOKEN", b"super-secret-value", description="a note")
    records = store.list()
    assert [record.name for record in records] == ["TOKEN"]
    assert not hasattr(records[0], "value")
    assert "super-secret-value" not in repr(records[0])

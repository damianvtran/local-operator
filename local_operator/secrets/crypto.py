"""Sealing and opening a secret record.

The whole of the encryption story lives here so that the storage layer never
touches a primitive directly: ``store.py`` moves opaque blobs and asks this
module to turn them into values. Splitting it that way is what makes the
tamper tests in ``tests/unit/secrets`` meaningful — they corrupt the database
rows the storage layer wrote and prove this module refuses them.

Design reference: ``docs/design/secret-store.md`` §3.

What this protects against, stated honestly because overclaiming it is worse
than not having it: an attacker who reads ``store.db`` alone learns nothing,
not even which secrets exist, and an attacker who can WRITE the database
cannot make a consumer fetch the wrong value under a trusted name. It does
NOT protect against an attacker who reads the key file next to the database
in the default ``keyfile`` mode — see §9 of the design. This is not a vault.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import unicodedata
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from local_operator.secrets.errors import InvalidSecretName, SecretCorrupt

#: Domain separator opening every AAD. Its presence means a ciphertext lifted
#: out of some other AES-GCM context in this codebase cannot be replayed into
#: the secret store, because the AAD it was sealed under starts differently.
_AAD_MAGIC = b"lopsec\x00"

#: Record format version, carried in the AAD *and* in its own column. Bumped
#: only when the sealed plaintext's layout changes. Kept distinct from the
#: schema version (``store.py``): a runtime can understand the table layout
#: while not understanding a record inside it, and §13 of the design requires
#: those two skews to be reported separately.
RECORD_FORMAT_VERSION = 1

#: AES-256-GCM. 32-byte key, 12-byte nonce — the size the NIST construction is
#: specified for and the only one where a random nonce's collision bound is the
#: familiar 2^-32-at-2^48-messages figure.
KEY_BYTES = 32
NONCE_BYTES = 12

#: Truncation length of the blind index. 128 bits of an HMAC tag: collision
#: resistance far beyond a store of thousands of records, and it keeps the
#: UNIQUE index narrow. Truncating an HMAC is sound — it is the standard
#: construction for exactly this "searchable but opaque" use.
NAME_INDEX_BYTES = 16

#: Names are the operator's own labels, so the ceiling only has to stop an
#: unbounded AAD, not to be tight.
MAX_NAME_LENGTH = 256

_SUBKEY_INFO_PREFIX = b"local-operator/secret-store/record/v1/gen="
_NAME_INDEX_INFO = b"local-operator/secret-store/name-index/v1"


def normalize_name(name: str) -> str:
    """Canonicalise a secret name for lookup and for the blind index.

    NFKC, then strip: two visually identical names that differ only by Unicode
    composition must land on the same blind index, or ``get`` silently misses a
    record the operator can see in ``list``. Case is deliberately NOT folded —
    environment-variable-shaped names are conventionally upper-case and the
    operator may legitimately want ``token`` and ``TOKEN`` apart.
    """
    return unicodedata.normalize("NFKC", name).strip()


def validate_name(name: str) -> str:
    """Return the normalised name, or raise :class:`InvalidSecretName`.

    Control characters are refused because the AAD encoding below separates
    fields with ``\\x00``: a name containing one could be split differently on
    the way back, which is exactly the ambiguity the canonical encoding exists
    to prevent. Newlines are refused for the same reason ``credentials.py``
    refuses them — they corrupt every line-oriented consumer downstream.
    """
    normalized = normalize_name(name)
    if not normalized:
        raise InvalidSecretName("Secret name cannot be empty.")
    if len(normalized) > MAX_NAME_LENGTH:
        raise InvalidSecretName(
            f"Secret name is {len(normalized)} characters; the maximum is {MAX_NAME_LENGTH}."
        )
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized):
        raise InvalidSecretName("Secret name cannot contain control characters.")
    return normalized


def generate_master_key() -> bytes:
    """A fresh 256-bit master key from the OS CSPRNG."""
    return os.urandom(KEY_BYTES)


def derive_record_key(master_key: bytes, key_generation: int) -> bytes:
    """The per-generation record encryption key.

    HKDF rather than using the master key directly, so that ``rotate`` can
    increment the generation and re-seal without the old and new keys sharing
    any material. Records carry their own generation, so a partially rotated
    store stays fully readable: the reader derives the subkey the record names.
    """
    return _hkdf(master_key, _SUBKEY_INFO_PREFIX + str(key_generation).encode("ascii"))


def derive_name_index_key(master_key: bytes) -> bytes:
    """The HMAC key behind the blind index.

    Deliberately NOT per-generation. The blind index has to stay stable across
    a key rotation or every lookup would break the moment the generation moved,
    and rewriting every index during rotation would be a second thing that can
    half-finish.
    """
    return _hkdf(master_key, _NAME_INDEX_INFO)


def name_index(master_key: bytes, name: str) -> bytes:
    """Truncated HMAC over the normalised name — the searchable, opaque label.

    This is what makes ``list`` and ``get`` work without storing a single name
    in the clear. A store whose file is stolen does not tell the thief that
    ``MINERVA_PROD_DB_PASSWORD`` exists, which is itself a disclosure worth
    preventing (design §4).
    """
    digest = hmac.new(
        derive_name_index_key(master_key),
        normalize_name(name).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return digest[:NAME_INDEX_BYTES]


def _hkdf(master_key: bytes, info: bytes) -> bytes:
    # salt=None is correct here and not an oversight: the input keying material
    # is already a uniformly random 256-bit key from the CSPRNG, which is the
    # case RFC 5869 §3.1 says a salt is optional for. The salt exists to
    # extract entropy from a non-uniform secret; there is none to extract.
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=info).derive(
        master_key
    )


@dataclass(frozen=True)
class RecordMetadata:
    """The fields bound into a record's AAD.

    Every one of these is a DATABASE COLUMN, and that is the load-bearing
    property: the AAD must be reconstructible before the ciphertext is opened,
    so it can only reference data that is outside the ciphertext. See
    :func:`build_aad` for why the name arrives here as its blind index rather
    than as the label the operator typed.
    """

    record_id: str
    name_index: bytes
    kind: str
    key_generation: int
    format_version: int = RECORD_FORMAT_VERSION


def build_aad(metadata: RecordMetadata) -> bytes:
    """The canonical associated-data encoding.

    ``AAD = b"lopsec\\x00" || format_version(u8) || key_generation(u32be) ||
    record_id || \\x00 || name_index || \\x00 || kind``

    Canonical means unambiguous: fixed-width integers first, then variable
    fields separated by ``\\x00`` — a byte :func:`validate_name` excludes from
    names and which cannot occur in the fixed-length ``name_index`` digest or
    in a ``kind`` drawn from a closed set. No two distinct metadata tuples
    encode to the same bytes.

    **Deviation from design §3, and why.** The design writes the AAD over the
    cleartext ``name`` and ``sha256(description)``. Both of those live *inside*
    the ciphertext (design §4: "``name`` and ``description`` are stored
    encrypted"), which makes the construction circular for any read that does
    not already know the name — ``list`` and ``rotate`` enumerate rows and have
    no candidate name to build an AAD from, so they could never open a record.
    Binding the ``name_index`` column instead is non-circular and preserves
    every property §3 asks for:

    - The blind index is a deterministic function of the name under a key
      derived from the master key, so **renaming a record by editing the
      database still fails closed** — the AAD changes and the tag check fails.
    - ``record_id`` is still bound, so **swapping two records' ciphertexts
      still fails closed**.
    - The cleartext name and description are still authenticated, because they
      are sealed *inside* the ciphertext; ``store.py`` additionally re-derives
      the blind index from the decrypted name and rejects a mismatch, so the
      inner copy and the outer column cannot disagree.

    The description is therefore covered by the AEAD rather than by the AAD,
    which is the same guarantee reached by a different route: editing it
    requires the key either way.
    """
    return b"".join(
        (
            _AAD_MAGIC,
            metadata.format_version.to_bytes(1, "big"),
            metadata.key_generation.to_bytes(4, "big"),
            metadata.record_id.encode("utf-8"),
            b"\x00",
            metadata.name_index,
            b"\x00",
            metadata.kind.encode("utf-8"),
        )
    )


def seal(master_key: bytes, metadata: RecordMetadata, value: bytes) -> tuple[bytes, bytes]:
    """Encrypt ``value``, returning ``(nonce, ciphertext)``.

    A FRESH random nonce on every call, including every update of an existing
    record. Reusing a nonce under the same key in GCM is catastrophic — it
    leaks the XOR of the two plaintexts and the authentication key — so the
    nonce is generated here rather than accepted as a parameter, leaving no
    call site able to supply a stale one.
    """
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(derive_record_key(master_key, metadata.key_generation)).encrypt(
        nonce, value, build_aad(metadata)
    )
    return nonce, ciphertext


def open_record(
    master_key: bytes, metadata: RecordMetadata, nonce: bytes, ciphertext: bytes
) -> bytes:
    """Decrypt a record, or raise :class:`SecretCorrupt`.

    ``InvalidTag`` is translated rather than propagated so no caller has to
    import ``cryptography`` to handle the one failure that matters, and so the
    message names the tampering explicitly. There is deliberately no path that
    returns a value when authentication fails.
    """
    try:
        return AESGCM(derive_record_key(master_key, metadata.key_generation)).decrypt(
            nonce, ciphertext, build_aad(metadata)
        )
    except InvalidTag as exc:
        raise SecretCorrupt(
            f"Secret record {metadata.record_id} failed authentication: its ciphertext or "
            "its metadata (name, description, kind, key generation) was modified after it "
            "was written, or it was sealed under a different key. Refusing to return a value."
        ) from exc

"""AAD binding: editing the database must fail closed, never mislead.

The attack these defend against is specific and worth stating, because it is
what makes AAD binding non-optional rather than a nicety. An attacker who can
WRITE the database but cannot decrypt it does not need the key to do damage:
if only the value were authenticated, they could point the row labelled
``PROD_DB_PASSWORD`` at the ciphertext of ``STAGING_DB_PASSWORD``, or rename a
record they control to a name a consumer trusts. Either one makes
``$(lop secret get NAME)`` return a real, correctly-decrypting secret that is
the WRONG secret — and nothing anywhere would report an error.

Every test here corrupts the real SQLite file on disk with a plain ``UPDATE``,
exactly as such an attacker would, and asserts the store refuses. They are
written to fail if someone later narrows the AAD: each one names the field it
removes from the binding, so a "simplification" that drops ``record_id`` or
``name_index`` from :func:`crypto.build_aad` turns the corresponding test red
rather than silently widening the attack surface.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from local_operator.secrets.crypto import RecordMetadata, build_aad, name_index
from local_operator.secrets.errors import SecretCorrupt
from local_operator.secrets.store import SecretStore


def _rows(store: SecretStore) -> list[tuple[Any, ...]]:
    with sqlite3.connect(store.path) as connection:
        return connection.execute(
            "SELECT id, name_index, key_generation, format_version, nonce, ciphertext, kind"
            " FROM secrets ORDER BY id"
        ).fetchall()


def test_renaming_a_record_in_the_database_fails_closed(store: SecretStore) -> None:
    """The rename attack: relabel a low-value secret as a trusted name.

    Done the way an attacker would — rewrite the ``name_index`` column so the
    row is found under a different name — and the AEAD must reject it, because
    that column is bound into the AAD. If this test ever passes by returning a
    value, ``get TRUSTED_NAME`` has silently returned the attacker's chosen
    secret.
    """
    store.set("LOW_VALUE", b"attacker-controlled")
    trusted_index = name_index(store._master_key, "PROD_DB_PASSWORD")

    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE secrets SET name_index = ?", (trusted_index,))

    with pytest.raises(SecretCorrupt, match="failed authentication"):
        store.get("PROD_DB_PASSWORD")


def test_swapping_two_records_ciphertexts_fails_closed(store: SecretStore) -> None:
    """The swap attack: leave the labels alone, exchange the payloads.

    ``record_id`` is bound into the AAD and is immutable, so a ciphertext moved
    to another row authenticates against the wrong id and the tag check fails.
    Without that binding both rows would decrypt cleanly and each name would
    resolve to the other's secret — the single most dangerous silent failure
    this store can have.
    """
    store.set("STAGING", b"staging-password")
    store.set("PRODUCTION", b"production-password")

    first, second = _rows(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE secrets SET nonce = ?, ciphertext = ? WHERE id = ?",
            (second[4], second[5], first[0]),
        )
        connection.execute(
            "UPDATE secrets SET nonce = ?, ciphertext = ? WHERE id = ?",
            (first[4], first[5], second[0]),
        )

    for name in ("STAGING", "PRODUCTION"):
        with pytest.raises(SecretCorrupt, match="failed authentication"):
            store.get(name)


def test_swapping_whole_logical_records_fails_closed(store: SecretStore) -> None:
    """The swap that ``name_index`` alone does NOT catch — this pins ``record_id``.

    :func:`test_swapping_two_records_ciphertexts_fails_closed` above is caught
    by the ``name_index`` binding, so it stays green even if ``record_id`` is
    dropped from the AAD — verified by mutation. This case moves the index AND
    the ciphertext together, leaving only the immutable ``id`` columns in
    place, so ``record_id`` is the sole remaining binding that can detect it.
    Remove ``record_id`` from :func:`crypto.build_aad` and this test goes red
    while the other stays green, which is exactly the coverage the pair is for.

    Why it matters beyond defence in depth: ``record_id`` is what the audit
    trail records. A store where ids can be shuffled under their records is one
    where the audit log attributes a retrieval to the wrong secret.
    """
    store.set("STAGING", b"staging-password")
    store.set("PRODUCTION", b"production-password")

    first, second = _rows(store)
    with sqlite3.connect(store.path) as connection:
        # Park one row out of the way first: name_index is UNIQUE, so a direct
        # exchange would collide mid-statement.
        connection.execute(
            "UPDATE secrets SET name_index = ? WHERE id = ?", (b"\xff" * 16, first[0])
        )
        connection.execute(
            "UPDATE secrets SET name_index = ?, nonce = ?, ciphertext = ? WHERE id = ?",
            (first[1], first[4], first[5], second[0]),
        )
        connection.execute(
            "UPDATE secrets SET name_index = ?, nonce = ?, ciphertext = ? WHERE id = ?",
            (second[1], second[4], second[5], first[0]),
        )

    for name in ("STAGING", "PRODUCTION"):
        with pytest.raises(SecretCorrupt, match="failed authentication"):
            store.get(name)


def test_editing_the_kind_column_fails_closed(store: SecretStore) -> None:
    """``kind`` drives how a value is materialised, so it is bound too.

    Flipping ``string`` to ``file`` would change how ``lop secret file`` treats
    the record; it is in the AAD so it cannot be flipped without the key.
    """
    store.set("TOKEN", b"value", kind="string")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE secrets SET kind = 'file'")

    with pytest.raises(SecretCorrupt, match="failed authentication"):
        store.get("TOKEN")


def test_editing_the_key_generation_column_fails_closed(store: SecretStore) -> None:
    """Generation selects the subkey, so a forged one must not decrypt.

    It fails twice over — wrong subkey AND wrong AAD — which is intended:
    binding it means an attacker cannot force a record to be opened under a
    generation whose key they may have obtained separately.
    """
    store.set("TOKEN", b"value")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE secrets SET key_generation = 99")

    with pytest.raises(SecretCorrupt, match="failed authentication"):
        store.get("TOKEN")


def test_flipping_one_ciphertext_bit_fails_closed(store: SecretStore) -> None:
    """The baseline AEAD property: no malleability, not even one bit."""
    store.set("TOKEN", b"value")
    row = _rows(store)[0]
    corrupted = bytearray(row[5])
    corrupted[0] ^= 0x01

    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE secrets SET ciphertext = ?", (bytes(corrupted),))

    with pytest.raises(SecretCorrupt, match="failed authentication"):
        store.get("TOKEN")


def test_flipping_one_nonce_bit_fails_closed(store: SecretStore) -> None:
    store.set("TOKEN", b"value")
    row = _rows(store)[0]
    corrupted = bytearray(row[4])
    corrupted[0] ^= 0x01

    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE secrets SET nonce = ?", (bytes(corrupted),))

    with pytest.raises(SecretCorrupt, match="failed authentication"):
        store.get("TOKEN")


def test_inner_and_outer_name_copies_cannot_disagree(store: SecretStore) -> None:
    """The belt-and-braces check, exercised on its own.

    The name exists twice: sealed inside the ciphertext (so ``list`` can read
    it back) and as the blind index column (so lookups work). The AAD binds the
    column; this test proves the store ALSO rejects a record whose two copies
    disagree, which is what stops anyone holding the key from constructing such
    a record. It is built by re-sealing deliberately — the AAD is honest here,
    so only the consistency check can catch it.
    """
    import json

    from local_operator.secrets.crypto import seal

    store.set("REAL_NAME", b"value")
    row = _rows(store)[0]
    record_id, stored_index, generation, format_version, _, _, kind = row

    # A payload claiming a different name than the index it is stored under.
    payload = json.dumps(
        {"name": "DIFFERENT_NAME", "description": "", "value": b"value".hex()},
        separators=(",", ":"),
    ).encode()
    metadata = RecordMetadata(
        record_id=record_id,
        name_index=bytes(stored_index),
        kind=kind,
        key_generation=generation,
        format_version=format_version,
    )
    nonce, ciphertext = seal(store._master_key, metadata, payload)

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE secrets SET nonce = ?, ciphertext = ? WHERE id = ?",
            (nonce, ciphertext, record_id),
        )

    with pytest.raises(SecretCorrupt, match="sealed name does not match"):
        store.get("REAL_NAME")


def test_aad_actually_covers_every_bound_field() -> None:
    """A direct guard on the binding itself.

    The tests above prove the behaviour end to end, but they would all still
    pass if a future refactor kept the fields and changed their encoding in a
    way that made two different tuples collide. This asserts the property
    directly: changing any single bound field changes the AAD bytes.
    """
    base = RecordMetadata(
        record_id="11111111-1111-1111-1111-111111111111",
        name_index=b"\x01" * 16,
        kind="string",
        key_generation=1,
    )
    variants = [
        RecordMetadata(
            record_id="22222222-2222-2222-2222-222222222222",
            name_index=base.name_index,
            kind=base.kind,
            key_generation=base.key_generation,
        ),
        RecordMetadata(
            record_id=base.record_id,
            name_index=b"\x02" * 16,
            kind=base.kind,
            key_generation=base.key_generation,
        ),
        RecordMetadata(
            record_id=base.record_id,
            name_index=base.name_index,
            kind="file",
            key_generation=base.key_generation,
        ),
        RecordMetadata(
            record_id=base.record_id,
            name_index=base.name_index,
            kind=base.kind,
            key_generation=2,
        ),
        RecordMetadata(
            record_id=base.record_id,
            name_index=base.name_index,
            kind=base.kind,
            key_generation=base.key_generation,
            format_version=2,
        ),
    ]
    encodings = {build_aad(base)} | {build_aad(variant) for variant in variants}
    assert len(encodings) == len(variants) + 1, "two distinct records share an AAD encoding"


def test_deleting_a_row_does_not_affect_the_others(store: SecretStore) -> None:
    """Records are independently sealed; damage does not spread."""
    store.set("ALPHA", b"alpha-value")
    store.set("BETA", b"beta-value")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM secrets WHERE ciphertext = (SELECT ciphertext" " FROM secrets LIMIT 1)"
        )
    remaining = store.list()
    assert len(remaining) == 1
    assert store.get(remaining[0].name).endswith(b"-value")

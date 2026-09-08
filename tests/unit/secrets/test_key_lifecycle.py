"""Rotation against the ~10-session workload, and against a power cut.

Two failures live here, both of which lost or stranded real secrets and both of
which were found by execution rather than by reading:

* a ``rotate`` running concurrently with another session's write used to POISON
  that write — it reported "stored", and the row was then unreachable by name
  and undecryptable, which in turn bricked ``list``, ``status`` and every later
  ``rotate`` store-wide;
* a crash between ``rotate``'s COMMIT and the installation of the new key used
  to lose EVERY secret in the store, because the database was re-sealed under a
  key that existed only in the dead process's memory.

Neither is exotic on this machine: the operator runs about ten sessions at
once, which is exactly when a rotation overlaps somebody else's write, and a
power cut needs no adversary at all.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
from pathlib import Path

import pytest

from local_operator.secrets.access import resolve_master_key
from local_operator.secrets.crypto import generate_master_key, key_fingerprint
from local_operator.secrets.errors import SecretCorrupt, SecretStoreError, StaleKeyEpoch
from local_operator.secrets.keys import (
    key_path,
    load_master_key,
    replace_master_key,
    stage_master_key,
    staged_key_path,
)
from local_operator.secrets.store import SecretStore

# --- R1: a concurrent write must never be orphaned by a rotation -------------


def test_a_write_under_a_pre_rotation_key_is_refused_not_poisoned(
    config_root: Path, master_key: bytes
) -> None:
    """The store-bricking interleave, at its smallest.

    Session A loads the key and is still holding it when session B rotates.
    A's next write must FAIL — a write that succeeds here is sealed under a key
    that no longer exists and indexed under an index key that no longer
    matches, so it is unreachable and undecryptable while having reported
    success. Failing closed costs a retry; succeeding costs the secret.
    """
    session_a = SecretStore(master_key, base=config_root)
    session_a.initialize()
    session_a.set("EXISTING", b"v0")

    new_key = generate_master_key()
    SecretStore(master_key, base=config_root).rotate(new_key)

    with pytest.raises(StaleKeyEpoch):
        session_a.set("POISONED", b"secret-value")

    # Nothing was committed, and the store is entirely healthy afterwards.
    rotated = SecretStore(new_key, base=config_root)
    assert [record.name for record in rotated.list()] == ["EXISTING"]
    assert rotated.damaged_records() == []
    assert rotated.get("EXISTING") == b"v0"


@pytest.mark.parametrize("verb", ["set", "update", "delete", "rotate"])
def test_every_write_verb_checks_the_key_epoch(
    config_root: Path, master_key: bytes, verb: str
) -> None:
    """Not just ``set``: any write under a superseded key corrupts the store."""
    stale = SecretStore(master_key, base=config_root)
    stale.initialize()
    stale.set("EXISTING", b"v0")
    SecretStore(master_key, base=config_root).rotate(generate_master_key())

    calls = {
        "set": lambda: stale.set("NEW", b"v"),
        "update": lambda: stale.update("EXISTING", b"v1"),
        "delete": lambda: stale.delete("EXISTING"),
        "rotate": lambda: stale.rotate(generate_master_key()),
    }
    with pytest.raises(StaleKeyEpoch):
        calls[verb]()


def test_ten_sessions_writing_through_a_rotation_never_brick_the_store(
    config_root: Path, master_key: bytes
) -> None:
    """The operator's real shape: nine writers, one rotator, no delays.

    Every write either succeeds or raises :class:`StaleKeyEpoch`. What must not
    happen is a write that reports success and leaves a row nobody can read —
    so the assertion is not "no errors", it is that the store is fully
    enumerable and every surviving record decrypts afterwards.
    """
    seed = SecretStore(master_key, base=config_root)
    seed.initialize()
    seed.set("ANCHOR", b"anchor-value")

    current_key = master_key
    stored: list[str] = []
    for round_index in range(9):
        stale_session = SecretStore(current_key, base=config_root)
        if round_index % 3 == 2:
            next_key = generate_master_key()
            SecretStore(current_key, base=config_root).rotate(next_key)
            current_key = next_key
        try:
            stale_session.set(f"SECRET_{round_index}", f"value-{round_index}".encode())
            stored.append(f"SECRET_{round_index}")
        except StaleKeyEpoch:
            pass

    final = SecretStore(current_key, base=config_root)
    assert final.damaged_records() == [], "a write was orphaned by a rotation"
    assert sorted(record.name for record in final.list()) == sorted(["ANCHOR", *stored])
    for name in stored:
        assert final.get(name) == f"value-{name.rsplit('_', 1)[1]}".encode()
    assert final.get("ANCHOR") == b"anchor-value"


# --- R1, second half: one bad row must not take down enumeration -------------


def _damage_one_row(store: SecretStore, name: str) -> str:
    """Corrupt a single record's ciphertext; returns its id."""
    index = store.describe(name).record_id
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE secrets SET ciphertext = ? WHERE id = ?", (b"\x00" * 64, index))
    return index


def test_list_survives_a_record_it_cannot_decrypt(store: SecretStore) -> None:
    """One unreadable row must never make the whole store unusable.

    ``list`` used to propagate, so a single damaged record took down ``list``,
    ``status`` and every subsequent ``rotate`` — removing the operator's only
    tools for seeing what survived, at the moment they needed them most.
    """
    store.set("GOOD_ONE", b"a")
    store.set("BROKEN", b"b")
    store.set("GOOD_TWO", b"c")
    damaged_id = _damage_one_row(store, "BROKEN")

    assert [record.name for record in store.list()] == ["GOOD_ONE", "GOOD_TWO"]
    assert store.damaged_records() == [damaged_id]
    # The healthy records are still fully usable.
    assert store.get("GOOD_ONE") == b"a"
    assert store.get("GOOD_TWO") == b"c"


def test_a_damaged_record_can_be_removed_by_id(store: SecretStore) -> None:
    """The repair path. ``rm NAME`` cannot reach it — its name is unreadable."""
    store.set("GOOD_ONE", b"a")
    store.set("BROKEN", b"b")
    damaged_id = _damage_one_row(store, "BROKEN")

    assert store.delete_record_id(damaged_id) is True
    assert store.damaged_records() == []
    assert [record.name for record in store.list()] == ["GOOD_ONE"]
    assert store.delete_record_id(damaged_id) is False


def test_rotate_moves_the_healthy_records_past_a_damaged_one(store: SecretStore) -> None:
    """A damaged row must not block the response to a suspected key compromise."""
    store.set("GOOD_ONE", b"a")
    store.set("BROKEN", b"b")
    damaged_id = _damage_one_row(store, "BROKEN")

    new_key = generate_master_key()
    assert store.rotate(new_key) == 1  # only the readable record moved

    rotated = SecretStore(new_key, base=store._base)
    assert rotated.get("GOOD_ONE") == b"a"
    assert rotated.damaged_records() == [damaged_id]


# --- R2: a crash at any step of a rotation must leave the store openable -----


def _rotate_to_step(base: Path, key: bytes, new_key: bytes, step: int) -> None:
    """Run ``handlers._rotate``'s sequence and stop after ``step``.

    Mirrors the handler exactly: stage the new key, commit the re-seal, install
    the key. Stopping partway is the power cut.
    """
    if step >= 1:
        stage_master_key(base, new_key)
    if step >= 2:
        SecretStore(key, base=base).rotate(new_key)
    if step >= 3:
        replace_master_key(base, new_key)


@pytest.mark.parametrize("crash_after_step", [0, 1, 2, 3])
def test_a_crash_at_any_rotation_step_leaves_the_store_openable(
    config_root: Path, crash_after_step: int
) -> None:
    """At EVERY instant of a rotation, some key on disk opens the store.

    Step 2 is the one that used to be fatal: the database was committed under
    the new key while the only copy of that key was in the dying process's
    memory, so an ordinary power cut destroyed every secret irrecoverably —
    while the code's docstring claimed the case was recoverable.
    """
    key = load_master_key(config_root, create=True)
    store = SecretStore(key, base=config_root)
    store.initialize()
    for index in range(3):
        store.set(f"KEY_{index}", f"value-{index}".encode())

    _rotate_to_step(config_root, key, generate_master_key(), crash_after_step)

    # A brand new process opens the store, knowing only what is on disk.
    survivor = SecretStore(resolve_master_key(config_root), base=config_root)
    assert sorted(record.name for record in survivor.list()) == ["KEY_0", "KEY_1", "KEY_2"]
    for index in range(3):
        assert survivor.get(f"KEY_{index}") == f"value-{index}".encode()


def test_an_interrupted_rotation_completes_the_key_install(config_root: Path) -> None:
    """Recovery is finished, not merely tolerated.

    Leaving the store readable-but-unrepaired would mean the next crash finds
    the same half-done state, so ``resolve_master_key`` installs the staged key
    and clears the staging file.
    """
    key = load_master_key(config_root, create=True)
    store = SecretStore(key, base=config_root)
    store.initialize()
    store.set("ALPHA", b"alpha-value")

    new_key = generate_master_key()
    _rotate_to_step(config_root, key, new_key, 2)  # crash before the install
    assert staged_key_path(config_root).exists()
    assert key_path(config_root).read_bytes() == key  # still the OLD key

    assert resolve_master_key(config_root) == new_key
    assert key_path(config_root).read_bytes() == new_key
    assert not staged_key_path(config_root).exists(), "the staged key was not cleared"


def test_a_stale_staged_key_is_never_adopted(config_root: Path) -> None:
    """A leftover staging file must not replace a perfectly good key.

    The staged key is adopted only when its fingerprint matches the one the
    DATABASE records, so an abandoned rotation's file is inert rather than
    destructive.
    """
    key = load_master_key(config_root, create=True)
    store = SecretStore(key, base=config_root)
    store.initialize()
    store.set("ALPHA", b"alpha-value")

    # A rotation that died BEFORE committing: staged, database untouched.
    stage_master_key(config_root, generate_master_key())

    assert resolve_master_key(config_root) == key
    assert key_path(config_root).read_bytes() == key
    assert SecretStore(resolve_master_key(config_root), base=config_root).get("ALPHA") == (
        b"alpha-value"
    )


def test_concurrent_key_installs_do_not_consume_each_others_temp_file(
    config_root: Path,
) -> None:
    """``replace_master_key`` is a concurrent path now, so its temp is per-pid.

    Recovery means several sessions can complete the same interrupted rotation
    at once. With a FIXED temporary name, one process's ``os.replace`` consumed
    the file another had just written, which surfaced under load as a spurious
    ``FileNotFoundError`` from an install that had actually succeeded.
    """
    key = load_master_key(config_root, create=True)
    store = SecretStore(key, base=config_root)
    store.initialize()
    store.set("ALPHA", b"alpha-value")

    new_key = generate_master_key()
    _rotate_to_step(config_root, key, new_key, 2)  # committed, not installed

    # Several sessions open the store simultaneously; each completes the
    # install. None may fail, and all must agree on the resulting key.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        resolved = [
            future.result()
            for future in [pool.submit(resolve_master_key, config_root) for _ in range(8)]
        ]

    assert resolved == [new_key] * 8
    assert key_path(config_root).read_bytes() == new_key
    assert SecretStore(new_key, base=config_root).get("ALPHA") == b"alpha-value"
    # No temporary files left behind in the secrets directory.
    leftovers = sorted(p.name for p in key_path(config_root).parent.glob("master.key.*"))
    assert leftovers == [], f"temporary key files were left behind: {leftovers}"


def test_the_fingerprint_identifies_the_key_without_revealing_it(master_key: bytes) -> None:
    """It is written to the database in the clear, so it must not be the key."""
    fingerprint = key_fingerprint(master_key)
    assert fingerprint != master_key[: len(fingerprint)]
    assert fingerprint not in master_key
    assert key_fingerprint(master_key) == fingerprint  # deterministic
    assert key_fingerprint(generate_master_key()) != fingerprint


def test_a_corrupt_meta_row_is_a_clean_error_not_a_traceback(store: SecretStore) -> None:
    """Q2: ``meta.v`` holding TEXT escaped as a raw ``TypeError``.

    Only reachable by hand-corrupting the database, and it never leaked a
    value — but it bypassed the CLI's deliberate one-line error contract and
    printed a 29-line traceback into a command run inside ``$( )``.
    """
    store.set("ALPHA", b"a")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE meta SET v = 'not-a-number' WHERE k = 'schema_version'")

    with pytest.raises(SecretStoreError):
        store.get("ALPHA")


def test_a_damaged_record_is_reported_as_corrupt_by_get(store: SecretStore) -> None:
    """Enumeration steps over damage; a direct ``get`` still fails closed."""
    store.set("BROKEN", b"b")
    _damage_one_row(store, "BROKEN")
    with pytest.raises(SecretCorrupt):
        store.get("BROKEN")

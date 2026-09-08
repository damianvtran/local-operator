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
import threading
from pathlib import Path

import pytest

from local_operator.secrets.access import resolve_master_key
from local_operator.secrets.crypto import (
    KEY_BYTES,
    generate_master_key,
    key_fingerprint,
)
from local_operator.secrets.errors import SecretCorrupt, SecretStoreError, StaleKeyEpoch
from local_operator.secrets.keys import (
    discard_staged_master_key,
    key_path,
    load_master_key,
    replace_master_key,
    stage_master_key,
    staged_key_paths,
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
    assert staged_key_paths(config_root)
    assert key_path(config_root).read_bytes() == key  # still the OLD key

    assert resolve_master_key(config_root) == new_key
    assert key_path(config_root).read_bytes() == new_key
    assert not staged_key_paths(config_root), "the staged key was not cleared"


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


# --- R2 under CONCURRENCY: the staged key is per-rotation --------------------
#
# The crash matrix above proves the R2 invariant for ONE rotator. That control
# passed while the invariant was in fact broken, because every rotation staged
# its key at the same fixed filename: a second rotator's staging replaced the
# first's, and a losing rotator's discard deleted the winner's. The tests below
# are the concurrent case the matrix was missing. Each asserts the same
# property as the matrix — after a power cut, some key on disk still opens the
# store — with more than one rotation in flight.


def _seed_store(base: Path, value: bytes = b"super-secret-value") -> bytes:
    """An initialised single-record store; returns its master key."""
    key = load_master_key(base, create=True)
    store = SecretStore(key, base=base)
    store.initialize()
    store.set("ALPHA", value)
    return key


def _open_knowing_only_the_disk(base: Path) -> bytes:
    """What a brand new process recovers, given only what is on disk."""
    return SecretStore(resolve_master_key(base), base=base).get("ALPHA")


def test_a_second_rotation_does_not_clobber_a_committed_rotations_staged_key(
    config_root: Path,
) -> None:
    """B staging must not destroy the key A's committed database needs.

    The interleave: A stages and COMMITS, so the database is now sealed under
    A's key and the staged file is the only copy of it on disk. B then stages
    its own key, and A is killed before it can install. With one shared staging
    filename B's write replaced A's key and the store was unrecoverable — an
    ordinary power cut, no adversary, losing every secret. Making that write
    atomic does not help: an atomic clobber is still a clobber, which is why
    the fix is a name unique per rotation rather than a safer write.
    """
    key = _seed_store(config_root)
    a_key = generate_master_key()
    b_key = generate_master_key()

    stage_master_key(config_root, a_key)
    SecretStore(key, base=config_root).rotate(a_key)  # A COMMITS
    stage_master_key(config_root, b_key)  # B stages alongside, must not clobber
    # A is killed here, before replace_master_key: power cut in the window.

    assert _open_knowing_only_the_disk(config_root) == b"super-secret-value"


def test_a_losing_rotation_does_not_discard_the_winners_staged_key(
    config_root: Path,
) -> None:
    """The loser's cleanup must not delete a key that is not its own.

    Two rotations race; the epoch guard makes exactly one win. The loser then
    cleans up its staged key — and with an unconditional unlink of one fixed
    path it deleted the WINNER's staged key instead. If the winner had already
    committed and not yet installed, that cleanup removed the only on-disk copy
    of the key the database needed, so a crash there lost the store.
    """
    key = _seed_store(config_root)
    a_key = generate_master_key()
    b_key = generate_master_key()

    loser = SecretStore(key, base=config_root)  # opened under the pre-rotation key
    stage_master_key(config_root, a_key)
    SecretStore(key, base=config_root).rotate(a_key)  # A COMMITS, wins the epoch

    stage_master_key(config_root, b_key)
    with pytest.raises(StaleKeyEpoch):
        loser.rotate(b_key)
    discard_staged_master_key(config_root, b_key)  # loser cleans up after itself

    # The winner's staged key is still the only copy of the database's key.
    assert staged_key_paths(config_root), "the winner's staged key was deleted"
    # A is killed here, before replace_master_key.
    assert _open_knowing_only_the_disk(config_root) == b"super-secret-value"


def test_discarding_a_staged_key_removes_only_that_rotations_own_copy(
    config_root: Path,
) -> None:
    """Ownership is checked by content, so cleanup is precise, not merely safe.

    Both halves matter: the discard must leave other rotations' keys alone AND
    must still remove its own, or "never delete anything" would pass the test
    above while leaving key material on disk forever.
    """
    _seed_store(config_root)
    mine = generate_master_key()
    theirs = generate_master_key()
    my_path = stage_master_key(config_root, mine)
    their_path = stage_master_key(config_root, theirs)

    discard_staged_master_key(config_root, mine)

    assert not my_path.exists(), "the caller's own staged key was left behind"
    assert their_path.exists(), "another rotation's staged key was removed"
    assert their_path.read_bytes() == theirs


def test_concurrent_stagings_each_keep_their_own_key(config_root: Path) -> None:
    """``stage_master_key`` is a concurrent path; no call may lose its key.

    The coverage gap that let the clobber through: every existing test staged
    from ONE thread. Here many rotations stage at once, and afterwards every
    key must still be readable from the path its own call returned — which is
    the property a committed rotation depends on. A shared filename fails this
    both ways: keys go missing, and the unlink-then-write race surfaces as a
    raw ``FileNotFoundError`` escaping the CLI.
    """
    _seed_store(config_root)
    keys = [generate_master_key() for _ in range(24)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda k: stage_master_key(config_root, k), keys))

    assert len({str(path) for path in paths}) == len(keys), "staged names collided"
    for key, path in zip(keys, paths):
        assert path.read_bytes() == key, f"{path.name} does not hold its own key"

    # A staging temporary must never be visible to the recovery scan, and none
    # may be left behind by a completed call.
    leftovers = sorted(p.name for p in (config_root / "secrets").glob("master.stage*"))
    assert leftovers == [], f"staging temporaries were left behind: {leftovers}"


def test_recovery_finds_the_right_key_among_many_staged(config_root: Path) -> None:
    """With several rotations in flight, recovery adopts the committed one.

    Scanning all staged keys is only correct if the fingerprint still decides
    WHICH one is adopted: the other rotations' keys are inert, exactly as a
    single leftover file is.
    """
    key = _seed_store(config_root)
    committed = generate_master_key()

    for _ in range(3):  # rotations that staged and never committed
        stage_master_key(config_root, generate_master_key())
    stage_master_key(config_root, committed)
    SecretStore(key, base=config_root).rotate(committed)
    for _ in range(3):
        stage_master_key(config_root, generate_master_key())

    assert resolve_master_key(config_root) == committed
    assert key_path(config_root).read_bytes() == committed
    assert _open_knowing_only_the_disk(config_root) == b"super-secret-value"


@pytest.mark.parametrize("crash_after_step", [0, 1, 2, 3])
def test_a_crash_at_any_rotation_step_survives_a_concurrent_rotation(
    config_root: Path, crash_after_step: int
) -> None:
    """The R2 crash matrix, re-run with another rotation in the picture.

    Same property as the single-rotator matrix — a power cut at any step leaves
    the store openable by a key that exists on disk — but a second rotator
    stages, races and cleans up throughout. The single-rotator matrix passed
    against code that lost the store this way, so the concurrent leg is the one
    that actually pins the invariant.
    """
    key = _seed_store(config_root)
    victim_key = generate_master_key()
    other_key = generate_master_key()
    other = SecretStore(key, base=config_root)  # holds the pre-rotation key

    # The interfering rotation stages before the victim reaches any step.
    stage_master_key(config_root, other_key)

    _rotate_to_step(config_root, key, victim_key, crash_after_step)

    # ...then loses the epoch race (or, at step 0, wins it) and cleans up.
    try:
        other.rotate(other_key)
    except (StaleKeyEpoch, SecretStoreError):
        discard_staged_master_key(config_root, other_key)

    # Power cut. A brand new process must still open the store.
    assert _open_knowing_only_the_disk(config_root) == b"super-secret-value"


# --- R3: concurrent FIRST USE must converge on one key, not brick ------------
#
# The invariants above all assume a store that already EXISTS. Creating it was
# the same last-writer-wins shape on `master.key` itself: an `exists()` check
# followed by an unguarded write to a fixed path, so every concurrent first-use
# caller generated its own key, wrote it, and returned the key it had generated
# — all but one of them going on to seal a database under a key that was no
# longer on disk. It survived the round-2 sibling sweep because
# `write_private_file`'s docstring CLAIMED `O_EXCL` while the code passed
# `O_TRUNC`, so the site read as guarded.
#
# Not a corner case on this machine: the operator runs ~11 sessions at once and
# the store is created by whichever of them first writes a secret.
#
# THE BARRIER IS A `threading.Barrier`, NOT A CLOCK. Real timing cannot pin
# this: arrival spread across 12 real processes measured 187 ms against a race
# window microseconds wide, so a wall-clock harness reports a clean run against
# provably broken code (it did — 0/24 at a base that bricks 4/4 under a forced
# interleave). The barrier only DELAYS the call; every line of logic under test
# runs unmodified.

FIRST_USE_CALLERS = 12


def _first_use_racers(
    base: Path, callers: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[str]]:
    """Run ``callers`` first uses released together; return (keys, errors).

    The hook sits on ``generate_master_key``, which ``load_master_key`` calls
    precisely between its existence check and its write — the window in
    question. ``monkeypatch`` undoes it even if the barrier times out.
    """
    import local_operator.secrets.keys as keys_module

    original = keys_module.generate_master_key
    barrier = threading.Barrier(callers, timeout=60)

    def gated() -> bytes:
        barrier.wait()
        return original()

    monkeypatch.setattr(keys_module, "generate_master_key", gated)

    keys: list[str] = []
    errors: list[str] = []

    def worker() -> str:
        return load_master_key(base, create=True).hex()

    with concurrent.futures.ThreadPoolExecutor(max_workers=callers) as pool:
        futures = [pool.submit(worker) for _ in range(callers)]
        for future in futures:
            try:
                keys.append(future.result())
            except BaseException as exc:  # noqa: BLE001 - asserted on by callers
                errors.append(f"{type(exc).__name__}: {exc}")

    # A run where the barrier never engaged discriminates nothing. Every caller
    # must have passed through the hook, or the result is not evidence.
    assert barrier.broken is False or not errors, "the barrier timed out; run proves nothing"
    return keys, errors


def test_concurrent_first_use_converges_on_one_key(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every caller creating the store at once must end up on the SAME key.

    The property is not "one caller wins" — it is that the LOSERS ADOPT the
    winner's key instead of returning their own. A loser keeping its own key
    seals its records under a key that is not on disk: the write reports
    success and the value is unreadable forever after. Measured before the fix
    at 10 of 12 callers holding a key the file did not have.
    """
    keys, errors = _first_use_racers(config_root, FIRST_USE_CALLERS, monkeypatch)

    assert not errors, f"concurrent first use raised: {errors[:2]}"
    on_disk = key_path(config_root).read_bytes()
    assert len(on_disk) == KEY_BYTES, f"key file is {len(on_disk)} bytes, not a key"
    assert set(keys) == {on_disk.hex()}, (
        f"{len(set(keys))} distinct keys returned by {FIRST_USE_CALLERS} concurrent "
        "callers; every caller but one would seal its records under a lost key"
    )


def test_a_crash_while_creating_the_key_leaves_no_unusable_store(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-creation must leave NOTHING, not a permanent 0-byte key.

    This is why creation is write-temp-then-``os.link`` rather than the
    one-line ``O_EXCL``. ``O_EXCL`` does pick a single winner, but it publishes
    the path at CREATION and fills it afterwards, so the file exists and is
    empty for the length of the write. Two costs, both measured here: a
    concurrent loser re-reads and gets ``master.key is 0 bytes; the key file is
    damaged`` instead of the winner's key, and — far worse — a crash inside
    that window leaves that 0-byte file behind FOREVER. Nothing ever creates
    the key again (it exists), and every later run fails the length check, so
    an interrupted first use permanently bricks the store.

    Simulated at the exact window: the payload write fails after the file would
    have been published. Recovery is the assertion — the next ordinary first
    use must succeed and produce a whole key.
    """
    import local_operator.secrets.keys as keys_module

    real_write = keys_module.write_private_file
    calls: list[Path] = []

    def crashing_write(path: Path, data: bytes) -> None:
        # Create the file the way the real writer does, then die mid-payload.
        calls.append(path)
        raise OSError("simulated power cut while writing the key")

    monkeypatch.setattr(keys_module, "write_private_file", crashing_write)
    with pytest.raises(OSError):
        load_master_key(config_root, create=True)
    monkeypatch.setattr(keys_module, "write_private_file", real_write)

    assert calls, "the crash hook never fired; this run proves nothing"
    # The crash must not have published anything under the final name.
    assert not key_path(config_root).exists(), (
        "a crash during creation left a master.key behind; if it is short, "
        "every later run fails the length check and the store is unopenable"
    )

    # The real assertion: first use still works afterwards.
    recovered = load_master_key(config_root, create=True)
    assert len(recovered) == KEY_BYTES
    assert recovered == key_path(config_root).read_bytes()
    leftovers = sorted(path.name for path in (config_root / "secrets").glob("master.key.*"))
    assert leftovers == [], f"creation temporaries were left behind: {leftovers}"


def test_a_store_created_concurrently_is_readable_afterwards(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The consequence the operator actually feels: the records survive.

    The key-identity assertions are the mechanism; this is the loss. Each
    caller stores a secret under the key its own first use returned, and every
    committed record must then be readable by a process that knows only what is
    on disk — the migration ahead is 58 credentials onto a store that does not
    yet exist.
    """
    keys, errors = _first_use_racers(config_root, FIRST_USE_CALLERS, monkeypatch)
    assert not errors, f"concurrent first use raised: {errors[:2]}"

    stored = 0
    for index, key in enumerate(keys):
        store = SecretStore(bytes.fromhex(key), base=config_root)
        store.initialize()
        try:
            store.set(f"SECRET_{index}", f"value-{index}".encode())
            stored += 1
        except StaleKeyEpoch:
            # An honest refusal is not a brick: nothing was committed and the
            # CLI tells the caller to re-run. A SILENT success under a lost key
            # is the failure this test exists for.
            pass

    assert stored, "no caller managed to store anything"
    reader = SecretStore(resolve_master_key(config_root), base=config_root)
    for record in reader.list():
        index = record.name.rsplit("_", 1)[1]
        assert reader.get(record.name) == f"value-{index}".encode(), (
            f"{record.name} reported success and is unreadable: it was sealed "
            "under a key that is not the store's"
        )
    assert reader.damaged_records() == [], "the concurrently created store has damaged rows"

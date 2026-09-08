"""File modes: 0600 files in a 0700 directory, and a hostile umask must lose.

In the default ``keyfile`` mode the entire at-rest story is the mode bits, so
these are not hygiene assertions — they are the control itself. The umask cases
matter because ``mkdir(mode=...)`` and ``os.open(..., mode)`` are both MASKED
by the process umask: code that looks like it requests 0700 silently produces
0777 under ``umask 000``, and nothing reports it. The explicit ``chmod`` in
``keys.py`` and ``store.py`` is what these pin.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from local_operator.secrets.errors import InsecurePermissions
from local_operator.secrets.keys import (
    DIR_MODE,
    FILE_MODE,
    key_path,
    load_master_key,
    secrets_dir,
    store_path,
)
from local_operator.secrets.store import SecretStore

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX mode bits do not carry their Unix meaning on Windows"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def hostile_umask():
    """Run the body under ``umask 000`` — the mode-loosening case.

    Restored in a finally: umask is process-global, so leaking it would corrupt
    every later test in the same xdist worker.
    """
    previous = os.umask(0o000)
    try:
        yield
    finally:
        os.umask(previous)


def test_directory_is_0700(store: SecretStore, config_root: Path) -> None:
    assert _mode(secrets_dir(config_root)) == DIR_MODE


def test_database_is_0600(store: SecretStore, config_root: Path) -> None:
    store.set("TOKEN", b"value")
    assert _mode(store_path(config_root)) == FILE_MODE


def test_wal_sidecars_are_0600(store: SecretStore, config_root: Path) -> None:
    """The WAL holds the same ciphertext as the database.

    A 0600 database beside a 0644 ``-wal`` is not a private store, and SQLite
    creates those sidecars itself under the process umask.
    """
    store.set("TOKEN", b"value")
    database = store_path(config_root)
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.exists():
            assert _mode(sidecar) == FILE_MODE, f"{sidecar.name} is {_mode(sidecar):04o}"


def test_keyfile_is_0600(config_root: Path) -> None:
    load_master_key(config_root, create=True)
    assert _mode(key_path(config_root)) == FILE_MODE


def test_hostile_umask_does_not_loosen_the_directory(
    config_root: Path, hostile_umask: None
) -> None:
    """``umask 000`` must not produce a world-traversable secrets directory."""
    load_master_key(config_root, create=True)
    store = SecretStore(b"\x00" * 32, base=config_root)
    store.initialize()

    assert _mode(secrets_dir(config_root)) == DIR_MODE
    assert _mode(key_path(config_root)) == FILE_MODE
    assert _mode(store_path(config_root)) == FILE_MODE


def test_hostile_umask_does_not_loosen_a_written_record(
    config_root: Path, hostile_umask: None
) -> None:
    """Writing a record re-asserts the modes, including on the fresh WAL."""
    key = load_master_key(config_root, create=True)
    store = SecretStore(key, base=config_root)
    store.set("TOKEN", b"value")

    database = store_path(config_root)
    assert _mode(database) == FILE_MODE
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.exists():
            assert _mode(sidecar) == FILE_MODE


def test_a_group_readable_keyfile_is_refused(config_root: Path) -> None:
    """Loosened after the fact — the case worth catching, and NOT repaired.

    Silently tightening the mode would hide from the operator that the key was
    ever exposed. By the time this is observable the exposure has happened, so
    it is reported.
    """
    load_master_key(config_root, create=True)
    path = key_path(config_root)
    os.chmod(path, 0o640)

    with pytest.raises(InsecurePermissions, match="group or others"):
        load_master_key(config_root)

    # Not repaired behind the operator's back.
    assert _mode(path) == 0o640


def test_a_world_readable_database_is_refused(config_root: Path) -> None:
    key = load_master_key(config_root, create=True)
    store = SecretStore(key, base=config_root)
    store.set("TOKEN", b"value")
    os.chmod(store_path(config_root), 0o644)

    with pytest.raises(InsecurePermissions, match="group or others"):
        store.get("TOKEN")


def test_a_truncated_keyfile_is_reported_clearly(config_root: Path) -> None:
    """A damaged key must not be silently padded into a valid-looking one."""
    load_master_key(config_root, create=True)
    path = key_path(config_root)
    path.write_bytes(b"too-short")
    os.chmod(path, FILE_MODE)

    from local_operator.secrets.errors import SecretStoreError

    with pytest.raises(SecretStoreError, match="master key is 32"):
        load_master_key(config_root)


def test_the_value_is_not_findable_in_the_raw_database_file(
    store: SecretStore, config_root: Path
) -> None:
    """The point of the whole feature, asserted against the bytes on disk.

    ``grep -r 'API_KEY' ~`` is the baseline this store exists to beat, so the
    test greps the actual file: neither the value NOR the name may appear.
    """
    store.set("MINERVA_PROD_DB_PASSWORD", b"hunter2-the-real-password", description="prod")
    # Checkpoint the WAL so the record is definitely in the main file too.
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    database = store_path(config_root)
    blob = database.read_bytes()
    for sidecar_suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + sidecar_suffix)
        if sidecar.exists():
            blob += sidecar.read_bytes()

    assert b"hunter2-the-real-password" not in blob
    assert b"MINERVA_PROD_DB_PASSWORD" not in blob, "the secret NAME leaked in the clear"
    assert b"prod" not in blob, "the description leaked in the clear"

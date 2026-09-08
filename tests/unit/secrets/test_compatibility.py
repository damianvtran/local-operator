"""Version skew must be refused with a clear message, never mis-parsed.

Design §13: the operator runs ~10 sessions and updates the runtime with
`lop-update` while they are live, so an older runtime meeting a newer store is
routine rather than exceptional. The failure mode being prevented is not a
crash — it is an older runtime reading a record layout it half-understands and
returning something plausible.
"""

from __future__ import annotations

import sqlite3

import pytest

from local_operator.secrets.crypto import RECORD_FORMAT_VERSION
from local_operator.secrets.errors import IncompatibleStore
from local_operator.secrets.store import SCHEMA_VERSION, SecretStore


def test_a_newer_schema_version_is_refused_on_read(store: SecretStore) -> None:
    store.set("TOKEN", b"value")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE meta SET v = ? WHERE k = 'schema_version'",
            (str(SCHEMA_VERSION + 1).encode("ascii"),),
        )

    with pytest.raises(IncompatibleStore, match="newer local-operator"):
        store.get("TOKEN")


def test_a_newer_schema_version_is_refused_on_write(store: SecretStore) -> None:
    """Refusing to WRITE is the important half: a downward migration is
    unrecoverable, and a newer runtime's records must not be clobbered."""
    store.set("TOKEN", b"value")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE meta SET v = ? WHERE k = 'schema_version'",
            (str(SCHEMA_VERSION + 5).encode("ascii"),),
        )

    with pytest.raises(IncompatibleStore, match="upgrade this runtime"):
        store.update("TOKEN", b"new-value")
    with pytest.raises(IncompatibleStore):
        store.set("OTHER", b"value")


def test_the_message_names_the_versions_and_the_fix(store: SecretStore) -> None:
    """A version error the operator cannot act on is barely better than a crash."""
    store.set("TOKEN", b"value")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE meta SET v = ? WHERE k = 'schema_version'",
            (str(SCHEMA_VERSION + 1).encode("ascii"),),
        )

    with pytest.raises(IncompatibleStore) as caught:
        store.list()
    message = str(caught.value)
    assert str(SCHEMA_VERSION + 1) in message
    assert str(SCHEMA_VERSION) in message
    assert "lop update" in message


def test_a_newer_record_format_is_refused(store: SecretStore) -> None:
    """Per-record, not just per-store.

    §13 requires records to carry their own format version so a partially
    migrated store stays coherent. A record from the future is refused
    individually rather than being decoded under this runtime's assumptions.
    """
    store.set("TOKEN", b"value")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE secrets SET format_version = ?", (RECORD_FORMAT_VERSION + 1,))

    with pytest.raises(IncompatibleStore, match="record format"):
        store.get("TOKEN")


def test_an_older_schema_version_is_also_refused(store: SecretStore) -> None:
    """No silent upward migration either.

    Not because the operator will hit it today — this is schema 1 — but because
    the moment schema 2 exists, "read an old store" must be a deliberate
    migration someone wrote, not an accident of tolerant parsing.
    """
    store.set("TOKEN", b"value")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE meta SET v = ? WHERE k = 'schema_version'", (b"0",))

    with pytest.raises(IncompatibleStore, match="older local-operator"):
        store.get("TOKEN")


def test_a_missing_store_reports_how_to_create_one(config_root) -> None:
    from local_operator.secrets.errors import SecretStoreError

    store = SecretStore(b"\x00" * 32, base=config_root)
    with pytest.raises(SecretStoreError, match="lop secret set"):
        store.get("TOKEN")

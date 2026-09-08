"""The audit hash chain: appending verifies, editing breaks it detectably.

Design §12. The claim is tamper-EVIDENT, not tamper-proof — whoever owns the
file can still delete the whole thing — so what these prove is the narrower and
achievable property: an attacker cannot silently EDIT history. Either the chain
verifies, or it reports the first entry that does not.
"""

from __future__ import annotations

import sqlite3

from local_operator.secrets.store import SecretStore


def test_every_operation_appends_an_entry(store: SecretStore) -> None:
    store.set("TOKEN", b"value")
    store.get("TOKEN")
    store.update("TOKEN", b"new-value")
    store.get("TOKEN")
    store.delete("TOKEN")

    events = [entry.event for entry in store.audit_entries()]
    assert events == ["set", "get", "update", "get", "delete"]


def test_an_intact_chain_verifies(store: SecretStore) -> None:
    for index in range(10):
        store.set(f"SECRET_{index}", b"value")
        store.get(f"SECRET_{index}")

    ok, position, message = store.verify_audit()
    assert ok, message
    assert position is None
    assert message == "audit chain intact"


def test_an_empty_chain_verifies(store: SecretStore) -> None:
    ok, position, _ = store.verify_audit()
    assert ok and position is None


def test_no_entry_ever_records_a_value(store: SecretStore) -> None:
    """The audit log must not become the leak it exists to detect."""
    store.set("TOKEN", b"super-secret-value")
    store.get("TOKEN")
    for entry in store.audit_entries():
        assert "super-secret-value" not in repr(entry)
        assert entry.secret_id is not None
        assert "TOKEN" not in repr(entry), "the audit log records ids, not names"


def test_editing_an_entry_in_place_breaks_the_chain(store: SecretStore) -> None:
    """The attack: rewrite a retrieval you made to look like something else."""
    store.set("TOKEN", b"value")
    for _ in range(4):
        store.get("TOKEN")

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE audit SET event = 'describe' WHERE rowid = "
            "(SELECT rowid FROM audit ORDER BY rowid LIMIT 1 OFFSET 2)"
        )

    ok, position, message = store.verify_audit()
    assert not ok
    assert position == 3
    assert "edited after it was written" in message


def test_deleting_a_middle_entry_breaks_the_chain(store: SecretStore) -> None:
    """The other attack: remove the retrieval entirely.

    ``prev_hash`` of the following entry no longer matches, so the gap is
    reported at the entry that now sits where the deleted one used to link.
    """
    store.set("TOKEN", b"value")
    for _ in range(4):
        store.get("TOKEN")

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM audit WHERE rowid = "
            "(SELECT rowid FROM audit ORDER BY rowid LIMIT 1 OFFSET 2)"
        )

    ok, position, message = store.verify_audit()
    assert not ok
    assert position == 3
    assert "removed or reordered" in message


def test_reordering_entries_breaks_the_chain(store: SecretStore) -> None:
    store.set("ALPHA", b"value")
    store.set("BETA", b"value")
    store.get("ALPHA")

    with sqlite3.connect(store.path) as connection:
        rows = connection.execute("SELECT rowid, ts FROM audit ORDER BY rowid").fetchall()
        # Swap the timestamps of entries 1 and 2, which is enough to change the
        # canonical row bytes without touching the hashes.
        connection.execute("UPDATE audit SET ts = ? WHERE rowid = ?", (rows[1][1], rows[0][0]))
        connection.execute("UPDATE audit SET ts = ? WHERE rowid = ?", (rows[0][1], rows[1][0]))

    ok, position, _ = store.verify_audit()
    assert not ok
    assert position == 1


def test_appending_a_forged_entry_breaks_the_chain(store: SecretStore) -> None:
    """An attacker cannot append plausible history without the chain state.

    They CAN compute a correct hash if they read the previous row — the chain
    is not a MAC and §12 does not claim it is. What this pins is that a naive
    insert, which is what an attacker scripting sqlite3 actually does, is
    detected.
    """
    store.set("TOKEN", b"value")

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO audit(ts, event, secret_id, session_id, pid, exe, outcome,"
            " prev_hash, hash) VALUES(?, 'get', 'x', NULL, 1, NULL, 'ok', ?, ?)",
            (1.0, b"\x00" * 32, b"\x11" * 32),
        )

    ok, position, message = store.verify_audit()
    assert not ok
    assert position == 2
    assert "removed or reordered" in message


def test_the_chain_survives_a_reopen(store: SecretStore) -> None:
    """The chain is state on disk, not in the object; a new process continues it."""
    store.set("TOKEN", b"value")
    reopened = SecretStore(store._master_key, base=store.path.parent.parent)
    reopened.get("TOKEN")
    reopened.get("TOKEN")

    ok, _, message = reopened.verify_audit()
    assert ok, message
    assert len(reopened.audit_entries()) == 3


def test_a_rotation_is_recorded(store: SecretStore) -> None:
    from local_operator.secrets.crypto import generate_master_key

    store.set("TOKEN", b"value")
    store.rotate(generate_master_key())

    assert "rotate" in [entry.event for entry in store.audit_entries()]
    ok, _, message = store.verify_audit()
    assert ok, message

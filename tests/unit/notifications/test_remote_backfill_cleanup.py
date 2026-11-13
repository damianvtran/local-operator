"""The one-time cleanup of the v0.102.2 remote backfill, and its undo.

WHY (operator rule, 2026-11-13): the first serve poll after v0.102.2 wrote 127
unread ``remote-done:`` rows and 34 unread ``remote-notice:`` rows into the
head's inbox. Opening the store marks those READ once (never deletes, never
touches parks/asks/other items), and the undo restores exactly what it marked.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from local_operator.notifications import backfill_cleanup
from local_operator.notifications import store as store_mod
from local_operator.notifications.store import NotificationStore


def _items(root: Path) -> dict[str, dict[str, Any]]:
    """Every row, dismissed ones included (``list_items`` hides those), by key."""
    store = NotificationStore(root)
    with sqlite3.connect(store.path) as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM items")]
    out: dict[str, dict[str, Any]] = {}
    for item_id in ids:
        row = store.get(item_id)
        assert row is not None
        out[row["key"]] = row
    return out


def _pre_fix_store(root: Path) -> dict[str, str]:
    """A store as v0.102.2 left it: the flag absent, the flood unread."""
    store = NotificationStore(root)
    ids = {}
    for key in (
        "remote-done:d_node:s1",
        "remote-done:d_node:s2",
        "remote-notice:d_node:s1:k",
        "remote-gate:d_node:s1:w1",
        "gate:approval:w2",
        "pr-9",
    ):
        ids[key] = store.notify("s1", title=key, key=key).item_id
    already = store.notify("s3", title="read before", key="remote-done:d_node:s3")
    store.apply("read", ids=[(already.item_id, already.revision)], via="explicit")
    store.notify("s4", title="dismissed", key="remote-notice:d_node:s4:k")
    dismissed = _items(root)["remote-notice:d_node:s4:k"]
    store.apply("dismiss", ids=[(dismissed["id"], dismissed["revision"])])
    # This build's own first open already set the flag; take it away so the next
    # open behaves as the first open after the upgrade.
    with sqlite3.connect(store.path) as conn:
        conn.execute("DELETE FROM meta WHERE k=?", (store_mod.CLEANUP_META_KEY,))
        conn.execute(
            "UPDATE items SET read_rev=0, read_at=NULL, read_via=NULL WHERE read_via=?",
            (store_mod.CLEANUP_VIA,),
        )
    return ids


def test_the_first_open_marks_only_the_flood_read_and_deletes_nothing(tmp_path: Path) -> None:
    _pre_fix_store(tmp_path)
    before = _items(tmp_path)
    record = NotificationStore(tmp_path).remote_backfill_cleanup_record()
    after = _items(tmp_path)
    assert set(after) == set(before), "nothing deleted"
    marked = {key for key, row in after.items() if row["read_via"] == store_mod.CLEANUP_VIA}
    assert marked == {
        "remote-done:d_node:s1",
        "remote-done:d_node:s2",
        "remote-notice:d_node:s1:k",
        "remote-notice:d_node:s4:k",
    }
    assert all(after[key]["read"] for key in marked)
    for untouched in ("remote-gate:d_node:s1:w1", "gate:approval:w2", "pr-9"):
        assert after[untouched]["read"] is False and after[untouched]["read_via"] is None
    assert after["remote-done:d_node:s3"]["read_via"] == "explicit", "already read: left alone"
    assert after["remote-notice:d_node:s4:k"]["dismissed"] is True
    assert record is not None and len(record["rows"]) == 4


def test_the_cleanup_runs_once(tmp_path: Path) -> None:
    _pre_fix_store(tmp_path)
    NotificationStore(tmp_path).remote_backfill_cleanup_record()  # the first write open
    store = NotificationStore(tmp_path)
    # A forwarded row written AFTER the cleanup is never swept by a later open.
    fresh = store.notify("s9", title="new notice", key="remote-notice:d_node:s9:k")
    NotificationStore(tmp_path).remote_backfill_cleanup_record()  # a later write open
    row = NotificationStore(tmp_path).get(fresh.item_id)
    assert row is not None and row["read"] is False


def test_undo_restores_exactly_the_rows_it_marked(tmp_path: Path) -> None:
    _pre_fix_store(tmp_path)
    store = NotificationStore(tmp_path)
    marked = store.remote_backfill_cleanup_record()
    assert marked is not None
    # The operator re-read one row themselves; another was updated since.
    rows = _items(tmp_path)
    own = rows["remote-done:d_node:s1"]
    store.apply("unread", ids=[(own["id"], own["revision"])])
    store.apply("read", ids=[(own["id"], own["revision"])], via="explicit")
    store.notify("s1", title="newer", key="remote-done:d_node:s2")

    assert store.undo_remote_backfill_cleanup(dry_run=True) == 2
    assert _items(tmp_path)["remote-notice:d_node:s1:k"]["read"] is True, "dry run writes nothing"
    assert store.undo_remote_backfill_cleanup() == 2
    after = _items(tmp_path)
    assert after["remote-notice:d_node:s1:k"]["read"] is False
    assert after["remote-notice:d_node:s1:k"]["read_via"] is None
    assert after["remote-notice:d_node:s4:k"]["dismissed"] is True, "dismiss marks untouched"
    assert after["remote-done:d_node:s1"]["read_via"] == "explicit", "the operator's own read"
    assert after["remote-done:d_node:s2"]["title"] == "newer"
    assert store.undo_remote_backfill_cleanup() == 0, "idempotent"
    # The flag stays, so the next write open does not mark them again.
    NotificationStore(tmp_path).notify("s5", title="unrelated", key="x")
    assert _items(tmp_path)["remote-notice:d_node:s1:k"]["read"] is False
    record = store.remote_backfill_cleanup_record()
    assert record is not None and record.get("undone_at")


def test_the_maintenance_entry_point_reports_and_undoes(tmp_path: Path, capsys: Any) -> None:
    _pre_fix_store(tmp_path)
    assert backfill_cleanup.main(["--config-dir", str(tmp_path)]) == 0
    assert "cleanup marked read: 4 rows" in capsys.readouterr().out
    assert backfill_cleanup.main(["--config-dir", str(tmp_path), "--undo", "--dry-run"]) == 0
    assert "would restore: 4 rows" in capsys.readouterr().out
    assert backfill_cleanup.main(["--config-dir", str(tmp_path), "--undo"]) == 0
    assert "restored: 4 rows" in capsys.readouterr().out

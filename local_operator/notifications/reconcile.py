"""Bounded list-row reconciliation after a receipt-backed notification gesture.

This is a read, never another selection. Use the existing list projection (no
full body or ask answers), and give it a separate snapshot fence: replaying an
old operation must not stamp current rows with the old operation's sequence.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from local_operator.notifications import bulk_asks
from local_operator.notifications import items as inbox
from local_operator.notifications.store import NotificationStore, _row_dict, ask_ref


def read_rows(store: NotificationStore, ids: Sequence[str]) -> dict[str, Any]:
    """Read requested metadata within this desktop's already-authorized store.

    The SQLite rows/marks and watermark share one read transaction. Ask lifecycle
    projection comes from the same derived index as GET, not an invented
    cross-store transaction. Unknown/other-store IDs are uniformly absent.
    """
    requested = list(dict.fromkeys(ids))
    requested_set = set(requested)
    asks = (
        {
            ask_ref(str(row["session_id"]), str(row["ask_id"])): row
            for row in inbox.open_asks(store.config_dir)
        }
        if any(item.startswith("ask:") for item in requested)
        else {}
    )
    asks = bulk_asks.current_asks(
        store.config_dir, {ref: row for ref, row in asks.items() if ref in requested_set}
    )
    sessions = inbox.SessionFacts(store.config_dir)
    projects = inbox.ProjectFacts(store.config_dir)

    def body(conn: sqlite3.Connection) -> dict[str, Any]:
        conn.execute("BEGIN")
        row = conn.execute("SELECT v FROM meta WHERE k='next_seq'").fetchone()
        high_water = int(row[0]) - 1 if row else 0
        marks = ",".join("?" for _ in requested)
        updates = (
            {
                row["id"]: inbox.notify_item(_row_dict(row), sessions, projects)
                for row in conn.execute(f"SELECT * FROM items WHERE id IN ({marks})", requested)
            }
            if requested
            else {}
        )
        states = (
            {
                row["ref"]: dict(row)
                for row in conn.execute(
                    f"SELECT * FROM ref_state WHERE ref IN ({marks})", requested
                )
            }
            if requested
            else {}
        )
        projections = {
            **updates,
            **{
                ref: inbox.ask_item(row, states, sessions, projects)
                for ref, row in asks.items()
                if ref in requested_set
            },
        }
        return {
            "items": [projections[ref] for ref in requested if ref in projections],
            "absent_ids": [ref for ref in requested if ref not in projections],
            "high_water": high_water,
        }

    return store._read(body, {"items": [], "absent_ids": requested, "high_water": 0}, "reconcile")

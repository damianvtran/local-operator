"""Frozen ask-notification read marks, not authority over ask settlement.

Ask identities are outside the notifications sequence. Capture them explicitly
at preview, then fence each read mark by its sequence and the displayed ask's
lifecycle fingerprint. Revalidation reads the ask index once per operation;
there is deliberately no transaction, lock or write on the ask log. Settlement
racing the read may leave a harmless notification mark but cannot reopen an ask.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_operator.notifications.store import NotificationStore, ask_ref


def _lifecycle(row: Mapping[str, Any]) -> str:
    # Ignore delivery bookkeeping, but not a replaced/reworded question or a
    # changed deadline. The public revision is always 1 and cannot fence these.
    identity = {key: row.get(key) for key in ("created_at", "expires_at", "questions")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def capture(conn: sqlite3.Connection, token: str, asks: Sequence[Mapping[str, Any]]) -> None:
    """Persist identities and notification read-state preimages in one transaction."""
    conn.executemany(
        "INSERT INTO bulk_ask_refs(token,ref,lifecycle) VALUES (?,?,?)",
        [
            (token, ask_ref(str(row["session_id"]), str(row["ask_id"])), _lifecycle(row))
            for row in asks
        ],
    )
    # Indexed lookups on ref_state's primary key, not a repeated full-store scan.
    conn.execute(
        "UPDATE bulk_ask_refs SET expected_seq=coalesce("
        "(SELECT seq FROM ref_state WHERE ref=bulk_ask_refs.ref),0),"
        "read_at=(SELECT read_at FROM ref_state WHERE ref=bulk_ask_refs.ref) WHERE token=?",
        (token,),
    )


def snapshots(conn: sqlite3.Connection, token: str, excluded: Sequence[str] = ()) -> list[Any]:
    rows = conn.execute("SELECT * FROM bulk_ask_refs WHERE token=?", (token,)).fetchall()
    exclusions = set(excluded)
    return [row for row in rows if row["ref"] not in exclusions]


def valid_rows(
    conn: sqlite3.Connection,
    token: str,
    root: Path,
    excluded: Sequence[str] = (),
    *,
    undo: bool = False,
) -> list[Any]:
    from local_operator.notifications.items import open_asks

    # Compare under the notification write lock. Another notification client
    # cannot change the mark between this read and the receipt write/Undo.
    rows = conn.execute(
        "SELECT r.*,s.read_at AS current_read_at FROM bulk_ask_refs r "
        "LEFT JOIN ref_state s ON s.ref=r.ref WHERE r.token=? AND "
        + ("s.seq=r.post_seq" if undo else "coalesce(s.seq,0)=r.expected_seq"),
        (token,),
    ).fetchall()
    if not rows:
        return []
    live = {
        ask_ref(str(row["session_id"]), str(row["ask_id"])): _lifecycle(row)
        for row in open_asks(root)
    }
    exclusions = set(excluded)
    candidates = [
        row
        for row in rows
        if row["ref"] not in exclusions and live.get(row["ref"]) == row["lifecycle"]
    ]
    # The index can lag a committed settlement (e.g. a crash before publishing
    # its projection). Read each selected session's log once, not once per ref.
    # A session without a local log retains the existing derived-index path
    # for mirrored/legacy asks; an existing but unreadable log fails closed.
    from local_operator.asks import store as ask_store

    logs: dict[str, dict[str, Any] | None] = {}
    valid = []
    for row in candidates:
        _, sid, ask_id = row["ref"].split(":", 2)
        if sid not in logs:
            directory = ask_store.session_dir(root, sid)
            logs[sid] = (
                {
                    str(record["ask_id"]): record
                    for record in ask_store.fold(
                        ask_store.read_events(directory), ask_store.now_ms(), session_id=sid
                    )
                }
                if ask_store.asks_log_path(directory).exists()
                else None
            )
        records = logs[sid]
        if records is not None:
            record = records.get(ask_id)
            if record is None or not ask_store.is_outstanding(record.get("status")):
                continue
            if not record.get("relay") and _lifecycle(record) != row["lifecycle"]:
                continue
        valid.append(row)
    return valid


def preview_counts(aggregates: dict[str, Any], refs: Sequence[Any]) -> dict[str, Any]:
    """Split acknowledgement targets from update-only dismissal eligibility."""

    def extend(counts: dict[str, int], rows: Sequence[Any]) -> None:
        counts["eligible_update_count"] = counts["eligible_count"]
        counts["eligible_unread_update_count"] = counts["eligible_unread_count"]
        counts["eligible_ask_count"] = len(rows)
        counts["eligible_unread_ask_count"] = sum(row["read_at"] is None for row in rows)
        counts["eligible_count"] += counts["eligible_ask_count"]
        counts["eligible_unread_count"] += counts["eligible_unread_ask_count"]

    extend(aggregates, refs)
    # At most 50 requested scopes; one grouping pass avoids scanning N asks for
    # each scope on a large waiting list.
    sessions: dict[str, list[Any]] = {sid: [] for sid in aggregates["session_counts"]}
    for row in refs:
        sid = row["ref"].split(":", 2)[1]
        if sid in sessions:
            sessions[sid].append(row)
    for sid, rows in sessions.items():
        extend(aggregates["session_counts"][sid], rows)
    aggregates["operation_counts"] = {
        op: {
            "eligible_count": (
                aggregates["eligible_count"]
                if op in ("read", "unread")
                else aggregates["eligible_update_count"]
            ),
            "skipped_ask_count": 0 if op in ("read", "unread") else len(refs),
        }
        for op in ("read", "unread", "dismiss", "undismiss")
    }
    aggregates["selection_version"] = 2
    return aggregates


def apply(
    conn: sqlite3.Connection,
    store: NotificationStore,
    token: str,
    op: str,
    stamp: int,
    excluded: Sequence[str],
) -> list[tuple[str, int]]:
    if op not in ("read", "unread"):
        return []
    changes = []
    for row in valid_rows(conn, token, store.config_dir, excluded):
        before = row["read_at"] is not None
        if before == (op == "read"):
            continue
        post_seq = store.write_ref(conn, row["ref"], stamp if op == "read" else None)
        conn.execute(
            "UPDATE bulk_ask_refs SET post_seq=? WHERE token=? AND ref=?",
            (post_seq, token, row["ref"]),
        )
        changes.append((row["ref"], -1 if op == "read" else 1))
    return changes


def undo(conn: sqlite3.Connection, store: NotificationStore, token: str) -> list[tuple[str, int]]:
    changes = []
    for row in valid_rows(conn, token, store.config_dir, undo=True):
        # Restore the prior read timestamp, but retain a new seq tombstone even
        # when the preimage had no row. Deleting would erase the ABA fence.
        store.write_ref(conn, row["ref"], row["read_at"])
        delta = int(row["read_at"] is None) - int(row["current_read_at"] is None)
        changes.append((row["ref"], delta))
    return changes


def result_counts(answer: dict[str, Any], changes: Sequence[tuple[str, int]]) -> None:
    """Preserve update badge deltas; waiting count never changes with a read mark."""
    answer["selection_version"] = 2
    answer["changed_update_count"] = answer["changed_count"]
    answer["changed_ask_count"] = len(changes)
    answer["changed_count"] += len(changes)
    answer["skipped_count"] -= len(changes)
    answer["waiting_unread_delta"] = sum(delta for _, delta in changes)
    for counts in answer["session_counts"].values():
        counts["changed_update_count"] = counts["changed_count"]
        counts["changed_ask_count"] = 0
        counts["waiting_unread_delta"] = 0
    for ref, delta in changes:
        sid = ref.split(":", 2)[1]
        if sid in answer["session_counts"]:
            counts = answer["session_counts"][sid]
            counts["changed_count"] += 1
            counts["changed_ask_count"] += 1
            counts["waiting_unread_delta"] += delta

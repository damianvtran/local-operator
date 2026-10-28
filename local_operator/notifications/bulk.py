"""Bounded inbox selections and transactional notification-state Undo receipts.

The SQLite sequence changes for *every* row write, not only content revisions.
A filter and a server-captured watermark therefore describe a safe subset even
without shipping every matching id to the client: anything arriving or changing
after preview is excluded. V1 selections remain update-only. V2 captures bounded
ask refs separately and may acknowledge their notification read marks. Ask
settlement lives elsewhere and is never part of this transaction or Undo.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from local_operator.notifications import bulk_asks
from local_operator.notifications.store import NotificationStore, now_ms

TOKEN_TTL_MS = 10 * 60 * 1000
MAX_TOKENS = 128
MAX_ROWS = 100_000
# Capture the complete state, including read provenance. Dismiss does NOT imply
# read, and Undo must not turn a previously read notification into an unread one.
_STATE = ("read_rev", "read_at", "read_via", "read_surface", "dismissed_at", "dismissed_via")
_COLUMNS = "id,seq,revision,session_id," + ",".join(_STATE)


def _aggregates(
    conn: sqlite3.Connection, where: str, params: list[Any], sessions: Sequence[str]
) -> dict[str, Any]:
    total = conn.execute(
        "SELECT count(*),coalesce(sum(read_rev<revision),0) FROM items WHERE " + where, params
    ).fetchone()
    by_session = {sid: {"eligible_count": 0, "eligible_unread_count": 0} for sid in sessions}
    if sessions:
        marks = ",".join("?" for _ in sessions)
        for row in conn.execute(
            "SELECT session_id,count(*),sum(read_rev<revision) FROM items WHERE "
            + where
            + f" AND session_id IN ({marks}) GROUP BY session_id",
            [*params, *sessions],
        ):
            by_session[row[0]] = {"eligible_count": row[1], "eligible_unread_count": row[2]}
    return {
        "eligible_count": total[0],
        "eligible_unread_count": total[1],
        "session_counts": by_session,
    }


def _exclude(where: str, params: list[Any], excluded: Sequence[str]) -> tuple[str, list[Any]]:
    if excluded:
        where += " AND id NOT IN (" + ",".join("?" for _ in excluded) + ")"
        params = [*params, *excluded]
    return where, params


def _badge(row: Mapping[str, Any]) -> int:
    return int(row["dismissed_at"] is None and row["read_rev"] < row["revision"])


def _delta(row: Mapping[str, Any], op: str) -> int:
    before = _badge(row)
    after = dict(row)
    if op == "read":
        after["read_rev"] = row["revision"]
    elif op == "unread":
        after["read_rev"] = 0
    elif op == "dismiss":
        after["dismissed_at"] = 1
    else:
        after["dismissed_at"] = None
    return _badge(after) - before


class SelectionError(ValueError):
    """Machine-readable refusal: no mutation has committed when this escapes."""

    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def _housekeep(conn: sqlite3.Connection, stamp: int) -> None:
    # Expire only our own metadata. Never evict a live Undo receipt to make room
    # for a new preview, which would silently break a promise already made.
    conn.execute(
        "DELETE FROM bulk_rows WHERE token IN (SELECT token FROM bulk_tokens WHERE expires_at<=?)",
        (stamp,),
    )
    conn.execute(
        "DELETE FROM bulk_ask_refs WHERE token IN "
        "(SELECT token FROM bulk_tokens WHERE expires_at<=?)",
        (stamp,),
    )
    conn.execute("DELETE FROM bulk_tokens WHERE expires_at<=?", (stamp,))


def _retained_rows(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT (SELECT count(*) FROM bulk_rows)+(SELECT count(*) FROM bulk_ask_refs)"
        ).fetchone()[0]
    )


def _session_badges(conn: sqlite3.Connection, answer: dict[str, Any]) -> None:
    """Fence authoritative bounded scope badges with their own coherent sequence.

    The route's global counts are read after commit and may include another
    writer. They cannot lend their newer high_water to these transaction-local
    scope counts, or a client could pin stale data ahead of a newer event.
    """
    scopes = answer["session_counts"]
    for entry in scopes.values():
        entry["unread_count"] = 0
    if scopes:
        marks = ",".join("?" for _ in scopes)
        for row in conn.execute(
            "SELECT session_id,count(*) FROM items WHERE dismissed_at IS NULL "
            f"AND read_rev<revision AND session_id IN ({marks}) GROUP BY session_id",
            list(scopes),
        ):
            scopes[row[0]]["unread_count"] = row[1]
    row = conn.execute("SELECT v FROM meta WHERE k='next_seq'").fetchone()
    answer["counts_high_water"] = int(row[0]) - 1 if row else 0


def _room(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT count(*) FROM bulk_tokens").fetchone()[0] >= MAX_TOKENS:
        raise SelectionError("selection_capacity", 429)


def _where(filters: Mapping[str, Any], through_seq: int) -> tuple[str, list[Any]]:
    clauses = ["seq<=?"]
    params: list[Any] = [through_seq]
    state = filters["state"]
    clauses.append("dismissed_at IS NOT NULL" if state == "dismissed" else "dismissed_at IS NULL")
    if state == "unread":
        clauses.append("read_rev<revision")
    if filters["kind"] == "asks":
        clauses.append("0")
    priority = filters["bulk_priority"]
    if priority == "high":
        clauses.append("priority='high'")
    elif priority == "normal_up":
        clauses.append("priority IN ('high','normal')")
    if filters.get("session_id"):
        clauses.append("session_id=?")
        params.append(filters["session_id"])
    return " AND ".join(clauses), params


def preview(
    store: NotificationStore,
    filters: Mapping[str, Any],
    *,
    scope: str,
    skipped_ask_count: int,
    ask_rows: Sequence[Mapping[str, Any]] = (),
    now: int | None = None,
) -> dict[str, Any]:
    """Freeze a selection without changing any notification or ask.

    Only bounded selection metadata is persisted by this read-only preview.
    The count and watermark share a transaction, avoiding a count/boundary race.
    """
    stamp = now_ms() if now is None else now

    def body(conn: sqlite3.Connection) -> dict[str, Any]:
        _housekeep(conn, stamp)
        _room(conn)
        row = conn.execute("SELECT v FROM meta WHERE k='next_seq'").fetchone()
        high_water = int(row[0]) - 1 if row else 0
        where, params = _where(filters, high_water)
        aggregates = _aggregates(conn, where, params, filters.get("count_session_ids", []))
        v2 = bool(filters.get("include_ask_reads"))
        count = aggregates["eligible_count"] + (len(ask_rows) if v2 else 0)
        if count > MAX_ROWS:
            raise SelectionError("selection_too_large", 413)
        if v2 and _retained_rows(conn) + len(ask_rows) > MAX_ROWS:
            raise SelectionError("receipt_capacity", 429)
        token = secrets.token_urlsafe(32)
        expires = stamp + TOKEN_TTL_MS
        conn.execute(
            "INSERT INTO bulk_tokens(token,scope,expires_at,filters_json,through_seq,"
            "eligible_count,skipped_ask_count) VALUES (?,?,?,?,?,?,?)",
            (
                token,
                scope,
                expires,
                json.dumps(dict(filters)),
                high_water,
                count,
                skipped_ask_count,
            ),
        )
        if v2:
            bulk_asks.capture(conn, token, ask_rows)
            bulk_asks.preview_counts(aggregates, bulk_asks.snapshots(conn, token))
        return {
            "token": token,
            "through_seq": high_water,
            "expires_at": expires,
            **aggregates,
            "skipped_ask_count": 0 if v2 else skipped_ask_count,
        }

    return store._write(body, "bulk_preview")


def _token(conn: sqlite3.Connection, token: str, scope: str, stamp: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM bulk_tokens WHERE token=?", (token,)).fetchone()
    # Deliberately indistinguishable: possession of a foreign/expired handle
    # must not become an oracle for another desktop authorization generation.
    if row is None or row["scope"] != scope or row["expires_at"] <= stamp:
        raise SelectionError("selection_unavailable")
    return row


def refresh(
    store: NotificationStore, token: str, *, scope: str, excluded_ids: Sequence[str] = ()
) -> dict[str, Any]:
    """Recount a frozen selection, never advance its boundary or extend its TTL."""

    def body(conn: sqlite3.Connection) -> dict[str, Any]:
        selection = _token(conn, token, scope, now_ms())
        if selection["op"] is not None:
            raise SelectionError("selection_already_committed")
        filters = json.loads(selection["filters_json"])
        where, params = _where(filters, selection["through_seq"])
        where, params = _exclude(where, params, sorted(set(excluded_ids)))
        aggregates = _aggregates(conn, where, params, filters.get("count_session_ids", []))
        v2 = bool(filters.get("include_ask_reads"))
        if v2:
            bulk_asks.preview_counts(
                aggregates, bulk_asks.valid_rows(conn, token, store.config_dir, excluded_ids)
            )
        return {
            "token": token,
            "through_seq": selection["through_seq"],
            "expires_at": selection["expires_at"],
            **aggregates,
            "skipped_ask_count": 0 if v2 else selection["skipped_ask_count"],
        }

    return store._write(body, "bulk_refresh")


def apply(
    store: NotificationStore,
    op: str,
    *,
    scope: str,
    selection_token: str | None = None,
    ids: Sequence[Mapping[str, Any]] | None = None,
    through_seq: int | None = None,
    excluded_ids: Sequence[str] = (),
    now: int | None = None,
) -> dict[str, Any]:
    """One atomic mutation/receipt, with a bounded response at every cardinality."""
    if op not in ("read", "unread", "dismiss", "undismiss"):
        raise ValueError("invalid bulk operation")
    stamp = now_ms() if now is None else now
    exclusions = sorted(set(excluded_ids))
    request_json = json.dumps({"op": op, "excluded_ids": exclusions}, sort_keys=True)

    def body(conn: sqlite3.Connection) -> dict[str, Any]:
        _housekeep(conn, stamp)
        sessions: list[str] = []
        excluded_count = 0
        v2 = False
        if selection_token is not None:
            selection = _token(conn, selection_token, scope, stamp)
            if selection["op"] is not None:
                if selection["request_json"] != request_json:
                    raise SelectionError("selection_operation_conflict")
                return {**json.loads(selection["result_json"]), "replayed": True}
            token = selection_token
            boundary = int(selection["through_seq"])
            eligible = int(selection["eligible_count"])
            asks = int(selection["skipped_ask_count"])
            filters = json.loads(selection["filters_json"])
            v2 = bool(filters.get("include_ask_reads"))
            sessions = filters.get("count_session_ids", [])
            where, params = _where(filters, boundary)
            if exclusions:
                marks = ",".join("?" for _ in exclusions)
                excluded_count = conn.execute(
                    "SELECT count(*) FROM items WHERE " + where + f" AND id IN ({marks})",
                    [*params, *exclusions],
                ).fetchone()[0]
            eligible -= excluded_count
            if v2:
                selected_asks = len(bulk_asks.snapshots(conn, token, exclusions))
                excluded_asks = asks - selected_asks
                excluded_count += excluded_asks
                eligible -= excluded_asks
                asks = 0 if op in ("read", "unread") else selected_asks
                # All v2 identities were frozen at select-all; choosing dismiss
                # only subtracts asks, never discovers a fresh operation scope.
                eligible -= asks
            where, params = _exclude(where, params, exclusions)
            rows = conn.execute(
                "SELECT " + _COLUMNS + " FROM items WHERE " + where, params
            ).fetchall()
        else:
            if ids is None or through_seq is None:
                raise ValueError("receipt ids require through_seq")
            _room(conn)
            token = secrets.token_urlsafe(32)
            boundary = through_seq
            # Deduplicate before accounting: one gesture cannot mutate a row
            # twice or create two competing snapshots for the same id.
            expected = {str(item["id"]): int(item["revision"]) for item in ids}
            asks = sum(item_id.startswith("ask:") for item_id in expected)
            eligible = len(expected) - asks
            placeholders = ",".join("?" for _ in expected)
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM items WHERE id IN ({placeholders}) AND seq<=?",
                [*expected, boundary],
            ).fetchall()
            rows = [row for row in rows if row["revision"] == expected[row["id"]]]
            conn.execute(
                "INSERT INTO bulk_tokens(token,scope,expires_at,through_seq,eligible_count,"
                "skipped_ask_count) VALUES (?,?,?,?,?,?)",
                (token, scope, stamp + TOKEN_TTL_MS, boundary, eligible, asks),
            )
        changed = [row for row in rows if store.item_changes(row, op)]
        retained = _retained_rows(conn)
        if len(changed) > MAX_ROWS or retained + len(changed) > MAX_ROWS:
            raise SelectionError("receipt_capacity", 429)
        for row in changed:
            before = json.dumps({key: row[key] for key in _STATE})
            post_seq = store.write_item(conn, row, op, stamp, "explicit")
            conn.execute(
                "INSERT INTO bulk_rows(token,item_id,revision,post_seq,before_json) "
                "VALUES (?,?,?,?,?)",
                (token, row["id"], row["revision"], post_seq, before),
            )
        session_deltas = {sid: {"changed_count": 0, "unread_delta": 0} for sid in sessions}
        for row in changed:
            if row["session_id"] in session_deltas:
                entry = session_deltas[row["session_id"]]
                entry["changed_count"] += 1
                entry["unread_delta"] += _delta(row, op)
        answer = {
            "changed_count": len(changed),
            "unread_delta": sum(_delta(row, op) for row in changed),
            "session_counts": session_deltas,
            "excluded_count": excluded_count,
            "skipped_count": eligible - len(changed),
            "skipped_ask_count": asks,
            "eligible_count": eligible,
            "operation_token": token,
            "through_seq": boundary,
            "expires_at": stamp + TOKEN_TTL_MS,
            "replayed": False,
        }
        if v2:
            ask_changes = bulk_asks.apply(conn, store, token, op, stamp, exclusions)
            bulk_asks.result_counts(answer, ask_changes)
            _session_badges(conn, answer)
        conn.execute(
            "UPDATE bulk_tokens SET op=?,result_json=?,expires_at=?,request_json=? WHERE token=?",
            (op, json.dumps(answer), answer["expires_at"], request_json, token),
        )
        return answer

    return store._write(body, "bulk_apply")


def undo(
    store: NotificationStore, token: str, *, scope: str, now: int | None = None
) -> dict[str, Any]:
    """Restore only untouched post-images, atomically and at most once.

    A later content or state write advances seq, including a state ABA cycle,
    and conservatively skipping it is safer than overwriting it. A delivery
    claim deliberately does not advance seq (``store.claim_delivery``); that is
    safe only because Undo restores ``_STATE`` (read/dismiss) columns alone and
    never touches the delivery fields a claim writes.
    """
    stamp = now_ms() if now is None else now

    def body(conn: sqlite3.Connection) -> dict[str, Any]:
        selection = _token(conn, token, scope, stamp)
        if selection["op"] is None:
            raise SelectionError("operation_not_committed")
        if selection["undo_json"] is not None:
            return {**json.loads(selection["undo_json"]), "replayed": True}
        # One indexed join, not a scan for every selected id. Only state fields
        # are restored; content, delivery metadata and the global seq never roll back.
        rows = conn.execute(
            "SELECT r.item_id,r.before_json,i.revision,i.read_rev,i.dismissed_at,i.session_id "
            "FROM bulk_rows r JOIN items i ON i.id=r.item_id "
            "AND i.seq=r.post_seq AND i.revision=r.revision WHERE r.token=?",
            (token,),
        ).fetchall()
        original = json.loads(selection["result_json"])
        session_deltas = {
            sid: {"changed_count": 0, "unread_delta": 0} for sid in original["session_counts"]
        }
        unread_delta = 0
        for row in rows:
            before = json.loads(row["before_json"])
            delta = _badge({**before, "revision": row["revision"]}) - _badge(row)
            unread_delta += delta
            if row["session_id"] in session_deltas:
                session_deltas[row["session_id"]]["changed_count"] += 1
                session_deltas[row["session_id"]]["unread_delta"] += delta
            conn.execute(
                "UPDATE items SET seq=?," + ",".join(f"{key}=?" for key in _STATE) + " WHERE id=?",
                [store._next_seq(conn), *(before[key] for key in _STATE), row["item_id"]],
            )
        answer = {
            "changed_count": len(rows),
            "unread_delta": unread_delta,
            "session_counts": session_deltas,
            "skipped_count": original["changed_count"] - len(rows),
            "skipped_ask_count": 0,
            "eligible_count": original["changed_count"],
            "operation_token": token,
            "through_seq": selection["through_seq"],
            "expires_at": selection["expires_at"],
            "replayed": False,
        }
        if original.get("selection_version") == 2:
            bulk_asks.result_counts(answer, bulk_asks.undo(conn, store, token))
            _session_badges(conn, answer)
        conn.execute(
            "UPDATE bulk_tokens SET undo_json=? WHERE token=?", (json.dumps(answer), token)
        )
        return answer

    return store._write(body, "bulk_undo")

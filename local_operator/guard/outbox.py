"""The durable local queue of Guard decision records (RFC v0.6, slice L1).

WHAT. A small SQLite ledger at ``<config_dir>/guard/outbox.sqlite3`` holding
:class:`~local_operator.guard.decisions.GuardDecisionRecord` rows until a later
slice uploads them as a decision-event batch. It follows
``team_bridge/outbox.py``: its own file with its own lifetime, a
``user_version`` schema generation, short per-call connections, and a refusal
(never a re-create) when a non-empty file carries no generation.

INERT IN L1. Nothing in the product constructs this class yet, and the module
imports no network client: it cannot send anything. Admission (the member's
host-level consent, L2a) is what will install :class:`OutboxDecisionSink` and
a drainer. Construction does no I/O, so even a caller that builds one but never
appends leaves no file behind -- the "no artefact for non-Team installs" rule.

IDEMPOTENT APPEND. ``record_id`` is the primary key. Appending the same record
twice is a no-op; appending a DIFFERENT record under an existing id is refused
(``record_conflict``), because silently keeping either would make the upload's
idempotency key mean two things.

BOUNDED. Rows are capped at :data:`MAX_OUTBOX_ROWS`. The bound is derived, not
guessed: the ``decisions`` family releases ``outcome x rule_class`` cells
(4 x 4 = 16) and counts at most c = 20 units per contributor per cell per day
(RFC §10'), so one install can contribute at most 320 countable units per day;
a decision batch lives at most 72 h (§9.2', 3 days), so 960 rows is the most
that can ever matter. 2048 leaves 2x headroom. At the cap the outbox first
drops what can no longer count (expired rows, already-sent rows); if it is
still full the NEW record is dropped -- the oldest live rows are the ones
closest to upload, and a capped install already exceeds what the grid counts.
"""

from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .decisions import GuardDecisionRecord

#: Our own schema generation; bump only for additive migrations.
#: 2 (slice L3) adds ``outbox_meta.high_water``: ``seq`` was ``MAX(seq)+1``,
#: which REUSES numbers once sent rows are pruned, so a per-tenant floor
#: ("send only what was recorded after this grant") could hide new rows
#: forever. A stored high-water mark never rewinds. Version 1 files migrate
#: in place (one additive table, seeded from ``MAX(seq)``).
GUARD_OUTBOX_SCHEMA_VERSION: Final = 2
_MIGRATABLE_FROM: Final = 1

#: See the module docstring for the derivation (16 cells x c=20 x 3 days, x2).
MAX_OUTBOX_ROWS: Final = 2048

#: Decision-event batches have a 72 h queue TTL (RFC §9.2'); a record whose
#: UTC day is more than this many days before today can no longer be accepted.
DECISION_TTL_DAYS: Final = 3

#: Bound on how long a writer waits for another lop process holding the file.
_BUSY_TIMEOUT_S: Final = 5.0

ROW_STATES: Final = ("pending", "sent")


class GuardOutboxRefused(Exception):
    """The outbox cannot serve this request (damage, conflicting record)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def guard_outbox_path(config_dir: Path | str) -> Path:
    """The outbox's own file, beside (not inside) any other store."""

    return Path(config_dir) / "guard" / "outbox.sqlite3"


@dataclass(frozen=True)
class AppendResult:
    """``stored``: a new row; ``duplicate``: the same record was already there;
    ``dropped``: the outbox was full of rows that still count."""

    status: str


def _today() -> _dt.date:
    return _dt.datetime.now(_dt.UTC).date()


class GuardOutbox:
    """The durable decision ledger. Every method opens its own short connection."""

    def __init__(self, config_dir: Path | str, *, max_rows: int = MAX_OUTBOX_ROWS) -> None:
        # No I/O here: see "INERT IN L1" above.
        self._path = guard_outbox_path(config_dir)
        self._max_rows = max_rows

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Re-asserted on EVERY open, not only at creation (L1 review MINOR-3):
        # a directory made by an older build, a restore or a umask 022 shell
        # would otherwise stay world-listable / the file world-readable.
        os.chmod(self._path.parent, 0o700)
        existed = self._path.exists() and self._path.stat().st_size > 0
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        try:
            if existed:
                os.chmod(self._path, 0o600)
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            except sqlite3.DatabaseError as exc:
                # A file that is not SQLite at all is damage too (L1 review
                # NIT-2): the same refusal, never a raw DatabaseError.
                raise GuardOutboxRefused(
                    "outbox_damaged", "the Guard outbox is not a database; refusing it"
                ) from exc
            if existed and version not in (GUARD_OUTBOX_SCHEMA_VERSION, _MIGRATABLE_FROM):
                # Re-running DDL over an unknown file is not repair (house rule,
                # team_bridge/outbox.py); a newer generation is not ours to edit.
                raise GuardOutboxRefused(
                    "outbox_damaged",
                    "the Guard outbox has an unknown schema generation; refusing it",
                )
            if not version:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS decision_records ("
                    " record_id TEXT PRIMARY KEY,"
                    " schema TEXT NOT NULL,"
                    " day TEXT NOT NULL,"
                    " outcome TEXT NOT NULL,"
                    " rule_class TEXT NOT NULL,"
                    " tool_class TEXT NOT NULL,"
                    " state TEXT NOT NULL DEFAULT 'pending',"
                    " seq INTEGER NOT NULL"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS decision_records_state"
                    " ON decision_records (state, seq)"
                )
                conn.execute(f"PRAGMA user_version = {_MIGRATABLE_FROM}")
                version = _MIGRATABLE_FROM
                conn.commit()
                # Enum-only content, but still the member's activity: owner-only.
                os.chmod(self._path, 0o600)
            if version == _MIGRATABLE_FROM:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute("PRAGMA user_version").fetchone()[0] == _MIGRATABLE_FROM:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS outbox_meta ("
                        " id INTEGER PRIMARY KEY CHECK (id = 1),"
                        " high_water INTEGER NOT NULL)"
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO outbox_meta (id, high_water)"
                        " SELECT 1, COALESCE(MAX(seq), 0) FROM decision_records"
                    )
                    conn.execute(f"PRAGMA user_version = {GUARD_OUTBOX_SCHEMA_VERSION}")
                conn.commit()
        except BaseException:
            conn.close()
            raise
        return conn

    def append(self, record: GuardDecisionRecord, *, today: _dt.date | None = None) -> AppendResult:
        """Store ``record`` durably (committed before return); idempotent on ``record_id``."""
        conn = self._connect()
        try:
            # BEGIN IMMEDIATE: the room check and the insert must see the same
            # row count even when two lop processes append at once.
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM decision_records WHERE record_id = ?", (record.record_id,)
            ).fetchone()
            if existing is not None:
                conn.rollback()
                if _row_to_record(existing) != record:
                    raise GuardOutboxRefused(
                        "record_conflict", "a different record already holds this record_id"
                    )
                return AppendResult("duplicate")
            if not self._make_room(conn, today or _today()):
                conn.rollback()
                return AppendResult("dropped")
            conn.execute("UPDATE outbox_meta SET high_water = high_water + 1 WHERE id = 1")
            seq = conn.execute("SELECT high_water FROM outbox_meta WHERE id = 1").fetchone()[0]
            conn.execute(
                "INSERT INTO decision_records"
                " (record_id, schema, day, outcome, rule_class, tool_class, state, seq)"
                " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    record.record_id,
                    record.schema,
                    record.day,
                    record.outcome,
                    record.rule_class,
                    record.tool_class,
                    seq,
                ),
            )
            conn.commit()
            return AppendResult("stored")
        finally:
            conn.close()

    def _make_room(self, conn: sqlite3.Connection, today: _dt.date) -> bool:
        count = conn.execute("SELECT COUNT(*) FROM decision_records").fetchone()[0]
        if count < self._max_rows:
            return True
        _delete_spent(conn, today)
        count = conn.execute("SELECT COUNT(*) FROM decision_records").fetchone()[0]
        return count < self._max_rows

    def pending(self, limit: int = 500) -> list[GuardDecisionRecord]:
        """Unsent records, oldest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM decision_records WHERE state = 'pending' ORDER BY seq LIMIT ?",
                (limit,),
            ).fetchall()
            return [_row_to_record(row) for row in rows]
        finally:
            conn.close()

    def pending_after(
        self, seq_floor: int, *, before_day: str, limit: int = 500
    ) -> list[tuple[int, GuardDecisionRecord]]:
        """``(seq, record)`` for unsent rows past ``seq_floor`` on CLOSED days, oldest first.

        Slice L3's uploader reads through this. ``seq_floor`` is the outbox's
        high-water mark when a tenant's current consent was first seen live, so
        a queue built under one authority is never sent under a later grant
        (RFC §22: "must not retry an old queue under newly granted authority").
        ``before_day`` keeps the open UTC day local: v0.3 §7 sends a day only
        after it closes, which also keeps intra-day timing off the wire.
        """
        if not self._path.exists():
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM decision_records WHERE state = 'pending' AND seq > ?"
                " AND day < ? ORDER BY seq LIMIT ?",
                (seq_floor, before_day, limit),
            ).fetchall()
            return [(int(row["seq"]), _row_to_record(row)) for row in rows]
        finally:
            conn.close()

    def max_seq(self) -> int:
        """The high-water mark (0 for an absent or empty outbox). Creates no file."""
        if not self._path.exists():
            return 0
        conn = self._connect()
        try:
            return int(
                conn.execute("SELECT high_water FROM outbox_meta WHERE id = 1").fetchone()[0]
            )
        finally:
            conn.close()

    def purge(self) -> int:
        """Drop every row (a revoke or ``namespace_reset``). Creates no file."""
        if not self._path.exists():
            return 0
        conn = self._connect()
        try:
            removed = conn.execute("DELETE FROM decision_records").rowcount
            conn.commit()
            return removed
        finally:
            conn.close()

    def mark_sent(self, record_ids: Iterable[str]) -> int:
        """Settle records the server confirmed; returns how many moved out of pending."""
        ids = list(record_ids)
        if not ids:
            return 0
        conn = self._connect()
        try:
            moved = conn.executemany(
                "UPDATE decision_records SET state = 'sent'"
                " WHERE record_id = ? AND state = 'pending'",
                [(rid,) for rid in ids],
            ).rowcount
            conn.commit()
            return moved
        finally:
            conn.close()

    def prune(self, *, today: _dt.date | None = None) -> int:
        """Delete sent rows and rows past the 72 h queue TTL; returns rows removed."""
        conn = self._connect()
        try:
            removed = _delete_spent(conn, today or _today())
            conn.commit()
            return removed
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM decision_records").fetchone()[0]
        finally:
            conn.close()


def _delete_spent(conn: sqlite3.Connection, today: _dt.date) -> int:
    cutoff = (today - _dt.timedelta(days=DECISION_TTL_DAYS)).isoformat()
    # ISO dates compare correctly as text.
    return conn.execute(
        "DELETE FROM decision_records WHERE state = 'sent' OR day < ?", (cutoff,)
    ).rowcount


def _row_to_record(row: sqlite3.Row) -> GuardDecisionRecord:
    # Re-validated through the record's own constructor: a row edited on disk
    # to carry a non-enum value is refused here rather than uploaded later.
    return GuardDecisionRecord(
        outcome=row["outcome"],
        rule_class=row["rule_class"],
        tool_class=row["tool_class"],
        day=row["day"],
        record_id=row["record_id"],
        schema=row["schema"],
    )


class OutboxDecisionSink:
    """A decision sink that appends to a :class:`GuardOutbox`.

    Not installed anywhere in L1; L2a installs it once the install is
    admitted. ``emit_decision`` already swallows its failures.
    """

    def __init__(self, outbox: GuardOutbox) -> None:
        self._outbox = outbox

    def emit(self, record: GuardDecisionRecord) -> None:
        self._outbox.append(record)


__all__ = [
    "DECISION_TTL_DAYS",
    "GUARD_OUTBOX_SCHEMA_VERSION",
    "MAX_OUTBOX_ROWS",
    "ROW_STATES",
    "AppendResult",
    "GuardOutbox",
    "GuardOutboxRefused",
    "OutboxDecisionSink",
    "guard_outbox_path",
]

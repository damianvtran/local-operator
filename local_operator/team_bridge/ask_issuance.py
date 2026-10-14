"""Durable issuance mapping for bridge request asks (H5a / F3 — the W1 basis).

WHY THIS EXISTS. The wake that asks a human about a screened bridge request can
be re-delivered: the connector's claim lease expires, a reconcile re-walks the
feed, a process dies between the row and the ask. Every one of those paths must
find the SAME answer — "this message id was already raised as ask X, or its
issuance is durably PENDING (open-ask-cap / no queue), never silently dropped
and never a second ask" — and a trusted local boundary may retry a PENDING
issuance as an ASK only, never a decide. A volatile in-memory check loses that
survives-restart fact at the first crash, which is exactly when it matters.

The row carries the ORIGINAL scope block (S-3, ``scope_from_runtime`` captured
at ISSUANCE): a retry later can therefore never attach the CURRENT binding to
an old screened request — the scope a human is asked under is the scope frozen
when the wake was validated, and a same-account generation change REJECTS the
old row instead of rebinding it (the generation is part of the identity).

Plain sqlite because the rest of this package's stores are, and because this
record must be written SYNCHRONOUSLY on the receive hook's stack, before any
await — the same discipline the decide-intent ledger keeps.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

from .store import _owner_key


def bridge_scope_json(scope: Mapping[str, Any]) -> str:
    """The ONE canonical JSON of a bridge scope block (R1-5, PR #2378).

    The keyed queue door (``asks/queue.py``), the session's tri-state lookup
    and this ledger's stored ``scope_json`` all normalize through HERE, so a
    future divergence between the three is a test failure — not a silent
    "scope mismatch" refusal on an honest retry.
    """

    return json.dumps(dict(scope), separators=(",", ":"), sort_keys=True)


def bridge_scope_digest(scope: Mapping[str, Any]) -> str:
    """Canonical digest of a bridge scope block (identity comparison only)."""

    return hashlib.sha256(bridge_scope_json(scope).encode("utf-8")).hexdigest()

#: The issuance row's lifecycle. ``claimed`` means the durable record exists
#: and the enqueue attempt is in flight or was interrupted (a retry may still
#: look the ask up in the ask log); ``raised`` is terminal until the ask is
#: answered or the row is settled; ``pending`` is a durable, VISIBLE "not
#: asked yet" the trusted boundary may retry.
STATE_CLAIMED = "claimed"
STATE_RAISED = "raised"
STATE_PENDING = "pending"
#: The tri-state lookup's HOLD (S-10): the ask log could not be read, or a
#: hit was ambiguous — NEVER permission to enqueue. The trusted boundary
#: re-checks and only a CONFIRMED-ABSENT lookup proceeds to enqueue.
STATE_HELD = "held"
#: Legacy/missing/invalid producer envelope: visible, never retried into an
#: ask (no current-scope backfill, no arrival mint).
STATE_UNRECOVERABLE = "unrecoverable"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ask_issuance (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    state TEXT NOT NULL,
    ask_id TEXT NOT NULL DEFAULT '',
    issuance_id TEXT NOT NULL DEFAULT '',
    delivery_ref TEXT NOT NULL DEFAULT '',
    scope_json TEXT NOT NULL DEFAULT '',
    sender TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    -- RETIRED as a content column (M1, §13.9.10/§12.1.3): the ledger carries
    -- references/actors only; the column is kept for schema continuity and is
    -- never written with third-party body text.
    excerpt TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    local_row_id TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL,
    PRIMARY KEY (owner_key, message_id, generation)
)
"""


class RowIdRequired(ValueError):
    """A NEW issuance row was refused because no trusted local row id came.

    The atomic rule: the mapping rides the same INSERT that creates the row,
    so a fresh row can never be committed UNMAPPED. Existing (legacy) rows
    are returned as-is — the display path then HOLDS — but a new row with no
    trusted actual local id refuses before any INSERT (the transaction rolls
    back; nothing is left behind).
    """


class AskIssuance:
    """One sqlite file under the config dir; same-owner single-writer per row."""

    def __init__(self, config_dir: str | Path) -> None:
        self._path = Path(config_dir) / "team_bridge" / "ask_issuance.db"

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute(_SCHEMA)
        # Additive, guarded migration for stores created before the local-row
        # binding existed: the column carries the write-once mapping from the
        # server message id to the LOCAL transcript row id (two id domains:
        # ``bmsg…`` selects the bridge message; the transcript row is keyed by
        # the accepted ``peer-<32hex>`` id the receiver stamped).
        try:
            conn.execute(
                "ALTER TABLE ask_issuance ADD COLUMN local_row_id TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError as failure:
            # ONLY the duplicate-column indication is the benign "already
            # migrated" case; a locked/full/IO failure is NOT swallowed — it
            # propagates so the caller fails closed instead of running against
            # a store missing the column.
            if "duplicate column name" not in str(failure).lower():
                raise
        return conn

    def generation_is_live(self, owner: Any) -> int:
        """The owner's CURRENT generation, or ``0`` when it has none."""

        return int(getattr(owner, "enrollment_generation", 0) or 0)

    def read(self, owner: Any, message_id: str) -> dict[str, Any] | None:
        """The issuance row for this message — ANY generation — or ``None``.

        The caller distinguishes generations itself: a row under a different
        generation is not this generation's to retry (no silent rebind), and
        the row's stored scope is the only scope a retry may reuse.
        """

        key = _owner_key(owner)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM ask_issuance WHERE owner_key = ? AND message_id = ? "
                    "ORDER BY generation DESC LIMIT 1",
                    (key, str(message_id)),
                ).fetchone()
                return dict(row) if row is not None else None
        except sqlite3.Error:
            return None

    def local_row_id(self, owner: Any, message_id: str) -> str:
        """The durable LOCAL transcript row id bound to this server message id.

        Empty string = NO mapping (legacy rows included): the caller HOLDS —
        no same-id guess, no newest-row fallback, no backfill.
        """

        key = _owner_key(owner)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT local_row_id FROM ask_issuance WHERE owner_key = ? "
                    "AND message_id = ? ORDER BY generation DESC LIMIT 1",
                    (key, str(message_id)),
                ).fetchone()
                return str(row["local_row_id"] or "") if row is not None else ""
        except sqlite3.Error:
            return ""

    def claim(
        self,
        owner: Any,
        message_id: str,
        scope: Mapping[str, Any],
        *,
        sender: str = "",
        summary: str = "",
        delivery_ref: str = "",
        local_row_id: str,
    ) -> dict[str, Any]:
        """Get or create the durable issuance row (generation-scoped).

        Written synchronously BEFORE any enqueue: a kill between this and the
        enqueue leaves ``claimed`` — a state the retry path resolves through
        the TRI-STATE ask-log lookup (found-exact / confirmed-absent /
        unavailable) rather than minting a duplicate. ``scope`` is the
        validated STORED original block, frozen here once and reused verbatim
        by every retry (S-3); the ask's sender/summary references are stored
        beside it so a retry never re-derives them from a mutable source —
        body text never enters this ledger (M1, §13.9.10/§12.1.3); and
        ``issuance_id`` (minted HERE, never by a model/UI caller) is the
        preallocated durable identity the crash windows are reasoned about in.
        """

        import uuid as _uuid

        key = _owner_key(owner)
        generation = self.generation_is_live(owner)
        scope_json = bridge_scope_json(scope)
        now = time.time()
        local_id = str(local_row_id or "")
        with self._connect() as conn:
            if not local_id:
                # A NEW row may NEVER be committed UNMAPPED: an EXISTING
                # (legacy) row is returned unchanged — the display path then
                # HOLDS — but a would-be INSERT without a trusted actual local
                # id refuses here, before any write.
                existing_row = conn.execute(
                    "SELECT 1 FROM ask_issuance WHERE owner_key = ? AND message_id = ? "
                    "AND generation = ?",
                    (key, str(message_id), generation),
                ).fetchone()
                if existing_row is None:
                    raise RowIdRequired("local_row_id_required")
            # ATOMIC: the ACTUAL trusted local transcript row id (when the
            # receive path supplies one) rides the INITIAL INSERT inside this
            # one transaction — the row is never durably committed UNMAPPED,
            # so there is no pre-insert window to close. A redelivery
            # (INSERT OR IGNORE no-op) keeps the FIRST mapping untouched; a
            # legacy row claimed before the column existed stays empty and
            # the display path HOLDS.
            conn.execute(
                "INSERT OR IGNORE INTO ask_issuance "
                "(owner_key, message_id, generation, state, scope_json, sender, "
                "summary, delivery_ref, issuance_id, local_row_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    str(message_id),
                    generation,
                    STATE_CLAIMED,
                    scope_json,
                    str(sender)[:200],
                    str(summary)[:500],
                    str(delivery_ref),
                    _uuid.uuid4().hex,
                    local_id,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM ask_issuance WHERE owner_key = ? AND message_id = ? "
                "AND generation = ?",
                (key, str(message_id), generation),
            ).fetchone()
            conn.execute(
                "UPDATE ask_issuance SET attempts = attempts + 1, updated_at = ? "
                "WHERE owner_key = ? AND message_id = ? AND generation = ?",
                (now, key, str(message_id), generation),
            )
        return dict(row) if row is not None else {}

    def mark_raised(self, owner: Any, message_id: str, ask_id: str) -> None:
        """Record the RAISED ask id — terminal for re-issuance of this message."""

        key = _owner_key(owner)
        generation = self.generation_is_live(owner)
        with self._connect() as conn:
            conn.execute(
                "UPDATE ask_issuance SET state = ?, ask_id = ?, reason = '', updated_at = ? "
                "WHERE owner_key = ? AND message_id = ? AND generation = ?",
                (STATE_RAISED, str(ask_id), time.time(), key, str(message_id), generation),
            )

    def mark_pending(self, owner: Any, message_id: str, reason: str) -> None:
        """A durable, visible "not asked yet" — never a silent drop."""

        key = _owner_key(owner)
        generation = self.generation_is_live(owner)
        with self._connect() as conn:
            conn.execute(
                "UPDATE ask_issuance SET state = ?, reason = ?, updated_at = ? "
                "WHERE owner_key = ? AND message_id = ? AND generation = ?",
                (STATE_PENDING, str(reason)[:200], time.time(), key, str(message_id), generation),
            )

    def mark_held(self, owner: Any, message_id: str, reason: str) -> None:
        """The lookup was UNAVAILABLE or AMBIGUOUS (S-10): hold, never enqueue.

        A held row is retried at the trusted boundary — the lookup is
        re-run FIRST, and only a CONFIRMED-ABSENT answer proceeds to enqueue.
        """

        key = _owner_key(owner)
        generation = self.generation_is_live(owner)
        with self._connect() as conn:
            conn.execute(
                "UPDATE ask_issuance SET state = ?, reason = ?, updated_at = ? "
                "WHERE owner_key = ? AND message_id = ? AND generation = ?",
                (STATE_HELD, str(reason)[:200], time.time(), key, str(message_id), generation),
            )

    def mark_unrecoverable(self, owner: Any, message_id: str, reason: str) -> None:
        """Legacy/missing/invalid producer envelope: visible, never retried.

        No current-scope backfill and no arrival mint — the row exists so the
        operator can SEE that a bridge arrival could not be authorized.
        """

        key = _owner_key(owner)
        generation = self.generation_is_live(owner)
        with self._connect() as conn:
            conn.execute(
                "UPDATE ask_issuance SET state = ?, reason = ?, updated_at = ? "
                "WHERE owner_key = ? AND message_id = ? AND generation = ?",
                (STATE_UNRECOVERABLE, str(reason)[:200], time.time(), key, str(message_id), generation),
            )

    def pending(self, owner: Any) -> list[dict[str, Any]]:
        """Rows this generation may still retry (ASK only; never a decide)."""

        key = _owner_key(owner)
        generation = self.generation_is_live(owner)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM ask_issuance WHERE owner_key = ? AND generation = ? "
                    "AND state IN (?, ?, ?) ORDER BY updated_at",
                    (key, generation, STATE_CLAIMED, STATE_PENDING, STATE_HELD),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []


__all__ = [
    "AskIssuance",
    "STATE_CLAIMED",
    "STATE_HELD",
    "STATE_PENDING",
    "STATE_RAISED",
    "STATE_UNRECOVERABLE",
    "bridge_scope_digest",
    "bridge_scope_json",
]

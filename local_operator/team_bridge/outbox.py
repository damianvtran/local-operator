"""H5c — the durable send outbox: write-intent-BEFORE-dispatch (§6.2, §13.5).

WHY A SECOND LEDGER. The bridge send carries its idempotency key in the BODY
(``client_ref``, §6.2's stated deviation from the house header vocabulary) and
"the client retries ONLY with the same ``client_ref``" (§13.5) — so the ref
must survive a crash between minting and dispatch, or the retry after restart
is a NEW message. This module mirrors ``ack_intents``' discipline exactly
(write-once intent recorded before any I/O, replay of the SAME tuple, a
terminal state recorded after): an intent row is written first, the dispatch
uses the row's ref verbatim, and only a confirmed server answer moves the row
out of ``pending``.

WHY ITS OWN FILE rather than the native store's schema. The native store's
schema is versioned and damage-guarded by H3's invariants (V5, additive-only,
row preservation); the outbox is H1/H5 client-side bookkeeping with an
independent lifetime (a re-enrolled or disabled account still has outbound
intents) and no cross-table transactions with screening. Keeping it at
``<config_dir>/team_bridge/outbox.sqlite3`` keeps both guarantees intact.

DAMAGE POLICY, proportionate: a non-empty database without this module's
``user_version`` is REFUSED (never re-created — the house rule that re-running
DDL is not repair), and only then is the table created.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import BridgeRefusal, BridgeUnreachable, TeamBridgeClient

#: Our own schema generation; bump only for additive migrations.
OUTBOX_SCHEMA_VERSION = 1

#: The three states an intent can hold. ``pending`` is the retry pool.
INTENT_STATES = ("pending", "sent", "refused")


class OutboxRefused(Exception):
    """The outbox cannot serve this request (damage, unknown intent, bad state)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def outbox_db_path(config_dir: Path | str) -> Path:
    """The outbox lives beside the native store, in its own file."""

    return Path(config_dir) / "team_bridge" / "outbox.sqlite3"


@dataclass(frozen=True)
class SendIntent:
    """One durable send intent — the tuple every retry replays verbatim."""

    intent_id: str
    owner_key: str
    client_ref: str
    to: dict[str, str]
    type: str
    text: str
    thread_id: str | None
    expires_at: str | None
    state: str
    message_id: str | None
    error_code: str | None


def _row_to_intent(row: sqlite3.Row) -> SendIntent:
    import json

    return SendIntent(
        intent_id=row["intent_id"],
        owner_key=row["owner_key"],
        client_ref=row["client_ref"],
        to=dict(json.loads(row["to_json"])),
        type=row["type"],
        text=row["text"],
        thread_id=row["thread_id"],
        expires_at=row["expires_at"],
        state=row["state"],
        message_id=row["message_id"],
        error_code=row["error_code"],
    )


class Outbox:
    """The durable send ledger. Every method opens its own short connection."""

    def __init__(self, config_dir: Path | str) -> None:
        self._path = outbox_db_path(config_dir)

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existed = self._path.exists() and self._path.stat().st_size > 0
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if existed and not version:
                raise OutboxRefused(
                    "outbox_damaged",
                    "a non-empty outbox database has no schema generation; refusing "
                    "authority (re-running DDL is not repair)",
                )
            if not version:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS send_intents ("
                    " intent_id TEXT PRIMARY KEY,"
                    " owner_key TEXT NOT NULL,"
                    " client_ref TEXT NOT NULL,"
                    " to_json TEXT NOT NULL,"
                    " type TEXT NOT NULL,"
                    " text TEXT NOT NULL,"
                    " thread_id TEXT, expires_at TEXT,"
                    " state TEXT NOT NULL,"
                    " message_id TEXT, error_code TEXT,"
                    " created_at TEXT NOT NULL DEFAULT (datetime('now')),"
                    " UNIQUE (owner_key, client_ref)"
                    ")"
                )
                conn.execute(f"PRAGMA user_version = {OUTBOX_SCHEMA_VERSION}")
                conn.commit()
        finally:
            pass
        return conn

    def record_send(
        self,
        owner_key: str,
        *,
        to: dict[str, str],
        type: str,
        text: str,
        thread_id: str | None = None,
        expires_at: str | None = None,
    ) -> SendIntent:
        """Record the intent and mint its ``client_ref`` — BEFORE any dispatch.

        The ref is minted HERE, once, and persists: a restart between record
        and dispatch finds the row in :meth:`pending` and retries with the
        SAME ref, which is the idempotency the server deduplicates on.
        """
        import json

        intent_id = str(uuid.uuid4())
        client_ref = str(uuid.uuid4())
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO send_intents (intent_id, owner_key, client_ref, to_json,"
                " type, text, thread_id, expires_at, state)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                (
                    intent_id,
                    owner_key,
                    client_ref,
                    json.dumps(to, sort_keys=True),
                    type,
                    text,
                    thread_id,
                    expires_at,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM send_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            return _row_to_intent(row)
        finally:
            conn.close()

    def pending(self, owner_key: str) -> list[SendIntent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM send_intents WHERE owner_key = ? AND state = 'pending'"
                " ORDER BY created_at, intent_id",
                (owner_key,),
            ).fetchall()
            return [_row_to_intent(row) for row in rows]
        finally:
            conn.close()

    def confirm(self, intent_id: str, message_id: str) -> None:
        self._settle(intent_id, state="sent", message_id=message_id, error_code=None)

    def refuse(self, intent_id: str, error_code: str) -> None:
        self._settle(intent_id, state="refused", message_id=None, error_code=error_code)

    def _settle(
        self,
        intent_id: str,
        *,
        state: str,
        message_id: str | None,
        error_code: str | None,
    ) -> None:
        if state not in INTENT_STATES:
            raise OutboxRefused("invalid_state", f"unknown intent state {state!r}")
        conn = self._connect()
        try:
            current = conn.execute(
                "SELECT state FROM send_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            if current is None:
                raise OutboxRefused("unknown_intent", f"no send intent {intent_id!r}")
            if current["state"] != "pending":
                raise OutboxRefused(
                    "intent_already_settled",
                    f"send intent {intent_id!r} is already {current['state']!r} "
                    "(write-once: a settled intent is never re-settled)",
                )
            conn.execute(
                "UPDATE send_intents SET state = ?, message_id = ?, error_code = ?"
                " WHERE intent_id = ?",
                (state, message_id, error_code, intent_id),
            )
            conn.commit()
        finally:
            conn.close()


async def send_from_outbox(
    client: TeamBridgeClient,
    outbox: Outbox,
    intent: SendIntent,
    *,
    tenant_id: str,
    expected_account_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch ONE recorded intent, settling it on a definitive outcome.

    Outcomes: a server answer confirms the intent (the message id is kept);
    a SERVER REFUSAL settles it ``refused`` with the code — retrying a refused
    send would repeat a rejected request, and §6.2's replay rule covers only
    identical retries, which a refusal is not. An UNREACHABLE transport raises
    and leaves the intent ``pending``: that is the one retryable outcome, and
    the retry reuses this row's ``client_ref`` by construction.
    """
    try:
        body = await client.send(
            tenant_id=tenant_id,
            to=intent.to,
            type=intent.type,
            text=intent.text,
            client_ref=intent.client_ref,
            thread_id=intent.thread_id,
            expires_at=intent.expires_at,
            expected_account_id=expected_account_id,
        )
    except BridgeUnreachable:
        # Nothing answered — the ONE retryable outcome. The intent stays
        # PENDING by construction (we simply do not settle it), and the next
        # attempt replays this row's client_ref.
        raise
    except BridgeRefusal as refusal:
        # A server refusal is a verdict, not a transport hiccup: settle the
        # intent so the retry pool never re-sends a rejected request.
        outbox.refuse(intent.intent_id, refusal.code)
        raise
    message = body.get("message") if isinstance(body, dict) else None
    message_id = str((message or {}).get("id") or "") or None
    outbox.confirm(intent.intent_id, message_id or "")
    return body


__all__ = [
    "INTENT_STATES",
    "OUTBOX_SCHEMA_VERSION",
    "Outbox",
    "OutboxRefused",
    "SendIntent",
    "outbox_db_path",
    "send_from_outbox",
]

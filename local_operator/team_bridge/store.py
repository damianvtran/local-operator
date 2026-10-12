"""The team-bridge metadata store: one small stdlib SQLite database (H2).

WHAT THIS IS. Local operational metadata for the H2 primitives: the committed
owner, sync checkpoints, delivery refs, write-once ack intents and claim
leases. It is deliberately NOT a session store, NOT a message cache, and NOT
a wake/admission source. The contract (§13.9) permits operational metadata
outside the session stores and forbids a body cache; the H2 handoff's privacy
list is enforced here by SCHEMA + WHITELIST: there is no column for a body,
excerpt, summary, sender name/email, model reason, token or raw error.

WHY SQLITE. The alternative (one JSON document + lock + fsync) needs a
whole-document rewrite to keep refs + checkpoint + ack intent + claim CAS
consistent; SQLite transactions are the smaller correctness mechanism, and it
is an existing dependency-free pattern in this codebase
(``providers/auth_store.py``). Every mutation runs inside ONE transaction;
the connection uses an explicit bounded busy timeout and refuses to run
unlocked on contention.

FAIL-CLOSED LAW. Missing, corrupt, or unsupported-schema state means **no
authority**: :func:`Store.active_owner` returns ``None`` and the caller does
not proceed — it never re-enrolls from a generic preference. Every mutation
checks the live generation inside the same transaction, so a result that
returns after :meth:`Store.invalidate` cannot commit under the new run.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from local_operator.team_bridge.types import (
    ACK_TARGETS,
    AckTarget,
    FeedSeq,
    InboxCursor,
    MessageRef,
    OwnerSnapshot,
    ScreeningReceipt,
)

__all__ = [
    "ACK_PENDING",
    "ACK_CONFIRMED",
    "ACK_CONFLICT",
    "SCHEMA_VERSION",
    "Store",
    "StoreRefused",
    "state_db_path",
]

SCHEMA_VERSION = 1

#: Ack intent lifecycle: exactly one of these states, no others.
ACK_PENDING = "pending"
ACK_CONFIRMED = "confirmed"
ACK_CONFLICT = "conflict"

#: A contention bound, not a retry loop: SQLite waits up to this long for a
#: writer lock, then refuses. Callers are one-shot primitives (H4 owns any
#: scheduling), so waiting forever would stall a turn.
_BUSY_TIMEOUT_MS = 2_000


class StoreRefused(RuntimeError):
    """A store mutation refused. ``code`` is a stable machine-readable tag."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def state_db_path(config_dir: Path) -> Path:
    """The one database path: ``<config_dir>/team_bridge/state.sqlite3``."""

    return Path(config_dir) / "team_bridge" / "state.sqlite3"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS enrollments (
    owner_key TEXT PRIMARY KEY,
    credential_id INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    canonical_base_url TEXT NOT NULL,
    enrollment_generation INTEGER NOT NULL,
    consent_version TEXT,
    consent_ref TEXT,
    designated_session TEXT,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sync (
    owner_key TEXT PRIMARY KEY,
    feed_seq INTEGER NOT NULL DEFAULT 0,
    pending_resync INTEGER NOT NULL DEFAULT 0,
    inbox_checkpoint TEXT,
    walk_generation INTEGER NOT NULL DEFAULT 0,
    last_observed_at REAL
);
CREATE TABLE IF NOT EXISTS deliveries (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    seq INTEGER,
    msg_type TEXT,
    origin TEXT,
    sender_account_id TEXT,
    sender_endpoint_id TEXT,
    recipient_account_id TEXT,
    recipient_endpoint_id TEXT,
    body_hash TEXT,
    delivery_state TEXT,
    state_at TEXT,
    request_state TEXT,
    request_expires_at TEXT,
    request_ref TEXT,
    pending_reconciliation INTEGER NOT NULL DEFAULT 0,
    local_hold_code TEXT,
    retry_at REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    blocked_reviewed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (owner_key, message_id)
);
CREATE TABLE IF NOT EXISTS ack_intents (
    intent_id TEXT PRIMARY KEY,
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    target TEXT NOT NULL,
    client_ref TEXT NOT NULL,
    verdict TEXT,
    model TEXT,
    prompt_version TEXT,
    template_sha256 TEXT,
    body_sha256 TEXT,
    categories TEXT,
    state TEXT NOT NULL,
    error_code TEXT,
    retry_after REAL,
    created_at REAL NOT NULL,
    UNIQUE (owner_key, message_id, target)
);
CREATE TABLE IF NOT EXISTS feed_refs (
    owner_key TEXT NOT NULL,
    ref TEXT NOT NULL,
    seq INTEGER NOT NULL,
    PRIMARY KEY (owner_key, ref)
);
CREATE TABLE IF NOT EXISTS claims (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    token TEXT NOT NULL,
    generation INTEGER NOT NULL,
    expires_at REAL NOT NULL,
    state TEXT NOT NULL,
    PRIMARY KEY (owner_key, message_id)
);
"""


def _owner_key(owner: OwnerSnapshot) -> str:
    """The namespace key: hub origin + account + tenant + endpoint.

    JSON rather than a delimiter join: ids can contain any printable
    character, and a joined key would make two different owners collide on
    one crafted id (the handoff's composite-namespace requirement).
    """

    return json.dumps(
        [
            owner.canonical_base_url,
            owner.account_id,
            owner.tenant_id,
            owner.endpoint_id,
        ],
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class OwnerReceipt:
    """The server's consent receipt reference — non-secret, authority-bearing.

    H4 invokes :meth:`Store.commit_enrollment` ONLY after the real
    captured-owner consent flow produced this. It carries the version/ref the
    server consented on, never a token or payload.
    """

    consent_version: str
    consent_ref: str


class Store:
    """Transactional metadata access. Construct per operation, not long-lived."""

    def __init__(self, config_dir: Path) -> None:
        self._path = state_db_path(config_dir)

    # -- connection / schema -------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_MS / 1000.0)
        try:
            try:
                conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
            except sqlite3.Error as error:
                # A file that is not a database is NO AUTHORITY (fail-closed):
                # typed refusal, never a partial read or a fresh enrollment.
                raise StoreRefused("state_unreadable", str(error)) from error
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn
        finally:
            conn.close()

    def _init(self, conn: sqlite3.Connection) -> None:
        try:
            conn.executescript(_SCHEMA)
        except sqlite3.Error as error:
            raise StoreRefused("state_unreadable", str(error)) from error
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            return
        if row["value"] != str(SCHEMA_VERSION):
            # Unsupported schema is NO AUTHORITY, not a migration: nothing in
            # H2 may guess at a future/unknown layout's semantics.
            raise StoreRefused(
                "schema_unsupported",
                f"state.sqlite3 schema {row['value']!r} != {SCHEMA_VERSION}",
            )

    def _live_generation(self, conn: sqlite3.Connection, owner: OwnerSnapshot) -> int | None:
        """The committed generation for this EXACT owner, or None."""

        row = conn.execute(
            "SELECT enrollment_generation FROM enrollments "
            "WHERE owner_key = ? AND credential_id = ? AND tenant_id = ? AND endpoint_id = ? "
            "AND state = 'committed'",
            (
                _owner_key(owner),
                owner.credential_id,
                owner.tenant_id,
                owner.endpoint_id,
            ),
        ).fetchone()
        return int(row["enrollment_generation"]) if row is not None else None

    def _require_live(self, conn: sqlite3.Connection, owner: OwnerSnapshot) -> None:
        live = self._live_generation(conn, owner)
        if live is None:
            raise StoreRefused("not_enrolled", "no committed enrollment for this exact owner")
        if live != owner.enrollment_generation:
            raise StoreRefused(
                "stale_generation",
                f"owner generation {owner.enrollment_generation} != live {live}",
            )

    # -- enrollment seam (H4 invokes; generic prefs never do) ----------------

    def active_owner(self, *, credential_id: int | None = None) -> OwnerSnapshot | None:
        """The committed owner, or None — never a fresh enrollment.

        A missing/corrupt/unsupported database reads as "no authority"; the
        caller must stop (the handoff's fail-closed law). ``credential_id``
        narrows to the EXACT pinned credential: a same-account re-login with a
        new credential row is a DIFFERENT owner and needs explicit rebinding.
        """

        try:
            with self._connect() as conn:
                self._init(conn)
                query = "SELECT * FROM enrollments WHERE state = 'committed'" + (
                    " AND credential_id = ?" if credential_id is not None else ""
                )
                params = (credential_id,) if credential_id is not None else ()
                row = conn.execute(query, params).fetchone()
                if row is None:
                    return None
                return OwnerSnapshot(
                    credential_id=int(row["credential_id"]),
                    account_id=row["account_id"],
                    tenant_id=row["tenant_id"],
                    endpoint_id=row["endpoint_id"],
                    canonical_base_url=row["canonical_base_url"],
                    enrollment_generation=int(row["enrollment_generation"]),
                ).validated()
        except StoreRefused:
            return None
        except sqlite3.Error:
            return None

    def commit_enrollment(
        self,
        owner: OwnerSnapshot,
        receipt: OwnerReceipt,
        *,
        cos_session_id: str | None = None,
    ) -> None:
        """Atomically commit authority: exact owner + server consent receipt.

        This is H2's SEAM. H4 calls it only after the real captured-owner
        consent flow succeeded; a generic ``{enabled, session}`` preference
        has no path here, and a failed consent flow cannot produce a receipt.
        """

        owner.validated()
        if not receipt.consent_version.strip() or not receipt.consent_ref.strip():
            raise StoreRefused("invalid_receipt", "consent receipt must name version + ref")
        with self._connect() as conn:
            self._init(conn)
            conn.execute(
                "INSERT OR REPLACE INTO enrollments (owner_key, credential_id, account_id, "
                "tenant_id, endpoint_id, canonical_base_url, enrollment_generation, "
                "consent_version, consent_ref, designated_session, state) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,'committed')",
                (
                    _owner_key(owner),
                    owner.credential_id,
                    owner.account_id,
                    owner.tenant_id,
                    owner.endpoint_id,
                    owner.canonical_base_url,
                    owner.enrollment_generation,
                    receipt.consent_version,
                    receipt.consent_ref,
                    cos_session_id,
                ),
            )
            conn.execute("INSERT OR IGNORE INTO sync (owner_key) VALUES (?)", (_owner_key(owner),))

    def invalidate(self, owner: OwnerSnapshot) -> int:
        """Fence pending results/claims and bump the generation. Returns it.

        Fences (marks ``pending_reconciliation``) rather than erases: an
        already-SENT remote mutation cannot be recalled, so its refs-only
        uncertain state stays under the OLD owner for explicit reconciliation.
        No approval/denial, no server history change, no sibling selection.
        """

        with self._connect() as conn:
            self._init(conn)
            self._require_live(conn, owner)
            new_generation = owner.enrollment_generation + 1
            conn.execute(
                "UPDATE enrollments SET enrollment_generation = ? WHERE owner_key = ?",
                (new_generation, _owner_key(owner)),
            )
            conn.execute(
                "UPDATE claims SET state = 'fenced' WHERE owner_key = ? AND state = 'held'",
                (_owner_key(owner),),
            )
            return new_generation

    # -- sync / inbox --------------------------------------------------------

    def merge_page(
        self, owner: OwnerSnapshot, refs: Sequence[MessageRef], cursor: InboxCursor | None
    ) -> None:
        """Persist a page's refs + checkpoint in ONE transaction.

        ``cursor`` is the server's NEXT cursor (``None`` = last page). Refs
        were already projected (no content); this method stores the whitelist
        columns only. A page that arrives after an invalidation is refused by
        the generation check — the sender may then retry under the new
        enrollment explicitly.
        """

        with self._connect() as conn:
            self._init(conn)
            self._require_live(conn, owner)
            for ref in refs:
                if ref.tenant_id != owner.tenant_id:
                    raise StoreRefused("wrong_tenant", "item tenant differs from owner")
                conn.execute(
                    "INSERT OR REPLACE INTO deliveries (owner_key, message_id, thread_id, "
                    "tenant_id, seq, msg_type, origin, sender_account_id, sender_endpoint_id, "
                    "recipient_account_id, recipient_endpoint_id, body_hash, delivery_state, "
                    "state_at, request_state, request_expires_at, request_ref) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        _owner_key(owner),
                        ref.message_id,
                        ref.thread_id,
                        ref.tenant_id,
                        ref.seq,
                        ref.message_type,
                        ref.origin,
                        ref.sender_account_id,
                        ref.sender_endpoint_id,
                        ref.recipient_account_id,
                        ref.recipient_endpoint_id,
                        ref.body_hash,
                        ref.delivery_state,
                        ref.state_at,
                        ref.request_state,
                        ref.request_expires_at,
                        ref.request_ref,
                    ),
                )
            conn.execute(
                "UPDATE sync SET inbox_checkpoint = ?, last_observed_at = ? WHERE owner_key = ?",
                (None if cursor is None else str(cursor), time.time(), _owner_key(owner)),
            )

    def record_feed_batch(
        self,
        owner: OwnerSnapshot,
        cursor: FeedSeq,
        event_refs: Sequence[str],
        *,
        resync_required: bool,
    ) -> None:
        """Persist refs + feed cursor in one transaction. Distinct from inbox."""

        with self._connect() as conn:
            self._init(conn)
            self._require_live(conn, owner)
            current = conn.execute(
                "SELECT pending_resync FROM sync WHERE owner_key = ?", (_owner_key(owner),)
            ).fetchone()
            pending = int(current["pending_resync"]) if current is not None else 0
            for ref in event_refs:
                conn.execute(
                    "INSERT OR IGNORE INTO feed_refs (owner_key, ref, seq) VALUES (?,?,?)",
                    (_owner_key(owner), str(ref), int(cursor)),
                )
            if resync_required or pending:
                # A fresh cursor may be LOWER than the stored one (the server
                # resets on ttl/gap), so this is not a compare-max rule: once
                # resync is pending, the walk owns the cursor until finished.
                conn.execute(
                    "UPDATE sync SET pending_resync = 1, feed_seq = ? WHERE owner_key = ?",
                    (int(cursor), _owner_key(owner)),
                )
            else:
                conn.execute(
                    "UPDATE sync SET feed_seq = ? WHERE owner_key = ?",
                    (int(cursor), _owner_key(owner)),
                )

    def finish_resync(self, owner: OwnerSnapshot, snapshot_checkpoint: InboxCursor) -> None:
        """Complete ONLY when every page was merged: clears the pending flag."""

        with self._connect() as conn:
            self._init(conn)
            self._require_live(conn, owner)
            conn.execute(
                "UPDATE sync SET pending_resync = 0, inbox_checkpoint = ?, "
                "walk_generation = walk_generation + 1, last_observed_at = ? "
                "WHERE owner_key = ?",
                (str(snapshot_checkpoint), time.time(), _owner_key(owner)),
            )

    def sync_state(self, owner: OwnerSnapshot) -> dict[str, Any]:
        """Read-only introspection of the sync row (no authority implied)."""

        with self._connect() as conn:
            self._init(conn)
            row = conn.execute(
                "SELECT * FROM sync WHERE owner_key = ?", (_owner_key(owner),)
            ).fetchone()
            if row is None:
                return {"feed_seq": 0, "pending_resync": False, "inbox_checkpoint": None}
            return {
                "feed_seq": int(row["feed_seq"]),
                "pending_resync": bool(row["pending_resync"]),
                "inbox_checkpoint": row["inbox_checkpoint"],
            }

    # -- ack intents ---------------------------------------------------------

    def record_ack_intent(
        self,
        owner: OwnerSnapshot,
        message_id: str,
        target: AckTarget,
        receipt: ScreeningReceipt | None,
        client_ref: str,
    ) -> str:
        """Write-once ack intent, recorded BEFORE any dispatch.

        ``read`` requires an allow receipt and ``blocked`` a reject receipt
        (the server refuses ``read`` without a verdict; H2 records the local
        intent with the EXACT immutable tuple it will send, so a crash between
        intent and dispatch replays identically — never a re-screen).
        Re-recording the same (owner, message, target) with the same tuple is
        idempotent; a DIFFERENT tuple is refused as a write-once conflict.
        """

        if target not in ACK_TARGETS:
            raise StoreRefused("invalid_target", f"ack target {target!r}")
        if receipt is None or receipt.verdict != ("allow" if target == "read" else "reject"):
            raise StoreRefused("receipt_required", f"{target} needs a matching screening receipt")
        intent_id = str(uuid.uuid4())
        with self._connect() as conn:
            self._init(conn)
            self._require_live(conn, owner)
            existing = conn.execute(
                "SELECT * FROM ack_intents WHERE owner_key = ? AND message_id = ? AND target = ?",
                (_owner_key(owner), message_id, target),
            ).fetchone()
            if existing is not None:
                if (
                    existing["body_sha256"] == receipt.body_sha256
                    and existing["verdict"] == receipt.verdict
                    and json.loads(existing["categories"]) == sorted(receipt.categories)
                ):
                    return str(existing["intent_id"])
                raise StoreRefused("intent_conflict", "write-once intent tuple differs")
            conn.execute(
                "INSERT INTO ack_intents (intent_id, owner_key, message_id, target, client_ref, "
                "verdict, model, prompt_version, template_sha256, body_sha256, categories, "
                "state, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    intent_id,
                    _owner_key(owner),
                    message_id,
                    target,
                    client_ref,
                    receipt.verdict,
                    receipt.model,
                    receipt.prompt_version,
                    receipt.template_sha256,
                    receipt.body_sha256,
                    json.dumps(sorted(receipt.categories)),
                    ACK_PENDING,
                    time.time(),
                ),
            )
            return intent_id

    def ack_intent(self, owner: OwnerSnapshot, intent_id: str) -> dict[str, Any] | None:
        """Read one intent (recovery). ``None`` means NOT reconstructable."""

        try:
            with self._connect() as conn:
                self._init(conn)
                row = conn.execute(
                    "SELECT * FROM ack_intents WHERE owner_key = ? AND intent_id = ?",
                    (_owner_key(owner), intent_id),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "intent_id": row["intent_id"],
                    "message_id": row["message_id"],
                    "target": row["target"],
                    "client_ref": row["client_ref"],
                    "receipt": (
                        {
                            "verdict": row["verdict"],
                            "model": row["model"],
                            "prompt_version": row["prompt_version"],
                            "template_sha256": row["template_sha256"],
                            "body_sha256": row["body_sha256"],
                            "categories": json.loads(row["categories"]),
                        }
                        if row["verdict"] is not None
                        else None
                    ),
                    "state": row["state"],
                    "error_code": row["error_code"],
                    "retry_after": row["retry_after"],
                }
        except sqlite3.Error:
            return None

    def mark_ack_result(
        self, owner: OwnerSnapshot, intent_id: str, *, state: str, error_code: str | None
    ) -> None:
        """Confirm or conflict ONE intent. Confirmed is terminal; pending stays."""

        if state not in (ACK_CONFIRMED, ACK_CONFLICT, ACK_PENDING):
            raise StoreRefused("invalid_state", state)
        with self._connect() as conn:
            self._init(conn)
            row = conn.execute(
                "SELECT state FROM ack_intents WHERE owner_key = ? AND intent_id = ?",
                (_owner_key(owner), intent_id),
            ).fetchone()
            if row is None:
                raise StoreRefused("unknown_intent", intent_id)
            if row["state"] == ACK_CONFIRMED and state != ACK_CONFIRMED:
                raise StoreRefused("already_confirmed", "confirmed intents are terminal")
            conn.execute(
                "UPDATE ack_intents SET state = ?, error_code = ? WHERE owner_key = ? "
                "AND intent_id = ?",
                (state, error_code, _owner_key(owner), intent_id),
            )

    # -- claims (CAS lease) --------------------------------------------------

    def claim(
        self,
        owner: OwnerSnapshot,
        message_id: str,
        session_id: str,
        lease_token: str,
        expiry: float,
    ) -> None:
        """Compare-and-set: at most ONE live local claim per message.

        A live claim (state ``held`` and unexpired) refuses a second
        contender; an expired or fenced claim is replaceable but ONLY by a new
        token — a stale token never releases or renews a successor.
        """

        if expiry <= time.time():
            raise StoreRefused("expired_lease", "expiry must be in the future")
        with self._connect() as conn:
            self._init(conn)
            self._require_live(conn, owner)
            row = conn.execute(
                "SELECT * FROM claims WHERE owner_key = ? AND message_id = ?",
                (_owner_key(owner), message_id),
            ).fetchone()
            if (
                row is not None
                and row["state"] == "held"
                and float(row["expires_at"]) > time.time()
            ):
                raise StoreRefused("claimed", "a live claim already holds this message")
            conn.execute(
                "INSERT OR REPLACE INTO claims (owner_key, message_id, session_id, token, "
                "generation, expires_at, state) VALUES (?,?,?,?,?,?,'held')",
                (
                    _owner_key(owner),
                    message_id,
                    session_id,
                    lease_token,
                    owner.enrollment_generation,
                    expiry,
                ),
            )

    def renew_claim(
        self, owner: OwnerSnapshot, message_id: str, lease_token: str, expiry: float
    ) -> None:
        with self._connect() as conn:
            self._init(conn)
            row = conn.execute(
                "SELECT * FROM claims WHERE owner_key = ? AND message_id = ?",
                (_owner_key(owner), message_id),
            ).fetchone()
            if row is None or row["token"] != lease_token or row["state"] != "held":
                raise StoreRefused("not_holder", "renew refused: token is not the live holder")
            conn.execute(
                "UPDATE claims SET expires_at = ? WHERE owner_key = ? AND message_id = ?",
                (expiry, _owner_key(owner), message_id),
            )

    def release_claim(self, owner: OwnerSnapshot, message_id: str, lease_token: str) -> None:
        """Conditional release: only the CURRENT token may release."""

        with self._connect() as conn:
            self._init(conn)
            row = conn.execute(
                "SELECT token FROM claims WHERE owner_key = ? AND message_id = ?",
                (_owner_key(owner), message_id),
            ).fetchone()
            if row is None:
                return
            if row["token"] != lease_token:
                raise StoreRefused("not_holder", "release refused: token is not the live holder")
            conn.execute(
                "UPDATE claims SET state = 'released' WHERE owner_key = ? AND message_id = ?",
                (_owner_key(owner), message_id),
            )

    def drain_feed_refs(self, owner: OwnerSnapshot, upto_seq: int) -> list[str]:
        """Refs durably recorded at or before ``upto_seq``, for the H4 drainer.

        Crash after commit replays nothing (refs are durable); crash BEFORE
        commit replays the feed batch — the same either way: a ref is acted on
        only after it is on disk here.
        """

        with self._connect() as conn:
            self._init(conn)
            rows = conn.execute(
                "SELECT ref FROM feed_refs WHERE owner_key = ? AND seq <= ? ORDER BY seq, ref",
                (_owner_key(owner), int(upto_seq)),
            ).fetchall()
            return [row["ref"] for row in rows]

    def merge_refs(self, owner: OwnerSnapshot, refs: Sequence[MessageRef]) -> None:
        """Persist refs WITHOUT touching the inbox checkpoint (single reads)."""

        with self._connect() as conn:
            self._init(conn)
            self._require_live(conn, owner)
            for ref in refs:
                if ref.tenant_id != owner.tenant_id:
                    raise StoreRefused("wrong_tenant", "item tenant differs from owner")
                conn.execute(
                    "INSERT OR REPLACE INTO deliveries (owner_key, message_id, thread_id, "
                    "tenant_id, seq, msg_type, origin, sender_account_id, sender_endpoint_id, "
                    "recipient_account_id, recipient_endpoint_id, body_hash, delivery_state, "
                    "state_at, request_state, request_expires_at, request_ref) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        _owner_key(owner),
                        ref.message_id,
                        ref.thread_id,
                        ref.tenant_id,
                        ref.seq,
                        ref.message_type,
                        ref.origin,
                        ref.sender_account_id,
                        ref.sender_endpoint_id,
                        ref.recipient_account_id,
                        ref.recipient_endpoint_id,
                        ref.body_hash,
                        ref.delivery_state,
                        ref.state_at,
                        ref.request_state,
                        ref.request_expires_at,
                        ref.request_ref,
                    ),
                )

    def list_metadata(self, owner: OwnerSnapshot) -> list[dict[str, Any]]:
        """Whitelisted metadata rows for the egress-free consumers (H4 later).

        Deliberately excludes ack/intent contents and any body-adjacent field;
        the local ``blocked_reviewed`` attention marker is local-only and the
        server ``blocked`` state is never cleared by it.
        """

        with self._connect() as conn:
            self._init(conn)
            rows = conn.execute(
                "SELECT message_id, thread_id, tenant_id, seq, msg_type, origin, "
                "sender_account_id, sender_endpoint_id, recipient_account_id, "
                "recipient_endpoint_id, body_hash, delivery_state, state_at, "
                "request_state, request_expires_at, local_hold_code, attempts, "
                "blocked_reviewed, pending_reconciliation "
                "FROM deliveries WHERE owner_key = ?",
                (_owner_key(owner),),
            ).fetchall()
            return [dict(row) for row in rows]

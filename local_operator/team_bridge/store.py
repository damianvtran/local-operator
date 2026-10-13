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

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from local_operator.team_bridge.types import (
    ACK_TARGETS,
    AckTarget,
    FeedSeq,
    InboxCursor,
    MessageRef,
    OwnerSnapshot,
    ScreeningReceipt,
    TeamBridgeTypeError,
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

SCHEMA_VERSION = 5

#: Every table a stamped-V3 database must carry. A stamped store missing any
#: of these is DAMAGED and refuses: re-running DDL is NOT repair — it
#: silently recreates an empty ``screening_attempts`` and resets known spend
#: (H3-Q4's fail-open quota reproduction). The refusal happens pre-DDL.
_V3_TABLES = frozenset(
    {
        "schema_meta",
        "enrollments",
        "ack_intents",
        "claims",
        "deliveries",
        "feed_refs",
        "sync",
        "screening_attempts",
        "screening_receipts",
        "screening_holds",
        "screening_policy",
        "screening_breaker",
    }
)
#: The closed screening-category set (mirrors screening.SCREENING_CATEGORIES;
#: duplicated here so the store validates persisted receipts without an
#: import cycle). Unknown strings are damaged proof, never re-emitted.
_CLOSED_CATEGORIES = frozenset(
    {"prompt_injection", "social_engineering", "harassment", "malicious_link", "spam"}
)

#: The V4 addition: the schedule table is NEW in V4 — a stamped-V3 store
#: legitimately lacks it (R5: an own R4-era V3 backup was declared sound and
#: then raised OperationalError on the first cycle), so V3 upgrades
#: additively and only V4+ requires it.
_V4_ADDED = frozenset({"screening_schedule"})

#: A V2 (H2-era) store migrates ADDITIVELY: only the screening tables are
#: created and the stamp advances; every existing row survives.
_FLOORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS text_floors (
    owner_key TEXT NOT NULL,
    hub TEXT NOT NULL,
    kind TEXT NOT NULL,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (owner_key, hub, kind)
);
"""

_V2_ADDED = frozenset(
    {
        "screening_attempts",
        "screening_receipts",
        "screening_holds",
        "screening_policy",
        "screening_breaker",
    }
)
#: Column floors for the screening tables: an earlier unreleased V3 (no
#: binding/policy columns) is indistinguishable from damage — both refuse.
_V3_COLUMNS = {
    "screening_attempts": frozenset(
        {
            "attempt_id",
            "kind",
            "account_scope",
            "owner_key",
            "message_id",
            "fingerprint",
            "at",
            "committed_message_id",
        }
    ),
    "screening_receipts": frozenset(
        {
            "owner_key",
            "message_id",
            "body_sha256",
            "template_sha256",
            "prompt_version",
            "model",
            "verdict",
            "categories",
            "envelope_digest",
            "admission_revision",
            "policy_epoch",
            "creator_generation",
            "created_at",
        }
    ),
    "screening_holds": frozenset(
        {"owner_key", "message_id", "hold_code", "retry_at", "created_at"}
    ),
    "screening_policy": frozenset(
        {
            "owner_key",
            "version",
            "template_sha256",
            "default_model",
            "revision",
            "epoch",
            "accepted_at",
        }
    ),
    "screening_breaker": frozenset({"account_scope", "failures", "open_until", "updated_at"}),
}

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


_CORE_SCHEMA = """
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
    walk_terminal INTEGER NOT NULL DEFAULT 0,
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
_SCREENING_SCHEMA = """
-- -- H3 screening: budget ledger, receipts, holds, policy epoch, breaker ----
-- The attempts ledger counts TWO independent buckets per account ('logical'
-- screenings and actual 'http' inference dispatches, 40/hour each): the
-- conservative physical counter exists because one logical screening can
-- reach up to three wire attempts (auth re-resolve + fast resend), and both
-- must be reserved durably BEFORE their I/O.
CREATE TABLE IF NOT EXISTS screening_attempts (
    attempt_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    account_scope TEXT NOT NULL,
    owner_key TEXT NOT NULL,
    message_id TEXT,
    fingerprint TEXT,
    at REAL NOT NULL,
    committed_message_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_screening_attempts_scope_kind_at
    ON screening_attempts (account_scope, kind, at);
CREATE TABLE IF NOT EXISTS screening_receipts (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    body_sha256 TEXT NOT NULL,
    template_sha256 TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    verdict TEXT NOT NULL,
    categories TEXT NOT NULL,
    envelope_digest TEXT NOT NULL,
    admission_revision TEXT NOT NULL,
    policy_epoch INTEGER NOT NULL,
    creator_generation INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_key, message_id)
);
CREATE TABLE IF NOT EXISTS screening_holds (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    hold_code TEXT NOT NULL,
    retry_at REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_key, message_id)
);
CREATE TABLE IF NOT EXISTS screening_schedule (
    owner_key TEXT PRIMARY KEY,
    last_sender TEXT,
    lease_holder TEXT,
    lease_expires REAL
);
CREATE TABLE IF NOT EXISTS screening_policy (
    owner_key TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    template_sha256 TEXT NOT NULL,
    default_model TEXT,
    revision TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    accepted_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS screening_breaker (
    account_scope TEXT PRIMARY KEY,
    failures INTEGER NOT NULL,
    open_until REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""
_SCHEMA = _CORE_SCHEMA + _SCREENING_SCHEMA


def _fingerprint_revision(fingerprint: str) -> str:
    """The admission revision embedded in a policy fingerprint string."""

    return fingerprint.split("|", 1)[0]


def _version_rank(version: str) -> tuple[int, str]:
    """Rank a template version for the monotonic floor: trailing digits first."""

    digits = "".join(ch for ch in version if ch.isdigit())
    return (int(digits) if digits else -1, version)


def _owner_frame_digest(owner: OwnerSnapshot) -> str:
    """Digest of the INITIATING owner frame (identity + gen), embedded in
    lease tokens so another live owner on the same account cannot borrow
    them (R7/Q15-6)."""

    material = (
        f"{owner.credential_id}|{owner.tenant_id}|{owner.endpoint_id}"
        f"|{owner.enrollment_generation}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _owner_frame_ok(turn: str, owner: OwnerSnapshot) -> bool:
    """The presented token must carry THIS caller's own frame digest."""

    return _turn_frame(turn) == _owner_frame_digest(owner)


def _turn_frame(turn: str) -> str:
    return turn.rsplit(":", 1)[-1] if ":" in turn else ""


#: Markers in the one available attempts column (R7/R8/Q3): at reservation
#: the row holds "proof:<canonical json>" (or the legacy packed fingerprint);
#: after the ONE legal bind it holds "job:<canonical digest>"; a second bind
#: must replay exactly.
_PROOF_FIELDS = (
    "owner",
    "body",
    "template",
    "version",
    "provider",
    "model",
    "approved",
    "envelope",
    "request",
    "display",
    "revision",
    "epoch",
    "message",
    "tenant",
    "logical",
)
_JOB_BOUND_PREFIX = "job:"
_PROOF_PREFIX = "proof:"


def _canonical_digest_from_fields(fields: list[Any]) -> str:
    """The ONE canonical digest encoder (stable order, compact separators).

    Both the job mint and the retained-registration re-derivation hash this
    exact list shape — one encoder, so the two sides can never drift into
    duplicate serializers (R12/F1).
    """

    payload = json.dumps(fields, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_proof_digest(registered: Any, attempt_id: Any, turn: Any) -> str | None:
    """Re-derive the canonical digest from a RETAINED registration record.

    The retained record carries the semantic fields but not the attempt id or
    the turn token (both are native/transaction context), so they are
    supplied from the actual row and the finishing transaction. Any altered
    field — model, provider, request, display, logical reference, approved
    edit that keeps membership — changes this derivation and so cannot equal
    the bound digest. Returns ``None`` when the record's owner namespace is
    unparseable (the caller refuses).
    """

    owner = registered.get("owner")
    if not isinstance(owner, str):
        return None
    parts = owner.split("|")
    if len(parts) != 6:
        return None
    credential_id_s, account_id, tenant_id, endpoint_id, hub_url, owner_generation_s = parts

    # The canonical digest is taken over NATIVE scalar types (the job mint
    # hashes ints for credential id and generation), while the retained
    # record stores the owner namespace joined as text — so re-coerce the
    # numeric parts back to their minted types here. A non-numeric part keeps
    # its string form, which then simply fails to equal the bound digest and
    # refuses, never silently matching a different shape.
    def _as_int(value: str) -> Any:
        try:
            return int(value)
        except ValueError:
            return value

    credential_id = _as_int(credential_id_s)
    generation = _as_int(owner_generation_s)
    return _canonical_digest_from_fields(
        [
            generation,
            credential_id,
            account_id,
            tenant_id,
            endpoint_id,
            hub_url,
            registered.get("message"),
            attempt_id,
            registered.get("body"),
            registered.get("template"),
            registered.get("version"),
            registered.get("provider"),
            registered.get("model"),
            list(registered.get("approved") or ()),
            registered.get("envelope"),
            registered.get("request"),
            registered.get("display"),
            registered.get("revision"),
            registered.get("epoch"),
            turn,
        ]
    )


def _canonical_job_digest(job: Any) -> str:
    """Independently derive the canonical full-job digest from its FIELDS.

    R10/B: the caller's ``.digest`` attribute, the supplied fingerprint and
    any wrapper are NOT the derivation — QA bound a job that delegated every
    field but returned 64 zeros, and a digest-only check accepted it. The
    encoding is the same stable order AdmissionJob documents; deriving it
    here (metadata-only, no model imports) is what makes a forged digest
    impossible to bind, dispatch or receipt.
    """

    return _canonical_digest_from_fields(
        [
            getattr(job, "owner_generation", None),
            getattr(job, "credential_id", None),
            getattr(job, "account_id", None),
            getattr(job, "tenant_id", None),
            getattr(job, "endpoint_id", None),
            getattr(job, "hub_url", None),
            getattr(job, "message_id", None),
            getattr(job, "attempt_id", None),
            getattr(job, "body_sha256", None),
            getattr(job, "template_sha256", None),
            getattr(job, "template_version", None),
            getattr(job, "provider", None),
            getattr(job, "model", None),
            list(getattr(job, "approved", ()) or ()),
            getattr(job, "envelope_digest", None),
            getattr(job, "request_digest", None),
            getattr(job, "display_digest", None),
            getattr(job, "revision", None),
            getattr(job, "epoch", None),
            getattr(job, "turn", None),
        ]
    )


def _decode_bound_proof(raw: Any) -> tuple[str, dict[str, Any]]:
    """Strictly decode ``job:<64hex>|proof:<json>``; anything else refuses.

    R10/B: ONE closed decoder for the retained registration, reused at every
    native boundary. Null/truncated/extra/typed-wrong proofs and the legacy
    ``job:<digest>`` form (no retained registration) are typed refusals, not
    keywords to be substring-matched.
    """

    if not isinstance(raw, str) or not raw.startswith(_JOB_BOUND_PREFIX):
        raise StoreRefused("proof_corrupt", "attempt row is not in bound form")
    body = raw[len(_JOB_BOUND_PREFIX) :]
    if "|" not in body:
        raise StoreRefused("unproved_attempt", "registration missing; reconcile only")
    digest, _, proof_raw = body.partition("|")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise StoreRefused("proof_corrupt", "bound digest is not a sha256 hex digest")
    if not proof_raw.startswith(_PROOF_PREFIX):
        raise StoreRefused("proof_corrupt", "registration marker missing")
    try:
        registered = json.loads(proof_raw[len(_PROOF_PREFIX) :])
    except (TypeError, ValueError) as error:
        raise StoreRefused("proof_corrupt", "registered input proof is unreadable") from error
    if not isinstance(registered, dict):
        raise StoreRefused("proof_corrupt", "registered input proof is not a mapping")
    if set(registered) != set(_PROOF_FIELDS):
        raise StoreRefused("proof_corrupt", "registered input proof has an unexpected field set")
    for field in _PROOF_FIELDS:
        value = registered[field]
        if field == "approved":
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(entry, str) or not entry for entry in value)
            ):
                raise StoreRefused("proof_corrupt", "approved map has an unsupported shape")
            continue
        if field == "epoch":
            if not isinstance(value, int) or isinstance(value, bool):
                raise StoreRefused("proof_corrupt", "epoch has an unsupported shape")
            continue
        if not isinstance(value, str) or not value:
            raise StoreRefused("proof_corrupt", f"{field} has an unsupported shape")
    return digest, registered


def _input_proof_mismatch(registered: Any, fingerprint: str, job: Any) -> str | None:
    """Compare a DispatchJob against the input proof registered at
    reservation (R8/Q3). The FIRST bind must match EVERY field the record
    pinned — a field mutated between reservation and bind refuses before any
    I/O instead of dispatching on the strength of a digest the caller could
    recompute. Returns None when the record matches (or when no job is
    supplied for the check, which the legacy packed form uses)."""

    if not isinstance(registered, dict):
        return "registered proof is not a mapping"
    if job is None:
        # A bind WITHOUT the job cannot certify anything (R9/Q3-5): the
        # registered record exists to be compared against a real job, and a
        # legacy no-job bind is exactly the bypass that dispatched before.
        return "a dispatch job is required to certify registered input"
    owner_expected = registered.get("owner")
    owner_actual = (
        f"{getattr(job, 'credential_id', None)}|{getattr(job, 'account_id', None)}|"
        f"{getattr(job, 'tenant_id', None)}|{getattr(job, 'endpoint_id', None)}|"
        f"{getattr(job, 'hub_url', None)}|{getattr(job, 'owner_generation', None)}"
    )
    if owner_expected != owner_actual:
        return "dispatch job owner namespace does not match the registered input proof"
    for key, attr in (
        ("body", "body_sha256"),
        ("envelope", "envelope_digest"),
        ("request", "request_digest"),
        ("display", "display_digest"),
    ):
        if registered.get(key) != getattr(job, attr, None):
            return f"dispatch job {key} does not match the registered input proof"
    field_checks = {
        "provider": getattr(job, "provider", None),
        "model": getattr(job, "model", None),
        "template": getattr(job, "template_sha256", None),
        "version": getattr(job, "template_version", None),
        "tenant": getattr(job, "tenant_id", None),
        "revision": getattr(job, "revision", None),
        "message": getattr(job, "message_id", None),
    }
    for key, value in field_checks.items():
        if registered.get(key) != value:
            return f"dispatch job {key} does not match the registered input proof"
    if list(registered.get("approved") or []) != sorted(getattr(job, "approved", ()) or ()):
        return "dispatch job approved map does not match the registered input proof"
    if registered.get("epoch") != int(getattr(job, "epoch", -1)):
        return "dispatch job epoch does not match the registered input proof"
    # NOTE: the registered "logical" value is the PACKED pre-job fingerprint;
    # the ``fingerprint`` argument is the job digest this bind CREATES. The
    # digest is the product of the bind, not an input to compare — comparing
    # them would refuse every legitimate first bind.
    return None


def _bound_digest(raw: Any) -> Any:
    """The canonical digest of a bound row.

    A bound value is ``job:<digest>|proof:<json>`` (R9/Q3-5): the ORIGINAL
    registration is retained beside the digest so later native checks can
    compare against it. Only the digest part participates in digest
    comparisons.
    """

    if isinstance(raw, str) and raw.startswith(_JOB_BOUND_PREFIX):
        body = raw[len(_JOB_BOUND_PREFIX) :]
        return body.split("|", 1)[0]
    return raw


def _decode_categories_text(raw: Any) -> list[str]:
    """Decode persisted categories JSON for COMPARISON or consumption.

    EMPTY SQL TEXT (or invalid JSON, or a non-list) is DAMAGED proof, not an
    empty tuple (R7/Q2): it was never written by this codebase, so nothing
    may silently coalesce it to [] and admit the row. Callers surface the
    typed refusal; only a valid JSON array — "[]" included — passes.
    """

    if not isinstance(raw, str) or raw.strip() == "":
        raise StoreRefused("proof_corrupt", "empty categories text")
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise StoreRefused("proof_corrupt", "categories decode failed") from error
    if not isinstance(decoded, list):
        raise StoreRefused("proof_corrupt", "categories not a list")
    return decoded


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

    def _begin_write(self, conn: sqlite3.Connection) -> None:
        """Take the write lock BEFORE any read a generation fence depends on.

        Every mutating path runs its authority check and its writes inside ONE
        immediate transaction: a deferred read-then-write let another
        connection invalidate between the two (QA R2 H2-Q5/Q6), so a stale
        operation could still alter the new generation's rows. BEGIN IMMEDIATE
        serializes check+write; contention beyond the busy timeout is a typed
        refusal, never a silent retry."""

        if conn.in_transaction:
            # _init's fresh-schema stamp can leave an implicit DEFERRED
            # transaction open; it holds no user data (schema bootstrap only),
            # so committing it is safe and lets the authority check take the
            # write lock now rather than inheriting a deferred one.
            conn.commit()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            raise StoreRefused("busy", str(error)) from error

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
        """Validate BEFORE any DDL; refuse damage instead of repairing it.

        Order is the safety property (H3-Q4): read what the file IS, decide,
        and only then write. A stamped store missing required structures
        refuses loudly — never re-create a screening counter table empty,
        which would reset known spend to zero and fail open. Only a genuinely
        empty file gets the full schema; a known V2 (H2-era) store migrates
        ADDITIVELY, preserving every row; anything unknown refuses.
        """

        try:
            pre_existing = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            version = None
            if "schema_meta" in pre_existing:
                row = conn.execute(
                    "SELECT value FROM schema_meta WHERE key = 'schema_version'"
                ).fetchone()
                version = row["value"] if row is not None else None
        except sqlite3.Error as error:
            raise StoreRefused("state_unreadable", str(error)) from error

        if version is None:
            if pre_existing - {"schema_meta"}:
                raise StoreRefused(
                    "schema_unversioned",
                    "nonempty state.sqlite3 without a schema_version row; refusing authority",
                )
            try:
                conn.executescript(_SCHEMA)
                conn.executescript(_FLOORS_SCHEMA)
            except sqlite3.Error as error:
                raise StoreRefused("state_unreadable", str(error)) from error
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            return

        if version == "2":
            # Known H2-era layout: additive migration only. Every V2 table
            # must be present as V2 promised, then the screening tables are
            # added and the stamp advances. No existing row is touched, so
            # the H2 owner/generation/claims/ACK tuple/ref/state survive
            # byte-for-byte; a damaged V2 store refuses like a V3 one.
            missing = (_V3_TABLES - _V2_ADDED) - pre_existing
            if missing:
                raise StoreRefused("schema_damaged", f"v2 store missing {sorted(missing)}")
            try:
                conn.executescript(_SCREENING_SCHEMA)
                conn.executescript(_FLOORS_SCHEMA)
            except sqlite3.Error as error:
                raise StoreRefused("state_unreadable", str(error)) from error
            conn.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )
            return

        if version == "3":
            # Known V3 (R4-era) layout: ADDITIVE upgrade only. Every V3 table
            # must be present exactly as V3 promised; the new schedule table
            # is created (IF NOT EXISTS, so an already-carrying store is
            # untouched) and the stamp advances. No existing row is touched —
            # receipts, policies, attempts and the H2 enrollment/claim/ACK
            # history survive byte-for-byte.
            missing = _V3_TABLES - pre_existing
            if missing:
                raise StoreRefused("schema_damaged", f"stamped-v3 store missing {sorted(missing)}")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS screening_schedule ("
                    "owner_key TEXT PRIMARY KEY, last_sender TEXT, "
                    "lease_holder TEXT, lease_expires REAL)"
                )
                conn.executescript(_FLOORS_SCHEMA)
            except sqlite3.Error as error:
                raise StoreRefused("state_unreadable", str(error)) from error
            conn.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )
            return

        if version == "4":
            # Known V4 (R10-era) layout: ADDITIVE upgrade only — the
            # contentless accepted-text FLOORS table is created (IF NOT
            # EXISTS, so a carrying store is untouched) and the stamp
            # advances. Quotas, receipts, policies, attempts and the H2
            # enrollment/claim/ACK history survive byte-for-byte.
            missing = (_V3_TABLES | _V4_ADDED) - pre_existing
            if missing:
                raise StoreRefused("schema_damaged", f"stamped-v4 store missing {sorted(missing)}")
            schedule_columns = {r[1] for r in conn.execute("PRAGMA table_info(screening_schedule)")}
            if (
                not {"owner_key", "last_sender", "lease_holder", "lease_expires"}
                <= schedule_columns
            ):
                raise StoreRefused("schema_damaged", "screening_schedule lacks required columns")
            try:
                conn.executescript(_FLOORS_SCHEMA)
            except sqlite3.Error as error:
                raise StoreRefused("state_unreadable", str(error)) from error
            conn.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )
            return

        if version == str(SCHEMA_VERSION):
            # Stamped V5: VERIFY, never re-create. Missing tables or missing
            # binding/policy columns mean damage — refuse, no DDL at all.
            missing = (_V3_TABLES | _V4_ADDED) - pre_existing
            if missing:
                raise StoreRefused("schema_damaged", f"stamped-v3 store missing {sorted(missing)}")
            for table, required in _V3_COLUMNS.items():
                columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if not required <= columns:
                    raise StoreRefused(
                        "schema_damaged", f"{table} lacks {sorted(required - columns)}"
                    )
            # Current-version integrity floor includes the V4 schedule table.
            schedule_columns = {r[1] for r in conn.execute("PRAGMA table_info(screening_schedule)")}
            if (
                not {"owner_key", "last_sender", "lease_holder", "lease_expires"}
                <= schedule_columns
            ):
                raise StoreRefused("schema_damaged", "screening_schedule lacks required columns")
            floor_columns = {r[1] for r in conn.execute("PRAGMA table_info(text_floors)")}
            if not {"owner_key", "hub", "kind", "version", "sha256", "updated_at"} <= floor_columns:
                raise StoreRefused("schema_damaged", "text_floors lacks required columns")
            return

        raise StoreRefused(
            "schema_unsupported",
            f"state.sqlite3 schema {version!r} != {SCHEMA_VERSION}",
        )

    @staticmethod
    def _floor_key(owner: OwnerSnapshot) -> str:
        """The FLOOR version namespace: ACCOUNT + hub + kind (R11/3).

        Accepted text versions are global to the ACCOUNT's serving hub; the
        credential/tenant/endpoint/generation are authorization guards, not
        ways to reset them (a new organization must not accept an older
        served version the account already moved past). The consent RECEIPT
        stays exact-owner-specific.
        """

        return f"acct:{owner.account_id}"

    def _floor_row(self, conn: Any, owner: OwnerSnapshot, kind: str) -> tuple[str, str] | None:
        """Current floor value: account scope first, then the legacy
        owner-scoped row CONSERVATIVELY (known V5 metadata is reconciled, not
        erased; a carried legacy floor keeps authorising until advanced)."""

        row = conn.execute(
            "SELECT version, sha256 FROM text_floors WHERE owner_key = ? AND hub = ? AND kind = ?",
            (self._floor_key(owner), owner.canonical_base_url, kind),
        ).fetchone()
        if row is not None:
            return str(row["version"]), str(row["sha256"])
        legacy = conn.execute(
            "SELECT version, sha256 FROM text_floors WHERE owner_key = ? AND hub = ? AND kind = ?",
            (_owner_key(owner), owner.canonical_base_url, kind),
        ).fetchone()
        if legacy is not None:
            return str(legacy["version"]), str(legacy["sha256"])
        return None

    def advance_text_floors(self, owner: OwnerSnapshot, staged: list[tuple[str, str, str]]) -> None:
        """Advance accepted floors ATOMICALLY as one staged set (R11/3).

        ONE native transaction: live-owner/generation authorization, then for
        EVERY staged candidate re-read the CURRENT floor (account+hub+kind)
        and validate monotonic version + same-version immutable bytes. All
        floors advance ONLY when every candidate validates — a refused
        candidate (stale version, changed bytes, refused currency/raw origin
        upstream) leaves EVERY floor untouched, so nothing untrusted can be
        made durable. The compare and the write live INSIDE this transaction,
        not in a Python precheck a competing writer can invalidate.
        """

        from local_operator.team_bridge.client import version_number

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            resolved: list[tuple[str, str, str]] = []
            for kind, version, sha in staged:
                number = version_number(version)
                if number is None:
                    raise StoreRefused("policy_stale", f"{kind} version does not parse")
                prior = self._floor_row(conn, owner, kind)
                if prior is not None:
                    prior_number = version_number(prior[0])
                    if prior_number is None or number < prior_number:
                        raise StoreRefused("policy_stale", f"{kind} version moved backwards")
                    if number == prior_number and sha != prior[1]:
                        raise StoreRefused(
                            "policy_stale", f"{kind} bytes differ for an accepted version"
                        )
                resolved.append((kind, version, sha))
            for kind, version, sha in resolved:
                conn.execute(
                    "INSERT INTO text_floors (owner_key, hub, kind, version, sha256, updated_at) "
                    "VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(owner_key, hub, kind) DO UPDATE SET "
                    "version = excluded.version, sha256 = excluded.sha256, "
                    "updated_at = excluded.updated_at",
                    (
                        self._floor_key(owner),
                        owner.canonical_base_url,
                        kind,
                        version,
                        sha,
                        time.time(),
                    ),
                )

    def text_floor(self, owner: OwnerSnapshot, kind: str) -> tuple[str, str] | None:
        """The contentless accepted floor for one served-text kind.

        R10/C: the served guidance/consent texts are accepted through
        monotonic version + immutable bytes, and that floor must SURVIVE
        reads and restarts — a fresh in-memory accept set would forget it.
        Only version + digest are stored; never the text. Missing floors
        are None (the caller holds where no prior floor exists).
        """

        with self._connect() as conn:
            self._init(conn)
            return self._floor_row(conn, owner, kind)

    def set_text_floor(self, owner: OwnerSnapshot, kind: str, version: str, sha256: str) -> None:
        """Advance the contentless floor (same-version identical bytes is a
        no-op by construction). The database never sees served text."""

        self.advance_text_floors(owner, [(kind, version, sha256)])

    def owner_consent_receipt(self, owner: OwnerSnapshot) -> dict[str, str] | None:
        """The COMMITTED human consent receipt for the exact live owner.

        R9/Q3-2: this is the ONLY proof of acceptance — status
        ``consent.version``/``text`` are the CURRENT served shape, never an
        acceptance flag. Read from the existing enrollment columns under the
        native live/generation fence; missing, unreadable or stale rows
        return None (the caller holds — it never reconstructs consent from
        served text or auto-reconsents).
        """

        with self._connect() as conn:
            self._init(conn)
            if self._live_generation(conn, owner) != owner.enrollment_generation:
                return None
            row = conn.execute(
                "SELECT consent_version, consent_ref FROM enrollments "
                "WHERE owner_key = ? AND enrollment_generation = ?",
                (_owner_key(owner), owner.enrollment_generation),
            ).fetchone()
            if row is None:
                return None
            version = row["consent_version"]
            ref = row["consent_ref"]
            if not isinstance(version, str) or not version:
                return None
            if not isinstance(ref, str) or not ref:
                return None
            return {"consent_version": version, "consent_ref": ref}

    def designated_session_for(self, owner: OwnerSnapshot) -> str | None:
        """The designated CoS session id under the live-owner/generation fence.

        READ-ONLY accessor (H4) over the column H2's seam writes
        (``commit_enrollment(..., cos_session_id=...)``). The wake authority is
        the NATIVE enrollment record, never a generic settings preference: a
        preference with no committed enrollment reads ``None`` here, and a
        fenced or superseded generation reads ``None`` too, so a stale owner
        snapshot cannot name a wake destination. Empty/missing values are
        ``None`` (no destination — arrivals badge only, per §5.6).
        """

        with self._connect() as conn:
            self._init(conn)
            if self._live_generation(conn, owner) != owner.enrollment_generation:
                return None
            row = conn.execute(
                "SELECT designated_session FROM enrollments "
                "WHERE owner_key = ? AND enrollment_generation = ?",
                (_owner_key(owner), owner.enrollment_generation),
            ).fetchone()
            if row is None:
                return None
            value = row["designated_session"]
            if not isinstance(value, str) or not value:
                return None
            return value

    def current_generation_is_live(self, owner: OwnerSnapshot) -> bool:
        """Public fence probe (R6/Q15): is THIS snapshot's enrollment
        generation still the live one? Used to pick the accurate typed hold
        (owner_changed vs retry_later) at renew/final fences."""
        with self._connect() as conn:
            self._init(conn)
            return self._live_generation(conn, owner) == owner.enrollment_generation

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
                rows = conn.execute(query, params).fetchall()
                if len(rows) != 1:
                    # Zero rows is "not enrolled". MORE than one for a partial
                    # selector (the credential alone) is AMBIGUOUS: v1 commits
                    # one endpoint, but a future multi-namespace store must
                    # never be read through a guess — callers name the full
                    # namespace via resolve_owner instead (QA H2-Q5).
                    return None
                row = rows[0]
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

    def resolve_owner(
        self,
        *,
        credential_id: int,
        account_id: str,
        tenant_id: str,
        endpoint_id: str,
        canonical_base_url: str | None = None,
    ) -> OwnerSnapshot | None:
        """EXACT full-namespace resolution — the multi-org-safe selector.

        Every field of the namespace is required (the base URL is compared
        when given): a caller that can name the whole namespace gets exactly
        the row it named, and partial-credential ambiguity stays with
        :meth:`active_owner`'s refusal. Dispatch guards use THIS selector, so
        a future multi-namespace store cannot dispatch through a lookup that
        guessed between tenants; other namespaces' rows are retained, never
        globalized away.
        """

        try:
            with self._connect() as conn:
                self._init(conn)
                rows = conn.execute(
                    "SELECT * FROM enrollments WHERE state = 'committed' "
                    "AND credential_id = ? AND account_id = ? AND tenant_id = ? "
                    "AND endpoint_id = ?",
                    (credential_id, account_id, tenant_id, endpoint_id),
                ).fetchall()
                if canonical_base_url is not None:
                    rows = [r for r in rows if r["canonical_base_url"] == canonical_base_url]
                if len(rows) != 1:
                    return None
                row = rows[0]
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
            self._begin_write(conn)
            existing = conn.execute(
                "SELECT enrollment_generation, state FROM enrollments WHERE owner_key = ?",
                (_owner_key(owner),),
            ).fetchone()
            if (
                existing is not None
                and existing["state"] == "committed"
                and int(existing["enrollment_generation"]) >= owner.enrollment_generation
            ):
                # Monotonic fence: a re-commit at or below the live generation
                # would silently roll authority backwards past invalidate()
                # (QA H2-Q5 / reviewer R3). Going backwards needs a deliberate
                # operator act outside this seam, not a stale caller.
                raise StoreRefused(
                    "stale_enrollment",
                    "a commit must carry a strictly greater generation than the live one",
                )
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
            self._begin_write(conn)
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

    @staticmethod
    def _check_binding(owner: OwnerSnapshot, ref: MessageRef) -> None:
        """The item must be addressed to THIS owner before anything persists.

        The recipient endpoint is mandatory and exact; the account is compared
        when the server serves one. A missing or foreign binding refuses — an
        item addressed elsewhere is not this owner's delivery however the
        tenant reads (QA H2-Q1 / reviewer R4).
        """

        if ref.recipient_endpoint_id != owner.endpoint_id:
            raise StoreRefused(
                "wrong_recipient", "item recipient endpoint differs from (or is missing for) owner"
            )
        if ref.recipient_account_id is not None and ref.recipient_account_id != owner.account_id:
            raise StoreRefused("wrong_recipient", "item recipient account differs from owner")

    def merge_page(
        self, owner: OwnerSnapshot, refs: Sequence[MessageRef], cursor: InboxCursor | None
    ) -> None:
        """Persist a page's refs + checkpoint in ONE transaction.

        ``cursor`` is the server's NEXT cursor (``None`` = last page). Refs
        were already projected (no content); this method stores the whitelist
        columns only. A page that arrives after an invalidation is refused by
        the generation check. Each ref must bind to THIS owner (tenant,
        recipient endpoint, and recipient account when served): a foreign or
        unbound item refuses the whole page atomically — the caller may then
        retry under the new enrollment explicitly.
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            for ref in refs:
                if ref.tenant_id != owner.tenant_id:
                    raise StoreRefused("wrong_tenant", "item tenant differs from owner")
                self._check_binding(owner, ref)
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
            # walk_terminal records whether the LAST merged page was terminal:
            # finish_resync's witness, so an unfinished walk cannot be marked
            # complete (QA H2-Q3).
            conn.execute(
                "UPDATE sync SET inbox_checkpoint = ?, walk_terminal = ?, "
                "last_observed_at = ? WHERE owner_key = ?",
                (
                    None if cursor is None else str(cursor),
                    1 if cursor is None else 0,
                    time.time(),
                    _owner_key(owner),
                ),
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
            self._begin_write(conn)
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
                    "UPDATE sync SET pending_resync = 1, walk_terminal = 0, feed_seq = ? "
                    "WHERE owner_key = ?",
                    (int(cursor), _owner_key(owner)),
                )
            else:
                conn.execute(
                    "UPDATE sync SET feed_seq = ? WHERE owner_key = ?",
                    (int(cursor), _owner_key(owner)),
                )

    def finish_resync(self, owner: OwnerSnapshot, snapshot_checkpoint: InboxCursor) -> None:
        """Complete ONLY when a walk actually reached a terminal page.

        The witness: a resync is pending AND the LAST merged page was
        terminal (``walk_terminal``). The walk itself is H4's (it drives the
        pagination loop); H2 owns both the bookkeeping and this guard, so a
        caller that calls finish early — QA's secondary probe deliberately
        did — is REFUSED rather than silently recording an unfinished
        checkpoint. The baseline ``snapshot_checkpoint`` (captured before the
        walk) becomes the new resume point so arrivals during the walk are
        not skipped.
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            row = conn.execute(
                "SELECT pending_resync, walk_terminal FROM sync WHERE owner_key = ?",
                (_owner_key(owner),),
            ).fetchone()
            if row is None or not int(row["pending_resync"]) or not int(row["walk_terminal"]):
                raise StoreRefused(
                    "walk_incomplete",
                    "resync may complete only after every page was merged through a terminal page",
                )
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
        try:
            # The FULL immutable tuple is validated before anything is
            # persisted: a malformed hash/model/prompt is not "evidence with
            # gaps", it is not evidence (QA H2-Q7).
            receipt.validated()
        except TeamBridgeTypeError as error:
            raise StoreRefused("invalid_receipt", str(error)) from error
        if not isinstance(client_ref, str) or not client_ref:
            raise StoreRefused("invalid_client_ref", "client_ref must be a non-empty string")
        intent_id = str(uuid.uuid4())
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            existing = conn.execute(
                "SELECT * FROM ack_intents WHERE owner_key = ? AND message_id = ? AND target = ?",
                (_owner_key(owner), message_id, target),
            ).fetchone()
            if existing is not None:
                # The compared set is the WHOLE write-once tuple the server
                # checks (bridge_ack_service.go): verdict, model,
                # prompt_version, template_sha256, body_sha256, categories
                # (order-irrelevant) — plus the intent's own idempotency ref.
                # Comparing only a subset let a changed template/model reuse
                # the old intent silently (QA H2-Q7 / reviewer R2).
                same = (
                    existing["verdict"] == receipt.verdict
                    and existing["model"] == receipt.model
                    and existing["prompt_version"] == receipt.prompt_version
                    and existing["template_sha256"] == receipt.template_sha256
                    and existing["body_sha256"] == receipt.body_sha256
                    and _decode_categories_text(existing["categories"])
                    == sorted(receipt.categories)
                    and existing["client_ref"] == client_ref
                )
                if same:
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

    @staticmethod
    def _intent_dict(row: sqlite3.Row) -> dict[str, Any]:
        """The ONE read shape for an ack intent (shared by every lookup)."""

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
                return self._intent_dict(row)
        except sqlite3.Error:
            return None

    def find_intent_state(self, owner: OwnerSnapshot, message_id: str, target: str) -> str | None:
        """The ack-intent state for this exact tuple, ANY state (H3-Q1).

        The pipeline needs to know whether an UPSTREAM transport record for
        this message already exists (pending OR confirmed) before it may
        commit a fresh current local verdict: a confirmed historical read
        must never be turned into a new conflicting transport receipt.
        """

        try:
            with self._connect() as conn:
                self._init(conn)
                row = conn.execute(
                    "SELECT state FROM ack_intents WHERE owner_key = ? AND message_id = ? "
                    "AND target = ?",
                    (_owner_key(owner), message_id, target),
                ).fetchone()
                return row["state"] if row is not None else None
        except StoreRefused:
            return None

    def find_pending_intent(
        self, owner: OwnerSnapshot, message_id: str, target: str
    ) -> dict[str, Any] | None:
        """The namespace-bound recovery lookup (reviewer R5 / QA Q7-recovery).

        A crash between ``record_ack_intent`` and the caller receiving the
        intent id (the store call committed, the process died before the
        return) leaves a PENDING row that ``flush_ack`` cannot be pointed at.
        This finds it by the SAME exact-owner namespace plus the (message,
        target) the caller was performing, so recovery replays the STORED
        tuple and the STORED ref — never a fresh screening round. Confirmed
        and conflict intents are terminal and deliberately invisible here.
        """

        try:
            with self._connect() as conn:
                self._init(conn)
                row = conn.execute(
                    "SELECT * FROM ack_intents WHERE owner_key = ? AND message_id = ? "
                    "AND target = ? AND state = ?",
                    (_owner_key(owner), message_id, target, ACK_PENDING),
                ).fetchone()
                return None if row is None else self._intent_dict(row)
        except sqlite3.Error:
            return None

    def list_pending_intents(self, owner: OwnerSnapshot) -> list[dict[str, Any]]:
        """Every PENDING intent for this exact owner, oldest first (drain)."""

        try:
            with self._connect() as conn:
                self._init(conn)
                rows = conn.execute(
                    "SELECT * FROM ack_intents WHERE owner_key = ? AND state = ? "
                    "ORDER BY created_at, intent_id",
                    (_owner_key(owner), ACK_PENDING),
                ).fetchall()
                return [self._intent_dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def mark_ack_result(
        self,
        owner: OwnerSnapshot,
        intent_id: str,
        *,
        state: str,
        error_code: str | None,
        retry_after: float | None = None,
    ) -> None:
        """Record ONE flush outcome. Confirmed is terminal; pending stays replayable."""

        if state not in (ACK_CONFIRMED, ACK_CONFLICT, ACK_PENDING):
            raise StoreRefused("invalid_state", state)
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            # The write lock only SERIALIZES the write; authority must also be
            # checked INSIDE this same transaction, against the caller's OWN
            # snapshot (its generation is the fence): a post-await check that
            # passed moments earlier cannot vouch for the state now, so an
            # invalidated generation's completion must refuse here and leave
            # the intent exactly as it was — PENDING, stable tuple/ref — for
            # explicit reconciliation under the new generation (QA R3 H2-Q5).
            self._require_live(conn, owner)
            row = conn.execute(
                "SELECT state FROM ack_intents WHERE owner_key = ? AND intent_id = ?",
                (_owner_key(owner), intent_id),
            ).fetchone()
            if row is None:
                raise StoreRefused("unknown_intent", intent_id)
            if row["state"] == ACK_CONFIRMED and state != ACK_CONFIRMED:
                raise StoreRefused("already_confirmed", "confirmed intents are terminal")
            conn.execute(
                "UPDATE ack_intents SET state = ?, error_code = ?, retry_after = ? "
                "WHERE owner_key = ? AND intent_id = ?",
                (state, error_code, retry_after, _owner_key(owner), intent_id),
            )

    # -- H3 screening: budget ledger, receipts, policy, holds, breaker -----

    #: The frozen §8 budget, PER INDEPENDENT BUCKET: 40 logical screenings
    #: and (separately, conservatively) 40 actual inference HTTP dispatches,
    #: per rolling hour per Radient ACCOUNT, aggregated across every
    #: tenant/endpoint/credential/generation namespace for that account in
    #: the hub. Two buckets because one isolated logical screening can reach
    #: up to three wire attempts (auth re-resolve + fast-mode resend), and
    #: the physical cap must count those even when the logical one allows it.
    SCREENING_BUDGET_PER_HOUR = 40

    #: Floor for an account turn lease: the screening wall deadline is 45 s,
    #: so a lease SHORTER than this could expire under a live dispatch and
    #: admit a second worker mid-call. Short supplied TTLs are raised to
    #: safety (fail closed), never honoured as an unsafe parameter.
    _MIN_TURN_TTL = 60.0

    @staticmethod
    def _account_scope(owner: OwnerSnapshot) -> str:
        # The budget key is the ACCOUNT within its hub, deliberately NOT the
        # namespace: a new endpoint or a re-enrollment must not reset spend.
        return f"{owner.canonical_base_url}|{owner.account_id}"

    def _reserve_bucket(
        self,
        owner: OwnerSnapshot,
        *,
        kind: str,
        message_id: str | None,
        fingerprint: str | None,
    ) -> str:
        """Durably charge ONE attempt of ``kind`` BEFORE its I/O.

        Reservation IS spend: an uncertain send keeps its row (never
        refunded), the rolling window is counted from each row's own
        timestamp, so a clock rollback cannot erase reservations (a
        rolled-back clock makes the window MORE inclusive, which fails
        closed). Missing/corrupt schema or contention surfaces as a refusal
        — never a reset to 40.
        """

        now = time.time()
        attempt_id = str(uuid.uuid4())
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            scope = self._account_scope(owner)
            spent = conn.execute(
                "SELECT COUNT(*) FROM screening_attempts "
                "WHERE account_scope = ? AND kind = ? AND at > ?",
                (scope, kind, now - 3600.0),
            ).fetchone()[0]
            if int(spent) >= self.SCREENING_BUDGET_PER_HOUR:
                raise StoreRefused("budget_exhausted", f"{kind} attempts/hour/account exhausted")
            conn.execute(
                "INSERT INTO screening_attempts "
                "(attempt_id, kind, account_scope, owner_key, message_id, fingerprint, at) "
                "VALUES (?,?,?,?,?,?,?)",
                (attempt_id, kind, scope, _owner_key(owner), message_id, fingerprint, now),
            )
            return attempt_id

    def reserve_screening_attempt(
        self,
        owner: OwnerSnapshot,
        *,
        message_id: str | None = None,
        fingerprint: str | None = None,
        input_proof: Mapping[str, Any] | None = None,
    ) -> str:
        """The LOGICAL screening charge, optionally bound to message + tuple.

        The pipeline ALWAYS binds (message + full policy fingerprint); the
        unbound form exists for budget accounting talk (QA probes, breaker
        bookkeeping) and still charges the same bucket.

        ``input_proof`` (R8/Q3) registers the IMMUTABLE canonical input at
        reservation time — owner namespace/generation, body/template/version,
        provider/model/approved map, envelope/request/display digests and
        revision/epoch — as a ``proof:`` record in the same column; the
        single bind then DERIVES its DispatchJob and refuses any field that
        does not match the registered record BEFORE I/O.
        """

        stored = fingerprint
        if input_proof is not None:
            stored = _PROOF_PREFIX + json.dumps(
                dict(input_proof), sort_keys=True, separators=(",", ":")
            )
        return self._reserve_bucket(
            owner, kind="logical", message_id=message_id, fingerprint=stored
        )

    def reserve_http_attempt(
        self, owner: OwnerSnapshot, *, message_id: str | None, fingerprint: str | None = None
    ) -> str:
        """The PHYSICAL charge for ONE actual inference HTTP dispatch.

        The wire decorator calls this immediately before every nested request
        (primary, auth re-resolve, fast-mode resend): the logical bucket's
        allowance does not waive the physical cap, and a refusal is raised
        at the wire so no request leaves.
        """

        return self._reserve_bucket(
            owner, kind="http", message_id=message_id, fingerprint=fingerprint
        )

    def publish_screening_policy(
        self,
        owner: OwnerSnapshot,
        *,
        version: str,
        template_sha256: str,
        default_model: str | None,
        revision: str,
        expected_epoch: int | None = None,
        create_only: bool = False,
    ) -> int:
        """Publish the current accepted policy; return its durable epoch.

        TRUSTED PUBLISHER entry point (a live configuration source), NOT a
        screening caller: the pipeline never publishes from its caller
        object, it only CONSUMES the published row. With ``expected_epoch``
        this is a CAS guard — a publisher whose view is stale refuses rather
        than rewriting current authority.

        The epoch is the TRUSTED clock for admission: the pipeline captures
        it before dispatch and ``finish_screening`` re-reads it inside the
        receipt-writing transaction, so a policy published mid-run makes the
        old epoch stale and the commit refuses (H3-Q3). Monotonic: the epoch
        only advances, and a template FLOOR is enforced — once a template
        hash is accepted, a different one may only arrive as a later
        explicitly-known version (today only byte-equal bridge-screen-v1 is
        accepted upstream; the floor row is what a future version bumps).
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            key = _owner_key(owner)
            row = conn.execute(
                "SELECT version, template_sha256, default_model, revision, epoch "
                "FROM screening_policy WHERE owner_key = ?",
                (key,),
            ).fetchone()
            if row is not None and create_only:
                # FIRST-PUBLICATION CAS: a bootstrap must never race another
                # writer into owning the row (the caller re-reads and matches).
                raise StoreRefused("policy_exists", "bootstrap lost the first-publication race")
            if row is not None and _version_rank(version) < _version_rank(str(row["version"])):
                # MONOTONIC TEMPLATE FLOOR: a publisher may advance the
                # accepted template version, never rewind it.
                raise StoreRefused("version_floor", "template version would move backwards")
            if (
                row is not None
                and str(row["version"]) == version
                and str(row["template_sha256"]) != template_sha256
            ):
                # IMMUTABLE HASH WITHIN A VERSION (R8/Q3): a version's
                # template is fixed once accepted — a same-version, different
                # hash is a TYPED REFUSAL that preserves the original, never
                # an epoch-advancing overwrite. Rank and CAS alone cannot
                # catch this, because the rewritten row keeps its rank.
                raise StoreRefused(
                    "template_hash_conflict",
                    "same version with a different template hash",
                )
            if row is None:
                conn.execute(
                    "INSERT INTO screening_policy "
                    "(owner_key, version, template_sha256, default_model, revision, epoch, "
                    "accepted_at) VALUES (?,?,?,?,?,1,?)",
                    (key, version, template_sha256, default_model, revision, time.time()),
                )
                return 1
            if expected_epoch is not None and int(row["epoch"]) != expected_epoch:
                raise StoreRefused(
                    "policy_stale",
                    "published epoch moved under the publisher's view",
                )
            unchanged = (
                row["version"] == version
                and row["template_sha256"] == template_sha256
                and row["default_model"] == default_model
                and row["revision"] == revision
            )
            if unchanged:
                return int(row["epoch"])
            conn.execute(
                "UPDATE screening_policy SET version = ?, template_sha256 = ?, "
                "default_model = ?, revision = ?, epoch = ?, accepted_at = ? "
                "WHERE owner_key = ?",
                (
                    version,
                    template_sha256,
                    default_model,
                    revision,
                    int(row["epoch"]) + 1,
                    time.time(),
                    key,
                ),
            )
            return int(row["epoch"]) + 1

    def current_screening_policy(self, owner: OwnerSnapshot) -> dict[str, Any] | None:
        """The currently published policy row (contentless policy metadata)."""

        try:
            with self._connect() as conn:
                self._init(conn)
                row = conn.execute(
                    "SELECT version, template_sha256, default_model, revision, epoch "
                    "FROM screening_policy WHERE owner_key = ?",
                    (_owner_key(owner),),
                ).fetchone()
                return dict(row) if row is not None else None
        except sqlite3.Error:
            return None

    def find_screening_receipt(
        self,
        owner: OwnerSnapshot,
        *,
        body_sha256: str,
        template_sha256: str,
        model: str,
        admission_revision: str,
        envelope_digest: str,
    ) -> dict[str, Any] | None:
        """Reuse cache, keyed by the EXACT full admission fingerprint.

        A changed body, template, model, revision OR envelope digest misses
        (never replays a stale allow/reject); the cache holds no body, reason
        or reply text — only the closed receipt fields and the hashes.
        """

        try:
            with self._connect() as conn:
                self._init(conn)
                row = conn.execute(
                    "SELECT verdict, model, prompt_version, template_sha256, body_sha256, "
                    "categories, envelope_digest, admission_revision, policy_epoch, "
                    "creator_generation "
                    "FROM screening_receipts WHERE owner_key = ? AND body_sha256 = ? "
                    "AND template_sha256 = ? AND model = ? AND admission_revision = ? "
                    "AND envelope_digest = ?",
                    (
                        _owner_key(owner),
                        body_sha256,
                        template_sha256,
                        model,
                        admission_revision,
                        envelope_digest,
                    ),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "verdict": row["verdict"],
                    "model": row["model"],
                    "policy_epoch": row["policy_epoch"],
                    "creator_generation": row["creator_generation"],
                    "prompt_version": row["prompt_version"],
                    "template_sha256": row["template_sha256"],
                    "body_sha256": row["body_sha256"],
                    "categories": json.loads(row["categories"]),
                    "envelope_digest": row["envelope_digest"],
                    "admission_revision": row["admission_revision"],
                }
        except (sqlite3.Error, json.JSONDecodeError):
            # An unreadable cache row is a MISS (fresh screening may re-bill
            # under normal accounting); it can never silently certify.
            return None

    def bind_attempt_fingerprint(
        self,
        owner: OwnerSnapshot,
        *,
        attempt_id: str,
        fingerprint: str,
        job: Any = None,
    ) -> None:
        """Bind a reserved LOGICAL attempt to the immutable job digest.

        The attempt is reserved before the job exists (the job carries its
        attempt id); this narrow update records the canonical digest so the
        physical reservation can REQUIRE exact logical-job equality inside
        its own transaction.
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            row = conn.execute(
                "SELECT fingerprint, committed_message_id FROM screening_attempts "
                "WHERE attempt_id = ? AND owner_key = ? AND kind = 'logical'",
                (attempt_id, _owner_key(owner)),
            ).fetchone()
            if row is None:
                raise StoreRefused("unknown_attempt", attempt_id)
            if row["committed_message_id"] is not None:
                # A committed attempt is sealed: no rebinding of any kind.
                raise StoreRefused("write_once_conflict", "attempt already committed")
            raw = row["fingerprint"]
            if isinstance(raw, str) and raw.startswith(_PROOF_PREFIX):
                # REGISTERED INPUT PROOF (R8/Q3): the one legal bind must
                # match the IMMUTABLE record registered at reservation —
                # every compared field of the DispatchJob is checked BEFORE
                # any I/O, so a mutation between reservation and bind refuses
                # instead of dispatching. No comment-based "forged input is
                # impractical" argument: the fields are compared.
                try:
                    registered = json.loads(raw[len(_PROOF_PREFIX) :])
                except (TypeError, ValueError) as error:
                    raise StoreRefused(
                        "proof_corrupt", "registered input proof is unreadable"
                    ) from error
                derived = _canonical_job_digest(job)
                if derived != fingerprint or derived != getattr(job, "digest", None):
                    # The bind's digest argument must be the CANONICAL
                    # derivation from the job's own fields (R10/B): a forged
                    # nonempty/zero digest refuses here, before any I/O.
                    raise StoreRefused(
                        "attempt_mismatch", "fingerprint is not the canonical job digest"
                    )
                mismatched = _input_proof_mismatch(registered, fingerprint, job)
                if mismatched is not None:
                    raise StoreRefused("attempt_mismatch", mismatched)
                # The registered row is UNBOUND: fall through to the single
                # bind below, RETAINING the original registration beside the
                # bound digest (R9/Q3-5): later native reservations compare
                # against the retained record, never a reconstructed one.
                bound_value = f"{_JOB_BOUND_PREFIX}{fingerprint}" f"|{_PROOF_PREFIX}" + json.dumps(
                    registered, sort_keys=True, separators=(",", ":")
                )
                conn.execute(
                    "UPDATE screening_attempts SET fingerprint = ? "
                    "WHERE attempt_id = ? AND owner_key = ? AND kind = 'logical'",
                    (bound_value, attempt_id, _owner_key(owner)),
                )
                return
            if isinstance(raw, str) and raw.startswith(_JOB_BOUND_PREFIX):
                # BOUND ALREADY: one bind was legitimate; only the IDENTICAL
                # digest may replay, and a changed pending bind refuses while
                # the original binding stays on record. The retained record
                # itself must still decode (R10/B).
                bound_digest, retained = _decode_bound_proof(raw)
                if bound_digest != fingerprint:
                    raise StoreRefused(
                        "write_once_conflict", "logical attempt already bound to another job"
                    )
                if job is not None:
                    mismatched = _input_proof_mismatch(retained, bound_digest, job)
                    if mismatched is not None:
                        raise StoreRefused("attempt_mismatch", mismatched)
                return
            # FIRST bind only: the row still holds the packed logical
            # reservation fingerprint (or NULL for a legacy row); the bind
            # marker makes the state explicit so a second bind can tell
            # "packed logical fp" from "bound job digest".
            raise StoreRefused(
                "unproved_attempt",
                "reservation without a registered input proof cannot bind a job",
            )

    def reserve_screening_dispatch(
        self, owner: OwnerSnapshot, *, message_id: str, job: Any, turn: str
    ) -> str:
        """The PHYSICAL charge for one admitted dispatch — job-REQUIRED.

        ONE conditional transaction (no gap between verification and the
        charge): the live owner generation must match the job's, the CURRENT
        published policy must still equal the job's revision/template/
        default/epoch (a trusted publisher moving it refuses before any
        I/O), the LOGICAL attempt row must exist carrying the SAME canonical
        digest under the same message, and the account turn token must still
        hold this account's lease. Anything missing/stale refuses with NO
        physical row, so no handler can ever run behind a stale authority.
        """

        now = time.time()
        attempt_id = str(uuid.uuid4())
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            if self._live_generation(conn, owner) != job.owner_generation:
                raise StoreRefused("owner_changed", "job owner generation is stale")
            policy = conn.execute(
                "SELECT version, template_sha256, default_model, revision, epoch "
                "FROM screening_policy WHERE owner_key = ?",
                (_owner_key(owner),),
            ).fetchone()
            if policy is None:
                raise StoreRefused("policy_stale", "no published policy for this job")
            if not job.provider or not job.approved:
                # The reservation must check what the job CLAIMS (R6/Q3): a
                # job with no canonical provider/approvals is not authority.
                raise StoreRefused("policy_stale", "job lacks canonical provider/approvals")
            if (
                policy["revision"] != job.revision
                or policy["template_sha256"] != job.template_sha256
                or policy["default_model"] != job.model
                or int(policy["epoch"]) != int(job.epoch)
            ):
                raise StoreRefused("policy_stale", "published policy moved after job mint")
            logical = conn.execute(
                "SELECT fingerprint, message_id FROM screening_attempts "
                "WHERE attempt_id = ? AND owner_key = ? AND kind = 'logical'",
                (job.attempt_id, _owner_key(owner)),
            ).fetchone()
            raw_logical = logical["fingerprint"] if logical is not None else None
            if logical is None or _bound_digest(raw_logical) != job.digest:
                raise StoreRefused("unknown_attempt", "logical job binding missing")
            # Decode and COMPARE the retained registration (R10/B): a null,
            # corrupt, altered or extra-field proof refuses BEFORE any
            # physical charge, and the digest must be the canonical
            # derivation from the job's fields — not a keyword.
            bound_digest, registered = _decode_bound_proof(raw_logical)
            derived = _canonical_job_digest(job)
            if bound_digest != job.digest or derived != getattr(job, "digest", None):
                raise StoreRefused(
                    "attempt_mismatch", "bound digest is not the canonical job digest"
                )
            mismatched = _input_proof_mismatch(registered, job.digest, job)
            if mismatched is not None:
                raise StoreRefused("attempt_mismatch", mismatched)
            if logical["message_id"] != message_id:
                raise StoreRefused("attempt_mismatch", "job message mismatch")
            schedule = conn.execute(
                "SELECT lease_holder, lease_expires FROM screening_schedule WHERE owner_key = ?",
                (self._account_scope(owner),),
            ).fetchone()
            if (
                schedule is None
                or schedule["lease_holder"] != turn
                or float(schedule["lease_expires"] or 0.0) <= now
            ):
                raise StoreRefused("retry_later", "account turn not held by this job")
            scope = self._account_scope(owner)
            spent = conn.execute(
                "SELECT COUNT(*) FROM screening_attempts "
                "WHERE account_scope = ? AND kind = 'http' AND at > ?",
                (scope, now - 3600.0),
            ).fetchone()[0]
            if int(spent) >= self.SCREENING_BUDGET_PER_HOUR:
                raise StoreRefused("budget_exhausted", "http attempts/hour/account exhausted")
            conn.execute(
                "INSERT INTO screening_attempts "
                "(attempt_id, kind, account_scope, owner_key, message_id, fingerprint, at) "
                "VALUES (?,?,?,?,?,?,?)",
                (attempt_id, "http", scope, _owner_key(owner), message_id, job.digest, now),
            )
            return attempt_id

    def consume_screening_receipt(
        self,
        owner: OwnerSnapshot,
        *,
        body_sha256: str,
        template_sha256: str,
        model: str,
        admission_revision: str,
        envelope_digest: str,
    ) -> dict[str, Any] | None:
        """ONE native consumption point for a cached verdict (A2/A3).

        In a single transaction: the live committed owner must still be the
        caller's exact generation; the receipt must match the full
        fingerprint; the PUBLISHED policy must still equal the receipt's own
        template/model/revision AND the receipt's epoch and creator
        generation must still be current. Anything else returns ``None`` —
        the cache can never outlive the authority that minted it.
        """

        try:
            with self._connect() as conn:
                self._init(conn)
                # EXPLICIT bounded transaction: SQLite autocommits plain
                # SELECTs, so without BEGIN IMMEDIATE an independent Store can
                # invalidate the owner or advance the policy BETWEEN these
                # reads (QA's actual two-connection race). IMMEDIATE takes the
                # write lock so the whole consumption is one linearized
                # snapshot: either the change happened strictly before (we
                # see it and refuse) or it waits until we have returned.
                self._begin_write(conn)
                key = _owner_key(owner)
                if self._live_generation(conn, owner) != owner.enrollment_generation:
                    return None
                policy = conn.execute(
                    "SELECT version, template_sha256, default_model, revision, epoch "
                    "FROM screening_policy WHERE owner_key = ?",
                    (key,),
                ).fetchone()
                if policy is None:
                    return None
                row = conn.execute(
                    "SELECT verdict, model, prompt_version, template_sha256, body_sha256, "
                    "categories, envelope_digest, admission_revision, policy_epoch, "
                    "creator_generation FROM screening_receipts "
                    "WHERE owner_key = ? AND body_sha256 = ? AND template_sha256 = ? "
                    "AND model = ? AND admission_revision = ? AND envelope_digest = ?",
                    (key, body_sha256, template_sha256, model, admission_revision, envelope_digest),
                ).fetchone()
                if row is None:
                    return None
                if (
                    policy["template_sha256"] != row["template_sha256"]
                    or policy["default_model"] != row["model"]
                    or policy["revision"] != row["admission_revision"]
                    or int(policy["epoch"]) != int(row["policy_epoch"])
                    or int(row["creator_generation"]) != owner.enrollment_generation
                ):
                    return None
                # EMPTY SQL TEXT is damaged proof, not an empty tuple
                # (R7/Q2): coalescing it to [] would admit a receipt whose
                # categories were never written. Only valid JSON survives —
                # "[]" is the only legitimate empty.
                try:
                    decoded = _decode_categories_text(row["categories"])
                except (TypeError, ValueError) as error:
                    # A MATCHED row whose metadata cannot be decoded is
                    # DAMAGED authority, not an absent proof: refusing (hold)
                    # keeps the caller from re-screening and DOUBLE-BILLING
                    # behind a corrupted receipt.
                    raise StoreRefused("proof_corrupt", str(error)) from None
                if (
                    not isinstance(decoded, list)
                    or not all(isinstance(item, str) for item in decoded)
                    or not set(decoded) <= _CLOSED_CATEGORIES
                ):
                    # Unknown category strings (or wrong shapes) can never be
                    # handed to a receipt: typed uncertainty, zero spend.
                    raise StoreRefused("proof_corrupt", "categories failed validation")
                if row["verdict"] not in ("allow", "reject"):
                    raise StoreRefused("proof_corrupt", "unknown verdict")
                if row["verdict"] == "allow" and decoded:
                    # An ALLOW with rejection categories is internally
                    # inconsistent (injection is a reject): fail closed
                    # rather than deliver on damaged metadata. A reject with
                    # an empty tuple is NOT forbidden and stays valid.
                    raise StoreRefused("proof_corrupt", "allow verdict with categories")
                return {
                    "verdict": row["verdict"],
                    "model": row["model"],
                    "prompt_version": row["prompt_version"],
                    "template_sha256": row["template_sha256"],
                    "body_sha256": row["body_sha256"],
                    "categories": decoded,
                    "envelope_digest": row["envelope_digest"],
                    "admission_revision": row["admission_revision"],
                    "policy_epoch": row["policy_epoch"],
                    "creator_generation": row["creator_generation"],
                }
        except sqlite3.Error as error:
            # An interleaved writer that could not take the lock (or any
            # engine-level failure) leaves the authority UNVERIFIED, which is
            # a refusal — never a silent "no row" that lets a stale caller
            # re-screen behind a concurrent authority change.
            raise StoreRefused("state_busy", str(error)) from None
        except StoreRefused:
            raise

    def finish_screening(
        self,
        owner: OwnerSnapshot,
        message_id: str,
        *,
        attempt_id: str,
        fingerprint: str,
        expected_epoch: int,
        receipt: ScreeningReceipt | None = None,
        envelope_digest: str | None = None,
        hold_code: str | None = None,
        retry_at: float | None = None,
        verify: Any = None,
        admission_revision: str | None = None,
        turn: str | None = None,
    ) -> None:
        """Commit ONE logical outcome, fenced on every binding that matters.

        Inside ONE write transaction: live-owner/generation (native fence,
        H2 discipline), the attempt belongs to THIS owner AND message AND
        policy fingerprint AND is of the logical kind, the published policy
        epoch is still the one captured before dispatch, and the write is
        WRITE-ONCE per message: an existing receipt may only be replayed
        identically (idempotent recovery) — never overwritten with another
        verdict, never used to commit a second message (H3-Q6). A receipt
        persists only after full tuple validation; a hold records its closed
        code durably so a refreshed inbox projection cannot erase it.
        """

        if (receipt is None) == (hold_code is None):
            raise StoreRefused("invalid_outcome", "exactly one of receipt/hold_code")
        if receipt is not None:
            try:
                receipt.validated()
            except TeamBridgeTypeError as error:
                raise StoreRefused("invalid_receipt", str(error)) from error
            if envelope_digest is None:
                raise StoreRefused("invalid_outcome", "receipt outcome needs the envelope digest")
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            if turn is not None:
                # ATOMIC final admission (R7/Q15-6): the turn/frame fence
                # runs INSIDE this same write transaction, so a successor
                # releasing/claiming between the caller's last check and
                # this commit can never let the old pipeline admit — the
                # receipt is atomic with ownership. The initiating frame
                # must match the token's own frame digest.
                row = conn.execute(
                    "SELECT lease_holder, lease_expires FROM screening_schedule "
                    "WHERE owner_key = ?",
                    (self._account_scope(owner),),
                ).fetchone()
                if row is None or row["lease_holder"] != turn or not _owner_frame_ok(turn, owner):
                    raise StoreRefused("turn_lost", "account turn moved before commit")
                if float(row["lease_expires"] or 0.0) <= time.time():
                    # An EXPIRED turn with no successor must not admit a
                    # receipt either (R8 static adjunct to the successor
                    # case): the lease must be LIVE inside this transaction.
                    raise StoreRefused("turn_lost", "account turn expired before commit")
            if verify is not None and not verify():
                # The live source moved between the caller's last check and
                # this transaction: nothing commits (A2/A3 bridge — the live
                # check runs INSIDE the native write, not only outside it).
                raise StoreRefused("policy_stale", "live policy changed at commit")
            key = _owner_key(owner)
            attempt = conn.execute(
                "SELECT kind, message_id, fingerprint, committed_message_id "
                "FROM screening_attempts "
                "WHERE attempt_id = ? AND owner_key = ?",
                (attempt_id, key),
            ).fetchone()
            if attempt is None:
                # An outcome for an attempt this owner never reserved would
                # fabricate spend evidence; refuse instead.
                raise StoreRefused("unknown_attempt", attempt_id)
            if (
                attempt["kind"] != "logical"
                or attempt["message_id"] != message_id
                or _bound_digest(attempt["fingerprint"]) != fingerprint
            ):
                # One reserved ID is bound to ONE message/policy tuple; it
                # cannot commit another message or a differently-fingerprinted
                # outcome.
                raise StoreRefused("attempt_mismatch", attempt_id)
            if isinstance(attempt["fingerprint"], str) and attempt["fingerprint"].startswith(
                _JOB_BOUND_PREFIX
            ):
                # The retained registration must still DECODE and AGREE with
                # the outcome being committed (R10/B): corruption injected
                # after the POST refuses the receipt — the charged logical
                # and physical counts stay exactly as spent, no refund, no
                # reconstruction.
                bound_digest, registered = _decode_bound_proof(attempt["fingerprint"])
                if bound_digest != fingerprint:
                    raise StoreRefused("attempt_mismatch", "bound digest changed before finish")
                if registered.get("message") != message_id:
                    raise StoreRefused("proof_corrupt", "retained proof names another message")
                # R12/F1: ONE canonical re-derivation from the RETAINED
                # registration (the same encoder and field order as the job
                # mint), with the attempt id taken from the native row and
                # the turn token from the finishing transaction. This closes
                # EVERY registered field at once — staged provider/model/
                # request/display/logical references and approved-map edits
                # that keep membership included — against the digest bound
                # at the physical reservation. Any field altered after the
                # POST cannot re-derive the binding.
                turn_candidates = [turn] if isinstance(turn, str) and turn else []
                lease_row = conn.execute(
                    "SELECT lease_holder FROM screening_schedule WHERE owner_key = ?",
                    (self._account_scope(owner),),
                ).fetchone()
                if lease_row is not None and lease_row["lease_holder"]:
                    turn_candidates.append(lease_row["lease_holder"])
                re_derived = [
                    _canonical_proof_digest(registered, attempt_id, candidate)
                    for candidate in turn_candidates
                ]
                if fingerprint not in re_derived:
                    raise StoreRefused(
                        "attempt_mismatch",
                        "retained registration does not re-derive the bound digest",
                    )
                # The PACKED logical reference is the pre-bind fingerprint the
                # reservation pinned (revision | envelope | approved
                # identities); it is not part of the job digest, so it is
                # re-derived here with the same helper the scheduler used.
                from local_operator.team_bridge.screening import _identities_digest

                approved_set = frozenset(str(entry) for entry in (registered.get("approved") or []))
                logical_expected = (
                    f"{registered.get('revision')}|{registered.get('envelope')}"
                    f"|{_identities_digest(approved_set)}"
                )
                if registered.get("logical") != logical_expected:
                    raise StoreRefused(
                        "proof_corrupt",
                        "retained logical reference does not re-derive",
                    )
                # FULL closed comparison against the NATIVE authority the
                # finish actually has (R11/1): owner namespace, tenant, the
                # served model/template/version the receipt carries, approval
                # agreement, revision and current policy epoch. Without this,
                # a valid closed record whose NON-body fields were altered
                # after the POST committed an allow receipt.
                owner_expected = (
                    f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
                    f"|{owner.endpoint_id}|{owner.canonical_base_url}"
                    f"|{owner.enrollment_generation}"
                )
                if registered.get("owner") != owner_expected:
                    raise StoreRefused(
                        "proof_corrupt", "retained proof owner does not match the finishing owner"
                    )
                if registered.get("tenant") != owner.tenant_id:
                    raise StoreRefused(
                        "proof_corrupt", "retained proof tenant does not match the finishing owner"
                    )
                if registered.get("epoch") != int(expected_epoch):
                    raise StoreRefused(
                        "policy_stale", "retained proof epoch is not the finishing epoch"
                    )
                if receipt is not None:
                    if registered.get("body") != receipt.body_sha256:
                        raise StoreRefused(
                            "proof_corrupt", "retained proof body does not match the receipt"
                        )
                    # (The served model is checked by APPROVED-MAP
                    # membership below, never by equality to the registered
                    # default: a legitimate resolver alias serves an
                    # approved identity that need not be the default string.)
                    if registered.get("template") != receipt.template_sha256:
                        raise StoreRefused(
                            "proof_corrupt",
                            "retained proof template does not match the verdict's template",
                        )
                    if registered.get("version") != receipt.prompt_version:
                        raise StoreRefused(
                            "proof_corrupt",
                            "retained proof version does not match the verdict's prompt version",
                        )
                    if receipt.model not in list(registered.get("approved") or []):
                        raise StoreRefused(
                            "proof_corrupt",
                            "the served model is not in the retained approved map",
                        )
                if (
                    receipt is not None
                    and envelope_digest is not None
                    and registered.get("envelope") != envelope_digest
                ):
                    raise StoreRefused(
                        "proof_corrupt", "retained proof envelope does not match the receipt"
                    )
                if (
                    admission_revision is not None
                    and registered.get("revision") != admission_revision
                ):
                    raise StoreRefused(
                        "proof_corrupt",
                        "retained proof revision is not the committed admission revision",
                    )
            policy_row = conn.execute(
                "SELECT epoch FROM screening_policy WHERE owner_key = ?", (key,)
            ).fetchone()
            if policy_row is None or int(policy_row["epoch"]) != expected_epoch:
                # A policy published after dispatch invalidates this proof.
                raise StoreRefused("policy_stale", "accepted policy changed during the run")
            if receipt is not None:
                existing = conn.execute(
                    "SELECT verdict, model, prompt_version, template_sha256, body_sha256, "
                    "categories, envelope_digest, admission_revision "
                    "FROM screening_receipts WHERE owner_key = ? AND message_id = ?",
                    (key, message_id),
                ).fetchone()
                if existing is not None:
                    same = (
                        existing["verdict"] == receipt.verdict
                        and existing["model"] == receipt.model
                        and existing["prompt_version"] == receipt.prompt_version
                        and existing["template_sha256"] == receipt.template_sha256
                        and existing["body_sha256"] == receipt.body_sha256
                        and tuple(sorted(_decode_categories_text(existing["categories"])))
                        == tuple(sorted(receipt.categories))
                        and existing["envelope_digest"] == envelope_digest
                    )
                    if not same:
                        # Write-once: a committed allow is never replaced by a
                        # reject (or any rewrite) through this path.
                        raise StoreRefused("write_once_conflict", message_id)
                else:
                    conn.execute(
                        "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
                        "template_sha256, prompt_version, model, verdict, categories, "
                        "envelope_digest, admission_revision, policy_epoch, "
                        "creator_generation, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            key,
                            message_id,
                            receipt.body_sha256,
                            receipt.template_sha256,
                            receipt.prompt_version,
                            receipt.model,
                            receipt.verdict,
                            json.dumps(sorted(receipt.categories)),
                            envelope_digest,
                            # The revision is an explicit parameter now: the
                            # canonical job digest is no longer a packed
                            # "revision|digest" string the old helper could
                            # parse (R5), and a receipt must store the
                            # snapshot's own revision so cache consumption
                            # matches on it.
                            admission_revision or _fingerprint_revision(fingerprint),
                            expected_epoch,
                            owner.enrollment_generation,
                            time.time(),
                        ),
                    )
                conn.execute(
                    "UPDATE screening_attempts SET committed_message_id = ? "
                    "WHERE attempt_id = ?",
                    (message_id, attempt_id),
                )
                conn.execute(
                    "DELETE FROM screening_holds WHERE owner_key = ? AND message_id = ?",
                    (key, message_id),
                )
            else:
                # WRITE-ONCE per attempt (reviewer R7): once this attempt has
                # committed a receipt, a later hold for the same attempt is
                # refused — an attempt's outcome is exactly one thing.
                if attempt["committed_message_id"] is not None:
                    raise StoreRefused(
                        "write_once_conflict", "attempt already committed its outcome"
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO screening_holds (owner_key, message_id, "
                    "hold_code, retry_at, created_at) VALUES (?,?,?,?,?)",
                    (key, message_id, hold_code, retry_at, time.time()),
                )

    def screen_schedule(self, owner: OwnerSnapshot) -> dict[str, Any] | None:
        """The durable scheduling row (cursor + lease), or None when absent.

        Keyed by the ACCOUNT SCOPE (hub URL + account id) — the same key the
        aggregate budget uses. A credential, tenant or endpoint change must
        not create a second rotation/lease for the same Radient account
        (QA: two valid namespaces both owned a turn).
        """

        with self._connect() as conn:
            self._init(conn)
            row = conn.execute(
                "SELECT last_sender, lease_holder, lease_expires FROM screening_schedule "
                "WHERE owner_key = ?",
                (self._account_scope(owner),),
            ).fetchone()
            return dict(row) if row is not None else None

    def set_screen_cursor(self, owner: OwnerSnapshot, last_sender: str) -> None:
        """Persist the rotation cursor (metadata only, never message content)."""

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            conn.execute(
                "INSERT INTO screening_schedule (owner_key, last_sender) VALUES (?, ?) "
                "ON CONFLICT(owner_key) DO UPDATE SET last_sender = excluded.last_sender",
                (self._account_scope(owner), last_sender),
            )

    def claim_screen_turn(
        self, owner: OwnerSnapshot, *, holder_prefix: str, now: float, ttl: float
    ) -> str | None:
        """Conditional one-at-a-time lease (CAS); returns an UNFORGEABLE token.

        The minted token (never a reusable holder label) is what identifies
        THIS turn: release/renew compare the exact token, so a previous
        holder — even one sharing a process/worker label after re-enrollment
        or a nested acquisition — can never clear a successor's lease. The
        TTL is floored at ``_MIN_TURN_TTL`` so an admitted dispatch (bounded
        by the screening wall time) cannot outlive its own lease and let a
        second worker in mid-call; short TTLs fail CLOSED, not open.
        """

        ttl = max(ttl, self._MIN_TURN_TTL)
        # The token binds the INITIATING frame (credential/tenant/endpoint/
        # generation), not just hub/account (R7/Q15-6): release/renew/match
        # re-derive the caller's frame and compare it to the token's, so a
        # DIFFERENT but LIVE owner on the same account holding this exact
        # string still cannot borrow, release or renew the turn.
        token = f"{holder_prefix}:{uuid.uuid4()}:{_owner_frame_digest(owner)}"
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            row = conn.execute(
                "SELECT lease_holder, lease_expires FROM screening_schedule WHERE owner_key = ?",
                (self._account_scope(owner),),
            ).fetchone()
            if row is not None:
                held_by = row["lease_holder"]
                expires = float(row["lease_expires"] or 0.0)
                if held_by and expires > now:
                    return None
            conn.execute(
                "INSERT INTO screening_schedule (owner_key, lease_holder, lease_expires) "
                "VALUES (?, ?, ?) ON CONFLICT(owner_key) DO UPDATE SET "
                "lease_holder = excluded.lease_holder, lease_expires = excluded.lease_expires",
                (self._account_scope(owner), token, now + ttl),
            )
            return token

    def screen_turn_matches(self, owner: OwnerSnapshot, *, turn: str, now: float) -> bool:
        """Whether ``turn`` still holds this account's live lease."""

        with self._connect() as conn:
            self._init(conn)
            # Generation fence (R6/Q15): an OLD owner snapshot that kept a
            # successor's capability token must not be able to act on it —
            # only a snapshot whose enrollment generation is still LIVE may.
            if self._live_generation(conn, owner) != owner.enrollment_generation:
                return False
            # Initiating-frame fence (R7/Q15-6): the token must carry THIS
            # caller's own frame — a different live owner given the token
            # string cannot borrow it.
            if not _owner_frame_ok(turn, owner):
                return False
            row = conn.execute(
                "SELECT lease_holder, lease_expires FROM screening_schedule WHERE owner_key = ?",
                (self._account_scope(owner),),
            ).fetchone()
            if row is None or row["lease_holder"] != turn:
                return False
            return float(row["lease_expires"] or 0.0) > now

    def release_screen_turn(self, owner: OwnerSnapshot, *, turn: str) -> bool:
        """Conditional release: only the EXACT live token can clear the lease."""

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            if self._live_generation(conn, owner) != owner.enrollment_generation:
                return False
            if not _owner_frame_ok(turn, owner):
                return False
            row = conn.execute(
                "SELECT lease_holder FROM screening_schedule WHERE owner_key = ?",
                (self._account_scope(owner),),
            ).fetchone()
            if row is None or row["lease_holder"] != turn:
                return False
            conn.execute(
                "UPDATE screening_schedule SET lease_holder = NULL, lease_expires = NULL "
                "WHERE owner_key = ?",
                (self._account_scope(owner),),
            )
            return True

    def renew_screen_turn(self, owner: OwnerSnapshot, *, turn: str, now: float, ttl: float) -> bool:
        """Extend the SAME turn's lease (never a different holder's)."""

        ttl = max(ttl, self._MIN_TURN_TTL)
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            if self._live_generation(conn, owner) != owner.enrollment_generation:
                return False
            if not _owner_frame_ok(turn, owner):
                return False
            row = conn.execute(
                "SELECT lease_holder FROM screening_schedule WHERE owner_key = ?",
                (self._account_scope(owner),),
            ).fetchone()
            if row is None or row["lease_holder"] != turn:
                return False
            conn.execute(
                "UPDATE screening_schedule SET lease_expires = ? WHERE owner_key = ?",
                (now + ttl, self._account_scope(owner)),
            )
            return True

    def schema_problem(self) -> str | None:
        """The refusal code of the CURRENT schema/state, or None when sound.

        ``resolve_owner`` (H2 discipline) reports a damaged or unknown store
        as NO AUTHORITY (None) rather than an error; callers that must name
        WHY nothing resolved (a durable hold reason, H3-Q8) ask this next.
        """

        try:
            with self._connect() as conn:
                self._init(conn)
            return None
        except StoreRefused as failure:
            return failure.code
        except sqlite3.Error:
            return "state_unreadable"

    def persist_hold(
        self,
        owner: OwnerSnapshot,
        message_id: str,
        hold_code: str,
        *,
        retry_at: float | None = None,
    ) -> None:
        """Durably quarantine a VALID-LIVE-OWNER early refusal (H3-Q8).

        Early holds (no usable default, template floor mismatch, budget
        exhaustion) happen before any attempt row exists; their per-item
        reason must still survive an inbox refresh instead of disappearing.
        An invalid owner gets no fresh writes at all (this method refuses —
        a non-resolving namespace must not accumulate state).
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            conn.execute(
                "INSERT OR REPLACE INTO screening_holds (owner_key, message_id, hold_code, "
                "retry_at, created_at) VALUES (?,?,?,?,?)",
                (_owner_key(owner), message_id, hold_code, retry_at, time.time()),
            )

    def screening_hold(self, owner: OwnerSnapshot, message_id: str) -> dict[str, Any] | None:
        """The durable hold row, if any (H4 surfaces it; H3 only writes it)."""

        try:
            with self._connect() as conn:
                self._init(conn)
                row = conn.execute(
                    "SELECT hold_code, retry_at, created_at FROM screening_holds "
                    "WHERE owner_key = ? AND message_id = ?",
                    (_owner_key(owner), message_id),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "hold_code": row["hold_code"],
                    "retry_at": row["retry_at"],
                    "created_at": row["created_at"],
                }
        except sqlite3.Error:
            return None

    #: Consecutive provider failures that open the account breaker, and the
    #: fixed finite backoff ladder (seconds) it walks per further failure.
    BREAKER_THRESHOLD = 3
    BREAKER_BACKOFF_S = (60.0, 120.0, 240.0, 480.0, 900.0)

    def record_screening_failure(self, owner: OwnerSnapshot) -> float:
        """One provider failure; open/advance the breaker, return ``retry_at``.

        Persisted so a restart cannot forget an open breaker, and FINITE so
        a broken route cannot park an account forever: the ladder stops at
        its last rung until a success resets it.
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            scope = self._account_scope(owner)
            row = conn.execute(
                "SELECT failures FROM screening_breaker WHERE account_scope = ?", (scope,)
            ).fetchone()
            failures = (int(row["failures"]) if row is not None else 0) + 1
            open_until = 0.0
            if failures >= self.BREAKER_THRESHOLD:
                rung = min(failures - self.BREAKER_THRESHOLD, len(self.BREAKER_BACKOFF_S) - 1)
                open_until = time.time() + self.BREAKER_BACKOFF_S[rung]
            conn.execute(
                "INSERT OR REPLACE INTO screening_breaker "
                "(account_scope, failures, open_until, updated_at) VALUES (?,?,?,?)",
                (scope, failures, open_until, time.time()),
            )
            return open_until

    def record_screening_success(self, owner: OwnerSnapshot) -> None:
        """A provider success clears the account breaker."""

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            conn.execute(
                "DELETE FROM screening_breaker WHERE account_scope = ?",
                (self._account_scope(owner),),
            )

    def screening_breaker_state(self, owner: OwnerSnapshot) -> dict[str, Any] | None:
        """Current breaker row (contentless: counts + timestamps only)."""

        try:
            with self._connect() as conn:
                self._init(conn)
                row = conn.execute(
                    "SELECT failures, open_until, updated_at FROM screening_breaker "
                    "WHERE account_scope = ?",
                    (self._account_scope(owner),),
                ).fetchone()
                return dict(row) if row is not None else None
        except sqlite3.Error:
            return None

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

        now = time.time()
        if expiry <= now:
            raise StoreRefused("expired_lease", "expiry must be in the future")
        with self._connect() as conn:
            self._init(conn)
            # BEGIN IMMEDIATE takes the write lock BEFORE the read: the
            # SELECT-then-write window QA's scheduler interleaved across two
            # connections (H2-Q6) cannot exist — the second contender waits
            # on the lock, then reads the winner's committed row.
            self._begin_write(conn)
            self._require_live(conn, owner)
            row = conn.execute(
                "SELECT * FROM claims WHERE owner_key = ? AND message_id = ?",
                (_owner_key(owner), message_id),
            ).fetchone()
            if row is not None and row["state"] == "held" and float(row["expires_at"]) > now:
                raise StoreRefused("claimed", "a live claim already holds this message")
            if row is not None and row["token"] == lease_token:
                # Reusing the predecessor's token would let a stale release
                # (same token) hit the successor; a replacement MUST present a
                # fresh token.
                raise StoreRefused(
                    "reused_claim_token", "an expired/released claim needs a NEW lease token"
                )
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
        """Extend a LIVE claim only: expiry can never be resurrected.

        Refused for a claim that already expired (renewal would turn a dead
        lease back into authority — H2-Q6), for a token that is not the live
        holder, and for a claim minted under a fenced generation.
        """

        now = time.time()
        if expiry <= now:
            raise StoreRefused("expired_lease", "expiry must be in the future")
        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            row = conn.execute(
                "SELECT * FROM claims WHERE owner_key = ? AND message_id = ?",
                (_owner_key(owner), message_id),
            ).fetchone()
            if row is None or row["token"] != lease_token or row["state"] != "held":
                raise StoreRefused("not_holder", "renew refused: token is not the live holder")
            if int(row["generation"]) != owner.enrollment_generation:
                raise StoreRefused("stale_generation", "claim was minted under a fenced generation")
            if float(row["expires_at"]) <= now:
                raise StoreRefused("expired_claim", "an expired claim cannot be renewed")
            # The UPDATE re-states the SAME fences: even if this transaction's
            # view were somehow raced by another writer (it cannot be, under
            # BEGIN IMMEDIATE), the row predicate — exact token, held state,
            # exact generation — is what gates the write (QA R2 H2-Q6).
            updated = conn.execute(
                "UPDATE claims SET expires_at = ? WHERE owner_key = ? AND message_id = ? "
                "AND token = ? AND state = 'held' AND generation = ?",
                (expiry, _owner_key(owner), message_id, lease_token, owner.enrollment_generation),
            )
            if updated.rowcount != 1:
                raise StoreRefused("not_holder", "renew refused: holder changed under the write")

    def release_claim(self, owner: OwnerSnapshot, message_id: str, lease_token: str) -> None:
        """Conditional release: only the LIVE holder's exact token may release.

        A fenced or already-released row is not the caller's to touch; a stale
        token never releases a successor (H2-Q6).
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            row = conn.execute(
                "SELECT token, state, generation FROM claims "
                "WHERE owner_key = ? AND message_id = ?",
                (_owner_key(owner), message_id),
            ).fetchone()
            if row is None:
                return
            if row["token"] != lease_token or row["state"] != "held":
                raise StoreRefused("not_holder", "release refused: token is not the live holder")
            if int(row["generation"]) != owner.enrollment_generation:
                raise StoreRefused("stale_generation", "claim was minted under a fenced generation")
            # Fenced conditional write (QA R2 H2-Q6): a successor's row cannot
            # match the old token+generation predicate, so a stale release can
            # never touch it even if evaluation were to interleave.
            updated = conn.execute(
                "UPDATE claims SET state = 'released' WHERE owner_key = ? AND message_id = ? "
                "AND token = ? AND state = 'held' AND generation = ?",
                (_owner_key(owner), message_id, lease_token, owner.enrollment_generation),
            )
            if updated.rowcount != 1:
                raise StoreRefused("not_holder", "release refused: holder changed under the write")

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
        """Persist refs WITHOUT touching the inbox checkpoint (single reads).

        The same tenant + recipient binding checks as :meth:`merge_page`: a
        single-message read is not a loophole past the owner guard (QA H2-Q1).
        """

        with self._connect() as conn:
            self._init(conn)
            self._begin_write(conn)
            self._require_live(conn, owner)
            for ref in refs:
                if ref.tenant_id != owner.tenant_id:
                    raise StoreRefused("wrong_tenant", "item tenant differs from owner")
                self._check_binding(owner, ref)
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

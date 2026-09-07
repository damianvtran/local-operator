"""Durable, revision-bound completion receipts shared by local frontends.

A connection is not a read. Only an explicit acknowledgement of a published
completion token advances the watermark. Tokens are conversation-bound and do
not depend on a process epoch, a wall clock, or transcript modification time.
SQLite serializes the short transactions across TUI, relay and server processes;
callers on an event loop run these operations in a worker thread.

TWO WATERMARKS, DELIBERATELY SEPARATE. ``receipts.acknowledged`` says a human
demonstrably READ a result; ``deliveries.delivered`` says somebody already
NOTIFIED them about it. They ride the same monotonic ``completions.sequence``
but must never be conflated: notifying is cheap and reversible, marking-read is
destructive, and ``docs/SESSION_SIDEBAR.md`` pins the rule that routing a
notification never marks anything read. A session can be delivered-and-unread
forever, which is correct — the sidebar's checkmark stays until it is opened.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir

ATTENTION_CAPABILITY = "completion-ack-v1"
ATTENTION_CUSTOM_TYPE = "completion_attention"

#: The delivery watermark. `delivered` is a highwater mark on the SAME
#: `completions.sequence` that `receipts.acknowledged` uses, which is what makes
#: the two comparable and the arbitration clock-free: sequence is assigned by
#: SQLite AUTOINCREMENT under BEGIN IMMEDIATE, so it is a total order across
#: every process on this machine. `delivered_at` and `backend` are DIAGNOSTICS
#: ONLY and no decision may read them \u2014 in particular, a wall clock must never
#: enter the claim, or two observers whose clocks disagree would both deliver.
_CREATE_DELIVERIES = (
    "CREATE TABLE deliveries ("
    "conversation TEXT PRIMARY KEY, delivered INTEGER NOT NULL, "
    "delivered_at REAL NOT NULL, backend TEXT NOT NULL)"
)

#: The no-flood rule, as one statement: everything already published counts as
#: already delivered. Runs once, inside the transaction that creates the table
#: (see `_connect`), so a machine upgrading into background-completion
#: notifications starts from "nothing outstanding" rather than announcing its
#: entire history.
_BASELINE_DELIVERIES = (
    "INSERT INTO deliveries(conversation,delivered,delivered_at,backend) "
    "SELECT conversation, MAX(sequence), ?, 'baseline' FROM completions "
    "GROUP BY conversation"
)


def conversation_identity(directory: Path) -> str:
    """Use the durable namespace, never the currently selected agent profile."""
    namespace = "agent" if directory.parent.name == "agents" else "session"
    return f"{namespace}/{directory.name}"


def bootstrap_transcript(transcript: Any, store: AttentionStore | None = None) -> None:
    """Explicit one-time import; never called by GET, SSE or focus observation.

    Old baselines were memory-only. Unknown historical work keeps that no-flood
    baseline, while a persisted seen stamp older than the actual final assistant
    entry preserves unread. Metadata file mtimes are deliberately irrelevant.
    """
    store = store or AttentionStore()
    identity = conversation_identity(transcript.directory)
    saved = transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
    started = transcript.latest_custom("attention_started")
    if (
        isinstance(started, dict)
        and started.get("conversation_id") == identity
        and (not isinstance(saved, dict) or saved.get("token") != started.get("token"))
    ):
        token = started["token"]
        store.publish(identity, token, f"completion-{token}", "interrupted")
        return
    if isinstance(saved, dict) and saved.get("conversation_id") == identity:
        if saved.get("eligible", True):
            store.publish(identity, saved["token"], saved["anchor"], saved["kind"])
        return
    if store.state(identity)["completion_token"]:
        return
    history = transcript.build_llm_history()
    if not history:
        return
    final = history[-1]
    if (
        getattr(final, "role", None) != "assistant"
        or not getattr(final, "text", "")
        or getattr(final, "tool_calls", None)
    ):
        return
    entry = next((row for row in reversed(transcript.entries()) if row.id == final.id), None)
    if entry is None:
        return
    seen = None
    try:
        raw = json.loads((store.path.parent / "mobile-seen.json").read_text())
        value = raw.get("sessions", {}).get(transcript.directory.name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seen = value
    except (OSError, ValueError, AttributeError):
        pass
    token = str(uuid.uuid5(uuid.NAMESPACE_URL, f"local-operator:{identity}:{final.id}"))
    store.publish(
        identity, token, final.id, "complete", baseline_seen=seen is None or seen >= entry.ts
    )


class AttentionStore:
    """One database per config root; no in-memory authority to become stale."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else config_dir() / "attention.db"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A new placeholder must be private before SQLite writes any contents.
        self.path.touch(mode=0o600, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            # Publish the complete schema in one transaction. Concurrent readers
            # can see the positively identified empty database or both tables,
            # never an intermediate schema with a missing receipt table.
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                if self._uninitialized(conn):
                    conn.execute(
                        "CREATE TABLE completions ("
                        "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "conversation TEXT NOT NULL, token TEXT NOT NULL UNIQUE, "
                        "anchor TEXT NOT NULL, kind TEXT NOT NULL)"
                    )
                    conn.execute(
                        "CREATE INDEX completion_conversation "
                        "ON completions(conversation, sequence)"
                    )
                    conn.execute(
                        "CREATE TABLE receipts ("
                        "conversation TEXT PRIMARY KEY, acknowledged INTEGER NOT NULL)"
                    )
                    # A store being created here has no completions yet, so
                    # there is no backlog to baseline against — an empty
                    # delivery watermark IS the correct starting point.
                    conn.execute(_CREATE_DELIVERIES)
                else:
                    # Missing tables/columns in an established database are
                    # corruption, not permission to rebuild an empty watermark.
                    #
                    # `deliveries` IS DELIBERATELY ABSENT FROM THIS PROBE, and
                    # adding it is the one "tidy-up" that would brick every
                    # existing machine: a database written by any release
                    # before the background-completion notifier legitimately
                    # lacks the table, so probing for it would read every one
                    # of them as corrupt. The probe stays byte-identical to
                    # what shipped, which is what keeps the meaning of an
                    # established database unchanged.
                    conn.execute(
                        "SELECT sequence,conversation,token,anchor,kind FROM completions LIMIT 0"
                    )
                    conn.execute("SELECT conversation,acknowledged FROM receipts LIMIT 0")
                    # Additive migration, and the baseline rides the SAME
                    # transaction as the CREATE so no concurrent reader can
                    # ever observe an unbaselined table. `unseen` is a LEVEL,
                    # not an edge: without this, the first observer to upgrade
                    # would claim every historical completion still unread and
                    # fire a banner for each one (measured on the maintainer's
                    # live store: 171 unseen conversations out of 332
                    # completions; the "8" an earlier draft of this comment
                    # cited was the test fixture's count, not a live
                    # measurement — review round 1, n1). Baselining at creation
                    # also means the
                    # eleventh observer to start INHERITS the baseline rather
                    # than re-deriving one of its own — the watermark is a
                    # property of the database, not of a process.
                    if not conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='deliveries'"
                    ).fetchone():
                        conn.execute(_CREATE_DELIVERIES)
                        conn.execute(_BASELINE_DELIVERIES, (time.time(),))
            return conn
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def _uninitialized(conn: sqlite3.Connection) -> bool:
        # A first publisher's private 0600 placeholder is a valid zero-schema
        # SQLite database. Check BOTH facts rather than swallowing OperationalError:
        # corrupt bytes, dropped tables and partial schemas must remain errors.
        return (
            conn.execute("PRAGMA schema_version").fetchone()[0] == 0
            and conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
        )

    @staticmethod
    def _state(conn: sqlite3.Connection, conversation: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM completions WHERE conversation=? ORDER BY sequence DESC LIMIT 1",
            (conversation,),
        ).fetchone()
        receipt = conn.execute(
            "SELECT acknowledged FROM receipts WHERE conversation=?", (conversation,)
        ).fetchone()
        acknowledged = receipt[0] if receipt else 0
        return {
            "conversation_id": conversation,
            "completion_token": row["token"] if row else None,
            "anchor_id": row["anchor"] if row else None,
            "kind": row["kind"] if row else None,
            "unseen": bool(row and row["sequence"] > acknowledged),
            "revision": [row["sequence"] if row else 0, acknowledged],
        }

    def state_many(self, conversations: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Read a whole frontend list on one connection and one consistent snapshot.

        Call this in a worker, then merge/render the returned map on the UI loop.
        Chunking bounds SQL parameters, not connection count; empty/new stores
        remain genuinely read-only and return explicit no-completion states.
        """
        identities = list(dict.fromkeys(conversations))
        states = {
            identity: {
                "conversation_id": identity,
                "completion_token": None,
                "anchor_id": None,
                "kind": None,
                "unseen": False,
                "revision": [0, 0],
            }
            for identity in identities
        }
        if not identities or not self.path.exists():
            return states
        with closing(
            sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=2.0)
        ) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            if self._uninitialized(conn):
                return states
            for offset in range(0, len(identities), 500):
                chunk = identities[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT c.*, COALESCE(r.acknowledged,0) AS acknowledged FROM "
                    "(SELECT conversation, MAX(sequence) AS sequence FROM completions "
                    f"WHERE conversation IN ({placeholders}) GROUP BY conversation) latest "
                    "JOIN completions c ON c.sequence=latest.sequence "
                    "LEFT JOIN receipts r ON r.conversation=c.conversation",
                    chunk,
                )
                for row in rows:
                    states[row["conversation"]] = {
                        "conversation_id": row["conversation"],
                        "completion_token": row["token"],
                        "anchor_id": row["anchor"],
                        "kind": row["kind"],
                        "unseen": row["sequence"] > row["acknowledged"],
                        "revision": [row["sequence"], row["acknowledged"]],
                    }
        return states

    def state(self, conversation: str) -> dict[str, Any]:
        return self.state_many([conversation])[conversation]

    def publish(
        self,
        conversation: str,
        token: str,
        anchor: str,
        kind: str,
        *,
        baseline_seen: bool | None = None,
    ) -> dict[str, Any]:
        """Import a durable outcome idempotently, including after owner restart."""
        if kind not in {"complete", "error", "interrupted"} or not anchor:
            raise ValueError("invalid completion")
        if str(uuid.UUID(token)) != token:
            raise ValueError("invalid completion token")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if (
                baseline_seen is not None
                and conn.execute(
                    "SELECT 1 FROM completions WHERE conversation=? LIMIT 1", (conversation,)
                ).fetchone()
            ):
                return self._state(conn, conversation)
            existing = conn.execute(
                "SELECT conversation, anchor, kind FROM completions WHERE token=?", (token,)
            ).fetchone()
            if existing and tuple(existing) != (conversation, anchor, kind):
                raise ValueError("completion token belongs to another outcome")
            conn.execute(
                "INSERT OR IGNORE INTO completions(conversation,token,anchor,kind) VALUES(?,?,?,?)",
                (conversation, token, anchor, kind),
            )
            if baseline_seen:
                sequence = conn.execute(
                    "SELECT sequence FROM completions WHERE token=?", (token,)
                ).fetchone()[0]
                conn.execute(
                    "INSERT OR IGNORE INTO receipts(conversation,acknowledged) VALUES(?,?)",
                    (conversation, sequence),
                )
            return self._state(conn, conversation)

    def revision(self) -> tuple[int, int]:
        """Cheap process-independent change detector for existing polling loops."""
        if not self.path.exists():
            return (0, 0)
        with closing(
            sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=2.0)
        ) as conn:
            conn.execute("BEGIN")
            if self._uninitialized(conn):
                return (0, 0)
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0), "
                "(SELECT COALESCE(SUM(acknowledged),0) FROM receipts) FROM completions"
            ).fetchone()
            return (row[0], row[1])

    def acknowledge(self, conversation: str, token: str) -> dict[str, Any]:
        """Advance only through the observed token, never through server 'now'."""
        if not isinstance(token, str) or len(token) != 36 or not self.path.exists():
            raise ValueError("unknown completion token")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sequence FROM completions WHERE conversation=? AND token=?",
                (conversation, token),
            ).fetchone()
            if row is None:
                raise ValueError("unknown completion token")
            conn.execute(
                "INSERT INTO receipts(conversation,acknowledged) VALUES(?,?) "
                "ON CONFLICT(conversation) DO UPDATE SET acknowledged="
                "MAX(receipts.acknowledged,excluded.acknowledged)",
                (conversation, row[0]),
            )
            return self._state(conn, conversation)

    def claim_delivery(self, conversation: str, token: str, backend: str) -> bool:
        """True iff THIS caller may notify about ``token``. Exactly one ever wins.

        The arbitration point for N observer processes watching one shared
        store. Every frontend polls attention for every session, so without a
        claim, eleven running sessions would each announce the same completion
        eleven times. `BEGIN IMMEDIATE` is the serialization: both racers take
        the write lock, and the loser reads the winner's already-advanced
        watermark from inside its own transaction. This is the identical
        argument `receipts` already rests on, which is why the table lives here
        rather than beside the store in a lock file of its own.

        NEVER ACKNOWLEDGES. Delivering a toast about a session says nothing
        about whether the human read it, so this touches `deliveries` only and
        the sidebar's unseen mark survives untouched.

        CLAIM-THEN-DELIVER, deliberately. A process dying between the claim and
        the spawn loses that one toast; delivering first and claiming after
        would instead give a crash-looping process a repeating banner. The lost
        signal is only the transient nudge \u2014 `unseen` stays true, the checkmark
        stays in the sidebar, and `lop sessions` still reports it \u2014 whereas a
        duplicate is visible and repeats. Backends that can report their own
        failure hand the claim back through :meth:`release_delivery`.

        THE ACCEPTED RISK, stated plainly so the next reader does not have to
        rediscover it: a claimant killed between the claim and the spawn leaves
        that completion delivered-but-unannounced FOREVER. There is no lease,
        no owner pid and no expiry \u2014 a re-claim returns False for good, and
        nothing sweeps it. That is bounded in CONSEQUENCE (one transient toast,
        for one completion, on a process that crashed) and unbounded in TIME,
        and it is accepted here because the alternative trade is worse: a lease
        needs a clock, and a clock in this predicate is what lets two observers
        with disagreeing time both deliver. The durable signal survives
        regardless, which is what makes the transient one safe to lose. A lease
        carrying an owner pid is the natural fix if this ever stops being
        acceptable; the schema has no room for one today (review round 1 m1,
        QA round 1 Q3 \u2014 both judged the trade correct).

        A store that does not exist yet holds no completion to claim, so this
        never creates one: an arbitration read must not be the thing that
        materialises the database.
        """
        if not self.path.exists():
            return False
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sequence FROM completions WHERE conversation=? AND token=?",
                (conversation, token),
            ).fetchone()
            if row is None:
                # An unknown token is never invented into the watermark: a
                # caller racing a store that was cleared behind it must not be
                # able to write a sequence no completion ever had.
                return False
            sequence = int(row[0])
            current = conn.execute(
                "SELECT delivered FROM deliveries WHERE conversation=?", (conversation,)
            ).fetchone()
            if current is not None and int(current[0]) >= sequence:
                return False
            conn.execute(
                "INSERT INTO deliveries(conversation,delivered,delivered_at,backend) "
                "VALUES(?,?,?,?) ON CONFLICT(conversation) DO UPDATE SET "
                "delivered=MAX(deliveries.delivered,excluded.delivered), "
                "delivered_at=excluded.delivered_at, backend=excluded.backend",
                (conversation, sequence, time.time(), backend),
            )
            # MAX on conflict mirrors `acknowledge` exactly, so reordered
            # writes converge upward instead of regressing the watermark.
            return True

    def release_delivery(self, conversation: str, token: str) -> bool:
        """Hand a claim back when the backend that took it delivered nothing.

        Without this, a backend failing AFTER the claim (notifications turned
        off between the two, `osascript` missing, a spawn refused) leaves the
        watermark asserting a toast that never reached anyone \u2014 a silent hole
        of exactly the kind this feature exists to close.

        Compare-and-swap, not `MAX`: rolling a watermark BACK is the one
        operation monotonic convergence cannot express. The update fires only
        while `delivered` is still the sequence this caller claimed, so an
        observer that has since claimed something newer is never clobbered.
        Rolling back to ``sequence - 1`` rather than deleting the row keeps
        every OLDER completion of this conversation delivered (their sequences
        are lower still), so a released claim re-opens exactly one event.
        """
        if not self.path.exists():
            return False
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sequence FROM completions WHERE conversation=? AND token=?",
                (conversation, token),
            ).fetchone()
            if row is None:
                return False
            sequence = int(row[0])
            cursor = conn.execute(
                "UPDATE deliveries SET delivered=?, delivered_at=?, backend='released' "
                "WHERE conversation=? AND delivered=?",
                (sequence - 1, time.time(), conversation, sequence),
            )
            return cursor.rowcount > 0

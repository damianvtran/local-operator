"""Durable, revision-bound completion receipts shared by local frontends.

A connection is not a read. Only an explicit acknowledgement of a published
completion token advances the watermark. Tokens are conversation-bound and do
not depend on a process epoch, a wall clock, or transcript modification time.
SQLite serializes the short transactions across TUI, relay and server processes;
callers on an event loop run these operations in a worker thread.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir

ATTENTION_CAPABILITY = "completion-ack-v1"
ATTENTION_CUSTOM_TYPE = "completion_attention"


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
        # Exclusive creation sets permissions before SQLite can write contents.
        self.path.touch(mode=0o600, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=2.0)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS completions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE,
                anchor TEXT NOT NULL,
                kind TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS completion_conversation
                ON completions(conversation, sequence);
            CREATE TABLE IF NOT EXISTS receipts (
                conversation TEXT PRIMARY KEY,
                acknowledged INTEGER NOT NULL
            );
            """)
        return conn

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

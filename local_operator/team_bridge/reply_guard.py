"""The client half of the conversation loop/cost guard (wake contract).

WHY THIS EXISTS. An agent member is now woken for every message in every
conversation it belongs to. Two agents whose policy is "answer every wake"
would answer each other for ever, each turn costing a model call. The SERVER
is the authoritative guard (it stops fanning agent-sent messages out to other
agents at a soft run of 4 and hard run of 8 consecutive agent messages), but an
older hub has no such rule, and the server cannot see what one endpoint spends.
This module is the client's own bound, enforced at the one place an agent
writes into a conversation (``bridge channel_post``):

* **No agent-to-agent limit** (urgent hotfix, operator direction 2026-11-03:
  "no limit on the number of agent replies"). The per-window agent budget and
  the agent-to-agent cool-down are gone, and a stale ``runs.cooldown_until``
  row left by an earlier build is never read, so an upgraded machine is
  unblocked at once. The SERVER's soft/hard run caps remain the loop bound.
* **Absolute cap** — ``ABSOLUTE_POSTS`` (120) posts per conversation per
  ``BUDGET_WINDOW_S``, counting EVERY post. It is a floor so no bug or stuck
  loop can turn one conversation into an unbounded spend, not a policy limit.

The refusal is STRUCTURED (``code`` + ``retry_after_s``) so an agent keeps its
draft and posts it once the window frees.

WHO WRITES WHAT. Two processes share this state: the connector (which knows
what woke the agent) records each delivered wake's TRIGGER — sender kind and
mention, never a body or a name — and the session's bridge tool reserves each
post. They meet in ONE small SQLite file beside the bridge store,
``<config_dir>/team_bridge/reply_guard.sqlite3``. It is deliberately NOT a
table in ``state.sqlite3``: that store's schema is migration-versioned and a
hotfix lane is mid-change there, and this state is disposable (losing it only
resets budgets, it never loses a message). Keys are the agent ENDPOINT and the
conversation id; nothing content-bearing is stored.

FAIL CLOSED. If the file cannot be opened or written, a post is refused with
``reply_guard_unavailable``: loop/cost safety is the requirement, and a
silently disabled guard is how a runaway would start.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

BUDGET_WINDOW_S = 600.0
#: Hard ceiling per conversation per window, over EVERY post. Raised from 30 to
#: 120 so live agent coordination is never throttled while a runaway keeps a floor.
ABSOLUTE_POSTS = 120

#: A trigger older than this no longer explains a post: the agent is acting on
#: its own (or on a human's DM), not answering an agent.
TRIGGER_TTL_S = 1800.0

CODE_ABSOLUTE = "reply_absolute_cap"
CODE_UNAVAILABLE = "reply_guard_unavailable"

_BUSY_TIMEOUT_S = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS triggers (
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    at REAL NOT NULL,
    sender_kind TEXT NOT NULL,
    mention INTEGER NOT NULL,
    directed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (endpoint_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    at REAL NOT NULL,
    agent_to_agent INTEGER NOT NULL,
    human_prompted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS posts_by_conversation
    ON posts (endpoint_id, conversation_id, at);
-- RETIRED: neither read nor written since the cool-down was removed; kept so a
-- guard file from an earlier build opens unchanged.
CREATE TABLE IF NOT EXISTS runs (
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    consecutive INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (endpoint_id, conversation_id)
);
"""


class ReplyGuardUnavailable(RuntimeError):
    """The guard's state could not be read or written (fail closed)."""


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add ``human_prompted`` to a guard file a previous build wrote.

    The guard's state is DISPOSABLE (losing it only resets budgets), so an
    additive column is the whole migration — no version machinery. Same rule as
    ``store._ensure_card_posted_column`` on the durable store. Pre-existing rows
    default to 0, i.e. they keep spending the agent budget: the conservative
    direction for a cost guard.
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(posts)")}
    if "human_prompted" not in columns:
        conn.execute("ALTER TABLE posts ADD COLUMN human_prompted INTEGER NOT NULL DEFAULT 0")
    trigger_columns = {row[1] for row in conn.execute("PRAGMA table_info(triggers)")}
    if "directed" not in trigger_columns:
        # A DIAGNOSTIC record, not a read path: the brake's decision is taken in
        # ``record_trigger`` from the value its caller passed, and no query reads
        # this column back (round 6, m1 — the earlier comment described a read
        # that does not exist). An older row therefore defaults to 0 and nothing
        # depends on the value.
        conn.execute("ALTER TABLE triggers ADD COLUMN directed INTEGER NOT NULL DEFAULT 0")


@dataclass(frozen=True)
class Reservation:
    """A granted post slot. ``release`` it if the post did not land."""

    post_id: int
    agent_to_agent: bool


@dataclass(frozen=True)
class Refusal:
    code: str
    retry_after_s: int
    message: str

    def as_result(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "error": self.message,
            "retry_after_s": self.retry_after_s,
            # The cap frees as the window slides; the post is worth keeping.
            "next": "not sent; keep your draft and retry after retry_after_s",
        }


def guard_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "reply_guard.sqlite3"


@contextmanager
def _connect(config_dir: Path | str) -> Iterator[sqlite3.Connection]:
    path = guard_path(config_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
    except (OSError, sqlite3.Error) as failure:
        raise ReplyGuardUnavailable("reply guard state unavailable") from failure
    try:
        conn.executescript(_SCHEMA)
        _ensure_columns(conn)
        # IMMEDIATE: check-then-reserve is one decision. Two sessions of the
        # same endpoint posting at once must not both see the last free slot.
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    except sqlite3.Error as failure:
        raise ReplyGuardUnavailable("reply guard state unavailable") from failure
    finally:
        conn.close()


def record_trigger(
    config_dir: Path | str,
    *,
    endpoint_id: str,
    conversation_id: str,
    sender_kind: str,
    mention: bool,
    directed: bool | None = None,
    now: float | None = None,
) -> None:
    """Record what last woke ``endpoint_id`` in ``conversation_id``.

    Called by the connector for every admitted message it hands to the agent
    (coalesced or not, oldest → newest, so the newest wins). ``directed`` (the
    connector's strict addressedness predicate; ``None`` falls back to
    ``mention``) is written as a diagnostic record only: since the cool-down was
    removed nothing reads it and no run state is written here. Never raises: a
    trigger that cannot be recorded leaves the previous one, and the post side
    still fails closed on its
    own.
    """

    now = time.time() if now is None else now
    kind = sender_kind if sender_kind in ("human", "agent") else "unknown"
    try:
        with _connect(config_dir) as conn:
            addressed = mention if directed is None else directed
            conn.execute(
                "INSERT INTO triggers "
                "(endpoint_id, conversation_id, at, sender_kind, mention, directed) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(endpoint_id, conversation_id) DO UPDATE SET "
                "at = excluded.at, sender_kind = excluded.sender_kind, "
                "mention = excluded.mention, directed = excluded.directed",
                (
                    endpoint_id,
                    conversation_id,
                    now,
                    kind,
                    1 if mention else 0,
                    1 if addressed else 0,
                ),
            )
    except ReplyGuardUnavailable:
        pass


def reserve_post(
    config_dir: Path | str,
    *,
    endpoint_id: str,
    conversation_id: str,
    now: float | None = None,
) -> Reservation | Refusal:
    """Grant one post slot, or say why not. Raises ``ReplyGuardUnavailable``.

    Only the ABSOLUTE cap can refuse; it counts every post. The trigger still
    labels the slot (``agent_to_agent``, ``human_prompted``) for diagnostics.
    ``runs.cooldown_until`` is deliberately NOT read: a cool-down an earlier
    build armed must not keep blocking an upgraded machine.

    The slot is recorded BEFORE the post goes out (so a concurrent post sees
    it) and must be given back through :func:`release` when the hub did not
    accept the post.
    """

    now = time.time() if now is None else now
    with _connect(config_dir) as conn:
        window_start = now - BUDGET_WINDOW_S
        recent = [
            float(row[0])
            for row in conn.execute(
                "SELECT at FROM posts "
                "WHERE endpoint_id = ? AND conversation_id = ? AND at > ? ORDER BY at",
                (endpoint_id, conversation_id, window_start),
            )
        ]
        if len(recent) >= ABSOLUTE_POSTS:
            freeing = recent[len(recent) - ABSOLUTE_POSTS]
            return Refusal(
                code=CODE_ABSOLUTE,
                retry_after_s=max(1, int(freeing + BUDGET_WINDOW_S - now + 0.999)),
                message=(
                    f"absolute cap: {ABSOLUTE_POSTS} posts in this conversation within "
                    f"{int(BUDGET_WINDOW_S // 60)} minutes, whatever woke you"
                ),
            )
        trigger = conn.execute(
            "SELECT at, sender_kind FROM triggers WHERE endpoint_id = ? AND conversation_id = ?",
            (endpoint_id, conversation_id),
        ).fetchone()
        human_prompted = False
        agent_to_agent = False
        if trigger is not None and now - float(trigger[0]) <= TRIGGER_TTL_S:
            human_prompted = trigger[1] == "human"
            agent_to_agent = trigger[1] == "agent"
        cursor = conn.execute(
            "INSERT INTO posts (endpoint_id, conversation_id, at, agent_to_agent, human_prompted) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                endpoint_id,
                conversation_id,
                now,
                1 if agent_to_agent else 0,
                1 if human_prompted else 0,
            ),
        )
        # Bound the table: rows older than 2x the window carry no decision.
        conn.execute("DELETE FROM posts WHERE at <= ?", (now - 2 * BUDGET_WINDOW_S,))
        return Reservation(post_id=int(cursor.lastrowid or 0), agent_to_agent=agent_to_agent)


def release(
    config_dir: Path | str, reservation: Reservation, *, endpoint_id: str, conversation_id: str
) -> None:
    """Give back a slot whose post the hub did not accept.

    The slot is returned to the absolute cap's window.
    """

    try:
        with _connect(config_dir) as conn:
            conn.execute(
                "DELETE FROM posts WHERE id = ? AND endpoint_id = ? AND conversation_id = ?",
                (reservation.post_id, endpoint_id, conversation_id),
            )
    except ReplyGuardUnavailable:
        pass


__all__ = [
    "ABSOLUTE_POSTS",
    "BUDGET_WINDOW_S",
    "CODE_ABSOLUTE",
    "CODE_UNAVAILABLE",
    "Refusal",
    "ReplyGuardUnavailable",
    "Reservation",
    "guard_path",
    "record_trigger",
    "release",
    "reserve_post",
]

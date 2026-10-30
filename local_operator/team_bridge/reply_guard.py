"""The client half of the conversation loop/cost guard (wake contract).

WHY THIS EXISTS. An agent member is now woken for every message in every
conversation it belongs to. Two agents whose policy is "answer every wake"
would answer each other for ever, each turn costing a model call. The SERVER
is the authoritative guard (it stops fanning agent-sent messages out to other
agents at a soft run of 4 and hard run of 8 consecutive agent messages), but an
older hub has no such rule, and the server cannot see what one endpoint spends.
This module is the client's own bound, enforced at the one place an agent
writes into a conversation (``bridge channel_post``):

* **Reply budget** — at most ``BUDGET_POSTS`` agent posts per conversation per
  ``BUDGET_WINDOW_S`` (6 / 10 min).
* **Agent-to-agent cool-down** — after ``COOLDOWN_AFTER`` (3) consecutive
  agent-to-agent turns (a post whose triggering wake was AGENT-sent), further
  posts to that conversation are refused for ``COOLDOWN_S`` (10 min).
  A HUMAN-sent mention resets the consecutive counter and lifts a cool-down:
  a person asking the agent directly is exactly the case the guard must not
  block. A human post without a mention does not (see the contract).

Both refusals are STRUCTURED (``code`` + ``retry_after_s``) so an agent can
end its turn with ``no_reply`` instead of retrying.

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

#: Max agent posts per conversation per window (wake contract).
BUDGET_POSTS = 6
BUDGET_WINDOW_S = 600.0
#: Consecutive agent-to-agent turns before the cool-down starts.
COOLDOWN_AFTER = 3
COOLDOWN_S = 600.0

#: A trigger older than this no longer explains a post: the agent is acting on
#: its own (or on a human's DM), not answering an agent.
TRIGGER_TTL_S = 1800.0

CODE_BUDGET = "reply_budget_exhausted"
CODE_COOLDOWN = "reply_cooldown"
CODE_UNAVAILABLE = "reply_guard_unavailable"

_BUSY_TIMEOUT_S = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS triggers (
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    at REAL NOT NULL,
    sender_kind TEXT NOT NULL,
    mention INTEGER NOT NULL,
    PRIMARY KEY (endpoint_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    at REAL NOT NULL,
    agent_to_agent INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS posts_by_conversation
    ON posts (endpoint_id, conversation_id, at);
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
            # The one action that ends the turn cheaply and quietly.
            "next": "end this turn with no_reply; do not retry the post",
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
    now: float | None = None,
) -> None:
    """Record what last woke ``endpoint_id`` in ``conversation_id``.

    Called by the connector for every admitted message it hands to the agent
    (coalesced or not, oldest → newest, so the newest wins). A HUMAN mention
    resets the agent-to-agent run and lifts a cool-down. Never raises: a
    trigger that cannot be recorded leaves the previous one, and the post side
    still fails closed on its own.
    """

    now = time.time() if now is None else now
    kind = sender_kind if sender_kind in ("human", "agent") else "unknown"
    try:
        with _connect(config_dir) as conn:
            conn.execute(
                "INSERT INTO triggers (endpoint_id, conversation_id, at, sender_kind, mention) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(endpoint_id, conversation_id) DO UPDATE SET "
                "at = excluded.at, sender_kind = excluded.sender_kind, mention = excluded.mention",
                (endpoint_id, conversation_id, now, kind, 1 if mention else 0),
            )
            if kind == "human" and mention:
                conn.execute(
                    "INSERT INTO runs (endpoint_id, conversation_id, consecutive, cooldown_until) "
                    "VALUES (?, ?, 0, 0) ON CONFLICT(endpoint_id, conversation_id) DO UPDATE SET "
                    "consecutive = 0, cooldown_until = 0",
                    (endpoint_id, conversation_id),
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

    The slot is recorded BEFORE the post goes out (so a concurrent post sees
    it) and must be :func:`release`\\ d if the hub did not accept the post.
    """

    now = time.time() if now is None else now
    with _connect(config_dir) as conn:
        run = conn.execute(
            "SELECT consecutive, cooldown_until FROM runs "
            "WHERE endpoint_id = ? AND conversation_id = ?",
            (endpoint_id, conversation_id),
        ).fetchone()
        consecutive, cooldown_until = (int(run[0]), float(run[1])) if run else (0, 0.0)
        if cooldown_until > now:
            return Refusal(
                code=CODE_COOLDOWN,
                retry_after_s=max(1, int(cooldown_until - now + 0.999)),
                message=(
                    f"cool-down: {COOLDOWN_AFTER} consecutive replies to agents in this "
                    "conversation; posting resumes later or when a person @-mentions you"
                ),
            )
        window_start = now - BUDGET_WINDOW_S
        recent = [
            float(row[0])
            for row in conn.execute(
                "SELECT at FROM posts WHERE endpoint_id = ? AND conversation_id = ? AND at > ? "
                "ORDER BY at",
                (endpoint_id, conversation_id, window_start),
            )
        ]
        if len(recent) >= BUDGET_POSTS:
            oldest = recent[0]
            return Refusal(
                code=CODE_BUDGET,
                retry_after_s=max(1, int(oldest + BUDGET_WINDOW_S - now + 0.999)),
                message=(
                    f"reply budget: {BUDGET_POSTS} posts in this conversation within "
                    f"{int(BUDGET_WINDOW_S // 60)} minutes"
                ),
            )
        trigger = conn.execute(
            "SELECT at, sender_kind FROM triggers WHERE endpoint_id = ? AND conversation_id = ?",
            (endpoint_id, conversation_id),
        ).fetchone()
        agent_to_agent = bool(
            trigger is not None
            and trigger[1] == "agent"
            and now - float(trigger[0]) <= TRIGGER_TTL_S
        )
        cursor = conn.execute(
            "INSERT INTO posts (endpoint_id, conversation_id, at, agent_to_agent) "
            "VALUES (?, ?, ?, ?)",
            (endpoint_id, conversation_id, now, 1 if agent_to_agent else 0),
        )
        # A human-triggered (or unprompted) post breaks the agent-to-agent run.
        consecutive = consecutive + 1 if agent_to_agent else 0
        cooldown = 0.0
        if consecutive >= COOLDOWN_AFTER:
            # THIS post is allowed (it is the third); the next is refused.
            cooldown, consecutive = now + COOLDOWN_S, 0
        conn.execute(
            "INSERT INTO runs (endpoint_id, conversation_id, consecutive, cooldown_until) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(endpoint_id, conversation_id) DO UPDATE SET "
            "consecutive = excluded.consecutive, cooldown_until = excluded.cooldown_until",
            (endpoint_id, conversation_id, consecutive, cooldown),
        )
        # Bound the table: rows older than the window carry no decision.
        conn.execute("DELETE FROM posts WHERE at <= ?", (now - 2 * BUDGET_WINDOW_S,))
        return Reservation(post_id=int(cursor.lastrowid or 0), agent_to_agent=agent_to_agent)


def release(
    config_dir: Path | str, reservation: Reservation, *, endpoint_id: str, conversation_id: str
) -> None:
    """Give back a slot whose post the hub did not accept.

    The budget slot is returned. The run counter is NOT rolled back: a failed
    agent-to-agent attempt still spent a model turn, and erring towards an
    earlier cool-down is the safe direction for a cost guard.
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
    "BUDGET_POSTS",
    "BUDGET_WINDOW_S",
    "CODE_BUDGET",
    "CODE_COOLDOWN",
    "CODE_UNAVAILABLE",
    "COOLDOWN_AFTER",
    "COOLDOWN_S",
    "Refusal",
    "ReplyGuardUnavailable",
    "Reservation",
    "guard_path",
    "record_trigger",
    "release",
    "reserve_post",
]

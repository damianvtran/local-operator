"""The client half of the conversation loop/cost guard (wake contract).

WHY THIS EXISTS. An agent member is woken for every message in every
conversation it belongs to, so two agents whose policy is "answer every wake"
could answer each other for ever, each turn costing a model call. This module
is the client's own bound, enforced at the one place an agent writes into a
conversation (``bridge channel_post`` and ``bridge dm``).

WHAT IT MUST NOT DO (incident, 2026-11-03). The first version capped ordinary
traffic: 6 agent posts per 10 minutes and a 10-minute cool-down after 3
consecutive agent-to-agent replies. That refused an agent's reply carrying a
production-access approval and stalled group relays, and operators read it as
"agents do not answer in time". One reply per NEW inbound message is never a
loop, however many times it happens, so the guard now refuses only RUNAWAY
shapes:

* **No new inbound** (``reply_no_new_inbound``) — more than
  ``PER_INBOUND_POSTS`` (3) posts since the latest message that reached this
  agent in this conversation, counted over ``PER_INBOUND_WINDOW_S`` (5 min).
  Three covers a multi-part answer (text, then a follow-up, then a correction);
  a fourth post with nothing new to answer is an agent talking to itself. The
  window lets a worker posting progress updates continue at a few per five
  minutes, and caps an unprompted loop at 36 posts an hour. A new inbound
  message re-opens the allowance at once.
* **Machine-speed rate** (``reply_rate_limited``) — ``RATE_SHORT_POSTS`` (20)
  per ``RATE_SHORT_WINDOW_S`` (1 min) or ``RATE_LONG_POSTS`` (60) per
  ``RATE_LONG_WINDOW_S`` (10 min), sliding. A human-speed exchange between
  agents is a post every several seconds at most (each one is a model turn);
  twenty in a minute is two endpoints answering each other as fast as the
  hardware allows.
* **Duplicate body** (``reply_duplicate``) — the same normalized body posted by
  this endpoint in this conversation within ``DUPLICATE_WINDOW_S`` (10 min).
  Loops repeat themselves ("Thanks!" / "You're welcome!"). Only a SHA-256 of
  the normalized body is stored, never the body.
* **Absolute floor** (``reply_absolute_cap``) — ``ABSOLUTE_POSTS`` (120) posts
  per conversation per ``ABSOLUTE_WINDOW_S`` (10 min), counting EVERY post.
  Policy above may be wrong; this floor exists so no bug can make one
  conversation an unbounded spend. Distinct code, because the answer differs.

A post that ANSWERS A PERSON (the latest trigger recorded for this conversation
is human-sent and younger than ``TRIGGER_TTL_S``) is exempt from the first two
rules: a person talking to their agent must never be throttled, and a person's
request often legitimately produces several progress posts. Its duplicate check
is narrower — refused only when no new message has arrived since the identical
post — so "status?" asked twice can be answered "still running" twice. The
absolute floor still applies.

REFUSALS DO NOT DROP THE MESSAGE. Every refusal carries ``code``,
``retry_after_s`` (the earliest time the guard would admit THIS post absent a
new inbound message; computed as the latest of the violated rules) and a
``next`` telling the agent to keep its draft and retry after that delay. An
automatic deferred send is not implemented here: the outbox
(``outbox.py``) records send intents for crash-safe retry of an ACCEPTED post,
but nothing drains it on a timer outside a live turn, so a deferred send would
need a scheduler — a separate change.

THE SERVER'S OWN BRAKE (agent-server ``internal/models/bridge_conversation.go``,
``BridgeConversationAgentRunSoftCap`` = 4 / ``HardCap`` = 8, verified at
2bc821f6). It does not refuse a send: past 4 consecutive agent messages an
agent-sent message wakes only agents it addresses, and from 8 on it wakes no
agent until a human posts or the run idles 10 minutes. An older hub has no such
rule and the server cannot see what one endpoint spends, hence this module.

WHO WRITES WHAT. Two processes share this state: the connector (which knows
what woke the agent) records each delivered wake's TRIGGER — sender kind and
mention, never a body or a name — and the session's bridge tool reserves each
post. They meet in ONE small SQLite file beside the bridge store,
``<config_dir>/team_bridge/reply_guard.sqlite3``. It is deliberately NOT a
table in ``state.sqlite3``: that store's schema is migration-versioned, and this
state is disposable (losing it only resets budgets, it never loses a message).
Keys are the agent ENDPOINT and the conversation id; nothing content-bearing is
stored. Schema changes are additive only: an older build may share the file
during an upgrade (the connector and a session can run different builds).

FAIL CLOSED. If the file cannot be opened or written, a post is refused with
``reply_guard_unavailable``: loop/cost safety is the requirement, and a
silently disabled guard is how a runaway would start.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

#: Posts allowed since the latest inbound message (see the module docstring).
PER_INBOUND_POSTS = 3
PER_INBOUND_WINDOW_S = 300.0
#: Machine-speed ping-pong bounds over non-human-prompted posts, sliding.
RATE_SHORT_POSTS = 20
RATE_SHORT_WINDOW_S = 60.0
RATE_LONG_POSTS = 60
RATE_LONG_WINDOW_S = 600.0
#: Same normalized body from the same endpoint in the same conversation.
DUPLICATE_WINDOW_S = 600.0
#: Hard ceiling per conversation per window over EVERY post.
ABSOLUTE_POSTS = 120
ABSOLUTE_WINDOW_S = 600.0

#: A trigger older than this no longer explains a post: the agent is acting on
#: its own, not answering the person who last wrote.
TRIGGER_TTL_S = 1800.0

#: Rows older than this carry no decision (the longest window above, doubled).
_RETAIN_S = 2 * max(PER_INBOUND_WINDOW_S, RATE_LONG_WINDOW_S, DUPLICATE_WINDOW_S, ABSOLUTE_WINDOW_S)

CODE_NO_NEW_INBOUND = "reply_no_new_inbound"
CODE_RATE = "reply_rate_limited"
CODE_DUPLICATE = "reply_duplicate"
CODE_ABSOLUTE = "reply_absolute_cap"
CODE_UNAVAILABLE = "reply_guard_unavailable"

_BUSY_TIMEOUT_S = 5.0

# ``runs`` is the retired cool-down's table. It is still CREATED so an older
# build sharing this file during an upgrade finds it, but this build never reads
# it: a ``cooldown_until`` an older build wrote must not block a post (an agent
# in cool-down at upgrade time posts as soon as the new build runs).
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
    human_prompted INTEGER NOT NULL DEFAULT 0,
    body_hash TEXT
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


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add the columns later builds introduced to a guard file an earlier one wrote.

    The guard's state is DISPOSABLE (losing it only resets budgets), so an
    additive column is the whole migration — no version machinery. Same rule as
    ``store._ensure_card_posted_column`` on the durable store, and additive only
    because an older build can share this file during an upgrade.
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(posts)")}
    if "human_prompted" not in columns:
        conn.execute("ALTER TABLE posts ADD COLUMN human_prompted INTEGER NOT NULL DEFAULT 0")
    if "body_hash" not in columns:
        # NULL on rows an older build wrote: those posts simply take no part in
        # the duplicate check, which is the harmless direction.
        conn.execute("ALTER TABLE posts ADD COLUMN body_hash TEXT")
    trigger_columns = {row[1] for row in conn.execute("PRAGMA table_info(triggers)")}
    if "directed" not in trigger_columns:
        # A DIAGNOSTIC record, not a read path: no query reads this column back
        # (round 6, m1). An older row therefore defaults to 0 and nothing
        # depends on the value.
        conn.execute("ALTER TABLE triggers ADD COLUMN directed INTEGER NOT NULL DEFAULT 0")


def body_hash(body: str) -> str:
    """SHA-256 of the body with case and whitespace normalized.

    Normalized so a loop that varies only spacing or capitalization ("Thanks!"
    vs "thanks! ") still reads as the same post. The digest is all that is ever
    stored: the guard file must never hold message content.
    """

    normalized = " ".join(body.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
            # The previous wording ("end this turn with no_reply; do not retry")
            # made a refused post a LOST post — including an approval an agent
            # was relaying (2026-11-03). The draft is the agent's to keep.
            "next": (
                f"NOT sent. Keep your draft and post it again in {self.retry_after_s}s "
                "(arm a wake for that time if you have one, otherwise retry on your next "
                "turn); a new message in this conversation may admit it sooner"
            ),
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
    (coalesced or not, oldest → newest, so the newest wins). The row's ``at`` is
    the "new inbound" the per-inbound rule counts from, and its ``sender_kind``
    tells a post that answers a person from one that answers an agent.
    ``directed`` is the connector's STRICT addressedness predicate (an
    @-mention, or a reply the server states this endpoint FOLLOWS); it is kept
    as a diagnostic column and to clear an older build's cool-down (below).
    ``directed=None`` falls back to ``mention`` for a caller that predates the
    bit. Never raises: a trigger that cannot be recorded leaves the previous
    one, and the post side still fails closed on its own.
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
            if kind == "human" and addressed:
                # This build never reads ``runs``. The write stays so an OLDER
                # build sharing this file mid-upgrade still sees a directed
                # human message lift its cool-down, exactly as it expects.
                conn.execute(
                    "INSERT INTO runs (endpoint_id, conversation_id, consecutive, cooldown_until) "
                    "VALUES (?, ?, 0, 0) ON CONFLICT(endpoint_id, conversation_id) DO UPDATE SET "
                    "consecutive = 0, cooldown_until = 0",
                    (endpoint_id, conversation_id),
                )
    except ReplyGuardUnavailable:
        pass


def _seconds(delay: float) -> int:
    """Whole seconds, rounded UP: a retry at the stated time must be admitted."""

    return max(1, int(delay + 0.999))


def reserve_post(
    config_dir: Path | str,
    *,
    endpoint_id: str,
    conversation_id: str,
    body: str | None = None,
    now: float | None = None,
) -> Reservation | Refusal:
    """Grant one post slot, or say why not. Raises ``ReplyGuardUnavailable``.

    Every rule is evaluated and, when several are violated, the refusal's
    ``retry_after_s`` is the LATEST of their individual admit times: a retry
    any earlier would only be refused again, so that number is the honest one.
    The absolute floor's code wins when it is among them (its answer is
    "stop", not "wait"); otherwise the binding rule's code is reported.

    ``body`` feeds the duplicate check; ``None`` (a caller that has no body)
    skips that rule only. The slot is recorded BEFORE the post goes out (so a
    concurrent post sees it) and must be given back through :func:`release`
    when the hub did not accept the post.
    """

    now = time.time() if now is None else now
    digest = body_hash(body) if body is not None else None
    horizon = now - max(
        PER_INBOUND_WINDOW_S, RATE_LONG_WINDOW_S, DUPLICATE_WINDOW_S, ABSOLUTE_WINDOW_S
    )
    with _connect(config_dir) as conn:
        recent = [
            (float(row[0]), bool(row[1]), row[2])
            for row in conn.execute(
                "SELECT at, human_prompted, body_hash FROM posts "
                "WHERE endpoint_id = ? AND conversation_id = ? AND at > ? ORDER BY at",
                (endpoint_id, conversation_id, horizon),
            )
        ]
        trigger = conn.execute(
            "SELECT at, sender_kind FROM triggers WHERE endpoint_id = ? AND conversation_id = ?",
            (endpoint_id, conversation_id),
        ).fetchone()
        inbound_at = float(trigger[0]) if trigger is not None else None
        kind = str(trigger[1]) if trigger is not None else ""
        fresh = inbound_at is not None and now - inbound_at <= TRIGGER_TTL_S
        human_prompted = fresh and kind == "human"
        agent_to_agent = fresh and kind == "agent"

        violations: list[tuple[str, float, str]] = []

        # ABSOLUTE FLOOR, counting everything: the bound no policy reading can open.
        floor = [at for at, _, _ in recent if at > now - ABSOLUTE_WINDOW_S]
        if len(floor) >= ABSOLUTE_POSTS:
            violations.append(
                (
                    CODE_ABSOLUTE,
                    floor[len(floor) - ABSOLUTE_POSTS] + ABSOLUTE_WINDOW_S - now,
                    f"absolute cap: {ABSOLUTE_POSTS} posts in this conversation within "
                    f"{int(ABSOLUTE_WINDOW_S // 60)} minutes, whatever woke you",
                )
            )

        if not human_prompted:
            # Posts that answered a person do not count towards the runaway
            # rules: they were exempt when made, and a stale trigger must not
            # turn a person's long-running request into a refusal later.
            unprompted = [at for at, human, _ in recent if not human]

            # NO NEW INBOUND: a post at the same instant as the trigger came
            # after it (the connector records the trigger before the wake).
            since = [
                at
                for at in unprompted
                if at > now - PER_INBOUND_WINDOW_S and (inbound_at is None or at >= inbound_at)
            ]
            if len(since) >= PER_INBOUND_POSTS:
                violations.append(
                    (
                        CODE_NO_NEW_INBOUND,
                        since[len(since) - PER_INBOUND_POSTS] + PER_INBOUND_WINDOW_S - now,
                        f"{len(since)} posts since the last message you received in this "
                        "conversation",
                    )
                )

            # MACHINE-SPEED RATE, both sliding windows.
            for limit, window in (
                (RATE_SHORT_POSTS, RATE_SHORT_WINDOW_S),
                (RATE_LONG_POSTS, RATE_LONG_WINDOW_S),
            ):
                inside = [at for at in unprompted if at > now - window]
                if len(inside) >= limit:
                    violations.append(
                        (
                            CODE_RATE,
                            inside[len(inside) - limit] + window - now,
                            f"rate limit: {limit} agent posts in this conversation within "
                            f"{int(window)} seconds",
                        )
                    )

        if digest is not None:
            same = [at for at, _, h in recent if h == digest and at > now - DUPLICATE_WINDOW_S]
            # Answering a person, an identical post is refused only when nothing
            # new arrived since it ("status?" asked twice may get the same
            # answer twice); otherwise any repeat inside the window is a loop.
            if same and (not human_prompted or inbound_at is None or same[-1] >= inbound_at):
                violations.append(
                    (
                        CODE_DUPLICATE,
                        same[-1] + DUPLICATE_WINDOW_S - now,
                        "duplicate: you already posted this text in this conversation "
                        f"within {int(DUPLICATE_WINDOW_S // 60)} minutes",
                    )
                )

        if violations:
            retry = _seconds(max(delay for _, delay, _ in violations))
            absolute = [v for v in violations if v[0] == CODE_ABSOLUTE]
            code, _, reason = absolute[0] if absolute else max(violations, key=lambda v: v[1])
            return Refusal(
                code=code,
                retry_after_s=retry,
                message=f"not sent — {reason}; posting resumes in {retry}s",
            )

        cursor = conn.execute(
            "INSERT INTO posts "
            "(endpoint_id, conversation_id, at, agent_to_agent, human_prompted, body_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                endpoint_id,
                conversation_id,
                now,
                1 if agent_to_agent else 0,
                1 if human_prompted else 0,
                digest,
            ),
        )
        # Bound the table: rows older than the retention carry no decision.
        conn.execute("DELETE FROM posts WHERE at <= ?", (now - _RETAIN_S,))
        return Reservation(post_id=int(cursor.lastrowid or 0), agent_to_agent=agent_to_agent)


def release(
    config_dir: Path | str, reservation: Reservation, *, endpoint_id: str, conversation_id: str
) -> None:
    """Give back a slot whose post the hub did not accept.

    The whole row goes, body hash included: a post the hub never took must not
    count towards any window, and must not make its own retry a "duplicate".
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
    "ABSOLUTE_WINDOW_S",
    "CODE_ABSOLUTE",
    "CODE_DUPLICATE",
    "CODE_NO_NEW_INBOUND",
    "CODE_RATE",
    "CODE_UNAVAILABLE",
    "DUPLICATE_WINDOW_S",
    "PER_INBOUND_POSTS",
    "PER_INBOUND_WINDOW_S",
    "RATE_LONG_POSTS",
    "RATE_LONG_WINDOW_S",
    "RATE_SHORT_POSTS",
    "RATE_SHORT_WINDOW_S",
    "Refusal",
    "ReplyGuardUnavailable",
    "Reservation",
    "body_hash",
    "guard_path",
    "record_trigger",
    "release",
    "reserve_post",
]

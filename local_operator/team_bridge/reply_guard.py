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
* **Hourly agent-to-agent ceiling** (``reply_agent_hourly``) —
  ``AGENT_HOURLY_POSTS`` (120) posts answering an AGENT per
  ``AGENT_HOURLY_WINDOW_S`` (60 min), sliding, read from the stored
  ``agent_to_agent`` column. In a two-agent loop every post answers a new
  inbound message, so the per-inbound rule never binds, and at model speed
  (10-30 s a turn) neither rate rule does: without this ceiling such a loop
  runs for ever on a hub with no wake brake (review round 1, S-1; the dev hub
  at agent-server 96822331 has none). It is the client's STANDALONE bound. A
  new inbound message does not reset it, and a post answering a person never
  counts. 120/h is about two posts a minute sustained, far above any real
  exchange; the incident's relays and approvals are a handful of posts. It is
  deliberately not a consecutive-run cool-down: that shape is what the
  incident removed.
* **Duplicate body** (``reply_duplicate``) — the same normalized body posted by
  this endpoint in this conversation within ``DUPLICATE_WINDOW_S`` (10 min).
  Loops repeat themselves ("Thanks!" / "You're welcome!"). With no new message
  since the identical post, the 2nd copy is refused. Once a new message has
  arrived, the same short answer to a new request is legitimate ("Approved."
  to two distinct requests, review round 1, M1), so for an agent-prompted post
  only the 3rd copy inside the window is refused. The stored value is an
  HMAC-SHA256 under a random per-install key (``reply_guard.key``, 0600, beside
  the database), never the body: a plain hash of "approved" or "yes" would be
  recoverable with a small dictionary (review round 1, S-3). Equality within
  this install is all the rule needs; losing the key only resets the rule.
* **Absolute floor** (``reply_absolute_cap``) — ``ABSOLUTE_POSTS`` (120) posts
  per conversation per ``ABSOLUTE_WINDOW_S`` (10 min), counting EVERY post.
  Policy above may be wrong; this floor exists so no bug can make one
  conversation an unbounded spend. Distinct code, because the answer differs.

A post that ANSWERS A PERSON (the latest trigger recorded for this conversation
is human-sent and younger than ``HUMAN_TRIGGER_TTL_S``, 10 min) is exempt from
the per-inbound, rate and hourly rules: a person talking to their agent must
never be throttled, and a person's request often legitimately produces several
progress posts. ANY human message counts, ambient or addressed, because the
operator wants no cap where a person participates; the TTL is 10 minutes, not
30, so one ambient line cannot exempt a runaway for half an hour (review round
1, S-2), and the absolute floor bounds whatever remains. Its duplicate check is
narrower — refused only when no new message has arrived since the identical
post — so "status?" asked twice can be answered "still running" twice.

REFUSALS DO NOT DROP THE MESSAGE. Every refusal carries ``code``,
``retry_after_s`` (the earliest whole second at which the guard would admit
THIS post absent a new inbound message: the latest of the violated rules'
admit times, rounded UP) and a
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
stored (the body digest is keyed, see above). Schema changes are additive
only: an older build may share the file during an upgrade (the connector and a
session can run different builds), and they run inside the same IMMEDIATE
transaction as the decision, so two processes opening an old file at once
cannot both try the same ``ALTER`` (review round 1, m1).

FAIL CLOSED. If the file or the digest key cannot be opened, created or
written, a post is refused with ``reply_guard_unavailable`` (with a short
``retry_after_s`` and the same keep-your-draft guidance as every refusal):
loop/cost safety is the requirement, and a silently disabled guard is how a
runaway would start. The trigger side still fails OPEN (``record_trigger``
never raises), a pre-existing asymmetry left as is in this change.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import secrets
import sqlite3
import tempfile
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
#: The client's standalone bound on agent-to-agent posts (module docstring).
AGENT_HOURLY_POSTS = 120
AGENT_HOURLY_WINDOW_S = 3600.0
#: Same normalized body from the same endpoint in the same conversation.
DUPLICATE_WINDOW_S = 600.0
#: Hard ceiling per conversation per window over EVERY post.
ABSOLUTE_POSTS = 120
ABSOLUTE_WINDOW_S = 600.0

#: An agent trigger older than this no longer explains a post: the agent is
#: acting on its own, not answering the agent that last wrote.
TRIGGER_TTL_S = 1800.0
#: A human trigger exempts posts for this long (S-2: shorter than the agent
#: TTL, so one ambient human line cannot open a long exemption window).
HUMAN_TRIGGER_TTL_S = 600.0

#: The longest window any rule looks back over.
_HORIZON_S = max(
    PER_INBOUND_WINDOW_S,
    RATE_LONG_WINDOW_S,
    AGENT_HOURLY_WINDOW_S,
    DUPLICATE_WINDOW_S,
    ABSOLUTE_WINDOW_S,
)
#: Rows older than this carry no decision (the horizon, doubled).
_RETAIN_S = 2 * _HORIZON_S
#: The retry hint on ``reply_guard_unavailable``: the usual cause is a busy or
#: mid-upgrade file, which clears in seconds.
UNAVAILABLE_RETRY_S = 5

CODE_NO_NEW_INBOUND = "reply_no_new_inbound"
CODE_RATE = "reply_rate_limited"
CODE_AGENT_HOURLY = "reply_agent_hourly"
CODE_DUPLICATE = "reply_duplicate"
CODE_ABSOLUTE = "reply_absolute_cap"
CODE_UNAVAILABLE = "reply_guard_unavailable"

_BUSY_TIMEOUT_S = 5.0
_KEY_BYTES = 32

# ``runs`` is the retired cool-down's table. It is still CREATED so an older
# build sharing this file during an upgrade finds it, but this build never reads
# it: a ``cooldown_until`` an older build wrote must not block a post (an agent
# in cool-down at upgrade time posts as soon as the new build runs).
_SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS triggers (
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    at REAL NOT NULL,
    sender_kind TEXT NOT NULL,
    mention INTEGER NOT NULL,
    directed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (endpoint_id, conversation_id)
)""",
    """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    at REAL NOT NULL,
    agent_to_agent INTEGER NOT NULL,
    human_prompted INTEGER NOT NULL DEFAULT 0,
    body_hash TEXT
)""",
    """
CREATE INDEX IF NOT EXISTS posts_by_conversation
    ON posts (endpoint_id, conversation_id, at)""",
    """
CREATE TABLE IF NOT EXISTS runs (
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    consecutive INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (endpoint_id, conversation_id)
)""",
)


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


def body_hash(body: str, *, key: bytes) -> str:
    """HMAC-SHA256 of the body with case and whitespace normalized.

    Normalized so a loop that varies only spacing or capitalization ("Thanks!"
    vs "thanks! ") still reads as the same post. Keyed so the stored digest of
    a short reply cannot be reversed with a dictionary (S-3). The digest is all
    that is ever stored: the guard file must never hold message content.
    """

    normalized = " ".join(body.casefold().split())
    return hmac.new(key, normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def key_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "reply_guard.key"


def _digest_key(config_dir: Path | str) -> bytes:
    """The per-install digest key, created on first use. Raises ``ReplyGuardUnavailable``.

    CREATED ATOMICALLY: the random key is written to a private temp file (0600,
    ``mkstemp``) and published with ``os.link``, which fails if the name exists.
    Two processes racing on first use therefore never see a half-written key,
    and the loser adopts the winner's key, so both compute the same digests. A
    key of the wrong size is damage, not something to overwrite: refusing keeps
    the rule honest (fail closed, like the database). The key is never logged.
    """

    path = key_path(config_dir)
    try:
        try:
            key = path.read_bytes()
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(dir=path.parent, prefix=".reply_guard.key.")
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(secrets.token_bytes(_KEY_BYTES))
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temp, path)
                except FileExistsError:
                    pass  # another process published first; read its key below
            finally:
                os.unlink(temp)
            key = path.read_bytes()
    except OSError as failure:
        raise ReplyGuardUnavailable("reply guard key unavailable") from failure
    if len(key) != _KEY_BYTES:
        raise ReplyGuardUnavailable("reply guard key damaged")
    return key


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


def unavailable_result() -> dict[str, Any]:
    """The ``reply_guard_unavailable`` refusal, in the same shape as every other.

    Round 1, m3: this arm used to say only "not posting", which read as "drop
    it". The usual cause (a busy or mid-upgrade file) clears in seconds, so it
    carries a short ``retry_after_s`` and the same keep-your-draft guidance.
    """

    return Refusal(
        code=CODE_UNAVAILABLE,
        retry_after_s=UNAVAILABLE_RETRY_S,
        message=(
            "not sent — the reply guard could not read its state; "
            f"posting resumes in {UNAVAILABLE_RETRY_S}s if it recovers"
        ),
    ).as_result()


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
        # IMMEDIATE: check-then-reserve is one decision. Two sessions of the
        # same endpoint posting at once must not both see the last free slot.
        # The schema and the additive ALTERs run INSIDE it (round 1, m1): run
        # before it, two processes opening an older file both saw a column
        # missing and the loser failed closed on "duplicate column name".
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA:
                conn.execute(statement)
            _ensure_columns(conn)
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

    return max(1, math.ceil(delay))


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

    ``body`` feeds the duplicate check (digested under the per-install key, so
    a key that cannot be read or created fails closed); ``None`` (a caller that
    has no body) skips that rule only. The slot is recorded BEFORE the post goes out (so a
    concurrent post sees it) and must be given back through :func:`release`
    when the hub did not accept the post.
    """

    now = time.time() if now is None else now
    digest = body_hash(body, key=_digest_key(config_dir)) if body is not None else None
    horizon = now - _HORIZON_S
    with _connect(config_dir) as conn:
        rows = [
            (float(row[0]), bool(row[1]), row[2], bool(row[3]))
            for row in conn.execute(
                "SELECT at, human_prompted, body_hash, agent_to_agent FROM posts "
                "WHERE endpoint_id = ? AND conversation_id = ? AND at > ? ORDER BY at",
                (endpoint_id, conversation_id, horizon),
            )
        ]
        recent = [(at, human, digested) for at, human, digested, _ in rows]
        trigger = conn.execute(
            "SELECT at, sender_kind FROM triggers WHERE endpoint_id = ? AND conversation_id = ?",
            (endpoint_id, conversation_id),
        ).fetchone()
        inbound_at = float(trigger[0]) if trigger is not None else None
        kind = str(trigger[1]) if trigger is not None else ""
        age = now - inbound_at if inbound_at is not None else None
        human_prompted = age is not None and kind == "human" and age <= HUMAN_TRIGGER_TTL_S
        agent_to_agent = age is not None and kind == "agent" and age <= TRIGGER_TTL_S

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

            # HOURLY AGENT-TO-AGENT CEILING (S-1): the standalone bound on a
            # loop where every post answers a new message. Counts only posts
            # recorded as answering an agent, and only when THIS one does too:
            # an unprompted post is bounded by the per-inbound rule instead.
            if agent_to_agent:
                hourly = [
                    at
                    for at, _, _, to_agent in rows
                    if to_agent and at > now - AGENT_HOURLY_WINDOW_S
                ]
                if len(hourly) >= AGENT_HOURLY_POSTS:
                    violations.append(
                        (
                            CODE_AGENT_HOURLY,
                            hourly[len(hourly) - AGENT_HOURLY_POSTS] + AGENT_HOURLY_WINDOW_S - now,
                            f"hourly limit: {AGENT_HOURLY_POSTS} replies to agents in this "
                            f"conversation within {int(AGENT_HOURLY_WINDOW_S // 60)} minutes",
                        )
                    )

        if digest is not None:
            same = [at for at, _, h in recent if h == digest and at > now - DUPLICATE_WINDOW_S]
            nothing_new = bool(same) and (inbound_at is None or same[-1] >= inbound_at)
            if nothing_new:
                # Nothing arrived since the identical post: repeating it is a
                # loop, whoever prompted it. Admitted once that copy ages out.
                freeing: float | None = same[-1]
            elif same and not human_prompted and len(same) >= 2:
                # A new message arrived, so one repeat is a legitimate answer
                # to a new request ("Approved." twice, M1); a THIRD identical
                # answer inside the window is the loop ("Thanks!" / "You're
                # welcome!"). Admitted once fewer than two copies remain.
                freeing = same[-2]
            else:
                # Answering a person after a new message: "still running" may
                # answer "status?" as often as it is asked.
                freeing = None
            if freeing is not None:
                violations.append(
                    (
                        CODE_DUPLICATE,
                        freeing + DUPLICATE_WINDOW_S - now,
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
    "AGENT_HOURLY_POSTS",
    "AGENT_HOURLY_WINDOW_S",
    "CODE_ABSOLUTE",
    "CODE_AGENT_HOURLY",
    "CODE_DUPLICATE",
    "CODE_NO_NEW_INBOUND",
    "CODE_RATE",
    "CODE_UNAVAILABLE",
    "DUPLICATE_WINDOW_S",
    "HUMAN_TRIGGER_TTL_S",
    "PER_INBOUND_POSTS",
    "PER_INBOUND_WINDOW_S",
    "RATE_LONG_POSTS",
    "RATE_LONG_WINDOW_S",
    "RATE_SHORT_POSTS",
    "RATE_SHORT_WINDOW_S",
    "TRIGGER_TTL_S",
    "UNAVAILABLE_RETRY_S",
    "Refusal",
    "ReplyGuardUnavailable",
    "Reservation",
    "body_hash",
    "guard_path",
    "key_path",
    "unavailable_result",
    "record_trigger",
    "release",
    "reserve_post",
]

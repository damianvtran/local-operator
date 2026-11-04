"""The client half of the conversation wake contract: post bookkeeping, never a cap.

WHY THIS EXISTS. An agent member is woken for every message in every
conversation it belongs to. This module sits at the one place an agent writes
into a conversation (``bridge channel_post`` and ``bridge dm``) and keeps a
small per-endpoint, per-conversation record of what it posted and what last
woke it.

WHAT IT MUST NOT DO (operator direction, 2026-11-03). It never refuses a post
because of HOW MANY posts were made. Two earlier versions did — first a
10-minute cool-down after 3 consecutive agent-to-agent replies
(``reply_cooldown``), then runaway rules (≤3 posts per inbound, 20/min, 60/10
min, 120/h agent-to-agent, 120/10 min absolute). Each refused real traffic: an
agent's reply carrying a production-access approval, group relays, agents
keeping each other updated. The operator's instruction: "no limit on the
number of agent replies so that agents can keep each other updated and
notified … a warning instead of an error". So every count/velocity rule is
gone, and the only outputs are:

* **A one-line warning on long one-sided runs.** When this endpoint has posted
  N messages in this conversation with no new inbound message since, the
  SUCCESSFUL result carries ``warning`` once, at N = ``RUN_WARN_FIRST`` (5),
  ``RUN_WARN_SECOND`` (10), then every further ``RUN_WARN_EVERY`` (10). Five,
  because a multi-part answer (text, follow-up, correction) is three posts and
  a status stream that reaches five with nobody answering is the first point a
  nudge is worth its tokens. Then 10 and every 10, not every post: repeating
  the line on each post would itself be the spam it warns about, and every
  tenth post keeps a long monologue reminded at one line per ten posts. A new
  inbound message resets the run, so ordinary back-and-forth never sees it.
  The post is delivered either way; the warning is advice, not a gate.
* **An exact double send is skipped** (``skipped: true``, still ``ok``) — the
  one case the operator named. The same endpoint posting a byte-identical body
  to the same conversation, the same THREAD and the same addressees (the
  resolved mention set, sorted) within ``DUPLICATE_WINDOW_S`` (10 s) of the
  previous copy, with no new inbound message in between, is an accidental
  double send (a retried tool call, a model emitting the call twice). EXACT,
  not case/whitespace-folded as the round-1 rule was: a skip is the only path
  that drops a post, so its predicate must be the narrowest one; a changed
  word, case or spacing is a deliberate edit and posts, and so is "Done." to a
  second thread or "Approved." to a second person (review round 3, M-1: two
  questions coalesced into one wake share one trigger time, so the
  "no new inbound" clause alone could not tell those apart). The "no new
  inbound in between" clause keeps "Yes." answering two different questions
  3 s apart from being eaten. The stored value is an HMAC-SHA256 of that
  (body, thread, addressees) tuple under a random per-install key
  (``reply_guard.key``, 0600, beside the database), never the body: a plain
  hash of "approved" or "yes" would be recoverable with a small dictionary
  (review round 1, S-3). The skip matches a RESERVED row, not a confirmed one:
  if the earlier call then fails at the hub, its row is released but the
  skipped call has already answered (review round 3, m-2). Confirming rows
  would need the skipped call to wait on its sibling's outcome, so instead the
  skip never claims delivery — it says "identical to a message you sent Ns
  ago … not posted (if that send failed, resend with changed text)".

WHAT BOUNDS AN AGENT LOOP NOW (review round 3, S-8). Nothing on this client,
by the operator's decision of 2026-11-03. On the server, agent-server #294
removes the agent-run wake caps; what remains is the S-2 per-recipient ROOM
wake budget (20 wakes per endpoint per conversation and 40 per endpoint, per
10 min), and it covers only UNADDRESSED posts in a group or channel. A DM
loop, or a loop where each agent @-mentions or replies in a thread the other
follows, is unbounded in time by design: it costs 3,600 / (seconds per turn)
posts an hour until one agent ends its turn with ``no_reply``, which the wake
card's ``expect:`` line asks for on agent chatter. The client's part is
advisory: the one-sided run warning above, and #294's ``agent_wake`` notice
surfaced as a ``warning`` once per threshold (:func:`agent_run_warning`) so a
back-and-forth pair is told to be brief too.

WHO WRITES WHAT. Two processes share this state: the connector (which knows
what woke the agent) records each delivered wake's TRIGGER — sender kind and
mention, never a body or a name — and the session's bridge tool reserves each
post. They meet in ONE small SQLite file beside the bridge store,
``<config_dir>/team_bridge/reply_guard.sqlite3``. It is deliberately NOT a
table in ``state.sqlite3``: that store's schema is migration-versioned, and this
state is disposable (losing it only resets the run count and the duplicate
memory, it never loses a message). Keys are the agent ENDPOINT and the
conversation id; nothing content-bearing is stored. Schema changes are additive
only: an older build may share the file during an upgrade (the connector and a
session can run different builds), and they run inside the same IMMEDIATE
transaction as the decision, so two processes opening an old file at once
cannot both try the same ``ALTER`` (review round 1, m1). Rows an older build
wrote — ``runs.cooldown_until``, its ``posts`` — are never a reason to refuse:
nothing here refuses at all.

FAIL OPEN, WITH A WARNING (review round 3, m-3; supersedes round 2's S-6
refusals). With no count left to enforce, this state is advisory bookkeeping:
the duplicate memory and the run count. A refusal when it cannot be read would
therefore only DROP messages — round 2 refused every post on a damaged key or
a corrupt file until the operator repaired it, and QA showed the model never
even read the retry guidance. So :func:`reserve_post` still raises
``ReplyGuardUnavailable`` (classified ``transient`` for BUSY/LOCKED, otherwise
broken), and the tool POSTS ANYWAY with :func:`unavailable_warning` in the
result: "reply guard busy; posted without duplicate/run tracking", or "reply
guard state is broken (<reason>); posted without duplicate/run tracking — tell
the operator". That post has no duplicate skip and no run count; the cost is
that accidental double sends are not caught while the state is unusable
(review round 4, security n-1). The trigger side fails open too
(``record_trigger`` never raises); if a trigger is lost under a lock, an
identical short answer to that unrecorded question within 10 s can be skipped
(review round 3, S-10 — a narrow, pre-existing asymmetry, deferred).

ROWS AN OLDER BUILD WROTE never count towards the run warning (review round 3,
QA Q5): only rows with ``tracked = 1`` (written by this build) are counted, so
the first post after an upgrade does not warn "Posted 140 messages".
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

#: One-sided run warnings (module docstring): at 5, at 10, then every 10.
RUN_WARN_FIRST = 5
RUN_WARN_SECOND = 10
RUN_WARN_EVERY = 10
#: An exact body re-sent by the same endpoint to the same conversation, thread
#: and addressees within this many seconds, with no new inbound in between, is
#: an accidental double send and is skipped.
DUPLICATE_WINDOW_S = 10.0
#: agent-server #294's ``agent_wake.consecutive_agent_messages`` thresholds at
#: which the tool adds a warning: 8 (the run at which the server starts
#: sending the notice even when senders alternate), 10, then every 10. Levels,
#: not exact counts: in a back-and-forth each agent sees every OTHER value
#: (8, 10, 12 … or 9, 11, 13 …), so an equality test would miss half of them.
AGENT_RUN_WARN_FIRST = 8
AGENT_RUN_WARN_EVERY = 10
#: The largest ``consecutive_agent_messages`` read as a real run (S-11): far
#: past any conversation, and well inside SQLite's 64-bit INTEGER.
AGENT_RUN_COUNT_MAX = 10**6

#: Classification of a post for an OLDER build sharing this file (it reads the
#: ``agent_to_agent`` and ``human_prompted`` columns); this build decides
#: nothing from them. An agent trigger older than this no longer explains a post.
TRIGGER_TTL_S = 1800.0
HUMAN_TRIGGER_TTL_S = 600.0

#: Rows older than this are dropped. A one-sided run is counted over what is
#: retained, so a day bounds the table while covering any real monologue.
_RETAIN_S = 86_400.0

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
    body_hash TEXT,
    tracked INTEGER NOT NULL DEFAULT 0
)""",
    """
CREATE INDEX IF NOT EXISTS posts_by_conversation
    ON posts (endpoint_id, conversation_id, at)""",
    """
CREATE TABLE IF NOT EXISTS agent_run_notices (
    endpoint_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    level INTEGER NOT NULL,
    PRIMARY KEY (endpoint_id, conversation_id)
)""",
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
    """The guard's state could not be read or written.

    The tool posts anyway (module docstring, FAIL OPEN) and reports it with
    :func:`unavailable_warning`. ``transient`` is True only for a busy/locked
    database, which clears in seconds; every other cause is a broken install
    the operator has to repair, so only that warning asks for them (S-6).
    """

    def __init__(self, reason: str, *, transient: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.transient = transient


def _sqlite_unavailable(failure: sqlite3.Error) -> ReplyGuardUnavailable:
    """Classify a SQLite failure: only BUSY/LOCKED is worth a short retry."""

    code = getattr(failure, "sqlite_errorcode", None)
    busy = code is not None and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    # Older SQLite bindings may not attach the code; the message is stable.
    text = str(failure).casefold()
    busy = busy or "database is locked" in text or "database is busy" in text
    if busy:
        return ReplyGuardUnavailable("reply guard state is busy", transient=True)
    return ReplyGuardUnavailable("reply guard state file is unreadable or unwritable")


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add the columns later builds introduced to a guard file an earlier one wrote.

    The guard's state is DISPOSABLE (losing it only resets the run count and
    the duplicate memory), so an
    additive column is the whole migration — no version machinery. Same rule as
    ``store._ensure_card_posted_column`` on the durable store, and additive only
    because an older build can share this file during an upgrade.
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(posts)")}
    if "human_prompted" not in columns:
        conn.execute("ALTER TABLE posts ADD COLUMN human_prompted INTEGER NOT NULL DEFAULT 0")
    if "body_hash" not in columns:
        # NULL on rows an older build wrote: those posts simply take no part in
        # the duplicate check, which is the harmless direction (they post).
        conn.execute("ALTER TABLE posts ADD COLUMN body_hash TEXT")
    if "tracked" not in columns:
        # 0 on every row an older build wrote, so those rows never count towards
        # the run warning (round 3, QA Q5: the first post after an upgrade
        # warned "Posted 140 messages" from the previous build's history).
        conn.execute("ALTER TABLE posts ADD COLUMN tracked INTEGER NOT NULL DEFAULT 0")
    trigger_columns = {row[1] for row in conn.execute("PRAGMA table_info(triggers)")}
    if "directed" not in trigger_columns:
        # A DIAGNOSTIC record, not a read path: no query reads this column back
        # (round 6, m1). An older row therefore defaults to 0 and nothing
        # depends on the value.
        conn.execute("ALTER TABLE triggers ADD COLUMN directed INTEGER NOT NULL DEFAULT 0")


def body_hash(
    body: str, *, key: bytes, thread_root_id: str = "", addressees: Iterable[str] = ()
) -> str:
    """HMAC-SHA256 of the EXACT body plus where it went (thread, addressees).

    Exact because the duplicate rule is a skip — the one path that drops a
    post — so only a byte-identical re-send counts as a double send; any edit
    is deliberate and posts. The thread root and the addressee set are part of
    the digested input (round 3, M-1): "Done." to two threads, or "Approved."
    to two people, are two messages. Addressees are de-duplicated, stripped and
    sorted, so the order a caller listed them in cannot make a re-send look
    new. The tuple is JSON-encoded so no body can forge another field's
    boundary. Keyed so the stored digest of a short reply cannot be reversed
    with a dictionary (S-3). The digest is all that is ever stored: the guard
    file must never hold message content. Digests an earlier build stored
    never equal these, which only means an upgrade-time re-send posts.
    """

    targets = sorted({str(item).strip() for item in addressees if str(item).strip()})
    encoded = json.dumps([body, thread_root_id.strip(), targets], ensure_ascii=False)
    return hmac.new(key, encoded.encode("utf-8"), hashlib.sha256).hexdigest()


def key_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "reply_guard.key"


def _digest_key(config_dir: Path | str) -> bytes:
    """The per-install digest key, created on first use. Raises ``ReplyGuardUnavailable``.

    CREATED ATOMICALLY: the random key is written to a private temp file (0600,
    ``mkstemp``) and published with ``os.link``, which fails if the name exists.
    Two processes racing on first use therefore never see a half-written key,
    and the loser adopts the winner's key, so both compute the same digests. A
    key of the wrong size is damage, not something to overwrite: the post goes
    out untracked and the warning asks for the operator, because the damage is
    PERMANENT (S-6). The key is never logged.
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
        raise ReplyGuardUnavailable("reply guard key is unreadable or unwritable") from failure
    if len(key) != _KEY_BYTES:
        raise ReplyGuardUnavailable("reply guard key is damaged")
    return key


@dataclass(frozen=True)
class Reservation:
    """A granted post slot. ``release`` it if the post did not land.

    ``warning`` is the one-line run warning to attach to the SUCCESSFUL result
    (absent — ``None`` — on all but the threshold posts).
    """

    post_id: int
    agent_to_agent: bool
    warning: str | None = None


@dataclass(frozen=True)
class Skipped:
    """An exact double send within ``DUPLICATE_WINDOW_S``: not posted, not an error."""

    seconds_ago: int
    in_thread: bool = False

    @property
    def warning(self) -> str:
        # Says "not posted" ONCE (round 3, n-2: it is billed on every skip) and
        # never claims the earlier copy was delivered (m-2): it may still be in
        # flight, and if it failed its row is released, so a resend posts.
        where = "in this thread" if self.in_thread else "here"
        return (
            f"Skipped: identical to a message you sent {self.seconds_ago}s ago {where}; "
            "not posted. If that send failed, send it again or change the text."
        )

    def as_result(self, conversation_id: str) -> dict[str, Any]:
        # ``ok``: nothing went wrong. Reporting an error here would invite the
        # very retry being skipped. The tool appends ``warning`` to the text
        # the model reads, so ``content`` names only the destination.
        return {
            "ok": True,
            "skipped": True,
            "state": "skipped",
            "conversation_id": conversation_id,
            "warning": self.warning,
            "content": f"duplicate send to {conversation_id}",
        }


def unavailable_warning(failure: ReplyGuardUnavailable) -> str:
    """The one-line warning for a post that went out WITHOUT the guard (m-3).

    Busy clears on its own, so it asks nothing of anyone. Broken does not, so it
    names the cause and asks for the operator — once per post, never a retry
    loop, because the post itself already went out.
    """

    if failure.transient:
        return "reply guard busy; posted without duplicate/run tracking"
    return (
        f"reply guard state is broken ({failure.reason}); posted without "
        "duplicate/run tracking — tell the operator"
    )


def guard_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "reply_guard.sqlite3"


@contextmanager
def _connect(config_dir: Path | str) -> Iterator[sqlite3.Connection]:
    path = guard_path(config_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
    except OSError as failure:
        raise ReplyGuardUnavailable(
            "reply guard state directory is unreadable or unwritable"
        ) from failure
    except sqlite3.Error as failure:
        raise _sqlite_unavailable(failure) from failure
    try:
        # IMMEDIATE: check-then-reserve is one decision. Two sessions of the
        # same endpoint posting at once must not both count the same run
        # position, nor both miss each other's identical double send.
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
        raise _sqlite_unavailable(failure) from failure
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
    the "new inbound" that resets the one-sided run and separates a double send
    from an answer to a new message; its ``sender_kind`` classifies the post
    for an older build sharing the file.
    ``directed`` is the connector's STRICT addressedness predicate (an
    @-mention, or a reply the server states this endpoint FOLLOWS); it is kept
    as a diagnostic column and to clear an older build's cool-down (below).
    ``directed=None`` falls back to ``mention`` for a caller that predates the
    bit. Never raises: a trigger that cannot be recorded leaves the previous
    one (at worst a run warning comes early), and the post side fails open too
    (module docstring, FAIL OPEN; review round 4, security n-2).
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
            if kind == "human":
                # A person spoke: the server resets the conversation's agent
                # run to 0 on ANY human post, so the remembered notice level
                # goes too, or the next run's first threshold would compare
                # equal to the old one and be lost (review round 4, QA Q-2).
                conn.execute(
                    "DELETE FROM agent_run_notices WHERE endpoint_id = ? AND conversation_id = ?",
                    (endpoint_id, conversation_id),
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


def _run_warning(position: int) -> str | None:
    """The one-line warning for the ``position``-th post of a one-sided run, or None.

    Fires exactly once per threshold crossing — 5, 10, 20, 30, … — so a long
    monologue is nudged one line per ten posts rather than on every post.
    """

    if position < RUN_WARN_FIRST:
        return None
    if position not in (RUN_WARN_FIRST, RUN_WARN_SECOND) and (
        position < RUN_WARN_SECOND or position % RUN_WARN_EVERY
    ):
        return None
    return (
        f"Posted {position} messages in this conversation with no reply; "
        "be brief and avoid repeating yourself."
    )


def reserve_post(
    config_dir: Path | str,
    *,
    endpoint_id: str,
    conversation_id: str,
    body: str | None = None,
    thread_root_id: str = "",
    addressees: Iterable[str] = (),
    now: float | None = None,
) -> Reservation | Skipped:
    """Record one post. Never refuses on counts. Raises ``ReplyGuardUnavailable``.

    Returns :class:`Skipped` only for an exact double send (module docstring);
    otherwise a :class:`Reservation`, whose ``warning`` is set on the posts that
    cross a one-sided-run threshold. ``body`` feeds the double-send check
    together with ``thread_root_id`` and ``addressees`` (the resolved mention
    account ids), digested under the per-install key; ``None`` (a caller that
    has no body) skips that check only. Raises when the state is unusable; the
    caller posts anyway (module docstring, FAIL OPEN). The slot is recorded
    BEFORE the post goes out (so a concurrent post sees it) and must be given
    back through :func:`release` when the hub did not accept the post.
    """

    now = time.time() if now is None else now
    digest = (
        body_hash(
            body,
            key=_digest_key(config_dir),
            thread_root_id=thread_root_id,
            addressees=addressees,
        )
        if body is not None
        else None
    )
    with _connect(config_dir) as conn:
        trigger = conn.execute(
            "SELECT at, sender_kind FROM triggers WHERE endpoint_id = ? AND conversation_id = ?",
            (endpoint_id, conversation_id),
        ).fetchone()
        inbound_at = float(trigger[0]) if trigger is not None else None
        kind = str(trigger[1]) if trigger is not None else ""
        age = now - inbound_at if inbound_at is not None else None
        human_prompted = age is not None and kind == "human" and age <= HUMAN_TRIGGER_TTL_S
        agent_to_agent = age is not None and kind == "agent" and age <= TRIGGER_TTL_S

        if digest is not None:
            previous = conn.execute(
                "SELECT MAX(at) FROM posts WHERE endpoint_id = ? AND conversation_id = ? "
                "AND body_hash = ? AND at > ?",
                (endpoint_id, conversation_id, digest, now - DUPLICATE_WINDOW_S),
            ).fetchone()[0]
            # A trigger recorded AFTER the earlier copy means this answers a new
            # message, not an accident. A trigger at the same instant came first
            # (the connector records it before the wake), so it does not count.
            if previous is not None and (inbound_at is None or inbound_at <= float(previous)):
                return Skipped(
                    seconds_ago=max(0, math.floor(now - float(previous))),
                    in_thread=bool(thread_root_id.strip()),
                )

        # The one-sided run: this endpoint's posts since the latest inbound (a
        # post at the same instant as the trigger came after it), this one
        # included. With no trigger at all, everything retained counts — but
        # only rows THIS build wrote (``tracked``), never an older build's (Q5).
        since = inbound_at if inbound_at is not None else now - _RETAIN_S
        before = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE endpoint_id = ? AND conversation_id = ? "
            "AND at >= ? AND tracked = 1",
            (endpoint_id, conversation_id, since),
        ).fetchone()[0]

        cursor = conn.execute(
            "INSERT INTO posts "
            "(endpoint_id, conversation_id, at, agent_to_agent, human_prompted, body_hash, "
            "tracked) VALUES (?, ?, ?, ?, ?, ?, 1)",
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
        return Reservation(
            post_id=int(cursor.lastrowid or 0),
            agent_to_agent=agent_to_agent,
            warning=_run_warning(int(before) + 1),
        )


def _agent_run_level(count: int) -> int:
    """The highest notice threshold ``count`` has reached: 0, 8, 10, 20, 30, …"""

    if count < AGENT_RUN_WARN_FIRST:
        return 0
    if count < AGENT_RUN_WARN_EVERY:
        return AGENT_RUN_WARN_FIRST
    return count - count % AGENT_RUN_WARN_EVERY


def agent_run_warning(
    config_dir: Path | str, *, endpoint_id: str, conversation_id: str, notice: Any
) -> str | None:
    """The one-line warning for agent-server #294's ``agent_wake`` notice, or None.

    WHY (review round 3, I-1). The one-sided run warning resets on every
    inbound message, so two agents taking turns never see it — 0 warnings in
    8,640 measured alternating posts. The server counts the conversation's
    consecutive agent messages and returns ``agent_wake`` on an agent's own
    send answer once that run reaches 8 (or the sender has posted 3 in a row).
    This turns it into the operator's "warning instead of an error" for the
    back-and-forth case, once per threshold crossing (8, 10, 20, …), never on
    every post. The level reached is remembered per endpoint and conversation.

    RE-ARMING (review round 4, QA Q-2 / code m-B). #294 OMITS ``agent_wake``
    below its threshold (run < 8 and streak < 3), so in an alternating
    exchange a count below 8 is never seen: a reset that waited for one never
    happened, and a second run's warning at the old level was lost. So the
    caller passes ``notice=None`` for a SUCCESSFUL send whose answer carried no
    notice, and that is read as "below the threshold": the level resets to 0.
    A human trigger clears it too (:func:`record_trigger`), the server's own
    reset rule. On an older server that never sends the field the level is
    always 0, so the reset changes nothing — the feature stays ADDITIVE.

    BOUNDED (review round 4, security S-11). Only the hub controls the count;
    one outside 0..``AGENT_RUN_COUNT_MAX`` is not a run this client can name,
    so it yields no warning and is not stored (an int past SQLite's 64-bit
    range raised ``OverflowError`` after the post had landed). Best effort:
    unusable guard state returns None rather than a second warning about
    bookkeeping; the tool also treats any failure here as non-fatal.
    """

    count: Any = None
    if isinstance(notice, dict):
        count = notice.get("consecutive_agent_messages")
        if not isinstance(count, int) or isinstance(count, bool):
            return None
        if not 0 <= count <= AGENT_RUN_COUNT_MAX:
            return None
    elif notice is not None:
        return None
    level = _agent_run_level(count) if count is not None else 0
    try:
        with _connect(config_dir) as conn:
            row = conn.execute(
                "SELECT level FROM agent_run_notices WHERE endpoint_id = ? AND conversation_id = ?",
                (endpoint_id, conversation_id),
            ).fetchone()
            seen = int(row[0]) if row is not None else 0
            conn.execute(
                "INSERT INTO agent_run_notices (endpoint_id, conversation_id, level) "
                "VALUES (?, ?, ?) ON CONFLICT(endpoint_id, conversation_id) DO UPDATE SET "
                "level = excluded.level",
                (endpoint_id, conversation_id, level),
            )
    except ReplyGuardUnavailable:
        return None
    # A lower level than remembered is a NEW run (the server's count reset).
    if level == 0 or level == seen:
        return None
    return (
        f"{count} consecutive agent messages in this conversation; "
        "be brief and avoid repeating yourself."
    )


def release(
    config_dir: Path | str, reservation: Reservation, *, endpoint_id: str, conversation_id: str
) -> None:
    """Give back a slot whose post the hub did not accept.

    The whole row goes, body hash included: a post the hub never took must not
    count towards the run, and must not make its own retry a "duplicate".
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
    "AGENT_RUN_COUNT_MAX",
    "AGENT_RUN_WARN_EVERY",
    "AGENT_RUN_WARN_FIRST",
    "DUPLICATE_WINDOW_S",
    "HUMAN_TRIGGER_TTL_S",
    "RUN_WARN_EVERY",
    "RUN_WARN_FIRST",
    "RUN_WARN_SECOND",
    "TRIGGER_TTL_S",
    "ReplyGuardUnavailable",
    "Reservation",
    "Skipped",
    "body_hash",
    "guard_path",
    "key_path",
    "agent_run_warning",
    "unavailable_warning",
    "record_trigger",
    "release",
    "reserve_post",
]

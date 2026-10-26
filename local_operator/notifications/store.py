"""The operator's inbox: durable items agents raise with ``notify``.

WHY A STORE OF ITS OWN (``docs/design/notifications-inbox.md`` §1). A push or a
banner is gone the moment it is dismissed, so a priority finding raised by a
watcher at 3 a.m. is buried by the next morning's backlog. This store keeps the
item until the operator reads or dismisses it, and every surface (desktop page,
TUI page, relay, native) reads the same rows.

WHY ``notifications.db`` AND NOT ``attention.db``. ``attention.db`` is the most
contended store on the machine (``session/attention.py`` ``_BUSY_TIMEOUT_MS``:
~25 concurrent sessions publish completions into it). A second write stream
there would widen that contention and pay its schema check on every publish, so
items live in an independent file with the same connection discipline:
``BEGIN IMMEDIATE`` writes, a measured busy window, one retry on contention,
``mode=ro`` readers that can never create or migrate the file.

THIS MODULE MUST STAY STDLIB-ONLY (the ``asks/store.py`` rule). Its readers are
the desktop feed's doorbell, the relay, the TUI sidebar poll and the cleanup
sweep, none of which may load the harness for a count. It imports nothing from
``local_operator`` except the shared contention set, which is itself
stdlib-only; ``tests/unit/test_import_graph.py`` pins that in a fresh
interpreter.

TOKEN-BOUND, NEVER A SWEEP (``docs/ATTENTION.md`` rule R10, applied to items).
A read or dismiss by id acts only when the caller names the ``revision`` it
rendered, so a keyed update published after the render stays unread. A
by-session or all-items write acts only on rows whose ``seq`` is at or below
the ``through_seq`` the caller last saw. Nothing in this module reads an item
on a timer.

THE TWO STORES NEVER ACKNOWLEDGE EACH OTHER. Reading a completion never reads an
item and the reverse: the read marks here are this store's own columns, and
``attention.db`` is not opened from this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

# THE CONTENTION SET IS SHARED, NOT RESTATED: ``session/attention.py`` records
# the drift two hand-written lists produced (review round 1, MINOR-1 there).
from local_operator.session.store_failures import BUSY_ERRONAMES

logger = logging.getLogger(__name__)

#: The store's file name inside the config dir.
NOTIFICATIONS_DB_NAME = "notifications.db"

#: ``PRAGMA user_version`` of the shipped schema. Migrations are ADDITIVE only
#: (the ``attention.py`` rule): an older build must still read a newer file.
SCHEMA_VERSION = 1

#: Copied from ``session/attention.py`` (``_BUSY_TIMEOUT_MS``,
#: ``_CONNECT_TIMEOUT_S``, ``_CONTENTION_ATTEMPTS``, ``_CONTENTION_BACKOFF_S``)
#: with their measured rationale, not imported: those names are private to that
#: store, and the budgets are sized for the same population of writers (every
#: live session plus the servers). One retry; the measured worst case of two
#: attempts at a 5 s window is ~12 s for a write that meets a held RESERVED
#: lock, which is why a third attempt is not taken.
_BUSY_TIMEOUT_MS = 5000
_CONNECT_TIMEOUT_S = 5.0
_CONTENTION_ATTEMPTS = 2
_CONTENTION_BACKOFF_S = 0.2

#: WAL is adopted opportunistically on the write path (the ``attention.py``
#: ``_adopt_wal`` reasoning: readers stop blocking writers' commits). A short
#: window so an adoption that meets a lock never costs a caller the full budget.
_WAL_BUSY_TIMEOUT_MS = 250

PRIORITIES = ("low", "normal", "high")
_PRIORITY_RANK = {name: rank for rank, name in enumerate(PRIORITIES)}

#: The surfaces whose ``seen`` receipts are accepted. ``native`` is the phone's
#: own door (D11): its edges are touch/press/scroll and its evidence is
#: re-checked here exactly like the desktop's, so a receipt cannot be minted
#: from an assertion the server cannot examine.
SEEN_SURFACES = ("desktop", "tui", "native")

#: How long after the input edge a surface's receipt is admissible. The desktop
#: waits a second so a wheel tick that merely lands on the pane cannot read; the
#: TUI has already required a Key/MouseDown on a rendered card, so its own
#: positive-elapsed check is enough; the phone gets the desktop's second, because
#: a touch that scrolls a list is the same class of accidental edge.
_DWELL_MS = {"desktop": 1000, "tui": 1, "native": 1000}

#: Read provenance. There is no ``watching`` value: a new item is ALWAYS unread.
#: ``in_view`` requires a per-item anchor/revision receipt with a real input
#: edge; ``explicit`` is the operator's own gesture. The old presence evaluator
#: was deleted (N3), rather than left available for a future accidental re-wire.
READ_VIA = ("explicit", "in_view")

#: Wire size caps (contract §1.3). The tool's pydantic schema enforces the same
#: numbers; the store re-applies them so a second writer cannot exceed them.
TITLE_MAX = 120
BODY_MAX = 4000
LINKS_MAX = 5
LINK_LABEL_MAX = 80
LINK_URL_MAX = 500
KEY_MAX = 120
EXCERPT_MAX = 160

#: New rows per session per rolling hour. Keyed updates do not count. Over the
#: cap the write is REFUSED (never dropped) so the model learns to pass a key.
NEW_ITEMS_PER_HOUR = 30
_HOUR_MS = 3600 * 1000

#: Earlier versions kept per item for the "Earlier updates" trail.
HISTORY_CAP = 20

#: Retention (contract §1.4). Unread rows are never age-pruned: an unread item
#: is exactly the thing the operator has not seen yet.
#: Banner cooldown per item (contract §1.3, D15). A watcher that ticks every few
#: minutes must not raise a banner per tick: inside this window a keyed update
#: still bumps the row, its revision and the badge, and raises nothing. EVERY
#: banner rung honours it through ``claim_delivery``, so the desktop feed, the
#: runtime ladder and the TUI cannot disagree about what already spoke.
BANNER_COOLDOWN_MS = 30 * 60 * 1000

DISMISSED_TTL_MS = 14 * 24 * _HOUR_MS
READ_TTL_MS = 30 * 24 * _HOUR_MS
ROW_CAP = 10_000
PRUNE_INTERVAL_MS = _HOUR_MS

#: How fresh an input edge must be to count as "the operator is here", in ms.
#: The TUI's ``ATTENTION_INPUT_EVIDENCE_S`` (``tui/attention.py``), restated as a
#: literal because importing the TUI from a stdlib-only store is the layering
#: this module exists to keep; ``tests/unit/notifications`` pins the equality.
INPUT_EVIDENCE_MS = 120_000

#: The ref prefix for an open ask's reel-side read mark.
ASK_REF_PREFIX = "ask:"

_T = TypeVar("_T")


def store_path(config_dir: Path | str) -> Path:
    """``<config_dir>/notifications.db``."""
    return Path(config_dir) / NOTIFICATIONS_DB_NAME


def now_ms() -> int:
    return int(time.time() * 1000)


def new_item_id() -> str:
    """``n_`` + 16 lowercase base32 chars (80 random bits)."""
    return "n_" + base64.b32encode(secrets.token_bytes(10)).decode("ascii").lower()


def is_item_id(value: str) -> bool:
    return isinstance(value, str) and value.startswith("n_") and len(value) == 18


def ask_ref(session_id: str, ask_id: str) -> str:
    return f"{ASK_REF_PREFIX}{session_id}:{ask_id}"


def parse_ask_ref(value: str) -> tuple[str, str] | None:
    if not isinstance(value, str) or not value.startswith(ASK_REF_PREFIX):
        return None
    session_id, _, ask_id = value[len(ASK_REF_PREFIX) :].partition(":")
    if not session_id or not ask_id:
        return None
    return session_id, ask_id


def excerpt(text: str, limit: int = EXCERPT_MAX) -> str:
    """The first ``limit`` chars of ``text``, whitespace collapsed to one line."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def priority_rank(priority: str) -> int:
    return _PRIORITY_RANK.get(priority, _PRIORITY_RANK["normal"])


def content_hash(title: str, body_md: str, links_json: str, priority: str) -> str:
    joined = "\x1f".join((title, body_md, links_json, priority))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def normalize_links(links: Iterable[Mapping[str, Any]] | None) -> list[dict[str, str]]:
    """``[{label, url}]``, http(s) only, capped. A bad entry is dropped, not fatal.

    The tool's schema refuses a malformed link before it reaches here, so this
    is the second writer's guard rather than the model's feedback path.
    """
    out: list[dict[str, str]] = []
    for raw in links or ():
        if not isinstance(raw, Mapping):
            continue
        url = str(raw.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")) or len(url) > LINK_URL_MAX:
            continue
        label = str(raw.get("label") or "").strip()[:LINK_LABEL_MAX] or url[:LINK_LABEL_MAX]
        out.append({"label": label, "url": url})
        if len(out) >= LINKS_MAX:
            break
    return out


class NotificationStoreBusy(sqlite3.OperationalError):
    """A store operation gave up on contention, still classified as contention.

    Subclasses ``sqlite3.OperationalError`` and carries SQLite's certified
    ``sqlite_errorname`` for the reason ``attention.AttentionWriteDeferred``
    does: ``session/store_failures.py`` maps that code to the 503-busy sentence,
    so every surface ladder keeps classifying this as "busy, retry" rather than
    "the store is broken".
    """

    def __init__(self, message: str, *, error: sqlite3.OperationalError | None = None) -> None:
        super().__init__(message)
        code = getattr(error, "sqlite_errorcode", None)
        self.sqlite_errorcode = int(code) if code else sqlite3.SQLITE_BUSY
        name = str(getattr(error, "sqlite_errorname", "") or "")
        self.sqlite_errorname = name or "SQLITE_BUSY"


class NotifyRefused(Exception):
    """The write was refused for a reason the model can act on (the hourly cap)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


#: The corruption family: a file whose bytes are not a database.
_CORRUPT_ERRONAMES = frozenset({"SQLITE_CORRUPT", "SQLITE_NOTADB"})


def _is_corrupt(error: sqlite3.Error) -> bool:
    """True when SQLite says this file is not (or can no longer be) a database.

    THE ONLY CLASS THE STORE HEALS ITSELF. A damaged file is a condition no
    retry can improve and no reader can work around, and the operator cannot be
    asked to delete a database they never created. Every other failure keeps its
    own answer: contention is retried, and an unopenable store (permissions, a
    full disk) is reported, because moving a healthy file aside would destroy
    the data the operator is looking for. The certified errornames come first;
    the message checks are the fallback for paths that raise a bare
    ``DatabaseError`` (measured on 3.12: opening garbage makes
    ``PRAGMA user_version`` raise ``DatabaseError: file is not a database``
    with no errorname).
    """
    name = str(getattr(error, "sqlite_errorname", "") or "")
    if name:
        return name in _CORRUPT_ERRONAMES
    message = str(error).lower()
    return "file is not a database" in message or "malformed" in message


def _is_contention(error: sqlite3.OperationalError) -> bool:
    name = str(getattr(error, "sqlite_errorname", "") or "")
    if name:
        return name in BUSY_ERRONAMES
    message = str(error).lower()
    return "locked" in message or "busy" in message


@dataclass(frozen=True)
class NotifyOutcome:
    """What one ``notify`` call did to the store.

    ``action`` is one of ``created`` (a new row), ``unchanged`` (keyed, same
    content: no write), ``updated`` (keyed, content changed: new revision, row
    unread again), ``absorbed`` (keyed, the operator had dismissed it: content
    recorded silently, row stays dismissed) and ``revived`` (keyed, dismissed,
    but the priority rose above the stored one: un-dismissed and unread).
    """

    action: str
    item_id: str
    revision: int
    repeat_count: int
    read: bool
    priority: str
    seq: int


def _visible_for_row(
    entry: Mapping[str, Any], batch_visible: int, input_at_ms: int, updated_at: int
) -> bool:
    """Was THIS item's anchor visible after its last change, before the input?

    CD5/D13: the batch-level field let item B ride item A's visibility — both
    were read although nothing proved B's anchor had been painted since B's
    update. An item that reports its own ``visible_since_ms`` is judged on that
    value alone; one that does not keeps the older batch semantics, so a mixed
    batch is judged per row rather than by its most generous member.
    """
    visible = entry.get("visible_since_ms")
    stamp = int(visible) if visible is not None else int(batch_visible)
    return 0 < stamp <= input_at_ms and stamp >= updated_at


@dataclass
class ApplyResult:
    changed: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS items (
      id            TEXT PRIMARY KEY,
      seq           INTEGER NOT NULL UNIQUE,
      session_id    TEXT NOT NULL,
      session_title TEXT NOT NULL DEFAULT '',
      agent_id      TEXT NOT NULL DEFAULT '',
      team_id       TEXT NOT NULL DEFAULT '',
      priority      TEXT NOT NULL CHECK (priority IN ('high','normal','low')),
      title         TEXT NOT NULL,
      body_md       TEXT NOT NULL DEFAULT '',
      links_json    TEXT NOT NULL DEFAULT '[]',
      key           TEXT NOT NULL DEFAULT '',
      content_hash  TEXT NOT NULL,
      revision      INTEGER NOT NULL DEFAULT 1,
      repeat_count  INTEGER NOT NULL DEFAULT 1,
      anchor_id     TEXT NOT NULL DEFAULT '',
      created_at    INTEGER NOT NULL,
      updated_at    INTEGER NOT NULL,
      read_rev      INTEGER NOT NULL DEFAULT 0,
      read_at       INTEGER,
      read_via      TEXT,
      read_surface  TEXT,
      dismissed_at  INTEGER,
      dismissed_via TEXT,
      delivered_rev INTEGER NOT NULL DEFAULT 0,
      delivered_by  TEXT,
      delivered_at  INTEGER
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS items_key ON items(session_id, key) WHERE key <> ''",
    # LISTED BY CONTENT TIME, NOT BY ``seq``. ``seq`` is re-stamped on every
    # write (read, dismiss, claim) so the doorbell and ``through_seq`` see every
    # change; ordering the reel by it would move a row to the top of "All" the
    # moment the operator read it. ``updated_at`` moves only with content.
    "CREATE INDEX IF NOT EXISTS items_list ON items(dismissed_at, updated_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS items_session ON items(session_id, updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS item_history (
      item_id  TEXT NOT NULL,
      revision INTEGER NOT NULL,
      at       INTEGER NOT NULL,
      title    TEXT NOT NULL,
      excerpt  TEXT NOT NULL,
      PRIMARY KEY (item_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ref_state (
      ref     TEXT PRIMARY KEY,
      read_at INTEGER,
      seq     INTEGER NOT NULL
    )
    """,
    "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)",
)

_ITEM_COLUMNS = (
    "id, seq, session_id, session_title, agent_id, team_id, priority, title, body_md, "
    "links_json, key, content_hash, revision, repeat_count, anchor_id, created_at, "
    "updated_at, read_rev, read_at, read_via, read_surface, dismissed_at, dismissed_via, "
    "delivered_rev, delivered_by, delivered_at"
)


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    try:
        links = json.loads(data.get("links_json") or "[]")
    except ValueError:
        links = []
    data["links"] = links if isinstance(links, list) else []
    data["read"] = int(data.get("read_rev") or 0) >= int(data.get("revision") or 1)
    data["dismissed"] = data.get("dismissed_at") is not None
    return data


def _encode_cursor(updated_at: int, item_id: str) -> str:
    return f"{int(updated_at)}.{item_id}"


#: The largest integer SQLite can bind (it stores integers as int64). A cursor
#: stamp above this can only come from a corrupted or hostile client, and
#: handing it to sqlite raises ``OverflowError`` from deep inside a query.
CURSOR_STAMP_MAX = 2**63 - 1


def _decode_cursor(cursor: str | None) -> tuple[int, str] | None:
    """The opaque list cursor. Malformed is "from the top", never an error.

    OUT OF RANGE IS MALFORMED (QA round 2): the wire refuses such a stamp with
    a 422 (``items.cursor_in_range``), and this is the second line of defence
    for every other caller — a value sqlite cannot bind must never reach the
    query, because ``OverflowError`` there is not a store failure and would
    surface as an unclassified 500.
    """
    if not cursor:
        return None
    stamp, _, item_id = str(cursor).partition(".")
    try:
        value = int(stamp)
    except ValueError:
        return None
    if not 0 <= value <= CURSOR_STAMP_MAX:
        return None
    return value, item_id


class NotificationStore:
    """One config root's ``notifications.db``.

    Every method is synchronous and short; callers on an event loop run them in
    a worker thread (``asyncio.to_thread``), the ``AttentionStore`` contract.
    """

    def __init__(self, config_dir: Path | str) -> None:
        self.config_dir = Path(config_dir)
        self.path = store_path(config_dir)

    # -- connections ------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """A write connection, creating and migrating the schema when needed.

        A file SQLite cannot read at all is QUARANTINED and replaced (CD1): the
        inbox is an optional store, and one damaged file must not turn every
        ``notify`` call on this machine into a permanent refusal with no remedy
        the operator could carry out. Only the write path heals, because only a
        writer may create a file; readers still report the failure so a route can
        say "unknown" instead of silently answering zero (measured: the desktop
        feed, the relay and the tool all reached a corrupt store).
        """
        try:
            return self._connect_once()
        except sqlite3.DatabaseError as error:
            if not _is_corrupt(error):
                raise
            moved = self._quarantine()
            logger.warning(
                "notifications: store was unreadable (%s); %s and a fresh store is being created",
                error,
                f"moved to {moved.name}" if moved is not None else "could not be moved aside",
            )
            return self._connect_once()

    def _quarantine(self) -> Path | None:
        """Move a corrupt store (and its WAL sidecars) aside for diagnosis.

        The sidecars go with the database they belong to: a ``-wal`` left beside
        a fresh file would be applied to it. Best-effort — a file we cannot move
        still gets the recreate attempt, and the caller logs which happened.
        """
        stamp = int(time.time())
        moved: Path | None = None
        for suffix in ("", "-wal", "-shm"):
            source = self.path.with_name(self.path.name + suffix) if suffix else self.path
            if not source.exists():
                continue
            target = self.path.with_name(f"{self.path.name}{suffix}.corrupt-{stamp}")
            try:
                os.replace(source, target)
            except OSError:
                continue
            if suffix == "":
                moved = target
        return moved

    def _connect_once(self) -> sqlite3.Connection:
        """One connection attempt: create, migrate, adopt WAL."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            # Private before SQLite writes anything: item bodies are the
            # operator's own findings (metrics, PR names, abuse attempts).
            self.path.touch(mode=0o600)
        conn = sqlite3.connect(self.path, timeout=_CONNECT_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if int(conn.execute("PRAGMA user_version").fetchone()[0]) < SCHEMA_VERSION:
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    # Re-checked under the lock: two processes racing a fresh
                    # file both see version 0, and only the winner creates.
                    if int(conn.execute("PRAGMA user_version").fetchone()[0]) < SCHEMA_VERSION:
                        for statement in _SCHEMA:
                            conn.execute(statement)
                        conn.execute("INSERT OR IGNORE INTO meta(k, v) VALUES ('next_seq', '1')")
                        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._adopt_wal(conn)
            return conn
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def _adopt_wal(conn: sqlite3.Connection) -> None:
        """Best-effort WAL; a store that cannot convert keeps working as is."""
        conn.execute(f"PRAGMA busy_timeout={_WAL_BUSY_TIMEOUT_MS}")
        try:
            conn.execute("PRAGMA journal_mode=WAL").fetchone()
        except sqlite3.Error as error:
            logger.debug("notifications: WAL conversion deferred: %s", error)
        finally:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")

    def _connect_read_only(self) -> sqlite3.Connection | None:
        """A ``mode=ro`` connection, or ``None`` when the store does not exist.

        Readers never create the file: an absent store is the ordinary state of
        a machine where no agent has notified yet, and a reader that created it
        would move the doorbell it is about to be woken by.
        """
        if not self.path.exists():
            return None
        conn = sqlite3.connect(
            f"{self.path.as_uri()}?mode=ro", uri=True, timeout=_CONNECT_TIMEOUT_S
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if int(conn.execute("PRAGMA user_version").fetchone()[0]) < 1:
                # A placeholder a first writer has not populated yet.
                conn.close()
                return None
        except BaseException:
            conn.close()
            raise
        return conn

    def _retry(self, operation: Callable[[], _T], what: str) -> _T:
        last: sqlite3.OperationalError | None = None
        for attempt in range(_CONTENTION_ATTEMPTS):
            try:
                return operation()
            except sqlite3.OperationalError as error:
                if not _is_contention(error):
                    raise
                last = error
                if attempt + 1 < _CONTENTION_ATTEMPTS:
                    time.sleep(_CONTENTION_BACKOFF_S)
        assert last is not None
        raise NotificationStoreBusy(
            f"notifications store stayed busy through {_CONTENTION_ATTEMPTS} attempts "
            f"({what}): {last}",
            error=last,
        ) from last

    def _write(self, body: Callable[[sqlite3.Connection], _T], what: str) -> _T:
        def run() -> _T:
            with closing(self._connect()) as conn:
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    return body(conn)

        return self._retry(run, what)

    def _read(self, body: Callable[[sqlite3.Connection], _T], empty: _T, what: str) -> _T:
        def run() -> _T:
            conn = self._connect_read_only()
            if conn is None:
                return empty
            with closing(conn):
                return body(conn)

        return self._retry(run, what)

    @staticmethod
    def _next_seq(conn: sqlite3.Connection) -> int:
        """Stamp the next machine-wide sequence inside the caller's transaction."""
        row = conn.execute("SELECT v FROM meta WHERE k='next_seq'").fetchone()
        seq = int(row[0]) if row is not None else 1
        conn.execute(
            "INSERT INTO meta(k, v) VALUES ('next_seq', ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (str(seq + 1),),
        )
        return seq

    # -- the notify write -------------------------------------------------

    def notify(
        self,
        session_id: str,
        *,
        title: str,
        body_md: str = "",
        priority: str = "normal",
        key: str = "",
        links: Iterable[Mapping[str, Any]] | None = None,
        anchor_id: str = "",
        session_title: str = "",
        agent_id: str = "",
        team_id: str = "",
        now: int | None = None,
    ) -> NotifyOutcome:
        """Create or update one item (collapse table: contract §1.3).

        Every changed revision starts unread: session presence cannot prove
        the new callout rendered. Only a subsequent per-item receipt or an
        explicit operator read gesture may acknowledge it.
        Raises :class:`NotifyRefused` over the hourly new-item cap, and
        :class:`NotificationStoreBusy` when contention outlasts the budget.
        """
        if priority not in _PRIORITY_RANK:
            raise ValueError(f"priority must be one of {', '.join(PRIORITIES)}")
        title = " ".join(str(title or "").split())[:TITLE_MAX]
        if not title:
            raise ValueError("title must not be empty")
        body_md = str(body_md or "")[:BODY_MAX]
        key = str(key or "").strip()[:KEY_MAX]
        links_json = json.dumps(normalize_links(links), separators=(",", ":"), ensure_ascii=False)
        digest = content_hash(title, body_md, links_json, priority)
        stamp = now if now is not None else now_ms()

        def body(conn: sqlite3.Connection) -> NotifyOutcome:
            self._maybe_prune(conn, stamp)
            existing = None
            if key:
                existing = conn.execute(
                    f"SELECT {_ITEM_COLUMNS} FROM items WHERE session_id=? AND key=?",
                    (session_id, key),
                ).fetchone()
            if existing is None:
                recent = conn.execute(
                    "SELECT COUNT(*) FROM items WHERE session_id=? AND created_at>?",
                    (session_id, stamp - _HOUR_MS),
                ).fetchone()[0]
                if int(recent) >= NEW_ITEMS_PER_HOUR:
                    raise NotifyRefused(
                        "rate_limited",
                        f"refused: this conversation already raised {NEW_ITEMS_PER_HOUR} new "
                        "items in the last hour. Pass a `key` so repeats update one item "
                        "instead of adding another.",
                    )
                item_id = new_item_id()
                seq = self._next_seq(conn)
                read = False
                conn.execute(
                    "INSERT INTO items (id, seq, session_id, session_title, agent_id, team_id, "
                    "priority, title, body_md, links_json, key, content_hash, revision, "
                    "repeat_count, anchor_id, created_at, updated_at, read_rev, read_at, read_via) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?, ?)",
                    (
                        item_id,
                        seq,
                        session_id,
                        session_title,
                        agent_id,
                        team_id,
                        priority,
                        title,
                        body_md,
                        links_json,
                        key,
                        digest,
                        anchor_id,
                        stamp,
                        stamp,
                        1 if read else 0,
                        stamp if read else None,
                        None,
                    ),
                )
                return NotifyOutcome("created", item_id, 1, 1, read, priority, seq)

            current = _row_dict(existing)
            item_id = str(current["id"])
            dismissed = current["dismissed"]
            if current["content_hash"] == digest:
                return NotifyOutcome(
                    "absorbed" if dismissed else "unchanged",
                    item_id,
                    int(current["revision"]),
                    int(current["repeat_count"]),
                    bool(current["read"]),
                    str(current["priority"]),
                    int(current["seq"]),
                )
            revision = int(current["revision"]) + 1
            repeat = int(current["repeat_count"]) + 1
            escalated = priority_rank(priority) > priority_rank(str(current["priority"]))
            revive = dismissed and escalated
            stays_dismissed = dismissed and not revive
            conn.execute(
                "INSERT OR REPLACE INTO item_history (item_id, revision, at, title, excerpt) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    item_id,
                    int(current["revision"]),
                    int(current["updated_at"]),
                    str(current["title"]),
                    excerpt(str(current["body_md"])),
                ),
            )
            conn.execute(
                "DELETE FROM item_history WHERE item_id=? AND revision NOT IN ("
                "SELECT revision FROM item_history WHERE item_id=? ORDER BY revision DESC LIMIT ?)",
                (item_id, item_id, HISTORY_CAP),
            )
            seq = self._next_seq(conn)
            read = False
            conn.execute(
                "UPDATE items SET seq=?, priority=?, title=?, body_md=?, links_json=?, "
                "content_hash=?, revision=?, repeat_count=?, anchor_id=?, updated_at=?, "
                "read_rev=?, read_at=?, read_via=?, dismissed_at=?, dismissed_via=? WHERE id=?",
                (
                    seq,
                    priority,
                    title,
                    body_md,
                    links_json,
                    digest,
                    revision,
                    repeat,
                    anchor_id or str(current["anchor_id"]),
                    stamp,
                    revision if read else int(current["read_rev"]),
                    stamp if read else current["read_at"],
                    current["read_via"],
                    current["dismissed_at"] if stays_dismissed else None,
                    current["dismissed_via"] if stays_dismissed else None,
                    item_id,
                ),
            )
            action = "absorbed" if stays_dismissed else ("revived" if revive else "updated")
            return NotifyOutcome(action, item_id, revision, repeat, read, priority, seq)

        return self._write(body, "notify")

    # -- reads --------------------------------------------------------------

    def list_items(
        self,
        *,
        state: str = "unread",
        priority: str | None = None,
        session_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        through_seq: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One page of items, newest content first, and the next page's cursor.

        ``state``: ``unread`` (not dismissed, not read), ``all`` (not dismissed)
        or ``dismissed``.
        """
        if state not in ("unread", "all", "dismissed"):
            raise ValueError("state must be unread, all or dismissed")
        limit = max(1, min(int(limit), 100))
        clauses = ["dismissed_at IS NOT NULL" if state == "dismissed" else "dismissed_at IS NULL"]
        params: list[Any] = []
        if through_seq is not None:
            clauses.append("seq <= ?")
            params.append(through_seq)
        if state == "unread":
            clauses.append("read_rev < revision")
        if priority:
            clauses.append("priority = ?")
            params.append(priority)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        decoded = _decode_cursor(cursor)
        if decoded is not None:
            clauses.append("(updated_at < ? OR (updated_at = ? AND id < ?))")
            params.extend([decoded[0], decoded[0], decoded[1]])
        sql = (
            f"SELECT {_ITEM_COLUMNS} FROM items WHERE {' AND '.join(clauses)} "
            "ORDER BY updated_at DESC, id DESC LIMIT ?"
        )
        params.append(limit + 1)

        def body(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], str | None]:
            rows = [_row_dict(row) for row in conn.execute(sql, params)]
            next_cursor = None
            if len(rows) > limit:
                rows = rows[:limit]
                last = rows[-1]
                next_cursor = _encode_cursor(int(last["updated_at"]), str(last["id"]))
            return rows, next_cursor

        return self._read(body, ([], None), "list")

    def session_ids(self, ids: Sequence[str]) -> dict[str, str]:
        """One read for a batch of ids: ``{id: session_id}`` (CD6).

        A bulk gesture echoes up to 500 observed tokens, and asking the store
        one row at a time opened up to 500 read-only connections per "dismiss
        all from this conversation" — on the same store the feed stats every
        100 ms. One ``WHERE id IN (…)`` statement answers the one question the
        caller has (which session owns this id), and an id absent from the
        answer is a row this store does not own.
        """
        wanted = [str(one) for one in ids if one]
        if not wanted:
            return {}
        placeholders = ",".join("?" for _ in wanted)

        def body(conn: sqlite3.Connection) -> dict[str, str]:
            return {
                str(row["id"]): str(row["session_id"])
                for row in conn.execute(
                    f"SELECT id, session_id FROM items WHERE id IN ({placeholders})", wanted
                )
            }

        return self._read(body, {}, "session_ids")

    def get(self, item_id: str) -> dict[str, Any] | None:
        def body(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM items WHERE id=?", (item_id,)
            ).fetchone()
            return _row_dict(row) if row is not None else None

        return self._read(body, None, "get")

    def history(self, item_id: str) -> list[dict[str, Any]]:
        """Earlier versions of one item, newest first (at most :data:`HISTORY_CAP`)."""

        def body(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT revision, at, title, excerpt FROM item_history WHERE item_id=? "
                    "ORDER BY revision DESC",
                    (item_id,),
                )
            ]

        return self._read(body, [], "history")

    def counts(self) -> tuple[int, int]:
        """``(unread_count, high_water)`` from two indexed reads.

        ``high_water`` is the last ``seq`` stamped (rows and ref marks share the
        counter), so it moves on every write including reads and dismisses —
        which is what a client bounding "mark all read" by it needs.
        """

        def body(conn: sqlite3.Connection) -> tuple[int, int]:
            unread = conn.execute(
                "SELECT COUNT(*) FROM items WHERE dismissed_at IS NULL AND read_rev < revision"
            ).fetchone()[0]
            row = conn.execute("SELECT v FROM meta WHERE k='next_seq'").fetchone()
            return int(unread), (int(row[0]) - 1 if row is not None else 0)

        return self._read(body, (0, 0), "counts")

    def ref_states(self, refs: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Reel-side read marks for refs this store does not own (open asks)."""
        wanted = [ref for ref in refs if isinstance(ref, str)]
        if not wanted:
            return {}

        def body(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            # Chunked under SQLite's default variable limit.
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                marks = ",".join("?" * len(chunk))
                for row in conn.execute(
                    f"SELECT ref, read_at, seq FROM ref_state WHERE ref IN ({marks})", chunk
                ):
                    out[str(row["ref"])] = dict(row)
            return out

        return self._read(body, {}, "ref_states")

    # -- operator writes ---------------------------------------------------

    def apply(
        self,
        op: str,
        *,
        ids: Sequence[tuple[str, int]] | None = None,
        session_id: str | None = None,
        through_seq: int | None = None,
        all_items: bool = False,
        via: str = "explicit",
        now: int | None = None,
    ) -> ApplyResult:
        """Read / unread / dismiss / undismiss notify items. Token-bound (R10).

        Exactly one target: ``ids`` (each with the ``revision`` the caller
        rendered), or ``session_id`` + ``through_seq``, or ``all_items`` +
        ``through_seq``. Ask refs are NOT handled here: the route resolves them
        (an open ask is never dismissed through the reel) and calls
        :meth:`mark_refs` for their read marks.
        """
        if op not in ("read", "unread", "dismiss", "undismiss"):
            raise ValueError("op must be read, unread, dismiss or undismiss")
        targets = sum((ids is not None, session_id is not None, bool(all_items)))
        if targets != 1:
            raise ValueError("name exactly one target: ids, session_id or all")
        if ids is None and through_seq is None:
            raise ValueError("a by-session or all-items write needs through_seq")
        stamp = now if now is not None else now_ms()
        result = ApplyResult()

        def changes(row: Mapping[str, Any]) -> bool:
            read = int(row["read_rev"]) >= int(row["revision"])
            dismissed = row["dismissed_at"] is not None
            return {
                "read": not read,
                "unread": read,
                "dismiss": not dismissed,
                "undismiss": dismissed,
            }[op]

        def write(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
            seq = self._next_seq(conn)
            if op == "read":
                conn.execute(
                    "UPDATE items SET seq=?, read_rev=revision, read_at=?, read_via=? WHERE id=?",
                    (seq, stamp, via, row["id"]),
                )
            elif op == "unread":
                conn.execute(
                    "UPDATE items SET seq=?, read_rev=0, read_at=NULL, read_via=NULL, "
                    "read_surface=NULL WHERE id=?",
                    (seq, row["id"]),
                )
            elif op == "dismiss":
                conn.execute(
                    "UPDATE items SET seq=?, dismissed_at=?, dismissed_via=? WHERE id=?",
                    (seq, stamp, via, row["id"]),
                )
            else:
                conn.execute(
                    "UPDATE items SET seq=?, dismissed_at=NULL, dismissed_via=NULL WHERE id=?",
                    (seq, row["id"]),
                )
            result.changed.append(str(row["id"]))

        def body(conn: sqlite3.Connection) -> ApplyResult:
            if ids is not None:
                for item_id, revision in ids:
                    row = conn.execute(
                        "SELECT id, revision, seq, read_rev, dismissed_at FROM items WHERE id=?",
                        (item_id,),
                    ).fetchone()
                    if row is None:
                        result.skipped.append({"id": item_id, "reason": "unknown"})
                    elif int(row["revision"]) != int(revision):
                        result.skipped.append({"id": item_id, "reason": "stale_revision"})
                    elif through_seq is not None and int(row["seq"]) > through_seq:
                        result.skipped.append({"id": item_id, "reason": "stale_revision"})
                    elif not changes(row):
                        result.skipped.append({"id": item_id, "reason": "already"})
                    else:
                        write(conn, row)
                return result
            clauses = ["seq <= ?"]
            params: list[Any] = [int(through_seq or 0)]
            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)
            for row in conn.execute(
                "SELECT id, revision, read_rev, dismissed_at FROM items WHERE "
                + " AND ".join(clauses)
                + " ORDER BY seq",
                params,
            ).fetchall():
                if changes(row):
                    write(conn, row)
            return result

        return self._write(body, f"apply:{op}")

    def mark_refs(self, refs: Sequence[str], *, read: bool, now: int | None = None) -> list[str]:
        """Set or clear the reel-side read mark of refs (open asks). Never touches the ask."""
        stamp = now if now is not None else now_ms()

        def body(conn: sqlite3.Connection) -> list[str]:
            changed: list[str] = []
            for ref in refs:
                row = conn.execute("SELECT read_at FROM ref_state WHERE ref=?", (ref,)).fetchone()
                is_read = row is not None and row["read_at"] is not None
                if is_read == read:
                    continue
                seq = self._next_seq(conn)
                conn.execute(
                    "INSERT INTO ref_state(ref, read_at, seq) VALUES (?, ?, ?) "
                    "ON CONFLICT(ref) DO UPDATE SET read_at=excluded.read_at, seq=excluded.seq",
                    (ref, stamp if read else None, seq),
                )
                changed.append(ref)
            return changed

        return self._write(body, "mark_refs")

    def seen(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        session_id: str,
        input_at_ms: int,
        visible_since_ms: int = 0,
        surface: str = "",
        now: int | None = None,
    ) -> ApplyResult:
        """In-view auto-read: per item, token-bound, evidence re-checked here.

        Each item must belong to ``session_id``, name its CURRENT revision and
        its anchor, and the input edge must be no older than
        :data:`INPUT_EVIDENCE_MS` and no earlier than the item's last content
        change. Anything else lands in ``skipped`` with its reason. This is the
        per-result rule ``docs/ATTENTION.md`` already states for completions,
        applied to items; it is never a sweep.
        """
        stamp = now if now is not None else now_ms()
        result = ApplyResult()
        # THE EDGE is batch-level: a trusted input kind on a known surface, an
        # input time in the past but fresh, and the dwell the surface needs
        # (the desktop waits one second while the anchor stays visible).
        dwell_ms = _DWELL_MS.get(surface, 0)
        edge_fresh = (
            surface in SEEN_SURFACES
            and dwell_ms
            and 0 < input_at_ms <= stamp
            and dwell_ms <= stamp - input_at_ms <= INPUT_EVIDENCE_MS
        )

        def body(conn: sqlite3.Connection) -> ApplyResult:
            for entry in items:
                item_id = str(entry.get("id") or "")
                row = conn.execute(
                    "SELECT id, session_id, revision, read_rev, anchor_id, updated_at, "
                    "dismissed_at FROM items WHERE id=?",
                    (item_id,),
                ).fetchone()
                if row is None or str(row["session_id"]) != session_id:
                    result.skipped.append({"id": item_id, "reason": "unknown"})
                elif int(row["revision"]) != int(entry.get("revision") or 0):
                    result.skipped.append({"id": item_id, "reason": "stale_revision"})
                elif (
                    str(entry.get("anchor_id") or "") != str(row["anchor_id"] or "")
                    or not row["anchor_id"]
                ):
                    result.skipped.append({"id": item_id, "reason": "anchor_mismatch"})
                elif not edge_fresh or not _visible_for_row(
                    entry, visible_since_ms, input_at_ms, int(row["updated_at"])
                ):
                    result.skipped.append({"id": item_id, "reason": "stale_evidence"})
                elif int(row["read_rev"]) >= int(row["revision"]):
                    result.skipped.append({"id": item_id, "reason": "already"})
                else:
                    seq = self._next_seq(conn)
                    conn.execute(
                        "UPDATE items SET seq=?, read_rev=revision, read_at=?, read_via='in_view', "
                        "read_surface=? "
                        "WHERE id=?",
                        (seq, stamp, surface, item_id),
                    )
                    result.changed.append(item_id)
            return result

        return self._write(body, "seen")

    def undelivered(self, *, priority: str | None = None, limit: int = 4) -> list[dict[str, Any]]:
        """Items a banner leg may still raise: unread, undismissed, unclaimed.

        THE FILTER IS SQL, not a post-read pass: a caller that fetched N unread
        rows and filtered `delivered_rev` afterwards could starve the oldest
        item for ever once the newest N had been claimed, which is the one
        failure a drain-by-ticks design has to avoid.
        """
        clauses = ["dismissed_at IS NULL", "read_rev < revision", "delivered_rev < revision"]
        params: list[Any] = []
        if priority is not None:
            clauses.append("priority = ?")
            params.append(str(priority))
        params.append(max(1, int(limit)))

        def body(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            return [
                _row_dict(row)
                for row in conn.execute(
                    f"SELECT {_ITEM_COLUMNS} FROM items WHERE {' AND '.join(clauses)} "
                    "ORDER BY created_at, id LIMIT ?",
                    params,
                )
            ]

        return self._read(body, [], "undelivered")

    def claim_delivery(self, item_id: str, by: str, *, now: int | None = None) -> bool:
        """Claim the banner for an item's current revision, once, machine-wide.

        ``delivered_rev`` is the watermark a banner leg (desktop feed, TUI, the
        runtime's own OS banner, and a future phone push) claims before raising,
        so one revision raises at most one banner. Separate from the read mark
        on purpose: notifying is not reading (``attention.py``'s two-watermark
        rule).

        THE READ STATE IS PART OF THE CLAIM. A receipt that lands before the
        raise (the feed claims on its next tick) means the surface already
        showed the operator this content, so a banner would be redundant — the
        one suppression the grace allows. It suppresses nothing permanently: a
        later content change bumps ``revision``, which makes the row unread and
        claimable again (``notifications/delivery.py`` states the rule).
        """

        stamp = now if now is not None else now_ms()

        def body(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT revision, delivered_rev, read_rev, dismissed_at, delivered_at "
                "FROM items WHERE id=?",
                (item_id,),
            ).fetchone()
            if row is None or int(row["delivered_rev"]) >= int(row["revision"]):
                return False
            # THE COOLDOWN (contract §1.3, D15), checked before any other arm so
            # it covers every rung: a high-priority watcher updating every five
            # minutes used to banner twelve times an hour for one item. The row
            # stays unread and badged; the SECOND update inside the window is
            # simply not news a banner can carry.
            last = row["delivered_at"]
            if last is not None and stamp - int(last) < BANNER_COOLDOWN_MS:
                return False
            if int(row["read_rev"]) >= int(row["revision"]):
                return False
            # DISMISSED IS NOT CLAIMABLE (round-1 R1): a keyed update to a
            # dismissed row is absorbed "silently … no banner" (contract §1.3);
            # `undelivered` already filters it, and the claim must agree with
            # it, or a direct caller re-banners what the operator dismissed.
            if row["dismissed_at"] is not None:
                return False
            # NO ``seq`` ADVANCE (D14). A claim changes nothing a list renders:
            # it is a delivery watermark. Moving ``seq`` made an UNCHANGED high
            # item "stale" to an in-flight bulk confirm, so "mark all read"
            # skipped exactly the item that had just bannered.
            conn.execute(
                "UPDATE items SET delivered_rev=revision, delivered_by=?, delivered_at=? "
                "WHERE id=?",
                (by, stamp, item_id),
            )
            return True

        return self._write(body, "claim")

    def release_delivery(self, item_id: str, revision: int, by: str) -> bool:
        """Give back a claim whose raise never happened (a spawn that reported nothing).

        Scoped to the exact revision AND claimant: a stale release must not
        wipe a newer revision's claim, and nothing may undo another surface's
        banner. True when a claim was actually released.
        """

        def body(conn: sqlite3.Connection) -> bool:
            # NO ``seq`` ADVANCE (D14), the claim's own rule. ``delivered_at`` is
            # deliberately NOT cleared: a release means the raise produced
            # nothing, and the cooldown clock must still start from the attempt
            # or a failing spawn would retry every second forever.
            released = conn.execute(
                "UPDATE items SET delivered_rev=0, delivered_by='' "
                "WHERE id=? AND delivered_rev=? AND delivered_by=?",
                (item_id, int(revision), by),
            )
            return bool(released.rowcount)

        return self._write(body, "release")

    # -- retention ------------------------------------------------------------

    def delete_session(self, session_id: str) -> int:
        """Drop a deleted conversation's items (``session/cleanup.py``)."""
        if not self.path.exists():
            return 0

        def body(conn: sqlite3.Connection) -> int:
            ids = [
                str(row[0])
                for row in conn.execute("SELECT id FROM items WHERE session_id=?", (session_id,))
            ]
            for item_id in ids:
                conn.execute("DELETE FROM item_history WHERE item_id=?", (item_id,))
            conn.execute("DELETE FROM items WHERE session_id=?", (session_id,))
            conn.execute(
                "DELETE FROM ref_state WHERE ref LIKE ?", (f"{ASK_REF_PREFIX}{session_id}:%",)
            )
            if ids:
                self._next_seq(conn)
            return len(ids)

        return self._write(body, "delete_session")

    def _maybe_prune(self, conn: sqlite3.Connection, stamp: int) -> None:
        """At most once an hour (machine-wide, ``meta.last_prune_at``), in the write's txn."""
        row = conn.execute("SELECT v FROM meta WHERE k='last_prune_at'").fetchone()
        try:
            last = int(row[0]) if row is not None else 0
        except (TypeError, ValueError):
            last = 0
        if stamp - last < PRUNE_INTERVAL_MS:
            return
        conn.execute(
            "INSERT INTO meta(k, v) VALUES ('last_prune_at', ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (str(stamp),),
        )
        self.prune(conn, stamp)

    @staticmethod
    def prune(conn: sqlite3.Connection, stamp: int) -> int:
        """Age out dismissed (14 d) and read (30 d) rows, then enforce the row cap.

        Over the cap, the oldest READ rows go first, low before normal before
        high; unread rows are never deleted by age or by the cap.
        """
        before = conn.total_changes
        # CD7: a DISMISSED row is aged out by its LAST activity, not by the
        # dismiss alone. A watcher the operator dismissed keeps ticking (the
        # fraud/campaign case): every absorbed update moves ``updated_at``, so
        # pruning on ``dismissed_at`` alone eventually dropped the tombstone and
        # the next tick re-created the item as new and unread — undoing exactly
        # the rule that keeps a 5-minute wake out of the reel.
        conn.execute(
            "DELETE FROM items WHERE dismissed_at IS NOT NULL "
            "AND max(dismissed_at, updated_at) < ?",
            (stamp - DISMISSED_TTL_MS,),
        )
        conn.execute(
            "DELETE FROM items WHERE dismissed_at IS NULL AND read_rev >= revision "
            "AND COALESCE(read_at, updated_at) < ?",
            (stamp - READ_TTL_MS,),
        )
        total = int(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])
        excess = total - ROW_CAP
        if excess > 0:
            conn.execute(
                "DELETE FROM items WHERE id IN (SELECT id FROM items "
                "WHERE dismissed_at IS NOT NULL OR read_rev >= revision "
                "ORDER BY CASE priority WHEN 'low' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END, "
                "updated_at LIMIT ?)",
                (excess,),
            )
        conn.execute("DELETE FROM item_history WHERE item_id NOT IN (SELECT id FROM items)")
        return conn.total_changes - before

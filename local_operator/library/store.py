"""Derived Library index tables: schema, SQL query builder and read projections.

WHY A SECOND SET OF TABLES IN THE SAME DATABASE. The catalog was rebuilt in
Python on every list request: 620 session journals scanned, 22,485 ``Asset``
objects constructed, filtered, sorted and counted by iteration. On the
operator's real config that measured 26.5 s cold and 2.8-9.5 s warm, and the
desktop grid polls. Everything here is DERIVED — the ``assets``/``asset_tags``
tables in :mod:`local_operator.library.metadata` stay authoritative and a
``DELETE FROM idx_*`` costs one rebuild, never a lost edit.

WHY THE SAME FILE. Two databases would need a two-phase write to keep a rename
or a tag change visible to search (the FTS table is fed from the effective
name), and a reader would need two connections per query. One file gives the
metadata triggers in :data:`_SCHEMA` atomicity for free: a user rename lands in
``assets``, the trigger rewrites ``idx_assets.effective_name`` and the FTS row
in the same transaction.

WHY FTS5 WITH A STANDALONE TABLE. Search used to be ``q.lower() in haystack``
over every row. The index keeps token-prefix semantics (see
:func:`fts_expression`), which is what a search-as-you-type box needs and what
the old substring match could not give in SQL; prefix matching is an intentional
behaviour change, documented at the function.

``idx_counts`` exists because the sidebar counts used to iterate the whole
catalog to produce six integers. Triggers keep it exact across inserts,
updates, deletes and ``is_final`` transitions, so a listing pays one indexed
primary-key lookup per bucket instead of a scan.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, Protocol

from local_operator.library import classify, identity
from local_operator.library import index as library_index
from local_operator.library.metadata import (
    AUTHORITATIVE_TABLES,
    DB_DIRNAME,
    DB_FILENAME,
    MetadataStore,
    schema_sql,
)

logger = logging.getLogger(__name__)

#: The derived schema. Derived tables only: bumping this discards and rebuilds
#: them, so it is not the metadata store's version and must not be compared to
#: it.
SCHEMA_VERSION: Final = 2

#: Re-exported for the indexer and for ``library doctor``-style callers.
DB_PATH_PARTS: Final = (DB_DIRNAME, DB_FILENAME)

#: The ``idx_state`` keys the service and the indexer agree on.
STATE_SCHEMA: Final = "schema_version"
STATE_REVISION: Final = "revision"
STATE_BACKFILL_DONE: Final = "backfill_done"
STATE_SESSIONS_TOTAL: Final = "sessions_total"
STATE_SESSIONS_SCANNED: Final = "sessions_scanned"
STATE_BUILT_AT: Final = "built_at"
STATE_LAST_RECONCILE: Final = "last_reconcile_at"

#: Count buckets the sidebar asks for, and the scope each listing uses.
COUNT_SCOPES: Final = ("all", "final")
COUNT_BUCKETS: Final = ("category", "kind", "origin_group")

#: Token shape for the FTS query. ``[^\W_]+`` is "letters and digits" in any
#: script, which is the tokenizer's own rule, so a query token can never be
#: split into two MATCH terms by the indexer.
_Q_TOKEN: Final = re.compile(r"[^\W_]+", re.UNICODE)
#: A pasted paragraph is not a search. The cap keeps one request from building a
#: thousand-term MATCH expression (the route already caps the string at 256).
_MAX_Q_TERMS: Final = 16


#: One whole-database rebuild per process at a time. Two requests that both see
#: a broken file must not build two replacements and swap them in some order.
_REBUILD_LOCK: Final = threading.Lock()


class IndexUnavailable(RuntimeError):
    """The derived index cannot be read or written (corrupt, I/O, future schema).

    ``corrupt`` distinguishes the ONE class recovery may answer — damage to the
    file, where discarding the derived tables loses nothing — from transient
    failures (a busy lock, an I/O hiccup) that must NOT trigger a rebuild: the
    index is healthy and a rebuild would throw it away.
    """

    def __init__(self, message: str, *, corrupt: bool = True) -> None:
        super().__init__(message)
        self.corrupt = corrupt


#: ``SQLITE_CORRUPT`` and ``SQLITE_NOTADB``. BUSY (5) and IOERR (10) are
#: deliberately absent: they say "try again", not "this file is damaged".
_CORRUPT_CODES: Final = frozenset({11, 26})


def _is_corruption(error: sqlite3.DatabaseError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if code in _CORRUPT_CODES:
        return True
    if code is not None:
        return False
    # Older SQLite builds do not expose ``sqlite_errorcode``; the message is the
    # only signal left, and only these two phrases mean damage.
    text = str(error).lower()
    return "malformed" in text or "not a database" in text


class ListingQuery(Protocol):
    """The subset of :class:`local_operator.library.service.Query` the SQL reads.

    Structural on purpose: ``store`` must not import ``service`` (the service
    imports this module), and a protocol keeps the filter list in one place
    instead of a converted copy that can silently drop a field.
    """

    # Read-only properties, not attributes: a Protocol with writable
    # attributes is invariant, and the frozen dataclass that satisfies it is not.
    @property
    def q(self) -> str: ...

    @property
    def categories(self) -> frozenset[str]: ...

    @property
    def kinds(self) -> frozenset[str]: ...

    @property
    def origin_groups(self) -> frozenset[str]: ...

    @property
    def origins(self) -> frozenset[str]: ...

    @property
    def tag(self) -> str | None: ...

    @property
    def tag_mode(self) -> str: ...

    @property
    def highlighted(self) -> bool | None: ...

    @property
    def organized(self) -> bool | None: ...

    @property
    def favorite(self) -> bool | None: ...

    @property
    def session(self) -> str | None: ...

    @property
    def session_ids(self) -> frozenset[str]: ...

    @property
    def scope(self) -> str: ...

    @property
    def sort(self) -> str: ...

    @property
    def order(self) -> str | None: ...

    @property
    def effective_order(self) -> str: ...

    @property
    def limit(self) -> int: ...

    @property
    def cursor(self) -> str | None: ...


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _count_statements(*, verb: str, row: str, sign: int) -> list[str]:
    """The ``idx_counts`` statements for one row transition.

    ``verb`` is ``insert`` (add at NEW), ``delete`` (subtract at OLD) or
    ``update`` (both: OLD's buckets leave, NEW's arrive — the only way an
    ``is_final`` transition can move a count between scopes for all three
    buckets without a per-column WHEN clause for each).
    """
    statements: list[str] = []
    for bucket, column in (
        ("category", "category"),
        ("kind", "kind"),
        ("origin_group", "origin_group"),
    ):
        for target_scope in COUNT_SCOPES:
            guard = " AND {row}.is_final = 1".format(row=row) if target_scope == "final" else ""
            if verb in ("insert", "update"):
                statements.append(
                    "INSERT INTO idx_counts(scope, bucket, value, n) "
                    "SELECT '{scope}', '{bucket}', {row}.{column}, {sign} "
                    "WHERE 1{guard} "
                    "ON CONFLICT(scope, bucket, value) DO UPDATE SET n = n + {sign};".format(
                        scope=target_scope,
                        bucket=bucket,
                        row=row,
                        column=column,
                        sign=sign,
                        guard=guard,
                    )
                )
            if verb in ("delete", "update"):
                statements.append(
                    "UPDATE idx_counts SET n = n - {sign} "
                    "WHERE scope = '{scope}' AND bucket = '{bucket}' "
                    "AND value = {row}.{column}{guard};".format(
                        scope=target_scope,
                        bucket=bucket,
                        row=row,
                        column=column,
                        sign=sign,
                        guard=guard,
                    )
                )
    # One sweep at the end of the trigger: a bucket that reached zero while the
    # row moved has left the vocabulary, and a zero row is a lie in JSON.
    statements.append("DELETE FROM idx_counts WHERE n <= 0;")
    return statements


def _fts_values(row: str) -> str:
    """The four FTS columns for a row of ``idx_assets``.

    ``name`` carries the effective name AND the derived file name: a rename must
    not make the original filename unfindable. They are equal on most rows, so
    the concat is conditional rather than a permanent duplication.
    """
    return (
        "CASE WHEN {row}.effective_name = {row}.derived_name "
        "THEN {row}.effective_name "
        "ELSE {row}.effective_name || ' ' || {row}.derived_name END, "
        "{row}.description, {row}.tag_names, {row}.session_titles, "
        # The absolute path, which the old haystack reached through
        # ``display_path``: a folder name like ``CameraRoll`` is a token of it,
        # and dropping the column silently made that search return nothing.
        "{row}.path"
    ).format(row=row)


def _schema_sql() -> str:
    # ``is_final`` is stored, not computed: the sidebar's final scope and the
    # scope filter both read it, and a generated expression in a WHERE clause
    # cannot use an index.
    statements = [
        """
CREATE TABLE IF NOT EXISTS idx_assets(
    asset_id TEXT PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    path TEXT,
    digest TEXT,
    derived_name TEXT NOT NULL DEFAULT '',
    effective_name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'other',
    kind TEXT NOT NULL DEFAULT 'other',
    content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
    origin TEXT NOT NULL DEFAULT 'mentioned',
    origin_group TEXT NOT NULL DEFAULT 'mentioned',
    origins TEXT NOT NULL DEFAULT '[]',
    highlighted INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER,
    width INTEGER,
    height INTEGER,
    duration_s REAL,
    first_ts REAL NOT NULL DEFAULT 0,
    last_ts REAL NOT NULL DEFAULT 0,
    is_final INTEGER NOT NULL DEFAULT 0,
    tag_names TEXT NOT NULL DEFAULT '',
    session_titles TEXT NOT NULL DEFAULT '',
    indexed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_recent ON idx_assets(last_ts DESC, asset_id);
CREATE INDEX IF NOT EXISTS idx_assets_name ON idx_assets(effective_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_assets_size ON idx_assets(size_bytes);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON idx_assets(kind);
CREATE INDEX IF NOT EXISTS idx_assets_category ON idx_assets(category);
CREATE INDEX IF NOT EXISTS idx_assets_origin_group ON idx_assets(origin_group);
CREATE INDEX IF NOT EXISTS idx_assets_final ON idx_assets(is_final);
""",
        # The record fragments are carried per occurrence so an asset's
        # aggregate is a pure function of its occurrences: a session's removal
        # (or a rebuild from a pre-populated checkpoint) then recomputes exactly
        # what ``index._build_assets`` would have, without rescanning a journal.
        """
CREATE TABLE IF NOT EXISTS idx_occurrences(
    asset_id TEXT NOT NULL,
    -- The canonical key, not just its hash: an asset's aggregate is rebuilt
    -- from occurrences alone, and ``a:<digest>`` rows carry no path to recover
    -- the key from (``asset_id`` is a one-way hash of it).
    key TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'mentioned',
    child_id TEXT NOT NULL DEFAULT '',
    label TEXT,
    tool TEXT NOT NULL DEFAULT '',
    intent TEXT,
    first_ts REAL NOT NULL DEFAULT 0,
    last_ts REAL NOT NULL DEFAULT 0,
    first_seq INTEGER NOT NULL DEFAULT 0,
    last_seq INTEGER NOT NULL DEFAULT 0,
    touches INTEGER NOT NULL DEFAULT 1,
    highlighted INTEGER NOT NULL DEFAULT 0,
    name TEXT NOT NULL DEFAULT '',
    path TEXT,
    kind TEXT NOT NULL DEFAULT 'other',
    -- Nullable on purpose: the aggregate replays ``classify`` with the RECORD's
    -- MIME, and substituting a default here would make every text file
    -- "application/octet-stream".
    content_type TEXT,
    size_bytes INTEGER,
    width INTEGER,
    height INTEGER,
    duration_s REAL,
    PRIMARY KEY(asset_id, session_id, child_id)
);
CREATE INDEX IF NOT EXISTS idx_occurrences_asset ON idx_occurrences(asset_id);
CREATE INDEX IF NOT EXISTS idx_occurrences_session ON idx_occurrences(session_id);
""",
        # The resumable checkpoint. ``sig_*`` is the journal identity
        # (size/mtime_ns/inode): a session whose signature is unchanged is not
        # rescanned, which is what makes a restart cheap.
        """
CREATE TABLE IF NOT EXISTS idx_sessions(
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    sig_size INTEGER NOT NULL DEFAULT -1,
    sig_mtime REAL NOT NULL DEFAULT -1,
    sig_inode INTEGER NOT NULL DEFAULT -1,
    -- The stored-title and roster identities, not just the transcript's: a
    -- ``/rename`` and a roster-only membership edit both leave the journal
    -- untouched, and both change what the Library lists and searches.
    title_key TEXT NOT NULL DEFAULT '',
    roster_key TEXT NOT NULL DEFAULT '',
    revision REAL NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS idx_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idx_counts(
    scope TEXT NOT NULL, bucket TEXT NOT NULL, value TEXT NOT NULL, n INTEGER NOT NULL,
    PRIMARY KEY(scope, bucket, value));
-- One row PER WIDTH. A single slot on idx_assets could only ever serve the
-- width the worker happened to generate last, so a `w=240` caller was handed
-- the full-size original on every render. ``path`` is the cached WebP and
-- ``src`` the source's stat identity (a moved source invalidates its row).
CREATE TABLE IF NOT EXISTS idx_thumbnails(
    asset_id TEXT NOT NULL,
    width INTEGER NOT NULL,
    path TEXT NOT NULL,
    key TEXT NOT NULL,
    src TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(asset_id, width));
CREATE INDEX IF NOT EXISTS idx_thumbnails_asset ON idx_thumbnails(asset_id);
""",
        """
CREATE VIRTUAL TABLE IF NOT EXISTS idx_assets_fts USING fts5(
    name, description, tags, sessions, path,
    tokenize = 'unicode61 remove_diacritics 2');
""",
        # -- counts, maintained by the database rather than by a caller --------
        "CREATE TRIGGER IF NOT EXISTS idx_assets_counts_ins AFTER INSERT ON idx_assets BEGIN "
        + " ".join(_count_statements(verb="insert", row="NEW", sign=1))
        + " END;",
        "CREATE TRIGGER IF NOT EXISTS idx_assets_counts_del AFTER DELETE ON idx_assets BEGIN "
        + " ".join(_count_statements(verb="delete", row="OLD", sign=1))
        + " END;",
        "CREATE TRIGGER IF NOT EXISTS idx_assets_counts_upd AFTER UPDATE ON idx_assets "
        "WHEN OLD.category IS NOT NEW.category OR OLD.kind IS NOT NEW.kind "
        "OR OLD.origin_group IS NOT NEW.origin_group OR OLD.is_final IS NOT NEW.is_final "
        "BEGIN "
        + " ".join(
            _count_statements(verb="delete", row="OLD", sign=1)
            + _count_statements(verb="insert", row="NEW", sign=1)
        )
        + " END;",
        # -- full text ---------------------------------------------------------
        "CREATE TRIGGER IF NOT EXISTS idx_assets_fts_ins AFTER INSERT ON idx_assets BEGIN "
        "INSERT INTO idx_assets_fts(rowid, name, description, tags, sessions, path) "
        "VALUES (NEW.rowid, {values}); END;".format(values=_fts_values("NEW")),
        "CREATE TRIGGER IF NOT EXISTS idx_assets_fts_del AFTER DELETE ON idx_assets BEGIN "
        "DELETE FROM idx_assets_fts WHERE rowid = OLD.rowid; END;",
        "CREATE TRIGGER IF NOT EXISTS idx_assets_fts_upd AFTER UPDATE ON idx_assets "
        "WHEN OLD.effective_name IS NOT NEW.effective_name "
        "OR OLD.derived_name IS NOT NEW.derived_name "
        "OR OLD.path IS NOT NEW.path "
        "OR OLD.description IS NOT NEW.description "
        "OR OLD.tag_names IS NOT NEW.tag_names "
        "OR OLD.session_titles IS NOT NEW.session_titles "
        "BEGIN "
        "DELETE FROM idx_assets_fts WHERE rowid = OLD.rowid; "
        "INSERT INTO idx_assets_fts(rowid, name, description, tags, sessions, path) "
        "VALUES (NEW.rowid, {values}); END;".format(values=_fts_values("NEW")),
        # -- authoritative metadata flows INTO the index -----------------------
        #
        # These are the reason a rename, a description edit and a tag change are
        # searchable immediately without the service having to know the index
        # exists. Each is a no-op when the asset has no idx row yet (SQLite
        # UPDATE simply matches nothing).
        "CREATE TRIGGER IF NOT EXISTS meta_assets_idx_ins AFTER INSERT ON assets BEGIN "
        + _METADATA_REFRESH_TRIGGER_BODY
        + " END;",
        "CREATE TRIGGER IF NOT EXISTS meta_assets_idx_upd "
        "AFTER UPDATE OF name, description, name_source, description_source ON assets BEGIN "
        + _METADATA_REFRESH_TRIGGER_BODY
        + " END;",
        # A tag link, an unlink, and a tag rename each rewrite the joined names
        # for exactly the assets affected. ``merge_tags`` re-inserts and deletes
        # rows rather than updating ``tag_id``, so insert/delete covers it.
        "CREATE TRIGGER IF NOT EXISTS meta_asset_tags_ins AFTER INSERT ON asset_tags BEGIN "
        + _TAG_REFRESH_SQL.format(session="NEW")
        + " END;",
        "CREATE TRIGGER IF NOT EXISTS meta_asset_tags_del AFTER DELETE ON asset_tags BEGIN "
        + _TAG_REFRESH_SQL.format(session="OLD")
        + " END;",
        "CREATE TRIGGER IF NOT EXISTS meta_asset_tags_upd AFTER UPDATE ON asset_tags BEGIN "
        + _TAG_REFRESH_SQL.format(session="NEW")
        + _TAG_REFRESH_SQL.format(session="OLD")
        + " END;",
        "CREATE TRIGGER IF NOT EXISTS meta_tags_rename AFTER UPDATE OF name ON tags BEGIN "
        "UPDATE idx_assets SET tag_names = ("
        "  SELECT COALESCE(GROUP_CONCAT(t.name, ' '), '') FROM asset_tags at "
        "  JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = idx_assets.asset_id) "
        "WHERE asset_id IN (SELECT asset_id FROM asset_tags WHERE tag_id = NEW.id); END;",
    ]
    return "".join(statement.strip() + "\n" for statement in statements)


#: Effective name/description precedence, in SQL. Mirrors
#: ``service._effective`` exactly: a stored value counts only when its source is
#: one of the three deliberate writers, and an empty string is not a choice.
_METADATA_REFRESH_TRIGGER_BODY: Final = """
UPDATE idx_assets SET
  effective_name = CASE
    WHEN NEW.name IS NOT NULL AND NEW.name != ''
     AND NEW.name_source IN ('user', 'librarian', 'agent') THEN NEW.name
    ELSE idx_assets.derived_name END,
  description = CASE
    WHEN NEW.description IS NOT NULL AND NEW.description != ''
     AND NEW.description_source IN ('user', 'librarian', 'agent') THEN NEW.description
    ELSE '' END
WHERE asset_id = NEW.asset_id;
"""

_TAG_REFRESH_SQL: Final = (
    "UPDATE idx_assets SET tag_names = ("
    "  SELECT COALESCE(GROUP_CONCAT(t.name, ' '), '') FROM asset_tags at "
    "  JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = {session}.asset_id) "
    "WHERE asset_id = {session}.asset_id;"
)

#: Derived tables, dropped together when the schema version moves. Ordered so a
#: drop never trips a trigger on a table that still exists.
_DERIVED_TABLES: Final = (
    "idx_assets_fts",
    "idx_thumbnails",
    "idx_counts",
    "idx_occurrences",
    "idx_sessions",
    "idx_assets",
)


#: Every trigger this module creates. ``recover_derived`` drops these BEFORE
#: the tables: three of them live on the authoritative metadata tables, so
#: dropping ``idx_assets`` alone would leave a rename trigger writing to a table
#: that no longer exists and turn every metadata edit into an error.
_DERIVED_TRIGGERS: Final = (
    "idx_assets_counts_ins",
    "idx_assets_counts_del",
    "idx_assets_counts_upd",
    "idx_assets_fts_ins",
    "idx_assets_fts_del",
    "idx_assets_fts_upd",
    "meta_assets_idx_ins",
    "meta_assets_idx_upd",
    "meta_asset_tags_ins",
    "meta_asset_tags_del",
    "meta_asset_tags_upd",
    "meta_tags_rename",
)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_tokens(q: str) -> list[str]:
    """The terms a search box's text breaks into. Empty means "no text filter"."""
    return _Q_TOKEN.findall(q or "")[:_MAX_Q_TERMS]


def fts_expression(q: str) -> str | None:
    """An AND of quoted prefix terms, or ``None`` when the text has no term.

    INTENTIONAL CHANGE FROM SUBSTRING SEARCH. The old filter was
    ``q.lower() in haystack``: ``reen`` found ``screen.png``, and ``cat dog``
    found nothing unless that exact pair appeared in order. A search-as-you-type
    box wants token-prefix: each token must START a word (``scr`` finds
    ``screen``), and several tokens must all appear somewhere in the row. The
    change is deliberate — the old semantics are not expressible as an index
    lookup, and they surprised users by matching inside words.
    """
    tokens = search_tokens(q)
    if not tokens:
        return None
    # Every token is quoted, so FTS5 operators inside user text (``OR``, ``NEAR``,
    # ``-``, ``*``) stay literal text rather than becoming query syntax.
    return " AND ".join(f'"{token}"*' for token in tokens)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class LibraryStore:
    """The derived index. One connection per call, like the metadata store.

    Construct per config root; it is cheap. ``ensure_schema`` is called lazily
    by every connection so a process that only reads still creates the tables it
    needs, and a schema-version move drops the derived tables and rebuilds.
    """

    _bootstrap_lock = threading.Lock()
    #: Paths whose derived schema this PROCESS has verified. Class-level for
    #: the same reason as ``MetadataStore._ready_paths``: one store per list
    #: request must not re-run DDL on the read path.
    _ready_paths: set[str] = set()

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / DB_DIRNAME / DB_FILENAME
        self._key = str(self.path.resolve())

    # -- connections --------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5.0)
        except (OSError, sqlite3.Error) as error:
            raise IndexUnavailable(str(error)) from error
        try:
            connection.row_factory = sqlite3.Row
            try:
                # journal_mode is a DATABASE property, not a connection one: it is
                # persisted in the file, so switching it on every connect only
                # buys a write lock another connection may already hold (observed
                # as ``database is locked`` under the librarian's concurrent
                # reads). ``_ensure_schema`` sets it once, at bootstrap.
                connection.execute("PRAGMA busy_timeout=5000")
                connection.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError as error:
                raise IndexUnavailable(str(error)) from error
            # ``_ensure_schema`` answers True or raises; a False return would be
            # a silent "serve an empty index" that no caller can distinguish.
            self._ensure_schema(connection)
            yield connection
        finally:
            try:
                connection.close()
            except sqlite3.Error:
                pass

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """One read, with a torn page translated into ``IndexUnavailable``.

        A corrupted b-tree raises ``sqlite3.DatabaseError`` from whichever
        statement touches it — ``database disk image is malformed`` out of an
        occurrence lookup, for instance. Translated HERE so every read path is
        covered once, and the service's one-shot recovery can act on it instead
        of a bare 500 reaching the route.
        """
        with self._connect() as connection:
            try:
                yield connection
            except sqlite3.DatabaseError as error:
                raise IndexUnavailable(str(error), corrupt=_is_corruption(error)) from error

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction: a session's rows land together."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException as error:
                connection.rollback()
                if isinstance(error, sqlite3.DatabaseError):
                    raise IndexUnavailable(str(error), corrupt=_is_corruption(error)) from error
                raise
            connection.commit()

    def ensure_schema(self) -> bool:
        """Create or migrate the derived schema. Idempotent."""
        with self._connect() as connection:
            return self._ensure_schema(connection)

    def _ensure_schema(self, connection: sqlite3.Connection) -> bool:
        if self._key in self._ready_paths:
            return True
        with self._bootstrap_lock:
            if self._key in self._ready_paths:
                return True
            try:
                # Bootstrap only, and inside the guard: WAL mode needs no open
                # transaction, and a dead connection must surface as the typed
                # refusal, not as a bare ``ProgrammingError``.
                connection.execute("PRAGMA journal_mode=WAL")
                # The metadata tables are AUTHORITATIVE and owned by
                # ``MetadataStore``; three triggers below attach to them, so they
                # must exist before this script runs. A derived table may never
                # be the thing that creates ``assets``.
                if not MetadataStore(self.config_dir).ensure_schema():
                    raise IndexUnavailable("library metadata schema is unavailable")
                row = self._state_value(connection, STATE_SCHEMA)
                current = int(row) if row and row.isdigit() else None
                if current is not None and current != SCHEMA_VERSION:
                    # These tables are derived: a version move discards them and
                    # the indexer rebuilds. Never touched: the metadata tables.
                    self._drop_derived(connection)
                    # The COUNTERS are derived too. Left behind, ``backfill_done``
                    # still said "1" and ``sessions_total`` still described the
                    # dropped rows, so a rebuilding index reported state=ready
                    # with scanned=0 and the worker picked the 6 h cadence.
                    for key in (STATE_REVISION, STATE_BACKFILL_DONE):
                        connection.execute("DELETE FROM idx_state WHERE key = ?", (key,))
                    for key in (STATE_SESSIONS_TOTAL, STATE_SESSIONS_SCANNED):
                        connection.execute("DELETE FROM idx_state WHERE key = ?", (key,))
                    connection.commit()
                connection.executescript(_schema_sql())
                connection.execute(
                    "INSERT INTO idx_state(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (STATE_SCHEMA, str(SCHEMA_VERSION)),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO idx_state(key, value) VALUES(?, '0')",
                    (STATE_REVISION,),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO idx_state(key, value) VALUES(?, '0')",
                    (STATE_BACKFILL_DONE,),
                )
                connection.commit()
                self._ready_paths.add(self._key)
                return True
            except IndexUnavailable:
                raise
            except sqlite3.Error as error:
                logger.warning("library index schema failure", exc_info=True)
                raise IndexUnavailable(str(error)) from error

    @staticmethod
    def _drop_derived_statements() -> list[str]:
        """``DROP`` for every derived trigger and table, metadata untouched.

        Triggers first: three of them hang off ``assets``/``asset_tags``/``tags``
        and would reference a table that no longer exists.
        """
        return [f"DROP TRIGGER IF EXISTS {trigger}" for trigger in _DERIVED_TRIGGERS] + [
            f"DROP TABLE IF EXISTS {table}" for table in _DERIVED_TABLES
        ]

    @classmethod
    def _drop_derived(cls, connection: sqlite3.Connection) -> None:
        for statement in cls._drop_derived_statements():
            connection.execute(statement)
        connection.commit()

    def recover_derived(self) -> bool:
        """Discard the derived index and recreate an empty one. Never raises.

        Two paths, because a damaged index fails in two different ways:

        * A MALFORMED table cannot be repaired through ``_connect`` — its
          ``_ensure_schema`` fails on the very table recovery must drop — so the
          fast path opens a raw connection and runs DROP + CREATE + the counter
          reset as ONE script inside one ``BEGIN IMMEDIATE``. A failure halfway
          rolls back and the next attempt starts from the same state.
        * A TORN b-tree cannot even be dropped: SQLite cannot free a page whose
          root or header is damage, so the DROP itself answers ``database disk
          image is malformed`` (QA Q-1). That case falls back to
          :meth:`_rebuild_database`, which copies the authoritative tables into
          a fresh file and swaps it in.

        AUTHORITATIVE tables are never discarded, so names, tags, favourites and
        edits survive a corrupt index.
        """
        try:
            # The cached readiness would short-circuit the schema work.
            self._ready_paths.discard(self._key)
            # The derived triggers attach to these; they are authoritative and
            # ``MetadataStore`` owns them.
            if not MetadataStore(self.config_dir).ensure_schema():
                return False
            self._recreate_derived()
            self._ready_paths.add(self._key)
            logger.info("library: derived index rebuilt after an unreadable file")
            return True
        except sqlite3.DatabaseError as error:
            # The damage is inside a b-tree this script must free. Only a new
            # file can be built from what is still readable.
            logger.warning(
                "library: derived tables cannot be dropped (%s); rebuilding the database",
                error,
            )
        except Exception:  # noqa: BLE001 - a failed recovery is a 503, not a crash
            logger.warning("library: derived index recovery failed", exc_info=True)
            return False
        return self._rebuild_database()

    def _recreate_derived(self) -> None:
        """DROP + CREATE + reset in one transaction. Raises on failure."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.executescript(self._recreate_derived_script())
        finally:
            connection.close()

    def _recreate_derived_script(self) -> str:
        return "\n".join(
            [
                "BEGIN IMMEDIATE;",
                # Statements carry no trailing ";" of their own, and the schema
                # block below starts with one: without it the script reads
                # "DROP ... CREATE ..." and SQLite refuses.
                *[f"{statement};" for statement in self._drop_derived_statements()],
                # ``_schema_sql`` is ONE script string, not a list of
                # statements: unpacking it would feed SQLite its letters.
                _schema_sql(),
                *[f"{statement};" for statement in self._state_reset_statements()],
                "COMMIT;",
            ]
        )

    @staticmethod
    def _state_reset_statements() -> list[str]:
        return [
            "INSERT INTO idx_state(key, value) VALUES('{0}', '{1}') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value;".format(key, value)
            for key, value in (
                (STATE_SCHEMA, SCHEMA_VERSION),
                # ``STATE_SESSIONS_TOTAL`` is left UNSET rather than 0 by callers
                # that know better; the rebuild has no way to know the visible
                # set, so 0 is honest here and ``_maybe_finish_backfill`` treats
                # "0 owed" as unknown while journals exist.
                (STATE_REVISION, "0"),
                (STATE_BACKFILL_DONE, "0"),
                (STATE_SESSIONS_TOTAL, "0"),
                (STATE_SESSIONS_SCANNED, "0"),
            )
        ]

    def _rebuild_database(self) -> bool:
        """Copy the authoritative tables into a fresh file and swap it in.

        The metadata tables live in their own btrees, so a torn DERIVED table
        leaves them readable. This builds the replacement beside the original,
        copies row-for-row through ``ATTACH``, and ``os.replace``s it — the
        original is never edited, so a failure leaves it exactly as it was.

        Transactions and locking: one process-wide lock means two requests
        cannot swap at once, and the caller's cooldown keeps a persistently
        unreadable file from paying this repeatedly. The replacement is built in
        rollback-journal mode so the single file is complete when it is moved;
        the old ``-wal``/``-shm`` are removed after the swap because they belong
        to the file that no longer exists at that path.
        """
        with _REBUILD_LOCK:
            try:
                # A second request may have waited on this lock while the first
                # rebuilt the file. If the DDL now runs, the repair is done and
                # a second copy would only churn the disk.
                self._recreate_derived()
                self._ready_paths.add(self._key)
                return True
            except sqlite3.DatabaseError:
                pass
            temp = self.path.with_name(
                f"{self.path.name}.rebuild-{os.getpid()}-{threading.get_ident()}"
            )
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ("", "-wal", "-shm", "-journal"):
                    Path(f"{temp}{suffix}").unlink(missing_ok=True)
                fresh = sqlite3.connect(temp)
                try:
                    fresh.executescript(schema_sql())
                    fresh.execute("ATTACH DATABASE ? AS old", (str(self.path),))
                    try:
                        fresh.execute("BEGIN IMMEDIATE")
                        for table in AUTHORITATIVE_TABLES:
                            columns = self._copy_columns(fresh, table)
                            if not columns:
                                continue
                            listing = ", ".join(columns)
                            fresh.execute(
                                f"INSERT INTO main.{table}({listing}) "
                                f"SELECT {listing} FROM old.{table}"
                            )
                        fresh.execute("COMMIT")
                    finally:
                        fresh.execute("DETACH DATABASE old")
                    # Derived tables and triggers, then the reset. Authoritative
                    # rows are already in place, so the triggers fire on nothing.
                    fresh.executescript(_schema_sql())
                    fresh.executescript(
                        "BEGIN IMMEDIATE;" + "".join(self._state_reset_statements()) + "COMMIT;"
                    )
                finally:
                    fresh.close()
                self._swap_in_database(temp)
                logger.info("library: rebuilt the database from its metadata tables")
                return True
            except Exception:  # noqa: BLE001 - refusal, never a crash
                logger.warning("library: database rebuild failed", exc_info=True)
                for suffix in ("", "-wal", "-shm", "-journal"):
                    try:
                        Path(f"{temp}{suffix}").unlink(missing_ok=True)
                    except OSError:
                        continue
                return False

    def _swap_in_database(self, temp: Path) -> None:
        """Replace the database with ``temp`` and drop the old file's WAL.

        The ``-wal``/``-shm`` belong to the file that is being replaced: leaving
        them would attach another database's pages to this path. A method of its
        own so the tests can prove two concurrent recoveries swap exactly once.
        """
        for suffix in ("-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)
        os.replace(temp, self.path)

    @staticmethod
    def _copy_columns(connection: sqlite3.Connection, table: str) -> list[str]:
        """Columns present in BOTH files: a migrated table may differ by one."""
        fresh = [str(row[1]) for row in connection.execute(f"PRAGMA main.table_info({table})")]
        old = [str(row[1]) for row in connection.execute(f"PRAGMA old.table_info({table})")]
        return [name for name in fresh if name in set(old)]

    @staticmethod
    def _state_value(connection: sqlite3.Connection, key: str) -> str | None:
        try:
            row = connection.execute("SELECT value FROM idx_state WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        return str(row["value"]) if row is not None else None

    # -- state --------------------------------------------------------------

    def state(self) -> dict[str, str]:
        with self.read() as connection:
            return self._state(connection)

    def set_state(self, connection: sqlite3.Connection, **values: Any) -> None:
        for key, value in values.items():
            connection.execute(
                "INSERT INTO idx_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def revision(self) -> int:
        with self.read() as connection:
            return self._revision(connection)

    @staticmethod
    def _revision(connection: sqlite3.Connection) -> int:
        raw = LibraryStore._state_value(connection, STATE_REVISION)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    def bump_revision(self, connection: sqlite3.Connection) -> int:
        """Advance the catalog revision a list cursor binds to."""
        value = self._revision(connection) + 1
        self.set_state(connection, **{STATE_REVISION: value})
        return value

    def listing_state(self) -> tuple[str, int, int]:
        """``(state, sessions_scanned, sessions_total)`` for the list envelope.

        One connection for all three: a second one could observe a different
        moment and report ``scanned`` > ``total`` mid-write.
        """
        with self.read() as connection:
            state = self._state(connection)
            scanned = self._count_sessions(connection, complete=True)
            total = self._int_state(state, STATE_SESSIONS_TOTAL)
        if state.get(STATE_BACKFILL_DONE, "0") == "1" or self.sessions_dir_empty():
            # An EMPTY journal directory is a finished empty index, not a build
            # in progress: a fresh install must open on "ready" with an empty
            # Library, not on a scanning spinner that never ends.
            return ("ready", scanned, total)
        return ("building", scanned, total)

    def sessions_dir_empty(self) -> bool:
        """One ``scandir`` probe; it answers on the first entry of a real store."""
        try:
            with os.scandir(self.config_dir / "sessions") as entries:
                return next(entries, None) is None
        except OSError:
            return True

    @staticmethod
    def _state(connection: sqlite3.Connection) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM idx_state")
        }

    def _count_sessions(self, connection: sqlite3.Connection, *, complete: bool) -> int:
        clause = " WHERE complete = 1" if complete else ""
        row = connection.execute(f"SELECT COUNT(*) AS n FROM idx_sessions{clause}").fetchone()
        return int(row["n"]) if row is not None else 0

    @staticmethod
    def _int_state(state: dict[str, str], key: str) -> int:
        try:
            return int(state.get(key, "0"))
        except (TypeError, ValueError):
            return 0

    # -- sessions -----------------------------------------------------------

    def sessions(self) -> dict[str, dict[str, Any]]:
        """Every recorded session checkpoint, keyed by id."""
        with self.read() as connection:
            return {
                str(row["session_id"]): dict(row)
                for row in connection.execute("SELECT * FROM idx_sessions")
            }

    def session(self, session_id: str) -> dict[str, Any] | None:
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM idx_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def record_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        *,
        title: str,
        sig: dict[str, Any] | None,
        complete: bool,
        title_key: str = "",
        roster_key: str = "",
        revision: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Store the checkpoint for one scanned session.

        ``revision`` is the ledger's own journal revision, not this index's
        counter: it is what tells the next run whether the journal moved even
        when size and mtime happen to repeat.
        """
        sig = sig or {}
        connection.execute(
            "INSERT INTO idx_sessions(session_id, title, sig_size, sig_mtime, sig_inode, "
            "title_key, roster_key, revision, complete, indexed_at, last_error) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET title = excluded.title, "
            "sig_size = excluded.sig_size, sig_mtime = excluded.sig_mtime, "
            "sig_inode = excluded.sig_inode, title_key = excluded.title_key, "
            "roster_key = excluded.roster_key, revision = excluded.revision, "
            "complete = excluded.complete, indexed_at = excluded.indexed_at, "
            "last_error = excluded.last_error",
            (
                session_id,
                title,
                int(sig.get("size", -1)),
                float(sig.get("mtime", -1)),
                int(sig.get("inode", -1)),
                title_key,
                roster_key,
                float(revision),
                1 if complete else 0,
                time.time(),
                error,
            ),
        )

    def forget_session(self, connection: sqlite3.Connection, session_id: str) -> set[str]:
        """Drop a session's occurrences; return the asset ids that lost one."""
        affected = {
            str(row["asset_id"])
            for row in connection.execute(
                "SELECT DISTINCT asset_id FROM idx_occurrences WHERE session_id = ?",
                (session_id,),
            )
        }
        connection.execute("DELETE FROM idx_occurrences WHERE session_id = ?", (session_id,))
        connection.execute("DELETE FROM idx_sessions WHERE session_id = ?", (session_id,))
        return affected

    # -- write path ---------------------------------------------------------

    def replace_session_occurrences(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        rows: Sequence[dict[str, Any]],
    ) -> set[str]:
        """Replace one session's occurrence rows; return every asset they touch.

        The union includes assets the session STOPPED touching: their aggregate
        is now different (a dropped origin can demote ``origin_group``), so the
        caller must recompute them even though no new row mentions them.
        """
        affected: set[str] = set()
        for row in connection.execute(
            "SELECT DISTINCT asset_id FROM idx_occurrences WHERE session_id = ?", (session_id,)
        ):
            affected.add(str(row["asset_id"]))
        connection.execute("DELETE FROM idx_occurrences WHERE session_id = ?", (session_id,))
        connection.executemany(
            "INSERT OR REPLACE INTO idx_occurrences(asset_id, key, session_id, title, origin, "
            "child_id, label, tool, intent, first_ts, last_ts, first_seq, last_seq, touches, "
            "highlighted, name, path, kind, content_type, size_bytes, width, height, "
            "duration_s) VALUES(" + ", ".join(["?"] * 23) + ")",
            [
                (
                    row["asset_id"],
                    row.get("key") or "",
                    session_id,
                    row.get("title") or "",
                    row.get("origin") or "mentioned",
                    row.get("child_id") or "",
                    row.get("label"),
                    row.get("tool") or "",
                    row.get("intent"),
                    float(row.get("first_ts") or 0.0),
                    float(row.get("last_ts") or 0.0),
                    int(row.get("first_seq") or 0),
                    int(row.get("last_seq") or 0),
                    int(row.get("touches") or 1),
                    1 if row.get("highlighted") else 0,
                    row.get("name") or "",
                    row.get("path"),
                    row.get("kind") or "other",
                    # NULL, not the column default: a recorded MIME wins in
                    # ``classify``, so substituting one here would make every
                    # text file "application/octet-stream".
                    row.get("content_type") or None,
                    row.get("size_bytes"),
                    row.get("width"),
                    row.get("height"),
                    row.get("duration_s"),
                )
                for row in rows
            ],
        )
        affected.update(row["asset_id"] for row in rows)
        return affected

    def recompute_assets(
        self,
        connection: sqlite3.Connection,
        asset_ids: Iterable[str],
    ) -> None:
        """Rewrite ``idx_assets`` for the given ids from their occurrences.

        An asset with no occurrences left is deleted (and its counts/FTS row go
        with it through the triggers). The effective name/description are read
        from the ``assets`` table ON THIS CONNECTION, never passed in: a caller
        that forgot to would recompute from the derived name alone and silently
        undo a user's rename the next time any journal touched the asset.
        """
        for asset_id in asset_ids:
            rows = list(
                connection.execute(
                    # ``rowid`` is insertion order, so rescanning ONE session
                    # moved its rows to the end and flipped a shared asset's
                    # last-wins fields (derived name, kind, content type) to
                    # whatever that session recorded. The aggregate must be a
                    # function of the occurrence SET, not of scan history.
                    "SELECT * FROM idx_occurrences WHERE asset_id = ? "
                    "ORDER BY first_ts, session_id, child_id",
                    (asset_id,),
                )
            )
            existing = connection.execute(
                "SELECT key FROM idx_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if not rows:
                connection.execute("DELETE FROM idx_assets WHERE asset_id = ?", (asset_id,))
                continue
            aggregate = _aggregate(
                asset_id, str(existing["key"]) if existing else str(rows[0]["key"]), rows
            )
            if aggregate is None:
                connection.execute("DELETE FROM idx_assets WHERE asset_id = ?", (asset_id,))
                continue
            meta = connection.execute(
                "SELECT name, name_source, description, description_source FROM assets "
                "WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
            effective_name = _effective_from_row(
                meta["name"] if meta else None,
                meta["name_source"] if meta else None,
                aggregate.derived_name,
            )
            description = _effective_from_row(
                meta["description"] if meta else None,
                meta["description_source"] if meta else None,
                "",
            )
            connection.execute(
                "INSERT INTO idx_assets(asset_id, key, path, digest, derived_name, "
                "effective_name, description, category, kind, content_type, origin, "
                "origin_group, origins, highlighted, size_bytes, width, height, duration_s, "
                "first_ts, last_ts, is_final, tag_names, session_titles, indexed_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                # Tag names are the metadata store's, not the previous idx row's:
                # an asset whose metadata row survived a rebuild must come back
                # searchable by its tags without waiting for the next tag write.
                "  COALESCE((SELECT GROUP_CONCAT(t.name, ' ') FROM asset_tags at "
                "            JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = ?), ''), "
                "  ?, "
                "  ?) "
                "ON CONFLICT(asset_id) DO UPDATE SET key = excluded.key, path = excluded.path, "
                "digest = excluded.digest, derived_name = excluded.derived_name, "
                "effective_name = excluded.effective_name, description = excluded.description, "
                "category = excluded.category, kind = excluded.kind, "
                "content_type = excluded.content_type, origin = excluded.origin, "
                "origin_group = excluded.origin_group, origins = excluded.origins, "
                "highlighted = excluded.highlighted, size_bytes = excluded.size_bytes, "
                "width = excluded.width, height = excluded.height, "
                "duration_s = excluded.duration_s, first_ts = excluded.first_ts, "
                "last_ts = excluded.last_ts, is_final = excluded.is_final, "
                "session_titles = excluded.session_titles, indexed_at = excluded.indexed_at",
                (
                    asset_id,
                    aggregate.key,
                    aggregate.path,
                    aggregate.digest,
                    aggregate.derived_name,
                    effective_name,
                    description,
                    aggregate.category,
                    aggregate.kind,
                    aggregate.content_type,
                    aggregate.origin,
                    aggregate.origin_group,
                    json.dumps(aggregate.origins),
                    1 if aggregate.highlighted else 0,
                    aggregate.size_bytes,
                    aggregate.width,
                    aggregate.height,
                    aggregate.duration_s,
                    aggregate.first_ts,
                    aggregate.last_ts,
                    1 if is_final(aggregate) else 0,
                    asset_id,
                    " ".join(aggregate.session_titles),
                    time.time(),
                ),
            )

    # -- read path ----------------------------------------------------------

    def query_page(
        self, query: ListingQuery, *, offset: int = 0, limit: int = 100
    ) -> tuple[list[library_index.Asset], int]:
        """One page of assets plus the total the filters match.

        Ordering, filters and the count share ONE ``WHERE`` clause: a total that
        disagreed with the rows would make ``next_cursor`` lie.
        """
        where, params = _where_clause(query)
        match = fts_expression(query.q)
        if match is None and query.q.strip():
            # A NON-EMPTY query with no searchable token ("!!", "…") matches
            # NOTHING. Dropping the filter instead would answer "all rows" for
            # something the user typed, which reads as the search being broken.
            return [], 0
        if match is not None:
            where.append(
                "i.rowid IN (SELECT rowid FROM idx_assets_fts WHERE idx_assets_fts MATCH ?)"
            )
            params.append(match)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        # ``effective_order`` is the Query's own ``order or DEFAULT_ORDER[sort]``.
        # Reading ``query.order`` alone turned "kind" into a DESC sort for every
        # caller that leaves the order out (the TUI, the mobile API).
        order = _order_clause(query.sort, getattr(query, "effective_order", None) or query.order)
        with self.read() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM idx_assets i "
                    "LEFT JOIN assets m ON m.asset_id = i.asset_id" + clause,
                    params,
                ).fetchone()["n"]
            )
            rows = list(
                connection.execute(
                    "SELECT i.* FROM idx_assets i "
                    "LEFT JOIN assets m ON m.asset_id = i.asset_id"
                    + clause
                    + order
                    + " LIMIT ? OFFSET ?",
                    [*params, int(limit), int(offset)],
                )
            )
            occurrences = self._occurrences(connection, [str(row["asset_id"]) for row in rows])
        return [
            _asset_from_row(row, occurrences.get(str(row["asset_id"]), [])) for row in rows
        ], total

    def asset(self, asset_id: str) -> library_index.Asset | None:
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM idx_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if row is None:
                return None
            occurrences = self._occurrences(connection, [asset_id]).get(asset_id, [])
        return _asset_from_row(row, occurrences)

    def occurrences(self, asset_ids: Sequence[str]) -> dict[str, list[library_index.Occurrence]]:
        """Hydrate one page's occurrences. A page, never the catalog."""
        if not asset_ids:
            return {}
        with self.read() as connection:
            return self._occurrences(connection, asset_ids)

    def _occurrences(
        self, connection: sqlite3.Connection, asset_ids: Sequence[str]
    ) -> dict[str, list[library_index.Occurrence]]:
        found: dict[str, list[library_index.Occurrence]] = {}
        if not asset_ids:
            return found
        placeholders = ",".join("?" * len(asset_ids))
        for row in connection.execute(
            "SELECT * FROM idx_occurrences WHERE asset_id IN (%s) "
            "ORDER BY first_ts, session_id, child_id" % placeholders,
            list(asset_ids),
        ):
            asset_id = str(row["asset_id"])
            found.setdefault(asset_id, []).append(
                library_index.Occurrence(
                    session_id=str(row["session_id"]),
                    title=str(row["title"] or ""),
                    origin=str(row["origin"] or "mentioned"),
                    child_id=str(row["child_id"]) or None,
                    label=row["label"],
                    tool=str(row["tool"] or ""),
                    intent=row["intent"],
                    first_ts=float(row["first_ts"] or 0.0),
                    last_ts=float(row["last_ts"] or 0.0),
                    first_seq=int(row["first_seq"] or 0),
                    last_seq=int(row["last_seq"] or 0),
                    touches=int(row["touches"] or 1),
                    highlighted=bool(row["highlighted"]),
                )
            )
        return found

    def counts(self, scope: str) -> dict[str, dict[str, int]]:
        """The sidebar counts for one scope, from ``idx_counts``."""
        counts: dict[str, dict[str, int]] = {bucket: {} for bucket in COUNT_BUCKETS}
        with self.read() as connection:
            for row in connection.execute(
                "SELECT bucket, value, n FROM idx_counts WHERE scope = ?", (scope,)
            ):
                bucket = str(row["bucket"])
                if bucket in counts:
                    counts[bucket][str(row["value"])] = int(row["n"])
        return counts

    def scoped_tags(self, scope: str) -> list[dict[str, Any]]:
        """Tag vocabulary restricted to a scope, with counts inside it.

        ``metadata.list_tags`` counts the whole store; a ``scope=final`` rail
        must not advertise tags only scratch assets carry.
        """
        clause = " WHERE i.is_final = 1" if scope == "final" else ""
        with self.read() as connection:
            rows = connection.execute(
                "SELECT t.id, t.name, t.category, COUNT(*) AS n FROM tags t "
                "JOIN asset_tags at ON at.tag_id = t.id "
                "JOIN idx_assets i ON i.asset_id = at.asset_id"
                + clause
                + " GROUP BY t.id, t.name, t.category ORDER BY t.name COLLATE NOCASE",
            )
            return [
                {
                    "id": str(row["id"]),
                    "name": str(row["name"]),
                    "category": str(row["category"]),
                    "count": int(row["n"]),
                }
                for row in rows
            ]

    def page_tags(self, asset_ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        """Tags for the assets on one page (a page, never the catalog)."""
        found: dict[str, list[dict[str, Any]]] = {}
        if not asset_ids:
            return found
        placeholders = ",".join("?" * len(asset_ids))
        with self.read() as connection:
            for row in connection.execute(
                "SELECT at.asset_id AS asset_id, t.id, t.name, t.category FROM asset_tags at "
                "JOIN tags t ON t.id = at.tag_id WHERE at.asset_id IN (%s) "
                "ORDER BY t.name COLLATE NOCASE" % placeholders,
                list(asset_ids),
            ):
                found.setdefault(str(row["asset_id"]), []).append(
                    {
                        "id": str(row["id"]),
                        "name": str(row["name"]),
                        "category": str(row["category"]),
                    }
                )
        return found

    def metadata_rows(self, asset_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """The authoritative ``assets`` rows for a page, for the projection."""
        found: dict[str, dict[str, Any]] = {}
        if not asset_ids:
            return found
        placeholders = ",".join("?" * len(asset_ids))
        with self.read() as connection:
            for row in connection.execute(
                "SELECT * FROM assets WHERE asset_id IN (%s)" % placeholders, list(asset_ids)
            ):
                found[str(row["asset_id"])] = dict(row)
        return found

    def primary_tags(self, asset_ids: Sequence[str]) -> dict[str, str]:
        found: dict[str, str] = {}
        if not asset_ids:
            return found
        with self.read() as connection:
            for asset_id in asset_ids:
                row = connection.execute(
                    "SELECT value FROM state WHERE key = ?", (f"primary_tag:{asset_id}",)
                ).fetchone()
                if row is not None:
                    found[asset_id] = str(row["value"])
        return found

    # -- thumbnails ---------------------------------------------------------

    def thumbnail_candidates(
        self, asset_ids: Iterable[str] | None = None, *, width: int = 480
    ) -> list[tuple[str, int]]:
        """``(asset_id, width)`` pairs whose cached WebP is missing at that width.

        ORDERED ``is_final DESC, last_ts DESC``: a migration run must spend its
        first work on the images the operator actually kept, not on the oldest
        scratch screenshot in the catalog.

        Staleness of the SOURCE is not decided here: that needs one stat per
        row, and the cheap filter ("no file on disk") is what keeps the sweep
        bounded. :meth:`ensure_thumbnail` re-checks ``src`` and regenerates when
        the source moved.
        """
        params = list(asset_ids or [])
        # No clause at all when there are no ids: SQLite accepts ``IN ()`` as
        # always-false, which silently turned the startup sweep into a no-op.
        clause = (" AND i.asset_id IN (%s)" % ",".join("?" * len(params))) if params else ""
        return self._missing_thumbnails(params, clause, width)

    def _missing_thumbnails(
        self, params: list[str], clause: str, width: int
    ) -> list[tuple[str, int]]:
        with self.read() as connection:
            rows = list(
                connection.execute(
                    "SELECT i.asset_id, t.path AS thumb_path FROM idx_assets i "
                    "LEFT JOIN idx_thumbnails t ON t.asset_id = i.asset_id AND t.width = ? "
                    "WHERE i.category = 'image'" + clause + " "
                    "ORDER BY i.is_final DESC, i.last_ts DESC",
                    [width, *params],
                )
            )
        return [
            (str(row["asset_id"]), width)
            for row in rows
            if not row["thumb_path"] or not os.path.isfile(str(row["thumb_path"]))
        ]

    def thumbnail_source(self, asset_id: str) -> tuple[Path, str, str] | None:
        """``(source path, content type, suffix)`` for an image asset, or ``None``.

        A digest artifact has no path of its own: the attachment store is where
        its bytes live, and that is what gets resized.
        """
        row = self._thumb_row(asset_id)
        if row is None or str(row["kind"]) != "image":
            return None
        name = str(row["derived_name"] or "")
        digest = row["digest"]
        if digest:
            from local_operator.session.attachments import AttachmentStore

            path = AttachmentStore(self.config_dir / "attachments").path_for(str(digest))
        elif row["path"]:
            path = Path(str(row["path"]))
        else:
            return None
        return (
            path,
            str(row["content_type"] or "application/octet-stream"),
            Path(name).suffix.lower(),
        )

    def _thumb_row(self, asset_id: str) -> sqlite3.Row | None:
        with self.read() as connection:
            return connection.execute(
                "SELECT asset_id, path, digest, derived_name, kind, content_type "
                "FROM idx_assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()

    def thumb_is_live(self, asset_id: str, width: int) -> Path | None:
        """The cached file for THIS width when it is present and still matches."""
        with self.read() as connection:
            row = connection.execute(
                "SELECT path, key, src FROM idx_thumbnails WHERE asset_id = ? AND width = ?",
                (asset_id, width),
            ).fetchone()
        if row is None or not row["path"] or not row["src"]:
            return None
        source = self.thumbnail_source(asset_id)
        if source is None:
            return None
        path, _content_type, suffix = source
        if suffix in (".svg", ".gif"):
            return None
        target = Path(str(row["path"]))
        if not target.is_file():
            return None
        # ``src`` is the source's stat identity — one stat, never a content
        # hash: re-reading a 4 MB image per grid request is the cost this cache
        # exists to remove.
        return target if str(row["src"]) == thumb_source_stamp(path) else None

    def ensure_thumbnail(self, asset_id: str, *, width: int, quality: int) -> str | None:
        """Generate the cached WebP for one asset at one width.

        NEVER on a request path: the caller is the indexer's thumbnail thread.
        A no-op when a live file already exists at that width, and one row per
        ``(asset, width)`` — a single slot per asset could only ever serve the
        width generated last, which is what handed ``w=240`` callers the
        full-size original.
        """
        live = self.thumb_is_live(asset_id, width)
        if live is not None:
            return str(live)
        source = self.thumbnail_source(asset_id)
        if source is None:
            return None
        path, _content_type, suffix = source
        if suffix in (".svg", ".gif") or not path.is_file():
            # Rasterising an animated GIF or a vector loses the thing the user
            # wants; those are served original by the route.
            return None
        Image = _image_module()
        if Image is None:
            # Pillow is an optional extra. Serving the original is honest.
            return None
        cache_dir = self.config_dir / "cache" / "library" / "thumbs"
        key = _thumb_key(path, width)
        target = cache_dir / f"{key}.webp"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                image.load()
                if image.width > width:
                    height = max(1, round(image.height * width / image.width))
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA")
                # Atomic: a reader must never see a half-written WebP.
                temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
                image.save(temp, "WEBP", quality=quality)
                os.replace(temp, target)
        except Exception:  # noqa: BLE001 - a thumbnail is cosmetic; the row still works
            logger.debug("library: thumbnail failed for %s", asset_id, exc_info=True)
            return None
        stamp = thumb_source_stamp(path)
        with self.write() as connection:
            connection.execute(
                "INSERT INTO idx_thumbnails(asset_id, width, path, key, src, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(asset_id, width) DO UPDATE SET path = excluded.path, "
                "key = excluded.key, src = excluded.src, created_at = excluded.created_at",
                (asset_id, width, str(target), key, stamp, time.time()),
            )
        return str(target)

    def thumb_size(self) -> tuple[int, int]:
        """``(files, bytes)`` of the thumbnail cache, for the build log."""
        root = self.config_dir / "cache" / "library" / "thumbs"
        files = 0
        total = 0
        try:
            for entry in root.iterdir():
                try:
                    info = entry.stat()
                except OSError:
                    continue
                if info.st_size:
                    files += 1
                    total += info.st_size
        except OSError:
            return (0, 0)
        return (files, total)

    def refresh_session_title(
        self, connection: sqlite3.Connection, session_id: str, title: str
    ) -> set[str]:
        """Rewrite the denormalised title for one session; return affected assets.

        A ``/rename`` writes only the title sidecar, so no journal signature
        moves. Without this the Library would keep showing the old name until
        the conversation was used again.
        """
        affected = {
            str(row["asset_id"])
            for row in connection.execute(
                "SELECT DISTINCT asset_id FROM idx_occurrences WHERE session_id = ?",
                (session_id,),
            )
        }
        connection.execute(
            "UPDATE idx_occurrences SET title = ? WHERE session_id = ?", (title, session_id)
        )
        connection.execute(
            "UPDATE idx_sessions SET title = ?, title_key = ? WHERE session_id = ?",
            (title, title, session_id),
        )
        return affected

    def occurrence_assets(self, session_id: str) -> set[str]:
        """Asset ids that currently have an occurrence in one session."""
        with self.read() as connection:
            return {
                str(row["asset_id"])
                for row in connection.execute(
                    "SELECT DISTINCT asset_id FROM idx_occurrences WHERE session_id = ?",
                    (session_id,),
                )
            }

    def incomplete_sessions(self) -> int:
        """Sessions still owed a scan, from the state counters reconcile writes."""
        state = self.state()
        total = self._int_state(state, STATE_SESSIONS_TOTAL)
        with self.read() as connection:
            scanned = self._count_sessions(connection, complete=True)
        return max(0, total - scanned)

    def object(self, asset_id: str) -> library_index.Asset | None:
        """Alias kept explicit for callers that want the id lookup, not a page."""
        return self.asset(asset_id)


# ---------------------------------------------------------------------------
# SQL fragments
# ---------------------------------------------------------------------------


def _image_module() -> Any:
    """Pillow's ``Image`` module, or ``None`` when the optional extra is absent.

    A named seam rather than an inline import: the thumbnail tests need a fake
    decoder, and swapping ``sys.modules["PIL"]`` for one leaked into later tests
    (a real generation then fell back to "no Pillow").
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    return Image


def thumb_source_stamp(path: Path) -> str:
    """Identity of the SOURCE bytes: a moved file invalidates its thumbnail.

    Size plus nanosecond mtime, not a content hash: this is computed on every
    thumbnail READ to decide liveness, and hashing a 4 MB image per grid row is
    exactly the cost the off-request-path design exists to remove.
    """
    try:
        info = path.stat()
    except OSError:
        return ""
    return hashlib.sha256(
        f"{path}:{int(info.st_size)}:{int(info.st_mtime_ns)}".encode("utf-8")
    ).hexdigest()[:16]


def _thumb_key(path: Path, width: int) -> str:
    """Cache key: the source CONTENT hash (first 16 hex) plus the width.

    Content, not path, because two sessions can reference the same image by
    different spellings and one resized WebP serves both. Reading the file is
    affordable here and only here — generation already opens it.
    """
    try:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()[:16]
    except OSError:
        return ""
    return f"{digest}-{width}"


def _where_clause(query: ListingQuery) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if query.scope == "final":
        where.append("i.is_final = 1")
    for column, values in (
        ("category", query.categories),
        ("kind", query.kinds),
        ("origin_group", query.origin_groups),
    ):
        if values:
            where.append(f"i.{column} IN ({','.join('?' * len(values))})")
            params.extend(sorted(values))
    if query.origins:
        where.append(
            "EXISTS (SELECT 1 FROM json_each(i.origins) WHERE json_each.value IN (%s))"
            % ",".join("?" * len(query.origins))
        )
        params.extend(sorted(query.origins))
    if query.highlighted is not None:
        where.append("i.highlighted = ?")
        params.append(1 if query.highlighted else 0)
    if query.favorite is not None:
        where.append("COALESCE(m.favorite, 0) = ?")
        params.append(1 if query.favorite else 0)
    if query.organized is not None:
        where.append(
            "(m.organized_at IS NOT NULL AND m.organized_at != 0) = ?"
            if query.organized
            else "(m.organized_at IS NULL OR m.organized_at = 0)"
        )
        if query.organized:
            params.append(1)
    if query.tag:
        # One tag id today (``Query.tag``), so ``tag_mode`` is a no-op here; the
        # EXISTS form is what a multi-tag ``tag_mode=any`` would extend.
        where.append(
            "EXISTS (SELECT 1 FROM asset_tags at WHERE at.asset_id = i.asset_id AND at.tag_id = ?)"
        )
        params.append(query.tag)
    session_ids = set(query.session_ids)
    if query.session:
        session_ids.add(query.session)
    if session_ids:
        where.append(
            "EXISTS (SELECT 1 FROM idx_occurrences o WHERE o.asset_id = i.asset_id "
            "AND o.session_id IN (%s))" % ",".join("?" * len(session_ids))
        )
        params.extend(sorted(session_ids))
    return where, params


_ORDER_SQL: Final = {
    "recent": "i.last_ts",
    "modified": "i.last_ts",
    "name": "i.effective_name COLLATE NOCASE",
    "kind": "i.kind",
    "size": "COALESCE(i.size_bytes, 0)",
    "created": "i.first_ts",
    "organized": "COALESCE(m.organized_at, 0)",
}


def _order_clause(sort: str, order: str | None) -> str:
    """``ORDER BY`` for one sort, keeping the old ``(key, asset_id)`` tie-break.

    ``order`` decides BOTH directions, exactly as the old ``reverse=`` did: the
    tie-break reversed with the sort key, so a page boundary stayed stable.

    NAME ORDER IS ASCII-ONLY. ``COLLATE NOCASE`` folds A-Z and nothing else,
    where the removed Python sort used ``str.lower()`` and ordered Unicode case
    pairs ("École" before "école" here, after it there). Sorting in SQL is what
    makes a name sort an index scan instead of a full materialisation; a
    Unicode fold would need a stored ``casefold`` column kept in sync on every
    rename. Documented here, pinned by a test.
    """
    # Mirrors ``service.DEFAULT_ORDER``: "name" and "kind" are A-Z by default
    # and everything else is newest/largest first. A caller that passes a Query
    # supplies ``effective_order`` and never reaches this fallback.
    default = "asc" if sort in ("name", "kind") else "desc"
    direction = (order or default).upper()
    if direction not in ("ASC", "DESC"):
        direction = "ASC"
    terms = [_ORDER_SQL.get(sort, _ORDER_SQL["recent"])]
    if sort == "kind":
        # The old key was ``(kind, name.lower(), id)``: a kind bucket is a name
        # list, not an arbitrary order.
        terms.append("i.effective_name COLLATE NOCASE")
    terms.append("i.asset_id")
    return " ORDER BY " + ", ".join(f"{term} {direction}" for term in terms)


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def _effective_from_row(value: Any, source: Any, derived: str) -> str:
    """user > librarian > agent > derived, in Python (see ``service._effective``)."""
    if isinstance(value, str) and value and source in ("user", "librarian", "agent"):
        return value
    return derived or ""


def is_final(asset: library_index.Asset) -> bool:
    """The operator's kept set: a highlight, or an artifact the agent produced.

    Mentioned and merely-created scratch files are excluded from the ``final``
    scope; ``all`` still indexes and serves them. Defined here rather than in
    the service because the INDEXER has to write it into ``idx_assets.is_final``
    and the service has to filter on it — two spellings would drift.
    """
    return bool(asset.highlighted or asset.origin_group == "generated")


def _aggregate(
    asset_id: str,
    key: str,
    rows: Sequence[sqlite3.Row],
) -> library_index.Asset | None:
    """One asset from its occurrence rows — the SQL-side ``index._build_assets``.

    Deliberately the same merge rules, in the same order: first non-null ``path``
    / ``size`` / ``width`` / ``height``, last non-null ``duration_s``, strongest
    origin wins the row's ``origin`` while ``origins`` keeps every raw origin.
    """
    asset = library_index.Asset(key=key or _key_from_rows(rows), id=asset_id)
    if not asset.key:
        return None
    derived = ""
    for row in rows:
        path = row["path"]
        if asset.path is None and path:
            asset.path = str(path)
        if asset.digest is None and key.startswith("a:"):
            asset.digest = key[2:]
        if row["name"]:
            derived = str(row["name"])
        classification = classify.classify(derived or asset.path or "", row["content_type"])
        asset.kind = classification.kind
        asset.category = classification.category
        asset.content_type = classification.content_type
        for target, column in (
            ("size_bytes", "size_bytes"),
            ("width", "width"),
            ("height", "height"),
        ):
            value = row[column]
            if value is not None and getattr(asset, target) is None:
                setattr(asset, target, value)
        if row["duration_s"] is not None:
            asset.duration_s = float(row["duration_s"])
        origin = str(row["origin"] or "mentioned")
        if origin not in library_index.ORIGINS:
            origin = "mentioned"
        occurrence = library_index.Occurrence(
            session_id=str(row["session_id"]),
            title=str(row["title"] or ""),
            origin=origin,
            child_id=str(row["child_id"]) or None,
            label=row["label"],
            tool=str(row["tool"] or ""),
            intent=row["intent"],
            first_ts=float(row["first_ts"] or 0.0),
            last_ts=float(row["last_ts"] or 0.0),
            first_seq=int(row["first_seq"] or 0),
            last_seq=int(row["last_seq"] or 0),
            touches=int(row["touches"] or 1),
            highlighted=bool(row["highlighted"]),
        )
        asset.occurrences.append(occurrence)
        if origin not in asset.origins:
            asset.origins.append(origin)
        asset.highlighted = asset.highlighted or occurrence.highlighted or origin == "highlight"
    if not asset.derived_name:
        asset.derived_name = derived or _basename(asset.path) or (asset.digest or "") or asset.id
    if asset.kind == "other" and asset.digest:
        classification = classify.classify(asset.derived_name, asset.content_type)
        asset.kind, asset.category = classification.kind, classification.category
    # ``_build_assets`` sorted the raw origins by their vocabulary order, which
    # is what makes ``origin`` ("strongest first") the first element.
    asset.origins.sort(key=library_index.ORIGINS.index)
    return asset


def _key_from_rows(rows: Sequence[sqlite3.Row]) -> str:
    """Recover a key when the idx row was never written (a rebuild edge case)."""
    for row in rows:
        try:
            return identity.key_for(row["path"], None)
        except ValueError:
            continue
    return ""


def _basename(path: str | None) -> str:
    return os.path.basename(path) if path else ""


def _asset_from_row(
    row: sqlite3.Row, occurrences: Sequence[library_index.Occurrence]
) -> library_index.Asset:
    """Rebuild the :class:`Asset` an existing projection expects from an idx row."""
    asset = library_index.Asset(
        key=str(row["key"]),
        id=str(row["asset_id"]),
        path=row["path"],
        digest=row["digest"],
        derived_name=str(row["derived_name"] or ""),
        kind=str(row["kind"] or "other"),
        category=str(row["category"] or "other"),
        content_type=str(row["content_type"] or "application/octet-stream"),
        size_bytes=row["size_bytes"],
        width=row["width"],
        height=row["height"],
        duration_s=row["duration_s"],
        origins=list(json.loads(row["origins"] or "[]")),
        highlighted=bool(row["highlighted"]),
    )
    asset.occurrences = list(occurrences)
    return asset


__all__ = [
    "DB_PATH_PARTS",
    "IndexUnavailable",
    "LibraryStore",
    "SCHEMA_VERSION",
    "STATE_BACKFILL_DONE",
    "STATE_BUILT_AT",
    "STATE_LAST_RECONCILE",
    "STATE_REVISION",
    "STATE_SESSIONS_SCANNED",
    "STATE_SESSIONS_TOTAL",
    "fts_expression",
    "is_final",
    "search_tokens",
]

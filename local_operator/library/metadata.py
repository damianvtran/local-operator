"""Authoritative Library metadata in SQLite (design §3.1).

WHY SQLITE AND NOT THE DERIVED INDEX. The index is rebuilt whenever a journal
changes; anything the user typed must survive that rebuild and must survive the
file disappearing entirely (a returning file keeps its name, description and
tags). So this store is the source of truth for everything the user or the
Librarian decided, and the index only contributes derived names.

CONCURRENCY. One connection per call, never a shared connection: FastAPI runs
handlers on several threads, and sqlite3 connections are not safely shareable
across them. WAL plus ``busy_timeout=5000`` gives readers a concurrent writer
and makes a short write burst queue rather than fail. Every write is one
``BEGIN IMMEDIATE`` transaction; a multi-row tag replacement can never be
observed half-applied.

FAILURE POLICY. A corrupt, unreadable or FUTURE-version database answers an
empty catalog on every read instead of raising into a request. Writes raise
:class:`MetadataUnavailable`, which the route turns into a typed refusal: a
user edit that cannot be recorded must not silently vanish.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final = 1
DB_DIRNAME: Final = "library"
DB_FILENAME: Final = "library.sqlite3"

MAX_TAGS_PER_ASSET: Final = 12
TAG_CATEGORIES: Final = ("topic", "project", "format", "status", "source", "custom")
TAG_CREATORS: Final = ("user", "librarian", "system")
FIELD_SOURCES: Final = ("user", "librarian", "derived")

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS assets(
    asset_id TEXT PRIMARY KEY, key TEXT UNIQUE NOT NULL,
    name TEXT, description TEXT, name_source TEXT, description_source TEXT,
    favorite INTEGER NOT NULL DEFAULT 0, locked INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    organized_at REAL, organized_by TEXT, confidence REAL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tags(
    id TEXT PRIMARY KEY, name TEXT UNIQUE COLLATE NOCASE NOT NULL,
    category TEXT NOT NULL DEFAULT 'custom', created_by TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS asset_tags(
    asset_id TEXT NOT NULL, tag_id TEXT NOT NULL,
    source TEXT NOT NULL, confidence REAL, created_at REAL NOT NULL,
    PRIMARY KEY(asset_id, tag_id));
CREATE TABLE IF NOT EXISTS chats(
    asset_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
    created_at REAL NOT NULL, last_used_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS chat_history(
    asset_id TEXT NOT NULL, session_id TEXT NOT NULL,
    created_at REAL NOT NULL, last_used_at REAL NOT NULL,
    PRIMARY KEY(asset_id, session_id));
CREATE TABLE IF NOT EXISTS conventions(
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
    created_by TEXT, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS edits(
    id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id TEXT NOT NULL,
    actor TEXT NOT NULL, fields TEXT NOT NULL, before TEXT, after TEXT,
    at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

#: Primary tags live in ``state`` because §3.1 freezes the table set and
#: ``asset_tags`` has no primary column. One key per asset is the assignment;
#: deleting/merging the tag clears it. This is the same "small durable fact in
#: an existing row" pattern as the schema version, without a second table.
_PRIMARY_PREFIX: Final = "primary_tag:"


class MetadataError(Exception):
    """Base for metadata refusals a route can turn into a status code."""


class MetadataUnavailable(MetadataError):
    """The store cannot be read or written (corrupt, future schema, I/O)."""


class TagRefused(MetadataError):
    """A tag operation violates a frozen rule (limit, duplicate, unknown id)."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def _now() -> float:
    return time.time()


def _bool(value: Any) -> bool:
    return bool(value)


class MetadataStore:
    """The authoritative metadata store. Construct per config root, cheap.

    Thread-safe by construction: each method opens its own connection, and a
    process-local lock serialises the schema bootstrap (two first requests in
    one process must not both run ``CREATE TABLE`` against an empty file).
    """

    _bootstrap_lock = threading.Lock()

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / DB_DIRNAME / DB_FILENAME

    # -- connection / schema -------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection ready for one call. ``MetadataUnavailable`` on any failure."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5.0)
        except (OSError, sqlite3.Error) as error:
            raise MetadataUnavailable(str(error)) from error
        try:
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA busy_timeout=5000")
                connection.execute("PRAGMA foreign_keys=OFF")
            except sqlite3.DatabaseError as error:
                # A corrupt file may fail here (``file is not a database``).
                raise MetadataUnavailable(str(error)) from error
            yield connection
        finally:
            try:
                connection.close()
            except sqlite3.Error:
                pass

    def _ensure_schema(self, connection: sqlite3.Connection) -> bool:
        """Create/verify the schema. ``False`` means "serve an empty catalog"."""
        with self._bootstrap_lock:
            try:
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM state WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO state(key, value) VALUES('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                    connection.commit()
                    return True
                try:
                    version = int(row["value"])
                except (TypeError, ValueError):
                    raise MetadataUnavailable("corrupt schema version")
                if version > SCHEMA_VERSION:
                    # A newer local-operator wrote this store. Guessing at its
                    # columns is how a downgrade corrupts data; answer empty.
                    return False
                return True
            except MetadataUnavailable:
                return False
            except sqlite3.Error as error:
                logger.warning("library metadata is unavailable", exc_info=True)
                logger.debug("library metadata schema failure: %s", error)
                return False

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection | None]:
        try:
            with self._connect() as connection:
                if not self._ensure_schema(connection):
                    yield None
                    return
                yield connection
        except MetadataUnavailable:
            yield None

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._connect() as connection:
                if not self._ensure_schema(connection):
                    raise MetadataUnavailable("future schema version")
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
        except MetadataUnavailable:
            raise
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error

    # -- assets --------------------------------------------------------------

    def ensure_asset(
        self,
        asset_id: str,
        key: str,
        *,
        derived_name: str | None = None,
        derived_description: str | None = None,
        at: float | None = None,
    ) -> dict[str, Any] | None:
        """Create the row for an asset the index knows about.

        A derived field is written ONLY while it is still derived (or absent):
        the user's or Librarian's value always wins, which is the design's
        effective-name rule applied at the source instead of at every read.
        """
        stamp = _now() if at is None else at
        try:
            with self._write() as connection:
                row = connection.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO assets(asset_id, key, name, description, "
                        "name_source, description_source, created_at, updated_at) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            asset_id,
                            key,
                            derived_name,
                            derived_description,
                            "derived" if derived_name else None,
                            "derived" if derived_description else None,
                            stamp,
                            stamp,
                        ),
                    )
                else:
                    if derived_name is not None and row["name_source"] in (None, "derived"):
                        connection.execute(
                            "UPDATE assets SET name = ?, name_source = 'derived', updated_at = ? "
                            "WHERE asset_id = ?",
                            (derived_name, stamp, asset_id),
                        )
                    if derived_description is not None and row["description_source"] in (
                        None,
                        "derived",
                    ):
                        connection.execute(
                            "UPDATE assets SET description = ?, description_source = 'derived', "
                            "updated_at = ? WHERE asset_id = ?",
                            (derived_description, stamp, asset_id),
                        )
        except MetadataUnavailable:
            return None
        return self.asset(asset_id)

    def ensure_assets(
        self,
        items: Iterable[tuple[str, str, str | None]],
        *,
        at: float | None = None,
    ) -> None:
        """Upsert many index-derived rows in ONE transaction.

        A list request knows every asset in the page and would otherwise open a
        connection per row; that is the difference between a page costing one
        transaction and costing 200. Derived fields refresh only while their
        source is still ``derived`` — the effective-name rule, applied once at
        write time instead of re-checked by every reader.
        """
        rows = [(asset_id, key, name) for asset_id, key, name in items]
        if not rows:
            return
        stamp = _now() if at is None else at
        try:
            with self._write() as connection:
                for asset_id, key, name in rows:
                    existing = connection.execute(
                        "SELECT name_source, description_source FROM assets WHERE asset_id = ?",
                        (asset_id,),
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            "INSERT INTO assets(asset_id, key, name, name_source, "
                            "created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                            (asset_id, key, name, "derived" if name else None, stamp, stamp),
                        )
                    elif name is not None and existing["name_source"] in (None, "derived"):
                        connection.execute(
                            "UPDATE assets SET name = ?, name_source = 'derived', updated_at = ? "
                            "WHERE asset_id = ?",
                            (name, stamp, asset_id),
                        )
        except MetadataUnavailable:
            # The index is the only caller; an unavailable store leaves the rows
            # derived at projection time. Never fail a list because of it.
            return

    def asset(self, asset_id: str) -> dict[str, Any] | None:
        with self._read() as connection:
            if connection is None:
                return None
            row = connection.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            return self._asset_row(row)

    def assets(self, asset_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return {}
        found: dict[str, dict[str, Any]] = {}
        with self._read() as connection:
            if connection is None:
                return {}
            # SQLite's default parameter cap is 999; chunking keeps a large
            # page (limit 200) and a stats call well inside it.
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                for row in connection.execute(
                    f"SELECT * FROM assets WHERE asset_id IN ({placeholders})", chunk
                ):
                    found[row["asset_id"]] = self._asset_row(row)
        return found

    @staticmethod
    def _asset_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "asset_id": row["asset_id"],
            "key": row["key"],
            "name": row["name"],
            "description": row["description"],
            "name_source": row["name_source"],
            "description_source": row["description_source"],
            "favorite": _bool(row["favorite"]),
            "locked": _bool(row["locked"]),
            "archived": _bool(row["archived"]),
            "organized_at": row["organized_at"],
            "organized_by": row["organized_by"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_fields(
        self,
        asset_id: str,
        fields: Mapping[str, Any],
        *,
        actor: str,
        source: str | None = None,
        at: float | None = None,
    ) -> dict[str, Any] | None:
        """Write user/librarian fields, recording provenance and an edit row.

        ``source`` defaults to the actor (``user`` edits set ``user``; the
        Librarian sets ``librarian``). ``locked`` is honoured HERE for agent
        writes: the route and the tool both pass ``actor='librarian'``, and a
        locked asset accepts only ``archived``/``locked`` changes.
        """
        if actor not in TAG_CREATORS:
            raise TagRefused("unknown_actor", actor)
        provenance = source if source in FIELD_SOURCES else actor
        stamp = _now() if at is None else at
        allowed = {"name", "description", "favorite", "locked", "archived", "organized_at",
                   "organized_by", "confidence"}
        unknown = set(fields) - allowed
        if unknown:
            raise TagRefused("unknown_field", ",".join(sorted(unknown)))
        try:
            with self._write() as connection:
                before = connection.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
                ).fetchone()
                if before is None:
                    return None
                if (
                    actor == "librarian"
                    and before["locked"]
                    and not set(fields) <= {"locked", "archived"}
                ):
                    raise TagRefused("asset_locked")
                assignments: list[str] = []
                values: list[Any] = []
                for field, value in fields.items():
                    if field == "name":
                        if not isinstance(value, str) or not value.strip():
                            raise TagRefused("name_required")
                        assignments.append("name = ?")
                        values.append(value.strip()[:300])
                        assignments.append("name_source = ?")
                        values.append(provenance)
                    elif field == "description":
                        if value is not None and not isinstance(value, str):
                            raise TagRefused("description_invalid")
                        assignments.append("description = ?")
                        values.append((value or "").strip()[:4000] or None)
                        assignments.append("description_source = ?")
                        values.append(provenance)
                    elif field in ("favorite", "locked", "archived"):
                        assignments.append(f"{field} = ?")
                        values.append(1 if value else 0)
                    elif field == "organized_at":
                        assignments.append("organized_at = ?")
                        values.append(value)
                    elif field == "organized_by":
                        assignments.append("organized_by = ?")
                        values.append(value)
                    elif field == "confidence":
                        assignments.append("confidence = ?")
                        values.append(value)
                if not assignments:
                    return self._asset_row(before)
                assignments.append("updated_at = ?")
                values.append(stamp)
                values.append(asset_id)
                connection.execute(
                    f"UPDATE assets SET {', '.join(assignments)} WHERE asset_id = ?", values
                )
                after = connection.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
                ).fetchone()
                connection.execute(
                    "INSERT INTO edits(asset_id, actor, fields, before, after, at) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        asset_id,
                        actor,
                        json.dumps(sorted(fields), separators=(",", ":")),
                        json.dumps(self._asset_row(before), sort_keys=True),
                        json.dumps(self._asset_row(after), sort_keys=True),
                        stamp,
                    ),
                )
                return self._asset_row(after)
        except TagRefused:
            raise
        except MetadataUnavailable:
            raise
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error

    # -- tags ---------------------------------------------------------------

    def tag_by_id(self, tag_id: str) -> dict[str, Any] | None:
        with self._read() as connection:
            if connection is None:
                return None
            row = connection.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
            return self._tag_row(row)

    @staticmethod
    def _tag_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "category": row["category"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_tags(self) -> list[dict[str, Any]]:
        with self._read() as connection:
            if connection is None:
                return []
            rows = connection.execute(
                "SELECT t.*, COUNT(a.asset_id) AS count FROM tags t "
                "LEFT JOIN asset_tags a ON a.tag_id = t.id "
                "GROUP BY t.id ORDER BY LOWER(t.name)"
            )
            return [{**(self._tag_row(row) or {}), "count": int(row["count"])} for row in rows]

    def create_tag(
        self, name: str, category: str = "custom", *, actor: str = "user"
    ) -> dict[str, Any]:
        tag_name = _normalise_tag_name(name)
        if category not in TAG_CATEGORIES:
            raise TagRefused("bad_category", category)
        if actor not in TAG_CREATORS:
            raise TagRefused("unknown_actor", actor)
        stamp = _now()
        tag_id = _tag_id(tag_name)
        try:
            with self._write() as connection:
                existing = connection.execute(
                    "SELECT * FROM tags WHERE name = ? COLLATE NOCASE", (tag_name,)
                ).fetchone()
                if existing is not None:
                    return self._tag_row(existing) or {}
                connection.execute(
                    "INSERT INTO tags(id, name, category, created_by, created_at, updated_at) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (tag_id, tag_name, category, actor, stamp, stamp),
                )
                row = connection.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
                return self._tag_row(row) or {}
        except sqlite3.IntegrityError as error:
            # Two concurrent creates: the loser reads the winner's row.
            existing = self.tag_by_id(tag_id)
            if existing is not None:
                return existing
            raise TagRefused("tag_exists", tag_name) from error

    def rename_tag(self, tag_id: str, name: str) -> dict[str, Any]:
        tag_name = _normalise_tag_name(name)
        try:
            with self._write() as connection:
                row = connection.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
                if row is None:
                    raise TagRefused("tag_not_found", tag_id)
                other = connection.execute(
                    "SELECT id FROM tags WHERE name = ? COLLATE NOCASE AND id != ?",
                    (tag_name, tag_id),
                ).fetchone()
                if other is not None:
                    raise TagRefused("tag_exists", tag_name)
                connection.execute(
                    "UPDATE tags SET name = ?, updated_at = ? WHERE id = ?",
                    (tag_name, _now(), tag_id),
                )
                updated = connection.execute(
                    "SELECT * FROM tags WHERE id = ?", (tag_id,)
                ).fetchone()
                return self._tag_row(updated) or {}
        except MetadataError:
            raise
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error

    def delete_tag(self, tag_id: str) -> bool:
        try:
            with self._write() as connection:
                row = connection.execute("SELECT id FROM tags WHERE id = ?", (tag_id,)).fetchone()
                if row is None:
                    return False
                connection.execute("DELETE FROM asset_tags WHERE tag_id = ?", (tag_id,))
                connection.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
                for state_row in connection.execute(
                    "SELECT key, value FROM state WHERE key LIKE ?", (f"{_PRIMARY_PREFIX}%",)
                ):
                    if state_row["value"] == tag_id:
                        connection.execute("DELETE FROM state WHERE key = ?", (state_row["key"],))
                return True
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error

    def merge_tags(self, tag_id: str, into: str) -> dict[str, Any]:
        """Move every assignment from ``tag_id`` to ``into`` and delete the source.

        Assignments already present on the target keep their original source
        (``INSERT OR IGNORE``), and a primary pointing at the source follows.
        """
        if tag_id == into:
            raise TagRefused("tag_merge_self")
        try:
            with self._write() as connection:
                source = connection.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
                target = connection.execute("SELECT * FROM tags WHERE id = ?", (into,)).fetchone()
                if source is None or target is None:
                    raise TagRefused("tag_not_found")
                connection.execute(
                    "INSERT OR IGNORE INTO asset_tags(asset_id, tag_id, source, confidence, "
                    "created_at) SELECT asset_id, ?, source, confidence, created_at "
                    "FROM asset_tags WHERE tag_id = ?",
                    (into, tag_id),
                )
                connection.execute("DELETE FROM asset_tags WHERE tag_id = ?", (tag_id,))
                connection.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
                for state_row in connection.execute(
                    "SELECT key, value FROM state WHERE key LIKE ?", (f"{_PRIMARY_PREFIX}%",)
                ):
                    if state_row["value"] == tag_id:
                        connection.execute(
                            "UPDATE state SET value = ? WHERE key = ?", (into, state_row["key"])
                        )
                updated = connection.execute("SELECT * FROM tags WHERE id = ?", (into,)).fetchone()
                return self._tag_row(updated) or {}
        except MetadataError:
            raise
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error

    def asset_tags(self, asset_id: str) -> list[dict[str, Any]]:
        with self._read() as connection:
            if connection is None:
                return []
            rows = connection.execute(
                "SELECT t.*, a.source, a.confidence FROM asset_tags a "
                "JOIN tags t ON t.id = a.tag_id WHERE a.asset_id = ? ORDER BY LOWER(t.name)",
                (asset_id,),
            )
            return [{**(self._tag_row(row) or {}), "source": row["source"],
                     "confidence": row["confidence"]} for row in rows]

    def tags_for_assets(self, asset_ids: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return {}
        result: dict[str, list[dict[str, Any]]] = {asset_id: [] for asset_id in ids}
        with self._read() as connection:
            if connection is None:
                return result
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT a.asset_id, t.* FROM asset_tags a JOIN tags t ON t.id = a.tag_id "
                    f"WHERE a.asset_id IN ({placeholders}) ORDER BY LOWER(t.name)",
                    chunk,
                )
                for row in rows:
                    result.setdefault(row["asset_id"], []).append(self._tag_row(row) or {})
        return result

    def primary_tags(self, asset_ids: Iterable[str]) -> dict[str, str]:
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return {}
        found: dict[str, str] = {}
        with self._read() as connection:
            if connection is None:
                return {}
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                keys = [f"{_PRIMARY_PREFIX}{asset_id}" for asset_id in chunk]
                key_placeholders = ",".join("?" for _ in keys)
                for row in connection.execute(
                    f"SELECT key, value FROM state WHERE key IN ({key_placeholders})", keys
                ):
                    found[row["key"][len(_PRIMARY_PREFIX) :]] = row["value"]
        return found

    def set_asset_tags(
        self,
        asset_id: str,
        tag_ids: Sequence[str],
        *,
        primary: str | None = None,
        actor: str = "user",
        confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """Replace an asset's tag set. All-or-nothing, bounded, validated."""
        unique = list(dict.fromkeys(str(tag_id) for tag_id in tag_ids if tag_id))
        if len(unique) > MAX_TAGS_PER_ASSET:
            raise TagRefused("too_many_tags", str(len(unique)))
        if primary is not None and primary not in unique:
            raise TagRefused("primary_not_assigned")
        stamp = _now()
        try:
            with self._write() as connection:
                asset = connection.execute(
                    "SELECT asset_id FROM assets WHERE asset_id = ?", (asset_id,)
                ).fetchone()
                if asset is None:
                    raise TagRefused("asset_not_found")
                if unique:
                    placeholders = ",".join("?" for _ in unique)
                    known = {
                        row["id"]
                        for row in connection.execute(
                            f"SELECT id FROM tags WHERE id IN ({placeholders})", unique
                        )
                    }
                    missing = [tag_id for tag_id in unique if tag_id not in known]
                    if missing:
                        raise TagRefused("tag_not_found", ",".join(missing[:3]))
                connection.execute("DELETE FROM asset_tags WHERE asset_id = ?", (asset_id,))
                connection.executemany(
                    "INSERT INTO asset_tags(asset_id, tag_id, source, confidence, created_at) "
                    "VALUES(?, ?, ?, ?, ?)",
                    [(asset_id, tag_id, actor, confidence, stamp) for tag_id in unique],
                )
                if primary is None:
                    connection.execute(
                        "DELETE FROM state WHERE key = ?", (f"{_PRIMARY_PREFIX}{asset_id}",)
                    )
                else:
                    connection.execute(
                        "INSERT INTO state(key, value) VALUES(?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (f"{_PRIMARY_PREFIX}{asset_id}", primary),
                    )
                connection.execute(
                    "INSERT INTO edits(asset_id, actor, fields, before, after, at) "
                    "VALUES(?, ?, ?, NULL, ?, ?)",
                    (
                        asset_id,
                        actor,
                        json.dumps(["tags"], separators=(",", ":")),
                        json.dumps(unique),
                        stamp,
                    ),
                )
        except MetadataError:
            raise
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error
        return self.asset_tags(asset_id)

    # -- conventions --------------------------------------------------------

    def conventions(self) -> list[dict[str, Any]]:
        with self._read() as connection:
            if connection is None:
                return []
            rows = connection.execute("SELECT * FROM conventions ORDER BY kind, id")
            return [dict(row) for row in rows]

    def set_convention(
        self, convention_id: str, kind: str, body: str, *, actor: str = "librarian"
    ) -> dict[str, Any]:
        if not convention_id or not kind:
            raise TagRefused("convention_required")
        stamp = _now()
        try:
            with self._write() as connection:
                connection.execute(
                    "INSERT INTO conventions(id, kind, body, created_by, updated_at) "
                    "VALUES(?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, "
                    "body = excluded.body, created_by = excluded.created_by, "
                    "updated_at = excluded.updated_at",
                    (convention_id, kind, body, actor, stamp),
                )
                row = connection.execute(
                    "SELECT * FROM conventions WHERE id = ?", (convention_id,)
                ).fetchone()
                return dict(row)
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error

    # -- chat link ----------------------------------------------------------

    def set_chat(self, asset_id: str, session_id: str, *, at: float | None = None) -> dict[str, Any]:
        stamp = _now() if at is None else at
        try:
            with self._write() as connection:
                asset = connection.execute(
                    "SELECT asset_id FROM assets WHERE asset_id = ?", (asset_id,)
                ).fetchone()
                if asset is None:
                    raise TagRefused("asset_not_found")
                connection.execute(
                    "INSERT INTO chats(asset_id, session_id, created_at, last_used_at) "
                    "VALUES(?, ?, ?, ?) ON CONFLICT(asset_id) DO UPDATE SET "
                    "session_id = excluded.session_id, last_used_at = excluded.last_used_at",
                    (asset_id, session_id, stamp, stamp),
                )
                connection.execute(
                    "INSERT INTO chat_history(asset_id, session_id, created_at, last_used_at) "
                    "VALUES(?, ?, ?, ?) ON CONFLICT(asset_id, session_id) DO UPDATE SET "
                    "last_used_at = excluded.last_used_at",
                    (asset_id, session_id, stamp, stamp),
                )
        except MetadataError:
            raise
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error
        return {"session_id": session_id, "created_at": stamp, "last_used_at": stamp}

    def chats_for_assets(self, asset_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return {}
        found: dict[str, dict[str, Any]] = {}
        with self._read() as connection:
            if connection is None:
                return found
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                for row in connection.execute(
                    f"SELECT * FROM chats WHERE asset_id IN ({placeholders})", chunk
                ):
                    found[row["asset_id"]] = dict(row)
        return found

    def chat(self, asset_id: str) -> dict[str, Any] | None:
        with self._read() as connection:
            if connection is None:
                return None
            row = connection.execute(
                "SELECT * FROM chats WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def clear_chat(self, asset_id: str) -> bool:
        try:
            with self._write() as connection:
                cursor = connection.execute("DELETE FROM chats WHERE asset_id = ?", (asset_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as error:
            raise MetadataUnavailable(str(error)) from error

    # -- edit log / stats ---------------------------------------------------

    def edits(self, asset_id: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._read() as connection:
            if connection is None:
                return []
            if asset_id is None:
                rows = connection.execute(
                    "SELECT * FROM edits ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)
                )
            else:
                rows = connection.execute(
                    "SELECT * FROM edits WHERE asset_id = ? ORDER BY id DESC LIMIT ?",
                    (asset_id, max(1, min(limit, 500))),
                )
            return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._read() as connection:
            if connection is None:
                return {"assets": 0, "tags": 0, "chats": 0, "conventions": 0, "available": False}
            assets = connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            ordered = connection.execute(
                "SELECT COUNT(*) FROM assets WHERE organized_at IS NOT NULL"
            ).fetchone()[0]
            tags = connection.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
            chats = connection.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            conventions = connection.execute("SELECT COUNT(*) FROM conventions").fetchone()[0]
            return {
                "assets": int(assets),
                "organized": int(ordered),
                "tags": int(tags),
                "chats": int(chats),
                "conventions": int(conventions),
                "schema_version": SCHEMA_VERSION,
                "available": True,
            }

    def schema_version(self) -> int | None:
        with self._read() as connection:
            if connection is None:
                return None
            row = connection.execute(
                "SELECT value FROM state WHERE key = 'schema_version'"
            ).fetchone()
            return int(row["value"]) if row is not None else None


def _tag_id(name: str) -> str:
    # 10 hex chars, stable for a name: the design's "stable 10-hex id". Two
    # concurrent creates of the same name compute the same id, which is what
    # makes the IntegrityError loser path a plain read.
    import hashlib

    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def _normalise_tag_name(name: str) -> str:
    value = (name or "").strip().lower()
    if not value:
        raise TagRefused("tag_name_required")
    if len(value) > 24 or not _TAG_GRAMMAR.match(value):
        raise TagRefused("bad_tag_name", value[:40])
    return value


_TAG_GRAMMAR: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")


__all__ = [
    "FIELD_SOURCES",
    "MAX_TAGS_PER_ASSET",
    "MetadataError",
    "MetadataStore",
    "MetadataUnavailable",
    "SCHEMA_VERSION",
    "TAG_CATEGORIES",
    "TAG_CREATORS",
    "TagRefused",
]

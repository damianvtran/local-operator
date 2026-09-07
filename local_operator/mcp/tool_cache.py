"""SQLite-backed cache of MCP ``tools/list`` results.

Lets a slow server contribute callable (deferred) tools at startup: the cache
stores the JSON of each server's last successful ``tools/list`` so a deferred
tool knows its name, description, and input schema before the connection is
up. Every operation is best-effort — a locked/corrupt/unwritable database
degrades to ``None`` reads and no-op writes, never an exception.

The file lives under :func:`config_dir`, not ``Path.home()``. Tests isolate
via ``LOCAL_OPERATOR_CONFIG_DIR`` / ``HOME`` the same way catalogue and usage
do; writing under the real home from a cell that only overrode CONFIG_DIR is
how a previous cache leaked into ``~/.local-operator/mcp_cache.db``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_tool_cache (
    server TEXT NOT NULL,
    digest TEXT NOT NULL,
    tools_json TEXT NOT NULL,
    saved_at REAL NOT NULL,
    PRIMARY KEY (server, digest)
)
"""

_CACHE_FILENAME = "mcp_cache.db"


def default_cache_path() -> Path:
    """``<config_dir>/mcp_cache.db`` — honouring CONFIG_DIR and HOME every call."""
    return config_dir() / _CACHE_FILENAME


def config_digest(config: Any) -> str:
    """Stable digest of a server's config so a rewrite cannot serve old schemas.

    Command, args, url, env, headers, and tool allow/deny lists all change
    what ``tools/list`` would return. A cache keyed on the name alone would
    advertise the previous server's tools after the user pointed the same
    name at a different command. ``model_dump`` keeps this independent of
    which transport the config is.
    """
    if hasattr(config, "model_dump"):
        payload = config.model_dump(mode="json", exclude_none=True)
    elif isinstance(config, dict):
        payload = config
    else:
        payload = {"repr": repr(config)}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


class McpToolCache:
    """Persist per-server ``tools/list`` payloads under the config dir.

    Rows are keyed by ``(server, digest)``. :meth:`get` with a digest that
    does not match drops the stale row so a rewritten mcp.json cannot keep
    advertising the previous schema. The ``path`` argument exists for tests;
    production callers rely on :func:`default_cache_path`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else default_cache_path()

    def _connect(self) -> sqlite3.Connection | None:
        """Open the database, creating the parent dir and schema as needed."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # 0600 before sqlite opens it: tool schemas are not tokens, but
            # they name the user's servers and this is the same-account
            # boundary as auth.db / usage_cache.db. connect-then-chmod leaves
            # a readable window.
            if not self._path.exists():
                fd = os.open(self._path, os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(fd)
            conn = sqlite3.connect(str(self._path), timeout=1.0)
            conn.execute("PRAGMA busy_timeout=1000")
            conn.execute(_SCHEMA)
            self._migrate_legacy(conn)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
            return conn
        except (OSError, sqlite3.Error):
            logger.debug("MCP tool cache unavailable at %s", self._path, exc_info=True)
            return None

    @staticmethod
    def _migrate_legacy(conn: sqlite3.Connection) -> None:
        """Drop a pre-digest table so a leftover name-only row cannot be served.

        The old primary key was ``server`` alone. Keeping those rows would
        make :meth:`get` with a digest look like a miss AND leave a name-only
        row a future reader without a digest (there is none) could still hit.
        Dropping is the honest migration: the next live ``tools/list`` rewrites.
        """
        columns = {row[1] for row in conn.execute("PRAGMA table_info(mcp_tool_cache)").fetchall()}
        if "digest" not in columns:
            conn.execute("DROP TABLE IF EXISTS mcp_tool_cache")
            conn.execute(_SCHEMA)
            conn.commit()

    def get(self, server: str, digest: str | None = None) -> list[dict[str, Any]] | None:
        """Return cached tool definitions for ``server`` at ``digest``, or ``None``.

        Each entry is the raw ``tools/list`` item shape: ``{"name",
        "description", "inputSchema"}`` (extra keys pass through).

        ``digest`` is required for a hit. A name-only read cannot tell a
        rewritten config from the one that wrote the row, so it is a miss
        (and any leftover rows for that name are dropped). A digest mismatch
        deletes the stale row: the next connect will ``put`` the live list.
        """
        conn = self._connect()
        if conn is None:
            return None
        try:
            if not digest:
                conn.execute("DELETE FROM mcp_tool_cache WHERE server = ?", (server,))
                conn.commit()
                return None
            row = conn.execute(
                "SELECT tools_json FROM mcp_tool_cache WHERE server = ? AND digest = ?",
                (server, digest),
            ).fetchone()
            if row is None:
                # Stale digest(s) for this name: drop them so a removed or
                # rewritten server does not accumulate.
                conn.execute("DELETE FROM mcp_tool_cache WHERE server = ?", (server,))
                conn.commit()
                return None
            parsed = json.loads(row[0])
            return parsed if isinstance(parsed, list) else None
        except (sqlite3.Error, ValueError):
            logger.debug("MCP tool cache read failed for %r", server, exc_info=True)
            return None
        finally:
            conn.close()

    def put(self, server: str, tools: list[dict[str, Any]], digest: str | None = None) -> None:
        """Store ``tools`` for ``server`` at ``digest``; failures are swallowed.

        Without a digest the write is skipped: a name-only row is exactly the
        stale-schema bug this cache exists to close.
        """
        if not digest:
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute("DELETE FROM mcp_tool_cache WHERE server = ?", (server,))
            conn.execute(
                "INSERT OR REPLACE INTO mcp_tool_cache "
                "(server, digest, tools_json, saved_at) VALUES (?, ?, ?, ?)",
                (server, digest, json.dumps(tools), time.time()),
            )
            conn.commit()
        except (sqlite3.Error, ValueError, TypeError):
            logger.debug("MCP tool cache write failed for %r", server, exc_info=True)
        finally:
            conn.close()

    def delete(self, server: str) -> None:
        """Drop every cache entry for ``server`` (e.g. after a config removal)."""
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute("DELETE FROM mcp_tool_cache WHERE server = ?", (server,))
            conn.commit()
        except sqlite3.Error:
            logger.debug("MCP tool cache delete failed for %r", server, exc_info=True)
        finally:
            conn.close()

"""SQLite-backed cache of MCP ``tools/list`` results.

Lets a slow server contribute callable (deferred) tools at startup: the cache
stores the JSON of each server's last successful ``tools/list`` so a deferred
tool knows its name, description, and input schema before the connection is
up. Every operation is best-effort — a locked/corrupt/unwritable database
degrades to ``None`` reads and no-op writes, never an exception.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_tool_cache (
    server TEXT PRIMARY KEY,
    tools_json TEXT NOT NULL,
    saved_at REAL NOT NULL
)
"""


class McpToolCache:
    """Persist per-server ``tools/list`` payloads at ``~/.local-operator/mcp_cache.db``.

    The ``path`` argument exists for tests; production callers rely on the
    default location.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = (
            Path(path) if path is not None else Path.home() / ".local-operator" / "mcp_cache.db"
        )

    def _connect(self) -> sqlite3.Connection | None:
        """Open the database, creating the parent dir and schema as needed."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path), timeout=1.0)
            conn.execute(_SCHEMA)
            return conn
        except (OSError, sqlite3.Error):
            logger.debug("MCP tool cache unavailable at %s", self._path, exc_info=True)
            return None

    def get(self, server: str) -> list[dict[str, Any]] | None:
        """Return the cached tool definitions for ``server``, or ``None``.

        Each entry is the raw ``tools/list`` item shape: ``{"name",
        "description", "inputSchema"}`` (extra keys pass through).
        """
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT tools_json FROM mcp_tool_cache WHERE server = ?", (server,)
            ).fetchone()
            if row is None:
                return None
            parsed = json.loads(row[0])
            return parsed if isinstance(parsed, list) else None
        except (sqlite3.Error, ValueError):
            logger.debug("MCP tool cache read failed for %r", server, exc_info=True)
            return None
        finally:
            conn.close()

    def put(self, server: str, tools: list[dict[str, Any]]) -> None:
        """Store ``tools`` for ``server``; failures are swallowed."""
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT OR REPLACE INTO mcp_tool_cache (server, tools_json, saved_at) VALUES (?, ?, ?)",
                (server, json.dumps(tools), time.time()),
            )
            conn.commit()
        except (sqlite3.Error, ValueError, TypeError):
            logger.debug("MCP tool cache write failed for %r", server, exc_info=True)
        finally:
            conn.close()

    def delete(self, server: str) -> None:
        """Drop the cache entry for ``server`` (e.g. after a config removal)."""
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

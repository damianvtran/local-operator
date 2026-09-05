"""Durable at-most-once receipts for desktop control requests.

The owner already reserves natural prompt IDs across process replacement. Slash
controls do not have that property, so an HTTP retry must not re-run a side
effect merely because its response was lost. A pending receipt after a crash is
explicitly indeterminate; only replay-safe owner admissions may resume it. No
secret input or raw request body is journalled here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from pathlib import Path
from typing import Any


class ReceiptConflict(ValueError):
    pass


class DesktopReceipts:
    def __init__(self, root: Path) -> None:
        self.path = root / "desktop-receipts.db"
        self.locks: dict[str, tuple[asyncio.Lock, int]] = {}

    def _db(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        self.path.chmod(0o600)
        db.execute(
            "CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, fingerprint "
            "TEXT NOT NULL, result TEXT)"
        )
        return db

    def _claim(self, key: str, fingerprint: str, retry_safe: bool) -> dict[str, Any] | None:
        with closing(self._db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT fingerprint, result FROM receipts WHERE id = ?", (key,)
            ).fetchone()
            if row is not None:
                if row[0] != fingerprint:
                    raise ReceiptConflict("Request ID was already used with different input")
                if row[1] is not None:
                    return json.loads(row[1])
                if not retry_safe:
                    raise ReceiptConflict(
                        "Request outcome is indeterminate. Reconcile session state "
                        "before issuing a new request"
                    )
            else:
                db.execute("INSERT INTO receipts VALUES (?, ?, NULL)", (key, fingerprint))
        return None

    def _finish(self, key: str, result: dict[str, Any]) -> None:
        with closing(self._db()) as db, db:
            db.execute("UPDATE receipts SET result = ? WHERE id = ?", (json.dumps(result), key))

    async def run(
        self,
        key: str,
        body: dict[str, Any],
        operation: Callable[[], Awaitable[dict[str, Any]]],
        *,
        retry_safe: bool = False,
    ) -> dict[str, Any]:
        fingerprint = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        # Coalesce retries of THIS request, not unrelated sessions: an owner
        # compaction can wait on a provider, and must not block another session's
        # prompt admission. Cross-process races still use the SQLite transaction.
        lock, users = self.locks.get(key, (asyncio.Lock(), 0))
        self.locks[key] = (lock, users + 1)
        try:
            async with lock:
                cached = await asyncio.to_thread(self._claim, key, fingerprint, retry_safe)
                if cached is not None:
                    return {**cached, "replayed": True}
                result = await operation()
                await asyncio.to_thread(self._finish, key, result)
                return result
        finally:
            # Include waiters in the count so removing an entry can never admit
            # a second lock for a request whose previous operation is still live.
            remaining = self.locks[key][1] - 1
            if remaining:
                self.locks[key] = (lock, remaining)
            else:
                self.locks.pop(key)

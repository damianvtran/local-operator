"""Durable at-most-once receipts for desktop control requests.

The runtime already reserves natural prompt IDs across process replacement. Slash
controls do not have that property, so an HTTP retry must not re-run a side
effect merely because its response was lost. A pending receipt after a crash is
explicitly indeterminate; only replay-safe runtime admissions may resume it. No
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


class Unclaimed(Exception):
    """An operation's outcome that must NOT be recorded against its key.

    THE ONE WAY TO SAY "this refusal is not this request's outcome". ``run``
    records whatever the operation returns, and a recorded refusal is REPLAYED —
    the right answer for a request that may already have left something behind
    (a created conversation, a snapshot written before the base moved), and the
    wrong one for a refusal raised before any write at all, whose own sentence
    tells the caller to retry: replaying that would refuse a retry that could
    have succeeded, forever, for that id (review round 2, Q4).

    Raised from inside the operation rather than acted on afterwards, so the
    withdrawal happens INSIDE this journal's critical section: the row never
    exists in its recorded state, and a same-id request waiting on the lock sees
    a free key rather than a refusal it has to distinguish from an outcome.
    """

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("unrecorded outcome")
        #: What the operation wants to return; the caller still renders it.
        self.result = result


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

    def recorded(self, key: str) -> bool:
        """Has ``key`` already been claimed, WITHOUT claiming it or writing.

        WHY THE PROBE EXISTS AT ALL, and why it must stay a read rather than
        becoming a pre-emptive claim: the two answers it has to satisfy cannot be
        served by one write. A RECORDED key has to be REPLAYED, not re-executed —
        the create operation behind it migrates, leases and starts things, so
        running it a second time is precisely the duplicated side effect the
        at-most-once contract exists to prevent. A FRESH key has to be claimed by
        the request that OWNS it rather than by a mere reading — a claim is a
        durable row, and a row left behind here would make the request's own first
        attempt an "indeterminate" conflict with itself. Answering both at once is
        what makes this a read that neither claims nor executes; a caller that
        "simplifies" it into ``run`` gets one of those two wrong every time.

        WHY a separate read, concretely: the create route's pre-flight admissions
        have to be skipped for a request whose first attempt already succeeded, or
        a retry would be REFUSED (a directory that has since vanished, a model that
        has since been retired) instead of answered from its receipt — turning a
        success into a failure for the client the at-most-once contract exists for
        (review round 2, R7).

        It opens the store READ-ONLY on purpose: ``_db`` creates the file and
        chmods it, and a probe that runs before a refusal must write nothing. An
        absent store, and a store with no receipts table yet, both record nothing.

        The path reaches SQLite as a proper ``file:`` URI rather than interpolated
        raw, because a config root is arbitrary user data and two of its legal
        characters are URI delimiters. Interpolated raw, a root containing ``?`` or
        ``#`` truncates the filename there and takes ``?mode=ro`` with it, so the
        open lands on the truncated path — CREATING a 0-byte file there, on the
        very path whose docstring promises a probe writes nothing — and then
        answers ``False`` for a key the store does hold (review round 3, R11).
        ``Path.as_uri`` percent-encodes, so those characters survive as data.
        """
        if not self.path.exists():
            return False
        try:
            # The connect is INSIDE the try because a read-only open of a file
            # that vanished since the check above raises at connect, not at the
            # first statement.
            with closing(
                sqlite3.connect(self.path.absolute().as_uri() + "?mode=ro", uri=True, timeout=10)
            ) as db:
                row = db.execute("SELECT 1 FROM receipts WHERE id = ?", (key,)).fetchone()
        except sqlite3.OperationalError:
            # No table yet, or a write lock held elsewhere: "nothing recorded"
            # is the safe answer — the caller then runs the admissions and
            # ``run`` owns the claim, including its conflict semantics.
            return False
        return row is not None

    def _release(self, key: str) -> None:
        """Delete an UNFINISHED claim on ``key``, so the same id may run again.

        THE ``result IS NULL`` GUARD IS THE SAFETY, not a tidy-up: a recorded
        result — a success, or a refusal the route chose to keep — is never
        deleted, so nothing here can make a completed request repeatable. Called
        only from :meth:`run`, on :class:`Unclaimed`, which is the moment the row
        is exactly that: an unfinished claim.
        """
        with closing(self._db()) as db, db:
            db.execute("DELETE FROM receipts WHERE id = ? AND result IS NULL", (key,))

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
        # Coalesce retries of THIS request, not unrelated sessions: a runtime
        # compaction can wait on a provider, and must not block another session's
        # prompt admission. Cross-process races still use the SQLite transaction.
        lock, users = self.locks.get(key, (asyncio.Lock(), 0))
        self.locks[key] = (lock, users + 1)
        try:
            async with lock:
                cached = await asyncio.to_thread(self._claim, key, fingerprint, retry_safe)
                if cached is not None:
                    return {**cached, "replayed": True}
                try:
                    result = await operation()
                except Unclaimed as unclaimed:
                    # The operation refused this request in a way that leaves
                    # nothing behind, so its id must stay usable: withdraw the
                    # claim and hand the refusal back for rendering. See
                    # :class:`Unclaimed` for why this is not an ``_finish``.
                    await asyncio.to_thread(self._release, key)
                    return unclaimed.result
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

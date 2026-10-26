"""Durable admission receipts for keyed session creation, not a work queue.

A committed pending row is a permanent uncertainty fence: its SID is authoritative
but it does NOT prove that no child was constructed. Only an exact verified owner
can complete it after a restart. Unlike transfer receipts, these IDs never expire
or evict; forgetting one would turn a lost-response retry into another session.
SQLite supplies portable cross-process claim exclusion and durable transactions;
async tasks merely coalesce callers and shield the existing spawn from disconnects.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

RECEIPTS_NAME = "mobile-start-receipts.sqlite3"
RECEIPT_CAPACITY = 4096
CAPABILITY = {"version": 1, "durable_idempotency": True, "receipt_capacity": RECEIPT_CAPACITY}

#: Per-field bound on cwd/provider/model_id, in code points.
MAX_FIELD_CHARS = 4096
#: The start route's whole-body byte cap, enforced BEFORE any JSON parsing.
#:
#: The field bounds above limit what a receipt stores, not what the relay must
#: buffer and parse to find out that a body breaks them: an 8 MiB authenticated
#: body was fully read and parsed only to be refused afterwards. The cap is
#: derived from the schema, so it never refuses a compactly or ordinarily
#: formatted UTF-8 body within the field bounds: the worst JSON encoding of one
#: code point is an escaped surrogate pair (12 bytes), so three maximal fields
#: cost 3 x 4096 x 12 bytes; the 4 KiB slack covers the key (at most 36 escaped
#: characters), escaped field names, punctuation and ordinary whitespace. Bodies
#: padded past that slack, or sent as UTF-16/32 JSON, can exceed it; no real
#: client sends either. It applies to unkeyed callers too, because the route
#: cannot know whether a body is keyed until it has parsed it; no real legacy
#: body (a path plus two identifiers) comes near it.
START_BODY_MAX_BYTES = 3 * MAX_FIELD_CHARS * 12 + 4096


class StartRefusal(Exception):
    """Machine-readable refusal; clients own localized presentation."""

    def __init__(self, code: str, status: int = 409, session_id: str | None = None):
        super().__init__(code)
        self.code = code
        self.status = status
        self.session_id = session_id


def canonical_request(body: dict[str, Any]) -> tuple[str, str]:
    """Fingerprint intent, not today's defaults or JSON object ordering.

    Unknown fields are refused for keyed callers rather than silently letting an
    unrepresented future selection alias a previously admitted request.
    """
    key = body.get("create_request_id")
    if not isinstance(key, str) or len(key) > 36:
        raise StartRefusal("invalid_create_request_id", 400)
    try:
        key = str(uuid.UUID(key))
    except ValueError as exc:
        raise StartRefusal("invalid_create_request_id", 400) from exc
    if set(body) - {"create_request_id", "cwd", "provider", "model_id"}:
        raise StartRefusal("invalid_start_request", 400)
    intent: dict[str, str | None] = {}
    for field in ("cwd", "provider", "model_id"):
        value = body.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > MAX_FIELD_CHARS):
            raise StartRefusal("invalid_start_request", 400)
        if not isinstance(value, str):
            intent[field] = None
            continue
        intent[field] = value.strip() or None
    return key, json.dumps(intent, sort_keys=True, separators=(",", ":"))


class StartReceipts:
    def __init__(self, root: Path, capacity: int = RECEIPT_CAPACITY):
        self.path = root / RECEIPTS_NAME
        self.capacity = capacity
        self._tasks: dict[str, tuple[str, asyncio.Task[dict[str, Any]]]] = {}

    @contextmanager
    def _database(self):
        """Never repair an existing empty/malformed store into forgotten IDs."""
        connection = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
                created = True
            connection = sqlite3.connect(self.path, timeout=3)
            connection.execute("PRAGMA synchronous=EXTRA")
            connection.execute("PRAGMA fullfsync=ON")
            if created:
                connection.execute(
                    "CREATE TABLE receipts (request_id TEXT PRIMARY KEY, "
                    "fingerprint TEXT NOT NULL, request TEXT NOT NULL, "
                    "resolved TEXT NOT NULL, session_id TEXT NOT NULL UNIQUE, "
                    "result TEXT)"
                )
                connection.execute("PRAGMA user_version=1")
                connection.commit()
                # SQLite syncs the transaction; the first directory entry must
                # also survive a power loss, not just an orderly process exit.
                if os.name != "nt":
                    fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise StartRefusal("create_receipts_unavailable", 503)
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise StartRefusal("create_receipts_unavailable", 503)
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except StartRefusal:
            # A selection/directory refusal is the caller's answer, not a broken
            # store; it must keep its own status and code.
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
            raise StartRefusal("create_receipts_unavailable", 503) from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _validate(row: tuple[Any, ...]) -> None:
        key, fingerprint, request, resolved, sid, result = row
        digest = hashlib.sha256(request.encode()).hexdigest()
        if str(uuid.UUID(key)) != key or digest != fingerprint:
            raise ValueError("invalid_receipt")
        intent = json.loads(request)
        chosen = json.loads(resolved)
        if not isinstance(intent, dict) or set(intent) != {"cwd", "provider", "model_id"}:
            raise ValueError("invalid_receipt")
        if not isinstance(chosen, dict) or set(chosen) != set(intent):
            raise ValueError("invalid_receipt")
        if not all(isinstance(value, str) and value for value in chosen.values()):
            raise ValueError("invalid_receipt")
        if (
            not isinstance(sid, str)
            or len(sid) != 12
            or any(c not in "0123456789abcdef" for c in sid)
        ):
            raise ValueError("invalid_receipt")
        if result is not None:
            answer = json.loads(result)
            if (
                not isinstance(answer, dict)
                or answer.get("ok") is not True
                or answer.get("session_id") != sid
                or type(answer.get("pid")) is not int
                or answer["pid"] <= 0
            ):
                raise ValueError("invalid_receipt")

    def _claim(self, key: str, request: str, resolve: Callable[[], dict[str, str]]):
        fingerprint = hashlib.sha256(request.encode()).hexdigest()
        with self._database() as db:
            # The store is deliberately bounded; validate its whole small set so
            # corrupt unrelated receipts cannot coexist with new side effects.
            rows = db.execute(
                "SELECT request_id, fingerprint, request, resolved, session_id, result "
                "FROM receipts"
            ).fetchall()
            for row in rows:
                self._validate(row)
            existing = next((row for row in rows if row[0] == key), None)
            if existing is not None:
                if existing[1] != fingerprint or existing[2] != request:
                    raise StartRefusal("create_conflict")
                answer = json.loads(existing[5]) if existing[5] else None
                return False, json.loads(existing[3]), existing[4], answer
            if len(rows) >= self.capacity:
                raise StartRefusal("create_capacity", 503)
            # Resolve once INSIDE admission. Replays never re-read changed
            # defaults, reinterpret a relative directory, or validate a new one.
            chosen = resolve()
            sid = uuid.uuid4().hex[:12]
            resolved = json.dumps(chosen, sort_keys=True, separators=(",", ":"))
            self._validate((key, fingerprint, request, resolved, sid, None))
            db.execute(
                "INSERT INTO receipts VALUES (?, ?, ?, ?, ?, NULL)",
                (key, fingerprint, request, resolved, sid),
            )
        return True, chosen, sid, None

    def _finish(self, key: str, answer: dict[str, Any]) -> None:
        with self._database() as db:
            row = db.execute(
                "SELECT request_id, fingerprint, request, resolved, session_id, result "
                "FROM receipts WHERE request_id=?",
                (key,),
            ).fetchone()
            if row is None:
                raise StartRefusal("create_receipts_unavailable", 503)
            self._validate((*row[:5], json.dumps(answer)))
            if row[5] is not None and json.loads(row[5]) != answer:
                raise StartRefusal("create_indeterminate", session_id=row[4])
            db.execute("UPDATE receipts SET result=? WHERE request_id=?", (json.dumps(answer), key))

    async def start(
        self,
        body: dict[str, Any],
        resolve: Callable[[], dict[str, str]],
        spawn: Callable[[dict[str, str], str], Awaitable[int]],
        recover: Callable[[str], Awaitable[int | None]],
    ) -> dict[str, Any]:
        """`recover` resolves the CURRENT owner of an exact SID, or ``None``.

        It is deliberately the only pid source for replays: a pid written to a
        completed receipt is history, and the OS may have recycled that number
        to an unrelated process. Commanding it would address a different
        session, so an absent owner is reported as a typed hold carrying the
        authoritative SID, never as the stored pid.
        """
        key, request = canonical_request(body)
        current = self._tasks.get(key)
        if current is not None:
            if current[0] != request:
                raise StartRefusal("create_conflict")
            return await asyncio.shield(current[1])

        async def operation():
            fresh, chosen, sid, answer = await asyncio.to_thread(self._claim, key, request, resolve)
            if answer is not None:
                # COMPLETED REPLAY: the SID is authoritative, the live pid is
                # not. Reconcile the exact current handle instead of echoing
                # the recorded pid, which PID reuse may have reassigned.
                pid = await recover(sid)
                if pid is None:
                    raise StartRefusal("create_indeterminate", session_id=sid)
                return {"ok": True, "pid": pid, "session_id": sid}
            try:
                pid = await spawn(chosen, sid) if fresh else await recover(sid)
                if pid is None:
                    raise StartRefusal("create_indeterminate", session_id=sid)
            except StartRefusal:
                raise
            except Exception as exc:
                # Failure may follow constructor admission. Keep its permanent
                # fence; neither timeout nor disconnection proves no side effect.
                # A cancelled request is the same uncertainty, and the claim is
                # retained so its retry answers with this exact SID.
                raise StartRefusal("create_indeterminate", session_id=sid) from exc
            answer = {"ok": True, "pid": pid, "session_id": sid}
            await asyncio.to_thread(self._finish, key, answer)
            return answer

        task = asyncio.create_task(operation())
        self._tasks[key] = (request, task)

        def finished(done):
            if self._tasks.get(key, (None, None))[1] is done:
                self._tasks.pop(key, None)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        # Shield the entire receipt lifecycle, not just child construction: an
        # abandoned HTTP request must not cancel the durable completion write.
        return await asyncio.shield(task)

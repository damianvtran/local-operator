"""Durable at-most-once receipts for the relay's transfer route.

WHY THIS EXISTS. ``POST /api/sessions/{id}/transfer`` is the phone plane's one
destructive mesh verb, and a phone retrying after a lost response (a flaky
tunnel, a killed tab, a 5xx whose body never arrived) must not run a SECOND
move for one user intent. The ``request_id`` in the body is that intent's
identity: the first request claims it, its settled outcome is recorded against
it, and every later request with the same id replays the recorded outcome —
``replayed: true`` on a success — instead of dialling the mesh again.

THE SEMANTICS ARE ``server/utils/desktop_receipts.py``'S, deliberately: claim
before the operation runs, fingerprint the body so one id can never be reused
for different input, REPLAY a recorded outcome with ``replayed: true``, REFUSE
an id whose claim never finished ("indeterminate — reconcile before retrying"),
and release the claim on ``Unclaimed`` so a refusal that left nothing behind
does not answer from itself forever. What differs is the store, and only the
store: this plane keeps one small JSON file under the config root, written
staged-and-replaced like ``mobile/seen.py``, because the daemon is its ONLY
writer — a second daemon fails to bind and exits loudly (``docs/mobile.md``),
so the cross-process BEGIN IMMEDIATE the desktop store needs has no analogue
here. A change to those semantics belongs in BOTH stores or in neither.

WHAT IS DELIBERATELY DIFFERENT FROM THE DESKTOP STORE, stated so a reader does
not have to diff them: this file is BOUNDED (oldest entries evicted). The
desktop's sqlite journal grows without bound on a process that is long-lived;
a phone that loops with fresh ids must not grow this file without limit, so
entries past ``MAX_ENTRIES`` are dropped oldest-first. The cost is explicit: an
evicted id whose move already settled replays as a fresh request, and the
move's own guards decide what that second attempt does — the same position a
request minted before the daemon ever started is in. Failed reads do NOT
degrade to empty (that would silently re-open every recorded id): a store this
process cannot parse or read refuses, and the route renders it as a named 503.

THREADING. Claim/release/finish are blocking file work and run on worker
threads (``asyncio.to_thread`` at the call sites); a ``threading.Lock``
serialises the read-modify-write against OTHER keys' concurrent workers, and
the per-key ``asyncio.Lock`` coalesces retries of one id so the second waiter
replays the moment the first finishes. Both layers are load-bearing: the file
lock is what keeps two different transfers' claims from tearing each other's
JSON, and the key lock is what lets a retry arrive mid-move and still get the
recorded answer rather than an "indeterminate" both of them would have to
un-learn.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

#: The store file's name, directly under ``config_dir()`` beside the other
#: owner-private state (``mobile-seen.json`` sets the convention).
TRANSFER_RECEIPTS_NAME = "mobile-transfer-receipts.json"

#: The bound on recorded entries, oldest evicted first. Transfers are manual
#: and rare — this is a guard against a client looping with fresh ids, not a
#: working set; 2048 ids is far beyond any retry envelope's useful lifetime.
MAX_ENTRIES = 2048


class TransferConflict(ValueError):
    """The id is taken — by different input, or by an unsettled claim."""


class ReceiptsUnreadable(RuntimeError):
    """The store exists but this process cannot read it.

    NOT an empty store, and the distinction is the whole point: an empty answer
    here would make every recorded id re-runnable, which is a second move for a
    first intent — the exact failure this file exists to prevent.
    """


class Unclaimed(Exception):
    """An outcome that must NOT be recorded against its key.

    The one way to say "this refusal left nothing behind": the claim is
    withdrawn so the same id may run again, because a user who frees the
    session up and presses again must not be answered from a refusal forever.
    Raised from inside the operation so the withdrawal happens inside this
    journal's critical section — a same-id request waiting on the per-key lock
    then sees a free key rather than a refusal it has to distinguish.
    """

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("unrecorded outcome")
        #: What the operation wants to return; the caller still renders it.
        self.result = result


class TransferReceipts:
    """Durable receipts keyed by request id, over one JSON file per config root."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / TRANSFER_RECEIPTS_NAME
        #: Serialises every read-modify-write of the file across worker
        #: threads. The asyncio key locks are NOT enough on their own: two
        #: transfers with different ids claim in parallel workers, and without
        #: this lock their load→mutate→save cycles interleave and one claim
        #: disappears.
        self._file_lock = threading.Lock()
        #: per-key ``(lock, waiter count)`` — the desktop store's shape, so a
        #: waiting retry is counted and the entry is dropped only when the
        #: last user leaves.
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    # -- the file ------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise ReceiptsUnreadable(
                f"the transfer receipt store could not be read ({exc}); refusing "
                "rather than treating recorded requests as new"
            ) from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ReceiptsUnreadable(
                "the transfer receipt store is not readable JSON; refusing rather "
                "than treating recorded requests as new"
            ) from exc
        if not isinstance(data, dict):
            raise ReceiptsUnreadable(
                "the transfer receipt store is not a JSON object; refusing rather "
                "than treating recorded requests as new"
            )
        return data

    def _write(self, data: dict[str, Any]) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=directory, prefix=f".{TRANSFER_RECEIPTS_NAME}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- the claim lifecycle (blocking; callers hand these to a thread) ------

    def _claim(self, key: str, fingerprint: str) -> dict[str, Any] | None:
        with self._file_lock:
            data = self._load()
            entry = data.get(key)
            if entry is not None:
                if str(entry.get("fingerprint") or "") != fingerprint:
                    raise TransferConflict("Request ID was already used with different input")
                result = entry.get("result")
                if result is not None:
                    return dict(result)
                raise TransferConflict(
                    "Request outcome is indeterminate. Reconcile session state "
                    "before issuing a new request"
                )
            data[key] = {"fingerprint": fingerprint, "result": None}
            while len(data) > MAX_ENTRIES:
                data.pop(next(iter(data)))
            self._write(data)
            return None

    def _release(self, key: str) -> None:
        with self._file_lock:
            data = self._load()
            entry = data.get(key)
            if entry is not None and entry.get("result") is None:
                del data[key]
                self._write(data)

    def _finish(self, key: str, fingerprint: str, result: dict[str, Any]) -> None:
        with self._file_lock:
            data = self._load()
            # Upsert rather than update-in-place: the bounded store can evict a
            # claim while its operation is still running, and the outcome the
            # operation produced is still this id's answer.
            data[key] = {"fingerprint": fingerprint, "result": result}
            while len(data) > MAX_ENTRIES:
                data.pop(next(iter(data)))
            self._write(data)

    # -- the one entry point -------------------------------------------------

    async def run(
        self,
        key: str,
        body: dict[str, Any],
        operation: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run ``operation`` once per ``key``; replay its recorded answer after.

        ``body`` is fingerprinted as sent (canonical JSON) so a replayed id
        carries the same input it was claimed with, and a different body on a
        used id is a conflict, not a replay.
        """
        fingerprint = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        lock, users = self._locks.get(key, (asyncio.Lock(), 0))
        self._locks[key] = (lock, users + 1)
        try:
            async with lock:
                cached = await asyncio.to_thread(self._claim, key, fingerprint)
                if cached is not None:
                    return {**cached, "replayed": True}
                try:
                    result = await operation()
                except Unclaimed as unclaimed:
                    # Withdraw the claim and hand the refusal back for
                    # rendering: see Unclaimed for why this is not a finish.
                    await asyncio.to_thread(self._release, key)
                    return unclaimed.result
                await asyncio.to_thread(self._finish, key, fingerprint, result)
                return result
        finally:
            # Include waiters in the count so removing the entry can never
            # admit a second lock for a request whose operation is still live.
            remaining = self._locks[key][1] - 1
            if remaining:
                self._locks[key] = (lock, remaining)
            else:
                self._locks.pop(key, None)

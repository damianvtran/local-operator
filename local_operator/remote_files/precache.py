"""Precache the media of ACTIVE remote sessions so opening one is instant.

docs/design/remote-files-parity.md §4 "Precache". Metadata decides, bytes follow:
one owner listing per session (user attachments + the files index filtered to
image/audio/video), then each readable file at most :data:`PER_FILE_MAX_BYTES` is
pulled into the bounded cache through the SAME fetch-on-miss machinery the routes
use — streamed to disk, never held in memory.

- **Active**: a catalogue row that is remote, reachable, not archived and moved
  within :data:`ACTIVE_WINDOW_S`, plus the last :data:`OPENED_LRU` remote sessions
  this daemon opened; at most :data:`MAX_SESSIONS` per pass.
- **Trigger**: the desktop catalogue poll, at most once per :data:`RESCAN_S` per
  session and only when its ``mtime`` moved; opening a remote session triggers it
  at once.
- **Priority**: one precache pull in flight per owner (a nested semaphore of 1
  inside the owner's 2), so an on-screen request always has a slot.
- **Budget**: a pass stops before a landing would push the cache past its budget,
  so it never evicts what it (or the screen) landed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from local_operator.network.files import FILE_PULL_CLIENT_TIMEOUT_S
from local_operator.remote_files import cache as remote_cache
from local_operator.remote_files import fetch as remote_fetch

logger = logging.getLogger(__name__)

#: "Active" = moved within the last 2 h: 11 of 239 sessions on the measuring host
#: (design §0), so the set stays small.
ACTIVE_WINDOW_S = 2 * 3600.0
#: Remote sessions this daemon opened, kept regardless of age (cleared on restart).
OPENED_LRU = 8
#: Sessions per pass: above the 11 observed active, bounded for a large mesh.
MAX_SESSIONS = 16
#: A session is re-listed at most this often, and only when its ``mtime`` moved.
RESCAN_S = 60.0
#: The UI's own ``MAX_DOWNLOAD_BYTES`` (desktop-media.ts): anything larger cannot be
#: previewed through the media relay, so precaching it buys nothing.
PER_FILE_MAX_BYTES = 32 * 1024 * 1024
#: The kinds worth preparing ahead of a click.
MEDIA_KINDS = ("image", "audio", "video")
#: Precache pulls in flight per owner, nested INSIDE the owner's fetch semaphore.
PRECACHE_PER_OWNER = 1


def active_set(
    rows: list[dict[str, Any]], opened: list[str], *, now: float | None = None
) -> list[dict[str, Any]]:
    """The remote rows a pass should prepare: opened first, then newest active."""
    now = time.time() if now is None else now
    usable = {
        str(row.get("id") or ""): row
        for row in rows
        if row.get("locality") == "remote"
        and row.get("reachable", True)
        and not row.get("archived")
        and row.get("owner_device")
    }
    chosen: list[dict[str, Any]] = []
    for session_id in opened:
        row = usable.pop(session_id, None)
        if row is not None:
            chosen.append(row)
    recent = sorted(
        (row for row in usable.values() if now - float(row.get("mtime") or 0.0) <= ACTIVE_WINDOW_S),
        key=lambda row: float(row.get("mtime") or 0.0),
        reverse=True,
    )
    return (chosen + recent)[:MAX_SESSIONS]


class Precacher:
    """One daemon's precache state: the opened LRU, per-session pass gates, a worker."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.opened: OrderedDict[str, None] = OrderedDict()
        #: session id -> (monotonic time of its last pass, the mtime that pass saw)
        self.passes: dict[str, tuple[float, float]] = {}
        self.queue: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.worker: asyncio.Task[None] | None = None
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        #: The digests landed so far, newest last (a test reads it).
        self.landed: list[str] = []
        self.rows: dict[str, dict[str, Any]] = {}

    # -- triggers -----------------------------------------------------------

    def observe(self, rows: list[dict[str, Any]], *, now: float | None = None) -> None:
        """The catalogue poll's remote rows: queue each active session that moved."""
        if remote_cache.configured_budget(self.root) <= 0:
            return
        clock = time.monotonic()
        for row in rows:
            if row.get("id"):
                self.rows[str(row["id"])] = row
        for row in active_set(rows, list(reversed(self.opened)), now=now):
            session_id = str(row["id"])
            last = self.passes.get(session_id)
            mtime = float(row.get("mtime") or 0.0)
            if last is not None and (clock - last[0] < RESCAN_S or last[1] == mtime):
                continue
            self._enqueue(row)

    def opened_session(self, session_id: str, row: dict[str, Any] | None = None) -> None:
        """A remote session was opened here: remember it and prepare it now."""
        self.opened[session_id] = None
        self.opened.move_to_end(session_id)
        while len(self.opened) > OPENED_LRU:
            self.opened.popitem(last=False)
        row = row or self.rows.get(session_id)
        if row is not None and remote_cache.configured_budget(self.root) > 0:
            self._enqueue(row)

    def _enqueue(self, row: dict[str, Any]) -> None:
        self.queue[str(row["id"])] = row
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self.worker is None or self.worker.done():
            self.worker = loop.create_task(self._drain())

    # -- the pass -------------------------------------------------------------

    async def _drain(self) -> None:
        while self.queue:
            _session_id, row = self.queue.popitem(last=False)
            try:
                await self.prepare(row)
            except Exception:  # noqa: BLE001 — precache is an optimisation, never a fault
                logger.debug("precache of %s failed", row.get("id"), exc_info=True)

    def _semaphore(self, owner: str) -> asyncio.Semaphore:
        semaphore = self.semaphores.get(owner)
        if semaphore is None:
            semaphore = self.semaphores[owner] = asyncio.Semaphore(PRECACHE_PER_OWNER)
        return semaphore

    async def candidates(self, row: dict[str, Any]) -> list[tuple[str, dict[str, str], str, int]]:
        """``(member_id, source, key, size)`` for each readable media file, newest first."""
        from local_operator.network.types import MeshRefusal
        from local_operator.server.utils.remote_session_files import ask

        session_id = str(row["id"])
        found: list[tuple[str, dict[str, str], str, int]] = []
        seen: set[str] = set()
        try:
            attachments, _ = await ask(
                self.root, session_id, phase="list", index="user_attachments", query={}
            )
            files, _ = await ask(
                self.root,
                session_id,
                phase="list",
                index="files",
                query={"kinds": list(MEDIA_KINDS), "sort": "recent", "limit": 200},
            )
        except (MeshRefusal, KeyError):
            return []
        for item in reversed(list(attachments.get("items") or [])):
            digest = str(item.get("attachment") or "")
            kind = str(item.get("mime_type") or "").split("/", 1)[0]
            size = int(item.get("bytes") or 0)
            if (
                digest
                and digest not in seen
                and item.get("remote_read")
                and kind in MEDIA_KINDS
                and size <= PER_FILE_MAX_BYTES
            ):
                seen.add(digest)
                found.append((session_id, {"attachment": digest}, digest, size))
        for file_row in files.get("rows") or []:
            read = file_row.get("remote_read") or {}
            size = int(file_row.get("size_bytes") or 0)
            if file_row.get("kind") not in MEDIA_KINDS or not read or size > PER_FILE_MAX_BYTES:
                continue
            ref = file_row.get("ref") or {}
            member = str((file_row.get("session") or {}).get("id") or session_id)
            if read.get("kind") == "attachment" and ref.get("digest"):
                digest = str(ref["digest"])
                if digest in seen:
                    continue
                seen.add(digest)
                found.append((member, {"attachment": digest}, digest, size))
            elif read.get("kind") == "path":
                # The listing names the pad file in display form; ``row`` gives the
                # pull source, so the pass asks for it per file (owner-side cheap).
                found.append((member, {"file_id": str(file_row["id"])}, "", size))
        return found

    async def prepare(self, row: dict[str, Any]) -> int:
        """One pass over one session. Returns the bytes it landed."""
        from local_operator.network.types import MeshRefusal
        from local_operator.server.utils.remote_session_files import ask

        session_id = str(row["id"])
        owner = str(row.get("owner_device") or "")
        self.passes[session_id] = (time.monotonic(), float(row.get("mtime") or 0.0))
        budget = remote_cache.configured_budget(self.root)
        landed = 0
        for member, source, key, size in await self.candidates(row):
            if "file_id" in source:
                try:
                    detail, _ = await ask(
                        self.root, session_id, phase="row", file_id=source["file_id"]
                    )
                except (MeshRefusal, KeyError):
                    continue
                pull = detail.get("source")
                if not isinstance(pull, dict) or not pull.get("path"):
                    continue
                member = str(pull.get("session_id") or member)
                source = {"path": str(pull["path"])}
                key = f"path:{pull['path']}:{size}"
            elif await asyncio.to_thread(remote_cache.lookup, self.root, (member, session_id), key):
                continue
            total = await asyncio.to_thread(remote_cache.total_bytes, self.root)
            if total + size > budget:
                break
            try:
                async with self._semaphore(owner):
                    blob = await remote_fetch.fetch(
                        self.root,
                        owner=owner,
                        member_id=member,
                        source=source,
                        key=key,
                        session_ids=(member, session_id),
                        wait_s=FILE_PULL_CLIENT_TIMEOUT_S,
                    )
            except remote_fetch.RemoteFetchError as error:
                if error.kind in ("unreachable", "outdated"):
                    break
                continue
            self.landed.append(Path(blob.path).stem)
            landed += size
        return landed


_PRECACHERS: dict[str, Precacher] = {}


def for_root(root: Path) -> Precacher:
    key = str(Path(root))
    precacher = _PRECACHERS.get(key)
    if precacher is None:
        precacher = _PRECACHERS[key] = Precacher(Path(root))
    return precacher

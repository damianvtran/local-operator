"""Fetch-on-miss: pull one file of a peer's session into the viewer's cache.

docs/design/remote-files-parity.md §1. The desktop routes that serve a peer
session's bytes look locally first (``cache.lookup``) and, on a miss, wait here
for ONE pull from the owner through this device's relay (``session_file_pull``
with ``land="cache"``). The owner applies S1's confinement on every phase, so
nothing reaches the cache that a view member could not already pull.

- **Single-flight** in-process: one task per ``(owner, member, key)``; across
  processes the relay's ``_part_lock`` already serialises a digest.
- **Bounded wait**: a route waits :data:`REMOTE_FETCH_WAIT_S` under ``shield``;
  when that expires the pull KEEPS RUNNING (bounded by the relay's own
  ``FILE_PULL_CLIENT_TIMEOUT_S``) and the route answers "pending", so a retry
  joins the same transfer instead of starting another.
- **Concurrency**: at most :data:`OWNER_CONCURRENCY` pulls per owner device,
  S1's "<=2 inbound transfers per link".
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.remote_files import cache

logger = logging.getLogger(__name__)

#: How long a route waits for a pull before answering "pending". Basis: the loopback
#: max for a 32 MiB pull (6.0 s, n=3, design §0) x 3, and under the desktop media
#: relay's 120 s timeout so the renderer always gets an answer it can act on.
REMOTE_FETCH_WAIT_S = 20.0

#: Concurrent pulls per owner device: S1's "<=2 inbound transfers per link".
OWNER_CONCURRENCY = 2

#: Relay refusal codes that mean "try again later", not "never".
_RETRYABLE = frozenset({"peer_unreachable", "relay_unavailable"})
#: The owner predates mesh file transfer: a named update, never a fallback.
_OUTDATED = frozenset({"predates_file_transfer"})


class RemoteFetchError(Exception):
    """A pull that did not produce bytes. ``kind`` decides the HTTP answer."""

    def __init__(self, kind: str, code: str, message: str) -> None:
        super().__init__(message)
        #: ``unreachable`` | ``pending`` | ``outdated`` | ``refused``
        self.kind = kind
        #: The relay's own refusal code (``peer_unreachable``, ``not_referenced``…).
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FetchKey:
    owner: str
    member_id: str
    key: str


_TASKS: dict[tuple[int, FetchKey], asyncio.Task[cache.CachedBlob]] = {}
#: Digests landed by a pull whose waiters have not looked them up yet. Every
#: eviction pass keeps them, so one landing cannot evict another between its
#: pull and its ``lookup`` (#3872 R1-3); past the lookup, the caller opens an fd
#: and an unlink can no longer take the bytes.
_LANDED: dict[int, str] = {}
_SEMAPHORES: dict[tuple[int, str], asyncio.Semaphore] = {}
#: Counters a test reads to prove single-flight (one pull per key, however many waiters).
STATS: dict[str, int] = {"pulls": 0}


def owner_semaphore(owner: str) -> asyncio.Semaphore:
    """The per-owner semaphore, per event loop (a semaphore binds to its loop)."""
    key = (id(asyncio.get_running_loop()), owner)
    semaphore = _SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = _SEMAPHORES[key] = asyncio.Semaphore(OWNER_CONCURRENCY)
    return semaphore


def _classify(code: str, message: str) -> RemoteFetchError:
    if code in _OUTDATED:
        return RemoteFetchError("outdated", code, message)
    if code in _RETRYABLE:
        return RemoteFetchError("unreachable", code, message)
    return RemoteFetchError("refused", code, message)


def _pull(root: Path, owner: str, member_id: str, source: dict[str, str]) -> dict[str, Any]:
    from local_operator.network.files import FILE_PULL_CLIENT_TIMEOUT_S
    from local_operator.network.types import MeshRefusal
    from local_operator.server.utils import desktop_mesh

    STATS["pulls"] += 1
    try:
        return desktop_mesh._call(  # noqa: SLF001 — the desktop's one relay seam
            root,
            "session_file_pull",
            timeout=FILE_PULL_CLIENT_TIMEOUT_S,
            peer=owner,
            session_id=member_id,
            land="cache",
            **source,
        )
    except MeshRefusal as refusal:
        raise _classify(refusal.code, str(refusal)) from None


async def _run(
    root: Path, fetch: FetchKey, source: dict[str, str], session_ids: tuple[str, ...]
) -> cache.CachedBlob:
    async with owner_semaphore(fetch.owner):
        receipt = await asyncio.to_thread(_pull, root, fetch.owner, fetch.member_id, source)
    digest = str(receipt.get("attachment") or "")
    # The digest just landed is kept, and so is every other landing still on its
    # way to its lookup: I4-1 bounds the cache at budget + the files in flight, and
    # a budget of 0 then evicts them on a LATER pass, after they were served.
    marker = id(asyncio.current_task())
    _LANDED[marker] = digest
    try:
        keep = frozenset(_LANDED.values()) | {
            task_key.key for (_loop, task_key) in _TASKS if cache.is_digest(task_key.key)
        }
        await asyncio.to_thread(cache.evict, root, keep=keep)
        blob = await asyncio.to_thread(cache.lookup, root, session_ids, digest)
    finally:
        _LANDED.pop(marker, None)
    if blob is None:
        # Landed and verified, then gone before this stat: another process's
        # eviction pass with a smaller budget. Retryable — the next ask re-pulls.
        raise RemoteFetchError("unreachable", "evicted", "the pulled file was evicted")
    return blob


async def fetch(
    root: Path,
    *,
    owner: str,
    member_id: str,
    source: dict[str, str],
    key: str,
    session_ids: tuple[str, ...] = (),
    wait_s: float | None = None,
) -> cache.CachedBlob:
    """Pull ``source`` from ``owner``'s session ``member_id`` into the cache.

    ``source`` is ``{"attachment": <digest>}`` or ``{"path": "scratchpad://…"}``;
    ``key`` names the content (the digest, or the owner's stat sha256) for
    single-flight. Raises :class:`RemoteFetchError`; ``kind == "pending"`` when the
    wait expired while the pull continues.
    """
    loop_id = id(asyncio.get_running_loop())
    fetch_key = FetchKey(owner, member_id, key)
    slot = (loop_id, fetch_key)
    task = _TASKS.get(slot)
    if task is None:
        task = asyncio.ensure_future(
            _run(Path(root), fetch_key, source, session_ids or (member_id,))
        )
        _TASKS[slot] = task

        def _forget(done: asyncio.Task[cache.CachedBlob], slot: Any = slot) -> None:
            if _TASKS.get(slot) is done:
                del _TASKS[slot]
            if not done.cancelled():
                done.exception()  # retrieved: a pending waiter may have left

        task.add_done_callback(_forget)
    bound = REMOTE_FETCH_WAIT_S if wait_s is None else wait_s
    try:
        return await asyncio.wait_for(asyncio.shield(task), bound)
    except TimeoutError:
        raise RemoteFetchError(
            "pending", "fetch_pending", "the file is still on its way from the owner"
        ) from None


def pending_count() -> int:
    return sum(1 for task in _TASKS.values() if not task.done())

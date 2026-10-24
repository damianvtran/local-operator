"""The resumable, signature-keyed cache under the per-session file scan.

``scan.py`` is pure; this module owns the journal I/O and the question "is what I scanned
still a prefix of what is on disk?". It copies ``code_requests/ledger.py``'s ladder, which in
turn copies ``session/transcript_index.py``'s, and for the same measured reasons:

1. an EXACT ``(size, mtime, inode)`` match serves the cached scan - one ``stat``, no read;
2. a GROWN file on the SAME inode whose recorded tail still verifies is scanned from the saved
   byte offset only (O(appended bytes));
3. anything else - ``compact_file`` REPLACES the journal (new inode), a truncation, an in-place
   rewrite that kept the size - rescans from byte 0. Compaction can move rows, so only a whole
   pass sees the new arrangement.

DIFFERENCES FROM THE CODE-REQUEST LEDGER, each on purpose:

* **Resumable by construction.** The design's backfill rule is "lazy and resumable": the first
  GET for a 300 MB journal must answer inside its latency budget and keep going afterwards. So
  :func:`refresh` takes a ``budget_s`` and, when it runs out, PUBLISHES the partial scan with
  its byte offset and ``complete: false``. The next call is simply rung 2 above. There is no
  separate "partial" code path to get wrong - an interrupted scan and an appended journal are
  the same thing.
* **The tail check is a content hash, not an id probe.** ``code_requests`` compares the last
  row's ``id``; a compaction that rewrites a row IN PLACE (same id, new content) at an equal
  size would pass that. Hashing the last complete line fails it. Rows are bounded well under
  :data:`_MAX_TAIL_LINE` (the heaviest observed is ~900 KB); a longer line simply fails
  verification and costs a rescan.
* **One cache file, no separate "index".** ``code_requests`` keeps an index for cold readers
  that never want a scan cache; nothing here reads a machine-wide view, so the cache IS the
  answer.

The cache lives under ``<config>/cache/session_files/<session_id>.json`` - OUTSIDE the session
directory on purpose: writing a sidecar into it moves the directory's mtime, which would
reorder the recency-ranked session list (the reason ``transcript_index`` gives too).

A cache is DERIVED, never authoritative: a missing, torn, wrong-schema or wrong-cwd file just
costs a rescan. Writes are atomic (pid-suffixed temp + ``os.replace``; two processes building
one session wrote one fixed temp path and tore documents - ``transcript_index``'s measured
remedy). Reads never raise into a request.

Blocking throughout: call it from a worker thread. Process-wide residency (:data:`_RESIDENT`)
means a polling client costs one ``stat`` per member session rather than a JSON parse of a
multi-megabyte cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final

from local_operator.session_files.scan import SCAN_SCHEMA, FileScan

logger = logging.getLogger(__name__)

#: Bumped on an incompatible change to the cache DOCUMENT (the scan's own shape is versioned
#: by :data:`~local_operator.session_files.scan.SCAN_SCHEMA`). A reader that does not
#: understand a schema discards the file; the next scan rewrites it.
CACHE_SCHEMA: Final = 1
CACHE_DIRNAME: Final = "session_files"
TRANSCRIPT_FILENAME: Final = "transcript.jsonl"

#: Longest journal line the tail verification will read back. Rows reach ~900 KB in real
#: stores; a line past this is not worth a verification and falls through to a rescan.
_MAX_TAIL_LINE: Final = 4 << 20

#: Lines between deadline checks. A clock read per line would be the dominant cost on a journal
#: of tiny rows; 64 keeps overshoot to a few ms even when a line is a megabyte.
_DEADLINE_STRIDE: Final = 64

#: Scans kept resident per process. A scan is a few MB at the record cap; the desktop looks at
#: one session at a time, so a small LRU covers a session and its subagents with room to spare.
_RESIDENT_CAP: Final = 24

#: File stems are session ids (12 hex) or a fork's same-shaped id; refuse anything that could
#: contribute a separator or ``..`` to a path (``code_requests.ledger._derived_stem``'s guard,
#: which ``tests/unit/session/test_no_session_deletion.py`` relies on to argue its allow-list).
_STEM: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")


@dataclass(frozen=True)
class Snapshot:
    """One published scan plus the facts that say how far it got. Treat as immutable."""

    scan: FileScan
    #: ``{size, mtime, inode, offset, tail}`` of the journal when this was published.
    sig: dict[str, Any]
    #: ``True`` when the scan reached the end of the journal it saw. ``False`` is a partial
    #: scan cut by a deadline: its records are real but the list is not finished.
    complete: bool
    updated_at: float
    cwd: str
    home: str
    #: Bytes of journal covered / present, for the ``building`` progress the route reports.
    covered_bytes: int = 0
    total_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def revision(self) -> int:
        """Monotone per session: milliseconds of the publish time (``_revision``'s rule)."""
        return int(self.updated_at * 1000)


def transcript_path(session_dir: Path) -> Path:
    """The journal this scanner reads, materialised when it is compressed.

    THE one answer for this module and its callers (the desktop route's
    "missing/ready" probe, the library index's signature). A compressed session
    resolves to its decompressed cache copy — byte-identical to the journal — so
    the scanner's offsets and signatures keep their meaning. When the cache copy
    is re-made its inode changes, which costs one full rescan and never a wrong
    answer. ``None`` from the resolver means no journal at all, so the plain
    path is returned and the caller's stat raises the same ENOENT it always did.
    """
    from local_operator.session import compression

    return compression.transcript_for_read(Path(session_dir)) or (
        Path(session_dir) / TRANSCRIPT_FILENAME
    )


def cache_dir(config_dir: str | Path) -> Path:
    return Path(config_dir) / "cache" / CACHE_DIRNAME


def cache_path(config_dir: str | Path, session_id: str) -> Path:
    if not _STEM.fullmatch(session_id or ""):
        raise ValueError(f"not a session id: {session_id!r}")
    return cache_dir(config_dir) / f"{session_id}.json"


# ---------------------------------------------------------------------------
# Residency and locks
# ---------------------------------------------------------------------------

_RESIDENT: "OrderedDict[tuple[str, str], Snapshot]" = OrderedDict()
_RESIDENT_GUARD = threading.Lock()
_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def _key(config_dir: str | Path, session_id: str) -> tuple[str, str]:
    return (str(config_dir), session_id)


def _lock_for(key: tuple[str, str]) -> threading.Lock:
    with _RESIDENT_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def _remember(key: tuple[str, str], snapshot: Snapshot) -> None:
    with _RESIDENT_GUARD:
        _RESIDENT[key] = snapshot
        _RESIDENT.move_to_end(key)
        # The per-key LOCK is deliberately kept on eviction: dropping it while another thread
        # holds it would let a third caller mint a second lock for the same key and run two
        # scans of one session at once. A lock is a few dozen bytes per session ever viewed.
        while len(_RESIDENT) > _RESIDENT_CAP:
            _RESIDENT.popitem(last=False)


def _resident(key: tuple[str, str]) -> Snapshot | None:
    with _RESIDENT_GUARD:
        held = _RESIDENT.get(key)
        if held is not None:
            _RESIDENT.move_to_end(key)
        return held


def forget_resident() -> None:
    """Drop every resident scan. Tests only: a process restart is the production analogue."""
    with _RESIDENT_GUARD:
        _RESIDENT.clear()
        _LOCKS.clear()


# ---------------------------------------------------------------------------
# Cache file
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        logger.warning("session-files cache: could not write %s", path, exc_info=True)
        return False


def _load(
    config_dir: str | Path,
    session_id: str,
    *,
    cwd: str,
    home: str,
    deny: Callable[[str], bool] | None,
) -> Snapshot | None:
    """The cached snapshot for a session, or ``None`` for anything that is not trustworthy."""
    try:
        path = cache_path(config_dir, session_id)
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("session-files cache: unreadable cache for %s; rescanning", session_id)
        return None
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != CACHE_SCHEMA
        or doc.get("scan_schema") != SCAN_SCHEMA
        or not isinstance(doc.get("sig"), dict)
    ):
        return None
    try:
        scan = FileScan.from_payload(doc["scan"], cwd=cwd, home=home, deny=deny)
    except ValueError:
        logger.warning("session-files cache: malformed scan for %s; rescanning", session_id)
        return None
    sig = doc["sig"]
    return Snapshot(
        scan=scan,
        sig=sig,
        complete=bool(doc.get("complete")),
        updated_at=float(doc.get("updated_at") or 0.0),
        cwd=str(doc.get("cwd") or ""),
        home=str(doc.get("home") or ""),
        covered_bytes=int(sig.get("offset") or 0),
        total_bytes=int(sig.get("size") or 0),
    )


def _store(config_dir: str | Path, session_id: str, snapshot: Snapshot) -> None:
    try:
        path = cache_path(config_dir, session_id)
    except ValueError:
        return
    _write_json(
        path,
        {
            "schema": CACHE_SCHEMA,
            "scan_schema": SCAN_SCHEMA,
            "session_id": session_id,
            "updated_at": round(snapshot.updated_at, 3),
            "cwd": snapshot.cwd,
            "home": snapshot.home,
            "complete": snapshot.complete,
            "sig": snapshot.sig,
            "scan": snapshot.scan.to_payload(),
        },
    )


# ---------------------------------------------------------------------------
# The signature ladder
# ---------------------------------------------------------------------------


def _exact(sig: dict[str, Any], stat: os.stat_result) -> bool:
    return (
        sig.get("size") == stat.st_size
        and sig.get("mtime") == stat.st_mtime
        and sig.get("inode") == stat.st_ino
    )


def _line_digest(line: bytes) -> list[Any]:
    return [len(line), hashlib.sha1(line, usedforsecurity=False).hexdigest()]


def _verify_tail(path: Path, end: int, tail: Any) -> bool:
    """Whether the line ending at byte ``end`` is still the one the cache recorded.

    Reads back from ``end - 1`` to the previous newline (bounded by :data:`_MAX_TAIL_LINE`) and
    compares the line's length and SHA-1. A compaction that kept the size but moved any byte
    of that line fails here, which is the case an id-only probe cannot see.
    """
    if end <= 0 or not isinstance(tail, list) or len(tail) != 2:
        return False
    length = tail[0]
    if not isinstance(length, int) or length <= 0 or length > _MAX_TAIL_LINE or length > end:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(end - length)
            raw = handle.read(length)
    except OSError:
        return False
    if len(raw) != length or not raw.endswith(b"\n"):
        return False
    return _line_digest(raw) == tail


def _read_into(
    scan: FileScan, path: Path, offset: int, deadline: float | None
) -> tuple[int, list[Any] | None, bool]:
    """Feed journal lines from ``offset`` into ``scan``.

    Returns ``(new_offset, tail, complete)``. ``tail`` is the digest of the last complete line
    read (``None`` when no line was read). A final fragment without ``\\n`` is a row being
    written: it is left for the next pass, so the saved offset always ends on a row boundary.
    """
    tail: list[Any] | None = None
    complete = True
    with path.open("rb") as handle:
        handle.seek(offset)
        count = 0
        for line in iter(handle.readline, b""):
            if not line.endswith(b"\n"):
                break
            scan.feed(scan.rows, line)
            offset += len(line)
            tail = _line_digest(line) if len(line) <= _MAX_TAIL_LINE else [0, ""]
            count += 1
            if (
                deadline is not None
                and count % _DEADLINE_STRIDE == 0
                and time.monotonic() > deadline
            ):
                complete = False
                break
    return offset, tail, complete


def refresh(
    config_dir: str | Path,
    session_id: str,
    session_dir: Path,
    *,
    cwd: str,
    home: str,
    deny: Callable[[str], bool] | None = None,
    budget_s: float | None = None,
    force: bool = False,
) -> Snapshot | None:
    """Bring one session's scan up to date, or advance it by ``budget_s`` seconds.

    ``None`` when the session has no journal (a draft, or a fresh directory). Otherwise the
    returned :class:`Snapshot` is ``complete`` unless the budget ran out first; calling again
    resumes from the saved offset. ``cwd`` is part of the cache identity: relative ``bash``
    outputs resolve against it, so a session that changed directory is rescanned rather than
    served rows resolved against the old one.
    """
    path = transcript_path(session_dir)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = _key(config_dir, session_id)
    # A cold burst must not queue worker threads behind another reader's entire budget.
    # Published scans are immutable: a loser can safely serve the last partial snapshot.
    lock = _lock_for(key)
    if not lock.acquire(blocking=budget_s is None):
        cached = _resident(key)
        if cached is not None and cached.cwd == cwd and cached.home == home:
            return replace(cached, complete=False)
        return Snapshot(
            FileScan(cwd=cwd, home=home, deny=deny),
            {},
            False,
            0.0,
            cwd,
            home,
            total_bytes=stat.st_size,
        )
    try:
        cached = (
            None
            if force
            else _resident(key) or _load(config_dir, session_id, cwd=cwd, home=home, deny=deny)
        )
        if cached is not None and (cached.cwd != cwd or cached.home != home):
            cached = None
        if cached is not None and cached.complete and _exact(cached.sig, stat):
            _remember(key, cached)
            return cached

        base: FileScan | None = None
        offset = 0
        if cached is not None:
            sig = cached.sig
            saved = int(sig.get("offset") or 0)
            resumable = (
                sig.get("inode") == stat.st_ino
                and 0 < saved <= stat.st_size
                # A tail match proves append compatibility only under STRICT growth;
                # an mtime-changed same-size prefix rewrite must start from byte zero.
                and (
                    _exact(sig, stat)
                    or (
                        stat.st_size > int(sig.get("size") or 0)
                        and _verify_tail(path, saved, sig.get("tail"))
                    )
                )
            )
            if resumable:
                base = cached.scan.clone()
                offset = saved
        scan = base if base is not None else FileScan(cwd=cwd, home=home, deny=deny)
        deadline = None if budget_s is None else time.monotonic() + budget_s
        try:
            end, tail, complete = _read_into(scan, path, offset, deadline)
        except OSError:
            logger.warning("session-files: could not read %s", path, exc_info=True)
            return cached
        if tail is None and cached is not None and base is not None:
            tail = cached.sig.get("tail")
        # The stat is taken AFTER the read: the signature must describe a file at least as
        # new as the bytes consumed, or an append between the two would be skipped forever
        # (the next call would see an exact match with unread bytes behind it). A later
        # append just makes the next call take rung 2.
        try:
            final = path.stat()
        except OSError:
            return cached
        # Never publish an unread append as exact-ready. Nor may bytes read from the old
        # inode be certified against an atomic replacement's new inode/signature.
        if final.st_ino != stat.st_ino or final.st_size < end:
            return Snapshot(
                FileScan(cwd=cwd, home=home, deny=deny),
                {},
                False,
                0.0,
                cwd,
                home,
                total_bytes=final.st_size,
            )
        complete = complete and end == final.st_size
        previous = cached.updated_at if cached is not None else 0.0
        snapshot = Snapshot(
            scan=scan,
            sig={
                "size": final.st_size,
                "mtime": final.st_mtime,
                "inode": final.st_ino,
                "offset": end,
                "tail": tail,
            },
            complete=complete,
            updated_at=max(time.time(), previous + 0.001),
            cwd=cwd,
            home=home,
            covered_bytes=end,
            total_bytes=final.st_size,
        )
        _store(config_dir, session_id, snapshot)
        _remember(key, snapshot)
        return snapshot
    finally:
        lock.release()


def peek(config_dir: str | Path, session_id: str) -> Snapshot | None:
    """The resident snapshot, no I/O. What a ``building`` answer serves while a scan runs."""
    return _resident(_key(config_dir, session_id))


__all__ = [
    "CACHE_DIRNAME",
    "CACHE_SCHEMA",
    "Snapshot",
    "cache_dir",
    "cache_path",
    "forget_resident",
    "peek",
    "refresh",
    "transcript_path",
]

"""The Library's background indexer: journal watch, incremental scan, thumbnails.

WHY A WORKER AND NOT A REQUEST PATH. The catalog used to be built inside the
list request: 620 journals scanned, 22,485 assets materialised and sorted in
Python (26.5 s cold on the operator's config). Every surface polls that route.
The index now lives in SQLite (see :mod:`local_operator.library.store`) and this
module is the only thing that writes it, so a list request is one indexed query.

WHAT DRIVES IT. ``watchdog`` watches ``<config>/sessions`` recursively. FSEvents
is the only mechanism on macOS that reports journal appends without polling, and
a hand-rolled ctypes shim for it was measured NOT to deliver events on this
machine while watchdog 6.0.0 did — hence a base dependency (inotify on Linux,
ReadDirectoryChangesW on Windows) rather than a bespoke binding per platform.
Events are COALESCED: one agent turn writes several journal lines, a subagent
fan-out writes hundreds of files, and each of those must become one scan, not
one scan per event.

WHEN IT IS NOT RUNNING. A first run (or a migration from the JSON index) starts
with ``backfill_done = 0`` and walks the visible sessions newest-first, one
transaction per session, so the grid fills from the top instead of showing an
empty page for half a minute. ``idx_sessions`` records each session's journal
signature, so a restart resumes instead of rescanning. If the observer cannot
start (no FSEvents, a missing directory) the worker falls back to a slow
reconcile: every 60 s while backfilling, every 6 h when idle. It never spins.

ONE WRITER. Reads and writes share the metadata database; every write here is a
short ``BEGIN IMMEDIATE`` transaction per session, and ``MetadataStore`` opens
its own connection per call, so the two never nest a connection inside another
transaction (that is how a writer deadlocks against itself).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from local_operator import resume
from local_operator.library import identity
from local_operator.library import index as library_index
from local_operator.library.metadata import MetadataStore
from local_operator.library.store import (
    LibraryStore,
    STATE_BACKFILL_DONE,
    STATE_BUILT_AT,
    STATE_LAST_RECONCILE,
    STATE_SESSIONS_SCANNED,
    STATE_SESSIONS_TOTAL,
)
from local_operator.session_files import service as files_service

logger = logging.getLogger(__name__)

#: How long a burst keeps being absorbed after its first event. A journal
#: append arrives as several events within milliseconds; 0.5 s is enough for one
#: turn's writes to land while staying under a user's perception of "instant".
SETTLE_S: Final = 0.5
#: Hard cap on one coalescing window. A build that keeps writing (a long tool
#: call streaming output) must still be indexed while it runs.
MAX_COALESCE_S: Final = 2.0
#: Reconcile cadence when the observer is unavailable, and when it is idle. A
#: timer shorter than the idle value is the spin this module exists to avoid.
RECONCILE_BACKFILL_S: Final = 60.0
RECONCILE_IDLE_S: Final = 6 * 3600.0

#: Thumbnails: the same 480 px WebP the request path used to produce inline,
#: moved off it. Quality 82 is the value the route shipped with.
THUMB_MAX_WIDTH: Final = 480
THUMB_QUALITY: Final = 82

#: One indexer per resolved config root. Keyed on the resolved path so two
#: spellings of one config share one worker (and one observer).
_INSTANCES: dict[str, "LibraryIndexer"] = {}
_INSTANCES_LOCK = threading.Lock()

#: One writer per config root ACROSS PROCESSES. The desktop server, a TUI
#: session and a measurement harness can all reach the same config, and three
#: workers would race on the same SQLite file, three observers would each queue
#: the same session, and the thumbnail queue would generate the same WebP three
#: times. Readers are unaffected: any process may read the database.
_WRITERS: dict[str, "_ProcessLock"] = {}


class _ProcessLock:
    """An exclusive advisory lock on ``<config>/cache/library/indexer.lock``.

    ``flock`` on POSIX, ``msvcrt.locking`` on Windows — both are released by the
    kernel when the process dies, which is the property that matters: a crashed
    indexer must never leave a lock that blocks the next run forever. The lock
    file is created, never truncated to a meaningful body; its CONTENT is not
    the lock.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return False
        try:
            if os.name == "nt":  # pragma: no cover - Windows only
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - Windows only
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            logger.debug("library: indexer lock release failed", exc_info=True)
        finally:
            os.close(fd)


def _acquire_writer(config_dir: str | Path) -> bool:
    """Become (or confirm being) this config's single writer. Never blocks."""
    key = str(Path(config_dir).resolve())
    with _INSTANCES_LOCK:
        if key in _WRITERS:
            # Already ours in this process: a second fd would be a second lock
            # and the tests' synchronous rebuild would deadlock against its own
            # worker.
            return True
        lock = _ProcessLock(Path(key) / "cache" / "library" / "indexer.lock")
        if not lock.acquire():
            return False
        _WRITERS[key] = lock
        return True


def _release_writer(config_dir: str | Path) -> None:
    key = str(Path(config_dir).resolve())
    with _INSTANCES_LOCK:
        lock = _WRITERS.pop(key, None)
    if lock is not None:
        lock.release()


def writer_available(config_dir: str | Path) -> bool:
    """Whether this process may write the index (read-only when another owns it)."""
    return str(Path(config_dir).resolve()) in _WRITERS


def ensure_started(config_dir: str | Path) -> "LibraryIndexer":
    """Start (or return) the indexer for a config root. Idempotent, never raises.

    Called from the server lifespan and lazily from :func:`service.list_assets`,
    so the TUI and the phone portal get an index too. A failure to start is
    logged and reported by the returned object's :attr:`running` flag rather
    than raised: an index that cannot start must not break the app that owns
    the journal.
    """
    key = str(Path(config_dir).resolve())
    with _INSTANCES_LOCK:
        instance = _INSTANCES.get(key)
        if instance is None:
            instance = LibraryIndexer(Path(key))
            _INSTANCES[key] = instance
    instance.start()
    return instance


def reset_for_tests() -> None:
    """Stop every indexer and forget them. Tests own the process lifetime."""
    with _INSTANCES_LOCK:
        instances = list(_INSTANCES.values())
        _INSTANCES.clear()
    for instance in instances:
        instance.stop()


def _titles(
    entry: dict[str, Any], session_id: str, title: str
) -> list[tuple[str, str | None, list[dict[str, Any]], str]]:
    """``(child_id, label, records, title)`` for a parent entry and its children."""
    groups: list[tuple[str, str | None, list[dict[str, Any]], str]] = [
        ("", None, entry.get("entries") or [], title)
    ]
    children = entry.get("children")
    if isinstance(children, dict):
        for child_id, child in children.items():
            if isinstance(child, dict):
                groups.append(
                    (str(child_id), child.get("label"), child.get("entries") or [], title)
                )
    return groups


def _occurrence_rows(session_id: str, entry: dict[str, Any], title: str) -> list[dict[str, Any]]:
    """Flatten one scanned session (parent + proven children) into index rows.

    Each row carries the RECORD fragments the aggregate merges (name, size,
    dimensions, content type) as well as the occurrence identity. Without them a
    session's removal could not recompute an asset's fields from its remaining
    occurrences, and an asset seen in two sessions would keep whichever name it
    happened to be indexed with.
    """
    rows: list[dict[str, Any]] = []
    for child_id, label, records, group_title in _titles(entry, session_id, title):
        for record in records:
            if not isinstance(record, dict):
                continue
            path = record.get("path")
            digest = record.get("digest")
            try:
                key = identity.key_for(path, digest)
            except ValueError:
                # A record with neither path nor digest cannot be identified;
                # ``_build_assets`` skipped those too.
                continue
            rows.append(
                {
                    "asset_id": identity.asset_id(key),
                    "key": key,
                    "session_id": session_id,
                    "title": group_title,
                    "child_id": child_id,
                    "label": str(label) if label else None,
                    "origin": str(record.get("origin") or "mentioned"),
                    "tool": str(record.get("tool") or ""),
                    "intent": str(record["intent"]) if record.get("intent") else None,
                    "first_ts": float(record.get("first_ts") or 0.0),
                    "last_ts": float(record.get("last_ts") or 0.0),
                    "first_seq": int(record.get("first_seq") or 0),
                    "last_seq": int(record.get("last_seq") or 0),
                    "touches": int(record.get("touches") or 1),
                    "highlighted": 1 if record.get("highlighted") else 0,
                    "name": str(record.get("name") or ""),
                    "path": str(path) if path else None,
                    "kind": str(record.get("kind") or "other"),
                    "content_type": record.get("content_type"),
                    "size_bytes": record.get("size"),
                    "width": record.get("width"),
                    "height": record.get("height"),
                    "duration_s": record.get("duration_s"),
                }
            )
    return rows


def _scannable(config_dir: Path, visible: Iterable[tuple[str, float, str, bool]]) -> int:
    """How many visible sessions actually have a journal to scan.

    The denominator, not the discovered count: a session DIRECTORY with no
    ``transcript.jsonl`` (a draft, or the empty dir a desktop session pool leaves
    behind) can never produce a checkpoint, so counting it would hold
    ``backfill_done`` at 0 and leave the UI on "scanning" forever — measured on
    the operator's config, where ``sessions/000000000000`` is an empty dir.
    """
    return sum(
        1
        for session_id, _mtime, _title, _flag in visible
        if library_index._signature(config_dir / "sessions" / session_id) is not None
    )


def _roster_key(session_dir: Path) -> str:
    """Identity of the parent's roster sidecar, as stored in the checkpoint.

    A roster-only edit adds or retracts whole child journals without touching
    the parent transcript, so the journal signature cannot see it. This is what
    makes "a subagent appeared" a dirty event even when no file in the parent
    session changed.
    """
    signature = library_index._roster_signature(session_dir)
    if not signature:
        return ""
    return "{size}:{mtime}:{inode}:{digest}".format(
        size=signature.get("size", 0),
        mtime=signature.get("mtime", 0),
        inode=signature.get("inode", 0),
        digest=signature.get("digest", ""),
    )


class _JournalHandler:
    """Map every filesystem event under ``sessions/`` to a session id.

    A duck-typed ``watchdog`` handler: subclassing ``FileSystemEventHandler``
    would bind the import to this module even when no observer ever starts, and
    the fallback path must work on a host where watchdog is unavailable.
    """

    def __init__(self, root: Path, on_session: Any) -> None:
        self.root = root
        self.on_session = on_session

    def dispatch(self, event: Any) -> None:
        self.on_any_event(event)

    def on_any_event(self, event: Any) -> None:
        session_id = self.session_id(event.src_path)
        if session_id is not None:
            self.on_session(session_id)

    def session_id(self, raw: str) -> str | None:
        try:
            relative = Path(raw).relative_to(self.root)
        except (ValueError, OSError):
            return None
        if not relative.parts:
            return None
        candidate = relative.parts[0]
        # One directory level under ``sessions/`` is a session; anything deeper
        # (a child journal, a roster sidecar, a thumbnail) belongs to that
        # session. A file directly in ``sessions/`` is not a session of ours.
        if len(relative.parts) == 1 and not (self.root / candidate).is_dir():
            return None
        return candidate


class LibraryIndexer:
    """One background worker per config root. Use :func:`ensure_started`."""

    def __init__(self, config_dir: str | Path, *, home: str | None = None) -> None:
        self.config_dir = Path(config_dir)
        self.home = home or str(Path.home())
        self.store = LibraryStore(self.config_dir)
        self._queue: queue.Queue[str] = queue.Queue()
        self._thumb_queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thumb_thread: threading.Thread | None = None
        self._observer: Any = None
        self._cwd_cache: dict[str, str] = {}
        self._deny: Any = None
        self._lock = threading.Lock()
        #: Set when the observer could not start; drives the slow reconcile.
        self.observer_failed = False

    # -- lifecycle ----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker (and the thumbnail worker). Idempotent.

        The writer lock is taken HERE, on the caller's thread, not inside the
        worker: a caller that must know whether this process owns the index (the
        server, a diagnostic, a test) cannot race the thread's first instruction.
        A process that loses the race still starts its threads — they read the
        database and retry the lock on the idle cadence.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if not _acquire_writer(self.config_dir):
                logger.info("library: another process owns the index; reader only")
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="library-indexer", daemon=True)
            self._thread.start()
            # NO thumbnail worker without the writer lock. It WRITES
            # ``idx_assets`` and decodes every image; a reader-only process that
            # ran it would break the single-writer rule and duplicate the owner's
            # work on the same files.
            if writer_available(self.config_dir):
                self._thumb_thread = threading.Thread(
                    target=self._run_thumbnails, name="library-thumbs", daemon=True
                )
                self._thumb_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        # Unblock both queue waits without inventing work.
        self._queue.put("")
        self._thumb_queue.put("")
        for thread in (self._thread, self._thumb_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout)
        self._stop_observer()
        # Another process may now take over. The lock is also released by the
        # kernel on exit, which is why a crashed run cannot wedge the next one.
        _release_writer(self.config_dir)

    def _stop_observer(self) -> None:
        observer = self._observer
        self._observer = None
        if observer is None:
            return
        try:
            observer.stop()
            observer.join(timeout=2.0)
        except Exception:  # noqa: BLE001 - a stuck observer must not block exit
            logger.debug("library: observer shutdown failed", exc_info=True)

    # -- thread body --------------------------------------------------------

    def _run(self) -> None:
        try:
            self.store.ensure_schema()
        except Exception:  # noqa: BLE001 - an unavailable index must not kill the app
            logger.warning("library: index schema unavailable; indexing off", exc_info=True)
            return
        self._deny = files_service.make_deny(self.config_dir)
        while not self._stop.is_set() and not _acquire_writer(self.config_dir):
            # Another process owns the writer. Keep serving READS from the
            # database (that is what this process's surfaces need) and retry on
            # the idle cadence — never in a tight loop.
            logger.info("library: another process owns the index; reader only")
            self._stop.wait(RECONCILE_IDLE_S)
        if self._stop.is_set():
            return
        try:
            self.reconcile()
        except Exception:  # noqa: BLE001 - a first reconcile is best effort
            logger.warning("library: initial reconcile failed", exc_info=True)
        self._start_observer()
        try:
            while not self._stop.is_set():
                try:
                    first = self._queue.get(timeout=self._idle_timeout())
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    try:
                        self.reconcile()
                    except Exception:  # noqa: BLE001 - keep the worker alive
                        logger.warning("library: periodic reconcile failed", exc_info=True)
                    continue
                if self._stop.is_set() or not first:
                    continue
                batch = self.coalesce(first)
                try:
                    self.process_dirty(batch)
                except Exception:  # noqa: BLE001 - one bad session must not stop the index
                    logger.warning("library: session batch failed", exc_info=True)
        finally:
            self._stop_observer()

    def _idle_timeout(self) -> float:
        """How long to wait before a reconcile. Never a spin, never a short poll."""
        try:
            done = self.store.state().get(STATE_BACKFILL_DONE, "0") == "1"
        except Exception:  # noqa: BLE001 - an unreadable state means "still building"
            done = False
        return RECONCILE_IDLE_S if done else RECONCILE_BACKFILL_S

    def _start_observer(self) -> None:
        root = self.config_dir / "sessions"
        if not root.is_dir():
            self.observer_failed = True
            logger.info("library: no sessions directory at %s; using slow reconcile", root)
            return
        try:
            from watchdog.observers import Observer

            # Duck-typed handler (``dispatch``/``on_any_event``): typed as Any so
            # this module does not import watchdog at module scope. A host
            # without the package still runs the app — it falls back to the
            # slow reconcile instead of failing to import the Library.
            handler: Any = _JournalHandler(root, self._mark_dirty)
            observer = Observer()
            observer.schedule(handler, str(root), recursive=True)
            observer.start()
            self._observer = observer
            self.observer_failed = False
        except Exception:  # noqa: BLE001 - fall back to the timer, never to a crash
            self.observer_failed = True
            logger.warning(
                "library: journal watch unavailable; using slow reconcile", exc_info=True
            )

    def _mark_dirty(self, session_id: str) -> None:
        if session_id:
            self._queue.put(session_id)

    # -- coalescing ---------------------------------------------------------

    def coalesce(self, first: str) -> set[str]:
        """Absorb a burst into one batch: settle ~0.5 s, never longer than 2 s.

        Exposed (rather than inlined in the loop) because that is the unit the
        tests exercise: OS events are not reproducible in CI, the dirty-set
        contract is.
        """
        dirty: set[str] = {first} if first else set()
        start = time.monotonic()
        settle = start + SETTLE_S
        hard = start + MAX_COALESCE_S
        while True:
            now = time.monotonic()
            if now >= hard or self._stop.is_set():
                break
            if now < settle:
                timeout = min(settle, hard) - now
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    continue
                if item:
                    dirty.add(item)
                continue
            # The settle window has passed: take whatever is already queued and
            # stop, so one session's arrival does not delay the batch forever.
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item:
                dirty.add(item)
        return dirty

    # -- scanning -----------------------------------------------------------

    def process_dirty(self, session_ids: Iterable[str]) -> int:
        """Scan every dirty session once. Returns how many were rescanned."""
        scanned = 0
        for session_id in session_ids:
            if self._stop.is_set():
                break
            try:
                if self.scan_session(session_id):
                    scanned += 1
            except Exception:  # noqa: BLE001 - one session cannot poison the batch
                logger.warning("library: scan failed for %s", session_id, exc_info=True)
        return scanned

    def _cwd(self, session_id: str, session_dir: Path) -> str:
        """Cached per-session cwd: the registry fallback costs ~20 ms a call."""
        cached = self._cwd_cache.get(session_id)
        if cached is not None:
            return cached
        cwd = files_service.session_cwd(self.config_dir, session_id, session_dir)
        self._cwd_cache[session_id] = cwd
        return cwd

    def _forget(self, session_id: str) -> None:
        with self.store.write() as connection:
            affected = self.store.forget_session(connection, session_id)
            connection.execute("DELETE FROM idx_sessions WHERE session_id = ?", (session_id,))
            self.store.recompute_assets(connection, affected)
            self.store.bump_revision(connection)

    def scan_session(self, session_id: str) -> bool:
        """Index one session if its journal moved. Returns whether it was scanned."""
        session_dir = self.config_dir / "sessions" / session_id
        if not session_dir.is_dir():
            self._forget(session_id)
            return False
        signature = library_index._signature(session_dir)
        if signature is None:
            # A draft directory with no journal yet: nothing to index, and it is
            # not an error the next pass should retry every 60 s.
            return False
        known = self.store.session(session_id)
        title_key = self._stored_title(session_dir)
        roster_key = _roster_key(session_dir)
        if known is not None and known.get("complete") and self._same_signature(known, signature):
            # The journal did not move, but the LIST is filtered and searched by
            # conversation title and a child journal is only reachable through
            # the parent roster, so both are part of the checkpoint: a
            # ``/rename`` and a roster-only membership edit trigger no append.
            if str(known.get("title_key") or "") != title_key:
                self._refresh_title(session_id, title_key)
            if str(known.get("roster_key") or "") == roster_key:
                return False
        entry, revision = library_index._session_entry(
            self.config_dir,
            session_id,
            session_dir,
            title=title_key,
            cwd=self._cwd(session_id, session_dir),
            home=self.home,
            deny=self._deny or files_service.make_deny(self.config_dir),
            budget_s=None,
        )
        if entry is None:
            self._forget(session_id)
            return False
        # The resolved label: the stored title, else the scan's opening line.
        title = str(entry.get("title") or "") or title_key or session_id
        rows = _occurrence_rows(session_id, entry, title)
        assets = {row["asset_id"]: (row["key"], row["name"]) for row in rows if row["asset_id"]}
        previous = self.store.occurrence_assets(session_id)
        affected = set(assets) | previous
        # Outside the write transaction on purpose: MetadataStore opens its own
        # connection, and a writer waiting on its own lock is a deadlock, not a
        # slow write.
        MetadataStore(self.config_dir).ensure_assets(
            (asset_id, assets[asset_id][0], assets[asset_id][1] or None) for asset_id in assets
        )
        with self.store.write() as connection:
            self.store.replace_session_occurrences(connection, session_id, rows)
            self.store.recompute_assets(connection, affected)
            self.store.record_session(
                connection,
                session_id,
                title=title,
                title_key=title_key,
                roster_key=roster_key,
                sig=signature,
                complete=bool(entry.get("complete")),
                revision=revision,
            )
            self.store.bump_revision(connection)
        self._enqueue_thumbs(assets)
        logger.info(
            "library: indexed %s (%d records, %d assets, revision %d)",
            session_id,
            len(rows),
            len(affected),
            self.store.revision(),
        )
        self._maybe_finish_backfill()
        return True

    def _refresh_title(self, session_id: str, title: str) -> None:
        with self.store.write() as connection:
            affected = self.store.refresh_session_title(connection, session_id, title)
            self.store.recompute_assets(connection, affected)
            self.store.bump_revision(connection)

    @staticmethod
    def _same_signature(known: dict[str, Any], signature: dict[str, Any]) -> bool:
        return library_index._sig_matches(
            {
                "size": known.get("sig_size"),
                "mtime": known.get("sig_mtime"),
                "inode": known.get("sig_inode"),
            },
            signature,
        )

    def _stored_title(self, session_dir: Path) -> str:
        """The title the user last saw, or ``""`` when nobody ever titled it.

        Empty, NOT the session id: ``_session_entry`` falls back to the scan's
        opening-line capture for an untitled session, which is the label the
        Library has always shown. Returning the id here made that fallback
        unreachable and every untitled conversation read as a hex string.
        """
        try:
            return resume.stored_session_title(session_dir) or ""
        except Exception:  # noqa: BLE001 - a title is a label, never a failure
            return ""

    # -- backfill and reconcile ---------------------------------------------

    def _maybe_finish_backfill(self) -> None:
        state = self.store.state()
        if state.get(STATE_BACKFILL_DONE) == "1":
            return
        if self.store.incomplete_sessions():
            return
        with self.store.write() as connection:
            self.store.set_state(
                connection,
                **{
                    STATE_BACKFILL_DONE: 1,
                    STATE_SESSIONS_SCANNED: self.store._count_sessions(connection, complete=True),
                    STATE_BUILT_AT: time.time(),
                },
            )
        self._drop_legacy_index()

    def _drop_legacy_index(self) -> None:
        """Delete the JSON catalog this module replaced. It is derived.

        Left behind it costs 11 MB and misleads anyone reading the config dir
        into thinking it is still the source; the log line records the reclaim.
        """
        cache = self.config_dir / "cache" / "library"
        freed = 0
        for name in (library_index.INDEX_FILENAME, f"{library_index.INDEX_FILENAME}.lock"):
            path = cache / name
            try:
                freed += path.stat().st_size
                path.unlink()
            except OSError:
                continue
        if freed:
            logger.info("library: removed the superseded JSON index (%.1f MB)", freed / 1e6)

    def reconcile(self) -> None:
        """Enumerate visible sessions, drop the vanished ones, queue the moved.

        Cheap when nothing changed: one session-store listing, one stat per
        visible journal, and no write unless the visible set or a signature
        actually moved.
        """
        visible = library_index._visible_sessions(self.config_dir)
        live = {row[0] for row in visible}
        known = self.store.sessions()
        vanished = set(known) - live
        for session_id in vanished:
            self._forget(session_id)
            self._cwd_cache.pop(session_id, None)
        dirty = 0
        for session_id, _mtime, _title, _flag in visible:
            session_dir = self.config_dir / "sessions" / session_id
            signature = library_index._signature(session_dir)
            if signature is None:
                continue
            row = known.get(session_id)
            if row is not None and row.get("complete") and self._same_signature(row, signature):
                continue
            self._mark_dirty(session_id)
            dirty += 1
        with self.store.write() as connection:
            self.store.set_state(
                connection,
                **{
                    STATE_SESSIONS_TOTAL: _scannable(self.config_dir, visible),
                    STATE_LAST_RECONCILE: time.time(),
                },
            )
        if not visible:
            # An empty config is a finished (empty) index, not a build in
            # progress: the UI must not show "scanning" forever.
            self._maybe_finish_backfill()
        logger.debug(
            "library: reconciled %d visible sessions (%d dirty, %d removed)",
            len(visible),
            dirty,
            len(vanished),
        )

    # -- thumbnails ---------------------------------------------------------

    def enqueue_thumbnail(self, asset_id: str) -> None:
        """Ask for one thumbnail. Called from a read; the queue drops duplicates.

        Never blocks: the request path must not wait on an image decode, and the
        worker re-checks liveness so a duplicate costs one stat.
        """
        if asset_id and self.running and writer_available(self.config_dir):
            self._thumb_queue.put(asset_id)

    def _enqueue_thumbs(self, assets: Iterable[str]) -> None:
        """Queue freshly indexed images; the worker decides whether one is stale."""
        try:
            for asset_id in self.store.thumbnail_candidates(assets):
                self._thumb_queue.put(asset_id)
        except Exception:  # noqa: BLE001 - thumbnails are cosmetic
            logger.debug("library: thumbnail enqueue failed", exc_info=True)

    def _run_thumbnails(self) -> None:
        try:
            for asset_id in self.store.thumbnail_candidates():
                self._thumb_queue.put(asset_id)
        except Exception:  # noqa: BLE001 - a startup miss is retried per session
            logger.debug("library: thumbnail sweep failed", exc_info=True)
        while not self._stop.is_set():
            try:
                asset_id = self._thumb_queue.get(timeout=60.0)
            except queue.Empty:
                continue
            if self._stop.is_set() or not asset_id:
                continue
            if not writer_available(self.config_dir):
                # Ownership moved (or was never ours): stop writing, keep reading.
                continue
            try:
                self.store.ensure_thumbnail(asset_id, width=THUMB_MAX_WIDTH, quality=THUMB_QUALITY)
            except Exception:  # noqa: BLE001 - one bad image must not stop the queue
                logger.debug("library: thumbnail failed for %s", asset_id, exc_info=True)

    # -- synchronous entry points -------------------------------------------

    def rebuild(self) -> int:
        """Index every visible session to completion, in the calling thread.

        The tests' and the CLI's entry point: no observer, no thread, and a
        return value that says how many sessions were actually rescanned. This
        is also what makes a partial backfill resumable — it skips every session
        whose checkpoint still matches its journal.
        """
        if not _acquire_writer(self.config_dir):
            logger.warning("library: rebuild skipped, another process owns the index")
            return 0
        self.store.ensure_schema()
        self._deny = files_service.make_deny(self.config_dir)
        visible = library_index._visible_sessions(self.config_dir)
        live = {row[0] for row in visible}
        for session_id in set(self.store.sessions()) - live:
            self._forget(session_id)
        # The denominator goes in FIRST: ``_maybe_finish_backfill`` runs after
        # every session, and with an unset total the first scan looked like a
        # finished backfill (0 owed) and froze ``sessions_scanned`` at 1.
        with self.store.write() as connection:
            self.store.set_state(
                connection, **{STATE_SESSIONS_TOTAL: _scannable(self.config_dir, visible)}
            )
        scanned = 0
        for session_id, _mtime, title, _flag in visible:
            session_dir = self.config_dir / "sessions" / session_id
            if library_index._signature(session_dir) is None:
                continue
            if self.scan_session(session_id):
                scanned += 1
        self._maybe_finish_backfill()
        return scanned


def rebuild_now(config_dir: str | Path, *, home: str | None = None) -> int:
    """Build the index synchronously for a config root. Returns sessions scanned."""
    return LibraryIndexer(Path(config_dir), home=home).rebuild()


def index_state(config_dir: str | Path) -> dict[str, Any]:
    """``idx_state`` plus the two counters, for diagnostics and tests."""
    store = LibraryStore(config_dir)
    state = store.state()
    listing = store.listing_state()
    return {**state, "state": listing[0], "scanned": listing[1], "total": listing[2]}


def wait_until_idle(indexer: LibraryIndexer, timeout: float = 30.0) -> bool:
    """Block until the worker's queue is empty. Tests only; never on a request path."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if indexer._queue.empty():
            return True
        time.sleep(0.02)
    return False


__all__ = [
    "LibraryIndexer",
    "ensure_started",
    "index_state",
    "rebuild_now",
    "reset_for_tests",
    "writer_available",
    "wait_until_idle",
]

"""The derived Library catalog (design §3.2).

WHY A SECOND STORE. ``session_files`` answers per conversation and rebuilds
behind a 1.5 s request budget. The Library needs one cross-conversation view,
and paying every session's full journal scan on every list call would make the
catalog unusable on a large store. So this module keeps a DERIVED cache under
``<config>/cache/library/index.json``: per session, the scanner's records plus
a journal signature and revision. A signature match serves the cached rows with
one stat; anything else refreshes only that session through the existing
``session_files.ledger`` ladder.

DERIVED, NEVER AUTHORITATIVE. The file is discardable: missing, torn, or written
by another schema costs a rebuild, never an error. Metadata (names, tags,
chats) lives in ``library.sqlite3`` and is unaffected.

INCREMENTAL AND RESUMABLE. A cold build stops at a time budget, persists what
it scanned, and answers ``state: "building"`` with honest progress. The caller
finishes the rest in the background — the same shape the session-files route
uses, because it is the same problem.

SUBAGENT JOURNALS. The scanner hides subagent directories from the session
list. Their assets still belong to the conversation that launched them, so a
parent's visible children (proven by the existing containment helper) are
scanned into the parent's entry with the child id kept on the occurrence. A
journal without proof is excluded, exactly as the design says.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

try:  # POSIX advisory locking; Windows has no fcntl and writes stay serialised per process.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]

from local_operator import resume
from local_operator.library import classify, identity
from local_operator.session_files import ledger, service
from local_operator.session_files.scan import ORIGINS, FileRecord, resolve_path

logger = logging.getLogger(__name__)

INDEX_DIRNAME: Final = "library"
INDEX_FILENAME: Final = "index.json"
#: Bumped to 4 when the occurrence title stopped taking the scan's
#: opening-line capture over the stored session title (see ``_session_entry``).
#: The freshness comparison in ``_build_locked`` would heal a same-schema cache
#: on its next build, so this is not the only repair; the bump makes the
#: correction UNCONDITIONAL instead — a stored field changed meaning, and every
#: existing cache should be rebuilt once rather than depend on a comparison
#: that a future edit of this file could weaken. Index files are derived and
#: discardable and a schema mismatch rebuilds them: it costs one rescan, never
#: an error.
INDEX_SCHEMA: Final = 4
DEFAULT_BUDGET_S: Final = service.DEFAULT_BUDGET_S

#: How long a request-path caller may reuse the in-process aggregate without a
#: session sweep. The desktop grid polls every ~2 s; without this window an
#: actively-written journal changed its signature on every poll and dragged the
#: whole index rebuild onto the request path. ``budget_s=None`` (background
#: finisher, CLI, tests) always scans: it exists to make progress, not to poll.
FRESH_WINDOW_S: Final = 2.0

#: Non-structural churn (a revision bump inside an existing entry) rewrites the
#: 11 MB index at most this often. A session id that appears for the first time
#: or an entry that first becomes complete bypasses the limit: those are the
#: events a reader must not miss, and they happen once per conversation.
WRITE_INTERVAL_S: Final = 10.0

#: ``(index file identity, monotonic build time, aggregate)`` per resolved index
#: path. The identity is what makes returning the cached aggregate sound: an
#: unchanged file plus the signature sweep below proves the session map the
#: aggregate was built from is still the one on disk.
_CACHE: dict[str, tuple[tuple[int, int, int] | None, float, "Catalog"]] = {}
#: One ``threading.Lock`` per index path: one build (or sweep) at a time.
_LOCKS: dict[str, threading.Lock] = {}
#: One lock per index path for the persisted-index asset build in
#: :func:`catalog_for_read`; separate from the scan lock so a thumbnail never
#: queues behind a background full build.
_ASSET_LOCKS: dict[str, threading.Lock] = {}
#: Paths with a full build in flight, so concurrent callers can report building.
_BUILDING: set[str] = set()
_CACHE_LOCK = threading.Lock()

#: The three groups the Library shows. Raw origins stay on each occurrence.
ORIGIN_GROUPS: Final = ("generated", "created", "mentioned")
_CREATED_ORIGINS: Final = frozenset({"written", "screenshot", "download", "edited", "command"})
_GENERATED_ORIGINS: Final = frozenset({"generated"})

#: The parent's own sidecar naming every child/grandchild run. Its changes are
#: membership changes: a roster-only edit (the parent transcript untouched) can
#: add or retract a whole journal, and a signature that watched only
#: ``transcript.jsonl`` served the old family forever.
ROSTER_SIDECAR: Final = "subagent-roster.v1.json"
#: Last parsed digest per sidecar, keyed on its stat identity. The full hash is
#: only re-read when the file moved; an atomic replace changes the inode, which
#: is the writer's actual pattern.
_ROSTER_DIGESTS: dict[str, tuple[tuple[int, int, int], str]] = {}


def _roster_signature(session_dir: Path) -> dict[str, Any] | None:
    """Identity of the parent roster sidecar, ``None`` when there is none."""
    path = session_dir / ROSTER_SIDECAR
    try:
        info = path.stat()
    except OSError:
        return None
    key = (info.st_size, info.st_mtime_ns, info.st_ino)
    held = _ROSTER_DIGESTS.get(str(path))
    if held is not None and held[0] == key:
        digest = held[1]
    else:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = ""
        _ROSTER_DIGESTS[str(path)] = (key, digest)
    return {"size": info.st_size, "mtime": info.st_mtime_ns, "inode": info.st_ino, "digest": digest}


def _roster_matches(cached: Any, current: dict[str, Any] | None) -> bool:
    if cached is None and current is None:
        return True
    if not isinstance(cached, dict) or not isinstance(current, dict):
        return False
    return (
        cached.get("digest") == current.get("digest")
        and cached.get("size") == current.get("size")
        and cached.get("mtime") == current.get("mtime")
        and cached.get("inode") == current.get("inode")
    )


def origin_group(origins: list[str] | tuple[str, ...]) -> str:
    """Project raw origins onto the three user-facing groups, strongest first."""
    if any(origin in _GENERATED_ORIGINS for origin in origins):
        return "generated"
    if any(origin in _CREATED_ORIGINS for origin in origins):
        return "created"
    return "mentioned"


@dataclass
class Occurrence:
    """One session (or proven child) that saw the asset."""

    session_id: str
    title: str
    origin: str
    child_id: str | None = None
    label: str | None = None
    tool: str = ""
    intent: str | None = None
    first_ts: float = 0.0
    last_ts: float = 0.0
    first_seq: int = 0
    last_seq: int = 0
    touches: int = 1
    highlighted: bool = False

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Asset:
    """One deduplicated artifact across every conversation."""

    key: str
    id: str
    path: str | None = None
    digest: str | None = None
    derived_name: str = ""
    kind: str = "other"
    category: str = "other"
    content_type: str = "application/octet-stream"
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    occurrences: list[Occurrence] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)
    highlighted: bool = False

    @property
    def origin(self) -> str:
        return self.origins[0] if self.origins else "mentioned"

    @property
    def origin_group(self) -> str:
        return origin_group(self.origins)

    @property
    def ref_type(self) -> str:
        return "artifact" if self.digest else "path"

    @property
    def first_ts(self) -> float:
        values = [o.first_ts for o in self.occurrences if o.first_ts]
        return min(values) if values else 0.0

    @property
    def last_ts(self) -> float:
        values = [o.last_ts for o in self.occurrences if o.last_ts]
        return max(values) if values else 0.0

    @property
    def session_titles(self) -> list[str]:
        titles: list[str] = []
        for occurrence in self.occurrences:
            if occurrence.title and occurrence.title not in titles:
                titles.append(occurrence.title)
        return titles


@dataclass
class Catalog:
    """The aggregate one list request answers from."""

    assets: list[Asset]
    revision: int
    state: str
    sessions_total: int
    sessions_scanned: int
    built_at: float

    @property
    def building(self) -> bool:
        return self.state == "building"


def index_path(config_dir: str | Path) -> Path:
    return Path(config_dir) / "cache" / INDEX_DIRNAME / INDEX_FILENAME


def _record_dict(record: FileRecord) -> dict[str, Any]:
    return {
        "key": record.key,
        "path": record.path,
        "digest": record.digest,
        "name": record.name,
        "kind": record.kind,
        "content_type": record.content_type,
        "size": record.size,
        "width": record.width,
        "height": record.height,
        "duration_s": record.duration_s,
        "origin": record.origin,
        "tool": record.tool,
        "call_id": record.call_id,
        "intent": record.intent,
        "first_seq": record.first_seq,
        "last_seq": record.last_seq,
        "first_ts": record.first_ts,
        "last_ts": record.last_ts,
        "touches": record.touches,
    }


def _cache_key(config: Path) -> str:
    """Resolved index path: two spellings of one config share one aggregate."""
    return os.path.realpath(str(index_path(config)))


def _index_stat(path: Path) -> tuple[int, int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return (int(info.st_mtime_ns), int(info.st_size), int(getattr(info, "st_ino", 0) or 0))


def _cached(key: str) -> tuple[tuple[int, int, int] | None, float, "Catalog"] | None:
    with _CACHE_LOCK:
        return _CACHE.get(key)


def _remember(key: str, stat: tuple[int, int, int] | None, catalog: "Catalog") -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (stat, time.monotonic(), catalog)


def _lock_for(key: str) -> threading.Lock:
    with _CACHE_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _asset_lock_for(key: str) -> threading.Lock:
    with _CACHE_LOCK:
        lock = _ASSET_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _ASSET_LOCKS[key] = lock
        return lock


def _serve(key: str, catalog: "Catalog") -> "Catalog":
    """The cached aggregate, marked building while another caller's build runs."""
    with _CACHE_LOCK:
        busy = key in _BUILDING
    if busy and catalog.state != "building":
        return dataclasses.replace(catalog, state="building")
    return catalog


def _load(path: Path) -> dict[str, Any]:
    """Read the derived index. Any failure is an empty catalog, never an error."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != INDEX_SCHEMA:
        return {}
    sessions = payload.get("sessions")
    return payload if isinstance(sessions, dict) else {}


@contextmanager
def _index_lock(path: Path):
    """Advisory cross-process lock; yields ``False`` when another process holds it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
    except OSError:
        yield False
        return
    try:
        locked = True
        if _fcntl is not None:
            try:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            except OSError:
                locked = False
        try:
            yield locked
        finally:
            if locked and _fcntl is not None:
                try:
                    _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        handle.close()


def _entry_revision_value(entry: Any) -> float:
    try:
        return float(entry.get("revision") or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _merge_sessions(
    disk: dict[str, Any], incoming: dict[str, Any], *, keep: set[str]
) -> dict[str, Any]:
    """Merge a build into what another process may have persisted meanwhile.

    A COMPLETE incoming entry is authoritative for the key it covers: this
    build either refreshed it from the journal or verified its signature, and a
    revision comparison cannot decide freshness — retracting a child lowers the
    parent's revision below the stale disk entry's. An INCOMPLETE incoming entry
    must not replace a complete disk entry (a budget-limited scan that started
    and ran out); disk keeps the complete rows. ``keep`` names the sessions
    still visible to this build, so a complete disk entry for a session this
    build did not reach also survives.
    """
    merged: dict[str, Any] = {
        session_id: entry for session_id, entry in incoming.items() if isinstance(entry, dict)
    }
    for session_id, entry in disk.items():
        if not isinstance(entry, dict):
            continue
        held = merged.get(session_id)
        if isinstance(held, dict):
            if not held.get("complete") and entry.get("complete"):
                merged[session_id] = entry
            continue
        if session_id in keep and entry.get("complete"):
            merged[session_id] = entry
    return merged


def _structural_change(disk: dict[str, Any], merged: dict[str, Any]) -> bool:
    """A new session, a first completion, or a title change: what a poll must not miss.

    A title differs here only when the build rescanned an entry whose stored
    title moved, and ``/rename`` is the case that does it: it writes the sidecar
    and leaves the transcript — and so the journal signature — untouched. The
    write rate limit is keyed on this predicate, so leaving a title out would
    keep a request-path poll serving the pre-rename name for up to
    ``WRITE_INTERVAL_S`` (measured: converged at 10 s), which reads as a rename
    that did not take.
    """
    for session_id, entry in merged.items():
        held = disk.get(session_id)
        if not isinstance(held, dict):
            return True
        if entry.get("complete") and not held.get("complete"):
            return True
        if str(entry.get("title") or "") != str(held.get("title") or ""):
            return True
    return any(session_id not in merged for session_id in disk)


def _written_recently(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime < WRITE_INTERVAL_S
    except OSError:
        return False


def _persist(
    path: Path,
    sessions: dict[str, Any],
    *,
    visible: set[str],
    force: bool = False,
    partial: bool = False,
) -> tuple[dict[str, Any], float, bool, tuple[int, int, int] | None] | None:
    """Read-merge-write under the cross-process lock.

    Returns ``(merged sessions, built_at, wrote, stat)``, or ``None`` when
    another process holds the lock — the caller then serves its last aggregate
    rather than waiting. ``stat`` is the index identity observed UNDER the lock
    (after any write), which is the identity of the payload ``merged`` came
    from. The rate limit applies only to non-structural churn: a new session, a
    first completion or a stored-title change always lands. The top-level
    ``complete`` flag lets a reader tell a finished index from a budget-limited
    one that has only complete rows so far.
    """
    with _index_lock(path.with_name("index.lock")) as locked:
        if not locked:
            return None
        stat = _index_stat(path)
        disk = _load(path)
        raw = disk.get("sessions")
        disk_sessions: dict[str, Any] = raw if isinstance(raw, dict) else {}
        # ``sessions`` is authoritative for what it contains; ``visible`` lets a
        # complete disk entry survive for a session this build did not reach.
        merged = _merge_sessions(disk_sessions, sessions, keep=visible)
        built_at = float(disk.get("built_at") or 0.0)
        complete = not partial
        flag_changed = disk.get("complete") is not complete
        if merged == disk_sessions and not flag_changed:
            return merged, built_at, False, stat
        if (
            not force
            and not flag_changed
            and not _structural_change(disk_sessions, merged)
            and _written_recently(path)
        ):
            return merged, built_at, False, stat
        stamp = time.time()
        revision = int(sum(_entry_revision_value(entry) for entry in merged.values()))
        _write(
            path,
            {
                "version": INDEX_SCHEMA,
                "built_at": stamp,
                "revision": revision,
                "complete": complete,
                "sessions": merged,
            },
        )
        return merged, stamp, True, _index_stat(path)


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write; a torn cache costs a rebuild and must never half-replace one."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary, path)
    except OSError:
        logger.debug("library index write failed", exc_info=True)


def _signature(session_dir: Path) -> dict[str, int | float] | None:
    """Cheap journal identity. Same fields the scanner's ledger compares."""
    try:
        info = ledger.transcript_path(session_dir).stat()
    except OSError:
        return None
    return {
        "size": int(info.st_size),
        "mtime": float(info.st_mtime),
        "inode": int(getattr(info, "st_ino", 0) or 0),
    }


def _sig_matches(cached: Any, current: dict[str, int | float] | None) -> bool:
    if not isinstance(cached, dict) or current is None:
        return False
    try:
        return (
            int(cached.get("size", -1)) == current["size"]
            and float(cached.get("mtime", -1)) == current["mtime"]
            and int(cached.get("inode", -1)) == current["inode"]
        )
    except (TypeError, ValueError):
        return False


def _visible_sessions(config_dir: Path) -> list[tuple[str, float, str, bool]]:
    """User-visible sessions, newest activity first."""
    try:
        # ``_scan_sessions`` answers ``(rows, hidden_names)``; the hidden set is
        # the containment helper's input, not ours — subagent attribution is
        # proven per visible parent below.
        rows, _hidden = resume._scan_sessions(config_dir)
    except Exception:  # noqa: BLE001 - a store read is an empty catalog, not a 500
        logger.warning("library: session store could not be read", exc_info=True)
        return []
    visible = []
    for row in rows:
        session_id = str(row[0])
        session_dir = config_dir / "sessions" / session_id
        if not session_dir.is_dir():
            continue
        if not resume.is_user_session(session_dir):
            continue
        visible.append((session_id, float(row[1]), str(row[2]), bool(row[3])))
    visible.sort(key=lambda item: item[1], reverse=True)
    return visible


def _highlight_paths(
    snapshot: ledger.Snapshot | None, *, cwd: str, home: str, deny: Any
) -> set[str]:
    """Newest-version highlight paths for one session, resolved to absolute paths.

    ``scan.FileScan.anchors`` already keeps only the newest version per anchor,
    so a retracted callout is absent by construction. Highlight is a FLAG on
    the asset, not a separate asset, so these paths are merged into the record
    set below.
    """
    if snapshot is None:
        return set()
    found: set[str] = set()
    for anchor in snapshot.scan.anchors.values():
        for item in anchor.files:
            raw = item.get("path") if isinstance(item, dict) else None
            if not isinstance(raw, str) or not raw:
                continue
            path = resolve_path(raw, cwd=cwd, home=home)
            if path and not deny(path) and os.path.isfile(path):
                found.add(path)
        for raw in anchor.more:
            path = resolve_path(raw, cwd=cwd, home=home)
            if path and not deny(path) and os.path.isfile(path):
                found.add(path)
    return found


def _child_entry(
    config_dir: Path,
    child_id: str,
    *,
    label: str | None,
    cwd: str,
    home: str,
    deny: Any,
    budget_s: float | None,
) -> dict[str, Any] | None:
    """Refresh one PROVEN child journal into its parent's cache entry."""
    child_dir = config_dir / "sessions" / child_id
    snapshot = ledger.refresh(
        config_dir,
        child_id,
        child_dir,
        cwd=service.session_cwd(config_dir, child_id, child_dir) or cwd,
        home=home,
        deny=deny,
        budget_s=budget_s,
    )
    if snapshot is None:
        return None
    return {
        "label": label,
        "sig": dict(snapshot.sig),
        "revision": snapshot.revision,
        "complete": bool(snapshot.complete),
        "entries": [_record_dict(record) for record in snapshot.scan.records.values()],
    }


def _entry_revision(entry: dict[str, Any]) -> float:
    """Parent revision plus every child's: the number the catalog sums.

    Child revisions used to be computed and thrown away, so a child append left
    the catalog revision identical and every warm reader saw stale rows.
    """
    total = float(entry.get("parent_revision") or entry.get("revision") or 0.0)
    children = entry.get("children")
    if isinstance(children, dict):
        total += sum(
            float(child.get("revision") or 0.0)
            for child in children.values()
            if isinstance(child, dict)
        )
    return total


def _session_entry(
    config_dir: Path,
    session_id: str,
    session_dir: Path,
    *,
    title: str,
    cwd: str,
    home: str,
    deny: Any,
    budget_s: float | None,
) -> tuple[dict[str, Any] | None, float]:
    """Refresh one visible session (parent + proven children). Returns ``(entry, revision)``."""
    snapshot = ledger.refresh(
        config_dir,
        session_id,
        session_dir,
        cwd=cwd,
        home=home,
        deny=deny,
        budget_s=budget_s,
    )
    if snapshot is None:
        return None, 0.0
    # The stored title wins: it is the name the user last SAW (auto-generated on
    # the first substantive turn, or typed at ``/rename``), and the Library is
    # rendering a conversation reference, not a transcript excerpt. The scan's
    # ``title`` is ``sanitize_prompt_line(texts[0])`` — the session's OPENING
    # user message — so preferring it labelled every occurrence with the first
    # thing anyone typed, however long ago, and a renamed session never showed
    # its name. It survives only as the fallback for a session nobody ever
    # titled (one written before titles were journalled), where it is the best
    # label available and costs nothing.
    title = title or snapshot.scan.title
    entries = [_record_dict(record) for record in snapshot.scan.records.values()]
    highlights = _highlight_paths(snapshot, cwd=cwd, home=home, deny=deny)
    for entry in entries:
        if entry.get("path") in highlights:
            entry["highlighted"] = True
    known = {entry.get("path") for entry in entries}
    for path in highlights - known:
        classification = classify.classify(path)
        entries.append(
            {
                "key": f"p:{path}",
                "path": path,
                "digest": None,
                "name": os.path.basename(path) or path,
                "kind": classification.kind,
                "content_type": classification.content_type,
                "size": None,
                "width": None,
                "height": None,
                "duration_s": None,
                "origin": "highlight",
                "tool": "highlight",
                "call_id": None,
                "intent": None,
                "first_seq": 0,
                "last_seq": 0,
                "first_ts": 0.0,
                "last_ts": 0.0,
                "touches": 1,
                "highlighted": True,
            }
        )

    children: dict[str, Any] = {}
    total_revision = float(snapshot.revision)
    complete = bool(snapshot.complete)
    # The containment proof lives beside the desktop session pool, whose module
    # graph costs ~2.3 s to import. Only a session that actually HAS a roster
    # can have children, so a normal conversation never pays it.
    roster_sig = _roster_signature(session_dir)
    has_roster = roster_sig is not None or (
        isinstance(snapshot.scan.roster, dict) and bool(snapshot.scan.roster.get("details"))
    )
    proven: list[Any] = []
    if has_roster:
        try:
            from local_operator.server.utils.desktop_sessions import (
                contained_descendants,
            )

            proven, _skipped = contained_descendants(
                config_dir,
                session_id,
                indexed_roster=snapshot.scan.roster,
                indexed=True,
            )
        except Exception:  # noqa: BLE001 - an unprovable roster is an empty child set
            proven = []
    for child in proven:
        child_entry = _child_entry(
            config_dir,
            child.child_id,
            label=child.label,
            cwd=cwd,
            home=home,
            deny=deny,
            budget_s=budget_s,
        )
        if child_entry is None:
            continue
        children[child.child_id] = child_entry
        total_revision += float(child_entry["revision"])
        complete = complete and bool(child_entry["complete"])

    return (
        {
            "sig": dict(snapshot.sig),
            "roster_sig": roster_sig,
            "parent_revision": snapshot.revision,
            "revision": total_revision,
            "title": title,
            "cwd": cwd,
            "complete": complete,
            "entries": entries,
            "children": children,
        },
        total_revision,
    )


def _build_assets(sessions: dict[str, Any]) -> list[Asset]:
    """Merge per-session records into deduplicated assets.

    One row per key; every contributing session becomes an occurrence. A path
    seen in two sessions is one asset; the strongest origin wins the row's
    origin while ``origins`` keeps the full evidence list.
    """
    by_key: dict[str, Asset] = {}
    for session_id, session in sessions.items():
        if not isinstance(session, dict):
            continue
        title = str(session.get("title") or "")
        groups: list[tuple[str | None, str | None, list[dict[str, Any]]]] = [
            (None, None, session.get("entries") or [])
        ]
        children = session.get("children")
        if isinstance(children, dict):
            for child_id, child in children.items():
                if isinstance(child, dict):
                    groups.append((str(child_id), child.get("label"), child.get("entries") or []))
        for child_id, label, entries in groups:
            for record in entries:
                if not isinstance(record, dict):
                    continue
                try:
                    key = identity.key_for(record.get("path"), record.get("digest"))
                except ValueError:
                    continue
                asset = by_key.get(key)
                if asset is None:
                    asset = Asset(key=key, id=identity.asset_id(key))
                    by_key[key] = asset
                if asset.path is None and record.get("path"):
                    asset.path = str(record["path"])
                if asset.digest is None and record.get("digest"):
                    asset.digest = str(record["digest"])
                if record.get("name"):
                    asset.derived_name = str(record["name"])
                classification = classify.classify(
                    asset.derived_name or asset.path or "", record.get("content_type")
                )
                asset.kind = classification.kind
                asset.category = classification.category
                asset.content_type = classification.content_type
                for name in ("size", "width", "height"):
                    value = record.get(name)
                    target = "size_bytes" if name == "size" else name
                    if value is not None and getattr(asset, target) is None:
                        setattr(asset, target, value)
                if record.get("duration_s") is not None:
                    asset.duration_s = record.get("duration_s")
                origin = str(record.get("origin") or "mentioned")
                if origin not in ORIGINS:
                    origin = "mentioned"
                occurrence = Occurrence(
                    session_id=str(session_id),
                    title=title,
                    origin=origin,
                    child_id=child_id,
                    label=str(label) if label else None,
                    tool=str(record.get("tool") or ""),
                    intent=str(record["intent"]) if record.get("intent") else None,
                    first_ts=float(record.get("first_ts") or 0.0),
                    last_ts=float(record.get("last_ts") or 0.0),
                    first_seq=int(record.get("first_seq") or 0),
                    last_seq=int(record.get("last_seq") or 0),
                    touches=int(record.get("touches") or 1),
                    highlighted=bool(record.get("highlighted")),
                )
                asset.occurrences.append(occurrence)
                if origin not in asset.origins:
                    asset.origins.append(origin)
                asset.highlighted = (
                    asset.highlighted or occurrence.highlighted or origin == "highlight"
                )
    for asset in by_key.values():
        asset.origins.sort(key=lambda value: ORIGINS.index(value))
        if not asset.derived_name:
            asset.derived_name = os.path.basename(asset.path or "") or asset.digest or asset.id
        if asset.kind == "other" and asset.digest:
            classification = classify.classify(asset.derived_name, asset.content_type)
            asset.kind, asset.category = classification.kind, classification.category
    return list(by_key.values())


def build_catalog(
    config_dir: str | Path,
    *,
    budget_s: float | None = DEFAULT_BUDGET_S,
    home: str | None = None,
    refresh: bool = False,
) -> Catalog:
    """Build or refresh the catalog. Blocking: call from a worker thread.

    ``budget_s=None`` scans every uncovered session to completion (the
    background finisher and CLI callers). A positive budget publishes what it
    reached and answers ``building``.

    A warm aggregate is served from memory: within :data:`FRESH_WINDOW_S` for a
    request-path poll, and after a full signature sweep when the on-disk index
    and every visible journal are unchanged. The 11 MB write happens only when
    the session map moved, and non-structural churn is rate-limited so an
    actively-written journal cannot rewrite the index on every poll.

    ``refresh=True`` skips the freshness window (still bounded by ``budget_s``):
    a read that missed an asset uses it to look for an index entry created since
    the last write.
    """
    config = Path(config_dir)
    home = home or str(Path.home())
    path = index_path(config)
    key = _cache_key(config)
    cached = _cached(key)
    if (
        cached is not None
        and not refresh
        and budget_s is not None
        and cached[2].state == "ready"
        and cached[0] == _index_stat(path)
        and time.monotonic() - cached[1] < FRESH_WINDOW_S
    ):
        return _serve(key, cached[2])

    lock = _lock_for(key)
    if not lock.acquire(blocking=budget_s is None):
        # Single-flight: another caller is already scanning. Its last aggregate
        # is servable now; report building so the caller polls again. An
        # unbounded caller (background finisher, CLI) waits instead: it exists
        # to finish the build, not to serve a stale answer.
        if cached is not None:
            return _serve(key, cached[2])
        return catalog_for_read(config)
    with _CACHE_LOCK:
        _BUILDING.add(key)
    try:
        # Another thread may have finished while this one was waiting.
        cached = _cached(key)
        stat = _index_stat(path)
        if (
            cached is not None
            and not refresh
            and budget_s is not None
            and cached[2].state == "ready"
            and cached[0] == stat
            and time.monotonic() - cached[1] < FRESH_WINDOW_S
        ):
            return _serve(key, cached[2])
        return _build_locked(config, path, key, budget_s=budget_s, home=home)
    finally:
        with _CACHE_LOCK:
            _BUILDING.discard(key)
        lock.release()


def catalog_for_read(config_dir: str | Path) -> Catalog:
    """The aggregate for content/thumbnail/get reads.

    Never enumerates sessions and never writes. A warm Catalog is returned only
    when the index file is still the one it was built from; otherwise the
    persisted index is loaded and its assets built once. A thumbnail therefore
    cannot trigger a journal scan or an 11 MB rewrite on the request path.

    The stat is captured BEFORE the read and compared after the asset build.
    A writer that lands in between replaces the file atomically; caching the
    old payload under the new stat would make the fast path serve a catalog
    that predates that write (and made ``refresh=True`` skip the new session).
    One retry handles the race; a second move leaves the catalog uncached.
    """
    config = Path(config_dir)
    path = index_path(config)
    key = _cache_key(config)
    cached = _cached(key)
    if cached is not None and cached[0] == _index_stat(path):
        return _serve(key, cached[2])
    with _asset_lock_for(key):
        cached = _cached(key)
        stat = _index_stat(path)
        if cached is not None and cached[0] == stat:
            return _serve(key, cached[2])
        catalog = Catalog(
            assets=[],
            revision=0,
            state="building",
            sessions_total=0,
            sessions_scanned=0,
            built_at=0.0,
        )
        for _attempt in range(2):
            payload = _load(path)
            raw = payload.get("sessions")
            sessions: dict[str, Any] = raw if isinstance(raw, dict) else {}
            complete = sum(
                1
                for entry in sessions.values()
                if isinstance(entry, dict) and entry.get("complete")
            )
            # A partial index is how a cold/incremental build looks on disk;
            # a reader must not call it ready just because it has rows. An index
            # written before the flag existed, or by a build that stopped early,
            # may have only complete rows and still be missing sessions.
            incomplete = any(
                isinstance(entry, dict) and not entry.get("complete") for entry in sessions.values()
            )
            indexed_all = payload.get("complete") is True
            catalog = Catalog(
                assets=_build_assets(sessions),
                revision=int(payload.get("revision") or 0)
                or int(sum(_entry_revision_value(entry) for entry in sessions.values())),
                state="ready" if sessions and indexed_all and not incomplete else "building",
                sessions_total=len(sessions),
                sessions_scanned=complete,
                built_at=float(payload.get("built_at") or 0.0),
            )
            stat_after = _index_stat(path)
            if stat_after == stat:
                _remember(key, stat_after, catalog)
                return _serve(key, catalog)
            stat = stat_after
        # The file moved twice while we read it: answer from what we have, but
        # do not record an identity that would make the next read trust it.
        return _serve(key, catalog)


def _build_locked(
    config: Path,
    path: Path,
    key: str,
    *,
    budget_s: float | None,
    home: str,
) -> Catalog:
    stat_before = _index_stat(path)
    payload = _load(path)
    raw_sessions = payload.get("sessions")
    cached_sessions: dict[str, Any] = raw_sessions if isinstance(raw_sessions, dict) else {}
    sessions: dict[str, Any] = {}
    visible = _visible_sessions(config)
    total = len(visible)
    scanned = 0
    building = False
    deadline = None if budget_s is None else time.monotonic() + max(0.0, budget_s)
    deny = service.make_deny(config)

    for session_id, _activity, _origin, _archived in visible:
        session_dir = config / "sessions" / session_id
        current = _signature(session_dir)
        cached = cached_sessions.get(session_id)
        # ``/rename`` writes the title SIDECAR and never the transcript, and the
        # sidecar write deliberately preserves the directory's mtime — so the
        # journal signature cannot see a rename, and a cached entry would keep
        # the pre-rename title until some later turn moved the transcript. A
        # cached title that disagrees with the stored one is therefore stale and
        # forces a refresh of this entry. Only a NON-EMPTY stored title does: an
        # untitled session's opening-line title must not force a rescan on every
        # poll forever. ``stored_session_title`` is stat-memoised, so a match
        # costs one ``stat``, not a transcript read.
        stored_title = resume.stored_session_title(session_dir)
        if (
            isinstance(cached, dict)
            and cached.get("complete")
            and _sig_matches(cached.get("sig"), current)
            and _roster_matches(cached.get("roster_sig"), _roster_signature(session_dir))
            and (not stored_title or stored_title == str(cached.get("title") or ""))
        ):
            entry = cached
            entry_complete = True
            children = cached.get("children")
            if isinstance(children, dict):
                changed = False
                for child_id, child in list(children.items()):
                    if not isinstance(child, dict):
                        children.pop(child_id, None)
                        changed = True
                        continue
                    child_dir = config / "sessions" / str(child_id)
                    if _sig_matches(child.get("sig"), _signature(child_dir)):
                        continue
                    remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                    if deadline is not None and remaining is not None and remaining <= 0:
                        building = True
                        entry_complete = False
                        continue
                    refreshed = _child_entry(
                        config,
                        str(child_id),
                        label=child.get("label"),
                        cwd=str(cached.get("cwd") or ""),
                        home=home,
                        deny=deny,
                        budget_s=remaining,
                    )
                    if refreshed is None:
                        children.pop(child_id, None)
                    else:
                        children[child_id] = refreshed
                        entry_complete = entry_complete and bool(refreshed.get("complete"))
                    changed = True
                if changed:
                    entry = dict(cached)
                    entry["children"] = children
                    entry["revision"] = _entry_revision(entry)
                    entry["complete"] = bool(cached.get("complete")) and entry_complete
            sessions[session_id] = entry
            if entry.get("complete"):
                scanned += 1
            else:
                building = True
            continue
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if deadline is not None and remaining is not None and remaining <= 0:
            building = True
            if isinstance(cached, dict):
                # Keep serving the last published rows while the budget is exhausted;
                # an empty row set would blank the grid on every cold poll.
                sessions[session_id] = cached
            continue
        try:
            entry, _revision = _session_entry(
                config,
                session_id,
                session_dir,
                title=stored_title,
                cwd=service.session_cwd(config, session_id, session_dir),
                home=home,
                deny=deny,
                budget_s=remaining,
            )
        except Exception:  # noqa: BLE001 - one bad journal must not blank the catalog
            logger.warning("library: session %s could not be indexed", session_id, exc_info=True)
            if isinstance(cached, dict):
                sessions[session_id] = cached
            continue
        if entry is None:
            continue
        sessions[session_id] = entry
        if entry.get("complete"):
            scanned += 1
        else:
            building = True

    state = "building" if building else "ready"
    complete = not building
    changed = sessions != cached_sessions
    flag_written = payload.get("complete") is complete
    cached = _cached(key)
    if not changed and flag_written and cached is not None and cached[0] == stat_before:
        # Every persisted entry already matched its journal, and the cached
        # aggregate was built from this exact file: no rebuild, no write. Return
        # it directly — this build is finished, so its own in-flight marker must
        # not relabel a ready aggregate as building.
        return cached[2]

    merged = sessions
    built_at = float(payload.get("built_at") or 0.0)
    wrote = False
    disk_stat = stat_before
    if changed or not flag_written:
        persisted = _persist(
            path,
            sessions,
            visible={session_id for session_id, *_ in visible},
            # An unbounded build (background finisher, CLI, tests) must land its
            # result; only the poll path trades persistence for latency.
            force=budget_s is None,
            partial=building,
        )
        if persisted is None:
            # Another process holds the write lock; never wait on the request
            # path. Its write will be picked up by the next stat check.
            if cached is not None:
                return _serve(key, cached[2])
        else:
            merged, disk_built_at, wrote, disk_stat = persisted
            built_at = disk_built_at
            if not wrote and changed and cached is not None:
                # Rate-limited non-structural churn: keep the last aggregate and
                # let a later poll persist; refresh its freshness window so the
                # next poll does not repeat the scan for the same churn — but
                # only if the file it came from is still the one on disk.
                if disk_stat == stat_before:
                    _remember(key, disk_stat, cached[2])
                return _serve(key, cached[2])

    if changed:
        revision = int(sum(_entry_revision_value(entry) for entry in merged.values()))
    else:
        revision = int(payload.get("revision") or 0) or int(
            sum(_entry_revision_value(entry) for entry in sessions.values())
        )
    catalog = Catalog(
        assets=_build_assets(merged),
        revision=revision,
        state=state,
        sessions_total=total,
        sessions_scanned=scanned,
        built_at=built_at or time.time(),
    )
    stat_after = _index_stat(path)
    if stat_after == disk_stat:
        # Only an identity that matches the payload these assets came from may
        # be cached. ``disk_stat`` was captured under the write lock; a writer
        # that landed while ``_build_assets`` ran (outside that lock) leaves the
        # aggregate uncached and the next read reloads.
        _remember(key, stat_after, catalog)
    return catalog


def complete_in_background(config_dir: str | Path) -> Catalog:
    """Finish an uncovered catalog. Idempotent; safe to call from several threads."""
    return build_catalog(config_dir, budget_s=None)


__all__ = [
    "Asset",
    "Catalog",
    "DEFAULT_BUDGET_S",
    "INDEX_SCHEMA",
    "Occurrence",
    "ORIGIN_GROUPS",
    "build_catalog",
    "catalog_for_read",
    "complete_in_background",
    "index_path",
    "origin_group",
]

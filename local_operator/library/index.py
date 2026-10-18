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
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from local_operator import resume
from local_operator.library import classify, identity
from local_operator.session_files import ledger, service
from local_operator.session_files.scan import ORIGINS, FileRecord, resolve_path

logger = logging.getLogger(__name__)

INDEX_DIRNAME: Final = "library"
INDEX_FILENAME: Final = "index.json"
INDEX_SCHEMA: Final = 1
DEFAULT_BUDGET_S: Final = service.DEFAULT_BUDGET_S

#: The three groups the Library shows. Raw origins stay on each occurrence.
ORIGIN_GROUPS: Final = ("generated", "created", "mentioned")
_CREATED_ORIGINS: Final = frozenset({"written", "screenshot", "download", "edited", "command"})
_GENERATED_ORIGINS: Final = frozenset({"generated"})


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
    try:
        # The server-side containment helper is the ONE proof (`_prove_child`).
        # Imported lazily because it lives beside the desktop session pool and
        # this module is also used by the agent tool with no server at all.
        from local_operator.server.utils.desktop_sessions import contained_descendants

        proven, _skipped = contained_descendants(
            config_dir,
            session_id,
            indexed_roster=snapshot.scan.roster,
            indexed=True,
        )
    except Exception:  # noqa: BLE001 - an unprovable roster is an empty child set
        proven = []
    for child in proven:
        child_snapshot = ledger.refresh(
            config_dir,
            child.child_id,
            child.session_dir,
            cwd=service.session_cwd(config_dir, child.child_id, child.session_dir) or cwd,
            home=home,
            deny=deny,
            budget_s=budget_s,
        )
        if child_snapshot is None:
            continue
        child_entries = [
            _record_dict(record) for record in child_snapshot.scan.records.values()
        ]
        children[child.child_id] = {
            "label": child.label,
            "sig": dict(child_snapshot.sig),
            "revision": child_snapshot.revision,
            "complete": bool(child_snapshot.complete),
            "entries": child_entries,
        }
        total_revision += child_snapshot.revision
        complete = complete and bool(child_snapshot.complete)

    return (
        {
            "sig": dict(snapshot.sig),
            "revision": snapshot.revision,
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
                    if value is not None and getattr(asset, "size_bytes" if name == "size" else name) is None:
                        setattr(asset, "size_bytes" if name == "size" else name, value)
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
                asset.highlighted = asset.highlighted or occurrence.highlighted or origin == "highlight"
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
) -> Catalog:
    """Build or refresh the catalog. Blocking: call from a worker thread.

    ``budget_s=None`` scans every uncovered session to completion (the
    background finisher and CLI callers). A positive budget publishes what it
    reached and answers ``building``.
    """
    config = Path(config_dir)
    home = home or str(Path.home())
    path = index_path(config)
    payload = _load(path)
    cached_sessions = payload.get("sessions") if isinstance(payload.get("sessions"), dict) else {}
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
        if (
            isinstance(cached, dict)
            and cached.get("complete")
            and _sig_matches(cached.get("sig"), current)
        ):
            sessions[session_id] = cached
            scanned += 1
            continue
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if deadline is not None and remaining <= 0:
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
                title=resume.session_name(session_dir),
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

    if building:
        state = "building"
    else:
        state = "ready"

    revision = int(sum(float(entry.get("revision") or 0.0) for entry in sessions.values()))
    if sessions != cached_sessions or state == "ready":
        _write(
            path,
            {
                "version": INDEX_SCHEMA,
                "built_at": time.time(),
                "revision": revision,
                "sessions": sessions,
            },
        )
    return Catalog(
        assets=_build_assets(sessions),
        revision=revision,
        state=state,
        sessions_total=total,
        sessions_scanned=scanned,
        built_at=time.time(),
    )


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
    "complete_in_background",
    "index_path",
    "origin_group",
]

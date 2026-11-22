"""Opt-in, cold-only cleanup evidence for the desktop catalogue.

The feed's headline revision orders headlines. It cannot attest to absent work:
startup falls back to revision zero, and an unseen completion can mask a live
owner. This separate receipt is not a latch, a lock, or an extension of the
session's lifetime. Compare two observations, in the same uninterrupted feed
connection, before dropping a tab. Never persist or merge it into a row cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from sqlite3 import Error as SQLiteError
from typing import Any, NamedTuple

from local_operator.asks import store as asks
from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
from local_operator.session.archived import ARCHIVED_FILE
from local_operator.session.attention import AttentionStore
from local_operator.session.catalog import session_directory_name
from local_operator.session.placement import MESH_STAMP_NAME
from local_operator.session.retention import _ACTIVITY_FILES, TRANSCRIPT_FILENAME
from local_operator.session.row_clip import ROW_RANGE_MAX_BYTES
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import SessionRecord
from local_operator.session.transcript import ENTRY_CUSTOM
from local_operator.session_lease import LEASE_NAME, MIRROR_NAME, RECOVERY_LOCK_NAME
from local_operator.tui.sidebar_pins import PINS_FILE
from local_operator.wakes import store as wakes

# Per response, not per row. Each optional scan fits the existing maximum
# Desktop history byte range; this path cannot make the normal list fail.
_ASK_READ_BUDGET = ROW_RANGE_MAX_BYTES
_WAKE_READ_BUDGET = ROW_RANGE_MAX_BYTES
_REGISTRY_READ_BUDGET = ROW_RANGE_MAX_BYTES
# Count *all* directory entries, including junk and writer temporaries. Use
# the same ceiling as the existing Desktop catalogue page (500); this bounds
# directories with tiny/empty files even when the byte budget is not spent.
_REGISTRY_ENTRY_BUDGET = 500
_FileStamp = tuple[int, int, int, int, int]


class _ColdFence(NamedTuple):
    directory: _FileStamp
    activity: dict[str, _FileStamp | None]
    journal: _FileStamp | None


class _VetoFence(NamedTuple):
    pins: _FileStamp | None
    archives: _FileStamp | None
    wakes: _FileStamp | None
    registry: _FileStamp | None


_UNKNOWN = (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError)
_ASK_EVENTS = frozenset(
    {
        asks.EVENT_QUEUED,
        asks.EVENT_ANSWERED,
        asks.EVENT_DECLINED,
        asks.EVENT_DISMISSED,
        asks.EVENT_REVISED,
        asks.EVENT_WITHDRAWN,
        asks.EVENT_RELAY_SURFACED,
        asks.EVENT_CONTEXT,
        asks.EVENT_CONSENT_USED,
    }
)
_SETTLED_ASKS = frozenset(
    {asks.STATUS_ANSWERED, asks.STATUS_DECLINED, asks.STATUS_DISMISSED, asks.STATUS_WITHDRAWN}
)


def _stat(path: Path, *, directory: bool = False) -> os.stat_result | None:
    """Only actual absence is a negative. Refuse unreadable or indirect files."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError(f"unexpected file type in workspace rest proof: {path.name}")
    return info


def _fingerprint(info: os.stat_result | None) -> _FileStamp | None:
    if info is None:
        return None
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)


def _cold_fence(root: Path, directory: Path) -> _ColdFence | None:
    before = _stat(directory, directory=True)
    if before is None:
        return None
    # Modern runtimes claim the lease before starting and keep it through
    # disposal. Legacy owners have the mirror. A crash/recovery marker, a
    # mixed-build owner, and any mesh placement (even if currently local) are
    # deliberately not proved by this cold-only reader.
    if (
        any(
            _stat(directory / marker) is not None
            for marker in (LEASE_NAME, RECOVERY_LOCK_NAME, MIRROR_NAME, MESH_STAMP_NAME)
        )
        or _stat(wakes.entry_path(root, directory.name)) is not None
    ):
        # Presence is enough to veto, even if the wake projection is corrupt
        # or parked. Absence is NOT enough: its writer is best-effort.
        return None
    journal = _stat(asks.asks_log_path(directory))
    # An index is derived, best-effort data. It may establish a reason NOT to
    # proceed, but an empty index cannot prove no asks. An index without its
    # journal may be a legacy or inconsistent state; do not settle either way.
    if journal is None and _stat(asks.entry_path(root, directory.name)) is not None:
        return None
    activity = {name: _fingerprint(_stat(directory / name)) for name in sorted(_ACTIVITY_FILES)}
    fingerprint = _fingerprint(before)
    if fingerprint is None or fingerprint != _fingerprint(_stat(directory, directory=True)):
        return None
    # Directory ctime/mtime carries even a short-lived owner that claimed and
    # released between two observations without changing the feed headline or
    # appending to the transcript. File ctime sees edits that preserve mtime.
    return _ColdFence(fingerprint, activity, _fingerprint(journal))


def _read_witness(path: Path, expected: _FileStamp) -> bytes:
    # Callers reserve size + 1 before opening. Check the opened inode (not just
    # the path) and don't prefetch; a FIFO substituted after stat must not block.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags | getattr(os, "O_BINARY", 0)), "rb", buffering=0) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or _fingerprint(info) != expected:
            raise ValueError("workspace rest witness changed before read")
        raw = handle.read(expected[-1] + 1)
        if len(raw) != expected[-1] or _fingerprint(os.fstat(handle.fileno())) != expected:
            raise ValueError("workspace rest witness changed during read")
    return raw


def _no_open_asks(journal: Path, expected: _FileStamp) -> bool:
    # The usual tolerant JSONL reader is right for rendering, but a torn or
    # unsupported row cannot certify absence. Do not follow relay references
    # into a second session; those are unknown on this proof's bounded path.
    raw = _read_witness(journal, expected)
    if raw and not raw.endswith(b"\n"):
        return False
    events = [json.loads(line) for line in raw.splitlines()]
    if any(
        not isinstance(event, dict)
        or event.get("v") != asks.EVENT_SCHEMA
        or not isinstance(event.get("ask_id"), str)
        or not event["ask_id"]
        or event.get("kind") not in _ASK_EVENTS
        or type(event.get("at")) is not int
        or event["at"] <= 0
        for event in events
    ):
        return False
    # The display reducer can recover from out-of-order data. An absence proof
    # cannot use an answer (or any other update) from before the ask existed.
    ids: set[str] = set()
    for event in events:
        ask_id = event["ask_id"]
        if event["kind"] == asks.EVENT_QUEUED:
            if (
                ask_id in ids
                or type(event.get("expires_at")) is not int
                or event["expires_at"] <= 0
            ):
                return False
            ids.add(ask_id)
        elif ask_id not in ids:
            return False
    rows = asks.fold(events, asks.now_ms())
    return {row.get("ask_id") for row in rows} == ids and all(
        row.get("status") in _SETTLED_ASKS for row in rows
    )


def capture_veto_fence(root: Path) -> _VetoFence | None:
    """Metadata only: bracket the permissive catalogue, including normal reaping."""
    try:
        return _VetoFence(
            _fingerprint(_stat(root / PINS_FILE)),
            _fingerprint(_stat(root / ARCHIVED_FILE)),
            _fingerprint(_stat(wakes.wakes_dir(root), directory=True)),
            _fingerprint(_stat(registry.run_dir(root), directory=True)),
        )
    except (OSError, ValueError):
        return None


def _veto_ids(path: Path, expected: _FileStamp | None) -> set[str]:
    if expected is None:
        return set()
    if expected[-1] + 1 > ROW_RANGE_MAX_BYTES:
        raise ValueError("workspace rest veto index exceeds read budget")
    ids = json.loads(_read_witness(path, expected))
    if not isinstance(ids, list) or not all(session_directory_name(sid) for sid in ids):
        raise ValueError("workspace rest veto index is not a session id array")
    # Membership in the writer's raw array, not the display reader's pruned
    # and best-effort projection. Missing files are attested by both fences.
    return set(ids)


def _no_wake_schedule(path: Path, expected: _FileStamp) -> bool:
    # The latest wake snapshot in the transcript is authoritative even if the
    # index failed to write. No decompression, repair or unbounded replay on a
    # list request: only a complete affordable plain journal is a witness.
    raw = _read_witness(path, expected)
    if raw and not raw.endswith(b"\n"):
        return False
    for line in reversed(raw.split(b"\n")):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            return False
        if row.get("type") != ENTRY_CUSTOM:
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            return False
        if payload.get("custom_type") == WAKE_SCHEDULES_CUSTOM_TYPE:
            details = payload.get("details")
            return isinstance(details, dict) and details.get("schedules") == []
    return True


def _registry_owners(directory: Path, expected: _FileStamp | None) -> set[str]:
    if expected is None:
        return set()
    owners: set[str] = set()
    budget = _REGISTRY_READ_BUDGET
    # Discovery is keyed by PID, so a broken record may be any candidate's
    # legacy owner. Check the complete inventory or certify none. Do not use
    # iterdir/glob/sort here: even enumerating all paths defeats the entry cap.
    with os.scandir(directory) as entries:
        for index, entry in enumerate(entries):
            if index >= _REGISTRY_ENTRY_BUDGET:
                raise ValueError("workspace rest registry exceeds entry budget")
            path = Path(entry.path)
            if path.suffix != ".json":
                continue
            stamp = _fingerprint(_stat(path))
            if stamp is None or stamp[-1] + 1 > budget:
                raise ValueError("workspace rest registry exceeds read budget or changed")
            budget -= stamp[-1] + 1
            raw = json.loads(_read_witness(path, stamp))
            if not isinstance(raw, dict):
                raise ValueError("workspace rest registry record is not an object")
            record = SessionRecord.from_json(raw)
            if (
                type(record.pid) is not int
                or record.pid <= 0
                or path.stem != str(record.pid)
                or not session_directory_name(record.session_id)
            ):
                raise ValueError("workspace rest registry record has no usable identity")
            # Any surviving record for this id is enough to veto. Reaping and
            # PID classification are the catalogue's job, not a proof-side sweep.
            owners.add(record.session_id)
    return owners


def decorate_workspace_rest(
    root: Path, rows: Sequence[dict[str, Any]], epoch: str, fence: _VetoFence
) -> None:
    """Add v1 proofs to already-authorized page rows; unknown stays absent."""
    candidates = []
    for row in rows:
        attention = row.get("attention") or {}
        if (
            row.get("locality") == "local"
            and row.get("active") is False
            and row.get("background_work") is False
            and row.get("waiting_on") == 0
            and row.get("pending") is None
            and not row.get("degraded")
            and not row.get("archived")
            and not row.get("pinned")
            and attention.get("kind") == "complete"
            and attention.get("unseen") is False
            and attention.get("completion_token")
            and row.get("status", {}).get("code") == "complete"
        ):
            candidates.append(row)
    if not candidates or capture_veto_fence(root) != fence:
        return
    try:
        vetoed = _veto_ids(root / PINS_FILE, fence.pins) | _veto_ids(
            root / ARCHIVED_FILE, fence.archives
        )
    except _UNKNOWN:
        return

    ask_budget = _ASK_READ_BUDGET
    wake_budget = _WAKE_READ_BUDGET
    observed: list[tuple[dict[str, Any], Path, _ColdFence]] = []
    for row in candidates:
        session_id = row["id"]
        if session_id in vetoed:
            continue
        directory = root / "sessions" / session_id
        try:
            before = _cold_fence(root, directory)
            if before is None:
                continue
            journal = before.journal
            if journal is not None:
                cost = journal[-1] + 1
                if cost > ask_budget:
                    continue
                ask_budget -= cost
                if not _no_open_asks(asks.asks_log_path(directory), journal):
                    continue
            transcript = before.activity[TRANSCRIPT_FILENAME]
            if transcript is None or transcript[-1] + 1 > wake_budget:
                continue
            wake_budget -= transcript[-1] + 1
            if not _no_wake_schedule(directory / TRANSCRIPT_FILENAME, transcript):
                continue
        except _UNKNOWN:
            continue  # this candidate's evidence is unknown, not empty
        observed.append((row, directory, before))
    if not observed:
        return

    # No proof-only inventory work for a miss, off-page id, or ineligible row.
    # The pre-catalogue directory stamp ensures we do not trust a census merely
    # because the ordinary scan just hid/reaped an ambiguous record.
    try:
        owners = _registry_owners(registry.run_dir(root), fence.registry)
    except _UNKNOWN:
        return

    # Recheck the completion levels in one bulk read, BETWEEN the file fences.
    # A completion/ack written independently of the session directory must not
    # give a stale catalogue row today's proof (or inherit its old clock).
    try:
        attention = AttentionStore(root / "attention.db").state_many(
            f"session/{row['id']}" for row, _, _ in observed
        )
    except (OSError, SQLiteError):
        return
    proofs = []
    for row, directory, before in observed:
        try:
            if before != _cold_fence(root, directory):
                continue
        except (OSError, ValueError):
            continue
        session_id = row["id"]
        current = attention.get(f"session/{session_id}", {})
        completion = row["attention"]["completion_token"]
        if (
            session_id in owners
            or current.get("kind") != "complete"
            or current.get("unseen") is not False
            or current.get("completion_token") != completion
            or current.get("revision") != row["attention"].get("revision")
        ):
            continue
        # Pin/archives are whole-store authorities: their generations carry a
        # pin→unpin or archive→unarchive even between requests. For wakes and
        # registry the directory generations only fence this particular read.
        # The local transcript and lease/mirror carry this session's changes;
        # unrelated wake projections and heartbeats must not reset its clock.
        material = (epoch, session_id, completion, before, fence.pins, fence.archives)
        revision = hashlib.sha256(repr(material).encode()).hexdigest()
        proofs.append((row, {"version": 1, "epoch": epoch, "revision": revision}))

    # Atomic writers change the sampled generation even for an ABA edit. The
    # proof describes an observation bracketed by these reads, not a lock.
    if capture_veto_fence(root) != fence:
        return
    for row, proof in proofs:
        row["workspace_rest"] = proof

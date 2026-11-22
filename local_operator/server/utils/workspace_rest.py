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
from typing import Any

from local_operator.asks import store as asks
from local_operator.session.attention import AttentionStore
from local_operator.session.placement import MESH_STAMP_NAME
from local_operator.session.retention import _ACTIVITY_FILES
from local_operator.session.row_clip import ROW_RANGE_MAX_BYTES
from local_operator.session.runtime import registry
from local_operator.session_lease import LEASE_NAME, MIRROR_NAME, RECOVERY_LOCK_NAME

# Per response, not per row. This opt-in read shares the existing desktop
# history-range byte budget; a journal that cannot fit stays unknown. Ordinary
# catalogue requests and the feed ticker never read these journals.
_ASK_READ_BUDGET = ROW_RANGE_MAX_BYTES
_FileStamp = tuple[int, int, int, int, int]
_ColdFence = tuple[_FileStamp | None, tuple[_FileStamp | None, ...], _FileStamp | None]
_RegistryFence = tuple[set[str], _FileStamp | None]
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
    if any(
        _stat(directory / marker) is not None
        for marker in (LEASE_NAME, RECOVERY_LOCK_NAME, MIRROR_NAME, MESH_STAMP_NAME)
    ):
        return None
    journal = _stat(asks.asks_log_path(directory))
    # An index is derived, best-effort data. It may establish a reason NOT to
    # proceed, but an empty index cannot prove no asks. An index without its
    # journal may be a legacy or inconsistent state; do not settle either way.
    if journal is None and _stat(asks.entry_path(root, directory.name)) is not None:
        return None
    activity = tuple(_fingerprint(_stat(directory / name)) for name in sorted(_ACTIVITY_FILES))
    after = _stat(directory, directory=True)
    fingerprint = _fingerprint(before)
    if fingerprint != _fingerprint(after):
        return None
    # Directory ctime/mtime carries even a short-lived owner that claimed and
    # released between two observations without changing the feed headline or
    # appending to the transcript. File ctime sees edits that preserve mtime.
    return (fingerprint, activity, _fingerprint(journal))


def _no_open_asks(journal: Path, expected: _FileStamp) -> bool:
    # The usual tolerant JSONL reader is right for rendering, but a torn or
    # unsupported row cannot certify absence. Do not follow relay references
    # into a second session; those are unknown on this proof's bounded path.
    # Check the opened inode before reading (a replacement can land after the
    # path stat). A FIFO substitution must not block an optional catalogue read.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(journal, flags | getattr(os, "O_BINARY", 0)), "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or _fingerprint(info) != expected:
            return False
        size = expected[-1]
        raw = handle.read(size + 1)
    if len(raw) != size or (raw and not raw.endswith(b"\n")):
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
    births = [event for event in events if event["kind"] == asks.EVENT_QUEUED]
    if any(
        type(event.get("expires_at")) is not int or event["expires_at"] <= 0 for event in births
    ):
        return False
    ids = {event["ask_id"] for event in events}
    if len(births) != len(ids) or {event["ask_id"] for event in births} != ids:
        return False
    rows = asks.fold(events, asks.now_ms())
    return {row.get("ask_id") for row in rows} == ids and all(
        row.get("status") in _SETTLED_ASKS for row in rows
    )


def capture_workspace_registry(root: Path) -> _RegistryFence | None:
    """One read-only census, before the catalogue's best-effort reaping scan."""
    directory = registry.run_dir(root)
    try:
        first = _fingerprint(_stat(directory, directory=True))
        # FileNotFoundError after a positive directory stat is a race, not an
        # empty inventory. Strict scan also rejects malformed records that the
        # subsequent display scan might skip or reap.
        records = registry.scan(root, reap=False, strict=True) if first is not None else []
        if first != _fingerprint(_stat(directory, directory=True)):
            return None
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ):
        return None
    return ({record.session_id for record, state in records if state != "stale"}, first)


def decorate_workspace_rest(
    root: Path, rows: Sequence[dict[str, Any]], epoch: str, census: _RegistryFence
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
            and attention.get("unseen") is False
            and attention.get("completion_token")
            and row.get("status", {}).get("code") == "complete"
        ):
            candidates.append(row)
    if not candidates:
        return

    owners, registry_fingerprint = census
    budget = _ASK_READ_BUDGET
    observed: list[tuple[dict[str, Any], Path, _ColdFence]] = []
    for row in candidates:
        session_id = row["id"]
        if session_id in owners:
            continue
        directory = root / "sessions" / session_id
        try:
            before = _cold_fence(root, directory)
            if before is None:
                continue
            journal = before[2]
            if journal is not None:
                size = journal[-1]
                if size + 1 > budget:
                    continue
                budget -= size + 1
                if not _no_open_asks(asks.asks_log_path(directory), journal):
                    continue
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
            continue  # this candidate's evidence is unknown, not empty
        observed.append((row, directory, before))
    if not observed:
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
        if current.get("unseen") is not False or current.get("completion_token") != completion:
            continue
        # No raw path, time, completion token or filesystem identity goes to
        # the renderer. Include the epoch and id so receipts cannot be spliced
        # from an old daemon or another row with an identical headline.
        material = (epoch, session_id, completion, before)
        revision = hashlib.sha256(repr(material).encode()).hexdigest()
        proofs.append((row, {"version": 1, "epoch": epoch, "revision": revision}))

    try:
        if registry_fingerprint != _fingerprint(_stat(registry.run_dir(root), directory=True)):
            return
    except (OSError, ValueError):
        return
    for row, proof in proofs:
        row["workspace_rest"] = proof

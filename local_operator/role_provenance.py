"""Where a promoted role or team came from (LOH-9 S4).

WHY A DEVICE-LEVEL INDEX RATHER THAN A SIDECAR IN THE ROW
========================================================

Design §3.6 sketched ``<config>/agents/<id>/provenance.json`` (and the team
twin). The merged stores make that unshippable: ``TeamRegistry._save_team_locked``
writes exactly ``team.yml``, ``instructions.md``, ``project.md`` and publishes
by renaming a whole staged directory over the row — any OTHER file in a team
row is destroyed on the next save. Provenance that vanishes the first time the
user edits the team is worse than no provenance, because a receipt pointing at
it would be a promise the store does not keep.

One append-only index at ``<config>/role-provenance.jsonl`` is durable for both
kinds, needs no change to either store, and is the thing a receipt or a future
audit reads. It is deliberately NOT under ``sessions/`` (a session's retention
sweep owns that tree) and NOT inside either global registry (each store owns its
own files and would rewrite or ignore it).

BOUNDED BY CONSTRUCTION
=======================

An append-only file that nothing rotates grows without limit, so every append
first checks whether the batch would cross :data:`MAX_PROVENANCE_BYTES` and, if
it would, rewrites the file with the most recent rows before appending. Rotating
BEFORE the append is what makes an unwritable index fail with nothing recorded
rather than with rows for a promotion that then rolled back. The rewrite is
atomic (temp file + ``os.replace``) under the same lock, so a reader never sees a
truncated index and a crash leaves either the old file or the new one.

The lock is a separate ``.lock`` file rather than the data file itself: a
rotation REPLACES the data file's inode, and two processes flocking different
inodes of the same path is exactly the interleaving this is meant to stop. The
wait is bounded — an append happens inside a turn, so a holder that blocks is a
refusal the caller can retry, not a hang.
"""

from __future__ import annotations

import errno
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

#: The index's file name inside the config root.
PROVENANCE_FILENAME = "role-provenance.jsonl"
#: The sidecar the lock lives on (see the module docstring for why not the data file).
PROVENANCE_LOCKNAME = "role-provenance.lock"

#: How many rows the index keeps. Every promotion writes one row per definition
#: it creates or overwrites, so this is hundreds of promotions of history — far
#: more than any receipt is read for, and small enough to read into memory.
MAX_PROVENANCE_ROWS = 500
#: The size that triggers a rotation to :data:`MAX_PROVENANCE_ROWS`. A row is
#: ~150 bytes, so the steady state is well under this and a burst of long names
#: cannot push the file past it for more than one append.
MAX_PROVENANCE_BYTES = 256 * 1024

#: Bounded lock wait: 100 × 20 ms = 2 s, then a named refusal.
_LOCK_ATTEMPTS = 100
_LOCK_SLEEP_S = 0.02

#: The ONLY fields a row carries. A fixed set rather than a passthrough: this
#: file outlives the code that wrote it, so a field added casually becomes a
#: schema every later reader has to tolerate.
RECORD_FIELDS = ("kind", "id", "name", "source_session", "at", "authorized_by", "replaced")


def provenance_path(config_dir: str | Path) -> Path:
    """``<config>/role-provenance.jsonl`` — the one index."""
    return Path(config_dir) / PROVENANCE_FILENAME


def _record(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise ONE row, refusing unknown or missing fields."""
    unknown = sorted(set(fields) - set(RECORD_FIELDS))
    if unknown:
        raise ValueError(f"unknown provenance field(s): {', '.join(unknown)}")
    record: dict[str, Any] = {}
    for name in RECORD_FIELDS:
        value = fields.get(name)
        if name == "replaced":
            record[name] = str(value) if value else None
            continue
        if value is None or str(value) == "":
            raise ValueError(f"provenance field {name!r} is required")
        record[name] = str(value)
    return record


def _try_lock(fd: int) -> bool:
    """One non-blocking exclusive lock attempt (POSIX, with a Windows arm)."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):
                return False
            raise

    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise


def _unlock(fd: int) -> None:
    """Release a lock taken by :func:`_try_lock`."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


class _Locked:
    """The index's lock: a bounded exclusive hold on the sidecar file."""

    def __init__(self, config_dir: Path) -> None:
        self.path = config_dir / PROVENANCE_LOCKNAME
        self._fd: int | None = None

    def __enter__(self) -> "_Locked":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        for _ in range(_LOCK_ATTEMPTS):
            if _try_lock(fd):
                self._fd = fd
                return self
            time.sleep(_LOCK_SLEEP_S)
        os.close(fd)
        raise TimeoutError(
            "timed out waiting for the role-provenance index, so nothing was "
            "recorded and nothing was promoted; retry in a moment"
        )

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None:
            _unlock(self._fd)
            os.close(self._fd)
            self._fd = None


def _atomic_write(path: Path, text: str) -> None:
    """Publish one complete file without exposing a truncated target."""
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def append_promotions(config_dir: str | Path, records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Append ``records`` as one locked, fsynced batch; return their ids.

    ALL of the batch's rows land in ONE write call under the lock, so a failure
    either happens before anything is appended or leaves at most a trailing
    partial line that :func:`read_promotions` skips. A promotion that rolls back
    therefore never reports ids for rows it did not durably record.
    """
    prepared = [_record(record) for record in records]
    if not prepared:
        return []
    # ``_record`` already refused an empty id. It deliberately does NOT require a
    # UUID: an agent/team id is any safe path segment (``validate_agent_id``), so
    # a row mirrored from an older peer or imported from an archive can legitimately
    # carry another shape, and refusing it here would fail a promotion AFTER its
    # rows were written.
    root = Path(config_dir)
    target = provenance_path(root)
    with _Locked(root):
        lines = "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in prepared
        )
        # ROTATE BEFORE APPENDING, not after. Appending first and rotating
        # second made a rotation failure leave the batch's rows in place while
        # the caller rolled the promotion back: the index then named ids that no
        # longer existed, and the user was told nothing had been promoted (review
        # round 1, MINOR-1). Rotating first means a failure here has written
        # NEITHER the definitions nor their rows, which is the order the
        # all-or-nothing rule actually needs.
        #
        # The keep-count leaves room for this batch, so one append cannot push
        # the file past MAX_PROVENANCE_ROWS even when the batch is the bound
        # itself (``MAX_PROMOTION_ROWS``).
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if len(existing.encode("utf-8")) + len(lines.encode("utf-8")) > MAX_PROVENANCE_BYTES:
            keep = max(MAX_PROVENANCE_ROWS - len(prepared), 0)
            kept = read_promotions(root)[-keep:] if keep else []
            existing = "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in kept
            )
        # ONE atomic publication of old+new. A raw append was durable only up to
        # its own write: an EIO after the bytes reached the file, or a failure in
        # the fsync below it, left a row for a promotion whose caller then rolled
        # the definitions back — the index naming an id that no longer exists
        # (review round 2). ``_atomic_write`` publishes via a temp file and one
        # ``os.replace``, so a failure leaves the PREVIOUS index intact and the
        # caller's rollback is consistent with it.
        _atomic_write(target, existing + lines)
    return [record["id"] for record in prepared]


def read_promotions(config_dir: str | Path) -> list[dict[str, Any]]:
    """Every readable row, oldest first. A torn trailing line is skipped."""
    target = provenance_path(config_dir)
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("id"):
            rows.append(row)
    return rows


def require_writable(config_dir: str | Path) -> None:
    """Refuse BEFORE a promotion's first write if the index cannot be appended.

    A promotion that creates rows and only THEN discovers it cannot record them
    would have to roll back work that was otherwise fine, so this probe runs in
    preflight. It opens the real file in append mode (creating it) and closes it
    without writing, which is the smallest check that proves the directory, the
    permissions and the file's own mode all allow the append that follows.
    """
    root = Path(config_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(provenance_path(root), "a", encoding="utf-8"):
        pass

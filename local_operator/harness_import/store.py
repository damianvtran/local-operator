"""Read-only source helpers: streaming JSONL, SQLite snapshots, IDs, lock.

**The read-only contract is enforced by construction, not by discipline.**
JSONL is opened ``"r"``. SQLite is never opened in place: a private snapshot
copies the database plus any ``-wal``/``-shm`` into our config root, and the
copy is opened ``immutable=1``. Opening the original in ``mode=ro`` would make
SQLite create a ``-shm`` beside a WAL database — a write to a foreign root,
which the design forbids.

Everything else here is deliberately small: bounded streaming, stable ids, a
process-wide import lock and atomic writes for TARGET files only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: A single JSONL line larger than this is skipped rather than materialised.
MAX_LINE_BYTES = 16 * 1024 * 1024
#: Hard ceiling for one source body read; a larger file is refused, not streamed
#: forever (the design's resource bound).
MAX_SOURCE_BYTES = 512 * 1024 * 1024
#: Prefix hashed for the manifest fingerprint. Hashing all 33 MB of a job
#: transcript on every listing is wasted I/O; a 1 MiB prefix plus the stat tuple
#: distinguishes an in-place append from a rotation in practice.
FINGERPRINT_PREFIX = 1024 * 1024
#: Import lock older than this is stale even if its pid cannot be checked.
LOCK_STALE_SECONDS = 30 * 60


def imports_root(config: Path) -> Path:
    root = Path(config) / "imports"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def is_confined(root: Path, candidate: Path) -> bool:
    """True when ``candidate`` resolves inside ``root`` after symlinks.

    The source roots are the user's own harness directories, but a planted
    symlink must not turn "list my conversations" into "read /etc". A link whose
    real path escapes the detected root is skipped by the adapters.
    """
    try:
        root_real = Path(root).resolve()
        candidate_real = Path(candidate).resolve()
    except OSError:
        return False
    return candidate_real == root_real or root_real in candidate_real.parents


def fingerprint(path: Path) -> dict[str, Any]:
    """Cheap identity for a source body: stat plus a hashed prefix."""
    try:
        stat = Path(path).stat()
    except OSError:
        return {"size": 0, "mtime_ns": 0, "sha256_prefix": ""}
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            remaining = FINGERPRINT_PREFIX
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError:
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256_prefix": ""}
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256_prefix": digest.hexdigest(),
    }


def sha256_prefix(path: Path, length: int) -> str:
    """Hash exactly ``length`` leading bytes (bounded by a short read)."""
    digest = hashlib.sha256()
    remaining = max(0, int(length))
    with Path(path).open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(65536, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(
    path: Path, *, stats: dict[str, int] | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream ``(line_number, object)`` from a JSONL file, bounded and tolerant.

    Malformed and oversized lines are skipped INDIVIDUALLY: a truncated tail
    (a live writer) must not discard the conversation before it. The one
    exception is a file over the total byte ceiling, which raises; importing
    half of a 2 GB body silently would be worse than telling the user.

    ``stats`` is the observability half: when supplied it receives
    ``skipped_lines``/``oversized_lines``/``malformed_lines`` so a caller can
    report that the source was partial instead of importing it as complete.
    Passing it is optional so the read-only listing paths stay allocation-free.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size > MAX_SOURCE_BYTES:
        raise ValueError(f"source exceeds the {MAX_SOURCE_BYTES} byte import cap")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            if len(line) > MAX_LINE_BYTES:
                if stats is not None:
                    stats["skipped_lines"] = stats.get("skipped_lines", 0) + 1
                    stats["oversized_lines"] = stats.get("oversized_lines", 0) + 1
                continue
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                if stats is not None:
                    stats["skipped_lines"] = stats.get("skipped_lines", 0) + 1
                    stats["malformed_lines"] = stats.get("malformed_lines", 0) + 1
                continue
            if isinstance(value, dict):
                yield number, value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [value for _, value in iter_jsonl(path)]


def stable_session_id(source: str, source_id: str, *, suffix: str = "") -> str:
    """Deterministic target id; ``suffix`` marks a copy of a changed source."""
    payload = f"lop-harness-import/v1\0{source}\0{source_id}{suffix}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def stable_entry_id(source: str, source_id: str, source_event_id: str, ordinal: int) -> str:
    payload = (
        f"lop-harness-import/v1/entry\0{source}\0{source_id}\0" f"{source_event_id}\0{ordinal}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Publish complete bytes with ``os.replace`` and an fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, prefix=f".{path.name}."
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


@dataclass
class SourceSnapshot:
    """One copied SQLite database and the columns read from it."""

    path: Path
    original: Path


@contextlib.contextmanager
def sqlite_snapshot(config: Path, database: Path) -> Iterator[SourceSnapshot]:
    """Copy ``database`` (+ WAL/SHM) under the config root, then open the COPY.

    The copy is deleted on exit. A missing database raises ``FileNotFoundError``
    so the adapter can report "metadata unavailable" rather than an empty list.
    ``connect_snapshot`` deliberately uses a normal read-only connection so the
    copied WAL is replayed (see its docstring for the measured failure).
    """
    database = Path(database)
    if not database.exists():
        raise FileNotFoundError(database)
    stat = database.stat()
    key = hashlib.sha256(f"{database}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()[
        :16
    ]
    directory = imports_root(config) / "snapshots" / key
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = directory / database.name
    try:
        shutil.copy2(database, target)
        for suffix in ("-wal", "-shm"):
            side = Path(str(database) + suffix)
            if side.exists():
                shutil.copy2(side, Path(str(target) + suffix))
        yield SourceSnapshot(path=target, original=database)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def connect_snapshot(snapshot: SourceSnapshot) -> sqlite3.Connection:
    """Open the private COPY read-only, allowing SQLite to replay its WAL.

    Deliberately NOT ``immutable=1``: that flag declares the database
    unchanging, so SQLite skips the ``-wal`` we just copied and a live Codex
    store (writer holding the WAL open) lists zero threads or fails with
    "no such table". The copy is already private to us; a normal read-only
    connection cannot reach the foreign original and still gets WAL replay.
    (Documented deviation from design §7, whose ``immutable=1`` prescription
    is what caused the invisible-thread failure.)
    """
    return sqlite3.connect(f"file:{snapshot.path}?mode=ro", uri=True, timeout=5)


class ImportLock:
    """One import job per process root; stale after 30 minutes or dead pid."""

    def __init__(self, config: Path):
        # Do not create <config>/imports here: constructing a service for a
        # read-only sources/preview call must not write. acquire() creates it.
        self.path = Path(config) / "imports" / ".lock"

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps({"pid": os.getpid(), "at": time.time()})
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if not self._stale():
                return False
            with contextlib.suppress(OSError):
                self.path.unlink()
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                return False
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        return True

    def _stale(self) -> bool:
        try:
            stat = self.path.stat()
        except OSError:
            return True
        if time.time() - stat.st_mtime > LOCK_STALE_SECONDS:
            return True
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except (OSError, ValueError, TypeError):
            return True
        if pid <= 0:
            return True
        if os.name != "posix":
            # On Windows ``os.kill(pid, 0)`` is TerminateProcess, not a
            # liveness probe: it would KILL the process it is checking. The
            # mtime rule above is the only safe cross-platform signal there.
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def release(self) -> None:
        with contextlib.suppress(OSError):
            self.path.unlink()

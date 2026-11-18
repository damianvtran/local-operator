"""Cooperating install writers share one permanent INSTALLROOT lock inode.

The lock serialises mutations, not preparation, and does not drain the fleet or
exclude old/non-cooperating installers. Callers still owe admission and fresh
promotion checks. Observers never open a lock for writing or try to acquire it.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from local_operator import procstate

LOCK_NAME = "install.lock"
ATTEMPT_NAME = ".install-attempt.json"
DEFAULT_TIMEOUT_S = 10.0


class InstallTransactionError(Exception):
    """Actionable refusal shared by CLI, TUI and mesh install writers."""


class RecoveryCapacityError(InstallTransactionError):
    """Bounded active namespace is full, not a transient installer failure.

    The machine code is additive to the frozen update failure vocabulary. The
    count is a LOWER bound observed during admission, never a census or a claim
    that any legacy operation can be retired.
    """

    code = "update_recovery_capacity"
    active_limit = 256
    observed_at_least = 256


class LockUnsupported(InstallTransactionError):
    """No validated lock implementation on this platform; fail before mutation."""


class LockBusy(InstallTransactionError):
    """A cooperating writer holds the root, or this process could not take it.

    A CLASS, not a sentence, because the update report classifies a failed
    install's cause structurally: the caller knows a bounded wait happened and
    that nothing was mutated, which is a different fact from an installer that
    exited non-zero. Matching this by message text would make the report depend
    on wording a future edit is free to improve.
    """


# flock locks file descriptions, not Python threads; reentrancy belongs to the
# owning thread, so nested primitive calls cannot unlock a parent's transaction.
_MUTEXES: dict[Path, threading.RLock] = {}
_MUTEX_GUARD = threading.Lock()
_LOCAL = threading.local()


def canonical_root(root: Path) -> Path:
    return root.expanduser().resolve()


def _identity() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "birth_scheme": procstate.birth_scheme(),
        "birth_token": procstate.self_birth_token(),
    }


def owner_alive(record: dict[str, Any]) -> bool | None:
    """Unknown identity is not reclaimable; PID reuse is not ownership."""
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    if procstate.pid_liveness(pid) is False:
        return False
    return procstate.same_birth(record.get("birth_scheme"), record.get("birth_token"), pid)


# Readers must bound input before JSON parsing as well as writers bounding output:
# recovery examines untrusted/legacy records and cannot allocate an arbitrary file.
MAX_RECOVERY_RECORD_BYTES = 64 * 1024
#: Structural bound, enforced BEFORE parsing. Real records nest at most a few
#: levels (record -> handoff -> occurrences -> entry). Relying on the
#: interpreter's recursion limit instead is not a bound: since Python 3.12 the
#: C JSON scanner no longer trips ``sys.getrecursionlimit()``, so a 2000-deep
#: 4 KiB record parsed successfully there and then reached callers that recurse
#: over it (``copy.deepcopy``, ``json.dumps``) and do not expect RecursionError.
MAX_RECOVERY_RECORD_DEPTH = 32
#: Only the characters that change string/escape/depth state. A single-char
#: class cannot backtrack, so ``finditer`` is one linear pass. (A string-masking
#: ``re.sub`` was quadratic on an unterminated string of escaped quotes: 13.4 s
#: for 65,533 B in review round 1, M1.)
_JSON_STRUCTURE = re.compile(r'["\\\[\]{}]')


def _within_depth(text: str) -> bool:
    """One linear scan: is no container nested deeper than the bound?

    Tracks in-string and escape state so brackets inside strings are not
    structure. A backslash inside a string escapes exactly the next character,
    whichever it is. Malformed input can only make this stricter or leave it to
    ``json.loads`` to refuse; it never accepts what the parser would reject.
    """
    depth = 0
    in_string = False
    escaped_at = -1
    for match in _JSON_STRUCTURE.finditer(text):
        at = match.start()
        if at == escaped_at:
            continue
        character = match.group()
        if in_string:
            if character == "\\":
                escaped_at = at + 1
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_RECOVERY_RECORD_DEPTH:
                return False
        elif character in "]}":
            depth -= 1
    return True


def parse_record_bytes(data: bytes) -> dict[str, Any] | None:
    """Decode one bounded recovery record; ``None`` for anything not trustworthy.

    Shared by every recovery reader so the byte, encoding and nesting bounds
    cannot drift between the live reader and the dry-run classifier. The depth
    check runs BEFORE ``json.loads`` and is linear in the input.
    """
    if len(data) > MAX_RECOVERY_RECORD_BYTES:
        return None
    try:
        text = data.decode("utf-8")
        if not _within_depth(text):
            return None
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def read_record(path: Path) -> dict[str, Any] | None:
    # Observation must not follow an unsafe lock/attempt name into unrelated
    # bytes, nor block on a FIFO. No create/acquire/update is involved here.
    descriptor: int | None = None
    data = b""
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return None
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        sample = os.fstat(descriptor)
        if (
            not stat.S_ISREG(sample.st_mode)
            or sample.st_nlink != 1
            or sample.st_size > MAX_RECOVERY_RECORD_BYTES
            or (os.name == "posix" and sample.st_uid != os.getuid())
        ):
            return None
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            # Read bytes, not characters: multibyte JSON must not multiply the
            # 64 KiB writer contract, including when a file grows after fstat.
            data = stream.read(MAX_RECOVERY_RECORD_BYTES + 1)
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return parse_record_bytes(data)


@dataclass(frozen=True)
class Transaction:
    root: Path
    descriptor: int
    pid: int
    thread: int
    root_device: int
    root_inode: int

    def validate(self, root: Path) -> None:
        if getattr(_LOCAL, "tokens", {}).get(self.root) is not self:
            raise InstallTransactionError("install transaction no longer holds its root inode")
        if (
            self.root != canonical_root(root)
            or self.pid != os.getpid()
            or self.thread != threading.get_ident()
        ):
            raise InstallTransactionError("install transaction belongs to another root or owner")
        try:
            held = os.fstat(self.descriptor)
            named = (self.root / LOCK_NAME).lstat()
            directory = self.root.lstat()
        except OSError as exc:
            raise InstallTransactionError(
                "install transaction no longer holds its root inode"
            ) from exc
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != os.getuid()
            or (directory.st_dev, directory.st_ino) != (self.root_device, self.root_inode)
            or not stat.S_ISREG(named.st_mode)
            or named.st_uid != os.getuid()
            or named.st_nlink != 1
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise InstallTransactionError("install lock inode was replaced; refusing mutation")


def _validate_lock_identity(root: Path, root_fd: int, fd: int) -> None:
    # Callers already refuse unsupported locking, but this helper states its
    # own precondition: os.getuid is absent off POSIX, including on Windows.
    if os.name != "posix":
        raise LockUnsupported("install lock identity requires POSIX; nothing was changed")
    held = os.fstat(fd)
    named = os.stat(LOCK_NAME, dir_fd=root_fd, follow_symlinks=False)
    directory = os.fstat(root_fd)
    visible = root.lstat()
    if (
        not stat.S_ISREG(held.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or held.st_uid != os.getuid()
        or held.st_nlink != 1
        or held.st_mode & 0o022
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        or not stat.S_ISDIR(visible.st_mode)
        or (directory.st_dev, directory.st_ino) != (visible.st_dev, visible.st_ino)
    ):
        raise InstallTransactionError(
            "unsafe or changed install lock/root identity; nothing was changed"
        )


@contextmanager
def transaction(
    root: Path, *, operation: str, timeout: float = DEFAULT_TIMEOUT_S
) -> Iterator[Transaction]:
    """Bounded, fail-closed, process- and thread-exclusive root mutation.

    The inode is NEVER removed, including after crash or cancellation: unlinking
    a stale-looking lock gives new writers a second inode while a holder still
    owns the first. Metadata is decoration; only the kernel lock grants entry.
    Windows is explicitly refused until its equivalent semantics are validated.
    """
    try:
        import fcntl
    except ImportError as exc:
        raise LockUnsupported(
            "install transactions require POSIX advisory locking; this platform is "
            "unsupported, so nothing was changed"
        ) from exc
    root = canonical_root(root)
    with _MUTEX_GUARD:
        mutex = _MUTEXES.setdefault(root, threading.RLock())
    deadline = time.monotonic() + max(0.0, timeout)
    if not mutex.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise LockBusy(f"another install mutation owns {root}; retry when it settles")
    tokens = getattr(_LOCAL, "tokens", None)
    if tokens is None:
        tokens = _LOCAL.tokens = {}
    fd: int | None = None
    root_fd: int | None = None
    locked = False
    try:
        existing = tokens.get(root)
        if existing is not None:
            # Reuse the actual held handle, never acquire a second file lock.
            existing.validate(root)
            yield existing
            return
        root.mkdir(parents=True, exist_ok=True)
        # Anchor creation to the owned directory; never follow a lock symlink
        # or truncate unrelated bytes merely because flock accepted its inode.
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        directory = os.fstat(root_fd)
        if directory.st_uid != os.getuid() or not stat.S_ISDIR(directory.st_mode):
            raise InstallTransactionError(
                "install root is not an owned directory; nothing was changed"
            )
        fd = os.open(
            LOCK_NAME,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=root_fd,
        )
        _validate_lock_identity(root, root_fd, fd)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise LockBusy(f"cannot lock install root {root}: {exc}") from exc
                if time.monotonic() >= deadline:
                    raise LockBusy(
                        f"another install mutation owns {root}; retry when it settles "
                        "(lop install status shows observational details)"
                    ) from None
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        _validate_lock_identity(root, root_fd, fd)
        payload = {
            **_identity(),
            "root": str(root),
            "operation": operation,
            "started_at": time.time(),
        }
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(payload).encode())
        token = Transaction(
            root, fd, os.getpid(), threading.get_ident(), directory.st_dev, directory.st_ino
        )
        tokens[root] = token
        try:
            token.validate(root)
            yield token
        finally:
            tokens.pop(root, None)
    except OSError as exc:
        raise InstallTransactionError(f"cannot mutate install root {root}: {exc}") from exc
    finally:
        if fd is not None:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)
        mutex.release()


OPERATION_DIR = ".update-operations"
#: The future-only fenced protocol's namespace (see ``fenced_operation``). Named
#: HERE so legacy admission can count it without importing that module (which
#: imports this one): both protocols share ONE active-recovery budget.
V2_OPERATION_DIR = ".update-operations-v2"
#: The v2 record's protocol tag, also named here so the read-only capacity dry
#: run can tell a well-formed v2 record from an unreadable one.
V2_PROTOCOL = "update-operation.v2"
#: Generation sidecar names, held HERE as the single source of truth for both
#: protocols (``fenced_operation`` and ``recovery_manifest`` alias them). The
#: legacy name is the literal the legacy functions below already read/write.
LEGACY_SIDECAR_NAME = ".update-operation.json"
V2_SIDECAR_NAME = ".update-operation-v2.json"
#: The repo's one platform question for owner checks (see ``secrets/protocol.py``):
#: ``os.getuid`` does not exist off POSIX, and a Windows ``st_uid`` is always 0,
#: so ownership is only compared where a uid means something. Elsewhere the
#: other refusals (symlink, type, identity, lock) still apply.
_UID_IS_MEANINGFUL = os.name == "posix"
_OPERATION_PHASES = {
    "queued",
    "preparing",
    "prepared",
    "promoted",
    "services",
    "partial",
    "failed",
    "capacity",
}


def _operation_path(root: Path, identifier: str, *, create: bool = False) -> Path:
    if len(identifier) != 32 or any(
        character not in "0123456789abcdef" for character in identifier
    ):
        raise InstallTransactionError("update operation identity is invalid")
    directory = root / OPERATION_DIR
    if create and not directory.exists() and not directory.is_symlink():
        directory.mkdir(mode=0o700)
    if directory.exists() or directory.is_symlink():
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise InstallTransactionError(
                "update operation directory is unsafe; no record was written"
            )
    return directory / f"{identifier}.json"


def _persist_operation(path: Path, record: dict[str, Any]) -> None:
    data = json.dumps(record, sort_keys=True).encode("utf-8")
    if len(data) > MAX_RECOVERY_RECORD_BYTES:
        raise InstallTransactionError("update operation exceeds its bounded recovery record")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    staged = f".{uuid.uuid4().hex}.tmp"
    try:
        before = os.fstat(directory)
        after = path.parent.lstat()
        if (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ) or before.st_uid != os.getuid():
            raise InstallTransactionError(
                "update operation directory identity changed; no record written"
            )
        descriptor = os.open(
            staged,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        after = path.parent.lstat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise InstallTransactionError(
                "update operation directory moved; preserve prior recovery"
            )
        os.replace(staged, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(staged, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


#: A capacity request's states. ``requested`` is open until the controller has
#: acted; the others record the outcome so a re-entry is idempotent rather than a
#: second request for the same pressure.
CAPACITY_STATES = ("requested", "moved", "reclaimed", "partial")


@dataclass(frozen=True)
class UpdateOperation:
    """Durable intent/receipts, not proof of OS ownership or service completion.

    One record is advanced under the EXISTING canonical-root mutex, while builds
    remain outside it. A crash cannot erase the last committed phase/receipts;
    recovery still owes fresh publication/current/leadership checks. No caller
    may turn an unknown or pending central observation into a complete phase.
    """

    root: Path
    identifier: str

    def advance(self, phase: str, **observations: Any) -> None:
        if phase not in _OPERATION_PHASES:
            raise InstallTransactionError(
                "update operation phase has no verified completion contract"
            )
        if set(observations) - {
            "candidate",
            "current",
            "receipts",
            "continuations",
            "failure",
            "handoff",
        }:
            raise InstallTransactionError("update operation observation is unsupported")
        with transaction(self.root, operation=f"update operation {phase}") as held:
            held.validate(self.root)
            path = _operation_path(self.root, self.identifier)
            record = read_record(path)
            if record is None or record.get("operation_id") != self.identifier:
                raise InstallTransactionError(
                    "durable update operation is unreadable; retry recovery"
                )
            record.update(observations)
            record.update(phase=phase, updated_at=time.time(), phase_actor=_identity())
            _persist_operation(path, record)
            held.validate(self.root)


def _namespace_entries(root: Path, bound: int) -> Iterator[tuple[str, Path | None]]:
    """THE one enumeration both admission and the status dry run use.

    Yields ``(namespace, entry)`` over the legacy then the v2 namespace, and
    stops after ``bound + 1`` entries in TOTAL (one overflow sentinel), so a
    flooded directory cannot make either caller unbounded. An unsafe namespace
    directory (symlink, foreign owner, not a directory) yields
    ``(namespace, None)`` once and is never followed: admission refuses on it,
    the dry run reports it. Read-only: no create, no lock, no record read.
    """
    seen = 0
    for name in (OPERATION_DIR, V2_OPERATION_DIR):
        directory = root / name
        if not directory.exists() and not directory.is_symlink():
            continue
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (_UID_IS_MEANINGFUL and info.st_uid != os.getuid()):
            yield name, None
            continue
        for entry in directory.iterdir():
            seen += 1
            yield name, entry
            if seen > bound:
                return


def active_recovery_entries(root: Path, *, limit: int | None = None) -> int:
    """Count legacy AND v2 namespace entries against ONE shared budget.

    Every entry counts, unknown names included, exactly like the original legacy
    loop: neither namespace may manufacture spare slots or exclude protected
    legacy obligations. Bounded: stops one past ``limit`` so a flooded directory
    cannot make admission unbounded. An unsafe directory refuses admission
    rather than being skipped, because skipping would undercount.
    """
    bound = RecoveryCapacityError.active_limit if limit is None else limit
    count = 0
    for _name, entry in _namespace_entries(root, bound):
        if entry is None:
            raise InstallTransactionError(
                "update operation directory is unsafe; no record was written"
            )
        count += 1
    return count


#: In-root archive for RETIRED recovery records (operator request 2026-11-16:
#: "make sure that this store properly rotates so that users don't run into this
#: hard limit"). It lives INSIDE the INSTALLROOT on purpose:
#:
#: * a retirement is then a same-filesystem ``os.rename`` -- atomic, and the
#:   record keeps its inode, so its bytes, mode and mtime are preserved without
#:   a copy (no content store is needed; see ``_archive_moves``);
#: * rotation and aging never read, age or write anything OUTSIDE the root. A
#:   sibling such as ``<parent>/lop-record-archive/`` (the reviewed manual
#:   archive of 2026-11-16) is never discovered, let alone aged;
#: * it is NOT one of the namespaces ``_namespace_entries`` enumerates, so an
#:   archived record frees its admission slot while staying restorable with
#:   :func:`restore_archived` (a rename back).
ARCHIVE_DIR = ".update-operations-archive"
#: Rotation switch. A module flag, not configuration: it exists so the negative
#: control (rotation off -> the 256 refusal still fires) and an emergency
#: bisect can turn retirement off without touching admission itself.
RECORD_ROTATION_ENABLED = True
#: Sealed v2 records kept LIVE after rotation, newest first, for diagnostics
#: (``lop install status``/``operation_status`` read them without the archive).
ROTATION_KEEP_SEALED = 32
#: Archive bounds: the archive must not become the next unbounded store. Count
#: and age are both enforced on every rotation; the oldest archive entries go
#: first. These unlinks are the ONLY deletions this module performs on records.
ARCHIVE_MAX_ENTRIES = 512
ARCHIVE_MAX_AGE_S = 30 * 24 * 60 * 60
#: "Near the cap" for LEGACY auto-migration. Legacy records have no positive
#: finality, so they are only archived under pressure, never routinely: 240 of
#: 256 leaves 16 admissions of headroom (several updates' worth of retries) so
#: the migration runs well before a user can hit the refusal, yet a store with
#: ordinary history (tens of records) is never touched.
LEGACY_MIGRATION_THRESHOLD = 240
#: A legacy candidate whose record changed within this window is left live even
#: under pressure. Age is never finality proof; it is used ONLY to exclude (like
#: phase in ``recovery_manifest``), so a controller tail or report that is still
#: advancing a just-superseded operation (``operation_for_generation`` ->
#: ``advance``) does not find its record gone. Installer attempts expire after
#: 1800 s, so one hour covers a whole update plus its service tail.
LEGACY_MIGRATION_GRACE_S = 60 * 60
#: Archive entry names: ``<archived_at_ms>.<v1|v2>.<operation id>.json``. The
#: millisecond stamp is the archive time (rename keeps the record's own mtime,
#: so the mtime cannot carry it); the tag says which namespace a restore
#: returns the record to. Anything else in the archive is foreign: never
#: indexed, aged or deleted.
_ARCHIVE_ENTRY = re.compile(r"^([0-9]{13,})\.(v1|v2)\.([0-9a-f]{32})\.json$")
_ARCHIVE_TAGS = {OPERATION_DIR: "v1", V2_OPERATION_DIR: "v2"}
#: Bounded directory scans: a flooded archive or namespace cannot make
#: admission unbounded. Entries past this bound are simply not considered.
_ROTATION_SCAN_BOUND = 4096
_CAPACITY_SENTENCE = "update recovery records are at capacity; inspect pending operations"
_log = logging.getLogger(__name__)


def _archive_directory(root: Path, *, create: bool) -> Path | None:
    """The validated in-root archive directory, or ``None`` when absent.

    Same refusals as the namespace directories: a symlink, non-directory or
    (where a uid means something) foreign-owned archive is never followed,
    so a planted link cannot redirect retired records -- or aging unlinks --
    outside the root.
    """
    directory = root / ARCHIVE_DIR
    if create and not directory.exists() and not directory.is_symlink():
        directory.mkdir(mode=0o700)
    if not directory.exists() and not directory.is_symlink():
        return None
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or (_UID_IS_MEANINGFUL and info.st_uid != os.getuid()):
        raise InstallTransactionError("update operation archive is unsafe; nothing was moved")
    return directory


def _archive_index(directory: Path | None) -> dict[str, tuple[int, str, str]]:
    """``operation id -> (archived_at_ms, tag, entry name)`` for OUR entries only.

    Bounded (``_ROTATION_SCAN_BOUND``) and read-only. Foreign names are left
    out, which is what keeps them out of aging too.
    """
    index: dict[str, tuple[int, str, str]] = {}
    if directory is None:
        return index
    with os.scandir(directory) as entries:
        for seen, entry in enumerate(entries):
            if seen >= _ROTATION_SCAN_BOUND:
                break
            match = _ARCHIVE_ENTRY.match(entry.name)
            if match is not None and entry.is_file(follow_symlinks=False):
                index[match.group(3)] = (int(match.group(1)), match.group(2), entry.name)
    return index


def _fsync_directory(directory: Path) -> None:
    """Make renames into/out of ``directory`` durable (review M3 of the manual run)."""
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _move_entry(source: Path, target: Path) -> None:
    """THE one rename for retirement and restore; never replaces an existing name.

    ``os.rename`` would silently replace ``target``, so absence is checked
    first. That check is sufficient because every caller holds the root
    transaction and record names are uuid4 ids no cooperating writer reuses.
    Same filesystem by construction (both ends are inside the root; callers
    also compare ``st_dev`` before the first move), so the rename is atomic
    and keeps the inode: bytes, mode and mtime are unchanged.
    """
    if target.exists() or target.is_symlink():
        raise InstallTransactionError(f"refusing to replace {target.name}; nothing was moved")
    os.rename(source, target)


def _archive_moves(
    root: Path, namespace: str, moves: Sequence[tuple[str, Path]], now: float
) -> list[str]:
    """Move verified records of ONE namespace into the archive; return their ids.

    Every source and every target is checked BEFORE the first rename (review
    M1): a refusal leaves the store exactly as it was. A crash between renames
    leaves each record wholly in one place (rename is atomic), and the next
    rotation simply re-plans from what is still live (review M2). Both
    directories are fsynced afterwards, including on failure (review M3).
    """
    if not moves:
        return []
    source_dir = root / namespace
    info = source_dir.lstat()
    if not stat.S_ISDIR(info.st_mode) or (_UID_IS_MEANINGFUL and info.st_uid != os.getuid()):
        raise InstallTransactionError("update operation directory is unsafe; nothing was moved")
    archive = _archive_directory(root, create=True)
    if archive is None:
        raise InstallTransactionError("update operation archive is missing; nothing was moved")
    if archive.lstat().st_dev != info.st_dev:
        raise InstallTransactionError("update operation archive is on another filesystem")
    stamp = max(0, int(now * 1000))
    tag = _ARCHIVE_TAGS[namespace]
    archived = _archive_index(archive)
    plan: list[tuple[str, Path, Path]] = []
    for identifier, source in moves:
        target = archive / f"{stamp:013d}.{tag}.{identifier}.json"
        if (
            source.parent != source_dir
            or source.is_symlink()
            or not source.is_file()
            or identifier in archived
            or target.exists()
            or target.is_symlink()
        ):
            raise InstallTransactionError(
                f"cannot archive update operation {identifier}; nothing was moved"
            )
        plan.append((identifier, source, target))
    moved: list[str] = []
    try:
        for identifier, source, target in plan:
            _move_entry(source, target)
            moved.append(identifier)
    finally:
        _fsync_directory(source_dir)
        _fsync_directory(archive)
    return moved


def _mtime_ns(path: Path) -> int:
    """A sort key only; an entry that vanished or cannot be stat'ed sorts first."""
    try:
        return path.lstat().st_mtime_ns
    except OSError:
        return 0


def _age_archive(root: Path, now: float) -> list[str]:
    """Bound the archive by age, then by count; return the names removed.

    The ONLY unlink of a recovery record anywhere in the update modules, and
    only of OUR archive names (``_ARCHIVE_ENTRY``), only inside the validated
    in-root archive, through one ``O_DIRECTORY|O_NOFOLLOW`` descriptor whose
    identity matches the validated directory. Nothing outside the root is
    ever looked at. Oldest archive time goes first; a future stamp (clock
    skew) counts as new rather than being dropped early.
    """
    directory = _archive_directory(root, create=False)
    if directory is None:
        return []
    now_ms = int(now * 1000)
    # Order by archive stamp, then by the record's OWN mtime (rename preserves
    # it: the last write, i.e. the seal for v2 or the last advance for legacy).
    # One rotation stamps a whole batch alike, so without the mtime the count
    # trim inside a batch would follow uuid order, i.e. be random (QA Q4, #4126).
    ordered = sorted(
        (stamp, _mtime_ns(directory / name), name)
        for stamp, _tag, name in _archive_index(directory).values()
    )
    expired = [name for stamp, _m, name in ordered if now_ms - stamp > ARCHIVE_MAX_AGE_S * 1000]
    gone = set(expired)
    kept = [name for _stamp, _m, name in ordered if name not in gone]
    overflow = kept[: max(0, len(kept) - ARCHIVE_MAX_ENTRIES)]
    drop = expired + overflow
    if not drop:
        return []
    expected = directory.lstat()
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    removed: list[str] = []
    try:
        held = os.fstat(handle)
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise InstallTransactionError("update operation archive moved; nothing was aged")
        for name in drop:
            info = os.stat(name, dir_fd=handle, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                continue
            os.unlink(name, dir_fd=handle)
            removed.append(name)
        os.fsync(handle)
    finally:
        os.close(handle)
    return removed


def rotate_for_admission(
    root: Path, held: Transaction, *, now: float | None = None
) -> dict[str, list[str]]:
    """Free admission slots by archiving ONLY records that are provably done.

    Called by both admission paths (through ``_admit``) while the root
    transaction is HELD, so two updaters can never plan against the same
    records: the second waits on the lock (or gets ``LockBusy``) and then sees
    the first one's result. Observers (``operation_status``,
    ``recovery_capacity_status``, ``observe_generation``) never call this.

    1. Sealed v2 records whose generations no top-level pointer references are
       archived oldest-seal first, keeping the newest ``ROTATION_KEEP_SEALED``
       live. A seal is positive finality: ``_fenced`` refuses every write to a
       sealed record, so no writer can be cut off by the move.
    2. Only when legacy+v2 entries are at ``LEGACY_MIGRATION_THRESHOLD`` or
       more, the reviewed legacy classifier (``recovery_manifest``) runs and
       its candidates (never an excluded class) are archived -- legacy records
       cannot be sealed, so this is the operator-approved abandonment of
       finality-unproven history, not a finality claim.
    3. The archive is aged by count and time.

    ``now`` is injectable for tests (no sleeping); production uses the clock.
    """
    report: dict[str, list[str]] = {"v2_archived": [], "legacy_archived": [], "aged": []}
    if not RECORD_ROTATION_ENABLED:
        return report
    # Imported here, not at module level: both modules import this one.
    from local_operator import recovery_manifest as _manifest

    # Legacy admission passes the caller's spelling of the root; every path
    # comparison below (and in the classifiers) is against the canonical one.
    root = canonical_root(root)
    held.validate(root)
    clock = time.time() if now is None else now
    try:
        retirable = _manifest.retirable_sealed_v2(root, keep=ROTATION_KEEP_SEALED)
        report["v2_archived"] = _archive_moves(root, V2_OPERATION_DIR, retirable, clock)
        if active_recovery_entries(root) >= LEGACY_MIGRATION_THRESHOLD:
            manifest = _manifest.dry_run_manifest(root)
            if not manifest["root_uncertain"] and manifest["candidates"]:
                # The digest binds the moves to exactly the bytes just classified;
                # approval itself is the standing operator policy quoted above.
                result = _manifest.archive_candidates(
                    manifest,
                    approved_sha256=manifest["manifest_sha256"],
                    now=clock,
                    min_age_s=LEGACY_MIGRATION_GRACE_S,
                )
                report["legacy_archived"] = result["archived"]
    finally:
        # Aging runs even when a step above raised (review R3 on #4126): the
        # archive's 512/30-day bound must not depend on every admission's
        # migration succeeding, or a persistent legacy refusal would let v2
        # retirements grow the archive without limit. The step's exception
        # still propagates to ``_admit`` after this.
        report["aged"] = _age_archive(root, clock)
    held.validate(root)
    return report


def _retirable_estimate(root: Path, observed: int, now: float) -> int:
    """How many entries the next admission's rotation would retire; read-only.

    The status dry run's half of review R2 on #4126: it runs the SAME selection
    functions as :func:`rotate_for_admission` (``retirable_sealed_v2``, then
    the legacy classifier only when the remainder is at the migration
    threshold, minus the grace-window deferrals), without a lock and without
    moving anything. Any failure answers 0, which is what admission does when
    rotation fails: it proceeds on the unrotated count.
    """
    if not RECORD_ROTATION_ENABLED:
        return 0
    from local_operator import recovery_manifest as _manifest

    try:
        v2 = len(_manifest.retirable_sealed_v2(root, keep=ROTATION_KEEP_SEALED))
        legacy = 0
        if observed - v2 >= LEGACY_MIGRATION_THRESHOLD:
            manifest = _manifest.dry_run_manifest(root)
            if not manifest["root_uncertain"]:
                legacy = sum(
                    1
                    for candidate in manifest["candidates"]
                    if now - candidate["mtime_ns"] / 1e9 >= LEGACY_MIGRATION_GRACE_S
                )
    except Exception:  # noqa: BLE001
        return 0
    return v2 + legacy


def _admit(root: Path, held: Transaction) -> None:
    """THE shared admission gate for legacy and v2 ``start_operation``.

    Rotation first, then the unchanged cap check. A rotation failure never
    becomes a new hard stop: below the cap the update proceeds (the failure is
    logged), and at the cap the SAME typed ``RecoveryCapacityError`` sentence
    and code are raised, chained to the rotation failure for diagnosis. When
    only protected records remain, rotation frees nothing and the refusal is
    exactly the pre-rotation one.
    """
    failure: Exception | None = None
    try:
        rotate_for_admission(root, held)
    except Exception as exc:  # noqa: BLE001
        # Broad on purpose (review R1 / QA Q1 on #4126): a malformed record can
        # raise TypeError/KeyError/ValueError from a classifier, and NO rotation
        # defect may become a refusal the base build would not have made.
        # BaseException (KeyboardInterrupt, SystemExit) still propagates.
        failure = exc
        _log.warning("update record rotation failed; checking the cap without it: %s", exc)
        # Whatever failed, the lock must still be ours before admission writes.
        held.validate(root)
    if active_recovery_entries(root) >= RecoveryCapacityError.active_limit:
        if failure is not None:
            raise RecoveryCapacityError(_CAPACITY_SENTENCE) from failure
        raise RecoveryCapacityError(_CAPACITY_SENTENCE)


def restore_archived(root: Path, identifier: str) -> Path:
    """Rename ONE archived record back into its namespace; return its live path.

    The documented undo for rotation and legacy migration. Refuses when the
    live name exists (never replaces) and when the shared cap is full (a
    restore must not mint the slot admission refuses). Note: a restored record
    that is still retirable is archived again by the next admission; restore
    to inspect, or with ``RECORD_ROTATION_ENABLED`` off to keep it live.
    """
    root = canonical_root(root)
    _operation_path(root, identifier)  # validates the id shape only
    with transaction(root, operation="restore archived update operation") as held:
        held.validate(root)
        archive = _archive_directory(root, create=False)
        found = _archive_index(archive).get(identifier)
        if archive is None or found is None:
            raise InstallTransactionError("no archived update operation has that id")
        _stamp, tag, name = found
        namespace = OPERATION_DIR if tag == "v1" else V2_OPERATION_DIR
        if active_recovery_entries(root) >= RecoveryCapacityError.active_limit:
            raise RecoveryCapacityError(_CAPACITY_SENTENCE)
        directory = root / namespace
        if not directory.exists() and not directory.is_symlink():
            directory.mkdir(mode=0o700)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (_UID_IS_MEANINGFUL and info.st_uid != os.getuid()):
            raise InstallTransactionError("update operation directory is unsafe; nothing moved")
        target = directory / f"{identifier}.json"
        try:
            _move_entry(archive / name, target)
        finally:
            _fsync_directory(archive)
            _fsync_directory(directory)
        held.validate(root)
    return target


def start_operation(root: Path, *, baseline: str, target: str, route: str) -> UpdateOperation:
    identifier = uuid.uuid4().hex
    with transaction(root, operation="register durable update operation") as held:
        held.validate(root)
        # Bound namespace growth without deleting unknown/unfinished operations.
        # v2 entries count too: otherwise each protocol would see its own 256 and
        # together they would form the second capacity pool the cap forbids.
        # Counted BEFORE the legacy directory is created, so a refusal (full, or
        # an unsafe v2 directory) leaves no new directory behind (round 1, Q1);
        # creating an empty directory adds no entry, so the count is unchanged.
        # ``_admit`` first retires provably-finished records (rotation), so the
        # cap is a backstop for protected records rather than a lifetime limit.
        _admit(root, held)
        path = _operation_path(root, identifier, create=True)
        _persist_operation(
            path,
            {
                **_identity(),
                "operation_id": identifier,
                "baseline": baseline,
                "target": target,
                "route": route,
                "phase": "queued",
                "candidate": "",
                "receipts": [],
                "continuations": [],
                "updated_at": time.time(),
            },
        )
    return UpdateOperation(root, identifier)


def operation_for_generation(generation: Path) -> UpdateOperation | None:
    """Find the immutable candidate's operation, not 'latest operation' by time.

    A service tail may run in another process after C supersedes B. Selecting by
    timestamp would overwrite C's recovery receipts on B's behalf. The candidate
    sidecar only identifies intent; it grants no install/collection permission.
    """
    record = read_record(generation / ".update-operation.json")
    if record is None or not isinstance(record.get("operation_id"), str):
        return None
    root = generation.parent.parent
    path = _operation_path(root, record["operation_id"])
    saved = read_record(path)
    if saved is None or saved.get("candidate") != str(generation):
        raise InstallTransactionError("candidate recovery identity disagrees; preserve operation")
    return UpdateOperation(root, record["operation_id"])


def link_operation(operation: UpdateOperation, generation: Path) -> None:
    with transaction(operation.root, operation="link candidate recovery identity") as held:
        held.validate(operation.root)
        if generation.is_symlink() or generation.parent != operation.root / "generations":
            raise InstallTransactionError(
                "candidate recovery is outside the canonical generation store"
            )
        _persist_operation(
            generation / ".update-operation.json",
            {"operation_id": operation.identifier},
        )


def handoff_status(root: Path, identifier: str) -> dict[str, Any] | None:
    """Read the transfer record; a read never advances or repairs it."""
    row = operation_status(root, identifier)
    if row is None:
        return None
    handoff = row.get("handoff")
    return handoff if isinstance(handoff, dict) else None


#: Occurrence record vocabulary. ``running``/``terminal`` is the only durable
#: signal a successor can trust; a crash between the two writes leaves a
#: readable ``running`` entry, which is exactly the signal that must survive.
OCCURRENCE_KINDS = ("scheduler", "relay")
OCCURRENCE_TERMINALS = ("done", "failed", "cancelled")


def record_occurrence(
    root: Path,
    identifier: str,
    occurrence_id: str,
    *,
    kind: str,
    state: str,
    terminal: str = "",
) -> None:
    """Write one occurrence marker into the EXISTING durable operation record.

    No new lock and no new ledger: this rides ``handoff.occurrences`` on the
    operation the transaction already guards. The successor's exactly-once
    guarantee is that it suppresses an inherited id ONLY while this entry is
    present and non-terminal, so a crash between the ``running`` write and the
    ``terminal`` write leaves a running entry — the conservative, readable
    direction — rather than a silent gap.
    """
    from local_operator import procstate

    if kind not in OCCURRENCE_KINDS:
        raise InstallTransactionError("occurrence kind has no durable contract")
    if state not in ("running", "terminal"):
        raise InstallTransactionError("occurrence state has no durable contract")
    if state == "terminal" and terminal not in OCCURRENCE_TERMINALS:
        raise InstallTransactionError("terminal occurrence needs a terminal outcome")
    with transaction(root, operation=f"occurrence {state}") as held:
        held.validate(root)
        path = _operation_path(root, identifier)
        record = read_record(path)
        if record is None or record.get("operation_id") != identifier:
            raise InstallTransactionError("occurrence has no durable operation; preserving work")
        # ``record.get("handoff")`` is split out and NARROWED into a typed local: the
        # inline conditional makes pyright read ``handoff`` as ``object | None``,
        # which then fails every read and write below it. ``raw`` keeps the
        # runtime rule (a non-dict handoff is discarded, not trusted) while giving
        # the analyzer the dict it can subscript.
        raw_handoff = record.get("handoff")
        handoff: dict[str, Any] = raw_handoff if isinstance(raw_handoff, dict) else {}
        occurrences = handoff.get("occurrences")
        if not isinstance(occurrences, dict):
            occurrences = {}
        occurrences[occurrence_id] = {
            "kind": kind,
            "state": state,
            "started_by_pid": os.getpid(),
            "birth": f"{procstate.birth_scheme()}:{procstate.self_birth_token()}",
            "terminal": terminal,
            "recorded_at": time.time(),
        }
        handoff["occurrences"] = occurrences
        record["handoff"] = handoff
        record["updated_at"] = time.time()
        _persist_operation(path, record)
        held.validate(root)


def note_capacity(
    root: Path,
    identifier: str,
    *,
    state: str,
    threshold: str,
    cohort: Sequence[str] = (),
    requested_at: float | None = None,
) -> dict[str, Any]:
    """Write the capacity sub-map into the EXISTING durable operation record.

    Rides ``handoff.capacity`` (never a new key), under the same root transaction
    the operation already holds — the same vehicle as ``handoff.occurrences``, and
    the reason a re-entry is idempotent rather than a second queue: the reader sees
    an OPEN request and stands down. It MERGES into the existing ``handoff`` dict
    rather than replacing it, because ``advance()`` sets the whole key and would
    otherwise drop the occurrence map written beside it.

    ``threshold`` and ``state`` are closed vocabularies so a torn or foreign record
    cannot smuggle an unbounded request; ``cohort`` is the generation NAMES the
    request is about, stored as strings because the record is JSON.
    """
    if state not in CAPACITY_STATES:
        raise InstallTransactionError("capacity state has no verified contract")
    if threshold not in ("count", "free", "both"):
        raise InstallTransactionError("capacity threshold has no verified contract")
    with transaction(root, operation=f"capacity {state}") as held:
        held.validate(root)
        path = _operation_path(root, identifier)
        record = read_record(path)
        if record is None or record.get("operation_id") != identifier:
            raise InstallTransactionError("capacity has no durable operation; preserving roots")
        raw_handoff = record.get("handoff")
        handoff: dict[str, Any] = raw_handoff if isinstance(raw_handoff, dict) else {}
        raw_previous = handoff.get("capacity")
        previous: dict[str, Any] = raw_previous if isinstance(raw_previous, dict) else {}
        # Annotated so ``update``/``return`` see a ``dict[str, Any]`` rather than the
        # ``dict[Unknown, Unknown]`` a bare ``dict(previous)`` infers from the
        # narrowed value.
        entry: dict[str, Any] = dict(previous)
        entry.update(
            state=state,
            threshold=threshold,
            cohort=[str(name) for name in cohort],
            # PRESERVE the original request time on a re-entry: the durable fact is
            # when the pressure was first seen, not when the caller last looked.
            requested_at=(
                previous.get("requested_at") or time.time()
                if requested_at is None
                else requested_at
            ),
            recorded_at=time.time(),
        )
        handoff["capacity"] = entry
        record["handoff"] = handoff
        record["updated_at"] = time.time()
        _persist_operation(path, record)
        held.validate(root)
        return entry


def capacity_status(root: Path, identifier: str) -> dict[str, Any] | None:
    """Read the capacity sub-map; ``None`` means unreadable/missing (not empty)."""
    row = operation_status(root, identifier)
    if row is None:
        return None
    handoff = row.get("handoff")
    if not isinstance(handoff, dict):
        return None
    capacity = handoff.get("capacity")
    if not isinstance(capacity, dict):
        return None
    return capacity


def occurrence_status(root: Path, identifier: str) -> dict[str, dict[str, Any]] | None:
    """Read the occurrence map; ``None`` means unreadable/missing (not empty).

    The distinction matters: an EMPTY map is "nothing in flight" (feature inert),
    while ``None`` is "unknown" — and the successor must treat unknown as *not
    proven still running* rather than suppressing on it.
    """
    row = operation_status(root, identifier)
    if row is None:
        return None
    handoff = row.get("handoff")
    if not isinstance(handoff, dict):
        return None
    occurrences = handoff.get("occurrences")
    if not isinstance(occurrences, dict):
        return None
    return {key: value for key, value in occurrences.items() if isinstance(value, dict)}


def advance_handoff(
    root: Path,
    identifier: str,
    state: str,
    **fields: Any,
) -> dict[str, Any]:
    """Move the transfer record one state under the EXISTING root transaction.

    The legacy ordering is enforced, not just documented: ``commit`` requires a
    prior ``ready`` from the successor, and ``ack`` requires ``commit``. A state
    skips nothing and an unknown/absent prior state refuses, because a successor
    that accepted before A committed would be a second leader for one port. The
    record is a handoff, not a lock: unknown or unreadable state leaves BOTH roots
    and both processes alive for the owning recovery path to reconcile.
    """
    from local_operator.service_handoff import HANDOFF_STATES

    if state not in HANDOFF_STATES:
        raise InstallTransactionError("handoff state has no verified contract")
    allowed = {
        "nonce",
        "from_pid",
        "from_birth",
        "to_pid",
        "to_birth",
        "target_generation",
        "listener_ino",
        "control_fd_role",
        "inherited",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise InstallTransactionError("handoff field is unsupported")
    with transaction(root, operation=f"handoff {state}") as held:
        held.validate(root)
        path = _operation_path(root, identifier)
        record = read_record(path)
        if record is None or record.get("operation_id") != identifier:
            raise InstallTransactionError("handoff has no durable operation; preserve both roots")
        previous = record.get("handoff") if isinstance(record.get("handoff"), dict) else None
        # `ready` is the FIRST state (no prior); each later state names exactly
        # its predecessor. A `commit` with no `ready`, or an `ack` with no
        # `commit`, refuses — a successor that accepted before A committed would
        # be a second leader for one port.
        required = {"ready": None, "commit": "ready", "ack": "commit", "aborted": None}
        prior_state = (previous or {}).get("state")
        if prior_state != required[state]:
            expected = required[state] or "no prior handoff"
            raise InstallTransactionError(
                f"handoff {state} requires {expected}; refusing out of order"
            )
        merged = dict(previous or {})
        merged.update(fields)
        merged["state"] = state
        record["handoff"] = merged
        record["updated_at"] = time.time()
        _persist_operation(path, record)
        held.validate(root)
        return merged


def _retirement_status(record: dict[str, Any] | None) -> dict[str, Any]:
    """No legacy phase, report, or dead owner is positive writer-finality proof.

    The current receipt protocol has no exhaustive writer-release certificate:
    old occurrence/capacity/handoff writers may still advance an operation. Even
    a purported seal must therefore remain protected, not become a new trust
    input. Keep this dry-run conclusion separate from phase and owner liveness.
    """
    reason = "unreadable_or_unsafe_record"
    if record is not None:
        reason = (
            "unverified_terminal_seal" if "terminal_seal" in record else "writer_finality_unproven"
        )
    return {"eligible": False, "reason": reason}


def recovery_capacity_status(root: Path) -> dict[str, Any]:
    """Read-only, bounded dry run; NEVER retire, repair, or acquire a lock.

    Count every entry of BOTH namespaces exactly as admission does, through the
    same enumeration (``_namespace_entries``), including foreign names and
    unfinished writes, so this diagnostic can never disagree with the refusal it
    explains. At most 256 entries are inspected in total, plus one overflow
    sentinel with an explicit incomplete-scan flag; no historical scan or
    liveness probe can turn unknown entries into available slots. Every entry
    occupies its slot NOW; ``retirable_entries`` is how many of them the next
    admission's rotation would archive (the same selection functions, run
    read-only), and ``at_limit`` is admission's decision after that rotation.
    The one way the two can still differ is a rotation that fails while moving
    (e.g. an unsafe archive directory): admission then refuses on the unrotated
    count. This API itself never moves, migrates or locks anything.
    """
    limit = RecoveryCapacityError.active_limit
    counts: dict[str, int] = {}
    namespaces: dict[str, int] = {OPERATION_DIR: 0, V2_OPERATION_DIR: 0}
    observed = 0
    complete = True
    for name, path in _namespace_entries(root, limit):
        if path is None:
            # Admission refuses on this directory; it was not followed, so the
            # count below is a lower bound.
            complete = False
            counts["unsafe_namespace_directory"] = counts.get("unsafe_namespace_directory", 0) + 1
            continue
        observed += 1
        namespaces[name] += 1
        if observed > limit:
            complete = False
            counts["uninspected_overflow"] = 1
            break
        identifier = path.stem
        if (
            path.name != f"{identifier}.json"
            or len(identifier) != 32
            or any(character not in "0123456789abcdef" for character in identifier)
        ):
            reason = "unknown_namespace_entry"
        else:
            record = read_record(path)
            if record is not None and record.get("operation_id") != identifier:
                record = None
            if name == OPERATION_DIR:
                reason = _retirement_status(record)["reason"]
            elif record is None or record.get("protocol") != V2_PROTOCOL:
                reason = "unreadable_or_unsafe_record"
            else:
                reason = "new_protocol_sealed" if record.get("sealed") else "new_protocol_open"
        counts[reason] = counts.get(reason, 0) + 1
    # What admission's rotation would free first (review R2 on #4126), from the
    # same selection functions, read-only. Without it a store of 256 sealed v2
    # records reads "at limit" while ``start_operation`` admits.
    # Only on a COMPLETE scan: the estimate reads each candidate record, and a
    # store past the bound must stay a bounded read (256 records + sentinel).
    # Such a store is reported at the limit without an estimate -- admission
    # can never have produced it, so only a flood or foreign writer reaches it.
    retirable = 0
    if complete:
        retirable = min(observed, _retirable_estimate(root, observed, time.time()))
    return {
        "active_limit": limit,
        "observed_entries": observed,
        "namespace_entries": namespaces,
        # The comparison admission makes AFTER rotation; with an incomplete scan
        # the observed count is a lower bound that already reaches the limit.
        "at_limit": observed - retirable >= limit,
        "scan_complete": complete,
        "protected_entries": observed - retirable,
        "retirable_entries": retirable,
        "protected_reasons": counts,
        "mutation_performed": False,
    }


def operation_status(root: Path, identifier: str) -> dict[str, Any] | None:
    """Observe only; never acquire/create/clear the transaction or its records."""
    path = _operation_path(root, identifier)
    if not path.exists() and not path.is_symlink():
        return None
    record = read_record(path)
    if record is None or record.get("operation_id") != identifier:
        raise InstallTransactionError("update operation status is unreadable; ownership is unknown")
    alive = owner_alive(record)
    state = "live" if alive is True else "gone" if alive is False else "unknown"
    return {**record, "owner_state": state, "retirement": _retirement_status(record)}


def register_attempt(
    generation: Path, *, target: str, mode: str, baseline: str, source_identity: str = ""
) -> str:
    """Called while locked; ownership survives the unlocked build phase."""
    nonce = uuid.uuid4().hex
    payload = {
        **_identity(),
        "nonce": nonce,
        "root": str(canonical_root(generation.parent.parent)),
        "candidate": str(generation.resolve()),
        "target": target,
        "source_identity": source_identity,
        "mode": mode,
        "baseline": baseline,
        "phase": "building",
        "started_at": time.time(),
        "deadline": time.time() + 1800,
    }
    path = generation / ATTEMPT_NAME
    staged = path.with_name(f"{ATTEMPT_NAME}.{nonce}.tmp")
    staged.write_text(json.dumps(payload), encoding="utf-8")
    staged.replace(path)
    return nonce


def attempt_alive(record: dict[str, Any]) -> bool | None:
    owner = owner_alive(record)
    if owner is not False:
        return owner
    worker = record.get("worker")
    if isinstance(worker, dict):
        return owner_alive(worker)
    # Hard death between intent and child identity cannot prove absence of a
    # writer. Keep it, rather than age-reclaiming beneath a possibly live uv.
    return None if record.get("worker_pending") else False


def attempt_protected(generation: Path) -> bool:
    path = generation / ATTEMPT_NAME
    if not path.exists():
        return False
    record = read_record(path)
    return record is None or bool(record.get("ever_exposed")) or attempt_alive(record) is not False


def preserve_publication(generation: Path, nonce: str) -> None:
    """Conservatively retain a candidate the instant its pointer swap lands.

    This existing attempt record grants NO lifetime-index completeness. It is
    written immediately AFTER a SUCCESSFUL ``flip_pointer`` return — the atomic
    ``os.rename`` either swapped the pointer or raised before it did, so a
    successful return IS the proof that a runtime can now observe B, and a flip
    that raised (pointer never written) must NOT be marked exposed. Its caller
    (``install_into_generation``) enforces that ordering; this function only
    records the boundary. Failed exposure stays protected; successful completion
    removes the attempt normally; the post-promotion prune
    (``update.reclaim_after_promotion``) still keeps every root a record names,
    the running tree and the keep margin.
    """
    path = generation / ATTEMPT_NAME
    record = read_record(path)
    if record is None or record.get("nonce") != nonce or record.get("pid") != os.getpid():
        raise InstallTransactionError("candidate publication ownership could not be verified")
    record["ever_exposed"] = True
    staged = path.with_name(f"{ATTEMPT_NAME}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(staged, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def finish_attempt(generation: Path, nonce: str) -> None:
    record = read_record(generation / ATTEMPT_NAME)
    if record is None or record.get("nonce") != nonce or record.get("pid") != os.getpid():
        raise InstallTransactionError(
            f"candidate ownership changed for {generation}; nothing was promoted"
        )
    (generation / ATTEMPT_NAME).unlink()


def run_installer(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    executable: str | None = None,
    timeout: float = 1800.0,
    on_output: Callable[[str], None] | None = None,
) -> int:
    """Bound and reap the exact child group, including cancellation and leaks.

    The existing orphan-group ledger provides best-effort recovery if the owner
    is hard-killed; normal returns and all Python exceptions reap synchronously.
    This never signals a discovered runtime or any group we did not spawn.
    """
    from local_operator.tools import group_reaper

    process: subprocess.Popen[bytes] | None = None
    # A file avoids both pipe backpressure and unbounded in-memory capture.
    # Read only the bounded head used by mesh failure receipts, after reaping.
    output = tempfile.TemporaryFile() if on_output is not None else None
    try:
        generation = Path(env["UV_TOOL_DIR"]).parent if env and "UV_TOOL_DIR" in env else None
        if generation is None:
            process = subprocess.Popen(
                argv,
                env=env,
                executable=executable,
                start_new_session=True,
                stdout=output,
                stderr=subprocess.STDOUT if output else None,
            )
        else:
            with transaction(generation.parent.parent, operation="register candidate worker"):
                path = generation / ATTEMPT_NAME
                record = read_record(path)
                if record is None or record.get("pid") != os.getpid():
                    raise InstallTransactionError(
                        "candidate worker has no owned attempt; refusing spawn"
                    )
                record["worker_pending"] = True
                path.write_text(json.dumps(record), encoding="utf-8")
                process = subprocess.Popen(
                    argv,
                    env=env,
                    executable=executable,
                    start_new_session=True,
                    stdout=output,
                    stderr=subprocess.STDOUT if output else None,
                )
                record["worker"] = {
                    "pid": process.pid,
                    "birth_scheme": procstate.birth_scheme(),
                    "birth_token": procstate.birth_token(process.pid),
                }
                path.write_text(json.dumps(record), encoding="utf-8")
        group_reaper.register_group(process.pid, "install candidate")
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise InstallTransactionError(
            f"installer exceeded {timeout:g}s; its owned process group was stopped; "
            "retry the update"
        ) from exc
    finally:
        try:
            if process is not None:
                _reap_installer(process)
        finally:
            if output is not None:
                try:
                    output.seek(0)
                    if on_output is not None:
                        on_output(output.read(65536).decode("utf-8", "replace"))
                finally:
                    output.close()


def _reap_installer(process: subprocess.Popen[bytes]) -> None:
    from local_operator.tools import group_reaper

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # A leader that already returned can still have descendants. Escalation
    # targets its exact spawned PGID even on the successful-exit path.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)
    group_reaper.unregister_group(process.pid)


def status(root: Path) -> dict[str, Any]:
    """Pure observation: stale metadata never means a kernel lock is held.

    Linux exposes flock owners in /proc/locks. macOS lsof may report open
    descriptors WITHOUT flock state (measured with a held lock); a blank lock
    field is unknown, not proof of an unlocked inode. Report unknown,
    not a fabricated owner inferred from a still-live metadata PID.
    """
    root = canonical_root(root)
    path = root / LOCK_NAME
    metadata = read_record(path)
    owners: list[int] | None = None
    try:
        sample = path.lstat()
        if (
            not stat.S_ISREG(sample.st_mode)
            or sample.st_nlink != 1
            or (os.name == "posix" and sample.st_uid != os.getuid())
        ):
            return {"root": str(root), "os_owners": None, "metadata": None, "metadata_alive": None}
    except FileNotFoundError:
        return {"root": str(root), "os_owners": [], "metadata": None, "metadata_alive": None}
    except OSError:
        return {"root": str(root), "os_owners": None, "metadata": metadata, "metadata_alive": None}
    try:
        if sys.platform.startswith("linux"):
            owners = []
            for line in Path("/proc/locks").read_text().splitlines():
                fields = line.split()
                if len(fields) >= 6 and fields[1] == "FLOCK":
                    major, minor, inode = fields[5].split(":")
                    if (int(major, 16), int(minor, 16), int(inode)) == (
                        os.major(sample.st_dev),
                        os.minor(sample.st_dev),
                        sample.st_ino,
                    ):
                        owners.append(int(fields[4]))
        elif sys.platform == "darwin":
            result = subprocess.run(
                ["/usr/sbin/lsof", "-Fpl", str(path)], capture_output=True, text=True, timeout=3
            )
            if result.returncode in (0, 1):
                owners = [] if not result.stdout.strip() else None
                reported: list[int] = []
                pid = 0
                for line in result.stdout.splitlines():
                    if line.startswith("p"):
                        pid = int(line[1:])
                    elif line.startswith("l") and line[1:] in ("W", "w", "R", "r"):
                        reported.append(pid)
                if reported:
                    owners = reported
    except (OSError, ValueError, subprocess.TimeoutExpired):
        owners = None
    return {
        "root": str(root),
        "os_owners": owners,
        "metadata": metadata,
        "metadata_alive": owner_alive(metadata) if metadata else None,
    }

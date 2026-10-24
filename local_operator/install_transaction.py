"""Cooperating install writers share one permanent INSTALLROOT lock inode.

The lock serialises mutations, not preparation, and does not drain the fleet or
exclude old/non-cooperating installers. Callers still owe admission and fresh
promotion checks. Observers never open a lock for writing or try to acquire it.
"""

from __future__ import annotations

import errno
import json
import os
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


def read_record(path: Path) -> dict[str, Any] | None:
    # Observation must not follow an unsafe lock/attempt name into unrelated
    # bytes, nor block on a FIFO. No create/acquire/update is involved here.
    descriptor: int | None = None
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
            or (os.name == "posix" and sample.st_uid != os.getuid())
        ):
            return None
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = None
            value = json.load(stream)
    except (OSError, ValueError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return value if isinstance(value, dict) else None


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
    if len(data) > 64 * 1024:
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


def start_operation(root: Path, *, baseline: str, target: str, route: str) -> UpdateOperation:
    identifier = uuid.uuid4().hex
    with transaction(root, operation="register durable update operation") as held:
        held.validate(root)
        path = _operation_path(root, identifier, create=True)
        # Bound namespace growth without deleting unknown/unfinished operations.
        for index, _path in enumerate(path.parent.iterdir()):
            if index >= 255:
                raise InstallTransactionError(
                    "update recovery records are at capacity; inspect pending operations"
                )
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
    return {**record, "owner_state": state}


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
    removes the attempt normally, while rolling automatic GC remains frozen for
    all loaded roots.
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

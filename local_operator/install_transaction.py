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
from typing import Any, Callable, Iterator

from local_operator import procstate

LOCK_NAME = "install.lock"
ATTEMPT_NAME = ".install-attempt.json"
DEFAULT_TIMEOUT_S = 10.0


class InstallTransactionError(Exception):
    """Actionable refusal shared by CLI, TUI and mesh install writers."""


class LockUnsupported(InstallTransactionError):
    """No validated lock implementation on this platform; fail before mutation."""


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
        raise InstallTransactionError(
            f"another install mutation owns {root}; retry when it settles"
        )
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
                    raise InstallTransactionError(
                        f"cannot lock install root {root}: {exc}"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise InstallTransactionError(
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
    return record is None or attempt_alive(record) is not False


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

"""One connector at a time per machine (account scope), across processes.

WHY A LOCK OF ITS OWN rather than reusing ``wakes/lock.py``'s per-session
lock: this exclusion is machine-wide and account-scoped (contract §13.3: "one
connector per account per machine"), while the wake lock guards one session's
schedule file. The MECHANISM is deliberately identical — non-blocking
``flock`` retried to a deadline, on a worker-free blocking loop — because the
failure this prevents is the same shape: two long-poll connectors would each
park a waiter against the server's ≤4 waiters/account cap and double every
delivery attempt, and a second connector on the same account can even race
the other's claim/ack bookkeeping.

REFUSES RATHER THAN DEGRADES. A timed-out acquire raises
:class:`ConnectorLockBusy`; running unlocked is never the fallback (the
pre-existing behaviour here would be duplicate wakes, not a missing feature).
The fd is opened, locked and closed on one path so no other thread can close
a descriptor this one is calling ``flock`` on.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

from local_operator.procstate import O_BINARY

#: Beside the bridge state database, in the same family as the wake lock's
#: dotted sidecars: a listing that walks the config dir must not mistake it
#: for a user file.
CONNECTOR_LOCK_NAME = "connector.lock"

#: Bounded retry before a refusal, and the poll between attempts. The holder
#: is a long-lived process, so the deadline is a courtesy for a crash-stale
#: holder; flock is released by the kernel on fd close/exit, so a dead
#: connector cannot leave this locked.
DEFAULT_LOCK_DEADLINE_S = 5.0
POLL_S = 0.05


class ConnectorLockBusy(RuntimeError):
    """Another connector already holds this account's machine-wide lock."""


class ConnectorLock:
    """The machine-wide connector singleton for one config dir."""

    def __init__(self, config_dir: Path, *, deadline_s: float = DEFAULT_LOCK_DEADLINE_S):
        self._path = Path(config_dir) / "team_bridge" / CONNECTOR_LOCK_NAME
        self._deadline_s = deadline_s
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> None:
        import fcntl

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, O_BINARY | os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + self._deadline_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return
            except OSError as failure:
                if failure.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    os.close(fd)
                    raise
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise ConnectorLockBusy(
                        "another bridge connector holds this account's machine-wide lock"
                    )
                time.sleep(POLL_S)

    def release(self) -> None:
        import fcntl

        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "ConnectorLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

"""One writer at a time, per session, across processes.

WHY A LOCK FILE OF ITS OWN. The track this PR is on makes an external writer of
a session's wakes ORDINARY rather than exotic: the desktop's create dialog, the
`lop wake create` CLI and a second app can all arm the same conversation while
nobody owns it. The write is read-merge-append — the base is the transcript's
latest snapshot and every append REPLACES the whole list — so two writers that
interleave their reads both see the same base, and the loser's row is either
dropped or, worse, appended twice: measured on the route, five concurrent arms
left SIX rows and five 200s, one request's row written at two ids, and three
rounds answered 409 for a write that had in fact landed.

THE LEASE IS NOT THIS LOCK, and that distinction is the whole design. A session
already has `.execution-lease`, held by whichever RUNTIME process owns the
transcript, and arbitration there is between "a runtime may host this session"
and "nobody may" — it is what the supervisor and the attach paths read, and it
carries a pid and a generation that other surfaces draw conclusions from. An
HTTP request taking it would refuse every attach and every engage for the
duration of a write, and would put this server's pid in `.session.pid` while it
held it. This lock is narrower and purely mutual: it excludes the OTHER external
writers of one session's schedules, lives beside those sidecars, and is released
with the request.

WHY THE RUNTIME DOES NOT TAKE IT, stated because the omission is deliberate. A
lock shared with `Session._persist_wake_schedules` would serialise the appends
and still lose the row: the owner's in-memory list was loaded BEFORE our append,
so its next persist writes that stale list over ours no matter who went first.
The mechanism that saves the row is refusing to write behind an owner at all
(:func:`local_operator.wakes.arm._refuse_if_owned`), which is a guard, not a
lock; adding the owner to this mutex would buy nothing and would couple an HTTP
write to the runtime's own write path.

BOUNDED, NON-BLOCKING, AND IT REFUSES RATHER THAN DEGRADING. Every attempt is
`LOCK_NB` and retried to a deadline, so a waiter never parks a worker thread in
the kernel — on macOS/BSD a thread inside `flock()` blocks a sibling's
`os.close()` of the same fd until that call returns, which is the shape of
#401 and of the group reaper's `_ledger_lock`. But where that lock DEGRADES to
running unlocked (its worst case is the pre-existing behaviour), this one must
NOT: it guards correctness, and running unlocked is the duplicate row. So a
timed-out acquire raises :class:`WakeLockBusy` and the caller refuses with a
retryable answer. The wait is bounded and off the event loop, and the holder
cannot block forever (it is appending to a file, not waiting on a peer or a
provider), so unlike the group reaper's bound this one is not on a process-exit
path and there is no hang for a long wait to relocate.

The fd is opened, retried on, and closed on the SAME worker thread, so no other
thread can close a descriptor this one is inside a lock call on.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

#: Beside the transcript, in the family the per-session sidecars already use
#: (``.execution-lease``, ``.session.pid``). Dotted so a listing that walks a
#: session directory never reads it as a session.
WAKE_LOCK_NAME = ".wake-write.lock"

#: How long an external writer waits for a peer before refusing. Generous on
#: purpose: the write it serialises is a read plus an append, which is
#: sub-millisecond on an ordinary transcript but MEASURED at ~0.5 s on a 25 MB
#: one, and five concurrent arms on a large transcript therefore legitimately
#: queue for seconds. Refusing early would turn contention into a false
#: failure; refusing at all is what keeps a duplicated row impossible.
LOCK_WAIT_S = 15.0
_LOCK_RETRY_SLEEP_S = 0.02
_LOCK_RETRY_SLEEP_MAX_S = 0.1


class WakeLockBusy(RuntimeError):
    """The lock was held by a peer for the whole of :data:`LOCK_WAIT_S`."""


def _try_lock(fd: int) -> bool:
    """One NON-BLOCKING exclusive attempt, on every platform that ships this.

    The Windows branch mirrors ``teams._try_lock_exclusive`` (``msvcrt`` has no
    ``fcntl`` and locks a byte range), including its "a zero-length file cannot
    be locked" bootstrap byte; the POSIX branch is the one every other lock in
    this tree uses.
    """
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


class WakeWriteLock:
    """An exclusive, cross-process mutex on one session's wake write path.

    Held ACROSS awaits (the transcript append is async), so it is acquired and
    released on worker threads while the event loop stays free::

        lock = WakeWriteLock(session_dir)
        await asyncio.to_thread(lock.acquire)
        try:
            ...
        finally:
            await asyncio.to_thread(lock.release)

    It is also usable as a plain synchronous context manager, which is how a
    caller with no event loop holds it — and how the tests drive two holders at
    once. ``release`` is idempotent, so a `finally` after a failed acquire is
    safe.

    The lock file is never unlinked, by anyone, while it may be contended: the
    only thing that removes it is the create+arm rollback's `rmtree` of the
    directory it lives in, and that path runs with no peer able to reach the
    same session (``arm._apply`` refuses a session directory that is not
    there, so a writer cannot target a path that has just gone). That is why
    this needs no answer to the "pathname replaced under a waiter" race the
    lease's own recovery lock documents.
    """

    def __init__(self, session_dir: Path, *, timeout_s: float = LOCK_WAIT_S) -> None:
        self.path = Path(session_dir) / WAKE_LOCK_NAME
        self.timeout_s = timeout_s
        self._fd: int | None = None

    def acquire(self) -> None:
        """Take the lock, or raise :class:`WakeLockBusy` at the deadline."""
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + self.timeout_s
        sleep_s = _LOCK_RETRY_SLEEP_S
        try:
            while True:
                if _try_lock(fd):
                    self._fd = fd
                    return
                if time.monotonic() >= deadline:
                    raise WakeLockBusy(
                        "Another writer is applying a change to this conversation's wakes. "
                        "Retry in a moment."
                    )
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 1.5, _LOCK_RETRY_SLEEP_MAX_S)
        except BaseException:
            # Closed HERE rather than by the caller: the fd belongs to this
            # call on every path that does not leave the lock held, and letting
            # it escape would leak one descriptor per contended write.
            os.close(fd)
            raise

    def release(self) -> None:
        """Release the lock. Idempotent, and safe to call when never held."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            # The close below is what matters: an unlock that fails because the
            # descriptor is already gone still drops the lock with the process's
            # last reference to it.
            pass
        os.close(fd)

    def __enter__(self) -> "WakeWriteLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

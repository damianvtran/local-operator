"""The cold-session inbox: messages that arrive when nothing is running.

A quiet peer note (``lop send --no-wake``, the ``send`` tool with
``wake=False``) to a session with no runtime has nowhere to go. Spawning a
whole runtime to receive it would contradict what "quiet" means — the point of
``wake=False`` is that the peer reads it on its NEXT turn, not that it starts
one — and dropping it would lose the message. So it is appended here, and the
runtime drains it the moment one exists.

**The ordering guarantee, and where it comes from.** ``process.py`` drains this
file after the session is constructed and BEFORE ``RuntimeServer`` begins
listening. That ordering *is* the guarantee: rows spooled while the session was
cold are delivered ahead of anything a socket client could send, because no
socket client can send anything yet. The alternative — draining after the
server starts — would race an errand that arrived in the same instant and
deliver messages out of the order they were written.

**Never a blocking ``flock``.** Every lock here is ``LOCK_NB`` with a bounded
retry. This is the #401 class: a blocking ``flock`` in the MCP OAuth refresh
lock deadlocked the Textual event loop, and on macOS/BSD a sibling ``flock``
even makes ``close()`` block. The appending side of this file runs from a
short-lived sender process, but ``peek_inbox`` is read by the VIEWER while it
paints, so a blocking lock here would freeze a terminal because an unrelated
process was mid-append. Contention is genuinely rare (two peers writing to the
same cold session in the same millisecond), and the correct response to losing
the race is to retry briefly and then give up — never to wait indefinitely.

**Not a sidecar.** ``inbox.jsonl`` is deliberately absent from
``retention._SIDECAR_NAMES``, so a directory holding one reads as having
CONTENT and the junk reap will not delete it. A spooled message the user has
not seen is exactly the thing that must survive a sweep.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The spool file inside ``sessions/<id>/``.
INBOX_NAME = "inbox.jsonl"

#: Non-blocking lock retries, and the pause between them. Deliberately small:
#: the critical section is one ``write()`` of a few hundred bytes, so a
#: contender that cannot get in within ~50 ms is not merely slow, and waiting
#: longer trades a caller's responsiveness for a case that barely happens.
_LOCK_ATTEMPTS = 10
_LOCK_RETRY_S = 0.005

#: Refuse to spool past this many rows for one session. A cold session that
#: something is hammering must not grow an unbounded file that then has to be
#: replayed into a transcript in one go. Well above any legitimate use (a
#: handful of notes between sessions) and far below "this file is a problem".
MAX_INBOX_ROWS = 500


@dataclass(frozen=True, slots=True)
class InboxLine:
    """One spooled message, in the order it was written."""

    text: str
    sender: dict[str, Any]
    mode: str = "mailbox"
    written_at: float = 0.0

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "InboxLine":
        sender = payload.get("sender")
        return cls(
            text=str(payload.get("text", "") or ""),
            sender=dict(sender) if isinstance(sender, dict) else {},
            mode=str(payload.get("mode", "mailbox") or "mailbox"),
            written_at=float(payload.get("written_at", 0.0) or 0.0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sender": self.sender,
            "mode": self.mode,
            "written_at": self.written_at,
        }


def inbox_path(session_dir: Path) -> Path:
    return session_dir / INBOX_NAME


class _NonBlockingLock:
    """``LOCK_EX | LOCK_NB`` with a bounded retry, or nothing at all.

    Failing to acquire is NOT an error: the append below is ``O_APPEND`` on a
    line-sized write, which the kernel already keeps atomic on every platform
    this runs on. The lock is what protects the read-then-rewrite in
    :func:`drain_inbox`, and for the writer it is belt-and-braces. So a
    contended writer proceeds unlocked rather than blocking a caller — the
    opposite trade from a correctness lock, and deliberate.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self.acquired = False

    def __enter__(self) -> "_NonBlockingLock":
        if os.name == "nt":
            # No fcntl on Windows, and msvcrt.locking's non-blocking mode
            # raises rather than waiting. The atomic O_APPEND write stands on
            # its own there.
            return self
        import fcntl

        for attempt in range(_LOCK_ATTEMPTS):
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = True
                return self
            except OSError:
                if attempt == _LOCK_ATTEMPTS - 1:
                    break
                time.sleep(_LOCK_RETRY_S)
        logger.debug("inbox lock contended; proceeding on the atomic append")
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self.acquired or os.name == "nt":
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass


def append_inbox(session_dir: Path, line: InboxLine) -> bool:
    """Spool one message for a session that is not running. True if written.

    ``O_APPEND`` so concurrent writers interleave whole lines rather than
    overwriting each other at a shared offset, and one ``write()`` per row so
    the atomicity that guarantees applies to the entire line.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    path = inbox_path(session_dir)
    payload = json.dumps(line.to_json(), separators=(",", ":")).encode() + b"\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    except OSError:
        logger.warning("could not open inbox for %s", session_dir.name, exc_info=True)
        return False
    try:
        with _NonBlockingLock(fd):
            # Checked under the lock when we hold it: the bound exists to stop
            # a runaway producer, and an unlocked contender overshooting it by
            # a row or two is harmless.
            if _count_rows(path) >= MAX_INBOX_ROWS:
                logger.warning(
                    "inbox for %s is at its %d-row cap; message dropped",
                    session_dir.name,
                    MAX_INBOX_ROWS,
                )
                return False
            os.write(fd, payload)
        return True
    except OSError:
        logger.warning("inbox append failed for %s", session_dir.name, exc_info=True)
        return False
    finally:
        os.close(fd)


def _count_rows(path: Path) -> int:
    try:
        with open(path, "rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _parse(raw: bytes) -> list[InboxLine]:
    lines: list[InboxLine] = []
    for row in raw.splitlines():
        if not row.strip():
            continue
        try:
            payload = json.loads(row.decode("utf-8", "replace"))
        except ValueError:
            continue  # a torn line is skipped, never fatal
        if isinstance(payload, dict):
            lines.append(InboxLine.from_json(payload))
    return lines


def peek_inbox(session_dir: Path) -> list[InboxLine]:
    """Read the spool WITHOUT consuming it — the cold viewer's read.

    A viewer showing a session that is not running renders these as pending
    messages; the runtime is what actually delivers them. Deliberately
    lock-free: a reader that saw a half-written final line simply drops it
    (``_parse`` skips unparseable rows), which is cheaper and safer than
    taking a lock on a UI path.
    """
    try:
        return _parse(inbox_path(session_dir).read_bytes())
    except OSError:
        return []


def drain_inbox(session_dir: Path) -> list[InboxLine]:
    """Consume every spooled message, in write order. Called once, at open.

    Read and removal happen under one non-blocking lock so a concurrent
    appender cannot have its row consumed-but-not-delivered. Rows that arrive
    while the caller is delivering are NOT lost: the file is emptied here, and
    a later append creates it again for the next open.

    **The crash contract is at-least-once, not exactly-once.** The file is
    truncated after it is read, so a runtime killed between the read and its
    ``receive_peer_message`` calls loses the messages; a runtime killed between
    reading and truncating re-delivers them on the next open. The second is the
    safe direction and is the one this ordering chooses — a duplicated note is
    visible and harmless, a dropped one is neither.
    """
    path = inbox_path(session_dir)
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("could not open inbox for %s", session_dir.name, exc_info=True)
        return []
    try:
        with _NonBlockingLock(fd) as lock:
            raw = _read_all(fd)
            lines = _parse(raw)
            if not lines:
                return []
            if lock.acquired or os.name == "nt":
                os.ftruncate(fd, 0)
            else:
                # Unlocked, a truncate could discard a row an appender wrote
                # between the read and here. Stage the remainder instead: the
                # rename is atomic, and the worst case is that a racing
                # appender's row lands in a file we just replaced, which the
                # NEXT open drains.
                _replace_remainder(path, raw)
            return lines
    except OSError:
        logger.warning("inbox drain failed for %s", session_dir.name, exc_info=True)
        return []
    finally:
        os.close(fd)


def _read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _replace_remainder(path: Path, consumed: bytes) -> None:
    """Rewrite the spool with only the bytes written after ``consumed``.

    Staged through ``os.replace`` so a reader never sees a truncated file.
    """
    staged = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        current = path.read_bytes()
    except OSError:
        current = consumed
    remainder = current[len(consumed) :] if current.startswith(consumed) else b""
    try:
        staged.write_bytes(remainder)
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    except OSError:
        logger.warning("inbox rewrite failed for %s", path.parent.name, exc_info=True)
        try:
            staged.unlink()
        except OSError:
            pass

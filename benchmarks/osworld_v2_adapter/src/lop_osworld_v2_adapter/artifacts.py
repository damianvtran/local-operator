"""Atomic publish for the adapter's content-addressed artifacts.

The artifact root is a flat directory whose FILE NAMES ARE THE SHA-256 OF
THEIR CONTENTS. The parent reopens ``<root>/<sha256>`` with ``O_NOFOLLOW``,
re-hashes the bytes and compares them against the size and digest the event
declared (``evaluation.adapters.supervisor.verify_artifact``), so the name and
the bytes are meant to be ONE fact. The adapter has only three publishes and
both of the byte-bearing ones (``observation``'s frame, ``scoring``'s detail)
go through here, because that "name implies bytes" contract is a property of
the write and not of its caller.

A create-then-write cannot honour it: the name exists from the instant of the
create, while the bytes are still on their way. The interrupting failure is
unremarkable -- the disk fills between the two, the process is killed, the
write raises ``EIO`` -- and its effect is permanent, because every later
attempt then finds the name already present and skips its own write.

That is not a hypothetical. A live episode (88 steps of real work, preserved
at ``runs/batch-deepseek-flash-canary9/task_012`` of the OSWorld worktree) died
fatally on exactly this: its publish raised ``OSError: [Errno 28] No space left
on device``, the create had already landed a 0-byte file under the digest name,
the retry skipped the write and declared the frame's true ``byte_count`` against
those 0 bytes, and the parent refused it as "artifact is not a matching regular
file" -- a corrupted-bundle refusal, i.e. ``retryable: false``. One transient
disk-full moment permanently poisoned a content address and killed the episode
that held it, and no retry could ever clear it.

So a publish here is indivisible: the bytes go to a temporary name in the same
directory and are ``os.replace``d onto the digest name, which is a single atomic
directory operation. A concurrent reader sees the previous complete file or this
complete one, never a partial one, and a failure before the replace leaves the
content address exactly as it found it.

This module changes nothing about what the harness VERIFIES -- the parent's
``verify_artifact`` remains the sole authority on whether published bytes are
acceptable, and a byte-count mismatch must stay as fatal as it is. All this
removes is a way for a transient write failure to become a permanent one.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

#: How an in-flight publish names its temporary file: ``.<digest>.<token>.tmp``,
#: the shape ``evaluation.evidence.store`` already uses for the same job.
#:
#: It has to be impossible for one to be mistaken for a published artifact, and
#: ``mkstemp``'s random token does not guarantee that on its own -- for a
#: 64-character digest name it leaves a name that could still be a digest. The
#: LEADING DOT is what makes it unambiguous: a valid content address cannot
#: begin with one, so the parent can never read a temp file as an artifact and a
#: human listing the directory can never mistake one for a stuck publish.
#: Carrying the digest is for that human: a leftover says which address it was
#: publishing when it died.
_TEMP_PREFIX_FORMAT = ".{name}."
_TEMP_SUFFIX = ".tmp"


def publish(path: Path, data: bytes) -> None:
    """Publish ``data`` under its content address ``path``, indivisibly.

    ``path.parent`` must already exist and be the parent the reader will open:
    the temporary name is created beside the destination because ``os.replace``
    is atomic only within one filesystem, and a cross-device rename would itself
    become a copy of the bytes we are trying not to expose half-written. (The
    parent creates the root with ``mkdir(parents=True)`` before the worker is
    ever handed it, so "already exists" is a contract, not an assumption.)

    The file mode is ``mkstemp``'s ``0o600``, matching both the parent's
    ``0o700`` artifact root -- which no other user can enter, so nothing wider
    would be readable anyway -- and the evidence store's own publishes.
    ``mkstemp`` rather than a hand-built name because it IS the stdlib's
    O_EXCL-unique-name primitive, collision retry and mode included; the store's
    ``.{name}.{token}.tmp`` spelling is reproduced by the prefix and suffix here
    so a leftover reads the same way, without reimplementing the uniqueness.

    ``fsync`` before the replace, then no directory ``fsync``: the file's own
    flush is what orders "bytes durable" before "name visible", which is the
    only ordering this contract needs (the crash that omits the directory
    entry simply leaves no name, and the one that omits the file leaves a
    temporary name that is not an address). Directory durability is the
    evidence store's concern, not a worker-staged artifact's -- the bundle it
    writes is what has to survive a crash.
    """

    if _already_published(path, len(data)):
        return
    fd, temporary = tempfile.mkstemp(
        prefix=_TEMP_PREFIX_FORMAT.format(name=path.name),
        suffix=_TEMP_SUFFIX,
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        # A failed publish must leave NOTHING behind: not a temp file, and above
        # all not a name at the content address. Best-effort because the failure
        # that brought us here is usually a full disk or a full inode table,
        # where the cleanup can fail too -- the original error is the one worth
        # propagating, so it is never replaced by a cleanup error.
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _already_published(path: Path, size: int) -> bool:
    """Whether ``path`` already holds a complete copy of a ``size``-byte payload.

    Skipping an existing name is sound ONLY because :func:`publish` is atomic:
    the name appears in one indivisible step with its bytes already flushed
    behind it, so a file at a content address is a finished publish rather than
    a create whose write never landed. Where the older non-atomic publish left
    the name behind without the bytes, this stops treating it as published.

    The size comparison is not a substitute for the digest -- the parent
    re-hashes every artifact and is the authority on content, and hashing here
    would read every published frame back for nothing. It is the one cheap
    signal that separates a complete predecessor from a truncated one, and it
    is what lets a poisoned address HEAL: a 0-byte or short leftover has the
    wrong size, so the next publish of those bytes rewrites it instead of
    trusting it forever. A same-size mismatch cannot come from an interrupted
    write -- landing every byte is what completing means -- and if it came from
    corruption instead, the parent's digest check is the thing that refuses it.

    ``follow_symlinks=False`` because a symlink is not something this adapter
    published; replacing it with the real file is the correct repair, and the
    parent refuses to follow one anyway (``O_NOFOLLOW``). Any ``OSError``
    (absent, dangling, unreadable) reports "not published" so the write is
    attempted and the real error surfaces from the write itself.
    """

    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size == size

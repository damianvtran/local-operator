"""Transcript compression: shrink the one bulk file a session owns, and nothing else.

WHY THIS MODULE EXISTS
======================

A session directory is a handful of small sidecars beside one big file. On the
operator's store the 48.6 GB under ``sessions/`` held 7.65 GB of
``transcript.jsonl`` — and that total only grows: every turn appends, and
nothing ever rewrites a transcript. Retention trims *sessions* (and the
delegated content pass trims *scratchpads*), so the remaining unbounded term is
the transcript itself. This module bounds that term by storing old transcripts
compressed, and by making every read path resolve the compressed form
transparently.

WHAT IT IS NOT: it does not delete, age out or rewrite a session's sidecars.
``title.json``, ``origin.json``, ``created_at.json``, ``attachment.json``, the
subagent roster, ``inbox.jsonl``, the tail anchor, ``scratchpad/`` and
``attachments/`` are never compressed and never touched — listing, the
``/resume`` picker, titles, mail and every attachment keep reading exactly the
files they read before. Only ``transcript.jsonl`` is a candidate, and the
original is removed only after a byte-for-byte verified replacement exists.

THE CODEC DECISION, WITH THE NUMBERS
====================================

Primary: stdlib :mod:`compression.zstd` (PEP 784, Python 3.14+), level 3.
Fallback: :mod:`gzip`, level 6, when that module is absent (the pyproject floor
is 3.12). No third-party dependency — ``zstandard`` and ``pyzstd`` are not
options for a stdlib-first harness.

Measured on real transcripts from this store (444-629 MB/s compress):

=================  =========  ======================  =======================
codec              ratio      compress throughput     decompress throughput
=================  =========  ======================  =======================
zstd level 3       2.3-4.7x   ~444-629 MB/s           ~1.4 GB/s
gzip level 6       2.1-3.3x   10-14x slower           proportionally slower
=================  =========  ======================  =======================

At 3.7x the 7.65 GB of live transcripts becomes ~2 GB once the corpus ages past
the threshold. The level is a module constant (:data:`ZSTD_LEVEL`,
:data:`GZIP_LEVEL`) rather than a setting: compression ratio is not a decision
a user should have to make, and a per-store level would fork the archive format.

Archives SELF-DESCRIBE by suffix — ``transcript.jsonl.zst`` or
``transcript.jsonl.gz`` — and the suffix is the only thing that decides how a
file is read (:func:`transcript_source`). An existing archive is NEVER rewritten
into another codec: a store can therefore hold both forms side by side, written
by 3.14 and 3.12 runtimes, and each is decoded by whatever can decode it.

THE CROSS-VERSION CONSEQUENCE, STATED PLAINLY
=============================================

A 3.12/3.13 runtime cannot decode a ``.zst`` archive written by a 3.14 runtime.
There is no stdlib zstd before 3.14 and this module will not grow a dependency
for it, so on such a build a ``.zst`` transcript is READ-ONLY unavailable: the
readers raise :class:`ArchiveUnavailable` naming the codec, they never report an
empty transcript, and :func:`has_transcript` still answers ``True`` so listing,
the picker and stored-session search still find the row. Such a build also never
*writes* ``.zst`` (it compresses with gzip), so the situation can only arise from
a newer runtime's archives — and the production ``lop`` runtime here is 3.14.3.

THE SAFETY ORDERING (the whole point of this module)
====================================================

The original transcript is deleted only after a replacement that was decoded
back and hashed to the SAME sha256 is durably on disk under its final name.
Exactly, in this order — :func:`compress_session` is the transcript of it:

1. acquire the per-session writer lock, non-blocking; a held lock means "session
   in use" and the session is skipped untouched;
2. stat the source into ``sig0 = (st_dev, st_ino, st_size, st_mtime_ns)``;
3. stream source -> ``.transcript.jsonl.<pid>.<rand>.tmp`` in the SAME
   directory, created ``O_EXCL`` (a pre-existing tmp-looking file is never
   adopted, and a same-directory tmp is what makes the publish atomic), hashing
   the original bytes on the way through;
4. decode the tmp in-process and hash the result; a mismatch or a decode error
   unlinks the tmp, keeps the original, logs at ERROR and counts a failure;
5. fsync the tmp and set its mtime to the SOURCE's (the archive must carry the
   activity clock — see below);
6. re-stat the source; anything other than ``sig0`` abandons the attempt (the
   transcript changed under us) and keeps the original;
7. ``os.replace(tmp, archive)`` — the only publish, so a half-written archive is
   never observable under the archive name;
8. RENAME the source to ``.transcript.jsonl.quarantine`` in the same directory,
   then stat the RENAMED file. Unchanged -> unlink the quarantine. Changed ->
   rename it back over the plain name and unlink the archive just published. A
   writer holding an fd opened before the rename appends into the quarantined
   inode, which is exactly what that stat sees; the old order (re-stat, then
   unlink) deleted a row that landed in the window between the two calls;
9. fsync the directory, rewrite the search index's physical signature for this
   session (no re-digest), and drop the now-stale tail anchor;
10. release the lock.

Invariants the tests pin: never unlink before a verified archive exists; a
verification failure leaves the original byte-identical and no ``.tmp`` behind;
a source that grows is never deleted; the archive is never observable partially
written.

A crash between the quarantine rename and its unlink leaves the plain file under
the quarantine name. Nothing a reader looks at is missing (readers only ever ask
for the two known transcript names), and :func:`_recover_quarantine` puts it back
from three places: :func:`restore_for_write`, the start of a compression attempt,
and the pass's scan. The name is FIXED rather than pid-unique for exactly that
reason: recovery must find it with one stat, and the per-session writer lock
already guarantees at most one compressor.

THE LISTING SIDECAR
===================

A list row asks two questions of the transcript tail — the last assistant reply
and the newest failure — and asking them of an ARCHIVE means decompressing it
end to end, once per row per process. So compression writes ``preview.json``
(schema 1) beside the archive, holding exactly those two condensed strings, from
a bounded tail read of the plain file before it is retired. Readers stat it and
read ~400 B; only an archive from a build that predates the sidecar falls back
to the tail read, once. ``restore_for_write`` deletes it, because the plain file
is authoritative again; the next pass recreates it.

THE ACTIVITY CLOCK MUST NOT MOVE
================================

``retention.session_activity`` is the one ranking clock for ``/resume``, the
sidebar catalog and the cleanup policy, and it reads the mtime of a small fixed
set of names (``_ACTIVITY_FILES``). Compression therefore (a) preserves the
source transcript's mtime on the archive and (b) adds the archive names to that
set. ``/resume`` order is byte-identical before and after a pass; a compressed
session does not look "touched now" and does not jump to the top of the picker.

READING: STREAMING, BOUNDED, NO WRITES
======================================

:func:`read_head` / :func:`read_tail` / :func:`iter_lines` stream the source
through a decompressor with a fixed byte budget, so a listing scan never
materialises a 300 MB transcript. :func:`transcript_for_read` is the one
resolver for callers that need a real seekable path (page readers, forks): plain
file if present, else the archive decompressed into a bounded cache
(``<config>/cache/decompressed/<id>/``), and ``None`` only when there is
neither. The cache is validated against the archive's stat, LRU-evicted at
:data:`CACHE_MAX_BYTES` / :data:`CACHE_MAX_ENTRIES`, and skipped entirely when
free disk space is tight — in which case the archive is restored in place
instead. A read never fails because the cache could not be written.

THE WRITER LOCK
===============

``<session>/.transcript.lock``, ``flock(LOCK_EX|LOCK_NB)`` on POSIX and
``msvcrt.locking`` on Windows, a no-op where neither exists. The compressor
takes it non-blocking and skips when busy. A :class:`~local_operator.session.
transcript.Transcript` (stage B) takes it for the instance's whole lifetime —
the exact statement "a runtime has this session open" — so compress-then-append
cannot interleave with a live writer. Process death releases ``flock``, so the
lock is crash-safe by construction; :func:`acquire_writer` keeps an in-process
refcount registry so two ``Transcript`` objects in one process do not steal or
release each other's lock.

IMPORT DISCIPLINE
=================

Module-level imports are stdlib ONLY, so this module can be imported (and its
reader helpers exercised) standalone under a 3.14 interpreter. Everything that
belongs to the harness — ``cleanup`` for the hard guards, ``retention`` for the
clock, ``delegated_retention`` for the keep rules, ``search_index`` for the
signature rewrite — is imported lazily inside the functions that need it. That
also keeps the import graph from growing an edge cycle through ``retention``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import secrets
import stat as _stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

try:  # PEP 784 — Python 3.14+. The only codec import that may be absent.
    # The 3.12/3.13 typeshed the repo type-checks against has no ``compression.zstd``
    # symbol; the branch is real and CI's 3.14 leg executes it.
    from compression import zstd as _zstd  # pyright: ignore[reportAttributeAccessIssue]
except ImportError:  # 3.12/3.13: gzip is the working codec.
    _zstd = None

try:  # POSIX advisory locks; absent on Windows.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    _fcntl = None

try:  # Windows advisory locks; absent on POSIX.
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX only
    _msvcrt = None


logger = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_FILENAMES",
    "CACHE_MAX_BYTES",
    "CACHE_MAX_ENTRIES",
    "COMPRESSION_LOG_NAME",
    "COMPRESSION_PATH",
    "CompressionPolicy",
    "CompressionResult",
    "DEFAULT_AFTER_DAYS",
    "DEFAULT_BUDGET_S",
    "DEFAULT_ENABLED",
    "LAST_COMPRESSION_NAME",
    "LOCK_NAME",
    "MAX_AFTER_DAYS",
    "MIN_AFTER_DAYS",
    "Outcome",
    "PLAIN_FILENAME",
    "PREVIEW_SIDECAR_NAME",
    "PREVIEW_TAIL_BYTES",
    "QUARANTINE_NAME",
    "WRITER_LOCK_WAIT_S",
    "WRITER_WAIT_POLL_S",
    "acquire_writer",
    "archive_path",
    "available_codec",
    "compress_from_config",
    "compress_session",
    "has_transcript",
    "iter_lines",
    "plain_path",
    "policy_from_config",
    "read_head",
    "read_listing_preview",
    "read_tail",
    "read_tail_window",
    "release_writer",
    "transcript_state",
    "restore_for_write",
    "run_compression",
    "transcript_for_read",
    "transcript_source",
    "unreadable_archive_error",
    "write_last_compression",
]


class ArchiveError(Exception):
    """A transcript archive could not be read.

    Deliberately NOT caught by the readers themselves: a listing that silently
    turned an undecodable archive into an empty transcript would hide the
    session's whole body while still showing its row (see
    ``tests/unit/session/test_compression.py``). Callers that enumerate a store
    catch this and render the row with a degraded preview.
    """


class ArchiveUnavailable(ArchiveError):
    """The archive exists but THIS build has no decoder for its codec.

    The cross-version case documented in the module docstring: a ``.zst``
    archive read by a 3.12/3.13 runtime. Distinct from a corrupt archive so the
    message can say which build could read it instead of implying data loss.
    """


class ArchiveCorrupt(ArchiveError):
    """The archive exists but did not decode."""


# ---------------------------------------------------------------------------
# Names, defaults, bounds
# ---------------------------------------------------------------------------

#: The one bulk file this module may compress. Kept in step with
#: ``retention.TRANSCRIPT_FILENAME`` by a test rather than imported at module
#: level: this module must import standalone, and ``retention`` is a lazy import
#: everywhere else here.
PLAIN_FILENAME = "transcript.jsonl"

#: Archive suffixes, keyed by codec. The suffix IS the format statement — there
#: is no sidecar and no header this module writes (zstd and gzip both carry
#: their own framing), so a reader that finds ``transcript.jsonl.zst`` knows
#: exactly what to do with it.
ARCHIVE_SUFFIXES: dict[str, str] = {"zstd": ".zst", "gzip": ".gz"}

#: The archive names, in preference order. Present so a single place can tell
#: ``retention._ACTIVITY_FILES`` (and anything else enumerating sidecar names)
#: what a compressed transcript is called.
ARCHIVE_FILENAMES: tuple[str, ...] = tuple(
    PLAIN_FILENAME + suffix for suffix in ARCHIVE_SUFFIXES.values()
)

#: The per-session writer lock (see the module docstring's lock protocol).
LOCK_NAME = ".transcript.lock"

#: Where the plain transcript is parked while compression decides whether it may
#: be retired. Dot-prefixed so a directory listing reads it as bookkeeping, and
#: FIXED rather than pid-unique because crash recovery has to find it with one
#: stat: the per-session writer lock already guarantees at most one renamer, so
#: uniqueness would buy a glob per session directory and nothing else. Readers
#: never look at this name — only :func:`_recover_quarantine` and the protocol
#: do.
QUARANTINE_NAME = f".{PLAIN_FILENAME}.quarantine"

#: The listing sidecar written beside an archive at compression time, and its
#: schema. Holding one condensed assistant line and one failure line, it is what
#: keeps a cold listing from decompressing every archive to paint its rows.
PREVIEW_SIDECAR_NAME = "preview.json"
PREVIEW_SIDECAR_SCHEMA = 1

#: Bytes of the plain tail read at compression time to build that sidecar.
#: Equal to ``resume.PREVIEW_SCAN_BYTES`` by construction — a test pins the two
#: together, because this module must import standalone (no harness import at
#: module scope) while the reader that consumes the value lives in ``resume``.
PREVIEW_TAIL_BYTES = 64_000

#: A sidecar read is bounded: the document holds two strings capped at
#: ``resume.PREVIEW_MAX_CHARS`` each. Anything larger is not ours.
PREVIEW_SIDECAR_MAX_BYTES = 4096

#: Append-only record of every compression that landed, inside the store it
#: describes: ``{"at", "session", "codec", "bytes_before", "bytes_after",
#: "actor", "pid"}``. Mirrors ``cleanup.CLEANUP_LOG_NAME``; dry runs are not
#: recorded. Unlike the cleanup log this is a RECEIPT, not a gate: nothing is
#: removed unverified, so a write failure costs an audit row and never a file.
COMPRESSION_LOG_NAME = ".compression-log.jsonl"

#: The last pass that actually compressed something, beside the log. Written
#: only when ``result.compressed`` is non-empty (a pass that did nothing is not
#: news).
LAST_COMPRESSION_NAME = "last-compression.json"

#: Where the policy lives. MUST be the same tuple ``settings_io`` writes —
#: the historical bug this pattern exists to prevent is a switch written under
#: one path and read under another (see ``cleanup.CLEANUP_PATH``).
COMPRESSION_PATH: tuple[str, ...] = ("session", "compression")

DEFAULT_ENABLED = True
DEFAULT_AFTER_DAYS = 30
MIN_AFTER_DAYS = 1
MAX_AFTER_DAYS = 3650

#: Codec levels. See the module docstring's measured table.
ZSTD_LEVEL = 3
GZIP_LEVEL = 6

#: The startup pass's wall-clock budget. The store maintenance sequence runs
#: after the first frame and must not hold a launch hostage: this pass stops
#: between sessions when the budget is spent, and the next launch (or the hourly
#: sweep in a long-lived runtime) continues. A CLI caller passes ``float("inf")``
#: or a larger budget — a person who typed a command waits for it.
DEFAULT_BUDGET_S = 20.0

#: Streaming chunk for compression, verification and tail reads. Bounded memory
#: is a property of the readers, not an accident of small test fixtures.
CHUNK = 1024 * 1024

#: The decompressed cache's bounds (bytes of decompressed transcript, entries).
#: A cache that outgrows either is trimmed least-recently-used first.
CACHE_MAX_BYTES = 256 * 1024 * 1024
CACHE_MAX_ENTRIES = 64

#: Free space that must remain on the cache's volume after materialising an
#: archive into it. Under that, the archive is restored in place instead (which
#: is also how a read succeeds on a store whose cache volume is full).
CACHE_FREE_RESERVE_BYTES = 512 * 1024 * 1024

#: How much bigger than its archive a cache copy is budgeted to be. The gate in
#: :func:`_materialise` must reserve room for the bytes it WRITES, which are the
#: DECOMPRESSED ones; the archive's size would under-book by the compression
#: ratio. The measured corpus expands ~3.7x, so 5x leaves headroom for the tail
#: of the distribution. Over-budget answers fall back to the documented in-place
#: restore; they never fail a read.
CACHE_EXPANSION_ESTIMATE = 5

#: A session's cold-read hint; invalidated when the transcript's physical form
#: changes (the anchor stores byte offsets and sizes of the PLAIN file).
TAIL_ANCHOR_NAME = "tail-anchor.v1.json"

#: How often the cache meta's ``last_used`` is rewritten on a reuse. Every
#: materialise would otherwise be a write on a read path; a stale-by-minutes
#: LRU order is not worth that.
CACHE_TOUCH_INTERVAL_S = 300.0


# ---------------------------------------------------------------------------
# Codec availability and file names
# ---------------------------------------------------------------------------


def available_codec() -> str:
    """The codec this build writes: ``"zstd"`` on 3.14+, else ``"gzip"``.

    Reading is NEVER routed through this: a reader dispatches on the archive's
    own suffix (:func:`_codec_for`), so a gzip-only build still decodes the gzip
    archives it shares with a zstd build, and reports a zstd archive as
    :class:`ArchiveUnavailable` rather than as absent.
    """
    return "zstd" if _zstd is not None else "gzip"


def plain_path(directory: Path) -> Path:
    """The plain transcript path, whether or not it exists."""
    return Path(directory) / PLAIN_FILENAME


def _codec_for(path: Path) -> str | None:
    """``"zstd"`` / ``"gzip"`` from the file's suffix, else ``None``."""
    name = Path(path).name
    for codec, suffix in ARCHIVE_SUFFIXES.items():
        if name.endswith(suffix):
            return codec
    return None


def archive_path(directory: Path) -> Path | None:
    """The existing archive for ``directory``, or ``None``.

    ``.zst`` is preferred when both forms are present. Both can legitimately
    coexist for one session (a 3.12 runtime compressed it, later a 3.14 runtime
    restored and re-compressed it; or a restore that left the old archive of the
    other codec behind after a failed unlink). Reading the newer codec is the
    right preference; and because :func:`restore_for_write` drops every archive
    name it finds, the ambiguity is transient.
    """
    directory = Path(directory)
    for name in ARCHIVE_FILENAMES:
        candidate = directory / name
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - is_file already folds OSError
            continue
    return None


#: The three answers :func:`transcript_state` gives. ``unreadable`` is NOT
#: absence: a directory this process may not traverse must never be read as
#: "nothing durable here", because every remover in the product treats that
#: answer as permission.
TRANSCRIPT_PRESENT = "present"
TRANSCRIPT_ABSENT = "absent"
TRANSCRIPT_UNREADABLE = "unreadable"


def transcript_state(directory: Path) -> str:
    """``present`` / ``absent`` / ``unreadable`` for the session in ``directory``.

    THE existence question, with the unreadable case kept separate instead of
    folded into ``absent``. A caller that only needs "can I show this row" wants
    :func:`has_transcript`; a caller that DELETES on absence — the two
    failed-create removals, the audited removal primitive, a sweep that deletes
    an unreferenced attachment — must ask this, because an ``EACCES`` on the
    directory is not evidence that the transcript is gone.
    """
    directory = Path(directory)
    try:
        if plain_path(directory).exists():
            return TRANSCRIPT_PRESENT
    except OSError:
        return TRANSCRIPT_UNREADABLE
    for name in ARCHIVE_FILENAMES:
        try:
            if (directory / name).is_file():
                return TRANSCRIPT_PRESENT
        except OSError:
            return TRANSCRIPT_UNREADABLE
    return TRANSCRIPT_ABSENT


def has_transcript(directory: Path) -> bool:
    """Whether this session has a transcript in ANY physical form.

    THE existence check. Every surface that asks "is there anything here?" must
    ask this rather than statting ``transcript.jsonl``, or a compressed session
    reads as an empty one: no row in a listing, no hit in search, an "empty"
    conversation on open. A zero-byte plain file counts as present — it is a
    transcript that exists and holds nothing, which is a different fact from
    absence and is the caller's to interpret.

    An unreadable directory answers ``False`` here (this is the tolerant form,
    used by listing surfaces whose failure mode is a missing row). Anything that
    acts ON the answer must use :func:`transcript_state`.
    """
    return transcript_state(directory) == TRANSCRIPT_PRESENT


def transcript_source(directory: Path) -> tuple[Path, str | None]:
    """``(path, codec)`` for whatever is physically on disk.

    ``(plain_path, None)`` when the plain file exists; ``(archive, codec)`` when
    only an archive does. When NEITHER exists the plain path is returned with
    ``None`` so a caller can stat it and get the same ``FileNotFoundError`` it
    always handled — this function does not raise.

    This is the resolver for content reads and for the search index's PHYSICAL
    signature: the signature must describe the file on disk, not the logical
    transcript, or a compressed session re-digests on every picker open.
    """
    directory = Path(directory)
    plain = plain_path(directory)
    try:
        if plain.exists():
            return plain, None
    except OSError:
        pass
    archive = archive_path(directory)
    if archive is not None:
        return archive, _codec_for(archive)
    return plain, None


# ---------------------------------------------------------------------------
# Streaming readers
# ---------------------------------------------------------------------------


#: A decoded archive reader. ``gzip.GzipFile`` and ``compression.zstd.ZstdFile``
#: are both file-like but neither is typed as ``BinaryIO`` (and the zstd type
#: does not exist at all on a 3.12 build), so the annotation is ``Any`` with this
#: note rather than a Protocol that has to restate two stdlib signatures.
_ArchiveReader = Any


def _open_archive_reader(archive: Path, codec: str | None) -> _ArchiveReader:
    """Open ``archive`` for streaming decompression.

    Raises :class:`ArchiveUnavailable` when this build has no decoder for the
    codec and :class:`ArchiveCorrupt` when opening or reading fails. Callers
    translate those into a degraded row or an honest error — never into an
    empty transcript.
    """
    if codec == "zstd":
        if _zstd is None:
            raise ArchiveUnavailable(
                "transcript archive uses zstd, which this Python build cannot decode"
            )
        try:
            return _zstd.ZstdFile(os.fspath(archive), "rb")
        except Exception as exc:  # noqa: BLE001 - any decoder failure is corruption
            raise ArchiveCorrupt(f"{archive.name}: {exc}") from exc
    if codec == "gzip":
        try:
            return gzip.open(os.fspath(archive), "rb")
        except OSError as exc:
            raise ArchiveCorrupt(f"{archive.name}: {exc}") from exc
    raise ArchiveCorrupt(f"{archive.name}: not a known archive form")


def _open_reader(directory: Path) -> tuple[BinaryIO | _ArchiveReader, str | None]:
    """``(reader, codec)`` for the transcript of ``directory``."""
    path, codec = transcript_source(directory)
    if codec is None:
        return path.open("rb"), None
    return _open_archive_reader(path, codec), codec


def read_head(directory: Path, limit: int) -> bytes:
    """The first ``limit`` bytes of the transcript, decompressing only that much.

    The work is bounded by the BUDGET, not by the transcript: zstd and gzip both
    yield incrementally, so a 300 MB archive costs a few chunks to read the first
    256 kB. Raises :class:`ArchiveError` for an archive this build cannot decode
    — an honest failure, not ``b""``, because ``b""`` reads as "empty
    conversation" everywhere a caller looks.
    """
    budget = max(0, int(limit))
    if budget == 0:
        return b""
    try:
        reader, codec = _open_reader(Path(directory))
    except (OSError, ValueError):
        # Same contract as ``read_tail``: a plain journal that cannot be opened
        # reads as empty; an archive's ArchiveError propagates.
        return b""
    try:
        with reader:
            try:
                return reader.read(budget)
            except ArchiveError:
                raise
            except Exception as exc:  # noqa: BLE001 - gzip/zstd raise their own types
                if codec is None:
                    return b""
                raise ArchiveCorrupt(f"{Path(directory).name}: {exc}") from exc
    except (FileNotFoundError, IsADirectoryError):
        return b""
    except OSError as exc:
        if codec is None:
            return b""
        raise ArchiveCorrupt(f"{Path(directory).name}: {exc}") from exc


def read_tail_window(directory: Path, limit: int) -> tuple[bytes, bool]:
    """The last ``limit`` bytes, plus whether the source is LONGER than them.

    The flag is the difference between a window that starts mid-row and one
    that starts on a whole row, and callers that parse the window depend on it:
    a source of exactly ``limit`` bytes is not truncated, so its first line is
    real and dropping it would lose the oldest entry of the window (review n1).

    A PLAIN journal is seeked to ``size - limit``: the read is bounded by the
    caller's budget and independent of the file's size. Only an archive streams
    — neither codec offers a seek from the end — so its time stays linear in the
    archive, and the listing avoids even that through
    :func:`read_listing_preview`.
    """
    budget = max(0, int(limit))
    if budget == 0:
        return b"", False
    path, codec = transcript_source(Path(directory))
    if codec is None:
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if size > budget:
                    handle.seek(size - budget)
                return handle.read(budget), size > budget
        except OSError:
            # An unopenable plain journal reads as empty, which is what an
            # unreadable `read` always answered. Archive failures are
            # ArchiveError and deliberately propagate: they are never "empty".
            return b"", False
    reader = _open_archive_reader(path, codec)
    tail = bytearray()
    total = 0
    try:
        with reader:
            while True:
                try:
                    chunk = reader.read(CHUNK)
                except ArchiveError:
                    raise
                except Exception as exc:  # noqa: BLE001 - decoder-specific types
                    raise ArchiveCorrupt(f"{Path(directory).name}: {exc}") from exc
                if not chunk:
                    break
                total += len(chunk)
                tail.extend(chunk)
                if len(tail) > budget:
                    del tail[: len(tail) - budget]
    except (FileNotFoundError, IsADirectoryError):
        return b"", False
    except OSError as exc:
        raise ArchiveCorrupt(f"{Path(directory).name}: {exc}") from exc
    return bytes(tail), total > budget


def read_tail(directory: Path, limit: int) -> bytes:
    """The last ``limit`` bytes of the transcript.

    Thin wrapper over :func:`read_tail_window`, which is where the seek and the
    streaming live; callers that parse the window should ask that function so
    they can tell a truncated window from a whole one.
    """
    return read_tail_window(directory, limit)[0]


def read_listing_preview(directory: Path) -> dict[str, Any] | None:
    """The archived session's listing sidecar, or ``None`` when there is none.

    THE cheap answer for a compressed session's list row: one stat and a
    sub-kilobyte read, instead of decompressing the archive to reach its tail.
    ``None`` means "ask the transcript": the session is plain (the sidecar
    never exists beside a plain file), the archive predates the sidecar, its
    write failed, or the document is foreign or unreadable. Every caller falls
    back to :func:`read_tail_window`, so a missing sidecar costs speed, never a
    preview.
    """
    directory = Path(directory)
    try:
        if plain_path(directory).exists() or archive_path(directory) is None:
            return None
        info = (directory / PREVIEW_SIDECAR_NAME).stat()
        if info.st_size > PREVIEW_SIDECAR_MAX_BYTES:
            return None
        payload = json.loads((directory / PREVIEW_SIDECAR_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != PREVIEW_SIDECAR_SCHEMA:
        return None
    return payload


def iter_lines(directory: Path) -> Iterator[str]:
    """Yield the transcript's lines as text, decompressing on the fly.

    Decoded with ``errors="replace"``: a transcript is JSON lines and a single
    byte of damage must not stop a caller that is scanning for titles or
    bookkeeping entries. Consumers that need strictness call :func:`read_head`
    and decide for themselves.

    LINE TERMINATORS ARE DROPPED, and ``\r\n`` folds to the same line as
    ``\n``: every consumer parses one JSON document per line, and the
    terminator is the framing rather than the data. A caller rebuilding the
    file must put them back.

    Raises ``FileNotFoundError`` when the session has no transcript at all, and
    :class:`ArchiveError` for an archive this build cannot decode.
    """
    reader, codec = _open_reader(Path(directory))
    with reader:
        while True:
            try:
                line = reader.readline()
            except Exception as exc:  # noqa: BLE001 - decoder-specific types
                if codec is None:
                    return
                raise ArchiveCorrupt(f"{Path(directory).name}: {exc}") from exc
            if not line:
                return
            yield line.decode("utf-8", "replace").rstrip("\r\n")


# ---------------------------------------------------------------------------
# The decompressed cache
# ---------------------------------------------------------------------------


def _config_dir_for(directory: Path) -> Path:
    """The config dir that owns ``directory``, or a temp root when there is none.

    A store session always sits at ``<config>/sessions/<id>``. A test's
    ``tmp_path`` store, a fork's staging directory or a directory someone copied
    out of a store does not, and those still deserve a cache — so they get a
    stable temp root instead of writing into a tree that is not ours.
    """
    parent = Path(directory).parent
    from local_operator.session.retention import SESSIONS_DIRNAME

    if parent.name == SESSIONS_DIRNAME:
        return parent.parent
    return Path(tempfile.gettempdir()) / "local-operator-decompressed"


def _cache_root(directory: Path) -> Path:
    return _config_dir_for(directory) / "cache" / "decompressed"


def _cache_entry_dir(directory: Path) -> Path:
    return _cache_root(directory) / Path(directory).name


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def _cached_plain(directory: Path, archive: Path, codec: str) -> Path | None:
    """The validated cache copy of ``archive``, or ``None`` (dropping a stale one)."""
    entry = _cache_entry_dir(directory)
    plain = entry / PLAIN_FILENAME
    meta = _read_json(entry / ".meta.json")
    if isinstance(meta, dict):
        try:
            archive_stat = archive.stat()
            cached_stat = plain.stat()
            valid = (
                meta.get("archive") == archive.name
                and meta.get("codec") == codec
                and meta.get("archive_size") == archive_stat.st_size
                and meta.get("archive_mtime_ns") == archive_stat.st_mtime_ns
                and meta.get("plain_size") == cached_stat.st_size
            )
        except OSError:
            valid = False
        if valid:
            _touch_cache_meta(entry / ".meta.json", meta)
            return plain
        # A cache copy that does not match the archive is worse than none: it
        # would serve a reader the bytes of a previous version of the file.
        _unlink_file(plain)
        _unlink_file(entry / ".meta.json")
        return None
    _unlink_file(plain)
    return None


def _touch_cache_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    """Update ``last_used`` for LRU order, at most every few minutes."""
    last = meta.get("last_used")
    if isinstance(last, (int, float)) and time.time() - last < CACHE_TOUCH_INTERVAL_S:
        return
    meta = dict(meta)
    meta["last_used"] = time.time()
    _write_json_atomic(meta_path, meta)


def _free_bytes(path: Path) -> int:
    try:
        info = os.statvfs(os.fspath(path))
    except (AttributeError, OSError):
        return CACHE_FREE_RESERVE_BYTES  # cannot tell: do not claim there is room
    return info.f_bavail * info.f_frsize


def _drop_cache(directory: Path) -> None:
    """Forget the cache copy for ``directory`` (files only; see :func:`_evict_cache`)."""
    entry = _cache_entry_dir(directory)
    _unlink_file(entry / PLAIN_FILENAME)
    _unlink_file(entry / ".meta.json")


def _evict_cache(root: Path) -> None:
    """Trim the cache to :data:`CACHE_MAX_ENTRIES` / :data:`CACHE_MAX_BYTES`.

    Only the FILES are removed. Empty entry directories stay behind deliberately:
    ``tests/unit/session/test_no_session_deletion.py`` permits a directory
    remover in exactly one function in this package (the cleanup remover), so a
    cache trimming that ``rmdir``-ed would either fail that gate or need an
    exemption far weaker than leaving a few empty directories in a cache.
    """
    try:
        children = [child for child in root.iterdir() if child.is_dir()]
    except OSError:
        return
    entries: list[tuple[float, int, Path]] = []
    for child in children:
        try:
            size = (child / PLAIN_FILENAME).stat().st_size
        except OSError:
            continue  # evicted or partially written: not a cache entry
        meta = _read_json(child / ".meta.json")
        last_used = meta.get("last_used") if isinstance(meta, dict) else None
        entries.append((float(last_used or 0.0), size, child))
    total = sum(size for _last, size, _child in entries)
    entries.sort(key=lambda row: (row[0], row[2].name))
    remaining = len(entries)
    for _last, size, child in entries:
        if remaining <= CACHE_MAX_ENTRIES and total <= CACHE_MAX_BYTES:
            break
        _unlink_file(child / PLAIN_FILENAME)
        _unlink_file(child / ".meta.json")
        total -= size
        remaining -= 1


def _materialise(directory: Path, archive: Path, codec: str) -> Path | None:
    """Decompress ``archive`` into the cache; ``None`` on any failure.

    Callers fall back to :func:`restore_for_write` when this returns ``None``,
    so a full or unwritable cache volume costs a restore, never a read.
    """
    entry = _cache_entry_dir(directory)
    plain = entry / PLAIN_FILENAME
    try:
        entry.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.info("transcript compression: cache unavailable for %s: %s", directory.name, exc)
        return None
    try:
        archive_stat = archive.stat()
    except OSError:
        return None
    needed = archive_stat.st_size * CACHE_EXPANSION_ESTIMATE
    if _free_bytes(entry) < needed + CACHE_FREE_RESERVE_BYTES:
        logger.info(
            "transcript compression: low disk space, restoring %s in place instead of caching",
            directory.name,
        )
        return None
    tmp = entry / f".{PLAIN_FILENAME}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        _reserve(tmp)
        digest, size = _decode_to_file(archive, codec, tmp)
        _fsync_file(tmp)
        os.utime(tmp, ns=(archive_stat.st_mtime_ns, archive_stat.st_mtime_ns))
        _publish(tmp, plain)
    except (OSError, ArchiveError) as exc:
        logger.warning("transcript compression: cannot cache %s: %s", directory.name, exc)
        _unlink_file(tmp)
        _unlink_file(plain)
        return None
    _write_json_atomic(
        entry / ".meta.json",
        {
            "archive": archive.name,
            "codec": codec,
            "archive_size": archive_stat.st_size,
            "archive_mtime_ns": archive_stat.st_mtime_ns,
            "plain_size": size,
            "sha256": digest,
            "last_used": time.time(),
        },
    )
    _evict_cache(_cache_root(directory))
    return plain


def transcript_for_read(directory: Path) -> Path | None:
    """A real file path holding this session's transcript, or ``None``.

    THE resolver for callers that need a seekable file — page readers, forks,
    anything that reopens by offset. Plain file when it exists; otherwise the
    archive is materialised into the bounded cache and that path is returned.
    ``None`` only when the session has neither form.

    Raises :class:`ArchiveError` when an archive exists but cannot be decoded
    (including the cross-version zstd case). A caller that only needs "does this
    session have a body" wants :func:`has_transcript`, which never raises.
    """
    directory = Path(directory)
    plain = plain_path(directory)
    try:
        if plain.exists():
            return plain
    except OSError:
        pass
    archive = archive_path(directory)
    if archive is None:
        return None
    codec = _codec_for(archive)
    if codec is None:
        raise ArchiveCorrupt(f"{archive.name}: not a known archive form")
    # Touch the decoder first: an unavailable or corrupt archive must raise here,
    # not after this function has spent disk making room for a cache copy.
    reader = _open_archive_reader(archive, codec)
    reader.close()
    cached = _cached_plain(directory, archive, codec)
    if cached is not None:
        return cached
    materialised = _materialise(directory, archive, codec)
    if materialised is not None:
        return materialised
    if restore_for_write(directory):
        return plain
    raise ArchiveCorrupt(f"{archive.name}: could not be decoded into a readable copy")


# ---------------------------------------------------------------------------
# Compression primitives
# ---------------------------------------------------------------------------


def _reserve(path: Path) -> None:
    """Create ``path`` exclusively, or raise. Never adopt a pre-existing file."""
    fd = os.open(os.fspath(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)


def _unlink_file(path: Path) -> bool:
    """Remove ``path`` if it is a REGULAR FILE; ``False`` for anything else.

    THE only unlink in this module, so there is exactly one place to audit. Its
    callers pass only names this module constructs itself: a ``.tmp`` it just
    reserved, the archive it just published, the source it has just verified a
    published replacement for, or a cache copy/record. The regular-file check is
    the belt: a directory, symlink or device is refused even if a future caller
    gets its path wrong.

    Deliberately not an error when the file is already gone — every caller is
    cleaning up something that may have been concurrently removed, and a failed
    cleanup must never raise into a maintenance pass.
    """
    try:
        info = os.lstat(os.fspath(path))
    except OSError:
        return False
    if not _stat.S_ISREG(info.st_mode):
        logger.error("transcript compression: refusing to unlink non-regular file %s", path)
        return False
    try:
        os.unlink(os.fspath(path))
    except OSError:
        return False
    return True


def _publish(tmp: Path, destination: Path) -> None:
    """Atomically displace a fully-written file to ``destination``.

    THE only path displacer in this module. Destinations are always regular
    files this module named itself — a session's transcript archive, the
    QUARANTINE the plain journal is parked under while compression decides
    whether it may be retired, the plain name when that journal is put back, a
    cache ``.meta.json``, a cache transcript, or
    ``sessions/last-compression.json``. Never a session directory: compression
    replaces files, and a session directory is not this module's to rename or
    remove.
    """
    os.replace(os.fspath(tmp), os.fspath(destination))


def _fsync_file(path: Path) -> None:
    """Best-effort durability for a file this module just wrote."""
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some filesystems refuse
        pass
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Best-effort durability for a directory entry (the rename's durability)."""
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - Windows refuses to open a directory
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some filesystems refuse
        pass
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: Any) -> bool:
    """tmp + publish, so no reader ever sees a half-written record."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        _reserve(tmp)
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        _fsync_file(tmp)
        _publish(tmp, path)
    except OSError as exc:
        logger.warning("transcript compression: cannot write %s: %s", path.name, exc)
        _unlink_file(tmp)
        return False
    return True


@contextmanager
def _archive_writer(path: Path, codec: str) -> Iterator[Any]:
    """Streaming compressed writer for ``codec``."""
    if codec == "zstd":
        if _zstd is None:  # pragma: no cover - available_codec prevents this
            raise ArchiveUnavailable("zstd is not available in this Python build")
        # A FRAME CHECKSUM, not just the level: gzip has always carried a CRC32
        # over the payload, so a flipped byte in a ``.gz`` archive fails at decode
        # time. A zstd frame without ``checksum_flag`` is happy to hand back
        # corrupted bytes, which would make the gzip half of this codec pair the
        # only half that detects damage. ``level`` and ``options`` are mutually
        # exclusive on ``ZstdFile``, so the level rides inside the options dict.
        options = {
            _zstd.CompressionParameter.compression_level: ZSTD_LEVEL,
            _zstd.CompressionParameter.checksum_flag: 1,
        }
        with _zstd.ZstdFile(os.fspath(path), "wb", options=options) as handle:
            yield handle
        return
    with gzip.GzipFile(os.fspath(path), "wb", compresslevel=GZIP_LEVEL) as handle:
        yield handle


def _write_compressed(source: Path, tmp: Path, codec: str) -> tuple[str, int]:
    """Stream ``source`` into ``tmp`` compressed; return (sha256, bytes read)."""
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as reader, _archive_writer(tmp, codec) as writer:
        while True:
            chunk = reader.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            writer.write(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _decoded_digest(archive: Path, codec: str) -> tuple[str, int]:
    """Decode ``archive`` fully; return (sha256 of the decoded bytes, byte count).

    THE verification step. It reads the compressed file back through the real
    decoder, so it fails on a truncated stream, a bad gzip CRC, a zstd frame
    error and a mismatch between what was written and what can be read — not
    merely on a difference between two in-memory buffers.
    """
    digest = hashlib.sha256()
    total = 0
    reader = _open_archive_reader(archive, codec)
    try:
        with reader:
            while True:
                try:
                    chunk = reader.read(CHUNK)
                except Exception as exc:  # noqa: BLE001 - decoder-specific types
                    raise ArchiveCorrupt(f"{archive.name}: {exc}") from exc
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
    except OSError as exc:
        raise ArchiveCorrupt(f"{archive.name}: {exc}") from exc
    return digest.hexdigest(), total


def _decode_to_file(archive: Path, codec: str, destination: Path) -> tuple[str, int]:
    """Decode ``archive`` into ``destination``; return (sha256, byte count).

    Used by the restore path, where there is no original left to compare
    against: the verification is that the stream decodes completely (both
    formats carry their own integrity check over the payload) and that the
    bytes written are the bytes counted.
    """
    digest = hashlib.sha256()
    total = 0
    reader = _open_archive_reader(archive, codec)
    try:
        with reader, destination.open("wb") as writer:
            while True:
                try:
                    chunk = reader.read(CHUNK)
                except Exception as exc:  # noqa: BLE001 - decoder-specific types
                    raise ArchiveCorrupt(f"{archive.name}: {exc}") from exc
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                total += len(chunk)
    except OSError as exc:
        raise ArchiveCorrupt(f"{archive.name}: {exc}") from exc
    return digest.hexdigest(), total


def _stat_sig(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = os.stat(os.fspath(path))
    except OSError:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


# ---------------------------------------------------------------------------
# The writer lock
# ---------------------------------------------------------------------------

#: ``realpath -> [fd, refcount]`` for writers held by THIS process. The refcount
#: is what lets two ``Transcript`` objects in one process share the lock: the
#: first acquires the kernel lock, the second only bumps the count, and neither
#: can release the other's hold. Without it the second object's close would
#: unlock a session the first is still writing.
_WRITERS: dict[str, list[Any]] = {}
_WRITERS_GUARD = threading.Lock()


def _lock_path(directory: Path) -> Path:
    return Path(directory) / LOCK_NAME


def _flock(fd: int) -> bool:
    """Take an exclusive non-blocking lock on ``fd``; ``False`` when busy."""
    if _fcntl is not None:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            return False
        return True
    if _msvcrt is not None:  # pragma: no cover - exercised on Windows only
        # `getattr` rather than attributes: pyright's typeshed for `msvcrt` on a
        # POSIX analysis host does not carry the locking members, and this branch
        # never runs there anyway.
        locker = getattr(_msvcrt, "locking", None)
        if locker is None:
            return True
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            locker(fd, getattr(_msvcrt, "LK_NBLCK", 2), 1)
        except OSError:
            return False
        return True
    return True  # neither primitive: a no-op lock, documented in the docstring


def _funlock(fd: int) -> None:
    if _fcntl is not None:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        except OSError:
            pass
    elif _msvcrt is not None:  # pragma: no cover - exercised on Windows only
        unlocker = getattr(_msvcrt, "locking", None)
        if unlocker is None:
            return
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            unlocker(fd, getattr(_msvcrt, "LK_UNLCK", 0), 1)
        except OSError:
            pass


def _try_acquire_lock(directory: Path) -> int | None:
    """Open and lock the session's writer lock; ``None`` when it is busy.

    Fail CLOSED: a lock file that cannot be opened or locked counts as "someone
    else has this session", because the one thing this lock exists to prevent is
    rewriting a transcript under a live writer.
    """
    try:
        fd = os.open(os.fspath(_lock_path(directory)), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return None
    if _flock(fd):
        return fd
    try:
        os.close(fd)
    except OSError:
        pass
    return None


def _release_lock_fd(fd: int) -> None:
    _funlock(fd)
    try:
        os.close(fd)
    except OSError:  # pragma: no cover
        pass


def lock_is_busy(directory: Path) -> bool:
    """Whether a live writer holds this session, WITHOUT creating the lock file.

    A dry run must be able to report "session in use" without changing anything,
    so this opens an EXISTING lock file and tries the lock. A missing lock file
    means no holder (the only way to hold the lock is to have created it), and
    an unreadable one reads as busy — fail closed, like every guard here.
    """
    path = _lock_path(directory)
    try:
        fd = os.open(os.fspath(path), os.O_RDWR)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        if _flock(fd):
            _funlock(fd)
            return False
        return True
    finally:
        try:
            os.close(fd)
        except OSError:  # pragma: no cover
            pass


#: How often :func:`acquire_writer` re-attempts a held lock while it waits.
#: Short enough that a single-session compress (well under a second) is invisible
#: to the opener, long enough that a second real runtime costs a handful of
#: syscalls rather than a spin.
WRITER_WAIT_POLL_S = 0.025

#: How long ``Transcript.__init__`` waits for the compressor before it opens
#: WITHOUT the writer lock. Long enough that one session's compress (measured in
#: well under a second) never loses a runtime its history, short enough that a
#: genuinely contended open is not a stalled launch.
WRITER_LOCK_WAIT_S = 5.0


def acquire_writer(directory: Path, *, wait_s: float = 0.0) -> bool:
    """Take the session's writer lock for this process; ``False`` when busy.

    Held for the lifetime of a ``Transcript`` (stage B), which is exactly the
    statement "a runtime has this session open": the compressor's non-blocking
    attempt then fails and the session is skipped rather than rewritten under a
    live writer. Refcounted per path so two holders in ONE process share the
    kernel lock instead of fighting over it.

    ``wait_s`` BOUNDS THE COMPRESSOR RACE, and it exists because the loser of
    that race is the conversation. ``Transcript.__init__`` restores a compressed
    journal and then reads it; a compressor that held the lock while it did so
    would publish its archive and unlink the plain file AFTER the restore's
    no-op check and BEFORE the runtime's first append — which then recreates an
    empty plain file that the reader prefers, hiding the archive's real history.
    Waiting for the lock makes "the compressor is mid-pass" a short pause rather
    than a silent wipe. It is deliberately BOUNDED (the caller passes a few
    seconds): the other holder may be a second real runtime, and the documented
    two-writer hazard is not something a longer wait fixes.
    """
    key = os.path.realpath(os.fspath(directory))
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        with _WRITERS_GUARD:
            held = _WRITERS.get(key)
            if held is not None:
                held[1] = int(held[1]) + 1
                return True
            fd = _try_acquire_lock(Path(directory))
            if fd is not None:
                _WRITERS[key] = [fd, 1]
                return True
        if time.monotonic() >= deadline:
            return False
        # OUTSIDE the guard: sleeping under it would serialize every other
        # session's acquisition in this process behind one held lock.
        time.sleep(WRITER_WAIT_POLL_S)


def release_writer(directory: Path) -> None:
    """Release one hold on the session's writer lock (no-op if none is held)."""
    key = os.path.realpath(os.fspath(directory))
    with _WRITERS_GUARD:
        held = _WRITERS.get(key)
        if held is None:
            return
        held[1] = int(held[1]) - 1
        if held[1] > 0:
            return
        _WRITERS.pop(key, None)
        _release_lock_fd(held[0])


# ---------------------------------------------------------------------------
# The outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """One session's result. Statuses: ``compressed`` / ``skipped`` / ``failed``."""

    session: str
    status: str
    reason: str = ""
    bytes_before: int = 0
    bytes_after: int = 0
    codec: str = ""
    elapsed_s: float = 0.0

    @property
    def compressed(self) -> bool:
        return self.status == "compressed"


@dataclass
class CompressionResult:
    """One pass, with enough detail for a dry run and for an audit row."""

    scanned: int = 0
    #: Directories that are candidates and NOT guarded (dry run rows).
    candidates: list[Outcome] = field(default_factory=list)
    compressed: list[Outcome] = field(default_factory=list)
    #: Every per-session skip this pass decided, in scan order.
    skipped_rows: list[Outcome] = field(default_factory=list)
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    elapsed_s: float = 0.0
    dry_run: bool = False
    #: The pass stopped because its wall-clock budget ran out (between sessions).
    budget_exhausted: bool = False
    stopped: bool = False
    #: Why the pass did not run at all ("disabled", "no store", ...).
    skipped: str | None = None


@dataclass(frozen=True)
class CompressionPolicy:
    """``session.compression.*`` — the switch and the age threshold."""

    enabled: bool = DEFAULT_ENABLED
    after_days: int = DEFAULT_AFTER_DAYS


def _coerce_bool(value: Any, default: bool) -> bool:
    """The settings registry's strict parser, so the page and the pass agree."""
    from local_operator.settings_io import strict_bool

    return strict_bool(value, default)


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number


def policy_from_config(config_manager: Any) -> CompressionPolicy:
    """Read ``session.compression.*`` through ``ConfigManager.get_nested_value``.

    Duck-typed, like ``cleanup.policy_from_config``: a manager that cannot be
    read yields the DISABLED policy rather than the module default. Compression
    is ON by default, so guessing "enabled" from an unreadable config would let
    a stub arm a pass that rewrites transcripts.
    """
    getter: Callable[..., Any] | None = getattr(config_manager, "get_nested_value", None)
    if getter is None:
        return CompressionPolicy(enabled=False)

    def leaf(name: str, default: Any) -> Any:
        try:
            return getter(COMPRESSION_PATH + (name,), default)
        except Exception:  # noqa: BLE001 — a broken config is not a reason to run
            return default

    enabled = _coerce_bool(leaf("enabled", DEFAULT_ENABLED), DEFAULT_ENABLED)
    days = _coerce_int(leaf("after_days", DEFAULT_AFTER_DAYS), DEFAULT_AFTER_DAYS)
    if days < MIN_AFTER_DAYS or days > MAX_AFTER_DAYS:
        # A value the page would refuse cannot arm the pass at an unexpected
        # threshold; the shipped default is the one a broken file gets.
        days = DEFAULT_AFTER_DAYS
    return CompressionPolicy(enabled=enabled, after_days=days)


# ---------------------------------------------------------------------------
# The pass's helpers
# ---------------------------------------------------------------------------


def _after_archive(directory: Path, archive: Path, codec: str) -> None:
    """Post-publish bookkeeping for a session whose transcript is now archived.

    Both steps are ADVISORY. The archive is already verified and published, and
    the source is about to be removed, so a failure here costs a search entry's
    freshness or a stale cold-read hint — never data. Hence no exception escapes:
    raising would leave a session with a verified archive AND a live source, and
    the pass would count an error that did not lose anything.
    """
    try:
        from local_operator.session.search_index import record_archive

        record_archive(_config_dir_for(directory), directory.name, source=archive, codec=codec)
    except Exception as exc:  # noqa: BLE001 — a cache rewrite is advisory
        logger.warning(
            "transcript compression: search index not updated for %s: %s", directory.name, exc
        )
    anchor = Path(directory) / TAIL_ANCHOR_NAME
    try:
        if anchor.is_file():
            # The anchor names byte offsets and a size in the PLAIN file; the
            # file it describes no longer exists.
            _unlink_file(anchor)
    except OSError:  # pragma: no cover - is_file already folds OSError
        pass


def _listing_payload(directory: Path) -> dict[str, str]:
    """The two listing fields the sidecar carries, from the PLAIN tail.

    Called while the session is still plain (a seek of at most
    :data:`PREVIEW_TAIL_BYTES`), so the cost is the same bounded read a listing
    pays today, once per compression instead of once per row per process.

    ``resume`` owns the preview rules — the condensed assistant line and the
    failure record — so they are imported HERE rather than reimplemented: this
    module stays stdlib-only at module scope (see the module docstring), and a
    second copy of the rules is how the sidecar and a live preview drift apart.
    """
    try:
        window, truncated = read_tail_window(directory, PREVIEW_TAIL_BYTES)
    except ArchiveError:  # pragma: no cover - the session is plain by contract here
        return {"preview": "", "failure": ""}
    try:
        from local_operator.resume import listing_preview_from_tail
    except Exception:  # noqa: BLE001 - a missing helper costs a preview, not a compression
        logger.warning("transcript compression: preview rules unavailable", exc_info=True)
        return {"preview": "", "failure": ""}
    return listing_preview_from_tail(window, truncated=truncated)


def _write_preview_sidecar(directory: Path, listing: dict[str, str]) -> None:
    """Write ``preview.json`` atomically. Advisory: a failure costs a slow row."""
    payload = {
        "schema": PREVIEW_SIDECAR_SCHEMA,
        "preview": listing.get("preview", ""),
        "failure": listing.get("failure", ""),
        "at": time.time(),
    }
    if _write_json_atomic(directory / PREVIEW_SIDECAR_NAME, payload):
        _fsync_dir(directory)


def _ensure_title_sidecar(directory: Path) -> None:
    """Guarantee ``title.json`` exists before the plain file stops being read.

    An archived session's stored title is read from its HEAD window only (see
    ``resume.stored_session_title``); reaching for the tail would decompress the
    whole archive once per listing row. That is lossless only when the newest
    title sits in the sidecar, so compression resolves it HERE — the transcript
    is still plain, so the existing head+tail window scan is cheap — and writes
    it through the same API a rename uses (which preserves the directory's
    mtime). No title resolvable ⇒ nothing written, and the opener still names
    the row from the head window.
    """
    try:
        from local_operator.resume import TITLE_SIDECAR_NAME as title_name
        from local_operator.resume import title_seed_for_sidecar, write_session_title
    except Exception:  # noqa: BLE001 - a missing helper costs a title, not a compression
        logger.warning("transcript compression: title rules unavailable", exc_info=True)
        return
    try:
        if (directory / title_name).is_file():
            return
    except OSError:  # pragma: no cover - is_file already folds OSError
        return
    seed = title_seed_for_sidecar(directory)
    if seed is None:
        return
    text, user_set, past_names = seed
    write_session_title(directory, text, user_set=user_set, past_names=past_names)


def _recover_quarantine(directory: Path) -> bool:
    """Undo a compression interrupted between the quarantine rename and its unlink.

    Returns whether it restored anything. Readers only ever ask for the two
    known transcript names, so a leftover quarantine makes a session look
    EMPTY; putting it back is the difference between "the conversation is
    gone" and "the pass will retry tomorrow".

    Both files present is the one case this does NOT repair: a writer recreated
    the plain journal after the crash, and merging two journals is not a
    recovery. Both are left where they are and reported, which keeps every row
    on disk.
    """
    directory = Path(directory)
    quarantine = directory / QUARANTINE_NAME
    try:
        if not quarantine.is_file():
            return False
    except OSError:
        return False
    plain = plain_path(directory)
    try:
        if plain.exists():
            logger.error(
                "transcript compression: %s holds both a plain transcript and a "
                "quarantine (%s); leaving both in place",
                directory.name,
                QUARANTINE_NAME,
            )
            return False
        _publish(quarantine, plain)
    except OSError as exc:
        logger.error("transcript compression: cannot recover %s: %s", directory.name, exc)
        return False
    # The archive published just before the crash is a copy of these bytes and
    # the plain file is authoritative again: drop it (and the cache copy built
    # from it) so the next pass starts from a clean slate.
    for name in ARCHIVE_FILENAMES:
        _unlink_file(directory / name)
    _unlink_file(directory / PREVIEW_SIDECAR_NAME)
    _drop_cache(directory)
    _fsync_dir(directory)
    logger.warning(
        "transcript compression: recovered %s from an interrupted compression", directory.name
    )
    return True


def unreadable_archive_error(directory: Path) -> ArchiveError | None:
    """The typed reason this build cannot read ``directory``'s archive, or ``None``.

    :func:`restore_for_write` answers a bool, which is enough for a caller that
    has a fallback and useless for one that must not open an EMPTY conversation
    over a full one. ``Transcript.__init__`` asks this after a failed restore: an
    archive that exists and cannot be decoded is the same class of fact as a
    byte-corrupt plain journal, which already raises, so it names the archive
    and the codec instead of loading zero rows (review m4).
    """
    directory = Path(directory)
    archive = archive_path(directory)
    if archive is None:
        return None
    codec = _codec_for(archive)
    if codec is None:  # pragma: no cover - archive_path only names known suffixes
        return ArchiveCorrupt(f"{archive}: not a known transcript archive form")
    try:
        _open_archive_reader(archive, codec).close()
    except ArchiveError as exc:
        # Re-typed with the same class so a caller can branch on decoder-missing
        # vs corrupt, and prefixed with the file and codec so the message names
        # both facts a reader needs.
        return type(exc)(f"{archive}: {codec}: {exc}")
    return ArchiveCorrupt(
        f"{archive}: {codec}: the archive could not be decoded into a readable copy"
    )


def compress_session(
    directory: Path, *, codec: str | None = None, now: float | None = None
) -> Outcome:
    """Compress ONE session's transcript, following the documented protocol.

    The mechanical half of the pass: it does not decide whether the session is
    ELIGIBLE (age, guards, policy) — :func:`run_compression` does. It does take
    the writer lock itself, so a direct caller cannot compress a session a live
    runtime has open.

    Returns an :class:`Outcome`; it does not raise for an ordinary failure, so a
    pass can keep going and count it.
    """
    directory = Path(directory)
    started = time.monotonic()
    chosen = codec or available_codec()
    lock = _try_acquire_lock(directory)
    if lock is None:
        return Outcome(directory.name, "skipped", "session in use", codec=chosen)
    try:
        return _compress_session_locked(directory, chosen, started)
    finally:
        _release_lock_fd(lock)


def _compress_session_locked(directory: Path, codec: str, started: float) -> Outcome:
    def done(status: str, reason: str, *, before: int = 0, after: int = 0) -> Outcome:
        return Outcome(
            directory.name,
            status,
            reason,
            bytes_before=before,
            bytes_after=after,
            codec=codec,
            elapsed_s=time.monotonic() - started,
        )

    source = plain_path(directory)
    # A crash between the quarantine rename and its unlink leaves the plain file
    # parked under the quarantine name. Put it back before anything reads it as
    # "no plain transcript" — the archive published just before the crash is a
    # copy of the same bytes, so recovering is a rename and a drop.
    _recover_quarantine(directory)
    if archive_path(directory) is not None:
        # Never rewrite an existing archive, and never keep two forms of the same
        # transcript: the plain file wins for readers, and a second pass is a
        # no-op until something restores it.
        return done("skipped", "archive already present")
    sig0 = _stat_sig(source)
    if sig0 is None:
        return done("skipped", "no plain transcript")
    if sig0[2] == 0:
        return done("skipped", "empty transcript")
    # The listing sidecar and the title sidecar are computed while the plain file
    # is still here: after this point the only readable copy is the archive, and
    # both values exist to spare a reader a decompression.
    listing = _listing_payload(directory)
    _ensure_title_sidecar(directory)
    tmp = directory / f".{PLAIN_FILENAME}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        _reserve(tmp)
    except OSError as exc:
        # A tmp name that already exists means we must NOT adopt it; a new random
        # name would be needed, which is the caller's next pass.
        logger.error(
            "transcript compression: cannot create a tmp beside %s: %s", directory.name, exc
        )
        return done("failed", "cannot create a tmp file")
    try:
        original_digest, original_size = _write_compressed(source, tmp, codec)
    except (OSError, ArchiveError) as exc:
        logger.error("transcript compression: write failed for %s: %s", directory.name, exc)
        _unlink_file(tmp)
        return done("failed", f"write failed: {exc}")
    try:
        # The tmp carries a ".tmp" suffix, so the codec is passed explicitly
        # rather than sniffed from the name: this is the one verification that
        # must decode with the codec the writer actually used.
        decoded_digest, decoded_size = _decoded_digest(tmp, codec)
    except ArchiveError as exc:
        logger.error("transcript compression: verification failed for %s: %s", directory.name, exc)
        _unlink_file(tmp)
        return done("failed", "verification failed")
    if decoded_digest != original_digest or decoded_size != original_size:
        logger.error(
            "transcript compression: verification mismatch for %s (wrote %d, read back %d)",
            directory.name,
            original_size,
            decoded_size,
        )
        _unlink_file(tmp)
        return done("failed", "verification mismatch")
    _fsync_file(tmp)
    # The archive carries the SOURCE's clock: the activity ranking must not move.
    try:
        os.utime(tmp, ns=(sig0[3], sig0[3]))
    except OSError as exc:
        # NOT advisory. The archive's mtime IS the session's activity clock
        # (``retention._ACTIVITY_FILES``), so an archive stamped "now" would jump
        # to the top of ``/resume`` and displace a recent conversation from the
        # cleanup policy's recent-N guard. Keeping the source untouched is the
        # only outcome that leaves the listing byte-identical.
        logger.error("transcript compression: cannot stamp %s: %s", directory.name, exc)
        _unlink_file(tmp)
        return done("failed", "cannot preserve the transcript clock")
    if _stat_sig(source) != sig0:
        _unlink_file(tmp)
        return done("skipped", "the transcript changed while it was being compressed")
    archive = directory / (PLAIN_FILENAME + ARCHIVE_SUFFIXES[codec])
    try:
        _publish(tmp, archive)
    except OSError as exc:
        logger.error("transcript compression: cannot publish %s: %s", archive.name, exc)
        _unlink_file(tmp)
        return done("failed", "cannot publish the archive")
    # THE REMOVAL IS ATOMIC WITH THE DECISION (review M2). Any re-stat followed by
    # an unlink is check-then-act: an append that lands between the two calls is
    # deleted with the file, and the archive — a copy of the PRE-append bytes —
    # becomes the only copy, silently dropping the new row. So the plain file is
    # RENAMED first and the renamed inode is what gets stat'ed. A writer holding
    # an fd opened before the rename appends into the quarantined inode, which is
    # exactly what the stat below sees, and the file is put back.
    #
    # ``Transcript._append`` opens the journal by path per append (no long-lived
    # fd), so a writer that opens AFTER the rename creates a new plain file
    # rather than writing into the quarantine; the check after the unlink reports
    # that case. Nothing here merges it — merging two journals is not this
    # pass's job, and refusing to lose the archive is.
    quarantine = directory / QUARANTINE_NAME
    try:
        _publish(source, quarantine)
    except FileNotFoundError:
        # The source vanished between the publish and the rename: the archive is
        # the ONLY surviving copy of the conversation, so retire nothing. Only a
        # source that is present AND different may withdraw its archive.
        _fsync_dir(directory)
        _write_preview_sidecar(directory, listing)
        _after_archive(directory, archive, codec)
        try:
            after = archive.stat().st_size
        except OSError:  # pragma: no cover - just published
            after = 0
        return done(
            "compressed",
            "source vanished; the archive is the surviving copy",
            before=sig0[2],
            after=after,
        )
    except OSError as exc:
        # Cannot park the source, so cannot prove it is safe to retire: withdraw
        # the archive and leave the plain file exactly as it was.
        logger.error("transcript compression: cannot quarantine %s: %s", source.name, exc)
        _unlink_file(archive)
        return done("failed", "cannot quarantine the source")
    quarantined = _stat_sig(quarantine)
    if quarantined is not None and quarantined != sig0:
        # An append landed after the publish — the documented writer that timed
        # out on the lock. Put the plain file back (the appended rows are in
        # this inode) and withdraw the archive: it is a copy of the pre-append
        # bytes and must not shadow the grown source.
        try:
            _publish(quarantine, source)
        except OSError as exc:  # pragma: no cover - the recovery pass reports this
            logger.error(
                "transcript compression: cannot restore the quarantined transcript for %s: %s",
                directory.name,
                exc,
            )
            _unlink_file(archive)
            return done("failed", "cannot restore the quarantined transcript")
        _unlink_file(archive)
        _fsync_dir(directory)
        return done("skipped", "the transcript changed while it was being compressed")
    _unlink_file(quarantine)
    if _stat_sig(source) is not None:  # pragma: no cover - microsecond window
        # A writer reopened the plain path after the rename and wrote a row into
        # a NEW file. Both copies hold rows the other does not, so neither is
        # removed; the archive stays the only copy of the history before the
        # append. Reported, never merged.
        logger.error(
            "transcript compression: %s reappeared during the quarantine window; "
            "the archive and the new plain file both stand",
            source.name,
        )
    _fsync_dir(directory)
    _write_preview_sidecar(directory, listing)
    _after_archive(directory, archive, codec)
    try:
        after = archive.stat().st_size
    except OSError:  # pragma: no cover - just published
        after = 0
    return done("compressed", "", before=sig0[2], after=after)


def _append_log(sessions_dir: Path, record: dict[str, Any]) -> bool:
    """Append one row to the store's compression log; True only if DURABLE.

    Same shape as ``cleanup._append_cleanup_log`` (fsync before returning)
    because it is the same promise: a row left in the page cache and lost with
    the host is a row that never existed. It is a RECEIPT here, not a gate —
    the archive is verified before anything is removed, so a failure to record
    costs an audit line, not data.
    """
    path = Path(sessions_dir) / COMPRESSION_LOG_NAME
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        logger.error("transcript compression: cannot append to %s: %s", path, exc)
        return False
    return True


def write_last_compression(sessions_dir: Path, result: CompressionResult, *, actor: str) -> None:
    """Record a PASS THAT DID WORK in :data:`LAST_COMPRESSION_NAME`."""
    payload = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "actor": actor,
        "pid": os.getpid(),
        "scanned": result.scanned,
        "compressed": len(result.compressed),
        "bytes_before": result.bytes_before,
        "bytes_after": result.bytes_after,
        "elapsed_s": round(result.elapsed_s, 3),
        "codec": available_codec(),
        "record": str(Path(sessions_dir) / COMPRESSION_LOG_NAME),
    }
    _write_json_atomic(Path(sessions_dir) / LAST_COMPRESSION_NAME, payload)


def _keep_reason(
    directory: Path,
    config_dir: Path,
    now: float,
    *,
    live_dir: Path | None,
    parents: set[str],
    projects: set[str],
) -> str | None:
    """Why this session must NOT be compressed (a short phrase), else ``None``.

    Every rule is a KEEP, and every probe fails closed: an exception becomes
    ``"guard failed: <type>"`` for this session rather than "no objection". The
    store-wide sets (``parents``, ``projects``) are computed once per pass; an
    unreadable one stops the whole pass in :func:`run_compression`.

    ``cleanup._transcript_write_guard`` is reused rather than copied, so an
    operator hold, a live claim, a lease, an armed wake and a monitor mean here
    exactly what they mean to the cleanup policy — one definition, and a new
    hazard added there protects this pass too.

    The CONTENT probes are deliberately NOT reused. ``cleanup._guard`` also
    refuses on unread spooled mail and on an open ask, and it must: deleting the
    directory destroys a message the user never read or a question they can
    still answer. Compression is reversible (a verified restore happens on open)
    and never touches ``inbox.jsonl``/``asks.jsonl``, so those two are not
    hazards to a rewrite — asking them here made the pass skip every old
    transcript that had ever exchanged mail, which on a real store meant none of
    the operator's sessions older than the threshold were compressed at all.
    """
    name = directory.name
    if live_dir is not None:
        try:
            if directory.resolve() == Path(live_dir).resolve():
                return "the current session"
        except OSError:
            return "cannot resolve path"
    if name in parents:
        return "parent session still active"
    if name in projects:
        return "linked to an open project"
    from local_operator.session import cleanup, delegated_retention

    try:
        guard = cleanup._transcript_write_guard(directory, config_dir, now)
    except Exception as exc:  # noqa: BLE001 — a guard that cannot answer keeps
        return f"guard failed: {type(exc).__name__}"
    if guard is not None:
        return guard
    try:
        hazard = delegated_retention.scratchpad_git_hazard(directory)
    except Exception as exc:  # noqa: BLE001 — cannot inspect is keep
        return f"guard failed: {type(exc).__name__}"
    if hazard:
        return hazard
    # The writer lock is the statement "a runtime has this session open", and
    # the pass must answer it the same way in a rehearsal and for real. The real
    # path re-attempts it inside `compress_session` (the authoritative check,
    # and the one that cannot be raced by a session opening mid-pass).
    try:
        if lock_is_busy(directory):
            return "session in use"
    except OSError as exc:
        return f"guard failed: {type(exc).__name__}"
    return None


def run_compression(
    config_dir: Path,
    policy: CompressionPolicy,
    *,
    live_dir: Path | None = None,
    now: float | None = None,
    dry_run: bool = False,
    actor: str = "startup",
    budget_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    delegated_window_hours: float | None = None,
) -> CompressionResult:
    """Compress every eligible transcript under ``config_dir/sessions``.

    ELIGIBILITY, in the order the pass evaluates it (cheap local facts first,
    then the per-candidate guards — the outcome is the same set, and this order
    is what keeps a 2,700-directory store's launch pass to two stats per
    directory): the session is not ``live_dir``; its activity clock exists and is
    older than ``policy.after_days``; its id is not in the roster of a live or
    recent session (transitively); it is not linked to an open project; and
    ``cleanup._transcript_write_guard`` raises no objection (the deletion-only
    content probes are deliberately excluded — see ``_keep_reason``). Then, per
    candidate, an unpushed or
    uncommitted scratchpad repo keeps it, and a held writer lock skips it as
    "session in use".

    A DRY RUN changes nothing: it lists the candidates and every skip with its
    reason, and does not create a lock file or even a cache directory.

    ``budget_s`` bounds the pass between sessions (never mid-session, so a
    compression is atomic-or-nothing); ``should_stop`` is polled at the same
    points, so a leaving runtime finishes the session in hand and stands down.
    The pass never raises: an error is counted and the next session is tried.
    """
    result = CompressionResult(dry_run=dry_run)
    started = time.monotonic()
    if not policy.enabled and not dry_run:
        result.skipped = "disabled"
        result.elapsed_s = time.monotonic() - started
        return result
    sessions_dir = Path(config_dir) / "sessions"
    if not sessions_dir.is_dir():
        result.skipped = "no store"
        result.elapsed_s = time.monotonic() - started
        return result
    if dry_run and not policy.enabled:
        # Listed (safe and useful), but the caller is told why nothing would run.
        result.skipped = "disabled"
    moment = now if now is not None else time.time()
    cutoff = moment - policy.after_days * 86400.0
    budget = DEFAULT_BUDGET_S if budget_s is None else float(budget_s)
    deadline = float("inf") if dry_run else started + budget

    from local_operator.session import delegated_retention, retention

    try:
        # The SAME window the delegated reap pass uses, when the caller knows it:
        # a parent kept alive by a widened ``cleanup.delegated.max_age_hours``
        # could still resume its children, so compressing them under a narrower
        # module default would protect the parent and strand the child.
        parents = delegated_retention.active_parent_sessions(
            config_dir,
            moment,
            **(
                {"window_hours": float(delegated_window_hours)}
                if delegated_window_hours is not None
                else {}
            ),
        )
        projects = delegated_retention.open_project_sessions(config_dir)
    except Exception as exc:  # noqa: BLE001 — cannot read the keep rules: keep everything
        logger.warning("transcript compression: cannot read the store's keep rules: %s", exc)
        result.skipped = f"cannot read the store's keep rules: {exc}"
        result.errors += 1
        result.elapsed_s = time.monotonic() - started
        return result

    try:
        # A context manager so the scandir descriptor is released at the end of
        # the walk rather than whenever CPython finalises the exhausted iterator.
        with os.scandir(sessions_dir) as scan:
            entries = [entry for entry in scan if entry.is_dir(follow_symlinks=False)]
    except OSError as exc:
        logger.warning("transcript compression: cannot scan %s: %s", sessions_dir, exc)
        result.skipped = "cannot scan the store"
        result.errors += 1
        result.elapsed_s = time.monotonic() - started
        return result
    entries = [entry for entry in entries if not entry.name.startswith(".")]

    # One stat for the clock per directory, then oldest first, then name: a
    # deterministic order makes the budget cut and a dry run agree about what a
    # pass would do. A clockless directory ranks LAST and is skipped below.
    ranked: list[tuple[float, str, Path, float | None]] = []
    for entry in entries:
        result.scanned += 1
        directory = Path(entry.path)
        try:
            activity = retention.session_activity(directory)
        except Exception:  # noqa: BLE001 — no clock is a keep, not a reason to stop
            activity = None
        ranked.append((0.0 if activity is None else activity, entry.name, directory, activity))
    ranked.sort(key=lambda row: (row[3] is None, row[0], row[1]))

    codec = available_codec()

    def skip(name: str, reason: str) -> None:
        """Record one skipped session and its reason (the dry run's whole value)."""
        result.skipped_rows.append(Outcome(name, "skipped", reason, codec=codec))
        result.skipped_by_reason[reason] = result.skipped_by_reason.get(reason, 0) + 1

    for _activity_key, name, directory, activity in ranked:
        if should_stop is not None and should_stop():
            result.stopped = True
            break
        if not dry_run and time.monotonic() > deadline:
            result.budget_exhausted = True
            break
        # Which file this session keeps its history in, decided ONCE here: a
        # second look could disagree with the first if a pass ran concurrently.
        if not dry_run:
            # A crash between the quarantine rename and its unlink leaves the
            # plain file parked under a dot-name. Without this the session reads
            # as "already compressed" AND has no readable transcript, so it
            # would never be revisited (review M2). One stat per candidate.
            _recover_quarantine(directory)
        plain_present = plain_path(directory).exists()
        if not plain_present:
            skip(
                name,
                "already compressed" if archive_path(directory) is not None else "no transcript",
            )
            continue
        if activity is None:
            skip(name, "no activity clock")
            continue
        if activity >= cutoff:
            skip(name, f"active within {policy.after_days}d")
            continue
        reason = _keep_reason(
            directory,
            config_dir,
            moment,
            live_dir=live_dir,
            parents=parents,
            projects=projects,
        )
        if reason is not None:
            skip(name, reason)
            continue
        if dry_run:
            # No lock FILE is created by a rehearsal (see `lock_is_busy`), so a
            # dry run leaves the session directory byte-identical.
            if lock_is_busy(directory):
                skip(name, "session in use")
                continue
            sample = _stat_sig(plain_path(directory))
            result.candidates.append(
                Outcome(
                    name,
                    "candidate",
                    "",
                    bytes_before=sample[2] if sample is not None else 0,
                    codec=codec,
                )
            )
            continue
        outcome = compress_session(directory, codec=codec, now=moment)
        if outcome.compressed:
            result.compressed.append(outcome)
            result.bytes_before += outcome.bytes_before
            result.bytes_after += outcome.bytes_after
            _append_log(
                sessions_dir,
                {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "session": outcome.session,
                    "codec": outcome.codec,
                    "bytes_before": outcome.bytes_before,
                    "bytes_after": outcome.bytes_after,
                    "actor": actor,
                    "pid": os.getpid(),
                },
            )
        elif outcome.status == "failed":
            result.errors += 1
            skip(name, outcome.reason or "failed")
        else:
            skip(name, outcome.reason or "skipped")
    result.elapsed_s = time.monotonic() - started
    return result


def compress_from_config(
    config_manager: Any,
    config_dir: Path,
    *,
    live_dir: Path | None = None,
    now: float | None = None,
    dry_run: bool = False,
    actor: str = "startup",
    budget_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CompressionResult:
    """The startup entry point: read the policy, run it, record what it did.

    NEVER raises into the store-maintenance sequence. A broken policy read, a
    raising guard or an error inside the pass becomes a result with the reason
    on it and ``errors`` counted — the maintenance sequence has seven other
    passes to run and a leaving runtime to respect, so one pass's defect must
    not abort the boot-time work.
    """
    try:
        policy = policy_from_config(config_manager)
        if not policy.enabled and not dry_run:
            return CompressionResult(skipped="disabled")
        # The keep rule must agree with the delegated reap pass about which
        # parents are still active, so read the CONFIGURED window through that
        # pass's own reader (which clamps a hand-edited value) rather than the
        # module constant.
        from local_operator.session.cleanup import policy_from_config as cleanup_policy

        delegated_hours = cleanup_policy(config_manager).delegated_max_age_hours
        result = run_compression(
            config_dir,
            policy,
            live_dir=live_dir,
            now=now,
            dry_run=dry_run,
            actor=actor,
            budget_s=budget_s,
            should_stop=should_stop,
            delegated_window_hours=delegated_hours,
        )
    except Exception as exc:  # noqa: BLE001 — see the docstring
        logger.warning("transcript compression: pass failed: %s", exc, exc_info=True)
        return CompressionResult(errors=1, skipped=f"pass failed: {type(exc).__name__}")
    if result.compressed:
        logger.warning(
            "transcript compression: compressed %d of %d sessions (%d -> %d bytes; see %s)",
            len(result.compressed),
            result.scanned,
            result.bytes_before,
            result.bytes_after,
            Path(config_dir) / "sessions" / COMPRESSION_LOG_NAME,
        )
        write_last_compression(Path(config_dir) / "sessions", result, actor=actor)
    return result


def restore_for_write(directory: Path) -> bool:
    """Turn an archived transcript back into the plain file, in place.

    Used by ``Transcript.__init__`` (stage B) BEFORE it reads rows or creates the
    file: a runtime that is about to append needs the plain file, and must not
    discover "no transcript" and start a fresh empty one over a session whose
    history is sitting in an archive beside it.

    Verified/atomic, like the compression: decode into a same-directory tmp,
    require a clean stream end, fsync, publish with ``os.replace`` (so the plain
    name never names a partial file), stamp the ARCHIVE's mtime, then unlink the
    archive and drop any cache copy. ``True`` when a plain transcript exists
    afterwards; ``False`` when there is nothing to restore or it could not be
    decoded (the archive is then left exactly as it was).

    A quarantine left by a crashed compression is recovered first: it IS the
    plain file, parked under a dot-name, so decoding the archive beside it would
    rewrite bytes that are already on disk.
    """
    directory = Path(directory)
    _recover_quarantine(directory)
    plain = plain_path(directory)
    try:
        if plain.exists():
            return True
    except OSError:
        return False
    archive = archive_path(directory)
    if archive is None:
        return False
    codec = _codec_for(archive)
    if codec is None:
        logger.error("transcript compression: %s has an unknown archive form", archive.name)
        return False
    try:
        archive_stat = archive.stat()
    except OSError:
        return False
    tmp = directory / f".{PLAIN_FILENAME}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        _reserve(tmp)
        _decode_to_file(archive, codec, tmp)
        _fsync_file(tmp)
        os.utime(tmp, ns=(archive_stat.st_mtime_ns, archive_stat.st_mtime_ns))
        _publish(tmp, plain)
    except (OSError, ArchiveError) as exc:
        logger.error("transcript compression: cannot restore %s: %s", directory.name, exc)
        _unlink_file(tmp)
        return False
    # Every archive form goes, not only the one read: a second form would
    # otherwise shadow the restored plain file on the next read. The listing
    # sidecar goes with it — it describes a compressed form that no longer
    # exists, and the plain file is authoritative again.
    for name in ARCHIVE_FILENAMES:
        _unlink_file(directory / name)
    _unlink_file(directory / PREVIEW_SIDECAR_NAME)
    _fsync_dir(directory)
    _drop_cache(directory)
    return True

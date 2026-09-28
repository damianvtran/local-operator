"""The episode record's durability boundary: a sink a full disk cannot void.

WHY THIS EXISTS. On 2026-09-28 two session-arm episodes completed their whole
interaction and were then destroyed by a single sink write that met ENOSPC
mid-run:

* ``a1696-p1/task_013`` finished 46 turns (finish accepted through the
  completion gate), wrote a 64,863,846-byte ``events.jsonl``, took its score --
  and sealed ``status: failed, steps: 0`` because the seal path re-raised the
  sink's first ``OSError(28)`` instead of reporting what the run had done.
* ``a1696-p2/task_004`` ran 57 batches / 63 model calls ($1.6388), wrote
  79,773,126 bytes, scored ``partial_ppm 359700`` -- and sealed the same
  ``failed / steps: 0`` shape. Its ``score.json`` was on disk, orphaned,
  because the outcome that would have linked it was replaced by the failure.

The host was shared with ~25 sessions and free space swung between 188 MiB and
17 GiB while those runs were alive, so this is an operating condition rather
than a freak: **the work and the spend are committed long before the volume
fills, and the product of that spend must not die with the seal.**

The same durability work already exists one package over for the sealed bundle
(``evidence/store.py``'s writer poisons itself and the runner treats a publish
failure as best-effort), and for the desktop stores
(``session/store_failures.py``). This module is the session-arm pilot record's
half of that rule -- general in shape, so any long-running record writer can
adopt it, not benchmark-shaped.

FOUR GUARANTEES, each one a measured failure turned into a property:

1. PRE-FLIGHT HEADROOM. :class:`RecordSink` refuses to open when the volume
   cannot hold :data:`RECORD_EXPECTED_BYTES` plus its :data:`SEAL_RESERVE_BYTES`
   before ``launch`` -- so a hopeless volume costs a refusal, not a paid episode
   that discovers the wall at the seal.
2. A RESERVED SEAL MARGIN. The reserve is pre-allocated at open and released at
   the seal's moment of need (:meth:`RecordSink.seal_write`), which is what
   converts "dies at 99%" into "the record stops at 100% minus the margin, and
   the seal still lands on top of it".
3. LINE-COMPLETE, INCREMENTAL WRITES. Every record line is one append, and a
   failed write is truncated back to the last complete line; a truncate that
   itself fails (a dying volume) is not swallowed -- the failure sentence says
   the file ends in a torn fragment, because a reader cannot tell a cut that
   happened from one that did not. So every COMPLETE line on disk is parseable
   and a partial record is USABLE -- the exact property whose absence made
   ``steps=0`` unreadable as an artifact.
4. A LEGIBLE, CLASSIFIED FAILURE. :attr:`RecordSink.failure` carries a
   :class:`RecordSinkError` whose sentence separates "the environment ran out of
   room" (``out_of_room``, naming the volume and how many bytes did land) from
   "the record could not be written" for any other cause. A record write never
   raises into the run; only the pre-flight refusal and the seal artifacts do.

MEASUREMENTS BEHIND THE NUMBERS (all from the two runs above, read off the
sealed roots; they are the arithmetic the constants below rest on):

* completed records: 64,863,846 B (46 turns) and 79,773,126 B (63 calls)
  -- ``79,773,126 B = 76.07 MiB``, the larger one, is the "76 MB partial";
* record growth: ~1.3-1.4 MB per model turn (79.77 MB / 63, 64.86 MB / 46);
* seal artifacts actually written: ``score.json`` 322-328 B, ``outcome.json``
  488 B, one terminal record line < 1 KB;
* largest single record line observed: 0.84 MB (an ``agent_event``).

A DELIBERATE LIMIT, stated so it is not read as a guarantee: on a shared volume
the free space can be consumed by neighbours between the pre-flight check and
the seal. The floor is a refusal, not a lock; what protects a run that is
already going is the reserve (the seal always has room) and the line-complete
sink (the record never becomes unreadable). Neither makes the record infinite.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from local_operator.session.store_failures import SPACE_ERRNOS

#: The seal's reserved margin, pre-allocated at open and kept until the seal
#: needs it. Derived, not chosen: the seal artifacts measured 322-328 B
#: (``score.json``) + 488 B (``outcome.json``) + one terminal line < 1 KB, and
#: the largest single record line observed is 0.84 MB -- so 1 MiB covers the
#: measured seal with ~600x headroom and still covers one full-size line if the
#: seal has to append after the reserve is released. It is small next to the
#: ~65-80 MB a record actually reaches (1.3% of the larger measured record), so
#: holding it does not meaningfully move when the record meets the wall.
SEAL_RESERVE_BYTES = 1024 * 1024

#: The reserve's name inside the record root. Plain (not a dotfile) so a
#: look at the record directory shows what the space is being held for.
SEAL_RESERVE_NAME = "seal.reserve"

#: The record size pre-flight expects, and refuses to start without.
#: Derived: the two completed records measured 64.86 MB and 79.77 MB
#: (76.07 MiB) for 46-63 model turns, so 128 MiB is 1.68x the larger record
#: and 2.07x the smaller one -- ~54 MB of growth room, about 40 further turns
#: at the measured 1.3-1.4 MB/turn, before the size itself would refuse.
#: This is a FLOOR: it refuses only a volume that cannot hold
#: one full-length record plus the reserve (129 MiB total), because refusing a
#: run that could plausibly complete would waste the wall time the benchmark is
#: actually short of, while the reserve and the line-complete sink are what
#: cover the volume filling up mid-run (see the module docstring's limit).
RECORD_EXPECTED_BYTES = 128 * 1024 * 1024

#: How much of a reserve file to write per syscall while pre-allocating.
_RESERVE_CHUNK = 64 * 1024


class RecordSinkError(RuntimeError):
    """One record operation could not be performed, with its cause classified.

    ``sentence`` is the legible half: it names the path and separates the
    environment running out of room from every other cause, and states how many
    bytes of the record DID land. ``out_of_room`` is the machine half for a
    caller that wants to act (retry elsewhere, surface a disk warning), and
    ``preflight`` marks the refusals raised before anything was launched.
    """

    def __init__(
        self,
        sentence: str,
        *,
        path: Path,
        cause: BaseException | None = None,
        out_of_room: bool = False,
        preflight: bool = False,
        bytes_written: int | None = None,
    ) -> None:
        super().__init__(sentence)
        self.sentence = sentence
        self.path = Path(path)
        self.cause = cause
        self.out_of_room = out_of_room
        self.preflight = preflight
        self.bytes_written = bytes_written


def space_error(error: BaseException) -> bool:
    """Whether ``error`` is the volume (or a quota) having no room left.

    The one definition is ``session/store_failures.SPACE_ERRNOS`` -- the desktop
    ladder classifies the same fact, and a second copy of the set is how two
    answers to one question start disagreeing. ``None``/non-``OSError`` errors
    are never space errors: "the disk is full" is a claim that must be earned.
    """

    return isinstance(error, OSError) and error.errno in SPACE_ERRNOS


def _space_code(error: BaseException) -> str:
    code: int | None = getattr(error, "errno", None)
    return errno.errorcode.get(code, f"errno {code}") if code is not None else "no-room"


class _SinkCalls:
    """The syscall seam crash cutpoints are injected through.

    The same shape ``evidence/store.py`` uses for its writer: tests drive a
    failure at an exact syscall (a short write, an ENOSPC, an unlink that
    fails) without a monkeypatch race or a real full disk.
    """

    def write(self, fd: int, data: Any) -> int:
        return os.write(fd, data)

    def ftruncate(self, fd: int, length: int) -> None:
        os.ftruncate(fd, length)

    def unlink(self, path: str) -> None:
        os.unlink(path)

    def fsync(self, fd: int) -> None:
        os.fsync(fd)


_CALLS = _SinkCalls()


class RecordSink:
    """Append-only JSONL record for one run, durable under a full disk.

    Construction is the PRE-FLIGHT: it refuses when the volume holding ``path``
    cannot hold :data:`RECORD_EXPECTED_BYTES` plus :data:`SEAL_RESERVE_BYTES`
    (raising :class:`RecordSinkError` with ``preflight=True``), and it
    pre-allocates the reserve so the seal's writes have bounded room no matter
    how the rest of the run goes.

    :meth:`write` never raises for a storage failure: the first failure is
    classified into :attr:`failure`, the file is truncated back to its last
    complete line, and the run continues -- a torn record must stay a record
    of what happened, not the reason the paid work is lost.

    :meth:`seal_write` is for the two artifacts that make a record USABLE
    (``score.json``, ``outcome.json``): it spends the reserve when it must and
    raises when even the released reserve cannot make the artifact land, so the
    caller can state that in the outcome it still returns.

    The reserve's lifecycle is explicit, not implicit in ``close``: the arm
    releases it after the seal first, in a ``finally``, and a process killed
    before that leaves one 1 MiB ``seal.reserve`` inside the record root --
    small, and itself evidence the run was cut rather than sealed.
    """

    def __init__(
        self,
        path: Path,
        *,
        expected_bytes: int = RECORD_EXPECTED_BYTES,
        reserve_bytes: int = SEAL_RESERVE_BYTES,
        calls: _SinkCalls | None = None,
    ) -> None:
        self._path = Path(path)
        self._expected_bytes = int(expected_bytes)
        self._reserve_bytes = int(reserve_bytes)
        # A negative budget would make the pre-flight accept LESS room than the
        # reserve it is about to hold, which is nonsense a caller could inject
        # without seeing it: refuse it here rather than arm a weaker floor.
        if self._expected_bytes < 0 or self._reserve_bytes < 0:
            raise ValueError("expected_bytes and reserve_bytes must be non-negative")
        self._calls = calls if calls is not None else _CALLS
        self._fd: int | None = None
        self._offset = 0
        self._lines = 0
        self._closed = False
        self._failure: RecordSinkError | None = None
        self._failures = 0

        try:
            # The record root is created here rather than by the caller so this
            # refusal path cannot be preceded by a traceback out of a mkdir.
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            raise RecordSinkError(
                f"the record could not be started: {self._path.parent} could not be "
                f"created: {type(error).__name__}: {error}",
                path=self._path,
                cause=error,
                out_of_room=space_error(error),
                preflight=True,
                bytes_written=0,
            ) from error

        # PRE-FLIGHT. A root that cannot be measured proceeds: a failed stat is
        # not evidence of a full disk, and the write path still classifies what
        # it meets. A measured volume smaller than the budget refuses HERE,
        # before launch, so the refusal costs no allocation.
        budget = self._expected_bytes + self._reserve_bytes
        free = self._free_bytes()
        if free is not None and free < budget:
            raise RecordSinkError(
                f"refusing to start: {free} bytes free on the volume holding "
                f"{self._path}, below the {self._expected_bytes} bytes a record is "
                f"expected to need plus its {self._reserve_bytes}-byte seal reserve; "
                "free space on that volume and run again",
                path=self._path,
                out_of_room=True,
                preflight=True,
                bytes_written=0,
            )

        self._reserve_path = self._path.parent / SEAL_RESERVE_NAME
        self._reserve_held = False
        if self._reserve_bytes > 0:
            try:
                self._create_reserve()
                self._reserve_held = True
            except OSError as error:
                # A half-written reserve is this refusal's own litter; removing
                # it is a different operation from the write that failed, so the
                # cleanup is safe to attempt and suppressed if it fails too.
                try:
                    self._reserve_path.unlink()
                except OSError:
                    pass
                raise RecordSinkError(
                    f"refusing to start: the {self._reserve_bytes}-byte seal reserve "
                    f"could not be created at {self._reserve_path}: "
                    f"{type(error).__name__}: {error}",
                    path=self._path,
                    cause=error,
                    out_of_room=space_error(error),
                    preflight=True,
                    bytes_written=0,
                ) from error

        try:
            fd = os.open(
                self._path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except OSError as error:
            if self._reserve_held:
                self._release_reserve()
            raise RecordSinkError(
                f"the record could not be started at {self._path}: "
                f"{type(error).__name__}: {error}",
                path=self._path,
                cause=error,
                out_of_room=space_error(error),
                preflight=True,
                bytes_written=0,
            ) from error
        self._fd = fd
        self._offset = os.fstat(fd).st_size

    # -- state ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def failure(self) -> RecordSinkError | None:
        """The first storage failure this record met, classified; else ``None``.

        Only the FIRST is kept: one cause explains the truncated tail, and
        later failures on a full volume are the same volume saying the same
        thing. ``failures`` counts them for a caller that wants the scale.
        """

        return self._failure

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def bytes_written(self) -> int:
        """Bytes of COMPLETE lines on disk -- never a torn partial line."""

        return self._offset

    @property
    def lines_written(self) -> int:
        return self._lines

    @property
    def closed(self) -> bool:
        return self._closed

    # -- the record ----------------------------------------------------------

    def write(self, kind: str, payload: Mapping[str, Any]) -> None:
        """Append one record line. A storage failure is captured, never raised.

        The line is serialized first (a payload that cannot be serialized is
        itself a failure to record, classified like any other), then appended
        in ONE logical write, and the file's offset advances only after the
        line is fully on disk. A failure truncates the file back to the last
        complete line -- and a fixup that itself fails leaves a torn tail the
        failure sentence NAMES -- so a reader gets either a clean cut or a
        stated torn tail, never a silent one: the property that makes a
        partial record analyzable instead of a void.
        """

        if self._closed:
            raise RecordSinkError(
                f"the record sink for {self._path} is closed; the {kind!r} line "
                "arrived after the record stopped",
                path=self._path,
                bytes_written=self._offset,
            )
        try:
            line = (
                json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str).encode(
                    "utf-8"
                )
                + b"\n"
            )
        except Exception as error:  # noqa: BLE001 - a record write must never break the run
            self._note_failure(error, last_complete=self._offset)
            return
        last_complete = self._offset
        try:
            self._write_all(self._fd, line)
        except Exception as error:  # noqa: BLE001 - see above
            self._note_failure(error, last_complete=last_complete)
            return
        self._offset += len(line)
        self._lines += 1

    def seal_write(self, target: Path, text: str) -> None:
        """Write one seal artifact, spending the reserve when it must.

        These are the writes that make a record usable -- the score link and
        the outcome -- and they are bounded (measured under 1 KB each), which
        is what lets the reserve cover them exactly. The reserve is spent
        LAZILY, on the first artifact that meets no-room, so a run that never
        fills the volume keeps its margin to the end; the write is attempted
        first and the reserve released only when the attempt actually failed,
        which keeps the release-to-write window to the retry itself.

        Raises :class:`RecordSinkError` when the artifact cannot land even so.
        The caller still holds the outcome and prints it, so the failure costs
        the disk copy, not the run.
        """

        target = Path(target)
        data = text.encode("utf-8")
        while True:
            try:
                self._write_artifact(target, data)
                return
            except OSError as error:
                if space_error(error) and self._release_reserve():
                    continue
                raise self._seal_failure(target, error) from error

    def release_reserve(self) -> bool:
        """Free the reserved seal margin now. Idempotent.

        Returns whether this call freed it. Called by the owner at the end of
        the run (``finally``) and internally by :meth:`seal_write` when an
        artifact needs the room.
        """

        return self._release_reserve()

    def close(self) -> None:
        """Close the record file. The reserve is deliberately NOT released here."""

        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                os.close(fd)
            finally:
                self._closed = True
        else:
            self._closed = True

    # -- internals -----------------------------------------------------------

    def _free_bytes(self) -> int | None:
        try:
            return shutil.disk_usage(self._path.parent).free
        except OSError:
            return None

    def _create_reserve(self) -> None:
        chunk = bytes(_RESERVE_CHUNK)
        with self._reserve_path.open("wb") as handle:
            remaining = self._reserve_bytes
            while remaining > 0:
                block = chunk if remaining >= len(chunk) else chunk[:remaining]
                handle.write(block)
                remaining -= len(block)
            handle.flush()

    def _release_reserve(self) -> bool:
        if not self._reserve_held:
            return False
        try:
            self._calls.unlink(str(self._reserve_path))
        except FileNotFoundError:
            # Already gone (another path released it, or it never landed): the
            # space is not ours any more, and a second report of "freed" would
            # be a lie the retry would trust.
            self._reserve_held = False
            return False
        except OSError:
            return False
        self._reserve_held = False
        return True

    def _write_all(self, fd: int | None, data: bytes) -> None:
        if fd is None:  # pragma: no cover - write() refuses a closed sink first
            raise RecordSinkError(f"the record sink for {self._path} is closed", path=self._path)
        view = memoryview(data)
        interrupted = 0
        while view:
            try:
                written = self._calls.write(fd, view)
            except InterruptedError:
                interrupted += 1
                if interrupted >= 16:
                    raise
                continue
            interrupted = 0
            if written <= 0:
                raise OSError(errno.EIO, "the record write made no progress")
            view = view[written:]

    def _write_artifact(self, target: Path, data: bytes) -> None:
        """One atomic artifact write: same-directory temp, then replace.

        Atomic because a torn ``outcome.json`` is the same void this module
        exists to remove, one layer up; the discipline is
        ``declare_action_server``'s and ``mcp/config.py``'s.
        """

        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0),
                0o600,
            )
            try:
                self._write_all(fd, data)
                self._calls.fsync(fd)
            finally:
                os.close(fd)
            temporary.replace(target)
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    def _note_failure(self, error: BaseException, *, last_complete: int) -> None:
        # The truncate runs FIRST so the failure sentence can tell the truth
        # about it: the fixup is best effort -- a dying volume can refuse it
        # too -- and a file left with a torn tail must SAY so rather than read
        # as a clean cut (review round 1, R1-N2).
        fd = self._fd
        torn_tail = False
        if fd is not None:  # pragma: no cover - write() refuses a closed sink first
            try:
                self._calls.ftruncate(fd, last_complete)
            except Exception:  # noqa: BLE001 - the fixup is best effort; named below
                torn_tail = True
        if self._failure is None:
            if space_error(error):
                sentence = (
                    f"the volume holding {self._path} ran out of room "
                    f"({_space_code(error)}) after {last_complete} bytes; the record "
                    "is complete through its last finished line"
                )
            else:
                sentence = (
                    f"the record file {self._path} could not be written: "
                    f"{type(error).__name__}: {error}"
                )
            if torn_tail:
                sentence += (
                    "; the trailing partial line could not be removed, so the file "
                    "ends in a torn fragment a reader must drop"
                )
            self._failure = RecordSinkError(
                sentence,
                path=self._path,
                cause=error,
                out_of_room=space_error(error),
                bytes_written=last_complete,
            )
        self._failures += 1

    def _seal_failure(self, target: Path, error: OSError) -> RecordSinkError:
        if space_error(error):
            if self._reserve_held:
                note = f"and its {self._reserve_bytes}-byte seal reserve could not be released"
            elif self._reserve_bytes > 0:
                note = "with the seal reserve already spent"
            else:
                note = "and no seal reserve was configured for this sink"
            sentence = (
                f"the seal artifact {target} could not be written: the volume holding "
                f"{target} ran out of room ({_space_code(error)}) {note}"
            )
            return RecordSinkError(
                sentence,
                path=target,
                cause=error,
                out_of_room=True,
                bytes_written=self._offset,
            )
        return RecordSinkError(
            f"the seal artifact {target} could not be written: " f"{type(error).__name__}: {error}",
            path=target,
            cause=error,
            out_of_room=False,
            bytes_written=self._offset,
        )

"""Retain the evaluator's own scoring-path output as part of the score evidence.

WHY this module exists. OSWorld's per-task evaluators compute the values that
decide a score and then throw almost all of them away: ``task_002`` computes
four checkpoint booleans and returns their mean; ``task_016`` computes
``email_avg``/``linkedin_avg`` and returns only the pairing's mean;
``task_098`` writes its normalised results into ``env.cache_dir`` and returns a
scalar. The adapter's score detail is the evaluator's RETURN VALUE, so a
``0.00%`` row cannot be distinguished from a ``0.00%`` row whose evaluator
bailed out internally. That single distinction decides "the apparatus is
broken" versus "the agent really failed", and without this capture the
archived bundles cannot answer it (the worker's stdout/stderr and the episode
cache were both discarded at scoring time).

WHAT is captured, on the scoring path only:

* everything the evaluator writes to ``sys.stdout`` (its ``print`` lines) and
  to ``sys.stderr`` (its direct writes);
* the log records its modules emit (``desktopenv.*``, ``desktop_env.*``, the
  judge's ``llm_metrics``, and the ``osworld_task_<id>`` names the adapter
  register task modules under -- see ``EVALUATOR_LOG_PREFIXES``);
* the state the evaluator fetched, as a bounded manifest of the task's cache
  directory with the text of the small files in it.

CONSTRAINTS, each load-bearing:

* NOTHING an evaluator returns changes, and no score changes. This module
  collects bytes the evaluator already produced; ``scoring.score_to_artifact``
  still reads the score out of the raw return exactly as before.
* No benchmark task file is touched. Upstream evaluators are called unmodified
  and un-rewritten, so the apparatus digest is unchanged and rows stay
  comparable with published ones.
* The capture is ADDITIVE. An evaluator that emits nothing and fetched nothing
  produces no payload at all, and the score detail stays byte-identical to what
  it was before this module existed.
* The captured bytes are bounded here (per stream and per file) and validated
  again in :mod:`lop_osworld_v2_adapter.scoring` against the score-detail
  limits. Over-budget content is CUT, and every cut is reported as a count on
  the artifact rather than silently absorbed.
* The worker's own output is not diverted. Both the stream sinks and the log
  handler tee back to the original destination, so what the supervisor's
  stderr tail and the process's stdout carried before this module existed,
  they still carry.

One deliberate gap, documented rather than papered over: a write to
``sys.stdout.buffer`` (raw bytes) is forwarded untouched and is NOT captured,
because the evidence artifact is text and decoding arbitrary element writes
would be a guess. Upstream's scoring path writes text.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, TextIO

#: The logger namespaces whose records are retained. Closed on purpose: a
#: handler attached to the root logger would also collect third-party INFO
#: noise (botocore and friends) into sealed evidence, and widening the ROOT
#: level would emit that noise on the worker's stderr too. Records outside this
#: set are untouched: WARNING and above still reach stderr exactly as before.
#:
#: ``desktopenv`` covers every task module (``desktopenv.task002``) and OSWorld
#: itself; ``desktop_env`` covers the metric modules that log through
#: ``getLogger(__name__)``; ``llm_metrics`` is the judge's undotted name; and
#: ``osworld_task_`` is what ``vendor_bridge.instantiate_task`` names a task
#: module, so a task module logging ``__name__`` lands there.
EVALUATOR_LOG_PREFIXES: tuple[str, ...] = (
    "desktopenv",
    "desktop_env",
    "llm_metrics",
    "osworld_task_",
)

#: Schema tag on the retained block, so a reader can tell which shape it holds.
DIAGNOSTICS_SCHEMA = "lop-evaluator-diagnostics-v1"

#: Bounds, chosen so a maximal capture always fits the score-detail limits
#: (``scoring.MAX_SCORE_DETAIL_BYTES`` is 1 MiB, ``..._NODES`` 100_000) instead
#: of being refused there. Measured on the true worst case -- both stream rings
#: full of NUL characters, the most expensive input canonical JSON accepts (six
#: bytes each), plus the aggregate file text -- the artifact is 786_975 bytes
#: against the 1 MiB ceiling; with the printable characters a real evaluator
#: emits (quotes and newlines are the next dearest at two bytes) the same
#: maximal capture is 262_687 bytes. ``scoring``'s refusal marker is therefore a
#: belt, not the path any captured output takes.
MAX_STREAM_CHARS = 32 * 1024
MAX_RECORD_CHARS = 2_000
MAX_FETCHED_ENTRIES = 128
MAX_FETCHED_DIRECTORIES = 256
MAX_FETCHED_TEXT_CHARS = 8 * 1024
MAX_FETCHED_TEXT_TOTAL_CHARS = 64 * 1024

#: Why a fetched file's content is absent, as a closed set. ``over_budget``
#: is the aggregate budget running out (the cut ``text_cut`` reports);
#: ``too_large`` is a single file over the per-file bound, which cuts nothing
#: the aggregate had room for.
_OMITTED_OVER_BUDGET = "over_budget"
_OMITTED_TOO_LARGE = "too_large"
_OMITTED_NOT_UTF8 = "not_utf8"
_OMITTED_BINARY = "binary"
_OMITTED_SYMLINK = "symlink"
_OMITTED_UNREADABLE = "unreadable"


def _evidence_text(text: str) -> str:
    """``text`` as something canonical UTF-8 JSON can carry.

    A lone surrogate (a mis-decoded path or guest payload) makes the whole
    score detail unencodable, and the score-detail contract refuses such bytes
    as a protocol violation. Refusing the SCORE over one bad character in a
    diagnostic would trade the number for the explanation, so the character is
    replaced instead -- and only when the text actually holds one.
    """

    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "replace").decode("utf-8")
    return text


class _BoundedTextRing:
    """The last ``limit`` characters added, and how many were dropped.

    The TAIL is kept, not the head, because a scoring path explains itself at
    its end (the summary line, the traceback, the final checkpoint dump) --
    upstream's own bounded diagnostic tail makes the same choice.

    Locked because upstream's evaluators spawn threads (the screenshot
    controller, the model client) and a diagnostic line can be written from a
    worker thread while this one appends. The supervisor's own ``_Tail`` holds
    a lock for the same reason.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: deque[str] = deque()
        self._kept = 0
        self._dropped = 0
        self._wrote = False
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._wrote = True
            self._chunks.append(text)
            self._kept += len(text)
            excess = self._kept - self._limit
            # Drop whole chunks while an older one can carry the whole excess.
            while excess > 0 and len(self._chunks) > 1:
                first = self._chunks.popleft()
                self._kept -= len(first)
                self._dropped += len(first)
                excess = self._kept - self._limit
            # The newest chunk alone can still be over the limit: keep its tail.
            if excess > 0:
                first = self._chunks[0]
                self._chunks[0] = first[excess:]
                self._kept -= excess
                self._dropped += excess

    @property
    def wrote(self) -> bool:
        return self._wrote

    @property
    def dropped(self) -> int:
        return self._dropped

    def text(self) -> str:
        with self._lock:
            kept = "".join(self._chunks)
        return _evidence_text(kept)


class _BoundedStreamSink(io.TextIOBase):
    """A bounded, tee'ing stand-in for ``sys.stdout``/``sys.stderr``.

    Everything written lands in the ring AND is written on to the stream it
    replaced, so the worker's own output -- which the supervisor drains into
    its stderr tail and which the parent reads on a failure path -- is
    unchanged by the capture. A failed write propagates exactly as it would
    have without this object in the way.
    """

    def __init__(self, original: TextIO | None, limit: int = MAX_STREAM_CHARS) -> None:
        self._original = original
        self._ring = _BoundedTextRing(limit)

    @property
    def original(self) -> TextIO | None:
        return self._original

    @property
    def ring(self) -> _BoundedTextRing:
        return self._ring

    # ``print`` looks up ``sys.stdout`` per call, so replacing the attribute is
    # what routes it here; ``TextIOBase`` supplies the rest of the protocol.
    def write(self, text: str) -> int:
        if not text:
            return 0
        self._ring.add(text)
        if self._original is not None:
            self._original.write(text)
        return len(text)

    def writelines(self, lines: Any) -> None:  # type: ignore[override]
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        if self._original is not None:
            self._original.flush()

    def record(self, text: str) -> None:
        """Store text that was already emitted elsewhere (a log record).

        Separate from :meth:`write` because a retained record must not be
        teed: whether the record still reaches the process's stderr is decided
        by the logging machinery, not by this sink (see
        :meth:`_EvaluatorLogHandler.emit`).
        """

        self._ring.add(text)

    @property
    def buffer(self) -> Any:  # pragma: no cover - element writes are not captured
        """Forward raw element writes untouched.

        ``sys.stdout.buffer`` is a real API: handing back the replaced
        stream's own buffer keeps a byte-level writer working exactly as it
        did, at the documented cost that those bytes are not retained.
        """

        return getattr(self._original, "buffer", None)


def _chain_has_other_handler(name: str, own: logging.Handler) -> bool:
    """Whether any OTHER logger on ``name``'s chain would emit the record.

    Mirrors ``logging.Logger.callHandlers``' walk: ``lastResort`` only fires
    when that walk finds no handler at all. This capture's own handler is
    attached to the root logger, so it MUST be excluded -- counting it would
    make every record look handled and silently divert the records that
    previously left through ``lastResort``.
    """

    current = name
    while True:
        logger = logging.getLogger(current)
        if any(handler is not own for handler in logger.handlers):
            return True
        if not logger.propagate or not current:
            return False
        current = current.rpartition(".")[0]


def _is_evaluator_logger(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in EVALUATOR_LOG_PREFIXES)


class _EvaluatorLogHandler(logging.Handler):
    """Retains evaluator log records without changing where they go.

    A handler attached to the root logger is ADDITIVE for every log record --
    logging calls each handler on the propagation chain -- so the judge capture
    in ``providers.aws`` and any handler upstream installs still see what they
    saw. The one behaviour a root handler does change is ``logging.lastResort``,
    which fires only when NOTHING has a handler; a record that previously
    reached stderr that way is therefore written back to the stream by
    :meth:`_tee_if_only_last_resort_would_emit`, preserving the worker's stderr
    byte-for-byte.
    """

    def __init__(self, sink: _BoundedStreamSink) -> None:
        super().__init__(logging.NOTSET)
        self._sink = sink
        self._formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        # ``lastResort`` carries no formatter, so what it writes is the plain
        # message plus the handler terminator. Formatting the tee the same way
        # is what keeps the worker's stderr output identical.
        self._plain = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if _is_evaluator_logger(record.name):
                self._sink.record(_bounded_record(self._formatter, record))
            self._tee_if_only_last_resort_would_emit(record)
        except Exception:  # noqa: BLE001 - a capture must never break scoring
            # ``Handler.handleError`` would print "--- Logging error ---" onto
            # the worker's stderr, which is output the evaluator never
            # produced. Diagnostics are dropped instead; the score is not at
            # risk either way.
            return

    def _tee_if_only_last_resort_would_emit(self, record: logging.LogRecord) -> None:
        resort = logging.lastResort
        if resort is None or record.levelno < resort.level:
            return
        if _chain_has_other_handler(record.name, self):
            return
        stream = self._sink.original
        if stream is None:
            return
        stream.write(self._plain.format(record) + "\n")


def _bounded_record(formatter: logging.Formatter, record: logging.LogRecord) -> str:
    """:meth:`_EvaluatorLogHandler`'s formatted record, per-record bounded.

    The stream ring already bounds the total; this stops one pathological
    record (a request body, a dumped file) from evicting every other line
    before it, and labels the cut so the artifact never reads as complete.
    """

    line = formatter.format(record)
    if len(line) <= MAX_RECORD_CHARS:
        return line + "\n"
    return f"{line[:MAX_RECORD_CHARS]} [truncated {len(line) - MAX_RECORD_CHARS} chars]\n"


def _evaluator_loggers() -> list[logging.Logger]:
    """Every existing logger under :data:`EVALUATOR_LOG_PREFIXES`, plus the
    dotted prefixes themselves.

    Both halves are needed. A dotted prefix widened as a logger is how a
    ``desktopenv.*`` module imported DURING scoring still inherits INFO through
    its parent. An undotted name like ``osworld_task_002`` is nobody's parent,
    so the logger itself has to be found and widened -- which works because the
    task module was imported at ``reset_start``, before scoring begins.

    Deduplicated by name: the prefix loggers and the registry walk overlap, and
    a logger appearing twice would be recorded (and restored) twice, leaving
    the restoration order to decide its final level.
    """

    by_name: dict[str, logging.Logger] = {}
    for prefix in EVALUATOR_LOG_PREFIXES:
        by_name[prefix] = logging.getLogger(prefix)
    for candidate in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(candidate, logging.Logger) and _is_evaluator_logger(candidate.name):
            by_name.setdefault(candidate.name, candidate)
    return list(by_name.values())


def _widen_evaluator_loggers() -> list[tuple[logging.Logger, int]]:
    """Let the evaluator's INFO lines through for the capture window.

    OSWorld never configures logging, so the root logger's WARNING gate is what
    decides whether a record EXISTS: ``logger.info("Task002 partials: ...")`` is
    dropped before any handler can see it, which is exactly why the archived
    runs could not be diagnosed. Raising the level is what upstream's own
    runner does for the same reason (``lib_run_single.py`` sets its per-task
    logger to DEBUG so the evaluator's partials reach a file).

    Widening the gate changes NO upstream control flow: the pinned corpus
    contains no ``isEnabledFor`` call (checked over ``desktop_env/`` and
    ``evaluation_examples/``), so nothing branches on whether a record exists --
    only whether it is emitted. Levels are restored on exit; no other logging
    configuration is touched.
    """

    widened: list[tuple[logging.Logger, int]] = []
    for logger in _evaluator_loggers():
        if logger.level <= logging.INFO and logger.getEffectiveLevel() <= logging.INFO:
            continue
        widened.append((logger, logger.level))
        logger.setLevel(logging.INFO)
    return widened


def _restore_logger_levels(widened: list[tuple[logging.Logger, int]]) -> None:
    for logger, level in widened:
        logger.setLevel(level)


def _snapshot_fetched_state(directory: Path | None) -> dict[str, Any] | None:
    """A bounded manifest of the state the evaluator fetched for this task.

    ``None`` when the directory does not exist, which is what an evaluator that
    read nothing leaves behind -- and what keeps the additive contract: no
    directory, no block, no byte change in the sealed detail.
    """

    if directory is None:
        return None
    try:
        if not directory.is_dir():
            return None
        found: list[Path] = []
        directories = 0
        directories_cut = False
        for current, dirnames, filenames in os.walk(directory, followlinks=False):
            directories += 1
            if directories > MAX_FETCHED_DIRECTORIES:
                directories_cut = True
                break
            dirnames.sort()
            filenames.sort()
            for name in dirnames:
                # A symlinked directory is not descended into, so it would
                # otherwise vanish from the manifest entirely; naming it is
                # what makes "no entry" mean "nothing there".
                if (Path(current) / name).is_symlink():
                    found.append(Path(current) / name)
            for name in filenames:
                found.append(Path(current) / name)
            if len(found) > MAX_FETCHED_ENTRIES:
                break
        entries_cut = len(found) > MAX_FETCHED_ENTRIES or directories_cut
        entries: list[dict[str, Any]] = []
        text_total = 0
        text_cut = False
        for path in sorted(found[:MAX_FETCHED_ENTRIES]):
            entry = _fetched_entry(path, directory, text_total)
            if entry is None:
                continue
            text = entry.get("text")
            if isinstance(text, str):
                text_total += len(text)
            if entry.get("text_omitted") == _OMITTED_OVER_BUDGET:
                text_cut = True
            entries.append(entry)
        if not entries:
            # An existing but EMPTY task cache is not state the evaluator
            # fetched, and upstream creates this directory during setup for
            # every task. Reporting an empty manifest would change the detail
            # bytes of an episode whose evaluator said nothing, which is
            # exactly what the additive contract forbids.
            return None
        return {"entries": entries, "entries_cut": entries_cut, "text_cut": text_cut}
    except OSError:
        # A cache directory that vanished or is unreadable is not a scoring
        # failure: the score detail simply carries no fetched-state block.
        return None


def _fetched_entry(path: Path, directory: Path, text_total: int) -> dict[str, Any] | None:
    try:
        relative = path.relative_to(directory).as_posix()
    except ValueError:
        return None
    if path.is_symlink():
        # Never followed: a fetched symlink can point outside the cache, and
        # the file it names is not state the evaluator fetched under this root.
        return {"path": relative, "text_omitted": _OMITTED_SYMLINK}
    try:
        size = path.stat().st_size
    except OSError:
        return {"path": relative, "text_omitted": _OMITTED_UNREADABLE}
    entry: dict[str, Any] = {"path": relative, "bytes": size}
    if size > MAX_FETCHED_TEXT_CHARS:
        entry["text_omitted"] = _OMITTED_TOO_LARGE
        return entry
    if text_total >= MAX_FETCHED_TEXT_TOTAL_CHARS:
        entry["text_omitted"] = _OMITTED_OVER_BUDGET
        return entry
    try:
        data = path.read_bytes()
    except OSError:
        entry["text_omitted"] = _OMITTED_UNREADABLE
        return entry
    if b"\x00" in data:
        entry["text_omitted"] = _OMITTED_BINARY
        return entry
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        entry["text_omitted"] = _OMITTED_NOT_UTF8
        return entry
    entry["text"] = _evidence_text(text)
    return entry


class EvaluatorCapture:
    """The scoring-path capture window.

    Entered around the one call that runs the evaluator. Process-global state
    (``sys.stdout``/``sys.stderr``, one root-logger handler and a set of
    evaluator logger levels) is installed for the window, which is safe here
    because the worker serves one adapter operation at a time and scoring is
    the last of them. The fetched-state snapshot is taken on exit, so it sees
    the state the evaluator left behind rather than the state it started from.
    """

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._stdout: _BoundedStreamSink | None = None
        self._stderr: _BoundedStreamSink | None = None
        self._handler: _EvaluatorLogHandler | None = None
        self._widened: list[tuple[logging.Logger, int]] = []
        self._fetched: dict[str, Any] | None = None

    def __enter__(self) -> "EvaluatorCapture":
        self._stdout = _BoundedStreamSink(sys.stdout)
        self._stderr = _BoundedStreamSink(sys.stderr)
        self._widened = _widen_evaluator_loggers()
        self._handler = _EvaluatorLogHandler(self._stderr)
        logging.getLogger().addHandler(self._handler)
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Restore first and unconditionally: leaving ``sys.stdout`` pointing at
        # a sink would outlive the capture and corrupt every later write. None
        # of these four steps can raise on a well-formed interpreter.
        if self._stdout is not None:
            sys.stdout = self._stdout.original if self._stdout.original is not None else sys.stdout
        if self._stderr is not None:
            sys.stderr = self._stderr.original if self._stderr.original is not None else sys.stderr
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
        _restore_logger_levels(self._widened)
        self._fetched = _snapshot_fetched_state(self._cache_dir)
        return None

    def payload(self) -> dict[str, Any] | None:
        """The retained block, or ``None`` when there is nothing to retain.

        ``None`` is the additive contract: no payload means
        ``score_to_artifact`` stages exactly the bytes it staged before this
        module existed.
        """

        assert self._stdout is not None and self._stderr is not None
        block: dict[str, Any] = {"schema": DIAGNOSTICS_SCHEMA}
        stdout = _stream_block(self._stdout)
        if stdout is not None:
            block["stdout"] = stdout
        stderr = _stream_block(self._stderr)
        if stderr is not None:
            block["stderr"] = stderr
        if self._fetched is not None:
            block["fetched_state"] = self._fetched
        return block if len(block) > 1 else None


def _stream_block(sink: _BoundedStreamSink) -> dict[str, Any] | None:
    """One stream's captured text, cut count included when it was bounded."""

    if not sink.ring.wrote:
        return None
    block: dict[str, Any] = {"text": sink.ring.text()}
    if sink.ring.dropped:
        block["truncated"] = sink.ring.dropped
    return block


def capture_evaluator_diagnostics(*, cache_dir: Path | None = None) -> EvaluatorCapture:
    """The capture window for one scoring call. See :class:`EvaluatorCapture`."""

    return EvaluatorCapture(cache_dir=cache_dir)

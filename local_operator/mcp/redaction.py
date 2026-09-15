"""Scrub resolved MCP credentials out of transport diagnostics.

**Why this exists rather than the transcript filter.** A resolved ``${NAME}``
value reaches sinks that run BEFORE anything model-visible: the stdio child
echoes it to stderr and :class:`~local_operator.mcp.manager.McpServerStderr`
logs and retains the line, the connect error quotes that tail, and the MCP SDK
logs its own transport failures with the request that carried the header.
Review BI-1 reproduced exactly that — an invalid-token diagnostic put the
sentinel in the raised exception and in two ``local_operator.mcp`` log records.
Registering the value with the session redactor alone is too late for all three,
because the log record is already written by then.

**Two layers, and the first one is the control.** Every MCP sink scrubs at the
SOURCE: the stderr pump scrubs bytes before they are split into lines
(a credential can straddle a chunk boundary, so the scrub has to be
stream-aware), and ``report_failure``/``explain`` scrub the text they build.
That is what the canary probe exercises. The logging filter below is defence in
depth for records this package does not write itself — the SDK's and httpx's.

**The filter is attached to the ROOT logger's HANDLERS, not to named loggers.**
Filters do not propagate: a filter on ``local_operator.mcp`` is never consulted
for a record logged through ``local_operator.mcp.server.<name>``, and a list of
SDK logger names is a denylist that goes stale the moment the SDK renames a
module — silently, which is the failure mode this whole finding is about.
Handlers see every propagated record regardless of which logger emitted it, so
attaching there needs no list and cannot miss a logger nobody remembered.

Handlers installed AFTER a value is registered would not carry the filter, so
:func:`register` re-scans on every call and :func:`attach` is exported for the
logging setup to call once it has built its handlers.

**What the scrubbers will and will not rewrite.** Only values at or above
:data:`MIN_SCRUBBED_LENGTH`, because a byte-for-byte replacement of a shorter
one corrupts ordinary text process-wide rather than protecting anything; and
registration is REVERSIBLE (:func:`unregister`) so a test or a short-lived
resolver cannot leave the process scrubbing a value whose reason to exist has
gone away. Both are there because the first cut of this module was global,
append-only and unbounded, and that combination was measured corrupting
unrelated diagnostics (agent review R-1 / QA Q1).

This registers values for SCRUBBING ONLY. It never adds them to a credential
map, so nothing here makes a value injectable into a child environment or
readable by a tool — the distinction
:class:`~local_operator.variables.VariableStore` draws between ``_credentials``
and ``_redactions``, used here for the second.
"""

from __future__ import annotations

import codecs
import logging
import threading
from contextlib import suppress

from local_operator.variables import VariableStore

#: Redaction-only store: values to scrub, never to inject or advertise.
#: ``env={}`` so it never reads the process environment.
_STORE = VariableStore(env={})

#: Guards the store against a resolve running on a worker thread while a log
#: record is being scrubbed on the event loop.
_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

#: Shortest value the scrubbers will rewrite.
#:
#: **Why a floor, and why here.** The scrubbers replace a registered value
#: byte-for-byte wherever it appears, so below this length a value cannot be told
#: apart from ordinary prose and the rewrite is damage rather than protection —
#: and it is process-wide damage, because the filter below sees every record and
#: the sinks below build user-visible text. Measured on a three-character value
#: registered by a store test: an unrelated provider warning became ``expected
#: [redacted] of minimal, low, medium, high, xhigh, max`` and a rendered MCP
#: failure became ``ne[redacted]rk: cannot reach …`` (agent review R-1 / QA Q1).
#: Ordinary diagnostics are full of one-to-seven-character words, so no shorter
#: bound works; the same collision is why a WORD-shaped credential of any length
#: still redacts that word, which is the cost of scrubbing at all.
#:
#: The residual is named rather than hidden: a credential shorter than this is
#: NOT scrubbed from diagnostics, and :func:`register` says so at DEBUG (never
#: the value itself). In practice MCP credentials are long opaque tokens; the
#: alternative — rewriting every occurrence of a three-character value — protects
#: nothing and corrupts every log line in the process.
MIN_SCRUBBED_LENGTH = 8


def scrub(text: str) -> str:
    """``text`` with every registered MCP credential replaced by ``[redacted]``."""
    with _LOCK:
        values = _STORE.redaction_values()
    if not values:
        return text
    from local_operator.variables import redact_secret_values

    return redact_secret_values(text, values)


def values() -> list[str]:
    """Registered values, for a stream-aware filter."""
    with _LOCK:
        return _STORE.redaction_values()


class StderrRedactor:
    """Scrub a streaming stderr feed, holding back only a PARTIAL line.

    **Why this is not ``_PipeRedactor``.** That class is right for a bash pipe,
    where output keeps flowing: it holds back a fixed lookbehind window
    regardless of line structure, and its caller never needs a line promptly.
    A stdio child's stderr is different — the sinks here are LINE-oriented (the
    pump splits on ``\\n``, and the retained tail is quoted into an error
    message) and a child may print one line and then go quiet for the rest of
    the session. Reusing the pipe redactor therefore withheld a line's own
    newline inside its holdback, so that line was never handed to
    ``McpServerStderr.feed`` until the stream ended, and a server whose startup
    line is its only output (a PID announcement, a readiness banner, an
    ``initialize`` diagnostic) reported nothing at all. That is a real
    regression, not a test artefact: it reproduced as hangs across
    ``tests/unit/mcp`` once any value was registered.

    The rule here: **release every complete line, hold back only the trailing
    run of bytes with no newline after it, then widen the cut backwards** so a
    value that spans the release boundary is never split across two releases.
    The cost of the weaker holdback is stated rather than hidden: a credential
    that itself contains a newline cannot be scrubbed by a line-oriented sink,
    which was already true of the per-line ``feed`` this replaces.
    """

    def __init__(self, values: list[str]) -> None:
        self._set(values)
        self._pending = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _set(self, values: list[str]) -> None:
        self._secrets = sorted({value for value in values if value}, key=len, reverse=True)

    def feed(self, chunk: bytes, *, final: bool = False) -> str:
        text = self._pending + self._decoder.decode(chunk, final=final)
        if final:
            cut = len(text)
        elif not self._secrets:
            # Nothing to protect: hold nothing back, so a child's line is fed the
            # instant it arrives.
            cut = len(text)
        else:
            # Release through the last newline; inside a trailing partial line,
            # hold back only the bytes that could still be a value prefix.
            newline = text.rfind("\n")
            cut = max(newline + 1, len(text) - max(len(s) for s in self._secrets) + 1)
            cut = min(cut, len(text))
        while True:
            previous = cut
            for secret in self._secrets:
                start = text.find(secret, max(cut - len(secret) + 1, 0))
                if 0 <= start < cut < start + len(secret):
                    cut = start
            if cut == previous:
                break
        ready, self._pending = text[:cut], text[cut:]
        for secret in self._secrets:
            ready = ready.replace(secret, "[redacted]")
        return ready


class _Filter(logging.Filter):
    """Scrub a record in place, including its exception and stack text.

    ``getMessage()`` is resolved here and ``args`` cleared: a secret can sit in
    an ARGUMENT rather than in the format string, and a formatter downstream
    would otherwise interpolate the raw value back in.

    **A filter must never raise.** ``Logger.handle`` consults filters before any
    handler's own error handling, so an exception here does not degrade logging —
    it turns the CALLER's ``logger.warning(...)`` into a raise, which is a
    user-visible break for anyone who has ever resolved an MCP credential. The
    guard below is therefore belt AND braces: the one shape that did raise
    (``exc_info=False``) is handled where it belongs, and anything left
    unforeseen passes the record through unscrubbed rather than taking the
    process down with it. The SOURCE-side scrubs are the control; this filter is
    defence in depth, so a skipped record is a smaller fault than a crash.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return self._scrub_record(record)
        except Exception:  # noqa: BLE001 — a filter must never break user logging
            return True

    def _scrub_record(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — a bad format string is not our business
            return True
        scrubbed = scrub(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        # TRUTHINESS, not ``is not None``: CPython stores a falsy non-None
        # ``exc_info`` verbatim, and this repo passes one — ``harness/loop.py``
        # logs ``exc_info=not isinstance(exc, RenderedStreamError)``, i.e. exactly
        # ``False`` for the case that line exists to handle, and
        # ``Formatter.formatException(False)`` does ``tb = ei[2]`` and raises
        # ``TypeError``. CPython's own ``Formatter.format`` guards the same way
        # (agent review / QA Q2).
        if record.exc_info:
            # Rendered HERE so the traceback text (which carries the exception's
            # own message, and therefore the child's echoed line) is scrubbed
            # before any handler formats it.
            record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif isinstance(record.exc_text, str):
            record.exc_text = scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub(str(record.stack_info))
        return True


_FILTER = _Filter()


def attach() -> None:
    """Put the filter on every root handler. Idempotent; safe to call often."""
    for handler in logging.getLogger().handlers:
        if _FILTER not in handler.filters:
            handler.addFilter(_FILTER)


def register(value: str) -> None:
    """Register one resolved credential for scrubbing across MCP sinks.

    Returns without registering when the value is shorter than
    :data:`MIN_SCRUBBED_LENGTH` — see that constant for the measurement. Nothing
    is logged but the LENGTH: this is on the path a credential travels.
    """
    trimmed = value.strip()
    if not trimmed:
        return
    if len(trimmed) < MIN_SCRUBBED_LENGTH:
        logger.debug(
            "MCP credential of %d characters is too short to scrub without rewriting "
            "unrelated diagnostics; it will not be redacted",
            len(trimmed),
        )
        return
    with _LOCK:
        _STORE.register_redaction(trimmed)
    attach()


def unregister(value: str) -> None:
    """Drop one registration, for a caller whose reason to scrub has ended.

    The deterministic half of :func:`register`. This store is process-global and
    was append-only, which made any test that registered a value order-dependent
    against every later test sharing the worker — the shape that turned a
    store test's value into a failure in an unrelated provider test, and the
    reason QA asked for registration to be reversible rather than merely
    bounded. Only values registered here can be dropped; a value that is also a
    session CREDENTIAL stays registered through that store.
    """
    if not value:
        return
    with _LOCK:
        _STORE.unregister_redaction(value)


def _scrub_held_text(exc: BaseException) -> None:
    """Scrub message text an exception keeps OUTSIDE ``args``, IN PLACE.

    **Why ``args`` is not enough.** The MCP SDK's ``MCPError`` stores its text in
    ``self.error = ErrorData(message=...)`` and defines ``__str__`` from that
    field, so the args rewrite above changed nothing a reader could see:

        after the args pass: args=(-32000, 'rejected credential [redacted]', None)
                             str(exc)='rejected credential <the value>'

    That is the shape a server echoing a rejected credential in a JSON-RPC error
    arrives in — the credential is in the ERROR MESSAGE, which is the one string
    every sink publishes — so a scrub that only rewrites ``args`` leaves the
    leak on both the raised exception and its chained cause.

    The write is attempted on the value READ BACK from the object rather than on
    the one we asked for, so ``MCPError.message`` (a read-only property over
    ``error.message``) is skipped instead of raised on once the first write has
    landed, and a frozen or validating model is left alone: a scrub must never
    turn an error path into a NEW exception. Exactly two shapes are covered —
    ``error.message``, which is what an SDK error uses, then a plain ``message``
    attribute. A type composing its text from anything else is not rewritten
    here; :meth:`McpServerStderr.explain` fail-closes on that residue.
    """
    for holder in (getattr(exc, "error", None), exc):
        message = getattr(holder, "message", None)
        if not isinstance(message, str):
            continue
        scrubbed = scrub(message)
        if scrubbed == message:
            continue
        # setattr, not ``holder.message = ``: the holder is typed as object here
        # (an SDK ErrorData, or a bare exception whose attribute only some types
        # have), and the point of the suppression is that the write is ALLOWED to
        # fail on a read-only property, a frozen model or a slotted type.
        with suppress(Exception):  # read-only property, frozen model, slots
            setattr(holder, "message", scrubbed)


def sanitize_exception(exc: BaseException, _seen: set[int] | None = None) -> None:
    """Scrub a credential out of an exception's own message, IN PLACE.

    **Why in place, and why not by dropping the chain.** A raised exception is
    retained by the caller's traceback and by whatever logs it, so an
    ``McpConnectionError`` built from scrubbed text is not enough on its own:
    the exception it is chained to (``raise ... from exc``) still carries its own
    message, and a remote server that echoes the rejected header back in an error
    BODY puts the credential exactly there. Rewriting the message is what makes
    the value unreachable through every later reader of that object — ``args``
    for a builtin, plus the attribute shapes in :func:`_scrub_held_text` for an
    SDK error, which is the one that carried this leak (agent review R-1).

    Suppressing the chain instead (``from None``) is NOT an option here, and that
    was measured rather than assumed: the connect round reads the chained
    exception to tell a cancellation apart from a network failure, so dropping it
    changed how a bare-cancellation attempt settled and left the startup round
    waiting for its ceiling
    (``test_a_bare_cancellation_settles_the_round_as_a_network_failure``). The
    chain is evidence; only its text is sanitized.

    Non-string args (``OSError(errno, strerror)``, a structured error payload)
    are left alone rather than coerced: this must not change an exception's
    SHAPE, only the strings that could carry a value. The walk is bounded by an
    identity set because ``__cause__``/``__context__`` chains can revisit an
    object.
    """
    if _seen is None:
        _seen = set()
    if id(exc) in _seen:
        return
    _seen.add(id(exc))
    scrubbed = tuple(scrub(arg) if isinstance(arg, str) else arg for arg in exc.args)
    if scrubbed != exc.args:
        exc.args = scrubbed
    _scrub_held_text(exc)
    for chained in (exc.__cause__, exc.__context__):
        if chained is not None:
            sanitize_exception(chained, _seen)


__all__ = [
    "MIN_SCRUBBED_LENGTH",
    "StderrRedactor",
    "attach",
    "register",
    "sanitize_exception",
    "scrub",
    "unregister",
    "values",
]

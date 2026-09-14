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

from local_operator.variables import VariableStore

#: Redaction-only store: values to scrub, never to inject or advertise.
#: ``env={}`` so it never reads the process environment.
_STORE = VariableStore(env={})

#: Guards the store against a resolve running on a worker thread while a log
#: record is being scrubbed on the event loop.
_LOCK = threading.Lock()


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
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — a bad format string is not our business
            return True
        scrubbed = scrub(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        if record.exc_info is not None:
            # Rendered HERE so the traceback text (which carries the exception's
            # own message, and therefore the child's echoed line) is scrubbed
            # before any handler formats it.
            record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub(record.stack_info)
        return True


_FILTER = _Filter()


def attach() -> None:
    """Put the filter on every root handler. Idempotent; safe to call often."""
    for handler in logging.getLogger().handlers:
        if _FILTER not in handler.filters:
            handler.addFilter(_FILTER)


def register(value: str) -> None:
    """Register one resolved credential for scrubbing across MCP sinks."""
    if not value:
        return
    with _LOCK:
        _STORE.register_redaction(value)
    attach()


def sanitize_exception(exc: BaseException, _seen: set[int] | None = None) -> None:
    """Scrub a credential out of an exception's own message, IN PLACE.

    **Why in place, and why not by dropping the chain.** A raised exception is
    retained by the caller's traceback and by whatever logs it, so an
    ``McpConnectionError`` built from scrubbed text is not enough on its own:
    the exception it is chained to (``raise ... from exc``) still carries its own
    message, and a remote server that echoes the rejected header back in an error
    BODY puts the credential exactly there. Rewriting ``args`` is what makes the
    value unreachable through every later reader of that object.

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
    for chained in (exc.__cause__, exc.__context__):
        if chained is not None:
            sanitize_exception(chained, _seen)


__all__ = [
    "StderrRedactor",
    "attach",
    "register",
    "sanitize_exception",
    "scrub",
    "values",
]

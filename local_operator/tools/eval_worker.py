"""Persistent Python kernel worker behind the ``eval`` tool.

WHY a separate process
----------------------
The ``eval`` tool promises the model a session whose state SURVIVES across
calls — variables, imports and functions defined in one call are present in
the next. The only honest way to give arbitrary code a persistent namespace
without leaking that state into the harness process (where a stray ``import``
or a monkeypatch would silently corrupt every later tool call) is to keep the
namespace in a child process the tool owns and can kill.

Protocol
--------
One JSON object per line, both directions, request then response, forever:

    request:  {"id": "<hex>", "code": "<python source>", "stream": bool}
    response: {"id": "<hex>", "ok": true|false, "stdout": str, "stderr": str,
               "error": str|null, "result": str|null, "display": [str, ...]}

When the request sets ``stream``, the worker ALSO writes zero or more frames
before the response, one per write to stdout/stderr:

    stream:   {"id": "<hex>", "stream": "stdout"|"stderr", "text": str}

A stream frame carries the same ``id`` and is distinguished by the ``stream``
key. They are advisory progress only — the response still carries the complete
captured streams — so a reader may ignore them entirely without losing data.
That redundancy is deliberate: it keeps a dropped or malformed frame from
costing output the caller would otherwise never see.

The ``id`` is echoed verbatim so the tool can discard lines that are not its
own response — user code writing to file descriptor 1 directly (``os.write``)
cannot be intercepted from inside the process, and skipping such noise beats
dying on it. Exactly one response line is written per request and flushed
immediately; the process is also launched with ``python -u`` so no buffering
layer can reorder or delay it.

``result`` is the ``repr`` of the code's trailing expression when the last
statement is an expression (the notebook-repl convention: exec everything
before it, ``eval`` the last), else ``null``. ``display`` carries what user
code passed to the built-in ``display()`` — shown to the human, never merged
into the model-visible text, which is why it rides a SEPARATE key rather than
being printed into stdout.

Run only via ``python -u -m local_operator.tools.eval_worker`` (see
``local_operator.tools.eval``); the ``__main__`` guard keeps the module
importable without side effects.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import reprlib
import shlex
import subprocess
import sys
import threading
import traceback
import uuid
from collections.abc import Callable
from typing import Any

#: Worker-side protocol caps. Parent-side spill_truncate is intentionally
#: later; these caps prevent a giant print/display/repr from allocating and
#: JSON-encoding gigabytes BEFORE the parent gets a chance to spill it.
STREAM_CHAR_LIMIT = 1_000_000
DISPLAY_CHAR_LIMIT = 256_000
TRUNCATED_MARKER = "\n[…worker output truncated before protocol serialization]"

#: The protocol's own stdout, captured at import BEFORE any request can
#: redirect ``sys.stdout``. Streaming frames are written through this handle so
#: they reach the parent even though user code's ``print`` is being captured at
#: the same moment — writing to ``sys.stdout`` there would land in the capture
#: buffer and never be seen.
_PROTOCOL_OUT = sys.stdout
_PROTOCOL_IN = sys.stdin

# A worker can ask the parent to execute a harness tool, but cannot grant
# itself one. The parent resolves only the current session's available or
# explicitly discovered tools and repeats normal validation/approval. Keeping
# the wire synchronous makes a Python fetch/filter/join pipeline one eval cell
# without another model turn or concurrent mutations of the worker namespace.
_BRIDGE_LOCK = threading.Lock()
_ACTIVE_BRIDGE: tuple[str, bool] = ("", False)
_BRIDGE_FRAME_LIMIT = 1_000_000
_EXECUTION_THREAD = threading.get_ident()


def _call_tool(name: str, **arguments: Any) -> dict[str, Any]:
    """Execute one approved harness tool and return its structured ToolResult.

    Read ``result['is_error']`` and ``result['content']`` before consuming it.
    Results stay in Python unless the cell prints or returns them, allowing a
    pipeline to project just the fields needed by the next reasoning step.
    Errors remain data exactly as with direct calls. This helper is rebound
    per cell, and reads the current cell identity even through an old alias.
    """
    # Only the foreground cell may consume protocol input. A Python thread can
    # outlive its cell and otherwise steal the next cell's request/response or
    # inherit its capability through the global bridge state. Check BEFORE the
    # lock so stale threads cannot block the valid foreground reader either.
    if threading.get_ident() != _EXECUTION_THREAD:
        raise RuntimeError("Harness tool calls require the foreground eval execution thread.")
    with _BRIDGE_LOCK:
        request_id, enabled = _ACTIVE_BRIDGE
        if not enabled:
            raise RuntimeError("Harness tool calls are unavailable in this eval context.")
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a nonempty string")
        call_id = uuid.uuid4().hex
        frame = json.dumps(
            {
                "id": request_id,
                "tool_call": {
                    "name": name,
                    "arguments": arguments,
                    "call_id": call_id,
                },
            }
        )
        if len(frame.encode()) > _BRIDGE_FRAME_LIMIT:
            raise ValueError("tool call exceeds the 1 MB bridge request limit")
        _PROTOCOL_OUT.write(frame + "\n")
        _PROTOCOL_OUT.flush()
        line = _PROTOCOL_IN.readline()
        if not line:
            raise RuntimeError("Harness tool bridge closed before returning a result.")
        response = json.loads(line)
        if response.get("id") != request_id or response.get("call_id") != call_id:
            raise RuntimeError("Harness tool bridge returned an unmatched result.")
        result = response.get("tool_result")
        if not isinstance(result, dict):
            raise RuntimeError("Harness tool bridge returned an invalid result.")
        return result


# ---------------------------------------------------------------------------
# Disclosure-gated redaction of subprocess argv in error rendering
# ---------------------------------------------------------------------------
# A cell that spawns a subprocess routinely puts a credential in argv
# (``curl -H "Authorization: Bearer …"``, ``psql <dsn>``, ``mongosh <uri>``).
# Python re-renders that argv when the spawn fails and nobody told it which
# argument is a secret: ``CalledProcessError.__str__`` and
# ``TimeoutExpired.__str__`` interpolate ``self.cmd``, and ``format_exception``
# writes the whole ``Command '[...]'`` into the traceback. That traceback is the
# model-visible tool result, so the credential lands in the transcript even
# though the model asked for none of it. A timing-out child is the worst case:
# there is no output, so the argv is the entire content of the error.
#
# The guard is PRIOR-DISCLOSURE gating, not secret detection — guessing which
# argv elements "look like" secrets fails in both directions. An argument is
# re-rendered only when its exact bytes already appear in code the model itself
# wrote in this kernel; those bytes are already in the transcript, so echoing
# them discloses nothing new. Anything else — a value read from the
# environment, a file, or a session credential — is replaced by its length
# alone, ``<redacted:71c>``, which keeps the argument count and the failure
# shape without carrying a single byte of the value (not even a digest, which
# would be a small partial oracle).
#
# Consequences, all deliberate:
#  - argv[0] is always rendered: an executable path is the most useful part of
#    a spawn failure and is not a credential.
#  - Nothing else about the failure changes — exception type, exit code,
#    timeout and captured stdout/stderr are untouched. Only the invocation is
#    filtered, and only in the UNCAUGHT rendering; a cell that catches the
#    error and prints ``e.cmd`` is making an explicit choice and still sees it.
#  - An all-disclosed command line comes back byte-identical, so the guard is
#    invisible for ordinary failures like ``['git', 'status']``.
#  - A value assembled at runtime reads as undisclosed even when its halves
#    are in the source. That over-redacts a computed path; safe direction.
#
# The ledger is fed cell source AS EXECUTED, so a literal written in cell 1 can
# reach argv in cell 5 and still count as disclosed. Retention is bounded; an
# evicted entry only costs fidelity (over-redaction), never safety. Mirrors the
# TypeScript policy in ``omp`` (utils/argv-disclosure.ts); the
# ``<redacted:<N>c>`` shape must stay in sync with it.

#: Cell sources retained for disclosure checks, newest last.
_LEDGER_MAX_ENTRIES = 64
#: Total bytes of retained cell source. Bounds a long-lived kernel's ledger.
_LEDGER_MAX_BYTES = 256 * 1024


class _ArgvDisclosureLedger:
    """Bounded record of the code a model has run in one kernel.

    Decides whether re-rendering a string discloses anything new. Retention is
    bounded because a long session's cell history is otherwise unbounded.
    """

    def __init__(self) -> None:
        self._entries: list[str] = []
        self._bytes = 0

    def record(self, source: str) -> None:
        """Record cell source as disclosed to the transcript."""
        if not source:
            return
        self._entries.append(source)
        self._bytes += len(source)
        while len(self._entries) > _LEDGER_MAX_ENTRIES or (
            self._bytes > _LEDGER_MAX_BYTES and len(self._entries) > 1
        ):
            self._bytes -= len(self._entries.pop(0))

    def discloses(self, text: str) -> bool:
        """True when ``text``'s exact bytes already appear in recorded source.

        Substring containment is the right test and not a weakening: a match
        means those bytes are literally present in code the model wrote.
        """
        if not text:
            return True
        return any(text in entry for entry in self._entries)


#: One ledger per kernel process; the worker is single-threaded over stdin.
_ARGV_LEDGER = _ArgvDisclosureLedger()


def _redacted_arg(value: str) -> str:
    """Length-only placeholder for an undisclosed argument.

    Space-free so it reads as one token inside a rendered command line, and
    length-only so it carries no bytes of the value — not even a digest.
    """
    return f"<redacted:{len(value)}c>"


def _redact_cmd_arg(arg: Any, seen_first: list[bool]) -> str:
    """Render one ``cmd`` element, redacting it when the ledger has not seen it.

    ``seen_first`` is a one-element cell so argv[0] is preserved verbatim (see
    the module comment) without exposing the index here.
    """
    text = arg if isinstance(arg, str) else str(arg)
    if not seen_first[0]:
        seen_first[0] = True
        return text
    return text if _ARGV_LEDGER.discloses(text) else _redacted_arg(text)


def _redact_cmd(cmd: Any) -> Any:
    """Return ``cmd`` (list or string) with undisclosed arguments redacted.

    A list is rebuilt element-wise; a string is split with ``shlex`` so a
    quoted argument stays one piece (quote awareness is fidelity, not safety:
    without it ``sh -c 'exit 3'`` half-redacts an innocent line). A string
    ``shlex`` cannot parse (one unbalanced quote from the normal case) is
    ledger-gated whole: if its exact bytes appear in the model's source they
    are already disclosed; otherwise it collapses to a single ``<redacted:Nc>``
    rather than being rendered verbatim with the secret inside.
    """
    seen = [False]
    if isinstance(cmd, (list, tuple)):
        return [_redact_cmd_arg(a, seen) for a in cmd]
    if isinstance(cmd, str):
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return cmd if _ARGV_LEDGER.discloses(cmd) else _redacted_arg(cmd)
        return " ".join(_redact_cmd_arg(a, seen) for a in parts)
    return cmd


def _redact_process_exception(exc: BaseException) -> None:
    """Redact a process-invocation exception's argv in place before formatting.

    Gating matters: this must never touch an ordinary error. It fires only for
    the two stdlib types that embed argv — ``CalledProcessError`` and
    ``TimeoutExpired`` — whose ``.cmd`` attribute carries the invocation.
    """
    if not isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        return
    cmd = getattr(exc, "cmd", None)
    if cmd is None:
        return
    redacted = _redact_cmd(cmd)
    if redacted is cmd or redacted == cmd:
        return
    with contextlib.suppress(Exception):
        exc.cmd = redacted  # type: ignore[attr-defined]


def _redact_process_exception_chain(exc: BaseException) -> None:
    """Redact argv in ``exc`` AND every process error in its context/cause chain.

    ``format_exception`` renders the WHOLE ``__context__``/``__cause__`` chain,
    so redacting only the outermost exception leaks a secret embedded in an
    inner one — ``except CalledProcessError as e: raise RuntimeError(str(e))``
    puts the undisclosed argv in the traceback twice. The walk is iterative
    with a visited set so a cyclic chain (``raise ... from`` pointing back)
    terminates instead of hanging.
    """
    visited: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        _redact_process_exception(current)
        stack.append(current.__cause__)
        # __context__ is the implicit chain (an exception raised while handling
        # another); it is what a bare ``raise`` inside ``except`` produces.
        stack.append(current.__context__)
        # format_exception renders ExceptionGroup LEAVES, so a CalledProcessError
        # raised inside an asyncio.TaskGroup task is rendered too — and leaks
        # argv if the walk stops at the group. Descend into .exceptions under
        # the same visited set. BaseExceptionGroup carries the attribute; a
        # plain exception simply does not, hence the getattr default.
        stack.extend(getattr(current, "exceptions", ()))


_REPR = reprlib.Repr()
_REPR.maxstring = 4096
_REPR.maxother = 4096
_REPR.maxlist = 100
_REPR.maxtuple = 100
_REPR.maxdict = 100


class _CappedTextIO(io.TextIOBase):
    """Text sink retaining at most ``limit`` chars while reporting all writes
    successful, so user code cannot distinguish it from StringIO."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._parts: list[str] = []
        self._used = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        text = str(value)
        remaining = self.limit - self._used
        if remaining > 0:
            kept = text[:remaining]
            self._parts.append(kept)
            self._used += len(kept)
        if len(text) > max(remaining, 0):
            self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        body = "".join(self._parts)
        return body + (TRUNCATED_MARKER if self.truncated else "")


class _DisplaySink:
    """Bounded list-shaped display channel (the wire contract stays list[str])."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.items: list[str] = []
        self.used = 0
        self.truncated = False

    def add(self, value: str) -> None:
        remaining = self.limit - self.used
        if remaining <= 0:
            self.truncated = True
            return
        kept = value[:remaining]
        self.items.append(kept)
        self.used += len(kept)
        if len(value) > remaining:
            self.truncated = True

    def finish(self) -> list[str]:
        if self.truncated:
            self.items.append("[…display output truncated before protocol serialization]")
        return self.items


#: Filename stamped into compiled code so tracebacks name the tool's cell
#: instead of a real path the model never wrote (or worse, one it did).
_FILENAME = "<eval>"

#: The namespace docstring. The session is the one thing user code cannot
#: discover by guessing, so it is spelled out where ``help()`` and curious
#: models will find it.
NAMESPACE_DOC = """\
Persistent Python session.

State SURVIVES across calls: variables, imports and functions defined in one
call are still present in the next. The kernel runs in the session's working
directory. Use display(value) to show a value to the USER only — it is never
added to the model's context.
"""


def _safe_repr(value: Any) -> str:
    """``repr(value)`` that cannot raise.

    A user-defined ``__repr__`` is arbitrary code; one that raises would turn
    a successful computation into a lost result, so the failure is reported
    in-band instead.
    """
    try:
        return _REPR.repr(value)
    except BaseException as exc:  # noqa: BLE001 — user code, any failure is data
        return f"<unrepresentable result: {type(exc).__name__}: {exc}>"


def _make_display(sink: _DisplaySink) -> Callable[..., None]:
    """The per-request ``display`` builtin, appending formatted values to
    ``sink``.

    Rebound on every request so the sink is per-call: a value displayed in
    call N must not reappear in call N+1's response, the same way stdout from
    call N does not.
    """

    def display(value: Any, *more: Any) -> None:
        sink.add(_safe_repr(value))
        for extra in more:
            sink.add(_safe_repr(extra))

    return display


def _split_trailing_expression(code: str) -> tuple[ast.Module, ast.Expression | None]:
    """``(module_of_all_but_last, expression_of_last | None)``.

    Compiling the whole source would make ``x = 1`` evaluate to ``None`` even
    when the model clearly means the notebook-repl contract, and ``eval``-ing
    the whole source would reject statements outright. Splitting the parsed
    tree keeps both: statements run via ``exec``, and a trailing expression is
    returned via ``eval``. A re-``compile`` of already-located AST nodes needs
    no ``fix_missing_locations`` — parsing attached the locations.
    """
    tree = ast.parse(code, filename=_FILENAME, mode="exec")
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return tree, None
    trailing = tree.body[-1]
    head = ast.Module(body=tree.body[:-1], type_ignores=[])
    return head, ast.Expression(body=trailing.value)


def _format_error(exc: BaseException) -> str:
    """One error string per failure kind.

    Compile errors get exception-only formatting: a traceback whose only
    frame is the worker's own ``exec`` adds noise, not information, and the
    model can fix a syntax error from the message alone. Runtime failures
    keep the traceback — the frames name the user's own lines, which IS the
    information the model needs.
    """
    if isinstance(exc, SyntaxError):
        return "".join(traceback.format_exception_only(type(exc), exc)).strip()
    # Redact embedded subprocess argv BEFORE the traceback is rendered: the
    # exception's ``__str__`` interpolates ``cmd`` at format time, so this is
    # the single choke point. ``CalledProcessError.__str__`` reads ``self.cmd``
    # fresh, and ``TimeoutExpired.__str__`` likewise, so rewriting the
    # attribute is what the formatted output reflects. The whole context/cause
    # chain is walked because format_exception renders all of it.
    _redact_process_exception_chain(exc)
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _execute(namespace: dict[str, Any], code: str) -> str | None:
    """Run ``code`` in ``namespace``; return the trailing expression's repr.

    ``BaseException`` is deliberate: user code calling ``sys.exit()`` or
    raising ``KeyboardInterrupt`` is a reportable error, not a reason to tear
    down a session whose earlier state is still valuable. (Fatal signals and
    ``os._exit`` cannot be caught here at all — that is the tool's crash
    path, and it says so honestly.)
    """
    head, trailing = _split_trailing_expression(code)
    exec(compile(head, _FILENAME, "exec"), namespace)
    if trailing is None:
        return None
    value = eval(compile(trailing, _FILENAME, "eval"), namespace)
    return _safe_repr(value)


class _StreamingTextIO(_CappedTextIO):
    """A capped sink that also forwards each write to the parent immediately.

    Plain ``_CappedTextIO`` only surrenders its contents in the final response,
    which is correct for a short call and useless for a long one: a training
    loop printing an epoch a minute would show nothing for an hour and then
    everything at once. When a request asks to stream, each write is ALSO
    emitted as its own protocol frame so the parent can publish it to the
    job's peek buffer while the code is still running.

    It still caps and still accumulates: the final response stays byte-for-byte
    what a non-streaming run would produce, so streaming changes only WHEN the
    parent learns something, never WHAT it ends up with.
    """

    def __init__(self, limit: int, emit: Callable[[str], None]) -> None:
        super().__init__(limit)
        self._emit = emit

    def write(self, value: str) -> int:
        written = super().write(value)
        if value:
            # A failure to emit must never break the user's code, which is what
            # an exception raised inside ``print`` would do. The frame is
            # advisory; the authoritative copy rides the final response.
            with contextlib.suppress(Exception):
                self._emit(value)
        return written


def _handle(namespace: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Execute one request and build its response mapping."""
    display_sink = _DisplaySink(DISPLAY_CHAR_LIMIT)
    # Rebound per request (see _make_display).
    namespace["display"] = _make_display(display_sink)
    namespace["tool"] = _call_tool
    global _ACTIVE_BRIDGE
    _ACTIVE_BRIDGE = (str(request.get("id", "")), bool(request.get("tool_bridge")))

    request_id = request.get("id", "")
    # Record the source BEFORE running it: a failure in this very cell can
    # reference literals written here, and those are disclosed by definition.
    _ARGV_LEDGER.record(str(request.get("code", "")))
    if request.get("stream"):

        def _emit(channel: str, text: str) -> None:
            # Written straight to the real stdout while the redirect is active:
            # ``sys.stdout`` is the captured object during the run, so the
            # protocol keeps its own handle (``_PROTOCOL_OUT``) captured at
            # import, before any redirect could reach it.
            frame = {"id": request_id, "stream": channel, "text": text}
            _PROTOCOL_OUT.write(json.dumps(frame) + "\n")
            _PROTOCOL_OUT.flush()

        stdout: _CappedTextIO = _StreamingTextIO(
            STREAM_CHAR_LIMIT, lambda text: _emit("stdout", text)
        )
        stderr: _CappedTextIO = _StreamingTextIO(
            STREAM_CHAR_LIMIT, lambda text: _emit("stderr", text)
        )
    else:
        stdout = _CappedTextIO(STREAM_CHAR_LIMIT)
        stderr = _CappedTextIO(STREAM_CHAR_LIMIT)
    ok = True
    error: str | None = None
    result: str | None = None
    try:
        # stdout/stderr swap, not fd redirection: print/warnings/logging write
        # through sys.stdout/sys.stderr and are captured, while the protocol
        # keeps fd 1 for itself.
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = _execute(namespace, str(request.get("code", "")))
    except BaseException as exc:  # noqa: BLE001 — user code failing is data
        ok = False
        error = _format_error(exc)
        if len(error) > STREAM_CHAR_LIMIT:
            error = error[:STREAM_CHAR_LIMIT] + TRUNCATED_MARKER
    finally:
        # Background threads left behind by arbitrary Python code cannot issue
        # tool calls between cells using an expired turn's authority.
        _ACTIVE_BRIDGE = ("", False)
    return {
        "id": request.get("id", ""),
        "ok": ok,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "error": error,
        "result": result,
        "display": display_sink.finish(),
    }


def main() -> None:
    """Read requests, run them, answer — until stdin closes or the tool kills
    the process. Either ending is normal from this side: the tool owns the
    lifecycle (idle reaping, LRU eviction, timeout kill)."""
    namespace: dict[str, Any] = {
        "__name__": "__eval__",
        "__doc__": NAMESPACE_DOC,
    }
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request is not a JSON object")
        except ValueError:
            # The tool only writes well-formed lines; anything else on stdin
            # means the pipe is not worth trusting further.
            continue
        response = _handle(namespace, request)
        _PROTOCOL_OUT.write(json.dumps(response) + "\n")
        _PROTOCOL_OUT.flush()


if __name__ == "__main__":
    main()

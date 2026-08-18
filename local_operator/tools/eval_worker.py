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
import sys
import traceback
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

    request_id = request.get("id", "")
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

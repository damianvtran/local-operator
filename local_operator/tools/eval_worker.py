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

    request:  {"id": "<hex>", "code": "<python source>"}
    response: {"id": "<hex>", "ok": true|false, "stdout": str, "stderr": str,
               "error": str|null, "result": str|null, "display": [str, ...]}

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
import sys
import traceback
from collections.abc import Callable
from typing import Any

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
        return repr(value)
    except BaseException as exc:  # noqa: BLE001 — user code, any failure is data
        return f"<unrepresentable result: {type(exc).__name__}: {exc}>"


def _make_display(sink: list[str]) -> Callable[..., None]:
    """The per-request ``display`` builtin, appending formatted values to
    ``sink``.

    Rebound on every request so the sink is per-call: a value displayed in
    call N must not reappear in call N+1's response, the same way stdout from
    call N does not.
    """

    def display(value: Any, *more: Any) -> None:
        sink.append(_safe_repr(value))
        for extra in more:
            sink.append(_safe_repr(extra))

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
    return "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).strip()


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


def _handle(namespace: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Execute one request and build its response mapping."""
    display_sink: list[str] = []
    # Rebound per request (see _make_display).
    namespace["display"] = _make_display(display_sink)

    stdout, stderr = io.StringIO(), io.StringIO()
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
        "display": display_sink,
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
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

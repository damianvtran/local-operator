"""Which tool's bytes are being redacted — carried for the host's redaction hook.

**Why this exists.** The harness hands the host ONE callable for every
model-visible result (``LoopConfig.redact_tool_result``), and calls it with the
text alone. That is the right contract for masking — the hook needs the bytes —
but it is not enough for the other half of the job: a result that was rewritten
by the credential-SHAPE pass has to be reported, and the report has to name the
tool, which means the tool identity has to travel to a hook that cannot be
handed another argument without breaking every existing host that passes a
bound method (``VariableStore.redact``, in this tree's own session code).

So it rides a :class:`~contextvars.ContextVar`, the same mechanism
``tools/builtin`` uses for its file-scan guard and for ``_ADVERTISED_EFFORT``.
Two properties make it the right one here rather than a field somewhere:

* it is PER TASK, so two tool calls in flight at once — or a subagent's own
  task running beside its parent's — cannot attribute one's output to the
  other's name; and
* it costs nothing when unused: the default is the empty pair and no hook is
  obliged to read it.

The summary is bounded and SCRUBBED before it is ever stored, because a
credential can be typed into the call itself (``mysql -p…``, ``curl -u…``,
``--password=…``) and a report about a redaction must not be the next place the
value appears.
"""

from __future__ import annotations

import inspect
import json
import re
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from local_operator.redaction_shapes import scrub_shapes

#: ``(tool name, one-line argument summary)`` for the call whose result is being
#: redacted. Empty when nothing published one, which is the honest answer for a
#: hook called from a test or from a surface that has no call in hand.
_SOURCE: ContextVar[tuple[str, str]] = ContextVar("tool_redaction_source", default=("", ""))

#: Longest argument summary an incident may carry. Long enough for a real
#: command line, short enough that a notice stays one line in a transcript.
SUMMARY_LIMIT = 200

#: Argument names whose value is worth quoting in the summary, most specific
#: first. A summary that named nothing would tell an operator which tool fired
#: and not what it was asked to do.
_SUMMARY_KEYS = ("command", "path", "file_path", "url", "query", "pattern", "expression")

_WHITESPACE = re.compile(r"\s+")


#: The session's incident sink, published for the layers that mask BEFORE a
#: result exists. Empty when no session is running, which is the honest answer
#: for a bare tool call.
_HIT_REPORTER: ContextVar[Any] = ContextVar("shape_hit_reporter", default=None)


def _hold(report: Any) -> Any:
    """Hold a sink WEAKLY when it is a bound method.

    The sink is a session's own method, and a ``ContextVar`` value is COPIED into
    every derived context — a child task, a subagent run, a ``to_thread`` worker.
    Holding the bound method strongly therefore kept the session alive through
    contexts that outlive it: measured as a disposed child Session that was never
    collectable and a child holding its parent (``tests/unit/session/
    test_launch_subagent.py``, three weakref assertions). The reporter is
    best-effort by contract — a hit report must never break a mask — so a sink
    that has been collected is simply no sink.
    """
    if report is None:
        return None
    if inspect.ismethod(report):
        # A bound method of an object that supports weak references — the session's
        # own sink, and the only case that needs the weak form.
        try:
            return weakref.WeakMethod(report)
        except TypeError as exc:  # a bound method of a non-weak-referenceable object
            raise TypeError(
                "the shape-hit sink must be a bound method of a weak-referenceable "
                "object, or a plain callable: a strong reference to it is what kept "
                "a disposed child session alive (see tests/unit/session/"
                "test_launch_subagent.py)"
            ) from exc
    if callable(report):
        # A plain function or closure owns no session, so there is nothing to
        # release: held directly, and that is the documented fallback.
        return report
    raise TypeError(
        "the shape-hit sink must be a bound method of a weak-referenceable object, "
        f"or a plain callable; got {type(report).__name__}"
    )


def _resolve(report: Any) -> Any:
    """The callable behind :func:`_hold`, or ``None`` once it has been collected."""
    if isinstance(report, weakref.WeakMethod):
        return report()
    return report


@contextmanager
def shape_hit_reporting(report: Any) -> Iterator[None]:
    """Publish ``report`` as the sink for shape hits observed during the block.

    The pipe filter masks the bytes of a command that is still running — and for
    the case this feature was written for (``kubectl exec … env``, where the
    credential is in the OUTPUT and nowhere in the command) the mask happens
    BEFORE the loop's result hook ever sees the text. By then the value is gone,
    the shape pass finds nothing to match, and the incident that was supposed to
    become a rotation ticket is never filed at all. Measured: 0 incidents and 0
    live notices for the output-only shape.
    """
    token = _HIT_REPORTER.set(_hold(report))
    try:
        yield
    finally:
        _HIT_REPORTER.reset(token)


def set_shape_hit_reporter(report: Any) -> Any:
    """Install the sink for the CURRENT context, returning the reset token.

    A context manager is the tidy shape and the wrong one at the call site: the
    loop run is a long ``async for`` whose body would have to be re-indented to
    sit inside a ``with``. The sink is the session's own bound method and lives
    as long as the session does, so the token is kept only for symmetry with
    :func:`reset_shape_hit_reporter` (used by tests).
    """
    return _HIT_REPORTER.set(_hold(report))


def reset_shape_hit_reporter(token: Any) -> None:
    """Undo :func:`set_shape_hit_reporter`."""
    _HIT_REPORTER.reset(token)


def report_shape_hits(labels: list[str]) -> None:
    """Hand shape labels to the session's incident queue, if one is attached."""
    if not labels:
        return
    reporter = _resolve(_HIT_REPORTER.get())
    if reporter is None:
        return
    try:
        reporter(labels)
    except Exception:  # noqa: BLE001 — a report must never break a mask
        pass


def current_tool_source() -> tuple[str, str]:
    """``(tool name, summary)`` for the call being redacted, or ``("", "")``."""
    return _SOURCE.get()


@contextmanager
def tool_source(tool_name: str, arguments: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Publish which call the bytes being redacted came from, for its duration."""
    token = _SOURCE.set((tool_name or "", summarize_arguments(arguments)))
    try:
        yield
    finally:
        _SOURCE.reset(token)


def summarize_arguments(arguments: Mapping[str, Any] | None) -> str:
    """One bounded, credential-scrubbed line describing what a call asked for.

    Scrubbed with the SHAPE pass rather than the exact-value pass on purpose:
    this function has no store to ask, and the value it is protecting is one the
    session may have just learned about — the same call that dumps a credential
    is often the one that spells it.
    """
    if not arguments:
        return ""
    text = ""
    for key in _SUMMARY_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            text = value
            break
    if not text:
        try:
            text = json.dumps({str(k): v for k, v in arguments.items()}, default=str)
        except (TypeError, ValueError):
            text = " ".join(f"{key}=…" for key in arguments)
    collapsed = _WHITESPACE.sub(" ", scrub_shapes(text)).strip()
    if len(collapsed) <= SUMMARY_LIMIT:
        return collapsed
    return collapsed[: SUMMARY_LIMIT - 1] + "…"

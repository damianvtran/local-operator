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

import json
import re
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

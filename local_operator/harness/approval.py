"""The tool-approval gate: its type, and the ONE place its arity is resolved.

A host installs a gate to answer "may this tool run?" for write- and
exec-tier calls. The gate has two accepted shapes and both must keep working:

- ``(tool_name, description)`` — the original, and what every host in the tree
  wrote against (the CLI's stdin prompt, the TUI's approval card, the FastAPI
  facade, and a long tail of test fakes);
- ``(tool_name, description, job_id)`` — the same question plus the provenance
  a host needs to tell a foreground ask apart from a background one.

Widening the callback in place rather than versioning it is what keeps every
existing host assignable and type-clean: :data:`ApprovalGate` is a union of the
two shapes, so a two-argument handler satisfies it unchanged.

Arity is resolved HERE, once, and never at a call site. There are two call
sites — the loop's tier gate and the builtin self-gate — they must agree or the
same host is asked two different ways depending on which tool is running, and
the failure mode of getting it wrong is a ``TypeError`` that both call sites
catch and turn into a silent denial. Resolution is by signature inspection and
NOT by calling with three arguments and retrying on ``TypeError``: a
``TypeError`` raised from inside the host's own body is indistinguishable from
an arity mismatch, and the retry would then invoke a handler that had already
mounted a prompt or written a log line.

The rule is BY NAME — a parameter literally called ``job_id`` — and not by
counting parameters, because counting is wrong in both directions and both
wrong answers are silent:

- ``(tool_name, description, *, job_id=None)`` is the natural way to add
  provenance to an existing handler without breaking its callers, and it has
  two POSITIONAL parameters. Counting calls it with two and the host's
  ``job_id`` stays ``None`` forever, so a host trying to scope a denial to a
  background job simply never can.
- ``(tool_name, description, timeout=30)`` is a pre-existing handler whose
  third parameter means something else. Counting hands it a job id as its
  timeout.
- ``async def wrapper(*args)`` forwarding to a two-argument gate is the shape
  this codebase actually writes. Counting ``*args`` as wide calls the wrapper
  with three, the INNER gate raises ``TypeError``, and both call sites turn
  that into exactly the invisible denial this module exists to prevent.

So a differently-named third parameter, ``*args``, and an unreadable signature
all degrade to the two-argument shape — the one that always works. A host that
wants provenance names it ``job_id`` and gets it.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Literal, Union, cast

#: Transcript row written when an approval gate expires with nobody attached.
#:
#: Lives here, beside the gate concept itself, because three layers need to
#: agree on it and none of them may import the others: the runtime WRITES it
#: (``session/runtime/owned.py``), the session RENDERS it for the model
#: (``session/session.py``), and the TUI renders it for the user. A copy in
#: any one of them would be a fourth place for the string to drift.
#:
#: The distinction it preserves is that an expiry is not a decision: without
#: the row, the next turn reads a plain denial and re-plans around a choice
#: nobody made.
GATE_TIMEOUT_CUSTOM_TYPE = "gate_timed_out_unattended"

#: The two accepted host gate shapes. Declared as a union rather than a
#: Protocol with an optional parameter because Python cannot express "callable
#: of two OR three arguments" in one signature, and a Protocol declaring the
#: third would make every existing two-argument host a type error.
ApprovalGate = Union[
    Callable[[str, str], Awaitable[bool]],
    Callable[[str, str, "str | None"], Awaitable[bool]],
]

#: How (and whether) a gate takes the job id. ``keyword`` covers the ordinary
#: third parameter as well as the keyword-only one: passing by keyword is
#: unambiguous for either, and it is the only spelling that works for the
#: keyword-only case.
_JobIdStyle = Literal["none", "positional", "keyword"]


def _job_id_style(gate: ApprovalGate) -> _JobIdStyle:
    """Find this gate's ``job_id`` parameter, or report that it has none.

    Positional-only (``..., job_id, /``) has to be passed positionally; every
    other binding takes the keyword. A gate whose signature cannot be read at
    all — a C callable, some ``functools.partial`` shapes — is the
    two-argument form, because guessing wide there raises inside the gate.
    """
    try:
        parameters = inspect.signature(gate).parameters.values()
    except (TypeError, ValueError):
        return "none"
    for parameter in parameters:
        if parameter.name != "job_id":
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return "positional"
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            return "keyword"
    return "none"


async def ask_approval(
    gate: ApprovalGate,
    tool_name: str,
    description: str,
    job_id: str | None = None,
) -> bool:
    """Put one approval question to the host. ``True`` means proceed."""
    style = _job_id_style(gate)
    if style == "positional":
        wide = cast("Callable[[str, str, str | None], Awaitable[bool]]", gate)
        return bool(await wide(tool_name, description, job_id))
    if style == "keyword":
        by_keyword = cast("Callable[..., Awaitable[bool]]", gate)
        return bool(await by_keyword(tool_name, description, job_id=job_id))
    narrow = cast("Callable[[str, str], Awaitable[bool]]", gate)
    return bool(await narrow(tool_name, description))


__all__ = ["ApprovalGate", "ask_approval"]

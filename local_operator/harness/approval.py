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
#: (``session/runtime/serving.py``), the session RENDERS it for the model
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


def loosening_is_authorised(*, source: str, gate_is_here: bool) -> bool:
    """Whether a write may LOOSEN a live approval gate (``ask`` -> ``auto``).

    The gate's own boundary, and the whole of it: a loosening is an operator
    action only when it is a write this process made through the operator's own
    settings facade (``settings_io`` -> ``config_watch.notify_local``, which
    delivers ``source="local"``) *in the process that holds the gate*.
    Everything else is unattributed from here and may only tighten:

    * a model tool's own file write (the party being gated is not the authority
      that may lower its own gate -- the reason this predicate exists);
    * an editor, a second pane, or any other process's edit;
    * the settings API or ``lop config edit`` in ANOTHER process, which is a
      genuine operator action but happened where this gate is not, so this
      process cannot tell it apart from the first case;
    * the embedded TUI's ``/settings`` page while an attached runtime owns the
      gate (``gate_is_here=False``): the file write is attributed in the app's
      own process, but the engine consults the runtime's flag.

    ``source`` is ``ConfigChange.source`` and ``gate_is_here`` is whether the
    caller is the process whose flag the engine reads -- ``True`` for the
    runtime/daemon host, ``not self._gate_is_owned_elsewhere()`` for the TUI.
    Spelled ``source == "local"`` rather than "not disk" so widening what
    counts as ``local`` in ``config_watch`` widens who may loosen the gate;
    that is the direction the literal deliberately makes visible at the
    call site rather than absorbing here.

    Tightening is NOT this predicate's business: ``auto`` -> ``ask`` and every
    other hardening path stay live and unconditional. Nor is ``--yolo``, which
    is an explicit pin on the run rather than a transition of the gate.
    """
    return source == "local" and gate_is_here


#: The ONE sentence both hosts print when a write tried to loosen a live gate
#: without being authorised to (see :func:`loosening_is_authorised`).
#:
#: Lives here for the same reason :data:`GATE_TIMEOUT_CUSTOM_TYPE` does: two
#: layers emit it and neither may import the other — the runtime
#: (``session/runtime/serving.py``) as a ``NoticeEvent``, the embedded TUI
#: (``tui/app.py``) into its own transcript — and a second copy is a second
#: chance for one surface to describe the rule differently from the other. The
#: wording is also load-bearing rather than decorative: it names the RULE ("a
#: write from outside this session") rather than the author, because the
#: emitting process cannot know who wrote the file, and in the attached-pane
#: case the person reading it is the one who just clicked the row (design round
#: 1, D3). ``/approvals auto`` is the route that does loosen the gate.
LOOSENING_REFUSED_NOTICE = (
    "keeping tool approvals: ask — config.yml now says auto, but a write from outside "
    "this session cannot loosen it; /approvals auto loosens it here"
)


#: The sibling sentence, and the reason it is a constant too (agent review round
#: 2, n2's family): the same event — the file says ``auto``, this session keeps
#: its typed ``ask`` — is emitted by the runtime and by the embedded pane, and
#: round 1 left the two copies inline in each host. Two copies of one sentence is
#: the drift U5 found for the receipts, one surface over.
LOOSENING_KEPT_BY_ASK_NOTICE = (
    "keeping tool approvals: ask — set with /approvals in this session; config.yml "
    "now says auto, /approvals auto adopts it"
)


__all__ = [
    "ApprovalGate",
    "LOOSENING_KEPT_BY_ASK_NOTICE",
    "LOOSENING_REFUSED_NOTICE",
    "ask_approval",
    "loosening_is_authorised",
]

"""One rule, two entry points: what a command an agent ran may open.

WHY THIS EXISTS. On 2026-09-18 the operator's desktop sidebar listed two chats
they never opened — ``lo-1281-review`` (role ``reviewer``) and ``lo-1281-qa``
(``qa-tester``) — created 7 ms apart by a subagent that owed a review round on
PR #1281 and reached for the CLI:

    lop exec --profile reviewer --yolo --background \
        --run-in ~/workspace/repos/lo-classify-cost \
        --name lo-1281-review < /tmp/reviewer_brief.md

Nothing about that command was broken. What it produced, though, is a TOP-LEVEL
session by every test the store applies: an ordinary conversation directory with
no ``origin.json``, so :func:`local_operator.resume.is_user_session` reports a
human started it and the desktop feed, the ``/resume`` picker and the phone's
history all offer it as the operator's own work. It also runs outside the
parent's job manager, so the parent cannot see, steer, cancel or account for it,
and it is not a child by this harness's definition (children are
capacity-gated, observed through the parent's roster, and navigable as a tree in
the TUI and the desktop UI). Two facts made it the
path of least resistance for a model trying to do the right thing: ``--profile``
is advertised in ``lop exec --help``, and a role that does not delegate holds no
``task`` tool to use instead — see ``harness.subagent``'s prune, which drops
``task``/``wait``/``wake`` for exactly the ``coder``-shaped role that was
running here.

The other half of that incident is the role's ALLOWANCE, and it is a role
question rather than a guard question (operator, 2026-09-18): a subagent that
holds ``task`` delegates with it, and one that does not hold it may not create
subagents at all — it does the work itself. So the fix for a team brief that owes
a review round to a slice is to give that slice a role that may delegate, or to
keep the round with the session that delegates. What the 2026-09-19 relaxation
below changes is the pair of words "at all": a session that HOLDS ``task`` may
now also open sessions with ``exec`` when that is what the user asked for, while
a session that does not hold it is still refused — for that session the CLI is a
way around a missing tool rather than a way to delegate.

So the rule is enforced where the act happens, and the refusal tells the reader
what to do instead — ``task`` when this session holds it, ``hub`` back to the
session that delegated when it does not, and, for work that belongs later,
``wake`` when the reader holds it and the session that delegated when it does
not.

THE RULE AFTER THE 2026-09-19 RELAXATION, AND WHY IT IS ASYMMETRIC. The
incident's root cause was never ``lop exec`` — it was a session reaching for the
CLI because it held no ``task`` tool, which is a ROLE's answer
(``agent_profiles``' ``delegate``). So the hard block became an allowance that
follows the session's OWN live inventory, and the two entry points part company
because they are not the same act:

* ``lop exec`` is the supported way to open separate TOP-LEVEL sessions, and it
  is available to an agent whose session holds ``task``. That is the case the
  operator asked for: a delegating agent told to fully delegate work, or to fan
  out a large number of independent long-lived workstreams, may open real
  sessions — the shape a ``task`` child cannot give it, since a child is one
  prompt and ends. A session that does NOT hold ``task`` is refused exactly as
  before, because for that session the CLI is a way AROUND a missing tool rather
  than a way to delegate.
* the interactive path (``lop``, ``lop --resume ID``, ``--tui``) stays refused
  for ANY agent shell, delegating or not. An agent has no terminal, so what that
  path opens is a front end on the OPERATOR's screen — not the delegation shape
  above, and not something a command should be able to put in front of a person.
  The relaxation answers "may this agent open sessions", and only ``exec`` is a
  session.

The allowance is carried by :data:`MAY_DELEGATE_ENV`, from the LIVE TOOL
INVENTORY rather than from a role name or a declared field: a role that may not
delegate has ``task`` pruned from its inventory (``harness.subagent``), and a
declared inventory narrows the same list (``Session._filter_declared``), so
"holds ``task``" and "may delegate" are one fact wherever the inventory is
trusted. Its writer is the ``bash`` tool, and that writer SIGNS IT IN BOTH
DIRECTIONS: an absent marker and an empty one both mean "may not delegate",
and the empty write is what stops a value INHERITED from the parent's
environment — the allowed route's own `lop` child is exactly that — from
outliving the session it described.

WHAT THE MARKER IS, AND WHAT IT IS NOT. ``LOCAL_OPERATOR_AGENT_SHELL`` is set by
the ``bash`` tool on every command it runs (see
:data:`local_operator.tools.builtin.NON_INTERACTIVE_ENV`), so it means exactly
"a process an agent's tool call started". This module does not pretend it is a
security boundary, and the message says so in words rather than leaning on the
marker: a subprocess spawned by the ``eval`` tool, or a command that scrubs its
own environment, does not carry it. What it stops is the naive path from
succeeding QUIETLY, and that is the path that actually occurred — a model that
finds the refusal answered in words has what it needs to route the work
correctly, which is the whole intent.

THE ESCAPE, AND WHY IT IS NOT SILENT. ``LOCAL_OPERATOR_ALLOW_NESTED_SESSION=1``
exists for this harness's own tests and for the QA runs that must drive the real
CLI (a standing rule of the operator's: testing evidence comes from exercising
the real path). Both entry points honour it, and
:func:`harness_child_env` is how a harness declares itself to its children. It
is deliberately NOT named in :func:`refusal_message` — that text is
model-facing, and its job is to route the model to ``task``/``hub``/``wake`` —
but it is documented for the human in ``docs/EXEC.md``, and (obscurity, not
secrecy) named for agents in this repository's own ``AGENTS.md`` section on the
rule: a QA run that legitimately needs the real front end must be able to find
it, and the source is readable either way. It is not an equivalent path: a
session opened under it is stamped :data:`local_operator.resume.ORIGIN_AGENT_SHELL`,
so even then the run stays out of the picker, the sidebar and the phone's list
rather than appearing as a chat the operator opened
(:func:`stamp_agent_shell_session`). Neither is the delegating route silent — it
gets the SAME stamp, which is why :func:`agent_shell_opened_run` covers both.

The pytest suite never sees any of these variables. ``LOCAL_OPERATOR_AGENT_SHELL``,
``LOCAL_OPERATOR_ALLOW_NESTED_SESSION`` and ``LOCAL_OPERATOR_AGENT_MAY_DELEGATE``
are all scrubbed by ``tests/conftest.py``'s ambient-environment fixture, so a
suite run by an agent does not refuse the sessions it builds on purpose, and a
guard test that wants a marker sets it for itself instead of inheriting it from
whichever harness happened to launch pytest. The scrub of the allowance matters
for a reason of its own: an inherited ``=1`` would make every guard test assert
the ALLOW path while looking like it tested the refusal.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

#: Set by the ``bash`` tool on every command it runs. Names the first of the two
#: facts this module acts on: the process descended from an agent's tool call.
AGENT_SHELL_ENV = "LOCAL_OPERATOR_AGENT_SHELL"

#: The documented escape for harness tests and QA runs that must drive the real
#: CLI. Named in ``docs/EXEC.md``, never in the refusal the model reads.
ALLOW_NESTED_SESSION_ENV = "LOCAL_OPERATOR_ALLOW_NESTED_SESSION"

#: Signed by the ``bash`` tool on every command it runs, from
#: :attr:`local_operator.harness.types.ToolContext.may_delegate`, in THREE arms:
#: "1" when the session may delegate, the empty string when the name is
#: inherited from the launcher and must be cleared, and NOT WRITTEN AT ALL
#: otherwise — the name is the mechanism (a shell that knows the spelling can
#: self-grant), and a session that never had the allowance is not handed it.
#: Names the second fact this module acts on: this shell's session may delegate,
#: so ``exec`` is a delegation route for it rather than a way around one.
#: Deliberately NOT named in :func:`refusal_message` — a reader told how the
#: allowance is spelled learns how to look for it — but documented for the human
#: in ``docs/EXEC.md``.
#:
#: WHY THE CLEAR MATTERS AS MUCH AS THE SET, and why the OMIT is not a gap: a
#: ``bash`` child's environment starts as a copy of the harness process's own
#: (``shell_env``'s default ``inherit`` mode), so a session that inherited the
#: marker from an ancestor would otherwise carry it for life however its own role
#: is configured — and the allowed route is what puts it there, since an allowed
#: `lop exec` runs `lop` as a child of the delegating shell. ``_on("")`` is False,
#: so the empty string is the "no"; an absent name reads the same way, which is
#: why a session that never had the allowance is not told how it is spelled.
MAY_DELEGATE_ENV = "LOCAL_OPERATOR_AGENT_MAY_DELEGATE"

#: Values that read as "on". Matches the convention the rest of the package
#: uses for boolean-ish environment flags (``agent_profiles`` reads ``delegate``
#: the same way), so ``=true`` from a shell script behaves as written rather
#: than being silently ignored for not being ``1``.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _on(value: str) -> bool:
    return value.strip().lower() in _TRUTHY


def in_agent_shell() -> bool:
    """True when this process descends from an agent's ``bash`` tool call."""
    return _on(os.environ.get(AGENT_SHELL_ENV, ""))


def nested_session_allowed() -> bool:
    """True when the escape hatch for real-CLI testing has been set."""
    return _on(os.environ.get(ALLOW_NESTED_SESSION_ENV, ""))


def may_delegate_from_shell() -> bool:
    """True when the session that ran this command holds ``task``.

    Read from :data:`MAY_DELEGATE_ENV`, which the ``bash`` tool signs from
    :attr:`local_operator.harness.types.ToolContext.may_delegate` — itself
    derived by the session from its live tool inventory. ABSENT MEANS NO, and so
    does EMPTY, and both directions are load-bearing: a shell that did not export
    the marker (a child built by an older runtime, a command that scrubbed its
    own environment, a tool double with no context) is treated as a session that
    may not delegate and is refused, while the empty value is how the writer
    CLEARS a marker the child inherited from an ancestor — without it the
    allowance would outlive the session it described and the block the operator
    asked to keep would not hold one hop down.
    """
    return _on(os.environ.get(MAY_DELEGATE_ENV, ""))


def agent_shell_opened_run() -> bool:
    """True when an AGENT'S SHELL opened the session about to be created.

    Two routes reach a created session from inside a marker-carrying shell, and
    both describe a session the operator did not open:

    * the documented escape (:data:`ALLOW_NESTED_SESSION_ENV`) waived the
      refusal for a harness or a QA run;
    * a DELEGATING session took the ``exec`` route the rule now allows it
      (:data:`MAY_DELEGATE_ENV`).

    Both halves matter to :func:`stamp_agent_shell_session`: the marker says who
    is asking, the second says the refusal was waived or never applied, and only
    the pair describes a machine-started session. That second half is why this
    is no longer named for the escape hatch alone — since the 2026-09-19
    relaxation an allowed agent-shell ``exec`` is an ORDINARY run, and an
    unstamped one would list in the operator's sidebar as a chat they opened,
    which is the exact bug the original incident produced.
    """
    return in_agent_shell() and (nested_session_allowed() or may_delegate_from_shell())


def without_agent_shell_marker(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` with the marker removed: a SESSION opening a conversation.

    The marker answers one question — "is this process a command an agent's
    tool call started" — and the codebase has places where a session, not an
    agent, opens a conversation for its user: the TUI's own restart
    (:func:`local_operator.reexec.replace_self`), ``/fork``'s new window
    (``tui/app.py``) and BOTH rungs of a notification click
    (``tui/resume_click.py``: the terminal, and the desktop app beside it,
    which is long-lived and would pass the claim on to everything it later
    spawns). Each passes the session's environment to a child
    that runs `lop --resume`, and each is a user gesture, so that child must not
    inherit the parent's answer.

    Measured consequence of not doing it (review round 1, F1): `/fork` and the
    click open a window that dies on the refusal whenever the session they were
    running in was itself started from an agent's shell — which is exactly what
    a QA session driving the real TUI is.
    """
    merged = dict(env)
    merged.pop(AGENT_SHELL_ENV, None)
    # BOTH markers, because the contract above is one sentence: this child is a
    # SESSION opening a conversation. Leaving the delegation allowance behind
    # would hand it an answer about its PARENT's role — and a child that then ran
    # `lop exec` would be admitted on an allowance it does not hold, which is the
    # one thing this rule exists to stop. The escape variable is deliberately NOT
    # stripped: without the marker it decides nothing, and dropping it would
    # rewrite a caller's environment beyond the question being answered.
    merged.pop(MAY_DELEGATE_ENV, None)
    return merged


def harness_child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a HARNESS gives the real CLI it drives.

    The benches and the eval driver under ``scripts/`` run `lop exec` — or the
    TUI in a pty — as subprocesses with a copy of ``os.environ``. Run from an
    agent's shell, which is how this repository tells agents to measure, they
    inherit the marker and every inner invocation would be refused: the bench
    would record "the product is broken" instead of numbers, and the failure
    would read as a regression rather than as a guard (review round 1, F2).
    They declare themselves harnesses instead, which is what
    :data:`ALLOW_NESTED_SESSION_ENV` is for.

    IT ALSO SILENCES THE CHILD'S NOTIFICATIONS, and that is the same rule rather
    than a second one: a harness is a throwaway driver, its child is a session
    nobody is watching, and the benches seed ``hosting: test`` — whose only
    reply is ``Hello from the mock provider!``, a notification body being a
    snippet of the session's own last assistant line. Without this the child
    finishes its turn and puts that sentence on the operator's lock screen: 17
    recorded banner attempts across scratch stores in two days, every one of them
    a drive-by rig. The pair comes from
    :data:`local_operator.tui.notify.ENV_DISABLE` so the switch has ONE
    definition, and hence is a function-local import: ``tui.notify`` pulls the
    terminal and settings modules in with it, and this module is imported from
    the CLI's own startup path.

    A NEW script that drives the real CLI belongs here too, and an existing one
    that stops using it is the drift this helper exists to make visible.
    """
    from local_operator.tui.notify import ENV_DISABLE, ENV_DISABLE_VALUE

    merged = dict(os.environ if env is None else env)
    merged[ALLOW_NESTED_SESSION_ENV] = "1"
    merged[ENV_DISABLE] = ENV_DISABLE_VALUE
    return merged


def refusal_message() -> str:
    """The refusal, as the model that caused it should read it.

    Three routes, named in the order a session discovers whether it has them:
    ``task`` (what the user's instruction asks for and what most sessions that
    reach here will hold), ``hub`` back to the delegating session (the route for
    a role that does not delegate — the case this guard was written for), and
    ``wake`` for anything that must happen later, named as the DELEGATING
    session's to arm rather than the reader's: ``harness.subagent`` prunes
    ``wake`` from every child, so a reader who reached for it would find the
    tool missing on top of the refusal it already got. The text deliberately
    does not mention :data:`ALLOW_NESTED_SESSION_ENV`: teaching the bypass to the
    reader it exists to stop would be the whole change talking itself out of a
    job, and the human-facing documentation is where a person (or a QA run that
    needs the real CLI) looks for it.
    """
    return (
        "this `lop` invocation from inside an agent session cannot open one — "
        "the session it would start is a top-level conversation the operator never "
        "opened, listed in their session list and desktop sidebar as if they "
        "had, and running outside the job manager that lets this session see, "
        "steer, cancel and account for delegated work.\n"
        "Delegated work is launched with the `task` tool. A session that does "
        "not hold `task` may not create subagents at all: do the work yourself, "
        "and say so with `hub` if the slice genuinely cannot be done alone — "
        "`hub` reaches the session that delegated to you and the brief travels "
        "in the message. Work that must happen later is not a child session's to "
        "arm — `wake` is pruned from every child — so a child routes it back to "
        "the session that delegated to it, while a session that holds `wake` "
        "arms it there itself."
    )


def _session_refusal(*, exec_may_be_opened_by_a_delegating_shell: bool) -> str | None:
    """The one evaluation both entry points share.

    The parameter is the ONLY thing that differs between them, and it is a
    parameter rather than two copies of the condition for the reason the module
    has always given: the rule must not come to mean one thing for a scripted
    run and another for a terminal. Read it as: is this entry point a SESSION,
    such that a delegating shell may reach it?
    """
    if not in_agent_shell() or nested_session_allowed():
        return None
    if exec_may_be_opened_by_a_delegating_shell and may_delegate_from_shell():
        return None
    return refusal_message()


def exec_session_refusal() -> str | None:
    """The refusal for ``lop exec``: ``None`` when this run may open a session.

    Allowed when there is no agent-shell marker at all (the operator's own
    terminal), when the documented escape is set, or when this shell's session
    HOLDS ``task`` — the delegation case the rule allows. See the module
    docstring for why the relaxation stops here, and note the direction of the
    last test: it is the marker the ``bash`` tool exports from the session's own
    inventory, so a shell that did not export it is refused.
    """
    return _session_refusal(exec_may_be_opened_by_a_delegating_shell=True)


def interactive_session_refusal() -> str | None:
    """The refusal for ``lop``/``lop --resume ID``/``--tui``.

    Refused for EVERY agent shell, delegating or not: an agent has no terminal,
    so this path puts a front end on the operator's screen rather than opening
    the separate top-level session the relaxation is about (module docstring).
    The escape still waives it — a pty harness drives this front end exactly as
    a bench drives ``exec``.
    """
    return _session_refusal(exec_may_be_opened_by_a_delegating_shell=False)


def stamp_agent_shell_session(
    directory: Path, *, created_here: bool, delegated_workstream: bool = False
) -> bool:
    """Mark a session an agent's shell opened as machine-started.

    Returns whether it did. The seatbelt under BOTH routes in
    :func:`agent_shell_opened_run`, and the reason neither is silent: a run that
    lands in the operator's own store (rather than the isolated one the rules
    ask for) must not become a chat they appear to have opened — and since the
    relaxation the DELEGATING route is an ordinary, unremarkable one, so this is
    now the only thing standing between an allowed `lop exec` and the sidebar
    listing it as the operator's own conversation. ``origin.json`` is the marker
    every listing already filters on — ``is_user_session`` hides anything that
    is not the user's — and the value ``agent-shell`` distinguishes it from a
    ``task`` child.

    ``created_here`` is the caller's answer to "did this call make the
    directory", and it is not decoration: `--resume` adopts a directory that
    may be the operator's OWN conversation, and marking one of those would hide
    their chat — the mirror image of the bug this marks against. Only the
    caller can answer it (an empty directory a moment old and one from last
    week look identical from here), so it is passed rather than guessed.

    ``delegated_workstream`` chooses WHICH machine-started value is written, and
    it is the flag the run's own caller set (`lop exec --workstream`): the
    operator asked for this one as a long-lived parallel workstream, so it is
    registered as a USER origin and every listing offers it — with the opener
    recorded beside it (:func:`_origin_attribution`), because a visible
    machine-started row that reads as the operator's own conversation is the
    2026-09-18 confusion rather than the fix. Absent, the run is an ephemeral
    agent-shell run and stays hidden, which is what every caller that predates
    the flag gets.

    The GUARD above is deliberately not part of that choice: a run is stamped
    only when this call created the directory and an agent's shell started it,
    in both modes — `--workstream` on the operator's own terminal is a no-op (no
    marker at all, so the run reads as their own session), and
    `--workstream --resume <their conversation>` is not re-stamped either.

    Called from :func:`local_operator.session_factory._prepare`, which is
    the one place every session — foreground exec, the detached worker, the
    interactive viewer's runtime, the server — gets its directory. It began in
    ``exec_session.run_session`` and moved up after review round 2: the exec
    path stamped, the interactive path did not, and the docs promised both
    (F1 of that round). Best-effort by contract, like
    :func:`local_operator.resume.mark_session_origin` itself: the session is
    about to do real work, and failing to write bookkeeping about it must not
    take that work down.
    """
    if not created_here or not agent_shell_opened_run():
        return False
    from local_operator.resume import (
        ORIGIN_AGENT_SHELL,
        ORIGIN_AGENT_WORKSTREAM,
        mark_session_origin,
    )

    if delegated_workstream:
        mark_session_origin(
            Path(directory), ORIGIN_AGENT_WORKSTREAM, opened_by=_origin_attribution()
        )
        return True
    mark_session_origin(Path(directory), ORIGIN_AGENT_SHELL)
    return True


#: The marker key a workstream records its opener under. One nested object
#: rather than flat keys because it is one fact, and because the desktop wire
#: publishes exactly this object's members (``resume.OPENED_BY_KEYS``): a flat
#: spelling here would need a translation step, which is a second place for the
#: two to drift.
ORIGIN_OPENED_BY_KEY = "opened_by"


def _origin_attribution() -> dict[str, str | None]:
    """WHO asked for this workstream, as far as THIS process can honestly tell.

    Read at stamp time and never re-derived, because this is the only moment
    the answer exists: the requesting session's identity reaches the child as
    the environment the ``bash`` tool signed, and nothing in the child's own
    directory says who started it.

    The sources, all of them read-only and best-effort:

    * the requesting session's directory, taken from
      :data:`local_operator.scratchpad.SCRATCHPAD_PATH_ENV` — the one variable a
      session's shell exports that NAMES the session it belongs to (its parent
      directory IS that session's directory, and its name is that session's id).
      Absent for a run whose parent has no session directory, which is why every
      member is nullable rather than this being an error;
    * that session's conversation name (``resume.session_name``), recorded for the
      durable marker;
    * that session's own task label and role, when it is itself a delegated
      child — the same ``label``/``agent`` pair ``harness.subagent`` writes into
      a child's marker, read back from its own, so a fan-out records WHICH
      delegated slice asked for the workstream. A top-level session has no such
      marker and honestly reports ``None``.

    EVERY member that cannot be read is ``None``, never invented and never
    inferred from a neighbouring field: an attribution that guessed would be
    worse than none, because the row it decorates is the operator's only way to
    tell whose work this is.
    """
    import json

    attribution: dict[str, str | None] = {
        key: None for key in ("agent", "label", "session", "name")
    }
    from local_operator.scratchpad import SCRATCHPAD_PATH_ENV

    raw = os.environ.get(SCRATCHPAD_PATH_ENV, "").strip()
    if not raw:
        return attribution
    # ``<config dir>/sessions/<session id>/scratchpad``: the ROOT is what the
    # child is handed, so the session's own directory is its parent. A value
    # shaped some other way names no session this rule can describe, so it is
    # reported as no attribution rather than as a guess.
    requesting = Path(raw).parent
    if not requesting.name:
        return attribution
    attribution["session"] = requesting.name
    try:
        from local_operator.resume import session_name

        attribution["name"] = session_name(requesting) or None
    except Exception:  # noqa: BLE001 — attribution never breaks a stamp
        logger.debug("workstream opener name unavailable", exc_info=True)
    try:
        from local_operator.resume import ORIGIN_NAME

        payload = json.loads((requesting / ORIGIN_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return attribution
    if not isinstance(payload, dict):
        return attribution
    for key in ("agent", "label"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            attribution[key] = value
    return attribution

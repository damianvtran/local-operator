"""One rule: a command an agent ran may not open a session of its own.

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
and it is not a child by this harness's definition (children are one level deep,
capacity-gated and observed through the parent's roster). Two facts made it the
path of least resistance for a model trying to do the right thing: ``--profile``
is advertised in ``lop exec --help``, and a role that does not delegate holds no
``task`` tool to use instead — see ``harness.subagent``'s prune, which drops
``task``/``wait``/``wake`` for exactly the ``coder``-shaped role that was
running here.

So the rule is enforced where the act happens: a ``lop`` invocation that
descends from an agent's shell may not open a session, and is told what to do
instead — ``task`` when this session holds it, ``hub`` back to the session that
delegated when it does not, and ``wake`` for work that belongs later.

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
the real path). It is deliberately NOT named in :func:`nested_session_refusal`
— that text is model-facing, and its job is to route the model to
``task``/``hub``/``wake`` — but it is documented for the human in
``docs/EXEC.md``. It is not an equivalent path: a session opened under it is
stamped :data:`local_operator.resume.ORIGIN_AGENT_SHELL`, so even then the run
stays out of the picker, the sidebar and the phone's list rather than appearing
as a chat the operator opened (:func:`stamp_escaped_session`).

The pytest suite never sees either variable. ``LOCAL_OPERATOR_AGENT_SHELL`` and
``LOCAL_OPERATOR_ALLOW_NESTED_SESSION`` are both scrubbed by
``tests/conftest.py``'s ambient-environment fixture, so a suite run by an agent
does not refuse the sessions it builds on purpose, and a guard test that wants
the marker sets it for itself instead of inheriting it from whichever harness
happened to launch pytest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: Set by the ``bash`` tool on every command it runs. Names the one fact this
#: module acts on: the process descended from an agent's tool call.
AGENT_SHELL_ENV = "LOCAL_OPERATOR_AGENT_SHELL"

#: The documented escape for harness tests and QA runs that must drive the real
#: CLI. Named in ``docs/EXEC.md``, never in the refusal the model reads.
ALLOW_NESTED_SESSION_ENV = "LOCAL_OPERATOR_ALLOW_NESTED_SESSION"

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


def escaped_agent_shell_run() -> bool:
    """True for a session opened under the escape: in an agent shell, allowed.

    Both halves matter to the caller in :func:`stamp_escaped_session`: the
    marker says who is asking, the escape says the refusal was waived, and only
    the pair describes a session that must be marked as machine-started.
    """
    return in_agent_shell() and nested_session_allowed()


def refusal_message() -> str:
    """The refusal, as the model that caused it should read it.

    Three routes, named in the order a session discovers whether it has them:
    ``task`` (what the user's instruction asks for and what most sessions that
    reach here will hold), ``hub`` back to the delegating session (the route for
    a role that does not delegate — the case this guard was written for), and
    ``wake`` for anything that must happen later. The text deliberately does not
    mention :data:`ALLOW_NESTED_SESSION_ENV`: teaching the bypass to the reader
    it exists to stop would be the whole change talking itself out of a job, and
    the human-facing documentation is where a person (or a QA run that needs the
    real CLI) looks for it.
    """
    return (
        "a `lop` invocation from inside an agent session cannot open one — the "
        "session it would start is a top-level conversation the operator never "
        "opened, listed in their session list and desktop sidebar as if they "
        "had, and running outside the job manager that lets this session see, "
        "steer, cancel and account for delegated work.\n"
        "Launch delegated work with the `task` tool instead. A session without "
        "it — a role that does not delegate runs one level deep and loses "
        "`task`/`wait`/`wake` — asks the session that delegated to it, with "
        "`hub`, to launch the child; the brief travels in the message. Work "
        "that must happen later belongs in `wake`."
    )


def nested_session_refusal() -> str | None:
    """The refusal to print, or ``None`` when starting a session is allowed.

    The single predicate both entry points call (``cli``'s ``exec`` branch and
    its interactive path), so the rule cannot come to mean one thing for a
    scripted run and another for a terminal.
    """
    if not in_agent_shell() or nested_session_allowed():
        return None
    return refusal_message()


def stamp_escaped_session(session: Any) -> bool:
    """Mark an escaped run's session as machine-started. Returns whether it did.

    The seatbelt under the escape hatch, and the reason the hatch is not silent:
    an opted-in run that lands in the operator's own store (rather than the
    isolated one the rules ask for) must not become a chat they appear to have
    opened. ``origin.json`` is the marker every listing already filters on —
    ``is_user_session`` hides anything that is not the user's — and the value
    ``agent-shell`` distinguishes it from a ``task`` child.

    Best-effort by contract, like :func:`local_operator.resume.mark_session_origin`
    itself: the session already exists and is about to do real work, so failing
    to write bookkeeping about it must not take that work down. Called from
    :func:`local_operator.exec_session.run_session` — the one place both exec
    shapes (foreground and the detached worker) hold the built session — rather
    than at session construction, because ``--resume`` can adopt a directory
    that is already the user's and must not be re-marked as this module's.
    """
    if not escaped_agent_shell_run():
        return False
    directory = getattr(getattr(session, "transcript", None), "directory", None)
    if directory is None:
        # A reduced host with an in-memory store: there is no directory to
        # mark, which is also no store for a picker to mis-offer.
        return False
    from local_operator.resume import ORIGIN_AGENT_SHELL, mark_session_origin

    mark_session_origin(Path(directory), ORIGIN_AGENT_SHELL)
    return True

"""Bounded exec counterparts of owner-side startup commands.

Resolve before provider construction AND again in a detached worker: a saved
registry may change between preflight and spawn. Never interpret prompt text as
slash commands, and never reinterpret the legacy --agent selector as a role.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

#: Startup selectors that survive the ``exec --background`` process boundary.
#: Order is the order :func:`local_operator.exec_mode.build_worker_argv` emits
#: them in, and each one must exist as a field on ``ExecArgs`` and as an option
#: on the worker's parser (both are fed by this module).
STARTUP_FIELDS = (
    "team",
    "profile",
    "goal",
    "clear_goal",
    "loop",
    "loop_goal",
    "name",
    "effort",
    "tools",
    # Not a selector a session applies (`apply_startup` never reads it): this one
    # decides what the run IS to every listing, and it belongs in this tuple for
    # the property the tuple exists for — a `--background` run is the same
    # request run elsewhere, so a flag dropped here would be accepted by the
    # front end and silently lost, and the workstream the operator asked for
    # would come back hidden.
    "workstream",
)

#: Separator between tool names in ``--tools``. A COMMA, not a space: the flag's
#: value crosses the ``--background`` boundary through ``build_worker_argv``'s
#: ``--opt=value`` form, which is one argv item by construction — a space would
#: split the declaration into two items and the worker's parser would read the
#: second half as a stray positional prompt.
_TOOLS_SEPARATOR = ","


def parse_tool_inventory(text: str | None) -> tuple[str, ...] | None:
    """Parse a ``--tools`` declaration into tool names, or ``None`` when unset.

    ``None`` and ``()`` are DIFFERENT answers and the difference is
    load-bearing at the one call site: ``None`` means "no declaration, this
    session reaches whatever the host built", while an empty tuple would be a
    session that may reach nothing at all. A value that parses to nothing
    (``--tools ''``, ``--tools ,,``) is refused by :func:`resolve_startup`
    rather than silently becoming the second one, because an operator who typed
    a declaration and got a tool-less session would read that as a harness bug.
    """
    if text is None:
        return None
    names = (name.strip() for name in text.split(_TOOLS_SEPARATOR))
    return tuple(dict.fromkeys(name for name in names if name))


def _name_tuple(value: Any) -> tuple[str, ...] | None:
    """A session-supplied name sequence, or ``None`` when the host cannot answer.

    Deliberately NOT a bare ``getattr(..., ())``: a reduced host fabricates any
    attribute it is asked for (a ``Mock`` session in a test, a front end that
    predates the member), so the default is never reached and the caller gets
    something truthy and uniterable. Both reads below sit on the startup path,
    where raising costs the run, so "cannot say" has to be distinguishable from
    "declares nothing" — hence ``None`` rather than ``()``.
    """
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(value)


def report_unresolved_declared_tools(session: Any, args: Any) -> None:
    """Report a ``--tools`` declaration that named tools the run never reached.

    Called at the END of the run (see ``exec_session``), not at startup — and
    that placement is the whole point rather than tidiness. MCP servers connect
    in the background, so at startup an unreachable name and a server that has
    not finished connecting are INDISTINGUISHABLE: a startup report either fires
    a false alarm on the ordinary path, or — if it waits for the round to settle
    — never fires at all. Both halves of that were measured on a live run. By the
    end nothing is in flight, so the answer is definitive.

    Why report at all: a declaration that matches nothing fails CLOSED — the
    session simply has no tools — which is the right direction, but from the
    model's side it is indistinguishable from a harness fault. The only symptom
    is "Tool not found" for every call and a run that answers nothing, so a typo
    in a security control would read as a broken harness.

    Silent for the ROLE-derived case, deliberately: a profile naming a tool that
    exists on another machine is ordinary and documented (see
    ``agent_profiles.filter_tools``), where a name the operator typed into
    ``--tools`` for THIS run is a typo worth a log line.
    """
    declared = parse_tool_inventory(getattr(args, "tools", None))
    if not declared:
        return
    unresolved = getattr(session, "unresolved_declared_tools", None)
    if not callable(unresolved):
        return
    try:
        unreached = [name for name in (_name_tuple(unresolved()) or ()) if name in set(declared)]
    except Exception:  # noqa: BLE001 — a report must never cost the run
        logger.debug("declared-tool resolution check failed", exc_info=True)
        return
    if not unreached:
        return
    reachable = _name_tuple(getattr(session, "tool_inventory", None)) or ()
    print(
        "Warning: --tools names no tool this run could reach: "
        + ", ".join(unreached)
        + f". The session reached {len(reachable)} tool(s).",
        file=sys.stderr,
    )


def add_startup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--team", metavar="NAME", help="Attach a saved team with its manager, roster and briefs"
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="Attach a reusable role/specialist (/agent counterpart; not legacy --agent)",
    )
    goals = parser.add_mutually_exclusive_group()
    # Deliberately NOT the TUI's /goal, which also submits the text as a message
    # so one Enter starts the work. Here the positional prompt IS the message
    # channel, so auto-submitting would double-send whenever both are given.
    goals.add_argument(
        "--goal",
        metavar="TEXT",
        help=(
            "Set a literal standing goal. Unlike the TUI's /goal this does NOT also "
            "send the text as a message: pair it with a prompt or --loop"
        ),
    )
    goals.add_argument("--clear-goal", action="store_true", help="Clear the resumed standing goal")
    loops = parser.add_mutually_exclusive_group()
    loops.add_argument(
        "--loop",
        type=int,
        metavar="N",
        help="Run N continuation iterations after the optional prompt; needs a standing goal",
    )
    loops.add_argument(
        "--loop-goal",
        metavar="TEXT",
        help="Continue and judge this goal until achieved (no fixed iteration count)",
    )
    parser.add_argument("--name", metavar="TEXT", help="Set the persisted conversation name")
    parser.add_argument(
        "--tools",
        metavar="NAMES",
        help=(
            "Comma-separated tools this run may reach, and the ONLY ones. Excluded "
            "tools are unreachable, not merely unapproved. The declaration stands as "
            "the APPROVAL for its own members only where nobody can be asked — a "
            "non-tty run without --control; on a terminal, and under --control, every "
            "write/exec call is still decided by the approval gate. A name this build "
            "does not have is unreachable and reported"
        ),
    )
    parser.add_argument(
        "--effort",
        metavar="LEVEL",
        help="Set reasoning effort; validated against the selected model",
    )
    parser.add_argument(
        "--workstream",
        action="store_true",
        help=(
            "Publish this run as a long-lived parallel WORKSTREAM the operator asked "
            "for: it is listed in the sidebar, /resume and the phone list, carries the "
            "session that opened it, and can be followed and steered. Without it an "
            "agent-opened run stays ephemeral: hidden everywhere and silent. Implies "
            "--control, because a row can only be steered where a live discovery "
            "record exists. A no-op outside an agent's shell"
        ),
    )


def resolve_startup(args: Any) -> Any:
    """Validate independent inputs without constructing a session or model."""
    if getattr(args, "agent_name", None) and getattr(args, "agent_id", None):
        raise ValueError("--agent and --agent-id are mutually exclusive")
    if getattr(args, "goal", None) is not None and getattr(args, "clear_goal", False):
        raise ValueError("--goal and --clear-goal are mutually exclusive")
    count = getattr(args, "loop", None)
    goal = getattr(args, "loop_goal", None)
    if count is not None:
        from local_operator.session.goal_loop import MAX_LOOP_ITERATIONS

        if goal is not None:
            raise ValueError("--loop and --loop-goal are mutually exclusive")
        if not 1 <= count <= MAX_LOOP_ITERATIONS:
            raise ValueError(f"--loop must be between 1 and {MAX_LOOP_ITERATIONS}")
    if goal is not None and not goal.strip():
        raise ValueError("--loop-goal must not be empty")
    # Decidable WITHOUT constructing anything: --loop needs a standing goal, and
    # only a resumed session can supply one this run did not pass. Refusing in
    # apply_startup instead left an empty session directory behind on every
    # typo, which the goal-alone path (refused at preflight) never does.
    if (
        count is not None
        and getattr(args, "goal", None) is None
        and not getattr(args, "resume", None)
    ):
        raise ValueError("--loop requires --goal, or --resume a session that has one")
    # Refused at PREFLIGHT, like ``--loop`` above, and for the same reason: a
    # declaration that can never match anything strands the session with no
    # tools, and refusing it in apply_startup would leave a session directory
    # behind to explain that to the operator.
    if getattr(args, "tools", None) is not None and not parse_tool_inventory(args.tools):
        raise ValueError(
            "--tools must name at least one tool; omit it to leave the session unrestricted"
        )
    team = None
    if getattr(args, "team", None):
        from local_operator.paths import config_dir
        from local_operator.teams import TeamRegistry

        team = TeamRegistry(config_dir()).get_team_by_name(args.team)
        if team is None:
            raise ValueError(f"No team named {args.team!r}; use 'lop teams list'")
    if getattr(args, "profile", None):
        from local_operator.agent_profiles import resolve_profile_or_specialist
        from local_operator.agents import AgentRegistry
        from local_operator.paths import config_dir

        if (
            resolve_profile_or_specialist(args.profile, registry=AgentRegistry(config_dir()))[0]
            is None
        ):
            # NOT 'lop agents list': that lists legacy AgentData records, which
            # is the --agent/--agent-id world this flag is distinct from. On a
            # fresh config it prints "No agents found." while --profile reviewer
            # resolves a packaged seed perfectly well, so the one message whose
            # job is "here is how to find the right name" pointed at the surface
            # that structurally cannot contain it. Name the seeds instead.
            from local_operator.agent_profiles import list_seeds

            available = ", ".join(sorted(list_seeds()))
            raise ValueError(
                f"No role or specialist named {args.profile!r}. "
                f"Available roles: {available}. Add your own with 'lop agents create'"
            )
    return team


def declared_tool_inventory(session: Any, args: Any) -> tuple[str, ...] | None:
    """The inventory this run declares, or ``None`` for an unrestricted session.

    ``--tools`` is the explicit form and WINS over an attached role's own
    ``tools:`` allow-list: a supervisor that composes a prompt and names the
    tools for THIS run is stating the run's reach, and a role's allow-list is a
    weaker statement about the role.

    Falling back to the role's allow-list is what makes a role-declared surface
    hold for a headless session at all. ``Session.attach_agent_profile`` stamps
    a profile's INSTRUCTIONS onto the session it is attached to, and its
    ``tools:`` allow-list was enforced only where the profile is launched as a
    subagent (``harness.subagent``). A ``lop exec --profile reviewer`` therefore
    ran a reviewer that could still ``write`` and ``edit`` — the precise
    capability the reviewer seed's allow-list exists to remove, and the failure
    that forces a re-review of a diff the reviewer itself changed — with no way
    for the unattended run to say otherwise.

    ``None`` means "unrestricted", which is the default for both an absent
    ``--tools`` and a role that declares no ``tools:`` of its own: the negative
    case must stay byte-for-byte today's behaviour.
    """
    explicit = parse_tool_inventory(getattr(args, "tools", None))
    if explicit is not None:
        return explicit
    attached = _name_tuple(getattr(session, "attached_profile_tools", None)) or ()
    return attached or None


def apply_startup(session: Any, args: Any, team: Any) -> None:
    """Apply explicit overrides after ordinary resume restored its attachment."""
    if team is not None:
        session.attach_team(team)
    if getattr(args, "profile", None) and not session.attach_agent_profile(args.profile):
        raise ValueError(f"Could not attach profile {args.profile!r}")
    # AFTER the attach, because a role's own allow-list is one of the two
    # sources; and here rather than in the session factory, because an ordinary
    # resume restores the attachment INSIDE the session and only this step runs
    # afterwards. ``--control`` reads as attended: the supervisor's gate replaces
    # the session's (see ``session.runtime.exec_control``), so a declared tool
    # must still be asked about rather than waved through.
    inventory = declared_tool_inventory(session, args)
    if inventory is not None:
        # "No supervisor" is NOT "no human", and conflating them silently
        # removed a live safety prompt: a declaration stands as the APPROVAL for
        # its own members only where there is nobody to ask, and the tree's
        # existing test for that is a tty — ``session_factory._make_request_approval``
        # prompts y/N on one and denies without one (CL-04), which is the
        # contract ``docs/EXEC.md`` states. So `lop exec --tools bash,write "…"`
        # typed at a terminal keeps the per-call prompt on THIS command, while a
        # piped or ``--background`` run stays auto-approved (the detached worker
        # is spawned with ``stdin=DEVNULL``, so it answers this the same way a
        # pipe does). Deriving it from the terminal rather than from the flag
        # also keeps this correct for any future host that runs attended without
        # ``--control``.
        #
        # ``sys.stdin`` is checked for ``None`` BEFORE ``isatty()`` is called
        # because a launcher or daemoniser may start the process with fd 0
        # CLOSED, and Python leaves ``sys.stdin`` as ``None`` for that shape
        # rather than raising on it. The unconditional call crashed every such
        # run at startup with ``'NoneType' object has no attribute 'isatty'`` —
        # before it reached a single provider request — so an absent stdin has to
        # read the same way a pipe does: nobody can be asked. Same guard, and the
        # same reason, as ``session_factory._make_request_approval``'s own tty
        # test at its first approval.
        stdin_is_tty = sys.stdin is not None and sys.stdin.isatty()
        session.set_tool_inventory(
            inventory,
            unattended=not getattr(args, "control", False) and not stdin_is_tty,
        )
    if getattr(args, "clear_goal", False):
        session.set_goal("")
    elif getattr(args, "goal", None) is not None:
        session.set_goal(args.goal)
    if getattr(args, "name", None) is not None:
        session.set_conversation_name(args.name)
    # The second half of the check resolve_startup starts: only here, with the
    # session built, can a RESUMED standing goal be consulted. resolve_startup
    # already rejected the decidable case (no --goal and no --resume) before
    # anything was constructed, so reaching this raise means the resumed session
    # genuinely has no goal to continue.
    if getattr(args, "loop", None) is not None and not session.goal:
        raise ValueError("--loop requires --goal or a resumed standing goal")

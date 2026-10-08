"""Bounded exec counterparts of owner-side startup commands.

Resolve before provider construction AND again in a detached worker: a saved
registry may change between preflight and spawn. Never interpret prompt text as
slash commands, and never reinterpret the legacy --agent selector as a role.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
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
    # The final-response output contract, appended here rather than in a
    # bespoke branch of ``build_worker_argv``: membership is what makes a
    # ``--background`` run the same request run elsewhere, AND what makes the
    # ``--status`` no-run-options guard cover them without a second list to
    # drift from (its guard reads this tuple).
    "output_format",
    "output_schema",
    "output_retries",
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
            "agent-opened run stays ephemeral: hidden everywhere and silent. Changes "
            "only the row's visibility, never how approvals are handled (pair it with "
            "--control for that). Stamps nothing outside an agent's shell"
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("markdown", "json", "yaml", "toml"),
        default=None,
        help=(
            "Enforce the format of the assistant's FINAL response. The run fails "
            "(exit 1) if no valid payload is produced within --output-retries. "
            "Default: no enforcement — behaviour is unchanged."
        ),
    )
    parser.add_argument(
        "--output-schema",
        metavar="PATH",
        help=(
            "JSON Schema file the decoded JSON/YAML/TOML payload must satisfy. For "
            "--output-format markdown: a JSON object {'required_sections': [...]}. "
            "Requires --output-format."
        ),
    )
    parser.add_argument(
        "--output-retries",
        type=int,
        metavar="N",
        help=(
            "Max retries after a failed final response (0-5; default 2). "
            "Requires --output-format."
        ),
    )


def resolve_output_contract(args: Any) -> Any:
    """The output contract this invocation declares, or ``None`` when unenforced.

    Validation lives HERE, next to the other preflight refusals and before any
    session exists: the schema FILE is read, parsed and meta-schema-checked now,
    so a bad schema fails as ``exec failed: …`` with no session directory left
    behind — and the detached worker re-resolves identically on its own side of
    the process boundary.

    All three reads use ``getattr(..., None)`` defaults, because the SDK's wider
    namespace deliberately does not carry these keys: an SDK schema is a type or
    mapping rather than a file path, so ``sdk._build_session`` constructs the
    contract itself (through the ``OutputContract`` validation both paths
    share) and an absent key here must read as "not declared", never raise.
    """
    output_format = getattr(args, "output_format", None)
    output_schema = getattr(args, "output_schema", None)
    retries = getattr(args, "output_retries", None)
    if output_format is None:
        if output_schema is not None:
            raise ValueError("--output-schema requires --output-format")
        if retries is not None:
            raise ValueError("--output-retries requires --output-format")
        return None
    if retries is not None and not (isinstance(retries, int) and 0 <= retries <= 5):
        raise ValueError("--output-retries must be between 0 and 5")
    if isinstance(output_schema, str) and not output_schema.strip():
        # An empty value is REFUSED rather than read as "no schema": the run
        # would otherwise enforce the format while silently dropping the
        # schema the operator asked for, and the SDK refuses the same value
        # through ``OutputContract`` — the two paths must agree (review R-4).
        # Whitespace is included: no file path is only whitespace.
        raise ValueError("--output-schema must name a schema file (an empty value is refused)")
    from local_operator.output_contract import OutputContract, OutputContractError

    schema = _load_output_schema(output_format, output_schema) if output_schema else None
    try:
        return OutputContract(
            format=output_format,
            schema=schema,
            retries=2 if retries is None else retries,
        )
    except OutputContractError as error:
        # Re-raised as a plain ValueError so the caller's existing
        # ``except (ValueError, OSError)`` arm prints it behind ``exec failed:``
        # with the flag vocabulary the operator actually typed.
        raise ValueError(str(error)) from error


def _load_output_schema(output_format: str, path: str) -> Any:
    """Read one ``--output-schema`` FILE into the contract's schema vocabulary.

    The file speaks JSON in every mode — a JSON Schema for json/yaml/toml, or
    the markdown sections object — and every failure names the flag and the
    path, because this message is the operator's only channel for a file they
    wrote and cannot see re-parsed.
    """
    import json

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read --output-schema file '{path}': {error}") from error
    try:
        loaded = json.loads(text)
    except ValueError as error:
        raise ValueError(f"--output-schema file is not valid JSON: {error}") from error
    if output_format == "markdown":
        # The one schema vocabulary markdown has, checked to the exact shape
        # the flag documents: a file that "almost" matches refuses now rather
        # than validating nothing later.
        if (
            not isinstance(loaded, dict)
            or set(loaded) != {"required_sections"}
            or not isinstance(loaded["required_sections"], list)
            or not all(isinstance(item, str) for item in loaded["required_sections"])
        ):
            raise ValueError(
                '--output-schema for markdown must be {"required_sections": ["<section>", ...]}'
            )
        return loaded
    from jsonschema import Draft202012Validator

    try:
        Draft202012Validator.check_schema(loaded)
    except Exception as error:  # noqa: BLE001 — jsonschema's own failure classes
        raise ValueError(
            f"--output-schema is not a valid JSON Schema: {_schema_failure(error)}"
        ) from error
    return loaded


def _schema_failure(error: Exception) -> str:
    """jsonschema's ``SchemaError`` carries a one-line ``message``."""
    message = getattr(error, "message", None)
    return " ".join(str(message if message else error).split())


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
    if team is not None and getattr(args, "profile", None):
        # ISSUE #2014, refused at PREFLIGHT — before a session or a model exists
        # — rather than in ``apply_startup``: a team owns the agent slot, so the
        # pair can never be honoured, and refusing it late would leave a session
        # directory behind to explain a command that was never going to run
        # (the same reasoning as ``--loop`` and ``--tools`` above).
        #
        # NO MANAGER CARVE-OUT: ``--team X --profile <X's manager>`` is refused
        # too, because the manager is already the speaker the team attached —
        # the pair restates what is in force rather than changing anything.
        #
        # The sentence is the ONE the runtime raises
        # (``team_owns_the_agent_slot_message``): the desktop's header lane keys
        # on those words, and a --profile that reads like /agent (docs/EXEC.md)
        # must be refused in the words of /agent.
        from local_operator.session.errors import team_owns_the_agent_slot_message

        raise ValueError(
            # The flag-level fact FIRST (design round 1, D6): the shared sentence
            # below speaks to a user of ``/agent`` in a session, and a shell
            # user who typed two flags needs to be told which pair is wrong
            # before being told how the SESSION feels about it. The sentence
            # itself stays VERBATIM — the desktop lane keys on those words.
            "--profile cannot be combined with --team: a team owns the session's "
            "agent slot. "
            + team_owns_the_agent_slot_message(
                str(getattr(team, "name", "") or args.team),
                str(getattr(team, "manager", "") or ""),
            )
        )
    if getattr(args, "profile", None):
        from local_operator.agent_profiles import resolve_profile_or_specialist
        from local_operator.agents import AgentRegistry, agents_store_present
        from local_operator.paths import config_dir

        config_root = config_dir()
        # GUARDED, and it is the FIRST writer on the launch path (review round
        # 1, finding 2): the registry CONSTRUCTOR creates ``config_dir`` and
        # ``config_dir/agents`` (then runs migrations), so building one just to
        # validate a NAME wrote the operator's roots on every ``--profile`` run
        # — before the advisory probe downstream could even reach its
        # fresh-root branch. With no store on disk there are no registered
        # roles to shadow the packaged seeds with, so ``registry=None``
        # resolves the same names; a root whose roles live only in a legacy
        # ``agents.json`` still constructs, because the registry is the code
        # that reads — and migrates — it. Shared predicate
        # (``agents_store_present``), so both writers agree on what "a store is
        # present" means.
        registry = AgentRegistry(config_root) if agents_store_present(config_root) else None
        if resolve_profile_or_specialist(args.profile, registry=registry)[0] is None:
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
    # The output contract's flags are validated HERE, before anything is
    # constructed, exactly like ``--tools`` above: the schema file is read and
    # meta-schema-checked now, so a bad schema is ``exec failed: …`` with no
    # session directory behind it — and the detached worker refuses the same
    # way when it re-resolves on its own side of the process boundary. The
    # result is discarded on purpose: the contract is built again from the same
    # flags in ``apply_startup``, the one place a contract can be installed.
    resolve_output_contract(args)
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
    # The output contract is built and installed HERE — re-resolved from the
    # same flags, not carried over from resolve_startup's validation pass — so
    # the object the loop reads belongs to this session's invocation. ``None``
    # (no --output-format) installs nothing: the byte-identical default.
    contract = resolve_output_contract(args)
    if contract is not None:
        session.set_output_contract(contract)
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

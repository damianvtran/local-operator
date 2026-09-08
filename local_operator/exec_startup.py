"""Bounded exec counterparts of owner-side startup commands.

Resolve before provider construction AND again in a detached worker: a saved
registry may change between preflight and spawn. Never interpret prompt text as
slash commands, and never reinterpret the legacy --agent selector as a role.
"""

from __future__ import annotations

import argparse
from typing import Any

STARTUP_FIELDS = ("team", "profile", "goal", "clear_goal", "loop", "loop_goal", "name", "effort")


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
        "--effort",
        metavar="LEVEL",
        help="Set reasoning effort; validated against the selected model",
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
            raise ValueError(f"No role or specialist named {args.profile!r}; use 'lop agents list'")
    return team


def apply_startup(session: Any, args: Any, team: Any) -> None:
    """Apply explicit overrides after ordinary resume restored its attachment."""
    if team is not None:
        session.attach_team(team)
    if getattr(args, "profile", None) and not session.attach_agent_profile(args.profile):
        raise ValueError(f"Could not attach profile {args.profile!r}")
    if getattr(args, "clear_goal", False):
        session.set_goal("")
    elif getattr(args, "goal", None) is not None:
        session.set_goal(args.goal)
    if getattr(args, "name", None) is not None:
        session.set_conversation_name(args.name)
    if getattr(args, "loop", None) is not None and not session.goal:
        raise ValueError("--loop requires --goal or a resumed standing goal")

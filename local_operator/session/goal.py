"""The session goal — a durable objective the agent keeps in view.

Why a shared mutable holder rather than a plain string on the session: the
system-prompt provider closure is built BEFORE the session facade exists (the
session is constructed with the provider already wired), so the two cannot
reference each other directly. Both are handed the same ``GoalState``, which
makes a ``/goal`` change visible to the very next turn's prompt without
rebuilding the session or reaching through private attributes.

The goal rides the VOLATILE TAIL of the system blocks (see
``prompts_api.build_system_blocks``). That placement is a cache decision:
editing the goal invalidates only the tail, never the stable instruction /
tool-inventory / env prefix that keeps the provider cache warm.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Hard cap on a stored goal. A goal is a short objective, not a spec dump;
#: capping it keeps the volatile tail small and bounds the per-turn cost.
MAX_GOAL_CHARS = 2000


@dataclass
class GoalState:
    """Mutable holder for the session's current goal (empty = unset)."""

    text: str = ""
    #: Team brief stamped by ``/team``. Separate from ``text`` so attaching a
    #: team cannot overwrite a standing ``/goal``, and clearing a goal cannot
    #: drop the roster the manager is coordinating.
    team_brief: str = ""
    #: Agent-profile brief stamped by ``/agent``. Its OWN field rather than a
    #: suffix of ``team_brief`` because the two are attached by different
    #: commands with different lifetimes: a later ``/agent`` replaces only the
    #: previous agent brief, and it must never eat the roster a running
    #: ``/team`` manager is still coordinating (nor vice versa).
    agent_brief: str = ""

    def set(self, text: str) -> str:
        """Store a trimmed, length-capped goal and return what was stored."""
        cleaned = (text or "").strip()
        if len(cleaned) > MAX_GOAL_CHARS:
            cleaned = cleaned[:MAX_GOAL_CHARS]
        self.text = cleaned
        return self.text

    def clear(self) -> None:
        self.text = ""

    def is_set(self) -> bool:
        return bool(self.text)

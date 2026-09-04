"""The session goal — a durable objective the agent keeps in view.

Why a shared mutable holder rather than a plain string on the session: the
system-prompt provider closure is built BEFORE the session facade exists (the
session is constructed with the provider already wired), so the two cannot
reference each other directly. Both are handed the same ``GoalState``, which
makes a ``/goal`` change visible to the very next turn's prompt without
rebuilding the session or reaching through private attributes.

The goal is part of the desired session-state section. Production sessions
retain their first system-prefix snapshot and append subsequent goal changes
as host-state records before the next model request. This preserves history's
cache prefix while keeping the newest goal authoritative.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    #: The DISPLAY NAME of the profile ``agent_brief`` was stamped from ("" when
    #: none). Kept beside the brief rather than derived from it because the band
    #: needs to NAME the active profile (U2), and the brief is an opaque
    #: instruction blob with no reliable name inside it — a role preamble, a
    #: wrapped specialist prompt, or empty for a resolved-but-hollow profile
    #: (A2), which still counts as attached. The two move together: every stamp
    #: sets both, and ``clear_agent_profile`` blanks both.
    agent_name: str = ""
    #: Live probe answering "is an interactive surface watching this session
    #: right now?" — set by the runtime, which is the only component that
    #: knows (it owns the control socket's connection table). ``None`` means
    #: "no probe installed", which every non-runtime host leaves alone and
    #: which reads as interactive: a plain CLI or a test has a person in
    #: front of it by construction.
    #:
    #: A PROBE rather than a stored flag on purpose. Attach state changes
    #: whenever a viewer opens or closes, and a cached copy would need an
    #: event per change — the token accumulation this exists to avoid. The
    #: prompt closure calls this at turn start and the answer costs one line
    #: whatever happened in between.
    interactive_probe: "Callable[[], bool] | None" = None

    def is_interactive(self) -> bool:
        """Whether a surface can answer a question right now (default True)."""
        probe = self.interactive_probe
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 — an unreadable probe must not kill a turn
            return True

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

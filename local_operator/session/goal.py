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

#: The ``/goal`` arguments that UNSET the standing goal instead of becoming one.
#:
#: ONE set for the three hosts that implement ``/goal`` — the TUI's local handler,
#: its routed one, and the detached runtime's — because a word honoured on one
#: host and stored as a goal body on another is the worst of the two outcomes:
#: the user's intent is executed in one window and silently becomes the standing
#: objective in another. The bare words predate the flag; ``--clear`` is the
#: discoverable form the palette and the argument picker now teach.
#:
#: The flag is matched as the WHOLE argument, never as a prefix: ``/goal --clear``
#: is a flag, while ``/goal --clear the flaky job`` is still free text the user
#: meant as an objective. Eating the tail of a real goal would be silent data
#: loss in the one command whose argument the MODEL is told.
GOAL_CLEAR_ARGS = frozenset({"clear", "none", "reset", "--clear"})

#: How much of a cleared goal the receipt echoes.
#:
#: ``MAX_GOAL_CHARS`` is a spec-sized bound no single receipt should carry: the
#: echo exists so a mistaken clear is visible and can be retyped by eye, and a
#: goal longer than a terminal row defeats exactly that. 96 is the same bound the
#: ``goal restored`` notice already clips to, so the two lines the app prints
#: about the same value read alike. Characters rather than cells because this
#: module is the shared, non-UI half of the app — the surfaces that paint the
#: string own its cell clipping.
CLEARED_GOAL_ECHO_CHARS = 96


def cleared_goal_receipt(cleared: str) -> str:
    """The ``/goal --clear`` receipt: name what went, or say there was nothing.

    A standing goal is deliberately invisible in the UI — the band does not carry
    it, and the only echo is the one-time ``goal restored`` notice on adopt — and
    there is no undo. So this string is the user's whole chance to see what a
    mistaken clear took away and type it back, which is why it names the goal
    rather than reporting the event (design D4 / UX U3, round 1).

    Flattened to ONE line: a receipt is a single terminal row, and a goal is
    free text a user may have pasted newlines into.
    """
    text = " ".join((cleared or "").split())
    if not text:
        # Nothing was set, so there is nothing to name — and "goal cleared: "
        # with an empty tail reads as a rendering bug rather than an empty goal.
        return "goal cleared"
    if len(text) > CLEARED_GOAL_ECHO_CHARS:
        text = text[:CLEARED_GOAL_ECHO_CHARS].rstrip() + "…"
    return f"goal cleared: {text}"


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

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
#: echo exists so a mistaken clear is visible and can be retyped by eye, and an
#: echo long enough to bury the transcript defeats exactly that. 96 is the same
#: bound the ``goal restored`` notice already clips to, so the two lines the app
#: prints about the same value read alike.
#:
#: CHARACTERS, not cells, because this module is the shared, non-UI half of the
#: app and the surfaces that paint the string own its cell clipping. The two
#: bounds are not the same number and it is worth knowing which one is quoted
#: here: the 14-cell ``goal cleared: `` prefix plus 96 ASCII characters plus the
#: clip's ``…`` is 111 cells, and the same 96 characters in CJK glyphs is 207.
#: So a maximum-length receipt WRAPS, and that is intended: measured as a
#: painted notice, a 96-character goal with no spaces fills 3 rows at both 80 and
#: 100 columns (Rich drops the unbreakable word to its own rows) and its CJK twin
#: fills 4 — the continuation indents under the text column and reads as one
#: notice, which is how a long objective stays legible at all. What the clip
#: guarantees is a bounded length and one LOGICAL line. This comment and the
#: helper below used to claim a single terminal ROW, which the code has never
#: held (round 2: reviewer NIT-6, UX U8).
CLEARED_GOAL_ECHO_CHARS = 96


def cleared_goal_receipt(cleared: str) -> str:
    """The ``/goal --clear`` receipt: name what went, or say there was nothing.

    A standing goal is deliberately invisible in the UI — the band does not carry
    it, and the only echo is the one-time ``goal restored`` notice on adopt — and
    there is no undo. So this string is the user's whole chance to see what a
    mistaken clear took away and type it back, which is why it names the goal
    rather than reporting the event (design D4 / UX U3, round 1).

    Flattened to ONE line — one LOGICAL line, which is all the flattening
    promises: a goal is free text a user may have pasted newlines into, and a
    multi-row payload dump in the transcript is what that must not become. The
    clip below is a CHARACTER bound, so a long receipt does wrap onto more than
    one painted row; :data:`CLEARED_GOAL_ECHO_CHARS` carries the measurements
    and why it is characters rather than cells.
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
    #: THAT FAIL-OPEN IS FOR THE DECISIONS THAT NEED A DIRECTION — parking a
    #: gate, the browser tool's attached probe (:meth:`is_interactive`). The
    #: MODEL-FACING block asks :meth:`interactivity`, whose ``None`` third value
    #: means "nothing measured" and renders no claim at all; the two must not be
    #: collapsed, or an ``exec`` run is told about an interface it never probed.
    #:
    #: A PROBE rather than a stored flag on purpose. Attach state changes
    #: whenever a viewer opens or closes, and a cached copy would need an
    #: event per change — the token accumulation this exists to avoid. The
    #: prompt closure calls this at turn start and the answer costs one line
    #: whatever happened in between.
    interactive_probe: "Callable[[], bool] | None" = None
    #: Live probe answering "can THIS session put a question in front of a
    #: person?" — the ``ask`` hook, published here by ``Session.__init__`` for
    #: exactly the reason the probe above is: the system-prompt provider closure
    #: is built BEFORE the session facade exists, so a shared holder is the only
    #: seam through which a fact the session learns later — the TUI installs the
    #: hook from a worker, in ``_adopt_session`` — reaches the next turn's prompt.
    #:
    #: ``None`` means "this holder's session was not built by the session facade"
    #: (a delegated child, a bare test), which the block builder reads as "not
    #: stated" and resolves from tool membership. Never a copied flag: the hook is
    #: installed AND uninstalled mid-session by ``set_ask_handler``.
    ask_probe: "Callable[[], bool] | None" = None

    def is_interactive(self) -> bool:
        """Whether a surface can answer a question right now (default True)."""
        probe = self.interactive_probe
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 — an unreadable probe must not kill a turn
            return True

    def interactivity(self) -> bool | None:
        """Tier A as a MEASURED fact: attached, detached, or nothing measured.

        The tri-state twin of :meth:`is_interactive`, for the model-facing block
        only, where ``None`` must render nothing: a host that installed no probe
        (a plain CLI, an ``exec`` run, a scheduled run) has no attachment answer at
        all, and a block asserting one is the same unmeasured claim the incident
        was made of — a scheduled run is not told about an interface nobody
        checked for.

        :meth:`is_interactive` keeps its fail-open ``True`` for the decisions that
        need a direction (parking a gate, the browser tool's attached probe): an
        unmeasured answer there must resolve to attached. Here an unreadable probe
        is ``None`` for the same reason, in the other register — a probe that
        raised measured nothing, so the block may say nothing.
        """
        probe = self.interactive_probe
        if probe is None:
            return None
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 — an unreadable probe must not kill a turn
            return None

    def can_ask(self) -> bool | None:
        """Whether THIS session can put a question in front of a person.

        ``None`` means the answer was not stated here, and the block builder falls
        back to tool membership — which is the honest answer for a delegated child,
        whose inventory holds no ``ask`` tool because ``build_ask_tool`` is gated on
        the hook every child is built without.
        """
        probe = self.ask_probe
        if probe is None:
            return None
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 — an unreadable probe states nothing
            return None

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

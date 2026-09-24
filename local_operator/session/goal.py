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

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

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
#: discoverable form the palette and the argument picker now teach, and ``delete``
#: joins it because a user reaching for the destructive act types either word
#: (RULINGS R7) — it is an ALIAS of ``--clear``, so it records nothing, which is
#: what makes it different from ``--done`` rather than a second spelling of it.
#:
#: The flag is matched as the WHOLE argument, never as a prefix: ``/goal --clear``
#: is a flag, while ``/goal --clear the flaky job`` is still free text the user
#: meant as an objective. Eating the tail of a real goal would be silent data
#: loss in the one command whose argument the MODEL is told.
GOAL_CLEAR_ARGS = frozenset({"clear", "none", "reset", "delete", "--clear"})

#: The OTHER flags ``/goal`` honours, each matched as the WHOLE argument exactly
#: as ``GOAL_CLEAR_ARGS`` is. The bare verbs are aliases of the flag (``done`` /
#: ``--done``) rather than a second grammar, because the desktop sends the bare
#: word — the ``clear`` precedent — and a word honoured on one host while stored
#: as a goal body on another is the host-disagreement class ``GOAL_CLEAR_ARGS``
#: exists to remove.
GOAL_DONE_ARGS = frozenset({"done", "--done"})
GOAL_HISTORY_ARGS = frozenset({"history", "--history"})
GOAL_DISMISS_ARGS = frozenset({"dismiss", "--dismiss"})

#: EVERY argument this command reads as a flag rather than as an objective.
#:
#: One set so the three things that must agree cannot drift apart: each host's
#: flag branch, and :func:`local_operator.slash_commands.unknown_flag_refusal`,
#: whose ``known`` map decides whether ``/goal --done`` is honoured or refused as
#: an unknown flag. A new flag that is not added here is REFUSED rather than
#: silently stored as the goal body, which is the failure mode that matters.
GOAL_FLAG_ARGS = GOAL_CLEAR_ARGS | GOAL_DONE_ARGS | GOAL_HISTORY_ARGS | GOAL_DISMISS_ARGS

#: How many settled goals the record keeps. The cap is the SAVED list a user can
#: act on — 20 settled goals is more than a session's worth — and it is what
#: bounds the record at ENTRY, so the process cannot grow one without end.
GOAL_HISTORY_MAX = 20

#: How much of a settled goal's text a history row carries. 400 characters is a
#: summary of an objective, not the objective: ``MAX_GOAL_CHARS`` = 2000 is a
#: spec-sized bound no LIST row should carry, the same argument
#: :data:`CLEARED_GOAL_ECHO_CHARS` makes for the receipt echo. The full text is
#: still recoverable from the transcript, which is what a history row points AT.
GOAL_HISTORY_TEXT_CHARS = 400

#: How much of a cleared goal the receipt echoes: THE GOAL'S OWN CAP.
#:
#: The echo exists so a mistaken clear is visible and can be retyped by eye, and
#: there is no undo on the terminal — so it is the user's WHOLE recovery artifact,
#: and a bound below the goal's own cap made it incomplete exactly where it
#: mattered: a 152-character goal came back as 96 characters and an ellipsis,
#: which cannot be retyped (design D4 / UX U2, measured on the real card).
#:
#: The old 96 was justified as "the same bound the `goal restored` notice clips
#: to, so the two lines read alike" — and that is the argument this revision
#: rejects: the notice is a one-time greeting about a goal that is still there,
#: while the receipt is the only record of one that is GONE. Two surfaces reading
#: alike is worth less than one of them being usable.
#:
#: CHARACTERS, not cells, because this module is the shared, non-UI half of the
#: app and the surfaces that paint the string own its cell clipping. A
#: maximum-length receipt therefore WRAPS — measured as a painted notice, a long
#: goal with no spaces fills several rows (Rich drops the unbreakable word to its
#: own rows) and its CJK twin more, with the continuation indenting under the text
#: column so it still reads as one notice. What the clip guarantees is a bounded
#: length and one LOGICAL line; the record's own cap is what makes the bound
#: bounded.
CLEARED_GOAL_ECHO_CHARS = MAX_GOAL_CHARS


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


def goal_done_receipt(done: str) -> str:
    """The ``/goal --done`` receipt: name what was settled, or say there was nothing.

    The mistaken-mark-done risks are the same false step a mistaken clear has
    (see :func:`cleared_goal_receipt`), so this sibling shares the bound and the
    one-logical-line flattening: the goal itself is KEPT by a mark-done (it stays
    in ``text`` until dismissal), but the user still needs to see WHICH objective
    was struck, because a later dismissal of the chip is what takes it off the
    live surface. Deliberately the same shape, so the two lines read alike.
    """
    text = " ".join((done or "").split())
    if not text:
        # Nothing was active, so nothing was settled. A "goal done: " with an
        # empty tail would read as a rendering bug rather than as a no-op.
        return "no goal to mark done"
    if len(text) > CLEARED_GOAL_ECHO_CHARS:
        text = text[:CLEARED_GOAL_ECHO_CHARS].rstrip() + "…"
    return f"goal done: {text}"


def goal_done_answer(text: str, entry: "GoalHistoryEntry | None") -> str:
    """The ``/goal --done`` answer: three cases, and it must say WHICH happened.

    ``mark_done`` returns ``None`` both when there is no goal and when the goal
    is already settled, and printing a second "goal done: <text>" for the latter
    would report a settle that did not happen — the same class of false report
    the dismissal receipt exists to avoid. So the caller passes the fact it has
    (was there an entry) and the text, and the wording lives here, once, for all
    four hosts.
    """
    if entry is not None:
        return goal_done_receipt(text)
    if text:
        return "goal already done — /goal --dismiss clears it"
    return goal_done_receipt("")


def _now_iso() -> str:
    """A fixed-width ISO-8601 UTC stamp (``2026-09-22T11:04:07Z``).

    ``%Z``-less and always UTC: these stamps are a SORT KEY and a display fact in
    one, and a local-time stamp with an offset would make two runs of the same
    history compare differently on two hosts. The fixed width is what makes the
    lexical order the chronological one.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_id() -> str:
    """A stable identity for one settled goal, minted once.

    ``uuid4`` rather than a counter or a hash of the text: the id is the key a
    delete-by-id names, and the user may settle the SAME text twice — the second
    mark-done is new work, not a duplicate to collapse.
    """
    return uuid4().hex


#: The judge's state vocabulary. A CLOSED set the writer keeps to and a reader
#: must tolerate: a newer writer's member renders as unknown rather than
#: crashing, which is why the wire type is ``dict[str, Any]`` and not a
#: ``Literal``-validated model.
GOAL_JUDGE_STATES = frozenset(
    {
        # A goal is active and no judge tick is in flight: the turn ended in
        # error or was aborted, or nothing has happened since the goal was set.
        # Never auto-continued from — spending tokens on a broken provider is
        # what this member exists to prevent.
        "waiting",
        # A judge call is in flight.
        "judging",
        # Verdict CONTINUE and an internal continuation turn was admitted.
        "continuing",
        # Verdict ACHIEVED and the goal was marked done.
        "done",
        # The consecutive-failure breaker or the continuation cap stopped
        # auto-continuation. The goal stays ACTIVE; only the harness stopped.
        "stalled",
        # Nothing has run yet (no goal, or a goal nothing has judged).
        "idle",
    }
)


def goal_flag_form(arg: str) -> str:
    """Which FLAG a whole argument names, or "" when it is an objective.

    The whole-argument rule (:data:`GOAL_CLEAR_ARGS`) resolved in ONE place,
    because four hosts read it: a word honoured on one host and stored as the
    goal body on another is the host-disagreement class these vocabularies exist
    to remove, and four copies of the branch chain is how that returns. Returns
    the canonical form name (not the token), so a host switches on the act.
    """
    token = (arg or "").strip().lower()
    if token in GOAL_DONE_ARGS:
        return "done"
    if token in GOAL_DISMISS_ARGS:
        return "dismiss"
    if token in GOAL_HISTORY_ARGS:
        return "history"
    if token in GOAL_CLEAR_ARGS:
        return "clear"
    return ""


def goal_report(text: str, status: str) -> str:
    """The bare ``/goal`` report: the objective plus the state it is in.

    Shared by every host so the status cannot be shown on one and omitted on
    another. A DONE goal names the way to drop the chip, because a settled goal
    has no live work to do and the user's next question is how to get rid of it.
    """
    if not text:
        return "no goal set — /goal <text> to set one"
    if status == "done":
        return f"goal: {text} — done; /goal --dismiss clears it"
    return f"goal: {text} — active"


def goal_dismissed_receipt(dismissed: bool) -> str:
    """The ``/goal --dismiss`` receipt, which must say when nothing was there.

    The flag is a no-op unless the chip is up, and a bare "goal dismissed" for
    that case would report a state change that did not happen.
    """
    return "goal dismissed" if dismissed else "nothing to dismiss"


def goal_history_notice(count: int) -> str:
    """The one-line notice beside ``/goal --history``'s rows.

    A COUNT, never the rows: the entries ride the receipt's ``data`` because a
    multi-row payload dump in the transcript is exactly what the receipt
    discipline exists to prevent (see :func:`cleared_goal_receipt`).
    """
    if count <= 0:
        return "no settled goals"
    return f"{count} settled goal{'s' if count != 1 else ''} — newest first"


def goal_history_items(entries: "list[dict[str, Any]]") -> list[list[str]]:
    """``/goal --history``'s rows: ``[objective, facts]`` per settled goal.

    ONE builder for the hosts, so the typed receipt and the pane cannot disagree
    about order or clip. The objective is clipped per row with the receipt echo's
    own bound rather than the full goal bound, for the reason that bound exists:
    a history row is a pointer to the objective, and a list row should not carry
    a spec. ``facts`` is the short metadata column — status, when it settled, and
    the judge's reason when there was one ("" for a supersede, and for a
    user-typed mark-done, which no model spoke).
    """
    rows: list[list[str]] = []
    for entry in entries:
        text = " ".join(str(entry.get("text") or "").split())
        if len(text) > CLEARED_GOAL_ECHO_CHARS:
            text = text[:CLEARED_GOAL_ECHO_CHARS].rstrip() + "…"
        facts = str(entry.get("status") or "")
        settled = str(entry.get("settled_at") or "")
        if settled:
            facts = f"{facts} · {settled}"
        reason = " ".join(str(entry.get("reason") or "").split())
        if reason:
            if len(reason) > CLEARED_GOAL_ECHO_CHARS:
                reason = reason[:CLEARED_GOAL_ECHO_CHARS].rstrip() + "…"
            facts = f"{facts} · {reason}"
        rows.append([text, facts])
    return rows


@dataclass
class GoalHistoryEntry:
    """One SETTLED goal.

    ``done`` is the record the operator asked for (the project-management half of
    A4). ``superseded`` exists because ``/goal B`` over ``/goal A`` would
    otherwise destroy A silently — the same class of loss
    :func:`cleared_goal_receipt` exists to prevent. A DELETE (``/goal --clear``)
    records nothing at all: that is the whole difference between delete and
    mark-done, and the reason the two are separate commands.
    """

    id: str
    text: str
    status: str  # "done" | "superseded"
    created_at: str
    settled_at: str
    #: The judge's reason for a ``done``; "" for ``superseded`` AND for a
    #: user-typed ``/goal --done`` — that is a person's judgement, not a model
    #: verdict, and dressing it in a model's voice would be a lie about who spoke.
    reason: str = ""

    def __post_init__(self) -> None:
        # Clipped AT CONSTRUCTION, not at read: this is the one place the text
        # enters the record, so a later reader cannot forget the bound and every
        # copy of the entry is already the bounded one.
        self.text = (self.text or "").strip()[:GOAL_HISTORY_TEXT_CHARS]

    def to_wire(self) -> dict[str, Any]:
        """The JSON-safe shape, exactly the six keys the contract names."""
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "created_at": self.created_at,
            "settled_at": self.settled_at,
            "reason": self.reason,
        }

    @classmethod
    def from_wire(cls, payload: Any) -> "GoalHistoryEntry | None":
        """Rebuild one entry, or ``None`` when the row is unusable.

        Tolerant by construction: the sidecar is a file a killed process or an
        older build may have written, and one unreadable row must cost that row
        rather than the whole record (the same rule ``read_session_attachment``
        follows at the file level).
        """
        if not isinstance(payload, dict):
            return None
        text = payload.get("text")
        status = payload.get("status")
        if not isinstance(text, str) or status not in {"done", "superseded"}:
            return None

        def _stamp(key: str) -> str:
            value = payload.get(key)
            return value if isinstance(value, str) else ""

        entry_id = payload.get("id")
        return cls(
            id=entry_id if isinstance(entry_id, str) and entry_id else _new_id(),
            text=text,
            status=status,
            created_at=_stamp("created_at"),
            settled_at=_stamp("settled_at"),
            reason=_stamp("reason"),
        )


@dataclass
class GoalJudgeState:
    """The LIVE judge's state, held beside the goal it judges.

    Bounded by construction: four scalars plus one string that is clipped where
    the model's answer enters the state (``_parse_loop_verdict``), so the whole
    record is a fixed-shape set the frame guard can price — the property
    ``goal_loop``'s own state was given for the same reason.
    """

    state: str = "idle"
    #: Consecutive auto-continuations admitted for THIS goal. Capped by
    #: ``MAX_GOAL_CONTINUATIONS``.
    run: int = 0
    verdict: str = ""  # "" | "achieved" | "continue" | "unknown"
    #: Clipped at LOOP_REASON_CHARS where the verdict is parsed.
    reason: str = ""
    #: Consecutive unreadable verdicts. NOT a wire field: it is the breaker's
    #: own counter, and publishing it would invite a surface to render a number
    #: whose only meaning is internal (``stalled`` is the state it produces).
    failures: int = 0

    def to_wire(self) -> dict[str, Any]:
        """The four decided keys — ``failures`` deliberately does not ride."""
        return {
            "state": self.state,
            "run": int(self.run),
            "verdict": self.verdict,
            "reason": self.reason,
        }

    def to_payload(self) -> dict[str, Any]:
        """The SIDECAR shape: the wire keys plus the breaker's counter.

        The sidecar is not the wire. ``failures`` must survive a restart because
        it is what makes the breaker survive one — a resumed session that lost
        it would hand a broken provider a fresh three strikes every boot — while
        publishing it on the frame would invite a surface to render an internal
        count. One deliberate difference between the two documents, stated once.
        """
        return {**self.to_wire(), "failures": int(self.failures)}

    @classmethod
    def from_payload(cls, payload: Any) -> "GoalJudgeState":
        """Rebuild from a ``goal.json`` judge block (tolerant; see from_wire)."""
        return cls.from_wire(payload)

    @classmethod
    def from_wire(cls, payload: Any) -> "GoalJudgeState":
        """Rebuild from one payload, tolerant of anything a newer writer sent.

        An unknown ``state`` member is KEPT rather than dropped: the field is a
        closed vocabulary the reader must tolerate, and rewriting a newer
        writer's member to ``idle`` here would make a resumed session look idle
        on disk the moment anything journalled it.
        """
        if not isinstance(payload, dict):
            return cls()
        state = payload.get("state")
        verdict = payload.get("verdict")
        reason = payload.get("reason")
        run = payload.get("run")
        failures = payload.get("failures")
        return cls(
            state=state if isinstance(state, str) and state else "idle",
            run=int(run) if isinstance(run, int) else 0,
            verdict=verdict if isinstance(verdict, str) else "",
            reason=reason if isinstance(reason, str) else "",
            failures=int(failures) if isinstance(failures, int) else 0,
        )


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

    # --- the judged-goal record (see the dataclasses above) -------------------
    #: "" (none) | "active" | "done". `done` coexists with a non-empty `text`
    #: ONLY between the judge's ACHIEVED verdict and the chip's dismissal.
    status: str = ""
    #: The live judge's state. Held beside the goal it judges because the two
    #: move on different clocks (the goal on the user's command, the judge on
    #: every turn end) and a surface that could not see them together could not
    #: say why a goal it can still read is not being pursued.
    judge: GoalJudgeState = field(default_factory=GoalJudgeState)
    #: SETTLED goals, newest FIRST (see :meth:`_record` for why the order is
    #: storage rather than a view concern).
    history: list[GoalHistoryEntry] = field(default_factory=list)
    #: Minted per ARMING. The judge captures it before its call and drops a
    #: verdict whose token has moved: the identity `text` cannot provide, because
    #: the user may set the SAME text twice and the second arming is new work
    #: whose in-flight verdict must not be the first one's.
    token: str = ""
    #: When the CURRENT goal was armed, so a history entry can record when the
    #: work it settles began. Empty for a goal restored from a pre-lifecycle
    #: build, which never recorded one.
    created_at: str = ""

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

    # --- the judged-goal record --------------------------------------------

    def arm(self, text: str) -> str:
        """``/goal <text>``: store the objective, mark it active, arm the judge.

        The ORDER is why this lives here rather than at the four call sites: an
        outgoing goal is recorded ``superseded`` BEFORE the new text is stored,
        and that ordering is what makes ``/goal B`` non-destructive to ``/goal
        A``. Two cases deliberately settle nothing: a goal already ``done``
        (it is in the history — the chip window is a display state, not a second
        settle), and a holder with no goal at all.

        The judge starts at ``waiting``, not ``idle``: a goal IS active, so the
        honest state is "no tick in flight yet" rather than "no goal".
        """
        if self.text and self.status == "done":
            self.dismiss()
        elif self.text:
            self.record_superseded()
        stored = self.set(text)
        if not stored:
            self.status = ""
            self.token = ""
            self.created_at = ""
            self.judge = GoalJudgeState()
            return stored
        self.status = "active"
        self.created_at = _now_iso()
        self.token = _new_id()
        self.judge = GoalJudgeState(state="waiting")
        return stored

    def mark_done(self, reason: str = "") -> GoalHistoryEntry | None:
        """Settle the active goal as DONE: strike it, and record it.

        The text STAYS in ``text`` until :meth:`dismiss` (that window is what
        lets every surface show WHAT was done), so ``text`` and ``status``
        together mean "this is the objective that was completed" until the chip
        is dropped. Returns the entry, or ``None`` when there was nothing to
        settle or it was already settled — a caller's receipt has to tell those
        apart from a real settle, and a second call must not duplicate a row.

        ``verdict`` is untouched: it is the parser's record of what a model
        SAID, and a user-typed ``/goal --done`` is a person's judgement with no
        model verdict behind it.
        """
        if not self.text or self.status == "done":
            return None
        entry = GoalHistoryEntry(
            id=_new_id(),
            text=self.text,
            status="done",
            created_at=self.created_at or _now_iso(),
            settled_at=_now_iso(),
            reason=reason,
        )
        self._record(entry)
        self.status = "done"
        # The judge state vocabulary's own member for "ACHIEVED and marked
        # done". Leaving the tick state behind would tell a surface a judge was
        # still coming on a goal that has been settled.
        self.judge.state = "done"
        if not reason:
            # A PERSON'S settle carries no model sentence, and the previous
            # tick's REASON is a sentence about work nothing is tracking any
            # more. Kept, it printed a live-looking quote about an objective the
            # user had just closed by hand — `judge: achieved · 2/12 · — still
            # drafting`, on a card whose own history entry correctly recorded no
            # reason at all (QA round 1, Q4). A VERDICT's settle passes its own
            # sentence instead, and that one is the record's reason, published
            # by the judge beside it.
            self.judge.reason = ""
        return entry

    def record_superseded(self, reason: str = "") -> GoalHistoryEntry | None:
        """Settle the CURRENT goal as superseded by a newly armed one.

        The record exists because ``/goal B`` over ``/goal A`` would otherwise
        destroy A silently — the same class of loss ``cleared_goal_receipt``
        prevents for an explicit clear.
        """
        if not self.text or self.status == "done":
            return None
        entry = GoalHistoryEntry(
            id=_new_id(),
            text=self.text,
            status="superseded",
            created_at=self.created_at or _now_iso(),
            settled_at=_now_iso(),
            reason=reason,
        )
        return self._record(entry)

    def delete(self) -> str:
        """``/goal --clear``: blank the goal and record NOTHING.

        Returns the text that went, so the receipt can name it. The missing
        history entry IS the delete-versus-mark-done distinction: an explicit
        delete is the user saying this objective is not worth keeping, and
        recording it would leave the two commands differing only in wording.

        The HISTORY survives — it is the settled record, not the live slot, and
        ``/goal --clear`` is not ``/goal --history --clear``. The judge is reset
        because it belongs to the goal that just left.
        """
        went = self.text
        self.text = ""
        self.status = ""
        self.token = ""
        self.created_at = ""
        self.judge = GoalJudgeState()
        return went

    def dismiss(self) -> bool:
        """Drop the done chip: forget the text and the status.

        Only meaningful while ``status == "done"``; returns ``False`` otherwise
        so a caller can say "nothing to dismiss" rather than print a no-op that
        reads like a success.
        """
        if self.status != "done":
            return False
        self.text = ""
        self.status = ""
        self.token = ""
        self.created_at = ""
        self.judge = GoalJudgeState()
        return True

    def ensure_token(self) -> bool:
        """Give a goal that has NO token one, so the judge can run. ``True`` if minted.

        The other half of RULINGS R9, and since the plain ``set_goal`` arms the
        goal it writes (agent review round 2, MAJOR-5), the ONE caller left is
        the pre-lifecycle restore fold: it reads a ``goal.json`` written before
        this record existed, and the fold reads a goal with no status as
        ``active`` because it is standing work the user expects pursued. But
        ``GoalJudge._enabled`` also requires a TOKEN (it is the judge's staleness
        guard), and ``arm`` was the only minter, so that path produced a goal
        every surface called ``active`` and nothing ever ran on: the silent half
        of "the goal sitting inert", with no error to see (agent review round 1,
        MAJOR-3).

        Minting here is exactly as safe as minting in :meth:`arm`. The token
        exists so a verdict captured before a goal was REPLACED is dropped; a
        goal that never had one has no such verdict, so there is nothing this
        can invalidate. A ``done`` goal is deliberately left alone — it is
        retained so the surfaces can show what was achieved, and minting would
        arm the judge against work that is already finished.
        """
        if not self.text or self.token or self.status == "done":
            return False
        self.token = _new_id()
        return True

    def reset_judge(self, *, state: str) -> None:
        """Move the judge to ``state`` without touching the goal or its record.

        Keyword-only on purpose: the state is the whole point of the call, and a
        positional would let ``reset_judge("waiting")`` read as a text argument
        at a call site where the goal's own text is a parameter nearby.
        """
        self.judge.state = state

    def history_view(self, limit: int | None = None) -> list[dict[str, Any]]:
        """``/goal --history``'s payload: newest-first wire dicts.

        ``limit`` is what the caller's renderer can show. The default is the
        whole record, which is already capped at :data:`GOAL_HISTORY_MAX` — so
        the receipt is bounded by the same rule the frame is, rather than by a
        second bound that could drift from it.
        """
        rows = self.history if limit is None else self.history[: max(0, limit)]
        return [entry.to_wire() for entry in rows]

    def _record(self, entry: GoalHistoryEntry) -> GoalHistoryEntry:
        """Insert newest-FIRST and evict past the cap.

        The order is storage, not a view concern: the eviction is then a tail
        slice rather than a search, and every reader (the frame, the receipt,
        the pane) wants the newest entry at index 0.
        """
        self.history.insert(0, entry)
        del self.history[GOAL_HISTORY_MAX:]
        return entry

    def to_payload(self) -> dict[str, Any]:
        """The ``goal.json`` document: the RECORD half of this holder.

        Deliberately not the whole holder — ``team_brief``/``agent_brief``/
        ``agent_name``/``interactive_probe`` are the ATTACHMENT's, journalled by
        ``attachment.json``. Writing them here too would give two files that can
        disagree about one value, and the sidecar's whole point (see
        ``resume.read_goal_record``) is to be invisible to the writers that
        rebuild that other document.
        """
        return {
            "goal": self.text,
            "status": self.status,
            "created_at": self.created_at,
            "token": self.token,
            "judge": self.judge.to_payload(),
            "history": [entry.to_wire() for entry in self.history],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "GoalState":
        """Rebuild a holder from a ``goal.json`` document.

        ``None``, a non-mapping or an empty document yields an EMPTY state (the
        pre-lifecycle shape) rather than raising: an unreadable sidecar must
        cost the record, never the resume. Anything unusable inside a readable
        document costs only itself, for the same reason one bad row must not
        take a history with it.
        """
        state = cls()
        if not isinstance(payload, dict):
            return state
        goal = payload.get("goal")
        if isinstance(goal, str) and goal.strip():
            # Straight onto the holder: this is a read of disk state, not a user
            # action, so it must not re-journal (nor re-mint a token).
            state.set(goal)
            status = payload.get("status")
            # Same rule as the fold's migration default: a goal with no readable
            # status is ACTIVE, because reading it as absent would silently
            # un-set an objective the user still expects to be pursued.
            state.status = status if status in {"active", "done"} else "active"
            created_at = payload.get("created_at")
            state.created_at = created_at if isinstance(created_at, str) else ""
            token = payload.get("token")
            state.token = token if isinstance(token, str) else ""
            state.judge = GoalJudgeState.from_payload(payload.get("judge"))
        rows = payload.get("history")
        if isinstance(rows, list):
            rebuilt = [GoalHistoryEntry.from_wire(row) for row in rows]
            state.history = [entry for entry in rebuilt if entry is not None][:GOAL_HISTORY_MAX]
        return state

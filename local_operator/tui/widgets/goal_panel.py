"""The ``/goal`` overlay: the standing goal, its judge and the settled record.

Why an overlay rather than more lines in the transcript. The goal is STANDING
state, and the two acts the user has over it (mark done, delete) are different in
kind from everything else in the conversation: a transcript row for them scrolls
away exactly as the user decides to act on it, and a re-run of ``/goal`` would
add a second row describing the same state. The panel is read while it is acted
on, so it holds focus and is dismissed by a key — the same shape
``widgets/usage_panel.py`` established, with the same host/placement rules
(:mod:`local_operator.tui.widgets.overlay`).

Whatever it says must be TRUE of the record: a goal marked done is struck
THROUGH (``Style(strike=True)``, not a colour or a tag alone — the operator asked
for the struck state, and a tag is what a struck row keeps readable), a stalled
judge says which bound stopped it, and the settled list is the record's own
history, newest first.

Everything above the class is a pure function of the record, so the body is
testable at any width without a screen (``tests/unit/tui/test_goal_panel.py``).
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.message import Message
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import overlay
from local_operator.tui.widgets.tool_card import truncate_cells

#: How many settled goals the panel lists before it says how many it did not.
#: The record itself holds `GOAL_HISTORY_MAX`; a card that printed all of them
#: would be taller than most terminals, and the newest few are what a user acts
#: on — so this bound is the CARD's, and it names what it dropped.
MAX_HISTORY_ROWS = 6

#: The card's width bounds. ``MIN`` is what the body's own columns need to read
#: (a 40-cell goal line plus the gutter); ``MAX`` keeps it from becoming a
#: full-width band on a wide terminal, where the eye has to travel back to the
#: left edge to read a two-word goal.
MIN_PANEL_WIDTH = 46
MAX_PANEL_WIDTH = 100

#: Cells the card's own padding spends, two per side (the stylesheet's
#: `padding: 1 2`). The BODY is built to the content box, not the outer width:
#: a line one cell too long WRAPS, and a wrapped line pushes the card's last row
#: out of the box it pinned — which is how the key hint went missing on the first
#: captured frame (a `─` rule built to `outer - 2` against a content box of
#: `outer - 4`).
PANEL_PADDING_CELLS = 4

#: Rows the card's own padding spends, one per side (the stylesheet's
#: `padding: 1 2`). Named here because the widget SIZES ITSELF: Textual is
#: border-box, so a pinned height that does not add these back hands the gutter
#: the last content rows and clips the hint row off the bottom of the card — the
#: mistake the usage card documents and this one made first (caught by looking at
#: a rendered frame, which is why it is a named constant and a test now).
PANEL_PADDING_ROWS = 2

#: Rows above the dock at which the card can still afford its vertical gutter.
#: Below it the gutter is dropped (the ``-squeezed`` class in the stylesheet),
#: because the alternative is a card that covers the prompt — the sibling
#: ``UsagePanel`` made this exact trade and this card had no answer for it at all
#: (design D3 / UX U5).

#: Cells the ``  — done`` tag on the standing row spends, needed by the clip
#: budget below: the goal row was built to the FULL content width and then had
#: the tag appended, so at 80 columns a 54-cell goal wrapped and pushed the key
#: hint off the card entirely (design D2).
_GOAL_TAG_CELLS = 8

#: The stamp tail a settled row prints (`` · 2026-09-23 11:04``). Named because
#: the same budget has to subtract it, per row, alongside the mark and the
#: status word.
_HISTORY_STAMP_CELLS = 19

#: What the card says when the WIRE dropped settled goals (the frame ran out of
#: line and the record yielded its history, ``frontend_state``'s
#: ``goal_history_truncated``). A list that under-reports must say so, and a
#: clipped record is the case where it cannot even count how many are missing.
_HISTORY_CLIPPED_NOTE = "… older settled goals were dropped from this record"

#: The theme token the judge line wears when the harness has STOPPED working on
#: the goal. The other states are passive and print dim; this one is the only row
#: asking the user for something, and it was indistinguishable from `waiting`
#: (design D6). Resolved per call rather than at import so a theme switch is
#: reflected on the next repaint.
_STALLED_TOKEN = "warning"

#: Rows the card spends on chrome, i.e. everything that is not a goal or a
#: history row: title, rule, judge line, blank, history heading, hint.
_CHROME_ROWS = 6

#: Rows above the dock below which the card spends its gutter rather than cover
#: the docked prompt. Defined AFTER ``_CHROME_ROWS`` because it is a sum of the
#: two, and a forward reference here would be a NameError at import.
_SQUEEZE_ROWS = PANEL_PADDING_ROWS + _CHROME_ROWS + 1
_CHROME_ROWS = 6

#: The ink a settled goal is struck with. A real SGR 9, not a colour: a
#: strikethrough survives on terminals with no colour and reads as "finished"
#: on every one that has it — which is what the operator's ask named.
_DONE_STYLE = Style(strike=True, dim=True)

#: The judge states, in the words the panel prints. Kept as a table rather than
#: an f-string so an unknown member of the closed vocabulary (a newer writer's)
#: prints as itself instead of being dropped or crashing — and kept IDENTITY
#: rather than translated, because the state word is the WIRE's own everywhere
#: else: the notice a user read a moment earlier says `goal stalled: …`, and a
#: card calling the same state `stopped` put two names on one state in the same
#: minute (design D5 / UX U8). `idle` is the one member with a meaning worth
#: stating, and it means the judge has NOT RUN — never "there is no goal".
_JUDGE_WORDS = {
    "": "idle",
    "idle": "idle (the judge has not run)",
    "waiting": "waiting",
    "judging": "judging",
    "continuing": "continuing",
    "done": "done",
    "stalled": "stalled",
}


def _stalled_style() -> Style:
    """The ink for the one judge state that asks something of the user.

    Every other state is passive and prints `dim`; a stall is the row that owes
    the user an action, and it wore the identical dim ink as `waiting` and
    `continuing` (design D6). Resolved through the theme's semantic token per
    call so a theme switch lands on the next repaint, the way the notice that
    announces the same stall already paints.
    """
    return Style(color=theme_mod.semantic_color(_STALLED_TOKEN))


def judge_line(judge: dict[str, Any] | None, *, cap: int) -> Text:
    """The one line that says what the judge is doing, or why it stopped.

    ``run``/``cap`` are printed as a LABELLED fraction (``run 2/12``): the cap
    is the answer to "why did it stop" whenever the cap was the bound, and a
    bare `2/12` left the reader asking "of what" (design D7 / UX U8).
    ``reason`` is printed verbatim — it is the judge's own sentence for a stall
    (see ``session/goal_judge.py``) or the model's for a mark-done — joined with
    the em dash ALONE, because ` · ` before a sentence that already opens with
    `— ` painted `· —` (same finding).
    """
    if not judge:
        return Text("judge: —", style="dim")
    state = str(judge.get("state") or "idle")
    words = _JUDGE_WORDS.get(state, state)
    parts = [f"judge: {words}"]
    run = judge.get("run")
    if isinstance(run, int) and run:
        parts.append(f"run {run}/{cap}")
    line = " · ".join(parts)
    reason = str(judge.get("reason") or "").strip()
    if reason:
        line = f"{line} — {reason}"
    return Text(line, style=_stalled_style() if state == "stalled" else "dim")


def _hint_row(*, status: str, goal: str, actions: bool, armed: bool) -> str:
    """What the card advertises, given the state it is painting.

    §5.4's own rule for this panel — *a key that silently does nothing is a
    defect, so it says so* — has a corollary the card was missing: it must not
    advertise an act the state cannot honour. The empty card offered
    `d done · c delete` and answered them with "no goal to mark done" and a
    `goal cleared` receipt for deleting nothing; the done card offered `d done`
    for an act that is a no-op in that state, and named the act that IS
    available only in a notice (design D8 / UX U3).

    The word for the erase is `clear`, the one the palette's `--clear`, the
    `goal cleared:` receipt and the typed flag all use: the card had a third
    name (`delete`) for one act, and the done state's `c` performs what the
    palette calls `--dismiss` — so "Clear" named both a benign and an
    irreversible act two rows apart (UX U4).
    """
    if not actions or not goal:
        # A follower may only READ (the two acts write the record, and a second
        # writer of one record is the class the ownership rule exists to
        # prevent); and with no goal set there is nothing to act on at all.
        return "q close"
    if armed:
        # The rehearsal the two-press rule asks for (DESIGN-UX §2.1/§5.4): the
        # first press paints what the second one does, and how to back out.
        return "c again to clear · esc cancels"
    if status == "done":
        return "c dismiss · q close"
    return "d done · c clear · q close"


def build_goal_body(
    *,
    goal: str,
    status: str,
    judge: dict[str, Any] | None,
    history: list[dict[str, Any]],
    width: int,
    cap: int,
    actions: bool = True,
    max_rows: int = MAX_HISTORY_ROWS,
    history_truncated: bool = False,
    armed: bool = False,
) -> Text:
    """The card's whole body, as one Rich ``Text``.

    ``actions`` is what a FOLLOWER terminal turns off: it can read this session's
    record, but the two keys write it, and a second writer of one record is the
    class the ownership rule exists to prevent — so the hint row drops them
    rather than advertising keys that would be refused.

    ``width`` is the CONTENT box the card will render into — the caller has
    already taken the padding off — because every line here has to fit without
    wrapping: see :data:`PANEL_PADDING_CELLS`.

    ``history`` is the WHOLE settled list and ``max_rows`` is the bound, and that
    division of labour is the fix for a MAJOR defect rather than a tidier
    signature: a caller that pre-sliced left the dropped-rows branch below
    unreachable, so the card painted six of a record's eleven settled goals with
    nothing to say it had, and at a budget of zero it printed `settled none yet`
    against those eleven — a pane that does not merely under-report but denies
    (design D1 / UX U1, RULINGS R1). ``history_truncated`` is the WIRE's own
    flag (the frame ran out of line and the record yielded its history), and it
    is a separate statement because in that case the card cannot even count what
    is missing.
    """
    inner = max(20, width)
    out = Text()
    done = status == "done"
    title = "Goal"
    if not goal:
        title += " — none set"
    elif done:
        title += " — done"
    else:
        title += " — active"
    out.append(title, style="bold")
    out.append("\n")
    out.append("─" * inner, style="dim")
    out.append("\n")
    if not goal:
        out.append("no goal set — /goal <text> to set one", style="dim")
    else:
        # The transcript's own user-block spine, so the standing objective reads
        # as the human's words rather than as app chrome.
        out.append("▌ ", style="dim" if done else "bold")
        # STRUCK, and the tag is not: a struck row must stay readable, which is
        # the rule the to-do rows already follow. The TAG IS SUBTRACTED from the
        # clip budget because it is appended AFTER the clip: a 54-cell goal at 80
        # columns used `inner - 2` and then had `  — done` added, so the row
        # measured 74 cells in a 66-cell box, wrapped, and pushed the key hint
        # off the card entirely (design D2).
        text = truncate_cells(
            " ".join(goal.split()), max(10, inner - 2 - (_GOAL_TAG_CELLS if done else 0))
        )
        out.append(text, style=_DONE_STYLE if done else "")
        if done:
            out.append("  — done", style="dim")
    out.append("\n")
    out.append_text(judge_line(judge, cap=cap))
    out.append("\n\n")
    out.append("settled", style="bold")
    # `none yet` is a claim about the RECORD, so it is withheld when the record
    # itself was clipped: the card cannot say there are none when it has been
    # told some were dropped.
    if not history and not history_truncated:
        out.append("  none yet", style="dim")
    shown = history[: max(0, max_rows)]
    for entry in shown:
        out.append("\n")
        entry_status = str(entry.get("status") or "")
        # `done` was finished; `superseded` was replaced. Only the first is
        # struck — a line through a goal that was abandoned claims it was met.
        out.append("✓ " if entry_status == "done" else "· ")
        settled_at = str(entry.get("settled_at") or "")
        # Measured from the parts THIS row prints rather than subtracted as a
        # constant: the mark (2), the two-space gap plus the status word, and the
        # stamp tail when there is one. `  superseded` is six cells longer than
        # `  done`, and a row with no stamp carries six fewer still — either one
        # wraps the row and takes the hint with it, which is the same defect D2
        # found on the goal row.
        tail = 2 + 2 + len(entry_status) + (_HISTORY_STAMP_CELLS if settled_at else 0)
        text = truncate_cells(" ".join(str(entry.get("text") or "").split()), max(10, inner - tail))
        out.append(text, style=_DONE_STYLE if entry_status == "done" else "")
        out.append(f"  {entry_status}", style="dim")
        if settled_at:
            out.append(f" · {settled_at[:16].replace('T', ' ')}", style="dim")
    hidden = len(history) - len(shown)
    if hidden > 0:
        out.append("\n")
        out.append(f"… {hidden} more settled", style="dim")
    if history_truncated:
        out.append("\n")
        out.append(_HISTORY_CLIPPED_NOTE, style="dim")
    out.append("\n")
    out.append(_hint_row(status=status, goal=goal, actions=actions, armed=armed), style="dim")
    return out


def clamp_history_rows(
    rows: int, available_rows: int, *, gutter_rows: int = PANEL_PADDING_ROWS
) -> int:
    """How many settled rows fit the ground the card may use.

    The card must never cover the docked prompt (:func:`overlay.rows_above_dock`
    is the ground), and on a short terminal that means showing fewer settled
    goals and SAYING so rather than pushing the title off the top.

    ``gutter_rows`` is the card's own vertical padding, which the ``-squeezed``
    state drops: a squeeze that spent CSS but left the ROW BUDGET still charging
    for the gutter would show one row fewer than the terminal could hold, which
    is the kind of disagreement between the sheet and the arithmetic the sibling
    card's ``_fit()`` exists to prevent.

    A ROW IS RESERVED for the dropped-rows notice whenever rows are dropped, and
    it comes out of the same budget as the rows themselves: a card that paints a
    single settled goal and cannot say how many it left behind is the silent
    under-report this bound exists to prevent, and on the shortest terminals the
    notice is the ONLY part of the list that fits — which is exactly where the
    card used to paint `settled none yet` against a record full of them (design
    D1 / UX U1).
    """
    wanted = min(rows, MAX_HISTORY_ROWS)
    budget = max(0, available_rows - _CHROME_ROWS - gutter_rows)
    if wanted <= budget:
        return wanted
    return max(0, min(wanted, budget - 1))


class GoalDismissed(Message):
    """The user closed the overlay."""


class GoalMarkDone(Message):
    """The user asked for the standing goal to be struck and recorded."""


class GoalDelete(Message):
    """The user asked for the standing goal to be DELETED (no history entry)."""


class GoalPanel(Static):
    """The overlay itself. It holds no record state — the session does.

    State is only what the widget can own: the snapshot it was last handed and
    whether it is displayed. Every mutation is reported as a message and applied
    by the app against the session, which is the same split the usage panel makes
    and the reason this widget cannot drift from the durable record: it repaints
    from what the app hands it, never from its own copy.
    """

    can_focus = True

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        # `d` and `c` are the SHORT forms of the two flags the command already
        # teaches (`/goal --done`, `/goal --clear`), so the keys need no new
        # vocabulary — and both are free HERE because this panel HOLDS FOCUS
        # while it is open, so its bindings are consulted before the app's (the
        # composer's TextArea cursor keys included).
        #
        # "HERE" IS THE LOAD-BEARING WORD: the app owns a bare `c`
        # (`action_subagent_child`) and bare `p`/`r`/`[`/`]` for the subagent
        # dock, so the claim this comment used to make — that *both are free* —
        # is true only while focus stays put. After Tab the card stays up with
        # the same hint row and the keys no longer reach it (design D9, whose
        # behaviour is the overlay family's and is left as it is; what is fixed
        # is the claim).
        #
        # Deliberately NOT app-level chords: the app's global vocabulary is
        # remappable and PERSISTED (`local_operator/keymap.py`), so a new global
        # action would be a keymap id and therefore a config migration for one
        # panel's two gestures.
        ("d", "mark_done", "Done"),
        ("c", "delete", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__(id="goal-panel")
        self._goal = ""
        self._status = ""
        self._judge: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        #: Whether this terminal may act on the record at all (see ``actions``
        #: on :func:`build_goal_body`).
        self._actions = True
        self._cap = 0
        #: Whether the WIRE dropped settled goals from the frame this record came
        #: off (``frontend_state``'s ``goal_history_truncated``). The card owes the
        #: reader that statement too: the flag had no consumer anywhere under
        #: ``local_operator/tui/``, so a history clipped at the frame budget was
        #: silent here as well (design D1 / UX U1).
        self._history_truncated = False
        #: Whether `c` has armed the ERASE and is waiting for its second press.
        #: `d` never arms: it records an act, and the design's rule is "a recorded
        #: act is one press; an erased act is two" (DESIGN-UX §2.1).
        self._armed = False
        self.display = False

    # -- state ---------------------------------------------------------------
    def show(
        self,
        *,
        goal: str,
        status: str,
        judge: dict[str, Any] | None,
        history: list[dict[str, Any]],
        cap: int,
        actions: bool = True,
        history_truncated: bool = False,
    ) -> None:
        """Display the record as it stands, and repaint.

        The WHOLE ``history`` is kept: the bound belongs to the formatter, which
        can say how many rows it dropped because it is the one doing the dropping
        (see :func:`build_goal_body`).
        """
        self._goal = goal or ""
        self._status = status or ""
        self._judge = judge
        self._history = list(history)
        self._cap = cap
        self._actions = actions
        self._history_truncated = bool(history_truncated)
        self.display = True
        self._repaint()

    def close(self) -> None:
        """Hide the card. The record is untouched — this is a surface, not an act."""
        self.display = False

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    @property
    def record(self) -> tuple[str, str, dict[str, Any] | None, list[dict[str, Any]]]:
        """The snapshot on screen: ``(goal, status, judge, history)``."""
        return self._goal, self._status, self._judge, list(self._history)

    # -- geometry ------------------------------------------------------------
    def panel_width(self) -> int:
        """The card's width, never wider than the screen it is on.

        ``MIN_PANEL_WIDTH`` is what the body's own columns need to READ, and it
        used to be applied against the screen unconditionally: at 34 columns the
        card was 46 cells on a 34-cell terminal, so the rule, the fill and any
        long goal line ran off the right edge (UX U6). Below the measure the
        frame wins — a card nobody can read the end of is worse than a narrow
        one — and the two narrow-degradation guards this card already has (the
        content-box rule and the history clamp) are joined by the one that
        matters most.
        """
        screen_width, _ = overlay.screen_size(self)
        usable = screen_width - 8
        if usable < MIN_PANEL_WIDTH:
            return max(20, screen_width - 2)
        return min(MAX_PANEL_WIDTH, usable)

    def _fit(self) -> tuple[int, int, int]:
        """``(rows above the dock, gutter rows, settled rows that fit)``.

        One measurement, because the three are one sum: the pinned height is
        exactly ``body rows + gutter``, and it has to come out no greater than the
        ground. The sibling ``UsagePanel`` splits this the same way for the same
        reason and calls it ``_fit()``; naming it the same is what stops a reader
        from thinking the two cards answer this question differently.
        """
        rows = overlay.rows_above_dock(self)
        # Below the squeeze threshold the decoration is what goes: the gutter is
        # the only part of the card that carries no information, and spending it
        # is what keeps the card off the docked prompt (design D3 / UX U5).
        gutter = PANEL_PADDING_ROWS if rows >= _SQUEEZE_ROWS else 0
        return rows, gutter, clamp_history_rows(len(self._history), rows, gutter_rows=gutter)

    @staticmethod
    def _rendered_rows(body: Text, width: int) -> int:
        """Rows the body will actually PAINT, wrapping included.

        Pinned height is border-box and the padding is added back, so this count
        is what the card must add it to — and it has to be measured from the
        rendered rows rather than from the `\n` count, because a line one cell too
        long wraps and the card is then a row short of its own content. That is
        the second shape design D2 found: the wrapped row pushed the key hint off
        the card while the pin still claimed it fit.
        """
        cells = max(1, width)
        return sum(max(1, -(-cell_len(line) // cells)) for line in body.plain.split("\n"))

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        width = self.panel_width()
        available, gutter, shown = self._fit()
        self.set_class(gutter == 0, "-squeezed")
        content_width = max(20, width - PANEL_PADDING_CELLS)
        body = build_goal_body(
            goal=self._goal,
            status=self._status,
            judge=self._judge,
            history=self._history,
            width=content_width,
            cap=self._cap,
            actions=self._actions,
            max_rows=shown,
            history_truncated=self._history_truncated,
            armed=self._armed,
        )
        self.styles.width = width
        # Pinned rather than `auto` for the reason the usage card pins its own:
        # `auto` measures against a guessed width before layout and settles a row
        # tall, and this card is repainted after every action.
        #
        # The PADDING ROWS ARE ADDED BACK, and that is load-bearing: Textual
        # sizes border-box, so pinning the content count alone gives the gutter
        # the last two content rows and the hint row falls off the bottom of the
        # card. Measured on the first captured frame: the card painted its title,
        # rule, goal and judge rows and then ran out, with `settled` and the key
        # hint invisible but still occupying height.
        rows = self._rendered_rows(body, content_width)
        # …and the sum is CLAMPED to the ground: the card may never cover a docked
        # surface, and on a terminal too short for its own chrome the honest
        # failure is to clip its own content rather than paint over the composer
        # (design D3 / UX U5, measured at 100×14 / 80×12 / 60×12).
        outer_height = min(available, rows + gutter)
        self.styles.height = outer_height
        overlay.recentre(self, width, outer_height)
        self.update(body)

    def on_mount(self) -> None:
        if self.display:
            self._repaint()

    # -- actions -------------------------------------------------------------
    def action_dismiss(self) -> None:
        # A close cancels a pending rehearsal: the arm is modal state about the
        # card that is going away, and leaving it set would fire the NEXT `c`
        # unarmed-open on a card the user has since reopened.
        self._armed = False
        self.close()
        self.post_message(GoalDismissed())

    def action_mark_done(self) -> None:
        if not self._actions:
            return
        # Whatever the state, the arm does not survive another keypress: the
        # rehearsal is "the next `c`", not "the next `c` whenever".
        was_armed = self._armed
        self._armed = False
        if was_armed:
            self._repaint()
        # A `done` card has nothing to settle, and the app's own answer says so;
        # the hint row no longer offers the key in that state (design D8).
        self.post_message(GoalMarkDone())

    def action_delete(self) -> None:
        """The erase, in TWO presses — or the done chip's dismissal in one.

        DESIGN-UX §2.1/§5.4: *a recorded act is one press; an erased act is two,
        because delete removes the goal's words with no history entry and no
        undo*. The card fired an immediate `GoalDelete` on a single `c` — one key
        directly under `d` on a QWERTY board, on a surface that holds focus, with
        no undo to lean on (design D4 / UX U2), where the desktop keeps a
        deliberate danger path of its own. The first press now ARMS and paints
        the rehearsal; the second fires.

        The DONE state is exempt on purpose: there the same key drops the chip
        the palette already calls `--dismiss`, which is the benign act and not an
        erase — the record's text is kept until it is dismissed, and the history
        entry is what says the goal was met.
        """
        if not self._actions:
            return
        if self._status == "done":
            self._armed = False
            self.post_message(GoalDelete())
            return
        if not self._armed:
            self._armed = True
            self._repaint()
            return
        self._armed = False
        self._repaint()
        self.post_message(GoalDelete())

    def on_blur(self) -> None:
        """A focus change drops the arm.

        An armed card that has lost focus would fire on a later `c` the user
        pressed for something else entirely — the card's keys only reach it while
        it holds focus (design D9), so the rehearsal must not outlive that.
        """
        if self._armed:
            self._armed = False
            self._repaint()

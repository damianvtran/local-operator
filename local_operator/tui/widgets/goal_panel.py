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

from rich.style import Style
from rich.text import Text
from textual.message import Message
from textual.widgets import Static

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

#: Rows the card spends on chrome, i.e. everything that is not a goal or a
#: history row: title, rule, judge line, blank, history heading, hint.
_CHROME_ROWS = 6

#: The ink a settled goal is struck with. A real SGR 9, not a colour: a
#: strikethrough survives on terminals with no colour and reads as "finished"
#: on every one that has it — which is what the operator's ask named.
_DONE_STYLE = Style(strike=True, dim=True)

#: The judge states, in the words the panel prints. Kept as a table rather than
#: an f-string so an unknown member of the closed vocabulary (a newer writer's)
#: prints as itself instead of being dropped or crashing.
_JUDGE_WORDS = {
    "idle": "no goal",
    "waiting": "waiting",
    "judging": "judging",
    "continuing": "continuing",
    "done": "achieved",
    "stalled": "stopped",
}


def judge_line(judge: dict[str, Any] | None, *, cap: int) -> Text:
    """The one line that says what the judge is doing, or why it stopped.

    ``run``/``cap`` are printed as a fraction because the cap is the answer to
    "why did it stop" whenever it was the cap that stopped it, and a bare state
    word would leave the user guessing which of the two bounds fired. ``reason``
    is printed verbatim — it is the judge's own sentence for a stall (see
    ``session/goal_judge.py``) or the model's for a mark-done.
    """
    if not judge:
        return Text("judge: —", style="dim")
    state = str(judge.get("state") or "idle")
    words = _JUDGE_WORDS.get(state, state)
    parts = [f"judge: {words}"]
    run = judge.get("run")
    if isinstance(run, int) and run:
        parts.append(f"{run}/{cap}")
    reason = str(judge.get("reason") or "").strip()
    if reason:
        parts.append(f"— {reason}")
    return Text(" · ".join(parts), style="dim")


def build_goal_body(
    *,
    goal: str,
    status: str,
    judge: dict[str, Any] | None,
    history: list[dict[str, Any]],
    width: int,
    cap: int,
    actions: bool = True,
) -> Text:
    """The card's whole body, as one Rich ``Text``.

    ``actions`` is what a FOLLOWER terminal turns off: it can read this session's
    record, but the two keys write it, and a second writer of one record is the
    class the ownership rule exists to prevent — so the hint row drops them
    rather than advertising keys that would be refused.
    """
    inner = max(20, width - 2)
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
        # the rule the to-do rows already follow.
        text = truncate_cells(" ".join(goal.split()), inner - 2)
        out.append(text, style=_DONE_STYLE if done else "")
        if done:
            out.append("  — done", style="dim")
    out.append("\n")
    out.append_text(judge_line(judge, cap=cap))
    out.append("\n\n")
    out.append("settled", style="bold")
    if not history:
        out.append("  none yet", style="dim")
    shown = history[:MAX_HISTORY_ROWS]
    for entry in shown:
        out.append("\n")
        entry_status = str(entry.get("status") or "")
        # `done` was finished; `superseded` was replaced. Only the first is
        # struck — a line through a goal that was abandoned claims it was met.
        out.append("✓ " if entry_status == "done" else "· ")
        text = truncate_cells(
            " ".join(str(entry.get("text") or "").split()), max(10, inner - 24)
        )
        out.append(text, style=_DONE_STYLE if entry_status == "done" else "")
        out.append(f"  {entry_status}", style="dim")
        settled_at = str(entry.get("settled_at") or "")
        if settled_at:
            out.append(f" · {settled_at[:16].replace('T', ' ')}", style="dim")
    hidden = len(history) - len(shown)
    if hidden > 0:
        out.append("\n")
        out.append(f"… {hidden} more settled", style="dim")
    out.append("\n")
    keys = "q close" if not actions else "d done · c delete · q close"
    out.append(keys, style="dim")
    return out


def clamp_history_rows(rows: int, available_rows: int) -> int:
    """How many history rows fit the ground the card may use.

    The card must never cover the docked prompt (:func:`overlay.rows_above_dock`
    is the ground), and on a short terminal that means showing fewer settled
    goals and SAYING so rather than pushing the title off the top.
    """
    budget = max(0, available_rows - _CHROME_ROWS - 2)
    return max(0, min(rows, budget, MAX_HISTORY_ROWS))


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
        # vocabulary — and both are free here for a reason worth stating: this
        # panel HOLDS FOCUS while it is open, so its bindings are consulted
        # before the app's (the composer's TextArea cursor keys included).
        # Deliberately NOT app-level chords: the app's global vocabulary is
        # remappable and PERSISTED (`local_operator/keymap.py`), so a new global
        # action would be a keymap id and therefore a config migration for one
        # panel's two gestures.
        ("d", "mark_done", "Done"),
        ("c", "delete", "Delete"),
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
    ) -> None:
        """Display the record as it stands, and repaint."""
        self._goal = goal or ""
        self._status = status or ""
        self._judge = judge
        self._history = list(history)
        self._cap = cap
        self._actions = actions
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
        screen_width, _ = overlay.screen_size(self)
        return max(MIN_PANEL_WIDTH, min(MAX_PANEL_WIDTH, screen_width - 8))

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        width = self.panel_width()
        _, rows_above = overlay.screen_size(self)
        available = overlay.rows_above_dock(self)
        history = self._history[: clamp_history_rows(len(self._history), available)]
        body = build_goal_body(
            goal=self._goal,
            status=self._status,
            judge=self._judge,
            history=history,
            width=width,
            cap=self._cap,
            actions=self._actions,
        )
        self.styles.width = width
        # Pinned rather than `auto` for the reason the usage card pins its own:
        # `auto` measures against a guessed width before layout and settles a row
        # tall, and this card is repainted after every action.
        self.styles.height = len(body.plain.split("\n"))
        if rows_above < 0:  # pragma: no cover - defensive, `rows_above` is floored
            rows_above = 0
        overlay.recentre(self, width, self.styles.height.value or 1)
        self.update(body)

    def on_mount(self) -> None:
        if self.display:
            self._repaint()

    # -- actions -------------------------------------------------------------
    def action_dismiss(self) -> None:
        self.close()
        self.post_message(GoalDismissed())

    def action_mark_done(self) -> None:
        if self._actions:
            self.post_message(GoalMarkDone())

    def action_delete(self) -> None:
        if self._actions:
            self.post_message(GoalDelete())

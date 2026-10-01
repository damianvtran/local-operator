"""The queued-ask surfaces: the minimized bar and the open-ask list.

Design: ``docs/design/ask-nonblocking.md`` §5.0 (R7) and §5.1 (PR B).

Two widgets for one feature, and the split is the interaction model rather than
a filing convention:

* :class:`AskBar` — the **MINIMIZED** state: one line above the composer that
  says how many questions are waiting and expands on a click. It is the default
  presentation of a queued ask, so it must exist while the answer surface is
  NOT mounted: it is a child of the composer's own panel, never of
  ``#prompt-host`` (which reserves rows only while a card is in it).
* :class:`AskQueueList` — the **list** the bar opens when more than one ask is
  open. One row per ask (status glyph, first question, expiry), Enter mounts
  that ask's picker. It is mounted in ``#prompt-host`` beside the picker,
  because both are the *expanded* state and only one of them is ever up.

WHY THE BAR IS NOT THE NOTICE IT REPLACES. A queued ask used to surface as a
one-line transcript notice. A notice scrolls away, is not clickable, and
cannot say how many questions are behind it — and R7 needs exactly those three
things: persistent, clickable, countable. One affordance, not two: the app
replaces the notice with this bar rather than painting both.

COLOUR IS PERSISTENT, NEVER ANIMATED. §5.0 is explicit: the accent is a fixed
colour on the glyph. A pulse would be the focus-steal this design exists to
avoid, expressed in colour instead of keys, so nothing here owns a timer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget

from local_operator.tui import theme as theme_mod

#: The one glyph the ask surfaces share. ``?`` is the question mark the
#: composer chip vocabulary already leaves free (``!`` is the sidebar's
#: needs-you gate mark, ``❯`` the composer caret, ``$`` bang-mode), and using
#: ONE glyph for the bar and the sidebar mark is what lets a user connect the
#: two without a legend.
ASK_MARKER = "?"

#: The chevron at the bar's right edge. Two glyphs, one meaning each: the
#: direction the click will move the surface, exactly like the transcript's
#: expand affordance.
ASK_BAR_CHEVRON_COLLAPSED = "⌄"
ASK_BAR_CHEVRON_EXPANDED = "⌃"

#: The statuses a row can carry from the wire, spelled once so the bar, the
#: list and the transcript card cannot drift about what ``timed_out`` is called.
STATUS_OPEN = "open"
STATUS_TIMED_OUT = "timed_out"
STATUS_LATE = "late"
STATUS_ANSWERED = "answered"

#: Closed-set glyphs for the list's status column. Deliberately NOT a spinner
#: and not an animated set: a queued ask is not doing anything, it is waiting.
#: ``●`` open, ``◷`` timed out but still answerable (the wake glyph's meaning —
#: "a clock decided something"), ``↩`` answered late.
STATUS_MARKS: dict[str, str] = {
    STATUS_OPEN: "●",
    STATUS_TIMED_OUT: "◷",
    STATUS_LATE: "↩",
    STATUS_ANSWERED: "✓",
}


@dataclass(frozen=True)
class AskRow:
    """One wire ask, flattened to what a surface actually draws.

    A view-model rather than the wire model on purpose: the app receives
    ``PendingAskState`` objects on the frontend snapshot, the capture scripts
    and tests feed plain dicts, and the list must render both without a second
    code path per source. Reading happens once, here, so a missing key is a
    default rather than an ``AttributeError`` inside a paint.
    """

    ask_id: str
    status: str
    created_at: int
    expires_at: int
    urgent: bool
    questions: tuple[Mapping[str, Any], ...]

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def head_question(self) -> str:
        """The first question's own text, or "" when the row carries none.

        Empty rather than a placeholder: the list falls back to the ask id in
        its own copy, because a surface that invented a question would be
        showing the user words the model never wrote.
        """
        for question in self.questions:
            text = str(question.get("question") or "").strip()
            if text:
                return text
        return ""


def _read(row: Any, key: str, default: Any = None) -> Any:
    """One field off either a wire model or a plain mapping."""
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def ask_rows(rows: Iterable[Any] | None) -> list[AskRow]:
    """Flatten wire rows, DROPPING the ones no surface may show.

    ``answered`` and ``declined`` are terminal for the USER's purpose: the
    response card in the transcript is where they belong, and a list that kept
    them would make "how many questions are waiting for me" unanswerable —
    which is the list's only job. ``expired`` is dropped for §5.0's own reason:
    an expiry is not a failure and must not sit in a queue claiming attention.
    """
    out: list[AskRow] = []
    for row in rows or ():
        status = str(_read(row, "status", STATUS_OPEN) or STATUS_OPEN)
        if status in ("answered", "declined", "dismissed", "expired"):
            continue
        questions = _read(row, "questions", None) or ()
        out.append(
            AskRow(
                ask_id=str(_read(row, "ask_id", "") or ""),
                status=status,
                created_at=int(_read(row, "created_at", 0) or 0),
                expires_at=int(_read(row, "expires_at", 0) or 0),
                urgent=bool(_read(row, "urgent", False)),
                questions=tuple(q for q in questions if isinstance(q, Mapping)),
            )
        )
    return out


class AskBar(Widget):
    """The minimized ask affordance: one line, click to expand or collapse.

    Focusable so a keyboard can reach it, but it never TAKES focus on its own:
    §5.0's no-auto-mount/no-focus-steal rule covers this widget too. Clicking
    it focuses it as a side effect of the click (Textual focuses a focusable
    widget under a press), and Enter then toggles — so the pointer is the
    primary path and the keyboard is available without a second gesture.
    """

    can_focus = True

    class Toggled(Message):
        """The user asked to expand or collapse the answer surface."""

    # The action is NOT named ``toggle``: Textual's ``DOMNode`` already owns an
    # ``action_toggle(attribute_name)`` (it toggles a CSS class), and pyright
    # rejects the override as an incompatible signature. A name that says what
    # this one does is clearer than a shadowing one anyway.
    BINDINGS = [Binding("enter", "expand_or_collapse", "Expand/collapse", show=False)]

    def __init__(self, widget_id: str = "ask-bar") -> None:
        super().__init__(id=widget_id)
        self._count = 0
        self._head = ""
        self._expanded = False
        self._urgent = False
        #: Whether there is anything to open at all — see `set_state`.
        self._present = False
        #: How many of those are timed out (only meaningful while ``_count`` is
        #: 0, which is the one state where the two differ).
        self._timed_out = 0
        self.display = False

    @property
    def count(self) -> int:
        return self._count

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_state(
        self,
        *,
        count: int,
        head: str = "",
        expanded: bool,
        urgent: bool = False,
        present: bool | None = None,
        timed_out: int = 0,
    ) -> None:
        """Repaint for the current queue.

        ``head`` is the first question, drawn as a dim trailing hint so the bar
        names WHAT is waiting without becoming a second list. It is clipped by
        the caller to a length the narrowest supported terminal can hold; this
        widget never truncates the label itself, because a bar whose meaning
        ("3 questions waiting") can be cut is a bar that lies at 40 columns.

        ``count`` is what the user still OWES — the OPEN asks — while
        ``present`` says whether there is anything at all to open. The two
        differ for exactly one state, and it is the reason both exist: a queue
        of nothing but TIMED-OUT asks owes nobody a wait (the agent moved on),
        but those asks are still answerable (§5's copy), so a bar hidden by
        ``count == 0`` would make them unreachable from the TUI.
        """
        if present is None:
            present = count > 0
        changed = (
            count != self._count
            or head != self._head
            or expanded != self._expanded
            or urgent != self._urgent
            or present != self._present
            or timed_out != self._timed_out
        )
        self._count = count
        self._head = head
        self._expanded = expanded
        self._urgent = urgent
        self._present = present
        self._timed_out = timed_out
        # Nothing to open means no bar at all — not an empty one. A row of
        # blank chrome above the composer would push the dock down for nothing.
        self.display = present
        self.set_class(present, "-ask-visible")
        self.set_class(self._expanded, "-ask-expanded")
        if changed:
            self.refresh(layout=True)

    def action_expand_or_collapse(self) -> None:
        self.post_message(self.Toggled())

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """A click anywhere on the bar toggles, chevron included.

        ``event.stop()`` for §5.0's overlay rule's sibling: a press that
        reaches the dock must not also reach whatever is behind it.
        """
        event.stop()
        self.post_message(self.Toggled())

    def render(self) -> Text:
        # Explicit `Style(color=...)` rather than Rich markup names: a bare word
        # like "accent" is not a Rich style, and Rich silently drops what it
        # cannot parse — a colour that never paints and never errors. The theme
        # module is the one authority for the semantic colours, exactly as the
        # transcript blocks read it.
        accent = Style(color=theme_mod.semantic_color("accent"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(ASK_MARKER, style=Style(bold=True) + (warning if self._urgent else accent))
        if self._count:
            noun = "question waiting" if self._count == 1 else "questions waiting"
            text.append(f" {self._count} {noun}")
        else:
            # Only timed-out asks are left. Their count is not a count of what
            # anyone is waiting for, so it is not spelled as one — the row says
            # what the state actually is (see `set_state`).
            noun = "ask" if self._timed_out == 1 else "asks"
            text.append(f" {self._timed_out} {noun} timed out")
        if self._head:
            text.append(" · ", style=dim)
            text.append(self._head, style=dim)
        if self._expanded:
            verb = " — click to collapse"
        else:
            verb = " — click to answer"
        text.append(verb, style=dim)
        # The chevron sits at the RIGHT EDGE, per §5.0. Read from `self.size`
        # rather than assumed: a bar rendered before its first layout pass has
        # width 0, and padding to a negative width would raise out of a paint.
        chevron = ASK_BAR_CHEVRON_EXPANDED if self._expanded else ASK_BAR_CHEVRON_COLLAPSED
        width = getattr(self.size, "width", 0) or 0
        used = text.cell_len
        if width > used + 1:
            text.append(" " * (width - used - 1))
        text.append(chevron, style=Style(bold=True) + accent)
        return text


class AskQueueList(Widget):
    """The open asks, newest-first with the still-open ones in front.

    Mounted in ``#prompt-host`` — the same slot the picker uses, because both
    are the EXPANDED state and only one is ever up. It takes focus (the answer
    keys are ordinary characters the composer would otherwise swallow), and
    every path out of it hands focus back: Escape collapses, Enter opens an ask
    and the picker owns the keyboard from there.

    Movement CLAMPS rather than wraps. The wheel/clamp rule in AGENTS.md is
    written for a picker over a screen the user is still looking at; this list
    is short and ordered by arrival, so wrapping would teleport a reader who
    pressed ``up`` on the first row to the oldest ask in the queue.
    """

    can_focus = True

    class Picked(Message):
        """The user chose an ask to answer."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    class Collapse(Message):
        """The user asked to collapse back to the minimized bar."""

    class Decline(Message):
        """``d`` — decline the highlighted ask. Explicit, never an Escape side effect."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    class Dismiss(Message):
        """``x`` — take a TIMED-OUT ask out of the view. Injects nothing."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    BINDINGS = [
        Binding("escape", "collapse", "Collapse", show=False),
        Binding("enter", "pick", "Answer", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("k", "move(-1)", "Up", show=False),
        Binding("j", "move(1)", "Down", show=False),
        Binding("d", "decline", "Decline", show=False),
        Binding("x", "dismiss", "Dismiss", show=False),
    ]

    def __init__(self, rows: Sequence[AskRow], widget_id: str = "ask-queue-list") -> None:
        super().__init__(id=widget_id, classes="prompt-slot")
        self._rows: list[AskRow] = list(rows)
        self._index = 0
        self._now_ms = 0

    @property
    def rows(self) -> list[AskRow]:
        return list(self._rows)

    @property
    def index(self) -> int:
        return self._index

    def set_rows(self, rows: Sequence[AskRow], *, now_ms: int = 0) -> None:
        """Replace the list, keeping the highlight on the same ASK where it can.

        By ask id and not by index: a queue that answered one ask and dropped
        another would otherwise slide the highlight onto a different question
        under the user's finger, which is the same defect the picker's held
        answer key exists to prevent.
        """
        current = self._rows[self._index].ask_id if self._rows else ""
        self._rows = list(rows)
        self._now_ms = now_ms
        self._index = next(
            (i for i, row in enumerate(self._rows) if row.ask_id == current),
            0,
        )
        self.refresh(layout=True)

    def current(self) -> AskRow | None:
        if not self._rows:
            return None
        return self._rows[min(self._index, len(self._rows) - 1)]

    def action_move(self, delta: int) -> None:
        if not self._rows:
            return
        self._index = max(0, min(len(self._rows) - 1, self._index + delta))
        self.refresh()

    def action_pick(self) -> None:
        row = self.current()
        if row is not None:
            self.post_message(self.Picked(row.ask_id))

    def action_collapse(self) -> None:
        self.post_message(self.Collapse())

    def action_decline(self) -> None:
        row = self.current()
        if row is not None:
            self.post_message(self.Decline(row.ask_id))

    def action_dismiss(self) -> None:
        row = self.current()
        if row is not None:
            self.post_message(self.Dismiss(row.ask_id))

    def render(self) -> Text:
        accent = Style(color=theme_mod.semantic_color("accent"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        bold = Style(bold=True)
        text = Text()
        count = len(self._rows)
        text.append(f"{ASK_MARKER} {count} open ask")
        text.append("s" if count != 1 else "")
        text.append("  ·  enter answer · esc collapse\n", style=dim)
        now = self._now_ms
        for index, row in enumerate(self._rows):
            selected = index == self._index
            mark = STATUS_MARKS.get(row.status, "●")
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("❯ " if selected else "  ", style=bold if selected else dim)
            line.append(f"{mark} ", style=dim)
            line.append(
                row.head_question or f"(ask {row.ask_id})", style=accent if selected else None
            )
            expiry = _expiry_text(row, now)
            if expiry:
                line.append(f"  {expiry}", style=dim)
            text.append(line)
            if index + 1 < count:
                text.append("\n")
        return text


def _expiry_text(row: AskRow, now_ms: int) -> str:
    """``expires in 42m`` / ``timed out`` — one honest word per state.

    Derived from ``expires_at`` against the client's own clock, which is what
    §5 says the countdown is: the server states a deadline, the client decides
    how long that is from here. A row with no deadline says nothing rather than
    ``in 0s``.
    """
    if row.status == STATUS_TIMED_OUT or row.status == STATUS_LATE:
        return "timed out — still answerable"
    if not row.expires_at:
        return ""
    if not now_ms:
        return ""
    remaining_s = int((row.expires_at - now_ms) // 1000)
    if remaining_s <= 0:
        return "expiring"
    if remaining_s < 60:
        return f"expires in {remaining_s}s"
    if remaining_s < 3600:
        return f"expires in {remaining_s // 60}m"
    return f"expires in {remaining_s // 3600}h"


__all__ = [
    "ASK_BAR_CHEVRON_COLLAPSED",
    "ASK_BAR_CHEVRON_EXPANDED",
    "ASK_MARKER",
    "STATUS_MARKS",
    "AskBar",
    "AskQueueList",
    "AskRow",
    "ask_rows",
]

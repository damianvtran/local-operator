"""The queued-ask surfaces: the minimized bar and the open-ask list.

Design: ``docs/design/ask-nonblocking.md`` §5.0 (R7) and §5.1 (PR B).

Two widgets for one feature, and the split is the interaction model rather than
a filing convention:

* :class:`AskBar` — the **MINIMIZED** state: one line above the composer that
  says how many asks are waiting and expands on a click or
  :data:`ASK_TOGGLE_KEY`. It is the
  default presentation of a queued ask, so it must exist while the answer
  surface is NOT mounted: it is a child of the composer's own panel, never of
  ``#prompt-host`` (which reserves rows only while a card is in it).
* :class:`AskQueueList` — the **list** the bar opens when more than one ask is
  open. One row per ask (status glyph, first question, expiry), a click or
  Enter mounts that ask's picker. It is mounted in ``#prompt-host`` beside the
  picker, because both are the *expanded* state and only one is ever up.

WHY THE BAR IS NOT THE NOTICE IT REPLACES. A queued ask used to surface as a
one-line transcript notice. A notice scrolls away, is not clickable, and
cannot say how many questions are behind it — and R7 needs exactly those three
things: persistent, clickable, countable. One affordance, not two: the app
replaces the notice with this bar rather than painting both.

COLOUR IS PERSISTENT, NEVER ANIMATED. §5.0 is explicit: the accent is a fixed
colour on the glyph. A pulse would be the focus-steal this design exists to
avoid, expressed in colour instead of keys, so nothing here owns a timer.

INK IS CHOSEN AGAINST THE GROUND IT REALLY SITS ON, and that is a review
outcome rather than a preference (design round 1, D2/D3). The first cut painted
the bar's own message in ``dim`` on ``raised`` (3.81:1 dark, 3.09:1 LIGHT —
under the palette gate's 3.4:1 dim floor) and the list's rows in ``dim`` on
``overlay`` (3.43:1 dark, 2.72:1 light). The gate checks ``bg``/``surface``
only, which is exactly how both slipped through: a new surface has to solve its
own pairs. So the bar's ground is the composer's ``surface`` (the gate's own
ground, where ``fg`` 13.76/13.86, ``muted`` 7.93/6.58 and ``dim`` 4.18/3.46 all
clear their floors, and the accent glyph reads 8.02/4.43) and the list's rows
avoid ``dim`` entirely on ``overlay`` (``fg`` 11.30/10.92, ``muted`` 6.51/5.18,
``warning`` 7.09/4.78).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget

from local_operator.asks import store
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
#: The key that reaches the ask surface without a mouse, named ONCE so the bar's
#: copy cannot advertise a key the app does not bind.
#:
#: A FUNCTION key, continuing the app's f8/f9/f10 row, because the composer owns
#: the letters: ``TextArea`` binds ctrl+a/e/w/d/x/k/u/z/y/c/v and, measured in
#: Textual 8.2.8, ``f6``/``f7`` as well — so f7, the obvious neighbour, is
#: swallowed by the focused composer and never reaches an app binding (found by
#: driving the key in a pilot, round 1 remediation). f8/f9/f10 are the app's.
#: f5 is avoided too: ``editor.py``'s note claims TextArea binds f5 as a
#: selection chord, and a dispute about one key is not worth winning.
ASK_TOGGLE_KEY = "f4"

ASK_BAR_CHEVRON_COLLAPSED = "⌄"
ASK_BAR_CHEVRON_EXPANDED = "⌃"

#: The statuses a row can carry, re-exported from ``asks/store.py``'s fold rather
#: than spelled here. The bar, the list and the transcript card used to keep
#: their OWN literals "spelled once" in this module, which is exactly how a
#: surface ends up disagreeing with the wire about what ``timed_out`` means:
#: the only spelling that is allowed to be authoritative is the fold's.
STATUS_OPEN = store.STATUS_OPEN
STATUS_TIMED_OUT = store.STATUS_TIMED_OUT
STATUS_LATE = store.STATUS_LATE
STATUS_ANSWERED = store.STATUS_ANSWERED
STATUS_DECLINED = store.STATUS_DECLINED
STATUS_DISMISSED = store.STATUS_DISMISSED

#: Statuses a SURFACE drops, because the user has nothing left to do about
#: them. ``answered`` and ``declined`` are settled; ``late`` joins them because
#: it is the same settlement one deadline later — the agent was told, the
#: receipt is in the transcript, and offering an answer box for it can only be
#: refused (design round 1, D6/U3: ``late`` was painted "timed out — still
#: answerable" and stayed in the answerable set).
SETTLED_STATUSES = frozenset({STATUS_ANSWERED, STATUS_LATE, STATUS_DECLINED, STATUS_DISMISSED})
#: ...which is to say: everything the reader must not be asked to answer. THE
#: OUTSTANDING SET, taken from the fold rather than re-listed here: a row the
#: user can still answer is ``open`` or timed out and unanswered, and that ONE
#: rule decides the wire's tally, this surface's rows and the sidebar mark, so
#: it lives in ``asks.store.OUTSTANDING_STATUSES`` and nowhere else.
_ANSWERABLE = store.OUTSTANDING_STATUSES

#: Closed-set glyphs for the list's status column. Deliberately NOT a spinner
#: and not an animated set: a queued ask is not doing anything, it is waiting.
#: ``●`` open, ``◷`` timed out but still answerable (the wake glyph's meaning —
#: "a clock decided something"). A settled row has no glyph because it has no
#: row: see :data:`SETTLED_STATUSES`.
STATUS_MARKS: dict[str, str] = {
    STATUS_OPEN: "●",
    STATUS_TIMED_OUT: "◷",
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
        telling the reader something the model never said.
        """
        for question in self.questions:
            text = str(question.get("question") or "").strip()
            if text:
                return text
        return ""

    @property
    def answerable(self) -> bool:
        """Whether the user can still do something about this row."""
        return self.status in _ANSWERABLE

    @property
    def waiting(self) -> bool:
        """Whether anyone is still WAITING for this row's answer.

        Narrower than :attr:`answerable`, and the distinction is the copy's: a
        timed-out ask is answerable (the user may still reply, and the agent is
        told) but nobody is waiting on it — the agent moved on.
        """
        return self.status == STATUS_OPEN


def _value(row: Any, key: str) -> Any:
    """Read one field from a wire model OR a plain mapping."""
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def ask_rows(rows: Iterable[Any] | None) -> list[AskRow]:
    """Flatten the wire's asks into the surface view-model, dropping settled ones.

    ``None`` and ``[]`` both mean "no queued asks" — the wire is ABSENT, not
    empty, while this runtime runs the BLOCKING arm (the kill switch, or an
    older build), so a caller that mistook absence for a list would be
    rendering a feature the runtime does not have.
    """
    out: list[AskRow] = []
    for row in rows or ():
        status = str(_value(row, "status") or STATUS_OPEN)
        if status in SETTLED_STATUSES or status not in _ANSWERABLE:
            continue
        questions = _value(row, "questions") or ()
        out.append(
            AskRow(
                ask_id=str(_value(row, "ask_id") or ""),
                status=status,
                created_at=int(_value(row, "created_at") or 0),
                expires_at=int(_value(row, "expires_at") or 0),
                urgent=bool(_value(row, "urgent")),
                questions=tuple(q for q in questions if isinstance(q, Mapping)),
            )
        )
    return out


def queue_headline(open_count: int, timed_out_count: int, urgent_count: int = 0) -> str:
    """The queue's count, in ONE vocabulary for the bar and the list header.

    Two totals used to describe one queue: the bar counted the OPEN asks
    ("2 questions waiting") while the list header counted every row it drew
    ("3 open asks", the timed-out one included), and a reader with both on
    screen saw two numbers three rows apart (design D7 / UX U4 / QA Q4). The
    words are here, once, so the two surfaces cannot disagree about what they
    are counting — and the no-timed-out case keeps §5's own copy verbatim.
    """
    parts: list[str] = []
    if open_count:
        noun = "question" if open_count == 1 else "questions"
        parts.append(f"{open_count} {noun} waiting")
    if urgent_count:
        # URGENCY IN WORDS, because the bar's amber glyph is a hue and a hue is
        # not a channel a reader without colour has: D8's own principle, which
        # had reached the list row and not the bar (design round 2, D13). It
        # also removes the mismatch D13 found — the glyph is painted from
        # `any(row.urgent)`, so with no word for it the amber annotated the ONE
        # question the bar happened to name, which need not be the urgent one.
        parts.append(f"{urgent_count} urgent")
    if timed_out_count:
        noun = "ask" if timed_out_count == 1 else "asks"
        parts.append(f"{timed_out_count} {noun} timed out")
    return " · ".join(parts)


class AskBar(Widget):
    """The minimized ask affordance: one line, click or the toggle key to expand.

    Focusable so a keyboard can reach it, but it never TAKES focus on its own:
    §5.0's no-auto-mount/no-focus-steal rule covers this widget too. Clicking
    it focuses it as a side effect of the click (Textual focuses a focusable
    widget under a press), and Enter then toggles — so the pointer is the
    primary path and the keyboard is available without a second gesture. The
    app's own :data:`ASK_TOGGLE_KEY` binding is the route that needs no
    focus at all, and the tests pin that the key is one the composer does not
    claim (UX
    round 1, U5: the bar's own Enter binding was unreachable, because nothing
    focuses a bar the user cannot see is focusable).
    """

    can_focus = True

    class Toggled(Message):
        """The user asked to expand or collapse the answer surface."""

    #: NOT named ``toggle``: Textual's ``DOMNode`` already owns an
    #: ``action_toggle(attribute_name)`` (it toggles a CSS class), and pyright
    #: rejects the override as an incompatible signature. A name that says what
    #: this one does is clearer than a shadowing one anyway.
    BINDINGS = [Binding("enter", "expand_or_collapse", "Expand/collapse", show=False)]

    def __init__(self, widget_id: str = "ask-bar") -> None:
        super().__init__(id=widget_id)
        self._count = 0
        self._timed_out = 0
        self._head = ""
        self._expanded = False
        self._urgent = False
        self._urgent_count = 0
        #: Whether there is anything to open at all — see `set_state`.
        self._present = False
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
        timed_out: int = 0,
        head: str = "",
        expanded: bool,
        urgent: bool = False,
        urgent_count: int = 0,
        present: bool | None = None,
    ) -> None:
        """Repaint for the current queue.

        ``head`` is the first question, drawn as a dim trailing hint so the bar
        names WHAT is waiting without becoming a second list. It is clipped by
        THIS widget, against the width the row really has (design D5: a fixed
        clip plus a tail appended after it overflowed a 76-cell bar at 80x24
        and lost the chevron entirely).

        ``count`` is what the user still OWES — the OPEN asks — while
        ``present`` says whether there is anything at all to open. The two
        differ for exactly one state, and it is the reason both exist: a queue
        of nothing but TIMED-OUT asks owes nobody a wait (the agent moved on),
        but those asks are still answerable (§5's copy), so a bar hidden by
        ``count == 0`` would make them unreachable from the TUI.
        """
        if present is None:
            present = count > 0 or timed_out > 0
        changed = (
            count != self._count
            or timed_out != self._timed_out
            or head != self._head
            or expanded != self._expanded
            or urgent != self._urgent
            or urgent_count != self._urgent_count
            or present != self._present
        )
        self._count = count
        self._timed_out = timed_out
        self._head = head
        self._expanded = expanded
        self._urgent = urgent
        self._urgent_count = urgent_count
        self._present = present
        # Nothing to open means no bar at all — not an empty one. A row of
        # blank chrome above the composer would push the dock down for nothing.
        self.display = present
        self.set_class(present, "-ask-visible")
        self.set_class(self._expanded, "-ask-expanded")
        if changed:
            self.refresh(layout=True)

    def action_expand_or_collapse(self) -> None:
        self.post_message(self.Toggled())

    def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        """A click anywhere on the bar toggles, chevron included.

        ``event.stop()`` for §5.0's overlay rule's sibling: a press that
        reaches the dock must not also reach whatever is behind it.
        """
        event.stop()
        self.post_message(self.Toggled())

    def render(self) -> Text:
        """The bar, laid out against the width it actually has.

        Explicit ``Style(color=...)`` rather than Rich markup names: a bare word
        like "accent" is not a Rich style, and Rich silently drops what it
        cannot parse — a colour that never paints and never errors.

        THE SACRIFICE ORDER IS DELIBERATE and is what design D5 asked for: the
        chevron is reserved first (it is the only thing saying which way this
        control goes), then the count, then the affordance words, and the head
        QUESTION is what gives way — it is the one part that is also available
        by expanding, while a bar with no chevron or no count is a bar that
        stopped being a control.
        """
        accent = Style(color=theme_mod.semantic_color("accent"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        glyph_style = Style(bold=True) + (warning if self._urgent else accent)

        # Read from `self.size` rather than assumed: a bar rendered before its
        # first layout pass has width 0.
        width = max(0, int(getattr(self.size, "width", 0) or 0))
        chevron = ASK_BAR_CHEVRON_EXPANDED if self._expanded else ASK_BAR_CHEVRON_COLLAPSED
        # One cell for the chevron and one of separation from the copy; a bar
        # too narrow for both keeps the glyph and the count and drops the rest
        # rather than butting the chevron against its own words (UX U10).
        avail = width - 2
        if avail <= 1:
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(ASK_MARKER, style=glyph_style)
            return text
        base = f"{ASK_MARKER} {queue_headline(self._count, self._timed_out, self._urgent_count)}"
        verb = "collapse" if self._expanded else "answer"
        tail = f" — {ASK_TOGGLE_KEY} or click to {verb}"
        head_part = f" · {self._head}" if self._head else ""
        used = cell_len(base)
        if head_part and used + cell_len(tail) + 8 <= avail:
            # The whole room that is left, with no second ceiling on top of it:
            # the tail is reserved first and the head gives way, so a further
            # cap can only truncate where the row had space (design round 2,
            # D14 — the old 48-cell ceiling still bit at 130 columns while the
            # frame showed eleven empty cells before the chevron).
            room = avail - used - cell_len(tail) - 3
            head_part = f" · {_clip_cells(self._head, room)}" if room >= 8 else ""
        else:
            head_part = ""
        if used + cell_len(head_part) + cell_len(tail) > avail:
            tail = ""
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(ASK_MARKER, style=glyph_style)
        text.append(
            f" {queue_headline(self._count, self._timed_out, self._urgent_count)}", style=fg
        )
        if head_part:
            text.append(head_part, style=fg)
        if tail:
            text.append(tail, style=dim)
        padding = width - 1 - cell_len(text.plain)
        if padding > 0:
            text.append(" " * padding)
        text.append(chevron, style=Style(bold=True) + accent)
        return text


def _clip_cells(text: str, room: int) -> str:
    """``text`` clipped to ``room`` cells with an ellipsis, cheaply.

    The local spelling of ``tool_card.truncate_cells`` rather than an import:
    this module is on the dock's paint path and the tool-card module is not
    otherwise needed here. Both agree on the ellipsis and the cell count.
    """
    if room <= 0:
        return ""
    if cell_len(text) <= room:
        return text
    if room == 1:
        return "…"
    clipped = ""
    for char in text:
        if cell_len(clipped + char) > room - 1:
            break
        clipped += char
    return clipped + "…"


class AskQueueList(Widget):
    """The open-ask list: one row per ask, Enter or a click to answer one.

    A ``Widget`` rather than a ``Container`` of rows: the rows are painted from
    state and hit-tested by line, which is what keeps a queue of eight asks one
    widget instead of eight focusable children competing for the tab order.
    """

    can_focus = True

    class Picked(Message):
        """The user chose one ask to answer."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    class Collapse(Message):
        """The user closed the surface."""

    class Decline(Message):
        """The user declined one ask — terminal, so it is an explicit action."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    class Dismiss(Message):
        """The user dismissed one TIMED-OUT ask from the view."""

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

    #: The header occupies the first painted line; a click on it selects
    #: nothing, which is what `_row_at` returns for it. ONE painted row by
    #: construction — see `header_parts`, which drops whole hints rather than
    #: letting the line wrap.
    HEADER_ROWS = 1

    #: The header's hints, in the order they are SPENT — earlier entries are
    #: kept longest, so the irreversible `d` outlives the reversible tips when
    #: the row runs out of room (UX round 1, U7 asked for `d` to be named
    #: precisely because it cannot be undone).
    HEADER_HINTS = (
        "enter answer",
        "d decline",
        "x dismiss",
        "esc collapse",
    )

    def __init__(
        self, rows: Sequence[AskRow], widget_id: str = "ask-queue-list", now_ms: int = 0
    ) -> None:
        super().__init__(id=widget_id, classes="prompt-slot")
        self._rows: list[AskRow] = list(rows)
        self._index = 0
        self._now_ms = now_ms

    @property
    def rows(self) -> list[AskRow]:
        return list(self._rows)

    @property
    def index(self) -> int:
        return self._index

    def set_rows(self, rows: Sequence[AskRow], *, now_ms: int = 0) -> None:
        """Replace the list, keeping the highlight on the same ASK where it can.

        Called from the app's frontend-snapshot writer, which is what makes the
        panel follow the wire: without it an ask answered on another surface
        kept its row (and its answerability) until the user collapsed and
        re-expanded (QA Q3 / design D4).

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

    def set_now(self, now_ms: int) -> None:
        """Advance the countdown's clock without touching the rows.

        The app's countdown tick calls THIS rather than :meth:`set_rows`: no row
        changed, and re-deriving the highlight by ask id every 30 s is work that
        can only get the highlight wrong. A plain ``refresh`` and not
        ``refresh(layout=True)`` — the words ``expiry_text`` paints are clipped
        against the width read at paint time, so a countdown that gains a cell
        (``9m`` → ``10m``) shortens the question under it rather than reflowing
        the row, which is what keeps the one-painted-row-per-ask invariant the
        pointer hit test rests on.
        """
        if now_ms == self._now_ms:
            return
        self._now_ms = now_ms
        self.refresh()

    def select(self, index: int) -> None:
        """Put the cursor on a row by index, clamped — used to hand a card's
        user back to the row they came from (UX round 1, U6, where Escaping a
        card reset the highlight to row 1 and made them hunt for their place)."""
        self._index = max(0, min(len(self._rows) - 1, index)) if self._rows else 0
        self.refresh()

    def current(self) -> AskRow | None:
        if not self._rows:
            return None
        return self._rows[min(self._index, len(self._rows) - 1)]

    # -- keys ----------------------------------------------------------------

    def action_move(self, delta: int) -> None:
        if not self._rows:
            return
        # Clamped, not wrapped: the list is short, overlaid, and a wrap would
        # throw the user from the last row to the first under a held key.
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

    # -- the pointer ---------------------------------------------------------

    def _top_inset(self) -> int:
        """Rows between this widget's OUTER edge and its first painted row.

        Taken from Textual's own regions rather than from a copied constant: the
        panel carries ``padding: 1 1``, and ``event.y`` arrives relative to the
        OUTER region, so a hit test that treated ``y`` as content-relative was a
        row ahead of the paint — which is exactly what review round 2's MAJOR
        found (a click opened the ask BELOW the one under the pointer, and the
        last ask was unreachable). ``content_region - region`` is Textual's own
        arithmetic for padding plus border, so it cannot drift from the
        stylesheet the way a constant would.
        """
        try:
            return max(0, int(self.content_region.y - self.region.y))
        except Exception:  # noqa: BLE001 — pre-layout, when there is nothing to hit
            return 0

    def _row_at(self, y: int) -> int | None:
        """The row index under a widget-relative y, or ``None`` for the header.

        Both offsets are MEASURED: the widget's own top inset, and a header that
        is one painted line by construction (`header_parts` never wraps, and
        `HEADER_HINTS` is spent rather than broken mid-phrase).
        """
        index = y - self._top_inset() - self.HEADER_ROWS
        if 0 <= index < len(self._rows):
            return index
        return None

    def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        """A click on a row selects it and opens it; a click elsewhere is inert.

        Without this the list was mouse-dead AND worse than dead: Textual's own
        click-to-focus walked up from the row and landed on the composer, so a
        click both failed to choose and disarmed the keyboard the list was
        using (UX round 1, U2 — `down`/`enter` stopped moving the highlight
        while `❯` was still painted, so the panel read as live).
        """
        event.stop()
        row = self._row_at(int(event.y))
        if row is None:
            return
        self._index = row
        self.refresh()
        self.post_message(self.Picked(self._rows[row].ask_id))

    def on_mouse_move(self, event: events.MouseMove) -> None:  # type: ignore[override]
        """Hover moves the highlight, so the row under the pointer is the one
        a click would take — the picker card's own rule."""
        row = self._row_at(int(event.y))
        if row is None or row == self._index:
            return
        self._index = row
        self.refresh()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:  # type: ignore[override]
        event.stop()
        self.action_move(1)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:  # type: ignore[override]
        event.stop()
        self.action_move(-1)

    # -- paint ---------------------------------------------------------------

    def header_parts(self, width: int) -> tuple[str, str]:
        """``(headline, hint suffix)`` for a ONE-LINE header at this width.

        Hints are added left to right and dropped from the right the moment one
        would not fit, so the line never splits a key from its verb and never
        spends a second row (design round 2, D12 — the header was free to wrap,
        which both orphaned an ``x`` at the line end and shifted every pointer
        hit by a row). A dropped hint is still TRUE — the key works — it is
        merely not advertised, the same sacrifice order the bar uses one widget
        up.

        Public because the hit test and the tests both need the rule without
        restating it.
        """
        waiting_rows = [row for row in self._rows if row.status == STATUS_OPEN]
        # The SAME set the bar counts: OPEN rows. An urgent ask is one whose
        # deadline is imminent, and a timed-out ask has no imminent deadline —
        # counting it here made the list say `1 urgent · 1 ask timed out` over a
        # bar showing only the amber hue, i.e. two surfaces disagreeing about one
        # queue again (review round 3, MINOR-1).
        urgent = sum(1 for row in waiting_rows if row.urgent)
        timed_out = len(self._rows) - len(waiting_rows)
        headline = f"{ASK_MARKER} {queue_headline(len(waiting_rows), timed_out, urgent)}"
        kept: list[str] = []
        for hint in self.HEADER_HINTS:
            if cell_len("  ·  ".join([headline, *kept, hint])) > width:
                break
            kept.append(hint)
        return headline, ("  ·  " + "  ·  ".join(kept) if kept else "")

    def render(self) -> Text:
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        bold = Style(bold=True)
        # `content_size`, not `size`: the box the text is really painted in.
        headline, hints = self.header_parts(max(1, int(self.content_size.width)))
        text = Text()
        # `no_wrap` + ellipsis, so even a headline too long for one row is CUT
        # rather than wrapped: a wrapped header is what shifted every hit below
        # it (round 2, MAJOR-1).
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append(headline, style=fg)
        header.append(hints, style=muted)
        text.append_text(header)
        if self._rows:
            text.append("\n")
        for index, row in enumerate(self._rows):
            selected = index == self._index
            mark = STATUS_MARKS.get(row.status, "●")
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("❯ " if selected else "  ", style=bold if selected else muted)
            line.append(f"{mark} ", style=warning if row.urgent else muted)
            # The SELECTED question is `fg` and not `accent`: accent on this
            # panel's ground reads 3.49:1 in the light ramp, under the 4.0 the
            # palette gate sets for a state hue (design D3). Selection is
            # carried by the `❯` marker and the weight instead, which is also
            # the one cue that survives NO_COLOR.
            # THE QUESTION GIVES WAY, not the row's tail, and the cut is made
            # HERE rather than left to the widget's overflow: a row that is
            # wider than the box is hard-cropped by the painter (no glyph), and
            # because the expiry is appended last it was the expiry — the
            # per-row WORD for urgency, which design D13 added so meaning did
            # not rest on hue alone — that fell off the end (agent review round
            # 4, MINOR-1). Clipping the question to the room the tail leaves
            # puts the cut where it belongs and paints the ellipsis that says
            # something was cut.
            expiry = expiry_text(row, self._now_ms)
            # Padded to the widest countdown this row can paint, so the cut in
            # the question above stays where it was when the clock moves
            # (`expiry_room`): a row that filled the line used to re-truncate
            # itself on every tick. A row with no deadline has no tail and no
            # reservation — nothing about it can move.
            room = expiry_room(row, expiry) if expiry else 0
            tail = f"  {expiry}{' ' * max(0, room - 2 - cell_len(expiry))}" if expiry else ""
            fixed = 4 + cell_len(tail)  # two marker cells, two state-glyph cells
            question = _clip_cells(
                row.head_question or f"(ask {row.ask_id})",
                max(1, self.content_size.width - fixed),
            )
            line.append(question, style=(bold + fg) if selected else muted)
            if tail:
                line.append(tail, style=warning if row.urgent else muted)
            text.append(line)
            if index + 1 < len(self._rows):
                text.append("\n")
        return text


def expiry_text(row: AskRow, now_ms: int) -> str:
    """``expires in 42m`` / ``timed out`` / ``urgent`` — one honest word per state.

    Derived from ``expires_at`` against the client's own clock, which is what
    §5 says the countdown is: the server states a deadline, the client decides
    how long that is from here. A row with no deadline says nothing rather than
    ``in 0s``.

    URGENCY CARRIES WORDS, not only the amber glyph (design D8): the bar's `?`
    turning amber is invisible to a reader who cannot see the hue and to a
    transcript, so the row says ``urgent`` too.
    """
    if row.status == STATUS_TIMED_OUT:
        return "timed out — still answerable"
    if not row.expires_at or not now_ms:
        return "urgent" if row.urgent else ""
    remaining_s = int((row.expires_at - now_ms) // 1000)
    if remaining_s <= 0:
        return "urgent · expiring" if row.urgent else "expiring"
    if remaining_s < 60:
        left = f"{remaining_s}s"
    elif remaining_s < 3600:
        left = f"{remaining_s // 60}m"
    else:
        left = f"{remaining_s // 3600}h"
    return f"urgent · expires in {left}" if row.urgent else f"expires in {left}"


def expiry_room(row: AskRow, expiry: str) -> int:
    """Cells a row's ``  <expiry>`` TAIL may need, over its whole life.

    The tail CHANGES WIDTH as the clock runs — ``10m`` is one cell wider than
    ``9m``, the words change unit at the hour and minute boundaries, and
    ``expiring`` replaces them all — so a question that filled its line got
    RE-CUT by its own countdown: ``…which shard …`` became ``…which shard s…``
    without the user touching anything, on a surface whose whole promise is
    that a tick repaints and changes nothing else (design round 1, D2).
    Reserving the widest form the row can reach keeps the question's clip
    budget fixed, which is what makes the tick a pure repaint.

    Derived from the row's OWN span (``expires_at - created_at``, the deadline
    the ask was given) rather than a global constant: a constant wide enough
    for ``expires in 999h`` would eat ten cells of question from every ask
    forever. ``expiry_text`` prints minutes and seconds below an hour — always
    at most two digits — so only the hours form can be wider, and the digits
    it can reach are the ones the row's own span implies.

    The returned width includes the two separator cells, and it never returns
    less than what ``expiry`` already needs: a client clock skewed far ahead
    of the wire's can paint a longer word than the span implies, and a tail is
    the one thing on this row that must not be cropped (the expiry and its
    urgency word are the meaning the hue alone cannot carry).
    """
    if row.status == STATUS_TIMED_OUT:
        widest = cell_len("timed out — still answerable")
    else:
        prefix = "urgent · expires in " if row.urgent else "expires in "
        span_s = (row.expires_at - row.created_at) // 1000 if row.created_at else 0
        widest = cell_len(prefix) + max(2, len(str(max(0, span_s) // 3600))) + 1
    return 2 + max(widest, cell_len(expiry))


__all__ = [
    "ASK_BAR_CHEVRON_COLLAPSED",
    "ASK_BAR_CHEVRON_EXPANDED",
    "ASK_MARKER",
    "ASK_TOGGLE_KEY",
    "SETTLED_STATUSES",
    "STATUS_MARKS",
    "STATUS_OPEN",
    "STATUS_TIMED_OUT",
    "AskBar",
    "AskQueueList",
    "AskRow",
    "ask_rows",
    "expiry_room",
    "expiry_text",
    "queue_headline",
]

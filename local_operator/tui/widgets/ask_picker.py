"""The ``ask`` tool's picker: the agent's question as a real answerable list.

Why this exists: a model that needs a decision has, until now, had exactly one
shape available to it — prose. Observed verbatim in a live session:

    (A) Drop email … (B) Escalate it properly … (C) You have context I don't

Three options printed into the transcript, none of them clickable, none of them
selectable, and the user's answer arriving as free text the agent then has to
re-parse. This screen is the other half of the ``ask`` tool: one question at a
time, keyboard and mouse, answering with the label the model wrote.

It is ANCHORED IN THE DOCK, not pushed as a modal screen, and that is the
central design decision. A ``ModalScreen`` covers the terminal: the question
arrived and the conversation it is about — the tool output, the error, the plan
the agent is asking about — went behind it, unreadable and unscrollable. A user
who needed to look something up to answer had to abandon the question to do it.
Mounted in ``#prompt-host`` instead, the card is a row band at the top of the
input dock: the transcript keeps the rest of the screen, keeps its scrollback,
and the card stays put while the user scrolls up for the context they need.

It sits ABOVE the dock band (subagents, todos) rather than below it, because
those panels are status and this is a question the turn is parked on: the thing
being waited on belongs closest to the eye's resting place, and a status list
that pushed the question around as jobs came and went would move the card under
the user's cursor mid-answer.

Its frame is the ``/resume`` picker's (one ``Static``, content-sized, measured
every paint) because that surface already solved the parts that are easy to get
wrong here: the card clipping its own footer, the cursor sitting on a row that
was never drawn, and a click resolving to a row that was never painted.

Two things it does that no other picker in this app does:

- **Every question offers a free-text answer.** The prose surface this replaces
  needed one constantly — "(C) You have context I don't" is a model asking for
  an answer it could not enumerate. Selecting the ``Other`` row turns it into a
  text field, so the options never have to be exhaustive.
- **It answers with TEXT, not an index.** The free-text row hands back a string
  that was never in ``options``, which an index cannot express.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.harness.types import AskQuestion
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells
from local_operator.tui.widgets.transcript import wrap_cells

#: Cells of the card's own padding on EACH side, mirroring the horizontal half
#: of ``padding: 1 1`` in the stylesheet. One cell rather than the modal's two:
#: docked, this is the same rail every other panel in the dock uses (the
#: composer's ``❯`` column, ``.band-body { padding: 0 1 }``), so the question
#: starts in the column the rest of the dock starts in. Chrome that sets its
#: own left edge is what makes a docked panel look detached from the input.
ASK_PADDING_CELLS = 1


#: Rows the transcript claims even when it is showing nothing at all.
#:
#: It is ``height: 1fr`` and a sibling of the dock under the screen, so it takes
#: its rows whatever the dock does with the rest. A card that budgeted the
#: screen without them produced a dock taller than the screen — a scrollable
#: screen with the top of the question cut off. Distinct from
#: :data:`MIN_TRANSCRIPT_ROWS`, which is the anchoring PREFERENCE: this is the
#: layout engine's floor, and it applies even where the preference has yielded.
#:
#: TWO, not one: ``TranscriptView`` carries ``padding: 1 0 1 1`` — a row of the
#: conversation's own ground above and below — and padding is inside the
#: scrollable region, so both survive. Written as 1 the budget was short by
#: exactly one row and `virtual_size` exceeded `size` at 100x14, 54x14, 80x13,
#: 30x12 and 40x12 (F2, agent review round 1).
#:
#: It was invisible to every geometry test in this file because they all mount
#: into an app with an EMPTY transcript, which is still in the boot layout —
#: and ``Screen.boot TranscriptView`` drops the padding to ``0 0 0 1``,
#: cancelling the error exactly. The tests now seed a conversation, which is
#: also the only state in which this surface is ever actually used.
_TRANSCRIPT_MIN_ROWS = 2

#: Rows to assume the rest of the dock needs before it has been laid out — a
#: one-row composer, its chevron row's padding, and the status band. Used only
#: on the first paint of a session's first question, and deliberately an
#: OVER-estimate of the minimum dock: reserving rows the dock does not need
#: costs one windowed row for one frame, while reserving too few lets the card
#: lay out over the composer and lose its footer to the clip.
_DOCK_ROWS_FALLBACK = 4

#: Rows the card costs the dock beyond its own text: one padding row above, one
#: below (the vertical half of ``padding: 1 1``), plus the one rhythm row its
#: slot owns underneath (``.prompt-slot { padding: 0 0 1 0 }``), which separates
#: the question from the band or the composer below it.
#:
#: Kept separate from the cell budget above rather than shared with it: the two
#: were equal by coincidence under the old ``padding: 1 2`` and spending one for
#: the other is a bug waiting for the stylesheet to move, which it since has.
CARD_PADDING_ROWS = 3

#: The most of the terminal the card may ever claim, as a share of its rows.
#:
#: This is the anchoring rule expressed as a number, and it is the whole reason
#: the surface was moved out of a ``ModalScreen``. A modal took the screen, so
#: the conversation the question is ABOUT went behind it: the tool output that
#: prompted the ask, the error being asked about, the plan under discussion. A
#: user who needed to re-read any of that to answer had to dismiss the question
#: first. Capped here, the card is always a BAND and the transcript is always
#: the rest of the screen.
#:
#: A share rather than a row count because the thing being protected is
#: proportional: two rows of transcript under a card is not "some conversation
#: visible", it is a sliver that answers nothing. The floor below is what keeps
#: the rule meaningful on a short terminal, where a share alone rounds to almost
#: everything.
#:
#: 0.7 is the same ratio omp's ask dialog settled on (``DIALOG_HEIGHT_RATIO``),
#: and the agreement is not a coincidence: it is the point where a six-option
#: question can still show the CONSEQUENCE line under each option on an
#: ordinary 30-row terminal. Measured at 0.6 the card had 15 body rows against
#: the 17 that question needs, so every description was dropped — and the
#: descriptions are what the user is comparing when they choose. At 0.7 the same
#: question draws in full and still leaves six rows of conversation.
PROMPT_HEIGHT_SHARE = 0.7

#: Conversation rows the card will not take, whatever the share above allows.
#:
#: The share is a ceiling on the CARD; this is a floor under the TRANSCRIPT, and
#: both are needed because they bind at opposite ends of the size range. On a
#: tall terminal the share is what stops a twelve-option question from filling
#: the screen; on a short one the share alone would still leave the conversation
#: with a row or two, so this takes over and the card windows its list instead.
#:
#: Four rows is the smallest number that shows an exchange rather than a
#: fragment: a user block, its first line of reply, and the beginning of what
#: came next. Below that the card is better off windowing — it can say "3 of 9"
#: about its own list, and the transcript cannot say anything about what it is
#: cut off from.
#:
#: It is a FLOOR and not the target. On a terminal with room to spare the share
#: above is what governs, and it leaves considerably more; this only takes over
#: where the share alone would leave the conversation a sliver. Both are needed
#: because they bind at opposite ends of the size range — see
#: :meth:`AskPickerScreen._body_rows`.
MIN_TRANSCRIPT_ROWS = 4

#: The smallest body :meth:`AskPickerScreen._allocate` can say anything about
#: the QUESTION in: one option row, the windowing line that admits the rest of
#: the list is hidden, and the footer. Under it the plan collapses to the footer
#: alone, because one option row with no question above it and no count beside
#: it is an answer to nothing, while the keys are still how the user leaves.
#:
#: It is a THRESHOLD in the allocator, not a floor on the budget. As a floor it
#: changed no observable at all: the card went on laying out these three lines
#: into a body with room for one or two, and Textual clips the tail — so what a
#: five- or six-row terminal painted was an option row and no footer (round 3,
#: R9-R11, measured on the composited screen rather than the card's own text).
MIN_BODY_ROWS = 3

#: The cursor glyph, matching the ``/resume`` and command pickers. A caret plus
#: a tinted label rather than a reversed row: an inverted block reads as a
#: selection the user made rather than the position they are on.
CURSOR = "❯"

#: Cells the cursor gutter always occupies, so labels start at one column
#: whether or not the row is the current one.
GUTTER_CELLS = 2

#: Cells the ``1``..``9`` jump number occupies, including its trailing space.
#: Rows past nine get the same indent with no number: the digits are a shortcut,
#: and re-flowing the list at ten options to reclaim two cells would move every
#: row under the cursor. The free-text row is the exception — see
#: :data:`OTHER_JUMP_KEY`.
NUMBER_CELLS = 3

#: The glyph naming Tab in the footer, for the one question the composer
#: cannot answer (a multi-select). A glyph rather than the word, to keep the
#: parallel with ``↑↓`` and because the row it shares is the tightest on the
#: card; an experimental press is free, since Escape leaves.
TAB_HINT_KEY = "⇥"

#: The digit that always reaches the free-text row. ``0`` because it is the one
#: digit the ordinals never claim, and because the row it reaches is the row a
#: long list pushes past nine — where a blank gutter left the only answer that
#: is not on the list unreachable by any key (D13).
OTHER_JUMP_KEY = "0"

#: The key that reveals the selected row's description in full, as the footer
#: names it. ``^e`` and not ``enter``: Enter ANSWERS here, and on the approval
#: gate it authorises a tool call, so it is not available to overload. Not hover
#: either — the text this uncovers is what decides an authorisation, and gating
#: it behind a mouse loses it for a keyboard user, an ssh session with no mouse
#: reporting, and every screen reader.
#:
#: Free, audited against the running app rather than against the binding tables:
#: no node from this card up to the app resolves ``ctrl+e`` to anything, and
#: :meth:`on_key` only swallows PRINTABLE single characters, which ``\x05`` is
#: not — so the free-text row does not eat it either.
REVEAL_HINT_KEY = "^e"

#: The multi-select checkbox, including its trailing space. ``[x]``/``[ ]`` and
#: not a filled glyph pair: this app already says "done/not done" that way in
#: the todo panel, and a box reads as toggleable where ``◉`` reads as decoration.
CHECK_ON = "[x] "
CHECK_OFF = "[ ] "

#: The scrollbar track and thumb, painted over the rightmost column of the
#: windowed body rows so the user can see WHERE in a taller list the window is.
#: The same two glyphs and the same proportional maths as ``usage_panel.py``
#: (``SCROLLBAR_TRACK``/``SCROLLBAR_THUMB``, :meth:`_scrollbar_thumb`), so the
#: bar matches the rest of the app. The gutter is NOT reserved (unlike
#: usage_panel, whose right-aligned numbers must not slide when the bar
#: toggles): this card's rows are left-aligned labels with nothing at the right
#: edge to keep stable, and a persistent column would shift the approval gate's
#: byte-identical frame one column. So the thumb is painted only while the list
#: windows — keyed on ``layout.show_position`` (see the paint site). The list
#: windows when the visual line list is taller than ``body_line_budget``, and
#: at tight heights the D1 collapse drops the position row while the list is
#: still windowed; painting a thumb with no count beside it would split the one
#: overflow signal in two, so both hang off ``show_position``. When it windows,
#: the option rows are built at ``content_width`` (``width - 1``) so the glyph
#: has its own column — a state no pinned golden reaches, since the approval
#: gate never windows above ~44 columns (its labels are kept whole by
#: :meth:`_labels_must_all_fit`).
SCROLLBAR_TRACK = "│"
SCROLLBAR_THUMB = "█"

#: The free-text row's label while it holds nothing and is not selected.
OTHER_LABEL = "Other (type your own)"
#: Its label once it is selected or carries text; the typed string follows.
OTHER_PREFIX = "Other: "
#: The text caret drawn at the end of the field while the row is selected.
FIELD_CARET = "▌"
#: What the free-text row's second line says, so the row explains itself before
#: it is selected rather than after.
OTHER_HINT = "an answer that is not on the list — type it here"

#: A secret question has no options: the one row IS the paste field.
SECRET_LABEL = "Paste the value (hidden)"
SECRET_PREFIX = "Value: "
SECRET_HINT = "hidden as you type — enter stores, esc skips"
#: Same glyph the login key prompt uses, so a secret paste looks like one.
SECRET_MASK = "•"

#: The tag marking the option the model recommends.
#:
#: Uppercase behind a marker glyph, and drawn at ``fg`` + bold where it heads a
#: description (:meth:`_description_text`) — three signals that are not hue,
#: because hue is not available here. Contrast was measured
#: for every candidate against BOTH themes and both of this card's grounds
#: (``overlay`` for a normal row, ``raised`` for the selected one): amber
#: (3.97:1), label violet (3.62:1) and signal blue (3.54:1) all fall under the
#: 4.5:1 AA floor on the light theme, whose chromatic ramp is tuned against
#: ``surface`` rather than against a card. ``$lo-accent`` is unavailable for a
#: second reason — local_operator.tcss spends it on an exhaustive list of four
#: sites and site 4 is already on this card ("what ENTER will take"), so a green
#: badge would make the accent say two things on one frame.
#:
#: Words and not a bare glyph: this is the one row the user is being nudged
#: toward, and a nudge nobody can read is just an unexplained difference in
#: colour. It was `recommended` at ``muted`` for a release — the identical style
#: to the prose it sits in, with no weight — and the designer reported being
#: unable to find the word in the rendered frame without searching for it (D4).
RECOMMENDED_TAG = "▸ RECOMMENDED"

#: Cells a label must keep for its row to say anything. Below this the row is
#: more honest showing its number alone than a one-character stub, which names a
#: category and hides the instance — on this card that means two different
#: answers painting the same text.
LABEL_MIN_CELLS = 6

#: The most description lines ``ctrl+e`` lifts the SELECTED row's cap to,
#: however much room the terminal has (:meth:`_cap_for_row`).
#:
#: Under line-granular windowing the reveal is a per-row cap LIFT inside the one
#: scrolling viewport, not a constant-height block (§4). ``ctrl+e`` raises the
#: selected row's description cap from :data:`DEFAULT_DESC_CAP` (2) to this,
#: making that one row up to 9 visual lines tall (a label plus 8 description
#: lines) in place; :meth:`_move_to` then scrolls the list to keep the taller
#: row's span visible, and the thumb reports the rest honestly. There is no
#: block to pad, no tallest-in-list reservation, and no BLOCKER-1 column trade
#: — lifting one row's cap never removes another row's prose, it only pushes
#: other rows further out of the viewport.
#:
#: A DRAWING cap, applied to the line list, and deliberately not a cut on the
#: wrap itself: :meth:`_description_lines` still returns the whole wrap, so the
#: last kept line is ``…``-marked wherever the wrap exceeds the cap, exactly the
#: "say that it continues" discipline every capped row follows. This is why D5
#: (a reveal that truncated in silence) cannot recur: the cut is a decision
#: about the FRAME, computed against the full wrap.
#:
#: Eight, and the number is the reported size's: at 150x40 the longest
#: description on ``scripts/ask_user_repro.py`` (option 1, 1023 characters)
#: wraps to exactly 8 lines, so an 8-line cap shows that whole consequence in
#: place. Above 8 buys nothing at any size the card was verified at — the
#: viewport binds first — and a taller row only pushes more of the list off
#: screen, which the thumb and the position row already report.
#:
#: Widths here are TERMINAL columns, and the card is 4 cells narrower (padding
#: one cell each side, plus the dock's own two): terminal 100 is card width 96,
#: and the prose column is 5 narrower again (:meth:`_description_indent`). That
#: conversion is stated because getting it backwards is what produced the "44
#: columns" figure this file carried for a release — see :data:`DEFAULT_DESC_CAP`.
REVEAL_MAX_ROWS = 8

#: The most description lines the DEFAULT list draws for any one row.
#:
#: A different number for a different job than :data:`REVEAL_MAX_ROWS`: 2 is
#: how much prose the LIST shows per row, 8 is how much the REVEAL shows for
#: one row. Uncapped, the pool spent every spare row of a roomy terminal on prose
#: and the card stopped being a list: measured on ``scripts/ask_user_repro.py``
#: (three described options plus the free-text row) at 190x50 under the real
#: ``OperatorApp``, an 18-row budget of which 14 were prose, options drawn
#: 7/5/4/2 rows tall with no blank line between one option's paragraph and the
#: next option's label. The labels are ``fg`` bold against ``muted`` prose, so
#: the contrast ranking was intact and the list was still uncountable — a bold
#: line every six lines does not separate anything at that density.
#:
#: Two, not three. The label/prose pair still reads as one unit at 2 lines and
#: stops being perceptible from 3. Measured on the same fixture and size, a
#: 3-cap draws rows 4/4/4/2 and a 24-row card against the 2-cap's 3/3/3/2 and
#: 17 — seven rows of the conversation spent to add a third line to prose the
#: reveal already reaches in full.
#:
#: The cap is what makes the reveal reachable at all. The two halves of the
#: previous round fought each other: the pool spent the budget step 7a needed,
#: so at 190x50 ``ctrl+e`` produced a byte-identical frame and the footer never
#: offered it, and at 150x40 all three descriptions were ellipsised with no way
#: to read them — the ORIGINAL truncation bug, at the size it was reported from
#: (D2). Fixing the hierarchy with SPACE is also the only move left: prose is
#: ``muted`` 6.51:1 and the next step down is ``dim`` 3.43:1, under the AA floor
#: the description was deliberately walked UP to, and on the approval gate that
#: text authorises a tool call.
#:
#: The APPROVAL gate is untouched by this cap, and the reason is a width the
#: file previously mis-stated. Its three consequences are 37, 36 and 28 cells,
#: so each wraps to one line — and therefore never asks for a second — down to a
#: prose column of 37, which is card width 42, which is TERMINAL width 46. Not
#: "44 columns": that figure was a card width transcribed as a terminal one, and
#: it is the number the gate's byte-identity argument used to rest on. Below 46
#: the consequences do wrap and the cap does apply to them; the first wrap going
#: down is at terminal 45 (card 41, column 36), where *Allow*'s takes two lines.
#: Measured through the same ``wrap_cells`` the card wraps with.
DEFAULT_DESC_CAP = 2


@dataclass
class _QuestionState:
    """One question's live answer, kept while the user moves between questions.

    Held per question rather than reset on advance so going back over a
    multi-question ask (or re-drawing after a resize) cannot silently lose a
    selection the user already made.
    """

    selected: int = 0
    checked: set[int] = field(default_factory=set)
    typed: str = ""
    #: Whether ``ctrl+e`` has traded this question's one-line-per-row list for
    #: the selected row's description in full.
    #:
    #: Per QUESTION and not per card, for the same reason the selection is: a
    #: multi-question ask that carried the mode forward would change how the
    #: NEXT question is drawn on the strength of a key pressed against the
    #: previous one, and coming back to a question would show it differently
    #: from how it was left.
    revealed: bool = False


@dataclass(frozen=True)
class _CardLayout:
    """How one paint divides the card's rows, decided before anything is drawn.

    Every renderer below reads this rather than measuring the screen for itself.
    The header, the question, the spacers, the window and the footer each used
    to decide independently, and their sum came to more lines than the region
    had — so the card drew chrome it could not show and Textual clipped whatever
    was laid out last (D1). One division, one budget, one answer.
    """

    #: Content cells the card is drawing in, padding excluded.
    width: int
    #: Cells an OPTION ROW's content (label and description) may use — ``width``
    #: normally, ``width - 1`` when the list windows so the row reserves the
    #: scrollbar column. Descriptions wrap into ``content_width - indent`` and
    #: labels truncate to ``content_width``, so a windowed row is never cut by
    #: the ``width - 1`` thumb reservation in :meth:`_card_text` — that cut used
    #: to eat the tail of every full-width description line behind the thumb, and
    #: mark a spurious ``…`` on blank continuation lines (the R11/R15/D5 class,
    #: reintroduced by the line-granular wrap). This is OMP's own fix: when the
    #: list overflows it re-renders one column narrower (``ask-dialog.ts:889-892``,
    #: ``renderRows(width - 1)``). CHROME (header rule, question, footer, the
    #: position row) is not under the thumb and keeps the full ``width``.
    content_width: int
    #: The question's wrapped lines that fit, the last marked ``…`` if any were
    #: cut off.
    question: tuple[str, ...]
    #: The title and its rule, which are shown or dropped together.
    show_title: bool
    space_above: bool
    space_below: bool
    #: Description lines each row contributes to the line list, by row index.
    #: Absent or 0 means the row is a bare label. Uniform at
    #: :data:`DEFAULT_DESC_CAP` for every row that has prose — except the
    #: SELECTED row while ``ctrl+e`` is on, lifted to :data:`REVEAL_MAX_ROWS`
    #: in place (§4). The VIEWPORT clips lines uniformly; there is no longer an
    #: all-or-nothing first-line decision (C5 retired, §3.4). ``page`` and
    #: ``reveal_rows`` are gone with the row window and the reveal block.
    description_rows: dict[int, int]
    #: First line index of every row in the OMP-style line list,
    #: ``lineStartByRow`` (``ask-dialog.ts:868-887``). Length ``row_count + 1``;
    #: the last entry is ``len(line_list)``. The sole source of truth for the
    #: cursor-visibility math (:meth:`_move_to`), the thumb's ``total`` (§5),
    #: the position row's range (§7) and the paging step (§2.6).
    line_start_by_row: tuple[int, ...]
    #: Visual LINES the option-list viewport draws. The line list is windowed
    #: into exactly this many lines — the viewport clips past its budget and
    #: pads blank past the list's end (§2.3), so the body is a rigid rectangle
    #: and the footer (bought first) is never clipped. This is C1 in line terms
    #: (§3.4): ``body_line_budget == remaining`` after step 8, so the plan's
    #: implied line count never exceeds ``budget``.
    body_line_budget: int
    #: Whether the windowing line and the thumb are drawn. Now
    #: ``len(line_list) > body_line_budget`` (OMP's ``#shouldRenderScrollbar``),
    #: the allocator's overflow decision. The renderer reads this instead of
    #: re-deriving it, which is how a row nobody had paid for got drawn and took
    #: the footer off the tail (round 3, R11).
    show_position: bool
    #: False only when the body has no drawable line at all. Everywhere else the
    #: footer is the first row bought, so it is the last thing that can go.
    show_footer: bool

    #: Whether the list carries an option description COLUMN at all: True
    #: whenever any OPTION has real prose (the free-text row's hint and the
    #: recommended badge do not count). Independent of the budget — descriptions
    #: now COEXIST with scroll rather than being dropped when the list windows
    #: (§3.4, C5 retired). Read by :meth:`_row_text` to place the recommendation
    #: badge: inline on the label when there is no column, in the column when
    #: there is.
    show_descriptions: bool


class AskPickerScreen(Container):
    """Put the ``ask`` tool's questions to the user; settle with the answers.

    Settles with ``question id -> chosen strings``, or ``None`` when nothing at
    all was answered. A PARTIAL mapping is deliberate: escaping out of the third
    question does not throw away the first two, because a user who answered and
    then stopped answering has still told the agent something, and the tool
    reports the rest as not answered.

    A ``Container`` mounted in the dock, not a ``ModalScreen``: see the module
    docstring. The name is kept — it is what the app, the tests and the harness
    already call this surface, and renaming a class to describe its parent
    widget rather than its job would churn every call site to say nothing new.

    It TAKES FOCUS, and that is what makes the keys work. The alternative is
    app-level bindings while the composer keeps focus, and that cannot work
    here: the answer keys are digits, letters and Space, and the composer is a
    text buffer that would swallow every one of them as input. Focus is handed
    back on settle (:meth:`restore_focus`), so answering never silently leaves
    the user somewhere other than the composer.
    """

    #: Focusable so the answer keys reach the card rather than the composer's
    #: buffer. Same rule the approval prompt follows, for the same reason.
    can_focus = True

    class DrawableChanged(Message):
        """The card started or stopped having anything to paint.

        Raised on the edge only, so the host is not told the same thing on
        every keystroke. The host reserves a separation row for the prompt and
        must give it back when the terminal gets too short for the card to draw
        at all — otherwise the dock keeps a row for a question painting nothing
        and pushes itself past the screen.
        """

        def __init__(self, card: "AskPickerScreen") -> None:
            super().__init__()
            self.card = card

    class QuestionAdvanced(Message):
        """The card moved to a NEW question within a multi-question ask.

        Raised whenever ``_index`` advances, by ANY route — the terminal's
        Enter (:meth:`action_accept`) and the phone's routed answer
        (:meth:`answer_current`) both post it. The host relays it to the mobile
        bridge so the phone re-projects the now-current question and follows the
        terminal to Q2..Qn (UX round 2, U8): without it, a terminal advance left
        the phone showing the previous question and a phone tap there was
        recorded against the question the terminal had moved to.
        """

        def __init__(self, card: "AskPickerScreen") -> None:
            super().__init__()
            self.card = card

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "accept", "Answer", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        # j/k as well as the arrows: this card has no filter to type into, so
        # the vi pair is free and is the movement a hand on the home row
        # reaches for. They are swallowed by :meth:`on_key` while the free-text
        # row is selected, where they are letters the user is typing.
        Binding("k", "move(-1)", "Up", show=False),
        Binding("j", "move(1)", "Down", show=False),
        Binding("ctrl+p", "move(-1)", "Up", show=False),
        Binding("ctrl+n", "move(1)", "Down", show=False),
        # PageUp/PageDown move the cursor a page and let `_move_to` autoscroll
        # the window to keep it drawn. CLAMPED, unlike the arrows: paging past
        # an end lands on the end rather than wrapping, so a page key can never
        # jump the cursor to the far end of the list.
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        # ``toggle_row``, not ``toggle``: ``DOMNode.action_toggle`` already
        # exists and takes an attribute name, so an ``action_toggle(self)``
        # here would override a live Textual action with an incompatible
        # signature — the shadowing class that breaks a widget from the inside
        # and reports the traceback somewhere else entirely.
        Binding("space", "toggle_row", "Toggle", show=False),
        Binding("ctrl+e", "toggle_reveal", "Reveal", show=False),
        Binding("backspace", "backspace", "Edit answer", show=False),
        *[Binding(str(digit), f"jump({digit})", "Jump", show=False) for digit in range(1, 10)],
        # ``0`` is not an ordinal: it reaches the free-text row, which is the
        # row a list of ten or more pushes past the digits (D13).
        Binding(OTHER_JUMP_KEY, "jump_other", "Other", show=False),
    ]

    def __init__(
        self,
        questions: Sequence[AskQuestion],
        on_settle: Callable[[dict[str, list[str]] | None], None] | None = None,
        *,
        allow_free_text: bool = True,
        # Defaulted per INSTANCE rather than to a shared literal: two prompts
        # can be attached at once for the few pump hops between one settling
        # and the awaiting frame unmounting it, and a fixed id makes Textual
        # refuse the second mount outright (`DuplicateIds`). Callers that want
        # a stable handle pass their own.
        widget_id: str | None = None,
        title: str = "the agent needs your decision",
        exit_hint: tuple[str, str] = ("esc", "skip"),
    ) -> None:
        super().__init__(id=widget_id or f"ask-picker-{id(self):x}", classes="prompt-slot")
        #: Whether the list carries the trailing "Other (type your own)" row.
        #:
        #: A property of the SURFACE rather than of ``AskQuestion``, so the
        #: model-facing tool schema is unchanged: the ``ask`` tool's contract is
        #: that every question it asks accepts an answer nobody enumerated, and
        #: that stays true. What varies is the host — the approval prompt reuses
        #: this widget for a question with exactly three answers (allow, deny,
        #: allow all), where a free-text row would offer to send prose to a
        #: gate that can only return a boolean.
        self._allow_free_text = allow_free_text
        #: The card's title row. Varies by host because the two surfaces are
        #: asking different kinds of question: ``ask`` wants a decision the
        #: agent cannot make, and the approval gate wants permission to act.
        self._title = title
        #: The last hint in the footer, and the one the width ladder defends
        #: hardest — a card with no stated way out is unusable. Its WORD differs
        #: by host and the difference is load-bearing: "skip" is honest for a
        #: question the agent can proceed without, and a lie for an approval,
        #: where escaping denies the tool and stops the turn.
        self._exit_hint = exit_hint
        # Both the question and every label are MODEL-CONTROLLED and reach a
        # real terminal, so both are stripped on the way in — the discipline the
        # approval prompt and the tool cards already apply. A label carrying CSI
        # could erase the rows above it and repaint a forged question over them,
        # and would mis-measure every width budget below (``cell_len`` counts the
        # escape bytes) before being cut mid-sequence by truncation.
        self._questions = [_sanitize(question) for question in questions]
        self._index = 0
        self._answers: dict[str, list[str]] = {}
        self._states = [_QuestionState() for _ in self._questions]
        for state, question in zip(self._states, self._questions):
            # A recommendation is preselected as well as marked: the whole point
            # of recommending is that Enter alone should take it.
            if question.recommended is not None:
                state.selected = question.recommended
            elif question.secret:
                # The only row. Parking the cursor anywhere else would be a
                # destination that does not exist.
                state.selected = 0
        self._offset = 0
        self._hovered: int | None = None
        #: Body-relative line index -> the row it belongs to. Recorded while the
        #: card is built rather than recomputed as arithmetic, because rows are
        #: one OR two lines tall and the header's height depends on how the
        #: question wrapped; a click resolved by arithmetic landed on the row
        #: below whenever a description wrapped or the question did not.
        self._line_rows: list[int | None] = []
        #: ``(row index, card width)`` -> that description's wrapped lines.
        #:
        #: ``_layout`` is called three times per paint and ``_repaint`` runs on
        #: every keystroke, so wrapping every description on the paint path is
        #: work the card does not need to repeat while nothing it depends on has
        #: moved. Width is part of the key, so a resize serves nothing stale;
        #: the question advancing is not visible in the key at all, so that path
        #: clears it (:meth:`_invalidate_description_wraps`).
        self._description_wraps: dict[tuple[int, int], list[str]] = {}
        #: Called once with the answers when the card settles. The app resolves
        #: the waiting tool call from it.
        self._on_settle = on_settle
        #: Guards the callback to EXACTLY ONE call. Several paths end this card
        #: — Enter on the last question, Escape, and the app tearing it down on
        #: a stop or an abort — and the tool call behind it is parked on a
        #: future that raises if it is resolved twice. The same idempotence the
        #: approval prompt's :meth:`resolve` has, for the same reason.
        self._settled = False
        #: What held focus when the card appeared, so answering hands it back
        #: rather than leaving the user out of the composer.
        self._restore_target: object | None = None
        #: The footer inputs as of the last paint, for :meth:`repaint_if_stale`.
        #: ``None`` until the first paint, which is never equal to a real
        #: fingerprint, so the first check always repaints.
        self._painted_fingerprint: tuple[object, ...] | None = None
        #: Whether the last paint produced any line at all. Starts True so the
        #: first undrawable paint is an EDGE and is announced: starting False
        #: would make "nothing to draw" the unremarkable case and leave the host
        #: holding a row nobody asked it to give back.
        self._drawable = True
        #: Set when Enter was pressed on a state that cannot answer, so the
        #: footer can say WHY rather than the card doing nothing at all (D4).
        #: Cleared by every key that changes the answer, so the complaint can
        #: never outlive the state it describes.
        self._rejected = False
        self._body: Static

    # -- state ---------------------------------------------------------------
    # ``visible_rows``/``_card_text``, not ``visible``/``_render``: both short
    # names are already Textual's (``Widget.visible``, ``Widget._render``) and
    # shadowing them breaks focus, layout or paint from inside the screen, with
    # a traceback that points somewhere else entirely.
    @property
    def question(self) -> AskQuestion:
        return self._questions[self._index]

    @property
    def state(self) -> _QuestionState:
        return self._states[self._index]

    @property
    def question_index(self) -> int:
        return self._index

    @property
    def row_count(self) -> int:
        """Options, plus the free-text row where this surface offers one.

        A secret question has no options: the one row IS the paste field, so
        the count is 1 regardless of ``_allow_free_text``.
        """
        if self.question.secret:
            return 1
        return len(self.question.options) + (1 if self._allow_free_text else 0)

    @property
    def other_row(self) -> int:
        """Index of the free-text row: always last, so its POSITION never moves.

        Its NUMBER does move, and past nine it has none of its own, which is
        why :data:`OTHER_JUMP_KEY` is bound to it separately.

        ``-1`` where the surface has no free-text row, which is deliberately an
        index no row can equal: every ``index == self.other_row`` test in this
        file then answers False without a second condition beside it, and the
        one place the value is used as a destination (:meth:`action_jump_other`)
        guards on the flag instead.

        A secret question's only row is this row: there is nothing else to
        select, and the typed value is the whole answer.
        """
        if self.question.secret:
            return 0
        return self.row_count - 1 if self._allow_free_text else -1

    @property
    def selected_index(self) -> int:
        return self.state.selected

    @property
    def typed_text(self) -> str:
        return self.state.typed

    @property
    def checked_indexes(self) -> list[int]:
        return sorted(self.state.checked)

    @property
    def visible_rows(self) -> list[str]:
        """The row labels the card is currently drawing, in order.

        Named for what it is — the DRAWN window, not the whole list — because on
        a short terminal those differ, and a test that asserted the full list
        would agree with a card that clipped half of it. Empty on a body with no
        room for a row at all, where the card is the footer or nothing.
        """
        window = self._window()
        return [self._row_label(index) for index in window]

    def answers_so_far(self) -> dict[str, list[str]] | None:
        """What has been answered up to now, for a host tearing the card down.

        The screen's own Escape path settles through ``dismiss``; this is for the
        app killing the picker from OUTSIDE it (a stop, a cancelled tool call,
        teardown), which has to answer the waiting tool call without going
        through the dismiss callback.
        """
        return dict(self._answers) or None

    # -- settling ------------------------------------------------------------
    def settle(self, answers: dict[str, list[str]] | None) -> None:
        """Hand ``answers`` to the waiting tool call, exactly once.

        Idempotent, because this card has several ends and they can race: the
        user's Enter on the last question, their Escape, and the app tearing the
        card down for a stop, an abort, or teardown. The tool call is parked on
        a future, and resolving that twice raises out of whichever path lost.
        """
        if self._settled:
            return
        self._settled = True
        callback = self._on_settle
        self._on_settle = None
        if callback is not None:
            callback(answers)

    @property
    def settled(self) -> bool:
        """Whether the answers have already been handed back."""
        return self._settled

    def restore_focus(self) -> None:
        """Return focus to whatever held it when the question appeared."""
        widget = self._restore_target
        self._restore_target = None
        if widget is not None and getattr(widget, "is_attached", False):
            widget.focus()  # type: ignore[attr-defined]

    # -- actions -------------------------------------------------------------
    def action_cancel(self) -> None:
        """Escape: stop answering. Whatever was already answered still counts."""
        self.settle(self._answers or None)

    def action_move(self, delta: int) -> None:
        """Arrow/vi movement WRAPS: a discrete, deliberate keypress."""
        self._move_to((self.state.selected + delta) % self.row_count)

    def action_page(self, delta: int) -> None:
        """PageUp/PageDown move the cursor a page and let the window follow.

        CLAMPED, not wrapped: paging past the end lands on the end, unlike
        :meth:`action_move` which wraps a discrete keypress. The step is
        ``max(1, rows_per_body - 1)`` ROWS — the line-model analogue of OMP's
        ``Math.max(1, bodyRows - 1)`` (``ask-dialog.ts:702-708``) — where
        ``rows_per_body`` is how many WHOLE option rows fit in one viewport
        height at the current line list (§2.6). Paging moves the CURSOR;
        :meth:`_move_to` then scrolls the viewport to follow, so a page still
        means "about a screenful" while the window stays line-granular.
        """
        layout = self._layout()
        step = max(1, self._rows_per_body(layout.line_start_by_row, layout.body_line_budget) - 1)
        target = self.state.selected + delta * step
        self._move_to(max(0, min(self.row_count - 1, target)))

    def action_jump(self, number: int) -> None:
        """``1``..``9`` jumps straight to a row, the free-text row included."""
        if 1 <= number <= self.row_count:
            self._move_to(number - 1)

    def action_jump_other(self) -> None:
        """``0`` reaches the free-text row from anywhere in the list.

        Its ordinal is past the digits on a list of ten or more, so without this
        the one row that can express an answer nobody enumerated is the one row
        no key reaches — on exactly the questions where scanning the list is
        hardest (D13).

        A no-op where the surface has no such row, rather than a jump to
        ``-1``: the key is bound unconditionally, and the approval prompt that
        reuses this widget has three answers and no fourth to reach.
        """
        if self._allow_free_text:
            self._move_to(self.other_row)

    def action_toggle_reveal(self) -> None:
        """``ctrl+e``: show the selected row's description in full, or stop.

        The list's one line per row is what the card can afford to draw for
        EVERY option; this trades it for one option's whole paragraph. An
        explicit, reversible keypress, so a card that windows its answers to
        buy the prose (measured at 130x30 and below) does so because the user
        asked, never on its own — and the count row it buys says how many
        answers went off screen.

        Reset here rather than in :meth:`_move_to`: the whole point of the
        reveal is to read one row and then the next, so movement RETARGETS the
        block instead of closing it. It stays the same height either way.

        Turning it ON is refused wherever the footer is not OFFERING it, which
        is a stronger rule than "would it show anything new" and is why it is
        asked of the footer rather than of :meth:`_reveal_is_useful` directly.
        The hint is dropped on a narrow card to keep `esc deny` whole, and a key
        that still fired there would be an unadvertised gesture that trades two
        of an authorisation's three consequences for one — measured on the
        approval card at 24-40 columns. The card must not do anything by a key
        it is not willing to name.

        Turning it OFF is always allowed: the terminal can grow under a revealed
        card until the list draws every description on its own, and a mode that
        could only be left while it was still needed would be a trap.
        """
        if not self.state.revealed and not self._offers_reveal():
            return
        self.state.revealed = not self.state.revealed
        # A new frame is a new state, so a refused Enter stops describing the
        # one before it — the same rule every other key here follows.
        self._rejected = False
        self._repaint()

    def action_toggle_row(self) -> None:
        """Space toggles a multi-select row; on a single-select it does nothing.

        A single-select question has exactly one answer, so a "toggle" there
        would either be Enter under another name or a selection the user cannot
        see the effect of.
        """
        if not self.question.multi or self.state.selected == self.other_row:
            return
        index = self.state.selected
        if index in self.state.checked:
            self.state.checked.discard(index)
        else:
            self.state.checked.add(index)
        self._rejected = False
        self._repaint()

    def action_backspace(self) -> None:
        if self.state.selected == self.other_row and self.state.typed:
            self.state.typed = self.state.typed[:-1]
            self._rejected = False
            self._repaint()

    def action_accept(self) -> None:
        """Enter: answer this question and move to the next, or dismiss.

        An empty answer is a NO-OP rather than a skip. Advancing on Enter with
        nothing chosen would record a question as asked-and-answered when the
        user had done nothing, and the model would then act on an answer nobody
        gave; Escape is how a question is left unanswered, and it says so in the
        footer.

        Refusing is only half of the job. The card NAMES this key in its footer,
        and a named key that does nothing and says nothing leaves the user
        pressing it again: the exported frame of the rejected press was
        BYTE-IDENTICAL to the frame before it, in two reachable states (D4). The
        refusal now answers in the footer's own row — see :meth:`_rejection`,
        which reverts as soon as there is something to take.
        """
        # Nothing is committed from a card that drew no options.
        #
        # The cursor still sits on a row — often the RECOMMENDED one, which is
        # preselected — so Enter on a collapsed card would take an answer the
        # user was never shown. The footer already refuses to advertise `enter`
        # there for exactly this reason; this makes the key agree with the hint
        # rather than merely going unadvertised (D9, design round 2).
        if not self.visible_rows:
            self._rejected = True
            self._repaint()
            return
        chosen = self._chosen()
        if not chosen:
            self._rejected = True
            self._repaint()
            return
        self._rejected = False
        self._answers[self.question.id] = chosen
        if self._index + 1 >= len(self._questions):
            self.settle(self._answers)
            return
        self._index += 1
        self._offset = 0
        self._hovered = None
        self._invalidate_description_wraps()
        self._repaint()
        # Tell the host the card moved to a new question so the phone follows
        # the terminal to it (U8). Posted AFTER _index advances so the handler
        # projects the now-current question, not the one just answered.
        self._notify_advanced()

    def answer_current(self, values: list[str]) -> bool:
        """Answer the CURRENT question with ``values`` and advance, or settle.

        The external-answer twin of :meth:`action_accept`: a phone answer routed
        in over the mobile bridge records this question's answer and moves the
        SAME picker to the next question, only settling after the last one. This
        is what makes a multi-question ask answerable question-by-question from
        the phone instead of the whole card resolving on the first answer and
        silently discarding the rest (UX round 1, U1).

        Returns True when this answer settled the card (it was the last
        question), False when the picker advanced and is now waiting on the next
        question — the caller re-projects the new current question on False.

        ``values`` is already the chosen text (labels for options, the typed
        string for free-text/secret); an empty list means "nothing chosen",
        which settles with None on the FIRST question (the harness's "user did
        not answer" signal) rather than recording an empty answer. On a later
        question an empty answer keeps whatever earlier questions collected, the
        same partial-report rule Escape follows.
        """
        # Guarded like action_accept: never commit an answer for a question the
        # card could not even draw.
        if self.settled:
            return True
        if not values:
            # Nothing chosen. On Q0 this is "the user answered nothing" — settle
            # with None so the tool falls back to its recommendation. Past Q0,
            # keep the partial map (Escape's rule) rather than throwing away the
            # answers already given.
            self.settle(self._answers or None)
            return True
        self._answers[self.question.id] = values
        if self._index + 1 >= len(self._questions):
            self.settle(self._answers)
            return True
        # Advance the live picker so a terminal user watching sees the next
        # question too, exactly as Enter would move it.
        self._index += 1
        self._offset = 0
        self._hovered = None
        self._rejected = False
        self._invalidate_description_wraps()
        self._repaint()
        # Same re-projection seam as the terminal Enter path: the phone that
        # DIDN'T drive this advance (or reconnects mid-ask) must still be
        # shown the current question (U8).
        self._notify_advanced()
        return False

    def _notify_advanced(self) -> None:
        """Announce that the card advanced to a new question.

        Posted from every ``_index`` advance so the host can re-project the
        current question to the phone regardless of which surface drove the
        move (U8). Guarded because a widget detached from the DOM (a card torn
        down as it settled) has no message pump; the advance still happened for
        the terminal, and a failed notify must never break it — the same
        best-effort contract the app's mobile-notify helpers keep.
        """
        try:
            self.post_message(self.QuestionAdvanced(self))
        except Exception:  # noqa: BLE001 - a detached card cannot post; harmless
            pass

    def _rejection(self) -> str:
        """What the footer says instead of the keys, or ``""`` to keep them.

        Derived from the state rather than stored as prose, so the complaint
        cannot contradict the card it sits on: the moment something is typed or
        ticked there is an answer, and the keys come back on the next paint.

        Which complaint depends on where the cursor is, not on the mode: a
        multi-select whose cursor is parked on the free-text row is answered by
        typing, and telling that user about Space would point at a key that is a
        letter while the field holds the cursor.
        """
        if not self._rejected or self._chosen():
            return ""
        if self.state.selected == self.other_row:
            return "type an answer first"
        return "nothing ticked — space toggles"

    def _chosen(self) -> list[str]:
        """This question's answer as text, or ``[]`` when there is none yet.

        The free-text row's content counts in BOTH modes and is appended last,
        so a multi-select answer reads in the order the card drew it.
        """
        state = self.state
        typed = state.typed.strip()
        if not self.question.multi:
            if state.selected == self.other_row:
                return [typed] if typed else []
            return [self.question.options[state.selected].label]
        labels = [
            self.question.options[index].label
            for index in sorted(state.checked)
            if index < len(self.question.options)
        ]
        if typed:
            labels.append(typed)
        return labels

    # -- keys ----------------------------------------------------------------
    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Printable keys type into the free-text row while it is selected.

        Handled here rather than as bindings because the field accepts every
        character; a binding per key would be a table of ninety-five entries
        that still missed the ninety-sixth. Textual dispatches the focused
        widget's handlers before its bindings, which is what lets this take
        ``j``, ``k``, the digits and Space back from the movement keys — on this
        row they are letters the user is typing, and a field that silently
        dropped every ``j`` would be worse than no field.

        Everywhere else the key is left alone: this card has no filter, its
        rows are numbered, and swallowing letters would cost the vi movement
        for nothing.
        """
        if self.state.selected != self.other_row:
            return
        char = event.character
        if char is not None and char.isprintable() and len(char) == 1:
            event.stop()
            event.prevent_default()
            self.state.typed += char
            self._rejected = False
            self._repaint()

    def on_paste(self, event) -> None:  # type: ignore[no-untyped-def]
        """A bracketed paste into the free-text (or secret) row.

        Pasting is the primary input for a secret question — a token arrives
        as one ``Paste`` event, not as keystrokes — and without this the
        value would land in the composer behind the card instead.
        """
        if self.state.selected != self.other_row:
            return
        text = getattr(event, "text", "") or ""
        if not text:
            return
        event.stop()
        event.prevent_default()
        # A paste routinely carries a trailing newline; keep it out of the
        # field so Enter is not fighting a leftover ``\\n``.
        self.state.typed += text.replace("\r", "").rstrip("\n")
        self._rejected = False
        self._repaint()

    # -- mouse ---------------------------------------------------------------
    # The wheel moves the cursor a row at a time, which scrolls the window with
    # it. CLAMPED, unlike the arrows: a scroll gesture that wrapped to the other
    # end of the list reads as the card resetting itself. Every handler stops
    # the event so one gesture does not also scroll the transcript behind.
    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._move_to(min(self.row_count - 1, self.state.selected + 1))

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._move_to(max(0, self.state.selected - 1))

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """A click on a row selects it, and on a single-select answers with it.

        Button 1 only, for the reason the ``/resume`` picker gives: this commits
        an answer the agent will act on, which is not something a right-click
        asking for a context menu or a stray middle-click paste should do.

        A multi-select click TOGGLES instead of answering — the gesture has to
        be repeatable there, and a click that confirmed the list would make the
        first pick the last one.

        Any button-1 click on the card also takes FOCUS back, before the row
        hit-test and whether or not it landed on a row. The card can lose focus
        without being answered — the user clicks into the composer to look
        something up while deciding — and without this the question sits there
        advertising keys with nothing to receive them. Clicking the thing you
        are being asked is the discoverable way back, and it is the affordance
        the approval prompt already offers.
        """
        if getattr(event, "button", 1) != 1:
            return
        if not self.has_focus:
            self.focus()
        index = self._index_at(event)
        if index is None:
            return
        event.stop()
        self._move_to(index)
        if self.question.multi:
            self.action_toggle_row()
        elif index != self.other_row:
            self.action_accept()

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            self._repaint()

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._hovered is not None:
            self._hovered = None
            self._repaint()

    def _index_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """Row under a mouse event, or ``None`` anywhere else on the screen.

        Three guards, all load-bearing, because a false positive here answers
        the agent's question on the user's behalf:

        - the point must be inside the BODY's region. The body now spans the
          card's full column, so the columns either side of a row are the card's
          one-cell padding rather than the wide empty band they were while the
          text was capped — but the guard is unchanged and still load-bearing:
          the padding ROWS above and below the body carry no row at all, and a
          click on one arrives with an ``x`` that looks perfectly valid. Region
          containment is what separates the card's ink from the container it is
          painted in, in both axes;
        - the line must map to a row in ``_line_rows``, which is recorded while
          painting, so the header, the blank spacers and the footer resolve to
          nothing rather than to whichever row the arithmetic lands on;
        - and the row must exist in THIS question, since the map is rebuilt per
          paint and a click can race an advance.
        """
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return None
        region = body.region
        if not region.contains(event.screen_x, event.screen_y):
            return None
        line = event.screen_y - region.y
        if not 0 <= line < len(self._line_rows):
            return None
        index = self._line_rows[line]
        if index is None or not 0 <= index < self.row_count:
            return None
        return index

    # -- geometry ------------------------------------------------------------
    def _screen_size(self) -> tuple[int, int]:
        """The SCREEN's content box, which is what the card budgets its ROWS against.

        Rows, and — only while :meth:`_card_width` has no laid-out column to
        read (before the card is placed, or while it is hidden) — the fallback
        width. The laid-out width does NOT come from here: it is
        imposed by the stylesheet's ``width: 1fr`` and read back off the card's
        own box. See :meth:`_card_width` for why the two axes differ.

        Deliberately not ``self.size``, which is the inverse of what the modal
        version did and is forced by the move into the dock. A docked container
        is sized BY its content: ``height: auto`` means ``self.size`` is
        whatever the card asked for on the last paint, so budgeting against it
        would be the card measuring its own shadow — it could never shrink,
        because every measurement would confirm the height it already had.

        And deliberately not ``self.app.size`` either, which is the raw
        TERMINAL. ``Screen { padding: 1 }`` insets the box the dock is actually
        laid out in, so the two differ by two rows: measured at an 80x12
        terminal the screen's content box is 78x10, and a card budgeting
        against 12 handed the dock two rows that did not exist. The dock came
        out 11 rows tall inside a 10-row screen, ``virtual_size`` exceeded
        ``size``, and the screen went scrollable with the top of the question
        pushed off the frame — the condition AGENTS.md calls always a bug here.
        Falls back to the app's own size only when the screen is not available,
        which is the pre-layout case where nothing is drawn yet anyway.
        """
        try:
            screen = self.screen if self.is_attached else None
            size = screen.size if screen is not None else self.app.size
        except Exception:  # pragma: no cover - only before the app has a screen
            return 80, 24
        if not size.width:  # pragma: no cover - pre-layout
            return 80, 24
        if not size.height:
            # A screen with ZERO content rows is a real state, not an
            # unmeasured one: `Screen { padding: 1 }` consumes two rows, so a
            # terminal two rows tall leaves nothing at all. Falling back to the
            # pre-layout default here (24 rows) was a defect — the card budgeted
            # against a screen four times the terminal, laid out its full
            # thirteen-line layout, and drove `virtual_size` to 20 rows on a
            # screen of 0. Zero rows in, zero rows out, and the card is not
            # drawn; the next resize brings it back.
            return max(1, size.width), 0
        # No floor on the HEIGHT. Reporting at least eight rows on a six-row
        # screen is the same mistake the row budget used to make one level up:
        # every caller then divides rows the region does not have, and the
        # difference comes off the bottom silently. At 20x8 that floor alone
        # clipped the windowing line and the whole footer back off a card that
        # had budgeted for both (D1). The allocator handles a two-row budget.
        return max(1, size.width), max(1, size.height)

    def _dock_reserved_rows(self) -> int:
        """Rows the rest of the dock needs below this card, measured live.

        The composer, the status band, and whatever the dock band is showing
        (subagent list, todos) all sit under the question, and the card may not
        push any of them off: the composer is where the answer to a free-text
        row is typed, and the status band carries the working line that says the
        turn is parked. Measured from the live widgets rather than assumed as a
        constant, because the band's height is genuinely variable — it is zero
        on an idle session and several rows with two subagents and a todo list —
        and a constant would be wrong in whichever direction the session was not
        in when it was written.

        Falls back to a conservative estimate when the dock has not been laid
        out yet (the first paint of the first question in a session), which
        errs toward a SHORTER card: budgeting rows the dock turns out to need
        is the failure that clips a footer, and budgeting too few only windows
        the list one row earlier for one frame.
        """
        screen = self.screen if self.is_attached else None
        if screen is None:
            return _DOCK_ROWS_FALLBACK
        # A host with no composer at all reserves nothing, and saying otherwise
        # is not conservative, it is wrong: the reduced hosts (the widget tests,
        # and any embedder mounting this card beside a transcript) have no dock,
        # so charging them for one shrinks the card by four rows it was given.
        # Distinguished from "the dock exists but has not been laid out yet" by
        # asking whether the composer is THERE, not how tall it currently is.
        if not screen.query("#input-shell"):
            return 0
        # The DOCK's own outer height, less whatever this card is currently
        # contributing to it, rather than the sum of its other children.
        #
        # Summing the siblings misses what the dock spends on itself: measured
        # at 30x12, `#prompt-host` (4) plus `#input-shell` (5) came to 9 while
        # `#input-dock` measured 10, and the missing row put the whole dock past
        # the screen — `virtual_size` 12 against `size` 10, permanently, because
        # the card kept re-deriving the same one-row-too-many budget. Reading
        # the container is also robust to a fourth child appearing later, which
        # a hardcoded list of siblings is not.
        dock = None
        try:
            dock = screen.query_one("#input-dock")
        except Exception:  # pragma: no cover - only before the dock is composed
            dock = None
        if dock is not None and dock.display and dock.outer_size.height:
            # This card's own contribution comes OUT of the reservation: what
            # is being computed is the room left FOR it, and counting the rows
            # it currently occupies would shrink the budget by whatever the card
            # already claimed — a feedback loop that can only ratchet downward.
            host = self.parent
            mine = 0
            if isinstance(host, Widget) and host.display:
                mine = host.outer_size.height
            return max(0, dock.outer_size.height - mine)
        reserved = 0
        seen = False
        for selector in ("#band", "#input-shell"):
            try:
                widget = screen.query_one(selector)
            except Exception:
                continue
            if not widget.display:
                continue
            # ``region.height``, NOT ``size.height``. Textual sizes border-box:
            # ``size`` is the CONTENT box, and the rows the dock actually
            # reserves are the outer ones. ``#input-shell`` carries
            # ``padding: 1``, so the two differ by two rows there — and reading
            # the content box let the card claim those two rows twice. Measured
            # at 80x12: the dock came out one row taller than the screen, its
            # region started at y=-1, and ``virtual_size`` (78, 12) exceeded
            # ``size`` (78, 10) — the scrollable screen AGENTS.md calls always a
            # bug on this app, with the top of the question scrolled off.
            height = widget.region.height
            if height:
                reserved += height
                seen = True
        return reserved if seen else _DOCK_ROWS_FALLBACK

    def _card_width(self) -> int:
        """Content cells the card may use: the COLUMN the dock laid it out in.

        This is ``self.size.width`` and not a budget re-derived from the screen,
        which is the opposite of what :meth:`_screen_size` does for ROWS — and
        the asymmetry is the point. Height here is content-driven (``height:
        auto``), so measuring it against the card's own box is measuring its own
        shadow. Width is not: the stylesheet pins the card at ``width: 1fr``, so
        its content box is IMPOSED by the dock and reading it back asks the
        layout engine what column this panel occupies rather than guessing.

        What it replaces was a modal-era cap — 74 cells, less a floating margin,
        measured against the terminal — and it survived the move into the dock
        as a number nothing re-derived. The slab had been full-width since that
        move, so the two disagreed and the gap grew with the terminal: measured
        with a question and four options up, the card's text stopped at 74 cells
        inside a 116-cell panel at 120 columns and inside a 156-cell panel at
        160, leaving a third of the card as blank fill and every row — the rule
        under the title, the selected row's tint, the footer — ending short of a
        composer that ran the full width directly beneath it. A floating modal's
        margin holds it off the terminal EDGE; docked, there is no edge there to
        hold off, only the panel the card is part of.

        Padding is not subtracted: Textual is border-box, so ``size`` is already
        the box inside this widget's ``padding``. ``ASK_PADDING_CELLS`` is still
        the stylesheet's mirror, and it is what the fallback below spends on the
        two paths that have no laid-out width to read:

        - before the card has been PLACED, because ``compose`` paints once on
          the way in; and
        - while it is HIDDEN, because a card that found no drawable line sets
          ``display = False`` (see :meth:`_repaint`) and an undisplayed widget
          reports a zero-width box. A terminal that shrinks past the card and
          then grows back re-measures through here (agent review round 2, F6).

        Both frames are replaced as soon as there is a real column to read, but
        by DIFFERENT events, and the difference is the whole reason
        :meth:`remeasure` exists: a placed card gets ``on_resize``, while a
        hidden one is not laid out and therefore receives no ``Resize`` at all —
        the app drives it from outside instead (round 3, F7). Measured, the
        fallback returns the same number the layout then assigns, because
        ``#prompt-host`` adds no horizontal padding.
        """
        mine = self.size.width if self.is_mounted else 0
        if mine:
            return max(1, mine)
        width, _ = self._screen_size()
        return max(1, width - ASK_PADDING_CELLS * 2)

    def _question_lines(self, width: int) -> list[str]:
        """The question, wrapped. Never truncated: it is what is being asked.

        Wrapping makes the header's height depend on content, which is why the
        row budget below is computed from this rather than from a constant.
        """
        return wrap_cells(self.question.question, width) or [""]

    def _description_indent(self) -> int:
        """Cells a description line is inset by, so it sits under the LABEL.

        One definition for the wrap and for the paint: measuring the room in
        one place and indenting by another is how a continuation line comes out
        one cell wider than the card it is drawn in.
        """
        return GUTTER_CELLS + NUMBER_CELLS + (cell_len(CHECK_ON) if self.question.multi else 0)

    def _description_lines(self, index: int, width: int) -> list[str]:
        """One row's description, wrapped into the card's own cell model, capped.

        ``wrap_cells`` and NOT a wrappable ``Text``: ``Content.from_rich_text``
        discards ``no_wrap``/``overflow`` when a ``Text`` crosses into a widget
        (command_picker.py:31-39, and the reason :func:`_cut_row` exists), so
        handing Textual a wrappable Text would let the card choose its own
        width — the one condition AGENTS.md calls always a bug here. Wrapping in
        the same width model the rest of the card measures in is what keeps
        every line inside the column :func:`_fit_row` then pads it to.

        The WHOLE wrap, uncapped, and memoised because ``_layout`` runs three
        times per paint and ``_repaint`` runs on every keystroke. Callers cap
        what they DRAW (:data:`DEFAULT_DESC_CAP` for the list,
        :data:`REVEAL_MAX_ROWS` for the block) and both then mark the cut,
        because both can still see the lines they are not showing.

        Capped here for a release, and that is precisely how the reveal came to
        stop mid-clause in silence (D5): a truncated return value made every
        caller's ``len(wrapped) > len(kept)`` test compare the cut against
        itself, so the card could not tell a description it had finished from
        one it had abandoned. A cut is a decision about a FRAME; this is the
        text.

        Returns the PROSE only. The recommendation tag is charged to the FIRST
        line, and this reserves its cells there by wrapping with a hanging
        indent — so the tag introduces the paragraph without narrowing the rest
        of it. Billed against every line instead, the same paragraph would wrap
        into a column fourteen cells short of the one it is drawn in, and the
        promoted option would be the one row on the card with a ragged edge.
        """
        cached = self._description_wraps.get((index, width))
        if cached is not None:
            return cached
        room = max(1, width - self._description_indent())
        description = self._row_description(index)
        tag_cells = 0
        if self.question.recommended == index:
            # The tag plus its ` · ` separator, exactly as `_description_text`
            # spends them, so what is reserved here is what is drawn there.
            tag_cells = cell_len(RECOMMENDED_TAG) + 3
            if not description or room - tag_cells <= 0:
                # No room beside the tag for prose, or no prose to put there.
                # The tag still earns its line: it is the only thing marking the
                # row the model is pointing at once the badge has moved off the
                # label (D6).
                self._description_wraps[(index, width)] = [""]
                return [""]
        if not description:
            self._description_wraps[(index, width)] = []
            return []
        if tag_cells:
            # The tag's line is wrapped NARROW and the rest wide, both through
            # `wrap_cells` so it keeps owning the word-breaking for over-long
            # words (URLs, paths) and no second wrapper drifts from it.
            #
            # Not a placeholder word wrapped in one pass, which is what this
            # was: a first token longer than `room - tag_cells` does not fit
            # beside a filler either, so `wrap_cells` put the filler on a line
            # of its own and slicing it off left line 0 EMPTY. At a grant of one
            # the row then drew `recommended` alone and lost every cell of its
            # prose — worse than the single truncated line the pre-wrap card
            # drew, and reached by exactly the descriptions a model writes
            # (measured at 70x22 on a description opening with a URL). The head
            # is a character prefix of the description, so the remainder is a
            # slice of it rather than a rejoin, and nothing is invented at the
            # seam.
            head = wrap_cells(description, room - tag_cells)[0]
            rest = description[len(head) :].lstrip(" ")
            lines = [head, *wrap_cells(rest, room)] if rest else [head]
        else:
            lines = wrap_cells(description, room)
        self._description_wraps[(index, width)] = lines
        return lines

    def _reveal_wrap(self, index: int, width: int) -> list[str]:
        """The lines ``ctrl+e`` may uncover for row ``index`` — none for a FIELD.

        The reveal exists for the prose that decides the answer: an option's
        consequence, which the model wrote and the user is comparing. The
        free-text and secret rows' "descriptions" are neither — they are chrome
        this app wrote to explain a paste box (``OTHER_HINT``, ``SECRET_HINT``),
        and a card that spent rows elaborating on `type it here` would be
        padding a text field. Measured on a two-option question with no
        descriptions at all: the free-text hint alone made the footer offer
        `^e` at 32 columns, so the key was advertised on a card whose answers
        carry no consequences.

        Blank wraps are dropped for the same reason. A row with no description
        still wraps to one EMPTY line where it is the recommended one, because
        the tag reserves its line there (:meth:`_description_lines`).
        """
        if index == self.other_row:
            return []
        wrap = self._description_lines(index, width)
        return wrap if any(wrap) else []

    def _invalidate_description_wraps(self) -> None:
        """Drop the wrap cache: its inputs (the question's text, the width) moved.

        Keyed by ``(index, width)``, so a resize alone cannot serve a stale
        entry — but the INDEX means a different row once the card advances to
        the next question, and that is the one input the key cannot see.
        """
        self._description_wraps.clear()

    def _body_rows(self, question_lines: int) -> int:
        """Lines the card's BODY may draw, with the transcript's share reserved.

        Three limits, and the card takes the smallest. Each one exists because
        the other two do not cover its case:

        1. **What is left of the terminal** once the rest of the dock (composer,
           status band, subagent/todo panels) and this card's own padding are
           paid for. This is a hard limit — exceeding it does not make a taller
           card, it makes a clipped one, and Textual clips the TAIL, which is
           the footer the allocator buys first precisely because it is the only
           statement of how to leave.
        2. **The anchoring share** (:data:`PROMPT_HEIGHT_SHARE`): the card is a
           band over a conversation, never a replacement for it. This is what
           binds on a tall terminal, where limit 1 would happily hand a
           twelve-option question the whole screen — which is the modal
           behaviour this rework exists to remove.
        3. **The transcript floor** (:data:`MIN_TRANSCRIPT_ROWS`): a share of a
           short terminal still rounds to nearly all of it, so below roughly
           fifteen rows the share stops protecting anything and this takes over.

        A CAP and not a fill: the body is ``height: auto``, so a card with
        little to say still draws short, and a two-option question does not
        reserve room for twelve.

        There is no floor under the result. Zero is a legitimate answer on a
        terminal with nothing to spare, and it is the honest one: a plan of no
        lines draws no card, where a floor would lay out lines the region cannot
        paint and let the compositor decide which to lose (round 3, R9-R11).
        :meth:`_allocate` spends this budget line by line and never overdraws
        it, so nothing is clipped rather than the wrong thing being.
        """
        _, height = self._screen_size()
        # 1. What physically remains for this card, inside its own padding.
        #
        # ``MIN_TRANSCRIPT_ROWS`` is NOT what is subtracted here — that is the
        # anchoring rule, applied below. This subtracts ONE row, because the
        # transcript is ``height: 1fr`` and a flexible child still claims a row:
        # the dock and the transcript are siblings under the screen, so a dock
        # sized to the whole screen leaves the transcript to overflow it. Left
        # out, the dock came to 11 rows on a 10-row screen and the screen went
        # scrollable, which took the top of the question off the frame.
        available = max(
            0, height - self._dock_reserved_rows() - CARD_PADDING_ROWS - _TRANSCRIPT_MIN_ROWS
        )
        # 2. and 3. Both anchoring rules are CEILINGS on the card, so the card
        # takes the tighter of the two rather than the looser: they bind at
        # opposite ends of the size range (the share on a tall terminal, the
        # transcript floor on a short one), and taking the looser would let
        # whichever is slack at this size cancel the one that is doing the work.
        #
        # The share is taken over the rows the card and the transcript actually
        # DIVIDE — the screen less the composer and the dock band — rather than
        # over the whole screen. Over the whole screen the dock's rows come out
        # of the transcript's side of the split alone: measured at 100x30, a
        # four-option question took 19 of 28 rows and left the conversation 4,
        # which is the sliver this rework exists to prevent. Over the divisible
        # rows the same question takes 16 and leaves 7.
        divisible = max(0, height - self._dock_reserved_rows())
        share = int(divisible * PROMPT_HEIGHT_SHARE) - CARD_PADDING_ROWS
        floor = height - MIN_TRANSCRIPT_ROWS - self._dock_reserved_rows() - CARD_PADDING_ROWS
        # The anchoring caps are ceilings on a card that has room to spare. They
        # must not become a gag: on a short terminal both go to zero or below,
        # and a question the agent is parked on that draws NO LINES AT ALL is a
        # worse outcome than a thin strip of conversation. Measured at 40x10
        # before this floor: the card was not drawn at all and the turn waited
        # on an answer with nothing on screen to give it. So the two caps yield
        # to :data:`MIN_BODY_ROWS` — the question, one option, and the footer.
        #
        # ``available`` is applied LAST and outside that floor, because it is
        # the only one of the three that is not a preference. The floor says
        # what the card would like to keep; ``available`` says what the screen
        # physically has, and a floor allowed to win over it is not a taller
        # card but a clipped one — measured at 80x12, where a 3-row floor over
        # 2 rows of room put the dock one row past the screen and made the
        # screen scrollable, cutting the top off the question.
        anchored = max(min(share, floor), MIN_BODY_ROWS)
        room = max(0, min(available, anchored))
        # Never taller than it needs to be: the caps above are ceilings, and a
        # card that padded itself out to its allowance would push the
        # conversation up to show blank rows.
        #
        # `wanted` is the card at its FULL natural height — title and rule, the
        # wrapped question, both spacer rows, every option with its description
        # line, and the footer — and getting that wrong in the other direction
        # is a real defect rather than a rounding difference. Computed as the
        # MINIMAL card (one line per option, no descriptions, no spacers) this
        # cap sat below what `_allocate` needed to buy the comforts, so the
        # allocator was handed a budget that could never afford a description
        # and the card silently lost every second line: measured as options
        # drawn with no consequences under them, and a list that windowed at 13
        # options on a terminal with room for all of them.
        # A description is now worth as many lines as it WRAPS to, capped, so
        # the natural height counts those lines rather than one per row. Left at
        # one, this cap sits below what `_allocate` can now spend and the
        # allocator is handed a budget that can never buy a continuation line:
        # the same failure recorded above, one step further in — every
        # description would silently stay a single ellipsised line however much
        # room the terminal had.
        # Every row's description at its cap, through the SAME line list the
        # allocator windows: :data:`DEFAULT_DESC_CAP` for each row, lifted to
        # :data:`REVEAL_MAX_ROWS` for the selected row while ``ctrl+e`` is on
        # (:meth:`_cap_for_row`). So the reveal is NOT a separate `wanted` term
        # any more — it is just the one row that is taller — which is the whole
        # simplification of the cap-lift model (§4): one line list, one height.
        #
        # Left counting one line per row, this cap sat below what `_allocate`
        # can spend and the card silently lost every description's second line
        # (measured: options drawn with no consequences under them, a list
        # windowing at 13 options on a terminal with room for all). Counting the
        # lifted cap for the selected row is what keeps the revealed frame's
        # natural height honest, so the key draws its extra lines on exactly the
        # terminals with room for them.
        line_list, _ = self._build_line_list(
            range(self.row_count),
            self._card_width(),
            revealed=self.state.revealed,
            selected=self.state.selected,
        )
        # ``len(line_list)`` is already every row's label PLUS its capped
        # description lines, so it stands in for both the per-row label row and
        # the ``described`` term the old arithmetic summed separately.
        wanted = (
            2  # title and its rule
            + question_lines
            + 2  # the spacer above the list and below it
            + len(line_list)  # every option's label and its capped description
            + 1  # the windowing line, where the list turns out to need one
            + 1  # the footer
        )
        return min(room, wanted)

    def _layout(self, *, reveal: bool | None = None) -> _CardLayout:
        """Divide the body's rows BEFORE anything is drawn into them.

        The old arithmetic reserved chrome and then handed the options
        ``max(1, budget - chrome)`` rows — a floor the region did not have. The
        card then laid out more lines than it could draw and Textual dropped
        whatever came last, which is the footer: at 100x14 the frame showed a
        question, one option of four and NO keys, and at 30x12 only the title
        and the question — zero options and nothing saying how to leave a card
        the turn is parked on (D1, round 1).

        The fix is an order, not a bigger floor. Nothing is drawn that was not
        paid for, and it is paid for in the order the card cannot do without it:

        1. the footer — the only statement of how to leave;
        2. the FIRST LINE OF THE QUESTION — what is being asked;
        3. one option row — a question with no answers is not a question;
        4. the windowing line, whenever the line list is taller than the
           viewport, because a card quietly showing one of four has hidden three;
        5. the rest of the question, every wrapped line of it, marked ``…`` if
           even that cannot fit;
        6. the title and its rule, which travel together — a rule under a title
           is a caption, a rule under nothing is the edge of a box;
        7. the rest of the option rows;
        8. the blank spacers, which are rhythm and nothing else;
        9. the option list is a fixed-height LINE VIEWPORT of whatever
           ``remaining`` is left. Every row carries its 2-line-clamped
           description (the selected row lifted to :data:`REVEAL_MAX_ROWS` while
           ``ctrl+e`` is on) into a line list, and the viewport windows it,
           clipping partial rows at the edges (§2.3). Descriptions and scroll
           COEXIST — there is no all-or-nothing column decision (C5 retired) and
           no separate reveal block (§4): the reveal is a per-row cap lift inside
           this one viewport, so there is nothing to fight it.

        The reveal is no longer a step: it is the cap lift step 9 already reads.
        The old steps 7a (the constant-height block) and 10/11 (continuation
        lines bought from a leftover pool) are gone with it — the viewport draws
        exactly ``min(body_line_budget, len(line_list))`` lines, so wrapping is
        free at the tight sizes (the list simply windows) without a pool to
        drain.

        **The question outranks the options, and that ordering is a safety
        property rather than a preference.** It used to sit below them, which
        was defensible while this card only ever asked the ``ask`` tool's
        questions: an option row at least says what one of the answers is. It
        stopped being defensible when the approval gate started using this same
        surface. Measured at 60x16 before this change, the approval card
        rendered exactly three lines:

            ❯ 1. Allow
            showing 1–1 of 3
            ↑↓ move · 1-9 jump · enter answer · esc deny

        — an authorisation prompt for ``rm -rf /Users/damian/project/data``
        that never names the tool, the command, or the target, with the cursor
        parked on *Allow*. A user cannot consent to something the card declines
        to state, and "Allow" without its object is worse than no card at all,
        because it looks answerable. The question is now bought immediately
        after the exit, so the last thing to go before the card gives up is
        what it is asking about (D1, design round 1).

        A budget under :data:`MIN_BODY_ROWS` cannot buy even the first three, so
        the plan collapses to the footer alone — and at a budget of zero, which
        is every terminal four rows tall and under, to nothing at all. Laying
        out a card the screen cannot paint is how the keys went missing in the
        first place.

        ``reveal`` overrides the question's own ``ctrl+e`` state, for asking
        what the OTHER state would draw. A parameter rather than a temporary
        write to ``_QuestionState``: this runs on the paint path, and a trial
        that mutates the state it is trialling leaves the card one exception
        away from being stuck in a mode nobody selected.
        """
        width = self._card_width()
        question = self._question_lines(width)
        budget = self._body_rows(len(question))
        revealed = self.state.revealed if reveal is None else reveal
        plan = self._allocate(width, question, budget, position=False, reveal=revealed)
        # Would the line list overflow the first pass's viewport? The first pass
        # never buys the position row (``position=False`` forces ``show_position``
        # False), so ask the geometry directly: the list is taller than the body
        # AND the body draws at least one line (a card under MIN_BODY_ROWS draws
        # nothing to window).
        first_total = plan.line_start_by_row[-1] if plan.line_start_by_row else 0
        if plan.body_line_budget > 0 and first_total > plan.body_line_budget:
            # The list windows after all, so the line saying how much is hidden
            # has to be bought. This branch is the only place the row can be
            # bought, which is the only place the renderer will draw it from.
            windowed = self._allocate(width, question, budget, position=True, reveal=revealed)
            # ...unless paying for it costs the QUESTION. The count is a
            # refinement of the answers on offer; the question is what the card
            # is for. Measured at 60x16 with a 3-row budget: buying the count
            # took the last row the question had, leaving `❯ 1. Allow` over
            # `showing 1–1 of 3` — a card that says how many answers it is
            # hiding while hiding what the answers are TO (D1). Where the two
            # compete, the question wins and the list stays honest by other
            # means: the option rows it did draw are still numbered.
            #
            # No R11-inversion guard is needed any more. ``show_position`` is now
            # ``position and len(line_list) > body_line_budget``, and buying the
            # position row only SHRINKS ``body_line_budget`` (``position=True``
            # subtracts a row up front), so a list that overflowed the first
            # pass overflows the retry too — the retry can never claim to hide
            # answers it is drawing, which the old row-granular retry could when
            # a released title handed back more rows than the count cost.
            if windowed.question or not plan.question:
                plan = windowed
        return plan

    def _labels_must_all_fit(self) -> bool:
        """Whether every option LABEL must stay on screen, never windowed off.

        False here: the ``ask`` picker is a scannable list that SHOULD scroll
        its labels with descriptions kept (OMP-style coexist, design §0). The
        approval gate overrides it True — an authorisation prompt that hides
        *Allow all* behind a scroll is the C3/D1 safety defect this file's
        priority order exists to prevent (:class:`ApprovalPrompt`).
        """
        return False

    def _cap_for_row(self, index: int, revealed: bool, selected: int, cap: int) -> int:
        """Visual-line cap for row ``index``'s description in the line list.

        ``cap`` (normally :data:`DEFAULT_DESC_CAP`) for every row, LIFTED to
        :data:`REVEAL_MAX_ROWS` for the SELECTED row while ``ctrl+e`` is on.
        This is the ENTIRE reveal (§4): the selected row grows in place inside
        the one scrolling viewport rather than opening a competing block, so
        there is no second mechanism to fight the scroll (AGENTS.md:595). OMP
        has no analogue — its descriptions are always 2 lines — so the lift is
        ours, expressed as one number the line list reads.

        ``revealed``/``selected`` are passed rather than read off
        :attr:`state`, because :meth:`_allocate` trials the OTHER reveal state
        (``_layout(reveal=...)``) on the paint path and a cap read from the
        live state would size the trial's line list wrong. ``cap`` is lowered
        below the default only where :meth:`_labels_must_all_fit` forces
        descriptions down to keep every label visible.
        """
        if revealed and index == selected:
            return max(cap, REVEAL_MAX_ROWS)
        return cap

    def _build_line_list(
        self,
        rows: Iterable[int],
        width: int,
        *,
        revealed: bool,
        selected: int,
        cap: int = DEFAULT_DESC_CAP,
    ) -> tuple[list[tuple[int, str]], tuple[int, ...]]:
        """The OMP-style line list and its ``lineStartByRow`` map.

        Every row contributes, in order, its LABEL line (one — labels are
        truncated here, never wrapped, §2.5) then up to :meth:`_cap_for_row`
        description lines. This is ``renderRowLabel``'s output shape
        (``ask-dialog.ts:329-341``) in this card's cell model: a flat list of
        ``(row_index, kind)`` pairs whose length is the list's full visual
        height, plus ``line_start_by_row[i]`` = the first line index of row
        ``i`` (``ask-dialog.ts:872``, ``lineStartByRow.push(allLines.length)``
        before each row). The last map entry is ``len(line_list)``.

        The sole source of truth for the cursor-visibility math
        (:meth:`_move_to`), the thumb's ``total`` (§5) and the visible-row
        resolution. Cheap: :meth:`_description_lines` is memoised, so this is
        ``O(row_count)`` slicing over cached wraps.

        ``kind`` is ``"label"`` or ``"desc"`` \u2014 the paint reads it to draw a
        row line versus a description line, and a partial row at a viewport edge
        keeps whichever of its lines fall inside the window.
        """
        line_list: list[tuple[int, str]] = []
        line_start_by_row: list[int] = []
        for index in rows:
            line_start_by_row.append(len(line_list))
            line_list.append((index, "label"))
            row_cap = self._cap_for_row(index, revealed, selected, cap)
            desc = self._description_lines(index, width)
            for _ in desc[:row_cap]:
                line_list.append((index, "desc"))
        line_start_by_row.append(len(line_list))
        return line_list, tuple(line_start_by_row)

    def _granted_lines(
        self, index: int, width: int, revealed: bool, selected: int, cap: int = DEFAULT_DESC_CAP
    ) -> int:
        """Description lines row ``index`` draws: ``min(cap, wrap)``.

        The per-row grant under line-granular windowing — no budget starvation
        at the row level, the VIEWPORT clips (§2.3). Zero for a bare label.
        ``cap`` drops below the default only where :meth:`_labels_must_all_fit`
        forces descriptions down to keep every label visible.
        """
        row_cap = self._cap_for_row(index, revealed, selected, cap)
        return min(row_cap, len(self._description_lines(index, width)))

    @staticmethod
    def _rows_per_body(line_start_by_row: tuple[int, ...], body: int) -> int:
        """Rows whose WHOLE span fits in one ``body``-tall viewport from the top.

        OMP pages the offset by ``bodyRows-1`` (``ask-dialog.ts:702-708``); we
        page the CURSOR by the equivalent row count and let :meth:`_move_to`
        follow (§2.6). At least one, so a row taller than the body still steps.
        """
        if body <= 0 or len(line_start_by_row) < 2:
            return 1
        count = 0
        for i in range(len(line_start_by_row) - 1):
            if line_start_by_row[i + 1] - line_start_by_row[0] <= body:
                count = i + 1
            else:
                break
        return max(1, count)

    def _scroll_offset_for_cursor(
        self, offset: int, cur_start: int, cur_end: int, body: int, total: int
    ) -> int:
        """Line offset keeping the selected row's WHOLE span visible (§2.2).

        OMP's ``#scrollOffsetForCursor`` (``ask-dialog.ts:974-993``) for the
        non-manual case. If the row is off either end: pull its BOTTOM to the
        body's bottom when it fits, else pin its TOP so the label anchors the
        view (a row taller than the body shows its label and as many
        description lines as fit, the honest degradation, §4.3).
        """
        max_off = max(0, total - body)
        if max_off == 0:
            return 0
        span = cur_end - cur_start
        if cur_start < offset or cur_end > offset + body:
            offset = cur_end - body if span <= body else cur_start
        return max(0, min(offset, max_off))

    def _allocate(
        self,
        width: int,
        question: list[str],
        budget: int,
        *,
        position: bool,
        reveal: bool,
    ) -> _CardLayout:
        """One trial division of ``budget`` body rows, in the order above.

        Past the collapse below, every line the returned plan implies is bought
        from ``remaining`` and nothing is bought that ``remaining`` cannot pay
        for, so the plan is exactly ``budget`` less whatever is left of it —
        never more than the budget, whichever branches are taken. It could be
        more before: handed 1 or 2 rows this ran ``remaining`` negative, still
        returned ``page=1``, and the renderer drew three lines into a body with
        room for one (round 3, R11).
        """
        if budget < MIN_BODY_ROWS:
            # Too little for the question, one answer and the exit together.
            #
            # Two rows go to the QUESTION and the exit, not to an option row and
            # the exit. A bare `❯ 1. Allow` over `esc deny` is a prompt asking
            # for authorisation while refusing to say what it would authorise —
            # and it looks answerable, so the cursor sits on the permissive
            # option with nothing on screen to weigh it against (D1). Naming the
            # thing and stating how to leave is the honest minimum; the answers
            # come back the moment there is a third row to put them on.
            #
            # ONE row goes to the question, not to the exit.
            #
            # This is the same rule as the two-row case and it took a second
            # round to get right. A single `esc deny` row was reachable at every
            # width on a 13-row terminal, and it is the worst frame this card
            # can draw: it names nothing, and the answer letters still work — so
            # `y` approved `rm -rf /Users/x/project/data` from a card whose
            # entire content was the word for refusing (D9, design round 2).
            #
            # The exit is the one thing a user can always guess, and Escape is
            # the app's stop key everywhere regardless of what this footer says.
            # What they cannot guess is what they are being asked. So on a
            # single row the question wins and the footer goes.
            #
            # No rows at all is a card that cannot be drawn, and drawing it
            # anyway is the clip itself (round 4, R15).
            first = question[:1] if budget >= 1 else ()
            _, line_start_by_row = self._build_line_list(
                range(self.row_count), width, revealed=reveal, selected=self.state.selected
            )
            return _CardLayout(
                width=width,
                content_width=width,
                question=tuple(first),
                show_title=False,
                space_above=False,
                space_below=False,
                show_descriptions=False,
                description_rows={},
                show_position=False,
                show_footer=budget >= 2,
                line_start_by_row=line_start_by_row,
                body_line_budget=0,
            )
        # The footer, the first line of the question, and one option row: the
        # three lines the card cannot say anything useful without. The question
        # line is charged here rather than out of `remaining` below, which is
        # what puts it ahead of the option rows in the priority order.
        remaining = budget - 3
        if position:
            remaining -= 1
        # The first line is already paid for above; the rest competes at step 5.
        kept = list(question[: remaining + 1])
        if len(kept) < len(question) and kept:
            # Say that the question continues. A silently halved question is
            # the one clip on this card the reader cannot detect: every other
            # abbreviation leaves a count, a caret or an empty gutter behind.
            tail = truncate_cells(kept[-1], max(1, width - 2))
            # `truncate_cells` marks its OWN cut, and two ellipses in a row read
            # as a rendering fault rather than as "there is more question".
            kept[-1] = f"{tail[:-1].rstrip() if tail.endswith('…') else tail} …"
        # Less ONE, because the question's first line was already bought above
        # with the footer and the option row. Charging `len(kept)` here would
        # bill it twice and hand the rest of the plan a budget short by a row.
        remaining -= len(kept) - 1
        show_title = remaining >= 2
        if show_title:
            remaining -= 2
        extra = max(0, min(self.row_count - 1, remaining))
        remaining -= extra
        # 7a is GONE. The reveal is no longer a constant-height block bought
        # here out of the option rows' remainder; it is a per-row cap LIFT on
        # the SELECTED row inside the ONE line viewport (§4, :meth:`_cap_for_row`).
        # Deleting the block — the ~90-line BLOCKER-1 `affords_column` search,
        # the spacer/title yield, the `reveal_rows` reservation — is what
        # removes the "two mechanisms fighting for the viewport" bug by
        # construction (AGENTS.md:595): there is one viewport, and `_move_to`
        # owns it. `reveal` still reaches the plan through the line list the
        # lifted cap makes taller, so `_body_rows`' `wanted` still sizes the
        # card for a revealed row.
        #
        # 8. The blank spacers, which are rhythm and nothing else. No longer a
        # yield to the block (there is none): the ordinary ranking — a list
        # needs air around it — applies unconditionally, so every frame is
        # byte-identical to before this change wherever the reveal was off,
        # which is every frame the old `spacer_floor` also left alone.
        space_above = remaining - 1 >= 0
        if space_above:
            remaining -= 1
        space_below = remaining - 1 >= 0
        if space_below:
            remaining -= 1
        # 9. The option list is a fixed-height LINE VIEWPORT, not a row window.
        # ``body_line_budget`` is the label rows step 3+7 bought PLUS whatever
        # ``remaining`` the chrome left — the full height the viewport draws
        # into. This is C1 in line terms (§3.4): the viewport draws EXACTLY
        # ``min(body_line_budget, len(line_list))`` lines, clipping past its
        # budget and padding blank past the list's end (§2.3), so the plan's
        # implied line count never exceeds ``budget`` and the footer (bought
        # first, step 1) is never clipped.
        #
        # The line list is every row at its 2-line clamp (the selected row
        # lifted to REVEAL_MAX_ROWS while ``ctrl+e`` is on). Step 9 no longer
        # decides "descriptions: all or none" (C5 retired): every row that has
        # prose carries it and the VIEWPORT decides how many lines are visible.
        # This is where descriptions and scroll COEXIST — the headline change.
        body_line_budget = max(0, (1 + extra) + remaining)

        # The wrap width descriptions and labels are measured against. When the
        # list windows, an OPTION ROW reserves the scrollbar's column: the paint
        # cuts each windowed row to ``width - 1`` before appending the thumb, so
        # a description that wrapped to the full ``width`` would lose its tail
        # (and a padded short line would gain a spurious ``…``) exactly there —
        # the R11/R15/D5 truncation-behind-the-thumb class. OMP has the same
        # problem and the same fix: when the list overflows it re-renders one
        # column narrower (``ask-dialog.ts:889-892``, ``renderRows(width - 1)``).
        # So build at full width first; if it overflows, rebuild the option-row
        # content at ``width - 1`` so nothing is ever cut under the thumb.
        def build(content_width: int, cap: int) -> tuple[list[tuple[int, str]], tuple[int, ...]]:
            return self._build_line_list(
                range(self.row_count),
                content_width,
                revealed=reveal,
                selected=self.state.selected,
                cap=cap,
            )

        desc_cap = DEFAULT_DESC_CAP
        content_width = width
        line_list, line_start_by_row = build(content_width, desc_cap)
        # A surface may forbid hiding an option's LABEL behind the scroll — the
        # approval gate's authorisation contract, where a user who cannot see
        # that *Allow all* exists cannot weigh it (C3/D1, the same safety
        # property the question-outranks-options order protects). When the full
        # 2-line-clamp list would overflow, such a surface DROPS descriptions to
        # the largest uniform cap whose list still fits every label, rather than
        # windowing a label off. Only if even the labels-only (cap 0) list
        # overflows does it window — there is then no way to show every label at
        # once and the scroll is the honest degradation.
        #
        # The ask picker overrides this to keep OMP-style coexist: a long list
        # SHOULD scroll its labels, with descriptions kept and the thumb honest
        # (design §0, the headline). So the cap reduction runs only where
        # :meth:`_labels_must_all_fit` is True and is a no-op elsewhere.
        if self._labels_must_all_fit() and len(line_list) > body_line_budget:
            for candidate in range(DEFAULT_DESC_CAP - 1, -1, -1):
                desc_cap = candidate
                line_list, line_start_by_row = build(content_width, desc_cap)
                if len(line_list) <= body_line_budget:
                    break
        # OMP's ``#shouldRenderScrollbar`` (``scroll-view.ts:222-227``): the list
        # windows iff it is taller than the viewport. Only claimed on the
        # ``position=True`` retry — the first pass never buys the position row,
        # exactly as before, so the thumb and the count appear together (§5).
        show_position = position and len(line_list) > body_line_budget
        if show_position:
            # Reserve the thumb column: re-measure the option rows one cell
            # narrower so no windowed line reaches the cut. Narrowing can only
            # add wrapped lines, so a list that overflowed at ``width`` still
            # overflows at ``width - 1`` — the window decision does not flip.
            content_width = max(1, width - 1)
            line_list, line_start_by_row = build(content_width, desc_cap)
        # The description COLUMN exists whenever any OPTION carries real prose;
        # a property of the QUESTION, not of the budget (§3.4). The free-text
        # row's hint is not a description (:meth:`_reveal_wrap` returns nothing
        # for it), so it does not count. Read by the renderer to place the
        # recommendation badge inline vs in the column.
        descriptions = any(
            index != self.other_row
            and self._granted_lines(index, content_width, reveal, self.state.selected, desc_cap)
            >= 1
            for index in range(self.row_count)
        )
        # The per-row grants, uniform at the cap — what the paint draws under
        # each row. Not budget-starved: the viewport clips lines, not rows.
        grants = {
            index: g
            for index in range(self.row_count)
            if (
                g := self._granted_lines(
                    index, content_width, reveal, self.state.selected, desc_cap
                )
            )
            >= 1
        }
        return _CardLayout(
            width=width,
            content_width=content_width,
            question=tuple(kept),
            show_title=show_title,
            space_above=space_above,
            space_below=space_below,
            show_descriptions=descriptions,
            description_rows=grants,
            show_position=show_position,
            show_footer=True,
            line_start_by_row=line_start_by_row,
            body_line_budget=body_line_budget,
        )

    def _window(self, layout: "_CardLayout | None" = None) -> list[int]:
        """The row indexes at least partially drawn, after clamping the offset.

        Under line-granular windowing a row is "drawn" when any of its visual
        lines fall inside the viewport ``[_offset, _offset + body_line_budget)``
        — a partial row at an edge counts as visible (§2.3). Row-oriented
        readers (``visible_rows``, ``answer_keys``, the position row, the
        footer) keep reading this; the PAINT reads :meth:`_window_lines`.

        Not a pure reader: it clamps and writes back :attr:`_offset` in VISUAL
        LINES, the same idempotent side effect the row-granular ``_window`` had.
        """
        if layout is None:
            layout = self._layout()
        lsbr = layout.line_start_by_row
        body = layout.body_line_budget
        total = lsbr[-1] if lsbr else 0
        offset = max(0, min(self._offset, max(0, total - body)))
        self._offset = offset
        if body <= 0:
            return []
        lo, hi = offset, offset + body
        return [i for i in range(self.row_count) if lsbr[i] < hi and lsbr[i + 1] > lo]

    def _scrollbar_thumb(self, total: int, budget: int) -> tuple[int, int]:
        """``(thumb_top, thumb_len)`` inside a ``budget``-tall track.

        The standard proportional model, copied from
        ``usage_panel.py``:meth:`_scrollbar_thumb` so the two bars agree: the
        thumb is the fraction of the track the viewport is of the content
        (``budget / total``), and its top is that fraction of the free travel
        the offset is of its range. Here ``total`` is ``len(line_list)`` and
        ``budget`` is ``body_line_budget`` — both VISUAL LINE counts, which is
        OMP's ``#thumbRange`` model (``scroll-view.ts:229-238``, over
        ``totalRows``/``height``, its own visual-line counts) — and the offset
        is :attr:`_offset`, now in lines, whose range is ``total - budget``
        (the same clamp :meth:`_window_lines` applies). Guards a zero-travel
        range so a thumb as tall as the track never moves. The arithmetic never
        cared what the units were; only the callers moved rows→lines.
        """
        track = max(1, budget)
        thumb = max(1, min(track, round(track * budget / total))) if total > 0 else track
        span = track - thumb
        max_off = max(0, total - budget)
        top = round(span * self._offset / max_off) if (span > 0 and max_off > 0) else 0
        return max(0, min(span, top)), thumb

    # -- internals -----------------------------------------------------------
    def _move_to(self, index: int) -> None:
        self.state.selected = max(0, min(self.row_count - 1, index))
        # Any movement is a new state, so a refused Enter stops describing it.
        self._rejected = False
        # Scroll just far enough to keep the SELECTED row's whole line-span
        # visible (§2.2), so a row's description is never clipped under the
        # cursor. A body of zero draws nothing (a card under MIN_BODY_ROWS), and
        # there is no keeping a cursor drawn on a list that is not: leave the
        # offset for the terminal to grow back into.
        layout = self._layout()
        lsbr = layout.line_start_by_row
        body = layout.body_line_budget
        if not lsbr or body <= 0:
            self._repaint()
            return
        total = lsbr[-1]
        cur_start = lsbr[self.state.selected]
        cur_end = lsbr[self.state.selected + 1]
        self._offset = self._scroll_offset_for_cursor(self._offset, cur_start, cur_end, body, total)
        self._repaint()

    # -- rendering -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(classes="ask-picker"):
            self._body = Static(self._card_text(), id="ask-picker-body")
            yield self._body

    def on_mount(self) -> None:
        """Take focus — unless the user is in the middle of typing.

        The card takes focus by default so its full keymap (arrows, Enter, the
        digits) works without a click, and remembers what held it so answering
        hands it back.

        It yields to a NON-EMPTY composer, because a question is raised by the
        AGENT and can land at any moment, including mid-sentence. Taking the
        caret then is not merely rude: the answer keys are ordinary characters,
        so the rest of the user's sentence starts landing on the card, and on an
        approval the first `y` AUTHORISES the call. Measured before this guard:
        typing `please ` and then `yes do it` through the mount approved
        `rm -rf /Users/x/project/data` and left `please es do it` in the buffer
        (D12, design round 3).

        Nothing is lost by yielding. The keys the card advertises are ROUTED
        from the composer (`OperatorApp.route_key_to_live_prompt`), the footer
        names only the ones that work from there, and clicking the card takes
        focus deliberately.
        """
        screen = self.screen
        self._restore_target = screen.focused if screen is not None else None
        if not self._composer_has_draft():
            self.focus()
        self._repaint()

    def _composer_has_draft(self) -> bool:
        """Whether the user has text in the composer right now.

        Defensive on purpose: this decides whether to take the caret, and a
        host with no composer (the reduced test harnesses) must fall through to
        the ordinary "take focus" path rather than raise out of a mount.
        """
        from local_operator.tui.widgets.editor import Editor

        # `self.screen` RAISES `NoScreen` on an unmounted node rather than
        # returning None, and this is reached from the footer, which is
        # rendered in contexts where the card is not attached (the width tests
        # build a card and ask it to lay out). Guarded so a layout question can
        # never raise out of a render.
        try:
            screen = self.screen
        except Exception:
            return False
        if screen is None:
            return False
        try:
            return bool(screen.query_one(Editor).text)
        except Exception:
            return False

    def on_focus(self, event) -> None:  # type: ignore[no-untyped-def]
        """Repaint: the footer describes a different keyboard on each side.

        Deferred like the blur, and for the mirror reason: the widget is told
        it is GAINING focus before `has_focus` reports it.

        `_footer_hints` branches on `has_focus`, and `has_focus` is not a
        reactive on this widget — nothing schedules a refresh when it changes.
        Without these two handlers the branch was correct in the model and
        INVISIBLE on screen: the card went on painting whatever `_repaint` last
        pushed into its body, so with the caret in the composer it still
        advertised `↑↓ move` and `enter answer`, both dead (D13, design round
        4). The test that pinned the fix read `render_lines_for_test`, which
        re-derives the text and therefore could not see the staleness — it was
        measuring the intent, not the pixels.
        """
        self.call_after_refresh(self._repaint)

    def on_blur(self, event) -> None:  # type: ignore[no-untyped-def]
        """Repaint for the same reason, in the other direction.

        Deferred by one frame: on the blur event `has_focus` is still True (the
        widget is told it is LOSING focus, not that it has lost it), so a
        repaint here re-derives the focused footer and paints exactly the stale
        text this handler exists to replace.
        """
        self.call_after_refresh(self._repaint)

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: every quantity the card lays out against has just moved.

        The width comes from the COLUMN the dock assigned this card
        (:meth:`_card_width`); the row budget, the page size and whether
        descriptions are affordable come from the screen (:meth:`_screen_size`).
        Two different sources, and this is the event that invalidates both.
        """
        self._move_to(self.state.selected)

    def remeasure(self) -> None:
        """Re-run the layout against the CURRENT screen, from outside.

        A hidden widget is not laid out, so it receives no ``Resize`` event —
        which means a card that hid itself on a terminal too short to draw it
        could never learn that the terminal had grown back. The question stayed
        invisible for the rest of the turn while the tool went on waiting: a
        shrink was a one-way door onto a permanently unanswerable prompt (D10,
        design round 2).

        The app calls this on every terminal resize instead, because the app
        still gets the event when this widget does not.
        """
        self._repaint()

    def footer_fingerprint(self) -> tuple[object, ...]:
        """Everything the FOOTER's content depends on, as one comparable value.

        The card's key hints are derived from state the card does not own:
        whether it holds focus, and whether the composer holds a draft. Neither
        emits anything this widget hears, so the same defect — model correct,
        screen stale — arrived three rounds running on three different inputs
        (D13 focus, F7/D14 the buffer, and D15's row set before it).

        Each was fixed by adding one more repaint trigger, which is a fix per
        input and leaves the next input to be discovered by a reviewer. This
        exists so the card can instead be ASKED whether what it is showing is
        still what it would draw — see :meth:`repaint_if_stale`, which the app
        calls on its 1 Hz tick. That makes a missed trigger a frame late rather
        than permanently wrong.

        Not a pure reader: ``_window()`` clamps and writes back ``_offset``.
        It is idempotent, so calling this never changes what the next paint
        produces — but it is worth knowing before using it anywhere the
        side effect would matter.
        """
        return (
            self.has_focus,
            self._composer_has_draft(),
            tuple(self._window()),
            self.question_index,
            # The refused-Enter complaint replaces the key hints entirely, and
            # it is cleared by any key that changes the answer — so it moves
            # without the window or the question moving.
            self._rejected,
            # What the footer's ladder offers on a multi-select depends on
            # whether anything is ticked (`_chosen` feeds `_rejection`), and on
            # a free-text row on what has been typed into it.
            self.state.selected,
            tuple(sorted(self.state.checked)),
            self.state.typed,
            # `^e more` becomes `^e less` and back, and the hint disappears
            # entirely once the reveal has drawn everything there was.
            self.state.revealed,
        )

    def repaint_if_stale(self) -> None:
        """Repaint when the footer's inputs have moved since the last paint.

        The backstop for the class of defect described in
        :meth:`footer_fingerprint`. Cheap: it compares a small tuple and does
        nothing in the overwhelming majority of ticks.
        """
        fingerprint = self.footer_fingerprint()
        if fingerprint != self._painted_fingerprint:
            self._repaint()

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        self._painted_fingerprint = self.footer_fingerprint()
        text = self._card_text()
        body.update(text)
        drawable = bool(text.plain)
        card = body.parent
        if card is not None:
            # A card with no drawable line is not a smaller card: its padding is
            # rows the screen has not got, so at three terminal rows and under
            # the container pushed the screen's virtual height past its size — a
            # scrollable screen, which AGENTS.md calls always a bug here, over a
            # card painting nothing. Nothing to draw, nothing laid out; the next
            # resize brings it back.
            card.display = drawable
        # And the same for THIS widget, which is the one that carries the
        # padding now that the card is docked rather than modal. Hiding only the
        # inner holder left the outer one claiming its own two padding rows
        # around zero rows of content: measured at 40x10, that was two rows of
        # pure chrome for a card drawing nothing, and `virtual_size` (38, 9)
        # over `size` (38, 8) — a scrollable screen, which AGENTS.md calls
        # always a bug here.
        #
        # The HOST's own row is not this widget's to write, and an earlier
        # revision that wrote it directly was a bug: the app shows the host when
        # it mounts a prompt, so two owners disagreed and which won depended on
        # ordering. The card ANNOUNCES instead, and the app (the single writer)
        # decides. A message rather than a direct call because the answer
        # changes on any repaint — a resize, a question advancing, a typed
        # character rewrapping the card — and each of those has to be able to
        # bring the host's row back as well as take it away.
        was_drawable = self._drawable
        self._drawable = drawable
        self.display = drawable
        if was_drawable != drawable and self.is_attached:
            self.post_message(self.DrawableChanged(self))

    def answer_keys(self) -> frozenset[str]:
        """Keys that answer this card directly from the COMPOSER.

        Deliberately narrow. The card holds focus in the ordinary case and its
        full keymap works there; this is only the set worth intercepting while
        the caret is in the composer, where every character is otherwise the
        user's text. On the ``ask`` picker that is the row ordinals — a digit
        with an empty composer is unambiguous — and nothing else: Enter is left
        to the composer because it SUBMITS a prompt there, and Escape is the
        app's stop key.

        The FREE-TEXT row is excluded, because a digit cannot answer it: it is
        answered by typing into it, which needs the card to hold the caret. Left
        in, the footer advertised `1-3 answer` on a three-row card whose third
        row was `Other` — and pressing `3` selected it and then refused, since
        there is nothing typed to accept. A hint that names a key which lands on
        a dead end is the same defect as one that names a key that does nothing.

        Empty on a MULTI-SELECT, which no single key can answer: it is answered
        by ticking rows with Space and confirming with Enter, and both of those
        belong to the composer while the caret is there. Advertised as
        `1-2 answer`, a digit only moved the cursor and left the question
        unanswered with `nothing ticked — space toggles` (D15b, design round 4).

        Empty while the card is drawing no rows, so a key can never commit an
        answer the user was not shown (the rule :meth:`action_accept` follows).
        """
        if self.question.multi or not self.visible_rows:
            return frozenset()
        return frozenset(
            str(index + 1) for index in self._window() if index < 9 and index != self.other_row
        )

    def routed_hint(self) -> tuple[str, str] | None:
        """How the footer names the keys that still work from the composer.

        A range for the ordinals (`1-3 answer`), because they are contiguous
        and a list of them would cost more width than it explains.
        """
        digits = sorted(key for key in self.answer_keys() if key.isdigit())
        if not digits:
            return None
        span = digits[0] if len(digits) == 1 else f"{digits[0]}-{digits[-1]}"
        return (span, "answer")

    def answer_from_key(self, character: str) -> None:
        """Take ``character`` as an answer routed from the composer."""
        if character.isdigit():
            self.action_jump(int(character))
            self.action_accept()

    @property
    def is_drawable(self) -> bool:
        """Whether the card has any line to paint at this terminal size.

        False on a terminal too short to show even the footer, where the card
        hides itself rather than laying out rows the screen cannot draw. The
        host reads this to decide whether to keep its own separation row, so
        that row cannot outlive the card it separates.
        """
        return self._drawable

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads.

        Empty when the body has no drawable line at all, because ``Text``
        splits an empty card into ONE empty line and a caller counting that
        against the room the screen has would be told the card overflows a
        terminal it is painting nothing into.

        Also empty while the card is HIDDEN. This method re-derives the text
        rather than reading back what was painted, so on a terminal too short
        to draw the card it would otherwise report the line the card WOULD
        have drawn — and a caller then asserts that line reached the terminal,
        which it never did. Measured as an intermittent failure at 30x12 where
        the card reported `esc skip` against a frame that contained only the
        composer. Whether the card is drawn is `display`'s answer, so this
        defers to it rather than keeping a second opinion.
        """
        if self.is_mounted and not self.display:
            return []
        text = self._card_text()
        if not text.plain:
            return []
        return [line.plain for line in text.split("\n")]

    def _row_label(self, index: int) -> str:
        """One row's label text, free-text row included, without styling."""
        if index != self.other_row:
            return self.question.options[index].label
        typed = self.state.typed
        if self.question.secret:
            if typed or self.state.selected == self.other_row:
                # Masked: a secret that paints into the transcript is the
                # failure this surface exists to prevent. The length is the
                # one property a paste needs to check (did it arrive whole).
                return f"{SECRET_PREFIX}{SECRET_MASK * len(typed)}"
            return SECRET_LABEL
        if typed or self.state.selected == self.other_row:
            return f"{OTHER_PREFIX}{typed}"
        return OTHER_LABEL

    def _row_description(self, index: int) -> str:
        if index == self.other_row:
            return SECRET_HINT if self.question.secret else OTHER_HINT
        return self.question.options[index].description

    def _card_text(self) -> Text:
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        edge = Style(color=theme_mod.semantic_color("edge"))
        layout = self._layout()
        width = layout.width

        out = Text()
        lines: list[int | None] = []

        def newline(row: int | None) -> None:
            """Start a line that belongs to ``row`` (``None`` = chrome)."""
            if lines:
                out.append("\n")
            lines.append(row)

        if layout.show_title:
            newline(None)
            out.append_text(self._header(width, fg, muted))
            newline(None)
            # The raised card ground needs the raised hairline: `edge` is tuned
            # against the app background and nearly vanishes on `overlay`.
            out.append("─" * width, style=faint)
        for line in layout.question:
            newline(None)
            out.append(line, style=fg)
        if layout.space_above:
            newline(None)

        # The option list is a fixed-height LINE VIEWPORT. Every row is rendered
        # once into (row_index, Text) visual lines — its label line, then its
        # granted description lines — into ``rendered``, the paint-side twin of
        # :meth:`_build_line_list`. The viewport then draws the ``_offset`` slice
        # of ``body_line_budget`` lines, clipping partial rows at the edges
        # (§2.3) so the body is a rigid rectangle. Descriptions and scroll now
        # COEXIST: a windowed list keeps every visible row's 2-line clamp.
        # OPTION-ROW content is measured against ``content_width`` — ``width``
        # normally, ``width - 1`` when the list windows so the row reserves the
        # thumb column (:attr:`_CardLayout.content_width`). Rows come out exactly
        # ``content_width`` wide, so the windowing cut below only pads and
        # appends the glyph; it never truncates real text or marks a spurious
        # ``…`` on a padded line (the R11/R15/D5 defect QA caught).
        cwidth = layout.content_width
        rendered: list[tuple[int, Text]] = []
        for index in range(self.row_count):
            ground = self._row_ground(index)
            rendered.append((index, self._row_text(index, cwidth, ground, fg, dim, faint, layout)))
            granted = layout.description_rows.get(index, 0)
            if granted:
                # `fg` for the TAG and `muted` for the separator and the prose.
                # Passed `muted` for both, the badge was the identical style to
                # the text beside it and had neither weight nor hue to win on
                # (D4); the tag carries the label's own ink and the prose keeps
                # the ramp step it was walked up to.
                for line in self._description_text(
                    index, cwidth, ground, fg, muted, granted, layout
                ):
                    rendered.append((index, line))
        body = layout.body_line_budget
        total = len(rendered)
        offset = max(0, min(self._offset, max(0, total - body)))
        self._offset = offset
        viewport = rendered[offset : offset + body]
        # The scrollbar thumb spans the full body height, one cell per VISUAL
        # LINE (not per option row), keyed on ``show_position`` — the
        # allocator's overflow decision (``len(line_list) > body_line_budget``).
        # Keyed there rather than on a raw comparison so the D1 collapse, which
        # drops the position row at a tight height to protect the question, also
        # drops the thumb: the two are one overflow signal in two renderings and
        # appear and vanish together (§5). ``total``/``body`` are line counts, so
        # the thumb reaches OMP byte-parity (``scroll-view.ts:229-238``).
        thumb_top, thumb_len = (
            self._scrollbar_thumb(total, body) if layout.show_position else (0, 0)
        )
        for line_pos, (index, row) in enumerate(viewport):
            # ``_line_rows`` maps every DRAWN body line back to the row it
            # belongs to (label OR description line), which the hit-test reads
            # (:meth:`_index_at`). A partial row's visible lines still map to it.
            newline(index)
            if layout.show_position:
                # Pad the ``content_width``-wide row to the thumb's column and
                # append the track/thumb glyph, so the bar never widens the
                # card. ``content_width`` already reserved this column (the row
                # wrapped to ``width - 1``), so the ``_cut_row`` here is a guard
                # against an over-long LABEL, not the description-truncating cut
                # it used to be. No gutter is reserved when the bar is absent
                # (unlike usage_panel, whose right-aligned numbers must not
                # slide): these rows are left-aligned, so the reservation would
                # buy nothing and would shift the approval gate's byte-identical
                # frame — which never windows at its pinned sizes and so never
                # reaches here.
                _cut_row(row, width - 1)
                pad = (width - 1) - cell_len(row.plain)
                if pad > 0:
                    row.append(" " * pad, style=self._row_ground(index))
                on_thumb = thumb_top <= line_pos < thumb_top + thumb_len
                row.append(
                    SCROLLBAR_THUMB if on_thumb else SCROLLBAR_TRACK,
                    style=self._row_ground(index) + (muted if on_thumb else edge),
                )
            out.append_text(row)
        if layout.space_below:
            newline(None)

        # The rows at least partially in the viewport — the position row's range
        # (§7) and the footer's "any option drawn" test read this, in OPTION
        # units, from the same offset the viewport used.
        window = sorted({idx for idx, _ in viewport})
        # Both rows are drawn only where the plan BOUGHT them. The position line
        # is gated on ``show_position`` (the overflow decision), never on
        # `len(window) < row_count` alone: at 1 or 2 rows the allocator never
        # paid for it, and drawing it regardless took the footer off the tail
        # (round 3, R11). The allocator decides; this only reads the decision.
        if layout.show_position:
            newline(None)
            out.append_text(self._position_row(width, window, muted, dim))
        if layout.show_footer:
            newline(None)
            out.append_text(self._footer_row(width, muted, dim, drawn=bool(window)))
        self._line_rows = lines
        return out

    def _position_row(self, width: int, window: list[int], muted: Style, dim: Style) -> Text:
        """``showing 2–3 of 6`` — how much of the list is not on screen.

        The range is still OPTION indexes, not line numbers (§7): "showing
        options 2–3 of 6" is what a user wants, not "showing lines 14–27 of 38".
        A partially-visible row counts as visible. Matches OMP's
        ``#clipIndicator`` (``ask-dialog.ts:995-1002``), which is also about
        content presence rather than line numbers.

        Numerals at `muted` and the grammar at `dim`, both a step up: at
        `faint`, `showing`/`of` measured 1.49:1 on this card's own ground, so
        the row that says how much is hidden was itself hidden (D5).

        The counts outlive the word: at 20 columns `showing 1–1 of 4` is wider
        than the whole card, and the card is exactly where the row matters most.
        `1–1 of 4` says the same thing in half the cells, so the word is what
        goes rather than the row.
        """
        span = f"{window[0] + 1}–{window[-1] + 1}"
        total = str(self.row_count)
        row = Text(no_wrap=True, overflow="ellipsis")
        if cell_len(f"showing {span} of {total}") <= width:
            row.append("showing ", style=dim)
        row.append(span, style=muted)
        row.append(" of ", style=dim)
        row.append(total, style=muted)
        return _cut_row(row, width)

    def _footer_row(self, width: int, muted: Style, dim: Style, *, drawn: bool = True) -> Text:
        """The key hints — or what the last refused Enter has to say instead.

        One row either way, so a refusal never moves the card under the cursor,
        and the keys come back the moment the state can answer.

        ``drawn`` is False on the collapsed card, where the footer is the only
        line and no option row is on screen. Then the exit is the ONLY honest
        hint: `enter` would commit a selection the user cannot see, and the
        digits would jump within a list that is not there. Measured on the
        previous revision at a 5-row terminal: `down down down enter` committed
        an option nobody had been shown (round 4, R14). This file already
        refuses to advertise the digits on the free-text row for the same
        reason — a key offered where it does not do what it says is a lie.
        """
        row = Text(no_wrap=True, overflow="ellipsis")
        rejection = self._rejection()
        if rejection:
            row.append(rejection, style=muted)
            return _cut_row(row, width)
        hints = self._footer_hints(width) if drawn else [self._exit_hint]
        for position, (key, what) in enumerate(hints):
            if position:
                row.append(" · ", style=dim)
            # Keys at `muted` (6.51:1) and their grammar at `dim` (3.43:1), each
            # a step up from `dim`/`faint`. The row that explains how to use and
            # how to leave the card was its least legible text, its words at
            # 1.49:1 on the `overlay` ground — the same failure this file
            # diagnosed one method down for the descriptions and left standing
            # here (D5).
            row.append(key, style=muted)
            if what:
                row.append(f" {what}", style=dim)
        return _cut_row(row, width)

    def _header(self, width: int, fg: Style, ink: Style) -> Text:
        """Title on the left, ``Question 1 of 2`` on the right when there is
        more than one.

        Worded and at ``muted``, not ``1/2`` at ``dim``: this is the only thing
        on the card saying that more questions follow, and right-aligned at
        3.43:1 it was the faintest text in the header — two consecutive frames
        of a two-question run differed in exactly one character (D10).

        The counter is dropped before the title when the card is narrow: it says
        how much is left, and the title says what the card IS.
        """
        header = Text(no_wrap=True, overflow="ellipsis")
        title = self._title
        counter = (
            f"Question {self._index + 1} of {len(self._questions)}"
            if len(self._questions) > 1
            else ""
        )
        gap = 2
        if counter and cell_len(title) + gap + cell_len(counter) <= width:
            header.append(title, style=fg)
            header.append(" " * (width - cell_len(title) - cell_len(counter)))
            header.append(counter, style=ink)
        else:
            header.append(truncate_cells(title, width), style=fg)
        return header

    def _row_ground(self, index: int) -> Style:
        """The row's background: selection by HUE, hover additive on top of it.

        The same three steps the ``/resume`` picker paints on the same ``overlay``
        card, and for the reason recorded on ``tint-select`` in ``theme.py``: pure
        elevation cannot carry selection here (surface->raised measures 1.096:1),
        so a bare caret left a mouse user with almost nothing saying which row a
        click would take. Selection stays dominant under the pointer — written the
        other way round, hovering the selected row swapped its tinted ground for
        the faintest step in the ramp and the highlight vanished exactly when the
        user reached for the mouse.
        """
        hovered = index == self._hovered
        if index == self.state.selected:
            token = "tint-select-hi" if hovered else "tint-select"
        elif hovered:
            token = "tint-select"
        else:
            token = "overlay"
        return Style(bgcolor=theme_mod.semantic_color(token))

    def row_key(self, index: int) -> str:
        """The letter that answers row ``index`` directly, or ``""``.

        Empty on this class: the ``ask`` picker's rows are addressed by ordinal
        and have no letters of their own. :class:`ApprovalPrompt` overrides it,
        because its three rows DO answer to letters (`y`/`n`/`A`) that predate
        the list and that people have in their fingers.

        It exists here so the gutter has one place to ask, rather than the
        renderer growing a branch on which subclass it is drawing.
        """
        return ""

    def _row_number(self, index: int) -> str:
        """The digit gutter's contents for one row.

        Rows past nine keep the indent and lose the number: the digits are a
        shortcut, and re-flowing the list at ten options to reclaim two cells
        would move every row under the cursor. The free-text row is the
        exception, because it is the one row a long list ALWAYS pushes past the
        digits and the one row that can express an answer nobody enumerated —
        with twelve options it drew a blank gutter while the footer still
        offered `1-9 jump`, so `Other` was unreachable by digit exactly where
        scanning the list is hardest (D13).

        A row with a LETTER of its own shows that instead. On the approval card
        the letters are the older interface — `y`/`n`/`A` predate the list and
        people have them in their fingers — and after the rework they were live
        bindings rendered nowhere, so "allow all" had lost its only discoverable
        shortcut (D4, design round 1). Shown in the gutter the ordinal would
        occupy, they cost no width, and the ordinals still work for anyone who
        counts rows instead.
        """
        key = self.row_key(index)
        if key:
            return f"{key}."
        if index < 9:
            return f"{index + 1}."
        if index == self.other_row:
            return f"{OTHER_JUMP_KEY}."
        return ""

    def _row_text(
        self,
        index: int,
        width: int,
        ground: Style,
        fg: Style,
        dim: Style,
        faint: Style,
        layout: _CardLayout,
    ) -> Text:
        """One option row: cursor, number, checkbox, label.

        The accent green marks WHAT ENTER WILL TAKE — the cursor's row on a
        single-select, the ticked rows on a multi-select, and neither on a
        free-text row with nothing typed into it. It used to be the cursor's
        label in every mode, so one multi-select frame spent the ink on two
        different claims at once: `[x]` on row 1 (chosen) and the label of row 3
        (merely under the cursor, its own box empty). The accent's one-thing
        rule in `local_operator.tcss` was then not what the frame said (D11),
        and on a multi-select the ink was pointing at the row Enter does NOT
        take.

        The cursor stays unmistakable without it: `tint-select` is a HUE step
        rather than an elevation one (see theme.py — elevation alone measures
        1.096:1), and the caret sits two columns to its left. The caret is
        ``muted`` and not the accent for the reason the command picker records:
        a second green glyph beside a green label reads as a duplicated caret.
        """
        selected = index == self.state.selected
        accent = ground + Style(color=theme_mod.semantic_color("accent"))
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append(
            f"{CURSOR} " if selected else " " * GUTTER_CELLS,
            style=ground + Style(color=theme_mod.semantic_color("muted")),
        )
        row.append(
            f"{self._row_number(index):<{NUMBER_CELLS}}",
            style=ground + (dim if selected else faint),
        )
        spent = GUTTER_CELLS + NUMBER_CELLS
        typed = bool(self.state.typed.strip())
        if self.question.multi:
            checked = index in self.state.checked or (index == self.other_row and typed)
            # A ticked box is the accent: it is the statement the ink is spent
            # on — "Enter takes this one" — and a multi-select confirms several
            # rows at once, so the ticks have to be readable without moving the
            # cursor over each of them.
            row.append(
                CHECK_ON if checked else CHECK_OFF,
                style=accent if checked else ground + faint,
            )
            spent += cell_len(CHECK_ON)
            taken = checked
        else:
            taken = selected and (index != self.other_row or typed)

        budget = max(LABEL_MIN_CELLS, width - spent)
        text = self._row_label(index)
        if index == self.other_row and selected:
            # The typed string keeps its TAIL: a field that truncated the end
            # would hide the characters being typed, which is the only part of
            # it the user is looking at.
            budget -= cell_len(FIELD_CARET)
            if self.question.secret:
                prefix = SECRET_PREFIX
                rendered = SECRET_MASK * len(self.state.typed)
            else:
                prefix = OTHER_PREFIX
                rendered = _tail_cells(self.state.typed, max(1, budget - cell_len(OTHER_PREFIX)))
            row.append(prefix, style=accent if taken else ground + fg)
            row.append(
                truncate_cells(rendered, max(1, budget - cell_len(prefix))),
                style=ground + fg,
            )
            row.append(FIELD_CARET, style=accent if taken else ground + dim)
        else:
            row.append(truncate_cells(text, budget), style=accent if taken else ground + fg)
        if self.question.recommended == index and not layout.show_descriptions:
            # No description line to carry the tag, so it rides here — but only
            # when it fits AFTER the label, rather than out of the label's own
            # budget. Charged to the budget it made the promoted option the
            # shortest label on the card, and on a 28-cell screen the label
            # floor won the arithmetic and the 15-cell tag was appended on top
            # of the floor anyway, pushing the card two cells past the terminal
            # (D2/D6). A badge that truncates what it promotes, or that
            # overflows the screen to fit, is worth less than no badge: the
            # recommendation is PRESELECTED too, so the cursor is already there.
            #
            # Drawn exactly as :meth:`_description_text` draws it: the separator
            # at ``muted`` and the tag at ``fg`` + bold, the label's own ink.
            # The whole run sat at ``dim`` here — 3.43:1, under the 4.5:1 WCAG
            # AA floor the description text was deliberately walked UP to (D7),
            # and a second unscoped treatment of one badge besides. This is the
            # call site that fires where there is NO description column, so it
            # is the frame on which the badge is the only thing marking the
            # promoted row: the least legible place to spend the least legible
            # ink.
            #
            # ``muted`` is derived here rather than taken as a parameter, for
            # the same reason ``accent`` above is: the caller's style set is the
            # one the ROW needs, and a sixth Style threaded through the
            # signature for one branch would change a call shape the tests hold.
            separator = "  · "
            muted = ground + Style(color=theme_mod.semantic_color("muted"))
            if cell_len(row.plain) + cell_len(separator) + cell_len(RECOMMENDED_TAG) <= width:
                row.append(separator, style=muted)
                row.append(RECOMMENDED_TAG, style=ground + fg + Style(bold=True))
        return _fit_row(row, width, ground)

    def _description_text(
        self,
        index: int,
        width: int,
        ground: Style,
        tag_ink: Style,
        ink: Style,
        granted: int,
        layout: _CardLayout,
    ) -> list[Text]:
        """The row's description lines: the recommendation tag, then the consequence.

        Drawn at ``muted``, which measures 6.51:1 on this card's ``overlay``
        ground. It has been walked up this ramp twice for the same reason: the
        first captured frame had it at ``faint`` (1.49:1, barely present), and
        it then sat at ``dim`` for a release — 3.43:1, which is under the 4.5:1
        WCAG AA floor for body text (D7, design round 1).

        The floor is the right test rather than a nicety, because of what this
        line CARRIES: it is the consequence of choosing the row, which is the
        thing the user is reading the card to compare — and on the approval
        prompt it is the difference between "ask again next time" and "stop
        asking for this session". Text that decides an authorisation cannot be
        the least legible text on the card.

        It stays a step below the LABEL (``fg``, 11.30:1), so the ranking
        between a row's name and its explanation is intact; what changed is
        that the explanation is now readable in absolute terms and not only
        relative to the label above it.

        The tag lives HERE, ahead of the prose, rather than after the label: on
        the label line it was paid for out of the label, so the one row the
        model is pointing at carried the shortest text on the card (D6).

        It is drawn at ``tag_ink`` — ``fg`` + bold, the label's own ink — and
        NOT at the ``ink`` its prose uses. It sat at ``muted`` for a release,
        which is the same style as the prose it introduces: the docstring here
        claimed it was "the loudest thing on a line of ``dim`` prose", and that
        stopped being true the moment prose itself was walked up to ``muted``
        (D7, round 1). The frame that resulted had a badge nobody could find
        (D4). Ranking now: label (``fg`` bold, line N) > badge (``fg`` bold,
        line N+1, indented) > prose (``muted``). The badge matches the label's
        weight rather than exceeding it and loses on POSITION, which is the
        right ordering for a hint that qualifies a label. Its salience comes
        from weight, case and a glyph rather than from hue — see
        :data:`RECOMMENDED_TAG` for why colour is not available here.

        ``granted`` lines, not one. Where the wrap is longer than the grant the
        LAST KEPT line is marked ``…`` — the same "say that it continues"
        discipline the question uses when its own tail is cut. Marking every
        line would say each of them was cut when only the last one is, which is
        the reading a wrapped paragraph must not invite.

        ...unless the ``ctrl+e`` block is about to draw the REST of this very
        paragraph on the next line (``continued`` below). ``layout`` is what
        says so, and it is passed for the same reason :meth:`_row_text` takes
        it: whether a line is the last one the user sees is a fact about the
        FRAME, not about the row. The mark and the fill both exist to stand in
        for text that is not on screen; where it is on screen, immediately
        below and in the same column, they describe a cut that is not
        happening. The fill is the more damaging of the two — it re-draws the
        opening of the line the block then starts with, so the paragraph gains
        a stutter exactly at the seam it is being read across (F3).
        """
        indent = self._description_indent()
        wrapped = self._description_lines(index, width)
        # ``continued`` is dead under line-granular windowing: there is no
        # separate reveal block that draws the paragraph's remainder immediately
        # below, so nothing suppresses the last line's ``…`` mark. The reveal is
        # now a per-row cap LIFT (§4) — the selected row's OWN ``granted`` lines
        # grow to REVEAL_MAX_ROWS, and where the wrap still exceeds that
        # ``_prose_line`` marks the cut, exactly the "say that it continues"
        # discipline every other row follows.
        continued = False
        # An option with a description draws its ``granted`` lines. A row with
        # no prose contributes no line here at all (the caller only enters this
        # method for rows in ``description_rows``, which are exactly the rows
        # with ``granted >= 1``), so ``or [""]`` guards only the recommended row
        # whose sole "line" is the badge.
        kept = wrapped[:granted] or [""]
        rows: list[Text] = []
        for position, text in enumerate(kept):
            body = Text(no_wrap=True, overflow="ellipsis")
            body.append(" " * indent, style=ground)
            room = max(1, width - indent)
            if position == 0 and self.question.recommended == index:
                body.append(RECOMMENDED_TAG, style=ground + tag_ink + Style(bold=True))
                room -= cell_len(RECOMMENDED_TAG)
                if text and room > 3:
                    body.append(" · ", style=ground + ink)
                    room -= 3
                else:
                    text = ""
            if text:
                body.append(
                    truncate_cells(
                        self._prose_line(index, wrapped, kept, position, text, continued=continued),
                        room,
                    ),
                    style=ground + ink,
                )
            rows.append(_fit_row(body, width, ground))
        return rows

    def _prose_line(
        self,
        index: int,
        wrapped: list[str],
        kept: list[str],
        position: int,
        text: str,
        *,
        continued: bool = False,
    ) -> str:
        """One drawn line of ``index``'s prose: its own, or the SOURCE tail if last.

        The last kept line of a paragraph that continues is filled from the REST
        of the prose rather than from its own wrapped line; the caller's
        :func:`truncate_cells` then marks the cut it makes.

        Filling rather than marking the wrapped line matters where the grant is
        one, which is every description the approval gate draws on a narrow
        terminal: a wrapped line stops at a word boundary, so marking it would
        end the consequence several cells earlier than a single truncated line
        does. That is the authorisation frame getting WORSE. Filled, the line
        carries at least as much of the consequence as it carried before and
        says that there is more.

        The tail-fill matters where the grant is one, which is every description
        the approval gate draws on a narrow terminal: a wrapped line stops at a
        word boundary, so marking it would end the consequence several cells
        earlier than a single truncated line does. That is the authorisation
        frame getting WORSE. Filled, the line carries at least as much of the
        consequence as it carried before and says that there is more.

        ``continued`` is retained for callers that draw the paragraph's
        remainder immediately below, but the line-granular card no longer has a
        second region that does so (the reveal is an in-place cap lift, §4), so
        the sole live caller (:meth:`_description_text`) passes ``continued``
        False. Kept as a keyword because the default is the one that is safe to
        get wrong: an omitted argument marks a cut that did not happen.
        """
        if continued:
            return text
        if position == len(kept) - 1 and len(wrapped) > len(kept):
            return _wrap_tail(self._row_description(index), wrapped, position)
        return text

    def _footer_hints(self, width: int) -> list[tuple[str, str]]:
        """The key hints that fit, shedding WORDS before it sheds KEYS.

        Two passes, and the order between them is the point: a key with no word
        beside it still names a key that exists, while a dropped hint is a key
        nobody can discover. Shedding whole hints first took `space toggle` off
        a 46-column multi-select and kept `esc skip`, leaving five empty boxes
        and one offered key that does nothing until one of them is ticked (D3).

        Three ladders rather than one, because the keys genuinely differ by
        state: a multi-select confirms rather than answers, and while the
        free-text row is selected the digits and Space have become letters, so
        advertising them there would be a lie. Each ladder is ordered LEAST
        defended first and drives both passes, so the last word standing and the
        last key standing belong to the same hint.

        What sits at the end differs by state, and that is the whole ranking.
        Normally it is ``esc``: a card with no stated way out is unusable where
        a card with fewer keys is merely terse, and `skip` is the one word here
        nobody can guess (it leaves THIS question unanswered, it does not cancel
        the ones already answered). On a multi-select it is ``space``, which
        outranks even that: it is the ONLY key that can answer the question, and
        it used to be dropped while `esc skip` survived.
        """
        # While the CARD does not hold the caret, most of this keymap is not
        # reachable: the composer has focus and takes the arrows, Enter and the
        # printable keys as text. Only what the app routes still works — the
        # row ordinals and Escape — so those are all the footer claims.
        #
        # A footer describing one keyboard while the caret sits on another is
        # the same lie this row already refuses to tell about `enter` on a
        # collapsed card. Measured from the composer: `↑↓` moved nothing and
        # `enter` answered nothing, while both were advertised (D13, design
        # round 3).
        if not self.has_focus:
            hints = []
            # ...and not even those while the composer holds a DRAFT. The
            # routing stands down whenever there is text in the buffer, so
            # every character is the user's again — which is exactly the state
            # the mount-time focus yield creates (D12), so the keys would be
            # advertised precisely where they are dead (F6, agent review round
            # 4). The exit survives, because Escape works from anywhere.
            routed = None if self._composer_has_draft() else self.routed_hint()
            if routed is not None:
                hints.append(routed)
            elif self.visible_rows and not self.answer_keys():
                # A question the composer cannot answer at all — a multi-select,
                # answered by Space and Enter, which the composer owns. Tab is
                # the way to reach it, and it has to be NAMED or it is a key
                # nobody can discover. Inferring the handover from the buffer
                # instead cost two rounds and two lost messages (F9, D18).
                hints.append((TAB_HINT_KEY, "answer here"))
            hints.append(self._exit_hint)
            # Through the same ladder the focused footer uses, rather than
            # returned raw for `_cut_row` to ellipsise. Returned raw, a narrow
            # card cut the exit mid-word — `1 answer · esc sk…` at 22 columns —
            # and `skip` is the one word this row ranks as unsheddable, because
            # a card with no stated way out is unusable (D3, and D16 in design
            # round 4). Shedding the routed hint's WORD first and then the hint
            # itself keeps `esc skip` whole down to the narrowest card.
            # The ladder names whichever hint precedes the exit, so the exit is
            # never in it and can never be shed. Passing an EMPTY ladder when
            # the Tab hint is showing meant both shed passes iterated over
            # nothing and the row went to `_cut_row` raw — which is exactly what
            # this call exists to prevent: at 18-26 columns the multi-select
            # painted `⇥ answer here · esc…` and then `⇥ answer here…`, stating
            # no way out at all, on the one surface where Escape is the only
            # alternative to the handover (D19, design round 7).
            shed_first = [routed[0]] if routed else [TAB_HINT_KEY]
            return self._shed_to_fit(hints, shed_first, width)

        if self.state.selected == self.other_row:
            if self.question.secret:
                # No movement keys: there is only one row, and advertising
                # arrows or digits would describe a keyboard that does nothing.
                hints = [("type", "the value"), ("enter", "store")]
                ladder = ["type", "enter", "esc"]
            else:
                hints = [("type", "your answer"), ("↑↓", "move"), ("enter", "accept")]
                ladder = ["↑↓", "type", "enter", "esc"]
        elif self.question.multi:
            hints = [("↑↓", "move"), ("space", "toggle"), ("enter", "confirm")]
            ladder = ["↑↓", "enter", "esc", "space"]
        else:
            # `0-9` once the free-text row has taken `0`, because `1-9` then
            # advertised a range that stopped short of the list (D13).
            jump = f"{OTHER_JUMP_KEY}-9" if self.other_row >= 9 else "1-9"
            # Where the rows carry LETTERS of their own, the footer names those
            # instead of the digit range: the gutter is showing `y`/`n`/`A`, and
            # a hint reading `1-9 jump` beside it describes a different keyboard
            # from the one on screen.
            hints = [("↑↓", "move"), (jump, "jump"), ("enter", "answer")]
            ladder = ["↑↓", jump, "enter", "esc"]
            # Rows that carry their own LETTER need no range hint at all: the
            # letter is printed in each row's gutter, next to the label it
            # answers, which says it better than a footer can. Repeating it here
            # produced `y/n/A answer · enter answer` — two hints claiming the
            # same verb, and the digit range would have been a claim about a
            # keyboard the card is not showing.
            if any(self.row_key(index) for index in self._window()):
                hints = [pair for pair in hints if pair[0] != jump]
                ladder = [key for key in ladder if key != jump]
            # ...and only where there is somewhere to jump TO. The range was
            # advertised unconditionally, so a card windowed down to a single
            # row still offered `1-9 jump` — verified live on a 3-row approval
            # card, where `5`, `7` and `9` did nothing at all (D3, design round
            # 1). A key offered where it does nothing is the same lie the
            # collapsed card's footer already refuses to tell about `enter`.
            #
            # Keyed on the DRAWN page rather than on `row_count`: the digits
            # address rows by ordinal, and a row that is not on screen is one
            # the user cannot see they are committing to.
            drawn = len(self._window())
            if drawn < 2:
                hints = [pair for pair in hints if pair[0] != jump]
                ladder = [key for key in ladder if key != jump]
        reveal = self._reveal_hint()
        if reveal is not None:
            hints.append(reveal)
            # Immediately BEFORE the exit in the ladder, never after it. The
            # ladder is ordered LEAST DEFENDED FIRST, and `_shed_to_fit` also
            # refuses to drop its last entry outright — so appending `^e` past
            # `esc` makes the exit the cheaper of the two. Measured on the
            # approval card at 30 columns, exactly that: the frame read
            # `↑↓ · enter · ^e more · esc`, spending cells on a reading aid
            # while the word for REFUSING an authorisation had been shed. The
            # exit's word is the one this row defends hardest (D3/D16/D19), and
            # "deny" is not guessable from "skip".
            #
            # Located by the exit's KEY rather than by position: the ladders
            # differ in what they defend hardest, and the multi-select's ends
            # with `space` (the only key that can answer it) BEYOND the exit —
            # so "one from the end" would put `^e` on the wrong side of `esc`
            # there while looking right on the single-select.
            exit_key = self._exit_hint[0]
            cut = ladder.index(exit_key) if exit_key in ladder else len(ladder)
            ladder = [*ladder[:cut], REVEAL_HINT_KEY, *ladder[cut:]]
            # ...and shed OUTRIGHT before the second pass reaches the exit.
            # `_shed_to_fit` keeps the ladder's last entry whatever happens, so
            # on the ask card (`esc` last) `^e` is already safe — but the
            # approval card and the multi-select rank other keys past the exit,
            # and there `^e` survived while `deny`/`skip` had been shed.
            # Measured at 18-23 columns: `^e · esc`, a card spending its last
            # cells on a reading aid instead of the word for refusing.
            #
            # Dropped here rather than by reordering the ladder, because the
            # ladder's tail is a genuine ranking those two surfaces own (`space`
            # is the only key that can answer a multi-select) and `^e` outranks
            # nothing on this row.
            #
            # Measured against the row WITH the exit on it, which is not yet in
            # `hints` here: the exit is what the check is protecting, so leaving
            # it out would ask whether the row fits without the thing that has
            # to fit.
            if _hint_cells([*hints, self._exit_hint]) > width:
                hints = [pair for pair in hints if pair[0] != REVEAL_HINT_KEY]
                ladder = [key for key in ladder if key != REVEAL_HINT_KEY]
        hints.append(self._exit_hint)

        return self._shed_to_fit(hints, ladder, width)

    def _reveal_hint(self) -> tuple[str, str] | None:
        """``^e more`` / ``^e less``, or nothing where the key does nothing.

        This row already refuses to advertise dead keys — the digits on a
        one-row window, the whole keymap while the composer holds the caret —
        and on a roomy terminal the reveal IS dead: (A)'s continuation lines
        have already drawn every description in full, so the key would toggle
        a mode that changes nothing. Offered anyway it is the same lie, one
        surface further along.

        So: only where some DRAWN row has more prose than the lines it was
        granted, or where the reveal is already on and the user needs to be told
        how to leave it. Answered against the plan the card is actually drawing
        rather than against the descriptions alone, because "is there more" is a
        question about the grant, not about the text.

        Below the sizes where the reveal can buy a line at all (measured: budget
        8 and under, at 100x24 and below) it stays hidden — the plan grants
        nothing in either state, so neither branch fires.
        """
        if self.state.revealed:
            # `less` only while the lift actually draws a line the default view
            # did not. A revealed card whose selected row shows no more than its
            # 2-line clamp is byte-identical to the default one, so naming a key
            # to undo it would point at a change nothing on screen shows.
            return (REVEAL_HINT_KEY, "less") if self._reveal_is_useful() else None
        return (REVEAL_HINT_KEY, "more") if self._reveal_is_useful() else None

    def _offers_reveal(self) -> bool:
        """Is ``^e`` actually ON the footer as the card is drawn right now?

        The footer is the card's whole statement of what its keys do, and this
        asks it rather than re-deriving the answer: the hint can be dropped for
        reasons the predicate below knows nothing about — the shed ladder taking
        it to keep the exit whole, the composer holding focus, the card drawing
        no options at all. Re-derived, the key would work in states the card
        does not advertise it in, which is the same defect as advertising a key
        that does not work, inverted.
        """
        plan = self._layout()
        if not plan.show_footer or plan.body_line_budget <= 0:
            # No footer, or no option line drawn at all to reveal anything
            # about. `_footer_row` draws only the exit on a card with no options
            # drawn (`drawn=False`), so the hints below are not what is on
            # screen there.
            return False
        return any(key == REVEAL_HINT_KEY for key, _ in self._footer_hints(plan.width))

    def _reveal_is_useful(self) -> bool:
        """Would turning ``ctrl+e`` on put prose on screen that is not there?

        Re-resolved for the cap-lift model (§4.3). The reveal is now a per-row
        cap lift on the SELECTED row inside the one line viewport, so this is
        two conditions, and both are needed. Either alone offers a key that does
        nothing, which is the lie this footer already refuses to tell about the
        digits and about ``enter``:

        - the SELECTED row's description must be longer than :data:`DEFAULT_DESC_CAP`
          — there is more of ITS OWN prose to show. On a roomy terminal the row
          already draws its full wrap (2 lines is the whole description), so the
          lift would change nothing. The selected row and not "any drawn row":
          lifting one row's cap never removes another row's description (there is
          no shared block pool any more), so the old BLOCKER-1 "would this cost
          the column" refusal is gone — it cannot arise;
        - and the lift must actually put a NEW line of that row in the viewport:
          the revealed line list must draw at least one more of the selected
          row's lines than the default one does, given ``body_line_budget`` and
          where the row sits. Below the sizes where the viewport can show a
          third line of the row, the revealed frame equals the default one and
          the key stays hidden.

        Answered against the two PLANS the card would draw, not against the
        descriptions alone: "is there more on screen" is a question about the
        viewport, not about the text.
        """
        selected = self.state.selected
        width = self._card_width()
        # More of the selected row's OWN prose than the default clamp shows?
        if len(self._reveal_wrap(selected, width)) <= DEFAULT_DESC_CAP:
            return False
        default = self._layout(reveal=False)
        if selected not in self._window(default):
            # The cursor's row is not even partially drawn. Nothing on screen
            # would change in a way the user can attribute.
            return False
        revealed = self._layout(reveal=True)
        # How many of the selected row's lines each plan actually draws in its
        # viewport — the lift is useful iff the revealed frame shows strictly
        # more of THIS row than the default one.
        return self._visible_row_lines(revealed, selected) > self._visible_row_lines(
            default, selected
        )

    def _visible_row_lines(self, layout: "_CardLayout", index: int) -> int:
        """How many of row ``index``'s visual lines fall inside the viewport.

        Clamps the row's ``[start, end)`` line span against the viewport
        ``[_offset, _offset + body_line_budget)`` under ``layout``'s offset, so
        a partially-clipped row counts only its visible lines (§2.3). Used by
        :meth:`_reveal_is_useful` to compare what the default and revealed
        frames actually put on screen for the selected row.
        """
        lsbr = layout.line_start_by_row
        body = layout.body_line_budget
        if not lsbr or body <= 0:
            return 0
        total = lsbr[-1]
        offset = max(0, min(self._offset, max(0, total - body)))
        # The revealed plan may want a different offset (its taller row scrolls
        # into view); model _move_to's scroll so the comparison is honest.
        cur_start, cur_end = lsbr[index], lsbr[index + 1]
        offset = self._scroll_offset_for_cursor(offset, cur_start, cur_end, body, total)
        lo, hi = offset, offset + body
        return max(0, min(cur_end, hi) - max(cur_start, lo))

    def _shed_to_fit(
        self, hints: list[tuple[str, str]], ladder: list[str], width: int
    ) -> list[tuple[str, str]]:
        """Fit ``hints`` into ``width``, shedding WORDS before whole KEYS.

        ``ladder`` is ordered LEAST defended first and drives both passes, so
        the last word standing and the last key standing belong to the same
        hint. Keys not named in it are never shed — which is how the exit
        survives to the narrowest card.
        """

        cells = _hint_cells
        shown = list(hints)
        if cells(shown) <= width:
            return shown
        for key in ladder:
            shown = [(name, "" if name == key else what) for name, what in shown]
            if cells(shown) <= width:
                return shown
        for key in ladder[:-1] if len(ladder) > 1 else ladder:
            shown = [pair for pair in shown if pair[0] != key]
            if cells(shown) <= width:
                return shown
        # One bare key left, on a card too narrow for two. It is the one the
        # ladder defends hardest, and :func:`_cut_row` keeps even that inside
        # the card rather than letting it widen the screen.
        return shown


def _hint_cells(pairs: list[tuple[str, str]]) -> int:
    """Cells a footer row of ``key word`` hints takes, separators included.

    One definition, because two callers ask the same question and a drifting
    copy of it would let a hint be kept by one measure and shed by another — on
    the row where the disagreement costs the card its only stated way out.
    """
    return sum(cell_len(f"{key} {what}".strip()) for key, what in pairs) + 3 * max(
        0, len(pairs) - 1
    )


def _cut_row(row: Text, width: int) -> Text:
    """``row`` cut to the card's width, in the card's own width model.

    Load-bearing rather than defensive: every one of these Texts is
    ``append_text``-ed into ONE Text whose overflow governs, so the
    ``no_wrap``/``ellipsis`` pair they were built with never fires. The body is
    ``width: auto``, so a line that overshoots simply widens the card — which is
    how a 15-cell tag pushed the card two cells past a 28-cell screen and gave
    the app the one condition AGENTS.md calls always a bug (D2).
    """
    if cell_len(row.plain) > width:
        row.truncate(width, overflow="ellipsis")
    return row


def _fit_row(row: Text, width: int, ground: Style) -> Text:
    """``row`` at exactly the card's width, in its own ground.

    Without the padding the tinted selection ends where the label ends, so the
    highlight is a ragged patch behind the text rather than a row — and the
    shorter of two adjacent options looks less selected than the longer one.
    """
    _cut_row(row, width)
    remaining = width - cell_len(row.plain)
    if remaining > 0:
        row.append(" " * remaining, style=ground)
    return row


def _sanitize(question: AskQuestion) -> AskQuestion:
    """A copy of ``question`` with every model-authored string stripped of
    control sequences, so nothing reaching the terminal can move the cursor."""
    return question.model_copy(
        update={
            "id": strip_control_sequences(question.id),
            "question": strip_control_sequences(question.question),
            "options": [
                option.model_copy(
                    update={
                        "label": strip_control_sequences(option.label),
                        "description": strip_control_sequences(option.description),
                    }
                )
                for option in question.options
            ],
        }
    )


def _wrap_tail(source: str, wrapped: list[str], position: int) -> str:
    """``source`` from where ``wrapped[position]`` starts — the rest of the prose.

    What the last kept line of a cut-short description is FILLED from. Sliced
    out of the source rather than rejoined from ``wrapped[position:]``, because
    the pieces `wrap_cells` produces are not uniformly space-separated: it
    BREAKS a word longer than the row (a URL, a path — its documented behaviour
    and the reason it exists), and those pieces never had a space between them.
    Rejoining them with one fabricates a character the user is being asked to
    authorise against, which on a path or a URL misreads the string itself.

    Measured on the fill itself: 69,392 of 120,083 randomised fills contained a
    character sequence absent from the source; with the slice, none do. On a
    keycap run at room 8 the join produced ``1..1..1..1.. 1..\u2026`` where the
    source has no space at all.

    What that is NOT is a claim about the painted frame. Swept over 2,832
    drawn comparisons (7 description shapes, widths 12-121, both hosts, single
    and multi select, grants 1-4) the two fills painted the SAME line every
    time, because the row's outer ellipsis re-cuts ahead of the seam. So this
    is correctness at the boundary rather than a visible-defect fix, and it is
    worth having for the reason the surface exists: the fill is the string a
    user authorises against on the approval gate, and a widening card or a
    longer consequence moves the cut, not the invariant.

    Walked line by line rather than measured, because only `wrap_cells` knows
    which breaks consumed a space. Each row it returns is a character prefix of
    what is left once the previous row's trailing separator is dropped; that
    holds over 19,619 randomised inputs (CJK, keycap clusters, combining marks,
    RTL, multi-space runs, unbreakable words) and it is what makes the walk
    exact. A source whose walk does not align is not reconstructable, and the
    honest answer there is the wrapped line itself: shorter than the fill by a
    few cells, but never a character the source does not contain.
    """
    remainder = source.lstrip(" ")
    for line in wrapped[:position]:
        if not remainder.startswith(line):
            return wrapped[position]
        remainder = remainder[len(line) :].lstrip(" ")
    return remainder if remainder.startswith(wrapped[position]) else wrapped[position]


def _tail_cells(text: str, width: int) -> str:
    """``text`` reduced to ``width`` cells, keeping its END.

    The mirror of :func:`truncate_cells`, for the one string on this card whose
    tail is the part being read: the characters the user is typing right now.
    """
    if cell_len(text) <= width:
        return text
    kept: list[str] = []
    spent = 1  # the leading ellipsis
    for char in reversed(text):
        size = cell_len(char)
        if spent + size > width:
            break
        kept.append(char)
        spent += size
    return "…" + "".join(reversed(kept))

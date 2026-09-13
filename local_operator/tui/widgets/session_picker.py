"""The ``/resume`` picker: choose a past conversation by NAME, not by hash.

Why a screen rather than a notice. The recovery list used to be printed INTO
the transcript: a block of ``<12-hex id>   3h ago`` rows that pushed the
conversation up, could not be navigated, stayed on screen after it had been
used, and left the user to retype an id they had to read off the scrollback.
Choosing a conversation is a two-way question — the app offers the options,
the user picks one — so it takes a surface that can hold a selection and hand
an answer back. That is exactly a modal screen, and it is what the trajectory
viewer already does for the other "read a list, pick a row" case.

Why names. A column of hex ids is not something anyone recognises their own
work in; the id is what the machine resumes, not what a human picks by. The
name is the session's opening user message (see
:func:`local_operator.resume.session_name`), which is both the only
per-session title on disk and the thing the user actually remembers about the
session.

The list is filterable by typing because the ids are unmemorable and the names
are not: with a hundred sessions, "asteroids" finds the one you mean faster
than paging can. Filtering narrows without reordering FOR A FIXED QUERY; a new
query re-ranks by relevance (best match first) and re-homes the cursor to the
top match, matching this app's command and ask pickers. That preserves the "a
row must not move under the cursor" invariant, because the only event that
reorders — a query change — is the same one that moves the cursor to the new
rank-0 row; a fixed query's order is byte-for-byte stable across repaints.

That statement is about the QUERY TEXT and nothing else, which is stronger than
it sounds and was briefly untrue. Every input to the row list — which tiers run,
what they admit, how the result is ordered — is a function of
``(rows, query, digests)``, so the same visible query renders identically
however the user arrived at it: typed straight through, or typed past and
backspaced back. A rule that read run history instead (which rows the previous
keystroke showed, whether a tier had latched) made the same query answer two
ways, and a user cannot know which route they took, so they could not tell
which answer they were looking at. See ``_soft_tier_wanted`` for what that cost
and why it is paid.

**The filter also searches what was SAID in each conversation**, not only its
name — see ``session/search_index.py``. Matching on the name alone meant a
session could only be found by the words in its title, so a user who could not
recall how a conversation was named could not reach it at all, however
distinctive the work inside it was. The body digest also carries the session's
title and every PAST name it was renamed away from, so a topic-pivot session is
findable by the subject it ended on. **Matching is substring plus bounded soft
matching** (prefix, word-order-independent, small-typo-tolerant), not only
exact substring — see ``search_index.soft_search_digests``. A row matched on
its body, a past name, or a soft match rather than its visible name is marked,
because otherwise it looks like a result the filter had no reason to return.

**The card measures the terminal.** Every column here is a cell count derived
from the screen, not a constant: the first cut shipped a fixed 78-cell card
that a 70-column terminal simply clipped, which amputated the id column
mid-token and left a truncated hex string that still looked like a valid id —
the one field a user copies into ``/resume <id>``. Below the width the id
needs, the id column is DROPPED rather than cut, and the age after it. The
same applies down: the chrome is reserved first and the list takes what is
left, so a short terminal loses list rows (which scroll) instead of the
footer (which is the only place the card says how to get out).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.paths import config_dir

# The row's searchable text (``fork_haystack``) and the search itself both live
# in ``session_search`` now: two spellings of "what text does this row have" is
# how one surface ends up finding a fork the other cannot.
from local_operator.resume import SessionRow, format_age
from local_operator.session.preview import (
    SessionPreviews,
    clip_to_height,
    demark,
    grep_context,
    wrap_turns,
)
from local_operator.session.search_index import SoftSearchIndex, search_digests
from local_operator.tui import theme as theme_mod
from local_operator.tui.terminal_title import SPINNER_FRAMES, SPINNER_INTERVAL_S
from local_operator.tui.widgets.tool_card import truncate_cells

logger = logging.getLogger(__name__)

#: The narrowest terminal this picker still draws something usable in. The
#: full-screen redesign has no maximum to pair with it — the panes take the
#: terminal — but the floor survives because the shed ladders are tested
#: against it: it is the width at which the key hints must still state the way
#: out (``test_the_way_out_is_stated_at_every_width_the_picker_supports``).
PICKER_MIN_WIDTH = 30

#: Name column floor. Below this a name is not identifiable, so the id and then
#: the age give up their cells first — they are lookup keys, and the name is
#: the thing being looked up.
NAME_MIN_CELLS = 16

#: What both empty surfaces say when the picker has nothing to offer: the
#: card's own body, and the notice ``/resume`` prints instead of opening it.
#: ONE string because the two are the same statement made in two places, and
#: they contradicted each other the moment either was edited alone.
#:
#: It names WHOSE sessions rather than claiming none exist. Delegated subagent
#: runs share the directory and are deliberately unlisted, so on a machine
#: whose only surviving sessions are children — reachable through retention,
#: which evicts the older parent before its newer children — "no previous
#: sessions" was false and told the user nothing about why.
#:
#: It must also FIT. The picker is full-screen now, so there is no fixed cap to
#: measure against — but the constraint did not go away, it moved: this string
#: is the RESULTS PANE's empty body, and that pane is a fraction of the
#: terminal (``plan_layout``). The first wording ran to 76 cells and hung two
#: past the rule of the 74-cell card this replaced; at 80 columns the pane is
#: narrower still. It wraps rather than truncating (``_wrap_cells``), because
#: the "subagent runs are not listed" clause is the EXPLANATION and not
#: decoration. Anything edited here is measured against the narrow case.
RESUME_EMPTY_NOTICE = "no conversations of yours to resume — subagent runs are not listed"

#: Below this terminal width the two panes STACK vertically.
#:
#: MEASURED BY A SWEEP, NOT DERIVED ON PAPER, and that distinction is the whole
#: reason this constant has a docstring: deriving it is exactly how design
#: round 2 produced the D16 BLOCKER. The value is the smallest width at which
#: the RENDERED side-by-side name field reaches :data:`NAME_MAX` and so stops
#: being narrower than the stacked field at that same width.
#:
#: Not the value the same arithmetic predicts on paper: the rendered pane comes
#: out 2–3 cells narrower than ``split × cols`` because the preview's
#: ``border-left`` and both panes' ``padding: 0 1`` are taken before the text
#: width, and Textual's ``fr`` resolution rounds. Re-run the sweep after
#: touching :data:`LIST_FR`/:data:`PREVIEW_FR` or any fixed column, because all
#: three move this number.
#:
#: 161, raised from 159 when :data:`SOFT_GUTTER_CELLS` became an unconditional
#: reservation. That is this docstring's own instruction being followed rather
#: than a retuning: the gutter is a fixed column, so the sweep was re-run and
#: the width at which side-by-side reaches :data:`NAME_MAX` moved by two. A
#: reviewer found the previous pairing — gutter reserved, constant left at 159
#: — as a 64 -> 63 name shrink at the flip on the query path, which is exactly
#: the D16 BLOCKER reappearing from the constant and the layout disagreeing.
STACK_BELOW_COLS = 161

#: Cells the name column is capped at: the longest of the 141 real session
#: names (p95 48, p99 53, max 64), measured in CELLS rather than characters.
#:
#: THIS CAP IS WHAT MAKES THE BREAKPOINT SOUND. It is not cosmetic, and
#: removing it reopens round 2's BLOCKER. Uncapped, the fields are
#: ``stacked = W − chrome`` and ``side-by-side = split×W − chrome``, so stacked
#: gains a full cell per terminal column while side-by-side gains only
#: ``split``: their gap DIVERGES without limit (at split 0.6, −64 cells at
#: W=160 and −128 at W=320), and **no breakpoint value can satisfy** "side-by-
#: side is never narrower than stacked would be at the same width". Raising the
#: breakpoint is unsatisfiable, not merely expensive.
#:
#: A cap makes both layouts SATURATE at the same value, so past the width where
#: side-by-side reaches it the two fields are exactly EQUAL and the invariant
#: holds for every larger width. 64 is chosen because 0 of 141 real names
#: exceed it: at the cap nothing truncates, so equal fields also means equally
#: zero truncation rather than equally bad. Design round 3 measured the result:
#: capacity 53 → 69 → 69 → 69 → 69 across 80 → 180, zero shrink events.
#:
#: THE ONE EXCEPTION TO "NEVER NARROWS", stated here because an invariant with
#: an undocumented exception is an invariant nobody can check: the name DOES
#: give up 14 cells at the width where the id column first fits (72 on this
#: store's measurements). That is a deliberate trade rather than the D16
#: defect — the id is the field a user copies into ``/resume <id>``, so it
#: appears as soon as there is room for it — and it is PINNED by
#: ``test_the_name_field_never_narrows_as_the_terminal_grows`` rather than
#: excused: that test asserts the narrowing happens at exactly one width, that
#: the width is the id flip, and that it is the same width whether or not a
#: query is active. The last clause is the load-bearing one; see
#: :data:`SOFT_GUTTER_CELLS`.
NAME_MAX = 64

#: p75 of the 141 real session names (median 33, p90 43, p95 46). Gates whether
#: the id column is shown: the id is worth its 14 cells only while the name
#: still clears the length three quarters of real names fit inside.
NAME_P75 = 39

#: The side-by-side split, in the LIST's favour (60/40). Round 2 had this
#: backwards at 2fr/3fr — the pane that TRUNCATES got the minority share while
#: the preview showed visible slack (measured at 140: an 83-cell pane drawing a
#: 76-cell rule beside a 56-cell list truncating 33% of its names).
LIST_FR = 3
PREVIEW_FR = 2

#: Stacked-layout row split: a fraction with clamps, not a fixed count. A fixed
#: preview height either starves the list on a 24-row terminal or wastes half
#: of a 50-row one.
#:
#: ``PREVIEW_MIN`` is 3 header lines + 1 rule + body — below it the pane shows
#: metadata and no conversation, which is not a preview. Past ``PREVIEW_MAX``
#: the list starves for no gain, since the pane scrolls anyway. Fewer than
#: ``LIST_MIN`` rows is a menu, not a list.
PREVIEW_MIN = 8
PREVIEW_MAX = 14
LIST_MIN = 6

#: How stale a live marker may be while the picker is open. The spinner keeps
#: advancing at ``SPINNER_INTERVAL_S`` (motion), but the DATA behind the markers
#: — `registry.scan()` plus `read_index()` — is re-read at this cadence instead
#: of on every frame.
#:
#: Sized against what the markers can actually say, not against taste: the
#: states they report come from heartbeat recency, whose own resolution is
#: 45 s (``HEARTBEAT_TIMEOUT_S``), so a 1 s refresh is 45× finer than that
#: signal and no transition can be shown late at a scale a reader could
#: notice. It still bounds the cost — 12.5 scans/s became 1 — and it keeps the
#: repaint-on-change guard (`_tick`) working, which is what makes a row that
#: REORDERS (a session parking on a gate, with nothing animating) reach the
#: screen.
LIVE_REFRESH_INTERVAL_S = 1.0

#: Rows the picker's container costs on top of the lines its panes draw.
#:
#: A survivor of the card era, kept because it still has a reader: the
#: real-stylesheet height test adds it to the composed line count to check
#: nothing is clipped (``test_the_panes_fit_the_terminal_at_every_height_on_
#: the_real_stylesheet``). Textual clips SILENTLY — rows past the region are
#: simply not drawn and nothing reads back that it happened — so that test is
#: the only thing standing between a layout change and an invisible clip.
CARD_PADDING_ROWS = 2

#: The cursor glyph, matching the command picker's. A caret plus a row ground
#: rather than a reversed row: the transcript behind this card is dim, and a
#: block of inverted colour reads as a selection the user made rather than as
#: the position they are on.
CURSOR = "❯"
#: Cells the cursor gutter always occupies, so names start at one column.
GUTTER_CELLS = 2

#: Prefix on a row the filter admitted because the query appears in the
#: CONVERSATION rather than in the visible name.
#:
#: Two cells wide (``cell_len`` measured, not assumed), and a plain typographic
#: quote rather than an icon-set glyph, so it costs the same width in every
#: terminal and is present in any font that can already draw the curly quotes
#: this app's own prose uses.
#:
#: A QUOTE mark rather than the `·` first shipped: the footer on this same
#: card uses `·` as its separator forty cells below, and one glyph meaning
#: "and also" in the chrome and "found inside this conversation" in the list
#: is a collision the reader has to resolve every time (D4). A quote reads as
#: "something was said here", which is what the mark actually means.
BODY_MATCH_MARKER = "” "

#: Prefix on a FORK that is still wearing the title it inherited from its
#: parent, drawn in its own reserved column AHEAD of the name.
#:
#: A PREFIX and not a suffix, and that is the whole point of it. The first
#: shipped form spliced ``(fork)`` onto the end of the name, inside the name
#: field — where it is the first thing an ellipsis eats. The name is condensed
#: to ``resume.NAME_MAX_CHARS`` (64) before this module ever sees it and the
#: card's name column measures 48 cells at 100 columns, so any title over ~40
#: characters lost the mark at EVERY terminal width, not just narrow ones. On
#: this machine's real store 17% of titles exceed that, so roughly one fork in
#: six rendered byte-identical to its parent — the exact twin-row confusion the
#: mark exists to resolve, and long descriptive titles are the ones users fork
#: from most. At 70 columns the suffix additionally ran straight into the age
#: column with no separating gap.
#:
#: In the fixed chrome ahead of the name, nothing truncates it: the ellipsis
#: now eats the tail of the title instead of the metadata about the row.
#: Verified at 80 and 70 columns against the real app (docs/evidence/fork-ux).
#:
#: Seven cells (``"[fork] "``), reserved for EVERY row whenever any row in the
#: result set is forked, exactly as the body-match marker reserves its column
#: and for the identical reason — see ``plan_columns``.
FORK_MARKER = "[fork] "

#: A live ``lop exec`` run: a machine-driven session, not a terminal someone is
#: sitting at.
#:
#: These rows are not new — since #804 every ``lop exec`` publishes an ordinary
#: attachable record, and ``decorate_rows(include_live=True)`` has been folding
#: them into this list ever since. What was missing is that they rendered
#: IDENTICALLY to a conversation the user started: same glyph, same "Ready".
#: Two facts make that worth a tag rather than leaving it implicit.
#:
#: * **They are somebody else's work**, in the same sense a subagent directory
#:   is — a supervisor composed the prompt. Resuming one is legitimate (that is
#:   what an attachable record IS) but it is a deliberate reach, not the row a
#:   user means when scanning for the conversation they had this morning.
#: * **They are ephemeral by design.** ``exec_control`` calls its records
#:   "deliberately ephemeral": a fast one-shot can be published and reaped
#:   between the paint and the Enter. A row vanishing under the cursor reads as
#:   a bug unless the row said what it was.
#:
#: Follows :data:`FORK_MARKER`'s form exactly — a bracketed word before the
#: name, reserved as fixed chrome for the whole result set, painted `dim` as
#: metadata about the row rather than part of the title — because it makes the
#: same kind of statement about the same column, and a second visual vocabulary
#: for "this row is qualified" would be the defect that pattern exists to avoid.
#:
#: Deliberately NOT a state glyph: the state column answers "what is it doing",
#: which an exec run answers exactly like any other session (busy, idle, needs
#: you). This answers "what KIND of thing is it", which is orthogonal, and
#: folding it into the glyph would make the two unaskable at once.
EXEC_MARKER = "[exec] "

#: The record kinds this picker TAGS. ``"tui"`` is the unmarked default — the
#: overwhelming majority, and tagging it would put a badge on every row to say
#: "normal" — and ``"daemon"`` is deliberately absent for now: the phone daemon
#: does not publish per-session records this list reads, so a tag for it would
#: be untested chrome. Adding a kind here is one entry plus its marker.
TAGGED_KINDS = frozenset({"exec"})

#: The needs-you mark: this session has parked a question and is holding a
#: runtime resident until somebody answers it. The one marker here that is
#: about the USER's attention rather than the session's state, which is why it
#: is the only one that also reorders the list.
NEEDS_YOU_MARKER = "!"

#: A session with wakes armed. Dormant wakes (a stopped session) render the
#: same glyph a step quieter rather than a different one: it is the same fact
#: about the session, qualified.
#:
#: ONE CELL, like every other marker here, and that is a constraint rather
#: than a preference: the column reserves ``STATE_COL_CELLS`` for glyph plus
#: separator, so a two-cell glyph consumes the separator and the name starts
#: flush against it. The first spelling was ⏰ (two cells) and rendered
#: ``⏰Morning standup notes`` while every other row had its space — caught
#: in the rendered frame, not by a test.
WAKE_MARKER = "◷"

#: An attached session — another terminal is already watching it. Resuming is
#: still fine (that is what a viewer IS now), but the user should know they
#: will not be alone in there.
ATTACHED_MARKER = "○"

#: A runtime that is up and warm with NOBODY watching it. A DIFFERENT glyph
#: from ``ATTACHED_MARKER``, not the same one in a quieter ink: round 1 (D6)
#: measured `muted` against `dim` at **1.90:1**, below any threshold for
#: telling two states apart (WCAG's 3:1 non-text floor is the comparison), and
#: invisible on a mismatched palette or to a reader with reduced colour
#: discrimination. "Someone else is watching" and "nobody is, it is just warm"
#: are two different facts, and DESIGN §10 assigns them different glyphs.
#:
#: Filled against the hollow ``○`` so the pair reads as a presence contrast at
#: a glance rather than as a brightness one. One cell, like every marker here.
IDLE_MARKER = "●"

#: A live pid whose heartbeat went stale. Distinguished from cold because the
#: remedy differs: a wedged session is one to `lop stop`, not to reopen.
WEDGED_MARKER = "✗"

#: An unacknowledged completion, by kind. The sidebar paints these over the
#: live-state glyph for a row whose last turn finished unread; they live here
#: beside the other markers so one module owns the list's whole glyph
#: vocabulary and a future author can see at a glance which cells are taken.
#:
#: ``⊘`` for interrupted is NOT a new invention — it is the glyph this
#: codebase ALREADY uses for that exact state everywhere else it renders an
#: outcome: ``tool_card.ICON_INTERRUPTED``, the subagent panel's
#: ``⊘ cancelled`` and ``session_presentation``'s "shut `interrupted ⊘` row".
#: The sidebar was the lone surface collapsing interrupted into ``✗``, so this
#: removes a second vocabulary rather than adding one. ``tool_card``'s contract
#: — "``✓``/``✗``/``⊘`` separate success, failure and interruption without a
#: single colour" — is the reason the fix is a SHAPE and not a tint: a design
#: round on the neighbouring markers measured `muted` against `dim` at 1.90:1
#: and rejected distinguishing two states by ink alone, and an interruption
#: against a failure is exactly that comparison.
#:
#: WHY SHAPE IS THE SIGNAL AND INK IS ONLY REINFORCEMENT — the measurement,
#: so the next author choosing a marker does not have to re-derive it. A design
#: round simulated colour-vision deficiency over these three inks:
#:
#: ===========  =========  ==============  ============
#: ink          actual     deuteranopia    protanopia
#: ===========  =========  ==============  ============
#: ``warning``  ``#e0b04b``  ``#bfbf47``   ``#b6b64b``
#: ``danger``   ``#ef8078``  ``#aaaa73``   ``#929278``
#: ``success``  ``#57c785``  ``#afaf87``   ``#bebe84``
#: ===========  =========  ==============  ============
#:
#: Three near-identical olives. For a deuteranope these rows are told apart by
#: ``⊘`` against ``✗`` against ``✓`` and by NOTHING ELSE — the hue separation
#: that carries them for trichromatic vision (36.6° dark, 34.1° light) simply
#: is not there. So a tint-only fix for the interrupted-wears-the-error-mark
#: bug would have shipped a change that a colour-blind reader cannot see at
#: all. Any future state added here must earn a distinct GLYPH; recolouring an
#: existing one is not a fix.
#:
#: CONSTRAINT — ``warning`` is now a TWO-MEMBER class, not a synonym for
#: "answer this gate". ``NEEDS_YOU_MARKER`` (``!``) and ``interrupted``
#: (``⊘``) resolve to the identical fill and can sit rows apart in the same
#: list. That is intended: both mean "this wants you", and ``!`` against ``⊘``
#: is a large enough shape difference to carry it while the tooltips
#: disambiguate. But the class is at its capacity — a THIRD amber marker would
#: leave the eye grouping three states as one before it resolves any shape, so
#: adding one needs a design round, not a free slot.
#:
#: One cell each, like every marker above: ``STATE_COL_CELLS`` reserves glyph
#: plus separator, and a two-cell glyph eats the separator (the ⏰ regression
#: ``WAKE_MARKER`` records). Verified with ``rich.cells.cell_len``.
COMPLETION_MARKERS: dict[str, tuple[str, str]] = {
    # `complete` and `error` keep the ink they had. `interrupted` takes
    # `warning` rather than `danger`: an interrupted turn is unfinished work,
    # not a failure, and `danger` is the ramp's "something broke" hue — the
    # exact false alarm the operator reported when seven interrupted sessions
    # read as seven errors. `warning` is also the ink `NEEDS_YOU_MARKER`
    # already carries for "this needs you to do something", which is precisely
    # what an interrupted turn is asking for: somebody to resume it.
    "complete": ("✓", "success"),
    "error": ("✗", "danger"),
    "interrupted": ("⊘", "warning"),
}

#: Cells reserved for the live-state column when ANY row in the result set
#: carries state. One cell for the state glyph plus its separating space; the
#: spinner frames, the wake glyph and the markers above are all one cell wide.
STATE_COL_CELLS = 2


#: Cells the widest ``format_age`` string occupies (``1000d ago``); ``just now``
#: is 8. Measured with ``cell_len`` over the range the function can produce,
#: because the age column is FIXED and right-aligned (D3: it was rendering at 13
#: distinct start columns across 37 rows) and a guessed width re-rags it.
AGE_CELLS = 9

#: Cells a session id occupies plus its separator. The id is the field a user
#: copies into ``/resume <id>``, which is why it is DROPPED rather than cut.
ID_CELLS = 12

#: Cells reserved for the soft-match ``~`` gutter while a query is active.
#:
#: RESERVED rather than appended, which is the whole point: appending the mark
#: unreserved overflowed the pane (measured: a 59-cell row + 3 in a 59-cell
#: pane) and wrapped it onto its own line, breaking the one-row-per-session
#: arithmetic the cursor depends on.
#:
#: Reserved UNCONDITIONALLY — never "only while a query is active". Making the
#: reservation depend on the query makes the LAYOUT depend on it too: the id
#: column then first fits at a different width in each state (70 unfiltered
#: against 72 querying), and the name field narrows at a width that moves as
#: the user types. That is the conditional-gutter form of D16, and it is what
#: ``test_the_name_field_never_narrows_as_the_terminal_grows`` detects by
#: asserting the flip width is identical on both paths.
SOFT_GUTTER_CELLS = 2

#: Cells of indent the grep-style context line is drawn at, under its row.
CONTEXT_INDENT = 4

#: The mark on a row admitted by a SOFT/fuzzy match, which by definition has no
#: literal substring to build a context line from. Without it most rows under a
#: broad query would be unexplained and the user could not tell "no context"
#: from "not a body match".
SOFT_MATCH_MARKER = "~"

#: Cells of indent the preview's body lines are drawn at, under their role
#: gutter. Subtracted from the wrap budget so a wrapped line plus its indent
#: still fits the pane.
PREVIEW_BODY_INDENT = 2

#: Lines ``ctrl+u``/``ctrl+d`` move the preview. Half a small pane, so the eye
#: keeps its place rather than being handed an entirely new screen.
PREVIEW_SCROLL_LINES = 5


@dataclass(frozen=True)
class PickerLayout:
    """Every geometry quantity the picker draws with, for one ``(width, height)``.

    A frozen dataclass rather than loose returns because these values must be
    mutually consistent: the name width, the breakpoint and the row budget are
    three views of one arithmetic, and computing them at separate call sites is
    what let design round 2 argue D16 on paper while the rendered frames
    disagreed. Tests assert against this rather than re-deriving the
    breakpoint arithmetic they are checking.
    """

    mode: str
    screen_width: int
    list_width: int
    preview_width: int
    list_rows: int
    preview_rows: int
    name_width: int
    show_id: bool
    age_width: int
    context_width: int


def plan_layout(width: int, height: int, *, querying: bool = False) -> PickerLayout:
    """The whole picker geometry as a pure function of the terminal size.

    THE PANE ARITHMETIC IS MEASURED, NOT ASSUMED. Textual resolves ``3fr``/
    ``2fr`` against the container's own content box after both panes' padding
    and the preview's ``border-left`` are taken, and it rounds; a sweep of
    ``results.size.width`` across widths 80–240 under the production
    stylesheet matched the expressions below at every one of the 161 widths,
    and matched ``split × cols`` at none of them. Guessing here wrapped every
    row onto a second line at 80 columns.

    The row budget comes from ``height`` — the app's size — rather than from a
    widget's measured height, because a widget height LAGS the paint: on mount
    it reads one row ahead of the settled layout and after a resize it reads
    the PREVIOUS geometry, which made the footer counter claim ``13 drawn``
    over 12 rendered rows at 80x24.
    """
    width = max(1, width)
    height = max(1, height)
    # `Screen { padding: 1 }` insets the content box by two cells and two rows
    # before anything below is resolved.
    inner_w = max(1, width - 2)
    # One filter row, plus the two screen padding rows.
    cols_h = max(1, height - 3)
    mode = "side-by-side" if width >= STACK_BELOW_COLS else "stacked"

    if mode == "side-by-side":
        # `fr` resolution rounds DOWN on the first pane; the preview then takes
        # the remainder less its own `border-left`.
        outer_list = inner_w * LIST_FR // (LIST_FR + PREVIEW_FR)
        list_width = outer_list - 2
        preview_width = inner_w - outer_list - 1 - 2
        list_rows = cols_h
        preview_rows = cols_h
    else:
        list_width = inner_w - 2
        preview_width = inner_w - 2
        if cols_h >= LIST_MIN + PREVIEW_MIN:
            preview_rows = min(PREVIEW_MAX, max(PREVIEW_MIN, cols_h // 3))
        else:
            # Too short for both. Chrome is reserved first and the list takes
            # what is left and scrolls, rather than the preview collapsing to a
            # header with no conversation under it.
            preview_rows = max(4, cols_h - LIST_MIN)
        list_rows = max(1, cols_h - preview_rows)

    list_width = max(12, list_width)
    preview_width = max(12, preview_width)

    # ONE definition of the name width, and nothing else computes it. The
    # prototype's own note says inlining it meant D16 could only be argued on
    # paper — the invariant is a comparison between two layouts at the SAME
    # width, so the compared quantity needs a single source of truth.
    # THE SOFT GUTTER IS RESERVED UNCONDITIONALLY, AND NEVER OUT OF THE NAME.
    # Both halves of that are what keep D16 true on the query path.
    #
    # Charging it only while querying made the reservation a function of the
    # QUERY as well as the width, and the two layouts absorbed it differently:
    # stacked, the raw field is far past NAME_MAX so the cap swallowed the 2
    # cells; side-by-side at the flip the raw field is 65 — close enough to the
    # cap that the same 2 cells came straight out of the name. That is the
    # 64 -> 63 shrink at exactly 158 -> 159 with a query typed: one event, and
    # one a no-query sweep cannot see.
    #
    # Reserving it ALWAYS also stops the name column moving when the user
    # starts typing, which is the same argument `plan_columns` makes for the
    # marker columns: a column that appears and disappears moves every name
    # sideways. `querying` stays in the signature because callers state it and
    # the tests sweep both values, but it no longer changes the arithmetic.
    #
    # Reserving it always moves the breakpoint from 159 to 161, and the
    # CONSTANT FOLLOWS THE MEASUREMENT rather than the other way round — §3.3
    # defines the breakpoint as the width at which side-by-side reaches
    # NAME_MAX, and instructs re-running the sweep after touching any fixed
    # column. This reservation is one. Pinning 159 while the fields say 161
    # would leave the constant and the layout disagreeing, which IS the D16
    # defect rather than a fix for it; the alternative — shedding the id
    # across 159..172 to buy the 2 cells — spends a column the user copies
    # into `/resume <id>` to protect a number.
    del querying
    fixed = GUTTER_CELLS + 2 + AGE_CELLS + SOFT_GUTTER_CELLS
    show_id = (list_width - fixed - 2 - ID_CELLS) >= NAME_P75
    raw_name = list_width - fixed - ((2 + ID_CELLS) if show_id else 0)
    name_width = max(NAME_MIN_CELLS, min(NAME_MAX, raw_name))

    return PickerLayout(
        mode=mode,
        # The filter row spans the container and carries the same `padding: 0 1`
        # the panes do, so its text width is the container less those two cells
        # — MEASURED against the real stylesheet at 80/100/120/140/159/160/180,
        # where a `screen_width == inner_w` model was 2 too generous at every
        # one and the row was clipped mid-word by Textual.
        screen_width=max(1, inner_w - 2),
        list_width=list_width,
        preview_width=preview_width,
        list_rows=max(1, list_rows),
        preview_rows=max(1, preview_rows),
        name_width=name_width,
        show_id=show_id,
        age_width=AGE_CELLS,
        context_width=max(10, list_width - CONTEXT_INDENT),
    )


def fit_rows(costs: Sequence[int], top: int, budget: int) -> int:
    """How many rows from ``top`` fit in a LINE budget of ``budget``.

    Rows are not uniformly one line: a row drawing a grep context line costs
    two, so the window size is a function of WHICH rows are in it.
    """
    used = 0
    count = 0
    for cost in costs[top:]:
        if used + cost > budget:
            break
        used += cost
        count += 1
    return max(1, count)


def scroll_into_window(costs: Sequence[int], top: int, cursor: int, budget: int) -> int:
    """``top`` moved just far enough that ``cursor`` is drawn.

    Scrolls by the minimum, so the list is stable while paging through the
    middle of it — the same rule ``_move_to`` has always applied, restated
    against a LINE budget rather than a row count.
    """
    if cursor < top:
        return cursor
    while cursor >= top + fit_rows(costs, top, budget):
        top += 1
    return top


# ``filter_rows``, ``matched_in_body`` and ``rank_rows`` are RE-EXPORTED from
# ``local_operator.session.session_search`` rather than defined here: the phone
# daemon and the desktop catalogue search the same store, and three copies of
# "what admits a row and what orders it" is how the phone ended up unable to
# find a typo the picker resolves. The definitions and their rationale (the
# tiers, the exact-body-versus-soft split, the recency tie-break) live in that
# one module; this import is what keeps the picker's call sites and its tests
# pointed at exactly one implementation.
from local_operator.session.session_search import (  # noqa: E402  (kept beside its callers)
    filter_rows,
    matched_in_body,
    rank_rows,
    soft_tier_wanted,
)


def _pad_cells(text: str, width: int) -> str:
    """Pad ``text`` to exactly ``width`` CELLS (not characters).

    Wide glyphs — CJK, most emoji — occupy two cells each, so the character
    count a name pads to is not the width it renders at. ``str.ljust`` counts
    characters, which let a CJK name satisfy the pad at half its rendered
    width and push the row past the card, where the ellipsis overflow silently
    ate the age and id columns.
    """
    return text + " " * max(0, width - cell_len(text))


def _wrap_cells(text: str, width: int) -> list[str]:
    """Break ``text`` into lines of at most ``width`` CELLS, on word bounds.

    Cells rather than characters for the same reason :func:`_pad_cells`
    measures in them: a wide glyph occupies two, so a character-counted wrap
    overflows the card on exactly the scripts that can least afford it.

    A single word longer than the width is truncated rather than allowed to
    run past the card, which is the only case where losing text beats breaking
    the layout — every other case keeps all of the words and spends rows.
    """
    if width <= 0:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if cell_len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word if cell_len(word) <= width else truncate_cells(word, width)
    if current:
        lines.append(current)
    return lines or [""]


def row_state_mark(row: SessionRow, frame: int) -> tuple[str, str]:
    """``(glyph, ink)`` for one row's live state. Empty glyph when cold.

    The picker is the one place a user can see the whole fleet, so it is where
    "which of these is actually running, and which one wants me" has to be
    answerable at a glance. Precedence is by URGENCY, not by state machine:
    needs-you outranks everything (a person is blocked), then wedged (broken),
    then busy, then an ARMED wake, then the runtime's own presence.

    **An armed wake outranks the IDLE glyph**, which is a change from the
    original ordering and is what the user asked for: "show the wake symbol if
    it's just scheduled wakes, or the circle icon that there's a runtime but no
    activity". ``●`` idle is the least informative thing true of a row — every
    resident session has it — while "this one will act on its own at some
    point" is a fact about the future that nothing else on the row conveys.
    Under the old order the wake glyph was unreachable for any live session,
    because a session with a wake armed is by definition resident and
    ``idle``/``attached`` matched first; it could only ever appear on a COLD
    row, i.e. one whose wake had no runtime to fire in.

    **``attached`` stays ABOVE the wake**, because the "least informative"
    argument does not extend to it (round 1, D2). ``○`` does not mean merely
    "resident"; it means *a terminal is watching this session*, which on a list
    the user is scanning is the one mark that answers "where am I?". A wake is
    worth more than bare residency and less than presence.

    **A DORMANT wake does not**, and stays below presence. ``wakes_dormant``
    means the session was deliberately stopped, so the schedule is not going to
    fire; promoting it would advertise a future that is not coming, over a
    runtime that is genuinely here. On a cold row it still renders (dimmed) as
    the last thing worth saying about the session.

    The spinner reuses ``terminal_title.SPINNER_FRAMES`` rather than a second
    animation vocabulary — the same glyphs the band and the terminal title
    already animate with, so "this is working" looks the same everywhere.

    ``live_state == "busy"`` is now the CONVERSATION's activity rather than the
    runtime's residency (see ``ServingSessionHandle.is_conversationally_active``),
    which is what makes the spinner honest: it had been pinned on by any
    background job or subagent the session had ever launched.
    """
    if row.pending:
        return NEEDS_YOU_MARKER, "warning"
    if row.live_state == "wedged":
        return WEDGED_MARKER, "danger"
    if row.live_state == "busy":
        return SPINNER_FRAMES[frame % len(SPINNER_FRAMES)], "accent"
    if row.live_state == "attached":
        return ATTACHED_MARKER, "muted"
    if row.wakes and not row.wakes_dormant:
        return WAKE_MARKER, "muted"
    if row.live_state == "idle":
        return IDLE_MARKER, "muted"
    if row.wakes:
        return WAKE_MARKER, "dim"
    return "", "dim"


def sort_needs_you_first(rows: Sequence[SessionRow]) -> list[SessionRow]:
    """Rows with a parked question first, everything else in the given order.

    The ONE marker that reorders. A parked gate is a person being waited on
    and a runtime held resident until they answer; burying it under thirty
    recent conversations is how a session stays parked for a day. Stable
    otherwise, so the recency order the caller established is preserved within
    each group.
    """
    waiting = [row for row in rows if row.pending]
    rest = [row for row in rows if not row.pending]
    return waiting + rest


def plan_columns(
    rows: Sequence[SessionRow],
    width: int,
    ages: Sequence[str],
    marked: bool = False,
    forked: bool = False,
    stated: bool = False,
    tagged: bool = False,
) -> tuple[int, int, int]:
    """``(name, age, id)`` cell budgets for ``width``, dropping before cutting.

    A column that does not fit is removed, never truncated. The id is dropped
    first: a cut hex id still LOOKS like an id, and it is the one field a user
    copies into ``/resume <id>``. The age goes second — "4h" with the "ago"
    sliced off is noise. The name is last and truncates with an ellipsis,
    because a prefix of a sentence is still recognisable.

    ``marked`` reserves the body-match marker's cells as part of the FIXED
    chrome, for every row in the list rather than only the matched ones. Two
    reasons, and both were found by looking at rendered frames rather than at
    the arithmetic:

    * Subtracting the marker from the name AFTER this function had already
      spent the budget down to :data:`NAME_MIN_CELLS` rendered marked names at
      14 cells — under the floor this module documents as "not identifiable",
      and it let the marker jump a queue in which the id and the age are
      supposed to surrender their cells before the name gives up any.
    * Reserving it only on matched rows started names at a different column
      depending on how each row matched, so a filtered list rendered a ragged
      left edge for the one field the user is reading down.

    ``forked`` reserves :data:`FORK_MARKER`'s cells on exactly the same terms,
    for exactly the same two reasons. It is asked of the RESULT SET rather than
    of the page for the scroll-stability argument recorded below: a column that
    appears as a forked row scrolls into view and disappears as it scrolls out
    makes every name on the list jump sideways on one arrow press.

    ``tagged`` reserves :data:`EXEC_MARKER`'s cells identically. It is a
    SEPARATE budget from ``forked`` rather than one shared "qualifier" column
    because the two facts are independent — a forked session can be running
    under exec — and sharing the cells would make one tag hide the other.

    Reserved as FIXED CHROME rather than subtracted from the name afterwards,
    which is what keeps the drop ladder honest — the id surrenders its cells
    before the age, and the age before the name, and a marker that helped
    itself to the name's budget after the fact would jump that queue and could
    push a name under :data:`NAME_MIN_CELLS`.
    """
    marker_col = cell_len(BODY_MATCH_MARKER) if marked else 0
    marker_col += cell_len(FORK_MARKER) if forked else 0
    marker_col += cell_len(EXEC_MARKER) if tagged else 0
    # The live-state column follows the same reserve-for-the-RESULT-SET rule as
    # the two above, and for the same reason: a column that appears as a
    # running row scrolls into view makes every name jump sideways on one
    # arrow press.
    marker_col += STATE_COL_CELLS if stated else 0
    age_col = max((cell_len(age) for age in ages), default=0)
    # Measured rather than assumed at 12: an id written by an older build with
    # a different length must still line up instead of ragging the column.
    id_col = max((cell_len(row.id) for row in rows), default=0)
    fixed = GUTTER_CELLS + marker_col + 2 + age_col + 2 + id_col
    if width - fixed >= NAME_MIN_CELLS:
        return width - fixed, age_col, id_col
    fixed = GUTTER_CELLS + marker_col + 2 + age_col
    if width - fixed >= NAME_MIN_CELLS:
        return width - fixed, age_col, 0
    return max(NAME_MIN_CELLS, width - GUTTER_CELLS - marker_col), 0, 0


def render_rows(
    rows: Sequence[SessionRow],
    selected: int,
    width: int,
    now: float,
    hovered: int | None = None,
    body_matched: AbstractSet[str] = frozenset(),
    forked: bool | None = None,
    frame: int = 0,
    tagged: bool | None = None,
    name_max: int | None = None,
    age_width: int | None = None,
    show_id: bool | None = None,
    soft_gutter: bool = False,
    exact_matched: AbstractSet[str] = frozenset(),
) -> list[Text]:
    """One line per session: cursor, name, age, id.

    Every style here is at least the ``dim`` step. The first cut put the ids,
    the ages and the whole key footer at ``faint``, which is 1.49:1 against
    this card's raised ground — the ramp is calibrated against the app's own
    background, and an overlay lifts the ground two steps without lifting the
    text with it.
    """
    fg = theme_mod.semantic_color("fg")
    muted = theme_mod.semantic_color("muted")
    dim = theme_mod.semantic_color("dim")

    ages = [format_age(max(0.0, now - row.mtime)) for row in rows]
    # Whether the RESULT SET has any marked row decides the column, not whether
    # this PAGE does. `rows` here is one page of a scrolling list, so asking it
    # made the reservation appear and disappear as the marked row scrolled in
    # and out of view: every name jumped two cells sideways on a single arrow
    # press, and truncation changed for rows that had not changed. A column
    # that depends on scroll position is D2's ragged edge moved onto the time
    # axis, where it is worse — motion draws the eye, a static offset does not.
    marked = bool(body_matched)
    # Whether the RESULT SET carries a fork decides the fork column, on the
    # same page-versus-result-set argument as `marked` above: `rows` here is
    # one page of a scrolling list, so asking it would move every name two
    # columns sideways as a fork scrolled past. The picker therefore passes
    # the result-set fact; the default (None) is only for callers that have
    # no paging — tests, a one-page list — and then the page IS the set.
    any_forked = (
        bool(forked) if forked is not None else any(getattr(row, "forked", False) for row in rows)
    )
    # Same result-set question as `any_forked`, same scroll-stability reason.
    any_stated = any(
        getattr(row, "live_state", "") or getattr(row, "pending", None) or getattr(row, "wakes", 0)
        for row in rows
    )
    # And again for the kind tag. Asked of the result set, not the page, for the
    # third time and the same reason — but it matters MORE here than for a fork:
    # an exec record is ephemeral, so this column would otherwise appear and
    # vanish on its own as a one-shot is reaped, moving every name sideways
    # without the user touching anything.
    #
    # Result-set scoping alone is NOT ENOUGH here, and that is the difference
    # from `forked`. It stabilises the column against SCROLLING, because the
    # result set does not change as the page moves. It cannot stabilise it
    # against REAPING: when the one-shot ends, the result set itself loses the
    # tagged row, `any_tagged` flips to False, and every name shifts 7 cells
    # left with no keystroke (round 1, D1 — measured `(67,6,12) → (76,6,12)`).
    # That fires on the NORMAL END OF EVERY ONE-SHOT, so the first time a user
    # observes the ephemerality this tag exists to explain, the feedback is the
    # whole list lurching — the exact "reads as a bug" reaction `EXEC_MARKER`
    # was added to prevent. So the SCREEN latches the fact for the lifetime of
    # the open picker and passes it here; `None` keeps the derive-from-rows
    # behaviour for callers with no paging and no lifetime to latch against
    # (tests, a one-page list), exactly as `forked` above does.
    any_tagged = (
        bool(tagged)
        if tagged is not None
        else any(getattr(row, "kind", "") in TAGGED_KINDS for row in rows)
    )
    name_col, age_col, id_col = plan_columns(
        rows, width, ages, marked, any_forked, any_stated, any_tagged
    )
    # The SCREEN's layout overrides the drop-ladder's own arithmetic when it
    # supplies one, so the two panes agree about where the name ends. The
    # ladder still runs first and still owns the narrow cases — this only caps
    # what it produced (D16's saturation) and fixes the age column, which must
    # be one right-aligned column across every drawn row rather than sized to
    # whichever ages happen to be in view (D3: 13 distinct start columns).
    if age_width is not None:
        age_col = age_width if age_col else 0
    if show_id is not None and not show_id:
        id_col = 0
    if name_max is not None:
        name_col = min(name_col, name_max)
    # The soft-match ``~`` gets a RESERVED gutter: appending it unreserved
    # overflowed the pane and wrapped the mark onto its own line, breaking the
    # one-row-per-session arithmetic the cursor depends on.
    soft_col = SOFT_GUTTER_CELLS if soft_gutter else 0
    marker_col = cell_len(BODY_MATCH_MARKER) if marked else 0
    fork_col = cell_len(FORK_MARKER) if any_forked else 0
    state_col = STATE_COL_CELLS if any_stated else 0
    exec_col = cell_len(EXEC_MARKER) if any_tagged else 0

    lines: list[Text] = []
    for index, (row, age) in enumerate(zip(rows, ages)):
        current = index == selected
        # A ground behind the whole row, as the command picker paints: a bare
        # caret gives a mouse user almost nothing, and the ground is the only
        # selection signal an unnamed row would otherwise have.
        if current:
            ground = theme_mod.semantic_color(
                "tint-select-hi" if index == hovered else "tint-select"
            )
        elif index == hovered:
            ground = theme_mod.semantic_color("tint-select")
        else:
            ground = theme_mod.semantic_color("overlay")
        row_bg = Style(bgcolor=ground)

        line = Text(no_wrap=True, overflow="ellipsis")
        # The caret is MUTED, like both sibling pickers — command_picker's D17
        # note gives the reason and ask_picker restates it: the row GROUND says
        # "selected", so the mark only has to point. It was `label`, the ramp's
        # violet meta ink for tips and skill labels, which said "meta" where the
        # frame meant "position", made the one cool mark on a warm card, and on
        # paper measured 4.45:1 on `tint-select` — under AA on the one row being
        # read. `muted` is 7.53:1 dark / 6.37:1 light there (D5).
        line.append(
            _pad_cells(CURSOR if current else "", GUTTER_CELLS),
            style=row_bg + Style(color=muted),
        )
        # An unnamed session is one whose transcript could not be read or that
        # has no user turn yet. Saying so beats an empty cell, which reads as a
        # rendering fault. It takes the SAME selection step as a named row —
        # pinning it to the floor made selecting it darker than every
        # unselected row, so the highlight inverted.
        name = row.name or "(unnamed session)"
        if row.name:
            name_colour = fg if current else muted
        else:
            name_colour = muted if current else dim
        # A row that matched inside the conversation carries a mark, because
        # its NAME does not contain what was typed and an unexplained row makes
        # the whole result set read as broken.
        #
        # The COLUMN is reserved for every row whenever any row is marked (see
        # ``plan_columns``), and unmarked rows pad it with blanks. Painting it
        # only where it applies moved the start of the name between rows, which
        # ragged the left edge of the one field being read down the list.
        #
        # `muted`, not `dim`. The two are interchangeable for the id and the
        # age, which are redundant lookup keys, but this mark is the ONLY thing
        # explaining why a row with no visible match is in the results — and
        # `dim` measures 3.43:1 dark / 2.72:1 light on this card's raised
        # ground, under AA. `muted` is 6.51:1 / 5.18:1, and is already the
        # caret's ink for exactly this argument (D3).
        if marker_col:
            line.append(
                _pad_cells(BODY_MATCH_MARKER if row.id in body_matched else "", marker_col),
                style=row_bg + Style(color=muted),
            )
        # A fork still wearing its parent's title is otherwise a byte-identical
        # row to the parent — same name, same age — separable only by a hex id,
        # and that is precisely the moment a user opens this picker looking for
        # one of the two. The tag clears the instant the fork writes its own
        # name (``forked`` is only set while the title is inherited), so it
        # marks the ambiguous STATE rather than permanently labelling a session
        # by its ancestry.
        #
        # The INHERITED TITLE IS KEPT beside it, rather than the row reading
        # "[fork] untitled": for a fork made seconds ago the borrowed title is
        # the only text on the row that says which conversation this branched
        # from, and it is how the user recognises it. The tag says the title is
        # borrowed; it does not have to replace it.
        #
        # `dim`, NOT the name's own ink. The shipped suffix painted at name
        # weight and read as part of the name — as though the conversation were
        # called "Refactor the YAML loader (fork)". This is metadata about the
        # row, so it takes the ink the age and the id already use, which is the
        # correct signal for a lookup key. Deliberately a step quieter than
        # BODY_MATCH_MARKER's `muted`: that mark is load-bearing (it is the only
        # thing explaining why an unmatched row is in the results), whereas this
        # one qualifies a name the user is already reading.
        if fork_col:
            line.append(
                _pad_cells(FORK_MARKER if getattr(row, "forked", False) else "", fork_col),
                style=row_bg + Style(color=dim),
            )
        # Beside the fork tag and painted the same `dim`, for the reason given
        # there: this is metadata ABOUT the row, and at name weight it would
        # read as part of the conversation's title.
        if exec_col:
            line.append(
                _pad_cells(
                    EXEC_MARKER if getattr(row, "kind", "") in TAGGED_KINDS else "", exec_col
                ),
                style=row_bg + Style(color=dim),
            )
        # The live-state mark sits immediately before the name, where the eye
        # scanning the name column passes it anyway. Its ink is the state's own
        # semantic colour rather than a fixed one: `warning` for needs-you and
        # `danger` for wedged are the two the user must not miss, and painting
        # them at `dim` beside the age would file a blocked session as a lookup
        # key.
        if state_col:
            glyph, ink = row_state_mark(row, frame)
            line.append(
                _pad_cells(glyph, state_col),
                style=row_bg + Style(color=theme_mod.semantic_color(ink)),
            )
        line.append(
            _pad_cells(truncate_cells(name, name_col), name_col),
            style=row_bg + Style(color=name_colour),
        )
        if age_col:
            line.append("  ", style=row_bg)
            line.append(age.rjust(age_col), style=row_bg + Style(color=dim))
        if id_col:
            line.append("  ", style=row_bg)
            line.append(row.id, style=row_bg + Style(color=dim))
        if soft_col:
            # A soft/fuzzy hit has no literal substring to locate, so it gets
            # this mark and no context line. Keyed to the row being a body
            # match WITHOUT an exact hit, so a row whose context is merely not
            # drawn (stacked, off the cursor) is not mismarked as fuzzy.
            soft = row.id in body_matched and row.id not in exact_matched
            line.append(
                _pad_cells(f" {SOFT_MATCH_MARKER}" if soft else "", soft_col),
                style=row_bg + Style(color=theme_mod.semantic_color("warning")),
            )
        lines.append(line)
    return lines


def _content_width(widget: object) -> int:
    """Cells available INSIDE ``widget``'s own padding, or 0 when unresolved.

    Tolerant by design: the repaint-signature tests substitute a minimal stub
    for a pane, and an unmounted screen has no resolved geometry at all.
    Neither is an error — both mean "no measurement yet" and the caller falls
    back to the planned size.
    """
    if widget is None or not getattr(widget, "is_mounted", False):
        return 0
    region = getattr(widget, "content_region", None)
    if region is not None and getattr(region, "width", 0):
        return int(region.width)
    return 0


def _content_height(widget: object) -> int:
    """Rows available INSIDE ``widget``'s own border and padding, or 0.

    Tolerant in the same way and for the same reasons as :func:`_content_width`.
    """
    if widget is None or not getattr(widget, "is_mounted", False):
        return 0
    region = getattr(widget, "content_region", None)
    if region is not None and getattr(region, "height", 0):
        return int(region.height)
    return 0


def _widget_size(widget: object, axis: str) -> int:
    """``widget``'s resolved width/height, or 0 when it has none to report.

    Tolerant by design: the repaint-signature tests substitute a minimal stub
    for a pane to read what was painted, and a screen that has not been mounted
    has no resolved geometry at all. Neither is an error — both simply mean
    "no measurement yet", and the caller falls back to the planned size.
    """
    if widget is None or not getattr(widget, "is_mounted", False):
        return 0
    size = getattr(widget, "size", None)
    return int(getattr(size, axis, 0) or 0)


def _highlight(text: str, query: str) -> Text:
    """``text`` with every case-insensitive run of ``query`` lifted to amber-bold.

    THE HIGHLIGHT IS THE ENTIRE MECHANISM of the context line: design round 1
    identified the amber-bold hit as what makes the eye land on the match, and
    round 3 verified 28 of 28 populated context lines carrying it.
    """
    out = Text(text, style=Style(color=theme_mod.semantic_color("dim")))
    needle = query.strip().lower()
    if not needle:
        return out
    haystack = text.lower()
    start = haystack.find(needle)
    while start >= 0:
        out.stylize(
            Style(color=theme_mod.semantic_color("warning"), bold=True),
            start,
            start + len(needle),
        )
        start = haystack.find(needle, start + len(needle))
    return out


def _short_model(checkpoint: dict[str, object]) -> str:
    """``anthropic/claude-opus-5`` → ``claude-opus-5``."""
    effective = checkpoint.get("effective_model")
    model = (effective or {}).get("model_id") if isinstance(effective, dict) else ""
    return str(model).rsplit("/", 1)[-1] if model else "·"


def _short_cwd(checkpoint: dict[str, object]) -> str:
    """``/Users/x/workspace`` → ``~/workspace``."""
    cwd = str(checkpoint.get("cwd") or "")
    if not cwd:
        return "·"
    home = str(Path.home())
    return "~" + cwd[len(home) :] if cwd.startswith(home) else cwd


class SessionPickerScreen(ModalScreen[str | None]):
    """Pick a conversation to resume; dismisses with its id, or ``None``.

    Two-way by construction: the caller pushes the screen and acts on what it
    returns, so the picker owns navigation and the caller owns resuming. Esc
    answers ``None`` and the session on screen is left exactly as it was.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "choose", "Resume", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        # Ctrl+P/Ctrl+N as well as the arrows: every printable key belongs to
        # the filter, so the readline pair is the only other way to move a
        # hand that is already typing.
        Binding("ctrl+p", "move(-1)", "Up", show=False),
        Binding("ctrl+n", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("backspace", "backspace", "Edit filter", show=False),
        # EVERY NEW AFFORDANCE IS A CHORD, never a letter: printable keys type
        # into the filter (``on_key``) and that stays true.
        #
        # ``ctrl+e`` is this codebase's established reveal chord —
        # ``ask_picker.py:559`` binds it to ``toggle_reveal`` for exactly this
        # "show me the full text" gesture — so this adds a shortcut rather than
        # a second vocabulary. ``ctrl+u``/``ctrl+d``/``ctrl+g`` are listed in
        # the composer keymap and bound at app level, but a ``ModalScreen`` on
        # top owns its own bindings while it is up and the composer does not
        # have focus behind it; this picker already rebinds ``pageup``/
        # ``pagedown``/``home``/``end`` on the same terms. Verified against the
        # running app in a pty, not against the binding tables: with the picker
        # open all four act on the picker, nothing reaches the composer, and
        # ``ctrl+d`` does not quit.
        Binding("ctrl+e", "toggle_verbose", "Verbose preview", show=False),
        Binding("ctrl+u", "pane_scroll(-1)", "Preview up", show=False),
        Binding("ctrl+d", "pane_scroll(1)", "Preview down", show=False),
        Binding("ctrl+g", "pane_end", "Newest turn", show=False),
    ]

    def __init__(
        self,
        rows: Sequence[SessionRow],
        now: float,
        digests: dict[str, str] | None = None,
        refresh_live_state: Callable[[list[SessionRow]], list[SessionRow]] | None = None,
    ) -> None:
        super().__init__()
        self._all = list(rows)
        self._now = now
        self._query = ""
        self._selected = 0
        self._offset = 0
        self._hovered: int | None = None
        # ``{session id: conversation digest}``, built by the caller before the
        # screen is pushed (``search_index.build_index``). Optional so a host
        # without an index — tests, embedders — gets the name-and-id filter
        # unchanged instead of an error.
        self._digests = dict(digests or {})
        # Soft matching reruns on every keystroke; a per-screen index caches each
        # digest's token set (and a deduplicated vocabulary over them) so the
        # bounded edit-distance search costs ~13 ms per query change at real
        # store scale instead of the ~185 ms a stateless re-tokenise-everything
        # call costs there. Owned by the screen so the cache lives exactly as
        # long as the picker and is discarded with it.
        self._soft_index = SoftSearchIndex()
        # Filtering runs on every keystroke and again on every paint; the
        # result is cached against the query that produced it so a card with
        # several hundred sessions does not re-scan the list per repaint. The
        # body matches are cached on the SAME key, because they are recomputed
        # by the same keystroke and scanning 200 digests per repaint is the
        # cost this cache exists to avoid.
        self._filtered: list[SessionRow] = list(rows)
        self._filtered_for = ""
        # ``_body_matches`` is the EXACT-body match set; ``_admitted`` is the
        # union of exact-body and bounded-soft matches that ``filter_rows``
        # admits a row on. Two sets rather than one because they answer
        # different questions: ``_admitted`` decides whether a row is SHOWN,
        # while ``_body_matches`` (exact only) decides its ranking TIER and
        # feeds the body-match marker. Cached on the same key as the filter,
        # because the same keystroke recomputes all three.
        self._body_matches: set[str] = set()
        self._admitted: set[str] = set()
        self._body: Static
        #: Spinner phase for the running marker, advanced by ``_tick``.
        self._frame = 0
        #: ``time.monotonic()`` of the last liveness refresh, or ``-inf`` so the
        #: first tick always refreshes. The picker's data freshness is a stated
        #: bound (``LIVE_REFRESH_INTERVAL_S``) rather than a side effect of the
        #: frame rate; see ``_tick``.
        self._live_refreshed_at = float("-inf")
        #: Has this picker EVER shown an exec row? Latched, never cleared while
        #: the screen lives — see :meth:`_exec_column_latched` for why the
        #: column may widen but must not narrow.
        self._saw_tagged = False
        #: CONDENSED IS THE DEFAULT. Condensed content alone is median 1,346
        #: chars and p90 6,789 against a 60x40 pane holding ~2,400, so 22 of 60
        #: sessions overflow the pane with tool calls already stripped — a
        #: verbose default is a wall the user escapes from on most rows.
        #: Sticky for the LIFETIME of the open picker rather than per row: a
        #: mode that resets as the cursor moves is one the user re-sets on
        #: every row.
        self._verbose = False
        #: The preview's scroll offset, moved by ``ctrl+u``/``ctrl+d``/``ctrl+g``
        #: and deliberately INDEPENDENT of the list cursor.
        self._pane_top = 0
        #: Bounded, cached preview reads, built lazily and discarded with the
        #: screen. ``None`` until first use so a picker that never paints a
        #: preview — a test host, an embedder — pays nothing.
        self._preview_data: SessionPreviews | None = None
        self._sessions_dir: Path | None = None
        #: The layout last APPLIED to the widget tree, and the stacked preview
        #: height that went with it. Both are part of the restyle guard; see
        #: :meth:`_apply_layout`. ``None`` forces the first application on
        #: mount regardless of which side of the breakpoint we start on.
        self._applied_mode: str | None = None
        self._applied_pane_rows: int | None = None
        #: Re-reads each row's live state, supplied by the host that knows how
        #: (``OperatorApp._overlay_live_state``). Optional: a host that does
        #: not pass one gets the pre-refresh behaviour — markers from open,
        #: and no animation — which is what keeps this widget testable without
        #: a registry and usable by an embedder that has none.
        self._refresh_live_state = refresh_live_state

    # -- state ---------------------------------------------------------------
    # ``visible_rows``/``filter_query``/``_card_text``, not ``visible``/``query``/
    # ``_render``: all three of the shorter names are already Textual's
    # (``Widget.visible``, the ``DOMNode.query`` selector method, and the
    # internal ``Widget._render``), and shadowing them breaks the framework's
    # own focus, query and paint paths from inside the screen.
    @property
    def visible_rows(self) -> list[SessionRow]:
        """The rows the current filter admits, ranked by relevance to the query.

        Empty query -> recency order, unchanged. A non-empty query re-ranks the
        admitted subset by :func:`rank_rows` (name > id > body > soft, recency
        tie-break) and, in the same step that ``set_query`` re-homes the cursor
        to index 0, so the cursor tracks the best match rather than a row that
        ranking then slides away from. Ordering is a pure function of
        ``(rows, query, digests)`` — no run history, no memory of previous
        keystrokes — so a FIXED query never reorders across repaints or resizes,
        AND two routes to the same query produce the same order. Verified on the
        real store across a 20-word list: 0 route divergences.
        """
        if self._filtered_for != self._query:
            # Exact-body hits and bounded-soft hits are computed separately: the
            # union decides which rows are shown, the exact set decides ranking
            # tier and the body-match marker. Both are recomputed only on a
            # query change, never per repaint — scanning 200 digests per paint
            # is the cost this cache exists to avoid.
            #
            # This call is UNLOCKED and shares ``search_index``'s process-wide
            # memo with the server's locked path (``session_search``). Safe only
            # because the picker is a single thread and no process hosts it
            # beside a shared-path caller — the TUI does not run the server
            # in-process. An embedder that did both would need to take
            # ``session_search._SHARED_LOCK`` around this line, because the memo
            # is one entry that two threads can interleave.
            self._body_matches = search_digests(self._digests, self._query)
            # The soft tier is expensive on its first call for a given store —
            # it tokenises every digest and builds a vocabulary over them — so
            # it is not run on every keystroke. WHEN it runs is decided by
            # ``_soft_tier_wanted`` below, which exists because the obvious
            # answers are all wrong in ways that were measured on this surface.
            admitted = filter_rows(self._all, self._query, self._body_matches)
            if self._soft_tier_wanted(self._query):
                soft = self._soft_index.search(self._digests, self._query)
                self._admitted = self._body_matches | soft
                # Recomputed only on the soft branch: on the common path the
                # first pass is already the answer, so the uncapped row list is
                # scanned once per query change rather than twice.
                admitted = filter_rows(self._all, self._query, self._admitted)
            else:
                self._admitted = set(self._body_matches)
            self._filtered = rank_rows(admitted, self._query, self._body_matches)
            self._filtered_for = self._query
        return self._filtered

    def _soft_tier_wanted(self, query: str) -> bool:
        """Whether to run the bounded soft tier for ``query``.

        Delegates to ``session_search.soft_tier_wanted`` — the gate, the tier
        floor it counts against and the name/id test it mirrors all live there,
        with the measurements behind the floor. This exists as a method only
        because the caching in :attr:`visible_rows` reads better with the store
        it is gating in scope; the picker holds rows the phone and the desktop
        do not, which is why the predicate takes them as an argument.
        """
        return soft_tier_wanted(self._all, query)

    @property
    def body_matched_ids(self) -> set[str]:
        """Ids on screen because their CONVERSATION matched, not their name.

        Reads through :attr:`visible_rows` rather than the cached set directly,
        so the two can never answer for different queries.

        Keyed on ``_admitted`` (exact-body OR soft), not the exact-body set: a
        row surfaced only because a PAST name or a typo/prefix matched is just
        as much "found on something other than the visible name" as an exact
        body hit, and the marker means exactly that — otherwise a soft or
        past-name hit reads as the filter returning an arbitrary row.
        """
        rows = self.visible_rows
        return {row.id for row in rows if matched_in_body(row, self._query, self._admitted)}

    @property
    def filter_query(self) -> str:
        return self._query

    @property
    def selected_index(self) -> int:
        return self._selected

    def selected_id(self) -> str | None:
        """The highlighted session's id, or ``None`` when nothing matches."""
        rows = self.visible_rows
        if not rows:
            return None
        return rows[min(self._selected, len(rows) - 1)].id

    # -- actions -------------------------------------------------------------
    def _dismiss_result(self, result: str | None) -> None:
        """Dismiss after releasing a hovered row's pointer shape."""
        # The modal leaves without another mouse move; make the inline rule's
        # observer restore OSC 22 while the screen still owns the pointer.
        self.styles.pointer = "default"
        self.dismiss(result)

    def action_cancel(self) -> None:
        self._dismiss_result(None)

    def action_choose(self) -> None:
        # Enter on an empty result set is not a choice. Dismissing with None
        # here (rather than ignoring the key) means Enter always closes the
        # picker, which is what a user who has typed a bad filter expects.
        self._dismiss_result(self.selected_id())

    def action_move(self, delta: int) -> None:
        self._move_to(self._selected + delta)

    def action_page(self, delta: int) -> None:
        self._move_to(self._selected + delta * self._page_rows())

    def action_jump(self, to_end: int) -> None:
        self._move_to(len(self.visible_rows) - 1 if to_end else 0)

    def action_toggle_verbose(self) -> None:
        """Flip condensed/verbose. Sticky until the picker closes."""
        self._verbose = not self._verbose
        self._pane_top = 0
        self._repaint()

    def action_pane_scroll(self, direction: int) -> None:
        """Scroll the preview WITHOUT moving the list cursor."""
        self._pane_top = max(0, self._pane_top + direction * PREVIEW_SCROLL_LINES)
        self._repaint()

    def action_pane_end(self) -> None:
        """Jump the preview to the newest turn."""
        self._pane_top = max(0, len(self._preview_lines()) - self._pane_height())
        self._repaint()

    def action_backspace(self) -> None:
        if self._query:
            self.set_query(self._query[:-1])

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Printable keys type into the filter.

        Handled here rather than as bindings because the filter accepts every
        character; a binding per key would be a table of ninety-five entries
        that still missed the ninety-sixth.
        """
        char = event.character
        if char is not None and char.isprintable() and len(char) == 1:
            event.stop()
            event.prevent_default()
            self.set_query(self._query + char)

    # -- mouse ---------------------------------------------------------------
    # The wheel moves the cursor a row at a time, which scrolls the window with
    # it (``_move_to`` keeps the selection on screen). Clamped, like every
    # other movement here: a scroll gesture that wrapped to the other end of
    # the list would read as the picker resetting itself. Every handler stops
    # the event so one gesture does not also scroll the transcript behind.
    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self.action_move(1)

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self.action_move(-1)

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Primary-button click on a row resumes it.

        This card invited the mouse in with the wheel; a list you can scroll
        with the mouse and cannot click with it is a half-built affordance.

        Button 1 only. The action behind this disposes the live session and
        reboots, which is not something a right-click asking for a context
        menu, or a stray middle-click paste, should be able to trigger.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._index_at(event)
        if index is None:
            return
        event.stop()
        rows = self.visible_rows
        if 0 <= index < len(rows):
            self._selected = index
            self._dismiss_result(rows[index].id)

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            self._repaint()
        # Hand pointer over a row only (a click resumes it); the card's
        # padding and headers keep the default shape. The inline-rule
        # assignment drives `Screen.update_pointer_shape()` through the
        # property's own observer and no-ops when the shape did not change.
        self.styles.pointer = "pointer" if index is not None else "default"

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
        self.styles.pointer = "default"

    def _index_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """List index under a mouse event, or ``None`` anywhere else.

        Measured against the BODY's region rather than the event's own widget:
        the card is one ``Static``, so a click anywhere in it reports a y
        relative to the whole block — header and rule included.

        Three guards, all load-bearing, because this feeds ``on_click`` and a
        false positive there DISPOSES THE LIVE SESSION and reboots onto another
        one. The first cut had none of them: a click on the footer resolved to
        session #12, the blank spacer to #10, and the dimmed backdrop beside
        the card to row 0.

        - the point must be inside the body's region (the modal's backdrop
          covers the whole screen and bubbles clicks from well outside the card,
          including columns to its left where ``y`` alone still looks valid);
        - the row must be inside the DRAWN page, not merely inside the list —
          the footer and the spacer sit below the last row and their offsets
          resolved to real sessions further down the list;
        - and the resulting index must still be a row that exists.
        """
        results = getattr(self, "_results", None)
        if results is None or not results.is_mounted:
            return None
        region = results.region
        if not region.contains(event.screen_x, event.screen_y):
            return None
        line = event.screen_y - region.y - self._header_rows()
        rows = self.visible_rows
        costs = self._row_costs()
        drawn = fit_rows(costs, self._offset, self._layout().list_rows)
        if line < 0:
            return None
        # A ROW THAT DRAWS A CONTEXT LINE OCCUPIES TWO LINES, so the click's
        # line offset is walked against the real per-row costs rather than used
        # as an index. Treating it as an index resolved every click below the
        # first context line to the wrong session — and a false positive here
        # DISPOSES THE LIVE SESSION and reboots onto another one.
        used = 0
        for step in range(drawn):
            index = self._offset + step
            if index >= len(rows):
                return None
            cost = costs[index]
            if used <= line < used + cost:
                return index
            used += cost
        return None

    # -- geometry ------------------------------------------------------------
    def _layout(self) -> PickerLayout:
        """The geometry for the CURRENT terminal size.

        Derived from ``self.app.size`` rather than from a measured widget
        height, because a widget height LAGS the paint: on mount it reads one
        row ahead of the settled layout and after a resize it reads the
        PREVIOUS geometry, which made the footer counter claim ``13 drawn``
        over 12 rendered rows at 80x24.
        """
        try:
            size = self.app.size
            width, height = size.width, size.height
        except Exception:  # pragma: no cover - only before the app has a screen
            width, height = 80, 24
        return plan_layout(max(1, width), max(8, height), querying=bool(self._query.strip()))

    def _usable(self) -> int:
        """The results pane's REAL text width — the ONE definition.

        Measured off the RESOLVED widget rather than off a fraction of the
        app: ``3fr`` of the split is what the row actually gets, and guessing
        it wrapped every row onto a second line at 80 columns. The fallback
        matters on the first paint, before layout resolves.
        """
        measured = _widget_size(getattr(self, "_results", None), "width")
        if measured:
            # The widget's own padding is inside its reported width.
            return max(12, measured - 2)
        return max(12, self._layout().list_width)

    def _context_width(self) -> int:
        """Cells the grep context line gets: the indent off the pane width."""
        return max(10, self._usable() - CONTEXT_INDENT)

    def _pane_width(self) -> int:
        """The preview's REAL text width, measured off the resolved widget."""
        measured = _widget_size(getattr(self, "_preview", None), "width")
        if measured:
            return max(20, measured - 2)
        return max(20, self._layout().preview_width)

    def _pane_height(self) -> int:
        """Body lines the preview can show: its height less its own header.

        The header is the name, the two clocks, an optional ``model · cwd``
        line and the rule, so the reservation is asked of the header the pane
        will actually draw rather than assumed at a constant.
        """
        # The CONTENT region, not `size`: stacked, the pane carries a
        # `border-top`, and `size.height` counts that border row while the
        # text does not get it. Budgeting against `size` left one line too
        # many, so Textual clipped the last one AFTER `clip_to_height` had
        # already run — and what it clipped was the body line under a role
        # header, leaving the orphan `▪ lop` D30 exists to prevent. Measured
        # in a rendered frame at 100x30, not reasoned about.
        measured = _content_height(getattr(self, "_preview", None))
        rows = measured or self._layout().preview_rows
        return max(1, rows - self._preview_header_rows())

    def _page_rows(self) -> int:
        """Session rows the list can actually DRAW right now.

        No ``PAGE_ROWS_MAX``: the row count is bounded only by the terminal.
        That 10-row ceiling drew 10 rows out of 140 at every terminal height —
        7% — while a 60-row terminal has room for 41.
        """
        costs = self._row_costs()
        return fit_rows(costs, self._offset, self._layout().list_rows)

    def _header_rows(self) -> int:
        """Rows above the first session row.

        Zero: the results pane holds rows and their context lines and nothing
        else. The title, the tally and the keys live in the filter row, and the
        preview is a separate widget — which is precisely why the mouse
        hit-test can measure against this pane's own region.
        """
        return 0

    def _exec_column_latched(self, rows: Sequence[SessionRow]) -> bool:
        """Should the exec column be reserved? Once yes, yes until the picker closes.

        The column may WIDEN during a picker's life and must never NARROW, and
        the asymmetry is the whole point. Widening is caused by the user's own
        act — typing a filter that admits an exec row — so the movement has an
        author and reads as a response. Narrowing is caused by a one-shot
        ENDING, which nobody in front of the terminal did, and which happens on
        the normal completion of every ``lop exec`` run rather than in some edge
        case. Round 1 (D1) measured the result: `any_tagged` flips false,
        `exec_col` goes 7 → 0, `plan_columns` `(67,6,12) → (76,6,12)`, and every
        name on screen jumps 7 cells left with no keystroke — under the cursor,
        on a timer. A list that lurches when a row quietly stops being special
        reads as a glitch, which is precisely the reaction `EXEC_MARKER` exists
        to prevent ("a row vanishing under the cursor reads as a bug unless the
        row said what it was"). The reaped row still loses its own 7 characters
        of tag, so the change that actually happened is still visible; what it
        no longer does is move everything else.

        Latched on the SCREEN, not on the row set, because "the result set for
        one open picker" is the span the fixed-chrome rule is really about, and
        a closed-and-reopened picker legitimately starts over from what is live
        then. The cost is one reserved column of 7 cells persisting after the
        last exec row is gone, which is the same tax the column charged a moment
        earlier and which the drop ladder already knows how to shed at narrow
        widths (round 1, D4 — accepted deliberately).
        """
        if any(getattr(row, "kind", "") in TAGGED_KINDS for row in rows):
            self._saw_tagged = True
        return self._saw_tagged

    # -- internals -----------------------------------------------------------
    def set_query(self, query: str) -> None:
        """Apply a filter and put the cursor on the FIRST match.

        Not the nearest surviving row: clamping the old index meant narrowing
        a list usually landed the cursor on the LAST match, so the row Enter
        would take was the least related one still standing. Every finder the
        user has met — fzf, a command palette, this app's own command picker —
        answers a narrowing query with its best match at the top.
        """
        if query == self._query:
            return
        self._query = query
        self._selected = 0
        self._offset = 0
        self._repaint()

    def _move_to(self, index: int) -> None:
        rows = self.visible_rows
        if not rows:
            self._selected = 0
            self._offset = 0
            self._repaint()
            return
        # Clamped, never wrapping: a Down at the bottom that silently returned
        # to the top reads as the list having reset itself.
        self._selected = max(0, min(len(rows) - 1, index))
        # Scroll only far enough to keep the cursor on screen, so the list is
        # stable while paging through the middle of it.
        page = self._page_rows()
        if self._selected < self._offset:
            self._offset = self._selected
        elif self._selected >= self._offset + page:
            self._offset = self._selected - page + 1
        self._offset = max(0, min(self._offset, max(0, len(rows) - page)))
        self._repaint()

    # -- preview -------------------------------------------------------------
    def _previews(self) -> SessionPreviews:
        """The per-picker preview cache, built lazily and discarded with the screen."""
        if self._preview_data is None:
            self._preview_data = SessionPreviews(self._sessions_dir or config_dir() / "sessions")
        return self._preview_data

    def use_previews_for_test(self, sessions_dir: Path) -> None:
        """Point the preview layer at a fixture store. Tests only."""
        self._sessions_dir = Path(sessions_dir)
        self._preview_data = None
        self._repaint()

    def _preview_header_rows(self) -> int:
        """Rows the preview header occupies: name, clocks, optional meta, rule.

        Asked of the header that will actually be drawn, because the
        ``model · cwd`` line is omitted ENTIRELY when there is no checkpoint
        (D7) — reserving a row for it would leave the blank the placeholder was
        removed to avoid.
        """
        return 4 if self._selected_checkpoint() else 3

    def _selected_row(self) -> SessionRow | None:
        rows = self.visible_rows
        if not rows:
            return None
        return rows[min(self._selected, len(rows) - 1)]

    def _selected_checkpoint(self) -> dict[str, object]:
        row = self._selected_row()
        if row is None:
            return {}
        try:
            return self._previews().checkpoint(row.id)
        except Exception:  # pragma: no cover - a broken store must not stop the paint
            return {}

    def _preview_lines(self) -> list[tuple[str, str]]:
        """``(kind, line)`` for the selected row's conversation."""
        row = self._selected_row()
        if row is None:
            return []
        try:
            previews = self._previews()
            turns = previews.verbose(row.id) if self._verbose else previews.condensed(row.id)
            # The body's own two-space indent is subtracted from the wrap
            # budget, not added on afterwards: a line wrapped to the full pane
            # width becomes indent + width once drawn, which is one cell past
            # the pane. Textual then wraps it onto a second row, pushing the
            # real last line out and leaving the role header above it orphaned
            # — the D30 shape, reintroduced from the other end. Measured in a
            # rendered 100x30 frame, where a 94-cell line drew at 96 in a
            # 96-cell pane.
            return wrap_turns(turns, self._pane_width() - PREVIEW_BODY_INDENT, self._pane_height())
        except Exception:  # pragma: no cover - a broken transcript must not stop the paint
            return []

    def _raw_context(self, row: SessionRow) -> str | None:
        """The grep context for ``row``, or ``None`` when it matched only softly.

        D17 — requested at the width it will actually be DRAWN at. Asking for a
        wide window and truncating it into a narrow pane cuts the match off the
        right end: measured, 0 of 9 context lines contained the query.
        """
        query = self._query.strip()
        if not query or row.id not in self._body_matches:
            return None
        return grep_context(self._digests.get(row.id, ""), query, self._context_width())

    def _context_for(self, row: SessionRow) -> str | None:
        """The context line to DRAW for ``row``, or ``None`` when it draws none.

        Stacked, only the CURSOR row draws one: cells are scarce at 12 list
        rows, a line on each would halve the list, and the preview directly
        beneath already shows that row's content.
        """
        if self._layout().mode == "stacked":
            selected = self._selected_row()
            if selected is None or selected.id != row.id:
                return None
        return self._raw_context(row)

    def _row_costs(self) -> list[int]:
        """Lines each visible row occupies: 2 when it draws a context line."""
        return [2 if self._context_for(row) else 1 for row in self.visible_rows]

    # -- rendering -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        # Two panes side by side with the filter row beneath, following
        # ``settings_view.py:894`` (``self._columns = Horizontal(...)``) — the
        # repo's existing two-pane precedent, rather than a new idiom beside it.
        with Container(classes="session-picker"):
            self._results = Static(id="session-picker-results")
            self._preview = Static(id="session-picker-preview")
            # So arrows never silently move focus off the list.
            self._preview.can_focus = False
            self._filter = Static(id="session-picker-filter")
            with Horizontal(id="session-picker-cols"):
                yield self._results
                yield self._preview
            yield self._filter

    def _apply_layout(self) -> str:
        """Restyle ``#session-picker-cols`` in place. Returns the mode.

        NEVER REBUILDS THE WIDGET TREE: remounting the panes loses the
        preview's scroll offset, which the live-resize test pins.

        Guarded on the mode AND the stacked preview height. Mode alone is not
        enough — the preview height is a function of terminal HEIGHT, which
        changes without crossing the width breakpoint, and a mode-only guard
        left a stale 13-row preview after 160x45 → 80x24, starving the list to
        7 rows where 12 is required. Neither value moves on a cursor keypress,
        so this still restyles only on a real geometry change.
        """
        plan = self._layout()
        stacked = plan.mode == "stacked"
        pane_rows = plan.preview_rows if stacked else None
        results = getattr(self, "_results", None)
        if results is None or not results.is_mounted or not self.is_mounted:
            # Nothing to restyle yet; the mode is still the honest answer.
            return plan.mode
        if plan.mode == self._applied_mode and pane_rows == self._applied_pane_rows:
            return plan.mode
        self._applied_mode = plan.mode
        self._applied_pane_rows = pane_rows

        cols = self.query_one("#session-picker-cols")
        # Clearing a border takes the ``("none", colour)`` TUPLE. Assigning
        # ``None`` drops only the inline rule and lets ``DEFAULT_CSS`` back in,
        # and the literal string ``"none"`` reads the existing rule and raises
        # on an edge that never had one. Measured, not style.
        edge = theme_mod.semantic_color("edge")
        if stacked:
            cols.styles.layout = "vertical"
            self._results.styles.width = "100%"
            self._preview.styles.width = "100%"
            self._results.styles.height = "1fr"
            self._preview.styles.height = pane_rows
            self._preview.styles.border_left = ("none", edge)
            self._preview.styles.border_top = ("solid", edge)
        else:
            cols.styles.layout = "horizontal"
            self._results.styles.width = f"{LIST_FR}fr"
            self._preview.styles.width = f"{PREVIEW_FR}fr"
            self._results.styles.height = "1fr"
            self._preview.styles.height = "1fr"
            self._preview.styles.border_top = ("none", edge)
            self._preview.styles.border_left = ("solid", edge)
        return plan.mode

    def on_mount(self) -> None:
        self._repaint()
        # D1+D3, fixed together on purpose. The running marker borrowed the
        # band's spinner GLYPH but nothing advanced it, so it sat on frame 0 —
        # and a frozen braille dot does not read as "busy", it reads as a
        # static bullet, which is the marker for a DIFFERENT state. That
        # collapsed the one distinction the picker exists to make under this
        # release: which of these is actually working right now.
        #
        # The liveness data is refreshed on the SAME tick rather than only the
        # frame index, because animating a snapshot taken at open would be
        # worse than the freeze: motion is a stronger claim of liveness than a
        # still, so a convincing spinner over minutes-old state actively
        # misleads. If we cannot re-read the state we stop animating too (see
        # ``_tick``) — the two must never come apart.
        self._timer = self.set_interval(SPINNER_INTERVAL_S, self._tick)

    def _tick(self) -> None:
        """Advance the spinner, and re-read what the live markers claim.

        TWO RATES, deliberately, because they answer different questions.

        The FRAME advances at ``SPINNER_INTERVAL_S`` (12.5 Hz) while a row is
        busy: that is motion, and motion is what makes the running marker read
        as running rather than as a static bullet.

        The DATA is re-read at ``LIVE_REFRESH_INTERVAL_S`` — the refresh is
        `registry.scan()` (a glob, a JSON parse per record, a `ps` per
        quiet-heartbeat record) plus `read_index()`, and `_tick` used to run it
        unconditionally on every frame, so an idle picker re-scanned the whole
        store 12.5 times a second. That is not a per-open cost and the previous
        docstring's claim that it was "skipped entirely when no row is
        animating" described a guard that did not exist.

        The refresh is NOT skipped outright, because two things it feeds are
        real while nothing spins: the repaint-on-visible-change guard below and
        the rows themselves, which REORDER when a session parks on a
        gate — this release's headline event, and one that arrives with no row
        animating. So it is BOUNDED rather than dropped, and the bound is what
        keeps a frozen marker from claiming to be live: the picker's own
        freshness is now a stated number instead of an accident of the frame
        rate, and at 1 s it is 45× finer than the 45 s heartbeat the live
        states are derived from, which is the resolution they actually change
        at.
        """
        before = self._marker_signature()
        refresh = self._refresh_live_state
        now = time.monotonic()
        if refresh is not None and now - self._live_refreshed_at >= LIVE_REFRESH_INTERVAL_S:
            self._live_refreshed_at = now
            try:
                self._all = list(refresh(self._all))
                # The filter cache is keyed on the query, which has not
                # changed — invalidate it explicitly or the refreshed rows are
                # computed and then thrown away.
                self._filtered_for = "\x00 never a real query"
            except Exception:  # noqa: BLE001 — a stale marker is not worth the picker
                logger.debug("picker could not refresh live state", exc_info=True)
        # REPAINT ON ANY VISIBLE CHANGE, not only while something spins.
        #
        # The refresh REORDERS (`_overlay_live_state` sorts needs-you first)
        # and `_selected` is an index into that order, so skipping the repaint
        # left the screen painted in the old order while Enter resolved
        # against the new one — the cursor sat on `alpha` and Enter resumed
        # `beta` (round 3, D10). That fires on this release's headline event:
        # a detached session parking on a gate sorts itself to the top, and
        # nothing is spinning while it happens. The same early return also
        # froze every non-busy marker transition (idle→wedged, idle→attached,
        # record gone, wake armed).
        #
        # The frame counter still advances only while something is busy, which
        # keeps the property the previous comment wanted: a session that starts
        # working later picks the animation up from a clean phase.
        after = self._marker_signature()
        is_busy = any(getattr(row, "live_state", "") == "busy" for row in self._all)
        if is_busy:
            self._frame += 1
        elif before == after:
            return
        self._repaint()

    #: The `SessionRow` fields a repaint can actually show differently.
    #:
    #: DERIVED FROM THE ROW'S OWN FIELD NAMES, and asserted against them at
    #: import (below), because the round-3 version of this signature read
    #: ``session_id`` — a field `SessionRow` does not have. `getattr` with a
    #: default made that silent: identity was the empty string on EVERY row,
    #: so a pure reorder compared equal, `_tick` returned early, and the
    #: picker went on painting one session while Enter resumed another. That
    #: is D10, unfixed by its own fix, through 83 green picker tests
    #: (round 4, D10).
    #:
    _SIGNATURE_FIELDS = (
        "id",
        "name",
        "forked",
        "live_state",
        "pending",
        "wakes",
        "wakes_dormant",
        "kind",
    )

    #: Fields deliberately OUTSIDE the signature, each with the reason it is
    #: safe to omit. Stated as data rather than as prose because the assertion
    #: below consumes it — an exclusion nobody can name is not an exclusion.
    #:
    #: * `mtime` changes constantly and renders as a coarse "when", so
    #:   including it would repaint ten times a second for nothing.
    #: * `created_at` is immutable for the life of a session (#800 makes it the
    #:   ordering key precisely because it never moves), so it cannot differ
    #:   between two ticks of one open picker.
    _SIGNATURE_EXCLUDED = ("mtime", "created_at")

    # BIDIRECTIONAL, and that is the whole point of it. The previous form
    # checked only that every signature NAME is a real field, which catches a
    # rename but is blind to the opposite and more common error: a field ADDED
    # to `SessionRow`, rendered by this widget, and never added here. `kind`
    # shipped exactly that way (round 1, MAJOR-1/Q1) — the row painted `[exec]`,
    # `_tick` compared two kinds equal, and the frame stayed stale. Both earlier
    # D10 escapes are the same defect in the other direction, so the guard now
    # answers both questions at once: every name must be a field, AND every
    # field must be classified as either signature or documented exclusion.
    # A new `SessionRow` field therefore fails HERE, at the moment it is added,
    # rather than in a frame someone has to notice is stale.
    #
    # THIS ASSERTION AND THE PINNED TESTS ARE BOTH LOAD-BEARING; neither
    # subsumes the other, so do not simplify one away as redundant. This one
    # catches an UNCLASSIFIED field — the omission — at import. It cannot catch
    # a MISCLASSIFIED one: moving `kind` into `_SIGNATURE_EXCLUDED` keeps the
    # union equal and passes here, and only `tests/unit/tui/test_session_picker.py`
    # (which asserts `kind` drives a repaint, and that the exclusions are the
    # two fields whose immutability is argued above) fails. Verified by making
    # exactly that mutation: import succeeds, two tests go red.
    assert set(_SIGNATURE_FIELDS) | set(_SIGNATURE_EXCLUDED) == set(SessionRow._fields), (
        "picker signature is out of step with SessionRow — "
        f"unknown names: {sorted(set(_SIGNATURE_FIELDS) - set(SessionRow._fields))}; "
        "unclassified fields (add to _SIGNATURE_FIELDS, or to _SIGNATURE_EXCLUDED "
        "with the reason it cannot change under an open picker): "
        f"{sorted(set(SessionRow._fields) - set(_SIGNATURE_FIELDS) - set(_SIGNATURE_EXCLUDED))}"
    )

    def _marker_signature(self) -> tuple[tuple[object, ...], ...]:
        """Everything about the rows a repaint would show differently.

        Identity AND order: a reorder with no content change still has to
        repaint, because the cursor is an index into the order (D10). Kept to
        the fields the renderer reads so an unrelated churn (a heartbeat
        timestamp) does not force a repaint ten times a second.
        """
        return tuple(
            tuple(getattr(row, field, None) for field in self._SIGNATURE_FIELDS)
            for row in self._all
        )

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: every column, both pane budgets and the split.

        ``_move_to`` repaints, and ``_repaint`` re-applies the layout — which
        is what flips the split across the breakpoint without rebuilding the
        widget tree and losing the preview's scroll offset.
        """
        self._move_to(self._selected)

    def _repaint(self) -> None:
        results = getattr(self, "_results", None)
        if results is None or not results.is_mounted:
            return
        # The layout may have just flipped, and every measurement below — pane
        # widths, the row budget, the preview's height — depends on which side
        # of the breakpoint we are on.
        self._apply_layout()
        # TWO-PASS FIT. The window size depends on the rows in it (a context
        # line costs a second line) and the cursor clamp depends on the window,
        # so one pass can leave the cursor on an undrawn row — the exact defect
        # ``_page_rows`` exists to prevent. Fit → clamp → fit again.
        budget = self._layout().list_rows
        costs = self._row_costs()
        self._offset = scroll_into_window(costs, self._offset, self._selected, budget)
        self._offset = max(0, min(self._offset, max(0, len(costs) - 1)))
        results.update(self._results_text())
        # Each pane is guarded on its own rather than on `results` alone: the
        # repaint-signature tests stub a single pane to read what was painted,
        # and a partially-composed screen must still paint the panes it has.
        preview = getattr(self, "_preview", None)
        if preview is not None and preview.is_mounted:
            preview.update(self._preview_text())
        filter_row = getattr(self, "_filter", None)
        if filter_row is not None and filter_row.is_mounted:
            filter_row.update(self._filter_text())

    # -- test accessors ------------------------------------------------------
    # Three panes are no longer one list of lines, so the accessors name WHICH
    # pane they answer about. Every row-level assertion in the suite is really
    # about the results pane; the chrome assertions are really about the filter
    # row; and tests must not re-derive the breakpoint arithmetic they check.
    def render_lines_for_test(self) -> list[str]:
        """The RESULTS pane as plain strings: rows and their context lines."""
        return [line.plain for line in self._results_text().split("\n")]

    def render_preview_for_test(self) -> list[str]:
        """The PREVIEW pane as plain strings."""
        return [line.plain for line in self._preview_text().split("\n")]

    def render_footer_for_test(self) -> str:
        """The filter row as one plain string."""
        return self._filter_text().plain

    def layout_mode_for_test(self) -> str:
        """``"stacked"`` or ``"side-by-side"``."""
        return self._layout().mode

    def preview_mode_for_test(self) -> str:
        """``"condensed"`` (the default) or ``"verbose"``."""
        return "verbose" if self._verbose else "condensed"

    def preview_offset_for_test(self) -> int:
        """The preview's scroll offset, which must survive a resize round trip."""
        return self._pane_top

    # -- the three panes -----------------------------------------------------
    def _results_text(self) -> Text:
        """The list pane: one line per session, plus any grep context lines."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        rows = self.visible_rows
        width = self._usable()

        if not rows and not self._query:
            # The shared empty-store notice, whose "subagent runs are not
            # listed" clause is the EXPLANATION and not decoration — it must
            # survive the narrow case, so it wraps rather than truncating.
            out = Text()
            for index, line in enumerate(_wrap_cells(RESUME_EMPTY_NOTICE, width)):
                if index:
                    out.append("\n")
                out.append(line, style=dim)
            return out
        if not rows:
            # The filter row already echoes the query; repeating it here said
            # it twice in two grammars.
            return Text("no session matches that filter", style=dim)

        budget = self._layout().list_rows
        costs = self._row_costs()
        drawn = fit_rows(costs, self._offset, budget)
        window = rows[self._offset : self._offset + drawn]
        query = self._query.strip()

        out = Text()
        for index, line in enumerate(
            render_rows(
                window,
                self._selected - self._offset,
                width,
                self._now,
                None if self._hovered is None else self._hovered - self._offset,
                self.body_matched_ids,
                # The RESULT SET, not this page — a column that appears as a
                # fork scrolls into view and vanishes as it scrolls out makes
                # every name jump sideways on one arrow press.
                any(getattr(row, "forked", False) for row in rows),
                self._frame,
                # The LATCHED exec fact: the column may widen, never narrow.
                self._exec_column_latched(rows),
                name_max=self._layout().name_width,
                age_width=self._layout().age_width,
                show_id=self._layout().show_id,
                soft_gutter=bool(query),
                exact_matched=self._body_matches,
            )
        ):
            if index:
                out.append("\n")
            out.append_text(line)
            row = window[index]
            if query:
                context = self._context_for(row)
                if context:
                    out.append("\n")
                    out.append(" " * CONTEXT_INDENT)
                    # ``demark`` BEFORE highlighting (D19), so literal ``**``
                    # and backticks do not ride into the list pane and so the
                    # match offsets stay honest.
                    out.append_text(
                        _highlight(
                            truncate_cells(demark(context), self._context_width()),
                            query,
                        )
                    )
        return out

    def _preview_text(self) -> Text:
        """The conversation preview for the row under the cursor."""
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        fg_colour = theme_mod.semantic_color("fg")

        row = self._selected_row()
        if row is None:
            return Text("no session", style=dim)

        out = Text()
        out.append(f"{row.name or row.id}\n", style=Style(color=fg_colour, bold=True))
        created = 0.0
        try:
            created = self._previews().created_at(row.id) or row.created_at
        except Exception:  # pragma: no cover - a broken store must not stop the paint
            created = row.created_at
        started = format_age(max(0.0, self._now - created)) if created else "·"
        worked = format_age(max(0.0, self._now - row.mtime))
        out.append(f"started {started} · last worked {worked}\n", style=muted)
        # Omitted ENTIRELY when there is no checkpoint (D7), rather than drawn
        # as a bare `· · ·` that reads as a load that never resolved. Measured:
        # on all 113 rows that have one, both model and cwd are present, so the
        # line is fully populated or fully absent.
        checkpoint = self._selected_checkpoint()
        if checkpoint:
            out.append(
                f"{_short_model(checkpoint)} · {_short_cwd(checkpoint)}\n",
                style=dim,
            )
        # Spans the REAL preview width, or it runs off the edge when stacked.
        out.append("─" * max(10, self._pane_width()) + "\n", style=faint)

        lines = self._preview_lines()
        height = self._pane_height()
        # Clamped on every paint: the row budget changes with the terminal, and
        # an offset from a taller geometry would leave the pane blank.
        self._pane_top = max(0, min(self._pane_top, max(0, len(lines) - height)))
        if not lines:
            out.append("(no prose in this transcript)", style=dim)
            return out

        for kind, text in clip_to_height(lines, self._pane_top, height):
            if kind == "gutter":
                ink = "accent" if text.endswith("you") else "success"
                out.append(
                    f"{text}\n",
                    style=Style(color=theme_mod.semantic_color(ink), bold=True),
                )
            elif kind == "blank":
                out.append("\n")
            else:
                # A uniform two-space indent; the wrap already rstripped each
                # line so a continuation cannot turn it into three (D11).
                out.append(f"  {text}\n", style=Style(color=fg_colour))
        return out

    def _filter_text(self) -> Text:
        """One row: the query, the counters, the legends and the key hints.

        This row carries what the card's last three lines used to: the position
        is stated, the keys are stated, and they do not collide. The query is
        the user's only receipt that typing reached this modal, so narrow
        terminals shed the tally and then the hints before they shed the query.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        label = Style(color=theme_mod.semantic_color("label"))
        accent = Style(color=theme_mod.semantic_color("accent"))

        rows = self.visible_rows
        # Measured off the resolved widget when there is one. The accessors are
        # deliberately usable on an UNMOUNTED screen — a large part of the suite
        # asserts rendered text without a pilot — so this falls back to the
        # planned width rather than requiring `compose` to have run.
        # Measured off the widget's CONTENT region — the cells left inside its
        # own padding — rather than off `size` less a guessed inset. The guess
        # was one cell out at 140 columns and the row was clipped mid-word by
        # Textual, which is the silent-clip failure this file keeps warning
        # about: nothing reads back that it happened.
        measured = _content_width(getattr(self, "_filter", None))
        # The row spans the SCREEN, not the list pane, so the fallback is the
        # full planned width rather than `list_width` — which would shed the
        # tally and the keys on an unmounted screen, where much of the suite
        # reads this row. One source, so a stubbed layout and the real one
        # cannot disagree about how much room this row has.
        width = max(1, measured or self._layout().screen_width)
        costs = self._row_costs()
        drawn = fit_rows(costs, self._offset, self._layout().list_rows) if rows else 0

        out = Text(no_wrap=True, overflow="ellipsis")
        out.append("/ ", style=accent)
        if self._query:
            out.append(self._query, style=label)
        else:
            out.append("type to filter", style=dim)

        counter: tuple[int, int, int] | None = None
        if rows and len(rows) > drawn:
            counter = (self._offset + 1, self._offset + drawn, len(rows))

        # The tally, then the legends, then the keys — statements ABOUT the
        # list before the keys that OPERATE it, which is the card's own
        # grammar moved into one row.
        tail = Text(no_wrap=True, overflow="ellipsis")
        if counter is not None:
            first, last, total = counter
            tail.append("   showing ", style=faint)
            tail.append(f"{first:,}–{last:,}", style=dim)
            tail.append(" of ", style=faint)
            tail.append(f"{total:,}", style=dim)
        else:
            # THE FILTERED COUNT WHENEVER A FILTER IS ACTIVE, even though the
            # list fits one page and there is no scroll position to report.
            # A design round caught this reporting the whole store's size over
            # eleven visible rows (8 of 8 such frames): the counter branch is
            # skipped when everything fits, and the branch it falls through to
            # was answering a different question — "how many sessions are
            # there" rather than "how many matched". The unfiltered case still
            # states the store total, which is what it means there.
            count = len(rows) if self._query else len(self._all)
            word = "session" if count == 1 else "sessions"
            tail.append(f"   {count:,} {word}", style=dim)

        # LEGENDS TRAVEL WITH THE COUNTER, not with the keys, and that
        # placement is design round 2's D2 rather than a tidier arrangement.
        # A legend states what a mark in the list MEANS — a statement ABOUT the
        # list, which is what this half of the row already carries — while the
        # keys say how the picker is driven. Ranking the legend against the
        # hints is what made it paint only on a list short enough not to
        # scroll, i.e. never on a real store at any terminal width.
        for glyph, meaning in _meta_legends(
            width,
            has_marked=bool(self.body_matched_ids),
            has_exec=self._exec_column_latched(rows),
            counter_cells=_counter_cells(counter),
        ):
            tail.append(" · ", style=faint)
            tail.append(glyph, style=dim)
            tail.append(f" {meaning}", style=faint)

        # Asked for the room the KEYS actually have — the row already spent
        # cells on the query and the tally — so `_footer_hints`' own shed
        # ladder runs against the real budget instead of against the whole
        # row. Handing it the full width let it return 91 cells for an 80-cell
        # row, and the whole block was then dropped as one.
        # The preview-mode hint is built FIRST and MEASURED, not estimated: it
        # is the one hint that is also a status, so its width changes with the
        # mode. A constant estimate here over-fed `_footer_hints` by three
        # cells, which spent the tail's room on a hint and dropped the counter
        # and the legend from a row that had space for both.
        mode_hint = Text(no_wrap=True, overflow="ellipsis")
        mode_hint.append(" · ", style=faint)
        mode_hint.append("ctrl+e", style=dim)
        mode_hint.append(f" {self.preview_mode_for_test()}", style=faint)

        def key_row(room: int, *, budget: int | None = None) -> Text:
            """The key hints that fit ``room``, plus the mode hint if it also fits.

            ``room`` is the space for the HINTS; ``budget`` is the space for
            this whole row including the mode hint, and defaults to
            ``room`` + the hint's own width — i.e. the caller already reserved
            it. Passing them separately is what stops the reservation being
            counted twice, which dropped the hint at widths with room to spare.
            """
            out_keys = Text(no_wrap=True, overflow="ellipsis")
            for index, (key, what) in enumerate(
                _footer_hints(
                    max(0, room),
                    scrolls=counter is not None,
                    empty=not rows and bool(self._query),
                )
            ):
                out_keys.append(" · " if index else "   ", style=faint)
                out_keys.append(key, style=dim)
                if what:
                    out_keys.append(f" {what}", style=faint)
            # The mode hint is a STATUS, not a way out, so it is the first
            # thing in this row to go when the keys themselves are under
            # pressure. Keeping it ahead of `esc` is what let a 24-column row
            # read `esc · ctrl+e con…` — the exit truncated mid-word to make
            # room for a label saying which preview mode is on.
            room_total = budget if budget is not None else max(0, room) + cell_len(mode_hint.plain)
            if cell_len(out_keys.plain) + cell_len(mode_hint.plain) <= room_total:
                out_keys.append_text(mode_hint)
            return out_keys

        keys = key_row(
            width - cell_len(out.plain) - cell_len(tail.plain) - cell_len(mode_hint.plain)
        )

        # SHED IN ORDER OF NEED, and the query is never dropped: it is the
        # user's only receipt that typing reached the modal. The keys go next
        # to last because they are the only statement of how to leave, and the
        # tally — a nicety — goes first. Each candidate is appended only if the
        # WHOLE row still fits, so the row can never overflow the terminal.
        # THE KEYS ARE NEVER SHED. `_footer_hints` has already reduced itself
        # to `enter`/`esc` at its narrowest, and between them they are how the
        # picker is used and how it is left — the one row that cannot afford to
        # overflow is also the one that cannot afford to go silent. So only the
        # TAIL is optional here: the counter and the legends are a nicety, the
        # way out is not.
        room = width - cell_len(out.plain) - cell_len(keys.plain)
        if cell_len(tail.plain) > room:
            # The keys are already at their narrowest that still fits, so the
            # only cells left to find are in their LABELS. Shedding those before
            # the tail is design round 2's D2 ordering: the counter and the
            # legend explain what is on screen and have no other home, while a
            # bare `enter · esc` still states the way out. `_footer_hints` does
            # exactly this reduction itself when asked for less room.
            keys = key_row(0, budget=max(0, width - cell_len(out.plain)))
            room = width - cell_len(out.plain) - cell_len(keys.plain)
        if cell_len(tail.plain) <= room:
            out.append_text(tail)

        # THE WAY OUT IS SHED LAST, AND THE QUERY YIELDS TO IT. Everything
        # above trims the row from the right, but the ECHO on the left grows
        # with what the user types, so at the floor the two meet and a blind
        # truncation takes `esc` off the end — a modal with no stated exit,
        # which is the one state this row exists to prevent (see this module's
        # header). Below the width where both fit, the echo is truncated to
        # whatever is left after the keys are reserved: a shortened query is
        # still a receipt that typing landed, while a missing `esc` is a dead
        # end. Measured at 24 and 30 columns, where the full row is 33 cells.
        keys_cells = cell_len(keys.plain)
        if cell_len(out.plain) + keys_cells > width:
            room_for_echo = max(0, width - keys_cells)
            out = Text(truncate_cells(out.plain, room_for_echo), style=out.style)
        out.append_text(keys)
        # The row is `no_wrap` with ellipsis overflow, but Textual only applies
        # that against the widget's REAL width — and this text is also read
        # back by the accessors on an unmounted screen. Truncating here keeps
        # the two answers identical and makes the "never wider than the
        # terminal" invariant a property of the text, not of the paint.
        #
        # Truncating from the LEFT end of the row would take the keys with it,
        # so anything still over budget here means even the bare keys do not
        # fit; the assertion in `test_the_way_out_is_stated_at_every_width_the
        # _picker_supports` pins the floor at which that would start to happen.
        if cell_len(out.plain) > width:
            return Text(truncate_cells(out.plain, width), style=out.style)
        return out


#: Footer hints, MOST disposable first. ``enter``/``esc`` are never dropped:
#: between them they are how the card is used and how it is left.
_FOOTER_HINTS: tuple[tuple[str, str], ...] = (
    ("↑↓", "move"),
    ("pgup/pgdn", "page"),
    ("type", "to filter"),
    ("enter", "resume"),
    ("esc", "cancel"),
)
_FOOTER_DROP_ORDER = ("pgup/pgdn", "type", "↑↓")

#: Drop order for a plain (unmarked) list that SCROLLS. ``pgup/pgdn`` is the
#: first thing shed by the order above, which is right for a list that fits on
#: one page and wrong for one that does not: the picker advertised paging where
#: paging is a no-op and withdrew it where it is the fastest way through the
#: list. Uncapping the store made the bare scrolling picker the DEFAULT state
#: rather than an edge case, so this is the common path, not a rare one.
#: ``type`` sheds first instead — a user who is already filtering knows they can
#: type, and the filter they typed is echoed in the header regardless.
_FOOTER_DROP_ORDER_SCROLLING = ("type", "pgup/pgdn", "↑↓")

#: The ``"`` body-match marker is load-bearing but unlabelled in the list: a
#: first-time reader sees a lone right-quote at the start of some rows and can
#: read it as a rendering artifact rather than "this row matched inside the
#: conversation" (design round 1, D2). So when any visible row carries the
#: marker, the footer states what it means — keyed on the marker GLYPH itself
#: so the legend and the mark are unmistakably the same thing.
_MARKER_LEGEND: tuple[str, str] = (BODY_MATCH_MARKER.strip(), "matched inside")

#: Where the legends do NOT live: the key row below. The full key-hint row is 69
#: cells, and the card this replaced was capped at 74, so a 24-cell legend
#: could only appear there by evicting a key — and the keys must win,
#: because they OPERATE the card while a legend teaches. Design round 1 resolved
#: that by ranking the legend above the two disposable hints, which made it
#: paint on a list that FITS one page and never on one that scrolls; against
#: real stores in the hundreds that is the ordinary case, at any terminal width
#: (design round 2, D2) — and it was worse still under the old 10-row page cap
#: this redesign removed. The legends
#: therefore share the position counter's row and shed against their own budget
#: — see :func:`_meta_legends`.

#: The same treatment for :data:`EXEC_MARKER`, and for a sharper reason than the
#: body-match mark needed. `[exec]` is legible as a WORD — nobody mistakes it for
#: a rendering artifact the way a lone `”` is mistaken — but legibility is not
#: the problem it has. Its grammar is byte-identical to `[fork]`'s, deliberately,
#: and `[fork]` states a permanent fact about ancestry: a row that will still be
#: there tomorrow. An identical treatment therefore files the two under one
#: class, and the user learns "this row came from somewhere else" — true, and the
#: LESS important of the two things the tag means. What it does not tell them is
#: that the row may be gone by the time they press Enter (design round 1, D2).
#:
#: The words that do carry lifetime already exist and are well judged
#: (`CatalogEntry.status`'s "Running headless (exec)"), but they render only in
#: the sidebar's hover tooltip — a surface the picker never shows, and they
#: appeared in zero of nine captured picker frames. So the ephemerality is said
#: HERE, in the one place the picker already reserves for explaining a mark.
#:
#: "one-shot" rather than "headless": both are established (`cli.py` calls exec a
#: "one-shot headless task"), but headless describes HOW it runs and one-shot
#: describes how long it lasts, which is the fact the legend exists to add
#: (design round 1, D5). No new visual grammar, no third ink — an amber `[exec]`
#: was the other candidate and is rejected on the record, because `warning` is
#: already a two-member class this file's own comments call "at its capacity".
_EXEC_LEGEND: tuple[str, str] = (EXEC_MARKER.strip(), "one-shot, may end")


def _footer_drop_order(*, scrolls: bool) -> tuple[str, ...]:
    """The shed order for the key hints, which is the only thing this row holds.

    DERIVED rather than enumerated so the two documented constants above cannot
    drift from the rule they state; the assertion below proves this function
    still reproduces them byte for byte.

    The rule, stated once:

    * the genuinely disposable hints shed first (``pgup/pgdn`` and ``type``,
      conveniences a user discovers anyway), EXCEPT that a scrolling list drops
      ``type`` first instead — paging is the fastest way through a list that has
      more than one page, so it must not be the first thing sacrificed;
    * ``↑↓`` last of all the droppable hints;
    * ``enter``/``esc`` never appear here — they are how the card is used and
      how it is left.

    Legends are NOT shed here any more. They live on the counter's row and shed
    against their own budget in :func:`_meta_legends`, which is what made them
    reachable at all — see that function and the call site.
    """
    return ("type", "pgup/pgdn", "↑↓") if scrolls else ("pgup/pgdn", "type", "↑↓")


# The derivation must reproduce the documented constants exactly, or the
# comments above them are describing a policy the code no longer follows.
assert _footer_drop_order(scrolls=False) == _FOOTER_DROP_ORDER
assert _footer_drop_order(scrolls=True) == _FOOTER_DROP_ORDER_SCROLLING


#: The footer for a filter that matched nothing. Movement, paging and `enter
#: resume` all describe a list that is not there, so the only honest thing the
#: row can say is how to get back to one. `backspace` is the key that widens
#: the query, and it is the key a user in this state is already reaching for.
#: Stated as a hint pair like every other so it sheds and renders identically.
_EMPTY_HINT: tuple[str, str] = ("backspace", "to widen")


def _counter_cells(counter: tuple[int, int, int] | None) -> int:
    """Cells the position counter will occupy, or 0 when it is not drawn.

    Measured from the FORMATTED string rather than estimated, because the
    numerals are thousands-grouped and the width therefore depends on the store
    size: "showing 1–10 of 40" is 18 cells and "of 24,310" is 22. The legends
    share this row, so an estimate here is a legend that overflows the card on
    exactly the large stores that made the picker scroll in the first place.
    """
    if counter is None:
        return 0
    first, last, total = counter
    return cell_len(f"showing {first:,}–{last:,} of {total:,}")


def _meta_legends(
    width: int,
    *,
    has_marked: bool,
    has_exec: bool,
    counter_cells: int = 0,
) -> list[tuple[str, str]]:
    """The mark legends that fit beside the counter, dropping the least needed.

    Legends explain what a mark in the LIST means, which is a statement about
    the list — the same kind of statement the position counter makes — so they
    share its row. They used to lead the key row below and were unreachable
    there: see the call site, where the 69-vs-74 cell arithmetic is set out.

    Order and shed are unchanged from that row and keep their round-1 reasoning.
    DISPLAYED, the body mark leads because that is the order the marks appear in
    a row (body column, then the kind column to its right). SHED, ``[exec]`` goes
    first: it is a readable word that still means something without its gloss,
    while a lone right-quote with nothing explaining it is exactly the rendering
    artifact :data:`_MARKER_LEGEND` exists to prevent.

    A legend never survives as a bare glyph. Dropping the gloss would leave the
    unexplained mark the legend was added for, so the whole pair goes.
    """
    legends = [
        legend
        for legend, present in ((_MARKER_LEGEND, has_marked), (_EXEC_LEGEND, has_exec))
        if present
    ]
    # The counter's own cells plus the separator that would join it to the
    # first legend: the budget is what is LEFT of the row, not the row.
    room = width - counter_cells - (3 if counter_cells else 0)
    while legends and _row_cells(legends) > room:
        # Shed the readable word first — reverse of display order, per above.
        legends.pop()
    return legends


def _row_cells(pairs: Sequence[tuple[str, str]]) -> int:
    """Cells ``pairs`` occupy when joined by the card's ``" · "`` separator.

    One definition for both meta rows, so a legend and a key hint can never
    disagree about what a row costs.
    """
    return sum(cell_len(f"{key} {what}".strip()) for key, what in pairs) + 3 * max(
        0, len(pairs) - 1
    )


def _footer_hints(
    width: int,
    *,
    scrolls: bool = False,
    empty: bool = False,
) -> list[tuple[str, str]]:
    """The key hints that fit in ``width`` cells, dropping the least needed.

    Three stages, because the footer is the one row that must not overflow the
    card: shed whole hints in order of need; then, if even ``enter``/``esc``
    with their labels will not fit (a card under about 26 cells), drop the
    LABELS and keep the keys. Two bare keys still say which keys exist, which
    is more than a clipped row says.

    KEYS ONLY. The mark legends moved to the counter's row in design round 2 —
    see :func:`_meta_legends` — because the full key row is 69 cells and the
    card this replaced was capped at 74, so a legend could only appear here by
    evicting a key. This function no longer has to choose between teaching a
    mark and stating how to leave the picker.

    ``scrolls`` says the list is longer than one page, which REORDERS the shed:
    ``pgup/pgdn`` sheds first on a list that fits one page (paging there is a
    no-op) and ``type`` sheds first on one that does not, because paging is then
    the fastest way through the list (round 1, D3).
    """
    if empty:
        # Nothing to move through, page, or resume: offering those keys for an
        # empty list advertises actions that do nothing. `esc` stays because
        # leaving is still available and is the other thing a user wants here.
        return _shed_to_width([_EMPTY_HINT, ("esc", "cancel")], (_EMPTY_HINT[0],), width)

    return _shed_to_width(list(_FOOTER_HINTS), _footer_drop_order(scrolls=scrolls), width)


def _shed_to_width(
    hints: list[tuple[str, str]], drop_order: Sequence[str], width: int
) -> list[tuple[str, str]]:
    """``hints`` reduced to fit ``width`` cells, dropping in ``drop_order``.

    The last resort drops the LABELS and keeps the keys: two bare keys still say
    which keys exist, which is more than a clipped row says. Shared by every
    footer variant so a new one cannot quietly grow a second shed policy.
    """

    def cells(pairs: list[tuple[str, str]]) -> int:
        return sum(cell_len(f"{key} {what}".strip()) for key, what in pairs) + 3 * max(
            0, len(pairs) - 1
        )

    for droppable in drop_order:
        if cells(hints) <= width:
            return hints
        hints = [pair for pair in hints if pair[0] != droppable]
    if cells(hints) <= width:
        return hints
    return [(key, "") for key, _ in hints]

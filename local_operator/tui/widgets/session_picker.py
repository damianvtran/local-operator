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
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

# ``fork_haystack`` is imported rather than restated here: the phone's session
# search matches over the same rows, and two spellings of "what text does this
# row have" is how one surface ends up finding a fork the other cannot.
from local_operator.resume import SessionRow, fork_haystack, format_age
from local_operator.session.search_index import SoftSearchIndex, search_digests
from local_operator.tui import theme as theme_mod
from local_operator.tui.terminal_title import SPINNER_FRAMES, SPINNER_INTERVAL_S
from local_operator.tui.widgets.tool_card import truncate_cells

logger = logging.getLogger(__name__)

#: Width the card will take when the terminal allows it, and the floor it will
#: not go below. Both are CELL counts of the card's content, inside its
#: padding; :meth:`SessionPickerScreen._card_width` resolves the actual value
#: against the screen every paint.
PICKER_MAX_WIDTH = 74
PICKER_MIN_WIDTH = 30

#: Cells the card leaves between itself and the terminal's edges, on top of its
#: own padding, so it reads as floating rather than as a panel bolted to the
#: side.
PICKER_WIDTH_MARGIN = 6

#: Cells of the card's own padding on EACH side, mirroring the horizontal half
#: of ``padding: 1 2``. Distinct from :data:`CARD_PADDING_ROWS`: a cell budget
#: and a row budget happen to be the same number here, and spending one for
#: the other is a bug waiting for the stylesheet to change.
PICKER_PADDING_CELLS = 2

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
#: It must also FIT: this string is the card's own empty body, and the card
#: is capped at :data:`PICKER_MAX_WIDTH` cells. The first wording ran to 76
#: cells and hung two past the rule at full width — and much further on a
#: narrow terminal, where every neighbouring row sheds cells to fit. Anything
#: edited here is measured against that cap, not eyeballed.
RESUME_EMPTY_NOTICE = "no conversations of yours to resume — subagent runs are not listed"

#: Rows of sessions shown before the list scrolls, when the terminal has room.
#: A page that fills the screen makes the modal feel like a mode switch rather
#: than a popup; ten is enough to scan. A CEILING, not the page size — see
#: :meth:`SessionPickerScreen._page_rows`.
PAGE_ROWS_MAX = 10

#: Name/id matches at which the picker stops consulting the bounded soft tier
#: (see ``SessionPickerScreen._soft_tier_wanted``). Three, not one: a single
#: exact hit on a name is as often incidental as deliberate — ``spit`` matches
#: "De\ *spit*\ e" — and treating it as a real answer hid every genuinely
#: intended match behind it. Measured over 517 vocabulary-drawn typos, this
#: floor loses no rows against running the tier on every keystroke while
#: leaving the cursor exactly as stable.
_PRECISE_HITS_ENOUGH = 3

#: Non-row lines the card always draws: header, rule, blank spacer, the
#: position counter, and the key hints. Reserved UNCONDITIONALLY (the counter
#: included, even when the list fits) so the height never depends on content
#: the user is about to change by typing a filter — a footer that appeared and
#: vanished as the list narrowed would move the card under the cursor.
CARD_CHROME_ROWS = 5

#: The card's own padding rows, mirroring ``padding: 1 2`` in the stylesheet.
CARD_PADDING_ROWS = 2

#: Share of the terminal the card may occupy, mirroring ``max-height: 80%``.
#: Kept in step by hand because Textual clips SILENTLY: rows past the cap are
#: simply not drawn and nothing reads back that it happened.
CARD_MAX_HEIGHT_FRACTION = 0.8

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

#: Cells reserved for the live-state column when ANY row in the result set
#: carries state. One cell for the state glyph plus its separating space; the
#: spinner frames, the wake glyph and the markers above are all one cell wide.
STATE_COL_CELLS = 2


def filter_rows(
    rows: Sequence[SessionRow],
    query: str,
    body_matches: AbstractSet[str] | None = None,
) -> list[SessionRow]:
    """Rows whose name or id contains ``query``, or whose id is in ``body_matches``.

    A pure MEMBERSHIP filter: it decides which rows are shown and preserves the
    order it was handed, so on its own it never moves a row under the cursor.
    Relevance ORDERING lives in :func:`rank_rows`, applied by the screen only
    when a query is active — keeping the "filtering never reorders" property
    literally true of this function while the query-scoped ranking sits in the
    one place that also re-homes the cursor.

    The searchable name is composed by :func:`fork_haystack`, so a row visibly
    tagged ``[fork]`` is found by typing ``fork`` — the tag is on screen, so it
    has to be in the index.

    The name/id test stays exact substring: those fields are a sentence the user
    wrote and a hex id, where an exact match is what a precise query expects.
    Soft matching (prefix, word-order, bounded typo) is not done here; it is
    folded into ``body_matches`` by the caller via
    :func:`local_operator.session.search_index.soft_search_digests`, so a soft
    hit on the conversation surfaces the row exactly as an exact body hit does.

    ``body_matches`` is the set of ids admitted on their conversation text —
    exact body, a past name, or a bounded soft match — decided by the caller
    and passed in rather than computed here, so this function stays pure and
    cheap enough to run per keystroke, and a caller with no index (a test, an
    embedder) keeps exactly the old name-and-id behaviour.
    """
    needle = query.strip().lower()
    if not needle:
        return list(rows)
    matched = body_matches or frozenset()
    return [
        row
        for row in rows
        if needle in fork_haystack(row).lower() or needle in row.id.lower() or row.id in matched
    ]


def matched_in_body(row: SessionRow, query: str, body_matches: AbstractSet[str]) -> bool:
    """True when ``row`` is on screen only because its CONVERSATION matched.

    Drives the body-match marker. A row whose visible name already contains the
    query needs no explanation; one that does not would otherwise read as the
    filter returning something arbitrary, which is worse than no marker at all
    because it makes the whole result set look untrustworthy.
    """
    needle = query.strip().lower()
    if not needle:
        return False
    # Same haystack the filter admitted on, so a row surfaced by its VISIBLE
    # fork tag is not additionally explained as a body match — the tag is
    # already on the row and the two marks would contradict each other.
    if needle in fork_haystack(row).lower() or needle in row.id.lower():
        return False
    return row.id in body_matches


#: Relevance tiers, best (lowest) first, used to sort a FILTERED subset when a
#: query is active. A tier is a property of ``(query, row)`` alone — it does not
#: depend on the previous query or on the order rows arrived — which is what
#: lets ranking coexist with the "no reorder under the cursor" invariant: the
#: order changes only when the query changes, and a query change already
#: re-homes the cursor to the top match (see ``set_query``).
_RANK_NAME = 0  # exact substring in the visible name — the strongest signal
_RANK_ID = 1  # exact substring in the id
_RANK_BODY = 2  # exact substring in the body/past-name digest
_RANK_SOFT = 3  # soft (prefix / token-AND / edit-distance) match only


def rank_rows(
    rows: Sequence[SessionRow],
    query: str,
    body_matches: AbstractSet[str] | None = None,
) -> list[SessionRow]:
    """``rows`` ordered by relevance to ``query``; recency order when empty.

    Kept SEPARATE from :func:`filter_rows`, which stays a pure membership
    filter, because the module's invariant is stated about filtering: "filtering
    narrows; it never reorders". Ordering is a property of the QUERY, not of a
    keystroke within a query's growth — so it is applied here, in the one place
    that recomputes per query and re-homes the cursor, and only when a query is
    active.

    * **Empty query** -> ``rows`` unchanged (recency order, newest first,
      exactly as today). A fixed query likewise never reorders: the key is a
      pure function of ``(query, row)``, so repeated repaints and resizes
      produce byte-for-byte the same order.
    * **Non-empty query** -> a single deterministic ordering: the tier the row
      matched in (name > id > body > soft), with recency (mtime desc) as the
      stable tie-break WITHIN every tier. ``sorted`` is stable, so passing rows
      already in recency order makes the tie-break free.

    ``body_matches`` is the EXACT-body match set (from ``search_digests``),
    passed in for the same reason :func:`filter_rows` takes it — this stays pure
    and cheap, and a caller with no index gets name/id ranking unchanged. It is
    only the exact-body set, not the soft set: a row in ``rows`` that matched
    none of name, id, or exact body was admitted by the soft set and so takes
    the soft tier, which needs no separate membership check.
    """
    needle = query.strip().lower()
    if not needle:
        return list(rows)
    body = body_matches or frozenset()

    def tier(row: SessionRow) -> int:
        # Through the same composition :func:`filter_rows` admits on: a fork
        # admitted on its tag must rank in the NAME tier, not fall through to
        # the soft tier and sort below every incidental body hit.
        if needle in fork_haystack(row).lower():
            return _RANK_NAME
        if needle in row.id.lower():
            return _RANK_ID
        # Exact body hit outranks a soft-only hit: an exact substring in the
        # conversation is a stronger signal than a typo/prefix/word-order match.
        if row.id in body:
            return _RANK_BODY
        # Admitted by the soft set (it is in the already-filtered ``rows`` yet
        # matched neither name, id, nor exact body), so it ranks below every
        # exact tier.
        return _RANK_SOFT

    # Stable sort on the tier alone: ``rows`` arrives newest-first, so equal
    # tiers keep recency order without a second sort key. Sorting the tier as
    # the only key is what preserves the recency tie-break for free.
    return sorted(rows, key=tier)


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
    runtime's residency (see ``OwnedSessionHandle.is_conversationally_active``),
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

    Reserved as FIXED CHROME rather than subtracted from the name afterwards,
    which is what keeps the drop ladder honest — the id surrenders its cells
    before the age, and the age before the name, and a marker that helped
    itself to the name's budget after the fact would jump that queue and could
    push a name under :data:`NAME_MIN_CELLS`.
    """
    marker_col = cell_len(BODY_MATCH_MARKER) if marked else 0
    marker_col += cell_len(FORK_MARKER) if forked else 0
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
    name_col, age_col, id_col = plan_columns(rows, width, ages, marked, any_forked, any_stated)
    marker_col = cell_len(BODY_MATCH_MARKER) if marked else 0
    fork_col = cell_len(FORK_MARKER) if any_forked else 0
    state_col = STATE_COL_CELLS if any_stated else 0

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
        lines.append(line)
    return lines


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
        """Whether the bounded soft tier should run for ``query``.

        A pure function of ``(query, rows)``: the tier runs unless the query
        matched a session's NAME or ID. No run history, no latch, no memory of
        previous keystrokes — that purity is what keeps the same visible query
        rendering identically however the user reached it, and it took four
        attempts to get there (see the module docstring on route independence).

        **Why name/id and not "any exact hit".** Gating on an empty exact result
        looks equivalent and silently destroys typo search. The exact tier also
        admits BODY substring hits, and on a real store almost every typed token
        appears incidentally in some conversation: ``plin`` has 8 body hits,
        ``gren`` has 1. One incidental hit anywhere in the store then silenced
        the tier for the whole query, so the typo it exists to rescue could not
        be found. Measured on typos drawn from the store's own vocabulary, that
        gate lost the target row outright on 11 of 763 queries and shed 100+
        rows on 14 — a recall regression against shipped behaviour, wearing the
        appearance of correct gating.

        A name or id match is different in kind. Those fields are a sentence the
        user wrote and a hex id they can copy; an exact substring in either is a
        deliberate, precise hit, and when the user has one they are not asking
        for fuzzy help. A body substring is not that signal — it is as likely to
        be the word appearing in passing inside an unrelated conversation.

        Measured over 521 vocabulary-drawn typos against the base behaviour of
        running the tier on every keystroke:

        ==========================  ============  ==============
        gate                        recall loss   top-row swaps
        ==========================  ============  ==============
        base (tier always runs)     0/521         0/279
        any exact hit silences it   5/521         3/279
        this gate (name/id only)    0/521         1/279
        ==========================  ============  ==============

        So it costs no recall against base while running the expensive tier no
        more often than base does, and it disrupts the cursor LESS than the
        gate it replaces.

        What it does not do: prevent the tier engaging on a keystroke where rows
        are on screen, which can re-home the cursor onto a row the user had not
        seen. That is bounded (1/279 keystrokes here, against base's 0) and is
        properly a CURSOR policy question — keep the selection on its row across
        a re-rank when that row survives — not an ordering one. Ordering cannot
        fix it without reading run history, which is what reopened route
        divergence in an earlier round.
        """
        needle = query.strip().lower()
        if not needle:
            return False
        # Name and id only, deliberately NOT the body digests: see above. This
        # mirrors the first two admission tests in ``filter_rows`` so the gate
        # and the filter cannot drift apart on what "an exact hit" means.
        #
        # Counted against a small floor rather than tested for emptiness,
        # because ONE precise hit is not yet a useful answer and can easily be
        # incidental: ``spit`` matches the name "Failover Triggering Despite
        # Available Account", and on that single hit the previous form silenced
        # the tier and made every ``split`` session unreachable. Below the floor
        # the user has almost nothing to look at, so the extra recall is worth
        # more than the precision; at or above it they have a real answer and
        # fuzzy additions would only dilute it.
        precise = 0
        for row in self._all:
            if needle in fork_haystack(row).lower() or needle in row.id.lower():
                precise += 1
                if precise >= _PRECISE_HITS_ENOUGH:
                    return False
        return True

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
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return None
        region = body.region
        if not region.contains(event.screen_x, event.screen_y):
            return None
        row = event.screen_y - region.y - self._header_rows()
        rows = self.visible_rows
        drawn = min(self._page_rows(), max(0, len(rows) - self._offset))
        if not 0 <= row < drawn:
            return None
        index = self._offset + row
        return index if 0 <= index < len(rows) else None

    # -- geometry ------------------------------------------------------------
    def _screen_size(self) -> tuple[int, int]:
        """The box the card's ``max-height``/``max-width`` actually resolve in.

        ``self.size`` (this Screen's own CONTENT box), not ``self.app.size``
        (the terminal). ``Screen { padding: 1 }`` insets the content box by two
        rows and two cells, so measuring the terminal over-counted the room by
        exactly that — and since the stylesheet's ``max-height: 80%`` resolves
        against the content box, the card asked for more rows than the
        container would draw and Textual clipped the difference SILENTLY, off
        the bottom, taking the footer with it at every height from 14 to 23.
        ``UsagePanel`` already measures the screen for the same reason.
        """
        try:
            size = self.size
            if not size.width or not size.height:  # not laid out yet
                size = self.app.size
        except Exception:  # pragma: no cover - only before the app has a screen
            return 80, 24
        # Reported HONESTLY. Clamping the width up to ``PICKER_MIN_WIDTH`` here
        # made a 24-column terminal measure as 30, and every budget derived
        # from it then overflowed the screen by the difference — the floor
        # belongs where the preference is applied (``_card_width``), not in the
        # measurement it is applied to.
        return max(1, size.width), max(8, size.height)

    def _card_width(self) -> int:
        """Content cells the card may use, measured against the terminal.

        The floor is applied only while it FITS. ``max(PICKER_MIN_WIDTH, …)``
        alone outranked the margin and then the terminal itself: at 30 columns
        it returned a 30-cell content box inside 4 cells of padding, so the
        card was 38 wide on a 30-column screen and the rule and header were
        cut. A minimum width is a preference; the terminal is not.
        """
        width, _ = self._screen_size()
        padding = PICKER_PADDING_CELLS * 2
        room = width - PICKER_WIDTH_MARGIN - padding
        if room < PICKER_MIN_WIDTH:
            # No room for the preferred floor: take what the screen has, and
            # give up the breathing margin before giving up content.
            return max(1, width - padding)
        return min(PICKER_MAX_WIDTH, room)

    def _page_rows(self) -> int:
        """Session rows the card can actually DRAW right now.

        Chrome is reserved FIRST and the list takes what is left. A fixed page
        let the cursor sit on a row the card never rendered — Enter then
        resumed a session the user could not see — and let the clip eat the
        footer, which is the only statement of how to leave.
        """
        _, height = self._screen_size()
        budget = int(height * CARD_MAX_HEIGHT_FRACTION) - CARD_PADDING_ROWS - CARD_CHROME_ROWS
        return max(1, min(PAGE_ROWS_MAX, budget))

    def _header_rows(self) -> int:
        """Rows above the first session row: the header and its rule."""
        return 2

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

    # -- rendering -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(classes="session-picker"):
            self._body = Static(self._card_text(), id="session-picker-body")
            yield self._body

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
        """Advance the spinner and re-read what it is claiming.

        Cheap by construction: the refresh is the same one `registry.scan()` +
        `read_index()` pair the picker already budgets for as a per-open cost,
        and the repaint is one `Static.update`. It runs only while the picker
        is on screen — the timer dies with the screen.

        Skipped entirely when no row is animating, so a store of cold sessions
        costs nothing: without a running session there is no motion to drive,
        and re-scanning the registry ten times a second to discover that would
        be the picker's own idle cost.
        """
        before = self._marker_signature()
        refresh = self._refresh_live_state
        if refresh is not None:
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
    #: `mtime` is deliberately absent: it changes constantly and is rendered
    #: as a coarse "when", so including it would repaint ten times a second
    #: for nothing.
    _SIGNATURE_FIELDS = ("id", "name", "forked", "live_state", "pending", "wakes", "wakes_dormant")
    # A NAME THAT IS NOT A FIELD READS AS A CONSTANT. That is how the D10 fix
    # shipped broken, so the names are checked against the row type itself
    # rather than trusted: a rename in `resume.SessionRow` fails here loudly
    # instead of silently dropping a column out of the comparison.
    assert not set(_SIGNATURE_FIELDS) - set(SessionRow._fields), (
        f"picker signature names unknown SessionRow fields: "
        f"{sorted(set(_SIGNATURE_FIELDS) - set(SessionRow._fields))}"
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
        """Re-measure: every column and the page size come from the screen."""
        self._move_to(self._selected)

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        body.update(self._card_text())

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads."""
        return [line.plain for line in self._card_text().split("\n")]

    def _card_text(self) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        fg_colour = theme_mod.semantic_color("fg")
        faint = Style(color=theme_mod.semantic_color("faint"))
        label = Style(color=theme_mod.semantic_color("label"))
        width = self._card_width()
        rows = self.visible_rows

        # The active filter is the user's only receipt that typing reached this
        # modal, so narrow terminals shed the tally and then the title before
        # they shed the query. Truncating the assembled line did the opposite:
        # at 50 columns it preserved the static title and clipped
        # ``filter asteroid`` completely.
        title = "Resume a conversation"
        header = Text(no_wrap=True, overflow="ellipsis")
        if self._query:
            lead = "  filter "
            compact_lead = "filter "
            # Grouped: `24,310` is read at a glance where `24310` is parsed as
            # a digit string. Only worth doing since the picker was uncapped —
            # at a 200-row ceiling the number never reached four digits. One
            # cell per group, and at the widths where that matters the tally
            # has already been shed entirely (see the shed order below).
            tally = f"  {len(rows):,} of {len(self._all):,}"
            full_width = cell_len(title) + cell_len(lead) + cell_len(self._query) + cell_len(tally)
            titled_width = cell_len(title) + cell_len(lead) + cell_len(self._query)
            if full_width <= width:
                header.append(title, style=Style(color=fg_colour))
                header.append(lead, style=faint)
                header.append(self._query, style=label)
                header.append(tally, style=dim)
            elif titled_width <= width:
                header.append(title, style=Style(color=fg_colour))
                header.append(lead, style=faint)
                header.append(self._query, style=label)
            elif cell_len(compact_lead) < width:
                header.append(compact_lead, style=faint)
                header.append(
                    truncate_cells(self._query, width - cell_len(compact_lead)),
                    style=label,
                )
            else:
                header.append(truncate_cells(self._query, width), style=label)
        else:
            # Singular when there is one. Pre-existing, but filtering makes a
            # one-row list the common case rather than the rare one: a machine
            # whose delegated fan-out dominates now lands there routinely.
            count = len(self._all)
            tally = f"  {count:,} session" if count == 1 else f"  {count:,} sessions"
            if cell_len(title) + cell_len(tally) > width:
                header.append(truncate_cells(title, width), style=Style(color=fg_colour))
            else:
                header.append(title, style=Style(color=fg_colour))
                header.append(tally, style=dim)

        out = Text()
        out.append_text(header)
        out.append("\n")
        # Raised card ground needs the raised hairline too: ``edge`` is tuned
        # against the app background and nearly vanishes on ``overlay``.
        out.append("─" * width, style=faint)
        out.append("\n")

        page = self._page_rows()
        counter: tuple[int, int, int] | None = None
        if not self._all:
            # WRAPPED to the measured width, not printed flat. Every other line
            # in this card is bounded by the runtime width; this one was a
            # constant, so it fitted the 74-cell ceiling and still overflowed
            # the actual card on any narrow terminal — at 60 columns it was cut
            # to "…subagent runs are", losing the clause that explains why the
            # list is empty, which is the whole reason the wording changed.
            # Wrapped rather than truncated for that reason: the explanation is
            # the message, so it must survive the narrow case, not be the first
            # thing dropped.
            for index, line in enumerate(_wrap_cells(RESUME_EMPTY_NOTICE, width)):
                if index:
                    out.append("\n")
                out.append(line, style=dim)
        elif not rows:
            # The header already echoes the query; repeating it here — and via
            # ``repr``, whose quoting flips on an apostrophe — said it twice in
            # two grammars.
            out.append("no session matches that filter", style=dim)
        else:
            window = rows[self._offset : self._offset + page]
            for index, line in enumerate(
                render_rows(
                    window,
                    self._selected - self._offset,
                    width,
                    self._now,
                    None if self._hovered is None else self._hovered - self._offset,
                    self.body_matched_ids,
                    # The RESULT SET, not this page: a column that appears as a
                    # fork scrolls into view and disappears as it scrolls out
                    # makes every name jump sideways on one arrow press.
                    any(getattr(row, "forked", False) for row in rows),
                    # The animated phase. Without it every call took the
                    # default 0 and the running marker never moved (D1).
                    self._frame,
                )
            ):
                if index:
                    out.append("\n")
                out.append_text(line)
            if len(rows) > page:
                counter = (
                    self._offset + 1,
                    self._offset + len(window),
                    len(rows),
                )

        # Body, then one quiet row, then the card's META — the position and the
        # key hints, which are the same KIND of row (statements ABOUT the list,
        # not entries in it) and so travel together at the bottom. This is the
        # usage card's grammar; the two overlays differ only by whether the
        # position row is there at all. The counter is EMITTED only when the
        # list scrolls: printing an empty line in its place left two blank rows
        # and pushed the keys away from the block they belong to.
        out.append("\n\n")
        if counter is not None:
            # Numerals carry the fact at the readable ``dim`` step; grammar can
            # stay quiet at ``faint`` because it is adjacent to those anchors.
            first, last, total = counter
            out.append("showing ", style=faint)
            out.append(f"{first:,}–{last:,}", style=dim)
            out.append(" of ", style=faint)
            out.append(f"{total:,}", style=dim)
            out.append("\n")
        # Key NAMES at `dim` and their labels at `faint`, matching the usage
        # card: at `faint` on this ground the keys themselves were 1.49:1.
        # Hints DROP to fit, in reverse order of need — the same discipline the
        # columns use. A footer that overflowed the card was the one row that
        # could not afford to: it is the only statement of how to get out.
        # The marker legend appears only when a marked row is actually on
        # screen, so an empty query or a pure name match never advertises a mark
        # the user cannot see (D2: teach the glyph where it is used, not always).
        has_marked = bool(self.body_matched_ids)
        # ``counter`` is set exactly when the list is longer than a page, so it
        # is already the "does this scroll" fact the shed order needs.
        for index, (key, what) in enumerate(
            _footer_hints(
                width,
                has_marked=has_marked,
                scrolls=counter is not None,
                empty=not rows and bool(self._query),
            )
        ):
            if index:
                out.append(" · ", style=faint)
            out.append(key, style=dim)
            if what:
                out.append(f" {what}", style=faint)
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

#: Drop priority once the legend is in play. The full key-hint row is ~69 cells
#: against a ~74-cell card, so a legend can only appear by DISPLACING a hint —
#: dropping it "first" would make it never show, i.e. no fix at all. So it
#: outranks the two genuinely disposable hints (``pgup/pgdn``, ``type``, which
#: describe conveniences a user discovers anyway) and is shed BEFORE the
#: movement and action keys. Because it is dropped before the bare-key stage,
#: it never survives as an unlabelled glyph — a lone ``"`` in the footer would
#: be exactly the artifact-looking mark D2 flagged.
_FOOTER_DROP_ORDER_MARKED = ("pgup/pgdn", "type", _MARKER_LEGEND[0], "↑↓")

#: Drop order once the list actually SCROLLS: the marker legend goes before the
#: paging keys. With the fixed order above, a list that grew past one page shed
#: ``pgup/pgdn`` to make room for the legend — so the picker advertised paging
#: in the state where paging does nothing and withdrew it in the state where it
#: is the fastest way through the list (design round 1, D3). Uncapping the store
#: is what made scrolling the normal case rather than the rare one, so the shed
#: order has to know whether there is anything to page through.
_FOOTER_DROP_ORDER_MARKED_SCROLLING = ("type", _MARKER_LEGEND[0], "pgup/pgdn", "↑↓")


#: The footer for a filter that matched nothing. Movement, paging and `enter
#: resume` all describe a list that is not there, so the only honest thing the
#: row can say is how to get back to one. `backspace` is the key that widens
#: the query, and it is the key a user in this state is already reaching for.
#: Stated as a hint pair like every other so it sheds and renders identically.
_EMPTY_HINT: tuple[str, str] = ("backspace", "to widen")


def _footer_hints(
    width: int, *, has_marked: bool = False, scrolls: bool = False, empty: bool = False
) -> list[tuple[str, str]]:
    """The key hints that fit in ``width`` cells, dropping the least needed.

    Three stages, because the footer is the one row that must not overflow the
    card: shed whole hints in order of need; then, if even ``enter``/``esc``
    with their labels will not fit (a card under about 26 cells), drop the
    LABELS and keep the keys. Two bare keys still say which keys exist, which
    is more than a clipped row says.

    ``has_marked`` adds the ``"``-marker legend (see :data:`_MARKER_LEGEND`)
    so the glyph the list is drawing is explained where the reader already
    looks for meaning. It sits at the FRONT (adjacent to nothing that could be
    read as a key) and is shed under width pressure per
    :data:`_FOOTER_DROP_ORDER_MARKED` — above the disposable hints so it can
    actually appear on a normal card, below the movement and action keys.

    ``scrolls`` says the list is longer than one page, which REORDERS the shed
    rather than adding a hint: the paging keys outrank the legend exactly when
    there is something to page through (see
    :data:`_FOOTER_DROP_ORDER_MARKED_SCROLLING`). Without it the two features
    fought — a list long enough to scroll is also long enough to contain a
    marked row, so the legend evicted the very hint the reader needed.
    """
    if empty:
        # Nothing to move through, page, or resume: offering those keys for an
        # empty list advertises actions that do nothing, and the marker legend
        # explains a glyph no row is drawing. `esc` stays because leaving is
        # still available and is the other thing a user wants here.
        return _shed_to_width([_EMPTY_HINT, ("esc", "cancel")], (_EMPTY_HINT[0],), width)

    hints = list(_FOOTER_HINTS)
    if has_marked:
        hints = [_MARKER_LEGEND, *hints]
        drop_order = _FOOTER_DROP_ORDER_MARKED_SCROLLING if scrolls else _FOOTER_DROP_ORDER_MARKED
    else:
        drop_order = _FOOTER_DROP_ORDER_SCROLLING if scrolls else _FOOTER_DROP_ORDER

    return _shed_to_width(hints, drop_order, width)


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

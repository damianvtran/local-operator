"""Model picker — the scrollable, filterable list behind ``/model``.

Why a second picker instead of reusing :class:`CommandPicker`: the two answer
different questions. The command picker offers a fixed registry of fifteen names
where the interesting information is the description; this one offers a live
catalogue of *hundreds* of models across a dozen providers where the interesting
information is a comparison — how big is its context, what does it cost, am I
even logged in to the provider that serves it. That means columns, a separator
between what you can use and what you would have to log in for, and a window that
scrolls rather than an "N more" stub.

It shares the command picker's INTERACTION model deliberately, because a user
should not have to learn two: the editor keeps focus and routes Up/Down/PgUp/
PgDn/Enter/Esc in, the query is just the text after ``/model``, and the row is
chosen by a callback rather than by the widget touching the buffer.

The list is never truncated to hide models. A catalogue is only useful if the
model you are hunting for is reachable, so overflow scrolls and filtering is
fuzzy over the string the user can actually see (``provider/id``) — type
``opus``, ``anthropic/``, or ``anthopus`` and all three converge.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import NoScreen
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells

#: Cursor glyph and its gutter. Identical to the command picker's and the
#: session picker's, because all three lists appear in the same place and a
#: different caret would read as a different kind of control. This said
#: "identical" while shipping ``›`` against the others' ``❯`` — three pickers,
#: two glyphs — until the round that put four cards on one ramp made them a
#: family and the divergence had to be settled.
_CURSOR = "❯"
_GUTTER_CELLS = 2

#: One cell of breathing room at the right edge, matching the transcript's.
_EDGE_MARGIN = 1

#: Minimum gap between the id column and the numbers. Wider than the single
#: space inside a column so the two read as separate groups rather than one run
#: — the same rule the status band's group seam follows.
_COLUMN_GAP = 2

#: Hard ceiling on visible rows, and the fraction of the screen the list may
#: take. A picker that eats the transcript is worse than one that scrolls: the
#: user is choosing a model to use ON the conversation they can no longer see.
#: The lead of ``app.PERSIST_HINT``, used to RECOGNISE that clause inside the
#: `` · ``-joined status string so the footer can give it its own protected row
#: (it is the one clause that must survive verbatim at narrow widths). Matched
#: by PREFIX because the app composes it with other clauses; kept as a constant
#: here rather than imported from ``app`` because ``app`` imports this module.
#: ``test_persist_hint_prefix_matches_app`` asserts the two never drift.
PERSIST_HINT_PREFIX = "d in /model"

MAX_VISIBLE_ROWS = 14
_SCREEN_HEIGHT_FRACTION = 3

#: The seam the app joins footer clauses with (``_status_line``) and the one the
#: footer puts between its own count and the status, so the whole line reads as
#: one run of clauses rather than two kinds of separator.
_SEAM = " · "

#: Where a footer clause's first value ends — see `_fit_clauses`.
_FIRST_VALUE_END = re.compile(r", | — ")

#: Below this width the numbers are dropped and the row is just the selector.
#: Two columns of metadata in 40 cells leaves nothing for the id, which is the
#: part being chosen.
_NUMBERS_MIN_WIDTH = 56

#: Marker on the row that is the session's current model.
_CURRENT_MARK = "●"

#: Version-shaped numbers inside a model id: `4`, `4.1`, `2.5`, the `2` in `k2`,
#: the `3` in `qwen3:8b`.
#:
#: The lookbehind excludes digits and dots ONLY. Excluding word characters as well
#: looked tidier and silently broke every id that glues the version to a letter —
#: `kimi-k2` matched nothing at all, so its version came from the `0905` serial and
#: `kimi-k2-0905` outranked `kimi-k3`. The dot is what stops a decimal's own
#: fraction being counted a second time as a standalone number.
_VERSION_PATTERN = re.compile(r"(?<![\d.])(\d+(?:[.]\d+)?)")

#: A dash followed by a SHORT run of digits is a minor version (`opus-4-1` = 4.1),
#: rewritten to a decimal before the version scan. Capped at two digits on purpose:
#: `sonnet-4-20250514` is a dated snapshot, and reading it as 4.20250514 would put
#: it above every real version in the catalogue.
_MINOR_VERSION_PATTERN = re.compile(r"(?<![\d.])(\d+)-(\d{1,2})(?![\d])")


@dataclasses.dataclass(frozen=True)
class ModelRow:
    """One offerable model.

    ``connected`` is the provider's credential state, not the model's. The app
    filters unreachable rows out before they get here — a picker is a list of
    choices — so a False row is one of the two the filter deliberately keeps: the
    session's CURRENT model when its provider stopped being usable, or every row
    at once when the credential store could not be read. Both need to look
    different from a model that will run, which is what this flag drives (dim id,
    `login required` where the numbers go, and last place in the ranking).
    Choosing one starts a login instead of a switch — see
    :meth:`ModelPicker.highlighted`.
    """

    provider: str
    model_id: str
    #: The model's display name, already through ``model/naming.py``'s honesty
    #: rule upstream — so it is either a name that identifies this model alone or
    #: the selector itself, never a name two models answer to. Empty means the
    #: caller had none; the row then shows its selector and nothing more.
    label: str = ""
    context_window: int = 0
    default_context_window: int | None = dataclasses.field(default=None, kw_only=True)
    max_context_window: int | None = dataclasses.field(default=None, kw_only=True)
    input_price: float = 0.0
    output_price: float = 0.0
    connected: bool = True
    #: True when this row comes from a RESELLER rather than the model's own
    #: provider. Set by the caller, which is the only layer that knows the
    #: registry; the picker only needs it as a sort rung.
    aggregated: bool = False
    #: This row is a META-ROUTE — a router whose price is the price of whichever
    #: model it dispatches to. Set by the caller from the listing that said so
    #: (``CatalogueEntry.routed``), never inferred here: the renderer cannot
    #: tell a router's unknown price from any other unknown one, and deciding it
    #: from the id in this layer would be the second, divergent statement of the
    #: rule that :func:`format_price_pair`'s docstring exists to warn against.
    routed: bool = False

    @property
    def selector(self) -> str:
        """``provider/id`` — what ``/model`` takes and what the user types."""
        return f"{self.provider}/{self.model_id}"


def format_window(tokens: int) -> str:
    """``400k`` / ``1.0m`` / ``""`` when unknown.

    Empty rather than a placeholder for unknown: the column is right-aligned, so
    a blank simply leaves the cell empty while a ``—`` would draw the eye to the
    one row that has nothing to say.
    """
    if tokens <= 0:
        return ""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}m".replace(".0m", "m")
    if tokens >= 1_000:
        return f"{tokens // 1_000}k"
    return str(tokens)


#: What a meta-route's price column reads. Lower-case to match ``free``, the
#: only other word this column prints; hyphenated so it reads as one token
#: beside the window it sits next to.
#:
#: Measured against the layout before it was chosen, because it is materially
#: longer than ``free`` and this column is width-constrained. What the extra
#: seven cells actually cost, swept across 120->56 columns for both router ids:
#:
#: * NOT id truncation. No router id truncates under ANY label length (0, 5, 6
#:   or 11 cells) at any width in that range — the ids are short enough that
#:   the budget never binds. An earlier version of this comment claimed a long
#:   id truncated ~7 characters earlier; that was wrong, and it mattered
#:   because this is where a future maintainer looks to judge whether a longer
#:   label is affordable.
#: * The DISPLAY-NAME PARENTHETICAL drops earlier, and that is the whole cost.
#:   ``openrouter/openrouter/auto  (Auto Router)`` keeps its name down to 61
#:   columns with ``usage-based`` against a 56-column floor with a 4-6 cell
#:   label — a five-column band, inside which the row loses a secondary aid
#:   and keeps everything it is chosen by.
#:
#: Below ``_NUMBERS_MIN_WIDTH`` (56) the whole numbers run is dropped anyway,
#: so nothing here is reachable further down. The trade is deliberate: the
#: spellings that fit in 4-6 cells (``usage``, ``routed``, ``varies``) all
#: answer a different question than the one a user reading a price column is
#: asking, and a shorter word that has to be decoded is not cheaper than a
#: longer one that does not.
ROUTED_PRICE_LABEL = "usage-based"


def format_price_pair(input_price: float, output_price: float, *, routed: bool = False) -> str:
    """``$3/15`` per million, ``free``, ``usage-based`` for a router, else ``""``.

    FOUR states, and the split matters. A provider that quotes no pricing is NOT
    free — treating a missing price as zero would advertise a paid model as
    free, which is the one error in this column a user would act on. So an
    absent price is blank, only a genuine pair of zeroes says ``free``, and a
    router says :data:`ROUTED_PRICE_LABEL`.

    This function takes no "is it free" flag ON PURPOSE, even though that fact
    now travels as one (:attr:`DiscoveredModel.free`). ``0.0`` reaching here
    ALREADY means "stated free, or a keyless provider whose zero is real":
    ``providers.controller._price`` is the single place that decides, and it
    maps everything else to ``-1.0``. Re-deciding here would be a second, easily
    divergent statement of the same rule — and the reason the ``free`` label was
    dead in the first place was that the two layers disagreed, not that this one
    was wrong.

    ``routed`` IS a flag for the opposite reason, not in spite of it. The float
    vocabulary is full: a router has no price, so it can only reach here as
    ``-1.0``, which already means "nobody quoted this" — a different answer that
    must keep rendering blank. There is no value the caller could pass that
    would let this function work the state out, so the alternative to a flag is
    not "derive it here", it is "invent a fifth sentinel and teach every reader
    of these floats about it". The flag is decided once, by the parser that read
    the wire (:attr:`DiscoveredModel.routed`), and travels; this layer only
    prints it. It is checked FIRST because it is a statement about the endpoint
    rather than about a number, so it cannot be outvoted by a zero a stale
    listing happens to quote.
    """
    if routed:
        return ROUTED_PRICE_LABEL
    if input_price < 0 or output_price < 0:
        return ""
    if input_price == 0 and output_price == 0:
        return "free"
    return f"${_trim_price(input_price)}/{_trim_price(output_price)}"


def _is_parenthesised_tail(name: str) -> bool:
    """Whether ``name`` already ENDS in a parenthetical, so wrapping it doubles up.

    True for ``GLM-5.2 (Token Plan)`` and ``Claude Opus 4.5 (2025-11-01)``; false
    for a plain ``GPT-4.1 mini`` and for anything whose brackets are unbalanced,
    which is where a naive "ends with `)`" test would strip a bracket the name
    needs. The scan is balanced rather than a regex so a nested pair inside the
    qualifier cannot fool it.
    """
    if not name.endswith(")"):
        return False
    depth = 0
    for index, char in enumerate(name):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
            # The outermost group closed before the end: the name continues
            # past its own parenthetical (``(preview) thing``), so it is not a
            # trailing qualifier and must keep the wrapper.
            if depth == 0 and index != len(name) - 1:
                return False
    # A group must actually have opened somewhere after the first character;
    # a name that IS one bracketed span (``(Token Plan)``) is already annotated.
    return depth == 0 and "(" in name


def _trim_price(value: float) -> str:
    """Prices without trailing noise: ``3``, ``0.6``, ``15``, ``18.8``, ``0.075``.

    Rounding is not free here: `$18.75` became `$19`, which reads as a real
    quoted price that the provider does not charge. A column is allowed to be
    terse; it is not allowed to be wrong. That argument does not stop at one
    cent, which is where a flat two decimals broke it — `0.075` printed `$0.07`,
    a 6.7% under-quote on exactly the cheap models a user picks BECAUSE of the
    price, and in one list `gpt-5.2:batch` (0.875) rounded up to `$0.88` while
    `gpt-5.1:batch` (0.625) rounded down to `$0.62`, two adjacent rows resolving
    the same half-cent in opposite directions.

    So: one decimal above ten, and THREE SIGNIFICANT FIGURES below it, which is
    a relative bound rather than an absolute one and therefore says the same
    thing about a $5 model and a $0.05 one. Measured over every price in the
    real models.dev and OpenRouter catalogues (244 distinct values), it drops
    the worst error from 20% to 0.44% and costs the widest pair three cells —
    `$0.0481/0.193` at 13, still inside a column that carries `$1.25/10` beside
    it and clears the 60-column layout, where the id truncates before the
    numbers are touched (``_NUMBERS_MIN_WIDTH``).

    ``%g`` would also switch to exponent form, which is unreadable in a price
    column; it cannot trigger here (it needs a value below 1e-4 or above 1e5,
    and the catalogue's range is 0.017 to 600) but the format is pinned to ``f``
    after the rounding so a future outlier degrades to a long number rather than
    to ``1e-05``.
    """
    if value == int(value):
        return f"{int(value)}"
    if value >= 10:
        return f"{value:.1f}"
    # Round to 3 s.f., then render as a plain decimal and strip the padding.
    # ``math.floor(log10)`` gives the exponent; ``2 - exponent`` is the decimal
    # count that leaves three figures (0.075 -> exp -2 -> 4dp -> "0.0750").
    exponent = math.floor(math.log10(abs(value)))
    decimals = max(0, 2 - exponent)
    return f"{round(value, decimals):.{decimals}f}".rstrip("0").rstrip(".")


#: `(tier, -score, version_key, row)` — the shape `rank_rows` sorts.
_RankEntry = tuple[tuple[int, int], int, tuple[float, float, str], "ModelRow"]


def rank_rows(rows: list[ModelRow], query: str) -> list[ModelRow]:
    """Rows matching ``query``, best first, matched on the DISPLAYED string.

    Matching what the user can see (``provider/id``) rather than the id alone is
    what lets bare names, provider prefixes and scoped queries all flow through
    one matcher.

    SUBSTRING matches win outright when there are any; the subsequence matcher is
    the fallback. Ordering it the other way round is technically a superset and
    practically much worse: ``opus`` is a subsequence of
    ``anthropic/claude-sonnet-4`` (o and p from "anthropic", u from "claude", s
    from "sonnet"), so a user typing the name of one model got a list led by a
    different one. Keeping the fallback is what still resolves ``anthopus`` and
    ``sonnet4``, which are the typo and elision cases fuzzy matching exists for.

    Two tiers come before the score. Connected rows outrank unconnected ones,
    because a model you can use right now beats one that needs a login and
    interleaving them scatters the usable rows through a list of locked ones. Then
    DIRECT providers outrank aggregators: `openrouter/anthropic/claude-opus-5` and
    `anthropic/claude-opus-5` are the same model, and after logging in to Anthropic
    the direct route is the one the user meant.
    """
    needle = query.strip().lower()
    if not needle:
        return sorted(
            rows,
            key=lambda row: (not row.connected, row.aggregated, row.provider, _version_key(row)),
        )
    exact: list[_RankEntry] = []
    fuzzy: list[_RankEntry] = []
    for row in rows:
        target = row.selector.lower()
        score = _score(target, needle)
        if score is None:
            continue
        entry = (
            (0 if row.connected else 1, 1 if row.aggregated else 0),
            -score,
            _version_key(row),
            row,
        )
        (exact if needle in target else fuzzy).append(entry)
    pool = exact or fuzzy
    pool.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in pool]


def _version_key(row: ModelRow) -> tuple[float, float, str]:
    """Sort key placing the NEWEST-looking model first within its tier.

    Alphabetical order on model ids is actively wrong for this catalogue:
    `claude-opus-4-1` sorts before `claude-opus-5` and `gpt-4o` before `gpt-5.4`,
    so a plain sort leads every family with its oldest member — the one a user is
    least likely to be reaching for.

    The version is the FIRST number in the id, not the largest. Taking the largest
    looked equivalent and was not: `kimi-k2-0905` carries 2 and 905, so it scored
    905 and led a list in which `kimi-k3` came ninth. Every id in this catalogue
    puts the family version first and its serials, dates and parameter counts
    after, so position is the reliable signal and magnitude is not.

    Three rungs:

    1. **version**, descending — the first number, with a SHORT run after a dash
       folded in as a minor (`claude-opus-4-1` reads 4.1 and beats
       `claude-opus-4`). The run has to be short, or `claude-sonnet-4-20250514`
       would read as 4.20250514 and outrank every real version in the list.
    2. **remaining numbers**, descending — the dates and serials rung 1 ignores.
       Two snapshots of one model (`-20250514` vs `-20260101`, `-0905` vs bare)
       differ only here, and the later one is what a user wants.
    3. **id**, ascending, so ids with no numbers at all stay in a stable,
       predictable order rather than an arbitrary one.
    """
    normalized = _MINOR_VERSION_PATTERN.sub(r"\1.\2", row.model_id)
    numbers = [float(match) for match in _VERSION_PATTERN.findall(normalized)]
    version = numbers[0] if numbers else 0.0
    return (-version, -max(numbers[1:], default=0.0), row.model_id)


def _score(target: str, needle: str) -> int | None:
    """Subsequence score, or None when ``needle`` is not a subsequence.

    Density is what the score measures: consecutive matched characters are worth
    double, so ``opus`` scores ``claude-opus-5`` far above a model that merely
    happens to contain o, p, u and s in order. An exact substring therefore always
    wins, without needing a separate substring pass.
    """
    if not needle:
        return 0
    score = 0
    previous = -2
    index = 0
    for char in needle:
        found = target.find(char, index)
        if found < 0:
            return None
        score += 2 if found == previous + 1 else 1
        previous = found
        index = found + 1
    # A match that starts at the beginning is a prefix, which is the strongest
    # signal a short query can carry.
    if target.startswith(needle):
        score += len(needle)
    return score


class ModelPicker(Static):
    """The model list shown while the editor holds ``/model <query>``.

    Driven from outside for the keyboard, from its own handlers for the mouse —
    the same split the command picker uses, for the same reason: the caret must
    never leave the text the user is typing.
    """

    def __init__(self, on_choose: Callable[[ModelRow], None]) -> None:
        super().__init__()
        self._on_choose = on_choose
        self._rows: list[ModelRow] = []
        self._matches: list[ModelRow] = []
        self._current: str | None = None
        self._selected = 0
        self._window_start = 0
        self._hovered: int | None = None
        self._query = ""
        self._open = False
        self._status = ""
        # Whether this open has painted a status row (anything but the protected
        # persistent hint). Once it has, the row is kept — blank — until close:
        # the app paints `checking providers…` on the keystroke and clears it
        # when the live list lands, and letting the row collapse shrank the card
        # by one line and dropped everything above it while the user was
        # reading. A blank dim row is less motion than a reflow. Set on paint by
        # `_footer_rows`, which `render_text` reaches as well as `_repaint` —
        # so a bare `render_text` call (the unit tests do this) holds the row
        # too; only `close()` releases it.
        self._status_row_held = False
        # A closed picker takes no layout space at all; `visible: hidden` would
        # reserve the rows and leave a hole above the input.
        self.display = False

    # -- public API ---------------------------------------------------------
    def set_rows(
        self, rows: list[ModelRow], *, current: str | None = None, status: str = ""
    ) -> None:
        """Replace the offered catalogue.

        Callable while the picker is OPEN, because discovery is asynchronous: the
        list paints from whatever is already known and repaints when a provider's
        live fetch lands. The selection is preserved by SELECTOR rather than by
        index so a row arriving above the highlight does not move it under the
        user's fingers mid-keystroke.
        """
        held = self.highlighted_selector()
        self._rows = list(rows)
        self._current = current
        self._status = status
        self._refilter(keep=held)

    def query_text(self) -> str:
        """The filter text currently narrowing the list ("" when unfiltered).

        Exposed so the editor can tell an ACTION key from a FILTER key: every
        printable character here belongs to the query, so a key like `d` may
        only act while the query is empty (see the `d` branch in
        `Editor._on_key`). Without this the shortcut would eat the `d` of
        `deepseek`.
        """
        return self._query

    def is_open(self) -> bool:
        """True when the list is showing."""
        return self._open

    def rows(self) -> list[ModelRow]:
        """Every offered row, unfiltered."""
        return list(self._rows)

    def suggestions(self) -> list[ModelRow]:
        """Current matches, best first (not just the visible window)."""
        return list(self._matches)

    def highlighted(self) -> ModelRow | None:
        """The highlighted row, or None when closed or empty."""
        if not self._open or not self._matches:
            return None
        return self._matches[self._selected]

    def highlighted_selector(self) -> str | None:
        """``provider/id`` of the highlighted row, or None."""
        row = self.highlighted()
        return None if row is None else row.selector

    @property
    def selected_index(self) -> int:
        """Index of the highlight within :meth:`suggestions`."""
        return self._selected

    @property
    def hovered_index(self) -> int | None:
        """Index under the mouse, or None."""
        return self._hovered

    def visible_window(self) -> tuple[int, int, int]:
        """``(start, end, total)`` — which matches the rows are showing."""
        total = len(self._matches)
        end = min(total, self._window_start + self._row_budget())
        return self._window_start, end, total

    def open(self, query: str = "") -> None:
        """Show the list, filtered by ``query``.

        On an EMPTY query the highlight lands on the session's current model. That
        makes the first frame answer "what am I on" as well as "what could I be on",
        and it makes the first Enter a no-op instead of an unrequested switch to
        whatever happened to sort first. A non-empty query is the user having
        already narrowed, so the best match wins as usual.
        """
        self._open = True
        self._query = query
        self._refilter(keep=self._current if not query.strip() else None)

    def set_query(self, query: str) -> None:
        """Re-filter to ``query`` without changing open/closed state."""
        if query == self._query:
            return
        self._query = query
        # The highlight is NOT preserved across a query change: the candidate set
        # is different, so the row under the cursor means a different model. The
        # command picker makes the same choice for the same reason.
        self._refilter()

    def close(self) -> None:
        """Hide the list and release a row's pointer shape."""
        self._open = False
        self._matches = []
        self._selected = 0
        self._window_start = 0
        self._status_row_held = False
        self._hovered = None
        # A stationary pointer gets no mouse-move after this surface leaves;
        # the style observer updates OSC 22 before `display` hides the node.
        self.styles.pointer = "default"
        self.display = False

    def move(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends."""
        if not self._matches:
            return
        self._selected = (self._selected + delta) % len(self._matches)
        self._scroll_to_selection()
        self._repaint()

    def page(self, delta: int) -> None:
        """Move by a windowful, CLAMPED rather than wrapping.

        Paging wraps nowhere in any list users already know, and a PgDn that
        silently returns to the top of a 300-model catalogue looks like the list
        reset itself.
        """
        if not self._matches:
            return
        step = max(1, self._row_budget()) * (1 if delta > 0 else -1)
        self._selected = max(0, min(len(self._matches) - 1, self._selected + step))
        self._scroll_to_selection()
        self._repaint()

    def jump(self, *, to_end: bool) -> None:
        """Home/End, clamped."""
        if not self._matches:
            return
        self._selected = len(self._matches) - 1 if to_end else 0
        self._scroll_to_selection()
        self._repaint()

    def scroll_rows(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows for a WHEEL notch, clamped.

        Deliberately not :meth:`move`: that wraps, which is right for an arrow
        key (a deliberate, discrete press) and wrong for a wheel. A scroll
        gesture that silently teleports from the bottom of a 300-model list
        back to the top reads as the list having reset itself — the same
        reason :meth:`page` clamps.
        """
        if not self._matches:
            return
        self._selected = max(0, min(len(self._matches) - 1, self._selected + delta))
        self._scroll_to_selection()
        self._repaint()

    def choose(self, index: int) -> None:
        """Highlight ``index`` and hand its row to the editor."""
        if not 0 <= index < len(self._matches):
            return
        self._selected = index
        self._on_choose(self._matches[index])

    # -- rendering ----------------------------------------------------------
    def render_rows(self, width: int) -> list[Text]:
        """The visible window as rendered rows."""
        start, end, _ = self.visible_window()
        return [self._row(index, width) for index in range(start, end)]

    def render_text(self, width: int) -> Text:
        """The full renderable: a header, the visible rows, and a footer."""
        out = Text()
        for index, row in enumerate(self._chrome_rows(width)):
            if index:
                out.append("\n")
            out.append_text(row)
        return out

    def _chrome_rows(self, width: int) -> list[Text]:
        return [*self.render_rows(width), *self._footer_rows(width)]

    def _repaint(self) -> None:
        if not self._open or not self.is_mounted:
            return
        width = max(self.size.width, 20)
        rows = self._chrome_rows(width)
        # Pin the height for the same reason ToolCard does: `auto` measures the
        # content against a guessed width before layout and settles one row too
        # tall, which here would push the transcript up by a row per keystroke.
        self.styles.height = max(1, len(rows))
        self.display = True
        self.update(self.render_text(width))

    def _row(self, index: int, width: int) -> Text:
        row = self._matches[index]
        selected = index == self._selected
        hovered = index == self._hovered

        # Hover is additive and selection dominates — the ordering the command
        # picker settled on, where writing it the other way round erased the
        # highlight under the pointer.
        ground = theme_mod.semantic_color("surface")
        if hovered:
            ground = theme_mod.semantic_color("overlay")
        if selected:
            ground = theme_mod.semantic_color("tint-select-hi" if hovered else "tint-select")
        bg = Style(bgcolor=ground)

        # ONE green, and it is the highlight — the same budget line the rest of
        # the app holds. An unconnected row never takes it even when selected:
        # its accent would promise a model that cannot run yet.
        if selected and row.connected:
            id_style = bg + Style(color=theme_mod.semantic_color("accent"))
        elif row.connected:
            id_style = bg + Style(color=theme_mod.semantic_color("fg"))
        else:
            id_style = bg + Style(color=theme_mod.semantic_color("dim"))
        provider_style = bg + Style(color=theme_mod.semantic_color("dim"))
        number_style = bg + Style(color=theme_mod.semantic_color("dim"))
        cursor_style = bg + Style(color=theme_mod.semantic_color("muted"))
        mark_style = bg + Style(color=theme_mod.semantic_color("success"))

        line = Text()
        line.append(f"{_CURSOR} " if selected else " " * _GUTTER_CELLS, style=cursor_style)

        numbers = (
            self._numbers(row) if width >= _NUMBERS_MIN_WIDTH else self._window(row, compact=True)
        )
        mark = f" {_CURRENT_MARK}" if row.selector == self._current else ""
        reserved = _GUTTER_CELLS + _EDGE_MARGIN + cell_len(numbers) + cell_len(mark)
        budget = max(4, width - reserved - (_COLUMN_GAP if numbers else 0))

        prefix = f"{row.provider}/"
        if cell_len(prefix) + 4 <= budget:
            line.append(prefix, style=provider_style)
            line.append(truncate_cells(row.model_id, budget - cell_len(prefix)), style=id_style)
        else:
            # No room for both: the model id is the part being chosen, so the
            # provider prefix is what goes. The selector is still unambiguous in
            # practice because the list is scoped by whatever the user typed.
            line.append(truncate_cells(row.model_id, budget), style=id_style)
        # The DISPLAY NAME, second and parenthesised. The status band shows this
        # string and nothing else once a model is running, so a picker that
        # offered only the selector gave the user two names for one model with no
        # way to connect them. Second rather than first because the selector is
        # what `/model` takes and what the ranking matches, so it stays the row's
        # identity; this only has to be recognisable next to it.
        #
        # Bracketed rather than set off by whitespace because it can NEVER be a
        # column: names start at whatever column each selector ends at (27, 38
        # and 33 on one measured 120-cell frame), and sizing a real column over
        # the visible rows would make it jump on every keystroke — the same
        # failure `_numbers` documents and refuses. A parenthetical reads as
        # deliberate at a ragged origin where a bare second field reads as two
        # columns that failed to line up.
        #
        # Suppressed when it carries nothing the selector already said, which is
        # every model whose name resolution declined to shorten it — those rows
        # would otherwise print their own id twice.
        #
        # Compared case-INSENSITIVELY, which is what the rule above always
        # meant. ChatGPT's Codex catalogue titles its display names off the slug
        # (``gpt-5.6-luna`` -> ``GPT-5.6-Luna``), so an exact comparison let
        # every row in that family print `openai/gpt-5.6-luna  (GPT-5.6-Luna)`
        # and spend ~16 cells restating its own id. Names that genuinely differ
        # keep their parenthetical: `Claude Opus 5` and `GPT-4.1 mini` still
        # differ from their selectors once folded.
        name = row.label.strip()
        if name and name.casefold() not in (row.model_id.casefold(), row.selector.casefold()):
            # Measured against a layout that ALWAYS reserves the numbers run,
            # even at the widths where it is not painted. Sized against the
            # painted layout instead, the annotation GREW as the window shrank:
            # crossing below `_NUMBERS_MIN_WIDTH` freed ~13 cells and handed all
            # of them here, so at 56 columns the row read `Cla…` and at 55 it read
            # `Claude Opus 4.5…`. Content appearing as space disappears is the
            # kind of thing a reader stops trusting a layout over.
            always = self._numbers(row)
            room = (
                width
                - _EDGE_MARGIN
                - cell_len(always)
                - (_COLUMN_GAP if always else 0)
                - cell_len(mark)
                - cell_len(line.plain)
                - _COLUMN_GAP
            )
            # WHOLE OR NOTHING. Truncation keeps the head, and the head of a model
            # name is the vendor word every sibling row already shares: at 60
            # columns two anthropic rows both read `Claude…` while the part that
            # tells them apart — `4.5 (2025-11-01)` — is exactly what was cut. A
            # secondary aid that cannot be read should not spend cells.
            if room >= cell_len(name) + 2:
                line.append(" " * _COLUMN_GAP, style=bg)
                # A name that is ITSELF one parenthetical does not get a second
                # pair: `GLM-5.2 (Token Plan)` read `(GLM-5.2 (Token Plan))`,
                # and the stray `))` looks like a rendering fault rather than a
                # qualifier. Only a name whose trailing `)` closes a group that
                # opens mid-name is unwrapped — `Claude Opus 4.5 (2025-11-01)`
                # qualifies, a hypothetical `(preview) thing` does not, so the
                # brackets can never be dropped from a name that needs them to
                # read as an annotation at all.
                wrapped = name if _is_parenthesised_tail(name) else f"({name})"
                line.append(wrapped, style=provider_style)
        if mark:
            line.append(mark, style=mark_style)
        if numbers:
            gap = max(_COLUMN_GAP, width - _EDGE_MARGIN - cell_len(line.plain) - cell_len(numbers))
            line.append(" " * gap, style=bg)
            line.append(numbers, style=number_style)
        return _pad_to(line, width, bg)

    def _numbers(self, row: ModelRow) -> str:
        """The right-hand metadata run: context window, then price.

        Assembled as one string rather than as padded columns because the window
        is filtered: per-window column alignment would make the numbers jump
        every time the user typed a character, and a stable right EDGE reads
        better than columns that only line up sometimes.
        """
        if not row.connected:
            return "login required"
        parts = [part for part in (self._window(row), self._price(row)) if part]
        return "  ".join(parts)

    def _window(self, row: ModelRow, *, compact: bool = False) -> str:
        if not row.connected:
            return ""
        if (
            row.max_context_window
            and row.default_context_window
            and row.max_context_window != row.default_context_window
        ):
            maximum = f"{format_window(row.max_context_window)} max"
            default = f"provider default {format_window(row.default_context_window)}"
            if row.context_window == row.default_context_window:
                return (
                    f"{format_window(row.context_window)} active"
                    if compact
                    else f"{default} active · {maximum}"
                )
            if compact:
                return maximum
            return f"{maximum} · {default}"
        # Preserve the existing narrow-row layout for models without two limits.
        return "" if compact else format_window(row.context_window)

    def _price(self, row: ModelRow) -> str:
        return format_price_pair(row.input_price, row.output_price, routed=row.routed)

    def _footer_rows(self, width: int) -> list[Text]:
        """Count/status rows with the persistent-default instruction protected.

        Status is assembled upstream from independent `` · `` segments. The
        approved default-model instruction is the only one that must survive
        verbatim at narrow widths, so it gets its own row rather than losing
        ``default`` behind a longer hidden/login clause.
        """
        total = len(self._matches)
        start, end, _ = self.visible_window()
        status = [part for part in self._status.split(_SEAM) if part]
        persistent = next(
            (part for part in status if part.startswith(PERSIST_HINT_PREFIX)),
            "",
        )
        status = [part for part in status if part != persistent]

        bits: list[str] = []
        if total == 0:
            bits.append("no matching models" if self._query.strip() else "no models available")
        elif total > end - start:
            bits.append(f"{end - start} of {total}")
        bits.extend(status)
        ordinary = list(bits)
        if status:
            self._status_row_held = True
        if persistent:
            bits.append(persistent)
        if not bits and not self._status_row_held:
            return []

        dim = Style(
            color=theme_mod.semantic_color("dim"),
            bgcolor=theme_mod.semantic_color("surface"),
        )
        available = max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)
        rows: list[Text] = []
        # The held row renders as an empty string, which `_pad_to` fills, so the
        # card keeps its height when a status clause comes and goes.
        for text in [
            *([_fit_clauses(ordinary, available)] if ordinary or self._status_row_held else []),
            *([truncate_cells(persistent, available)] if persistent else []),
        ]:
            row = Text(" " * _GUTTER_CELLS, style=dim)
            row.append(text, style=dim)
            rows.append(_pad_to(row, width, dim))
        return rows

    # -- window -------------------------------------------------------------
    def _row_budget(self) -> int:
        try:
            screen_height = self.screen.size.height
        except NoScreen:
            screen_height = 0
        if screen_height <= 0:
            return MAX_VISIBLE_ROWS
        # Catalogue status occupies footer chrome. The persistent-default
        # instruction gets a dedicated second row so wider catalogues cannot
        # crowd it out non-monotonically.
        persistent = any(part.startswith(PERSIST_HINT_PREFIX) for part in self._status.split(_SEAM))
        status_rows = 2 if persistent else (1 if self._status or self._status_row_held else 0)
        return max(
            1,
            min(MAX_VISIBLE_ROWS, screen_height // _SCREEN_HEIGHT_FRACTION) - status_rows,
        )

    def _scroll_to_selection(self) -> None:
        budget = self._row_budget()
        if self._selected < self._window_start:
            self._window_start = self._selected
        elif self._selected >= self._window_start + budget:
            self._window_start = self._selected - budget + 1
        self._window_start = max(0, min(self._window_start, max(0, len(self._matches) - budget)))

    def _refilter(self, *, keep: str | None = None) -> None:
        self._matches = rank_rows(self._rows, self._query)
        # The unfiltered first frame answers "what am I on?" and keeps that
        # provider's alternatives beside it. Merely scrolling the highlight to
        # a current model buried at row 300 showed none of its family.
        if not self._query.strip() and self._current:
            current_provider = self._current.partition("/")[0]
            ranked = {row.selector: index for index, row in enumerate(self._matches)}
            self._matches.sort(
                key=lambda row: (
                    (
                        0
                        if row.selector == self._current
                        else (1 if row.provider == current_provider else 2)
                    ),
                    ranked[row.selector],
                )
            )
        self._selected = 0
        if keep is not None:
            for index, row in enumerate(self._matches):
                if row.selector == keep:
                    self._selected = index
                    break
        self._window_start = 0
        self._hovered = None
        self._scroll_to_selection()
        if self._open:
            self._repaint()

    # -- mouse --------------------------------------------------------------
    def on_mouse_move(self, event) -> None:  # noqa: ANN001 - Textual event type
        index = self._index_at(event.y)
        if index != self._hovered:
            self._hovered = index
            self._repaint()
        # Hand pointer over a row only (a click chooses it); the widget's
        # non-row rows keep the default shape. The inline-rule assignment
        # drives `Screen.update_pointer_shape()` through the property's own
        # observer and no-ops when the shape did not change.
        self.styles.pointer = "pointer" if index is not None else "default"

    def on_leave(self) -> None:
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
        self.styles.pointer = "default"

    def on_click(self, event) -> None:  # noqa: ANN001 - Textual event type
        index = self._index_at(event.y)
        if index is not None:
            self.choose(index)

    # The wheel is stopped here rather than left to bubble: this card floats
    # over the transcript, and without `stop()` a scroll aimed at the list
    # also scrolls the conversation behind it, which moves two surfaces for
    # one gesture.
    def on_mouse_scroll_down(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self.scroll_rows(1)

    def on_mouse_scroll_up(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self.scroll_rows(-1)

    def _index_at(self, y: int) -> int | None:
        """Match index under a widget-relative row, or None for the footer."""
        start, end, _ = self.visible_window()
        index = start + y
        return index if start <= index < end else None


def _fit_clauses(clauses: list[str], width: int) -> str:
    """Join footer clauses into ``width`` cells, cutting at a seam before a label.

    Plain ``truncate_cells`` on the joined line is right when the cut lands
    inside a list of values — ``stale list: anthropic…`` still says which list
    is stale and names one provider. It is wrong when the cut lands between a
    label and its first value: the three-clause composite (access note, stale,
    empty) at 100 columns rendered ``… · no live list:…``, a label with no
    payload and a dangling colon that reads as a rendering glitch. A trailing
    clause that cannot keep its label and first value whole is dropped, and
    so is everything after it; the seam is the cut, not the colon. Whatever
    survives then takes the plain cell cut, so every line that fitted before
    this rule renders exactly as it did. The leading clause is never dropped
    — a footer has to say something.

    "First value" is the text up to the first ``, `` (a provider list) or
    `` — `` (the reason behind ``stale list: all providers``, the instruction
    behind ``2 hidden``): the parts that already die first under truncation.
    Never called with the persistent hint, which has its own row.
    """
    kept = list(clauses)
    while len(kept) > 1 and cell_len(_SEAM.join(kept)) > width:
        head = _FIRST_VALUE_END.split(kept[-1], maxsplit=1)[0]
        # A head shorter than its clause is followed by the ellipsis, which
        # `truncate_cells` charges to the same budget.
        tail = 0 if head == kept[-1] else cell_len("…")
        if cell_len(_SEAM.join([*kept[:-1], head])) + tail <= width:
            break
        kept.pop()
    return truncate_cells(_SEAM.join(kept), width)


def _pad_to(row: Text, width: int, style: Style) -> Text:
    """Pad ``row`` to exactly ``width`` cells so its tint spans the full row."""
    missing = width - cell_len(row.plain)
    if missing > 0:
        row.append(" " * missing, style=style)
    return row

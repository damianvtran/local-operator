"""Status line — the full-width BAND on the kit's ``sunken`` ground (D3/D17).

The character refinement supersedes the thin border-row trick: the status
line is a full-width band at the bottom of the input dock, painted on the
``sunken`` ground with segments separated by ``·``:

    π openrouter/kimi-k2 · high · ~/local-operator
                    2 agents · 49.6%/1M · $0.12 · 41m1s · Parser rewrite

(one row on a wide terminal; wrapped here only to fit this docstring)

Left group: brand glyph · provider/model · effort · working dir (+ the
shimmering working indicator while a turn streams — faithful: the working
text rides the shimmer sweep; when shimmer is off a static dim spinner keeps
it legible, D26). Right group, right-aligned: subagents · context usage ·
cost · active duration · conversation name.

**Nine segments do not fit an 80-column terminal, and the band is exactly one
row at every width.** Both facts are load-bearing, so overflow is not left to
truncation — truncation would clip whichever segment happened to be last and
leave a half-written number on screen. Instead :data:`_DROP_LADDER` names an
explicit reduction order, applied step by step until the row fits. The
ordering protects what the operator steers by (provider/model, working dir,
context usage) and spends the transient counters (background jobs, elapsed
time) first. All widths measured with ``rich.cells.cell_len`` (one width
model).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from local_operator.model.naming import model_label as model_label_forms
from local_operator.tui import theme as theme_mod
from local_operator.tui.terminal_title import SPINNER_FRAMES as _SPINNER_FRAMES
from local_operator.tui.terminal_title import SPINNER_INTERVAL_S as _SPINNER_INTERVAL_S
from local_operator.tui.terminal_title import TerminalTitle, TitleState, cwd_label
from local_operator.tui.widgets.tool_card import format_duration, truncate_cells

# Spinner frames and cadence live in ``terminal_title`` and are imported here
# rather than declared twice. The band's spinner and the terminal title's
# animated separator report the SAME fact, and in a tiled terminal both are on
# screen at once — two braille sequences stepping at two cadences read as two
# things happening. Aliased to the module-private names the band has always
# used so the ladder's arithmetic and the tests keep their vocabulary.

#: Segment icons. Every one is a SINGLE cell in ``rich.cells.cell_len`` — this
#: is a hard requirement, not a preference: the band's whole layout is measured
#: arithmetic, and a glyph that renders two cells wide on one terminal and one
#: on another makes the right group's edge alignment drift. Emoji are excluded
#: for exactly that reason (the omp reference uses them; it can afford to).
#: Geometric and technical symbols only, so no Nerd font is required.
ICON_MODEL = "◆"
#: Reasoning effort. A CATEGORY marker like every other icon here — the value is
#: the word beside it (`▴ high`), and the glyph never varies with the rung. A
#: varying glyph would buy one cell of redundancy and cost a legend nobody is
#: given; the owner's own reference reads its effort glyph as a category mark
#: too.
#:
#: It was `◐`, which is the wrong constant for a category: that glyph belongs to
#: a series (`◐◑◒◓`) that encodes FRACTIONS, so a half-filled circle reads as a
#: gauge — and it showed the identical half-filled mark beside `low` and beside
#: `xhigh`, which is either a stuck meter or a mark contradicting the word next
#: to it. A triangle carries no fill to misread.
#:
#: Triangles are also the one glyph family this band does not otherwise use, so
#: it cannot be confused with the model diamond, the cost diamond, the MCP circle
#: or the agents circle. `●` was rejected for the opposite reason: it already
#: means "the current row" in the model picker and "the rung in force" in
#: `/effort`'s listing, and a glyph with two meanings in one feature is worse
#: than a plain one with none.
ICON_EFFORT = "▴"
ICON_CWD = "⌂"
ICON_AGENTS = "◍"
#: Background jobs. Kept inside **Geometric Shapes (U+25xx)** like every other
#: segment icon above, and that block is a hard constraint rather than a
#: stylistic preference. This was `⊞` (U+229E SQUARED PLUS, Mathematical
#: Operators) and it rendered as TOFU — an empty replacement box — in the
#: operator's terminal, while every U+25xx glyph beside it on the same row
#: painted correctly. One band screenshot showed `◆ ◍ ▦ ◈ ◷` all fine and the
#: single U+22xx glyph among them broken, which is what identifies the block
#: rather than the individual codepoint as the problem: terminal fonts that
#: cover Geometric Shapes routinely stop short of Mathematical Operators.
#: `▣` also stays visually distinct from `▦` (U+25A6), which sits directly
#: beside it in the right group — a filled-square icon that reads as a
#: crosshatched one at a glance would make two adjacent readings look alike.
#: **Any new icon here belongs in U+25xx unless it has been checked in a real
#: terminal**, because `cell_len` cannot see this: tofu still measures one cell,
#: so the band's width arithmetic stays correct while the glyph is invisible.
ICON_JOBS = "▣"
ICON_CONTEXT = "▦"
ICON_COST = "◈"
ICON_DURATION = "◷"
#: MCP servers. ``⊙`` (U+2299) is the reference's glyph and measures ONE cell
#: in ``rich.cells.cell_len``, so it satisfies the single-cell rule above; it
#: was checked rather than assumed, because the band's arithmetic is exact and
#: a two-cell glyph would drift the right group's edge by one column.
#:
#: NOTE: this is now the ONLY U+22xx (Mathematical Operators) glyph left in the
#: palette, and so carries the same latent tofu risk that took `ICON_JOBS` above
#: out of that block. It is deliberately left alone: it has not been observed
#: broken, and swapping a working glyph on suspicion would trade a known state
#: for an unknown one. Recorded so that a future report of an empty box next to
#: the MCP count is diagnosed in one step instead of rediscovered.
ICON_MCP = "⊙"
#: Disarmed-gate alarm. Deliberately NOT one of the geometric segment icons
#: above: this one is an ALARM, not a reading, and it reuses the app-wide
#: warning glyph so it reads the same as a warning notice in the transcript.
ICON_APPROVALS = "!"

#: The name's cell BUDGET: the widest box the trailing segment may take, and
#: the narrowest it is worth showing at. Everything between is available — the
#: segment is elastic, and which width it gets is decided by :meth:`_walk` from
#: the row's geometry ALONE, never from the length of the string in it.
#:
#: That last clause is the whole design of this segment, and it is worth stating
#: plainly because the obvious implementation gets it wrong. The name is
#: model-authored and it CHANGES UNDER THE READER: an opener excerpt lands on
#: submit, the generated title replaces it ~1.5 s later, and a re-title can
#: land at any time after that (see ``local_operator.session.naming``). Size the
#: box to the string and every one of those arrivals re-flows the row —
#: measured at a fixed 150 columns, the alarm glyph sat at column 144, then 101,
#: 102, 110 across four consecutive frames, two of those moves with no user
#: input at all, and a ONE-cell difference in title length flipped the whole
#: 13-cell `◍ 2 agents` segment in and out. Size the box to the geometry and the
#: name is the only thing on the band that a title can change: the box is
#: reserved, the name is left-aligned in it, and the leftover cells sit at the
#: band's own edge where nothing else lives.
#:
#: 40 for the cap: it fits every title the naming caps allow (``MAX_TITLE_CHARS``
#: is 80, but a 3-7 word answer lands well inside 40) and leaves the identity
#: group its ordinary width at 120 cells.
#:
#: 18 for the floor is a MEASURED choice, not a round number: it sets the width
#: below which the name cannot appear at all. Swept on a fully populated band
#: (model, effort, cwd, healthy MCP, context, cost, the alarm) the floor runs
#: 14 → 90 cells, 18 → 94, 22 → 98, 26 → 101. Anything above 22 loses the name
#: on a 100-column terminal, which is the width this feature is mostly read at,
#: and 14 leaves too little of a title to tell two sessions apart. 18 is the
#: widest floor that still survives 100 columns with everything else on screen.
NAME_CELLS = 40
NAME_CELLS_FLOOR = 18

#: Separators point INWARD: the left group's chevrons aim right and the right
#: group's aim left, so both runs converge on the centre gap and frame it. A
#: symmetric separator (the middot this replaced) left the two groups reading as
#: one long run once the gap narrowed.
_SEP_LEFT = "›"
_SEP_RIGHT = "‹"

#: Legacy separator, still used by nothing in the band. Kept only because the
#: intra-group width arithmetic in :data:`_MIN_GROUP_GAP` is stated relative to
#: it in the comments below.
_SEPARATOR = " · "

#: Minimum cells between the left and right groups. Deliberately WIDER than the
#: 3-cell intra-group separator so the seam between the groups always dominates
#: the seams inside them — otherwise the band's identity-left/numbers-right
#: architecture dissolves into one run at ordinary widths.
_MIN_GROUP_GAP = 4

#: Reduction order, cheapest loss first. Each entry is applied in turn until
#: the row fits the available width.
#:
#: The ladder DROPS, SHORTENS, and — for the name only — FLEXES. The name is
#: still on it twice, but the first entry is no longer a step from one fixed
#: width to another: ``flex-name`` is where the segment stops being a rigid box
#: of :data:`NAME_CELLS` and becomes elastic down to its floor, taking back
#: whatever the row needs and giving back whatever it does not. What remains
#: below is its DROP — the point where even the floor has stopped fitting.
#:
#: The ordering principle: shed what the user already knows or can re-derive,
#: and protect what predicts their next decision. In order —
#:
#: * ``duration`` is re-derivable from the transcript.
#: * ``subagents`` is a counter, but NOT re-derivable without scrolling, which
#:   is why it outlasts the one above rather than going first.
#: * ``flex-name`` comes next, and it comes AFTER those three drops for the
#:   reason its predecessor did: they cost the user nothing they cannot
#:   re-derive, and narrowing the one field that says which conversation this is
#:   in order to keep a counter would be the wrong way round.
#: * the two SHORTEN steps outrank every remaining drop, because they recover
#:   width while keeping the segment: a basename still answers "where am I" and
#:   a bare model id still answers "who is replying". Together they free ~35
#:   cells on a realistic label, more than cost and context cost combined, so
#:   dropping numbers to preserve a fully-qualified path would be a bad trade.
#: * ``effort`` is a static setting the user chose. It does not change while
#:   they watch, so it goes before either live number — an earlier version had
#:   it OUTLIVING context usage, which meant a band could show `high` but not
#:   `49.6%/1M`: it kept the field nobody re-reads and dropped the one that
#:   says compaction is coming.
#: * ``name`` drops once the floor has stopped fitting, ahead of the numbers: it
#:   is still the widest thing left, and a title predicts no action. The terminal
#:   tab still carries it, which is where it earns its keep anyway. Dropping it
#:   RE-WALKS the ladder from the top — see :meth:`StatusLine._fit`.
#: * ``cost``, then ``cwd``, then ``context``, then ``mcp``: context usage is the
#:   one an operator acts on, and the MCP indicator outlives even that WHEN IT IS
#:   AN ALARM — a healthy count is only a courtesy, so it sheds like one; see
#:   :func:`drop_ladder`.
#: * ``approvals`` is last in every variant: one cell, and an alarm. See the rung.
#:
#: The brand glyph, the streaming spinner and the model label are NEVER dropped.
#: When even the irreducible row overflows, ``_render`` emits spinner, glyph and a
#: TRUNCATED label rather than shedding any of them: `deepse…` still answers which
#: model is replying, a band reduced to a bare glyph on an empty tinted strip
#: reads as broken rather than as compressed, and a streaming band that renders
#: identically to an idle one is the one thing the colour budget must never allow.
_DROP_LADDER: tuple[str, ...] = (
    # Re-derivable from the transcript.
    "duration",
    # Counters. Not re-derivable without scrolling, so they outlast the one
    # above; jobs go before agents because a backgrounded shell command is
    # visible in the transcript as a tool card, while a subagent is not.
    "jobs",
    "subagents",
    # The name stops being a rigid box here and becomes elastic: from this rung
    # down it is measured at its FLOOR and handed back whatever the fitting row
    # left over, so its width runs continuously from 18 cells to 40 instead of
    # jumping between two values. This rung replaced a `shorten-name` that halved
    # it outright, and the halving is what made the segment idle: across 30-180
    # columns the name rendered at only 0, 18 or 40 cells, so a 130-column
    # terminal showed an 18-cell stub beside an 18-cell hole with 32 cells
    # available, and the excerpt-to-title moment this feature exists for read as
    # `The /resume picke…` → `Fix unnamed sessi…` at every width under 138.
    #
    # The flex is measured from the geometry, never from the string: a title is
    # model-authored and lands while the user is reading, so a box sized to it
    # would re-flow the row on its own. See :data:`NAME_CELLS`.
    "flex-name",
    # Shorten before dropping: a basename still answers "where am I" and a bare
    # model id still answers "who is replying".
    "shorten-cwd",
    "shorten-model",
    # A static session setting the user chose — it does not change while they
    # watch, so it goes before either live number.
    "effort",
    # The name's own drop, and the one rung that is not simply a step down. By
    # here the segment has already given back every cell between its cap and its
    # floor, and the three drops above were made to keep that floor on screen; if
    # the floor still does not fit, those concessions bought nothing. So this
    # rung is a RESTART rather than a step: ``_fit`` re-walks the whole ladder
    # with the name off the table, which hands `▴ high`, `◍ 2 agents` and the
    # duration back. Without the restart the walk was monotone and could not roll
    # a concession back, and a named session at 83-94 columns rendered strictly
    # worse than the same session unnamed — no effort segment, no name either,
    # and 9 more cells of hole in exchange.
    #
    # Its place in the order is unchanged: below it are only figures an operator
    # acts on, which is where a title stops competing.
    "name",
    # The one figure that is not a live reading of this turn.
    "cost",
    # The working directory goes before the context number. Its shorten step has
    # already reduced it to a basename by now, so what remains is ~7 cells of
    # "where am I" against ~9 cells of "how close is compaction" — and the second
    # is the one that predicts the operator's next action. An earlier version had
    # these the other way round, which contradicted this very ladder's rationale.
    "cwd",
    "context",
    # Second-to-last. Its failure branch is an ALARM: the danger-tinted glyph is
    # the only place the band admits the agent is missing tools it was
    # configured to have, and a cramped terminal is exactly where a user would
    # otherwise conclude the tools were never configured. Kept in the ladder
    # rather than omitted from it so the very narrowest widths still get one
    # graceful aligned step before ``_render`` falls back to the truncated tail.
    "mcp",
    # DEAD LAST, in every variant. This rung only EXISTS while the tool-approval
    # gate is disarmed, and it is now a BARE GLYPH — one cell, the narrowest
    # thing in the band by a factor of seven. Last place goes to the narrowest
    # bounded rung and an alarm outranks a reading; `!` is both, so it wins
    # outright and no variant needs a repair to keep it there.
    #
    # It used to be first, which looks like the opposite judgement and was the
    # same one: the segment then spelled out `! auto-approve always` at up to 20
    # cells, making it the WIDEST rung, and this ladder sheds by what a drop
    # buys. Shrinking it to the glyph moved it from one end of that argument to
    # the other, and took its ink with it — one cell of `warning` amber was the
    # same ink as `◈ $73.92` three cells away, so `danger` carries it now (see
    # :data:`ICON_APPROVALS`). What has not changed is the reason it is here at
    # all: nothing else in the UI says the gate is disarmed for longer than a
    # scroll — a saved `tool_approval_mode: auto` is adopted at boot with no
    # notice at all (see ``_adopt_saved_approvals``), so this is the only
    # standing indication that no tool will ask before it runs.
    "approvals",
)


#: Rungs whose painted width is not knowable from the ladder. ``cwd`` is as wide
#: as the user's path — a basename alone runs from 3 cells to 30 — so it can
#: never be relied on as the thing that still fits at the narrowest width. A
#: render walk only ever sheds (``_walk`` cannot add a segment back, and the one
#: restart ``_fit`` makes starts a fresh walk rather than un-shedding one), so a
#: ladder ending on one of these sheds a bounded segment to make room for a path
#: that then does not fit either, and paints neither. No ladder may end on one;
#: ``test_status_line`` holds every variant to it.
_UNBOUNDED_RUNGS = frozenset({"cwd"})


#: Last place is not repaired any more, and that is worth recording because a
#: whole helper used to do it. Promoting a rung out of the tail leaves whatever
#: FOLLOWED it at the end, and while ``approvals`` spelled out
#: `! auto-approve always` it was the widest rung in the band, authored first,
#: so every promotion stranded a 20-cell alarm in the one slot reserved for the
#: narrowest thing that reliably fits. The old fix re-seated it behind ``mcp``
#: whenever ``mcp`` would have landed last — a rule with an exception, a
#: docstring longer than the ladder, and four ladder variants to keep honest.
#:
#: The alarm is one cell now (see the ``approvals`` rung), which makes it the
#: narrowest bounded rung outright as well as an alarm — both halves of the rule
#: last place was always meant to follow. So it is simply authored last, every
#: promotion leaves it there, and the repair has nothing left to repair.


def _mcp_before_cwd(ladder: tuple[str, ...]) -> tuple[str, ...]:
    """``ladder`` with the mcp rung moved to just ahead of ``cwd``.

    A healthy `⊙ 2 MCP` is a nicety, and last place had it outranking both the
    working directory and the full model label. The rung moves rather than the
    render site special-casing the segment.
    """
    rungs = [step for step in ladder if step != "mcp"]
    rungs.insert(rungs.index("cwd"), "mcp")
    return tuple(rungs)


#: The ladder for a band whose MCP segment is NOT an alarm. Precomputed rather
#: than rebuilt per render: ``_render`` walks a ladder on every repaint, and the
#: spinner repaints it eight times a second.
_DROP_LADDER_QUIET: tuple[str, ...] = _mcp_before_cwd(_DROP_LADDER)


def _context_before_cwd(ladder: tuple[str, ...]) -> tuple[str, ...]:
    """``ladder`` with the context rung moved to just ahead of ``cwd``.

    The full order puts cwd first because the context number "predicts the
    operator's next action". That is true of a number the model reported, and
    false of the boot ESTIMATE: before a single turn, nothing has been spent,
    nothing is approaching compaction, and the figure is a static property of a
    session that has not done anything yet. Meanwhile "where am I" is exactly
    as load-bearing as it always was — more so, since a fresh session is when a
    user is most likely to be checking they opened the right directory.

    Shedding the estimate first is therefore the same trade ``_mcp_before_cwd``
    exists to make: rank by what the segment is worth RIGHT NOW, not by what
    its slot is usually worth.
    """
    rungs = [step for step in ladder if step != "context"]
    rungs.insert(rungs.index("cwd"), "context")
    return tuple(rungs)


#: Estimate variants of both ladders, precomputed for the same reason.
_DROP_LADDER_ESTIMATE: tuple[str, ...] = _context_before_cwd(_DROP_LADDER)
_DROP_LADDER_QUIET_ESTIMATE: tuple[str, ...] = _context_before_cwd(_DROP_LADDER_QUIET)


def drop_ladder(status: McpStatus, *, context_estimated: bool = False) -> tuple[str, ...]:
    """Which reduction order this band uses, given its MCP state.

    The mcp rung's place is earned by the ALARM, not by the segment. Unconditional
    last place meant a healthy `⊙ 2 MCP` outranked both the working directory and
    the full model label: at 40 cells the band read `◆ model › ⊙ 2 MCP` where the
    same terminal with no MCP configured showed `◆ test/model › ⌂ local-operator`.
    A healthy count is a nicety and a failed one is the only warning the band
    carries, so the rung moves rather than the segment being special-cased at the
    render site.

    ``context_estimated`` demotes the context rung for the same kind of reason —
    see :func:`_context_before_cwd`.
    """
    if mcp_semantic(status) == "danger":
        return _DROP_LADDER_ESTIMATE if context_estimated else _DROP_LADDER
    return _DROP_LADDER_QUIET_ESTIMATE if context_estimated else _DROP_LADDER_QUIET


def format_context_tokens(tokens: int) -> str:
    """Compact context estimate: ``12.4k`` / ``1.2m`` style, plain under 1k."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}m"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def format_window(window: int) -> str:
    """Abbreviate a context window for the denominator: ``1M``, ``200k``.

    Capital ``M`` and lower-case ``k`` are the conventional units for model
    windows, and a whole window renders without a decimal (``1M``, not
    ``1.0M``) — the denominator is a label, not a measurement.
    """
    if window >= 1_000_000:
        scaled = window / 1_000_000
        return f"{scaled:.0f}M" if scaled == int(scaled) else f"{scaled:.1f}M"
    if window >= 1_000:
        scaled = window / 1_000
        return f"{scaled:.0f}k" if scaled == int(scaled) else f"{scaled:.1f}k"
    return str(window)


def format_context_usage(tokens: int, window: int) -> str:
    """Context usage as ``49.6%/1M``; ``12.4k/—`` when the window is unknown.

    A percentage needs a denominator. When the registry has no window for the
    model, showing a bare percentage would invent one, so the spend is reported
    against an explicit unknown instead.

    The em dash rather than ``?`` because it is already this UI's glyph for an
    absent value — unknown cost renders ``$—`` and a provider with no stored
    credential renders ``—``. Two spellings of "unknown" in one row would read
    as two different states.
    """
    return context_spelling(tokens, window, form="full")


#: Context sizes at which the reading changes colour, largest first. ABSOLUTE
#: token counts, because the cost they warn about is absolute: 300k tokens is
#: slow and expensive to re-send on every request whether the window is 1M or
#: 200k, and a purely proportional ramp would leave a big window looking calm
#: at exactly the size that hurts most. This mirrors the ladder omp shows for
#: the same reading.
#:
#: These are only half the rule — see :data:`CONTEXT_COLOR_WINDOW_BANDS` for
#: the proportional half, which is what makes the ramp reachable at all on the
#: 200k-and-under models that are most of the registry.
#:
#: The hues are the palette's existing semantics rather than new tokens, so
#: both themes get a legible ramp for free: `signal` (blue) is the neutral
#: reading, `label` (purple) marks a context worth noticing, and `danger`
#: (red) marks one at or past the point where compaction is due.
CONTEXT_COLOR_BANDS: tuple[tuple[int, str], ...] = (
    (500_000, "danger"),
    (200_000, "label"),
)

#: Fractions of the context window that warm the reading to the same rungs,
#: applied as a UNION with the absolute bands above (whichever is warmer wins).
#:
#: The absolute bands alone cannot do the segment's job, because 70% of the
#: registry's windowed models are 200k or smaller: on those the reading could
#: never leave the base colour, so it stayed calm blue at 100% full with
#: compaction already overdue (the trigger for a 200k window is 160k). Once
#: colour means "how full am I", a calm blue at 100% is not neutral, it is
#: wrong. The fractions are set just under and at the compaction trigger's own
#: default (``threshold_percent`` 0.80) so the warm rung appears while there is
#: still headroom and the top rung coincides with the pass becoming due.
CONTEXT_COLOR_WINDOW_BANDS: tuple[tuple[float, str], ...] = (
    (0.80, "danger"),
    (0.55, "label"),
)

#: Colour of a context reading below every band in :data:`CONTEXT_COLOR_BANDS`.
CONTEXT_COLOR_BASE = "signal"

#: Rungs warmest-first, so a union of two ladders can be resolved by rank.
_CONTEXT_COLOR_RANK: tuple[str, ...] = ("danger", "label", CONTEXT_COLOR_BASE)


def context_semantic_color(tokens: int, window: int = 0) -> str:
    """Semantic colour name for a context reading of ``tokens``.

    The warmer of two ladders wins: an ABSOLUTE token count
    (:data:`CONTEXT_COLOR_BANDS`) and a FRACTION of ``window``
    (:data:`CONTEXT_COLOR_WINDOW_BANDS`). Each guards a failure the other
    cannot see, which is why this is a union rather than a choice:

    - The absolute ladder is what makes a very large window legible. Re-sending
      300k tokens on every request is slow and expensive whether the window is
      1M or 200k, and a purely proportional ramp would leave a 1M session
      looking calm at exactly the size that costs the most per turn.
    - The proportional ladder is what makes a SMALL window legible, and without
      it the segment was inert on most models: at 200k and below the absolute
      rungs are unreachable, so the reading stayed blue through 100% full while
      compaction was already due.

    ``window <= 0`` (unknown) falls back to the absolute ladder alone, since a
    percentage needs a denominator — the same rule the reading's text follows.

    Strictly greater-than on both ladders, so a value sitting exactly on a
    boundary keeps the calmer colour and a number hovering there does not
    flicker between two hues.
    """
    color = CONTEXT_COLOR_BASE
    for threshold, candidate in CONTEXT_COLOR_BANDS:
        if tokens > threshold:
            color = candidate
            break
    if window > 0:
        for fraction, candidate in CONTEXT_COLOR_WINDOW_BANDS:
            if tokens > window * fraction:
                # Warmest wins: rank is warmest-first, so a lower index is warmer.
                if _CONTEXT_COLOR_RANK.index(candidate) < _CONTEXT_COLOR_RANK.index(color):
                    color = candidate
                break
    return color


#: The forms a context reading may be written in, widest first. Named rather
#: than inlined at each site because the band and the subagent rows show the
#: SAME child's reading four rows apart, and they were spelling it two ways —
#: `▦ 24.1%/200k` above `24%/200k`. Cost never had that problem because both
#: sides go through `format_cost` and differ only by a documented magnitude
#: rule; this gives the context reading the same treatment.
#:
#: * ``full``  — ``24.1%/200k``. What the band always shows.
#: * ``nodec`` — ``24%/200k``. Two cells cheaper, denominator kept, so two
#:   children on different windows stay comparable.
#: * ``short`` — ``24%``. Seven cells cheaper; the denominator is the half a
#:   reader can recover by opening the child's page.
#:
#: Only the rows reduce. The band has one segment and the cells to spell it
#: whole, so a band that shortened would be shortening for no reason.
CONTEXT_FORMS = ("full", "nodec", "short")


def context_spelling(tokens: int, window: int, *, form: str = "full") -> str:
    """One context reading in one of :data:`CONTEXT_FORMS`.

    Empty when there is nothing to report, so every caller's segment
    disappears on the same test. A percentage needs a denominator: with no
    window the spend is reported against an explicit unknown rather than
    against an invented one, and that is true of every form.

    A rounded ``0%`` over a non-zero reading is refused in all of them, for
    the reason a confident ``$0.0000`` over real spend is refused — a child
    holding three thousand tokens is not holding none.
    """
    if tokens <= 0:
        return ""
    if window <= 0:
        return (
            f"{format_context_tokens(tokens)}/—"
            if form != "short"
            else (format_context_tokens(tokens))
        )
    percent = tokens / window * 100
    if form == "short":
        return f"{percent:.0f}%" if percent >= 1 else "<1%"
    denominator = format_window(window)
    if form == "nodec":
        return f"{max(percent, 0.5):.0f}%/{denominator}"
    if percent < 0.05:
        return f"<0.1%/{denominator}"
    return f"{percent:.1f}%/{denominator}"


def format_cost(cost: float) -> str:
    """Compact dollar cost: ``$0.0021`` under a cent, ``$0.12`` above."""
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 1.0:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


def format_agents(count: int) -> str:
    """``1 agent`` / ``3 agents``; empty at zero (the segment disappears).

    Zero renders as nothing rather than "0 agents": a band that permanently
    reports the absence of background work is nine characters of noise, and
    the segment appearing is itself the signal.
    """
    if count <= 0:
        return ""
    return f"{count} agent" if count == 1 else f"{count} agents"


def format_jobs(count: int) -> str:
    """``2 jobs`` — backgrounded shell work, empty at zero.

    Separate from the agent counter because they are different things an
    operator tracks: a subagent is delegated reasoning, a job is a process this
    session started and can outlive the turn. omp shows both, and collapsing
    them into one number would hide which kind is running.
    """
    if count <= 0:
        return ""
    return f"{count} job" if count == 1 else f"{count} jobs"


@dataclass(frozen=True)
class McpStatus:
    """The facts the MCP segment renders from.

    Passed as ONE value rather than four ``update`` keywords: the band's
    update signature is already at ten segments, and the count is meaningless
    without knowing whether anything was configured — splitting them invites a
    caller to set one and leave the others stale.

    ``configured`` is how many servers the config asked for, ``connected`` how
    many are actually up right now, and ``failed`` whether any of them did not
    come up. All three are read LIVE from the manager; ``configured`` alone
    decides whether the segment carries a COUNT, and ``discovery_failed``
    (below) is the one state that renders without one.
    """

    configured: int = 0
    connected: int = 0
    failed: bool = False
    #: Discovery itself failed, so there is no server list to count. Distinct
    #: from ``configured == 0``, which means "this machine does not use MCP":
    #: here the machine HAS a config and it could not be read, and a segment
    #: hidden for the same reason as an unconfigured one would report a broken
    #: setup as an absent feature.
    discovery_failed: bool = False


def format_mcp(status: McpStatus) -> str:
    """``3 MCP`` — connected SERVERS, never pluralised; empty when unconfigured.

    The count is servers rather than tools because a server is the thing that
    can drop: "31 MCP" would tell an operator nothing about the github server
    having died, which is the event this segment exists to show. The tool total
    belongs in the startup toast, where there is room to say both.

    ``MCP`` stays singular the way ``2 jobs`` does not, because it is an
    initialism naming the protocol, not a count of a noun — the reference
    renders it that way and "3 MCPs" reads as three protocols.

    Zero CONFIGURED servers renders nothing: a permanent ``⊙ 0 MCP`` on a
    machine with no ``.mcp.json`` is seven cells asserting the absence of a
    feature the user never asked for. Zero CONNECTED with servers configured
    does render, because that is a real and interesting state.

    A DISCOVERY failure renders the bare initialism with no number. The config
    layer never produced a server list on that path, so every count would be a
    fiction — ``0 MCP`` in particular would claim the machine asked for nothing,
    which is the opposite of what happened. The toast that reports the failure
    dismisses itself after ten seconds; without this the band it leaves behind
    is indistinguishable from a machine with no MCP at all.
    """
    if status.discovery_failed:
        return "MCP"
    if status.configured <= 0:
        return ""
    return f"{status.connected} MCP"


def mcp_semantic(status: McpStatus) -> str:
    """Which semantic tint the ``⊙`` glyph carries.

    An ALARM OR NOTHING. Only the failure branch gets a colour of its own:
    `danger` when a configured server did not come up, and the band's neutral
    ramp otherwise. FAILURE WINS even when other servers connected — a partial
    outcome is the dangerous one, because a `⊙ 2 MCP` that looks fine on a
    machine where a third server died costs the user a turn wondering why the
    agent cannot reach that server's tools.

    The healthy rung used to be `success` #57c785. That is 5.08 dE2000 from the
    accent #38c96a — for scale, this file's own comments reject 3.06 as
    imperceptible, and this glyph is ONE cell. The band's single accent site is
    the running indicator, so a healthy count put a second, indistinguishable
    green ten cells from the spinner and "green means a turn is live" stopped
    being true. The neutral ramp says the same three things without spending the
    accent: `muted` once something is up, `dim` while nothing is (16.66 dE2000
    apart, and both are already band vocabulary).

    Public because the startup toast paints the same lamp: two surfaces deriving
    the same state independently is how they end up disagreeing, and a toast
    saying green while the band says red is worse than either alone.
    """
    if status.failed or status.discovery_failed:
        return "danger"
    if status.connected > 0:
        return "muted"
    return "dim"


def format_cwd(cwd: str, *, short: bool) -> str:
    """The working dir as ``~/rel/path``, or just its basename when ``short``.

    Home-relative beats absolute (the ``/Users/<name>`` prefix is the same on
    every row of every session) and beats the bare basename, which is
    ambiguous the moment two checkouts share a directory name — the basename
    is what the overflow ladder falls back to, not what it starts from.
    """
    if not cwd:
        return ""
    path = Path(cwd)
    if short:
        return path.name or cwd
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        # Outside the home tree (/tmp, /opt, a mounted volume): the absolute
        # path is already the shortest honest rendering.
        return str(path)


def format_model_label(label: str, *, short: bool, name: str = "") -> str:
    """A model's human name, or its selector when no name can be trusted.

    ``label`` is the selector the rest of the app identifies a model by
    (``anthropic/claude-opus-5``); ``name`` is the resolved ``ModelInfo.name``
    when the caller has one, and is what lets an aggregator id no registry row
    covers still read as ``MoonshotAI: Kimi K2``. Callers without it still get
    the curated name, which is every direct provider.

    Both the choice of name and the ``short`` narrowing are
    ``model/naming.py``'s: it refuses a name that two models answer to, so the
    23 cells of ``anthropic/claude-opus-5`` become the 13 of ``Claude Opus 5``
    without the band ever printing a string that fails to say which model is
    replying. Where it refuses, the selector comes back and ``short`` keeps only
    the last path segment — the behaviour this segment has always had, now
    reached only when a name would have been a guess.
    """
    if not label:
        return ""
    provider, _, model_id = label.partition("/")
    forms = model_label_forms(provider, model_id, name)
    return forms.compact if short else forms.full


def truncate_name(name: str, cells: int) -> str:
    """The session name cut to ``cells``, on a word boundary where one is close.

    ``truncate_cells`` alone is width-aware and word-blind, and the string it is
    handed here was cut on a word boundary ON PURPOSE one call earlier:
    ``provisional_title`` stops on a word "so it reads as a quotation rather than
    as a string that ran out of buffer", and re-cutting it mid-word throws that
    away — `The /resume picker shows (unnamed session) for…` came back as `The
    /resume picker shows (unnamed sessi…`.

    A word cut is only taken when it costs less than a third of the box, which is
    the same trade ``resume._condense`` makes for the picker's row labels (there
    stated as a fixed 12 characters against a fixed cap; here proportional,
    because this box is elastic and runs from 18 cells to 40). Past that the
    segment has stopped doing its one job: `Add todo…` in an 18-cell box tells two
    tiled sessions apart less well than `Add todo guardrai…` does, and half a box
    of blank is not a tidier cut, it is a smaller one.

    Trailing punctuation goes with the cut word for the same reason ``_condense``
    strips it: `(unnamed session),…` reads as a rendering fault.
    """
    if cells <= 0 or cell_len(name) <= cells:
        return name[:cells] if cells <= 0 else name
    cut = truncate_cells(name, cells)
    stem = cut[:-1] if cut.endswith("…") else cut
    spaced = stem.rsplit(" ", 1)[0].rstrip(" ,.;:") if " " in stem else ""
    if spaced and cell_len(spaced) * 3 >= cells * 2:
        return spaced + "…"
    return cut


@dataclass(frozen=True)
class SubagentBand:
    """The CHILD's readings, shown in place of the session's own.

    While the full-page subagent view is up, the band four rows under it was
    still describing the PARENT — its model, its context, its spend — over a
    frame whose entire subject is a different session. A child frequently runs
    a different model than its parent (``run_subagent`` takes a ``model_spec``
    override), so that was not merely redundant, it was wrong.

    An OVERLAY rather than a save-and-restore of the band's own fields,
    because the parent does not stop while the page is open: its turn keeps
    ending, and every ``on_turn_ended`` writes a fresh cost and context into
    those fields. Saving them on the way in would hand back a snapshot that
    went stale in the reader's hand. Shadowed, the parent's numbers stay live
    underneath and are revealed intact the moment the overlay is dropped.

    Every field is OMITTABLE and empty means omitted, never zero: a child that
    has reported no usage has no context reading and no spend, and the band's
    own segments already disappear on exactly those emptiness tests. Printing
    the parent's figure in a child's frame is the one thing this class exists
    to prevent, so there is no field here that can fall back to it.
    """

    #: Never empty in practice: a child with no recorded model of its own is
    #: running the parent's, so the caller resolves that before constructing
    #: this. The band keeps the model segment down every rung of the ladder,
    #: and "which model is replying" has an answer here even when nothing else
    #: does — with ONE exception, which is the narrow end of an overlay. Below
    #: the last aligned rung the irreducible row carries `◍ <name>` alone
    #: (:meth:`StatusLine._render`), because a child's model with no owner
    #: beside it reads as the SESSION's model and misattributes the one fact
    #: this class exists to keep straight. The name is the segment that never
    #: drops; the model yields to it, and only there.
    model_label: str = ""
    #: The child's own name, which REPLACES the parent's running-agent counter
    #: while the overlay is up. Without it the band interleaves three owners
    #: with no mark — the model, context, cost and duration are the child's,
    #: the cwd and MCP are shared, and the parent's `◍ 2 agents` sat between
    #: them. Naming the child at the head of the right-hand group makes
    #: everything after it belong to one session, and the count it displaces
    #: is redundant to a reader who is already inside one of those agents and
    #: has the whole list four rows below.
    label: str = ""
    context_tokens: int = 0
    context_window: int = 0
    #: Already formatted, because the vocabulary for "billed but unpriceable"
    #: (``$—``) belongs to the caller that knows the difference between a
    #: child that spent nothing and one nobody can price.
    cost: str = ""
    #: The child's own age. ``None`` leaves the segment off rather than
    #: reporting the parent's cumulative active time under a child's title.
    duration: float | None = None


class StatusLine:
    """Draws status segments into the bottom status band widget.

    Owns no layout row: the band widget (a ``Static`` docked at the bottom of
    the input dock) is repainted through :meth:`refresh`. Call :meth:`update`
    whenever a segment changes; call :meth:`refresh` after resizes so the
    overflow ladder follows the new width (one width model:
    ``rich.cells.cell_len``).

    Every :meth:`update` parameter is optional and ``None`` means "leave this
    segment unchanged" — callers assemble partial updates from whatever the
    event they are handling actually knows.
    """

    def __init__(self, dock: Static, *, clock: Callable[[], float] = time.monotonic) -> None:
        # A `Static`, not a bare `Widget`: the band is repainted by handing
        # it a rich renderable, which only content widgets accept.
        self._dock = dock
        # Injected so duration is testable without sleeping. Monotonic, not
        # wall clock: a duration that jumps when the system clock is adjusted
        # is worse than no duration.
        self._clock = clock
        self._model_label: str = ""
        # The resolved ``ModelInfo.name`` for the selector above, "" when the
        # host could not supply one. Kept beside the selector rather than
        # replacing it because the selector is still what the segment falls back
        # to, and because a caller that knows neither (the boot frame, before a
        # session exists) must not have to invent one.
        self._model_name: str = ""
        self._effort: str = ""
        self._cwd: str = ""
        self._context_tokens: int = 0
        # Provenance of the reading above: True while it is the host's LOCAL
        # estimate of what is already loaded, False once a provider's exact
        # prompt_tokens has replaced it. Lives here rather than on the app
        # because both readers need it — the app for precedence, the band for
        # how hard to fight to keep the segment on screen.
        self._context_is_estimate: bool = False
        self._context_window: int = 0
        self._subagents: int = 0
        self._jobs: int = 0
        self._streaming: bool = False
        self._cost: str = ""
        self._conversation_name: str = ""
        self._mcp: McpStatus = McpStatus()
        # Which segments the drop ladder shed on the LAST render. Every segment
        # is dropped until something has been rendered, which is the honest
        # starting state: nothing has been shown yet.
        self._dropped: frozenset[str] = frozenset(_DROP_LADDER)
        # True while the tool-approval gate is disarmed — by the prompt's "allow
        # all", by `/approvals auto`, or by a saved default adopted at boot: a
        # session-wide mode with no persistent indicator is how a disarmed gate
        # gets forgotten about.
        self._approvals_auto: bool = False
        # True while the turn is parked on something the USER owes (currently
        # the tool-approval prompt). Only the terminal title reads it; see
        # :meth:`set_attention` for why it is not inferred from `_streaming`.
        self._attention: bool = False
        # And whether that mode is the SAVED default rather than this session's
        # choice. Two different promises — "until I close this window" and
        # "every window from now on" — and the alarm is the only place a user
        # sees either without running a command.
        self._approvals_always: bool = False
        # Cumulative ACTIVE processing time: the sum of turn durations, not
        # wall clock since launch. A session left open over lunch has not
        # been working for two hours, and reporting that it has makes the
        # number useless for judging what a task actually cost.
        self._active_seconds: float = 0.0
        self._turn_started_at: float | None = None
        self._spinner_index: int = 0
        self._spinner_timer = None
        #: The child's readings, shadowing the session's own while the
        #: full-page subagent view is up. ``None`` is the ordinary band.
        self._subagent: SubagentBand | None = None
        #: The window/tab title, driven from the band rather than from the app
        #: because the band is ALREADY the one object that knows all three
        #: inputs — the conversation name, whether a turn is streaming, and
        #: which spinner frame is current. Wiring the title from the app would
        #: mean a second subscription to the same three facts, and the failure
        #: mode of two subscriptions is the one this feature is about: a title
        #: still spinning after the band has gone idle. ``None`` until the app
        #: attaches one, which keeps the band usable by the lightweight test
        #: hosts that have no terminal to write to.
        self._title: TerminalTitle | None = None

    def set_terminal_title(self, title: TerminalTitle | None) -> None:
        """Attach (or detach) the window-title writer and paint it once.

        Called by the app once the driver exists. The immediate sync matters:
        the band is constructed and updated with the model and cwd BEFORE the
        session is awaited, so without it the title would stay blank for the
        whole of a slow boot — which is exactly when a user is most likely to
        be looking at a sidebar full of starting sessions.
        """
        self._title = title
        self._sync_terminal_title()

    def _title_state(self) -> TitleState:
        """The run state the title should show.

        Only two of the three states are derivable here: ``attention`` is the
        app's to declare (it owns the approval prompt), and it is applied by
        :meth:`set_attention` rather than inferred, because the band has no
        way to distinguish "streaming, working" from "streaming, parked on a
        question" — the session is streaming in both.
        """
        if self._attention:
            return "attention"
        return "working" if self._streaming else "idle"

    def set_attention(self, attention: bool) -> None:
        """Whether the turn is parked on something the USER owes.

        Currently the tool-approval prompt. Kept separate from ``streaming``
        because it outranks it: a turn waiting on an approval is technically
        streaming, and a session that shows a spinner while it is actually
        blocked on the user is precisely the session the user never comes back
        to. The band itself does not change — the transcript's approval card
        is unmissable once you are looking at it, and the whole point of the
        title is the case where you are not.
        """
        if attention == self._attention:
            return
        self._attention = attention
        self._sync_terminal_title()

    def _sync_terminal_title(self) -> None:
        """Push the band's current name/state/frame at the title writer.

        Cheap to call from anywhere: every setter on ``TerminalTitle``
        early-returns on an unchanged value, and the writer coalesces on the
        rendered string, so a redundant sync costs three comparisons and
        writes nothing.
        """
        title = self._title
        if title is None:
            return
        # The SUBAGENT's name wins while its page is up, matching the band
        # beneath it: everything on screen is the child's, and a title naming
        # the parent would be the one contradicting label in the frame.
        label = self._subagent.label if self._subagent is not None else self._conversation_name
        # No name in that slot falls back to the working directory. That used to
        # be every session for its first minute or two, when the only name came
        # from a detached model call made off the first substantive prompt. It
        # no longer is: `OperatorApp._show_provisional_name` paints an excerpt of
        # the opener in the SAME frame as the submit, with no provider call at
        # all. What still reaches the directory is a conversation nobody has
        # prompted yet — boot, `/new`, or a resume, since `ConversationName` is
        # never journalled — and one whose opener had no topic to quote ("hi",
        # "thanks"), which `naming.is_low_signal` refuses to label. A sidebar row
        # reading `lo ›` five times over is the exact failure this feature exists
        # to fix, and the directory is what the user opened the session for.
        title.set_label(label or cwd_label(self._cwd))
        title.set_state(self._title_state())
        title.set_frame(self._spinner_index)

    @property
    def context_tokens(self) -> int:
        """Tokens currently attributed to the context segment.

        Read by the boot-time estimator, which must not overwrite a provider's
        exact ``prompt_tokens`` if a turn beat it to the finish.
        """
        return self._context_tokens

    @property
    def context_is_estimate(self) -> bool:
        """Whether :attr:`context_tokens` is a local estimate, not the wire."""
        return self._context_is_estimate

    def set_subagent(self, band: SubagentBand | None) -> None:
        """Describe a CHILD session instead of this one, or stop (``None``).

        Idempotent and cheap enough to call from the 1 Hz refresh that keeps
        the open page live: an unchanged overlay repaints nothing.
        """
        if band == self._subagent:
            return
        self._subagent = band
        self.refresh()

    # -- segment setters ----------------------------------------------------
    def update(
        self,
        *,
        model_label: str | None = None,
        model_name: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        context_tokens: int | None = None,
        context_is_estimate: bool | None = None,
        context_window: int | None = None,
        subagents: int | None = None,
        jobs: int | None = None,
        streaming: bool | None = None,
        cost: str | None = None,
        conversation_name: str | None = None,
        mcp: McpStatus | None = None,
        approvals_auto: bool | None = None,
        approvals_always: bool | None = None,
    ) -> None:
        """Update any subset of segments and repaint the band."""
        if model_label is not None:
            self._model_label = model_label
        if model_name is not None:
            self._model_name = model_name
        if effort is not None:
            self._effort = effort
        if cwd is not None:
            self._cwd = cwd
        if context_tokens is not None:
            self._context_tokens = context_tokens
        if context_is_estimate is not None:
            self._context_is_estimate = context_is_estimate
        if context_window is not None:
            self._context_window = context_window
        if subagents is not None:
            self._subagents = subagents
        if jobs is not None:
            self._jobs = jobs
        if mcp is not None:
            self._mcp = mcp
        if approvals_auto is not None:
            self._approvals_auto = approvals_auto
        if approvals_always is not None:
            self._approvals_always = approvals_always
        if cost is not None:
            self._cost = cost
        if conversation_name is not None:
            self._conversation_name = conversation_name
        if streaming is not None and streaming != self._streaming:
            self._streaming = streaming
            self._mark_turn_boundary(streaming)
            self._sync_spinner_timer()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the band content, truncated to the dock's inner width.

        ``layout=False`` because the band's box is fixed by the sheet
        (``#status-band { height: 2 }``) and docked, so its content can never
        move anything: whatever the ladder sheds, the row count is two. The
        default reflows the WHOLE screen, and this runs on the 12.5 Hz spinner
        for the whole of every turn. A/B in one process, 161-block transcript,
        turn running, three-second idle windows: 9.6% of a core with the
        default against 7.2% with this, and 7.4% against 5.2% on a second run
        — about a quarter of the turn's idle burn, to animate one glyph.
        """
        width = max(self._dock.size.width, 10)
        self._dock.update(self._render(width), layout=False)
        # The window title rides the band's repaint rather than having its own
        # hook on each of the seven call sites that change a title input
        # (`update`, `set_subagent`, the spinner tick, …). One hook cannot be
        # forgotten by the eighth, and the two indicators are then updated by
        # construction on the same frame — the band saying "idle" while the tab
        # still spins is the specific incoherence this feature would otherwise
        # introduce. It is free when nothing changed: the writer coalesces on
        # the rendered string, so the 12.5 Hz spinner is the only thing here
        # that reaches the terminal, and only while a turn is live.
        self._sync_terminal_title()

    def is_showing(self, segment: str) -> bool:
        """Whether ``segment`` survived the drop ladder on the last render.

        Exists because "the band said it" is not the same claim as "the band was
        told it": at ordinary widths the ladder sheds segments, and a caller that
        treats the band as its only receipt has to know when the band could not
        deliver one. Reads the last render rather than re-measuring, so the
        answer is about the row the user is looking at.
        """
        return segment not in self._dropped

    def dispose(self) -> None:
        """Stop the spinner timer (idempotent).

        Deliberately does NOT stop the terminal title: the band borrows the
        writer, the app owns it. Restoring the terminal's own title is part of
        handing the terminal back, which is the app's job and has to happen
        exactly once even on the paths that never built a band (a boot that
        failed before ``on_mount`` finished).
        """
        self._stop_spinner()

    # -- duration ------------------------------------------------------------
    def _mark_turn_boundary(self, streaming: bool) -> None:
        """Start or bank the active-time clock on a streaming transition.

        Only called on an actual change (``update`` guards it), so a
        redundant ``streaming=False`` from the prompt worker's ``finally``
        cannot bank the same interval twice.
        """
        if streaming:
            self._turn_started_at = self._clock()
            # A new turn's first working frame is frame 0, by contract on both
            # the band and the title. Without resetting here the spinner resumed
            # from whichever frame the last turn happened to stop on, and the
            # title tests' "starts from the top" claim was true only for the
            # pure state holder and false for the real wiring through the band.
            self._spinner_index = 0
        elif self._turn_started_at is not None:
            self._active_seconds += self._clock() - self._turn_started_at
            self._turn_started_at = None

    def _elapsed(self) -> float:
        """Banked active time plus the turn in flight, so the segment ticks
        live off the spinner's repaint rather than needing its own timer.

        Under a subagent overlay this is the CHILD's age instead — wall clock
        since it was launched, which is the only duration a child has. The
        parent's banked active time is not a fact about the child and would be
        a bigger number than its whole life on a frame titled with its name.
        """
        if self._subagent is not None:
            return self._subagent.duration or 0.0
        if self._turn_started_at is None:
            return self._active_seconds
        return self._active_seconds + (self._clock() - self._turn_started_at)

    # -- overlay ---------------------------------------------------------------
    # Five segments answer from the overlay when one is set, and the rest are
    # left alone deliberately: cwd, MCP and the approval alarm describe the
    # HOST, which the child shares — it runs in the same directory, on the
    # parent's live MCP surface, under the same approval policy. Re-pointing
    # those would say something changed when nothing did. The counters (agents,
    # jobs) are the parent's ledger, and that ledger is what the page is a
    # window onto.
    #
    # EFFORT used to be in that host list and did not belong there: it is a
    # property of the MODEL, which is the one thing this overlay exists because
    # the child can change. The parent's level beside the child's model name was
    # not merely stale — a child on another family's model can be shown a level
    # that model's ladder does not contain.
    def _shown_effort(self) -> str:
        """The effort level for whichever model the band is describing.

        Same rule as :meth:`_shown_model_name`, for the same reason: a child
        built on the parent's spec IS running the parent's level (the spec is
        copied whole, effort included), and that is the normal path, so blanking
        it unconditionally would hide a true reading on the common case.

        A DIFFERENT model gets nothing rather than a guess. The overlay carries
        no level of its own, and the segment's whole contract is that it names
        the level in force — an empty segment says "not known here", which is
        the only honest thing the band can say about a ladder it cannot see.
        """
        if self._subagent is not None:
            if self._subagent.model_label == self._model_label:
                return self._effort
            return ""
        return self._effort

    def _shown_model_label(self) -> str:
        if self._subagent is not None:
            return self._subagent.model_label
        return self._model_label

    def _shown_model_name(self) -> str:
        """The resolved name for whichever model the segment is describing.

        Inherited under an overlay ONLY when the overlay names the same model.
        Returning nothing unconditionally looked like the safe reading of the
        overlay's rule and was not: ``job_stats`` documents an inherited spec as
        the NORMAL path and ``harness/subagent.py`` sets the child's label from a
        child built on the parent's spec, so the common case is a child on the
        parent's own model — and then the same model was spelled two ways seconds
        apart, ``MoonshotAI: Kimi K2`` with the page closed and
        ``openrouter/moonshotai/kimi-k2`` with it open, because the fallback
        lookup only reaches curated rows. A user reads that as a model switch.
        That is D10, the defect this app has already shipped once between its
        splash and its band.

        A DIFFERENT selector still gets nothing, which is the rule that actually
        matters: the parent's name must never be attributed to another model. The
        child then falls back to its curated name, exactly as any caller with no
        resolved metadata does.
        """
        if self._subagent is not None:
            if self._subagent.model_label == self._model_label:
                return self._model_name
            return ""
        return self._model_name

    def _shown_context(self) -> tuple[int, int]:
        """``(tokens, window)`` for the context segment."""
        if self._subagent is not None:
            return self._subagent.context_tokens, self._subagent.context_window
        return self._context_tokens, self._context_window

    def _shown_cost(self) -> str:
        if self._subagent is not None:
            return self._subagent.cost
        return self._cost

    def _shown_context_is_estimate(self) -> bool:
        """A child's reading is always the provider's own, never an estimate.

        Which matters beyond a tone: the drop ladder fights harder to keep an
        ESTIMATED reading on screen, and inheriting the parent's estimate flag
        would have the band making that trade on behalf of a number that has
        nothing to do with it.
        """
        return self._subagent is None and self._context_is_estimate

    # -- rendering ----------------------------------------------------------
    def _render(self, width: int) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        # Separators must sit BELOW the things they separate or they read as
        # content: `faint` is the ramp step under `dim` and exists for
        # exactly this — the dots recede and the segments group themselves.
        seam = Style(color=theme_mod.semantic_color("faint"))
        accent = Style(color=theme_mod.semantic_color("accent"))

        fitted = self._fit(width, dim, muted, seam, accent)
        if fitted is not None:
            dropped, left, right = fitted
            # Recorded so a caller can ask whether the band actually SAID
            # what it was asked to show. The effort segment needs it: it is
            # the one segment a keystroke changes, so when the ladder has
            # shed it the app owes the user a receipt somewhere else (see
            # ``OperatorApp.action_cycle_effort``). Set from the row that
            # SHIPPED — the rungs the walk tried on the way are attempts, and
            # so is a whole walk the restart threw away.
            self._dropped = dropped
            return self._compose(left, right, width, dim)

        # Even the irreducible row overflows. Truncate the model label rather
        # than shedding it (D7): `deepse…` still answers which model is
        # replying, whereas the ladder's old final rung dropped the label and
        # left a bare glyph on an empty tinted strip, which reads as broken
        # rather than as compressed.
        #
        # The SPINNER comes first and is never dropped, so "green means a turn is
        # live" holds at every width. Without it this path emitted glyph + label
        # for both states, making the streaming band byte-identical to the idle
        # one — the exact confusion the band's colour budget exists to prevent,
        # reintroduced at the narrow end. One cell of the label is a cheaper loss
        # than the liveness signal.
        tail = Text()
        if self._streaming:
            from local_operator.tui.shimmer import shimmer_enabled

            # Same gate the wide path uses: with animation off the band states
            # itself in words instead of leaning on a moving glyph.
            if shimmer_enabled():
                tail.append(_SPINNER_FRAMES[self._spinner_index], style=accent)
                tail.append(" ", style=dim)
        if self._subagent is not None and self._subagent.label:
            # The name goes on the IRREDUCIBLE row too. Without it this path
            # emitted `⣾ ◆ Gemini 2.5 Pro Preview` while the session's own
            # model was something else entirely — a child's model on screen
            # attributed to nobody, which is worse than dropping the model
            # with it. The docstring on the wide path calls this segment
            # never-dropped; below 48 columns that was a claim and not a fact.
            tail.append(f"{ICON_AGENTS} ", style=dim)
            tail.append(
                truncate_cells(self._subagent.label, max(1, width - cell_len(tail.plain))),
                style=Style(color=theme_mod.semantic_color("label")),
            )
            tail.truncate(width, overflow="ellipsis")
            return tail
        model_label = self._shown_model_label()
        label = (
            format_model_label(model_label, short=True, name=self._shown_model_name())
            if model_label
            else ""
        )
        if label:
            tail.append(f"{ICON_MODEL} ", style=dim)
            tail.append(
                truncate_cells(label, max(1, width - cell_len(tail.plain))),
                style=Style(color=theme_mod.semantic_color("fg")),
            )
        tail.truncate(width, overflow="ellipsis")
        return tail

    def _fit(
        self, width: int, dim: Style, muted: Style, seam: Style, accent: Style
    ) -> tuple[frozenset[str], Text, Text] | None:
        """The row this width gets: ``(dropped, left, right)``, or ``None``.

        ``None`` means not even the last rung fitted, and :meth:`_render` falls
        back to the irreducible tail.

        Two walks at most, and the second one is the point of this method. A walk
        is monotone — it sheds down the ladder and can never take a concession
        back — so every rung it spends BEFORE the name's own drop was spent to
        keep the name on screen, and when the name then goes those cells are
        already gone. Measured on a fully populated band at 83-94 columns (all
        four ladder variants; 79-90 with the alarm disarmed): a NAMED session lost
        `▴ high` **and** its name and gained 9 cells of hole, so naming a session
        made its band strictly worse than leaving it unnamed. So dropping the name
        restarts: the same ladder is re-walked with the name off the table
        entirely, and the wider-surviving of the two rows wins.

        The comparison is by segments SURVIVING rather than by cells painted,
        because the padded name box makes "cells" the wrong measure — a row can
        paint more cells and say less. The restart wins nearly always (a walk with
        21 fewer cells to place fits no later than one with them), and it is
        compared rather than assumed so the rule stays true if a future rung
        changes what a concession costs.
        """
        ladder = drop_ladder(self._mcp, context_estimated=self._shown_context_is_estimate())
        fitted = self._walk(ladder, width, dim, muted, seam, accent)
        if fitted is None or "name" not in fitted[0]:
            return fitted
        restart = self._walk(
            tuple(rung for rung in ladder if rung not in ("name", "flex-name")),
            width,
            dim,
            muted,
            seam,
            accent,
            forgo_name=True,
        )
        if restart is None:
            return fitted
        return restart if len(restart[0]) <= len(fitted[0]) else fitted

    def _walk(
        self,
        ladder: tuple[str, ...],
        width: int,
        dim: Style,
        muted: Style,
        seam: Style,
        accent: Style,
        *,
        forgo_name: bool = False,
    ) -> tuple[frozenset[str], Text, Text] | None:
        """One monotone pass down ``ladder``, stopping at the first row that fits.

        Building a row is a handful of string joins and the ladder is ten steps,
        so the worst case is eleven cheap rebuilds on a resize — far cheaper than
        measuring segments independently and getting the separator arithmetic
        subtly wrong.

        The name is measured at its RESERVE — the full :data:`NAME_CELLS` until
        the ``flex-name`` rung fires, its floor after — and then grown into
        whatever the fitting rung left over, capped. Three properties come out of
        that, and all three are the point:

        * The reserve is a function of the row's geometry and never of the name's
          text, so a title arriving or being rewritten cannot move a sibling.
        * A long title never sheds a neighbour that a short one would have kept,
          because the fit test measured the box either way.
        * The leftover goes to the one segment that can use it rather than into
          the hole between the groups.
        """
        dropped: set[str] = {"name"} if forgo_name else set()
        short: set[str] = set()
        flexed = False
        for step in (None, *ladder):
            if step == "flex-name":
                flexed = True
            elif step is not None:
                target = step.partition("shorten-")[2]
                (short if target else dropped).add(target or step)
            reserve = (
                0
                if "name" in dropped or not self._conversation_name
                else NAME_CELLS_FLOOR if flexed else NAME_CELLS
            )
            left = self._left_text(dropped, short, dim, muted, seam, accent)
            right = self._right_text(dropped, short, reserve, dim, seam)
            # The fit test reserves the group gap rather than asking whether the
            # composed row happens to fit. `_compose` pads with `max(1, …)`, so a
            # row could "fit" with the two groups ONE cell apart — tighter than
            # the 3-cell ` · ` separator used inside each group, which makes the
            # whole left/right architecture read as one undifferentiated run and
            # abuts a filesystem path against a percentage. Reachable by dragging
            # a window one cell at ordinary widths like 98 or 116 (D3).
            #
            # Reserved only when there IS a right group. Charged unconditionally,
            # the gap bought separation from nothing: the two-group layout was
            # abandoned four columns early and the spinner hopped across the model
            # label at a width where the row it reflowed to was no narrower.
            gap = _MIN_GROUP_GAP if right.plain else 0
            slack = width - cell_len(left.plain) - cell_len(right.plain) - gap
            if slack < 0:
                continue
            if reserve and slack and reserve < NAME_CELLS:
                right = self._right_text(
                    dropped, short, min(NAME_CELLS, reserve + slack), dim, seam
                )
            return frozenset(dropped), left, right
        return None

    def _left_text(
        self,
        dropped: set[str],
        short: set[str],
        dim: Style,
        muted: Style,
        seam: Style,
        accent: Style,
    ) -> Text:
        """model › effort › cwd › mcp (+ the working indicator).

        Each segment is ``icon value``, the icon a step dimmer than its value so
        it frames the number rather than competing with it. Separators point
        RIGHT here and LEFT in the other group, so the two chevron runs aim at
        the centre gap and frame it — which is what makes a borderless band read
        as two groups instead of one long run.

        The MCP segment is the one exception to the ramp: there the GLYPH carries
        the state colour and the text stays neutral foreground. Tinting `3 MCP`
        danger would read as "the number 3 is wrong"; tinting the glyph reads as
        a status lamp beside a plain count, which is what it is.
        """
        # (icon, value, value style, icon style) — the icon style is None for
        # every segment whose glyph is pure framing, and set only where the
        # glyph itself is the signal.
        parts: list[tuple[str, str, Style, Style | None]] = []
        model_label = self._shown_model_label()
        if model_label and "model" not in dropped:
            parts.append(
                (
                    ICON_MODEL,
                    format_model_label(
                        model_label,
                        short="model" in short,
                        name=self._shown_model_name(),
                    ),
                    Style(color=theme_mod.semantic_color("fg")),
                    None,
                )
            )
        effort = self._shown_effort()
        if effort and "effort" not in dropped:
            parts.append(
                (ICON_EFFORT, effort, Style(color=theme_mod.semantic_color("label")), None)
            )
        if self._cwd and "cwd" not in dropped:
            rendered = format_cwd(self._cwd, short="cwd" in short)
            if rendered:
                parts.append(
                    (ICON_CWD, rendered, Style(color=theme_mod.semantic_color("signal")), None)
                )
        if "mcp" not in dropped:
            mcp = format_mcp(self._mcp)
            if mcp:
                semantic = mcp_semantic(self._mcp)
                parts.append(
                    (
                        ICON_MCP,
                        mcp,
                        Style(color=theme_mod.semantic_color("fg")),
                        # BOLD when the lamp is an alarm, so weight tracks
                        # "needs attention" across the whole band rather than
                        # tracking which segment got a carrier first. The
                        # context reading is bold on its warm rungs (a
                        # colour-vision carrier), and without this the
                        # self-correcting red outweighed the one state where
                        # the agent is genuinely missing tools — worst on a
                        # narrow terminal, where the ladder drops the segments
                        # between the two and leaves them adjacent.
                        Style(
                            color=theme_mod.semantic_color(semantic),
                            bold=semantic == "danger",
                        ),
                    )
                )

        left = Text()
        for index, (icon, text, style, icon_style) in enumerate(parts):
            if index:
                left.append(f" {_SEP_LEFT} ", style=seam)
            left.append(f"{icon} ", style=icon_style or dim)
            left.append(text, style=style)
        if self._streaming:
            # The aggregate working LINE (WorkingBlock) carries the shimmer; the
            # band keeps a quiet activity glyph so a still frame still reads
            # "live". With shimmer off that line is static too, so the band
            # spells the state out rather than relying on a glyph the eye may
            # read as decoration.
            from local_operator.tui.shimmer import shimmer_enabled

            if parts:
                left.append(f" {_SEP_LEFT} ", style=seam)
            # ACCENT: the band's running indicator, and the accent budget's whole
            # point is that seeing green means a turn is live. The trailing
            # " working" word stays dim so the colour is the signal and the word
            # is only the caption.
            left.append(_SPINNER_FRAMES[self._spinner_index], style=accent)
            if not shimmer_enabled():
                left.append(" working", style=dim)
        return left

    def _right_text(
        self, dropped: set[str], short: set[str], name_cells: int, dim: Style, seam: Style
    ) -> Text:
        """agents ‹ jobs ‹ context ‹ cost ‹ duration ‹ [!] ‹ name.

        ``name_cells`` is the box :meth:`_walk` reserved for the trailing name —
        zero when the ladder dropped it. The name is PADDED to that box rather
        than measured, which is what pins every other segment's column; see
        :data:`NAME_CELLS`.

        Colour groups by KIND rather than giving every field its own hue, which
        would be a rainbow: counters share `label`, the two numbers an operator
        acts on take `signal` (context) and `warning` (cost, because spend is a
        caution), and the least volatile fields stay neutral. Green is not used
        at all — it belongs to the running indicator.
        """
        parts: list[tuple[str, str, Style]] = []
        if self._subagent is not None and self._subagent.label:
            # Never dropped: it is what says whose numbers follow it, and a
            # band that sheds the owner while keeping the figures is worse
            # than one that sheds a figure.
            parts.append(
                (
                    ICON_AGENTS,
                    truncate_cells(self._subagent.label, 24),
                    Style(color=theme_mod.semantic_color("label")),
                )
            )
        elif "subagents" not in dropped:
            agents = format_agents(self._subagents)
            if agents:
                parts.append((ICON_AGENTS, agents, Style(color=theme_mod.semantic_color("label"))))
        if "jobs" not in dropped:
            jobs = format_jobs(self._jobs)
            if jobs:
                parts.append((ICON_JOBS, jobs, Style(color=theme_mod.semantic_color("label"))))
        if "context" not in dropped:
            tokens, window = self._shown_context()
            usage = format_context_usage(tokens, window)
            if usage:
                # The one segment whose colour carries information: it warms
                # from blue through purple to red as the context fills, so a
                # session heading for compaction is visible at a glance
                # instead of having to be read. The window is passed because
                # the ramp is a union of an absolute and a proportional
                # ladder; on a 200k model the absolute rungs are unreachable
                # and the proportional half is the only thing that fires.
                #
                # BOLD on the warm rungs, because hue alone cannot carry this.
                # `signal` and `label` are 35 dE apart in normal vision and
                # 1.7 under deuteranopia (the commonest deficiency, ~6% of
                # men) — indistinguishable, so for those readers the 200k step
                # would simply not exist. Weight is orthogonal to hue, costs no
                # cells, and moves nothing in the drop ladder's arithmetic
                # since the text is unchanged. The base rung stays regular so
                # "warm" remains the marked state rather than the default.
                semantic = context_semantic_color(tokens, window)
                parts.append(
                    (
                        ICON_CONTEXT,
                        usage,
                        Style(
                            color=theme_mod.semantic_color(semantic),
                            bold=semantic != CONTEXT_COLOR_BASE,
                        ),
                    )
                )
        cost = self._shown_cost()
        if cost and "cost" not in dropped:
            parts.append((ICON_COST, cost, Style(color=theme_mod.semantic_color("warning"))))
        if "duration" not in dropped:
            elapsed = self._elapsed()
            # Zero means "nothing has run yet" — an idle band should not claim a
            # 0s task. Any real turn banks at least a few ms.
            if elapsed > 0:
                parts.append((ICON_DURATION, format_duration(elapsed), dim))
        if self._approvals_auto and "approvals" not in dropped:
            # A BARE GLYPH, and second-to-last rather than last. It used to spell
            # out `! auto-approve always` and own the band's right edge, on the
            # argument that an alarm which slides left as siblings appear is an
            # alarm the eye has to hunt for. That argument was sound and the slot
            # is now spoken for: the session NAME is what an operator reads this
            # corner for, and 20 cells of prose about a mode that has not changed
            # since boot was crowding out the one field that says which
            # conversation this is. The argument is answered rather than ignored —
            # the name's box is RESERVED (:data:`NAME_CELLS`), so this glyph has
            # one column per terminal width and a title landing under it cannot
            # move it.
            #
            # What survives is the alarm itself, because nothing else in the UI
            # carries it for longer than a scroll: `/approvals` answers on
            # demand, the notice that latched the mode scrolls away (and `/clear`
            # deletes it), and a saved `tool_approval_mode: auto` is adopted at
            # boot with NO notice at all — see ``OperatorApp._adopt_saved_approvals``.
            # So `!` alone stays, bold, and it still means exactly what it has
            # always meant here: no tool will ask before it runs. It no longer
            # distinguishes `auto` from `always`; that was the answer to "until
            # when", which is a question worth a command rather than worth the
            # band's last segment.
            #
            # `danger`, not `warning`, and that is the D4 fix rather than a
            # preference: at one cell the ink is ALL the segment has, and in
            # `warning` it was the identical hue to `◈ $73.92` three cells away
            # (both `#e0b04b`, both 8.64:1 on the band's ground), so the band's
            # only alarm read as another figure. `danger` reads as an alarm and
            # measures 6.62:1 dark / 4.94:1 on the paper ramp, and it is the same
            # ink the `⊙` lamp takes when MCP discovery fails.
            #
            # Red in this band means NEEDS-ATTENTION, and three segments may take
            # it: this one, the `⊙` MCP lamp, and the context reading's top rung.
            # They do not conflict because each is the warmest state of a
            # DIFFERENT segment, read off the glyph that precedes it, and none is
            # a routine value — `!` means no tool will ask before it runs, `⊙`
            # means configured tools are missing, and `▦` in red means the
            # context is at or past its compaction trigger OR past the absolute
            # re-send-cost rung. Those two coincide only where the window is
            # small enough for the proportional ladder to govern: above ~625k
            # the absolute rung fires at 500k while the resolved trigger is
            # capped at 600k, so a 1M session shows red for a band of ~100k
            # before a pass is actually due. It is the weakest of the three
            # either way (compaction resolves it without the user acting), which
            # is why it is the only one gated behind a threshold rather than
            # being a state the segment can simply be in.
            #
            # BOLD on all three, so weight tracks "needs attention" rather than
            # tracking which segment happened to get a carrier first. The
            # context reading is bold on its warm rungs as a colour-vision
            # carrier (see the ramp), and leaving the `⊙` lamp regular made the
            # self-correcting red heavier than the one state where the agent is
            # genuinely missing tools — most visible on a narrow terminal, where
            # the ladder sheds the segments between them.
            #
            # The glyph rides INSIDE the styled text rather than in the icon slot,
            # because the loop below paints icons `dim` — which made the one alarm
            # in the band its quietest mark (4.18:1, against the same `!` at
            # 9.4:1 in the transcript).
            parts.append(
                (
                    "",
                    ICON_APPROVALS,
                    Style(color=theme_mod.semantic_color("danger"), bold=True),
                )
            )
        if self._conversation_name and name_cells:
            # LAST, so the corner is the name's — the position the approval mode
            # used to hold, and the one the user asked for it in.
            #
            # `muted` and not the accent: the accent budget is spent on "what the
            # turn is on" (``local_operator.tcss`` enumerates every site), and a
            # session's name is metadata, so it takes the same secondary ink as
            # the duration beside it.
            #
            # Truncated, never wrapped: the name is model-generated or an excerpt
            # of the user's opener, so it has no natural bound, and the band is a
            # fixed two-row box.
            #
            # PADDED to its box, and RIGHT-aligned in it. The box is the
            # reserved width, so the padding is what holds every sibling's
            # column still while a model rewrites the string in it — that half
            # is the reserve's whole purpose and is unchanged.
            #
            # The alignment inside the box decides where the box's UNUSED cells
            # sit, and they have to sit somewhere: the fit was decided against
            # the full box, so those cells are already spoken for and cannot be
            # given back to the layout without moving every column left of the
            # name. `rjust` puts them before the title, which makes the title's
            # last cell the band's right edge at every length.
            #
            # `ljust` put them after it instead, and `_compose` then stripped
            # them — so the painted row simply ended early: flush for a long
            # title and up to 35 cells short for a brief one, which reads as a
            # rendering fault rather than as a band with less to say. (They are
            # ordinary styled cells, so anything READING the row carried them
            # too: the SVG export wrote all 35.)
            #
            # The cost is that the title's FIRST cell moves with its length.
            # That is the right way round — the reserve exists to stop the
            # SIBLINGS drifting (the `!` alarm holds its column at every width),
            # and the alternative, measuring the group by its ink, moves the
            # alarm itself, which is the defect the box was introduced to fix.
            parts.append(
                (
                    "",
                    truncate_name(self._conversation_name, name_cells).rjust(name_cells),
                    Style(color=theme_mod.semantic_color("muted")),
                )
            )

        right = Text()
        for index, (icon, text, style) in enumerate(parts):
            if index:
                right.append(f" {_SEP_RIGHT} ", style=seam)
            if icon:
                right.append(f"{icon} ", style=dim)
            right.append(text, style=style)
        return right

    def _compose(self, left: Text, right: Text, width: int, dim: Style) -> Text:
        """Left group, filler, right group — right-aligned to the band edge.

        With no right group there is nothing to align and nothing to separate, so
        the filler is not emitted at all. Padding unconditionally pushed a row
        that exactly fitted its frame `_MIN_GROUP_GAP` cells past it, on trailing
        blanks that aligned nothing.

        The right group is aligned by its BOX, not by its ink: the name arrives
        padded to the cells :meth:`_walk` reserved (:data:`NAME_CELLS`), so this
        arithmetic is the same at every title length and a short title leaves its
        leftover at the band's edge rather than shunting the group right.

        The box's unused cells are spent INSIDE the name segment, which
        right-aligns itself in them (see :meth:`_right_text`), so the row's last
        inked cell is the band's right edge at every title length and there is
        nothing here to trim.

        This method used to `rstrip` a LEFT-aligned name's trailing padding,
        which produced a row that ended early — flush for a long title and up to
        35 cells short for a brief one. Moving the padding to the other side of
        the title fixes that at the source: the cells still exist, so the fit
        arithmetic and every column left of the name are untouched, but they are
        no longer at the edge where their absence was visible.

        Aligning the group by its INK rather than its box is the other way to
        reach the edge, and the one this must not do: measured across four
        titles it walks the `!` alarm from column 74 to 91/111/151 at 100/120/160
        cells, which is exactly the drift the reserve was introduced to stop.
        """
        if not right.plain:
            return left
        gap = max(_MIN_GROUP_GAP, width - cell_len(left.plain) - cell_len(right.plain))
        row = Text()
        row.append_text(left)
        row.append(" " * gap, style=dim)
        row.append_text(right)
        return row

    def render_text(self, width: int) -> Text:
        """Public render entry (tests): segments joined, fitted to width."""
        return self._render(width)

    # -- spinner ------------------------------------------------------------
    def _sync_spinner_timer(self) -> None:
        if self._streaming and self._spinner_timer is None:
            self._spinner_timer = self._dock.set_interval(
                _SPINNER_INTERVAL_S, self._advance_spinner
            )
        elif not self._streaming and self._spinner_timer is not None:
            self._stop_spinner()

    def _advance_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        self.refresh()

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

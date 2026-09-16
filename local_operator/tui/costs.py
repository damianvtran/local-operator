"""Turning token usage into dollars, for every surface that shows money.

One computation, several readers. The status band, the subagent panel's rows and
the full-page subagent view all price spend, and before this module each of them
would have had to resolve a model and multiply for itself — which is how two
places on the same screen end up disagreeing about what a turn cost.

The arithmetic itself is NOT here: it lives in
:func:`local_operator.model.configure.cost_for_usage`, next to the pricing table
and the provider cache-token conventions it depends on. What this module adds is
the adaptation the TUI needs on top of it:

- a model LABEL (``provider/model_id``) rather than a resolved ``ModelInfo``,
  because that is what a session and a job carry;
- ``None`` for "this model has no published price" as distinct from ``0.0`` for
  "this cost nothing", which is the distinction the band's ``$—`` exists to make;
- an :class:`~local_operator.harness.jobs.AsyncJob` as an input, so a subagent's
  spend can be read straight off the ledger the panel already renders from.

Per-child only, deliberately: the parent's aggregate is NOT a sum over the live
ledger. Settled jobs are swept out of it after a retention window, so the app
keeps its own dict of last-observed figures and sums that instead — a spend
counter that falls when a finished child is evicted is worse than none.

One kind of money here is NOT token usage and is not priced by this module:
search spend. ``web_search`` bills per query or per provider-reported turn and
records it in its own process-wide ledger
(:data:`~local_operator.web_search.cost.SEARCH_SPEND`), keyed by session. What
this module adds for it is the same adaptation -- a frozen
:class:`SearchSpendSnapshot`/``SearchSpendRow`` pair that carries the
``_CostLike`` members the panels' one money formatter reads, so search dollars
and model dollars are spelled the same way (``$—`` unknown, ``+`` lower bound)
without a second formatter or a ledger import in the panels.

Nothing here raises. A price is never worth a broken frame.

The three pure pricing functions this module was built around — ``turn_cost``,
``cost_summary`` and ``job_cost`` — now live in ``local_operator.model.costs``,
next to the arithmetic they adapt, because the SESSION layer prices turns and
jobs too and must not have to import the TUI package to do it (see that
module's docstring). They are re-exported below, so every caller here is
unchanged. The formatter ladder (``format_usd`` and friends) stays here: it is
the TUI's own vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# The three pure pricing functions now live in `local_operator.model.costs`,
# next to the arithmetic they adapt — see that module's docstring for why the
# session layer must not have to import the TUI package to price a turn. They
# are re-exported here so this module's surface (and every caller's import)
# is unchanged.
from local_operator.model.costs import cost_summary, job_cost, turn_cost

if TYPE_CHECKING:  # pragma: no cover - typing only, see ``SearchSpendSnapshot.of``
    from local_operator.web_search.cost import SearchSpendTotals

__all__ = [
    "SearchSpendRow",
    "SearchSpendSnapshot",
    "format_usd",
    "format_usd_exact",
    "micro_from_usd",
    "turn_cost",
    "job_cost",
    "cost_summary",
]


#: Below this many micro-USD a 4dp figure round-trips to ``$0.0000``, which
#: reads as FREE — the more expensive of the two lies ``turn_cost`` warns about.
#: 50 µ$ is $0.00005, the rounding boundary of ``{"$%.4f"}``.
_SUB_CENT_VISIBLE_MICRO = 50

#: THE spelling for "money we cannot state", shared by the surfaces that can
#: show it. It lives here beside ``LOWER_BOUND_MARK`` for the same reason: two
#: literals for one honesty vocabulary is how the band and ``/session`` came to
#: disagree about the same state (QA round 1 Q1, round 2 Q1/Q3).
UNKNOWN_COST_CELL = "$—"

#: THE lower-bound mark, shared by every surface that has to say "this figure
#: cannot be whole". It lives here rather than on the band because the band and
#: ``/session`` both draw it, and two spellings of the same honesty vocabulary is
#: exactly the defect the panels' single-formatter rule exists to prevent (review
#: R2-1: ``/session`` printed an unmarked figure for a state the band marked).
LOWER_BOUND_MARK = "\u2265"


def format_usd(micro: int) -> str:
    """The ONE dollar ladder, from an EXACT integer micro-USD amount.

    Takes micro-USD rather than a float because the ladder's edge cases are
    decided on the true value: a float that arrived already rounded to ``0.0``
    cannot be told from a real zero, and the one reading the ladder must never
    print is ``$0.0000`` for money that was spent. INTEGER in, so the decision
    is about the money rather than about a rounded copy of it.

    Ladder (rounding, never truncation — an f-string rounds):

    - ``$1.90`` above a dollar, ``$0.213`` above a cent, ``$0.0042`` below it;
    - a NONZERO amount under half a ten-thousandth of a dollar renders
      ``<$0.0001`` — the one spelling where the ladder would otherwise lie, and
      8 cells wide, or 9 when marked (``≥<$0.0001``). The mark and the spelling
      DO co-occur: an unpriced call makes the total a lower bound regardless of
      the digits, and the digits round to zero regardless of the mark (review
      R1-3 corrected the design's earlier "cannot co-occur" claim, which would
      have budgeted one cell short);
    - zero renders ``$0.0000``, unchanged: for a genuine zero that spelling is
      correct, and the band's zero policy already drops the segment entirely.

    Every money surface reads THIS function for its digits; the callers keep
    their own mark handling (``≥`` on the band, a dim ``+`` in the panels). Two
    ladders is how ``/analytics`` came to print ``$1.2k`` beside the band's
    ``$1234.56`` for the same money.
    """
    if micro < _SUB_CENT_VISIBLE_MICRO:
        return "<$0.0001" if micro > 0 else "$0.0000"
    cost = micro / 1_000_000.0
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 1.0:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


def micro_from_usd(cost: Any) -> int | None:
    """The exact integer micro-USD of a FLOAT dollar figure, or ``None``.

    ``None`` means "not a figure this ladder can take": non-finite (a NaN
    restored from a corrupt checkpoint, an inf from a division the accounting
    did not guard) or not a number at all (a ``None`` from a reduced host).
    ``int(round(nan))`` RAISES, so the two display wrappers that still hold a
    float — the band's and ``/analytics``' — used to be able to take a frame
    down over a bad figure, against this module's own contract that nothing here
    raises (review R1-7). The caller decides what to print for a value the
    ladder cannot take; this function only refuses to lie about one.
    """
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost):
        return None
    return int(round(cost * 1_000_000))


def format_usd_exact(micro: int) -> str:
    """The exact micro-USD at full precision: ``$1.897843``.

    What ``/session`` shows on demand, so the reader can see that the cents on
    the band are not the whole number. Six decimal places is the record's own
    resolution (micro-USD), trailing zeros trimmed to at least the cents, so an
    exact ``$2.10`` still reads as ``$2.10`` rather than ``$2.1``.
    """
    if micro < 0:
        return f"-{format_usd_exact(-micro)}"
    whole, _, fraction = f"{micro / 1_000_000:.6f}".partition(".")
    fraction = fraction.rstrip("0")
    if len(fraction) < 2:
        fraction = (fraction + "00")[:2]
    return f"${whole}.{fraction}"


@dataclass(frozen=True)
class SearchSpendRow:
    """One provider's search spend, frozen.

    A copy rather than the ledger's own ``ProviderSearchSpend``, and frozen for
    the reason the whole snapshot exists: see :class:`SearchSpendSnapshot`.

    Carries the three members ``analytics_panel._CostLike`` names --
    ``cost_usd``/``cost_is_known``/``cost_is_partial`` -- so the panels render
    this money through the ONE formatter (``format_cost``) that already
    distinguishes an unknown price from a free one and marks a lower bound. A
    second money formatter for search spend would be a second honesty
    vocabulary on the same screen.
    """

    provider: str
    searches: int = 0
    usd: float = 0.0
    #: ``search`` or ``read``. ``web_read`` records its money under
    #: ``<provider>:read`` so the totals stay clean, and this is what lets the
    #: row SAY read: a row whose whole purpose is separating reads from searches
    #: must not be labelled as one of them.
    kind: str = "search"
    reads: int = 0
    #: Unpriced reads, so a note can attach the tally to the kind it belongs to.
    unpriced_reads: int = 0
    #: Searches whose provider publishes no rate. Counted, never rendered as
    #: $0: unknown and free are different facts (see ``format_cost``).
    unpriced_searches: int = 0
    #: Operations served free and paid for, counted exactly by the ledger (see
    #: :class:`~local_operator.web_search.cost.ProviderSearchSpend`), and the
    #: money each half came to. The pair the dollars alone cannot carry: six free
    #: searches and no searches at all both total $0.0000.
    free_operations: int = 0
    paid_operations: int = 0
    free_usd: float = 0.0
    paid_usd: float = 0.0

    @property
    def priced_searches(self) -> int:
        """Everything with a known price: searches AND reads.

        The name follows the common case, but a READ has a price too. Counting
        only ``searches`` here made a read-only row report ``cost_is_known``
        False, so a read that cost $0.002 rendered ``$—``: the ledger knew the
        money and the panel claimed it did not.
        """
        return self.count - self.unpriced_searches

    @property
    def cost_usd(self) -> float:
        return self.usd

    @property
    def cost_is_known(self) -> bool:
        # A provider with nothing priceable is ``$—``, not ``$0.0000``, even
        # when it served searches: the money is unknown, not measured at zero.
        return self.priced_searches > 0

    @property
    def cost_is_partial(self) -> bool:
        # A lower bound: some of this provider's searches are unpriced.
        return self.unpriced_searches > 0

    @property
    def count(self) -> int:
        """Everything this row covers: searches plus reads."""
        return self.searches + self.reads


@dataclass(frozen=True)
class SearchSpendSnapshot:
    """A search-spend total frozen for display, session or process-wide.

    The panels take THIS rather than the live ``SearchSpendTotals`` for two
    reasons, one of them a crash: ``SEARCH_SPEND.session()`` hands back the
    ledger's OWN mutable total, and a search finishing while a panel iterates
    ``by_provider`` mutates that dict mid-render (``RuntimeError: dictionary
    changed size during iteration``). The other is the promise ``/session``
    already makes -- a snapshot, not a live bill -- which a panel holding the
    ledger object would quietly break.

    ``rows`` is sorted biggest-first (spend, then search count) because that is
    the order every other per-entity table on these two screens reads in, and
    an unpriced provider sorts last on the strength of its zero spend: its
    ``$—`` is the honest cell, not a big number.

    Structurally satisfies ``_CostLike``, so ``format_cost``/``append_cost``
    price it without a ``web_search`` import reaching the formatter.

    NOT the place for the ledger's own session keys: this is one scope, already
    resolved by the caller, and the process-wide view is merged by the ledger
    (``overall()``).
    """

    searches: int = 0
    usd: float = 0.0
    unpriced_searches: int = 0
    #: Page reads, counted apart from searches (see ``SearchSpendRow.kind``).
    reads: int = 0
    #: Unpriced reads (see ``SearchSpendRow.unpriced_reads``).
    unpriced_reads: int = 0
    #: Free and paid operations across every provider (see
    #: ``SearchSpendRow.free_operations``).
    free_operations: int = 0
    paid_operations: int = 0
    free_usd: float = 0.0
    paid_usd: float = 0.0
    rows: tuple[SearchSpendRow, ...] = ()

    @classmethod
    def of(cls, totals: "SearchSpendTotals | None") -> "SearchSpendSnapshot":
        """Freeze one ``SearchSpendTotals`` (or ``None``) into a snapshot.

        Duck-typed on the four ledger members rather than annotated with the
        ledger's own type at runtime: ``tui.costs`` is imported by
        ``session.frontend_state`` and ``harness.subagent``, and pulling the
        web-search package into those import graphs for a display copy is not
        worth it. ``None`` freezes to the empty snapshot, which is what a
        session with no ledger row and a host with no ledger both mean.
        """
        if totals is None:
            return cls()
        by_provider = getattr(totals, "by_provider", None) or {}
        rows = tuple(
            SearchSpendRow(
                provider=str(getattr(entry, "provider", key) or key),
                searches=int(getattr(entry, "searches", 0) or 0),
                kind=str(getattr(entry, "kind", "search") or "search"),
                reads=int(getattr(entry, "reads", 0) or 0),
                unpriced_reads=int(getattr(entry, "unpriced_reads", 0) or 0),
                free_operations=int(getattr(entry, "free_operations", 0) or 0),
                paid_operations=int(getattr(entry, "paid_operations", 0) or 0),
                free_usd=float(getattr(entry, "free_usd", 0.0) or 0.0),
                paid_usd=float(getattr(entry, "paid_usd", 0.0) or 0.0),
                usd=float(getattr(entry, "usd", 0.0) or 0.0),
                unpriced_searches=int(getattr(entry, "unpriced_searches", 0) or 0),
            )
            for key, entry in by_provider.items()
        )
        return cls(
            searches=int(getattr(totals, "searches", 0) or 0),
            usd=float(getattr(totals, "usd", 0.0) or 0.0),
            unpriced_searches=int(getattr(totals, "unpriced_searches", 0) or 0),
            reads=int(getattr(totals, "reads", 0) or 0),
            unpriced_reads=int(getattr(totals, "unpriced_reads", 0) or 0),
            free_operations=int(getattr(totals, "free_operations", 0) or 0),
            paid_operations=int(getattr(totals, "paid_operations", 0) or 0),
            free_usd=float(getattr(totals, "free_usd", 0.0) or 0.0),
            paid_usd=float(getattr(totals, "paid_usd", 0.0) or 0.0),
            rows=tuple(sorted(rows, key=lambda row: (-row.usd, -row.count, row.provider))),
        )

    @property
    def count(self) -> int:
        """Searches plus reads: everything that could carry a price."""
        return self.searches + self.reads

    @property
    def priced_searches(self) -> int:
        """Known-price operations, reads included.

        The name follows the common case, but a READ has a price too: counting
        only ``searches`` made a read-only row or session report
        ``cost_is_known`` False, so a read that cost $0.002 rendered ``$—`` --
        the ledger knew the money and the panel claimed it did not.
        """
        return self.count - self.unpriced_searches

    @property
    def cost_usd(self) -> float:
        return self.usd

    @property
    def cost_is_known(self) -> bool:
        return self.priced_searches > 0

    @property
    def cost_is_partial(self) -> bool:
        return self.unpriced_searches > 0


@dataclass(frozen=True)
class SpendSummary:
    """One screen's money: the model half, the search half, and their total.

    THE place the two kinds of spend are combined, because they reached three
    surfaces (the status band, ``/session``, ``/analytics``) and each one had to
    answer the same three questions: what is the total, is any of it unknown
    rather than zero, and is the total a LOWER BOUND. Written out per surface,
    those answers drifted -- the band folded search in while both panels'
    headline said model-only, so one session showed two different costs, and
    the version that omitted retrieval was the one a person reads first.

    Deliberately not a money FORMATTER: the values here go to ``format_cost``,
    which owns the honesty vocabulary (``$—`` unknown, ``+`` lower bound).
    """

    model_usd: float | None
    search_usd: float
    #: True when NOTHING could be priced (an unpriced model with no search
    #: spend), so the figure must render ``$—`` rather than a confident zero.
    is_unknown: bool
    #: True when the figure covers only part of the work: an unpriceable model
    #: beside real search spend, or any unpriced search.
    is_floor: bool

    @property
    def total_usd(self) -> float:
        return (self.model_usd or 0.0) + self.search_usd


@dataclass(frozen=True)
class MoneyFigure:
    """One money figure with its honesty flags, for ``format_cost``.

    ``format_cost`` takes a SCOPE rather than a float because the marks it
    renders (``+`` lower bound, ``$—`` unknown) are properties of the scope's
    knowledge. A figure this module has already combined and classified still has
    to be spelled as a scope to go through the same formatter -- the alternative
    is a second money formatter on the same screen, which is the defect the
    panels' single-formatter rule exists to prevent.
    """

    cost_usd: float
    cost_is_known: bool = True
    cost_is_partial: bool = False

    @classmethod
    def of(cls, summary: "SpendSummary") -> "MoneyFigure":
        return cls(
            cost_usd=summary.total_usd,
            cost_is_known=not summary.is_unknown,
            cost_is_partial=summary.is_floor,
        )


def combined_spend(
    model_usd: float | None,
    search: "SearchSpendSnapshot | None",
    *,
    model_is_partial: bool = False,
) -> SpendSummary:
    """Combine model cost with search spend for one screen.

    ``model_usd`` is ``None`` when the model half is unpriceable; the search half
    is never ``None`` (an absent snapshot is an empty one), because a retrieval
    ledger that is missing means no searches were recorded HERE, which is a
    zero and not an unknown.

    ``model_is_partial`` carries the model figure's OWN floor (some calls priced,
    some not) into the combined one: the marks are per-figure, and dropping it
    here printed a bare total for a tree whose child had used an unpriced model
    -- the very mark the panel had, before this function existed, gone at the
    moment the two halves were added together.
    """
    if search is None:
        search = SearchSpendSnapshot()
    search_usd = float(search.usd or 0.0)
    if model_usd is None and search_usd <= 0.0:
        # Neither half is priceable: the honest figure is "unknown", and it is
        # not a floor because there is no known part for it to be a floor OF.
        return SpendSummary(None, search_usd, True, False)
    if model_usd is None:
        # Real search money beside an unpriceable model: a floor, and the one
        # case where a bare figure would understate the session by the whole
        # model half.
        return SpendSummary(None, search_usd, False, True)
    return SpendSummary(
        model_usd, search_usd, False, bool(model_is_partial or search.cost_is_partial)
    )


def search_spend_is_floor(search: "SearchSpendSnapshot | None") -> bool:
    """Whether a search figure is a lower bound (some of it unpriced).

    Asked by the status band, which shows the combined figure and needs only this
    flag. Deliberately NOT ``combined_spend(0.0, ...).is_floor``: that call hands
    the combiner a model half the band does not have, so the combiner's
    model-unknown branch was unreachable at that call site while the code read as
    though it handled the case (round-1 review MINOR-2).
    """
    return bool(search is not None and search.cost_is_partial)


def cost_label(has_search: bool) -> str:
    """The ``Est. cost`` row's label, carrying the search scope when there is one.

    Design review D1 (blocker): the search half was named only in the NOTE, and
    the note is cropped above ``_NOTE_MIN`` and shed below it -- so at 80 columns
    the row painted ``≈ list price × tokens · incl`` and at 60 it painted nothing,
    leaving a combined figure that looked model-only. The panel's own rule is that
    a distinction the reader must not lose belongs in the LABEL, which is never
    shed; the notes carry only the refinement. The label column is 22 cells and
    this is 18.
    """
    return "Est. cost · search" if has_search else "Est. cost"


def cost_note_rungs(spend: SpendSummary, *, search_component: str = "") -> tuple[str, ...]:
    """The note ladder for a combined ``Est. cost`` figure, widest first.

    Shared by ``/session`` and ``/analytics`` for the reason this module exists:
    one figure, one vocabulary. Two situations the ladder has to spell:

    * a model half that could NOT be priced beside real search money -- the
      figure is entirely retrieval, so a token-priced note would describe half a
      figure that is not in it (round-1 review MINOR-3);
    * a figure that includes search, where the note must keep naming the search
      component at widths that crop. A single wide string rendered
      ``≈ list price × tokens · incl`` at 80 columns, destroying the one fact the
      row was added to state (round-1 review MAJOR-2).
    """
    if spend.is_unknown:
        return ("no published price",)
    if spend.model_usd is None:
        # D2 (major): with no model price there is nothing the tokens multiplied,
        # and nothing for ``incl.`` to fold the search money INTO -- the search
        # half IS the figure. ``incl.`` here was a claim about a half that is not
        # in it, so the wording says what the figure is instead.
        return (
            "search only · model unpriced",
            "search only",
        )
    if search_component:
        return (f"≈ list price × tokens · {search_component}", search_component)
    return ("≈ list price × tokens",)

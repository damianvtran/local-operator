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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, see ``SearchSpendSnapshot.of``
    from local_operator.web_search.cost import SearchSpendTotals

__all__ = [
    "SearchSpendRow",
    "SearchSpendSnapshot",
    "turn_cost",
    "job_cost",
    "cost_summary",
]


def _resolve_for_paint(provider: str, model_id: str):
    """The paint-safe resolver, with a background refresh fired on a cold miss.

    One seam so ``turn_cost`` and the per-component loop share the exact
    policy: resolve from the warm memo or registry ONLY (never discovery,
    which is synchronous HTTP), and when the memo has no entry for this
    model, hand the full resolution to a background thread so the NEXT tick
    paints the real number. The refresh is what keeps a memo rollover from
    silently switching the band to the (possibly stale) registry row; the
    paint-only resolution is what keeps the keyboard live while it happens.
    """
    from local_operator.model.configure import (
        refresh_model_info_background,
        resolve_model_info_paint,
    )

    info, memo_hit = resolve_model_info_paint(provider, model_id)
    if not memo_hit:
        # Missed the paint memo: either a model this process never resolved
        # (the registry row may carry no price — exactly the population the
        # discovery legs exist for) or the TTL bucket rolled over mid-session
        # (the registry row carries a price that may be staler than the
        # discovery answer the band showed until that moment). Both want the
        # same thing: the real answer, fetched OFF the loop, landing next
        # tick. The band still shows this tick's honest answer — None when
        # unpriceable, the registry price otherwise — rather than blocking.
        refresh_model_info_background(provider, model_id)
    return info


def turn_cost(model_label: str, usage: Any) -> float | None:
    """What ``usage`` cost on the model named by ``model_label``, or ``None``.

    ``model_label`` is the ``provider/model_id`` spelling that
    :attr:`Session.model_label` produces.

    ``None`` means the price is genuinely unknown — no registry row, no provider
    listing and no aggregator entry could put a number on this model. It is NOT
    the same as ``0.0``, and a caller must not collapse the two: a confident
    ``$0.0000`` on a turn that billed tokens reads as "that was free", which is
    the more expensive lie of the two.

    Resolution for the PAINT path is memo-or-registry only
    (:func:`~local_operator.model.configure.resolve_model_info_paint`): the
    full resolver's discovery legs are synchronous HTTP (measured 418 ms
    warm-disk, 10 s + 3 s worst case for an unlisted model), and this
    function runs on the Textual loop at ``message_end" and on the 1 Hz
    subagent harvest — a blocking miss there is the frozen-keyboard
    regression, not a slow number. A cold miss fires one background refresh
    per model so the following tick prices from the warm memo; ``None" for
    one tick is the same honest degradation the band already renders.
    """
    if usage is None or not model_label:
        return None
    try:
        # A provider-reported dollar amount is authoritative without a table:
        # OpenRouter (and any aggregator that precomputes billing) returns the
        # exact charge it printed, per-routed-provider pricing and reasoning
        # splits included. It must win even when the model has no published price
        # row, because the provider's bill is the fact the table is an estimate of.
        #
        # Coerced and floored through the SAME helper the pricing path uses on
        # the wire values rather than a bare ``float()`` here: a negative or
        # non-numeric amount is malformed provider data and must fall back to the
        # estimate, not render an upside-down credit or degrade the whole turn to
        # unpriceable while a table price exists. (The wire client already drops
        # these to ``None``, but ``turn_cost`` also serves rehydrated mappings.)
        from local_operator.model.configure import cost_for_usage

        provider, _, model_id = model_label.partition("/")
        components = getattr(usage, "cost_components", None)
        if components:
            # A mixed aggregate cannot put a partial receipt in ``usd_cost``:
            # that would make the pricing helper skip estimates for every other
            # call. Price each original call on its serving identity instead.
            total = 0.0
            for component in components:
                component_provider = getattr(component, "provider", None) or provider
                component_model = getattr(component, "model_id", None) or model_id
                reported = _recorded_cost(component)
                if reported is not None:
                    total += reported
                    continue
                info = _resolve_for_paint(component_provider, component_model)
                if not (info.input_price or info.output_price):
                    return None
                total += cost_for_usage(component_provider, info, component)
            return total

        reported = _recorded_cost(usage)
        if reported is not None:
            return reported

        info = _resolve_for_paint(provider, model_id)
        if not (info.input_price or info.output_price):
            return None
        return cost_for_usage(provider, info, usage)
    except Exception:  # noqa: BLE001 — an unpriceable model is not a render error
        return None


def _recorded_cost(usage: Any) -> float | None:
    from local_operator.model.configure import _usage_cost

    receipt = _usage_cost(usage)
    if receipt is not None:
        return receipt
    return _usage_cost({"usd_cost": getattr(usage, "estimated_usd_cost", None)})


def cost_summary(
    components: Any, *, model_label: str = "", recorded_only: bool = False
) -> tuple[float | None, bool]:
    """Known spend and whether any component is unknown; never lose a lower bound.

    Components, not aggregate tokens, own price provenance. A failed or offline
    lookup must not erase already-priced siblings, and an empty ledger must not
    masquerade as a provider-reported zero.
    """
    total: float | None = None
    unknown = False
    for component in components:
        provider = getattr(component, "provider", None)
        model_id = getattr(component, "model_id", None)
        label = f"{provider}/{model_id}" if provider and model_id else model_label
        cost = _recorded_cost(component) if recorded_only else turn_cost(label, component)
        if cost is None:
            unknown = True
        else:
            total = (total or 0.0) + cost
    return total, unknown


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


def job_cost(job: Any, *, default_model_label: str | None = None) -> float | None:
    """What one subagent job has spent so far, or ``None`` when unpriceable.

    ``job`` is duck-typed: anything carrying ``usage`` and ``model_label``, which
    in production is an :class:`~local_operator.harness.jobs.AsyncJob`. A job with
    no recorded usage — a ``bash`` job, or a child that has not reported a turn
    yet — returns ``None`` rather than ``0.0``, because "spent nothing" and "has
    not told us yet" are different facts and only one of them is worth a number
    on screen.

    ``default_model_label`` is the PARENT's model, used when the job did not
    record one of its own. That is the common case rather than a fallback: every
    child inherits the parent's spec unless ``run_subagent`` was given a
    ``model_spec`` override, and a child that WAS overridden records its own
    label — so the two together price a mixed-model fan-out correctly.

    MUST NOT BLOCK, and every path it takes is now a warm-memo hit, pure
    arithmetic, or a fire-and-forget background refresh. It is called from
    the Textual event loop (`app.py`'s `_harvest_subagent_costs`, on the 1 Hz
    poll), so anything added here that can wait on I/O freezes the keyboard.
    The paint-safe resolver (:func:`turn_cost`'s seam) is what keeps that
    true on a cold memo: a miss returns the registry row immediately and
    resolves the real price in a thread, so a child on a model this process
    has never priced costs one tick of "unpriceable", not a stalled frame.
    A new caller on the event loop should assume the memo is cold.

    Duck-typed means the two field reads are guarded, not just the pricing. The
    TUI runs against embedder hosts and replayed ledgers whose job objects are
    not ``AsyncJob`` at all, so ``job.usage`` can be a property with real work
    behind it; an exception escaping here takes down the whole band repaint, so
    one unreadable ledger row would cost every other row its number too.
    """
    try:
        usage = getattr(job, "usage", None)
        if usage is None:
            return None
        label = getattr(job, "model_label", None)
    except Exception:  # noqa: BLE001 — an unreadable job is not a render error
        return None
    return turn_cost(label or default_model_label or "", usage)

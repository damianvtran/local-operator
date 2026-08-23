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

Nothing here raises. A price is never worth a broken frame.
"""

from __future__ import annotations

from typing import Any

__all__ = ["turn_cost", "job_cost"]


def turn_cost(model_label: str, usage: Any) -> float | None:
    """What ``usage`` cost on the model named by ``model_label``, or ``None``.

    ``model_label`` is the ``provider/model_id`` spelling that
    :attr:`Session.model_label` produces.

    ``None`` means the price is genuinely unknown — no registry row, no provider
    listing and no aggregator entry could put a number on this model. It is NOT
    the same as ``0.0``, and a caller must not collapse the two: a confident
    ``$0.0000`` on a turn that billed tokens reads as "that was free", which is
    the more expensive lie of the two.

    Resolution is memoized per model per TTL bucket
    (:func:`~local_operator.model.configure.resolve_model_info`), so calling this
    once per subagent row on a 1 Hz repaint is a dict hit and four multiplies.
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
        from local_operator.model.configure import (
            _usage_cost,
            cost_for_usage,
            resolve_model_info,
        )

        reported = _usage_cost(usage)
        if reported is not None:
            return reported

        provider, _, model_id = model_label.partition("/")
        info = resolve_model_info(provider, model_id)
        if not (info.input_price or info.output_price):
            return None
        return cost_for_usage(provider, info, usage)
    except Exception:  # noqa: BLE001 — an unpriceable model is not a render error
        return None


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

    MUST NOT BLOCK, and today does not: every path it takes is a memo hit or
    pure arithmetic. It is called from the Textual event loop (`app.py`'s
    `_harvest_subagent_costs`, on the 1 Hz poll), so anything added here that
    can wait on I/O freezes the keyboard. The one hazard is indirect and
    already bounded: `resolve_model_info` will consult a listing for a model id
    it has not seen this TTL bucket, which is why callers that can be on a
    hot paint path resolve off-thread instead (`subagent_panel.job_stats`).
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

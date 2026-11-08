"""Every route leaves the wire seam carrying money, or an honest "unknown".

WHY THIS FILE EXISTS. A direct provider states only tokens on the wire: DeepSeek's
``usage`` object has ``prompt_tokens``/``completion_tokens`` and no dollar field,
because only an aggregator (OpenRouter) precomputes ``usage.cost``. Measured on
OSWorld episode ``t27a-t001`` (route ``deepseek/deepseek-flash``, DIRECT
api.deepseek.com): all 56 ``message_end`` rows carried real token counts and
``"usd_cost": null`` — no receipt. The harness already has the field that is
supposed to answer this (``Usage.estimated_usd_cost``: "a record-time table
estimate is durable money, but NOT a provider receipt"), and nothing wrote it,
so a sealed record could not tell "this call was free" apart from "nobody ever
priced this call" by reading the usage alone.

The three-way split these tests pin:

- a provider receipt is never overwritten and never invented;
- a priced route with no receipt records the shared table's estimate;
- an UNPRICED route records nothing at all — ``0.0`` would be a bill, and the
  whole point is that "we did not look" must stay distinguishable from "free".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.harness.types import ModelSpec, StreamUsageEvent, Usage
from local_operator.model.costs import cost_summary
from local_operator.providers.failover import _stamp_serving_spec

#: 2026-11-03T02:00:00Z -- a Tuesday inside DeepSeek's peak window
#: (``tariff.DEEPSEEK_TOU``: 01-04 and 06-10 UTC, weekdays only, stored at its
#: PEAK list rates). Pinned because the schedule halves every rate off-peak, so
#: an unstamped usage would make these dollar assertions depend on the wall
#: clock: the same call priced 0.30 at 02:00 and 0.15 at 04:00.
PEAK_MS = 1_793_671_200_000


def _stamp(provider: str, model_id: str, usage: Usage) -> Usage:
    """One usage event through the seam every provider's stream passes."""
    spec = ModelSpec(provider=provider, model_id=model_id)
    stamped = _stamp_serving_spec(
        StreamUsageEvent(usage=usage), spec, access=SimpleNamespace(kind="api_key")
    )
    assert isinstance(stamped, StreamUsageEvent)
    return stamped.usage


def test_direct_provider_records_a_table_estimate():
    """DeepSeek's shape: tokens and nothing else, so the table must price it.

    One million input tokens on ``deepseek/deepseek-flash`` is $0.30 from the
    registry row (0.3 in / 1.2 out per million), which makes the assertion an
    independent calculation rather than a restatement of the code under test.
    """
    usage = _stamp("deepseek", "deepseek-flash", Usage(input_tokens=1_000_000, at_ms=PEAK_MS))

    assert usage.usd_cost is None  # still no receipt — the provider said nothing
    assert usage.estimated_usd_cost == pytest.approx(0.30)


def test_provider_receipt_is_never_replaced_by_an_estimate():
    """An aggregator's own bill is ground truth; an estimate must not shadow it."""
    usage = _stamp(
        "openrouter", "deepseek/deepseek-flash", Usage(input_tokens=1_000_000, usd_cost=0.5)
    )

    assert usage.usd_cost == 0.5
    assert usage.estimated_usd_cost is None


def test_unpriced_route_records_unknown_not_zero():
    """No receipt AND no price row stays unknown; a 0.0 here would be a lie."""
    usage = _stamp("acme", "unlisted-model", Usage(input_tokens=1_000_000))

    assert usage.usd_cost is None
    assert usage.estimated_usd_cost is None


def test_sealed_record_keeps_money_and_provenance_apart():
    """The record-time estimate is durable money but not a provider bill.

    ``cost_summary(recorded_only=True)`` is what the session arm's sealed
    ``requests`` row reads: on the measured direct route it reported
    ``recorded_usd: null, recorded_unknown: true`` for every call even though
    the same usages priced fine from the table. A record-time estimate makes
    that row carry known money, and ``usd_cost`` staying null is what still says
    the provider did not bill it.
    """
    stamped = _stamp("deepseek", "deepseek-flash", Usage(input_tokens=1_000_000, at_ms=PEAK_MS))

    total, unknown = cost_summary([stamped], recorded_only=True)

    assert unknown is False
    assert total == pytest.approx(0.30)
    assert stamped.usd_cost is None


@pytest.mark.asyncio
async def test_a_streamed_direct_provider_turn_lands_priced():
    """The end-to-end seam: a whole streamed turn, not just the stamp function.

    ``stream_with_failover`` is what every provider call actually goes through,
    so this pins the operator-visible artefact: the usage event a caller receives
    on a DIRECT DeepSeek route carries money without pretending to be a bill.
    Scripted client, no network -- the wire shape is the one measured on the
    route (tokens only, no ``cost`` field).
    """
    from local_operator.harness.types import StreamEndEvent
    from local_operator.providers.failover import stream_with_failover
    from tests.unit.providers.test_failover import FakeAuth, ScriptedClient, _request

    scripted = ScriptedClient(
        [
            StreamUsageEvent(usage=Usage(input_tokens=1_000_000, at_ms=PEAK_MS)),
            StreamEndEvent(stop_reason="stop"),
        ]
    )

    async def client_for(spec):
        return scripted

    usages = [
        event.usage
        async for event in stream_with_failover(
            _request("deepseek", "deepseek-flash"),
            FakeAuth({"deepseek": ["synthetic"]}),
            {"retry": {"enabled": False}},
            client_for,
        )
        if isinstance(event, StreamUsageEvent)
    ]

    assert len(usages) == 1
    assert usages[0].usd_cost is None
    assert usages[0].estimated_usd_cost == pytest.approx(0.30)

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

The split these tests pin:

- a provider receipt is never overwritten and never invented;
- a priced route with no receipt records the shared table's estimate;
- an UNPRICED route records nothing at all — ``0.0`` would be a bill, and the
  whole point is that "we did not look" must stay distinguishable from "free";
- so does a priced route whose COUNTERS the provider never stated: the adapters
  zero-fill a missing counter for display and flag it, and pricing that
  placeholder is the same lie by another route (review B1, QA Q-F1).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamUsageEvent,
    Usage,
)
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
    """No receipt AND no price stays unknown; a 0.0 here would be a lie.

    BOTH unpriced shapes, because the contract names two and only one was
    pinned (review n1): ``acme`` has no registry row at all, while the Google
    row SHIPS with no prices -- a resolvable row that ``turn_cost`` still
    cannot price, which is the branch the docstring actually describes.
    """
    for provider, model_id in (
        ("acme", "unlisted-model"),
        ("google", "gemini-2.0-flash-lite-preview-02-05"),
    ):
        usage = _stamp(provider, model_id, Usage(input_tokens=1_000_000))

        assert usage.usd_cost is None
        assert usage.estimated_usd_cost is None, (provider, model_id)


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


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(0, 0), (5_000, 0), (0, 120)],
)
def test_unreported_counters_are_never_priced(input_tokens: int, output_tokens: int):
    """A placeholder zero is not a measurement, so it cannot become money.

    The wire adapters replace a missing or malformed counter with a display-safe
    ``0`` and record that they did (``Usage._token_counts_reported``), so these
    numbers are placeholders. Pricing them stamped a durable ``0.0`` (or a
    partial-token number) that ``_recorded_cost`` then reported as KNOWN money:
    an unmeasured call reading as a settled, and cheap, one -- the exact
    "we could not look" -> "this cost nothing" collapse the unpriced branch
    exists to prevent, arriving by a second route (review B1, QA Q-F1). The
    evaluation runner's own pricer already refuses this shape
    (``runner/provider_client._usage_from`` -> ``known=False``); this pins the
    record to the same answer.
    """
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    usage._token_counts_reported = False

    stamped = _stamp("deepseek", "deepseek-flash", usage)

    assert stamped.usd_cost is None
    assert stamped.estimated_usd_cost is None
    assert cost_summary([stamped], recorded_only=True) == (None, True)


def test_counters_the_provider_reported_as_zero_are_priced():
    """The paired control: a REAL zero is a price, and must stay one.

    Without this the guard above could be reading "all-zero tokens" instead of
    "nobody stated the tokens", and a provider that genuinely served nothing --
    or stated an explicit zero -- would lose its (correct) $0.00.
    """
    usage = Usage(input_tokens=0, output_tokens=0)
    usage._token_counts_reported = True

    stamped = _stamp("deepseek", "deepseek-flash", usage)

    assert stamped.usd_cost is None
    assert stamped.estimated_usd_cost == 0.0


@pytest.mark.asyncio
async def test_a_streamed_anthropic_turn_without_stated_counters_stays_unknown():
    """The reachable shape, driven through the REAL adapter and then the seam.

    Anthropic's ``message_start``/``message_delta`` frames carry ``usage`` and
    the adapter yields whatever they hold, so a body that states no counters
    (``{}`` / ``{}``) arrives at the seam flagged incomplete. Hand-building a
    ``Usage`` would prove only that the guard reads an attribute; this pins that
    the adapter actually sets it on this body, and that the priced route -- the
    one that produced a confident $0.00 before the guard -- prices nothing.
    """
    import httpx

    from local_operator.providers.clients import AnthropicClient
    from tests.unit.providers.test_clients import _collect, _spec, _sse

    body = _sse(
        [
            {"type": "message_start", "message": {"usage": {}}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
        ]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    async with httpx.AsyncClient(transport=transport) as http:
        client = AnthropicClient(http_client=http)
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec(provider="anthropic"), messages=[Message.user("hi")]),
                "synthetic",
            )
        )
    usage = next(event.usage for event in events if isinstance(event, StreamUsageEvent))
    assert usage._token_counts_reported is False

    stamped = _stamp("deepseek", "deepseek-flash", usage)

    assert stamped.usd_cost is None
    assert stamped.estimated_usd_cost is None
    assert cost_summary([stamped], recorded_only=True) == (None, True)


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

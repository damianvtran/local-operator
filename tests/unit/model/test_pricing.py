"""Pricing resolution, the placeholder guard, and cache-aware cost arithmetic.

The defect these cover: the ten current-generation Claude rows and every
shipping `gpt-5.x` resolved to ``input_price=0.0``, so the TUI's status band
rendered "cost unavailable" for the models the app actually runs on while the
older OpenRouter-priced rows costed fine. Two separate causes — placeholder
prices nobody filled in, and no price source at all for a direct provider whose
own listing quotes none — so there are two separate guards here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import Usage
from local_operator.model.configure import (
    _from_aggregator_catalogue,
    _from_price_catalogue,
    calculate_cost,
    cost_for_usage,
    invalidate_model_info_cache,
    resolve_model_info,
)
from local_operator.model.discovery import DiscoveredModel
from local_operator.model.registry import ModelInfo, anthropic_models, static_models

#: Registry rows that are legitimately unpriced, with the reason. Every one was
#: checked against the provider's own pricing page on 2026-08-10 and found to
#: have no published price — Google's `-exp-`/`-preview-` Gemini ids were
#: free-of-charge-only releases that have since been shut down (see
#: https://ai.google.dev/gemini-api/docs/deprecations), and neither small
#: Qwen2.5-Coder appears in Alibaba's DashScope price list or in the OpenRouter
#: catalogue.
#:
#: This set is the WHOLE point of :func:`test_no_unexplained_placeholder_prices`.
#: A new row added with `input_price=0.0` fails that test until someone either
#: prices it or adds it here with a reason — which is what did not happen the
#: last time, when ten shipping Claude rows sat at 0.0 for a whole generation.
UNPRICED_BY_DESIGN: dict[str, str] = {
    "google/gemini-2.0-flash-lite-preview-02-05": "free-of-charge preview, shut down 2025-12-09",
    "google/gemini-2.0-pro-exp-02-05": "experimental, never carried a published price",
    "google/gemini-2.0-flash-thinking-exp-01-21": "experimental, no published price",
    "google/gemini-2.0-flash-thinking-exp-1219": "experimental, no published price",
    "google/gemini-2.0-flash-exp": "experimental, no published price",
    "google/gemini-1.5-flash-002": "retired; absent from the current pricing page",
    "google/gemini-1.5-flash-exp-0827": "experimental, no published price",
    "alibaba/qwen2.5-coder-1.5b-instruct": "no DashScope-international published price",
    "alibaba/qwen2.5-coder-0.5b-instruct": "no DashScope-international published price",
}

#: The providers whose shipped rows this module audits.
_AUDITED_PROVIDERS = ("anthropic", "openai", "google", "alibaba", "deepseek", "mistral")


def _static_rows() -> list[tuple[str, ModelInfo]]:
    rows: list[tuple[str, ModelInfo]] = []
    for provider in _AUDITED_PROVIDERS:
        for model_id, info in static_models(provider).items():
            rows.append((f"{provider}/{model_id}", info))
    return rows


def test_no_unexplained_placeholder_prices() -> None:
    """Every shipped registry row is priced, or is listed as unpriceable.

    Fails on a NEW `input_price=0.0` row, which is exactly how the current
    Claude generation shipped unpriced: the placeholder is invisible in review
    because it is a valid float, and the only symptom is a status band that
    quietly says nothing.
    """
    unpriced = {
        label for label, info in _static_rows() if not (info.input_price or info.output_price)
    }
    unexplained = sorted(unpriced - set(UNPRICED_BY_DESIGN))
    assert not unexplained, (
        "registry rows carry placeholder 0.0 prices with no published-price "
        f"exemption: {unexplained}. Add the real per-MILLION-token price from the "
        "provider's own pricing page, or add the id to UNPRICED_BY_DESIGN with the "
        "reason it has none."
    )


def test_no_cache_read_costs_as_much_as_a_fresh_input_token() -> None:
    """A cache hit must be cheaper than reading the token for the first time.

    Otherwise caching is a cost, which is backwards on its face — Anthropic's own
    published multiplier is 0.1x base input, and no provider has ever charged a
    premium for a hit.

    This class of bad data used to be INERT: `calculate_cost` took input and output
    only, so a nonsense `cache_reads_price` sat in the registry unread. Pricing the
    cache buckets made it billable, and it would have over-billed exactly the
    warm-cache turns this agent runs all day. Nineteen rows were wrong when the
    guard was written: five Claude rows using `cache_reads_price = input_price` as
    a "no separate rate known" placeholder, and fourteen Qwen rows that had the
    four price fields filled in the order (input, output, input, output), making a
    cached read cost up to 4x a fresh one.

    ``None`` is the right value when a provider publishes no cache rate;
    `calculate_cost` falls back to the input price for it, which is wrong by the
    unknown discount rather than wrong in the expensive direction.
    """
    offenders = [
        f"{label} (input {info.input_price}, cache read {info.cache_reads_price})"
        for label, info in _static_rows()
        if info.cache_reads_price is not None
        and info.input_price
        and info.cache_reads_price >= info.input_price
    ]
    assert not offenders, (
        "these rows charge as much or more to read a CACHED token as a fresh one, "
        f"which now bills real money: {offenders}. Use the provider's published "
        "cache rate, or None if it publishes none \u2014 never the input or output price."
    )


def test_unpriced_exemptions_have_not_rotted() -> None:
    """An exemption for a row that no longer exists, or is now priced, is stale."""
    by_label = {label: info for label, info in _static_rows()}
    for label in UNPRICED_BY_DESIGN:
        info = by_label.get(label)
        assert info is not None, f"{label} is exempted from pricing but is not in the registry"
        assert not (
            info.input_price or info.output_price
        ), f"{label} is now priced — drop its UNPRICED_BY_DESIGN entry"


@pytest.mark.parametrize(
    "model_id,input_price,output_price,cache_read,cache_write",
    [
        # Anthropic's own "Model pricing" table, read 2026-08-10:
        # https://platform.claude.com/docs/en/about-claude/pricing
        ("claude-opus-5", 5.0, 25.0, 0.50, 6.25),
        # Introductory rate, in effect until 2026-08-31; guarded below.
        ("claude-sonnet-5", 2.0, 10.0, 0.20, 2.50),
        ("claude-fable-5", 10.0, 50.0, 1.0, 12.50),
        ("claude-haiku-4-5-20251001", 1.0, 5.0, 0.10, 1.25),
    ],
)
def test_current_generation_claude_is_priced(
    model_id: str,
    input_price: float,
    output_price: float,
    cache_read: float,
    cache_write: float,
) -> None:
    """The rows the app actually runs on carry the published rate, cache included."""
    info = anthropic_models[model_id]
    assert (info.input_price, info.output_price) == (input_price, output_price)
    assert (info.cache_reads_price, info.cache_writes_price) == (cache_read, cache_write)


# ---------------------------------------------------------------------------
# The catalogue fallback (a placeholder self-heals)
# ---------------------------------------------------------------------------
#
# Leg 2 of resolution is the ranked keyless chain (`_from_price_catalogue` →
# `prices.price_row`): models.dev FIRST, and OpenRouter's public listing under the
# vendor namespace ONLY when models.dev has no priced row. A user signed in only
# to Anthropic gets a release-day price from models.dev without another
# aggregator's document being fresh — and a gap in models.dev alone does not
# unprice the model, because the independent secondary still answers. Leg 3
# (`_from_aggregator_catalogue`) survives for an aggregator's OWN ids only.


def _catalogue(*rows: DiscoveredModel):
    """Patch the aggregator listing to answer with exactly ``rows``."""
    return patch(
        "local_operator.model.discovery.available_models",
        return_value=(list(rows), "ok"),
    )


def _price_catalogue(row: DiscoveredModel | None):
    """Patch the models.dev lookup to answer with exactly ``row``."""
    return patch("local_operator.model.prices.price_catalogue_row", return_value=row)


def test_zero_priced_registry_row_falls_back_to_the_catalogue() -> None:
    """A direct provider's placeholder is healed by the neutral catalogue's price.

    This is the mechanism that stops the placeholder class of bug recurring: a
    row added tomorrow with `0.0` starts costing correctly instead of silently
    reading as free. Anthropic's `/v1/models` quotes no prices at all, so
    without this leg the registry is the ONLY price source for a direct provider
    and a missing row is permanent.
    """
    placeholder = ModelInfo(id="claude-nova-6", name="claude-nova-6", description="")
    row = DiscoveredModel(
        id="claude-nova-6",
        name="Claude Nova 6",
        input_price=7.0,
        output_price=35.0,
        cache_read_price=0.7,
        cache_write_price=8.75,
    )
    with _price_catalogue(row):
        healed = _from_price_catalogue("anthropic", "claude-nova-6", placeholder)
    assert (healed.input_price, healed.output_price) == (7.0, 35.0)
    assert healed.cache_reads_price == 0.7
    assert healed.supports_prompt_cache is True


def test_the_write_price_reaches_model_info() -> None:
    """`cache_writes_price` is the catalogue's number, not the input price.

    The input-price stand-in under-states an Anthropic 5m write by 20%; both
    consumers used to substitute it because `DiscoveredModel` had no write price.
    """
    placeholder = ModelInfo(id="claude-fable-5-1", name="x", description="")
    row = DiscoveredModel(
        id="claude-fable-5-1",
        input_price=10.0,
        output_price=50.0,
        cache_read_price=0.25,
        cache_write_price=12.5,
    )
    with _price_catalogue(row):
        healed = _from_price_catalogue("anthropic", "claude-fable-5-1", placeholder)
    assert healed.cache_writes_price == 12.5, "the input price was substituted for the write"


def test_the_input_price_fallback_survives_a_listing_with_no_write_price() -> None:
    """A catalogue that quotes a read price and no write price still gets the floor."""
    placeholder = ModelInfo(id="gpt-9", name="x", description="")
    row = DiscoveredModel(id="gpt-9", input_price=2.0, output_price=8.0, cache_read_price=0.2)
    with _price_catalogue(row):
        healed = _from_price_catalogue("openai", "gpt-9", placeholder)
    assert healed.cache_writes_price == 2.0


def test_a_direct_provider_is_priced_from_the_neutral_catalogue_not_openrouter() -> None:
    """Zero OpenRouter requests on an Anthropic-only install when models.dev answers.

    The provider's own listing answers unpriced (as Anthropic's does), models.dev
    prices it, and the OpenRouter document is never read for a direct id the
    primary already priced.
    """
    invalidate_model_info_cache()
    unpriced = DiscoveredModel(id="claude-nova-6", context_window=1_000_000, max_tokens=128_000)
    priced = DiscoveredModel(
        id="claude-nova-6",
        input_price=7.0,
        output_price=35.0,
        cache_read_price=0.7,
        cache_write_price=8.75,
    )
    consulted: list[str] = []

    def listing(provider_id, **kwargs):
        consulted.append(provider_id)
        return [unpriced], "ok"

    with (
        patch("local_operator.model.discovery.available_models", side_effect=listing),
        _price_catalogue(priced),
    ):
        info = resolve_model_info("anthropic", "claude-nova-6")
    invalidate_model_info_cache()

    assert (info.input_price, info.output_price) == (7.0, 35.0)
    assert info.cache_writes_price == 8.75
    assert info.context_window == 1_000_000, "the provider's own limit wins over the catalogue"
    assert consulted == ["anthropic"], f"OpenRouter was consulted for a direct id: {consulted}"


def test_openrouter_ids_still_fall_back_to_their_own_listing() -> None:
    """Leg 3 is for an aggregator's own ids, and nothing else."""
    placeholder = ModelInfo(id="some/model", name="x", description="")
    row = DiscoveredModel(id="some/model", input_price=9.0, output_price=27.0)
    with _catalogue(row) as listing:
        healed = _from_aggregator_catalogue("openrouter", "some/model", placeholder)
    assert healed.input_price == 9.0
    assert listing.call_args.kwargs["want_id"] == "some/model"


def test_the_aggregator_leg_refuses_a_direct_provider() -> None:
    """Leg 3 never prices a direct id: OpenRouter reaches those only as the
    SECONDARY step of leg 2's chain, behind models.dev, never on its own."""
    placeholder = ModelInfo(id="claude-nova-6", name="x", description="")
    row = DiscoveredModel(id="anthropic/claude-nova-6", input_price=7.0)
    with _catalogue(row) as listing:
        untouched = _from_aggregator_catalogue("anthropic", "claude-nova-6", placeholder)
    assert untouched.input_price == 0.0
    listing.assert_not_called()


def test_catalogue_fallback_supplies_a_missing_context_window() -> None:
    """A shipping model with no window is the same defect wearing a different hat.

    `openai/gpt-5.4` has no registry row and OpenAI's `/v1/models` is bare ids, so
    it resolved with no window at all: the band rendered `311.0k/—` and, worse,
    the compaction threshold derives from the window, so compaction silently never
    fires and the turn eventually 400s on the provider's real limit.
    """
    bare = ModelInfo(id="gpt-9", name="gpt-9", description="", context_window=-1, max_tokens=-1)
    row = DiscoveredModel(
        id="gpt-9",
        name="",
        input_price=2.0,
        output_price=8.0,
        context_window=400_000,
        max_tokens=64_000,
    )
    with _price_catalogue(row):
        healed = _from_price_catalogue("openai", "gpt-9", bare)
    assert (healed.context_window, healed.max_tokens) == (400_000, 64_000)


def test_catalogue_fallback_never_overrides_a_window_we_already_have() -> None:
    """The direct provider's own limits win where they exist.

    OpenRouter advertises the LARGEST window across its routes, which is the wrong
    number for one specific upstream endpoint — so this leg fills holes and never
    corrects.
    """
    described = ModelInfo(
        id="claude-opus-4-5-20251101",
        name="x",
        description="",
        context_window=200_000,
        max_tokens=64_000,
    )
    row = DiscoveredModel(
        id="claude-opus-4.5",
        name="",
        input_price=5.0,
        context_window=1_000_000,
        max_tokens=128_000,
    )
    with _price_catalogue(row):
        untouched = _from_price_catalogue("anthropic", "claude-opus-4-5-20251101", described)
    assert (untouched.context_window, untouched.max_tokens) == (200_000, 64_000)
    assert untouched.input_price == 5.0, "the price hole was still filled"


def test_catalogue_fallback_leaves_an_unknown_model_alone() -> None:
    """No match means no price, not a plausible neighbour's price."""
    placeholder = ModelInfo(id="claude-nova-6", name="x", description="")
    with _price_catalogue(None):
        untouched = _from_price_catalogue("anthropic", "claude-nova-6", placeholder)
    assert (untouched.input_price, untouched.output_price) == (0.0, 0.0)


def test_catalogue_fallback_ignores_a_stub_with_no_cost_and_no_limit() -> None:
    """A plan-catalogue stub answers neither question this leg is here for."""
    placeholder = ModelInfo(id="k3", name="x", description="")
    with _price_catalogue(DiscoveredModel(id="k3", name="Kimi K3")):
        untouched = _from_price_catalogue("kimi", "k3", placeholder)
    assert untouched == placeholder


def test_a_priced_row_never_reaches_the_aggregator_leg() -> None:
    """The SECOND listing is what a described row must not pay for.

    The aggregator leg re-evaluates ``_needs_enrichment`` after the provider's own
    listing rather than sharing one gate with it, precisely so this stays true. The
    provider's own listing IS consulted for `claude-opus-5` — that row's limits are
    a dated transcription of it (`limits_from_listing`) — but once it has answered,
    nothing is missing and the aggregator is never asked.
    """
    invalidate_model_info_cache()
    with patch(
        "local_operator.model.discovery.available_models", return_value=([], "ok")
    ) as listing:
        info = resolve_model_info("anthropic", "claude-opus-5")
    assert info.input_price == 5.0
    consulted = [call.args[0] for call in listing.call_args_list]
    assert "openrouter" not in consulted, (
        "a fully described row paid for the aggregator catalogue: " f"{consulted}"
    )


def test_a_transcribed_row_re_asks_the_provider_so_a_window_can_be_corrected() -> None:
    """Pricing the Claude rows must not freeze their limits at the transcription.

    Those ten rows entered enrichment only through the PRICE clause, by accident of
    carrying `0.0`. Pricing them closed that door, and with it Anthropic's ability
    to ever correct an Opus 5 window again — the exact `1.8%/200k`-on-a-1M-model
    failure the registry header blames for the rows existing. `limits_from_listing`
    is what reopens it.
    """
    invalidate_model_info_cache()
    live = DiscoveredModel(
        id="claude-opus-5", name="Claude Opus 5", context_window=2_000_000, max_tokens=200_000
    )
    with _catalogue(live):
        info = resolve_model_info("anthropic", "claude-opus-5")
    assert (info.context_window, info.max_tokens) == (2_000_000, 200_000)
    assert info.input_price == 5.0, "the listing quotes no price; the registry's must survive"
    invalidate_model_info_cache()


def test_an_opportunistic_refresh_gets_a_short_budget_and_a_blocking_one_does_not() -> None:
    """A re-ask must not be able to freeze a repaint for the full listing ceiling.

    `limits_from_listing` puts the ten Claude rows back on the listing path, and
    that path is reachable from the TUI's subagent panel, which resolves a child's
    model on its paint timer. A complete row already has a usable answer, so a slow
    provider must cost a stale number rather than the frame. A row that is genuinely
    missing its window keeps the full ceiling: the session cannot run without it.
    """
    from local_operator.model.discovery import DEFAULT_TIMEOUT_S

    seen: list[float | None] = []

    def record(provider_id, **kwargs):
        seen.append(kwargs.get("timeout"))
        return [], "ok"

    with patch("local_operator.model.discovery.available_models", side_effect=record):
        invalidate_model_info_cache()
        resolve_model_info("anthropic", "claude-opus-5")  # complete + transcribed
        refresh = seen[0]
        seen.clear()
        resolve_model_info("anthropic", "claude-nova-6")  # nothing known about it
        blocking = seen[0]
    invalidate_model_info_cache()

    assert (
        refresh is not None and refresh < DEFAULT_TIMEOUT_S
    ), f"an opportunistic refresh may block for {refresh}s on a paint path"
    assert blocking == DEFAULT_TIMEOUT_S, "a model with no window must still be waited for"


def test_the_two_listing_legs_share_one_ceiling_rather_than_summing() -> None:
    """An unresolvable model must not block for both budgets in series.

    Independent per-leg ceilings COMPOSE INTO THEIR SUM, so adding the aggregator
    leg silently took an unknown model's worst case above the one ceiling every
    caller of this module budgets for. That model is not exotic for the subagent
    panel — a child launched on a `model_spec` override the registry has never
    heard of is exactly what that panel exists to make visible.
    """
    from local_operator.model.discovery import DEFAULT_TIMEOUT_S

    # What is asserted is elapsed-so-far PLUS the next grant, not grant plus
    # grant: leg 1 is allowed the full ceiling, and the guarantee is that leg 2
    # only ever gets what leg 1 did not actually spend. `spent` makes leg 1 burn
    # nearly the whole budget so leg 2's share is squeezed to almost nothing —
    # the shape of the case the panel hits.
    spent = DEFAULT_TIMEOUT_S - 0.4
    seen: list[float | None] = []
    clock = [0.0]

    def slow(provider_id, **kwargs):
        seen.append(kwargs.get("timeout"))
        if len(seen) == 1:
            clock[0] += spent  # leg 1 blocks for nearly the whole ceiling
        return [], "ok"

    def price_leg(provider, model_id, *, timeout, **kwargs):
        seen.append(timeout)
        return None

    with (
        patch("local_operator.model.discovery.available_models", side_effect=slow),
        patch("local_operator.model.prices.price_catalogue_row", side_effect=price_leg),
        patch("local_operator.model.configure.time.monotonic", side_effect=lambda: clock[0]),
    ):
        invalidate_model_info_cache()
        resolve_model_info("anthropic", "claude-nova-6")
    invalidate_model_info_cache()

    assert len(seen) == 2, f"expected the provider leg then the price-catalogue leg, got {seen}"
    provider_leg, price_leg_budget = seen
    assert provider_leg == DEFAULT_TIMEOUT_S, "a blocking resolve must keep the full ceiling"
    assert price_leg_budget is not None
    assert spent + price_leg_budget <= DEFAULT_TIMEOUT_S + 0.01, (
        f"leg 1 spent {spent}s and leg 2 was still granted {price_leg_budget}s, "
        f"over the {DEFAULT_TIMEOUT_S}s ceiling"
    )


def test_sonnet_5_carries_the_standard_price() -> None:
    """Sonnet 5's $2/$10/$2.50/$0.20 is the permanent standard rate — pin it.

    This replaces a dated guard that watched for the introductory period ending
    2026-08-31. That expiry never happened: Anthropic cancelled the scheduled
    2026-09-01 rise to $3/$15 and made these rates permanent on 2026-08-10
    (launch-post changelog edit on https://www.anthropic.com/news/claude-sonnet-5,
    and the `claude-sonnet-5-introductory-pricing` note on
    https://platform.claude.com/docs/en/about-claude/pricing). The old guard duly
    fired on 2026-09-01 UTC and failed every PR in the repo while instructing an
    edit that would have over-reported every Sonnet 5 call by 50%.

    With no announced end date there is no longer a date worth watching, so the
    risk inverts: not "a correct value goes stale on a known day" but "someone
    copies one of the many third-party tables that still print the cancelled
    $3/$15 increase". Hence a pin on the published numbers rather than a clock —
    deliberately time-insensitive, so it can never fail for merely being run on a
    later date.
    """
    info = anthropic_models["claude-sonnet-5"]
    actual = (
        info.input_price,
        info.output_price,
        info.cache_writes_price,
        info.cache_reads_price,
    )
    assert actual == (2.0, 10.0, 2.50, 0.20), (
        f"Claude Sonnet 5's registry price is {actual}, not the published standard "
        "(2.0, 10.0, 2.50, 0.20). $2/$10 is permanent as of 2026-08-10 — the "
        "$3/$15 increase once scheduled for 2026-09-01 was cancelled. Re-verify "
        "against https://platform.claude.com/docs/en/about-claude/pricing before "
        "changing this, not against a third-party pricing table."
    )


# ---------------------------------------------------------------------------
# Cache-aware arithmetic
# ---------------------------------------------------------------------------


def _priced(**overrides: Any) -> ModelInfo:
    fields: dict[str, Any] = {
        "id": "m",
        "name": "m",
        "description": "",
        "input_price": 10.0,
        "output_price": 100.0,
        "cache_reads_price": 1.0,
        "cache_writes_price": 12.5,
    }
    fields.update(overrides)
    return ModelInfo(**fields)


def test_calculate_cost_prices_each_bucket_at_its_own_rate() -> None:
    cost = calculate_cost(_priced(), 1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert cost == pytest.approx(10.0 + 100.0 + 1.0 + 12.5)


def test_calculate_cost_falls_back_to_input_rate_without_a_cache_price() -> None:
    """Cached tokens were billed at SOMETHING; free is the one certainly-wrong answer."""
    model = _priced(cache_reads_price=None, cache_writes_price=None)
    assert calculate_cost(model, 0, 0, 1_000_000, 0) == pytest.approx(10.0)
    assert calculate_cost(model, 0, 0, 0, 1_000_000) == pytest.approx(10.0)


def test_calculate_cost_keeps_its_legacy_two_count_contract() -> None:
    """The three-argument form still prices input and output only."""
    assert calculate_cost(_priced(), 1_000_000, 1_000_000) == pytest.approx(110.0)


def test_anthropic_cache_tokens_are_billed_alongside_input_not_within_it() -> None:
    """Anthropic reports ``input_tokens`` EXCLUDING its cache buckets.

    So the three buckets are disjoint and each is priced at its own rate, with
    nothing added to or subtracted from the input count. Dropping the cache
    buckets would undercount a warm agent turn by most of its input: prompt
    caching is on and the cached prefix is the bulk of the prompt.
    """
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost_for_usage("anthropic", _priced(), usage) == pytest.approx(10.0 + 1.0 + 12.5)


def test_openai_usage_carves_cache_tokens_out_of_input() -> None:
    """OpenAI reports cached tokens as a SUBSET of ``input_tokens``.

    Pricing input and cache_read side by side would charge the cached prefix
    twice — once at the full rate and once at the cached rate.
    """
    usage = Usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=900_000)
    # 100k uncached at $10/MTok + 900k cached at $1/MTok.
    assert cost_for_usage("openai", _priced(), usage) == pytest.approx(1.0 + 0.9)


def test_cache_tokens_exceeding_input_never_bill_negative() -> None:
    """A malformed provider report is not a reason to hand back a credit."""
    usage = Usage(input_tokens=10, output_tokens=0, cache_read_tokens=1_000_000)
    assert cost_for_usage("openai", _priced(), usage) == pytest.approx(1.0)


def test_cost_for_usage_reads_a_serialized_usage_mapping() -> None:
    """A child's usage arrives rehydrated from a serialized event, as a dict."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert cost_for_usage("anthropic", _priced(), usage) == pytest.approx(10.0)


def test_cost_for_usage_prefers_a_provider_reported_dollar() -> None:
    """OpenRouter's ``usage.cost`` is the provider's own bill: it must be
    returned verbatim, never re-derived from tokens — a reconstruction cannot
    reproduce the per-route price, reasoning split, or override that produced it."""
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, usd_cost=0.0075)
    # Note the token-shaped usage would price to ~110.0 here; the reported
    # figure must WIN, so the assertion is the figure, not the estimate.
    assert cost_for_usage("openrouter", _priced(), usage) == pytest.approx(0.0075)


def test_cost_for_usage_reported_dollar_wins_even_against_big_table_price() -> None:
    """The preference is by presence, not by magnitude: even when the table
    would price the token counts far higher, the provider's printed number is the
    fact and the table is the estimate."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "usd_cost": 0.0075}
    assert cost_for_usage("openrouter", _priced(), usage) == pytest.approx(0.0075)


def test_cost_for_usage_falls_back_to_estimate_without_a_reported_dollar() -> None:
    """``usd_cost`` absent means the estimate is still the answer."""
    usage = Usage(input_tokens=1_000_000, output_tokens=0)
    assert cost_for_usage("openrouter", _priced(), usage) == pytest.approx(10.0)


def test_cost_for_usage_treats_a_malformed_reported_dollar_as_absent() -> None:
    """A non-numeric or negative reported amount is provider noise, not money:
    it must degrade to the estimate, not raise or bill a credit."""
    assert cost_for_usage(
        "openrouter", _priced(), Usage(input_tokens=1_000_000, usd_cost=-5)
    ) == pytest.approx(10.0)
    assert cost_for_usage(
        "openrouter", _priced(), {"input_tokens": 1_000_000, "usd_cost": "x"}
    ) == pytest.approx(10.0)


def test_cost_for_usage_non_finite_reported_dollar_falls_back_to_estimate() -> None:
    """A non-finite reported amount is malformed provider data and is
    wire-reachable (``json.loads`` accepts ``Infinity``/``NaN`` by default). An
    unfloored ``inf`` would poison every summed total forever, so it must degrade
    to the token estimate exactly like a negative or non-numeric amount, and the
    result must be finite."""
    import math

    for bad in (float("inf"), float("-inf"), float("nan")):
        result = cost_for_usage(
            "openrouter", _priced(), Usage(input_tokens=1_000_000, usd_cost=bad)
        )
        assert result == pytest.approx(10.0)
        assert math.isfinite(result)


def test_cost_for_usage_reported_zero_is_a_fact_not_an_absence() -> None:
    """A real ``0.0`` from the provider means "billed as free", which is not the
    same as "not reported" — it must return 0.0, not fall through to the estimate."""
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, usd_cost=0.0)
    assert cost_for_usage("openrouter", _priced(), usage) == 0.0

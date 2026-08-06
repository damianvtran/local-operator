"""Unit tests for the provider usage-quota fetchers and the ProviderController.

The fetchers are tested against canned HTTP responses (httpx MockTransport) so
the parsing contract is pinned without any live network. ``fetch_usage`` and
``usage_supported`` are the public dispatch surface; each provider's parser is
exercised through it.
"""

from __future__ import annotations

import pytest
import httpx

from local_operator.providers.usage import (
    UsageAmount,
    UsageLimit,
    fetch_usage,
    usage_supported,
)


def _client_for(payload, status: int = 200) -> httpx.AsyncClient:
    """An httpx async client that returns a canned JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_openrouter_credits_parse() -> None:
    payload = {"data": {"usage": 12.5, "limit": 100.0, "is_free_tier": False}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="sk-or-test")
    assert report is not None
    assert len(report.limits) == 1
    limit = report.limits[0]
    assert limit.amount.used == pytest.approx(12.5)
    assert limit.amount.limit == pytest.approx(100.0)
    assert limit.amount.fraction() == pytest.approx(0.125)
    assert limit.effective_status() == "ok"


@pytest.mark.asyncio
async def test_openrouter_free_tier_has_no_budget() -> None:
    payload = {"data": {"usage": 0.0, "limit": 0.0, "is_free_tier": True}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="sk-or-test")
    assert report is not None
    assert report.limits == []  # no fabricated 0/0 budget
    assert report.notes is not None and "free" in report.notes


@pytest.mark.asyncio
async def test_openrouter_no_limit_reports_spend_only() -> None:
    payload = {"data": {"usage": 519.7, "limit": 0.0, "is_free_tier": False}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="sk-or-test")
    assert report is not None
    limit = report.limits[0]
    assert limit.amount.used == pytest.approx(519.7)
    assert limit.amount.limit is None
    assert limit.amount.fraction() is None  # no limit -> unmeasurable
    assert limit.effective_status() == "unknown"


@pytest.mark.asyncio
async def test_kimi_api_key_reaches_the_balance_endpoint() -> None:
    """The widest gap this module had: the registry stores `KIMI_API_KEY` while the
    only Kimi fetcher wanted an OAuth token, so an API-key user was told Kimi
    reports quota and got an empty table forever."""
    payload = {
        "data": {"available_balance": 12.5, "voucher_balance": 2.5, "cash_balance": 10.0},
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "kimi", api_key="sk-moonshot")
    assert report is not None
    limit = report.limits[0]
    assert limit.id == "kimi:balance"
    assert limit.amount.remaining == pytest.approx(12.5)
    assert limit.amount.unit == "usd"
    assert report.notes is not None and "voucher" in report.notes


@pytest.mark.asyncio
async def test_an_oauth_token_still_wins_where_both_routes_exist() -> None:
    """For a provider with both, the OAuth route reports the SUBSCRIPTION the user
    is actually spending; the balance a subscription user never draws down is the
    less useful answer."""
    payload = {"data": {"limits": [{"name": "coding", "used": 1, "limit": 10}]}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "kimi", api_key="sk-moonshot", access_token="tok")
    assert report is not None
    assert all(lim.id != "kimi:balance" for lim in report.limits), report.limits


@pytest.mark.asyncio
async def test_deepseek_reports_a_balance_per_currency() -> None:
    """A CNY balance rendered as USD would be wrong by roughly a factor of seven,
    so the currency is part of the id and only USD claims the dollar unit."""
    payload = {
        "is_available": True,
        "balance_infos": [
            {"currency": "CNY", "total_balance": "70.00"},
            {"currency": "USD", "total_balance": "9.85"},
        ],
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "deepseek", api_key="sk-ds")
    assert report is not None
    ids = [lim.id for lim in report.limits]
    assert ids == ["deepseek:balance:cny", "deepseek:balance:usd"]
    units = {lim.id: lim.amount.unit for lim in report.limits}
    assert units["deepseek:balance:cny"] == "unknown"
    assert units["deepseek:balance:usd"] == "usd"


@pytest.mark.asyncio
async def test_an_unavailable_deepseek_account_says_so() -> None:
    """A zero balance and a suspended account look identical in the numbers."""
    payload = {"is_available": False, "balance_infos": [{"currency": "USD", "total_balance": 0.0}]}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "deepseek", api_key="sk-ds")
    assert report is not None
    assert report.notes == "account not available for requests"


@pytest.mark.asyncio
async def test_anthropic_oauth_parses_five_hour() -> None:
    payload = {
        "five_hour": {"used": 30, "limit": 100, "resets_at": "2026-08-05T12:00:00Z"},
        "limits": [],
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    labels = [lim.label for lim in report.limits]
    assert "5 hour" in labels


@pytest.mark.asyncio
async def test_openai_oauth_parses_primary_window() -> None:
    payload = {"primary": {"used_percent": 40, "window_minutes": 300}, "secondary": {}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openai", access_token="tok")
    assert report is not None
    primary = next(lim for lim in report.limits if lim.id == "openai:primary")
    assert primary.amount.fraction() == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_oauth_provider_without_token_returns_none() -> None:
    # OAuth-only provider with no access token -> None (no report).
    client = _client_for({})
    async with client:
        report = await fetch_usage(client, "anthropic", api_key="sk-key")
    assert report is None


@pytest.mark.asyncio
async def test_unknown_provider_returns_none() -> None:
    client = _client_for({})
    async with client:
        report = await fetch_usage(client, "nonsense", api_key="k")
    assert report is None


def test_usage_supported() -> None:
    assert usage_supported("openrouter") is True
    assert usage_supported("deepseek") is True
    assert usage_supported("nonsense") is False
    # `zai` had a working fetcher and no ProviderDefinition, so no code path could
    # ever supply its credential: `/login zai` raised, its env var was never read,
    # and the set advertised a provider nobody could reach.
    assert usage_supported("zai") is False


def test_usage_kinds_distinguishes_no_endpoint_from_no_credential() -> None:
    """Those look identical in an empty table, and only the second is actionable."""
    from local_operator.providers.usage import usage_kinds

    assert usage_kinds("anthropic") == (True, False)  # OAuth only
    assert usage_kinds("openrouter") == (False, True)  # API key only
    assert usage_kinds("kimi") == (True, True)  # both routes
    assert usage_kinds("google") == (False, False)  # no endpoint at all


def test_the_oauth_only_set_is_derived_from_the_dispatch_table() -> None:
    """The hand-written version had already drifted from the table and was read by
    nothing, which is how it stayed wrong."""
    from local_operator.providers.usage import OAUTH_USAGE_PROVIDERS

    assert "anthropic" in OAUTH_USAGE_PROVIDERS
    assert "kimi" not in OAUTH_USAGE_PROVIDERS, "kimi now has an API-key route"
    assert "openrouter" not in OAUTH_USAGE_PROVIDERS


@pytest.mark.asyncio
async def test_failure_is_not_a_crash() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="k")
    assert report is None


def test_usage_amount_fraction_precedence() -> None:
    # Explicit used_fraction wins over derived used/limit.
    a = UsageAmount(used=50, limit=100, used_fraction=0.25)
    assert a.fraction() == pytest.approx(0.25)
    # Derive from used/limit when no explicit fraction.
    b = UsageAmount(used=25, limit=100)
    assert b.fraction() == pytest.approx(0.25)
    # Derive from used+remaining when limit absent.
    c = UsageAmount(used=10, remaining=90)
    assert c.fraction() == pytest.approx(0.1)
    # Nothing measurable -> None.
    assert UsageAmount(used=10).fraction() is None


def test_usage_amount_status_thresholds() -> None:
    assert UsageAmount(used_fraction=0.5).status() == "ok"
    assert UsageAmount(used_fraction=0.9).status() == "warning"
    assert UsageAmount(used_fraction=1.0).status() == "exhausted"
    assert UsageAmount().status() == "unknown"


def test_usage_limit_effective_status_defaults_to_amount() -> None:
    limit = UsageLimit(id="x", label="x", amount=UsageAmount(used_fraction=0.95))
    assert limit.effective_status() == "warning"
    explicit = UsageLimit(id="y", label="y", amount=UsageAmount(), status="ok")
    assert explicit.effective_status() == "ok"

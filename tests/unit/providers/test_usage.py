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
async def test_zai_parses_limits_and_percentage() -> None:
    payload = {
        "data": {
            "limits": [
                {
                    "type": "tokens",
                    "usage": 800,
                    "currentValue": 800,
                    "percentage": 80,
                    "nextResetTime": "2026-08-05T00:00:00Z",
                }
            ]
        }
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "zai", api_key="rawtoken")
    assert report is not None
    limit = report.limits[0]
    assert limit.effective_status() == "ok"  # 80% < 0.85 warning threshold
    assert limit.amount.fraction() == pytest.approx(0.8)


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
    assert usage_supported("zai") is True
    assert usage_supported("deepseek") is False
    assert usage_supported("nonsense") is False


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

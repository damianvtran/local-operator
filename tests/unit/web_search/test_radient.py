"""Transport tests for Radient-hosted search (``web_search/radient.py``).

Wire shapes are the ones measured live against prod ``GET /v1/tools/search``
on 2026-11-24 (design §1): 200 with a ``cost`` object, 402 on insufficient
credits, 401 on a bad bearer, and a 200 that can still carry ``error``.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.web_search import radient as radient_module
from local_operator.web_search.models import WebSearchSettings

BASE = "https://radient.test/v1"

COST = {
    "v": 1,
    "amount_usd": 0.01,
    "basis": "billed",
    "units": 1,
    "unit": "search",
    "unit_price_usd": 0.01,
    "provider": "tavily",
    "tool_type": "web_search",
    "usage_record_id": "6b05ad92-usage",
}


@pytest.fixture(autouse=True)
def _signed_in(monkeypatch):
    radient_module.reset_radient_credit_cache_for_tests()
    monkeypatch.setattr(radient_module, "_radient_base_url", lambda _c: BASE)

    async def token(_c, _b):
        return "tok"

    monkeypatch.setattr(radient_module, "_radient_token", token)
    yield
    radient_module.reset_radient_credit_cache_for_tests()


async def _search(handler, calls: list[httpx.Request] | None = None):
    def record(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
        return await radient_module.search_radient(client, None, WebSearchSettings(), "q", 2)


@pytest.mark.asyncio
async def test_200_maps_results_and_records_the_billed_cost() -> None:
    calls: list[httpx.Request] = []
    response = await _search(
        lambda _r: httpx.Response(
            200,
            json={
                "query": "q",
                "provider": "tavily",
                "total_found": 3,
                "results": [
                    {"title": "A", "url": "https://a.test", "content": "alpha", "score": 0.9},
                    {"title": "B", "url": "https://b.test", "content": "beta", "score": 0.5},
                    {"title": "C", "url": "https://c.test", "content": "gamma", "score": 0.1},
                ],
                "cost": COST,
            },
        ),
        calls,
    )
    assert [s.url for s in response.sources] == ["https://a.test", "https://b.test"]
    assert response.sources[0].snippet == "alpha"
    # The 0-1 score is a different scale from the 0-100 relevance hint.
    assert response.sources[0].relevance is None
    assert response.provider == "radient" and response.auth_mode == "radient-credits"
    assert response.cost is not None
    assert response.cost.usd == 0.01
    assert response.cost.basis == "billed by Radient (tavily)"
    assert response.cost.priced_from_usage is True
    assert response.usage is not None and response.usage.credits == 1.0
    assert response.request_id == "6b05ad92-usage"
    (request,) = calls
    assert str(request.url).startswith(f"{BASE}/tools/search?")
    assert request.headers["authorization"] == "Bearer tok"
    assert request.url.params["provider"] == "tavily"
    assert request.url.params["search_depth"] == "basic"
    assert request.url.params["include_raw"] == "false"


@pytest.mark.asyncio
async def test_200_without_cost_falls_back_to_the_published_rate() -> None:
    response = await _search(
        lambda _r: httpx.Response(
            200, json={"query": "q", "results": [{"title": "A", "url": "https://a.test"}]}
        )
    )
    assert response.cost is not None
    assert response.cost.usd == 0.01
    assert response.cost.priced_from_usage is False
    assert "Radient sent no cost" in response.cost.basis


@pytest.mark.asyncio
async def test_402_is_cached_and_the_next_search_makes_no_request() -> None:
    calls: list[httpx.Request] = []
    refused = lambda _r: httpx.Response(402, json={"error": "insufficient credits"})  # noqa: E731
    with pytest.raises(RuntimeError, match="credit balance is too low"):
        await _search(refused, calls)
    assert len(calls) == 1
    remaining = radient_module.radient_credit_refusal_remaining()
    assert remaining is not None and 0 < remaining <= radient_module.RADIENT_CREDIT_TTL_SECONDS
    with pytest.raises(RuntimeError, match="credit balance is too low"):
        await _search(refused, calls)
    assert len(calls) == 1, "a fresh 402 verdict must skip the round trip"


@pytest.mark.asyncio
async def test_401_asks_to_sign_in_again_and_is_not_cached() -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(RuntimeError, match="local-operator login radient"):
        await _search(
            lambda _r: httpx.Response(401, json={"error": "invalid API key format"}), calls
        )
    assert radient_module.radient_credit_refusal_remaining() is None
    with pytest.raises(RuntimeError):
        await _search(lambda _r: httpx.Response(401), calls)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_200_with_error_raises_the_server_text() -> None:
    with pytest.raises(RuntimeError, match="Radient search failed: upstream boom"):
        await _search(
            lambda _r: httpx.Response(
                200, json={"query": "q", "results": [], "error": "upstream boom"}
            )
        )


@pytest.mark.asyncio
async def test_empty_results_return_an_empty_response_for_the_chain_to_skip() -> None:
    response = await _search(lambda _r: httpx.Response(200, json={"query": "q", "results": []}))
    assert response.sources == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "match"),
    [
        (lambda _r: httpx.Response(200, text="<html>"), "non-JSON"),
        (lambda _r: httpx.Response(200, json=["not", "an", "object"]), "non-object"),
        (lambda _r: httpx.Response(500, text="boom"), "Radient returned HTTP 500"),
    ],
)
async def test_malformed_and_server_errors_raise(handler, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        await _search(handler)


@pytest.mark.asyncio
async def test_a_missing_credential_at_call_time_names_the_login(monkeypatch) -> None:
    async def no_token(_c, _b):
        return ""

    monkeypatch.setattr(radient_module, "_radient_token", no_token)
    calls: list[httpx.Request] = []
    with pytest.raises(RuntimeError, match="not signed in to Radient"):
        await _search(lambda _r: httpx.Response(200, json={}), calls)
    assert calls == []


def test_the_sync_client_model_keeps_cost_and_total_found() -> None:
    from local_operator.clients.radient import RadientSearchResponse

    parsed = RadientSearchResponse.model_validate(
        {"query": "q", "results": [], "cost": COST, "total_found": 7}
    )
    assert parsed.cost == COST
    assert parsed.total_found == 7

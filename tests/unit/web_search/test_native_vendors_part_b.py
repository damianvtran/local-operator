"""Native-search vendors, part B (google, kimi, zai, mistral, openrouter).

Same seams as ``test_native_vendors.py`` (whose helpers are reused): credentials
are pinned at ``web_search.native.credentials`` and every transport runs over
``httpx.MockTransport``. ``openrouter_responses_live.json`` is a trimmed, redacted
LIVE capture (2026-11-24); the other part-B fixtures are documented shapes
(no credential for those legs on the probe machine; each file's ``_note`` says so).

The chain rows put the plan-class Z.AI leg in the NATIVE band before Radient and
the model-key legs after it, cheapest-first (design §6 rows 7-9 with part-B
vendors in place).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.web_search.models import WebSearchSettings
from local_operator.web_search.native import google, kimi, mistral
from local_operator.web_search.native import openai as openai_vendor
from local_operator.web_search.native import openrouter, zai
from tests.unit.web_search.test_native_vendors import DEFAULT_FREE, FREE_MODES
from tests.unit.web_search.test_native_vendors import WebSearchService as _Service
from tests.unit.web_search.test_native_vendors import (
    _creds,
    _fixture,
    _key_bands,
    _modes,
    _run,
)

PART_B = ("google", "kimi", "zai", "mistral", "openrouter")


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_part_b_vendors_are_registered_as_provider_ids() -> None:
    from local_operator.web_defaults import WEB_SEARCH_PROVIDER_IDS
    from local_operator.web_search.models import PROVIDER_IDS
    from local_operator.web_search.providers import NATIVE_VENDORS, PROVIDERS

    for vendor_id in PART_B:
        assert vendor_id in PROVIDER_IDS
        assert vendor_id in WEB_SEARCH_PROVIDER_IDS  # settable from /settings
        assert vendor_id in NATIVE_VENDORS
        assert PROVIDERS[vendor_id].search is not None


# -- OpenRouter (live fixture) ----------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_records_the_billed_cost_and_its_search_item(monkeypatch) -> None:
    _creds(monkeypatch, keys={"openrouter"})
    payload = _fixture("openrouter_responses_live.json")
    response, seen = await _run("openrouter", lambda _r: httpx.Response(200, json=payload))
    assert str(seen[0].url) == openrouter.RESPONSES_URL
    assert seen[0].headers["authorization"] == "Bearer test-key"
    tool = _body(seen[0])["tools"][0]
    assert tool["type"] == "openrouter:web_search"
    assert tool["parameters"] == {"engine": "exa", "max_results": 5, "max_uses": 1}
    assert [s.url for s in response.sources][:1] == [
        "https://www.python.org/downloads/release/python-3152/"
    ]
    assert len(response.sources) >= 3
    # The count comes from usage.server_tool_use_details, not a guessed item type.
    assert response.usage is not None
    assert response.usage.server_searches == 1
    assert response.cost is not None
    assert response.cost.usd == pytest.approx(0.007238)
    assert response.cost.basis == openrouter.BASIS_BILLED and response.cost.priced_from_usage
    assert response.request_id == "gen-REDACTED"


@pytest.mark.asyncio
async def test_openrouter_without_billed_cost_falls_back_to_the_estimate(monkeypatch) -> None:
    _creds(monkeypatch, keys={"openrouter"})
    payload = _fixture("openrouter_responses_live.json")
    payload["usage"].pop("cost")
    response, _seen = await _run("openrouter", lambda _r: httpx.Response(200, json=payload))
    assert response.cost is not None
    assert response.cost.basis != openrouter.BASIS_BILLED
    assert response.cost.usd is not None
    assert response.cost.usd >= openrouter.FEE_USD


def test_openrouter_search_item_counts_and_carries_sources() -> None:
    from local_operator.web_search.native import responses_wire

    payload = _fixture("openrouter_responses_live.json")
    del payload["usage"]["server_tool_use_details"]
    # Without the usage count the ``openrouter:web_search`` item is the count.
    assert responses_wire.search_call_count(payload) == 1
    for item in payload["output"]:
        if item["type"] == "message":
            item["content"][0]["annotations"] = []
    sources, _answer = responses_wire.parse_responses_search(payload, 5, "OpenRouter")
    assert sources, "consulted pages on the openrouter:web_search item were dropped"


# -- Google ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_google_grounding_sends_the_key_in_a_header_and_prices_fee_plus_tokens(
    monkeypatch,
) -> None:
    _creds(monkeypatch, keys={"google"}, key="AIza-SECRET")
    payload = _fixture("google_grounding_documented.json")
    response, seen = await _run("google", lambda _r: httpx.Response(200, json=payload))
    assert str(seen[0].url) == f"{google.API_BASE}/{google.HOST_MODEL}:generateContent"
    assert "AIza-SECRET" not in str(seen[0].url)
    assert seen[0].headers["x-goog-api-key"] == "AIza-SECRET"
    assert _body(seen[0])["tools"] == [{"google_search": {}}]
    assert [s.title for s in response.sources] == ["python.org", "blog.python.org"]
    assert response.sources[0].snippet == "Python 3.15.2 is the latest stable release."
    # Tool-use prompt tokens are input; thinking tokens are output.
    assert response.usage is not None
    assert response.usage.input_tokens == 912 and response.usage.output_tokens == 100
    assert response.usage.server_searches == 1
    assert response.cost is not None
    assert response.cost.priced_from_usage
    assert response.cost.usd is not None
    assert response.cost.usd > google.FEE_USD
    assert google.BASIS_NOTE in response.cost.basis


@pytest.mark.asyncio
async def test_google_ungrounded_answer_is_a_leg_failure(monkeypatch) -> None:
    _creds(monkeypatch, keys={"google"})
    payload = {"candidates": [{"content": {"parts": [{"text": "From memory: 3.12"}]}}]}
    with pytest.raises(RuntimeError, match="no grounding sources"):
        await _run("google", lambda _r: httpx.Response(200, json=payload))


# -- Kimi --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kimi_key_leg_is_a_search_api_priced_per_call(monkeypatch) -> None:
    _creds(monkeypatch, keys={"kimi"})
    payload = _fixture("kimi_search_documented.json")
    response, seen = await _run("kimi", lambda _r: httpx.Response(200, json=payload))
    assert [str(r.url) for r in seen] == [kimi.KEY_SEARCH_URLS[0]]
    assert _body(seen[0]) == {
        "text_query": "latest stable Python release",
        "limit": 5,
        "enable_page_crawling": False,
        "timeout_seconds": 30,
    }
    assert response.auth_mode == "api-key" and response.answer is None
    assert response.sources[0].published_date == "2026-10-25"
    assert response.cost is not None
    assert response.cost.usd == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_kimi_key_retries_the_international_host_only_on_401(monkeypatch) -> None:
    _creds(monkeypatch, keys={"kimi"})
    payload = _fixture("kimi_search_documented.json")

    def by_host(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.moonshot.cn":
            return httpx.Response(401, json={"error": {"message": "Invalid Authentication"}})
        return httpx.Response(200, json=payload)

    response, seen = await _run("kimi", by_host)
    assert [r.url.host for r in seen] == ["api.moonshot.cn", "api.moonshot.ai"]
    assert response.sources

    with pytest.raises(RuntimeError, match="HTTP 429"):
        _response, seen = await _run("kimi", lambda _r: httpx.Response(429, text="slow down"))


@pytest.mark.asyncio
async def test_kimi_empty_results_fail_and_are_not_charged(monkeypatch) -> None:
    _creds(monkeypatch, keys={"kimi"})
    with pytest.raises(RuntimeError, match="no results"):
        await _run("kimi", lambda _r: httpx.Response(200, json={"search_results": []}))


def test_kimi_code_plan_leg_is_not_offered_until_probed(monkeypatch) -> None:
    _creds(monkeypatch, oauth={"kimi"})
    assert kimi.PLAN_LEG_PROBED is False
    assert kimi._modes(None, WebSearchSettings()) == []


@pytest.mark.asyncio
async def test_kimi_code_plan_leg_sends_the_grant_and_device_headers(monkeypatch) -> None:
    monkeypatch.setattr(kimi, "PLAN_LEG_PROBED", True)
    _creds(monkeypatch, keys={"kimi"}, oauth={"kimi"})
    payload = _fixture("kimi_search_documented.json")
    response, seen = await _run("kimi", lambda _r: httpx.Response(200, json=payload), mode="plan")
    assert str(seen[0].url) == kimi.PLAN_SEARCH_URL
    assert seen[0].headers["authorization"] == "Bearer test-grant"
    assert seen[0].headers["x-msh-platform"] == "kimi_cli"
    assert response.cost is not None
    assert response.auth_mode == "plan" and response.cost.usd is None


# -- Z.AI --------------------------------------------------------------------------


def test_zai_is_plan_class_with_either_credential(monkeypatch) -> None:
    _creds(monkeypatch, keys={"zai"})
    assert zai._modes(None, WebSearchSettings()) == [("plan", "plan")]
    _creds(monkeypatch, oauth={"zai"})
    assert zai._modes(None, WebSearchSettings()) == [("plan", "plan")]
    _creds(monkeypatch)
    assert zai._modes(None, WebSearchSettings()) == []


@pytest.mark.asyncio
async def test_zai_calls_web_search_prime_with_the_grant_first(monkeypatch) -> None:
    _creds(monkeypatch, keys={"zai"}, oauth={"zai"})
    payload = _fixture("zai_mcp_documented.json")
    response, seen = await _run("zai", lambda _r: httpx.Response(200, json=payload))
    assert str(seen[0].url) == zai.MCP_URL
    assert seen[0].headers["authorization"] == "Bearer test-grant"
    body = _body(seen[0])
    assert body["method"] == "tools/call"
    assert body["params"]["name"] == "webSearchPrime"
    assert body["params"]["arguments"]["search_query"] == "latest stable Python release"
    assert [s.url for s in response.sources] == [
        "https://www.python.org/downloads/",
        "https://www.python.org/downloads/release/python-3152/",
    ]
    assert response.cost is not None
    assert response.cost.usd is None and "GLM Coding Plan" in response.cost.basis


@pytest.mark.asyncio
async def test_zai_http_200_refusal_body_is_a_leg_failure(monkeypatch) -> None:
    _creds(monkeypatch, keys={"zai"})
    # Reproduced 2026-11-24 with no credential: HTTP 200, not a JSON-RPC envelope.
    refusal = {"code": 1001, "msg": "Authentication parameter not received", "success": False}
    with pytest.raises(RuntimeError, match="refused: 1001"):
        await _run("zai", lambda _r: httpx.Response(200, json=refusal))


# -- Mistral -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mistral_reads_tool_references_and_counts_connector_tokens(monkeypatch) -> None:
    _creds(monkeypatch, keys={"mistral"})
    payload = _fixture("mistral_conversations_documented.json")
    response, seen = await _run("mistral", lambda _r: httpx.Response(200, json=payload))
    assert str(seen[0].url) == mistral.CONVERSATIONS_URL
    body = _body(seen[0])
    assert body["tools"] == [{"type": "web_search"}] and body["store"] is False
    assert [s.title for s in response.sources] == [
        "Python Release Python 3.15.2",
        "Download Python",
    ]
    assert response.answer == "Python 3.15.2 is the latest stable release."
    assert response.usage is not None
    assert response.usage.input_tokens == 188 + 7112
    assert response.usage.server_searches == 1
    # The -latest alias is not in the price registry: fee only, stated as a floor.
    assert response.cost is not None
    assert response.cost.usd == pytest.approx(0.03) and "floor" in response.cost.basis


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", PART_B)
async def test_part_b_http_errors_carry_the_body_never_the_credential(monkeypatch, vendor) -> None:
    if vendor == "kimi":
        monkeypatch.setattr(kimi, "PLAN_LEG_PROBED", False)
    _creds(monkeypatch, keys={vendor}, key="sk-SECRET")
    with pytest.raises(RuntimeError) as caught:
        await _run(vendor, lambda _r: httpx.Response(403, json={"error": "forbidden"}))
    assert "HTTP 403" in str(caught.value) and "sk-SECRET" not in str(caught.value)


# -- the chain (design §6) -----------------------------------------------------------


def _chain(monkeypatch, tmp_path, *, radient, plan=(), model_keys=(), **cfg):
    from local_operator.web_search import radient as radient_module

    radient_module.reset_radient_credit_cache_for_tests()
    modes = dict(FREE_MODES)
    if radient:
        modes["radient"] = "radient-credits"
    _modes(monkeypatch, modes)
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", "openai" in plan)
    _creds(monkeypatch, keys=set(model_keys), oauth=set(plan))
    settings = WebSearchSettings(strategy="ordered", **cfg)
    return [leg.label for leg in _Service(settings, tmp_path).candidate_legs()]


#: Every part-B model-key vendor plus two part-A ones, cheapest-first:
#: kimi $0.002 < xai $0.005 < openrouter $0.007 < anthropic $0.01 < google $0.014
#: < mistral $0.03, whatever their registration order.
PAID = ("mistral", "google", "anthropic", "openrouter", "xai", "kimi")
PAID_ORDER = ["kimi", "xai", "openrouter", "anthropic", "google", "mistral"]


def test_row07b_zai_plan_leg_joins_native_before_radient(tmp_path, monkeypatch) -> None:
    chain = _chain(monkeypatch, tmp_path, radient=True, plan=("zai",))
    assert chain == [*DEFAULT_FREE, "zai", "radient"]


def test_row08b_model_key_legs_sort_cheapest_first_after_radient(tmp_path, monkeypatch) -> None:
    chain = _chain(monkeypatch, tmp_path, radient=True, model_keys=PAID)
    assert chain == [*DEFAULT_FREE, *_key_bands(["radient"], [], PAID_ORDER)]


def test_row09b_plans_then_radient_then_model_keys(tmp_path, monkeypatch) -> None:
    # A stored Z.AI KEY is still a plan leg: the key is the Coding Plan's own.
    chain = _chain(monkeypatch, tmp_path, radient=True, plan=("openai",), model_keys=("zai", *PAID))
    assert chain == [*DEFAULT_FREE, "openai", "zai", *_key_bands(["radient"], [], PAID_ORDER)]


def test_row12b_listing_a_paid_leg_first_never_lifts_it_past_radient(tmp_path, monkeypatch):
    chain = _chain(
        monkeypatch,
        tmp_path,
        radient=True,
        plan=("zai",),
        model_keys=("mistral", "kimi"),
        providers=["mistral", "radient", "zai"],
    )
    assert chain == [*DEFAULT_FREE, "zai", *_key_bands(["radient"], [], ["mistral", "kimi"])]

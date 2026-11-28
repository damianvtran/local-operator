"""The native-search family scaffold: shared wire parsers, pricing, registration.

Vendor transports have their own test files; this one pins the shared pieces
every vendor leans on, so a vendor slice can trust them without re-proving them.
"""

from __future__ import annotations

import json
import typing
from dataclasses import replace

import httpx
import pytest

from local_operator.i18n.messages import Msg
from local_operator.web_search.models import (
    SearchResponse,
    SearchUsage,
    WebSearchSettings,
)
from local_operator.web_search.native import anthropic_wire, pricing, responses_wire
from local_operator.web_search.native.base import (
    NativeClass,
    NativeSearchVendor,
    leg_mode,
)


def _responses_payload() -> dict[str, typing.Any]:
    return {
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "latest stable python",
                    "sources": [
                        {"type": "url", "url": "https://www.python.org/downloads/"},
                        {"type": "url", "url": "https://docs.python.org/3/whatsnew/"},
                    ],
                },
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Python 3.14 is the latest stable release.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://docs.python.org/3/whatsnew/",
                                "title": "What's New In Python",
                            },
                            {"type": "url_citation", "url": "javascript:alert(1)", "title": "x"},
                        ],
                    }
                ],
            },
        ],
        "usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 400},
            "output_tokens": 100,
        },
    }


def test_responses_parser_puts_cited_pages_first_and_merges_titles() -> None:
    sources, answer = responses_wire.parse_responses_search(_responses_payload(), 5, "OpenAI")
    assert [s.url for s in sources] == [
        "https://docs.python.org/3/whatsnew/",
        "https://www.python.org/downloads/",
    ]
    assert sources[0].title == "What's New In Python"
    # An untitled consulted page is titled by its URL, never left blank.
    assert sources[1].title == "https://www.python.org/downloads/"
    assert answer == "Python 3.14 is the latest stable release."


def test_responses_parser_honours_limit_and_xai_citations() -> None:
    payload = {"output": [], "citations": ["https://a.example/", "https://b.example/"]}
    sources, answer = responses_wire.parse_responses_search(payload, 1, "xAI")
    assert [s.url for s in sources] == ["https://a.example/"]
    assert answer is None


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ([], "non-object"),
        ({"error": {"message": "bad tool"}}, "bad tool"),
        (
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]},
            "no web_search results",
        ),
    ],
)
def test_responses_parser_raises_rather_than_returning_prose(payload, needle) -> None:
    with pytest.raises(RuntimeError, match=needle):
        responses_wire.parse_responses_search(payload, 5, "OpenAI")


def test_responses_usage_counts_vendor_reported_searches() -> None:
    usage = responses_wire.responses_usage(_responses_payload())
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (1000, 400, 100)
    assert usage.server_searches == 1
    alibaba = {"output": [], "tool_usage": {"web_search": {"num_requests": 3}}}
    assert responses_wire.search_call_count(alibaba) == 3
    xai = {"output": [], "usage": {"server_side_tool_usage_details": {"web_search_calls": 2}}}
    assert responses_wire.search_call_count(xai) == 2


@pytest.mark.asyncio
async def test_sse_collector_uses_streamed_items_when_completed_output_is_empty() -> None:
    item = _responses_payload()["output"][0]
    done = json.dumps({"type": "response.output_item.done", "item": item})
    body = (
        "event: response.output_item.done\n"
        f"data: {done}\n\n"
        ": keep-alive\n\n"
        'data: {"type":"response.completed",'
        '"response":{"output":[],"usage":{"input_tokens":5}}}\n\n'
    )
    response = httpx.Response(200, text=body)
    payload = await responses_wire.collect_sse_response(response, "ChatGPT")
    assert payload["output"] == [item]
    assert payload["usage"] == {"input_tokens": 5}


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ('data: {"type":"response.failed","response":{"error":{"message":"quota"}}}\n\n', "quota"),
        ('data: {"type":"error","message":"nope"}\n\n', "nope"),
        ('data: {"type":"response.created"}\n\n', "without a completed response"),
    ],
)
@pytest.mark.asyncio
async def test_sse_collector_raises_on_failure_and_truncation(body, needle) -> None:
    with pytest.raises(RuntimeError, match=needle):
        await responses_wire.collect_sse_response(httpx.Response(200, text=body), "ChatGPT")


def test_messages_parser_is_the_deepseek_parser_with_a_label() -> None:
    from local_operator.web_search.providers import parse_deepseek_search

    payload = {"content": [{"type": "text", "text": "prose only"}]}
    with pytest.raises(RuntimeError) as deepseek:
        parse_deepseek_search(payload, 5)
    # Byte-identical to the pre-move message: the wrapper only names the vendor.
    assert str(deepseek.value) == (
        "DeepSeek returned no web_search results; the request may not have "
        "triggered native search"
    )
    with pytest.raises(RuntimeError, match="^Anthropic returned no web_search"):
        anthropic_wire.parse_messages_search(payload, 5, "Anthropic")


def test_messages_tool_result_error_code_is_surfaced() -> None:
    payload = {
        "content": [
            {
                "type": "web_search_tool_result",
                "content": {"type": "x", "error_code": "unavailable"},
            }
        ]
    }
    assert anthropic_wire.tool_result_error(payload) == "unavailable"
    assert anthropic_wire.tool_result_error({"content": []}) is None


def test_native_cost_prices_fee_and_tokens_per_wire() -> None:
    usage = SearchUsage(input_tokens=1_000_000, cache_read_tokens=400_000, output_tokens=0)
    # gpt-4.1-mini: $0.40/M input; cache reads priced from the registry row.
    inside = pricing.native_cost(
        fee_usd=0.01, usage=usage, hosting="openai", model="gpt-4.1-mini", cache="inside"
    )
    apart = pricing.native_cost(
        fee_usd=0.01, usage=usage, hosting="openai", model="gpt-4.1-mini", cache="apart"
    )
    assert inside.priced_from_usage and apart.priced_from_usage
    # "inside" bills 600k fresh + 400k cached; "apart" bills 1M fresh + 400k cached.
    assert apart.usd is not None and inside.usd is not None
    assert apart.usd - inside.usd == pytest.approx(0.6 * 0.40 - 0.2 * 0.40 + 0.2 * 0.40, rel=0.5)
    assert apart.usd > inside.usd > 0.01


def test_native_cost_charges_the_fee_per_reported_search_and_never_below_it() -> None:
    usage = SearchUsage(input_tokens=-50, output_tokens=-50, server_searches=2)
    cost = pricing.native_cost(
        fee_usd=0.01, usage=usage, hosting="openai", model="gpt-4.1-mini", cache="inside"
    )
    assert cost.usd == pytest.approx(0.02)


def test_native_cost_unpriced_model_is_a_labelled_floor() -> None:
    cost = pricing.native_cost(
        fee_usd=0.005,
        usage=SearchUsage(input_tokens=10, output_tokens=10),
        hosting="openai",
        model="no-such-model-xyz",
        cache="inside",
    )
    assert cost.usd == 0.005
    assert not cost.priced_from_usage
    assert "floor" in cost.basis


def test_plan_cost_is_counted_not_charged() -> None:
    cost = pricing.plan_cost("ChatGPT plan")
    assert cost.usd is None and cost.basis == "included in ChatGPT plan"


def test_native_class_literal_matches_the_band_code() -> None:
    from local_operator.web_search import native
    from local_operator.web_search.native import base
    from local_operator.web_search.providers import NativeClass

    assert typing.get_args(base.NativeClass) == typing.get_args(NativeClass)
    assert native.NATIVE_SEARCH_VENDORS is not None


def test_leg_mode_serves_the_walked_leg_and_refuses_a_vanished_one() -> None:
    from local_operator.web_search.providers import LEG_MODE

    modes: list[tuple[str, NativeClass]] = [("plan", "plan"), ("api-key", "key")]
    assert leg_mode("OpenAI", modes) == "plan"  # outside a walk: strongest first
    plan_only: list[tuple[str, NativeClass]] = [("plan", "plan")]
    token = LEG_MODE.set("api-key")
    try:
        assert leg_mode("OpenAI", modes) == "api-key"
        with pytest.raises(RuntimeError, match="OpenAI search has no usable credential"):
            leg_mode("OpenAI", plan_only)
    finally:
        LEG_MODE.reset(token)


def test_registration_adds_a_vendor_to_providers_and_bands(monkeypatch) -> None:
    from local_operator.web_search import native, providers

    async def search(_c, _d, _s, _q, _l) -> SearchResponse:
        return SearchResponse(provider="deepseek", auth_mode="api-key")

    fake = NativeSearchVendor(
        id="deepseek",
        label="Fake",
        access=Msg("web_search.native.access_key", {}),
        detail=Msg("web_search.native.access_key", {}),
        modes=lambda _d, _s: [("api-key", "key")],
        search=search,
        fee_usd=0.5,
    )
    monkeypatch.setitem(native.NATIVE_SEARCH_VENDORS, "deepseek", fake)
    monkeypatch.setitem(providers.PROVIDERS, "deepseek", providers.PROVIDERS["deepseek"])
    monkeypatch.setitem(providers.NATIVE_VENDORS, "deepseek", providers.NATIVE_VENDORS["deepseek"])
    providers._register_native_search_vendors()
    assert providers.PROVIDERS["deepseek"].search is search
    assert providers.NATIVE_VENDORS["deepseek"].fee_usd == 0.5
    legs = providers.provider_legs("deepseek", None, WebSearchSettings())
    assert [(leg.mode, leg.cost_class) for leg in legs] == [("api-key", "paid")]

    bogus = replace(fake, id="not-a-provider")
    monkeypatch.setitem(native.NATIVE_SEARCH_VENDORS, "not-a-provider", bogus)
    with pytest.raises(RuntimeError, match="not a SearchProviderId"):
        providers._register_native_search_vendors()

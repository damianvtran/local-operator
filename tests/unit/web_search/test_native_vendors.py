"""Native-search vendors (openai, anthropic, xai, alibaba): transports, class gates, chain.

Transports run over ``httpx.MockTransport`` on fixtures under ``fixtures/native/``:
``anthropic_subscription_live.json`` and ``openai_plan_429_live.json`` are trimmed,
redacted LIVE captures (2026-11-24); the ``*_documented.json`` files are built
from the vendors' documented shapes because no credential for that leg existed
on the probe machine (each file says so in ``_note``).

Credential state is pinned at the ``credentials`` seam the vendor modules read
(``has_api_key``/``has_oauth_row`` for availability, ``api_key``/``oauth_access``
at call time), so no test touches a real store or network.

The chain rows mirror design §6 rows 7-9, 11-13 and 27 with the REAL vendors
in place of PR1's fakes; the OpenAI plan leg is enabled for them through
``PLAN_LEG_PROBED`` (it ships off until a live probe returns sources).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.providers.auth_store import OAuthAccess
from local_operator.web_search.models import WebSearchSettings
from local_operator.web_search.native import alibaba, anthropic, credentials
from local_operator.web_search.native import openai as openai_vendor
from local_operator.web_search.native import pricing, xai
from local_operator.web_search.providers import LEG_MODE, PROVIDERS
from local_operator.web_search.service import WebSearchService

FIXTURES = Path(__file__).parent / "fixtures" / "native"
QUERY = "latest stable Python release"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def _creds(monkeypatch, *, keys=(), oauth=(), key="test-key", grant_kind="oauth") -> None:
    """Pin which namespaces hold which credential class (sync AND async halves)."""
    monkeypatch.setattr(credentials, "has_api_key", lambda ns, _d, *_e: ns in keys)
    monkeypatch.setattr(credentials, "has_oauth_row", lambda ns: ns in oauth)

    async def api_key(ns, _d, *_e):
        return key if ns in keys else ""

    async def oauth_access(ns):
        if ns not in oauth or grant_kind != "oauth":
            # The real helper returns None for an api_key-kind cascade result.
            return None
        return OAuthAccess(access_token="test-grant", credential_id=1, account_id="acct-1")

    monkeypatch.setattr(credentials, "api_key", api_key)
    monkeypatch.setattr(credentials, "oauth_access", oauth_access)


async def _run(provider, handler, *, mode=None, settings=None):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    token = LEG_MODE.set(mode)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
            response = await PROVIDERS[provider].search(
                client, None, settings or WebSearchSettings(), QUERY, 5
            )
    finally:
        LEG_MODE.reset(token)
    return response, seen


def _sse(payload: dict[str, Any]) -> str:
    events = [{"type": "response.output_item.done", "item": item} for item in payload["output"]] + [
        {"type": "response.completed", "response": {**payload, "output": []}}
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


# -- pinned spellings -------------------------------------------------------------


def test_spellings_match_the_shipped_clients() -> None:
    from local_operator.imagegen import rungs_openai_sub
    from local_operator.providers.clients import AnthropicClient

    assert openai_vendor.CODEX_RESPONSES_URL == rungs_openai_sub.CODEX_RESPONSES_URL
    assert openai_vendor.CODEX_BETA_HEADER == rungs_openai_sub.CODEX_BETA_HEADER
    assert anthropic.CLAUDE_CODE_IDENTITY == AnthropicClient.CLAUDE_CODE_IDENTITY


def test_vendors_are_registered_as_provider_ids() -> None:
    from local_operator.web_search.models import PROVIDER_IDS
    from local_operator.web_search.providers import NATIVE_VENDORS

    for vendor_id in ("openai", "anthropic", "xai", "alibaba"):
        assert vendor_id in PROVIDER_IDS
        assert vendor_id in NATIVE_VENDORS
        assert PROVIDERS[vendor_id].search is not None


# -- OpenAI -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_key_leg_posts_responses_and_prices_fee_plus_tokens(monkeypatch) -> None:
    _creds(monkeypatch, keys={"openai"})
    payload = _fixture("openai_responses_documented.json")
    response, seen = await _run("openai", lambda _r: httpx.Response(200, json=payload))
    (request,) = seen
    body = json.loads(request.content)
    assert str(request.url) == openai_vendor.RESPONSES_URL
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["tools"] == [{"type": "web_search"}]
    assert body["include"] == ["web_search_call.action.sources"]
    assert body["model"] == openai_vendor.API_KEY_MODEL
    assert response.auth_mode == "api-key"
    # Cited page first, then the consulted one.
    assert [s.url for s in response.sources] == [
        "https://www.python.org/downloads/release/python-3150/",
        "https://www.python.org/downloads/",
    ]
    # 6,000 fresh + 2,000 cached input, 120 output at gpt-4.1-mini list price.
    assert response.usage is not None
    expected = pricing.native_cost(
        fee_usd=0.01, usage=response.usage, hosting="openai", model="gpt-4.1-mini", cache="inside"
    )
    assert response.cost is not None
    assert response.cost == expected and response.cost.priced_from_usage
    assert response.cost.usd is not None
    assert response.cost.usd > 0.01


def test_openai_plan_leg_is_not_offered_until_probed(monkeypatch) -> None:
    _creds(monkeypatch, oauth={"openai"})
    assert openai_vendor.PLAN_LEG_PROBED is False
    assert openai_vendor._modes(None, WebSearchSettings()) == []
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", True)
    assert openai_vendor._modes(None, WebSearchSettings()) == [("plan", "plan")]


@pytest.mark.asyncio
async def test_openai_plan_leg_streams_the_codex_backend(monkeypatch) -> None:
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", True)
    _creds(monkeypatch, oauth={"openai"})
    stream = _sse(_fixture("openai_responses_documented.json"))
    response, seen = await _run(
        "openai",
        lambda _r: httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"}),
        mode="plan",
    )
    (request,) = seen
    body = json.loads(request.content)
    assert str(request.url) == openai_vendor.CODEX_RESPONSES_URL
    assert request.headers["authorization"] == "Bearer test-grant"
    assert request.headers["chatgpt-account-id"] == "acct-1"
    assert request.headers["openai-beta"] == "responses=experimental"
    assert body["stream"] is True and body["store"] is False
    assert response.auth_mode == "plan" and len(response.sources) == 2
    assert response.cost is not None
    assert response.cost.usd is None and response.cost.basis == "included in ChatGPT plan"


@pytest.mark.asyncio
async def test_openai_plan_quota_429_is_a_leg_failure(monkeypatch) -> None:
    """The live 2026-11-24 probe's answer: the vendor sentence reaches the digest."""
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", True)
    _creds(monkeypatch, oauth={"openai"})
    body = _fixture("openai_plan_429_live.json")
    with pytest.raises(RuntimeError, match="HTTP 429.*usage_limit_reached"):
        await _run("openai", lambda _r: httpx.Response(429, json=body), mode="plan")


@pytest.mark.asyncio
async def test_openai_plan_leg_never_sends_an_api_key_kind_credential(monkeypatch) -> None:
    """``get_oauth_access`` can rotate to an api_key row; that must not reach chatgpt.com."""
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", True)
    _creds(monkeypatch, oauth={"openai"}, grant_kind="api_key")
    with pytest.raises(RuntimeError, match="no usable credential"):
        _response, seen = await _run("openai", lambda _r: httpx.Response(200), mode="plan")
    # No request was sent: the failure is raised before the POST.


@pytest.mark.asyncio
async def test_a_walked_key_leg_spends_the_key_even_when_a_plan_is_stored(monkeypatch) -> None:
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", True)
    _creds(monkeypatch, keys={"openai"}, oauth={"openai"})
    payload = _fixture("openai_responses_documented.json")
    _response, seen = await _run(
        "openai", lambda _r: httpx.Response(200, json=payload), mode="api-key"
    )
    assert str(seen[0].url) == openai_vendor.RESPONSES_URL


# -- Anthropic --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_key_leg_sends_x_api_key_and_no_claude_code_identity(monkeypatch) -> None:
    _creds(monkeypatch, keys={"anthropic"})
    payload = _fixture("anthropic_subscription_live.json")
    response, seen = await _run("anthropic", lambda _r: httpx.Response(200, json=payload))
    (request,) = seen
    body = json.loads(request.content)
    assert request.headers["x-api-key"] == "test-key"
    assert "authorization" not in request.headers
    assert "system" not in body
    assert body["tools"][0] == {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    assert response.auth_mode == "api-key" and len(response.sources) == 5
    assert response.usage is not None
    assert response.usage.server_searches == 1
    assert response.cost is not None
    assert "Claude extra usage" not in response.cost.basis


def test_claude_subscription_leg_is_off_by_default(monkeypatch) -> None:
    from local_operator.settings_io import BY_KEY

    _creds(monkeypatch, oauth={"anthropic"})
    assert anthropic._modes(None, WebSearchSettings()) == []
    on = WebSearchSettings(claude_subscription_search=True)
    # Opt-in, and still `key` class: ordered after Radient, never in the plan band.
    assert anthropic._modes(None, on) == [("subscription", "key")]
    setting = BY_KEY["web_search.claude_subscription_search"]
    assert setting.default is False and setting.path == (
        "web_search",
        "claude_subscription_search",
    )


@pytest.mark.asyncio
async def test_claude_subscription_leg_sends_the_oauth_shape(monkeypatch) -> None:
    _creds(monkeypatch, oauth={"anthropic"})
    payload = _fixture("anthropic_subscription_live.json")
    response, seen = await _run(
        "anthropic",
        lambda _r: httpx.Response(200, json=payload),
        mode="subscription",
        settings=WebSearchSettings(claude_subscription_search=True),
    )
    (request,) = seen
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer test-grant"
    assert request.headers["anthropic-beta"] == "oauth-2025-04-20"
    assert "x-api-key" not in request.headers
    assert body["system"][0]["text"] == anthropic.CLAUDE_CODE_IDENTITY
    assert response.auth_mode == "subscription"
    assert response.cost is not None
    assert response.cost.basis.endswith("API equivalent; drawn from Claude extra usage")
    # Live usage: 13,501 in / 669 out at claude-haiku-5-5 + one $0.01 search.
    assert response.cost.usd == pytest.approx(0.01 + 13_501 * 0.1e-6 + 669 * 0.5e-6, abs=1e-6)


@pytest.mark.asyncio
async def test_anthropic_refused_search_names_the_error_code(monkeypatch) -> None:
    _creds(monkeypatch, keys={"anthropic"})
    refused = {
        "content": [
            {
                "type": "web_search_tool_result",
                "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
            }
        ]
    }
    with pytest.raises(RuntimeError, match="web_search refused: unavailable"):
        await _run("anthropic", lambda _r: httpx.Response(200, json=refused))


# -- xAI / Alibaba ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_xai_reads_citations_and_bills_every_reported_search(monkeypatch) -> None:
    _creds(monkeypatch, keys={"xai"})
    payload = _fixture("xai_responses_documented.json")
    response, seen = await _run("xai", lambda _r: httpx.Response(200, json=payload))
    assert str(seen[0].url) == xai.RESPONSES_URL
    assert [s.url for s in response.sources][:2] == [
        "https://www.python.org/downloads/release/python-3150/",
        "https://www.python.org/downloads/",
    ]
    assert response.usage is not None
    assert response.usage.server_searches == 2
    tokens = 6_000 * 1.25e-6 + 2_000 * 0.2e-6 + 120 * 2.5e-6
    assert response.cost is not None
    assert response.cost.usd == pytest.approx(2 * 0.005 + tokens)


@pytest.mark.asyncio
async def test_xai_grant_serves_when_no_key_is_stored(monkeypatch) -> None:
    _creds(monkeypatch, oauth={"xai"})
    payload = _fixture("xai_responses_documented.json")
    response, seen = await _run("xai", lambda _r: httpx.Response(200, json=payload), mode="oauth")
    assert seen[0].headers["authorization"] == "Bearer test-grant"
    assert response.cost is not None
    assert "unpublished" in response.cost.basis


@pytest.mark.asyncio
async def test_alibaba_uses_billed_count_and_reports_an_unpriced_model_as_a_floor(
    monkeypatch,
) -> None:
    _creds(monkeypatch, keys={"alibaba"})
    payload = _fixture("alibaba_responses_documented.json")
    response, seen = await _run("alibaba", lambda _r: httpx.Response(200, json=payload))
    assert str(seen[0].url) == alibaba.RESPONSES_URL
    assert response.usage is not None
    assert response.usage.server_searches == 3
    assert response.cost is not None
    assert response.cost.usd == pytest.approx(0.03)
    assert response.cost.priced_from_usage is False and "floor" in response.cost.basis


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["openai", "anthropic", "xai", "alibaba"])
async def test_http_errors_carry_the_vendor_body_never_the_credential(monkeypatch, vendor) -> None:
    _creds(monkeypatch, keys={vendor}, key="sk-SECRET-VALUE")
    with pytest.raises(RuntimeError) as caught:
        await _run(vendor, lambda _r: httpx.Response(401, json={"error": "invalid key"}))
    assert "HTTP 401" in str(caught.value) and "sk-SECRET" not in str(caught.value)


# -- the chain with real vendors (design §6) --------------------------------------

from tests.unit.web_search.test_search_chain_matrix import (  # noqa: E402
    DEFAULT_FREE,
    FREE_MODES,
    _failing,
    _modes,
    _radient_service,
)


def _chain(monkeypatch, tmp_path, *, radient, keys=(), plan=False, model_keys=(), **cfg):
    from local_operator.web_search import radient as radient_module

    radient_module.reset_radient_credit_cache_for_tests()
    modes = dict(FREE_MODES)
    modes.update({k: "api-key" for k in keys})
    if radient:
        modes["radient"] = "radient-credits"
    _modes(monkeypatch, modes)
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", plan)
    _creds(monkeypatch, keys=set(model_keys), oauth={"openai"} if plan else set())
    settings = WebSearchSettings(strategy="ordered", **cfg)
    return [leg.label for leg in WebSearchService(settings, tmp_path).candidate_legs()]


def _key_bands(radient: list[str], search_keys: list[str], model_key: list[str]) -> list[str]:
    from local_operator.web_search.providers import NATIVE_MODEL_KEY_BAND

    if NATIVE_MODEL_KEY_BAND == "native":
        return [*model_key, *radient, *search_keys]
    return [*radient, *search_keys, *model_key]


#: xAI ($0.005) sorts ahead of Anthropic ($0.01) whatever the registration order.
MODEL_KEYS = ("anthropic", "xai")
MODEL_KEY_LEGS = ["xai", "anthropic"]


def test_row07_plan_leg_joins_native_before_radient(tmp_path, monkeypatch) -> None:
    assert _chain(monkeypatch, tmp_path, radient=False, plan=True) == [*DEFAULT_FREE, "openai"]


def test_row08_model_key_legs_sort_cheapest_first(tmp_path, monkeypatch) -> None:
    chain = _chain(monkeypatch, tmp_path, radient=False, model_keys=MODEL_KEYS)
    assert chain == [*DEFAULT_FREE, *_key_bands([], [], MODEL_KEY_LEGS)]


def test_row09_every_band_in_order(tmp_path, monkeypatch) -> None:
    chain = _chain(
        monkeypatch,
        tmp_path,
        radient=True,
        keys=("tavily", "brave", "exa"),
        plan=True,
        model_keys=MODEL_KEYS,
    )
    search_keys = ["tavily (key)", "exa (key)", "brave"]
    assert chain == [
        *DEFAULT_FREE,
        "openai",
        *_key_bands(["radient"], search_keys, MODEL_KEY_LEGS),
    ]


def test_row11_listed_search_key_heads_api_keys_not_the_chain(tmp_path, monkeypatch) -> None:
    chain = _chain(
        monkeypatch,
        tmp_path,
        radient=True,
        keys=("brave",),
        model_keys=MODEL_KEYS,
        providers=["brave", "duckduckgo"],
    )
    assert chain == [*DEFAULT_FREE, *_key_bands(["radient"], ["brave"], MODEL_KEY_LEGS)]


def test_row12_listing_reorders_within_a_band_only(tmp_path, monkeypatch) -> None:
    chain = _chain(monkeypatch, tmp_path, radient=True, plan=True, providers=["radient", "openai"])
    assert chain == [*DEFAULT_FREE, "openai", "radient"]


def test_row13_excluded_radient(tmp_path, monkeypatch) -> None:
    chain = _chain(
        monkeypatch,
        tmp_path,
        radient=True,
        keys=("tavily",),
        model_keys=MODEL_KEYS,
        excluded_providers=["radient"],
    )
    assert chain == [*DEFAULT_FREE, *_key_bands([], ["tavily (key)"], MODEL_KEY_LEGS)]


def test_listing_a_native_vendor_never_moves_it_across_a_band(tmp_path, monkeypatch) -> None:
    chain = _chain(
        monkeypatch, tmp_path, radient=True, model_keys=("xai",), providers=["xai", "duckduckgo"]
    )
    # A listed model-key leg heads ITS band (wherever the policy constant puts
    # it) and never jumps to the free prefix ahead of the free legs.
    assert chain == [*DEFAULT_FREE, *_key_bands(["radient"], [], ["xai"])]


@pytest.mark.asyncio
async def test_row27_thirteen_real_legs_fail_and_every_one_is_named(tmp_path, monkeypatch):
    from rich.cells import cell_len

    from local_operator.web_search import providers as providers_module
    from local_operator.web_search import service as service_module

    def vendor_down(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "chatgpt.com":
            return httpx.Response(429, json=_fixture("openai_plan_429_live.json"))
        if host == "api.anthropic.com":
            return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded"}})
        if host == "api.x.ai":
            return httpx.Response(503, text="upstream unavailable")
        if host.startswith("dashscope"):
            return httpx.Response(401, json={"code": "InvalidApiKey"})
        return httpx.Response(503, text="All connection attempts failed")

    keys = {"tavily": "api-key", "brave": "api-key", "exa": "api-key"}
    service, calls = _radient_service(tmp_path, monkeypatch, vendor_down, keys=keys)
    monkeypatch.setitem(
        providers_module.PROVIDERS,
        "brave",
        replace(providers_module.PROVIDERS["brave"], search=_failing("brave")),
    )
    monkeypatch.setattr(openai_vendor, "PLAN_LEG_PROBED", True)
    _creds(monkeypatch, keys={"anthropic", "xai", "alibaba"}, oauth={"openai"})
    candidates = service.candidate_legs()
    legs = [leg.label for leg in candidates]
    assert len(legs) == 13, legs
    # The header counts PROVIDERS, not legs (PR #4553 R1-7): tavily and exa are
    # walked twice (free leg, then key), so 13 legs are 11 providers. The tried
    # list below must still name all 13 legs.
    providers = {leg.provider for leg in candidates}
    assert len(providers) == 11, providers
    with pytest.raises(RuntimeError) as caught:
        await service.search("q")
    message = str(caught.value)
    assert message.startswith("11/11 providers failed: ")
    for label in legs:
        assert f"{label}: " in message
    assert cell_len(message) <= service_module.REASON_MAX_CELLS
    # Every real native transport made exactly one request (the radient leg one more).
    hosts = sorted(request.url.host for request in calls)
    assert hosts.count("chatgpt.com") == 1
    assert hosts.count("api.anthropic.com") == 1
    assert hosts.count("api.x.ai") == 1

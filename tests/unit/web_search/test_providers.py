from __future__ import annotations

import json

import httpx
import pytest

from local_operator.credentials import CredentialManager
from local_operator.web_search.models import WebSearchSettings
from local_operator.web_search.providers import PROVIDERS, parse_duckduckgo_html


def _credentials(tmp_path) -> CredentialManager:
    return CredentialManager(tmp_path / "config")


def test_duckduckgo_parser_unwraps_links_and_inline_markup() -> None:
    page = (
        '<div class="result results_links">\n'
        '  <h2><a class="result__a" href="//duckduckgo.com/l/?uddg='
        'https%3A%2F%2Fexample.com%2Fdocument%252Fversion">Example <b>Doc</b></a></h2>\n'
        '  <a class="result__snippet">Useful &amp; current.</a>\n'
        "</div>\n"
        '<div class="nav-link"></div>'
    )

    rows = parse_duckduckgo_html(page, 5)

    assert len(rows) == 1
    assert rows[0].title == "Example Doc"
    assert rows[0].url == "https://example.com/document%2Fversion"
    assert rows[0].snippet == "Useful & current."


@pytest.mark.asyncio
async def test_tavily_uses_official_keyless_header_when_no_key_exists(tmp_path) -> None:
    seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = request
        return httpx.Response(
            200,
            json={
                "answer": "Current answer",
                "request_id": "req-1",
                "results": [
                    {"title": "Result", "url": "https://example.com", "content": "Snippet"}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["tavily"].search(
            client,
            _credentials(tmp_path),
            WebSearchSettings(),
            "current fact",
            3,
        )

    assert seen is not None
    assert seen.headers["X-Tavily-Access-Mode"] == "keyless"
    assert "Authorization" not in seen.headers
    assert json.loads(seen.content)["max_results"] == 3
    assert response.auth_mode == "keyless"
    assert response.sources[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_tavily_prefers_stored_key_over_keyless_mode(tmp_path) -> None:
    credentials = _credentials(tmp_path)
    credentials.set_credential("TAVILY_API_KEY", "secret-key")
    seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = request
        return httpx.Response(200, json={"results": [{"url": "https://example.com"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["tavily"].search(
            client, credentials, WebSearchSettings(), "query", 1
        )

    assert seen is not None
    assert seen.headers["Authorization"] == "Bearer secret-key"
    assert "X-Tavily-Access-Mode" not in seen.headers
    assert response.auth_mode == "api-key"


@pytest.mark.asyncio
async def test_serpapi_accepts_legacy_serp_api_key_name(tmp_path) -> None:
    credentials = _credentials(tmp_path)
    credentials.set_credential("SERP_API_KEY", "legacy-key")
    seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = request
        return httpx.Response(
            200,
            json={
                "search_metadata": {"id": "search-1"},
                "organic_results": [
                    {"title": "Result", "link": "https://example.com", "snippet": "Body"}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["serpapi"].search(
            client, credentials, WebSearchSettings(), "query", 2
        )

    assert seen is not None
    assert seen.url.params["api_key"] == "legacy-key"
    assert response.request_id == "search-1"
    assert response.sources[0].title == "Result"


@pytest.mark.asyncio
async def test_brave_uses_subscription_token_and_maps_extra_snippets(tmp_path) -> None:
    credentials = _credentials(tmp_path)
    credentials.set_credential("BRAVE_API_KEY", "brave-key")
    seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = request
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave result",
                            "url": "https://example.com/brave",
                            "description": "Primary",
                            "extra_snippets": ["Extra"],
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["brave"].search(
            client, credentials, WebSearchSettings(), "query", 2
        )

    assert seen is not None
    assert seen.headers["X-Subscription-Token"] == "brave-key"
    assert seen.url.params["count"] == "2"
    assert response.sources[0].snippet == "Primary\nExtra"


@pytest.mark.asyncio
async def test_exa_requests_query_summary_instead_of_full_page_text(tmp_path) -> None:
    credentials = _credentials(tmp_path)
    credentials.set_credential("EXA_API_KEY", "exa-key")
    seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = request
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Exa result",
                        "url": "https://example.com/exa",
                        "summary": "Query-grounded summary",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["exa"].search(
            client, credentials, WebSearchSettings(), "semantic query", 4
        )

    assert seen is not None
    payload = json.loads(seen.content)
    assert payload["contents"] == {"summary": {"query": "semantic query"}}
    assert "text" not in payload["contents"]
    assert response.sources[0].snippet == "Query-grounded summary"


@pytest.mark.asyncio
async def test_searxng_uses_configured_endpoint_and_json_contract(tmp_path) -> None:
    seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = request
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Private result",
                        "url": "https://example.com/private",
                        "content": "Private snippet",
                    }
                ]
            },
        )

    settings = WebSearchSettings(searxng_endpoint="https://search.example.com")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["searxng"].search(
            client, _credentials(tmp_path), settings, "query", 3
        )

    assert seen is not None
    assert str(seen.url).startswith("https://search.example.com/search?")
    assert seen.url.params["format"] == "json"
    assert response.sources[0].snippet == "Private snippet"


@pytest.mark.asyncio
async def test_perplexity_anonymous_sse_yields_answer_and_sources(tmp_path) -> None:
    source_event = {
        "uuid": "pplx-1",
        "blocks": [
            {
                "intended_usage": "web_results",
                "web_result_block": {
                    "web_results": [
                        {
                            "name": "Source",
                            "url": "https://example.com",
                            "snippet": "Evidence",
                        }
                    ]
                },
            }
        ],
    }
    answer_event = {
        "uuid": "pplx-1",
        "blocks": [
            {
                "intended_usage": "ask_text",
                "markdown_block": {"answer": "Grounded answer"},
            }
        ],
        "final": True,
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        body = (
            "data: "
            + json.dumps(source_event)
            + "\n\ndata: "
            + json.dumps(answer_event)
            + "\n\ndata: [DONE]\n"
        )
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["perplexity"].search(
            client,
            _credentials(tmp_path),
            WebSearchSettings(),
            "query",
            3,
        )

    assert response.auth_mode == "anonymous"
    assert response.answer == "Grounded answer"
    assert response.sources[0].url == "https://example.com"


# ---------------------------------------------------------------------------
# DeepSeek native search (Anthropic-format Messages API)
# ---------------------------------------------------------------------------


def _deepseek_payload(
    *,
    items: list[dict] | None = None,
    answer: str = "Synthesized answer",
    citations: list[dict] | None = None,
) -> dict:
    """One Messages response in the shape DeepSeek actually returns."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-v4-flash",
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "…", "signature": "sig"},
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srv_1",
                "content": items
                if items is not None
                else [
                    {"type": "web_search_result", "url": "https://example.com/a", "title": "A"},
                    {"type": "web_search_result", "url": "https://example.com/b", "title": "B"},
                ],
            },
            {"type": "text", "text": answer, "citations": citations},
        ],
        "usage": {
            "input_tokens": 14_521,
            "output_tokens": 863,
            "server_tool_use": {"web_search_requests": 1},
        },
    }


def test_deepseek_normalizer_maps_sources_joins_citations_and_carries_answer() -> None:
    from local_operator.web_search.providers import parse_deepseek_search

    sources, answer = parse_deepseek_search(
        _deepseek_payload(
            items=[
                {"type": "web_search_result", "url": "https://example.com/a", "title": "A"},
                # Same URL twice: a `max_uses > 1` request can surface one page
                # from more than one server-side search.
                {"type": "web_search_result", "url": "https://example.com/a", "title": "A again"},
                {"type": "web_search_result", "url": "https://example.com/b", "title": "B",
                 "page_age": "2026-08-05"},
            ],
            citations=[{"type": "web_search_result_location", "url": "https://example.com/a",
                        "cited_text": "Cited excerpt"}],
        ),
        5,
    )

    assert [s.url for s in sources] == ["https://example.com/a", "https://example.com/b"]
    assert sources[0].snippet == "Cited excerpt"
    assert sources[1].published_date == "2026-08-05"
    assert answer == "Synthesized answer"


def test_deepseek_normalizer_honours_limit_and_raises_without_results() -> None:
    from local_operator.web_search.providers import parse_deepseek_search

    sources, _ = parse_deepseek_search(_deepseek_payload(), 1)
    assert len(sources) == 1

    # Prose with no result block is NOT a search result. Returning the answer as
    # success would stop the fallback chain on nothing actionable -- the defect
    # the anonymous Perplexity transport has.
    payload = _deepseek_payload(items=None, answer="I did not search")
    payload["content"] = [{"type": "text", "text": "I did not search"}]
    with pytest.raises(RuntimeError):
        parse_deepseek_search(payload, 5)

    with pytest.raises(RuntimeError):
        parse_deepseek_search(_deepseek_payload(items=[]), 5)


def test_deepseek_availability_requires_env_key_or_login(tmp_path, monkeypatch) -> None:
    from local_operator.web_search import providers as module

    monkeypatch.setattr(module, "_deepseek_login_present", lambda: False)
    credentials = _credentials(tmp_path)
    assert module.provider_available("deepseek", credentials, WebSearchSettings()) is False

    monkeypatch.setattr(module, "_deepseek_login_present", lambda: True)
    assert module.provider_available("deepseek", credentials, WebSearchSettings()) is True

    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")
    assert module.provider_available("deepseek", credentials, WebSearchSettings()) is True


@pytest.mark.asyncio
async def test_deepseek_transport_uses_server_tool_and_primes_balance_off_path(
    tmp_path, monkeypatch
) -> None:
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()
    primes: list[str] = []
    monkeypatch.setattr(module, "_spawn_deepseek_balance_refresh", lambda key: primes.append(key))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_deepseek_payload())

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["deepseek"].search(
            client, credentials, WebSearchSettings(), "latest python", 5
        )

    # ONE request: the balance probe is a background refresh for the next search,
    # never an extra round trip on this one's path.
    assert len(requests) == 1
    assert requests[0].url.path == "/anthropic/v1/messages"
    assert primes == ["sk-test-not-real"]
    body = json.loads(requests[0].content)
    assert body["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    ]
    assert body["max_tokens"] == 1_024
    assert body["messages"][0]["content"][0]["text"] == (
        "Perform a web search for the query: latest python"
    )
    assert requests[0].headers["x-api-key"] == "sk-test-not-real"
    assert response.provider == "deepseek"
    assert response.auth_mode == "api-key"
    assert len(response.sources) == 2
    assert response.answer == "Synthesized answer"


@pytest.mark.asyncio
async def test_deepseek_balance_probe_parses_the_account_balance() -> None:
    from local_operator.web_search import providers as module

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/user/balance"
        assert request.headers["authorization"] == "Bearer sk-test-not-real"
        return httpx.Response(
            200,
            json={
                "is_available": True,
                "balance_infos": [{"currency": "USD", "total_balance": "28.48"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await module._deepseek_balance_ok(client, "sk-test-not-real") is True

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200,
                json={
                    "is_available": True,
                    "balance_infos": [{"currency": "USD", "total_balance": "0.10"}],
                },
            )
        )
    ) as client:
        assert await module._deepseek_balance_ok(client, "sk") is False

    # A balance endpoint that cannot be read must NOT remove a paid transport.
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(503))
    ) as client:
        assert await module._deepseek_balance_ok(client, "sk") is True


@pytest.mark.asyncio
async def test_deepseek_transport_skips_on_a_cached_low_balance_verdict(tmp_path) -> None:
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()
    module._remember_deepseek_balance(False)

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")

    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no search may be dispatched for an unfunded account")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="balance is below"):
            await PROVIDERS["deepseek"].search(
                client, credentials, WebSearchSettings(), "query", 5
            )

    module.reset_deepseek_balance_cache_for_tests()


@pytest.mark.asyncio
async def test_deepseek_transport_requires_a_key(tmp_path, monkeypatch) -> None:
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()

    async def no_key(_credentials) -> str:
        return ""

    monkeypatch.setattr(module, "_resolve_deepseek_key", no_key)
    credentials = _credentials(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request may leave without a credential")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="login deepseek"):
            await PROVIDERS["deepseek"].search(
                client, credentials, WebSearchSettings(), "query", 5
            )

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

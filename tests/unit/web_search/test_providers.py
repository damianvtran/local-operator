from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from local_operator.credentials import CredentialManager
from local_operator.web_search.models import (
    PROVIDER_IDS,
    SearchProviderId,
    SearchResponse,
    SearchSource,
    WebSearchSettings,
)
from local_operator.web_search.providers import (
    PROVIDERS,
    parse_duckduckgo_html,
    parse_exa_mcp_text,
    parse_parallel_mcp_text,
)


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
    items: list[dict[str, Any]] | None = None,
    answer: str = "Synthesized answer",
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
                "content": (
                    items
                    if items is not None
                    else [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/a",
                            "title": "A",
                            # DeepSeek always returns this opaque field; the evidence
                            # pass depends on replaying it untouched.
                            "encrypted_content": "opaque-page-content",
                        },
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/b",
                            "title": "B",
                            "encrypted_content": "opaque-page-content-b",
                        },
                    ]
                ),
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
                {
                    "type": "web_search_result",
                    "url": "https://example.com/b",
                    "title": "B",
                    "page_age": "2026-08-05",
                },
            ],
            citations=[
                {
                    "type": "web_search_result_location",
                    "url": "https://example.com/a",
                    "cited_text": "Cited excerpt",
                }
            ],
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
    assert body["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}]
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
            await PROVIDERS["deepseek"].search(client, credentials, WebSearchSettings(), "query", 5)

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
            await PROVIDERS["deepseek"].search(client, credentials, WebSearchSettings(), "query", 5)


# ---------------------------------------------------------------------------
# DeepSeek per-page evidence pass (the snippet source for this provider)
# ---------------------------------------------------------------------------


def _deepseek_blocks_payload() -> dict[str, Any]:
    return _deepseek_payload()


def test_deepseek_evidence_parser_skips_unparseable_lines() -> None:
    from local_operator.web_search.providers import parse_deepseek_evidence

    rows = parse_deepseek_evidence(
        "```json\n"
        '{"url": "https://example.com/a", "relevance": 92, "summary": "About A",'
        ' "quote": "Verbatim A"}\n'
        '{"url": "https://example.com/b", "relevance": 88, "quote": "Verbatim B"}\n'
        "not json at all\n"
        '{"url": "https://example.com/a", "relevance": 10}\n'
    )

    # First row for a URL wins, and a broken line costs only that line.
    assert set(rows) == {"https://example.com/a", "https://example.com/b"}
    assert rows["https://example.com/a"]["relevance"] == 92


def test_deepseek_evidence_merge_fills_snippets_and_ranks() -> None:
    from local_operator.web_search.models import SearchSource
    from local_operator.web_search.providers import _apply_deepseek_evidence

    sources = [
        SearchSource(title="A", url="https://example.com/a"),
        SearchSource(title="B", url="https://example.com/b"),
        SearchSource(title="C", url="https://example.com/c"),
    ]
    merged, applied = _apply_deepseek_evidence(
        sources,
        {
            "https://example.com/b": {"relevance": 95, "quote": "B is the relevant one"},
            "https://example.com/a": {"relevance": 40, "summary": "A is tangential"},
        },
    )

    # The pass supplied snippets, so the footer that follows this flag must be
    # the model-reported one, not the "page text" one.
    assert applied is True

    # Ranked by relevance, and the uncovered source is kept at the end rather
    # than dropped: the pass is a top-N view, not a verdict on the rest.
    assert [s.url for s in merged] == [
        "https://example.com/b",
        "https://example.com/a",
        "https://example.com/c",
    ]
    assert merged[0].snippet == "B is the relevant one"
    assert merged[1].snippet == "A is tangential"  # summary is the fallback
    assert merged[0].relevance == 95
    assert merged[2].relevance is None


@pytest.mark.asyncio
async def test_deepseek_evidence_pass_runs_a_second_turn_and_replays_blocks(tmp_path) -> None:
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json=_deepseek_blocks_payload())
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '{"url": "https://example.com/b", "relevance": 91, '
                            '"summary": "About B", "quote": "Verbatim from B"}\n'
                        ),
                    }
                ],
                "usage": {"input_tokens": 240, "output_tokens": 300},
            },
        )

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["deepseek"].search(
            client,
            credentials,
            WebSearchSettings(deepseek_evidence=True),
            "latest python",
            5,
        )

    assert len(seen) == 2
    replay = seen[1]["messages"]
    assert replay[0]["content"][0]["text"] == seen[0]["messages"][0]["content"][0]["text"]
    # The assistant turn is replayed verbatim -- the opaque encrypted_content is
    # what makes DeepSeek restore the page text, so it must survive the trip.
    replayed_items = [
        item
        for block in replay[1]["content"]
        if block["type"] == "web_search_tool_result"
        for item in block["content"]
    ]
    assert replayed_items[0]["encrypted_content"] == "opaque-page-content"
    # The search turn was trimmed: the triage turn writes the descriptive text.
    assert seen[0]["max_tokens"] == module.DEEPSEEK_SEARCH_ANSWER_MAX_TOKENS
    assert response.sources[0].url == "https://example.com/b"
    assert response.sources[0].snippet == "Verbatim from B"
    assert response.sources[0].relevance == 91


@pytest.mark.asyncio
async def test_deepseek_evidence_failure_leaves_the_sources_intact(tmp_path) -> None:
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()

    def handler(request: httpx.Request) -> httpx.Response:
        if len(handler.calls) == 0:  # type: ignore[attr-defined]
            handler.calls.append(1)  # type: ignore[attr-defined]
            return httpx.Response(200, json=_deepseek_blocks_payload())
        return httpx.Response(500, json={"error": {"message": "evidence boom"}})

    handler.calls = []  # type: ignore[attr-defined]
    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["deepseek"].search(
            client,
            credentials,
            WebSearchSettings(deepseek_evidence=True),
            "latest python",
            5,
        )

    assert len(response.sources) == 2
    assert all(source.snippet is None for source in response.sources)


@pytest.mark.asyncio
async def test_deepseek_evidence_is_off_by_default(tmp_path) -> None:
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_deepseek_payload())

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await PROVIDERS["deepseek"].search(
            client, credentials, WebSearchSettings(), "latest python", 5
        )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_deepseek_evidence_failure_is_reported_not_swallowed(tmp_path) -> None:
    """A failed evidence pass must say so, while keeping the search usable."""
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()

    def handler(request: httpx.Request) -> httpx.Response:
        if len(handler.calls) == 0:  # type: ignore[attr-defined]
            handler.calls.append(1)  # type: ignore[attr-defined]
            return httpx.Response(200, json=_deepseek_payload())
        return httpx.Response(500, json={"error": {"message": "evidence boom"}})

    handler.calls = []  # type: ignore[attr-defined]
    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["deepseek"].search(
            client,
            credentials,
            WebSearchSettings(deepseek_evidence=True),
            "latest python",
            5,
        )

    assert len(response.sources) == 2
    assert any("evidence pass" in note for note in response.failures)


@pytest.mark.asyncio
async def test_a_truncated_evidence_pass_is_reported_not_passed_off_as_complete(
    tmp_path,
) -> None:
    """A capped pass must say so, not silently drop the pages it never reached.

    ``evidence_failure`` used to be set only when zero rows came back, so a
    payload cut off at ``DEEPSEEK_EVIDENCE_MAX_TOKENS`` looked exactly like a
    complete one as long as the first row parsed -- and the rows it lost are the
    later, lower-ranked pages the pass exists to triage. ``stop_reason`` is in
    the same payload and was unused.
    """
    from local_operator.web_search import providers as module

    module.reset_deepseek_balance_cache_for_tests()
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json=_deepseek_blocks_payload())
        return httpx.Response(
            200,
            json={
                "stop_reason": "max_tokens",
                "content": [
                    {
                        "type": "text",
                        "text": '{"url": "https://example.com/b", "relevance": 91, "quote": "B"}\n',
                    }
                ],
                "usage": {"input_tokens": 240, "output_tokens": 300},
            },
        )

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "sk-test-not-real")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["deepseek"].search(
            client,
            credentials,
            WebSearchSettings(deepseek_evidence=True),
            "latest python",
            5,
        )

    # The rows it DID parse are applied -- the enrichment is partial, not lost.
    assert response.sources[0].url == "https://example.com/b"
    assert response.evidence_applied is True
    # ...and the truncation is on the record rather than invisible.
    assert any("token cap" in failure for failure in response.failures), response.failures


@pytest.mark.asyncio
async def test_perplexity_anonymous_wall_is_a_refusal_not_a_result(tmp_path) -> None:
    """A sign-in wall must not be served as the search the model asked for.

    The anonymous endpoint refuses with a normally completed stream: HTTP 200,
    ``status: COMPLETED``, and ``text`` asking the reader to sign up. Because the
    chain accepts an empty-source response when its answer is non-empty (a
    provider may legitimately answer without citations), that sentence was
    returned as a successful search -- observed five times in one real session,
    each one a search that returned nothing while looking like one that worked.
    The refusal is marked structurally by ``upsell_information``.
    """
    wall_event = {
        "uuid": "pplx-wall",
        "status": "COMPLETED",
        "final": True,
        "text": "Sign up and repeat your request.",
        "upsell_information": {
            "name": "fraud_authwall_upsell",
            "upsell_type": "LOGIN",
            "cta": "SIGN_UP_OR_LOGIN",
        },
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        body = "data: " + json.dumps(wall_event) + "\n\ndata: [DONE]\n"
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError) as caught:
            await PROVIDERS["perplexity"].search(
                client,
                _credentials(tmp_path),
                WebSearchSettings(),
                "query",
                3,
            )

    message = str(caught.value)
    # The reason names the wall and says what to do about it.
    assert "fraud_authwall_upsell" in message
    assert "PERPLEXITY_API_KEY" in message or "another provider" in message
    # ...and never the invitation itself, which is what used to be returned.
    assert "Sign up and repeat your request." not in message


@pytest.mark.asyncio
async def test_perplexity_keeps_sources_returned_alongside_a_wall(tmp_path) -> None:
    """Some pages plus an upsell is still a result: never discard real sources."""
    source_event = {
        "uuid": "pplx-2",
        "blocks": [
            {
                "intended_usage": "web_results",
                "web_result_block": {
                    "web_results": [
                        {"name": "Source", "url": "https://example.com", "snippet": "E"}
                    ]
                },
            }
        ],
        "upsell_information": {"name": "fraud_authwall_upsell", "upsell_type": "LOGIN"},
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        body = "data: " + json.dumps(source_event) + "\n\ndata: [DONE]\n"
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["perplexity"].search(
            client,
            _credentials(tmp_path),
            WebSearchSettings(),
            "query",
            3,
        )

    assert response.sources[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_a_chain_of_empty_providers_fails_loudly(tmp_path, monkeypatch) -> None:
    """When every enabled provider comes back empty, the search says so.

    The alternative -- returning the last provider's empty response -- is what
    made a real session's five empty searches look successful.
    """
    from local_operator.web_search import service as service_module
    from local_operator.web_search.service import WebSearchService

    def empty(*_args, **_kwargs):
        return SearchResponse(provider="duckduckgo", auth_mode="free", sources=[], answer=None)

    def walled(*_args, **_kwargs):
        raise RuntimeError("anonymous tier walled this request (fraud_authwall_upsell/LOGIN)")

    class _Table(dict[str, Any]):
        def __getitem__(self, key):
            return _Entry(empty if key == "duckduckgo" else walled)

    class _Entry:
        def __init__(self, fn):
            self.search = fn

    monkeypatch.setattr(service_module, "PROVIDERS", _Table())
    # Pin the chain: a bare providers=[...] now auto-joins every other usable
    # provider, and the fake table would answer for those too -- turning a
    # two-leg assertion into a six-leg one by accident.
    settings = WebSearchSettings(
        providers=["duckduckgo", "perplexity"],
        excluded_providers=[
            value for value in PROVIDER_IDS if value not in ("duckduckgo", "perplexity")
        ],
    )
    service = WebSearchService(settings, _credentials(tmp_path))

    with pytest.raises(RuntimeError) as caught:
        await service.search("query")

    message = str(caught.value)
    assert "duckduckgo" in message and "perplexity" in message


# ---------------------------------------------------------------------------
# Refusal shape matrix (round-1 review: MAJOR-1 and MINOR-1)
# ---------------------------------------------------------------------------

#: The in-thread sign-in NUDGE. Live probes of the anonymous endpoint found this
#: on refusals too, so it must keep raising; it is not, on its own, proof that
#: nothing was served -- that is what the served-block test settles.
SOFT_UPSELL = {
    "name": "logged_out_thread_sign_in",
    "upsell_type": "LOGIN",
    "app_location": "IN_THREAD_INPUT",
    "title": "Sign in to save your history and access more features",
}
WALL_UPSELL = {
    "name": "fraud_authwall_upsell",
    "upsell_type": "LOGIN",
    "app_location": "MODAL",
    "title": "Sign in to continue using Perplexity",
}
_ASK_ONLY_BLOCKS = [
    {
        "intended_usage": "ask_text",
        "markdown_block": {"answer": "Sign up and repeat your request."},
    }
]


def _sse(*events: dict[str, object]) -> str:
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n"


def _wall(body: str) -> str | None:
    from local_operator.web_search.providers import (
        _parse_perplexity_sse,
        _perplexity_authwall,
    )

    return _perplexity_authwall(_parse_perplexity_sse(body))


def test_refusal_carrying_only_the_soft_marker_is_still_a_refusal() -> None:
    """Three live refusals carried ONLY this marker, so it has to raise.

    Narrowing the test to ``fraud_authwall_upsell`` would reopen the original
    bug for that shape (round-1 review MAJOR-1).
    """
    body = _sse({"upsell_information": SOFT_UPSELL, "blocks": _ASK_ONLY_BLOCKS})
    assert _wall(body) is not None


def test_a_served_block_beside_a_soft_marker_is_a_result_not_a_refusal() -> None:
    """MAJOR-1: "no recognised sources" is not "nothing was served".

    A shopping, hotels, maps or media answer is served content that the source
    extractor has no rows for. Raising on its absence discarded a real answer
    that happened to carry a sign-in nudge.
    """
    for block_key in (
        "shopping_block",
        "hotels_mode_block",
        "maps_mode_block",
        "media_block",
        "web_result_block",
    ):
        body = _sse(
            {
                "upsell_information": SOFT_UPSELL,
                "blocks": [
                    {"intended_usage": "web_results", block_key: {"items": [{"name": "x"}]}}
                ],
            }
        )
        assert _wall(body) is None, f"a served {block_key} was mistaken for a refusal"


def test_a_double_encoded_wall_is_still_a_refusal() -> None:
    """MINOR-1: the JSON-string branch had one unwrap, and two escaped it.

    A wall arriving as a double-encoded string was served to the model as a
    search result -- this fix's own bug, in the shape the stream sometimes uses.
    """
    body = _sse(
        {
            "upsell_information": json.dumps(json.dumps(WALL_UPSELL)),
            "blocks": _ASK_ONLY_BLOCKS,
        }
    )
    assert _wall(body) is not None


@pytest.mark.asyncio
async def test_a_served_shopping_answer_survives_the_chain(tmp_path) -> None:
    """The end-to-end half of MAJOR-1, through the provider entry point."""
    body = _sse(
        {
            "upsell_information": SOFT_UPSELL,
            "blocks": [
                {
                    "intended_usage": "web_results",
                    "shopping_block": {"products": [{"name": "marathon shoe"}]},
                },
                {
                    "intended_usage": "ask_text",
                    "markdown_block": {"answer": "Here are the best marathon shoes."},
                },
            ],
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["perplexity"].search(
            client, _credentials(tmp_path), WebSearchSettings(), "query", 3
        )

    # Not raised away: the answer is the served one, and the sign-in nudge is
    # not treated as the reason to discard it.
    assert response.answer == "Here are the best marathon shoes."


@pytest.mark.asyncio
async def test_a_multi_leg_failure_leads_with_the_actionable_sentence(
    tmp_path, monkeypatch
) -> None:
    """Round-1 D4/U7: the cropped card showed the leg that could never run.

    At six legs the whole-chain summary was ~1200 cells while the collapsed tool
    card paints its first ~34 -- which used to be "All configured web search
    providers failed: brave: not configured", the one leg that was never going to
    serve. The message now leads with the last leg's own sentence (the paid
    backstop on a free-first chain), then names the count and every leg tried, and
    no longer calls auto-joined providers "configured".
    """
    from local_operator.web_search import providers as module
    from local_operator.web_search.service import WebSearchService

    monkeypatch.setattr(
        module,
        "provider_auth_mode",
        lambda provider_id, _credentials, _settings: {
            "duckduckgo": "credential-free",
            "tavily": "keyless",
            "exa": "keyless-mcp",
            "deepseek": "login",
        }.get(provider_id, ""),
    )
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo", "deepseek"]), _credentials(tmp_path)
    )

    async def walled(*_args: object, **_kwargs: object):
        raise RuntimeError("Fetch a page directly, or set DEEPSEEK_API_KEY: refused")

    for provider_id in ("duckduckgo", "tavily", "exa", "deepseek"):
        monkeypatch.setitem(PROVIDERS, provider_id, SimpleNamespace(search=walled))

    with pytest.raises(RuntimeError) as raised:
        await service.search("query")

    message = str(raised.value)
    # COUNT FIRST, then the cause: the collapsed tool card paints ~25 cells, so a
    # count sitting after the reason can never be seen (round-2 D2-1).
    assert message.startswith("4/4 providers failed: Fetch a page directly")
    # Rotation decides the order the legs were TRIED in, so the list is asserted as a
    # set: the breakdown names every leg, in the order this call attempted them, and
    # no other.
    tried = message.split("tried: ", 1)[1].rstrip(")")
    assert {entry.split(": ")[0] for entry in tried.split("; ")} == {
        "duckduckgo",
        "tavily",
        "exa",
        "deepseek",
    }
    assert "configured" not in message


@pytest.mark.asyncio
async def test_the_failure_lead_skips_empty_and_unconfigured_reasons(tmp_path, monkeypatch) -> None:
    """Round-2 N4/Q2-2: the lead must be the first reason that says something.

    A stalled leg reports `httpx.ReadTimeout`, whose `str()` is empty, and a listed
    leg with no credential reports `not configured` -- leading with either buried the
    leg that actually failed on the wire.
    """
    from local_operator.web_search import providers as module
    from local_operator.web_search.service import WebSearchService

    monkeypatch.setattr(
        module,
        "provider_auth_mode",
        lambda provider_id, _credentials, _settings: {
            "duckduckgo": "credential-free",
            "tavily": "keyless",
            # No mode for brave: listed, so it stays in the prefix and reports
            # `not configured` without touching the network.
            "exa": "keyless-mcp",
        }.get(provider_id, ""),
    )
    # `ordered`, so the legs are attempted in the stored order: under round_robin the
    # lead depends on which leg the rotation put first, and this test is about WHICH
    # REASON is chosen, not about the rotation.
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo", "tavily", "brave"], strategy="ordered"),
        _credentials(tmp_path),
    )

    async def stalled(*_args: object, **_kwargs: object):
        raise httpx.ReadTimeout("")

    async def refused(*_args: object, **_kwargs: object):
        raise RuntimeError("returned HTTP 503")

    monkeypatch.setitem(PROVIDERS, "duckduckgo", SimpleNamespace(search=refused))
    monkeypatch.setitem(PROVIDERS, "tavily", SimpleNamespace(search=stalled))
    monkeypatch.setitem(PROVIDERS, "exa", SimpleNamespace(search=stalled))

    with pytest.raises(RuntimeError) as raised:
        await service.search("query")

    message = str(raised.value)
    assert message.startswith("4/4 providers failed: returned HTTP 503")
    assert "not configured" in message  # every leg's reason is still in the digest


@pytest.mark.asyncio
async def test_a_stalled_leg_reports_its_error_class_not_an_empty_reason(
    tmp_path, monkeypatch
) -> None:
    """Round-2 N4: `str(httpx.ReadTimeout())` is `''`, so the digest said nothing."""
    from local_operator.web_search import providers as module
    from local_operator.web_search.service import WebSearchService

    monkeypatch.setattr(
        module,
        "provider_auth_mode",
        lambda provider_id, _credentials, _settings: {
            "duckduckgo": "credential-free",
            "brave": "api-key",
        }.get(provider_id, ""),
    )

    async def stalled(*_args: object, **_kwargs: object):
        raise httpx.ReadTimeout("")

    async def refused(*_args: object, **_kwargs: object):
        raise RuntimeError("BRAVE_API_KEY is not set")

    monkeypatch.setitem(PROVIDERS, "duckduckgo", SimpleNamespace(search=stalled))
    monkeypatch.setitem(PROVIDERS, "brave", SimpleNamespace(search=refused))
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo", "brave"], strategy="ordered"),
        _credentials(tmp_path),
    )

    with pytest.raises(RuntimeError) as raised:
        await service.search("query")

    message = str(raised.value)
    assert message.startswith("2/2 providers failed: ReadTimeout")
    assert "duckduckgo: ReadTimeout" in message
    assert "duckduckgo: ;" not in message


@pytest.mark.asyncio
async def test_the_failure_message_fits_the_card_reason_budget(tmp_path, monkeypatch) -> None:
    """Round-3 D3-2: the message must fit the card, with real reasons on six legs.

    Quoting each leg's FULL reason (a provider's 503 HTML body is ~200 cells) made
    the message ~1400 cells against the card's 432-cell reason budget, so the
    expansion folded at `… 11 more lines` and the reader could not see which other
    legs failed. The caps are what keep the whole diagnosis visible.
    """
    from local_operator.tui.widgets.tool_card import REASON_MAX_CELLS
    from local_operator.web_search.service import WebSearchService

    body = (
        "Tavily returned HTTP 503: <html><head><title>503 Service Unavailable</title>"
        "</head><body><h1>503 Service Unavailable</h1><p>No server is available to "
        "handle this request.</p></body></html>"
    )

    async def boom(*_args: object, **_kwargs: object):
        raise RuntimeError(body)

    providers = ("duckduckgo", "tavily", "perplexity", "exa", "parallel", "deepseek")
    for provider_id in providers:
        monkeypatch.setitem(PROVIDERS, provider_id, SimpleNamespace(search=boom))

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "stored")
    service = WebSearchService(
        WebSearchSettings(
            providers=["duckduckgo", "tavily", "perplexity", "deepseek"], strategy="ordered"
        ),
        credentials,
    )

    with pytest.raises(RuntimeError) as raised:
        await service.search("x")

    message = str(raised.value)
    assert len(message) <= REASON_MAX_CELLS, (len(message), message)
    # Every leg is still named, and its reason is recognisable rather than dropped.
    for provider_id in ("duckduckgo", "tavily", "perplexity", "exa", "parallel"):
        assert f"{provider_id}: " in message


def test_the_copy_helpers_are_total_for_an_id_outside_the_catalogue() -> None:
    """Round-2 N6: the refusal path must not raise a KeyError of its own.

    No validated surface can reach this (the tool schema is a closed Literal, the CLI
    uses `choices`), which is exactly why the error path is the wrong place to depend
    on it.
    """
    from local_operator.web_search.providers import provider_setup_hint

    hint = provider_setup_hint(cast(SearchProviderId, "not-a-provider"))
    assert "search setup not-a-provider" in hint


def test_chain_markers_name_paid_unready_and_best_effort_legs(tmp_path) -> None:
    """Round-2 D2-2/U2-6: the chain row must agree with each leg's state word."""
    from local_operator.web_search.providers import chain_label, provider_statuses

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "stored")
    settings = WebSearchSettings(providers=["duckduckgo", "brave", "perplexity", "deepseek"])
    label = chain_label(provider_statuses(settings, credentials))

    # A listed leg with no credential IS walked, so the row says why it is there.
    assert "Brave (setup needed)" in label
    assert label.endswith("DeepSeek (paid)")
    # A listed best-effort leg keeps its tier in the chain row as well as on its row.
    assert "Perplexity (best-effort)" in label


def test_the_two_best_effort_words_match_the_placements_the_resolver_makes(tmp_path) -> None:
    """Round-3 D3-1: each word describes ONE placement, so neither sentence is false.

    Only a METERED listed leg is hoisted out of the prefix, so a listed best-effort
    leg is always inside the rotating pool, and an auto-joined one is always in the
    fallback band after it. That is what lets the two meanings be properties of
    their words rather than a guess about where a given leg landed -- and this pins
    the resolver rule they depend on, so a future change to the hoisting rule fails
    here instead of shipping a legend that contradicts the chain row.
    """
    from local_operator.web_search.providers import (
        STATE_MEANINGS,
        provider_state_label,
        provider_statuses,
        resolve_provider_bands,
    )

    credentials = _credentials(tmp_path)

    listed_settings = WebSearchSettings(providers=["duckduckgo", "perplexity"])
    bands = resolve_provider_bands(listed_settings, credentials)
    assert "perplexity" in bands.prefix and "perplexity" not in bands.fallback
    listed = next(
        status
        for status in provider_statuses(listed_settings, credentials)
        if status.id == "perplexity"
    )
    assert provider_state_label(listed) == "enabled (best-effort)"
    assert "after the free pool" not in STATE_MEANINGS["enabled (best-effort)"]

    unlisted_settings = WebSearchSettings(providers=["duckduckgo"])
    bands = resolve_provider_bands(unlisted_settings, credentials)
    assert "perplexity" in bands.fallback and "perplexity" not in bands.prefix
    unlisted = next(
        status
        for status in provider_statuses(unlisted_settings, credentials)
        if status.id == "perplexity"
    )
    assert provider_state_label(unlisted) == "auto best-effort"
    assert "after the free pool" in STATE_MEANINGS["auto best-effort"]


def test_the_legend_defines_the_painted_words_and_only_those(tmp_path) -> None:
    """Round-2 U2-5/D2-4, round-3 D3-3: meanings, scoped to what is on screen.

    The full table is 477 cells and mostly defines states the install does not have,
    while the header rows of the same listing are what a narrow terminal folds away
    (round-3 measured 6 of 26 painted rows at 110x44, 8 of 15 at 80x24).
    """
    from local_operator.web_search.providers import (
        STATE_MEANINGS,
        provider_state_label,
        provider_statuses,
        state_legend,
    )

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "stored")
    statuses = provider_statuses(
        WebSearchSettings(providers=["duckduckgo"], excluded_providers=["tavily"]),
        credentials,
    )
    legend = state_legend(statuses)
    painted = [provider_state_label(status) for status in statuses]

    for word in painted:
        assert f"{word} = {STATE_MEANINGS[word]}" in legend
    # Nothing on screen is undefined, and nothing undefined is on the line.
    assert "enabled (paid)" not in legend or "enabled (paid)" in painted
    assert "excluded = " in legend  # tavily is excluded in this fixture

    # Scoping is real: drop the exclusion and the word leaves the line with it.
    without = state_legend(
        provider_statuses(WebSearchSettings(providers=["duckduckgo"]), credentials)
    )
    assert "excluded = " not in without
    assert len(without) < len(legend)


@pytest.mark.asyncio
async def test_a_forced_provider_failure_names_only_that_provider(tmp_path, monkeypatch) -> None:
    """Round-1 design review D2: the message described a chain that never ran.

    With one candidate -- a forced provider, or the only one enabled -- the old
    text said "All configured web search providers failed" while the others had
    not been asked.
    """

    from local_operator.web_search.service import WebSearchService

    manager = _credentials(tmp_path)
    settings = WebSearchSettings(providers=["duckduckgo", "perplexity"], strategy="ordered")
    service = WebSearchService(settings, manager)

    async def walled(*_args: object, **_kwargs: object):
        raise RuntimeError("Fetch a page directly, or set PERPLEXITY_API_KEY: refused")

    # Replace the whole registry ENTRY: ``PROVIDERS`` maps ids to provider
    # DEFINITIONS (objects, not dicts), and only ``.search`` is reached here.
    monkeypatch.setitem(PROVIDERS, "perplexity", SimpleNamespace(search=walled))
    with pytest.raises(RuntimeError) as raised:
        await service.search("query", forced_provider="perplexity")

    message = str(raised.value)
    assert "all configured web search providers failed" not in message.lower()
    assert "only provider tried" in message
    assert "duckduckgo" not in message
    # D1: the provider's actionable sentence leads, so it survives the card's
    # single-line crop; the scope note follows it rather than preceding it.
    assert message.index("Fetch a page directly") < message.index("only provider tried")


# ---------------------------------------------------------------------------
# Round-2 review: MINOR-1 (served key), NIT-1 (private key), MINOR-3 (strip)
# ---------------------------------------------------------------------------


def test_a_top_level_source_list_is_served_content_not_a_wall() -> None:
    """MINOR-1: only the ``blocks`` loop marked a payload as served.

    ``_perplexity_sources`` honours top-level ``sources_list`` and
    ``search_results``, so a real answer arriving that way plus the soft sign-in
    marker was raised away whenever the extractor rejected the entries -- the
    entries with no URL scheme reach the parser's notice but not its source map.
    """
    # The answer is a REAL one, not the wall sentence: the claim being pinned is
    # that a served answer survives, and a fixture whose text is the wall's own
    # could not show that (round-1 review NIT-2).
    real_answer = [
        {"intended_usage": "ask_text", "markdown_block": {"answer": "Paris is the capital."}}
    ]
    for key in ("sources_list", "search_results"):
        for row in ({"url": "https://example.com"}, {"url": "example.com"}):
            body = _sse({"upsell_information": SOFT_UPSELL, "blocks": real_answer, key: [row]})
            assert _wall(body) is None, f"a served top-level {key} was mistaken for a refusal"


def test_a_top_level_row_with_no_url_is_not_served_content() -> None:
    """MINOR-1: a dict row is not evidence of a result.

    ``_perplexity_sources`` builds nothing from a row without a URL, so a
    name-only or empty row beside a wall left the wall's own sentence as the
    response -- the original bug, reached through the new check.
    """
    # Truthiness is not enough either (round-2 review MINOR-1): ``_source``
    # strips and requires a scheme, so a blank url and a non-string url each
    # produce zero sources while suppressing the wall.
    for row in (
        {"name": "x"},
        {},
        {"title": "x"},
        {"url": "   "},
        {"url": 5},
        {"url": True},
        {"url": {}},
        {"url": None},
    ):
        for key in ("sources_list", "search_results"):
            body = _sse(
                {
                    "upsell_information": WALL_UPSELL,
                    "blocks": _ASK_ONLY_BLOCKS,
                    key: [row],
                }
            )
            assert _wall(body) is not None, f"a wall escaped behind a url-less {key} row"


# ---------------------------------------------------------------------------
# Issue #1070: the served-url predicate and the extractor's cap must agree
# ---------------------------------------------------------------------------


def _url_of(length: int) -> str:
    """A scheme-carrying url of exactly ``length`` characters.

    The scheme matters: a url with no scheme is rejected by ``_source`` for a
    different reason, so a length test built on one would pass whether or not
    the cap is honoured.
    """
    return "http://" + "x" * (length - len("http://"))


#: A real answer block, so the served cases below can show the answer that
#: survives rather than the wall's own sentence coming back.
_REAL_ANSWER_BLOCKS = [
    {"intended_usage": "ask_text", "markdown_block": {"answer": "Paris is the capital."}}
]


def _served_and_sourced(body: str) -> tuple[str | None, list[SearchSource]]:
    """The wall verdict and the extracted sources for one SSE body."""
    from local_operator.web_search.providers import (
        _parse_perplexity_sse,
        _perplexity_authwall,
        _perplexity_sources,
    )

    payload = _parse_perplexity_sse(body)
    return _perplexity_authwall(payload), _perplexity_sources(payload, 10)


@pytest.mark.parametrize("url", ["x" * 5_000, _url_of(4_097)])
@pytest.mark.parametrize("key", ["sources_list", "search_results"])
def test_a_row_whose_url_exceeds_the_extractor_cap_is_not_served_content(
    url: str, key: str
) -> None:
    """#1070: the predicate took any non-blank url, and ``_source`` caps it.

    A row past the cap suppressed the wall while the extractor built no source
    from it, so ``not sources`` never fired either and the refusal sentence
    (``Sign up and repeat your request.``) went back as the search result -- the
    #1061 bug one rule narrower. Both url shapes the issue names are covered:
    ``"x" * 5000`` is the issue's own repro (no scheme either, which the
    extractor rejects independently) and 4097 carries a valid scheme, so it is
    the shape that is over the cap and nothing else.
    """
    body = _sse(
        {
            "upsell_information": WALL_UPSELL,
            "blocks": _ASK_ONLY_BLOCKS,
            key: [{"url": url}],
        }
    )
    wall, sources = _served_and_sourced(body)
    assert wall is not None, f"a wall escaped behind an over-long {key} url"
    # The zero-sources half is the extractor's own rule, measured rather than
    # assumed: it is what made the suppressed wall reachable in the first place.
    assert sources == []


def test_the_served_url_cap_agrees_with_the_extractor_on_both_sides() -> None:
    """The boundary: exactly the cap serves, one character past it refuses.

    Both sides are derived from the url lengths the issue names, and the
    lengths are checked against the single constant ``_source`` and the
    predicate share -- so a second, drifting spelling of the cap fails here
    instead of leaving one of these two shapes unpinned.
    """
    from local_operator.web_search.providers import _MAX_URL_CHARS

    assert _MAX_URL_CHARS == 4_096

    at_cap, past_cap = _url_of(4_096), _url_of(4_097)
    assert (len(at_cap), len(past_cap)) == (4_096, 4_097)

    served_wall, served_sources = _served_and_sourced(
        _sse(
            {
                "upsell_information": WALL_UPSELL,
                "blocks": _REAL_ANSWER_BLOCKS,
                "sources_list": [{"url": at_cap}],
            }
        )
    )
    assert served_wall is None
    assert [source.url for source in served_sources] == [at_cap]

    refused_wall, refused_sources = _served_and_sourced(
        _sse(
            {
                "upsell_information": WALL_UPSELL,
                "blocks": _REAL_ANSWER_BLOCKS,
                "sources_list": [{"url": past_cap}],
            }
        )
    )
    assert refused_wall is not None
    assert refused_sources == []


def test_a_sources_list_of_bare_strings_is_not_served_content() -> None:
    """``["https://x"]`` is refused: ``_source`` reads object rows only.

    Correct by inspection before this test, but unpinned (issue #1070). A
    string is not a served result -- the extractor builds nothing from it -- so
    counting it as one would suppress the wall with zero sources.
    """
    body = _sse(
        {
            "upsell_information": WALL_UPSELL,
            "blocks": _ASK_ONLY_BLOCKS,
            "sources_list": ["https://x"],
        }
    )
    wall, sources = _served_and_sourced(body)
    assert wall is not None
    assert sources == []


def test_a_scheme_with_no_authority_is_served_with_one_source() -> None:
    """``{"url": "http://"}`` serves: a scheme is required, a host is not.

    Correct by inspection before this test, but unpinned (issue #1070). The
    extractor accepts it and builds one source, so the wall is suppressed and
    the result stands -- the shape the length cap must NOT start refusing.
    """
    body = _sse(
        {
            "upsell_information": WALL_UPSELL,
            "blocks": _REAL_ANSWER_BLOCKS,
            "sources_list": [{"url": "http://"}],
        }
    )
    wall, sources = _served_and_sourced(body)
    assert wall is None
    assert [source.url for source in sources] == ["http://"]


@pytest.mark.asyncio
async def test_a_wall_behind_an_over_long_row_refuses_through_the_provider(tmp_path) -> None:
    """#1070 end to end: the refusal is raised, not handed back as the answer.

    The parsing half is the unit above; this drives the provider entry point
    the chain calls, which is where the sentence used to reach the model.
    """
    wall_event = {
        "uuid": "pplx-long-url",
        "status": "COMPLETED",
        "final": True,
        "text": "Sign up and repeat your request.",
        "upsell_information": {
            "name": "fraud_authwall_upsell",
            "upsell_type": "LOGIN",
            "cta": "SIGN_UP_OR_LOGIN",
        },
        "sources_list": [{"url": "x" * 5_000}],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(wall_event))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError) as caught:
            await PROVIDERS["perplexity"].search(
                client,
                _credentials(tmp_path),
                WebSearchSettings(),
                "query",
                3,
            )

    message = str(caught.value)
    assert "fraud_authwall_upsell" in message
    assert "Sign up and repeat your request." not in message


def test_a_payload_carrying_our_own_served_key_cannot_suppress_a_wall() -> None:
    """NIT-1: the private key is the parser's, so the parser always sets it.

    Left to ``merged.update``, a payload that arrived carrying the key survived
    and suppressed the wall -- the original bug reached from the other side.
    """
    body = _sse(
        {
            "upsell_information": WALL_UPSELL,
            "blocks": _ASK_ONLY_BLOCKS,
            "_lo_served_blocks": ["web_result_block"],
        }
    )
    assert _wall(body) is not None


@pytest.mark.asyncio
async def test_a_single_candidate_message_keeps_the_provider_name_once(
    tmp_path, monkeypatch
) -> None:
    """MINOR-3: the prefix strip was load-bearing and unpinned.

    Reverting it left every test passing. It exists so the head of the cropped
    line carries the remedy rather than repeating the provider the reader just
    saw; the shape it strips is built by the same loop that formats the entry.
    """
    from local_operator.web_search.service import WebSearchService

    settings = WebSearchSettings(providers=["duckduckgo", "perplexity"], strategy="ordered")
    service = WebSearchService(settings, _credentials(tmp_path))

    async def walled(*_args: object, **_kwargs: object):
        raise RuntimeError("Fetch a page directly, or set PERPLEXITY_API_KEY: refused")

    monkeypatch.setitem(PROVIDERS, "perplexity", SimpleNamespace(search=walled))
    with pytest.raises(RuntimeError) as raised:
        await service.search("query", forced_provider="perplexity")

    message = str(raised.value)
    assert message.startswith("Web search failed: Fetch a page directly")
    # The provider is named once, in the scope note -- not again as a prefix.
    assert "perplexity: Fetch" not in message
    assert message.count("perplexity") == 1


@pytest.mark.asyncio
async def test_a_keyed_search_reports_usage_so_it_is_not_priced_free(tmp_path) -> None:
    """MAJOR-1, end to end: the keyed branch handed back a usage-less response."""
    payload = {
        "id": "req-1",
        "choices": [{"message": {"content": "Grounded answer"}}],
        "citations": ["https://example.com"],
        "usage": {"prompt_tokens": 14_500, "completion_tokens": 776},
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    manager = _credentials(tmp_path)
    manager.set_credential("PERPLEXITY_API_KEY", "test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["perplexity"].search(
            client, manager, WebSearchSettings(), "query", 3
        )

    assert response.auth_mode == "api-key"
    assert response.usage is not None
    # ``keyless=False`` is what routes it away from the free anonymous price.
    assert response.usage.keyless is False
    assert response.usage.input_tokens == 14_500
    assert response.usage.output_tokens == 776


# ---------------------------------------------------------------------------
# The two keyless MCP transports, and the provider-resolution truth table
# ---------------------------------------------------------------------------

#: The two framings the servers were observed to answer with (2026-09-18): Exa
#: sends SSE, Parallel sends a bare JSON body. Both are exercised for both
#: providers, because a parser that handled only the one it was written against
#: would report the other as unparseable.
_MCP_FRAMINGS = ("sse", "json")


def _mcp_request_body(**arguments: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "web_search_exa", "arguments": arguments},
    }


def _mcp_sse(envelope: dict[str, Any]) -> str:
    """SSE framing, as Exa answers: one ``event: message`` and a ``data:`` line."""
    return f"event: message\ndata: {json.dumps(envelope)}\n\n"


def _exa_text() -> str:
    return (
        "Title: Coroutines and tasks\n"
        "URL: https://docs.example/asyncio-task\n"
        "Published: 2026-01-02\n"
        "Author: N/A\n"
        "Highlights:\n"
        "first fragment\n"
        "...\n"
        "second fragment\n"
        "Title: A block with no URL\n"
        "Published: N/A\n"
        "Highlights:\n"
        "orphan body text\n"
        "Title: Second page\n"
        "URL: https://docs.example/second\n"
        "Published: N/A\n"
        "Author: Ada\n"
        "Highlights:\n"
        "second page body\n"
    )


@pytest.mark.parametrize("framing", _MCP_FRAMINGS)
@pytest.mark.asyncio
async def test_exa_keyless_mcp_parses_the_rendered_text_blob(tmp_path, framing) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content)
        envelope = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": _exa_text()}], "isError": False},
        }
        body = _mcp_sse(envelope) if framing == "sse" else json.dumps(envelope)
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["exa"].search(
            client, _credentials(tmp_path), WebSearchSettings(), "semantic query", 4
        )

    assert seen["payload"]["params"]["name"] == "web_search_exa"
    assert seen["payload"]["params"]["arguments"]["query"] == "semantic query"
    assert seen["headers"]["content-type"] == "application/json"
    # Exa refuses a JSON-only Accept with 406, reproduced live.
    assert "text/event-stream" in seen["headers"]["accept"]
    assert "application/json" in seen["headers"]["accept"]

    assert response.auth_mode == "keyless-mcp"
    assert [source.url for source in response.sources] == [
        "https://docs.example/asyncio-task",
        "https://docs.example/second",
    ]
    assert response.sources[0].title == "Coroutines and tasks"
    assert response.sources[0].published_date == "2026-01-02"
    assert response.sources[0].snippet == "first fragment\n...\nsecond fragment"
    assert response.sources[1].published_date is None
    assert response.usage is not None and response.usage.keyless is True


def test_exa_mcp_drops_a_block_without_a_usable_url() -> None:
    sources = parse_exa_mcp_text(_exa_text(), 10)

    assert [source.url for source in sources] == [
        "https://docs.example/asyncio-task",
        "https://docs.example/second",
    ]
    assert all("orphan" not in (source.snippet or "") for source in sources)


@pytest.mark.parametrize(
    "payload,message",
    [
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32602, "message": "Tool nope not found"},
            },
            "Exa MCP error -32602: Tool nope not found",
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": "MCP error -32602: Tool nope not found"}],
                    "isError": True,
                },
            },
            "Exa MCP error: MCP error -32602: Tool nope not found",
        ),
    ],
)
@pytest.mark.asyncio
async def test_an_mcp_error_envelope_raises_rather_than_returning_nothing(
    tmp_path, payload, message
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_mcp_sse(payload))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError) as caught:
            await PROVIDERS["exa"].search(
                client, _credentials(tmp_path), WebSearchSettings(), "query", 3
            )

    assert message in str(caught.value)


@pytest.mark.asyncio
async def test_an_mcp_http_error_carries_the_status_and_body(tmp_path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError) as caught:
            await PROVIDERS["parallel"].search(
                client, _credentials(tmp_path), WebSearchSettings(), "query", 3
            )

    assert "Parallel returned HTTP 429: rate limited" in str(caught.value)


def _parallel_envelope(text: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            # The vendor's own meter. It must NEVER become our `usd`: no account
            # exists to bill, and the endpoint is documented as free.
            "_meta": {"parallel/usage": [{"name": "sku_search", "count": 1, "cost_usd": 0.001}]},
            "content": [{"type": "text", "text": text}],
            "isError": False,
        },
    }


@pytest.mark.parametrize("framing", _MCP_FRAMINGS)
@pytest.mark.asyncio
async def test_parallel_keyless_mcp_maps_excerpts_and_ignores_the_vendor_meter(
    tmp_path, framing
) -> None:
    inner = {
        "search_id": "search_abc",
        "results": [
            {
                "url": "https://docs.example/one",
                "title": "One",
                "publish_date": "2026-03-04",
                "excerpts": ["first crop", "second crop", "third crop"],
            },
            {
                "url": "https://docs.example/two",
                "title": "Two",
                "publish_date": None,
                "excerpts": [],
            },
        ],
        "warnings": None,
    }
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        envelope = _parallel_envelope(json.dumps(inner))
        body = json.dumps(envelope) if framing == "json" else _mcp_sse(envelope)
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["parallel"].search(
            client, _credentials(tmp_path), WebSearchSettings(), "asyncio taskgroup", 3
        )

    assert seen["payload"]["params"]["name"] == "web_search"
    arguments = seen["payload"]["params"]["arguments"]
    assert arguments["objective"] == "asyncio taskgroup"
    assert arguments["search_queries"] == ["asyncio taskgroup"]
    # No key is stored: no Authorization header is sent, and the mode says so.
    assert "authorization" not in seen["headers"]
    assert response.auth_mode == "keyless-mcp"
    assert [source.url for source in response.sources] == [
        "https://docs.example/one",
        "https://docs.example/two",
    ]
    assert response.sources[0].snippet == "first crop\nsecond crop"
    assert response.sources[0].published_date == "2026-03-04"
    assert response.usage is not None and response.usage.keyless is True

    from local_operator.web_search.cost import estimate_search_cost

    cost = estimate_search_cost("parallel", response.usage)
    assert cost.usd == 0.0 and "keyless" in cost.basis, "the vendor's meter is not our spend"


@pytest.mark.asyncio
async def test_parallel_keyed_mode_sends_authorization_and_is_unpriced(tmp_path) -> None:
    credentials = _credentials(tmp_path)
    credentials.set_credential("PARALLEL_API_KEY", "stored-test-key")
    inner = {"results": [{"url": "https://docs.example/one", "title": "One", "excerpts": ["x"]}]}
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, text=json.dumps(_parallel_envelope(json.dumps(inner))))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PROVIDERS["parallel"].search(
            client, credentials, WebSearchSettings(), "query", 3
        )

    assert seen["headers"]["authorization"] == "Bearer stored-test-key"
    assert response.auth_mode == "api-key"
    assert response.usage is not None and response.usage.keyless is False

    from local_operator.web_search.cost import estimate_search_cost

    cost = estimate_search_cost("parallel", response.usage)
    assert cost.usd is None, "no published rate for the keyed Search API: unpriced, not free"
    assert cost.basis == "no published rate"


def test_parallel_mcp_unparseable_text_raises() -> None:
    with pytest.raises(RuntimeError, match="unparseable"):
        parse_parallel_mcp_text("not json at all", 3)


def test_parallel_parser_skips_rows_without_a_usable_url() -> None:
    sources = parse_parallel_mcp_text(
        json.dumps(
            {
                "results": [
                    {"url": "not-a-url", "title": "Relative", "excerpts": []},
                    {"url": "https://docs.example/one", "title": "One", "excerpts": ["body"]},
                ]
            }
        ),
        10,
    )

    assert [source.url for source in sources] == ["https://docs.example/one"]


def test_provider_auth_mode_truth_table_covers_the_whole_catalogue(tmp_path) -> None:
    """One function answers availability AND the transport each provider would use."""
    from local_operator.web_search import providers as module

    credentials = _credentials(tmp_path)
    settings = WebSearchSettings()

    # Keyless by default, in the order the catalogue declares them.
    assert module.provider_auth_mode("duckduckgo", credentials, settings) == "credential-free"
    assert module.provider_auth_mode("tavily", credentials, settings) == "keyless"
    assert module.provider_auth_mode("perplexity", credentials, settings) == "anonymous"
    assert module.provider_auth_mode("exa", credentials, settings) == "keyless-mcp"
    assert module.provider_auth_mode("parallel", credentials, settings) == "keyless-mcp"
    # Cannot serve without a credential, and none is stored.
    for provider_id in ("deepseek", "brave", "serpapi"):
        assert module.provider_auth_mode(provider_id, credentials, settings) == ""
    # SearXNG needs an endpoint, not a key.
    assert module.provider_auth_mode("searxng", credentials, settings) == ""
    assert (
        module.provider_auth_mode(
            "searxng", credentials, WebSearchSettings(searxng_endpoint="https://searx.example")
        )
        == "self-hosted"
    )

    for provider_id in PROVIDER_IDS:
        credentials.set_credential(
            {
                "tavily": "TAVILY_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "perplexity": "PERPLEXITY_API_KEY",
                "brave": "BRAVE_API_KEY",
                "exa": "EXA_API_KEY",
                "parallel": "PARALLEL_API_KEY",
                "serpapi": "SERPAPI_API_KEY",
            }.get(provider_id, "UNUSED_KEY"),
            "stored",
        )
    # With a credential stored, every keyed provider reports the api-key transport
    # -- including the two whose keyless mode is otherwise the default.
    for provider_id in ("tavily", "deepseek", "perplexity", "brave", "exa", "parallel", "serpapi"):
        assert (
            module.provider_auth_mode(provider_id, credentials, settings) == "api-key"
        ), provider_id


def test_provider_statuses_reports_the_resolved_chain_not_the_stored_list(tmp_path) -> None:
    from local_operator.web_search.providers import (
        provider_state_label,
        provider_statuses,
    )

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "stored")
    settings = WebSearchSettings(providers=["duckduckgo"], excluded_providers=["tavily"])
    statuses = {status.id: status for status in provider_statuses(settings, credentials)}

    # The bug report in one line: exa serves the chain while `listed` is False.
    assert statuses["exa"].enabled is True
    assert statuses["exa"].listed is False
    assert statuses["exa"].tier == "rotate"
    assert statuses["exa"].mode == "keyless-mcp"
    assert provider_state_label(statuses["exa"]) == "auto free"

    assert provider_state_label(statuses["duckduckgo"]) == "enabled"
    assert provider_state_label(statuses["tavily"]) == "excluded"
    assert statuses["tavily"].enabled is False
    # `off` became `needs setup`: it read as the master switch on the same screen
    # and repeated the readiness column word for word (round-1 D6).
    assert provider_state_label(statuses["brave"]) == "needs setup"
    assert provider_state_label(statuses["deepseek"]) == "auto paid"


def test_the_status_rows_follow_the_chain_and_the_paid_leg_is_never_plain_enabled(
    tmp_path,
) -> None:
    """Rounds 1's D1 and U2: `paid: (none)` beside a metered leg, and a plain `enabled`.

    The listing has to corroborate the chain line above it, and an install that
    LISTS a paid provider must not read as if nothing paid were involved.
    """
    from local_operator.web_search.providers import (
        chain_label,
        provider_state_label,
        provider_statuses,
        resolve_providers,
    )

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "stored")
    settings = WebSearchSettings(providers=["duckduckgo", "perplexity", "deepseek"])
    chain = resolve_providers(settings, credentials)
    statuses = provider_statuses(settings, credentials)
    order = [status.id for status in statuses]

    # DeepSeek is listed THIRD and resolves LAST: the free legs come first, and
    # the listed paid leg leads only the paid band.
    assert chain == ["duckduckgo", "perplexity", "tavily", "exa", "parallel", "deepseek"]
    assert order[: len(chain)] == chain, "the rows follow the chain, not the catalogue"
    # DeepSeek is STILL listed, and its state word says both facts.
    assert statuses[order.index("deepseek")].listed is True
    assert provider_state_label(statuses[order.index("deepseek")]) == "enabled (paid)"

    # The chain line spans the whole chain and marks the paid leg, so the summary
    # cannot print `(none)` while a metered leg is in the chain.
    label = chain_label(statuses)
    assert label.endswith("DeepSeek (paid)")
    assert "DuckDuckGo" in label and "Exa" in label


def test_the_landing_line_reuses_the_status_vocabulary(tmp_path) -> None:
    """`search enable` and `search list` must not describe the same provider differently.

    Both read `provider_state_label`, whose meanings live in `STATE_MEANINGS`, so a
    change to the vocabulary moves both surfaces; this pins the pairing rather than
    the prose.
    """
    from local_operator.web_search.providers import (
        STATE_MEANINGS,
        provider_landing_line,
        provider_state_label,
        provider_statuses,
        state_legend,
    )

    credentials = _credentials(tmp_path)
    credentials.set_credential("DEEPSEEK_API_KEY", "stored")
    settings = WebSearchSettings(providers=["duckduckgo"], excluded_providers=["tavily"])
    statuses = {status.id: status for status in provider_statuses(settings, credentials)}

    # No stutter for a listed leg (`deepseek enabled (enabled; …)` was round-1 D3),
    # and a landing sentence for every state the vocabulary can produce.
    assert provider_landing_line("duckduckgo", statuses["duckduckgo"]) == (
        "duckduckgo enabled (in your priority order)"
    )
    assert provider_landing_line("deepseek", statuses["deepseek"]) == (
        "deepseek enabled (auto paid; tried after the free providers, never before a free leg)"
    )
    # A provider that cannot serve is told the command that fixes it, instead of
    # "enabled (off; not usable yet)" (round-1 U4).
    assert provider_landing_line("brave", statuses["brave"]) == (
        "brave is allowed, but no search can use it yet: run "
        "`local-operator search setup brave` (BRAVE_API_KEY)"
    )

    # The legend is the vocabulary this listing paints, so no state word can be
    # printed without one (and none is defined that is not printed).
    legend = state_legend(list(statuses.values()))
    for state in {provider_state_label(status) for status in statuses.values()}:
        assert f"{state} = {STATE_MEANINGS[state]}" in legend
    # Every word the vocabulary can emit has a meaning, and the two placements of a
    # listed leg are distinguished by the word itself (paid = hoisted to the paid
    # band; best-effort = stays in the pool where the user put it), so the legend
    # cannot describe one of them with the other's placement.
    assert set(STATE_MEANINGS) == {provider_state_label(status) for status in statuses.values()} | {
        "enabled (paid)",
        "enabled (best-effort)",
    }
    assert "after the free pool" not in STATE_MEANINGS["enabled (best-effort)"]
    assert "pool" in STATE_MEANINGS["enabled (best-effort)"]

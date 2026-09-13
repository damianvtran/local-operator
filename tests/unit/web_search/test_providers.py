from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.credentials import CredentialManager
from local_operator.web_search.models import SearchResponse, WebSearchSettings
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
    settings = WebSearchSettings(providers=["duckduckgo", "perplexity"])
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
    for key in ("sources_list", "search_results"):
        for row in ({"url": "https://example.com"}, {"url": "example.com"}, {"name": "x"}):
            body = _sse(
                {
                    "upsell_information": SOFT_UPSELL,
                    "blocks": _ASK_ONLY_BLOCKS,
                    key: [row],
                }
            )
            assert _wall(body) is None, f"a served top-level {key} was mistaken for a refusal"


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
async def test_a_single_candidate_summary_without_the_prefix_is_untouched(
    tmp_path, monkeypatch
) -> None:
    """The strip must not eat text that merely looks like a prefix."""
    from local_operator.web_search.service import WebSearchService

    settings = WebSearchSettings(providers=["duckduckgo", "perplexity"], strategy="ordered")
    service = WebSearchService(settings, _credentials(tmp_path))

    async def odd(*_args: object, **_kwargs: object):
        raise RuntimeError("Fetch a page directly: refused")

    monkeypatch.setitem(PROVIDERS, "perplexity", SimpleNamespace(search=odd))
    with pytest.raises(RuntimeError) as raised:
        await service.search("query", forced_provider="perplexity")

    assert "Fetch a page directly: refused" in str(raised.value)

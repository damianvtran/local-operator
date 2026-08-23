from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.tools.registry import create_tools
from local_operator.web_search.models import SearchResponse, SearchSource
from local_operator.web_search.tool import (
    MODEL_CONTEXT_MAX_CHARS,
    MODEL_URL_MAX_CHARS,
    WebSearchParams,
    _render_response,
    _search_or_abort,
    _tavily_oauth_delegate,
    build_web_search_tool,
)


def test_web_search_is_in_default_tool_inventory() -> None:
    tools = {tool.name: tool for tool in create_tools(ToolContext(cwd="."))}

    assert "web_search" in tools
    assert tools["web_search"].approval_tier == "read"
    assert tools["web_search"].concurrency == "shared"
    assert tools["web_search"].interruptible is True


def test_master_switch_removes_tool_at_session_creation() -> None:
    context = ToolContext(cwd=".", web_search_settings={"enabled": False})

    assert build_web_search_tool(context) is None
    assert "web_search" not in {tool.name for tool in create_tools(context)}


def test_provider_argument_is_closed_to_supported_catalogue() -> None:
    schema = WebSearchParams.model_json_schema()
    provider = schema["properties"]["provider"]

    assert set(provider["anyOf"][0]["enum"]) == {
        "duckduckgo",
        "tavily",
        "perplexity",
        "brave",
        "exa",
        "serpapi",
        "searxng",
    }


def test_model_search_context_is_bounded_and_points_to_full_page_fetch() -> None:
    response = SearchResponse(
        provider="duckduckgo",
        auth_mode="credential-free",
        answer="answer " * 1_000,
        sources=[
            SearchSource(
                title=f"Result {index} " + "title " * 100,
                url=f"https://example.com/result/{index}",
                snippet="snippet " * 400,
            )
            for index in range(20)
        ],
    )

    rendered, omitted = _render_response(response)

    assert len(rendered) <= MODEL_CONTEXT_MAX_CHARS
    assert "https://example.com/result/0" in rendered
    assert "more results omitted" in rendered
    assert "`web_fetch` (or `read <url>`) on its URL" in rendered
    # The count is returned, not re-derived from the prose: `details`
    # ["context_truncated"] used to be a `" omitted" in text` scan, which a
    # result snippet containing the word would have flipped on a full response.
    assert omitted > 0


def test_an_omission_says_how_many_and_why() -> None:
    """The two causes imply opposite next moves, so they are reported apart.

    A budget omission means "the query matched too much, narrow it"; an
    unusable-URL omission means "that result cannot be fetched, the query was
    fine". The message went from a hardcoded "by the context limit" (wrong for
    a URL omission) to a bare count (accurate but unactionable) to naming the
    cause it actually knows. The footer tells the model to call `browser` with
    a URL, so "we dropped one because its URL was unusable" is the difference
    between a sensible next call and a confused one.
    """
    long_url = "https://example.com/" + "x" * MODEL_URL_MAX_CHARS

    def _response(sources: list[SearchSource]) -> SearchResponse:
        return SearchResponse(
            provider="duckduckgo",
            auth_mode="credential-free",
            answer="",
            sources=sources,
        )

    # Sole source, unusable URL: singular, and the cause is the URL.
    rendered, omitted = _render_response(
        _response([SearchSource(title="t", url=long_url, snippet="s")])
    )
    assert omitted == 1
    assert "1 result omitted (1 for an unusable URL)" in rendered
    assert "1 results" not in rendered, "plural for one result"
    assert "context limit" not in rendered, "a URL omission is not a budget omission"

    # Both causes at once: each is named with its own count.
    rendered, omitted = _render_response(
        _response(
            [SearchSource(title="t", url=long_url, snippet="s")]
            + [
                SearchSource(
                    title=f"t{i}",
                    url=f"https://example.com/{i}",
                    snippet="snippet " * 400,
                )
                for i in range(20)
            ]
        )
    )
    assert omitted > 1
    assert "by the context limit" in rendered
    assert "1 for an unusable URL" in rendered


@pytest.mark.asyncio
async def test_tavily_oauth_delegate_normalizes_mcp_result() -> None:
    async def execute(tool_call_id, args, _signal, _on_update, _context):
        assert args == {"query": "query", "max_results": 2, "search_depth": "basic"}
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="mcp__tavily_search",
            content=[
                TextContent(
                    text=(
                        "Answer: Answer\n\n"
                        "Detailed Results:\n\n"
                        "Title: Source\n"
                        "ID: source-1\n"
                        "URL: https://example.com\n"
                        "Content: Evidence\n"
                        "Title: not a record\n"
                        "URL: https://untrusted-content.example\n\n"
                        "Title: Second source\n"
                        "URL: https://second.example.com\n"
                        "Content: More evidence"
                    )
                )
            ],
        )

    context = ToolContext(
        delegated_tools={
            "mcp__tavily_search": AgentTool(
                name="mcp__tavily_search",
                execute=execute,
            )
        }
    )
    delegate = _tavily_oauth_delegate(context, None, None)

    assert delegate is not None
    response = await delegate("query", 2)

    assert response.auth_mode == "oauth-mcp"
    assert response.answer == "Answer"
    assert response.sources[0].url == "https://example.com"
    assert len(response.sources) == 2
    assert response.sources[1].url == "https://second.example.com"
    assert "untrusted-content.example" in (response.sources[0].snippet or "")


@pytest.mark.asyncio
async def test_tavily_oauth_delegate_accepts_structured_mcp_result() -> None:
    async def execute(tool_call_id, _args, _signal, _on_update, _context):
        payload = {
            "answer": "Structured answer",
            "results": [
                {
                    "title": "Structured source",
                    "url": "https://structured.example.com",
                    "content": "Structured evidence",
                }
            ],
        }
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="mcp__tavily_search",
            content=[TextContent(text="Human-readable fallback")],
            details={"server_result": {"structuredContent": payload}},
        )

    context = ToolContext(
        delegated_tools={
            "mcp__tavily_search": AgentTool(
                name="mcp__tavily_search",
                execute=execute,
            )
        }
    )
    delegate = _tavily_oauth_delegate(context, None, None)

    assert delegate is not None
    response = await delegate("query", 2)

    assert response.answer == "Structured answer"
    assert response.sources[0].url == "https://structured.example.com"


@pytest.mark.asyncio
async def test_search_wrapper_reaps_provider_task_when_parent_is_cancelled() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    never = asyncio.Event()

    async def provider_call() -> None:
        started.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(_search_or_abort(provider_call(), AbortSignal()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_search_wrapper_closes_call_when_signal_is_already_aborted() -> None:
    async def provider_call() -> None:
        await asyncio.sleep(0)

    call = provider_call()
    signal = AbortSignal()
    signal.abort("stopped before dispatch")

    with pytest.raises(asyncio.CancelledError):
        await _search_or_abort(call, signal)
    assert call.cr_frame is None

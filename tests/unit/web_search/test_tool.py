from __future__ import annotations

import json

import pytest

from local_operator.harness.types import AgentTool, TextContent, ToolContext, ToolResult
from local_operator.tools.registry import create_tools
from local_operator.web_search.models import SearchResponse, SearchSource
from local_operator.web_search.tool import (
    MODEL_CONTEXT_MAX_CHARS,
    WebSearchParams,
    _render_response,
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

    rendered = _render_response(response)

    assert len(rendered) <= MODEL_CONTEXT_MAX_CHARS
    assert "https://example.com/result/0" in rendered
    assert "more results omitted" in rendered
    assert "call `browser` with its URL" in rendered


@pytest.mark.asyncio
async def test_tavily_oauth_delegate_normalizes_mcp_result() -> None:
    async def execute(tool_call_id, args, _signal, _on_update, _context):
        assert args == {"query": "query", "max_results": 2, "search_depth": "basic"}
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="mcp__tavily_search",
            content=[
                TextContent(
                    text=json.dumps(
                        {
                            "answer": "Answer",
                            "results": [
                                {
                                    "title": "Source",
                                    "url": "https://example.com",
                                    "content": "Evidence",
                                }
                            ],
                        }
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

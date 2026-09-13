"""Read-from-search: page-context custody, replay, refusal and cost recording."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.web_search import read_tool
from local_operator.web_search.cost import SEARCH_SPEND
from local_operator.web_search.models import (
    SearchResponse,
    SearchSource,
    SearchUsage,
    WebSearchSettings,
)
from local_operator.web_search.pages import PAGE_CONTEXTS, PageContextStore

BLOCKS: list[dict[str, Any]] = [
    {"type": "thinking", "thinking": "…", "signature": "sig"},
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srv_1",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://example.com/a",
                "title": "A",
                "encrypted_content": "opaque-a",
            },
            {
                "type": "web_search_result",
                "url": "https://example.com/b",
                "title": "B",
                "encrypted_content": "opaque-b",
            },
        ],
    },
    {"type": "text", "text": "Search summary"},
]

SOURCES = [
    {"url": "https://example.com/a", "title": "A"},
    {"url": "https://example.com/b", "title": "B"},
]


@pytest.fixture(autouse=True)
def _clean_state():
    PAGE_CONTEXTS.reset()
    SEARCH_SPEND.reset()
    yield
    PAGE_CONTEXTS.reset()
    SEARCH_SPEND.reset()


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_store_keeps_blocks_verbatim_and_returns_a_context() -> None:
    store = PageContextStore()
    context = store.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)

    assert context is not None
    # Verbatim replay is what makes the pages readable; a "cleaned" block would
    # silently lose the encrypted payload.
    assert context.blocks == BLOCKS
    assert context.urls == ["https://example.com/a", "https://example.com/b"]


def test_store_refuses_a_search_with_no_result_blocks() -> None:
    store = PageContextStore()

    assert store.store(provider="duckduckgo", query="q", blocks=[], sources=[]) is None
    assert (
        store.store(
            provider="deepseek",
            query="q",
            blocks=[{"type": "web_search_tool_result", "content": []}],
            sources=[],
        )
        is None
    )


def test_contexts_are_session_scoped() -> None:
    """A read must never see another session's pages."""
    store = PageContextStore()
    context = store.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert context is not None
    store.attach("session-a", context.context_id)

    assert store.for_session("session-a") is not None
    assert store.for_session("session-b") is None
    assert store.for_session("") is None


def test_expired_contexts_are_not_served() -> None:
    store = PageContextStore(ttl_seconds=0.0)
    context = store.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert context is not None
    store.attach("s", context.context_id)

    # A stale page set answers from a world the session has moved past, so the
    # reader must fall back to a fetch rather than serve it.
    assert store.for_session("s") is None


def test_store_is_bounded_and_drops_session_references() -> None:
    store = PageContextStore(max_contexts=2)
    first = store.store(provider="deepseek", query="1", blocks=BLOCKS, sources=SOURCES)
    second = store.store(provider="deepseek", query="2", blocks=BLOCKS, sources=SOURCES)
    third = store.store(provider="deepseek", query="3", blocks=BLOCKS, sources=SOURCES)
    assert first and second and third
    store.attach("s", first.context_id)
    store.attach("s", third.context_id)

    assert store.get(first.context_id) is None
    assert store.for_session("s") is third
    assert store.get(third.context_id) is not None


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, status: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubClient:
    def __init__(self, response: _StubResponse) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def post(self, url: str, headers: dict | None = None, json: dict | None = None):
        self.requests.append({"url": url, "headers": headers or {}, "json": json or {}})
        return self.response


class _StubClientFactory:
    def __init__(self, stub: _StubClient) -> None:
        self.stub = stub

    async def __aenter__(self):
        return self.stub

    async def __aexit__(self, *exc) -> None:
        return None


def _patch_client(monkeypatch, stub: _StubClient) -> None:
    monkeypatch.setattr(read_tool, "_client", lambda _context: _StubClientFactory(stub))


def _context(session_id: str = "s1") -> ToolContext:
    return ToolContext(cwd=".", session_id=session_id)


async def _answer_payload(text: str, *, input_tokens: int = 200, output_tokens: int = 300):
    return _StubResponse(
        200,
        {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )


@pytest.mark.asyncio
async def test_read_refuses_actionably_when_no_pages_were_captured(monkeypatch) -> None:
    result = await read_tool.execute_web_read(
        "call-1", {"question": "What is the pricing?"}, None, None, _context()
    )

    assert result.is_error is True
    text = result.content[0].text
    assert "web_fetch" in text
    assert "DeepSeek" in text


@pytest.mark.asyncio
async def test_read_replays_blocks_verbatim_and_reports_cited_pages(monkeypatch) -> None:
    ctx = PAGE_CONTEXTS.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert ctx is not None
    PAGE_CONTEXTS.attach("s1", ctx.context_id)
    stub = _StubClient(
        await _answer_payload(
            "Agentic Disposition clears false positives.\nSOURCES: https://example.com/a"
        )
    )
    _patch_client(monkeypatch, stub)

    async def fake_key(_credentials):
        return "sk-test"

    monkeypatch.setattr(read_tool, "resolve_deepseek_key", fake_key)

    result = await read_tool.execute_web_read(
        "call-2", {"question": "What does Agentic Disposition do?"}, None, None, _context()
    )

    assert result.is_error is False
    body = stub.requests[0]["json"]
    messages = body["messages"]
    assert messages[0]["content"][0]["text"] == "Perform a web search for the query: q"
    assert messages[1]["content"] == BLOCKS
    assert "SOURCES:" not in result.content[0].text
    assert "Pages used: https://example.com/a" in result.content[0].text
    assert result.details["refused"] is False
    assert result.details["cited"] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_read_detects_the_refusal_marker_and_says_so(monkeypatch) -> None:
    ctx = PAGE_CONTEXTS.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert ctx is not None
    PAGE_CONTEXTS.attach("s1", ctx.context_id)
    stub = _StubClient(
        await _answer_payload("NOT IN PAGES These pages describe screening, not pricing.\nSOURCES:")
    )
    _patch_client(monkeypatch, stub)

    async def fake_key(_credentials):
        return "sk-test"

    monkeypatch.setattr(read_tool, "resolve_deepseek_key", fake_key)

    result = await read_tool.execute_web_read(
        "call-3", {"question": "pricing?"}, None, None, _context()
    )

    assert result.is_error is False
    assert result.details["refused"] is True
    # The user-visible text must not read like a finding.
    assert "do not answer this question" in result.content[0].text


@pytest.mark.asyncio
async def test_read_filters_invented_sources(monkeypatch) -> None:
    ctx = PAGE_CONTEXTS.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert ctx is not None
    PAGE_CONTEXTS.attach("s1", ctx.context_id)
    stub = _StubClient(
        await _answer_payload(
            "Something.\nSOURCES: https://example.com/a, https://invented.example/x"
        )
    )
    _patch_client(monkeypatch, stub)

    async def fake_key(_credentials):
        return "sk-test"

    monkeypatch.setattr(read_tool, "resolve_deepseek_key", fake_key)

    result = await read_tool.execute_web_read("call-4", {"question": "q"}, None, None, _context())

    assert result.details["cited"] == ["https://example.com/a"]
    assert "Ignored source(s)" in result.content[0].text


@pytest.mark.asyncio
async def test_read_records_spend_under_its_own_provider_key(monkeypatch) -> None:
    """A read is money, and it is not a search: both must stay visible."""
    ctx = PAGE_CONTEXTS.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert ctx is not None
    PAGE_CONTEXTS.attach("s1", ctx.context_id)
    _patch_client(monkeypatch, _StubClient(await _answer_payload("Answer.\nSOURCES:")))

    async def fake_key(_credentials):
        return "sk-test"

    monkeypatch.setattr(read_tool, "resolve_deepseek_key", fake_key)

    result = await read_tool.execute_web_read("call-5", {"question": "q"}, None, None, _context())

    totals = SEARCH_SPEND.session("s1")
    assert totals.searches == 1
    assert "deepseek:read" in totals.by_provider
    assert totals.usd > 0
    assert result.details["read_cost"]["usd"] == pytest.approx(totals.usd, abs=1e-6)


@pytest.mark.asyncio
async def test_read_reports_an_expired_context_as_a_refusal(monkeypatch) -> None:
    ctx = PAGE_CONTEXTS.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert ctx is not None
    PAGE_CONTEXTS.attach("s1", ctx.context_id)
    stub = _StubClient(_StubResponse(400, {"error": {"message": "invalid encrypted content"}}))
    _patch_client(monkeypatch, stub)

    async def fake_key(_credentials):
        return "sk-test"

    monkeypatch.setattr(read_tool, "resolve_deepseek_key", fake_key)

    result = await read_tool.execute_web_read("call-6", {"question": "q"}, None, None, _context())

    assert result.is_error is True
    assert "web_fetch" in result.content[0].text


@pytest.mark.asyncio
async def test_read_runs_a_search_first_when_asked(monkeypatch) -> None:
    """The one-call form pins DeepSeek ONLY when the session has it configured.

    Forcing a provider that is not in the session's chain raises before any
    search happens -- and that is the default install, where a DeepSeek model
    login makes ``provider_available`` true while ``web_search.providers`` holds
    duckduckgo/tavily/perplexity. Pinning on credential availability alone made
    every ``search=`` read fail with "provider 'deepseek' is disabled".
    """
    seen: dict[str, Any] = {}

    class StubService:
        def __init__(self, settings, credentials, **kwargs) -> None:
            pass

        async def search(self, query, *, limit=5, forced_provider=None):
            seen["query"] = query
            seen["forced"] = forced_provider
            ctx = PAGE_CONTEXTS.store(
                provider="deepseek", query=query, blocks=BLOCKS, sources=SOURCES
            )
            return SearchResponse(
                provider="deepseek",
                auth_mode="api-key",
                sources=[SearchSource(title="A", url="https://example.com/a")],
                usage=SearchUsage(input_tokens=1_000, output_tokens=100),
                page_context_id=ctx.context_id if ctx else None,
            )

    monkeypatch.setattr(read_tool, "WebSearchService", StubService)
    monkeypatch.setattr(read_tool, "provider_available", lambda *_args, **_kwargs: True)
    _patch_client(monkeypatch, _StubClient(await _answer_payload("Answer.\nSOURCES:")))

    async def fake_key(_credentials):
        return "sk-test"

    monkeypatch.setattr(read_tool, "resolve_deepseek_key", fake_key)

    # Configured + available: pin it, because only it captures readable pages.
    monkeypatch.setattr(
        read_tool,
        "load_read_settings",
        lambda _manager: WebSearchSettings(providers=["deepseek"]),
    )
    result = await read_tool.execute_web_read(
        "call-7", {"question": "q", "search": "minerva adverse media"}, None, None, _context()
    )
    assert seen == {"query": "minerva adverse media", "forced": "deepseek"}
    assert result.is_error is False

    # NOT configured: fall back to the session's own chain rather than forcing a
    # provider the chain does not contain.
    PAGE_CONTEXTS.reset()
    monkeypatch.setattr(
        read_tool,
        "load_read_settings",
        lambda _manager: WebSearchSettings(providers=["duckduckgo", "tavily"]),
    )
    seen.clear()
    result = await read_tool.execute_web_read(
        "call-8", {"question": "q", "search": "minerva adverse media"}, None, None, _context()
    )
    assert seen == {"query": "minerva adverse media", "forced": None}
    assert result.is_error is False


def test_read_tool_is_absent_when_search_or_reading_is_disabled() -> None:
    assert (
        read_tool.build_web_read_tool(ToolContext(cwd=".", web_search_settings={"enabled": False}))
        is None
    )
    assert (
        read_tool.build_web_read_tool(
            ToolContext(cwd=".", web_search_settings={"enabled": True, "read_enabled": False})
        )
        is None
    )
    tool = read_tool.build_web_read_tool(
        ToolContext(cwd=".", web_search_settings={"enabled": True})
    )
    assert tool is not None
    assert tool.name == "web_read"
    assert tool.approval_tier == "read"
    assert tool.concurrency == "shared"


@pytest.mark.asyncio
async def test_the_tool_dispatches_in_the_harness_order(monkeypatch) -> None:
    """The executor must be callable exactly as the loop calls it.

    ``ToolExecuteFn`` (harness/types.py) and the loop dispatch POSITIONALLY as
    ``(call.id, args, signal, on_update, context)``. An executor declared in any
    other order takes the signal as its context and fails at the session
    boundary -- while direct calls in tests, which pass what they named, still
    pass. This calls through the real tool object so the order is exercised the
    way a session exercises it, with a context carrying a session id that HAS
    captured pages: a swapped order cannot be mistaken for "no pages".
    """
    ctx = PAGE_CONTEXTS.store(provider="deepseek", query="q", blocks=BLOCKS, sources=SOURCES)
    assert ctx is not None
    PAGE_CONTEXTS.attach("s1", ctx.context_id)
    _patch_client(monkeypatch, _StubClient(await _answer_payload("Answer.\nSOURCES:")))

    async def fake_key(_credentials):
        return "sk-test"

    monkeypatch.setattr(read_tool, "resolve_deepseek_key", fake_key)
    tool = read_tool.build_web_read_tool(
        ToolContext(cwd=".", session_id="s1", web_search_settings={"enabled": True})
    )
    assert tool is not None

    # Exactly the loop's positional call shape.
    result = await tool.execute("call-9", {"question": "q"}, None, None, _context("s1"))

    assert result.is_error is False
    assert "Answer." in result.content[0].text

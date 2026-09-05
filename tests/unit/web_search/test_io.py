"""HTTP owner lifetime and subscriber cancellation are correctness contracts."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from local_operator.web_search.io import WebReadIO


@pytest.mark.asyncio
async def test_owner_reuses_only_same_origin_and_closes_all_clients():
    owner = WebReadIO()
    async with owner.client(("https", "one.example")) as first:
        pass
    async with owner.client(("https", "one.example")) as reused:
        assert reused is first
    async with owner.client(("https", "two.example")) as other:
        assert other is not first
    await owner.aclose()
    assert first.is_closed and other.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        async with owner.client(("https", "one.example")):
            pass


@pytest.mark.asyncio
async def test_pooling_does_not_add_cookie_session_state():
    seen = []

    def respond(request):
        seen.append(request.headers.get("cookie"))
        return httpx.Response(200, headers={"set-cookie": "secret=state; Path=/"})

    owner = WebReadIO()
    try:
        async with owner.client(("test",), transport=httpx.MockTransport(respond)) as client:
            await client.get("https://example.com/a")
            await client.get("https://example.com/b")
        assert seen == [None, None]
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_duplicate_reads_share_work_but_cancel_independently():
    owner = WebReadIO()
    entered = asyncio.Event()
    subscribed = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def read():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return "result"

    async def second():
        subscribed.set()
        return await owner.singleflight(("request",), read)

    first = asyncio.create_task(owner.singleflight(("request",), read))
    await entered.wait()
    other = asyncio.create_task(second())
    await subscribed.wait()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    assert not other.done()
    release.set()
    assert await other == "result"
    assert calls == 1
    # Sequential reads are fresh; this owner caches connections, not answers.
    assert await owner.singleflight(("request",), read) == "result"
    assert calls == 2
    await owner.aclose()


@pytest.mark.asyncio
async def test_final_subscriber_cancel_cancels_upstream():
    owner = WebReadIO()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def read():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    subscriber = asyncio.create_task(owner.singleflight(("request",), read))
    await entered.wait()
    subscriber.cancel()
    await asyncio.gather(subscriber, return_exceptions=True)
    assert cancelled.is_set()
    assert not owner._flights
    await owner.aclose()


@pytest.mark.asyncio
async def test_pool_bounds_idle_origins_without_closing_active_client():
    owner = WebReadIO()
    async with owner.client(("active",)) as active:
        for index in range(40):
            async with owner.client((index,)):
                pass
        assert len(owner._clients) <= 32
        assert not active.is_closed
    await owner.aclose()


@pytest.mark.asyncio
async def test_search_tool_collapses_only_matching_query_and_credentials(tmp_path, monkeypatch):
    from local_operator.harness.types import ToolContext
    from local_operator.web_search import tool
    from local_operator.web_search.models import (
        SearchResponse,
        SearchSource,
        WebSearchSettings,
    )
    from local_operator.web_search.service import WebSearchService

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("TAVILY_API_KEY", "first-test-value")
    monkeypatch.setattr(
        tool, "load_search_settings", lambda manager: WebSearchSettings(providers=["tavily"])
    )
    started, release, subscribed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def search(self, query, **kwargs):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return SearchResponse(
            provider="tavily",
            auth_mode="test",
            sources=[SearchSource(title="answer", url="https://example.com/")],
        )

    monkeypatch.setattr(WebSearchService, "search", search)
    owner = WebReadIO()
    context = ToolContext(cwd=str(tmp_path), web_io=owner)

    async def call(*, second=False):
        if second:
            subscribed.set()
        return await tool.execute_web_search("search", {"query": "same"}, context=context)

    try:
        first = asyncio.create_task(call())
        await started.wait()
        second = asyncio.create_task(call(second=True))
        await subscribed.wait()
        release.set()
        assert all(not result.is_error for result in await asyncio.gather(first, second))
        assert calls == 1

        # Recreate an overlap with a changed effective environment credential.
        # A new principal cannot join a request authorized by the prior value.
        started.clear()
        release.clear()
        subscribed.clear()
        first = asyncio.create_task(call())
        await started.wait()
        monkeypatch.setenv("TAVILY_API_KEY", "second-test-value")
        second = asyncio.create_task(call(second=True))
        await subscribed.wait()
        release.set()
        assert all(not result.is_error for result in await asyncio.gather(first, second))
        assert calls == 3
    finally:
        await owner.aclose()

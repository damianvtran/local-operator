"""Real service/tool wiring: shared reads preserve fetch policy and freshness."""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from local_operator.harness.types import ToolContext
from local_operator.web_fetch import service, tool
from local_operator.web_fetch.models import WebFetchSettings
from local_operator.web_search.io import WebReadIO


@pytest.mark.asyncio
async def test_pooled_fetch_partitions_virtual_hosts_and_validates_off_loop(monkeypatch):
    owner = WebReadIO()
    checks = []
    loop_thread = threading.get_ident()

    def resolve(host):
        checks.append((host, threading.get_ident()))
        return ["93.184.216.34"]

    monkeypatch.setattr(service, "_resolve_host_ips", resolve)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="body"))
    fetcher = service.WebFetchService(WebFetchSettings(enrich=False), io=owner, transport=transport)
    try:
        await fetcher.fetch("https://a.example/one")
        await fetcher.fetch("https://a.example/two")
        assert len(owner._clients) == 1
        await fetcher.fetch("https://b.example/three")
        assert len(owner._clients) == 2
        assert [host for host, _ in checks] == ["a.example", "a.example", "b.example"]
        assert all(thread != loop_thread for _, thread in checks)
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_duplicate_fetches_coalesce_and_refresh_stays_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(
        tool,
        "load_fetch_settings",
        lambda manager: WebFetchSettings(enrich=False, cache_ttl_seconds=0),
    )
    owner = WebReadIO()
    context = ToolContext(cwd=str(tmp_path), web_io=owner)
    entered, release, subscribed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, text="full article")

    transport = httpx.MockTransport(respond)

    async def fetch(*, second=False, refresh=False):
        if second:
            subscribed.set()
        return await tool.run_fetch(
            "https://example.com/article",
            tool_name="web_fetch",
            context=context,
            transport=transport,
            refresh=refresh,
        )

    try:
        first = asyncio.create_task(fetch())
        await entered.wait()
        second = asyncio.create_task(fetch(second=True))
        await subscribed.wait()
        release.set()
        outputs = await asyncio.gather(first, second)
        assert all(not output[2] for output in outputs)
        assert calls == 1
        await fetch(refresh=True)
        assert calls == 2
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_fetch_cache_cannot_survive_changed_private_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    settings = WebFetchSettings(enrich=False, allow_private=True)
    monkeypatch.setattr(tool, "load_fetch_settings", lambda manager: settings)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="private page"))
    first = await tool.run_fetch(
        "http://127.0.0.1/article", tool_name="web_fetch", transport=transport
    )
    assert not first[2]
    settings.allow_private = False
    second = await tool.run_fetch(
        "http://127.0.0.1/article", tool_name="web_fetch", transport=transport
    )
    assert second[2]
    assert "private/loopback/reserved" in second[0]

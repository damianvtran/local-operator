"""Tool-orchestration tests: spill/chunk-read, cache coupling, tool inventory.

These exercise the full ``run_fetch`` engine (fetch → render → spill → cache)
with an injected httpx MockTransport, so the spill store and cache index run for
real against an isolated config dir.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from local_operator.harness.types import ToolContext
from local_operator.tools import spill
from local_operator.tools.builtin import execute_read
from local_operator.tools.registry import create_tools
from local_operator.web_fetch import service, tool
from local_operator.web_fetch.tool import build_web_fetch_tool, run_fetch


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv(spill.SPILL_MAX_BYTES_ENV, raising=False)
    # Every SSRF check resolves to a fixed public address so the transport is the
    # only thing that decides an outcome (no live DNS in unit tests).
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    # Enrichment is disabled by pinning settings through the config default; tests
    # that want it exercise it explicitly.
    monkeypatch.setattr(
        service, "DEFAULT_WEB_FETCH_CONFIG", {**service.DEFAULT_WEB_FETCH_CONFIG, "enrich": False}
    )


@pytest.fixture
def context(tmp_path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="fetch-test")


class _CountingTransport(httpx.MockTransport):
    """A MockTransport that records how many requests it served.

    The cache-hit test asserts the SECOND fetch makes zero HTTP calls, which is
    only provable by counting real transport invocations.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.calls = 0

        def _counting(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            return handler(request)

        super().__init__(_counting)


def _html_page(n_lines: int) -> str:
    body = "".join(
        f"<p>Line number {i} of the article body content here.</p>" for i in range(n_lines)
    )
    return f"<html><body><h1>Big Page</h1>{body}</body></html>"


@pytest.mark.asyncio
async def test_large_page_spills_and_chunk_reads(context: ToolContext) -> None:
    """The central requirement: a large page returns a bounded preview plus a
    spill handle, and ``read spill://<d> range=`` / ``?q=`` resolve the full
    content."""
    page = _html_page(400)
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )
    preview, details, is_error = await run_fetch(
        "https://example.com/big", tool_name="web_fetch", context=context, transport=transport
    )
    assert is_error is False
    assert "spill" in details
    handle = details["spill"]["handle"]
    assert len(preview) < len(page)  # preview is bounded

    # Expand a range through the SAME read path any oversized output uses.
    ranged = await execute_read("t1", {"path": f"{handle}", "range": "1-5"}, None, None, context)
    assert "Line number 1" in ranged.text
    # Search within the spill.
    searched = await execute_read("t2", {"path": f"{handle}?q=Line number 42"}, None, None, context)
    assert "42" in searched.text


@pytest.mark.asyncio
async def test_cache_hit_makes_no_network_call(context: ToolContext) -> None:
    page = _html_page(50)
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )
    _p1, d1, _e1 = await run_fetch(
        "https://example.com/x", tool_name="web_fetch", context=context, transport=transport
    )
    assert d1["cache"] == "miss"
    first_calls = transport.calls
    assert first_calls >= 1

    _p2, d2, _e2 = await run_fetch(
        "https://example.com/x", tool_name="web_fetch", context=context, transport=transport
    )
    assert d2["cache"] == "hit"
    # Zero additional HTTP calls on the cache hit.
    assert transport.calls == first_calls


@pytest.mark.asyncio
async def test_refresh_bypasses_cache(context: ToolContext) -> None:
    page = _html_page(50)
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )
    await run_fetch(
        "https://example.com/x", tool_name="web_fetch", context=context, transport=transport
    )
    calls_after_first = transport.calls
    _p, d, _e = await run_fetch(
        "https://example.com/x",
        tool_name="web_fetch",
        context=context,
        transport=transport,
        refresh=True,
    )
    assert d["cache"] == "miss"
    assert transport.calls > calls_after_first


@pytest.mark.asyncio
async def test_cache_miss_when_spill_evicted(context: ToolContext) -> None:
    """If the spill entry is pruned between fetches, the cache degrades to a
    network fetch rather than returning a dead handle (design test 9)."""
    page = _html_page(50)
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )
    await run_fetch(
        "https://example.com/x", tool_name="web_fetch", context=context, transport=transport
    )
    calls_before = transport.calls

    # Evict all spilled content; the cache sidecar still points at the (now gone)
    # handle. The next fetch must NOT return that dead handle.
    spill.get_store().prune_all()

    _p, d, _e = await run_fetch(
        "https://example.com/x", tool_name="web_fetch", context=context, transport=transport
    )
    assert d["cache"] == "miss"
    assert transport.calls > calls_before


@pytest.mark.asyncio
async def test_json_body_pretty_printed(context: ToolContext) -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200, text='{"b":2,"a":1}', headers={"content-type": "application/json"}
        )
    )
    preview, details, _e = await run_fetch(
        "https://example.com/api", tool_name="web_fetch", context=context, transport=transport
    )
    assert details["render_method"] == "json"
    assert '"b": 2' in preview


@pytest.mark.asyncio
async def test_stdlib_backend_selected_by_config(context: ToolContext) -> None:
    """render_backend='stdlib' forces the fallback even when markdownify is
    present, and details reflect the method."""
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir

    manager = ConfigManager(config_dir())
    service.set_render_backend(manager, "stdlib")
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            text="<html><body><h1>H</h1><p>Body text long enough to render.</p></body></html>",
            headers={"content-type": "text/html"},
        )
    )
    _p, details, _e = await run_fetch(
        "https://example.com/", tool_name="web_fetch", context=context, transport=transport
    )
    assert details["render_method"] == "stdlib"


@pytest.mark.asyncio
async def test_ssrf_direct_returns_error_result(context: ToolContext, monkeypatch) -> None:
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["169.254.169.254"])
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text="x"))
    preview, _details, is_error = await run_fetch(
        "http://169.254.169.254/latest/meta-data/",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert is_error is True
    assert "private/loopback/reserved" in preview


# --- tool inventory / gate --------------------------------------------------


def test_web_fetch_in_default_inventory() -> None:
    tools = {t.name: t for t in create_tools(ToolContext(cwd="."))}
    assert "web_fetch" in tools
    assert tools["web_fetch"].approval_tier == "read"
    assert tools["web_fetch"].concurrency == "shared"
    assert tools["web_fetch"].interruptible is True


def test_master_switch_removes_tool() -> None:
    ctx = ToolContext(cwd=".", web_fetch_settings={"enabled": False})
    assert build_web_fetch_tool(ctx) is None
    assert "web_fetch" not in {t.name for t in create_tools(ctx)}


def test_params_forbid_extra() -> None:
    schema = tool.WebFetchParams.model_json_schema()
    assert schema.get("additionalProperties") is False


@pytest.mark.asyncio
async def test_enrichment_prefers_md_twin(context: ToolContext, monkeypatch) -> None:
    """A ``.md`` twin that yields clean markdown is preferred over scraping the
    HTML page (design test 16). Enrichment must be enabled for this path."""
    monkeypatch.setattr(
        service,
        "DEFAULT_WEB_FETCH_CONFIG",
        {**service.DEFAULT_WEB_FETCH_CONFIG, "enrich": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".md"):
            return httpx.Response(
                200,
                text=(
                    "# Clean Markdown\n\nThis is the enriched markdown twin content, "
                    "long enough to clear the substantiality gate that rejects a "
                    "too-short enrichment candidate in favour of the real page render."
                ),
                headers={"content-type": "text/markdown"},
            )
        if request.url.path == "/llms.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            text="<html><body><h1>Scraped</h1><p>HTML fallback body.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    preview, details, _e = await run_fetch(
        "https://example.com/docs/page",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert "Clean Markdown" in preview
    assert "Scraped" not in preview


@pytest.mark.asyncio
async def test_enrichment_falls_through_without_md(context: ToolContext, monkeypatch) -> None:
    """A site without a ``.md`` twin or llms.txt falls through to HTML render."""
    monkeypatch.setattr(
        service,
        "DEFAULT_WEB_FETCH_CONFIG",
        {**service.DEFAULT_WEB_FETCH_CONFIG, "enrich": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".md") or request.url.path == "/llms.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            text="<html><body><h1>Real Page</h1><p>The actual HTML content here.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    preview, _details, _e = await run_fetch(
        "https://example.com/docs/page",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert "Real Page" in preview

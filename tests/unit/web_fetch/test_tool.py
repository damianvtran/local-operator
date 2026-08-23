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
async def test_raw_and_rendered_do_not_share_cache_entry(context: ToolContext) -> None:
    """M3: a raw fetch and a rendered fetch of the SAME URL must not collide in
    the cache. The raw call stores verbatim source; the later rendered call must
    NOT get that raw HTML from cache — it renders to markdown."""
    html = "<html><body><h1>Heading</h1><p>Body paragraph with real content.</p></body></html>"
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text=html, headers={"content-type": "text/html"})
    )
    # raw=True stores the verbatim source under a raw-keyed entry.
    p_raw, d_raw, _e = await run_fetch(
        "https://example.com/p",
        tool_name="web_fetch",
        context=context,
        transport=transport,
        raw=True,
    )
    assert "<html>" in p_raw  # verbatim source
    assert d_raw["render_method"] == "text"

    # raw=False for the same URL must NOT hit the raw entry: it renders markdown.
    p_rendered, d_rendered, _e = await run_fetch(
        "https://example.com/p",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert d_rendered["cache"] == "miss"  # different variant → no collision
    assert d_rendered["render_method"] == "markdownify"
    assert "# Heading" in p_rendered
    assert "<html>" not in p_rendered

    # And a second rendered fetch DOES hit the rendered entry (variant is stable).
    _p3, d3, _e = await run_fetch(
        "https://example.com/p",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert d3["cache"] == "hit"


@pytest.mark.asyncio
async def test_non_2xx_not_cached_and_retried(context: ToolContext) -> None:
    """M4 + F1: a 503 is returned to the caller as an ERROR but NOT cached, so the
    next request re-hits the network instead of replaying the outage for the full
    TTL. The two paths (not-cached and not-successful-content) agree."""
    transport = _CountingTransport(
        lambda req: httpx.Response(503, text="<html><body>Service Unavailable</body></html>")
    )
    p1, d1, e1 = await run_fetch(
        "https://example.com/down", tool_name="web_fetch", context=context, transport=transport
    )
    assert d1["status"] == 503
    assert d1["cache"] == "miss"
    # F1: structurally an error, with the explicit flags the card branches on.
    assert e1 is True
    assert d1["http_error"] is True
    assert d1["ok"] is False
    # F1: the preview LEADS with the unmissable warning, not a benign status line.
    assert p1.startswith("⚠ HTTP 503 Service Unavailable")
    assert "not page content" in p1
    calls_after_first = transport.calls

    _p2, d2, _e = await run_fetch(
        "https://example.com/down", tool_name="web_fetch", context=context, transport=transport
    )
    # Not served from cache: the network was hit again (a retry could now succeed).
    assert d2["cache"] == "miss"
    assert transport.calls > calls_after_first


@pytest.mark.parametrize("status,reason", [(403, "Forbidden"), (404, "Not Found")])
@pytest.mark.asyncio
async def test_non_2xx_surfaced_as_error_not_content(
    context: ToolContext, status: int, reason: str
) -> None:
    """F1: a bot-block (403) or miss (404) returns is_error=True, leads with the
    prominent status line, and is NOT cached — so an agent doing research cannot
    mistake a block/error page for the requested content."""
    block_body = (
        "<html><body>Please enable JS and disable any ad blocker to continue.</body></html>"
    )
    transport = _CountingTransport(
        lambda req: httpx.Response(status, text=block_body, headers={"content-type": "text/html"})
    )
    preview, details, is_error = await run_fetch(
        "https://walled.example/page",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    # (a) structurally an error with explicit flags
    assert is_error is True
    assert details["http_error"] is True
    assert details["ok"] is False
    assert details["status"] == status
    # (b) leads with the prominent, unmissable status line
    assert preview.startswith(f"⚠ HTTP {status} {reason}")
    assert "error/block page, not page content" in preview
    assert "The body below is the error response" in preview
    # (c) not cached: a second fetch hits the network again
    calls = transport.calls
    _p2, d2, _e2 = await run_fetch(
        "https://walled.example/page",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert d2["cache"] == "miss"
    assert transport.calls > calls


@pytest.mark.asyncio
async def test_2xx_is_not_flagged_as_error(context: ToolContext) -> None:
    """F1 boundary: a normal 200 keeps ok=True/http_error=False and is_error
    False — the honest-failure signal must not fire on success."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            text="<html><body><h1>Real</h1><p>Genuine page content here.</p></body></html>",
            headers={"content-type": "text/html"},
        )
    )
    preview, details, is_error = await run_fetch(
        "https://ok.example/", tool_name="web_fetch", context=context, transport=transport
    )
    assert is_error is False
    assert details["ok"] is True
    assert details["http_error"] is False
    assert not preview.startswith("⚠")


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


@pytest.mark.asyncio
async def test_llms_txt_not_substituted_for_subpage(context: ToolContext, monkeypatch) -> None:
    """M2: for a SUBPAGE whose .md twin 404s, a substantial site-wide /llms.txt
    must NOT be returned as the result. The subpage's own HTML is rendered
    instead, so the agent never gets the site index attributed to a specific
    page it did not ask about."""
    monkeypatch.setattr(
        service,
        "DEFAULT_WEB_FETCH_CONFIG",
        {**service.DEFAULT_WEB_FETCH_CONFIG, "enrich": True},
    )
    llms_index = (
        "# Example Site Index\n\nThis is the site-wide llms.txt index listing "
        "every section of the documentation, which is the WRONG content to return "
        "for a specific subpage the agent requested by its own URL."
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".md"):
            return httpx.Response(404)  # no .md twin for this subpage
        if request.url.path == "/llms.txt":
            return httpx.Response(200, text=llms_index, headers={"content-type": "text/plain"})
        return httpx.Response(
            200,
            text="<html><body><h1>The Guide</h1><p>The guide's own body content.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    preview, details, _e = await run_fetch(
        "https://example.com/docs/guide",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    # The guide's own HTML is rendered; the site index is NOT substituted.
    assert "The Guide" in preview
    assert "Example Site Index" not in preview
    assert details["final_url"] == "https://example.com/docs/guide"


@pytest.mark.asyncio
async def test_llms_txt_used_for_site_root(context: ToolContext, monkeypatch) -> None:
    """M2 boundary: /llms.txt IS a legitimate enrichment win when the REQUESTED
    URL is the site root, where it genuinely represents the resource."""
    monkeypatch.setattr(
        service,
        "DEFAULT_WEB_FETCH_CONFIG",
        {**service.DEFAULT_WEB_FETCH_CONFIG, "enrich": True},
    )
    llms_index = (
        "# Example Site\n\nThe llms.txt index is the right representation for the "
        "site root, so it should be preferred over scraping the landing page HTML."
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/llms.txt":
            return httpx.Response(200, text=llms_index, headers={"content-type": "text/plain"})
        return httpx.Response(
            200,
            text="<html><body><h1>Landing</h1><p>Marketing splash.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    preview, _details, _e = await run_fetch(
        "https://example.com/",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert "Example Site" in preview

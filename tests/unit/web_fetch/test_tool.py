"""Tool-orchestration tests: spill/chunk-read, cache coupling, tool inventory.

These exercise the full ``run_fetch`` engine (fetch → render → spill → cache)
with an injected httpx MockTransport, so the spill store and cache index run for
real against an isolated config dir.
"""

from __future__ import annotations

import asyncio
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
    fresh_key = "https://example.com/x\x00raw=0|max_bytes=default"
    assert d1["supersede_key"] == fresh_key
    first_calls = transport.calls
    assert first_calls >= 1

    _p2, d2, _e2 = await run_fetch(
        "https://example.com/x", tool_name="web_fetch", context=context, transport=transport
    )
    assert d2["cache"] == "hit"
    # A hit is still a re-read of the same rendition. Pinning equality here
    # catches either call site dropping the variant as the result is shaped.
    assert d2["supersede_key"] == fresh_key
    assert d2["supersede_key"] == d1["supersede_key"]
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
    assert d_raw["supersede_key"] == "https://example.com/p\x00raw=1|max_bytes=default"

    # raw=False for the same URL must NOT hit the raw entry: it renders markdown.
    p_rendered, d_rendered, _e = await run_fetch(
        "https://example.com/p",
        tool_name="web_fetch",
        context=context,
        transport=transport,
    )
    assert d_rendered["cache"] == "miss"  # different variant → no collision
    assert d_rendered["render_method"] == "markdownify"
    assert d_rendered["supersede_key"] == "https://example.com/p\x00raw=0|max_bytes=default"
    assert d_rendered["supersede_key"] != d_raw["supersede_key"]
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
async def test_max_bytes_variants_emit_distinct_supersede_keys(context: ToolContext) -> None:
    """Byte caps change the returned content, so they name distinct resources."""
    page = _html_page(100)
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )

    _small_preview, small, _small_error = await run_fetch(
        "https://example.com/capped",
        tool_name="web_fetch",
        context=context,
        transport=transport,
        max_bytes=2048,
    )
    _large_preview, large, _large_error = await run_fetch(
        "https://example.com/capped",
        tool_name="web_fetch",
        context=context,
        transport=transport,
        max_bytes=4096,
    )

    assert small["supersede_key"] == "https://example.com/capped\x00raw=0|max_bytes=2048"
    assert large["supersede_key"] == "https://example.com/capped\x00raw=0|max_bytes=4096"
    assert small["supersede_key"] != large["supersede_key"]


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
    if status == 403:
        # A 403 is now classified: this body carries no vendor signature, so the
        # lead says the origin refused the request and names the ambiguity
        # instead of the generic "error/block page" wording. The F1 property
        # under test — an unmissable lead that cannot be mistaken for content —
        # is unchanged and asserted above; only the reason phrase is sharper.
        # The class also replaces the challenge body with an explanation, so the
        # "body below is the error response" note is deliberately absent (it
        # would describe the wrong thing).
        assert "the origin refused this request" in preview
        assert details["failure_kind"] == "blocked"
        assert details["suggested_tool"] == "browser"
    else:
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


@pytest.mark.asyncio
async def test_a_disabled_fetch_refuses_per_call_without_touching_the_network(
    context: ToolContext,
) -> None:
    """``web_fetch.enabled`` is LIVE: the file is re-read on EVERY call, so a
    tool still advertised (mid-turn, or in a subagent whose inventory is
    fixed at spawn) refuses the moment the switch is off. The transport
    counter is the proof no request left the process."""
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir

    service.set_fetch_enabled(ConfigManager(config_dir()), False)
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text="<p>x</p>", headers={"content-type": "text/html"})
    )
    preview, details, is_error = await run_fetch(
        "https://example.com/", tool_name="web_fetch", context=context, transport=transport
    )
    assert is_error is True
    assert preview == tool.WEB_FETCH_DISABLED_MESSAGE
    # The KEY, without its value: the key is what the user greps for, while
    # `: false` restates the condition they are living through and is the one
    # fragment that can go stale (design round 1, D4).
    assert "web_fetch.enabled" in preview
    assert "false" not in preview
    # NO details mapping (UX round 1, U3). `tool_card` renders any details
    # carrying a `url` as `Fetched: <url> · cache miss`, which told the user a
    # request had gone out for a call that never opened a connection — on
    # exactly the key a privacy- or airgap-minded user flips.
    assert not details
    assert transport.calls == 0

    # Flip it back on and the very next call fetches — no rebuild, no restart.
    service.set_fetch_enabled(ConfigManager(config_dir()), True)
    _p, _d, is_error = await run_fetch(
        "https://example.com/", tool_name="web_fetch", context=context, transport=transport
    )
    assert is_error is False
    assert transport.calls >= 1


# --- retries, cacheability, and the blocked result shape --------------------


CHALLENGE_HEADERS = {"content-type": "text/html", "server": "AkamaiGHost"}
#: The real shoppersdrugmart.ca body, entities intact (see test_failure.py).
AKAMAI_BODY = (
    "<HTML><HEAD>\n<TITLE>Access Denied</TITLE>\n</HEAD><BODY>\n"
    "<H1>Access Denied</H1>\n \n"
    "You don't have permission to access this server.<P>\n"
    "Reference&#32;&#35;18&#46;44182117&#46;1789250166&#46;2be4ae76\n</BODY>\n</HTML>"
)


def _cache_files() -> list[str]:
    try:
        return [p.name for p in service.cache_dir().iterdir() if p.suffix == ".json"]
    except OSError:
        return []


@pytest.mark.asyncio
async def test_repeated_5xx_is_never_cached_and_a_retried_200_is(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M4 still holds AFTER retries: a transient outage must never be replayed
    for the TTL, while a success that took two tries is an ordinary success.

    Asserted on the cache DIRECTORY rather than a mock, because "nothing was
    written" is a statement about the disk.
    """
    monkeypatch.setattr(service, "_backoff_sleep", lambda delay: asyncio.sleep(0))
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        # Count only the TARGET path: an enrichment probe (``/x.md``) is a real
        # request but not the one under test, and letting it consume a scripted
        # response would make the attempt numbers describe the optimisation
        # instead of the fetch.
        if not request.url.path.endswith("/x"):
            return httpx.Response(404, text="no probe here")
        state["n"] += 1
        if state["n"] <= 3:
            return httpx.Response(500, text="down")
        return httpx.Response(200, text="back up", headers={"content-type": "text/plain"})

    transport = _CountingTransport(handler)
    _p, details, is_error = await run_fetch(
        "https://flaky.example/x", tool_name="web_fetch", context=context, transport=transport
    )
    assert is_error is True
    assert details["attempts"] == 3
    assert _cache_files() == []  # three failures wrote nothing

    _p2, d2, e2 = await run_fetch(
        "https://flaky.example/x", tool_name="web_fetch", context=context, transport=transport
    )
    assert e2 is False
    assert d2["attempts"] == 1
    assert len(_cache_files()) == 1  # the success IS cached


@pytest.mark.asyncio
async def test_retried_success_is_cached_with_its_attempt_count(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 500-then-200 is a success that happened to cost two requests."""
    monkeypatch.setattr(service, "_backoff_sleep", lambda delay: asyncio.sleep(0))
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        # Only the target path is scripted; see the note in the test above.
        if not request.url.path.endswith("/x"):
            return httpx.Response(404, text="no probe here")
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, text="content here", headers={"content-type": "text/plain"})

    transport = _CountingTransport(handler)
    preview, details, is_error = await run_fetch(
        "https://slow.example/x", tool_name="web_fetch", context=context, transport=transport
    )
    assert is_error is False
    assert details["attempts"] == 2
    # The count rides in the header meta line, so a longer duration is legible
    # rather than looking like a slow network.
    assert "2 attempts" in preview
    assert len(_cache_files()) == 1


@pytest.mark.asyncio
async def test_cache_hit_reports_zero_attempts(context: ToolContext) -> None:
    """Truthful: a cache hit made no network attempt at all."""
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text="page", headers={"content-type": "text/plain"})
    )
    await run_fetch(
        "https://example.com/cached", tool_name="web_fetch", context=context, transport=transport
    )
    _p, details, _e = await run_fetch(
        "https://example.com/cached", tool_name="web_fetch", context=context, transport=transport
    )
    assert details["cache"] == "hit"
    assert details["attempts"] == 0


@pytest.mark.asyncio
async def test_blocked_result_drops_the_challenge_body_and_names_the_escalation(
    context: ToolContext,
) -> None:
    """§5.2: the challenge markup is REPLACED, not inlined.

    The body is unusable by construction — markup whose purpose is to be executed
    by a browser — and inlining it cost ~5.5 KB of context (measured on
    medium.com) to tell the agent nothing, under a lead that invited the misread
    that it was page content. What survives is the one durable fact (the origin's
    reference id) plus the named next step.
    """
    transport = _CountingTransport(
        lambda req: httpx.Response(403, text=AKAMAI_BODY, headers=CHALLENGE_HEADERS)
    )
    preview, details, is_error = await run_fetch(
        "https://walled.example/x", tool_name="web_fetch", context=context, transport=transport
    )

    assert is_error is True
    # The markup is gone.
    assert "Access Denied" not in preview
    assert "<html" not in preview.lower()
    # The useful parts are not.
    assert "18.44182117.1789250166.2be4ae76" in preview
    assert "browser" in preview
    assert "Akamai" in preview
    # And the escalation is data as well as prose, so a UI or a future caller
    # does not have to parse the sentence.
    assert details["failure_kind"] == "blocked"
    assert details["block_vendor"] == "akamai"
    assert details["block_reference"] == "18.44182117.1789250166.2be4ae76"
    assert details["suggested_tool"] == "browser"


@pytest.mark.asyncio
async def test_404_body_is_still_inlined(context: ToolContext) -> None:
    """The §5.2 narrowing applies to the ``blocked`` class ONLY: a 404's body,
    like a 451's and a 500's, often genuinely explains itself."""
    body = "<html><body><h1>Page moved</h1><p>See /guide/new-page instead.</p></body></html>"
    transport = _CountingTransport(
        lambda req: httpx.Response(404, text=body, headers={"content-type": "text/html"})
    )
    preview, details, is_error = await run_fetch(
        "https://docs.example.com/old", tool_name="web_fetch", context=context, transport=transport
    )
    assert is_error is True
    assert "Page moved" in preview
    assert "/guide/new-page" in preview
    assert details["failure_kind"] == "client"
    assert "suggested_tool" not in details


@pytest.mark.asyncio
async def test_existing_details_keys_keep_their_names_and_types(
    context: ToolContext,
) -> None:
    """§5.3 is ADDITIVE: the card, the UI row model and any stored transcript
    read these, so a rename or a retype here breaks a second repo silently."""
    transport = _CountingTransport(
        lambda req: httpx.Response(200, text="hello", headers={"content-type": "text/plain"})
    )
    _p, details, _e = await run_fetch(
        "https://example.com/x", tool_name="web_fetch", context=context, transport=transport
    )
    expected: dict[str, type] = {
        "url": str,
        "final_url": str,
        "status": int,
        "content_type": str,
        "render_method": str,
        "bytes": int,
        "complete": bool,
        "low_quality": bool,
        "cache": str,
        "ok": bool,
        "http_error": bool,
    }
    for key, kind in expected.items():
        assert key in details, f"{key} disappeared from details"
        assert isinstance(details[key], kind), f"{key} changed type"


@pytest.mark.asyncio
async def test_abort_during_a_backoff_returns_promptly(context: ToolContext) -> None:
    """§3.5: the backoff is a plain ``asyncio.sleep`` inside the raced coroutine,
    so an abort lands on it immediately rather than waiting the delay out.

    Structural, not timed: the signal is set while the sleep is in flight and the
    assertion is on the RESULT, so the test cannot flake under load.
    """
    from local_operator.harness.types import AbortSignal

    signal = AbortSignal()
    started = asyncio.Event()

    async def slow_backoff(delay: float) -> None:
        started.set()
        await asyncio.sleep(30)  # never completes; the abort must cut it short

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(service, "_backoff_sleep", slow_backoff)
        transport = _CountingTransport(lambda req: httpx.Response(503, text="down"))

        async def abort_when_sleeping() -> None:
            await started.wait()
            signal.abort()

        waiter = asyncio.create_task(abort_when_sleeping())
        preview, _details, is_error = await run_fetch(
            "https://example.com/x",
            tool_name="web_fetch",
            context=context,
            transport=transport,
            signal=signal,
        )
        await waiter

    assert is_error is True
    assert "aborted" in preview.lower()

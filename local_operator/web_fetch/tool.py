"""Model-facing ``web_fetch`` tool and the shared fetch orchestration.

:func:`run_fetch` is the ONE engine both doorways use — the ``web_fetch`` tool
and the ``read <url>`` sugar in :mod:`local_operator.tools.builtin` — so they
cannot drift into two pipelines. It wraps :class:`WebFetchService` with the
spill + cache layer:

- Full rendered content is written to the EXISTING spill store, exactly like any
  other oversized tool output, so the agent expands it through ``read`` with a
  range or ``?q=`` — no new expansion path, no new URL convention.
- A metadata-only URL cache points at that spill entry and self-checks against
  it (``stat``), so a cache hit within TTL returns with zero network calls and
  NEVER hands back a handle the spill store has already evicted.

The model-facing text is a bounded PREVIEW; the whole page lives behind the
``spill://`` handle. ``details`` never reaches the provider, so it costs no
tokens while still driving the TUI card and the transcript.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.config import ConfigManager
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.paths import config_dir
from local_operator.tools.builtin import spill_truncate
from local_operator.tools.spill import get_store
from local_operator.web_fetch.models import FetchResult
from local_operator.web_fetch.service import (
    CacheEntry,
    FetchError,
    WebFetchService,
    cache_variant,
    coerce_fetch_settings,
    load_fetch_settings,
    normalize_url,
    read_cache_entry,
    write_cache_entry,
)

#: The description the model reads. Points explicitly at ``web_fetch``/``read
#: <url>`` as the headless-safe verbs and reserves ``browser`` for the cases only
#: a real logged-in/JS session can serve — the whole reason this tool exists.
_DESCRIPTION = (
    "Fetch a web page or file over HTTP(S) once and make it readable without "
    "pulling the whole thing into context. Returns a short preview plus a "
    "`spill://` handle; expand it with `read` using a line range or `?q=<regex>` "
    "search, exactly like any other large output. Renders HTML to clean markdown, "
    "pretty-prints JSON, returns text as-is, and reports type and size for "
    "PDFs/binaries instead of inlining them. Re-fetching the same URL within the "
    "cache TTL reuses stored content with no network call. Use this (or "
    "`read <url>`) instead of `browser` for headless, subagent, and server "
    "contexts; use `browser` only when a page needs a real logged-in session or "
    "JavaScript rendering. Need several pages? Issue multiple web_fetch calls in "
    "one turn — they run in parallel instead of one-at-a-time."
)


class WebFetchParams(BaseModel):
    """Arguments accepted by the built-in fetch tool."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="Absolute http:// or https:// URL to fetch.", min_length=1)
    raw: bool = Field(
        default=False,
        description=(
            "Return the source verbatim (no HTML→markdown rendering, no JSON pretty-print)."
        ),
    )
    max_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Override the configured download ceiling for this call.",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Override the configured request timeout for this call.",
    )
    refresh: bool = Field(
        default=False,
        description="Bypass the cache and force a fresh network fetch.",
    )


def _result(
    tool_call_id: str,
    tool_name: str,
    text: str,
    *,
    error: bool = False,
    details: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text=text)],
        details=details,
        is_error=error,
    )


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


# Human-readable reason phrases for the statuses an agent most often hits on a
# bot-walled or broken page. Not exhaustive — an unlisted code falls back to a
# generic phrase — but the common blockers (403/429) and misses (404/5xx) get a
# name so the lead line reads as prose, not just a number.
_STATUS_REASONS: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    410: "Gone",
    429: "Too Many Requests",
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def _status_reason(status: int) -> str:
    """A short reason phrase for ``status`` (``"Forbidden"``), or a class name."""
    if status in _STATUS_REASONS:
        return _STATUS_REASONS[status]
    if 400 <= status < 500:
        return "Client Error"
    if 500 <= status < 600:
        return "Server Error"
    if 300 <= status < 400:
        return "Redirect"
    return "Unexpected Status"


def _header_line(result_like: dict[str, Any]) -> str:
    """The status line that leads a fetch preview.

    A 2xx leads with a compact ``[200] url`` + metadata line. A NON-2xx leads
    with an UNMISSABLE warning line (F1): live dogfooding showed bot-walled sites
    (Reddit/TripAdvisor 403s) whose block-page body rendered as if it were the
    requested content, and a benign ``[HTTP 403]`` lead was too easy for an agent
    to skim past. The warning states plainly that this is an error/block page, not
    the page content, so the agent's own judgement has a reliable signal. The
    rendered body still follows (it can carry a useful message) but the lead makes
    clear it is the error RESPONSE body, never the requested content.
    """
    status = int(result_like["status"])
    final = result_like["final_url"]
    ctype = result_like["content_type"]
    method = result_like["render_method"]
    cache = result_like["cache"]
    tail = f"{method} · {ctype} · cache {cache}"
    if 200 <= status < 300:
        quality = " · sparse/JS-gated (try `browser`)" if result_like.get("low_quality") else ""
        return f"[{status}] {final}\n{tail}{quality}"
    warn = (
        f"⚠ HTTP {status} {_status_reason(status)} — this is an error/block page, "
        f"not page content. {final}"
    )
    return f"{warn}\n{tail}\n(The body below is the error response, not the requested page.)"


def _shape_from_content(
    result_like: dict[str, Any],
    content: str,
    tool_name: str,
    context: ToolContext | None,
) -> tuple[str, dict[str, Any], str | None]:
    """Build ``(preview_text, details, spill_handle)`` from full ``content``.

    Two jobs kept deliberately distinct:

    - **Display shape** via :func:`spill_truncate` — the same footer, handle, and
      expansion path any oversized tool output gets. Content under the budget is
      returned inline with no footer (spill_truncate no-ops).
    - **A cache handle for EVERY fetch.** The full content is ALSO written to the
      store unconditionally, so even a small page that renders inline still has a
      stable ``spill://`` handle the URL cache can point at — otherwise re-reading
      a small page would always miss the cache and hit the network, defeating
      requirement #3. The store is content-addressed, so this write is idempotent
      and costs one entry, not a duplicate per fetch.

    The returned handle is the cache pointer (``None`` only when the store could
    not accept the content, e.g. a read-only config dir — then caching is simply
    skipped rather than pointing at nothing).
    """
    header = _header_line(result_like)
    body, spill_details = spill_truncate(content, tool_name, context)
    preview = f"{header}\n\n{body}" if body else header
    details = dict(result_like)
    details["lines"] = content.count("\n") + 1 if content else 0
    details["preview_chars"] = len(preview)
    if spill_details is not None:
        # Oversized: spill_truncate already wrote it and gave us the handle.
        details.update(spill_details)
        handle = spill_details["spill"].get("handle")
        return preview, details, handle
    # Small enough to show inline, but still persist it so the URL cache has a
    # handle to point at. Best-effort: a failed store write just means no cache.
    meta = get_store().write(
        content, tool_name=tool_name, session_id=(context.session_id if context else "") or ""
    )
    return preview, details, (meta.handle if meta is not None else None)


def _cache_hit_result(
    entry: CacheEntry,
    ttl_seconds: int,
    tool_name: str,
    context: ToolContext | None,
) -> tuple[str, dict[str, Any]] | None:
    """Rebuild a result from a cache entry, or ``None`` when the hit is unusable.

    A hit is usable only when it is within TTL AND its spill entry still stats —
    the coupling that guarantees the cache never returns a dead handle (design
    §8). A stale or spill-evicted entry returns ``None`` so the caller falls
    through to a fresh network fetch.
    """
    if ttl_seconds <= 0:
        return None
    age_ms = _now_ms() - entry.fetched_at_ms
    if age_ms > ttl_seconds * 1000:
        return None
    store = get_store()
    meta = store.stat(entry.spill_handle) if entry.spill_handle else None
    if meta is None:
        # The spill entry was evicted under its own LRU: treat as a miss rather
        # than hand back a handle that resolves to nothing.
        return None
    read = store.read_lines(entry.spill_handle, 1, None)
    if read is None:
        return None
    content = "\n".join(read[0])
    # Only 2xx responses are ever cached (M4), so a cache hit is always a
    # success — but carry the explicit ok/http_error flags anyway so the card
    # branches identically on cached and fresh results.
    http_ok = 200 <= entry.status < 300
    result_like = {
        "url": entry.url,
        "final_url": entry.final_url,
        "status": entry.status,
        "content_type": entry.content_type,
        "render_method": entry.render_method,
        "bytes": meta.bytes,
        "complete": entry.complete,
        "low_quality": entry.low_quality,
        "cache": "hit",
        "ok": http_ok,
        "http_error": not http_ok,
    }
    preview, details, _handle = _shape_from_content(result_like, content, tool_name, context)
    return preview, details


async def _fetch_or_abort(coro, signal: AbortSignal | None):
    """Race a fetch coroutine against the abort signal (mirrors web_search).

    A fetch is network-bound and must be abortable for steering. The coroutine
    is owned and reaped on every exit so a redirect chain cannot continue
    detached from its cancelled tool call.
    """
    if signal is None:
        return await coro
    if signal.aborted:
        if inspect.iscoroutine(coro):
            coro.close()
        raise asyncio.CancelledError(signal.reason or "aborted")
    fetch_task = asyncio.create_task(coro)
    abort_task = asyncio.create_task(signal.wait())
    try:
        done, _pending = await asyncio.wait(
            {fetch_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if abort_task in done:
            raise asyncio.CancelledError(signal.reason or "aborted")
        return await fetch_task
    finally:
        for task in (fetch_task, abort_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(fetch_task, abort_task, return_exceptions=True)


async def run_fetch(
    url: str,
    *,
    tool_name: str,
    raw: bool = False,
    max_bytes: int | None = None,
    timeout_seconds: float | None = None,
    refresh: bool = False,
    context: ToolContext | None = None,
    signal: AbortSignal | None = None,
    transport: Any = None,
) -> tuple[str, dict[str, Any], bool]:
    """Fetch → (cache lookup | network) → spill → cache. The shared engine.

    Returns ``(preview_text, details, is_error)``. Both ``web_fetch`` and the
    ``read <url>`` sugar call this so they return the identical shape; the caller
    only decides the ``tool_name`` recorded on the result and the card.
    """
    manager = ConfigManager(config_dir())
    settings = load_fetch_settings(manager)

    try:
        normalized = normalize_url(url)
    except FetchError as error:
        return str(error), {"url": url, "cache": "miss"}, True

    # The cache key folds in the render-affecting params (M3): a raw fetch and a
    # rendered fetch of the same URL are different renditions and must not share
    # an entry, and a byte-capped fetch must not satisfy a full-ceiling request.
    variant = cache_variant(raw=raw, max_bytes=max_bytes)

    # Cache lookup first (unless refreshing). A hit that still stats against the
    # spill store returns with zero network calls (requirement #3 / test 8).
    if not refresh and settings.cache_ttl_seconds > 0:
        entry = read_cache_entry(normalized, variant)
        if entry is not None:
            hit = _cache_hit_result(entry, settings.cache_ttl_seconds, tool_name, context)
            if hit is not None:
                preview, details = hit
                return preview, details, False

    # ``transport`` is a TEST-ONLY seam (httpx MockTransport): production callers
    # leave it None and the service opens real connections. Mirrors the
    # injectable transport WebSearchService uses for deterministic provider tests.
    service = WebFetchService(settings, transport=transport)
    try:
        result: FetchResult = await _fetch_or_abort(
            service.fetch(
                normalized,
                raw=raw,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
            ),
            signal,
        )
    except asyncio.CancelledError:
        return "web_fetch aborted.", {"url": normalized, "cache": "miss"}, True
    except FetchError as error:
        return str(error), {"url": normalized, "cache": "miss"}, True
    except Exception as error:  # pragma: no cover - defensive; unexpected transport bug
        return f"web_fetch failed for {normalized!r}: {error}", {"url": normalized}, True

    # ``ok``/``http_error`` are the explicit success booleans the card and any
    # downstream logic branch on without re-parsing the status int (F1). ``ok`` is
    # the friendly form; ``http_error`` is its negation, both carried so a reader
    # can use whichever reads better at the call site.
    http_ok = 200 <= result.status < 300
    result_like: dict[str, Any] = {
        "url": result.url,
        "final_url": result.final_url,
        "status": result.status,
        "content_type": result.content_type,
        "render_method": result.render_method,
        "bytes": result.bytes,
        "complete": result.complete,
        "low_quality": result.low_quality,
        "cache": "miss",
        "ok": http_ok,
        "http_error": not http_ok,
    }
    preview, details, handle = _shape_from_content(result_like, result.content, tool_name, context)

    # Record the cache entry pointing at the spill handle (metadata only). Only
    # 2xx responses are cached (M4): a 4xx/5xx cached for the full TTL would
    # replay a transient outage for 15 minutes with no retry, poisoning the URL —
    # a non-2xx result still returns to the caller (with its status leading the
    # preview) but is never stored, so the next request re-hits the network. A
    # binary notice is not worth caching (re-fetching a HEAD-sized notice is
    # cheap and the bytes were never stored), and a store write that failed
    # leaves ``handle`` None — skip caching rather than store a pointer to
    # nothing, which is exactly the dead-handle case test 9 guards against.
    cacheable = http_ok and result.render_method != "binary"
    if handle and settings.cache_ttl_seconds > 0 and cacheable:
        write_cache_entry(
            CacheEntry(
                url=normalized,
                final_url=result.final_url,
                spill_handle=str(handle),
                fetched_at_ms=_now_ms(),
                status=result.status,
                content_type=result.content_type,
                render_method=result.render_method,
                complete=result.complete,
                low_quality=result.low_quality,
            ),
            variant,
        )
    # F1: a non-2xx fetch returns as an ERROR result so it is structurally
    # distinct from a successful one — the model sees is_error and the card
    # renders the error treatment, closing the "block page read as content" gap
    # M4 only addressed for caching. The body still rides along (it can carry a
    # useful message) but is never presented as the requested page.
    return preview, details, not http_ok


async def execute_web_fetch(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Execute one fetch, respecting abort and the cache/spill layer."""
    try:
        params = WebFetchParams.model_validate(args)
    except ValidationError as error:
        return _result(
            tool_call_id, "web_fetch", f"Invalid web_fetch arguments: {error}", error=True
        )

    preview, details, is_error = await run_fetch(
        params.url,
        tool_name="web_fetch",
        raw=params.raw,
        max_bytes=params.max_bytes,
        timeout_seconds=params.timeout_seconds,
        refresh=params.refresh,
        context=context,
        signal=signal,
    )
    return _result(tool_call_id, "web_fetch", preview, error=is_error, details=details)


def build_web_fetch_tool(context: ToolContext | None = None) -> AgentTool | None:
    """Create the tool unless the startup configuration disables web fetch."""
    raw_settings = context.web_fetch_settings if context is not None else None
    if not coerce_fetch_settings(raw_settings).enabled:
        return None
    return AgentTool(
        name="web_fetch",
        label="Web Fetch",
        description=_DESCRIPTION,
        parameters=WebFetchParams.model_json_schema(),
        # Reads a remote resource and produces no local side effect beyond the
        # bounded spill/cache under config_dir(); the SSRF policy — not an
        # approval prompt on every call — is the guardrail (design §3, §15.2).
        approval_tier="read",
        concurrency="shared",
        interruptible=True,
        execute=execute_web_fetch,
    )

"""Model-facing ``web_search`` tool backed by the load-balancing service."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import uuid
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.paths import config_dir
from local_operator.tools.builtin import validation_error_result
from local_operator.web_search.models import (
    SearchProviderId,
    SearchResponse,
    WebSearchSettings,
)
from local_operator.web_search.providers import PROVIDERS, tavily_response_from_payload
from local_operator.web_search.service import (
    WebSearchService,
    coerce_search_settings,
    load_search_settings,
)

MODEL_CONTEXT_MAX_CHARS = 6_000
MODEL_ANSWER_MAX_CHARS = 1_200
MODEL_SNIPPET_MAX_CHARS = 320
MODEL_TITLE_MAX_CHARS = 240
MODEL_URL_MAX_CHARS = 2_048
_SOURCE_FOOTER = (
    "Snippets are intentionally capped. To read one result in full, use "
    "`web_fetch` (or `read <url>`) on its URL."
)
#: Used when a provider attached a judged relevance score. Measured on the
#: DeepSeek evidence pass, quotes verified verbatim against the live page in
#: about three quarters of checkable cases (two more were unverifiable: a 403 and
#: a JS-rendered page) -- so they are a fetch-priority signal, not citable page
#: text, and the wording must say so or a model will quote them as if they were.
_EVIDENCE_FOOTER = (
    "Relevance scores and quotes come from the provider's page-evidence pass and "
    "are model-reported: use them to choose what to fetch, and apply "
    "`web_fetch` (or `read <url>`) to a page before relying on its exact wording."
)


class WebSearchParams(BaseModel):
    """Arguments accepted by the built-in search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query.", min_length=1)
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return.")
    provider: SearchProviderId | None = Field(
        default=None,
        description="Optional enabled provider to use instead of load balancing.",
    )


def _result(
    tool_call_id: str,
    text: str,
    *,
    error: bool = False,
    details: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name="web_search",
        content=[TextContent(text=text)],
        details=details,
        is_error=error,
    )


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _hard_clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _omission_note(url_omitted: int, budget_omitted: int, *, more: bool) -> str:
    """One line saying how many sources were dropped AND why.

    The two causes are reported separately when both fired, because they point
    the model at different next actions and it cannot tell them apart from a
    bare count. ``more`` distinguishes "some results are shown, these are
    additional" from "nothing is shown at all".

    Previously this was one number with the cause hardcoded to the context
    limit, which was simply wrong for a URL omission; dropping the cause
    entirely fixed the inaccuracy and lost the actionability with it.
    """
    parts: list[str] = []
    if budget_omitted:
        parts.append(f"{budget_omitted} by the context limit")
    if url_omitted:
        parts.append(f"{url_omitted} for an unusable URL")
    total = url_omitted + budget_omitted
    lead = f"{total} {'more ' if more else ''}result{'s' if total != 1 else ''} omitted"
    return f"{lead} ({', '.join(parts)})" if parts else lead


def _render_response(response: SearchResponse) -> tuple[str, int]:
    """Bounded model context, and HOW MANY sources it left out.

    The count is returned rather than re-derived by the caller: it is known
    exactly here, and the only other way to recover it was scanning the
    rendered text for the word "omitted" — which a result snippet containing
    that word would have flipped on an untruncated response.
    """
    sections = [f"Provider: {response.provider} ({response.auth_mode})"]
    if response.answer:
        sections.append(_clip(response.answer, MODEL_ANSWER_MAX_CHARS))

    # One footer for the whole response: a mixed set of sources would otherwise
    # suggest some snippets are page text and others are not.
    # Follows the SNIPPET, not the score: see ``SearchResponse.evidence_applied``.
    footer = _EVIDENCE_FOOTER if response.evidence_applied else _SOURCE_FOOTER

    failures = ""
    if response.failures:
        failures = "Fallbacks: " + _clip("; ".join(response.failures), 600)

    source_blocks: list[str] = []
    # The two causes are counted apart because they imply OPPOSITE next moves
    # for the model: a budget omission means "your query matched too much,
    # narrow it", an unusable-URL omission means "that one result cannot be
    # fetched, the query was fine". Collapsing them into one number threw that
    # away, and the footer below tells the model to call `browser` with a URL
    # the response may have just declined to supply.
    omitted_url = 0
    omitted_budget = 0
    for source_position, source in enumerate(response.sources):
        # A clipped URL is not actionable. Omit the whole candidate instead so
        # every URL that reaches model context can be passed to browser verbatim.
        if len(source.url) > MODEL_URL_MAX_CHARS:
            omitted_url += 1
            continue
        index = len(source_blocks) + 1
        title = _clip(source.title, MODEL_TITLE_MAX_CHARS)
        # A judged relevance is a fetch-priority signal, so it sits on the title
        # line where the model reads it before choosing a URL. Providers that do
        # not judge pages leave it None and render exactly as before.
        heading = (
            f"{index}. [relevance {source.relevance}/100] {title}"
            if source.relevance is not None
            else f"{index}. {title}"
        )
        lines = [heading, f"   {source.url}"]
        if source.snippet:
            lines.append(f"   {_clip(source.snippet, MODEL_SNIPPET_MAX_CHARS)}")
        block = "\n".join(lines)
        trial_sources = "Sources:\n" + "\n".join([*source_blocks, block])
        tail = "\n\n".join(part for part in (failures, footer) if part)
        trial = "\n\n".join([*sections, trial_sources, tail])
        # Reserve enough room for the explicit omission marker so the hard cap
        # never silently cuts a URL or leaves a half-result in model context.
        if len(trial) + 80 > MODEL_CONTEXT_MAX_CHARS:
            omitted_budget += len(response.sources) - source_position
            break
        source_blocks.append(block)

    omitted = omitted_url + omitted_budget
    if source_blocks:
        sources = "Sources:\n" + "\n".join(source_blocks)
        if omitted:
            sources += f"\n… {_omission_note(omitted_url, omitted_budget, more=True)}"
        sections.append(sources)
    elif response.sources:
        sections.append(f"… {_omission_note(omitted_url, omitted_budget, more=False)}")
    if failures:
        sections.append(failures)
    sections.append(footer)
    rendered = "\n\n".join(sections)
    # The source-block budget above should make this unreachable, but the cap is
    # an invariant at the model boundary even if future sections are added.
    return _hard_clip(rendered, MODEL_CONTEXT_MAX_CHARS), omitted


async def _search_or_abort(service_call, signal: AbortSignal | None):
    if signal is None:
        return await service_call
    if signal.aborted:
        # The caller constructs the provider coroutine before handing it over.
        # Close an unscheduled coroutine or Python will warn at collection time.
        if inspect.iscoroutine(service_call):
            service_call.close()
        raise asyncio.CancelledError(signal.reason or "aborted")
    search_task = asyncio.create_task(service_call)
    abort_task = asyncio.create_task(signal.wait())
    try:
        done, _pending = await asyncio.wait(
            {search_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if abort_task in done:
            raise asyncio.CancelledError(signal.reason or "aborted")
        return await search_task
    finally:
        # Immediate steering cancels this coroutine itself rather than setting
        # AbortSignal. Own and reap both children on every exit so provider or
        # delegated MCP I/O can never continue detached from its tool call.
        for task in (search_task, abort_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(search_task, abort_task, return_exceptions=True)


def _parse_tavily_mcp_text(text: str) -> dict[str, Any]:
    """Parse the official Tavily MCP's unescaped ``formatResults`` text."""
    answer: list[str] = []
    results: list[dict[str, str]] = []
    current: dict[str, str] = {}
    in_results = False
    phase: str | None = None
    previous_blank = False

    def finish() -> None:
        nonlocal current
        if current.get("url"):
            content = current.get("content")
            if content is not None:
                current["content"] = content.strip()
            results.append(current)
        current = {}

    for raw_line in text.replace("\r\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            if phase == "content" and current.get("content"):
                current["content"] += "\n"
            previous_blank = True
            continue
        if line == "Detailed Results:":
            in_results = True
            phase = None
        elif not in_results and line.startswith("Answer:"):
            answer.append(line.removeprefix("Answer:").strip())
            phase = "answer"
        elif not in_results and phase == "answer":
            answer.append(line)
        elif in_results and previous_blank and line.startswith("Title:"):
            # Official records begin with a blank line before Title. Content is
            # unescaped, so Title:/URL: text inside an ordinary snippet must
            # remain content rather than becoming a forged source record.
            finish()
            current["title"] = line.removeprefix("Title:").strip()
            phase = "metadata"
        elif phase == "metadata" and line.startswith("ID:"):
            pass
        elif phase == "metadata" and line.startswith("URL:"):
            current["url"] = line.removeprefix("URL:").strip()
        elif phase == "metadata" and line.startswith("Content:"):
            current["content"] = line.removeprefix("Content:").strip()
            phase = "content"
        elif phase == "content":
            current["content"] = " ".join(part for part in (current.get("content"), line) if part)
        previous_blank = False
    finish()
    if not results:
        raise RuntimeError("Tavily OAuth MCP returned no parseable results")
    return {"answer": " ".join(answer).strip() or None, "results": results}


def _tavily_mcp_payload(result: ToolResult) -> dict[str, Any]:
    """Normalize official prose plus structured payloads from other servers."""
    server_result = (result.details or {}).get("server_result")
    if isinstance(server_result, dict):
        structured = server_result.get("structuredContent")
        if not isinstance(structured, dict):
            structured = server_result.get("structured_content")
        if isinstance(structured, dict):
            return structured
    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError:
        return _parse_tavily_mcp_text(result.text)
    if not isinstance(payload, dict):
        raise RuntimeError("Tavily OAuth MCP returned a non-object response")
    return payload


def _tavily_oauth_delegate(
    context: ToolContext | None,
    signal: AbortSignal | None,
    on_update: Callable[[AgentToolUpdate], None] | None,
):
    """Reuse the session's connected Tavily MCP tool when OAuth is configured."""
    if context is None:
        return None
    tool = context.delegated_tools.get("mcp__tavily_search")
    if not isinstance(tool, AgentTool):
        return None

    async def search(query: str, limit: int) -> SearchResponse:
        result = await tool.execute(
            f"web-search-tavily-{uuid.uuid4().hex}",
            {"query": query, "max_results": limit, "search_depth": "basic"},
            signal,
            on_update,
            context,
        )
        if result.is_error:
            raise RuntimeError(result.text or "Tavily OAuth MCP search failed")
        payload = _tavily_mcp_payload(result)
        return tavily_response_from_payload(payload, auth_mode="oauth-mcp", limit=limit)

    return search


#: The refusal a call gets while ``web_search.enabled`` is false. Names the
#: key and the page that flips it, so the model (and the human reading the
#: card) can tell a policy refusal from a provider outage.
#:
#: The KEY without its VALUE (design round 1, D4). The key is the thing the
#: user greps for in ``config.yml``; ``: false`` only restates the condition
#: they are already living through, and being hardcoded rather than read from
#: the value that triggered the refusal, it is the one fragment that can go
#: stale. Kept byte-parallel with ``WEB_FETCH_DISABLED_MESSAGE``.
WEB_SEARCH_DISABLED_MESSAGE = (
    "web_search is disabled by config (web_search.enabled) — " "turn it on in /settings › Web tools"
)


def _search_digest_key(
    service: "WebSearchService", settings: WebSearchSettings, params: "WebSearchParams"
) -> tuple[str, str, str, str, int, str | None]:
    """The singleflight key for one search call.

    Only a digest of the EXPORTED credential values behind the resolved chain
    scopes duplicate work, so two calls with different credentials do not
    coalesce. Store rows are deliberately NOT digested: they resolve identically
    for every consumer of the same config root, so they cannot vary between two
    calls in one process; the environment is the only per-call input. The
    plaintext ``credentials.env`` leg this used also to read is GONE (PR2a).

    The RESOLVED, non-rotating chain, not ``settings.providers``: an
    auto-joined provider's credential (EXA_API_KEY, PARALLEL_API_KEY,
    DEEPSEEK_API_KEY) sits outside the priority prefix, and two calls with
    different credentials behind it must not coalesce. ``resolve()`` rather than
    ``candidates()`` because the rotation offset moves per call and the key must
    be stable.

    A NAMED function rather than an inline block so the plaintext sweep
    (``tests/unit/secrets/test_no_reader_resolves_the_plaintext_file.py``) can
    prove the removed leg's value reaches no digest: a removal no callable
    reaches is removal nothing exercises.
    """
    auth = hashlib.sha256(
        json.dumps(
            {
                key: os.environ.get(key, "")
                for provider in service.resolve()
                for key in PROVIDERS[provider].credential_keys
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return (
        "search",
        settings.model_dump_json(),
        auth,
        params.query.strip(),
        params.max_results,
        params.provider,
    )


async def execute_web_search(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Execute one configured search, respecting abort and provider fallback."""
    try:
        params = WebSearchParams.model_validate(args)
    except ValidationError as error:
        # Shared with the builtins rather than hand-built here: the fault
        # classification must not depend on WHICH tool the model picked, or
        # `validity` acquires a hidden dependence on tool identity.
        return validation_error_result(tool_call_id, "web_search", error)

    manager = ConfigManager(config_dir())
    settings = load_search_settings(manager)
    # The master switch is re-checked PER CALL, not only at inventory build.
    # ``web_search.enabled`` is LIVE: a top-level session drops the tool from
    # its schema at the next turn boundary, but the model may already hold a
    # call against the old schema, and a subagent keeps its spawn inventory
    # for its whole life — so this line is what makes "disabled" true from the
    # moment the file changes. Same per-call manager the other knobs already
    # use, so no new parse cost and no new malformed-file exposure.
    if not settings.enabled:
        return _result(tool_call_id, WEB_SEARCH_DISABLED_MESSAGE, error=True)
    credentials = CredentialManager.readonly(config_dir())
    service = WebSearchService(
        settings,
        credentials,
        tavily_oauth_search=_tavily_oauth_delegate(context, signal, on_update),
        io=context.web_io if context is not None else None,
    )

    async def search() -> SearchResponse:
        return await service.search(
            params.query, limit=params.max_results, forced_provider=params.provider
        )

    try:
        # Credentials are resolved anew for each invocation. Only a digest
        # scopes duplicate work; no secret values enter keys or diagnostics.
        # OAuth delegates capture per-call signals/events, so they retain
        # independent calls rather than borrowing another caller's lifecycle.
        io = context.web_io if context is not None else None
        if io is not None and service.tavily_oauth_search is None:
            key = _search_digest_key(service, settings, params)
            work = io.singleflight(key, search)
        else:
            work = search()
        response = await _search_or_abort(
            work,
            signal,
        )
    except asyncio.CancelledError:
        return _result(tool_call_id, "Web search aborted.", error=True)
    except Exception as error:
        return _result(tool_call_id, str(error), error=True)

    text, omitted = _render_response(response)
    details = response.model_dump(mode="json")
    details["context_chars"] = len(text)

    # Search spend is its own cost line: model-token accounting never saw it, and
    # a search-heavy session can spend more on retrieval than on generation. The
    # ledger is session-keyed, so /session can show this session's total and
    # /analytics can show the cross-session one, with per-provider detail.
    from local_operator.web_search.cost import SEARCH_SPEND
    from local_operator.web_search.pages import PAGE_CONTEXTS

    session_id = context.session_id if context is not None else ""
    entry = SEARCH_SPEND.record(session_id, response.provider, response.cost)
    session_totals = SEARCH_SPEND.session(session_id)
    details["search_cost"] = {
        "usd": response.cost.usd if response.cost else None,
        "basis": response.cost.basis if response.cost else "",
        "priced_from_usage": bool(response.cost and response.cost.priced_from_usage),
        "session_usd": round(session_totals.usd, 6),
        "session_searches": session_totals.searches,
        "provider_searches": None if entry is None else entry.searches,
    }

    # Hand the captured page context to THIS session, so `web_read` can answer
    # from these pages without a fetch. Attaching is per-session by design: the
    # pages a search retrieved belong to the session that asked for them.
    PAGE_CONTEXTS.attach(session_id, response.page_context_id)
    details["context_max_chars"] = MODEL_CONTEXT_MAX_CHARS
    details["context_truncated"] = omitted > 0
    return _result(tool_call_id, text, details=details)


def build_web_search_tool(context: ToolContext | None = None) -> AgentTool | None:
    """Create the tool unless the startup configuration disables web search."""
    raw_settings = context.web_search_settings if context is not None else None
    if not coerce_search_settings(raw_settings).enabled:
        return None
    return AgentTool(
        name="web_search",
        label="Web Search",
        description=(
            "Search the public web with load balancing and automatic fallback. "
            "Results include bounded snippets and source URLs; use `web_fetch` (or "
            "`read <url>`) to read a result's full page. Use provider only for a "
            "specific enabled source. Running several independent searches? Issue "
            "the calls in one turn — they run in parallel."
        ),
        parameters=WebSearchParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=True,
        execute=execute_web_search,
    )

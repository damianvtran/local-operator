"""Model-facing ``web_read`` tool: answer from pages a search already retrieved.

A search through the ``deepseek`` provider captures the retrieved pages as
replayable blocks (:mod:`local_operator.web_search.pages`). Those pages can then
be interrogated WITHOUT fetching them over the network and WITHOUT billing a new
search: the replay restores them into the auxiliary model's context server-side,
and the answers come back as text. A third turn on the same pages keeps working
(measured), so one search can serve several questions.

WHY THIS IS NOT A "DEEPSEEK SESSION" FEATURE
--------------------------------------------
The obvious objection is that the page text only exists on DeepSeek's side, so
this could only help when the session itself runs on DeepSeek. It is the
opposite: the replay runs as an AUXILIARY call on the DeepSeek endpoint, exactly
like the search itself, and only distilled text crosses back into the harness.
The session's own model never needs to hold the pages, so the feature behaves
identically for every session model -- DeepSeek, Claude, GPT, a local Ollama
model. The permutations that DO matter are on the retrieval side, and each one
has a defined behaviour:

* **Session model is anything at all** -> supported. The reader is a separate
  DeepSeek call; the session model only reads the returned text.
* **Search ran through a non-DeepSeek provider** (DuckDuckGo, Tavily, Brave,
  Exa, SerpApi, Perplexity, SearXNG) -> no page payload was captured, so there is
  nothing to replay. The tool says so and points at ``web_fetch`` rather than
  quietly fetching on the caller's behalf: a silent fetch would cost the network
  round trip this tool exists to avoid, and would hide that the cheaper path was
  unavailable.
* **The DeepSeek key/balance is gone** (the provider is unavailable) -> same
  refusal. A read is a BILLED turn, so it must not be attempted on an account
  that cannot pay: the provider's own gate applies before the request.
* **The captured context expired or was evicted** -> same refusal. Contexts are
  bounded and TTL'd; a stale page set is worse than a fetch because it answers
  from a world the session has moved past.
* **A search is requested up front** (``search`` given) -> the tool runs one
  itself, preferring the DeepSeek provider so the read has pages to work with,
  and then reads. This is the one-call form for a fresh question.
* **Subagents** -> each session reads only the contexts IT captured, so a child
  cannot answer from the parent's pages (or vice versa). Pages a search retrieved
  belong to the session that asked for them.

The refusal contract matters as much as the mechanism: the auxiliary model is
told to answer ONLY from the captured pages and to reply with a fixed marker when
the pages do not contain the answer. Without that, an answer-from-pages feature
becomes a confident-sounding confabulation channel, which is strictly worse than
the fetch it replaced.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
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
from local_operator.web_search.cost import SEARCH_SPEND, estimate_search_cost
from local_operator.web_search.models import SearchUsage, WebSearchSettings
from local_operator.web_search.pages import PAGE_CONTEXTS, PageContext
from local_operator.web_search.providers import (
    DEEPSEEK_API_VERSION,
    DEEPSEEK_SEARCH_ENDPOINT,
    DEEPSEEK_SEARCH_MODEL,
    provider_available,
    resolve_deepseek_key,
)
from local_operator.web_search.service import WebSearchService, coerce_search_settings

MODEL_ANSWER_MAX_CHARS = 4_000
MODEL_SOURCES_MAX_CHARS = 600

#: The exact phrase the auxiliary model must use when the pages do not answer the
#: question. Fixed so the harness can detect it and say plainly that the pages
#: were silent, instead of presenting a hedge as a finding.
NOT_IN_PAGES = "NOT IN PAGES"

#: The search-spend ledger key a read is recorded under. Named once: the writer,
#: the tool-result details and the resume path all have to agree on it, and the
#: round-2 review caught them disagreeing.
READ_LEDGER_PROVIDER = "deepseek:read"

#: The reading instruction. Two constraints carry the whole feature's honesty:
#: answer only from the pages above (no priors), and refuse in a detectable way.
#: The source list is requested as a trailing machine-readable line so provenance
#: survives without asking the model to emit JSON around prose that may contain
#: newlines, quotes and code.
_READ_INSTRUCTION = (
    "Answer the question using ONLY the pages retrieved above. Do not use your "
    "own knowledge, and do not guess: if those pages do not contain the answer, "
    f"reply with exactly {NOT_IN_PAGES} followed by one short sentence naming "
    "what is missing.\n"
    "Otherwise answer in at most 200 words, and quote sparingly.\n"
    "End with one final line in exactly this form, listing only pages you "
    "actually used:\n"
    "SOURCES: <url>, <url>\n"
    f"Never invent a URL that is not among the retrieved pages."
)


class WebReadParams(BaseModel):
    """Arguments accepted by the built-in read-from-search tool."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="What to answer from the retrieved pages.", min_length=1)
    search: str | None = Field(
        default=None,
        description=(
            "Optional query to run first when this session has no recent search " "pages to read."
        ),
    )
    urls: list[str] = Field(
        default_factory=list,
        description="Optional subset of page URLs to answer from.",
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
        tool_name="web_read",
        content=[TextContent(text=text)],
        details=details,
        is_error=error,
    )


def _prompt_for(context: PageContext) -> str:
    """The user turn the search used, so the replayed turn is reproducible.

    The replay must be the SAME conversation that produced the blocks, or the
    restored pages and the follow-up question would belong to two different
    searches.
    """
    return f"Perform a web search for the query: {context.query}"


def _build_messages(context: PageContext, params: WebReadParams) -> list[dict[str, Any]]:
    question = params.question.strip()
    if params.urls:
        wanted = ", ".join(url.strip() for url in params.urls if url.strip())
        if wanted:
            question = f"{question}\n\nAnswer only from these pages: {wanted}"
    return [
        {"role": "user", "content": [{"type": "text", "text": _prompt_for(context)}]},
        # Verbatim: the opaque encrypted_content in these blocks is what makes
        # DeepSeek restore the pages. Rewriting them invalidates the request.
        {"role": "assistant", "content": context.blocks},
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{question}\n\n{_READ_INSTRUCTION}"}],
        },
    ]


def _split_answer(text: str) -> tuple[str, list[str]]:
    """Split the model's prose from its trailing ``SOURCES:`` line."""
    answer_lines: list[str] = []
    urls: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SOURCES:"):
            for chunk in stripped.split(":", 1)[1].split(","):
                candidate = chunk.strip().strip("<>[]`")
                if candidate.startswith("http"):
                    urls.append(candidate)
            continue
        answer_lines.append(line)
    return "\n".join(answer_lines).strip(), urls


def _usage_from(payload: object) -> SearchUsage:
    if not isinstance(payload, dict):
        return SearchUsage()
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return SearchUsage()

    def as_int(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) else None

    return SearchUsage(
        input_tokens=as_int("input_tokens"),
        output_tokens=as_int("output_tokens"),
        cache_read_tokens=as_int("cache_read_input_tokens"),
    )


async def _abortable(call, signal: AbortSignal | None):
    if signal is None:
        return await call
    if signal.aborted:
        if inspect.iscoroutine(call):
            call.close()
        raise asyncio.CancelledError(signal.reason or "aborted")
    task = asyncio.create_task(call)
    abort = asyncio.create_task(signal.wait())
    try:
        done, _pending = await asyncio.wait({task, abort}, return_when=asyncio.FIRST_COMPLETED)
        if abort in done:
            task.cancel()
            # AWAIT the cancellation rather than abandoning it. An un-awaited
            # cancelled task can surface later as "Task exception was never
            # retrieved", and any exception the request had already raised (a
            # 4xx, a transport error) would be swallowed by nobody. We are
            # raising the abort either way, so suppressing here loses nothing.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise asyncio.CancelledError(signal.reason or "aborted")
        return task.result()
    finally:
        if not abort.done():
            abort.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await abort


async def execute_web_read(
    tool_call_id: str,
    params: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Answer a question from pages a previous search captured.

    The parameter ORDER is the harness's (``ToolExecuteFn``,
    ``harness/types.py``): the loop dispatches positionally as
    ``(call.id, args, signal, on_update, context)``. An executor declared in any
    other order is not merely mis-typed -- it receives an ``AbortSignal`` where
    it expects a context and fails at the session boundary, while direct calls in
    tests still pass. Ordering it the harness way is the only version that runs.
    """
    try:
        parsed = WebReadParams.model_validate(params)
    except ValidationError as error:
        return validation_error_result(tool_call_id, "web_read", error)

    manager = ConfigManager(config_dir())
    settings: WebSearchSettings = load_read_settings(manager)
    if not settings.enabled or not settings.read_enabled:
        return _result(tool_call_id, "Web reading is disabled in settings.", error=True)
    credentials = CredentialManager(config_dir())

    session_id = context.session_id if context is not None else ""
    page_context = PAGE_CONTEXTS.for_session(session_id)

    # One-call form: search first, so a fresh question does not need the model to
    # remember to search and then read. DeepSeek is preferred because only its
    # payload captures readable pages; the chain is the fallback so a session
    # without the DeepSeek provider still gets an honest failure rather than a
    # hang.
    if page_context is None and parsed.search:
        # Only pin the provider when the session actually has it CONFIGURED.
        # ``provider_available`` answers whether a credential exists, not whether
        # the provider is in the session's chain, and forcing an unconfigured one
        # fails the whole call -- on a default install (duckduckgo, tavily,
        # perplexity) a DeepSeek model login would make every `search=` read fail
        # with "provider 'deepseek' is disabled". Falling back to the configured
        # chain is honest: if it captures no pages, the refusal below says so.
        forced = (
            "deepseek"
            if "deepseek" in settings.providers
            and provider_available("deepseek", credentials, settings)
            else None
        )
        service = WebSearchService(
            settings,
            credentials,
            io=context.web_io if context is not None else None,
        )
        try:
            response = await _abortable(
                service.search(parsed.search, limit=10, forced_provider=forced), signal
            )
        except asyncio.CancelledError:
            return _result(tool_call_id, "Web read aborted.", error=True)
        except Exception as error:
            return _result(tool_call_id, f"Search before read failed: {error}", error=True)
        SEARCH_SPEND.record(session_id, response.provider, response.cost)
        PAGE_CONTEXTS.attach(session_id, response.page_context_id)
        page_context = PAGE_CONTEXTS.for_session(session_id)

    if page_context is None:
        return _result(
            tool_call_id,
            (
                "No readable pages from a previous search in this session. "
                "Reading works only on pages a `web_search` through the DeepSeek "
                "provider retrieved (its results carry the page payload; other "
                "providers return links only), and the captured pages expire. "
                "Use `web_fetch` (or `read <url>`) on the URLs you need, or pass "
                "`search` to run a search first."
            ),
            error=True,
            details={"page_context": None},
        )

    key = await resolve_deepseek_key(credentials)
    if not key:
        return _result(
            tool_call_id,
            "Reading captured pages needs the DeepSeek API key; run "
            "`local-operator login deepseek`, or fetch the pages with `web_fetch`.",
            error=True,
        )

    async with _client(context) as client:

        async def call() -> tuple[int, dict[str, Any]]:
            response = await client.post(
                DEEPSEEK_SEARCH_ENDPOINT,
                headers={
                    "x-api-key": key,
                    "authorization": f"Bearer {key}",
                    "anthropic-version": DEEPSEEK_API_VERSION,
                    "content-type": "application/json",
                    "accept": "application/json",
                },
                json={
                    "model": DEEPSEEK_SEARCH_MODEL,
                    "max_tokens": 2_048,
                    "messages": _build_messages(page_context, parsed),
                },
            )
            try:
                return response.status_code, response.json()
            except Exception:
                return response.status_code, {"raw": response.text[:400]}

        try:
            status, payload = await _abortable(call(), signal)
        except asyncio.CancelledError:
            # The turn may have been accepted and billed before the abort landed,
            # and we cannot see its usage because we stopped reading the
            # response. Recorded as an UNPRICED read rather than left out
            # entirely: a suspected charge the ledger never mentions is how a
            # total quietly stops being a total. ``None`` (not 0.0) keeps the
            # module's rule that unknown and free are different facts.
            SEARCH_SPEND.record(
                str(getattr(context, "session_id", "") or ""),
                READ_LEDGER_PROVIDER,
                None,
                kind="read",
            )
            return _result(tool_call_id, "Web read aborted.", error=True)
        except Exception as error:
            return _result(tool_call_id, f"Web read failed: {error}", error=True)

    if status != 200:
        detail = payload.get("error") or payload.get("raw") or status
        return _result(
            tool_call_id,
            f"DeepSeek refused the page replay (HTTP {status}): {detail}. "
            "The captured pages may have expired; use `web_fetch` on the URLs.",
            error=True,
        )

    text = "\n".join(
        str(block.get("text") or "")
        for block in (payload.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    answer, urls = _split_answer(text)
    usage = _usage_from(payload)
    cost = estimate_search_cost("deepseek", usage)
    # A read is spend, and it is not a search: it is recorded under its own
    # provider key so the counts stay truthful while the money still lands in the
    # session's search-spend total.
    # ``kind="read"``: this is the ledger write for a SUCCESSFUL read, and the
    # one that matters. Without it the read is booked as a search -- the row
    # renders "1 search" and the session's search count is inflated by reads,
    # which is the opposite of why the key carries a ``:read`` suffix at all.
    entry = SEARCH_SPEND.record(session_id, READ_LEDGER_PROVIDER, cost, searches=1, kind="read")
    session_totals = SEARCH_SPEND.session(session_id)

    refused = answer.strip().upper().startswith(NOT_IN_PAGES)
    known = set(page_context.urls)
    cited = [url for url in urls if url in known] if known else urls
    unknown = [url for url in urls if known and url not in known]

    header = (
        f"Read {len(page_context.sources)} retrieved pages from a {page_context.provider} search."
    )
    body = answer or "(the model returned no text)"
    sections = [header, body]
    if refused:
        sections.append(
            "The captured pages do not answer this question. Fetch a specific page "
            "with `web_fetch` (or `read <url>`) if you need to look further."
        )
    if cited:
        sections.append("Pages used: " + ", ".join(cited)[:MODEL_SOURCES_MAX_CHARS])
    if unknown:
        sections.append(
            "Ignored source(s) not among the retrieved pages: "
            + ", ".join(unknown)[:MODEL_SOURCES_MAX_CHARS]
        )
    rendered = "\n\n".join(section for section in sections if section)
    if len(rendered) > MODEL_ANSWER_MAX_CHARS:
        rendered = rendered[: MODEL_ANSWER_MAX_CHARS - 1].rstrip() + "…"

    details = {
        "page_context_id": page_context.context_id,
        "pages": len(page_context.sources),
        "refused": refused,
        "cited": cited,
        "read_cost": {
            # The LEDGER KEY, stated by the writer that chose it: a resume that
            # re-derives it from a ``provider`` field this result does not carry
            # labelled the restored row ``:read`` rather than ``deepseek:read``.
            "ledger_provider": READ_LEDGER_PROVIDER,
            "usd": cost.usd,
            "basis": cost.basis,
            "session_usd": round(session_totals.usd, 6),
            "session_searches": session_totals.searches,
            "session_reads": session_totals.reads,
            "reads": entry.reads,
        },
        "usage": usage.model_dump(mode="json"),
    }
    return _result(tool_call_id, rendered, details=details)


class _client:
    """A pooled client when the session owns one, else a short-lived one."""

    def __init__(self, context: ToolContext | None) -> None:
        self._owner = getattr(context, "web_io", None) if context is not None else None
        self._client = None

    async def __aenter__(self):
        if self._owner is not None:
            self._pooled = self._owner.client(("web_read", 20.0), timeout=20.0)
            self._client = await self._pooled.__aenter__()
            return self._client
        import httpx

        self._own = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._client = self._own
        return self._client

    async def __aexit__(self, *exc) -> None:
        if self._owner is not None:
            await self._pooled.__aexit__(*exc)
        else:
            await self._own.aclose()


def load_read_settings(manager: ConfigManager) -> WebSearchSettings:
    """Search settings, which carry the reader's own switch."""
    from local_operator.web_search.service import load_search_settings

    return load_search_settings(manager)


def build_web_read_tool(context: ToolContext | None = None) -> AgentTool | None:
    """Create the tool unless search (or the reader) is disabled."""
    raw_settings = context.web_search_settings if context is not None else None
    settings = coerce_search_settings(raw_settings)
    if not settings.enabled or not settings.read_enabled:
        return None
    return AgentTool(
        name="web_read",
        label="Web Read",
        description=(
            "Answer a question from the pages a PREVIOUS `web_search` already "
            "retrieved, instead of fetching them again. No network fetch and no "
            "new search is billed; the pages are re-read by the search provider, "
            "so this works whatever model this session runs. Use it to pull "
            "details, quotes or comparisons out of pages you already found, and "
            "`web_fetch` when you need a page that was never in a search result. "
            "If the pages do not answer the question it says so rather than "
            "guessing. Pass `search` to run a search first when this session has "
            "no recent search pages."
        ),
        parameters=WebReadParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=True,
        execute=execute_web_read,
    )

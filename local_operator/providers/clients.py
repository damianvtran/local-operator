"""httpx wire clients streaming provider SSE into harness ``StreamEvent``s.

Four clients over the harness contract:

- :class:`OpenAICompatClient` — ``{base}/chat/completions`` (covers openai,
  openrouter, deepseek, kimi, alibaba, mistral, xai, ollama, radient).
- :class:`AnthropicClient` — ``/v1/messages`` with cache-control breakpoints
  on system blocks.
- :class:`GoogleClient` — ``generateContent`` / ``streamGenerateContent``
  (minimal).
- :class:`MockClient`` — deterministic canned events for ``--hosting test``.

All accept an injected ``httpx.AsyncClient`` so tests can use
``httpx.MockTransport`` without touching the network. Error mapping raises
:class:`~local_operator.providers.failover.ProviderError` with status/
retryable/auth flags for the failover layer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

from local_operator.harness.types import (
    ChatRequest,
    ImageContent,
    Message,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    Usage,
)
from local_operator.providers.failover import ProviderError


@runtime_checkable
class WireClient(Protocol):
    """The one method the harness needs from a provider client."""

    async def stream(self, request: ChatRequest, api_key: str | None) -> AsyncIterator[StreamEvent]:
        """Stream one completion. Raises :class:`ProviderError` on failure.

        Must be an async generator (``stream(...)`` called then iterated).
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_retry_after(response: httpx.Response) -> int | None:
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return int(float(header) * 1000)
    except ValueError:
        return None


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message", error))
        if isinstance(error, str):
            return error
        message = payload.get("message")
        if message:
            return str(message)
    return response.text[:500]


def _raise_for_status(response: httpx.Response) -> None:
    """Map HTTP errors onto ProviderError with failover-relevant flags."""
    status = response.status_code
    if status < 400:
        return
    message = _extract_error_message(response)
    auth_error = status in (401, 403)
    retryable = status == 429 or status >= 500 or status == 408
    raise ProviderError(
        status,
        message,
        retryable=retryable,
        retry_after_ms=_parse_retry_after(response),
        auth_error=auth_error,
    )


def _iter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield decoded ``data:`` payloads from an SSE byte stream."""

    async def _gen() -> AsyncIterator[str]:
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        yield data

    return _gen()


def _message_to_openai(message: Message) -> dict[str, Any]:
    """Render one harness message into OpenAI chat-completions shape."""
    if message.role == "assistant" and message.tool_calls:
        text = message.text
        entry: dict[str, Any] = {"role": "assistant"}
        if text:
            entry["content"] = text
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    if call.raw_arguments is not None
                    else json.dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
        return entry
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": _tool_content_openai(message),
        }
    parts: list[dict[str, Any]] = []
    plain_only = True
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            plain_only = False
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    role = message.role
    if plain_only:
        return {"role": role, "content": "".join(p["text"] for p in parts)}
    return {"role": role, "content": parts}


EMPTY_TOOL_RESULT_TEXT = "[tool returned no output]"


def _tool_content_openai(message: Message) -> str | list[dict[str, Any]]:
    """Render a tool result from its content blocks — never ``message.text``.

    Flattening via ``.text`` drops image-only results to ``""``; render text
    blocks as text and image blocks as data-URL ``image_url`` parts. An empty
    result is backfilled so providers never receive empty content.
    """
    parts: list[dict[str, Any]] = []
    has_image = False
    for block in message.content:
        if isinstance(block, TextContent):
            if block.text:
                parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            has_image = True
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    if not parts:
        return EMPTY_TOOL_RESULT_TEXT
    if not has_image:
        return "".join(part["text"] for part in parts)
    return parts


def _tools_to_openai(tools: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


_FINISH_TO_STOP_REASON = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "toolUse",
    "function_call": "toolUse",
    "content_filter": "stop",
}


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAICompatClient:
    """``POST {base_url}/chat/completions`` with SSE streaming.

    Tool-call deltas are assembled by ``index`` (name/id arrive on the first
    chunk, arguments stream in pieces). Usage comes from the final chunk
    (``stream_options={"include_usage": true}``), including
    ``prompt_tokens_details.cached_tokens``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._extra_headers = dict(extra_headers or {})
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_body(self, request: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model.model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                *[({"role": "system", "content": block}) for block in request.system_blocks],
                *[_message_to_openai(m) for m in request.messages],
            ],
        }
        if request.tools:
            body["tools"] = _tools_to_openai(request.tools)
            body["tool_choice"] = {"auto": "auto", "none": "none", "required": "required"}[
                request.tool_choice
            ]
        max_tokens = request.max_tokens or request.model.max_output_tokens
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = max_tokens
        temperature = request.temperature if request.temperature is not None else request.model.temperature
        body["temperature"] = temperature
        top_p = request.top_p if request.top_p is not None else request.model.top_p
        body["top_p"] = top_p
        if request.stop_sequences:
            body["stop"] = list(request.stop_sequences)
        return body

    async def stream(self, request: ChatRequest, api_key: str | None) -> AsyncIterator[StreamEvent]:
        url = f"{self._base_url}/chat/completions"
        finish_reason: str | None = None
        usage: Usage | None = None
        provider_payload: dict[str, Any] | None = None
        # index -> (id, name); arguments stream as deltas keyed by index.
        call_ids: dict[int, str] = {}
        call_names: dict[int, str] = {}

        async with self._http.stream(
            "POST", url, json=self._build_body(request), headers=self._headers(api_key)
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_status(response)
            async for data in _iter_sse_lines(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("usage"), Mapping):
                    raw = chunk["usage"]
                    details = raw.get("prompt_tokens_details") or {}
                    usage = Usage(
                        input_tokens=int(raw.get("prompt_tokens", 0)),
                        output_tokens=int(raw.get("completion_tokens", 0)),
                        cache_read_tokens=int(details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0),
                    )
                    yield StreamUsageEvent(usage=usage)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    yield StreamTextDelta(delta=text)
                for tool_delta in delta.get("tool_calls") or []:
                    index = int(tool_delta.get("index", 0))
                    function = tool_delta.get("function") or {}
                    call_id = tool_delta.get("id")
                    name = function.get("name")
                    if call_id:
                        call_ids[index] = call_id
                        yield StreamToolCallDelta(index=index, id=call_id)
                    if name:
                        call_names[index] = name
                        yield StreamToolCallDelta(index=index, name=name)
                    argument_delta = function.get("arguments")
                    if argument_delta:
                        yield StreamToolCallDelta(index=index, argument_delta=argument_delta)
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                if chunk.get("id") or chunk.get("system_fingerprint"):
                    provider_payload = {
                        "id": chunk.get("id"),
                        "system_fingerprint": chunk.get("system_fingerprint"),
                    }

        yield StreamEndEvent(
            stop_reason=_FINISH_TO_STOP_REASON.get(finish_reason or "", finish_reason or "stop"),
            usage=usage,
            provider_payload=provider_payload,
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient:
    """``POST {base}/v1/messages`` streaming.

    System blocks are sent as an array with ``cache_control: {type:
    "ephemeral"}`` on every block EXCEPT the last (omp breakpoint policy:
    the volatile tail stays un-cached). Content arrives as
    ``content_block_start/delta/stop`` events for ``text`` and ``tool_use``
    blocks; tool arguments stream via ``input_json_delta``.
    """

    def __init__(
        self,
        base_url: str = ANTHROPIC_API_URL,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"anthropic-version": ANTHROPIC_VERSION, "Content-Type": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    @staticmethod
    def _system_blocks(blocks: Sequence[str]) -> list[dict[str, Any]]:
        """System blocks → Anthropic ``system`` array with cache breakpoints.

        The harness sends [instructions, tool inventory, skills, env/date];
        the trailing blocks are VOLATILE (skills change per turn, env/date
        changes per day) and must stay breakpoint-free so the prompt-cache
        prefix covers only the stable head. Generic for any block count:
        every block except the last two gets an ephemeral breakpoint.
        """
        rendered: list[dict[str, Any]] = []
        stable_count = max(0, len(blocks) - 2)
        for index, block in enumerate(blocks):
            entry: dict[str, Any] = {"type": "text", "text": block}
            if index < stable_count:
                entry["cache_control"] = {"type": "ephemeral"}
            rendered.append(entry)
        return rendered

    @staticmethod
    def _message_blocks(message: Message) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.mime_type,
                            "data": block.data,
                        },
                    }
                )
        return blocks

    @staticmethod
    def _tool_result_blocks(message: Message) -> list[dict[str, Any]]:
        """Render a tool result's content as Anthropic blocks.

        Uses the message's content blocks (never ``message.text``) so
        image-only results survive; an empty result is backfilled because
        Anthropic 400s on empty ``tool_result`` content.
        """
        blocks = AnthropicClient._message_blocks(message)
        if not blocks:
            return [{"type": "text", "text": EMPTY_TOOL_RESULT_TEXT}]
        return blocks

    def _build_body(self, request: ChatRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "assistant" and message.tool_calls:
                content = self._message_blocks(message)
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments
                        if call.raw_arguments is None
                        else json.loads(call.raw_arguments or "{}"),
                    }
                    for call in message.tool_calls
                )
                messages.append({"role": "assistant", "content": content})
            elif message.role == "tool":
                # Anthropic groups tool results under one user message.
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": self._tool_result_blocks(message),
                        **({"is_error": True} if message.is_error else {}),
                    }
                ]
                if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
                    messages[-1]["content"].extend(content)
                else:
                    messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": message.role, "content": self._message_blocks(message)})

        body: dict[str, Any] = {
            "model": request.model.model_id,
            "stream": True,
            "messages": messages,
            "max_tokens": request.max_tokens or request.model.max_output_tokens,
        }
        if request.system_blocks:
            body["system"] = self._system_blocks(request.system_blocks)
        if request.tools:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters or {"type": "object", "properties": {}},
                }
                for tool in request.tools
            ]
            body["tool_choice"] = {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}[
                request.tool_choice
            ]
        temperature = request.temperature if request.temperature is not None else request.model.temperature
        body["temperature"] = temperature
        top_p = request.top_p if request.top_p is not None else request.model.top_p
        body["top_p"] = top_p
        if request.stop_sequences:
            body["stop_sequences"] = list(request.stop_sequences)
        return body

    async def stream(self, request: ChatRequest, api_key: str | None) -> AsyncIterator[StreamEvent]:
        url = f"{self._base_url}/v1/messages"
        stop_reason = "stop"
        usage = Usage()
        block_index_to_call: dict[int, tuple[str, str]] = {}

        async with self._http.stream(
            "POST", url, json=self._build_body(request), headers=self._headers(api_key)
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_status(response)
            async for data in _iter_sse_lines(response):
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "message_start":
                    raw_usage = (event.get("message") or {}).get("usage") or {}
                    usage.input_tokens = int(raw_usage.get("input_tokens", usage.input_tokens))
                    usage.cache_read_tokens = int(raw_usage.get("cache_read_input_tokens", 0))
                    usage.cache_write_tokens = int(raw_usage.get("cache_creation_input_tokens", 0))
                elif event_type == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        index = int(event.get("index", 0))
                        block_index_to_call[index] = (block.get("id", ""), block.get("name", ""))
                        yield StreamToolCallDelta(index=index, id=block.get("id"), name=block.get("name"))
                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        text = delta.get("text")
                        if text:
                            yield StreamTextDelta(delta=text)
                    elif delta_type == "input_json_delta":
                        index = int(event.get("index", 0))
                        partial = delta.get("partial_json")
                        if partial:
                            yield StreamToolCallDelta(index=index, argument_delta=partial)
                elif event_type == "message_delta":
                    delta = event.get("delta") or {}
                    if delta.get("stop_reason"):
                        stop_reason = str(delta["stop_reason"])
                    raw_usage = event.get("usage") or {}
                    if "output_tokens" in raw_usage:
                        usage.output_tokens = int(raw_usage["output_tokens"])
                elif event_type == "error":
                    error = event.get("error") or {}
                    raise ProviderError(None, str(error.get("message", error)), retryable=True)

        mapped = {"end_turn": "stop", "max_tokens": "length", "tool_use": "toolUse", "stop_sequence": "stop"}.get(
            stop_reason, stop_reason
        )
        yield StreamUsageEvent(usage=usage)
        yield StreamEndEvent(stop_reason=mapped, usage=usage)


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

GOOGLE_API_URL = "https://generativelanguage.googleapis.com"


class GoogleClient:
    """Minimal Gemini client: ``streamGenerateContent?alt=sse``."""

    def __init__(
        self,
        base_url: str = GOOGLE_API_URL,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    def _build_body(self, request: ChatRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        if request.system_blocks:
            contents.append(
                {"role": "user", "parts": [{"text": "\n\n".join(request.system_blocks)}]}
            )
        for message in request.messages:
            role = "user" if message.role == "user" else "model"
            parts: list[dict[str, Any]] = [{"text": message.text}] if message.text else []
            for block in message.content:
                if isinstance(block, ImageContent):
                    parts.append(
                        {"inline_data": {"mime_type": block.mime_type, "data": block.data}}
                    )
            if message.role == "assistant" and message.tool_calls:
                parts.extend(
                    {"functionCall": {"name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                )
            if message.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.tool_name or "",
                                    "response": {"content": message.text},
                                }
                            }
                        ],
                    }
                )
                continue
            if parts or message.role == "user":
                contents.append({"role": role, "parts": parts or [{"text": ""}]})

        body: dict[str, Any] = {"contents": contents}
        generation_config: dict[str, Any] = {}
        max_tokens = request.max_tokens or request.model.max_output_tokens
        if max_tokens and max_tokens > 0:
            generation_config["maxOutputTokens"] = max_tokens
        generation_config["temperature"] = (
            request.temperature if request.temperature is not None else request.model.temperature
        )
        generation_config["topP"] = request.top_p if request.top_p is not None else request.model.top_p
        if request.stop_sequences:
            generation_config["stopSequences"] = list(request.stop_sequences)
        body["generationConfig"] = generation_config
        if request.tools:
            body["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters or {"type": "object", "properties": {}},
                        }
                        for tool in request.tools
                    ]
                }
            ]
        return body

    async def stream(self, request: ChatRequest, api_key: str | None) -> AsyncIterator[StreamEvent]:
        url = (
            f"{self._base_url}/v1beta/models/{request.model.model_id}:streamGenerateContent?alt=sse"
        )
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key
        usage: Usage | None = None
        stop_reason = "stop"

        async with self._http.stream(
            "POST", url, json=self._build_body(request), headers=headers
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_status(response)
            async for data in _iter_sse_lines(response):
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for candidate in chunk.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        text = part.get("text")
                        if text:
                            yield StreamTextDelta(delta=text)
                        function_call = part.get("functionCall")
                        if function_call:
                            yield StreamToolCallDelta(
                                index=0,
                                id=f"fc_{function_call.get('name', 'call')}",
                                name=function_call.get("name"),
                                argument_delta=json.dumps(function_call.get("args") or {}),
                            )
                    if candidate.get("finishReason"):
                        reason = str(candidate["finishReason"])
                        stop_reason = {"MAX_TOKENS": "length", "TOOL_USE": "toolUse"}.get(reason, "stop")
                raw_usage = chunk.get("usageMetadata")
                if raw_usage:
                    usage = Usage(
                        input_tokens=int(raw_usage.get("promptTokenCount", 0)),
                        output_tokens=int(raw_usage.get("candidatesTokenCount", 0)),
                        cache_read_tokens=int(raw_usage.get("cachedContentTokenCount", 0)),
                    )

        if usage is not None:
            yield StreamUsageEvent(usage=usage)
        yield StreamEndEvent(stop_reason=stop_reason, usage=usage)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockClient:
    """Deterministic canned stream for ``--hosting test``.

    Emits two text deltas + usage + end; when the last user message contains
    ``[tool]`` it emits one tool call (``echo`` with ``{"text": "hi"}``) and
    stops with ``toolUse`` instead.
    """

    async def stream(self, request: ChatRequest, api_key: str | None) -> AsyncIterator[StreamEvent]:
        wants_tool = any("[tool]" in message.text for message in request.messages)
        if wants_tool:
            yield StreamToolCallDelta(index=0, id="call_mock_1", name="echo")
            yield StreamToolCallDelta(index=0, argument_delta=json.dumps({"text": "hi"}))
            yield StreamUsageEvent(usage=Usage(input_tokens=10, output_tokens=5))
            yield StreamEndEvent(stop_reason="toolUse", usage=Usage(input_tokens=10, output_tokens=5))
            return
        yield StreamTextDelta(delta="Hello")
        yield StreamTextDelta(delta=" from the mock provider!")
        yield StreamUsageEvent(usage=Usage(input_tokens=10, output_tokens=8))
        yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=8))


def client_for_spec(spec: Any, *, http_client: httpx.AsyncClient | None = None) -> WireClient:
    """Build the wire client for a ``ModelSpec`` via the provider registry."""
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(spec.provider)
    wire = definition.wire if definition else "openai-compat"
    if wire == "mock":
        return MockClient()
    if wire == "anthropic":
        base = spec.base_url or (definition.base_url if definition else None) or ANTHROPIC_API_URL
        return AnthropicClient(base_url=base, http_client=http_client)
    if wire == "google":
        base = spec.base_url or GOOGLE_API_URL
        return GoogleClient(base_url=base, http_client=http_client)
    base = spec.base_url or (definition.base_url if definition else None) or "http://localhost:11434/v1"
    extra_headers = None
    if spec.provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://local-operator.com",
            "X-Title": "Local Operator",
        }
    return OpenAICompatClient(base_url=base, http_client=http_client, extra_headers=extra_headers)

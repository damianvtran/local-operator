"""Native Bedrock Converse, without an OpenAI/Anthropic intermediate.

The same projection is used for signature-prefix proof, request admission and
wire bytes. In particular a restored signed turn is not authority to replay it
after earlier history changes. Claude stays on its existing InvokeModel path.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx

from local_operator.harness.types import (
    AttachmentContent,
    ChatRequest,
    DocumentContent,
    ImageContent,
    StreamEndEvent,
    StreamEvent,
    StreamReasoningDelta,
    StreamStartEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    Usage,
)
from local_operator.model import tariff
from local_operator.providers.bedrock import auth, connection, errors
from local_operator.providers.bedrock.client import _guarded_bytes, region_sentinel
from local_operator.providers.bedrock.models import (
    BedrockCapabilities,
    capabilities_for_spec,
)
from local_operator.providers.clients import (
    EMPTY_TOOL_RESULT_TEXT,
    _effective_max_tokens,
    _encode_json_body,
    _sampling_params,
    _StreamOutputEnvelope,
)
from local_operator.providers.context import bind_native_context
from local_operator.providers.documents import document_text_cap, fallback_document_text
from local_operator.providers.image_rejection import stream_with_rejected_image_retry
from local_operator.providers.replay import (
    credential_scope,
    native_payload,
    replay_items,
)

PROTOCOL = "bedrock-converse"
_DOCUMENT_FORMATS = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/csv": "csv",
    "text/html": "html",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}
_STOP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "toolUse",
    "guardrail_intervened": "refusal",
    "content_filtered": "refusal",
}


def _protocol_error(message: str):
    # ProviderError, rather than ValueError, participates in the normal failure
    # path; malformed wire output must never be treated as a successful reply.
    from local_operator.providers.failover import ProviderError

    return ProviderError(None, f"Bedrock Converse: {message}", kind="unknown", retryable=False)


def _prefix(system: list[dict[str, Any]], messages: list[dict[str, Any]]) -> str:
    """Hash actual provider-visible history BEFORE transport cache decoration.

    Cache points are excluded by construction: they affect billing, not content.
    Native blocks already selected for preceding turns are included verbatim,
    so invalidating one signature invalidates dependent later signatures too.
    """
    return hashlib.sha256(
        json.dumps(
            [system, messages], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _blob(data: str, limit: int, what: str) -> bytes:
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise _protocol_error(f"invalid base64 {what}") from exc
    if not raw or len(raw) > limit:
        raise _protocol_error(f"{what} exceeds Bedrock input size or is empty")
    return raw


def project_conversation(
    request: ChatRequest, caps: BedrockCapabilities, endpoint: str, scope: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any], str]:
    """Return system, messages, selected native per harness row, final prefix.

    An explicit unsupported modality fails rather than disappearing. Documents
    without positive native support use the existing retained-byte extraction
    policy; the original filesystem path is never consulted here.
    """
    system = [{"text": text} for text in request.system_blocks if text]
    rows: list[dict[str, Any]] = []
    selected: list[Any] = []
    image_count = document_count = 0
    pending_calls: set[str] = set()

    def content(blocks, role: str, tool: bool = False):
        nonlocal image_count, document_count
        out = []
        for block in blocks:
            if isinstance(block, TextContent):
                if block.text:
                    out.append({"text": block.text})
            elif isinstance(block, AttachmentContent):
                # Output-only artifact references are ledger/UI metadata, not
                # input modalities. Every wire omits them; matching that policy
                # keeps a generated artifact from breaking the next tool turn.
                continue
            elif isinstance(block, ImageContent):
                allowed = caps.tool_result_images if tool else caps.images
                if role != "user" or allowed is not True:
                    raise _protocol_error("image input is not supported for this model/placement")
                fmt = {
                    "image/png": "png",
                    "image/jpeg": "jpeg",
                    "image/gif": "gif",
                    "image/webp": "webp",
                }.get(block.mime_type)
                if not fmt:
                    raise _protocol_error("unsupported image MIME type")
                raw = _blob(block.data, 3_750_000, "image")
                # Reuse the existing images extra, not another decoder or a
                # dependency addition. A text-only install fails clearly rather
                # than sending unchecked dimensions or exposing ImportError.
                from io import BytesIO

                try:
                    from PIL import Image
                except ImportError as exc:
                    raise _protocol_error(
                        "image validation requires local-operator[images]"
                    ) from exc

                try:
                    with Image.open(BytesIO(raw)) as image:
                        if (image.format or "").lower() != fmt:
                            raise _protocol_error("image MIME type does not match its bytes")
                        if max(image.size) > 8000:
                            raise _protocol_error("image dimensions exceed 8000 pixels")
                except (OSError, ValueError) as exc:
                    raise _protocol_error("invalid image bytes") from exc
                image_count += 1
                if image_count > 20:
                    raise _protocol_error("more than 20 images in Converse request")
                out.append({"image": {"format": fmt, "source": {"bytes": block.data}}})
            elif isinstance(block, DocumentContent):
                fmt = _DOCUMENT_FORMATS.get(block.mime_type)
                formats = caps.tool_result_document_formats if tool else caps.document_formats
                if role == "user" and fmt in formats:
                    _blob(block.data, 4_500_000, "document")
                    document_count += 1
                    if document_count > 5:
                        raise _protocol_error("more than 5 documents in Converse request")
                    out.append(
                        {
                            "document": {
                                "format": fmt,
                                "name": f"Document {document_count}",
                                "source": {"bytes": block.data},
                            }
                        }
                    )
                else:
                    out.append(
                        {
                            "text": fallback_document_text(
                                block, document_text_cap(request.model.context_window)
                            )
                        }
                    )
            else:
                raise _protocol_error("unsupported Converse content modality")
        if any("document" in part for part in out) and not any("text" in part for part in out):
            out.insert(0, {"text": "Attached document."})
        return out

    for message in request.messages:
        native = None
        role = "assistant" if message.role == "assistant" else "user"
        if message.role == "assistant":
            pending_calls = {call.id for call in message.tool_calls}
            candidate = replay_items(message, request.model, endpoint, PROTOCOL, scope)
            payload = message.provider_payload or {}
            if candidate is not None and payload.get("converse_prefix") == _prefix(system, rows):
                native = candidate
            parts = copy.deepcopy(native) if native is not None else content(message.content, role)
            if native is None:
                for call in message.tool_calls:
                    if not isinstance(call.arguments, dict):
                        raise _protocol_error("historical tool arguments must be an object")
                    parts.append(
                        {
                            "toolUse": {
                                "toolUseId": call.id,
                                "name": call.name,
                                "input": call.arguments,
                            }
                        }
                    )
        elif message.role == "tool":
            if not message.tool_call_id:
                raise _protocol_error("tool result has no tool call ID")
            # Validate content before pairings so an unsupported mandatory
            # modality is never masked as an empty-result recovery.
            parts = content(message.content, role, tool=True) or [{"text": EMPTY_TOOL_RESULT_TEXT}]
            if message.tool_call_id not in pending_calls:
                raise _protocol_error(
                    "tool result is not paired with the preceding assistant calls"
                )
            pending_calls.remove(message.tool_call_id)
            if message.is_error and not caps.tool_result_status:
                parts.insert(0, {"text": "[tool error]"})
            result = {"toolUseId": message.tool_call_id, "content": parts}
            if caps.tool_result_status:
                result["status"] = "error" if message.is_error else "success"
            parts = [{"toolResult": result}]
        else:
            pending_calls.clear()
            parts = content(message.content, role)
        selected.append(native)
        if not parts:
            continue
        if rows and rows[-1]["role"] == role:
            rows[-1]["content"].extend(parts)
        else:
            rows.append({"role": role, "content": parts})
    return system, rows, selected, _prefix(system, rows)


def build_converse_body(
    request: ChatRequest, caps: BedrockCapabilities, endpoint: str, scope: str | None
) -> dict[str, Any]:
    request.validate_required_images()
    system, rows, _, _ = project_conversation(request, caps, endpoint, scope)
    body: dict[str, Any] = {
        "messages": rows,
        "inferenceConfig": {"maxTokens": _effective_max_tokens(request)},
    }
    if system:
        body["system"] = system
    inference = body["inferenceConfig"]
    inference.update(_sampling_params(request, top_p_key="topP"))
    if request.stop_sequences:
        inference["stopSequences"] = request.stop_sequences
    history_tools = any(
        "toolUse" in part or "toolResult" in part for row in rows for part in row["content"]
    )
    if request.tool_choice == "none" and history_tools:
        raise _protocol_error("tool_choice=none with historical tools is unsupported by Converse")
    if request.tools and request.tool_choice != "none":
        if caps.tools is False:
            raise _protocol_error("this model does not support client tools")
        tools = [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": t.parameters},
                }
            }
            for t in request.tools
        ]
        tool_config: dict[str, Any] = {"tools": tools}
        body["toolConfig"] = tool_config
        if request.tool_choice == "required":
            if caps.required_tools is not True:
                raise _protocol_error("required tool choice is not known supported by this model")
            tool_config["toolChoice"] = {"any": {}}
    elif history_tools:
        raise _protocol_error("historical tools require tool definitions")
    effort = request.model.reasoning_effort
    if caps.reasoning_config == "nova2" and effort is not None:
        if effort not in {"none", "off", "low", "medium", "high"}:
            raise _protocol_error("unsupported Nova reasoning effort")
        config = (
            {"type": "disabled"}
            if effort in {"none", "off"}
            else {"type": "enabled", "maxReasoningEffort": effort}
        )
        body["additionalModelRequestFields"] = {"reasoningConfig": config}
        if effort == "high":
            inference.pop("temperature", None)
            inference.pop("topP", None)
    # Cache only stable content, within the documented model maximum. Unknown
    # cache capabilities emit no points. Byte/4 is the existing local estimate;
    # short checkpoints are allowed to miss, never represented as guaranteed hits.
    if request.prompt_cache_key and caps.cache_max_points:
        from local_operator.compaction.tokens import count_text_tokens

        # Points cache the cumulative prefix (tools THEN system THEN messages),
        # not merely their containing block. Use the existing token ruler and
        # a provider-measured upper bound when present; unsafe prefixes skip.
        plain_system = copy.deepcopy(system)
        tool_prefix = body.get("toolConfig", {}).get("tools", [])

        def fits(prefix_rows):
            estimate = count_text_tokens(
                json.dumps([tool_prefix, plain_system, prefix_rows], ensure_ascii=False)
            )
            if request.context_tokens_hint_measured and request.context_tokens_hint is not None:
                estimate = max(estimate, request.context_tokens_hint)
            return caps.cache_max_tokens is None or estimate <= caps.cache_max_tokens

        count = 0
        if system and "system" in caps.cache_locations and fits([]):
            # 5m-only models reject the TTL field itself; omit it, do not send 5m.
            system.append({"cachePoint": {"type": "default"}})
            count += 1
        frozen = min(request.frozen_prefix_messages or 0, len(request.messages))
        if frozen and "messages" in caps.cache_locations and count < caps.cache_max_points:
            _, prefix_rows, _, _ = project_conversation(
                request.model_copy(update={"messages": request.messages[:frozen]}),
                caps,
                endpoint,
                scope,
            )
            if prefix_rows and fits(prefix_rows):
                # Harness rows coalesce on the wire. Put the point at the exact
                # prefix-content boundary, not after later same-role content.
                row = len(prefix_rows) - 1
                rows[row]["content"].insert(
                    len(prefix_rows[-1]["content"]), {"cachePoint": {"type": "default"}}
                )
    return body


def usage_from_converse(raw: Any) -> Usage:
    """Converse inputTokens excludes BOTH cache buckets (AWS caching guide).

    TTL details subdivide write tokens, not new tokens. Missing and invalid
    counts remain incomplete rather than fabricated provider-reported zero.
    """
    raw = raw if isinstance(raw, dict) else {}

    def valid(key):
        value = raw.get(key)
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def count(key):
        return raw[key] if valid(key) else 0

    written = count("cacheWriteInputTokens")
    splits = {"5m": 0, "1h": 0}
    details = raw.get("cacheDetails")
    for item in details if isinstance(details, list) else []:
        if isinstance(item, dict) and item.get("ttl") in splits:
            value = item.get("inputTokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                splits[item["ttl"]] += value
    if sum(splits.values()) > written:
        splits = {"5m": 0, "1h": 0}
    usage = Usage(
        input_tokens=count("inputTokens"),
        output_tokens=count("outputTokens"),
        cache_read_tokens=count("cacheReadInputTokens"),
        cache_write_tokens=written,
        cache_write_5m_tokens=splits["5m"],
        cache_write_1h_tokens=splits["1h"],
        context_tokens=(
            (count("inputTokens") + count("cacheReadInputTokens") + written)
            if valid("inputTokens")
            else None
        ),
        at_ms=tariff.now_ms(),
    )
    usage._cache_read_reported = valid("cacheReadInputTokens")
    usage._token_counts_reported = valid("inputTokens") and valid("outputTokens")
    return usage


async def iter_converse_events(response, *, model: str, region: str):
    from botocore.eventstream import EventStreamBuffer, ParserError

    buffer = EventStreamBuffer()

    def decoded():
        try:
            yield from buffer
        except ParserError as exc:
            raise _protocol_error("invalid event-stream framing or checksum") from exc

    async for chunk in _guarded_bytes(response):
        buffer.add_data(chunk)
        for message in decoded():
            headers = message.headers
            try:
                data = json.loads(message.payload)
            except (ValueError, TypeError) as exc:
                raise _protocol_error("malformed event JSON") from exc
            if not isinstance(data, dict):
                raise _protocol_error("event payload must be an object")
            name = (
                headers.get(":event-type")
                or headers.get(":exception-type")
                or headers.get(":error-code")
            )
            if headers.get(":message-type") in {"exception", "error"} or str(name).endswith(
                "Exception"
            ):
                raise errors.provider_error_for(
                    str(name), str(data.get("message", "")), None, model=model, region=region
                )
            if headers.get(":message-type") == "event":
                yield str(name), data
    # EventStreamBuffer intentionally retains partial frames between feeds. At
    # EOF, any remainder is truncated data even if messageStop was seen earlier.
    if buffer._data:
        raise _protocol_error("truncated event-stream frame")


class _Accumulator:
    """One block finalizer for streaming and Converse JSON output."""

    def __init__(self, max_tokens: int):
        self.blocks: dict[int, dict[str, Any]] = {}
        self.calls: dict[int, int] = {}
        self.raw_args: dict[int, str] = {}
        self.text: list[str] = []
        self.closed: set[int] = set()
        self.envelope = _StreamOutputEnvelope(max_tokens)
        self.started = False
        self.stopped = False
        self.reason = ""
        self.usage = usage_from_converse(None)
        self.replayable = True

    def _index(self, data):
        index = data.get("contentBlockIndex")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in self.closed
        ):
            raise _protocol_error("invalid or closed content block index")
        return index

    def event(self, name: str, data: dict[str, Any]) -> list[StreamEvent]:
        out = []
        if name == "messageStart":
            if self.started or self.stopped or data.get("role") != "assistant":
                raise _protocol_error("invalid messageStart")
            self.started = True
            return [StreamStartEvent()]
        if not self.started:
            raise _protocol_error("event before messageStart")
        if name == "metadata":
            if not self.stopped:
                raise _protocol_error("metadata before messageStop")
            self.usage = usage_from_converse(data.get("usage"))
            return []
        if self.stopped:
            raise _protocol_error("content after messageStop")
        if name == "messageStop":
            if set(self.blocks) != self.closed or not isinstance(data.get("stopReason"), str):
                raise _protocol_error("unfinished block or missing stop reason")
            self.stopped = True
            self.reason = data["stopReason"]
            return []
        if name not in {"contentBlockStart", "contentBlockDelta", "contentBlockStop"}:
            raise _protocol_error(f"unknown event {name}")
        index = self._index(data)
        if name == "contentBlockStart":
            if index in self.blocks:
                raise _protocol_error("duplicate contentBlockStart")
            start = data.get("start")
            if not isinstance(start, dict) or set(start) != {"toolUse"}:
                raise _protocol_error("unsupported content block start")
            tool = start["toolUse"]
            if (
                not isinstance(tool, dict)
                or not isinstance(tool.get("toolUseId"), str)
                or not isinstance(tool.get("name"), str)
            ):
                raise _protocol_error("malformed toolUse start")
            self.envelope.charge(tool["name"] + tool["toolUseId"])
            self.blocks[index] = {"toolUse": dict(tool)}
            self.calls[index] = len(self.calls)
            self.raw_args[index] = ""
            return [
                StreamToolCallDelta(
                    index=self.calls[index], id=tool["toolUseId"], name=tool["name"]
                )
            ]
        if name == "contentBlockStop":
            if index not in self.blocks:
                raise _protocol_error("stop for missing block")
            if index in self.raw_args:
                try:
                    value = json.loads(self.raw_args[index])
                    if not isinstance(value, dict):
                        raise ValueError("not object")
                    self.blocks[index]["toolUse"]["input"] = value
                except ValueError:
                    # Preserve streamed raw arguments for harness repair, but
                    # NEVER replay only the valid part of an assistant turn.
                    self.replayable = False
            self.closed.add(index)
            return []
        delta = data.get("delta")
        if not isinstance(delta, dict) or len(delta) != 1:
            raise _protocol_error("malformed content delta")
        if "text" in delta:
            text = delta["text"]
            if not isinstance(text, str):
                raise _protocol_error("non-text text delta")
            self.envelope.charge(text)
            block = self.blocks.setdefault(index, {"text": ""})
            if "text" not in block:
                raise _protocol_error("content block changed type")
            block["text"] += text
            self.text.append(text)
            out.append(StreamTextDelta(delta=text))
        elif "toolUse" in delta:
            tool = delta["toolUse"]
            if (
                index not in self.calls
                or not isinstance(tool, dict)
                or not isinstance(tool.get("input"), str)
            ):
                raise _protocol_error("tool delta without valid tool start")
            self.envelope.charge(tool["input"])
            self.raw_args[index] += tool["input"]
            out.append(StreamToolCallDelta(index=self.calls[index], argument_delta=tool["input"]))
        elif "reasoningContent" in delta:
            value = delta["reasoningContent"]
            if not isinstance(value, dict) or len(value) != 1:
                raise _protocol_error("malformed reasoning delta")
            key, piece = next(iter(value.items()))
            if key not in {"text", "signature", "redactedContent"} or not isinstance(piece, str):
                raise _protocol_error("unsupported reasoning delta")
            self.envelope.charge(piece)
            block = self.blocks.setdefault(index, {"reasoningContent": {}})
            if "reasoningContent" not in block:
                raise _protocol_error("reasoning block changed type")
            reasoning = block["reasoningContent"]
            if key == "redactedContent":
                if "reasoningText" in reasoning:
                    raise _protocol_error("mixed reasoning union")
                try:
                    old = base64.b64decode(reasoning.get(key, ""), validate=True)
                    new = base64.b64decode(piece, validate=True)
                except ValueError as exc:
                    raise _protocol_error("invalid redacted reasoning") from exc
                reasoning[key] = base64.b64encode(old + new).decode()
            else:
                if "redactedContent" in reasoning:
                    raise _protocol_error("mixed reasoning union")
                target = reasoning.setdefault("reasoningText", {})
                target[key] = target.get(key, "") + piece
                if key == "text":
                    out.append(StreamReasoningDelta(delta=piece))
        else:
            raise _protocol_error("unsupported output modality")
        return out

    def json_events(self, response: dict[str, Any]):
        try:
            message = response["output"]["message"]
            blocks = message["content"]
            if message["role"] != "assistant" or not isinstance(blocks, list):
                raise ValueError("invalid message")
        except (KeyError, TypeError, ValueError) as exc:
            raise _protocol_error("malformed Converse response") from exc
        yield from self.event("messageStart", {"role": "assistant"})
        for index, block in enumerate(blocks):
            if not isinstance(block, dict) or len(block) != 1:
                raise _protocol_error("malformed response content block")
            if "toolUse" in block:
                tool = block["toolUse"]
                if not isinstance(tool, dict) or not isinstance(tool.get("input"), dict):
                    raise _protocol_error("malformed response toolUse")
                yield from self.event(
                    "contentBlockStart",
                    {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {k: tool.get(k) for k in ("toolUseId", "name")}},
                    },
                )
                yield from self.event(
                    "contentBlockDelta",
                    {
                        "contentBlockIndex": index,
                        "delta": {"toolUse": {"input": json.dumps(tool["input"])}},
                    },
                )
            elif "reasoningContent" in block:
                reasoning = block["reasoningContent"]
                if not isinstance(reasoning, dict) or len(reasoning) != 1:
                    raise _protocol_error("malformed response reasoning")
                if "reasoningText" in reasoning:
                    values = reasoning["reasoningText"]
                    if not isinstance(values, dict):
                        raise _protocol_error("malformed reasoningText response")
                else:
                    values = reasoning
                for key, value in values.items():
                    yield from self.event(
                        "contentBlockDelta",
                        {"contentBlockIndex": index, "delta": {"reasoningContent": {key: value}}},
                    )
            else:
                yield from self.event(
                    "contentBlockDelta", {"contentBlockIndex": index, "delta": block}
                )
            yield from self.event("contentBlockStop", {"contentBlockIndex": index})
        yield from self.event("messageStop", {"stopReason": response.get("stopReason")})
        self.event("metadata", {"usage": response.get("usage")})

    def end(self, request, endpoint, scope, prefix):
        if not self.started or not self.stopped:
            raise _protocol_error("EOF before terminal messageStop")
        items = [self.blocks[i] for i in sorted(self.blocks)]
        text = "".join(self.text)
        # Fingerprint what the harness observed, not block-index ordering: tool
        # starts may interleave, and their stable harness index is start order.
        calls = [
            {
                "id": self.blocks[i]["toolUse"]["toolUseId"],
                "name": self.blocks[i]["toolUse"]["name"],
                "args": self.blocks[i]["toolUse"].get("input", {}),
            }
            for i in sorted(self.calls, key=self.calls.__getitem__)
        ]
        payload = {"converse_stop_reason": self.reason}
        if self.replayable:
            payload.update(
                native_payload(request.model, endpoint, PROTOCOL, items, text, calls, scope)
            )
            payload["converse_prefix"] = prefix
        return StreamEndEvent(
            stop_reason=_STOP.get(self.reason, "error"), usage=self.usage, provider_payload=payload
        )


class BedrockConverseClient:
    """Async native HTTP transport; an injected pool remains caller-owned."""

    def __init__(
        self,
        *,
        http_client=None,
        region=None,
        timeout=600.0,
        credential_timeout=auth.CREDENTIAL_TIMEOUT_S,
    ):
        self._http = http_client
        self._region = region
        self._timeout = timeout
        self._credential_timeout = credential_timeout

    async def stream(self, request, api_key, oauth_access=None) -> AsyncIterator[StreamEvent]:
        # Reuse the harness's one pre-output image refusal recovery, including
        # mandatory-image and rejected-image ledger semantics. The sent body is
        # this native projection, never an Anthropic approximation of its paths.
        async def attempt(candidate, sent):
            async for event in self._stream_once(candidate, api_key, oauth_access, sent):
                yield event

        async for event in stream_with_rejected_image_retry(request, attempt):
            yield event

    async def _stream_once(
        self, request, api_key, oauth_access, sent
    ) -> AsyncIterator[StreamEvent]:
        if not api_key:
            raise errors.no_credentials_error()
        try:
            resolved = await auth.resolve(api_key, timeout=self._credential_timeout)
        except auth.BedrockCredentialError as exc:
            raise exc.provider_error from exc
        region = self._region or resolved.region
        base = connection.runtime_base(region)
        endpoint = region_sentinel(region) + "|" + base
        scope = credential_scope(api_key, oauth_access)
        caps = capabilities_for_spec(request.model)
        _, _, selected, prefix = project_conversation(request, caps, endpoint, scope)
        request = bind_native_context(
            request, endpoint, PROTOCOL, scope, 1.0, selected_native=selected
        )
        body = build_converse_body(request, caps, endpoint, scope)
        sent["body"] = body
        operation = (
            "converse"
            if caps.streaming is False or ("toolConfig" in body and caps.streaming_tools is False)
            else "converse-stream"
        )
        encoded = _encode_json_body(body)
        client = self._http or httpx.AsyncClient(timeout=self._timeout)
        try:
            # At most one capability-only pre-accept retry. Never retry auth,
            # arbitrary validation, partial generation or malformed streams.
            for attempt in range(2):
                url = f"{base}/model/{quote(request.model.model_id, safe='')}/{operation}"
                headers = auth.sign_headers(
                    resolved,
                    "POST",
                    url,
                    encoded,
                    {
                        "Content-Type": "application/json",
                        "Accept": (
                            "application/json"
                            if operation == "converse"
                            else "application/vnd.amazon.eventstream"
                        ),
                    },
                )
                async with client.stream(
                    "POST", url, content=encoded, headers=headers, timeout=self._timeout
                ) as response:
                    if response.is_error:
                        await response.aread()
                        try:
                            value = response.json()
                        except ValueError:
                            value = {}
                        message = str(value.get("message", "")) if isinstance(value, dict) else ""
                        kind = response.headers.get("x-amzn-errortype", "").split(":")[0]
                        unsupported = "stream" in message.lower() and any(
                            s in message.lower()
                            for s in ("not support", "unsupported", "doesn't support")
                        )
                        if (
                            attempt == 0
                            and operation == "converse-stream"
                            and kind == "ValidationException"
                            and unsupported
                        ):
                            operation = "converse"
                            continue
                        errors.raise_for_status(
                            response, model=request.model.model_id, region=region
                        )
                    accumulator = _Accumulator(body["inferenceConfig"]["maxTokens"])
                    if operation == "converse":
                        await response.aread()
                        try:
                            value = response.json()
                        except ValueError as exc:
                            raise _protocol_error("malformed JSON response") from exc
                        for event in accumulator.json_events(value):
                            yield event
                    else:
                        async for name, data in iter_converse_events(
                            response, model=request.model.model_id, region=region
                        ):
                            for event in accumulator.event(name, data):
                                yield event
                    end = accumulator.end(request, endpoint, scope, prefix)
                    yield StreamUsageEvent(usage=accumulator.usage)
                    yield end
                    return
        finally:
            if self._http is None:
                await client.aclose()

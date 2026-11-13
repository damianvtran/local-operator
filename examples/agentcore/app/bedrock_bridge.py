"""A loopback OpenAI-compatible endpoint that forwards to Amazon Bedrock with SigV4.

Why this exists
---------------
Local Operator has no native Bedrock provider. It does have a generic
``openai-compatible`` hosting that talks the OpenAI Chat Completions wire to any
base URL. Bedrock's own OpenAI-compatible surfaces (``bedrock-runtime
/openai/v1`` and ``bedrock-mantle``) accept SigV4 but, when this sample was
written, served only open-weight models: a Claude inference profile answers
``404 model_not_found`` there. So this module translates OpenAI Chat
Completions (streaming and tool calls) to Bedrock **ConverseStream** and back.

Constraints that shaped it
--------------------------
* **No API keys.** Every Bedrock call is signed by botocore's default
  credential chain, which inside AgentCore Runtime resolves to the runtime's
  execution role. The local token lop sends (if any) is ignored.
* **Loopback only.** The server binds ``127.0.0.1``; it is a process-internal
  adapter, not a gateway. Anything that can reach it can already reach the
  execution role's credentials, so it adds no new authority.
* **Small and auditable.** Standard library HTTP server plus boto3 (which the
  AgentCore SDK already depends on). No LiteLLM: the translation lop needs is a
  few hundred lines and every line of it is visible here.
* **Model-agnostic.** Converse works for every Bedrock chat model that supports
  tool use, so the bridge never assumes Claude-specific request shapes.

What is deliberately NOT translated: images in tool results, reasoning/thinking
blocks, prompt-cache points and ``n > 1``. Unsupported inputs degrade to text or
are dropped, never fabricated.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger("lop_agentcore.bridge")

# Converse stop reasons mapped onto the OpenAI ``finish_reason`` vocabulary.
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}

# Bedrock error codes mapped onto HTTP statuses lop's OpenAI client classifies
# (auth vs. rate limit vs. bad request), so a denied model surfaces as a clear
# permission error instead of a generic 500.
_ERROR_STATUS = {
    "AccessDeniedException": 403,
    "UnrecognizedClientException": 401,
    "ExpiredTokenException": 401,
    "ResourceNotFoundException": 404,
    "ValidationException": 400,
    "ThrottlingException": 429,
    "ServiceQuotaExceededException": 429,
    "ModelNotReadyException": 503,
    "ServiceUnavailableException": 503,
    "ModelTimeoutException": 504,
}


class BridgeError(Exception):
    """A request the bridge cannot translate; reported to the caller as HTTP 400."""


# ---------------------------------------------------------------------------
# OpenAI request -> Converse request
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    """Flatten OpenAI message content (string or part list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return str(content)


def _image_block(url: str) -> dict[str, Any] | None:
    """A Converse image block from a ``data:`` URI; remote URLs are not fetched.

    Fetching arbitrary URLs from inside the bridge would turn it into an SSRF
    primitive, and lop inlines images as data URIs anyway.
    """
    if not url.startswith("data:image/") or ";base64," not in url:
        return None
    header, data = url.split(",", 1)
    fmt = header[len("data:image/") :].split(";", 1)[0].lower()
    fmt = "jpeg" if fmt == "jpg" else fmt
    if fmt not in ("png", "jpeg", "gif", "webp"):
        return None
    return {"image": {"format": fmt, "source": {"bytes": base64.b64decode(data)}}}


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind in ("text", "input_text") and part.get("text"):
                blocks.append({"text": part["text"]})
            elif kind == "image_url":
                image = part.get("image_url") or {}
                url = image.get("url", "") if isinstance(image, dict) else str(image)
                block = _image_block(url)
                if block is not None:
                    blocks.append(block)
        return blocks
    text = _text_of(content)
    return [{"text": text}] if text else []


def _tool_input(arguments: Any) -> Any:
    """Converse wants the tool input as a JSON object, OpenAI carries a string."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        # A model that emitted malformed JSON last turn must not wedge the
        # conversation: keep the raw text so the next turn can see it.
        return {"_raw_arguments": str(arguments)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _append(messages: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    """Converse requires strictly alternating roles: merge consecutive same-role turns."""
    if not blocks:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def _has_tool_blocks(messages: list[dict[str, Any]]) -> bool:
    return any(
        "toolUse" in block or "toolResult" in block
        for message in messages
        for block in message.get("content", [])
    )


def to_converse(request: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions body into ``converse_stream`` kwargs."""
    system: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for message in request.get("messages") or []:
        role = message.get("role")
        if role in ("system", "developer"):
            text = _text_of(message.get("content"))
            if text:
                system.append({"text": text})
        elif role == "user":
            _append(messages, "user", _user_blocks(message.get("content")))
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _text_of(message.get("content"))
            if text.strip():
                blocks.append({"text": text})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "name": function.get("name", ""),
                            "input": _tool_input(function.get("arguments")),
                        }
                    }
                )
            _append(messages, "assistant", blocks)
        elif role == "tool":
            # Converse carries tool results inside a USER turn; empty results
            # are rejected, so an empty string becomes an explicit marker.
            text = _text_of(message.get("content")) or "(no output)"
            _append(
                messages,
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.get("tool_call_id", ""),
                            "content": [{"text": text}],
                        }
                    }
                ],
            )
        else:
            raise BridgeError(f"unsupported message role: {role!r}")
    if not messages:
        raise BridgeError("request has no user/assistant messages")

    kwargs: dict[str, Any] = {"modelId": model_id, "messages": messages}
    if system:
        kwargs["system"] = system

    inference: dict[str, Any] = {}
    max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
    if max_tokens:
        inference["maxTokens"] = int(max_tokens)
    if request.get("temperature") is not None:
        inference["temperature"] = float(request["temperature"])
    if request.get("top_p") is not None:
        inference["topP"] = float(request["top_p"])
    stop = request.get("stop")
    if stop:
        inference["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    if inference:
        kwargs["inferenceConfig"] = inference

    tools = [
        {
            "toolSpec": {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description") or tool["function"]["name"],
                "inputSchema": {"json": tool["function"].get("parameters") or {"type": "object"}},
            }
        }
        for tool in request.get("tools") or []
        if tool.get("type") == "function" and tool.get("function", {}).get("name")
    ]
    choice = request.get("tool_choice")
    if tools and choice == "none" and not _has_tool_blocks(kwargs["messages"]):
        # Converse has no "none" tool choice; the only way to forbid tool calls
        # is to send no toolConfig. That is allowed only while the history has
        # no toolUse/toolResult blocks (Converse rejects those without a
        # toolConfig), so a "none" after a tool round degrades to auto below.
        tools = []
    if tools:
        tool_config: dict[str, Any] = {"tools": tools}
        if choice == "required":
            tool_config["toolChoice"] = {"any": {}}
        elif isinstance(choice, dict) and choice.get("type") == "function":
            tool_config["toolChoice"] = {"tool": {"name": choice["function"]["name"]}}
        elif choice in (None, "auto"):
            tool_config["toolChoice"] = {"auto": {}}
        # "none" reaching here: history references tools, so the specs must
        # stay and no choice is set (Converse then behaves as auto).
        kwargs["toolConfig"] = tool_config
    return kwargs


# ---------------------------------------------------------------------------
# Converse stream -> OpenAI chunks
# ---------------------------------------------------------------------------


def _chunk(
    completion_id: str, model: str, created: int, delta: dict[str, Any], finish: str | None
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _usage(metadata: dict[str, Any]) -> dict[str, int]:
    usage = metadata.get("usage") or {}
    prompt = int(usage.get("inputTokens", 0))
    completion = int(usage.get("outputTokens", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("totalTokens", prompt + completion)),
    }


def stream_chunks(events: Iterable[dict[str, Any]], model: str) -> Iterator[dict[str, Any]]:
    """Yield OpenAI ``chat.completion.chunk`` dicts for a ConverseStream event stream.

    Tool calls are numbered by their order among tool blocks (OpenAI's
    ``index``), not by Converse's ``contentBlockIndex``, which also counts text
    blocks. Usage arrives in Converse's trailing ``metadata`` event, after
    ``messageStop``, so the finish chunk is held until then and usage is
    emitted as OpenAI's final ``choices: []`` chunk.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    tool_index: dict[int, int] = {}
    finish: str | None = None
    yield _chunk(completion_id, model, created, {"role": "assistant", "content": ""}, None)
    for event in events:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]
            tool = (start.get("start") or {}).get("toolUse")
            if tool:
                index = tool_index.setdefault(start["contentBlockIndex"], len(tool_index))
                delta = {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool["toolUseId"],
                            "type": "function",
                            "function": {"name": tool["name"], "arguments": ""},
                        }
                    ]
                }
                yield _chunk(completion_id, model, created, delta, None)
        elif "contentBlockDelta" in event:
            block = event["contentBlockDelta"]
            delta = block.get("delta") or {}
            if "text" in delta and delta["text"]:
                yield _chunk(completion_id, model, created, {"content": delta["text"]}, None)
            elif "toolUse" in delta:
                index = tool_index.get(block["contentBlockIndex"])
                if index is not None:
                    tool_delta = {
                        "tool_calls": [
                            {"index": index, "function": {"arguments": delta["toolUse"]["input"]}}
                        ]
                    }
                    yield _chunk(completion_id, model, created, tool_delta, None)
        elif "messageStop" in event:
            finish = _FINISH_REASONS.get(event["messageStop"].get("stopReason", ""), "stop")
        elif "metadata" in event:
            yield _chunk(completion_id, model, created, {}, finish or "stop")
            finish = None
            usage_chunk = _chunk(completion_id, model, created, {}, None)
            usage_chunk["choices"] = []
            usage_chunk["usage"] = _usage(event["metadata"])
            yield usage_chunk
        else:
            # Converse delivers modelled errors as stream events after HTTP 200.
            for key in event:
                if key.endswith("Exception"):
                    raise StreamError(key, (event[key] or {}).get("message", key))
    if finish is not None:  # stream ended without a metadata event
        yield _chunk(completion_id, model, created, {}, finish)


class StreamError(Exception):
    """A Converse error delivered mid-stream (after headers were sent)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def collect_completion(chunks: Iterable[dict[str, Any]], model: str) -> dict[str, Any]:
    """Fold streamed chunks into one non-streaming ``chat.completion`` body."""
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish = "stop"
    usage: dict[str, int] | None = None
    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            text.append(delta.get("content") or "")
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    call["index"],
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    body: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }
    if usage:
        body["usage"] = usage
    return body


def error_status(exc: Exception) -> tuple[int, str, str]:
    """(HTTP status, error code, message) for a botocore or stream error."""
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error") or {}
        code = error.get("Code") or code
        message = error.get("Message") or message
    code = code or type(exc).__name__
    return _ERROR_STATUS.get(code, 502), code, message or str(exc)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


ConverseStream = Callable[..., dict[str, Any]]


def make_handler(model_id: str, converse_stream: ConverseStream) -> type[BaseHTTPRequestHandler]:
    """Build the request handler bound to one model and one ConverseStream callable.

    ``converse_stream`` is injected (normally ``bedrock_runtime.converse_stream``)
    so the tests drive the real HTTP path with a fake Bedrock.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            log.debug("bridge: " + format, *args)

        def _json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": code, "code": code}})

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": model_id, "object": "model", "owned_by": "bedrock"}],
                    },
                )
            else:
                self._error(404, "not_found", self.path)

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._error(404, "not_found", self.path)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
                # The served model is fixed by the deployment (and the IAM
                # policy); a different ``model`` in the body is honoured so a
                # session can opt into another granted model.
                model = request.get("model") or model_id
                kwargs = to_converse(request, model)
            except (BridgeError, ValueError, KeyError, TypeError) as exc:
                self._error(400, "invalid_request_error", str(exc))
                return
            started = time.monotonic()
            try:
                response = converse_stream(**kwargs)
            except Exception as exc:  # botocore ClientError and friends
                status, code, message = error_status(exc)
                log.warning("bedrock converse_stream failed model=%s code=%s", model, code)
                self._error(status, code, message)
                return
            chunks = stream_chunks(response["stream"], model)
            if not request.get("stream"):
                try:
                    self._json(200, collect_completion(chunks, model))
                # StreamError, or botocore's EventStreamError raised by the
                # stream iterator itself for a modelled mid-stream failure.
                except Exception as exc:
                    status, code, message = error_status(exc)
                    self._error(status, code, message)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            usage: dict[str, int] = {}
            try:
                for chunk in chunks:
                    usage = chunk.get("usage") or usage
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
            except Exception as exc:  # StreamError or botocore EventStreamError
                # Headers are gone; OpenAI clients read an ``error`` frame.
                _, code, message = error_status(exc)
                frame = {"error": {"message": message, "type": code, "code": code}}
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            # One INFO line per model call: the CloudWatch evidence that
            # inference went through boto3 (SigV4, role credentials).
            log.info(
                "bedrock converse_stream model=%s auth=sigv4 in=%s out=%s %.1fs",
                model,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                time.monotonic() - started,
            )

    return Handler


def start_bridge(
    model_id: str,
    *,
    port: int = 0,
    region: str | None = None,
    converse_stream: ConverseStream | None = None,
) -> ThreadingHTTPServer:
    """Start the bridge on ``127.0.0.1:port`` in a daemon thread and return the server.

    ``port=0`` picks a free port (read it back from ``server.server_address``).
    """
    if converse_stream is None:
        import boto3
        from botocore.config import Config

        session = boto3.Session()
        credentials = session.get_credentials()
        log.info(
            "bridge credentials method=%s (SigV4; no API key)",
            getattr(credentials, "method", "unknown") if credentials else "none",
        )
        client = session.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_REGION"),
            # Long agent turns stream for minutes; the default 60 s read
            # timeout would cut a slow model mid-answer.
            config=Config(read_timeout=600, retries={"max_attempts": 4, "mode": "adaptive"}),
        )
        converse_stream = client.converse_stream
    assert converse_stream is not None
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(model_id, converse_stream))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="bedrock-bridge", daemon=True).start()
    return server

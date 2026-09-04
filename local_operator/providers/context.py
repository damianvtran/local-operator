"""Reconcile provider counts with the exact conversation they measured.

A previous input-token count excludes the answer and every subsequent tool
result. It is useful only with its counted boundary. Keep that boundary in the
conversation owner, and estimate just appended material. Edited old messages,
compaction, tools, system instructions and model changes invalidate calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from local_operator.compaction.tokens import estimate_messages_tokens
from local_operator.harness.types import ChatRequest, ImageContent, Message, Usage
from local_operator.providers.replay import replay_items


def model_key(request: ChatRequest) -> str:
    return f"{request.model.provider}/{request.model.model_id}"


def _native_key(value: Any) -> Any:
    # Reuse immutable strings (and their cached hash), rather than serializing
    # encrypted continuations again or retaining another copy of their bytes.
    if isinstance(value, dict):
        return tuple((key, _native_key(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_native_key(item) for item in value)
    return hash(value) if isinstance(value, str) else value


def _message_key(message: Message) -> tuple[Any, ...]:
    # Strings cache their Python hash, so revisiting long immutable history
    # does not encode it again. Include content rather than only stable ids:
    # pruning and restored/editable transcripts can retain an id after an edit.
    return (
        message.id,
        message.role,
        tuple(
            (
                (block.type, hash(block.data), block.mime_type)
                if isinstance(block, ImageContent)
                else (block.type, hash(block.text))
            )
            for block in message.content
        ),
        tuple(
            (call.id, call.name, json.dumps(call.arguments, sort_keys=True))
            for call in message.tool_calls
        ),
        message.tool_call_id,
        message.tool_name,
        _native_key((message.provider_payload or {}).get("native_replay")),
        message.usage.reasoning_tokens if message.usage else 0,
    )


@dataclass(frozen=True)
class RequestMeasure:
    model: str
    prefix: tuple[Any, ...]
    messages: tuple[tuple[Any, ...], ...]
    tokens: int
    native_tokens: int = 0


def measure_request(request: ChatRequest) -> RequestMeasure:
    tools = tuple(
        (tool.name, tool.description, json.dumps(tool.parameters, sort_keys=True))
        for tool in request.tools
    )
    prefix_chars = sum(map(len, request.system_blocks)) + sum(
        len(name) + len(description) + len(schema) for name, description, schema in tools
    )
    return RequestMeasure(
        model_key(request),
        (request.model.base_url, tuple(request.system_blocks), tools),
        tuple(_message_key(message) for message in request.messages),
        estimate_messages_tokens(request.messages) + prefix_chars // 4,
    )


class ContextTokenTracker:
    """One bounded baseline per conversation, with no transcript serialization."""

    def __init__(self) -> None:
        self.baseline: RequestMeasure | None = None
        self.provider_tokens = 0

    def estimate(self, measured: RequestMeasure, slope: float) -> int | None:
        reconciled = self.reconcile(measured, slope)
        return reconciled[0] if reconciled is not None else None

    def reconcile(self, measured: RequestMeasure, slope: float) -> tuple[int, int] | None:
        """Return scaled admission sizing and unscaled refusal evidence.

        The output reservation may conservatively inflate new local tokens;
        a refusal must not turn that optional safety margin into measured
        usage. Both numbers retain the same authoritatively counted prefix.
        """
        previous = self.baseline
        if (
            previous is None
            or previous.model != measured.model
            or previous.prefix != measured.prefix
            or measured.messages[: len(previous.messages)] != previous.messages
        ):
            return None
        native_delta = max(0, measured.native_tokens - previous.native_tokens)
        delta = max(0, measured.tokens - previous.tokens - native_delta)
        # Provider-reported reasoning is already on the provider's ruler;
        # only the locally tokenized visible suffix needs the family margin.
        return (
            self.provider_tokens + native_delta + int(delta * slope),
            self.provider_tokens + native_delta + delta,
        )

    def record(self, measured: RequestMeasure, usage: Usage) -> None:
        # A fallback may have a different tokenizer and a transformed prefix.
        # Its usage cannot calibrate the request originally given to the router.
        if usage.provider and usage.model_id:
            if f"{usage.provider}/{usage.model_id}" != measured.model:
                self.baseline = None
                return
        if usage.context_tokens:
            self.baseline = measured
            self.provider_tokens = usage.context_tokens


class ContextBinding:
    """Bind a request's calibration to what its selected wire client replays.

    Credential choice happens after SessionStreamFn prepares a request. A
    pre-route native count could include reasoning that a different account,
    endpoint or protocol cannot replay. This small local object lets the body
    builder finalize the already-tokenized measure and the usage observer save
    that exact boundary. Failed attempts never advance the tracker.
    """

    def __init__(self, tracker: ContextTokenTracker, measured: RequestMeasure) -> None:
        self.tracker = tracker
        self.initial = measured
        self.measured = measured


def bind_native_context(
    request: ChatRequest, endpoint: str, protocol: str, scope: str | None, slope: float
) -> ChatRequest:
    """Finalize native admission after credentials and replay validity are known.

    OpenAI can discard reasoning preceding the latest user turn. Invalidate
    calibration at that boundary and count only continuations after it, using
    reported reasoning tokens. Google thought signatures are retained protocol
    bytes, but its thoughtsTokenCount measures generated work, not evidence of
    charged input on replay; never convert that output counter into input.
    """
    last_user = next(
        (
            index
            for index in range(len(request.messages) - 1, -1, -1)
            if request.messages[index].role == "user"
        ),
        -1,
    )
    validity = []
    reasoning = 0
    for index, message in enumerate(request.messages):
        native = replay_items(message, request.model, endpoint, protocol, scope)
        validity.append(native is not None)
        if (
            protocol == "openai-responses"
            and native is not None
            and any(
                item.get("type") == "reasoning" and item.get("encrypted_content") for item in native
            )
            and index > last_user
            and message.usage is not None
        ):
            reasoning += max(0, message.usage.reasoning_tokens)
    update: dict[str, Any] = {"native_context_tokens": reasoning}
    binding = request.context_binding
    if isinstance(binding, ContextBinding):
        base = binding.initial
        if base.model != model_key(request):
            base = measure_request(request)
        measured = replace(
            base,
            prefix=(
                *base.prefix,
                protocol,
                endpoint,
                scope,
                request.messages[last_user].id if last_user >= 0 else None,
            ),
            messages=tuple((key, valid) for key, valid in zip(base.messages, validity)),
            tokens=base.tokens + reasoning,
            native_tokens=reasoning,
        )
        binding.measured = measured
        pair = binding.tracker.reconcile(measured, slope)
        update.update(
            {
                "context_tokens_hint": pair[0] if pair else request.context_tokens_hint,
                "context_tokens_hint_measured": pair[1] if pair else None,
                "context_tokens_hint_model": model_key(request) if pair else None,
            }
        )
    return request.model_copy(update=update)

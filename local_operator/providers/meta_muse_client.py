"""Muse Responses compatibility without changing other OpenAI-compatible hosts.

Meta's observed contract accepts auto-selected function tools, not Codex headers,
custom tools or tool_choice. Keep this boundary explicit rather than pretending a
static Muse key is a ChatGPT OAuth bearer. Protocol source: OpenCodex #3191 and
its current meta-muse adapter; independently implemented, not vendor-supported.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import httpx

from local_operator.harness.types import (
    ChatRequest,
    StreamEndEvent,
    StreamEvent,
    StreamToolCallDelta,
)
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.failover import ProviderError

ATTRIBUTION_REFUSAL = (
    "Muse Code subscription billing could not be verified for this response. "
    "Run /login meta-muse and try again. If this repeats, update lop or contact support. "
    "No PAYG fallback was selected."
)
_ACTIVE_OBSERVATION: ContextVar[dict[str, Any] | None] = ContextVar(
    "meta_muse_observation", default=None
)

if TYPE_CHECKING:
    from local_operator.providers.auth_store import OAuthAccess


def tool_aliases(request: ChatRequest) -> dict[str, str]:
    """Give invalid names deterministic, collision-free Meta-compatible spellings."""
    originals = {tool.name for tool in request.tools}
    # Reserve original valid names first: a-b must not shadow a_b, regardless
    # of tool registration order or the contents of a prior replay transcript.
    occupied = {name for name in originals if re.fullmatch(r"[A-Za-z0-9_]{1,64}", name)}
    mapping: dict[str, str] = {}
    for name in sorted(originals):
        alias = re.sub(r"[^A-Za-z0-9_]", "_", name)[:64] or "tool"
        if alias != name:
            stem = alias
            digest = hashlib.sha256(name.encode()).hexdigest()[:12]
            attempt = 0
            # A hash candidate can itself be a reserved ORIGINAL name. Check
            # every final spelling; numeric retries guarantee progress without
            # assuming distinct digests and stay stable across replay/order.
            while alias in occupied:
                suffix = "_" + digest + (f"_{attempt}" if attempt else "")
                alias = stem[: 64 - len(suffix)] + suffix
                attempt += 1
        occupied.add(alias)
        mapping[name] = alias
    return mapping


def _rename_items(value: Any, mapping: dict[str, str]) -> None:
    """Only function-call names change; never mutate opaque encrypted reasoning."""
    if isinstance(value, dict):
        if value.get("type") == "function_call" and isinstance(value.get("name"), str):
            value["name"] = mapping.get(value["name"], value["name"])
        for child in value.values():
            _rename_items(child, mapping)
    elif isinstance(value, list):
        for child in value:
            _rename_items(child, mapping)


class MetaMuseClient(OpenAICompatClient):
    """Narrow public-Responses dialect; no account bearer or PAYG fallback."""

    last_subscription_observation: dict[str, Any] | None = None
    last_request_id: str | None = None

    def _observe_responses_headers(self, headers: httpx.Headers) -> None:
        request_id = headers.get("x-request-id") or headers.get("request-id")
        self.last_request_id = (
            request_id if request_id and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id) else None
        )
        state = _ACTIVE_OBSERVATION.get()
        if state is not None:
            state["request_id"] = self.last_request_id

    def _observe_responses_event(self, payload: dict[str, Any]) -> None:
        from local_operator.providers.meta_muse_usage import subscription_observation

        value = (
            payload.get("subscription")
            if payload.get("type") == "response.subscription_usage"
            else None
        )
        # The normalized fixture form remains useful for compatible gateways,
        # but the actual Meta frame is the named event above, not response.*.usage.
        if value is None:
            response = payload.get("response")
            nested = response if isinstance(response, dict) else {}
            value = payload.get("subscription_usage") or nested.get("subscription_usage")
        receipt = subscription_observation(value)
        if receipt is not None:
            self.last_subscription_observation = receipt
            state = _ACTIVE_OBSERVATION.get()
            if state is not None:
                state["receipt"] = receipt

    def _headers(
        self, api_key: str | None, oauth_access: "OAuthAccess | None" = None
    ) -> dict[str, str]:
        key = oauth_access.access_token if oauth_access is not None else api_key
        if not key or not key.startswith("LLM|"):
            raise ProviderError(
                401,
                "Muse Code subscription key unavailable; run /login meta-muse.",
                kind="auth",
                retryable=False,
                fail_closed=True,
            )
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-api-version": "1.0.0",
            "Authorization": f"Bearer {key}",
        }

    def _codex_responses_mode(self, oauth_access: "OAuthAccess | None") -> bool:
        return False

    def _build_responses_body(
        self, request: ChatRequest, *, endpoint: str | None = None, scope: str | None = None
    ) -> dict[str, Any]:
        if request.tools and request.tool_choice == "required":
            raise ProviderError(
                400, "Muse Code supports automatic function-tool selection only.", retryable=False
            )
        body = super()._build_responses_body(request, endpoint=endpoint, scope=scope)
        body.pop("tool_choice", None)
        body.pop("prompt_cache_retention", None)
        body.pop("prompt_cache_key", None)
        body["store"] = False
        body["include"] = ["reasoning.encrypted_content"]
        if request.tool_choice == "none":
            # Omitting tools is the only verified way to enforce a no-tools
            # aside. Sending tool_choice:none is rejected by this endpoint.
            body.pop("tools", None)
        aliases = tool_aliases(request)
        for tool in body.get("tools", []):
            if tool.get("type") != "function":
                raise ProviderError(
                    400, "Muse Code does not support custom tools.", retryable=False
                )
            tool["name"] = aliases.get(tool["name"], tool["name"])
        _rename_items(body.get("input"), aliases)
        return body

    async def stream(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        token = _ACTIVE_OBSERVATION.set({"receipt": None, "request_id": None})
        try:
            async for event in self._stream_muse(request, api_key, oauth_access):
                yield event
        except ProviderError as original:
            # Classify upstream before dropping possibly reflected token text.
            # Only our exact local attribution refusal survives as prose.
            message = (
                ATTRIBUTION_REFUSAL
                if original.message == ATTRIBUTION_REFUSAL
                else f"Unofficial Muse Code request failed "
                f"(HTTP {original.status or 'unavailable'}). "
                "Check your subscription or sign in again; no PAYG fallback was selected."
            )
            raise ProviderError(
                original.status,
                message,
                kind=original.kind,
                retryable=original.retryable,
                retry_after_ms=original.retry_after_ms,
                request_sent=original.request_sent,
                fail_closed=True,
            ) from None
        except httpx.HTTPError:
            raise ProviderError(
                None,
                "Unofficial Muse Code connection failed; no PAYG fallback was selected.",
                kind="transient",
                retryable=True,
                transport=True,
                fail_closed=True,
            ) from None
        finally:
            _ACTIVE_OBSERVATION.reset(token)

    async def _stream_muse(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        self.last_subscription_observation = None
        self.last_request_id = None
        reverse = {alias: name for name, alias in tool_aliases(request).items()}
        async for event in super().stream(request, api_key, oauth_access):
            if isinstance(event, StreamEndEvent) and event.error is not None:
                # Error bodies can echo a submitted key. Preserve the status
                # classification, never the provider body, and prohibit a
                # hidden billable cross-provider fallback from this plan route.
                raise ProviderError(None, "Muse Code stream failed.", fail_closed=True)
            if isinstance(event, StreamToolCallDelta) and event.name:
                event = event.model_copy(update={"name": reverse.get(event.name, event.name)})
            elif isinstance(event, StreamEndEvent) and event.provider_payload:
                payload = copy.deepcopy(event.provider_payload)
                _rename_items(payload, reverse)
                event = event.model_copy(update={"provider_payload": payload})
            if isinstance(event, StreamEndEvent):
                # The terminal Responses frame may PRECEDE the meter frame.
                # The parent yields End only after draining EOF/[DONE], so this
                # gate cannot discard a late valid receipt. Context-local state
                # prevents an aside/concurrent call from borrowing an old meter.
                state = _ACTIVE_OBSERVATION.get() or {}
                receipt = state.get("receipt")
                if receipt is None:
                    raise ProviderError(
                        None, ATTRIBUTION_REFUSAL, kind="unknown", retryable=False, fail_closed=True
                    )
                event = event.model_copy(
                    update={
                        "provider_payload": {
                            **(event.provider_payload or {}),
                            "meta_muse": {
                                "subscription_usage": receipt,
                                "request_id": state.get("request_id"),
                            },
                        }
                    }
                )
            yield event

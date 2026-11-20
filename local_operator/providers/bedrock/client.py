"""Claude on Amazon Bedrock: Anthropic Messages over ``InvokeModelWithResponseStream``.

WHY A SUBCLASS (design D1). Bedrock's InvokeModel accepts the Anthropic
Messages body almost verbatim, so ``AnthropicClient``'s whole tuned request
(cache breakpoints, Saver frozen prefix, 1h TTL policy, effort/adaptive
thinking, native thinking replay) and its event loop (usage incl. cache
counters, tool_use deltas, thinking signatures) are reused unchanged. Only the
three transport hooks differ:

* ``_messages_request``: URL ``/model/{id}/invoke-with-response-stream``, a
  small body transform (``anthropic_version``, betas moved from headers into
  ``anthropic_beta``, first-party-only keys dropped through an ALLOW-LIST so a
  future first-party key can never leak into a 400), and auth — bearer or
  SigV4 over the exact bytes sent.
* ``_iter_events``: ``application/vnd.amazon.eventstream`` frames, each
  ``chunk`` carrying one base64 Anthropic event; exception frames raise through
  the Bedrock error table; the final chunk's ``amazon-bedrock-invocationMetrics``
  becomes a usage-fill event so cost never under-reports.
* ``_raise_for_status``: ``x-amzn-ErrorType``-aware mapping.

``_base_url`` is the sentinel ``bedrock-runtime:{region}``, never
``ANTHROPIC_API_URL``, which (a) switches off first-party-only features gated
on that URL (cache diagnostics, wire TTL hints) and (b) scopes native thinking
replay per Region.

The Region comes from the credential (a stored connection carries its own; the
env/ambient tier resolves it per design §5.2), so one factory-built client
fans out to one regional instance per Region, all sharing the HTTP pool.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import quote, unquote

import httpx

from local_operator.providers.bedrock import auth as bauth
from local_operator.providers.bedrock import connection as bconn
from local_operator.providers.bedrock import errors as berrors
from local_operator.providers.clients import (
    STREAM_READ_TIMEOUT_S,
    USAGE_FILL_EVENT,
    AnthropicClient,
    _encode_json_body,
)

#: Bedrock's Messages API version string (body field, not a header).
BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

#: Top-level body keys Bedrock's Anthropic Messages schema accepts. Bedrock
#: rejects unknown top-level keys with ``ValidationException``, so this is an
#: ALLOW-list: ``model``/``stream`` (URL + operation carry them), ``speed``
#: (first-party fast mode) and ``diagnostics`` (first-party cache diagnosis)
#: and anything a future ``_build_body`` adds are dropped by construction.
ALLOWED_BODY_KEYS = frozenset(
    {
        "anthropic_version",
        "anthropic_beta",
        "max_tokens",
        "system",
        "messages",
        "tools",
        "tool_choice",
        "thinking",
        "output_config",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "metadata",
    }
)

#: Betas that are first-party-only and must never reach Bedrock.
_DROPPED_BETAS = frozenset({"oauth-2025-04-20"})

EVENTSTREAM_CONTENT_TYPE = "application/vnd.amazon.eventstream"


def region_sentinel(region: str) -> str:
    return f"bedrock-runtime:{region}"


def transform_body(body: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
    """First-party Messages body + headers -> Bedrock InvokeModel body."""
    out = {key: value for key, value in body.items() if key in ALLOWED_BODY_KEYS}
    out["anthropic_version"] = BEDROCK_ANTHROPIC_VERSION
    betas = [
        beta.strip()
        for beta in (headers.get("anthropic-beta") or "").split(",")
        if beta.strip() and beta.strip() not in _DROPPED_BETAS
    ]
    if betas:
        out["anthropic_beta"] = betas
    return out


def invoke_url(region: str, model_id: str) -> str:
    """``/model/{id}/invoke-with-response-stream`` with the id fully escaped.

    ``safe=""`` so a versioned id's ``:`` becomes ``%3A`` — the same canonical
    URI botocore itself signs (pinned by the SigV4 parity test T3).
    """
    base = bconn.runtime_base(region)
    return f"{base}/model/{quote(model_id, safe='')}/invoke-with-response-stream"


async def _guarded_bytes(response: httpx.Response) -> AsyncIterator[bytes]:
    """``aiter_bytes`` with the same post-first-byte stall watchdog as SSE."""
    iterator = response.aiter_bytes().__aiter__()
    started = False
    while True:
        try:
            if started:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=STREAM_READ_TIMEOUT_S)
            else:
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise httpx.ReadTimeout(
                f"stream stalled: no data for {STREAM_READ_TIMEOUT_S:.0f}s",
                request=response.request,
            ) from exc
        started = True
        yield chunk


def _metrics_usage(metrics: Mapping[str, Any]) -> dict[str, int]:
    """``amazon-bedrock-invocationMetrics`` -> Anthropic usage field names."""
    mapping = {
        "inputTokenCount": "input_tokens",
        "outputTokenCount": "output_tokens",
        "cacheReadInputTokenCount": "cache_read_input_tokens",
        "cacheWriteInputTokenCount": "cache_creation_input_tokens",
    }
    out: dict[str, int] = {}
    for src, dst in mapping.items():
        value = metrics.get(src)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[dst] = value
    return out


async def iter_eventstream_events(
    response: httpx.Response, *, model: str = "", region: str = ""
) -> AsyncIterator[Any]:
    """Decode an InvokeModelWithResponseStream body into Anthropic events."""
    from botocore.eventstream import EventStreamBuffer

    buffer = EventStreamBuffer()
    async for chunk in _guarded_bytes(response):
        buffer.add_data(chunk)
        for message in buffer:
            headers = message.headers
            message_type = headers.get(":message-type")
            if message_type == "exception" or message_type == "error":
                error_type = headers.get(":exception-type") or headers.get(":error-code") or ""
                try:
                    payload = json.loads(message.payload or b"{}")
                except ValueError:
                    payload = {}
                text = ""
                if isinstance(payload, dict):
                    text = str(payload.get("message") or payload.get("Message") or "")
                if not text:
                    text = str(headers.get(":error-message") or "")
                raise berrors.provider_error_for(
                    str(error_type), text, None, model=model, region=region
                )
            if message_type != "event" or headers.get(":event-type") != "chunk":
                continue
            try:
                envelope = json.loads(message.payload)
                event = json.loads(base64.b64decode(envelope["bytes"]))
            except (ValueError, KeyError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            metrics = event.pop("amazon-bedrock-invocationMetrics", None)
            yield event
            if isinstance(metrics, Mapping):
                fill = _metrics_usage(metrics)
                if fill:
                    yield {"type": USAGE_FILL_EVENT, "usage": fill}


class BedrockAnthropicClient(AnthropicClient):
    """Claude on Bedrock; see the module docstring for the design."""

    def __init__(
        self,
        *,
        region: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        cache_ttl_1h_min_context_tokens: int = 0,
        subagent_cache_ttl_1h_min_context_tokens: int = 0,
        credential_timeout: float = bauth.CREDENTIAL_TIMEOUT_S,
    ) -> None:
        super().__init__(
            base_url=region_sentinel(region or "auto"),
            http_client=http_client,
            timeout=timeout,
            cache_ttl_1h_min_context_tokens=cache_ttl_1h_min_context_tokens,
            subagent_cache_ttl_1h_min_context_tokens=subagent_cache_ttl_1h_min_context_tokens,
        )
        self._region = region
        self._credential_timeout = credential_timeout
        self._ttl_1h = cache_ttl_1h_min_context_tokens
        self._subagent_ttl_1h = subagent_cache_ttl_1h_min_context_tokens
        self._regional: dict[str, BedrockAnthropicClient] = {}

    def _for_region(self, region: str) -> "BedrockAnthropicClient":
        client = self._regional.get(region)
        if client is None:
            client = BedrockAnthropicClient(
                region=region,
                http_client=self._http,
                cache_ttl_1h_min_context_tokens=self._ttl_1h,
                subagent_cache_ttl_1h_min_context_tokens=self._subagent_ttl_1h,
                credential_timeout=self._credential_timeout,
            )
            self._regional[region] = client
        return client

    async def stream(self, request, api_key, oauth_access=None):  # type: ignore[override]
        if self._region is None:
            # The router: resolve the credential's Region (pure parse; a
            # profile's own region is a config-file read) and hand the turn to
            # that Region's instance, whose ``_base_url`` scopes replay.
            try:
                conn = bconn.parse(api_key or "")
            except bconn.DescriptorError:
                conn = None
            if conn is None:
                raise berrors.no_credentials_error()
            region = await asyncio.to_thread(bauth.region_for, conn)
            async for event in self._for_region(region).stream(request, api_key, oauth_access):
                yield event
            return
        async for event in super().stream(request, api_key, oauth_access):
            yield event

    def _cache_ttl_for(
        self,
        request: Any,
        body: dict[str, Any],
        *,
        oauth: bool = False,
        account: str = "",
    ) -> str | None:
        # Bedrock answers ``ValidationException`` to a ``ttl`` on a model that
        # only has the 5m cache. The shared policy picks 1h by context size;
        # keep it only for ids this build KNOWS carry the 1h tier (every row of
        # the static Bedrock catalogue — AWS Price List lists a 1h write SKU for
        # each, read 2026-11-18). An unknown/discovered id stays on 5m.
        from local_operator.model.registry import bedrock_supports_1h_cache

        ttl = super()._cache_ttl_for(request, body, oauth=oauth, account=account)
        if ttl and not bedrock_supports_1h_cache(request.model.model_id):
            return None
        return ttl

    async def _messages_request(
        self, body: dict[str, Any], headers: dict[str, str], api_key: str | None
    ) -> tuple[str, bytes, dict[str, str]]:
        assert self._region is not None  # only regional instances send
        resolved = await self._resolve(api_key)
        url = invoke_url(self._region, str(body.get("model") or ""))
        content = _encode_json_body(transform_body(body, headers))
        # Exactly the headers botocore's own client sends for
        # ``invoke_model_with_response_stream(contentType=..., accept=...)``:
        # ``accept`` is the MODEL body's MIME type and travels as
        # ``X-Amzn-Bedrock-Accept`` (the response framing is always
        # event-stream). Mirroring botocore keeps our SignedHeaders identical
        # to the SDK's (pinned by the SigV4 parity test).
        base_headers = {
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-Accept": "application/json",
        }
        return url, content, bauth.sign_headers(resolved, "POST", url, content, base_headers)

    async def _resolve(self, api_key: str | None) -> bauth.ResolvedAuth:
        if not api_key:
            raise berrors.no_credentials_error()
        try:
            resolved = await bauth.resolve(api_key, timeout=self._credential_timeout)
        except bauth.BedrockCredentialError as exc:
            raise exc.provider_error from exc
        if resolved.region != self._region:
            # A regional instance signs for ITS Region: the router chose it from
            # this same credential, so a mismatch only happens if config moved
            # mid-turn. Sign for the instance's Region, which the URL names.
            resolved = bauth.ResolvedAuth(
                region=str(self._region), bearer=resolved.bearer, credentials=resolved.credentials
            )
        return resolved

    def _iter_events(self, response: httpx.Response) -> AsyncIterator[Any]:
        return iter_eventstream_events(
            response, model=_model_from_url(response), region=self._region or ""
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        berrors.raise_for_status(
            response, model=_model_from_url(response), region=self._region or ""
        )


def _model_from_url(response: httpx.Response) -> str:
    try:
        path = response.request.url.raw_path.decode("ascii").split("?", 1)[0]
    except (RuntimeError, AttributeError, UnicodeDecodeError):
        return ""
    parts = path.split("/")
    if "model" in parts:
        index = parts.index("model")
        if index + 1 < len(parts):
            return unquote(parts[index + 1])
    return ""

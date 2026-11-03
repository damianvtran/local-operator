"""Conservative native billing qualification, not a live provider guarantee.

A stated charge survives malformed later metadata and transport failure. Completion,
identity and reconciliation are independent facts; none manufactures a zero bill.
Rates describe the reviewed historical boundary, not immutable vendor pricing.
"""

# pyright: strict
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal

from .json_types import JSONValue, amount, count, object_value, parse_json
from .sse_stream import iter_sse

REQUEST_MODELS = frozenset(("default", "claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"))
SERVED_MODELS = frozenset(("claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"))
PROVIDERS = frozenset(("anthropic", "Anthropic"))
RATE_VERSION = "sonnet5.5-native-source-observed@2026-11-03"
RATES = {
    "input_tokens": Decimal(".000002"),
    "output_tokens": Decimal(".00001"),
    "cache_read_input_tokens": Decimal(".0000001"),
    "cache_creation_input_tokens": Decimal(".0000025"),
}
STOPS = frozenset(("end_turn", "tool_use", "max_tokens", "stop_sequence", "pause_turn", "refusal"))
CACHE_FIELDS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
COST_FIELDS = (
    "upstream_inference_cost",
    "upstream_inference_prompt_cost",
    "upstream_inference_completions_cost",
)


@dataclass(frozen=True)
class ProviderEvidence:
    """Operator-asserted configuration provenance, never proof of live routing."""

    configured_provider: str
    allow_fallbacks: bool
    provenance_sha256: str

    def __post_init__(self) -> None:
        if self.configured_provider != "anthropic" or self.allow_fallbacks is not False:
            raise ValueError("only anthropic without fallbacks is supported")
        if re.fullmatch(r"[0-9a-fA-F]{64}", self.provenance_sha256) is None:
            raise ValueError("provenance_sha256 must be a full SHA-256 digest")


@dataclass(frozen=True)
class Observation:
    actual_usd: Decimal | None
    terminal_complete: bool
    supported_boundary: bool
    qualified_complete: bool
    faults: tuple[str, ...]
    served_model: str | None
    provider_evidence: str


def validate_request(body: dict[str, JSONValue], evidence: ProviderEvidence) -> None:
    """Validate request overrides without consulting private source-tree paths."""
    evidence.__post_init__()
    model = body.get("model")
    if model is not None and (not isinstance(model, str) or model not in REQUEST_MODELS):
        raise ValueError("unsupported requested model")
    provider = body.get("provider")
    if provider is not None:
        pin = object_value(provider)
        if (
            set(pin) != {"order", "allow_fallbacks"}
            or pin.get("order") != ["anthropic"]
            or pin.get("allow_fallbacks") is not False
        ):
            raise ValueError("unsupported requested provider")
    if body.get("service_tier") not in (None, "default") or body.get("speed") not in (
        None,
        "standard",
    ):
        raise ValueError("unsupported billing tier")


@dataclass
class _State:
    counts: dict[str, int] = field(default_factory=dict[str, int])
    cache: dict[str, int] = field(default_factory=dict[str, int])
    details: dict[str, Decimal] = field(default_factory=dict[str, Decimal])
    faults: set[str] = field(default_factory=set[str])
    models: list[JSONValue] = field(default_factory=list[JSONValue])
    providers: list[JSONValue] = field(default_factory=list[JSONValue])
    known: Decimal | None = None
    terminal: bool = False
    final_usage: bool = False
    stop: JSONValue = None
    stream_error: bool = False
    started: bool = False

    def identity(self, obj: dict[str, JSONValue]) -> None:
        if "model" in obj:
            self.models.append(obj["model"])
        if "provider" in obj:
            self.providers.append(obj["provider"])
        if obj.get("service_tier") not in (None, "default") or obj.get("speed") not in (
            None,
            "standard",
        ):
            self.faults.add("unsupported-billing-tier")

    def merge(self, value: JSONValue) -> bool:
        try:
            part = object_value(value)
        except ValueError:
            self.faults.add("usage-not-object")
            return False
        valid = True
        # Charges are extracted independently: malformed tokens cannot erase a bill.
        if "cost" in part:
            try:
                self.known = amount(part["cost"])
            except (ValueError, ArithmeticError):
                self.faults.add("invalid-cost")
                valid = False
        for key in RATES:
            if key in part:
                try:
                    self.counts[key] = count(part[key])
                except ValueError:
                    self.faults.add("invalid-token-count")
                    valid = False
        for key in ("cache_creation", "cost_details"):
            if key not in part:
                continue
            try:
                raw = object_value(part[key])
                if key == "cache_creation":
                    self.cache = {name: count(raw[name]) for name in CACHE_FIELDS if name in raw}
                else:
                    self.details = {name: amount(raw[name]) for name in COST_FIELDS if name in raw}
            except (ValueError, ArithmeticError):
                self.faults.add("invalid-usage-detail")
                valid = False
        if part.get("is_byok") is not None and part.get("is_byok") is not False:
            self.faults.add("unsupported-billing-basis")
            valid = False
        return valid

    def accept(self, value: JSONValue, *, sse: bool) -> None:
        try:
            obj = object_value(value)
        except ValueError:
            self.faults.add("response-not-object")
            return
        self.identity(obj)
        kind = obj.get("type")
        if kind == "error" or "error" in obj:
            self.stream_error = True
        nested = obj.get("message")
        if isinstance(nested, dict):
            self.identity(nested)
            if "usage" in nested:
                self.merge(nested["usage"])
        part = obj.get("usage")
        valid = self.merge(part) if "usage" in obj else False
        if not sse:
            self.stop = obj.get("stop_reason")
            self.terminal = (
                isinstance(self.stop, str) and self.stop in STOPS and not self.stream_error
            )
            self.final_usage = (
                valid
                and isinstance(part, dict)
                and all(key in part for key in ("input_tokens", "output_tokens", "cost"))
            )
            return
        if self.terminal:
            self.faults.add("event-after-terminal")
        if kind == "message_start":
            if self.started or self.terminal:
                self.faults.add("duplicate-or-late-start")
            self.started = True
        elif kind == "message_delta":
            if not self.started:
                self.faults.add("delta-before-start")
            delta = obj.get("delta")
            reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            if reason is not None:
                self.stop = reason
                self.final_usage = (
                    valid and isinstance(part, dict) and "cost" in part and "output_tokens" in part
                )
        elif kind == "message_stop":
            if self.terminal or not self.started:
                self.faults.add("invalid-stop-order")
            self.terminal = True

    def observation(self, evidence: ProviderEvidence) -> Observation:
        token_valid = all(key in self.counts for key in ("input_tokens", "output_tokens"))
        identity_ok = bool(self.models) and all(
            isinstance(model, str) and model in SERVED_MODELS for model in self.models
        )
        provider_ok = all(
            isinstance(provider, str) and provider in PROVIDERS for provider in self.providers
        )
        rates_ok = False
        if token_valid and self.known is not None:
            creation = self.counts.get("cache_creation_input_tokens", 0)
            cache_ok = (
                self.cache.get("ephemeral_5m_input_tokens", 0) == creation
                and self.cache.get("ephemeral_1h_input_tokens", 0) == 0
            )
            context = sum(
                self.counts.get(key, 0)
                for key in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            )
            expected = sum(
                (Decimal(self.counts.get(key, 0)) * rate for key, rate in RATES.items()), Decimal(0)
            )
            output = Decimal(self.counts["output_tokens"]) * RATES["output_tokens"]
            expected_details = {
                "upstream_inference_cost": expected,
                "upstream_inference_prompt_cost": expected - output,
                "upstream_inference_completions_cost": output,
            }
            rates_ok = (
                cache_ok and context <= 200000 and abs(expected - self.known) <= Decimal("1e-12")
            )
            rates_ok = rates_ok and all(
                abs(value - expected_details[key]) <= Decimal("1e-12")
                for key, value in self.details.items()
            )
        if not token_valid:
            self.faults.add("missing-valid-token-usage")
        if self.stream_error:
            self.faults.add("stream-error")
        terminal = (
            self.terminal
            and isinstance(self.stop, str)
            and self.stop in STOPS
            and self.final_usage
            and not self.stream_error
        )
        if not terminal:
            self.faults.add("missing-terminal-or-final-billing")
        if not identity_ok:
            self.faults.add("unsupported-served-model")
        if not provider_ok:
            self.faults.add("unsupported-provider-or-pin")
        if not rates_ok:
            self.faults.add("unsupported-rate-or-usage")
        supported = identity_ok and provider_ok and rates_ok
        model = self.models[-1] if identity_ok else None
        return Observation(
            actual_usd=self.known,
            terminal_complete=terminal
            and not self.faults.intersection(
                {
                    "event-after-terminal",
                    "duplicate-or-late-start",
                    "invalid-stop-order",
                    "delta-before-start",
                    "unparsed-sse-frame",
                    "invalid-sse-json",
                    "response-too-large",
                }
            ),
            supported_boundary=supported,
            qualified_complete=terminal and supported and not self.faults,
            faults=tuple(sorted(self.faults)),
            served_model=model if isinstance(model, str) else None,
            provider_evidence=(
                "asserted-config-provenance:"
                + evidence.provenance_sha256.lower()
                + (
                    "; response-provider-approved"
                    if self.providers and provider_ok
                    else (
                        "; response-provider-not-exposed"
                        if provider_ok
                        else "; response-provider-unqualified"
                    )
                )
                + "; not-a-live-routing-guarantee"
            ),
        )


class _ResponseTooLarge(ValueError):
    pass


async def inspect_response(
    content: bytes | AsyncIterable[bytes],
    ctype: str,
    evidence: ProviderEvidence,
    *,
    max_response_bytes: int,
) -> Observation:
    """Inspect bounded bytes; interrupted streams retain only already framed bills.

    Cancellation is converted to an unqualified observation so the caller can persist
    a known partial charge before deciding how to propagate its own cancellation.
    """
    evidence.__post_init__()
    if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")
    state = _State()

    async def source() -> AsyncIterator[bytes]:
        if isinstance(content, bytes):
            yield content
        else:
            async for chunk in content:
                yield chunk

    async def chunks() -> AsyncIterator[bytes]:
        consumed = 0
        async for chunk in source():
            remaining = max_response_bytes - consumed
            if len(chunk) > remaining:
                # Process complete frames within the bound before marking overflow.
                if remaining:
                    yield chunk[:remaining]
                raise _ResponseTooLarge
            consumed += len(chunk)
            yield chunk

    sse = ctype.split(";", 1)[0].strip().lower() == "text/event-stream"
    try:
        if sse:
            async for frame, data in iter_sse(chunks()):
                if data is None:
                    if any(line and not line.startswith(b":") for line in frame.splitlines()):
                        state.faults.add("unparsed-sse-frame")
                    continue
                if any(
                    line.startswith(b"event:") and line[6:].strip() == b"error"
                    for line in frame.splitlines()
                ):
                    state.stream_error = True
                if data.strip() == b"[DONE]":
                    continue
                try:
                    state.accept(parse_json(data), sse=True)
                except (ValueError, UnicodeError, ArithmeticError, RecursionError):
                    state.faults.add("invalid-sse-json")
        else:
            raw = bytearray()
            async for chunk in chunks():
                raw.extend(chunk)
            try:
                state.accept(parse_json(bytes(raw)), sse=False)
            except (ValueError, UnicodeError, ArithmeticError, RecursionError):
                state.faults.add("invalid-json")
    except _ResponseTooLarge:
        state.faults.add("response-too-large")
        state.stream_error = True
    except asyncio.CancelledError:
        state.faults.add("stream-cancelled")
        state.stream_error = True
    except Exception:
        # Transport exception messages may contain private URLs or payloads.
        state.stream_error = True
    return state.observation(evidence)

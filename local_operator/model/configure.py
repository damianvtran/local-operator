"""Model configuration on top of the new provider layer.

Rewritten for the harness rewrite (docs/REWRITE.md §B). The public surface
legacy code depends on is preserved:

- :class:`ModelConfiguration` (plus ``.spec``, the harness ``ModelSpec``).
- :func:`configure_model` — same signature, returns a ``ModelConfiguration``.
- :func:`validate_model` — same endpoints as the legacy if/elif chain, now a
  per-provider descriptor table.
- :func:`calculate_cost`, ``DEFAULT_TEMPERATURE``, ``DEFAULT_TOP_P``.

New: :func:`create_stream_fn` builds the ``LoopConfig.stream_fn`` from an
:class:`~local_operator.providers.auth_store.AuthStore`, resolving API keys
and dispatching to the right wire client through
:func:`~local_operator.providers.failover.stream_with_failover`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, Optional

import requests
from pydantic import SecretStr

from local_operator.harness.types import AbortSignal, ChatRequest, ModelSpec, StreamEvent
from local_operator.model.registry import ModelInfo, get_model_info

DEFAULT_TEMPERATURE = 0.2
"""Default temperature value for language models."""
DEFAULT_TOP_P = 0.9
"""Default top_p value for language models."""

# Per-hosting defaults preserved byte-for-byte from the legacy chain so
# existing config files and CLI invocations keep picking the same model.
DEFAULT_MODEL_NAMES: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o",
    "openrouter": "google/gemini-2.0-flash-001",
    "anthropic": "claude-3-5-sonnet-latest",
    "kimi": "moonshot-v1-32k",
    "alibaba": "qwen-plus",
    "google": "gemini-2.0-flash-001",
    "mistral": "mistral-large-latest",
    "radient": "auto",
    "xai": "grok-3",
}

# Sensible ModelSpec fallbacks when the legacy registry knows nothing.
UNKNOWN_CONTEXT_WINDOW = 128_000
UNKNOWN_MAX_OUTPUT = 8_192


class ModelConfiguration:
    """Configuration for one model on one hosting provider.

    Legacy attributes are unchanged (``hosting``, ``name``, ``instance``,
    ``info``, ``api_key``, sampling knobs); ``spec`` is the new harness
    descriptor consumed by wire clients. ``instance`` is ``None`` in the new
    engine — streaming happens through ``LoopConfig.stream_fn``.
    """

    hosting: str
    name: str
    instance: Any
    info: ModelInfo
    api_key: Optional[SecretStr]
    temperature: float
    top_p: float
    top_k: Optional[int]
    max_tokens: Optional[int]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop: Optional[list[str]]
    seed: Optional[int]
    spec: ModelSpec

    def __init__(
        self,
        hosting: str,
        name: str,
        instance: Any = None,
        info: ModelInfo | None = None,
        api_key: Optional[SecretStr] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
        spec: ModelSpec | None = None,
    ) -> None:
        self.hosting = hosting
        self.name = name
        self.instance = instance
        self.info = info or ModelInfo(id=name, name=name, description="Unknown model")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.seed = seed
        self.spec = spec or build_model_spec(hosting, name, self.info)


def build_model_spec(hosting: str, model_name: str, info: ModelInfo | None = None) -> ModelSpec:
    """Derive a harness ``ModelSpec`` from the legacy registry when known."""
    from local_operator.providers.registry import get_provider_definition

    canonical = "test" if hosting == "noop" else hosting
    if info is None:
        try:
            info = get_model_info(canonical, model_name)
        except (ValueError, KeyError):  # legacy registry KeyErrors on unknown openai models
            info = None

    context_window = UNKNOWN_CONTEXT_WINDOW
    max_output = UNKNOWN_MAX_OUTPUT
    supports_images = True
    supports_cache = False
    if info is not None:
        # Legacy sentinels: -1 means "no data", not a real limit.
        if info.context_window and info.context_window > 0:
            context_window = info.context_window
        if info.max_tokens and info.max_tokens > 0:
            max_output = info.max_tokens
        if info.supports_images is not None:
            supports_images = info.supports_images
        supports_cache = info.supports_prompt_cache

    definition = get_provider_definition(canonical)
    lowered = model_name.lower()
    reasoning = any(marker in lowered for marker in ("o1", "o3", "reasoner", "thinking", "deep-research"))

    return ModelSpec(
        provider=canonical,
        model_id=model_name,
        context_window=context_window,
        max_output_tokens=max_output,
        supports_tools=True,
        supports_images=supports_images,
        supports_prompt_cache=supports_cache,
        base_url=definition.base_url if definition else None,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Validation — legacy endpoints, table-driven
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ValidationDescriptor:
    """Where and how a provider lists its models for validation."""

    url: str
    header_style: str = "bearer"  # bearer | x-api-key | x-goog-api-key | none
    extra_headers: Mapping[str, str] = dataclasses.field(default_factory=dict)


VALIDATION_ENDPOINTS: dict[str, ValidationDescriptor] = {
    "deepseek": ValidationDescriptor("https://api.deepseek.com/v1/models"),
    "openai": ValidationDescriptor("https://api.openai.com/v1/models"),
    "openrouter": ValidationDescriptor("https://openrouter.ai/api/v1/models"),
    "radient": ValidationDescriptor("https://api.radienthq.com/v1/models"),
    "anthropic": ValidationDescriptor(
        "https://api.anthropic.com/v1/models",
        header_style="x-api-key",
        extra_headers={"anthropic-version": "2023-06-01"},
    ),
    "kimi": ValidationDescriptor("https://api.moonshot.cn/v1/models"),
    "alibaba": ValidationDescriptor("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"),
    "google": ValidationDescriptor(
        "https://generativelanguage.googleapis.com/v1/models", header_style="x-goog-api-key"
    ),
    "mistral": ValidationDescriptor("https://api.mistral.ai/v1/models"),
    "ollama": ValidationDescriptor("http://localhost:11434/api/tags", header_style="none"),
    "xai": ValidationDescriptor("https://api.x.ai/v1/models"),
}


def _check_model_exists_payload(hosting: str, model: str, response_data: dict[str, Any]) -> bool:
    """Check if a model exists in the provider's response data.

    Payload shapes differ per provider (Google nests under ``models`` with
    ``models/`` prefixes; Ollama uses ``name``; the rest use ``data`` with
    ``id`` or ``name``). Anthropic ``-latest`` aliases match by prefix.
    """
    if hosting == "google":
        models = response_data.get("models", [])
        return any(m.get("name", "").replace("models/", "") == model for m in models)

    if hosting == "ollama":
        models = response_data.get("models", [])
        return any(m.get("name", "") == model for m in models)

    models = response_data.get("data", [])
    if not models:
        return False

    if hosting == "anthropic" and model.endswith("-latest"):
        base_model = model.replace("-latest", "")
        return any(m.get("id", "").startswith(base_model) for m in models)

    for m in models:
        model_id = m.get("id") or m.get("name") or ""
        if model_id == model:
            return True
    return False


def validate_model(hosting: str, model: str, api_key: SecretStr | str) -> bool:
    """Validate that the model exists and the key is accepted.

    Same endpoints and semantics as the legacy chain; network errors raise
    ``requests.exceptions.RequestException`` (callers catch and report).
    """
    descriptor = VALIDATION_ENDPOINTS.get(hosting)
    if descriptor is None:
        return True

    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else str(api_key)
    headers: dict[str, str] = dict(descriptor.extra_headers)
    if descriptor.header_style == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif descriptor.header_style == "x-api-key":
        headers["x-api-key"] = key
    elif descriptor.header_style == "x-goog-api-key":
        headers["x-goog-api-key"] = key

    # Byte-compatible call shape: omit the headers kwarg entirely when empty
    # (legacy tests assert the exact call arguments, e.g. ollama).
    response = (
        requests.get(descriptor.url, headers=headers) if headers else requests.get(descriptor.url)
    )
    if response.status_code == 200:
        return _check_model_exists_payload(hosting, model, response.json())
    return False


# ---------------------------------------------------------------------------
# Model info via OpenRouter/Radient clients (duck-typed; no legacy imports)
# ---------------------------------------------------------------------------


def _extra(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field the client schema does not declare.

    The legacy listing models set ``extra="allow"``, so provider fields like
    ``context_length`` and ``top_provider`` arrive as pydantic extras rather
    than attributes. Reading them generically keeps this mapper working for
    every listing-backed provider (OpenRouter, Radient) without teaching the
    wire schemas about each field.
    """
    value = getattr(obj, key, None)
    if value is None and hasattr(obj, "model_extra"):
        extra = obj.model_extra or {}
        value = extra.get(key)
    return default if value is None else value


def _info_from_listing(listing: Any, model_name: str, template: ModelInfo, source: str) -> ModelInfo:
    """Find ``model_name`` in a ``list_models()`` payload and describe it.

    ``listing`` is the legacy clients' ``list_models()`` result (object with
    ``.data`` of items carrying ``id``, ``description``, ``pricing``).

    Beyond price this maps the fields the harness depends on at runtime, all of
    which used to fall through to the "unknown model" template:

    - ``context_window`` — compaction thresholds are derived from it, so an
      unknown (-1) window silently disables compaction for the whole session.
      A model routed through several providers advertises the largest window at
      the top level and the routed one under ``top_provider``; we take the
      smaller so a prompt sized to the window cannot 400 on the provider that
      actually serves it.
    - ``supports_prompt_cache`` — gates cache_control emission; inferred from
      the presence of a cache-read price.
    - ``supports_images`` — gates the snapcompact vision strategy.
    """
    for model in listing.data:
        if model.id != model_name:
            continue
        info = template.model_copy(deep=True)
        # Providers quote price per token here; normalize to per-million.
        info.input_price = float(model.pricing.prompt) * 1_000_000
        info.output_price = float(model.pricing.completion) * 1_000_000
        info.description = model.description

        top = _extra(model, "top_provider", {}) or {}
        if not isinstance(top, dict):
            top = getattr(top, "model_dump", lambda: {})()
        windows = [
            int(w)
            for w in (_extra(model, "context_length"), top.get("context_length"))
            if w
        ]
        if windows:
            info.context_window = min(windows)
        max_out = top.get("max_completion_tokens")
        if max_out:
            info.max_tokens = int(max_out)

        arch = _extra(model, "architecture", {}) or {}
        if not isinstance(arch, dict):
            arch = getattr(arch, "model_dump", lambda: {})()
        modalities = arch.get("input_modalities") or []
        if modalities:
            info.supports_images = "image" in modalities

        pricing_extra = getattr(model.pricing, "model_extra", None) or {}
        cache_read = pricing_extra.get("input_cache_read")
        cache_write = pricing_extra.get("input_cache_write")
        if cache_read is not None:
            info.supports_prompt_cache = True
            info.cache_reads_price = float(cache_read) * 1_000_000
            # Providers with implicit caching quote no write price; the read
            # price is the only signal that caching exists at all.
            info.cache_writes_price = (
                float(cache_write) * 1_000_000 if cache_write is not None else info.input_price
            )
        return info
    raise ValueError(f"Model not found from {source} models API: {model_name}")


def get_model_info_from_openrouter(client: Any, model_name: str) -> ModelInfo:
    """Model info from the OpenRouter models listing (legacy-compatible)."""
    from local_operator.model.registry import openrouter_default_model_info

    return _info_from_listing(client.list_models(), model_name, openrouter_default_model_info, "openrouter")


def get_model_info_from_radient(client: Any, model_name: str) -> ModelInfo:
    """Model info from the Radient models listing (legacy-compatible)."""
    from local_operator.model.registry import radient_default_model_info

    return _info_from_listing(client.list_models(), model_name, radient_default_model_info, "radient")


# ---------------------------------------------------------------------------
# configure_model
# ---------------------------------------------------------------------------


def configure_model(
    hosting: str,
    model_name: str,
    credential_manager: Any = None,
    model_info_client: Any = None,
    env_config: Any = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> ModelConfiguration:
    """Configure a model for ``hosting``.

    Key resolution happens lazily at stream time through the auth store in
    the new engine; this function only records a best-effort ``api_key`` for
    legacy consumers (no interactive prompting — headless-safe). Raises
    ``ValueError`` for missing hosting, unknown hosting, or ollama without a
    model name.
    """
    if not hosting:
        raise ValueError("Hosting is required")

    canonical = "test" if hosting == "noop" else hosting
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(canonical)
    if definition is None:
        raise ValueError(f"Unsupported hosting platform: {hosting}")

    if canonical == "ollama" and not model_name:
        raise ValueError("Model is required for ollama hosting")
    if not model_name:
        model_name = DEFAULT_MODEL_NAMES.get(canonical, "")

    # Best-effort static key for legacy consumers; the cascade at stream time
    # re-resolves (OAuth refresh, env, stored keys) — see AuthStore.
    api_key: Optional[SecretStr] = None
    if credential_manager is not None and isinstance(definition.env_keys, str):
        try:
            secret = credential_manager.get_credential(definition.env_keys)
        except Exception:
            secret = None
        if secret is not None and secret.get_secret_value():
            api_key = secret

    model_info: ModelInfo
    if model_info_client is not None:
        if canonical == "openrouter":
            model_info = get_model_info_from_openrouter(model_info_client, model_name)
        elif canonical == "radient":
            model_info = get_model_info_from_radient(model_info_client, model_name)
        else:
            raise ValueError(f"Model info client not supported for hosting: {hosting}")
    else:
        try:
            model_info = get_model_info(canonical, model_name)
        except (ValueError, KeyError):
            model_info = ModelInfo(id=model_name, name=model_name, description="Unknown model")

    spec = build_model_spec(canonical, model_name, model_info)
    # Radient base URL is env-overridable (legacy EnvConfig behaviour).
    if canonical == "radient" and env_config is not None:
        base_url = getattr(env_config, "radient_api_base_url", None)
        if base_url:
            spec = spec.model_copy(update={"base_url": base_url})

    return ModelConfiguration(
        hosting=hosting,
        name=model_name,
        instance=None,
        info=model_info,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        seed=seed,
        spec=spec,
    )


# ---------------------------------------------------------------------------
# stream_fn factory
# ---------------------------------------------------------------------------


def create_stream_fn(
    auth_store: Any, settings: Mapping[str, Any] | None = None
) -> Callable[[ChatRequest, AbortSignal | None], AsyncIterator[StreamEvent]]:
    """Build the ``LoopConfig.stream_fn`` for a session.

    Resolves the API key through ``auth_store`` (7-step cascade + OAuth
    refresh), picks the wire client from the request's ``ModelSpec``, and
    wraps the call in credential-rotation + model-fallback failover.
    """
    from local_operator.providers.clients import client_for_spec
    from local_operator.providers.failover import stream_with_failover

    def client_for(spec: ModelSpec) -> Any:
        return client_for_spec(spec)

    async def stream_fn(request: ChatRequest, signal: AbortSignal | None) -> AsyncIterator[StreamEvent]:
        async for event in stream_with_failover(
            request, auth_store, settings, client_for, signal=signal
        ):
            yield event

    return stream_fn


def calculate_cost(model_info: ModelInfo, input_tokens: int, output_tokens: int) -> float:
    """Cost of a request from per-million token pricing.

    Raises:
        ValueError: on any arithmetic failure (keeps the legacy contract).
    """
    try:
        input_cost = (float(input_tokens) / 1_000_000.0) * model_info.input_price
        output_cost = (float(output_tokens) / 1_000_000.0) * model_info.output_price
        total_cost = input_cost + output_cost
        return total_cost
    except Exception as e:
        raise ValueError(f"Error calculating cost: {e}") from e

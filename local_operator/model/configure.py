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
import functools
import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Optional

import requests
from pydantic import BaseModel, SecretStr

from local_operator.harness.types import AbortSignal, ChatRequest, ModelSpec, StreamEvent
from local_operator.model.catalogue import LISTING_PROVIDERS, cached_listing
from local_operator.model.registry import ModelInfo, get_model_info

logger = logging.getLogger("local_operator.model.configure")

if TYPE_CHECKING:
    # Import-time cost only for type checkers: the listing clients pull in the
    # whole requests/provider surface, and this module is on the CLI startup
    # path. Nothing here touches them at runtime.
    from typing import Protocol

    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.clients.radient import RadientListModelsResponse
    from local_operator.credentials import CredentialManager
    from local_operator.env import EnvConfig
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.clients import WireClient

    #: Either provider's ``list_models()`` payload. The two schemas are
    #: structurally identical (``data`` of items with ``id``/``description``/
    #: ``pricing``) and both allow extras, so one mapper serves both.
    ListingResponse = OpenRouterListModelsResponse | RadientListModelsResponse

    class ModelListingClient(Protocol):
        """A provider client that can enumerate its models.

        Structural on purpose: ``configure_model`` picks the mapper from the
        hosting name, so pinning the concrete client class here would only
        force a narrowing cast at the branch that already knows which one it
        holds.
        """

        def list_models(self) -> ListingResponse: ...


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
    reasoning = any(
        marker in lowered for marker in ("o1", "o3", "reasoner", "thinking", "deep-research")
    )

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
    "alibaba": ValidationDescriptor(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"
    ),
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
# Model info via the OpenRouter/Radient listing clients
# ---------------------------------------------------------------------------


def _extra(model: BaseModel, key: str) -> Any:
    """Read a provider field the wire schema does not declare.

    The listing schemas set ``extra="allow"``, so provider fields like
    ``context_length`` and ``top_provider`` land in ``model_extra`` instead of
    becoming declared attributes. Values are whatever JSON the provider sent,
    hence ``Any``.
    """
    return (model.model_extra or {}).get(key)


def _extra_mapping(model: BaseModel, key: str) -> Mapping[str, Any]:
    """Read an undeclared *nested object* out of the listing extras.

    Extras parsed from a live response are plain JSON dicts; a caller that
    hands the schema an already-built pydantic object instead is flattened
    back to a mapping. Anything else (a scalar, a missing key) reads as empty
    so the lookups at the call site stay uniform.
    """
    value = _extra(model, key)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump()
    return {}


def _info_from_listing(
    listing: ListingResponse, model_name: str, template: ModelInfo, source: str
) -> ModelInfo:
    """Find ``model_name`` in a ``list_models()`` payload and describe it.

    ``listing`` is either provider's ``list_models()`` result: a ``data`` list
    of items carrying ``id``, ``description`` and ``pricing``.

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

        top = _extra_mapping(model, "top_provider")
        windows = [
            int(w) for w in (_extra(model, "context_length"), top.get("context_length")) if w
        ]
        if windows:
            info.context_window = min(windows)
        max_out = top.get("max_completion_tokens")
        if max_out:
            info.max_tokens = int(max_out)

        modalities = _extra_mapping(model, "architecture").get("input_modalities") or []
        if modalities:
            info.supports_images = "image" in modalities

        pricing_extra = model.pricing.model_extra or {}
        cache_read = pricing_extra.get("input_cache_read")
        cache_write = pricing_extra.get("input_cache_write")
        # OpenRouter quotes "input_cache_read": "0" for models with no prompt
        # caching; presence alone would flip the flag and change request shape
        # for no benefit. Require a positive price.
        if cache_read is not None and float(cache_read) > 0:
            info.supports_prompt_cache = True
            info.cache_reads_price = float(cache_read) * 1_000_000
            # Providers with implicit caching quote no write price; the read
            # price is the only signal that caching exists at all.
            info.cache_writes_price = (
                float(cache_write) * 1_000_000 if cache_write is not None else info.input_price
            )
        return info
    raise ValueError(f"Model not found from {source} models API: {model_name}")


def get_model_info_from_openrouter(client: ModelListingClient, model_name: str) -> ModelInfo:
    """Model info from the OpenRouter models listing (legacy-compatible)."""
    from local_operator.model.registry import openrouter_default_model_info

    return _info_from_listing(
        client.list_models(), model_name, openrouter_default_model_info, "openrouter"
    )


def get_model_info_from_radient(client: ModelListingClient, model_name: str) -> ModelInfo:
    """Model info from the Radient models listing (legacy-compatible)."""
    from local_operator.model.registry import radient_default_model_info

    return _info_from_listing(
        client.list_models(), model_name, radient_default_model_info, "radient"
    )


def _has_real_window(info: ModelInfo) -> bool:
    """True when the registry actually knows this model's context window.

    ``-1`` and ``0`` are both "no data" sentinels in the legacy registry, and a
    placeholder entry for an aggregator carries one of them.
    """
    return bool(info.context_window and info.context_window > 0)


def _catalogue_source(provider: str) -> tuple[Any, type[Any]] | None:
    """``(client, response_model)`` for ``provider``, or None if unavailable.

    The response model comes back alongside the client because the cache stores
    the raw payload: something has to re-validate the dict into the shape
    ``_info_from_listing`` reads, and only the caller knows which shape.

    Returns None rather than raising when a client cannot be built. That is not
    hypothetical: ``OpenRouterClient`` raises ``RuntimeError`` on an empty key
    in its constructor, so a keyless install reached this path and failed
    ``configure_model`` outright — turning a metadata optimisation into "the CLI
    will not start". Enriching the catalogue is always optional.

    Imports are local: the client modules pull in provider response models that
    must stay out of the startup graph.
    """
    from pydantic import SecretStr

    try:
        if provider == "openrouter":
            from local_operator.clients.openrouter import (
                OpenRouterClient,
                OpenRouterListModelsResponse,
            )

            key = SecretStr(os.environ.get("OPENROUTER_API_KEY", ""))
            return OpenRouterClient(api_key=key), OpenRouterListModelsResponse
        if provider == "radient":
            from local_operator.clients.radient import (
                RadientClient,
                RadientListModelsResponse,
            )
            from local_operator.env import get_env_config

            key = SecretStr(os.environ.get("RADIENT_API_KEY", ""))
            base_url = get_env_config().radient_api_base_url or "https://api.radienthq.com/v1"
            return RadientClient(key, base_url), RadientListModelsResponse
    except Exception as exc:  # noqa: BLE001 - a missing key or import is not fatal
        logger.debug("no %s catalogue client: %s", provider, exc)
        return None
    return None


def _info_from_catalogue(provider: str, model_name: str, fallback: ModelInfo) -> ModelInfo:
    """Describe ``model_name`` from the cached catalogue, else ``fallback``.

    Never raises. A missing model, an unreachable listing, a malformed cache
    and provider schema drift all degrade to the caller's static fallback,
    because none of them is a reason to refuse to start a session.
    """
    source = _catalogue_source(provider)
    if source is None:
        return fallback
    client, response_model = source

    payload = cached_listing(provider, lambda: client.list_models().model_dump())
    if payload is None:
        return fallback

    try:
        listing = response_model.model_validate(payload)
    except Exception:  # noqa: BLE001 - schema drift must not break startup
        logger.debug("%s catalogue payload did not validate; using static fallback", provider)
        return fallback
    try:
        return _info_from_listing(listing, model_name, fallback, provider)
    except ValueError:
        # Not in the catalogue: a brand-new id, or a typo the provider will
        # reject anyway. The static fallback is the honest answer.
        return fallback


@functools.lru_cache(maxsize=64)
def resolve_model_info(provider: str, model_id: str) -> ModelInfo:
    """A model's real metadata: static registry first, catalogue to fill gaps.

    THE one resolution path, so the numbers a session runs on and the numbers a
    UI prices with cannot disagree. ``_cost_for`` in the TUI used to call
    ``get_model_info`` directly and therefore saw zero prices for every
    aggregator model — the session had already resolved the real ones, and the
    status band still rendered "cost unavailable".

    Memoized because callers are per-turn: the disk cache alone still costs a
    JSON parse (~25ms) on every call, which is real latency inside a render.
    Bounded because model ids are user-supplied — a typo per turn must not grow
    the map without limit. Clear it with ``resolve_model_info.cache_clear()``
    after a deliberate model switch in a long-lived process.
    """
    canonical = "test" if provider == "noop" else provider
    try:
        info = get_model_info(canonical, model_id)
    except (ValueError, KeyError):
        info = ModelInfo(id=model_id, name=model_id, description="Unknown model")
    if canonical in LISTING_PROVIDERS and not _has_real_window(info):
        info = _info_from_catalogue(canonical, model_id, info)
    return info


# ---------------------------------------------------------------------------
# configure_model
# ---------------------------------------------------------------------------


def configure_model(
    hosting: str,
    model_name: str,
    credential_manager: CredentialManager | None = None,
    model_info_client: ModelListingClient | None = None,
    env_config: EnvConfig | None = None,
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
        # Aggregators route hundreds of models, so their registry entry is a
        # placeholder (context_window -1, zero prices). Left at that, auto
        # compaction sizes itself off a 128k fallback on a 1M model and cost
        # cannot be reported at all. `resolve_model_info` fills the gap from a
        # disk-cached catalogue: one HTTP call a day, and never a blocked start.
        model_info = resolve_model_info(canonical, model_name)

    spec = build_model_spec(canonical, model_name, model_info)
    # Sampling rides on the ModelSpec: the loop builds its ChatRequest without
    # temperature/top_p, so the wire clients fall back to ``request.model.*``.
    # Without this copy an agent's stored temperature (and the server's
    # per-request ``options``) would be recorded on the ModelConfiguration and
    # then silently dropped on the way to the provider.
    spec = spec.model_copy(update={"temperature": temperature, "top_p": top_p})
    # Radient base URL is env-overridable (legacy EnvConfig behaviour).
    if canonical == "radient" and env_config is not None:
        base_url = env_config.radient_api_base_url
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


class SessionStreamFn:
    """The ``LoopConfig.stream_fn`` for one session, plus the pool it owns.

    One ``httpx.AsyncClient`` per session: every wire client shares it, and
    :meth:`close` releases the connection pool on session dispose — a fresh
    client per LLM round trip leaked one pool per turn for the process
    lifetime. ``close`` lives on the callable (rather than being returned
    alongside it) because the loop config carries nothing but the callable.
    """

    def __init__(
        self,
        auth_store: AuthStore,
        settings: Mapping[str, Any] | None,
        session_id: str | None,
    ) -> None:
        import httpx

        self._auth_store = auth_store
        self._settings = settings
        self._session_id = session_id
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))

    def _client_for(self, spec: ModelSpec) -> WireClient:
        from local_operator.providers.clients import client_for_spec

        return client_for_spec(spec, http_client=self._http)

    async def __call__(
        self, request: ChatRequest, signal: AbortSignal | None
    ) -> AsyncIterator[StreamEvent]:
        from local_operator.providers.failover import stream_with_failover

        async for event in stream_with_failover(
            request,
            self._auth_store,
            self._settings,
            self._client_for,
            signal=signal,
            session_id=self._session_id,
        ):
            yield event

    async def close(self) -> None:
        await self._http.aclose()


def create_stream_fn(
    auth_store: AuthStore,
    settings: Mapping[str, Any] | None = None,
    *,
    session_id: str | None = None,
) -> SessionStreamFn:
    """Build the ``LoopConfig.stream_fn`` for a session.

    Resolves the API key through ``auth_store`` (7-step cascade + OAuth
    refresh), picks the wire client from the request's ``ModelSpec``, and
    wraps the call in credential-rotation + model-fallback failover.

    ``session_id`` rides into the failover layer so the auth store keeps
    credential selection STICKY per session; without it the store round-robins
    on every resolve and multi-credential providers alternate accounts
    mid-conversation (cold cache prefix, alternating identity headers).
    """
    return SessionStreamFn(auth_store, settings, session_id)


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

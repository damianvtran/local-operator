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
import time
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Optional

import requests
from pydantic import BaseModel, SecretStr

from local_operator.harness.types import AbortSignal, ChatRequest, ModelSpec, StreamEvent
from local_operator.paths import config_dir
from local_operator.model.catalogue import (
    DEFAULT_TTL_S,
    LISTING_PROVIDERS,
    cached_listing,
)
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


class _UnmappableEntry(ValueError):
    """A catalogue entry that validated but whose fields could not be read.

    A subclass of ``ValueError`` on purpose. The legacy public helpers
    (:func:`get_model_info_from_openrouter` and friends) have always raised
    ``ValueError`` for "no usable answer", and callers outside this module catch
    exactly that; a fresh exception hierarchy would break them for no gain.
    Inside the module the subclass is what lets the two causes be told apart —
    "this model is not in this catalogue" is routine, while "this document says
    a context window is a dictionary" is a provider defect worth a warning.
    """


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
        try:
            return _map_entry(model, template)
        except (TypeError, ValueError, OverflowError) as exc:
            # Validation is NOT a guarantee that the mapping will succeed: the
            # listing schemas set ``extra="allow"``, so a payload whose extra
            # fields are the wrong shape validates cleanly and then blows up
            # inside `float()`/`int()`. All three types are real and reachable
            # from one upstream document:
            #
            #   {"context_length": {"max": 1000}}  -> TypeError
            #   {"context_length": "not-a-number"} -> ValueError
            #   {"context_length": NaN}            -> ValueError, since
            #                                         json.loads accepts the
            #                                         bare literal
            #   {"context_length": Infinity}       -> OverflowError, likewise
            #
            # Letting any of them escape fails session start outright — the
            # exact outcome this module exists to prevent.
            raise _UnmappableEntry(f"{source} entry for {model_name} did not map: {exc}") from exc
    raise ValueError(f"Model not found from {source} models API: {model_name}")


def _map_entry(model: Any, template: ModelInfo) -> ModelInfo:
    """One catalogue entry as a :class:`ModelInfo`, or raise on bad field shapes.

    Split out from the search loop so the conversions live inside ONE guarded
    region. Inline, the guard would have had to wrap the loop, which would make
    a legitimate "not in this catalogue" ValueError indistinguishable from a
    payload that cannot be read.
    """
    info = template.model_copy(deep=True)
    # The template is the PROVIDER's placeholder entry, so its id and name
    # describe the aggregator ("openrouter" / "OpenRouter") rather than the
    # model. Nothing in-tree reads them today, which is exactly why it is
    # worth correcting now: the next reader will reasonably expect
    # `info.id` to identify the model it just resolved.
    info.id = model.id
    info.name = getattr(model, "name", None) or model.id
    # Providers quote price per token here; normalize to per-million.
    info.input_price = float(model.pricing.prompt) * 1_000_000
    info.output_price = float(model.pricing.completion) * 1_000_000
    info.description = model.description

    top = _extra_mapping(model, "top_provider")
    windows = [int(w) for w in (_extra(model, "context_length"), top.get("context_length")) if w]
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


#: Sent as the bearer token when no key can be found. The listing endpoints are
#: PUBLIC catalogue data — verified: `GET https://openrouter.ai/api/v1/models`
#: returns 200 and all 340 models with no Authorization header at all, and 200
#: with a bogus one. The clients nevertheless refuse to construct on an empty
#: key, so a placeholder is what lets a keyless (or OAuth-only) install still
#: learn its real context window. If a provider ever starts gating the
#: catalogue, the request 401s and the whole path degrades to the static entry,
#: which is the same outcome as having no key today.
_PUBLIC_LISTING_TOKEN = "public-catalogue-read"


def _catalogue_api_key(provider: str) -> str:
    """A listing key for ``provider`` from the app's own stores, else "".

    Reading ONLY ``os.environ`` was a real defect rather than a shortcut: both
    sanctioned credential flows bypass the environment. ``local-operator
    credential update OPENROUTER_API_KEY`` writes the ``CredentialManager``
    file, and the TUI's ``/login`` writes the ``AuthStore``. So the users who
    configured credentials the app's own way were exactly the ones this
    enrichment silently skipped — their sessions streamed fine (the stream-time
    cascade reads those stores) while their band showed a 128k window and no
    cost, forever, with the failure recorded only at debug level. Every other
    key reader in the repo goes through ``CredentialManager``; this one was the
    outlier.

    The ``AuthStore`` cascade is deliberately NOT consulted: its accessor is
    async and this runs inside a synchronous render path, so awaiting it would
    mean either a nested event loop or making the whole resolver async for a
    value the public endpoint does not require. An OAuth-only user is covered by
    :data:`_PUBLIC_LISTING_TOKEN` instead.
    """
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("test" if provider == "noop" else provider)
    env_keys = getattr(definition, "env_keys", None)
    names = [env_keys] if isinstance(env_keys, str) else []
    for name in names:
        from_env = os.environ.get(name, "")
        if from_env:
            return from_env

    try:
        from local_operator.credentials import CredentialManager

        manager = CredentialManager(config_dir())
        for name in names:
            secret = manager.get_credential(name)
            if secret is not None and secret.get_secret_value():
                return secret.get_secret_value()
    except Exception as exc:  # noqa: BLE001 - an unreadable store is not fatal
        logger.debug("could not read %s key for the catalogue: %s", provider, exc)
    return ""


def _catalogue_source(provider: str) -> tuple[Any, type[Any]] | None:
    """``(client, response_model)`` for ``provider``, or None if unavailable.

    The response model comes back alongside the client because the cache stores
    the raw payload: something has to re-validate the dict into the shape
    ``_info_from_listing`` reads, and only the caller knows which shape.

    Returns None rather than raising when a client cannot be built, because
    enriching the catalogue is always optional and a metadata optimisation must
    never become "the CLI will not start".

    Imports are local: the client modules pull in provider response models that
    must stay out of the startup graph.
    """
    from pydantic import SecretStr

    try:
        key = SecretStr(_catalogue_api_key(provider) or _PUBLIC_LISTING_TOKEN)
        if provider == "openrouter":
            from local_operator.clients.openrouter import (
                OpenRouterClient,
                OpenRouterListModelsResponse,
            )

            return OpenRouterClient(api_key=key), OpenRouterListModelsResponse
        if provider == "radient":
            from local_operator.clients.radient import (
                RadientClient,
                RadientListModelsResponse,
            )
            from local_operator.env import get_env_config

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
    except _UnmappableEntry as exc:
        # The provider's document validated but this entry cannot be read. WARN,
        # because unlike a missing model this is upstream data the operator may
        # want to hear about: the session runs on fallback numbers, which means
        # the wrong context window and the wrong cost.
        #
        # The document is deliberately NOT purged. An earlier revision dropped it
        # here, reasoning that a payload which cannot be mapped should not be
        # served for the whole TTL — but the failure is scoped to the ONE entry
        # whose id matched, out of a listing that routinely carries several
        # hundred. Purging threw away metadata that was correct for all of them,
        # and since the upstream document is unchanged the refetch re-poisons the
        # cache immediately, so the only lasting effect was an extra HTTP call per
        # process start plus a refetch for every other model. Falling back for the
        # one bad entry costs nothing and keeps the rest of the catalogue.
        logger.warning("%s catalogue: %s; using static model metadata", provider, exc)
        return fallback
    except ValueError:
        # Not in the catalogue: a brand-new id, or a typo the provider will
        # reject anyway. The static fallback is the honest answer, and this one is
        # routine enough to stay at debug.
        logger.debug("%s catalogue has no entry for %s", provider, model_name)
        return fallback


@functools.lru_cache(maxsize=64)
def _resolve_model_info_cached(provider: str, model_id: str, _bucket: int) -> ModelInfo:
    """Memoized body of :func:`resolve_model_info`.

    ``_bucket`` is unused by the logic and present only to expire the memo: it
    is part of the cache KEY, so when the caller's bucket advances every entry
    for the previous one becomes unreachable and `lru_cache` evicts it in due
    course. Without it a bare `lru_cache` outlives the disk TTL entirely, and a
    long-lived process (the HTTP server, a scheduler worker) would pin whatever
    metadata it saw at boot for as long as it ran — the disk cache would refresh
    underneath it and nothing would ever read the new numbers.
    """
    canonical = "test" if provider == "noop" else provider
    try:
        info = get_model_info(canonical, model_id)
    except (ValueError, KeyError):
        info = ModelInfo(id=model_id, name=model_id, description="Unknown model")
    if canonical in LISTING_PROVIDERS and not _has_real_window(info):
        info = _info_from_catalogue(canonical, model_id, info)
    return info


def resolve_model_info(provider: str, model_id: str) -> ModelInfo:
    """A model's real metadata: static registry first, catalogue to fill gaps.

    THE one resolution path, so the numbers a session runs on and the numbers a
    UI prices with cannot disagree. ``_cost_for`` in the TUI used to call
    ``get_model_info`` directly and therefore saw zero prices for every
    aggregator model — the session had already resolved the real ones, and the
    status band still rendered "cost unavailable".

    Memoized in-process because callers are per-turn: the disk cache alone still
    costs a JSON parse (~25ms) per call, which is real latency inside a render.

    The memo's staleness is BOUNDED by one TTL window rather than pinned to the
    disk file's own age, and the distinction is worth being precise about. The
    file expires on AGE — 24h since its own ``fetched_at`` — while the memo
    expires on a wall-clock window aligned to epoch multiples of the TTL, i.e.
    at 00:00 UTC for the default. So when ANOTHER process refreshes the file
    mid-window, this process keeps serving the pre-refresh numbers until its
    window rolls. That is the same order of staleness as the disk cache, which is
    all this needs to be; what it replaces is a bare ``lru_cache`` whose
    staleness was unbounded, pinning boot-time metadata in a server for as long
    as it ran. Bounded at 64 entries because model ids are user-supplied — a
    typo per turn must not grow the map without limit.

    A switch to a DIFFERENT model needs no invalidation: the id is part of the
    key, so it simply misses.
    """
    return _resolve_model_info_cached(provider, model_id, int(time.time() // DEFAULT_TTL_S))


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

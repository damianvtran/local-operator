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
import inspect
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, SecretStr

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEvent,
)
from local_operator.model.catalogue import DEFAULT_TTL_S
from local_operator.model.effort import default_effort, supported_efforts
from local_operator.model.registry import (
    ModelInfo,
    anthropic_default_model_info,
    anthropic_family_model_info,
    get_model_info,
    unknown_model_info,
)
from local_operator.paths import config_dir

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
    "zai": "glm-5.3",
}

# Sensible ModelSpec fallbacks when the legacy registry knows nothing.
UNKNOWN_CONTEXT_WINDOW = 128_000
UNKNOWN_MAX_OUTPUT = 8_192

#: Per-provider FAMILY resolvers for an id the shipped registry does not carry,
#: tried before the flat templates below. A family answer is strictly better where
#: one exists: Anthropic's tiers no longer share a window (Opus 5 serves 1M, Opus
#: 4.5 serves 200k), so a single per-provider template necessarily reports one of
#: them wrongly, and it was the 200k one — a dated snapshot of Opus 5 ran with a
#: 160k compaction threshold on a model with 1M of room.
_FAMILY_MODEL_RESOLVERS: dict[str, Callable[[str], ModelInfo | None]] = {
    "anthropic": anthropic_family_model_info,
}

#: Per-provider fallback templates for an id neither the registry nor a family
#: resolver can describe. Only providers whose whole FAMILY shares a floor belong
#: here: an Anthropic id whose tier cannot be parsed still resolves to the global
#: 128k/8192/no-cache unknown without one — numbers no Claude has ever had. A
#: provider absent from this map keeps the existing behaviour and falls through to
#: ``unknown_model_info``.
_UNKNOWN_MODEL_TEMPLATES: dict[str, ModelInfo] = {
    "anthropic": anthropic_default_model_info,
}


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


#: Model families that reject ``temperature``/``top_p`` outright.
#:
#: Anthropic's Claude 5 generation answers HTTP 400 ``` `temperature` is
#: deprecated for this model.``` — and then the same for ``top_p`` once
#: ``temperature`` is dropped, so both have to go together. Verified live
#: against ``api.anthropic.com/v1/messages``: ``claude-opus-5`` and
#: ``claude-sonnet-5`` 400 on either parameter and 200 with neither, while
#: ``claude-opus-4-5``/``claude-sonnet-4-5``/``claude-haiku-4-5`` accept both —
#: hence the generation digit must sit directly after the tier, or the trailing
#: ``-5`` of the 4.5 models would match and silently lose their sampling
#: settings. ``[5-9]|\d{2,}`` reads forward rather than pinning to 5: a future
#: ``claude-opus-6`` is far likelier to keep the deprecation than to revert it,
#: and the two failure directions are not symmetric — a false negative makes
#: the model unusable on every single turn, a false positive only falls back to
#: the provider's own sampling defaults.
#:
#: OpenAI's o-series and ``gpt-5`` reject the same pair on both
#: ``/chat/completions`` and ``/responses``. This is deliberately NOT keyed on
#: the ``reasoning`` flag below even though it overlaps: ``reasoning`` also
#: matches the ``thinking``/``reasoner`` suffixes, and Gemini and DeepSeek
#: happily accept ``temperature`` on those variants. Dropping it there would
#: trade a loud 400 for a silent loss of a real setting, which is the worse
#: bug. Only families with observed rejection belong in this pattern.
# The Claude arm matches ANY tier name at generation 5 and above, not a fixed
# list of them. `opus|sonnet|haiku` was written when those were all there were,
# and `claude-fable-5` — a real tier no such list contained — sailed through it
# and sent `temperature`/`top_p` to an endpoint that rejects the pair. This is
# the same reasoning `_anthropic_family` uses for avoiding tier lists, applied
# to the one place that still had one.
_NO_SAMPLING_PARAMS = re.compile(
    r"claude-[a-z]+-(?:[5-9]|\d{2,})(?!\d)" r"|(?:^|[/:-])o[1-9](?:-|$)" r"|gpt-5"
)

#: OpenAI introduced the public Responses route for the GPT-5 generation. The
#: direct `/v1/models` listing exposes ids but no capability flags, so an
#: uncurated current snapshot still needs a family rule; older registry rows
#: remain explicitly off through ``ModelInfo.supports_responses_api``'s default.
_OPENAI_RESPONSES_API = re.compile(r"^gpt-5(?:[.-]|$)")


def build_model_spec(hosting: str, model_name: str, info: ModelInfo | None = None) -> ModelSpec:
    """Derive a harness ``ModelSpec`` from the model's resolved metadata.

    Resolution goes through :func:`resolve_model_info`, NOT ``get_model_info``.
    They differ by exactly the enrichment: the bare registry lookup returns the
    ``-1`` placeholder for any model it does not ship, which this function then
    normalises to the 128k unknown default. That is how a 1M-context model ended up
    running as a 128k one — the enrichment had already learned the real window and
    the spec was built from a path that never saw it.

    It matters because the spec IS what the session runs on: compaction thresholds
    are derived from ``context_window``, so an under-reported window compacts a
    conversation that had eight times the room, and an absent one disables
    compaction until the provider rejects the request.
    """
    from local_operator.providers.registry import get_provider_definition

    canonical = "test" if hosting == "noop" else hosting
    if info is None:
        try:
            info = resolve_model_info(canonical, model_name)
        except Exception:  # noqa: BLE001 - metadata is never worth a failed start
            info = None

    context_window = UNKNOWN_CONTEXT_WINDOW
    max_output = UNKNOWN_MAX_OUTPUT
    supports_images = True
    supports_cache = False
    supports_responses_api = False
    if info is not None:
        # Legacy sentinels: -1 means "no data", not a real limit.
        if info.context_window and info.context_window > 0:
            context_window = info.context_window
        if info.max_tokens and info.max_tokens > 0:
            max_output = info.max_tokens
        if info.supports_images is not None:
            supports_images = info.supports_images
        supports_cache = info.supports_prompt_cache
        supports_responses_api = bool(getattr(info, "supports_responses_api", False))

    definition = get_provider_definition(canonical)
    if canonical == "openai" and _OPENAI_RESPONSES_API.search(model_name.lower()):
        supports_responses_api = True
        supports_cache = True
    lowered = model_name.lower()
    effort_levels = supported_efforts(model_name)
    # A model with an effort ladder reasons BY DEFINITION, whatever its name
    # looks like: `claude-opus-5` matches none of the markers below — it says
    # neither "thinking" nor "reasoner" — so before the ladder existed the
    # status band reported nothing at all for the deepest-reasoning model the
    # app ships with.
    reasoning = bool(effort_levels) or any(
        marker in lowered for marker in ("o1", "o3", "reasoner", "thinking", "deep-research")
    )
    # Keyed on the model, not on the provider that fronts it. `claude-opus-5`
    # returns 200 through OpenRouter only because the aggregator strips the
    # parameters before forwarding — the model never honoured them on either
    # route, so omitting them everywhere loses nothing that was ever real,
    # while a provider-keyed rule would keep shipping a value that is provably
    # discarded and would start 400ing the day an aggregator stops normalising.
    supports_sampling_params = _NO_SAMPLING_PARAMS.search(lowered) is None
    # Seeded to the model's own documented default, not left unset, so the band
    # states a real level from the first frame. Safe only because the provider
    # says the two are the same request — Anthropic documents `effort: "high"`
    # as exactly equivalent to omitting the parameter — which is why OpenAI,
    # whose default varies per snapshot, is seeded with nothing instead.
    reasoning_effort = default_effort(model_name)
    # A GUARDED read, not `info.name`. `info` is duck-typed here — the legacy
    # public helpers and the tests hand in stand-ins, and `name` is the one
    # attribute name that collides with something that is not a string on almost
    # any object that has one (a `MagicMock`'s `.name` is its own identity, a
    # module's is its import path). Feeding that to a `str` pydantic field raises
    # a ValidationError, which would fail a session start over a display label.
    # Anything that is not already a string is treated as no name at all, which
    # is exactly what the readers fall back from.
    resolved_name = getattr(info, "name", "") if info is not None else ""
    if not isinstance(resolved_name, str):
        resolved_name = ""
    # The shared unknown-model singleton is named "Unknown". That word is a
    # STATUS, not a model: a live listing that fills the window but not the
    # name used to paint the status band ``Unknown`` for every unshipped id
    # (Grok 4.6 was the reported case). Treat it as no name so the band falls
    # back to the selector the operator typed.
    if resolved_name.casefold() == "unknown":
        resolved_name = ""
    resolved_id = getattr(info, "id", None)
    describes_this_model = isinstance(resolved_id, str) and _normalised_id(
        resolved_id
    ) == _normalised_id(model_name)
    if not describes_this_model:
        # The row is not ABOUT this model. Resolution degrades to a placeholder
        # whenever nothing describes the id — `ollama_default_model_info` for a
        # local tag, `anthropic_default_model_info` for an unshipped Claude,
        # `unknown_model_info` for anything else — and every one of those carries
        # a name for the PROVIDER: measured, `resolve_model_info("ollama",
        # "qwen3:32b").name` was "Ollama", so the band would have labelled every
        # local model identically and a user running two of them could not tell
        # which was answering.
        #
        # Compared under `_normalised_id` and NOT by equality, because equality
        # would throw away legitimate names. `_info_from_discovery` matches rows
        # on the normalised id and then writes the LISTING's spelling into
        # `info.id`, so a user who types the id Google's own docs show
        # (`models/gemini-2.5-pro`) resolves a row whose `id` is the bare
        # `gemini-2.5-pro` — the same model, a different string. This is the one
        # comparison in the file that has to agree with that matcher.
        resolved_name = ""

    return ModelSpec(
        provider=canonical,
        model_id=model_name,
        context_window=context_window,
        max_output_tokens=max_output,
        supports_tools=True,
        supports_images=supports_images,
        supports_prompt_cache=supports_cache,
        supports_responses_api=supports_responses_api,
        base_url=definition.base_url if definition else None,
        reasoning=reasoning,
        supports_sampling_params=supports_sampling_params,
        reasoning_efforts=effort_levels,
        reasoning_effort=reasoning_effort,
        display_name=resolved_name,
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
    # Validated against the coding-plan base so a key that works here is a key
    # that works for inference; the general `/api/paas/v4` listing would accept
    # keys that cannot spend coding-plan quota.
    "zai": ValidationDescriptor("https://api.z.ai/api/coding/paas/v4/models"),
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
    # ``requests`` is imported HERE, not at module scope. This module is loaded
    # by ``session_factory._prepare`` on every single session build, but the
    # only thing in it that speaks to ``requests`` is this one interactive
    # credential-validation call. Eagerly, requests costs 53.7 ms / +12.6 MB
    # RSS / +228 modules in a bare interpreter, and even alongside the httpx
    # stack the session already loads it still costs +5.8 ms / +2.9 MB / +127
    # modules — paid by every ``exec`` run that never validates a key.
    # Measured with scripts/bench_base_overhead.py; pinned by
    # tests/unit/test_import_graph.py. Note the whole ``local_operator.clients``
    # package still uses requests, so this defers the cost rather than removing
    # it: any run that reaches a client pays it then.
    import requests

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


def _needs_enrichment(info: ModelInfo) -> bool:
    """True when a live listing could still teach us something about this model.

    The window alone was the gate, and it left nine shipped rows priced at $0
    forever — `google/gemini-2.0-pro-exp-02-05`, `google/gemini-2.0-flash-exp`,
    the `alibaba/qwen2.5-coder-*` pair and five more all carry a real window and
    no price. This module's contract names prices as one of the things enrichment
    fixes, and a row that can never enter the enrichment path can never learn one:
    the status band renders "cost unavailable" for the whole life of the install.

    A second reason to enter costs nothing when the listing turns out to be terse,
    because :func:`_info_from_discovery` takes each field only when the listing
    actually carries it — a priced-lookup that comes back priceless leaves the row
    exactly as it was. What it does cost is one listing call for such a row, which
    is why the gate stays closed for a row that has BOTH: a fully described model
    still does zero HTTP, zero cache reads and zero listing scans.

    This is the INCOMPLETENESS question only. Whether a complete row should be
    re-asked anyway is a different one — see :func:`_listing_can_correct`, which
    is what gates the provider's own listing; this predicate gates the aggregator
    leg, where "we already have an answer" really is the end of it.
    """
    return not _has_real_window(info) or not (info.input_price or info.output_price)


def _listing_can_correct(info: ModelInfo) -> bool:
    """True when the PROVIDER's own listing is worth asking, complete row or not.

    Everything :func:`_needs_enrichment` covers, plus rows whose limits are
    second-hand. ``limits_from_listing`` marks the ones whose window and
    ``max_tokens`` were transcribed out of the provider's listing on a date — the
    ten current-generation Claude rows say so in their header comment. Nothing
    about that transcription is independent knowledge, so the provider can always
    be more right than it is, and skipping the listing pins every session to
    whenever a human last copied the numbers over.

    That clause is not theoretical tidiness: it repairs a regression those rows
    caused the moment they were priced. Until then they entered enrichment through
    the PRICE clause, purely by accident of carrying `0.0` — so pricing them
    silently stopped Anthropic from ever correcting an Opus 5 window again, and a
    live `image_input.supported: false` from ever reaching the compaction
    strategy. The first of those is the exact failure (`1.8%/200k` on a 1M model)
    that the registry header blames for these rows existing at all.

    The cost is bounded and is the cost these rows already had: one listing per
    provider per TTL bucket, disk-cached, memoized per model in
    :func:`_resolve_model_info_cached`. A row that is complete AND first-hand
    still does no I/O at all.
    """
    return _needs_enrichment(info) or info.limits_from_listing


#: Sent as the bearer token when no key can be found. The listing endpoints are
#: PUBLIC catalogue data — verified: `GET https://openrouter.ai/api/v1/models`
#: returns 200 and all 340 models with no Authorization header at all, and 200
#: with a bogus one. The clients nevertheless refuse to construct on an empty
#: key, so a placeholder is what lets a keyless (or OAuth-only) install still
#: learn its real context window. If a provider ever starts gating the
#: catalogue, the request 401s and the whole path degrades to the static entry,
#: which is the same outcome as having no key today.
_PUBLIC_LISTING_TOKEN = "public-catalogue-read"


def _credential_file_names(provider: str) -> list[str]:
    """The ``CredentialManager`` keys worth trying for ``provider``.

    ``env_key_name`` answers this for the plain-string ``env_keys`` form and
    returns ``None`` for the callable one, which today is exactly ``anthropic``.
    Stopping there would leave half of the defect in place: an install whose key
    came from ``local-operator credential update ANTHROPIC_API_KEY`` writes the
    credential FILE, so the listing would still go out unauthenticated. The
    provider table already declares the key name that command writes, so it is the
    right second source — a second hard-coded map here could drift from the one
    the CLI, the server schema and the setup prompt all read.
    """
    from local_operator.providers.registry import env_key_name

    name = env_key_name(provider)
    if name:
        return [name]

    from local_operator.model.registry import SupportedHostingProviders

    for detail in SupportedHostingProviders:
        if detail.id == provider:
            return list(detail.requiredCredentials)
    return []


def _catalogue_api_key(provider: str) -> str:
    """An explicit API key for ``provider`` from env or the credential file, else "".

    Reading ONLY ``os.environ`` was a real defect rather than a shortcut: both
    sanctioned credential flows bypass the environment. ``local-operator
    credential update OPENROUTER_API_KEY`` writes the ``CredentialManager`` file,
    and the TUI's ``/login`` writes the ``AuthStore``. So the users who configured
    credentials the app's own way were exactly the ones this enrichment silently
    skipped — their sessions streamed fine (the stream-time cascade reads those
    stores) while their band showed a 128k window and no cost, forever, with the
    failure recorded only at debug level. Every other key reader in the repo goes
    through ``CredentialManager``; this one was the outlier.

    The env leg goes through ``resolve_env_key`` rather than reading the
    definition's ``env_keys`` directly, because that field has TWO forms —
    ``str | Callable[[], str | None]`` — and an ``isinstance(..., str)`` test
    silently drops the callable one. Anthropic is the only provider using it, so
    the reader that skipped it skipped precisely the provider whose listing needs
    a credential most: its catalogue 401s unauthenticated, so enrichment never ran
    and every unshipped Claude id kept the 128k unknown default.

    The OAuth store is NOT read here — see :func:`_catalogue_credential`, which
    layers it underneath this and reports which kind of secret it found.
    """
    from local_operator.providers.registry import resolve_env_key

    canonical = "test" if provider == "noop" else provider
    from_env = resolve_env_key(canonical)
    if from_env:
        return from_env

    try:
        from local_operator.credentials import CredentialManager

        manager = CredentialManager(config_dir())
        for name in _credential_file_names(canonical):
            secret = manager.get_credential(name)
            if secret is not None and secret.get_secret_value():
                return secret.get_secret_value()
    except Exception as exc:  # noqa: BLE001 - an unreadable store is not fatal
        logger.debug("could not read %s key for the catalogue: %s", provider, exc)
    return ""


def _env_secret_is_oauth(secret: str) -> bool:
    """True when ``secret`` came ONLY out of OAuth-named env variables.

    The callable ``env_keys`` resolvers can hand back either kind of credential —
    ``_anthropic_env_key`` prefers ``ANTHROPIC_OAUTH_TOKEN`` over
    ``ANTHROPIC_API_KEY`` — and return only the VALUE, so the caller cannot tell
    which it got. Getting that wrong is not cosmetic: Anthropic rejects an OAuth
    token sent as ``x-api-key`` with a 401, which is exactly the "model cannot be
    described" outcome this whole path exists to avoid.

    Matching on the variable NAME keeps this a general rule instead of a second
    place that knows about Anthropic specifically, and it runs at most once per
    model id per TTL bucket.

    ``all`` and not ``any``, because the two misclassifications do not cost the
    same. A plain ``OPENAI_API_KEY`` whose value happens to equal that of any
    other variable with OAUTH in its name was reported as an OAuth token, and
    OpenAI's OAuth route needs a ChatGPT account id that an env key cannot
    supply — so the provider became unlistable outright, silently and for as
    long as the variables stayed set. An OAuth token misread as a key costs one
    401 on one provider and falls back to the bundled registry. When a value
    appears under both kinds of name it is genuinely ambiguous, and this resolves
    the ambiguity toward the cheaper mistake.
    """
    names = [name.upper() for name, value in os.environ.items() if value == secret]
    return bool(names) and all("OAUTH" in name for name in names)


def _catalogue_credential(provider: str) -> tuple[str, bool, str | None]:
    """``(secret, is_oauth, account_id)`` for a listing call.

    The OAuth flag selects provider-specific auth, while OpenAI additionally
    requires the stored ChatGPT account id to authorize its current Codex
    catalogue. Explicit keys keep precedence and are not account-scoped.

    Order is env, then the credential file, then the OAuth store — an explicit
    variable is the operator overriding config for one run, which is the same
    precedence every other key reader in the repo uses. That order was inverted in
    practice for Anthropic: the env leg could not see a callable ``env_keys``, so
    a stored OAuth row beat an explicitly exported ``ANTHROPIC_API_KEY``.
    """
    key = _catalogue_api_key(provider)
    if key:
        return key, _env_secret_is_oauth(key), None
    return _oauth_listing_token(provider)


def _oauth_listing_token(provider: str) -> tuple[str, bool, str | None]:
    """The newest stored token and account scope, or ``("", False, None)``.

    Best-effort by construction: an unreadable store, a missing table or a row
    without a token all mean "no listing", never an exception. Opened and closed
    per call because this is reached only when the registry could not describe a
    model, which is once per model id per TTL bucket per process.
    """
    store = None
    try:
        from local_operator.providers.auth_store import AuthStore
        from local_operator.providers.registry import credential_provider_id

        storage = credential_provider_id(provider)
        store = AuthStore()
        rows = store.list_credentials(provider=storage)
        for row in reversed(rows):
            token = str(row.data.get("access") or "")
            if token:
                account_id = row.data.get("account_id") or row.data.get("org_id")
                return (
                    token,
                    row.credential_type == "oauth",
                    (str(account_id) if account_id else None),
                )
    except Exception as exc:  # noqa: BLE001 - metadata is never worth a failed start
        logger.debug("could not read a stored %s token for the listing: %s", provider, exc)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001 - closing a broken handle is not fatal
                pass
    return "", False, None


def _normalised_id(model_id: str) -> str:
    """A model id in the one spelling both sides of a match can agree on.

    Discovery NORMALISES ids on ingest — ``_row_from_gemini_entry`` strips
    Google's ``models/`` resource prefix so the rest of the system sees a bare id
    — while the user types whatever the provider's own documentation shows, which
    for Gemini is ``models/gemini-2.5-pro``. An exact-match-only lookup therefore
    missed the spelling Google itself publishes and handed that session the 128k
    unknown default. Case is folded for the same reason: an id is a wire
    identifier, not prose, and no provider ships two models differing only in case.
    """
    trimmed = model_id.strip()
    prefix = "models/"
    if trimmed.startswith(prefix):
        trimmed = trimmed[len(prefix) :]
    return trimmed.casefold()


def _info_from_discovery(
    provider: str, model_name: str, fallback: ModelInfo, *, timeout: float | None = None
) -> ModelInfo:
    """Fill ``fallback``'s gaps from the provider's own live model listing.

    Never raises, and never returns worse data than it was given: every field is
    taken only when the listing actually has it. ``local_operator.model.discovery``
    has already merged the listing over the static registry and applied the rules
    that make a listing trustworthy — a zero price is unknown rather than free, a
    ``max_tokens`` of exactly 4096 is a lying OpenAI-compat default, an UNSTATED
    capability defers to the registry while a stated one (including a ``false``)
    is the provider's own answer — so this function is only the projection of that
    answer onto the legacy ``ModelInfo`` shape.

    ``timeout`` overrides ``discovery.DEFAULT_TIMEOUT_S``. It exists because the
    two reasons to call this are not equally urgent. When the registry cannot
    describe the model the answer is REQUIRED — the session runs with no context
    window until it arrives, so the full ceiling is the right budget and the user
    is watching a spinner for it. When the row is complete and is only being
    re-asked in case the provider has since corrected it
    (:func:`_listing_can_correct`), the answer is a bonus: a slow host must cost a
    stale-but-correct number, not the frame. That second call is reachable from a
    repaint, where ten seconds is a frozen keyboard rather than a slow start.

    Imported lazily. The discovery module pulls httpx and the provider registry,
    and this branch is only reached for a model the registry does not describe;
    putting that on the import path would cost every CLI invocation.
    """
    try:
        from local_operator.model.discovery import DEFAULT_TIMEOUT_S, available_models

        secret, is_oauth, account_id = _catalogue_credential(provider)
        rows, status = available_models(
            provider,
            api_key=secret or None,
            is_oauth=is_oauth,
            account_id=account_id,
            timeout=DEFAULT_TIMEOUT_S if timeout is None else timeout,
        )
    except Exception as exc:  # noqa: BLE001 — metadata is never worth a failed start
        logger.debug("%s discovery unavailable for %s: %s", provider, model_name, exc)
        return fallback

    row = next((candidate for candidate in rows if candidate.id == model_name), None)
    if row is None:
        # Exact first, normalised second: the exact hit is what every provider but
        # Google produces, and trying it alone costs one comparison per row.
        wanted = _normalised_id(model_name)
        row = next(
            (candidate for candidate in rows if _normalised_id(candidate.id) == wanted), None
        )
    if row is None:
        logger.debug("%s listing (%s) has no entry for %s", provider, status, model_name)
        return fallback

    info = fallback.model_copy(deep=True)
    info.id = row.id
    info.name = row.name or info.name or row.id
    if row.context_window > 0:
        info.context_window = row.context_window
    if row.max_tokens > 0:
        info.max_tokens = row.max_tokens
    if row.input_price > 0:
        info.input_price = row.input_price
    if row.output_price > 0:
        info.output_price = row.output_price
    if row.cache_read_price > 0:
        info.cache_reads_price = row.cache_read_price
        # A quoted cache-READ price is the only signal some providers give that
        # prompt caching exists at all; the write price is often absent because the
        # caching is implicit. Falling back to the input price keeps cost estimates
        # from reading as free rather than inventing a number.
        if not info.cache_writes_price:
            info.cache_writes_price = info.input_price
    if row.supports_images is not None:
        # The provider's own statement, including a ``false``: ``DiscoveredModel``
        # spells "the listing did not say" as ``None``, so the only thing an
        # OR would add here is the ability to ignore a denial. ``ModelInfo``
        # already carries ``Optional[bool]`` with the same meaning, and
        # ``build_model_spec`` reads ``is not None`` before trusting it.
        info.supports_images = row.supports_images
    info.supports_prompt_cache = info.supports_prompt_cache or row.supports_prompt_cache
    return info


#: The catalogue consulted for a direct provider's prices. OpenRouter rather than
#: Radient because it is the one whose listing is a PUBLIC document — no key, no
#: account — so this leg works on an install that has only, say, an Anthropic
#: OAuth login. See :data:`local_operator.model.discovery.PUBLIC_LISTING_PROVIDERS`.
_AGGREGATOR_CATALOGUE = "openrouter"

#: Seconds this leg may block. Well under ``discovery.DEFAULT_TIMEOUT_S`` (10.0)
#: because it runs BEHIND the provider's own listing on the same synchronous
#: call — two default ceilings would be a 20s session start for one unresolvable
#: model — and because it is reachable from the TUI's 1 Hz poll. Enrichment that
#: cannot be had in three seconds is worth skipping until the next TTL bucket;
#: the row degrades to "cost unavailable", which is the honest pre-existing state.
_AGGREGATOR_TIMEOUT_S = 3.0

#: A direct provider's id, mapped to the namespace the same models are published
#: under in the OpenRouter catalogue. Only providers whose OWN listing quotes no
#: prices need an entry, which is every direct provider in this tree: Anthropic's
#: `/v1/models` has no pricing object, OpenAI's `/v1/models` is bare ids, and
#: Gemini's listing carries token limits only. The aggregators (`openrouter`,
#: `radient`) are deliberately absent — their own listing IS the priced one.
#:
#: Spelled out rather than derived because three of the namespaces are renames
#: (`x-ai`, `qwen`, `moonshotai`), verified against
#: `GET https://openrouter.ai/api/v1/models` on 2026-08-10; a derived guess would
#: silently price a model from whatever else happened to match.
_AGGREGATOR_NAMESPACE: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "deepseek": "deepseek",
    "mistral": "mistralai",
    "xai": "x-ai",
    "alibaba": "qwen",
    "kimi": "moonshotai",
    # Z.AI's own listing quotes no prices, and GLM is resold on OpenRouter under
    # the `z-ai/` namespace (verified against GET https://openrouter.ai/api/v1/models
    # on 2026-08-17). Newer ids not yet carried there fall back to the static rows.
    "zai": "z-ai",
}

#: A trailing Anthropic-style release stamp: `claude-opus-4-5-20251101`.
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

#: A version separated by a dash between two digits: the `4-5` of `claude-opus-4-5`.
#: Anchored on digits BOTH sides so `qwen2.5-coder-1.5b` and `gpt-4o-mini` are left
#: alone — only a dash that is standing in for a decimal point is rewritten.
_DOTTED_VERSION_RE = re.compile(r"(?<=\d)-(?=\d)")

#: Seconds a NON-BLOCKING listing refresh may take — the case where the registry
#: already has a usable answer and is only checking whether the provider has since
#: corrected it (see :func:`_listing_can_correct`). Short because this path is
#: reachable from a TUI repaint rather than from a spinner: the full
#: ``discovery.DEFAULT_TIMEOUT_S`` there is a frozen keyboard, and the cost of
#: giving up is a number that is stale rather than a number that is missing.
_REFRESH_TIMEOUT_S = 2.0


def _remaining_budget(started: float) -> float:
    """What is left of one resolution's total listing budget, in seconds.

    Resolution can consult two listings — the provider's own, then the public
    aggregator catalogue. Given a ceiling each, they compose into their SUM, so a
    model neither can describe blocks for both. One deadline across the pair keeps
    the second leg free in the common case (the first answers in tens of
    milliseconds) and bounds the pathological one at the single ceiling every
    caller of this module already budgets for.

    Which leg gets STARVED by that is a deliberate priority, not an accident of
    ordering, and inverting it would be a real regression. On a degraded network
    this tends to yield a window but no price, because leg 1 supplies the limits
    and leg 2 is the only source of money for a direct provider. That is the right
    way round: the window is load-bearing — the compaction threshold derives from
    it, and getting it wrong 400s the turn — while a missing price costs a status
    segment that already knows how to say "unavailable".

    Floored just above zero rather than at it: ``httpx`` reads ``timeout=0`` as
    "fail immediately", which is the correct OUTCOME here but arrives as a
    connect error in the log rather than as the deliberate skip it is. A few
    milliseconds says the same thing without the noise.
    """
    from local_operator.model.discovery import DEFAULT_TIMEOUT_S

    return max(0.01, DEFAULT_TIMEOUT_S - (time.monotonic() - started))


def _aggregator_spellings(model_id: str) -> list[str]:
    """``model_id`` as the aggregator might spell it, most literal first.

    Providers and aggregators disagree about punctuation for the SAME model, and
    the disagreement is systematic rather than per-model: Anthropic ships
    `claude-opus-4-5-20251101` while OpenRouter lists `anthropic/claude-opus-4.5`.
    Trying only the literal id would leave every dated Claude snapshot unpriced,
    which is precisely the population this fallback exists for.

    Both rewrites are conservative — a date stamp is eight digits at the end, and a
    dash becomes a dot only between two digits — and every candidate must still be
    found in the catalogue before it is believed. A miss costs one dict lookup.

    Order is most-literal-first, and the DOTTED-WITH-DATE form comes before either
    date-stripped one on purpose: OpenRouter publishes dated snapshots under their
    own dated ids alongside the undated alias (`anthropic/claude-3.5-sonnet-20240620`
    as well as `anthropic/claude-3.5-sonnet`), so stripping the date first would
    answer a question about one snapshot with the alias's price. Harmless while
    snapshots of a family share a rate, wrong the day one does not.
    """
    stripped = _DATE_SUFFIX_RE.sub("", model_id)
    candidates = [model_id]
    for candidate in (
        _DOTTED_VERSION_RE.sub(".", model_id),
        stripped,
        _DOTTED_VERSION_RE.sub(".", stripped),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _from_aggregator_catalogue(
    provider: str, model_id: str, info: ModelInfo, *, timeout: float | None = None
) -> ModelInfo:
    """Describe a DIRECT provider's model from the public aggregator catalogue.

    The last leg of resolution, reached only when neither the registry nor the
    provider's own listing could finish the job. It exists because those two
    sources CANNOT close the gap between them: a direct provider's listing quotes
    no money at all, and for a model the registry has not been taught about it
    frequently carries no limits either. That is not a hypothetical —
    `openai/gpt-5.4` is a shipping model with no registry row, and on a
    fully-credentialled install it resolved to `input_price=0.0` AND no context
    window, so the status band read "cost unavailable" and `311.0k/—` for the
    whole session.

    Structurally this is the same trick the aggregator models already get for free:
    OpenRouter's listing describes the very same upstream models, and it is fetched
    with NO credential — ``available_models`` treats it as keyless because it is in
    :data:`~local_operator.model.discovery.PUBLIC_LISTING_PROVIDERS`, so the request
    simply goes out without an ``Authorization`` header — and cached on disk for a
    TTL. So the self-healing property is the point: a row added tomorrow with a 0.0
    placeholder starts costing correctly instead of silently reading as free until
    someone notices.

    Every field is taken ONLY where the direct sources left a hole. The provider's
    own answer is authoritative where it exists and can legitimately differ from
    what the aggregator's routing exposes — OpenRouter advertises the largest
    window across its routes, which is the wrong number for a specific upstream
    endpoint. ``supports_images`` is not taken at all: it carries a three-valued
    contract (see :func:`_info_from_discovery`) in which a stated ``false`` is the
    PROVIDER's denial, and a second-hand listing has no standing to issue one.
    ``supports_prompt_cache`` is inferred from a quoted cache-read price, which is
    the same inference :func:`_info_from_discovery` makes and is only ever
    widening.

    Never raises and never returns worse data than it was given.
    """
    namespace = _AGGREGATOR_NAMESPACE.get(provider)
    if namespace is None:
        return info
    try:
        from local_operator.model.discovery import available_models

        # Two ceilings, whichever is smaller. `_AGGREGATOR_TIMEOUT_S` is this leg's
        # own cap: it is pure enrichment stacked BEHIND the provider's listing, and
        # it is reachable from the TUI's 1 Hz poll (`_harvest_subagent_costs` →
        # `job_cost` → here, for a child on a model not yet in the memo), where a
        # 10s synchronous stall is input lag rather than a slow start. `timeout` is
        # what the CALLER has left of the whole resolution's budget, so the two
        # legs together can never cost more than the single ceiling every caller of
        # this module already assumes.
        budget = _AGGREGATOR_TIMEOUT_S if timeout is None else min(_AGGREGATOR_TIMEOUT_S, timeout)
        rows, _status = available_models(_AGGREGATOR_CATALOGUE, api_key=None, timeout=budget)
    except Exception as exc:  # noqa: BLE001 — metadata is never worth a failed start
        logger.debug("aggregator catalogue unavailable for %s/%s: %s", provider, model_id, exc)
        return info

    # Priced rows only. An unpriced aggregator row is a routing stub that can
    # answer neither of the two questions this leg is here for, and matching one
    # would shadow a better-spelled sibling further down the candidate list.
    priced = {row.id: row for row in rows if row.input_price > 0 or row.output_price > 0}
    row = None
    for spelling in _aggregator_spellings(model_id):
        row = priced.get(f"{namespace}/{spelling}")
        if row is not None:
            break
    if row is None:
        logger.debug("aggregator catalogue has no priced entry for %s/%s", provider, model_id)
        return info

    info = info.model_copy(deep=True)
    if not (info.input_price or info.output_price):
        info.input_price = row.input_price
        info.output_price = row.output_price
        if row.cache_read_price > 0:
            info.cache_reads_price = row.cache_read_price
            info.supports_prompt_cache = True
            # Same convention as `_info_from_discovery`: `DiscoveredModel` carries
            # no write price, and the input price is the closest defensible
            # stand-in. It under-states an Anthropic 5m write by 20% (1.25x base) —
            # which is why the shipped rows in the registry carry the real number
            # and this is only the floor for an id nobody has written down yet.
            if not info.cache_writes_price:
                info.cache_writes_price = info.input_price
    if not _has_real_window(info) and row.context_window > 0:
        # A missing window is not cosmetic and not merely a rendering gap: the
        # compaction threshold is derived from it, so an unknown window disables
        # compaction for the session and the turn eventually 400s on the
        # provider's real limit. The band's `311.0k/—` is the visible half.
        info.context_window = row.context_window
    # Independent of the window: an OpenAI-shaped gateway can quote a context
    # length and no completion cap, and a missing `max_tokens` falls back to
    # UNKNOWN_MAX_OUTPUT (8192), which truncates a long answer with no error.
    # ``None`` and ``-1`` are both "no data" here, same as the window.
    if not (info.max_tokens and info.max_tokens > 0) and row.max_tokens > 0:
        info.max_tokens = row.max_tokens
    return info


def _registry_fallback(provider: str, model_id: str) -> ModelInfo:
    """What the registry can say about ``model_id``, or the best stand-in for it.

    Three answers in descending order of confidence:

    1. The shipped row for exactly this id.
    2. The model's FAMILY, where the provider has one that can be read out of the
       id — see :func:`anthropic_family_model_info`. This is what keeps a dated
       snapshot of a shipped model (``claude-opus-5-20260112``) on its family's
       real 1M window instead of a family-blind floor.
    3. A per-provider template.

    The global ``unknown_model_info`` is the right answer only for a provider we
    know nothing structural about. For Anthropic it is actively wrong: an unshipped
    Claude id would keep 128k/8192/no-cache — numbers no Claude generation has ever
    had. The template carries the family floor instead, in the same shape
    ``openrouter_default_model_info`` and ``radient_default_model_info`` already use
    for the aggregators.

    The template's id and name are overwritten so a placeholder shared by every
    unknown id of a provider cannot leak its identity ("Anthropic Claude") into a
    band that is meant to name the model the session is running. A family resolver
    owns those two fields itself, because only it knows whether the match was the
    same model under another spelling (keep the real name) or a newer generation
    inheriting limits (the name would be a lie).

    The same overwrite applies to the global ``unknown_model_info`` singleton.
    Returning it as-is is how a live xAI listing that carries a 500k window and
    no display name painted the status band ``Unknown``: discovery copies the
    fallback, fills the window, and keeps the placeholder's name because an
    empty listing name is treated as "no name" rather than "this model is
    called Unknown". The id is this model's; the name must be too.
    """
    try:
        info = get_model_info(provider, model_id)
    except (ValueError, KeyError):
        info = None
    if info is not None and info is not unknown_model_info:
        return info

    resolver = _FAMILY_MODEL_RESOLVERS.get(provider)
    family = resolver(model_id) if resolver is not None else None
    if family is not None:
        return family

    template = _UNKNOWN_MODEL_TEMPLATES.get(provider)
    if template is not None:
        return template.model_copy(deep=True, update={"id": model_id, "name": model_id})
    if info is not None:
        return info.model_copy(deep=True, update={"id": model_id, "name": model_id})
    return ModelInfo(id=model_id, name=model_id, description="Unknown model")


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
    started = time.monotonic()
    info = _registry_fallback(canonical, model_id)
    if _listing_can_correct(info):
        # EVERY provider, not just the aggregators. The gate used to be
        # `canonical in LISTING_PROVIDERS`, which left a hole that the model picker
        # turned into a routine path: the picker offers whatever a provider's live
        # listing returns, so a user can now select `anthropic/claude-opus-5` — a
        # real model, absent from the shipped registry — and the session would run
        # with `context_window = -1`. Compaction thresholds derive from the window,
        # so that is not a cosmetic gap: compaction silently never fires and the
        # turn eventually 400s on the provider's real limit.
        #
        # Reached when the registry is missing the window or BOTH prices, and ALSO
        # when its limits are a dated transcription of this very listing — see
        # `_listing_can_correct`. A row that is complete and first-hand still costs
        # nothing: no HTTP call, no cache read, no listing scan.
        #
        # The two cases get different budgets. Missing data is BLOCKING — the
        # session has no context window until the listing answers — so it keeps the
        # full ceiling. A complete row being re-asked for a correction is not: it
        # already has a usable answer, and this path is reachable from a TUI
        # repaint (`subagent_panel.job_stats` resolves a child's model on the paint
        # timer), where a slow provider would otherwise freeze the keyboard for ten
        # seconds per distinct child model. Measured on this branch: 0.007ms warm
        # memo, 45ms warm disk, 222ms cold disk, 10s worst case. Capping the
        # non-blocking case costs a stale-but-correct number, which is exactly what
        # the registry row already is.
        info = _info_from_discovery(
            canonical,
            model_id,
            info,
            timeout=None if _needs_enrichment(info) else _REFRESH_TIMEOUT_S,
        )
    if _needs_enrichment(info):
        # STILL incomplete after the provider's own listing had its turn, which for
        # every DIRECT provider is the normal outcome rather than a failure: none of
        # them quote money in `/v1/models`, and for an id the registry has not been
        # taught about most of them carry no limits either. Without this leg the only
        # way such a model is ever described is a human editing the registry, and the
        # ten current-generation Claude rows plus every shipping `gpt-5.x` show how
        # that goes — they sat at 0.0 with no window on a fully working install.
        #
        # The SAME gate as above, deliberately re-evaluated rather than folded in:
        # the provider's own answer must get first refusal, and a model either
        # source fully described must not pay for a second catalogue read.
        #
        # Budgeted against ONE deadline for the whole resolution, not its own fresh
        # ceiling. Two independent budgets compose into their SUM, so adding this
        # leg silently took an unresolvable model's worst case from 10s to 13s —
        # and that model is not the exotic case for the subagent panel, it is the
        # motivating one (a child launched on a `model_spec` override the shipped
        # registry has never heard of). Spending what leg 1 left keeps this leg
        # free for the common case, where leg 1 answers in tens of milliseconds,
        # while guaranteeing the pair can never cost more than the one ceiling
        # callers already budget for.
        info = _from_aggregator_catalogue(
            canonical, model_id, info, timeout=_remaining_budget(started)
        )
    return info


def invalidate_model_info_cache() -> None:
    """Drop the in-process metadata memo.

    The memo is keyed on a TTL bucket, so a resolution that degraded for a fixable
    reason — no credential yet, provider briefly down — otherwise stays degraded
    for up to a full bucket (24h by default). A user who logs in or pastes a key
    mid-session has fixed the cause and should not have to restart to see real
    numbers, so the fix path gets a way to say so.
    """
    _resolve_model_info_cached.cache_clear()


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
    key, so it simply misses. :func:`invalidate_model_info_cache` handles the
    other direction, where the SAME key should be re-resolved because the reason
    it degraded (a missing credential) has just been fixed.

    Every caller gets its OWN copy. ``ModelInfo`` is a mutable pydantic model and
    the registry hands out module-level singletons, so handing back the memo entry
    made ``config.info.context_window = ...`` in one session rewrite the shipped
    registry for every later session in the process — the server and the TUI both
    resolve many models in one process. The copy is a few dozen field assignments
    against the ~25ms JSON parse this memo exists to avoid.
    """
    info = _resolve_model_info_cached(provider, model_id, int(time.time() // DEFAULT_TTL_S))
    return info.model_copy(deep=True)


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


def _openai_api_mode(settings: Mapping[str, Any] | None) -> str:
    """Resolve the direct OpenAI wire route, defaulting safely to Responses."""
    providers = settings.get("providers") if isinstance(settings, Mapping) else None
    openai = providers.get("openai") if isinstance(providers, Mapping) else None
    configured = openai.get("api") if isinstance(openai, Mapping) else None
    # Only the explicit opt-out disables Responses. Old config files have no
    # providers block, and malformed values must not accidentally change route.
    return "chat_completions" if configured == "chat_completions" else "responses"


# ---------------------------------------------------------------------------
# stream_fn factory
# ---------------------------------------------------------------------------


class SessionStreamFn:
    """One session's shared client pool and stateful failover router.

    Hard fallback stays pinned for the rest of a user message, so tool loops,
    compaction and naming do not re-send a warm prompt to a provider that just
    rejected it. Optional usage preflight runs once at the message boundary,
    rotates through same-provider OAuth accounts first, then selects the first
    configured provider/model/effort route with working auth.
    """

    USAGE_CHECK_TTL_S = 60.0
    DEFAULT_USAGE_BLOCK_MS = 5 * 60 * 1000

    def __init__(
        self,
        auth_store: AuthStore,
        settings: Mapping[str, Any] | None,
        session_id: str | None,
    ) -> None:
        import httpx

        from local_operator.providers.failover import FailoverRouteState

        self._auth_store = auth_store
        self._settings = settings
        self._session_id = session_id
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        self._notice_handler: Callable[[str, str], Awaitable[None] | None] | None = None
        self._route_state = FailoverRouteState(on_change=self._on_route_change)
        self._message_boundary_pending = True
        # Frozen for one user-message tool loop: choosing a new effort between
        # tool calls would bust the provider cache and make one task reason at
        # several depths.
        self._message_effort: str | None = None
        # The coarse tier (lo/med/hi) the level above was mapped from, kept so a
        # mid-message model switch can re-fit the SAME judgement onto the new
        # model's ladder instead of re-reading the conversation — see
        # ``_effort_for``.
        self._message_tier: str | None = None
        # A mid-message model switch owes the NEW model a quota check, and this
        # is deliberately not ``_message_boundary_pending``: that flag also gates
        # effort classification, and re-arming it to buy a quota check re-grades
        # the turn from an aside (see ``on_model_changed``).
        #
        # It holds the SELECTOR rather than a bare bool so only the model the
        # switch was made to can spend it — see ``preflight_usage`` (review F10).
        self._quota_recheck_for: str | None = None
        self._primary_selector: str | None = None
        self._usage_checked_selector: str | None = None
        self._usage_checked_at = 0.0

    def _client_for(self, spec: ModelSpec) -> WireClient:
        from local_operator.providers.clients import client_for_spec

        return client_for_spec(
            spec,
            http_client=self._http,
            openai_api=_openai_api_mode(self._settings),
        )

    def set_notice_handler(
        self, handler: Callable[[str, str], Awaitable[None] | None] | None
    ) -> None:
        """Install the owning session's event bridge."""
        self._notice_handler = handler

    def begin_message(self) -> None:
        """Mark the next model call as a user-message boundary."""
        self._message_boundary_pending = True
        # A switch nobody spent a check on does not carry into the next message:
        # this boundary re-checks whatever model it opens on anyway.
        self._quota_recheck_for = None
        self._message_effort = None
        self._message_tier = None

    def on_model_changed(self, model: ModelSpec) -> None:
        """The session switched model mid-message; re-open the new model's quota check.

        Only called when the provider/model pair genuinely changed — ``/effort``
        and per-request sampling overrides write the spec constantly and must
        not each pay for this (see ``Session.set_model``).

        Must stay SYNCHRONOUS: ``Session.set_model`` is a sync method and
        discards a returned awaitable, which is the right contract for what is
        only a cache invalidation.

        The EFFORT is deliberately not touched here. It is re-fitted at apply
        time against the request's own spec (:meth:`_effort_for`), because the
        model handed to this hook is not guaranteed to be the model the next
        request actually carries — the loop's resolver can fall back — and
        fitting to one while applying to the other is how the two drift
        (review F9).
        """
        # ONLY the quota gate, and it is its OWN token on purpose.
        # ``_message_boundary_pending`` also gates effort CLASSIFICATION, so
        # re-arming that to get a quota check would re-grade the turn from
        # whatever aside happens to be the newest user-role message — the exact
        # defect review F2 found.
        #
        # Without a token of its own the new provider went unchecked for the
        # rest of the turn (review F7): ``preflight_usage``'s body sits behind
        # the boundary token, which the turn's FIRST call already spent, so
        # clearing the memo behind that gate achieved nothing.
        #
        # The token names the SELECTOR it was armed for, because a bare bool is
        # spent by whichever request reaches the preflight first — and that is
        # not necessarily a request on the new model. A call built just before
        # the switch can still be in flight (the loop resolves the spec two
        # yields before it calls the stream, and its resolver may fall back to
        # the run's snapshot), so a bool let a stale call consume the check the
        # new provider was owed, reproducing F7 on a narrower path (review F10).
        self._quota_recheck_for = f"{model.provider}/{model.model_id}"

    def _effort_for(self, model: ModelSpec) -> str | None:
        """The frozen auto-effort as a rung ``model`` actually accepts.

        The level chosen at the message boundary is a rung on the ladder of the
        model in force AT THAT MOMENT, and ladders differ between models. A level
        the current model accepts is passed through unchanged; one it does not is
        replaced by re-fitting the same coarse judgement (the classifier's
        lo/med/hi tier) to this model's ladder. What is never done is sending the
        stored level blind, which is an HTTP 400 that reads as the switch having
        broken the session.

        Re-fitting rather than re-classifying is the point (review F2). The
        classifier reads the newest ``role="user"`` message, and mid-turn that is
        very often not the user's prompt: steering, wake, hub, job-result and
        todo-reminder asides all render as user turns. Re-classifying on a switch
        therefore graded the aside — a task opened at ``high`` continued at
        ``low`` after a "hurry up" nudge. The user's own prompt stays the thing
        that decided the depth, which is what freezing was for.

        Called per request rather than on the switch itself so the fit is always
        against the spec being sent (review F9).

        A level the new model ALREADY accepts is kept as-is rather than re-fitted
        (review F11). So a ``med`` prompt frozen as ``low`` on a two-rung ladder
        stays ``low`` on a three-rung one, where re-fitting would say ``medium``.
        That is deliberate: this function exists to keep requests legal, and
        silently deepening a level the user has been shown — and is being billed
        for — because the ladder got finer is a bigger surprise than a level that
        holds steady across a switch.
        """
        if self._message_effort is None:
            return None
        if self._message_effort in model.reasoning_efforts:
            return self._message_effort
        if self._message_tier is None:
            # A level with no tier behind it cannot be re-fitted, and it is not
            # on this model's ladder: dropping it costs one call's depth, where
            # sending it costs the call.
            return None
        from local_operator.model.effort_classifier import map_tier_to_effort

        cfg = self._settings.get("effort", {}) if isinstance(self._settings, Mapping) else {}
        allow_max = (
            bool(cfg.get("allowMax", cfg.get("allow_max", False)))
            if isinstance(cfg, Mapping)
            else False
        )
        return map_tier_to_effort(self._message_tier, model.reasoning_efforts, allow_max=allow_max)

    async def _notice(self, text: str, kind: str = "warning") -> None:
        if self._notice_handler is None:
            return
        result = self._notice_handler(text, kind)
        if inspect.isawaitable(result):
            await result

    async def _on_route_change(self, target: Any, reason: str) -> None:
        effort = f" ({target.effort} effort)" if target.effort else ""
        await self._notice(f"{reason} — falling back to {target.selector}{effort}")

    def _fallback_targets(self, model: ModelSpec) -> list[Any]:
        from local_operator.providers.failover import (
            RetrySettings,
            expand_fallback_targets,
            resolve_chain,
        )

        retry = RetrySettings.from_settings(self._settings)
        if not retry.enabled or not retry.model_fallback:
            return []
        selector = f"{model.provider}/{model.model_id}"
        chain = resolve_chain(selector, retry.fallback_chains)
        return expand_fallback_targets(selector, chain or [])

    async def _target_has_auth(self, target: Any) -> bool:
        from local_operator.providers.failover import parse_selector
        from local_operator.providers.registry import get_provider_definition

        provider, _model_id = parse_selector(target.selector)
        definition = get_provider_definition(provider)
        if definition is not None and definition.allows_missing_api_key:
            return True
        try:
            return bool(await self._auth_store.get_api_key(provider, self._session_id))
        except Exception:
            return False

    async def _first_available_fallback(
        self,
        model: ModelSpec,
        *,
        different_provider: bool = False,
    ) -> Any | None:
        from local_operator.providers.failover import parse_selector

        for target in self._fallback_targets(model):
            provider, _model_id = parse_selector(target.selector)
            if different_provider and provider == model.provider:
                continue
            if await self._target_has_auth(target):
                return target
        return None

    @staticmethod
    def _storage_provider(provider: str) -> str:
        from local_operator.providers.registry import credential_provider_id

        return credential_provider_id(provider)

    async def _primary_has_auth(self, model: ModelSpec) -> bool:
        from local_operator.providers.failover import FallbackTarget

        return await self._target_has_auth(
            FallbackTarget(f"{model.provider}/{model.model_id}", model.reasoning_effort)
        )

    async def preflight_usage(self, model: ModelSpec) -> None:
        """Check reliable OAuth quota once per user-message boundary.

        Unknown/unreachable usage fails open. A low account is suppressed only
        when a sibling or configured fallback is ready, so preflight can never
        turn usable reserve capacity into a dead end.
        """
        from local_operator.providers.failover import RetrySettings, parse_selector

        selector = f"{model.provider}/{model.model_id}"

        if selector != self._primary_selector:
            self._primary_selector = selector
            self._route_state.clear()
            self._usage_checked_at = 0.0
        # EITHER gate opens the check: the user-message boundary (the ordinary
        # once-per-message case) or a mid-message model switch, which brings a
        # provider this turn has never checked. The switch needs its own token
        # because the boundary one was already spent by the turn's first call —
        # without it the new provider went unchecked for the rest of the turn
        # (review F7).
        #
        # The switch token is honoured only for the selector it was armed for,
        # and is consumed only by that same selector, so a request still
        # carrying the pre-switch spec can neither open the gate nor spend the
        # check the new model is owed (review F10).
        recheck_due = self._quota_recheck_for == selector
        if not self._message_boundary_pending and not recheck_due:
            return
        self._message_boundary_pending = False
        if recheck_due:
            self._quota_recheck_for = None

        now = time.monotonic()
        if (
            selector == self._usage_checked_selector
            and now - self._usage_checked_at < self.USAGE_CHECK_TTL_S
        ):
            return
        self._usage_checked_selector = selector
        self._usage_checked_at = now

        retry = RetrySettings.from_settings(self._settings)
        if self._route_state.active is not None and not self._route_state.primary_retry_due():
            return
        if not retry.usage_aware_fallback:
            if self._route_state.active is not None and await self._primary_has_auth(model):
                self._route_state.clear()
            return

        attempted_ids: set[int] = set()
        while True:
            try:
                access = await self._auth_store.get_oauth_access(model.provider, self._session_id)
            except Exception:
                return
            if access is None:
                storage = self._storage_provider(model.provider)
                rows = self._auth_store.list_credentials(storage)
                if rows and all(self._auth_store.is_blocked(row.id, storage) for row in rows):
                    fallback = await self._first_available_fallback(
                        model,
                        different_provider=True,
                    )
                    if fallback is not None:
                        await self._route_state.activate(
                            fallback,
                            f"{model.provider} credentials temporarily unavailable",
                        )
                return
            if access.kind != "oauth" or access.credential_id in attempted_ids:
                return
            attempted_ids.add(access.credential_id)

            from local_operator.providers.usage import fetch_usage, usage_health

            report = await fetch_usage(
                self._http,
                model.provider,
                access_token=access.access_token,
                account_id=access.account_id,
            )
            if report is None:
                return
            health = usage_health(
                report,
                model.model_id,
                reserve_percent=retry.usage_reserve_percent,
            )
            if health.state == "healthy":
                self._route_state.clear()
                return
            if health.state == "unknown":
                return

            remaining = (
                ""
                if health.remaining_fraction is None
                else f" ({health.remaining_fraction * 100:.0f}% remaining)"
            )
            condition = "quota exhausted" if health.state == "depleted" else "quota low"
            if health.scope != "account":
                fallback = await self._first_available_fallback(model)
                if fallback is None:
                    await self._notice(
                        f"{model.provider} {condition}{remaining} for {model.model_id}; "
                        "no configured model fallback is available"
                    )
                    return
                await self._route_state.activate(
                    fallback,
                    f"{model.provider} {condition}{remaining} for {model.model_id}",
                )
                return

            storage = self._storage_provider(model.provider)
            row = self._auth_store.get_credential(access.credential_id)
            siblings = [
                candidate
                for candidate in self._auth_store.list_credentials(storage)
                if candidate.id != access.credential_id
                and (row is None or candidate.credential_type == row.credential_type)
                and not self._auth_store.is_blocked(candidate.id, storage)
            ]
            fallback = await self._first_available_fallback(
                model,
                # A different effort cannot revive a fully exhausted provider,
                # but it can preserve reserve quota by reducing token spend.
                different_provider=health.state == "depleted",
            )
            if not siblings and fallback is None:
                await self._notice(
                    f"{model.provider} {condition}{remaining}; no configured fallback is available"
                )
                return

            block_ms = max(
                60_000,
                health.reset_after_ms or self.DEFAULT_USAGE_BLOCK_MS,
            )
            if siblings:
                self._auth_store.block_credential(
                    access.credential_id,
                    storage,
                    block_ms=block_ms,
                )
                await self._notice(
                    f"{model.provider} {condition}{remaining} — trying another "
                    f"{model.provider} account before provider fallback"
                )
                continue

            assert fallback is not None
            fallback_provider, _model_id = parse_selector(fallback.selector)
            if fallback_provider != model.provider:
                self._auth_store.block_credential(
                    access.credential_id,
                    storage,
                    block_ms=block_ms,
                )
            await self._route_state.activate(
                fallback,
                f"{model.provider} {condition}{remaining}",
            )
            return

    async def __call__(
        self, request: ChatRequest, signal: AbortSignal | None
    ) -> AsyncIterator[StreamEvent]:
        from local_operator.providers.failover import stream_with_failover

        if request.isolated:
            # Decoration runs alongside the turn, so it must not consume or move
            # any of this session's shared state — see ``ChatRequest.isolated``.
            # Three things are skipped rather than one, and each was a real
            # route by which a title could have degraded a turn:
            #
            # * the message-boundary effort classification, which is CONSUMED by
            #   whoever reaches it first. A naming call arriving before the turn
            #   would spend the boundary, freeze `_message_effort` from its own
            #   prompt, and emit an "auto effort" notice for a request the user
            #   never made.
            # * the quota preflight, which can block a credential and activate a
            #   fallback route for the whole session.
            # * the session's prompt cache key, which identifies a request
            #   PREFIX. The naming call's prefix is a different system block, so
            #   sharing the key buys no hit and dirties the turn's cache entry.
            async for event in stream_with_failover(
                request,
                self._auth_store,
                self._settings,
                self._client_for,
                signal=signal,
                session_id=self._session_id,
            ):
                yield event
            return

        # Classify only at the user-message boundary, then freeze the chosen
        # effort for every tool-loop request under it. The tiny local linear
        # model is sub-millisecond / zero tokens — an extra "small LLM" call
        # would erase the saving on the short prompts most likely to go low.
        if self._message_boundary_pending:
            from local_operator.model.effort_classifier import auto_effort_for

            last_user = next(
                (message.text for message in reversed(request.messages) if message.role == "user"),
                "",
            )
            self._message_effort, classification = auto_effort_for(
                last_user,
                request.model.reasoning_efforts,
                self._settings,
            )
            # Remembered so a mid-message model switch can re-fit this same
            # judgement onto a different ladder (``_effort_for``).
            self._message_tier = classification.tier if classification is not None else None
            if classification is not None and self._message_effort is not None:
                await self._notice(
                    f"auto effort: {self._message_effort} ({classification.tier}, "
                    f"score {classification.score:.1f})",
                    "info",
                )
        # Fitted to THIS request's spec, every call. The frozen level belongs to
        # the model it was classified against, and mid-turn the model can change
        # under it — by the user's switch, or by the loop's resolver falling back
        # to the run's snapshot. Applying the stored level blind would send a rung
        # the current model may not have (review F9).
        effort = self._effort_for(request.model)
        if effort is not None:
            request = request.model_copy(
                update={"model": request.model.model_copy(update={"reasoning_effort": effort})}
            )

        if self._session_id and request.prompt_cache_key is None:
            # The transcript directory name is stable for the session and
            # already scopes credential stickiness. Reusing it here keeps every
            # turn on the same provider cache without coupling the harness loop
            # to session storage.
            request = request.model_copy(update={"prompt_cache_key": self._session_id})

        await self.preflight_usage(request.model)
        async for event in stream_with_failover(
            request,
            self._auth_store,
            self._settings,
            self._client_for,
            signal=signal,
            session_id=self._session_id,
            route_state=self._route_state,
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


def calculate_cost(
    model_info: ModelInfo,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Cost of a request from per-million token pricing.

    The four token counts are DISJOINT buckets: a token counted as a cache read
    must not also be counted as input. Providers disagree about whether their
    own ``input_tokens`` already contains the cached ones, so the caller
    normalizes that before getting here — :func:`cost_for_usage` is the one that
    knows, and is what every caller should use with a live ``Usage``.

    A cache price of ``None`` means the model has no separate cache rate, and the
    tokens fall back to the base input price rather than being priced at zero:
    they were read, so they were billed at something, and free is the one answer
    that is certainly wrong.

    Raises:
        ValueError: on any arithmetic failure (keeps the legacy contract).
    """
    try:
        cache_read_price = model_info.cache_reads_price
        if not cache_read_price:
            cache_read_price = model_info.input_price
        cache_write_price = model_info.cache_writes_price
        if not cache_write_price:
            cache_write_price = model_info.input_price
        total_cost = (
            float(input_tokens) * model_info.input_price
            + float(output_tokens) * model_info.output_price
            + float(cache_read_tokens) * cache_read_price
            + float(cache_write_tokens) * cache_write_price
        ) / 1_000_000.0
        return total_cost
    except Exception as e:
        raise ValueError(f"Error calculating cost: {e}") from e


def _cache_tokens_are_inside_input(provider: str) -> bool:
    """True when the provider's ``input_tokens`` already counts its cached tokens.

    The two conventions are real and the difference is money. Anthropic reports
    ``input_tokens`` EXCLUDING ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``, so the three add up to the context that was
    read; every OpenAI-shaped listing and Gemini report a total prompt count with
    the cached part called out as a SUBSET of it (``prompt_tokens_details.
    cached_tokens``, ``cachedContentTokenCount``). ``clients.py`` normalizes this
    for ``context_tokens`` and deliberately leaves the raw counts alone, which is
    correct — but it means anyone pricing a ``Usage`` has to do the same division.

    Getting it wrong is not a rounding error. Charging an OpenAI turn for
    ``input + cache_read`` double-counts the cached prefix at 11x its real rate;
    dropping Anthropic's cache buckets undercounts a warm agent turn by most of
    its input, since prompt caching is on and the prefix is the bulk of the prompt.

    Keyed on the WIRE FORMAT, not the provider id, so a new Anthropic-wire
    provider (or a new OpenAI-compatible one) gets the right answer without an
    edit here. An unknown provider is treated as OpenAI-shaped because that is
    what every OpenAI-compatible endpoint in the registry is.
    """
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider)
    return not (definition is not None and definition.wire == "anthropic")


def cost_for_usage(provider: str, model_info: ModelInfo, usage: Any) -> float:
    """What one turn's ``Usage`` cost on ``model_info``, in dollars.

    THE money computation. Everything that renders a cost — the parent's status
    band, a subagent row, a subagent page — goes through here, so two surfaces
    can never disagree about what a turn cost.

    ``usage`` is duck-typed rather than annotated ``Usage`` because it also
    arrives rehydrated from a serialized child event, where it is a plain mapping
    with the same field names.

    The caller is responsible for deciding whether ``model_info`` is priced at
    all; this returns 0.0 for a zero-priced model, which is arithmetically true
    and is exactly why a UI must not render it blindly.
    """
    read = _usage_field(usage, "cache_read_tokens")
    written = _usage_field(usage, "cache_write_tokens")
    plain = _usage_field(usage, "input_tokens")
    if _cache_tokens_are_inside_input(provider):
        # Subtract, floored at zero: the buckets must stay disjoint, and a
        # provider that reports more cached tokens than prompt tokens is
        # malformed rather than a reason to hand back a negative bill.
        plain = max(0, plain - read - written)
    return calculate_cost(
        model_info,
        plain,
        _usage_field(usage, "output_tokens"),
        read,
        written,
    )


def _usage_field(usage: Any, name: str) -> int:
    """One token count off a ``Usage`` or an equivalent mapping, as a count ≥ 0.

    Floored, not merely coerced. ``Usage`` declares plain ``int`` fields with no
    validator and the wire clients coerce with a bare ``int(raw.get(...))``, so a
    provider that spells "unknown" as ``-1`` — a convention ``discovery.py``
    documents meeting in the wild — reaches the arithmetic intact. Both signs of
    that are wrong in a way a user would see: a negative output count bills a
    CREDIT, and a negative cache-read count inflates an OpenAI-shaped bill because
    it is subtracted out of the prompt total. It would also break the one
    invariant the parent's running total depends on — a child whose latest figure
    came back smaller than its last would make the band's number go DOWN.
    """
    value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0

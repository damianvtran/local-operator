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
import math
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
    Usage,
)
from local_operator.model.catalogue import DEFAULT_TTL_S
from local_operator.model.defaults import DEFAULT_MODEL_NAMES as _DEFAULT_MODEL_NAMES
from local_operator.model.effort import default_effort, supported_efforts
from local_operator.model.ids import normalised_id as _normalised_id
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
    from local_operator.model.discovery import DiscoveredModel
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.clients import WireClient
    from local_operator.providers.usage import UsageReport
    from local_operator.providers.usage_cache import UsageCacheStore

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
# Re-exported from ``model.defaults`` (the stdlib-only home) so the preflight
# path can read the map without importing this heavy module. Kept as a name here
# because legacy callers and tests import ``configure.DEFAULT_MODEL_NAMES``.
DEFAULT_MODEL_NAMES = _DEFAULT_MODEL_NAMES

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

#: Kimi's coding-plan host pins sampling instead of rejecting the keys
#: outright: ``api.kimi.com/coding/v1/chat/completions`` answers HTTP 400
#: ``invalid temperature: only 1 is allowed for this model`` for any other
#: value, and ``invalid top_p: only 0.95 is allowed`` likewise — while
#: OMITTING both keys succeeds. Verified live against ``k3`` and
#: ``k2-thinking`` with temperatures 0.2/0.6/1/absent and top_p 0.9/1/absent.
#:
#: Scoped to the coding-plan model ids rather than folded into
#: :data:`_NO_SAMPLING_PARAMS`, because this is NOT a property of a model
#: family across routes: the mainland ``api.moonshot.cn`` host serves
#: ``kimi-k2-*``/``moonshot-*`` under the same ``kimi`` provider id and
#: accepts the pair. The rule is "any k-numbered coding-host id, now or
#: later": ``k3``, ``k3-256k``, ``k2-thinking`` (the live probe confirmed it
#: pins the pair too) and ``kimi-for-coding*`` all exist only on the coding
#: host, so matching the ``k<digit>``/``kimi-for-coding`` shapes is matching
#: the endpoint that pins the values. Anchored at the start:
#: ``kimi-k2-0711-preview`` must not match on its ``k2`` fragment.
_KIMI_PINNED_SAMPLING = re.compile(r"^(?:k\d+(?:-|$)|kimi-for-coding)")

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
    # The kimi coding-plan host pins temperature/top_p to fixed values and
    # 400s on any other; omitted keys pass. Unlike _NO_SAMPLING_PARAMS this
    # IS keyed on the provider too, because the same model family accepts the
    # pair on the mainland host — see _KIMI_PINNED_SAMPLING.
    if canonical == "kimi" and _KIMI_PINNED_SAMPLING.match(lowered):
        supports_sampling_params = False
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

    ``model_name`` is passed through as discovery's ``want_id``: a stored
    document old enough to predate the model is refetched once, inside this
    same ``timeout``, before the lookup below is allowed to miss. That is what
    prices a model released this morning on its FIRST resolution rather than
    after the memo bucket rolls over at midnight.

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
            want_id=model_name,
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
        if row.cache_write_price > 0:
            info.cache_writes_price = row.cache_write_price
        elif not info.cache_writes_price:
            # A quoted cache-READ price is the only signal some providers give
            # that prompt caching exists at all; a listing that quotes no write
            # price usually caches implicitly. Falling back to the input price
            # keeps cost estimates from reading as free rather than inventing a
            # number — it under-states an Anthropic 5m write by 20%, which is
            # why a quoted write price above takes precedence.
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


#: Seconds the price-catalogue leg and the aggregator leg may each block. Well
#: under ``discovery.DEFAULT_TIMEOUT_S`` (10.0) because they run BEHIND the
#: provider's own listing on the same synchronous call — two default ceilings
#: would be a 20s session start for one unresolvable model — and because they
#: are reachable from the TUI's 1 Hz poll. Enrichment that cannot be had in
#: three seconds is worth skipping until the next TTL bucket; the row degrades
#: to "cost unavailable", which is the honest pre-existing state.
_AGGREGATOR_TIMEOUT_S = 3.0
_PRICE_CATALOGUE_TIMEOUT_S = _AGGREGATOR_TIMEOUT_S

#: Seconds a NON-BLOCKING listing refresh may take — the case where the registry
#: already has a usable answer and is only checking whether the provider has since
#: corrected it (see :func:`_listing_can_correct`). Short because this path is
#: reachable from a TUI repaint rather than from a spinner: the full
#: ``discovery.DEFAULT_TIMEOUT_S`` there is a frozen keyboard, and the cost of
#: giving up is a number that is stale rather than a number that is missing.
_REFRESH_TIMEOUT_S = 2.0


def _remaining_budget(started: float) -> float:
    """What is left of one resolution's total listing budget, in seconds.

    Resolution can consult up to three listings — the provider's own, the neutral
    price catalogue, and (for an aggregator's own ids) the aggregator's listing.
    Given a ceiling each, they compose into their SUM, so a model none can
    describe blocks for all of them. One deadline across the legs keeps the later
    ones free in the common case (the first answers in tens of milliseconds) and
    bounds the pathological one at the single ceiling every caller of this module
    already budgets for.

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


def _fill_from_row(info: ModelInfo, row: "DiscoveredModel") -> ModelInfo:
    """``info`` with its HOLES filled from a second-hand catalogue row.

    Shared by the price-catalogue and aggregator legs, which have the same
    contract: every field is taken ONLY where the direct sources left one. The
    provider's own answer is authoritative where it exists and can legitimately
    differ from what a catalogue exposes — OpenRouter advertises the largest
    window across its routes, which is the wrong number for a specific upstream
    endpoint. ``supports_images`` is not taken at all: it carries a three-valued
    contract (see :func:`_info_from_discovery`) in which a stated ``false`` is
    the PROVIDER's denial, and a second-hand listing has no standing to issue
    one. ``supports_prompt_cache`` is inferred from a quoted cache-read price,
    the same inference :func:`_info_from_discovery` makes and only ever widening.
    """
    info = info.model_copy(deep=True)
    if not (info.input_price or info.output_price):
        info.input_price = row.input_price
        info.output_price = row.output_price
        if row.cache_read_price > 0:
            info.cache_reads_price = row.cache_read_price
            info.supports_prompt_cache = True
            if row.cache_write_price > 0:
                info.cache_writes_price = row.cache_write_price
            elif not info.cache_writes_price:
                # A catalogue that quotes a read price and no write price. The
                # input price is the closest defensible stand-in — it under-states
                # an Anthropic 5m write by 20% (1.25x base), which is why a quoted
                # write price above wins and this is only the floor.
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


def _from_price_catalogue(
    provider: str, model_id: str, info: ModelInfo, *, timeout: float | None = None
) -> ModelInfo:
    """Fill a model's price and limit holes from the neutral models.dev catalogue.

    The second leg of resolution, reached when the registry and the provider's
    own listing together could not finish the job — which for every DIRECT
    provider is the normal outcome rather than a failure: none of them quote money
    in their listing. Until this leg existed the only price source for such an id
    was the OpenRouter listing looked up under a per-provider namespace, which
    tied an Anthropic-only user's cost display to one aggregator's document and
    its id spellings; the day ``claude-fable-5-1`` shipped that document was six
    hours old and predated the row, so the session ran at $0.00. See
    :mod:`local_operator.model.prices` for what the catalogue is and why it is
    trusted for prices and limits but not capabilities.

    Never raises and never returns worse data than it was given. ``timeout`` is
    what the CALLER has left of the whole resolution's budget, capped at this
    leg's own ceiling: it is pure enrichment stacked BEHIND the provider's
    listing and is reachable from the TUI's 1 Hz poll (via
    ``refresh_model_info_background``'s executor thread), where a 10s stall on
    a 4.4 MB cold download is input lag rather than a slow start.
    """
    try:
        from local_operator.model.prices import price_catalogue_row

        budget = (
            _PRICE_CATALOGUE_TIMEOUT_S
            if timeout is None
            else min(_PRICE_CATALOGUE_TIMEOUT_S, timeout)
        )
        row = price_catalogue_row(provider, model_id, timeout=budget)
    except Exception as exc:  # noqa: BLE001 — metadata is never worth a failed start
        logger.debug("price catalogue unavailable for %s/%s: %s", provider, model_id, exc)
        return info
    if row is None:
        logger.debug("price catalogue has no entry for %s/%s", provider, model_id)
        return info
    if not (row.input_price > 0 or row.output_price > 0 or row.context_window > 0):
        # A key with no cost and no limit answers neither question this leg is
        # here for; models.dev carries such stubs for plan catalogues.
        return info
    return _fill_from_row(info, row)


def _from_aggregator_catalogue(
    provider: str, model_id: str, info: ModelInfo, *, timeout: float | None = None
) -> ModelInfo:
    """Describe an AGGREGATOR's own model from its public listing, as a last resort.

    In practice ``openrouter/*`` ids only. The gate is ``AGGREGATOR_PROVIDERS``,
    but the lookup below needs a listing readable with NO credential, and of the
    aggregators only OpenRouter's is public (``PUBLIC_LISTING_PROVIDERS``);
    Radient's needs a key, so a ``radient/*`` id returns ``info`` untouched from
    this leg and relies on leg 1 having read its listing with the credential.
    Leg 1 has normally already priced OpenRouter's ids too; this leg survives
    for the case where leg 1 was unavailable (a credential lookup that raised)
    and the public OpenRouter document can still answer.

    It used to be the price source for DIRECT providers too, through a
    per-provider namespace map (``anthropic`` → ``anthropic/``, ``xai`` →
    ``x-ai/``, ...). That coupling is gone: :func:`_from_price_catalogue` is the
    provider-neutral leg now, and a direct-provider id this function is handed
    is returned untouched.

    Every field is taken ONLY where the direct sources left a hole
    (:func:`_fill_from_row`). Never raises and never returns worse data than it
    was given.
    """
    from local_operator.providers.registry import AGGREGATOR_PROVIDERS

    if provider not in AGGREGATOR_PROVIDERS:
        return info
    try:
        from local_operator.model.discovery import (
            PUBLIC_LISTING_PROVIDERS,
            available_models,
        )

        # Radient's listing needs a key; only a PUBLIC listing can be read here
        # with no credential at all.
        if provider not in PUBLIC_LISTING_PROVIDERS:
            return info
        # Two ceilings, whichever is smaller — see `_from_price_catalogue`.
        budget = _AGGREGATOR_TIMEOUT_S if timeout is None else min(_AGGREGATOR_TIMEOUT_S, timeout)
        rows, _status = available_models(provider, api_key=None, timeout=budget, want_id=model_id)
    except Exception as exc:  # noqa: BLE001 — metadata is never worth a failed start
        logger.debug("aggregator catalogue unavailable for %s/%s: %s", provider, model_id, exc)
        return info

    # Priced rows only. An unpriced aggregator row is a routing stub that can
    # answer neither of the two questions this leg is here for.
    row = next(
        (r for r in rows if r.id == model_id and (r.input_price > 0 or r.output_price > 0)), None
    )
    if row is None:
        logger.debug("aggregator catalogue has no priced entry for %s/%s", provider, model_id)
        return info
    return _fill_from_row(info, row)


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
        # ceiling. Two independent budgets compose into their SUM, so adding a
        # leg silently took an unresolvable model's worst case from 10s to 13s —
        # and that model is not the exotic case for the subagent panel, it is the
        # motivating one (a child launched on a `model_spec` override the shipped
        # registry has never heard of). Spending what leg 1 left keeps this leg
        # free for the common case, where leg 1 answers in tens of milliseconds,
        # while guaranteeing the legs can never cost more than the one ceiling
        # callers already budget for.
        info = _from_price_catalogue(canonical, model_id, info, timeout=_remaining_budget(started))
    if _needs_enrichment(info):
        # Leg 3, for an AGGREGATOR's own ids only (the function refuses direct
        # providers). Normally leg 1 already priced these from the same listing;
        # this is the fallback for when leg 1 could not run. Same shared deadline.
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
    _paint_refreshing.clear()
    _paint_memo.clear()


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
    bucket = int(time.time() // DEFAULT_TTL_S)
    info = _resolve_model_info_cached(provider, model_id, bucket)
    # Feed the paint memo from the authoritative answer, so a renderer that
    # resolves AFTER the session does (the common order) paints the real row,
    # and so the background refresh is the only writer on a cold process.
    # Prior-bucket keys are evicted on write rather than left to accrete: the
    # dict has no TTL of its own, and a long-lived process (the server, a
    # scheduler worker) would otherwise gain one dead entry per model per
    # day for its whole lifetime. Evicting on write keeps it bounded by the
    # models resolved in the CURRENT bucket without a background sweeper.
    _paint_memo[(provider, model_id, bucket)] = info
    if len(_paint_memo) > 64:
        stale = [key for key in _paint_memo if key[2] != bucket]
        for key in stale:
            del _paint_memo[key]
    return info.model_copy(deep=True)


#: Keys with a paint-triggered background refresh already in flight, so a 1 Hz
#: poller that misses the paint cache once per tick does not spawn one thread
#: per tick for the same model. Cleared with the memo because a refresh that
#: has landed (or definitively failed) is the reason to stop gating: the next
#: paint miss after an invalidation SHOULD spawn a fresh fetch.
_paint_refreshing: set[tuple[str, str]] = set()

#: What the full resolver last answered, keyed with the same TTL bucket as
#: ``_resolve_model_info_cached``. The paint path reads this dict and NEVER
#: calls the cached body, because the body IS the discovery path — entering it
#: on a cold key would run the very HTTP legs this split exists to keep off the
#: loop. Populated only by :func:`resolve_model_info` (directly or via the
#: background refresh), so a cold process paints registry rows until the first
#: full resolve lands, which the background refresh schedules on the first miss.
_paint_memo: dict[tuple[str, str, int], ModelInfo] = {}


def resolve_model_info_paint(provider: str, model_id: str) -> tuple[ModelInfo, bool]:
    """The paint-safe metadata for a model, plus whether the memo answered.

    Returns ``(info, memo_hit)``. ``memo_hit`` is False on a cold memo — the
    TTL bucket rolled over mid-session, or the model was never fully
    resolved in this process — and then ``info`` is the STATIC registry row.
    The caller uses the flag to decide whether to schedule an off-loop
    refresh: a priced registry row served after a rollover may be staler
    than the discovery answer the band showed until that moment, and
    silently switching to it is the confident-wrong-number failure the
    module's own docstrings rule out.

    The memo is a SEPARATE dict rather than a call into
    ``_resolve_model_info_cached`` because that cached body is itself the
    discovery path — entering it on a cold key runs the synchronous
    ``httpx.Client`` legs (measured 418 ms warm-disk, 10 s + 3 s budgets for
    an unlisted model) on whatever loop called this, and the callers here
    are the Textual loop (the status band's ``message_end`` pricing and the
    1 Hz subagent harvest). A frozen keyboard is a worse failure than a
    segment that reads "cost unavailable" for one tick, which is the
    honest degradation the band already renders for a genuinely
    unpriceable model. The registry fallback is NOT a guess dressed as
    data either: a row with no prices prices as ``None`` exactly as
    discovery failing would.
    """
    bucket = int(time.time() // DEFAULT_TTL_S)
    info = _paint_memo.get((provider, model_id, bucket))
    if info is None:
        return _registry_fallback(provider, model_id).model_copy(deep=True), False
    return info.model_copy(deep=True), True


def refresh_model_info_background(provider: str, model_id: str) -> None:
    """Resolve one model off-loop so the NEXT paint sees the real price.

    Fired by the paint path on a miss (see :func:`resolve_model_info_paint`'s
    callers): the full resolver runs in a thread and lands in the shared memo,
    so the following tick prices from the warm cache. Gated per model by
    :data:`_paint_refreshing` — the 1 Hz poller re-misses until the fetch
    lands, and without the gate that is one thread per tick per model.

    Fire-and-forget BY CONTRACT: the caller is a renderer and must never
    wait on or observe this. A failure inside is logged and otherwise
    swallowed; the memo keeps the degraded row, the paint path keeps
    returning it, and the next TTL bucket or an explicit invalidation gets
    to try again. That is also why the flag is cleared in a ``finally":
    a refresh that died must not pin the gate shut forever.
    """
    import asyncio

    key = (provider, model_id)
    if key in _paint_refreshing:
        return
    _paint_refreshing.add(key)

    def _refresh() -> None:
        try:
            resolve_model_info(provider, model_id)
        except Exception:  # noqa: BLE001 — a refresh is never worth surfacing
            logger.debug("paint-triggered model refresh failed", exc_info=True)
        finally:
            _paint_refreshing.discard(key)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop. Refuse rather than run inline: the full resolver
        # is the discovery path (measured up to 13 s for an unlisted model),
        # and a synchronous caller adopting this "paint-safe" API has made
        # exactly the mistake the API exists to prevent. Better a loud no-op
        # with a log line than a silent multi-second block that the next
        # reader blames on the caller's own code.
        logger.warning(
            "refresh_model_info_background called off-loop for %s/%s; skipping",
            provider,
            model_id,
        )
        _paint_refreshing.discard(key)
        return
    loop.run_in_executor(None, _refresh)


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
    rotates through same-provider OAuth accounts first (including accounts
    under the reserve threshold), then selects the first configured
    provider/model/effort route with working auth and remaining quota.
    """

    USAGE_CHECK_TTL_S = 60.0
    DEFAULT_USAGE_BLOCK_MS = 5 * 60 * 1000

    #: How many blocked accounts the recovery walk may probe at once.
    #:
    #: The walk used to be strictly serial, which on a pool with several
    #: blocked rows is a multi-second network train on the time-to-usable
    #: path (an ``ensure_oauth_fresh`` plus a usage GET per row, one after
    #: another). Probing them concurrently removes the train.
    #:
    #: The bound is what keeps the cure from being worse than the disease,
    #: and it is NOT a tuning knob to be raised casually. Anthropic and
    #: OpenAI rate-limit their usage endpoints **per source IP regardless of
    #: account** (see the module docstring of
    #: :mod:`local_operator.providers.usage_cache`), so an unbounded gather
    #: over N blocked rows is a synchronized burst against one IP — exactly
    #: how this walk used to earn its own 429s, whose backoff then poisoned
    #: the NEXT boot. Three is deliberately small: it collapses the common
    #: 3-5 row pool to one or two waves while keeping the instantaneous
    #: request rate close to what a single interactive ``/usage`` already
    #: costs. Raising it trades a few hundred milliseconds for the 429 storm
    #: this constant exists to prevent.
    USAGE_RECOVERY_PROBE_CONCURRENCY = 3

    def __init__(
        self,
        auth_store: AuthStore,
        settings: Mapping[str, Any] | None,
        session_id: str | None,
        cache_lineage_id: str | None = None,
    ) -> None:
        import httpx

        from local_operator.providers.failover import FailoverRouteState

        self._auth_store = auth_store
        self._settings = settings
        self._session_id = session_id
        # The identity this session's PROVIDER CACHE is keyed under, which is
        # the session id for an ordinary session and the PARENT's id for a fork.
        #
        # Separate from ``_session_id`` on purpose, and the separation is the
        # safety property: ``_session_id`` alone scopes sticky CREDENTIAL
        # selection (``auth_store._set_sticky`` keys on ``(provider,
        # session_id)`` and is passed ``_session_id`` directly, never this
        # value). So a fork sharing a cache key with its parent shares a
        # routing hint and nothing else — in particular it does not share a
        # pinned credential row. Unifying the two would silently make it do so.
        self._cache_lineage_id = cache_lineage_id or session_id
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        self._notice_handler: Callable[[str, str], Awaitable[None] | None] | None = None
        # The session's route bridge: called with the pinned fallback target
        # (or None on recovery) so the host can keep its model display and its
        # persisted session state truthful about which model is actually
        # serving requests. Installed by the owning Session, exactly like the
        # notice handler above — the stream owns routing, the session owns
        # ordered event delivery.
        self._route_handler: Callable[[Any, str], Awaitable[None] | None] | None = None
        self._route_state = FailoverRouteState(
            on_change=self._on_route_change,
            on_settle=self._on_route_settle,
        )
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
        # Quota is re-probed at EVERY user-message boundary, but the user only
        # needs to hear about a CHANGE in quota standing — not the same "quota
        # low"/"quota exhausted" line echoed on every message they send while
        # the condition simply persists. This latch remembers, per
        # ``provider/model_id`` selector, WHICH quota conditions we have already
        # announced so a steady state stays silent and only a genuine transition
        # (none->low, low->exhausted, or a recurrence after recovery) speaks.
        # The whole selector entry is cleared the moment the selector reads
        # "healthy" again, so a later re-entry into low/exhausted counts as a
        # real new transition and re-announces. Account ROTATION churn is not
        # tracked here at all — it is suppressed outright as an internal
        # implementation detail.
        #
        # The value is a SET of announced condition TOKENS, not a single state
        # string. A single selector can be under more than one distinct quota
        # condition at different times — account-scope low/exhausted
        # (``account:<state>``), model/tier-scope low/exhausted
        # (``model:<state>``), and the tier-cap-spent-but-shared-remains notice
        # (``tier-spent:<state>``) all key on the SAME ``provider/model_id``.
        # If they shared one remembered state string they would alias: announcing
        # one would overwrite another's memory, so a still-holding condition could
        # wrongly re-announce or a genuinely new one be wrongly suppressed. A set
        # per selector lets each condition dedup against ITSELF only, while the
        # healthy edge still resets all of them together (recovery is a fresh
        # start for every condition on that selector).
        self._last_quota_state: dict[str, set[str]] = {}
        # Shared cross-process usage cache for preflight probes, built lazily on
        # the first routing check and reused for the session's lifetime. Its
        # sole job here is to collapse a concurrent PEER process's duplicate
        # fetch of the same account (the per-source-IP 429 storm this fixes) —
        # it never suppresses this process's own per-boundary re-probe. None
        # until first use, and stays None whenever the cache cannot open (a
        # permanent live-fetch miss, never an error). See ``_usage_cache_store``.
        self._usage_cache: "UsageCacheStore | None" = None

    def _client_for(self, spec: ModelSpec) -> WireClient:
        from local_operator.providers.clients import client_for_spec

        return client_for_spec(
            spec,
            http_client=self._http,
            openai_api=_openai_api_mode(self._settings),
        )

    @property
    def routing_settings(self) -> Mapping[str, Any]:
        """The settings mapping THIS stream will actually route on.

        Captured once at session build (``session_factory``) and held for the
        session's life, so it is not necessarily what is on disk: nothing
        watches ``config.yml``, and the only ``ConfigManager.reload`` caller is
        the first-run ``/login`` path. A read-only surface that wants to report
        what the SESSION will do — rather than what a later edit intends — has
        to read this rather than re-reading the file, or it shows a green light
        for a cascade the running session will not honour.

        Exposed read-only for exactly that reason; mutating routing mid-session
        is not what this is for.
        """
        return self._settings if isinstance(self._settings, Mapping) else {}

    def set_notice_handler(
        self, handler: Callable[[str, str], Awaitable[None] | None] | None
    ) -> None:
        """Install the owning session's event bridge."""
        self._notice_handler = handler

    def set_route_handler(
        self, handler: Callable[[Any, str], Awaitable[None] | None] | None
    ) -> None:
        """Install the owning session's fallback-route bridge.

        Called with the active ``FallbackTarget`` when a fallback pins, and with
        ``None`` when the route returns to the primary — both edges, because a
        model display that only ever learns "fell back" keeps naming the
        fallback after the primary has recovered.
        """
        self._route_handler = handler

    def restore_fallback(self, selector: str, effort: str | None, primary_selector: str) -> None:
        """Re-pin a fallback route persisted by a previous run of this session.

        A resumed session whose transcript says "requests were being served by
        the fallback" should keep being served by it rather than re-sending the
        first prompt to the provider that was failing when the session closed.
        Pinned WITHOUT a cooldown: the next message boundary's preflight is
        free to probe the primary immediately, so a recovered provider is
        picked back up on the first turn rather than after an arbitrary wait.

        ``primary_selector`` is the SELECTED model this pin belongs to, seeded
        into the preflight's selector memo because that memo starts ``None``:
        without it the first ``preflight_usage`` call reads the primary as "a
        different model than last time" and clears the pin it was just handed,
        making every restore a no-op.

        Set directly rather than through ``activate`` so NEITHER handler fires:
        the transcript replay already shows the original fallback notice
        (re-announcing it on every resume reads as a fresh failure that did not
        happen), and the restoring session sets its own display/persistence
        state from the same entry it restored this pin from — a settle edge
        here would only write that entry back to the transcript it came from.
        """
        from local_operator.providers.failover import FallbackTarget

        self._route_state.active = FallbackTarget(selector, effort)
        # The same minimum grace the live driver gives a fresh pin (60s):
        # without it the TUI's boot-time quota preflight — which runs seconds
        # after this and probes only that the primary HAS AUTH, not that it
        # recovered — would clear the pin before the first request proved
        # anything, turning every restore into an immediate un-restore. After
        # the grace, the ordinary boundary probe reclaims a recovered primary.
        self._route_state.primary_retry_at_ms = int(time.time() * 1000) + 60_000
        self._primary_selector = primary_selector

    def withdraw_fallback(self) -> None:
        """The user explicitly re-selected a model; drop the pinned fallback route.

        The inverse of :meth:`restore_fallback`. A fallback pin rescues the
        SELECTED model by routing around it; when the user deliberately picks a
        model again — including re-picking the very model a fallback displaced
        them from — the pin's premise is withdrawn and the next request must go
        to the primary.

        Needed because the ordinary clear is selector-driven: ``preflight_usage``
        resets the route only when it sees a DIFFERENT selector than the memoized
        primary. A same-model re-selection never changes the selector, so that
        lazy clear never fires and the session stays glued to the fallback until
        the user switches away and back — the reported stuck-fallback symptom.

        Silent on purpose — :meth:`FailoverRouteState.clear`, not
        ``clear_settled``. The owning Session has already moved its own display
        state and emitted the ``ModelChangeEvent`` for this withdrawal; firing a
        settle edge here would re-persist and re-announce what the session just
        recorded. Hosts without a route capability simply never call this.

        Must stay SYNCHRONOUS for the same reason :meth:`on_model_changed` does:
        ``Session.set_model`` is sync and discards a returned awaitable.
        """
        self._route_state.clear()
        # The selector memo still matches the re-selected model, so preflight
        # would otherwise trust the quota reading that pinned the fallback for
        # the rest of the TTL. Reset the clock so the explicit re-selection gets
        # a fresh probe at the next boundary instead of inheriting the stale
        # verdict.
        self._usage_checked_at = 0.0

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

    async def _announce_quota_change(
        self, selector: str, token: str, text: str, kind: str = "warning"
    ) -> None:
        """Emit a quota notice once per distinct CONDITION for ``selector``.

        The preflight runs on every user-message boundary, so a persistent
        low/exhausted verdict recurs on every message the user sends — but the
        user only needs to hear about the CHANGE, not the same line echoed
        forever. ``self._last_quota_state`` remembers the SET of condition
        ``token``s already announced for this ``provider/model_id``; we speak
        only when ``token`` is not yet in that set (a genuinely new condition,
        or a recurrence after a healthy edge cleared the whole selector entry),
        then record it. Steady state is silent.

        ``token`` must be distinct per CONDITION, not merely per state: several
        different conditions (account-scope low/exhausted, model-tier-scope
        low/exhausted, tier-cap-spent-but-shared-remains) key on the same
        selector, and a token that only encoded ``health.state`` would alias
        them — one overwriting another's memory. Callers pass a scoped token
        (``account:<state>``, ``model:<state>``, ``tier-spent:<state>``) so each
        condition dedups against itself alone.
        """
        announced = self._last_quota_state.get(selector)
        if announced is not None and token in announced:
            return
        if announced is None:
            announced = set()
            self._last_quota_state[selector] = announced
        announced.add(token)
        await self._notice(text, kind)

    def _clear_quota_latch(self, selector: str) -> None:
        """Drop every announced quota condition so a real recurrence re-announces.

        Called on the healthy edge: once quota recovers, the next slide back into
        ANY low/exhausted condition is a genuinely new transition the user should
        hear about, not a duplicate of a verdict we already announced before the
        recovery. Recovery resets all conditions on this selector together —
        dropping the whole entry — because a healthy reading clears the account
        as a whole, so every prior condition on it starts fresh.
        """
        self._last_quota_state.pop(selector, None)

    async def _on_route_change(self, target: Any, reason: str) -> None:
        effort = f" ({target.effort} effort)" if target.effort else ""
        await self._notice(f"{reason} — falling back to {target.selector}{effort}")

    async def _on_route_settle(self, target: Any, reason: str) -> None:
        """Forward the effective-route edge (fallback pinned / primary back)."""
        if target is None:
            # The recovery deserves the same narration the failure got: the
            # fallback edge printed "falling back to X", and without this line
            # the model display silently snapping back reads as a glitch, not
            # a recovery. Info, not warning — it is good news. One clause,
            # "back to" — the failure edge's "falling back to" and this pair
            # off the same preposition (design D1).
            await self._notice(f"back to {self._primary_selector}", "info")
        if self._route_handler is None:
            return
        result = self._route_handler(target, reason)
        if inspect.isawaitable(result):
            await result

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

    async def _provider_quota_availability(
        self,
        provider: str,
        model_id: str,
        *,
        reserve_percent: float,
        cache: dict[str, str],
        usage_memo: "dict[str, UsageReport | None] | None" = None,
    ) -> str:
        """Whether ``provider`` still has spendable quota for ``model_id``.

        ``usable`` means at least one account still has remaining > 0
        (healthy *or* reserve — reserve is still spendable). ``depleted``
        means every account that answered is at 0%. ``unknown`` is fail-open:
        no endpoint, no report, or a fetch error, so the caller must not
        skip the target on a missing signal.

        Cached per provider+model for one preflight walk so a chain that
        lists several models on the same host does not re-hit the usage
        endpoint, while still letting a Fable hop and an Opus hop on the
        same Anthropic pool disagree (their binding windows differ).

        ``usage_memo`` carries the walk's per-account reports, which is a
        LEVEL BELOW that verdict memo: two model-scoped verdicts on one pool
        may differ, but they read the same accounts, and this method
        enumerates every OAuth account of the provider. Without the report
        memo, a chain that lists the walk's own provider re-fetched usage for
        an account the primary probe had just read (measured: two GETs for
        one account in one boundary).
        """
        cache_key = f"{provider}/{model_id}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        from local_operator.providers.usage import (
            fetch_usage,
            usage_health,
            usage_supported,
        )
        from local_operator.providers.usage_cache import fingerprint_secret

        if not usage_supported(provider):
            cache[cache_key] = "unknown"
            return "unknown"

        # Enumerate configured rows before asking for refreshed accesses.
        # ``list_oauth_accesses`` deliberately omits a row whose refresh fails;
        # without this comparison, one omitted/unknown account plus one depleted
        # account looked like proof the WHOLE provider was depleted (review F1).
        rows = self._auth_store.list_credentials(provider)
        oauth_rows = [row for row in rows if row.credential_type == "oauth"]
        api_key_rows = [row for row in rows if row.credential_type == "api_key"]

        saw_depleted = False
        saw_unknown = False
        try:
            accesses = await self._auth_store.list_oauth_accesses(provider)
        except Exception:
            accesses = []
            saw_unknown = bool(oauth_rows)
        if {access.credential_id for access in accesses} != {row.id for row in oauth_rows}:
            saw_unknown = True

        for access in accesses:
            try:
                report = await self._cached_account_usage(
                    provider,
                    access.email or access.account_id or f"cred:{access.credential_id}",
                    lambda a=access: fetch_usage(
                        self._http,
                        provider,
                        access_token=a.access_token,
                        account_id=a.account_id,
                        oauth_creds=a.raw,
                    ),
                    usage_memo,
                )
            except Exception:
                saw_unknown = True
                continue
            if report is None:
                saw_unknown = True
                continue
            health = usage_health(report, model_id, reserve_percent=reserve_percent)
            if health.state == "unknown":
                saw_unknown = True
                continue
            if health.state != "depleted":
                cache[cache_key] = "usable"
                return "usable"
            saw_depleted = True

        # Probe the credential the wire cascade would ACTUALLY select now.
        # Usage enumeration includes blocked OAuth rows for visibility, while
        # routing correctly falls through to a healthy API key (review F4).
        # ``read_only`` keeps this quota question from moving stickiness.
        try:
            selected = await self._auth_store.get_oauth_access(
                provider, self._session_id, read_only=True
            )
        except Exception:
            selected = None
            saw_unknown = True
        if selected is not None and selected.kind == "api_key":
            try:
                report = await self._cached_account_usage(
                    provider,
                    fingerprint_secret(selected.access_token),
                    lambda: fetch_usage(self._http, provider, api_key=selected.access_token),
                    usage_memo,
                )
            except Exception:
                report = None
            if report is None:
                saw_unknown = True
            else:
                health = usage_health(report, model_id, reserve_percent=reserve_percent)
                if health.state == "depleted":
                    saw_depleted = True
                elif health.state == "unknown":
                    saw_unknown = True
                else:
                    cache[cache_key] = "usable"
                    return "usable"
            if any(row.id != selected.credential_id for row in api_key_rows):
                # The read-only cascade exposes the winning key, not every
                # lower-priority sibling's secret. A depleted selected key is
                # therefore not proof the provider is empty when another
                # enabled key row exists; fail open and let the stream's
                # credential rotation walk that pool (review F5).
                saw_unknown = True
        elif api_key_rows:
            # An unblocked OAuth row shadows lower API-key rows. The stream
            # reaches them after a provider-side quota error, but preflight
            # cannot safely resolve a lower tier without changing routing.
            # Unknown is fail-open: the key may still hold spendable balance.
            saw_unknown = True

        availability = "depleted" if saw_depleted and not saw_unknown else "unknown"
        cache[cache_key] = availability
        return availability

    async def _first_available_fallback(
        self,
        model: ModelSpec,
        *,
        different_provider: bool = False,
        reserve_percent: float = 10.0,
        quota_cache: dict[str, str] | None = None,
        usage_memo: "dict[str, UsageReport | None] | None" = None,
    ) -> Any | None:
        """The first configured fallback with working auth, bench- and quota-aware.

        "First configured" alone is what replayed the waterfall on every
        message boundary: with the chain's head providers down, each quota
        preflight re-selected the first entry, re-pinned it, and the stream
        walk then re-paid one failure notice and one serial timeout per dead
        target before landing back on the one provider that had actually been
        serving. Targets the stream driver recently benched (see
        ``FailoverRouteState.mark_target_failed``) are therefore passed over
        here, so a session that has settled on a working fallback stays on it.

        A target whose provider is *quota-depleted* is skipped the same way:
        pinning a maxed Kimi/Qwen hop just to watch it fail (or, worse, to
        treat its last 10% as another reason to hop) is how the cascade
        burned past a provider that still had spendable quota. Reserve is
        still usable — only a 0% remaining verdict skips. Unknown/unreachable
        usage fails open, matching preflight's own contract.

        The bench (and the depleted skip) is a preference, not a verdict:
        when EVERY authed candidate is benched or depleted, the first of
        them is returned anyway — returning ``None`` would report "no
        configured fallback" to a user who has several, and the stream
        walk's own retry machinery is the right place to discover which
        bench has expired.

        ``quota_cache`` is the memo of provider+model availability verdicts.
        It belongs to the CALLER's boundary walk, not to one invocation: a
        preflight that rotates through several accounts calls this once per
        rotation step, and a per-call cache re-probed every fallback
        provider's usage endpoint on each of them — a three-account
        Anthropic pool re-asked Kimi three times for one message boundary
        (review F7). The verdicts cannot disagree within a walk anyway: they
        are derived from usage reports fetched over a few hundred
        milliseconds, and ``_provider_quota_availability`` enumerates blocked
        rows too, so the recovery walk lifting a block mid-boundary does not
        change what a re-probe would answer. ``None`` keeps the standalone
        contract for a caller with no walk of its own.
        """
        from local_operator.providers.failover import parse_selector

        first_benched: Any | None = None
        first_depleted: Any | None = None
        if quota_cache is None:
            quota_cache = {}
        for target in self._fallback_targets(model):
            provider, target_model = parse_selector(target.selector)
            if different_provider and provider == model.provider:
                continue
            if not await self._target_has_auth(target):
                continue
            if not self._route_state.target_retry_due(target):
                if first_benched is None:
                    first_benched = target
                continue
            availability = await self._provider_quota_availability(
                provider,
                target_model,
                reserve_percent=reserve_percent,
                cache=quota_cache,
                usage_memo=usage_memo,
            )
            if availability == "depleted":
                if first_depleted is None:
                    first_depleted = target
                continue
            return target
        return first_benched or first_depleted

    @staticmethod
    def _storage_provider(provider: str) -> str:
        from local_operator.providers.registry import credential_provider_id

        return credential_provider_id(provider)

    def _usage_cache_store(self) -> "UsageCacheStore | None":
        """The shared usage cache, built lazily. A cache that cannot open is a
        permanent miss (live fetch), never an error — same contract as the warmer's."""
        if self._usage_cache is None:
            try:
                from local_operator.providers.usage_cache import UsageCacheStore

                self._usage_cache = UsageCacheStore()
            except Exception:  # noqa: BLE001 — no cache = live fetch, never fatal
                return None
        return self._usage_cache

    async def _cached_account_usage(
        self,
        provider: str,
        account_identity: str,
        fetch: "Callable[[], Awaitable[UsageReport | None]]",
        usage_memo: "dict[str, UsageReport | None] | None" = None,
    ) -> "UsageReport | None":
        """Route one preflight usage probe through the shared cross-process cache.

        Collapses a concurrent peer's duplicate fetch of the SAME account (the 429
        storm this fixes) while keeping every boundary free to re-probe live —
        ``leased_account_usage`` never serves a fresh row on the fast path. Fails open
        to a live fetch when the cache is unavailable, so routing can never be made
        WORSE than the pre-cache behaviour. See docs/specs/preflight-usage-cache.md.

        ``usage_memo`` is the boundary walk's per-ACCOUNT report memo, and it is the
        layer that stops one boundary asking the same account's usage endpoint twice.
        It is needed IN ADDITION to the two memos the walk already carries, because
        neither covers this:

        * ``attempted_ids`` dedupes credential rows the walk has JUDGED, but the
          rows enumerated by ``_provider_quota_availability`` (a fallback-chain
          question) are not judgements and are deliberately not recorded there.
        * ``quota_cache`` dedupes provider+model VERDICTS, and cannot be widened:
          a Fable hop and an Opus hop on one Anthropic pool legitimately disagree
          because their binding windows differ.

        The underlying per-account REPORT is identical for both, though — it is one
        GET against one account — so memoizing the report is sound where sharing the
        verdict is not. Measured: a reserve-state account on a chain that lists its
        own provider was fetched twice per boundary (once by the primary probe, once
        by ``_provider_quota_availability`` via ``_first_available_fallback``).

        The cross-process cache cannot collapse this. ``leased_account_usage``
        deliberately never serves a fresh row on its fast path so a routing probe can
        notice recovery on its own next boundary; that contract is right ACROSS
        boundaries and is exactly what leaves the duplication WITHIN one.

        A ``None`` result is memoized like any other: every caller treats it as
        fail-open (unknown usage, keep the existing verdict), so replaying it inside
        one boundary reaches the same conservative outcome the re-probe would — while
        a re-probe of an endpoint that failed milliseconds ago is precisely the burst
        that earns a 429. Freshness is unaffected: the memo lives only for this
        boundary walk and is discarded with it.
        """
        from local_operator.providers.usage_cache import (
            account_preflight_key,
            leased_account_usage,
        )

        storage = self._storage_provider(provider)
        store = self._usage_cache_store()
        key = account_preflight_key(storage, account_identity)
        # Keyed by the same string the shared cache keys on, so the memo can never
        # conflate two accounts that the cache would keep apart (storage aliasing
        # included: ``openai-device`` and ``openai`` are one pool).
        if usage_memo is not None and key in usage_memo:
            return usage_memo[key]
        report = await leased_account_usage(store, key, storage, fetch)
        if usage_memo is not None:
            usage_memo[key] = report
        return report

    async def _primary_has_auth(self, model: ModelSpec) -> bool:
        from local_operator.providers.failover import FallbackTarget

        return await self._target_has_auth(
            FallbackTarget(f"{model.provider}/{model.model_id}", model.reasoning_effort)
        )

    @staticmethod
    def _write_quota_block(
        auth_store: Any,
        credential_id: int,
        storage: str,
        health: Any,
        block_ms: int,
    ) -> None:
        """Record a quota verdict against a credential, scoped to what it binds.

        A verdict whose ONLY binding windows are model-family caps (Anthropic's
        ``7 day (Fable)`` at 100% beside a healthy shared 5-hour window) stops
        ONE family on the account, not the account: the block is written under
        ``model:<family>`` so requests for other families still resolve to the
        row and spend the shared headroom. The moment a shared window binds
        (``health.scope == "account"``) the account is out for every model and
        the block is account-wide, as before. Writing family verdicts
        account-wide is the defect that made an opus request report "all
        credentials unusable" on a pool whose only spent window was Fable's.
        """
        if health.scope == "model" and health.binding_families:
            for fam in health.binding_families:
                auth_store.block_credential(
                    credential_id,
                    storage,
                    block_scope=f"model:{fam}",
                    block_ms=block_ms,
                )
        else:
            auth_store.block_credential(credential_id, storage, block_ms=block_ms)

    async def preflight_usage(self, model: ModelSpec, *, consume_boundary: bool = True) -> None:
        """Check reliable OAuth quota once per user-message boundary.

        Unknown/unreachable usage fails open. A low account is suppressed only
        when a sibling or configured fallback is ready, so preflight can never
        turn usable reserve capacity into a dead end.
        """
        from local_operator.providers.failover import RetrySettings

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
        # The boundary token also gates effort CLASSIFICATION in ``__call__``;
        # a switch-time probe (``consume_boundary=False``) must not spend it,
        # or a mid-turn ``/model`` would silently skip the next request's
        # effort grading.
        if consume_boundary:
            self._message_boundary_pending = False
        if recheck_due:
            self._quota_recheck_for = None

        now = time.monotonic()
        if (
            selector == self._usage_checked_selector
            and now - self._usage_checked_at < self.USAGE_CHECK_TTL_S
            and not self._route_state.quota_pinned
        ):
            # The memo dedupes the several requests ONE message makes; a
            # quota-pinned route is re-probed at every boundary regardless,
            # or a session could sit on a fallback for a whole memo window
            # past the primary's recovery.
            return
        self._usage_checked_selector = selector
        self._usage_checked_at = now

        retry = RetrySettings.from_settings(self._settings)
        # Credential blocks are scoped to the model family that spent them,
        # so a spent family cap never takes the account out of rotation for
        # models of another family; the reads below are model-scoped.
        if (
            self._route_state.active is not None
            and not self._route_state.primary_retry_due()
            and not self._route_state.quota_pinned
        ):
            # A quota pin is re-probed at every boundary: the usage endpoint
            # answers definitively and cheaply, and its cooldown can be hours
            # long (a 24h advertised reset), which would otherwise glue the
            # session to a fallback past the primary's window reopening.
            # Transport pins keep the cooldown — their recovery is not
            # observable without re-paying the failure.
            return
        if not retry.usage_aware_fallback:
            if self._route_state.active is not None and await self._primary_has_auth(model):
                # A real recovery edge, not bookkeeping: a fallback was pinned
                # and requests are about to return to the primary, so the
                # host's model display has to hear about it (settled, not
                # silent — same reasoning as the stream driver's clear).
                await self._route_state.clear_settled("primary model recovered")
            return

        attempted_ids: set[int] = set()
        # One memo for the WHOLE boundary walk. Rotating through this
        # provider's accounts re-enters the loop, and each pass may consult
        # the fallback chain; scoping the memo per call re-probed every
        # fallback provider's usage endpoint once per rotation step (review
        # F7). See ``_first_available_fallback`` for why a verdict is stable
        # across one walk.
        quota_cache: dict[str, str] = {}
        # The per-ACCOUNT report memo for this same walk, one level below the
        # verdict memo above. Two model-scoped verdicts on one pool may
        # legitimately disagree, but they read the SAME accounts, and the
        # rotation/fallback steps each enumerate them again — so the reports
        # are what duplicates. See ``_cached_account_usage`` for why neither
        # ``attempted_ids`` nor ``quota_cache`` can cover this, and why the
        # cross-process cache deliberately does not either. Scoped to the
        # walk and discarded with it, so the next boundary still re-probes
        # live and can notice recovery.
        usage_memo: dict[str, UsageReport | None] = {}
        while True:
            try:
                access = await self._auth_store.get_oauth_access(
                    model.provider, self._session_id, model_id=model.model_id
                )
            except Exception:
                return
            if access is None:
                storage = self._storage_provider(model.provider)
                rows = self._auth_store.list_credentials(storage)
                if rows and all(
                    self._auth_store.is_blocked_for_model(row.id, storage, model.model_id)
                    for row in rows
                ):
                    # Every account is under a block, but "blocked" is only a
                    # verdict from an earlier probe — quota resets while the
                    # backoff is still on the clock, and a tier-scoped cap can
                    # block an account that still serves other models. Fail over
                    # to another provider only after re-checking the blocks
                    # themselves: exhaust every login first.
                    recovered = await self._recover_blocked_accounts(
                        model, storage, rows, retry, attempted_ids, usage_memo
                    )
                    if recovered is not None:
                        health, shared_remaining, tier_binding, access = recovered
                        if health.state != "healthy" and await self._apply_account_health(
                            model,
                            access,
                            storage,
                            health,
                            shared_remaining,
                            tier_binding,
                            retry,
                            attempted_ids,
                            quota_cache,
                            usage_memo,
                        ):
                            continue
                        return
                    fallback = await self._first_available_fallback(
                        model,
                        different_provider=True,
                        reserve_percent=retry.usage_reserve_percent,
                        quota_cache=quota_cache,
                        usage_memo=usage_memo,
                    )
                    if fallback is not None:
                        await self._route_state.activate(
                            fallback,
                            f"{model.provider} credentials temporarily unavailable",
                            quota=True,
                        )
                return
            if access.kind != "oauth" or access.credential_id in attempted_ids:
                return
            attempted_ids.add(access.credential_id)

            from local_operator.providers.usage import (
                fetch_usage,
                shared_tier_saturation,
                usage_health,
            )

            report = await self._cached_account_usage(
                model.provider,
                access.email or access.account_id or f"cred:{access.credential_id}",
                lambda a=access: fetch_usage(
                    self._http,
                    model.provider,
                    access_token=a.access_token,
                    account_id=a.account_id,
                ),
                usage_memo,
            )
            if report is None:
                return
            health = usage_health(
                report,
                model.model_id,
                reserve_percent=retry.usage_reserve_percent,
            )
            if health.state == "healthy":
                # Settled for the same reason as the auth-only path above: this
                # clear is the moment a pinned fallback stops serving requests.
                # Also drop the quota-notice latch: recovery means the next
                # slide back to low/exhausted is a real new transition to
                # announce, not a duplicate of what we said before recovery.
                self._clear_quota_latch(selector)
                await self._route_state.clear_settled("primary model recovered")
                return
            if health.state == "unknown":
                # Fail-open, indeterminate: NOT a transition. Leave the latch as
                # it stands so an unreadable probe between two low readings does
                # not reset the dedup and let the next low reading re-announce.
                return

            shared_remaining, tier_binding = shared_tier_saturation(
                report,
                reserve_percent=retry.usage_reserve_percent,
            )
            remaining = (
                ""
                if health.remaining_fraction is None
                else f" ({health.remaining_fraction * 100:.0f}% remaining)"
            )
            condition = "quota exhausted" if health.state == "depleted" else "quota low"
            storage = self._storage_provider(model.provider)
            if health.scope != "account":
                # A model-tier cap (Anthropic's ``7 day (Fable)`` against
                # ``claude-fable-5``) is still per ACCOUNT. Jumping to the
                # next provider here is what skipped three Anthropic logins
                # that still had Fable headroom — the reported cascade that
                # hopped Anthropic → Kimi (10% remaining) → Qwen (maxed) →
                # Grok while Fable quota sat idle. Rotate siblings first;
                # only the last account on this provider may leave it.
                row = self._auth_store.get_credential(access.credential_id)
                siblings = [
                    candidate
                    for candidate in self._auth_store.list_credentials(storage)
                    if candidate.id != access.credential_id
                    and candidate.id not in attempted_ids
                    and (row is None or candidate.credential_type == row.credential_type)
                    and not self._auth_store.is_blocked_for_model(
                        candidate.id, storage, model.model_id
                    )
                ]
                if not siblings and health.state == "depleted":
                    # No UNBLOCKED sibling can take this model, but blocked
                    # rows are earlier verdicts, not facts: a block written
                    # when the account was low (or by an older build that
                    # blocked reserve accounts for days) can be hiding the
                    # ONLY spendable quota for this model. The reported
                    # incident was exactly that — two accounts blocked at
                    # 8%/4% Fable while the one live account read 0% and the
                    # session hopped providers. Probe the blocked rows before
                    # leaving the provider; ``None`` means every account was
                    # re-checked and genuinely cannot serve, which is the
                    # only honest moment to fall back.
                    blocked_rows = [
                        candidate
                        for candidate in self._auth_store.list_credentials(storage)
                        if candidate.id != access.credential_id
                        and candidate.id not in attempted_ids
                        and (row is None or candidate.credential_type == row.credential_type)
                        and self._auth_store.is_blocked_for_model(
                            candidate.id, storage, model.model_id
                        )
                    ]
                    recovered = await self._recover_blocked_accounts(
                        model, storage, blocked_rows, retry, attempted_ids, usage_memo
                    )
                    if recovered is not None:
                        rec_health = recovered[0]
                        if rec_health.state in ("healthy", "reserve"):
                            # A blocked account holds spendable quota for
                            # this model and the walk pinned the session to
                            # it. Settle here: re-walking would let the
                            # sibling-rotation step demote the recovered
                            # account in favour of the depleted one it just
                            # replaced.
                            if rec_health.state == "reserve":
                                await self._notice(
                                    f"{model.provider} blocked account recovered "
                                    f"({rec_health.remaining_fraction * 100:.0f}% remaining) "
                                    f"for {model.model_id} — resuming {model.provider}",
                                    "info",
                                )
                                await self._route_state.clear_settled("recovered account has quota")
                            return
                        # Recovered but depleted for this model: the shared
                        # policy re-blocks and activates the fallback, naming
                        # the quota in its notice. Its True return means a
                        # sibling took over (with several blocked rows the
                        # walk unblocks them one at a time), and discarding
                        # that signal ended preflight with a depleted row
                        # unblocked and no fallback pinned (review F2) — so
                        # re-enter the walk exactly like the other callers.
                        if await self._apply_account_health(
                            model,
                            recovered[3],
                            storage,
                            rec_health,
                            recovered[1],
                            recovered[2],
                            retry,
                            attempted_ids,
                            quota_cache,
                            usage_memo,
                        ):
                            continue
                        return
                if siblings:
                    if health.state == "depleted":
                        self._write_quota_block(
                            self._auth_store,
                            access.credential_id,
                            storage,
                            health,
                            max(60_000, health.reset_after_ms or self.DEFAULT_USAGE_BLOCK_MS),
                        )
                    else:
                        self._auth_store.deprioritize_credential(
                            model.provider, access.credential_id
                        )
                    # Rotating to another same-provider account is an internal
                    # implementation detail — the user's request is still being
                    # served on this provider, just on a different login. It
                    # used to emit a notice per rotation, which spammed the
                    # transcript on every boundary. Rotate silently.
                    continue
                if health.state == "reserve":
                    # Last account, still holding this model's quota. Same
                    # rule as the account-scope path: reserve is not a
                    # licence to leave the provider. Deduped per condition:
                    # "quota low, continuing" is worth one line on the
                    # transition, not one per message for as long as the account
                    # stays low. ``model:`` scope token — this is the model-tier
                    # branch, and its token must stay distinct from the
                    # account-scope branch's ``account:`` token so the two
                    # conditions cannot alias on this shared selector.
                    await self._announce_quota_change(
                        selector,
                        f"model:{health.state}",
                        f"{model.provider} {condition}{remaining} for {model.model_id} "
                        f"— continuing until {model.provider} quota is exhausted",
                        "info",
                    )
                    await self._route_state.clear_settled("primary model still has quota")
                    return
                fallback = await self._first_available_fallback(
                    model,
                    reserve_percent=retry.usage_reserve_percent,
                    quota_cache=quota_cache,
                    usage_memo=usage_memo,
                )
                if fallback is None:
                    # Deduped per condition: the quota is spent and nothing can
                    # take over, but the user only needs that told once per
                    # transition — not on every message while the condition
                    # holds and no fallback appears. ``model:`` scope token,
                    # distinct from the account-scope ``account:`` token.
                    await self._announce_quota_change(
                        selector,
                        f"model:{health.state}",
                        f"{model.provider} {condition}{remaining} for {model.model_id}; "
                        "no configured model fallback is available",
                    )
                    return
                await self._route_state.activate(
                    fallback,
                    f"{model.provider} {condition}{remaining} for {model.model_id}",
                    quota=True,
                )
                return

            if await self._apply_account_health(
                model,
                access,
                storage,
                health,
                shared_remaining,
                tier_binding,
                retry,
                attempted_ids,
                quota_cache,
                usage_memo,
            ):
                continue
            return

    async def _apply_account_health(
        self,
        model: ModelSpec,
        access: Any,
        storage: str,
        health: Any,
        shared_remaining: float | None,
        tier_binding: bool,
        retry: Any,
        attempted_ids: set[int],
        quota_cache: dict[str, str] | None = None,
        usage_memo: "dict[str, UsageReport | None] | None" = None,
    ) -> bool:
        """Act on a low/depleted account-scope verdict.

        Returns True when the caller should re-resolve credentials (a sibling
        account took over), False when the routing decision is final.

        ``quota_cache`` is the caller's boundary-walk memo of fallback
        availability, threaded through so the chain is probed once per
        boundary rather than once per account rotation (review F7).
        ``usage_memo`` is the same walk's per-account report memo, threaded
        for the same reason one level down: the fallback chain may list this
        walk's own provider, whose accounts have already been read.

        The binding windows that produced ``health`` can be scoped to a model
        tier while the shared windows still hold quota (Anthropic's
        ``7 day (Fable)`` at 100% beside an 11%-free 5-hour window, when the
        model being routed never draws on Fable). Taking that account out of
        rotation — and, once every account reports the same shape, failing
        over to another provider — strands real shared headroom. Rotation is
        reserved for accounts whose SHARED windows are genuinely binding; a
        tier-only cap keeps the account in service, and the last account with
        any shared headroom — including remaining under the reserve
        threshold — is always allowed to spend it down to zero before
        a provider fallback is even considered. Reserve is a preference
        between siblings of the same provider, not a hop to the next one.
        """
        from local_operator.providers.failover import parse_selector

        # Same dedup key the boundary check uses: quota notices out of this
        # method latch on ``provider/model_id`` so a persistent verdict is
        # announced once per transition, not once per message (see
        # ``_announce_quota_change``).
        selector = f"{model.provider}/{model.model_id}"
        threshold = min(100.0, max(0.0, float(retry.usage_reserve_percent))) / 100.0
        # ``None`` means no shared window carried a number — indeterminate, not
        # headroom, so the tier-cap guard stays off and the cautious rotate /
        # failover path runs.
        shared_above_reserve = shared_remaining is not None and shared_remaining > threshold
        remaining = (
            ""
            if health.remaining_fraction is None
            else f" ({health.remaining_fraction * 100:.0f}% remaining)"
        )
        condition = "quota exhausted" if health.state == "depleted" else "quota low"

        row = self._auth_store.get_credential(access.credential_id)
        siblings = [
            candidate
            for candidate in self._auth_store.list_credentials(storage)
            if candidate.id != access.credential_id
            and candidate.id not in attempted_ids
            and (row is None or candidate.credential_type == row.credential_type)
            and not self._auth_store.is_blocked_for_model(candidate.id, storage, model.model_id)
        ]
        fallback = await self._first_available_fallback(
            model,
            # A different effort cannot revive a fully exhausted provider,
            # but it can preserve reserve quota by reducing token spend.
            different_provider=health.state == "depleted",
            reserve_percent=retry.usage_reserve_percent,
            quota_cache=quota_cache,
            usage_memo=usage_memo,
        )
        if not siblings and health.state == "reserve":
            # Last account on this provider, still holding spendable quota.
            # Crossing the reserve threshold used to hop to the next chain
            # entry (Kimi at 10% remaining → Qwen maxed → Grok) while this
            # account could still serve. Reserve is a preference BETWEEN
            # siblings of the same provider, not a licence to leave the
            # provider; spend it to zero, then fail over. A same-provider
            # lower-effort hop is still allowed — it reduces token spend
            # without abandoning remaining quota.
            if fallback is not None:
                fallback_provider, _model_id = parse_selector(fallback.selector)
                if fallback_provider == model.provider:
                    await self._route_state.activate(
                        fallback,
                        f"{model.provider} {condition}{remaining}",
                        quota=True,
                    )
                    return False
            # ``account:`` scope token: this is the account-scope path, and its
            # condition must dedup separately from the model-tier branch that
            # shares this selector.
            await self._announce_quota_change(
                selector,
                f"account:{health.state}",
                f"{model.provider} {condition}{remaining} — continuing until "
                f"{model.provider} quota is exhausted",
                "info",
            )
            await self._route_state.clear_settled("primary model still has quota")
            return False

        if not siblings and health.state == "depleted":
            # About to leave the provider on a depleted verdict while other
            # accounts sit under blocks. Blocks are earlier verdicts — an
            # older build's days-long reserve block can be hiding the only
            # spendable quota left (the incident behind this split: two
            # accounts blocked at 8%/4% while the live one read 0%). Probe
            # the blocked rows before the hop; ``None`` means every account
            # was re-checked and genuinely cannot serve.
            #
            # The account under verdict is blocked FIRST: its depletion is a
            # definite reading, and a walk that settles on a recovered
            # sibling returns before the tail of this method — which is
            # where the block used to be written — leaving a spent account
            # in the unblocked pool (review F2's second half).
            self._write_quota_block(
                self._auth_store,
                access.credential_id,
                storage,
                health,
                max(60_000, health.reset_after_ms or self.DEFAULT_USAGE_BLOCK_MS),
            )
            blocked_rows = [
                candidate
                for candidate in self._auth_store.list_credentials(storage)
                if candidate.id != access.credential_id
                and candidate.id not in attempted_ids
                and (row is None or candidate.credential_type == row.credential_type)
                and self._auth_store.is_blocked_for_model(candidate.id, storage, model.model_id)
            ]
            recovered = await self._recover_blocked_accounts(
                model, storage, blocked_rows, retry, attempted_ids, usage_memo
            )
            if recovered is not None:
                rec_health = recovered[0]
                if rec_health.state in ("healthy", "reserve"):
                    # A blocked account holds spendable quota and the walk
                    # pinned the session to it: settle instead of leaving
                    # the provider on the depleted account under verdict.
                    if rec_health.state == "reserve":
                        await self._notice(
                            f"{model.provider} blocked account recovered "
                            f"({rec_health.remaining_fraction * 100:.0f}% remaining) "
                            f"— resuming {model.provider}",
                            "info",
                        )
                        await self._route_state.clear_settled("recovered account has quota")
                    return False
                # Recovered but depleted: the shared policy re-blocks and
                # activates the fallback; whether a sibling then takes over
                # decides our return.
                return await self._apply_account_health(
                    model,
                    recovered[3],
                    storage,
                    rec_health,
                    recovered[1],
                    recovered[2],
                    retry,
                    attempted_ids,
                    quota_cache,
                    usage_memo,
                )

        if not siblings and fallback is None:
            # Deduped per condition: spent with nowhere to fall back is worth
            # one line on the transition, not a repeat on every subsequent
            # message while the account stays spent. ``account:`` scope token,
            # distinct from the model-tier branch's ``model:`` token.
            await self._announce_quota_change(
                selector,
                f"account:{health.state}",
                f"{model.provider} {condition}{remaining}; no configured fallback is available",
            )
            return False

        if tier_binding and shared_above_reserve:
            # The tight window is a scoped tier cap, not the shared pool, and
            # the shared windows still hold reserve. The current model does not
            # draw on that tier, so the account keeps serving: spend the shared
            # headroom down to zero instead of rotating or failing over on a
            # cap that does not gate this request.
            binding = "/".join(health.binding_labels) or "a model-tier window"
            # Route through the dedup helper, not a raw ``_notice``: preflight
            # runs on every message boundary, so a tier cap that stays spent
            # while shared quota holds would echo this "continuing…" line on
            # every message — the exact per-boundary spam this latch exists to
            # kill. The ``tier-spent:`` token must be DISTINCT from the plain
            # ``account:``/``model:`` state tokens: this is a separate condition
            # (tier cap spent but shared remains) that can hold at the same time
            # as, and on the same selector as, a plain low/exhausted verdict, so
            # sharing a token would let the two conditions alias — one masking
            # the other. A distinct token dedups this condition against itself
            # alone.
            await self._announce_quota_change(
                selector,
                f"tier-spent:{health.state}",
                f"{model.provider} {binding} spent; shared quota remains{remaining} "
                "— continuing until shared windows are exhausted",
                "info",
            )
            self._route_state.clear()
            return False

        # How the account is taken out of the running depends on WHAT the
        # verdict was, and conflating the two is the incident this split
        # comes from. "Depleted" is a fact about the provider: it will 429
        # every request until the spent window resets, so a cross-process
        # SQLite block until that reset merely records reality. "Reserve"
        # is the opposite of unusable — the account still HAS quota, held
        # back so it is there when nothing better remains. Writing a block
        # for it (as this code once did) stood the reserve on its head:
        # accounts at 90% of a seven-day window were blocked for DAYS, one
        # by one, until the last live account genuinely depleted and every
        # session died reporting "all credentials unusable" while three
        # accounts still held quota.
        #
        # So a reserve account is DEPRIORITIZED instead: an in-process,
        # self-expiring routing preference (see
        # ``AuthStore.deprioritize_credential``) that steers this walk and
        # the session's next resolve toward healthier siblings, while the
        # cascade's ignore-demotions second pass still serves the account
        # the moment it is the only thing left. The mark is short-lived on
        # purpose; the preflight re-checks and re-applies it while the
        # preference still holds.
        if health.state == "depleted":

            def take_out_of_rotation(credential_id: int) -> None:
                self._write_quota_block(
                    self._auth_store,
                    credential_id,
                    storage,
                    health,
                    max(60_000, health.reset_after_ms or self.DEFAULT_USAGE_BLOCK_MS),
                )

        else:

            def take_out_of_rotation(credential_id: int) -> None:
                # Keyed by ``model.provider``, not ``storage``: demotions
                # are consulted by ``_resolve`` under the provider name the
                # request resolves with.
                self._auth_store.deprioritize_credential(model.provider, credential_id)

        if siblings:
            take_out_of_rotation(access.credential_id)
            # Silent: rotating to another same-provider account is an internal
            # implementation detail. The request is still served on the same
            # provider, so the per-rotation notice this used to emit was pure
            # churn on the transcript (once per boundary while quota was low).
            return True

        assert fallback is not None
        fallback_provider, _model_id = parse_selector(fallback.selector)
        if fallback_provider != model.provider:
            take_out_of_rotation(access.credential_id)
        await self._route_state.activate(
            fallback,
            f"{model.provider} {condition}{remaining}",
            quota=True,
        )
        return False

    async def _recover_blocked_accounts(
        self,
        model: ModelSpec,
        storage: str,
        rows: list[Any],
        retry: Any,
        attempted_ids: set[int],
        usage_memo: "dict[str, UsageReport | None] | None" = None,
    ) -> tuple[Any, float | None, bool, Any] | None:
        """Re-check blocked accounts before a provider failover.

        A block is a stale verdict the moment a window resets — and preflight
        takes accounts out of rotation on nothing more than crossing a reserve
        threshold, so a pool that still has spendable quota can look exactly
        like a dead one. Each blocked row is probed with its OWN refreshed
        token (asking the cascade would resolve to whichever credential
        outranks the row, and with a healthy unblocked sibling in the pool
        every probe answered for that sibling — re-blocking rows that held
        the only spendable quota). Refresh failures and unknown/unreachable
        reports leave the row's existing block standing and move on. A
        usable verdict (healthy or reserve) in the wave wins over a depleted
        one: the row's block is lifted, the session is pinned to it, and the
        (health, shared, tier, access) tuple goes back to the caller's
        shared policy. A depleted verdict is returned only when the whole
        wave is genuinely out — which is what re-blocks that row and, if
        nothing later recovers, lets the caller hop. ``None`` means every
        blocked account was re-checked and none gave a verdict — only then
        is a provider fallback honest.

        ``attempted_ids`` is the preflight's record of which credentials this
        message boundary has already judged, and it is BOTH read and written
        here. That is what terminates the walk. A depleted verdict sends the
        caller back into ``_apply_account_health``, which re-blocks the row
        and walks the blocked pool again; without recording the probe, the
        row this frame just cleared is blocked again by the next frame and
        re-enumerated by the one after, so two depleted blocked accounts
        ping-pong A→B→A until the recursion limit kills the turn. Every row
        the walk touches is recorded, whatever the outcome: a refresh failure
        or an unreadable report is still a decision taken about that account
        for this boundary, and re-probing it costs a network round trip to
        reach the same answer. Rows are finite and the set only grows, so
        each recursive step strictly shrinks the candidate pool.

        **Probes run concurrently, in bounded waves, but the VERDICT is still
        decided in row order.** The walk used to be strictly serial, which on
        a pool with several blocked rows is a network train (a refresh plus a
        usage GET per row, one after another) on the time-to-usable path, and
        it is what generated the self-inflicted 429 burst whose backoff then
        poisoned the next boot. Three properties keep the concurrent form
        equivalent to the serial one, and each is load-bearing:

        * **Ordering.** ``asyncio.gather`` preserves result order regardless
          of completion order, and the verdict is selected by scanning that
          ordered list. A usable recovery is preferred over a depleted one
          in the same wave (see the scan below); among equals, the first in
          ROW order wins, not whichever probe happened to answer first —
          which is what makes the choice deterministic and reproducible
          rather than a race between siblings.
        * **Attribution.** Each probe reads its own row's refreshed token and
          builds its own ``OAuthAccess``; nothing is shared between probes, so
          running them together cannot cross a verdict onto another row. This
          is the invariant the serial form protected by construction and the
          one whose breakage would take a healthy credential out of rotation.
        * **Termination.** Every row in a launched wave is recorded in
          ``attempted_ids`` BEFORE that wave starts, so the walk's shrinking
          candidate pool is unchanged. Rows in LATER waves are not reserved:
          once a verdict is found, the remaining waves are never launched, and
          reserving them would mark accounts as judged that were never probed
          — retiring, for this whole boundary, credentials nobody ever looked
          at. The serial walk left them untouched for exactly that reason.

        The wave size is capped by
        :data:`SessionStreamFn.USAGE_RECOVERY_PROBE_CONCURRENCY`; see that
        constant for why an unbounded gather is the wrong shape here.
        """
        import asyncio

        from local_operator.providers.registry import get_provider_definition
        from local_operator.providers.usage import fetch_usage, usage_health

        async def probe(row: Any) -> tuple[Any, dict[str, Any], str] | None:
            """Read one row's own usage, or None when it yields no verdict.

            Every failure mode the serial walk handled with ``continue`` is a
            ``None`` here — refresh failure, missing token, unreachable
            endpoint — so an exception can never escape one probe and cancel
            its siblings' gather. Returns the row alongside its credentials
            and token because the caller needs all three to build the access,
            and re-reading them after the gather would refresh twice.
            """
            try:
                # Probe the row's OWN refreshed token. Clearing the block and
                # re-asking the cascade (the first shape of this walk)
                # attributed the verdict to whatever the cascade returned —
                # with a healthy unblocked sibling in the pool, EVERY probe
                # resolved to that sibling, re-blocked the row just lifted,
                # and the walk ended "nothing recovered" while blocked
                # accounts held spendable quota. Reading the row directly
                # makes the verdict about the row, and leaves the pool's
                # blocks and stickiness untouched until a verdict says
                # otherwise. Concurrent refreshes of DISTINCT rows are safe:
                # ``AuthStore`` holds a per-row refresh lock.
                creds = await self._auth_store.ensure_oauth_fresh(row.id)
            except Exception:
                creds = None
            if creds is None:
                return None  # refresh failed: the block stands
            definition = get_provider_definition(model.provider)
            key_fn = definition.get_api_key if definition is not None else None
            token = key_fn(creds) if key_fn else creds.get("access")
            if not token:
                return None
            try:
                report = await self._cached_account_usage(
                    model.provider,
                    creds.get("email") or creds.get("account_id") or f"cred:{row.id}",
                    lambda t=token, c=creds: fetch_usage(
                        self._http,
                        model.provider,
                        access_token=t,
                        account_id=c.get("account_id"),
                    ),
                    usage_memo,
                )
            except Exception:
                # The serial form let an exception here propagate to
                # preflight's own guard. Inside a gather it would cancel the
                # siblings, so it is contained and read as "no verdict" —
                # the same outcome an unreachable endpoint already produces,
                # and one that leaves the row's block standing.
                return None
            if report is None:
                return None  # unreachable quota endpoint: keep the block
            return report, creds, token

        # Reserve and probe one bounded wave at a time. Slicing (rather than a
        # semaphore over all rows) is what keeps the unprobed tail out of
        # ``attempted_ids``: a semaphore would still have to launch — and so
        # reserve — every row up front.
        pending = [row for row in rows if row.id not in attempted_ids]
        for start in range(0, len(pending), self.USAGE_RECOVERY_PROBE_CONCURRENCY):
            wave = pending[start : start + self.USAGE_RECOVERY_PROBE_CONCURRENCY]
            # Recorded BEFORE the wave's probes run, so every outcome —
            # refresh failure, missing token, unreachable endpoint, unreadable
            # report, or a definite verdict — leaves these rows out of the
            # next enumeration. See the docstring: this is the walk's
            # termination guarantee, not an optimisation.
            for row in wave:
                attempted_ids.add(row.id)
            results = await asyncio.gather(*(probe(row) for row in wave))
            # Row order, not completion order: see the docstring's ordering
            # property. A later row's verdict must never pre-empt an earlier
            # row's just because its request finished first.
            #
            # Prefer a USABLE verdict (healthy/reserve) over a depleted one
            # in the same wave. The serial walk returned the first definite
            # verdict of any kind, then the caller re-entered
            # ``_apply_account_health`` which walked the remaining blocked
            # rows — so a depleted first row never hid a later sibling that
            # still held quota (review F2/F5). Reserving the whole wave in
            # ``attempted_ids`` (required for termination of THIS gather)
            # would make that re-entry skip the rest of the wave, so a
            # depleted-then-healthy pair in one wave would hop providers
            # while the healthy row sat reserved and unconsulted. Scanning
            # the already-paid reports for a usable recovery first produces
            # the same observable (depleted rows stay blocked, the healthy
            # sibling serves) without the ping-pong, and only returns a
            # depleted verdict when the whole wave is genuinely out — which
            # is when the caller is honest to hop.
            first_depleted: tuple[Any, Any, Any, dict[str, Any], str] | None = None
            for row, result in zip(wave, results):
                if result is None:
                    continue
                report, creds, token = result
                health = usage_health(
                    report,
                    model.model_id,
                    reserve_percent=retry.usage_reserve_percent,
                )
                if health.state == "unknown":
                    continue  # unreadable: the block stands, try the next row
                if health.state in ("healthy", "reserve"):
                    return await self._settle_recovered_account(
                        model,
                        storage,
                        row,
                        report,
                        health,
                        creds,
                        token,
                        retry,
                    )
                if first_depleted is None:
                    first_depleted = (row, report, health, creds, token)
            if first_depleted is not None:
                row, report, health, creds, token = first_depleted
                return await self._settle_recovered_account(
                    model,
                    storage,
                    row,
                    report,
                    health,
                    creds,
                    token,
                    retry,
                )
        return None

    async def _settle_recovered_account(
        self,
        model: ModelSpec,
        storage: str,
        row: Any,
        report: "UsageReport",
        health: Any,
        creds: dict[str, Any],
        token: str,
        retry: Any,
    ) -> tuple[Any, float | None, bool, Any]:
        """Apply a definite recovery verdict to the row it was read for.

        Split out of the walk so the concurrent form has exactly ONE place
        that mutates blocks and stickiness, reached only after the verdict has
        been selected in row order. Probes must not write here: two siblings
        settling at once is how a pinned session ends up on the account that
        merely answered first.
        """
        from local_operator.providers.auth_store import OAuthAccess
        from local_operator.providers.usage import shared_tier_saturation

        # A definite verdict about the row just probed. The block is a
        # stale claim this probe has now superseded: lift it and pin the
        # session to the exact credential the usage was read for, then
        # hand the verdict to the caller's shared policy (settle, rotate,
        # or block-again-and-fall-back). A depleted verdict is returned,
        # not swallowed — the policy decides its fate, and the caller's
        # fallback notice must name the quota, not the credential pool.
        # A definite verdict supersedes exactly the blocks that could
        # hide this model: the account-wide backoff and every scoped
        # block whose family gates it. Blocks for OTHER families stay
        # standing — a probe that proves opus serviceable says nothing
        # about a fable weekly that is still spent.
        self._auth_store.clear_blocks_for_model(row.id, storage, model.model_id)
        self._auth_store.pin_session_credential(model.provider, self._session_id, row.id)
        shared_remaining, tier_binding = shared_tier_saturation(
            report,
            reserve_percent=retry.usage_reserve_percent,
        )
        if health.state == "healthy":
            # A recovered account is a healthy edge like the boundary
            # probe's: drop the quota latch so a later re-entry into
            # low/exhausted announces afresh rather than being deduped
            # against the pre-recovery verdict.
            self._clear_quota_latch(f"{model.provider}/{model.model_id}")
            await self._notice(
                f"{model.provider} account quota recovered — resuming {model.provider}",
                "info",
            )
            self._route_state.clear()
        access = OAuthAccess(
            access_token=token,
            credential_id=row.id,
            account_id=creds.get("account_id"),
            email=creds.get("email"),
            org_id=creds.get("org_id"),
            kind="oauth",
            raw=creds,
        )
        return health, shared_remaining, tier_binding, access

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
            async for event in self._record_stream(
                request,
                stream_with_failover(
                    request,
                    self._auth_store,
                    self._settings,
                    self._client_for,
                    signal=signal,
                    session_id=self._session_id,
                ),
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
        # The harness loop steps the effort down one rung when a reasoning
        # model spends its whole output budget thinking and produces nothing
        # (empty ``length`` truncation); that retreat rides on the request as
        # ``effort_ceiling``. The frozen override holds a classification
        # steady — it must not raise the retry back to the rung that just
        # produced silence.
        ceiling = request.effort_ceiling
        ladder = request.model.reasoning_efforts
        if effort is not None and ceiling is not None and ceiling in ladder and effort in ladder:
            if ladder.index(effort) > ladder.index(ceiling):
                effort = ceiling
        if effort is not None:
            request = request.model_copy(
                update={"model": request.model.model_copy(update={"reasoning_effort": effort})}
            )

        if self._cache_lineage_id and request.prompt_cache_key is None:
            # The transcript directory name is stable for the session, so
            # reusing it keeps every turn on the same provider cache without
            # coupling the harness loop to session storage. For a FORK this is
            # the PARENT's id (see ``_cache_lineage_id``): the fork replays a
            # byte-identical transcript, so it really is the same prefix, and a
            # routing/stickiness hint is exactly what should follow it. Without
            # the inheritance a fork's first request routes as a fresh prefix —
            # the same class of regression measured when this key was stripped
            # entirely, which moved cache-read rates from ~97-98% to ~89-90%.
            #
            # Only the OpenAI-shaped wire reads this key
            # (``OpenAICompatClient._build_responses_body``); Anthropic keys its
            # cache on prefix CONTENT, so a fork hits there on byte-identity
            # alone and is unaffected either way.
            request = request.model_copy(update={"prompt_cache_key": self._cache_lineage_id})

        await self.preflight_usage(request.model)
        async for event in self._record_stream(
            request,
            stream_with_failover(
                request,
                self._auth_store,
                self._settings,
                self._client_for,
                signal=signal,
                session_id=self._session_id,
                route_state=self._route_state,
            ),
        ):
            yield event

    async def _record_stream(
        self, request: ChatRequest, stream: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[StreamEvent]:
        """Forward a provider stream unchanged, then record its usage analytics.

        Why the recording lives HERE, wrapping the one place every provider
        call already funnels through: this method sees both the ``ChatRequest``
        (system blocks, tools, messages — the component breakdown) and the
        final ``Usage`` (authoritative provider counts), for turns, tool loops,
        compaction summaries, auto-naming, and every subagent, with no per-call
        wiring anywhere else. A universal view falls out of one wrapper.

        Latency contract: recording happens ONLY after the stream is fully
        consumed (``async for`` completes), so it adds nothing to the response
        the caller is awaiting. The single piece of event-loop work — reading
        the request's component character lengths — is done up front into a
        scalar snapshot because the transcript mutates the messages after the
        call returns; tokenising, apportioning, and the SQLite write all happen
        on the recorder's background thread. Everything is wrapped so a failure
        in analytics can never break a turn.
        """
        # Snapshot char lengths BEFORE streaming: cheap (string length reads,
        # sub-millisecond even on a very large context) and safe to hand a
        # background thread, unlike the live message objects.
        try:
            from local_operator.analytics import snapshot_component_chars

            component_chars = snapshot_component_chars(request)
        except Exception:  # noqa: BLE001 — analytics must never break a turn
            component_chars = None

        final_usage: Usage | None = None
        ok = True
        try:
            async for event in stream:
                usage = getattr(event, "usage", None)
                if usage is not None:
                    final_usage = usage
                stop_reason = getattr(event, "stop_reason", None)
                if stop_reason in ("error", "aborted") or getattr(event, "error", None):
                    ok = False
                yield event
        finally:
            # In a ``finally`` so an aborted/failed stream (which still cost
            # input tokens) is recorded too — best-effort and never raising.
            if component_chars is not None and final_usage is not None:
                self._record_usage(request, component_chars, final_usage, ok)

    def _record_usage(
        self,
        request: ChatRequest,
        component_chars: dict[str, int],
        usage: Usage,
        ok: bool,
    ) -> None:
        """Enqueue one call sample. Off the hot path; never raises."""
        try:
            import time as _time

            from local_operator.analytics import CallSnapshot, record_call

            context_tokens = usage.context_tokens
            if not context_tokens:
                # Providers that omit an explicit context size: reconstruct the
                # full input the same way the wire clients normalise
                # ``context_tokens`` (clients.py) — input plus BOTH cache
                # halves — so the component split has the right denominator and
                # the headline total is not short by the cache-write volume on
                # a cache-writing provider (review A2). Cache-inclusive
                # providers report ``context_tokens`` directly, so this fallback
                # only fires when nothing was reported at all.
                context_tokens = (
                    usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
                )
            # Cost is NOT priced here (review C1). The snapshot carries the
            # provider, model id, and every token count, which is everything
            # ``cost_for_usage`` needs — so the pricing (including the
            # ``resolve_model_info`` lookup, which can block for seconds on a
            # COLD memo: a TTL rollover, an lru_cache eviction, or a subagent on
            # a registry-unknown model override) runs on the recorder's
            # background thread, next to the SQLite write, and never on the event
            # loop this turn is unwinding on. The same hazard ``subagent_panel``
            # deliberately takes off-thread. It is still the SAME
            # ``cost_for_usage`` the status band uses, so the analytics dollar
            # total cannot disagree with the live band.
            # Serving identity comes off the usage event the failover layer
            # stamped from the on-the-wire request. Falling back to
            # ``request.model`` would reintroduce the bug this exists to
            # close: after a primary→fallback walk the original ChatRequest
            # still names the session primary, so every Grok call was stored
            # as anthropic and priced at Opus rates. Isolated/naming calls
            # disable ``route_state``, so that pin is not an honest source
            # either. Unstamped usage (a test drain, a client that never
            # went through failover) still has ``request.model``.
            serving_provider = getattr(usage, "provider", None) or request.model.provider
            serving_model = getattr(usage, "model_id", None) or request.model.model_id
            from local_operator.providers.registry import credential_provider_id

            # Login flavours (``xai-oauth``, ``openai-device``) are the same
            # billable provider as their storage id. Canonicalize at record
            # time so By-provider does not split one vendor into two rows.
            serving_provider = credential_provider_id(serving_provider)
            record_call(
                CallSnapshot(
                    ts_ms=int(_time.time() * 1000),
                    session_id=self._session_id or "",
                    provider=serving_provider,
                    model_id=serving_model,
                    input_tokens=int(usage.input_tokens),
                    output_tokens=int(usage.output_tokens),
                    cache_read_tokens=int(usage.cache_read_tokens),
                    cache_write_tokens=int(usage.cache_write_tokens),
                    reasoning_tokens=int(getattr(usage, "reasoning_tokens", 0)),
                    context_tokens=int(context_tokens or 0),
                    component_chars=component_chars,
                    ok=ok,
                    usd_cost=getattr(usage, "usd_cost", None),
                )
            )
        except Exception:  # noqa: BLE001 — recording is best-effort
            logger.debug("analytics: usage record failed", exc_info=True)

    async def close(self) -> None:
        await self._http.aclose()
        if self._usage_cache is not None:
            try:
                self._usage_cache.close()
            except Exception:  # noqa: BLE001 — teardown, never fatal
                self._usage_cache = None
            else:
                self._usage_cache = None


def create_stream_fn(
    auth_store: AuthStore,
    settings: Mapping[str, Any] | None = None,
    *,
    session_id: str | None = None,
    cache_lineage_id: str | None = None,
) -> SessionStreamFn:
    """Build the ``LoopConfig.stream_fn`` for a session.

    Resolves the API key through ``auth_store`` (7-step cascade + OAuth
    refresh), picks the wire client from the request's ``ModelSpec``, and
    wraps the call in credential-rotation + model-fallback failover.

    ``session_id`` rides into the failover layer so the auth store keeps
    credential selection STICKY per session; without it the store round-robins
    on every resolve and multi-credential providers alternate accounts
    mid-conversation (cold cache prefix, alternating identity headers).

    ``cache_lineage_id`` overrides ONLY the provider cache key, defaulting to
    ``session_id``. A ``/fork`` passes its parent's id so the branch keeps the
    warm prefix it inherited byte-for-byte. It is a separate parameter rather
    than a reused ``session_id`` because the two govern different things:
    credential stickiness must stay scoped to the real session (a fork sharing
    a pinned credential row with its parent would be a genuine bug), while the
    cache key is a routing hint whose whole purpose is to follow an identical
    prefix.
    """
    return SessionStreamFn(auth_store, settings, session_id, cache_lineage_id)


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

    When ``usage`` carries a provider-reported dollar amount
    (``usd_cost``, e.g. OpenRouter's ``usage.cost``), that is returned verbatim
    and the token arithmetic is skipped entirely. The provider already applied
    per-route pricing, reasoning-token splits, cache discounts and any overrides
    that a single flat table price cannot express, so a reconstruction here can
    only be wronger than the number the provider printed on the bill.

    The caller is responsible for deciding whether ``model_info`` is priced at
    all; this returns 0.0 for a zero-priced model, which is arithmetically true
    and is exactly why a UI must not render it blindly.
    """
    reported = _usage_cost(usage)
    if reported is not None:
        return reported
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


def _usage_cost(usage: Any) -> float | None:
    """The provider-reported dollar cost on a ``Usage``, or ``None`` when absent.

    Duck-typed for the same reason as :func:`_usage_field`: ``usage`` can be a
    ``Usage`` model or a rehydrated mapping. ``None`` is "not reported" — a
    caller must not collapse it into ``0.0`` ("billed as free"). Coerced and
    floored so a malformed report (negative, non-numeric) degrades to the
    estimate instead of aborting the pricing path.
    """
    value = (
        usage.get("usd_cost") if isinstance(usage, Mapping) else getattr(usage, "usd_cost", None)
    )
    if value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    # Require finiteness as well as a non-negative sign. ``inf`` is wire-reachable
    # (``json.loads`` accepts the non-standard ``Infinity``/``NaN`` literals by
    # default), passes a bare ``>= 0`` guard, and — because ``inf + x == inf`` —
    # would permanently pin every summed turn/session/child total at infinity with
    # no recovery. Non-finite falls back to the estimate, exactly like negatives.
    return cost if (math.isfinite(cost) and cost >= 0) else None


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

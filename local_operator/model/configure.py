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

from pydantic import BaseModel, SecretStr

from local_operator.harness.types import AbortSignal, ChatRequest, ModelSpec, StreamEvent
from local_operator.model.catalogue import DEFAULT_TTL_S
from local_operator.model.registry import (
    ModelInfo,
    anthropic_default_model_info,
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
}

# Sensible ModelSpec fallbacks when the legacy registry knows nothing.
UNKNOWN_CONTEXT_WINDOW = 128_000
UNKNOWN_MAX_OUTPUT = 8_192

#: Per-provider fallback templates for an id the shipped registry does not carry.
#: Only providers whose whole FAMILY shares a floor belong here: Anthropic's
#: listing names models without describing them, so without a template an id the
#: provider confirms exists still resolves to the global 128k/8192/no-cache
#: unknown — numbers no Claude has ever had. A provider absent from this map keeps
#: the existing behaviour and falls through to ``unknown_model_info``.
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
    """
    return not _has_real_window(info) or not (info.input_price or info.output_price)


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
    """True when ``secret`` was picked up out of an OAuth-named env variable.

    The callable ``env_keys`` resolvers can hand back either kind of credential —
    ``_anthropic_env_key`` prefers ``ANTHROPIC_OAUTH_TOKEN`` over
    ``ANTHROPIC_API_KEY`` — and return only the VALUE, so the caller cannot tell
    which it got. Getting that wrong is not cosmetic: Anthropic rejects an OAuth
    token sent as ``x-api-key`` with a 401, which is exactly the "model cannot be
    described" outcome this whole path exists to avoid.

    Matching on the variable NAME keeps this a general rule instead of a second
    place that knows about Anthropic specifically, and it runs at most once per
    model id per TTL bucket.
    """
    return any(value == secret and "OAUTH" in name.upper() for name, value in os.environ.items())


def _catalogue_credential(provider: str) -> tuple[str, bool]:
    """``(secret, is_oauth)`` for a listing call, preferring an explicit key.

    The flag matters and is not cosmetic: Anthropic authenticates an API key with
    ``x-api-key`` and an OAuth token with ``Authorization: Bearer``, so sending the
    wrong header shape is a 401 and therefore a model that cannot be described.

    Order is env, then the credential file, then the OAuth store — an explicit
    variable is the operator overriding config for one run, which is the same
    precedence every other key reader in the repo uses. That order was inverted in
    practice for Anthropic: the env leg could not see a callable ``env_keys``, so
    a stored OAuth row beat an explicitly exported ``ANTHROPIC_API_KEY``.
    """
    key = _catalogue_api_key(provider)
    if key:
        return key, _env_secret_is_oauth(key)
    return _oauth_listing_token(provider)


def _oauth_listing_token(provider: str) -> tuple[str, bool]:
    """The newest stored token for ``provider``, or ``("", False)``.

    Best-effort by construction: an unreadable store, a missing table or a row
    without a token all mean "no listing", never an exception. Opened and closed
    per call because this is reached only when the registry could not describe a
    model, which is once per model id per TTL bucket per process.
    """
    store = None
    try:
        from local_operator.providers.auth_store import AuthStore
        from local_operator.providers.registry import get_provider_definition

        definition = get_provider_definition(provider)
        storage = (definition.store_credentials_as or provider) if definition else provider
        store = AuthStore()
        rows = store.list_credentials(provider=storage)
        for row in reversed(rows):
            token = str(row.data.get("access") or "")
            if token:
                return token, row.credential_type == "oauth"
    except Exception as exc:  # noqa: BLE001 - metadata is never worth a failed start
        logger.debug("could not read a stored %s token for the listing: %s", provider, exc)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001 - closing a broken handle is not fatal
                pass
    return "", False


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


def _info_from_discovery(provider: str, model_name: str, fallback: ModelInfo) -> ModelInfo:
    """Fill ``fallback``'s gaps from the provider's own live model listing.

    Never raises, and never returns worse data than it was given: every field is
    taken only when the listing actually has it. ``local_operator.model.discovery``
    has already merged the listing over the static registry and applied the rules
    that make a listing trustworthy — a zero price is unknown rather than free, a
    ``max_tokens`` of exactly 4096 is a lying OpenAI-compat default, capabilities
    are OR-ed so a terse listing cannot downgrade a model — so this function is
    only the projection of that answer onto the legacy ``ModelInfo`` shape.

    Imported lazily. The discovery module pulls httpx and the provider registry,
    and this branch is only reached for a model the registry does not describe;
    putting that on the import path would cost every CLI invocation.
    """
    try:
        from local_operator.model.discovery import available_models

        secret, is_oauth = _catalogue_credential(provider)
        rows, status = available_models(
            provider,
            api_key=secret or None,
            is_oauth=is_oauth,
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
    info.supports_images = info.supports_images or row.supports_images
    info.supports_prompt_cache = info.supports_prompt_cache or row.supports_prompt_cache
    return info


def _registry_fallback(provider: str, model_id: str) -> ModelInfo:
    """What the registry can say about ``model_id``, or the best template for it.

    The global ``unknown_model_info`` is the right answer only for a provider we
    know nothing structural about. For Anthropic it is actively wrong: the listing
    confirms an id exists but describes NOTHING about it (ids and display names
    only), so an unshipped Claude id would keep 128k/8192/no-cache — numbers no
    Claude generation has ever had. The per-provider template carries the family
    floor instead, in the same shape ``openrouter_default_model_info`` and
    ``radient_default_model_info`` already use for the aggregators.

    The id and name are overwritten so a template shared by every unknown id of a
    provider cannot leak its placeholder identity ("Anthropic Claude") into a band
    that is meant to name the model the session is running.
    """
    try:
        info = get_model_info(provider, model_id)
    except (ValueError, KeyError):
        info = None
    if info is not None and info is not unknown_model_info:
        return info

    template = _UNKNOWN_MODEL_TEMPLATES.get(provider)
    if template is not None:
        return template.model_copy(deep=True, update={"id": model_id, "name": model_id})
    if info is not None:
        return info
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
    info = _registry_fallback(canonical, model_id)
    if _needs_enrichment(info):
        # EVERY provider, not just the aggregators. The gate used to be
        # `canonical in LISTING_PROVIDERS`, which left a hole that the model picker
        # turned into a routine path: the picker offers whatever a provider's live
        # listing returns, so a user can now select `anthropic/claude-opus-5` — a
        # real model, absent from the shipped registry — and the session would run
        # with `context_window = -1`. Compaction thresholds derive from the window,
        # so that is not a cosmetic gap: compaction silently never fires and the
        # turn eventually 400s on the provider's real limit.
        #
        # Only reached when the registry is missing the window or BOTH prices, so a
        # fully described model still costs nothing: no HTTP call, no cache read,
        # no listing scan.
        info = _info_from_discovery(canonical, model_id, info)
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

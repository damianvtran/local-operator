"""Live per-provider model discovery, layered OVER the static registry.

``model/registry.py`` only knows the models that were current when it was last
edited, so a user who logs into Anthropic today cannot reach a model released
since: the picker never lists it and there is no way to learn its id in order to
type it. This module asks each provider for its own model list and folds the
answer into the registry.

Three properties are load-bearing, and each exists because the obvious version
of this feature broke in exactly that way:

- **The listing never subtracts.** Ids are UNIONed and every numeric field falls
  back to the registry, because a listing is only as rich as the provider chose
  to make it and several are far poorer than the bundled data. A lean
  OpenAI-compatible gateway returns nothing but an id; replacing a registry row
  with that is how a session ends up with ``context_window = -1`` and auto
  compaction that never fires. Where a listing IS rich it wins outright, which is
  the other half of the same rule: Anthropic's ``/v1/models`` reports
  ``max_input_tokens``, and that is the only place a model released after this
  package can state its real 1M window.
- **Failure and emptiness are different answers.** A transport error yields
  ``None`` ("keep what we had") and a successful listing with no models yields
  ``[]`` ("this provider really has nothing"). Collapsing them turns a flaky
  network into a picker that claims the provider has no models at all.
- **Nothing here raises.** A provider that is down, misconfigured or unreachable
  is a status annotation in the UI, not a failed startup.
  :func:`available_models` degrades to the registry instead of propagating.

Results are cached on disk through
:func:`local_operator.model.catalogue.cached_listing`, so the picker opens at
disk speed and a listing outage is invisible for as long as the cache holds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import httpx

from local_operator.model.catalogue import DEFAULT_TTL_S, cached_listing, invalidate
from local_operator.model.registry import ModelInfo, static_models
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    WireFormat,
    get_provider_definition,
)

logger = logging.getLogger("local_operator.model.discovery")

#: Ceiling on ONE listing, pagination included. A model picker and session start
#: both call this synchronously while the user waits, so an unreachable -- or
#: merely slow -- host must fail in seconds rather than hang on the default socket
#: timeout. ``_fetch_gemini`` spends it as a deadline across its pages rather than
#: per request, because 25 pages x this value is not "seconds".
DEFAULT_TIMEOUT_S = 10.0

#: Anthropic pins its wire format with a dated header and rejects requests that
#: omit it. Duplicated from the wire client rather than imported, so listing a
#: provider's models does not drag the whole chat-client module into startup.
ANTHROPIC_VERSION = "2023-06-01"

#: Claude Pro/Max OAuth tokens are only accepted alongside this beta opt-in --
#: the same pairing the chat client and the token refresh already use.
ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"

#: ChatGPT subscription OAuth cannot call the public ``api.openai.com/v1/models``
#: endpoint; the same bearer that serves Codex inference lists its models here.
#: ``client_version=0.0.0`` is the OpenAI Codex source-build version and is
#: deliberately not local-operator's unrelated package version. The backend uses
#: this value for compatibility filtering.
CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
CODEX_MODELS_CLIENT_VERSION = "0.0.0"

#: Gemini's ``v1`` surface omits ``inputTokenLimit`` on newer models, and
#: ``v1beta`` is the version the generation client already talks to, so the two
#: agree about which models exist.
GEMINI_API_VERSION = "v1beta"

#: Hard stop on ``nextPageToken`` following. A provider bug that returns a fresh
#: token forever would otherwise turn opening the picker into an unbounded run of
#: requests; 25 pages of 100 is far more models than any provider serves.
GEMINI_MAX_PAGES = 25

#: OpenAI-compatible gateways very commonly report ``max_tokens: 4096`` as a
#: hardcoded default rather than a real limit. Believing it silently caps a
#: 32k-output model at 4k, so an exact 4096 loses to a larger bundled value.
LYING_MAX_TOKENS = 4096

#: Stamped into every cached listing document and required when one is read back.
#: Bump it whenever a transport starts capturing a FIELD, or a MEANING, that the
#: previous writer could not express.
#:
#: Version 2 is ``_fetch_anthropic`` reading ``max_input_tokens``, ``max_tokens``
#: and ``capabilities.image_input``, and ``supports_images`` becoming three-state
#: (``null`` = the listing did not say, distinct from ``false``). Without the
#: stamp, a document written by version 1 has a perfectly valid shape full of
#: zeros, is served as a fresh cache hit for the rest of its 24h TTL, and the
#: upgrade that fixed the numbers appears to have done nothing — which is
#: precisely the state the reported ``1.8%/200k`` install was left in.
#: Stamped PER TRANSPORT, because only one of them changed. A single global
#: number invalidated every provider's cache on upgrade — including transports
#: whose payload was already correct — and for an aggregator with no static rows
#: to fall back on, the replacement answer was an EMPTY model list.
LISTING_CAPTURE_VERSIONS: dict[str, int] = {"anthropic": 2}
#: What a transport not named above is stamped with. Version 1 is the original
#: shape; a transport only earns a bump when its own reader starts needing a
#: field its writer did not record.
LISTING_CAPTURE_DEFAULT = 1


def listing_capture_version(provider_id: str) -> int:
    """Capture stamp this provider's cached listing is written and read with."""
    return LISTING_CAPTURE_VERSIONS.get(provider_id, LISTING_CAPTURE_DEFAULT)


#: Providers whose MODEL LISTING is public even though inference is not.
#:
#: Distinct from the registry's ``allows_missing_api_key``, which describes the
#: inference transport (a local server that needs no bearer at all). An
#: aggregator is the opposite shape: the catalogue is a public marketing surface
#: while every completion needs a key. Conflating the two costs the most valuable
#: listing in the tree — OpenRouter serves ~340 models and is the one provider
#: whose catalogue cannot be approximated from the registry, since its entry there
#: is a single placeholder describing the router itself.
PUBLIC_LISTING_PROVIDERS = frozenset({"openrouter", "radient"})

#: What :func:`available_models` managed to do, for the UI to annotate:
#: ``ok`` fetched live now, ``cached`` served a stored document, ``static``
#: registry only, ``unauthenticated`` needs a credential it was not given,
#: ``empty`` the provider answered and listed nothing.
ListingStatus = Literal[
    "ok",
    "cached",
    "static",
    "unavailable",
    "unauthenticated",
    "empty",
]


@dataclasses.dataclass(frozen=True)
class DiscoveredModel:
    """One model as the UI needs it: from a listing, the registry, or both.

    Every numeric field uses ``0`` for "unknown" rather than ``None`` or ``-1``.
    Sentinel negatives were the original source of the ``context_window = -1``
    bug: they survive arithmetic and produce a plausible-looking compaction
    threshold, whereas a zero is falsy and so is caught by the merge's fallback
    at the first read.

    ``supports_images`` is the exception, and deliberately THREE-state: ``None``
    means the listing said nothing, which is a different answer from ``False``
    ("this model does not accept images"). A boolean cannot express the
    difference, and collapsing them costs the distinction in the direction that
    matters: the merge falls back to the registry for an unknown, so an explicit
    ``false`` on the wire would be overruled by a hand-transcribed ``True`` and a
    text-only model would keep advertising vision forever. The provider is the
    authority on its own capabilities; only a silent listing defers to us.

    ``supports_prompt_cache`` stays a plain boolean because NO listing in the tree
    states it: Anthropic's ``capabilities`` object has no prompt-caching key at
    all, and the OpenAI-compatible wires only imply it through a priced cache-read
    leg. There is no explicit ``false`` to preserve, so there is nothing for a
    third state to carry.
    """

    id: str
    name: str = ""
    context_window: int = 0
    max_tokens: int = 0
    input_price: float = 0.0
    output_price: float = 0.0
    cache_read_price: float = 0.0
    supports_images: bool | None = None
    supports_prompt_cache: bool = False


class _ListingUnavailable(RuntimeError):
    """Raised inside the ``cached_listing`` thunk when a live listing failed.

    ``cached_listing`` chooses between a stale document and ``None`` by catching
    an exception from the thunk, so a failed transport has to be re-raised rather
    than returned: handing back an empty payload would overwrite a good cache
    with nothing and then serve that for a full day.
    """


@dataclasses.dataclass(frozen=True)
class _FetchContext:
    """Everything a transport needs, so all transports share one signature."""

    provider_id: str
    base_url: str
    api_key: str | None
    is_oauth: bool
    account_id: str | None
    client: httpx.Client
    timeout: float


# -- scalar coercion ---------------------------------------------------------


def _positive_int(value: object) -> int:
    """``value`` as a usable token count, or ``0`` when it is not one.

    Listings put ints, floats, decimal strings, ``null`` and ``-1`` in these
    fields. Everything that is not a finite positive number collapses to ``0`` so
    that the single ``or static`` fallback in the merge covers all of them at
    once; the registry's own placeholder rows carry ``-1`` and land here too.
    """
    if isinstance(value, bool):
        # bool is an int subclass, and a ``True`` context window is nonsense.
        return 0
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return 0
    else:
        return 0
    if not math.isfinite(number) or number <= 0:
        return 0
    return int(number)


def _positive_float(value: object) -> float:
    """``value`` as a usable price, or ``0.0`` when it is not one.

    Zero is deliberately indistinguishable from missing: the cost display prints
    the literal word ``free`` when both legs are zero, so a listing that simply
    omits pricing must not be able to claim a paid model costs nothing.
    """
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return 0.0
    else:
        return 0.0
    if not math.isfinite(number) or number <= 0:
        return 0.0
    return number


def _per_million(value: object) -> float:
    """A per-TOKEN price scaled to the per-million unit used everywhere else.

    OpenRouter-style gateways quote ``"0.000003"`` per token, while the registry
    and the whole cost display work in dollars per million tokens. Mixing the two
    misstates a session's cost by six orders of magnitude.
    """
    return _positive_float(value) * 1_000_000.0


def _first_str(*values: object) -> str:
    """The first non-blank string among ``values``, else ``""``.

    Display names hide behind different keys per provider (``name``,
    ``display_name``, ``displayName``). A blank string counts as absent, so a
    provider that sends ``""`` does not beat the registry's real name.
    """
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_positive_int(*values: object) -> int:
    """The first usable token count among ``values``, else ``0``."""
    for value in values:
        number = _positive_int(value)
        if number:
            return number
    return 0


def _stated_bool(value: object) -> bool | None:
    """``value`` as a capability the listing STATED, or ``None`` when it did not.

    Only a real boolean counts as a statement. A listing that sends ``null``, omits
    the key, or answers with an object where a flag belongs has not said anything,
    and the difference matters: a stated ``False`` overrules the registry while an
    absent one defers to it. ``bool(value)`` cannot express that, and it also reads
    ``{}`` as False, which would turn "the key exists but is empty" into a denial.
    """
    return value if isinstance(value, bool) else None


def _mapping(value: object) -> Mapping[str, object]:
    """``value`` when it is a mapping, else an empty one.

    Nested listing objects (``pricing``, ``top_provider``, ``architecture``) are
    optional and are occasionally ``null``, so without this every read of them
    would need its own guard.
    """
    if isinstance(value, Mapping):
        return value
    return {}


# -- HTTP --------------------------------------------------------------------


def _get_json(
    ctx: _FetchContext,
    url: str,
    *,
    headers: Mapping[str, str],
    # `str | int` rather than `object`: httpx's own parameter type is a mapping of
    # primitives, and widening it here only moved the mismatch to the call site.
    # Every caller passes strings and page sizes, so nothing is lost.
    params: Mapping[str, str | int],
    timeout: float | None = None,
) -> object | None:
    """One GET, decoded, or ``None`` for any answer we cannot use.

    A non-2xx status returns ``None`` rather than raising, because the caller's
    only distinction is failure-versus-listing: a 401 and a 500 both mean "keep
    the registry", and the status is useful solely in the debug log.

    ``timeout`` overrides the context's per-request ceiling, which is what lets a
    paginating transport spend ONE ceiling across several requests instead of
    granting each hop a fresh one. Passed here rather than by rebuilding the
    context, so the single-request transports allocate nothing extra.
    """
    ceiling = ctx.timeout if timeout is None else timeout
    response = ctx.client.get(url, headers=dict(headers), params=dict(params), timeout=ceiling)
    status = int(getattr(response, "status_code", 0))
    if not 200 <= status < 300:
        logger.debug("%s model listing returned HTTP %s for %s", ctx.provider_id, status, url)
        return None
    return response.json()


def _entry_list(body: object, *keys: str) -> list[Mapping[str, object]] | None:
    """The model array inside ``body``, or ``None`` when there is not one.

    ``None`` here has to mean "malformed" rather than "empty": an unrecognised
    envelope must degrade to the registry, whereas a present-but-empty array is a
    real answer the caller is required to preserve. Non-mapping members are
    dropped instead of failing the page, so one junk entry cannot hide a
    provider's entire catalogue.
    """
    if isinstance(body, list):
        return [entry for entry in body if isinstance(entry, Mapping)]
    if isinstance(body, Mapping):
        for key in keys:
            value = body.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, Mapping)]
    return None


# -- transports --------------------------------------------------------------


def _has_image_input(architecture: Mapping[str, object]) -> bool | None:
    """Image support from either OpenRouter modality encoding, or ``None``.

    The current listing exposes ``input_modalities: ["text", "image"]``; the
    older one packs the same fact into ``modality: "text+image->text"``. Both are
    still in the wild across gateways that mirror OpenRouter's schema, and
    reading only one silently marks vision models as text-only.

    A gateway that describes modalities and does not list ``image`` has SAID the
    model is text-only, and that answer beats the registry. One that describes no
    modalities at all — every lean OpenAI-compatible endpoint, which sends an id
    and little else — has said nothing, and returning ``False`` for it would
    downgrade every bundled vision model the moment such a gateway is listed.
    """
    modalities = architecture.get("input_modalities")
    if isinstance(modalities, (list, tuple)):
        return any(isinstance(item, str) and item.strip().lower() == "image" for item in modalities)
    modality = architecture.get("modality")
    if isinstance(modality, str):
        # Only the left of the arrow is INPUT. A model that GENERATES images is
        # not a model you can send an image to.
        return "image" in modality.split("->")[0].lower()
    return None


def _row_from_openai_entry(entry: Mapping[str, object]) -> DiscoveredModel | None:
    """One OpenAI-compatible listing entry, or ``None`` when it has no id.

    Nothing beyond ``id`` is standardised: OpenRouter nests the real limits under
    ``top_provider`` and the modalities under ``architecture``, while leaner
    gateways send ``id`` alone. Each field therefore reads a list of candidate
    keys and falls through to ``0``/``""`` so the merge can supply the bundled
    value.
    """
    model_id = _first_str(entry.get("id"))
    if not model_id:
        # An entry with no addressable id cannot be selected, and admitting it
        # would put a blank row in the picker.
        return None

    pricing = _mapping(entry.get("pricing"))
    top_provider = _mapping(entry.get("top_provider"))
    architecture = _mapping(entry.get("architecture"))

    cache_read_price = _per_million(pricing.get("input_cache_read"))
    return DiscoveredModel(
        id=model_id,
        name=_first_str(entry.get("name"), entry.get("display_name")),
        context_window=_first_positive_int(
            entry.get("context_length"),
            entry.get("context_window"),
            entry.get("max_context_length"),
            top_provider.get("context_length"),
        ),
        max_tokens=_first_positive_int(
            entry.get("max_tokens"),
            entry.get("max_output_tokens"),
            entry.get("max_completion_tokens"),
            top_provider.get("max_completion_tokens"),
        ),
        input_price=_per_million(pricing.get("prompt")),
        output_price=_per_million(pricing.get("completion")),
        cache_read_price=cache_read_price,
        supports_images=_has_image_input(architecture),
        # A priced cache-read leg is the only machine-readable evidence of prompt
        # caching in these listings; there is no capability flag for it.
        supports_prompt_cache=cache_read_price > 0,
    )


def _row_from_codex_entry(entry: Mapping[str, object]) -> DiscoveredModel | None:
    """One picker-visible model from ChatGPT's Codex catalogue.

    This endpoint is richer than OpenAI's public ``/v1/models``: it carries the
    context window and input modalities used by the connected subscription. It
    does not quote token prices because a ChatGPT plan is quota-billed rather
    than per-token billed, so prices stay unknown unless the static registry has
    a published value for the same id.
    """
    if _first_str(entry.get("visibility")).lower() != "list":
        return None
    model_id = _first_str(entry.get("slug"))
    if not model_id:
        return None
    modalities = entry.get("input_modalities")
    supports_images = (
        any(
            isinstance(modality, str) and modality.strip().lower() == "image"
            for modality in modalities
        )
        if isinstance(modalities, (list, tuple))
        else None
    )
    return DiscoveredModel(
        id=model_id,
        name=_first_str(entry.get("display_name")),
        context_window=_first_positive_int(
            entry.get("context_window"),
            entry.get("max_context_window"),
        ),
        supports_images=supports_images,
    )


def _fetch_openai_compat(ctx: _FetchContext) -> list[DiscoveredModel] | None:
    """List an OpenAI-shaped provider, including ChatGPT subscription OAuth."""
    headers = {"Accept": "application/json"}
    if ctx.api_key:
        headers["Authorization"] = f"Bearer {ctx.api_key}"
    if ctx.provider_id == "openai" and ctx.is_oauth:
        # ChatGPT OAuth tokens are rejected by the public Models API. The Codex
        # endpoint is the model authority for the subscription-backed inference
        # path this same token uses.
        headers.update(
            {
                "originator": "local-operator",
                "User-Agent": "local-operator",
            }
        )
        if ctx.account_id:
            headers["ChatGPT-Account-Id"] = ctx.account_id
        body = _get_json(
            ctx,
            CODEX_MODELS_URL,
            headers=headers,
            params={"client_version": CODEX_MODELS_CLIENT_VERSION},
        )
        entries = _entry_list(body, "models") if body is not None else None
        if entries is None:
            return None
        rows = (_row_from_codex_entry(entry) for entry in entries)
        return [row for row in rows if row is not None]

    body = _get_json(ctx, f"{ctx.base_url}/models", headers=headers, params={})
    if body is None:
        return None
    entries = _entry_list(body, "data", "models")
    if entries is None:
        return None
    rows = (_row_from_openai_entry(entry) for entry in entries)
    return [row for row in rows if row is not None]


def _anthropic_models_url(base_url: str) -> str:
    """``/v1/models`` under ``base_url``, without doubling an existing ``/v1``.

    The registry stores Anthropic's base as the bare host because the wire client
    appends ``/v1`` itself, but a user override or a proxy may already include
    it, and ``/v1/v1/models`` 404s.
    """
    if base_url.endswith("/v1"):
        return f"{base_url}/models"
    return f"{base_url}/v1/models"


def _capability_supported(capabilities: Mapping[str, object], name: str) -> bool | None:
    """What Anthropic's ``capabilities`` object says about ``name``, or ``None``.

    Each capability is an OBJECT with its own ``supported`` boolean rather than a
    bare flag (``{"image_input": {"supported": true}}``), because several of them
    carry sub-variants alongside it (``thinking`` lists ``adaptive``/``enabled``,
    ``effort`` lists five tiers). Reading the object's truthiness instead would
    report every listed capability as supported, including the ones explicitly
    marked ``false``.

    All three answers are distinct and all three occur on this wire: the live
    listing marks ``effort.xhigh`` supported on Opus 4.8 and NOT supported on
    Sonnet 4.6, while an older API version omits ``capabilities`` entirely. A
    stated ``false`` overrules the registry; an absent one lets the registry
    answer, so a terse wire cannot downgrade a vision model to text-only.
    """
    return _stated_bool(_mapping(capabilities.get(name)).get("supported"))


def _fetch_anthropic(ctx: _FetchContext) -> list[DiscoveredModel] | None:
    """Anthropic's listing: ids, display names, limits and capabilities.

    The response carries ``max_input_tokens`` (the context window),
    ``max_tokens`` (the output cap) and a ``capabilities`` object per model —
    verified against ``api.anthropic.com/v1/models`` on 2026-08-07, which reported
    1,000,000 / 128,000 for ``claude-opus-5`` while the shipped registry's family
    floor said 200,000. Reading them here is the only way a model released after a
    release of this package gets its real window, and the window is what the
    compaction threshold is derived from.

    Prices are still absent from this listing and are NOT invented: a zero price
    means "unknown" downstream, and the merge restores whatever the registry knows.
    The limits are equally optional — a proxy, an older API version or a future
    schema change may omit them — so each field falls through to ``0`` and lets
    :func:`merge_models` supply the bundled number rather than zeroing the one the
    session runs on.
    """
    headers = {"anthropic-version": ANTHROPIC_VERSION, "Accept": "application/json"}
    if ctx.is_oauth:
        # An OAuth access token sent as ``x-api-key`` is rejected with a 401, and
        # an API key sent as a bearer likewise: the two schemes are not
        # interchangeable, so the caller's ``is_oauth`` has to decide.
        headers["Authorization"] = f"Bearer {ctx.api_key or ''}"
        headers["anthropic-beta"] = ANTHROPIC_OAUTH_BETA
    elif ctx.api_key:
        headers["x-api-key"] = ctx.api_key

    body = _get_json(
        ctx,
        _anthropic_models_url(ctx.base_url),
        headers=headers,
        # The default page is 20 models, which already truncates the catalogue.
        # 1000 is the documented maximum and fits every release in one page, so
        # this endpoint needs no pagination loop.
        params={"limit": 1000},
    )
    if body is None:
        return None
    entries = _entry_list(body, "data")
    if entries is None:
        return None

    rows: list[DiscoveredModel] = []
    for entry in entries:
        model_id = _first_str(entry.get("id"))
        if not model_id:
            continue
        capabilities = _mapping(entry.get("capabilities"))
        rows.append(
            DiscoveredModel(
                id=model_id,
                name=_first_str(entry.get("display_name")),
                context_window=_positive_int(entry.get("max_input_tokens")),
                max_tokens=_positive_int(entry.get("max_tokens")),
                supports_images=_capability_supported(capabilities, "image_input"),
            )
        )
    return rows


def _row_from_gemini_entry(entry: Mapping[str, object]) -> DiscoveredModel | None:
    """One Gemini entry, or ``None`` when it is not a chat model.

    Ids arrive as the resource path ``models/gemini-2.5-pro`` while every other
    part of the system uses the bare id, so the prefix is stripped here rather
    than at each call site.

    The capability filter is deliberately strict: an entry that does not
    advertise ``generateContent`` is dropped, including one that omits the field.
    This listing is dominated by embedding and token-counting models, and
    offering one in the picker yields a model that 400s on every message -- a
    worse outcome than omitting a model whose capabilities the API declined to
    state.
    """
    resource = _first_str(entry.get("name"))
    if not resource:
        return None
    model_id = resource.split("/", 1)[1] if resource.startswith("models/") else resource
    if not model_id:
        return None
    methods = entry.get("supportedGenerationMethods")
    if not isinstance(methods, (list, tuple)) or "generateContent" not in methods:
        return None
    return DiscoveredModel(
        id=model_id,
        name=_first_str(entry.get("displayName")),
        context_window=_positive_int(entry.get("inputTokenLimit")),
        max_tokens=_positive_int(entry.get("outputTokenLimit")),
    )


def _fetch_gemini(ctx: _FetchContext) -> list[DiscoveredModel] | None:
    """Google's paginated listing, filtered to models that can generate text."""
    if not ctx.api_key:
        # Gemini authenticates the listing with a query parameter, so a keyless
        # request is a guaranteed 403. Skipping it keeps the picker responsive for
        # a provider the user has not logged into.
        return None

    url = f"{ctx.base_url}/{GEMINI_API_VERSION}/models"
    rows: list[DiscoveredModel] = []
    seen: set[str] = set()
    page_token = ""
    # ``ctx.timeout`` is a PER-REQUEST ceiling, but this is the only transport that
    # issues more than one request: at the documented 10 s ceiling, a
    # slow-but-alive endpoint handing back a fresh ``nextPageToken`` each time
    # cost 25 x 10 s = 250 s for ONE `resolve_model_info()` -- measured with a stub
    # that always paginates. That call is on the synchronous session-start path and
    # on the TUI's `_cost_for`, so the whole run has to share one ceiling: the
    # deadline is taken once here and each page gets only what is left of it.
    deadline = time.monotonic() + ctx.timeout
    for _ in range(GEMINI_MAX_PAGES):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            # Same choice as a failed page below, for the same reason: the pages
            # that did arrive are not the catalogue, and passing them off as one
            # silently deletes every model on the pages that did not.
            logger.debug(
                "%s model listing exceeded its %.1fs budget after %d models",
                ctx.provider_id,
                ctx.timeout,
                len(rows),
            )
            return None
        params: dict[str, str | int] = {"key": ctx.api_key, "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        body = _get_json(
            ctx,
            url,
            headers={"Accept": "application/json"},
            params=params,
            timeout=remaining,
        )
        if body is None:
            # A failure part-way through pagination fails the WHOLE listing.
            # Returning the pages that did arrive would present a truncated
            # catalogue as authoritative, and the merge cannot tell the
            # difference: a model the user runs today would simply vanish.
            return None
        page = _mapping(body)
        entries = _entry_list(page, "models")
        if entries is None:
            return None
        for entry in entries:
            row = _row_from_gemini_entry(entry)
            if row is not None and row.id not in seen:
                seen.add(row.id)
                rows.append(row)
        next_token = _first_str(page.get("nextPageToken"))
        if not next_token or next_token == page_token:
            # An unchanged token means the server is not advancing. Continuing
            # would re-request the same page until the cap, spending 25 round
            # trips to collect one page of models.
            break
        page_token = next_token
    return rows


_Transport = Callable[[_FetchContext], list[DiscoveredModel] | None]

#: One transport per wire shape, because the wire -- not the vendor -- decides
#: how a listing is requested and parsed. Keying on ``wire`` rather than on
#: provider ids means a provider added to ``PROVIDER_REGISTRY`` later is
#: discoverable without editing this module, which is the failure mode a
#: hardcoded id list has: the newest provider is always the one that silently
#: gets no listing.
_WIRE_TRANSPORTS: dict[WireFormat, _Transport | None] = {
    # openai, openai-device, kimi, xai, xai-oauth, deepseek, mistral, ollama,
    # openrouter, radient and alibaba all expose OpenAI's ``/models``.
    "openai-compat": _fetch_openai_compat,
    # Anthropic's listing has its own envelope and its own auth scheme.
    "anthropic": _fetch_anthropic,
    # Gemini paginates and authenticates by query parameter.
    "google": _fetch_gemini,
    # The mock wire has no server behind it, so there is nothing to list and a
    # request would have nowhere to go; its registry rows are the whole truth.
    "mock": None,
}


def _build_transports() -> dict[str, _Transport | None]:
    """Provider id to transport, derived from the registry at import time.

    A provider with no ``base_url`` maps to ``None``: no host means no listing
    endpoint. That is the shape the credential-brokered providers take (Vertex,
    Bedrock and Azure, none of which exist in this tree today), so adding one
    later gets the correct "registry only" behaviour instead of a request to the
    empty string.
    """
    transports: dict[str, _Transport | None] = {}
    for definition in PROVIDER_REGISTRY:
        if not definition.base_url:
            transports[definition.id] = None
            continue
        transports[definition.id] = _WIRE_TRANSPORTS.get(definition.wire)
    return transports


_TRANSPORTS: dict[str, _Transport | None] = _build_transports()

#: Providers that cannot be listed at all, exposed so a UI can say so without
#: first attempting a request that is guaranteed to fail.
NO_LISTING_PROVIDERS: frozenset[str] = frozenset(
    provider_id for provider_id, transport in _TRANSPORTS.items() if transport is None
)


def _static_rows(provider_id: str) -> dict[str, ModelInfo]:
    """The registry's rows for ``provider_id``, resolving credential aliases.

    ``xai-oauth`` and ``openai-device`` are login flavours of ``xai`` and
    ``openai``, not separate catalogues, and ``store_credentials_as`` is the
    registry's own statement of that. Without following it they would come back
    with no static rows and lose every bundled price and window -- the exact
    regression this module is meant to prevent.
    """
    definition = get_provider_definition(provider_id)
    if definition is None:
        return {}
    return static_models(definition.store_credentials_as or definition.id)


def fetch_models(
    provider_id: str,
    *,
    api_key: str | None,
    is_oauth: bool = False,
    account_id: str | None = None,
    base_url: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[DiscoveredModel] | None:
    """``provider_id``'s live model list, or ``None`` when it cannot be had.

    ``None`` covers every failure -- unknown provider, no listing endpoint,
    transport error, non-2xx, unparseable body -- because callers treat all of
    them identically: keep the registry. ``[]`` means the provider answered and
    listed nothing, and the two must never be conflated. An empty list is a
    positive statement that the picker should show registry rows only, whereas
    ``None`` says the question went unanswered.

    Args:
        provider_id: Registry provider id, or a legacy alias.
        api_key: API key or OAuth access token; ``None`` for keyless hosts.
        is_oauth: ``api_key`` is an OAuth access token, which changes the header
            scheme on the Anthropic wire.
        account_id: ChatGPT account id paired with an OpenAI OAuth bearer.
        base_url: Overrides the registry base (proxies, local gateways).
        client: Reused HTTP client. When omitted, one is created and closed here;
            a caller listing several providers should pass one so the connection
            pool and TLS handshake are shared.
        timeout: Ceiling in seconds for the whole listing. Google's transport
            paginates and spends it as a single deadline; every other transport
            issues one request, so for them it is also the per-request ceiling.
    """
    definition = get_provider_definition(provider_id)
    if is_oauth and definition is not None and definition.store_credentials_as:
        definition = get_provider_definition(definition.store_credentials_as) or definition
    transport = _TRANSPORTS.get(definition.id) if definition is not None else None
    if definition is None or transport is None:
        return None
    resolved_base = (base_url or definition.base_url or "").rstrip("/")
    if not resolved_base:
        return None

    owned: httpx.Client | None = None
    try:
        active = client
        if active is None:
            owned = httpx.Client(timeout=timeout)
            active = owned
        return transport(
            _FetchContext(
                provider_id=definition.id,
                base_url=resolved_base,
                api_key=api_key,
                is_oauth=is_oauth,
                account_id=account_id,
                client=active,
                timeout=timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001 - every failure degrades identically
        # Deliberately broad: transports raise httpx errors, TLS errors, JSON
        # decode errors, and whatever a provider's next schema change produces.
        # All of them mean the same thing to the caller, and a model listing must
        # never be able to abort a session.
        logger.debug("could not list %s models: %s", provider_id, exc)
        return None
    finally:
        if owned is not None:
            # Only a client this function created. Closing the caller's would
            # break the next provider in a batch.
            owned.close()


def merge_models(
    static_rows: Mapping[str, ModelInfo],
    live: list[DiscoveredModel] | None,
) -> list[DiscoveredModel]:
    """Fold a listing into the registry so the result is never the poorer of the two.

    Live rows come first because providers list newest-first, which puts a model
    released this week where the user will actually see it -- the entire point of
    discovery. Registry-only ids follow, in the registry's curated order.
    """
    merged: list[DiscoveredModel] = []
    seen: set[str] = set()

    for row in live or []:
        if row.id in seen:
            # A listing that repeats an id (paginated gateways do) must not
            # produce two identical picker entries.
            continue
        seen.add(row.id)
        merged.append(_merge_one(row, static_rows.get(row.id)))

    for model_id, info in static_rows.items():
        if model_id in seen:
            continue
        seen.add(model_id)
        # Static ids are never pruned: a listing that omits an id we know works
        # -- a gateway filtering by entitlement, a page we failed to fetch --
        # would otherwise make a model the user is running today unreachable.
        merged.append(_from_static(model_id, info))

    return merged


def _from_static(model_id: str, info: ModelInfo) -> DiscoveredModel:
    """A registry row as a :class:`DiscoveredModel`, sentinels normalised.

    The registry spells "unknown" as ``-1`` in some rows and ``None`` in others;
    both become ``0`` here so downstream code has one unknown-marker to check
    instead of three.

    ``supports_images`` is carried across unchanged, including ``None``: a bundled
    row that never stated the capability has not denied it, and this row is what a
    registry-only model is resolved from. Collapsing it to ``False`` here would
    hand such a model a denial nothing ever wrote down.
    """
    return DiscoveredModel(
        id=model_id,
        name=info.name,
        context_window=_positive_int(info.context_window),
        max_tokens=_positive_int(info.max_tokens),
        input_price=_positive_float(info.input_price),
        output_price=_positive_float(info.output_price),
        cache_read_price=_positive_float(info.cache_reads_price),
        supports_images=_stated_bool(info.supports_images),
        supports_prompt_cache=bool(info.supports_prompt_cache),
    )


def _merge_one(row: DiscoveredModel, info: ModelInfo | None) -> DiscoveredModel:
    """One live row reconciled with its registry twin, field by field.

    The bias is uniform: the live answer wins only where it is actually the more
    informative one. Every exception below is a bug that happened when the naive
    "live overwrites static" version shipped.
    """
    if info is None:
        # Live-only: a model the registry has never heard of, which is exactly
        # the case this module exists to surface.
        return row

    live_max = _positive_int(row.max_tokens)
    static_max = _positive_int(info.max_tokens)
    if live_max == LYING_MAX_TOKENS and static_max > live_max:
        # OpenAI-compat gateways hardcode 4096 in their listing regardless of the
        # model, so an exact 4096 beside a larger bundled cap is read as a
        # default rather than a limit. Believing it truncates long outputs.
        max_tokens = static_max
    else:
        max_tokens = live_max or static_max

    return DiscoveredModel(
        id=row.id,
        name=_merge_name(row.name, info.name, row.id),
        # Only a positive live window beats the registry. A listing that omits the
        # field (a lean OpenAI-compatible gateway, or an Anthropic proxy on an API
        # version predating ``max_input_tokens``) must not zero out the number auto
        # compaction derives its threshold from.
        context_window=_positive_int(row.context_window) or _positive_int(info.context_window),
        max_tokens=max_tokens,
        # A zero or absent price means "unknown", never "free": the cost display
        # prints the literal word for a genuinely free model, so letting a silent
        # listing win here would advertise a paid model as free.
        input_price=_positive_float(row.input_price) or _positive_float(info.input_price),
        output_price=_positive_float(row.output_price) or _positive_float(info.output_price),
        cache_read_price=(
            _positive_float(row.cache_read_price) or _positive_float(info.cache_reads_price)
        ),
        # The provider decides its own capabilities WHEN IT SPEAKS: a stated
        # ``false`` is an answer, and OR-ing it against the registry made it
        # unreachable — every bundled Anthropic row carries
        # ``supports_images=True``, so a live ``image_input.supported: false``
        # merged back to True and a text-only model went on advertising vision.
        # A listing that says NOTHING still defers to the registry, which is what
        # keeps a terse wire from downgrading a vision model.
        supports_images=(
            row.supports_images
            if row.supports_images is not None
            else _stated_bool(info.supports_images)
        ),
        # OR, not three-state: no listing in the tree states prompt caching, so
        # there is no explicit denial to respect — only silence, and silence must
        # not disable ``cache_control`` on the most expensive models we ship.
        supports_prompt_cache=bool(row.supports_prompt_cache or info.supports_prompt_cache),
    )


def _merge_name(live_name: str, static_name: str, model_id: str) -> str:
    """The better display name of the two.

    Live wins normally -- that is how a renamed model gets its new label. Two
    exceptions: a blank live name is no name at all, and a live name equal to the
    id is the very common case of an endpoint echoing its key back, which must
    not replace a real bundled label like "Claude Sonnet 4" with the raw id.
    """
    candidate = live_name.strip()
    if not candidate:
        return static_name
    if candidate == model_id and static_name and static_name != model_id:
        return static_name
    return candidate


def _cache_key(provider_id: str, *, is_oauth: bool, account_id: str | None) -> str:
    """Cache document name for one provider listing and credential surface.

    ``.listing`` separates this shape from legacy raw ``list_models()`` payloads.
    OpenAI needs a second split: API keys list the public API catalogue, while
    ChatGPT OAuth lists the plan-specific Codex catalogue. The account id is
    hashed so switching plans cannot serve another account's availability from
    cache and the private identifier never reaches a filename.
    """
    if provider_id == "openai" and is_oauth:
        identity = hashlib.sha256((account_id or "unknown").encode()).hexdigest()[:12]
        return f"{provider_id}.codex.{identity}.listing"
    return f"{provider_id}.listing"


def _rows_from_payload(
    payload: Mapping[str, object] | None, expected_capture: int
) -> list[DiscoveredModel] | None:
    """Cached rows, or ``None`` when the document is absent, unrecognised or stale.

    An unrecognised document is treated as no document, so a change to the
    payload shape degrades to the registry rather than to a silently empty model
    list that looks like a provider with nothing to offer.

    A document from an older capture stamp (see :func:`listing_capture_version`)
    is rejected for a different reason: its SHAPE is fine and every field maps,
    so nothing else here could notice that the transport now reads a field the
    writer left at zero.

    Either way the caller drops the document and refetches IN THE SAME CALL. An
    earlier version deferred to "the next call", which was wrong for exactly the
    providers this matters to: the drop happens on a call already served from a
    fresh cache hit, so no fetch ran, and the answer fell back to the registry's
    static rows — of which an aggregator has none. The user saw an empty model
    list and had to invoke twice to get a catalogue.
    """
    if payload is None:
        return None
    # An ABSENT stamp is version 1, not version 0. Documents written before the
    # stamp existed carry no `capture` key at all, and reading that as 0 rejected
    # every one of them for every transport — including the aggregators the
    # per-transport map exists precisely to spare, whose registry has no static
    # rows to answer with. The unstamped shape IS the original shape.
    stamp = _positive_int(payload.get("capture")) or LISTING_CAPTURE_DEFAULT
    if stamp != expected_capture:
        return None
    entries = payload.get("models")
    if not isinstance(entries, list):
        return None
    rows: list[DiscoveredModel] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        model_id = _first_str(entry.get("id"))
        if not model_id:
            continue
        rows.append(
            DiscoveredModel(
                id=model_id,
                name=_first_str(entry.get("name")),
                context_window=_positive_int(entry.get("context_window")),
                max_tokens=_positive_int(entry.get("max_tokens")),
                input_price=_positive_float(entry.get("input_price")),
                output_price=_positive_float(entry.get("output_price")),
                cache_read_price=_positive_float(entry.get("cache_read_price")),
                # ``null`` in the document is the listing's silence, faithfully
                # stored by ``dataclasses.asdict``. Reading it as False would let
                # a cache round-trip turn "unstated" into a denial, so the same
                # model would resolve differently live than from disk.
                supports_images=_stated_bool(entry.get("supports_images")),
                supports_prompt_cache=bool(entry.get("supports_prompt_cache")),
            )
        )
    return rows


def available_models(
    provider_id: str,
    *,
    api_key: str | None,
    is_oauth: bool = False,
    account_id: str | None = None,
    base_url: str | None = None,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[list[DiscoveredModel], ListingStatus]:
    """Every model a UI should offer for ``provider_id``, plus how it was obtained.

    The result is always at least the registry, so a provider outage costs the
    user discovery of new models and nothing else. This never raises and never
    blocks longer than ``timeout``: it is called from the model picker and from
    session start while the user waits, and a fresh cache skips the request
    entirely.

    Returns:
        ``(models, status)``: ``"ok"`` fetched live; ``"cached"`` used a stored
        listing; ``"static"`` means no listing endpoint exists; ``"unavailable"``
        means a reachable listing failed without a cache; ``"unauthenticated"``
        needs a credential; and ``"empty"`` is an authoritative zero-row answer.
    """
    rows = _static_rows(provider_id)
    try:
        return _available_models(
            provider_id,
            rows,
            api_key=api_key,
            is_oauth=is_oauth,
            account_id=account_id,
            base_url=base_url,
            ttl_s=ttl_s,
            cache_dir=cache_dir,
            client=client,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - a broken provider is an annotation
        # The layers below already swallow transport failures, so reaching here
        # means a cache or coding fault. Even then the picker has to open: the
        # registry alone is a usable answer, a traceback is not.
        logger.debug("falling back to the static registry for %s: %s", provider_id, exc)
        definition = get_provider_definition(provider_id)
        has_transport = definition is not None and _TRANSPORTS.get(definition.id) is not None
        return merge_models(rows, None), ("unavailable" if has_transport else "static")


def _available_models(
    provider_id: str,
    rows: Mapping[str, ModelInfo],
    *,
    api_key: str | None,
    is_oauth: bool,
    account_id: str | None,
    base_url: str | None,
    ttl_s: float,
    cache_dir: Path | None,
    client: httpx.Client | None,
    timeout: float,
) -> tuple[list[DiscoveredModel], ListingStatus]:
    definition = get_provider_definition(provider_id)
    if is_oauth and definition is not None and definition.store_credentials_as:
        definition = get_provider_definition(definition.store_credentials_as) or definition
    transport = _TRANSPORTS.get(definition.id) if definition is not None else None
    if definition is None or transport is None:
        return merge_models(rows, None), "static"

    resolved_base = base_url or definition.base_url
    if not resolved_base:
        return merge_models(rows, None), "static"

    keyless_listing = definition.allows_missing_api_key or definition.id in PUBLIC_LISTING_PROVIDERS
    if not api_key and not keyless_listing:
        # Reported apart from "static" because it is the one status the user can
        # act on: log in and the listing appears. The keyless exceptions are local
        # servers (Ollama needs no credential at all) and the aggregators, whose
        # catalogue is a public page even though inference is not.
        return merge_models(rows, None), "unauthenticated"

    # ``cached_listing`` signals "the fetch failed, serve the stale document" by
    # catching an exception from the thunk, and reports nothing about which path
    # it took. This flag is how a live answer is told from a cached one, which is
    # the whole difference between the "ok" and "cached" statuses.
    fetched = False

    capture = listing_capture_version(definition.id)

    def fetch() -> dict[str, Any]:
        nonlocal fetched
        live = fetch_models(
            definition.id,
            api_key=api_key,
            is_oauth=is_oauth,
            account_id=account_id,
            base_url=resolved_base,
            client=client,
            timeout=timeout,
        )
        if live is None:
            raise _ListingUnavailable(definition.id)
        fetched = True
        return {
            "capture": capture,
            "models": [dataclasses.asdict(row) for row in live],
        }

    capture = listing_capture_version(definition.id)
    key = _cache_key(definition.id, is_oauth=is_oauth, account_id=account_id)
    payload = cached_listing(key, fetch, ttl_s=ttl_s, cache_dir=cache_dir)
    live_rows = _rows_from_payload(payload, capture)
    if live_rows is None:
        if payload is not None:
            # A document `cached_listing` could read but this reader cannot use:
            # a `models` key that is not an array (a payload-shape change, a
            # truncated write) or a capture older than the transports that will
            # read it. It is written before anything interprets it, so leaving it
            # in place serves the same failure as a FRESH cache hit on every start
            # until the TTL expires: a planted document with a dict `models`
            # produced three consecutive `static` results and zero fetches.
            # Dropping it costs one refetch.
            invalidate(key, cache_dir=cache_dir)
            # Re-enter once: the first call served the malformed fresh document
            # without invoking the fetch thunk.
            payload = cached_listing(key, fetch, ttl_s=ttl_s, cache_dir=cache_dir)
            live_rows = _rows_from_payload(payload, capture)
        if live_rows is None:
            # The registry remains usable offline, but the status tells the
            # picker it is provisional rather than entitlement-authoritative.
            return merge_models(rows, None), "unavailable"

    if definition.id == "openai":
        # OpenAI's API-key and subscription catalogues are entitlement lists,
        # not partial metadata overlays. Re-adding registry rows absent from the
        # response resurrects explicitly hidden or unavailable models.
        listed_static = {row.id: rows[row.id] for row in live_rows if row.id in rows}
        merged = merge_models(listed_static, live_rows)
    else:
        merged = merge_models(rows, live_rows)
    if not fetched:
        return merged, "cached"
    return merged, ("empty" if not live_rows else "ok")

"""Provider-neutral prices and limits from models.dev, cached like a listing.

WHY THIS EXISTS
---------------
No direct provider quotes money in its model listing: Anthropic's ``/v1/models``
carries ``id``, ``display_name``, ``created_at`` and limits, OpenAI's is bare
ids, Gemini's has token limits only. So for an id the shipped registry has not
been taught, the ONLY price source used to be the OpenRouter listing, looked up
under a per-provider namespace. That coupled every direct provider's cost
display to one aggregator's public document and its id spellings — and on the
day Anthropic published ``claude-fable-5-1`` the OpenRouter document on disk was
six hours old and predated the row, so a user signed in only to Anthropic ran
the whole day at ``$0.00``.

`models.dev <https://models.dev>`_ is a community-maintained, keyless JSON of
every provider's models with ``cost {input, output, cache_read, cache_write}``
in $/MTok and ``limit {context, output}``, provider-keyed, updated the day a
model ships (verified: ``anthropic.models["claude-fable-5-1"]`` carried
10/50/0.25/12.5 and a 1M/128k limit on its release date). It answers a weak
ETag and honours ``If-None-Match`` with a 0-byte 304, so keeping it fresh
hourly costs one header round trip per machine.

WHAT IS STORED
--------------
The source is 4.4 MB and 200+ providers. What lands on disk is a PROJECTION:
only the providers this tree maps (:data:`_PRICE_CATALOGUE_KEYS`) and only the
five fields a :class:`DiscoveredModel` can carry — measured at ~141 KB, a
quarter of the OpenRouter listing, so the JSON parse the resolution memo exists
to avoid stays in the tens of milliseconds. The ETag rides in the document so
the next fetch can be conditional. The document goes through
:func:`local_operator.model.catalogue.read_listing` under the key
``models-dev.listing`` — same directory, same lease, same stale-while-revalidate
state machine as every provider listing.

WHAT IS NOT TAKEN
-----------------
``supports_images`` is left ``None``: it carries a three-valued contract in
which a stated ``false`` is the PROVIDER's denial, and a second-hand catalogue
has no standing to issue one. ``supports_prompt_cache`` is inferred from a
quoted cache-read price, the same inference discovery makes for an
OpenAI-compatible listing and only ever widening.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from local_operator.model.catalogue import (
    DEFAULT_TTL_S,
    MISS_REFETCH_MIN_AGE_S,
    SOFT_TTL_S,
    _schedule_revalidate,
    invalidate,
    peek_listing,
    read_listing,
)
from local_operator.model.discovery import (
    DEFAULT_TIMEOUT_S,
    DiscoveredModel,
    _positive_float,
    _positive_int,
)
from local_operator.model.ids import id_spellings, normalised_id

logger = logging.getLogger("local_operator.model.prices")

#: The public document. Keyless; see the module docstring for what it carries.
MODELS_DEV_URL = "https://models.dev/api.json"

#: Document name in the catalogue cache. Ends in ``.listing`` so the legacy
#: purge glob (``*.models.json``) can never match it.
PRICE_CATALOGUE_KEY = "models-dev.listing"

#: Stamped into the projection and required when it is read back, for the same
#: reason ``discovery.LISTING_CAPTURE_VERSIONS`` exists: a reader that starts
#: needing a field the writer did not record must be able to force one refetch
#: rather than serve zeros for a day. Version 1 is the five-field projection.
PRICE_CATALOGUE_CAPTURE = 1

#: A local (canonical) provider id, mapped to the models.dev provider keys that
#: describe the same models — FIRST MATCH WINS, per model id. The local id is the
#: canonical one after ``credential_provider_id``, so the login flavours
#: (``xai-oauth``, ``openai-device``, ...) need no entry.
#:
#: ``_lookup`` returns the first key's entry for an id OUTRIGHT; it never
#: merges a second key's fields into it. The second keys are subscription/plan
#: catalogues models.dev lists apart from the pay-per-token one (Kimi's coding
#: plan lists ``k3``, which ``moonshotai`` does not; Z.AI's coding plan
#: likewise), reached only for an id the first key lacks entirely.
#: ``alibaba-token-plan`` is a plan of ``alibaba``'s models with its own key
#: first, so an id it lists answers with the plan's numbers alone — which
#: models.dev quotes as 0/0 on purpose, because the plan bills credits rather
#: than USD per token. Those zeros are the intended answer ("cost unknown", per
#: ``_fill_from_row``), NOT a hole for the pay-per-token key to fill: quoting
#: ``alibaba``'s USD rate for a credit plan would print a cost the user is not
#: paying. The pay-per-token key is there for a model the plan does not list.
#:
#: ``radient`` is deliberately absent: models.dev does not list it, and its own
#: listing quotes prices, so leg 1 of resolution already covers it.
_PRICE_CATALOGUE_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "openai": ("openai",),
    "google": ("google",),
    "xai": ("xai",),
    "deepseek": ("deepseek",),
    "mistral": ("mistral",),
    "kimi": ("moonshotai", "kimi-for-coding"),
    "zai": ("zai", "zai-coding-plan"),
    "alibaba": ("alibaba",),
    "alibaba-token-plan": ("alibaba-token-plan", "alibaba"),
    "openrouter": ("openrouter",),
}

#: Every models.dev key the projection keeps, derived from the map above so the
#: two cannot disagree about what is on disk.
_PROJECTED_PROVIDERS: frozenset[str] = frozenset(
    key for keys in _PRICE_CATALOGUE_KEYS.values() for key in keys
)


def project(body: Mapping[str, Any], etag: str | None) -> dict[str, Any]:
    """The on-disk document for a models.dev body: mapped providers, five fields.

    Pure, so a test can hand it a captured body. Anything that is not the
    expected shape is skipped rather than raised — a key rename upstream must
    land as "no prices" (holes stay holes), never as a failed resolution.
    """
    providers: dict[str, dict[str, Any]] = {}
    for provider_key in _PROJECTED_PROVIDERS:
        provider = body.get(provider_key)
        if not isinstance(provider, Mapping):
            continue
        models = provider.get("models")
        if not isinstance(models, Mapping):
            continue
        projected: dict[str, Any] = {}
        for model_id, model in models.items():
            if not isinstance(model_id, str) or not isinstance(model, Mapping):
                continue
            cost = model.get("cost")
            limit = model.get("limit")
            projected[model_id] = {
                "name": model.get("name") if isinstance(model.get("name"), str) else "",
                "cost": dict(cost) if isinstance(cost, Mapping) else {},
                "limit": dict(limit) if isinstance(limit, Mapping) else {},
                "attachment": (
                    model.get("attachment") if isinstance(model.get("attachment"), bool) else None
                ),
                "release_date": (
                    model.get("release_date") if isinstance(model.get("release_date"), str) else ""
                ),
            }
        providers[provider_key] = projected
    return {"capture": PRICE_CATALOGUE_CAPTURE, "etag": etag, "providers": providers}


def _fetch(timeout: float, cache_dir: Path | None) -> dict[str, Any]:
    """The ``read_listing`` thunk: conditional GET, projected on the way in.

    A 304 hands the previous document back unchanged so ``_write_cache`` re-stamps
    its ``fetched_at`` — "validated just now" — without re-parsing 4.4 MB. Any
    other non-200, or a body that is not an object, raises so ``read_listing``
    applies its stale-beats-absent rule instead of overwriting a good document.

    ``httpx`` is imported here rather than at module level for the same reason
    ``configure`` imports discovery lazily: this module is reached from the
    resolution path, and a CLI invocation that resolves a fully described model
    never needs an HTTP client.
    """
    import httpx

    previous = peek_listing(PRICE_CATALOGUE_KEY, cache_dir=cache_dir).payload
    headers: dict[str, str] = {}
    etag = previous.get("etag") if previous is not None else None
    if isinstance(etag, str) and etag:
        headers["If-None-Match"] = etag
    response = httpx.get(MODELS_DEV_URL, headers=headers, timeout=timeout, follow_redirects=True)
    if response.status_code == 304 and previous is not None:
        logger.debug("models.dev catalogue unchanged (304, etag %s)", etag)
        return dict(previous)
    if response.status_code != 200:
        raise RuntimeError(f"models.dev answered {response.status_code}")
    body = response.json()
    if not isinstance(body, Mapping):
        raise RuntimeError("models.dev body is not an object")
    document = project(body, response.headers.get("etag"))
    logger.debug(
        "models.dev catalogue fetched: %d providers projected, etag %s",
        len(document["providers"]),
        document["etag"],
    )
    return document


def _usable(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The ``providers`` map of a document this reader can use, else ``None``.

    A capture mismatch is treated as no document, same as discovery's reader:
    the shape may parse perfectly while a field this version needs is missing.
    """
    if payload is None:
        return None
    if _positive_int(payload.get("capture")) != PRICE_CATALOGUE_CAPTURE:
        return None
    providers = payload.get("providers")
    return providers if isinstance(providers, Mapping) else None


def _row(model_id: str, entry: Mapping[str, Any]) -> DiscoveredModel:
    """One projected entry as the struct every other resolution leg speaks."""
    cost = entry.get("cost")
    cost = cost if isinstance(cost, Mapping) else {}
    limit = entry.get("limit")
    limit = limit if isinstance(limit, Mapping) else {}
    cache_read = _positive_float(cost.get("cache_read"))
    name = entry.get("name")
    return DiscoveredModel(
        id=model_id,
        name=name if isinstance(name, str) else "",
        context_window=_positive_int(limit.get("context")),
        max_tokens=_positive_int(limit.get("output")),
        input_price=_positive_float(cost.get("input")),
        output_price=_positive_float(cost.get("output")),
        cache_read_price=cache_read,
        cache_write_price=_positive_float(cost.get("cache_write")),
        # Never from here: a second-hand catalogue cannot issue the provider's
        # denial that a stated ``False`` means. See the module docstring.
        supports_images=None,
        supports_prompt_cache=cache_read > 0,
    )


def _lookup(providers: Mapping[str, Any], provider: str, model_id: str) -> DiscoveredModel | None:
    """``model_id`` under the first mapped key that lists it, or ``None``.

    Three spellings per key, most literal first: the id as given, the normalised
    id (``models/`` prefix stripped, case folded, matched against normalised keys)
    and the dotted/dashed rewrites in :func:`id_spellings`. For ``openrouter``
    the id already carries its ``vendor/`` namespace, which is how models.dev
    keys that provider too, so no namespace mapping is needed.
    """
    for key in _PRICE_CATALOGUE_KEYS.get(provider, ()):
        models = providers.get(key)
        if not isinstance(models, Mapping):
            continue
        for spelling in id_spellings(model_id):
            entry = models.get(spelling)
            if isinstance(entry, Mapping):
                return _row(spelling, entry)
        wanted = normalised_id(model_id)
        for candidate, entry in models.items():
            if isinstance(candidate, str) and isinstance(entry, Mapping):
                if normalised_id(candidate) == wanted:
                    return _row(candidate, entry)
    return None


def price_catalogue_row(
    provider: str,
    model_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: Path | None = None,
) -> DiscoveredModel | None:
    """Prices and limits for ``provider/model_id`` from models.dev, or ``None``.

    Never raises. ``timeout`` bounds the ONE request this may make and is the
    caller's remaining resolution budget (see ``configure._remaining_budget``);
    a cold machine with no document pays one 4.4 MB download inside it.

    The ``want_id`` rule from discovery applies here too: an id absent from a
    document at least ``MISS_REFETCH_MIN_AGE_S`` old is refetched once,
    synchronously — for this document that is usually a 0-byte 304 unless the
    row really did just land, which is the case this exists for.
    """
    if provider not in _PRICE_CATALOGUE_KEYS:
        return None
    try:
        return _price_catalogue_row(provider, model_id, timeout, ttl_s, cache_dir)
    except Exception as exc:  # noqa: BLE001 — a price is never worth a failed start
        logger.debug("models.dev catalogue unavailable for %s/%s: %s", provider, model_id, exc)
        return None


def _price_catalogue_row(
    provider: str, model_id: str, timeout: float, ttl_s: float, cache_dir: Path | None
) -> DiscoveredModel | None:
    def fetch() -> dict[str, Any]:
        # On the calling path: the leg's budget (3 s at most, less when the
        # provider's own listing spent some of the shared deadline).
        return _fetch(timeout, cache_dir)

    def revalidate() -> dict[str, Any]:
        # Off the calling path, where the leg budget is the wrong ceiling: a 4.4 MB
        # download that needs more than 3 s would fail every background attempt
        # and leave the document to the 24 h sync path. Same full ceiling the
        # cold-miss retry below already uses.
        return _fetch(DEFAULT_TIMEOUT_S, cache_dir)

    def lacks_model(document: Mapping[str, Any], age_s: float) -> bool:
        # The `want_id` rule: a usable document old enough to predate the id
        # is refetched once — usually a 0-byte 304 unless the row really did
        # just land, which is the case this exists for. An unusable document
        # is handled by the invalidate-and-re-enter below, not here.
        if age_s < MISS_REFETCH_MIN_AGE_S:
            return False
        known = _usable(document)
        return known is not None and _lookup(known, provider, model_id) is None

    listing = read_listing(
        PRICE_CATALOGUE_KEY,
        fetch,
        soft_ttl_s=min(SOFT_TTL_S, ttl_s),
        ttl_s=ttl_s,
        cache_dir=cache_dir,
        refetch_if=lacks_model,
        revalidate=revalidate,
    )
    providers = _usable(listing.payload)
    if providers is None and listing.payload is not None:
        # A document from an older capture, or one whose shape this reader does
        # not recognise: drop it and refetch IN THIS CALL, the same recovery
        # discovery performs, so the fix is not invisible for a day.
        invalidate(PRICE_CATALOGUE_KEY, cache_dir=cache_dir)
        listing = read_listing(
            PRICE_CATALOGUE_KEY, fetch, soft_ttl_s=0, ttl_s=0, cache_dir=cache_dir
        )
        providers = _usable(listing.payload)
    if providers is None:
        if listing.failed:
            # COLD MISS THAT FAILED. The first resolution on a machine with no
            # document pays a 4.4 MB download inside a 3 s leg budget, and on a
            # slow link that times out — evidence the budget was too small, not
            # that the network is down. The one case where a failed sync fetch
            # is followed by a background retry, with the full default timeout,
            # so the NEXT resolution finds the document instead of the session
            # running unpriced all day. Backoff-bounded like every other
            # revalidation; a genuinely offline machine gets one thread per
            # five minutes, not one per repaint.
            _schedule_revalidate(
                PRICE_CATALOGUE_KEY, lambda: _fetch(DEFAULT_TIMEOUT_S, cache_dir), cache_dir
            )
        return None
    return _lookup(providers, provider, model_id)

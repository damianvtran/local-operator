"""Provider-neutral prices and limits: models.dev first, OpenRouter's public
listing second, cached like a listing.

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

WHY TWO SOURCES
---------------
One community-maintained JSON is one point of failure: a day-0 gap (the row
not merged yet), a shape drift, or the host being down all land as "unpriced"
for every direct provider at once. OpenRouter's public ``/api/v1/models`` is an
INDEPENDENT keyless document that quotes prices for the same models under a
per-vendor namespace (``anthropic/claude-fable-5.1``), so it is the secondary
leg of a ranked chain (:func:`price_row`): models.dev answers first; OpenRouter
is consulted ONLY for an id models.dev has no PRICED row for; the shipped
registry is what the caller falls back to when both miss. OpenRouter never
overrides a price models.dev or the provider's own listing quoted, and it
stays the authority for ``openrouter/*`` ids, which never enter this chain.
Both documents sit under the same stale-while-revalidate, lease and ``want_id``
machinery, so neither adds a request to a path that already has an answer.

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

import dataclasses
import logging
import math
import time
from collections.abc import Iterable, Mapping
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

#: The models.dev keys whose ``0/0`` means "billed in plan credits", NOT "free".
#:
#: These catalogues describe subscription plans: the vendor genuinely does not
#: quote a USD-per-token rate, so models.dev records zeros as a way of saying
#: "not priced in dollars". A row from one of them therefore STATES a zero and
#: still has an unknowable cost — the one shape where the stated-zero signal
#: must not become the word ``free`` on screen, because the user IS paying, just
#: not per token. The distinction is per KEY rather than per row because it is a
#: property of the billing arrangement the catalogue describes: every id under
#: ``alibaba-token-plan`` is credit-billed and every id under ``zai`` is not,
#: whatever their individual costs happen to be.
#:
#: Contrast ``zai``/``moonshotai``/``google``, which are pay-per-token
#: catalogues: a zero there is a quoted zero (``zai/glm-4.7-flash`` is $0 on
#: Z.AI's own pricing page) and is exactly what ``free`` is for.
_PLAN_BILLED_KEYS: frozenset[str] = frozenset(
    {"alibaba-token-plan", "kimi-for-coding", "zai-coding-plan"}
)

#: Every models.dev key the projection keeps, derived from the map above so the
#: two cannot disagree about what is on disk.
_PROJECTED_PROVIDERS: frozenset[str] = frozenset(
    key for keys in _PRICE_CATALOGUE_KEYS.values() for key in keys
)

#: A direct provider's id, mapped to the namespace OpenRouter publishes the same
#: models under — the SECONDARY leg of :func:`price_row`. Only a provider whose
#: OWN listing quotes no prices needs an entry, which is every direct provider in
#: this tree. The aggregators are deliberately absent: their own listing IS the
#: priced one, and ``openrouter/*`` ids already carry their namespace.
#:
#: Spelled out rather than derived because three of the namespaces are renames
#: (``x-ai``, ``qwen``, ``moonshotai``), verified against
#: ``GET https://openrouter.ai/api/v1/models`` on 2026-09-02; a derived guess
#: would silently price a model from whatever else happened to match.
#:
#: ``alibaba-token-plan`` has NO entry on purpose. models.dev quotes its models
#: as 0/0 because the plan bills credits rather than USD per token, and that zero
#: is the intended answer ("cost unknown"). Falling through to ``qwen/`` here
#: would print the pay-per-token USD rate for a plan the user is not paying it
#: on — precisely the number this chain must never invent.
OPENROUTER_NAMESPACE: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "deepseek": "deepseek",
    "mistral": "mistralai",
    "xai": "x-ai",
    "alibaba": "qwen",
    "kimi": "moonshotai",
    "zai": "z-ai",
}

#: The OpenRouter document's name in the catalogue cache, for the keyless read
#: the secondary leg performs. Same key ``discovery._cache_key`` derives for the
#: provider's own listing (no credential, so no scope suffix): the picker's
#: OpenRouter rows and this leg share ONE document on disk.
OPENROUTER_CATALOGUE_KEY = "openrouter.listing"

#: Relative gap past which two sources pricing the same id are logged as
#: disagreeing. Diagnostic only — nothing acts on it — so a loose bound that
#: ignores rounding (models.dev quotes $10, OpenRouter 1e-5/token) is the point.
PRICE_DISAGREEMENT_RATIO = 0.05


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


def _stated_price(value: object) -> float | None:
    """``value`` as a price models.dev actually STATED, or ``None``.

    Deliberately not :func:`_positive_float`, which collapses "stated zero",
    "absent" and "malformed" into the same ``0.0`` — this module's whole
    ranking turns on keeping the first apart from the other two. A bool is an
    int subclass and a ``True`` price is nonsense; a negative or non-finite
    number is not a price any vendor can charge, so both read as UNSTATED and
    the chain keeps looking rather than recording a zero the vendor never said.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _row(model_id: str, entry: Mapping[str, Any], key: str = "") -> DiscoveredModel:
    """One projected entry as the struct every other resolution leg speaks.

    ``key`` is the models.dev catalogue the entry was found under, and it decides
    what a stated zero MEANS: a plan catalogue (:data:`_PLAN_BILLED_KEYS`) quotes
    zeros because it bills credits rather than dollars, so its rows stop the
    chain exactly as before but are never marked ``free`` for display.

    Two facts live in a ``cost`` mapping and the chain needs them apart:

    * **models.dev STATED a price** — both ``input`` and ``output`` are real
      numbers. That alone is what stops the chain (see :func:`_priced`): a
      quoted price is an answer whatever its value.
    * **the stated price is ZERO** — both of those numbers are exactly ``0``.
      Only then is the row marked with :data:`_STATED_ZERO`, because only then
      does the struct lose the fact: a ``DiscoveredModel`` spells "free" and
      "unknown" the same way (``0.0``), so a symmetric zero would otherwise
      read as "nobody said anything" and fall through to the secondary.

    An ASYMMETRIC row (``input: 0, output: 15``) is stated but not zero, so it
    keeps its numbers and stops the chain on the ordinary positive-leg test —
    marking it would flatten a real $15 output price to free. Anything else —
    a missing or empty ``cost``, or one without numeric ``input`` and
    ``output`` — stays ``0.0`` and therefore "unanswered".
    """
    cost = entry.get("cost")
    cost = cost if isinstance(cost, Mapping) else {}
    limit = entry.get("limit")
    limit = limit if isinstance(limit, Mapping) else {}
    cache_read = _positive_float(cost.get("cache_read"))
    name = entry.get("name")
    stated_input = _stated_price(cost.get("input"))
    stated_output = _stated_price(cost.get("output"))
    stated_zero = stated_input == 0.0 and stated_output == 0.0
    input_price = _positive_float(cost.get("input"))
    output_price = _positive_float(cost.get("output"))
    return DiscoveredModel(
        id=model_id,
        name=name if isinstance(name, str) else "",
        context_window=_positive_int(limit.get("context")),
        max_tokens=_positive_int(limit.get("output")),
        input_price=_STATED_ZERO if stated_zero else input_price,
        output_price=_STATED_ZERO if stated_zero else output_price,
        # Nearly the same fact as the marker, in the field that OUTLIVES this
        # module. ``_STATED_ZERO`` is an in-flight marker the chain strips on
        # the way out (:func:`_unmark`) and answers "does this row stop the
        # chain"; ``free`` is the durable statement a display reads and answers
        # "may this row say the word". They differ on exactly the plan
        # catalogues, whose zeros are an answer to the first question and not to
        # the second — a credit-billed model's real cost is unknowable, and
        # printing ``free`` for it would be a lie the user could act on.
        free=stated_zero and key not in _PLAN_BILLED_KEYS,
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

    The MATCHED key is handed to :func:`_row`, not just the entry: whether a
    stated zero may be displayed as ``free`` depends on which catalogue answered
    (see :data:`_PLAN_BILLED_KEYS`), and this is the only place that knows.
    """
    for key in _PRICE_CATALOGUE_KEYS.get(provider, ()):
        models = providers.get(key)
        if not isinstance(models, Mapping):
            continue
        for spelling in id_spellings(model_id):
            entry = models.get(spelling)
            if isinstance(entry, Mapping):
                return _row(spelling, entry, key)
        wanted = normalised_id(model_id)
        for candidate, entry in models.items():
            if isinstance(candidate, str) and isinstance(entry, Mapping):
                if normalised_id(candidate) == wanted:
                    return _row(candidate, entry, key)
    return None


#: The in-flight marker for "models.dev priced this id and the price is zero".
#: A listing row's PRICE FIELDS conflate "free" with "unknown" (0.0 either way),
#: so a stated zero cannot survive the trip through them; the marker lets the
#: chain tell the two apart internally and is stripped by :func:`_unmark` before
#: a row leaves this module. Module-private: no caller ever sees a negative
#: price.
#:
#: Purely in-flight, and NOT how the fact reaches a display —
#: :attr:`DiscoveredModel.free` is, set beside this marker in :func:`_row` and
#: never stripped. The two answer different questions (see the comment there),
#: which is why both exist rather than one doing double duty.
#:
#: NOT the same ``-1.0`` as ``providers.controller._price``'s UNKNOWN sentinel,
#: which the picker renders blank. The two spell opposite facts — "the vendor
#: stated zero" here, "nobody knows" there — and they never meet, because
#: :func:`_unmark` strips this one before any row reaches that function. Widen
#: either one's reach and they collide silently, so keep them apart.
_STATED_ZERO = -1.0


def _stated_zero(row: DiscoveredModel | None) -> bool:
    """Whether ``row`` carries the marker — i.e. models.dev said 0/0 on purpose.

    Both legs, because :func:`_row` only marks a row whose stated ``input`` and
    ``output`` are BOTH zero; a half-marked row cannot be constructed.
    """
    return row is not None and row.input_price == _STATED_ZERO and row.output_price == _STATED_ZERO


def _unmark(row: DiscoveredModel | None) -> DiscoveredModel | None:
    """The chain's exit: a stated-zero marker back to the ``0.0`` the struct speaks.

    ``price_row`` and ``price_catalogue_row`` both return through this so the
    marker cannot leak to a caller. ``0.0`` is the same zero the registry uses.

    ``free`` is deliberately NOT stripped with it: ``dataclasses.replace``
    carries it through, and it is what tells the display apart the two things
    this ``0.0`` could mean. ``providers.controller._price`` still maps a bare
    ``0.0`` to its own ``-1.0`` unknown marker for any provider that wants an
    API key — blank beats printing a third party's rate for the vendor's own
    endpoint — but it now consults ``free`` first, so a vendor-quoted zero
    reaches the picker as the word instead of an empty cell.
    """
    if _stated_zero(row):
        assert row is not None  # for the type checker; ``_stated_zero`` implies it
        return dataclasses.replace(row, input_price=0.0, output_price=0.0)
    return row


def _priced(row: DiscoveredModel | None) -> bool:
    """Whether a row answers the MONEY question — i.e. models.dev STATED a price.

    The chain exits on "stated", not on "non-zero" and not on "zero": a quoted
    price is an answer whatever its value, and the OpenRouter secondary — which
    prices the same weights hosted by a THIRD party — must never overwrite one.
    The two shapes a stated price takes are tested separately here because the
    struct records them differently:

    * a positive leg (``2.5/15``, and also the asymmetric ``0/15``) survives
      into the struct as itself, so the ordinary positive-leg test finds it;
    * a symmetric zero (``zai/glm-4.7-flash``, ``kimi-for-coding/k3``: free, or
      billed in plan credits) cannot, so :func:`_row` marks it and
      :func:`_stated_zero` reads the marker back.

    Those two are exhaustive over stated prices: :func:`_stated_price` refuses
    negatives and non-finite values, so a stated row always has a positive leg
    or is marked. Only an ABSENT price — no entry, or no ``cost`` mapping
    (``google/gemma-4-31b-it``) — leaves the money question open.

    An OpenRouter row passes through :func:`openrouter_lookup` unchanged and
    carries no marker, so for it this is the plain positive-leg test: a
    zero-priced OpenRouter route is not trusted to price the vendor's own API.
    """
    return row is not None and (row.input_price > 0 or row.output_price > 0 or _stated_zero(row))


def openrouter_lookup(
    rows: Iterable[DiscoveredModel], provider: str, model_id: str
) -> DiscoveredModel | None:
    """``provider/model_id`` in OpenRouter's rows, under the vendor namespace.

    Priced rows only: an unpriced OpenRouter row is a routing stub that can answer
    neither question a price lookup asks, and matching one would shadow a
    better-spelled sibling further down the candidate list. Spellings come from
    :func:`id_spellings` (``claude-fable-5-1`` → ``claude-fable-5.1``), the same
    map the models.dev lookup trusts, and the date-suffixed forms are tried
    before the stripped ones for the reason that function documents.

    Returns ``None`` for a provider with no namespace — the aggregators, whose
    ``openrouter/*`` ids are their own listing's business, and plan providers
    whose USD rate must not be borrowed (see :data:`OPENROUTER_NAMESPACE`).
    """
    namespace = OPENROUTER_NAMESPACE.get(provider)
    if namespace is None:
        return None
    wanted = {f"{namespace}/{spelling}" for spelling in id_spellings(model_id)}
    for row in rows:
        if row.id in wanted and _priced(row):
            return row
    return None


def _disagree(primary: DiscoveredModel, secondary: DiscoveredModel) -> bool:
    for mine, theirs in (
        (primary.input_price, secondary.input_price),
        (
            primary.output_price,
            secondary.output_price,
        ),
    ):
        if (
            mine > 0
            and theirs > 0
            and abs(mine - theirs) / max(mine, theirs) > (PRICE_DISAGREEMENT_RATIO)
        ):
            return True
    return False


def price_row(
    provider: str,
    model_id: str,
    *,
    models_dev: Mapping[str, Any] | None,
    openrouter: Iterable[DiscoveredModel] | None,
) -> DiscoveredModel | None:
    """THE ranked price chain over two pre-read documents, or ``None``.

    Pure: it reads nothing and fetches nothing, which is what lets the resolver
    (one id, budgeted, may fetch) and the picker's row builder (hundreds of
    ids, one read per document, never fetches) share it without drifting.

    1. models.dev (``models_dev``: the projection's ``providers`` map) — first
       refusal. A row here is the answer outright, INCLUDING one that states
       ``0/0``: the vendor quoted zero (or bills by plan credits) and the
       secondary's third-party hosting rate must not replace it. The zero
       arrives marked (:data:`_STATED_ZERO`) and is stripped on the way out,
       so callers see the same ``0.0`` they always have.
    2. OpenRouter (``openrouter``: the public listing's rows) — ONLY when
       models.dev has no row answering the money question: a day-0 gap, a
       cost-less stub, a shape drift, or the host being down. It fills the
       money; limits models.dev already gave (from a cost-less stub, say) are
       kept because the primary's limits are native while OpenRouter
       advertises the widest window across its routes.
    3. ``None`` — the caller keeps whatever the shipped registry says.

    When both price an id and disagree by more than
    :data:`PRICE_DISAGREEMENT_RATIO`, the fact is logged at debug and nothing
    else happens: the primary still wins. The log is how a drifting source gets
    noticed without either source being allowed to "correct" the other.
    """
    primary = _lookup(models_dev, provider, model_id) if models_dev is not None else None
    secondary = openrouter_lookup(openrouter, provider, model_id) if openrouter else None
    if _priced(primary):
        assert primary is not None  # for the type checker; ``_priced`` implies it
        if secondary is not None and _disagree(primary, secondary):
            logger.debug(
                "price sources disagree for %s/%s: models.dev %s/%s vs openrouter %s/%s",
                provider,
                model_id,
                primary.input_price,
                primary.output_price,
                secondary.input_price,
                secondary.output_price,
            )
        return _unmark(primary)
    if secondary is None:
        return _unmark(primary)
    if primary is None:
        return secondary
    # A models.dev stub with limits but no cost, and an OpenRouter price: take the
    # money from the secondary and keep the primary's native limits where it has
    # them. Every field is "first source that has it", never a blend.
    # ``free`` rides with the MONEY, so it comes from the secondary along with
    # the prices and is never taken from the primary: the primary reaching here
    # is by definition one that answered no money question at all (an empty
    # ``cost``), so it has nothing to say about whether the model is free.
    return dataclasses.replace(
        secondary,
        name=primary.name or secondary.name,
        context_window=primary.context_window or secondary.context_window,
        max_tokens=primary.max_tokens or secondary.max_tokens,
    )


def price_catalogue_row(
    provider: str,
    model_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: Path | None = None,
) -> DiscoveredModel | None:
    """Prices and limits for ``provider/model_id`` through the ranked chain.

    Never raises. ``timeout`` bounds the requests this may make and is the
    caller's remaining resolution budget (see ``configure._remaining_budget``),
    shared across both legs: a cold machine with no models.dev document pays
    one 4.4 MB download inside it, and the OpenRouter leg gets what is left.

    The ``want_id`` rule from discovery applies to both documents: an id absent
    from a document at least ``MISS_REFETCH_MIN_AGE_S`` old is refetched once,
    synchronously — for models.dev that is usually a 0-byte 304 unless the row
    really did just land, which is the case this exists for.

    The OpenRouter document is read ONLY when models.dev has no row answering
    the money question — and a stated ``0/0`` ANSWERS it, so a vendor-free or
    plan-billed model never touches the secondary either. A resolution
    models.dev answers costs no second parse, no second request, nothing on
    the paint path in the common case.
    """
    if provider not in _PRICE_CATALOGUE_KEYS:
        return None
    started = time.monotonic()
    try:
        models_dev = _models_dev_providers(provider, model_id, timeout, ttl_s, cache_dir)
    except Exception as exc:  # noqa: BLE001 — a price is never worth a failed start
        logger.debug("models.dev catalogue unavailable for %s/%s: %s", provider, model_id, exc)
        models_dev = None
    # The single "is the primary's answer complete" decision lives HERE so the
    # document read and the read-OpenRouter skip share one rule. The final
    # assembly — primary wins, secondary fills, limits merge — is all in
    # ``price_row``; this function only decides which documents it sees.
    primary = _lookup(models_dev, provider, model_id) if models_dev is not None else None
    if _priced(primary):
        return price_row(provider, model_id, models_dev=models_dev, openrouter=None)
    openrouter: list[DiscoveredModel] | None = None
    namespace = OPENROUTER_NAMESPACE.get(provider)
    if namespace is not None:
        try:
            openrouter = openrouter_rows(
                want_id=f"{namespace}/{model_id}",
                # What the primary left of the caller's budget, floored above
                # zero for the reason ``configure._remaining_budget`` gives.
                timeout=max(0.01, timeout - (time.monotonic() - started)),
                ttl_s=ttl_s,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 — same rule as the primary
            logger.debug("openrouter catalogue unavailable for %s/%s: %s", provider, model_id, exc)
    return price_row(provider, model_id, models_dev=models_dev, openrouter=openrouter)


def openrouter_rows(
    *,
    want_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: Path | None = None,
) -> list[DiscoveredModel]:
    """OpenRouter's public rows via discovery, credential-free.

    Goes through ``available_models`` rather than reading the document directly
    so the SAME stale-while-revalidate, lease and capture-version rules govern
    it as when the picker lists the ``openrouter`` provider — one document, one
    state machine. ``want_id`` is the NAMESPACED spelling (``anthropic/…``) so
    the miss-refetch rule fires for the row we are about to look up; the
    spelling map is applied by ``discovery._lists_id`` on the other side.
    Imported lazily: discovery pulls in httpx and this module is on the
    resolution path of CLI invocations that never need a listing.
    """
    from local_operator.model.discovery import available_models

    rows, _status = available_models(
        "openrouter",
        api_key=None,
        timeout=timeout,
        ttl_s=ttl_s,
        cache_dir=cache_dir,
        want_id=want_id,
    )
    return rows


def models_dev_providers(
    *, ttl_s: float = DEFAULT_TTL_S, cache_dir: Path | None = None
) -> Mapping[str, Any] | None:
    """The projection's ``providers`` map from DISK ONLY, or ``None``.

    For a caller that is about to price many ids at once (the picker's row
    builder) and must not put a fetch on its path: it reads what is there,
    whatever its age, and never schedules anything. A document from a capture
    this reader cannot use reads as absent; the next single-id resolution
    performs the drop-and-refetch, not this.
    """
    try:
        return _usable(peek_listing(PRICE_CATALOGUE_KEY, cache_dir=cache_dir).payload)
    except Exception as exc:  # noqa: BLE001 — a price is never worth a failed paint
        logger.debug("models.dev catalogue unreadable: %s", exc)
        return None


def _models_dev_providers(
    provider: str, model_id: str, timeout: float, ttl_s: float, cache_dir: Path | None
) -> Mapping[str, Any] | None:
    """The projection's ``providers`` map, read through the SWR state machine
    with the ``want_id`` miss rule armed for ``provider/model_id``."""

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
    return providers

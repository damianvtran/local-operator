"""Live per-provider model discovery, layered OVER the static registry.

``model/registry.py`` only knows the models that were current when it was last
edited, so a user who logs into Anthropic today cannot reach a model released
since: the picker never lists it and there is no way to learn its id in order to
type it. This module asks each provider for its own model list and folds the
answer into the registry.

Three properties are load-bearing, and each exists because the obvious version
of this feature broke in exactly that way:

- **List authority matches the endpoint.** ChatGPT's account-scoped Codex
  catalogue is the user's actual selectable set, so it replaces OpenAI's stale
  static ids when available. Generic compatibility endpoints are often partial
  entitlement snapshots, so those ids are UNIONed instead. In both cases fields
  fall back individually: a lean listing that returns only an id must not erase
  the registry's context window and stop compaction from firing, while a rich
  listing such as Anthropic's ``/v1/models`` supplies the real current limit.
- **Failure and emptiness are different answers.** A transport error yields
  ``None`` ("keep what we had") and a successful listing with no models yields
  ``[]`` ("this provider really has nothing"). Collapsing them turns a flaky
  network into a picker that claims the provider has no models at all.
- **Nothing here raises.** A provider that is down, misconfigured or unreachable
  is a status annotation in the UI, not a failed startup.
  :func:`available_models` degrades to the registry instead of propagating.

Results are cached on disk through
:func:`local_operator.model.catalogue.read_listing`, so the picker opens at
disk speed and a listing outage is invisible for as long as the cache holds.
An ageing document is served and refreshed in the background, and a requested
id missing from a document old enough to be wrong triggers one synchronous
refetch (``want_id``) — together, a model released today is offered today.
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

from local_operator.model.catalogue import (
    DEFAULT_TTL_S,
    MISS_REFETCH_MIN_AGE_S,
    SOFT_TTL_S,
    Listing,
    invalidate,
    invalidate_documents,
    peek_listing,
    read_listing,
)
from local_operator.model.effort import EFFORT_ORDER
from local_operator.model.ids import id_spellings, normalised_id
from local_operator.model.registry import ModelInfo, static_models
from local_operator.providers.registry import (
    AGGREGATOR_PROVIDERS,
    PROVIDER_REGISTRY,
    WireFormat,
    credential_provider_id,
    get_provider_definition,
)

logger = logging.getLogger("local_operator.model.discovery")

#: Ceiling on ONE listing, pagination included. A model picker and session start
#: both call this synchronously while the user waits, so an unreachable -- or
#: merely slow -- host must fail in seconds rather than hang on the default socket
#: timeout. ``_fetch_gemini`` spends it as a deadline across its pages rather than
#: per request, because 25 pages x this value is not "seconds".
DEFAULT_TIMEOUT_S = 10.0
#: ChatGPT OAuth tokens are subscription credentials, not OpenAI API keys:
#: ``api.openai.com/v1/models`` rejects them. The account-scoped Codex endpoint
#: is the catalogue used by OpenAI's own client and reports the currently
#: available slugs plus their display metadata and context windows.
OPENAI_CHATGPT_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
#: The endpoint requires a compatibility version and rejects a versionless
#: request. This describes the catalogue schema parsed below, not this package's
#: release number; using local-operator's current 0.x version would incorrectly
#: hide models whose minimum Codex client version is numerically newer.
OPENAI_MODELS_CLIENT_VERSION = "1.0.0"
OPENAI_OAUTH_PROVIDERS = frozenset({"openai", "openai-device"})
#: Visibility values this catalogue uses for rows that are internal helpers
#: rather than models an operator may select. A DENYLIST rather than the
#: inverse test: this listing may prune the registry, so an unrecognised or
#: renamed value must not be read as "hide every row" and empty the picker.
_OPENAI_CODEX_HIDDEN_VISIBILITY = frozenset({"hide", "hidden", "internal", "none"})

#: Anthropic pins its wire format with a dated header and rejects requests that
#: omit it. Duplicated from the wire client rather than imported, so listing a
#: provider's models does not drag the whole chat-client module into startup.
ANTHROPIC_VERSION = "2023-06-01"

#: Claude Pro/Max OAuth tokens are only accepted alongside this beta opt-in --
#: the same pairing the chat client and the token refresh already use.
ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"

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

#: Fraction of the context window above which a LISTING'S advertised output cap
#: is treated as a routing artifact rather than a limit worth seeding a spec
#: with. See :func:`sane_listing_max_tokens` for why the line sits here.
IMPLAUSIBLE_MAX_TOKENS_RATIO = 0.85

#: What an implausible cap is reduced TO, as a fraction of the window. Leaves
#: half the window for prompt and half for output — generous for any real
#: workload, and the reduction is a FLOOR under the request-time clamp in
#: ``providers.clients``, not the number that finally goes on the wire.
REDUCED_MAX_TOKENS_RATIO = 0.5


def sane_listing_max_tokens(max_tokens: int, context_window: int) -> int:
    """A listing's output cap, reduced when it is an implausible share of the window.

    Second line of defence behind the request-time clamp in
    ``providers.clients._effective_max_tokens``. That clamp is the real fix
    because it alone knows the prompt size; this one exists so an absurd number
    never reaches a ``ModelSpec`` in the first place, which protects the paths
    that never build a request body — the picker's displayed limits, any
    provider we never measure, and anything that reads ``max_output_tokens``
    directly.

    Providers count ``prompt + max_tokens`` against the window at admission, so
    a cap at 0.9 of the window leaves 10% for input. OpenRouter advertises
    ``meta/muse-spark-1.3`` as ``context_length: 1048576`` with
    ``top_provider.max_completion_tokens: 943718``, and a real session 400'd at
    ~113k of input on a 1M model.

    The threshold is measured, not guessed. Across the 419 live OpenRouter rows
    that quote both numbers, the ratio distribution has an empty band exactly
    where this line sits:

    * **~80 rows sit at exactly 0.9** — ``x-ai/grok-4.6`` (500000/450000),
      ``x-ai/grok-4.20`` (2000000/1800000), ``openai/gpt-oss-120b``
      (131072/117964). A round 0.9 across unrelated vendors is a gateway
      formula, not 80 independent engineering decisions.
    * **Nothing at all sits above 0.9**, so the guard has no upper tail.
    * **The band 0.80 < r < 0.87 is empty.** Below it sits a dense, plainly
      deliberate Mistral cluster at exactly 0.800 (``mistral-large``
      128000/102400), which must not be touched and is not. Above it the next
      rows are 0.870 (``nvidia/nemotron-3-nano``) and 0.879
      (``meta-llama/llama-3.3-70b``, ``stepfun/step-3.7-flash``). 0.85 splits
      that empty band.

    The 0.87-0.88 rows ARE reduced, and deliberately so rather than as an
    accepted false positive: ``llama-3.3-70b`` at 131072/115200 leaves 15,872
    tokens of prompt on a 128k model, which fails on any real agent turn. After
    the guard it serves 65,536 output tokens with 65,536 of prompt room — a cap
    no ordinary turn reaches, in exchange for a window that is usable at all.
    Every row this reduces has the same shape; none loses output capacity a
    caller could realistically have spent.

    Crucially this takes ``max_tokens`` from a LISTING only, never from the
    bundled registry. 52 shipped rows exceed this ratio legitimately — ``gpt-4o``
    is 128000/128000 and every ``grok-3`` row is 1.0 — because a hand-transcribed
    registry entry states the model's documented maximum rather than a gateway's
    formula. Applying the ratio there would reduce specs that work correctly
    today, which is exactly the regression this must not cause.

    A large cap on a large window is NOT the target and must keep working: a
    model serving 64k output on a 200k window is 0.32 and passes untouched.
    """
    if max_tokens <= 0 or context_window <= 0:
        # Nothing to compare against. Zero is this module's "unknown", and
        # inventing a limit from an unknown is how the -1 sentinel bugs started.
        return max_tokens
    if max_tokens <= int(context_window * IMPLAUSIBLE_MAX_TOKENS_RATIO):
        return max_tokens
    return int(context_window * REDUCED_MAX_TOKENS_RATIO)


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
#:
#: Version 2 for ``openrouter`` and ``radient`` is ``_row_from_openai_entry``
#: reading ``pricing.input_cache_write`` into ``cache_write_price``. A version-1
#: document parses to rows with a zero write price, which is harmless in itself,
#: but without the bump the real number would stay invisible for a day on every
#: install — the same argument as above. Only the two OpenAI-compatible
#: aggregators quote a write price, so only they pay the one-time refetch — and
#: it is paid ON the calling path: a version-1 document is invalidated and
#: refetched synchronously (up to ``DEFAULT_TIMEOUT_S``) by the first resolution
#: after the upgrade, not in the background. Exactly once per install, which is
#: acceptable; a bump here is not free for the user whose only provider is
#: OpenRouter.
#:
#: Version 3 for ``openrouter`` and ``radient`` is ``_row_from_openai_entry``
#: reading an explicitly quoted ``0`` into ``DiscoveredModel.free`` — a MEANING
#: the version-2 writer could not express, which is exactly the case this stamp
#: is for. A version-2 document parses to rows with ``free=False``, a perfectly
#: valid shape in which the 18 ``:free`` routes render a blank price cell
#: instead of the word ``free``; nothing else could notice, so without the bump
#: the fix would stay invisible for a day on every install. Same one-time
#: synchronous refetch the version-2 bump described, and paid by the same two
#: aggregators, since they are the only transports that quote a price at all.
#:
#: Version 4 for ``openrouter`` and ``radient`` is the modality gate on that
#: same flag (:func:`_bills_in_tokens`). Unlike the version-3 bump this one is
#: not about a field the writer omitted: the document records ``free`` as a
#: COMPUTED boolean, so a version-3 document written by the ungated parser
#: carries ``free: true`` for the two per-song-billed lyria rows and the reader
#: has no modality left to re-judge it with. Without the bump every existing
#: install keeps making the false claim on screen until its listing TTL expires
#: — the fix would ship and change nothing for a day. Same one-time synchronous
#: refetch, paid by the same two aggregators.
#:
#: Version 5 for ``openrouter`` and ``radient`` is ``_row_from_openai_entry``
#: reading ``reasoning.supported_efforts`` and ``reasoning.default_effort``
#: into :attr:`DiscoveredModel.reasoning_efforts` and
#: ``reasoning_default_effort`` — a FIELD the version-4 writer never recorded,
#: which is the plain case for this stamp. A version-4 document parses to rows
#: whose ladder is ``None``, and ``None`` is precisely "the listing said
#: nothing", so the spec builder falls back to ``model.effort``'s table and the
#: install behaves exactly as it does today: nothing crashes and nothing lies,
#: the fix is simply INVISIBLE for up to a day. That invisibility is the whole
#: reason the stamp exists, so the bump is required rather than optional — the
#: reported bug (no effort segment on ``openrouter/google/gemini-3.8-flash``)
#: would appear unfixed on every existing install until its TTL rolled. Same
#: one-time synchronous refetch on the calling path, paid by the same two
#: aggregators, since they are the only transports whose wire carries the field
#: at all (the anthropic/zai/xai/kimi/alibaba/codex documents and models.dev do
#: not, so ``anthropic`` stays at 2 and every default-1 transport stays at 1).
#: Version 6 for ``openrouter`` and ``radient`` is ``_row_from_openai_entry``
#: recording :attr:`DiscoveredModel.routed` — a MEANING the version-5 writer
#: could not express, the same case as the version-3 ``free`` bump. A version-5
#: document parses to rows with ``routed=False``, a valid shape in which
#: ``radient/auto`` keeps rendering the word ``free`` (its listing quotes a
#: symmetric zero) and ``openrouter/auto`` keeps rendering a blank cell. Both
#: are the exact wrong labels this change exists to replace, so without the
#: bump the fix would ship and change nothing on screen for up to a day on
#: every install that already has a listing. Same one-time synchronous refetch
#: the earlier bumps describe, paid by the same two aggregators, since they are
#: the only transports that quote a price at all.
LISTING_CAPTURE_VERSIONS: dict[str, int] = {
    "anthropic": 2,
    "openrouter": 6,
    "radient": 6,
    "radient-key": 6,
}
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
PUBLIC_LISTING_PROVIDERS = frozenset({"openrouter", "radient", "radient-key"})

#: What :func:`available_models` managed to do, for the UI to annotate:
#: ``ok`` fetched live now, ``cached`` served a stored document with no fetch
#: needed (fresh enough, or refreshing in the background), ``stale`` a fetch was
#: attempted on this call and FAILED so the stored document is what you got,
#: ``static`` registry only, ``unauthenticated`` needs a credential it was not
#: given, ``empty`` the provider answered and listed nothing.
#:
#: ``cached`` and ``stale`` used to be one value, which is why the picker footer
#: could not tell "fresh enough" from "offline" — the only case a user hunting
#: for a model released this morning actually needs to hear about.
ListingStatus = Literal["ok", "cached", "stale", "static", "unauthenticated", "empty"]


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

    ``free`` is the second exception to the "``0`` means unknown" rule above, and
    it exists for the same reason ``supports_images`` is three-state: the source
    said something the struct could not otherwise record. Both price legs are
    ``0.0`` for "nobody quoted a price" AND for "the vendor quotes zero", and the
    picker renders those two opposite facts as a blank cell and the word ``free``
    respectively. Only the parser that read the wire knows which it saw, so it
    sets this flag rather than leaving the display to guess from a float.

    ``reasoning_efforts`` is the third, and it is three-state for the reason
    ``supports_images`` is: the listing's silence and its denial are different
    answers. ``None`` means the listing said nothing about efforts and the
    resolution defers to ``model.effort``'s hand-transcribed table; a populated
    tuple is the router's own statement about the request it will accept, and it
    WINS over that table. An empty tuple would read as a denial, and no listing
    in this tree issues one — a ``reasoning`` object carrying ``mandatory`` but
    no ``supported_efforts`` answers a different question and omits ours, so it
    defers too. The parser therefore never produces ``()``: only ``None`` or a
    populated tuple.
    """

    id: str
    name: str = ""
    context_window: int = 0
    max_tokens: int = 0
    input_price: float = 0.0
    output_price: float = 0.0
    cache_read_price: float = 0.0
    #: $/MTok to WRITE a prompt-cache entry. Zero is "not quoted", and the
    #: consumers then fall back to the input price — which under-states an
    #: Anthropic 5-minute write by 20% (1.25x base), so a quoted number is
    #: worth carrying rather than guessing.
    cache_write_price: float = 0.0
    #: The SOURCE stated that both price legs are zero — the model is free at the
    #: point of use, not merely unpriced. Never inferred from the prices
    #: themselves (that is precisely the inference this field exists to replace):
    #: a row is only ``True`` here when a parser read an explicit zero off a wire
    #: or a document. A row that carries a positive price must never set it.
    free: bool = False
    #: The SOURCE stated that this endpoint is a META-ROUTE whose price depends
    #: on the model it dispatches to — a quoted NEGATIVE price, which is how
    #: OpenRouter spells it on ``openrouter/auto`` and Radient on ``auto``.
    #:
    #: The fourth price state, and a sibling of ``free`` rather than a variant
    #: of it: both exist because the wire said something a pair of floats
    #: cannot record. Where ``free`` distinguishes a quoted zero from silence,
    #: this distinguishes "unknowable by construction" from "nobody quoted it",
    #: which the picker renders as ``usage-based`` and a blank cell.
    #:
    #: Never true beside a positive price and never true beside ``free``: a
    #: route that quotes real money is priced, and a router cannot be free at
    #: the point of use when its own listing declines to price it.
    routed: bool = False
    supports_images: bool | None = None
    supports_prompt_cache: bool = False
    #: The effort ladder the LISTING stated, ASCENDING and normalised to
    #: ``EFFORT_ORDER`` at ingest (see :func:`_effort_ladder`). Three-state —
    #: see the class docstring. Deliberately shares its name with
    #: ``ModelSpec.reasoning_efforts`` even though the two differ in exactly one
    #: way: on the SPEC, ``()`` means "no knob" because the spec is the resolved
    #: answer and has no silence left to record. Nothing may assign one to the
    #: other without going through the resolution that collapses the third state.
    reasoning_efforts: tuple[str, ...] | None = None
    #: The rung the listing says the model runs at when nothing is sent, or
    #: ``None`` when unstated. Only meaningful beside a populated ladder, and
    #: the parser drops it when it is not a member of that ladder — a default
    #: off its own ladder could be neither selected by ``/effort`` nor reached
    #: by ``shift+tab``, so seeding it would strand the band on a level the
    #: cycle can never return to.
    reasoning_default_effort: str | None = None


class _ListingUnavailable(RuntimeError):
    """Raised inside the ``read_listing`` thunk when a live listing failed.

    ``read_listing`` chooses between a stale document and ``None`` by catching
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


def _stated_zero_price(value: object) -> bool:
    """Whether ``value`` is a price the wire explicitly quoted as ZERO.

    The counterpart to :func:`_positive_float`, which deliberately collapses
    "absent", "zero" and "nonsense" into one falsy answer. That collapse is right
    for arithmetic and wrong for display, so this reads the one bit it discards:
    did the source actually write a zero here?

    OpenRouter is why the distinction has to be read off the WIRE rather than
    inferred later. Its listing spells three different things in this field, and
    two of them arrive as ``0.0`` once :func:`_positive_float` is done:
    ``"0"`` for a model that is genuinely free at the point of use
    (``google/gemma-4-31b-it:free``), ``"-1"`` for a meta-route whose cost
    depends on which model it picks (``openrouter/auto``), and an absent leg for
    a listing that quoted nothing. Only the first may ever render as ``free``;
    reading a negative as a statement would put that word on a router whose real
    cost is unknowable, which is the one error this column must not make.
    """
    if isinstance(value, bool):
        # bool is an int subclass, and ``False`` is not a quoted price.
        return False
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return False
    else:
        return False
    return math.isfinite(number) and number == 0.0


def _stated_routed_price(value: object) -> bool:
    """Whether ``value`` is the META-ROUTE sentinel: a NEGATIVE quoted price.

    The fourth thing this field can mean, beside the three
    :func:`_stated_zero_price` enumerates. OpenRouter writes ``"-1"`` on
    ``openrouter/auto`` and Radient writes it on ``auto``: the endpoint is a
    router, so its cost is neither a number nor zero nor unknown — it is
    whatever the model it dispatches to charges, which cannot be known until
    the turn is routed.

    Read off the WIRE for the same reason the stated zero is: only the parser
    that saw the character ``-`` can tell this apart from a listing that
    quoted nothing, since :func:`_positive_float` collapses both to ``0.0``.
    Inferring it downstream by pattern-matching an id would put the label on
    any future model whose name happens to end in ``auto`` and miss every
    router that does not.

    NOT merged with the stated zero into one "unpriced" answer. They render
    differently on purpose — ``free`` is a promise the user can act on and
    ``usage-based`` is a warning that they cannot — and a router quoted at
    ``"0"`` by a server that has not been fixed yet is exactly the row that
    must not read as free.
    """
    if isinstance(value, bool):
        # bool is an int subclass, and ``False`` is not a quoted price.
        return False
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return False
    else:
        return False
    return math.isfinite(number) and number < 0.0


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


def _effort_ladder(value: object) -> tuple[str, ...] | None:
    """A listing's effort list as an ASCENDING ladder, or ``None`` when unstated.

    Sorted HERE, at ingest, rather than at each reader. ``EFFORT_ORDER`` is the
    one place the word order is defined and ``ModelSpec.reasoning_efforts`` is
    contractually ascending — ``next_effort`` indexes it and the loop's retreat
    path walks it downward — so a row that reaches the picker, the merge and the
    spec builder unsorted would be normalised three times and disagree once.
    Sorting also makes a cache round-trip an identity instead of a re-sort.

    Sorted, not reversed, even though every row on the wire arrives strictly
    descending today: a reverse is a bet on the wire's ordering and a sort is
    not.

    DEDUPED on the way through, via the ``set`` the sort reads from. A listing
    that repeats a rung (or states ``High`` beside ``high``, since the words are
    lowercased first) would otherwise put a duplicate on the ladder, and the
    ladder is a CYCLE: ``next_effort`` steps by index, so a repeated rung is a
    ``shift+tab`` that appears not to move. Collapsing here keeps that a
    property of ingest rather than something each reader has to defend against.

    Words outside ``EFFORT_ORDER`` are DROPPED rather than kept or passed
    through. The nearest-rung clamp indexes ``EFFORT_ORDER`` for every rung of
    the ladder it is clamping toward, and an unknown word made that raise —
    on a failover hop, i.e. on the request meant to rescue a turn. Dropping
    costs one rung on a model whose vocabulary grew; keeping costs the turn.
    Extending ``EFFORT_ORDER`` at runtime is not the alternative it looks like:
    its POSITIONS encode the semantic ordering the clamp depends on, and a word
    arriving from a listing carries no information about where it belongs. A new
    rung is a human decision.

    A list that is ENTIRELY unknown words returns ``None`` (unstated), not
    ``()``: ``()`` is a denial this codebase does not want to invent, and the
    table is a better answer than nothing.
    """
    if not isinstance(value, (list, tuple)):
        return None
    known = {
        word.lower() for word in value if isinstance(word, str) and word.lower() in EFFORT_ORDER
    }
    if not known:
        return None
    return tuple(sorted(known, key=EFFORT_ORDER.index))


def _effort_default(value: object, ladder: tuple[str, ...] | None) -> str | None:
    """The listing's default rung, but only when it is ON the stated ladder.

    A default the ladder does not contain can be neither selected by ``/effort``
    nor reached by ``shift+tab``, so seeding it would put a level on the status
    band that the cycle can never return to. No row on the wire violates this
    today (0 of 153 measured); the guard is for the day one does — including the
    day the rung is real but got dropped above as unrankable.
    """
    if not ladder or not isinstance(value, str):
        return None
    lowered = value.lower()
    return lowered if lowered in ladder else None


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


def _bills_in_tokens(architecture: Mapping[str, object]) -> bool:
    """Whether a token price of zero can be this model's WHOLE price.

    A quoted ``0`` per token only means "free" for a model whose product IS
    tokens. OpenRouter prices ``google/lyria-3-pro-preview`` at $0.08 per SONG
    and ``google/lyria-3-clip-preview`` at $0.04 per CLIP, and quotes
    ``{"prompt": "0", "completion": "0"}`` for both — not because they are free
    but because the leg that bills is not denominated in tokens at all. The
    symmetric zero there is a silence about the charge, and reading it as
    ``free`` advertises a paid model as free, the one error
    ``format_price_pair`` exists to prevent.

    The wire carries the discriminator: ``architecture.output_modalities`` is
    ``["text", "audio"]`` on exactly those two rows against ``["text"]`` on all
    19 genuinely-free routes in the live listing. Gating on the modality closes
    the whole class — any future image/audio/video generator priced per artifact
    is covered — rather than naming two ids that go stale.

    ABSENT modalities — the key missing, or an empty list, which is how the
    models.dev projection normalises "said nothing" — default to text
    (permissive), and that direction is deliberate. Every lean
    OpenAI-compatible gateway sends an id and little else, and the same
    reasoning :func:`_has_image_input` documents applies: silence is not a
    statement. The conservative default would strip ``free`` from every
    genuinely-free route on any gateway that omits the field — a large, silent
    regression against a hypothetical one, where the permissive default is wrong
    only for a gateway that both omits modalities AND quotes a token zero for a
    non-token-billed model. OpenRouter, the only transport that quotes zeros at
    all today, states the field on 100% of its rows.
    """
    modalities = architecture.get("output_modalities")
    if isinstance(modalities, (list, tuple)) and modalities:
        return all(isinstance(item, str) and item.strip().lower() == "text" for item in modalities)
    modality = architecture.get("modality")
    if isinstance(modality, str) and "->" in modality:
        # The older encoding packs the same fact to the RIGHT of the arrow.
        # Only the output side decides what the model bills for.
        # Lowercased like the list branch above and like ``_has_image_input``'s
        # own arrow branch: the encoding is the gateway's, not a normalised
        # field, so ``text->TEXT`` must not read as a non-text output and strip
        # ``free`` from a genuinely free model.
        return all(
            part.strip().lower() == "text" for part in modality.split("->")[-1].split("+") if part
        )
    return True


#: Listing ids that ARE a router rather than a model, by aggregator spelling.
#:
#: Deliberately tiny and deliberately here rather than in the renderer. These
#: are not "models we know about" — enumerating those is what the live listing
#: is for — they are the two endpoints whose entire product is dispatching to
#: another model, which is a structural property of the aggregator's API and
#: changes only when the aggregator adds a router.
_META_ROUTE_IDS = frozenset({"auto", "openrouter/auto"})


def is_meta_route_id(model_id: str, provider_id: str) -> bool:
    """Whether ``provider_id``/``model_id`` names a ROUTER rather than a model.

    The id half of :func:`_is_meta_route`, exported for the one caller that
    has no listing row to read a price from: the controller's rescue entry,
    which describes the session's CURRENT model from the registry when a live
    listing could not be had. That path is reached exactly when the router's
    own pricing is unavailable, so the price signal it would prefer does not
    exist there — and it is the path a user on ``radient/auto`` with a cold
    cache actually hits.

    Takes the provider for the same reason :func:`_is_meta_route` does, and it
    is not hypothetical here either: ``ollama/auto`` reaches the rescue entry
    by exactly the same route as ``radient/auto``, so an ungated id test would
    mislabel a local model the user happened to name ``auto``.

    Kept as one exported predicate over one module-private set so the two
    layers cannot drift into disagreeing about what a router is.
    """
    return model_id in _META_ROUTE_IDS and provider_id in AGGREGATOR_PROVIDERS


def _is_meta_route(model_id: str, provider_id: str, pricing: Mapping[str, object]) -> bool:
    """Whether this entry is a ROUTER whose cost depends on where it dispatches.

    Two signals, either of which is sufficient, and the second exists only
    because of a bug the first cannot see through.

    The PRICE is the principled signal: a quoted negative leg means exactly
    this and nothing else (:func:`_stated_routed_price`), it is what OpenRouter
    publishes for ``openrouter/auto``, and it keeps working for any router any
    aggregator adds later without this module being told about it. It needs no
    provider gate — a quoted negative is self-describing wherever it appears.

    The ID is the fallback, and it is gated on the provider being an
    AGGREGATOR. Radient's listing currently declares ``auto`` with
    ``{"prompt": "0", "completion": "0"}``, and a symmetric quoted zero is
    INDISTINGUISHABLE from a genuinely free route by price alone — that is the
    whole reason ``_stated_zero_price`` exists. So for that one shape there is
    no principled signal to read, and the honest options are to name the router
    or to let it keep claiming to be free.

    WHY the gate is load-bearing rather than defensive: this parser serves the
    ``openai-compat`` wire GENERALLY, not just the two aggregators — ollama,
    deepseek, mistral, kimi, xai, zai, alibaba and openai all reach it. On
    Ollama the "listing" is the user's own filesystem, so the id space is
    user-controlled and a local model named ``auto`` is a thing a user can
    simply have. Ungated, that row rendered ``usage-based`` where it had
    correctly read ``free`` — and ``ollama`` is the ONE provider with
    ``allows_missing_api_key``, whose genuine zero is exactly what ``_price``
    exists to preserve. So the mislabel landed on the honest-pricing axis this
    module is most careful about.

    The blast radius was the PRICE CELL ONLY, and the bound is worth stating so
    nobody re-derives it: ``get_model_info`` dispatches the router's 1M row on
    its own aggregator-scoped set, so no session ever ran on wrong metadata —
    a local model kept its real window and its real image capability. A
    cosmetic mislabel on the one column that must not lie, rather than a
    correctness bug in the spec.

    The set is checked against the LISTING id, which is the id as that
    aggregator spells it (``auto`` on Radient, ``openrouter/auto`` on
    OpenRouter). The fallback expires on its own: once the Radient server
    quotes ``"-1"`` (radient-ml/agent-server #8) and cached listings have
    rolled past the capture bump, the price leg carries the row and the id leg
    can be deleted.
    """
    if model_id in _META_ROUTE_IDS and provider_id in AGGREGATOR_PROVIDERS:
        return True
    # BOTH legs, matching the stated-zero rule directly above it: a
    # half-negative row is a listing bug, not a router, and reading one
    # leg would let it suppress a real price on the other.
    return _stated_routed_price(pricing.get("prompt")) and _stated_routed_price(
        pricing.get("completion")
    )


def _row_from_openai_entry(
    entry: Mapping[str, object], provider_id: str = ""
) -> DiscoveredModel | None:
    """One OpenAI-compatible listing entry, or ``None`` when it has no id.

    ``provider_id`` says whose wire this entry came off, which only
    :func:`_is_meta_route`'s router-id fallback needs. It DEFAULTS TO EMPTY and
    that direction is deliberate: ``""`` is not in ``AGGREGATOR_PROVIDERS``, so
    a caller that omits it gets the price-only answer — the conservative one —
    rather than a router label it did not ask for. The alternative, a required
    argument, would fail closed too but at import time for every existing
    caller; this fails closed at the only place the value changes an answer.

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
    reasoning = _mapping(entry.get("reasoning"))

    cache_read_price = _per_million(pricing.get("input_cache_read"))
    effort_ladder = _effort_ladder(reasoning.get("supported_efforts"))
    return DiscoveredModel(
        id=model_id,
        name=_first_str(entry.get("name"), entry.get("display_name")),
        # OpenRouter also publishes ``input_cache_write_1h`` for the one-hour
        # Anthropic tier; the five-minute rate is the one every write is billed
        # at unless a caller asks otherwise, so that is the one carried.
        cache_write_price=_per_million(pricing.get("input_cache_write")),
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
        # BOTH legs, and only when the wire wrote a zero in each. A half-stated
        # row (``prompt: 0`` beside a priced ``completion``) is a model you pay
        # for, and calling it free would understate it by the whole output bill.
        # The modality gate is the same guard one level up: a zero per TOKEN is
        # only a whole price for a model that bills in tokens (see
        # :func:`_bills_in_tokens`).
        # A router is never free, whichever way its listing spells the
        # non-price. The ``routed`` test below claims the row first, so a
        # meta-route quoting ``"0"`` — which is what Radient's listing does
        # for ``auto`` today — drops out of this expression rather than
        # advertising a frontier-model dispatch as costing nothing.
        free=(
            not _is_meta_route(model_id, provider_id, pricing)
            and _stated_zero_price(pricing.get("prompt"))
            and _stated_zero_price(pricing.get("completion"))
            and _bills_in_tokens(architecture)
        ),
        routed=_is_meta_route(model_id, provider_id, pricing),
        cache_read_price=cache_read_price,
        supports_images=_has_image_input(architecture),
        # A priced cache-read leg is the only machine-readable evidence of prompt
        # caching in these listings; there is no capability flag for it.
        supports_prompt_cache=cache_read_price > 0,
        # The router's own statement about which efforts it accepts for this
        # model, which is the thing that returns 400 when we get it wrong. Only
        # the OpenAI-compatible AGGREGATORS publish it (`_row_from_gemini_entry`
        # and `_fetch_anthropic` are untouched: neither wire carries the field),
        # and it overrules `model.effort`'s table because that table is
        # second-hand by construction and its one extrapolating arm — one
        # transcribed `gpt-5.4` ladder applied to every `gpt-[5-9]` id — is
        # measurably wrong 34 times in BOTH directions against this listing.
        reasoning_efforts=effort_ladder,
        reasoning_default_effort=_effort_default(reasoning.get("default_effort"), effort_ladder),
    )


def _serves_account_scoped_catalogue(
    provider_id: str,
    *,
    is_oauth: bool,
    api_key: str | None,
    account_id: str | None,
) -> bool:
    """Whether a listing for ``provider_id`` is ChatGPT's account-scoped one.

    ONE predicate, read by all three places that need the answer: the
    transport that chooses the endpoint, the cache key that isolates the
    document, and the merge that decides whether the listing may prune
    registry ids. They used to be spelled separately and disagreed — the
    prune test omitted ``account_id`` and keyed off the credential-storage id
    rather than the provider actually fetched, so the weakest of the three
    was the only destructive one. An OAuth run with no account scope then
    read the shared API-KEY cache document (``_cache_key`` degrades to the
    unscoped name without an account id) and pruned the registry against a
    listing written under a different credential, taking 11 shipped OpenAI
    ids down to 1 with no request issued. A pre-change document parses
    cleanly too, so an upgrade hit the same path.

    Authority is therefore defined once, by the conditions under which the
    account-scoped request is actually issuable, and every site derives from
    it instead of restating it.
    """
    return bool(
        is_oauth and provider_id in OPENAI_OAUTH_PROVIDERS and api_key and account_id,
    )


def _row_from_openai_codex_entry(entry: Mapping[str, object]) -> DiscoveredModel | None:
    """One selectable model from ChatGPT's account-scoped Codex catalogue.

    This is deliberately separate from the public OpenAI-compatible shape:
    Codex addresses a model by ``slug`` and places modalities at the top level.

    Both the id and the visibility test are deliberately permissive, because
    this listing is now allowed to PRUNE the registry: a schema change at an
    endpoint nobody here controls used to cost "no new models" and would
    otherwise cost the whole OpenAI picker. So ``id`` is accepted as a
    fallback spelling of ``slug`` (as in the public shape), and a row is
    dropped only for a visibility value KNOWN to mean hidden. An unrecognised
    value leaves the row listed: showing one internal helper model is a far
    smaller harm than dropping every row and emptying the picker.
    """
    visibility = _first_str(entry.get("visibility")).casefold()
    if visibility in _OPENAI_CODEX_HIDDEN_VISIBILITY:
        return None
    model_id = _first_str(entry.get("slug")) or _first_str(entry.get("id"))
    if not model_id:
        return None
    return DiscoveredModel(
        id=model_id,
        name=_first_str(entry.get("display_name")),
        context_window=_first_positive_int(
            entry.get("context_window"),
            entry.get("max_context_window"),
        ),
        max_tokens=_first_positive_int(
            entry.get("max_tokens"),
            entry.get("max_output_tokens"),
        ),
        supports_images=_has_image_input({"input_modalities": entry.get("input_modalities")}),
    )


def _fetch_openai_oauth(ctx: _FetchContext) -> list[DiscoveredModel] | None:
    """The current ChatGPT/Codex models available to one logged-in account."""
    if not ctx.api_key or not ctx.account_id:
        # The account header is part of the authorization boundary. A token
        # without it cannot safely ask which workspace's models are available.
        return None
    body = _get_json(
        ctx,
        OPENAI_CHATGPT_MODELS_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {ctx.api_key}",
            "ChatGPT-Account-ID": ctx.account_id,
        },
        params={"client_version": OPENAI_MODELS_CLIENT_VERSION},
    )
    if body is None:
        return None
    entries = _entry_list(body, "models")
    if entries is None:
        return None
    rows = [row for row in (_row_from_openai_codex_entry(e) for e in entries) if row is not None]
    if not rows:
        # A 200 that yields nothing selectable is indistinguishable from "this
        # account has no models" at the call site, and the two have very
        # different remedies. The pinned client version is the likeliest cause
        # of a filtered catalogue, so it is named here: without it, a listing
        # filtered by a version this package pins is invisible in the logs.
        logger.warning(
            "openai model listing at %s returned %d entries but no selectable "
            "models (client_version=%s); keeping the bundled catalogue",
            OPENAI_CHATGPT_MODELS_URL,
            len(entries),
            OPENAI_MODELS_CLIENT_VERSION,
        )
    return rows


def _fetch_openai_compat(ctx: _FetchContext) -> list[DiscoveredModel] | None:
    """The provider's model catalogue, including ChatGPT's OAuth-only route."""
    if _serves_account_scoped_catalogue(
        ctx.provider_id,
        is_oauth=ctx.is_oauth,
        api_key=ctx.api_key,
        account_id=ctx.account_id,
    ):
        return _fetch_openai_oauth(ctx)
    if ctx.is_oauth and ctx.provider_id in OPENAI_OAUTH_PROVIDERS:
        # An OAuth token for these providers cannot be spent on
        # ``api.openai.com/v1/models`` (it rejects subscription credentials),
        # and without an account scope the Codex catalogue is not askable
        # either. Falling through to the generic wire would spend the token on
        # a request that answers 401 and read as a listing outage; saying "no
        # listing" keeps the bundled catalogue and issues no request.
        return None

    headers = {"Accept": "application/json"}
    if ctx.api_key:
        # Bearer serves both API keys and OAuth access tokens on this wire, so
        # ``is_oauth`` needs no branch here; Anthropic is the wire that differs.
        headers["Authorization"] = f"Bearer {ctx.api_key}"
    body = _get_json(ctx, f"{ctx.base_url}/models", headers=headers, params={})
    if body is None:
        return None
    entries = _entry_list(body, "data", "models")
    if entries is None:
        return None
    # The provider is threaded in because the ROUTER-id fallback in
    # ``_is_meta_route`` is only valid for an aggregator: this same parser
    # serves every ``openai-compat`` provider, and on Ollama the listing is the
    # user's own filesystem (see R1).
    rows = (_row_from_openai_entry(entry, ctx.provider_id) for entry in entries)
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
    return static_models(credential_provider_id(definition.id))


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
            scheme on the Anthropic wire and selects ChatGPT's Codex catalogue
            for OpenAI instead of ``api.openai.com/v1/models``.
        account_id: ChatGPT account scope required by the Codex listing. Other
            providers ignore it.
        base_url: Overrides the registry base (proxies, local gateways).
        client: Reused HTTP client. When omitted, one is created and closed here;
            a caller listing several providers should pass one so the connection
            pool and TLS handshake are shared.
        timeout: Ceiling in seconds for the whole listing. Google's transport
            paginates and spends it as a single deadline; every other transport
            issues one request, so for them it is also the per-request ceiling.
    """
    definition = get_provider_definition(provider_id)
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
    *,
    include_static_only: bool = True,
) -> list[DiscoveredModel]:
    """Fold a listing into the registry without losing known per-model metadata.

    Most provider listings are incomplete entitlement snapshots, so their static
    ids remain reachable by default. A caller with an authoritative account-scoped
    catalogue can set ``include_static_only=False``: listed ids still inherit
    missing specs, but obsolete registry-only ids are not presented as available.
    Live rows always come first because providers list newest-first.
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

    if include_static_only:
        for model_id, info in static_rows.items():
            if model_id in seen:
                continue
            seen.add(model_id)
            # Static ids are retained for non-authoritative listings: gateways
            # filter by entitlement and may omit a model the user runs today.
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
        cache_write_price=_positive_float(info.cache_writes_price),
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
        # the case this module exists to surface — and exactly the case the
        # output-cap guard is FOR, since there is no bundled value to fall back
        # on and nothing else downstream re-examines the number. muse-spark
        # reaches the spec through this branch.
        sane_max = sane_listing_max_tokens(
            _positive_int(row.max_tokens), _positive_int(row.context_window)
        )
        if sane_max == row.max_tokens:
            return row
        return dataclasses.replace(row, max_tokens=sane_max)

    # Reduced BEFORE the 4096 comparison below, and before it can win over the
    # static value: an implausible live cap is a bad answer whichever branch
    # consumes it. The window used is the one this merge will actually publish,
    # so the ratio is judged against the same pair that ends up on the row.
    #
    # One coincidence worth naming: on an 8192 window a 0.9 cap (7372) reduces to
    # exactly 4096, so a reduced value can LAND ON the lying-default sentinel and
    # hand the branch below to a larger bundled cap. That outcome is correct
    # rather than merely tolerable — a hand-transcribed registry entry states the
    # model's documented maximum, which is better information than either the
    # gateway's formula or this guard's halving.
    merged_window = _positive_int(row.context_window) or _positive_int(info.context_window)
    live_max = sane_listing_max_tokens(_positive_int(row.max_tokens), merged_window)
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
        # The LIVE listing is the only source that can state this, and it only
        # survives when the registry does not then supply a price: the two lines
        # above fall back to the bundled numbers for a silent listing, and a row
        # that ends up quoting $3/15 must not also claim to be free. The
        # registry has no field of its own here — a bundled row cannot say
        # "free", only "priced" or "unknown".
        free=row.free
        and not (_positive_float(info.input_price) or _positive_float(info.output_price)),
        # Same shape and same reason as ``free`` above: only the LIVE listing
        # can state it (no bundled row describes a router's pricing), and it
        # does not survive a registry price. The aggregator templates carry no
        # prices, so in practice this passes straight through — but a future
        # priced row must not be able to quote $3/15 and claim to be
        # unpriceable at the same time, which is the invariant, not the
        # current data.
        routed=row.routed
        and not (_positive_float(info.input_price) or _positive_float(info.output_price)),
        cache_read_price=(
            _positive_float(row.cache_read_price) or _positive_float(info.cache_reads_price)
        ),
        cache_write_price=(
            _positive_float(row.cache_write_price) or _positive_float(info.cache_writes_price)
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
        # Passed through, and that is not the no-op it reads as: this function
        # CONSTRUCTS a fresh row, so an omitted field silently defaults — the
        # exact class of bug the comments above keep recording. There is nothing
        # to merge against, because the registry has no ladder of its own: no
        # bundled row can state one, so silence here leaves ``None`` and the
        # SPEC BUILDER falls back to ``model.effort``. The merge deliberately
        # does not consult that table itself, so a row stays a faithful record
        # of what the wire said and exactly one function owns the fallback.
        reasoning_efforts=row.reasoning_efforts,
        reasoning_default_effort=row.reasoning_default_effort,
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


def _cache_key(
    storage_id: str,
    *,
    account_scoped: bool = False,
    account_id: str | None = None,
    host_scoped: bool = False,
) -> str:
    """Cache document name, isolated by credential scope where required.

    ``.listing`` avoids the incompatible legacy ``<provider>.models.json``
    layout. OpenAI's OAuth catalogue is account-scoped and has a different
    schema from its API-key catalogue, so it gets a hashed account suffix: an
    API-key cache or another workspace must never decide which models this
    account can select. The raw account id is intentionally absent from disk.

    ``storage_id`` is the credential-storage identity rather than the provider
    id, so ``openai-device`` shares one document with ``openai`` — they are one
    logged-in account and listing twice under two names would ask the same
    endpoint for the same answer. ``account_scoped`` is passed in rather than
    re-derived here: :func:`_serves_account_scoped_catalogue` is the single
    place that decides it, and a second spelling of the same test is how the
    scoping and the pruning came to disagree.
    """
    if account_scoped and account_id:
        account_scope = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:12]
        return f"{storage_id}.oauth.{account_scope}.listing"
    if host_scoped:
        # A provider whose OAuth sign-in is served by a different HOST than its
        # API key (``oauth_base_url``; Kimi's coding plan) returns a genuinely
        # different catalogue per credential kind -- the subscription lists
        # ``k3``, the mainland API-key platform does not. One document for both
        # would let whichever ran last decide what the other kind can select.
        return f"{storage_id}.oauth.listing"
    return f"{storage_id}.listing"


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
        # Computed ONCE and handed to both fields: the default is only valid
        # against its own ladder, so re-deriving it would let the two disagree.
        stored_ladder = _effort_ladder(entry.get("reasoning_efforts"))
        rows.append(
            DiscoveredModel(
                id=model_id,
                name=_first_str(entry.get("name")),
                context_window=_positive_int(entry.get("context_window")),
                max_tokens=_positive_int(entry.get("max_tokens")),
                input_price=_positive_float(entry.get("input_price")),
                output_price=_positive_float(entry.get("output_price")),
                cache_read_price=_positive_float(entry.get("cache_read_price")),
                cache_write_price=_positive_float(entry.get("cache_write_price")),
                # A stored ``false`` and an absent key mean the same thing here
                # (not stated free), so a plain bool is enough — unlike
                # ``supports_images`` below, this field has no third state.
                free=bool(entry.get("free")),
                # Stored as a computed boolean exactly like ``free``, and read
                # back the same way: a stored ``false`` and an absent key both
                # mean "not a router". A document written before this field
                # existed therefore reads as ``false``, which is why the
                # capture stamp is bumped rather than left to the TTL.
                routed=bool(entry.get("routed")),
                # ``null`` in the document is the listing's silence, faithfully
                # stored by ``dataclasses.asdict``. Reading it as False would let
                # a cache round-trip turn "unstated" into a denial, so the same
                # model would resolve differently live than from disk.
                supports_images=_stated_bool(entry.get("supports_images")),
                supports_prompt_cache=bool(entry.get("supports_prompt_cache")),
                # The SAME coercers the parser used, which is what makes the
                # round-trip faithful rather than merely plausible: a stored
                # ``null`` reads back as ``None`` (the listing's silence, which
                # the table then answers) instead of as ``()``, and a stored
                # list re-sorts and re-filters identically. Reading ``null`` as
                # ``()`` would turn silence into a denial and strip the table's
                # answer — the same trap ``supports_images`` documents two
                # fields up, where the cost was a model resolving differently
                # from disk than it did live.
                reasoning_efforts=stored_ladder,
                reasoning_default_effort=_effort_default(
                    entry.get("reasoning_default_effort"), stored_ladder
                ),
            )
        )
    return rows


def invalidate_listing(provider_id: str, *, cache_dir: Path | None = None) -> int:
    """Drop ``provider_id``'s cached listing so the next call refetches.

    For callers reacting to an event that can CHANGE WHAT THE LISTING RETURNS
    but that no TTL can observe -- a login, a re-auth, a logout. The picker's
    15-minute TTL and the hourly background revalidation bound ordinary drift,
    but neither knows that the CREDENTIAL changed: a different account or plan
    can list a different catalogue, and a listing fetched anonymously (or under
    the account just removed) must not decide what the next credential can
    select.

    Keyed on the CREDENTIAL identity, not the provider id, because that is what
    names the documents: ``openai-device`` and ``openai`` are one logged-in
    account sharing one document, and invalidating under the literal id would
    miss it. Returns how many documents were dropped -- 0 is the ordinary answer
    for a provider that was never listed, not an error.
    """
    definition = get_provider_definition(provider_id)
    storage_id = credential_provider_id(definition.id if definition else provider_id)
    return invalidate_documents(storage_id, cache_dir=cache_dir)


def cached_available_models(
    provider_id: str,
    *,
    cache_dir: Path | None = None,
) -> tuple[list[DiscoveredModel], ListingStatus]:
    """Every model cached on disk for ``provider_id`` with zero network calls.

    Synchronous, non-blocking, and I/O-isolated: uses :func:`peek_listing` so it
    never acquires fetch leases, spawns background revalidation threads, or
    attempts network requests.

    Returns:
        ``(models, status)`` where status is ``"cached"`` if a valid cached
        listing was found on disk, or ``"static"`` if no cache exists or the
        cache payload is unusable, in which case the bundled static registry rows
        are returned as fallback.
    """
    rows = _static_rows(provider_id)
    definition = get_provider_definition(provider_id)
    if definition is None:
        return merge_models(rows, None), "static"

    storage_id = credential_provider_id(definition.id)
    key = _cache_key(storage_id)
    listing = peek_listing(key, cache_dir=cache_dir)
    capture = listing_capture_version(storage_id)
    live_rows = _rows_from_payload(listing.payload, capture)
    if not live_rows:
        return merge_models(rows, None), "static"

    return merge_models(rows, live_rows, include_static_only=True), "cached"


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
    want_id: str | None = None,
) -> tuple[list[DiscoveredModel], ListingStatus]:
    """Every model a UI should offer for ``provider_id``, plus how it was obtained.

    A provider outage falls back to the registry, so failed discovery never
    prevents selection. A successful authoritative catalogue may intentionally
    replace registry-only ids. This never raises and never blocks longer than
    ``timeout``; a fresh cache skips the request entirely.

    ``ttl_s`` is the HARD TTL beyond which the call blocks on a fetch. A
    document older than ``catalogue.SOFT_TTL_S`` but inside it is served as-is
    and refreshed in the background (stale-while-revalidate), so callers on a
    boot or paint path pay nothing for freshness; a caller that wants a fresher
    answer NOW (the model picker) passes a shorter ``ttl_s``.

    ``want_id`` is the id the caller is about to look up. If it is absent from a
    stored document at least ``catalogue.MISS_REFETCH_MIN_AGE_S`` old, the
    document is refetched ONCE, synchronously, within ``timeout`` — the one
    trigger that beats every TTL for a model released this morning, because it
    fires for exactly the id the user asked for. A young document that lacks the
    id is believed (a typo must not refetch on every resolution), and the
    refetched document is zero seconds old, so a second miss is believed too.

    Returns:
        ``(models, status)``, where status is ``"ok"`` (fetched live just now),
        ``"cached"`` (served a stored document with no fetch needed), ``"stale"``
        (a fetch was attempted and failed; the stored document is what you got),
        ``"static"`` (registry only -- no listing endpoint, or no cache and no
        successful fetch), ``"unauthenticated"`` (the provider needs a credential
        to list and none was supplied) or ``"empty"`` (the provider answered and
        listed no models).
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
            want_id=want_id,
        )
    except Exception as exc:  # noqa: BLE001 - a broken provider is an annotation
        # The layers below already swallow transport failures, so reaching here
        # means a cache or coding fault. Even then the picker has to open: the
        # registry alone is a usable answer, a traceback is not.
        logger.debug("falling back to the static registry for %s: %s", provider_id, exc)
        return merge_models(rows, None), "static"


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
    want_id: str | None = None,
) -> tuple[list[DiscoveredModel], ListingStatus]:
    definition = get_provider_definition(provider_id)
    transport = _TRANSPORTS.get(definition.id) if definition is not None else None
    if definition is None or transport is None:
        return merge_models(rows, None), "static"

    # An OAuth credential may be served by a different host than the provider's
    # API-key base (Kimi's coding plan; see ``oauth_base_url``). An explicit
    # caller-supplied base still wins, because that is a deliberate override.
    resolved_base = base_url or (
        (definition.oauth_base_url if is_oauth else None) or definition.base_url
    )
    if not resolved_base:
        return merge_models(rows, None), "static"

    keyless_listing = definition.allows_missing_api_key or definition.id in PUBLIC_LISTING_PROVIDERS
    if not api_key and not keyless_listing:
        # Reported apart from "static" because it is the one status the user can
        # act on: log in and the listing appears. The keyless exceptions are local
        # servers (Ollama needs no credential at all) and the aggregators, whose
        # catalogue is a public page even though inference is not.
        return merge_models(rows, None), "unauthenticated"

    capture = listing_capture_version(definition.id)

    def fetch_within(ceiling: float) -> dict[str, Any]:
        live = fetch_models(
            definition.id,
            api_key=api_key,
            is_oauth=is_oauth,
            account_id=account_id,
            base_url=resolved_base,
            client=client,
            timeout=ceiling,
        )
        if live is None:
            raise _ListingUnavailable(definition.id)
        return {
            "capture": capture,
            "models": [dataclasses.asdict(row) for row in live],
        }

    def fetch() -> dict[str, Any]:
        # On the calling path: the caller's budget, which from a repaint is 2 s.
        return fetch_within(timeout)

    def revalidate() -> dict[str, Any]:
        # Off the calling path, so the caller's budget is the wrong ceiling: a
        # background refresh that inherited 2 s failed on every link slower
        # than that, backed off, and left the document to the 24 h sync path.
        # The provider's full default is what an unhurried listing gets, the
        # same choice ``prices._price_catalogue_row`` makes for its retry.
        return fetch_within(DEFAULT_TIMEOUT_S)

    storage_id = credential_provider_id(definition.id)
    # Derived ONCE and reused for both the document name and the pruning
    # decision, so the two cannot disagree about whether this listing is the
    # account's authoritative catalogue.
    account_scoped = _serves_account_scoped_catalogue(
        definition.id,
        is_oauth=is_oauth,
        api_key=api_key,
        account_id=account_id,
    )
    key = _cache_key(
        storage_id,
        account_scoped=account_scoped,
        account_id=account_id,
        host_scoped=bool(is_oauth and definition.oauth_base_url),
    )
    # The soft TTL never exceeds the hard one: a picker asking for a 15-minute
    # document wants it fetched NOW, not served-and-refreshed-later.
    soft_ttl_s = min(SOFT_TTL_S, ttl_s)

    def lacks_want_id(document: Mapping[str, object], age_s: float) -> bool:
        # The id the caller is about to resolve is not in a document old enough
        # to predate it: declare the document expired so `read_listing` fetches
        # ONCE, synchronously, under the lease and inside the caller's budget.
        # Not a loop: the refetched document is 0s old, so the next miss for the
        # same id is believed for ten minutes. A young document that lacks the
        # id is right (a typo), and a document this reader cannot map is handled
        # by the invalidate-and-re-enter below rather than here.
        if want_id is None or age_s < MISS_REFETCH_MIN_AGE_S:
            return False
        mapped = _rows_from_payload(document, capture)
        return mapped is not None and not _lists_id(mapped, want_id)

    listing = read_listing(
        key,
        fetch,
        soft_ttl_s=soft_ttl_s,
        ttl_s=ttl_s,
        cache_dir=cache_dir,
        refetch_if=lacks_want_id,
        revalidate=revalidate,
    )
    payload = listing.payload
    live_rows = _rows_from_payload(payload, capture)
    if live_rows is None:
        if payload is not None:
            # A document `read_listing` could read but this reader cannot use:
            # a `models` key that is not an array (a payload-shape change, a
            # truncated write) or a capture older than the transports that will
            # read it. It is written before anything interprets it, so leaving it
            # in place serves the same failure as a FRESH cache hit on every start
            # until the TTL expires: a planted document with a dict `models`
            # produced three consecutive `static` results and zero fetches.
            # Dropping it costs one refetch.
            invalidate(key, cache_dir=cache_dir)
            # RE-ENTER once. Dropping the document without retrying meant the
            # fetch that was supposed to replace it never ran: `read_listing`
            # had already served this call from a fresh hit, so the thunk was
            # never invoked, and the answer fell through to the registry's static
            # rows — of which an aggregator has NONE. Offline that is an empty
            # model list on every start; online it memoises a session booted at
            # default context, no prompt cache and zero prices.
            listing = read_listing(
                key,
                fetch,
                soft_ttl_s=soft_ttl_s,
                ttl_s=ttl_s,
                cache_dir=cache_dir,
                revalidate=revalidate,
            )
            payload = listing.payload
            live_rows = _rows_from_payload(payload, capture)
        if live_rows is None:
            # Neither a listing nor a cache: the registry is all there is.
            return merge_models(rows, None), "static"

    # An authoritative listing may replace the bundled ids -- but only when it
    # actually listed something. A 200 that parses to nothing (a renamed field,
    # an unrecognised visibility value, a catalogue filtered by the pinned
    # client version) would otherwise prune the registry to EMPTY and cache the
    # emptiness for the 24h TTL, leaving the picker with no OpenAI models at
    # all and no request issued to recover. Before this path existed the same
    # answer still offered every bundled id, which is the behaviour to keep:
    # upstream schema drift should cost "no new models", never "no models".
    authoritative_live = account_scoped and bool(live_rows)
    merged = merge_models(
        rows,
        live_rows,
        include_static_only=not authoritative_live,
    )
    return merged, _status_of(listing, live_rows)


def _status_of(listing: Listing, live_rows: list[DiscoveredModel]) -> ListingStatus:
    """The status a served document earns, from how :func:`read_listing` got it."""
    if listing.fetched:
        return "empty" if not live_rows else "ok"
    if listing.failed:
        return "stale"
    return "cached"


def _lists_id(rows: list[DiscoveredModel], model_id: str) -> bool:
    """Whether ``rows`` already accounts for ``model_id`` under ANY known spelling.

    This answers "could a refetch plausibly list this id?", not "will the lookup
    find a row?" — the two differ, and conflating them put a synchronous listing
    fetch on every process start. Anthropic's ``/v1/models`` lists dated
    snapshots only (``claude-sonnet-4-5-20250929``) while its API accepts the
    undated alias, and ``claude-sonnet-4-5`` is a common configured id. Matched
    exactly or normalised, the alias was a miss against a document that DID
    list its snapshot, so every process older than ``MISS_REFETCH_MIN_AGE_S``
    paid a blocking round trip on boot for a document the refetch could not
    improve — indefinitely, since the refetched document lacked the alias too.

    So a row counts as a hit when the wanted id's LITERAL (normalised) is one
    of the row's :func:`id_spellings` (the id as given, date-stripped, dotted),
    or the row's literal is one of the wanted id's. That covers both directions
    — an alias whose snapshot is listed, and a snapshot whose alias is listed
    (the aggregators' habit) — so neither direction can refetch for a row the
    provider spells differently. The rewrites are the conservative ones
    ``prices._lookup`` already trusts.

    The match is literal-against-spellings on purpose, NOT spellings-against-
    spellings: both sides date-stripped coincide for ANY two snapshots of one
    family, so a document listing only ``claude-opus-4-5-20251101`` counted
    ``claude-opus-4-5-20260315`` as present — the day a new snapshot landed,
    the one trigger built for that case stayed silent, the lookup missed, and
    the memo pinned the fallback for the rest of the day. A literal never loses
    its date, so two dated ids match only when they are the same id.
    """
    if any(row.id == model_id for row in rows):
        return True
    wanted_literal = normalised_id(model_id)
    wanted_spellings = {normalised_id(spelling) for spelling in id_spellings(model_id)}
    for row in rows:
        row_literal = normalised_id(row.id)
        if row_literal in wanted_spellings:
            return True
        if any(normalised_id(spelling) == wanted_literal for spelling in id_spellings(row.id)):
            return True
    return False

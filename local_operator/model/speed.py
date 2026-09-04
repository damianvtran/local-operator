"""Which models can be served at a provider's FAST tier, and how to ask for it.

A leaf module, deliberately shaped like its sibling :mod:`local_operator.model.effort`:
``model.configure`` derives the spec from it, ``providers.failover`` re-derives it
when a fallback swaps the model out, and the TUI reads the RESULT off the spec
(never this table) so no widget carries model-name knowledge.

**Fast mode is not reasoning effort.** The two dials are orthogonal and this
module exists because conflating them would be a lie in both directions.
``effort`` buys *fewer or more thinking tokens* — it changes what the model
does. Fast mode buys the *same tokens sooner*: the same weights, the same
answer distribution, served off faster (and more expensive) capacity. Anthropic
states it outright — "Fast mode runs the same model with a faster inference
configuration. There is no change to intelligence or capabilities" — so a user
who wants a cheaper, shallower answer wants ``/effort low``, and a user who
wants the same answer with less waiting wants ``/fast``. Neither substitutes
for the other, which is why this ships beside ``/effort`` rather than as
another rung on its ladder.

Two facts vary independently, so this module answers them separately:

* **The dialect** — which key carries the request and what value means "fast" —
  is a property of the ROUTE, not of the model. ``anthropic/claude-opus-5``
  reached directly takes ``speed: "fast"`` behind a beta header; the SAME model
  reached through OpenRouter takes ``service_tier: "priority"``, which
  OpenRouter then translates to ``speed: "fast"`` upstream. A model-keyed
  dialect would send Anthropic's spelling down an OpenAI-shaped pipe.
* **Availability** — whether this particular model can be served fast at all —
  is a property of the model within that route.

Every row below was transcribed from the provider's own documentation AND
probed against the live endpoint on 2026-09-04, because on this axis the docs
and the wire disagree in ways that are silent until they are expensive. The
measurements are recorded next to the rows they justify; the three that changed
the design:

* ``service_tier: "fast"`` is NOT universal, even though OpenAI's own Fast mode
  guide leads with it. The ChatGPT/Codex backend answered
  ``HTTP 400 {"detail":"Unsupported service_tier: fast"}`` and xAI answered
  ``HTTP 422 unknown variant `fast`, expected one of `auto`, `default`,
  `flex`, `standard`, `priority```. ``"priority"`` is the value every
  service-tier route accepts — OpenAI documents it as equivalent ("Setting
  service_tier to priority provides the same behavior"), xAI and OpenRouter
  name it as their only fast tier, and the Codex backend parses it — so it is
  the single value sent here. One accepted word beats a per-route value table
  that has to be right four times.
* Anthropic's ``speed`` key REQUIRES its beta header. Without it the request is
  rejected outright (``HTTP 400 "speed: Extra inputs are not permitted"``), so
  the header and the key are written together or not at all.
* Fast mode is ENTITLED, not merely supported. A Claude subscription that
  serves ``claude-opus-5`` perfectly well at standard speed answered
  ``HTTP 429 {'type': 'rate_limit_error', 'message': 'Usage credits are
  required for fast mode.'}``. That is a 429 that does NOT mean "you ran out" —
  see ``providers.failover._is_fast_mode_refusal`` for why that distinction had
  to be taught to the classifier before this feature could ship.

Families NOT here offer nothing, which is the honest answer rather than a gap.
Google (Vertex/AI Studio) does sell priority capacity, but our ``GoogleClient``
speaks ``generateContent``, whose request has no service-tier field at all —
the tier is bought through a Vertex provisioning path we do not use — so a
dial here would be a switch wired to nothing. DeepSeek, Kimi, Z.ai, Alibaba and
Ollama document no fast tier; sending an unknown top-level key to them is a
400 on a turn the user did not ask to risk.
"""

from __future__ import annotations

import dataclasses
import re

#: Anthropic's fast mode is a beta, and the key is refused without this header.
#: Measured 2026-09-04: ``speed`` sent without it answers ``HTTP 400
#: "speed: Extra inputs are not permitted"``, so the two travel together.
ANTHROPIC_FAST_BETA = "fast-mode-2026-02-01"

#: The one value every service-tier route accepts. See the module docstring for
#: the two live rejections of ``"fast"`` that pinned this choice.
SERVICE_TIER_FAST = "priority"

#: Anthropic's own spelling of the same idea.
ANTHROPIC_SPEED_FAST = "fast"

#: ``body["speed"] = "fast"`` plus :data:`ANTHROPIC_FAST_BETA`.
DIALECT_ANTHROPIC_SPEED = "anthropic_speed"

#: ``body["service_tier"] = "priority"``, top-level, on OpenAI-shaped routes.
DIALECT_SERVICE_TIER = "service_tier"


@dataclasses.dataclass(frozen=True)
class FastModeSupport:
    """How one route asks for fast service, when the model can be served that way."""

    #: Which key/value pair carries the request; one of the ``DIALECT_*`` constants.
    dialect: str
    #: The wire value meaning "serve this fast".
    value: str
    #: Beta header the key requires, or ``None`` when it needs none.
    beta_header: str | None = None


_ANTHROPIC_FAST = FastModeSupport(
    DIALECT_ANTHROPIC_SPEED, ANTHROPIC_SPEED_FAST, beta_header=ANTHROPIC_FAST_BETA
)
_SERVICE_TIER_FAST = FastModeSupport(DIALECT_SERVICE_TIER, SERVICE_TIER_FAST)


#: Anthropic ships fast mode on an EXPLICIT list of two models, and the failure
#: modes off that list are why this is a list rather than a generation range:
#: ``claude-opus-4-7`` ERRORS on ``speed: "fast"``, while ``claude-opus-4-6``
#: accepts the key, silently serves the request at standard speed and bills it
#: at standard rates. The second is the dangerous one — a status band claiming
#: "fast" over a request that is not — so a model is offered the dial only when
#: the vendor names it.
#:
#: ``[.-]`` for the generation separator, for the reason ``model.effort``'s
#: table records: Anthropic hyphenates its own snapshot ids
#: (``claude-opus-4-8``) while aggregators spell the same model with a dot.
_ANTHROPIC_FAST_MODELS = re.compile(r"claude-opus-(?:5(?!\d)|4[.-]8(?!\d))")

#: OpenAI documents Fast mode for the GPT-5 generation and later, and says
#: plainly that support "isn't guaranteed for every model". A family rule is
#: therefore the honest reading: it tracks the generation the feature shipped
#: in without asserting a per-snapshot list the vendor declines to promise.
#: A model that turns out not to serve fast is not a hazard here the way the
#: Anthropic case is — the response reports the tier that actually served it,
#: and billing follows that tier, so an unsupported model is billed standard.
_OPENAI_FAST_MODELS = re.compile(r"gpt-(?:[5-9]|\d{2,})")

#: xAI's Priority Processing is documented for its text inference endpoints
#: (Chat Completions and Responses) rather than for a model list, so the gate
#: is the family that reaches those endpoints. Confirmed on the wire to be a
#: recognised enum member (2026-09-04): xAI's rejection of ``"fast"`` named
#: ``priority`` as an accepted variant.
_XAI_FAST_MODELS = re.compile(r"grok")


def fast_mode_support(provider: str, model_id: str) -> FastModeSupport | None:
    """How to ask ``provider`` to serve ``model_id`` fast, or ``None`` if it cannot.

    Keyed on the pair rather than on the model alone: the dialect belongs to the
    route (see the module docstring), so the same model answers differently
    depending on which door it is reached through.
    """
    route = (provider or "").strip().lower()
    lowered = (model_id or "").strip().lower()
    if not route or not lowered:
        return None
    if route == "anthropic":
        return _ANTHROPIC_FAST if _ANTHROPIC_FAST_MODELS.search(lowered) else None
    if route == "openai":
        return _SERVICE_TIER_FAST if _OPENAI_FAST_MODELS.search(lowered) else None
    if route == "xai":
        return _SERVICE_TIER_FAST if _XAI_FAST_MODELS.search(lowered) else None
    if route == "openrouter":
        # OpenRouter is the one route that may be asked for a tier it cannot
        # serve without penalty, so it is gated on the UNDERLYING model rather
        # than opened wide. The aggregator prefixes the publisher
        # (``anthropic/claude-opus-5``, ``openai/gpt-5.4``), and its own docs
        # list exactly which upstreams sell a priority tier through it —
        # OpenAI, Anthropic, Google and xAI. Routing is by endpoint, and the
        # response reports the tier that actually served the request with
        # billing following it, so a model whose provider sheds the request to
        # its default tier costs the default rate.
        return (
            _SERVICE_TIER_FAST
            if (
                _ANTHROPIC_FAST_MODELS.search(lowered)
                or _OPENAI_FAST_MODELS.search(lowered)
                or _XAI_FAST_MODELS.search(lowered)
            )
            else None
        )
    return None


def supports_fast_mode(provider: str, model_id: str) -> bool:
    """Whether this route can serve this model fast at all."""
    return fast_mode_support(provider, model_id) is not None

"""Per-provider SUGGESTED models, in a stdlib-only module.

This is the one table that answers "which model should a user who just connected
provider X start on?". Every surface asks it: the CLI's ``login``, the TUI's
``/login``, the desktop sign-in and API-key routes (through
``providers.login_defaults``), the startup resolver when a config names a
hosting but no model, and the desktop catalogue that SHOWS the suggestion
before anyone signs in. Keeping it here -- rather than in the UI, or beside
each caller -- is what stops those surfaces from recommending different models
for the same provider; the desktop renderer reads it over the wire
(``suggested_model`` on ``GET /v1/auth/providers``) instead of carrying a copy.

Extracted from ``model.configure`` so the startup/preflight path can ask it
without importing the model configuration stack (pydantic, the registry, the
wire clients). ``configure`` re-exports :data:`DEFAULT_MODEL_NAMES` from here
for backward compatibility, so there is still exactly one map.

**What a suggestion is.** The provider's current frontier model that the route
can actually serve -- not the cheapest, and not the most conservative. The
previous table (``gpt-4o``, ``claude-3-5-sonnet-latest``, ``grok-3``,
``gemini-2.0-flash-001``...) was chosen as "broadly available" two model
generations ago and had quietly become the worst model most providers still
serve, so a first-run user who took the default got a visibly weaker agent than
the one they signed up for. Each row names its source; re-verify them when a
provider ships a new generation, because a stale row degrades silently.

Why a default exists at all: with a hosting chosen but ``model_name`` empty (a
fresh ``config edit hosting <provider>``, or a ``--hosting`` flag with no
``--model``), the app used to raise "Model name is not configured." and die.
There is a reasonable default per provider, so it resolves to that and prints
what it picked instead of turning a one-field omission into a dead end.
"""

from __future__ import annotations

from typing import NamedTuple


class SuggestedModel(NamedTuple):
    """A suggested model id and the name a person should see for it.

    The display name travels WITH the id rather than being looked up in the
    model registry, because this module must stay importable without that
    registry (see the module docstring) and because two suggestions -- an
    aggregator's namespaced id and Radient's router -- have no registry row to
    look a name up in. ``tests/unit/providers/test_login_defaults.py`` pins that
    every non-aggregator suggestion resolves to a real registry row, so an id
    cannot be suggested that the registry would describe as "unknown".
    """

    id: str
    name: str


#: The suggested model per HOSTING id, with the source each was verified against
#: (all read 2026-09-24). A hosting absent from this table has no suggestion:
#: local runtimes serve whatever the user pulled, and a decision-only provider
#: (TypeSafe) cannot serve a chat at all.
SUGGESTED_MODELS: dict[str, SuggestedModel] = {
    # platform.claude.com models overview: "start with Claude Opus 5.5 for most
    # workloads"; also served by the live /v1/models listing under a Claude
    # Pro/Max OAuth grant, so the subscription route can use it too.
    "anthropic": SuggestedModel("claude-opus-5-5", "Claude Opus 5.5"),
    # developers.openai.com/api/docs/models/gpt-6-astra: "our most capable
    # model". The ChatGPT-subscription (Codex) route serves the same id, which is
    # what scripts/bench_openai_oauth_cache.py drives over OAuth.
    "openai": SuggestedModel("gpt-6-astra", "GPT-6 Astra"),
    # DeepSeek's direct API names V4.1 Flash ``deepseek-flash`` -- its /models
    # listing returns exactly ``deepseek-flash`` and ``deepseek-v4-pro``, and it
    # answers HTTP 400 "The supported API model names are deepseek-flash,
    # deepseek-v4-pro" to anything else (see ``configure``). The
    # ``deepseek-v4.1-flash`` spelling exists only as OpenRouter's namespaced id.
    "deepseek": SuggestedModel("deepseek-flash", "DeepSeek V4.1 Flash"),
    # models.dev's zai and zai-coding-plan catalogues: GLM-5.3 is the current
    # flagship, served under this id to the API key and to the OAuth-minted key.
    "zai": SuggestedModel("glm-5.3", "GLM-5.3"),
    # alibabacloud.com/help/en/model-studio/qwen3-8-max (Model Studio,
    # pay-as-you-go) and the Token Plan catalogue both serve ``qwen3.8-max``.
    "alibaba": SuggestedModel("qwen3.8-max", "Qwen3.8 Max"),
    "alibaba-token-plan": SuggestedModel("qwen3.8-max", "Qwen3.8 Max"),
    # docs.x.ai/developers/models: "For everything else, including code, use
    # Grok 4.7. It is the most capable model we've built." Same id on the
    # API-key and the xai-oauth route (both store under ``xai``).
    "xai": SuggestedModel("grok-4.7", "Grok 4.7"),
    # ai.google.dev pricing page: "Gemini 3.8 Flash ... our most intelligent
    # Flash model, engineered for long-horizon software engineering, autonomous
    # agents". Chosen over a Pro model at the operator's direction: it is the
    # model an agent harness should start on for price/latency.
    "google": SuggestedModel("gemini-3.8-flash", "Gemini 3.8 Flash"),
    # The API-key host (api.moonshot.cn) spells Kimi K3 ``kimi-k3`` (models.dev,
    # OpenRouter's moonshotai/kimi-k3). An OAuth sign-in reaches the coding-plan
    # host instead, which spells it ``k3`` -- see OAUTH_SUGGESTED_MODELS.
    "kimi": SuggestedModel("kimi-k3", "Kimi K3"),
    # docs.mistral.ai models overview: Mistral Medium 3.5 is "our frontier-class
    # multimodal model optimized for agentic and coding use cases", and it is the
    # named replacement for Magistral Medium and Medium 3 in the deprecation
    # table. Large 3 is an open-weight general model and ranks below it here.
    "mistral": SuggestedModel("mistral-medium-latest", "Mistral Medium 3.5"),
    # An aggregator: the suggestion is the frontier model a new user most
    # plausibly came for, under OpenRouter's own namespaced id (verified in its
    # live /models listing). Any model it lists works; this is only a start.
    "openrouter": SuggestedModel("anthropic/claude-opus-5.5", "Claude Opus 5.5"),
    # Radient's router picks the model per request, which is the product's own
    # recommendation -- a pinned model would second-guess it.
    "radient": SuggestedModel("auto", "Automatic"),
}

#: Where an OAuth sign-in reaches a DIFFERENT host than an API key does, and that
#: host spells the suggested model differently. Only Kimi does today: its OAuth
#: grant is accepted only by the coding-plan host (``oauth_base_url``), whose
#: catalogue names K3 ``k3``; ``kimi-k3`` there is not a model. Every other
#: OAuth flavour serves the same ids as its API key (the comments above say so
#: per provider), so it has no entry and falls through to SUGGESTED_MODELS.
OAUTH_SUGGESTED_MODELS: dict[str, SuggestedModel] = {
    "kimi": SuggestedModel("k3", "Kimi K3"),
}

#: The model id used when a provider is selected but no model is named -- the
#: API-key-route suggestion, as a plain ``hosting -> id`` map. Kept under its old
#: name because the startup resolver and legacy callers import it; it is DERIVED
#: from SUGGESTED_MODELS, so the two cannot disagree.
DEFAULT_MODEL_NAMES: dict[str, str] = {
    hosting: suggestion.id for hosting, suggestion in SUGGESTED_MODELS.items()
}


def _canonical(hosting: str) -> str:
    # ``noop`` maps to ``test`` for the same reason the rest of the code treats
    # them as one.
    return "test" if hosting == "noop" else hosting


def suggested_model_for(hosting: str, *, oauth: bool = False) -> SuggestedModel | None:
    """The suggested model for ``hosting``, or ``None`` when there is none.

    ``oauth`` selects the route-specific spelling where one exists (see
    :data:`OAUTH_SUGGESTED_MODELS`). ``hosting`` must already be a HOSTING id
    (``xai``, not ``xai-oauth``); resolving a login flavour to its hosting is the
    caller's job, done by ``providers.login_defaults`` through
    ``credential_provider_id`` so it happens in one place.
    """
    canonical = _canonical(hosting)
    if oauth and canonical in OAUTH_SUGGESTED_MODELS:
        return OAUTH_SUGGESTED_MODELS[canonical]
    return SUGGESTED_MODELS.get(canonical)


def default_model_for(hosting: str) -> str | None:
    """The default model id for ``hosting``, or ``None`` when there is none.

    An unknown provider returns ``None`` so the caller can keep "no default, ask
    the user" distinct from "the default is empty".
    """
    return DEFAULT_MODEL_NAMES.get(_canonical(hosting))

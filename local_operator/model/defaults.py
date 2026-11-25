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

import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from local_operator.providers.auth_store import AuthStore


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


#: The suggested model per HOSTING id, with its route evidence below. A hosting
#: absent from this table has no suggestion:
#: local runtimes serve whatever the user pulled, and a decision-only provider
#: (TypeSafe) cannot serve a chat at all.
SUGGESTED_MODELS: dict[str, SuggestedModel] = {
    # Official models documentation names this wire ID; no published Full
    # wire ID exists, so a marketing label must not become a broken alias.
    "meta-muse": SuggestedModel("muse-wonder-1.0", "Muse Wonder 1.0"),
    # platform.claude.com models overview: "start with Claude Opus 5.5 for most
    # workloads"; also served by the live /v1/models listing under a Claude
    # Pro/Max OAuth grant, so the subscription route can use it too.
    "anthropic": SuggestedModel("claude-opus-5-5", "Claude Opus 5.5"),
    # Bedrock: the same model through the GLOBAL cross-Region inference
    # profile — first-party price (regional/geo ids cost 1.10x, AWS Price
    # List 2026-11-18) and callable from every commercial source Region.
    # Bedrock prompt-caching docs list ``anthropic.claude-opus-5-5`` (read
    # 2026-11-18). A worker that needs data residency picks ``us.``/``eu.``.
    "bedrock": SuggestedModel("global.anthropic.claude-opus-5-5", "Claude Opus 5.5 (Bedrock)"),
    # OpenAI's API model page recommends Sol for everyday agent work; its
    # GPT-6.2 launch announcement also lists Sol for ChatGPT subscribers in Codex.
    "openai": SuggestedModel("gpt-6.2-sol", "GPT-6.2 Sol"),
    # DeepSeek serves V4.1 Flash on the stable direct API id, not a dated id.
    "deepseek": SuggestedModel("deepseek-flash", "DeepSeek V4.1 Flash"),
    # docs.z.ai/guides/llm/glm-5.4: Coding Plan only at launch. Both lop Z.AI
    # logins go to /api/coding/paas/v4, including the pasted-key login.
    "zai": SuggestedModel("glm-5.4", "GLM-5.4"),
    # Alibaba's qwen4-max and Token Plan product updates both list the stable
    # id. Usage OAuth on Token Plan still sends the same inference key.
    "alibaba": SuggestedModel("qwen4-max", "Qwen4 Max (Model Studio)"),
    "alibaba-token-plan": SuggestedModel("qwen4-max", "Qwen4 Max"),
    # docs.x.ai/developers/models: "For everything else, including code, use
    # Grok 4.7. It is the most capable model we've built." Same id on the
    # API-key and the xai-oauth route (both store under ``xai``).
    "xai": SuggestedModel("grok-4.7", "Grok 4.7"),
    # AI Studio offers Neon on paid tiers only; free-tier keys can explicitly
    # select gemini-3.8-flash. The key itself does not disclose its billing tier.
    "google": SuggestedModel("gemini-4-neon", "Gemini 4 Neon"),
    # Key auth uses the publisher-only endpoint (including Express Mode). The
    # express-mode docs list 2.5 Flash but not Neon; Cloud login can use Neon.
    "google-vertex": SuggestedModel("gemini-2.5-flash", "Vertex Gemini 2.5 Flash"),
    # platform.kimi.com documents this id on the mainland API-key host; the
    # Kimi Code subscription has a different wire spelling below.
    "kimi": SuggestedModel("kimi-k3.1-pro", "Kimi K3.1 Pro (Moonshot API)"),
    # The Mistral Large 4 launch names this exact hosted API id.
    "mistral": SuggestedModel("mistral-large-4", "Mistral Large 4"),
    # An aggregator: the suggestion is the frontier model a new user most
    # plausibly came for, under OpenRouter's own namespaced id (verified in its
    # live /models listing). Any model it lists works; this is only a start.
    "openrouter": SuggestedModel("anthropic/claude-opus-5.5", "Claude Opus 5.5"),
    # Radient's router picks the model per request, which is the product's own
    # recommendation -- a pinned model would second-guess it.
    "radient": SuggestedModel("auto", "Automatic"),
}

#: Overrides are for a different inference route, not just a different sign-in
#: flow. Kimi Code documents ``k3.1-pro`` for the same plan tiers as ``k3``;
#: Vertex uses the project/location endpoint for Cloud and service accounts.
OAUTH_SUGGESTED_MODELS: dict[str, SuggestedModel] = {
    "kimi": SuggestedModel("k3.1-pro", "Kimi K3.1 Pro"),
    "google-vertex": SuggestedModel("gemini-4-neon", "Vertex Gemini 4 Neon"),
}

_VERTEX_NEON_LOCATIONS = frozenset({"global", "us", "eu"})
_VERTEX_REGIONAL_SUGGESTION = SuggestedModel("gemini-3.8-flash", "Vertex Gemini 3.8 Flash")


def vertex_location(stored: str = "") -> str:
    """Vertex Cloud auth location precedence, shared by selection and dispatch."""
    return (
        stored.strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or os.environ.get("VERTEX_LOCATION", "").strip()
        or "global"
    )


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


def suggested_model_for(
    hosting: str, *, oauth: bool = False, vertex_locations: tuple[str, ...] = ()
) -> SuggestedModel | None:
    """The suggested model for ``hosting``, or ``None`` when there is none.

    ``oauth`` selects the route-specific spelling where one exists (see
    :data:`OAUTH_SUGGESTED_MODELS`). ``hosting`` must already be a HOSTING id
    (``xai``, not ``xai-oauth``); resolving a login flavour to its hosting is the
    caller's job, done by ``providers.login_defaults`` through
    ``credential_provider_id`` so it happens in one place.
    """
    canonical = _canonical(hosting)
    if oauth and canonical in OAUTH_SUGGESTED_MODELS:
        if canonical == "google-vertex" and any(
            vertex_location(location) not in _VERTEX_NEON_LOCATIONS
            for location in (vertex_locations or ("",))
        ):
            # Keep the previous Cloud suggestion outside Neon's published
            # locations. Never silently rewrite the user's inference region.
            return _VERTEX_REGIONAL_SUGGESTION
        return OAUTH_SUGGESTED_MODELS[canonical]
    suggestion = SUGGESTED_MODELS.get(canonical)
    if suggestion is not None and (canonical, suggestion.id) in _KEYED_NAMES:
        return suggestion._replace(name=_keyed_name(canonical, suggestion))
    return suggestion


#: Suggestions whose display name is product COPY rather than a model's proper
#: name, keyed by ``(hosting, id)``. Every other name in the tables above is a
#: vendor's product name ("Claude Opus 5.5") and stays as written in every
#: language; Radient's router has no product name, only a description of what it
#: does ("Automatic"), so a French connect dialog read "Suggère Automatic" (the
#: desktop fr census, 2026-11). The table keeps that English word as the source
#: and the fallback; the id (``auto``) is a machine token and never changes.
#: Values name the generated ``wire.providers.connect`` key function.
_KEYED_NAMES: dict[tuple[str, str], str] = {("radient", "auto"): "suggested_automatic"}


def _keyed_name(canonical: str, suggestion: SuggestedModel) -> str:
    """The display name in the language the config resolves NOW.

    Rendered here, at the one read every surface goes through (the auth and
    models routes' ``suggested_model`` and ``login_defaults``' "Default model
    set to ..." label), because each of them hands the name on as finished text.
    Lazy imports keep this module importable without the i18n stack, the same
    constraint the module docstring states for the model registry; a render
    failure keeps the English name rather than failing the listing.
    """
    try:
        from local_operator.i18n.keys import wire_providers_connect
        from local_operator.i18n.messages import render_resolved

        key = getattr(wire_providers_connect, _KEYED_NAMES[(canonical, suggestion.id)])
        return render_resolved(key())
    except Exception:  # noqa: BLE001 — a label is copy; the suggestion still stands
        pass
    return suggestion.name


def stored_route_suggestion(hosting: str, store: AuthStore) -> SuggestedModel | None:
    """Suggest for the eligible local OAuth tier without contacting the vendor."""
    locations = store.oauth_locations_for_default(hosting)
    return suggested_model_for(hosting, oauth=bool(locations), vertex_locations=locations)


def local_route_suggestion(hosting: str, config_dir: Path) -> SuggestedModel | None:
    """Preview a new turn's route without refreshing a token or borrowing a key.

    Startup and the provider census need a useful local answer before a request
    exists. Exclude known unusable grants, but keep refreshable Cloud accounts;
    when multiple eligible Cloud rows remain, don't guess which region wins.
    This is presentation only: dispatch has the actual credential winner.
    """
    canonical = _canonical(hosting)
    if canonical in OAUTH_SUGGESTED_MODELS:
        from local_operator.providers.auth_store import AuthStore

        try:
            store = AuthStore.open_read_only(config_dir / "auth.db")
            try:
                return stored_route_suggestion(canonical, store)
            finally:
                store.close()
        except Exception:  # noqa: BLE001 — damaged auth metadata must not prevent startup
            pass
    return suggested_model_for(canonical)


class RouteDefaultModel(str):
    """An omitted model's provisional display ID, not an explicit selection.

    Startup, resume and re-home still pass string pairs through their model
    selection APIs. Keep omission distinguishable until the shared spec
    builder; an identical string named by the caller must not be retargeted.
    """


def default_model_for(hosting: str, *, config_dir: Path | None = None) -> str | None:
    """Resolve an omitted model; only inspect auth for routes that disagree."""
    canonical = _canonical(hosting)
    if canonical in OAUTH_SUGGESTED_MODELS:
        suggestion = (
            local_route_suggestion(canonical, config_dir)
            if config_dir is not None
            else suggested_model_for(canonical)
        )
        return RouteDefaultModel(suggestion.id) if suggestion is not None else None
    return DEFAULT_MODEL_NAMES.get(canonical)

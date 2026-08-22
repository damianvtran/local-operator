"""Per-provider default model ids, in a stdlib-only module.

Extracted from ``model.configure`` so the startup/preflight path can answer
"what model does this provider default to?" without importing the model
configuration stack (pydantic, the registry, the wire clients). ``configure``
re-exports :data:`DEFAULT_MODEL_NAMES` from here for backward compatibility, so
there is still exactly one map; this module is only about WHERE it can be
imported cheaply from.

Why a default exists at all: with a hosting chosen but ``model_name`` empty (a
fresh ``config edit hosting <provider>``, or a ``--hosting`` flag with no
``--model``), the app used to raise "Model name is not configured." and die.
There is a reasonable default per provider, so it resolves to that and prints
what it picked instead of turning a one-field omission into a dead end.
"""

from __future__ import annotations

#: The model id used when a provider is selected but no model is named. Values
#: are deliberately conservative, broadly-available ids for each provider; a
#: user who wants a specific model sets ``model_name`` and this is never read.
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


def default_model_for(hosting: str) -> str | None:
    """The default model id for ``hosting``, or ``None`` when there is none.

    ``noop`` maps to ``test`` for the same reason the rest of the code treats
    them as one; an unknown provider returns ``None`` so the caller can keep
    "no default, ask the user" distinct from "the default is empty".
    """
    canonical = "test" if hosting == "noop" else hosting
    return DEFAULT_MODEL_NAMES.get(canonical)

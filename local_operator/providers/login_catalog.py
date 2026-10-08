"""How the login choices are PRESENTED: one order, one set of words, every host.

WHY THIS EXISTS. ``/login`` (TUI) and ``lop login`` (CLI) each printed the raw
registry in registry order: ``openai`` first, ``radient`` 20th, speech and
decision-only rows mixed in with chat logins, labels like ``ChatGPT Plus/Pro``
beside rows with no description at all, and a ``*`` marker nobody explained
(audit D3/U2/U9/Q3/D13). The desktop already groups its provider page
(``local-operator-ui`` ``provider-catalog.ts``: "Use a subscription" / "Use an
API key" / "Run models on this computer"); a first-run user who meets the TUI
first saw none of that.

So the presentation lives HERE, beside the registry, and both terminal hosts
read it:

* :data:`RECOMMENDED_LOGIN` is the ONE recommended provider. The setup splash,
  the rejected-message line, the headless quickstart, the hints table and the
  README all name it through this constant — five surfaces that used to say
  ``/login openai`` independently.
* :func:`login_groups` orders the loginable rows into the desktop's groups with
  a ``Recommended`` group first, a human label and a one-line description for
  every row.
* ``include_non_chat`` keeps the speech/decision-only rows OFF the setup-state
  picker (a first-run user who picks ElevenLabs gets a stored key and still
  cannot chat), while the normal ``/login`` and ``lop login`` list them in a
  labelled ``Not for chat`` group so a user who needs one can still find it.

Pure: reads only the registry. The registry stays the authority for WHAT can be
logged into; this module only decides how that is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_operator.providers.registry import (
    ProviderDefinition,
    is_decision_only,
    is_speech_only,
    list_login_providers,
)

#: The provider every first-run surface recommends. Radient is the one login
#: that needs no existing AI subscription or key (a browser sign-in that can
#: create an account), routes to many model vendors, and is what the phone
#: relay signs in through — so it is the right first step for someone who
#: does not yet know which provider they want.
RECOMMENDED_LOGIN = "radient"

#: The setup command, spelled once so the splash, the refusal line, the hints
#: and the quickstart cannot drift.
RECOMMENDED_LOGIN_COMMAND = f"/login {RECOMMENDED_LOGIN}"

GROUP_RECOMMENDED = "Recommended"
GROUP_SUBSCRIPTION = "Use a subscription"
GROUP_API_KEY = "Use an API key"
GROUP_LOCAL = "Run models on this computer"
GROUP_NOT_CHAT = "Not for chat"

#: The display order of the groups. The three middle headings are the
#: desktop's own (``provider-catalog.ts`` ``GROUP_HEADINGS``) so a user moving
#: between the two front ends reads one vocabulary.
GROUP_ORDER: tuple[str, ...] = (
    GROUP_RECOMMENDED,
    GROUP_SUBSCRIPTION,
    GROUP_API_KEY,
    GROUP_LOCAL,
    GROUP_NOT_CHAT,
)

#: A one-line description per row: what the user needs to HAVE for it to work.
#: Every loginable row has one; a row missing here falls back to a description
#: derived from its kind (:func:`_fallback_description`), and
#: ``tests/unit/providers/test_login_catalog.py`` fails if any shipped row
#: needs the fallback, so a new provider cannot land wordless.
DESCRIPTIONS: dict[str, str] = {
    "radient": "Browser sign-in or new account; many models, phone relay included",
    "radient-key": "Paste a Radient Pass key",
    "openai": "Sign in with your ChatGPT Plus/Pro plan",
    "openai-device": "ChatGPT plan, one-time code (no browser on this machine)",
    "openai-api-key": "Paste an OpenAI platform API key",
    "anthropic": "Sign in with your Claude Pro/Max plan",
    "anthropic-key": "Paste an Anthropic Console API key",
    "kimi": "Sign in with your Moonshot Kimi account",
    "xai": "Paste an xAI (Grok) API key",
    "xai-oauth": "Sign in with SuperGrok, one-time code",
    "deepseek": "Paste a DeepSeek API key",
    "zai": "Paste a Z.AI (GLM) API key",
    "zai-oauth": "Sign in with your Z.AI coding plan",
    "google": "Paste a Google AI Studio (Gemini) API key",
    "mistral": "Paste a Mistral API key",
    "openrouter": "Paste an OpenRouter key; hundreds of models",
    "alibaba": "Paste an Alibaba Cloud DashScope key (Qwen)",
    "alibaba-token-plan": "Paste a QwenCloud Token Plan key",
    "alibaba-token-plan-oauth": "QwenCloud Token Plan key plus usage sign-in",
    "lmstudio": "Use a running LM Studio server",
    "ollama": "Use a running Ollama server",
    "vllm": "Use a running vLLM server",
    "llamacpp": "Use a running llama.cpp server",
    "openai-compatible": "Any OpenAI-compatible endpoint you run",
    "elevenlabs": "Speech only (voice on the phone); not for chat",
    "openai-key": "Speech only (voice); not for chat",
    "typesafe": "Classification only; not for chat",
}


@dataclass(frozen=True)
class LoginRow:
    """One row as a terminal host shows it."""

    id: str
    #: The human label (the registry's brand plus flavour), e.g. ``OpenAI (API key)``.
    label: str
    description: str
    group: str
    recommended: bool = False


#: The PICKER's description for a row: the same fact as :data:`DESCRIPTIONS`,
#: said in the room a one-line list row actually has.
#:
#: WHY A SECOND MAP (design round 1, D3). The `/login` picker paints
#: name | description | state on ONE line inside a column that is a fraction of
#: the terminal, so the full sentences above ellipsized on 11 of the first 14
#: rows at 120 columns and on every visible row at 80 — the cut landing on the
#: distinguishing words (``one-ti…``, ``coding…``, ``Gemi…``) that are the only
#: reason two near-twin rows (``zai``/``zai-oauth``, ``xai``/``xai-oauth``,
#: ``alibaba-token-plan``/``alibaba-token-plan-oauth``) can be told apart.
#: ``lop login`` prints one row per line over a whole terminal, so it keeps the
#: long form: the two hosts answer the same question at different lengths
#: rather than disagreeing about the answer. Every shipped row needs an entry —
#: ``tests/unit/providers/test_login_catalog.py`` fails a row that would fall
#: back, the same rule :data:`DESCRIPTIONS` is held to.
#:
#: Budget: ≤ 36 cells, so the widest of these still fits the picker's description
#: column at an 80-column terminal alongside the widest name.
PICKER_DESCRIPTIONS: dict[str, str] = {
    "radient": "Browser sign-in; no key needed",
    "radient-key": "Paste a Radient Pass key",
    "openai": "ChatGPT plan sign-in",
    "openai-device": "ChatGPT plan; code, no browser",
    "openai-api-key": "Paste an OpenAI platform key",
    "anthropic": "Claude Pro/Max sign-in",
    "anthropic-key": "Paste an Anthropic Console key",
    "kimi": "Moonshot Kimi sign-in",
    "xai": "Paste an xAI (Grok) key",
    "xai-oauth": "SuperGrok sign-in, one-time code",
    "deepseek": "Paste a DeepSeek key",
    "zai": "Paste a Z.AI (GLM) key",
    "zai-oauth": "Z.AI coding plan, browser sign-in",
    "google": "Paste a Google AI Studio key",
    "mistral": "Paste a Mistral key",
    "openrouter": "Paste an OpenRouter key",
    "alibaba": "Paste an Alibaba DashScope key",
    "alibaba-token-plan": "Paste a QwenCloud Token key",
    "alibaba-token-plan-oauth": "QwenCloud plan key + sign-in",
    "lmstudio": "A running LM Studio server",
    "ollama": "A running Ollama server",
    "vllm": "A running vLLM server",
    "llamacpp": "A running llama.cpp server",
    "openai-compatible": "Your own OpenAI-compatible endpoint",
    "elevenlabs": "Speech only; not for chat",
    "openai-key": "Speech only; not for chat",
    "typesafe": "Classification only; not for chat",
}

#: The picker's NAME for a row: the short human label, where the registry's
#: ``name`` is the full flavour ("Alibaba Cloud (QwenCloud Token Plan)") that a
#: one-line list row cannot afford.
#:
#: WHY A THIRD FORM (design round 1, D3, second measurement). Painting the
#: registry label fixed the jargon (`zai-oauth`) but spent the name column on
#: the parenthetical, so at 120 columns the LABEL itself was the thing
#: ellipsized ("OpenAI (ChatGPT Plus/Pro)" → "…n-in") and the descriptions went
#: on being cut. The name column and the description column compete for one
#: line, so both halves have to be short: the brand here, the flavour in
#: PICKER_DESCRIPTIONS. `lop login` prints one row per line and keeps the full
#: registry name — the same fact, at the length that surface can afford.
#:
#: Uniqueness matters as much as length: two rows for one brand ("OpenAI API
#: key" against "OpenAI speech") must stay tellable apart, and the catalogue
#: test enforces it.
PICKER_LABELS: dict[str, str] = {
    "radient": "Radient",
    "radient-key": "Radient Pass",
    "openai": "ChatGPT plan",
    "openai-device": "ChatGPT code",
    "openai-api-key": "OpenAI API key",
    "anthropic": "Claude plan",
    "anthropic-key": "Anthropic key",
    "kimi": "Kimi",
    "xai": "xAI key",
    "xai-oauth": "SuperGrok",
    "deepseek": "DeepSeek",
    "zai": "Z.AI key",
    "zai-oauth": "Z.AI plan",
    "google": "Google AI",
    "mistral": "Mistral",
    "openrouter": "OpenRouter",
    "alibaba": "DashScope",
    "alibaba-token-plan": "QwenCloud key",
    "alibaba-token-plan-oauth": "QwenCloud plan",
    "lmstudio": "LM Studio",
    "ollama": "Ollama",
    "vllm": "vLLM",
    "llamacpp": "llama.cpp",
    "openai-compatible": "Your endpoint",
    "elevenlabs": "ElevenLabs",
    "openai-key": "OpenAI speech",
    "typesafe": "Typesafe",
}

#: The widest a PICKER label may be, in cells (see :data:`PICKER_LABELS`). The
#: longest shipped entry is 14; the budget leaves the arrow and gutter room.
PICKER_LABEL_BUDGET = 16


def picker_label(provider_id: str, fallback: str = "") -> str:
    """The picker row's NAME for ``provider_id`` (``fallback`` for embedders).

    Falls back to the registry's own name rather than a blank: an embedder's
    provider has no catalogue entry and must still be choosable, and a name is
    better than an id.
    """
    return PICKER_LABELS.get(provider_id) or fallback or provider_id


#: The widest a PICKER description may be, in cells. MEASURED from the picker's
#: own columns at 80 columns wide (the narrowest width the setup screen is
#: designed for): the name column takes the longest label, the state column the
#: longest state, and what is left is the description column. Enforced by the
#: catalogue test, so a new row cannot land ellipsized.
PICKER_DESCRIPTION_BUDGET = 36

#: The single spelling of the recommendation tag, so the picker, `lop login`,
#: the setup splash and the README cannot each say it their own way (design
#: round 1, D10). One word, one meaning: this row is the one we suggest first.
RECOMMENDED_TAG = "recommended"


def picker_description(provider_id: str) -> str:
    """The picker row's description for ``provider_id``.

    Falls back to the LONG description rather than an empty string: a row whose
    short form is missing should say too much, never nothing — a blank
    description column is the defect the description field exists to fix
    (first-run audit D3/U9/D13). The catalogue test is what keeps the fallback
    from being the normal path.
    """
    return PICKER_DESCRIPTIONS.get(provider_id) or DESCRIPTIONS.get(provider_id, "")


def is_non_chat(provider_id: str) -> bool:
    """A login that stores a credential no chat turn can use."""
    return is_speech_only(provider_id) or is_decision_only(provider_id)


def _is_key_login(definition: ProviderDefinition) -> bool:
    return bool(getattr(definition.login, "__lo_api_key_login__", False))


def group_of(definition: ProviderDefinition) -> str:
    """Which group a row belongs to (the desktop's rule, plus two of ours)."""
    if definition.id == RECOMMENDED_LOGIN:
        return GROUP_RECOMMENDED
    if is_non_chat(definition.id):
        return GROUP_NOT_CHAT
    if definition.local_setup:
        return GROUP_LOCAL
    if _is_key_login(definition):
        return GROUP_API_KEY
    return GROUP_SUBSCRIPTION


def _fallback_description(definition: ProviderDefinition) -> str:
    group = group_of(definition)
    if group == GROUP_LOCAL:
        return "Use a model server on this computer"
    if group == GROUP_API_KEY:
        return "Paste an API key"
    return "Sign in in your browser"


def login_groups(*, include_non_chat: bool = True) -> list[tuple[str, list[LoginRow]]]:
    """The loginable rows, grouped and ordered for display.

    Within a group the registry order is kept (it is already roughly
    "most-used first" and is the order the desktop's tests pin). Empty groups
    are dropped so a host never prints a bare heading.
    """
    buckets: dict[str, list[LoginRow]] = {name: [] for name in GROUP_ORDER}
    for definition in list_login_providers():
        group = group_of(definition)
        if group == GROUP_NOT_CHAT and not include_non_chat:
            continue
        buckets[group].append(
            LoginRow(
                id=definition.id,
                label=definition.name,
                description=DESCRIPTIONS.get(definition.id) or _fallback_description(definition),
                group=group,
                recommended=definition.id == RECOMMENDED_LOGIN,
            )
        )
    return [(name, rows) for name, rows in buckets.items() if rows]


def ordered_rows(*, include_non_chat: bool = True) -> list[LoginRow]:
    """:func:`login_groups` flattened, for a host that shows one list."""
    return [row for _, rows in login_groups(include_non_chat=include_non_chat) for row in rows]


#: Browser logins whose provider ALSO ships a device-code flavour that needs no
#: browser on this machine. Keyed by the browser login id.
DEVICE_CODE_ALTERNATIVES: dict[str, str] = {
    "openai": "openai-device",
    "anthropic": "anthropic-key",
    "radient": "radient-key",
    "zai-oauth": "zai",
}


def headless_display(environ: "dict[str, str] | None" = None, platform: str | None = None) -> bool:
    """Whether a browser opened HERE cannot reach the person (SSH, no display).

    ``SSH_CONNECTION``/``SSH_TTY`` means the keyboard is on another machine, so
    the loopback redirect lands on a browser that is not theirs; a Linux host
    with neither ``DISPLAY`` nor ``WAYLAND_DISPLAY`` cannot open one at all
    (audit Q8). macOS and Windows always have a display session when a user
    runs the CLI, so only the SSH signal applies there.
    """
    import os
    import sys

    env = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        return True
    if plat.startswith("linux"):
        return not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    return False


def remote_login_hint(provider_id: str, *, command: str = "lop login") -> str | None:
    """One line pointing a remote/headless user at a flow that works for them.

    ``None`` when this machine can show a browser, or when the provider has no
    browser-free alternative to suggest. ``command`` is the host's spelling
    (``lop login`` in a shell, ``/login`` in the TUI) so the line names a
    command its reader can actually run.
    """
    if not headless_display():
        return None
    alternative = DEVICE_CODE_ALTERNATIVES.get(provider_id)
    if alternative is None:
        return None
    return (
        "No browser on this machine (SSH or no display)? "
        f"Use `{command} {alternative}` instead — it works without one."
    )

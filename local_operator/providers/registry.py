"""Provider registry — one ``ProviderDefinition`` per provider.

Field presence is the feature
flag (``login`` present ⇒ interactive login, ``callback_port`` ⇒ loopback
flow, ...). Heavy OAuth modules are reached through lazy-import thunks so
they stay out of the eager startup graph.

Every legacy ``--hosting`` name MUST resolve here (the 11 names in
``local_operator.model.registry.SupportedHostingProviders`` plus ``test``
and ``noop``).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.harness.types import AbortSignal
from local_operator.providers.local import LOCAL_PRESETS

if TYPE_CHECKING:
    from local_operator.providers.oauth.callback_server import LoginCallbacks

WireFormat = Literal["openai-compat", "anthropic", "google", "mock"]

# A login yields either the OAuth credentials mapping or, for paste-a-key
# providers, the bare key string.
LoginFn = Callable[..., Awaitable[str | dict[str, Any]]]
RefreshFn = Callable[..., Awaitable[dict[str, Any]]]
GetApiKeyFn = Callable[[dict[str, Any]], str]
#: How a provider names its API key's environment variable(s).
#:
#: THREE forms, and the tuple one is not decoration: TypeSafe publishes its
#: decision model under ``TYPESAFE_API_KEY`` and its original name
#: ``JEV_API_KEY``, and an operator who already exported the older spelling must
#: not have to rename it. Written as a tuple rather than a callable because the
#: NAMES are data the credential surfaces need (``credential_file_names`` turns
#: them into the legacy-store rungs the login status reads), whereas a callable
#: only ever answers with a value. ``resolve_env_key`` reads them in order, which
#: is what makes "primary first" a property of the registry row rather than of
#: whichever reader happens to be asking.
EnvKeys = str | tuple[str, ...] | Callable[[], str | None] | None

#: Attribute a login callable sets on itself to declare "I cannot complete
#: without reading text from the user". Read by
#: ``ProviderDefinition.__post_init__``, which is what carries the requirement
#: from the flow that has it out to the hosts that must honour it.
PASTE_PROMPT_ATTR = "__lo_requires_paste_prompt__"

#: Attribute a login callable sets on itself to declare "the text my prompt
#: reads is an API KEY" -- as opposed to an OAuth authorization code, which is
#: what Anthropic's and Z.AI's fallback box reads. Read by
#: ``ProviderDefinition.__post_init__`` into ``paste_is_api_key``, the question a
#: host asks before sending a paste to a provider as a key (the desktop's
#: ``POST /v1/auth/operations/{id}/input`` runs ``key_check`` on it).
#:
#: A separate tag from :data:`PASTE_PROMPT_ATTR` because the two are different
#: facts: a required prompt is not always a key, and an optional one could be.
#: It is NOT the login flavour either: keying the check on
#: ``login_kind == "api_key"`` let ``alibaba-token-plan-oauth`` -- a DEVICE
#: login whose first step reads the ``sk-sp-`` inference key -- store a
#: fabricated key unchecked (review round 2, MAJOR 1).
PASTE_IS_API_KEY_ATTR = "__lo_paste_is_api_key__"

#: The closed vocabulary ``ProviderDefinition.capabilities`` accepts. A member
#: outside this set is a construction error (see ``__post_init__``), not a
#: capability a consumer silently never matches on: the strings are consumed by
#: surfaces that switch on them (the speech lane's cascade, the desktop's
#: composer), and a typo'd ``"speech"`` would look like a provider that simply
#: serves nothing rather than like a mistake.
CAPABILITY_VOCABULARY = frozenset({"chat", "tts", "stt"})


@dataclasses.dataclass(frozen=True)
class ProviderDefinition:
    """The whole per-provider auth/routing record.

    - ``env_keys``: env var name, a TUPLE of names read in order (primary
      first), or a zero-arg callable returning the key's value (picking among
      several vars where the choice has to be computed, feature-flag style).
    - ``allows_missing_api_key``: transport needs no bearer (local servers).
    - ``store_credentials_as``: alias the credential row under another
      provider id (xai-oauth ⇒ xai; openai-device ⇒ openai).
    - ``oauth_base_url``: host to use when the resolved credential is an OAuth
      token, for providers that serve subscription sign-ins and pay-as-you-go
      API keys from DIFFERENT hosts. Kimi is the case that forced it: the
      coding-plan OAuth grant is only accepted at ``api.kimi.com/coding/v1``
      (which is where ``k3`` lives), while ``KIMI_API_KEY`` belongs to the
      mainland ``api.moonshot.cn/v1`` platform and 401s there. One ``base_url``
      per provider therefore cannot be right for both credential kinds, and the
      symptom is silent: model discovery 401s, falls back to the static
      registry, and the picker shows a years-old model list.
      ``None`` means the provider serves both kinds from ``base_url``.
    - ``wire``: which wire client serves this provider.
    - ``search_aliases``: other names a USER would type for this provider —
      almost always the model family it is known by (``claude`` for anthropic,
      ``qwen`` for alibaba). Provider metadata rather than view state, so the
      TUI picker, the CLI and any later surface all offer the same vocabulary.
      Nothing routes on these; they only make the provider findable, which
      matters because the company name and the name on the model a user came
      here for are frequently different.

    Two DIFFERENT things describe "this login may read text from the user", and
    conflating them is what broke every paste-a-key login (see
    ``paste_prompt_required`` below):

    - ``paste_code_flow`` — the login has a loopback/device path AND *also*
      accepts a pasted code as a FALLBACK, raced against the callback. Anthropic
      only. Attaching a prompt to any other loopback provider deadlocks the
      terminal, which is why hosts gate on it.
    - ``paste_prompt_required`` — the login has no other path at all: reading
      the key IS the flow, so a host that attaches no prompt cannot log in and
      the flow can only fail. Derived, not stored (see the property), because
      the login callable already knows.
    """

    id: str
    name: str
    env_keys: EnvKeys = None
    allows_missing_api_key: bool = False
    #: Endpoint setup shares the login surface, but does not mint an OAuth grant.
    local_setup: bool = False
    login: LoginFn | None = None
    refresh_token: RefreshFn | None = None
    get_api_key: GetApiKeyFn | None = None
    store_credentials_as: str | None = None
    callback_port: int | None = None
    paste_code_flow: bool = False
    base_url: str | None = None
    oauth_base_url: str | None = None
    wire: WireFormat = "openai-compat"
    search_aliases: tuple[str, ...] = ()
    #: Whether this login cannot complete without reading text from the user.
    #: NOT a host preference and not a fallback marker: a host that ignores it
    #: turns the login into a guaranteed error, which is the defect this field
    #: exists to make impossible to reintroduce.
    #:
    #: Left unset in every registration below and DERIVED in ``__post_init__``
    #: from the login callable itself, which is the only thing that actually
    #: knows whether it prompts. Declaring it per provider would be one more
    #: line to forget, and forgetting it is exactly how the paste-a-key
    #: providers shipped unloggable-into: the requirement lived in the login
    #: body and nothing carried it out to the hosts.
    requires_paste_prompt: bool = False
    #: Whether the text this login's paste prompt reads is an API KEY -- the
    #: value a host must validate with the provider before the flow stores it.
    #: Derived in ``__post_init__`` from :data:`PASTE_IS_API_KEY_ATTR` on the
    #: login callable, for the same reason ``requires_paste_prompt`` is: the
    #: flow that reads the paste is the only thing that knows what it is, and a
    #: per-row flag would be one more line a new provider could forget.
    paste_is_api_key: bool = False
    #: This provider serves DECISION-model calls, never chat completions.
    #:
    #: TypeSafe's Jev is the case this exists for: every host we reach it
    #: through rejects ``chat/completions`` outright, so a user who selected it
    #: in ``/model`` would get a session that cannot answer a single turn, and
    #: a fallback chain naming it would route a failing turn onto a provider
    #: that can only fail again. The flag therefore marks a login-capable,
    #: selectable-NOWHERE provider: it keeps its ``login`` (the key has to be
    #: storable) while every surface that offers or resolves a CHAT model must
    #: ask :func:`is_decision_only` and skip it.
    #:
    #: Honoured in four places, kept in step by
    #: ``tests/unit/providers/test_decision_only.py``: the catalogue
    #: (``providers.controller``, i.e. discovery/listing for every surface),
    #: ``/model``'s ranking (``model.ranking``), session model resolution
    #: (``session_factory.resolve_hosting_model_with_source``) and the failover
    #: chain (``providers.failover.expand_fallback_targets``).
    decision_only: bool = False
    #: This provider serves SPEECH-TO-TEXT, never chat completions.
    #:
    #: ElevenLabs is the case this exists for: the row is login-capable ON
    #: PURPOSE (a bring-your-own key has to be storable for the mobile voice
    #: path, ``clients/stt.py``) while its wire answers no chat completion, so
    #: a session could never run a turn on it. Same shape as ``decision_only``
    #: above and enforced at the same doors — every surface that offers or
    #: resolves a CHAT model asks :func:`is_speech_only` beside
    #: :func:`is_decision_only`, kept in step by
    #: ``tests/unit/providers/test_speech_only.py``.
    speech_only: bool = False
    #: What this provider's WIRE actually serves, from
    #: :data:`CAPABILITY_VOCABULARY` (``chat``/``tts``/``stt``).
    #:
    #: The default ``{"chat"}`` is today's truth for every entry whose wire
    #: answers chat completions -- every row that predates this field. A row
    #: declares its own set when that stops being true: ``elevenlabs`` carries
    #: ``{"stt", "tts"}`` because its wire is speech in both directions and never
    #: chat, ``openai-key`` carries ``{"tts"}``, and a future TTS
    #: provider sets ``{"tts"}``. Nothing consumes it yet (the speech lane's
    #: cascade reads its own rung list), so declaring it changes no behaviour;
    #: it is the fact the surfaces WILL gate on, kept next to the wire it
    #: describes so there is no second table to drift.
    #:
    #: Deliberately NOT derived from ``speech_only``/``decision_only``, and a
    #: follow-up may derive ``speech_only`` from ``"chat" not in capabilities``
    #: instead -- not this change, because both those flags have enforcement
    #: sites pinned by their own tests and the derivation would be a behaviour
    #: change dressed as a refactor.
    capabilities: frozenset[str] = frozenset({"chat"})
    #: Provider-class STORE-row names (the encrypted ``lop credential`` namespace,
    #: e.g. ``OPENAI_API_KEY``) a persisted-credential probe may read for this row
    #: when it has no ``env_keys`` of its own.
    #:
    #: Exists for ``openai-key`` alone. That row declares NO ``env_keys`` so an
    #: ambient ``OPENAI_API_KEY`` can never surface on a view of it, which also
    #: leaves the probe with no name to look up the legacy store row under -- the
    #: row ``lop credential update OPENAI_API_KEY`` wrote before this login
    #: existed, and which the OpenAI speech rung must keep honouring so nobody
    #: loses a rung they had. This field names that STORE row and nothing else:
    #: it is never read from the process environment, never shown by a view and
    #: never consulted by the call-time cascade (``resolve_env_key`` ignores it).
    legacy_store_keys: tuple[str, ...] = ()
    #: The clean title a surface shows for this provider: the name MINUS the
    #: trailing parenthetical that distinguishes login flavours (``"OpenAI"``
    #: for ``"OpenAI (ChatGPT Plus/Pro)"``).
    #:
    #: ``""`` means DERIVE (:func:`provider_brand`) rather than store the same
    #: string twice: the registry ``name`` is written for a CLI listing where
    #: the id is not adjacent, and the derivation is right for every row whose
    #: parenthetical is a flavour qualifier. An explicit value is for the rows
    #: where derivation is WRONG -- ``alibaba-token-plan`` is the case
    #: (``brand="QwenCloud"``, the product its users know, against a name whose
    #: first two words are the vendor's cloud brand).
    brand: str = ""

    def __post_init__(self) -> None:
        """Adopt the login callable's own paste requirement.

        ``create_api_key_login`` and the QwenCloud thunk tag themselves with
        :data:`PASTE_PROMPT_ATTR`; a provider built from one inherits the tag
        with no per-provider bookkeeping, so adding a paste-a-key provider
        cannot silently produce a login no host can drive. An explicit ``True``
        passed by a caller is honoured and never downgraded.
        """
        if not self.requires_paste_prompt and getattr(self.login, PASTE_PROMPT_ATTR, False):
            # The dataclass is frozen (definitions are shared, module-level and
            # read from every surface), so the derivation goes through
            # ``object.__setattr__`` — the documented way to complete a frozen
            # instance's own construction.
            object.__setattr__(self, "requires_paste_prompt", True)
        if not self.paste_is_api_key and getattr(self.login, PASTE_IS_API_KEY_ATTR, False):
            object.__setattr__(self, "paste_is_api_key", True)
        # LOUD on an unknown member: the set is consumed by surfaces that
        # switch on it, so a member nothing matches would read as "this
        # provider serves nothing", and finding that at a call site instead of
        # at construction is the difference between a one-line fix and a hunt.
        unknown = self.capabilities - CAPABILITY_VOCABULARY
        if unknown:
            raise ValueError(
                f"provider {self.id!r} declares unknown capabilit"
                f"{'y' if len(unknown) == 1 else 'ies'}: {sorted(unknown)!r}"
                f" (vocabulary: {sorted(CAPABILITY_VOCABULARY)!r})"
            )

    @property
    def accepts_api_key(self) -> bool:
        """Whether a pasted/saved API key is a valid credential for this row.

        The ONE answer behind the census's ``accepts_api_key`` field and the
        ``PUT /v1/auth/providers/{id}/key`` route, which used to each read
        ``env_keys is not None`` -- true for every provider with a key variable
        and FALSE for ``openai-key``, a paste-a-key login that declares no
        variable on purpose. Reading the login's own kind as well keeps the two
        consistent: a row whose login IS "paste an API key" accepts one.
        """
        return self.env_keys is not None or self.login_kind == "api_key"

    @property
    def login_kind(self) -> str | None:
        """The flow a host should offer, derived from the callable that owns it.

        A required paste is not necessarily an API-key-only login: QwenCloud
        collects a key and then runs device authorization. Keeping the marker
        on the key factory prevents desktop clients from guessing by provider.
        """
        if self.login is None:
            return None
        if getattr(self.login, "__lo_api_key_login__", False):
            return "api_key"
        return "browser" if self.callback_port is not None else "device"

    @property
    def paste_prompt_required(self) -> bool:
        """Whether a host MUST attach ``on_manual_code_input`` to log in.

        The one question a host should ask. ``paste_code_flow`` answers a
        narrower one ("may a prompt race the loopback callback?"), and hosts
        that asked it instead attached no prompt to the paste-a-key providers —
        whose login is nothing but that prompt — so `/login alibaba` and every
        sibling failed with "requires an interactive code prompt" before it
        could ask for anything.
        """
        return self.requires_paste_prompt

    @property
    def accepts_paste_prompt(self) -> bool:
        """Whether a host may offer a paste prompt for this provider at all.

        The union of the two cases: a required prompt, and Anthropic's optional
        fallback. Hosts attach a prompt on this and on nothing else, so a
        loopback-only provider still never gets one (attaching it there races
        the HTTP callback and leaves the terminal blocked — the trap the
        ``paste_code_flow`` gate was originally protecting).
        """
        return self.requires_paste_prompt or self.paste_code_flow


def _lazy_login(
    module: str,
    attr: str,
    *,
    requires_paste_prompt: bool = False,
    paste_is_api_key: bool = False,
) -> LoginFn:
    """Dynamic-import thunk: keeps OAuth deps out of startup imports.

    ``requires_paste_prompt`` and ``paste_is_api_key`` are carried on the
    returned callable rather than passed to the provider entry, because the
    whole point of the thunk is that the real login module is NOT imported at
    registry build time — so the only place either fact can be stated without
    paying that import is here.
    """

    async def login(
        callbacks: LoginCallbacks,
        *,
        signal: AbortSignal | None = None,
        open_browser: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        fn = getattr(importlib.import_module(module), attr)
        # Forward the opener to EVERY login that takes one, decided by the
        # callee's own signature rather than a name list. The list this replaced
        # named only Anthropic and OpenAI, so Z.AI and Radient -- which accept
        # ``open_browser`` too -- silently fell back to ``webbrowser.open``: the
        # desktop passes a no-op opener because its renderer opens the URL, and
        # a dropped opener made the BACKEND open a second browser tab (QA round
        # 1, Q3). Device flows take no opener and no ``**kwargs``, so passing it
        # unconditionally would be a TypeError there instead.
        if "open_browser" in inspect.signature(fn).parameters:
            return await fn(callbacks, signal=signal, open_browser=open_browser, **kwargs)
        return await fn(callbacks, signal=signal, **kwargs)

    if requires_paste_prompt:
        setattr(login, PASTE_PROMPT_ATTR, True)
    if paste_is_api_key:
        setattr(login, PASTE_IS_API_KEY_ATTR, True)
    return login


def _lazy_refresh(module: str, attr: str) -> RefreshFn:
    async def refresh(creds: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        fn = getattr(importlib.import_module(module), attr)
        return await fn(creds, **kwargs)

    return refresh


def _oauth_api_key(creds: dict[str, Any]) -> str:
    return creds["access"]


def _token_plan_wire_key(creds: dict[str, Any]) -> str:
    """The Token Plan's OAuth row carries TWO tokens with different jobs.

    ``access`` is the QwenCloud management token (usage:read) and is rejected
    by the inference endpoint; ``api_key`` is the pasted ``sk-sp-…`` key the
    wire actually wants. Falls back to ``access`` for hand-written rows so the
    credential still authenticates something rather than raising KeyError.
    """
    return creds.get("api_key") or creds["access"]


def create_api_key_login(provider_label: str, auth_url: str, instructions: str = "") -> LoginFn:
    """Paste-an-API-key "login" for providers without real OAuth.

    Opens the dashboard URL, prompts for a paste, returns the trimmed key:
    prompt for a paste, return the trimmed key (a ``str`` — AuthStore
    stores it as an ``api_key`` credential with ``source="login"``).

    The prompt is the ENTIRE flow, which is why the returned callable tags
    itself with :data:`PASTE_PROMPT_ATTR`: a host that attaches no
    ``on_manual_code_input`` cannot log in to any of these providers, so the
    requirement has to reach the host rather than being discovered as an error
    after the browser has already been opened.
    """

    async def login(callbacks: LoginCallbacks, **_kwargs: Any) -> str:
        # Imported HERE, not at module scope. ``callback_server`` pulls in
        # http.server, ssl, email and 150-odd other modules (~138 ms), and this
        # registry is on the model picker's path — every interactive session
        # paid for a loopback HTTP server it only needs if the user logs in.
        from local_operator.providers.oauth.callback_server import (
            LoginCancelledError,
            maybe_await,
        )

        if callbacks.on_manual_code_input is None:
            # Checked BEFORE the URL is surfaced. A host with no prompt cannot
            # finish this flow, and announcing "opening your browser to
            # authorize" first told the user to go and get a key that was then
            # refused — the failure this check replaces. Every shipped host now
            # attaches a prompt (see ``accepts_paste_prompt``), so this is a
            # contract violation by an embedder rather than a user-facing path,
            # and it says which hook is missing instead of naming a key prompt
            # the user was never offered.
            raise ValueError(
                f"{provider_label} login reads an API key, but this interface "
                "provided no on_manual_code_input callback to read it with."
            )
        if callbacks.on_auth_url is not None:
            await maybe_await(callbacks.on_auth_url(auth_url, instructions=instructions or None))
        pasted = await maybe_await(callbacks.on_manual_code_input())
        if pasted is None:
            raise LoginCancelledError(f"{provider_label} login cancelled")
        key = pasted.strip()
        if not key:
            # An empty paste is a CANCEL, not a credential. Storing it would
            # write a blank api_key row that shadows a working env key and turns
            # every later request into an auth error with no visible cause.
            raise LoginCancelledError(f"{provider_label} login cancelled")
        return key

    setattr(login, PASTE_PROMPT_ATTR, True)
    setattr(login, PASTE_IS_API_KEY_ATTR, True)
    setattr(login, "__lo_api_key_login__", True)
    return login


def _anthropic_env_key() -> str | None:
    # OAuth-issued tokens win over raw API keys.
    return os.environ.get("ANTHROPIC_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")


PROVIDER_REGISTRY: list[ProviderDefinition] = [
    ProviderDefinition(
        id="openai",
        search_aliases=(
            "gpt",
            "chatgpt",
            "codex",
        ),
        name="OpenAI (ChatGPT Plus/Pro)",
        env_keys="OPENAI_API_KEY",
        login=_lazy_login("local_operator.providers.oauth.openai", "login_openai"),
        refresh_token=_lazy_refresh(
            "local_operator.providers.oauth.openai", "refresh_openai_token"
        ),
        get_api_key=_oauth_api_key,
        callback_port=1455,
        base_url="https://api.openai.com/v1",
    ),
    ProviderDefinition(
        id="openai-device",
        search_aliases=(
            "gpt",
            "chatgpt",
            "codex",
        ),
        name="OpenAI (ChatGPT device code)",
        env_keys="OPENAI_API_KEY",
        login=_lazy_login("local_operator.providers.oauth.openai", "login_openai_device"),
        refresh_token=_lazy_refresh(
            "local_operator.providers.oauth.openai", "refresh_openai_token"
        ),
        get_api_key=_oauth_api_key,
        store_credentials_as="openai",
        base_url="https://api.openai.com/v1",
    ),
    ProviderDefinition(
        id="anthropic",
        search_aliases=(
            "claude",
            "sonnet",
            "opus",
            "haiku",
        ),
        name="Anthropic (Claude Pro/Max)",
        env_keys=_anthropic_env_key,
        login=_lazy_login("local_operator.providers.oauth.anthropic", "login_anthropic"),
        refresh_token=_lazy_refresh(
            "local_operator.providers.oauth.anthropic", "refresh_anthropic_token"
        ),
        get_api_key=_oauth_api_key,
        callback_port=54545,
        paste_code_flow=True,
        base_url="https://api.anthropic.com",
        wire="anthropic",
    ),
    ProviderDefinition(
        id="kimi",
        search_aliases=(
            "moonshot",
            "k2",
            "k3",
        ),
        name="Kimi (Moonshot)",
        env_keys="KIMI_API_KEY",
        login=_lazy_login("local_operator.providers.oauth.kimi", "login_kimi"),
        refresh_token=_lazy_refresh("local_operator.providers.oauth.kimi", "refresh_kimi_token"),
        get_api_key=_oauth_api_key,
        base_url="https://api.moonshot.cn/v1",
        # The coding-plan host, used only for OAuth sign-ins. It is a different
        # PLATFORM from the mainland API-key host above, not merely a different
        # path: it is where the subscription's models live (`k3`, `k3-256k`,
        # `kimi-for-coding`), and the mainland host rejects the OAuth bearer
        # with 401 "Invalid Authentication". Confirmed live against both hosts.
        oauth_base_url="https://api.kimi.com/coding/v1",
    ),
    ProviderDefinition(
        id="xai",
        search_aliases=("grok",),
        name="xAI (Grok API key)",
        env_keys="XAI_API_KEY",
        login=create_api_key_login("xAI", "https://console.x.ai/"),
        base_url="https://api.x.ai/v1",
    ),
    ProviderDefinition(
        id="xai-oauth",
        search_aliases=("grok",),
        name="xAI (Grok OAuth)",
        login=_lazy_login("local_operator.providers.oauth.xai", "login_xai"),
        refresh_token=_lazy_refresh("local_operator.providers.oauth.xai", "refresh_xai_token"),
        get_api_key=_oauth_api_key,
        store_credentials_as="xai",
        base_url="https://api.x.ai/v1",
    ),
    ProviderDefinition(
        id="deepseek",
        search_aliases=("ds",),
        name="DeepSeek",
        env_keys="DEEPSEEK_API_KEY",
        login=create_api_key_login("DeepSeek", "https://platform.deepseek.com/api_keys"),
        base_url="https://api.deepseek.com/v1",
    ),
    ProviderDefinition(
        id="zai",
        # Z.AI sells GLM under two names and users type both. "zhipu"/"bigmodel"
        # are the CN-facing brand for the same models, so a user who came here
        # for "glm" or "zhipu" finds the provider they actually want.
        search_aliases=(
            "glm",
            "zhipu",
            "bigmodel",
            "z-ai",
        ),
        # Names the CREDENTIAL, not the plan. Both Z.AI entries route to the
        # same coding-plan base URL, so a name implying a plan difference sends
        # a user to the wrong row for the wrong reason. `xai`/`xai-oauth` set
        # the precedent this follows.
        name="Z.AI (GLM API key)",
        env_keys="ZAI_API_KEY",
        # No instruction line: the prompt row below it already reads "Paste your
        # Z.AI API key", and Z.AI's dashboard calls it exactly that, so there is
        # no vendor-specific term worth a second line. Matches the convention
        # #139 established when it dropped four identical duplicates.
        login=create_api_key_login("Z.AI", "https://z.ai/manage-apikey/apikey-list"),
        # The CODING-plan path, not the general `/api/paas/v4` endpoint. Requests
        # sent here consume GLM Coding Plan quota; the general endpoint bills the
        # account balance instead, which is the same key silently spending the
        # wrong budget. Mirrors omp's `zhipuCodingPlanModelManagerOptions`.
        base_url="https://api.z.ai/api/coding/paas/v4",
    ),
    ProviderDefinition(
        id="zai-oauth",
        search_aliases=(
            "glm",
            "zhipu",
            "bigmodel",
            "z-ai",
        ),
        name="Z.AI (GLM browser sign-in)",
        # Browser sign-in rather than a pasted key. The flow ends by minting a
        # durable `id.secret` API key, which is what `access` holds and what the
        # wire receives -- so this shares `zai`'s credential row, base URL and
        # models, exactly as `xai-oauth` shares `xai`'s.
        login=_lazy_login("local_operator.providers.oauth.zai", "login_zai"),
        get_api_key=_oauth_api_key,
        store_credentials_as="zai",
        # Pinned by the provider's OAuth client registration; port fallback is
        # refused, which `ZaiOAuthFlow` states again as `allow_port_fallback`.
        callback_port=54548,
        # Paste fallback for when the browser cannot reach this machine (a
        # remote or headless session), as for anthropic. The prompt accepts the
        # whole redirect URL from the address bar, which is what a user has in
        # front of them in that situation, as well as a bare authorization
        # code; `_parse_pasted_callback` owns the shapes.
        paste_code_flow=True,
        base_url="https://api.z.ai/api/coding/paas/v4",
        # No refresh_token: the minted key never expires, so there is nothing to
        # refresh. `expires: None` stops AuthStore from ever trying.
    ),
    ProviderDefinition(
        id="google",
        search_aliases=(
            "gemini",
            "vertex",
            "aistudio",
        ),
        name="Google (Gemini)",
        env_keys="GOOGLE_AI_STUDIO_API_KEY",
        login=create_api_key_login(
            "Google AI Studio",
            "https://aistudio.google.com/apikey",
            "The console calls it an AI Studio API key.",
        ),
        base_url="https://generativelanguage.googleapis.com",
        wire="google",
    ),
    ProviderDefinition(
        id="mistral",
        search_aliases=(
            "codestral",
            "magistral",
        ),
        name="Mistral AI",
        env_keys="MISTRAL_API_KEY",
        login=create_api_key_login("Mistral", "https://console.mistral.ai/api-keys"),
        base_url="https://api.mistral.ai/v1",
    ),
    *[
        ProviderDefinition(
            id=provider_id,
            name=name,
            search_aliases=("local", "self-hosted"),
            allows_missing_api_key=True,
            local_setup=True,
            base_url=base_url,
        )
        for provider_id, (name, base_url, _url) in LOCAL_PRESETS.items()
    ],
    ProviderDefinition(
        id="openrouter",
        search_aliases=(
            "or",
            "router",
        ),
        name="OpenRouter",
        env_keys="OPENROUTER_API_KEY",
        login=create_api_key_login("OpenRouter", "https://openrouter.ai/keys"),
        base_url="https://openrouter.ai/api/v1",
    ),
    ProviderDefinition(
        id="elevenlabs",
        search_aliases=("eleven",),
        name="ElevenLabs",
        env_keys="ELEVENLABS_API_KEY",
        login=create_api_key_login(
            "ElevenLabs",
            "https://elevenlabs.io/app/settings/api-keys",
            "Paste the API key from your ElevenLabs account settings.",
        ),
        base_url="https://api.elevenlabs.io/v1",
        # SPEECH-ONLY (mobile STT, 2026-09-28). The key is storable so the
        # phone's voice path can run ``provider_stt_elevenlabs``; the provider
        # is deliberately absent from every surface that offers or resolves a
        # CHAT model (`speech_only` above documents the enforcement sites).
        speech_only=True,
        # The wire fact behind the flag above, declared where the wire is:
        # ElevenLabs serves speech-to-text AND text-to-speech (the voicing
        # settings' provider voice, ``tts``) and never chat. ``speech_only``
        # above is deliberately untouched: it is the chat-enforcement flag with
        # its own pinned call sites, and "never chat" is still true.
        capabilities=frozenset({"stt", "tts"}),
    ),
    ProviderDefinition(
        id="openai-key",
        search_aliases=("openai-api-key", "openai-speech"),
        name="OpenAI (API key)",
        # NO ``env_keys``, on purpose, and that is the whole safety property of
        # this row: this login exists for OpenAI SPEECH under the user's own API
        # key, and the speech availability rule is "advertised only from a key
        # the user stored here". With an env name declared, an ambient
        # ``OPENAI_API_KEY`` would surface as ``has_credential`` on every view
        # of this row (the registry view's env-OR-store hint) and invite a
        # reader to treat it as a login. Declaring none makes that impossible
        # by construction. The cost is accepted: this row's key is not
        # resolvable from the environment, and the chat ``openai`` row keeps
        # its own ``OPENAI_API_KEY`` for chat. Provider-key checks
        # (``key_check``) and the desktop's "save a key" route are keyed on
        # ``env_keys is not None`` for the SYNTHESIZED api-key method, but this
        # row carries its own real ``create_api_key_login`` method, so the
        # desktop still lists it as an API-key method.
        login=create_api_key_login(
            "OpenAI",
            "https://platform.openai.com/api-keys",
            "Paste an API key from the OpenAI platform. This is separate from a "
            "ChatGPT sign-in, which cannot be used for speech.",
        ),
        # Its OWN credential namespace (``storage_id`` on the wire): the row is
        # stored under ``openai-key``, never under ``openai``. That is the
        # DEFAULT -- ``store_credentials_as`` is deliberately left unset, because
        # setting it names a flavour ALIAS onto another provider (``openai-device``
        # -> ``openai``) and every "is this a flavour" rule keys on that field being
        # truthy (the catalogue drops such rows, the CLI marks them). Aliasing onto
        # ``openai`` is exactly what this row must not do: it would put a platform
        # API key beside the ChatGPT OAuth rows the chat provider's cascade walks,
        # and the key would be routed to the ChatGPT backend as a chat credential.
        base_url="https://api.openai.com/v1",
        # SPEECH-ONLY, same reasoning as ElevenLabs: the row is a login the
        # speech cascade reads, and must stay off every surface that offers or
        # resolves a CHAT model (the chat provider is ``openai``). Declared in
        # the enforcement flag and in the wire fact.
        speech_only=True,
        capabilities=frozenset({"tts"}),
        # The legacy provider-class store row the OpenAI speech rung keeps
        # honouring (see ``legacy_store_keys``); a store row, never the env.
        legacy_store_keys=("OPENAI_API_KEY",),
    ),
    ProviderDefinition(
        id="radient",
        search_aliases=("radient-oauth",),
        name="Radient",
        env_keys="RADIENT_API_KEY",
        login=_lazy_login("local_operator.providers.oauth.radient", "login_radient"),
        refresh_token=_lazy_refresh(
            "local_operator.providers.oauth.radient", "refresh_radient_token"
        ),
        get_api_key=_oauth_api_key,
        callback_port=54549,
        # Quoted from the single constant rather than spelled again: the desktop
        # hub transport derives its hub base from THIS definition
        # (``server/routes/desktop_radient.py`` ``base_url()``), so a separate
        # literal here was a third source for a host ``env.py`` calls the one
        # place it is written — and the one a hand-set bare value has to agree
        # with on the ``/v1`` segment.
        base_url=DEFAULT_RADIENT_API_BASE_URL,
    ),
    ProviderDefinition(
        id="radient-key",
        search_aliases=("radient-api-key",),
        name="Radient (API key)",
        env_keys="RADIENT_API_KEY",
        login=create_api_key_login(
            "Radient", "https://radienthq.com/", "The console calls it a Radient Pass key."
        ),
        store_credentials_as="radient",
        base_url=DEFAULT_RADIENT_API_BASE_URL,
    ),
    ProviderDefinition(
        id="alibaba",
        search_aliases=(
            "qwen",
            "dashscope",
            "tongyi",
        ),
        name="Alibaba Cloud (Qwen)",
        env_keys="ALIBABA_CLOUD_API_KEY",
        login=create_api_key_login(
            "Alibaba Cloud",
            "https://dashscope-intl.console.aliyun.com/",
            "The console calls it a DashScope API key.",
        ),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ),
    ProviderDefinition(
        id="alibaba-token-plan",
        search_aliases=(
            "token-plan",
            "tokenplan",
            "qwencloud",
        ),
        name="QwenCloud Token Plan",
        # The brand its users know: the name's first two words are the vendor's
        # cloud brand, and the desktop's own override table carried exactly this
        # value -- moved here so the registry is the one place it lives.
        brand="QwenCloud",
        env_keys="ALIBABA_TOKEN_PLAN_API_KEY",
        login=create_api_key_login(
            "QwenCloud Token Plan",
            "https://home.qwencloud.com/billing/subscription/token-plan-individual",
            "It starts with sk-sp-.",
        ),
        get_api_key=_token_plan_wire_key,
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    ),
    ProviderDefinition(
        id="alibaba-token-plan-oauth",
        search_aliases=("token-plan",),
        name="QwenCloud Token Plan (usage OAuth)",
        # Paste-requiring as well as device-flow: this login collects the
        # ``sk-sp-…`` inference key BEFORE it starts the device flow, so it is
        # unusable from a host that cannot prompt (see
        # ``login_qwencloud_token_plan``).
        login=_lazy_login(
            "local_operator.providers.oauth.qwencloud",
            "login_qwencloud_token_plan",
            requires_paste_prompt=True,
            # The prompt reads the inference key the wire then sends, so it is
            # checked like any pasted key even though the login is a device flow.
            paste_is_api_key=True,
        ),
        store_credentials_as="alibaba-token-plan",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    ),
    ProviderDefinition(
        id="typesafe",
        search_aliases=("jev", "typesafe-jev"),
        name="TypeSafe (Jev)",
        # Tuple form, primary first: TypeSafe's own console issues a
        # ``TYPESAFE_API_KEY`` and the model's original name is ``JEV_API_KEY``,
        # so the release reads whichever the operator already exported. Two
        # NAMES rather than a callable because the login-status surfaces read
        # the names themselves (see ``credential_file_names``).
        env_keys=("TYPESAFE_API_KEY", "JEV_API_KEY"),
        login=create_api_key_login(
            "TypeSafe",
            "https://typesafe.ai/",
            "Paste the API key from the TypeSafe console.",
        ),
        base_url="https://api.typesafe.ai/v1",
        # DECISION-ONLY (docs/design/classification-layer.md §9): the
        # classification layer's second cascade leg reaches Jev through
        # ``/v1/systemone``, and ``chat/completions`` is rejected on every host.
        # The login is here because the KEY has to be storable — the provider is
        # deliberately absent from every surface that offers or resolves a CHAT
        # model, which is what ``decision_only`` records.
        decision_only=True,
    ),
    ProviderDefinition(
        id="test",
        search_aliases=("mock",),
        name="Test (mock)",
        allows_missing_api_key=True,
        wire="mock",
    ),
]

_BY_ID: dict[str, ProviderDefinition] = {p.id: p for p in PROVIDER_REGISTRY}

# Legacy ``--hosting`` aliases (noop behaved like the mock host).
_ALIASES: dict[str, str] = {"noop": "test"}


def get_provider_definition(provider_id: str) -> ProviderDefinition | None:
    """Look up a provider by id or legacy alias; ``None`` when unknown."""
    return _BY_ID.get(_ALIASES.get(provider_id, provider_id))


def is_mock_provider(provider_id: str) -> bool:
    """Whether ``provider_id`` serves the deterministic TEST wire.

    The ONE PREDICATE for "is this the test hosting", shared by the places that
    ask the question from a provider ID — ``model/configure.py::configure_model``
    (which has only the id) and the store-side reader
    (``session.model_selection.session_uses_test_hosting``), plus the harness
    callers that gate themselves. Spelling the wire name in a second place is how
    two answers to one question drift, and drift here fails OPEN — a mock session
    that banners the operator — which is the failure this whole rule exists to
    stop.

    ``providers/clients.py::client_for_spec`` does NOT call it, and that is not
    drift: it already holds the resolved ``ProviderDefinition``, so asking this
    function for the same object's ``wire`` would be a second lookup of a value
    in hand. It tests ``definition.wire == "mock"`` on that object, and this
    predicate reads the same field through :func:`get_provider_definition` — one
    fact, two readers, each as direct as its starting point allows.

    Keyed on the WIRE rather than on the ``test`` id so a future mock provider
    is covered by construction; resolved through
    :func:`get_provider_definition`, so the legacy ``noop`` alias answers too.
    """
    definition = get_provider_definition(provider_id)
    return definition is not None and definition.wire == "mock"


def known_provider_ids() -> tuple[str, ...]:
    """Every id :func:`get_provider_definition` answers for, aliases included.

    Public because the desktop catalogue PUBLISHES this vocabulary to a renderer
    that has to reproduce the same lookup: the admission rule refuses a
    whole-draft ``/login <id>`` only when the id resolves, so a renderer holding
    just the primary ids would plan a message for an alias the backend still
    runs the command for. One alias exists today (``noop`` -> ``test``); the
    point is that it is derived here rather than hand-listed on the wire.
    """
    return tuple(sorted(set(_BY_ID) | set(_ALIASES)))


def provider_brand(definition: ProviderDefinition) -> str:
    """The clean title for ``definition``: ``brand`` when set, else derived.

    ONE derivation for every surface that shows a provider's name away from its
    id -- the desktop's picker rows and the TUI's rows read this rather than
    each carrying the same parenthetical-stripping fallback (the UI's
    ``brandOf`` keeps its copy only as the pre-existing fallback for a census
    that predates ``brand``).

    The derivation is the UI's, moved server-side: strip ONE trailing
    parenthetical -- ``"OpenAI (ChatGPT Plus/Pro)"`` -> ``"OpenAI"`` -- because
    the parenthetical is a login-flavour qualifier that the row's id and detail
    column already carry. A name with no parenthetical is its own brand; the
    rows where that is wrong (``alibaba-token-plan``) say so explicitly in
    ``ProviderDefinition.brand``.
    """
    if definition.brand:
        return definition.brand
    name = definition.name.strip()
    if name.endswith(")") and "(" in name:
        return name[: name.index("(")].strip()
    return name


def list_login_providers() -> list[ProviderDefinition]:
    """Providers offering interactive login or server setup, in registry order."""
    return [p for p in PROVIDER_REGISTRY if p.login is not None or p.local_setup]


#: Providers that RESELL other providers' models rather than serving their own.
#:
#: Their catalogues overlap the direct providers almost entirely, so the same model
#: is reachable two ways and something has to decide which the user meant. The
#: direct route wins: it is one hop instead of two, it is the credential the user
#: just created when they logged in, and provider-native behaviour (Anthropic's
#: cache-control breakpoints, OpenAI's reasoning effort) is only reliable there.
#: An aggregator remains the right answer for models with no direct route, which is
#: most of its list.
AGGREGATOR_PROVIDERS = frozenset({"openrouter", "radient", "radient-key"})


def is_decision_only(provider_id: str | None) -> bool:
    """Whether ``provider_id`` serves decision calls and never chat completions.

    ONE exported predicate rather than a ``definition.decision_only`` read at
    each site, for the reason :func:`local_operator.model.discovery.is_meta_route_id`
    is one too: an unknown id, an empty string and a legacy alias have to answer
    the same way everywhere, and a second ``get_provider_definition`` call spelled
    out per surface is how one of them ends up testing the raw string.

    CASE AND PADDING ARE NORMALISED HERE, which is load-bearing rather than tidy:
    ``get_provider_definition`` is a plain dict lookup over lowercase ids, so
    ``TypeSafe``/``TYPESAFE``/`` typeSafe `` used to answer ``False`` — a spelling
    in a wire frame or a hand-edited ``config.yml`` was enough to put a session on
    a provider that 400s every turn (review round 3, MAJOR 1, reproduced on a real
    session). Normalising here fixes every door at once and can only ever refuse
    MORE, never less: it never turns a ``True`` into a ``False``.

    Alias-aware through :func:`get_provider_definition`, and tolerant of ``None``
    because every call site here is a filter over data that may carry no provider
    at all. ``False`` for an unknown provider: "do not offer this" is a claim
    only a resolved definition may make.
    """
    if not provider_id:
        return False
    canonical = str(provider_id).strip().lower()
    definition = get_provider_definition(canonical)
    return bool(definition is not None and definition.decision_only)


def decision_only_message(provider_id: str) -> str:
    """The ONE sentence for a provider that can serve no chat completion.

    Shared by every door that refuses one, and there are now five: the config /
    agent / flag hosting preflight and a resume's stored row (``session_factory``),
    the desktop pick boundary, the draft-marker reader, and ``build_model_spec`` —
    which is where a LIVE switch lands, since ``/model``, the viewer's slash result
    and the wire's ``set_model`` all build their spec through it. A second spelling
    of this fact would be a second chance to tell a user something the other surface
    does not say, and the message is the only place the reason is explained at all.

    The sentence stops at the fact: each caller appends the remedy that fits its own
    surface (``/model`` from inside a session, ``local-operator config edit`` from a
    config file), which is why this is a fragment of a message rather than a whole
    one.
    """
    return (
        f"Hosting '{provider_id}' serves decision-model calls, not chat completions, "
        "so no session can run on it."
    )


def is_speech_only(provider_id: str | None) -> bool:
    """Whether ``provider_id`` serves speech (STT and/or TTS) and never chat completions.

    The sibling of :func:`is_decision_only`, with the same contract: ONE
    exported predicate (alias-aware, case/padding-normalised, tolerant of
    ``None``, ``False`` for unknown ids), so every door that skips a
    decision-only provider skips a speech-only one the same way. The two flags
    answer the same question for two provider classes, and a site that asks
    only one of them is how a provider whose wire cannot answer a turn becomes
    selectable again.
    """
    if not provider_id:
        return False
    canonical = str(provider_id).strip().lower()
    definition = get_provider_definition(canonical)
    return bool(definition is not None and definition.speech_only)


def speech_wire_noun(provider_id: str) -> str:
    """What a speech-only provider's wire serves, as a noun phrase for a sentence.

    ONE derivation for every door that refuses a speech-only hosting
    (:func:`speech_only_message` and ``model/configure.py``'s selection refusal),
    read off ``capabilities`` rather than spelled per door: a TTS-only row
    (``openai-key``) is not "speech-to-text", and a hand-written sentence in a
    second place is how that stayed wrong for one door. ``stt`` is tested FIRST
    and by membership, so ElevenLabs (``{stt, tts}``) keeps the sentence its
    pinned tests and users know, and any set that is not exactly one of the two
    -- empty, unknown id, a future member -- says the honest generic "speech"
    rather than a wrong specific one.
    """
    definition = get_provider_definition(provider_id)
    capabilities = definition.capabilities if definition is not None else frozenset()
    if "stt" in capabilities:
        return "speech-to-text"
    if "tts" in capabilities:
        return "text-to-speech"
    return "speech"


def speech_only_message(provider_id: str) -> str:
    """The ONE sentence for a speech-only provider, refused as a chat hosting.

    Shared by every door that refuses one, for the same reason
    :func:`decision_only_message` is: the fact must have one spelling, and the
    sentence stops at the fact so each caller can append the remedy its own
    surface can offer. Named apart from ``decision_only_message`` because the
    two providers ARE different things to a reader — one is a classification
    backend, this one is a voice provider — and collapsing them into one
    sentence would tell the reader something false about whichever it is not.
    """
    return (
        f"Hosting '{provider_id}' serves {speech_wire_noun(provider_id)}, not chat "
        "completions, so no session can run on it."
    )


def credential_provider_id(provider_id: str) -> str:
    """The provider id a credential for ``provider_id`` is actually STORED under.

    Login flavours do not own their credentials. ``xai-oauth``, ``openai-device``,
    ``alibaba-token-plan-oauth`` and ``zai-oauth`` are alternate ways of
    authenticating ``xai``, ``openai``, ``alibaba-token-plan`` and ``zai``, and
    their login writes ONE row under the aliased name (``store_credentials_as``)
    so that logging in either way yields one account rather than two
    half-configured ones.

    Every credential lookup therefore has to be translated before it reaches a
    store whose queries are exact (``WHERE provider = ?``). Asking for the
    literal flavour id matches no row, which does not read as "translate this" —
    it reads as "this provider has no credential", i.e. the "No API key
    configured for provider 'xai-oauth'" that an OAuth login is specifically
    supposed to make impossible.

    Unknown ids and providers that store under their own name pass through, so
    this is safe to call unconditionally on any provider id.
    """
    definition = get_provider_definition(provider_id)
    return (definition.store_credentials_as or definition.id) if definition else provider_id


def resolve_env_key(provider_id: str) -> str | None:
    """Resolve the provider's API key from the environment.

    Handles all three forms of ``env_keys``: a plain variable name, a tuple of
    names read PRIMARY FIRST, or a callable that picks among several
    (feature-flag style).

    Alias-aware: a login flavour declares no env var of its own (there is no
    ``XAI_OAUTH_API_KEY``), but it serves the same endpoint as the provider it
    stores under, so the base provider's var authenticates it too. This is the
    ONE env reader the store's ``_env_api_key``, the controller's ``is_usable``
    and the catalogue enrichment share; a second, literal one is how a flavour
    resolves at stream time yet reads "needs login" on every status surface.
    """
    definition = get_provider_definition(provider_id)
    if definition is None or definition.env_keys is None:
        definition = get_provider_definition(credential_provider_id(provider_id))
    if definition is None or definition.env_keys is None:
        return None
    keys = definition.env_keys
    if callable(keys):
        return keys()
    # The tuple is ordered by the REGISTRY ROW, not by the caller: a host with
    # both variables set must prefer the one the provider's own login writes,
    # or an old spelling left in a shell profile silently outranks it.
    for name in (keys,) if isinstance(keys, str) else keys:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _config_root(base: Path | None) -> Path | None:
    """Normalise a config-root argument: a ``Path``/``str``, else ``None``.

    The store and file legs take ``base`` straight into filesystem calls, so a
    value that is not a path is treated as "no override" and the HOME-derived
    default applies rather than being handed to ``open()``. Callers pass
    ``getattr(manager, "config_dir", None)``, which on a test double is an
    auto-generated attribute rather than a path.
    """
    if base is None or isinstance(base, Path):
        return base
    if isinstance(base, str):
        return Path(base)
    return None


def provider_secret_value(env_key: str, *, base: Path | None = None) -> str | None:
    """The VALUE of the provider-owned store row for ``env_key``, else ``None``.

    The store reader every provider-key resolution leg shares, so the cascade
    (``auth_store._env_api_key``), the static-key readers and the search
    transports cannot disagree about whether a key the operator stored runs a
    provider. ``env_key`` is the ENV VAR NAME (``OPENROUTER_API_KEY``); the row
    lives under ``LOP_PROVIDER_<env_key>`` and is read with ``role="provider"``,
    the only namespace that may touch the reserved prefix.

    **Never raises, and never imports the crypto stack at module scope.** The
    store is reached lazily inside the function: this module is imported on the
    CLI startup path, and ``tests/unit/test_import_graph.py`` pins that the CLI
    import must not pull the store's crypto machinery. A store that is absent,
    locked, corrupted or served by an incompatible broker degrades to ``None``
    — a MISSING credential, which the caller resolves by falling through to its
    next leg — rather than turning a resolution into an error. That trade is
    deliberate: the store is an optional capability (design §13), and a
    credential lookup that raised would take down the provider picker on any
    host whose store has not been created.
    """
    if not env_key:
        return None
    from local_operator.secrets.access import retrieve_secret
    from local_operator.secrets.errors import SecretStoreError
    from local_operator.secrets.keys import store_path
    from local_operator.secrets.store import provider_secret_name

    root = _config_root(base)
    # **The existence guard is load-bearing, not an optimisation (R1).**
    # `retrieve_secret` reaches `ensure_broker` BEFORE it considers whether a
    # store exists, and `ensure_broker` creates `secrets/` and spawns
    # `python -m local_operator.secrets.brokerd`. So an unguarded READ on the
    # very common host that has never run `lop secret set` leaves both a
    # directory and a detached daemon behind — the "leaves new state on a read"
    # fault class `info/collect.py` and `mcp/credentials.py` explicitly ban, and
    # this is the hottest such path: the credential cascade, the model
    # catalogue, the classification legs and every search transport all read
    # through here. `stored_provider_env_keys` below and the qwencloud ticket
    # reader in `providers/controller.py` already carry the same gate.
    if not store_path(root).exists():
        return None
    try:
        raw = retrieve_secret(provider_secret_name(env_key), root, role="provider")
    except (SecretStoreError, OSError, ValueError):
        return None
    # ``surrogateescape``, matching the legacy plaintext reader this consolidates
    # (``secrets.legacy_env.read_credentials``, the migration's only reader): a
    # credential is an arbitrary byte string, and ``errors="replace"`` would
    # substitute U+FFFD so the provider authenticates with a corrupted key
    # instead of the stored one.
    value = raw.decode("utf-8", "surrogateescape").strip()
    return value or None


def provider_env_key(provider_id: str, *, base: Path | None = None) -> str | None:
    """Resolve ``provider_id``'s API key value, STORE FIRST then environment.

    Returns the same VALUE :func:`resolve_env_key` does, but its resolution order
    is the one the credential consolidation introduces: a provider-class store
    row wins and the process environment is the second leg.

    Why store-first: the store is the sanctioned home for a key the harness now
    writes (``lop credential update``, ``lop search setup``), and a stale export
    left in a shell profile must not outrank the value the operator deliberately
    saved. The env leg stays behind the store for the opposite reason it always
    did — an explicit ``ANTHROPIC_API_KEY`` export is still a real instruction,
    and is not what PR2a removes.

    Alias-aware like :func:`resolve_env_key`: a login flavour resolves through
    the name of the provider it stores under.
    """
    definition = get_provider_definition(provider_id)
    if definition is None or definition.env_keys is None:
        definition = get_provider_definition(credential_provider_id(provider_id))
    if definition is None or definition.env_keys is None:
        return None
    names = env_key_names(provider_id) or credential_file_names(provider_id)
    return first_provider_key(names, resolve_env_key(provider_id), base=base)


def first_provider_key(
    names: Iterable[str], env_value: str | None = None, *, base: Path | None = None
) -> str | None:
    """First configured value across the store, then ``env_value``.

    The rung ORDER every provider-key reader shares, split out from
    :func:`provider_env_key` for the callers that carry their OWN key names and
    have already resolved their env value. Two spellings of that order is how
    one surface ends up reading a different credential than the one a login
    wrote.

    ``names`` are ENV VAR NAMES; each is looked up as a ``LOP_PROVIDER_<name>``
    store row. ``env_value`` is the caller's already-resolved environment value,
    if it has one, and is consulted when no store row holds the key.

    The plaintext ``credentials.env`` leg this function used to consult last is
    GONE (PR2a): the file is no longer a credential source, so a name the store
    does not hold and the environment does not export resolves to nothing rather
    than to a file read.
    """
    for name in names:
        stored = provider_secret_value(name, base=base)
        if stored:
            return stored
    if env_value:
        return env_value
    return None


def store_provider_key(
    env_key: str, value: str, *, description: str = "", base: Path | None = None
) -> None:
    """Write a provider-class store row for ``env_key`` (set or update).

    The ONE writer for the provider namespace, so every surface that saves a
    key — ``lop credential update``, ``lop search setup``, the credentials
    route's PATCH — files it under the same name with the same ``role``. Raises
    :class:`~local_operator.secrets.errors.SecretStoreError` for the caller to
    report; unlike the READERS this does not swallow failures, because a write
    the operator asked for and did not get must not look like success.

    ``base`` selects the config root the store lives under; ``None`` is the
    HOME-derived default, which is what the CLI wants. The credentials route
    passes the manager's own ``config_dir`` so a server configured with a
    non-default root writes its provider rows to the SAME root it reads its
    other state from.

    ``update`` is tried first and ``set`` second: ``set`` refuses an existing
    name by design (a mistyped name must not clobber a live credential), so a
    re-run that changed a key would otherwise fail with ``SecretExists``.
    """
    from local_operator.secrets.access import open_store, session_id
    from local_operator.secrets.errors import SecretExists, SecretStoreError
    from local_operator.secrets.store import provider_secret_name

    name = provider_secret_name(env_key)
    payload = value.encode("utf-8")
    store = open_store(_config_root(base), create=True)
    # ``update`` first, then ``set``: ``set`` refuses an existing name by design
    # (a mistyped name must not clobber a live credential), so a re-run that
    # changed a key would otherwise fail with ``SecretExists``. The fall-through
    # is on any store error — a missing row AND a store that does not exist yet
    # both land here — and ``set`` surfaces its own, more specific failure rather
    # than this branch hiding it.
    try:
        store.update(name, payload, role="provider", session_id=session_id())
        return
    except SecretStoreError:
        pass
    try:
        store.set(
            name,
            payload,
            description=description or f"Provider API key {env_key}",
            role="provider",
            session_id=session_id(),
        )
    except SecretExists:
        # Another writer created the row between the update and the set; the
        # caller's intent is still "this name holds this value".
        store.update(name, payload, role="provider", session_id=session_id())


def remove_provider_key(env_key: str, *, base: Path | None = None) -> bool:
    """Delete the provider-class row for ``env_key``; ``True`` if one was removed.

    Best-effort on an absent store (returns ``False``): a delete of something
    that is not there is the desired end state, not an error.

    Passes ``role="provider"``, which the store now REQUIRES for this name: the
    reserved namespace check on ``delete`` that keeps an agent surface from
    destroying a provider credential also refuses a caller that has not
    declared itself as the provider side.
    """
    from local_operator.secrets.access import open_store, session_id
    from local_operator.secrets.errors import SecretNotFound, SecretStoreError
    from local_operator.secrets.store import provider_secret_name

    try:
        open_store(_config_root(base)).delete(
            provider_secret_name(env_key), role="provider", session_id=session_id()
        )
        return True
    except (SecretNotFound, SecretStoreError, OSError):
        return False


def stored_provider_env_keys(base: Path | None = None) -> set[str]:
    """ENV KEY names that have a provider-class row in the store.

    The name-source reader the controller's ``persisted_providers`` rung uses in
    place of enumerating the plaintext file: it returns the env-key spelling
    (``OPENROUTER_API_KEY``), stripped of the reserved ``LOP_PROVIDER_`` prefix,
    so a caller can match it against :func:`credential_file_names` exactly as it
    matched the legacy file's keys.

    An absent, locked or damaged store yields an EMPTY set — "no provider rows I
    can see" — never an error, for the same availability reason
    :func:`provider_secret_value` documents. Enumeration must not become the one
    path that takes down the provider picker, which is precisely the failure the
    store's ``list`` verb was reworked to avoid.

    That promise has to cover the STORE'S OWN exception family and a malformed
    ROW, not just this module's: the store is a sqlite file, so a damaged one raises
    ``sqlite3.DatabaseError`` ("file is not a database") and a ``chmod 000`` one
    raises ``sqlite3.OperationalError`` — neither is an ``OSError`` — while a row
    whose BLOB columns hold text raises ``TypeError`` out of the decryptor. An
    earlier version of this tuple caught none of the three, so
    `GET /v1/desktop/models?live=true` (which reaches here through
    ``persisted_providers``) answered 500 for states this function is documented to
    survive (Q2-1, R3-1). For the row case the deeper fix belongs to
    ``secrets.store._decode``, which should raise ``SecretCorrupt`` for a byte
    column that is not bytes-like; see the clause's comment.
    ``sqlite3.ProgrammingError`` is re-raised the way ``usable_providers``
    re-raises it: a connection used from the wrong thread is a caller BUG, and a bug
    that dresses itself as "no provider rows" is one nobody finds.
    """
    from local_operator.secrets.access import open_store
    from local_operator.secrets.errors import SecretStoreError
    from local_operator.secrets.keys import store_path
    from local_operator.secrets.store import PROVIDER_SECRET_PREFIX

    try:
        root = _config_root(base)
        if not store_path(root).exists():
            return set()
        return {
            record.name[len(PROVIDER_SECRET_PREFIX) :]
            for record in open_store(root).list()
            if record.name.startswith(PROVIDER_SECRET_PREFIX)
        }
    except (SecretStoreError, OSError, ValueError):
        return set()
    except sqlite3.ProgrammingError:
        # Re-raised BEFORE the family below, which it belongs to: a connection
        # crossing threads (or a closed handle) is a caller bug, not an unreadable
        # store, and dressing it as "no provider rows" hides it forever. Same
        # precedent and same rationale as ``usable_providers``.
        raise
    except sqlite3.Error:
        # Damaged, locked or otherwise unopenable: the sqlite family's own answer
        # to "this file is not a store I can read". Degrades exactly like an
        # absent store, so no GET can be taken down by one.
        return set()
    except TypeError:
        # A ROW whose SHAPE is wrong, rather than a store that cannot be read: the
        # decryptor's ``bytes(row[1])`` (``name_index``), ``bytes(row[4])``
        # (``nonce``) and ``bytes(row[5])`` (``ciphertext``) each raise "string
        # argument without an encoding" when a column that must hold a BLOB holds
        # text instead — hand-tampering, or a record-header desync left by disk
        # damage. Caught for the same reason as the sqlite family above (this
        # reader's contract is "no provider rows I can see", never an error into a
        # GET), and NOT because this is the right layer: the principled fix lives
        # in ``secrets.store._decode``, which should validate that the byte columns
        # are bytes-like and raise ``SecretCorrupt`` so ``_enumerate`` reports the
        # row through ``damaged_records`` — the treatment ``_read_meta_int``
        # already got for the same class of hand-corrupted row. That is the
        # secrets layer's contract to keep, out of scope for a listing fix, and
        # recorded here rather than skipped (R3-1).
        #
        # THE PRICE of catching this broadly, stated because the ``ProgrammingError``
        # clause directly above re-raises for the same reason: a ``TypeError``
        # raised by a genuine BUG in the store's decode path, rather than by a
        # malformed row, is now reported as "no provider rows I can see" instead
        # of failing loudly. Round 3 accepted that trade knowingly (the alternative
        # leaves a 500 standing on a GET); the row-level fix above is what removes
        # the need for it.
        return set()


def env_key_names(provider_id: str) -> tuple[str, ...]:
    """Every env var NAME ``provider_id`` reads, primary first.

    The ONE reader of the tuple form. :func:`env_key_name` (the display name) and
    :func:`credential_file_names` (the legacy-store rungs the login status and
    the controller's ``is_usable`` walk) both take their answer from here, so a
    second variable cannot be honoured by one surface and invisible to another —
    the asymmetry that made ``lop credential update ANTHROPIC_API_KEY`` leave the
    phone's model sheet empty while the desktop listed every row.

    Empty for a provider with no name of its own — a callable resolver
    (:func:`_anthropic_env_key`), an unknown id, or an empty string — so it is
    safe to call unconditionally and iterate over.
    """
    definition = get_provider_definition(provider_id)
    if definition is None or definition.env_keys is None:
        return ()
    keys = definition.env_keys
    if callable(keys):
        return ()
    if isinstance(keys, str):
        return (keys,) if keys else ()
    return tuple(name for name in keys if name)


def env_key_name(provider_id: str) -> str | None:
    """The PRIMARY env var NAME for display (None for callable resolvers)."""
    names = env_key_names(provider_id)
    return names[0] if names else None


def credential_file_names(provider_id: str) -> list[str]:
    """The key NAMES ``provider_id`` can be configured under, every ``env_keys`` form.

    The name-source for the store-first rungs: ``env_key_name`` answers only for
    the plain-string form and returns ``None``
    for the callable one — today exactly ``anthropic`` — so any caller built on
    it alone silently drops the provider whose key the user is most likely to
    have set by hand. That is not hypothetical: it is how an install configured
    with ``lop credential update ANTHROPIC_API_KEY`` came back with an empty
    model sheet on the phone while the desktop listed 18 rows.

    Walks :func:`env_key_names`, not the single-name reader: a provider that reads
    two variables (``typesafe``: ``TYPESAFE_API_KEY``, else ``JEV_API_KEY``) is
    configured under EITHER of them, and a rung list carrying only the primary
    would report an operator who set the second one as not logged in.

    ``SupportedHostingProviders.requiredCredentials`` is the right second source
    rather than a hard-coded map here, because it is the same table the CLI, the
    server schema and the setup prompt already read — a private map beside them
    is free to drift from the name the command actually writes.

    Alias-aware in the same way :func:`resolve_env_key` is: a login flavour
    (``xai-oauth``) declares no key name of its own but the provider it stores
    under does, and that is the name a store row or the environment holds.

    Returns an empty list for an unknown provider or one with no key name at
    all, so it is safe to call unconditionally and iterate over.
    """
    from local_operator.model.registry import SupportedHostingProviders

    names: list[str] = []
    for candidate in (provider_id, credential_provider_id(provider_id)):
        for name in env_key_names(candidate):
            if name not in names:
                names.append(name)
        for detail in SupportedHostingProviders:
            if detail.id != candidate:
                continue
            for required in detail.requiredCredentials:
                if required and required not in names:
                    names.append(required)
    return names

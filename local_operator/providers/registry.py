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
import os
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

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
EnvKeys = str | Callable[[], str | None] | None

#: Attribute a login callable sets on itself to declare "I cannot complete
#: without reading text from the user". Read by
#: ``ProviderDefinition.__post_init__``, which is what carries the requirement
#: from the flow that has it out to the hosts that must honour it.
PASTE_PROMPT_ATTR = "__lo_requires_paste_prompt__"


@dataclasses.dataclass(frozen=True)
class ProviderDefinition:
    """The whole per-provider auth/routing record.

    - ``env_keys``: env var name OR a zero-arg callable returning the key's
      value (picking among several vars, feature-flag style).
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


def _lazy_login(module: str, attr: str, *, requires_paste_prompt: bool = False) -> LoginFn:
    """Dynamic-import thunk: keeps OAuth deps out of startup imports.

    ``requires_paste_prompt`` is carried on the returned callable rather than
    passed to the provider entry, because the whole point of the thunk is that
    the real login module is NOT imported at registry build time — so the only
    place the requirement can be stated without paying that import is here.
    """

    async def login(
        callbacks: LoginCallbacks,
        *,
        signal: AbortSignal | None = None,
        open_browser: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        fn = getattr(importlib.import_module(module), attr)
        if attr in ("login_anthropic", "login_openai"):
            return await fn(callbacks, signal=signal, open_browser=open_browser, **kwargs)
        return await fn(callbacks, signal=signal, **kwargs)

    if requires_paste_prompt:
        setattr(login, PASTE_PROMPT_ATTR, True)
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
        base_url="https://api.radienthq.com/v1",
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
        base_url="https://api.radienthq.com/v1",
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
        ),
        store_credentials_as="alibaba-token-plan",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
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

    Handles both forms of ``env_keys``: a plain variable name, or a callable
    that picks among several (feature-flag style).

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
    if callable(definition.env_keys):
        return definition.env_keys()
    return os.environ.get(definition.env_keys) or None


def env_key_name(provider_id: str) -> str | None:
    """The env var NAME for display (None for callable resolvers)."""
    definition = get_provider_definition(provider_id)
    if definition is None or definition.env_keys is None or callable(definition.env_keys):
        return None
    return definition.env_keys

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

if TYPE_CHECKING:
    from local_operator.providers.oauth.callback_server import LoginCallbacks

WireFormat = Literal["openai-compat", "anthropic", "google", "mock"]

# A login yields either the OAuth credentials mapping or, for paste-a-key
# providers, the bare key string.
LoginFn = Callable[..., Awaitable[str | dict[str, Any]]]
RefreshFn = Callable[..., Awaitable[dict[str, Any]]]
GetApiKeyFn = Callable[[dict[str, Any]], str]
EnvKeys = str | Callable[[], str | None] | None


@dataclasses.dataclass(frozen=True)
class ProviderDefinition:
    """The whole per-provider auth/routing record.

    - ``env_keys``: env var name OR a zero-arg callable returning the key's
      value (picking among several vars, feature-flag style).
    - ``allows_missing_api_key``: transport needs no bearer (local servers).
    - ``store_credentials_as``: alias the credential row under another
      provider id (xai-oauth ⇒ xai; openai-device ⇒ openai).
    - ``wire``: which wire client serves this provider.
    """

    id: str
    name: str
    env_keys: EnvKeys = None
    allows_missing_api_key: bool = False
    login: LoginFn | None = None
    refresh_token: RefreshFn | None = None
    get_api_key: GetApiKeyFn | None = None
    store_credentials_as: str | None = None
    callback_port: int | None = None
    paste_code_flow: bool = False
    base_url: str | None = None
    wire: WireFormat = "openai-compat"


def _lazy_login(module: str, attr: str) -> LoginFn:
    """Dynamic-import thunk: keeps OAuth deps out of startup imports."""

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

    return login


def _lazy_refresh(module: str, attr: str) -> RefreshFn:
    async def refresh(creds: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        fn = getattr(importlib.import_module(module), attr)
        return await fn(creds, **kwargs)

    return refresh


def _oauth_api_key(creds: dict[str, Any]) -> str:
    return creds["access"]


def create_api_key_login(provider_label: str, auth_url: str, instructions: str = "") -> LoginFn:
    """Paste-an-API-key "login" for providers without real OAuth.

    Opens the dashboard URL, prompts for a paste, returns the trimmed key:
    prompt for a paste, return the trimmed key (a ``str`` — AuthStore
    stores it as an ``api_key`` credential with ``source="login"``).
    """

    async def login(callbacks: LoginCallbacks, **_kwargs: Any) -> str:
        # Imported HERE, not at module scope. ``callback_server`` pulls in
        # http.server, ssl, email and 150-odd other modules (~138 ms), and this
        # registry is on the model picker's path — every interactive session
        # paid for a loopback HTTP server it only needs if the user logs in.
        from local_operator.providers.oauth.callback_server import maybe_await

        if callbacks.on_auth_url is not None:
            await maybe_await(callbacks.on_auth_url(auth_url, instructions=instructions or None))
        if callbacks.on_manual_code_input is None:
            raise ValueError(f"{provider_label} login requires an interactive code prompt")
        pasted = await maybe_await(callbacks.on_manual_code_input())
        if pasted is None:
            raise ValueError(f"{provider_label} login cancelled")
        return pasted.strip()

    return login


def _anthropic_env_key() -> str | None:
    # OAuth-issued tokens win over raw API keys.
    return os.environ.get("ANTHROPIC_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")


PROVIDER_REGISTRY: list[ProviderDefinition] = [
    ProviderDefinition(
        id="openai",
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
        name="Kimi (Moonshot)",
        env_keys="KIMI_API_KEY",
        login=_lazy_login("local_operator.providers.oauth.kimi", "login_kimi"),
        refresh_token=_lazy_refresh("local_operator.providers.oauth.kimi", "refresh_kimi_token"),
        get_api_key=_oauth_api_key,
        base_url="https://api.moonshot.cn/v1",
    ),
    ProviderDefinition(
        id="xai",
        name="xAI (Grok API key)",
        env_keys="XAI_API_KEY",
        login=create_api_key_login("xAI", "https://console.x.ai/", "Paste your xAI API key"),
        base_url="https://api.x.ai/v1",
    ),
    ProviderDefinition(
        id="xai-oauth",
        name="xAI (Grok OAuth)",
        login=_lazy_login("local_operator.providers.oauth.xai", "login_xai"),
        refresh_token=_lazy_refresh("local_operator.providers.oauth.xai", "refresh_xai_token"),
        get_api_key=_oauth_api_key,
        store_credentials_as="xai",
        base_url="https://api.x.ai/v1",
    ),
    ProviderDefinition(
        id="deepseek",
        name="DeepSeek",
        env_keys="DEEPSEEK_API_KEY",
        login=create_api_key_login(
            "DeepSeek", "https://platform.deepseek.com/api_keys", "Paste your DeepSeek API key"
        ),
        base_url="https://api.deepseek.com/v1",
    ),
    ProviderDefinition(
        id="google",
        name="Google (Gemini)",
        env_keys="GOOGLE_AI_STUDIO_API_KEY",
        login=create_api_key_login(
            "Google AI Studio",
            "https://aistudio.google.com/apikey",
            "Paste your Google AI Studio API key",
        ),
        base_url="https://generativelanguage.googleapis.com",
        wire="google",
    ),
    ProviderDefinition(
        id="mistral",
        name="Mistral AI",
        env_keys="MISTRAL_API_KEY",
        login=create_api_key_login(
            "Mistral", "https://console.mistral.ai/api-keys", "Paste your Mistral API key"
        ),
        base_url="https://api.mistral.ai/v1",
    ),
    ProviderDefinition(
        id="ollama",
        name="Ollama (local)",
        allows_missing_api_key=True,
        base_url="http://localhost:11434/v1",
    ),
    ProviderDefinition(
        id="openrouter",
        name="OpenRouter",
        env_keys="OPENROUTER_API_KEY",
        login=create_api_key_login(
            "OpenRouter", "https://openrouter.ai/keys", "Paste your OpenRouter API key"
        ),
        base_url="https://openrouter.ai/api/v1",
    ),
    ProviderDefinition(
        id="radient",
        name="Radient",
        env_keys="RADIENT_API_KEY",
        login=create_api_key_login(
            "Radient", "https://radienthq.com/", "Paste your Radient Pass key"
        ),
        base_url="https://api.radienthq.com/v1",
    ),
    ProviderDefinition(
        id="alibaba",
        name="Alibaba Cloud (Qwen)",
        env_keys="ALIBABA_CLOUD_API_KEY",
        login=create_api_key_login(
            "Alibaba Cloud",
            "https://dashscope-intl.console.aliyun.com/",
            "Paste your DashScope API key",
        ),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ),
    ProviderDefinition(
        id="test",
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
    """Providers offering an interactive login, in registry order."""
    return [p for p in PROVIDER_REGISTRY if p.login is not None]


#: Providers that RESELL other providers' models rather than serving their own.
#:
#: Their catalogues overlap the direct providers almost entirely, so the same model
#: is reachable two ways and something has to decide which the user meant. The
#: direct route wins: it is one hop instead of two, it is the credential the user
#: just created when they logged in, and provider-native behaviour (Anthropic's
#: cache-control breakpoints, OpenAI's reasoning effort) is only reliable there.
#: An aggregator remains the right answer for models with no direct route, which is
#: most of its list.
AGGREGATOR_PROVIDERS = frozenset({"openrouter", "radient"})


def resolve_env_key(provider_id: str) -> str | None:
    """Resolve the provider's API key from the environment.

    Handles both forms of ``env_keys``: a plain variable name, or a callable
    that picks among several (feature-flag style).
    """
    definition = get_provider_definition(provider_id)
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

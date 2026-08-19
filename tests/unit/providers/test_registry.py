"""Registry tests: legacy hosting resolution, field semantics, env keys."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    env_key_name,
    get_provider_definition,
    list_login_providers,
    resolve_env_key,
)

LEGACY_HOSTING_NAMES = [
    "radient",
    "openai",
    "anthropic",
    "google",
    "mistral",
    "ollama",
    "openrouter",
    "deepseek",
    "kimi",
    "alibaba",
    "xai",
    "zai",
]


@pytest.mark.parametrize("hosting", LEGACY_HOSTING_NAMES + ["test", "noop"])
def test_every_legacy_hosting_name_resolves(hosting: str) -> None:
    definition = get_provider_definition(hosting)
    assert definition is not None, f"legacy --hosting name must resolve: {hosting}"


def test_unknown_provider_returns_none() -> None:
    assert get_provider_definition("definitely-not-a-provider") is None


def test_registry_ids_unique() -> None:
    ids = [p.id for p in PROVIDER_REGISTRY]
    assert len(ids) == len(set(ids))


def test_openai_definition_oauth_fields() -> None:
    definition = get_provider_definition("openai")
    assert definition is not None
    assert definition.login is not None
    assert definition.refresh_token is not None
    assert definition.callback_port == 1455
    assert definition.paste_code_flow is False
    assert definition.wire == "openai-compat"
    # The device variant aliases into the same credential row.
    device = get_provider_definition("openai-device")
    assert device is not None
    assert device.store_credentials_as == "openai"


def test_zai_definition() -> None:
    """Z.AI is an API-key provider on the CODING-plan base URL.

    The base URL is asserted because the general `/api/paas/v4` endpoint accepts
    the same key but bills the account balance instead of coding-plan quota — a
    silent wrong-budget bug rather than a visible failure.
    """
    definition = get_provider_definition("zai")
    assert definition is not None
    assert definition.login is not None
    assert definition.env_keys == "ZAI_API_KEY"
    assert definition.wire == "openai-compat"
    assert definition.base_url == "https://api.z.ai/api/coding/paas/v4"
    # Search vocabulary only \u2014 nothing ROUTES on these (that is what the
    # registry's own docstring promises), but the picker must offer the name
    # users came here for, which is the model family rather than the company.
    assert set(definition.search_aliases) == {"glm", "zhipu", "bigmodel", "z-ai"}


def test_anthropic_definition() -> None:
    definition = get_provider_definition("anthropic")
    assert definition is not None
    assert definition.callback_port == 54545
    assert definition.paste_code_flow is True
    assert definition.wire == "anthropic"
    assert definition.base_url == "https://api.anthropic.com"


def test_kimi_definition() -> None:
    definition = get_provider_definition("kimi")
    assert definition is not None
    assert definition.base_url == "https://api.moonshot.cn/v1"
    assert definition.login is not None  # RFC 8628 device code
    assert definition.callback_port is None  # no loopback server


def test_xai_pair() -> None:
    key_provider = get_provider_definition("xai")
    oauth_provider = get_provider_definition("xai-oauth")
    assert key_provider is not None and oauth_provider is not None
    assert oauth_provider.store_credentials_as == "xai"
    assert key_provider.login is not None  # paste-key login
    assert key_provider.env_keys == "XAI_API_KEY"


def test_ollama_allows_missing_api_key() -> None:
    definition = get_provider_definition("ollama")
    assert definition is not None
    assert definition.allows_missing_api_key is True
    assert definition.base_url == "http://localhost:11434/v1"


def test_test_provider_is_mock_wire() -> None:
    definition = get_provider_definition("test")
    assert definition is not None
    assert definition.wire == "mock"


def test_list_login_providers_excludes_keyless_hosts() -> None:
    ids = {p.id for p in list_login_providers()}
    assert {"openai", "anthropic", "kimi", "xai", "xai-oauth"} <= ids
    assert "ollama" not in ids
    assert "test" not in ids


def test_env_key_resolution_plain_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    assert resolve_env_key("deepseek") == "sk-deepseek-test"
    assert env_key_name("deepseek") == "DEEPSEEK_API_KEY"


def test_env_key_resolution_callable_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic's callable resolver prefers the OAuth token var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "raw-key")
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    assert resolve_env_key("anthropic") == "raw-key"
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "oauth-token")
    assert resolve_env_key("anthropic") == "oauth-token"
    # Callable resolvers have no single display name.
    assert env_key_name("anthropic") is None


def test_env_key_resolution_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert resolve_env_key("mistral") is None
    assert resolve_env_key("unknown-provider") is None


def test_alibaba_token_plan_is_separate_from_payg_dashscope() -> None:
    """The Token Plan is a distinct product from DashScope pay-as-you-go:
    its own region-locked endpoint, its own env key, findable by the names a
    user would actually type."""
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("alibaba-token-plan")
    assert definition is not None
    assert (
        definition.base_url
        == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    )
    assert definition.env_keys == "ALIBABA_TOKEN_PLAN_API_KEY"
    # Aliases are picker search vocabulary, not id lookups.
    assert "tokenplan" in definition.search_aliases

    dashscope = get_provider_definition("alibaba")
    assert dashscope is not None and dashscope is not definition


def test_token_plan_oauth_row_spends_the_api_key_on_the_wire() -> None:
    """The OAuth login stores two tokens with different jobs; the wire bearer
    must be the pasted sk-sp key, never the management token."""
    from local_operator.providers.registry import (
        _token_plan_wire_key,
        get_provider_definition,
    )

    assert _token_plan_wire_key({"api_key": "sk-sp-x", "access": "mgmt"}) == "sk-sp-x"
    # Hand-written rows without the embedded key still resolve to something.
    assert _token_plan_wire_key({"access": "mgmt"}) == "mgmt"

    oauth_variant = get_provider_definition("alibaba-token-plan-oauth")
    assert oauth_variant is not None
    assert oauth_variant.store_credentials_as == "alibaba-token-plan"


class TestOAuthHostSplit:
    """Providers serving OAuth and API keys from DIFFERENT hosts.

    Kimi is the case: the coding-plan OAuth grant is only accepted at
    ``api.kimi.com/coding/v1`` -- which is where ``k3`` lives -- while
    ``KIMI_API_KEY`` belongs to the mainland ``api.moonshot.cn`` platform and
    401s there. Verified live against both hosts.
    """

    def test_kimi_declares_the_coding_plan_host_for_oauth(self) -> None:
        from local_operator.providers.registry import get_provider_definition

        kimi = get_provider_definition("kimi")
        assert kimi is not None
        assert kimi.base_url == "https://api.moonshot.cn/v1"
        assert kimi.oauth_base_url == "https://api.kimi.com/coding/v1"

    def test_an_oauth_bearer_is_sent_to_the_oauth_host(self) -> None:
        """Listing the subscription's models is worthless if inference then
        sends them to the API-key host, where they 404."""
        from local_operator.providers.auth_store import OAuthAccess
        from local_operator.providers.clients import OpenAICompatClient

        client = OpenAICompatClient(
            base_url="https://api.moonshot.cn/v1",
            oauth_base_url="https://api.kimi.com/coding/v1",
        )
        oauth = OAuthAccess(access_token="tok", credential_id=1, kind="oauth")
        api_key = OAuthAccess(access_token="sk-x", credential_id=2, kind="api_key")

        assert client._request_base_url(oauth) == "https://api.kimi.com/coding/v1"
        assert client._request_base_url(api_key) == "https://api.moonshot.cn/v1"
        assert client._request_base_url(None) == "https://api.moonshot.cn/v1"

    def test_the_registry_base_on_a_spec_does_not_suppress_the_oauth_host(self) -> None:
        """`build_model_spec` copies `definition.base_url` onto EVERY spec, so a
        naive "did the spec pin a base?" test disables the OAuth host for every
        request -- which is how a live k3 call reached the API-key platform and
        401'd despite all of this being wired up. Only a base the registry did
        NOT supply counts as a deliberate override.
        """
        from local_operator.model.configure import build_model_spec
        from local_operator.providers.clients import OpenAICompatClient, client_for_spec

        def oauth_host_of(spec: Any) -> str | None:
            client = client_for_spec(spec)
            assert isinstance(client, OpenAICompatClient)
            return client._oauth_base_url

        spec = build_model_spec("kimi", "k3")
        assert spec.base_url == "https://api.moonshot.cn/v1"  # the registry's own
        assert oauth_host_of(spec) == "https://api.kimi.com/coding/v1"

        # A genuine override (a gateway) still wins and is never second-guessed.
        assert oauth_host_of(spec.model_copy(update={"base_url": "https://gw.internal/v1"})) is None

    def test_zai_oauth_shares_the_zai_credential_row_and_catalogue(self) -> None:
        """The sign-in mints a durable key for the SAME provider, exactly as
        ``xai-oauth`` does -- not a second catalogue."""
        from local_operator.providers.registry import get_provider_definition

        zai_oauth = get_provider_definition("zai-oauth")
        zai = get_provider_definition("zai")
        assert zai_oauth is not None and zai is not None
        assert zai_oauth.store_credentials_as == "zai"
        assert zai_oauth.base_url == zai.base_url
        assert zai_oauth.login is not None
        # No refresh: the minted key never expires, so there is nothing to refresh.
        assert zai_oauth.refresh_token is None

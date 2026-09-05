"""Tests for the rewritten model/configure.py.

The legacy langchain plumbing (ChatOpenAI/ChatAnthropic/...) is gone:
``configure_model`` now returns a ``ModelConfiguration`` whose ``.spec`` is
the harness ``ModelSpec`` consumed by wire clients, and ``validate_model``
hits the same endpoints as before through a descriptor table.
"""

import asyncio
import json
import time
import zlib
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests
from pydantic import SecretStr

from local_operator.credentials import CredentialManager
from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.model.configure import (
    build_model_spec,
    calculate_cost,
    configure_model,
    create_stream_fn,
    get_model_info_from_openrouter,
    validate_model,
)
from local_operator.model.registry import ModelInfo
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.failover import FallbackTarget, RetrySettings
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport


@pytest.fixture
def mock_credential_manager():
    manager = MagicMock(spec=CredentialManager)
    manager.get_credential = MagicMock(return_value=SecretStr("test_key"))
    manager.prompt_for_credential = MagicMock(return_value=SecretStr("test_key"))
    return manager


@pytest.fixture
def mock_requests_get():
    # Patch target is ``requests.get``, NOT
    # ``local_operator.model.configure.requests.get``: configure.py imports
    # requests inside ``validate_model`` (it is on the per-session import path
    # and requests costs +2.9 MB / +127 modules there), so the module has no
    # ``requests`` attribute to reach through. Patching the real module works
    # for both binding styles — the function-local import resolves to the same
    # already-patched module object from sys.modules. Every ``@patch`` in this
    # file uses the same target for the same reason.
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [{"id": "test-model"}]}
        yield mock_get


# ---------------------------------------------------------------------------
# configure_model
# ---------------------------------------------------------------------------


def test_configure_model_deepseek(mock_credential_manager):
    config = configure_model("deepseek", "deepseek-chat", mock_credential_manager)
    assert config.name == "deepseek-chat"
    assert config.hosting == "deepseek"
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "test_key"
    assert config.info is not None
    assert config.instance is None  # new engine: streaming via stream_fn
    assert config.spec.provider == "deepseek"
    assert config.spec.model_id == "deepseek-chat"
    assert config.spec.base_url == "https://api.deepseek.com/v1"


def test_configure_model_openai(mock_credential_manager):
    config = configure_model("openai", "gpt-4", mock_credential_manager)
    assert config.name == "gpt-4"
    assert config.spec.provider == "openai"
    assert config.spec.base_url == "https://api.openai.com/v1"
    assert config.api_key is not None


def test_configure_model_openai_unknown_model_uses_defaults(mock_credential_manager):
    """Unknown models get sensible ModelSpec defaults, not a KeyError."""
    config = configure_model("openai", "gpt-9-turbo-invented", mock_credential_manager)
    assert config.spec.context_window > 0
    assert config.spec.max_output_tokens > 0


def test_configure_model_ollama(mock_credential_manager):
    config = configure_model("ollama", "llama2", mock_credential_manager)
    assert config.spec.base_url == "http://localhost:11434/v1"
    assert config.spec.provider == "ollama"


def test_configure_model_ollama_missing_model(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("ollama", "", mock_credential_manager)
    assert "Model is required for ollama hosting" in str(exc_info.value)


def test_configure_model_noop_maps_to_test(mock_credential_manager):
    """Legacy 'noop' hosting resolves to the mock wire (unchanged CLI UX)."""
    config = configure_model("noop", "noop", mock_credential_manager)
    assert config.spec.provider == "test"


def test_configure_model_invalid_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("invalid", "model", mock_credential_manager)
    assert "Unsupported hosting platform: invalid" in str(exc_info.value)


def test_configure_model_missing_hosting(mock_credential_manager):
    with pytest.raises(ValueError) as exc_info:
        configure_model("", "model", mock_credential_manager)
    assert "Hosting is required" in str(exc_info.value)


@pytest.mark.parametrize(
    "hosting, default_model",
    [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet-latest"),
        ("deepseek", "deepseek-chat"),
        ("kimi", "moonshot-v1-32k"),
        ("alibaba", "qwen-plus"),
        ("google", "gemini-2.0-flash-001"),
        ("mistral", "mistral-large-latest"),
        ("openrouter", "google/gemini-2.0-flash-001"),
        ("radient", "auto"),
        ("xai", "grok-3"),
    ],
)
def test_configure_model_default_names(hosting: str, default_model: str):
    config = configure_model(hosting, "", None)
    assert config.name == default_model


def test_configure_model_anthropic_spec(mock_credential_manager):
    config = configure_model("anthropic", "claude-3-5-sonnet-latest", mock_credential_manager)
    assert config.spec.provider == "anthropic"
    assert config.spec.base_url == "https://api.anthropic.com"
    # No caller-supplied value, so the per-family sampling policy decides. This
    # model falls through to the OMIT fallback: the app no longer asserts its
    # own 0.2 over a vendor default it never established. DEFAULT_TEMPERATURE
    # remains exported for the explicit-override path.
    assert config.temperature is None


def test_configure_model_kimi_base_url(mock_credential_manager):
    config = configure_model("kimi", "moonshot-v1-8k", mock_credential_manager)
    assert config.spec.base_url == "https://api.moonshot.cn/v1"
    assert config.spec.model_id == "moonshot-v1-8k"


def test_configure_model_alibaba_base_url(mock_credential_manager):
    config = configure_model("alibaba", "qwen-plus", mock_credential_manager)
    assert config.spec.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_configure_model_no_credential_manager_yields_no_key():
    """configure_model never prompts; missing keys resolve at stream time."""
    config = configure_model("deepseek", "deepseek-chat", None)
    assert config.api_key is None
    assert config.spec.model_id == "deepseek-chat"


def test_configure_model_known_model_pulls_registry_info():
    config = configure_model("google", "gemini-2.0-flash-001", None)
    assert config.info.id == "gemini-2.0-flash-001"
    assert config.spec.context_window == config.info.context_window
    assert config.spec.max_output_tokens == config.info.max_tokens


def test_calculate_cost() -> None:
    model_info = ModelInfo(
        id="test-model",
        name="test-model",
        description="Mock model",
        input_price=1,
        output_price=2,
        recommended=True,
    )
    input_tokens = 1000
    output_tokens = 2000
    expected_cost = (input_tokens / 1_000_000) * model_info.input_price + (
        output_tokens / 1_000_000
    ) * model_info.output_price
    assert calculate_cost(model_info, input_tokens, output_tokens) == pytest.approx(expected_cost)
    assert calculate_cost(model_info, 0, 0) == 0.0


@pytest.fixture
def mock_openrouter_client():
    client = MagicMock()
    mock_model_data = [
        MagicMock(
            id="openai/gpt-4o",
            name="GPT-4o",
            description="Mock description before",
            pricing=MagicMock(prompt=1.0 / 1_000_000, completion=2.0 / 1_000_000),
        ),
        MagicMock(
            id="google/gemini-2.0-flash-001",
            name="Gemini 2.0 Flash",
            description="Mock description",
            pricing=MagicMock(prompt=5.0 / 1_000_000, completion=10.0 / 1_000_000),
        ),
        MagicMock(
            id="anthropic/claude-3-5-sonnet-latest",
            name="Claude 3.5 Sonnet",
            description="Mock description after",
            pricing=MagicMock(prompt=15.0 / 1_000_000, completion=20.0 / 1_000_000),
        ),
    ]
    client.list_models.return_value = MagicMock(data=mock_model_data)
    return client


def test_get_model_info_from_openrouter(mock_openrouter_client):
    model_info = get_model_info_from_openrouter(
        mock_openrouter_client, "google/gemini-2.0-flash-001"
    )
    assert model_info.input_price == 5
    assert model_info.output_price == 10


def test_get_model_info_from_openrouter_no_match(mock_openrouter_client):
    with pytest.raises(
        ValueError, match="Model not found from openrouter models API: non-existent-model"
    ):
        get_model_info_from_openrouter(mock_openrouter_client, "non-existent-model")


def test_configure_model_openrouter_via_client(mock_openrouter_client):
    config = configure_model(
        "openrouter", "google/gemini-2.0-flash-001", None, model_info_client=mock_openrouter_client
    )
    assert config.info.input_price == 5
    assert config.spec.model_id == "google/gemini-2.0-flash-001"


def test_create_stream_fn_returns_callable():
    stream_fn = create_stream_fn(MagicMock(), None)
    assert callable(stream_fn)


@pytest.mark.asyncio
async def test_openai_chat_completions_override_routes_gpt5_to_legacy_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    stream = create_stream_fn(
        MagicMock(),
        {"providers": {"openai": {"api": "chat_completions"}}},
        session_id="session-override",
    )
    await stream._http.aclose()
    stream._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    spec = build_model_spec("openai", "gpt-5.4")
    try:
        client = stream._client_for(spec)
        await _collect_stream(
            client.stream(ChatRequest(model=spec, messages=[Message.user("hi")]), "sk-test")
        )
    finally:
        await stream.close()

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_session_stream_supplies_stable_prompt_cache_key_to_public_responses(
    tmp_path,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(
            200,
            content=(
                b'data: {"type":"response.completed","response":{"id":"resp_1",'
                b'"usage":{"input_tokens":1,"output_tokens":1}}}\n\ndata: [DONE]\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(store, {}, session_id="session-cache-key")
    await stream._http.aclose()
    stream._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    spec = build_model_spec("openai", "gpt-5.4")
    try:
        await _collect_stream(stream(ChatRequest(model=spec, messages=[Message.user("hi")]), None))
    finally:
        await stream.close()
        store.close()

    assert captured["url"] == "https://api.openai.com/v1/responses"
    body = json.loads(captured["body"])
    assert body["prompt_cache_key"] == "session-cache-key"
    assert body["prompt_cache_retention"] == "24h"


async def _collect_stream(stream: Any) -> list[Any]:
    return [event async for event in stream]


def _oauth(access: str, account_id: str) -> dict[str, Any]:
    return {
        "access": access,
        "refresh": f"refresh-{access}",
        "expires": 10**15,
        "account_id": account_id,
    }


def _anthropic_usage(used_percent: float) -> UsageReport:
    return UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                amount=UsageAmount(used=used_percent, limit=100.0, unit="percent"),
                window="5h",
                shared=True,
                resets_at_ms=10**15,
            )
        ],
    )


def _anthropic_tier_capped_usage(five_hour_used: float, fable_used: float) -> UsageReport:
    """Shared 5h headroom beside a fully-spent scoped tier cap.

    This is the shape behind the false "credentials temporarily unavailable":
    the Fable weekly is at 100% but the shared 5-hour window still serves,
    and the model being routed (``claude-opus-5``) never draws on Fable."""
    return UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                amount=UsageAmount(used=five_hour_used, limit=100.0, unit="percent"),
                window="5h",
                shared=True,
                resets_at_ms=10**15,
            ),
            UsageLimit(
                id="anthropic:7d:fable",
                label="7 day (Fable)",
                amount=UsageAmount(used=fable_used, limit=100.0, unit="percent"),
                window="7d",
                shared=False,
                tier="fable",
                resets_at_ms=10**15,
            ),
        ],
    )


def _anthropic_tier_only_usage(fable_used: float) -> UsageReport:
    """A scoped tier cap with NO shared window reported beside it.

    The F8 shape: the only observation about the account is that its Fable
    weekly is spent. ``usage_health`` for a fable model therefore returns a
    ``scope="model"`` verdict binding the ``fable`` family alone, and nothing
    in the report speaks to what the account can still do for other models.
    """
    return UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:7d:fable",
                label="7 day (Fable)",
                amount=UsageAmount(used=fable_used, limit=100.0, unit="percent"),
                window="7d",
                shared=False,
                tier="fable",
                resets_at_ms=10**15,
            ),
        ],
    )


def _anthropic_tier_spent_shared_remains(extra_used: float, shared_used: float) -> UsageReport:
    """A spent per-tier cap that leaves the account under an ACCOUNT-scope
    verdict while shared quota still remains.

    ``usage_health`` reads the paid extra-usage window (non-shared, non-tier)
    as an ACCOUNT-scope binding, so a high ``extra_used`` drives the account to
    reserve/depleted; but ``shared_tier_saturation`` re-reads the report and
    still sees the shared 5-hour window with headroom. That is the exact
    combination that reaches the ``tier_binding and shared_above_reserve``
    branch inside ``_apply_account_health`` — the "continuing until shared
    windows are exhausted" notice this test exercises. The spent Fable tier
    row supplies ``tier_binding``."""
    return UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                amount=UsageAmount(used=shared_used, limit=100.0, unit="percent"),
                window="5h",
                shared=True,
                resets_at_ms=10**15,
            ),
            UsageLimit(
                id="anthropic:5h:extra",
                label="5 hour (extra)",
                amount=UsageAmount(used=extra_used, limit=100.0, unit="percent"),
                window="5h",
                shared=False,
                resets_at_ms=10**15,
            ),
            UsageLimit(
                id="anthropic:7d:fable",
                label="7 day (Fable)",
                amount=UsageAmount(used=100.0, limit=100.0, unit="percent"),
                window="7d",
                shared=False,
                tier="fable",
                resets_at_ms=10**15,
            ),
        ],
    )


def _anthropic_model_tier_reserve() -> UsageReport:
    """A MODEL-scope reserve verdict: the tier cap that gates this exact model
    is in reserve while the shared pool is healthy.

    ``usage_health("claude-opus-5")`` keeps only the shared windows plus the
    tier windows whose name appears in the model id, so the Opus weekly (5%
    remaining, in reserve) is the binding limit and — being ``tier`` and not
    ``shared`` — makes ``scope == "model"``. The shared 5-hour window sits at
    10% used (healthy) so it never binds. Paired with
    ``_anthropic_tier_spent_shared_remains`` this yields TWO ``reserve`` verdicts
    on one selector that differ ONLY in scope (``model`` vs ``account``): the
    exact same-state/different-scope case the scoped-token widening exists to
    keep separate. A state-only latch token would collapse both to ``reserve``
    and suppress the second."""
    return UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                amount=UsageAmount(used=10.0, limit=100.0, unit="percent"),
                window="5h",
                shared=True,
                resets_at_ms=10**15,
            ),
            UsageLimit(
                id="anthropic:7d:opus",
                label="7 day (Opus)",
                amount=UsageAmount(used=95.0, limit=100.0, unit="percent"),
                window="7d",
                shared=False,
                tier="opus",
                resets_at_ms=10**15,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_usage_preflight_rotates_accounts_before_providers(tmp_path) -> None:
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_usage(100.0 if access_token == "oauth-a" else 25.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert store.is_blocked(first.id, "anthropic")
        assert not store.is_blocked(second.id, "anthropic")
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == second.id
        assert stream._route_state.active is None
        # Rotating to a sibling account is an internal detail and is now silent:
        # the user still gets served on anthropic, so no notice should fire.
        assert not any("trying another anthropic account" in notice for notice in notices)
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_tier_scoped_cap_with_shared_headroom_keeps_the_account_serving(tmp_path) -> None:
    """A spent tier cap is not an exhausted account when shared quota remains.

    The reported incident: every Anthropic account showed a full Fable weekly
    while the shared 5-hour window still had headroom, and preflight treated
    the tier cap as account exhaustion — rotating, then failing over to z.ai
    while usable Anthropic quota sat idle. A tier-scoped binding window that
    the routed model never draws on must leave the account in service until
    the SHARED windows are the ones that run out."""
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("zai", {"key": "sk-zai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["zai/glm-5.3"]},
            }
        },
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _anthropic_tier_capped_usage(89.0, 100.0),
        ):
            await stream.preflight_usage(model)

        # The account is neither blocked nor deprioritized, and no failover fires.
        assert not store.is_blocked(account.id, "anthropic")
        assert stream._route_state.active is None
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == account.id
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_shared_window_exhaustion_still_rotates_even_with_a_tier_cap(tmp_path) -> None:
    """The tier guard must not mask a genuinely spent shared window.

    When the shared 5-hour window is itself at 100%, the account IS out for
    this model regardless of what the tier caps say, so the sibling rotation
    still happens."""
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    session = _session_hashing_to_first_row(2)
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id=session,
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        # First account: shared window spent. Second: plenty of shared headroom.
        if access_token == "oauth-a":
            return _anthropic_tier_capped_usage(100.0, 100.0)
        return _anthropic_tier_capped_usage(20.0, 100.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert store.is_blocked(first.id, "anthropic")
        assert not store.is_blocked(second.id, "anthropic")
        selected = await store.get_oauth_access("anthropic", session)
        assert selected is not None and selected.credential_id == second.id
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_usage_preflight_exhausts_accounts_then_uses_provider_fallback(tmp_path) -> None:
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {
                    "default": [
                        {
                            "provider": "anthropic",
                            "model": "claude-opus-5",
                            "effort": "low",
                        },
                        {
                            "provider": "openai",
                            "model": "gpt-5.3-codex",
                            "effort": "high",
                        },
                    ]
                },
            }
        },
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _anthropic_usage(100.0),
        ):
            await stream.preflight_usage(model)

        assert store.is_blocked(first.id, "anthropic")
        assert store.is_blocked(second.id, "anthropic")
        assert stream._route_state.active == FallbackTarget("openai/gpt-5.3-codex", "high")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_preflight_fallback_selection_skips_benched_targets(tmp_path) -> None:
    """Quota preflight must not re-pin a fallback the stream driver just
    watched fail.

    The reported annoyance: with the chain's head fallbacks down, every
    message boundary re-selected the FIRST configured fallback, re-pinned it,
    and the stream walk then replayed the whole waterfall — one "provider
    failure" notice and one serial timeout per dead target — before landing
    back on the provider that had been serving. A target benched by
    ``mark_target_failed`` is passed over until its cooldown expires.
    """
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("zai", {"key": "sk-zai", "source": "login"})
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {"default": ["zai/glm-5.3", "openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    try:
        # The stream driver benched zai after it exhausted its provider.
        stream._route_state.mark_target_failed(FallbackTarget("zai/glm-5.3"), cooldown_ms=300_000)

        async def usage_for_access(_client, provider, **_kwargs):
            # Anthropic is the exhausted primary; fallbacks must look usable
            # so the quota-aware skip does not hide the bench preference.
            return _anthropic_usage(100.0 if provider == "anthropic" else 20.0)

        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        # Preflight pinned the SECOND fallback, not the benched head.
        assert stream._route_state.active == FallbackTarget("openai/gpt-5.3-codex")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_preflight_fallback_uses_a_benched_target_when_nothing_else_remains(
    tmp_path,
) -> None:
    """An all-benched chain is still a chain — never "no configured fallback".

    The bench is a preference between working candidates. When every authed
    candidate is benched, the first is returned anyway: reporting a dead end
    to a user who configured several fallbacks would turn a routing hint into
    an outage, and the stream walk owns discovering which bench has expired.
    """
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("zai", {"key": "sk-zai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {"default": ["zai/glm-5.3"]},
            }
        },
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    try:
        stream._route_state.mark_target_failed(FallbackTarget("zai/glm-5.3"), cooldown_ms=300_000)
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda _client, provider, **_kwargs: _anthropic_usage(
                100.0 if provider == "anthropic" else 20.0
            ),
        ):
            await stream.preflight_usage(model)

        assert stream._route_state.active == FallbackTarget("zai/glm-5.3")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_usage_reserve_can_reduce_effort_without_blocking_account(tmp_path) -> None:
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {
                    "default": [
                        {
                            "provider": "anthropic",
                            "model": "claude-opus-5",
                            "effort": "low",
                        }
                    ]
                },
            }
        },
        session_id="session-a",
    )
    model = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning=True,
        reasoning_effort="high",
    )

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _anthropic_usage(95.0),
        ):
            await stream.preflight_usage(model)

        assert not store.is_blocked(account.id, "anthropic")
        assert stream._route_state.active == FallbackTarget("anthropic/claude-opus-5", "low")
    finally:
        await stream.close()
        store.close()


def _session_hashing_to_first_row(row_count: int) -> str:
    """A session id whose crc32 lands the selection order on row index 0.

    ``AuthStore._base_selection_order`` rotates the pool by
    ``crc32(session_id) % len(rows)`` when no sticky credential is set, so a
    fixed literal makes "which account does the walk meet first" an accident
    of the string. These tests need the FIRST-INSERTED account visited first
    for their scenario to exist at all, so the id is derived, not guessed.
    """
    candidates = (f"session-{i}" for i in range(64))
    return next(s for s in candidates if zlib.crc32(s.encode()) % row_count == 0)


@pytest.mark.asyncio
async def test_usage_preflight_demotes_rather_than_blocks_a_reserve_account(tmp_path) -> None:
    """An account in reserve still has quota, so it must never be written into
    the cross-process block table.

    The block preflight used to write lasted until the BINDING window's reset
    — for a seven-day window at 90 % that is a multi-day outage recorded in
    SQLite against an account that could still serve. Reserve is a routing
    preference, so it gets the in-process, self-expiring demotion instead."""
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    session = _session_hashing_to_first_row(2)
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id=session,
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_usage(95.0 if access_token == "oauth-a" else 25.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert not store.is_blocked(first.id, "anthropic")
        assert not store.is_blocked(second.id, "anthropic")
        # The preference still holds: the healthy sibling serves next.
        selected = await store.get_oauth_access("anthropic", session)
        assert selected is not None and selected.credential_id == second.id
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_reserve_on_the_sticky_account_keeps_the_session_there(tmp_path) -> None:
    """A reserve verdict is a preference for NEW picks, never an eviction.

    The measured defect: a session sticky to account A (its whole conversation
    cached there) hit a boundary where A read 8% remaining and B was healthy.
    The preflight demoted A process-wide, the cascade dropped A from the tier,
    the session was re-pinned to B and its 150-500k-token prefix was rewritten
    on B at cache-write price — then, 120s later with B also low, back again.
    Now the session stays on A with one info line, and only a FRESH session
    prefers B."""
    store = AuthStore(tmp_path / "auth.db")
    warm = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    cold = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
    # The session has been transacting on A: that is where its cache is warm.
    store.pin_session_credential("anthropic", "session-a", warm.id)

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_usage(92.0 if access_token == "oauth-a" else 40.0)

    async def boundary() -> None:
        stream.begin_message()
        stream._usage_checked_at = 0.0

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await boundary()
            await stream.preflight_usage(model)

            selected = await store.get_oauth_access("anthropic", "session-a")
            assert selected is not None and selected.credential_id == warm.id
            assert not store.is_blocked(warm.id, "anthropic")
            # Not demoted either: the mark is for a fresh pick the walk is
            # moving off, and this walk moved off nothing.
            assert store._active_demotions("anthropic") == set()
            assert stream._route_state.active is None
            assert notices == [
                "info:anthropic quota low (8% remaining) — staying on this account to keep "
                "the prompt cache warm"
            ]

            # Steady state is silent, and the session still does not move.
            await boundary()
            await stream.preflight_usage(model)
            selected = await store.get_oauth_access("anthropic", "session-a")
            assert selected is not None and selected.credential_id == warm.id
            assert len(notices) == 1

        # A session with nothing cached anywhere still prefers the healthy
        # account: that is what the reserve threshold is FOR.
        fresh = create_stream_fn(
            store,
            {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
            session_id=_session_hashing_to_first_row(2),
        )
        try:
            with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
                await fresh.preflight_usage(model)
            picked = await store.get_oauth_access("anthropic", fresh._session_id)
            assert picked is not None and picked.credential_id == cold.id
        finally:
            await fresh.close()
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_depleted_on_the_sticky_account_still_moves_the_session(tmp_path) -> None:
    """The one verdict that may move a warm session: at 0% the rewrite is
    unavoidable, so the block is written and the session re-pins to a sibling."""
    store = AuthStore(tmp_path / "auth.db")
    warm = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    cold = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
    store.pin_session_credential("anthropic", "session-a", warm.id)

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_usage(100.0 if access_token == "oauth-a" else 40.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert store.is_blocked(warm.id, "anthropic")
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == cold.id
        assert store.session_credential_id("anthropic", "session-a") == cold.id
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_reserve_on_a_fresh_pick_moves_on_and_the_walk_terminates(tmp_path) -> None:
    """A row the walk only just landed on holds nothing cached, so reserve on it
    still steers the walk to a sibling — and the walk must END there.

    The trap: the store keeps a demoted STICKY row in service, and the walk's
    own resolve pinned the session to this fresh pick. Demoting without
    releasing the pin would re-resolve the same row, find it in
    ``attempted_ids``, and return with the session parked on the account it
    meant to leave. Three accounts, the first two in reserve, so the walk has
    to move twice; every probe is counted so a re-probe of the same account
    would show up as a fourth fetch."""
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    third = store.upsert_credential("anthropic", _oauth("oauth-c", "account-c"))
    session = _session_hashing_to_first_row(3)
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id=session,
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
    probed: list[str] = []

    async def usage_for_access(_client, _provider, *, access_token="", **_kwargs):
        probed.append(access_token)
        return _anthropic_usage({"oauth-a": 95.0, "oauth-b": 93.0}.get(access_token, 20.0))

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert probed == ["oauth-a", "oauth-b", "oauth-c"]
        selected = await store.get_oauth_access("anthropic", session)
        assert selected is not None and selected.credential_id == third.id
        assert store._active_demotions("anthropic") == {first.id, second.id}
        assert not store.is_blocked(first.id, "anthropic")
        assert not store.is_blocked(second.id, "anthropic")
        # Silent: fresh-pick rotation is the internal detail it always was.
        assert notices == []
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_reserve_on_the_sticky_account_is_silent_after_the_walk_settles(
    tmp_path,
) -> None:
    """The stay decision is taken against the sticky captured BEFORE the walk,
    so a session whose warm account is reached only after the walk moves off a
    fresh reserve pick is still recognised as warm there — and not demoted.

    Shape: sticky on C (warm). The walk starts on C (sticky first), finds it
    in reserve, and must settle immediately, never touching A or B."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    warm = store.upsert_credential("anthropic", _oauth("oauth-c", "account-c"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
    store.pin_session_credential("anthropic", "session-a", warm.id)
    probed: list[str] = []

    async def usage_for_access(_client, _provider, *, access_token="", **_kwargs):
        probed.append(access_token)
        return _anthropic_usage(94.0 if access_token == "oauth-c" else 20.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert probed == ["oauth-c"]
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == warm.id
        assert store._active_demotions("anthropic") == set()
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_concurrent_walks_on_one_stream_fn_each_keep_their_own_boundary_sticky(
    tmp_path,
) -> None:
    """Review F2: the boundary sticky is walk-local, not stream-fn state.

    A child session runs on its PARENT's stream fn (``harness/subagent.py``),
    so two boundary walks can interleave on one instance. Shape: no sticky
    yet; the parent's walk hash-picks A (reserve) and its resolve pins the
    session to A, then suspends in the usage fetch. The child's walk enters
    meanwhile, reads the store sticky — now A, the parent's fresh pick — and
    judges A warm. Held on the instance, that reading leaked into the parent's
    walk when it resumed: A compared equal to "its" boundary sticky, so the
    parent settled on a reserve account it had every reason to leave. Kept
    per walk, the parent still sees ``None`` (nothing was warm when it began),
    demotes the fresh pick and moves to B."""
    store = AuthStore(tmp_path / "auth.db")
    low = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    healthy = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    session = _session_hashing_to_first_row(2)
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id=session,
    )
    parent_model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
    child_model = ModelSpec(provider="anthropic", model_id="claude-sonnet-5")
    parent_suspended = asyncio.Event()
    child_done = asyncio.Event()
    probed: list[str] = []

    async def usage_for_access(_client, _provider, *, access_token="", **_kwargs):
        probed.append(access_token)
        if len(probed) == 1:
            # The parent's first probe: park it until the child's whole walk
            # has run, so the child's reading of the sticky is the one the
            # parent would have inherited from a shared attribute.
            parent_suspended.set()
            await child_done.wait()
        return _anthropic_usage(95.0 if access_token == "oauth-a" else 20.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            parent = asyncio.create_task(stream.preflight_usage(parent_model))
            await parent_suspended.wait()
            # The parent's resolve has pinned the session to its fresh pick.
            assert store.session_credential_id("anthropic", session) == low.id

            # The child opens its own boundary on the shared stream fn.
            stream.begin_message()
            await stream.preflight_usage(child_model)
            child_done.set()
            await parent

        # The parent judged A against ITS boundary reading (None): a fresh
        # pick in reserve, so it was demoted and the walk moved on to B.
        selected = await store.get_oauth_access("anthropic", session)
        assert selected is not None and selected.credential_id == healthy.id
        assert store._active_demotions("anthropic") == {low.id}
        assert not store.is_blocked(low.id, "anthropic")
        assert probed == ["oauth-a", "oauth-a", "oauth-b"]
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_a_warm_stay_keeps_the_current_effort(tmp_path) -> None:
    """Review F4: staying on the warm account must not take the same-provider
    lower-effort hop the chain offers. Anthropic drops the cached message
    prefix when the thinking parameters change, so the hop would rewrite the
    very conversation the stay exists to keep warm — the last-account branch
    still takes it (``test_usage_reserve_can_reduce_effort_without_blocking_account``),
    because there nothing else can serve and the per-request saving wins."""
    store = AuthStore(tmp_path / "auth.db")
    warm = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {
                    "default": [
                        {"provider": "anthropic", "model": "claude-opus-5", "effort": "low"}
                    ]
                },
            }
        },
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
    model = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning=True,
        reasoning_effort="high",
    )
    store.pin_session_credential("anthropic", "session-a", warm.id)

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_usage(95.0 if access_token == "oauth-a" else 20.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == warm.id
        # No effort route activated: the request goes out at the effort the
        # conversation is cached under.
        assert stream._route_state.active is None
        assert notices == [
            "info:anthropic quota low (5% remaining) — staying on this account to keep "
            "the prompt cache warm"
        ]
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_model_tier_reserve_on_the_sticky_account_keeps_the_session_there(
    tmp_path,
) -> None:
    """The model-tier branch (a scoped weekly cap in reserve) keeps the same
    promise as the account-scope one: warm account stays, fresh pick moves."""
    store = AuthStore(tmp_path / "auth.db")
    warm = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    cold = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
    model = ModelSpec(provider="anthropic", model_id="claude-fable-5")
    store.pin_session_credential("anthropic", "session-a", warm.id)

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_fable_usage(95.0 if access_token == "oauth-a" else 20.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == warm.id
        assert store._active_demotions("anthropic") == set()
        assert not store.is_blocked(warm.id, "anthropic")
        assert len(notices) == 1 and "staying on this account" in notices[0]
        assert "for claude-fable-5" in notices[0]

        # And a session that is NOT warm on A moves to B (silently).
        fresh = create_stream_fn(
            store,
            {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
            session_id=_session_hashing_to_first_row(2),
        )
        try:
            with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
                await fresh.preflight_usage(model)
            picked = await store.get_oauth_access("anthropic", fresh._session_id)
            assert picked is not None and picked.credential_id == cold.id
        finally:
            await fresh.close()
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_reserve_account_still_serves_after_the_healthy_sibling_depletes(tmp_path) -> None:
    """The incident this change fixes, replayed end to end.

    One account sits in reserve (95 % of a window used), the other is healthy.
    Preflight steers to the healthy account; later that account genuinely
    exhausts and the 429 path blocks it for its advertised reset. The reserve
    account is now the only thing left — and it MUST serve. When preflight
    recorded reserve as a SQLite block, this exact sequence left the provider
    with four configured credentials and zero usable ones, and every live
    session died with "all credentials not usable" while quota remained."""
    from local_operator.providers.failover import ProviderError

    store = AuthStore(tmp_path / "auth.db")
    reserve = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    healthy = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    session = _session_hashing_to_first_row(2)
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id=session,
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_usage(95.0 if access_token == "oauth-a" else 25.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)
        # The healthy account served and is sticky; now it runs out for real.
        store.rotate_sibling(
            "anthropic",
            session,
            ProviderError(429, "rate limited", retryable=True, retry_after_ms=3 * 3_600_000),
        )
        assert store.is_blocked(healthy.id, "anthropic")

        survivor = await store.get_oauth_access("anthropic", session)
        assert survivor is not None and survivor.credential_id == reserve.id
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_reserve_account_is_not_blocked_when_falling_back_cross_provider(tmp_path) -> None:
    """The no-siblings branch takes the same depleted/reserve split.

    A lone reserve account with a cross-provider fallback used to be blocked
    until its window reset before the session moved to the fallback. The block
    is cross-process, so it also stranded every OTHER session — including ones
    whose fallback chains could not rescue them.

    Crossing the reserve threshold is also no longer a reason to LEAVE the
    provider: the last account still holding spendable quota stays in
    service until it is genuinely at 0%. The fallback is configured so the
    old hop-at-10% behaviour would have pinned openai; staying on anthropic
    is the contract this test now pins."""
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _anthropic_usage(95.0),
        ):
            await stream.preflight_usage(model)

        assert not store.is_blocked(account.id, "anthropic")
        assert stream._route_state.active is None
        # Another session (or process) with no fallback still reaches the account.
        access = await store.get_oauth_access("anthropic", "some-other-session")
        assert access is not None and access.credential_id == account.id
    finally:
        await stream.close()
        store.close()


def _anthropic_fable_usage(fable_used: float, five_hour_used: float = 0.0) -> UsageReport:
    """The shape of a Fable request: the scoped weekly is the binding window.

    Shared 5h/7d still apply (they always do), but a Fable model is also gated
    by ``7 day (Fable)``. That is ``scope="model"`` when the Fable window is
    the tightest — the path that used to hop providers without rotating
    siblings."""
    return UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                amount=UsageAmount(used=five_hour_used, limit=100.0, unit="percent"),
                window="5h",
                shared=True,
                resets_at_ms=10**15,
            ),
            UsageLimit(
                id="anthropic:7d:fable",
                label="7 day (Fable)",
                amount=UsageAmount(used=fable_used, limit=100.0, unit="percent"),
                window="7d",
                shared=False,
                tier="fable",
                resets_at_ms=10**15,
            ),
        ],
    )


def _kimi_usage(used_percent: float) -> UsageReport:
    return UsageReport(
        provider="kimi",
        limits=[
            UsageLimit(
                id="kimi:total",
                label="Total quota",
                amount=UsageAmount(used=used_percent, limit=100.0, unit="percent"),
                shared=True,
                resets_at_ms=10**15,
            )
        ],
    )


@pytest.mark.asyncio
async def test_model_tier_cap_rotates_siblings_before_leaving_the_provider(tmp_path) -> None:
    """A spent Fable weekly is per account, not per provider.

    The reported cascade: primary on ``claude-fable-5``, first Anthropic
    login's Fable window at 100% while siblings still had Fable headroom
    (and every login still had 5h/7d), yet preflight hopped to Kimi because
    ``scope="model"`` skipped sibling rotation. The first account is taken
    out of rotation; the sibling with remaining Fable quota serves; no
    provider failover fires."""
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    store.upsert_credential("kimi", _oauth("oauth-kimi", "kimi-a"))
    session = _session_hashing_to_first_row(2)
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["kimi/k3"]},
            }
        },
        session_id=session,
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))
    model = ModelSpec(provider="anthropic", model_id="claude-fable-5")

    async def usage_for_access(_client, provider, *, access_token=None, **_kwargs):
        if provider == "kimi":
            return _kimi_usage(90.0)
        return _anthropic_fable_usage(100.0 if access_token == "oauth-a" else 16.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        # The spent Fable weekly takes the first account out of rotation FOR
        # FABLE — a family-scoped block, not an account-wide one: the shared
        # 5h/7d windows still serve other families, so an opus resolve must
        # still see the row (the defect this guards: "all credentials
        # unusable" on a pool whose only spent window was Fable's).
        assert store.is_blocked_for_model(first.id, "anthropic", "claude-fable-5")
        assert not store.is_blocked(first.id, "anthropic")
        assert not store.is_blocked_for_model(first.id, "anthropic", "claude-opus-4-8")
        assert not store.is_blocked(second.id, "anthropic")
        selected = await store.get_oauth_access("anthropic", session)
        assert selected is not None and selected.credential_id == second.id
        opus = await store.get_oauth_access("anthropic", session, model_id="claude-opus-4-8")
        assert opus is not None  # the family-blocked account still serves opus
        assert stream._route_state.active is None
        # Sibling rotation is silent now — an internal detail, not a transcript event.
        assert not any("trying another anthropic account" in notice for notice in notices)
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_last_account_in_reserve_stays_on_the_provider(tmp_path) -> None:
    """Reserve is a preference between siblings, not a hop to the next provider.

    One Anthropic account at 5% remaining, chain next is Kimi at 10% remaining
    then a maxed Qwen then Grok. The old policy pinned Kimi the moment the
    last Anthropic account crossed the 10% threshold, then pinned past Kimi
    for the same reason. Spendable quota on the current provider must be
    emptied before the cascade moves."""
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("kimi", _oauth("oauth-kimi", "kimi-a"))
    store.upsert_credential("xai", _oauth("oauth-xai", "xai-a"))
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {
                    "default": ["kimi/k3", "alibaba-token-plan/qwen3.8-max", "xai/grok-4.6"]
                },
            }
        },
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
    model = ModelSpec(provider="anthropic", model_id="claude-fable-5")

    async def usage_for_access(_client, provider, *, access_token=None, **_kwargs):
        if provider == "kimi":
            return _kimi_usage(90.0)
        if provider == "xai":
            return _kimi_usage(0.0)
        return _anthropic_fable_usage(95.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert not store.is_blocked(account.id, "anthropic")
        assert stream._route_state.active is None
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == account.id
        assert any("continuing until anthropic quota is exhausted" in n for n in notices)
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_quota_notice_is_deduped_to_one_per_state_transition(tmp_path) -> None:
    """The user hears about a quota CHANGE, not the same line every message.

    Preflight runs on every user-message boundary, so a persistent low/exhausted
    verdict recurs on every message the user sends. Before the dedup latch that
    echoed the identical "quota low"/"quota exhausted" line forever. The
    contract now: announce once per state TRANSITION per provider/model —
    silent while a state holds, a fresh line when it changes, and a fresh line
    again after a recovery resets the latch. This is the real-execution evidence
    for that behaviour, driving successive boundaries through the account-scope
    ``_apply_account_health`` path (single account, no configured fallback)."""
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    # A knob the mocked usage endpoint reads, so each boundary can present a
    # different quota shape without re-patching.
    used_percent = {"value": 95.0}

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return _anthropic_usage(used_percent["value"])

    async def boundary() -> None:
        # Simulate a fresh user message far enough apart that the 60s TTL memo
        # (which only dedupes the several requests ONE message makes) does not
        # itself swallow the check — resetting the memo clock is what a real
        # inter-message gap does.
        stream.begin_message()
        stream._usage_checked_at = 0.0

    reserve_line = "continuing until anthropic quota is exhausted"
    depleted_line = "no configured fallback is available"
    recovered_line = "account quota recovered"

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            # 1) First slide to low (reserve, 5% remaining): one notice.
            used_percent["value"] = 95.0
            await boundary()
            await stream.preflight_usage(model)
            assert sum(reserve_line in n for n in notices) == 1

            # 2) Still low on the next boundary: silent — no duplicate line.
            await boundary()
            await stream.preflight_usage(model)
            assert sum(reserve_line in n for n in notices) == 1

            # 3) low -> exhausted is a real transition: a second, different line.
            used_percent["value"] = 100.0
            await boundary()
            await stream.preflight_usage(model)
            assert sum(depleted_line in n for n in notices) == 1
            assert store.is_blocked(account.id, "anthropic")

            # 4) Recovery (healthy) clears the latch via the blocked-account
            #    re-probe, so the exhausted verdict is no longer "already said".
            used_percent["value"] = 0.0
            await boundary()
            await stream.preflight_usage(model)
            assert sum(recovered_line in n for n in notices) == 1
            assert not store.is_blocked(account.id, "anthropic")

            # 5) Sliding back to low AFTER recovery is a genuine new transition:
            #    it re-announces rather than being deduped against step 1.
            used_percent["value"] = 95.0
            await boundary()
            await stream.preflight_usage(model)
            assert sum(reserve_line in n for n in notices) == 2
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_tier_spent_shared_remains_notice_is_deduped_across_boundaries(tmp_path) -> None:
    """The "continuing until shared windows are exhausted" line is announced
    once, then silent while the condition holds.

    ``_apply_account_health`` reaches this branch when a per-tier cap is spent
    (Fable at 100%) but the shared window still has headroom. Preflight runs on
    every message boundary, so before the dedup this notice recurred on every
    message — the per-boundary spam this PR eliminates. Two same-provider
    accounts skip the ``not siblings`` guards so the tier-spent branch is the
    one that fires. Real execution: repeated boundaries, one notice."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    async def boundary() -> None:
        stream.begin_message()
        stream._usage_checked_at = 0.0

    tier_spent_line = "continuing until shared windows are exhausted"
    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_a, **_k: _anthropic_tier_spent_shared_remains(95.0, 50.0),
        ):
            for _ in range(4):
                await boundary()
                await stream.preflight_usage(model)
        # One line on the first boundary, silent on the three that follow.
        assert sum(tier_spent_line in n for n in notices) == 1
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_distinct_quota_conditions_on_one_selector_do_not_alias(tmp_path) -> None:
    """Two quota conditions that share a STATE but differ in SCOPE each get their
    own single announcement rather than masking each other.

    This is the guard for the scoped-token widening specifically, so the two
    conditions are chosen to be indistinguishable to the PRE-fix state-only
    latch: BOTH read ``health.state == "reserve"``, and differ ONLY in scope.

    - Phase 1 (``tier-spent``): a spent extra-usage window drives an
      ACCOUNT-scope reserve verdict while the shared 5-hour window still has
      headroom → the ``tier-spent:reserve`` "continuing until shared windows are
      exhausted" line.
    - Phase 2 (``model-reserve``): the Opus weekly tier cap that gates this exact
      model is in reserve while the shared pool is healthy → a MODEL-scope reserve
      verdict and the ``model:reserve`` "for <model> — staying on this account to
      keep the prompt cache warm" line (the session is already sticky to this
      account from phase 1, so a reserve verdict keeps it there).

    Under the fix both are announced once (tokens ``tier-spent:reserve`` and
    ``model:reserve`` are distinct). Under the bug — a token that encoded only
    ``health.state`` — both would collapse to a single ``"reserve"`` entry and the
    SECOND would be wrongly suppressed. A regression guard whose two conditions
    differed in state (as an earlier version of this test did) stays green even
    against that bug, so it proved nothing; keeping the state fixed is the whole
    point. See ``test_state_only_quota_token_would_alias_distinct_scopes`` for the
    companion that pins the bug direction by forcing a state-only token.

    Two same-provider accounts are configured so a sibling exists (the shape
    under which phase 2 takes the model-tier stay-on-sticky branch rather than
    the lone-account one), and so the account-scope tier-spent branch is
    reachable."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    # A knob the mocked endpoint reads so each boundary presents a different
    # quota shape. Both shapes read ``reserve`` — only the SCOPE differs.
    phase = {"value": "tier-spent"}

    def report_for_phase(*_a, **_k):
        if phase["value"] == "tier-spent":
            # Extra window spent (account-scope reserve) while the shared 5-hour
            # window still holds headroom → the account-scope tier-spent branch.
            return _anthropic_tier_spent_shared_remains(95.0, 50.0)
        # Opus weekly cap in reserve, shared pool healthy → model-scope reserve.
        return _anthropic_model_tier_reserve()

    async def boundary() -> None:
        stream.begin_message()
        stream._usage_checked_at = 0.0

    tier_spent_line = "continuing until shared windows are exhausted"
    model_reserve_line = "for claude-opus-5 — staying on this account to keep the prompt cache warm"
    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=report_for_phase):
            # 1) The account-scope (tier-spent) reserve condition announces once
            #    and stays quiet across a second identical boundary.
            phase["value"] = "tier-spent"
            await boundary()
            await stream.preflight_usage(model)
            await boundary()
            await stream.preflight_usage(model)
            assert sum(tier_spent_line in n for n in notices) == 1

            # 2) A DIFFERENT-SCOPE reserve condition (model-tier) on the SAME
            #    selector and the SAME state is announced too — the scoped tokens
            #    keep the two conditions from aliasing where a state-only token
            #    (``reserve``) would have suppressed this second line.
            phase["value"] = "model-reserve"
            await boundary()
            await stream.preflight_usage(model)
            assert sum(model_reserve_line in n for n in notices) == 1
            # The earlier condition's single announcement is untouched.
            assert sum(tier_spent_line in n for n in notices) == 1
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_state_only_quota_token_would_alias_distinct_scopes(tmp_path) -> None:
    """Pins the DIRECTION of the aliasing bug the scoped tokens fix.

    Same two same-state/different-scope reserve conditions as
    ``test_distinct_quota_conditions_on_one_selector_do_not_alias``, but the
    latch's per-condition ``token`` is monkeypatched back to a bare
    ``health.state`` — the PRE-fix behaviour. With that collapse the second
    condition (model-scope reserve) shares the first's ``"reserve"`` entry and is
    wrongly suppressed, so only ONE line is heard. This is the failure the
    strengthened guard above must be able to catch: if the production code ever
    regresses to a state-only token, the guard flips red instead of staying
    green. Together the two tests bracket the bug — one proves the fix announces
    both, this one proves the bug would silence the second."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    # Re-derive the state-only token from the announcement text so the shim needs
    # no access to ``health`` here: both conditions in this scenario are reserve,
    # so a state-only latch keys every one of them on the single token "reserve".
    original = stream._announce_quota_change

    async def state_only_token(selector, token, text, kind="warning"):
        # The bug: collapse the scoped token (``tier-spent:reserve`` /
        # ``model:reserve``) back to the state alone, so distinct conditions on
        # one selector alias onto a single latch entry.
        return await original(selector, "reserve", text, kind)

    stream._announce_quota_change = state_only_token  # type: ignore[method-assign]

    phase = {"value": "tier-spent"}

    def report_for_phase(*_a, **_k):
        if phase["value"] == "tier-spent":
            return _anthropic_tier_spent_shared_remains(95.0, 50.0)
        return _anthropic_model_tier_reserve()

    async def boundary() -> None:
        stream.begin_message()
        stream._usage_checked_at = 0.0

    tier_spent_line = "continuing until shared windows are exhausted"
    model_reserve_line = "for claude-opus-5 — staying on this account to keep the prompt cache warm"
    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=report_for_phase):
            phase["value"] = "tier-spent"
            await boundary()
            await stream.preflight_usage(model)
            assert sum(tier_spent_line in n for n in notices) == 1

            phase["value"] = "model-reserve"
            await boundary()
            await stream.preflight_usage(model)
            # The bug in action: same "reserve" token as phase 1, so the
            # model-scope line is swallowed. This assertion documents the wrong
            # behaviour the scoped tokens prevent.
            assert sum(model_reserve_line in n for n in notices) == 0
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_preflight_skips_a_depleted_fallback_provider(tmp_path) -> None:
    """A maxed fallback is not a place to land.

    Anthropic genuinely exhausted, Kimi at 10% remaining, Qwen at 100%. The
    cascade must pin Kimi (spendable reserve) rather than Qwen (already
    empty) or hopping past Kimi because 10% looks like a reason to leave."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("kimi", _oauth("oauth-kimi", "kimi-a"))
    store.upsert_credential("xai", _oauth("oauth-xai", "xai-a"))
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["kimi/k3", "xai/grok-4.6"]},
            }
        },
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    async def usage_for_access(_client, provider, **_kwargs):
        if provider == "anthropic":
            return _anthropic_usage(100.0)
        if provider == "kimi":
            return _kimi_usage(90.0)
        return _kimi_usage(100.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert stream._route_state.active == FallbackTarget("kimi/k3")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_preflight_skips_a_fully_spent_fallback_to_the_next_usable(tmp_path) -> None:
    """When the chain's head fallback is at 0%, land on the next that isn't."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("kimi", _oauth("oauth-kimi", "kimi-a"))
    store.upsert_credential("xai", _oauth("oauth-xai", "xai-a"))
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["kimi/k3", "xai/grok-4.6"]},
            }
        },
        session_id="session-a",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    async def usage_for_access(_client, provider, **_kwargs):
        if provider == "anthropic":
            return _anthropic_usage(100.0)
        if provider == "kimi":
            return _kimi_usage(100.0)
        return _kimi_usage(20.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert stream._route_state.active == FallbackTarget("xai/grok-4.6")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_one_boundary_walk_probes_each_fallback_provider_once(tmp_path) -> None:
    """Review F7: the fallback quota memo belongs to the WALK, not the call.

    Three exhausted Anthropic accounts make preflight rotate three times, and
    each rotation consults the fallback chain. With the memo scoped per call,
    every rotation re-hit Kimi's and Z.AI's usage endpoints — six network
    probes to answer a question whose answer cannot change inside one
    boundary, against endpoints that rate-limit per source IP. One probe per
    provider+model now serves the whole walk, and the route it picks is
    unchanged.
    """
    store = AuthStore(tmp_path / "auth.db")
    for suffix in ("a", "b", "c"):
        store.upsert_credential("anthropic", _oauth(f"oauth-{suffix}", f"account-{suffix}"))
    store.upsert_credential("kimi", {"key": "sk-kimi", "source": "login"})
    store.upsert_credential("zai", {"key": "sk-zai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["kimi/k3", "zai/glm-5.3"]},
            }
        },
        session_id=_session_hashing_to_first_row(3),
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
    probed: list[str] = []

    async def usage_for_access(_client, provider, **_kwargs):
        probed.append(provider)
        if provider == "anthropic":
            return _anthropic_usage(100.0)  # every account spent: rotate
        if provider == "kimi":
            return _kimi_usage(100.0)  # depleted: the walk continues past it
        return _kimi_usage(20.0)  # zai has headroom and takes the route

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(model)

        assert probed.count("kimi") == 1
        assert probed.count("zai") == 1
        # The saving must not have changed where the session lands.
        assert stream._route_state.active == FallbackTarget("zai/glm-5.3")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_a_tier_depletion_leaves_other_models_on_the_account(tmp_path) -> None:
    """Review F8: a spent model-tier window must not strand other models.

    Three accounts whose ONLY reported window is a ``7 day (Fable)`` cap at
    100%. Routing ``claude-fable-5`` is therefore a pure model-tier
    depletion: nothing observed says these accounts cannot serve
    ``claude-opus-5``. The verdict is recorded as a ``model:fable``-scoped
    block (see ``_write_quota_block``), so Fable stops and opus keeps every
    account. An account-wide block here is the regression this pins — it
    would take the pool out of service for every model until the Fable
    weekly reset, days away.
    """
    store = AuthStore(tmp_path / "auth.db")
    rows = [
        store.upsert_credential("anthropic", _oauth(f"oauth-{s}", f"account-{s}"))
        for s in ("a", "b", "c")
    ]
    store.upsert_credential("kimi", {"key": "sk-kimi", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["kimi/k3"]},
            }
        },
        session_id=_session_hashing_to_first_row(3),
    )

    async def usage_for_access(_client, provider, **_kwargs):
        if provider == "anthropic":
            return _anthropic_tier_only_usage(100.0)
        return _kimi_usage(20.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-fable-5"))

        for row in rows:
            # Never account-wide: that is what strands the other models.
            assert not store.is_blocked(row.id, "anthropic")
            assert not store.is_blocked_for_model(row.id, "anthropic", "claude-opus-5")
        # Opus still resolves on this provider rather than falling to Kimi.
        opus = await store.get_oauth_access("anthropic", "session-opus", model_id="claude-opus-5")
        assert opus is not None
        # The Fable cap itself is still honoured on the rotated accounts.
        assert any(
            store.is_blocked_for_model(row.id, "anthropic", "claude-fable-5") for row in rows
        )
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_fallback_quota_is_unknown_when_an_oauth_account_is_omitted(tmp_path) -> None:
    """One unread OAuth sibling prevents a provider-wide depleted verdict.

    ``list_oauth_accesses`` omits refresh failures. If the one access it does
    return is at 0%, the missing row is still UNKNOWN — not proof that every
    account is depleted (agent review F1)."""
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(store, {"retry": {"usageAwareFallback": True}})

    access = await store.get_oauth_access("anthropic", "session-a")
    assert access is not None
    survivor = first if access.credential_id == first.id else second

    try:
        with (
            patch.object(store, "list_oauth_accesses", return_value=[access]),
            patch(
                "local_operator.providers.usage.fetch_usage",
                side_effect=lambda *_args, **_kwargs: _anthropic_usage(100.0),
            ),
        ):
            verdict = await stream._provider_quota_availability(
                "anthropic", "claude-opus-5", reserve_percent=10, cache={}
            )

        assert survivor.id in {first.id, second.id}
        assert verdict == "unknown"
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_reserve_only_pool_settles_back_from_a_pinned_fallback(tmp_path) -> None:
    """After every sibling is checked, reserve quota wins over the old pin.

    Both Anthropic accounts have 5% Fable quota. The previous fallback must
    clear after the second probe; attempted accounts are not counted as fresh
    siblings that send the walk around once more (agent review F2)."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True, "usageReservePercent": 10}},
        session_id=_session_hashing_to_first_row(2),
    )
    model = ModelSpec(provider="anthropic", model_id="claude-fable-5")
    stream._primary_selector = "anthropic/claude-fable-5"
    await stream._route_state.activate(FallbackTarget("kimi/k3"), "previous failure", cooldown_ms=0)
    # Permit the boundary probe of the primary while retaining the active pin.
    stream._route_state.primary_retry_at_ms = 0

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _anthropic_fable_usage(95.0),
        ):
            await stream.preflight_usage(model)

        assert stream._route_state.active is None
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_blocked_depleted_oauth_does_not_hide_healthy_api_key(tmp_path) -> None:
    """Availability follows the credential tier routing will actually use.

    A blocked OAuth row is still enumerated for usage and reports 0%, but the
    wire cascade falls through to the API key. Its healthy report prevents the
    provider from being skipped (agent review F4)."""
    store = AuthStore(tmp_path / "auth.db")
    oauth = store.upsert_credential("kimi", _oauth("oauth-kimi", "kimi-a"))
    store.block_credential(oauth.id, "kimi", block_ms=60_000)
    api = store.upsert_credential("kimi", {"key": "sk-kimi", "source": "login"})
    stream = create_stream_fn(store, {"retry": {"usageAwareFallback": True}})

    selected = await store.get_oauth_access("kimi", "session-a", read_only=True)
    assert selected is not None and selected.credential_id == api.id

    async def usage_for_access(_client, provider, *, access_token=None, api_key=None, **_kwargs):
        assert provider == "kimi"
        return _kimi_usage(100.0 if access_token else 20.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            verdict = await stream._provider_quota_availability(
                "kimi", "k3", reserve_percent=10, cache={}
            )

        assert verdict == "usable"
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_depleted_selected_api_key_does_not_hide_an_unprobed_sibling(tmp_path) -> None:
    """One empty key is not proof every key on the provider is empty.

    The credential store exposes only the selected secret. When another API
    key row exists, a depleted selected key therefore fails open so the stream
    rotation can reach the sibling instead of skipping the provider (review
    F5)."""
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("kimi", {"key": "sk-empty", "source": "login"})
    second = store.upsert_credential("kimi", {"key": "sk-funded", "source": "login"})
    session = _session_hashing_to_first_row(2)
    stream = create_stream_fn(store, {"retry": {"usageAwareFallback": True}}, session_id=session)

    selected = await store.get_oauth_access("kimi", session, read_only=True)
    assert selected is not None and selected.credential_id == first.id

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _kimi_usage(100.0),
        ):
            verdict = await stream._provider_quota_availability(
                "kimi", "k3", reserve_percent=10, cache={}
            )

        assert second.id != selected.credential_id
        assert verdict == "unknown"
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_transport_fallback_cooldown_skips_quota_probe(tmp_path) -> None:
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True}},
        session_id="session-a",
    )
    stream._primary_selector = "anthropic/claude-opus-5"
    await stream._route_state.activate(
        FallbackTarget("openai/gpt-5.3-codex"),
        "provider failure",
        cooldown_ms=60_000,
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")

    try:
        with patch("local_operator.providers.usage.fetch_usage") as fetch:
            await stream.preflight_usage(model)
        fetch.assert_not_called()
        assert stream._route_state.active == FallbackTarget("openai/gpt-5.3-codex")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_withdraw_fallback_clears_the_route_and_reopens_the_quota_probe(
    tmp_path,
) -> None:
    """An explicit re-selection ends the rescue route, whatever the cooldown says.

    The selector-driven clear in ``preflight_usage`` never fires for a
    same-model re-selection, so the session tells the stream fn directly. Two
    things must move: the pinned route (including its cooldown — the user's
    choice outranks the backoff that pinned the fallback), and the quota memo,
    so the next boundary re-probes the primary instead of trusting the reading
    that caused the pin. Silent: the owning session already announced this
    withdrawal, so no settle edge fires here.
    """
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    stream = create_stream_fn(
        store,
        {"retry": {"usageAwareFallback": True}},
        session_id="session-a",
    )
    stream._primary_selector = "anthropic/claude-opus-5"
    await stream._route_state.activate(
        FallbackTarget("openai/gpt-5.3-codex"),
        "provider failure",
        cooldown_ms=3_600_000,  # an hour of cooldown must not survive the withdrawal
    )
    settles: list[Any] = []
    # Installed AFTER the pin: ``activate`` settles the pin itself, and the
    # assertion below is that the WITHDRAWAL adds nothing to this list.
    stream._route_state.on_settle = lambda target, reason: settles.append((target, reason))
    # A fresh quota reading just pinned the fallback; the withdrawal must not
    # inherit it.
    stream._usage_checked_selector = "anthropic/claude-opus-5"
    stream._usage_checked_at = time.monotonic()

    try:
        stream.withdraw_fallback()
        assert stream._route_state.active is None
        assert stream._route_state.primary_retry_due()  # cooldown gone with the pin
        assert stream._usage_checked_at == 0.0  # memo reset: next boundary re-probes
        assert settles == []  # silent — the session owns this announcement
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_startup_preflight_recovers_a_blocked_account_that_still_has_quota(
    tmp_path,
) -> None:
    """A block is a stale verdict, not ground truth — re-probe before failover.

    The all-blocked branch used to fail over to another provider without ever
    asking whether the blocked accounts had come back to life. A window reset
    (or a block written for a tier cap the current model does not draw on)
    leaves live quota stranded behind the backoff. The blocked account is
    re-probed, found healthy, unblocked, and kept in service — no failover."""
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.block_credential(account.id, "anthropic", block_ms=60_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _anthropic_usage(25.0),
        ) as fetch:
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
        fetch.assert_called()
        # The recovered account serves again; no provider failover.
        assert not store.is_blocked(account.id, "anthropic")
        assert stream._route_state.active is None
        assert any("recovered" in notice for notice in notices)
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_startup_preflight_fails_over_only_when_every_account_is_exhausted(
    tmp_path,
) -> None:
    """The re-probe confirms exhaustion before the provider fallback fires.

    With every account still genuinely at 100%, recovery finds no usable
    login, the block is stood back up, and only then does the session move to
    the configured cross-provider fallback."""
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.block_credential(account.id, "anthropic", block_ms=60_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))

    try:
        with patch(
            "local_operator.providers.usage.fetch_usage",
            side_effect=lambda *_args, **_kwargs: _anthropic_usage(100.0),
        ) as fetch:
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
        fetch.assert_called()
        assert store.is_blocked(account.id, "anthropic")
        assert stream._route_state.active == FallbackTarget("openai/gpt-5.3-codex")
        assert notices == [
            "anthropic quota exhausted (0% remaining) — falling back to openai/gpt-5.3-codex"
        ]
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_blocked_recovery_probes_run_bounded_and_concurrently(tmp_path) -> None:
    """The recovery walk probes blocked rows in bounded waves, not serially.

    The serial walk was a network train on the time-to-usable path (a refresh
    plus a usage GET per row, one after another), and it is what generated the
    self-inflicted 429 burst whose backoff then poisoned the next boot. The
    bound matters as much as the concurrency: Anthropic/OpenAI rate-limit the
    usage endpoint per source IP regardless of account, so an unbounded gather
    would just make the burst faster. Asserting the PEAK is the regression
    guard against someone later "optimizing" the cap away."""
    store = AuthStore(tmp_path / "auth.db")
    rows = [
        store.upsert_credential("anthropic", _oauth(f"oauth-{i}", f"acct-{i}")) for i in range(6)
    ]
    for row in rows:
        store.block_credential(row.id, "anthropic", block_ms=60_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )

    inflight = 0
    peak = 0

    async def fetch(*_args: Any, **kwargs: Any) -> Any:
        nonlocal inflight, peak
        token = str(kwargs.get("access_token") or kwargs.get("api_key"))
        if not token.startswith("oauth-"):
            return _anthropic_usage(100.0)
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0)  # yield so siblings can overlap
        inflight -= 1
        return _anthropic_usage(100.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=fetch):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
        assert peak > 1, "probes ran serially — the walk is still a network train"
        assert peak <= stream.USAGE_RECOVERY_PROBE_CONCURRENCY
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_blocked_recovery_verdict_follows_row_order_not_completion_order(
    tmp_path,
) -> None:
    """Concurrency must not turn "first definite verdict" into a race.

    ``asyncio.gather`` preserves result order, and the walk scans that ordered
    list — so the winner is the first row, not the fastest responder. Here the
    LAST row answers first; picking it would mis-attribute the recovery and
    pin the session to an account chosen by network timing."""
    store = AuthStore(tmp_path / "auth.db")
    rows = [
        store.upsert_credential("anthropic", _oauth(f"oauth-{i}", f"acct-{i}")) for i in range(3)
    ]
    for row in rows:
        store.block_credential(row.id, "anthropic", block_ms=60_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )

    async def fetch(*_args: Any, **kwargs: Any) -> Any:
        token = str(kwargs.get("access_token") or kwargs.get("api_key"))
        if not token.startswith("oauth-"):
            return _anthropic_usage(100.0)
        # Earlier rows are SLOWER, so completion order is the reverse of row
        # order and a completion-order bug would settle on oauth-2.
        index = int(token.rsplit("-", 1)[1])
        for _ in range((3 - index) * 4):
            await asyncio.sleep(0)
        return _anthropic_usage(25.0)  # every row is healthy

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=fetch):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
        # The FIRST row recovered; its siblings keep their blocks.
        assert not store.is_blocked(rows[0].id, "anthropic")
        assert store.is_blocked(rows[1].id, "anthropic")
        assert store.is_blocked(rows[2].id, "anthropic")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_blocked_recovery_does_not_reserve_rows_it_never_probed(tmp_path) -> None:
    """Only rows in a LAUNCHED wave are recorded as judged.

    ``attempted_ids`` is the walk's termination guarantee, but reserving the
    unprobed tail up front would retire — for the whole boundary — credentials
    nobody ever looked at. With a verdict found in the first wave, the later
    rows must remain un-probed AND un-reserved."""
    store = AuthStore(tmp_path / "auth.db")
    rows = [
        store.upsert_credential("anthropic", _oauth(f"oauth-{i}", f"acct-{i}")) for i in range(6)
    ]
    for row in rows:
        store.block_credential(row.id, "anthropic", block_ms=60_000)
    stream = create_stream_fn(
        store, {"retry": {"usageAwareFallback": True}}, session_id="session-a"
    )

    probed: list[str] = []

    def fetch(*_args: Any, **kwargs: Any) -> Any:
        token = str(kwargs.get("access_token") or kwargs.get("api_key"))
        probed.append(token)
        return _anthropic_usage(25.0)  # healthy: the first wave settles it

    attempted: set[int] = set()
    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=fetch):
            recovered = await stream._recover_blocked_accounts(
                ModelSpec(provider="anthropic", model_id="claude-opus-5"),
                "anthropic",
                rows,
                RetrySettings.from_settings({"retry": {"usageAwareFallback": True}}),
                attempted,
            )
        assert recovered is not None
        bound = stream.USAGE_RECOVERY_PROBE_CONCURRENCY
        assert len(probed) <= bound, "probed beyond the first wave"
        # The tail was never probed, so it must still be a candidate later.
        assert attempted == {row.id for row in rows[:bound]}
        for row in rows[bound:]:
            assert row.id not in attempted
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_preflight_reads_each_account_usage_once_per_boundary(tmp_path) -> None:
    """One boundary asks one account's usage endpoint exactly once.

    Eight requests to one provider's usage endpoint in a single boot is the
    defect behind the self-inflicted 429s. The walk carries two memos already
    — ``attempted_ids`` (rows judged) and ``quota_cache`` (provider+model
    verdicts) — and NEITHER dedupes the underlying per-account report: a
    fallback chain that lists the walk's own provider re-enumerates accounts
    the primary probe already read. This is that regression guard."""
    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "acct-a"))
    store.block_credential(account.id, "anthropic", block_ms=0)  # unblocked
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                # Lists anthropic itself, so the fallback scan re-enumerates
                # the very account the primary probe just read.
                "fallbackChains": {"default": ["anthropic/claude-fable-5"]},
            }
        },
        session_id="session-a",
    )

    calls: list[str] = []

    def fetch(*_args: Any, **kwargs: Any) -> Any:
        calls.append(str(kwargs.get("access_token") or kwargs.get("api_key")))
        return _anthropic_usage(92.0)  # reserve: drives the fallback scan

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=fetch):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
        assert calls.count("oauth-a") == 1, f"account probed {calls.count('oauth-a')}x: {calls}"
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_recovery_wave_prefers_a_usable_sibling_over_an_earlier_depleted_one(
    tmp_path,
) -> None:
    """A depleted first row in a wave must not hide a later sibling with quota.

    The serial walk returned the first definite verdict of any kind, then the
    caller re-entered and walked the remaining blocked rows — so depleted
    never hid a later healthy sibling (review F2). The concurrent form
    reserves the whole wave in ``attempted_ids`` before the gather, which
    would make that re-entry skip the rest of the wave. Preferring a usable
    verdict already in hand is what keeps the observable identical: the
    depleted row stays blocked, the healthy sibling serves, no provider hop.
    """
    store = AuthStore(tmp_path / "auth.db")
    depleted = store.upsert_credential("anthropic", _oauth("oauth-a", "acct-a"))
    healthy = store.upsert_credential("anthropic", _oauth("oauth-b", "acct-b"))
    for row in (depleted, healthy):
        store.block_credential(row.id, "anthropic", block_ms=60_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )

    def fetch(*_args: Any, **kwargs: Any) -> Any:
        token = str(kwargs.get("access_token") or kwargs.get("api_key"))
        if token == "oauth-a":
            return _anthropic_usage(100.0)
        if token == "oauth-b":
            return _anthropic_usage(25.0)
        return _anthropic_usage(100.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=fetch):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
        assert stream._route_state.active is None
        assert store.is_blocked(depleted.id, "anthropic")
        assert not store.is_blocked(healthy.id, "anthropic")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_recovery_probe_failure_leaves_the_block_standing(tmp_path) -> None:
    """A transient failure keeps the existing verdict; it never invents one.

    The concurrent form contains exceptions inside each probe rather than
    letting one escape into the gather, where it would cancel its siblings.
    A raising probe must read exactly as the serial walk's "unreachable
    endpoint" did: no verdict, block stands, siblings still evaluated."""
    store = AuthStore(tmp_path / "auth.db")
    first = store.upsert_credential("anthropic", _oauth("oauth-a", "acct-a"))
    second = store.upsert_credential("anthropic", _oauth("oauth-b", "acct-b"))
    for row in (first, second):
        store.block_credential(row.id, "anthropic", block_ms=60_000)
    stream = create_stream_fn(
        store, {"retry": {"usageAwareFallback": True}}, session_id="session-a"
    )

    def fetch(*_args: Any, **kwargs: Any) -> Any:
        token = str(kwargs.get("access_token") or kwargs.get("api_key"))
        if token == "oauth-a":
            raise RuntimeError("transient transport failure")
        return _anthropic_usage(25.0)

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=fetch):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
        # The failed probe's block stands; the healthy sibling still recovered,
        # so one bad account never suppresses the rest of the pool.
        assert store.is_blocked(first.id, "anthropic")
        assert not store.is_blocked(second.id, "anthropic")
    finally:
        await stream.close()
        store.close()


# ---------------------------------------------------------------------------
# validate_model — same endpoints/headers as the legacy chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hosting, model, status_code, response_json, expected_result, expected_url, expected_headers",
    [
        (
            "openai",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "openai",
            "test_model",
            404,
            {},
            False,
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "openai",
            "test_model",
            200,
            {"data": []},
            False,
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "ollama",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "http://localhost:11434/v1/models",
            None,
        ),
        ("ollama", "test_model", 404, {}, False, "http://localhost:11434/v1/models", None),
        (
            "deepseek",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.deepseek.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "openrouter",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://openrouter.ai/api/v1/models",
            {
                "HTTP-Referer": "https://local-operator.com",
                "X-OpenRouter-Title": "Local Operator",
                "X-Title": "Local Operator",
                "X-OpenRouter-Categories": "cli-agent,personal-agent",
                "Authorization": "Bearer test_key",
            },
        ),
        (
            "anthropic",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.anthropic.com/v1/models",
            {"x-api-key": "test_key", "anthropic-version": "2023-06-01"},
        ),
        (
            "anthropic",
            "test-model-latest",
            200,
            {"data": [{"id": "test-model-1234"}]},
            True,
            "https://api.anthropic.com/v1/models",
            {"x-api-key": "test_key", "anthropic-version": "2023-06-01"},
        ),
        (
            "kimi",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.moonshot.cn/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "alibaba",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "google",
            "test_model",
            200,
            {"models": [{"name": "test_model"}]},
            True,
            "https://generativelanguage.googleapis.com/v1/models",
            {"x-goog-api-key": "test_key"},
        ),
        (
            "mistral",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.mistral.ai/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "xai",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.x.ai/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
        (
            "radient",
            "test_model",
            200,
            {"data": [{"id": "test_model"}]},
            True,
            "https://api.radienthq.com/v1/models",
            {"Authorization": "Bearer test_key"},
        ),
    ],
)
def test_validate_model(
    mock_requests_get: MagicMock,
    hosting: str,
    model: str,
    status_code: int,
    response_json: dict[str, Any],
    expected_result: bool,
    expected_url: str,
    expected_headers: dict[str, Any] | None,
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_requests_get.return_value = mock_response

    api_key = SecretStr("test_key")
    result = validate_model(hosting, model, api_key)
    assert result == expected_result

    if hosting == "ollama":
        mock_requests_get.assert_called_once_with(
            expected_url,
            headers={"Authorization": "Bearer test_key"},
            timeout=10,
            allow_redirects=False,
        )
    elif expected_headers:
        mock_requests_get.assert_called_once_with(expected_url, headers=expected_headers)
    else:
        mock_requests_get.assert_called_once_with(expected_url)


def test_validate_model_unknown_hosting_returns_true():
    """Providers without a validation endpoint pass (legacy behaviour)."""
    assert validate_model("test", "anything", SecretStr("k")) is True


@patch("requests.get")
def test_validate_model_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    assert validate_model("openai", "test_model", SecretStr("test_key")) is False


@patch("requests.get")
def test_validate_model_exception(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException("API error")
    with pytest.raises(requests.exceptions.RequestException, match="API error"):
        validate_model("openai", "test_model", SecretStr("test_key"))


@patch("requests.get")
def test_validate_model_no_model_found(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}
    mock_get.return_value = mock_response
    assert validate_model("openai", "test_model", SecretStr("test_key")) is False


@patch("requests.get")
def test_validate_model_ollama_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "test_model"}]}
    mock_get.return_value = mock_response
    assert validate_model("ollama", "test_model", SecretStr("test_key")) is True


@patch("requests.get")
def test_validate_model_ollama_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": []}
    mock_get.return_value = mock_response
    assert validate_model("ollama", "test_model", SecretStr("test_key")) is False


class TestInfoFromListing:
    """Listing-backed model metadata (OpenRouter/Radient).

    The harness derives compaction thresholds from ``context_window`` and gates
    cache_control / snapcompact on the capability flags, so a listing that only
    yields price leaves compaction silently disabled — these pin the mapping.
    """

    @staticmethod
    def _listing(**overrides: object):
        from local_operator.clients.openrouter import (
            OpenRouterListModelsResponse,
            OpenRouterModelData,
            OpenRouterModelPricing,
        )

        payload = {
            "id": "vendor/model",
            "name": "Model",
            "description": "d",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "context_length": 1_000_000,
            "top_provider": {"context_length": 200_000, "max_completion_tokens": 64_000},
            "architecture": {"input_modalities": ["image", "text"]},
        }
        payload.update(overrides)
        return (
            OpenRouterListModelsResponse(data=[OpenRouterModelData.model_validate(payload)]),
            OpenRouterModelPricing,
        )

    def test_window_takes_the_routed_provider_not_the_headline(self):
        from local_operator.model.configure import _info_from_listing
        from local_operator.model.registry import openrouter_default_model_info

        listing, _ = self._listing()
        info = _info_from_listing(listing, "vendor/model", openrouter_default_model_info, "or")
        # 1M is the best any provider offers; 200k is what the routed one serves.
        assert info.context_window == 200_000
        assert info.max_tokens == 64_000

    def test_single_window_source_is_used_when_top_provider_absent(self):
        from local_operator.model.configure import _info_from_listing
        from local_operator.model.registry import openrouter_default_model_info

        listing, _ = self._listing(top_provider={})
        info = _info_from_listing(listing, "vendor/model", openrouter_default_model_info, "or")
        assert info.context_window == 1_000_000

    def test_cache_read_price_implies_prompt_cache_support(self):
        from local_operator.model.configure import _info_from_listing
        from local_operator.model.registry import openrouter_default_model_info

        listing, _ = self._listing(
            pricing={
                "prompt": "0.00000009",
                "completion": "0.00000018",
                "input_cache_read": "0.000000018",
            }
        )
        info = _info_from_listing(listing, "vendor/model", openrouter_default_model_info, "or")
        assert info.supports_prompt_cache is True
        assert info.cache_reads_price == pytest.approx(0.018)
        # Implicit-cache providers quote no write price: fall back to input.
        assert info.cache_writes_price == pytest.approx(0.09)

    def test_no_cache_price_leaves_cache_unsupported(self):
        from local_operator.model.configure import _info_from_listing
        from local_operator.model.registry import openrouter_default_model_info

        listing, _ = self._listing()
        info = _info_from_listing(listing, "vendor/model", openrouter_default_model_info, "or")
        assert info.supports_prompt_cache is False

    def test_image_modality_maps_to_supports_images(self):
        from local_operator.model.configure import _info_from_listing
        from local_operator.model.registry import openrouter_default_model_info

        listing, _ = self._listing()
        assert (
            _info_from_listing(listing, "vendor/model", openrouter_default_model_info, "or")
        ).supports_images is True
        text_only, _ = self._listing(architecture={"input_modalities": ["text"]})
        assert (
            _info_from_listing(text_only, "vendor/model", openrouter_default_model_info, "or")
        ).supports_images is False

    def test_missing_model_raises(self):
        from local_operator.model.configure import _info_from_listing
        from local_operator.model.registry import openrouter_default_model_info

        listing, _ = self._listing()
        with pytest.raises(ValueError, match="Model not found"):
            _info_from_listing(listing, "vendor/other", openrouter_default_model_info, "or")


class TestAnIsolatedRequestTouchesNoneOfTheSessionsSharedState:
    """``SessionStreamFn`` skips three session-wide steps for decoration.

    Auto-naming runs CONCURRENTLY with the user's turn now, so a title call and
    the turn reach this object at the same moment. Each step below is consumed
    or mutated by whichever request arrives first, which is why a decorative
    call must not take part in any of them.
    """

    @staticmethod
    def _handler(captured: dict[str, Any]) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=(
                    b'data: {"type":"response.completed","response":{"id":"resp_1",'
                    b'"usage":{"input_tokens":1,"output_tokens":1}}}\n\ndata: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        return handler

    @staticmethod
    def _stream(tmp_path: Any) -> tuple[Any, Any]:
        store = AuthStore(tmp_path / "auth.db")
        store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
        # Auto effort is off by default, and off it cannot consume the message
        # boundary — which would make "the isolated call did not consume it"
        # true of every request and prove nothing.
        stream = create_stream_fn(store, {"effort": {"auto": True}}, session_id="session-isolation")
        return stream, store

    async def _run(self, tmp_path: Any, *, isolated: bool) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        stream, store = self._stream(tmp_path)
        await stream._http.aclose()
        stream._http = httpx.AsyncClient(transport=httpx.MockTransport(self._handler(captured)))
        preflights: list[Any] = []
        real_preflight = stream.preflight_usage

        # Wraps rather than replaces: `preflight_usage` is also what CONSUMES
        # the message boundary, so a stub would make the boundary assertion
        # below pass for the wrong reason on the ordinary-request control.
        async def spy_preflight(model: Any) -> None:
            preflights.append(model)
            await real_preflight(model)

        stream.preflight_usage = spy_preflight  # type: ignore[method-assign]
        notices: list[tuple[str, str]] = []
        stream.set_notice_handler(lambda text, kind: notices.append((text, kind)))
        spec = build_model_spec("openai", "gpt-5.4")
        try:
            await _collect_stream(
                stream(
                    ChatRequest(
                        model=spec,
                        messages=[
                            Message.user(
                                "refactor the compaction cut-point selector and prove it "
                                "with an end-to-end test against the real transcript store"
                            )
                        ],
                        isolated=isolated,
                    ),
                    None,
                )
            )
        finally:
            await stream.close()
            store.close()
        return {
            "body": captured["body"],
            "preflights": preflights,
            "notices": notices,
            "boundary_pending": stream._message_boundary_pending,
            "message_effort": stream._message_effort,
        }

    @pytest.mark.asyncio
    async def test_it_leaves_the_message_boundary_for_the_turn(self, tmp_path) -> None:
        """The boundary is CONSUMED by the first request through. A title call
        arriving first would classify effort from its own prompt, freeze it for
        the turn's whole tool loop, and emit an "auto effort" notice for a
        request the user never made."""
        result = await self._run(tmp_path, isolated=True)
        assert result["boundary_pending"] is True
        assert result["message_effort"] is None
        assert result["notices"] == []

    @pytest.mark.asyncio
    async def test_it_runs_no_quota_preflight(self, tmp_path) -> None:
        """The preflight can BLOCK a credential and activate a fallback route
        for the whole session — an outcome no title is worth."""
        result = await self._run(tmp_path, isolated=True)
        assert result["preflights"] == []

    @pytest.mark.asyncio
    async def test_it_does_not_ride_the_sessions_prompt_cache_key(self, tmp_path) -> None:
        """The key identifies a request PREFIX. A naming call's prefix is a
        different system block, so sharing the key buys no hit and writes a
        competing entry under the name the turn's prefix is cached as."""
        result = await self._run(tmp_path, isolated=True)
        assert "prompt_cache_key" not in result["body"]

    @pytest.mark.asyncio
    async def test_an_ordinary_request_does_all_three(self, tmp_path) -> None:
        """The control. Without it, each assertion above could be passing
        because this fixture never reaches those code paths at all."""
        result = await self._run(tmp_path, isolated=False)
        assert result["boundary_pending"] is False
        assert result["message_effort"] is not None
        assert result["preflights"], "no preflight ran on an ordinary request"
        assert result["body"]["prompt_cache_key"] == "session-isolation"


class TestRouteSettleBridge:
    """The stream fn's route bridge: what the session's model display hangs off.

    ``set_route_handler`` receives BOTH edges — the pinned target on a
    fallback, ``None`` on recovery — because a display that only ever hears
    "fell back" keeps naming the fallback after the primary recovered.
    """

    def _stream(self, store, settings=None):
        return create_stream_fn(
            store,
            settings
            or {
                "retry": {
                    "usageAwareFallback": True,
                    "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
                }
            },
            session_id="session-route",
        )

    @pytest.mark.asyncio
    async def test_fallback_activation_reaches_the_route_handler(self, tmp_path) -> None:
        store = AuthStore(tmp_path / "auth.db")
        stream = self._stream(store)
        edges: list[tuple[Any, str]] = []
        stream.set_route_handler(lambda target, reason: edges.append((target, reason)))
        try:
            await stream._route_state.activate(
                FallbackTarget("openai/gpt-5.3-codex"), "provider failure"
            )
            assert len(edges) == 1
            target, reason = edges[0]
            assert target == FallbackTarget("openai/gpt-5.3-codex")
            assert reason == "provider failure"
        finally:
            await stream.close()
            store.close()

    @pytest.mark.asyncio
    async def test_preflight_recovery_settles_and_narrates(self, tmp_path) -> None:
        """The auth-only preflight path clears a pinned route as a SETTLED
        edge (handler told, notice printed), not as silent bookkeeping —
        otherwise the display keeps naming a fallback that stopped serving."""
        store = AuthStore(tmp_path / "auth.db")
        store.upsert_credential("anthropic", {"key": "sk-ant", "source": "login"})
        stream = create_stream_fn(
            store,
            {"retry": {"usageAwareFallback": False}},
            session_id="session-route",
        )
        edges: list[Any] = []
        notices: list[str] = []
        stream.set_route_handler(lambda target, reason: edges.append(target))
        stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
        model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
        stream._primary_selector = "anthropic/claude-opus-5"
        stream._route_state.active = FallbackTarget("openai/gpt-5.3-codex")
        try:
            await stream.preflight_usage(model)
            assert edges == [None]
            assert any("back to anthropic/claude-opus-5" in notice for notice in notices)
            assert any(notice.startswith("info:") for notice in notices)
        finally:
            await stream.close()
            store.close()

    @pytest.mark.asyncio
    async def test_restore_fallback_pins_without_announcing(self, tmp_path) -> None:
        """A resume's re-pin fires NEITHER handler — the transcript replay
        already narrates the original failure, and the restoring session set
        its own state from the same entry — but must seed the primary memo and
        a probe grace, or the next preflight clears the pin it was handed."""
        store = AuthStore(tmp_path / "auth.db")
        stream = self._stream(store)
        edges: list[Any] = []
        notices: list[str] = []
        stream.set_route_handler(lambda target, reason: edges.append(target))
        stream.set_notice_handler(lambda text, kind: notices.append(text))
        try:
            stream.restore_fallback("openai/gpt-5.3-codex", None, "anthropic/claude-opus-5")
            assert stream._route_state.active == FallbackTarget("openai/gpt-5.3-codex")
            assert stream._primary_selector == "anthropic/claude-opus-5"
            assert not stream._route_state.primary_retry_due()
            assert edges == []
            assert notices == []
        finally:
            await stream.close()
            store.close()

    @pytest.mark.asyncio
    async def test_restored_pin_survives_the_boot_preflight(self, tmp_path) -> None:
        """The TUI runs a quota preflight seconds after boot; a restored pin
        must survive it. The auth-only clear path would otherwise probe only
        "does the primary HAVE auth" — true throughout an outage — and unpin
        a fallback the provider is still failing behind."""
        store = AuthStore(tmp_path / "auth.db")
        store.upsert_credential("anthropic", {"key": "sk-ant", "source": "login"})
        stream = create_stream_fn(
            store,
            {"retry": {"usageAwareFallback": False}},
            session_id="session-route",
        )
        edges: list[Any] = []
        stream.set_route_handler(lambda target, reason: edges.append(target))
        try:
            stream.restore_fallback("openai/gpt-5.3-codex", None, "anthropic/claude-opus-5")
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-opus-5"))
            assert stream._route_state.active == FallbackTarget("openai/gpt-5.3-codex")
            assert edges == []
        finally:
            await stream.close()
            store.close()


# ---------------------------------------------------------------------------
# Blocked-sibling recovery: the stale-block trap (session f3c058d1, 2026-08-21)
# ---------------------------------------------------------------------------


def _fable_report(five_hour_used: float, seven_day_used: float, fable_used: float) -> Any:
    """The live incident shape: shared windows + a scoped Fable weekly."""
    return UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                amount=UsageAmount(used=five_hour_used, limit=100.0, unit="percent"),
                window="5h",
                shared=True,
                resets_at_ms=10**15,
            ),
            UsageLimit(
                id="anthropic:7d",
                label="7 day",
                amount=UsageAmount(used=seven_day_used, limit=100.0, unit="percent"),
                window="7d",
                shared=True,
                resets_at_ms=10**15,
            ),
            UsageLimit(
                id="anthropic:7d:fable",
                label="7 day (Fable)",
                amount=UsageAmount(used=fable_used, limit=100.0, unit="percent"),
                window="7d",
                shared=False,
                tier="fable",
                resets_at_ms=10**15,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_preflight_recovers_a_blocked_sibling_holding_model_quota(tmp_path) -> None:
    """The reported incident: the selected account is model-depleted (Fable at
    0%) while two accounts sit under stale blocks holding 8%/4% Fable. The
    unblocked pool has no sibling to rotate to, so preflight must probe the
    blocked rows and settle on the recovered one instead of hopping providers.

    The walk must attribute each verdict to the row it probed — a healthy
    unblocked sibling in the pool must not swallow the probes."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-live", "account-live"))
    blocked_a = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    blocked_b = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    for row in (blocked_a, blocked_b):
        store.block_credential(row.id, "anthropic", block_ms=3_600_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )
    notices: list[str] = []
    stream.set_notice_handler(lambda text, kind: notices.append(text))

    reports = {
        "oauth-live": _fable_report(0.0, 58.0, 100.0),  # Fable depleted
        "oauth-a": _fable_report(0.0, 92.0, 16.0),  # 16% Fable left
        "oauth-b": _fable_report(0.0, 96.0, 6.0),  # 6% Fable left
    }

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        assert access_token is not None
        return reports[access_token]

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-fable-5"))

        # No provider failover: a blocked account recovered with Fable quota.
        assert stream._route_state.active is None
        # The recovered account's block is lifted and the session is pinned to
        # it; the other blocked row keeps its block (its verdict was never read).
        assert not store.is_blocked(blocked_a.id, "anthropic")
        assert store.is_blocked(blocked_b.id, "anthropic")
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == blocked_a.id
        assert any("recovered" in notice for notice in notices)
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_preflight_falls_back_when_blocked_siblings_are_genuinely_depleted(
    tmp_path,
) -> None:
    """The honest-failover half: when every blocked account re-probes as
    depleted for the model, the provider fallback fires and the blocks stand."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-live", "account-live"))
    blocked = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.block_credential(blocked.id, "anthropic", block_ms=3_600_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        if access_token == "oauth-live":
            return _fable_report(0.0, 58.0, 100.0)
        return _fable_report(0.0, 100.0, 100.0)  # blocked account truly out

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-fable-5"))

        assert stream._route_state.active is not None
        assert stream._route_state.active.selector == "openai/gpt-5.3-codex"
        assert store.is_blocked(blocked.id, "anthropic")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_recovery_walks_past_a_depleted_blocked_row_to_one_with_quota(
    tmp_path,
) -> None:
    """Review F2: with several blocked rows, a first re-probe that says
    depleted must not end the walk — the row is re-blocked by the shared
    policy and the NEXT blocked row (holding quota) is probed and serves."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-live", "account-live"))
    blocked_a = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    blocked_b = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    for row in (blocked_a, blocked_b):
        store.block_credential(row.id, "anthropic", block_ms=3_600_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )

    reports = {
        "oauth-live": _fable_report(0.0, 58.0, 100.0),  # selected: Fable spent
        "oauth-a": _fable_report(0.0, 100.0, 100.0),  # blocked, truly out
        "oauth-b": _fable_report(0.0, 96.0, 6.0),  # blocked, 6% Fable left
    }

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        assert access_token is not None
        return reports[access_token]

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-fable-5"))

        # No provider hop: the second blocked row held quota and serves.
        assert stream._route_state.active is None
        assert store.is_blocked(blocked_a.id, "anthropic")
        assert not store.is_blocked(blocked_b.id, "anthropic")
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == blocked_b.id
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_two_depleted_blocked_rows_do_not_ping_pong_the_recovery_walk(
    tmp_path,
) -> None:
    """Review F5: the walk terminates, and a healthy row enumerated AFTER two
    depleted ones is still reached.

    The recursion is the point. A depleted re-probe hands the verdict back to
    ``_apply_account_health``, which re-blocks that row and walks the blocked
    pool again — so without recording which rows this boundary has already
    judged, the row cleared in one frame is blocked again by the next and
    re-enumerated by the one after. Two depleted blocked accounts ping-pong
    A→B→A until ``RecursionError`` kills the turn, and it kills it inside
    ``preflight_usage``, which ``__call__`` awaits unguarded.

    This is the live shape of the reported incident, not a constructed one:
    four Anthropic logins, the selected account model-depleted, two blocked
    accounts genuinely out, and one blocked account still holding quota that
    the crash meant nobody ever probed.
    """
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", _oauth("oauth-live", "account-live"))
    spent_a = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    spent_b = store.upsert_credential("anthropic", _oauth("oauth-b", "account-b"))
    healthy = store.upsert_credential("anthropic", _oauth("oauth-c", "account-c"))
    for row in (spent_a, spent_b, healthy):
        store.block_credential(row.id, "anthropic", block_ms=3_600_000)
    store.upsert_credential("openai", {"key": "sk-openai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["openai/gpt-5.3-codex"]},
            }
        },
        session_id="session-a",
    )

    reports = {
        "oauth-live": _fable_report(0.0, 58.0, 100.0),  # selected: Fable spent
        "oauth-a": _fable_report(0.0, 100.0, 100.0),  # blocked, truly out
        "oauth-b": _fable_report(47.0, 100.0, 34.0),  # blocked, truly out
        "oauth-c": _fable_report(10.0, 30.0, 5.0),  # blocked, but HEALTHY
    }
    probed: list[str] = []

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        assert access_token is not None
        probed.append(access_token)
        return reports[access_token]

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-fable-5"))

        # Terminated, and on the right account: no provider hop while an
        # Anthropic login still holds quota for this model.
        assert stream._route_state.active is None
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == healthy.id
        assert not store.is_blocked(healthy.id, "anthropic")
        assert store.is_blocked(spent_a.id, "anthropic")
        assert store.is_blocked(spent_b.id, "anthropic")

        # Each account is probed at most once per boundary. A count, not a
        # bare "it did not crash": unbounded re-probing would still terminate
        # by luck of ordering while spending a network round trip per frame.
        assert sorted(probed) == ["oauth-a", "oauth-b", "oauth-c", "oauth-live"]
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_family_scoped_blocks_leave_other_families_serving(tmp_path) -> None:
    """A spent family cap blocks the family, never the account.

    The reported state: four Anthropic accounts, one of them (gominerva)
    holding shared headroom but a 100% Fable weekly. Preflight for
    ``claude-fable-5`` must take that account out of Fable rotation while
    leaving it resolvable for opus — the account-wide block the old code
    wrote is what later made an opus session report "all credentials
    unusable" on a pool whose only spent window was Fable's."""
    store = AuthStore(tmp_path / "auth.db")
    gominerva = store.upsert_credential("anthropic", _oauth("oauth-gominerva", "a1"))
    radienthq = store.upsert_credential("anthropic", _oauth("oauth-radienthq", "a2"))
    pergamon = store.upsert_credential("anthropic", _oauth("oauth-pergamon", "a3"))
    gmail = store.upsert_credential("anthropic", _oauth("oauth-gmail", "a4"))
    store.upsert_credential("zai", {"key": "sk-zai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["zai/glm-5.3"]},
            }
        },
        session_id="session-a",
    )
    reports = {
        "oauth-gominerva": _fable_report(2.0, 59.0, 100.0),  # fable spent, shared fine
        "oauth-radienthq": _fable_report(0.0, 100.0, 34.0),  # shared 7d spent
        "oauth-pergamon": _fable_report(0.0, 100.0, 100.0),  # everything spent
        "oauth-gmail": _fable_report(100.0, 41.0, 79.0),  # shared 5h spent
    }

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return reports[access_token] if access_token is not None else None

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            stream.begin_message()
            await stream.preflight_usage(ModelSpec(provider="anthropic", model_id="claude-fable-5"))

        # The fable verdict on gominerva is family-scoped; the shared-window
        # verdicts on the other three are account-wide.
        assert not store.is_blocked(gominerva.id, "anthropic")
        assert store.is_blocked_for_model(gominerva.id, "anthropic", "claude-fable-5")
        assert not store.is_blocked_for_model(gominerva.id, "anthropic", "claude-opus-4-8")
        assert store.is_blocked(radienthq.id, "anthropic")
        assert store.is_blocked(pergamon.id, "anthropic")
        assert store.is_blocked(gmail.id, "anthropic")

        # Switching to opus resolves gominerva directly — no recovery probe
        # needed, no failover — because the family block never hid the row
        # from an opus resolve.
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            opus_spec = ModelSpec(provider="anthropic", model_id="claude-opus-4-8")
            stream.on_model_changed(opus_spec)
            stream.begin_message()
            await stream.preflight_usage(opus_spec)
        selected = await store.get_oauth_access(
            "anthropic", "session-a", model_id="claude-opus-4-8"
        )
        assert selected is not None and selected.credential_id == gominerva.id
        assert stream._route_state.active is None
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_reactive_stream_recovers_stale_account_blocks(tmp_path) -> None:
    """A request that skips preflight must not die on stale quota blocks.

    The reported terminal error: "All 4 OAuth sign-in credentials ... not
    usable right now" on an opus request, because every account carried a
    block written by an earlier fable-scoped verdict. Isolated requests
    (errands, subagents) never run the preflight that would have re-scoped
    those blocks, so the stream driver probes the blocked rows itself: the
    account whose shared windows still hold headroom is unblocked and serves."""
    import httpx

    store = AuthStore(tmp_path / "auth.db")
    gominerva = store.upsert_credential("anthropic", _oauth("oauth-gominerva", "a1"))
    store.upsert_credential("anthropic", _oauth("oauth-radienthq", "a2"))
    store.upsert_credential("anthropic", _oauth("oauth-pergamon", "a3"))
    gmail = store.upsert_credential("anthropic", _oauth("oauth-gmail", "a4"))
    stream = create_stream_fn(
        store,
        {"retry": {"enabled": True, "usageAwareFallback": True, "usageReservePercent": 10}},
        session_id="session-a",
    )
    reports = {
        "oauth-gominerva": _fable_report(2.0, 59.0, 100.0),
        "oauth-radienthq": _fable_report(0.0, 100.0, 34.0),
        "oauth-pergamon": _fable_report(0.0, 100.0, 100.0),
        "oauth-gmail": _fable_report(100.0, 41.0, 79.0),
    }
    # Legacy state: every account under an ACCOUNT-WIDE block, as an older
    # build wrote for the same fable verdicts.
    for row in store.list_credentials("anthropic"):
        store.block_credential(row.id, "anthropic", block_ms=60 * 60 * 1000)

    anthropic_ok = (
        'data: {"type": "message_start", "message": {"usage": {"input_tokens": 10}}}\n\n'
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}}\n\n'
        'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"},'
        ' "usage": {"output_tokens": 1}}\n\n'
        'data: {"type": "message_stop"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "anthropic" in str(request.url):
            if "oauth-gominerva" in request.headers.get("authorization", ""):
                return httpx.Response(
                    200, text=anthropic_ok, headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(
                429,
                json={"error": {"message": "This request would exceed your account's rate limit."}},
            )
        return httpx.Response(500, json={})

    await stream._http.aclose()
    stream._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return reports[access_token] if access_token is not None else None

    try:
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            events = [
                event
                async for event in stream(
                    ChatRequest(
                        model=ModelSpec(provider="anthropic", model_id="claude-opus-4-8"),
                        messages=[Message.user("hi")],
                        isolated=True,
                    ),
                    None,
                )
            ]
        assert events  # the turn completed on the recovered account
        assert not store.is_blocked(gominerva.id, "anthropic")
        # The genuinely spent accounts keep their blocks.
        assert store.is_blocked(gmail.id, "anthropic")
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_quota_pinned_fallback_is_reprobed_at_the_next_boundary(tmp_path) -> None:
    """A quota pin never outlives the primary's recovery.

    The reported symptom: a session ran fine, tripped onto a fallback, then
    stayed glued to it — the pin's cooldown (sized to an advertised quota
    reset, hours long) suppressed the re-probe that would have shown the
    primary serving again. Quota pins are re-probed at every message
    boundary; a healthy primary withdraws the pin immediately."""
    from local_operator.providers.failover import FallbackTarget

    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("zai", {"key": "sk-zai", "source": "login"})
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "usageAwareFallback": True,
                "usageReservePercent": 10,
                "fallbackChains": {"default": ["zai/glm-5.3"]},
            }
        },
        session_id="session-a",
    )
    state = {"report": _fable_report(100.0, 41.0, 79.0)}  # 5h spent: depleted

    async def usage_for_access(_client, _provider, *, access_token=None, **_kwargs):
        return state["report"]

    try:
        opus_spec = ModelSpec(provider="anthropic", model_id="claude-opus-4-8")
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            stream.begin_message()
            await stream.preflight_usage(opus_spec)
        assert stream._route_state.active is not None

        # A long cooldown lands on the pin (a 24h advertised reset); the next
        # boundary must STILL re-probe because the pin is quota evidence.
        await stream._route_state.activate(
            FallbackTarget("zai/glm-5.3", None),
            "provider failure",
            cooldown_ms=24 * 60 * 60 * 1000,
            quota=True,
        )
        state["report"] = _fable_report(5.0, 41.0, 79.0)  # 5h window reset
        with patch("local_operator.providers.usage.fetch_usage", side_effect=usage_for_access):
            stream.begin_message()
            await stream.preflight_usage(opus_spec)
        assert stream._route_state.active is None
        selected = await store.get_oauth_access("anthropic", "session-a")
        assert selected is not None and selected.credential_id == account.id
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_preflight_reuses_stale_row_when_a_peer_holds_the_lease(tmp_path) -> None:
    """Cross-process fan-out collapse, through the real configure→helper wiring.

    Two sessions share one HOME (hence one ``usage_cache.db``). Session A holds
    the account's preflight fetch lease, standing in for a peer mid-fetch. Session
    B's ``preflight_usage`` must reach a real verdict off A's last-good row WITHOUT
    the patched fetcher ever being invoked for that account — proving the ``:pf:``
    lease diverts the duplicate network hit that earned the endpoint its 429 storm
    (BUG 3). This exercises the whole path (``_cached_account_usage`` →
    ``account_preflight_key`` → ``leased_account_usage``), not the helper alone."""
    from local_operator.providers.usage_cache import account_preflight_key

    store = AuthStore(tmp_path / "auth.db")
    account = store.upsert_credential("anthropic", _oauth("oauth-a", "account-a"))
    store.upsert_credential("zai", {"key": "sk-zai", "source": "login"})
    settings = {
        "retry": {
            "usageAwareFallback": True,
            "usageReservePercent": 10,
            "fallbackChains": {"default": ["zai/glm-5.3"]},
        }
    }
    stream_a = create_stream_fn(store, settings, session_id="session-a")
    stream_b = create_stream_fn(store, settings, session_id="session-b")
    model = ModelSpec(provider="anthropic", model_id="claude-opus-4-8")
    # The reactive probe keys on ``access.account_id`` here (``_oauth`` sets no
    # email), so this is the exact row site 2150 will read for account-a.
    key = account_preflight_key("anthropic", "account-a")

    calls: list[dict[str, Any]] = []

    async def recorder(*_args, **kwargs):
        # If the lease-loser ever crossed the network this would fire (and, being
        # a depleted 99% report, would flip the verdict) — so an empty ``calls``
        # is a two-sided proof the network was NOT crossed.
        calls.append(kwargs)
        return _anthropic_usage(99.0)

    try:
        # A seeds a healthy last-good row and holds the lease (peer mid-fetch).
        cache_a = stream_a._usage_cache_store()
        assert cache_a is not None
        cache_a.set(key, "anthropic", [_anthropic_usage(20.0)], expires_at_ms=cache_a._now_ms() - 1)
        assert cache_a.try_lease(key) is True

        with patch("local_operator.providers.usage.fetch_usage", side_effect=recorder):
            stream_b.begin_message()
            await stream_b.preflight_usage(model)

        # Verdict reached off the stale healthy row: account stays in service, no
        # fallback pinned — and the patched fetcher was never called for account-a.
        assert calls == []
        assert stream_b._route_state.active is None
        assert not store.is_blocked(account.id, "anthropic")
    finally:
        await stream_a.close()
        await stream_b.close()
        store.close()


# ---------------------------------------------------------------------------
# Anthropic 1h prompt-cache TTL: settings → client, last usage → next request
# ---------------------------------------------------------------------------


def test_anthropic_cache_ttl_threshold_setting_reads_like_openai_api() -> None:
    """Same resolution rules as ``_openai_api_mode``: missing/malformed → the
    default, an explicit non-negative int (including the 0 off switch) wins."""
    from local_operator.model.configure import ANTHROPIC_CACHE_TTL_1H_MIN_CONTEXT_TOKENS
    from local_operator.model.configure import (
        _anthropic_cache_ttl_1h_min_context_tokens as read,
    )

    assert read(None) == ANTHROPIC_CACHE_TTL_1H_MIN_CONTEXT_TOKENS
    assert read({}) == ANTHROPIC_CACHE_TTL_1H_MIN_CONTEXT_TOKENS
    assert read({"providers": {"openai": {"api": "responses"}}}) == (
        ANTHROPIC_CACHE_TTL_1H_MIN_CONTEXT_TOKENS
    )
    assert read({"providers": {"anthropic": {"cache_ttl_1h_min_context_tokens": 0}}}) == 0
    assert read({"providers": {"anthropic": {"cache_ttl_1h_min_context_tokens": 42}}}) == 42
    for bad in ("150000", -1, True, None, 1.5):
        assert read({"providers": {"anthropic": {"cache_ttl_1h_min_context_tokens": bad}}}) == (
            ANTHROPIC_CACHE_TTL_1H_MIN_CONTEXT_TOKENS
        )


def _anthropic_sse(context_tokens: int, *, tool_call: bool = False) -> bytes:
    """One mocked Anthropic stream whose usage adds up to ``context_tokens``
    (the client derives ``Usage.context_tokens`` as input + cache read +
    cache write). ``tool_call`` ends the message in a ``tool_use`` for the
    ``echo`` tool so the harness loop makes a second call in the SAME turn."""
    head = (
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":5,'
        b'"cache_read_input_tokens":' + str(context_tokens - 5).encode() + b","
        b'"cache_creation_input_tokens":0}}}\n\n'
    )
    if tool_call:
        return (
            head + b'data: {"type":"content_block_start","index":0,"content_block":'
            b'{"type":"tool_use","id":"tu_1","name":"echo"}}\n\n'
            b'data: {"type":"content_block_delta","index":0,"delta":'
            b'{"type":"input_json_delta","partial_json":"{\\"text\\":\\"x\\"}"}}\n\n'
            b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
            b'"usage":{"output_tokens":1}}\n\n'
        )
    return (
        head + b'data: {"type":"content_block_start","index":0,"content_block":'
        b'{"type":"text","text":""}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":'
        b'{"type":"text_delta","text":"ok"}}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":1}}\n\n'
    )


def _cache_markers(body: dict[str, Any]) -> list[dict[str, Any]]:
    found = [e["cache_control"] for e in body["system"] if "cache_control" in e]
    for message in body["messages"]:
        found.extend(b["cache_control"] for b in message["content"] if "cache_control" in b)
    return found


def _echo_tool():
    from local_operator.harness.types import AgentTool, TextContent, ToolResult

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    return AgentTool(
        name="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )


def _anthropic_session(tmp_path, name: str, stream, *, blocks: list[str]):
    """A real ``Session`` on the real stream fn — the integration under test
    is the harness loop stamping the hint per request, so nothing here may
    touch the hint by hand."""
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    return Session(
        model=build_model_spec("anthropic", "claude-opus-4-8"),
        stream_fn=stream,
        tools=[_echo_tool()],
        transcript=Transcript(tmp_path / name),
        system_blocks_provider=lambda: blocks,
    )


def _ttl_stream(tmp_path, handler):
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", {"key": "sk-ant", "source": "login"})
    settings = {"providers": {"anthropic": {"cache_ttl_1h_min_context_tokens": 150_000}}}
    stream = create_stream_fn(store, settings, session_id="session-ttl")
    stream._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stream._transport.http = stream._http
    return store, stream


@pytest.mark.asyncio
async def test_session_stream_feeds_last_context_into_anthropic_ttl(tmp_path) -> None:
    """The provider's reported context on call N picks the TTL on call N+1 —
    INSIDE the same turn, not only at the turn boundary (review F9).

    A real ``Session`` drives a real stream fn; nothing sets the hint by
    hand. Turn 1, call 1 has no hint and a tiny body, so the byte estimate
    keeps it at 5m even though the threshold is low; the mock answers with
    a tool call and a 200k context. Call 2 — same turn, after the tool ran
    — goes out with ``ttl: 1h`` on every marker: a subagent is one turn for
    its whole life and a long tool loop crosses the threshold mid-run, so a
    hint advanced only at the turn edge would never reach either. An
    isolated errand (``complete_once``) between the turns neither reads nor
    moves the hint: its own body decides (5m) and its SMALL report (5k)
    does not enter the conversation, so turn 2's first call still goes 1h
    on the session's own 200k. Dropping the isolation guard would have to
    fail this test, not leave it green.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        # The errand is the one request without the turn's tool schema (it
        # carries no tools and no history). Deliberately DIFFERENT counts:
        # 200k for the turn, 5k for the errand — a mock that reports the same
        # figure for every call cannot prove the errand was excluded.
        errand = not body.get("tools")
        content = (
            _anthropic_sse(5_000) if errand else _anthropic_sse(200_000, tool_call=len(bodies) == 1)
        )
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    store, stream = _ttl_stream(tmp_path, handler)
    session = _anthropic_session(
        tmp_path, "sess", stream, blocks=["instructions", "inventory", "skills", "env"]
    )
    try:
        await session.prompt("first")
        await session.complete_once("name it", "title?")
        await session.prompt("second")
    finally:
        await session.dispose()
        await stream.close()
        store.close()

    assert len(bodies) == 4
    # Turn 1 / call 1: no hint yet, tiny body → 5m by the estimate.
    assert _cache_markers(bodies[0]) and all(
        m == {"type": "ephemeral"} for m in _cache_markers(bodies[0])
    )
    # Turn 1 / call 2 (after the tool ran): call 1 reported 200k → 1h NOW.
    assert bodies[1]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert _cache_markers(bodies[1]) and all(
        m == {"type": "ephemeral", "ttl": "1h"} for m in _cache_markers(bodies[1])
    )
    # Isolated errand: the session's hint says 200k, but its request carries
    # none and its small body stays 5m by the estimate.
    assert not bodies[2].get("tools")
    assert [b["text"] for m in bodies[2]["messages"] for b in m["content"]] == ["title?"]
    assert all(m == {"type": "ephemeral"} for m in _cache_markers(bodies[2]))
    # Turn 2 / call 1: the conversation's own 200k (the errand's 5k never
    # entered it) → 1h on every marker.
    assert _cache_markers(bodies[3]) and all(
        m == {"type": "ephemeral", "ttl": "1h"} for m in _cache_markers(bodies[3])
    )
    assert session._context_tokens_hint == 200_000


@pytest.mark.asyncio
async def test_session_stream_hint_is_per_conversation_not_per_stream_fn(
    tmp_path,
) -> None:
    """Parent and child share a transport, with independent counted boundaries.

    A child's first call cannot inherit its parent's 300k calibration and its
    completion cannot overwrite that calibration. Real Session requests drive
    both owners, without manually supplying corrected hints.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        is_child = body["system"][0]["text"] == "child instructions"
        content = _anthropic_sse(10_000 if is_child else 300_000)
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    store, stream = _ttl_stream(tmp_path, handler)
    parent = _anthropic_session(
        tmp_path, "parent", stream, blocks=["instructions", "inventory", "skills", "env"]
    )
    child = None
    child_stream = None
    try:
        await parent.prompt("parent turn")
        child_stream = stream.fork("child")
        child = _anthropic_session(tmp_path, "child", child_stream, blocks=["child instructions"])
        await child.prompt("child errand")
        await parent.prompt("parent again")
    finally:
        if child is not None:
            await child.dispose()
            if child_stream is not None:
                await child_stream.close()
        await parent.dispose()
        await stream.close()
        store.close()

    assert len(bodies) == 3
    # Parent, first call: no hint, small body → 5m; the mock then reports 300k.
    assert _cache_markers(bodies[0]) and all(
        m == {"type": "ephemeral"} for m in _cache_markers(bodies[0])
    )
    assert parent._context_tokens_hint == 300_000
    # Child (same stream fn, NO hint of its own): its small body decides and
    # stays 5m — contamination (a), the parent's 300k on a fresh ~10k prefix.
    assert bodies[1]["system"][0]["text"] == "child instructions"
    assert all(m == {"type": "ephemeral"} for m in _cache_markers(bodies[1]))
    assert child is not None and child._context_tokens_hint == 10_000
    # Parent again after the child: 1h on its OWN 300k, proving the child's
    # construction and its 10k report changed nothing the parent reads —
    # contamination (b), the parent downgraded to 5m at the resume moment.
    assert bodies[2]["system"][0]["text"] == "instructions"
    assert _cache_markers(bodies[2]) and all(
        m == {"type": "ephemeral", "ttl": "1h"} for m in _cache_markers(bodies[2])
    )
    assert parent._context_tokens_hint == 300_000


# -- the muse-spark overflow, end to end ---------------------------------------


#: OpenRouter's listing entry for ``meta/muse-spark-1.3``, trimmed to the fields
#: the resolver reads. The two numbers are verbatim from the live catalogue on
#: 2026-09-02 and are the whole bug: ``max_completion_tokens`` is exactly 0.9 of
#: ``context_length``, and providers count prompt + max_tokens against the window
#: at admission, so the advertised cap alone consumes 90% of the model.
_MUSE_SPARK_LISTING_ENTRY = {
    "id": "meta/muse-spark-1.3",
    "name": "Meta: Muse Spark 1.3",
    "context_length": 1_048_576,
    "top_provider": {"context_length": 1_048_576, "max_completion_tokens": 943_718},
    "pricing": {"prompt": "0.00000125", "completion": "0.00000425"},
    "architecture": {"input_modalities": ["text"]},
}


def test_muse_spark_listing_no_longer_overflows_its_window(monkeypatch) -> None:
    """The reported incident, reproduced through the REAL resolver.

    A session on ``openrouter/meta/muse-spark-1.3`` died with HTTP 400::

        This endpoint's maximum context length is 1048576 tokens. However, you
        requested about 1057079 tokens (102961 of text input, 10400 of tool
        input, 943718 in the output).

    The 943718 was never generated text — it is the ``max_tokens`` the harness
    put on the wire, copied from the listing through ``DiscoveredModel`` and
    ``ModelSpec``. Compaction could not rescue it: the trigger is a fraction of
    the window (~838k at the default 0.8), far above where the 400 lands.

    This pins the whole chain — listing entry, spec, wire body — rather than any
    single link, because each one alone looked reasonable.
    """
    from local_operator.model.discovery import _row_from_openai_entry
    from local_operator.providers.clients import OpenAICompatClient

    row = _row_from_openai_entry(_MUSE_SPARK_LISTING_ENTRY)
    assert row is not None
    # The listing really does advertise this, and the parser really does read it.
    assert row.context_window == 1_048_576

    info = ModelInfo(
        id=row.id,
        name=row.name,
        description="live",
        context_window=row.context_window,
        max_tokens=row.max_tokens,
    )
    spec = build_model_spec("openrouter", "meta/muse-spark-1.3", info=info)

    # ~113k tokens of prompt: the size the real session failed at.
    request = ChatRequest(model=spec, messages=[Message.user("word " * 120_000)])
    body = OpenAICompatClient("https://openrouter.ai/api/v1")._build_body(request)

    prompt_tokens = len("word " * 120_000) // 4
    assert body["max_tokens"] + prompt_tokens < spec.context_window


@pytest.mark.asyncio
async def test_a_fast_mode_refusal_is_narrated_once_and_forwarded_to_the_session() -> None:
    """The stream fn's half of review F1: the provider's own words reach the
    user as a warning, the session bridge is called so the dial comes off,
    and `forget_fast_refusal` re-opens the latch for an explicit `/fast on`."""
    stream = create_stream_fn(MagicMock(), None)
    notices: list[str] = []
    forwarded: list[tuple[str, str]] = []
    stream.set_notice_handler(lambda text, kind: notices.append(f"{kind}:{text}"))
    stream.set_fast_refused_handler(lambda sel, msg: forwarded.append((sel, msg)))

    state = stream._route_state
    assert state.on_fast_refused is not None
    first = await state.record_fast_refusal(
        "anthropic/claude-opus-5", "Usage credits are required for fast mode."
    )
    second = await state.record_fast_refusal("anthropic/claude-opus-5", "again")

    assert (first, second) == (True, False), "latched: the second report is silent"
    assert notices == [
        "warning:fast mode: refused by anthropic/claude-opus-5 — Usage credits are "
        "required for fast mode; switched off, serving at standard speed"
    ]
    assert forwarded == [("anthropic/claude-opus-5", "Usage credits are required for fast mode.")]
    assert state.fast_refused_for("anthropic/claude-opus-5")

    stream.forget_fast_refusal()
    assert not state.fast_refused_for("anthropic/claude-opus-5")

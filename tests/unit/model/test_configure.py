"""Tests for the rewritten model/configure.py.

The legacy langchain plumbing (ChatOpenAI/ChatAnthropic/...) is gone:
``configure_model`` now returns a ``ModelConfiguration`` whose ``.spec`` is
the harness ``ModelSpec`` consumed by wire clients, and ``validate_model``
hits the same endpoints as before through a descriptor table.
"""

import json
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
    DEFAULT_TEMPERATURE,
    build_model_spec,
    calculate_cost,
    configure_model,
    create_stream_fn,
    get_model_info_from_openrouter,
    validate_model,
)
from local_operator.model.registry import ModelInfo
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.failover import FallbackTarget
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
    assert config.temperature == DEFAULT_TEMPERATURE


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
        assert any("trying another anthropic account" in notice for notice in notices)
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

        assert store.is_blocked(first.id, "anthropic")
        assert not store.is_blocked(second.id, "anthropic")
        selected = await store.get_oauth_access("anthropic", session)
        assert selected is not None and selected.credential_id == second.id
        assert stream._route_state.active is None
        assert any("trying another anthropic account" in notice for notice in notices)
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
            {"models": [{"name": "test_model"}]},
            True,
            "http://localhost:11434/api/tags",
            None,
        ),
        ("ollama", "test_model", 404, {}, False, "http://localhost:11434/api/tags", None),
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
            {"Authorization": "Bearer test_key"},
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

    if expected_headers:
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
    mock_response.json.return_value = {"models": [{"name": "test_model"}]}
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

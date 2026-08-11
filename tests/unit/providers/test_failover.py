"""Failover tests: a/b/c rotation, fallback chains, backoff, end-to-end
streaming with fake clients/auth."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import (
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.providers import failover as failover_module
from local_operator.providers.failover import (
    AUTH_RETRY_MAX_ATTEMPTS,
    AuthRetryKeyState,
    FailoverRouteState,
    FallbackTarget,
    ProviderError,
    RetrySettings,
    backoff_delay_ms,
    expand_fallback_candidates,
    expand_fallback_targets,
    is_direct_credential_rotation_error,
    resolve_chain,
    resolve_next_key,
    spec_for_target,
    stream_with_failover,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Tier 1 — a/b/c rotation
# ---------------------------------------------------------------------------


async def test_initial_resolve_returns_first_key() -> None:
    state = AuthRetryKeyState()

    async def resolver(ctx: Any) -> str:
        assert ctx.error is None
        return "key-a"

    assert await resolve_next_key(state, resolver) == "key-a"
    assert state.last_key == "key-a"


async def test_refresh_step_then_sibling_rotation() -> None:
    """401 path: one force-refresh of the same account, then rotate."""
    state = AuthRetryKeyState()
    state.attempted_keys = {"key-a"}
    state.last_key = "key-a"
    sequence: list[str] = ["key-a-refreshed", "key-b"]
    contexts: list[Any] = []

    async def resolver(ctx: Any) -> str | None:
        contexts.append(ctx)
        if not ctx.last_chance:
            return sequence[0] if sequence[0] not in state.attempted_keys else None
        return sequence[1]

    error = ProviderError(401, "unauthorized", auth_error=True)
    first = await resolve_next_key(state, resolver, error)
    assert first == "key-a-refreshed"  # (b) refresh same account
    assert contexts[0].last_chance is False and contexts[0].error is error

    second = await resolve_next_key(state, resolver, error)
    assert second == "key-b"  # (c) sibling
    assert any(ctx.last_chance for ctx in contexts)


async def test_usage_limit_skips_refresh_step() -> None:
    """429/usage-limit and 403 are valid-but-denied: skip (b), go to (c)."""
    for error in (
        ProviderError(429, "rate limit", retryable=True),
        ProviderError(403, "quota exhausted"),
        ProviderError(None, "Usage limit reached"),
    ):
        assert is_direct_credential_rotation_error(error)
        state = AuthRetryKeyState()
        state.attempted_keys = {"key-a"}
        state.last_key = "key-a"

        async def resolver(ctx: Any) -> str | None:
            assert ctx.last_chance, f"refresh step must be skipped for {error}"
            return "key-b"

        assert await resolve_next_key(state, resolver, error) == "key-b"


async def test_attempted_key_dedupe_and_cap() -> None:
    state = AuthRetryKeyState()
    calls = {"n": 0}

    async def resolver(ctx: Any) -> str:
        calls["n"] += 1
        return "key-loop"  # same key forever: dedupe must stop it

    first = await resolve_next_key(state, resolver)
    assert first == "key-loop"
    # b and c steps re-query the resolver, but the attempted-keys set
    # rejects the duplicate, so rotation still terminates.
    assert await resolve_next_key(state, resolver, ProviderError(401, "x")) is None
    assert calls["n"] == 3

    # Hard cap at 64 attempts.
    capped = AuthRetryKeyState()
    capped.attempts = AUTH_RETRY_MAX_ATTEMPTS
    assert await resolve_next_key(capped, resolver) is None


async def test_abort_signal_stops_rotation() -> None:
    from local_operator.harness.types import AbortSignal

    state = AuthRetryKeyState()
    signal = AbortSignal()
    signal.abort("user cancelled")

    async def resolver(ctx: Any) -> str:
        return "key-a"

    assert await resolve_next_key(state, resolver, ProviderError(401, "x"), signal=signal) is None


# ---------------------------------------------------------------------------
# Tier 2 — fallback chains
# ---------------------------------------------------------------------------


CHAINS: dict[str, list[str]] = {
    "default": ["openai/gpt-4o-mini"],
    "google/*": ["openrouter/google/*", "anthropic/claude-3-5-sonnet-latest"],
    "openrouter/anthropic/claude-opus": ["anthropic/claude-3-5-sonnet-latest"],
}


def test_resolve_chain_exact_beats_wildcard() -> None:
    chain = resolve_chain("openrouter/anthropic/claude-opus", CHAINS)
    assert chain == ["anthropic/claude-3-5-sonnet-latest"]


def test_resolve_chain_wildcard_specificity() -> None:
    chain = resolve_chain("google/gemini-2.5-pro", CHAINS)
    assert chain == CHAINS["google/*"]
    # Unmatched selectors fall through to the default chain.
    assert resolve_chain("mistral/mistral-large", CHAINS) == CHAINS["default"]


def test_resolve_chain_none_without_default() -> None:
    assert resolve_chain("mistral/x", {"google/*": ["a/b"]}) is None
    assert resolve_chain("mistral/x", {}) is None


def test_expand_candidates_provider_wildcard_keeps_model_id() -> None:
    candidates = expand_fallback_candidates(
        "google/gemini-2.5-pro", ["openrouter/google/*", "fixed/model"]
    )
    assert candidates == ["openrouter/google/gemini-2.5-pro", "fixed/model"]


def test_expand_candidates_id_prefixed_wildcard_reprefixes_bare_id() -> None:
    """``provider/sub/*`` entries re-prefix the bare id after the provider."""
    candidates = expand_fallback_candidates(
        "openrouter/google/gemini-x", ["google-antigravity/google/*"]
    )
    # Bare id is everything after the FIRST slash of the current selector.
    assert candidates == ["google-antigravity/google/google/gemini-x"]


def test_expand_candidates_never_emits_current_selector() -> None:
    candidates = expand_fallback_candidates("openai/gpt-4o", ["openai/gpt-4o", "anthropic/claude"])
    assert candidates == ["anthropic/claude"]


def test_structured_fallback_entry_carries_effort() -> None:
    targets = expand_fallback_targets(
        "anthropic/claude-opus-5",
        [{"provider": "openai", "model": "gpt-5.3-codex", "effort": "high"}],
    )
    assert targets == [FallbackTarget("openai/gpt-5.3-codex", "high")]
    spec = spec_for_target(
        ModelSpec(
            provider="anthropic",
            model_id="claude-opus-5",
            base_url="https://api.anthropic.com",
        ),
        targets[0],
    )
    assert spec.provider == "openai"
    assert spec.base_url == "https://api.openai.com/v1"
    assert spec.reasoning_effort == "high"


def test_invalid_structured_effort_is_not_silently_dropped() -> None:
    assert (
        expand_fallback_targets(
            "anthropic/claude",
            [{"provider": "openai", "model": "gpt-5", "effort": "turbo"}],
        )
        == []
    )


def test_retry_settings_from_config() -> None:
    settings = RetrySettings.from_settings(
        {
            "retry": {
                "enabled": True,
                "maxRetries": 3,
                "baseDelayMs": 250,
                "modelFallback": False,
                "fallbackChains": {"default": ["a/b"]},
            }
        }
    )
    assert settings.max_retries == 3
    assert settings.base_delay_ms == 250
    assert settings.model_fallback is False
    assert settings.fallback_chains == {"default": ["a/b"]}
    assert settings.usage_aware_fallback is False
    assert settings.usage_reserve_percent == 10
    empty = RetrySettings.from_settings(None)
    assert empty.enabled and empty.max_retries == 10 and empty.base_delay_ms == 500


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def test_backoff_growth_and_cap() -> None:
    rng = random.Random(42)
    for attempt, expected_base in [(1, 500), (2, 1000), (3, 2000), (4, 4000), (5, 8000), (9, 8000)]:
        delay = backoff_delay_ms(500, attempt, rng=random.Random(0))
        # 25% downward jitter: delay ∈ [0.75*base, base].
        assert int(expected_base * 0.75) <= delay <= expected_base
    _ = rng


# ---------------------------------------------------------------------------
# stream_with_failover
# ---------------------------------------------------------------------------


class FakeAuth:
    """AuthStore stand-in: rotates through a fixed key list per provider."""

    def __init__(self, keys: dict[str, list[str]]) -> None:
        self.keys = {provider: list(keys) for provider, keys in keys.items()}
        self.rotations: list[tuple[str, Any]] = []

    async def get_api_key(
        self, provider: str, session_id: str | None = None, **kwargs: Any
    ) -> str | None:
        pool = self.keys.get(provider, [])
        return pool[0] if pool else None

    def rotate_sibling(
        self, provider: str, session_id: str | None, error: Any, api_key: str | None = None
    ) -> bool:
        self.rotations.append((provider, api_key))
        pool = self.keys.get(provider, [])
        if api_key in pool:
            pool.remove(api_key)
        return len(pool) > 0


class ScriptedClient:
    """Yields scripted events or raises on construction/iteration."""

    def __init__(self, events: list[Any] | Exception) -> None:
        self._events = events
        self.calls = 0

    async def stream(
        self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        self.calls += 1
        if isinstance(self._events, Exception):
            raise self._events
        for event in self._events:
            yield event


def _request(provider: str = "openai", model_id: str = "gpt-4o") -> ChatRequest:
    return ChatRequest(model=ModelSpec(provider=provider, model_id=model_id))


async def test_stream_success_passes_events_through() -> None:
    events: list[Any] = [StreamTextDelta(delta="hi"), StreamEndEvent(stop_reason="stop")]
    client = ScriptedClient(events)

    async def client_for(spec: ModelSpec) -> Any:
        return client

    got = [
        event
        async for event in stream_with_failover(
            _request(), FakeAuth({"openai": ["k"]}), None, client_for
        )
    ]
    assert got == events


async def test_stream_auth_error_rotates_credential() -> None:
    """First key 401s; rotation yields a second key and the retry succeeds."""
    failing = ScriptedClient(ProviderError(401, "invalid api key", auth_error=True))
    succeeding = ScriptedClient([StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")])
    used_keys: list[str | None] = []

    async def client_for(spec: ModelSpec) -> Any:
        def wrapper(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            used_keys.append(api_key)
            return (failing if api_key == "bad-key" else succeeding).stream(request, api_key)

        return _FnClient(wrapper)

    auth = FakeAuth({"openai": ["bad-key", "good-key"]})
    got = [
        event
        async for event in stream_with_failover(
            _request(), auth, {"retry": {"baseDelayMs": 1}}, client_for
        )
    ]
    assert used_keys == ["bad-key", "good-key"]
    assert [e for e in got if isinstance(e, StreamTextDelta)]
    assert auth.rotations and auth.rotations[0][1] == "bad-key"


class _FnClient:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    async def stream(
        self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        async for event in self._fn(request, api_key, oauth_access):
            yield event


async def test_stream_fallback_chain_walks_to_next_model() -> None:
    """Primary 500s (non-auth); the fallback selector takes over."""
    specs_seen: list[ModelSpec] = []

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(spec)
        if spec.model_id == "gpt-4o":
            return ScriptedClient(ProviderError(500, "boom", retryable=True))
        return ScriptedClient(
            [StreamTextDelta(delta="fallback"), StreamEndEvent(stop_reason="stop")]
        )

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}}
    auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})
    got = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]
    assert [s.model_id for s in specs_seen] == ["gpt-4o", "claude-x"]
    assert any(isinstance(e, StreamTextDelta) and e.delta == "fallback" for e in got)


async def test_fallback_chain_deduplicates_current_model_and_effort() -> None:
    specs_seen: list[str] = []

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(f"{spec.provider}/{spec.model_id}/{spec.reasoning_effort}")
        if spec.provider == "openai":
            return ScriptedClient(ProviderError(400, "unavailable"))
        return ScriptedClient([StreamEndEvent(stop_reason="stop")])

    request = _request()
    request = request.model_copy(
        update={"model": request.model.model_copy(update={"reasoning_effort": "low"})}
    )
    settings = {
        "retry": {
            "fallbackChains": {
                "default": [
                    {"provider": "openai", "model": "gpt-4o", "effort": "low"},
                    {"provider": "anthropic", "model": "claude-x", "effort": "high"},
                ]
            }
        }
    }

    _ = [
        event
        async for event in stream_with_failover(
            request,
            FakeAuth({"openai": ["k1"], "anthropic": ["k2"]}),
            settings,
            client_for,
        )
    ]
    assert specs_seen == ["openai/gpt-4o/low", "anthropic/claude-x/high"]


async def test_successful_fallback_stays_pinned_for_same_message() -> None:
    """A tool loop must not probe the exhausted primary before every model call."""
    specs_seen: list[str] = []

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(f"{spec.provider}/{spec.model_id}")
        if spec.provider == "openai":
            return ScriptedClient(ProviderError(400, "model unavailable"))
        return ScriptedClient([StreamEndEvent(stop_reason="stop")])

    settings = {
        "retry": {
            "fallbackChains": {
                "default": [{"provider": "anthropic", "model": "claude-opus-5", "effort": "high"}]
            }
        }
    }
    state = FailoverRouteState()
    auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})

    for _ in range(2):
        _ = [
            event
            async for event in stream_with_failover(
                _request(), auth, settings, client_for, route_state=state
            )
        ]

    assert specs_seen == [
        "openai/gpt-4o",
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-5",
    ]
    assert state.active == FallbackTarget("anthropic/claude-opus-5", "high")


@pytest.mark.asyncio
async def test_the_primary_cooldown_is_a_deadline_not_a_sliding_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-entering the SAME fallback route must not push the retry deadline out.

    ``stream_with_failover`` calls ``activate`` on every request that lands on
    the sticky fallback, so arming the cooldown before the "already on this
    route" early return made it slide forward on each use. A user sending
    messages more often than the cooldown never reached ``primary_retry_due``
    and stayed pinned to the fallback for the whole session, even after the
    primary recovered — the opposite of the fixed post-failure probe the class
    docstring promises.

    The clock MUST advance between calls or this test cannot discriminate:
    the arming is ``max(existing, now + cooldown)``, so six re-entries inside
    one millisecond compute the same deadline whether the bug is present or
    not. Verified by mutation — with a frozen clock the buggy order passes.
    """
    now_ms = 1_000_000

    def fake_time() -> float:
        return now_ms / 1000

    # `failover.py` does `import time` and calls `time.time()`, so the patch
    # goes on the module's own `time` binding — swapping a stand-in rather
    # than mutating the real `time` module, which every other test shares.
    monkeypatch.setattr(failover_module, "time", SimpleNamespace(time=fake_time), raising=True)

    target = FallbackTarget("anthropic/claude-opus-5", "high")
    state = FailoverRouteState()

    await state.activate(target, "quota", cooldown_ms=60_000)
    armed = state.primary_retry_at_ms
    assert armed == now_ms + 60_000, "the route CHANGED, so the cooldown must arm"

    # Five more requests on the route that is already active, ten seconds
    # apart — the real shape of a user who keeps typing.
    for _ in range(5):
        now_ms += 10_000
        await state.activate(target, "quota", cooldown_ms=60_000)

    assert state.primary_retry_at_ms == armed, "re-entry re-armed the cooldown"
    # 50s of use later the deadline has arrived, so the primary gets probed.
    assert state.primary_retry_due(now_ms=now_ms + 10_001)


async def test_stream_exhaustion_raises_last_error() -> None:
    async def client_for(spec: ModelSpec) -> Any:
        return ScriptedClient(ProviderError(500, "still down", retryable=True))

    settings = {"retry": {"baseDelayMs": 1, "maxRetries": 1, "fallbackChains": {}}}
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(
            _request(), FakeAuth({"openai": ["k1"]}), settings, client_for
        ):
            pass
    assert excinfo.value.status == 500


async def test_stream_non_retryable_stops_provider_immediately() -> None:
    """A non-retryable 400 abandons the provider after one call and walks
    the fallback chain instead of re-trying the same key."""
    calls = {"n": 0}

    def fail(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        calls["n"] += 1
        raise ProviderError(400, "bad request", retryable=False)

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "anthropic":
            return ScriptedClient([StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")])
        return _FnClient(fail)

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}}
    got = [
        event
        async for event in stream_with_failover(
            _request(), FakeAuth({"openai": ["k"], "anthropic": ["k2"]}), settings, client_for
        )
    ]
    assert calls["n"] == 1  # 400 is not retried on the same provider
    assert any(isinstance(e, StreamTextDelta) and e.delta == "b" for e in got)


async def test_transport_retries_honor_budget_same_key_first() -> None:
    """PR-06: retryable 5xx consumes retry.maxRetries on the SAME key with
    backoff BEFORE any credential rotation."""
    attempts: list[str | None] = []

    def flaky(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts.append(api_key)
        raise ProviderError(503, "flaky", retryable=True)

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(flaky)

    auth = FakeAuth({"openai": ["k1", "k2"]})
    settings = {"retry": {"baseDelayMs": 1, "maxRetries": 2, "fallbackChains": {}}}
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(_request(), auth, settings, client_for):
            pass
    assert excinfo.value.status == 503
    # Two same-key retries before rotation, then the sibling once.
    assert attempts == ["k1", "k1", "k1", "k2", "k2", "k2"]
    assert len(auth.rotations) == 1  # rotation only after the budget is spent


async def test_long_rate_limit_retry_after_rotates_without_sleep(monkeypatch) -> None:
    """A quota-reset header must not freeze an interactive session for minutes."""
    attempts: list[str | None] = []
    sleeps: list[int] = []

    def rate_limited(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts.append(api_key)
        raise ProviderError(429, "quota reset pending", retryable=True, retry_after_ms=600_000)

    async def no_sleep(delay_ms: int, signal: Any) -> None:
        sleeps.append(delay_ms)

    monkeypatch.setattr("local_operator.providers.failover._abortable_sleep", no_sleep)

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(rate_limited)

    auth = FakeAuth({"openai": ["k1", "k2"]})
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(
            _request(),
            auth,
            {"retry": {"baseDelayMs": 1, "maxRetries": 10, "fallbackChains": {}}},
            client_for,
        ):
            pass

    assert excinfo.value.status == 429
    assert attempts == ["k1", "k2"]
    assert sleeps == []
    assert [key for _provider, key in auth.rotations] == ["k1", "k2"]


async def test_short_rate_limit_retries_once_per_credential(monkeypatch) -> None:
    """Brief throttles get one chance, not the generic ten-retry budget."""
    attempts: list[str | None] = []
    sleeps: list[int] = []

    def rate_limited(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts.append(api_key)
        raise ProviderError(429, "brief throttle", retryable=True, retry_after_ms=5)

    async def record_sleep(delay_ms: int, signal: Any) -> None:
        sleeps.append(delay_ms)

    monkeypatch.setattr("local_operator.providers.failover._abortable_sleep", record_sleep)

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(rate_limited)

    auth = FakeAuth({"openai": ["k1", "k2"]})
    with pytest.raises(ProviderError):
        async for _ in stream_with_failover(
            _request(),
            auth,
            {"retry": {"baseDelayMs": 1, "maxRetries": 10, "fallbackChains": {}}},
            client_for,
        ):
            pass

    assert attempts == ["k1", "k1", "k2", "k2"]
    assert len(sleeps) == 2


async def test_403_rotates_once_per_credential_no_double_rotation() -> None:
    """PR-04/26: a 403 must go through resolve_next_key ONLY — one
    rotate_sibling per failed credential, never the old direct double rotate."""
    used: list[str | None] = []

    def denied(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        used.append(api_key)
        raise ProviderError(403, "forbidden", auth_error=True)

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(denied)

    auth = FakeAuth({"openai": ["k1", "k2"]})
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(
            _request(), auth, {"retry": {"baseDelayMs": 1, "fallbackChains": {}}}, client_for
        ):
            pass
    assert excinfo.value.status == 403
    assert used == ["k1", "k2"]
    # Exactly one rotation per credential (old code rotated twice per 403).
    assert [key for _provider, key in auth.rotations] == ["k1", "k2"]


async def test_ordinary_401_single_refresh_plus_single_switch() -> None:
    """PR-07: an ordinary 401 gets exactly one refresh + one sibling switch
    (legacyAuthSwitchUsed), even when more siblings exist."""
    used: list[str | None] = []

    def unauthorized(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        used.append(api_key)
        raise ProviderError(401, "unauthorized", auth_error=True)

    class SwitchOnceAuth(FakeAuth):
        async def get_api_key(
            self, provider: str, session_id: str | None = None, **kwargs: Any
        ) -> str | None:
            # Force-refresh returns the SAME (failed) bearer; only the
            # last_chance leg yields the next sibling.
            if kwargs.get("force_refresh"):
                return self.keys[provider][0]
            return await super().get_api_key(provider, session_id, **kwargs)

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(unauthorized)

    auth = SwitchOnceAuth({"openai": ["k1", "k2", "k3"]})
    with pytest.raises(ProviderError):
        async for _ in stream_with_failover(
            _request(), auth, {"retry": {"baseDelayMs": 1, "fallbackChains": {}}}, client_for
        ):
            pass
    assert used == ["k1", "k2"]  # k3 never touched: switch spent


async def test_abort_interrupts_backoff_sleep() -> None:
    """PR-19: the backoff sleep races the abort signal and loses."""
    from local_operator.harness.types import AbortSignal

    def flaky(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        raise ProviderError(503, "flaky", retryable=True)

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(flaky)

    signal = AbortSignal()
    settings = {"retry": {"baseDelayMs": 60_000, "fallbackChains": {}}}

    async def abort_soon() -> None:
        await asyncio.sleep(0.05)
        signal.abort("user cancelled")

    task = asyncio.create_task(abort_soon())
    with pytest.raises(ProviderError):
        async for _ in stream_with_failover(
            _request(), FakeAuth({"openai": ["k"]}), settings, client_for, signal=signal
        ):
            pass
    await task

"""Failover tests: a/b/c rotation, fallback chains, backoff, end-to-end
streaming with fake clients/auth."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ChatRequest, ModelSpec, StreamEndEvent, StreamTextDelta
from local_operator.model.configure import build_model_spec
from local_operator.providers.failover import (
    AUTH_RETRY_MAX_ATTEMPTS,
    AuthRetryKeyState,
    ProviderError,
    RetrySettings,
    backoff_delay_ms,
    classify_provider_error,
    expand_fallback_candidates,
    is_auth_error,
    is_direct_credential_rotation_error,
    is_transient_error,
    is_usage_limit_error,
    resolve_chain,
    resolve_next_key,
    spec_for_selector,
    stream_with_failover,
    wrap_transport_error,
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


class TestChainsAreNormalizedAtTheBoundary:
    """A config that is wrong in a way YAML cannot catch must not kill turns.

    ``fallbackChains`` declares ``Sequence[str]``, and the parser used to check
    only that the outer value was a mapping. A config written with structured
    entries therefore parsed, resolved, and died in
    ``expand_fallback_candidates`` with ``'dict' object has no attribute
    'endswith'`` — on EVERY turn, because the selector list is built eagerly at
    the top of ``stream_with_failover`` before any provider is called.
    """

    def test_the_mapping_form_is_accepted_not_merely_survived(self) -> None:
        """The exact config that reproduced the crash, from a real machine."""
        settings = RetrySettings.from_settings(
            {
                "retry": {
                    "fallbackChains": {
                        "anthropic/claude-opus-5": [
                            {"effort": "low", "model": "claude-opus-5", "provider": "anthropic"},
                            {"effort": "high", "model": "gpt-5.4", "provider": "openai"},
                        ]
                    }
                }
            }
        )
        assert settings.fallback_chains == {
            "anthropic/claude-opus-5": ["anthropic/claude-opus-5", "openai/gpt-5.4"]
        }
        # And it reaches the wire as the user plainly intended.
        chain = resolve_chain("anthropic/claude-opus-5", settings.fallback_chains)
        assert chain is not None
        assert expand_fallback_candidates("anthropic/claude-opus-5", chain) == ["openai/gpt-5.4"]

    def test_an_unsupported_key_is_reported_rather_than_swallowed(self, caplog) -> None:
        """``effort`` on a chain entry is still not honoured, now for a reason
        rather than for a missing field: ``ModelSpec`` HAS a
        ``reasoning_effort``, but a chain is a flat list of selectors deduped by
        selector and expanded from wildcards that name no model until failure
        time, so a level here could be neither validated nor attached to one
        attempt. A chain that quietly drops half of what the user wrote is the
        next bug report, so it is named in the log — see
        ``_normalize_chain_entry`` for the full argument, and
        ``spec_for_selector`` for what DOES carry across a hop."""
        with caplog.at_level("WARNING"):
            RetrySettings.from_settings(
                {
                    "retry": {
                        "fallbackChains": {"a/b": [{"provider": "x", "model": "y", "effort": 1}]}
                    }
                }
            )
        assert any("effort" in record.getMessage() for record in caplog.records)

    def test_a_string_chain_is_untouched(self) -> None:
        """The declared form keeps working, byte for byte."""
        settings = RetrySettings.from_settings(
            {"retry": {"fallbackChains": {"default": ["a/b", "c/*"]}}}
        )
        assert settings.fallback_chains == {"default": ["a/b", "c/*"]}

    def test_junk_degrades_instead_of_raising(self) -> None:
        """A preference about what to do when a model fails must not be more
        disruptive than the failure it was written to handle."""
        settings = RetrySettings.from_settings(
            {
                "retry": {
                    "fallbackChains": {
                        "a/b": [None, 42, {"provider": "x"}, {"model": "y"}, "", "  "],
                        "c/d": "not-a-list",
                        "e/f": ["ok/one"],
                    }
                }
            }
        )
        # Every unreadable entry dropped, the readable chain kept, nothing raised.
        assert settings.fallback_chains == {"e/f": ["ok/one"]}

    def test_a_wholly_bad_chain_cannot_reach_the_wire(self) -> None:
        """The regression that mattered: whatever the config says, the selectors
        handed to ``expand_fallback_candidates`` are strings."""
        settings = RetrySettings.from_settings(
            {"retry": {"fallbackChains": {"a/b": [{"nonsense": True}]}}}
        )
        chain = resolve_chain("a/b", settings.fallback_chains)
        assert chain is None  # dropped entirely rather than half-formed
        for entries in settings.fallback_chains.values():
            assert all(isinstance(entry, str) for entry in entries)


class TestEffortDoesNotOutliveItsModelAcrossAHop:
    """A fallback swaps the model; the level chosen for the old one may not fit.

    ``spec_for_selector`` carries every other knob over unchanged, which is
    right for context windows and cache flags and wrong for exactly this one:
    the valid values are a property of the MODEL, so a carried-over level is
    either rejected outright or discarded while the band goes on claiming it.
    """

    def test_a_level_both_models_accept_survives_the_hop(self) -> None:
        """A user who dropped to `low` for cost still gets cheap thinking on
        whichever model answers."""
        base = build_model_spec("anthropic", "claude-opus-5").model_copy(
            update={"reasoning_effort": "low"}
        )
        swapped = spec_for_selector(base, "openai/gpt-5.4")
        assert swapped.reasoning_effort == "low"
        assert swapped.reasoning_efforts == ("none", "low", "medium", "high", "xhigh")

    def test_a_level_the_fallback_lacks_becomes_that_models_default(self) -> None:
        """`xhigh` reaching a 4.5-generation model is a 400 on the request that
        was supposed to rescue the turn."""
        base = build_model_spec("anthropic", "claude-opus-5").model_copy(
            update={"reasoning_effort": "xhigh"}
        )
        swapped = spec_for_selector(base, "anthropic/claude-opus-4-5-20251101")
        assert swapped.reasoning_effort == "high"

    def test_falling_back_to_a_model_without_the_knob_drops_it(self) -> None:
        base = build_model_spec("anthropic", "claude-opus-5").model_copy(
            update={"reasoning_effort": "max"}
        )
        swapped = spec_for_selector(base, "openai/gpt-4.1")
        assert swapped.reasoning_effort is None
        assert swapped.reasoning_efforts == ()


# ---------------------------------------------------------------------------
# Error kinds, and who owns the reported-error slot
# ---------------------------------------------------------------------------


class TestProviderErrorKinds:
    """The reported defect: a quota exhaustion surfaced as ``✕ HTTP 404:``.

    A status is not a diagnosis and an empty message is not an error, so both
    the classification and the rendered line are pinned here. Retryability is
    asserted alongside the kind on purpose — the two decisions are the same
    decision, and a kind that reads "transient" while ``is_transient_error``
    says otherwise would send the driver and the user different stories.
    """

    @pytest.mark.parametrize(
        ("error", "kind", "transient"),
        [
            # quota: the reported case, plus the shapes providers actually use.
            (ProviderError(429, "Too many requests", retryable=True), "quota", False),
            (ProviderError(402, "Insufficient credits"), "quota", False),
            (ProviderError(403, "Quota exceeded for model"), "quota", False),
            (ProviderError(None, "Usage limit reached"), "quota", False),
            # auth: a 401 is always the bearer, even when its body says
            # "insufficient permissions" — which the quota matcher would claim.
            (ProviderError(401, "insufficient permissions", auth_error=True), "auth", False),
            (ProviderError(403, "Forbidden"), "auth", False),
            # timeout ahead of transient: both retry, one names the cause.
            (ProviderError(408, "Request Timeout", retryable=True), "timeout", True),
            (ProviderError(504, "Gateway Timeout", retryable=True), "timeout", True),
            (ProviderError(None, "ReadTimeout: stream stalled", retryable=True), "timeout", True),
            # transient
            (ProviderError(500, "Internal Server Error", retryable=True), "transient", True),
            (ProviderError(529, "Overloaded", retryable=True), "transient", True),
            (
                ProviderError(None, "ConnectError: connection reset", retryable=True),
                "transient",
                True,
            ),
            # a request the provider READ and refused
            (ProviderError(400, "`temperature` is deprecated"), "request", False),
            (ProviderError(404, "model not found"), "request", False),
            (ProviderError(422, "input too long"), "request", False),
        ],
    )
    def test_each_kind_is_named_and_its_retryability_agrees(
        self, error: ProviderError, kind: str, transient: bool
    ) -> None:
        assert error.kind == kind
        assert classify_provider_error(error) == kind
        assert is_transient_error(error) is transient

    def test_the_three_kinds_that_must_never_be_retried(self) -> None:
        """Retrying these burns quota and delays the honest answer: a quota
        reset needs a wait, no retry mints a valid bearer, and the same bytes
        get the same 400."""
        for error in (
            ProviderError(429, "rate limited", retryable=True),
            ProviderError(401, "invalid api key", auth_error=True),
            ProviderError(400, "bad request"),
        ):
            assert is_transient_error(error) is False

    def test_a_bare_exception_gets_the_same_vocabulary(self) -> None:
        """Not every path wraps: the embedding backend and the TUI hold raw
        exceptions and need the same answer."""
        assert classify_provider_error(TimeoutError()) == "timeout"
        assert classify_provider_error(ValueError("malformed payload")) == "unknown"
        assert is_transient_error(ValueError("malformed payload")) is False

    def test_no_5xx_is_ever_read_as_a_quota_exhaustion(self) -> None:
        """Found by enumerating every ``HTTPStatus`` phrase against the quota
        markers: an empty-bodied 507 was classified ``quota``, because the
        empty-message floor fills the text from the status phrase and
        "Insufficient Storage" contains ``insufficient`` — the harness's own
        words classifying the harness's own error, and a retryable server fault
        turned into "your quota is gone".

        The general rule this pins: a 5xx is the server failing, so a 5xx that
        MENTIONS a limit is still the server failing and still worth retrying.
        """
        for status in HTTPStatus:
            if int(status) < 500:
                continue
            error = ProviderError(int(status), "", retryable=True)
            assert error.kind in ("transient", "timeout"), (int(status), status.phrase, error.kind)
        assert ProviderError(507, "").kind == "transient"
        assert ProviderError(503, "rate limit on the upstream pool", retryable=True).kind == (
            "transient"
        )
        # And the bound does not cost the real quota shapes, which are all 4xx
        # or carry no status at all.
        assert ProviderError(403, "Quota exceeded for model").kind == "quota"
        assert ProviderError(None, "Usage limit reached").kind == "quota"

    def test_a_403_about_permissions_is_auth_not_quota(self) -> None:
        """Google's real 403 PERMISSION_DENIED text is "Request had insufficient
        authentication scopes." — which the combined marker set read as a quota
        exhaustion and rendered `rate limit or quota exceeded (HTTP 403)`, sending
        the user to wait out a problem only a re-login or scope grant clears.
        This commit's own failure mode, inverted, so it is pinned both ways."""
        for message in (
            "Request had insufficient authentication scopes.",
            "insufficient permissions for this model",
            "The caller does not have permission; usage of this API is restricted",
        ):
            error = ProviderError(403, message)
            assert error.kind == "auth", message
            assert "quota" not in str(error)
        # A 403 that really IS exhaustion still says so — google and openrouter
        # both report quota this way, which is why the branch exists at all.
        for message in ("Quota exceeded for model", "RESOURCE_EXHAUSTED", "rate limit reached"):
            assert ProviderError(403, message).kind == "quota", message
        # Rotation is unchanged either way: a denied 403 must still rotate.
        assert is_auth_error(ProviderError(403, "insufficient authentication scopes"))
        assert is_direct_credential_rotation_error(ProviderError(403, "whatever"))

    def test_the_harness_never_diagnoses_its_own_bug_as_a_quota_problem(self) -> None:
        """`wrap_transport_error` used to hand its own synthesized text to the
        text classifier, so a `KeyError('usage')` raised while a client parsed a
        usage block rendered as `rate limit or quota exceeded`. Nothing in an
        exception's text is evidence about the user's quota."""
        for exc in (
            KeyError("usage"),
            ValueError("insufficient data in chunk"),
            RuntimeError("rate limit bookkeeping failed"),
        ):
            wrapped = wrap_transport_error(exc)
            assert wrapped.kind == "transient", (exc, wrapped.kind)
            assert wrapped.retryable is True
        # The exception CLASS is still legitimate evidence of a timeout.
        assert wrap_transport_error(httpx.ConnectTimeout("")).kind == "timeout"
        assert wrap_transport_error(TimeoutError()).kind == "timeout"

    def test_402_is_named_as_quota_without_claiming_its_credential_is_worth_keeping(
        self,
    ) -> None:
        """A spent balance IS a quota problem to the user and a refresh cannot fix
        it, so it reads as quota and skips the refresh step. But it must not be a
        ``usage_limit`` to ``AuthStore.rotate_sibling``, which PRESERVES the
        sticky credential on the reasoning that the window reopens — a balance of
        zero does not reopen in sixty seconds."""
        error = ProviderError(402, "Your credit balance is too low")
        assert error.kind == "quota"
        assert is_direct_credential_rotation_error(error) is True
        assert is_usage_limit_error(error) is False
        assert is_transient_error(error) is False

    @pytest.mark.parametrize(
        ("error", "rendered"),
        [
            (
                ProviderError(
                    429, "Limit: 200000 tokens/min.", retryable=True, retry_after_ms=41600
                ),
                "rate limit or quota exceeded (HTTP 429, retry in 42s): Limit: 200000 tokens/min.",
            ),
            (
                ProviderError(503, "upstream reset", retryable=True),
                "transient provider error (HTTP 503): upstream reset",
            ),
            (
                ProviderError(408, "request timed out", retryable=True),
                "provider timeout (HTTP 408): request timed out",
            ),
            (
                ProviderError(401, "invalid x-api-key", auth_error=True),
                "authentication failed (HTTP 401): invalid x-api-key",
            ),
            (
                ProviderError(400, "`temperature` is deprecated"),
                "invalid request (HTTP 400): `temperature` is deprecated",
            ),
            (
                ProviderError(None, "ConnectError: refused", retryable=True),
                "transient provider error: ConnectError: refused",
            ),
        ],
    )
    def test_the_rendered_line_leads_with_the_kind(
        self, error: ProviderError, rendered: str
    ) -> None:
        """This string IS the error frame — ``RenderedStreamError`` tells the
        loop that ``str()`` is the whole diagnosis, and the TUI prints it
        verbatim into a ``NoticeBlock``. The wait rides in the parenthetical so
        it survives a long provider message being wrapped."""
        assert str(error) == rendered

    def test_an_abort_is_not_dressed_up_as_a_diagnosis(self) -> None:
        """The user pressed the key; prefixing their own reason with a label
        states the obvious twice."""
        assert str(ProviderError(None, "user cancelled", kind="aborted")) == "user cancelled"

    @pytest.mark.parametrize("status", [None, 404, 429, 599])
    def test_no_provider_error_can_print_nothing(self, status: int | None) -> None:
        """The other half of the reported frame. ``__str__`` would happily render
        ``HTTP 404:`` forever, and no downstream care can recover text that was
        never captured — so the floor is set at construction."""
        for blank in ("", "   ", "\n"):
            error = ProviderError(status, blank)
            assert error.message.strip()
            assert str(error).rstrip().endswith(error.message)
            assert not str(error).rstrip().endswith(":")

    def test_the_provider_own_words_are_never_editorialised(self) -> None:
        """``message`` stays the provider's text: the classifiers read it, and
        it is the half that says WHICH limit and WHEN it clears. Composition
        happens only in ``__str__``."""
        error = ProviderError(429, "  Quota exceeded. Retry after 41.6s.  ", retryable=True)
        assert error.message == "Quota exceeded. Retry after 41.6s."


class TestTheReportedErrorIsTheMostDiagnosticOne:
    """The reported ``✕ HTTP 404:`` frame was a real 429 on the requested model,
    overwritten by a bare 404 from a fallback selector the account cannot serve.

    ``last_error`` was last-wins, so a longer failover walk reported the LEAST
    informative failure of the set. The primary selector owns the slot: the user
    asked for that model, so its failure is the news of the turn.
    """

    def _settings(self, chain: list[str]) -> dict[str, Any]:
        return {"retry": {"baseDelayMs": 1, "maxRetries": 1, "fallbackChains": {"default": chain}}}

    async def _run(self, errors: dict[str, ProviderError]) -> ProviderError:
        async def client_for(spec: ModelSpec) -> Any:
            return ScriptedClient(errors[spec.model_id])

        auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})
        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(
                _request(), auth, self._settings(["anthropic/claude-x"]), client_for
            ):
                pass
        return excinfo.value

    async def test_a_fallbacks_bare_404_cannot_mask_the_primarys_quota_error(self) -> None:
        """The reproduction, at the driver level."""
        error = await self._run(
            {
                "gpt-4o": ProviderError(
                    429, "You exceeded your current quota.", retryable=True, retry_after_ms=42_000
                ),
                "claude-x": ProviderError(404, ""),
            }
        )
        assert error.status == 429
        assert str(error) == (
            "rate limit or quota exceeded (HTTP 429, retry in 42s): "
            "You exceeded your current quota."
        )

    async def test_the_primary_wins_even_when_it_is_the_duller_failure(self) -> None:
        """Not "highest rank overall": the requested model's 500 is the answer to
        "why did my turn fail", and the fallback's 401 is a broken chain entry."""
        error = await self._run(
            {
                "gpt-4o": ProviderError(500, "boom", retryable=True),
                "claude-x": ProviderError(401, "no key for anthropic", auth_error=True),
            }
        )
        assert error.status == 500

    async def test_a_fallback_failure_is_logged_rather_than_lost(self, caplog) -> None:
        """It can no longer reach the frame, so a chain entry the account cannot
        serve would otherwise fail invisibly forever."""
        with caplog.at_level("WARNING", logger="local_operator.providers.failover"):
            await self._run(
                {
                    "gpt-4o": ProviderError(429, "quota", retryable=True),
                    "claude-x": ProviderError(404, ""),
                }
            )
        assert any(
            "fallback selector failed" in record.getMessage() and "404" in record.getMessage()
            for record in caplog.records
        )

    async def test_the_primary_owns_the_slot_even_when_it_never_reached_a_client(self) -> None:
        """Named for what it actually pins, which is not what it first claimed.

        A provider with no key configured breaks BEFORE calling anyone, and that
        "No API key configured" still records against the primary — so it beats
        the fallback's 500, and the frame names the missing key rather than a
        model the user did not ask for. That is right: the missing key is the
        thing to fix.

        It also means the primary records on every path it can take, which makes
        "a fallback wins the slot" unreachable in practice and the terminal
        ``Failover exhausted`` line defensive rather than live. Recorded here so
        the next reader does not go looking for a test of either.
        """

        async def client_for(spec: ModelSpec) -> Any:
            return ScriptedClient(ProviderError(500, "fallback is down", retryable=True))

        auth = FakeAuth({"anthropic": ["k2"]})
        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(
                _request(), auth, self._settings(["anthropic/claude-x"]), client_for
            ):
                pass
        assert "No API key configured" in excinfo.value.message
        assert "fallback is down" not in excinfo.value.message
        # And it renders without a `provider error:` stutter in front of a
        # sentence that already names itself.
        assert str(excinfo.value) == "No API key configured for provider 'openai'"


class TestTransientFailuresAreRetriedOnEveryCallPath:
    """ "Transient errors are automatically retried on any invocation."

    A turn, a subagent, an aside and the one-shot errands all reach the provider
    through ``stream_with_failover``, so the first test covers the shared floor.
    The one-shot path had a REAL gap: it collects the whole stream before
    returning a string, but the driver forwarded events as they arrived, so a
    stall part-way through was permanent — and the compaction summary is one of
    those calls, which means the context it was meant to shrink kept growing.
    """

    async def _drive(
        self, script: list[Any], *, replayable: bool = False, max_retries: int = 3
    ) -> tuple[list[Any], int]:
        """Run one request whose client raises ``script`` in order, then
        succeeds. Returns the events seen and the number of attempts made."""
        attempts = {"n": 0}

        def flaky(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            index = attempts["n"]
            attempts["n"] += 1

            async def gen() -> AsyncIterator[Any]:
                if index < len(script):
                    step = script[index]
                    if isinstance(step, tuple):
                        # (partial events, then the failure) — a mid-stream death.
                        for event in step[0]:
                            yield event
                        raise step[1]
                    raise step
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(flaky)

        request = _request().model_copy(update={"replayable": replayable})
        settings = {"retry": {"baseDelayMs": 1, "maxRetries": max_retries, "fallbackChains": {}}}
        events = [
            event
            async for event in stream_with_failover(
                request, FakeAuth({"openai": ["k"]}), settings, client_for
            )
        ]
        return events, attempts["n"]

    async def test_a_turn_retries_a_transient_failure_before_it_streams(self) -> None:
        events, attempts = await self._drive(
            [ProviderError(503, "overloaded", retryable=True), httpx.ConnectError("refused")]
        )
        assert attempts == 3
        assert [e.delta for e in events if isinstance(e, StreamTextDelta)] == ["done"]

    async def test_a_one_shot_errand_retries_a_stall_MID_stream(self) -> None:
        """The gap this closes. ``replayable`` buffers instead of forwarding, so
        a half-finished attempt can be discarded whole; the caller sees exactly
        one clean stream and no duplicated text."""
        events, attempts = await self._drive(
            [
                ([StreamTextDelta(delta="half a summ")], httpx.ReadTimeout("stream stalled")),
                ([StreamTextDelta(delta="half a summ")], httpx.ReadTimeout("stream stalled")),
            ],
            replayable=True,
        )
        assert attempts == 3
        assert [e.delta for e in events if isinstance(e, StreamTextDelta)] == ["done"]

    async def test_a_turn_does_NOT_replay_output_the_user_already_read(self) -> None:
        """The default, and it must stay the default: those deltas are already in
        the transcript, and re-streaming them would write the answer twice. It
        arrives NAMED all the same — re-raised raw, a mid-stream
        ``httpx.ReadTimeout("")`` painted an empty frame and a traceback."""
        with pytest.raises(ProviderError) as excinfo:
            await self._drive(
                [([StreamTextDelta(delta="partial")], httpx.ReadTimeout(""))],
                replayable=False,
            )
        assert excinfo.value.kind == "timeout"
        assert str(excinfo.value) == "provider timeout: ReadTimeout"

    async def test_the_one_shot_call_the_session_makes_is_marked_replayable(self) -> None:
        """Wiring, not policy: ``_one_shot_complete`` is the compaction summary
        and the auto-naming call, and the flag is what buys them the retry
        above. Asserted on the request the session actually builds so the two
        cannot drift apart."""
        from local_operator.session.session import Session

        seen: list[ChatRequest] = []

        async def stream_fn(request: ChatRequest, signal: Any) -> AsyncIterator[Any]:
            seen.append(request)
            yield StreamTextDelta(delta="summary")

        # `__new__` plus the two attributes `_one_shot_complete` reads. Brittle by
        # nature — it is coupled to the private attrs that method happens to touch
        # — but the alternative is standing a whole Session up to assert one flag.
        session = Session.__new__(Session)
        spec = ModelSpec(provider="openai", model_id="gpt-4o")
        session._model = spec  # type: ignore[attr-defined]
        session._stream_fn = stream_fn  # type: ignore[attr-defined]
        assert await session._one_shot_complete("sys", "prompt") == "summary"
        assert seen[0].replayable is True

    @pytest.mark.parametrize(
        "error",
        [
            ProviderError(400, "`temperature` is deprecated"),
            ProviderError(404, "no such model"),
        ],
    )
    async def test_a_refused_request_is_never_retried(self, error: ProviderError) -> None:
        """One call, one answer. The same bytes get the same refusal, so a retry
        only delays it."""
        attempts = {"n": 0}

        def refuse(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts["n"] += 1
            raise error

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(refuse)

        with pytest.raises(ProviderError):
            async for _ in stream_with_failover(
                _request(),
                FakeAuth({"openai": ["k1", "k2"]}),
                {"retry": {"baseDelayMs": 1, "maxRetries": 5, "fallbackChains": {}}},
                client_for,
            ):
                pass
        assert attempts["n"] == 1, "a refused request must not be re-sent, nor rotated onto"

    async def test_an_auth_failure_rotates_but_never_re_sends_the_same_key(self) -> None:
        """Auth is the one failure a retry cannot fix and a DIFFERENT credential
        can, so the budget must not be spent on the rejected one."""
        used: list[str | None] = []

        def denied(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            used.append(api_key)
            raise ProviderError(401, "invalid api key", auth_error=True)

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(denied)

        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(
                _request(),
                FakeAuth({"openai": ["k1", "k2"]}),
                {"retry": {"baseDelayMs": 1, "maxRetries": 5, "fallbackChains": {}}},
                client_for,
            ):
                pass
        assert excinfo.value.kind == "auth"
        # Each key tried once. Five same-key retries would have been five
        # guaranteed 401s before the sibling that might have worked.
        assert sorted(set(used)) == used == ["k1", "k2"]

    async def test_a_long_quota_reset_surfaces_instead_of_sleeping_through_it(self) -> None:
        """A quota exhaustion with a long reset is not transient. The user gets
        the named error and the wait, immediately, instead of a frozen UI."""
        attempts = {"n": 0}

        def limited(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts["n"] += 1
            raise ProviderError(429, "quota exhausted", retryable=True, retry_after_ms=3_600_000)

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(limited)

        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(
                _request(),
                FakeAuth({"openai": ["k1"]}),
                {"retry": {"baseDelayMs": 1, "maxRetries": 10, "fallbackChains": {}}},
                client_for,
            ):
                pass
        assert attempts["n"] == 1
        assert "retry in 1h" in str(excinfo.value)


class TestTransportErrorsKeepTheirIdentity:
    """``ProviderError(None, str(exc))`` printed NOTHING for the whole httpx
    family that raises with no arguments, which is most of it."""

    @pytest.mark.parametrize(
        ("exc", "kind"),
        [
            (httpx.ConnectTimeout(""), "timeout"),
            (httpx.ReadTimeout(""), "timeout"),
            (httpx.RemoteProtocolError(""), "transient"),
            (httpx.ConnectError("[Errno 61] Connection refused"), "transient"),
        ],
    )
    async def test_an_argumentless_transport_error_still_says_what_it_was(
        self, exc: Exception, kind: str
    ) -> None:
        def boom(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            raise exc

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(boom)

        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(
                _request(),
                FakeAuth({"openai": ["k"]}),
                {"retry": {"baseDelayMs": 1, "maxRetries": 1, "fallbackChains": {}}},
                client_for,
            ):
                pass
        assert type(exc).__name__ in excinfo.value.message
        assert excinfo.value.kind == kind
        assert excinfo.value.retryable is True

"""Failover tests: a/b/c rotation, fallback chains, backoff, end-to-end
streaming with fake clients/auth."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamStartEvent,
    StreamTextDelta,
    StreamUsageEvent,
    Usage,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers import failover as failover_module
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.failover import (
    AUTH_RETRY_MAX_ATTEMPTS,
    BACKOFF_CAP_MS,
    CHAIN_EFFORT_LADDER,
    CONNECTIVITY_BACKOFF_CAP_MS,
    CONNECTIVITY_MAX_RETRIES,
    SUPPORTED_EFFORTS,
    AuthRetryKeyState,
    FailoverRouteState,
    FallbackTarget,
    ProviderError,
    RetrySettings,
    _request_has_rotated,
    backoff_delay_ms,
    classify_provider_error,
    connectivity_backoff_delay_ms,
    expand_fallback_candidates,
    expand_fallback_targets,
    is_auth_error,
    is_connectivity_loss,
    is_direct_credential_rotation_error,
    is_fast_mode_refusal,
    is_fast_mode_refusal_for,
    is_image_rejection,
    is_transient_error,
    is_usage_limit_error,
    resolve_chain,
    resolve_chain_key,
    resolve_next_key,
    spec_for_selector,
    spec_for_target,
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


def test_resolve_chain_key_reports_which_key_won() -> None:
    """The KEY, by the same specificity that picks the entries.

    `/failovers` shows the user which of several plausible keys is in force, and
    it must be this function's answer rather than a second matching loop that
    can disagree with routing.
    """
    assert resolve_chain_key("openrouter/anthropic/claude-opus", CHAINS) == (
        "openrouter/anthropic/claude-opus"
    )
    assert resolve_chain_key("google/gemini-2.5-pro", CHAINS) == "google/*"
    # No specific key matches, so the default chain is what actually serves.
    assert resolve_chain_key("mistral/mistral-large", CHAINS) == "default"


def test_resolve_chain_key_longest_wildcard_wins() -> None:
    chains = {"a/*": ["x/y"], "a/b/*": ["z/w"]}
    assert resolve_chain_key("a/b/c", chains) == "a/b/*"


def test_resolve_chain_key_none_without_default() -> None:
    # Same "nothing applies" verdict resolve_chain returns, so the two cannot
    # disagree about whether a cascade exists at all.
    assert resolve_chain_key("mistral/x", {"google/*": ["a/b"]}) is None
    assert resolve_chain_key("mistral/x", {}) is None


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


def test_failover_hop_does_not_reimpose_the_primarys_sampling_on_the_target() -> None:
    """Trap: ``spec_for_target`` carried ``base.temperature``/``top_p`` onto the
    target unconditionally, so a hop from a 0.2-configured primary re-imposed
    on the target exactly the value its own vendor documents against — a live
    wire bug on any target that still sends the pair.

    A sampling value belongs to the MODEL, so the target's own policy wins; the
    session's value rides only into a knob the target leaves open.
    """
    base = ModelSpec(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        temperature=0.2,
        top_p=0.9,
    )

    # A target that documents its own values keeps them across the hop.
    two_x = spec_for_target(base, FallbackTarget("google/gemini-2.5-pro", None))
    assert two_x.temperature == 1.0
    assert two_x.top_p == 0.95

    # A target whose policy is to omit is not handed the primary's 0.2.
    three_x = spec_for_target(base, FallbackTarget("google/gemini-3.8-flash", None))
    assert three_x.temperature is None
    assert three_x.top_p is None

    # Per-knob, not per-pair: Qwen seeds both, but a family that seeded only one
    # must not have the other clobbered.
    qwen = spec_for_target(base, FallbackTarget("alibaba/qwen3-coder-plus", None))
    assert qwen.temperature == 0.7
    assert qwen.top_p == 0.8


def test_invalid_structured_effort_is_not_silently_dropped(caplog) -> None:
    """The name is the assertion: dropped, and NOT silently.

    The entry really is discarded - an effort outside the vocabulary is not a
    routing decision this can honour. But a typo deleting a whole fallback hop
    with nothing in the log is how an operator ends up with no failover during
    an outage and no way to trace it to their YAML (review round 29). This
    previously asserted only the `== []`, which the silence also satisfied.
    """
    chain = [{"provider": "openai", "model": "gpt-5", "effort": "turbo"}]
    with caplog.at_level("WARNING"):
        settings = RetrySettings.from_settings(
            {"retry": {"fallbackChains": {"anthropic/claude": chain}}}
        )

    # Expand what NORMALIZATION produced, not the raw config: production always
    # walks the normalized chain, and `_fallback_target` honours a raw mapping
    # on its own, so expanding `chain` here would bypass the function under test.
    assert (
        expand_fallback_targets("anthropic/claude", settings.fallback_chains["anthropic/claude"])
        == []
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "turbo" in messages and "openai/gpt-5" in messages
    # The message has to be actionable: name the vocabulary, not just the typo,
    # and name it as the LADDER it is - `sorted()` puts `max` between `low` and
    # `medium`, which reads as noise to anyone who knows the scale from
    # `/effort` (design round 28).
    # Anchored on the prefix, not a bare substring: a SUPERSET satisfies
    # `"minimal, low, ..." in messages`, so re-admitting `none` to the
    # advertised list would pass while making the sentence contradict itself
    # (`'none' is not accepted ...; expected one of none, ...`) - review round 32
    # built exactly that mutant and it survived.
    assert "expected one of minimal, low, medium, high, xhigh, max" in messages
    # And the claim must be true: `none` IS an effort - `/effort` offers it -
    # it is just not accepted in a chain hop. Copy that overreaches sends the
    # reader to check the wrong thing.
    assert "is not an effort" not in messages
    assert "is not accepted in a fallback chain hop" in messages
    # The chain itself survives: one bad hop is not a reason to lose the rest.
    assert settings.fallback_chains["anthropic/claude"]


def test_the_advertised_ladder_is_exactly_what_is_accepted() -> None:
    """The message's list and the gate that refuses a value must be one set.

    `CHAIN_EFFORT_LADDER` filters `EFFORT_ORDER` by `SUPPORTED_EFFORTS`, so the
    drift is asymmetric: a rung added to `EFFORT_ORDER` alone is filtered out
    and can never be advertised while refused (safe), but a rung added to
    `SUPPORTED_EFFORTS` alone is ACCEPTED and silently missing from the list -
    something the `sorted(SUPPORTED_EFFORTS)` this replaced could not do
    (review round 32). Cheap to pin, and the failure it prevents is a user
    being told a value they just used successfully is not one of the options.
    """
    assert set(CHAIN_EFFORT_LADDER) == SUPPORTED_EFFORTS
    # And it is a ladder, stated LITERALLY. Re-deriving it as
    # `[e for e in EFFORT_ORDER if e in SUPPORTED_EFFORTS]` re-runs the
    # implementation's own comprehension against the same source, so reversing
    # `EFFORT_ORDER` reversed the ladder and the assertion still passed - it
    # could only ever restate the code (review round 33). Written out, it is a
    # decision about what the user reads, and a reordering upstream has to come
    # here and be agreed to.
    assert list(CHAIN_EFFORT_LADDER) == ["minimal", "low", "medium", "high", "xhigh", "max"]


def test_effort_survives_an_unknown_sibling_key(caplog) -> None:
    """`effort` is honoured, so an unrelated `note:` must not cost the user
    their routing - and the warning must not list `effort` among the keys it
    is ignoring while claiming in the same breath that effort is supported.

    Both halves are asserted. The routing is the damage; the self-contradicting
    sentence is what would send the reader looking in the wrong place for it.
    """
    chain = [{"provider": "openai", "model": "gpt-5", "effort": "low", "note": "cheap hop"}]
    with caplog.at_level("WARNING"):
        settings = RetrySettings.from_settings(
            {"retry": {"fallbackChains": {"anthropic/claude": chain}}}
        )

    # The NORMALIZED chain, for the reason above: asserting against `chain`
    # passes even when `_normalize_chain_entry` flattens the effort away, which
    # is the exact bug this test is named for (review round 30).
    normalized = settings.fallback_chains["anthropic/claude"]
    assert [
        (t.selector, t.effort) for t in expand_fallback_targets("anthropic/claude", normalized)
    ] == [("openai/gpt-5", "low")]

    ignored = [
        r.getMessage() for r in caplog.records if "ignoring unsupported key" in r.getMessage()
    ]
    assert len(ignored) == 1 and "note" in ignored[0]
    # The sentence claims effort is honoured; it must not also list it as ignored.
    key_list = ignored[0].split("key(s) ", 1)[1].split(" on entry", 1)[0]
    assert "effort" not in key_list


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
        self,
        provider: str,
        session_id: str | None,
        error: Any,
        api_key: str | None = None,
        *,
        model_id: str = "",
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


async def test_failover_stamps_serving_spec_on_usage() -> None:
    """After a primary→xai walk the usage event names xai, not the primary.

    The analytics recorder reads serving identity off Usage. Without this stamp
    it would keep attributing the successful Grok call to anthropic — the
    missing-provider bug. A primary success is covered by the same helper
    (it stamps whatever ``current_request.model`` is) so this cannot
    mis-attribute a call that never left the primary.
    """
    usage = Usage(input_tokens=10, output_tokens=4)

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "anthropic":
            return ScriptedClient(ProviderError(500, "boom", retryable=True))
        return ScriptedClient(
            [
                StreamTextDelta(delta="ok"),
                StreamUsageEvent(usage=usage),
                StreamEndEvent(stop_reason="stop", usage=usage),
            ]
        )

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "fallbackChains": {"default": ["xai/grok-4.6"]},
        }
    }
    auth = FakeAuth({"anthropic": ["k1"], "xai": ["k2"]})
    got = [
        event
        async for event in stream_with_failover(
            _request("anthropic", "claude-opus-4-8"), auth, settings, client_for
        )
    ]
    usage_events = [e for e in got if isinstance(e, StreamUsageEvent)]
    end_events = [e for e in got if isinstance(e, StreamEndEvent)]
    assert usage_events and usage_events[0].usage.provider == "xai"
    assert usage_events[0].usage.model_id == "grok-4.6"
    assert end_events and end_events[0].usage is not None
    assert end_events[0].usage.provider == "xai"
    assert end_events[0].usage.model_id == "grok-4.6"


async def test_primary_success_stamps_primary_spec() -> None:
    """A call that never failed over must still name the session primary."""
    usage = Usage(input_tokens=3, output_tokens=1)

    async def client_for(spec: ModelSpec) -> Any:
        return ScriptedClient(
            [StreamUsageEvent(usage=usage), StreamEndEvent(stop_reason="stop", usage=usage)]
        )

    got = [
        event
        async for event in stream_with_failover(
            _request("anthropic", "claude-opus-4-8"),
            FakeAuth({"anthropic": ["k"]}),
            None,
            client_for,
        )
    ]
    stamped = next(e for e in got if isinstance(e, StreamUsageEvent)).usage
    assert stamped.provider == "anthropic"
    assert stamped.model_id == "claude-opus-4-8"


async def test_fallback_chain_deduplicates_current_model_and_effort() -> None:
    specs_seen: list[str] = []

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(f"{spec.provider}/{spec.model_id}/{spec.reasoning_effort}")
        if spec.provider == "openai":
            # A generic "primary is down, walk to the fallback" trigger. Uses an
            # UNKNOWN-kind failure (status=None) rather than a request-shape 400:
            # a primary kind=="request" 400 now ABORTS the turn (the storm guard),
            # so a 400 here would no longer walk. status=None takes the identical
            # non-retryable break->walk path without being classified "request",
            # which keeps this test about fallback dedup, not the abort policy.
            return ScriptedClient(ProviderError(None, "unavailable"))
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
            # Unknown-kind (status=None) "primary down" trigger, not a 400: a
            # primary request-shape 400 now aborts the turn rather than walking.
            # See test_primary_request_400_aborts_without_walking_chain.
            return ScriptedClient(ProviderError(None, "model unavailable"))
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


async def test_exhausted_fallback_is_benched_for_the_next_walk(monkeypatch) -> None:
    """Message 2 must not re-pay message 1's failed fallback.

    The reported annoyance: with the chain's head providers down, every user
    message replayed the whole waterfall — one "provider failure" notice and
    one serial timeout per dead target — before landing back on the one
    provider that had been serving all along. A fallback that exhausted its
    provider is benched for ``FALLBACK_FAILURE_COOLDOWN_MS``, so the next
    walk goes straight from the failed primary to the working tail.
    """
    now_ms = 1_000_000
    monkeypatch.setattr(
        failover_module, "time", SimpleNamespace(time=lambda: now_ms / 1000), raising=True
    )
    specs_seen: list[str] = []

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(f"{spec.provider}/{spec.model_id}")
        if spec.provider in ("openai", "anthropic"):
            # Unknown-kind (status=None) "target down" trigger, not a 400: a
            # primary request-shape 400 now aborts instead of walking. status=None
            # keeps the identical break->walk path for this bench-mechanics test.
            return ScriptedClient(ProviderError(None, "model unavailable"))
        return ScriptedClient([StreamEndEvent(stop_reason="stop")])

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "fallbackChains": {"default": ["anthropic/claude-x", "groq/llama-x"]},
        }
    }
    state = FailoverRouteState()
    auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"], "groq": ["k3"]})

    async def run() -> None:
        _ = [
            event
            async for event in stream_with_failover(
                _request(), auth, settings, client_for, route_state=state
            )
        ]

    await run()
    assert specs_seen == ["openai/gpt-4o", "anthropic/claude-x", "groq/llama-x"]
    # The dead fallback is benched; the one that served is not.
    assert not state.target_retry_due(FallbackTarget("anthropic/claude-x"), now_ms=now_ms)
    assert state.target_retry_due(FallbackTarget("groq/llama-x"), now_ms=now_ms)

    # A preflight that believed the primary recovered clears the pin — which
    # must NOT clear the bench, or the next walk replays the waterfall.
    state.clear()
    specs_seen.clear()
    now_ms += 30_000  # well inside the cooldown
    await run()
    assert specs_seen == [
        "openai/gpt-4o",
        "groq/llama-x",
    ], "the benched fallback was re-walked inside its cooldown"

    # Past the cooldown the bench expires on its own and the target is asked
    # again — the bench is a delay, not a removal.
    state.clear()
    specs_seen.clear()
    now_ms += failover_module.FALLBACK_FAILURE_COOLDOWN_MS
    await run()
    assert specs_seen == ["openai/gpt-4o", "anthropic/claude-x", "groq/llama-x"]


async def test_bench_never_strands_a_turn_when_every_fallback_is_benched(monkeypatch) -> None:
    """An all-benched chain must still serve via the loop-back sweep.

    The bench is advisory: it shapes the first pass, and the loop-back sweep
    re-walks whatever the first pass never asked. A turn may be slower here,
    but it must never die reporting exhaustion while a benched target would
    have served.
    """
    now_ms = 1_000_000
    monkeypatch.setattr(
        failover_module, "time", SimpleNamespace(time=lambda: now_ms / 1000), raising=True
    )
    specs_seen: list[str] = []

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(f"{spec.provider}/{spec.model_id}")
        if spec.provider == "openai":
            # Unknown-kind (status=None) "primary down" trigger, not a 400: a
            # primary request-shape 400 now aborts instead of walking. status=None
            # keeps the identical break->walk path for this all-benched test.
            return ScriptedClient(ProviderError(None, "model unavailable"))
        return ScriptedClient([StreamEndEvent(stop_reason="stop")])

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["groq/llama-x"]}}}
    state = FailoverRouteState()
    state.mark_target_failed(
        FallbackTarget("groq/llama-x"),
        cooldown_ms=failover_module.FALLBACK_FAILURE_COOLDOWN_MS,
        now_ms=now_ms,
    )
    got = [
        event
        async for event in stream_with_failover(
            _request(),
            FakeAuth({"openai": ["k1"], "groq": ["k3"]}),
            settings,
            client_for,
            route_state=state,
        )
    ]
    assert specs_seen == ["openai/gpt-4o", "groq/llama-x"]
    assert any(isinstance(e, StreamEndEvent) for e in got)
    # Serving cleared the bench mark.
    assert state.target_retry_due(FallbackTarget("groq/llama-x"), now_ms=now_ms)


async def test_pinned_route_is_exempt_from_its_own_bench(monkeypatch) -> None:
    """The route the session is running on is never skipped by the bench.

    A pinned fallback that hiccups gets benched like any other; the pin
    exemption keeps the session's own route in the walk so the trim that
    starts the walk from the pin cannot produce an empty first pass.
    """
    now_ms = 1_000_000
    monkeypatch.setattr(
        failover_module, "time", SimpleNamespace(time=lambda: now_ms / 1000), raising=True
    )
    specs_seen: list[str] = []

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(f"{spec.provider}/{spec.model_id}")
        return ScriptedClient([StreamEndEvent(stop_reason="stop")])

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["groq/llama-x"]}}}
    state = FailoverRouteState()
    target = FallbackTarget("groq/llama-x")
    await state.activate(target, "provider failure", cooldown_ms=60_000)
    state.mark_target_failed(
        target,
        cooldown_ms=failover_module.FALLBACK_FAILURE_COOLDOWN_MS,
        now_ms=now_ms,
    )
    _ = [
        event
        async for event in stream_with_failover(
            _request(),
            FakeAuth({"openai": ["k1"], "groq": ["k3"]}),
            settings,
            client_for,
            route_state=state,
        )
    ]
    # The pin trim starts the walk at groq, and the bench must not skip it.
    assert specs_seen == ["groq/llama-x"]


async def test_bench_deadline_extends_but_never_shrinks(monkeypatch) -> None:
    """A fresh short failure cannot shorten a longer advertised reset."""
    state = FailoverRouteState()
    target = FallbackTarget("groq/llama-x")
    state.mark_target_failed(target, cooldown_ms=600_000, now_ms=1_000_000)
    state.mark_target_failed(target, cooldown_ms=60_000, now_ms=1_000_000)
    assert not state.target_retry_due(target, now_ms=1_000_000 + 599_999)
    assert state.target_retry_due(target, now_ms=1_000_000 + 600_000)


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


async def test_primary_request_400_aborts_without_walking_chain() -> None:
    """A PRIMARY request-shape 400 aborts the turn instead of walking the chain.

    ``kind=="request"`` is a 4xx the provider READ and refused: the same bytes
    fail identically on every other provider, so walking the fallback chain is
    a storm — one malformed Anthropic 400 marched through z.ai/kimi/etc, each
    400ing on the same messages. When the PRIMARY (the user's own model)
    refuses, the turn now raises its 400 immediately and never touches the
    fallback. (Was: this test asserted the anthropic fallback served "b" — the
    behaviour the storm-stopper corrects.)"""
    calls = {"n": 0}
    anthropic = ScriptedClient([StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")])

    def fail(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        calls["n"] += 1
        raise ProviderError(400, "bad request", retryable=False)

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "anthropic":
            return anthropic
        return _FnClient(fail)

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}}
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(
            _request(), FakeAuth({"openai": ["k"], "anthropic": ["k2"]}), settings, client_for
        ):
            pass
    # The primary's own 400 surfaces, not a fallback's error.
    assert excinfo.value.status == 400
    assert excinfo.value.kind == "request"
    assert calls["n"] == 1  # 400 is not retried on the same provider
    assert anthropic.calls == 0  # the fallback chain was never walked


async def test_fallback_request_400_still_walks() -> None:
    """A request-shape 400 on a FALLBACK target still walks to the next target.

    The abort is gated on ``is_primary``: a ``kind=="request"`` failure on a
    fallback can mean that one fallback rejected a field the primary accepted
    (an effort rung its ladder lacks) — a per-target defect the walk should
    route around, NOT a request-shape defect. So a fallback 400 keeps walking
    to a later target that can serve."""
    served = ScriptedClient([StreamTextDelta(delta="c"), StreamEndEvent(stop_reason="stop")])

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "openai":  # the primary — hand off with a provider fault
            return ScriptedClient(ProviderError(500, "boom", retryable=True))
        if spec.provider == "anthropic":  # first fallback — request-shape 400
            return ScriptedClient(ProviderError(400, "bad request", retryable=False))
        return served  # zai — the later target that serves

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 1,
            "fallbackChains": {"default": ["anthropic/claude-x", "zai/glm-x"]},
        }
    }
    auth = FakeAuth({"openai": ["k"], "anthropic": ["k2"], "zai": ["k3"]})
    got = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]
    # The walk continued PAST the fallback 400 to the later target.
    assert any(isinstance(e, StreamTextDelta) and e.delta == "c" for e in got)
    assert served.calls == 1


async def test_primary_image_rejection_400_aborts_turn() -> None:
    """A PRIMARY image-rejection 400 aborts too, and keeps its image marker.

    Image rejections are ``kind=="request"`` (a plain 400 on the provider's own
    wording). Their recovery is the session's ``_degrade_if_image_rejected``,
    fired from the turn end event INDEPENDENT of this walk, plus a resend — not
    the fallback chain (a poisoned image 400s everywhere too). So aborting the
    walk costs the image path nothing, and the raised error must still satisfy
    ``is_image_rejection`` so the session's degrade fires."""
    anthropic = ScriptedClient([StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")])

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "anthropic":
            return anthropic
        return ScriptedClient(ProviderError(400, "could not process image", retryable=False))

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}}
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(
            _request(), FakeAuth({"openai": ["k"], "anthropic": ["k2"]}), settings, client_for
        ):
            pass
    assert excinfo.value.status == 400
    # The error object still carries the image marker, so the session degrade fires.
    assert is_image_rejection(excinfo.value)
    assert anthropic.calls == 0  # aborted, not walked


async def test_primary_unknown_400_style_not_aborted_if_not_request_kind() -> None:
    """An ``unknown``-kind failure still walks — the abort is request-only.

    An empty-bodied gateway failure classifies as ``unknown`` (not ``request``)
    and may be a transient edge a fallback survives, so the guard is gated
    strictly on ``kind=="request"``. This pins that a primary ``unknown``
    failure does NOT abort but walks to a serving fallback (guards against
    over-broad aborting)."""
    served = ScriptedClient([StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")])

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "anthropic":
            return served
        # status=None, non-retryable, no auth/timeout markers -> kind "unknown".
        return ScriptedClient(ProviderError(None, "gateway boom", retryable=False))

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 1,
            "fallbackChains": {"default": ["anthropic/claude-x"]},
        }
    }
    auth = FakeAuth({"openai": ["k"], "anthropic": ["k2"]})
    got = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]
    assert any(isinstance(e, StreamTextDelta) and e.delta == "ok" for e in got)
    assert served.calls == 1


async def test_opaque_aggregator_400_fails_over_instead_of_aborting() -> None:
    """The session e13d092c093c failure, end to end: an aggregator answers the
    PRIMARY with HTTP 400 / "Provider returned error" / ``metadata.raw``
    "ERROR" — no actionable diagnostics. Classified ``request`` it aborted the
    turn before rotation or fallback could serve; classified transient it must
    consume a same-credential attempt, rotate to the sibling key, and only then
    walk the chain — all under the driver's existing server-fault budget."""
    from local_operator.providers.clients import raise_for_status

    def _opaque_400() -> ProviderError:
        # Built through the real wire mapper so the test exercises the exact
        # body OpenRouter sent, not a hand-stamped classification.
        try:
            raise_for_status(
                httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": "Provider returned error",
                            "code": 400,
                            "metadata": {"raw": "ERROR", "provider_name": "Stealth"},
                        }
                    },
                )
            )
        except ProviderError as exc:
            return exc
        raise AssertionError("raise_for_status must raise")

    attempts: list[str | None] = []
    served = ScriptedClient([StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")])

    def fail_like_the_session(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts.append(api_key)
        raise _opaque_400()

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "anthropic":
            return served
        return _FnClient(fail_like_the_session)

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 2,
            "fallbackChains": {"default": ["anthropic/claude-x"]},
        }
    }
    auth = FakeAuth({"openai": ["k1", "k2"], "anthropic": ["k3"]})
    got = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]
    assert any(isinstance(e, StreamTextDelta) and e.delta == "ok" for e in got)
    assert served.calls == 1
    # The opaque 400 was retried on the SAME credential first (transient
    # semantics), then rotated to the sibling before the chain took over.
    assert len(attempts) >= 2
    assert attempts[0] == "k1"
    assert any(k == "k2" for k in attempts[1:])
    assert auth.rotations, "the sibling rotation must actually have run"


async def test_opaque_aggregator_400_in_band_fails_over_instead_of_aborting() -> None:
    """The same failure on the aggregator's OTHER relay channel.

    When the gateway has already committed HTTP 200 it relays the identical
    opaque body as an in-band error chunk instead of a status line. That path is
    mapped by ``_compat_stream_error``, which classified from ``code`` alone and
    so still aborted the turn with zero fallback calls after the pre-stream path
    was fixed. Driven end to end through a REAL ``OpenAICompatClient`` over a
    mock transport, so the chunk travels the actual SSE parser rather than a
    hand-stamped ProviderError: the cascade must rotate the sibling key and let
    the chain serve the turn."""
    body = json.dumps(
        {
            "id": "gen-1",
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {"raw": "ERROR", "provider_name": "Stealth"},
            },
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
        }
    )
    primary_requests: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        primary_requests.append(request)
        # The error is the FIRST chunk, so nothing has been forwarded yet and
        # failover is genuinely still possible at that point.
        return httpx.Response(
            200,
            content=f"data: {body}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    primary = OpenAICompatClient(
        "https://api.test.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    served = ScriptedClient([StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")])

    async def client_for(spec: ModelSpec) -> Any:
        return served if spec.provider == "anthropic" else primary

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 2,
            "fallbackChains": {"default": ["anthropic/claude-x"]},
        }
    }
    auth = FakeAuth({"openai": ["k1", "k2"], "anthropic": ["k3"]})
    got = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]
    assert any(isinstance(e, StreamTextDelta) and e.delta == "ok" for e in got)
    assert served.calls == 1, "the fallback chain must have served the turn"
    assert auth.rotations, "the sibling rotation must actually have run"
    # Bounded, not unbounded: the driver's server-fault budget still caps the
    # in-band path, so the primary is not retried forever before the chain runs.
    assert len(primary_requests) <= 12


async def test_relayed_upstream_404_recovers_instead_of_killing_the_turn() -> None:
    """The session 2be018a98088 failure, end to end.

    An aggregator answered the PRIMARY with HTTP 404 and the RELAYED upstream
    words "The requested model was not found." for a model id that worked six
    times in the surrounding 75 seconds. Classified ``request`` on the strength
    of that wording, the turn was aborted instantly with no retry and no
    failover; classified as the relayed upstream blip it is, the same transient
    machinery that serves an opaque sentinel must serve it and the turn must
    still produce an answer.

    Built through the real wire mapper so the test pins the behaviour against
    the exact bytes OpenRouter sent, not a hand-stamped classification."""
    from local_operator.providers.clients import raise_for_status

    def _relayed_404() -> ProviderError:
        try:
            raise_for_status(
                httpx.Response(
                    404,
                    json={
                        "error": {
                            "message": "Provider returned error",
                            "code": 404,
                            "metadata": {
                                "raw": json.dumps(
                                    {
                                        "error": {
                                            "message": "The requested model was not found.",
                                            "type": "invalid_request_error",
                                        }
                                    }
                                ),
                                "provider_name": "Meta",
                                "is_byok": False,
                            },
                        }
                    },
                )
            )
        except ProviderError as exc:
            return exc
        raise AssertionError("raise_for_status must raise")

    # Precondition: the frame the user reads names the upstream host rather
    # than the ambiguous generic "Provider".
    assert "Meta returned error" in str(_relayed_404())

    attempts: list[str | None] = []
    served = ScriptedClient([StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")])

    def fail_like_the_session(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts.append(api_key)
        raise _relayed_404()

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "anthropic":
            return served
        return _FnClient(fail_like_the_session)

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 2,
            "fallbackChains": {"default": ["anthropic/claude-x"]},
        }
    }
    auth = FakeAuth({"openai": ["k1", "k2"], "anthropic": ["k3"]})
    got = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]
    assert any(isinstance(e, StreamTextDelta) and e.delta == "ok" for e in got)
    assert served.calls == 1, "the turn must be served instead of dying on the 404"
    assert len(attempts) >= 2, "the 404 must be retried, not aborted on first sight"
    assert auth.rotations, "the sibling rotation must actually have run"


async def test_flat_unknown_model_404_still_aborts_immediately() -> None:
    """The other side of the line, end to end.

    A model id the AGGREGATOR itself rejects is flat (no ``metadata.raw``) and
    actionable, so it must still abort the turn at once. Without this the
    widening would trade a dead turn for minutes of pointless backoff on a
    request no retry can fix."""
    from local_operator.providers.clients import raise_for_status

    attempts: list[str | None] = []

    def refuse(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts.append(api_key)
        raise_for_status(
            httpx.Response(
                400,
                json={"error": {"message": "openai/nope-9.9 is not a valid model ID", "code": 400}},
            )
        )
        raise AssertionError("unreachable")

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(refuse)

    auth = FakeAuth({"openai": ["k1", "k2"]})
    settings = {"retry": {"baseDelayMs": 1, "maxRetries": 2, "fallbackChains": {}}}
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(_request(), auth, settings, client_for):
            pass
    assert excinfo.value.kind == "request"
    assert "is not a valid model ID" in str(excinfo.value)
    assert len(attempts) == 1, "a bad model id must surface at once, not after a cascade"


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
    # Two same-key retries before rotation, then the sibling gets the same --
    # the sequence this test has always asserted, and the one `main` produces.
    # The restore-once path hands back the REMAINDER of the configured budget
    # rather than a fresh one, so it adds nothing here.
    assert attempts == ["k1", "k1", "k1", "k2", "k2", "k2"]
    # Every credential in the pool is asked before the turn is declared dead: a
    # 5xx is a PROVIDER fault, so the next ACCOUNT is the thing most likely to
    # succeed. Capping rotation at one switch is what stranded two healthy
    # Anthropic accounts during a 529 storm.
    assert [key for _provider, key in auth.rotations] == ["k1", "k2"]


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


async def test_short_rate_limit_retries_twice_per_credential(monkeypatch) -> None:
    """Brief throttles get two chances, not the generic ten-retry budget.

    Two because the throttles that advertise a short delay are burst and
    concurrency limits, which clear in seconds: the first retry often lands
    inside the same collision that produced the 429, and giving up after it
    turned a recoverable blip into a credential rotation plus a provider
    fallback — the early "provider failure" churn a live session reads as
    flakiness. Still not the full budget: each attempt sleeps the ADVERTISED
    delay, so the cap bounds how long one credential can hold the screen.
    """
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

    assert attempts == ["k1", "k1", "k1", "k2", "k2", "k2"]
    # Four in-place retry sleeps, plus ONE loop-back pause: a short throttle
    # arms the end-of-cascade sweep, which sleeps the advertised delay and
    # re-resolves — finding nothing (both keys rotated out), hence no seventh
    # attempt. The attempts list above is the budget contract; the fifth sleep
    # is the sweep's, not a fifth retry.
    assert len(sleeps) == 5
    assert sleeps[-1] == 5


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
        # The structured form is PRESERVED, not flattened: the `effort` key is
        # what makes the entry mean something, so collapsing it to a selector
        # would silently answer a different question. Main's per-attempt effort
        # (quota-aware failover) is what this test predates and now pins.
        assert settings.fallback_chains == {
            "anthropic/claude-opus-5": [
                {"effort": "low", "model": "claude-opus-5", "provider": "anthropic"},
                {"effort": "high", "model": "gpt-5.4", "provider": "openai"},
            ]
        }
        # And it reaches the wire as the user plainly intended, effort and all.
        chain = resolve_chain("anthropic/claude-opus-5", settings.fallback_chains)
        assert chain is not None
        # The first entry IS the current model at a LOWER effort, so it is a
        # real fallback route (retry cheaper) rather than a self-reference, and
        # the expansion keeps it. The second is the provider hop.
        targets = expand_fallback_targets("anthropic/claude-opus-5", chain)
        assert [(t.selector, t.effort) for t in targets] == [
            ("anthropic/claude-opus-5", "low"),
            ("openai/gpt-5.4", "high"),
        ]

    def test_an_unsupported_key_is_reported_rather_than_swallowed(self, caplog) -> None:
        """``effort`` is now honoured on a chain entry - quota-aware failover
        made the chain's shape (selector, effort) real, which retired the
        reason this key used to be dropped. What must still be named is a key
        that ISN'T understood: a chain that quietly drops half of what the
        user wrote is the next bug report."""
        with caplog.at_level("WARNING"):
            RetrySettings.from_settings(
                {
                    "retry": {
                        "fallbackChains": {
                            "a/b": [{"provider": "x", "model": "y", "urgency": "low"}]
                        }
                    }
                }
            )
        assert any("urgency" in record.getMessage() for record in caplog.records)

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

    def test_an_unwrapped_exception_never_reads_as_the_users_quota(self) -> None:
        """The harness must not diagnose its OWN bugs as an exhausted account.

        ``wrap_transport_error`` was taught to state its kind outright for this,
        but ``classify_provider_error`` is the entry point every caller holding
        an unwrapped exception uses, and it still ran the exception's text
        through the quota markers: a client's ``KeyError('usage')`` came back
        ``quota``, which renders as ``rate limit or quota exceeded`` and sends
        the user to check a billing page over a defect in this repo.

        A class name is legitimate evidence of a timeout, so that stays.
        """
        for exc in (
            KeyError("usage"),
            ValueError("insufficient data in chunk"),
            RuntimeError("quota bookkeeping failed"),
            AttributeError("'NoneType' object has no attribute 'rate limit'"),
        ):
            assert classify_provider_error(exc) == "unknown", exc
            assert is_transient_error(exc) is False
        # The class name still carries the one thing it honestly knows.
        assert classify_provider_error(httpx.ReadTimeout("")) == "timeout"
        # And a real ProviderError keeps the kind it was classified with.
        assert classify_provider_error(ProviderError(429, "rate limited")) == "quota"

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

    async def test_an_openrouter_error_chunk_after_partial_output_arrives_named(self) -> None:
        """The reported incident, end to end: visible deltas, then an in-band
        error chunk on an HTTP 200 stream.

        The real OpenAI-compatible client raises the named ``ProviderError``
        from the error chunk; ``stream_with_failover`` has already forwarded
        output, so it must re-raise it NAMED — no retry (which would duplicate
        the visible deltas) and no silent wordless end. This is the exact
        combination the client-level and driver-level tests each cover only
        half of.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            lines = [
                "data: " + json.dumps({"choices": [{"delta": {"content": "Gate"}, "index": 0}]}),
                "data: "
                + json.dumps(
                    {
                        "id": "gen-9",
                        "error": {
                            "code": 429,
                            "message": "Rate limit exceeded",
                            "metadata": {"error_type": "rate_limit_exceeded"},
                        },
                        "choices": [
                            {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
                        ],
                    }
                ),
                "data: [DONE]",
            ]
            return httpx.Response(
                200,
                content="\n\n".join(lines).encode() + b"\n\n",
                headers={"content-type": "text/event-stream"},
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def client_for(spec: ModelSpec) -> Any:
            return OpenAICompatClient("https://api.test.example/v1", http_client=http)

        seen: list[Any] = []
        # Closed in ``finally`` — the shape every other client-owning test in
        # this file already uses. With the close after the assertions, a
        # failing assertion left the client open and the reader chasing an
        # "unclosed client" warning attached to whichever later test happened
        # to trip the collector, instead of the assertion that actually failed.
        try:
            with pytest.raises(ProviderError) as excinfo:
                async for event in stream_with_failover(
                    _request(),
                    FakeAuth({"openai": ["k"]}),
                    {"retry": {"baseDelayMs": 1}},
                    client_for,
                ):
                    seen.append(event)
            error = excinfo.value
            assert error.status == 429
            assert error.kind == "quota"
            assert "Rate limit exceeded" in str(error)
            # The visible delta was forwarded exactly once; nothing was replayed.
            assert [e for e in seen if isinstance(e, StreamTextDelta)] == [
                StreamTextDelta(delta="Gate")
            ]
        finally:
            await http.aclose()

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
        #
        # ``api_key`` is optional on the wire (an OAuth provider sends none), so
        # the real keys are separated out first: a ``None`` reaching the client
        # under a keyed provider would be its own bug, and it would otherwise
        # crash ``sorted`` here instead of naming itself.
        keys = [key for key in used if key is not None]
        assert keys == used, "every rotation must carry a real credential"
        assert sorted(set(keys)) == keys == ["k1", "k2"]

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


def test_an_image_rejection_is_recognised_in_both_the_raised_and_rendered_forms() -> None:
    """The predicate has two callers holding two different things.

    The client layer catches a ``ProviderError``; ``AgentEndEvent.error``
    carries the already-rendered string the UI shows. Both must agree, because
    what they drive is a STICKY degrade — a session that stops sending images
    for the rest of its life.
    """
    raised = ProviderError(400, "Could not process image")
    assert is_image_rejection(raised)
    # The rendered form is what the session actually receives.
    assert str(raised) == "invalid request (HTTP 400): Could not process image"
    assert is_image_rejection(str(raised))


@pytest.mark.parametrize(
    "message",
    [
        "Could not process image",
        "Image could not be processed",
        "messages.31.content.4.image.source.base64.data: Image does not match "
        "the provided media type image/jpeg",
        "Unsupported image format",
        # The DIMENSION refusals. Both were missed by the original marker list,
        # and the many-image one is the wording that wedged a real session on
        # 2026-08-18: the composer had attached a 2206x266 screenshot verbatim,
        # and the request crossed twenty images a hundred turns later.
        "messages.0.content.2.image.source.base64.data: At least one of the image "
        "dimensions exceed max allowed size for many-image requests: 2000 pixels",
        "messages.5.content.1.image.source.base64.data: At least one of the image "
        "dimensions exceed max allowed size: 8000 pixels",
    ],
)
def test_the_wordings_providers_actually_use_are_all_recognised(message: str) -> None:
    """Sampled from the reports this degrade exists for, not invented:
    anthropics/claude-code#12009, #13594, #31142, #50708, #12351, #39185."""
    assert is_image_rejection(ProviderError(400, message))


def test_the_many_image_refusal_degrades_in_the_rendered_form_too() -> None:
    """The form the session actually receives, for the case that wedged a
    session.

    ``AgentEndEvent.error`` carries the rendered string, and that is the value
    ``_degrade_if_image_rejected`` is handed. A marker that matched only the
    raised form would leave the session bricked exactly as before, so this pins
    the path rather than the predicate's convenient half.

    The many-image limit is worth its own test because of how it ARRIVES: no
    image changed and no request was malformed, the conversation simply grew
    past twenty images and a block that had been accepted for a hundred turns
    started being refused forever.
    """
    refusal = ProviderError(
        400,
        "messages.0.content.2.image.source.base64.data: At least one of the image "
        "dimensions exceed max allowed size for many-image requests: 2000 pixels",
    )
    assert str(refusal).startswith("invalid request (HTTP 400)")
    assert is_image_rejection(str(refusal))


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (503, "Could not process image"),  # weather, not a bad block
        (500, "Could not process image"),
        (400, "max_tokens: must be less than the model's context window"),
        (400, "credit balance is too low"),
        (429, "rate limit exceeded"),
    ],
)
def test_only_a_client_side_image_refusal_degrades_the_session(status: int, message: str) -> None:
    """The cost of a false positive is silent and permanent — every image
    stripped from the rest of the session — so a 5xx that merely mentions an
    image must not trip it, and neither may any other 4xx."""
    assert not is_image_rejection(ProviderError(status, message))


def test_the_rendered_form_is_gated_on_the_kind_not_on_a_guessed_status() -> None:
    """The string carries no parseable status, so the gate is the kind's own
    label. A transient error whose body mentions an image renders as
    ``transient provider error (...)`` and must not match."""
    transient = ProviderError(503, "Could not process image")
    assert str(transient).startswith("transient provider error")
    assert not is_image_rejection(str(transient))


class TestAnIsolatedRequestCannotDegradeTheTurnBesideIt:
    """``ChatRequest.isolated``: decoration runs concurrently with a user turn.

    Auto-naming stopped waiting for the turn to settle, so a title call is now a
    SECOND in-flight request on the same session. Each test here pins one route
    by which that call's failure used to be able to reach the turn, and each
    fails if its denial is removed, because every one of these behaviours is on
    by default for an ordinary request: the first five come from
    ``stream_with_failover``'s ``isolated`` branch, the sixth from the read-only
    credential resolve it asks for.
    """

    @staticmethod
    def _isolated(provider: str = "openai", model_id: str = "gpt-4o") -> ChatRequest:
        return ChatRequest(model=ModelSpec(provider=provider, model_id=model_id), isolated=True)

    async def test_it_never_walks_the_model_fallback_chain(self) -> None:
        """A fallback hop would put the title on a model the turn is not using —
        and, with a route state, would drag the turn there too."""
        specs_seen: list[str] = []

        async def client_for(spec: ModelSpec) -> Any:
            specs_seen.append(spec.model_id)
            return ScriptedClient(ProviderError(500, "boom", retryable=True))

        settings = {
            "retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}
        }
        auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})
        with pytest.raises(ProviderError):
            _ = [
                event
                async for event in stream_with_failover(
                    self._isolated(), auth, settings, client_for
                )
            ]
        assert specs_seen == ["gpt-4o"], "an isolated call walked onto a fallback model"

    async def test_it_never_rotates_the_sessions_credential(self) -> None:
        """``rotate_sibling`` moves the session's STICKY credential, so an auth
        failure on a title would re-point the account the turn transacts on."""
        auth = FakeAuth({"openai": ["bad-key", "good-key"]})
        used_keys: list[str | None] = []

        async def client_for(spec: ModelSpec) -> Any:
            def wrapper(
                request: ChatRequest, api_key: str | None, oauth_access: Any = None
            ) -> AsyncIterator[Any]:
                used_keys.append(api_key)
                return ScriptedClient(
                    ProviderError(401, "invalid api key", auth_error=True)
                ).stream(request, api_key)

            return _FnClient(wrapper)

        with pytest.raises(ProviderError):
            _ = [
                event
                async for event in stream_with_failover(
                    self._isolated(), auth, {"retry": {"baseDelayMs": 1}}, client_for
                )
            ]
        assert used_keys == ["bad-key"], "an isolated call retried on a second credential"
        assert auth.rotations == [], "an isolated call rotated the session's credential"

    async def test_it_neither_pins_nor_clears_the_sticky_route(self) -> None:
        """The route state is session-wide: a title pinning a fallback would move
        the TURN's model, and a title CLEARING a pin would send the turn back to
        a primary the turn already knows is down."""
        pinned = FallbackTarget("anthropic/claude-opus-5", "high")
        state = FailoverRouteState()
        await state.activate(pinned, "an earlier turn failed over")

        async def client_for(spec: ModelSpec) -> Any:
            return ScriptedClient([StreamEndEvent(stop_reason="stop")])

        auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})
        _ = [
            event
            async for event in stream_with_failover(
                self._isolated(), auth, None, client_for, route_state=state
            )
        ]
        assert state.active == pinned, "an isolated call cleared the turn's sticky route"

    async def test_it_never_sleeps_on_a_backoff(self) -> None:
        """A retry budget would hold the naming worker (and the 15-second
        ceiling ``TITLE_TIMEOUT_S`` puts on it) open across sleeps for a result
        nobody is waiting on."""
        slept: list[float] = []

        async def spy_sleep(delay_ms: float, signal: Any = None) -> None:
            slept.append(delay_ms)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = spy_sleep  # type: ignore[assignment]
        try:

            async def client_for(spec: ModelSpec) -> Any:
                return ScriptedClient(ProviderError(500, "boom", retryable=True))

            auth = FakeAuth({"openai": ["k1"]})
            with pytest.raises(ProviderError):
                _ = [
                    event
                    async for event in stream_with_failover(
                        self._isolated(),
                        auth,
                        {"retry": {"baseDelayMs": 50, "maxRetries": 3}},
                        client_for,
                    )
                ]
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]
        assert slept == [], "an isolated call spent a backoff sleep"

    async def test_one_attempt_is_all_it_gets(self) -> None:
        """The sum of the four above: exactly one call reaches the wire."""
        client = ScriptedClient(ProviderError(429, "rate limited", retryable=True))

        async def client_for(spec: ModelSpec) -> Any:
            return client

        auth = FakeAuth({"openai": ["k1", "k2"]})
        settings = {
            "retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}
        }
        with pytest.raises(ProviderError):
            _ = [
                event
                async for event in stream_with_failover(
                    self._isolated(), auth, settings, client_for
                )
            ]
        assert client.calls == 1

    async def test_its_credential_read_neither_blocks_nor_repoints_the_turn(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sixth route, and the only one the two switches above do not reach:
        the cascade takes routing decisions on what looks like a READ.

        ``AuthStore._resolve`` passes an OAuth row whose refresh RAISED to
        ``block_credential`` — which hides it from ``_usable_key_rows`` on every
        later resolve, and the turn re-resolves on each tool-loop request — and
        writes session stickiness to whichever row wins instead. So a transient
        token-endpoint failure inside a decorative title call could take the
        credential the turn is transacting on out of service and move the turn
        onto a sibling account mid-conversation. Both switches are downstream of
        this: it happens inside the resolver, before any error reaches the retry
        or route logic.

        A real ``AuthStore`` on a temp DB, because the mutation being denied is
        the real cascade's, not a fake's.
        """
        store = AuthStore(db_path=tmp_path / "auth.db")
        session_id = "session-with-a-live-turn"
        # Row A is the account the turn is on and its refresh is due; row B is a
        # healthy sibling, so the cascade has somewhere to move to.
        turn_row = store.upsert_credential(
            "openai",
            {"refresh": "r-a", "access": "stale-a", "expires": 0, "account_id": "acct-a"},
        )
        sibling_row = store.upsert_credential(
            "openai",
            {
                "refresh": "r-b",
                "access": "good-b",
                "expires": int(time.time() * 1000) + 3_600_000,
                "account_id": "acct-b",
            },
        )

        async def refresh_always_fails(creds: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("token endpoint 503")

        monkeypatch.setattr(store, "_refresh_fn", lambda provider: refresh_always_fails)

        async def client_for(spec: ModelSpec) -> Any:
            return ScriptedClient([StreamEndEvent(stop_reason="stop")])

        def the_turn_is_on_row_a() -> None:
            store._sticky[("openai", session_id)] = turn_row.id

        try:
            the_turn_is_on_row_a()
            _ = [
                event
                async for event in stream_with_failover(
                    self._isolated(), store, None, client_for, session_id=session_id
                )
            ]
            assert not store.is_blocked(
                turn_row.id, "openai"
            ), "an isolated call blocked the credential the turn is transacting on"
            assert (
                store._sticky[("openai", session_id)] == turn_row.id
            ), "an isolated call repointed the session's sticky credential"

            # CONTROL, on the same store and the same failing refresh: an
            # ordinary request DOES both, so neither assertion above can be
            # passing because this fixture never reaches the path.
            the_turn_is_on_row_a()
            _ = [
                event
                async for event in stream_with_failover(
                    _request(), store, None, client_for, session_id=session_id
                )
            ]
            assert store.is_blocked(turn_row.id, "openai"), "the refresh failure blocked nothing"
            assert (
                store._sticky[("openai", session_id)] == sibling_row.id
            ), "stickiness did not move on an ordinary request"
        finally:
            store.close()

    async def test_the_same_failure_on_an_ORDINARY_request_does_all_four(self) -> None:
        """The control. Without this, every assertion above could be passing
        because the fake never triggers those paths at all."""
        client = ScriptedClient(ProviderError(500, "boom", retryable=True))
        specs_seen: list[str] = []
        slept: list[float] = []

        async def spy_sleep(delay_ms: float, signal: Any = None) -> None:
            slept.append(delay_ms)

        async def client_for(spec: ModelSpec) -> Any:
            specs_seen.append(spec.model_id)
            return client

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = spy_sleep  # type: ignore[assignment]
        try:
            auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})
            settings = {
                "retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}
            }
            state = FailoverRouteState()
            with pytest.raises(ProviderError):
                _ = [
                    event
                    async for event in stream_with_failover(
                        _request(), auth, settings, client_for, route_state=state
                    )
                ]
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]
        assert specs_seen == ["gpt-4o", "claude-x"], "the chain was not walked"
        assert slept, "no backoff was spent"
        assert client.calls > 1, "only one attempt was made"
        assert state.active is not None, "no route was pinned"


class TestProviderOutageWalksTheWholePool:
    """A provider-side fault must ask every account, and blame none of them.

    The reported incident: four Anthropic OAuth accounts, a 529
    ``overloaded_error`` storm, and a turn that died reporting the 529 after
    trying TWO of them. Two accounts with quota were never asked.
    """

    async def test_a_529_storm_tries_every_credential_in_the_pool(self) -> None:
        """The regression: rotation stopped at the ordinary-401 switch cap."""
        tried: list[str | None] = []

        def overloaded(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            tried.append(api_key)
            raise ProviderError(529, "overloaded_error: Overloaded", retryable=True)

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(overloaded)

        auth = FakeAuth({"openai": ["acct-a", "acct-b", "acct-c", "acct-d"]})
        settings = {"retry": {"baseDelayMs": 0, "maxRetries": 0, "fallbackChains": {}}}
        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(_request(), auth, settings, client_for):
                pass

        assert excinfo.value.status == 529
        assert set(tried) == {"acct-a", "acct-b", "acct-c", "acct-d"}, tried

    async def test_an_ordinary_401_still_stops_after_one_switch(self) -> None:
        """The cap is kept where it belongs: a rejected bearer is not a pool problem.

        Cycling every sibling on a 401 only delays the login prompt the user has
        to answer anyway, so widening rotation must not have widened this.
        """
        tried: list[str | None] = []

        def unauthorized(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            tried.append(api_key)
            raise ProviderError(401, "invalid bearer", retryable=False, auth_error=True)

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(unauthorized)

        auth = FakeAuth({"openai": ["acct-a", "acct-b", "acct-c", "acct-d"]})
        settings = {"retry": {"baseDelayMs": 0, "maxRetries": 0, "fallbackChains": {}}}
        with pytest.raises(ProviderError):
            async for _ in stream_with_failover(_request(), auth, settings, client_for):
                pass

        assert len(tried) < 4, f"a 401 walked the whole pool: {tried}"


class TestCredentialErrorSaysWhichProblemItIs:
    """ "No API key configured" must not be said to a user who is signed in."""

    async def test_a_rate_limited_oauth_account_is_not_reported_as_unconfigured(
        self, tmp_path: Any
    ) -> None:
        """The reported frame: `No API key configured for provider 'openai'`
        shown while an OAuth sign-in existed and was merely blocked."""
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        row = store.upsert_credential(
            "openai",
            {"type": "oauth", "access": "tok", "refresh": "r", "expires": None},
        )
        store.block_credential(row.id, "openai", block_ms=600_000)

        def unreachable(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            raise AssertionError("no request should be sent without a bearer")

        async def client_for(spec: ModelSpec) -> Any:
            # Built before the credential is resolved, so the guard belongs on
            # the STREAM rather than on construction.
            return _FnClient(unreachable)

        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(
                _request(), store, {"retry": {"enabled": True}}, client_for
            ):
                pass

        message = str(excinfo.value)
        assert "No API key configured" not in message, message
        assert "not usable right now" in message
        # The three causes a present-but-unresolvable row can have. A message
        # naming only the first two told an R21 user to wait for a limit that
        # was never the problem.
        assert "rate limited" in message
        assert "token refresh" in message
        assert "could not be read" in message
        assert "OAuth sign-in" in message

    async def test_a_provider_with_no_credential_still_says_so(self, tmp_path: Any) -> None:
        """The original wording is correct when it is actually true."""
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")

        def unreachable(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            raise AssertionError("no request should be sent without a bearer")

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(unreachable)

        with pytest.raises(ProviderError) as excinfo:
            async for _ in stream_with_failover(
                _request(), store, {"retry": {"enabled": True}}, client_for
            ):
                pass

        assert "No API key configured for provider 'openai'" in str(excinfo.value)


async def _clean_stream() -> AsyncIterator[Any]:
    """A stream that completes without events: the attempt SUCCEEDED."""
    return
    yield


class TestAnOverloadedProviderIsNotFloodedWhileRotating:
    """Widening rotation must not multiply the load aimed at the outage.

    Round 1 review (R2): the same-credential transport budget is spent once per
    ACCOUNT, so the default `maxRetries: 10` became 10 x pool size requests sent
    to a provider that had just answered "overloaded" -- measured at 44 requests
    over ~190s for a four-account pool, against 22/~100s before the widening.
    """

    async def test_a_529_storm_is_bounded_per_account(self, tmp_path: Any) -> None:
        sent: list[str | None] = []
        slept: list[int] = []

        def overloaded(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            sent.append(api_key)
            raise ProviderError(529, "overloaded_error: Overloaded", retryable=True)

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            slept.append(delay_ms)

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(overloaded)

        # A REAL store: the cap only applies when the pool is enumerable and has
        # more than one member, which is the condition it is justified by.
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        for i in range(1, 5):
            store.upsert_credential(
                "openai",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            # The DEFAULT budget, which is the configuration that misbehaved.
            with pytest.raises(ProviderError):
                async for _ in stream_with_failover(
                    _request(), store, {"retry": {"enabled": True}}, client_for
                ):
                    pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        # Every account is still tried -- the pool walk is the point.
        assert set(sent) == {"k1", "k2", "k3", "k4"}, sent
        # Each is asked a bounded number of times, not `maxRetries` times, and
        # the TURN total is bounded too -- that is the quantity the provider
        # actually receives.
        for key in ("k1", "k2", "k3", "k4"):
            assert sent.count(key) <= 4, (key, sent)
        assert len(sent) <= 12, sent

    async def test_a_timeout_storm_is_bounded_too(self, tmp_path: Any) -> None:
        """R10: the cap was installed only on the `except ProviderError` arm.

        Raw transport failures take the `except Exception` arm -- no client in
        `clients.py` catches httpx, and `_guarded_chunks` deliberately raises
        `ReadTimeout` on a stall -- which kept its own uncapped budget. That is
        the failure kind a provider degradation most often produces, so the
        bound was missing exactly where it was needed: 44 requests/~196s
        measured, twice the pre-change behaviour.
        """
        import httpx

        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        for i in range(4):
            store.upsert_credential(
                "openai",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )
        sent: list[str | None] = []

        def stalled(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            sent.append(api_key)
            raise httpx.ReadTimeout("stream stalled")

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(stalled)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(ProviderError):
                async for _ in stream_with_failover(
                    _request(), store, {"retry": {"enabled": True}}, client_for
                ):
                    pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert len(sent) <= 12, f"a timeout storm sent {len(sent)} requests"

    async def test_a_lone_credential_keeps_the_budget_the_user_configured(
        self, tmp_path: Any
    ) -> None:
        """R11: the cap is justified by pool MULTIPLICATION, so it must not bite
        where there is no pool. With one credential the earlier form failed a
        500 blip that cleared on attempt 4 -- a regression against `main` for
        the users least able to absorb one.
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential(
            "openai",
            {"type": "oauth", "access": "only", "refresh": "r", "expires": None},
        )
        attempts = [0]

        def blip_then_recover(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts[0] += 1
            if attempts[0] < 4:
                raise ProviderError(500, "blip", retryable=True)
            return _clean_stream()

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(blip_then_recover)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request(), store, {"retry": {"enabled": True}}, client_for
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        # The 4th attempt has to be REACHED; a cap of 2 would have stopped at 3.
        assert attempts[0] == 4, attempts[0]


class TestTheRetryCapAsksWhatRotationWouldAnswer:
    """R16: the cap is justified by a sibling the turn can ACTUALLY rotate onto.

    Counting raw credential rows claimed a pool in two configurations where
    rotation reaches nothing, and the cap then removed retries with nowhere to
    spend them -- R11's symptom through a different door.
    """

    @staticmethod
    def _blip_client(attempts: list[int]) -> Any:
        def blip_then_recover(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts[0] += 1
            if attempts[0] < 4:
                raise ProviderError(500, "blip", retryable=True)
            return _clean_stream()

        return blip_then_recover

    async def _recovers(self, store: Any) -> int:
        attempts = [0]
        client = self._blip_client(attempts)

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(client)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request(), store, {"retry": {"enabled": True}}, client_for, session_id="s"
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]
        return attempts[0]

    async def test_a_blocked_sibling_is_not_a_pool(self, tmp_path: Any) -> None:
        """This PR's own headline shape: several accounts spent during a
        degradation, one healthy. The survivor must keep its full budget."""
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        rows = [
            store.upsert_credential(
                "openai",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )
            for i in range(4)
        ]
        for row in rows[1:]:
            store.block_credential(row.id, "openai", block_ms=600_000)

        assert await self._recovers(store) == 4

    async def test_a_different_credential_type_is_not_a_sibling(self, tmp_path: Any) -> None:
        """`rotate_sibling` only walks rows of the SAME credential_type, so an
        OAuth sign-in beside a pasted API key is two rows and zero rotation."""
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential(
            "openai", {"type": "oauth", "access": "o1", "refresh": "r", "expires": None}
        )
        store.upsert_credential("openai", {"type": "api_key", "key": "sk-1", "source": "login"})

        assert await self._recovers(store) == 4


class TestTheCapEngagesOnObservedRotation:
    """Direct coverage for `_request_has_rotated`.

    Five rounds of review each found the previous PREDICTION of "is there a
    sibling?" drifting from what the cascade actually does -- rows, then
    unblocked rows, then same-type rows, then rows behind an override, then
    rows split across cascade tiers. Each drift cost a real user retries.

    The question is now answered by observation: `attempted_keys` is the set of
    distinct bearers `resolve_next_key` has already returned, so the cap
    engages exactly when a second one exists and cannot be wrong about a
    configuration it never enumerates. These tests pin that contract.
    """

    def test_no_bearer_yet_is_not_a_pool(self) -> None:
        assert not _request_has_rotated(AuthRetryKeyState())

    def test_one_bearer_is_not_a_pool(self) -> None:
        """A lone credential, an override bearer, or a pool whose siblings are
        all spent: whatever the table says, ONE bearer cannot multiply."""
        state = AuthRetryKeyState(attempted_keys={"only"})
        assert not _request_has_rotated(state)

    def test_a_second_distinct_bearer_is_a_pool(self) -> None:
        state = AuthRetryKeyState(attempted_keys={"k1", "k2"})
        assert _request_has_rotated(state)


class TestAnOverrideBearerKeepsItsBudget:
    """End to end through the cascade tier no failover test previously used.

    R20 was invisible because every failover test resolves through the stored
    credential tiers. A user who pasted API keys and then pointed at a gateway
    (`--api-key`) or exported `ANTHROPIC_API_KEY` got 3 requests where `main`
    gave 11 -- the cap firing on a bearer with nothing to rotate to.
    """

    async def test_a_runtime_override_is_not_capped_by_the_rows_beside_it(
        self, tmp_path: Any
    ) -> None:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        for i in range(2):
            store.upsert_credential(
                "openai", {"type": "api_key", "key": f"sk-{i}", "source": "login"}
            )
        store.set_runtime_api_key("openai", "gateway-key")

        sent: list[str | None] = []

        def overloaded(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            sent.append(api_key)
            raise ProviderError(529, "overloaded_error: Overloaded", retryable=True)

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(overloaded)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(ProviderError):
                async for _ in stream_with_failover(
                    _request(),
                    store,
                    {"retry": {"enabled": True, "maxRetries": 10, "baseDelayMs": 0}},
                    client_for,
                ):
                    pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        # The override served every attempt: there is no sibling to rotate to,
        # so it is never starved by a pool that does not exist. It gets the
        # pre-rotation allowance and then, once rotation reports exhaustion, the
        # configured budget again -- well beyond the 3 requests the earlier
        # table-counting versions allowed it.
        assert set(sent) == {"gateway-key"}, sent
        assert len(sent) >= 8, len(sent)


class TestTheTurnWideCeiling:
    """The per-credential cap bounds each account; this bounds their PRODUCT.

    Widening rotation to walk the whole pool multiplied the load by the pool
    size at exactly the moment the provider was asking for less. The quantity
    that reaches the provider is the total, so that is what is bounded.
    """

    async def test_a_storm_never_exceeds_the_turn_ceiling(self, tmp_path: Any) -> None:
        from local_operator.providers.auth_store import AuthStore
        from local_operator.providers.failover import MAX_SERVER_FAULT_REQUESTS_PER_TURN

        store = AuthStore(db_path=tmp_path / "auth.db")
        for i in range(6):  # a pool larger than the ceiling would allow per-account
            store.upsert_credential(
                "openai",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )
        sent: list[str | None] = []

        def overloaded(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            sent.append(api_key)
            raise ProviderError(529, "overloaded_error: Overloaded", retryable=True)

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(overloaded)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(ProviderError):
                async for _ in stream_with_failover(
                    _request(), store, {"retry": {"enabled": True, "maxRetries": 10}}, client_for
                ):
                    pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert len(sent) <= MAX_SERVER_FAULT_REQUESTS_PER_TURN + 1, len(sent)
        # And it still rotated rather than hammering one account.
        assert len(set(sent)) > 1, sent

    async def test_api_key_rows_split_across_cascade_tiers_keep_their_budget(
        self, tmp_path: Any
    ) -> None:
        """R22: `api_key` rows live in cascade tier 4 (`source == "login"`) and
        tier 6, and a tier-4 row always wins -- so two rows of the same type can
        still yield exactly one reachable bearer. Predicting this from the table
        is what the observation model removes the need to do."""
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential(
            "openai", {"type": "api_key", "key": "login-key", "source": "login"}
        )
        store.upsert_credential("openai", {"type": "api_key", "key": "migrated-key"})

        attempts = [0]

        def blip_then_recover(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts[0] += 1
            if attempts[0] < 6:
                raise ProviderError(500, "blip", retryable=True)
            return _clean_stream()

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(blip_then_recover)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request(),
                store,
                {"retry": {"enabled": True, "maxRetries": 10}},
                client_for,
                session_id="s",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        # Reached attempt 6 rather than being capped at 3.
        assert attempts[0] == 6, attempts[0]


class TestTheCeilingIsPerProviderNotPerTurn:
    """R23: a chain stops being a chain if the primary spends its allowance.

    The server-fault ceiling was counted across the whole turn, so a primary
    outage consumed all of it and every fallback target got ONE attempt with no
    retries -- a fallback that would have succeeded on its second try never got
    one. Each target is a different service having a different day.
    """

    async def test_a_fallback_gets_a_real_budget_after_a_primary_storm(self, tmp_path: Any) -> None:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        for i in range(4):  # enough accounts to exhaust the primary's ceiling
            store.upsert_credential(
                "openai",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )
        store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "fallback", "refresh": "r", "expires": None},
        )

        per_provider: dict[str, int] = {}

        def overloaded_then_fallback_recovers(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            provider = request.model.provider
            per_provider[provider] = per_provider.get(provider, 0) + 1
            # The fallback succeeds on its SECOND attempt -- it must be given one.
            if provider == "anthropic" and per_provider[provider] >= 2:
                return _clean_stream()
            raise ProviderError(529, "overloaded_error: Overloaded", retryable=True)

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(overloaded_then_fallback_recovers)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request(),
                store,
                {
                    "retry": {
                        "enabled": True,
                        "fallbackChains": {"default": ["anthropic/claude-opus-5"]},
                    }
                },
                client_for,
                session_id="s",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert per_provider.get("anthropic", 0) >= 2, per_provider


class TestARestoredBudgetIsTheConfiguredOne:
    """R24: the restore-once path must express the budget the user asked for.

    Zeroing the counter but re-entering the predicate as "not yet rotated"
    re-capped the restored pass at the first-bearer allowance: a lone credential
    got the same 8 requests whatever `maxRetries` said -- short of a configured
    10, and double a configured 2. Wrong in both directions, which is the tell
    that the mechanism was not saying what it meant.
    """

    @staticmethod
    async def _requests_for(store: Any, max_retries: int) -> int:
        attempts = [0]

        def always_500(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts[0] += 1
            raise ProviderError(500, "blip", retryable=True)

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(always_500)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(ProviderError):
                async for _ in stream_with_failover(
                    _request(),
                    store,
                    {"retry": {"enabled": True, "maxRetries": max_retries}},
                    client_for,
                    session_id="s",
                ):
                    pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]
        return attempts[0]

    @pytest.mark.parametrize("max_retries", list(range(0, 12)))
    async def test_a_lone_credential_spends_exactly_its_configured_budget(
        self, tmp_path: Any, max_retries: int
    ) -> None:
        """`max_retries + 1` requests -- the same total as before this change.

        The restore-once path hands back the REMAINDER of the user's budget; it
        neither zeroes the counter nor reopens the budget. An earlier form let
        the token-change reset zero it first, so a lone credential spent
        `2 x (max_retries + 1)` requests -- twice what was asked for, inside a
        change whose whole purpose is to stop hammering a provider that is
        already failing.

        Swept across EVERY value rather than a sample. The first version of
        this test checked 0/1/3/5/10 and stepped straight over the only setting
        that was wrong: at `maxRetries: 4` the pre-rotation allowance lands
        exactly ON the budget, and an exclusive `<` comparison skipped the
        restore and cost the user their last request. A sampled sweep is how a
        boundary bug survives a test written to catch boundary bugs.
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / f"auth-{max_retries}.db")
        store.upsert_credential(
            "openai",
            {"type": "oauth", "access": "only", "refresh": "r", "expires": None},
        )

        assert await self._requests_for(store, max_retries) == max_retries + 1


class TestRotationPermutations:
    """The credential shapes users actually hold, each driven END TO END
    through ``stream_with_failover`` on a real ``AuthStore``.

    Rotation policy and cascade tiers each had unit coverage, but the
    incident that motivated this class was a composition failure: every
    session died reporting four credentials unusable while three still held
    quota. So each permutation here asserts the thing a user cares about —
    after the failure, SOMEBODY serves — not the mechanism that got there.
    """

    @staticmethod
    def _no_sleep():
        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        return no_sleep

    async def test_oauth_429_rotates_to_the_sibling_account(self, tmp_path: Any) -> None:
        """OAuth → OAuth inside one provider: the first account's 429 blocks it
        for the advertised reset and the sibling serves the SAME request."""
        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential(
            "anthropic",
            # Distinct emails: OAuth rows without an identity field dedupe to
            # one per-provider row (see `_identity_key_for`), silently turning
            # a two-account pool into a single account.
            {
                "type": "oauth",
                "access": "acct-a",
                "refresh": "r",
                "expires": None,
                "email": "a@example.com",
            },
        )
        store.upsert_credential(
            "anthropic",
            {
                "type": "oauth",
                "access": "acct-b",
                "refresh": "r",
                "expires": None,
                "email": "b@example.com",
            },
        )
        served: list[str | None] = []

        def first_account_exhausted(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            if api_key == "acct-a":
                raise ProviderError(
                    429, "quota reset pending", retryable=True, retry_after_ms=3 * 3_600_000
                )
            served.append(api_key)
            return _clean_stream()

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(first_account_exhausted)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = self._no_sleep()  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request("anthropic", "claude-opus-5"),
                store,
                {"retry": {"enabled": True}},
                client_for,
                session_id="s0",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert served == ["acct-b"]
        rows = {r.data["access"]: r for r in store.list_credentials("anthropic")}
        # usageAwareFallback is OFF here: no preflight probe exists to upgrade
        # a family block to an account-wide one on this path, so rotation keeps
        # the pre-existing account-wide semantics.
        assert store.is_blocked(rows["acct-a"].id, "anthropic")
        assert not store.is_blocked(rows["acct-b"].id, "anthropic")

    async def test_usage_aware_rotation_scopes_the_429_block_to_the_family(
        self, tmp_path: Any
    ) -> None:
        """With usage-aware routing on, a usage-limit 429 blocks the family.

        The opt-in counterpart of the previous test: the preflight's usage
        probe can upgrade a family block to an account-wide one the moment a
        shared window is the binding limit, so rotation may scope the block
        to the family the request ran on — the account stays in rotation for
        every other family."""
        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential(
            "anthropic",
            {
                "type": "oauth",
                "access": "acct-a",
                "refresh": "r",
                "expires": None,
                "email": "a@example.com",
            },
        )
        store.upsert_credential(
            "anthropic",
            {
                "type": "oauth",
                "access": "acct-b",
                "refresh": "r",
                "expires": None,
                "email": "b@example.com",
            },
        )
        served: list[str | None] = []

        def first_account_exhausted(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            if api_key == "acct-a":
                raise ProviderError(
                    429, "quota reset pending", retryable=True, retry_after_ms=3 * 3_600_000
                )
            served.append(api_key)
            return _clean_stream()

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(first_account_exhausted)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = self._no_sleep()  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request("anthropic", "claude-opus-5"),
                store,
                {"retry": {"enabled": True, "usageAwareFallback": True}},
                client_for,
                session_id="s0",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert served == ["acct-b"]
        rows = {r.data["access"]: r for r in store.list_credentials("anthropic")}
        # Scoped to the family the request ran on: opus is blocked, fable
        # still resolves the row, and no account-wide block was written.
        assert store.is_blocked_for_model(rows["acct-a"].id, "anthropic", "claude-opus-5")
        assert not store.is_blocked_for_model(rows["acct-a"].id, "anthropic", "claude-fable-5")
        assert not store.is_blocked(rows["acct-a"].id, "anthropic")

    async def test_a_429d_oauth_row_falls_through_to_an_api_key_row(self, tmp_path: Any) -> None:
        """OAuth → API key inside one provider. ``rotate_sibling`` only counts
        same-type siblings, so the cross-type hop happens in the cascade: the
        blocked OAuth tier empties and tier 6 hands over the stored key. This
        is the whole request surviving that hop, not the tier unit test."""
        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "oauth-token", "refresh": "r", "expires": None},
        )
        store.upsert_credential("anthropic", {"type": "api_key", "key": "sk-stored"})
        served: list[str | None] = []

        def oauth_exhausted(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            if api_key == "oauth-token":
                raise ProviderError(
                    429, "quota reset pending", retryable=True, retry_after_ms=3 * 3_600_000
                )
            served.append(api_key)
            return _clean_stream()

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(oauth_exhausted)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = self._no_sleep()  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request("anthropic", "claude-opus-5"),
                store,
                {"retry": {"enabled": True}},
                client_for,
                session_id="s0",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert served == ["sk-stored"]

    async def test_a_multi_key_pool_walks_every_key_until_one_serves(self, tmp_path: Any) -> None:
        """API key → API key → API key: the token-plan shape (several pasted
        keys under one provider). Two keys 429; the third serves."""
        store = AuthStore(db_path=tmp_path / "auth.db")
        for i in range(3):
            store.upsert_credential(
                "alibaba-token-plan", {"type": "api_key", "key": f"plan-key-{i}"}
            )
        served: list[str | None] = []
        rejected: list[str | None] = []

        def two_exhausted(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            # Order-independent on purpose: which key the walk meets first is a
            # session-hash accident, so the first two DISTINCT keys are the
            # exhausted ones and whichever remains serves.
            if len(rejected) < 2:
                rejected.append(api_key)
                raise ProviderError(
                    429, "quota reset pending", retryable=True, retry_after_ms=3 * 3_600_000
                )
            served.append(api_key)
            return _clean_stream()

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(two_exhausted)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = self._no_sleep()  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request("alibaba-token-plan", "qwen3-max"),
                store,
                {"retry": {"enabled": True}},
                client_for,
                session_id="s0",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert len(rejected) == 2 and len(set(rejected)) == 2
        assert len(served) == 1
        assert {*rejected, *served} == {"plan-key-0", "plan-key-1", "plan-key-2"}

    async def test_cross_provider_fallback_resolves_an_api_key_provider(
        self, tmp_path: Any
    ) -> None:
        """OAuth primary exhausted → chain hops providers → the fallback's
        LOGIN API key resolves through the real cascade and serves."""
        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "acct-a", "refresh": "r", "expires": None},
        )
        store.upsert_credential(
            "openai", {"type": "api_key", "key": "sk-openai", "source": "login"}
        )
        served: list[tuple[str, str | None]] = []

        def primary_exhausted(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            if request.model.provider == "anthropic":
                raise ProviderError(
                    429, "quota reset pending", retryable=True, retry_after_ms=3 * 3_600_000
                )
            served.append((request.model.provider, api_key))
            return _clean_stream()

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(primary_exhausted)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = self._no_sleep()  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request("anthropic", "claude-opus-5"),
                store,
                {
                    "retry": {
                        "enabled": True,
                        "fallbackChains": {"default": ["openai/gpt-5.4"]},
                    }
                },
                client_for,
                session_id="s0",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert served == [("openai", "sk-openai")]


class TestTheLoopBackSweep:
    """An exhausted walk gets ONE more look at the waterfall before the verdict.

    "Every target failed once" is stale evidence by the time the walk ends:
    blocks written by other processes expire, short throttles clear, and
    targets the route-state pin trimmed off were never asked at all. The sweep
    is bounded (one revisit per call) and discriminating (never-walked targets,
    short-throttle quota failures, and blocked-not-absent credential pools —
    never server faults, whose budgets were already spent in place).
    """

    async def test_the_sweep_revisits_targets_the_route_pin_excluded(self) -> None:
        """The 'sample upwards' case: pinned to the last fallback, which dies —
        the sweep walks back up and the recovered primary serves the turn."""
        served: list[str] = []
        sleeps: list[int] = []

        def fallback_dead_primary_alive(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            if request.model.provider == "openai":
                raise ProviderError(400, "schema rejected", retryable=False, kind="request")
            served.append(request.model.provider)
            return _clean_stream()

        async def record_sleep(delay_ms: int, signal: Any) -> None:
            sleeps.append(delay_ms)

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(fallback_dead_primary_alive)

        route_state = FailoverRouteState()
        fallback = FallbackTarget("openai/gpt-5.4")
        await route_state.activate(fallback, "provider failure", cooldown_ms=600_000)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = record_sleep  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request("anthropic", "claude-opus-5"),
                FakeAuth({"anthropic": ["acct"], "openai": ["sk"]}),
                {
                    "retry": {
                        "enabled": True,
                        "baseDelayMs": 1,
                        "fallbackChains": {"default": ["openai/gpt-5.4"]},
                    }
                },
                client_for,
                session_id="s0",
                route_state=route_state,
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        # The pinned walk asked only openai; the sweep asked the primary.
        assert served == ["anthropic"]

    async def test_a_short_throttle_is_swept_once_and_can_serve(self) -> None:
        """A bearer with no store row (env-var shaped) 429s through its budget;
        the sweep waits the advertised delay and the SAME bearer serves."""
        attempts: list[str | None] = []
        sleeps: list[int] = []

        class EnvAuth:
            """get_api_key-only store: one bearer, nothing to block."""

            async def get_api_key(
                self, provider: str, session_id: str | None = None, **kwargs: Any
            ) -> str | None:
                return "env-key"

            def rotate_sibling(
                self,
                provider: str,
                session_id: str | None,
                error: Any,
                api_key: str | None = None,
                *,
                model_id: str = "",
            ) -> bool:
                return False

        def throttled_then_clear(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts.append(api_key)
            if len(attempts) < 4:
                raise ProviderError(429, "brief throttle", retryable=True, retry_after_ms=5)
            return _clean_stream()

        async def record_sleep(delay_ms: int, signal: Any) -> None:
            sleeps.append(delay_ms)

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(throttled_then_clear)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = record_sleep  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request(),
                EnvAuth(),
                {"retry": {"enabled": True, "baseDelayMs": 1, "fallbackChains": {}}},
                client_for,
                session_id="s0",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        # Three in-place attempts (initial + the short-throttle pair), then the
        # sweep's fourth after sleeping the advertised 5 ms.
        assert attempts == ["env-key"] * 4
        assert sleeps[-1] == 5

    async def test_the_sweep_runs_at_most_once(self) -> None:
        """A throttle that never clears gets exactly one revisit — the sweep
        must not turn the walk into a spin."""
        attempts: list[str | None] = []

        class EnvAuth:
            async def get_api_key(
                self, provider: str, session_id: str | None = None, **kwargs: Any
            ) -> str | None:
                return "env-key"

            def rotate_sibling(
                self,
                provider: str,
                session_id: str | None,
                error: Any,
                api_key: str | None = None,
                *,
                model_id: str = "",
            ) -> bool:
                return False

        def always_throttled(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            attempts.append(api_key)
            raise ProviderError(429, "brief throttle", retryable=True, retry_after_ms=5)

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(always_throttled)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(ProviderError) as excinfo:
                async for _ in stream_with_failover(
                    _request(),
                    EnvAuth(),
                    {"retry": {"enabled": True, "baseDelayMs": 1, "fallbackChains": {}}},
                    client_for,
                    session_id="s0",
                ):
                    pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert excinfo.value.status == 429
        # Three attempts per walk, exactly two walks.
        assert attempts == ["env-key"] * 6

    async def test_a_block_expiring_mid_walk_is_caught_by_the_sweep(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """The incident, inverted: every credential blocked at resolve time is
        a RETRYABLE verdict, so the sweep re-resolves — and a block that
        expired while the walk was busy failing puts the account back in
        service instead of the turn dying on stale evidence."""
        clock = {"now": 1_000_000}
        monkeypatch.setattr(AuthStore, "_now_ms", staticmethod(lambda: clock["now"]))
        store = AuthStore(db_path=tmp_path / "auth.db")
        row = store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "acct-a", "refresh": "r", "expires": None},
        )
        store.block_credential(row.id, "anthropic", block_ms=2_000)
        served: list[str | None] = []

        def serves(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            served.append(api_key)
            return _clean_stream()

        async def sleep_advances_clock(delay_ms: int, signal: Any) -> None:
            clock["now"] += 5_000

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(serves)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = sleep_advances_clock  # type: ignore[assignment]
        try:
            async for _ in stream_with_failover(
                _request("anthropic", "claude-opus-5"),
                store,
                {"retry": {"enabled": True, "baseDelayMs": 1}},
                client_for,
                session_id="s0",
            ):
                pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert served == ["acct-a"]

    async def test_a_still_blocked_pool_keeps_the_quota_verdict(self, tmp_path: Any) -> None:
        """When the sweep finds the pool still blocked, the verdict is the
        retryable quota error naming the blocked credentials — never the
        'No API key configured' misdiagnosis."""
        store = AuthStore(db_path=tmp_path / "auth.db")
        for access, email in (("acct-a", "a@example.com"), ("acct-b", "b@example.com")):
            row = store.upsert_credential(
                "anthropic",
                {
                    "type": "oauth",
                    "access": access,
                    "refresh": "r",
                    "expires": None,
                    "email": email,
                },
            )
            store.block_credential(row.id, "anthropic", block_ms=3_600_000)

        def never_reached(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            raise AssertionError("no bearer should resolve")

        async def no_sleep(delay_ms: int, signal: Any) -> None:
            return None

        async def client_for(spec: ModelSpec) -> Any:
            return _FnClient(never_reached)

        original = failover_module._abortable_sleep
        failover_module._abortable_sleep = no_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(ProviderError) as excinfo:
                async for _ in stream_with_failover(
                    _request("anthropic", "claude-opus-5"),
                    store,
                    {"retry": {"enabled": True, "baseDelayMs": 1}},
                    client_for,
                    session_id="s0",
                ):
                    pass
        finally:
            failover_module._abortable_sleep = original  # type: ignore[assignment]

        assert excinfo.value.kind == "quota"
        assert excinfo.value.retryable
        assert "not usable right now" in excinfo.value.message


@pytest.mark.asyncio
async def test_route_edges_reach_the_settle_handler_in_both_directions() -> None:
    """The settle hook hears the fallback pin AND the recovery.

    The recovery direction is the one a display cannot live without: the
    driver's success path used to `clear()` silently, so a band that learned
    "now serving from the fallback" had no edge to learn "back on the
    primary" from — the same stale frame in the opposite direction.
    """
    primary_healthy = False

    async def client_for(spec: ModelSpec) -> Any:
        if spec.provider == "openai" and not primary_healthy:
            # Unknown-kind (status=None) "primary down" trigger, not a 400: a
            # primary request-shape 400 now aborts instead of walking, so it
            # would never reach the settle-handler pin this test asserts.
            return ScriptedClient(ProviderError(None, "model unavailable"))
        return ScriptedClient([StreamEndEvent(stop_reason="stop")])

    settings = {"retry": {"fallbackChains": {"default": ["anthropic/claude-opus-5"]}}}
    edges: list[Any] = []
    state = FailoverRouteState(on_settle=lambda target, reason: edges.append(target))
    auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})

    _ = [
        event
        async for event in stream_with_failover(
            _request(), auth, settings, client_for, route_state=state
        )
    ]
    assert edges == [FallbackTarget("anthropic/claude-opus-5")]

    # The primary recovers; a probe reaching it must settle the route back.
    primary_healthy = True
    state.primary_retry_at_ms = 0  # the cooldown has elapsed
    state.active = None  # preflight's probe path re-opens the primary
    _ = [
        event
        async for event in stream_with_failover(
            _request(), auth, settings, client_for, route_state=state
        )
    ]
    # No new edge: nothing was pinned when the primary served, so the clear
    # is a no-op rather than a spurious "back on" for a route never left.
    assert edges == [FallbackTarget("anthropic/claude-opus-5")]


@pytest.mark.asyncio
async def test_primary_success_with_a_pinned_route_settles_the_recovery() -> None:
    """`clear_settled` fires exactly on the pinned→primary transition."""
    edges: list[Any] = []
    state = FailoverRouteState(on_settle=lambda target, reason: edges.append(target))
    state.active = FallbackTarget("anthropic/claude-opus-5")

    async def client_for(spec: ModelSpec) -> Any:
        return ScriptedClient([StreamEndEvent(stop_reason="stop")])

    # The pinned target is not in this request's chain (no chains configured),
    # so the walk starts at the primary — the "primary probe succeeds" shape.
    _ = [
        event
        async for event in stream_with_failover(
            _request(), FakeAuth({"openai": ["k1"]}), None, client_for, route_state=state
        )
    ]
    assert edges == [None]
    assert state.active is None


# --- Auth recovery hints (item 9) -------------------------------------------


def test_append_auth_recovery_adds_hint_to_auth_error() -> None:
    from local_operator.providers.failover import append_auth_recovery

    rendered = "authentication failed (HTTP 401): invalid x-api-key"
    out = append_auth_recovery(rendered, "openai")
    assert rendered in out
    assert "/login openai" in out
    assert "credential update <OPENAI_API_KEY>" in out


def test_append_auth_recovery_ignores_non_auth_errors() -> None:
    from local_operator.providers.failover import append_auth_recovery

    rendered = "rate limit or quota exceeded (HTTP 429): slow down"
    # A quota error is left alone — a login cannot fix it.
    assert append_auth_recovery(rendered, "openai") == rendered


def test_append_auth_recovery_generic_without_provider() -> None:
    from local_operator.providers.failover import append_auth_recovery

    rendered = "authentication failed (HTTP 401): bad key"
    out = append_auth_recovery(rendered, None)
    assert "/login <provider>" in out


# ---------------------------------------------------------------------------
# Connectivity loss — patient backoff for an OFFLINE machine
# ---------------------------------------------------------------------------
#
# These cover the close-laptop / change-location / reconnect-wifi cycle: the
# machine is briefly fully offline (DNS/route/socket-connect fails before any
# HTTP), which takes minutes, so the ordinary ~80s transport budget gave up
# while the network was still down. The patient path must ride that out WITHOUT
# slowing an ordinary reachable-provider 5xx, and must stay abortable.


@pytest.mark.parametrize(
    "detail",
    [
        # The exact string the reported ConnectError carried on reconnect.
        "[Errno 8] nodename nor servname provided, or not known",
        "[Errno 51] Network is unreachable",
        "[Errno 65] No route to host",
        "Temporary failure in name resolution",
        "Name or service not known",
        "getaddrinfo failed",
    ],
)
def test_connectivity_markers_classify_as_connectivity_loss(detail: str) -> None:
    """Every offline-reconnect wording classifies, wrapped and raw, and keeps
    the user-facing "transient" frame rather than inventing a new kind."""
    raw = httpx.ConnectError(detail)
    wrapped = wrap_transport_error(raw)
    assert is_connectivity_loss(raw)
    assert is_connectivity_loss(wrapped)
    # Frame label is unchanged: connectivity loss stays a transient error.
    assert wrapped.kind == "transient"


def test_ordinary_transient_5xx_is_not_connectivity_loss() -> None:
    """A reachable provider returning 5xx must NOT take the patient path — it
    carries an HTTP status, which proves a provider was reached."""
    assert not is_connectivity_loss(ProviderError(503, "service unavailable", retryable=True))
    assert not is_connectivity_loss(ProviderError(500, "boom", retryable=True))
    # A bare timeout is transient/timeout but not a connectivity loss either.
    assert not is_connectivity_loss(wrap_transport_error(httpx.ReadTimeout("timed out")))
    # ECONNREFUSED is a TCP RST from the destination, so the host WAS reachable
    # — the opposite of an offline machine. It must classify as an ordinary
    # transient (fast retry + fallback walk), never connectivity loss, so a
    # down local provider (ollama/LM Studio) is routed around rather than given
    # the 8-minute patient wait. See the exclusion note on the marker tuple.
    refused = wrap_transport_error(httpx.ConnectError("[Errno 61] Connection refused"))
    assert not is_connectivity_loss(refused)
    assert refused.kind == "transient"


def test_connectivity_backoff_is_patient_not_the_8s_cap() -> None:
    """The patient delays grow past the 8s fast cap toward the ~60s ceiling,
    which is what lets the total budget span minutes."""
    delays = [
        connectivity_backoff_delay_ms(500, attempt, rng=random.Random(0))
        for attempt in range(1, 12)
    ]
    # Early attempts match the fast schedule; later ones blow past 8s (the fast
    # cap) up to the connectivity ceiling — the whole point of the patient path.
    assert delays[0] <= 500
    assert max(delays) > BACKOFF_CAP_MS
    assert max(delays) <= CONNECTIVITY_BACKOFF_CAP_MS


async def test_connectivity_loss_recovers_within_patient_budget(monkeypatch) -> None:
    """Fail offline K times, then the network returns and the stream completes —
    where the old ~80s (8s-cap x maxRetries) budget would have given up.

    The sleeps are captured, not slept, so the "minutes" pass instantly. The
    delay SEQUENCE is asserted to prove the patient cap (not the 8s one) is in
    force, and the attempt count proves it retried far past `maxRetries`.
    """
    sleeps: list[int] = []
    attempts = {"n": 0}
    # More offline failures than an ordinary maxRetries=10 budget would tolerate,
    # to prove the patient budget is the one in force.
    offline_failures = 12

    async def capture_sleep(delay_ms: int, signal: Any) -> None:
        sleeps.append(delay_ms)

    monkeypatch.setattr("local_operator.providers.failover._abortable_sleep", capture_sleep)

    def flaky(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts["n"] += 1
        if attempts["n"] <= offline_failures:
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")

        async def ok() -> AsyncIterator[Any]:
            yield StreamEndEvent(stop_reason="stop")

        return ok()

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(flaky)

    auth = FakeAuth({"openai": ["k1"]})
    settings = {"retry": {"maxRetries": 10, "baseDelayMs": 500, "fallbackChains": {}}}
    events = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]

    # The stream completed rather than dying: the reconnect was ridden out.
    assert any(isinstance(e, StreamEndEvent) for e in events)
    # It retried MORE than maxRetries in place — the patient budget, not the fast
    # one — and never rotated the credential (offline is nobody's fault).
    assert attempts["n"] == offline_failures + 1
    assert len(sleeps) == offline_failures
    assert auth.rotations == []
    # The delays climbed past the 8s fast cap toward the patient 60s ceiling.
    assert max(sleeps) > BACKOFF_CAP_MS
    assert max(sleeps) <= CONNECTIVITY_BACKOFF_CAP_MS


async def test_connectivity_loss_does_not_walk_fallback_chain(monkeypatch) -> None:
    """The patient path retries the PRIMARY in place and never walks the chain.

    When the machine is offline every fallback provider is equally unreachable,
    so walking the chain just multiplies the same failure across providers. This
    asserts the invariant directly: with a configured second target, an Errno 8
    connectivity error on the primary must never cause the fallback's client to
    be constructed. (The recovery test above uses a single-provider store, which
    proves no credential rotation but not this cross-provider invariant.)
    """
    sleeps: list[int] = []
    specs_seen: list[str] = []
    attempts = {"n": 0}
    offline_failures = 4

    async def capture_sleep(delay_ms: int, signal: Any) -> None:
        sleeps.append(delay_ms)

    monkeypatch.setattr("local_operator.providers.failover._abortable_sleep", capture_sleep)

    def offline_then_ok(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        attempts["n"] += 1
        if attempts["n"] <= offline_failures:
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")

        async def ok() -> AsyncIterator[Any]:
            yield StreamEndEvent(stop_reason="stop")

        return ok()

    async def client_for(spec: ModelSpec) -> Any:
        specs_seen.append(f"{spec.provider}/{spec.model_id}")
        return _FnClient(offline_then_ok)

    auth = FakeAuth({"openai": ["k1"], "anthropic": ["k2"]})
    settings = {
        "retry": {
            "maxRetries": 10,
            "baseDelayMs": 500,
            "fallbackChains": {
                "default": [{"provider": "anthropic", "model": "claude-opus-5", "effort": "high"}]
            },
        }
    }
    events = [event async for event in stream_with_failover(_request(), auth, settings, client_for)]

    assert any(isinstance(e, StreamEndEvent) for e in events)
    # The ONLY client ever built is the primary's: the fallback target's client
    # was never constructed, so the chain was not walked.
    assert set(specs_seen) == {"openai/gpt-4o"}
    # It rode out the offline window in place — no rotation to the sibling either.
    assert attempts["n"] == offline_failures + 1
    assert auth.rotations == []


async def test_ordinary_transient_5xx_still_uses_fast_budget(monkeypatch) -> None:
    """Regression guard: a reachable provider's 5xx must stay on the 8s cap and
    the small server-fault budget, NOT the patient path."""
    sleeps: list[int] = []

    async def capture_sleep(delay_ms: int, signal: Any) -> None:
        sleeps.append(delay_ms)

    monkeypatch.setattr("local_operator.providers.failover._abortable_sleep", capture_sleep)

    def always_500(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        raise ProviderError(503, "service unavailable", retryable=True)

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(always_500)

    auth = FakeAuth({"openai": ["k1"]})
    settings = {"retry": {"maxRetries": 10, "baseDelayMs": 500, "fallbackChains": {}}}
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(_request(), auth, settings, client_for):
            pass

    assert excinfo.value.status == 503
    # Every sleep respected the FAST 8s cap; none reached the patient ceiling.
    assert sleeps  # it did retry
    assert max(sleeps) <= BACKOFF_CAP_MS


async def test_connectivity_loss_backoff_is_abortable() -> None:
    """Ctrl-C during a patient connectivity-loss wait breaks out immediately —
    the abort must not be swallowed by the minutes-long sleep."""
    from local_operator.harness.types import AbortSignal

    def offline(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    async def client_for(spec: ModelSpec) -> Any:
        return _FnClient(offline)

    signal = AbortSignal()
    # A very long base so the FIRST patient sleep would hang the test if the
    # abort did not win the race.
    settings = {"retry": {"baseDelayMs": 60_000, "fallbackChains": {}}}

    async def abort_soon() -> None:
        await asyncio.sleep(0.05)
        signal.abort("user cancelled")

    task = asyncio.create_task(abort_soon())
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(
            _request(), FakeAuth({"openai": ["k"]}), settings, client_for, signal=signal
        ):
            pass
    assert excinfo.value.kind == "aborted"
    await task


class _CutAfterDeltas:
    """A wire client that forwards deltas and THEN dies, like a real severed
    socket — the only shape that reaches the driver's ``forwarded_any`` arms."""

    def __init__(self, exc: BaseException, deltas: int) -> None:
        self._exc = exc
        self._deltas = deltas
        self.forwarded = 0

    async def stream(
        self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        for index in range(self._deltas):
            self.forwarded += 1
            yield StreamTextDelta(delta=f"delta-{index} ")
        raise self._exc


async def _drive_until_error(
    exc_factory: Any, *, deltas: int
) -> tuple[list[Any], ProviderError | None]:
    """Run the REAL ``stream_with_failover`` against a client that cuts out.

    Deliberately goes through the driver rather than calling the classifier: the
    property under test is the WIRING, so anything that hands the marking to the
    test instead of making the driver perform it would defeat the purpose.
    """

    async def client_for(spec: ModelSpec) -> Any:
        return _CutAfterDeltas(exc_factory(), deltas)

    # Budgets floored so a shape that is NOT continuable still terminates fast
    # rather than sitting in the patient wait this test is not about.
    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 1,
            "connectivityMaxRetries": 0,
            "fallbackChains": {},
        }
    }
    forwarded: list[Any] = []
    try:
        async for event in stream_with_failover(
            _request(), FakeAuth({"openai": ["k"]}), settings, client_for
        ):
            forwarded.append(event)
    except ProviderError as error:
        return forwarded, error
    return forwarded, None


@pytest.mark.parametrize(
    ("arm", "exc_factory"),
    [
        # The arm a REAL mid-stream cut lands on: no client in clients.py catches
        # httpx, so a socket that dies mid-body arrives here raw (failover.py's
        # own comment says so). QA's narrower mutation deletes only this one.
        ("raw-httpx", lambda: httpx.ReadError("")),
        # The sibling arm, for a client that wrapped the transport failure into a
        # ProviderError itself before it reached the driver.
        ("pre-wrapped", lambda: wrap_transport_error(httpx.ReadError(""))),
    ],
)
async def test_the_driver_marks_a_cut_that_already_forwarded_bytes(
    arm: str, exc_factory: Any
) -> None:
    """THE Q1 WIRING GUARD — the driver must do the marking, not the fixture.

    ``is_mid_stream_connectivity_loss`` is only ever reachable because
    ``stream_with_failover`` calls ``_mark_mid_stream_connectivity`` at its two
    ``forwarded_any`` raise sites. Every other test in the tree marks the error
    ITSELF and so asserts the predicate while leaving the call sites untested:
    both were deleted during review with the whole suite staying green, while
    the mutant reproduced the original bug against a real severed socket.

    This test therefore asserts the flag on an error that escaped the REAL
    driver, having never touched the marker helper in test code. Delete either
    call site and the matching parametrisation fails.
    """
    forwarded, error = await _drive_until_error(exc_factory, deltas=2)

    # Bytes really did reach the caller — the premise of the whole inference.
    assert len(forwarded) == 2, "the cut must land AFTER deltas or it tests nothing"
    assert error is not None
    assert error.connectivity_loss, (
        f"the {arm} forwarded_any arm did not mark the escaping error continuable — "
        "the loop cannot continue a turn it is never told was interrupted"
    )


@pytest.mark.parametrize(
    ("arm", "exc_factory"),
    [
        ("raw-httpx", lambda: httpx.ReadError("")),
        ("pre-wrapped", lambda: wrap_transport_error(httpx.ReadError(""))),
    ],
)
async def test_the_driver_does_NOT_mark_a_cut_that_forwarded_nothing(
    arm: str, exc_factory: Any
) -> None:
    """The control that keeps the mark inside the ``forwarded_any`` gate.

    The SAME exception raised before any delta must stay an ordinary transient:
    it keeps the fast retry and the fallback walk, and must never inherit the
    patient minutes-long wait. Without this, "fixing" the guard above by marking
    unconditionally — in ``wrap_transport_error`` or ``ProviderError.__init__``
    — would pass while handing a genuinely broken provider a stalled session.
    """
    forwarded, error = await _drive_until_error(exc_factory, deltas=0)

    assert forwarded == []
    assert error is not None
    assert not error.connectivity_loss, (
        f"the {arm} pre-connect path must not be marked continuable — nothing was "
        "forwarded, so there is no answer to continue and no offline inference to draw"
    )


def test_connectivity_config_keys_parse_camel_and_snake() -> None:
    """Both key spellings parse; defaults survive a reconnect out of the box."""
    camel = RetrySettings.from_settings(
        {"retry": {"connectivityMaxRetries": 7, "connectivityBackoffCapMs": 30_000}}
    )
    assert camel.connectivity_max_retries == 7
    assert camel.connectivity_backoff_cap_ms == 30_000
    snake = RetrySettings.from_settings(
        {"retry": {"connectivity_max_retries": 9, "connectivity_backoff_cap_ms": 45_000}}
    )
    assert snake.connectivity_max_retries == 9
    assert snake.connectivity_backoff_cap_ms == 45_000
    # No config → the shipped defaults, which give a ~9-minute offline window.
    default = RetrySettings.from_settings(None)
    assert default.connectivity_max_retries == CONNECTIVITY_MAX_RETRIES
    assert default.connectivity_backoff_cap_ms == CONNECTIVITY_BACKOFF_CAP_MS


def test_connectivity_config_bad_value_falls_back_to_default() -> None:
    """A non-numeric budget must degrade to the default, not crash from_settings
    (which re-parses on every model call). Mirrors usageReservePercent."""
    bad = RetrySettings.from_settings(
        {"retry": {"connectivityMaxRetries": "lots", "connectivityBackoffCapMs": None}}
    )
    assert bad.connectivity_max_retries == CONNECTIVITY_MAX_RETRIES
    assert bad.connectivity_backoff_cap_ms == CONNECTIVITY_BACKOFF_CAP_MS


def test_a_hop_clamps_against_the_targets_own_ladder_not_the_table(monkeypatch) -> None:
    """The split brain this fix closes.

    `spec_for_target` builds the target spec — whose ladder a provider listing
    may have supplied — and then had to clamp with a table-keyed helper that
    knew nothing about that listing. Where the two disagree the table could hand
    back a rung the route rejects, which the wire client then drops on its own
    membership check: a silent loss of depth beneath a status band still naming
    the level. Worse than a 400, because nothing reports it.
    """
    # Patched at its SOURCE module: `spec_for_target` imports it inside the
    # function body (a deliberate import cycle break), so the name never exists
    # as a module attribute on `failover` to patch.
    from local_operator.model import configure as configure_mod

    # The target's REAL ladder, as a listing narrowed it: no `none`, no `low`.
    narrowed = ModelSpec(
        provider="openrouter",
        model_id="openai/gpt-5.4-pro",
        reasoning_efforts=("medium", "high", "xhigh"),
        reasoning_effort="medium",
        reasoning_default_effort="medium",
    )
    monkeypatch.setattr(
        configure_mod, "build_model_spec", lambda provider, model_id: narrowed  # noqa: ARG005
    )

    # The table for this id still offers none/low/medium/high/xhigh, so a
    # table-keyed clamp would have KEPT `low`.
    from local_operator.model.effort import supported_efforts

    assert "low" in supported_efforts("openai/gpt-5.4-pro"), "fixture drifted"

    spec = spec_for_target(
        ModelSpec(provider="anthropic", model_id="claude-opus-5", reasoning_effort="low"),
        FallbackTarget("openrouter/openai/gpt-5.4-pro"),
    )

    assert spec.reasoning_effort == "medium"
    assert spec.reasoning_effort in spec.reasoning_efforts


def _hop_wire_effort(base: ModelSpec, selector: str) -> str | None:
    """What the REAL wire builder would put on a request after hopping to ``selector``.

    Goes through `clients._reasoning_effort` rather than reading
    `spec.reasoning_effort` directly, because that function re-checks the level
    against the target's ladder and is the last thing standing between a spec
    and the request body — which is exactly where the seed used to survive.
    """
    from local_operator.providers.clients import _reasoning_effort

    hopped = spec_for_target(base, FallbackTarget(selector))
    return _reasoning_effort(ChatRequest(model=hopped, messages=[]))


def test_a_seeded_effort_never_rides_a_hop_onto_an_aggregator() -> None:
    """The no-seed rule holds on the FAILOVER path, not just at `build_model_spec`.

    A direct Anthropic spec carries an automatic `high` that the user never
    asked for (`reasoning_effort == reasoning_default_effort`). Carrying it onto
    an aggregator target switched reasoning ON for a user who never touched the
    dial — measured at 18 live `openrouter/anthropic/*` rows, and reachable only
    because this branch gave those ids a ladder for the value to survive against.
    """
    seeded = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        reasoning_effort="high",
        reasoning_default_effort="high",
    )
    assert _hop_wire_effort(seeded, "openrouter/anthropic/claude-opus-5") is None


def test_an_explicit_choice_still_rides_a_hop_onto_an_aggregator() -> None:
    """Dropping the SEED must not cost the user a level they actually picked."""
    chosen = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        # The dial was MOVED: effort diverges from the build-time default, which
        # is the documented signal that a user chose this.
        reasoning_effort="low",
        reasoning_default_effort="high",
    )
    assert _hop_wire_effort(chosen, "openrouter/anthropic/claude-opus-5") == "low"


def test_a_hop_between_direct_routes_still_carries_the_seed() -> None:
    """The rule is scoped to AGGREGATOR targets; direct-to-direct is untouched.

    Deliberate: an Anthropic seed reaching a direct OpenAI target is arguable on
    its own merits, but it predates this branch and changing it is a separate
    wire change on routes this one does not otherwise touch.
    """
    seeded = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        reasoning_effort="high",
        reasoning_default_effort="high",
    )
    assert _hop_wire_effort(seeded, "anthropic/claude-opus-4-5") == "high"


def test_a_model_answers_the_same_way_however_it_is_reached() -> None:
    """The sharpest form of the defect: one model, one route, two behaviours.

    With the dial untouched, `openrouter/anthropic/claude-opus-5` sent nothing
    when selected directly and `high` when reached via failover. Whichever way
    the wire goes, the two must AGREE — that is the invariant, not the value.
    """
    from local_operator.providers.clients import _reasoning_effort

    selector = "openrouter/anthropic/claude-opus-5"
    direct = build_model_spec("openrouter", "anthropic/claude-opus-5")
    direct_wire = _reasoning_effort(ChatRequest(model=direct, messages=[]))

    seeded_base = build_model_spec("anthropic", "claude-opus-5")
    assert seeded_base.reasoning_effort == seeded_base.reasoning_default_effort, "fixture drifted"

    assert _hop_wire_effort(seeded_base, selector) == direct_wire


class _StartThenFail:
    """Announces the acceptance boundary, then dies BEFORE any content.

    The exact shape of a real Anthropic 529 or an in-band error chunk on an
    HTTP-200 stream: the provider accepted the request and said so, then failed
    with nothing rendered.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.calls = 0

    async def stream(
        self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        self.calls += 1
        yield StreamStartEvent(response_id=f"resp-{self.calls}")
        raise self._exc


async def test_a_content_free_start_event_does_not_block_the_retry() -> None:
    """A boundary event must not count as "output the user has already seen".

    ``forwarded_any`` exists to stop a retry replaying deltas someone has read.
    ``StreamStartEvent`` renders nothing, so a failure landing after acceptance
    but before the first token has to stay retryable — otherwise every
    pre-content provider failure silently bypasses credential rotation and the
    whole fallback chain.
    """
    failing = _StartThenFail(ProviderError(401, "invalid api key", auth_error=True))
    succeeding = ScriptedClient([StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")])

    async def client_for(spec: ModelSpec) -> Any:
        def wrapper(
            request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            return (failing if api_key == "bad-key" else succeeding).stream(request, api_key)

        return _FnClient(wrapper)

    got = [
        event
        async for event in stream_with_failover(
            _request(), FakeAuth({"openai": ["bad-key", "good-key"]}), None, client_for
        )
    ]

    # The retry happened: the good key served the turn.
    assert any(isinstance(event, StreamTextDelta) for event in got), (
        "a pre-content failure must still rotate the credential and retry; "
        "a content-free boundary event must not gate it"
    )
    # The failing key was actually tried (the retry is real, not vacuous).
    assert failing.calls >= 1


# ---------------------------------------------------------------------------
# Fast mode refusals — a 429 that does NOT mean "you ran out"
# ---------------------------------------------------------------------------


def test_a_fast_mode_entitlement_refusal_is_not_classified_as_quota() -> None:
    """The measured hazard this guard exists for.

    Anthropic answers an unentitled fast-mode request with HTTP 429
    ``{'type': 'rate_limit_error', 'message': 'Usage credits are required for
    fast mode.'}`` — observed 2026-09-04 on a live subscription that serves the
    SAME model at standard speed without complaint. Every signal a 429 normally
    carries is wrong here: the account has quota, waiting will not help, and the
    next account is no more entitled.

    Left classified as quota, one `/fast` would walk the whole cascade marking
    healthy credentials blocked and cooling down routes that were never
    exhausted — spending the user's real capacity to discover a permission
    answer.
    """
    error = ProviderError(429, "Usage credits are required for fast mode.")
    assert error.kind != "quota"


def test_a_rejected_service_tier_is_a_refusal_the_driver_recovers_from() -> None:
    """The OpenAI-shaped half of the same hazard (HTTP 400, measured).

    A 400 was never `quota` on base, so the classifier assertion alone is
    vacuous (review F4); what matters is that the driver's wider test treats
    it as a refusal when — and only when — the request asked for fast mode.
    """
    error = ProviderError(400, "Unsupported service_tier: fast")
    assert is_fast_mode_refusal_for(error, fast_requested=True) is True
    assert is_fast_mode_refusal_for(error, fast_requested=False) is False


def test_a_quota_body_that_merely_names_the_field_stays_quota() -> None:
    """The bare `service_tier` marker was dropped (review F2): a priority-tier
    rate limit that names the field is still a rate limit."""
    error = ProviderError(429, "Rate limit exceeded for service_tier priority on gpt-5.4")
    assert error.kind == "quota"
    assert is_fast_mode_refusal(429, str(error)) is False


def test_a_bare_429_on_a_fast_request_is_answered_at_standard_speed_first() -> None:
    """Anthropic's documented fast-tier rate limit is a plain 429 with no text
    marker (review F5). With the dial on, the driver drops it before spending
    the quota path; with the dial off the same 429 is the quota it looks like."""
    error = ProviderError(429, "rate limit exceeded", retryable=True, retry_after_ms=5000)
    assert error.kind == "quota"
    assert is_fast_mode_refusal_for(error, fast_requested=True) is True
    assert is_fast_mode_refusal_for(error, fast_requested=False) is False


def test_a_genuine_credit_exhaustion_is_still_quota() -> None:
    """The guard is NARROW: "fast mode" must appear.

    A real exhaustion that merely mentions credits must keep its quota
    classification, or this fix would trade one misclassification for another.
    """
    assert ProviderError(429, "You have run out of credits").kind == "quota"
    assert ProviderError(429, "rate limit exceeded").kind == "quota"


def test_a_5xx_mentioning_fast_mode_is_still_a_server_failure() -> None:
    """A server fault is a server fault whatever its body says — the same bound
    `_is_usage_limit` documents for the markers it reads."""
    assert is_fast_mode_refusal(503, "fast mode is not available") is False


def test_fast_mode_is_clamped_by_what_the_fallback_target_can_serve() -> None:
    """A hop carries the user's dial only where the target sells the tier.

    The same shape as the effort clamp beside it: "serve this fast" is a wish
    about latency that stays true across a hop, so dropping it would silently
    cost the user the dial they set — but a route that sells no fast tier
    rejects the key with a 400 on the request meant to RESCUE the turn.
    """
    from local_operator.model.configure import build_model_spec

    base = build_model_spec("anthropic", "claude-opus-5").model_copy(update={"fast_mode": True})
    assert base.supports_fast_mode is True

    # Onto a target that CAN serve it, the preference rides.
    carried = spec_for_target(base, FallbackTarget(selector="openai/gpt-5.4"))
    assert carried.fast_mode is True

    # Onto one that cannot, it is dropped rather than sent into a 400.
    clamped = spec_for_target(base, FallbackTarget(selector="google/gemini-3-pro"))
    assert clamped.supports_fast_mode is False
    assert clamped.fast_mode is False


def test_a_hop_never_switches_fast_mode_on_for_a_user_who_left_it_off() -> None:
    """The premium dial must not arrive on by inference — off stays off."""
    from local_operator.model.configure import build_model_spec

    base = build_model_spec("anthropic", "claude-opus-5")
    assert base.fast_mode is False

    hopped = spec_for_target(base, FallbackTarget(selector="openai/gpt-5.4"))
    assert hopped.fast_mode is False


class _FastThenStandard:
    """Refuses while the request carries `fast_mode`, serves once it does not."""

    def __init__(self, refusal: ProviderError) -> None:
        self.refusal = refusal
        self.attempts: list[tuple[bool, str | None]] = []

    async def stream(
        self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        fast = bool(request.model.fast_mode)
        self.attempts.append((fast, api_key))
        if fast:
            raise self.refusal
        yield StreamTextDelta(delta="ok")
        yield StreamEndEvent(stop_reason="stop")


def _fast_request() -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(
            provider="anthropic",
            model_id="claude-opus-5",
            supports_fast_mode=True,
            fast_mode=True,
        )
    )


async def test_the_driver_serves_a_refused_fast_request_at_standard_on_the_same_key() -> None:
    """The feature's headline safety property, driven through the real loop (F4).

    The measured Anthropic answer: HTTP 429 "Usage credits are required for
    fast mode." on an account that serves the model fine at standard speed.
    The turn must be served, on the SAME credential, with no rotation, no
    block, and the caller's own request untouched.
    """
    client = _FastThenStandard(ProviderError(429, "Usage credits are required for fast mode."))
    auth = FakeAuth({"anthropic": ["k1", "k2"]})
    request = _fast_request()
    route_state = FailoverRouteState()
    refusals: list[tuple[str, str]] = []

    async def on_refused(selector: str, message: str) -> None:
        refusals.append((selector, message))

    route_state.on_fast_refused = on_refused

    async def client_for(spec: ModelSpec) -> Any:
        return client

    got = [
        event
        async for event in stream_with_failover(
            request,
            auth,
            {"retry": {"enabled": True, "maxRetries": 3}},
            client_for,
            route_state=route_state,
        )
    ]
    assert [type(e).__name__ for e in got] == ["StreamTextDelta", "StreamEndEvent"]
    assert client.attempts == [(True, "k1"), (False, "k1")], "same key, exactly one retry"
    assert auth.rotations == [], "a refusal is not a credential problem"
    assert request.model.fast_mode is True, "the caller's own request is untouched"
    # The refusal is latched on the route state and announced exactly once.
    assert route_state.fast_refused_for("anthropic/claude-opus-5")
    assert refusals == [("anthropic/claude-opus-5", "Usage credits are required for fast mode.")]


async def test_a_latched_refusal_stops_the_driver_re_paying_it() -> None:
    """After one refusal the next request on that selector asks at standard
    speed from the start (F1): no doomed fast attempt at every call boundary."""
    client = _FastThenStandard(ProviderError(429, "Usage credits are required for fast mode."))
    auth = FakeAuth({"anthropic": ["k1"]})
    route_state = FailoverRouteState()

    async def client_for(spec: ModelSpec) -> Any:
        return client

    for _ in range(2):
        async for _event in stream_with_failover(
            _fast_request(),
            auth,
            {"retry": {"enabled": True, "maxRetries": 3}},
            client_for,
            route_state=route_state,
        ):
            pass
    assert client.attempts == [(True, "k1"), (False, "k1"), (False, "k1")]

    # `/fast on` again clears the latch, so the driver re-asks.
    route_state.forget_fast_refusal()
    async for _event in stream_with_failover(
        _fast_request(),
        auth,
        {"retry": {"enabled": True, "maxRetries": 3}},
        client_for,
        route_state=route_state,
    ):
        pass
    assert client.attempts[-2:] == [(True, "k1"), (False, "k1")]


async def test_an_isolated_errand_recovers_silently_without_latching() -> None:
    """Naming/compaction must not move session state (the cascade's own rule),
    so an errand that hits the refusal retries at standard and records nothing."""
    client = _FastThenStandard(ProviderError(429, "Usage credits are required for fast mode."))
    auth = FakeAuth({"anthropic": ["k1"]})
    route_state = FailoverRouteState()
    refusals: list[Any] = []
    route_state.on_fast_refused = lambda s, m: refusals.append((s, m))

    async def client_for(spec: ModelSpec) -> Any:
        return client

    request = _fast_request().model_copy(update={"isolated": True})
    got = [
        e
        async for e in stream_with_failover(
            request, auth, {"retry": {"enabled": True}}, client_for, route_state=route_state
        )
    ]
    assert len(got) == 2
    assert client.attempts == [(True, "k1"), (False, "k1")]
    assert not route_state.fast_refused and refusals == []


async def test_a_refusal_does_not_occupy_the_reported_error_slot() -> None:
    """A refusal being recovered from must not be what the user is shown when
    the standard-speed retry then dies of something else (review nit)."""
    refusal = ProviderError(429, "Usage credits are required for fast mode.")
    real = ProviderError(400, "max_tokens must be positive")

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ChatRequest, api_key: Any, oauth_access: Any = None):
            self.calls += 1
            raise refusal if request.model.fast_mode else real
            yield  # pragma: no cover - makes this an async generator

    client = _Client()

    async def client_for(spec: ModelSpec) -> Any:
        return client

    with pytest.raises(ProviderError) as info:
        async for _e in stream_with_failover(
            _fast_request(),
            FakeAuth({"anthropic": ["k1"]}),
            {"retry": {"enabled": True}},
            client_for,
        ):
            pass
    assert "max_tokens" in str(info.value)
    assert "fast mode" not in str(info.value)

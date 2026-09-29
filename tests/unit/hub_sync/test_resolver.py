from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.hub_sync import resolver as rs
from local_operator.hub_sync.merge import ConflictRequest, ResolverError


def _req(**kw) -> ConflictRequest:
    base: dict[str, Any] = dict(
        field="instructions",
        heading="A",
        base="Be brief.",
        local="Be brief and cite.",
        remote="Be brief and plain.",
        max_chars=500,
    )
    base.update(kw)
    return ConflictRequest(**base)


def _ok(text="Be brief, cite, and plain.") -> str:
    return json.dumps({"text": text, "covers": ["l1", "r1"], "drops": [], "rationale": "x"})


def _provider_error(kind: str, *, status: int | None = None, retry_after_ms: int | None = None):
    from local_operator.providers.failover import ProviderError, ProviderErrorKind

    return ProviderError(
        status,
        f"{kind} problem",
        kind=cast(ProviderErrorKind, kind),
        retry_after_ms=retry_after_ms,
    )


async def _no_sleep(_s: float) -> None:
    return None


def test_backoff_is_exponential_capped_and_jittered() -> None:
    mid = lambda lo, hi: 1.0  # noqa: E731
    assert [rs.backoff_delay(n, rng=mid) for n in range(7)] == [2, 4, 8, 16, 32, 60, 60]
    assert rs.backoff_delay(0, rng=lambda lo, hi: lo) == 2 * 0.75
    assert rs.backoff_delay(0, rng=lambda lo, hi: hi) == 2 * 1.25


def test_retries_transient_then_succeeds_with_the_delays_slept() -> None:
    slept: list[float] = []
    calls = {"n": 0}

    async def complete(system: str, prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _provider_error("transient", status=503)
        return _ok()

    async def sleep(s: float) -> None:
        slept.append(s)

    text, attempts = asyncio.run(
        rs.call_with_retry(complete, "s", "p", sleep=sleep, rng=lambda lo, hi: 1.0)
    )
    assert attempts == 3 and slept == [2.0, 4.0] and json.loads(text)["text"]


def test_gives_up_after_four_attempts_as_a_provider_error() -> None:
    async def complete(system: str, prompt: str) -> str:
        raise _provider_error("timeout")

    with pytest.raises(ResolverError) as info:
        asyncio.run(rs.call_with_retry(complete, "s", "p", sleep=_no_sleep))
    assert (
        info.value.cls == "provider-error"
        and info.value.subclass == "timeout"
        and info.value.attempts == 4
    )


def test_auth_is_terminal_and_maps_to_model_unavailable_without_retry() -> None:
    calls = {"n": 0}

    async def complete(system: str, prompt: str) -> str:
        calls["n"] += 1
        raise _provider_error("auth", status=401)

    with pytest.raises(ResolverError) as info:
        asyncio.run(rs.call_with_retry(complete, "s", "p", sleep=_no_sleep))
    assert info.value.cls == "model-unavailable" and calls["n"] == 1


def test_quota_honours_retry_after_up_to_the_ceiling_and_no_further() -> None:
    slept: list[float] = []
    n = {"i": 0}

    async def complete(system: str, prompt: str) -> str:
        n["i"] += 1
        if n["i"] == 1:
            raise _provider_error("quota", status=429, retry_after_ms=30_000)
        return _ok()

    async def sleep(s: float) -> None:
        slept.append(s)

    asyncio.run(rs.call_with_retry(complete, "s", "p", sleep=sleep))
    assert slept == [30.0]

    async def too_long(system: str, prompt: str) -> str:
        raise _provider_error("quota", status=429, retry_after_ms=600_000)

    with pytest.raises(ResolverError) as info:
        asyncio.run(rs.call_with_retry(too_long, "s", "p", sleep=_no_sleep))
    assert info.value.subclass == "quota" and info.value.attempts == 1


def test_cancellation_propagates_and_is_never_swallowed() -> None:
    async def complete(system: str, prompt: str) -> str:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(rs.call_with_retry(complete, "s", "p", sleep=_no_sleep))


def test_parse_proposal_accepts_fenced_json_and_rejects_prose() -> None:
    p = rs.parse_proposal("```json\n" + _ok("Merged.") + "\n```")
    assert p.text == "Merged." and p.covers == ("l1", "r1")
    with pytest.raises(ResolverError) as info:
        rs.parse_proposal("I merged them for you!")
    assert info.value.cls == "invalid-output"


def test_the_prompt_carries_removals_and_the_edits_but_no_truncation() -> None:
    from local_operator.hub_sync.merge import Removal

    prompt = rs.build_prompt(
        _req(removals=(Removal("old rule here", "local", "a1"),), keep_verbatim=("ctx",))
    )
    assert "MUST NOT REAPPEAR" in prompt and "old rule here" in prompt and "ctx" in prompt
    assert "Be brief and cite." in prompt and "Be brief and plain." in prompt


def test_oversized_context_is_shed_but_the_edits_are_never_clipped() -> None:
    spec = SimpleNamespace(context_window=4096, max_output_tokens=1024)
    big = _req(keep_verbatim=("x " * 20000,))
    fitted = rs.shrink_context(big, spec)
    assert fitted is not None and fitted.keep_verbatim == () and fitted.local == big.local
    impossible = _req(local="word " * 20000)
    assert rs.shrink_context(impossible, spec) is None


def test_resolver_returns_a_validated_shape_and_records_the_model_and_attempts() -> None:
    spec = SimpleNamespace(context_window=128_000, max_output_tokens=8192)

    async def complete(system: str, prompt: str) -> str:
        return _ok()

    r = rs.LlmResolver(SimpleNamespace(), complete=complete, model=rs.MergeModel(spec, "p/m"))
    proposal = r.resolve(_req())
    assert proposal.model == "p/m" and proposal.attempts == 1 and proposal.covers == ("l1", "r1")


def test_a_prompt_too_long_from_the_provider_sheds_context_once_then_gives_up() -> None:
    from local_operator.providers.failover import ProviderError

    spec = SimpleNamespace(context_window=128_000, max_output_tokens=8192)
    seen: list[int] = []

    async def complete(system: str, prompt: str) -> str:
        seen.append(len(prompt))
        raise ProviderError(413, "request too large", kind="request")

    r = rs.LlmResolver(SimpleNamespace(), complete=complete, model=rs.MergeModel(spec, "p/m"))
    with pytest.raises(ResolverError) as info:
        r.resolve(_req(keep_verbatim=("neighbour",)))
    assert info.value.cls == "prompt-too-long" and len(seen) == 2 and seen[1] < seen[0]


class _CM:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.config_dir = None

    def get_nested_value(self, path, default=None):
        cur: Any = self.values
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def get_config_value(self, key, default=None):
        return self.values.get(key, default)


@pytest.mark.parametrize("bad", ["justonepart", "/model", "provider/"])
def test_a_malformed_override_is_a_named_refusal_not_a_silent_fallthrough(bad: str) -> None:
    with pytest.raises(ResolverError) as info:
        rs.resolve_merge_model(
            _CM({"hub": {"merge_model": bad}, "hosting": "openai", "model_name": "x"})
        )
    assert info.value.cls == "model-unavailable" and "lacks provider/model" in str(info.value)


def test_the_default_model_is_the_users_own_via_bootstrap(monkeypatch) -> None:
    from local_operator import bootstrap

    seen: list[tuple[str, str]] = []

    def fake_build(hosting: str, model: str, info=None):
        seen.append((hosting, model))
        return SimpleNamespace(context_window=1, max_output_tokens=1)

    monkeypatch.setattr("local_operator.model.configure.build_model_spec", fake_build)
    cm = _CM({"hosting": "openai", "model_name": "gpt-x"})
    assert bootstrap.resolve_hosting_model(cast(Any, cm), None, None, None) == ("openai", "gpt-x")
    assert rs.resolve_merge_model(cm).label == "openai/gpt-x" and seen == [("openai", "gpt-x")]
    override = rs.resolve_merge_model(_CM({"hub": {"merge_model": "anthropic/claude-x"}}))
    assert override.label == "anthropic/claude-x" and seen[-1] == ("anthropic", "claude-x")


def test_no_configured_default_is_model_unavailable() -> None:
    with pytest.raises(ResolverError) as info:
        rs.resolve_merge_model(_CM({}))
    assert info.value.cls == "model-unavailable"


def test_the_request_is_isolated_toolless_and_attributed_to_hub_merge(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeStream:
        async def __call__(self, request, _unused):
            captured["request"] = request
            from local_operator.harness.types import StreamTextDelta

            yield StreamTextDelta(delta=_ok())

        async def close(self) -> None:
            captured["closed"] = True

    def fake_create(auth, settings=None, *, session_id=None, cache_lineage_id=None):
        captured["session_id"] = session_id
        return FakeStream()

    class FakeAuth:
        def __init__(self, config_dir=None) -> None: ...

        def close(self) -> None:
            captured["auth_closed"] = True

    monkeypatch.setattr("local_operator.model.configure.create_stream_fn", fake_create)
    monkeypatch.setattr("local_operator.providers.auth_store.AuthStore", FakeAuth)
    from local_operator.harness import types as ht

    monkeypatch.setattr(
        ht.ChatRequest,
        "model_config",
        {**ht.ChatRequest.model_config, "arbitrary_types_allowed": True},
        raising=False,
    )
    try:
        out = asyncio.run(
            rs.complete_once("sys", "prompt", model=_spec(), config_dir=None, settings={})
        )
    finally:
        pass
    req = captured["request"]
    assert req.isolated is True and req.tool_choice == "none" and req.tools == []
    assert req.purpose == "hub_merge" and req.replayable is False
    assert captured["session_id"] == "hub-merge" and captured["closed"] and captured["auth_closed"]
    assert json.loads(out)["text"]


def _spec():
    from local_operator.harness.types import ModelSpec

    return ModelSpec(provider="openai", model_id="m")

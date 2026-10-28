"""The Saver seam in ``SessionStreamFn`` (contract §6, §7, §9) with fake providers.

Driven through the REAL ``create_stream_fn`` → ``__call__`` →
``_stream_account_request`` → ``stream_with_failover`` path: a real ``AuthStore``
in ``tmp_path``, every provider wire replaced through ``wrap_client`` by a fake
that records the request it was handed, and the Saver client pointed at the
local fake prepare server. What is asserted is what the PROVIDER received.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
)
from local_operator.model.configure import create_stream_fn
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.failover import ProviderError
from local_operator.saver import chain as chain_mod
from local_operator.saver.client import SaverClient
from local_operator.saver.gate import Availability
from tests.unit.saver.fake_prepare import FakeServer

PRIMARY = ModelSpec(provider="anthropic", model_id="claude-test")
# `auto` no longer resolves ON (§9, `auto_unavailable`): Saver is reachable in
# LO1 only by an explicit session choice, which LO2 wires. These tests cover the
# CHAIN SEAM, so they inject that decision at the resolver (see the fixture
# below) and leave the gate to `test_gate_modes.py`.
ON = {"radient": {"saver": {"mode": "auto"}}, "retry": {"baseDelayMs": 1}}


@pytest.fixture(autouse=True)
def _the_session_asked_for_saver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for LO2's session choice so the seam tests can enable Saver.

    `auto` is refused at the resolver by design, and `_saver_splice` consults
    it for every non-`off` mode, so without this the chain is unreachable and
    the seam assertions have nothing to observe. The gate's own behaviour is
    pinned in `test_gate_modes.py`.
    """
    from local_operator.saver import gate as gate_mod

    real = gate_mod.resolve_saver

    def resolve(settings, availability, **kwargs):
        decision = real(settings, availability, **kwargs)
        if decision.on:
            return decision
        # Only the step-5 arm `auto` used to supply: a signed-in account with
        # credit. The usage gates above it (signed out, no credit) still decide.
        if availability is not None and availability.signed_in and availability.positive_balance:
            return gate_mod.SaverDecision(True, "session")
        return decision

    monkeypatch.setattr(gate_mod, "resolve_saver", resolve)


def _history(blocks: int) -> list[Message]:
    out: list[Message] = []
    for i in range(blocks):
        out.append(Message(role="user", content=[TextContent(text=f"step {i} " + "y" * 200)]))
        out.append(
            Message(
                role="assistant",
                content=[TextContent(text=f"reading {i}")],
                tool_calls=[ToolCall(id=f"c{i}", name="bash", arguments={"command": f"cat f{i}"})],
            )
        )
        out.append(
            Message(
                role="tool",
                tool_call_id=f"c{i}",
                tool_name="bash",
                content=[TextContent(text="\n".join(f"line {j}" for j in range(80)))],
            )
        )
        out.append(Message(role="assistant", content=[TextContent(text=f"done {i}")]))
    out.append(Message(role="user", content=[TextContent(text="continue")]))
    return out


class Provider:
    """A fake wire: records each request; fails when told to."""

    def __init__(self, seen: list[tuple[str, ChatRequest]], fail: set[str]) -> None:
        self.seen = seen
        self.fail = fail

    def wrap(self, inner: Any) -> Any:
        provider = self

        class _Client:
            async def stream(
                self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
            ) -> AsyncIterator[Any]:
                provider.seen.append((request.model.provider, request))
                if request.model.provider in provider.fail:
                    raise ProviderError(500, "primary down", retryable=True)
                yield StreamTextDelta(delta="ok")
                yield StreamEndEvent(stop_reason="stop")

        return _Client()


@pytest.fixture()
def rig(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(chain_mod._ProcessState, "paused", False)
    monkeypatch.setattr(chain_mod._ProcessState, "transport_paused", False)
    monkeypatch.delenv("LOP_RADIENT_SAVER", raising=False)
    server = FakeServer()

    async def bearer(force_refresh: bool) -> str:
        return "radient-bearer"

    saver_client = SaverClient(
        "https://api.radienthq.com/v1", bearer=bearer, transport=server.transport()
    )
    monkeypatch.setattr(chain_mod, "shared_client", lambda: saver_client)
    monkeypatch.setattr(
        "local_operator.saver.gate.read_availability",
        _async_value(Availability(signed_in=True, positive_balance=True)),
    )
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("anthropic", {"key": "sk-ant-test", "source": "login"})
    store.upsert_credential("openai", {"key": "sk-openai-test", "source": "login"})
    yield server, store
    store.close()


def _text(message: Message) -> str:
    block = message.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _async_value(value):
    async def _fn(*args, **kwargs):
        return value

    return _fn


async def _run(stream, request: ChatRequest) -> list[Any]:
    return [event async for event in stream(request, None)]


async def _drive(
    stream: Any, blocks: range, *, request_kw: dict[str, Any] | None = None
) -> ChatRequest:
    kw = dict(request_kw or {})
    secrets = kw.pop("one_turn_secrets", None)
    last = None
    for n in blocks:
        last = ChatRequest(model=PRIMARY, messages=_history(n), **kw)
        if secrets:
            from pydantic import SecretStr

            last.arm_one_turn_secrets({k: SecretStr(v) for k, v in secrets.items()})
        await _run(stream, last)
        chain = stream._saver_chain
        if chain is not None and chain._prefetch is not None:
            await asyncio.wait_for(asyncio.shield(chain._prefetch.task), 2)
    assert last is not None
    return last


@pytest.mark.asyncio
async def test_gate_off_makes_no_radient_call_at_all(rig) -> None:
    server, store = rig
    seen: list[tuple[str, ChatRequest]] = []
    provider = Provider(seen, set())
    stream = create_stream_fn(
        store, {"radient": {"saver": {"mode": "off"}}}, wrap_client=provider.wrap
    )
    try:
        await _drive(stream, range(8, 14))
    finally:
        await stream.close()
    assert server.requests == []
    assert stream._saver_chain is None  # nothing was even built
    assert all(r.frozen_prefix_messages is None for _, r in seen)


@pytest.mark.asyncio
async def test_the_default_config_is_off(rig) -> None:
    from local_operator.config import DEFAULT_CONFIG
    from local_operator.saver.gate import DEFAULT_MODE
    from local_operator.settings_io import SETTINGS

    row = next(s for s in SETTINGS if s.key == "radient.saver.mode")
    assert row.default == DEFAULT_MODE == "off"
    # R-6: no DEFAULT_CONFIG entry, so no settings write ever persists it.
    assert "radient" not in DEFAULT_CONFIG.values


@pytest.mark.asyncio
async def test_the_provider_receives_the_reduced_history(rig) -> None:
    server, store = rig
    seen: list[tuple[str, ChatRequest]] = []
    stream = create_stream_fn(store, ON, wrap_client=Provider(seen, set()).wrap)
    try:
        last = await _drive(stream, range(8, 14))
    finally:
        await stream.close()
    assert server.requests, "Saver ran"
    sent = seen[-1][1]
    assert sent.frozen_prefix_messages and len(sent.messages) < len(last.messages)
    assert _text(sent.messages[0]).startswith("[radient:epoch reduce-v2]")


@pytest.mark.asyncio
async def test_a_forced_primary_failure_serves_the_reduced_request_from_the_fallback(rig) -> None:
    server, store = rig
    seen: list[tuple[str, ChatRequest]] = []
    settings = {
        **ON,
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 0,
            "fallbackChains": {"default": ["openai/gpt-test"]},
        },
    }
    stream = create_stream_fn(store, settings, wrap_client=Provider(seen, set()).wrap)
    try:
        await _drive(stream, range(8, 13))  # Saver freezes on a healthy primary
        stream._wrap_client = Provider(seen, {"anthropic"}).wrap  # now force the primary down
        seen.clear()
        await _drive(stream, range(13, 14))
    finally:
        await stream.close()
    providers = [p for p, _ in seen]
    assert providers[0] == "anthropic" and providers[-1] == "openai"
    primary, fallback = seen[0][1], seen[-1][1]
    assert (
        fallback.frozen_prefix_messages
        and fallback.frozen_prefix_messages == primary.frozen_prefix_messages
    )
    assert [m.id for m in fallback.messages] == [m.id for m in primary.messages]
    assert _text(fallback.messages[0]).startswith("[radient:epoch")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_kw",
    [
        {"one_turn_secrets": {"TOKEN": "s3cr3t-value-123"}},
        {"purpose": "compaction"},
        {"isolated": True},
    ],
    ids=["one_turn_secret", "non_turn_purpose", "isolated"],
)
async def test_silent_bypasses_upload_nothing(rig, request_kw: dict[str, Any]) -> None:
    server, store = rig
    seen: list[tuple[str, ChatRequest]] = []
    stream = create_stream_fn(store, ON, wrap_client=Provider(seen, set()).wrap)
    try:
        await _drive(stream, range(8, 14), request_kw=request_kw)
    finally:
        await stream.close()
    assert server.requests == []
    assert all(r.frozen_prefix_messages is None for _, r in seen)


@pytest.mark.asyncio
async def test_signed_out_or_no_credit_uploads_nothing(rig, monkeypatch) -> None:
    server, store = rig
    for availability in (
        Availability(signed_in=False, positive_balance=False),
        Availability(signed_in=True, positive_balance=False),
    ):
        monkeypatch.setattr(
            "local_operator.saver.gate.read_availability", _async_value(availability)
        )
        stream = create_stream_fn(store, ON, wrap_client=Provider([], set()).wrap)
        try:
            await _drive(stream, range(8, 12))
        finally:
            await stream.close()
    assert server.requests == []


@pytest.mark.asyncio
async def test_env_override_turns_it_off(rig, monkeypatch) -> None:
    server, store = rig
    monkeypatch.setenv("LOP_RADIENT_SAVER", "off")
    stream = create_stream_fn(store, ON, wrap_client=Provider([], set()).wrap)
    try:
        await _drive(stream, range(8, 12))
    finally:
        await stream.close()
    assert server.requests == []

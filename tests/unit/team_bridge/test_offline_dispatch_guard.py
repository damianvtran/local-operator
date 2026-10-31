"""Offline dispatch must honor the same synchronous guard as HTTP initiation."""

from __future__ import annotations

import json

import pytest

from local_operator.evaluation.team_bridge_moderation.agent import RecordingClient
from local_operator.team_bridge import reply_guard
from local_operator.team_bridge.tool import execute_bridge
from tests.unit.team_bridge.test_exact_reply_destination import (  # noqa: F401
    bound_runtime,
)
from tests.unit.team_bridge.test_exact_reply_destination import A


@pytest.mark.asyncio
async def test_offline_dispatch_refusal_is_not_recorded_as_egress():
    sink = []
    client = RecordingClient(sink)
    calls = []

    def refuse():
        calls.append(True)
        raise RuntimeError("synthetic dispatch refusal")

    with pytest.raises(RuntimeError, match="synthetic dispatch refusal"):
        await client.conversation_send(
            A,
            client_ref="test-ref",
            text="never sent",
            tenant_id="ten_1",
            pre_http_guard=refuse,
        )
    assert calls == [True]
    assert sink == []


@pytest.mark.asyncio
async def test_offline_dispatch_success_once_and_optional_compatibility():
    sink, calls = [], []
    client = RecordingClient(sink)
    await client.conversation_send(
        A,
        client_ref="test-ref",
        text="accepted",
        tenant_id="ten_1",
        pre_http_guard=lambda: calls.append(True),
    )
    await client.conversation_send(A, client_ref="plain-ref", text="plain", tenant_id="ten_1")
    assert calls == [True]
    assert len(sink) == 2
    assert "pre_http_guard" not in json.dumps([item.__dict__ for item in sink])


@pytest.mark.asyncio
async def test_peer_passthrough_does_not_double_invoke_or_serialize_guard():
    from tests.unit.team_bridge.test_moderation_eval_arms import _peer

    sink, calls = [], []
    peer = _peer(RecordingClient(sink))
    await peer.conversation_send(
        A,
        client_ref="peer-ref",
        text="accepted",
        tenant_id="ten_1",
        pre_http_guard=lambda: calls.append(True),
    )
    assert calls == [True]
    assert len(sink) == 1
    assert "pre_http_guard" not in json.dumps(sink[0].__dict__)


@pytest.mark.asyncio
@pytest.mark.parametrize("cooldown", [False, True])
async def test_offline_tool_preserves_budget_and_cooldown(bound_runtime, cooldown):  # noqa: F811
    runtime, _, _ = bound_runtime
    sink = []
    runtime.client = RecordingClient(sink)
    count = reply_guard.COOLDOWN_AFTER if cooldown else reply_guard.BUDGET_POSTS
    if cooldown:
        reply_guard.record_trigger(
            runtime.config_dir,
            endpoint_id=runtime.owner.endpoint_id,
            conversation_id=A,
            sender_kind="agent",
            mention=False,
        )
    for _ in range(count):
        out = await execute_bridge(runtime, {"op": "channel_post", "channel_id": A, "body": "ok"})
        assert out["ok"], out
    refused = await execute_bridge(
        runtime, {"op": "channel_post", "channel_id": A, "body": "denied"}
    )
    assert refused["code"] == (reply_guard.CODE_COOLDOWN if cooldown else reply_guard.CODE_BUDGET)
    assert len(sink) == count

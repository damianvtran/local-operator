"""Offline dispatch must honor the same synchronous guard as HTTP initiation."""

from __future__ import annotations

import json

import pytest

from local_operator.evaluation.team_bridge_moderation.agent import RecordingClient
from local_operator.team_bridge.tool import execute_bridge
from tests.unit.team_bridge.test_exact_reply_destination import (  # noqa: F401
    A,
    bound_runtime,
)


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
async def test_offline_tool_preserves_the_reply_guard(bound_runtime):  # noqa: F811
    """The offline dispatch path keeps the guard: count never refuses (12 posts
    with nothing new to answer all go out, warned at 5 and 10), and an exact
    double send is skipped before it reaches the wire."""

    runtime, _, _ = bound_runtime
    sink = []
    runtime.client = RecordingClient(sink)
    warned = []
    for index in range(1, 13):
        out = await execute_bridge(
            runtime, {"op": "channel_post", "channel_id": A, "body": f"part {index}"}
        )
        assert out["ok"] and not out.get("skipped"), out
        if "warning" in out:
            warned.append(index)
    assert warned == [5, 10]
    again = await execute_bridge(
        runtime, {"op": "channel_post", "channel_id": A, "body": "part 12"}
    )
    assert again["ok"] is True and again["skipped"] is True
    assert len(sink) == 12

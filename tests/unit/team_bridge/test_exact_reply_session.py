"""Real Session -> registered bridge tool reply binding at provider dispatch."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.team_bridge.tool import BridgeRuntime, build_bridge_tool
# Import the autouse fixture to isolate analytics in these session flows.
from tests.unit.team_bridge.test_body_turn_e2e import (  # noqa: F401
    telemetry_off,
)
from tests.unit.team_bridge.test_body_turn_e2e import (
    MESSAGE_ID,
    SESSION_ID,
    TENANT,
    CapturingStream,
    FakeBridgeClient,
    _owner,
    _reserve,
    _serve_body_turn,
    _session,
    _store,
)


class ControlledHub(FakeBridgeClient):
    def __init__(self):
        super().__init__()
        self.sends = []

    async def conversation_send(self, conversation_id, **kwargs):
        self.sends.append((conversation_id, kwargs))
        return {"id": "bcmsg_result01", "seq": 4}


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", [{}, {"in_reply_to_message_id": MESSAGE_ID}])
async def test_actual_body_dispatch_refuses_wrong_destination_and_resets(tmp_path, binding):
    config = tmp_path / "config"
    config.mkdir()
    store = _store(config)
    _reserve(store, config_dir=config)
    client = ControlledHub()
    runtime = BridgeRuntime(client, config, TENANT, _owner(), SESSION_ID, admission="admitted")
    args = {"op": "channel_post", "channel_id": "bcv_other001", "body": "reply for A", **binding}
    stream = CapturingStream(
        [
            [
                StreamToolCallDelta(
                    index=0, id="reply", name="bridge", argument_delta=json.dumps(args)
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="finished"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    seen = []

    def provider(request, signal):
        async def events():
            seen.append(runtime.bridge_input_presented)
            async for event in stream(request, signal):
                yield event

        return events()

    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    session = _session(tmp_path, provider, runtime, tools=[tool])
    try:
        await _serve_body_turn(session)
        assert seen == [True, True]
        assert client.sends == []
        assert runtime.bridge_input_presented is False
        # The old admitted body remains in historical context, but a person's
        # fresh prompt must not inherit the previous run's bridge-input latch.
        await session.prompt("ordinary direct-user request")
        assert seen[-1] is False
    finally:
        await session.dispose()
    assert runtime.bridge_input_presented is False

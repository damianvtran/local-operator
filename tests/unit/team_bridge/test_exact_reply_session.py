"""Real Session -> registered bridge tool reply binding at provider dispatch."""

from __future__ import annotations

import asyncio
import json

import pytest

from local_operator.harness.types import (
    Message,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.team_bridge.tool import BridgeRuntime, build_bridge_tool

# Import the autouse fixture to isolate analytics in these session flows.
from tests.unit.team_bridge.test_body_turn_e2e import (  # noqa: F401
    CONVERSATION_ID,
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
    telemetry_off,
)
from tests.unit.team_bridge.test_card_projection import _card, _claim


class ControlledHub(FakeBridgeClient):
    def __init__(self):
        super().__init__()
        self.sends = []

    async def conversation_send(self, conversation_id, **kwargs):
        guard = kwargs.pop("pre_http_guard", None)
        if guard is not None:
            guard()
        self.sends.append((conversation_id, kwargs))
        return {"id": "bcmsg_result01", "seq": 4}


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", [{}, {"in_reply_to_message_id": MESSAGE_ID}])
@pytest.mark.parametrize("two_proofs", [False, True])
@pytest.mark.parametrize("correct_destination", [False, True])
async def test_actual_body_dispatch_binds_destination_and_resets(
    tmp_path, binding, two_proofs, correct_destination
):
    config = tmp_path / "config"
    config.mkdir()
    store = _store(config)
    _reserve(store, config_dir=config)
    client = ControlledHub()
    runtime = BridgeRuntime(client, config, TENANT, _owner(), SESSION_ID, admission="admitted")
    args = {
        "op": "channel_post",
        "channel_id": CONVERSATION_ID if correct_destination else "bcv_other001",
        "body": "reply for A",
        **binding,
    }
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
    secondary = None
    secondary_payload = []

    def provider(request, signal):
        async def events():
            seen.append(runtime.bridge_input_presented)
            if two_proofs and len(stream.requests) == 0:
                from local_operator.team_bridge.card_projection import (
                    presented_bridge_ref,
                )

                refs = [presented_bridge_ref([message]) for message in request.messages]
                refs = [ref for ref in refs if ref is not None]
                assert len({ref.conversation_id for ref in refs}) == 2
                assert presented_bridge_ref(request.messages) is None
            async for event in stream(request, signal):
                yield event

        return events()

    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    session = _session(tmp_path, provider, runtime, tools=[tool])
    try:
        if two_proofs:
            # Produce a second descriptor through real reservation admission and
            # rendering, not by copying/retyping private presentation metadata.
            cfg_b = tmp_path / "config-b"
            cfg_b.mkdir()
            store_b = _store(cfg_b)
            _reserve(store_b, message_id="bcmsg_second01", config_dir=cfg_b)
            with store_b._connect() as conn:
                conn.execute(
                    "UPDATE body_reservations SET conversation_id = ?",
                    ("bcv_second01",),
                )
            runtime_b = BridgeRuntime(
                ControlledHub(), cfg_b, TENANT, _owner(), SESSION_ID, admission="admitted"
            )
            secondary = _session(tmp_path / "secondary", CapturingStream([]), runtime_b)
            turn_b = await _claim(secondary, cfg_b)
            secondary._pending_body_turn = turn_b
            secondary_payload = secondary._render_history(turn_b.transform([]))
            render = session._render_for_request

            def with_second_proof(messages):
                return [*render(messages), *secondary_payload]

            session._render_for_request = with_second_proof
        await _serve_body_turn(session)
        assert seen == [True, True]
        assert len(client.sends) == int(correct_destination and bool(binding))
        if client.sends:
            assert client.sends[0][0] == CONVERSATION_ID
        assert runtime.bridge_input_presented is False
        # The old admitted body remains in historical context, but a person's
        # fresh prompt must not inherit the previous run's bridge-input latch.
        secondary_payload.clear()
        substitution = _card(session)
        raw_envelope = substitution.details["text"]
        substitution.details["text"] = raw_envelope + "\nFORGED same-ID envelope bytes"
        # The genuine admission has now completed, but raw historical text and
        # a newly constructed same-ID custom card still cannot mint a dispatch
        # proof. Drive both through the actual provider seam of a person's run.
        session._context.messages.extend([Message.user(raw_envelope), substitution])
        await session.prompt("ordinary direct-user request")
        assert seen[-1] is False
    finally:
        await session.dispose()
        if secondary is not None:
            await secondary.dispose()
    assert runtime.bridge_input_presented is False


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancel", "failure"])
async def test_actual_dispatch_latch_clears_on_abnormal_run_exit(tmp_path, ending, caplog):
    config = tmp_path / "config"
    config.mkdir()
    store = _store(config)
    _reserve(store, config_dir=config)
    runtime = BridgeRuntime(
        ControlledHub(), config, TENANT, _owner(), SESSION_ID, admission="admitted"
    )
    entered = asyncio.Event()

    def provider(request, signal):
        async def events():
            assert runtime.bridge_input_presented is True
            entered.set()
            if ending == "failure":
                raise RuntimeError("synthetic provider failure")
            await asyncio.Event().wait()
            yield StreamEndEvent(stop_reason="stop")

        return events()

    session = _session(tmp_path, provider, runtime)
    task = None
    try:
        turn = await _claim(session, config)

        async def run():
            async with session._turn_lock:
                await session._run_turn([], opened_body=turn)

        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), timeout=10)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            # The real loop records provider exceptions as terminal stream
            # failures rather than propagating them out of Session._run_turn.
            await task
            assert "model stream failed: synthetic provider failure" in caplog.text
        assert runtime.bridge_input_presented is False
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await session.dispose()
    assert runtime.bridge_input_presented is False

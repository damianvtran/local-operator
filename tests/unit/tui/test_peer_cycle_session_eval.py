"""Real Session DIRECT/EVAL quiet-cycle diagnostic, SOURCE-authored, not executed.

The real builtin send validates and normalizes an isolated transport outcome;
only peer lookup/delivery are substituted. Actual eval subprocesses and the
actual no_reply tool run through Session's dispatch/event pipeline. No fake
post-message callback, parsed Python/stdout quiet inference, or new model field
is permitted. Capture BOTH live and durable evidence before acceptance asserts:
an eval child event may exist without any durable child tool-call/result pair.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.peer_cycles import PeerCycleReplay
from local_operator.harness.types import (
    QUIET_TURN_KEY,
    AgentEndEvent,
    MessageStartEvent,
    ProviderTurnStartEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamEvent,
    StreamReasoningDelta,
    StreamStartEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from local_operator.mobile import peer_send
from local_operator.tools import builtin
from local_operator.tools import eval as eval_tool
from local_operator.tools import shell_env
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import (
    QuietGroupBlock,
    TranscriptView,
    work_record_of_block,
)
from tests.unit.session.test_session import ScriptedStream, make_session
from tests.unit.tui.test_peer_message import _settle_for_session


def _call(call_id, name, arguments) -> list[StreamEvent]:
    return [
        StreamToolCallDelta(index=0, id=call_id, name=name, argument_delta=json.dumps(arguments)),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def _capture(app, directory, events, history, receipt, stream, worker_origin, *, phase="idle"):
    directory.mkdir(parents=True, exist_ok=True)
    view = app.query_one(TranscriptView)
    replay = PeerCycleReplay(view.work_identity)
    closures = [close for message in history if (close := replay.step(message)) is not None]
    inventory = []
    for block in view.blocks():
        group = getattr(block, "group", None)
        inventory.append(
            {
                "kind": type(block).__name__,
                "source_id": work_record_of_block(block).source_id,
                "display": bool(block.display),
                "dormant": bool(getattr(block, "work_dormant", False)),
                "tool_name": getattr(block, "tool_name", None),
                "expanded": getattr(block, "_expanded", None),
                "member_ids": list(getattr(group, "row_ids", ())),
                "receipts": getattr(group, "count", None),
                "actions": getattr(group, "actions", None),
                "y": block.region.y - view.region.y,
            }
        )
    typed = []
    for event in events:
        row = {"type": event.type}
        if isinstance(event, ReasoningDeltaEvent):
            row["message_id"] = event.message_id
        if isinstance(event, ProviderTurnStartEvent):
            row["response_id"] = event.response_id
        if isinstance(event, (ToolExecutionStartEvent, ToolExecutionEndEvent)):
            row.update(tool_name=event.tool_name, tool_call_id=event.tool_call_id)
        if isinstance(event, ToolExecutionStartEvent):
            row["arguments"] = event.args
        if isinstance(event, ToolExecutionEndEvent):
            row.update(is_error=event.is_error, result=event.result.model_dump(mode="json"))
        if isinstance(event, AgentEndEvent):
            # Preserve the actual typed outcome/end reason without duplicating
            # its whole message list or guessing quiet from the final text.
            row.update(event.model_dump(mode="json", exclude={"messages"}))
        typed.append(row)
    raw = [message.model_dump(mode="json") for message in history]
    report = {
        "phase": phase,
        "trusted_display": (
            session_display
            if (session_display := getattr(app._session.frontend_state, "peer_cycle_display", None))
            else None
        ),
        "worker_origin": worker_origin,
        "peer_ack_delivery": getattr(receipt, "delivery", None),
        "provider_request_count": len(stream.requests),
        "events": typed,
        "raw_agent_messages": raw,
        "provider_payload_schema": [
            {
                "id": message.get("id"),
                "role": message.get("role"),
                "custom_type": message.get("custom_type"),
                "fields": sorted(message),
                "provider_payload": message.get("provider_payload"),
                "details": message.get("details"),
            }
            for message in raw
        ],
        "raw_closures": [
            {"member_ids": list(close.member_ids), "quiet_call_id": close.quiet_call_id}
            for close in closures
        ],
        "canonical_inventory": inventory,
        "viewport": [view.size.width, view.size.height],
        "virtual": [view.virtual_size.width, view.virtual_size.height],
        "scroll_y": view.scroll_y,
    }
    # BEFORE any one-bar/quiet/durable assertion: a failure still leaves the
    # discriminating child events, actual outer result and rendered inventory.
    app.save_screenshot(str(directory / f"{phase}.svg"))
    (directory / f"{phase}.json").write_text(json.dumps(report, indent=2, default=str))
    return report, closures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "direct",
        "eval",
        "mixed",
        "wrapped-no-send",
        "same-eval-root",
        "eval-send-direct-quiet",
        "mixed-reasoning",
    ],
)
async def test_real_session_direct_eval_peer_send_quiet_cycle_records_live_and_durable_evidence(
    tmp_path,
    monkeypatch,
    mode,
):
    scenario = mode
    with_reasoning = mode == "mixed-reasoning"
    if with_reasoning:
        # Reuse the exact real mixed path, not a flattened imitation of its
        # nested child. Reasoning is a provider channel, never message content.
        mode = "mixed"
    is_mixed = mode in {"mixed", "mixed-observer-failure", "wrapped-no-send", "same-eval-root"}
    expected_actions = 0 if mode == "wrapped-no-send" else 1
    own_root = Path(__file__).resolve().parents[3]
    assert Path(eval_tool.__file__).resolve().is_relative_to(own_root)
    deliveries = []
    record = SimpleNamespace(
        session_id="synthetic-diagnostic-target",
        pid=700001,
        conversation_name="isolated-target",
        cwd=str(tmp_path),
    )

    def resolve(**kwargs):
        return record, [], None

    async def deliver(target_record, **kwargs):
        assert target_record is record
        deliveries.append(
            {
                "session_id": kwargs["session_id"],
                "text": kwargs["text"],
                "mode": kwargs["mode"],
                "wake": kwargs["wake"],
            }
        )
        return peer_send.DeliveryOutcome(
            peer_send.DELIVERY_DELIVERED,
            "synthetic ack",
            "peer-" + "b" * 32,
            peer_send.WAKE_NOT_REQUESTED,
            1,
            "",
            "live",
            "isolated-target",
        )

    async def sender(context):
        return {"pid": 700002, "conversation_name": "isolated-diagnostic"}

    monkeypatch.setattr(peer_send, "resolve_peer_target", resolve)
    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", deliver)
    monkeypatch.setattr(builtin, "_send_sender_identity", sender)
    # The test never enumerates the operator's team/peer directories.
    monkeypatch.setattr(builtin, "_role_words", lambda context: set())

    # Use the EXISTING child-environment policy reader; no production spawn,
    # globals/config/security rule is changed. -P still honors this explicitly
    # allowed PYTHONPATH, so the worker must import this worktree, not the
    # primary venv's editable checkout. Actual child __file__/hash are checked.
    monkeypatch.setenv("PYTHONPATH", str(own_root))
    policy = shell_env.ShellEnvironmentPolicy(
        mode=shell_env.MODE_ALLOWLIST, inherit=("PYTHONPATH",), exclude=frozenset()
    )
    monkeypatch.setattr(shell_env, "load_policy", lambda: policy)
    message_args = {"target": "isolated-target", "message": "synthetic reply", "wake": False}
    if mode == "direct":
        scripts = [
            _call("direct-send", "send", message_args),
            _call("direct-quiet", "no_reply", {}),
        ]
    elif mode == "wrapped-no-send":
        scripts = [_call("outer-quiet", "eval", {"code": "tool('no_reply')", "timeout": 10})]
    elif mode == "same-eval-root":
        scripts = [
            _call(
                "outer-cycle",
                "eval",
                {
                    "code": (
                        "tool('send', target='isolated-target', "
                        "message='synthetic reply', wake=False); tool('no_reply')"
                    ),
                    "timeout": 10,
                },
            )
        ]
    elif mode == "eval-send-direct-quiet":
        scripts = [
            _call(
                "outer-send",
                "eval",
                {
                    "code": (
                        "tool('send', target='isolated-target', "
                        "message='synthetic reply', wake=False)"
                    ),
                    "timeout": 10,
                },
            ),
            _call("direct-quiet", "no_reply", {}),
        ]
    elif is_mixed:
        scripts = [
            _call("direct-send", "send", message_args),
            _call("outer-quiet", "eval", {"code": "tool('no_reply')", "timeout": 10}),
        ]
    else:
        scripts = [
            _call(
                "outer-send",
                "eval",
                {
                    "code": (
                        "tool('send', target='isolated-target', "
                        "message='synthetic reply', wake=False)"
                    ),
                    "timeout": 10,
                },
            ),
            _call("outer-quiet", "eval", {"code": "tool('no_reply')", "timeout": 10}),
        ]
    if with_reasoning:
        scripts[0][:0] = [
            StreamStartEvent(response_id="synthetic-reasoning-response"),
            StreamReasoningDelta(delta="peer-cycle-private-reasoning-control"),
        ]
    # If nested no_reply does not end the outer eval turn, capture an ACTUAL
    # final stop, not ScriptedStream exhaustion masquerading as a quiet end.
    scripts.append(
        [StreamTextDelta(delta="diagnostic final sentinel"), StreamEndEvent(stop_reason="stop")]
    )
    final_requested = asyncio.Event()
    release_final = asyncio.Event()

    class RecordingStream(ScriptedStream):
        def __init__(self, turns):
            super().__init__(turns)
            self.original_request_bytes = []

        def __call__(self, request, signal):
            self.original_request_bytes.append(request.model_dump_json())
            return super().__call__(request, signal)

    class HeldFinalStream(RecordingStream):
        def __call__(self, request, signal):
            response = super().__call__(request, signal)
            call_number = len(self.requests)

            async def generate():
                if call_number == (2 if mode in {"wrapped-no-send", "same-eval-root"} else 3):
                    # Hold the PROVIDER RESPONSE, before any final-message
                    # event can make grouping appear to work after the fact.
                    final_requested.set()
                    await asyncio.wait_for(release_final.wait(), timeout=20)
                async for event in response:
                    yield event

            return generate()

    stream = HeldFinalStream(scripts) if is_mixed else RecordingStream(scripts)
    context = ToolContext(cwd=str(tmp_path))
    send = builtin.build_send_tool(context)
    assert send is not None
    # The existing index resolves config/sessions/<id>, unlike the small
    # generic make_session helper's arbitrary tmp/sess directory. Use the real
    # owner layout so cold discovery exercises its actual public path.
    session = make_session(
        tmp_path / "sessions",
        stream,
        tools=[send, eval_tool.build_eval_tool()],
        cwd=str(tmp_path),
        has_ui=True,
    )
    quiet_context = ToolContext(cwd=str(tmp_path), quiet_end=session._quiet_end_callable())
    quiet = builtin.build_no_reply_tool(quiet_context)
    assert quiet is not None, "real Session must advertise its real quiet-end door"
    session.refresh_tools([send, eval_tool.build_eval_tool(), quiet])
    events = []
    ended = asyncio.Event()

    def observe(event):
        events.append(event)
        if isinstance(event, AgentEndEvent):
            ended.set()

    unsubscribe = session.subscribe(observe)
    directory = (
        Path(os.environ.get("OWNED_WORK_EVIDENCE_DIR", str(tmp_path / "evidence"))) / scenario
    )
    directory.mkdir(parents=True, exist_ok=True)
    worker_origin = None

    async def existing_session():
        # This path deliberately owns a REAL Session, not the pilot's FakeSession.
        return session

    app = OperatorApp(existing_session)
    try:
        if mode in {
            "eval",
            "mixed",
            "mixed-observer-failure",
            "wrapped-no-send",
            "same-eval-root",
            "eval-send-direct-quiet",
        }:
            worker_path = Path(eval_tool.__file__).resolve().with_name("eval_worker.py")
            expected_hash = hashlib.sha256(worker_path.read_bytes()).hexdigest()
            probe_path = directory / "worker-origin.json"
            probe = (
                "import json, hashlib\nfrom pathlib import Path\n"
                "import local_operator.tools.eval_worker as actual_worker\n"
                "p = Path(actual_worker.__file__).resolve()\n"
                f"Path({str(probe_path)!r}).write_text(json.dumps("
                f"{{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}}))"
            )
            tool = eval_tool.build_eval_tool()
            assert tool.execute is not None
            probe_result = await tool.execute(
                "source-probe",
                {"code": probe, "timeout": 10},
                None,
                None,
                ToolContext(cwd=str(tmp_path), session_id=session.session_id),
            )
            assert (
                probe_result.is_error is False
            ), "worker provenance cell must complete before any child tool calls"
            worker_origin = json.loads(probe_path.read_text())
            assert worker_origin == {"path": str(worker_path), "sha256": expected_hash}
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle_for_session(pilot, app)

            async def allow_synthetic_tools(tool_name, description):
                return tool_name in {"send", "eval", "no_reply"}

            session.set_approval_handler(allow_synthetic_tools)
            app.save_screenshot(str(directory / "before.svg"))
            receipt = await session.receive_peer_message(
                "synthetic incoming",
                wake=True,
                sender={"pid": 700003, "conversation_name": "isolated-incoming"},
                message_id="peer-" + "a" * 32,
            )
            prefinal_report = None
            if is_mixed:
                try:
                    await asyncio.wait_for(final_requested.wait(), timeout=20)
                    await pilot.pause()
                    await pilot.pause()
                    prefinal_report, _ = _capture(
                        app,
                        directory,
                        events,
                        session.history(),
                        receipt,
                        stream,
                        worker_origin,
                        phase="pre-final",
                    )
                    prefinal_report["final_response_held"] = not release_final.is_set()
                    (directory / "pre-final.json").write_text(
                        json.dumps(prefinal_report, indent=2, default=str)
                    )
                finally:
                    release_final.set()
            await asyncio.wait_for(ended.wait(), timeout=20)
            await pilot.pause()
            await pilot.pause()
            history = session.history()
            report, closures = _capture(
                app, directory, events, history, receipt, stream, worker_origin
            )
            print(
                json.dumps(
                    {
                        "mode": scenario,
                        "evidence": str(directory),
                        "raw_closures": report["raw_closures"],
                        "inventory": report["canonical_inventory"],
                        "events": report["events"],
                    },
                    default=str,
                )
            )
            ends = [event for event in events if isinstance(event, AgentEndEvent)]
            assert len(ends) == 1 and ends[0].error is None
            if mode in {"direct", "eval-send-direct-quiet"}:
                assert (
                    len(stream.requests) == 2
                ), "direct quiet must end before the sentinel provider call"
            assert not any(getattr(message, "role", None) == "user" for message in history)
            assert [
                request.model_dump_json() for request in stream.requests
            ] == stream.original_request_bytes
            assert all(
                "frontend_peer_cycle_display_v1" not in payload
                and '"peer_cycle_display"' not in payload
                for payload in stream.original_request_bytes
            )
            assert len(deliveries) == (0 if mode == "wrapped-no-send" else 1), (
                "real builtin sends must reach only the synthetic transport; "
                "zero-send mode sends nothing"
            )
            quiet_events = [
                event
                for event in events
                if isinstance(event, ToolExecutionEndEvent) and event.tool_name == "no_reply"
            ]
            assert len(quiet_events) == 1
            quiet_details = quiet_events[0].result.details
            assert quiet_details is not None
            assert quiet_details.get(QUIET_TURN_KEY) is True
            bars = [
                block
                for block in app.query_one(TranscriptView).blocks()
                if isinstance(block, QuietGroupBlock)
            ]
            if is_mixed:
                # Both frames already exist BEFORE the expected RED. Assert the
                # genuine final exclusion first, so a grouping failure cannot
                # silently skip this important negative control.
                finals = [
                    row for row in report["canonical_inventory"] if row["kind"] == "AssistantBlock"
                ]
                assert len(finals) == 1 and finals[0]["display"] is True
                assert all(finals[0]["source_id"] not in bar.group.row_ids for bar in bars)
                assert (
                    prefinal_report is not None and prefinal_report["final_response_held"] is True
                )
                if with_reasoning:
                    reasoning = [
                        event for event in events if isinstance(event, ReasoningDeltaEvent)
                    ]
                    assert len(reasoning) == 1
                    assert any(
                        isinstance(event, ProviderTurnStartEvent)
                        and event.response_id == "synthetic-reasoning-response"
                        for event in events
                    )
                    assert any(
                        isinstance(event, MessageStartEvent)
                        and event.message.id == reasoning[0].message_id
                        for event in events
                    )
                    assert events.index(reasoning[0]) < next(
                        index
                        for index, event in enumerate(events)
                        if isinstance(event, ToolExecutionStartEvent)
                    )
                    assert all(
                        "peer-cycle-private-reasoning-control" not in payload
                        for payload in stream.original_request_bytes
                    )
                    assert "peer-cycle-private-reasoning-control" not in json.dumps(
                        report["raw_agent_messages"]
                    )
                assert not any(
                    row["kind"] == "AssistantBlock"
                    for row in prefinal_report["canonical_inventory"]
                )
                physical = ["peer-" + "a" * 32 + ":0", "tool:direct-send:0", "tool:outer-quiet:0"]
                if mode == "wrapped-no-send":
                    physical = [physical[0], "tool:outer-quiet:0"]
                elif mode == "same-eval-root":
                    sends = [
                        event
                        for event in events
                        if isinstance(event, ToolExecutionEndEvent) and event.tool_name == "send"
                    ]
                    assert len(sends) == 1
                    physical = [
                        physical[0],
                        "tool:outer-cycle:0",
                        f"tool:{sends[0].tool_call_id}:0",
                    ]
                for captured in (prefinal_report, report):
                    original = [
                        row["source_id"]
                        for row in captured["canonical_inventory"]
                        if row["kind"] in {"PeerMessageBlock", "ToolCard"}
                    ]
                    assert original == physical
                prefinal_bars = [
                    row
                    for row in prefinal_report["canonical_inventory"]
                    if row["kind"] == "QuietGroupBlock"
                ]
                if mode == "mixed-observer-failure":
                    # Fault injection must not abort Session._emit, alter the
                    # three real provider calls, hide the final, or publish a
                    # bogus proof. The ordinary rows remain fully visible.
                    assert len(stream.requests) == 3
                    assert report["trusted_display"] is None
                    assert not bars and not prefinal_bars
                    assert all(
                        row["display"]
                        for row in report["canonical_inventory"]
                        if row["kind"] in {"PeerMessageBlock", "ToolCard"}
                    )
                    return
                assert (
                    len(prefinal_bars) == 1
                ), "mixed: typed nested quiet must condense before any genuine final lands"
                assert (prefinal_bars[0]["receipts"], prefinal_bars[0]["actions"]) == (
                    1,
                    expected_actions,
                )
                assert prefinal_bars[0]["member_ids"] == physical
            # Requested acceptance, NOT an assertion that EVAL already works.
            # A RED here has typed live events + raw outer data captured above;
            # it must lead to a separate display-persistence architectural ruling.
            assert (
                len(bars) == 1
            ), f"{mode}: one actual peer communication plus typed quiet must fold at idle"
            assert (bars[0].group.count, bars[0].group.actions) == (1, expected_actions)
            if mode == "direct":
                assert len(closures) == 1, "direct raw quiet proof stays unchanged"
            else:
                # Raw AgentMessages still contain NO nested quiet metadata.
                # Only the separate display archive may certify this exchange.
                assert len(closures) == 0, "no model-visible nested proof may be invented"
                assert len((report["trusted_display"] or {}).get("records", [])) == 1
                from local_operator.harness.peer_cycle_display import (
                    discover_records,
                    validate_record,
                )

                trusted = report["trusted_display"]["records"][0]
                assert validate_record(trusted, history, session.session_id) is not None
                archived = discover_records(session._transcript, history, session.session_id)
                assert archived is not None and [item["id"] for item in archived["records"]] == [
                    trusted["id"]
                ]
            bars[0].toggle_expanded()
            await pilot.pause()
            expanded_report, _ = _capture(
                app,
                directory,
                events,
                session.history(),
                receipt,
                stream,
                worker_origin,
                phase="expanded",
            )
            assert all(
                row["display"]
                for row in expanded_report["canonical_inventory"]
                if row["kind"] in {"PeerMessageBlock", "ToolCard"}
            )
    finally:
        release_final.set()
        unsubscribe()
        await session.dispose()
        # Existing exact-worker-group teardown, including a provenance kernel
        # created before Session's managed execution context took ownership.
        kernel = eval_tool._KERNELS.pop(session.session_id, None)
        if kernel is not None:
            await eval_tool._close_kernel(kernel)


@pytest.mark.asyncio
async def test_display_observer_failure_preserves_real_agent_final_and_provider_requests(
    tmp_path, monkeypatch, caplog
):
    from local_operator.harness.peer_cycle_display import DisplayCycleAccumulator

    original = DisplayCycleAccumulator.observe

    def broken_observer(self, event):
        # Actual typed producer event, not a Namespace/duck `.arguments` mimic.
        if isinstance(event, ToolExecutionStartEvent) and event.tool_call_id == "direct-send":
            raise RuntimeError("synthetic display observer failure")
        return original(self, event)

    monkeypatch.setattr(DisplayCycleAccumulator, "observe", broken_observer)
    await test_real_session_direct_eval_peer_send_quiet_cycle_records_live_and_durable_evidence(
        tmp_path, monkeypatch, "mixed-observer-failure"
    )
    refusals = [getattr(record, "peer_display_refusal", None) for record in caplog.records]
    assert {
        "stage": "observe",
        "event_type": "tool_execution_start",
        "error_type": "RuntimeError",
    } in refusals

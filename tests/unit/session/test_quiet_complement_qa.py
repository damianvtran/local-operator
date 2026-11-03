"""SOURCE ONLY. Ordinary fixture-based complements; no repo edits/runtime.

Reuse real Session, real serving subscription/GoalJudge and actual eval worker.
Only provider and peer wire delivery are synthetic; native notifier stays gated by
ordinary root fixtures. No internal quiet/user/goal status flags are assigned.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.peer_cycle_display import (
    DisplayCycleAccumulator,
    validate_record,
)
from local_operator.harness.peer_cycles import PeerCycleReplay
from local_operator.harness.types import AgentEndEvent, AgentMessage, CustomMessage
from local_operator.mobile import peer_send
from local_operator.monitors.delivery import MonitorDelivery
from local_operator.session.goal import GoalState
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools import builtin
from local_operator.tools import eval as ev
from local_operator.tui.app import OperatorApp
from tests.unit.session.test_quiet_product_qa import (
    REPORTS,
    WT,
    RecordingProvider,
    batch,
    call,
    emit_proof,
    final,
    require_worker_identity,
    serialized,
    settled,
    tool_result_text,
    worker_identity_code,
)
from tests.unit.session.test_session import make_session


def bind_home(monkeypatch):
    home = Path.home().resolve()
    assert home.is_absolute() and home != WT
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / "config"))
    monkeypatch.chdir(home)
    REPORTS.mkdir(parents=True, exist_ok=True)
    return home


def save(name, value):
    emit_proof(name, value)
    print("QA_COMPLEMENT=" + json.dumps({"case": name}), flush=True)


async def wait_until(predicate, timeout=8):
    async def spin():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(spin(), timeout)


@pytest.mark.parametrize("state", ["active", "legacy-constructor", "done", "deleted"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_goal_runtime_and_real_serving(state, wrapped, tmp_path, monkeypatch):
    asyncio.run(goal_case(state, wrapped, tmp_path, monkeypatch))


async def goal_case(state, wrapped, tmp_path, monkeypatch, host="serving"):
    home = bind_home(monkeypatch)
    name = host + "-goal-" + state + ("-wrapped" if wrapped else "-direct")
    timeline = []
    main_requests = []
    judge_requests = []
    other_requests = []
    events = []
    initial = (
        GoalState(text="QA legacy persisted objective") if state == "legacy-constructor" else None
    )
    initial_token = initial.token if initial is not None else None
    quiet = (
        call("eval", {"code": worker_identity_code() + "tool('no_reply')", "timeout": 10})
        if wrapped
        else call("no_reply")
    )
    scripts = [batch(quiet), final("QA_GOAL_STILL_OWED")]

    def provider(request, signal):
        texts = "\n".join(str(getattr(m, "text", "")) for m in request.messages)
        purpose = getattr(request, "purpose", None)
        is_judge = "judging whether a standing goal" in texts
        is_supplement = purpose in {"supplement_render", "supplement_repair"}
        bucket = (
            other_requests if is_supplement else (judge_requests if is_judge else main_requests)
        )
        bucket.append(request.model_dump(mode="json"))
        timeline.append(
            {"kind": "provider", "purpose": purpose, "judge": is_judge, "ordinal": len(bucket)}
        )

        async def stream():
            if is_supplement:
                from local_operator.harness.types import StreamEndEvent, StreamTextDelta

                yield StreamTextDelta(delta="QA_UNEXPECTED_SUPPLEMENT_DISPATCH")
                yield StreamEndEvent(stop_reason="stop")
            elif is_judge:
                from local_operator.harness.types import StreamEndEvent, StreamTextDelta

                yield StreamTextDelta(delta="VERDICT: ACHIEVED\nQA scripted actual judge verdict")
                yield StreamEndEvent(stop_reason="stop")
            else:
                assert len(main_requests) <= len(scripts), "unexpected main/supplement inference"
                for item in scripts[len(main_requests) - 1]:
                    yield item

        return stream()

    session = make_session(
        tmp_path,
        provider,
        tools=[ev.build_eval_tool()],
        cwd=str(home),
        has_ui=True,
        goal_state=initial,
    )
    session.set_conversation_name("QA isolated named goal owner", user_set=True)
    if state != "legacy-constructor":
        session.arm_goal("QA actual standing objective")
        if state == "done":
            session.mark_goal_done("QA actual user marked done")
        if state == "deleted":
            session.delete_goal()
    start = {
        "goal": session.goal,
        "status": session.goal_status,
        "token": session.goal_token,
        "legacy_initial_token": initial_token,
    }
    handle = None
    pilot_manager = None
    app = None
    if host == "serving":
        handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(home))
        handle.subscribe(lambda: None)  # Installs the REAL serving end subscriber
    else:

        async def factory():
            return session

        app = OperatorApp(factory)
        pilot_manager = app.run_test(size=(100, 30))
        await pilot_manager.__aenter__()
        await wait_until(lambda: getattr(app, "_session", None) is session)

    def observe(event):
        events.append(event)
        timeline.append({"kind": "event", "event": serialized(event)})

    unsub = session.subscribe(observe)

    async def approve(*args):
        return True

    session.set_approval_handler(approve)
    try:
        await session.receive_peer_message(
            "QA goal peer result",
            wake=True,
            sender={"pid": 700009, "conversation_name": "qa-complement"},
            message_id="peer-" + "d" * 32,
        )
        await settled(session, lambda: bool(main_requests))
        if state in {"active", "legacy-constructor"}:
            await wait_until(lambda: session.goal_status == "done")
            await settled(session)
        if handle is not None:
            task = getattr(handle._supplements, "_task", None)
            if task is not None:
                await asyncio.wait_for(asyncio.shield(task), 8)
        ends = [e for e in events if isinstance(e, AgentEndEvent)]
        save(
            name,
            {
                "start": start,
                "finish": {
                    "goal": session.goal,
                    "status": session.goal_status,
                    "judge": session.goal_judge,
                    "history": session.history_view(),
                },
                "main_requests": main_requests,
                "judge_requests": judge_requests,
                "supplement_requests": other_requests,
                "timeline": timeline,
                "attention": await session.refresh_attention(),
                "goal_sidecar": (
                    (tmp_path / "sess" / "goal.json").read_text()
                    if (tmp_path / "sess" / "goal.json").exists()
                    else None
                ),
                "coverage_note": "Real "
                + host
                + " owner/subscriber; supplements default settings, no forced enabled supplement",
            },
        )
        # Rendered-frame capture belongs to separate visual evidence, not JSON proof uploads.
        if wrapped:
            require_worker_identity(events)
        assert ends
        if state in {"active", "legacy-constructor"}:
            assert any(
                getattr(event, "tool_name", None) == "no_reply"
                and getattr(event, "is_error", False)
                and "active runtime goal still requires work" in tool_result_text(event)
                for event in events
            ), "actual standing-goal refusal was not observed"
            assert not any(e.quiet_accepted for e in ends), "standing goal was silenced"
            assert len(main_requests) == 2 and len(judge_requests) == 1
            assert session.goal_status == "done", "actual judge verdict not persisted"
        else:
            assert len(main_requests) == 1 and not judge_requests and not other_requests
            assert len(ends) == 1 and ends[0].quiet_accepted
    finally:
        unsub()
        if handle is not None:
            await handle.dispose()
        if pilot_manager is not None:
            await pilot_manager.__aexit__(None, None, None)
        await session.dispose()
        await ev.close_session_kernel(session.session_id)


@pytest.mark.parametrize("kind", ["monitor-quiet", "monitor-required", "job-typed"])
def test_other_real_typed_delivery(kind, tmp_path, monkeypatch):
    asyncio.run(delivery_case(kind, tmp_path, monkeypatch))


async def delivery_case(kind, tmp_path, monkeypatch):
    home = bind_home(monkeypatch)
    timeline = []
    stream = RecordingProvider([batch(call("no_reply")), final()], timeline)
    session = make_session(tmp_path, stream, cwd=str(home), has_ui=True)
    events = []
    unsub = session.subscribe(events.append)
    try:
        if kind.startswith("monitor"):
            await session._deliver_monitor(
                MonitorDelivery(
                    monitor_id="qa-monitor",
                    name="QA monitor",
                    tool="qa_synthetic_source",
                    changes=1,
                    checks=1,
                    skipped=0,
                    delta_text="QA material delta",
                    at_ms=0,
                    final=True,
                    notify=kind == "monitor-required",
                )
            )
        else:
            # Actual Session typed host ingress, not a spawned background job.
            initial: list[AgentMessage] = [
                CustomMessage(
                    custom_type="job_result",
                    attribution="user",
                    details={"job_id": "qa-job", "text": "QA synthetic settled job result"},
                )
            ]
            await session._prompt_messages(initial)
        await settled(session, lambda: bool(stream.requests))
        ends = [e for e in events if isinstance(e, AgentEndEvent)]
        save(
            kind,
            {
                "requests": len(stream.requests),
                "events": [serialized(e) for e in events],
                "timeline": timeline,
                "history": [serialized(m) for m in session.history()],
                "attention": await session.refresh_attention(),
                "seam": (
                    "scheduler delivery seam"
                    if kind.startswith("monitor")
                    else "typed _prompt_messages; NOT real JobManager completion"
                ),
            },
        )
        assert len(ends) == 1
        assert bool(ends[0].quiet_accepted) == (kind != "monitor-required")
        assert len(stream.requests) == (2 if kind == "monitor-required" else 1)
    finally:
        unsub()
        await session.dispose()


@pytest.mark.parametrize("arrival", ["person", "arm-goal"])
def test_actual_host_post_compaction_recheck(arrival, tmp_path, monkeypatch):
    asyncio.run(barrier_case(arrival, tmp_path, monkeypatch))


async def barrier_case(arrival, tmp_path, monkeypatch):
    home = bind_home(monkeypatch)
    timeline = []
    stream = RecordingProvider(
        [batch(call("no_reply")), final("QA_REQUIRED_AFTER_BARRIER")], timeline
    )
    session = make_session(tmp_path, stream, cwd=str(home), has_ui=True)
    events = []
    unsub = session.subscribe(events.append)
    original = session._maybe_compact
    injected = False

    async def boundary(*args, **kwargs):
        nonlocal injected
        result = await original(*args, **kwargs)  # Preserve actual compaction/check path
        if not injected:
            injected = True
            if arrival == "person":
                session.steer("QA actual boundary person", message_id="qa-barrier-person")
            else:
                session.arm_goal("QA goal armed at actual host recheck")
        return result

    monkeypatch.setattr(session, "_maybe_compact", boundary)
    try:
        await session.receive_peer_message(
            "QA barrier peer", wake=True, sender={"pid": 700010, "conversation_name": "qa-boundary"}
        )
        await settled(session, lambda: bool(stream.requests))
        ends = [e for e in events if isinstance(e, AgentEndEvent)]
        save(
            "barrier-" + arrival,
            {
                "injected": injected,
                "timeline": timeline,
                "ends": [serialized(e) for e in ends],
                "history": [serialized(m) for m in session.history()],
                "goal": session.goal,
                "seam": (
                    "Wrapper calls original _maybe_compact, injects real ingress; "
                    "NOT a forced compaction-provider run"
                ),
            },
        )
        assert injected and ends
        assert not any(e.quiet_accepted for e in ends)
        assert len(stream.requests) == 2
        if arrival == "person":
            assert (
                sum(getattr(m, "id", None) == "qa-barrier-person" for m in session.history()) == 1
            )
    finally:
        unsub()
        await session.dispose()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("delivery_error", [False, True])
def test_actual_peer_exchange_live_and_cold(wrapped, delivery_error, tmp_path, monkeypatch):
    asyncio.run(certificate_case(wrapped, delivery_error, tmp_path, monkeypatch))


async def certificate_case(wrapped, delivery_error, tmp_path, monkeypatch):
    home = bind_home(monkeypatch)
    target = SimpleNamespace(
        session_id="qa-synthetic-target", pid=700011, conversation_name="qa-target", cwd=str(home)
    )
    deliveries = []
    monkeypatch.setattr(peer_send, "resolve_peer_target", lambda **kw: (target, [], None))

    async def deliver(record, **kw):
        deliveries.append({"target": record.session_id, "text": kw["text"]})
        if delivery_error:
            raise RuntimeError("QA synthetic wire rejection")
        return peer_send.DeliveryOutcome(
            peer_send.DELIVERY_DELIVERED,
            "QA wire ack",
            "peer-" + "f" * 32,
            peer_send.WAKE_NOT_REQUESTED,
            1,
            "",
            "live",
            "qa-target",
        )

    async def sender(context):
        return {"pid": 700012, "conversation_name": "qa-sender"}

    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", deliver)
    monkeypatch.setattr(builtin, "_send_sender_identity", sender)
    monkeypatch.setattr(builtin, "_role_words", lambda context: set())
    sendargs = {"target": "qa-target", "message": "QA finite reply", "wake": False}
    if wrapped:
        scripts = [
            batch(
                call(
                    "eval",
                    {
                        "code": worker_identity_code()
                        + "tool('send',target='qa-target',message='QA finite reply',wake=False); "
                        "tool('no_reply')",
                        "timeout": 10,
                    },
                )
            ),
            final(),
        ]
    else:
        scripts = [batch(call("send", sendargs)), batch(call("no_reply")), final()]
    timeline = []
    stream = RecordingProvider(scripts, timeline)
    session = make_session(
        tmp_path, stream, tools=[ev.build_eval_tool()], cwd=str(home), has_ui=True
    )
    send = builtin.build_send_tool(session._build_tool_context())
    assert send is not None
    session.refresh_tools(
        [*session._tools, send]
    )  # Adds actual send, never a quiet door substitute
    accumulator = DisplayCycleAccumulator(session.session_id, "qa-epoch", redact=lambda text: text)
    records = []
    events = []

    def observe(event):
        events.append(event)
        record = accumulator.observe(event)
        if record is not None:
            records.append(record)

    unsub = session.subscribe(observe)

    async def approve(*args):
        return True

    session.set_approval_handler(approve)
    try:
        await session.receive_peer_message(
            "QA finite incoming",
            wake=True,
            sender={"pid": 700013, "conversation_name": "qa-incoming"},
            message_id="peer-" + "a" * 32,
        )
        await settled(session, lambda: bool(stream.requests))
        history = session.history()
        cold = [validate_record(record, history, session.session_id) for record in records]
        replay = PeerCycleReplay((session.session_id, 0))
        raw = []
        for message in history:
            item = replay.step(message)
            if item is not None:
                raw.append(serialized(item))
        name = (
            "certificate-"
            + ("wrapped" if wrapped else "direct")
            + ("-error" if delivery_error else "-ok")
        )
        save(
            name,
            {
                "deliveries": deliveries,
                "requests": len(stream.requests),
                "events": [serialized(e) for e in events],
                "records": records,
                "cold": cold,
                "raw_replay": raw,
                "seam": (
                    "Actual builtin send/Session/eval; synthetic resolution+wire only; "
                    "identity redactor safe ONLY synthetic payloads"
                ),
            },
        )
        if wrapped:
            require_worker_identity(events)
        assert len(deliveries) == 1
        if delivery_error:
            assert not records and not raw, "failed exchange got finite success certificate"
        elif wrapped:
            assert records and all(item is not None for item in cold)
            cross_session = dict(records[0], session_id="qa-other-owner")
            assert validate_record(cross_session, history, session.session_id) is None
        else:
            assert raw, "direct successful pair lost raw cold proof"
    finally:
        unsub()
        await session.dispose()
        await ev.close_session_kernel(session.session_id)


@pytest.mark.parametrize("state", ["active", "done"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_real_tui_goal_end_subscriber(state, wrapped, tmp_path, monkeypatch):
    asyncio.run(goal_case(state, wrapped, tmp_path, monkeypatch, host="tui"))

"""SOURCE ONLY: focused remaining acceptance gates, ordinary fixtures.

Actual compactor/provider/journal, counted existing supplement admission boundary,
and typed job input across real Session ownership boundary. No new framework.
"""

import asyncio

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import AgentEndEvent, CustomMessage
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_fold_forward_compaction import (
    TEXT_MODEL,
    _compaction_entries,
)
from tests.unit.session.test_quiet_complement_qa import bind_home, save
from tests.unit.session.test_quiet_product_qa import (
    RecordingProvider,
    batch,
    call,
    final,
    serialized,
    settled,
)
from tests.unit.session.test_session import CompactionAwareStream, make_session


def test_paired_stale_length_producer_seam(tmp_path):
    from tests.unit.session.test_quiet_acceptance import (
        test_compaction_producer_does_not_revive_a_quiet_request,
    )

    asyncio.run(test_compaction_producer_does_not_revive_a_quiet_request(tmp_path))


def test_actual_compaction_finishes_before_quiet_host_acceptance_without_resume(
    tmp_path, monkeypatch
):
    asyncio.run(compaction_case(tmp_path, monkeypatch))


async def compaction_case(tmp_path, monkeypatch):
    home = bind_home(monkeypatch)
    timeline = []
    session = None
    scripts = [final("QA seed reply") for _ in range(3)] + [
        batch(call("no_reply")),
        final("QA next owner answer"),
    ]

    class Stream(CompactionAwareStream):
        def __call__(self, request, signal):
            purpose = getattr(request, "purpose", None)
            held = getattr(session, "_held_end", None)
            timeline.append(
                {
                    "kind": "request",
                    "purpose": purpose,
                    "request": request.model_dump(mode="json"),
                    "held_requested": bool(held and held.quiet_requested),
                    "held_accepted": bool(held and held.quiet_accepted),
                }
            )
            if purpose == "compaction":
                assert held is not None and held.quiet_requested and not held.quiet_accepted
            return super().__call__(request, signal)

    stream = Stream(scripts)
    session = make_session(
        tmp_path,
        stream,
        model=TEXT_MODEL,
        cwd=str(home),
        has_ui=True,
        compaction_settings=CompactionSettings(
            keep_recent_tokens=40, auto_continue=True, threshold_tokens=400000
        ),
    )
    session.set_conversation_name("QA actual compaction owner", user_set=True)
    events = []

    def observe(event):
        events.append(event)
        timeline.append({"kind": "event", "event": serialized(event)})

    unsub = session.subscribe(observe)
    original = session._maybe_compact
    try:
        for index in range(3):
            await session.prompt("QA seed " + str(index) + " detail " * 2000)
        assert not [r for r in stream.requests if getattr(r, "purpose", None) == "compaction"]
        before = list(session._context.messages)

        async def due_real_pass(*args, **kwargs):
            session._compaction_settings = CompactionSettings(
                keep_recent_tokens=40, auto_continue=True, threshold_tokens=4000
            )
            return await original(*args, **kwargs)

        monkeypatch.setattr(session, "_maybe_compact", due_real_pass)
        await session.receive_peer_message(
            "QA compaction quiet input",
            wake=True,
            sender={"pid": 700020, "conversation_name": "qa-compaction"},
        )
        await settled(session, lambda: len(stream.requests) >= 4)
        after = list(session._context.messages)
        entries = _compaction_entries(session)
        accepted = [e for e in events if isinstance(e, AgentEndEvent) and e.quiet_accepted]
        summary_requests = [
            r for r in stream.requests if getattr(r, "purpose", None) == "compaction"
        ]
        save(
            "real-compaction-before-acceptance",
            {
                "timeline": timeline,
                "before_context": [serialized(m) for m in before],
                "after_context": [serialized(m) for m in after],
                "entries": [{"payload": e.payload} for e in entries],
                "continuations": [serialized(m) for m in session._continuation_queue],
                "summary_requests": len(summary_requests),
                "request_count": len(stream.requests),
            },
        )
        assert (
            summary_requests
        ), "Actual planner declined compaction; not a passing interface-only test"
        assert entries and any(
            isinstance(m, CustomMessage) and m.custom_type == "compaction_summary" for m in after
        )
        assert len(after) < len(before), "actual compacted context did not shrink"
        assert len(accepted) == 1 and not session._continuation_queue
        starts = [
            i
            for i, row in enumerate(timeline)
            if row["kind"] == "request" and row["purpose"] == "compaction"
        ]
        ends = [
            i
            for i, row in enumerate(timeline)
            if row["kind"] == "event" and row["event"].get("quiet_accepted")
        ]
        assert max(starts) < min(ends)
        assert any(getattr(e, "type", "") == "compaction_end" for e in events)
        ordinary_before = sum(getattr(r, "purpose", None) != "compaction" for r in stream.requests)
        monkeypatch.setattr(session, "_maybe_compact", original)
        # Raise trigger for next input so this check concerns stale resume only.
        session._compaction_settings = CompactionSettings(
            keep_recent_tokens=40, auto_continue=True, threshold_tokens=400000
        )
        await session.prompt("QA normal next owner")
        await settled(session)
        ordinary_after = sum(getattr(r, "purpose", None) != "compaction" for r in stream.requests)
        owner_ends = [e for e in events if isinstance(e, AgentEndEvent)]
        save(
            "real-compaction-next-owner",
            {
                "ordinary_before": ordinary_before,
                "ordinary_after": ordinary_after,
                "ends": [serialized(e) for e in owner_ends],
                "continuations": [serialized(m) for m in session._continuation_queue],
            },
        )
        assert ordinary_before == 4 and ordinary_after == 5 and not session._continuation_queue
        assert owner_ends[-1].generation != accepted[0].generation
        # Pair with repo test_compaction_producer_does_not_revive_a_quiet_request:
        # toolUse integration alone does NOT mutation-discriminate stale length guard.
    finally:
        unsub()
        await session.dispose()


@pytest.mark.parametrize("quiet", [False, True])
def test_actual_serving_supplement_admission_boundary(quiet, tmp_path, monkeypatch):
    asyncio.run(supplement_case(quiet, tmp_path, monkeypatch))


async def supplement_case(quiet, tmp_path, monkeypatch):
    home = bind_home(monkeypatch)
    timeline = []
    stream = RecordingProvider(
        [batch(call("no_reply"))] if quiet else [final("QA ordinary user answer")], timeline
    )
    session = make_session(tmp_path, stream, cwd=str(home), has_ui=True)
    session.set_conversation_name("QA supplement boundary", user_set=True)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(home))
    admissions = []
    original = handle._supplements.on_agent_end

    def counted(event, *, goal_loop_running):
        admissions.append({"event": serialized(event), "goal_loop_running": goal_loop_running})
        return original(event, goal_loop_running=goal_loop_running)

    monkeypatch.setattr(handle._supplements, "on_agent_end", counted)
    handle.subscribe(lambda: None)
    try:
        if quiet:
            await session.receive_peer_message(
                "QA accepted quiet supplement fence",
                wake=True,
                sender={"pid": 700021, "conversation_name": "qa-supplement"},
            )
        else:
            await session.prompt("QA real user ordinary answer")
        await settled(session, lambda: bool(stream.requests))
        task = getattr(handle._supplements, "_task", None)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), 8)
        save(
            "supplement-admission-" + str(quiet),
            {
                "calls": admissions,
                "requests": len(stream.requests),
                "timeline": timeline,
                "seam": (
                    "Actual serving subscription and original on_agent_end; "
                    "spy counts admission without changing eligibility"
                ),
            },
        )
        assert len(admissions) == (0 if quiet else 1)
        if admissions:
            assert admissions[0]["event"]["type"] == "agent_end"
            assert not admissions[0]["event"].get("quiet_accepted")
    finally:
        await handle.dispose()
        await session.dispose()


@pytest.mark.parametrize("when", ["before-host-commit", "accepted-subscriber"])
def test_typed_job_ingress_is_retained_across_owning_boundary(when, tmp_path, monkeypatch):
    asyncio.run(job_boundary(when, tmp_path, monkeypatch))


async def job_boundary(when, tmp_path, monkeypatch):
    home = bind_home(monkeypatch)
    timeline = []
    stream = RecordingProvider([batch(call("no_reply")), final("QA typed job handled")], timeline)
    session = make_session(tmp_path, stream, cwd=str(home), has_ui=True)
    events = []
    tasks = []
    injected = False
    message = CustomMessage(
        custom_type="job_result",
        attribution="job",
        id="qa-owned-job-boundary",
        details={"job_id": "qa-finite-job", "text": "QA settled job"},
    )

    def admit():
        nonlocal injected
        if not injected:
            injected = True
            tasks.append(asyncio.create_task(session._prompt_messages([message])))

    original = session._maybe_compact

    async def barrier(*args, **kwargs):
        result = await original(*args, **kwargs)
        admit()  # Real typed host ingress queues behind current lock, not private queue mutation
        return result

    if when == "before-host-commit":
        monkeypatch.setattr(session, "_maybe_compact", barrier)

    def observe(event):
        events.append(event)
        timeline.append({"kind": "event", "event": serialized(event)})
        if (
            when == "accepted-subscriber"
            and isinstance(event, AgentEndEvent)
            and event.quiet_accepted
        ):
            admit()

    unsub = session.subscribe(observe)
    try:
        await session.receive_peer_message(
            "QA initial quiet owner",
            wake=True,
            sender={"pid": 700022, "conversation_name": "qa-job-boundary"},
        )
        await settled(session, lambda: bool(stream.requests))
        await asyncio.wait_for(asyncio.gather(*tasks), 10)
        await settled(session)
        ends = [e for e in events if isinstance(e, AgentEndEvent)]
        save(
            "typed-job-" + when,
            {
                "timeline": timeline,
                "ends": [serialized(e) for e in ends],
                "history": [serialized(m) for m in session.history()],
                "seam": "Actual typed _prompt_messages, not JobManager spawn/storage lifecycle",
            },
        )
        assert injected and len(stream.requests) == 2 and len(ends) == 2
        assert ends[0].quiet_accepted and ends[0].generation != ends[1].generation
        assert sum(getattr(m, "id", None) == message.id for m in session.history()) == 1
        firstend = next(
            i
            for i, row in enumerate(timeline)
            if row["kind"] == "event" and row["event"].get("quiet_accepted")
        )
        requests = [i for i, row in enumerate(timeline) if row["kind"] == "provider"]
        assert firstend < requests[1]
    finally:
        unsub()
        await session.dispose()

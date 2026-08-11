"""Parent↔subagent messaging (``harness/comms.py`` and the ``hub`` tool).

Two levels. The unit half drives :class:`SubagentComms` against a stand-in
child so every branch — including the ones a live child rarely reaches
(unknown id, question to a finished child, double cancel) — is exercised
deterministically. The integration half runs real parent and child Sessions
through a scripted provider, because the whole feature rests on claims about
the real injection path: that an aside reaches a busy child's context, that
the child's own reply resolves the parent's question, and that a resumed
child replays the stopped one's transcript instead of starting fresh.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest

from local_operator.harness.comms import HUB_MESSAGE_TYPE, SubagentComms
from local_operator.harness.types import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    AsideResult,
    ChatRequest,
    CustomMessage,
    Message,
    MessageEndEvent,
    ModelSpec,
    StaleAside,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    SubagentEndEvent,
    TextContent,
    ToolCall,
    ToolContext,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import execute_hub
from local_operator.tools.registry import create_tools

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


async def wait_for(predicate, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


# --- unit level: comms against a stand-in child -------------------------------


class FakeJob:
    def __init__(self, job_id: str, status: str = "running") -> None:
        self.id = job_id
        self.status = status


class FakeJobs:
    """The slice of AsyncJobManager comms uses."""

    def __init__(self) -> None:
        self.jobs: dict[str, FakeJob] = {}
        self.cancelled: list[str] = []

    def add(self, job_id: str, status: str = "running") -> FakeJob:
        job = FakeJob(job_id, status)
        self.jobs[job_id] = job
        return job

    def get(self, job_id: str, **_kwargs):
        return self.jobs.get(job_id)

    async def cancel(self, job_id: str, **_kwargs) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status != "running":
            return False
        job.status = "cancelled"
        self.cancelled.append(job_id)
        return True


class FakeChild:
    """A child session as comms sees it: aside queue, steering, event stream."""

    def __init__(self) -> None:
        self.asides: list[Callable[[], AsideResult]] = []
        self.steers: list[str] = []
        self.handlers: list[Callable[[AgentEvent], Any]] = []

    def queue_aside(self, thunk: Callable[[], AsideResult]) -> None:
        self.asides.append(thunk)

    def steer(self, text: str) -> None:
        self.steers.append(text)

    def subscribe(self, handler: Callable[[AgentEvent], Any]) -> Callable[[], None]:
        self.handlers.append(handler)
        return lambda: self.handlers.remove(handler)

    def materialize(self) -> list[AsideResult]:
        """Run the queued thunks the way the loop's injection boundary does —
        including firing ``on_discard`` on a withdrawn aside, which is what
        the real loop's ``_materialize_asides`` does and the only way the
        stale-question paths are exercised at all."""
        out = []
        for thunk in self.asides:
            result = thunk()
            if isinstance(result, StaleAside) and result.message.on_discard is not None:
                result.message.on_discard()
            out.append(result)
        self.asides.clear()
        return out

    async def emit(self, event: AgentEvent) -> None:
        for handler in list(self.handlers):
            await handler(event)


class FakeParent:
    """A parent session as comms sees it."""

    def __init__(self, jobs: FakeJobs) -> None:
        self.jobs = jobs
        self.asides: list[Callable[[], AsideResult]] = []

    def queue_aside(self, thunk: Callable[[], AsideResult]) -> None:
        self.asides.append(thunk)


def body(result: ToolResult) -> str:
    """The text of a tool result. Asserted rather than assumed: every hub
    result is a single text block, and a result that stopped being one is a
    regression the reader should see named."""
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def hub_aside(result: AsideResult) -> CustomMessage:
    """One materialized aside, narrowed to the hub message it should be. An
    aside can also be a plain Message, a StaleAside or nothing, and each of
    those is a different bug \u2014 naming the expectation here says which."""
    assert isinstance(result, CustomMessage)
    return result


def payload(result: ToolResult) -> dict[str, Any]:
    """The machine details of a tool result; every hub result carries them."""
    assert result.details is not None
    return result.details


def status_of(session: Session, job_id: str) -> str:
    """A job row's status, asserting the row still exists \u2014 retention could
    sweep it, and "None has no status" would hide which job vanished."""
    job = session.jobs.get(job_id)
    assert job is not None, f"job {job_id} is gone from the manager"
    return job.status


def first_text(history: list[AgentMessage]) -> str:
    """The text of a replayed history's first entry, asserting it is a real
    message rather than a custom marker."""
    message = history[0]
    assert isinstance(message, Message)
    return message.text


def assistant(text: str, *, tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text=text)],
        tool_calls=tool_calls or [],
    )


def wire(*, attach: bool = True) -> tuple[SubagentComms, FakeJobs, FakeChild, FakeParent]:
    jobs = FakeJobs()
    parent = FakeParent(jobs)
    comms = SubagentComms(parent)  # type: ignore[arg-type]
    jobs.add("job-1")
    comms.record_launch("job-1", "parser")
    child = FakeChild()
    if attach:
        comms.attach("job-1", child, tmp_dir())
    return comms, jobs, child, parent


def tmp_dir():
    from pathlib import Path

    return Path("/tmp")


def test_send_reaches_a_live_child_as_an_aside():
    comms, _jobs, child, _parent = wire()

    delivery = comms.send("job-1", "the fixture API moved to v2")

    assert delivery.outcome == "injected"
    assert delivery.label == "parser"
    [message] = child.materialize()
    assert isinstance(message, CustomMessage)
    assert message.custom_type == HUB_MESSAGE_TYPE
    assert "the fixture API moved to v2" in message.details["text"]
    # A note must not ask for an answer, or every child stops to reply.
    assert message.details["expects_reply"] is False
    assert "not a question" in message.details["text"]


def test_send_to_an_unstarted_child_buffers_until_it_attaches():
    """A job parked behind the capacity gate has no session yet. The parent
    must still be able to address it — the message is what stops the child
    doing the wrong thing for ten minutes."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1")
    comms.record_launch("job-1", "parser")

    delivery = comms.send("job-1", "skip the migration, it landed already")
    assert delivery.outcome == "queued"

    child = FakeChild()
    comms.attach("job-1", child, tmp_dir())
    [aside] = child.materialize()
    assert "skip the migration" in hub_aside(aside).details["text"]


def test_send_to_an_unknown_job_fails_without_raising():
    comms, _jobs, _child, _parent = wire()
    delivery = comms.send("nope", "hello")
    assert delivery.outcome == "failed"
    assert "unknown subagent" in (delivery.error or "")


def test_send_to_a_finished_child_says_so_and_points_at_resume():
    comms, jobs, _child, _parent = wire()
    comms.detach("job-1")
    jobs.jobs["job-1"].status = "completed"

    delivery = comms.send("job-1", "one more thing")

    assert delivery.outcome == "failed"
    assert "completed" in (delivery.error or "")
    assert "resume" in (delivery.error or "")


def test_steer_is_a_real_user_message_not_an_aside():
    """A course change belongs in the child's instructions, so it goes through
    the steering queue; a note goes through the aside queue."""
    comms, _jobs, child, _parent = wire()

    delivery = comms.steer("job-1", "stop after the first failing test")

    assert delivery.outcome == "injected"
    assert child.asides == []
    assert "stop after the first failing test" in child.steers[0]
    assert "changes your instructions" in child.steers[0]


def test_resolve_addresses_by_id_label_and_all():
    comms, jobs, _child, _parent = wire()
    jobs.add("job-2")
    comms.record_launch("job-2", "docs")
    comms.attach("job-2", FakeChild(), tmp_dir())

    assert comms.resolve("job-1") == (["job-1"], None)
    assert comms.resolve("parser") == (["job-1"], None)
    ids, error = comms.resolve("all")
    assert error is None and set(ids) == {"job-1", "job-2"}


def test_resolve_refuses_an_ambiguous_label():
    """Two children called "reviewer" is normal. Picking one silently is not."""
    comms, jobs, _child, _parent = wire()
    jobs.add("job-2")
    comms.record_launch("job-2", "parser")

    ids, error = comms.resolve("parser")

    assert ids == []
    assert "ambiguous" in (error or "")


def test_resolve_all_with_nothing_running_is_an_error_not_an_empty_success():
    comms, jobs, _child, _parent = wire()
    jobs.jobs["job-1"].status = "completed"
    ids, error = comms.resolve("all")
    assert ids == []
    assert "no running subagents" in (error or "")


@pytest.mark.asyncio
async def test_ask_is_answered_by_the_childs_own_reply():
    comms, _jobs, child, _parent = wire()

    question = asyncio.create_task(comms.ask("job-1", "are you stuck?", 5_000))
    await wait_for(lambda: bool(child.asides))
    # The loop materializes the aside at the injection boundary; only then is
    # the question really in the child's context.
    [aside] = child.materialize()
    assert hub_aside(aside).details["expects_reply"] is True

    assert comms.reply_to_parent("job-1", "not stuck, tests are slow") == (
        "answered the parent's question"
    )
    reply = await question

    assert reply.text == "not stuck, tests are slow"
    assert reply.timed_out is False
    assert reply.error is None


@pytest.mark.asyncio
async def test_a_child_that_answers_in_prose_still_resolves_the_question():
    """The child is told to use its hub tool. A model that ignores the tool and
    just talks must not strand the parent for the whole timeout, so a
    text-only assistant message counts as the answer."""
    comms, _jobs, child, _parent = wire()

    question = asyncio.create_task(comms.ask("job-1", "are you stuck?", 5_000))
    await wait_for(lambda: bool(child.asides))
    child.materialize()

    await child.emit(MessageEndEvent(message=assistant("no, waiting on the fixtures")))
    reply = await question

    assert reply.text == "no, waiting on the fixtures"


@pytest.mark.asyncio
async def test_narration_before_a_tool_call_is_not_read_as_the_answer():
    """ "Let me check the logs" is the child still working, not its reply."""
    comms, _jobs, child, _parent = wire()

    question = asyncio.create_task(comms.ask("job-1", "are you stuck?", 5_000))
    await wait_for(lambda: bool(child.asides))
    child.materialize()

    await child.emit(
        MessageEndEvent(
            message=assistant("let me check the logs", tool_calls=[ToolCall(name="bash")])
        )
    )
    await asyncio.sleep(0.01)
    assert not question.done()

    await child.emit(MessageEndEvent(message=assistant("not stuck")))
    assert (await question).text == "not stuck"


@pytest.mark.asyncio
async def test_an_assistant_message_before_the_question_lands_is_not_the_answer():
    """The child may be mid-sentence about something else when the question is
    asked. Only a message after the aside actually reached its context can be
    a reply to it."""
    comms, _jobs, child, _parent = wire()

    question = asyncio.create_task(comms.ask("job-1", "are you stuck?", 5_000))
    await wait_for(lambda: bool(child.asides))

    await child.emit(MessageEndEvent(message=assistant("rewriting the tokenizer now")))
    await asyncio.sleep(0.01)
    assert not question.done()

    child.materialize()  # NOW the question is in context
    await child.emit(MessageEndEvent(message=assistant("not stuck")))
    assert (await question).text == "not stuck"


@pytest.mark.asyncio
async def test_ask_times_out_without_erroring_and_leaves_the_child_alone():
    comms, jobs, child, _parent = wire()

    reply = await comms.ask("job-1", "are you stuck?", 20)

    assert reply.timed_out is True
    assert reply.error is None
    assert jobs.jobs["job-1"].status == "running"  # a timeout is not a kill


@pytest.mark.asyncio
async def test_a_timed_out_question_cannot_hijack_the_next_one():
    """A question that timed out while the child sat in a long tool call is
    still queued as an aside. When it finally reaches the boundary it must
    withdraw itself — not arm the NEXT question's future, which would return
    the answer to the old question as the answer to the new one."""
    comms, _jobs, child, _parent = wire()

    stale = await comms.ask("job-1", "how far along are you?", 20)
    assert stale.timed_out is True

    fresh = asyncio.create_task(comms.ask("job-1", "are you stuck?", 5_000))
    await wait_for(lambda: len(child.asides) == 2)
    materialized = child.materialize()

    assert isinstance(materialized[0], StaleAside)  # the abandoned question
    # Asserted as a type before its payload is read: without the guard the
    # live question is withdrawn too, and dereferencing .details would fail
    # with an AttributeError instead of naming what actually went wrong.
    assert not isinstance(materialized[1], StaleAside), "the live question was withdrawn too"
    assert "are you stuck?" in hub_aside(materialized[1]).details["text"]
    comms.reply_to_parent("job-1", "not stuck")
    assert (await fresh).text == "not stuck"


@pytest.mark.asyncio
async def test_ask_a_finished_child_fails_immediately():
    comms, jobs, _child, _parent = wire()
    comms.detach("job-1")
    jobs.jobs["job-1"].status = "completed"

    reply = await comms.ask("job-1", "are you stuck?", 5_000)

    assert reply.text is None
    assert "completed" in (reply.error or "")


@pytest.mark.asyncio
async def test_ask_an_unknown_job_fails_immediately():
    comms, _jobs, _child, _parent = wire()
    reply = await comms.ask("nope", "are you stuck?", 5_000)
    assert "unknown subagent" in (reply.error or "")


@pytest.mark.asyncio
async def test_a_child_finishing_mid_question_fails_the_wait_at_once():
    """Otherwise the parent blocks for the full timeout on an agent that no
    longer exists."""
    comms, _jobs, child, _parent = wire()

    question = asyncio.create_task(comms.ask("job-1", "are you stuck?", 30_000))
    await wait_for(lambda: bool(child.asides))
    child.materialize()

    comms.detach("job-1")
    reply = await question

    assert "finished before answering" in (reply.error or "")


@pytest.mark.asyncio
async def test_a_withdrawn_question_does_not_reach_a_child_that_settled_first():
    """The aside is a thunk precisely so a question queued before the child
    stopped is not injected into a corpse."""
    comms, _jobs, child, _parent = wire()

    question = asyncio.create_task(comms.ask("job-1", "are you stuck?", 30_000))
    await wait_for(lambda: bool(child.asides))
    comms.detach("job-1")
    await question

    from local_operator.harness.types import StaleAside

    assert all(isinstance(item, StaleAside) for item in child.materialize())


@pytest.mark.asyncio
async def test_two_questions_at_once_are_refused_rather_than_racing():
    comms, _jobs, child, _parent = wire()
    first = asyncio.create_task(comms.ask("job-1", "are you stuck?", 1_000))
    await wait_for(lambda: bool(child.asides))

    second = await comms.ask("job-1", "and now?", 1_000)
    assert "already pending" in (second.error or "")

    child.materialize()
    comms.reply_to_parent("job-1", "fine")
    assert (await first).text == "fine"


def test_an_unprompted_child_message_lands_in_the_parents_context():
    comms, _jobs, _child, parent = wire()

    outcome = comms.reply_to_parent("job-1", "I am blocked: no credentials for staging")

    assert "delivered to the parent" in outcome
    [thunk] = parent.asides
    message = hub_aside(thunk())
    assert message.custom_type == HUB_MESSAGE_TYPE
    assert "I am blocked" in message.details["text"]
    assert "parser" in message.details["text"]


def test_resume_refuses_a_second_child_on_the_same_transcript():
    """Two children on one transcript directory is data loss, not just
    confusion: each holds its own in-memory entry list and ``compact_file``
    rewrites the whole file from one of them."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    comms.record_launch("job-1", "parser")
    comms._records["job-1"].session_dir = tmp_dir()
    comms._records["job-1"].settled = True
    # The first resume is live, on job-1's directory.
    jobs.add("job-2")
    comms.record_launch("job-2", "parser")
    comms._records["job-2"].session_dir = tmp_dir()

    new_id, error = comms.resume("job-1", "carry on")

    assert new_id is None
    assert "already resumed as job job-2" in (error or "")


def test_a_label_still_addresses_the_live_child_after_a_resume():
    """Resume reuses the label, so the stopped record and its continuation
    both match it. Refusing there would revoke the handle the system prompt
    tells the model to use, at exactly the moment it is managing a child that
    went wrong."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    comms.record_launch("job-1", "parser")
    jobs.add("job-2")
    comms.record_launch("job-2", "parser")

    assert comms.resolve("parser") == (["job-2"], None)

    # Both settled: the ambiguity is real again and is reported.
    jobs.jobs["job-2"].status = "completed"
    ids, error = comms.resolve("parser")
    assert ids == []
    assert "ambiguous" in (error or "")


def test_records_are_bounded_even_when_children_never_started():
    """A job cancelled while parked behind the capacity gate never reaches
    attach/detach, so keying eviction on "was detached" left those records
    permanently unevictable and the cap was not a cap."""
    from local_operator.harness import comms as comms_module

    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    for index in range(comms_module.MAX_RECORDS + 10):
        job_id = f"job-{index}"
        jobs.add(job_id, status="cancelled")  # never started, never detached
        comms.record_launch(job_id, f"child-{index}")

    assert len(comms._records) <= comms_module.MAX_RECORDS
    # The newest survive: the oldest are the ones least likely to be resumed.
    assert "job-260" in comms._records


@pytest.mark.asyncio
async def test_cancel_stops_a_running_child_and_a_second_cancel_reports_the_state():
    comms, jobs, _child, _parent = wire()

    first = await comms.cancel("job-1")
    second = await comms.cancel("job-1")

    assert first.outcome == "cancelled"
    assert jobs.cancelled == ["job-1"]
    assert second.outcome == "failed"
    assert "already cancelled" in (second.error or "")


@pytest.mark.asyncio
async def test_cancel_an_unknown_job_fails_without_raising():
    comms, _jobs, _child, _parent = wire()
    delivery = await comms.cancel("nope")
    assert delivery.outcome == "failed"
    assert "unknown job" in (delivery.error or "")


def test_resume_refuses_a_child_that_never_started():
    """No transcript means no context to continue, and a resume that quietly
    spawns a stranger is worse than none: the parent would believe the
    history survived."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    comms.record_launch("job-1", "parser")

    new_id, error = comms.resume("job-1", "carry on")

    assert new_id is None
    assert "no transcript" in (error or "")


def test_resume_refuses_a_running_child():
    comms, _jobs, _child, _parent = wire()
    new_id, error = comms.resume("job-1", "carry on")
    assert new_id is None
    assert "still running" in (error or "")


def test_resume_refuses_an_unknown_job():
    comms, _jobs, _child, _parent = wire()
    new_id, error = comms.resume("nope", "carry on")
    assert new_id is None
    assert "unknown subagent" in (error or "")


def test_resume_refuses_when_the_transcript_is_gone_from_disk(tmp_path):
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    comms.record_launch("job-1", "parser")
    comms.attach("job-1", FakeChild(), tmp_path / "vanished")
    comms.detach("job-1")

    new_id, error = comms.resume("job-1", "carry on")

    assert new_id is None
    assert "gone from disk" in (error or "")


# --- the hub tool surface -----------------------------------------------------


def hub_tool(*, job_id: str | None, comms: Any) -> AgentTool:
    context = ToolContext(cwd=".", subagent_comms=comms, job_id=job_id)
    return create_tools(context, enabled=("hub",))[0]


def test_the_tool_is_not_advertised_without_a_subagent_engine():
    assert create_tools(ToolContext(cwd="."), enabled=("hub",)) == []


def test_a_parent_and_a_child_are_offered_different_tools():
    """A child has one peer and no children, so the four parent ops would be
    context spent on things it cannot do."""
    comms, _jobs, _child, _parent = wire()

    parent_tool = hub_tool(job_id=None, comms=comms)
    child_tool = hub_tool(job_id="job-1", comms=comms)

    assert set(parent_tool.parameters["properties"]) >= {"op", "to", "message"}
    assert set(child_tool.parameters["properties"]) == {"i", "message"}
    # Redirecting/killing autonomous work rides the same gate as starting it;
    # a child answering its own parent starts nothing.
    assert parent_tool.approval_tier == "write"
    assert child_tool.approval_tier == "read"


@pytest.mark.asyncio
async def test_the_tool_broadcasts_and_reports_one_receipt_per_child():
    comms, jobs, child_one, _parent = wire()
    jobs.add("job-2")
    comms.record_launch("job-2", "docs")
    child_two = FakeChild()
    comms.attach("job-2", child_two, tmp_dir())

    result = await execute_hub(
        "call-1",
        {"op": "send", "to": ["all"], "message": "the API moved"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    text = body(result)
    assert "2/2" in text
    assert "parser" in text and "docs" in text
    assert payload(result)["acted"] == 2
    assert child_one.asides and child_two.asides


@pytest.mark.asyncio
async def test_the_tool_refuses_to_ask_several_children_at_once():
    comms, jobs, _child, _parent = wire()
    jobs.add("job-2")
    comms.record_launch("job-2", "docs")
    comms.attach("job-2", FakeChild(), tmp_dir())

    result = await execute_hub(
        "call-1",
        {"op": "ask", "to": ["all"], "message": "stuck?"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    assert result.is_error
    assert "one subagent at a time" in body(result)


@pytest.mark.asyncio
async def test_the_tool_requires_a_body_for_everything_but_cancel():
    comms, _jobs, _child, _parent = wire()
    context = ToolContext(cwd=".", subagent_comms=comms)

    empty = await execute_hub("call-1", {"op": "send", "to": ["job-1"]}, None, None, context)
    assert empty.is_error and "needs a message" in body(empty)

    killed = await execute_hub("call-2", {"op": "cancel", "to": ["job-1"]}, None, None, context)
    assert not killed.is_error


@pytest.mark.asyncio
async def test_a_delivery_that_reached_nobody_is_flagged_useless():
    """Compaction and the renderer both read this: a receipt list of pure
    failures is not an observation worth keeping in context."""
    comms, jobs, _child, _parent = wire()
    comms.detach("job-1")
    jobs.jobs["job-1"].status = "completed"

    result = await execute_hub(
        "call-1",
        {"op": "send", "to": ["job-1"], "message": "hello"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    assert result.useless is True
    assert payload(result)["acted"] == 0


@pytest.mark.asyncio
async def test_the_child_tool_answers_the_parents_question():
    comms, _jobs, child, _parent = wire()
    question = asyncio.create_task(comms.ask("job-1", "stuck?", 5_000))
    await wait_for(lambda: bool(child.asides))
    child.materialize()

    result = await execute_hub(
        "call-1",
        {"message": "no, the fixtures are just slow"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms, job_id="job-1"),
    )

    assert "answered the parent's question" in body(result)
    assert (await question).text == "no, the fixtures are just slow"


# --- integration: real sessions, real injection -------------------------------


def tool_call_stream(name: str, args: dict[str, Any]):
    async def gen():
        yield StreamToolCallDelta(
            index=0, id=f"c-{name}", name=name, argument_delta=json.dumps(args)
        )
        yield StreamEndEvent(stop_reason="toolUse")

    return gen()


def text_stream(body: str):
    async def gen():
        yield StreamTextDelta(delta=body)
        yield StreamEndEvent(stop_reason="stop")

    return gen()


class ScriptedProvider:
    """Children that keep working until the parent's message changes their
    mind — the only way to test injection into a BUSY agent."""

    def __call__(self, request: ChatRequest, signal=None):
        body = "\n".join(
            message.text
            for message in request.messages
            if isinstance(message, Message) and message.role == "user"
        )
        if "Answer it now" in body and "answered" not in body:
            return tool_call_stream("hub", {"message": "not stuck; fixtures are slow"})
        if "changes your instructions" in body:
            return text_stream("acknowledged; stopping now")
        if "You were interrupted" in body:
            return text_stream("resumed and wrapped up")
        return tool_call_stream("bash", {"command": "sleep 0.1; echo tick"})


def make_parent(tmp_path, provider) -> Session:
    return Session(
        model=MODEL,
        stream_fn=provider,
        tools=[],
        transcript=Transcript(tmp_path / "parent"),
        system_blocks_provider=lambda: ["parent", "env"],
        cwd=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_a_question_reaches_a_busy_child_and_its_answer_comes_back(tmp_path, monkeypatch):
    """The end-to-end claim: an aside injected into a child that is mid tool
    loop reaches its context, and the child's own hub call answers the
    parent."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_parent(tmp_path, ScriptedProvider())
    await parent.async_init()
    job_id = parent._launch_subagent(label="parser", prompt="Rewrite the parser.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)

    reply = await comms.ask(job_id, "are you stuck?", 20_000)

    assert reply.error is None and reply.timed_out is False
    assert "not stuck" in (reply.text or "")

    # It really landed in the child's own transcript, not just in a future.
    history = Transcript(comms.session_dir_of(job_id)).build_llm_history()
    assert any(
        isinstance(message, CustomMessage)
        and message.custom_type == HUB_MESSAGE_TYPE
        and "are you stuck?" in message.details["text"]
        for message in history
    )
    await parent.dispose()


@pytest.mark.asyncio
async def test_steering_a_child_changes_what_it_does(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_parent(tmp_path, ScriptedProvider())
    await parent.async_init()
    events: list[AgentEvent] = []
    parent.subscribe(events.append)
    job_id = parent._launch_subagent(label="parser", prompt="Rewrite the parser.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)

    assert comms.steer(job_id, "stop after the first failing test").outcome == "injected"

    await wait_for(lambda: any(isinstance(e, SubagentEndEvent) for e in events))
    end = next(e for e in events if isinstance(e, SubagentEndEvent))
    assert end.status == "completed"
    assert "acknowledged" in (end.result_text or "")
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_resumed_child_replays_the_stopped_ones_transcript(tmp_path, monkeypatch):
    """The claim resume lives or dies on: the new child holds what the old one
    said and did, rather than being a fresh agent with a summary."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_parent(tmp_path, ScriptedProvider())
    await parent.async_init()
    job_id = parent._launch_subagent(label="docs", prompt="Update the docs.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)
    # Let it do some work, then stop it. A running turn's messages are
    # persisted when the turn ENDS (Session.prompt writes them after its loop
    # finishes, and a hard cancel pre-empts that), so the stopped child's
    # transcript only settles after cancel — the state resume has to read.
    await asyncio.sleep(0.3)
    await comms.cancel(job_id)
    await wait_for(lambda: status_of(parent, job_id) == "cancelled")
    await wait_for(lambda: len(Transcript(comms.session_dir_of(job_id)).build_llm_history()) >= 2)
    before = Transcript(comms.session_dir_of(job_id)).build_llm_history()

    new_id, error = comms.resume(job_id, "You were interrupted. Wrap up.")

    assert error is None and new_id is not None
    await wait_for(lambda: status_of(parent, new_id) != "running")
    assert comms.session_dir_of(new_id) == comms.session_dir_of(job_id)
    after = Transcript(comms.session_dir_of(new_id)).build_llm_history()
    assert len(after) > len(before)
    assert first_text(after) == first_text(before) == "Update the docs."
    assert status_of(parent, new_id) == "completed"
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_cancelled_child_keeps_the_work_it_had_already_done(tmp_path, monkeypatch):
    """What makes resume worth having. ``Session.prompt`` persists a turn when
    the turn ends, so a hard-cancelled child used to leave a transcript
    holding only its launch prompt — resuming it would forget every tool call
    it had completed."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_parent(tmp_path, ScriptedProvider())
    await parent.async_init()
    job_id = parent._launch_subagent(label="docs", prompt="Update the docs.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)
    await asyncio.sleep(0.3)  # let it complete at least one tool round

    await comms.cancel(job_id)
    await wait_for(lambda: len(Transcript(comms.session_dir_of(job_id)).build_llm_history()) > 1)

    history = Transcript(comms.session_dir_of(job_id)).build_llm_history()
    assert any(
        isinstance(message, Message) and message.role == "assistant" and message.tool_calls
        for message in history
    )
    assert any(isinstance(message, Message) and message.role == "tool" for message in history)
    # Written once, not once per pass: a duplicated entry replays twice.
    ids = [message.id for message in history]
    assert len(ids) == len(set(ids))
    # ...and nothing it could not replay: an assistant tool-use block with no
    # matching result is a 400 from every major provider, so the batch the
    # cancel landed inside must not be persisted half-finished.
    answered = {
        message.tool_call_id
        for message in history
        if isinstance(message, Message) and message.role == "tool"
    }
    assert all(
        call.id in answered
        for message in history
        if isinstance(message, Message)
        for call in message.tool_calls
    )
    await parent.dispose()


def test_the_persisted_prefix_keeps_finished_batches_and_drops_the_interrupted_one():
    """Only the tail can be incoherent, so a cancel must cost the child the
    batch it was inside and nothing earlier."""
    from local_operator.harness.subagent import _answered_prefix

    done = ToolCall(id="t1", name="bash")
    interrupted = ToolCall(id="t2", name="bash")
    messages = [
        Message(role="user", content=[TextContent(text="go")]),
        assistant("", tool_calls=[done]),
        Message(role="tool", content=[TextContent(text="ok")], tool_call_id="t1"),
        assistant("", tool_calls=[interrupted]),
    ]

    kept = _answered_prefix(messages)

    assert kept == messages[:3]
    # A history with no dangling call is returned untouched.
    assert _answered_prefix(messages[:3]) == messages[:3]

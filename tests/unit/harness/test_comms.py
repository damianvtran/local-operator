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
import time
from typing import Any, Callable

import pytest

from local_operator.harness.comms import (
    HUB_MESSAGE_TYPE,
    SubagentComms,
    extract_parent_message,
)
from local_operator.harness.subagent import MCP_DENIED_ATTR
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
from local_operator.session.transcript import TRANSCRIPT_FILENAME, Transcript
from local_operator.tools.builtin import execute_hub
from local_operator.tools.registry import create_tools

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


#: Ceiling for the signal-free pump below, counted in LOOP TURNS rather than
#: in seconds. That unit is the whole point (#122): machine contention
#: stretches how long a turn takes but not how many turns the awaited work
#: needs, so a loaded box cannot exhaust this budget the way it exhausted the
#: 10 s wall-clock deadline this replaced. It is a deadlock guard — reached
#: only when the awaited thing is never going to happen — not a timing
#: assumption, and raising it is never the fix for a failure here.
MAX_PUMP_TURNS = 3000

#: Same role for the signal-driven wait: a wedged test must fail rather than
#: hang a CI job forever (this suite has no pytest-timeout). It can never be
#: the discriminator for TIMING — the success path blocks on an
#: ``asyncio.Event`` set by the code under test and never compares elapsed
#: time — so the only run that reaches it is one where the signal is not
#: coming at all.
#:
#: It is still a COST, though, which is why it is 30 s and not the 120 s this
#: started at (review round 1, F2): a wedge is paid per affected test, and four
#: waits here share the bound, so an over-generous guard turns one real
#: regression in child persistence into minutes of CI. These tests complete in
#: about a second, so 30 s is still ~30x the observed work — far too wide to
#: fire on slowness, narrow enough to keep the feedback loop short.
DEADLOCK_GUARD_S = 30.0


class ChangeSignal:
    """An edge fed by the code under test's OWN notifications.

    The deterministic synchronisation point #122 asks for. A child's progress
    is published twice over — ``SubagentComms.notify_detail_persisted`` fires
    after each durable transcript append, and the parent's event stream
    carries ``SubagentStartEvent``/``SubagentProgressEvent`` — so a test that
    needs "the child got somewhere" can block on those publications instead of
    re-reading the filesystem on a timer and betting the answer arrives before
    a deadline.

    Both sources are watched because neither alone covers every predicate:
    the detail registry sees nested children the parent stream never routes,
    and the parent stream carries the attach moment (``SubagentStartEvent`` is
    emitted immediately after ``comms.attach``) that makes ``session_dir_of``
    non-None before any transcript append has happened.

    Level-triggered on purpose: :meth:`settle` clears the flag BEFORE the
    caller re-tests its predicate, so a notification racing in between the
    test and the wait is kept rather than lost.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._unsubscribes: list[Callable[[], None]] = []

    def _fire(self, *_: Any) -> None:
        self._event.set()

    def watch_comms(self, comms) -> "ChangeSignal":
        """Wake on every durable child-transcript append."""
        self._unsubscribes.append(comms.subscribe_detail_changes(self._fire))
        return self

    def watch_session(self, session: Session) -> "ChangeSignal":
        """Wake on every event the parent's stream carries."""
        self._unsubscribes.append(session.subscribe(self._fire))
        return self

    def arm(self) -> None:
        """Drop any pending edge so the next wait blocks on a NEW one."""
        self._event.clear()

    async def settle(self) -> None:
        """Block until something publishes."""
        await self._event.wait()
        self._event.clear()

    def close(self) -> None:
        for unsubscribe in self._unsubscribes:
            unsubscribe()
        self._unsubscribes.clear()


async def _await_signalled(predicate, signal: "ChangeSignal") -> None:
    """Re-test ``predicate`` once per publication, forever. Bounded by caller."""
    while True:
        # Armed BEFORE the test, never after: a publication landing between
        # the test and the wait is then still on the flag, so the edge that
        # would satisfy the predicate can never be missed.
        signal.arm()
        if predicate():
            return
        await signal.settle()


async def wait_for(predicate, *, signal: "ChangeSignal | None" = None) -> None:
    """Block until ``predicate`` holds, on an event rather than on a clock.

    With a ``signal``, each re-test follows a publication from the code under
    test, so this waits exactly as long as the work takes and no longer: there
    is no elapsed-time comparison anywhere on the success path. That is the
    property #122 asks for, and it is why the integration cases below pass one.

    Without one — the in-process cases over ``FakeChild``, where the awaited
    step is a coroutine this test itself scheduled and there is nothing to
    subscribe to — it pumps the loop and bounds itself by TURN COUNT. See
    :data:`MAX_PUMP_TURNS` for why that unit survives contention where seconds
    did not.
    """
    if signal is not None:
        try:
            await asyncio.wait_for(_await_signalled(predicate, signal), DEADLOCK_GUARD_S)
        except asyncio.TimeoutError:
            raise AssertionError(
                "the code under test published no change satisfying the condition in "
                f"{DEADLOCK_GUARD_S:.0f}s — it is wedged, not merely slow"
            ) from None
        return

    for _ in range(MAX_PUMP_TURNS):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"condition never held across {MAX_PUMP_TURNS} loop turns")


def has_complete_tool_round(session_dir) -> bool:
    """Whether a child's transcript already holds one FINISHED tool round.

    "Finished" means an assistant message carrying tool calls plus a tool
    result for every one of them. A round that is only half written is the
    thing the cancel tests must not act on: cancelling there leaves a dangling
    tool-use block, which ``_answered_prefix`` then trims away, and the test
    asserting the child "kept the work it had already done" would be asserting
    against an empty prefix.

    Reading the file mid-append is safe by construction: the transcript is
    JSONL and ``TranscriptEntry.from_json`` drops a malformed line on its own
    rather than failing the parse, so a torn final row simply reads as not
    there yet and the next poll sees it.
    """
    history = Transcript(session_dir).build_llm_history()
    answered = {
        message.tool_call_id
        for message in history
        if isinstance(message, Message) and message.role == "tool"
    }
    return any(
        isinstance(message, Message)
        and message.role == "assistant"
        and message.tool_calls
        and all(call.id in answered for call in message.tool_calls)
        for message in history
    )


# --- unit level: comms against a stand-in child -------------------------------


class FakeJob:
    def __init__(
        self,
        job_id: str,
        status: str = "running",
        queued: bool = False,
        started_at: float | None = None,
    ) -> None:
        self.id = job_id
        self.status = status
        #: Mirrors ``AsyncJob.started_at``: stamped as ``_run_job``'s FIRST
        #: statement, so it is the only field that answers "has the runner
        #: actually begun". ``status`` reads ``running`` from registration.
        #: This fixture not modelling it is why the suite agreed for so long
        #: that a half-hour-old child had "not started yet".
        self.started_at = started_at
        #: Mirrors ``AsyncJob.queued``: admitted to the ledger but holding no
        #: execution slot. Distinct from "running but still building its
        #: session", which is what a job carries between registration and
        #: ``attach``.
        self.queued = queued


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
        self.steer_messages: list[Message] = []
        self.handlers: list[Callable[[AgentEvent], Any]] = []

    def queue_aside(self, thunk: Callable[[], AsideResult]) -> None:
        self.asides.append(thunk)

    def steer_message(self, message: Message) -> None:
        self.steers.append(message.text)
        self.steer_messages.append(message)

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


def test_lineage_queries_are_recursive_cycle_safe_and_snapshot_durable(tmp_path) -> None:
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    for job_id in ("a", "b", "c", "d"):
        jobs.add(job_id)
    comms.record_launch("a", "architect", prompt="root", agent_role="architect")
    comms.record_launch("b", "coder", parent_job_id="a", effort="hi")
    comms.record_launch("c", "reviewer", parent_job_id="a")
    comms.record_launch("d", "scout", parent_job_id="b")
    for job_id in ("a", "b", "c", "d"):
        comms._records[job_id].session_dir = tmp_path / job_id

    assert [node.job_id for node in comms.ancestors("d")] == ["a", "b"]
    assert [node.job_id for node in comms.children("a")] == ["b", "c"]
    assert [node.job_id for node in comms.peers("b")] == ["c"]
    assert comms.parent("a") is None

    restored = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    restored.restore(comms.snapshot())
    assert [node.label for node in restored.ancestors("d")] == ["architect", "coder"]
    restored_a = restored.node("a")
    restored_b = restored.node("b")
    assert restored_a is not None and restored_a.prompt == "root"
    assert restored_b is not None and restored_b.effort == "hi"

    restored._records["a"].parent_job_id = "d"
    assert len(restored.ancestors("d")) <= 4


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


def test_record_launch_after_attach_keeps_the_attached_child():
    """attach() can run BEFORE record_launch(): under Textual's eager task
    factory the runner registered by ``jobs_manager.register`` executes
    synchronously up to its first suspension, so it may build its child and
    call attach() before ``register`` returns and ``run_subagent`` records
    the launch. record_launch() must merge into that record, not replace it —
    replacing it discards the live child, the reply watcher and the session
    directory, and every later send/steer/ask then buffers into ``pending``
    on a record whose flush (attach) already happened. Observed live: a
    healthy reviewer ran 41 minutes while two ``hub ask`` status checks never
    reached it, and the roster reported the settled child as "never started".
    """
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1")
    child = FakeChild()
    session_dir = tmp_dir() / "child-session"
    comms.attach("job-1", child, session_dir)  # eager runner attaches first
    comms.record_launch("job-1", "reviewer")  # then registration records

    # The live child survived the late record_launch.
    delivery = comms.send("job-1", "scope is only the diff")
    assert delivery.outcome == "injected"
    [aside] = child.materialize()
    assert "scope is only the diff" in hub_aside(aside).details["text"]

    # The roster keeps the transcript directory, so a settled child stays
    # resumable instead of reading "never started, so it has no transcript".
    [row] = [info for info in comms.roster() if info.job_id == "job-1"]
    assert row.label == "reviewer"
    assert row.session_id == session_dir.name


def test_record_launch_replacement_is_still_the_normal_path():
    """The merge must not change the ordinary ordering: a child that has not
    attached yet gets its launch record, and a buffered note still flushes at
    the later attach."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1")
    comms.record_launch("job-1", "parser")
    comms.record_launch("job-1", "parser")  # a duplicate launch is a no-op

    delivery = comms.send("job-1", "note")
    assert delivery.outcome == "queued"

    child = FakeChild()
    comms.attach("job-1", child, tmp_dir())
    [aside] = child.materialize()
    assert "note" in hub_aside(aside).details["text"]


@pytest.mark.asyncio
async def test_a_starting_child_is_not_reported_as_parked_behind_the_gate():
    """Two states reach the no-live-child branch and an operator acts on them
    differently.

    A PARKED job is waiting for a slot and may not start for minutes. An
    ADMITTED job is waiting only on this event loop — ``register`` schedules
    the runner without entering it, so one yield starts it. Both used to
    report the capacity gate, which is a false statement about an admitted
    job: it says "nothing is happening" about a child that is already
    spending tokens, so a healthy run reads as stuck and gets cancelled
    instead of waited out.

    Neither string may claim a session is being BUILT (in the state this
    branch was written for the runner has not been entered at all), neither
    may stutter the label the render site already prints, and both owe the
    caller a next step — the parked one most of all, because that is the
    state where a caller most needs to be told to stop waiting.
    """
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]

    jobs.add("job-starting")  # running, not queued, no child attached yet
    comms.record_launch("job-starting", "reviewer")
    starting = await comms.ask("job-starting", "status?", timeout_ms=50)

    jobs.add("job-parked").queued = True
    comms.record_launch("job-parked", "designer")
    parked = await comms.ask("job-parked", "status?", timeout_ms=50)

    assert "scheduled" in (starting.error or "")
    assert "capacity gate" not in (starting.error or "")
    assert "being built" not in (starting.error or "")
    # The genuinely parked job keeps the message that names its real cause,
    # and now carries the advice its sibling always had.
    assert "capacity gate" in (parked.error or "")
    assert "do not wait on it" in (parked.error or "")
    assert starting.error != parked.error
    # The only render site prefixes `{label} ({job_id}): `, and the card
    # truncates the reason as content — a stuttered label spends the cells
    # that would otherwise have carried the state word.
    assert "reviewer" not in (starting.error or "")
    assert "designer" not in (parked.error or "")


@pytest.mark.asyncio
async def test_ask_waits_for_a_child_the_loop_has_not_entered_yet():
    """The reported bug: checking in on a just-launched subagent is refused.

    ``register`` calls ``ensure_future``, which SCHEDULES the runner without
    entering it, so a parent that launches a child and asks it something in
    its very next tool call finds no live child — not because anything is
    wrong, but because the loop has not reached the runner. Refusing with
    "it starts when this session next yields; retry in a moment" is an answer
    the model cannot act on: it cannot yield except by making another call, so
    it either polls or stops checking in. Both were observed live.

    Awaiting the attach is self-fulfilling — the wait IS the yield that lets
    the runner start — so the question must survive a late attach rather than
    being rejected the instant it finds ``child is None``.
    """
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1")
    comms.record_launch("job-1", "designer")

    question = asyncio.create_task(comms.ask("job-1", "status?", 30_000))
    await asyncio.sleep(0)  # the ask is now parked on the attach, not refused
    assert not question.done(), "the question was refused instead of waiting"

    # The runner finally gets its turn on the loop and attaches its session.
    child = FakeChild()
    comms.attach("job-1", child, tmp_dir())
    await wait_for(lambda: bool(child.asides))
    child.materialize()
    comms.reply_to_parent("job-1", "two findings so far")

    reply = await question
    assert reply.text == "two findings so far"
    assert reply.error is None


@pytest.mark.asyncio
async def test_ask_does_not_burn_its_timeout_on_a_parked_child():
    """A parked job holds no slot and may not start for minutes, so waiting on
    it is waiting on nothing. It must be reported at once — the opposite call
    from the scheduled-but-not-entered child above, which one yield starts."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-parked").queued = True
    comms.record_launch("job-parked", "designer")

    loop = asyncio.get_running_loop()
    started = loop.time()
    reply = await comms.ask("job-parked", "status?", 30_000)
    elapsed = loop.time() - started

    assert "capacity gate" in (reply.error or "")
    assert elapsed < 1.0, f"waited {elapsed:.1f}s on a job that cannot start"


@pytest.mark.asyncio
async def test_ask_gives_up_when_the_child_never_starts():
    """The wait is bounded. A child that never attaches must produce the honest
    not-started reason rather than consuming the caller's whole timeout: the
    grace is a FRACTION of it, so the caller still gets an answer in time to
    do something else."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1")
    comms.record_launch("job-1", "designer")

    loop = asyncio.get_running_loop()
    started = loop.time()
    reply = await comms.ask("job-1", "status?", 400)
    elapsed = loop.time() - started

    assert "scheduled" in (reply.error or "")
    # Half of 400 ms, and comfortably less than the full timeout.
    assert elapsed < 0.4, f"consumed the whole timeout ({elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_a_child_that_settles_before_starting_fails_the_wait_at_once():
    """``detach`` must wake a waiter: the attach it is parked on is never
    coming, and burning the full grace on a child that already died is the
    same stall this wait exists to remove."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1")
    comms.record_launch("job-1", "designer")

    question = asyncio.create_task(comms.ask("job-1", "status?", 30_000))
    await asyncio.sleep(0)
    assert not question.done()

    jobs.jobs["job-1"].status = "failed"
    comms.detach("job-1")

    reply = await asyncio.wait_for(question, timeout=2.0)
    assert reply.text is None
    # The REASON matters, not merely that there is one: a child that died must
    # not be reported as "not started yet ... retry in a moment", which invites
    # exactly the polling loop this wait exists to remove.
    assert "failed" in (reply.error or "")
    assert "retry in a moment" not in (reply.error or "")


@pytest.mark.asyncio
async def test_ask_never_exceeds_the_timeout_it_was_given():
    """The attach grace is deducted from the caller's budget, not added to it.

    ``timeout_ms`` is the whole budget a caller planned around. Spending the
    grace first and then granting the full timeout for the answer let a
    1000 ms request block for ~1450 ms while reporting it had waited 1000 ms,
    and it scales with the timeout — the 600 s schema maximum could block for
    900 s.
    """
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1")
    comms.record_launch("job-1", "designer")

    # Attach late enough to consume part of the grace, then never answer, so
    # the call spends grace AND answer wait: the sum must still fit the budget.
    async def attach_soon() -> None:
        # Late enough to burn most of the 300 ms grace (half of 600 ms).
        await asyncio.sleep(0.28)
        comms.attach("job-1", FakeChild(), tmp_dir())

    loop = asyncio.get_running_loop()
    started = loop.time()
    attacher = asyncio.create_task(attach_soon())
    reply = await comms.ask("job-1", "status?", 600)
    elapsed = loop.time() - started
    await attacher

    assert reply.timed_out
    # Measured: 601 ms with the deadline honoured, 884 ms without it. The bound
    # sits between the two, with enough slack above the correct figure that a
    # loaded CI box does not flake it.
    assert elapsed < 0.75, f"overshot a 600 ms budget: took {elapsed * 1000:.0f} ms"


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


def test_steer_passes_the_journaled_communication_id_to_the_child():
    """The persisted steer row must carry the SAME id as the journaled
    communication fact.

    Human-facing surfaces (the TUI subagent view, the mobile projection)
    correlate the fact with the replay-visible row by id and render the fact
    instead of the model-facing ``<parent-message>`` envelope. Before this fix
    ``comms.steer`` went through ``steer`` — which mints a fresh Message id —
    so the correlation could never match and the envelope rendered beside the
    fact. The fix routes through the identity-preserving ``steer_message``
    seam with a caller-built Message carrying the fact's id.
    """
    comms, _jobs, child, _parent = wire()

    journaled_ids: list[str | None] = []
    original_journal = comms._journal_communication

    def record_journal(record, *, direction, body, communication_id=None, **kwargs):
        journaled_ids.append(communication_id)
        return original_journal(
            record, direction=direction, body=body, communication_id=communication_id, **kwargs
        )

    comms._journal_communication = record_journal  # type: ignore[method-assign]

    delivery = comms.steer("job-1", "stop after the first failing test")

    assert delivery.outcome == "injected"
    assert len(journaled_ids) == 1
    assert journaled_ids[0] is not None
    assert [message.id for message in child.steer_messages] == [journaled_ids[0]]


@pytest.mark.parametrize(
    ("steer", "expects_reply", "kind"),
    [(True, False, "steer"), (False, True, "ask"), (False, False, "note")],
)
def test_every_envelope_the_builder_emits_round_trips_through_the_parser(
    steer: bool, expects_reply: bool, kind: str
):
    """The builder and its inverse share the tag and instruction constants, so
    every envelope `_format_to_child` emits must parse back to its own kind and
    the exact body. This is the guard that keeps the pair from drifting: a
    reworded instruction that missed `TO_CHILD_INSTRUCTIONS` fails here rather
    than silently disabling extraction on every new transcript."""
    envelope = SubagentComms._format_to_child(
        "Focus on retries", expects_reply=expects_reply, steer=steer
    )

    parsed = extract_parent_message(envelope)

    assert parsed is not None
    assert (parsed.kind, parsed.body) == (kind, "Focus on retries")


@pytest.mark.parametrize(
    "text",
    [
        "just a normal message",
        # The tag shape a HUMAN can type: no builder preamble, so not ours.
        "<parent-message>\nwhy does my log show this?\n\nsecret plan\n</parent-message>",
        # Right tags, no blank-line separator — not the builder's shape.
        "<parent-message>\nFocus on retries</parent-message>",
        # Opening tag with no close.
        "<parent-message>\nThis changes your instructions.\n\nFocus on retries",
    ],
)
def test_text_the_builder_did_not_emit_is_not_extracted(text: str):
    """Extraction is keyed on the exact instruction preamble, not on tag shape
    alone. A user asking about this very wrapper must keep their own words:
    rewriting them as a parent communication would misattribute a human's
    message and strip its text."""
    assert extract_parent_message(text) is None


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


@pytest.mark.asyncio
async def test_a_note_sent_before_the_question_lands_does_not_answer_it():
    """R3, ordering A. A child volunteering "I am blocked" while the parent's
    question is still an un-injected thunk must not resolve it: the child has
    not read the question. Two things used to be destroyed at once — the
    answer was wrong, the question then materialized as a StaleAside and was
    never asked, AND the note the child actually sent was consumed instead of
    delivered."""
    comms, _jobs, child, parent = wire()

    question = asyncio.create_task(comms.ask("job-1", "are you stuck?", 5_000))
    await wait_for(lambda: bool(child.asides))  # queued, NOT yet in context

    outcome = comms.reply_to_parent("job-1", "I am blocked: no staging credentials")

    assert "delivered to the parent" in outcome
    assert "I am blocked" in hub_aside(parent.asides[0]()).details["text"]
    assert not question.done()

    # The question survives, reaches the child, and is answered normally.
    [aside] = child.materialize()
    assert hub_aside(aside).details["expects_reply"] is True
    comms.reply_to_parent("job-1", "not stuck, the fixtures are slow")
    assert (await question).text == "not stuck, the fixtures are slow"


@pytest.mark.asyncio
async def test_an_answer_to_a_timed_out_question_does_not_answer_the_next_one():
    """R3, ordering B. The child is answering Q1, which nobody is waiting for
    any more; Q2 is queued but unread. Accepting it as Q2's answer would hand
    the parent a reply to a question it did not ask."""
    comms, _jobs, child, parent = wire()

    first = await comms.ask("job-1", "how far along are you?", 20)
    assert first.timed_out is True

    second = asyncio.create_task(comms.ask("job-1", "are you stuck?", 5_000))
    await wait_for(lambda: len(child.asides) == 2)

    outcome = comms.reply_to_parent("job-1", "about halfway")  # answering Q1

    assert "delivered to the parent" in outcome
    assert "about halfway" in hub_aside(parent.asides[0]()).details["text"]
    assert not second.done()

    child.materialize()  # Q1 withdraws itself, Q2 lands
    comms.reply_to_parent("job-1", "not stuck")
    assert (await second).text == "not stuck"


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


def test_a_resume_carries_the_childs_role_and_tier_forward(tmp_path, monkeypatch):
    """A resumed child comes back as WHAT IT WAS, not as a generic task.

    ``run_subagent`` defaults ``agent`` to ``"task"``, so a resume that omits
    it silently rebuilt every reviewer, designer, scout and specialist as a
    no-role child. That is a capability regression and not a cosmetic one: the
    role decides the tool ALLOWLIST, so a resumed reviewer regained ``edit``
    and could rewrite the code it was reviewing, and a resumed scout lost its
    read-only promise. It also decides the preamble, the MCP exclusion, and
    the ``agent`` re-stamped into ``origin.json``.

    Asserted at the ``run_subagent`` seam because that is where the fact is
    lost; the end-to-end consequences are exercised in the subagent tests.
    """
    from local_operator.harness import comms as comms_mod

    seen: dict[str, object] = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return "job-2"

    monkeypatch.setattr("local_operator.harness.subagent.run_subagent", spy)
    jobs = FakeJobs()
    parent = FakeParent(jobs)
    # The parent prices the tier exactly as the LAUNCH path does; a resume is
    # a second launch and has to perform the same resolution, or the child
    # comes back on the parent's model while the panel still says `hi`.
    resolved = comms_mod.ModelSpec(provider="anthropic", model_id="claude-opus-5")
    parent._resolve_subagent_model = lambda agent, effort: resolved  # type: ignore[attr-defined]
    comms = SubagentComms(parent)  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    comms.record_launch("job-1", "review-301-r2", agent_role="reviewer", effort="hi")
    comms.attach("job-1", FakeChild(), tmp_path / "child")
    (tmp_path / "child").mkdir(parents=True, exist_ok=True)
    (tmp_path / "child" / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    comms.detach("job-1")

    new_id, error = comms.resume("job-1", "carry on")

    assert error is None and new_id == "job-2"
    assert seen["agent"] == "reviewer"
    assert seen["effort"] == "hi"
    assert seen["model_spec"] is resolved
    assert seen["resume_dir"] == tmp_path / "child"


def test_a_resume_carries_the_mcp_denial_a_role_cannot_express(tmp_path, monkeypatch):
    """An inherited MCP activation denial survives the resume rebuild.

    Review round 2, R5. This is the half ``agent_role`` cannot carry: a denial
    is inherited from the LINEAGE, so the child that leaks is a plain ``task``
    grandchild of a restricted ``manager`` whose own role says "unrestricted".
    ``resume`` also rebuilds with ``parent_session=self._session`` -- the
    comms-owning ROOT, not the child's real parent -- so the parent-session
    read in ``_build_child_session`` finds an unrestricted session too. With
    neither source able to recover the fact, a grandchild that was correctly
    refused an ``approval_tier="exec"`` write while live came back from a
    resume able to activate it.

    Asserted at the ``run_subagent`` seam, matching the role/effort tests
    above: this is where the fact was being dropped. The end-to-end
    consequence is exercised in the subagent tests.
    """
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "local_operator.harness.subagent.run_subagent",
        lambda **kwargs: (seen.update(kwargs), "job-2")[1],
    )
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    # A plain no-role child: the role carries nothing about the denial, which
    # is exactly why the flag has to be persisted separately.
    comms.record_launch("job-1", "worker")
    child = FakeChild()
    # What ``_build_child_session`` stamps on a child built under a denial.
    setattr(child, MCP_DENIED_ATTR, True)
    comms.attach("job-1", child, tmp_path / "child")
    (tmp_path / "child").mkdir(parents=True, exist_ok=True)
    (tmp_path / "child" / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    comms.detach("job-1")

    new_id, error = comms.resume("job-1", "carry on")

    assert error is None and new_id == "job-2"
    assert seen["agent"] == "task", "the role still says unrestricted"
    assert seen["restricted"] is True, "the denial must ride separately from the role"

    # ...and again ACROSS A RESTART, which is the common case: a child that
    # settled hours ago is resumed from the roster sidecar, not from a live
    # ``_records`` map. Resuming in-process only proved the flag reaches
    # ``run_subagent``; it did not prove the flag still exists after
    # ``snapshot()`` (review round 3, R6).
    seen.clear()
    revived = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    revived.restore(comms.snapshot())
    new_id, error = revived.resume("job-1", "carry on after a restart")

    assert error is None and new_id == "job-2"
    assert seen["restricted"] is True, "the denial must survive a restart too"


def test_a_resume_of_an_unrestricted_child_does_not_invent_a_denial(tmp_path, monkeypatch):
    """The counter-check: the carry must not restrict what was never restricted.

    A flag that defaulted to True (or was set from something broader than the
    child's own stamp) would quietly strip MCP activation from every resumed
    ordinary child, which no assertion above would catch."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "local_operator.harness.subagent.run_subagent",
        lambda **kwargs: (seen.update(kwargs), "job-2")[1],
    )
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    comms.record_launch("job-1", "worker")
    comms.attach("job-1", FakeChild(), tmp_path / "child")
    (tmp_path / "child").mkdir(parents=True, exist_ok=True)
    (tmp_path / "child" / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    comms.detach("job-1")

    new_id, error = comms.resume("job-1", "carry on")

    assert error is None and new_id == "job-2"
    assert seen["restricted"] is False


def test_a_resume_of_a_plain_child_stays_a_plain_child(tmp_path, monkeypatch):
    """The no-role default round-trips as the default rather than as ``""``."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "local_operator.harness.subagent.run_subagent",
        lambda **kwargs: (seen.update(kwargs), "job-2")[1],
    )
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    comms.record_launch("job-1", "docs")
    comms.attach("job-1", FakeChild(), tmp_path / "child")
    (tmp_path / "child").mkdir(parents=True, exist_ok=True)
    (tmp_path / "child" / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    comms.detach("job-1")

    new_id, error = comms.resume("job-1", "carry on")

    assert error is None and new_id == "job-2"
    assert seen["agent"] == "task"
    assert seen["effort"] is None
    # A parent that cannot price a tier still resumes, on its own model.
    assert seen["model_spec"] is None


def test_a_restored_record_still_knows_what_its_child_was(tmp_path):
    """The role has to survive the PROCESS, not just the job row.

    ``_ChildRecord`` is in memory, so a parent session that is itself resumed
    rebuilds its children from the persisted snapshot. If the role did not
    round-trip there, resuming a child after a restart would hit exactly the
    same defect from the other side.
    """
    jobs = FakeJobs()
    first = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("job-1", status="cancelled")
    first.record_launch("job-1", "review-301-r2", agent_role="reviewer", effort="hi")
    first.attach("job-1", FakeChild(), tmp_path / "child")
    first.detach("job-1")

    restored = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    restored.restore(first.snapshot())

    record = restored._records["job-1"]
    assert record.agent_role == "reviewer"
    assert record.effort == "hi"


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
async def test_the_tool_refuses_to_peek_several_children_at_once():
    """peek renders ONE transcript window, so like ask it stays single-subject
    even though resume beside it now fans out."""
    comms, jobs, _child, _parent = wire()
    jobs.add("job-2")
    comms.record_launch("job-2", "docs")
    comms.attach("job-2", FakeChild(), tmp_dir())

    result = await execute_hub(
        "call-1",
        {"op": "peek", "to": ["all"]},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    assert result.is_error
    assert "one subagent at a time" in body(result)


def _stopped_resumable(comms, jobs, tmp_path, job_id: str, label: str) -> None:
    """Bring a child into the exact state ``comms.resume`` will act on: a
    settled job row, a session dir with a transcript on disk, and detached so
    no live twin blocks the resume."""
    jobs.add(job_id, status="cancelled")
    comms.record_launch(job_id, label)
    session_dir = tmp_path / job_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    comms.attach(job_id, FakeChild(), session_dir)
    comms.detach(job_id)


@pytest.mark.asyncio
async def test_resume_fans_one_message_out_to_several_stopped_children(tmp_path, monkeypatch):
    """The point of the fan-out: a batch that all stalled comes back in one
    call, each as its own new job replaying its own transcript."""
    spawned: list[str] = []

    def spy(**kwargs):
        new_id = f"resumed-{len(spawned) + 1}"
        spawned.append(new_id)
        return new_id

    monkeypatch.setattr("local_operator.harness.subagent.run_subagent", spy)
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    _stopped_resumable(comms, jobs, tmp_path, "job-1", "parser")
    _stopped_resumable(comms, jobs, tmp_path, "job-2", "docs")

    result = await execute_hub(
        "call-1",
        {"op": "resume", "to": ["job-1", "job-2"], "message": "carry on"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    text = body(result)
    assert "resume: 2/2 subagent(s)" in text
    # Both targets are named beside the NEW job id they were resumed as.
    assert "parser (job-1): resumed as job resumed-1" in text
    assert "docs (job-2): resumed as job resumed-2" in text
    details = payload(result)
    assert details["job_ids"] == ["resumed-1", "resumed-2"]
    assert details["resumed_from"] == ["job-1", "job-2"]
    assert details["acted"] == 2
    assert result.useless is False


@pytest.mark.asyncio
async def test_resume_batch_reports_success_and_failure_side_by_side(tmp_path, monkeypatch):
    """A mixed batch must not fail the whole call: the resumable child comes
    back and the one that refuses carries its reason on its own line."""
    monkeypatch.setattr(
        "local_operator.harness.subagent.run_subagent",
        lambda **kwargs: "resumed-1",
    )
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    _stopped_resumable(comms, jobs, tmp_path, "job-1", "parser")
    # A still-running child is not resumable and refuses with a reason.
    jobs.add("job-2", status="running")
    comms.record_launch("job-2", "docs")
    comms.attach("job-2", FakeChild(), tmp_path / "job-2")

    result = await execute_hub(
        "call-1",
        {"op": "resume", "to": ["job-1", "job-2"], "message": "carry on"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    text = body(result)
    assert "resume: 1/2 subagent(s)" in text
    assert "parser (job-1): resumed as job resumed-1" in text
    assert "docs (job-2): failed" in text
    assert "still running" in text
    details = payload(result)
    assert details["job_ids"] == ["resumed-1"]
    assert details["resumed_from"] == ["job-1"]
    assert details["acted"] == 1
    # One child was reached, so the receipt is a real observation, not useless.
    assert result.useless is False


@pytest.mark.asyncio
async def test_a_resume_batch_that_reaches_nobody_is_flagged_useless(tmp_path):
    """Pure-failure fan-out mirrors send/steer/cancel: nothing was resumed, so
    the receipt is flagged useless for compaction to prune."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    # Two running children: both refuse, so the batch reaches nobody.
    jobs.add("job-1", status="running")
    comms.record_launch("job-1", "parser")
    comms.attach("job-1", FakeChild(), tmp_path / "job-1")
    jobs.add("job-2", status="running")
    comms.record_launch("job-2", "docs")
    comms.attach("job-2", FakeChild(), tmp_path / "job-2")

    result = await execute_hub(
        "call-1",
        {"op": "resume", "to": ["job-1", "job-2"], "message": "carry on"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    assert "resume: 0/2 subagent(s)" in body(result)
    assert payload(result)["acted"] == 0
    assert result.useless is True


@pytest.mark.asyncio
async def test_the_tool_accepts_the_string_forms_models_send_for_to():
    """Observed live (2026-08-19): a parent model retried ``op='ask'`` against
    a running reviewer and never once emitted a real array — a bare id, the
    JSON of the list as a string (which the TUI then prints so it *looks*
    like a list), and the bracketed id without quotes. Every attempt failed
    ``to: Input should be a valid list`` while the child worked unheard. The
    before-validator coerces those shapes; a real array is untouched."""
    comms, _jobs, child, _parent = wire()
    context = ToolContext(cwd=".", subagent_comms=comms)

    for to_value in ('["job-1"]', "[job-1]", "job-1"):
        result = await execute_hub(
            "call-1", {"op": "send", "to": to_value, "message": "hi"}, None, None, context
        )
        assert not result.is_error, to_value
        assert child.asides, to_value
        child.asides.clear()

    listed = await execute_hub("call-2", {"op": "list"}, None, None, context)
    assert not listed.is_error

    for bad_to in (42, ""):
        bad = await execute_hub(
            "call-3", {"op": "send", "to": bad_to, "message": "hi"}, None, None, context
        )
        assert bad.is_error and "valid list" in body(bad), bad_to

    # '[null]' drops its non-string item and fails as a missing target, not
    # as a fabricated "None" target name.
    nulls = await execute_hub(
        "call-4", {"op": "send", "to": "[null]", "message": "hi"}, None, None, context
    )
    assert nulls.is_error and "needs a 'to' target" in body(nulls)


def test_the_hub_schema_still_advertises_a_nullable_array_not_a_union():
    """The coercion lives in a validator because a ``str | list[str]`` schema
    would render a non-nullable ``anyOf`` — the one construct the provider
    matrix rejects for every request in the session. Pin the nullable-array
    shape the comment on ``HubParams.to`` promises."""
    tool = hub_tool(job_id=None, comms=wire()[0])
    kinds = [entry.get("type") for entry in tool.parameters["properties"]["to"]["anyOf"]]
    assert kinds == ["array", "null"]


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
        if "background job" in body:
            # Job auto-delivery now re-wakes the TOP-LEVEL parent. The fixture's
            # old default is an intentional infinite bash loop (it models a
            # busy child), so without an acknowledgement for this new parent
            # turn it spins forever after the child has already completed.
            return text_stream("job result received")
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
    communication_rows = [
        entry.payload["details"]
        for entry in Transcript(comms.session_dir_of(job_id)).entries()
        if entry.payload.get("custom_type") == "hub_communication"
    ]
    question = next(row for row in communication_rows if row.get("kind") == "ask")
    answer = next(row for row in communication_rows if row.get("direction") == "to_parent")
    assert question["body"] == "are you stuck?"
    assert answer["body"] == "not stuck; fixtures are slow"
    assert answer["reply_to"] == question["communication_id"]
    assert "<parent-message>" not in question["body"]
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
    # Every wait below rides the child's own publications rather than a
    # deadline (#122): both sources are watched because the parent stream
    # carries attach and settle while the detail registry carries each durable
    # transcript append, and this test waits on all three kinds of fact.
    signal = ChangeSignal().watch_comms(comms).watch_session(parent)
    try:
        await wait_for(lambda: comms.session_dir_of(job_id) is not None, signal=signal)
        # Let it finish real work before stopping it, and wait on THAT rather
        # than on a duration: the child persists each tool round at its loop
        # boundary (``Session._persist_progress``) roughly 0.29s after launch,
        # so the fixed 0.3s sleep this originally replaced was inside one
        # standard deviation of the thing it was betting against. Resume is
        # only meaningful over a child that got somewhere, so the condition is
        # a completed round — and it is now awaited on the append that creates
        # it, so a loaded machine makes this slower, never red.
        await wait_for(lambda: has_complete_tool_round(comms.session_dir_of(job_id)), signal=signal)
        await comms.cancel(job_id)
        await wait_for(lambda: status_of(parent, job_id) == "cancelled", signal=signal)
        await wait_for(
            lambda: len(Transcript(comms.session_dir_of(job_id)).build_llm_history()) >= 2,
            signal=signal,
        )
        before = Transcript(comms.session_dir_of(job_id)).build_llm_history()
        original = parent.jobs.get(job_id)
        assert original is not None and original.launch_message_id == before[0].id

        new_id, error = comms.resume(job_id, "You were interrupted. Wrap up.")

        assert error is None and new_id is not None
        continuation = parent.jobs.get(new_id)
        assert continuation is not None and continuation.launch_message_id
        assert continuation.launch_message_id != original.launch_message_id
        await wait_for(lambda: status_of(parent, new_id) != "running", signal=signal)
        assert comms.session_dir_of(new_id) == comms.session_dir_of(job_id)
        after = Transcript(comms.session_dir_of(new_id)).build_llm_history()
        assert len(after) > len(before)
        assert any(message.id == continuation.launch_message_id for message in after)
        resumed_node = comms.node(new_id)
        assert resumed_node is not None
        assert resumed_node.launch_message_id == continuation.launch_message_id
        assert first_text(after) == first_text(before) == "Update the docs."
        assert status_of(parent, new_id) == "completed"
    finally:
        signal.close()
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
    signal = ChangeSignal().watch_comms(comms).watch_session(parent)
    try:
        await wait_for(lambda: comms.session_dir_of(job_id) is not None, signal=signal)
        # Wait for the round itself, not for how long a round usually takes.
        # The assertions below are entirely about work the child had ALREADY
        # completed before the cancel, so cancelling early does not fail
        # loudly — it silently asserts over a transcript holding only the
        # launch prompt.
        await wait_for(lambda: has_complete_tool_round(comms.session_dir_of(job_id)), signal=signal)

        await comms.cancel(job_id)
        await wait_for(
            lambda: len(Transcript(comms.session_dir_of(job_id)).build_llm_history()) > 1,
            signal=signal,
        )
    finally:
        signal.close()

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


# --- roster (op='list') and pause ---------------------------------------------
#
# The gap these close: before them the only ways to enumerate children were
# ``resolve("all")``, which returns RUNNING ones only, and the ``jobs`` tool,
# whose rows the manager sweeps a few minutes after they settle. A child that
# failed or was stopped therefore became unaddressable long before it stopped
# being resumable, so the case an operator most wants to act on — a stuck or
# crashed subagent — was the one they could not see.


def test_the_roster_lists_a_running_child_as_not_resumable():
    """Resumability is the opposite of the intuitive reading of the status, so
    the roster states it rather than leaving it to be inferred: a RUNNING child
    is the one that cannot be resumed."""
    comms, _jobs, _child, _parent = wire()

    [row] = comms.roster()

    assert row.job_id == "job-1" and row.label == "parser"
    assert row.status == "running"
    assert row.resumable is False
    assert "still running" in (row.detail or "")


def test_the_roster_still_lists_a_child_whose_job_row_was_swept(tmp_path):
    """The whole point of the feature. Job rows are swept minutes after they
    settle while comms records outlive them so a child stays resumable; a
    roster that read the row would go blank exactly when resume still works."""
    comms, jobs, _child, _parent = wire()
    comms.attach("job-1", FakeChild(), tmp_path)
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.record_outcome("job-1", "failed", "provider 500")
    comms.detach("job-1")
    del jobs.jobs["job-1"]  # what AsyncJobManager._sweep_due does

    [row] = comms.roster()

    assert row.status == "failed"
    assert row.error_text == "provider 500"
    assert row.result_text is None
    assert row.resumable is True
    assert "provider 500" in (row.detail or "")


@pytest.mark.parametrize(
    ("first", "second", "expected", "result", "error"),
    [
        ("completed", "cancelled", "completed", "finished report", None),
        ("failed", "cancelled", "failed", None, "provider 500"),
        ("cancelled", "failed", "failed", None, "provider 500"),
        ("failed", "completed", "completed", "finished report", None),
    ],
)
def test_terminal_outcomes_only_advance_in_precedence(first, second, expected, result, error):
    """Late runner observations cannot regress a stronger terminal fact.

    ``completed`` wins because a returned result proves the work finished;
    ``failed`` next because a captured exception proves how it stopped; bare
    cancellation is weakest because it only says the task was interrupted.
    """
    comms, _jobs, _child, _parent = wire()
    payloads = {
        "completed": {"result_text": "finished report"},
        "failed": {"error_text": "provider 500"},
        "cancelled": {},
    }

    comms.record_outcome("job-1", first, **payloads[first])
    comms.record_outcome("job-1", second, **payloads[second])
    [row] = comms.roster()

    assert row.status == expected
    assert row.result_text == result
    assert row.error_text == error


def test_completed_payload_survives_snapshot_restore_and_job_sweep(tmp_path):
    comms, _jobs, _child, _parent = wire()
    comms.attach("job-1", FakeChild(), tmp_path)
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.record_outcome("job-1", "completed", result_text="finished report")
    comms.detach("job-1")

    restored = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    restored.restore(comms.snapshot())
    [row] = restored.roster()

    assert row.status == "completed"
    assert row.result_text == "finished report"
    assert row.error_text is None


def test_the_roster_reports_a_never_started_child_as_unresumable():
    """No transcript means no context to continue, and a resume that quietly
    starts a stranger is worse than none — so the roster says so up front
    rather than letting the caller discover it from a failed resume."""
    comms, jobs, _child, _parent = wire(attach=False)
    jobs.jobs["job-1"].status = "cancelled"

    [row] = comms.roster()

    assert row.resumable is False
    assert "never started" in (row.detail or "")


def test_the_roster_marks_a_child_whose_transcript_is_gone(tmp_path):
    """``resumable`` must agree with what ``resume`` will actually do; a row
    promising a resume that then refuses is worse than an honest refusal."""
    comms, jobs, _child, _parent = wire()
    comms.attach("job-1", FakeChild(), tmp_path / "missing")
    comms.record_outcome("job-1", "completed")
    comms.detach("job-1")
    # The manager stamps the row once the runner returns. Without this the
    # record is settled while the row still says running, which is a real
    # (brief) state but a different one -- and it is reported as "still
    # settling", not as the missing transcript this test is about.
    jobs.jobs["job-1"].status = "completed"

    [row] = comms.roster()

    assert row.resumable is False
    assert "gone from disk" in (row.detail or "")
    # ...and the two surfaces agree, which is the invariant that matters.
    assert comms.resume("job-1", "carry on")[0] is None


def test_the_roster_replaces_a_terminal_attempt_with_its_continuation(tmp_path):
    """A resumed transcript is one logical child, not one row per attempt."""
    comms, jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)
    comms.record_outcome("job-1", "cancelled")
    comms.detach("job-1")
    jobs.jobs["job-1"].status = "cancelled"
    # The continuation, live on the same directory.
    jobs.add("job-2")
    comms.record_launch("job-2", "parser")
    comms.attach("job-2", FakeChild(), tmp_path)

    rows = comms.roster()

    assert [row.job_id for row in rows] == ["job-2"]
    assert comms.session_dir_of("job-1") == tmp_path


@pytest.mark.asyncio
async def test_pause_stops_the_child_and_keeps_it_resumable(tmp_path):
    """A pause is a cancel plus intent: same physical state, but the roster
    has to keep saying which one the parent meant."""
    comms, jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)

    delivery = await comms.pause("job-1")

    assert delivery.outcome == "paused"
    assert jobs.jobs["job-1"].status == "cancelled"  # mechanically a cancel
    comms.record_outcome("job-1", "cancelled")  # what the runner's teardown does
    comms.detach("job-1")
    [row] = comms.roster()
    assert row.status == "paused"  # ...but reported as the intent, not the mechanism
    assert row.resumable is True


@pytest.mark.asyncio
async def test_resume_ends_the_pause(tmp_path, monkeypatch):
    """Left set, the stopped record would advertise ``paused`` forever beside
    the running child that replaced it, inviting a resume of a live run."""
    comms, jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)
    await comms.pause("job-1")
    comms.record_outcome("job-1", "cancelled")
    comms.detach("job-1")
    monkeypatch.setattr(
        "local_operator.harness.subagent.run_subagent",
        lambda **kwargs: jobs.add("job-2").id,
    )

    new_id, error = comms.resume("job-1", "carry on")

    assert error is None and new_id == "job-2"
    rows = {row.job_id: row for row in comms.roster()}
    assert rows["job-1"].status != "paused"


@pytest.mark.asyncio
async def test_pause_refuses_a_child_with_no_transcript():
    """A pause that cannot be resumed is a cancel wearing the wrong name; the
    caller should have to say what it means."""
    comms, _jobs, _child, _parent = wire(attach=False)

    delivery = await comms.pause("job-1")

    assert delivery.outcome == "failed"
    assert "no transcript to pause" in (delivery.error or "")


@pytest.mark.asyncio
async def test_pause_reports_an_already_settled_child_rather_than_acting():
    comms, jobs, _child, _parent = wire()
    jobs.jobs["job-1"].status = "completed"

    delivery = await comms.pause("job-1")

    assert delivery.outcome == "failed"
    assert "already completed" in (delivery.error or "")
    assert "resume" in (delivery.error or "")


@pytest.mark.asyncio
async def test_the_list_op_needs_no_target_and_names_every_child(tmp_path):
    """``to`` is optional only for ``list``; the schema had it required, which
    is why this op could not simply reuse the existing dispatch."""
    comms, jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    jobs.add("job-2")
    comms.record_launch("job-2", "docs")
    comms.attach("job-2", FakeChild(), tmp_path)
    comms.record_outcome("job-2", "failed", "boom")
    comms.detach("job-2")
    jobs.jobs["job-2"].status = "failed"  # the manager stamps the row last

    result = await execute_hub(
        "call-1", {"op": "list"}, None, None, ToolContext(cwd=".", subagent_comms=comms)
    )

    text = body(result)
    assert not result.is_error
    assert "parser" in text and "docs" in text
    assert "failed" in text and "boom" in text
    assert payload(result)["count"] == 2
    assert {entry["job_id"] for entry in payload(result)["children"]} == {"job-1", "job-2"}


@pytest.mark.asyncio
async def test_the_list_op_says_so_when_nothing_was_launched():
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]

    result = await execute_hub(
        "call-1", {"op": "list"}, None, None, ToolContext(cwd=".", subagent_comms=comms)
    )

    assert "no subagents" in body(result)
    # Nothing to act on: compaction should not keep this in context.
    assert result.useless is True
    assert payload(result)["useless"] is True


@pytest.mark.asyncio
async def test_every_op_but_list_still_requires_a_target():
    """Making ``to`` optional for ``list`` must not make it optional for the
    ops that act on a child — a send with no target would otherwise be a
    silent no-op."""
    comms, _jobs, _child, _parent = wire()

    result = await execute_hub(
        "call-1",
        {"op": "send", "message": "hello"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    assert result.is_error
    assert "needs a 'to' target" in body(result)


@pytest.mark.asyncio
async def test_the_pause_op_needs_no_body(tmp_path):
    comms, _jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)

    result = await execute_hub(
        "call-1",
        {"op": "pause", "to": ["job-1"]},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    assert not result.is_error
    assert "paused" in body(result)


@pytest.mark.asyncio
async def test_a_paused_child_survives_its_swept_job_row_and_resumes(tmp_path, monkeypatch):
    """The end-to-end claim the roster and pause exist for, on real sessions.

    A child is paused mid-run; its job row is then swept the way retention
    sweeps it minutes later. The roster must still show the child, still call
    it resumable, and a resume driven from that roster must replay the paused
    run's transcript rather than starting a stranger.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_parent(tmp_path, ScriptedProvider())
    await parent.async_init()
    job_id = parent._launch_subagent(label="docs", prompt="Update the docs.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)

    # Wait on the CONDITION, not the clock: the point is that the child has
    # actually done a round of work worth resuming, and a fixed sleep is the
    # line that goes first on a loaded CI box. The transcript cannot be the
    # probe here — ``Session.prompt`` persists a turn when the turn ENDS, so it
    # stays at one entry until the pause below pre-empts it (measured: still 1
    # after 1.5 s) — so this watches the job's live trajectory instead, which
    # the runner appends to as the child completes tool calls.
    # A COMPLETED turn, not merely a non-empty trajectory: the trajectory is
    # populated at ``agent_start``, long before any tool has run, so a length
    # probe would pass instantly and the pause would pre-empt the child before
    # it had produced anything worth replaying.
    def finished_a_turn() -> bool:
        job = parent.jobs.get(job_id)
        return any(
            entry.get("type") == "turn_end" for entry in (getattr(job, "trajectory", None) or [])
        )

    await wait_for(finished_a_turn)

    paused = await comms.pause(job_id)
    assert paused.outcome == "paused"
    await wait_for(lambda: status_of(parent, job_id) == "cancelled")
    await wait_for(lambda: len(Transcript(comms.session_dir_of(job_id)).build_llm_history()) >= 2)
    before = Transcript(comms.session_dir_of(job_id)).build_llm_history()

    # Exactly what AsyncJobManager._sweep_due does once retention elapses.
    # Reaching for the private dict rather than driving the real sweep because
    # ``Session`` builds its own manager with no retention knob to shorten;
    # the equivalence was verified against the real ``_sweep_due`` (row gone,
    # signals cleared, roster unchanged).
    parent.jobs._jobs.pop(job_id)
    assert parent.jobs.get(job_id) is None

    [row] = [row for row in comms.roster() if row.job_id == job_id]
    assert row.status == "paused"  # not "gone", and not "cancelled"
    assert row.resumable is True

    new_id, error = comms.resume(job_id, "You were interrupted. Wrap up.")
    assert error is None and new_id is not None
    await wait_for(lambda: status_of(parent, new_id) != "running")

    after = Transcript(comms.session_dir_of(new_id)).build_llm_history()
    assert len(after) > len(before)
    assert first_text(after) == first_text(before) == "Update the docs."
    assert status_of(parent, new_id) == "completed"
    await parent.dispose()


@pytest.mark.asyncio
async def test_the_roster_records_a_real_childs_failure(tmp_path, monkeypatch):
    """A crashed child is the case an operator most wants to find later, so
    the roster must carry its failure across the job row's sweep."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    async def exploding(request, signal=None):
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover - makes this an async generator

    parent = make_parent(tmp_path, exploding)
    await parent.async_init()
    job_id = parent._launch_subagent(label="doomed", prompt="Do the thing.")
    comms = parent.subagent_comms
    await wait_for(lambda: status_of(parent, job_id) != "running")

    parent.jobs._jobs.pop(job_id)  # retention sweep
    [row] = [row for row in comms.roster() if row.job_id == job_id]

    assert row.status == "failed"
    assert "provider exploded" in (row.detail or "")
    await parent.dispose()


def test_the_roster_does_not_promise_a_resume_during_the_pause_window(tmp_path):
    """Roster and ``resume`` must agree even in a paused-but-still-running state.

    This state is NOT currently reachable in production, and the test says so
    rather than implying otherwise: ``pause`` sets the flag and then awaits
    ``jobs.cancel``, which stamps the row before its first suspension point, so
    no concurrent ``list`` can observe the gap. The state is constructed here
    directly.

    It is worth a test anyway, because the guard's job is to make the
    roster/``resume`` invariant hold structurally rather than by luck about
    where an ``await`` sits in ``AsyncJobManager``. If someone later adds one
    inside ``cancel``, this is what stops the divergence reopening silently.
    """
    comms, _jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)
    # Constructed, not raced into: see the docstring.
    comms._records["job-1"].paused = True

    [row] = comms.roster()

    assert row.status == "pausing"
    assert row.resumable is False
    # The invariant that matters: the two surfaces give the same answer.
    assert comms.resume("job-1", "carry on")[0] is None


def test_the_roster_does_not_promise_a_resume_during_the_settle_window(tmp_path):
    """F1: ``resumable`` and ``resume()`` must consult the same fact.

    ``record_outcome`` runs inside the runner's settle path, which then still
    awaits ``emit(SubagentEndEvent)`` — the parent's whole handler fan-out —
    before returning, and only then does the manager stamp the job row. For
    that stretch the record says ``failed`` while the row says ``running``.
    Deriving ``resumable`` from the status alone advertised "failed —
    resumable" while ``resume()``, which reads the row via ``_is_running``,
    refused with "still running". Measured at ~256 ms with one 250 ms handler:
    wide enough for a concurrent ``list`` in the same tool batch to land in.
    """
    comms, _jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)
    # Exactly the window: the outcome is recorded, the row is not yet stamped.
    comms.record_outcome("job-1", "failed", "provider 500")

    [row] = comms.roster()

    assert row.status == "failed"  # the status string is correct...
    assert row.resumable is False  # ...and must not invite a resume yet
    # The invariant: whatever the roster promises, resume() must deliver.
    assert comms.resume("job-1", "carry on")[0] is None


@pytest.mark.asyncio
async def test_cancelling_a_paused_child_abandons_the_pause(tmp_path):
    """F2: a pause must not be a one-way door.

    Only ``resume`` used to clear the flag, so a parent that paused a child and
    then decided it was wrong had no way to say so: ``cancel`` refused a job
    that was already stopped, and the roster went on advertising a pause the
    parent had abandoned.
    """
    comms, jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)
    await comms.pause("job-1")
    comms.record_outcome("job-1", "cancelled")
    comms.detach("job-1")
    assert comms.roster()[0].status == "paused"

    delivery = await comms.cancel("job-1")

    assert delivery.outcome == "cancelled"
    [row] = comms.roster()
    assert row.status == "cancelled"  # no longer advertised as paused
    # The transcript is untouched, so anyone holding the id can still resume.
    assert row.resumable is True


def test_an_unknown_status_is_not_born_resumable(tmp_path):
    """F4: resumability is enumerated, not defaulted.

    A status nobody has reasoned about is one the parent should not be invited
    to resume; the old default meant any status added to ``_describe`` was
    silently born resumable.
    """
    comms, jobs, _child, _parent = wire()
    # A real transcript on disk and a settled record, so every EARLIER branch
    # (still running, no session_dir, transcript missing, live twin) is passed
    # and the enumeration is genuinely what decides. Without this the test
    # passed on the old fail-open code too, proving nothing: ``session_dir is
    # None`` short-circuited long before the status was consulted.
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)
    comms.detach("job-1")
    jobs.jobs["job-1"].status = "some_future_state"

    [row] = comms.roster()

    assert row.status == "some_future_state"  # reached the enumeration...
    assert row.resumable is False  # ...and was not born resumable


def test_a_swept_row_with_a_transcript_is_still_resumable(tmp_path):
    """R3: ``gone`` says the job row was swept, not that the work is lost.

    ``resume`` asks only whether the record has a readable transcript and no
    live twin — it never consults the row's status. Leaving ``gone`` out of the
    resumable enumeration therefore refused a resume that would have succeeded:
    the same roster/``resume`` disagreement as F1, pointing the safe way. It is
    still a lie about the child, and this is the state a parent reaches by
    coming back to a long-settled subagent, which is the feature's whole point.
    """
    comms, jobs, _child, _parent = wire()
    (tmp_path / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.attach("job-1", FakeChild(), tmp_path)
    # Row swept with no recorded outcome: the ``gone`` fallback.
    del jobs.jobs["job-1"]

    [row] = comms.roster()

    assert row.status == "gone"
    assert row.resumable is True
    # And the invariant: nothing in resume() disagrees.
    assert comms._live_twin(comms._records["job-1"]) is None


def test_the_roster_carries_the_session_id_a_resume_would_take(tmp_path):
    """The roster is now the ONLY surface that can show a child's session id.

    Children are deliberately kept out of the `/resume` picker — they are the
    machine's own runs, not the user's conversations — and the job row that
    carries `job_id` is swept minutes after a child settles. Without the
    session id here, an operator investigating a subagent that crashed an hour
    ago has no in-product path to its transcript, which is precisely the case
    this roster exists to cover.

    Note it is NOT the job id: `--resume` takes the transcript directory name.
    """
    session_dir = tmp_path / "9f2c1a0b7e44"
    session_dir.mkdir()
    comms, jobs, _child, _parent = wire()
    comms.attach("job-1", FakeChild(), session_dir)
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.record_outcome("job-1", "failed", "provider 500")
    comms.detach("job-1")
    del jobs.jobs["job-1"]  # the sweep that used to take the id with it

    [row] = comms.roster()

    assert row.resumable is True
    assert row.session_id == "9f2c1a0b7e44"
    assert row.session_id != row.job_id


def test_a_child_that_never_started_has_no_session_id_to_offer():
    """No transcript directory, so there is nothing to print. A blank is
    honest; the job id in its place would be an id that resumes nothing."""
    comms, jobs, _child, _parent = wire(attach=False)
    jobs.jobs["job-1"].status = "cancelled"

    [row] = comms.roster()

    assert row.session_id is None


def test_the_roster_text_tells_the_two_ids_apart(tmp_path):
    """Both ids are `uuid4().hex[:12]`, so they are visually identical.

    The roster prints a job id and a transcript id two lines apart, and the
    line under them said "Resume one with hub op='resume'" — which takes the
    JOB id and rejects the transcript id with `unknown subagent`. Two
    indistinguishable ids beside an instruction that fits only one is a
    coin-flip, so each id now names what it is for.
    """
    from local_operator.tools.builtin import _hub_list

    session_dir = tmp_path / "9f2c1a0b7e44"
    session_dir.mkdir()
    comms, jobs, _child, _parent = wire()
    comms.attach("job-1", FakeChild(), session_dir)
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n")
    comms.record_outcome("job-1", "failed", "provider 500")
    comms.detach("job-1")
    del jobs.jobs["job-1"]

    block = _hub_list("call-1", comms).content[0]
    assert isinstance(block, TextContent)
    text = block.text

    # The transcript id is shown with the command that actually takes it.
    assert "lop --resume 9f2c1a0b7e44" in text
    # And the resume instruction says which id it wants, so the transcript id
    # sitting above it is not read as the argument.
    assert "JOB id" in text
    assert "not a job id" in text


# --- the running-but-unattached window ----------------------------------------
#
# ``_await_child`` already covers the child that is merely SCHEDULED: it yields
# the loop, which is what lets the runner start, so a parent asking its brand
# new child a question gets an answer instead of a refusal.
#
# It does not cover the child whose runner is well underway but whose session
# has not attached (a slow child build, a resumed child replaying a large
# transcript, or a parent monopolising the loop). The grace is capped at
# ``ATTACH_WAIT_MAX_S``, so even a 300 s request waits 30 s and is then told
# the subagent "has not started yet ... retry in a moment" — about an agent
# half an hour into its work. Observed live, and acted on: a healthy reviewer
# subagent was cancelled on the strength of that message.


def test_the_not_started_reason_never_denies_a_running_child():
    """``started_at`` is the authoritative "has the runner begun". A child that
    has been working for half an hour must not be described as unstarted."""
    comms, jobs, _child, _parent = wire(attach=False)
    jobs.jobs["job-1"].started_at = time.time() - 1800

    reason = comms._not_started_reason(comms._records["job-1"])

    assert "has not started" not in reason, reason
    assert "1800s ago" in reason and "RUNNING" in reason
    # And it must steer away from the action that destroyed the work.
    assert "cancelling" in reason


def test_the_not_started_reason_still_distinguishes_parked_from_scheduled():
    """The two genuine not-started states keep their own advice: a parked child
    may not run for minutes (do not wait), a scheduled one starts on the next
    yield (retry)."""
    comms, jobs, _child, _parent = wire(attach=False)
    jobs.jobs["job-1"].started_at = None

    scheduled = comms._not_started_reason(comms._records["job-1"])
    jobs.jobs["job-1"].queued = True
    parked = comms._not_started_reason(comms._records["job-1"])

    assert "session next yields" in scheduled
    assert "capacity gate" in parked and "do not wait on it" in parked


@pytest.mark.asyncio
async def test_ask_buffers_rather_than_refusing_a_started_but_unattached_child():
    """The reported bug. Past the attach grace, with the runner underway, the
    question is BUFFERED for the child instead of refused — and the caller
    spends its own budget waiting, not 30 s waiting to be told no."""
    comms, jobs, _child, _parent = wire(attach=False)
    jobs.jobs["job-1"].started_at = time.time() - 1800

    seen: list[str] = []
    original = comms._withdraw_pending

    def spy(record, message):
        # The buffer is WITHDRAWN when the ask gives up (round 2, R5), so the
        # queue is empty by the time ``ask`` returns. Capture it mid-flight
        # instead of asserting on the wreckage afterwards.
        seen.append(message.details["body"])
        return original(record, message)

    comms._withdraw_pending = spy  # type: ignore[method-assign]

    reply = await comms.ask("job-1", "status?", 400)

    assert reply.error is None, f"refused a working child: {reply.error!r}"
    assert reply.timed_out is True  # nobody attached inside the budget
    assert seen == ["status?"], "the question was never buffered for the child"
    assert not comms._records["job-1"].pending, "an abandoned question was left queued"


@pytest.mark.asyncio
async def test_a_question_buffered_past_the_grace_is_answered_once_it_attaches():
    """End to end: ``attach`` must re-arm the buffered question with the
    waiter's future. Flushed as a plain note it would be injected and answered
    while the asker sat out its whole timeout on a reply already returned."""
    comms, jobs, child, _parent = wire(attach=False)
    jobs.jobs["job-1"].started_at = time.time() - 1800

    # 4 s budget -> a 2 s attach grace, so the question buffers well inside
    # ``wait_for``'s default. (A 20 s budget would spend 10 s in the grace and
    # time the WAIT out, not the ask.)
    asking = asyncio.create_task(comms.ask("job-1", "stuck?", 4_000))
    await wait_for(lambda: bool(comms._records["job-1"].pending))

    comms.attach("job-1", child, tmp_dir())  # the window finally closes
    await wait_for(lambda: bool(child.asides))
    child.materialize()

    result = await execute_hub(
        "call-1",
        {"message": "no, just slow"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms, job_id="job-1"),
    )

    assert "answered the parent's question" in body(result)
    reply = await asking
    assert reply.error is None and reply.timed_out is False, f"{reply.error!r}"
    assert reply.text == "no, just slow"


@pytest.mark.asyncio
async def test_a_stale_buffered_question_cannot_answer_the_next_asker():
    """A timed-out buffered question must be WITHDRAWN, not left in the queue.

    ``_thunk``'s identity check exists so a question whose asker gave up cannot
    arm the next question's future. The buffered path defeated it by binding at
    flush time — ``attach`` asked "is this a question?" and reached for
    whatever ``record.ask`` then held, so the parent asked "are you blocked?"
    and was handed the answer to "what is your ETA?", with ``error=None`` and
    ``timed_out=False``. Wrong and indistinguishable from right (round 2, R5).
    """
    comms, jobs, child, _parent = wire(attach=False)
    jobs.jobs["job-1"].started_at = time.time() - 1800

    stale = await comms.ask("job-1", "OLD: what is your ETA?", 200)
    assert stale.timed_out is True
    assert not comms._records["job-1"].pending, "the abandoned question was not withdrawn"

    # 4 s budget -> 2 s attach grace, comfortably inside ``wait_for``.
    asking = asyncio.create_task(comms.ask("job-1", "NEW: are you blocked?", 4_000))
    await wait_for(lambda: bool(comms._records["job-1"].pending))

    comms.attach("job-1", child, tmp_dir())
    await wait_for(lambda: bool(child.asides))
    materialized = child.materialize()
    assert all(isinstance(message, CustomMessage) for message in materialized)
    bodies = [
        message.details["body"] for message in materialized if isinstance(message, CustomMessage)
    ]
    assert bodies == ["NEW: are you blocked?"], f"a stale question reached the child: {bodies}"

    await execute_hub(
        "call-1",
        {"message": "no, not blocked"},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms, job_id="job-1"),
    )

    reply = await asking
    assert reply.text == "no, not blocked", "the asker got someone else's answer"


@pytest.mark.asyncio
async def test_a_second_question_is_refused_on_the_buffered_path_too():
    """The concurrent-ask guard has to cover BOTH paths.

    The buffered path used to overwrite a live ``record.ask``, orphaning the
    first caller: nothing held its future afterwards, so it could receive
    neither an answer nor a failure and burned its whole budget — up to the
    600 s schema maximum (round 2, R6).
    """
    comms, jobs, _child, _parent = wire(attach=False)
    jobs.jobs["job-1"].started_at = time.time() - 1800

    first = asyncio.create_task(comms.ask("job-1", "Q-A", 4_000))
    # Past the attach grace (half the budget, so 2 s here), where Q-A buffers.
    await wait_for(lambda: bool(comms._records["job-1"].pending))
    held = comms._records["job-1"].ask

    second = await comms.ask("job-1", "Q-B", 4_000)

    assert second.error == "a question is already pending for this subagent"
    assert comms._records["job-1"].ask is held, "the first caller's future was replaced"
    assert [m.details["body"] for m in comms._records["job-1"].pending] == ["Q-A"]
    first_reply = await first
    assert first_reply.timed_out is True  # it ended on its OWN terms, not orphaned


@pytest.mark.asyncio
async def test_two_concurrent_asks_cannot_both_pass_the_guard():
    """The guard is check-then-act, so the buffered path re-checks after its
    attach grace.

    The top-of-``ask`` guard runs BEFORE the grace await, and the buffered path
    only claims ``record.ask`` after it — so two asks entering together both
    passed and the second orphaned the first (review round 4, R8). Unreachable
    through the `hub` tool today, which is ``concurrency="exclusive"`` and is
    therefore batched alone, but that is a property of a tool declaration
    elsewhere rather than an invariant of this layer, and it becomes reachable
    the moment `hub` goes ``shared``.
    """
    comms, jobs, _child, _parent = wire(attach=False)
    jobs.jobs["job-1"].started_at = time.time() - 1800

    first, second = await asyncio.gather(
        comms.ask("job-1", "Q-A", 4_000),
        comms.ask("job-1", "Q-B", 4_000),
    )

    refused = [
        reply for reply in (first, second) if reply.error and "already pending" in reply.error
    ]
    served = [reply for reply in (first, second) if reply.error is None]
    assert len(refused) == 1, "both asks passed the guard; one caller is orphaned"
    assert len(served) == 1 and served[0].timed_out is True
    assert comms._records["job-1"].ask is None, "the surviving ask leaked its future"
    assert not comms._records["job-1"].pending_asks


# --- peek: ranged, read-only observation of a child's transcript ------------


async def _until_peek(comms, job_id, want, timeout: float = 10.0):
    """Poll an async peek until the child's transcript satisfies ``want``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        window = await comms.peek(job_id, steps=50)
        if want(window):
            return window
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the child's transcript")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_peek_refuses_an_unknown_child():
    comms, _jobs, _child, _parent = wire()
    window = await comms.peek("nope")
    assert window.error is not None and "unknown subagent" in window.error


@pytest.mark.asyncio
async def test_peek_reports_a_child_that_never_started():
    comms, _jobs, _child, _parent = wire(attach=False)
    window = await comms.peek("job-1")
    assert window.error is not None and "no transcript" in window.error
    assert window.total == 0


@pytest.mark.asyncio
async def test_peek_with_an_end_past_the_transcript_still_shows_the_tail():
    """A public caller can pass end beyond the transcript; the window must
    clamp to the last steps rather than collapse to lo > hi (round 2)."""
    from local_operator.harness.comms import _resolve_peek_range

    lo, hi, error = _resolve_peek_range(10, start=None, end=100, steps=None)
    assert error is None
    assert (lo, hi) == (6, 10)
    # A non-positive end is a range error, not an empty window (round 3).
    for bad in (0, -3):
        _lo, _hi, err = _resolve_peek_range(10, start=None, end=bad, steps=None)
        assert err is not None and "end must be >= 1" in err


@pytest.mark.asyncio
async def test_peek_shows_a_running_childs_recent_steps(tmp_path, monkeypatch):
    """The end-to-end claim: a RUNNING child's transcript is readable, ranged,
    and stable — the parent sees what the child is doing without touching it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_parent(tmp_path, ScriptedProvider())
    await parent.async_init()
    job_id = parent._launch_subagent(label="parser", prompt="Rewrite the parser.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)
    # Let the child run a few tool steps so there is something to peek at.
    await _until_peek(comms, job_id, lambda w: w.total >= 4)

    window = await comms.peek(job_id)  # default: last few steps
    assert window.error is None
    assert window.status == "running"
    assert window.total >= 4
    assert window.steps, "a running child with a transcript must show steps"
    kinds = {step.kind for step in window.steps}
    assert "tool" in kinds, "the child's bash steps must be visible"

    # Ranges are stable 1-based positions: paging forward from the last seen
    # step yields exactly the steps after it.
    last = window.steps[-1].index
    ahead = await comms.peek(job_id, start=last + 1)
    assert ahead.error is not None or all(step.index > last for step in ahead.steps)

    # steps= is the "last N" shorthand and clamps to the transcript.
    three = await comms.peek(job_id, steps=3)
    assert len(three.steps) <= 3
    assert three.steps[-1].index == three.total

    # An out-of-range start is a legible error, not an empty dump.
    beyond = await comms.peek(job_id, start=window.total + 50)
    assert beyond.error is not None and "nothing at step" in beyond.error
    await parent.dispose()


@pytest.mark.asyncio
async def test_peek_through_the_hub_tool_is_ranged_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_parent(tmp_path, ScriptedProvider())
    await parent.async_init()
    job_id = parent._launch_subagent(label="parser", prompt="Rewrite the parser.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)
    await _until_peek(comms, job_id, lambda w: w.total >= 4)

    result = await execute_hub(
        "call-1",
        {"op": "peek", "to": [job_id], "steps": 2},
        None,
        None,
        ToolContext(cwd=str(tmp_path), subagent_comms=comms),
    )
    text = body(result)
    assert result.is_error is False
    assert "2 of" in text and "transcript step(s)" in text
    # A long tool result must be clipped, not dumped: the child's bash output
    # is the bulk of any transcript and the one thing peek must not re-send.
    assert len(text) < 4000

    # A nonsense range is refused with a legible error.
    bad = await execute_hub(
        "call-2",
        {"op": "peek", "to": [job_id], "range": "5-2"},
        None,
        None,
        ToolContext(cwd=str(tmp_path), subagent_comms=comms),
    )
    assert bad.is_error is True
    await parent.dispose()


def test_an_explicit_range_cannot_bypass_the_step_ceiling():
    """The cap is "whatever the caller asks for": range='1-1000' against a
    long transcript must not inject 1000 steps through the one op that
    exists to bound context (review round 1, major)."""
    from local_operator.harness.comms import PEEK_MAX_STEPS, _resolve_peek_range

    for total in (PEEK_MAX_STEPS, PEEK_MAX_STEPS * 4, 1000):
        lo, hi, error = _resolve_peek_range(total, start=1, end=total, steps=None)
        assert error is None
        assert (
            hi - lo + 1 <= PEEK_MAX_STEPS
        ), f"explicit range returned {hi - lo + 1} steps of a {total}-step transcript"
    # The clamp keeps the requested HEAD and pages forward from there.
    lo, hi, _error = _resolve_peek_range(200, start=100, end=200, steps=None)
    assert (lo, hi) == (100, 100 + PEEK_MAX_STEPS - 1)


def test_hub_peek_and_list_are_read_tier_while_control_stays_write():
    """Observing children must never prompt; controlling them still does."""
    from local_operator.tools.builtin import build_hub_tool

    comms, _jobs, _child, _parent = wire()

    class Ctx:
        subagent_comms = comms
        job_id = None

    tool = build_hub_tool(Ctx())  # type: ignore[arg-type]
    assert tool is not None
    assert tool.approval_tier == "write"
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"op": "peek"}) == "read"
    assert tool.call_approval_tier({"op": "list"}) == "read"
    assert tool.call_approval_tier({"op": "cancel", "to": ["job-1"]}) == "write"
    assert tool.call_approval_tier({"op": "resume", "message": "x"}) == "write"


def test_a_child_hub_reply_with_parent_shaped_args_still_reaches_the_parent():
    """Children mirror the parent tool shape they see (``op``/``to``); the
    child surface must drop those keys and deliver the message anyway.

    ``model_validate`` is the construction the child tool actually performs
    (``HubChildParams(**args)``), and it is what runs the ``mode="before"``
    validator that strips the parent-shaped keys. Passing them as keyword
    arguments to the constructor would be a type error the validator never
    sees, so the dict form is the one under test.
    """
    comms, _jobs, _child, parent = wire()

    from local_operator.tools.builtin import HubChildParams

    params = HubChildParams.model_validate(
        {"op": "send", "to": ["parent"], "message": "review posted"}
    )
    assert params.message == "review posted"

    # The dropped keys must not stop delivery: the reply lands as an aside on
    # the parent, which is what the child's ``hub`` call resolves to.
    outcome = comms.reply_to_parent("job-1", params.message)
    assert outcome == "delivered to the parent (it will read this at its next step)"
    [aside] = parent.asides
    message = aside()
    assert isinstance(message, CustomMessage)
    assert message.custom_type == HUB_MESSAGE_TYPE
    assert "review posted" in message.details["text"]

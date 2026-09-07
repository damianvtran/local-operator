"""Owned-session handle behaviours that have no terminal to fall back on:
full-auto approval, headless conversation naming, and the concurrent-approval
queue wired through the fold.

These are the phone-started-session equivalents of things the TUI's
OperatorApp does (adopt ``tool_approval_mode: auto`` at boot, run the naming
worker). The handle is exercised directly with a minimal fake session so the
tests stay off the real provider and event loop machinery.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from local_operator.harness.types import (
    AskOption,
    AskQuestion,
    NoticeEvent,
    SteeringDeliveredEvent,
)
from local_operator.session.runtime import owned as owned_mod
from local_operator.session.runtime.owned import OwnedSessionHandle


class FakeSession:
    """The slice of Session the OwnedSessionHandle touches in these tests."""

    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.model_label = "test/model"
        self.effective_model_label = "test/model"
        self.model = None
        self.conversation_name = ""
        self.is_streaming = False
        self._handlers: list[Any] = []
        self._admission_handlers: list[Any] = []
        self._steer_rejection_handlers: list[Any] = []
        self._admitted_ids: set[str] = set()
        self._named: list[tuple[str, bool]] = []
        self._complete_calls: list[tuple[str, str]] = []
        self.prompt_calls: list[str] = []
        self.steer_calls: list[str] = []
        #: Reasons `abort` was called with, so a stop test can assert the turn
        #: was stopped and not only the children.
        self.aborts: list[str] = []
        self.prompt_release = asyncio.Event()
        #: The MCP manager the `/mcp` handlers read. ``None`` matches a real
        #: session before its servers connect; the grant tests substitute a
        #: double. Declared here rather than attached per-test so the shape is
        #: part of the double's contract.
        self.mcp_manager: Any = None
        #: The session's event-emission seam. The runtime reports a settled
        #: MCP grant through it, since the grant outlives the request that
        #: started it. Tests replace it to capture what viewers would see.
        self._emit: Any = None
        # Tagged or a short untagged title both parse; the default stays
        # tagged so these tests stay independent of the untagged heuristics.
        self.title_reply = "<title>A Neat Title</title>"
        from local_operator.harness.jobs import AsyncJobManager

        self.jobs = AsyncJobManager()

    # -- naming seams ----------------------------------------------------------
    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        self.conversation_name = text
        self._named.append((text, user_set))
        return text

    async def complete_once(self, system: str, prompt: str) -> str:
        self._complete_calls.append((system, prompt))
        return self.title_reply

    # -- gate registration -----------------------------------------------------
    def set_approval_handler(self, handler) -> None:
        self._approval_handler = handler

    def set_ask_handler(self, handler) -> None:
        self._ask_handler = handler

    # -- subscribe/selectors the handle reads at construction ------------------
    def subscribe(self, handler):
        # Capture the handler so a test can drive the fold with real events,
        # the way the live session's event stream does.
        self._handlers.append(handler)

        def _unsub() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return _unsub

    def emit(self, event) -> None:
        for handler in list(getattr(self, "_handlers", [])):
            handler(event)

    def history(self):  # pragma: no cover - not exercised here
        return []

    def has_admitted_command(self, command_id: str) -> bool:
        return command_id in self._admitted_ids

    def subscribe_admitted_commands(self, handler):  # noqa: ANN001, ANN202
        self._admission_handlers.append(handler)

        def unsubscribe() -> None:
            self._admission_handlers.remove(handler)

        return unsubscribe

    def admit(self, command_id: str) -> None:
        self._admitted_ids.add(command_id)
        for handler in list(self._admission_handlers):
            handler(command_id)

    def subscribe_rejected_steering(self, handler):  # noqa: ANN001, ANN202
        self._steer_rejection_handlers.append(handler)

        def unsubscribe() -> None:
            self._steer_rejection_handlers.remove(handler)

        return unsubscribe

    def reject_steer(self, command_id: str, reason: str) -> None:
        for handler in list(self._steer_rejection_handlers):
            handler(command_id, reason)
        self.emit(
            NoticeEvent(
                text=(
                    f"steering command {command_id} was not saved: {reason}; "
                    "retry with the same command ID"
                )
            )
        )
        self.emit(SteeringDeliveredEvent(count=1))

    def running_subagents(self) -> int:
        return 0

    def abort(self, reason: str = "interrupted") -> None:
        """Part of SessionProtocol, and the control op's first act. Recorded
        rather than ignored so a test can prove the turn was stopped as well as
        the children."""
        self.aborts.append(reason)

    async def prompt(self, text: str, images=None) -> None:  # noqa: ANN001
        self.prompt_calls.append(text)
        self.is_streaming = True
        await self.prompt_release.wait()
        self.is_streaming = False

    def steer(self, text: str, images=None) -> None:  # noqa: ANN001
        self.steer_calls.append(text)

    async def dispose(self) -> None:
        self.prompt_release.set()

    @property
    def reasoning_effort(self):  # pragma: no cover
        return "auto"


def make_handle(auto_approve: bool = False) -> tuple[OwnedSessionHandle, FakeSession]:
    # The handle records whichever loop it is built on; inside an async test
    # that is the running loop, and the sync construction test never awaits so
    # a fresh loop is fine. get_event_loop_policy().get_event_loop() avoids the
    # "no current event loop" deprecation of the bare accessor.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    session = FakeSession()
    handle = OwnedSessionHandle(session, loop, cwd="/tmp", auto_approve=auto_approve)
    return handle, session


def test_default_conversation_name_is_empty_not_a_placeholder() -> None:
    """A fresh mobile session shows nothing until named, so the phone's own
    fallback ("untitled" / cwd) applies — never a frozen "mobile session"."""
    handle, _ = make_handle()
    assert handle.session_projection_seed.conversation_name == ""


@pytest.mark.asyncio
async def test_full_auto_approves_inline_without_a_card() -> None:
    """With the owner's saved default at full-auto, the gate answers True
    inline and never parks a pending card — matching the TUI's auto mode."""
    handle, _ = make_handle(auto_approve=True)
    gate = handle._approval_gate
    approved = await gate("bash", "rm -rf build/")
    assert approved is True
    # No card was ever queued.
    assert handle._fold.projection.pending is None
    assert handle._fold.projection.pending_count == 0


@pytest.mark.asyncio
async def test_ask_mode_parks_a_card_then_resolves() -> None:
    """Without full-auto, the gate queues a card and blocks until answered —
    and a second concurrent gate queues behind it rather than overwriting."""
    handle, _ = make_handle(auto_approve=False)
    gate = handle._approval_gate

    first = asyncio.ensure_future(gate("bash", "one"))
    second = asyncio.ensure_future(gate("write", "two"))
    await asyncio.sleep(0)  # let both gates enqueue

    assert handle._fold.projection.pending_count == 2
    front = handle._fold.projection.pending
    assert front is not None and front.title == "bash"

    # Answer the front; the second card surfaces, count drops.
    await handle.approval_answer(front.request_id, True, False)
    assert await first is True
    await asyncio.sleep(0)
    assert handle._fold.projection.pending_count == 1
    nxt = handle._fold.projection.pending
    assert nxt is not None and nxt.title == "write"

    await handle.approval_answer(nxt.request_id, False, False)
    assert await second is False
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is None


@pytest.mark.asyncio
async def test_same_id_concurrent_steers_are_admitted_once() -> None:
    handle, session = make_handle()
    command_id = "same-steer"

    receipts = await asyncio.gather(
        handle.steer("correction", command_id=command_id),
        handle.steer("correction", command_id=command_id),
    )

    assert receipts == ["steering queued", "already admitted"]
    assert session.steer_calls == ["correction"]
    assert [row.text for row in handle._fold.projection.transcript] == ["correction"]


@pytest.mark.asyncio
async def test_async_steer_rejection_releases_owned_slot_and_same_id() -> None:
    handle, session = make_handle()
    notified = 0

    def notify() -> None:
        nonlocal notified
        notified += 1

    handle.subscribe(notify)
    assert await handle.steer("first", command_id="retry-id") == "steering queued"
    assert handle._command_reservations._pending_steers == 1
    assert handle.session_projection_seed.queued_count == 1

    session.reject_steer("retry-id", "disk full")

    assert handle._command_reservations._pending_steers == 0
    assert "retry-id" not in handle._command_reservations._commands
    assert handle.session_projection_seed.queued_count == 0
    assert handle.session_projection_seed.transcript[-1].text == (
        "steering command retry-id was not saved: disk full; retry with the same command ID"
    )
    assert notified >= 2
    assert await handle.steer("retry", command_id="retry-id") == "steering queued"
    assert session.steer_calls == ["first", "retry"]


@pytest.mark.asyncio
async def test_steer_ack_loss_retry_remains_admitted() -> None:
    handle, session = make_handle()

    assert await handle.steer("once", command_id="lost-ack") == "steering queued"
    assert await handle.steer("once", command_id="lost-ack") == "already admitted"
    assert session.steer_calls == ["once"]


@pytest.mark.asyncio
async def test_stalled_owned_steers_apply_backpressure_and_drain_frees_capacity() -> None:
    handle, session = make_handle()

    for index in range(32):
        assert await handle.steer(f"steer {index}", command_id=f"id-{index}") == "steering queued"
    assert len(session.steer_calls) == 32
    assert await handle.steer("duplicate", command_id="id-0") == "already admitted"
    with pytest.raises(RuntimeError, match=r"steering queue is full \(32\)"):
        await handle.steer("overflow", command_id="overflow")
    assert len(session.steer_calls) == 32

    session.admit("id-0")
    assert await handle.steer("replacement", command_id="replacement") == "steering queued"
    assert await handle.steer("old retry", command_id="id-0") == "already admitted"


@pytest.mark.asyncio
async def test_terminal_steer_rejection_releases_identity() -> None:
    handle, session = make_handle()
    original = session.steer
    attempts = 0

    def reject_once(text, images=None):  # noqa: ANN001, ANN202
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("not accepted")
        original(text, images)

    session.steer = reject_once  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="not accepted"):
        await handle.steer("retry", command_id="retry-id")
    assert await handle.steer("retry", command_id="retry-id") == "steering queued"
    assert session.steer_calls == ["retry"]


@pytest.mark.asyncio
async def test_prompt_streaming_rejection_transfers_identity_to_steer() -> None:
    handle, session = make_handle()

    async def reject_prompt(  # noqa: ANN202
        text, images=None, *, message_id=None, admitted=None  # noqa: ANN001
    ):
        raise RuntimeError("session is already streaming; use steer() to inject mid-turn")

    session.prompt = reject_prompt  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="already streaming"):
        await handle.prompt("raced", command_id="fallback-id")
    assert await handle.steer("raced", command_id="fallback-id") == "steering queued"
    assert await handle.steer("raced", command_id="fallback-id") == "already admitted"
    assert session.steer_calls == ["raced"]


@pytest.mark.asyncio
async def test_distinct_concurrent_steers_keep_fifo_order() -> None:
    handle, session = make_handle()

    receipts = await asyncio.gather(
        *(
            handle.steer(text, command_id=f"id-{index}")
            for index, text in enumerate(["a", "b", "c"])
        )
    )

    assert receipts == ["steering queued"] * 3
    assert session.steer_calls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_concurrent_ordinary_prompts_are_admitted_fifo() -> None:
    handle, session = make_handle()
    first, second, third = await asyncio.gather(
        handle.prompt("mobile"),
        handle.prompt("attach one"),
        handle.prompt("attach two"),
    )
    assert first == "prompt admitted"
    assert second == "prompt queued (2)"
    assert third == "prompt queued (3)"
    await asyncio.sleep(0)
    assert session.prompt_calls == ["mobile"]
    assert handle.is_busy() is True

    session.prompt_release.set()
    deadline = asyncio.get_running_loop().time() + 5
    while len(session.prompt_calls) < 3:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.01)
    assert session.prompt_calls == ["mobile", "attach one", "attach two"]
    assert len(set(session.prompt_calls)) == 3


@pytest.mark.asyncio
async def test_failed_admitted_prompt_is_visible_and_later_fifo_progresses() -> None:
    handle, session = make_handle()
    session.prompt_release.set()
    calls: list[str] = []

    async def prompt(text: str, images=None) -> None:  # noqa: ANN001
        calls.append(text)
        if text == "first":
            raise ValueError("provider exploded")

    session.prompt = prompt
    assert await handle.prompt("first") == "prompt admitted"
    assert await handle.prompt("second") == "prompt queued (2)"
    drain = handle._prompt_drain_task
    assert drain is not None
    await drain

    assert calls == ["first", "second"]
    assert not handle._prompt_queue
    assert handle.is_busy() is False
    notices = [
        entry.text for entry in handle.session_projection_seed.transcript if entry.kind == "notice"
    ]
    assert any("provider exploded" in notice for notice in notices)
    assert drain.exception() is None


@pytest.mark.asyncio
async def test_dispose_rejects_queued_admissions_without_unhandled_task_error() -> None:
    handle, session = make_handle()
    assert await handle.prompt("running") == "prompt admitted"
    assert await handle.prompt("queued") == "prompt queued (2)"
    await asyncio.sleep(0)

    await handle.dispose()

    assert not handle._prompt_queue
    assert handle.is_busy() is False
    notices = [
        entry.text for entry in handle.session_projection_seed.transcript if entry.kind == "notice"
    ]
    assert sum("session closed" in notice for notice in notices) == 2
    drain = handle._prompt_drain_task
    assert drain is not None and drain.cancelled()


@pytest.mark.asyncio
async def test_queue_overflow_rejects_before_admission(monkeypatch) -> None:
    monkeypatch.setattr(owned_mod, "MAX_QUEUED_PROMPTS", 1)
    handle, _ = make_handle()
    assert await handle.prompt("first") == "prompt admitted"
    with pytest.raises(RuntimeError, match="prompt queue is full"):
        await handle.prompt("overflow")
    assert [text for text, _ in handle._prompt_queue] == ["first"]
    drain = handle._prompt_drain_task
    assert drain is not None
    drain.cancel()
    await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_gate_answers_have_one_authoritative_winner() -> None:
    handle, _ = make_handle()
    gate = asyncio.create_task(handle._approval_gate("bash", "Allow?"))
    await asyncio.sleep(0)
    request_id = next(iter(handle._pending_futures))
    results = await asyncio.gather(
        handle.approval_answer(request_id, True, False),
        handle.approval_answer(request_id, False, False),
        return_exceptions=True,
    )
    assert gate.done()
    assert await gate in (True, False)
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert sum(isinstance(result, str) for result in results) == 1
    assert request_id not in handle._pending_futures


@pytest.mark.asyncio
async def test_concurrent_explicit_steers_preserve_dispatch_order() -> None:
    handle, session = make_handle()
    receipts = await asyncio.gather(
        handle.steer("mobile steer"),
        handle.steer("attach steer one"),
        handle.steer("attach steer two"),
    )
    assert receipts == ["steering queued"] * 3
    assert session.steer_calls == ["mobile steer", "attach steer one", "attach steer two"]


@pytest.mark.asyncio
async def test_pending_gate_is_busy_until_ordinary_timeout(monkeypatch) -> None:
    """The child drain cannot deny WAITING_INPUT ahead of its 30s policy."""
    monkeypatch.setattr(owned_mod, "PENDING_REQUEST_TIMEOUT_S", 0.05)
    handle, _ = make_handle(auto_approve=False)
    waiting = asyncio.ensure_future(handle._approval_gate("bash", "one"))
    await asyncio.sleep(0)
    assert handle.is_busy() is True
    assert await waiting is False
    assert handle.is_busy() is False
    assert owned_mod.PENDING_REQUEST_TIMEOUT_S == 0.05


@pytest.mark.asyncio
async def test_real_background_bash_job_is_busy_until_done(tmp_path) -> None:
    """The reaper reads the real bash job Session.dispose would terminate."""
    from local_operator.harness.types import ToolContext
    from local_operator.tools import builtin

    handle, session = make_handle()
    context = ToolContext(cwd=str(tmp_path), session_id="owned-bash", jobs=session.jobs)
    tool = builtin.build_bash_tool()
    result = await tool.execute(  # type: ignore[operator]
        "call",
        {"command": "sleep 0.4; echo settled", "background": True, "timeout": 5},
        None,
        None,
        context,
    )
    job_id = str((result.details or {})["job_id"])
    job = session.jobs.get(job_id)
    assert job is not None and job.type == "bash"
    assert handle.is_busy() is True
    deadline = asyncio.get_running_loop().time() + 5
    while job.status == "running":
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.01)
    assert job.status == "completed"
    assert handle.is_busy() is False
    await session.jobs.dispose()


@pytest.mark.asyncio
async def test_detached_background_work_is_busy_until_done() -> None:
    handle, _ = make_handle()
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    handle._background_tasks.add(future)
    assert handle.is_busy() is True
    future.set_result(None)
    assert handle.is_busy() is False


@pytest.mark.asyncio
async def test_ask_gate_projects_serializable_options_with_descriptions() -> None:
    """An owned ask WITH options must project a JSON-serializable card that
    carries each option's consequence line (U3).

    Regression origin: the gate once pushed raw AskOption pydantic models into
    PendingRequest, whose ``to_json`` is ``asdict`` and leaves those models as
    objects ``json.dumps`` cannot encode — crashing the projection push. The
    wire now carries {label, description} dicts (still JSON-serializable), and
    the phone renders both. The ``json.dumps`` below is what raised on the old
    object-valued shape.
    """
    handle, _ = make_handle(auto_approve=False)
    gate = handle._ask_gate

    question = AskQuestion(
        id="stale",
        question="What should happen to the stale rows?",
        options=[
            AskOption(label="Drop them", description="nothing reads the column"),
            AskOption(label="Backfill", description="slower, keeps history"),
        ],
    )
    asked = asyncio.ensure_future(gate([question]))
    await asyncio.sleep(0)  # let the gate enqueue its card

    pending = handle._fold.projection.pending
    assert pending is not None
    assert pending.kind == "ask"
    assert [o.label for o in pending.options] == ["Drop them", "Backfill"]
    assert [o.description for o in pending.options] == [
        "nothing reads the column",
        "slower, keeps history",
    ]
    assert pending.secret is False
    assert (pending.question_index, pending.question_total) == (0, 1)

    # The whole point: the projection round-trips over the wire. This is the
    # line that raised on the old object-valued shape.
    wire = json.dumps(handle._fold.projection.to_json())
    assert '"Drop them"' in wire and '"nothing reads the column"' in wire

    # Answer it back with a label, exactly as the phone does, so the parked
    # gate resolves under the question's id and the test leaves nothing hanging.
    await handle.ask_answer(pending.request_id, "Backfill")
    assert await asyncio.wait_for(asked, 1) == {"stale": ["Backfill"]}


@pytest.mark.asyncio
async def test_ask_gate_projects_secret_flag_without_the_value() -> None:
    """D1/U2: a secret ask projects secret=True and no options (paste field),
    and the pasted value never rides the projection or the wire."""
    handle, _ = make_handle(auto_approve=False)
    gate = handle._ask_gate

    question = AskQuestion(id="OPENAI_API_KEY", question="Paste your key", secret=True)
    asked = asyncio.ensure_future(gate([question]))
    await asyncio.sleep(0)

    pending = handle._fold.projection.pending
    assert pending is not None
    assert pending.secret is True
    assert pending.options == []

    await handle.ask_answer(pending.request_id, "sk-topsecret")
    # The value never appeared on the wire while the card was live or after.
    assert "sk-topsecret" not in json.dumps(handle._fold.projection.to_json())
    assert await asyncio.wait_for(asked, 1) == {"OPENAI_API_KEY": ["sk-topsecret"]}


@pytest.mark.asyncio
async def test_ask_gate_asks_multiple_questions_one_at_a_time() -> None:
    """U1: an owned multi-question ask projects and resolves question by
    question — answering Q1 advances to Q2 rather than settling the whole set,
    and the gate returns BOTH answers only after the last is answered."""
    handle, _ = make_handle(auto_approve=False)
    gate = handle._ask_gate

    q1 = AskQuestion(
        id="env",
        question="Which environment?",
        options=[AskOption(label="prod"), AskOption(label="staging")],
    )
    q2 = AskQuestion(
        id="confirm",
        question="Confirm?",
        options=[AskOption(label="yes"), AskOption(label="no")],
    )
    asked = asyncio.ensure_future(gate([q1, q2]))
    await asyncio.sleep(0)

    first = handle._fold.projection.pending
    assert first is not None
    assert first.title == "Which environment?"
    assert (first.question_index, first.question_total) == (0, 2)

    await handle.ask_answer(first.request_id, "prod")
    # The answer resolves via call_soon_threadsafe; let the gate resume, pop Q1,
    # and push Q2 across a few loop turns before inspecting the new front card.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not asked.done(), "the gate resolved after only Q1 (U1 truncation)"

    second = handle._fold.projection.pending
    assert second is not None
    assert second.title == "Confirm?"
    assert (second.question_index, second.question_total) == (1, 2)
    # A distinct request id per question: Q1's stale request must no longer
    # resolve anything.
    assert second.request_id != first.request_id

    await handle.ask_answer(second.request_id, "yes")
    assert await asyncio.wait_for(asked, 1) == {"env": ["prod"], "confirm": ["yes"]}
    assert handle._fold.projection.pending is None


@pytest.mark.asyncio
async def test_naming_worker_stores_a_title_once() -> None:
    """The first substantive prompt names an unnamed session; a low-signal
    opener does not consume the one attempt."""
    handle, session = make_handle()

    # Low-signal opener: skipped, latch not spent.
    handle._maybe_name_conversation("hi")
    assert handle._name_requested is False

    handle._maybe_name_conversation("please refactor the billing importer")
    assert handle._name_requested is True
    # Let the background naming task run.
    for _ in range(5):
        await asyncio.sleep(0)
    assert session.conversation_name == "A Neat Title"
    assert session._named == [("A Neat Title", False)]

    # A second prompt does not re-name.
    handle._maybe_name_conversation("and now the invoices too")
    for _ in range(5):
        await asyncio.sleep(0)
    assert session._named == [("A Neat Title", False)]


@pytest.mark.asyncio
async def test_first_prompt_wears_a_provisional_title_before_the_model_answers() -> None:
    """The phone list must not stay "untitled" for the whole first turn.

    Isolated naming is a round trip (and often a 429 on a dead primary). The
    opener excerpt is already in hand, so the projection wears it the same
    frame the prompt is accepted — matching the TUI band.
    """
    handle, session = make_handle()
    session.title_reply = ""  # naming will fail; the stand-in must still land

    handle._maybe_name_conversation("review what regressed in mobile titles")
    assert handle._fold.projection.conversation_name == "Review what regressed in mobile titles"
    assert session.conversation_name == ""

    for _ in range(5):
        await asyncio.sleep(0)
    # Failure released the latch and stashed the opener for a route-edge retry.
    assert handle._name_requested is False
    assert handle._pending_name_text == "review what regressed in mobile titles"


@pytest.mark.asyncio
async def test_a_failed_name_retries_once_a_fallback_pins() -> None:
    """Quota-exhausted naming is isolated; the turn's fallback must re-fire it."""
    from local_operator.harness.types import ModelChangeEvent

    handle, session = make_handle()
    session.title_reply = ""
    handle.subscribe(lambda: None)
    handle._maybe_name_conversation("review what regressed in mobile titles")
    for _ in range(5):
        await asyncio.sleep(0)
    assert session.conversation_name == ""
    assert handle._pending_name_text

    session.title_reply = "<title>Mobile title sync</title>"
    # The real session updates effective_model BEFORE emitting; _refresh_state
    # then re-reads that label. A fake that only emits would have the fold
    # paint the fallback and the refresh clobber it back to the selection.
    session.effective_model_label = "xai/grok-4.6"
    session.emit(
        ModelChangeEvent(
            provider="xai",
            model_id="grok-4.6",
            is_fallback=True,
            reason="quota exhausted",
        )
    )
    for _ in range(8):
        await asyncio.sleep(0)
    assert session.conversation_name == "Mobile title sync"
    assert handle._pending_name_text == ""
    assert handle._fold.projection.model_label == "xai/grok-4.6"


@pytest.mark.asyncio
async def test_agent_end_clears_streaming_despite_stale_is_streaming() -> None:
    """Regression: the phone stayed pinned to "in progress" after a turn ended.

    The session emits ``AgentEndEvent`` while its ``is_streaming`` flag is
    STILL True — the flag clears only in the turn's ``finally`` block, after
    the event has been emitted and folded. The per-event ``_refresh_state``
    used to re-read that stale True and overwrite the fold's correct
    ``streaming=False``; because the end event is the turn's last event, no
    later push ever corrected it and the session list shimmered forever.

    This drives the real handler wiring (subscribe → emit) and asserts the
    projection settles to ``streaming=False`` even though ``is_streaming`` is
    left True, exactly as the live session leaves it at the emit point.
    """
    from local_operator.harness.types import AgentEndEvent, AgentStartEvent

    handle, session = make_handle()
    handle.subscribe(lambda: None)

    # Turn starts: the session marks itself streaming and emits the start.
    session.is_streaming = True
    session.emit(AgentStartEvent(generation=1))
    assert handle._fold.projection.streaming is True

    # Turn ends: the session emits AgentEndEvent BEFORE clearing the flag,
    # reproducing the real ordering (is_streaming still True at emit time).
    session.emit(AgentEndEvent(aborted=False, generation=1))

    assert handle._fold.projection.streaming is False, (
        "projection stuck streaming=True after AgentEndEvent — the per-event "
        "refresh clobbered the fold with the not-yet-cleared is_streaming flag"
    )
    assert handle._fold.projection.stop_reason == "completed"
    assert handle._fold.projection.activity == ""


@pytest.mark.asyncio
async def test_command_boundary_reconcile_cannot_restick_after_abort() -> None:
    """F1 regression: on the abort/error path the session emits AgentEndEvent
    INLINE while its ``is_streaming`` flag is still True (it clears several
    awaits later, in the turn's ``finally``). A mobile command landing in that
    window runs ``refresh`` → ``_reconcile_streaming`` with the stale True. The
    fold must ignore it because it has already folded a terminal event, so the
    projection stays ``streaming=False`` instead of re-sticking to "in
    progress" with no later event to correct it.
    """
    from local_operator.harness.types import AgentEndEvent, AgentStartEvent

    handle, session = make_handle()
    handle.subscribe(lambda: None)

    session.is_streaming = True
    session.emit(AgentStartEvent(generation=1))
    # Aborted turn: end event folded, but the session flag has NOT cleared yet.
    session.emit(AgentEndEvent(aborted=True, generation=1))
    assert handle._fold.projection.streaming is False

    # A command lands before the finally clears is_streaming: reconcile sees
    # the stale True and must NOT raise streaming back up.
    await handle.refresh()
    assert handle._fold.projection.streaming is False, (
        "command-boundary reconcile re-stuck streaming=True from the stale "
        "is_streaming flag on the abort path"
    )

    # A genuine next turn still reconciles up normally (latch cleared on start).
    session.emit(AgentStartEvent(generation=2))
    assert handle._fold.projection.streaming is True


@pytest.mark.asyncio
async def test_attach_seeds_streaming_from_flag_for_mid_turn_subscriber() -> None:
    """A phone that subscribes mid-turn never saw the AgentStartEvent, so the
    fold alone would open on a stale ``streaming=False``. Attach seeds the live
    flag once from the session so the working line paints immediately."""
    handle, session = make_handle()
    session.is_streaming = True
    handle.subscribe(lambda: None)
    assert handle._fold.projection.streaming is True


# --- next_wake_due_at: the reaper's warmth signal ----------------------------


def test_next_wake_due_at_reads_the_live_scheduler() -> None:
    from types import SimpleNamespace

    from local_operator.harness.wake import WakeSchedule

    loop = asyncio.new_event_loop()
    try:
        scheduler = SimpleNamespace(
            disposed=False,
            schedules=(
                WakeSchedule(id="a", message="x", next_due_at=5_000, created_at=0),
                WakeSchedule(id="b", message="y", next_due_at=2_000, created_at=0),
            ),
        )
        session = SimpleNamespace(
            session_id="s", wake_scheduler=scheduler, subscribe=lambda h: None
        )
        handle = OwnedSessionHandle.__new__(OwnedSessionHandle)
        handle._session = session  # type: ignore[attr-defined]
        handle._loop = loop  # type: ignore[attr-defined]
        assert handle.next_wake_due_at() == 2_000
        scheduler.schedules = ()
        assert handle.next_wake_due_at() is None
        scheduler.schedules = (WakeSchedule(id="a", message="x", next_due_at=5_000, created_at=0),)
        scheduler.disposed = True
        assert handle.next_wake_due_at() is None, "a disposed scheduler can no longer fire"
        handle._session = SimpleNamespace(session_id="s")  # type: ignore[attr-defined]
        assert handle.next_wake_due_at() is None
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_a_parked_gate_spawns_no_desktop_notifier_under_the_suite_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suite must never put a toast on the developer's real desktop.

    A runtime with no attached client announces a parked gate through
    ``detached_notify``, which on darwin spawns a real ``osascript display
    notification``. Six tests in this file drive real gates with zero attached
    clients, so before the suite-wide gate they delivered 100 genuine
    notifications to Notification Centre — titled "lop needs you", bodies
    taken verbatim from these fixtures.

    Nothing about a green suite reveals that: the spawn is fire-and-forget and
    every failure is swallowed by ``detached_notify``'s contract. This test is
    the tripwire — it asserts on the SPAWN, which is the only observable the
    leak has, so a future change that reintroduces an ungated OS-facing path
    fails here instead of on the operator's screen.

    The gate itself lives in ``tests/conftest.py::isolate_environment``
    (autouse), which is what makes the whole suite silent; this asserts that
    gate is actually in force on the production announce path.
    """
    spawned: list[list[str]] = []

    def _record(argv: list[str], *args: Any, **kwargs: Any) -> bool:
        spawned.append(argv)
        return True

    monkeypatch.setattr(owned_mod, "spawn_detached", _record, raising=False)
    import local_operator.tui.notify as notify_mod

    monkeypatch.setattr(notify_mod, "spawn_detached", _record, raising=False)
    # Also patch the helper that runs AFTER a platform notifier is found. On a
    # runner with neither `osascript` nor `notify-send` (CI's Linux images) a
    # spawn-level assertion passes for the wrong reason — the binary was
    # missing, not the gate. This one cannot: it fires whenever the gate lets
    # execution reach the spawn at all.
    monkeypatch.setattr(
        notify_mod, "_spawn_detached_ok", lambda argv: bool(spawned.append(argv)) or True
    )
    # And prove the gate the fixture sets is the reason, rather than an
    # unrelated early return: with it removed, this same path DOES spawn.
    assert notify_mod.notifications_enabled() is False

    handle, session = make_handle()
    # No attached clients: this is exactly the condition that routes the
    # announcement out of band to the OS.
    assert handle._attached_clients() == 0
    handle._announce_pending("approval", "bash", "rm -rf build/")
    await asyncio.sleep(0)

    assert spawned == [], f"the suite spawned an OS notifier: {spawned}"


@pytest.mark.parametrize(
    ("watching", "expect_toast"),
    [
        (frozenset({"attach"}), False),
        (frozenset({"viewer"}), False),
        (frozenset({"attach", "viewer"}), False),
        (frozenset(), True),
    ],
)
@pytest.mark.asyncio
async def test_a_pending_announcement_routes_to_whoever_is_watching(
    monkeypatch: pytest.MonkeyPatch,
    watching: frozenset[str],
    expect_toast: bool,
) -> None:
    """A notification goes to the surface that is watching; the OS is the
    fallback for nobody, not the default.

    The old test was ``attached_clients() > 0``, which counts terminals only —
    so a user whose PHONE was watching got a desktop toast for a card already
    on their phone, on the one surface they were not looking at. Both watching
    surfaces already deliver this card (the terminal paints it in-band, the
    mobile relay carries it in the projection push ``_notify`` has already
    made), which is why routing is a predicate and not a second transport.

    NOTE the kinds here: ``viewer`` is a phone with the session actually
    OPEN, never the relay's mere presence. This test injects the set as a
    premise, which is deliberately not enough on its own — see
    ``test_watching_surfaces_is_derived_from_real_connections`` in
    ``test_server.py``, which derives it from a real dial and is what catches
    the class of defect this parametrisation cannot (round 3, B1).
    """
    import local_operator.tui.notify as notify_mod

    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    spawned: list[list[str]] = []
    # Patch at `detached_notify`, not at the spawn helper: the helper is only
    # reached once a platform notifier has been FOUND, and CI's Linux runners
    # have no `notify-send` (nor `osascript`), so a spawn-level probe measures
    # the runner's installed binaries instead of this module's routing
    # decision — which is the only thing under test here.
    monkeypatch.setattr(
        notify_mod,
        "detached_notify",
        lambda title, body, **kwargs: bool(spawned.append([title, body])) or True,
    )

    handle, _session = make_handle()

    class _Registrant:
        record = type("R", (), {"session_id": "route0000001"})()

        def watching_surfaces(self) -> frozenset[str]:
            return watching

        def set_record_pending(self, kind: str | None) -> None:
            return None

    handle._registrant = _Registrant()
    handle._announce_pending("approval", "bash", "rm -rf build/")

    assert bool(spawned) is expect_toast


@pytest.mark.asyncio
async def test_background_desktop_owns_notification_without_becoming_interactive(
    monkeypatch,
) -> None:
    import local_operator.tui.notify as notify_mod

    deliveries = []
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    monkeypatch.setattr(
        notify_mod, "detached_notify", lambda *args, **kwargs: deliveries.append(args)
    )
    handle, _session = make_handle()

    class DesktopRegistrant:
        active = True
        record = type("R", (), {"session_id": "desktop00001"})()

        def watching_surfaces(self):
            return frozenset()

        def notification_surfaces(self):
            return frozenset({"desktop"}) if self.active else frozenset()

        def set_record_pending(self, _kind):
            return None

    registrant = DesktopRegistrant()
    handle._registrant = registrant
    handle._announce_pending("approval", "bash", "build the project")
    assert not deliveries
    assert not handle._watching_surfaces()
    registrant.active = False
    handle._announce_pending("approval", "bash", "build the project")
    assert len(deliveries) == 1


@pytest.mark.asyncio
async def test_an_old_registrant_without_surface_kinds_keeps_the_previous_behaviour() -> None:
    """A runtime published by an older release cannot answer by kind.

    It still knows the attach COUNT, and treating "a terminal is attached" as
    "something is watching" reproduces the previous behaviour exactly rather
    than inventing a toast that release never sent.
    """
    handle, _session = make_handle()

    class _OldRegistrant:
        def attach_clients(self) -> int:
            return 1

    handle._registrant = _OldRegistrant()
    assert handle._watching_surfaces() == frozenset({"attach"})

    class _OldIdle:
        def attach_clients(self) -> int:
            return 0

    handle._registrant = _OldIdle()
    assert handle._watching_surfaces() == frozenset()


@pytest.mark.parametrize(
    ("tool", "description", "expected"),
    [
        # `describe_approval` already leads with the action word, and the
        # title IS the tool name, so prefixing rendered every approval toast
        # as "write: write: /path" — on the release's headline surface, every
        # time (round 4, Q3).
        ("write", "write: /tmp/notes.txt", "write: /tmp/notes.txt"),
        ("bash", "bash: rm -rf build/", "bash: rm -rf build/"),
        # A tool whose description does NOT name itself still gets the prefix.
        ("browser", "https://example.com", "browser: https://example.com"),
        # No description: the tool name alone says less than the shared
        # vocabulary, so BODIES answers instead of a bare "write".
        ("write", "", "Waiting for approval"),
    ],
)
def test_the_toast_body_never_repeats_the_tool_name(
    tool: str, description: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a user reads on the banner when nothing is attached."""
    import asyncio
    from types import SimpleNamespace

    from local_operator.session.runtime.owned import OwnedSessionHandle

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "local_operator.tui.notify.detached_notify",
        lambda title, body, **kwargs: sent.append((title, body)) or True,
    )
    monkeypatch.setattr("local_operator.tui.notify.notifications_enabled", lambda *a, **k: True)

    handle = OwnedSessionHandle.__new__(OwnedSessionHandle)
    handle._session = SimpleNamespace(conversation_name="a session")  # type: ignore[attr-defined]
    handle._registrant = None  # type: ignore[attr-defined]
    handle._parked_announcement = None  # type: ignore[attr-defined]
    handle._loop = asyncio.new_event_loop()  # type: ignore[attr-defined]
    handle._session_id_for_resume = lambda: "abc123def456"  # type: ignore[attr-defined]

    try:
        handle._announce_pending("approval", tool, description)
    finally:
        handle._loop.close()  # type: ignore[attr-defined]

    assert sent, "no notification was produced"
    assert sent[0][1] == expected


@pytest.mark.asyncio
async def test_a_compaction_that_refuses_corrects_its_own_receipt(tmp_path) -> None:
    """`/compact` answers optimistically, so a refusal MUST be reported.

    A pass that runs narrates itself through the canonical compaction events;
    a refusal emits nothing at all, which is what made it invisible on the
    routed path — the runtime replied "compacting context…" and then discarded
    the outcome, so the user was told a pass had started and nothing ever
    contradicted it (round 5, U17).

    Driven against a real empty session, whose genuine answer is
    `nothing_to_compact`, rather than a stubbed outcome: the copy the user
    reads comes from the session and a fake would not prove it arrives.
    """
    import json
    from pathlib import Path

    from local_operator.compaction.marker import COMPACTION_REFUSED_TYPE
    from local_operator.providers.clients import MockClient
    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.runtime.owned import OwnedSessionHandle
    from tests.e2e.harness import build_session

    session = build_session(tmp_path, MockClient().stream)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle._slash_result("compact", "", SlashResult)
        assert result.text == "compacting context…"

        # The reporting task is fire-and-forget by design (a long pass cannot
        # be awaited inside a request/response op), so settle it explicitly
        # rather than sleeping — a sleep here measures a race, not the answer.
        for task in list(handle._background_tasks):
            await asyncio.shield(task)

        rows = [
            json.loads(line)["payload"]
            for line in Path(session.transcript.path).read_text().splitlines()
            if line.strip()
            and json.loads(line).get("payload", {}).get("custom_type") == COMPACTION_REFUSED_TYPE
        ]
        assert len(rows) == 1, "the refusal never reached the transcript"
        detail = (rows[0].get("details") or {}).get("detail") or ""
        assert "nothing to compact" in detail, detail
    finally:
        await session.dispose()


# --- /mcp grant verbs on a DETACHED runtime ----------------------------------
#
# The regression these cover: a detached runtime refused every grant verb with
# "run it from a terminal on that machine" while the user was sitting at that
# machine. The control socket binds 127.0.0.1 only, so a client that reached
# the runtime is on its host by construction; the refusal fired on the one case
# it was meant to protect and left `/mcp reauth` with no working path at all
# once a session detached — exactly when an expired credential needs it.


class _GrantCfg:
    """An http server with no declared auth block: the shape that can OAuth."""

    auth = None
    url = "https://mcp.example.com/mcp"


@pytest.fixture
def fake_mcp_logout(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Never let a unit test delete a row from the developer's real auth.db.

    ``reauth`` forgets the stored grant before reconnecting, and the helper it
    uses writes the shared credential store. Patched at the definition site so
    the runtime's late import picks it up.
    """
    removed: list[str] = []

    def _fake(name: str, cwd: str) -> str | None:
        removed.append(name)
        return None

    monkeypatch.setattr("local_operator.mcp.auth.mcp_logout_server", _fake)
    return removed


class _GrantManager:
    def __init__(self, *, supports: bool = True) -> None:
        self._supports = supports
        self.connected: list[str] = []
        self.disconnected: list[str] = []

    def get_server_config(self, name: str):  # noqa: ANN202
        return _GrantCfg()

    async def server_supports_oauth_login(self, cfg) -> bool:  # noqa: ANN001
        return self._supports

    async def disconnect_server(self, name: str) -> None:
        self.disconnected.append(name)

    async def connect_configured_server(
        self, name: str, *, timeout_ms=None
    ):  # noqa: ANN001, ANN202
        self.connected.append(name)
        return type("_Conn", (), {"tools": [1, 2]})()


@pytest.mark.asyncio
async def test_routed_mcp_reauth_runs_instead_of_refusing_the_local_user(
    fake_mcp_logout: list[str],
) -> None:
    """The bug report: /mcp reauth on a detached session must actually reauth."""
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()
    session.mcp_manager = _GrantManager()

    result = await handle._slash_result("mcp", "reauth notion", SlashResult)

    assert "run it from a terminal on that machine" not in result.text
    assert "authorizing MCP server 'notion'" in result.text
    assert result.style == "info"

    for task in list(handle._mcp_grant_tasks):
        await asyncio.gather(task, return_exceptions=True)
    # The stored grant is forgotten BEFORE the reconnect, or the manager's
    # auto-reconnect re-authenticates the session the user just reset.
    assert fake_mcp_logout == ["notion"]
    assert session.mcp_manager.disconnected == ["notion"]
    assert session.mcp_manager.connected == ["notion"]


@pytest.mark.asyncio
async def test_a_relayed_remote_client_still_gets_the_locality_refusal(
    fake_mcp_logout: list[str],
) -> None:
    """The refusal is kept for the topology it actually describes.

    A phone's command reaching the runtime through a relay must NOT open a
    browser on the host: the tab would be in front of nobody and the credential
    would land in a store the phone's owner cannot use.
    """
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()
    session.mcp_manager = _GrantManager()

    result = await handle._slash_result("mcp", "reauth notion", SlashResult, "remote")

    assert "run it from a terminal on that machine" in result.text
    assert result.style == "warning"
    assert session.mcp_manager.connected == []
    assert session.mcp_manager.disconnected == []
    assert fake_mcp_logout == [], "a refused grant must not touch the credential"


@pytest.mark.asyncio
async def test_a_grant_verb_without_a_server_name_is_refused_by_arity(
    fake_mcp_logout: list[str],
) -> None:
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()
    session.mcp_manager = _GrantManager()

    result = await handle._slash_result("mcp", "reauth", SlashResult)
    assert result.text == "usage: /mcp reauth <name>"
    assert result.style == "warning"

    result = await handle._slash_result("mcp", "reauth a b", SlashResult)
    assert "takes one server name" in result.text
    assert session.mcp_manager.connected == []


@pytest.mark.asyncio
async def test_the_settled_grant_reaches_viewers_as_a_notice(
    fake_mcp_logout: list[str],
) -> None:
    """The receipt cannot be the result frame, so it must be an event.

    The invoking client abandons the request after ACK_TIMEOUT_S; a NoticeEvent
    is the channel that already fans out to every attached front end.
    """
    from local_operator.harness.types import NoticeEvent
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()
    session.mcp_manager = _GrantManager()
    emitted: list[object] = []

    async def _emit(event: object) -> None:
        emitted.append(event)

    session._emit = _emit

    await handle._slash_result("mcp", "login notion", SlashResult)
    # The grant settles first; the notice it emits is a SEPARATE task on the
    # ordinary holder (a notice must not go through the superseding path, or
    # it would cancel the grant reporting it).
    for task in list(handle._mcp_grant_tasks):
        await asyncio.gather(task, return_exceptions=True)
    for task in list(handle._mcp_reload_tasks):
        await asyncio.gather(task, return_exceptions=True)

    notices = [e for e in emitted if isinstance(e, NoticeEvent)]
    assert notices, "the settled grant never reached the event stream"
    assert "authenticated MCP server 'notion'" in notices[0].text


@pytest.mark.asyncio
async def test_a_second_grant_supersedes_the_first(fake_mcp_logout: list[str]) -> None:
    """F3: every grant binds the same loopback redirect port, so only one runs.

    Two concurrent exchanges race for that port and the loser fails with a bind
    error describing nothing the user did. The TUI has always serialised these
    through an exclusive worker group; the runtime had no equivalent.
    """
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()

    class _Blocking(_GrantManager):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def connect_configured_server(self, name, *, timeout_ms=None):  # noqa: ANN001, ANN202
            await self.release.wait()
            return await super().connect_configured_server(name, timeout_ms=timeout_ms)

    session.mcp_manager = _Blocking()
    notices: list[str] = []

    async def _emit(event: object) -> None:
        notices.append(getattr(event, "text", ""))

    session._emit = _emit

    await handle._slash_result("mcp", "login one", SlashResult)
    await asyncio.sleep(0)
    first = list(handle._mcp_grant_tasks)
    assert len(first) == 1

    await handle._slash_result("mcp", "login two", SlashResult)
    await asyncio.sleep(0)
    # The first was cancelled to make room, not left racing the second. The
    # count is of LIVE grants: the cancelled task's done-callback has not run
    # yet, so it is briefly still in the set — what matters is that exactly one
    # is still contending for the redirect port.
    assert first[0].cancelling() or first[0].cancelled() or first[0].done()
    live = [t for t in handle._mcp_grant_tasks if not (t.done() or t.cancelling())]
    assert len(live) == 1

    session.mcp_manager.release.set()
    for task in list(handle._mcp_grant_tasks):
        await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    # The superseded grant still got an ending rather than vanishing.
    assert any("cancelled" in n for n in notices), notices


@pytest.mark.asyncio
async def test_dispose_cancels_a_grant_parked_on_a_browser(
    fake_mcp_logout: list[str],
) -> None:
    """F4: a grant waiting on a human must not outlive the session.

    Ten minutes is long enough for the session to be disposed underneath it,
    and a notice written into a disposed session is at best noise.
    """
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()

    class _Parked(_GrantManager):
        async def connect_configured_server(self, name, *, timeout_ms=None):  # noqa: ANN001, ANN202
            # Never returns: the "browser tab" the user never gets to.
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    session.mcp_manager = _Parked()
    await handle._slash_result("mcp", "login notion", SlashResult)
    await asyncio.sleep(0)
    tasks = list(handle._mcp_grant_tasks)
    assert len(tasks) == 1 and not tasks[0].done()

    await handle.dispose()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert tasks[0].cancelled() or tasks[0].done()


@pytest.mark.asyncio
async def test_model_saved_adopts_the_configured_default_on_a_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``/model saved`` WORKS on a detached runtime (QA round 2, Q49).

    ``OperatorApp`` intercepts ``saved`` before routing, so a local pane always
    honoured it while this handler — the one serving a DETACHED runtime's
    ``/model`` — saw a bare word with no ``/`` and answered the
    ``<provider>/<model-id>`` usage error. That made the keep notice emitted by
    this same change ("config.yml default changed, /model saved adopts it")
    a dead end on the phone and on any viewer of a runtime-owned session, and
    contradicted the ``/help`` text round 1's U5 added.
    """
    from local_operator.config import ConfigManager
    from local_operator.session.frontend_state import SlashResult

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "openai")
    manager.set_config_value("model_name", "gpt-5")

    handle, session = make_handle()
    switched: list[tuple[str, str]] = []

    async def _set_model(provider: str, model_id: str) -> str:
        switched.append((provider, model_id))
        session.model_label = f"{provider}/{model_id}"
        return f"model: {session.model_label}"

    # Substituted rather than run for real: `set_model` resolves provider
    # metadata over the network, and what this pins is the ROUTING — that
    # `saved` reaches the same mutation a `<provider>/<id>` switch does.
    monkeypatch.setattr(handle, "set_model", _set_model)

    result = await handle._model_slash(session, "saved", SlashResult)

    assert switched == [("openai", "gpt-5")], result.text
    assert "usage:" not in result.text
    assert "openai/gpt-5" in result.text


@pytest.mark.asyncio
async def test_model_saved_with_no_configured_default_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An empty config gets the app's own words, not a usage error: the two
    surfaces must answer "there is nothing to go back to" identically."""
    from local_operator.session.frontend_state import SlashResult

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    handle, session = make_handle()
    result = await handle._model_slash(session, "saved", SlashResult)
    assert "no boot default saved yet" in result.text
    assert "/model default <provider>/<model-id>" in result.text


@pytest.mark.asyncio
async def test_the_abort_op_stops_the_children_too() -> None:
    """The control op has NO second rung, so its one press must reach children.

    The keyboard's Esc ladder can afford a narrow first press because a second
    press is offered on screen. Nothing on this path has that: the mobile
    relay, a supervisor and `lop` peers send `abort` once into a session they
    cannot see. Reusing the keyboard's narrow semantics gave those callers a
    stop that could never end a runaway — the operator sent `abort`, was acked
    "stopping", and watched the meter run (QA Q-1).
    """
    handle, session = make_handle()
    cancelled: list[str] = []
    session.cancel_subagents = lambda reason="interrupted": (  # type: ignore[attr-defined]
        cancelled.append(reason),
        3,
    )[1]

    receipt = await handle.abort()

    assert session.aborts, "the turn itself must still be stopped"
    assert cancelled, "the abort op must reach the children; it is the only rung it has"
    assert "stopped 3 subagents" in receipt


@pytest.mark.asyncio
async def test_the_abort_receipt_does_not_claim_more_than_it_did() -> None:
    """The ack must not say "stopping" while things keep running.

    Returning the literal "stopping" whatever survived is how the operator was
    told the problem was handled while the meter ran. Backgrounded `bash` jobs
    are deliberately spared (`background=true` exists so a build outlives the
    turn), so the receipt has to name them rather than imply they stopped.
    """
    handle, session = make_handle()
    session.cancel_subagents = lambda reason="interrupted": 0  # type: ignore[attr-defined]

    async def never(job_id, signal, report_progress):  # noqa: ANN001, ANN202
        await asyncio.sleep(30)

    session.jobs.register("bash", "a long build", never)
    await asyncio.sleep(0.05)

    receipt = await handle.abort()

    assert "1 background job still running" in receipt
    assert "jobs cancel" in receipt
    # No children ran, so the receipt must not invent a number for them.
    assert "subagent" not in receipt


@pytest.mark.asyncio
async def test_the_abort_op_survives_a_session_that_cannot_stop_children() -> None:
    """A reduced host must get a stop, not an exception.

    `cancel_subagents` is getattr-probed like every other optional capability
    in this file: a stop must never fail because the thing it was asked to stop
    is not implemented.
    """
    handle, session = make_handle()
    assert not hasattr(session, "cancel_subagents")

    receipt = await handle.abort()

    assert "stopping this turn" in receipt

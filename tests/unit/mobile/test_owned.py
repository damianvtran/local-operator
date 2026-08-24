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

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.mobile import owned as owned_mod
from local_operator.mobile.owned import OwnedSessionHandle


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
        self._named: list[tuple[str, bool]] = []
        self._complete_calls: list[tuple[str, str]] = []
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

    def running_subagents(self) -> int:
        return 0

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

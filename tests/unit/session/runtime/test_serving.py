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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import (
    AskOption,
    AskQuestion,
    ModelSpec,
    NoticeEvent,
    SteeringDeliveredEvent,
)
from local_operator.providers.model_access import REHOME_BUSY_REPLY
from local_operator.session.frontend_state import SlashResult as _SlashResult
from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.session.naming import (
    TITLE_CORRECTIVE_ADDENDUM,
    ConversationName,
    fallback_from_opener,
)
from local_operator.session.protocol import RuntimeLocality
from local_operator.session.runtime import serving as serving_mod
from local_operator.session.runtime.serving import ServingSessionHandle


class FakeSession:
    """The slice of Session the ServingSessionHandle touches in these tests."""

    # Runtime role (SessionProtocol). This fake stands in for an OWNER:
    # it carries no attached runtime, which is what the absent legacy
    # `is_remote` meant.
    owns_runtime = True
    outcome_is_synchronous = True
    runtime_locality: RuntimeLocality = "this-process"

    #: Declared, deliberately UNASSIGNED: the naming tests set these per-case,
    #: and `_title_refresh_slash` probes the public one with
    #: `getattr(session, "conversation_name_state", None)`. A default here
    #: would make the attribute exist on every fake and silently move those
    #: tests onto the other branch, so the annotation states the surface this
    #: fake stands in for without creating it.
    _name_state: ConversationName
    conversation_name_state: ConversationName

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
        #: The boot record the listing consults when there is NO manager, since
        #: discovery that raised never assigns one and records itself here
        #: instead. ``None`` is the not-yet-wired state; a test wanting the
        #: failure state sets a real :class:`McpStartupOutcome` with failures.
        #: Declared for the same reason as ``mcp_manager``: the shape is part
        #: of the double's contract, and an undeclared attribute is invisible
        #: to the type checker every gate runs.
        self.mcp_startup: Any = None
        #: The session's event-emission seam. The runtime reports a settled
        #: MCP grant through it, since the grant outlives the request that
        #: started it. Tests replace it to capture what viewers would see.
        self._emit: Any = None
        #: The holder ``_install_interactivity_probe`` installs the model-facing
        #: probe onto. ``None`` matches a host that never grew one — an absent
        #: attribute reads the same way — and a test that pins the install puts a
        #: real :class:`GoalState` here. Declared for the same reason as
        #: ``mcp_manager``: an undeclared attribute is invisible to the type
        #: checker every gate runs.
        self._goal_state: Any = None
        # Tagged or a short untagged title both parse; the default stays
        # tagged so these tests stay independent of the untagged heuristics.
        self.title_reply = "<title>A Neat Title</title>"
        #: Per-call replies, consumed in order BEFORE ``title_reply``. The
        #: acceptance cascade spends up to two samples per attempt (sample +
        #: corrective/hedged resample), so a test that wants a wrapped FIRST
        #: sample needs the second reply spelled out — see the cascade tests.
        self.title_replies: list[str] = []
        from local_operator.harness.jobs import AsyncJobManager

        self.jobs = AsyncJobManager()

    # -- naming seams ----------------------------------------------------------
    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        self.conversation_name = text
        self._named.append((text, user_set))
        return text

    async def complete_once(self, system: str, prompt: str) -> str:
        self._complete_calls.append((system, prompt))
        if self.title_replies:
            return self.title_replies.pop(0)
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

    #: Children this double reports as live. A test that stages a cancel must
    #: move this too: the abort receipt counts what the roster says SURVIVED,
    #: not what `cancel_subagents` claims it dispatched, so a stub that returns
    #: a number while this stays put proves only that a call was made.
    running_children = 0

    def running_subagents(self) -> int:
        return self.running_children

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

    @property
    def variables(self) -> Any:
        """Memory-only store for ``/credential``, created on first use. The
        handle's own ``credential_op`` probes this attribute BY NAME on the
        session it wraps, so the double carries the real store the probe
        expects instead of answering ``None`` (the table's refusal)."""
        store = getattr(self, "_variables", None)
        if store is None:
            from local_operator.variables import VariableStore

            store = self._variables = VariableStore(cwd="/tmp", env={})
        return store

    async def variables_op(
        self, action: str, key: str = "", value: str = "", value_type: str = ""
    ) -> dict[str, Any]:
        """The REAL verb table against this fake's (empty) kernel registry.

        ``SessionProtocol`` declares code memory for every session shape and the
        desktop route reaches it BY NAME through the bridge's facade, so a double
        without it does not type as a session at all — the drift the declaration
        exists to catch rather than a test-only nuisance.

        The fake owns no interpreter, so the table answers exactly what a real
        session whose runtime has never run a cell answers: observed/absent for a
        read, ``no_kernel`` for a write. Delegating rather than hand-writing that
        envelope keeps ONE copy of the frozen shape in the tree, so the double
        cannot certify a branch the real session does not have.
        """
        from local_operator.session.variable_ops import run_variable_verb

        return await run_variable_verb(
            f"fake-{id(self):x}",
            action,
            key,
            value,
            value_type,
            redact=getattr(getattr(self, "variables", None), "redact", None),
        )

    async def credential_op(self, action: str, key: str = "", value: str = "") -> dict[str, Any]:
        """The REAL verb table against this fake's store, not a stub of it.

        ``SessionProtocol`` declares this verb for every session shape, and
        the TUI's submit seam probes it BY NAME — a double lacking it
        silently degrades every credential gesture driven through it to
        "this session cannot hold credentials", and a fake that swallows the
        verb is how #891 passed four review streams on an unreachable path.
        The canonical delegation rationale lives on
        ``test_app_pilot.FakeSession.credential_op``.
        """
        from local_operator.session.credential_ops import run_credential_verb

        return await run_credential_verb(
            self.variables, getattr(self, "journal_credential_change", None), action, key, value
        )


def make_handle(auto_approve: bool = False) -> tuple[ServingSessionHandle, FakeSession]:
    # The handle records whichever loop it is built on; inside an async test
    # that is the running loop, and the sync construction test never awaits so
    # a fresh loop is fine. get_event_loop_policy().get_event_loop() avoids the
    # "no current event loop" deprecation of the bare accessor.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    session = FakeSession()
    handle = ServingSessionHandle(session, loop, cwd="/tmp", auto_approve=auto_approve)
    return handle, session


def test_default_conversation_name_is_empty_not_a_placeholder() -> None:
    """A fresh mobile session shows nothing until named, so the phone's own
    fallback ("untitled" / cwd) applies — never a frozen "mobile session"."""
    handle, _ = make_handle()
    assert handle.session_projection_seed.conversation_name == ""


def test_the_started_hook_reaches_the_registrant() -> None:
    """``_publish_session_started`` (wired onto the session and called from
    ``_run_turn_pipeline``) flips the registrant's record bit, and is a no-op
    for a host that never grew one."""
    handle, _ = make_handle()
    started_calls: list[bool] = []

    class _Registrant:
        def set_record_started(self, started: bool) -> None:
            started_calls.append(started)

    handle._registrant = _Registrant()
    handle._publish_session_started()
    assert started_calls == [True]

    # A reduced host with no registrant must not fail the turn.
    handle._registrant = None  # type: ignore[attr-defined]
    handle._publish_session_started()
    assert started_calls == [True]

    # A registrant that predates the setter (a reduced test double) is skipped.
    handle._registrant = object()
    handle._publish_session_started()
    assert started_calls == [True]


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
async def test_a_refused_prompt_retried_as_a_prompt_is_really_admitted() -> None:
    """QA on PR #1528, Q1-5: the retry of a refused id is RUN, never swallowed.

    A prompt refused with ``TurnInFlight`` after the drain took it used to park
    its id as ``prompt-transfer``. The desktop's receipt journal and the phone
    retry a failed send with the SAME id as a PROMPT, and that retry answered
    "already admitted" without queueing anything: the message reached no
    transcript while the client was told it had. Here the retry must reach the
    session a second time and its receipt must be the real admission.
    """
    from local_operator.session.errors import TURN_IN_FLIGHT, TurnInFlight

    handle, session = make_handle()
    delivered: list[str] = []

    async def refuse_then_admit(  # noqa: ANN202
        text, images=None, *, message_id=None, admitted=None  # noqa: ANN001
    ):
        if not delivered:
            delivered.append("refused")
            raise TurnInFlight(TURN_IN_FLIGHT)
        delivered.append(text)
        assert message_id is not None and admitted is not None
        session.admit(message_id)
        admitted.set_result(None)

    session.prompt = refuse_then_admit  # type: ignore[method-assign]
    with pytest.raises(TurnInFlight):
        await handle.prompt("raced", command_id="retried-id")
    assert await handle.prompt("raced", command_id="retried-id") == "prompt admitted"
    assert delivered == ["refused", "raced"], "the retry never reached the session"
    # And from then on the durable index answers, so a THIRD send is the dedupe.
    assert await handle.prompt("raced", command_id="retried-id") == "already admitted"
    assert delivered == ["refused", "raced"]


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
    monkeypatch.setattr(serving_mod, "MAX_QUEUED_PROMPTS", 1)
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
    monkeypatch.setattr(serving_mod, "PENDING_REQUEST_TIMEOUT_S", 0.05)
    handle, _ = make_handle(auto_approve=False)
    waiting = asyncio.ensure_future(handle._approval_gate("bash", "one"))
    await asyncio.sleep(0)
    assert handle.is_busy() is True
    assert await waiting is False
    assert handle.is_busy() is False
    assert serving_mod.PENDING_REQUEST_TIMEOUT_S == 0.05


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
async def test_a_parked_ask_names_the_question_it_is_waiting_on() -> None:
    """The announcement surfaces must carry the QUESTION, not a placeholder.

    Regression origin (#868): both announcing reads in ``ask_gate`` asked for a
    ``text`` attribute :class:`AskQuestion` does not have — and, with
    ``extra="forbid"``, can never be given — so both always fell through to the
    literal string ``"question"``. The reads that degraded are
    ``_parked_announcement`` (which ``reannounce_pending`` replays) and the
    timeout row REPLAYED TO THE MODEL. Two neighbouring surfaces were NOT
    affected and must not be claimed here: the projection card reads the field
    directly and was always correct, and ``lop sessions`` publishes only the
    gate KIND, so prose never travelled to it at all.

    It stayed invisible because this entire gate was unreachable: both of its
    hosts build ``has_ui=False``, which the ``build_ask_tool`` clause removed in
    the same change vetoed, so ``ask`` was never advertised and the body never
    ran. Fixing that gate is what makes this reachable, which is why the two
    land together.

    Both reads are asserted, and so is ``ask_pending_request`` on the SAME
    question. That pairing is the point: the projection seam was always correct,
    so a card that renders the prose while these two do not is what proves a
    wrong-attribute bug rather than a question nobody supplied.
    """
    handle, session = make_handle(auto_approve=False)

    appended: list[Any] = []

    class _Transcript:
        async def append_message(self, message):  # noqa: ANN001
            appended.append(message)

    setattr(session, "transcript", _Transcript())

    prose = "Deploy to production or roll back?"
    question = AskQuestion(
        id="deploy",
        question=prose,
        options=[AskOption(label="Deploy"), AskOption(label="Roll back")],
    )

    # 1. The live announcement: what the notification and the record's pending
    #    state are handed while the gate is parked.
    asked = asyncio.ensure_future(handle._ask_gate([question]))
    await asyncio.sleep(0)

    assert handle._parked_announcement is not None
    kind, title, detail = handle._parked_announcement
    assert kind == "ask"
    assert title == prose, f"the parked ask announced {title!r} instead of the question"
    # DETAIL, not just title: ``_announce_pending`` composes the notification
    # body from ``detail`` alone, so a banner built from the title is
    # unreachable. Passing prose only as the title left the ask toast reading
    # the static "Waiting for your answer" while the approval toast beside it
    # named its action (review R1 / UX U2 / QA Q2).
    assert detail == prose, (
        "the notification body reads `detail`, so an ask that passes prose only "
        f"as the title cannot name the question; got {detail!r}"
    )

    # 2. The projection seam, on the same question. Correct before this fix and
    #    after it — which is what localises the defect to the two reads above.
    pending = handle._fold.projection.pending
    assert pending is not None
    assert pending.title == prose

    pending_request_id = pending.request_id
    await handle.ask_answer(pending_request_id, "Deploy")
    assert await asyncio.wait_for(asked, 1) == {"deploy": ["Deploy"]}

    # 3. The timeout row, which is the read the MODEL sees on replay. Driven
    #    through the REAL gate rather than by calling ``_record_gate_timeout``
    #    directly: the defect was in what the gate PASSES that recorder, so a
    #    direct call would assert the argument this test supplied itself.
    handle._gate_timeout_s = lambda: 0.01  # type: ignore[method-assign]
    timed_out = await asyncio.wait_for(handle._ask_gate([question]), 2)

    assert timed_out is None, "an unanswered ask must settle to None, not hang"
    assert appended, "the expiry was not recorded at all"
    row = appended[0]
    assert row.details["kind"] == "ask"
    assert row.details["description"] == prose, (
        "the timeout row replayed to the model named "
        f"{row.details['description']!r} instead of the question"
    )


@pytest.mark.asyncio
async def test_a_parked_secret_ask_keeps_its_prose_off_the_notification() -> None:
    """A secret question must not name the credential on an OS banner.

    The counterpart to the prose-in-`detail` fix above, and the reason that fix
    is conditional rather than unconditional. A secret ask's prose names the key
    being requested ("Paste your OpenAI key"), and a desktop notification is
    delivered to a lock screen — the one surface that must not enumerate which
    of the user's credentials a session is missing. The VALUE was never exposed
    here (the announcement happens before any answer exists), so what this
    guards is the QUESTION.

    The card still carries it: a person who opens the session needs to know what
    is being asked, and the projection is not a lock screen. So this asserts the
    two surfaces DISAGREE on purpose — the terse fallback out of band, the full
    prose in band.
    """
    handle, _ = make_handle(auto_approve=False)

    question = AskQuestion(id="OPENAI_API_KEY", question="Paste your OpenAI key", secret=True)
    asked = asyncio.ensure_future(handle._ask_gate([question]))
    await asyncio.sleep(0)

    assert handle._parked_announcement is not None
    _kind, title, detail = handle._parked_announcement
    assert detail == "", (
        "a secret question's prose reached the notification body, which is "
        f"delivered to a lock screen; got {detail!r}"
    )
    # The title is only consumed in-process (``reannounce_pending``), and the
    # notification's own composition cannot promote it to the body when detail
    # is empty — so carrying the prose here costs nothing and keeps the replay
    # honest.
    assert title == "Paste your OpenAI key"

    # In band, the card names it: this is what a person opening the session
    # reads, and it must not be degraded to protect a banner.
    pending = handle._fold.projection.pending
    assert pending is not None
    assert pending.title == "Paste your OpenAI key"
    assert pending.secret is True

    await handle.ask_answer(pending.request_id, "sk-topsecret")
    assert await asyncio.wait_for(asked, 1) == {"OPENAI_API_KEY": ["sk-topsecret"]}
    # And the pasted value never rode either surface.
    assert "sk-topsecret" not in json.dumps(handle._fold.projection.to_json())
    assert handle._parked_announcement is None


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
async def test_title_refresh_retitles_a_detached_session_and_republishes() -> None:
    """``/title refresh`` on a runtime nobody is attached to.

    The receipt is the RETURN VALUE here, not a painted notice, so the call is
    awaited rather than detached: a handle that answered before the naming call
    settled would report a title that had not been decided. The republish is
    what keeps ``lop sessions`` and the resume picker from listing the session
    under the name it just stopped having.
    """
    handle, session = make_handle()
    session.conversation_name = "Fix the login flow"
    session._name_state = ConversationName(text="Fix the login flow", user_set=True)
    session.conversation_name_state = session._name_state
    session.history = lambda: [  # type: ignore[attr-defined]
        SimpleNamespace(role="user", text="fix the login redirect loop"),
        SimpleNamespace(role="assistant", text="done"),
        SimpleNamespace(role="user", text="now rewrite the billing importer"),
    ]
    session.title_reply = "<title>Billing importer rewrite</title>"
    republished: list[bool] = []
    handle._registrant = SimpleNamespace(  # type: ignore[attr-defined]
        _republish=lambda: republished.append(True)
    )

    result = await handle._rename_slash(session, "refresh", _SlashResult)

    assert result.text == "title refreshed: Billing importer rewrite"
    assert session.conversation_name == "Billing importer rewrite"
    # Stored as a GENERATED title, and the human latch released with it: asking
    # for a fresh name withdraws the one you typed.
    assert session._named[-1] == ("Billing importer rewrite", False)
    assert not session._name_state.user_set
    assert republished == [True], "a renamed session was left stale in the registry"


@pytest.mark.asyncio
async def test_a_renamed_session_reaches_the_projection_and_the_record() -> None:
    """A name nobody can read is not a rename.

    The projection's ``conversation_name`` has exactly one writer
    (``_refresh_state``), the heartbeat republishes the PROJECTION's copy every
    15 s, and ``_republish`` does not carry the name field at all. So a slash
    path that skipped ``_refresh_state`` did not merely lag — the stale name was
    re-asserted for the life of the session, and an attached phone kept the old
    header forever. Covers both writers through the one seam they share.
    """
    for args, expected in (
        ("refresh", "Billing importer rewrite"),
        ("Typed by hand", "Typed by hand"),
    ):
        handle, session = make_handle()
        session.conversation_name = "Fix the login flow"
        session._name_state = ConversationName(text="Fix the login flow")
        session.conversation_name_state = session._name_state
        session.history = lambda: [  # type: ignore[attr-defined]
            SimpleNamespace(role="user", text="fix the login redirect loop"),
            SimpleNamespace(role="assistant", text="done"),
            SimpleNamespace(role="user", text="now rewrite the billing importer"),
        ]
        session.title_reply = "<title>Billing importer rewrite</title>"

        await handle._rename_slash(session, args, _SlashResult)

        assert handle.session_projection_seed.conversation_name == expected, args


@pytest.mark.asyncio
async def test_the_rename_receipt_says_she_was_renamed_too_for_her_session() -> None:
    """UX round 1, U3: renaming HER conversation renames the assistant.

    The config row (``applied: aida.name``) is the audit trail; the receipt
    is where the expectation forms, so the product-wide half is said out
    loud. ``_aida_duty`` is the same gate the config sync itself reads, so
    the clause cannot claim a rename that did not sync.
    """
    handle, session = make_handle()
    session._aida_duty = True  # type: ignore[attr-defined]

    result = await handle._rename_slash(session, "Vega", _SlashResult)

    assert result.text == "renamed to Vega — she is now called Vega everywhere", result.text


@pytest.mark.asyncio
async def test_the_rename_receipt_stays_plain_for_every_other_session() -> None:
    """The coupling is HERS alone — an ordinary conversation's receipt is
    exactly the words it always was."""
    handle, session = make_handle()

    result = await handle._rename_slash(session, "Vega", _SlashResult)

    assert result.text == "renamed to Vega", result.text


@pytest.mark.asyncio
async def test_title_refresh_that_changes_nothing_keeps_the_name_and_the_latch() -> None:
    """A refresh is not a rename: "the name still fits" must leave both the
    title and the user's claim on it exactly as they were."""
    handle, session = make_handle()
    session.conversation_name = "Ledger reconciliation"
    session._name_state = ConversationName(text="Ledger reconciliation", user_set=True)
    session.conversation_name_state = session._name_state
    session.history = lambda: [  # type: ignore[attr-defined]
        SimpleNamespace(role="user", text="reconcile the ledger"),
        SimpleNamespace(role="assistant", text="done"),
    ]
    session.title_reply = "<title/>"  # the model says: unchanged

    result = await handle._rename_slash(session, "refresh", _SlashResult)

    assert result.text == "title unchanged: Ledger reconciliation"
    assert session.conversation_name == "Ledger reconciliation"
    assert session._name_state.user_set, "the rename was quietly revoked"
    assert session._named == []


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
async def test_a_wrapped_reply_is_corrected_before_it_reaches_the_store() -> None:
    """The operator's defect at the runtime worker: the wrapped reply must not
    be stored, and the corrective resample's clean answer must be."""
    handle, session = make_handle()
    session.title_replies = ["<PROBE#1>Probe session title", "Probe title recovered"]

    handle._maybe_name_conversation("recover the stale PR work")
    for _ in range(10):
        await asyncio.sleep(0)

    assert session._named == [("Probe title recovered", False)]
    assert session.conversation_name == "Probe title recovered"
    assert len(session._complete_calls) == 2
    assert TITLE_CORRECTIVE_ADDENDUM in session._complete_calls[1][0]
    assert handle._name_heal_text == ""


@pytest.mark.asyncio
async def test_an_exhausted_attempt_stores_the_fallback_and_arms_the_heal_once() -> None:
    """Both samples wrapped -> the opener lands (never the markup), and the
    one-shot heal is armed with the opener for the next completed turn."""
    handle, session = make_handle()
    opener = "recover the stale PR work"
    session.title_replies = ["<PROBE#1>Probe session title", "<PROBE#2>Probe session title"]

    handle._maybe_name_conversation(opener)
    for _ in range(10):
        await asyncio.sleep(0)

    assert session.conversation_name == fallback_from_opener(opener)
    assert "<" not in session.conversation_name
    assert handle._name_heal_text == opener
    assert len(session._complete_calls) == 2, "the attempt is bounded at two samples"


@pytest.mark.asyncio
async def test_the_heal_upgrades_the_fallback_once_and_never_loops() -> None:
    handle, session = make_handle()
    opener = "recover the stale PR work"
    session.title_replies = ["<first wrapped>", "<second wrapped>"]

    handle._maybe_name_conversation(opener)
    for _ in range(10):
        await asyncio.sleep(0)
    assert session.conversation_name == fallback_from_opener(opener)
    assert handle._name_heal_text == opener

    # The heal: one more acceptance attempt, whose clean sample replaces the
    # opener quote. (Two replies: the sample plus its classifier-less hedge.)
    session.title_replies = ["Probe acceptance landed", "Probe acceptance landed"]
    handle._maybe_heal_name()
    for _ in range(10):
        await asyncio.sleep(0)
    assert session.conversation_name == "Probe acceptance landed"
    assert handle._name_heal_text == ""
    assert len(session._complete_calls) == 4

    # SINGLE-SHOT: a later completed turn spends nothing, even though the
    # model is still leaking markup.
    session.title_replies = ["<again>", "<again again>"]
    handle._maybe_heal_name()
    for _ in range(10):
        await asyncio.sleep(0)
    assert session.conversation_name == "Probe acceptance landed"
    assert len(session._complete_calls) == 4


@pytest.mark.asyncio
async def test_the_turn_boundary_hook_spends_the_armed_heal() -> None:
    """``_on_turn_settled`` is where "the next completed turn" exists for every
    opener; the chaining is pinned here with its sibling consumers stubbed."""
    handle, session = make_handle()
    opener = "recover the stale PR work"
    session.title_replies = [
        "<first wrapped>",
        "<second wrapped>",
        "Probe acceptance landed",
        "Probe acceptance landed",
    ]
    handle._maybe_name_conversation(opener)
    for _ in range(10):
        await asyncio.sleep(0)
    assert handle._name_heal_text == opener

    handle._publish_busy_soon = lambda: None  # type: ignore[method-assign]
    handle._schedule_completion_announce = lambda **_: None  # type: ignore[method-assign]
    handle._on_turn_settled()
    for _ in range(10):
        await asyncio.sleep(0)

    assert session.conversation_name == "Probe acceptance landed"
    assert handle._name_heal_text == ""


@pytest.mark.asyncio
async def test_the_heal_spends_no_call_after_a_human_rename() -> None:
    handle, session = make_handle()
    opener = "recover the stale PR work"
    session.title_replies = ["<first wrapped>", "<second wrapped>"]
    handle._maybe_name_conversation(opener)
    for _ in range(10):
        await asyncio.sleep(0)
    assert handle._name_heal_text == opener

    state = ConversationName()
    state.set("Ledger reconciliation", user_set=True)
    session.conversation_name_state = state  # type: ignore[attr-defined]
    session.title_replies = ["never spent"]
    handle._maybe_heal_name()
    for _ in range(10):
        await asyncio.sleep(0)

    assert handle._name_heal_text == "", "the latch is consumed either way"
    assert len(session._complete_calls) == 2, "the rename spend no further call"


@pytest.mark.asyncio
async def test_an_unusable_opener_arms_the_heal_with_an_empty_store() -> None:
    """Tier 3 with nothing usable to fall back to: nothing is stored, and the
    one-shot latch STILL arms with the opener — the retry the next completed
    turn spends is exactly what an unnamed session has left (the once-only
    latch stays spent on this path, and no fallback store released it).
    """
    handle, session = make_handle()
    opener = "<\u56d7>"
    session.title_replies = ["<first wrapped>", "<second wrapped>"]

    handle._maybe_name_conversation(opener)
    for _ in range(10):
        await asyncio.sleep(0)

    assert session.conversation_name == "", "nothing usable -> nothing stored"
    assert session._named == []
    assert handle._name_heal_text == opener, "the heal survives an empty fallback"
    assert len(session._complete_calls) == 2


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
        handle = ServingSessionHandle.__new__(ServingSessionHandle)
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

    monkeypatch.setattr(serving_mod, "spawn_detached", _record, raising=False)
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


def test_the_model_facing_probe_reads_attachment_not_attention() -> None:
    """The probe behind ``<interactivity>`` asks the PRESENTATION question.

    Pinned as its own case because the two predicates have to stay separable: the
    desktop pane below has been dropped by the attention predicate (notification
    routing, rung 1) and is at the same time exactly the surface a question from
    this session would appear on. Reading attention here is what told the
    operator's own focused, visible app that nobody was at a screen.

    The probe is exercised through the real install seam, on a real ``GoalState``,
    so this also pins that it is LIVE: an answer cached at install time would
    freeze the block for the life of the session.
    """
    from local_operator.session.goal import GoalState

    handle, session = make_handle()

    class _PaneAttached:
        def __init__(self) -> None:
            self.attached = True

        def attached_surfaces(self) -> frozenset[str]:
            return frozenset({"desktop"}) if self.attached else frozenset()

        def watching_surfaces(self) -> frozenset[str]:
            return frozenset()

        def set_record_pending(self, _pending: str | None) -> None:
            return None

    registrant = _PaneAttached()
    session._goal_state = GoalState()
    handle._registrant = registrant
    handle._install_interactivity_probe()

    assert not handle._watching_surfaces()
    assert session._goal_state.is_interactive() is True
    # ...and it re-reads rather than reporting what it saw at install time.
    registrant.attached = False
    assert session._goal_state.is_interactive() is False


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

    from local_operator.session.runtime.serving import ServingSessionHandle

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "local_operator.tui.notify.detached_notify",
        lambda title, body, **kwargs: sent.append((title, body)) or True,
    )
    monkeypatch.setattr("local_operator.tui.notify.notifications_enabled", lambda *a, **k: True)

    handle = ServingSessionHandle.__new__(ServingSessionHandle)
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
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import build_session

    session = build_session(tmp_path, MockClient().stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
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

    # Mirrors the real helper's full signature, ``store`` included: the reauth
    # gate passes it positionally, and a narrower stub raises TypeError that
    # the gate reports as a failed removal — a refusal caused by the stub
    # rather than by the code under test.
    def _fake(name: str, cwd: str, store: object | None = None) -> str | None:
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

    # ``"local"`` is STATED rather than left to the default: the helpers this
    # reaches read an un-forwarded locality as relayed now (round 2, R2-1), and
    # this cell is the user at THIS machine — the one the regression above is
    # about. The sibling below spells ``"remote"`` for the refusal it covers.
    result = await handle._slash_result("mcp", "reauth notion", SlashResult, "local")

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

    await handle._slash_result("mcp", "login notion", SlashResult, "local")
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

    await handle._slash_result("mcp", "login one", SlashResult, "local")
    await asyncio.sleep(0)
    first = list(handle._mcp_grant_tasks)
    assert len(first) == 1

    await handle._slash_result("mcp", "login two", SlashResult, "local")
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
    await handle._slash_result("mcp", "login notion", SlashResult, "local")
    await asyncio.sleep(0)
    tasks = list(handle._mcp_grant_tasks)
    assert len(tasks) == 1 and not tasks[0].done()

    await handle.dispose()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert tasks[0].cancelled() or tasks[0].done()


@pytest.mark.asyncio
async def test_a_chosen_effort_is_applied_with_the_model_on_this_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The level arrives WITH the pair, so the first turn runs at the chosen depth.

    ``build_model_spec`` seeds the model's OWN default level, and a pair-only
    switch therefore replaces a chosen level with that seed — on the very send
    that consumes the viewer's intent. Asserted on the SPEC THE SESSION
    RECEIVES, because that is where the level either survives or is silently
    lost, and against a control built the same way for the no-effort case: an
    absent level must leave exactly the spec this RPC has always produced.
    """
    from local_operator.model.configure import build_model_spec

    handle, session = make_handle()
    applied: list[tuple[Any, bool]] = []
    # ``raising=False``: ``FakeSession`` deliberately has no ``set_model`` — this
    # test is about what the HANDLE sends it.
    monkeypatch.setattr(
        session,
        "set_model",
        lambda spec, explicit=False: applied.append((spec, explicit)),
        raising=False,
    )
    monkeypatch.setattr(handle, "_refresh_state", lambda: None)

    await handle.set_model_effort("deepseek", "deepseek-flash", "max")
    await handle.set_model("deepseek", "deepseek-flash")

    assert [(spec.provider, spec.model_id, spec.reasoning_effort) for spec, _ in applied] == [
        ("deepseek", "deepseek-flash", "max"),
        (
            "deepseek",
            "deepseek-flash",
            build_model_spec("deepseek", "deepseek-flash").reasoning_effort,
        ),
    ]
    assert [explicit for _, explicit in applied] == [
        True,
        True,
    ], "a switch is a deliberate choice, so a pinned fallback must be withdrawn"


@pytest.mark.asyncio
async def test_a_level_this_model_cannot_express_is_clamped_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored level outliving its ladder lands on the nearest rung it can have.

    The refusal belongs to the moment the user chooses (the create/preview
    routes, 422); by the time a level reaches an owner it is a stored decision,
    and failing the send over it would turn a stale record into a dead turn.
    ``xhigh`` is on no ``deepseek-flash`` ladder, which tops out at ``max``.
    """
    handle, session = make_handle()
    applied: list[Any] = []
    monkeypatch.setattr(
        session, "set_model", lambda spec, explicit=False: applied.append(spec), raising=False
    )
    monkeypatch.setattr(handle, "_refresh_state", lambda: None)

    await handle.set_model_effort("deepseek", "deepseek-flash", "xhigh")

    assert applied[0].reasoning_effort == "high"


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
    # The double reports its children through the SAME predicate the receipt
    # reads, so "stopped" has to be earned by the roster emptying rather than
    # by the stub's return value — the hole QA named in the first version of
    # this test, where a stubbed count proved a call was made and nothing more.
    session.running_children = 3  # type: ignore[attr-defined]

    def _cancel(reason: str = "interrupted") -> int:
        cancelled.append(reason)
        dispatched = session.running_children  # type: ignore[attr-defined]
        session.running_children = 0  # type: ignore[attr-defined]
        return dispatched

    session.cancel_subagents = _cancel  # type: ignore[attr-defined]

    receipt = await handle.abort()

    assert session.aborts, "the turn itself must still be stopped"
    assert cancelled, "the abort op must reach the children; it is the only rung it has"
    assert session.running_subagents() == 0, "the roster must actually drain"
    assert "stopped 3 subagents" in receipt
    assert "did NOT stop" not in receipt


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
async def test_an_abort_clears_a_card_that_outlived_its_turn() -> None:
    """C2, and the case that FAILED before this change: the ORPHAN card.

    A card parked in a LIVE turn was already cleared by the cancellation that
    follows ``_session.abort`` — the batch's abort watcher unwinds the parked
    await through the gate closure's ``finally`` — which is why the hole went
    unnoticed. A card that OUTLIVED its turn is parked on a future nothing will
    resolve: the drain gave up on a tool whose cleanup outran
    ``ABORT_DRAIN_TIMEOUT_S``, or the turn ended while the question was still on
    screen. The user's own stop then left the question up while the receipt
    said a turn had been stopped.

    No turn is live here on purpose — that IS the orphan — so the assertion is
    that the press settles the question anyway. Driven through the ask gate
    (the harder shape: an approval would resolve ``False`` either way) and
    asserted on the folded card the surface paints.
    """
    handle, _ = make_handle(auto_approve=False)
    parked = asyncio.ensure_future(
        handle._ask_gate(
            [
                AskQuestion(
                    id="env",
                    question="Which environment?",
                    options=[AskOption(label="prod"), AskOption(label="staging")],
                )
            ]
        )
    )
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None

    receipt = await handle.abort()
    assert "refused 1 waiting prompt" in receipt, receipt
    assert await asyncio.wait_for(parked, 2) is None, "the orphan card was never answered"
    assert handle._fold.projection.pending is None, "the question is still on screen"
    assert handle._pending_futures == {}


@pytest.mark.asyncio
async def test_an_abort_leaves_an_aborted_call_not_a_denial(tmp_path) -> None:
    """C1: settling the card first must not turn a STOP into a user refusal.

    The gate is denied BEFORE the turn is cut (that order is what closes the
    window where a fresh answer could start a tool mid-teardown), which puts a
    resolved deny on the awaiting gate one loop pass before the AbortSignal. If
    the loop acted on that value the tool would be recorded as ``User denied
    approval`` — blaming the user for a stop they did make, the same
    misattribution class as the crashed-gate bug this repo fixed once already.

    Driven against a REAL session, a REAL turn and a real parked card, and
    asserted on the TRANSCRIPT: that entry is what the user reads and what the
    next turn is shown, and a stub's return value cannot stand in for it.
    """
    from local_operator.harness.types import (
        AgentTool,
        ModelSpec,
        StreamEndEvent,
        StreamTextDelta,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import ENTRY_MESSAGE, Transcript

    def stream(request, signal):  # noqa: ANN001, ANN202
        async def gen():
            yield StreamTextDelta(delta="working")
            yield StreamToolCallDelta(index=0, id="c1", name="gated", argument_delta="{}")
            yield StreamEndEvent(stop_reason="toolUse")
            yield StreamTextDelta(delta="after the batch")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    executed: list[str] = []

    async def execute(tool_call_id, args, signal, on_update, context):  # noqa: ANN001, ANN202
        executed.append(tool_call_id)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="gated", content=[TextContent(text="it ran")]
        )

    transcript = Transcript(tmp_path / "sess")
    session = Session(
        model=ModelSpec(provider="test", model_id="T", context_window=100_000),
        stream_fn=stream,
        # The default tier is ``exec``, which is what parks a card.
        tools=[AgentTool(name="gated", execute=execute)],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable", "env"],
    )
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    turn = asyncio.ensure_future(session.prompt("run the gated tool"))
    try:
        for _ in range(500):
            if handle._pending_futures:
                break
            await asyncio.sleep(0.01)
        assert handle._pending_futures, "no card ever parked, so nothing was interrupted"

        receipt = await handle.abort()
        await asyncio.wait_for(turn, 10)

        assert handle._pending_futures == {}, "the card must not survive the stop"
        assert receipt.startswith("stopping this turn"), receipt
        assert executed == [], "a denied call must not have run"
        tool_rows = [
            entry.payload
            for entry in transcript.entries()
            if entry.type == ENTRY_MESSAGE and entry.payload.get("role") == "tool"
        ]
        assert tool_rows, "the tool came back with no result row at all"
        text = json.dumps(tool_rows)
        assert "aborted" in text, text
        assert "User denied approval" not in text, (
            "a stop was recorded as the user's own refusal: " + text
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_abort_receipt_names_a_turn_only_when_one_was_running() -> None:
    """C3: the receipt must not claim a turn on a press that found none.

    The counterpart to ``_abort_receipt``'s existing rule that survivors are
    named only when there ARE any. The orphan case above is exactly where the
    old opening clause lied: the user pressed stop with nothing running, saw a
    card still on screen, and was told a turn had been stopped. Both halves are
    asserted together because the contrast IS the rule.
    """
    handle, session = make_handle()
    session.cancel_subagents = lambda reason="interrupted": 0  # type: ignore[attr-defined]

    idle = await handle.abort()
    assert "stopping this turn" not in idle, idle
    assert idle.startswith("no turn was running"), idle
    # Nothing was denied either, and a receipt that named a refused prompt on a
    # press that settled no gate would be the same overstatement in the other
    # direction (the route answers `idle` for this state rather than reporting
    # an interrupt, precisely because there is nothing here to report).
    assert "refused" not in idle, idle

    session.is_streaming = True
    live = await handle.abort()
    assert live.startswith("stopping this turn"), live


@pytest.mark.asyncio
async def test_the_receipt_names_children_that_refused_to_die(tmp_path) -> None:
    """MAJOR-1: the count must be what DIED, not what was asked to die.

    `cancel_subagents` returns `len(running)` sampled at DISPATCH time, and
    per-child cancellation is fire-and-forget with failures swallowed by
    design. Reporting that number said "stopped 3 subagents" while two children
    were still running and spending — the exact overstatement this PR exists to
    remove, reintroduced one layer up.

    Deliberately driven against a REAL Session and real `task` rows, and
    asserted on the ROSTER rather than on a stubbed return: a double that
    reports its own count can only prove a call was made, never that anything
    died.
    """
    from local_operator.harness.types import StreamEndEvent
    from local_operator.session.session import ModelSpec, Session
    from local_operator.session.transcript import Transcript

    def stream(request, signal):  # noqa: ANN001, ANN202
        async def gen():
            await asyncio.sleep(0.01)
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    session = Session(
        model=ModelSpec(
            provider="test", model_id="T", context_window=100_000, supports_images=True
        ),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    started = asyncio.Event()

    async def child(job_id, signal, report_progress):  # noqa: ANN001, ANN202
        started.set()
        await asyncio.sleep(30)
        return "never"

    ids = [session.jobs.register("task", f"c{n}", child) for n in range(3)]
    await asyncio.wait_for(started.wait(), timeout=5)
    await asyncio.sleep(0.2)

    # Two of three children refuse to die: their cancel raises, which is the
    # failure `_cancel_job_quietly` swallows by design.
    real_cancel = session.jobs.cancel

    async def flaky_cancel(job_id, *, registrant_id=None):  # noqa: ANN001, ANN202
        if job_id in (ids[1], ids[2]):
            raise RuntimeError("child refuses to die")
        return await real_cancel(job_id, registrant_id=registrant_id)

    session.jobs.cancel = flaky_cancel  # type: ignore[assignment]

    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    receipt = await handle.abort()

    # The rows are the ground truth, exactly as QA read them.
    statuses = [session.jobs.get(job_id) for job_id in ids]
    assert [row.status for row in statuses if row is not None] == [
        "cancelled",
        "running",
        "running",
    ]
    assert "stopped 1 subagent" in receipt
    assert "2 subagents did NOT stop" in receipt
    assert "stopped 3 subagents" not in receipt, "the receipt must not claim the two that survived"

    session.jobs.cancel = real_cancel  # type: ignore[assignment]
    await session.dispose()


@pytest.mark.asyncio
async def test_the_abort_op_survives_a_session_that_cannot_stop_children() -> None:
    """A reduced host must get a stop, not an exception.

    `cancel_subagents` is getattr-probed like every other optional capability
    in this file: a stop must never fail because the thing it was asked to stop
    is not implemented.
    """
    handle, session = make_handle()
    assert not hasattr(session, "cancel_subagents")
    # A stop arrives with a turn running — the normal case — and the receipt's
    # opening clause says which state it found (see
    # ``test_the_abort_receipt_names_a_turn_only_when_one_was_running``).
    session.is_streaming = True

    receipt = await handle.abort()

    assert "stopping this turn" in receipt


@pytest.mark.asyncio
async def test_the_abort_op_really_terminates_live_children(tmp_path) -> None:
    """End to end on a REAL Session: the children are gone, not just counted.

    The receipt naming stopped subagents is worth nothing if they keep running,
    so this drives real jobs through the real job manager rather than a double
    and asserts on their status afterwards. The children here would run for 30
    seconds on their own, so a "cancelled" status cannot be them finishing.
    """
    from local_operator.harness.types import StreamEndEvent
    from local_operator.session.session import ModelSpec, Session
    from local_operator.session.transcript import Transcript

    def stream(request, signal):  # noqa: ANN001, ANN202
        async def gen():
            await asyncio.sleep(0.01)
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    session = Session(
        model=ModelSpec(
            provider="test", model_id="T", context_window=100_000, supports_images=True
        ),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    started = asyncio.Event()

    async def child(job_id, signal, report_progress):  # noqa: ANN001, ANN202
        started.set()
        await asyncio.sleep(30)
        return "a child that never finishes on its own"

    job_ids = [session.jobs.register("task", f"child-{n}", child) for n in range(3)]
    await asyncio.wait_for(started.wait(), timeout=5)
    await asyncio.sleep(0.2)
    assert session.running_subagents() == 3

    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    receipt = await handle.abort()

    def statuses() -> list[str]:
        rows = [session.jobs.get(job_id) for job_id in job_ids]
        assert all(row is not None for row in rows), "a registered job vanished"
        return [row.status for row in rows if row is not None]

    for _ in range(100):
        await asyncio.sleep(0.02)
        if all(status == "cancelled" for status in statuses()):
            break
    assert statuses() == ["cancelled"] * 3
    assert session.running_subagents() == 0
    assert "stopped 3 subagents" in receipt
    await session.dispose()


# --- /mcp LISTING on a routed session -----------------------------------------
#
# The regression these cover: the listing test read `manager.servers`, an
# attribute `McpManager` has never had, so its emptiness branch was taken on
# every session whose slash command routes to the owner. An explicit
# `/mcp list` therefore answered "no MCP servers configured." — every fresh
# viewer, and the phone projection, which shares this handler — while the same
# session's transcript was listing its configured servers failing to start by
# name. The sibling producers in the TUI never had the bug, which is why the
# bare `/mcp` typed locally rendered the right listing and `/mcp list` did not.


class _NameManager:
    """A manager exposing the REAL roster accessor, and nothing else.

    Deliberately without ``servers``: that phantom attribute is what the
    listing test used to read, and a double that carried it would keep the
    old bug green.
    """

    def __init__(self, names: list[str]) -> None:
        self._names = list(names)

    def get_all_server_names(self) -> list[str]:
        return list(self._names)


@pytest.mark.asyncio
async def test_a_routed_mcp_listing_asks_the_manager_for_its_servers() -> None:
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()
    session.mcp_manager = _NameManager(["alpha-stdio", "beta-oauth"])

    result = await handle._slash_result("mcp", "list", SlashResult)

    # A BLOCK is the instruction to render the listing; the notice below is the
    # refusal to. Which one comes back is the whole defect.
    assert result.kind == "block", result
    assert result.data == {"type": "mcp"}
    # THE ROSTER TRAVELS WITH THE BLOCK (review round 1, R1-1). A terminal draws
    # this block from its own live panel, but a surface WITHOUT one -- the phone --
    # was handed a bare type marker and answered "ran /mcp" with the roster it had
    # just read thrown away. ``text`` is that surface's line, and the terminal's
    # renderer returns on the block type before reading it, so nothing paints
    # twice.
    assert "alpha-stdio" in result.text and "beta-oauth" in result.text, result.text
    assert "2 MCP servers" in result.text, result.text


@pytest.mark.parametrize(
    ("manager", "startup"),
    [
        (None, None),
        (_NameManager([]), None),
        # A record that coexists with a manager and names NO failure — the
        # ordinary arm at ``session_factory.py:2462-2469``, which a host with no
        # MCP config reaches as ``configured=()`` / ``failures={}``. (The
        # import-gap arm at ``2377`` records an empty outcome too, but it returns
        # BEFORE any manager exists, so it cannot produce this pair.) Pinned so a
        # later predicate change cannot drop the arm (review round 2, R2-NIT-2;
        # citation corrected in review round 3, NIT-1).
        (_NameManager([]), McpStartupOutcome()),
    ],
    ids=["no-manager", "zero-name-manager", "deliberate-empty-outcome"],
)
@pytest.mark.asyncio
async def test_a_routed_mcp_listing_keeps_the_honest_empty_answer(
    manager: Any, startup: Any
) -> None:
    """The states that MAY say this, none of them with a failure recorded.

    Three shapes reach here and all are genuinely empty: a session whose wiring
    has not run yet (no boot record at all), a host that really asked for
    nothing — including the zero-name manager, which is the same answer the real
    one gives when no config file names a server (QA round 1, row 3) — and the
    recorded-but-empty outcome the ordinary wiring arm produces. An empty roster
    with a failure on the boot record is a different answer entirely; see the
    three tests below.
    """
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()
    session.mcp_manager = manager
    session.mcp_startup = startup

    result = await handle._slash_result("mcp", "list", SlashResult)

    assert result.kind == "notice"
    assert result.text == "no MCP servers configured."
    assert result.style == "info"


@pytest.mark.asyncio
async def test_a_recorded_but_empty_discovery_message_is_still_a_failure() -> None:
    """MEMBERSHIP decides, not the value (review round 3, MINOR-1).

    ``session_factory`` stores ``str(entry.get("error", ...))`` with no falsy
    filter, and ``str(exc)`` is ``""`` for an exception raised with no args — so
    a record can carry the discovery key with an empty message. Testing the
    VALUE fell through to the empty-state sentence and had `/mcp list` deny a
    failure it was holding; the fallback keeps that arm non-empty and says what
    is missing rather than inventing a cause.
    """
    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.mcp_status import MCP_DISCOVERY_KEY

    handle, session = make_handle()
    session.mcp_manager = _NameManager([])
    session.mcp_startup = McpStartupOutcome(failures={MCP_DISCOVERY_KEY: ""})

    result = await handle._slash_result("mcp", "list", SlashResult)

    assert result.kind == "notice"
    assert "no MCP servers configured." not in result.text
    assert "MCP discovery failed" in result.text
    assert "no error detail was recorded" in result.text
    assert result.style == "warning"


@pytest.mark.asyncio
async def test_a_routed_mcp_listing_does_not_blame_a_server_the_roster_lost() -> None:
    """A STALE per-server entry must not be spoken for an EMPTY roster.

    The boot record is written at boot and by its settle sink only, and that
    sink fires only when a round DEFERRED something — so `/mcp remove` reloads
    the manager into an empty roster while the record still names the server it
    just removed (``mcp/manager.py:2032`` assigns ``_configs`` before
    validating, so a fresh boot cannot produce this pair, but a config change
    can). Speaking that entry would have `/mcp list` announce "MCP server
    github failed: …" on a session that configures nothing, where the empty
    sentence is the true answer (review round 2, R2-MINOR-1).
    """
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()
    session.mcp_manager = _NameManager([])
    session.mcp_startup = McpStartupOutcome(failures={"github": "command not found: gh"})

    result = await handle._slash_result("mcp", "list", SlashResult)

    assert result.kind == "notice"
    assert result.text == "no MCP servers configured."
    assert "github" not in result.text


@pytest.mark.asyncio
async def test_a_routed_mcp_listing_names_a_failure_when_the_roster_is_empty() -> None:
    """The REACHABLE hard-failure shape — empty roster, manager present.

    ``discover_and_load_mcp_tools`` never raises for a discovery failure: it
    catches, logs, and returns the manager alongside a synthetic
    ``{"path": ".mcp.json"}`` error entry, which ``session_factory`` keys as
    ``discovery`` (``mcp/__init__.py:145-149``). So the state a user actually
    reaches is a MANAGER whose roster came back empty plus a boot record that
    says why, and keying the honest answer on ``manager is None`` missed it
    (QA round 1, Q2 — where this branch probe returned the old sentence).
    """
    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.mcp_status import MCP_DISCOVERY_KEY, McpStartupOutcome

    handle, session = make_handle()
    session.mcp_manager = _NameManager([])
    session.mcp_startup = McpStartupOutcome(
        failures={MCP_DISCOVERY_KEY: "the config layer could not be read"}
    )

    result = await handle._slash_result("mcp", "list", SlashResult)

    assert result.kind == "notice"
    assert "no MCP servers configured." not in result.text
    assert "the config layer could not be read" in result.text
    assert result.style == "warning"


@pytest.mark.asyncio
async def test_a_routed_mcp_listing_names_a_discovery_failure_instead_of_denying_it() -> None:
    """A discovery RAISE is the other shape that reaches the failure answer.

    ``wire_mcp_into_session`` never assigns ``mcp_manager`` when its own call
    raises, and records the exception on the boot record instead; that session
    is a machine which HAS an MCP setup that could not be read. The old guard
    answered "no MCP servers configured." there — the operator's reported
    sentence in the state where it is least true. The band refuses to say it
    (``_mcp_status`` reads ``startup.failed`` for exactly this reason), so the
    slash answer must not either. The reachable sibling of this state keeps the
    same answer: see the zero-name-manager test above.
    """
    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.mcp_status import MCP_DISCOVERY_KEY, McpStartupOutcome

    handle, session = make_handle()
    assert session.mcp_manager is None
    session.mcp_startup = McpStartupOutcome(
        failures={MCP_DISCOVERY_KEY: "no such file or directory: mcp.json"}
    )

    result = await handle._slash_result("mcp", "list", SlashResult)

    assert result.kind == "notice"
    assert "no MCP servers configured." not in result.text
    assert "no such file or directory: mcp.json" in result.text
    # A failure is not an empty state, and it is not styled as one either.
    assert result.style == "warning"


@pytest.mark.asyncio
async def test_a_manager_that_cannot_name_its_servers_does_not_kill_the_command() -> None:
    """A slash surface has no error page to render an exception on.

    Reading one attribute too far takes the whole app down with it, so a
    manager that cannot answer has to degrade. It must NOT degrade to "none
    configured" though: a roster we could not READ is not an empty roster, and
    borrowing the empty state's sentence is the same lie in a quieter place
    (review round 1, MAJOR-1, second half).
    """
    from local_operator.session.frontend_state import SlashResult

    handle, session = make_handle()

    class _Mute:
        def get_all_server_names(self) -> list[str]:
            raise RuntimeError("no roster")

    session.mcp_manager = _Mute()

    result = await handle._slash_result("mcp", "list", SlashResult)

    assert result.kind == "notice"
    assert result.text != "no MCP servers configured."
    assert "could not read" in result.text
    assert result.style == "warning"


@pytest.mark.asyncio
async def test_the_wire_op_refuses_a_decision_only_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The phone's model sheet takes bare strings, so the handle must refuse too.

    ``server.py``'s ``set_model`` arm passes whatever ``provider``/``model_id`` a
    client sent straight here, and any client can send any pair — this is the one
    live-switch surface with no picker, no catalogue and no config file in front of
    it. The refusal must leave the session untouched: the spec is built (and
    therefore refused) BEFORE ``Session.set_model`` is reached, which is what the
    ``applied`` list asserts.
    """
    handle, session = make_handle()
    applied: list[Any] = []
    monkeypatch.setattr(
        session, "set_model", lambda spec, explicit=False: applied.append(spec), raising=False
    )
    monkeypatch.setattr(handle, "_refresh_state", lambda: None)

    with pytest.raises(ValueError) as refused:
        # MIXED CASE, deliberately: the wire takes the frame's strings verbatim, so the
        # spelling is the client's and not a user's — a lowercase-only test would pass
        # on the very hole this closes (review round 3, MAJOR 1).
        await handle.set_model_effort("TypeSafe", "jev-1.13", None)

    assert "serves decision-model calls, not chat completions" in str(refused.value)
    assert applied == [], "the session must not be switched onto a provider that cannot chat"


# =============================================================================
# The aside seam: the instruction and the stream, applied where every remote
# caller converges
# =============================================================================


class _AsideSession:
    """A session that only knows how to be asked off the record.

    Deliberately NOT ``FakeSession``: the subject here is the SEAM — what the
    handle hands the primitive and what it forwards back — so the primitive is a
    recorder rather than a stand-in for the whole Session.
    """

    session_id = "aside-seam"
    _frontend_state_store = None

    def __init__(self, *, accepts_delta: bool = True) -> None:
        self.turns: list[list[Any]] = []
        self._accepts_delta = accepts_delta

    async def complete_aside(
        self, turns: list[Any], *, on_delta: Any = None, on_usage: Any = None
    ) -> str:
        # The REAL ``Session.complete_aside`` signature, spelled out rather than
        # swallowed by ``**kwargs``: the handle probes these NAMES by signature,
        # so a catch-all here would silently test the legacy path instead.
        self.turns.append(list(turns))
        if self._accepts_delta and on_delta is not None:
            on_delta("part one. ")
            on_delta("part two.")
        return "part one. part two."


class _LegacyAsideSession:
    """A session built before the stream existed: it takes ``turns`` alone."""

    session_id = "aside-seam-legacy"
    _frontend_state_store = None

    def __init__(self) -> None:
        self.turns: list[list[Any]] = []
        self.deltas: list[Any] = []

    async def complete_aside(self, turns: list[Any]) -> str:
        self.turns.append(list(turns))
        return "answer."


def _aside_handle(session: Any) -> ServingSessionHandle:
    return ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd="/tmp", install_gates=False
    )


@pytest.mark.asyncio
async def test_complete_aside_wraps_the_turn_and_forwards_the_stream() -> None:
    """ONE placement, both halves: the instruction at the seam and the chunks out.

    The wrap lives here rather than in ``Session.complete_aside`` because the
    goal-loop judge calls that in-process and must never receive an aside
    instruction; the cost of getting this wrong was a desktop ``/asides`` route
    that sent the raw question and no instruction at all, which is why the
    desktop path showed tool calls and the TUI did not.
    """
    from local_operator.session.aside import ASIDE_PROMPT

    session = _AsideSession()
    deltas: list[str] = []

    answer = await _aside_handle(session).complete_aside(
        [{"role": "user", "content": [{"type": "text", "text": "why?"}]}],
        on_delta=deltas.append,
    )

    assert answer == "part one. part two."
    assert deltas == ["part one. ", "part two."]
    (sent,) = session.turns
    assert sent[-1].text == ASIDE_PROMPT.format(question="why?")


@pytest.mark.asyncio
async def test_complete_aside_wraps_a_continuation_only_at_its_new_question() -> None:
    """The earlier pairs are the aside's own exchanges and stay RAW.

    Wrapping them would ask the model to answer a question it already answered;
    leaving them raw is also what lets an adopted exchange contain the words the
    user typed (``DesktopSessionBridge`` stores this same list).
    """
    from local_operator.session.aside import ASIDE_PROMPT

    session = _AsideSession()

    await _aside_handle(session).complete_aside(
        [
            {"role": "user", "content": [{"type": "text", "text": "first?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first."}]},
            {"role": "user", "content": [{"type": "text", "text": "second?"}]},
        ]
    )

    (sent,) = session.turns
    assert [m.text for m in sent[:2]] == ["first?", "first."]
    assert sent[2].text == ASIDE_PROMPT.format(question="second?")


@pytest.mark.asyncio
async def test_complete_aside_with_the_flag_off_sends_the_turns_untouched() -> None:
    """``aside_instruction=False`` is the caller's own instruction, honoured.

    This is the TUI's ``/btw`` overlay (which formats ``ASIDE_PROMPT`` itself)
    and its goal-loop judge (whose question is ``LOOP_JUDGE_PROMPT``). Both
    reach this handle when the TUI is merely VIEWING another owner, and the
    judge's case is the one that must never be wrapped: the model would be told
    to answer a question about session state briefly and off the record, from a
    request whose whole purpose is to report a verdict about a running goal.
    """
    from local_operator.session.goal_loop import LOOP_JUDGE_PROMPT

    session = _AsideSession()
    asked = LOOP_JUDGE_PROMPT.format(goal="finish the report")

    await _aside_handle(session).complete_aside(
        [{"role": "user", "content": [{"type": "text", "text": asked}]}],
        aside_instruction=False,
    )

    (sent,) = session.turns
    assert [m.text for m in sent] == [asked]
    assert "<aside>" not in sent[0].text
    assert "OFF\nTHE RECORD" not in sent[0].text


@pytest.mark.asyncio
async def test_complete_aside_does_not_double_wrap_a_pre_wrapped_turn() -> None:
    """The BELT, exercised through the seam: the default flag cannot double it.

    A caller that supplies its own instruction and forgets the flag is the
    regression this PR fixes on the TUI-attached-to-a-remote-owner path, where
    the measured result was two ``<aside>`` blocks and two ``Question:`` lines in
    one turn. ``wrap_aside_turns`` being idempotent is what makes that caller
    harmless rather than merely unlikely.
    """
    from local_operator.session.aside import ASIDE_PROMPT

    session = _AsideSession()
    already = ASIDE_PROMPT.format(question="why?")

    await _aside_handle(session).complete_aside(
        [{"role": "user", "content": [{"type": "text", "text": already}]}]
    )

    (sent,) = session.turns
    assert sent[0].text == already
    assert sent[0].text.count("<aside>") == 1
    assert sent[0].text.count("Question:") == 1


@pytest.mark.asyncio
async def test_complete_aside_tolerates_a_primitive_without_on_delta() -> None:
    """A session built before the stream must still answer, in one piece.

    The capability is probed by signature, so an older session is handed turns
    alone rather than a keyword it would raise on — the failure a caller would
    see as "the aside is broken" against precisely the deployments that have
    asked for nothing.
    """
    session = _LegacyAsideSession()

    answer = await _aside_handle(session).complete_aside(
        [{"role": "user", "content": [{"type": "text", "text": "why?"}]}],
        on_delta=lambda _delta: None,
    )

    assert answer == "answer."
    assert len(session.turns) == 1


@pytest.mark.asyncio
async def test_the_runtimes_routed_seam_refuses_a_delete_verb_with_no_capability_set() -> None:
    """R3-3 on the RUNTIME host: the capability half, with the keyword left out.

    ``ServingSessionHandle._slash_result`` is the seam a relayed frame lands on, and
    its delete-scoped gate reads ``capabilities``. Until this cell existed, the
    mutation ``may_run_delete_scoped_slash = lambda loc, caps: loc == "local" or caps
    is None or "delete" in caps`` passed every delete-gate test in the suite — the
    routed cells all carried a set — so a relayed caller that forwarded NO set could
    archive on this host. Both spellings of "nothing" are pinned, and the allowed
    direction with them, so a gate that simply refused every relayed caller could
    not satisfy this either.
    """
    from local_operator.session.frontend_state import SlashResult

    handle, _session = make_handle()
    for locality in ("remote", None):
        for capabilities in (None, frozenset()):
            refused = await handle._slash_result(
                "archive", "", SlashResult, locality, None, capabilities
            )
            assert refused.kind == "notice", refused
            assert refused.style == "warning", refused
            assert "delete" in refused.text, refused
    allowed = await handle._slash_result(
        "archive", "", SlashResult, "remote", None, frozenset({"slash", "delete"})
    )
    assert "capability" not in allowed.text, allowed


# -- the child-roster republish is coalesced off the per-event path -----------
#
# WHY. ``set_subagent_details`` is one linear pass over a registry capped at 256
# records, and it ran inline for EVERY root event -- token rate on a parent with
# live lanes, on the loop that admits the user's next message. Measured with
# ``scripts/bench_send_admission.py --condition roster``: 30% of loop samples,
# and a prompt waited 0.8 s p50 behind it. Structural pins: calls counted, never
# a clock.


class _CountingComms:
    """A registry double that counts roster passes and reports none of them."""

    def __init__(self) -> None:
        self.passes = 0

    def roster_pass(self, now: Any = None) -> Any:
        self.passes += 1
        return SimpleNamespace(
            roster=lambda: [], lifecycles=lambda: {}, nodes=lambda: [], job=lambda _job_id: None
        )


@pytest.mark.asyncio
async def test_a_burst_of_root_events_costs_one_roster_pass() -> None:
    """Fifty streamed events inside one coalescing window -> one registry pass."""
    session = FakeSession()
    comms = _CountingComms()
    session._subagent_comms = comms  # type: ignore[attr-defined]
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    projections: list[int] = []
    handle.subscribe(lambda: projections.append(comms.passes))
    comms.passes = 0  # the attach seed is a deliberate, separate pass

    for _ in range(50):
        session.emit(NoticeEvent(text="tick", kind="info"))
    assert comms.passes == 0, "the per-event path must not walk the registry inline"
    projections.clear()

    # Wait on the coalescer's own flag rather than a sleep sized to the window.
    for _ in range(200):
        if not handle._roster_refresh_scheduled:
            break
        await asyncio.sleep(serving_mod._ROSTER_REFRESH_COALESCE_S / 5)
    assert comms.passes == 1, f"a burst must fold into ONE pass, saw {comms.passes}"
    # The deferred pass must also be PUBLISHED: it is what carries a quiet
    # session's final roster to the phone and desktop. Review round 1 (F2)
    # turned the flush's ``_notify()`` into ``pass`` and 727 tests stayed
    # green; a projection callback that observed the pass is what was missing.
    assert (
        projections and projections[-1] == 1
    ), "the deferred roster pass was folded but never published to the viewer"


@pytest.mark.asyncio
async def test_a_subagent_start_still_republishes_the_roster_inline() -> None:
    """The fold REBUILDS a started child's row without its session id; the
    registry's identity must be re-applied before this event's push goes out."""
    from local_operator.harness.types import SubagentStartEvent

    session = FakeSession()
    comms = _CountingComms()
    session._subagent_comms = comms  # type: ignore[attr-defined]
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    handle.subscribe(lambda: None)
    comms.passes = 0

    session.emit(SubagentStartEvent(job_id="child", label="child"))

    assert comms.passes == 1


# =============================================================================
# The spend + context glance (mobile parity phase 1)
# =============================================================================


def test_the_refresh_path_publishes_the_stores_spend_and_context_block() -> None:
    """The phone's per-event refresh reads the canonical store, field for field.

    This doubles as the allow-list guard: ``read_field`` raises ``KeyError``
    for any name the shareable set does not carry, and this path runs after
    every folded event, so a missing name would be a per-event crash on a live
    session rather than a lag. Every value is compared so a refresh that stops
    reading one fails here.

    ``None`` is asserted through as a VALUE (the store holds nulls; the
    projection must too) — see ``set_spend_context`` for why a
    skip-on-``None`` assignment would be the wrong shape.
    """
    from local_operator.harness.types import Usage
    from local_operator.session.frontend_state import (
        CostKnowledge,
        FrontendSessionState,
        FrontendStateStore,
    )

    handle, session = make_handle()
    store = FrontendStateStore(FrontendSessionState(session_id="sess-1", epoch="t"))
    session._frontend_state_store = store  # type: ignore[attr-defined]
    store.mutate(
        cumulative_parent_cost=1.25,
        child_costs={"job-a": 0.5},
        subagent_cost=None,
        subagent_cost_knowledge=None,
        cost_knowledge=CostKnowledge.PARTIAL,
        context_tokens=12_400,
        context_window=200_000,
        context_is_estimate=True,
        last_usage=Usage(input_tokens=9_000, output_tokens=100),
    )

    handle._refresh_state()

    p = handle._fold.projection
    assert p.cumulative_parent_cost == 1.25
    assert p.child_costs == {"job-a": 0.5}
    assert p.subagent_cost is None, "the store's null must publish as a null"
    assert p.subagent_cost_knowledge is None
    assert p.cost_knowledge == CostKnowledge.PARTIAL
    assert p.context_tokens == 12_400
    assert p.context_window == 200_000
    assert p.context_is_estimate is True
    assert p.usage == {"input_tokens": 9_000, "output_tokens": 100}


def test_a_store_less_session_keeps_the_spend_and_context_defaults() -> None:
    """A reduced host (no store) must not invent numbers, and must not crash.

    ``FakeSession`` carries no store, like an embedder facade: the defaults
    stand — ``None``/``{}``/``"unknown"`` — which is what a durable-only
    rebuild lands on too. This is the other half of the refresh contract: the
    block is populated only from the store, never from a synthesised zero.
    """
    handle, _ = make_handle()
    handle._refresh_state()
    p = handle._fold.projection
    assert p.cumulative_parent_cost is None
    assert p.child_costs == {}
    assert p.subagent_cost is None
    assert p.subagent_cost_knowledge is None
    assert p.cost_knowledge == "unknown"
    assert p.context_tokens is None
    assert p.context_window is None
    assert p.context_is_estimate is None
    assert p.usage == {}


@pytest.mark.asyncio
async def test_a_restored_receipt_pair_is_re_materialized_when_the_host_takes_the_session_up(
    tmp_path,
) -> None:
    """QA round 1, Q1: a durable resume keeps the pair the phone's ``$—`` reads.

    The turn-end checkpoint is written while the turn's final ``AgentEndEvent``
    is still held (``Session._held_end`` folds at the held-end flush, after
    ``_run_turn``'s durable write), and that fold is what lands the aggregate
    receipt in ``last_usage`` — so a session restored from the checkpoint
    carries the money and the context but ``last_usage: null``. Without a
    refresher the serving host published ``usage={}``, and the phone dropped
    its ``$—``: a resumed session silently read "we have not spent" where the
    live one read "we spent something we cannot price".

    The TUI host re-materializes the pair at its adopt edge
    (``OperatorApp._adopt_session`` -> ``refresh_frontend_usage``); this pins
    the serving host doing the same when it takes the session up, from the
    transcript replay, before any frame is served. Both the store accessor the
    refresh path uses and the folded projection are asserted, so removing the
    constructor call fails here on the defect's own shape.
    """
    from unittest.mock import patch

    from local_operator.harness.types import Message, ModelSpec, TextContent, Usage
    from local_operator.model.registry import ModelInfo
    from local_operator.session.frontend_state import (
        CostKnowledge,
        FrontendSessionState,
        FrontendStateStore,
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    sid = "restored-receipt"
    directory = tmp_path / "sessions" / sid
    directory.mkdir(parents=True)

    # The durable prior conversation: one settled assistant call whose receipt
    # the turn-end checkpoint could not carry, and the checkpoint itself —
    # money and context present, `last_usage` absent, exactly the shape QA
    # read off the resumed wire.
    writer = Transcript(directory)
    await writer.append_message(Message.user("what did this cost?"))
    await writer.append_message(
        Message(
            role="assistant",
            content=[TextContent(text="answer")],
            stop_reason="stop",
            usage=Usage(
                provider="openai",
                model_id="e2e-oracle",
                input_tokens=9_000,
                output_tokens=100,
                context_tokens=12_400,
            ),
        )
    )
    checkpoint_store = FrontendStateStore(FrontendSessionState(session_id=sid, epoch="t"))
    checkpoint_store.mutate(
        cumulative_parent_cost=None,
        cost_knowledge=CostKnowledge.UNKNOWN,
        context_tokens=12_400,
        context_is_estimate=False,
    )
    await checkpoint_store.checkpoint(writer)

    def _no_price(provider: str, model_id: str) -> ModelInfo:
        # No price row: the call is unpriceable, deterministically (a real
        # resolve would fire discovery HTTP on a worker thread).
        return ModelInfo(id=model_id, name=model_id, description="")

    async def _stream(request: Any, signal: Any = None):  # pragma: no cover — never called
        if False:
            yield None

    with patch.multiple(
        "local_operator.model.configure",
        resolve_model_info=_no_price,
        resolve_model_info_paint=lambda provider, model_id: (_no_price(provider, model_id), True),
    ):
        # The resumed session, replayed from disk exactly as `--resume` builds it.
        session = Session(
            model=ModelSpec(provider="openai", model_id="e2e-oracle", context_window=0),
            stream_fn=_stream,
            tools=[],
            transcript=Transcript(directory),
            session_id=sid,
            system_blocks_provider=lambda: ["system"],
        )
        store = session._frontend_state_store

        # PREMISE, on the defect's own shape: the replay has the receipt; the
        # durable state the store restored does not.
        assert session.restored_usage() is not None
        assert store.spend_context_copy() == ({}, {}), "the restored checkpoint must lack the pair"

        # The fix: taking the session up re-materializes the pair before any
        # frame is served.
        handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))

        _child_costs, usage = store.spend_context_copy()
        assert usage == {"input_tokens": 9_000, "output_tokens": 100}
        assert store.state.last_usage is not None
        # The readings the checkpoint DID carry are kept, not invented away.
        assert store.read_field("context_tokens") == 12_400
        assert store.read_field("cost_knowledge") == CostKnowledge.UNKNOWN

        # And they reach the phone's projection, which is what reads `usage`
        # as the `$—` billed signal.
        handle._refresh_state()
        assert handle._fold.projection.usage == {"input_tokens": 9_000, "output_tokens": 100}

        # Drain the one fire-and-forget rebuild the adoption starts (a session
        # with usage rows and no record); its publish is not this test's claim,
        # and it must not outlive the loop.
        if session._spend_tasks:
            await asyncio.gather(*list(session._spend_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_audio_is_probed_away_for_a_session_that_predates_the_keyword() -> None:
    """A reduced session must never receive a keyword its signature lacks.

    Same contract as the silent input metadata above: the handle decodes and
    holds the blocks, and the drain adds ``audio`` only when the SESSION's own
    signature takes the keyword. ``FakeSession`` here carries the pre-carriage
    production shape (``prompt(text, images)``), so a leaked keyword would be a
    ``TypeError`` inside the drain — the assertion is that the call lands and
    the recorded shape is unchanged. The accepting half of the probe is pinned
    against a real ``Session`` in ``test_audio_carriage.py``.
    """
    import base64

    wav = base64.b64encode(b"RIFF" + b"\x00" * 4 + b"WAVE").decode("ascii")
    handle, session = make_handle()
    session.prompt_release.set()

    await handle.prompt(
        "with a recording",
        audio=[{"data_b64": wav, "mime_type": "audio/wav"}],
        command_id="audio-probe",
    )

    # The drain runs as a background task; poll like ``test_concurrent_ordinary
    # _prompts_are_admitted_fifo`` does rather than sleeping a fixed turn count.
    deadline = asyncio.get_running_loop().time() + 5
    while not session.prompt_calls:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.01)
    assert session.prompt_calls == ["with a recording"]


# ---------------------------------------------------------------------------
# ``rehome_if_current`` — the compare-and-set behind a sign-in's repair
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Collects what the handle emits; notices ride a fire-and-forget task."""

    def __init__(self) -> None:
        self.events: list[object] = []

    async def __call__(self, event: object) -> None:
        self.events.append(event)

    def notices(self) -> list[str]:
        return [e.text for e in self.events if isinstance(e, NoticeEvent)]


def _rehome_handle(
    monkeypatch: pytest.MonkeyPatch, current: str = "radient/auto"
) -> tuple[Any, FakeSession, list[Any], _EventRecorder]:
    """A handle whose session is pinned to ``current`` and records switches.

    Returns ``(handle, session, applied_specs, recorder)`` — the records live in
    the test's own variables rather than on the fake, so pyright checks every
    read (the fake declares no such attributes, and a typo here would otherwise
    be a passing test that asserts nothing).
    """
    handle, session = make_handle()
    session.model_label = current
    applied: list[Any] = []

    def _set_model(spec: Any, *, explicit: bool = False) -> None:
        applied.append(spec)
        session.model_label = f"{spec.provider}/{spec.model_id}"

    monkeypatch.setattr(session, "set_model", _set_model, raising=False)
    recorder = _EventRecorder()
    session._emit = recorder
    return handle, session, applied, recorder


def _patch_access(monkeypatch: pytest.MonkeyPatch, accessible: set[str] | None) -> None:
    """Point the owner-side credential read at a fixed set (None = unreadable)."""
    from local_operator.providers import model_access

    monkeypatch.setattr(
        model_access, "credentialed_chat_providers_here", lambda **kwargs: accessible
    )


async def _settle_notices(handle: Any) -> None:
    """A notice is emitted on the handle's own task holder — await it, do not sleep."""
    for task in list(handle._mcp_reload_tasks):
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rehome_switches_a_stranded_idle_session_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported bug's second half: the pin outranks config, so the OWNER moves it.

    Asserted on three things at once, because any one alone can lie: the receipt
    word the caller counts, the SPEC the session received (the switch really
    happened), and the emitted notice (the user is told why).
    """
    handle, session, applied, recorder = _rehome_handle(monkeypatch)
    _patch_access(monkeypatch, {"deepseek"})

    detail = await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")

    assert detail == "rehomed: radient/auto → deepseek/deepseek-flash"
    assert [(spec.provider, spec.model_id) for spec in applied] == [("deepseek", "deepseek-flash")]
    await _settle_notices(handle)
    assert recorder.notices() == ["Switched to deepseek/deepseek-flash — not signed in to radient."]


@pytest.mark.asyncio
async def test_rehome_loses_to_a_pick_that_landed_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compare-and-set's whole reason: a /model that landed in between WINS.

    The desktop's candidate list is a snapshot; this re-check is what keeps a
    stale one from clobbering a deliberate choice with a model the user did not pick.
    """
    handle, session, applied, _recorder = _rehome_handle(
        monkeypatch, current="anthropic/claude-opus-5-5"
    )
    _patch_access(monkeypatch, {"deepseek", "anthropic"})

    detail = await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")

    assert detail == "kept: the model moved to anthropic/claude-opus-5-5 since the sign-in"
    assert applied == []


class _AccessSession(FakeSession):
    """A session with a canonical store and a real model spec — the claim's inputs.

    ``FakeSession`` is deliberately storeless, so the serve-side publication
    early-outs on it and every other test in this file keeps its shape; this
    subclass is the one double that stands in for a real runtime session, whose
    ``_frontend_state_store`` is what carries the claim.
    """

    def __init__(self, label: str = "radient/auto") -> None:
        super().__init__()
        from local_operator.session.frontend_state import (
            FrontendSessionState,
            FrontendStateStore,
        )

        self._frontend_state_store = FrontendStateStore(
            FrontendSessionState(session_id="sess-1", epoch="e1")
        )
        provider, _, model_id = label.partition("/")
        self.model = ModelSpec(provider=provider, model_id=model_id)
        self.model_label = label
        self.effective_model_label = label

    def set_model(self, spec: Any, *, explicit: bool = False) -> None:
        self.model = spec
        self.model_label = f"{spec.provider}/{spec.model_id}"
        self.effective_model_label = self.model_label


def _store_with_credential(root: Path, provider: str = "deepseek") -> None:
    """A real credential store under ``root`` holding one api_key row."""
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(root / "auth.db", config_dir=root)
    store.upsert_credential(provider, {"type": "api_key", "source": "login", "key": "probe"})
    store.close()


@pytest.mark.asyncio
async def test_the_serve_side_publishes_the_access_claim_at_open_and_on_switch(
    tmp_path: Path,
) -> None:
    """A desktop-only session carries the band's claim: at open, and after a switch.

    The TUI host publishes ``model_access`` from its own controller, so a
    session served by a runtime — no TUI anywhere — had no claim at all and the
    band's "not signed in to <provider>" sentence could never render. The
    serve-side host publishes the same claim from its own store on the two
    edges that change it: taking the session up, and every model switch (the
    re-home lands through ``set_model_effort`` too).
    """
    root = tmp_path / "cfg"
    root.mkdir()
    _store_with_credential(root)

    session = _AccessSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp", config_dir=root)

    opened = session._frontend_state_store.state.model_access
    assert opened is not None
    assert (opened.state, opened.provider, opened.label) == ("signed_out", "radient", "Radient")

    await handle.set_model_effort("deepseek", "deepseek-flash", None)
    switched = session._frontend_state_store.state.model_access
    assert switched is not None
    assert (switched.state, switched.provider, switched.label) == (
        "ok",
        "deepseek",
        "DeepSeek",
    )

    # THE TUI PARITY, for the same store: the serve-side predicate is the
    # controller call the TUI publishes from, and the claim is the SHARED
    # builder's answer over it — one computation, two hosts.
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController
    from local_operator.providers.model_access import usable_providers_here
    from local_operator.session.frontend_state import model_access_claim

    store = AuthStore(root / "auth.db", config_dir=root)
    try:
        tui_usable = ProviderController(store, root).usable_providers()
    finally:
        store.close()
    assert usable_providers_here(config_dir=root) == tui_usable
    tui_ok = model_access_claim("deepseek/deepseek-flash", tui_usable)
    tui_out = model_access_claim("radient/auto", tui_usable)
    assert tui_ok is not None and tui_ok.state == switched.state
    assert tui_out is not None and tui_out.state == opened.state


@pytest.mark.asyncio
async def test_an_unreadable_store_clears_the_claim_rather_than_leaving_a_stale_ok(
    tmp_path: Path,
) -> None:
    """A stale ``ok`` is the one wrong answer a reader cannot detect.

    When the store cannot be read the host says nothing (``None``, the absent
    field) rather than keeping the previous claim: "could not check" must not
    re-present as "you are signed in".
    """
    root = tmp_path / "cfg"
    root.mkdir()
    _store_with_credential(root)

    session = _AccessSession("deepseek/deepseek-flash")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp", config_dir=root)
    claim = session._frontend_state_store.state.model_access
    assert claim is not None and claim.state == "ok"

    # The store becomes unreadable: the db file is replaced by a directory, the
    # shape both the picker and the re-home degrade on (D18's sibling case).
    (root / "auth.db").unlink()
    (root / "auth.db").mkdir()

    handle.publish_model_access()

    assert session._frontend_state_store.state.model_access is None


@pytest.mark.asyncio
async def test_a_pick_during_the_credential_read_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compare re-runs immediately before the switch (review MINOR-1).

    The entry checks run, then the credential read awaits the loop — and the
    dispatch chain serializes same-connection frames, so a pick landing inside
    that window comes from ANOTHER connection (a phone or peer) or from a turn
    starting. Either way the user's decision is live and the CAS exists to lose
    to it, so the pre-switch re-read must catch it.
    """
    from local_operator.providers import model_access

    handle, session, applied, _recorder = _rehome_handle(monkeypatch)

    def _pick_then_read(**kwargs: Any) -> set[str]:
        # Exactly the window the reviewer reproduced: the set arrives, but a
        # selection landed while it was being read.
        session.model_label = "openai/gpt-6-astra"
        return {"deepseek"}

    monkeypatch.setattr(model_access, "credentialed_chat_providers_here", _pick_then_read)

    detail = await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")

    assert detail == "kept: the model moved to openai/gpt-6-astra since the sign-in"
    assert applied == []
    assert session.model_label == "openai/gpt-6-astra"


@pytest.mark.asyncio
async def test_rehome_never_cuts_across_live_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleeping on a stranded model is not a defect: switching mid-turn is.

    And the refusal is SPOKEN (UX review U1): the caller only counts the word,
    so the conversation itself gets the one sentence naming what happens next —
    said once per attempt, for either flavour of busy (a turn, or subagents
    still running).
    """
    handle, session, applied, recorder = _rehome_handle(monkeypatch)
    _patch_access(monkeypatch, {"deepseek"})

    session.is_streaming = True
    assert (
        await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")
        == REHOME_BUSY_REPLY
    )

    session.is_streaming = False
    session.running_children = 2
    assert (
        await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")
        == REHOME_BUSY_REPLY
    )
    await _settle_notices(handle)
    assert (
        recorder.notices()
        == [
            "This conversation stays on radient/auto until the current turn ends — "
            "/model switches it now.",
        ]
        * 2
    )
    assert applied == []


@pytest.mark.asyncio
async def test_rehome_refuses_when_either_side_of_the_pair_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both re-checks on the owner: the old provider signed back in, or the new one
    is not signed in HERE (the desktop runs under its own config root)."""
    handle, session, applied, _recorder = _rehome_handle(monkeypatch)

    _patch_access(monkeypatch, {"deepseek", "radient"})
    assert (
        await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")
        == "kept: radient is signed in again"
    )

    _patch_access(monkeypatch, {"openai"})
    assert (
        await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")
        == "kept: deepseek is not signed in on this device"
    )

    _patch_access(monkeypatch, None)
    assert (
        await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")
        == "kept: the credential store could not be read"
    )
    assert applied == []


@pytest.mark.asyncio
async def test_rehome_reports_a_switch_that_did_not_take(monkeypatch: pytest.MonkeyPatch) -> None:
    """The read-back decides, never the setter's own return (the ``set_model``
    contract: it assigns the spec before its journal writes, so a later raise can
    still leave a switch in force — and a silent no-op must not read as one)."""
    handle, session, applied, recorder = _rehome_handle(monkeypatch)
    _patch_access(monkeypatch, {"deepseek"})

    def _set_model(spec: Any, *, explicit: bool = False) -> None:
        # A host that swallowed the switch: the selection does not move.
        return None

    monkeypatch.setattr(session, "set_model", _set_model, raising=False)

    detail = await handle.rehome_if_current("radient/auto", "deepseek", "deepseek-flash")

    assert detail == "kept: the switch did not take (still radient/auto)"
    assert session.model_label == "radient/auto"

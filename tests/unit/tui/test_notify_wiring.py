"""Notifications, as the REAL app fires them.

``test_notify.py`` pins the notifier's own contract in isolation. This file
asserts the half that only exists in ``OperatorApp``: which of its handlers
call the notifier, with what, and — the part a unit test of the notifier
cannot see at all — how many times.

Three things are checked here because each has a failure mode that a passing
notifier unit test would not catch:

- **A parked turn notifies once, not once per repaint.** The waiting state is
  re-derived by ``_refresh_working_activity``, which runs on every event that
  moves the turn. Firing there without the edge latch sends a toast per
  repaint, which is the classic way a notification feature becomes something
  users disable.
- **A delegating turn does not report a false finish.** The ``task`` tool
  returns at registration, so the parent's ``agent_end`` lands while children
  work; the app must pass the live child count through.
- **A stopped turn says nothing.** The user pressed the key themselves a
  moment ago.

The app is driven through ``run_test`` and its real handlers rather than by
calling the notifier directly — a test that poked the notifier would pass with
the wiring connected to nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.events import TurnEnded
from local_operator.tui.widgets.approval import ApprovalBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class RecordingNotifier:
    """Stands in for ``Notifier``, recording the calls the app makes.

    Deliberately NOT a mock of the delivery path: what is under test here is
    the app's policy (which event, how often, with which count), and the wire
    is already pinned in ``test_notify.py``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.labels: list[str] = []
        #: Mirrors the real notifier's focus gate, which is what makes the
        #: DELIVERY-vs-derivation distinction observable here: a suppressed
        #: toast must return False so the app leaves its edge armed.
        self.focused = False

    def set_label(self, label: str) -> None:
        self.labels.append(label)

    def set_focused(self, focused: bool) -> None:
        self.focused = focused
        self.calls.append(("focus", focused))

    def notify_turn_complete(self, *, running_children: int) -> bool:
        self.calls.append(("complete", running_children))
        return running_children == 0 and not self.focused

    def notify_waiting(self, kind: str) -> bool:
        if self.focused:
            return False
        self.calls.append((kind, None))
        return True

    def notify_error(self) -> bool:
        self.calls.append(("error", None))
        return not self.focused

    @property
    def kinds(self) -> list[str]:
        """Just the notification kinds, in order — focus changes excluded."""
        return [kind for kind, _ in self.calls if kind != "focus"]


class JobsSession(FakeSession):
    """A fake whose ``jobs`` manager reports a settable set of live children.

    ``queued_tasks`` and ``running_bash`` exist because the completion gate has
    to see work the STATUS BAND deliberately hides: a child parked at the
    capacity gate is registered ``status="running", queued=True`` and has not
    started at all, and a backgrounded bash job also re-enters the conversation
    when it settles.
    """

    def __init__(
        self,
        running_tasks: int = 0,
        queued_tasks: int = 0,
        running_bash: int = 0,
    ) -> None:
        super().__init__()
        self.running_tasks = running_tasks
        self.queued_tasks = queued_tasks
        self.running_bash = running_bash
        session = self

        class _Manager:
            def list(self) -> list[Any]:
                # Shaped exactly like `AsyncJob` where the counts read it (see
                # `harness/jobs.py`).
                from types import SimpleNamespace

                jobs = [
                    SimpleNamespace(status="running", type="task", queued=False)
                    for _ in range(session.running_tasks)
                ]
                jobs += [
                    SimpleNamespace(status="running", type="task", queued=True)
                    for _ in range(session.queued_tasks)
                ]
                jobs += [
                    SimpleNamespace(status="running", type="bash", queued=False)
                    for _ in range(session.running_bash)
                ]
                return jobs

        self.jobs = _Manager()


async def _boot(pilot: Any, app: OperatorApp) -> None:
    """Settle until the session worker has actually attached the session.

    NOT optional, and not a style choice. The app paints before it awaits the
    session (a worker builds it), so a single `pause()` leaves `app._session`
    None — and every subagent-count assertion here reads the job manager
    THROUGH that session. An unbooted app reports 0 live children whatever the
    fake was staged with, so the "children are still running" test passed for
    the wrong reason until this landed: it was asserting against a session that
    did not exist yet. Mirrors `_boot` in `test_approvals_ux.py`.
    """
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("the session worker never attached a session")


async def _app_with_notifier(
    session: FakeSession,
) -> tuple[OperatorApp, RecordingNotifier]:
    """An app whose notifier is the recorder (the real one needs a terminal)."""
    app = OperatorApp(lambda: _factory(session))
    return app, RecordingNotifier()


@pytest.mark.asyncio
async def test_a_completed_turn_notifies_with_no_children_running() -> None:
    session = JobsSession(running_tasks=0)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert notifier.kinds == ["complete"]
    assert ("complete", 0) in notifier.calls


@pytest.mark.asyncio
async def test_a_turn_that_ends_while_children_run_passes_the_live_count() -> None:
    """The parent's `agent_end` is not the task finishing when it delegated:
    `task` returns at registration, so the children are still working."""
    session = JobsSession(running_tasks=3)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 3)


@pytest.mark.asyncio
async def test_an_aborted_turn_notifies_nothing() -> None:
    """The user pressed Ctrl+C or Esc a moment ago; telling them their own stop
    worked is the definition of a notification nobody wants."""
    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=True, error=None))
        await pilot.pause()
    assert notifier.kinds == []


@pytest.mark.asyncio
async def test_a_failed_turn_notifies_the_error() -> None:
    """The case a user who walked away most needs pulled to their attention."""
    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error="provider refused"))
        await pilot.pause()
    assert notifier.kinds == ["error"]


@pytest.mark.asyncio
async def test_an_unanswered_approval_notifies_exactly_once() -> None:
    """The edge latch, which is the whole reason `_waiting_on_user` exists.

    `_refresh_working_activity` re-derives the waiting state on every event
    that moves the turn — a tool card settling behind the prompt, the working
    line's own re-derivation. Without the latch each of those sends another
    toast for the same unanswered question.
    """
    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app._approval = ApprovalBlock("bash", "rm -rf /tmp/x", on_answer=lambda _: None)
        for _ in range(5):
            app._refresh_working_activity()
        await pilot.pause()
    assert notifier.kinds == ["approval"]


@pytest.mark.asyncio
async def test_answering_rearms_the_notification_for_the_next_question() -> None:
    """The latch is an edge detector, not a once-per-session fuse: a turn that
    asks twice must notify twice."""
    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app._approval = ApprovalBlock("bash", "first", on_answer=lambda _: None)
        app._refresh_working_activity()
        app._approval = None  # answered
        app._refresh_working_activity()
        app._approval = ApprovalBlock("write", "second", on_answer=lambda _: None)
        app._refresh_working_activity()
        await pilot.pause()
    assert notifier.kinds == ["approval", "approval"]


@pytest.mark.asyncio
async def test_the_ask_picker_notifies_as_a_question_not_an_approval() -> None:
    """`ask` outranks approval exactly as it does in `_current_activity`: the
    picker is modal and drawn over the card, so it is what the user is actually
    being asked for. The two share one phase, so only the notification kind
    distinguishes them — and 'Waiting for approval' over a question the model
    asked would send the user looking for a tool prompt that is not there.
    """
    import asyncio

    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        # An unresolved future is exactly what `_current_activity` reads: the
        # `ask` tool parks on one for the life of the question.
        app._ask_pending = asyncio.get_running_loop().create_future()
        app._refresh_working_activity()
        await pilot.pause()
        app._ask_pending.set_result(None)
    assert notifier.kinds == ["ask"]


@pytest.mark.asyncio
async def test_the_toast_is_titled_with_the_conversation_name() -> None:
    """A user with five sessions open otherwise gets five identical toasts."""
    session = JobsSession()
    session.set_conversation_name("Fix quota reporting")
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert notifier.labels[-1] == "Fix quota reporting"


@pytest.mark.asyncio
async def test_focus_changes_reach_the_notifier() -> None:
    """`AppFocus`/`AppBlur` are what make "only when the user is elsewhere"
    true; a notifier that never heard about them would notify always."""
    from textual.events import AppBlur, AppFocus

    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_app_blur(AppBlur())
        app.on_app_focus(AppFocus())
        await pilot.pause()
    assert notifier.calls == [("focus", False), ("focus", True)]


@pytest.mark.asyncio
async def test_an_app_without_a_notifier_still_ends_turns() -> None:
    """The headless/disabled path: every call site funnels through `_notify`,
    which must be inert rather than absent."""
    session = JobsSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        assert app._notifier is None  # no terminal under the test driver
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        app._refresh_working_activity()
        await pilot.pause()


# -- round 1 review: the wiring fixes ----------------------------------------


@pytest.mark.asyncio
async def test_a_question_raised_while_focused_is_told_when_the_user_looks_away() -> None:
    """The feature's primary use case, and it was inverted.

    The ordinary sequence is: start a turn, watch it for a few seconds, tab
    away. That raises the approval WHILE focused, where the toast is correctly
    suppressed — but latching the derived state there consumed the edge on a
    notification nobody received, so the question was never announced
    afterwards. Indefinitely: the latch re-arms only by answering the question
    the user does not know exists.
    """
    from textual.events import AppBlur

    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        notifier.focused = True
        app._approval = ApprovalBlock("bash", "rm -rf /tmp/x", on_answer=lambda _: None)
        app._refresh_working_activity()
        assert notifier.kinds == []  # suppressed: the user is watching
        app.on_app_blur(AppBlur())
        await pilot.pause()
    assert notifier.kinds == ["approval"]


@pytest.mark.asyncio
async def test_a_child_parked_at_the_capacity_gate_still_blocks_completion() -> None:
    """`run_subagent` registers with `queued=True` when the manager is full, so
    the child has not merely failed to finish — it has not STARTED. The status
    band's running count excludes it by design, which made the completion gate
    fire over delegated work that had yet to run."""
    session = JobsSession(running_tasks=0, queued_tasks=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 1)


@pytest.mark.asyncio
async def test_a_backgrounded_bash_job_does_not_produce_a_second_toast() -> None:
    """Both job types re-enter the conversation as a fresh turn when they
    settle (`Session._on_job_completed` delivers `task` AND `bash`), so counting
    only `task` announced the turn twice for one event."""
    session = JobsSession(running_bash=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 1)


@pytest.mark.asyncio
async def test_a_suppressed_completion_is_delivered_when_the_last_child_settles() -> None:
    """The "nothing is lost" guarantee, made true.

    It relied on the settled job re-entering as a notifiable turn, which
    `Session._on_job_completed` does not promise: it returns early for a
    cancelled, consumed, nested or mid-stream job, and the manager's cancel
    branch never delivers at all. So the app remembers the suppressed finish
    and flushes it when the delegated work is actually over.
    """
    from local_operator.tui.events import SubagentEnded

    session = JobsSession(running_tasks=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert notifier.kinds == ["complete"]  # suppressed (returned False)
        # The child is CANCELLED, which produces no re-entering turn at all.
        session.running_tasks = 0
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="child", status="cancelled"))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 0)


@pytest.mark.asyncio
async def test_siblings_still_working_keep_the_deferred_completion_waiting() -> None:
    """Only the LAST child to settle announces the finish."""
    from local_operator.tui.events import SubagentEnded

    session = JobsSession(running_tasks=2)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        before = len(notifier.calls)
        session.running_tasks = 1  # one settled, one still going
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="a", status="completed"))
        await pilot.pause()
        assert len(notifier.calls) == before  # nothing yet
        session.running_tasks = 0
        app.on_subagent_ended(SubagentEnded(job_id="j2", label="b", status="completed"))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 0)


@pytest.mark.asyncio
async def test_an_ask_raised_over_a_live_approval_is_still_announced() -> None:
    """`ask` and approval share one activity phase, so a bool latch could not
    see the transition — and the two are worth telling apart, since "waiting
    for approval" over a question the model asked sends the user hunting for a
    tool prompt that is not there."""
    import asyncio

    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app._approval = ApprovalBlock("bash", "x", on_answer=lambda _: None)
        app._refresh_working_activity()
        app._ask_pending = asyncio.get_running_loop().create_future()
        app._refresh_working_activity()
        await pilot.pause()
        app._ask_pending.set_result(None)
    assert notifier.kinds == ["approval", "ask"]


# -- self-review: the remediation's own hazards ------------------------------


@pytest.mark.asyncio
async def test_a_new_turn_drops_a_deferred_completion() -> None:
    """Found while probing my own round-1 remediation.

    A completion suppressed because children were running is owned by the turn
    that finished. If a NEW turn starts before the last child settles, the
    child's settle announced "task complete" while the agent was mid-stream —
    a finish reported for a session that is visibly working. The new turn will
    raise its own completion when it settles, so the stale one is dropped.
    """
    from local_operator.tui.events import SubagentEnded, TurnStarted

    session = JobsSession(running_tasks=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert app._completion_deferred is True
        app.on_turn_started(TurnStarted())
        assert app._completion_deferred is False
        before = len(notifier.calls)
        session.running_tasks = 0
        app.on_subagent_ended(SubagentEnded(job_id="j", label="c", status="completed"))
        await pilot.pause()
    assert len(notifier.calls) == before  # nothing stale fired


@pytest.mark.asyncio
async def test_blurring_twice_does_not_announce_one_question_twice() -> None:
    """`_flush_pending_question` runs on every blur, and a user alt-tabbing
    back and forth must not be told about the same unanswered question each
    time — the latch is what makes the flush idempotent."""
    from textual.events import AppBlur, AppFocus

    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app._approval = ApprovalBlock("bash", "x", on_answer=lambda _: None)
        app._refresh_working_activity()
        for _ in range(3):
            app.on_app_blur(AppBlur())
            app.on_app_focus(AppFocus())
        await pilot.pause()
    assert notifier.kinds == ["approval"]


@pytest.mark.asyncio
async def test_blurring_an_idle_session_announces_nothing() -> None:
    """The flush must fire only for a turn actually parked on the user."""
    from textual.events import AppBlur

    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_app_blur(AppBlur())
        await pilot.pause()
    assert notifier.kinds == []

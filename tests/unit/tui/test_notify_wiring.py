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

# `ApprovalPrompt`, not `ApprovalBlock`: the live question is the docked card
# that `_approval` holds, and the block is now only the transcript receipt
# written after an answer. These tests stand a PENDING approval into that slot
# to drive the notifier, so they need the type that can be pending.
from local_operator.tui.widgets.approval import ApprovalPrompt
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

                # Ids match the `SubagentEnded.job_id`s the tests post, so a
                # test can reproduce the REAL manager's ordering: the child's
                # end event arrives while its own row is still `running`,
                # because the manager settles it only after the coroutine that
                # emitted the event returns (`harness/subagent.py`).
                jobs = [
                    SimpleNamespace(status="running", type="task", queued=False, id=f"j{i + 1}")
                    for i in range(session.running_tasks)
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
        app._approval = ApprovalPrompt("bash", "rm -rf /tmp/x", on_answer=lambda _: None)
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
        app._approval = ApprovalPrompt("bash", "first", on_answer=lambda _: None)
        app._refresh_working_activity()
        app._approval = None  # answered
        app._refresh_working_activity()
        app._approval = ApprovalPrompt("write", "second", on_answer=lambda _: None)
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
async def test_the_toast_uses_the_current_provisional_session_title() -> None:
    """Naming runs concurrently with the opening turn, so the display title is
    often still provisional when completion or an ask edge arrives. Falling
    through to the cwd here produced notifications titled ``tmp`` while the
    status band and terminal tab already showed the useful session title."""
    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app._provisional_name = "Improve notification context"
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert session.conversation_name == ""
    assert notifier.labels[-1] == "Improve notification context"


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
        app._approval = ApprovalPrompt("bash", "rm -rf /tmp/x", on_answer=lambda _: None)
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
async def test_a_backgrounded_bash_job_does_not_hold_the_completion() -> None:
    """A turn that backgrounded a shell command IS finished.

    Round 1 read the double-toast symptom as "count `bash` too"; round 2 showed
    that inverts the feature (see the B4/B5 tests below), because a background
    job may outlive the session and emits no `SubagentEnded`. The turn that
    spawned it is genuinely over — the command is a side effect the user
    started deliberately and can watch in its own tool card — so the completion
    fires, and the later turn that reacts to the job's output is a separate
    finish worth its own notification.
    """
    session = JobsSession(running_bash=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 0)


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
        # `j2` settles; `j1` is still listed and still running. The exclusion
        # must drop only the job whose end this is.
        session.running_tasks = 1
        app.on_subagent_ended(SubagentEnded(job_id="j2", label="b", status="completed"))
        await pilot.pause()
        assert len(notifier.calls) == before  # nothing yet
        session.running_tasks = 0
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="a", status="completed"))
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
        app._approval = ApprovalPrompt("bash", "x", on_answer=lambda _: None)
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
        app._approval = ApprovalPrompt("bash", "x", on_answer=lambda _: None)
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


@pytest.mark.asyncio
async def test_a_question_after_an_aborted_turn_is_still_announced() -> None:
    """The waiting latch must not survive a turn boundary.

    `_refresh_working_activity` clears the latch when a question is ANSWERED,
    but an aborted turn never makes that transition: Esc denies the parked
    approval and ends the turn with the latch still reading `approval`. The
    next turn's question was therefore not a change, and was silently never
    announced — a stale latch costing exactly the notification this feature
    exists to send. Caught by the round-2 reviewer's probing, reproduced, and
    pinned here.
    """
    from local_operator.tui.events import TurnStarted

    session = JobsSession()
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app._approval = ApprovalPrompt("bash", "first", on_answer=lambda _: None)
        app._refresh_working_activity()
        # An abort ends the turn without the answered-transition that clears it.
        app.on_turn_ended(TurnEnded(aborted=True, error=None))
        await pilot.pause()
        app.on_turn_started(TurnStarted())
        app._approval = ApprovalPrompt("write", "second", on_answer=lambda _: None)
        app._refresh_working_activity()
        await pilot.pause()
    assert notifier.kinds == ["approval", "approval"]


# -- round 2 review: the B2/M2 fix's own regressions --------------------------


@pytest.mark.asyncio
async def test_a_long_lived_background_job_does_not_disable_notifications() -> None:
    """Round 2, B4. Counting `bash` in the completion gate looked symmetrical
    with `task` and silently switched the feature off: a backgrounded
    `npm run dev` never settles, so EVERY later completion in the session was
    suppressed. The turn that spawned it genuinely is over — a background
    command is a side effect the user started deliberately and can watch in its
    own tool card."""
    from local_operator.tui.events import TurnStarted

    session = JobsSession(running_bash=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        for _ in range(3):
            app.on_turn_started(TurnStarted())
            await pilot.pause()
            app.on_turn_ended(TurnEnded(aborted=False, error=None))
            await pilot.pause()
    delivered = [call for call in notifier.calls if call == ("complete", 0)]
    assert len(delivered) == 3


@pytest.mark.asyncio
async def test_a_background_job_cannot_strand_a_deferred_completion() -> None:
    """Round 2, B5. `SubagentEnded` is posted only for `task` children, so a
    `bash` job counted by the gate could CAUSE a deferral that nothing was able
    to flush — losing the completion for good, on exactly the paths the
    deferral was introduced to protect."""
    session = JobsSession(running_bash=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
    assert app._completion_deferred is False
    assert notifier.calls[-1] == ("complete", 0)


@pytest.mark.asyncio
async def test_a_completion_watched_by_the_user_is_not_replayed_on_the_blur() -> None:
    """Round 3, M7. The completion debt must NOT outlive the work.

    Round 2's M6 fix treated a suppressed completion like a suppressed
    question, and the two are not alike. A question outlives the moment it was
    asked — it is still unanswered later, so a suppressed one stays owed (B1).
    A completion is an instant: once the work finished in front of the user
    there is nothing left to tell them. Retaining the debt meant the ONLY toast
    the blur flush could ever deliver was one whose user had already watched it
    land, announced whenever they next alt-tabbed away, unboundedly later.

    So the flag is cleared once the work is over, delivered or not.
    """
    from textual.events import AppBlur

    from local_operator.tui.events import SubagentEnded

    session = JobsSession(running_tasks=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        # The child settles while the user is watching: nothing to announce.
        notifier.focused = True
        session.running_tasks = 0
        app.on_subagent_ended(SubagentEnded(job_id="j", label="c", status="completed"))
        await pilot.pause()
        assert app._completion_deferred is False  # the debt is settled, not held
        notifier.focused = False
        before = len(notifier.calls)
        app.on_app_blur(AppBlur())
        await pilot.pause()
    assert notifier.calls[before:] == [("focus", False)]  # no stale completion


@pytest.mark.asyncio
async def test_a_deferred_completion_is_not_announced_over_live_children() -> None:
    """Round 3, N5. The outstanding-jobs guard, which no test pinned.

    Without it, one child settling while siblings still work announces a
    finish over live subagents — the B2 false finish, reintroduced from the
    flush path instead of the turn-end path.
    """
    from local_operator.tui.events import SubagentEnded

    session = JobsSession(running_tasks=2)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        before = len(notifier.calls)
        # `j2` settles while `j1` is still running: excluding the settling job
        # must NOT hide the sibling that is genuinely still working.
        session.running_tasks = 1
        app.on_subagent_ended(SubagentEnded(job_id="j2", label="a", status="completed"))
        await pilot.pause()
    assert notifier.calls[before:] == []
    assert app._completion_deferred is True  # still owed, correctly


@pytest.mark.asyncio
async def test_the_last_child_still_counts_itself_when_its_end_arrives() -> None:
    """Round 4, M8. The ordering the real manager actually produces.

    `SubagentEndEvent` is emitted from INSIDE the job coroutine, and the
    manager flips `job.status` to settled only once that coroutine returns —
    with an awaited transcript flush in between. So the handler routinely runs
    while the ending child's own row still reads `running`, and a guard that
    counts it concludes work is outstanding and drops the completion. Worse,
    the flag then latches, swallowing every later completion in the session.

    Every other test here hand-sets the count to 0 before posting the event,
    which hard-codes an ordering the real manager does not guarantee — so this
    one deliberately leaves the child listed as running.
    """
    from local_operator.tui.events import SubagentEnded

    session = JobsSession(running_tasks=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert app._completion_deferred is True
        # NOT cleared: the manager has not settled it yet, exactly as in
        # production when the event is drained inside that window.
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="c", status="completed"))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 0)
    assert app._completion_deferred is False


@pytest.mark.asyncio
async def test_a_batch_of_children_settling_together_still_notifies_once() -> None:
    """Round 5, B6. The `task(tasks=[...])` batch, which is the common shape.

    A batch settles its children inside one another's teardown windows, so
    every end event arrives while EVERY one of those rows still reads
    `running` (the manager settles a job only after the coroutine that emitted
    its end event returns). Excluding only the job being handled left each
    handler seeing its siblings as outstanding: nobody was last, nobody
    delivered, and the deferred flag latched — swallowing every later
    completion in the session too.

    Every other test here exercises exactly one child in that window, which is
    why this survived the previous round. Note that `running_tasks` is NOT
    decremented: all three rows stay `running` for the whole exchange.
    """
    from local_operator.tui.events import SubagentEnded

    session = JobsSession(running_tasks=3)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert app._completion_deferred is True
        for job_id in ("j1", "j2", "j3"):
            app.on_subagent_ended(SubagentEnded(job_id=job_id, label=job_id, status="completed"))
        await pilot.pause()
    assert [call for call in notifier.calls if call == ("complete", 0)] == [("complete", 0)]
    assert app._completion_deferred is False


@pytest.mark.asyncio
async def test_a_later_batch_is_not_masked_by_the_previous_ones_ids() -> None:
    """The handled-set must not outlive the deferral it serves.

    Ids held past a completion would let the NEXT turn's guard skip a child
    that really is running, which is the B2 false finish arriving through the
    exclusion instead of the count.
    """
    from local_operator.tui.events import SubagentEnded, TurnStarted

    session = JobsSession(running_tasks=1)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="a", status="completed"))
        await pilot.pause()
        assert app._settled_child_ids == set()
        # A new turn delegates again, and `j1` is a DIFFERENT live child now.
        app.on_turn_started(TurnStarted())
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        before = len(notifier.calls)
        session.running_tasks = 2
        app.on_subagent_ended(SubagentEnded(job_id="j2", label="b", status="completed"))
        await pilot.pause()
    assert notifier.calls[before:] == []  # `j1` still running: correctly silent


@pytest.mark.asyncio
async def test_the_handled_set_is_empty_whenever_a_deferral_arms() -> None:
    """Round 6, N6. The safety invariant behind `_settled_child_ids`.

    The set is cleared in four places — on delivery, on a fresh deferral, at a
    turn boundary and on session reload — and those clears mutually mask under
    mutation, so removing any one of them left the suite green while the shipped
    code stayed correct only by redundancy. What actually matters is the
    invariant they collectively maintain: **an arming deferral must never carry
    ids from an earlier one**, or the next batch's guard skips a child that is
    genuinely still running (the B2 false finish, arriving through the
    exclusion instead of through the count).

    Asserted directly, across the interleavings that reach the arming point by
    different routes, so a future editor who removes a clear breaks a test that
    names the property rather than one that happens to notice.
    """
    from local_operator.tui.events import SubagentEnded, TurnStarted

    session = JobsSession(running_tasks=2)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]

        # Route 1: deferral armed by a turn ending with children live.
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert app._completion_deferred is True
        assert app._settled_child_ids == set()

        # Handle one child, then arm a FRESH deferral from a new turn: the
        # previous batch's id must not survive into it.
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="a", status="completed"))
        await pilot.pause()
        app.on_turn_started(TurnStarted())
        assert app._settled_child_ids == set()
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert app._completion_deferred is True
        assert app._settled_child_ids == set()

        # Route 2: through delivery. Drain the batch, then arm again.
        session.running_tasks = 2
        for job_id in ("j1", "j2"):
            app.on_subagent_ended(SubagentEnded(job_id=job_id, label=job_id, status="completed"))
        await pilot.pause()
        assert app._completion_deferred is False
        assert app._settled_child_ids == set()
        app.on_turn_started(TurnStarted())
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert app._settled_child_ids == set()


@pytest.mark.asyncio
async def test_a_deferral_armed_without_a_turn_boundary_starts_empty() -> None:
    """Round 6, N6, second half: the clear on the ARMING path specifically.

    The four clear points mask one another — a deferral usually re-arms after a
    turn boundary that has already emptied the set, so removing the clear on
    the arming path alone leaves every other test green. This reaches the
    arming point WITHOUT an intervening boundary (a job result re-enters and
    settles the turn directly), which is the one route where the arming clear
    is the only thing standing between a stale id and a false finish.
    """
    from local_operator.tui.events import SubagentEnded

    session = JobsSession(running_tasks=2)
    app, notifier = await _app_with_notifier(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        # `j1` is handled; its id is now in the set and the manager has not
        # settled it, so the row is still listed.
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="a", status="completed"))
        await pilot.pause()
        assert app._settled_child_ids == {"j1"}
        # Another turn settles with both children still live and NO boundary in
        # between. If `j1` survived into this deferral, the guard would count
        # only `j2`, and `j2`'s end would then announce a finish over a child
        # that is still running.
        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        assert app._settled_child_ids == set()
        before = len(notifier.calls)
        app.on_subagent_ended(SubagentEnded(job_id="j2", label="b", status="completed"))
        await pilot.pause()
    assert notifier.calls[before:] == []  # `j1` still running: correctly silent

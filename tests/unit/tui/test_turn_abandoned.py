"""Turns that end without an ``agent_end``, and what the app still owes them.

THE DEFECT, in one sentence: a turn's status band was retired by a `finally`
that always runs, while the working line and EVERY terminal side effect the
turn owed the user — the notification most of all — hung off `on_turn_ended`,
which fires only when an `agent_end` reaches the controller. So a turn that
died on the way to one left a session that looked finished (band idle, composer
free) with `thinking 15m42s` ticking forever, and told the user nothing at all.
The reported incident: an MCP authorization failure killed the turn, two
subagents had already settled, and the operator found the work done only by
scanning their session list by hand.

The spinner was cosmetic. The swallowed notification is what cost time, and it
is the half most of these tests are about.

The seam is now: ONE `_finalize_turn`, reached either from the authoritative
`agent_end` path or from a `TurnAbandoned` the turn worker posts from its own
`finally`. This file pins one test per route a turn can die by, plus the four
things the obvious version of this fix gets wrong — a fallback that runs ahead
of the real end and double-notifies, a follower's mid-turn `prompt()` return
read as a finish, a `CancelledError` reported as a completion, and a deferral
armed for children that have already settled and will never report again.

Assertions are on the REAL notifier seam (`RecordingNotifier`, as
`test_notify_wiring.py` drives it) and on the mounted widget, never on whether
some internal method was called.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from local_operator.harness.types import ToolExecutionStartEvent
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    CompactionStarted,
    SubagentEnded,
    ToolStarted,
    TurnAbandoned,
    TurnBoundaryEnd,
    TurnEnded,
    TurnStarted,
)
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptView,
    WorkingBlock,
)

from .test_app_pilot import FakeSession, _factory
from .test_notify_wiring import JobsSession, RecordingNotifier, _boot


def _working(app: OperatorApp) -> WorkingBlock | None:
    """The mounted working line, or None when the turn has settled."""
    lines = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, WorkingBlock)]
    assert len(lines) <= 1, "D25: there is exactly ONE aggregate working line"
    return lines[0] if lines else None


def _started(tool_call_id: str, tool_name: str, **args: Any) -> ToolStarted:
    return ToolStarted(
        ToolExecutionStartEvent(tool_call_id=tool_call_id, tool_name=tool_name, args=args)
    )


async def _armed(pilot: Any, app: OperatorApp, notifier: RecordingNotifier) -> None:
    """Attach the recorder and open a turn, the way ``agent_start`` does."""
    app._notifier = notifier  # type: ignore[assignment]
    app.post_message(TurnStarted())
    await pilot.pause()


class ExplodingSession(JobsSession):
    """Route (a): ``prompt()`` raises, so no ``agent_end`` is ever emitted.

    This is the reported incident, reproduced through the app's own turn
    worker rather than by posting a message the worker would have posted.
    """

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        raise RuntimeError("provider exploded mid-turn")


class SilentSession(JobsSession):
    """Routes (d)/(e): ``prompt()`` returns CLEANLY but no ``TurnEnded`` lands.

    Stands in for a subscriber raising inside ``Session._emit`` (which catches
    and logs every handler exception, so the event is simply never posted) or
    ``on_turn_ended`` raising partway through its own body. From the app's
    side the two are indistinguishable, and both are silent today.
    """

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        return None


class HangingFollowerSession(JobsSession):
    """Route (f): a FOLLOWER, whose ``prompt()`` returns while the turn runs.

    ``RemoteSession.prompt`` returns on the OWNER's acknowledgement — the owner
    acks as soon as the user's row is durable ("prompt admitted"), not when the
    turn ends — so the worker's `finally` fires mid-turn on every attached TUI.
    ``is_streaming`` stays True until the owner's `agent_end` arrives over the
    wire. On a REAL socket it is the SECOND of two independent guards to drop
    this fallback — the epoch check fires first, because the worker captures the
    epoch before the relayed `agent_start` bumps it. This fake holds the epoch
    still on purpose, so the `is_streaming` guard is the one under test here
    rather than merely the one that happens to run first.
    """

    #: What `RemoteSession` declares (`remote.py:203`), and the app reads it to
    #: know this worker's `error=None` means "the owner did not tell me" rather
    #: than "the turn succeeded". A follower fake without it models the wire
    #: ORDER while claiming the authority of an in-process session.
    is_remote = True

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        self.streaming = True
        return None


class OwnerEndRacingFollowerSession(JobsSession):
    """The follower interleaving that breaks a latch placed at the CALL SITE.

    `RemoteSession._on_wire_event` clears `_streaming` and only THEN emits the
    relayed `agent_end` (remote.py: the `AgentEndEvent` arm), and it runs on the
    socket read pump — a different task from Textual's message pump. So the
    owner's end can land inside the fallback's own post-to-dispatch window:
    guard 2 reads a just-cleared False, the fallback proceeds, and the real
    `TurnEnded` dispatches BEHIND it.

    This fake reproduces exactly that window. ``relay`` is armed with the app,
    and the relayed end is scheduled with ``call_soon`` rather than posted
    inline: the turn worker's `finally` runs synchronously when `prompt()`
    returns (nothing between them awaits), so a callback scheduled here lands
    just AFTER the fallback is queued and just BEFORE the pump drains — which
    is the interleaving, and the only one that puts the two messages in the
    queue in the order the socket produces.
    """

    #: See :class:`HangingFollowerSession`. Load-bearing here: it is what makes
    #: this fallback decline to announce an outcome it was never told, so the
    #: owner's real end behind it is what speaks.
    is_remote = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.relay: Callable[[], None] | None = None

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        # Cleared BEFORE returning, mirroring the wire handler's order — this is
        # what makes guard 2 read False for a turn whose real end is still in
        # flight.
        self.streaming = False
        if self.relay is not None:
            asyncio.get_running_loop().call_soon(self.relay)
        return None


@pytest.mark.asyncio
async def test_a_turn_whose_prompt_raises_is_retired_and_notified() -> None:
    """Route (a), the reported incident, end to end through the real worker.

    Both halves in one assertion set: the line goes (the visible symptom) AND
    exactly one error notification is delivered (the half that cost the
    operator an afternoon).
    """
    app = OperatorApp(lambda: _factory(ExplodingSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)
        assert _working(app) is not None

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None, "the line outlived the turn it describes"
        assert app._status is not None and not app._status._streaming
    assert notifier.kinds == ["error"]


@pytest.mark.asyncio
async def test_a_turn_that_returns_without_an_agent_end_still_completes() -> None:
    """Routes (d)/(e): the session thinks the turn finished; the event was lost.

    Deliberately NOT reported as an error. The session considers the work done
    and the missing event is a transport or handler gap, so claiming a failure
    would invent one the user cannot act on.
    """
    app = OperatorApp(lambda: _factory(SilentSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None
        assert app._status is not None and not app._status._streaming
    assert notifier.kinds == ["complete"]


@pytest.mark.asyncio
async def test_a_superseded_turns_fallback_never_touches_the_live_turn() -> None:
    """Route (b): turn A's fallback dispatches after turn B has already opened.

    Dropping the superseded turn's completion is DELIBERATE and matches what
    the controller's own supersede guard and `on_turn_started` already do — the
    user is told when the work they can see is over. What must not happen is
    the stale fallback tearing down turn B's working line, which is what an
    epoch-blind version of this handler does.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)
        stale_epoch = app._turn_epoch

        # Turn B opens before A's fallback is dispatched.
        app.post_message(TurnStarted())
        await pilot.pause()
        assert app._turn_epoch != stale_epoch

        app.post_message(TurnAbandoned(stale_epoch, aborted=False, error=None))
        await pilot.pause()

        assert _working(app) is not None, "the stale fallback retired the LIVE turn"
    assert notifier.kinds == [], "a superseded turn must not announce itself"


@pytest.mark.asyncio
async def test_a_follower_prompt_returning_mid_turn_is_a_no_op() -> None:
    """Route (f), and THE test that catches the obvious fix.

    A follower's `prompt()` returns on the owner's ACK, so the worker's
    `finally` runs while the turn is very much alive. Without the `is_streaming`
    guard every attached TUI would report a false finish the instant its prompt
    was accepted — and would lift the working line off a turn still streaming
    into it.
    """
    session = HangingFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert session.is_streaming, "the fake must still report a live turn"
        assert _working(app) is not None, "the follower's line was lifted mid-turn"
        assert notifier.kinds == [], "a follower's ACK is not a finish"

        # The owner's real end arrives over the wire; NOW it settles, once.
        session.streaming = False
        app.post_message(TurnEnded(aborted=False, error=None))
        await pilot.pause()
        assert _working(app) is None
    assert notifier.kinds == ["complete"]


@pytest.mark.asyncio
async def test_a_normal_turn_notifies_exactly_once() -> None:
    """The race the fallback MUST NOT lose: a real end, then the fallback.

    Posted in this exact order rather than left to the harness to produce,
    because this ordering is the one the FIFO guarantees for an in-process
    session: `prompt()` returns only after the pipeline has emitted `agent_end`
    synchronously, so the real `TurnEnded` is already queued when the worker's
    `finally` posts. A fallback that did its work inline — or that did not check
    the latch — would double-notify every normal turn in the product.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app.post_message(TurnEnded(aborted=False, error=None))
        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error=None))
        await pilot.pause()

        assert _working(app) is None
    assert notifier.kinds == ["complete"], "the fallback double-fired a normal turn"


@pytest.mark.asyncio
async def test_the_fallback_then_the_real_end_notifies_exactly_once() -> None:
    """THE REVERSE ORDER — and the one a call-site latch cannot guard.

    The test above pins fallback-after-real-end, which a check inside
    `on_turn_abandoned` does catch. This pins the other direction, and it is the
    direction that matters: `on_turn_ended` calls `_finalize_turn`
    UNCONDITIONALLY, so once the fallback has run the ladder, a real `TurnEnded`
    arriving behind it runs the whole ladder again. Measured at the revision
    that latched at the call site: `['complete', 'complete']`.

    That is why the latch lives on the shared path inside `_finalize_turn`.
    At-most-once is a property of the TURN, not of either mechanism, so the only
    correct place to assert it is the code both routes go through.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        # Fallback FIRST, real end behind it — the follower ordering below.
        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error=None))
        app.post_message(TurnEnded(aborted=False, error=None))
        await pilot.pause()
        await pilot.pause()

        assert _working(app) is None
    assert notifier.kinds == ["complete"], "the real end re-ran the notification ladder"


@pytest.mark.asyncio
async def test_an_aborted_turn_gets_one_interrupted_notice_in_either_order() -> None:
    """The same at-most-once property, on the notice the ABORT path prints.

    The notification ladder is not the only thing in the gated tail: an aborted
    turn with nothing in flight appends a standalone `interrupted` row. Running
    the tail twice produced two of them stacked in the transcript, which is the
    visible half of the same defect.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=True, error=None))
        app.post_message(TurnEnded(aborted=True, error=None))
        await pilot.pause()
        await pilot.pause()

        notices = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        assert len(notices) == 1, "the abort notice was printed twice"
    assert notifier.kinds == []


@pytest.mark.asyncio
async def test_a_follower_whose_owner_end_races_the_fallback_notifies_once() -> None:
    """The interleaving above, driven through the REAL turn worker.

    The two tests above post both messages by hand. This one lets
    `_start_turn`'s worker post the `TurnAbandoned` from its own `finally` —
    with `is_streaming` already cleared, exactly as the wire handler leaves it —
    and delivers the owner's relayed end into the window that opens behind it.

    This is the regression the PR would otherwise have INTRODUCED: `main` has no
    `TurnAbandoned` at all, so a follower cannot double-notify there today.
    """
    session = OwnerEndRacingFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        # The owner's end is relayed from inside `prompt()`'s return, so it is
        # queued behind the worker's own `TurnAbandoned` rather than ahead of it.
        def relay_owner_end() -> None:
            app.post_message(TurnEnded(aborted=False, error=None))

        session.relay = relay_owner_end

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None, "the turn never retired"
    assert notifier.kinds == ["complete"], "the follower double-notified its owner's turn"


@pytest.mark.asyncio
async def test_an_abandoned_turn_that_was_aborted_stays_silent() -> None:
    """A `CancelledError` is a `BaseException` and slides past `except Exception`.

    Without the explicit branch that classifies it, a Ctrl+C or a `/reload`
    would post `error=None` and this fallback would congratulate the user on
    completing a turn they had just killed — the one outcome here that actively
    breaks trust. The line still has to go.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=True, error=None))
        await pilot.pause()

        assert _working(app) is None
    assert notifier.kinds == []


@pytest.mark.asyncio
async def test_a_cancelled_worker_is_classified_as_an_abort_not_a_completion() -> None:
    """The same guarantee, driven through the REAL worker's exception path.

    The test above pins the handler's behaviour given `aborted=True`; this one
    pins that a cancelled `prompt()` actually produces it, which is the part the
    `except asyncio.CancelledError` branch exists for.
    """

    class CancellingSession(JobsSession):
        async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
            raise asyncio.CancelledError()

    app = OperatorApp(lambda: _factory(CancellingSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None, "a cancelled turn still leaves no spinner"
    assert notifier.kinds == [], "a turn the user killed must not report success"


@pytest.mark.asyncio
async def test_a_fallback_with_children_running_defers_its_completion() -> None:
    """The deferral arms exactly as a normal turn's does, and the last child
    flushes it through the existing path — no new deferral machinery."""
    session = JobsSession(running_tasks=1)
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error=None))
        await pilot.pause()
        assert app._completion_deferred is True
        assert notifier.calls[-1] == ("complete", 1)  # suppressed, not delivered

        session.running_tasks = 0
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="child", status="completed"))
        await pilot.pause()
    assert notifier.calls[-1] == ("complete", 0)


@pytest.mark.asyncio
async def test_a_fallback_whose_children_already_settled_notifies_immediately() -> None:
    """THE STRAND, and the expensive half of the reported defect.

    `_completion_deferred` is only ever set in the turn-end tail, and
    `on_subagent_ended` returns immediately while it is False. So when children
    settle BEFORE the parent turn dies, each of their end events is discarded,
    the turn then dies without `agent_end`, and NO further `SubagentEnded` will
    ever arrive — which is why merely ARMING the flag here would strand the
    completion permanently rather than fix it. It has to be EVALUATED: the
    children are not running, the count is 0, and the toast goes out now.

    This is the operator's incident exactly: two subagents already settled, one
    of them cancelled (which produces no re-entering turn at all), and no
    notification of any kind.
    """
    session = JobsSession(running_tasks=0)
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        # The children settled while the flag was False — discarded, as today.
        app.on_subagent_ended(SubagentEnded(job_id="j1", label="a", status="completed"))
        app.on_subagent_ended(SubagentEnded(job_id="j2", label="b", status="cancelled"))
        await pilot.pause()
        assert notifier.kinds == [], "nothing is recorded before the turn ends"

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error=None))
        await pilot.pause()

        assert app._completion_deferred is False, "an armed flag here strands forever"
    assert notifier.calls[-1] == ("complete", 0)


@pytest.mark.asyncio
async def test_a_held_goal_loop_still_owns_its_single_release_toast() -> None:
    """The canary for "the fallback goes THROUGH the ladder, not around it".

    `/loop <goal>` fires one toast when the judge says the goal is achieved.
    A fallback that notified directly would restore the per-turn notification
    fatigue that goal mode exists to remove — on an unbounded loop.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)
        app._loop_suppress_completion = True

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error=None))
        await pilot.pause()

        assert _working(app) is None, "the line is still retired; only the toast is held"
    assert notifier.kinds == []


@pytest.mark.asyncio
async def test_a_fallback_does_not_disturb_a_compactions_working_line() -> None:
    """`/compact` outside a turn owns the line, and no turn is open to retire.

    A manual compaction cannot overlap a turn (`compact_now` refuses while the
    session is streaming), so its line is the only one on screen and the latch
    is what keeps a stray fallback from lifting it.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._notifier = notifier  # type: ignore[assignment]

        app.post_message(CompactionStarted("manual"))
        await pilot.pause()
        assert _working(app) is not None
        assert app._compaction_owns_working_block is True

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error=None))
        await pilot.pause()

        assert _working(app) is not None, "the fallback lifted the compaction's line"
    assert notifier.kinds == []


@pytest.mark.asyncio
async def test_live_tool_cards_settle_when_a_turn_is_abandoned() -> None:
    """A turn that emits no `agent_end` emits no `turn_end` either.

    So nothing else reconciles its running rows: `on_turn_ended` never called
    `_retire_live_tool_cards` — `on_turn_boundary_end` did — and an abandoned
    turn reaches neither. Left alone the card animates forever beside a
    retired working line, which is the same defect one surface down.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()
        assert app._tool_cards, "the card must be live before the turn dies"
        card = app._tool_cards["c0"]

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error=None))
        await pilot.pause()

        assert not app._tool_cards, "a stranded live card outlived its turn"
        # Asserted on the row the user reads, which is the repo's convention
        # for this state (see `test_tool_card.py`): `⊘ interrupted`.
        row = card._build_row(80).plain
        assert "⊘" in row and "interrupted" in row, row


@pytest.mark.asyncio
async def test_an_abandoned_turn_with_live_cards_skips_the_standalone_notice() -> None:
    """`+=`, not `=`, on the interrupted count.

    The per-card `⊘ interrupted` mark is the more useful of the two because it
    names WHICH tool stopped, so the standalone notice is suppressed whenever
    anything was in flight. Clobbering the count with a bare assignment from
    `_retire_live_tool_cards` would resurrect the duplicate row this rule
    exists to remove.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=True, error=None))
        await pilot.pause()

        blocks = app.query_one(TranscriptView).blocks()
        notices = [b for b in blocks if isinstance(b, NoticeBlock) and "interrupted" in b.text()]
        assert notices == [], "the card already says it was interrupted"


@pytest.mark.asyncio
async def test_a_banked_interrupted_count_survives_the_fallback() -> None:
    """`+=`, NOT `=` — and this is the case where the two differ.

    On a turn that DID reach its boundary, `on_turn_boundary_end` has already
    banked the count and emptied the card dicts, so `_retire_live_tool_cards`
    now returns 0. A bare assignment would overwrite the banked figure with
    that 0, and the abort would regain the standalone "interrupted" notice the
    per-card marks exist to replace. `+=` adds nothing and preserves it.

    The previous test cannot catch this: there the boundary never ran, so both
    spellings happen to agree.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()

        # The boundary reconciles the orphaned card and BANKS the count.
        app.post_message(TurnBoundaryEnd())
        await pilot.pause()
        assert app._interrupted_cards == 1
        assert not app._tool_cards, "the boundary already emptied the ledger"

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=True, error=None))
        await pilot.pause()

        notices = [
            b
            for b in app.query_one(TranscriptView).blocks()
            if isinstance(b, NoticeBlock) and "interrupted" in b.text()
        ]
        assert notices == [], "the banked count was clobbered and the notice came back"


@pytest.mark.asyncio
async def test_a_reloaded_turns_queued_fallback_is_inert() -> None:
    """`/reload` throws the turn away, so its worker's fallback must drop.

    The reload path already resets the turn-scoped cluster; the latch is part
    of that cluster. Left open, a fallback dispatched after the reload would
    announce a completion for a conversation the user can no longer see — the
    same cross-session leak the other resets there exist to prevent.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)
        epoch = app._turn_epoch

        app._session_factory = lambda: _factory(JobsSession())  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        app.post_message(TurnAbandoned(epoch, aborted=False, error=None))
        await pilot.pause()

        assert _working(app) is None
    assert notifier.kinds == [], "a discarded turn announced itself after the reload"


@pytest.mark.asyncio
async def test_a_turn_that_never_started_leaves_the_band_idle() -> None:
    """The case the seam deliberately drops, and why the band's own line stays.

    `prompt()` can refuse BEFORE any `agent_start` — a disposed session, an
    "already streaming" refusal, an MCP failure while building the request. No
    `TurnStarted` ever arrives, so no turn is open and `_finalize_turn` is
    never reached; the `finally`'s own `streaming=False` is the only thing that
    clears the band, which is why removing it in favour of the seam would be
    wrong.
    """

    class RefusingSession(FakeSession):
        async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
            raise RuntimeError("session is already streaming")

    app = OperatorApp(lambda: _factory(RefusingSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert app._turn_open is False
        assert app._status is not None and not app._status._streaming
        assert _working(app) is None


class ExplodingLoopSession(JobsSession):
    """A loop iteration whose ``prompt()`` raises — the reported incident, in
    ``/loop``.

    Goal mode needs the judge primitive too, so one fake serves both workers;
    the verdict never matters here because the first prompt is what dies.
    """

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        raise RuntimeError("MCP authorization failed for 'minerva-qa'")

    async def complete_aside(self, turns: Any, **kwargs: Any) -> str:
        return "CONTINUE: not yet"


@pytest.mark.asyncio
async def test_a_dying_loop_iteration_retires_its_turn() -> None:
    """`/loop` drives `session.prompt()` too, and owes the same promise.

    Only the composer's call site was on this seam at first, so a numeric loop
    reproduced the WHOLE incident one surface over: the working line span
    forever, the band read idle, and nothing was announced. `/loop` is the case
    where that hurts most — it is the mode a user starts precisely so they can
    walk away from it.
    """
    app = OperatorApp(lambda: _factory(ExplodingLoopSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        await app._loop_worker(2)
        for _ in range(6):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None, "the loop left its thinking clock climbing"
        assert app._turn_open is False, "the latch stayed open on a dead loop turn"
        assert app._status is not None and not app._status._streaming
    assert notifier.kinds == ["error"], "a loop that died went unannounced"


@pytest.mark.asyncio
async def test_a_dying_goal_loop_reports_an_error_without_a_per_turn_toast() -> None:
    """The goal loop's single-toast contract, held across the new retirement.

    A held `/loop <goal>` owns ONE release toast, so a dying turn must not fire
    a per-turn completion — but it must still lift the spinner, clear the band
    and close the latch, and a loop that dies OUTRIGHT must not go silent.
    `_loop_suppress_completion` gates only the completion branch of the ladder,
    so carrying the error through takes the ERROR branch instead: exactly one
    notification, and it is the true one.
    """
    app = OperatorApp(lambda: _factory(ExplodingLoopSession()))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        await app._loop_goal_worker("ship the thing")
        for _ in range(8):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None
        assert app._turn_open is False
        assert app._loop_suppress_completion is False, "the deferred clear never ran"
    assert notifier.kinds == ["error"], "a held loop died silently, or double-announced"


#: The reported incident's error, in the form the transcript actually holds.
MCP_AUTH_ERROR = "MCP error: MCP OAuth authorization required for https://mcp.linear.app/mcp"


class _NamedServers:
    """The manager seam the hint path reads.

    `auth_recovery_hint` is what decides the VERB, and it lives on the manager
    precisely because the answer depends on state only the manager holds —
    whether a grant is stored for the server, and whether the server can take
    one at all (review R5). This stub stands in for a manager that holds a
    stored grant for `linear`, which is the shape where `reauth` is the truthful
    advice; `tests/unit/mcp/test_manager.py` drives the real derivation across
    all three auth shapes against real configs.
    """

    def get_all_server_names(self) -> list[str]:
        return ["linear", "minerva-qa"]

    def auth_recovery_hint(self, rendered_error: str) -> str | None:
        if "linear" not in rendered_error:
            return None
        return "run /mcp reauth linear — authorization expired"


class McpAuthRaisingSession(JobsSession):
    """`prompt()` dies on expired MCP auth — the motivating incident."""

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        raise RuntimeError(MCP_AUTH_ERROR)


def _error_notices(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_both_error_routes_offer_the_same_recovery() -> None:
    """The asymmetry, stated as the property that failed.

    The `agent_end` path appended a recovery hint and the worker's `except`
    printed a bare `str(error)`, so the IDENTICAL failure came with or without
    instructions purely by internal route — and the incident took the bare one,
    because an MCP auth failure is what makes `prompt()` raise. Asserted as
    "the two routes agree" rather than against a fixed string, so the next
    person to change the wording cannot re-open the gap on one side only.
    """
    raising = OperatorApp(lambda: _factory(McpAuthRaisingSession()))
    async with raising.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, raising)
        await _armed(pilot, raising, RecordingNotifier())
        raising._session.mcp_manager = _NamedServers()  # type: ignore[union-attr]

        raising._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)
        from_raise = _error_notices(raising)

    ending = OperatorApp(lambda: _factory(JobsSession()))
    async with ending.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, ending)
        await _armed(pilot, ending, RecordingNotifier())
        ending._session.mcp_manager = _NamedServers()  # type: ignore[union-attr]

        ending.post_message(TurnEnded(aborted=False, error=MCP_AUTH_ERROR))
        await pilot.pause()
        from_event = _error_notices(ending)

    assert from_raise == from_event, "the same failure was explained two different ways"
    assert any("/mcp reauth linear" in text for text in from_raise)


@pytest.mark.asyncio
async def test_an_mcp_failure_is_never_told_to_relogin_the_provider() -> None:
    """The wrong remedy is worse than none.

    An expired grant on `linear` has nothing to do with the Anthropic key, and
    `/login anthropic` sends the user to re-authorize a provider that was never
    broken: they do the work, the failure survives, and the real cause is now
    further away.
    """
    app = OperatorApp(lambda: _factory(McpAuthRaisingSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        app._session.mcp_manager = _NamedServers()  # type: ignore[union-attr]

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        joined = "\n".join(_error_notices(app))
        assert "/mcp reauth linear" in joined
        assert "/login" not in joined
        assert "credential update" not in joined


@pytest.mark.asyncio
async def test_a_session_that_died_with_an_error_does_not_read_idle() -> None:
    """The surface the incident was FOUND in, and the one it read wrong.

    `TitleState` was idle/working/attention only, so a session whose turn died
    with an error rendered `lo › name` — identical to one that finished
    cleanly. That is the exact surface a user scans to find what needs them,
    which is why the operator found this bug by opening sessions one at a time.

    The mark is turn-scoped, so the two ends of its life are pinned together
    here: a failure shows it, and the next turn takes it away.
    """
    from local_operator.tui.terminal_title import build_title

    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        assert app._status is not None

        app.post_message(TurnEnded(aborted=False, error="MCP authorization failed"))
        await pilot.pause()
        assert app._status._title_state() == "failed"
        assert build_title("ledger", app._status._title_state()) == "lo ✗ ledger"

        # A live turn outranks the mark: it describes the last SETTLED outcome.
        app.post_message(TurnStarted())
        await pilot.pause()
        assert app._status._title_state() == "working"


@pytest.mark.asyncio
async def test_an_abandoned_turn_marks_the_title_too() -> None:
    """The fallback owes the title exactly what the `agent_end` path owes it.

    `_finalize_turn` is the ONE exit, so the mark is set there rather than in
    `on_turn_ended` — a turn that died without an `agent_end` is precisely the
    case that used to read `idle`, and it is the case this PR exists for.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        assert app._status is not None

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error="boom"))
        await pilot.pause()
        assert app._status._title_state() == "failed"


@pytest.mark.asyncio
async def test_a_turn_the_user_stopped_is_not_marked_failed() -> None:
    """An abort is not a failure: the user stopped it on purpose.

    The same reasoning the notification ladder already applies (an aborted turn
    fires no toast, because the user was at the keyboard a moment ago). A cross
    on a session the user stopped themselves would be a false alarm in the
    surface whose whole job is telling them where a real one is.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        assert app._status is not None

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=True, error=None))
        await pilot.pause()
        assert app._status._title_state() == "idle"


@pytest.mark.asyncio
async def test_a_failing_turn_never_flashes_idle_in_the_title() -> None:
    """Asserted on the EMITTED BYTES, because that is where the defect was.

    The title dedupes on the rendered string and every band setter syncs it, so
    ORDER decides what a tab actually shows. Applying the failure mark after the
    band went idle wrote `lo › name` and then `lo ✗ name` — a one-tick flash of
    "finished cleanly" on a turn that failed, in the surface whose whole job is
    saying which session went wrong. Invisible to a state assertion, which only
    sees where things settled; visible in the write log.
    """
    from local_operator.tui.terminal_title import TerminalTitle

    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        assert app._status is not None

        writes: list[str] = []
        title = TerminalTitle(writes.append, enabled=True)
        # `emit` is silent until `start` has saved the terminal's own title,
        # and the app installs its own writer during boot — so attach after.
        title.start()
        app._status.set_terminal_title(title)
        await pilot.pause()

        app.post_message(TurnEnded(aborted=False, error="MCP authorization failed"))
        await pilot.pause()

        titles = [w.split("\x07")[0].split(";", 1)[1] for w in writes if "]0;" in w]
        assert titles[-1].startswith("lo ✗"), titles
        # The LAST write before the mark must still be a working frame: an idle
        # separator in between is the flash.
        assert not titles[-2].startswith("lo ›"), f"flashed idle before failing: {titles}"


class FailedOwnerFollowerSession(OwnerEndRacingFollowerSession):
    """The R4 interleaving: the same wire order, on a turn that FAILED.

    Identical to its base but for the owner's outcome, which is the whole point
    — the follower's own worker sees `error=None` either way, because its
    `prompt()` returned on the ACK long before the owner's turn ended. What
    distinguishes a clean turn from a failed one exists only on the owner's
    relayed end, arriving behind the fallback.
    """

    error: str | None = "MCP authorization failed for 'minerva-qa'"


@pytest.mark.asyncio
async def test_a_followers_failed_turn_is_never_toasted_as_complete() -> None:
    """R4. The fallback must not assert a clean finish it was never told about.

    On a follower the fallback WINS the gate — `prompt()` returns on the owner's
    ACK, so the worker's `finally` runs mid-turn — carrying the local worker's
    `error=None`. Read as an outcome, that announced "task complete" for a turn
    that had failed on the owner, while the terminal title on the SAME turn read
    `✗` because the real end behind it still reached the mark. Two new surfaces
    contradicting each other, on the path the PR names as its riskiest, and a
    REGRESSION: `main` has no `TurnAbandoned`, so the follower's `agent_end` is
    the only route there and the kind is always right.

    The fix is that `error=None` from a route that cannot know stops meaning
    "it succeeded" and starts meaning "ask the route that does".
    """
    session = FailedOwnerFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        def relay_owner_end() -> None:
            app.post_message(TurnEnded(aborted=False, error=session.error))

        session.relay = relay_owner_end

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None, "the turn never retired"
        assert app._status is not None
        # The two new surfaces AGREE, which is the actual claim.
        assert app._status._title_state() == "failed"
    assert notifier.kinds == ["error"], "a failed follower turn was announced as complete"


@pytest.mark.asyncio
async def test_a_followers_clean_turn_is_still_announced_once() -> None:
    """The other half: deferring the outcome must not swallow the notification.

    A fix that simply dropped the fallback's announcement would trade a WRONG
    toast for a MISSING one. The owner's real end is what speaks, and it speaks
    exactly once.
    """
    session = OwnerEndRacingFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        def relay_owner_end() -> None:
            app.post_message(TurnEnded(aborted=False, error=None))

        session.relay = relay_owner_end

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None
    assert notifier.kinds == ["complete"], "the follower's clean turn went unannounced"


@pytest.mark.asyncio
async def test_a_follower_whose_owner_never_reports_still_retires_the_turn() -> None:
    """Deferring the ANNOUNCEMENT must not defer the RETIREMENT.

    Everything above the notification gate — the working line, the band, the
    latch, the accrual, stranded cards — is what the fallback exists to reclaim,
    and a follower whose owner says nothing at all must still get all of it.
    Only the toast waits for a route that knows.
    """
    session = HangingFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        # `is_streaming` is still True, so guard 2 drops this fallback and the
        # turn stays open for the owner's end — the pre-existing contract.
        assert app._turn_open is True
        assert _working(app) is not None
        # And the BAND holds with it (#642): the worker's own write applies the
        # same predicate, so a live owner keeps the title on `working` rather
        # than dropping it to `idle` for a turn this viewer was never told about.
        assert app._status is not None and app._status._streaming is True
    assert notifier.kinds == [], "a mid-turn follower ACK was read as a finish"


@pytest.mark.asyncio
async def test_an_abandoned_failing_turn_never_flashes_idle_in_the_title() -> None:
    """D6/U8. The no-flash claim, on the route it did NOT hold for.

    The sibling test above covers the `agent_end` route, where `_finalize_turn`
    owns both writes. The ABANDONED route has a second, EARLIER writer: the turn
    worker's own `finally` calls `update(streaming=False)` directly, so a failed
    turn emitted `lo ›` ("finished cleanly") and only then `lo ✗`.

    It was ~0.5-1ms on an idle app — nobody saw it — but it was won by a RACE
    rather than structurally ordered: 43.5ms with 300 queued messages, 781ms at
    1500 queued blocks, well inside a rendered frame. This asserts the ordering
    is now structural, by giving the worker's write the outcome it already holds
    so there is no interval in which the title is wrong.
    """
    from local_operator.tui.terminal_title import TerminalTitle

    app = OperatorApp(lambda: _factory(ExplodingSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        assert app._status is not None

        writes: list[str] = []
        title = TerminalTitle(writes.append, enabled=True)
        title.start()
        app._status.set_terminal_title(title)
        await pilot.pause()

        # Dropped so the window under test is the TURN, not the attach: the
        # app writes an ordinary idle title while wiring the band up, and that
        # `lo ›` is correct — there is no turn yet.
        writes.clear()

        app._start_turn("Run the audit")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        titles = [w.split("\x07")[0].split(";", 1)[1] for w in writes if "]0;" in w]
        assert titles, "the turn wrote no title at all"
        assert titles[-1].startswith("lo ✗"), titles
        # NOT MERELY "the last write is right": the claim is that no write in
        # the whole retirement ever said the turn finished cleanly. A `[-2]`
        # check would pass on a sequence that flashed idle and then corrected
        # itself twice, which is the shape a race produces.
        assert not any(
            t.startswith("lo ›") for t in titles
        ), f"flashed 'finished cleanly' on a failed turn: {titles}"


@pytest.mark.asyncio
async def test_a_dying_loop_names_the_remedy_not_just_the_failure() -> None:
    """U6. `/loop` is the flow where a bare error costs the most.

    The recovery hint reached the composer's path but NEITHER loop worker, so
    the flow this PR just repaired retired cleanly and then told an
    away-from-keyboard user WHAT broke without WHAT TO DO. The operator's
    scenario is `/loop 20` overnight, a grant expiring at hour two, and a naked
    401 waiting in the morning — the same asymmetry U2 was raised about, on the
    UNATTENDED flow where a remedy is worth most.
    """
    app = OperatorApp(lambda: _factory(ExplodingLoopSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())

        await app._loop_worker(2)
        for _ in range(6):
            await pilot.pause()
            await asyncio.sleep(0.02)

        stopped = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock) and "loop stopped" in block.text()
        ]
        assert stopped, "the loop's failure was never printed"
        assert "/mcp" in stopped[0].text(), f"named no remedy: {stopped[0].text()!r}"


@pytest.mark.asyncio
async def test_a_dying_goal_loop_names_the_remedy_too() -> None:
    """The same, on the goal worker — the most unattended surface in the app."""
    app = OperatorApp(lambda: _factory(ExplodingLoopSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())

        await app._loop_goal_worker("ship it")
        for _ in range(6):
            await pilot.pause()
            await asyncio.sleep(0.02)

        stopped = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock) and "loop stopped" in block.text()
        ]
        assert stopped, "the goal loop's failure was never printed"
        assert "/mcp" in stopped[0].text(), f"named no remedy: {stopped[0].text()!r}"


@pytest.mark.asyncio
async def test_a_reload_clears_the_failure_mark_for_the_next_conversation() -> None:
    """R7. `StatusLine` is built once and outlives the session swap.

    Every other turn-scoped flag is explicitly cleared on this path with a
    comment about cross-session leaks; `_failed` was not, so between the reload
    and the replacement session's first turn the title carried a cross for a
    conversation the user can no longer see. Self-correcting on the next prompt,
    which is why it is minor — but the window is exactly the leak the
    neighbouring resets exist to prevent.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        assert app._status is not None

        app.post_message(TurnAbandoned(app._turn_epoch, aborted=False, error="boom"))
        await pilot.pause()
        assert app._status._title_state() == "failed"

        await app._reload_session()
        await pilot.pause()
        assert app._status._title_state() != "failed", "the cross outlived its conversation"


@pytest.mark.asyncio
async def test_a_crashed_loop_does_not_also_sign_off_as_finished() -> None:
    """U9. Two adjacent statements about one event, and the wrong one last.

    A numeric loop whose first prompt died printed `loop stopped: <error>` and
    then, directly beneath it, `loop finished after 0 iteration(s)` — in the
    quieter ink, so the reassuring line is the one the eye lands on last. The
    error line already says the loop stopped and why; "finished" is a claim
    about completion that a crashed loop cannot make.
    """
    app = OperatorApp(lambda: _factory(ExplodingLoopSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())

        await app._loop_worker(3)
        for _ in range(6):
            await pilot.pause()
            await asyncio.sleep(0.02)

        notices = [
            block.text()
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        assert any("loop stopped" in text for text in notices), notices
        assert not any("loop finished" in text for text in notices), notices


@pytest.mark.asyncio
async def test_a_loop_that_ran_out_of_iterations_still_signs_off() -> None:
    """The other side of U9: a healthy loop must keep its receipt.

    Suppressing the sign-off on a crash is only correct if a loop that actually
    completed still says so — otherwise the fix trades a contradictory line for
    a missing one.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())

        await app._loop_worker(2)
        for _ in range(6):
            await pilot.pause()
            await asyncio.sleep(0.02)

        notices = [
            block.text()
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        assert any("loop finished after 2 iteration(s)" in text for text in notices), notices


@pytest.mark.asyncio
async def test_a_reloaded_turns_late_error_does_not_mark_the_new_session() -> None:
    """R8. The late `agent_end` a `/reload` deliberately still delivers.

    The reload keeps its controller subscribed precisely so the dying turn's
    late `agent_end` can settle stranded tool cards — so that event arrives
    AFTER the swap, and it carries the dead conversation's error. Applying its
    outcome marked the REPLACEMENT session's title `failed`, a cross for work
    the user can no longer see. Cosmetic and self-correcting on the next turn,
    which is why it is a nit; it is also the reason clearing `_failed` on the
    reload path is not sufficient on its own.

    The cards must still settle: gating the whole handler is the opposite of
    this seam's fix, so only the OUTCOME is withheld.
    """
    app = OperatorApp(lambda: _factory(JobsSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        assert app._status is not None

        await app._reload_session()
        await pilot.pause()

        # The dead conversation's late end, arriving behind the swap.
        app.post_message(TurnEnded(aborted=False, error="the old conversation's failure"))
        await pilot.pause()

        assert app._status._title_state() != "failed", "a dead turn marked the live session"


@pytest.mark.parametrize(
    ("error", "expected_kind", "expected_title"),
    [
        pytest.param(None, "complete", "idle", id="clean"),
        pytest.param("MCP authorization failed for 'minerva-qa'", "error", "failed", id="failed"),
    ],
)
@pytest.mark.asyncio
async def test_the_toast_kind_and_the_title_agree_on_one_turn(
    error: str | None, expected_kind: str, expected_title: str
) -> None:
    """R4's INVARIANT, stated directly: one turn, one outcome, two surfaces.

    The R4 defect was not that either surface was wrong in isolation — it was
    that they DISAGREED: a toast reading "task complete" beside a title reading
    `✗` for the same turn. Each surface already has its own test, and both
    passed while the pair contradicted each other, because neither one looks at
    the other. This pins the relationship itself, on the follower path where the
    fallback and the owner's real end race, which is where they came apart.

    Parametrized over BOTH outcomes on purpose. A gate that suppressed the
    fallback's announcement entirely would satisfy the failing half while
    silently breaking the clean half into a missing toast, so the clean case is
    what keeps the fix honest rather than merely quiet.
    """
    session = FailedOwnerFollowerSession()
    session.error = error
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)

        def relay_owner_end() -> None:
            app.post_message(TurnEnded(aborted=False, error=session.error))

        session.relay = relay_owner_end

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert _working(app) is None, "the turn never retired"
        assert app._status is not None
        # Read inside the app context: the two facts are asserted about the SAME
        # turn, which is the only way the contradiction is observable at all.
        title = app._status._title_state()
    assert (notifier.kinds, title) == (
        [expected_kind],
        expected_title,
    ), f"the toast said {notifier.kinds} while the title said {title!r}"


def _titles(writes: list[str]) -> list[str]:
    """The rendered title strings, in write order, from a `TerminalTitle` spy."""
    return [w.split("\x07")[0].split(";", 1)[1] for w in writes if "]0;" in w]


async def _spy_title(pilot: Any, app: OperatorApp) -> list[str]:
    """Attach a write-recording `TerminalTitle` and drop the attach-time idle write."""
    from local_operator.tui.terminal_title import TerminalTitle

    assert app._status is not None
    writes: list[str] = []
    title = TerminalTitle(writes.append, enabled=True)
    title.start()
    app._status.set_terminal_title(title)
    await pilot.pause()
    # The band writes an ordinary `lo ›` while wiring up, and that one is
    # correct — there is no turn yet. The window under test is the TURN.
    writes.clear()
    return writes


async def _hold(pilot: Any, seconds: float) -> None:
    """Keep the pump running for ``seconds`` — the owner's relay latency."""
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        await asyncio.sleep(0.02)


@pytest.mark.parametrize("relay_delay", [0.05, 0.3])
@pytest.mark.asyncio
async def test_a_followers_failed_turn_holds_working_until_the_owner_reports(
    relay_delay: float,
) -> None:
    """#642 (D-1 from #619). "Ignorant" must not render as "finished cleanly".

    On a follower, `prompt()` returns on the owner's ACK — mid-turn — so the
    worker's `finally` runs with the outcome genuinely unknown. It used to
    write `streaming=False, failed=None` anyway, and with no failure mark set
    the title resolved to `idle`: `lo ›` for exactly the owner's relay latency
    (measured 501.7 ms at a 0.5 s relay, 2011 ms at 2 s), then `lo ✗`. A user
    glancing at the tab strip in that window read success on a failed turn.

    The band now HOLDS `working` while `is_streaming` is still True, and the
    outcome lands through `_finalize_turn` when the relayed end arrives. The
    delay is parametrized because the window WAS the relay: a fix that merely
    shrank it would pass at one delay and fail at the other. Asserted on the
    emitted bytes, as the sibling flash tests are, because that is where the
    defect lived.
    """
    session = HangingFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, RecordingNotifier())
        writes = await _spy_title(pilot, app)

        app._start_turn("Run the audit")
        await _hold(pilot, relay_delay)
        assert app._status is not None
        assert app._status._title_state() == "working", "the hold dropped while the owner was live"

        # The owner's relayed end: the turn FAILED.
        session.streaming = False
        app.post_message(TurnEnded(aborted=False, error="MCP authorization failed"))
        for _ in range(5):
            await pilot.pause()
            await asyncio.sleep(0.02)

    titles = _titles(writes)
    assert titles, "the turn wrote no title at all"
    assert titles[-1].startswith("lo ✗"), titles
    assert not any(
        t.startswith("lo ›") for t in titles
    ), f"flashed 'finished cleanly' on a failed follower turn: {titles}"


@pytest.mark.asyncio
async def test_a_followers_clean_turn_settles_idle_once_the_owner_reports() -> None:
    """The other half of #642: the hold releases to `›` on a clean relay.

    A hold that never let go would trade a false `idle` for a permanent
    `working`. The clean relay lands `lo ›`, and the completion toast fires
    exactly once — the hold changes WHEN the band settles, not whether the
    owner's end is announced.
    """
    session = HangingFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)
        writes = await _spy_title(pilot, app)

        app._start_turn("Run the audit")
        await _hold(pilot, 0.1)
        assert app._status is not None
        assert app._status._title_state() == "working"
        assert not any(t.startswith("lo ›") for t in _titles(writes)), _titles(writes)

        session.streaming = False
        app.post_message(TurnEnded(aborted=False, error=None))
        for _ in range(5):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert app._status._title_state() == "idle"
        assert _working(app) is None
    assert _titles(writes)[-1].startswith("lo ›"), _titles(writes)
    assert notifier.kinds == ["complete"], "the held clean turn went unannounced or doubled"


@pytest.mark.asyncio
async def test_a_follower_whose_prompt_is_refused_before_start_still_clears_the_band() -> None:
    """The hold is scoped to a LIVE owner, not to every follower.

    A follower's `prompt()` can refuse before any `agent_start` reaches the
    owner (disposed session, queue full, owner reconnecting). `is_streaming`
    is False, no `TurnStarted` ever arrives, and the worker's `finally` is the
    only thing that clears the band `_start_turn` lit — exactly the "KEPT"
    case the in-process refusal test pins. Gating the write on `is_remote`
    alone would have left THIS band lit forever.
    """

    class RefusingFollowerSession(JobsSession):
        is_remote = True

        async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
            raise RuntimeError("session owner is reconnecting")

    app = OperatorApp(lambda: _factory(RefusingFollowerSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        app._start_turn("do the thing")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert app._turn_open is False
        assert app._status is not None and not app._status._streaming
        assert _working(app) is None


@pytest.mark.asyncio
async def test_a_followers_held_turn_settles_when_the_owner_dies() -> None:
    """Owner death releases the hold: `›`, an "interrupted" notice, no toast.

    `RemoteSession._end_turn_locally` synthesises an aborted `AgentEndEvent`
    on EOF, which reaches the app as `TurnEnded(aborted=True)`. The held band
    must read that as the terminal state it is — not a failure (the owner did
    not report one) and not a completion (nothing completed).
    """
    session = HangingFollowerSession()
    app = OperatorApp(lambda: _factory(session))
    notifier = RecordingNotifier()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _armed(pilot, app, notifier)
        writes = await _spy_title(pilot, app)

        app._start_turn("Run the audit")
        await _hold(pilot, 0.1)
        assert app._status is not None
        assert app._status._title_state() == "working"

        session.streaming = False
        app.post_message(TurnEnded(aborted=True, error=None))
        for _ in range(5):
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert app._status._title_state() == "idle"
        assert _working(app) is None
        notices = [
            b.text() for b in app.query_one(TranscriptView).blocks() if isinstance(b, NoticeBlock)
        ]
        assert "interrupted" in notices, notices
    assert _titles(writes)[-1].startswith("lo ›"), _titles(writes)
    assert notifier.kinds == [], "an owner-death abort is not announced"

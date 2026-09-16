"""The ledger row for a call that was ANNOUNCED and has not started.

The reported defect, in the operator's words: "a stuck wake compose that never
completes and just hangs out at the edge of the conversation until the wait
completes". A composing row is the UI's optimistic PREDICTION that a call
exists, and the producer used to announce the prediction's beginning and then
only one of its three endings. Two of them left the row stranded:

* the call is QUEUED — `harness/loop.py` runs an `exclusive` call in a second
  group after the batch's `shared` ones, so a `wake` composed into the same
  step as `wait(wait_ms=1800000)` waits out the whole sibling before it starts;
* the call NEVER RUNS — a planning failure, a duplicate id, a steering skip —
  and its row then lived until the turn died, where it was labelled
  `interrupted` for a call that was never interrupted.

These tests drive the REAL loop through the REAL `EventController` into the
REAL `OperatorApp` under `run_test`, and read the painted frame back, because
every claim here is about what a frame SAYS. Where the state is set by the
reveal path instead of the live one, the app's own painter is called directly —
the reveal path's own tests live beside this file.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.loop import AgentLoop, LoopConfig, LoopContext
from local_operator.harness.types import (
    AgentEvent,
    AgentTool,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import EventController, ToolEnded, TurnBoundaryEnd
from local_operator.tui.widgets.tool_card import ToolCard

from .test_app_pilot import FakeSession, _factory
from .test_steering_approval import _boot

MODEL = ModelSpec(provider="test", model_id="m")

WAIT_ID = "call_wait_bf321"
#: Composed in the SAME step as the wait and executed in a later group, because
#: `wake` is `exclusive` and the batch runner does not mix the two kinds.
WAKE_ID = "call_wake_9d1c"


def _rows(app: OperatorApp) -> list[str]:
    return [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]


def _rows_for(app: OperatorApp, needle: str) -> list[str]:
    return [row for row in _rows(app) if needle in row]


def _tool_cards(app: OperatorApp) -> list[ToolCard]:
    """The mounted ledger rows, structurally — the frame is not the only view.

    Counted from the widgets rather than from the painted text because a
    transcript NOTICE can name the same tool and put the tool's name in a row
    of its own (`✗ Tool not found: wake`), which a text scan cannot tell from
    the card.
    """
    return [block for block in app._transcript_view().blocks() if isinstance(block, ToolCard)]


# ---------------------------------------------------------------------------
# The producer, driven in real time
# ---------------------------------------------------------------------------


class ScriptedStream:
    """Fake ``stream_fn``: replays a per-call script of stream events."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def _delta(index: int, *, id: str | None = None, name: str | None = None, args: str = ""):
    return StreamToolCallDelta(index=index, id=id, name=name, argument_delta=args)


class StrandedTurn:
    """One step composing `wait` and `wake`, handed to a test event by event.

    The events are pushed onto a queue by the producer task and pulled by
    ``pump``, so a frame can be read at the exact instant under test — the
    described frame is the one where the wait is running and the wake has not
    started, and a test that drained the whole turn could not see it.
    """

    def __init__(self, *, offer_wake: bool = True) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.seen: list[str] = []
        self.executed: list[str] = []
        self.release = asyncio.Event()
        self.offer_wake = offer_wake
        self._task: asyncio.Task[None] | None = None

    # -- the tools ---------------------------------------------------------

    def _wait_tool(self) -> AgentTool:
        async def execute(tool_call_id, args, signal, on_update, context):
            # Blocks until the test says so: the sibling's "whole duration" is
            # whatever the test needs it to be, and nothing here races a clock.
            await self.release.wait()
            self.executed.append("wait")
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="wait",
                content=[TextContent(text="waited")],
            )

        return AgentTool(
            name="wait",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            concurrency="shared",
            execute=execute,
        )

    def _wake_tool(self) -> AgentTool:
        async def execute(tool_call_id, args, signal, on_update, context):
            self.executed.append("wake")
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="wake",
                content=[TextContent(text="scheduled")],
            )

        return AgentTool(
            name="wake",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            concurrency="exclusive",
            execute=execute,
        )

    def tools(self) -> list[AgentTool]:
        tools = [self._wait_tool()]
        if self.offer_wake:
            tools.append(self._wake_tool())
        return tools

    # -- the driver --------------------------------------------------------

    def _steps(self) -> ScriptedStream:
        return ScriptedStream(
            [
                [
                    _delta(0, id=WAIT_ID, name="wait", args='{"text":"1800000"}'),
                    _delta(1, id=WAKE_ID, name="wake", args='{"text":"30m"}'),
                    StreamEndEvent(stop_reason="toolUse"),
                ],
                [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
            ]
        )

    def start(self) -> None:
        self._task = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        config = LoopConfig(
            model=MODEL,
            convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
            stream_fn=self._steps(),
        )
        context = LoopContext(system_blocks=["sys"], tools=self.tools())
        async for event in AgentLoop().run([Message.user("go")], context, config, None):
            self.seen.append(f"{event.type}:{getattr(event, 'tool_call_id', '')}")
            await self.queue.put(event)
        await self.queue.put(None)

    async def pump(self, controller: EventController, until: Any) -> AgentEvent:
        """Feed producer events into the app until ``until`` holds."""
        while True:
            event = await asyncio.wait_for(self.queue.get(), timeout=30)
            if event is None:
                raise AssertionError(f"the turn ended before the target: {self.seen}")
            controller._on_event(event)
            if until(event):
                return event

    async def finish(self) -> None:
        assert self._task is not None
        await asyncio.wait_for(self._task, timeout=30)


async def _parked(pilot: Any) -> None:
    """The producer's own pause: let the message pump run one cycle."""
    await pilot.pause()


# ---------------------------------------------------------------------------
# The operator's frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_queued_row_stops_saying_composing_the_moment_dictation_ends() -> None:
    """A `wake` behind a running `wait`: queued, not composing, and not ticking.

    `message_end` precedes the first `tool_execution_start` — the model is
    demonstrably done writing BOTH calls — so a row that still says `composing…`
    (let alone one whose clock is still moving) is claiming work that has
    stopped. The sibling's row is asserted too: the fix must not have frozen the
    turn, only told the truth about the call that is waiting.
    """
    turn = StrandedTurn()
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        # The app drops `SessionEvent`s whose `origin` is not its CURRENT
        # controller (its queue outlives subscriptions), so the controller under
        # test has to be the app's own.
        app._controller = controller
        turn.start()

        await turn.pump(controller, lambda e: isinstance(e, ToolExecutionStartEvent))
        await _parked(pilot)

        dictation_end = next(
            index for index, item in enumerate(turn.seen) if item.startswith("message_end")
        )
        first_start = next(
            index for index, item in enumerate(turn.seen) if item.startswith("tool_execution_start")
        )
        assert dictation_end < first_start, "dictation must be over before the batch starts"

        wake_rows = _rows_for(app, "wake")
        assert wake_rows, f"the queued call has a row: {_rows(app)}"
        assert not [
            row for row in wake_rows if "composing" in row
        ], f"the row claims the model is still writing it: {wake_rows!r}"
        assert "queued" in wake_rows[0], wake_rows
        assert _rows_for(app, "wait"), "the running sibling is still on screen"

        card = app._composing_cards[WAKE_ID]
        assert card.state == "queued"
        assert card._clock_timer is None, "the dictation clock is stopped, not merely quiet"

        # The row does not move. This is the operator's symptom read directly:
        # a frozen frame is what a stopped claim LOOKS like, and a ticking one
        # is what they reported for thirty to sixty minutes. Compared on THE
        # ROW, not on the whole frame: the frame also carries the status bar's
        # own clock and spinner, which tick for their own reasons.
        before = _rows_for(app, "wake")
        await asyncio.sleep(1.1)
        await _parked(pilot)
        assert _rows_for(app, "wake") == before, "the queued row must not tick"

        # The band must not say the model is composing either — it reads the
        # same registry, and the row's claim is what it echoes.
        label, phase, _clock, _from, _epoch = app._current_activity()
        assert phase == "running", "the running sibling owns the band's phase"
        assert "composing" not in label, label

        # Release the sibling: the queued call runs, and it runs in the SAME
        # row — one card per call id, adopted by its own start.
        turn.release.set()
        await turn.pump(
            controller,
            lambda e: isinstance(e, ToolExecutionStartEvent) and e.tool_call_id == WAKE_ID,
        )
        await _parked(pilot)
        assert not app._composing_cards, "the start adopted the announcement row"
        assert app._tool_cards.get(WAKE_ID) is card, "adopted, not re-mounted"
        assert card.state == "running"
        assert "composing" not in "".join(_rows_for(app, "wake"))

        await turn.pump(controller, lambda e: isinstance(e, ToolExecutionEndEvent))
        await turn.finish()
        await _parked(pilot)
        assert card.state == "success"
        assert app._tool_cards == {}, "both calls settled; nothing is left live"
        assert len(_rows_for(app, "wake")) == 1, "one row for one call, start to finish"


@pytest.mark.asyncio
async def test_a_joiner_replays_the_queued_row_without_a_viewer_relative_clock() -> None:
    """The same row, handed to a surface that becomes visible mid-turn.

    The seed is the REAL one: the producer's own events folded through the REAL
    ``FrontendStateStore``, snapshotted while the sibling still runs. Before the
    fix that snapshot carried a compose frame with nothing able to settle it, and
    every attach/sidebar-switch/`/resume` re-mounted the row with a clock
    relative to the VIEWER (the reported `8s`/`1m58s` beside a `7m8s`/`9m25s`).

    The assertion is the honest one available to a frame: no number is drawn,
    and the row says the one thing that is true about it.
    """
    turn = StrandedTurn()
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="p", cwd="/r"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller
        turn.start()

        live: list[dict[str, Any]] = []
        while True:
            event = await asyncio.wait_for(turn.queue.get(), timeout=30)
            if event is None:
                raise AssertionError(f"the turn ended too early: {turn.seen}")
            store.observe_event(None, event)
            if isinstance(event, ToolExecutionStartEvent):
                live = list(store.state.live_events)
                break

        seeded = [item for item in live if item.get("tool_call_id") == WAKE_ID]
        assert seeded, "the seed retains the queued call's own frames"
        assert not [
            item for item in seeded if item.get("type") == "tool_execution_start"
        ], "nothing has started it: that is the fact the row must render"

        controller.restore_live_projection(
            SimpleNamespace(streaming=True, generation=1, live_events=live), set(), set()
        )
        await _parked(pilot)

        wake_rows = _rows_for(app, "wake")
        assert len(wake_rows) == 1, f"one row for one call: {_rows(app)}"
        assert "queued" in wake_rows[0], wake_rows
        assert "composing" not in wake_rows[0], wake_rows
        card = app._composing_cards[WAKE_ID]
        assert card.state == "queued"
        assert card._clock_timer is None
        before = _rows_for(app, "wake")
        await asyncio.sleep(1.1)
        await _parked(pilot)
        assert _rows_for(app, "wake") == before, "a replayed queued row carries no clock to tick"

        turn.release.set()
        await turn.finish()


# ---------------------------------------------------------------------------
# The never-run call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_never_run_call_settles_when_the_verdict_exists() -> None:
    """`wake` is unknown to the harness: the row settles at the verdict.

    The harness already surfaces the failure as a notice; what was missing was
    the LEDGER's own ending, so the row went on composing until the turn died
    and was retired as `interrupted` — a word for a call that was interrupted.
    Now it settles mid-turn, with the harness's own reason, and the turn's death
    changes nothing about it.
    """
    turn = StrandedTurn(offer_wake=False)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller
        turn.release.set()
        turn.start()

        await turn.pump(controller, lambda e: isinstance(e, ToolExecutionEndEvent))
        await _parked(pilot)

        assert app._composing_cards == {}, "the announcement is settled, not still live"
        assert WAKE_ID not in app._tool_cards
        # One CARD for the call. The harness's own notice (`✗ Tool not found:
        # wake`) is a transcript row that also names the tool, so the frame scan
        # below is filtered on the ledger's wording rather than on the tool
        # name — the notice is a second account of the same verdict and is
        # expected to be there. The sibling's card is on screen too, by design:
        # this test's frame is a turn that ran one call and refused another.
        wake_cards = [card for card in _tool_cards(app) if card.tool_call_id == WAKE_ID]
        assert len(wake_cards) == 1
        assert wake_cards[0].state == "error"
        wake_rows = [row for row in _rows_for(app, "wake") if "never sent" in row]
        assert len(wake_rows) == 1, _rows(app)
        assert "Tool not found: wake" in wake_rows[0], wake_rows
        assert "composing" not in wake_rows[0]

        await turn.finish()
        # Turn death is now a no-op for this row: it has an outcome, and the
        # retirement pass only touches cards that are still live.
        app.post_message(TurnBoundaryEnd())
        await _parked(pilot)
        assert wake_cards[0].state == "error"
        assert [row for row in _rows_for(app, "wake") if "never sent" in row] == wake_rows


# ---------------------------------------------------------------------------
# One card per call id — the adoption rule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_seeded_compose_frame_never_mounts_a_second_row() -> None:
    """A replayed row plus the seed's frame for the same call is still ONE row.

    The reveal path paints a row for a live call from the transcript and the
    seed replays the announcement for it. `on_tool_composing` used to mount
    blindly, so the screen carried two rows for one call; when the call started,
    the composing card won `_tool_cards` and the replayed row was left in
    neither registry — unreachable by `on_tool_ended` and by turn-end
    retirement, stranded as `running` for the life of the process.

    The row is asserted to survive the whole lifecycle here: replay, seeded
    announcement, real start, real end, turn end.
    """

    class QueuedSession(FakeSession):
        """A session whose live turn reports this call as unanswered and unstarted."""

        def pending_display_tool_ids(self) -> set[str]:
            return set()

        def executing_display_tool_ids(self) -> set[str]:
            return {WAKE_ID}

        def live_tool_start_epochs(self) -> dict[str, float | None]:
            return {}

    session = QueuedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller

        # (a) The replayed row, registered exactly as the reveal path does.
        replayed = ToolCard(WAKE_ID, "wake", {"text": "30m"})
        app._append_block(replayed)
        app._mark_pending_tool_rows([replayed], session, app._tool_cards, app._composing_cards)
        await _parked(pilot)
        assert replayed.state == "queued"
        assert app._composing_cards[WAKE_ID] is replayed

        # (b) The seed's announcement for the SAME call.
        controller._on_event(
            ToolCallComposeEvent(
                tool_call_id=WAKE_ID,
                tool_name="wake",
                argument_bytes=14,
                dictation_complete=True,
            )
        )
        await _parked(pilot)
        assert app._composing_cards[WAKE_ID] is replayed, "a SECOND card was mounted"
        assert len(_tool_cards(app)) == 1
        assert len(_rows_for(app, "wake")) == 1, _rows(app)

        # (c) The call starts, adopting that row; (d) it ends, settling it.
        controller._on_event(
            ToolExecutionStartEvent(tool_call_id=WAKE_ID, tool_name="wake", args={"text": "30m"})
        )
        await _parked(pilot)
        assert app._tool_cards[WAKE_ID] is replayed
        assert not app._composing_cards

        controller._on_event(
            ToolExecutionEndEvent(
                tool_call_id=WAKE_ID,
                tool_name="wake",
                result=ToolResult(
                    tool_call_id=WAKE_ID,
                    tool_name="wake",
                    content=[TextContent(text="scheduled")],
                ),
            )
        )
        await _parked(pilot)
        assert replayed.state == "success"
        assert app._tool_cards == {}

        # (e) And nothing is left behind for the retirement pass to mislabel.
        app.post_message(TurnBoundaryEnd())
        await _parked(pilot)
        assert replayed.state == "success", "the row has an outcome; turn end must not relabel it"


@pytest.mark.asyncio
async def test_a_never_run_verdict_is_undone_when_the_twin_of_a_duplicate_id_runs() -> None:
    """Two calls, one id: the verdict settles the shared row, the winner revives it.

    A duplicate id is the one never-run case where the row belongs to TWO calls,
    so the verdict and the execution arrive for the same id in that order. The
    row must take both: settle under the verdict (that is the honest reading of
    one of the two calls) and then return to the live tier when the call that
    DID win actually executes — without a second card, and without the failure's
    dressing surviving into the running row.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller

        controller._on_event(
            ToolCallComposeEvent(tool_call_id="call_dup", tool_name="echo", argument_bytes=9)
        )
        await _parked(pilot)
        card = app._composing_cards["call_dup"]

        controller._on_event(
            ToolCallComposeEvent(
                tool_call_id="call_dup",
                tool_name="echo",
                argument_bytes=9,
                dictation_complete=True,
                not_run_reason="Duplicate call id 'call_dup' skipped.",
            )
        )
        await _parked(pilot)
        assert card.state == "error"
        assert not app._composing_cards
        assert "Duplicate call id" in _rows_for(app, "echo")[0]

        controller._on_event(
            ToolExecutionStartEvent(tool_call_id="call_dup", tool_name="echo", args={"text": "x"})
        )
        await _parked(pilot)
        assert card.state == "running", "the winner's start revives the row it shares"
        assert card._state == "running"
        assert "tool-error" not in card.classes, "the verdict's dressing must not survive"
        assert len(_tool_cards(app)) == 1

        controller._on_event(
            ToolExecutionEndEvent(
                tool_call_id="call_dup",
                tool_name="echo",
                result=ToolResult(
                    tool_call_id="call_dup",
                    tool_name="echo",
                    content=[TextContent(text="ok")],
                ),
            )
        )
        await _parked(pilot)
        assert card.state == "success"
        assert len(_rows_for(app, "echo")) == 1


@pytest.mark.asyncio
async def test_an_announcement_for_a_call_already_running_mounts_nothing() -> None:
    """A stale announcement must not colour a row whose call has started.

    Reachable whenever a viewer replays frames out of order — the seed's compose
    entry is dropped by the call's own start, but a surface that receives both
    can apply them in either order. The running registry is consulted first for
    exactly this: the row is the present, the frame is history.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller

        controller._on_event(
            ToolExecutionStartEvent(tool_call_id="call_x", tool_name="echo", args={"text": "x"})
        )
        await _parked(pilot)
        running = app._tool_cards["call_x"]
        assert running.state == "running"

        controller._on_event(
            ToolCallComposeEvent(tool_call_id="call_x", tool_name="echo", argument_bytes=9)
        )
        controller._on_event(
            ToolCallComposeEvent(
                tool_call_id="call_x",
                tool_name="echo",
                argument_bytes=9,
                dictation_complete=True,
            )
        )
        await _parked(pilot)
        assert app._composing_cards == {}, "no second card for a call that has started"
        assert running.state == "running", "and the running row is untouched"
        assert len(_tool_cards(app)) == 1


@pytest.mark.asyncio
async def test_a_frame_without_the_new_fields_still_mounts_and_composes() -> None:
    """BACKWARD COMPATIBILITY: an older owner's frames are ordinary frames.

    Built through ``AgentEvent.model_validate`` with only the keys that runtime
    knows, which is how ``deserialize_event`` rehydrates a relayed frame — so
    the payload genuinely lacks the keys rather than carrying nulls. The row must
    compose, exactly as it did before the fields existed; the absence of an
    ending is the pre-fix behaviour, not a crash.
    """
    from local_operator.harness.types import AgentEvent

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller

        legacy = AgentEvent.model_validate(
            {
                "type": "tool_call_compose",
                "tool_call_id": "call_legacy",
                "tool_name": "write",
                "argument_bytes": 14079,
            }
        )
        controller._on_event(legacy)
        await _parked(pilot)

        card = app._composing_cards["call_legacy"]
        assert card.state == "composing"
        assert [row for row in _rows_for(app, "write") if "composing" in row]


# ---------------------------------------------------------------------------
# The working line and turn death
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_working_line_says_waiting_to_run_and_holds_its_clock() -> None:
    """The band reads the same registry, so it has to read the same states.

    All three cards here are in `_composing_cards` — the announcement registry —
    but only one of them has a model still writing it. The line must not say
    `composing a call` for a call whose dictation ended minutes ago, and it must
    not print a number: the dictation clock has ended and there is no other zero
    to count from, so the honest reading is no reading.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)

        app._composing_cards["call_q"] = ToolCard("call_q", "wake", {"text": "30m"})
        app._composing_cards["call_q"].mark_queued()
        label, phase, clock, clock_from, clock_from_epoch = app._current_activity()
        assert label == "waiting to run a call", label
        assert phase == "queued"
        assert clock is False, "no number can be true here"
        assert clock_from is None and clock_from_epoch is None

        # A model that is STILL dictating outranks it, exactly as before: the
        # composing arm is the more specific fact.
        app._composing_cards["call_c"] = ToolCard("call_c", "write")
        app._composing_cards["call_c"].set_composing(120, "write")
        label, phase, clock, _from, _epoch = app._current_activity()
        assert label == "composing a call", label
        assert phase == "composing"
        assert clock is True


@pytest.mark.asyncio
async def test_a_queued_row_is_retired_as_never_sent_when_the_turn_dies() -> None:
    """Invariant: the new state lives in a registry the retirement pass reaches.

    A queued call may never start — the turn is stopped, the runtime is
    reloaded, the session is swapped — and its row must be settled by the same
    pass that settles every other live card. `never sent · N composed` is the
    correct wording for it (the call was never sent to a tool, and the size is
    how far the model got); `interrupted` is not, and a spinner definitely is
    not.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        card = ToolCard("call_q", "wake", {"text": "30m"})
        app._append_block(card)
        app._composing_cards["call_q"] = card
        card.set_composing(14, "wake")
        card.mark_queued()

        app.post_message(TurnBoundaryEnd())
        await _parked(pilot)

        assert app._composing_cards == {} and app._tool_cards == {}
        assert card.state == "interrupted"
        row = _rows_for(app, "wake")[0]
        assert "never sent" in row and "14 B composed" in row, row
        assert "queued" not in row
        # The outcome column stays BLANK. `mark_interrupted` used to print this
        # row's own age there — `_elapsed` measures from the moment the ROW was
        # built, i.e. the dictation plus the whole queue, printed in the column
        # where the sibling's real execution time lives; at the reported
        # half-hour wait the two endings of one fact disagreed (`never sent`
        # beside `⊘ 30m`). `mark_not_run` blanks the same number for the same
        # reason, and both rows now agree: nothing executed, so there is no
        # interval to draw.
        assert "⊘" in row, row
        assert re.search(r"\d+(?:\.\d+)?s\b", row) is None, row


# ---------------------------------------------------------------------------
# Round-1 remediation: what the terminal frame hands over, and what it clears
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verdict_that_is_the_only_frame_keeps_the_size_it_carried() -> None:
    """The terminal frame is the whole dictation on some surfaces.

    The live relay keeps one compose frame per call, in place, and the reconnect
    seed keeps exactly one entry per call id — so a viewer whose interim frames
    were compacted away, or that learns the call from the seed at all (attach,
    sidebar switch, `/resume`), receives this frame and NOTHING else. It carries
    the final `argument_bytes`, and a row born from it has never been through
    `set_composing`: without the value the row printed `nothing composed` over a
    frame that said how far the model got — the same lie as the ticking clock,
    one state later.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller

        controller._on_event(
            ToolCallComposeEvent(
                tool_call_id="call_solo",
                tool_name="mystery",
                argument_bytes=12,
                dictation_complete=True,
                not_run_reason="Tool not found: mystery",
            )
        )
        await _parked(pilot)

        rows = [row for row in _rows_for(app, "mystery") if "never sent" in row]
        assert rows, _rows(app)
        assert "12 B composed" in rows[0], rows
        assert "nothing composed" not in rows[0], rows


@pytest.mark.asyncio
async def test_a_joiner_replaying_a_never_run_verdict_keeps_the_size_too() -> None:
    """The same ending on a surface that becomes visible AFTER it.

    The seed is the real one — the producer's own events folded through the real
    ``FrontendStateStore`` — and it retains exactly ONE compose entry per call
    id, which for a never-run call IS the terminal frame. The live row's size
    came from frames the joiner never receives, so this is the frame that has to
    carry it: the two surfaces must paint the same words AND the same number.

    Reachability caveat, stated rather than implied: the switch path replays with
    `settled_tools` = the transcript's own results, and a never-run call is in
    that set once its synthetic result is persisted — the row is then painted
    from the transcript and this frame is skipped. The SEED path exercised here
    (restore_live_projection, `settled_tools` empty) is the attach and
    sidebar-switch case, where the owner's live projection is what the new
    surface folds.
    """
    turn = StrandedTurn(offer_wake=False)
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="p", cwd="/r"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        controller = EventController(session, app)
        app._controller = controller
        turn.start()

        live: list[dict[str, Any]] = []
        while True:
            event = await asyncio.wait_for(turn.queue.get(), timeout=30)
            if event is None:
                raise AssertionError(f"the turn ended too early: {turn.seen}")
            store.observe_event(None, event)
            if isinstance(event, ToolCallComposeEvent) and getattr(event, "not_run_reason", None):
                live = list(store.state.live_events)
                break

        seeded = [item for item in live if item.get("tool_call_id") == WAKE_ID]
        assert seeded, "the seed retains the call's terminal frame"
        assert not [
            item for item in seeded if item.get("type") == "tool_execution_start"
        ], "nothing has started it: that is the fact the row must render"

        controller.restore_live_projection(
            SimpleNamespace(streaming=True, generation=1, live_events=live), set(), set()
        )
        await _parked(pilot)

        rows = [row for row in _rows_for(app, "wake") if "never sent" in row]
        assert rows, _rows(app)
        assert "14 B composed" in rows[0], rows
        assert "nothing composed" not in rows[0], rows

        turn.release.set()
        await turn.finish()


@pytest.mark.asyncio
async def test_an_end_for_a_call_still_being_announced_clears_that_registry() -> None:
    """A settled card must not be left where the DEATH PASS can relabel it.

    `_retire_live_tool_cards` is deliberately unconditional: it marks everything
    still in `_composing_cards` as `interrupted` when the turn dies. A queued row
    now sits in that registry for a sibling's whole execution group — the
    reported half-hour — so an end arriving ahead of its start (a relay out of
    order, a replay) has to settle the card AND take it out of the registry, or
    the death pass overwrites an outcome that really happened.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        card = ToolCard("call_q", "wake", {"text": "30m"})
        app._append_block(card)
        app._composing_cards["call_q"] = card
        card.set_composing(14, "wake")
        card.mark_queued()

        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="call_q",
                    tool_name="wake",
                    result=ToolResult(
                        tool_call_id="call_q",
                        tool_name="wake",
                        content=[TextContent(text="scheduled")],
                    ),
                )
            )
        )
        await _parked(pilot)

        assert app._composing_cards == {}, "the outcome took it out of the registry"
        assert card.state == "success"

        app.post_message(TurnBoundaryEnd())
        await _parked(pilot)
        assert card.state == "success", "the death pass cannot reach a settled card"

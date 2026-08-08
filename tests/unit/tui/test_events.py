"""EventController tests — turn generations and orphaned tool-end buffering.

Drives the controller with a ``FakeSession`` implementing ``SessionProtocol``
and a stub app that records posted messages (no Textual run needed).
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentStartEvent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    NoticeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.session.protocol import SessionProtocol
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    EventController,
    NoticePosted,
    StartFlushTimer,
    ToolEnded,
    ToolStarted,
    TurnEnded,
    TurnStarted,
)


class FakeTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeApp:
    """Records posted Textual messages; supplies a manual flush timer.

    When the controller posts ``StartFlushTimer`` the fake app thread does
    what the real one would: call ``controller.start_flush_timer()``.
    """

    def __init__(self) -> None:
        self.posted: list[Any] = []
        self.intervals: list[tuple[float, Any]] = []
        self.timers: list[FakeTimer] = []
        self.controller: EventController | None = None

    def post_message(self, message: Any) -> None:
        self.posted.append(message)
        if isinstance(message, StartFlushTimer) and self.controller is not None:
            self.controller.start_flush_timer()

    def set_interval(self, interval: float, callback: Any) -> FakeTimer:
        timer = FakeTimer()
        self.timers.append(timer)
        self.intervals.append((interval, callback))
        return timer

    def flush(self) -> None:
        """Fire the pending interval callback (simulated 30 fps tick)."""
        if self.intervals:
            _interval, callback = self.intervals.pop(0)
            callback()


class FakeSession:
    """Minimal SessionProtocol that can emit events synchronously."""

    def __init__(self) -> None:
        self._handlers: list[Any] = []

    @property
    def session_id(self) -> str:
        return "sess"

    @property
    def agent_id(self) -> str:
        return "agent"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def model_label(self) -> str:
        return "test/model"

    @property
    def model(self) -> Any:
        return None

    def set_model(self, model: Any) -> None:
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
        pass

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    def history(self) -> list[Any]:
        return getattr(self, "_history", [])

    def steer(self, text: str) -> None:
        pass

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        pass

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            self._handlers.remove(handler)

        return unsubscribe

    @property
    def conversation_name(self) -> str:
        return getattr(self, "_name", "")

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        self._name = (text or "").strip()
        return self._name

    async def complete_once(self, system: str, prompt: str) -> str:
        return ""

    async def dispose(self) -> None:
        pass

    def emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)


def _controller() -> tuple[EventController, FakeSession, FakeApp]:
    session = FakeSession()
    app = FakeApp()
    controller = EventController(session, app)
    app.controller = controller
    controller.subscribe()
    return controller, session, app


def test_fake_session_satisfies_protocol() -> None:
    assert isinstance(FakeSession(), SessionProtocol)


def test_agent_start_bumps_generation() -> None:
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    assert controller.generation == 1
    session.emit(AgentEndEvent())
    assert [type(m) for m in app.posted] == [TurnStarted, TurnEnded]


def test_superseded_agent_end_is_ignored() -> None:
    """TUI-025: a stamped agent_end for an OLDER generation is dropped.

    Driven entirely through the controller's real event path: two stamped
    starts (the second supersedes the first), then the stale stamped end —
    no self-manufactured shortcuts, only ``session.emit``.
    """
    controller, session, app = _controller()
    # Turn 1 starts (stamped)…
    session.emit(AgentStartEvent(generation=1))
    assert controller.generation == 1
    # …but before its end arrives, turn 2 has already started.
    session.emit(AgentStartEvent(generation=2))
    assert controller.generation == 2
    # Now the stale turn-1 end arrives (dispatch crossed an async hop).
    session.emit(AgentEndEvent(generation=1))
    # It must be ignored: no TurnEnded for the live turn.
    assert [type(m) for m in app.posted] == [TurnStarted, TurnStarted]
    # The live turn still ends normally.
    session.emit(AgentEndEvent(generation=2))
    assert [type(m) for m in app.posted][-1] is TurnEnded


def test_unstamped_agent_end_falls_back_to_turn_counter() -> None:
    """Older producers (generation absent/0) use the monotonic counter."""
    controller, session, app = _controller()
    session.emit(AgentStartEvent(generation=0))
    session.emit(AgentStartEvent(generation=0))
    assert controller.generation == 2  # counter, not stamp
    # An unstamped end always belongs to the current turn.
    session.emit(AgentEndEvent(generation=0))
    assert isinstance(app.posted[-1], TurnEnded)


def test_unstamped_agent_end_tears_down() -> None:
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(AgentEndEvent())
    assert isinstance(app.posted[-1], TurnEnded)


def test_orphaned_tool_end_buffered_until_start() -> None:
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    result = ToolResult(tool_call_id="t1", tool_name="bash")
    # End arrives BEFORE its start: must be buffered, not crash, not posted.
    session.emit(ToolExecutionEndEvent(tool_call_id="t1", tool_name="bash", result=result))
    assert "t1" in controller.pending_tool_ends
    assert not any(isinstance(m, ToolEnded) for m in app.posted)
    # The start then arrives: card posted AND buffered end attached.
    session.emit(ToolExecutionStartEvent(tool_call_id="t1", tool_name="bash"))
    kinds = [type(m) for m in app.posted]
    assert ToolStarted in kinds and ToolEnded in kinds
    assert kinds.index(ToolStarted) < kinds.index(ToolEnded)
    assert "t1" not in controller.pending_tool_ends


def test_orphaned_buffer_dropped_on_agent_end() -> None:
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    result = ToolResult(tool_call_id="t2", tool_name="read")
    session.emit(ToolExecutionEndEvent(tool_call_id="t2", tool_name="read", result=result))
    assert "t2" in controller.pending_tool_ends
    session.emit(AgentEndEvent())
    assert controller.pending_tool_ends == {}


def test_paired_tool_end_posts_immediately() -> None:
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(ToolExecutionStartEvent(tool_call_id="t3", tool_name="grep"))
    result = ToolResult(tool_call_id="t3", tool_name="grep")
    session.emit(ToolExecutionEndEvent(tool_call_id="t3", tool_name="grep", result=result))
    assert [type(m) for m in app.posted] == [TurnStarted, ToolStarted, ToolEnded]


def test_message_update_coalesces_and_guards_equality() -> None:
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(MessageStartEvent(message=Message.assistant("")))
    session.emit(MessageUpdateEvent(message=Message.assistant("He"), delta="He"))
    session.emit(MessageUpdateEvent(message=Message.assistant("Hello"), delta="llo"))
    # Nothing flushed yet — the 30 Hz timer owns the flush.
    assert not any(isinstance(m, AssistantDelta) for m in app.posted)
    controller._flush_assistant()
    deltas = [m for m in app.posted if isinstance(m, AssistantDelta)]
    assert len(deltas) == 1
    assert deltas[0].text == "Hello"
    # Identical flush is a no-op (equality guard).
    controller._flush_assistant()
    assert len([m for m in app.posted if isinstance(m, AssistantDelta)]) == 1


def test_message_update_posts_start_flush_timer() -> None:
    """TUI-024: the timer starts via a posted message, app-thread only."""
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(MessageUpdateEvent(message=Message.assistant("x"), delta="x"))
    # The app thread handled StartFlushTimer and actually started a timer.
    assert any(isinstance(m, StartFlushTimer) for m in app.posted)
    assert len(app.timers) == 1
    assert not app.timers[0].stopped


def test_message_end_stops_flush_timer() -> None:
    """TUI-006: message_end stops the timer after its final flush."""
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(MessageStartEvent(message=Message.assistant("")))
    session.emit(MessageUpdateEvent(message=Message.assistant("x"), delta="x"))
    assert app.timers and not app.timers[0].stopped
    session.emit(MessageEndEvent(message=Message.assistant("x")))
    assert app.timers[0].stopped
    # The final flush delivered the buffered text before stopping.
    deltas = [m for m in app.posted if isinstance(m, AssistantDelta)]
    assert deltas[-1].text == "x"


def test_message_end_adopts_authoritative_text() -> None:
    """TUI-020: the block adopts event.message.text, not the local buffer."""
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(MessageStartEvent(message=Message.assistant("")))
    session.emit(MessageUpdateEvent(message=Message.assistant("partial"), delta="partial"))
    # The authoritative message carries MORE than the buffered deltas.
    session.emit(MessageEndEvent(message=Message.assistant("partial and complete")))
    ends = [m for m in app.posted if isinstance(m, AssistantMessageEnd)]
    assert ends[-1].text == "partial and complete"


def test_agent_end_final_flush_delivers_buffered_tail() -> None:
    """TUI-005: agent_end runs a final flush BEFORE stopping the timer."""
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(MessageStartEvent(message=Message.assistant("")))
    session.emit(MessageUpdateEvent(message=Message.assistant("tail"), delta="tail"))
    session.emit(AgentEndEvent())
    # The buffered tail reached the app even without a timer tick.
    deltas = [m for m in app.posted if isinstance(m, AssistantDelta)]
    assert deltas[-1].text == "tail"
    assert app.timers[0].stopped


def test_dispose_stops_flush_timer() -> None:
    """Timer lifecycle: dispose stops any running flush timer."""
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    session.emit(MessageUpdateEvent(message=Message.assistant("x"), delta="x"))
    assert app.timers and not app.timers[0].stopped
    controller.dispose()
    assert app.timers[0].stopped


def test_notice_forwarded() -> None:
    controller, session, app = _controller()
    session.emit(NoticeEvent(text="heads up", kind="warning"))
    notices = [m for m in app.posted if isinstance(m, NoticePosted)]
    assert notices[0].text == "heads up"
    assert notices[0].kind == "warning"


def test_dispose_unsubscribes() -> None:
    controller, session, app = _controller()
    controller.dispose()
    session.emit(AgentStartEvent())
    assert app.posted == []


@pytest.mark.asyncio
async def test_controller_async_handler_compat() -> None:
    """Sessions may deliver events from async contexts; sync handlers stay safe."""
    controller, session, app = _controller()
    session.emit(AgentStartEvent())
    assert controller.generation == 1

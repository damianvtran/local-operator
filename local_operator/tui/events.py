"""Event controller — marshals SessionProtocol events into Textual messages.

The agent emits typed :class:`~local_operator.harness.types.AgentEvent`s on
whatever loop/task they originate from. The TUI must only mutate widgets on
the Textual thread, so the controller converts each engine event into a
Textual :class:`~textual.message.Message` and ``app.post_message``s it. All
widget mutation then happens in the app's message handlers, on the right
thread. This keeps the engine → UI boundary exactly as omp has it: the agent
never imports the TUI.

Threading: ``post_message`` is safe only from the app's own asyncio loop.
``SessionProtocol`` delivery is documented same-loop, so every ``_post``
here runs on the right loop. Cross-thread producers must use
``app.call_from_thread``; the flush timer is started by POSTING an internal
:class:`StartFlushTimer` message instead of calling ``set_interval`` from an
event callback (TUI-024).

Ordering hazards ported from omp's ``EventController``:

- A **superseded ``agent_end``** can arrive AFTER the next ``agent_start``
  (dispatch crosses an async hop). Running teardown then would kill the live
  turn's loader. The controller adopts the generation stamped on each
  ``agent_start`` and drops ``agent_end`` events carrying an older
  generation. When the producer does not stamp generations (older harnesses
  emit ``generation == 0``), a monotonic turn counter inside the controller
  keys the same guard off ``agent_start`` arrivals.
- **Orphaned ``tool_execution_end``** events (an end with no matching start
  card yet) are buffered keyed by ``tool_call_id`` and attached when the
  start arrives; the buffer is dropped at turn end.

``message_update`` deltas are coalesced: the controller buffers incoming
deltas and flushes them on a ~30 Hz timer rather than posting per-token,
with an equality guard so identical text is free.
"""

from __future__ import annotations

from typing import Any, Callable

from textual.message import Message

from local_operator.harness.types import (
    AgentEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    NoticeEvent,
    RetryEndEvent,
    RetryStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

#: Streaming updates flush at ~30 fps (omp's coalesced cadence).
FLUSH_INTERVAL_S = 1.0 / 30.0


# --- Textual messages posted by the controller -----------------------------
# Each wraps the engine event so the app can mutate widgets on its own thread.


class StartFlushTimer(Message):
    """Internal: start the 30 Hz flush timer on the app thread (TUI-024)."""


class TurnStarted(Message):
    """An ``agent_start`` for the CURRENT generation."""


class TurnEnded(Message):
    """An ``agent_end`` that matched the current generation."""

    def __init__(
        self,
        aborted: bool,
        error: str | None,
        *,
        context_tokens: int = 0,
        usage: Any = None,
    ) -> None:
        super().__init__()
        self.aborted = aborted
        self.error = error
        self.context_tokens = context_tokens
        self.usage = usage


class TurnBoundaryStart(Message):
    """A ``turn_start`` engine event: one model call is beginning."""


class TurnBoundaryEnd(Message):
    """A ``turn_end`` engine event: one model call finished."""


class AssistantDelta(Message):
    """Flushed, coalesced assistant text (full accumulated text)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class AssistantMessageStart(Message):
    """A new assistant message began streaming."""


class AssistantMessageEnd(Message):
    """The streaming assistant message is complete."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ToolStarted(Message):
    """A tool execution started; a buffered completion may follow."""

    def __init__(self, event: ToolExecutionStartEvent) -> None:
        super().__init__()
        self.event = event


class ToolEnded(Message):
    """A tool execution finished."""

    def __init__(self, event: ToolExecutionEndEvent) -> None:
        super().__init__()
        self.event = event


class ToolUpdated(Message):
    """A running tool streamed a partial result."""

    def __init__(self, event: ToolExecutionUpdateEvent) -> None:
        super().__init__()
        self.event = event


class NoticePosted(Message):
    """A notice to surface in the transcript."""

    def __init__(self, text: str, kind: str) -> None:
        super().__init__()
        self.text = text
        self.kind = kind


class CompactionStarted(Message):
    """Context compaction began."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


class CompactionEnded(Message):
    """Context compaction finished."""

    def __init__(self, reason: str, success: bool) -> None:
        super().__init__()
        self.reason = reason
        self.success = success


class RetryStarted(Message):
    """A provider call is being retried."""

    def __init__(self, attempt: int, error: str, fallback_model: str | None) -> None:
        super().__init__()
        self.attempt = attempt
        self.error = error
        self.fallback_model = fallback_model


class RetryEnded(Message):
    """A provider retry resolved."""

    def __init__(self, success: bool) -> None:
        super().__init__()
        self.success = success


class EventController:
    """Subscribes to a session and posts Textual messages for each event.

    Construct with the session and the app (for ``post_message`` / timers).
    The controller is engine-agnostic; it works against any
    ``SessionProtocol``, including the test ``FakeSession``.

    Generation guard (TUI-001): the app-side current generation is adopted
    from each stamped ``agent_start``; an ``agent_end`` stamped with an
    OLDER generation is dropped so the live turn's loader survives. When the
    producer does not stamp generations (absent/None/0 on both sides), a
    monotonic turn counter inside the controller — bumped on every
    ``agent_start`` — provides the same guarantee for older harnesses.
    """

    def __init__(self, session: Any, app: Any) -> None:
        self._session = session
        self._app = app
        self._generation: int = 0
        self._turn_counter: int = 0  # monotonic fallback for older producers
        self._pending_tool_ends: dict[str, ToolExecutionEndEvent] = {}
        self._started_tools: set[str] = set()
        self._assistant_buffer: str = ""
        self._assistant_seen: str = ""
        self._flush_timer = None
        self._unsubscribe: Callable[[], Any] | None = None

    # -- lifecycle ----------------------------------------------------------
    def subscribe(self) -> None:
        """Register with the session; the returned callable unsubscribes."""
        self._unsubscribe = self._session.subscribe(self._on_event)

    def dispose(self) -> None:
        """Unsubscribe and stop the flush timer (idempotent)."""
        self._stop_flush_timer()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    @property
    def generation(self) -> int:
        """Current turn generation (test/inspection hook)."""
        return self._generation

    @property
    def pending_tool_ends(self) -> dict[str, ToolExecutionEndEvent]:
        """Orphaned tool ends awaiting their start (test hook)."""
        return self._pending_tool_ends

    # -- event dispatch -----------------------------------------------------
    def _on_event(self, event: AgentEvent) -> None:
        """Route one engine event to its handler (sync or async-safe)."""
        handler = self._HANDLERS.get(event.type)
        if handler is not None:
            handler(self, event)

    # -- per-event handlers -------------------------------------------------
    def _handle_agent_start(self, event: AgentEvent) -> None:
        # New turn: adopt the stamped generation when present; older
        # producers fall back to the controller's monotonic turn counter.
        gen = getattr(event, "generation", None)
        if gen:  # stamped by the harness (per-session monotonic, >= 1)
            self._generation = int(gen)
        else:
            self._generation += 1
        self._turn_counter += 1
        self._pending_tool_ends.clear()
        self._started_tools.clear()
        self._assistant_buffer = ""
        self._assistant_seen = ""
        self._post(TurnStarted())

    def _handle_agent_end(self, event: AgentEvent) -> None:
        # Superseded hazard (omp event-controller): an agent_end can arrive
        # AFTER the next agent_start when dispatch crosses an async hop.
        # Drop ends stamped with a generation OLDER than the current one;
        # unstamped ends (older producers) belong to the current turn.
        gen = getattr(event, "generation", None)
        if gen and gen < self._generation:
            return  # superseded: belongs to an earlier turn
        # Final flush BEFORE stopping the timer so no buffered tail is lost
        # (TUI-005); message_end also stops the timer after its own flush
        # (TUI-006).
        self._flush_assistant()
        self._pending_tool_ends.clear()
        self._stop_flush_timer()
        # Feed the latest usage into the status band (D10).
        usage = None
        for message in getattr(event, "messages", None) or []:
            message_usage = getattr(message, "usage", None)
            if message_usage is not None:
                usage = message_usage
        context_tokens = 0
        if usage is not None:
            context_tokens = getattr(usage, "context_tokens", None) or getattr(
                usage, "input_tokens", 0
            ) or 0
        self._post(
            TurnEnded(
                getattr(event, "aborted", False),
                getattr(event, "error", None),
                context_tokens=context_tokens,
                usage=usage,
            )
        )

    def _handle_turn_start(self, event: TurnStartEvent) -> None:
        self._post(TurnBoundaryStart())

    def _handle_turn_end(self, event: TurnEndEvent) -> None:
        self._post(TurnBoundaryEnd())

    def _handle_message_start(self, event: MessageStartEvent) -> None:
        self._assistant_buffer = ""
        self._assistant_seen = ""
        self._post(AssistantMessageStart())

    def _handle_message_update(self, event: MessageUpdateEvent) -> None:
        # Buffer the delta; flush on the 30 Hz timer (coalescing).
        self._assistant_buffer += event.delta
        self._request_flush_timer()

    def _handle_message_end(self, event: MessageEndEvent) -> None:
        # Adopt the authoritative message text (TUI-020), not the local
        # buffer; final flush, then stop the timer (TUI-006).
        text = getattr(event.message, "text", None)
        if text is None:
            text = self._assistant_buffer
        self._assistant_buffer = text
        self._flush_assistant()
        self._stop_flush_timer()
        self._post(AssistantMessageEnd(text))

    def _handle_tool_start(self, event: ToolExecutionStartEvent) -> None:
        self._started_tools.add(event.tool_call_id)
        self._post(ToolStarted(event))
        # Attach a buffered (orphaned) end now that the card exists.
        buffered = self._pending_tool_ends.pop(event.tool_call_id, None)
        if buffered is not None:
            self._post(ToolEnded(buffered))

    def _handle_tool_update(self, event: ToolExecutionUpdateEvent) -> None:
        self._post(ToolUpdated(event))

    def _handle_tool_end(self, event: ToolExecutionEndEvent) -> None:
        # Orphaned-end hazard (omp): an end with no matching start card must
        # be BUFFERED, not crash — it attaches when the start arrives, and is
        # dropped at agent_end if the start never comes.
        if event.tool_call_id in self._started_tools:
            self._post(ToolEnded(event))
        else:
            self._pending_tool_ends[event.tool_call_id] = event

    def _handle_notice(self, event: NoticeEvent) -> None:
        self._post(NoticePosted(event.text, event.kind))

    def _handle_compaction_start(self, event: CompactionStartEvent) -> None:
        self._post(CompactionStarted(event.reason))

    def _handle_compaction_end(self, event: CompactionEndEvent) -> None:
        self._post(CompactionEnded(event.reason, event.success))

    def _handle_retry_start(self, event: RetryStartEvent) -> None:
        self._post(RetryStarted(event.attempt, event.error, event.fallback_model))

    def _handle_retry_end(self, event: RetryEndEvent) -> None:
        self._post(RetryEnded(event.success))

    _HANDLERS = {
        "agent_start": _handle_agent_start,
        "agent_end": _handle_agent_end,
        "turn_start": _handle_turn_start,
        "turn_end": _handle_turn_end,
        "message_start": _handle_message_start,
        "message_update": _handle_message_update,
        "message_end": _handle_message_end,
        "tool_execution_start": _handle_tool_start,
        "tool_execution_update": _handle_tool_update,
        "tool_execution_end": _handle_tool_end,
        "notice": _handle_notice,
        "compaction_start": _handle_compaction_start,
        "compaction_end": _handle_compaction_end,
        "retry_start": _handle_retry_start,
        "retry_end": _handle_retry_end,
    }

    # -- flush timer --------------------------------------------------------
    def _request_flush_timer(self) -> None:
        """Ask the app thread to start the flush timer (TUI-024)."""
        if self._flush_timer is None:
            self._post(StartFlushTimer())

    def start_flush_timer(self) -> None:
        """App-thread entry: actually start the 30 Hz interval."""
        if self._flush_timer is None:
            self._flush_timer = self._app.set_interval(FLUSH_INTERVAL_S, self._flush_assistant)

    def _stop_flush_timer(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None

    def _flush_assistant(self) -> None:
        """Post the buffered assistant text, guarded by equality."""
        if self._assistant_buffer == self._assistant_seen:
            return  # equality guard — identical text = no work
        self._assistant_seen = self._assistant_buffer
        self._post(AssistantDelta(self._assistant_buffer))

    def _post(self, message: Message) -> None:
        self._app.post_message(message)

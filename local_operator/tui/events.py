"""Event controller — marshals SessionProtocol events into Textual messages.

The agent emits typed :class:`~local_operator.harness.types.AgentEvent`s on
whatever loop/task they originate from. The TUI must only mutate widgets on
the Textual thread, so the controller converts each engine event into a
Textual :class:`~textual.message.Message` and ``app.post_message``s it. All
widget mutation then happens in the app's message handlers, on the right
thread. This keeps the engine → UI boundary clean: the agent
never imports the TUI.

Threading: ``post_message`` is safe only from the app's own asyncio loop.
``SessionProtocol`` delivery is documented same-loop, so every ``_post``
here runs on the right loop. Cross-thread producers must use
``app.call_from_thread``; the flush timer is started by POSTING an internal
:class:`StartFlushTimer` message instead of calling ``set_interval`` from an
event callback (TUI-024).

Ordering hazards handled by the controller:

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
    HistoryDeltaEvent,
    ImageContent,
)
from local_operator.harness.types import Message as HarnessMessage
from local_operator.harness.types import (
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    NoticeEvent,
    PeerMessageDeliveredEvent,
    RetryEndEvent,
    RetryStartEvent,
    SteeringDeliveredEvent,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
    WakeDeliveredEvent,
)
from local_operator.tui.widgets.transcript import NoticeKind

#: Streaming updates flush at ~30 fps (the coalesced cadence).
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
        context_is_estimate: bool = False,
    ) -> None:
        super().__init__()
        self.aborted = aborted
        self.error = error
        self.context_tokens = context_tokens
        self.usage = usage
        self.context_is_estimate = context_is_estimate


class ContextUsageReported(Message):
    """One model call reported how large the context was when it ran.

    Separate from :class:`TurnEnded` because it fires per CALL, not per turn.
    An agentic turn is many calls over many minutes, and the context reading
    used to move only when the whole turn settled — so during exactly the
    stretch a user is watching it grow, the band reported the size the context
    had before the turn began. On a long tool-using turn that is a number tens
    of thousands of tokens stale, and it looks live.

    ``usage`` is the whole reading the call reported, not just its size, so the
    band's COST segment can move on the same signal for the same reason. Before
    it did, money appeared only at ``agent_end``: a first turn that spent ten
    minutes in tools showed no cost at all while it ran, which reads as a free
    session precisely while it is becoming an expensive one. ``context_tokens``
    stays a separate field rather than being read back off ``usage`` because the
    controller resolves it with a fallback (``context_tokens or input_tokens``)
    that the raw object does not carry.
    """

    def __init__(self, context_tokens: int, usage: Any = None) -> None:
        super().__init__()
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


class UserMessageStart(Message):
    """A user message reached the session from ANY front end. Carries the
    prompt (and image count) so the TUI can paint it — the mobile→TUI half of
    keeping the two surfaces in step. The TUI's own prompt path already paints
    its UserBlock optimistically, so the app de-dupes on arrival. The field is
    ``prompt``, not ``text``: Textual's ``Message`` reserves ``text``.

    ``message_id`` is the announced ``Message``'s own id — the correlation key
    the app's pending-echo registry matches on, in the same role
    :class:`PeerMessageDelivered` uses it for. It is what tells a prompt this
    TUI painted apart from a DISTINCT message that happens to carry identical
    words (repeated "yes" / "continue" from the phone, issue #228); matching
    those by text swallowed the foreign row. Defaults to empty for reduced
    event producers and synthetic tests, where the app falls back to its
    historical text match.
    """

    def __init__(self, prompt: str, images: list[ImageContent] | int, message_id: str = "") -> None:
        super().__init__()
        self.prompt = prompt
        self.message_id = message_id
        # Integer remains accepted for older synthetic tests and reduced event
        # producers. Production carries immutable blocks; only that path can
        # render thumbnails, while the compatibility path preserves its receipt.
        self.images = tuple(images) if not isinstance(images, int) else ()
        self.image_count = images if isinstance(images, int) else len(images)


class AssistantMessageEnd(Message):
    """The streaming assistant message is complete."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class HistoryRowsSettled(Message):
    """Durable rows that settled while no frontend painted them (reconnect gap).

    Carries the rows verbatim so the app projects them through the SAME
    role-aware settled-history renderer a cold resume uses — user rows as
    user blocks with their images, assistant rows as prose plus paired tool
    cards, custom rows through their own block paths. Routing these through
    :class:`AssistantMessageEnd` painted every role as assistant speech
    (review round 3, MAJOR-1/U7/D1).
    """

    def __init__(self, messages: list[Any]) -> None:
        super().__init__()
        self.messages = messages


class ToolComposing(Message):
    """The model is still dictating a tool call's arguments."""

    def __init__(self, event: ToolCallComposeEvent) -> None:
        super().__init__()
        self.event = event


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


class WakeDelivered(Message):
    """A scheduled wake's prompt was delivered — render the expandable receipt."""

    def __init__(self, text: str, catchup: bool, wake_id: str = "", occurrence: int = 0) -> None:
        super().__init__()
        self.text = text
        self.catchup = catchup
        self.wake_id = wake_id
        self.occurrence = occurrence


class PeerMessageDelivered(Message):
    """A message from another local lop session (`lop send`) was delivered.

    Carries the raw body and the advisory sender identity so the app can paint
    the cross-session indicator immediately, even while the session is idle.
    ``message_id`` is the persisted transcript row's id, recorded by the app so
    a later history replay does not mount a second copy of the same delivery.
    """

    def __init__(self, body: str, sender: dict[str, Any], message_id: str = "") -> None:
        super().__init__()
        self.body = body
        self.sender = sender
        self.message_id = message_id


class NoticePosted(Message):
    """A notice to surface in the transcript."""

    def __init__(self, text: str, kind: NoticeKind) -> None:
        super().__init__()
        self.text = text
        # Annotated, not inferred: pyright WIDENS a literal to ``str`` when it
        # infers an attribute's type from an assignment, which silently undid the
        # typing at the one hop that carries a kind across a thread boundary.
        self.kind: NoticeKind = kind


class CompactionStarted(Message):
    """Context compaction began."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


class CompactionEnded(Message):
    """Context compaction finished.

    Carries what the pass achieved (the history size either side of it, and the
    strategy that ran) so the notice can report a result rather than only the
    fact that something happened.
    """

    def __init__(
        self,
        reason: str,
        success: bool,
        strategy: str = "",
        tokens_before: int = 0,
        tokens_after: int = 0,
        detail: str | None = None,
    ) -> None:
        super().__init__()
        self.reason = reason
        self.success = success
        self.strategy = strategy
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        #: Optional clause explaining a pass whose timing the numbers do not
        #: explain (the compaction advisor firing below the threshold).
        self.detail = detail


class RetryStarted(Message):
    """A provider call is being retried."""

    def __init__(self, attempt: int, error: str, fallback_model: str | None) -> None:
        super().__init__()
        self.attempt = attempt
        self.error = error
        self.fallback_model = fallback_model


class EffectiveModelChanged(Message):
    """The model actually serving requests changed (fallback or recovery).

    What the status band repaints its model segment from: ``provider`` and
    ``model_id`` name the model now answering, ``is_fallback`` says whether it
    is a fallback detour or the selected model back in force. Carried as data
    rather than read off the session at handling time because the event
    crosses the controller's thread boundary — by the time the app thread
    handles it, the session may already be mid-way through the NEXT edge.
    """

    def __init__(
        self,
        provider: str,
        model_id: str,
        effort: str | None,
        reason: str,
        is_fallback: bool,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model_id = model_id
        self.effort = effort
        self.reason = reason
        self.is_fallback = is_fallback


class RetryEnded(Message):
    """A provider retry resolved."""

    def __init__(self, success: bool) -> None:
        super().__init__()
        self.success = success


class SteeringDelivered(Message):
    """Queued mid-turn messages reached the model's context.

    The receipt the ``queued — sends when this step finishes`` row was missing:
    the app settles that row against this instead of leaving a promise about the
    future standing after the future arrived. ``count`` is how many messages
    went in at the one boundary.

    ``origin`` is the controller that posted it, and it is what makes the
    receipt safe across a session swap (issue #160, F3). A generation number
    cannot answer this hazard: generations are per-controller, and the stale
    event comes from a DIFFERENT controller with its own counter, so the two
    numbers are not comparable. Identity is, and the app drops a delivery whose
    origin is no longer the controller it is listening to — see
    ``OperatorApp.on_steering_delivered``. ``None`` means "unstamped, treat as
    current": a hand-posted message in a test is simulating the live session,
    and only a stamped origin can be PROVEN stale.
    """

    def __init__(self, count: int, origin: "EventController | None" = None) -> None:
        super().__init__()
        self.count = count
        self.origin = origin


class SubagentStarted(Message):
    """A child session was registered as a background ``task`` job."""

    def __init__(self, job_id: str, label: str, agent_id: str | None = None) -> None:
        super().__init__()
        self.job_id = job_id
        self.label = label
        self.agent_id = agent_id


class SubagentProgress(Message):
    """A throttled relay of a child session's activity (tool starts/ends,
    message ends — never per-token deltas; the relay bounds that)."""

    def __init__(self, job_id: str, label: str, progress: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.label = label
        self.progress = progress


class SubagentEnded(Message):
    """A child session settled; the app repaints the band's subagent row.

    The band is the only surface this touches — the row flips from its
    spinner to the ✓/✗ outcome glyph on the next refresh. No transcript line
    is appended; the live band row IS the news."""

    def __init__(
        self,
        job_id: str,
        label: str,
        status: str,
        result_text: str | None = None,
        error_text: str | None = None,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.label = label
        self.status = status
        self.result_text = result_text
        self.error_text = error_text


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
        self._stamped_start_turn: int | None = None
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
        # The turn counter of the run whose start currently owns the UI:
        # an unstamped agent_end after a newer start must be dropped (see
        # _handle_agent_end).
        if gen:
            self._stamped_start_turn = None
        else:
            self._stamped_start_turn = self._turn_counter
        self._pending_tool_ends.clear()
        self._started_tools.clear()
        self._assistant_buffer = ""
        self._assistant_seen = ""
        self._post(TurnStarted())

    def _handle_agent_end(self, event: AgentEvent) -> None:
        # Superseded hazard: an agent_end can arrive
        # AFTER the next agent_start when dispatch crosses an async hop.
        # Drop ends stamped with a generation OLDER than the current one;
        # unstamped ends (older producers) belong to the current turn.
        gen = getattr(event, "generation", None)
        if gen:
            if gen < self._generation:
                return  # superseded: belongs to an earlier (newer stamped) turn
        elif self._stamped_start_turn is not None and self._turn_counter > self._stamped_start_turn:
            # Unstamped end (older producer) but the controller has seen a
            # NEWER start since the one that opened this run: the end belongs
            # to the superseded earlier turn and must not tear down the live
            # one.
            return
        # Final flush BEFORE stopping the timer so no buffered tail is lost
        # (TUI-005); message_end also stops the timer after its own flush
        # (TUI-006).
        self._flush_assistant()
        self._pending_tool_ends.clear()
        self._stop_flush_timer()
        # Feed usage into the status band (D10). Cost must SUM every model
        # call in the turn — the harness emits one assistant message per call
        # and a tool-using turn spends most of its tokens in the earlier
        # calls; the old overwrite-on-each-iteration loop counted only the
        # last one. context_tokens is a point-in-time size, taken from the
        # last message that reports it.
        usage = None
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        cost_components: list[Usage] = []
        context_tokens = 0
        for message in getattr(event, "messages", None) or []:
            message_usage = getattr(message, "usage", None)
            if message_usage is None:
                continue
            totals["input"] += getattr(message_usage, "input_tokens", 0) or 0
            totals["output"] += getattr(message_usage, "output_tokens", 0) or 0
            totals["cache_read"] += getattr(message_usage, "cache_read_tokens", 0) or 0
            totals["cache_write"] += getattr(message_usage, "cache_write_tokens", 0) or 0
            # Preserve each billed call rather than collapsing a partial receipt
            # beside every call's tokens. The consumer prices these components
            # independently, so reported calls stay authoritative while calls
            # without receipts still receive their estimate exactly once.
            components = getattr(message_usage, "cost_components", None)
            cost_components.extend(
                component.model_copy() for component in components or [message_usage]
            )
            usage = message_usage
            context_tokens = (
                getattr(message_usage, "context_tokens", None)
                or getattr(message_usage, "input_tokens", 0)
                or context_tokens
            )
        if usage is not None:
            usage = Usage(
                input_tokens=totals["input"],
                output_tokens=totals["output"],
                cache_read_tokens=totals["cache_read"],
                cache_write_tokens=totals["cache_write"],
                context_tokens=context_tokens or None,
                cost_components=cost_components,
            )
        # The aggregate still prices the calls. A post-turn compaction can stamp
        # a newer occupancy level on the boundary without corrupting that bill.
        settled_context = getattr(event, "context_tokens", None)
        self._post(
            TurnEnded(
                getattr(event, "aborted", False),
                getattr(event, "error", None),
                context_tokens=(settled_context if settled_context is not None else context_tokens),
                usage=usage,
                context_is_estimate=settled_context is not None,
            )
        )

    def _handle_turn_start(self, event: TurnStartEvent) -> None:
        self._post(TurnBoundaryStart())

    def _handle_turn_end(self, event: TurnEndEvent) -> None:
        self._post(TurnBoundaryEnd())

    def _handle_message_start(self, event: MessageStartEvent) -> None:
        # User turns reach here too now (the session emits MessageStartEvent
        # for them so every front end stays in step). Route them apart from
        # assistant starts: an assistant start resets the stream buffer, a
        # user start carries the prompt for the app to paint.
        message = event.message
        # Narrow to Message (not CustomMessage) before touching role/content:
        # a user MessageStartEvent is always a Message, but the field's type
        # is the AgentMessage union, so pyright needs the isinstance.
        if isinstance(message, HarnessMessage) and message.role == "user":
            images = [b for b in message.content if isinstance(b, ImageContent)]
            # The message id rides along as the echo-dedup correlation key: the
            # app registered the same id when it painted the row optimistically,
            # so an event whose id it does not recognise is a genuinely new
            # message and must be painted even when the words repeat (#228).
            self._post(UserMessageStart(message.text, images, message.id))
            return
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
        # Keep the context reading live THROUGH the turn, not just after it.
        # One agentic turn is many model calls; each reports the context size
        # it ran against, and that is the only signal the band can move on
        # while the agent is still working.
        message_usage = getattr(event.message, "usage", None)
        if message_usage is not None:
            size = (
                getattr(message_usage, "context_tokens", None)
                or getattr(message_usage, "input_tokens", 0)
                or 0
            )
            if size:
                self._post(ContextUsageReported(int(size), message_usage))
            else:
                # A call that reported tokens but no size still cost money, and
                # the cost segment must not be gated on the context segment
                # having something to say. Posting with size 0 lets the app take
                # the money and leave the reading alone.
                self._post(ContextUsageReported(0, message_usage))

    def _handle_history_delta(self, event: HistoryDeltaEvent) -> None:
        # Settled history, not a live stream: nothing here touches the
        # assistant buffer or the flush timer, because these rows were never
        # streaming in THIS frontend. The app owns the role-aware projection.
        self._post(HistoryRowsSettled(list(event.messages)))

    def _handle_tool_compose(self, event: ToolCallComposeEvent) -> None:
        self._post(ToolComposing(event))

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
        # Orphaned-end hazard: an end with no matching start card must
        # be BUFFERED, not crash — it attaches when the start arrives, and is
        # dropped at agent_end if the start never comes.
        if event.tool_call_id in self._started_tools:
            self._post(ToolEnded(event))
        else:
            self._pending_tool_ends[event.tool_call_id] = event

    def _handle_notice(self, event: NoticeEvent) -> None:
        self._post(NoticePosted(event.text, event.kind))

    def _handle_wake_delivered(self, event: WakeDeliveredEvent) -> None:
        # No generation guard: a wake receipt is a state fact about the
        # session (this prompt was delivered), not a turn-scoped boundary, so
        # a superseded turn cannot invalidate it.
        self._post(WakeDelivered(event.text, event.catchup, event.wake_id, event.occurrence))

    def _handle_peer_message_delivered(self, event: PeerMessageDeliveredEvent) -> None:
        # No generation guard: a peer delivery is a state fact about the
        # session (this message landed), not a turn-scoped boundary, so a
        # superseded turn cannot invalidate it — same rationale as a wake.
        self._post(PeerMessageDelivered(event.body, dict(event.sender), event.message_id))

    def _handle_steering_delivered(self, event: SteeringDeliveredEvent) -> None:
        # No TURN guard, deliberately: the drain belongs to whichever turn is
        # running, and the app settles a row it is holding a direct reference to
        # rather than looking one up by turn. Within one session a late event
        # finds nothing held and does nothing.
        #
        # A SESSION guard, though, because that reasoning stops holding at a
        # swap (issue #160, F3). `/reload` disposes this controller, but a
        # `steering_delivered` already dispatched is a Textual message sitting
        # in the app's queue, and unsubscribing cannot recall it. It is handled
        # after the swap cleared the held lists — by which time the user may
        # have steered into the REPLACEMENT session, whose row is then held and
        # would be falsely settled by the dying session's drain.
        #
        # Stamped with `self`, not with a generation: generations are
        # per-controller counters, so the outgoing controller's number is not
        # comparable with the incoming one's and could collide with it outright.
        # The app compares against the controller it currently listens to, which
        # is the question actually being asked.
        self._post(SteeringDelivered(getattr(event, "count", 1), origin=self))

    def _handle_compaction_start(self, event: CompactionStartEvent) -> None:
        self._post(CompactionStarted(event.reason))

    def _handle_compaction_end(self, event: CompactionEndEvent) -> None:
        self._post(
            CompactionEnded(
                event.reason,
                event.success,
                event.strategy,
                event.tokens_before,
                event.tokens_after,
                # getattr: a host or session predating the field emits an
                # event without it, same tolerance the token pair had.
                getattr(event, "detail", None),
            )
        )

    def _handle_retry_start(self, event: RetryStartEvent) -> None:
        self._post(RetryStarted(event.attempt, event.error, event.fallback_model))

    def _handle_model_change(self, event: ModelChangeEvent) -> None:
        # No generation guard: the route edge belongs to the session, not to a
        # turn, and a fallback pinned by a stale turn's last request is still
        # the route the next request will take.
        self._post(
            EffectiveModelChanged(
                event.provider,
                event.model_id,
                event.effort,
                event.reason,
                event.is_fallback,
            )
        )

    def _handle_retry_end(self, event: RetryEndEvent) -> None:
        self._post(RetryEnded(event.success))

    # Subagent events ride the PARENT session stream: the child session's own
    # stream is the job manager's problem, and the TUI subscribes to exactly
    # one session. No generation guard here — a child's lifecycle events are
    # not the parent loop's boundary events, so a stale turn cannot supersede
    # them; the job id groups them.
    def _handle_subagent_start(self, event: SubagentStartEvent) -> None:
        self._post(SubagentStarted(event.job_id, event.label, event.agent_id))

    def _handle_subagent_progress(self, event: SubagentProgressEvent) -> None:
        self._post(SubagentProgress(event.job_id, event.label, event.progress))

    def _handle_subagent_end(self, event: SubagentEndEvent) -> None:
        self._post(
            SubagentEnded(
                event.job_id, event.label, event.status, event.result_text, event.error_text
            )
        )

    _HANDLERS = {
        "agent_start": _handle_agent_start,
        "agent_end": _handle_agent_end,
        "turn_start": _handle_turn_start,
        "turn_end": _handle_turn_end,
        "message_start": _handle_message_start,
        "message_update": _handle_message_update,
        "message_end": _handle_message_end,
        "history_delta": _handle_history_delta,
        "tool_call_compose": _handle_tool_compose,
        "tool_execution_start": _handle_tool_start,
        "tool_execution_update": _handle_tool_update,
        "tool_execution_end": _handle_tool_end,
        "notice": _handle_notice,
        "wake_delivered": _handle_wake_delivered,
        "peer_message_delivered": _handle_peer_message_delivered,
        "steering_delivered": _handle_steering_delivered,
        "compaction_start": _handle_compaction_start,
        "compaction_end": _handle_compaction_end,
        "retry_start": _handle_retry_start,
        "retry_end": _handle_retry_end,
        "model_change": _handle_model_change,
        "subagent_start": _handle_subagent_start,
        "subagent_progress": _handle_subagent_progress,
        "subagent_end": _handle_subagent_end,
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

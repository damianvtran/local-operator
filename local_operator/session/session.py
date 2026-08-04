"""Session facade — composes loop + tools + transcript + wake + jobs.

The one object every front end talks to (see ``SessionProtocol``). It wires
the host callbacks of :class:`~local_operator.harness.types.LoopConfig` to
session state and keeps the loop itself ignorant of persistence, wakes, and
event fan-out.

Ported semantics:

- Event handlers receive ``AgentEvent``s in emission order; a handler may be
  sync or async, and one raising never breaks the others (isolated with a
  warning).
- Steering messages interrupt tool batches (``interrupt_mode="immediate"``);
  ``steer()`` is the public injection point.
- Wakes persist as a ``wake_schedules`` custom transcript entry (newest wins)
  and deliver through the prompt path as a user-attributed
  ``wake_prompt`` custom message.
- Compaction is checked after each turn via a LAZY import of
  ``local_operator.compaction.api`` — a missing module degrades to
  no-compaction. Binding wiring: prune tool outputs BEFORE the trigger math,
  trigger on ``compaction_context_tokens``, default threshold
  ``min(int(window * 0.8), 600_000)`` when both knobs are unset, strategy
  resolution with snapcompact preferred for vision models, and the recovery
  band gating auto-continuation.
- ``agent_start``/``agent_end`` carry a per-session monotonic ``generation``
  so UIs can drop a superseded ``agent_end`` arriving after the next start.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentTool,
    ChatRequest,
    CompactionEndEvent,
    CompactionStartEvent,
    CustomMessage,
    EventHandler,
    ImageContent,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEvent,
    StreamTextDelta,
    TextContent,
    ToolContext,
    Usage,
)
from local_operator.harness.wake import (
    WAKE_PROMPT_MESSAGE_TYPE,
    WAKE_SCHEDULES_CUSTOM_TYPE,
    DueWake,
    WakeSchedule,
    WakeScheduler,
    format_wake_delivery_text,
)
from local_operator.session.transcript import Transcript

logger = logging.getLogger(__name__)

#: Self-prompt scheduled after a compaction pass that cleared the recovery
#: band, so the model resumes where the summary left off.
_CONTINUATION_PROMPT = (
    "The conversation context was just compacted into a summary. "
    "Continue the task from where it left off."
)


def _archive_to_json(archive: Any) -> dict[str, Any]:
    """JSON-safe dump of a snapcompact ``Archive``: base64 frames + ISO
    timestamp (``model_dump(mode='json')`` chokes on raw PNG bytes)."""
    return {
        "frames": [base64.b64encode(frame).decode("ascii") for frame in archive.frames],
        "text": archive.text,
        "text_head": archive.text_head,
        "text_tail": archive.text_tail,
        "shape_id": archive.shape_id,
        "truncated_chars": archive.truncated_chars,
        "created_at": archive.created_at.isoformat(),
    }


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Default transcript→LLM rendering.

    ``compaction_summary`` markers become a user message carrying the summary;
    a snapcompact archive in ``preserve_data`` is rendered back into
    text_head → imaged middle → text_tail blocks (base64 ``ImageContent``
    between ``TextContent`` edges). ``wake_prompt`` deliveries become user
    messages of their formatted text; other custom entries are dropped
    (bookkeeping never enters LLM context). ``provider_payload`` rides along
    untouched.
    """
    out: list[Message] = []
    for message in messages:
        if isinstance(message, Message):
            out.append(message)
        elif message.custom_type == "compaction_summary":
            out.append(_render_compaction_marker(message))
        elif message.custom_type == WAKE_PROMPT_MESSAGE_TYPE:
            out.append(
                Message(role="user", content=[TextContent(text=message.details.get("text", ""))])
            )
    return out


def _render_compaction_marker(marker: CustomMessage) -> Message:
    """Render one compaction marker into an LLM-visible message.

    Snapcompact archives replay via ``history_blocks`` (lazy import; any
    failure degrades to the plain-text summary so a malformed archive never
    breaks the turn).
    """
    summary = marker.details.get("summary", "")
    preserve = marker.details.get("preserve_data") or {}
    archive_payload = preserve.get("snapcompact")
    if archive_payload:
        try:
            from local_operator.compaction import snapcompact

            archive = snapcompact.Archive.model_validate(archive_payload)
            content: list[TextContent | ImageContent] = []
            for block in snapcompact.history_blocks(archive):
                if block["kind"] == "text":
                    content.append(TextContent(text=block["text"]))
                elif block["kind"] == "images":
                    for frame_b64 in block["frames"]:
                        content.append(ImageContent(data=frame_b64, mime_type="image/png"))
            if content:
                return Message(role="user", content=content)
        except Exception:
            logger.warning("snapcompact replay failed; falling back to text summary", exc_info=True)
    return Message(
        role="user",
        content=[
            TextContent(
                text="<previous-context-summary>\n" f"{summary}\n" "</previous-context-summary>"
            )
        ],
    )


class Session:
    """The session facade. Satisfies ``SessionProtocol``."""

    def __init__(
        self,
        model: ModelSpec,
        stream_fn: Callable[[ChatRequest, AbortSignal | None], AsyncIterator[StreamEvent]],
        tools: Sequence[AgentTool],
        transcript: Transcript,
        *,
        session_id: str | None = None,
        agent_id: str = "main",
        convert_to_llm: Callable[[list[AgentMessage]], list[Message]] | None = None,
        compaction_settings: Any | None = None,
        yolo: bool = False,
        has_ui: bool = False,
        cwd: str | None = None,
        skill_resolver: Callable[[str], str | None] | None = None,
        request_approval: Callable[[str, str], Awaitable[bool]] | None = None,
        system_blocks_provider: Callable[[], list[str]] | Callable[[], Awaitable[list[str]]],
    ) -> None:
        self._model = model
        self._stream_fn = stream_fn
        self._tools = list(tools)
        self._transcript = transcript
        self._session_id = session_id or transcript.directory.name
        self._agent_id = agent_id
        self._system_blocks_provider = system_blocks_provider
        self._convert_to_llm = convert_to_llm or _default_convert_to_llm
        self._compaction_settings = compaction_settings
        self._yolo = yolo
        self._has_ui = has_ui
        self._cwd = cwd or "."
        self._skill_resolver = skill_resolver
        self._request_approval = request_approval

        self._loop = AgentLoop()
        self._context = LoopContext(
            system_blocks=[],
            messages=list(transcript.build_llm_history()),
            tools=self._tools,
        )
        self._handlers: list[EventHandler] = []
        self._steering_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._aside_thunks: list[Callable[[], AgentMessage | None]] = []
        self._continuation_queue: list[AgentMessage] = []
        self._last_usage: Usage | None = None  # latest provider-reported usage
        self._last_activity_ms: int = 0  # epoch ms; drives idle-flush pruning
        self._generation = 0  # monotonic turn counter for agent_start/end
        self._signal: AbortSignal | None = None
        self._is_streaming = False
        self._turn_lock = asyncio.Lock()  # serializes prompt() and wake deliveries
        self._disposed = False

        self.jobs = AsyncJobManager()
        self._wake = WakeScheduler(
            now=lambda: int(time.time() * 1000),
            deliver=self._deliver_wake,
            persist=self._persist_wake_schedules,
        )
        self._load_wake_schedules()

    # -- identity / state (SessionProtocol) ----------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def model_label(self) -> str:
        return f"{self._model.provider}/{self._model.model_id}"

    @property
    def wake_scheduler(self) -> WakeScheduler:
        """Exposed so the wake tool can list/create/cancel schedules."""
        return self._wake

    # -- driving turns --------------------------------------------------------

    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
        """Run one user turn to completion (awaitable) or raise.

        ``attachments`` is reserved for the wire clients (image blocks); this
        engine turn carries text only.
        """
        if self._is_streaming:
            raise RuntimeError("session is already streaming; use steer() to inject mid-turn")
        await self._prompt_messages([Message.user(text)])

    def steer(self, text: str) -> None:
        """Inject a steering message into the running turn (interrupts tool
        batches at the next boundary)."""
        self._steering_queue.put_nowait(Message.user(text))

    def abort(self, reason: str = "interrupted") -> None:
        """Abort the running turn; the engine emits an aborted agent_end."""
        if self._signal is not None:
            self._signal.abort(reason)

    # -- events ---------------------------------------------------------------

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register an event handler; returns an unsubscribe callable. Sync or
        async handlers are called in registration order; one raising never
        breaks the others."""
        self._handlers.append(handler)

        def _unsubscribe() -> None:
            try:
                self._handlers.remove(handler)
            except ValueError:
                pass

        return _unsubscribe

    async def _emit(self, event: AgentEvent) -> None:
        for handler in list(self._handlers):
            try:
                outcome = handler(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.warning("event handler failed for %s", event.type, exc_info=True)

    # -- turn machinery --------------------------------------------------------

    async def _prompt_messages(self, initial: list[AgentMessage]) -> None:
        """Shared turn runner for user prompts and wake deliveries."""
        if self._disposed:
            raise RuntimeError("session is disposed")
        async with self._turn_lock:
            await self._run_turn(initial)
            await self._drain_continuation()

    async def _run_turn(self, initial: list[AgentMessage]) -> None:
        """One loop run + persistence. Caller holds ``_turn_lock``."""
        self._is_streaming = True
        self._generation += 1  # monotonic; stamped on start AND end events
        self._last_activity_ms = int(time.time() * 1000)
        signal = AbortSignal()
        self._signal = signal
        try:
            for message in initial:
                await self._transcript.append_message(message)

            blocks = self._system_blocks_provider()
            if inspect.isawaitable(blocks):
                blocks = await blocks
            self._context.system_blocks = list(blocks)
            self._context.tool_context = self._build_tool_context()

            config = LoopConfig(
                model=self._model,
                convert_to_llm=self._convert_to_llm,
                stream_fn=self._stream_fn,
                get_steering_messages=self._drain_steering,
                has_steering_messages=lambda: not self._steering_queue.empty(),
                get_aside_messages=self._drain_asides,
                resolve_fallback_tool=None,
                interrupt_mode="immediate",
            )

            new_messages: list[AgentMessage] = []
            async for event in self._loop.run(
                initial, self._context, config, signal, generation=self._generation
            ):
                if isinstance(event, AgentEndEvent):
                    new_messages = list(event.messages)
                await self._emit(event)

            # Track the latest provider usage for compaction trigger math.
            for message in reversed(new_messages):
                if isinstance(message, Message) and message.usage is not None:
                    self._last_usage = message.usage
                    break
            self._last_activity_ms = int(time.time() * 1000)

            # Persist everything the turn produced (initial messages were
            # written before the run).
            for message in new_messages:
                if message in initial:
                    continue
                await self._transcript.append_message(message)

            await self._maybe_compact()
        finally:
            self._signal = None
            self._is_streaming = False

    async def _drain_continuation(self) -> None:
        """Run queued auto-continuation prompts (post-compaction resume) as
        sequential turns inside the same lock hold."""
        while self._continuation_queue and not self._disposed:
            message = self._continuation_queue.pop(0)
            await self._run_turn([message])

    def _build_tool_context(self) -> ToolContext:
        return ToolContext(
            cwd=self._cwd,
            session_id=self._session_id,
            agent_id=self._agent_id,
            has_ui=self._has_ui,
            resolve_internal_url=self._skill_resolver,
            request_approval=None if self._yolo else self._request_approval,
        )

    async def _drain_steering(self) -> list[AgentMessage]:
        """Consume the steering queue. Steering messages are real injected
        turns, so they are persisted here — the loop never returns them in its
        ``new_messages``."""
        messages: list[AgentMessage] = []
        while not self._steering_queue.empty():
            message = self._steering_queue.get_nowait()
            await self._transcript.append_message(message)
            messages.append(message)
        return messages

    async def _drain_asides(self) -> list[Any]:
        thunks = self._aside_thunks
        self._aside_thunks = []
        return list(thunks)

    def queue_aside(self, thunk: Callable[[], AgentMessage | None]) -> None:
        """Queue a lazy aside message for the next injection boundary. The thunk
        is wrapped so a materialized (non-None) message is persisted exactly
        once, at the moment it actually reaches the model."""

        def _wrapped() -> AgentMessage | None:
            message = thunk()
            if message is not None:
                asyncio.ensure_future(self._transcript.append_message(message))
            return message

        self._aside_thunks.append(_wrapped)

    # -- compaction ------------------------------------------------------------

    async def _maybe_compact(self) -> None:
        """Post-turn compaction check; lazy-imports the compaction API so a
        missing module degrades to no-compaction.

        Order (binding orchestrator decisions):

        1. ``prune_tool_outputs`` over the LLM history (in-place blanking of
           superseded/useless tool outputs) BEFORE the trigger math.
        2. Trigger on ``compaction_context_tokens`` (max of provider-reported
           context size and the local estimate).
        3. Default threshold when both knobs are unset:
           ``min(int(context_window * 0.8), 600_000)``.
        4. Strategy resolution: snapcompact for vision models (archive stored
           under ``preserve_data['snapcompact']``), context-full otherwise;
           any snapcompact failure falls back to LLM summarization.
        5. After a successful pass, schedule auto-continue only when the
           residual cleared the recovery band (``residual <= 0.8 * threshold``).
        """
        try:
            from local_operator.compaction import api as compaction_api
        except ImportError:
            return

        settings = self._compaction_settings or compaction_api.CompactionSettings()
        if not settings.enabled:
            return

        strategy = self._resolve_strategy(settings)
        if strategy == "off":
            return
        if settings.threshold_tokens <= 0 and settings.threshold_percent <= 0:
            default_threshold = min(int(self._model.context_window * 0.8), 600_000)
            settings = settings.model_copy(update={"threshold_tokens": default_threshold})

        llm_history = self._convert_to_llm(list(self._context.messages))

        # (1) Prune before deciding: blanked outputs shrink the estimate and
        # may avoid a compaction pass entirely. Mutates messages in place.
        now_ms = int(time.time() * 1000)
        try:
            compaction_api.prune_tool_outputs(llm_history, now_ms, self._last_activity_ms)
        except TypeError:
            pass  # signature in flux; degrade to no pruning

        # (2) Trigger math: prefer the provider's ground-truth context size.
        local_estimate = compaction_api.estimate_messages_tokens(llm_history)
        provider_reported = (
            self._last_usage.context_tokens if self._last_usage is not None else None
        )
        context_tokens = compaction_api.compaction_context_tokens(provider_reported, local_estimate)
        if not compaction_api.should_compact(context_tokens, self._model.context_window, settings):
            return

        cut = compaction_api.find_cut_point(llm_history, settings.keep_recent_tokens)
        if cut is None or cut <= 0:
            return

        await self._emit(CompactionStartEvent(reason="context-window"))
        try:
            to_summarize = llm_history[:cut]
            kept = llm_history[cut:]
            summary, preserve_data = await self._produce_summary(
                compaction_api, to_summarize, strategy
            )
            first_kept_entry_id = kept[0].id
            await self._transcript.append_compaction(
                summary, first_kept_entry_id, context_tokens, preserve_data=preserve_data
            )
            marker_details: dict[str, Any] = {"summary": summary}
            if preserve_data is not None:
                marker_details["preserve_data"] = preserve_data
            marker = CustomMessage(
                custom_type="compaction_summary",
                attribution="system",
                details=marker_details,
            )
            self._context.messages = [marker, *kept]
            await self._emit(CompactionEndEvent(reason="context-window", success=True))

            # (5) Recovery band: only schedule a continuation when the pass
            # actually created headroom (omp issue #2275 anti-thrash).
            if getattr(settings, "auto_continue", False):
                threshold = compaction_api.resolve_threshold_tokens(
                    self._model.context_window, settings
                )
                residual = compaction_api.estimate_messages_tokens(
                    self._convert_to_llm(list(self._context.messages))
                )
                if residual <= compaction_api.RECOVERY_BAND * threshold:
                    self._continuation_queue.append(Message.user(_CONTINUATION_PROMPT))
        except Exception:
            logger.warning("compaction failed", exc_info=True)
            await self._emit(CompactionEndEvent(reason="context-window", success=False))

    def _resolve_strategy(self, settings: Any) -> str:
        """'auto' | 'context-full' | 'snapcompact' | 'off'.

        Uses ``compaction.thresholds.resolve_strategy`` when it has landed;
        until then, an explicit ``strategy`` setting wins and ``auto`` defers
        to ``strategy_for_model``."""
        explicit = getattr(settings, "strategy", "auto")
        if explicit in ("context-full", "snapcompact", "off"):
            return str(explicit)
        try:
            from local_operator.compaction import thresholds as _thresholds

            resolver = getattr(_thresholds, "resolve_strategy", None)
            if resolver is not None:
                return str(resolver(settings, self._model))
        except ImportError:
            pass
        try:
            from local_operator.compaction.snapcompact import strategy_for_model

            return strategy_for_model(self._model)
        except ImportError:
            return "context-full"

    async def _produce_summary(
        self, compaction_api: Any, to_summarize: list[Message], strategy: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Summary text + optional ``preserve_data`` for one compaction pass.

        Snapcompact stores ``{"snapcompact": <archive dump>}`` (JSON-safe:
        base64 frames) and uses the archive text as the summary; any error —
        including ImportError — falls back to the one-shot LLM summary.
        """
        if strategy == "snapcompact":
            try:
                from local_operator.compaction import snapcompact

                archive = snapcompact.compact_to_archive(
                    to_summarize,
                    self._model.provider,
                    self._model.model_id,
                    self._previous_archive_text(),
                )
                return archive.text or " ", {"snapcompact": _archive_to_json(archive)}
            except Exception:
                logger.warning("snapcompact failed; falling back to context-full", exc_info=True)
        summary = await compaction_api.summarize_messages(to_summarize, self._one_shot_complete)
        return summary, None

    def _previous_archive_text(self) -> str | None:
        """The latest compaction's archive text, so snapcompact re-renders from
        accumulated history instead of carrying old PNGs forward."""
        for entry in reversed(self._transcript.entries()):
            if entry.type != "compaction":
                continue
            preserve = entry.payload.get("preserve_data") or {}
            snap = preserve.get("snapcompact")
            if isinstance(snap, dict) and snap.get("text"):
                return str(snap["text"])
            return entry.payload.get("summary")
        return None

    async def _one_shot_complete(self, system: str, prompt: str) -> str:
        """One non-tool provider call used to produce the compaction summary."""
        request = ChatRequest(
            model=self._model,
            system_blocks=[system],
            messages=[Message.user(prompt)],
            tools=[],
            tool_choice="none",
        )
        parts: list[str] = []
        async for event in self._stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
        return "".join(parts)

    # -- wakes -------------------------------------------------------------------

    def _load_wake_schedules(self) -> None:
        details = self._transcript.latest_custom(WAKE_SCHEDULES_CUSTOM_TYPE)
        if not details:
            return
        schedules: list[WakeSchedule] = []
        for raw in details.get("schedules", []):
            try:
                schedules.append(WakeSchedule.model_validate(raw))
            except Exception:
                logger.warning("dropping malformed persisted wake schedule: %r", raw)
        self._wake.load(schedules)

    async def _persist_wake_schedules(self, schedules: list[WakeSchedule]) -> None:
        await self._transcript.append_custom(
            WAKE_SCHEDULES_CUSTOM_TYPE,
            {"schedules": [schedule.model_dump() for schedule in schedules]},
        )

    async def set_wake_schedules(self, schedules: list[WakeSchedule]) -> None:
        """Full-list update from the wake tool: persists then re-arms."""
        await self._wake.update(schedules)

    async def _deliver_wake(self, due: DueWake) -> None:
        """Deliver one fired wake through the prompt path as a user-attributed
        ``wake_prompt`` custom message."""
        text = format_wake_delivery_text(due)
        wake_message = CustomMessage(
            custom_type=WAKE_PROMPT_MESSAGE_TYPE,
            attribution="user",
            details={"wake_id": due.schedule.id, "occurrence": due.occurrence, "text": text},
        )
        if self._is_streaming:
            # Busy: ride the next steering boundary instead of racing the turn.
            self._steering_queue.put_nowait(wake_message)
            return
        asyncio.ensure_future(self._prompt_messages([wake_message]))

    # -- lifecycle ----------------------------------------------------------------

    async def dispose(self) -> None:
        """Cancel jobs, dispose the wake scheduler, flush the transcript."""
        if self._disposed:
            return
        self._disposed = True
        await self.jobs.dispose()
        self._wake.dispose()
        self._transcript.flush()

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
import contextlib
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    Aside,
    AsideResult,
    BrowserSurface,
    ChatRequest,
    CompactionEndEvent,
    CompactionStartEvent,
    Content,
    CustomMessage,
    EventHandler,
    ImageContent,
    LoopConfig,
    Message,
    ModelSpec,
    NoticeEvent,
    StaleAside,
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
from local_operator.session.goal import GoalState
from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.session.naming import ConversationName
from local_operator.session.transcript import Transcript

if TYPE_CHECKING:
    # Type-only: the session must never pull the MCP stack in at import time.
    # It only holds the manager the composition root hands it.
    from local_operator.mcp.manager import McpManager

logger = logging.getLogger(__name__)

#: Self-prompt scheduled after a compaction pass that cleared the recovery
#: band, so the model resumes where the summary left off.
_CONTINUATION_PROMPT = (
    "The conversation context was just compacted into a summary. "
    "Continue the task from where it left off."
)

#: Cap on auto-continuation turns within one user turn
#: MAX_PAUSED_TURN_CONTINUATIONS): a compaction pass that keeps clearing the
#: recovery band must not re-prompt forever.
_MAX_CONTINUATIONS = 8


def _coerce_compaction_settings(settings: Any) -> Any:
    """Defensive coercion (CL-01 belt): a dict-shaped ``compaction_settings``
    is validated into ``CompactionSettings``; already-typed or ``None`` pass
    through. The factory coerces too — this keeps direct constructors (the
    server facade, benchmarks, tests feeding a raw config dict) safe."""
    if settings is None or not isinstance(settings, dict):
        return settings
    try:
        from local_operator.compaction.api import CompactionSettings
    except ImportError:
        return None
    try:
        return CompactionSettings.model_validate(settings)
    except Exception as exc:  # noqa: BLE001 — degrade, never break the session
        logger.warning("Invalid compaction settings dict, using defaults: %s", exc)
        return CompactionSettings()


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
            # Pass the ORIGINAL entry id through the render: the transcript
            # persists custom entries with their CustomMessage.id, so a
            # compaction cut landing on a rendered marker can still locate
            # ``first_kept_entry_id`` on replay.
            out.append(_render_compaction_marker(message, entry_id=message.id))
        elif message.custom_type == WAKE_PROMPT_MESSAGE_TYPE:
            out.append(
                Message(
                    role="user",
                    content=[TextContent(text=message.details.get("text", ""))],
                    id=message.id,
                )
            )
    return out


def _replayed_user_message(content: list[Content], entry_id: str | None) -> Message:
    """Build a replayed user message, preserving its transcript entry id.

    A message rendered from a persisted entry MUST keep that entry's id:
    ``first_kept_entry_id`` references it, so minting a fresh uuid here would
    make replay unable to find the cut point. A message with no originating
    entry keeps the model's default id.
    """
    message = Message(role="user", content=content)
    if entry_id:
        message.id = entry_id
    return message


def _pruned_ids(messages: Sequence[AgentMessage]) -> set[str]:
    """Ids of messages already carrying the pruning pass's ``pruned`` marker.

    Taken BEFORE a prune pass so the pass's effect is a set difference. The
    pass reports only a boolean ``changed``, and re-journalling every
    already-blanked message on every turn would grow the transcript with
    duplicate entries faster than the blanking shrinks it.
    """
    return {
        message.id
        for message in messages
        if isinstance(message, Message) and (message.provider_payload or {}).get("pruned")
    }


def _render_compaction_marker(marker: CustomMessage, entry_id: str | None = None) -> Message:
    """Render one compaction marker into an LLM-visible message. ``entry_id``
    (the marker's transcript entry id) rides onto the rendered message.

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
                return _replayed_user_message(content, entry_id)
        except Exception:
            logger.warning("snapcompact replay failed; falling back to text summary", exc_info=True)
    return _replayed_user_message(
        [
            TextContent(
                text="<previous-context-summary>\n" f"{summary}\n" "</previous-context-summary>"
            )
        ],
        entry_id,
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
        goal_state: GoalState | None = None,
        conversation_name: ConversationName | None = None,
        system_blocks_provider: Callable[[], list[str]] | Callable[[], Awaitable[list[str]]],
    ) -> None:
        self._model = model
        self._stream_fn = stream_fn
        self._tools = list(tools)
        self._transcript = transcript
        self._session_id = session_id or transcript.directory.name
        self._agent_id = agent_id
        # The goal rides the prompt's volatile tail; the holder is shared with
        # the system-blocks provider so an edit applies from the next turn.
        self._goal_state = goal_state if goal_state is not None else GoalState()
        # The conversation's title. A holder rather than a plain string for
        # the same reason the goal is one: the title arrives on a DETACHED
        # naming task after the host already built its status chrome, and
        # both sides must see the same object rather than a stale copy.
        self._conversation_name = (
            conversation_name if conversation_name is not None else ConversationName()
        )
        self._system_blocks_provider = system_blocks_provider
        self._convert_to_llm = convert_to_llm or _default_convert_to_llm
        self._compaction_settings = _coerce_compaction_settings(compaction_settings)
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
        # Host-registered teardown (see add_dispose_hook): resources the
        # composition root owns but the session's lifetime governs.
        self._dispose_hooks: list[Callable[[], Awaitable[None] | None]] = []
        # Set by the composition root when MCP servers are wired in, and read
        # only for diagnostics — the session never drives the manager itself,
        # it just governs its lifetime through a dispose hook.
        self.mcp_manager: McpManager | None = None
        # What that wiring actually achieved, recorded as data so a front end
        # can render it. ``None`` means MCP wiring never ran on this session at
        # all, which is distinct from "it ran and found nothing configured" —
        # the latter is an empty outcome. See session/mcp_status.py for why the
        # record does not live in the mcp package.
        self.mcp_startup: McpStartupOutcome | None = None
        self._aside_thunks: list[Aside] = []
        self._continuation_queue: list[AgentMessage] = []
        self._last_usage: Usage | None = None  # latest provider-reported usage
        self._last_activity_ms: int = 0  # epoch ms; drives idle-flush pruning
        self._generation = 0  # monotonic turn counter for agent_start/end
        # Boundary-event suppression across a post-compaction continuation:
        # `_held_end` parks the loop's agent_end until compaction has decided
        # whether the run continues, `_logical_generation` remembers which
        # agent_start the eventual end belongs to. Both are None outside a run.
        self._held_end: AgentEndEvent | None = None
        self._abort_requested = False  # sticky across the continuation gap
        # Last completed provider request (epoch ms). Distinct from
        # _last_activity_ms: the idle-flush pruning must measure provider-cache
        # age, and stamping turn bookkeeping right before the check made the
        # 90-minute flush dead code.
        self._last_provider_request_ms = 0
        self._seeded = False  # seed_history is once-only, pre-prompt
        self._logical_generation: int | None = None
        self._fallback_tool_resolver: Callable[[str], AgentTool | None] | None = None

        self._disposed = False
        # Session-scoped task group (HC-11): wake deliveries and aside
        # persistence are routed through it so dispose() cancels them
        # deterministically and a delivery after dispose never raises into an
        # unobserved task. A TaskGroup must be ENTERED inside a running loop,
        # so construction only allocates the slot — :meth:`async_init` opens
        # it (code that skips async_init degrades to ensure_future).
        self._tg_stack: contextlib.AsyncExitStack | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._signal: AbortSignal | None = None
        self._is_streaming = False
        self._turn_lock = asyncio.Lock()  # serializes prompt() and wake deliveries
        self._turn_task: asyncio.Task[None] | None = None  # in-flight turn (wake deliveries)

        self.jobs = AsyncJobManager()
        self._wake = WakeScheduler(
            now=lambda: int(time.time() * 1000),
            deliver=self._deliver_wake,
            persist=self._persist_wake_schedules,
        )
        self._load_wake_schedules()
        # Owned here, not by the browser tool, for the same reason the wake
        # scheduler is: _build_tool_context runs at the start of EVERY turn, so
        # a handle the tool stored on the ToolContext lived exactly one turn.
        # The visible symptoms were that multi-turn browsing could not work at
        # all ("open X", then next message "click Y" → "no browser surface
        # open") and that every turn which opened a browser stranded a cmux tab
        # the agent could never close. dispose() closes whatever is still open.
        self._browser = BrowserSurface()

    async def async_init(self) -> None:
        """Async second half of construction.

        Opens the session-scoped task group and re-arms the wake scheduler:
        a scheduler armed during sync ``__init__`` (no running loop yet)
        could not create its timer — one ``pump()`` here fires overdue wakes
        and arms properly (see ``WakeScheduler.needs_rearm``). Safe to call
        more than once; sessions that skip it degrade to ``ensure_future``
        for background work.
        """
        if self._tg_stack is None and not self._disposed:
            stack = contextlib.AsyncExitStack()
            await stack.__aenter__()
            try:
                self._task_group = await stack.enter_async_context(asyncio.TaskGroup())
            except BaseException:
                await stack.aclose()
                raise
            self._tg_stack = stack
        await self._wake.pump()

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
    def model(self) -> ModelSpec:
        """The spec every provider call is built from."""
        return self._model

    def set_model(self, model: ModelSpec) -> None:
        """Swap the model spec mid-session.

        The loop reads ``config.model`` fresh on every turn, so the new spec
        takes effect from the next turn onward. Hosts use this for per-request
        overrides that are not part of the agent record — the FastAPI server
        applies ``ChatRequest.options`` (temperature / top_p) this way, since
        sampling rides on the spec (see ``model/configure.build_model_spec``).
        """
        self._model = model

    @property
    def goal(self) -> str:
        """The session's standing objective ("" when unset)."""
        return self._goal_state.text

    def set_goal(self, text: str) -> str:
        """Set (or clear, with an empty string) the standing objective.

        Returns what was actually stored (trimmed and length-capped). The
        goal rides the system prompt's volatile tail, so it applies from the
        next turn and only invalidates that tail — never the cached prefix.
        """
        return self._goal_state.set(text)

    @property
    def conversation_name(self) -> str:
        """The conversation's title ("" until one is set or generated)."""
        return self._conversation_name.text

    @property
    def conversation_name_state(self) -> ConversationName:
        """The title holder itself — hosts need the ``user_set`` precedence
        flag and the once-only request latch, not just the string."""
        return self._conversation_name

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        """Name the conversation; returns the title in force afterwards.

        ``user_set=True`` (an explicit rename) wins permanently: a generated
        title landing later is discarded rather than overwriting a name the
        user chose. Auto-naming passes ``user_set=False``.
        """
        return self._conversation_name.set(text, user_set=user_set)

    @property
    def wake_scheduler(self) -> WakeScheduler:
        """Exposed so the wake tool can list/create/cancel schedules."""
        return self._wake

    # -- driving turns --------------------------------------------------------
    async def prompt(self, text: str) -> None:
        """Run one user turn to completion (awaitable) or raise.

        Reentrancy: ``_turn_lock`` is consulted FIRST — if a live turn (user
        prompt or wake delivery) holds it, a concurrent ``prompt`` is
        rejected outright instead of queueing behind it. ``_is_streaming`` is
        then re-checked under the lock to close the race where streaming was
        set between the lock probe and the acquire.
        """
        if self._disposed:
            raise RuntimeError("session is disposed")
        if self._turn_lock.locked():
            raise RuntimeError("session is already streaming; use steer() to inject mid-turn")
        await self._turn_lock.acquire()
        try:
            # A fresh user prompt supersedes any earlier interrupt request.
            self._abort_requested = False
            if self._is_streaming:
                raise RuntimeError("session is already streaming; use steer() to inject mid-turn")
            await self._run_turn_pipeline([Message.user(text)])
        finally:
            self._turn_lock.release()

    async def seed_history(self, messages: list[Message]) -> None:
        """Prime the conversation from a host-supplied history.

        The server facade uses this for the two paths where the transcript is
        not already the history source: stateless ``/v1/chat`` (the caller's
        ``context`` array) and agent chat with ``persist_conversation=false``
        (the registry's stored conversation). Without it the provider sees
        the bare prompt while the response envelope echoes history the model
        never read.

        Once-only and pre-prompt: a no-op when the context already carries
        messages (transcript replay populated them) or after the first turn.
        Messages are persisted as replayed entries so later turns, pruning and
        compaction treat them like any other history.
        """
        if self._seeded or self._context.messages or self._is_streaming:
            return
        self._seeded = True
        for message in messages:
            await self._transcript.append_message(message)
            self._context.messages.append(message)

    def steer(self, text: str) -> None:
        """Inject a steering message into the running turn (interrupts tool
        batches at the next boundary)."""
        self._steering_queue.put_nowait(Message.user(text))

    def set_approval_handler(self, handler: Callable[[str, str], Awaitable[bool]] | None) -> None:
        """Install the host's tool-approval gate (see SessionProtocol).

        Read when the per-turn tool context is built rather than captured once,
        so a front end that installs its own gate after the session is already
        constructed (the TUI resolves its session in a worker, well after the
        factory ran) governs every tool call from the next one onward.
        """
        self._request_approval = handler

    def abort(self, reason: str = "interrupted") -> None:
        """Abort the running turn; the engine emits an aborted agent_end.

        Sticky: between a turn and its post-compaction continuation the live
        signal is None, so a Ctrl+C in that window would otherwise be dropped
        and the agent would run the continuation the user just tried to stop.
        The flag is checked at the top of the continuation drain and pre-aborts
        the next turn's signal.
        """
        self._abort_requested = True
        if self._signal is not None:
            self._signal.abort(reason)

    # -- live tool refresh (MCP late-connect / reconnect) ---------------------

    def refresh_tools(self, tools: Sequence[AgentTool]) -> None:
        """Replace the full tool inventory mid-session.

        THE committed hook for MCP ``set_on_tools_changed`` (orchestrator
        MCP-20): the caller passes the merged set (builtins + all currently
        loaded MCP tools) and this swaps it in. The loop reads
        ``context.tools`` fresh on every model call and every tool resolution,
        so the new set is effective from the NEXT model call onward — and even
        mid-turn at the next tool batch — with no restart.
        """
        self._tools = list(tools)
        self._context.tools = self._tools

    def set_fallback_tool_resolver(
        self, resolver: Callable[[str], AgentTool | None] | None
    ) -> None:
        """Install a resolver for tool names NOT in the inventory (deferred /
        lazy MCP tools). Wired to ``LoopConfig.resolve_fallback_tool`` so the
        loop can dispatch calls to tools not yet materialized. ``None`` clears
        it."""
        self._fallback_tool_resolver = resolver

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
        """Shared turn runner for wake deliveries (prompt() owns its own lock
        handling so it can REJECT reentrants instead of queueing)."""
        if self._disposed:
            raise RuntimeError("session is disposed")
        async with self._turn_lock:
            await self._run_turn_pipeline(initial)

    async def _run_turn_pipeline(self, initial: list[AgentMessage]) -> None:
        """One turn + its auto-continuations. Caller holds ``_turn_lock``.

        A post-compaction continuation is a CONTINUATION of the same logical
        run, not a new one: compaction happens after the loop has already
        yielded its ``agent_end``, so forwarding that end (and the next run's
        ``agent_start``) would tell every UI the task finished and then started
        again. The pipeline therefore holds the boundary events and emits
        exactly one ``agent_start`` / ``agent_end`` pair per user prompt, with
        ``compaction_start`` / ``compaction_end`` and further turns in between.
        The generation stamp on the emitted end is the one from the start that
        opened the run, so the TUI's supersede guard still pairs them.
        """
        self._turn_task = asyncio.current_task()
        try:
            await self._run_turn(initial)
            await self._drain_continuation()
        finally:
            self._turn_task = None
            await self._flush_held_end()

    async def _flush_held_end(self) -> None:
        """Emit the boundary event the pipeline was holding, if any."""
        held = self._held_end
        self._held_end = None
        if held is None:
            return
        generation = self._logical_generation or held.generation
        self._logical_generation = None
        await self._emit(
            held
            if held.generation == generation
            else held.model_copy(update={"generation": generation})
        )

    async def _run_turn(self, initial: list[AgentMessage]) -> None:
        """One loop run + persistence. Caller holds ``_turn_lock``."""
        if self._wake.needs_rearm:
            # HC-20: the scheduler could not arm without a running loop at
            # construction; the first turn (with a loop) re-arms via pump().
            await self._wake.pump()
        self._is_streaming = True
        self._generation += 1  # monotonic; stamped on start AND end events
        self._last_activity_ms = int(time.time() * 1000)
        signal = AbortSignal()
        self._signal = signal
        if self._abort_requested:
            # A Ctrl+C landed in the gap between turns (signal was None);
            # honour it on the fresh signal instead of running the
            # continuation the user tried to stop.
            signal.abort("interrupted")
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
                resolve_fallback_tool=self._fallback_tool_resolver,
                interrupt_mode="immediate",
            )

            new_messages: list[AgentMessage] = []
            async for event in self._loop.run(
                initial, self._context, config, signal, generation=self._generation
            ):
                if isinstance(event, AgentStartEvent):
                    if self._logical_generation is None:
                        # First run of the pipeline: this start opens the run
                        # the UI sees, and its generation stamps the eventual
                        # end. Continuation runs re-enter here and are silent.
                        self._logical_generation = event.generation
                        await self._emit(event)
                    continue
                if isinstance(event, AgentEndEvent):
                    new_messages = list(event.messages)
                    if event.aborted or event.error:
                        # A failed or interrupted run is a real boundary: never
                        # hold it behind a compaction that may not happen. The
                        # end is re-stamped with the generation of the start
                        # the UI saw (the continuation's own stamp would split
                        # the pair), and the continuation queue is dropped so
                        # no further run can open a second boundary inside the
                        # same prompt (§B).
                        self._abort_requested = True
                        self._continuation_queue.clear()
                        self._held_end = None
                        generation = self._logical_generation or event.generation
                        self._logical_generation = None
                        await self._emit(
                            event
                            if event.generation == generation
                            else event.model_copy(update={"generation": generation})
                        )
                    else:
                        # Held until _maybe_compact has had its say; the
                        # pipeline flushes it if no continuation is queued.
                        self._held_end = event
                    continue
                await self._emit(event)

            # Track the latest provider usage for compaction trigger math.
            for message in reversed(new_messages):
                if isinstance(message, Message) and message.usage is not None:
                    self._last_usage = message.usage
                    break
            self._last_activity_ms = int(time.time() * 1000)
            # The run just completed provider round-trips; the idle flush
            # measures provider-cache age from this stamp, not turn
            # bookkeeping (which would always read ~0 and kill the flush).
            self._last_provider_request_ms = int(time.time() * 1000)

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
        sequential turns inside the same lock hold, capped at
        ``_MAX_CONTINUATIONS`` — the remainder is dropped with a warning
        notice so a thrashing recovery band cannot re-prompt forever."""
        continuations = 0
        while self._continuation_queue and not self._disposed:
            if self._abort_requested:
                # Ctrl+C landed in the gap between turns; the continuation
                # the user interrupted must not run.
                self._continuation_queue.clear()
                return
            if continuations >= _MAX_CONTINUATIONS:
                dropped = len(self._continuation_queue)
                self._continuation_queue.clear()
                logger.warning(
                    "auto-continuation cap (%d) reached; dropping %d queued continuation(s)",
                    _MAX_CONTINUATIONS,
                    dropped,
                )
                await self._emit(
                    NoticeEvent(
                        text=(
                            f"Auto-continuation limit ({_MAX_CONTINUATIONS}) reached; " "stopping."
                        ),
                        kind="warning",
                    )
                )
                return
            message = self._continuation_queue.pop(0)
            await self._run_turn([message])
            continuations += 1

    def _spawn_background(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Route a fire-and-forget coroutine through the session task group
        when one is open (wake deliveries, aside persistence); otherwise fall
        back to ``ensure_future``. Every spawned task is tracked so
        :meth:`dispose` can cancel and await it. After dispose nothing is
        spawned, so a late wake delivery cannot raise into an unobserved task.

        The coroutine is wrapped so its failure is logged, never raised: an
        exception escaping into a TaskGroup would cancel every sibling task.
        """
        if self._disposed:
            return

        async def _guarded() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("background session task failed", exc_info=True)

        if self._task_group is not None:
            task = self._task_group.create_task(_guarded())
        else:
            task = asyncio.ensure_future(_guarded())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _build_tool_context(self) -> ToolContext:
        # This context is REBUILT on every turn, so anything that must outlive
        # a turn is owned by the session and injected here. ``wake_scheduler``
        # is a declared ToolContext field: the wake tool's createIf check and
        # executor read it off the context, and a session without a scheduler
        # must not advertise the tool at all. ``browser`` is the same shape —
        # the surface handle has to survive to the next turn for browsing to
        # work, and for teardown to be able to close the tab.
        return ToolContext(
            cwd=self._cwd,
            session_id=self._session_id,
            agent_id=self._agent_id,
            has_ui=self._has_ui,
            resolve_internal_url=self._skill_resolver,
            request_approval=None if self._yolo else self._request_approval,
            wake_scheduler=self._wake,
            browser=self._browser,
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

    async def _drain_asides(self) -> list[Aside]:
        """Drain queued aside thunks (the loop materializes them at the
        injection boundary and fires commit/discard hooks)."""
        thunks = self._aside_thunks
        self._aside_thunks = []
        return list(thunks)

    def queue_aside(self, thunk: Callable[[], AsideResult]) -> None:
        """Queue a lazy aside message for the next injection boundary. The thunk
        is wrapped so a materialized (non-None, non-stale) message is
        persisted exactly once, at the moment it actually reaches the model.
        A :class:`StaleAside` result passes through unpersisted; the loop
        fires its ``on_discard``."""

        def _wrapped() -> AsideResult:
            message = thunk()
            if message is not None and not isinstance(message, StaleAside):
                self._spawn_background(self._transcript.append_message(message))
            return message

        self._aside_thunks.append(_wrapped)

    # -- compaction ------------------------------------------------------------

    async def _journal_prunes(
        self, llm_history: Sequence[AgentMessage], pruned_before: set[str]
    ) -> None:
        """Persist the blanking that ``prune_tool_outputs`` just did in memory.

        Pruning is the one place where the live context and the transcript
        drift apart: the session throws away a multi-kilobyte tool output,
        the transcript keeps it, and the next resume replays the original
        back into the prompt — so a resumed session costs MORE per turn than
        the session it resumed. Journalling closes that gap and, once the
        dead bytes are worth a rewrite, ``compact_file`` takes them off disk
        too. Best-effort: a transcript write that fails must never abort the
        turn's compaction, which is load-bearing where this is an optimisation.
        """
        newly = [
            message
            for message in llm_history
            if isinstance(message, Message)
            and message.id not in pruned_before
            and (message.provider_payload or {}).get("pruned")
        ]
        if not newly:
            return
        try:
            for message in newly:
                await self._transcript.append_prune(message.id, message.text)
            await self._transcript.compact_file()
        except OSError as exc:
            logger.warning("could not journal pruned tool outputs: %s", exc)

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
        # The idle flush compares against the last PROVIDER request, not turn
        # bookkeeping, so a genuinely idle session reclaims the warm region.
        now_ms = int(time.time() * 1000)
        pruned_before = _pruned_ids(llm_history)
        try:
            compaction_api.prune_tool_outputs(llm_history, now_ms, self._last_provider_request_ms)
        except (ImportError, AttributeError):
            pass  # optional pruning hook absent; degrade to no pruning
        await self._journal_prunes(llm_history, pruned_before)

        # (2) Trigger math: prefer the provider's ground-truth context size.
        provider_reported = (
            self._last_usage.context_tokens if self._last_usage is not None else None
        )

        # Cheap proof first. ``should_compact`` is strictly monotonic in
        # context_tokens and ``compaction_context_tokens`` is monotonic in the
        # local estimate, so a rigorous UPPER bound that already fails the
        # threshold test proves the exact estimate fails it too — same early
        # return, same observable behaviour. This matters because the first
        # exact estimate in a process loads tiktoken's cl100k_base table
        # (~84 ms, ~43.6 MB RSS, measured with scripts/bench_base_overhead.py),
        # and compaction runs on EVERY turn while the typical session never
        # comes near its threshold — so every short run was buying a 43.6 MB
        # tokenizer to be told "no".
        bound = compaction_api.messages_tokens_upper_bound(llm_history)
        if not compaction_api.should_compact(
            compaction_api.compaction_context_tokens(provider_reported, bound),
            self._model.context_window,
            settings,
        ):
            return

        local_estimate = compaction_api.estimate_messages_tokens(llm_history)
        context_tokens = compaction_api.compaction_context_tokens(provider_reported, local_estimate)
        if not compaction_api.should_compact(context_tokens, self._model.context_window, settings):
            return

        cut = compaction_api.find_cut_point(llm_history, settings.keep_recent_tokens)
        if cut is None or cut <= 0:
            return

        # The kept window must start at an entry the transcript can replay:
        # first_kept_entry_id is persisted and matched on resume, and a cut
        # whose first kept message has no transcript entry (a converter-minted
        # id) would make replay drop the whole kept window silently.
        entry_ids = {entry.id for entry in self._transcript.entries()}
        if llm_history[cut].id not in entry_ids:
            logger.warning(
                "compaction cut rejected: kept[0].id %s is not a transcript entry",
                llm_history[cut].id,
            )
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
            # actually created headroom (an anti-thrash guard).
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
        base64 frames) and renders the frames back into context through the
        converter. The TEXT slot must therefore be a real short summary —
        never ``archive.text``: the archive is the full bounded history, and
        replaying it as text while the frames are dropped means the pass
        reduces nothing and re-fires on the next turn. Any error — including
        ImportError — falls back to the one-shot LLM summary.
        """
        if strategy == "snapcompact":
            try:
                from local_operator.compaction import snapcompact

                archive = snapcompact.compact_to_archive(
                    to_summarize,
                    self._model.provider,
                    self._model.model_id,
                    self._previous_archive_text(),
                    context_window=self._model.context_window,
                )
                # The frames are the durable record; the text slot is a
                # compact digest for hosts that render summaries as text.
                summary = await compaction_api.summarize_messages(
                    to_summarize, self._one_shot_complete
                )
                return summary or " ", {"snapcompact": _archive_to_json(archive)}
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

    async def complete_once(self, system: str, prompt: str) -> str:
        """One non-tool provider call, exposed for host-side helpers.

        Hosts need the session's configured provider and credentials for
        small side errands — conversation auto-naming is the first — and
        rebuilding a client from the spec would duplicate the whole auth
        cascade. The call carries no tools, no history and no abort signal:
        it is not a turn, must not appear in the transcript, and must never
        be awaited on the turn's critical path.
        """
        return await self._one_shot_complete(system, prompt)

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
        self._spawn_background(self._prompt_messages([wake_message]))

    # -- lifecycle ----------------------------------------------------------------

    def add_dispose_hook(self, hook: Callable[[], Awaitable[None] | None]) -> None:
        """Register teardown that runs after the session's own dispose.

        The composition root owns resources the session never created — MCP
        connections, the auth store's SQLite handle, the shared HTTP client —
        and every front end calls ``dispose`` exactly once, so this is the one
        place to hang them. Hooks run in registration order, and one that
        raises is logged rather than propagated: teardown must never mask the
        dispose that triggered it.
        """
        self._dispose_hooks.append(hook)

    async def _close_browser_surface(self) -> None:
        """Close a cmux browser surface the agent left open.

        The surface is session-scoped (see ``BrowserSurface``), so nothing else
        will ever close it: the model is not guaranteed to call ``browser
        close``, and the handle dies with the process.

        The tool layer is imported HERE rather than at module scope: this is a
        teardown-only call, and ``tools.builtin`` is otherwise absent from the
        session's import graph. Failures are logged, never raised — a terminal
        emulator that will not answer must not be able to break dispose.

        Bounded, because ``_run_cmux`` allows a cmux call 30 s and a wedged
        socket is exactly the state a session is likely to be torn down in.
        Cancelling the wait kills the cmux child (``_run_cmux`` handles
        ``CancelledError``), so the timeout does not trade a stranded tab for an
        orphaned process.
        """
        if not self._browser.surface_id:
            return
        try:
            from local_operator.tools.builtin import close_browser_surface

            problem = await asyncio.wait_for(close_browser_surface(self._browser), timeout=5.0)
            if problem:
                logger.warning("could not close browser surface: %s", problem)
        except Exception:
            logger.warning("closing the browser surface failed", exc_info=True)

    async def dispose(self) -> None:
        """Abort any in-flight turn, close the browser surface, cancel
        background work, dispose jobs and the wake scheduler, flush the
        transcript, then run dispose hooks.

        Order matters: the running turn is aborted and AWAITED (bounded)
        before anything is torn down, so its persistence and event emission
        complete against live objects; background wake deliveries and aside
        writes are cancelled next; the task group is closed last.
        """
        if self._disposed:
            return
        self._disposed = True
        try:
            # HC-14: abort the in-flight turn and await its completion (bounded)
            # before flushing — its persistence must land on a live transcript.
            turn = self._turn_task
            if turn is not None and not turn.done() and self._signal is not None:
                self.abort("session disposed")
                try:
                    await asyncio.wait_for(asyncio.shield(turn), timeout=5.0)
                except BaseException:  # noqa: BLE001 — dispose must always proceed
                    pass
            # The browser surface is session-scoped and lives in the user's own
            # cmux pane, so an unclosed one is a tab THEY have to close by hand.
            # After the turn has stopped (so nothing is mid-navigation on it)
            # and before the task group closes, since this awaits a subprocess.
            await self._close_browser_surface()
            # HC-11: cancel tracked background tasks (wake deliveries, aside
            # persistence), then close the session task group.
            for task in list(self._background_tasks):
                if not task.done():
                    task.cancel()
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            if self._tg_stack is not None:
                with contextlib.suppress(Exception):
                    await self._tg_stack.aclose()
                self._tg_stack = None
            self._task_group = None
            await self.jobs.dispose()
            self._wake.dispose()
            self._transcript.flush()
        finally:
            # ``finally``: host-owned resources must be released even when the
            # session's own teardown blew up part way through.
            for hook in self._dispose_hooks:
                try:
                    outcome = hook()
                    if inspect.isawaitable(outcome):
                        await outcome
                except Exception:
                    logger.warning("session dispose hook failed", exc_info=True)

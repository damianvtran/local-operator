"""Core harness types — the binding contract between all rewrite streams.

This module is the Python port of the ``packages/agent`` type surface in
oh-my-pi (omp). It deliberately knows NOTHING about sessions, providers,
persistence, skills, or UI. Every other stream programs against these types:

- ``Message`` / ``CustomMessage`` — LLM-visible conversation entries.
- ``ToolCall`` / ``ToolResult`` / ``AgentTool`` — the tool protocol.
- ``AgentEvent`` — the ONLY boundary between engine and UI (TUI, print mode,
  server websockets all subscribe to these).
- ``LoopConfig`` — the callback bundle injected into the loop. The loop never
  imports session or provider code; everything is a callback here.
- ``ModelSpec`` / ``ChatRequest`` / ``StreamEvent`` — the provider wire
  contract implemented by ``local_operator.providers.clients``.

Design notes carried over from omp:

- Messages carry an optional ``provider_payload`` for provider-native replay
  data (e.g. OpenAI Responses ids, Anthropic encrypted thinking). It rides
  through history untouched and is consumed only by wire clients.
- ``CustomMessage`` is the extension point for host-authored entries
  (compaction summaries, skill prompts, wake deliveries). It renders to an
  LLM-visible message via the session's ``convert_to_llm`` and NEVER goes to
  a provider raw.
- Aside commit/discard: omp attaches symbols to message objects; here they
  are explicit optional callables excluded from serialization
  (``compare=False``), invoked by the loop when an aside message is actually
  injected (commit) or dropped as stale (discard).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    """A plain text block inside a message."""

    model_config = ConfigDict(frozen=False)

    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(BaseModel):
    """An image block. ``data`` is base64-encoded bytes of ``mime_type``."""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = "image/png"


Content = TextContent | ImageContent


# ---------------------------------------------------------------------------
# Tool calls and results
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """One requested tool invocation as emitted by the model."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Raw JSON argument string when the provider gives us one; wire clients
    # replay this verbatim for providers that require it.
    raw_arguments: str | None = None


class ToolResult(BaseModel):
    """The outcome of executing one ``ToolCall``.

    ``is_error`` marks a NON-throwing failure that should go back to the model
    as a normal tool result (never raise into the loop for model-recoverable
    errors). ``useless`` flags a contextually worthless result (zero-match
    search, timed-out wait) that compaction may elide once consumed; it must
    never be set together with ``is_error`` (errors win).
    """

    tool_call_id: str
    tool_name: str = ""
    content: list[Content] = Field(default_factory=list)
    details: Any = None  # structured payload for renderers/logs, not serialized to providers
    is_error: bool = False
    useless: bool = False

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

MessageRole = Literal["user", "assistant", "tool"]


class Message(BaseModel):
    """One LLM-visible message.

    Identity matters: compaction memoizes token estimates per message object.
    Mutating a message in place (pruning, streaming finalize) MUST call
    ``invalidate_message_cache`` on the compaction cache — see
    ``local_operator.compaction``.
    """

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: list[Content] = Field(default_factory=list)
    # assistant only: requested tool calls for this turn
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # tool only: which call this result answers
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    stop_reason: str | None = None  # stop | length | toolUse | error | aborted
    usage: "Usage | None" = None
    # Provider-native replay payload (opaque to the harness). NOTE: the loop
    # stores harness bookkeeping under ``provider_payload["details"]`` (tool
    # result metadata for compaction) — wire clients MUST NOT replay that key
    # to providers; it is not provider data.
    provider_payload: dict[str, Any] | None = None
    # Stable id for transcript entries and cache memoization.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @staticmethod
    def user(text: str, **extra: Any) -> "Message":
        return Message(role="user", content=[TextContent(text=text)], **extra)

    @staticmethod
    def assistant(text: str = "", **extra: Any) -> "Message":
        return Message(role="assistant", content=[TextContent(text=text)] if text else [], **extra)

    @staticmethod
    def tool_result(result: ToolResult) -> "Message":
        return Message(
            role="tool",
            content=list(result.content),
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            is_error=result.is_error,
        )


class Usage(BaseModel):
    """Token accounting reported by a provider (or estimated locally)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    context_tokens: int | None = None  # provider-reported full context size if given

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CustomMessage(BaseModel):
    """Host-authored transcript entry that renders into LLM context via
    ``convert_to_llm``. Subclass-free extension: ``custom_type`` discriminates
    (``"compaction_summary"``, ``"skill_prompt"``, ``"wake_prompt"``,
    ``"handoff"``, ...), ``details`` carries the typed payload.

    ``id`` is the stable transcript entry id: the transcript persists it
    verbatim (never mints a new one) and ``convert_to_llm`` must carry it
    onto the rendered message, so ``first_kept_entry_id`` can reference a
    rendered custom entry and replay still finds it.

    The two callables are aside commit/discard hooks (see module docstring);
    they are never serialized. ``on_commit`` fires when the message is
    actually injected into context; ``on_discard`` fires when the aside is
    dropped as stale at injection time.
    """

    model_config = ConfigDict(extra="allow")

    custom_type: str
    attribution: Literal["user", "agent", "system"] = "system"
    details: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    on_commit: Callable[[], None] | None = Field(default=None, exclude=True, json_schema_extra={'compare': False})
    on_discard: Callable[[], None] | None = Field(default=None, exclude=True, json_schema_extra={'compare': False})


class StaleAside:
    """Returned by an aside thunk when its payload is stale at injection
    time. Carries the originating :class:`CustomMessage` so the loop can fire
    its ``on_discard`` hook; the message itself is never injected. (A plain
    ``None`` thunk result is dropped silently — producers that need the
    discard receipt must return this instead.)"""

    __slots__ = ("message",)

    def __init__(self, message: CustomMessage) -> None:
        self.message = message


AgentMessage = Message | CustomMessage


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class AgentToolUpdate(BaseModel):
    """A partial result streamed from a running tool."""

    content: list[Content] = Field(default_factory=list)
    details: Any = None


class ToolContext(BaseModel):
    """Minimal host-provided context handed to tool execution.

    Kept tiny on purpose (omp's 100-field ToolSession is a symptom of a
    9000-line session class); grow by demand.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    cwd: str = "."
    session_id: str = ""
    agent_id: str = ""
    has_ui: bool = False
    # Resolver hook for internal URLs (``skill://...``); returns content or
    # None when the URL is not handled. Installed by the session.
    resolve_internal_url: Callable[[str], str | None] | None = None
    # Approval callback: returns True when the user approved. Tools with an
    # approval tier call this before mutating side effects.
    request_approval: Callable[[str, str], Awaitable[bool]] | None = None


ToolExecuteFn = Callable[
    [str, dict[str, Any], "AbortSignal | None", Callable[[AgentToolUpdate], None] | None, ToolContext],
    Awaitable[ToolResult],
]


class AgentTool(BaseModel):
    """A tool the model can call.

    ``parameters`` is a JSON Schema object (pydantic model's
    ``model_json_schema()`` output). ``concurrency`` controls batch
    scheduling: ``"shared"`` tools run in parallel, ``"exclusive"`` alone.
    ``interruptible`` tools may be aborted mid-run to deliver steering.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    label: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    approval_tier: Literal["read", "write", "exec"] = "exec"
    concurrency: Literal["shared", "exclusive"] = "shared"
    interruptible: bool = False
    hidden: bool = False
    execute: ToolExecuteFn = Field(exclude=True)


# ---------------------------------------------------------------------------
# Abort signal
# ---------------------------------------------------------------------------


class AbortSignal:
    """asyncio-flavored AbortSignal. Composable via :meth:`any_of`.

    The loop and every tool receive one; aborting sets the event, and long
    operations ``await signal.wait()``-race their work against it.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None
        # Watcher tasks created by ``any_of``; kept so they can be cancelled
        # when the combined signal fires or is no longer needed (otherwise
        # they leak for the lifetime of the watched signals).
        self._watchers: set[asyncio.Task[None]] = set()

    def abort(self, reason: str = "aborted") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()
        self._cancel_watchers()

    def cancel(self) -> None:
        """Cancel any watcher tasks without aborting (the combined signal is
        no longer needed — e.g. the run that wired it has ended)."""
        self._cancel_watchers()

    def _cancel_watchers(self) -> None:
        watchers, self._watchers = self._watchers, set()
        for task in watchers:
            if not task.done():
                task.cancel()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def watchers(self) -> tuple[asyncio.Task[None], ...]:
        return tuple(self._watchers)

    async def wait(self) -> None:
        await self._event.wait()

    @staticmethod
    def any_of(*signals: "AbortSignal") -> "AbortSignal":
        """Combine signals: aborts when any input aborts. Watcher task
        references live on the combined signal and are cancelled when it
        fires or :meth:`cancel` is called. With no running event loop the
        watchers cannot be created — return an already-aborted signal rather
        than silently dropping aborts."""
        combined = AbortSignal()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            combined.abort("no running event loop")
            return combined

        for sig in signals:
            if sig.aborted:
                combined.abort(sig.reason or "aborted")
                return combined

        async def _watch(sig: "AbortSignal") -> None:
            await sig.wait()
            combined.abort(sig.reason or "aborted")

        for sig in signals:
            task = loop.create_task(_watch(sig))
            combined._watchers.add(task)
            task.add_done_callback(combined._watchers.discard)
        return combined


# ---------------------------------------------------------------------------
# Events — the engine→UI boundary
# ---------------------------------------------------------------------------


class AgentEvent(BaseModel):
    """Base event. ``type`` discriminates; UIs match exhaustively."""

    model_config = ConfigDict(extra="allow")

    type: str


class AgentStartEvent(AgentEvent):
    type: Literal["agent_start"] = "agent_start"
    # Per-session monotonic turn counter; lets UIs drop a superseded
    # agent_end that arrives after the next agent_start.
    generation: int = 0


class AgentEndEvent(AgentEvent):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)
    aborted: bool = False
    error: str | None = None
    generation: int = 0


class TurnStartEvent(AgentEvent):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(AgentEvent):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)


class MessageStartEvent(AgentEvent):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(AgentEvent):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    delta: str = ""  # incremental text for this update (UIs should append, not re-read)


class MessageEndEvent(AgentEvent):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(AgentEvent):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    intent: str | None = None


class ToolExecutionUpdateEvent(AgentEvent):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    partial_result: AgentToolUpdate


class ToolExecutionEndEvent(AgentEvent):
    """A tool finished. ``is_error`` mirrors ``result.is_error``.

    The flag is kept as a serialized field because UI clients and the JSON
    exec stream read it directly, but it is NOT an independent input: a
    producer that sets only ``result.is_error`` (or only the flag) would
    otherwise ship an event whose two halves disagree, and a UI reading the
    flag renders a failed tool as a success. The validator ORs them so the
    two can never drift.
    """

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool = False

    @model_validator(mode="after")
    def _sync_error_flag(self) -> "ToolExecutionEndEvent":
        if self.result.is_error and not self.is_error:
            object.__setattr__(self, "is_error", True)
        return self


class NoticeEvent(AgentEvent):
    type: Literal["notice"] = "notice"
    text: str
    kind: Literal["info", "warning", "error"] = "info"


class CompactionStartEvent(AgentEvent):
    type: Literal["compaction_start"] = "compaction_start"
    reason: str


class CompactionEndEvent(AgentEvent):
    type: Literal["compaction_end"] = "compaction_end"
    reason: str
    success: bool


class RetryStartEvent(AgentEvent):
    type: Literal["retry_start"] = "retry_start"
    attempt: int
    error: str
    fallback_model: str | None = None


class RetryEndEvent(AgentEvent):
    type: Literal["retry_end"] = "retry_end"
    success: bool


EventHandler = Callable[[AgentEvent], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# Loop configuration — the host extension surface
# ---------------------------------------------------------------------------


class LoopConfig(BaseModel):
    """Everything the loop needs from its host, injected as callbacks.

    Python port of omp's ``AgentLoopConfig``. Only ``convert_to_llm`` and a
    model streamer are required; everything else has a neutral default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # The model to call (see providers registry).
    model: "ModelSpec"

    # Required: render transcript messages (incl. custom entries) into the
    # LLM-visible list sent to the provider.
    convert_to_llm: Callable[[list[AgentMessage]], list[Message]] = Field(exclude=True)

    # Required in practice: stream one assistant response. Providers implement
    # this; the loop only knows the signature.
    stream_fn: Callable[["ChatRequest", AbortSignal | None], Any] = Field(exclude=True)

    # Host context shaping, applied before every provider call.
    transform_context: Callable[[list[AgentMessage]], Awaitable[list[AgentMessage]] | list[AgentMessage]] | None = (
        Field(default=None, exclude=True)
    )

    # Steering (CONSUMING) interrupts tool batches; peek (non-consuming) is
    # polled between calls. Asides never interrupt.
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = Field(default=None, exclude=True)
    has_steering_messages: Callable[[], bool] | None = Field(default=None, exclude=True)
    get_aside_messages: Callable[[], Awaitable[list[Any]]] | None = Field(default=None, exclude=True)
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = Field(default=None, exclude=True)

    # Gates and hooks.
    before_model_call: Callable[[], Awaitable[bool] | bool] | None = Field(default=None, exclude=True)
    on_turn_end: Callable[[list[AgentMessage]], Awaitable[None] | None] | None = Field(default=None, exclude=True)
    on_before_yield: Callable[[], Awaitable[None] | None] | None = Field(default=None, exclude=True)

    # Fallback routing for unknown tool names (e.g. deferred MCP tools).
    resolve_fallback_tool: Callable[[str], AgentTool | None] | None = Field(default=None, exclude=True)

    interrupt_mode: Literal["immediate", "wait"] = "wait"
    # Epoch-ms deadline for the whole run, if any.
    deadline: float | None = None

    # Guardrails.
    max_paused_turn_continuations: int = 8


# ---------------------------------------------------------------------------
# Provider wire contract
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """A provider/model pair plus the knobs wire clients need."""

    provider: str  # registry id: openai, anthropic, kimi, xai, ollama, ...
    model_id: str
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    supports_tools: bool = True
    supports_images: bool = True
    supports_prompt_cache: bool = False
    base_url: str | None = None  # override for OpenAI-compatible endpoints
    temperature: float = 0.2
    top_p: float = 0.9
    reasoning: bool = False


class ChatRequest(BaseModel):
    """One provider call. System prompt is a LIST of blocks so providers can
    place cache breakpoints per block (stable instruction block first,
    volatile context last)."""

    model: ModelSpec
    system_blocks: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    tools: list[AgentTool] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] = "auto"


class StreamTextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    delta: str


class StreamToolCallDelta(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    id: str | None = None
    name: str | None = None
    argument_delta: str = ""


class StreamUsageEvent(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage


class StreamEndEvent(BaseModel):
    type: Literal["end"] = "end"
    stop_reason: str  # stop | length | toolUse | error | aborted
    usage: Usage | None = None
    provider_payload: dict[str, Any] | None = None
    error: str | None = None


StreamEvent = StreamTextDelta | StreamToolCallDelta | StreamUsageEvent | StreamEndEvent

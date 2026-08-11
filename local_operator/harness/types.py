"""Core harness types — the binding contract between all rewrite streams.

This module defines the ``packages/agent`` type surface. It deliberately knows
NOTHING about sessions, providers, persistence, skills, or UI. Every other
stream programs against these types:

- ``Message`` / ``CustomMessage`` — LLM-visible conversation entries.
- ``ToolCall`` / ``ToolResult`` / ``AgentTool`` — the tool protocol.
- ``AgentEvent`` — the ONLY boundary between engine and UI (TUI, print mode,
  server websockets all subscribe to these).
- ``LoopConfig`` — the callback bundle injected into the loop. The loop never
  imports session or provider code; everything is a callback here.
- ``ModelSpec`` / ``ChatRequest`` / ``StreamEvent`` — the provider wire
  contract implemented by ``local_operator.providers.clients``.

Design notes carried over from the reference engine:

- Messages carry an optional ``provider_payload`` for provider-native replay
  data (e.g. OpenAI Responses ids, Anthropic encrypted thinking). It rides
  through history untouched and is consumed only by wire clients.
- ``CustomMessage`` is the extension point for host-authored entries
  (compaction summaries, skill prompts, wake deliveries). It renders to an
  LLM-visible message via the session's ``convert_to_llm`` and NEVER goes to
  a provider raw.
- Aside commit/discard: the reference engine attaches symbols to message
  objects; here they
  are explicit optional callables excluded from serialization
  (``compare=False``), invoked by the loop when an aside message is actually
  injected (commit) or dropped as stale (discard).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Protocol,
    Sequence,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

# TypeVar comes from typing_extensions, NOT typing: the ``default=`` parameter
# below is PEP 696, which landed in typing only in 3.13, while this package
# supports 3.12. On 3.12 ``typing.TypeVar(default=...)`` raises TypeError at
# import time, which surfaces as an undiagnosable "preflight failed" on every
# command. typing_extensions backports it and is a declared dependency for
# exactly this reason (it also arrives with pydantic).
from typing_extensions import TypeVar

# One-way, in-package dependencies: ``approval`` (the gate's type plus the one
# arity resolver) and ``wake`` (pure schedule data plus a timer) each import
# nothing else from the harness, so naming their types here cannot cycle. They
# buy the approval-gate and wake-scheduler contracts on ``ToolContext`` real
# types instead of ``Any``.
from local_operator.harness.approval import ApprovalGate
from local_operator.harness.wake import WakeSchedule

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RenderedStreamError(Exception):
    """A stream failure whose ``str()`` is the whole story for the user.

    Raised by wire clients for provider responses (``HTTP 400: ...``) as
    opposed to defects. The loop catches both, but only the latter is worth a
    traceback: a handled provider answer that the UI already prints as one
    clean line does not also need forty lines of stack painted over the
    interface. Lives here rather than in ``providers`` because the harness must
    not import the provider layer — the dependency only runs the other way.
    """


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
    # Structured payload for renderers, logs and compaction pruning. Never
    # serialized to providers; always a JSON-ish mapping so consumers can
    # index it without probing the value's shape first.
    details: dict[str, Any] | None = None
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
    def user(text: str, images: "Sequence[ImageContent] | None" = None, **extra: Any) -> "Message":
        """A user turn, optionally carrying attachments.

        Images follow the text rather than leading it: the prompt says what to
        do with them, and a model reading the instruction first knows what it
        is looking for. Empty text still yields a text block, so a message that
        is nothing but a pasted screenshot keeps the shape every provider
        serializer expects.
        """
        content: list[Content] = [TextContent(text=text)]
        content.extend(images or ())
        return Message(role="user", content=content, **extra)

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
    on_commit: Callable[[], None] | None = Field(
        default=None, exclude=True, json_schema_extra={"compare": False}
    )
    on_discard: Callable[[], None] | None = Field(
        default=None, exclude=True, json_schema_extra={"compare": False}
    )


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

#: What evaluating an aside yields: a live message to inject, a stale-receipt
#: so the producer's ``on_discard`` still fires, or nothing at all.
AsideResult = AgentMessage | StaleAside | None
#: A queued aside is either a ready message or a thunk evaluated at the
#: injection boundary. The thunk form is the point: a payload that went stale
#: while the turn ran can withdraw itself instead of being injected.
Aside = AsideResult | Callable[[], AsideResult]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class AgentToolUpdate(BaseModel):
    """A partial result streamed from a running tool."""

    content: list[Content] = Field(default_factory=list)
    # Same contract as ``ToolResult.details``: a mapping or nothing.
    details: dict[str, Any] | None = None


# --- capability contracts carried on ToolContext ---------------------------
#
# These are Protocols rather than concrete classes for two reasons: the
# harness must not import the session/app layers that own the
# implementations, and hosts (and tests) legitimately supply their own
# stand-ins. They are ``runtime_checkable`` because pydantic validates an
# arbitrary-type field with ``isinstance``.


@runtime_checkable
class WakeSchedulerProtocol(Protocol):
    """The slice of a wake scheduler the ``wake`` tool drives.

    Implemented by :class:`local_operator.harness.wake.WakeScheduler`. The
    tool only ever reads the current list and writes a replacement, so the
    arming/timer half of the scheduler stays out of the contract.
    """

    @property
    def schedules(self) -> Sequence[WakeSchedule]: ...

    async def update(self, schedules: list[WakeSchedule]) -> None: ...


@runtime_checkable
class VariableStoreProtocol(Protocol):
    """The slice of a variable store the variables tools read.

    Implemented by :class:`local_operator.variables.VariableStore`. Listing
    yields names only — values are pulled one at a time — so the store's
    denylist stays the single gate on what the model can see.
    """

    def names(self) -> list[str]: ...

    def get(self, name: str) -> str | None: ...

    def read(self, name: str) -> str:
        """Resolve ``name`` or raise ``KeyError`` when unknown or denied."""
        ...


@runtime_checkable
class BrowserSurfaceProtocol(Protocol):
    """Mutable handle to the host browser surface a session has open.

    The browser tool records the handle here on ``open`` and every later
    action drives that surface instead of leaking a fresh one per call.
    """

    surface_id: str


class BrowserSurface:
    """The concrete :class:`BrowserSurfaceProtocol` a HOST owns.

    Deliberately host-owned rather than created by the tool on demand: the
    session rebuilds its :class:`ToolContext` at the start of EVERY turn, so a
    handle the tool stashed on the context survived only the turn that opened
    it. That broke the ordinary shape of browsing ("open X" then, next
    message, "click Y") and stranded a cmux tab per turn that nothing could
    close. Injected like ``wake_scheduler``, the surface outlives the context
    and session teardown can close it.

    Lives here beside the protocol, and holds no cmux knowledge, so a host can
    own one without importing the tool layer.
    """

    __slots__ = ("surface_id",)

    def __init__(self) -> None:
        self.surface_id = ""


@runtime_checkable
class JobManagerProtocol(Protocol):
    """The slice of ``harness.jobs.AsyncJobManager`` the tools drive.

    Declared HERE rather than importing the manager because ``harness.jobs``
    imports THIS module: annotating ``ToolContext.jobs`` with the concrete
    class would be an import cycle. Exposes exactly what ``wait``/``job``
    and cancellation need (get, list, cancel) and nothing more — spawning
    reaches the manager through session-installed launcher closures, never
    through this surface, so the tools cannot touch the manager's
    registration or delivery internals. Return types stay ``Any`` for the
    same edge reason: the ``AsyncJob`` row type cannot be named on this side
    of the cycle.
    """

    def get(self, job_id: str, *, owner_id: str | None = None) -> Any: ...

    def list(self, *, owner_id: str | None = None) -> list[Any]: ...

    async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool: ...


class ToolContext(BaseModel):
    """Minimal host-provided context handed to tool execution.

    Kept tiny on purpose (a 100-field monolithic session object is a symptom
    of a 9000-line session class); grow by demand. ``extra="allow"`` remains
    so a host can stash something bespoke, but every capability the built-in
    tools look for is DECLARED below — a tool must not have to probe for an
    undeclared attribute to find out what its host supports.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    cwd: str = "."
    session_id: str = ""
    agent_id: str = ""
    # Which BACKGROUND JOB this execution belongs to, so a host can scope an
    # approval decision to the work that provoked it instead of to every
    # request that follows. ``None`` is the foreground: the session's own turn.
    # Not telemetry and not an identity — ``session_id``/``agent_id`` already
    # say who is asking; this says on whose behalf, which is the part a host
    # cannot otherwise recover. Live failure it exists for: a subagent running
    # past the end of its parent's turn inherited that turn's approval state
    # and had its tools denied with no prompt shown to anyone.
    job_id: str | None = None
    has_ui: bool = False
    # Resolver hook for lazy internal URLs (``skill://`` and ``guide://``);
    # returns content or None when the URL is not handled. Installed by session.
    resolve_internal_url: Callable[[str], str | None] | None = None
    # Approval callback: returns True when the user approved. Tools with an
    # approval tier call this before mutating side effects. Two shapes are
    # accepted (see harness/approval.py); call it through ``ask_approval``
    # rather than directly, which is what picks the one this host wrote.
    request_approval: ApprovalGate | None = None
    # Session-named variables behind the list_variables / read_variable tools.
    # Values are never baked into the prompt; the model lists names and reads
    # single values on demand. ``None`` degrades those tools to the process
    # environment only.
    variables: VariableStoreProtocol | None = None
    # Wake scheduling. ``None`` means the host has no scheduler, and the wake
    # tool is then not advertised at all (createIf) rather than advertised and
    # always failing.
    wake_scheduler: WakeSchedulerProtocol | None = None
    # Durable todo lists keyed by session id. A host that attaches one gets
    # todo state it can persist alongside the transcript; otherwise the tool
    # falls back to a process-local table.
    todos: dict[str, list[dict[str, str]]] | None = None
    # Injected by the HOST (see BrowserSurface), not created by the tool: this
    # context is rebuilt every turn, so a tool-owned handle would not survive
    # to the next one. ``None`` degrades the browser tool to a single-call
    # surface the session can never close.
    browser: BrowserSurfaceProtocol | None = None
    # Launcher for one-shot child sessions run as background jobs (the
    # ``task`` tool). ``(label, prompt) -> job_id``, registering the run as
    # an AsyncJob on the host's job manager. The session installs a closure
    # over its own emit and job manager; ``None`` means the host has no
    # subagent engine and the task tool is then not advertised at all
    # (createIf) rather than advertised and always failing — the same
    # convention ``wake_scheduler`` uses.
    subagent_launcher: Callable[[str, str], str] | None = None
    # The session's background job manager. Declared as a Protocol because
    # the concrete class lives in ``harness.jobs``, which imports this
    # module (import cycle). The ``wait``/``job`` tools read it; ``None``
    # means no background work is tracked and both tools are then not
    # advertised at all (createIf) rather than advertised and always
    # failing.
    jobs: JobManagerProtocol | None = None
    # The parent↔child messaging surface behind the ``hub`` tool
    # (``harness.comms.SubagentComms``). Typed ``Any`` for the same import-
    # cycle reason ``jobs`` is a Protocol: ``harness.comms`` imports this
    # module. A CHILD carries its PARENT's instance — that is what lets
    # ``hub`` inside a subagent reach the agent that delegated to it, and
    # what ``is_child(job_id)`` uses to decide which shape of the tool to
    # advertise. ``None`` means no subagent engine, and the tool is then not
    # advertised at all (createIf).
    subagent_comms: Any | None = None


ToolExecuteFn = Callable[
    [
        str,
        dict[str, Any],
        "AbortSignal | None",
        Callable[[AgentToolUpdate], None] | None,
        ToolContext,
    ],
    Awaitable[ToolResult],
]


#: Renders the human sentence an approval prompt shows for one call. Takes the
#: call's parsed arguments and the session's working directory, and returns
#: ``"<verb>: <target>"`` — the shape the read-tier tools already use —
#: optionally led by one of two hazard markers:
#:
#: * ``[outside workspace]``: the target resolved, and it is not under the root.
#: * ``[unresolvable]``: the target could not be characterised at all, so nothing
#:   can be said about where it is. A describer may also return this in front of
#:   a sentence that is NOT ``<verb>: <target>`` — ``[unresolvable] unparsed url:
#:   <raw>`` — precisely because no verb-and-target pair could be determined.
#:
#: Both markers escalate identically; they differ only in the words the renderer
#: spells out, because a target visibly inside the workspace described as being
#: outside it teaches the reader to distrust the clause that matters. The cwd is
#: a parameter and not read
#: from the process because a session can be rooted anywhere (the server and the
#: scheduler both pass one), and "outside the workspace" is measured against the
#: session's root or it means nothing.
#:
#: This exists because the fallback the loop can build unaided
#: (``name({...json...})``) is the wrong string to put in front of a human who is
#: deciding whether to authorise something: the decision-relevant argument is
#: buried between quoting and irrelevant fields, and no amount of clever
#: truncation in the UI can recover which end of a JSON blob matters. Only the
#: tool knows which of its arguments IS the decision.
ApprovalDescribeFn = Callable[[dict[str, Any], str], str]


class AgentTool(BaseModel):
    """A tool the model can call.

    ``parameters`` is a JSON Schema object (pydantic model's
    ``model_json_schema()`` output). ``concurrency`` controls batch
    scheduling: ``"shared"`` tools run in parallel, ``"exclusive"`` alone.
    ``interruptible`` tools may be aborted mid-run to deliver steering.

    ``describe_approval`` is what the approval prompt says. Every write/exec
    tier tool should set it; without one the loop falls back to a JSON dump,
    which is legible to a reviewer of logs and not to a user answering a
    question under time pressure.
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
    describe_approval: ApprovalDescribeFn | None = Field(default=None, exclude=True)


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


#: The discriminator literal of a concrete event, defaulting to plain ``str``
#: so an unparameterized ``AgentEvent`` still means "any event". Declaring it
#: covariant is what lets a handler take ``AgentEvent`` and receive any
#: concrete subclass while each subclass keeps its exact ``Literal`` type —
#: the discriminator is written once at construction and never reassigned.
EventTypeT = TypeVar("EventTypeT", bound=str, default=str, covariant=True)


class AgentEvent(BaseModel, Generic[EventTypeT]):
    """Base event. ``type`` discriminates; UIs match exhaustively."""

    model_config = ConfigDict(extra="allow")

    type: EventTypeT


class AgentStartEvent(AgentEvent[Literal["agent_start"]]):
    type: Literal["agent_start"] = "agent_start"
    # Per-session monotonic turn counter; lets UIs drop a superseded
    # agent_end that arrives after the next agent_start.
    generation: int = 0


class AgentEndEvent(AgentEvent[Literal["agent_end"]]):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)
    aborted: bool = False
    error: str | None = None
    generation: int = 0


class TurnStartEvent(AgentEvent[Literal["turn_start"]]):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(AgentEvent[Literal["turn_end"]]):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)


class MessageStartEvent(AgentEvent[Literal["message_start"]]):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(AgentEvent[Literal["message_update"]]):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    delta: str = ""  # incremental text for this update (UIs should append, not re-read)


class MessageEndEvent(AgentEvent[Literal["message_end"]]):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolCallComposeEvent(AgentEvent[Literal["tool_call_compose"]]):
    """The model is STILL WRITING a tool call; nothing has run yet.

    Emitted while the arguments stream in, because for a large one they stream
    for a long time: asking for a file of any size means tens of kilobytes of
    `content` arriving token by token, and until the last one lands there is no
    call to execute, no `tool_execution_start`, and — before this event — nothing
    on screen. The user watches a spinner for minutes and reasonably concludes
    the agent has hung. It has not; it is dictating.

    ``argument_bytes`` is the running size of the arguments seen so far, which is
    the only honest progress signal available: the model never says how much is
    left. ``tool_call_id`` is the provider's id when it has arrived and an
    index-derived placeholder before that, so a UI can correlate this with the
    ``tool_execution_start`` that eventually follows.

    ``intent`` is the model's own ``i`` narration, scraped out of the partial
    JSON as soon as its string closes and ``None`` until then — never a
    half-word, because a label growing character by character on a repainting
    row reads worse than no label. This is the highest-value place the field
    appears: ``argument_bytes`` says the agent is alive, and only the intent
    says what for, across the longest silence of the turn.
    """

    type: Literal["tool_call_compose"] = "tool_call_compose"
    tool_call_id: str
    tool_name: str
    argument_bytes: int = 0
    intent: str | None = None


class ToolExecutionStartEvent(AgentEvent[Literal["tool_execution_start"]]):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    intent: str | None = None


class ToolExecutionUpdateEvent(AgentEvent[Literal["tool_execution_update"]]):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    partial_result: AgentToolUpdate


class ToolExecutionEndEvent(AgentEvent[Literal["tool_execution_end"]]):
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


class NoticeEvent(AgentEvent[Literal["notice"]]):
    type: Literal["notice"] = "notice"
    text: str
    kind: Literal["info", "warning", "error"] = "info"


class SubagentStartEvent(AgentEvent[Literal["subagent_start"]]):
    """A child session was registered as a background job.

    Subagent events are NOT the parent loop's own boundary events: they relay
    one CHILD session's lifecycle onto the parent's stream so a front end can
    render the child's progress. ``job_id`` is the AsyncJob id every event of
    this subagent carries, which is what lets a UI group them.
    """

    type: Literal["subagent_start"] = "subagent_start"
    job_id: str
    label: str
    agent_id: str | None = None


class SubagentProgressEvent(AgentEvent[Literal["subagent_progress"]]):
    """A throttled relay of a child session's activity.

    The relay emits one of these on tool starts/ends and message ends —
    NEVER on every stream delta — because a child streaming a file token by
    token would otherwise flood the parent stream with per-delta events.
    ``progress`` is a short human-readable description of the step.
    """

    type: Literal["subagent_progress"] = "subagent_progress"
    job_id: str
    label: str
    progress: str


class SubagentEndEvent(AgentEvent[Literal["subagent_end"]]):
    """A child session settled. The front end renders completion from THIS
    event; the runner deliberately adds no NoticeEvent or transcript write
    for it, so there is exactly one delivery path."""

    type: Literal["subagent_end"] = "subagent_end"
    job_id: str
    label: str
    status: str  # completed | failed | cancelled
    result_text: str | None = None
    error_text: str | None = None


class CompactionStartEvent(AgentEvent[Literal["compaction_start"]]):
    """A compaction pass began. ``reason`` is ``context-window`` for the
    automatic trigger, ``manual`` when the user asked for it."""

    type: Literal["compaction_start"] = "compaction_start"
    reason: str


class CompactionEndEvent(AgentEvent[Literal["compaction_end"]]):
    """A compaction pass settled.

    ``tokens_before``/``tokens_after`` size the LLM-visible history either side
    of the pass, by the same local ruler, so a host can report what the pass
    ACHIEVED — compaction is slow and its effect is invisible in the
    transcript, so "context compacted" alone asks the user to take it on faith.
    Both are zero when the pass failed, and ``strategy`` is the concrete
    mechanism that ran (``snapcompact`` or ``context-full``).
    """

    type: Literal["compaction_end"] = "compaction_end"
    reason: str
    success: bool
    strategy: str = ""
    tokens_before: int = 0
    tokens_after: int = 0


class RetryStartEvent(AgentEvent[Literal["retry_start"]]):
    type: Literal["retry_start"] = "retry_start"
    attempt: int
    error: str
    fallback_model: str | None = None


class RetryEndEvent(AgentEvent[Literal["retry_end"]]):
    type: Literal["retry_end"] = "retry_end"
    success: bool


EventHandler = Callable[[AgentEvent], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# Loop configuration — the host extension surface
# ---------------------------------------------------------------------------


class LoopConfig(BaseModel):
    """Everything the loop needs from its host, injected as callbacks.

    The ``AgentLoopConfig`` configuration record. Only ``convert_to_llm``
    and a model streamer are required; everything else has a neutral default.
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
    transform_context: (
        Callable[[list[AgentMessage]], Awaitable[list[AgentMessage]] | list[AgentMessage]] | None
    ) = Field(default=None, exclude=True)

    # Steering (CONSUMING) interrupts tool batches; peek (non-consuming) is
    # polled between calls. Asides never interrupt.
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = Field(
        default=None, exclude=True
    )
    has_steering_messages: Callable[[], bool] | None = Field(default=None, exclude=True)
    get_aside_messages: Callable[[], Awaitable[list[Aside]]] | None = Field(
        default=None, exclude=True
    )
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = Field(
        default=None, exclude=True
    )

    # Gates and hooks.
    before_model_call: Callable[[], Awaitable[bool] | bool] | None = Field(
        default=None, exclude=True
    )
    on_turn_end: Callable[[list[AgentMessage]], Awaitable[None] | None] | None = Field(
        default=None, exclude=True
    )
    on_before_yield: Callable[[], Awaitable[None] | None] | None = Field(default=None, exclude=True)

    # Fallback routing for unknown tool names (e.g. deferred MCP tools).
    resolve_fallback_tool: Callable[[str], AgentTool | None] | None = Field(
        default=None, exclude=True
    )

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
    # Whether the model accepts ``temperature``/``top_p`` at all. Some families
    # (Anthropic's Claude 5 generation, OpenAI's reasoning models) reject the
    # parameters outright with HTTP 400, so the defaults above are unsendable
    # and the wire clients must omit the keys rather than send a value.
    # Derived once in ``build_model_spec`` so the wire clients stay free of
    # model-name knowledge; see the note there for why it keys on the model
    # rather than the provider.
    supports_sampling_params: bool = True
    # The reasoning-effort ladder this model accepts, ASCENDING, and the level
    # it is currently set to. Same division of labour as the flag above: derived
    # once in ``build_model_spec`` from ``model.effort`` so no wire client and no
    # widget has to recognise a model name, and the KEY IS OMITTED rather than
    # sent with a null when the model exposes no knob — a provider that rejects
    # the key rejects it just as hard with an empty value.
    #
    # Two fields rather than one because the pair answers two different
    # questions asked by different callers: the wire clients need "what do I
    # send", the status band and ``/effort`` need "what else could this be set
    # to". An empty ``reasoning_efforts`` is the non-reasoning model, and it is
    # what makes ``/effort`` able to say so instead of accepting a level the
    # request would silently drop.
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_effort: str | None = None
    # The model's HUMAN name as metadata resolution found it — "Claude Opus 5"
    # for ``anthropic/claude-opus-5``, "MoonshotAI: Kimi K2" for an OpenRouter
    # id no registry row covers. Carried on the spec rather than looked up by
    # the reader because ``build_model_spec`` already holds the resolved
    # ``ModelInfo`` and then threw the name away, which left the status band
    # with nothing to print but the selector and left an aggregator model — the
    # case with no curated name at all — permanently unnamable without a disk
    # read inside a repaint.
    #
    # A RAW name, not a display decision: ``model/naming.py`` owns whether it is
    # safe to show and how it narrows. Empty means resolution had none, which is
    # different from "no name exists" and is why the readers fall back rather
    # than treat this as authoritative.
    display_name: str = ""


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
    #: This call's output has NOT been shown to anyone yet, so a failed attempt
    #: may be discarded and retried whole.
    #:
    #: Transport policy rather than wire content (no ``_build_body`` reads it):
    #: it lives here because the loop's ``stream_fn`` signature is
    #: ``(request, signal)`` in every host and fake, and the one fact the
    #: failover driver is missing is a property of the CALL.
    #:
    #: ``False`` for a turn and for an aside: their deltas reach the transcript
    #: as they arrive, and a retry would re-render text the user already read.
    #: ``True`` for the one-shot errands that collect the whole stream before
    #: returning a string — the compaction summary and auto-naming — where a
    #: stalled read (``_guarded_chunks`` gives up after 180s of silence) used to
    #: be a permanent failure because the driver had already forwarded events it
    #: could not take back. A failed compaction is not cosmetic: the context it
    #: was meant to shrink keeps growing.
    replayable: bool = False


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

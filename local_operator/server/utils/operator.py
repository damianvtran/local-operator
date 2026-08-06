"""Session facade for the Local Operator API.

Why this module exists
----------------------
The HTTP surface (44 endpoints) and the shipped Electron UI are frozen: paths,
verbs, envelopes, status codes and websocket message shapes must not move. The
engine underneath them was replaced wholesale — the classify/plan/act triple
round-trip ``Operator`` + ``LocalCodeExecutor`` pair is gone, and the rewritten
harness exposes a very different contract (``Session.prompt`` driving an
``AgentEvent`` stream, JSONL transcripts, native tool calling).

This module is the adapter between the two. It keeps the legacy call shape the
routes and job processors were written against::

    operator = create_operator(...)          # sync, no I/O
    await operator.handle_user_input(prompt) # -> (ResponseJsonSchema|None, str)
    operator.executor.agent_state.conversation
    operator.executor.model_configuration

and implements it over ``session_factory.create_session``. Three deliberate
design points:

1. **Lazy session construction.** ``create_operator`` must stay synchronous
   (every call site invokes it without ``await``), but ``create_session`` is
   async and does real I/O (skill index, MCP discovery, auth store). The
   session is therefore built on the first ``handle_user_input`` call and
   disposed before it returns, so nothing leaks across requests.
2. **``agent_state`` is materialized from the turn's events, never faked.**
   The engine's durable memory is the JSONL transcript; the server's
   ``/v1/agents/{id}/conversation`` endpoints read ``AgentState`` from the
   agent registry. After a turn, the messages carried by the terminal
   ``agent_end`` event are projected into ``ConversationRecord`` /
   ``CodeExecutionResult`` and appended to the in-memory state (and persisted
   to the registry when ``persist_conversation`` is set), which is exactly
   what the routes serialize into their response envelopes.
3. **Events are translated, not re-invented.** ``AgentEventBridge`` maps the
   engine's event stream onto the ``CodeExecutionResult`` payloads the
   websocket manager already broadcasts, so the UI sees byte-identical frames.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, List, Literal, Optional, Protocol, Sequence

from local_operator.agents import AgentData, AgentRegistry
from local_operator.bootstrap import initialize_operator, resolve_model_configuration
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ChatRequest,
    CustomMessage,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    NoticeEvent,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from local_operator.model.configure import ModelConfiguration
from local_operator.types import (
    ActionType,
    AgentState,
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    ExecutionType,
    OperatorType,
    ProcessResponseStatus,
    ResponseJsonSchema,
)

logger = logging.getLogger("local_operator.server.utils")


class StatusQueue(Protocol):
    """The producer half of the multiprocessing status queue.

    The job processors create a ``multiprocessing.Queue`` in the parent
    process and hand it down to the child; on this side only ``put`` is ever
    called. Declaring just that keeps the annotation honest without dragging
    in the unparameterizable ``multiprocessing.Queue`` alias, and lets tests
    substitute a list-backed stub.
    """

    def put(self, obj: object, /) -> None: ...


class ExecutorInitError(Exception):
    """Raised when conversation history is initialized twice.

    Preserved from the deleted ``local_operator.executor`` because the chat
    routes branch on a second ``initialize_conversation_history`` call.
    """

    def __init__(self, message: str = "Failed to initialize executor") -> None:
        self.message = message
        super().__init__(self.message)


class ModelResponse:
    """Minimal stand-in for the langchain ``BaseMessage`` the legacy
    ``invoke_model`` returned. Only ``.content`` was ever read."""

    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content


#: Harness message roles that map onto a legacy conversation role. Tool
#: messages and ``CustomMessage`` plumbing entries have no legacy equivalent
#: and are dropped from the conversation projection rather than invented.
_ROLE_MAP: dict[str, ConversationRole] = {
    "user": ConversationRole.USER,
    "assistant": ConversationRole.ASSISTANT,
}

#: The same mapping in the other direction, for seeding the engine from the
#: legacy conversation projection. Spelled out rather than reading
#: ``ConversationRole.value`` so the harness ``Message.role`` literal stays
#: checkable.
_WIRE_ROLE_MAP: dict[ConversationRole, Literal["user", "assistant"]] = {
    ConversationRole.USER: "user",
    ConversationRole.ASSISTANT: "assistant",
}


def _project_conversation_to_messages(records: Sequence[ConversationRecord]) -> list[Message]:
    """Legacy ``ConversationRecord`` list → harness ``Message`` list.

    Only user/assistant text records project; system-prompt records, empty
    content and non-text roles are dropped (the engine's system blocks carry
    the instructions, and inventing tool history would break pairing).
    """
    out: list[Message] = []
    for record in records:
        wire_role = _WIRE_ROLE_MAP.get(record.role)
        content = record.content or ""
        if wire_role is None:
            continue
        if not content.strip() or record.is_system_prompt:
            continue
        out.append(Message(role=wire_role, content=[TextContent(text=content)]))
    return out


def _message_text(message: Message | CustomMessage) -> str:
    """Text of a harness ``Message``; empty for ``CustomMessage`` plumbing
    entries, which carry no text blocks at all."""
    return message.text if isinstance(message, Message) else ""


def _now() -> datetime:
    return datetime.now(timezone.utc)


#: Per-string ceiling for values forwarded on the event stream. A tool result
#: can be megabytes (a build log, a large file read); pushing that through a
#: multiprocessing queue and then to every listener would stall the pump and
#: blow out browser memory for content the transcript already holds. The
#: authoritative copy is reachable over REST and, for oversized tool output,
#: through the spill store - so the stream carries a readable prefix and says
#: so, rather than either truncating silently or shipping everything.
STREAM_VALUE_LIMIT = 16 * 1024

#: Marker appended to a clipped value. Distinct from the tool-output truncation
#: marker so a consumer can tell "clipped for streaming" from "clipped by the
#: tool", which have different recovery paths.
STREAM_TRUNCATION_MARKER = "\n\n[... truncated for streaming; fetch the record for full content]"


def _cap_stream_payload(payload: dict[str, object], _depth: int = 0) -> None:
    """Clip oversized strings in a serialised event, in place.

    Recursion is depth-limited because event payloads nest (an event holds a
    message which holds content blocks which hold tool results); a cycle is
    impossible in JSON-dumped data, but a pathological depth would still be
    a needless stack risk.
    """
    if _depth > 6:
        return
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > STREAM_VALUE_LIMIT:
            payload[key] = value[:STREAM_VALUE_LIMIT] + STREAM_TRUNCATION_MARKER
        elif isinstance(value, dict):
            _cap_stream_payload(value, _depth + 1)  # type: ignore[arg-type]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _cap_stream_payload(item, _depth + 1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Event translation
# ---------------------------------------------------------------------------


class AgentEventBridge:
    """Translate the engine's ``AgentEvent`` stream into the websocket frames
    the UI expects.

    The UI consumes ``CodeExecutionResult`` dumps (see
    ``WebSocketManager.broadcast_update``) pushed through the multiprocessing
    ``status_queue`` as ``("message_update", <message id>, result)`` and
    ``("execution_update", <job id>, result)``. Both tuple shapes and the
    payload model are unchanged from the legacy executor; only their producer
    moved.

    Streaming semantics preserved from the legacy operator:

    - one long-lived record per assistant message, ``is_streamable=True``,
      re-broadcast on every delta with the accumulated text in ``message``;
    - ``status`` stays ``in_progress`` until the message ends, then flips to
      ``success`` with ``is_complete=True``;
    - tool executions get their own record keyed by tool-call id, mirroring
      the legacy ACTION records (code in, stdout/stderr out).
    """

    def __init__(self, status_queue: StatusQueue | None = None, job_id: str | None = None) -> None:
        self._status_queue = status_queue
        self._job_id = job_id
        self._streams: dict[str, CodeExecutionResult] = {}
        self._tools: dict[str, CodeExecutionResult] = {}
        # Creation-ordered record list: execution_records used to concatenate
        # the two dicts (all assistant records, then all tool records), so a
        # turn's persisted activity log no longer reflected what happened.
        self._ordered: list[CodeExecutionResult] = []
        #: Text of the last completed assistant message — the value the HTTP
        #: envelope reports as ``response``.
        self.final_response: str = ""

    # -- queue plumbing ----------------------------------------------------

    def _put(self, payload: tuple[str, str, object]) -> None:
        if self._status_queue is None:
            return
        try:
            self._status_queue.put(payload)
        except Exception:  # noqa: BLE001 — a dead queue must not kill the turn
            logger.warning("failed to publish status update", exc_info=True)

    def _broadcast(self, record: CodeExecutionResult) -> None:
        self._put(("message_update", record.id, record))

    def _execution(self, record: CodeExecutionResult) -> None:
        if self._job_id:
            self._put(("execution_update", self._job_id, record))

    def _raw(self, event: AgentEvent) -> None:
        """Forward the engine event itself, for the SSE transport only.

        The record frames above are lossy by design - they collapse a whole
        message into repeated cumulative snapshots and drop tool progress,
        turn boundaries and retries entirely, because that is all the legacy
        websocket contract could express. SSE consumers get the real taxonomy
        through this path, so no existing client changes and a new one need not
        reverse-engineer state from snapshots.

        Serialised to plain JSON-compatible data here rather than in the parent:
        the payload crosses a process boundary by pickle, and a dict of
        primitives cannot fail to reconstruct on the far side the way a model
        carrying engine types can.
        """
        if self._status_queue is None or not self._job_id:
            return
        try:
            payload = event.model_dump(mode="json")
        except Exception:  # noqa: BLE001 — a serialisation quirk must not kill the turn
            logger.warning("failed to serialise agent event for stream", exc_info=True)
            return
        _cap_stream_payload(payload)
        self._put(("agent_event", self._job_id, payload))

    # -- the handler -------------------------------------------------------

    def handle(self, event: AgentEvent) -> None:
        # Every event goes to the SSE path first, unconditionally. The record
        # translation below is a lossy projection for the legacy websocket
        # contract; forwarding before it means a new event type reaches SSE
        # consumers whether or not anyone taught the projection about it.
        self._raw(event)

        # isinstance rather than a ``.type`` string switch: each concrete
        # event declares its own fields, so narrowing here is what lets the
        # handlers below read them without probing.
        if isinstance(event, AgentStartEvent):
            self._on_agent_start()
        elif isinstance(event, (MessageStartEvent, MessageUpdateEvent, MessageEndEvent)):
            self._on_message(event)
        elif isinstance(event, ToolExecutionStartEvent):
            self._on_tool_start(event)
        elif isinstance(event, ToolExecutionEndEvent):
            self._on_tool_end(event)
        elif isinstance(event, NoticeEvent):
            self._on_notice(event)

    def _on_agent_start(self) -> None:
        # Legacy parity: the UI's "working" indicator is driven by an
        # in-progress ACTION execution record emitted before the first token.
        self._execution(
            CodeExecutionResult(
                message="Thinking about my next action",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                execution_type=ExecutionType.ACTION,
                timestamp=_now(),
            )
        )

    def _on_message(self, event: MessageStartEvent | MessageUpdateEvent | MessageEndEvent) -> None:
        message = event.message
        # ``CustomMessage`` entries are UI plumbing with no role and no text,
        # so they never become a streamed assistant record.
        if not isinstance(message, Message) or message.role != "assistant":
            return
        message_id = message.id or uuid.uuid4().hex
        record = self._streams.get(message_id)
        if record is None:
            record = CodeExecutionResult(
                id=message_id,
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                execution_type=ExecutionType.RESPONSE,
                is_streamable=True,
                timestamp=_now(),
            )
            self._streams[message_id] = record
            self._ordered.append(record)

        record.message = _message_text(message)
        if isinstance(event, MessageEndEvent):
            record.status = ProcessResponseStatus.SUCCESS
            record.is_complete = True
            if record.message:
                self.final_response = record.message
        self._broadcast(record)
        self._execution(record)

    def _on_tool_start(self, event: ToolExecutionStartEvent) -> None:
        call_id = event.tool_call_id or uuid.uuid4().hex
        tool_name = event.tool_name
        args = event.args or {}
        record = CodeExecutionResult(
            id=call_id,
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.IN_PROGRESS,
            execution_type=ExecutionType.ACTION,
            action=ActionType.CODE,
            message=event.intent or tool_name,
            code=_render_tool_args(tool_name, args),
            timestamp=_now(),
        )
        self._tools[call_id] = record
        self._ordered.append(record)
        self._broadcast(record)
        self._execution(record)

    def _on_tool_end(self, event: ToolExecutionEndEvent) -> None:
        record = self._tools.get(event.tool_call_id)
        if record is None:
            return
        text = event.result.text
        if event.is_error:
            record.stderr = text
            record.status = ProcessResponseStatus.ERROR
        else:
            record.stdout = text
            record.status = ProcessResponseStatus.SUCCESS
        record.is_complete = True
        self._broadcast(record)
        self._execution(record)

    def _on_notice(self, event: NoticeEvent) -> None:
        record = CodeExecutionResult(
            message=event.text,
            role=ConversationRole.SYSTEM,
            status=ProcessResponseStatus.SUCCESS,
            execution_type=ExecutionType.INFO,
            is_complete=True,
            timestamp=_now(),
        )
        self._broadcast(record)
        self._execution(record)

    # -- projection --------------------------------------------------------

    def execution_records(self) -> list[CodeExecutionResult]:
        """Every record produced by the turn, in creation order."""
        return list(self._ordered)


def _render_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    """A compact one-line rendering of a tool call for the UI's code slot."""
    if not args:
        return tool_name
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{tool_name}({rendered})"


# ---------------------------------------------------------------------------
# Executor facade
# ---------------------------------------------------------------------------


class ServerExecutor:
    """The ``operator.executor`` surface the routes program against.

    Holds the conversation projection plus a one-shot ``invoke_model`` used by
    the inline-edit and speech endpoints, both of which need a single
    completion rather than an agent turn.
    """

    def __init__(
        self,
        model_configuration: ModelConfiguration,
        credential_manager: Optional[CredentialManager] = None,
        config_manager: Optional[ConfigManager] = None,
        agent_registry: Optional[AgentRegistry] = None,
        agent: Optional[AgentData] = None,
        agent_state: Optional[AgentState] = None,
        persist_conversation: bool = False,
        job_id: Optional[str] = None,
        status_queue: StatusQueue | None = None,
    ) -> None:
        self.model_configuration = model_configuration
        self.credential_manager = credential_manager
        self.config_manager = config_manager
        self.agent_registry = agent_registry
        self.agent = agent
        self.persist_conversation = persist_conversation
        self.job_id = job_id
        self.status_queue = status_queue
        self.agent_state = agent_state or AgentState(
            version="",
            conversation=[],
            execution_history=[],
            learnings=[],
            schedules=[],
            current_plan=None,
            instruction_details=None,
            agent_system_prompt=None,
        )

    # -- conversation ------------------------------------------------------

    def initialize_conversation_history(
        self,
        new_conversation_history: Sequence[ConversationRecord] = (),
        overwrite: bool = False,
    ) -> None:
        """Seed the conversation with a system record plus caller-supplied
        history.

        Same contract as the legacy executor: calling it twice without
        ``overwrite`` raises. The system record is a marker only — the real
        system prompt is assembled per turn by the harness from
        ``prompts_md/system.md`` — but the UI's conversation envelope has
        always carried a leading system entry and tests assert its presence.
        """
        if overwrite:
            self.agent_state.conversation = []

        if len(self.agent_state.conversation) != 0:
            raise ExecutorInitError("Conversation history already initialized")

        history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=self.agent_state.agent_system_prompt or "System prompt",
                is_system_prompt=True,
            )
        ]
        history.extend(record for record in new_conversation_history if not record.is_system_prompt)
        self.agent_state.conversation = history

    def append_to_history(self, record: ConversationRecord) -> None:
        """Append one record to the conversation projection."""
        self.agent_state.conversation.append(record)

    def get_conversation_history(self) -> List[ConversationRecord]:
        return self.agent_state.conversation

    def add_to_code_history(
        self, code_execution_result: CodeExecutionResult
    ) -> CodeExecutionResult:
        self.agent_state.execution_history.append(code_execution_result)
        return code_execution_result

    # -- one-shot completion ----------------------------------------------

    async def invoke_model(self, messages: Iterable[ConversationRecord]) -> ModelResponse:
        """Run a single non-agentic completion and return its text.

        Used by the inline-edit endpoint and the speech gender classifier.
        Goes straight through the provider wire clients (``stream_fn``), so no
        tools, no transcript, and no session lifecycle are involved.
        """
        from local_operator.model.configure import create_stream_fn
        from local_operator.providers.auth_store import AuthStore

        system_blocks: list[str] = []
        wire_messages: list[Message] = []
        for record in messages:
            content = record.content or ""
            if record.role == ConversationRole.SYSTEM or record.is_system_prompt:
                if content:
                    system_blocks.append(content)
                continue
            if record.role in (ConversationRole.ASSISTANT, ConversationRole.AI):
                wire_messages.append(Message.assistant(content))
            else:
                wire_messages.append(Message.user(content))

        settings = (
            self.config_manager.get_config().values if self.config_manager is not None else None
        )
        auth_store = AuthStore(credential_manager=self.credential_manager)
        try:
            stream_fn = create_stream_fn(auth_store, settings=settings)
            request = ChatRequest(
                model=self.model_configuration.spec,
                system_blocks=system_blocks,
                messages=wire_messages,
                tools=[],
                temperature=self.model_configuration.temperature,
                max_tokens=self.model_configuration.max_tokens,
                tool_choice="none",
            )
            chunks: list[str] = []
            async for event in stream_fn(request, None):
                # The stream union's members are distinct models; narrow so
                # each branch reads only the fields it actually declares.
                if isinstance(event, StreamTextDelta):
                    chunks.append(event.delta)
                elif isinstance(event, StreamEndEvent) and event.error:
                    raise RuntimeError(event.error)
            return ModelResponse("".join(chunks))
        finally:
            try:
                auth_store.close()
            except Exception:  # noqa: BLE001 — teardown must not mask errors
                logger.debug("auth store close failed", exc_info=True)


# ---------------------------------------------------------------------------
# Operator facade
# ---------------------------------------------------------------------------


class ServerOperator:
    """The ``Operator`` surface the routes and job processors call.

    One instance serves one HTTP request (or one background job). The harness
    session is created on demand inside :meth:`handle_user_input` and disposed
    before it returns.
    """

    def __init__(
        self,
        executor: ServerExecutor,
        config_manager: ConfigManager,
        credential_manager: CredentialManager,
        agent_registry: AgentRegistry,
        env_config: EnvConfig,
        hosting: str,
        model: str,
        current_agent: Optional[AgentData] = None,
        persist_conversation: bool = False,
        job_id: Optional[str] = None,
        status_queue: StatusQueue | None = None,
    ) -> None:
        self.executor = executor
        self.config_manager = config_manager
        self.credential_manager = credential_manager
        self.agent_registry = agent_registry
        self.env_config = env_config
        self.hosting = hosting
        self.model = model
        self.current_agent = current_agent
        self.persist_agent_conversation = persist_conversation
        self.job_id = job_id
        self.status_queue = status_queue

    # -- the one entry point ----------------------------------------------

    async def handle_user_input(
        self,
        user_input: str,
        user_message_id: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        additional_instructions: Optional[str] = None,
    ) -> tuple[ResponseJsonSchema | None, str]:
        """Run one agent turn and project it back onto the legacy shapes.

        Returns ``(response_json, final_response)`` exactly like the legacy
        operator: the routes read ``response_json.response`` for token
        accounting and ``final_response`` for the envelope's ``response``
        field.
        """
        attachments = list(attachments or [])
        prompt = user_input
        if additional_instructions:
            prompt = f"{prompt}\n\n## Additional Instructions\n\n{additional_instructions}"
        if attachments:
            listing = "\n".join(f"- {path}" for path in attachments)
            prompt = f"{prompt}\n\n## Attachments\n\n{listing}"

        # The user's turn is part of the conversation projection regardless of
        # how the engine turn resolves, so record it up front (legacy parity:
        # the UI shows the user bubble before the first token arrives).
        self.executor.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=user_input,
                files=attachments or None,
                timestamp=_now(),
            )
        )
        self.executor.add_to_code_history(
            CodeExecutionResult(
                id=user_message_id or str(uuid.uuid4()),
                message=user_input,
                files=attachments,
                role=ConversationRole.USER,
                status=ProcessResponseStatus.SUCCESS,
                execution_type=ExecutionType.USER_INPUT,
                is_complete=True,
                timestamp=_now(),
            )
        )

        # ``job_processor_queue`` assigns the queue onto the executor AFTER
        # construction (legacy call shape), so the executor's value wins.
        status_queue = self.executor.status_queue or self.status_queue
        bridge = AgentEventBridge(status_queue=status_queue, job_id=self.job_id)
        session = await initialize_operator(
            operator_type=OperatorType.SERVER,
            config_manager=self.config_manager,
            credential_manager=self.credential_manager,
            agent_registry=self.agent_registry,
            env_config=self.env_config,
            # The routes mutate ``executor.model_configuration`` to apply the
            # request's ``options``; that object is the server's source of
            # truth for sampling, so hand its values to the session.
            sampling_overrides={
                "temperature": self.executor.model_configuration.temperature,
                "top_p": self.executor.model_configuration.top_p,
            },
            request_hosting=self.hosting,
            request_model=self.model,
            current_agent=self.current_agent,
            persist_conversation=self.persist_agent_conversation,
            job_id=self.job_id,
            status_queue=status_queue,
            verbosity_level=VerbosityLevel.QUIET,
        )
        end_events: list[AgentEndEvent] = []

        def _capture(event: AgentEvent) -> None:
            if isinstance(event, AgentEndEvent):
                end_events.append(event)
            bridge.handle(event)

        # The frozen endpoints hand the engine their history through the
        # legacy projection: stateless /v1/chat loads the caller's context
        # array into agent_state.conversation, and non-persist agent chat
        # loads the registry's stored conversation. The transcript is NOT the
        # history source on either path, so without seeding the provider sees
        # the bare prompt while the envelope echoes history it never read.
        # The persist path replays its transcript and seed_history no-ops.
        # The current turn's user record was appended to the projection up
        # front; it is excluded so the prompt is not delivered twice.
        records = list(self.executor.agent_state.conversation)
        if (
            records
            and records[-1].role == ConversationRole.USER
            and records[-1].content == user_input
        ):
            records = records[:-1]
        seeded = _project_conversation_to_messages(records)

        unsubscribe = session.subscribe(_capture)
        try:
            if seeded:
                await session.seed_history(seeded)
            await session.prompt(prompt)
        finally:
            try:
                unsubscribe()
            except Exception:  # noqa: BLE001
                pass
            try:
                await session.dispose()
            except Exception:  # noqa: BLE001 — disposal must not mask the turn
                logger.exception("failed to dispose server session")

        end_event = end_events[-1] if end_events else None
        if end_event is not None and end_event.error:
            raise RuntimeError(f"Agent turn failed: {end_event.error}")

        messages: list[Message | CustomMessage] = list(end_event.messages) if end_event else []
        final_response = bridge.final_response or _last_assistant_text(messages)

        self._project_turn(messages, bridge)

        response_json = ResponseJsonSchema(
            response=final_response,
            action=ActionType.DONE,
        )
        return response_json, final_response

    # -- projection --------------------------------------------------------

    def _project_turn(
        self, messages: Sequence[Message | CustomMessage], bridge: AgentEventBridge
    ) -> None:
        """Fold the turn's engine messages into the legacy ``AgentState`` and
        persist it when the request asked for conversation persistence.

        Only user/assistant messages carry into ``conversation``: tool
        messages have no ``ConversationRecord`` equivalent and the UI renders
        tool activity from ``execution_history`` instead, which
        ``AgentEventBridge`` already populated with one record per tool call.
        """
        state = self.executor.agent_state
        for message in messages:
            # ``CustomMessage`` plumbing entries have no role and no legacy
            # equivalent, so they never reach the conversation projection.
            role = _ROLE_MAP.get(message.role) if isinstance(message, Message) else None
            if role is None:
                continue
            text = _message_text(message)
            if not text:
                continue
            if role is ConversationRole.USER:
                # The prompt was already recorded before the turn ran.
                continue
            state.conversation.append(ConversationRecord(role=role, content=text, timestamp=_now()))

        state.execution_history.extend(bridge.execution_records())

        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            try:
                self.agent_registry.update_agent_state(
                    agent_id=self.current_agent.id,
                    agent_state=state,
                )
            except Exception:  # noqa: BLE001 — persistence is best-effort
                logger.exception("failed to persist agent state after turn")


def _last_assistant_text(messages: Sequence[Message | CustomMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, Message) and message.role == "assistant":
            text = _message_text(message)
            if text:
                return text
    return ""


# ---------------------------------------------------------------------------
# Factory — signature frozen for the routes and job processors
# ---------------------------------------------------------------------------


def create_operator(
    request_hosting: str,
    request_model: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
    current_agent: Optional[AgentData] = None,
    persist_conversation: bool = False,
    job_id: Optional[str] = None,
    status_queue: StatusQueue | None = None,
) -> ServerOperator:
    """Build the per-request session facade.

    Cheap and synchronous by design: it resolves hosting/model precedence and
    the agent's stored state, but does not touch the network, the skill index,
    or MCP. The harness session itself is constructed on the first
    ``handle_user_input`` call.

    Raises:
        ValueError: when hosting/model configuration is missing or invalid.
    """
    logger.info(
        f"Creating server operator for Hosting: {request_hosting}, Model: {request_model}, "
        f"Agent: {current_agent.name if current_agent else 'None'}, Job ID: {job_id}"
    )

    agent_state: Optional[AgentState] = None
    if current_agent is not None:
        agent_state = agent_registry.load_agent_state(current_agent.id)

    model_configuration, hosting, model_name = resolve_model_configuration(
        config_manager,
        credential_manager,
        env_config,
        request_hosting=request_hosting,
        request_model=request_model,
        current_agent=current_agent,
    )

    executor = ServerExecutor(
        model_configuration=model_configuration,
        credential_manager=credential_manager,
        config_manager=config_manager,
        agent_registry=agent_registry,
        agent=current_agent,
        agent_state=agent_state,
        persist_conversation=persist_conversation,
        job_id=job_id,
        status_queue=status_queue,
    )
    operator = ServerOperator(
        executor=executor,
        config_manager=config_manager,
        credential_manager=credential_manager,
        agent_registry=agent_registry,
        env_config=env_config,
        hosting=hosting,
        model=model_name,
        current_agent=current_agent,
        persist_conversation=persist_conversation,
        job_id=job_id,
        status_queue=status_queue,
    )
    logger.info("Server operator created successfully.")
    return operator

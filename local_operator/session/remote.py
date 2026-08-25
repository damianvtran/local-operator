"""Full-fidelity remote session facade for a follower TUI (protocol v4).

``RemoteSession`` implements the same :class:`SessionProtocol` the standard
``OperatorApp`` already consumes. Durable history comes from the transcript;
live rendering comes from the owner's raw ``AgentEvent`` relay; every mutation
goes back over the authenticated loopback control socket. The app therefore
hosts its normal transcript, tool cards, composer, slash registry and gate
widgets. There is no attach-specific UI and no inverse-folding of the phone
projection.

Connection loss is plumbing, not a user decision. The facade silently
re-discovers a replacement owner or attempts the normal resume factory. The
existing sole-writer lease arbitrates simultaneous followers: one becomes the
owner, losers observe ``SessionLeaseHeldError`` and redial the winner. The app
installs a takeover callback at adoption so the winning real Session replaces
this facade without clearing the painted transcript.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from local_operator.harness.approval import ApprovalGate
from local_operator.harness.approval import ask_approval as call_approval_gate
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AskOption,
    AskQuestion,
    AskUserFn,
    CompactionEndEvent,
    CompactionStartEvent,
    EventHandler,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    ModelSpec,
    NoticeEvent,
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
from local_operator.mobile.attach_client import AttachClient, find_owner_record
from local_operator.mobile.types import (
    ContinuationCommand,
    LiveTurnSeed,
    PendingRequest,
    SessionProjection,
    SessionRecord,
)
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome
from local_operator.session.transcript import Transcript
from local_operator.session_lease import SessionLeaseHeldError

logger = logging.getLogger(__name__)

_EVENT_TYPES: dict[str, type[AgentEvent[Any]]] = {
    cls.model_fields["type"].default: cls
    for cls in (
        AgentStartEvent,
        AgentEndEvent,
        TurnStartEvent,
        TurnEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        MessageEndEvent,
        ToolCallComposeEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        ToolExecutionEndEvent,
        NoticeEvent,
        WakeDeliveredEvent,
        SteeringDeliveredEvent,
        SubagentStartEvent,
        SubagentProgressEvent,
        SubagentEndEvent,
        CompactionStartEvent,
        CompactionEndEvent,
        RetryStartEvent,
        ModelChangeEvent,
        RetryEndEvent,
    )
}


def deserialize_event(data: dict[str, Any]) -> AgentEvent[Any]:
    """Rehydrate one relayed event into its concrete pydantic subclass.

    Unknown future event types remain base ``AgentEvent`` instances. The base
    allows extra fields and EventController ignores unknown types, so a newer
    owner can relay through an older follower without killing its stream.
    """
    cls = _EVENT_TYPES.get(str(data.get("type", "")), AgentEvent)
    return cls.model_validate(data)


class RemoteSession:
    """A SessionProtocol facade backed by one owner's v4 attach socket."""

    is_remote = True

    def __init__(
        self,
        *,
        config_dir: Path,
        session_id: str,
        takeover_factory: Callable[[], Any],
    ) -> None:
        self._config_dir = config_dir
        self._session_id = session_id
        self._takeover_factory = takeover_factory
        self._client: AttachClient | None = None
        self._projection: SessionProjection | None = None
        self._sync_future: asyncio.Future[LiveTurnSeed] | None = None
        self._history: list[Any] = []
        self._history_ids: set[str] = set()
        self._handlers: list[EventHandler] = []
        # Events arriving before OperatorApp adopts/subscribes are retained;
        # otherwise a fast owner can stream between factory return and app
        # adoption and the first visible delta vanishes.
        self._buffered_events: list[AgentEvent[Any]] = []
        self._ready_for_events = False
        self._approval_handler: ApprovalGate | None = None
        self._ask_handler: AskUserFn | None = None
        self._gate_task: asyncio.Task[None] | None = None
        # One ask card keeps its request id while advancing through questions.
        # The question index is therefore part of the gate identity: request id
        # alone made Q2 look like a duplicate of Q1 and stranded the owner gate.
        self._gate_key: tuple[str, str, int] | None = None
        self._disposed = False
        self._recovering = False
        self._recovery_task: asyncio.Task[None] | None = None
        self._takeover_callback: Callable[[Any], Any] | None = None
        # Input submitted while the owner rotates waits here instead of failing
        # out of the composer's turn worker. On reattach it goes over the fresh
        # socket; on takeover it goes straight to the real Session after the
        # preserving adoption callback completes. Keystrokes remain editable in
        # the standard composer throughout — no attach/recovery UI state.
        self._owner_ready = asyncio.Event()
        self._takeover_target: Any | None = None
        self._streaming = False
        self._generation = 0
        self._queued_steers: list[Message] = []
        self._name_state = ConversationName()
        self._model = ModelSpec(provider="unknown", model_id="unknown")

    @classmethod
    async def connect(
        cls,
        record: SessionRecord,
        session_id: str,
        *,
        config_dir: Path,
        takeover_factory: Callable[[], Any],
    ) -> "RemoteSession":
        if record.protocol < 4:
            raise ConnectionError(
                f"owner runs protocol v{record.protocol}; full-TUI attach needs >= 4"
            )
        self = cls(
            config_dir=config_dir,
            session_id=session_id,
            takeover_factory=takeover_factory,
        )
        await self._dial(record)
        await self._load_history()
        seed = await self._await_seed()
        self._finish_sync(seed)
        return self

    async def _dial(self, record: SessionRecord) -> None:
        # Freeze relay delivery until attach_sync is installed ahead of any
        # event frames that follow it on the socket. The pump may process both
        # before the coroutine awaiting the future resumes.
        self._ready_for_events = False
        loop = asyncio.get_running_loop()
        self._sync_future = loop.create_future()
        client = AttachClient(
            self._on_projection,
            self._on_disconnected,
            events=True,
            on_event=self._on_wire_event,
            on_attach_sync=self._on_attach_sync,
        )
        await client.connect(record, self._session_id)
        self._client = client

    async def _await_seed(self) -> LiveTurnSeed:
        future = self._sync_future
        if future is None:
            raise ConnectionError("owner did not start attach synchronization")
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=15.0)
        except TimeoutError as exc:
            raise ConnectionError("owner did not send attach synchronization") from exc

    async def _load_history(self) -> None:
        """Read model-facing durable history without acquiring the writer lease."""

        # BOTH the transcript construction and the history build are threaded.
        # ``Transcript.__init__`` eagerly reads and parses the whole file, so
        # constructing it on the loop was half the stall the to_thread below
        # was meant to remove; a long session's replay is file I/O plus JSON
        # parsing from end to end, with nothing the loop needs until the
        # result is bound. ``restore_todos`` stays ON the loop: it mutates
        # process-private state (the todo panel's store) that the seed events
        # replayed immediately after must observe.
        def _replay() -> tuple[list[Any], dict[str, Any] | None]:
            transcript = Transcript(self._config_dir / "sessions" / self._session_id)
            history = transcript.build_llm_history()
            details = transcript.latest_custom("todo_snapshot")
            return history, details

        history, details = await asyncio.to_thread(_replay)
        self._history = history
        self._history_ids = {str(message.id) for message in history if getattr(message, "id", None)}
        # Frames that arrived during the threaded replay were deduped against
        # a still-empty id set and buffered. The replay answer is now
        # authoritative: re-filter before anything drains, so a message that
        # landed durably mid-replay is not painted twice (once from history,
        # once from the buffered relay frame).
        self._filter_known_messages()
        if details and details.get("items"):
            # The standard TUI todo panel reads a process-local store. Restore
            # the newest durable snapshot so a follower's panel starts where
            # the owner did; relayed todo tool-end events trigger its normal
            # 1 Hz refresh afterward.
            from local_operator.tools.builtin import restore_todos

            restore_todos(self._session_id, list(details["items"]))

    def _finish_sync(self, seed: LiveTurnSeed) -> None:
        """Install seed events before post-sync frames and resume delivery.

        ``AttachClient._pump`` processes attach_sync and the next event in one
        loop turn. The future wakeup runs later, so the next event may already
        be buffered here. PREPENDING the seed is therefore load-bearing; an
        append would paint a tool result before the running card it settles.
        """
        seeded: list[AgentEvent[Any]] = []
        if not seed.streaming:
            self._streaming = False
        else:
            self._streaming = True
            self._generation = seed.generation
            seeded.append(AgentStartEvent(generation=seed.generation))
            if seed.assistant_open:
                identity = {"id": seed.assistant_message_id} if seed.assistant_message_id else {}
                message = Message.assistant(seed.assistant_text, **identity)
                seeded.append(MessageStartEvent(message=message))
                if seed.assistant_text:
                    seeded.append(MessageUpdateEvent(message=message, delta=seed.assistant_text))
            seeded.extend(deserialize_event(data) for data in seed.open_tools)
        self._buffered_events[0:0] = seeded
        self._ready_for_events = True
        self._owner_ready.set()
        self._drain_buffered_events()

    def _drain_buffered_events(self) -> None:
        """Deliver buffered sync frames once both ordering and a subscriber exist."""
        if not self._ready_for_events or not self._handlers or not self._buffered_events:
            return
        buffered, self._buffered_events = self._buffered_events, []
        for event in buffered:
            self._emit_or_buffer(event)

    def _on_attach_sync(self, data: dict[str, Any]) -> None:
        future = self._sync_future
        if future is not None and not future.done():
            future.set_result(LiveTurnSeed.from_json(data))

    def _on_wire_event(self, data: dict[str, Any]) -> None:
        event = deserialize_event(data)
        message = getattr(event, "message", None)
        message_id = str(getattr(message, "id", "") or "")
        # History was read after the socket began buffering. If a turn landed
        # durably in that window, its message-grade relay events are already in
        # history; dropping by stable message id prevents double painting.
        # The check runs again at DRAIN time (see ``_filter_known_messages``)
        # because the replay runs in a thread: a frame that arrives while the
        # ids are still empty passes HERE, sits in the buffer, and would
        # otherwise double-paint once the replayed history — which already
        # contains that message — is handed to the app.
        if message_id and message_id in self._history_ids:
            return
        if isinstance(event, AgentStartEvent):
            self._streaming = True
            self._generation = event.generation
        elif isinstance(event, AgentEndEvent):
            self._streaming = False
        self._emit_or_buffer(event)

    def _filter_known_messages(self) -> None:
        """Drop buffered events whose message the replayed history contains.

        The SECOND half of the double-paint guard above. ``_load_history``
        yields to the loop for the whole transcript replay (that is the A3
        fix), so relay frames can arrive between the socket opening and the
        ids binding — each one checked against a still-empty set and
        buffered. Anything that landed durably in that window is ALREADY in
        the replayed history, so re-filtering the buffer against the bound
        ids before delivery drops exactly those. Non-message events (tool
        cards, notices) keep flowing: they have no stable id to compare and
        their replay equivalent is not painted from history.
        """
        if not self._buffered_events:
            return
        kept: list[AgentEvent[Any]] = []
        for event in self._buffered_events:
            message = getattr(event, "message", None)
            message_id = str(getattr(message, "id", "") or "")
            if message_id and message_id in self._history_ids:
                continue
            kept.append(event)
        self._buffered_events = kept

    def _emit_or_buffer(self, event: AgentEvent[Any]) -> None:
        if not self._ready_for_events or not self._handlers:
            self._buffered_events.append(event)
            return
        for handler in list(self._handlers):
            result = handler(event)
            if inspect.isawaitable(result):
                asyncio.create_task(_await_handler(result))

    def _on_projection(self, projection: SessionProjection) -> None:
        self._projection = projection
        self._streaming = projection.streaming
        self._refresh_model(projection)
        self._sync_gate(projection.pending)

    def _refresh_model(self, projection: SessionProjection) -> None:
        selector = projection.model_selector or projection.model_label
        provider, _, model_id = selector.partition("/")
        if not model_id:
            provider, model_id = "unknown", selector or "unknown"
        effort = projection.effort or None
        self._model = ModelSpec(
            provider=provider,
            model_id=model_id,
            reasoning_effort=effort,
            reasoning_efforts=tuple(projection.effort_ladder),
        )

    # -- gate bridging ------------------------------------------------------

    @staticmethod
    def _gate_identity(pending: PendingRequest | None) -> tuple[str, str, int] | None:
        if pending is None:
            return None
        # Approvals never advance in place, so their synthetic index stays at
        # zero. Ask position must travel end-to-end because one request id names
        # the whole picker rather than one question within it.
        question_index = pending.question_index if pending.kind == "ask" else 0
        return (pending.kind, pending.request_id, question_index)

    def _sync_gate(self, pending: PendingRequest | None) -> None:
        key = self._gate_identity(pending)
        if key == self._gate_key:
            return
        if self._gate_task is not None:
            self._gate_task.cancel()
            self._gate_task = None
        self._gate_key = key
        if pending is not None:
            self._maybe_start_gate(pending)

    def _maybe_start_gate(self, pending: PendingRequest | None = None) -> None:
        if pending is None:
            projection = self._projection
            pending = projection.pending if projection else None
        if pending is None or self._gate_task is not None:
            return
        if pending.kind == "approval" and self._approval_handler is not None:
            self._gate_task = asyncio.create_task(self._run_approval(pending))
        elif pending.kind == "ask" and self._ask_handler is not None:
            self._gate_task = asyncio.create_task(self._run_ask(pending))

    async def _run_approval(self, pending: PendingRequest) -> None:
        try:
            handler = self._approval_handler
            client = self._client
            if handler is None or client is None:
                return
            approved = await call_approval_gate(handler, pending.title, pending.detail)
            await client.approval_answer(pending.request_id, approved)
        except (asyncio.CancelledError, RuntimeError):
            # Cancellation means another front end settled it. RuntimeError is
            # the owner's stale-request answer to the losing race. Both are an
            # ordinary first-valid-answer-wins outcome; the projection removes
            # the card.
            pass
        finally:
            if self._gate_key == self._gate_identity(pending):
                self._gate_task = None

    async def _run_ask(self, pending: PendingRequest) -> None:
        try:
            handler = self._ask_handler
            client = self._client
            if handler is None or client is None:
                return
            options = [
                AskOption(label=option.label, description=option.description)
                for option in pending.options
            ]
            question = AskQuestion(
                id=pending.request_id,
                question=pending.title,
                options=options,
                secret=pending.secret,
            )
            answer = await handler([question])
            if not answer:
                return
            values = answer.get(pending.request_id) or []
            if values:
                await client.ask_answer(
                    pending.request_id,
                    values[0],
                    question_index=pending.question_index,
                )
        except (asyncio.CancelledError, RuntimeError):
            pass
        finally:
            if self._gate_key == self._gate_identity(pending):
                self._gate_task = None

    # -- owner loss ---------------------------------------------------------

    def _on_disconnected(self, _reason: str) -> None:
        if self._disposed or self._recovering:
            return
        self._recovering = True
        self._owner_ready.clear()
        # A killed owner factually aborted the in-flight turn. Mark it through
        # the normal event path; no card/banner or attach vocabulary appears.
        if self._streaming:
            self._emit_or_buffer(
                AgentEndEvent(aborted=True, generation=self._generation, error=None)
            )
            self._streaming = False
        self._recovery_task = asyncio.create_task(self._recover_owner())

    async def _recover_owner(self) -> None:
        delay = 0.1
        try:
            while not self._disposed:
                record, _ = await asyncio.to_thread(
                    find_owner_record, self._config_dir, self._session_id
                )
                if record is not None and record.protocol >= 4:
                    try:
                        await self._dial(record)
                        seed = await self._await_seed()
                        self._finish_sync(seed)
                        return
                    except (ConnectionError, OSError, TimeoutError):
                        pass
                else:
                    try:
                        local = await self._takeover_factory()
                    except SessionLeaseHeldError:
                        # Another follower won the kernel-arbitrated stale
                        # recovery lock. Back off, then discover its fresh
                        # registrant record and reattach.
                        pass
                    except Exception:
                        logger.debug("remote takeover attempt failed", exc_info=True)
                    else:
                        callback = self._takeover_callback
                        if callback is not None:
                            result = callback(local)
                            if inspect.isawaitable(result):
                                await result
                            self._takeover_target = local
                            self._owner_ready.set()
                            return
                        # Adoption normally installed the callback before a
                        # disconnect can happen; if it did not, avoid leaking
                        # the writer lease we just won.
                        await local.dispose()
                await asyncio.sleep(delay)
                delay = min(delay * 1.7, 0.5)
        finally:
            self._recovering = False

    def set_takeover_callback(self, callback: Callable[[Any], Any]) -> None:
        self._takeover_callback = callback

    # -- SessionProtocol identity/state ------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_id(self) -> str:
        return "main"

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def model_label(self) -> str:
        return self._projection.model_label if self._projection else "unknown/unknown"

    @property
    def model(self) -> ModelSpec:
        return self._model

    @property
    def effective_model(self) -> ModelSpec:
        return self._model

    @property
    def effective_model_label(self) -> str:
        return self.model_label

    def set_model(self, model: ModelSpec, *, explicit: bool = False) -> None:
        old = self._model
        self._model = model
        client = self._client
        if client is None:
            return
        # /effort changes only the reasoning rung; /model changes identity.
        if (model.provider, model.model_id) == (old.provider, old.model_id):
            effort = model.reasoning_effort or "auto"
            asyncio.create_task(client.set_effort(effort))
        else:
            asyncio.create_task(client.set_model(model.provider, model.model_id))

    @property
    def goal(self) -> str:
        return ""

    def set_goal(self, text: str) -> str:
        client = self._client
        if client is not None:
            asyncio.create_task(client.slash("goal", text))
        return text.strip()

    @property
    def conversation_name(self) -> str:
        return self._projection.conversation_name if self._projection else ""

    @property
    def conversation_name_state(self) -> ConversationName:
        self._name_state.set(self.conversation_name, user_set=False)
        return self._name_state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        client = self._client
        if client is not None:
            asyncio.create_task(client.slash("rename", text))
        return text.strip()

    # -- history / host errands --------------------------------------------

    def history(self) -> list[Any]:
        return list(self._history)

    async def complete_once(self, system: str, prompt: str) -> str:
        raise RuntimeError("provider errands run on the session owner")

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> str:
        raise RuntimeError("/btw is unavailable while another process owns the session")

    async def adopt_aside(self, messages: list[Message]) -> None:
        raise RuntimeError("aside adoption is unavailable while another process owns the session")

    async def route_shared_slash(self, command: str, args: str) -> str:
        """Run a conversation-mutating slash command on the authoritative host.

        OperatorApp handles process-local navigation/config itself. This seam is
        only for commands whose normal local implementation depends on owner
        registries or turn workers (/agent, /team, /loop), so the follower does
        not maintain a second copy of shared orchestration state.
        """
        client = self._client
        if client is None:
            raise ConnectionError("session owner is reconnecting")
        return await client.slash(command, args)

    async def compact_now(self) -> CompactionOutcome:
        client = self._client
        if client is None:
            return CompactionOutcome(False, "unavailable", "session owner is reconnecting")
        detail = await client.slash("compact", "")
        return CompactionOutcome(True, detail=detail)

    # -- driving turns ------------------------------------------------------

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        await self._owner_ready.wait()
        target = self._takeover_target
        if target is not None:
            await target.prompt(text, images)
            return
        client = self._client
        if client is None or not client.connected:
            raise ConnectionError("session owner is reconnecting")
        command = ContinuationCommand.create(
            self._session_id,
            text,
            [_image_to_wire(image) for image in (images or [])],
        )
        await client.send_command(command, streaming=self._streaming)

    async def seed_history(self, messages: list[Message]) -> None:
        if self._history:
            return
        self._history = list(messages)

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.steer_message(Message.user(text, images))

    def steer_message(self, message: Message) -> None:
        self._queued_steers.append(message)
        asyncio.create_task(self._send_steer_when_ready(message))

    async def _send_steer_when_ready(self, message: Message) -> None:
        """Retain a queued steer across silent reattach/takeover."""
        await self._owner_ready.wait()
        target = self._takeover_target
        if target is not None:
            target.steer_message(message)
            return
        client = self._client
        if client is None or not client.connected:
            return
        command = ContinuationCommand(
            command_id=message.id,
            session_id=self._session_id,
            text=message.text,
            images=[
                _image_to_wire(block)
                for block in message.content
                if isinstance(block, ImageContent)
            ],
        )
        await client.send_command(command, streaming=True)

    def queued_steering(self) -> list[Any]:
        return list(self._queued_steers)

    def recall_steering(self, message: Any) -> bool:
        if message not in self._queued_steers:
            return False
        self._queued_steers.remove(message)
        client = self._client
        if client is not None:
            asyncio.create_task(client.recall_steer(str(message.id)))
        return True

    def abort(self, reason: str = "interrupted") -> None:
        if self._client is not None:
            asyncio.create_task(self._client.abort())

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        if self._client is not None:
            asyncio.create_task(self._client.slash("stop", ""))
        return 0

    def running_subagents(self) -> int:
        projection = self._projection
        if projection is None:
            return 0
        return sum(1 for row in projection.subagents if row.status == "running")

    def set_approval_handler(self, handler: ApprovalGate | None) -> None:
        self._approval_handler = handler
        self._maybe_start_gate()

    def set_ask_handler(self, handler: AskUserFn | None) -> None:
        self._ask_handler = handler
        self._maybe_start_gate()

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._handlers.append(handler)
        if len(self._handlers) == 1:
            self._drain_buffered_events()

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def dispose(self) -> None:
        self._disposed = True
        if self._gate_task is not None:
            self._gate_task.cancel()
        if self._recovery_task is not None and self._recovery_task is not asyncio.current_task():
            # The takeover callback adopts the real Session and disposes this
            # facade FROM the recovery task. Cancelling the current task there
            # interrupts adoption halfway through and strands the lease winner.
            self._recovery_task.cancel()
        if self._client is not None:
            self._client.close()
            self._client = None


async def _await_handler(result: Any) -> None:
    """Turn an EventHandler's generic Awaitable into a concrete coroutine.

    ``asyncio.create_task`` intentionally requires a coroutine rather than the
    broader Awaitable protocol. This wrapper preserves the SessionProtocol's
    sync-or-async handler contract without weakening types at the call site.
    """
    await result


def _image_to_wire(image: ImageContent) -> dict[str, str]:
    return {"data_b64": image.data, "mime_type": image.mime_type}

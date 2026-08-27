"""Full-fidelity remote session facade for a follower TUI (protocol v5).

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
    HistoryDeltaEvent,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    ModelSpec,
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
from local_operator.mobile.attach_client import AttachClient, find_owner_record
from local_operator.mobile.types import (
    ContinuationCommand,
    PendingRequest,
    SessionRecord,
)
from local_operator.session.frontend_state import (
    FRONTEND_CAPABILITY,
    FrontendSessionState,
    FrontendStateStore,
    FrontendSync,
    FrontendUpdate,
    SnapshotJobs,
    SnapshotMcpManager,
    SnapshotSubagentComms,
    SnapshotWakeScheduler,
)
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome
from local_operator.session.transcript import Transcript
from local_operator.session_lease import SessionLeaseHeldError

logger = logging.getLogger(__name__)

#: User-facing refusal for a routed slash submitted while the owner is being
#: recovered. One string, formatted with the command, so every seam that can
#: race the gap says the same thing — never the transport's ``not attached``.
_RECONNECTING_SLASH_NOTICE = "session is reconnecting; try /{command} again in a moment"

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
        HistoryDeltaEvent,
        ToolCallComposeEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        ToolExecutionEndEvent,
        NoticeEvent,
        PeerMessageDeliveredEvent,
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


#: Delivery rank of one message-grade event within its id's lifecycle. A
#: replay at or below the delivered rank is a duplicate; anything above it is
#: the legitimate next beat of the SAME live message.
_MESSAGE_PHASE: dict[type, int] = {
    MessageStartEvent: 1,
    MessageUpdateEvent: 2,
    MessageEndEvent: 3,
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
    """A SessionProtocol facade backed by one owner's v5 attach socket."""

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
        # The projection callback authenticates the welcome identity only. Full
        # TUI semantics come exclusively from the canonical v5 state stream.
        self._frontend_future: asyncio.Future[FrontendSync] | None = None
        self._frontend_store: FrontendStateStore | None = None
        self.jobs = SnapshotJobs()
        self.wake_scheduler = SnapshotWakeScheduler()
        self.mcp_manager = SnapshotMcpManager()
        # The full-page subagent view's hierarchy keys read this facade; built
        # from the canonical jobs so a follower's parent/peer/child navigation
        # works on the same authoritative graph the owner's does (U5).
        self._subagent_comms = SnapshotSubagentComms()
        self.mcp_startup: Any | None = None
        self._history: list[Any] = []
        self._history_ids: set[str] = set()
        # Message ids whose row the follower has ALREADY painted live. The sync
        # seed and relayed stream are filtered against this set as well as
        # ``_history_ids``, so a turn that became durable mid-join — or a
        # completed turn re-advertised by a fresh sync after reconnect — can
        # never paint twice (M4). Rebuilt on each sync from durable rows +
        # painted ids.
        self._message_events: set[str] = set()
        # Lifecycle-progress filter for the live relay, separate from
        # ``_message_events`` by necessity: one message id stamps its START,
        # UPDATE and END events alike, so dedupe by id alone dropped the update
        # and end of every live message (BLOCKER-1, review round 2). The value
        # ranks the phases 0→3 and only a phase at or BELOW the rank already
        # delivered for that id is a true replay duplicate — a legitimately
        # later phase for the same id must always pass.
        self._live_message_phase: dict[str, int] = {}
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
        self._name_state = ConversationName()
        self._model: ModelSpec | None = None
        # Double-Esc subagent cancel: the synchronous protocol method issues
        # the authoritative op, and the owner's confirmed count replaces the
        # optimistic notice through this app-installed callback.
        self._cancel_resolution: Callable[[int], None] | None = None
        self._cancel_task: asyncio.Task[None] | None = None

    @classmethod
    async def connect(
        cls,
        record: SessionRecord,
        session_id: str,
        *,
        config_dir: Path,
        takeover_factory: Callable[[], Any],
    ) -> "RemoteSession":
        if record.protocol < 5 or FRONTEND_CAPABILITY not in record.capabilities:
            raise ConnectionError(
                f"owner lacks {FRONTEND_CAPABILITY}; canonical full-TUI attach needs protocol >= 5"
            )
        self = cls(
            config_dir=config_dir,
            session_id=session_id,
            takeover_factory=takeover_factory,
        )
        await self._dial(record)
        frontend = await self._await_frontend()
        self._install_frontend(frontend.snapshot)
        await self._load_history(frontend.live_cursor)
        self._finish_sync()
        return self

    async def _dial(self, record: SessionRecord) -> None:
        # Freeze relay delivery until the canonical sync is installed ahead of
        # raw event frames that follow it on the same socket.
        self._ready_for_events = False
        loop = asyncio.get_running_loop()
        self._frontend_future = loop.create_future()
        client = AttachClient(
            lambda _projection: None,
            self._on_disconnected,
            events=True,
            on_event=self._on_wire_event,
            frontend_state=True,
            on_frontend_sync=self._on_frontend_sync,
            on_frontend_update=self._on_frontend_update,
        )
        await client.connect(record, self._session_id)
        self._client = client

    async def _await_frontend(self) -> FrontendSync:
        future = self._frontend_future
        if future is None:
            raise ConnectionError("owner did not start frontend synchronization")
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=15.0)
        except TimeoutError as exc:
            raise ConnectionError("owner did not send frontend synchronization") from exc

    async def _load_history(
        self, live_cursor: str | None = None, *, drop_history_duplicates: bool = True
    ) -> None:
        """Read durable history exactly up to the sync's advertised boundary.

        ``live_cursor`` is the owner's ``history_cursor``: the id of the newest
        transcript entry that was durable when the sync snapshot was taken. The
        atomic sync boundary means durable rows <= cursor are already captured
        in the snapshot's history, and events > cursor arrive as the live
        suffix. Filtering the durable read to <= cursor (rather than reading
        the whole transcript) is what makes the boundary EXACT: a message that
        became durable between snapshot and this read is NOT double-loaded,
        because its live event is what paints it.

        BOTH the transcript construction and the history build are threaded
        (#300): ``Transcript.__init__`` eagerly reads and parses the whole
        file, so a long session's replay is file I/O plus JSON parsing from
        end to end, with nothing the loop needs until the result is bound.
        """
        entries, history = await self._read_transcript()
        self._bind_history(
            entries, history, live_cursor, drop_history_duplicates=drop_history_duplicates
        )

    async def _read_transcript(self) -> tuple[list[Any], list[Any]]:
        """Parse the durable transcript off-loop, once per sync.

        The single threaded read shared by initial connect AND reconnect:
        review round 3 (MAJOR-2) found the reconnect path re-running this
        exact parse synchronously on the event loop — a 60 MB transcript
        blocked it for ~90 ms, past the 50 ms no-stall bar #300 established
        for the connect path. Both callers now consume ONE threaded result
        (gap projection and ``_history`` reconciliation), so the file is
        parsed once per sync and never on the loop.
        """

        def _replay() -> tuple[list[Any], list[Any]]:
            transcript = Transcript(self._config_dir / "sessions" / self._session_id)
            return transcript.entries(), transcript.build_llm_history()

        return await asyncio.to_thread(_replay)

    def _bind_history(
        self,
        entries: list[Any],
        history: list[Any],
        live_cursor: str | None,
        *,
        drop_history_duplicates: bool,
    ) -> None:
        """Adopt one parsed transcript as ``_history``, bounded by the cursor."""
        if live_cursor is not None:
            # Keep only the message entries at or before the advertised cursor.
            # The cursor names a transcript ENTRY id (any type); walk to it and
            # drop the durable suffix past the boundary the sync already sealed.
            boundary_index = None
            for index, entry in enumerate(entries):
                if entry.id == live_cursor:
                    boundary_index = index
                    break
            if boundary_index is not None:
                kept_ids = {entry.id for entry in entries[: boundary_index + 1]}
                history = [
                    message
                    for message in history
                    if not getattr(message, "id", None) or str(message.id) in kept_ids
                ]
        self._history = history
        self._history_ids = {
            str(message.id) for message in self._history if getattr(message, "id", None)
        }
        # Frames that arrived during the threaded replay were deduped against
        # a still-empty id set and buffered (#300 F2). The replay answer is now
        # authoritative: re-filter before anything drains, so a message that
        # landed durably mid-replay is not painted twice (once from history,
        # once from the buffered relay frame). INITIAL connect only: the app
        # renders the loaded history there, so a buffered frame for a durable
        # row is a duplicate. On RECONNECT nothing re-renders history — the
        # buffered gap-replay events (U6) and any live frame that settled
        # mid-reload are the ONLY paint those rows get, so dropping them here
        # would reproduce the invisible-recovery bug the gap replay closes.
        if drop_history_duplicates:
            self._filter_known_messages()
        # Anything durable is by definition already accounted for; seed the
        # painted-id filter from it so the live seed cannot repaint those rows.
        self._message_events |= self._history_ids
        # A reconnect loads fresh durable rows and reseeds the in-flight suffix
        # from the snapshot, so the per-id lifecycle rank is rebuilt with the
        # paint stream. Ids that ended stay in ``_message_events`` regardless;
        # clearing the rank here only affects messages whose lifecycle is still
        # open and will be re-seeded by ``_finish_sync``.
        self._live_message_phase = {}

    def _finish_sync(self) -> None:
        """Install the canonical in-flight seed before post-sync events.

        The seed shares the snapshot boundary with every other frontend field,
        so there is no second live-turn reducer whose cursor can race state.
        Each seeded event is filtered against the durable/painted id sets — a
        turn that became durable between snapshot and history load is dropped
        here rather than painted twice.
        """
        seeded = []
        for data in self.frontend_state.live_events:
            event = deserialize_event(data)
            if self._is_duplicate(event):
                continue
            self._track(event)
            seeded.append(event)
        if self.frontend_state.streaming:
            seeded.insert(0, AgentStartEvent(generation=self.frontend_state.generation))
        # Durable-before-live is the transcript's paint invariant. On
        # reconnect the buffer's head can hold the gap's HistoryDeltaEvent
        # (rows OLDER than everything the seed and relay carry); inserting
        # the seed at 0 unconditionally would mount the in-flight turn above
        # the durable rows it follows. Initial connect buffers no delta, so
        # this degenerates to the plain front-insert it always was.
        insert_at = 0
        while insert_at < len(self._buffered_events) and isinstance(
            self._buffered_events[insert_at], HistoryDeltaEvent
        ):
            insert_at += 1
        self._buffered_events[insert_at:insert_at] = seeded
        self._ready_for_events = True
        self._owner_ready.set()
        self._drain_buffered_events()

    def _replay_durable_suffix(self, history: list[Any]) -> None:
        """Emit ONE typed history delta for durable rows nothing ever painted.

        ``history`` is the reconnect's single threaded transcript parse (the
        same result ``_bind_history`` adopts): on the reconnect path this runs
        before the fresh history bind, and the pre-disconnect ``_history`` is
        exactly the rows that need no replay. Only rows whose id is absent
        from ``_message_events`` (the painted set) are gathered, so a
        reconnect after a quiet gap replays nothing and a reconnect across a
        missed turn repaints exactly that turn. Each replayed id is claimed,
        so the follow-up sync's live seed, the history bind and any later
        relay dedupe against it rather than re-painting.

        The gap goes out as a single :class:`HistoryDeltaEvent` rather than
        per-row ``message_end`` events. A ``message_end`` is a LIVE assistant
        contract — the controller adopts its text into the streaming block —
        so replaying a user prompt, a tool call/result pair, or a custom row
        through it painted every role as assistant prose and dropped tools,
        images and custom blocks entirely (review round 3, MAJOR-1/U7/D1).
        The typed delta hands the settled rows — INCLUDING role-less tool
        results, which the settled renderer pairs with their calls — to the
        same role-aware projector a cold resume uses, so a recovered
        transcript is indistinguishable from one that never disconnected.
        """
        gap: list[Any] = []
        claimed: list[str] = []
        for message in history:
            message_id = str(getattr(message, "id", "") or "")
            if not message_id or message_id in self._message_events:
                continue
            claimed.append(message_id)
            gap.append(message)
        if not gap:
            # Nothing new became durable in the gap.
            return
        # A gap of ONLY tool results still delivers: the calls painted live
        # before the disconnect (their cards sit in ``_tool_cards`` marked
        # ``interrupted``), and the settled renderer now resolves each
        # recovered result back onto its painted card rather than painting a
        # new row (review round 4, MINOR-1). The old early-return dropped the
        # gap entirely, leaving those cards interrupted forever while
        # ``history()`` carried their real output.
        self._message_events.update(claimed)
        # Durable-before-live is a DELIVERY invariant, not just a seed-vs-delta
        # one. The recovery sequence dials, then awaits the frontend sync (a
        # network round trip) and the threaded transcript parse before this
        # method runs — and the reader task keeps buffering live relay frames
        # throughout, so a reconnect to a streaming replacement owner leaves
        # the buffer as [live frames…, delta]. A plain append delivers those
        # frames first and paints the durable gap rows BELOW the in-flight
        # turn (review round 4, MAJOR-1), which no cold boot of the same
        # transcript can ever look like. Placing the delta at the buffer's
        # head — after any delta already sitting there, the mirror of
        # ``_finish_sync``'s skip loop, so multiple recovery cycles keep their
        # own order — makes the guarantee positional, never a timing
        # accident. Initial connect buffers no delta, so nothing changes there.
        insert_at = 0
        while insert_at < len(self._buffered_events) and isinstance(
            self._buffered_events[insert_at], HistoryDeltaEvent
        ):
            insert_at += 1
        self._buffered_events[insert_at:insert_at] = [HistoryDeltaEvent(messages=gap)]

    def _is_duplicate(self, event: AgentEvent[Any]) -> bool:
        """Whether this event is a true replay duplicate, never a lifecycle peer.

        Two independent seams can replay a row — durable history and the sync
        seed — and the live relay's own phases are NOT a third. A message id
        already in the painted/durable set means the row is complete: a START
        or END for it is a replay. In between, the phases must flow: a START
        does not make its UPDATE or END a duplicate, because those are the
        events that carry the content the start only announced. The phase
        ranks in ``_MESSAGE_PHASE`` make "at or below what we already
        delivered" the duplicate test.
        """
        message = getattr(event, "message", None)
        message_id = str(getattr(message, "id", "") or "")
        if not message_id:
            return False
        if isinstance(event, (MessageStartEvent, MessageEndEvent)):
            # Durable or already-painted-complete: a replayed row, never the
            # same live message's first/last beat.
            if message_id in self._message_events:
                return True
        phase = _MESSAGE_PHASE.get(type(event))
        if phase is None:
            return False
        delivered = self._live_message_phase.get(message_id, -1)
        # A phase BELOW the delivered rank is a replay. At the SAME rank it
        # depends on the phase: a second update for one in-flight message is
        # the next legitimate beat (deltas are incremental and the UIs
        # coalesce them), while a repeated start or end is a true duplicate.
        if phase < delivered:
            return True
        if phase == delivered and not isinstance(event, MessageUpdateEvent):
            return True
        return False

    def _track(self, event: AgentEvent[Any]) -> None:
        """Record a painted live message id so a later sync/durable row skips it."""
        message = getattr(event, "message", None)
        message_id = str(getattr(message, "id", "") or "")
        if not message_id:
            return
        phase = _MESSAGE_PHASE.get(type(event))
        if phase is None:
            return
        # Monotonic per id: a regressed phase is ignored by ``_is_duplicate``
        # anyway, so the rank only ever moves forward.
        if phase > self._live_message_phase.get(message_id, -1):
            self._live_message_phase[message_id] = phase
        # The row is SETTLED once its complete form is known — an END event,
        # or a START whose message already carries its content (the durable
        # seed folds completed rows in as message_start entries, and the
        # join-time seed is exactly a replay of durable-looking events; M4
        # then needs the id claimed immediately or a snapshot taken just
        # after the end event would repaint the row). A bare START claims
        # nothing — its update and end share the id.
        if isinstance(event, MessageEndEvent) or (
            isinstance(event, MessageStartEvent)
            and bool(getattr(message, "text", "") or getattr(message, "tool_calls", None))
        ):
            self._message_events.add(message_id)

    def _drain_buffered_events(self) -> None:
        """Deliver buffered sync frames once both ordering and a subscriber exist."""
        if not self._ready_for_events or not self._handlers or not self._buffered_events:
            return
        buffered, self._buffered_events = self._buffered_events, []
        for event in buffered:
            self._emit_or_buffer(event)

    def _on_frontend_sync(self, data: dict[str, Any]) -> None:
        future = self._frontend_future
        if future is not None and not future.done():
            future.set_result(FrontendSync.model_validate(data))

    def _on_frontend_update(self, data: dict[str, Any]) -> None:
        update = FrontendUpdate.model_validate(data)
        if self._frontend_store is None:
            raise ConnectionError("frontend update arrived before synchronization")
        state = self._frontend_store.apply_update(update)
        self._apply_frontend_facades(state)

    def _install_frontend(self, state: FrontendSessionState, *, publish: bool = False) -> None:
        if state.session_id != self._session_id:
            raise ConnectionError("frontend state belongs to another session")
        if self._frontend_store is None:
            self._frontend_store = FrontendStateStore(state)
        elif publish:
            self._frontend_store.replace_and_notify(state)
        else:
            self._frontend_store.replace(state)
        self._apply_frontend_facades(state)

    def _apply_frontend_facades(self, state: FrontendSessionState) -> None:
        """Refresh compatibility facades after one canonical install."""
        self._streaming = state.streaming
        self._generation = state.generation
        self._model = state.selected_model
        self.jobs.replace(state.jobs)
        self._subagent_comms.replace(state.jobs)
        self.wake_scheduler.replace(state.wakes)
        self.mcp_manager.replace(state.mcp_servers)
        startup = state.mcp_startup
        if isinstance(startup, dict):
            from local_operator.session.mcp_status import McpStartupOutcome

            startup = McpStartupOutcome(
                configured=tuple(startup.get("configured", ()) or ()),
                connected=tuple(startup.get("connected", ()) or ()),
                failures=dict(startup.get("failures", {}) or {}),
                tool_count=int(startup.get("tool_count", 0) or 0),
                settling=bool(startup.get("settling", False)),
            )
        self.mcp_startup = startup
        self._name_state.set(state.conversation_title, user_set=state.conversation_title_user_set)
        self._apply_pending_gate(_pending_request(state.pending_gate))

    def _on_wire_event(self, data: dict[str, Any]) -> None:
        event = deserialize_event(data)
        # A message-grade event whose row is already durable (history was read
        # after the socket began buffering) or already painted live is dropped
        # by stable message id — the single dedup rule for both seams. The
        # check runs again at DRAIN time (see ``_filter_known_messages``)
        # because the replay runs in a thread: a frame that arrives while the
        # ids are still empty passes HERE, sits in the buffer, and would
        # otherwise double-paint once the replayed history — which already
        # contains that message — is handed to the app.
        if self._is_duplicate(event):
            return
        if isinstance(event, AgentStartEvent):
            self._streaming = True
            self._generation = event.generation
        elif isinstance(event, AgentEndEvent):
            self._streaming = False
        self._track(event)
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

    def _apply_pending_gate(self, pending: PendingRequest | None) -> None:
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
            pending = _pending_request(self.frontend_state.pending_gate)
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
                AskOption(
                    label=(option.get("label", "") if isinstance(option, dict) else option.label),
                    description=(
                        option.get("description", "")
                        if isinstance(option, dict)
                        else option.description
                    ),
                )
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
                if (
                    record is not None
                    and record.protocol >= 5
                    and FRONTEND_CAPABILITY in record.capabilities
                ):
                    try:
                        await self._dial(record)
                        frontend = await self._await_frontend()
                        self._install_frontend(frontend.snapshot, publish=True)
                        # ONE threaded parse feeds both the gap replay and the
                        # history bind: reconnect must not re-parse the file on
                        # the event loop (review round 3, MAJOR-2 — a 60 MB
                        # transcript stalled it ~90 ms) and must not parse it
                        # twice. The replay must still run BEFORE the bind:
                        # ``_bind_history`` seeds the painted-id set from every
                        # durable row it adopts, so binding first would mark
                        # the gap rows painted before their delta was ever
                        # emitted — recovery then "succeeds" with the rows in
                        # ``history()`` but never on screen (U6, review round
                        # 2). The replay claims exactly the durable ids this
                        # follower has not painted; the bind afterwards brings
                        # ``_history`` to the same point and the live seed
                        # dedupes against the ids the replay just claimed (M4).
                        entries, history = await self._read_transcript()
                        self._replay_durable_suffix(history)
                        self._bind_history(
                            entries,
                            history,
                            frontend.live_cursor,
                            drop_history_duplicates=False,
                        )
                        self._finish_sync()
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

    def set_cancel_resolution(self, resolver: Callable[[int], None] | None) -> None:
        """Install the app's handler for an owner-confirmed subagent cancel count.

        Called with the REAL number the owner stopped (or ``-1`` on a failed
        request) so the double-Esc notice can be rewritten from the optimistic
        count to the authoritative one. ``None`` disarms it.
        """
        self._cancel_resolution = resolver

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
    def frontend_state(self) -> FrontendSessionState:
        if self._frontend_store is None:
            raise RuntimeError("frontend state has not synchronized")
        return self._frontend_store.state

    def subscribe_frontend(self, handler):  # type: ignore[no-untyped-def]
        if self._frontend_store is None:
            raise RuntimeError("frontend state has not synchronized")
        return self._frontend_store.subscribe(handler)

    @property
    def model_label(self) -> str:
        return self.frontend_state.model_label

    @property
    def model(self) -> ModelSpec:
        model = self.frontend_state.selected_model
        if model is None:
            raise RuntimeError("owner has no selected model spec")
        return model

    @property
    def effective_model(self) -> ModelSpec:
        model = self.frontend_state.effective_model or self.frontend_state.selected_model
        if model is None:
            raise RuntimeError("owner has no effective model spec")
        return model

    @property
    def effective_model_label(self) -> str:
        return self.frontend_state.effective_model_label

    def set_model(self, model: ModelSpec, *, explicit: bool = False) -> None:
        old = self.model
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
        return self.frontend_state.goal

    def set_goal(self, text: str) -> str:
        client = self._client
        if client is not None:
            asyncio.create_task(client.slash("goal", text))
        return text.strip()

    @property
    def conversation_name(self) -> str:
        return self.frontend_state.conversation_title

    @property
    def conversation_name_state(self) -> ConversationName:
        return self._name_state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        client = self._client
        if client is not None:
            asyncio.create_task(client.slash("rename", text))
        return text.strip()

    # -- history / host errands --------------------------------------------

    def history(self) -> list[Any]:
        return list(self._history)

    def context_breakdown(self) -> dict[str, int]:
        return dict(self.frontend_state.context_breakdown or {})

    async def complete_once(self, system: str, prompt: str) -> str:
        raise RuntimeError("provider errands run on the session owner")

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> str:
        client = self._client
        if client is None:
            raise ConnectionError("session owner is reconnecting")
        # The authoritative request currently returns a settled answer. Feed it
        # through the normal delta callback once so the existing aside widget
        # uses the same rendering path without inventing remote-only UI state.
        answer = await client.complete_aside(
            [turn.model_dump(mode="json") for turn in turns if hasattr(turn, "model_dump")]
        )
        if answer and on_delta is not None:
            on_delta(answer)
        return answer

    async def adopt_aside(self, messages: list[Message]) -> None:
        """Promote the aside exchange into the conversation through the owner.

        The Ctrl+F fork is advertised on the standard aside card, so it must
        work on a follower too. The owner appends the pair to its live context
        and transcript (the same idle-turn guard and durable-first order as a
        local :meth:`Session.adopt_aside`), then the canonical frontend update
        carries the new rows to every terminal — the follower does not splice
        anything itself.
        """
        client = self._client
        if client is None:
            raise ConnectionError("session owner is reconnecting")
        await client.adopt_aside(
            [
                message.model_dump(mode="json")
                for message in messages
                if hasattr(message, "model_dump")
            ]
        )

    async def route_shared_slash(
        self,
        command: str,
        args: str,
        images: Sequence[ImageContent] | None = None,
    ) -> Any:
        """Run a conversation-mutating slash command on the authoritative host.

        OperatorApp handles process-local navigation/config itself. This seam
        carries every command the owner's capability list marks
        ``authoritative_session``, so the follower never maintains a second
        copy of shared orchestration state. The owner returns a typed
        :class:`SlashResult` dict the invoker renders locally — the answer
        never paints in the owner's terminal.

        During owner recovery this REFUSES, in user vocabulary, rather than
        waiting like ``prompt()`` does. A prompt is fire-and-forget so queuing
        it across the gap is invisible; a slash is request/response, and a
        command that silently blocks until an owner returns minutes later
        answers a question the user has stopped asking — against whatever
        state the replacement owner has by then. The refusal names the retry,
        and the transport's own ``not attached`` wording must never surface
        (review round 3, MINOR-1/U8): the disconnect can also land mid-request,
        so the raced ``ConnectionError`` is rewritten below too.
        """
        client = self._client
        if client is None or self._recovering or not client.connected:
            raise ConnectionError(_RECONNECTING_SLASH_NOTICE.format(command=command))
        try:
            return await client.slash_result(
                command,
                args,
                [_image_to_wire(image) for image in (images or [])],
            )
        except ConnectionError as error:
            raise ConnectionError(_RECONNECTING_SLASH_NOTICE.format(command=command)) from error

    async def compact_now(self) -> CompactionOutcome:
        client = self._client
        if client is None or self._recovering or not client.connected:
            return CompactionOutcome(False, "unavailable", "session owner is reconnecting")
        try:
            detail = await client.slash("compact", "")
        except ConnectionError:
            # The disconnect can land mid-request (review round 4, NIT-1): the
            # transport's own ``not attached`` must never surface as a
            # compaction receipt, so race the same rewrite the routed-slash
            # seam performs above.
            return CompactionOutcome(False, "unavailable", "session owner is reconnecting")
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
        return [
            Message.user(
                str(item.get("text", "") or ""),
                id=str(item.get("id", "") or "remote-steer"),
            )
            for item in self.frontend_state.queued_steering
        ]

    def recall_steering(self, message: Any) -> bool:
        ids = {str(item.get("id", "") or "") for item in self.frontend_state.queued_steering}
        if str(getattr(message, "id", "") or "") not in ids:
            return False
        client = self._client
        if client is not None:
            asyncio.create_task(client.recall_steer(str(message.id)))
        return True

    def abort(self, reason: str = "interrupted") -> None:
        if self._client is not None:
            asyncio.create_task(self._client.abort())

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """Optimistic cancel: returns the running count the offer promised.

        ``SessionProtocol.cancel_subagents`` is synchronous (the Esc handler
        reads its count inline), but a follower's authoritative count lives on
        the owner across an async socket. The follower issues the typed
        ``cancel_subagents`` op and, when the owner confirms the REAL number,
        replaces its optimistic notice via the ``_cancel_resolution`` callback
        the app installs — so the completion text always reflects what actually
        stopped, never a guessed zero. Returning the current running count
        keeps the synchronous contract honest for the frame it is read on.
        """
        client = self._client
        if client is None:
            return 0
        offered = self.running_subagents()
        task = asyncio.ensure_future(self._resolve_cancel(client))
        self._cancel_task = task
        return offered

    async def _resolve_cancel(self, client: AttachClient) -> None:
        try:
            stopped = await client.cancel_subagents()
        except Exception:
            stopped = -1
        resolver = self._cancel_resolution
        if resolver is not None:
            resolver(stopped)

    @property
    def active_agent(self) -> str:
        return self.frontend_state.active_agent

    @property
    def active_team_name(self) -> str:
        return self.frontend_state.active_team

    def restored_usage(self) -> Usage | None:
        return self.frontend_state.last_usage

    def running_subagents(self) -> int:
        return sum(
            1
            for row in self.frontend_state.jobs
            if row.type == "task" and row.status == "running" and not row.queued
        )

    def owner_model_catalogue(self) -> list[dict[str, Any]]:
        """The owner's offerable model rows, as published canonical state.

        A follower's own provider controller describes the follower's
        credentials, which are not the ones the shared session can run on —
        the picker must offer the owner's rows (D3, review round 2).
        """
        return [dict(row) for row in self.frontend_state.model_catalogue]

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


def _pending_request(state: Any) -> PendingRequest | None:
    if state is None:
        return None
    return PendingRequest(
        request_id=state.request_id,
        kind=state.kind,
        title=state.title,
        detail=state.detail,
        options=state.options,
        secret=state.secret,
        question_index=state.question_index,
        question_total=state.question_total,
    )


def _image_to_wire(image: ImageContent) -> dict[str, str]:
    return {"data_b64": image.data, "mime_type": image.mime_type}

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
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

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
    ToolResult,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
    WakeDeliveredEvent,
)
from local_operator.mobile.attach_client import (
    RETIRING_REASON,
    STOPPED_REASON,
    AttachClient,
    find_owner_record,
)
from local_operator.mobile.types import (
    SLASH_ACTION_RECEIPTS,
    ContinuationCommand,
    PendingRequest,
    SessionRecord,
)
from local_operator.session.frontend_state import (
    FRONTEND_CAPABILITY,
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
    FrontendStateStore,
    FrontendSync,
    FrontendUpdate,
    JobState,
    SnapshotJobs,
    SnapshotMcpManager,
    SnapshotSubagentComms,
    SnapshotWakeScheduler,
)
from local_operator.session.history_window import DisplayHistoryWindow
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome
from local_operator.session.transcript import Transcript
from local_operator.session_lease import SessionLeaseHeldError

logger = logging.getLogger(__name__)

#: User-facing refusal for a routed slash submitted while the owner is being
#: recovered. One string, formatted with the command, so every seam that can
#: race the gap says the same thing — never the transport's ``not attached``.
_RECONNECTING_SLASH_NOTICE = "session is reconnecting; try /{command} again in a moment"

#: How long a VIEWER chases a vanished runtime before unbinding and going cold.
#: A runtime exits by design when it has nothing left to do, so owner loss is
#: usually not a crash at all — but a restart after a `kill -9` publishes a new
#: record within a second or two, so the window has to be wide enough to catch
#: that before concluding nothing is coming. Only the viewer path uses it; the
#: legacy attach path still recovers by taking over (see ``_can_go_cold``).
COLD_FALLBACK_S = 8.0

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


def _restored_job_rows(jobs: Sequence[Any]) -> list[Any]:
    """Roster rows as they must appear with NO runtime alive.

    The persisted roster records what each job's state WAS when it was
    written. With no runtime there is by definition nothing running, so a row
    restored verbatim paints a spinner for a child that cannot be working and
    the band counts it as live activity (UX review round 1, U1). That is worse
    than the empty panel this change replaces: an empty panel is obviously
    incomplete, while phantom activity is confidently wrong, and it invites a
    cancel that finds nothing. Non-terminal rows are common on disk — a
    session whose terminal was closed mid-run persists them by design.

    The rule is the one ``AsyncJobManager.restore`` already applies on the
    owner path, reproduced here because the cold viewer never builds a
    manager:

    * a ``running`` row that was PARKED (``queued``) never started, so it has
      no transcript to show or resume and is DROPPED — an ``interrupted`` row
      would invite a resume that finds nothing;
    * any other non-terminal row becomes ``interrupted``, the restore-only
      status that means "was cut off mid-run"; live readers already treat it
      as terminal, and it is what lets the panel offer to resume the child.

    Anything already terminal is untouched: ``completed``/``failed``/
    ``cancelled`` are facts the last runtime settled and this process must not
    relitigate.
    """
    rows: list[Any] = []
    for job in jobs:
        status = str(getattr(job, "status", "") or "")
        if status == "running":
            if bool(getattr(job, "queued", False)):
                continue
            rows.append(job.model_copy(update={"status": "interrupted", "restored": True}))
            continue
        rows.append(job.model_copy(update={"restored": True}))
    return rows


class RemoteSession:
    """A SessionProtocol facade backed by one owner's v5 attach socket."""

    is_remote = True

    def __init__(
        self,
        *,
        config_dir: Path,
        session_id: str,
        takeover_factory: Callable[[], Any],
        surface: str = "terminal",
    ) -> None:
        self._config_dir = config_dir
        self._session_id = session_id
        self._takeover_factory = takeover_factory
        self._surface = surface
        self._desktop_visible = False
        self._desktop_can_notify = False
        self._desktop_seen = 0.0
        self._client: AttachClient | None = None
        #: The pid of the runtime this viewer is dialed into, ``None`` while
        #: cold or between owners. Read by the TUI's startup-cleanup notice to
        #: tell "MY runtime removed sessions" from "some other launch did":
        #: the record on disk names the removing pid, and the viewer attached
        #: to that runtime is the one that should announce (UX round 3, U14).
        self._runtime_pid: int | None = None
        #: Where a runtime for this session should be started. Only a cold
        #: viewer needs it (an attached one inherits the runtime's own cwd);
        #: ``cold()`` sets the real value.
        self._cwd = ""
        #: Serialises ``_ensure_bound`` so concurrent first-writes engage one
        #: runtime between them rather than one each.
        self._bind_lock = asyncio.Lock()
        #: Whether owner loss may end in an unbound viewer rather than a
        #: takeover. True for a viewer (the runtime owns the lease and this
        #: process must never take it); left False for the legacy attach path,
        #: whose contract is still "recover the conversation into this
        #: process" and whose tests assert exactly that.
        #:
        #: The DESKTOP viewer takes the viewer contract too: its host survives
        #: the runtime and re-dials, so owner loss must leave it unbound rather
        #: than dragging the lease into a process the user cannot see. Every
        #: other surface keeps the legacy attach contract above.
        self._can_go_cold = surface == "desktop"
        #: The BUILD the runtime on the other end is running, read off its
        #: discovery record at dial. ``""`` on a facade that has never bound,
        #: and on one bound to a runtime older than the field — the TUI treats
        #: those two the same way (it has no build to compare against) and
        #: distinguishes them by whether the facade is cold, not by this
        #: value. See ``app.py::_check_build_skew``.
        self.owner_version: str = ""
        self.owner_source_ref: str = ""
        #: Why this viewer opened WITHOUT live state, when that was not the
        #: ordinary "no runtime was running" case. Set by the launcher when an
        #: attach to a live runtime failed and it fell back to cold; the TUI
        #: prints it once on adoption so the user is told why the session came
        #: up bare instead of being left to guess (UX round 1, U2).
        self.degraded_reason: str = ""
        #: Told when the runtime vanished for good; see ``_go_cold``.
        self._went_cold_callback: Callable[[], Any] | None = None
        #: Told when the runtime retired ITSELF for a newer build (the
        #: ``retiring`` frame): the app re-engages eagerly so the band never
        #: shows the cold state for a refresh the user did not ask for. See
        #: ``_go_cold(refresh=True)``.
        self._refresh_callback: Callable[[], Any] | None = None
        #: True once THIS follower asked the owner to stop the session
        #: (``request_stop`` acked) or the wire evidence says the session was
        #: deliberately ended (the owner served the stop and unpublished).
        #: Owner loss after THAT is the request landing, not a death:
        #: ``_recover_owner`` must not take over the conversation a stop
        #: just ended (it would republish a live record for a stopped
        #: session, and its next prompt would be refused against the
        #: ``stopped_at`` marker the stop stamped).
        self._deliberate_stop = False
        #: Told to the app when this viewer's session ends deliberately, so
        #: the screen can say so instead of reporting an owner-death recovery
        #: that is not happening.
        self._stopped_callback: Callable[[], Any] | None = None
        self._stopped_announced = False
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
        self._live_history: dict[str, Any] = {}
        self._display_window_requested = False
        self._owner_record: SessionRecord | None = None
        self._display_refresh_lock = asyncio.Lock()
        self._hydrated_once = False
        self._display_history: DisplayHistoryWindow | None = None
        self._history_hydrated = True
        self._durable_seed_ids: set[str] = set()
        self._durable_seed_tool_ids: set[str] = set()
        self._history_ids: set[str] = set()
        #: The durable frontend checkpoint, handed from the threaded history
        #: read to the cold path's restore so the roster/todo/title recovery
        #: costs no second parse. A small dict rather than the parsed
        #: ``Transcript`` deliberately: retaining the transcript pinned every
        #: entry (1.23x file size) for the life of a warm attach that never
        #: wanted it (review round 1, C2). Only ``cold()`` asks for it, and it
        #: is cleared the moment that consumes it.
        self._cold_checkpoint: dict[str, Any] | None = None
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
        self._gates_detached = False
        self._background_approval = False
        self._keep_gate_reply = False
        self._gate_answered_key: tuple[str, str, int] | None = None
        # Snapshot creation is not retryable: navigation must retire the UI,
        # not close the response channel after the owner has begun a copy.
        self._snapshot_clients: dict[AttachClient, int] = {}
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
        # Socket delivery can outrun the coroutine installing its initial sync.
        # Buffer that suffix until its epoch, rather than the cold disk epoch,
        # is authoritative. None means the canonical boundary is installed.
        self._pending_frontend_updates: list[FrontendUpdate] | None = None
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
        # Teams and agent profiles are LOCAL CONFIG, not runtime state: they
        # live in `<config_dir>/teams` and `<config_dir>/agents`, the same
        # files `lop team`/`lop agents` read with no session at all. So a
        # viewer answers them from its OWN config dir rather than asking a
        # runtime — which is what makes `/team` and `/agent` work on a COLD
        # session, where there is no runtime to ask and never will be until
        # the first message engages one.
        #
        # This is the invariant that regressed in the viewer transition: the
        # TUI reads both registries off the SESSION object
        # (`_team_registry()`, `_agent_profile_rows()`), and `Session`
        # supplied them while `RemoteSession` did not, so every `/team` and
        # `/agent` surface silently answered "unavailable" once `lop` stopped
        # building a `Session`. Anything the TUI reads off the session has to
        # exist on BOTH implementations or it fails only on the viewer path.
        #
        # Built lazily and cached: constructing a registry walks the config
        # tree (and `TeamRegistry.__init__` runs crash recovery), which a
        # session that never types `/team` must not pay at boot.
        self._team_registry_cache: Any | None = None
        self._agent_registry_cache: Any | None = None
        # FAILURE is latched as well as success (R3). Returning None out of the
        # `except` without recording it left the cache empty, so the next read
        # re-entered the whole constructor: 25 property reads against a raising
        # constructor measured 25 constructions. The picker re-derives its rows
        # on EVERY keystroke, so an unreadable registry would put a directory
        # walk plus a recovery probe on the typing path — the exact cost
        # `teams.py` keeps off that loop ("a reader that waited turned an
        # ordinary keystroke into a multi-second freeze").
        #
        # The latch is a COOLDOWN, not a tombstone (R7). A permanent latch
        # makes any transient failure — a full disk that clears, a directory
        # being rewritten by a concurrent `lop team` — cost `/team` and
        # `/agent` for the entire life of the session, silently and with no way
        # back short of restarting. Timestamps rather than a bool, so a burst
        # of keystrokes still pays exactly one construction while a genuinely
        # repaired registry heals on its own.
        #
        # Same shape and the same budget as `TeamRegistry`'s own read-path
        # recovery cooldown (`_READ_RECOVERY_COOLDOWN_S`), deliberately: one
        # retry convention in the codebase, not a second one invented here.
        self._team_registry_failed_at: float | None = None
        self._agent_registry_failed_at: float | None = None

    def _within_registry_cooldown(self, failed_at: float | None) -> bool:
        """Whether a failed registry construction is still too recent to retry.

        Keeps a burst of keystrokes to ONE construction (the picker re-derives
        its rows per character) while letting a repaired registry recover
        without restarting the session — see the constructor's note on why a
        permanent latch is the wrong trade.

        ``monotonic`` because this is an elapsed-time question: a wall-clock
        source would let an NTP correction either suppress the retry for hours
        or defeat the cooldown entirely.
        """
        if failed_at is None:
            return False
        from local_operator.teams import _READ_RECOVERY_COOLDOWN_S

        return (time.monotonic() - failed_at) < _READ_RECOVERY_COOLDOWN_S

    @property
    def team_registry(self) -> Any | None:
        """The teams on this machine, read from the viewer's own config dir.

        Mirrors the attribute ``Session`` carries so the TUI's
        ``_team_registry()`` — a plain ``getattr(session, "team_registry")`` —
        resolves identically whichever session implementation it holds.

        Failure degrades ONE feature rather than the session, the same
        discipline ``session_factory`` applies to its own construction: a
        stranded backup or an unreadable ``teams`` directory leaves ``/team``
        reporting "teams are unavailable" instead of taking the conversation
        down with it. The registry itself still refuses to answer with a
        half-truth (see ``TeamRegistry._raise_if_recovery_failed``).
        """
        if self._team_registry_cache is None:
            if self._within_registry_cooldown(self._team_registry_failed_at):
                return None
            from local_operator.teams import TeamRegistry

            try:
                self._team_registry_cache = TeamRegistry(self._config_dir)
            except Exception:  # noqa: BLE001 — one feature must not break the session
                self._team_registry_failed_at = time.monotonic()
                return None
        return self._team_registry_cache

    @property
    def agent_registry(self) -> Any | None:
        """The agent profiles on this machine, from the viewer's own config dir.

        The sibling of :attr:`team_registry`, and broken by the same
        mechanism: ``/agent``'s listing and launch both read this off the
        session (``_agent_profile_rows``, ``_cmd_agent``). Same lazy
        construction, same degrade-one-feature guard, same failure latch.

        NOT symmetric with :attr:`team_registry` in one respect, which is
        recorded here rather than silently inherited (R4):
        ``AgentRegistry.__init__`` creates ``<config_dir>`` and
        ``<config_dir>/agents`` and runs its two migrations, so READING this
        property writes to disk. ``TeamRegistry`` deliberately does the
        opposite ("No mkdir here: every interactive session constructs a
        registry, and an unused feature must not litter the config dir").

        The asymmetry is pre-existing and left alone on purpose: every other
        host that offers `/agent` — the CLI, `exec`, the server, the mobile
        daemon — constructs the same registry the same way, so the directory a
        viewer creates is one every other entry point would have created
        anyway. Changing the constructor to match ``TeamRegistry`` is a change
        to shared behaviour with its own blast radius, not a fix belonging to
        this regression. What is new here is only that the construction now
        also happens on a viewer.
        """
        if self._agent_registry_cache is None:
            if self._within_registry_cooldown(self._agent_registry_failed_at):
                return None
            from local_operator.agents import AgentRegistry

            try:
                self._agent_registry_cache = AgentRegistry(self._config_dir)
            except Exception:  # noqa: BLE001 — one feature must not break the session
                self._agent_registry_failed_at = time.monotonic()
                return None
        return self._agent_registry_cache

    @classmethod
    async def connect(
        cls,
        record: SessionRecord,
        session_id: str,
        *,
        config_dir: Path,
        takeover_factory: Callable[[], Any],
        surface: str = "terminal",
        display_window: bool = False,
    ) -> "RemoteSession":
        if record.protocol < 5 or FRONTEND_CAPABILITY not in record.capabilities:
            raise ConnectionError(
                f"owner lacks {FRONTEND_CAPABILITY}; canonical full-TUI attach needs protocol >= 5"
            )
        self = cls(
            config_dir=config_dir,
            session_id=session_id,
            takeover_factory=takeover_factory,
            surface=surface,
        )
        self._display_window_requested = display_window
        await self._dial(record)
        try:
            frontend = await self._await_frontend()
            self._install_frontend(frontend.snapshot)
            await self._load_frontend_history(frontend)
        except BaseException:
            # The caller gets the error and no facade — so nothing would ever
            # close the connection ``_dial`` just opened. See
            # ``_discard_rejected_client`` for the leak this closes.
            self._discard_rejected_client()
            raise
        self._finish_sync()
        return self

    @classmethod
    async def cold(
        cls,
        session_id: str,
        *,
        config_dir: Path,
        cwd: str,
        takeover_factory: Callable[[], Any],
        surface: str = "terminal",
    ) -> "RemoteSession":
        """A viewer bound to NOTHING: durable history and a spool, no runtime.

        The state ``lop`` boots into. There is no process to attach to yet and
        deliberately none is started — opening a terminal is not work, and a
        session that is only being LOOKED at should cost nothing. The first
        mutating call (a prompt, a steer, an answered gate) runs
        :meth:`_ensure_bound`, which engages a runtime and attaches to it.

        Canonical state is synthesised rather than read from an owner, because
        there is no owner: the model comes from config, the roster is empty,
        and the wakes come from the on-disk index. It is a real
        ``FrontendSessionState`` so every widget renders a cold session through
        exactly the same path it renders an attached one — the alternative, a
        second "cold" rendering mode, is how two vocabularies for one screen
        get built.
        """
        self = cls(
            config_dir=config_dir,
            session_id=session_id,
            takeover_factory=takeover_factory,
            surface=surface,
        )
        self._cwd = cwd
        self._can_go_cold = True
        state = await self._synthesise_cold_state(cwd)
        # A session that has never run has no transcript to read; one being
        # reopened has its whole history here, off the loop as always.
        if (config_dir / "sessions" / session_id / "transcript.jsonl").exists():
            # ``want_checkpoint`` extracts the durable status row during the
            # SAME threaded parse the history comes from, so the restore below
            # costs no second read of a file that is 103 MB at the top end.
            await self._load_history(None, want_checkpoint=True)
            # Ordered before ``_install_frontend`` so the roster, todos and
            # title are present in the FIRST state the widgets ever see —
            # installing twice would paint an empty panel and then repaint it,
            # which is the visible flicker this whole change exists to remove.
        # Children can persist a roster before the parent's first transcript
        # row. Absence of that file must not hide independently durable spend.
        state = self._restore_cold_details(state)
        self._cold_checkpoint = None
        self._install_frontend(state)
        self._finish_sync()
        # Nothing is queued behind an owner that will never arrive: a cold
        # viewer is READY, and it is _ensure_bound that supplies the runtime
        # when one is actually needed.
        self._owner_ready.set()
        return self

    def _restore_cold_details(self, state: FrontendSessionState) -> FrontendSessionState:
        """Fold the durable turn-end checkpoint over synthesised cold state.

        A cold viewer synthesises canonical state because there is no owner to
        ask — but "no owner" is not "nothing is known". The session's last
        runtime wrote a full ``FrontendSessionState`` to the transcript at every
        turn end (``FrontendStateStore.checkpoint``), and that row already holds
        the subagent roster, the todo list, the conversation title, the goal and
        the accumulated spend.

        Before this, none of it was read: a resumed session opened with an empty
        subagent panel and no todos, and stayed that way until the user sent a
        message and a runtime started. The old in-process TUI restored exactly
        this state at boot (``Session.__init__`` calls ``_load_subagent_roster``
        and ``_load_todo_snapshot``), so the viewer model regressed it — the
        details were not slow to arrive, they were never going to arrive.

        The checkpoint is authoritative for what it carries and the synthesised
        state is authoritative for the rest, so the two are merged rather than
        one replacing the other: ``cwd`` and the model come from THIS process
        (the config may have changed since; the checkpoint's copy is history),
        while the roster, todos, title and costs come from disk. ``jobs`` are
        stamped ``restored`` for the same reason the session's own restore does
        — a restored row has no in-process trajectory, and the panel says so
        rather than rendering a busy child as empty.

        Best-effort by construction: an unreadable, absent or malformed
        checkpoint leaves the synthesised state untouched. Opening a
        conversation must never fail because its last status row did.
        """
        checkpoint = self._cold_checkpoint
        if checkpoint is None:
            return self._restore_cold_subagents(state)
        try:
            raw = checkpoint.get("state") if isinstance(checkpoint, dict) else None
            if not isinstance(raw, dict):
                raise ValueError(f"checkpoint 'state' is {type(raw).__name__}, not a mapping")
            durable = FrontendSessionState.model_validate(raw)
        except Exception as error:  # noqa: BLE001 — a bad row must not stop the open
            # LOUDLY. Falling back leaves exactly the pre-fix experience — an
            # empty roster and no todos — and at DEBUG that is indistinguishable
            # from the bug this change fixes, so the next report of it would be
            # re-diagnosed from scratch (UX round 1, U5). The open still
            # succeeds: a status row must never cost the user their
            # conversation.
            logger.warning(
                "session %s: durable checkpoint unreadable (%s); opening without the "
                "restored roster, todos and title",
                self._session_id,
                error,
            )
            self.degraded_reason = (
                "the saved session details could not be read, so the subagent and "
                "todo panels start empty"
            )
            return self._restore_cold_subagents(state)
        # A fork's transcript carries the PARENT's checkpoints verbatim (#573),
        # and the parent's children are not this session's to list — the same
        # reason ``fork.EXCLUDED_SIDECARS`` leaves the roster behind. The
        # runtime applies the identical rule when it restores
        # (``frontend_state._inherited_identity_fixups``), so the cold frame
        # and the attached frame agree on what the fork has.
        inherited = durable.session_id != self._session_id
        restored = state.model_copy(
            update={
                # Everything the last runtime knew and this process cannot
                # derive. The panel reads these directly, so restoring them is
                # what puts the session's details on the FIRST frame.
                "jobs": [] if inherited else list(durable.jobs),
                "todos": list(durable.todos),
                "conversation_title": durable.conversation_title,
                "conversation_title_user_set": durable.conversation_title_user_set,
                "conversation_title_forked": durable.conversation_title_forked,
                "goal": durable.goal,
                "active_agent": durable.active_agent,
                "active_team": durable.active_team,
                # Spend and occupancy are the conversation's history, not this
                # process's: a resumed session that already cost money must not
                # open reading zero (the same argument as
                # ``_restore_reported_usage`` on the old owner path).
                "cumulative_parent_cost": durable.cumulative_parent_cost,
                "child_costs": dict(durable.child_costs),
                "subagent_cost": durable.subagent_cost,
                "subagent_cost_knowledge": durable.subagent_cost_knowledge,
                "cost_knowledge": durable.cost_knowledge,
                "last_usage": durable.last_usage,
                **self._consistent_context(state, durable),
                # MCP servers are the last runtime's connection report and
                # there is no live manager to ask while cold, so the durable
                # copy is the only thing that can populate this chrome
                # (review round 1, C5). Restored as HISTORY: the panel shows
                # what the session was connected to, and the runtime's own
                # state replaces it wholesale on first engage.
                "mcp_servers": list(durable.mcp_servers),
                # Wakes are deliberately NOT taken from the checkpoint. The
                # synthesised state already read them from the wake index
                # (``_synthesise_cold_state``), which is the live derived file
                # a supervisor rewrites without opening the session — so the
                # index is fresher than any checkpoint and overwriting it with
                # a stale copy would re-show a wake that already fired. Only
                # fall back to the durable rows when the index gave nothing,
                # which is the corrupt/deleted-index case its own self-healing
                # rebuild is designed around.
                "wakes": list(state.wakes) if state.wakes else list(durable.wakes),
                **self._restored_model_specs(state, durable),
            }
        )

        return self._restore_cold_subagents(restored)

    def _restore_cold_subagents(self, state: FrontendSessionState) -> FrontendSessionState:
        """Overlay the independently committed roster and lifetime ledger.

        A child can settle without a parent turn, so the sidecar is newer than
        the frontend checkpoint and may be the ONLY durable state. Read it once
        for both rows and money; summing visible rows loses swept/prior work.
        """
        from local_operator.session.frontend_state import CostKnowledge
        from local_operator.session.session import (
            SUBAGENT_ROSTER_SIDECAR,
            _read_roster_sidecar,
        )
        from local_operator.tui.costs import cost_summary

        payload = (
            _read_roster_sidecar(
                self._config_dir / "sessions" / self._session_id / SUBAGENT_ROSTER_SIDECAR
            )
            or {}
        )
        changes: dict[str, Any] = {
            "jobs": _restored_job_rows(self._durable_roster(state, payload=payload))
        }
        if isinstance(payload.get("accounting"), list):
            try:
                # Validate the complete checkpoint before replacing money. A
                # corrupt component must not silently turn into a smaller bill.
                components = [Usage.model_validate(row) for row in payload["accounting"]]
                cost, unknown = cost_summary(components, recorded_only=True)
                changes.update(
                    subagent_cost=cost,
                    subagent_cost_knowledge=(
                        CostKnowledge.PARTIAL if unknown else CostKnowledge.EXACT
                    ),
                )
            except (TypeError, ValueError):
                logger.warning(
                    "session %s: subagent accounting checkpoint unreadable", self._session_id
                )
        return state.model_copy(update=changes)

    def _durable_roster(
        self, durable: FrontendSessionState, *, payload: dict[str, Any] | None = None
    ) -> Sequence[Any]:
        """The roster to restore: the SIDECAR's rows, falling back to the
        checkpoint's.

        The two stores are written on different triggers — the sidecar on every
        roster move (``_persist_subagent_roster``), the checkpoint at turn end
        (``FrontendStateStore.checkpoint``) — so they disagree whenever a child
        settles after the last turn boundary. On the reference session they
        differ in BOTH directions: 18 rows against 17, with two children only
        the sidecar knows and one only the checkpoint knows (UX round 1, U4).

        The sidecar wins because it is the roster's own store and the fresher
        of the two, which is the same reason ``_load_subagent_roster`` reads it
        first on the owner path. Its rows are merged OVER the checkpoint's
        rather than replacing them, so a child the sidecar has since dropped
        but the checkpoint still records is not silently lost — a resumed
        session should show every child it ever had, and neither store alone is
        a complete list.

        Best-effort: an unreadable or absent sidecar leaves the checkpoint's
        rows exactly as they were.
        """
        from local_operator.session.session import (
            SUBAGENT_ROSTER_SIDECAR,
            _read_roster_sidecar,
        )

        rows: dict[str, Any] = {str(job.id): job for job in durable.jobs}
        try:
            if payload is None:
                payload = _read_roster_sidecar(
                    self._config_dir / "sessions" / self._session_id / SUBAGENT_ROSTER_SIDECAR
                )
            for raw in (payload or {}).get("jobs") or []:
                job = JobState.model_validate(raw)
                if job.usage is not None:
                    from local_operator.session.frontend_state import _cost_knowledge
                    from local_operator.tui.costs import cost_summary

                    # The strict AsyncJob sidecar cannot grow frontend-only
                    # fields without breaking older owners. Reconstruct only
                    # from persisted bills/estimates here: a daemonless viewer
                    # must not need credentials or trigger model discovery.
                    cost, unknown = cost_summary(
                        job.usage.cost_components or [job.usage], recorded_only=True
                    )
                    previous = rows.get(str(job.id))
                    if cost is not None or previous is None:
                        job = job.model_copy(
                            update={
                                "direct_cost": cost,
                                "direct_cost_knowledge": _cost_knowledge(cost, unknown),
                            }
                        )
                    else:
                        job = job.model_copy(
                            update={
                                "direct_cost": previous.direct_cost,
                                "direct_cost_knowledge": (
                                    _cost_knowledge(previous.direct_cost, True)
                                    if unknown
                                    else previous.direct_cost_knowledge
                                ),
                            }
                        )
                rows[str(job.id)] = job
        except Exception:  # noqa: BLE001 — a bad sidecar must not stop the open
            logger.debug("cold state could not read the roster sidecar", exc_info=True)
        return list(rows.values())

    @staticmethod
    def _consistent_context(
        state: FrontendSessionState, durable: FrontendSessionState
    ) -> dict[str, Any]:
        """The restored context reading, but only where it still means something.

        A token count is only interpretable against the window it was measured
        against. When the user has switched models since the checkpoint was
        written, the stored numerator and the current denominator describe
        different things, and dividing one by the other produces a confident
        wrong percentage — the D1 failure in its other direction.

        There is no honest way to convert the reading, so it is DROPPED rather
        than converted: the band renders ``—`` for an unknown context, which is
        the same honest degradation it already shows for a model it cannot
        price. The first real turn replaces it with a live reading anyway.
        """
        configured = state.selected_model
        stored = durable.selected_model
        same_model = bool(
            configured is not None
            and stored is not None
            and configured.provider == stored.provider
            and configured.model_id == stored.model_id
        )
        if not same_model:
            return {}
        return {
            "context_tokens": durable.context_tokens,
            "context_is_estimate": durable.context_is_estimate,
            "context_window": durable.context_window,
        }

    @staticmethod
    def _restored_model_specs(
        state: FrontendSessionState, durable: FrontendSessionState
    ) -> dict[str, Any]:
        """Model specs for the restored state, keeping the window and the
        measured context consistent with each other.

        The synthesised cold spec is built from ``config.yml``, which names the
        provider and model but carries no metadata — so ``ModelSpec`` supplies
        its **128k default** for ``context_window``. The restored
        ``context_tokens`` were measured against the window the runtime
        actually had (1M on the reference session), and the band divides the
        restored tokens by the SPEC's window (``_context_window`` in
        ``tui/app.py`` reads the effective spec, deliberately, because the
        percentage predicts when the next request overflows). Dividing 322,546
        by a defaulted 128,000 is how a resumed session painted **268.2%**
        (design review round 1, D1) — a number that cannot be true, on the one
        surface that exists to tell the user how much room is left.

        The checkpoint's own spec is the one those tokens were measured
        against, so it is the honest denominator. Taken ONLY when the config
        names the same model: if the user switched models since, the
        configured spec is right and the stale window would be the wrong
        answer in the other direction. In that case the numerator is dropped
        instead (see ``_consistent_context``) rather than divided by a window
        it was never measured against.
        """
        configured = state.selected_model
        stored = durable.selected_model
        if configured is None or stored is None:
            return {}
        same_model = (
            configured.provider == stored.provider and configured.model_id == stored.model_id
        )
        if not same_model:
            return {}
        # Fresh route metadata outranks an old owner's active window (which
        # may predate maximum-context support or a changed opt-out setting).
        if (
            configured.context_metadata_resolved
            or configured.default_context_window
            or configured.max_context_window
        ):
            return {}
        # Only the window is adopted. Everything else on the configured spec
        # reflects THIS process's config, which is current by definition.
        window = int(getattr(stored, "context_window", 0) or 0)
        if window <= 0:
            return {}
        update = {
            "context_window": window,
            "default_context_window": stored.default_context_window,
            "max_context_window": stored.max_context_window,
        }
        return {
            "selected_model": configured.model_copy(update=update),
            "effective_model": (
                state.effective_model.model_copy(update=update)
                if state.effective_model is not None
                else None
            ),
        }

    async def _synthesise_cold_state(self, cwd: str) -> FrontendSessionState:
        """Canonical state for a session with no runtime to ask.

        Off the loop: it reads the config file and the wake index.
        """

        def _build() -> FrontendSessionState:
            from local_operator.session.frontend_state import (
                FrontendModelSpec,
                WakeState,
            )

            model: FrontendModelSpec | None = None
            try:
                from local_operator.config import ConfigManager

                config = ConfigManager(config_dir=self._config_dir)
                provider = str(config.get_config_value("hosting", "") or "")
                model_id = str(config.get_config_value("model_name", "") or "")
                model = FrontendModelSpec(provider=provider, model_id=model_id)
            except Exception:  # noqa: BLE001 — an unreadable config is not fatal
                logger.debug("cold state could not read the configured model", exc_info=True)
            if model is None:
                # NEVER None. ``RemoteSession.model`` raises without a spec, and
                # a cold viewer is exactly the state where config may be empty
                # (a first run, before `/login`) — so the band would crash on
                # the very screen that exists to help the user fix it. An empty
                # spec renders as "no model" and is replaced by the runtime's
                # own on first engage.
                model = FrontendModelSpec(provider="", model_id="")

            wakes: list[WakeState] = []
            try:
                from local_operator.wakes.store import read_entry

                entry = read_entry(self._config_dir, self._session_id)
                for schedule in (entry or {}).get("schedules", []) or []:
                    if isinstance(schedule, dict):
                        try:
                            wakes.append(WakeState.model_validate(schedule))
                        except Exception:  # noqa: BLE001 — skip an unreadable row
                            continue
            except Exception:  # noqa: BLE001 — no index is the common case
                logger.debug("cold state could not read the wake index", exc_info=True)

            return FrontendSessionState(
                session_id=self._session_id,
                epoch=f"cold-{self._session_id}",
                cwd=cwd,
                selected_model=model,
                effective_model=model,
                wakes=wakes,
            )

        state = await asyncio.to_thread(_build)
        if state.selected_model is not None and state.selected_model.provider == "openai":
            from local_operator.config import ConfigManager
            from local_operator.model.configure import context_spec_for_access
            from local_operator.providers.auth_store import AuthStore
            from local_operator.providers.failover import (
                AuthRetryKeyState,
                _resolve_access_for_provider,
            )

            # Resolve the same account as dispatch, not a saved denominator
            # from before maximum-context support. Never move account stickiness
            # merely because a viewer opened a cold session.
            configured_model = state.selected_model

            async def _resolve() -> ModelSpec:
                # AuthStore's SQLite connection is thread-affine. Creation,
                # credential resolution and close belong to this ONE worker's
                # event loop, not separately scheduled default-executor jobs.
                auth = AuthStore(self._config_dir / "auth.db")
                try:
                    access = await _resolve_access_for_provider(
                        auth,
                        "openai",
                        self._session_id,
                        AuthRetryKeyState(),
                        None,
                        read_only=True,
                        model_id=configured_model.model_id,
                        scoped_blocks=True,
                    )
                    settings = ConfigManager(config_dir=self._config_dir).get_config().values
                    return context_spec_for_access(configured_model, access, settings)
                finally:
                    auth.close()

            try:
                model = await asyncio.to_thread(lambda: asyncio.run(_resolve()))
                state = state.model_copy(update={"selected_model": model, "effective_model": model})
            except Exception:  # noqa: BLE001 — metadata must not prevent viewing saved work
                logger.debug("cold context metadata unavailable", exc_info=True)
        return state

    @property
    def is_cold(self) -> bool:
        """No fully synchronized runtime is attached to this viewer."""
        return self._client is None or not self._client.connected or not self._ready_for_events

    async def attach_existing(self) -> bool:
        """Attach if an owner exists, without turning a history read into work.

        Desktop read/subscription requests use the cold viewer's recovery policy
        even when an owner is already live: losing that owner must never move
        execution into the HTTP worker or start a replacement just for a reader.
        """
        from local_operator.mobile.attach_client import find_owner_record

        async with self._bind_lock:
            if self._disposed or not self.is_cold:
                return not self.is_cold
            record, _ = await asyncio.to_thread(
                find_owner_record, self._config_dir, self._session_id
            )
            if record is None or self._disposed:
                return False
            await self._bind_to(record)
            return True

    async def admit_prompt(
        self, text: str, *, command_id: str, images: list[dict[str, str]], steer: bool = False
    ) -> tuple[str, bool]:
        """Return the owner's admission receipt, not a fictitious completed turn.

        Retrying the caller's stable ID crosses the existing durable reservation
        boundary. Unlike submit_response this does not wait for model completion,
        so an HTTP disconnect cannot cancel work the owner already accepted.
        """
        await self._ensure_bound()
        client = self._client
        if client is None or not client.connected:
            raise ConnectionError(self._unavailable_reason())
        return await client.request_ack_with_duplicate(
            "steer" if steer else "prompt", text=text, images=images, command_id=command_id
        )

    async def bind_runtime(self) -> None:
        """Bind a viewer before an explicitly requested owner control operation."""
        await self._ensure_bound()

    async def update_desktop_watch(self, *, visible: bool, can_notify: bool) -> None:
        """Update the existing attach lease; a proxy socket alone is not a human."""
        if self._surface != "desktop":
            raise ValueError("only a desktop viewer can renew a desktop lease")
        self._desktop_visible = visible
        self._desktop_can_notify = can_notify
        self._desktop_seen = time.monotonic()
        if self._client is not None and self._client.connected:
            await self._client.desktop_watch(visible=visible, can_notify=can_notify)

    async def answer_gate(
        self,
        request_id: str,
        *,
        value: str | None = None,
        approved: bool | None = None,
        question_index: int | None = None,
    ) -> str:
        """Answer the current owner gate without a terminal-local prompt task.

        The owner validates again across the socket. This early identity check
        prevents a stale desktop popup from accidentally answering a newer gate
        while a reconnect or a multi-question ask advances in another window.
        """
        pending = self.frontend_state.pending_gate
        client = self._client
        if (
            pending is None
            or pending.request_id != request_id
            or client is None
            or not client.connected
        ):
            raise ValueError("this question is no longer pending")
        if pending.kind == "approval" and type(approved) is bool:
            return await client.approval_answer(request_id, approved)
        if pending.kind == "ask" and value is not None and question_index == pending.question_index:
            return await client.ask_answer(request_id, value, question_index=question_index)
        raise ValueError("the answer does not match the current question")

    async def _ensure_bound(self) -> None:
        """Attach to a runtime, starting one if none exists. Idempotent.

        The seam between "looking at a session" and "working in one", and the
        only place a viewer creates a process. Serialised by a lock because
        several mutating calls can arrive in the same tick (a prompt racing the
        speculative warm engage the first keystroke started) and each must
        wait for the SAME engagement rather than starting a second.

        Scoped to VIEWER facades (``_can_go_cold``). The legacy attach path
        keeps its own contract for a lost owner — recover the conversation into
        this process, or report the deliberate stop — and engaging a runtime
        there would both contradict that and start a process for a session the
        caller is about to take over itself.
        """
        # Recovery already owns its dial/sync and signals _owner_ready. A
        # prompt or steer must wait on that promise, not start a competing
        # initial attachment merely because its connected socket is not ready.
        if not self._can_go_cold or not self.is_cold or self._disposed or self._recovering:
            return
        async with self._bind_lock:
            if not self.is_cold or self._disposed or self._recovering:
                return
            from local_operator.mobile.attach_client import find_owner_record
            from local_operator.session.runtime.launch import (
                ActionableConnectionError,
                RuntimeStartupError,
                WarmErrand,
                engage_runtime,
            )

            try:
                await engage_runtime(
                    self._session_id,
                    self._cwd,
                    WarmErrand(),
                    config_dir=self._config_dir,
                )
            except RuntimeStartupError as error:
                # engage_runtime now fails FAST once no candidate can start,
                # carrying the child's own cause. Re-raised as ConnectionError
                # so it takes the existing owner-unavailable path, but keeping
                # the vetted user-facing sentence when there is one, instead of
                # a generic timeout nobody can act on (QA Q1).
                logger.warning("engage failed for %s: %s", self._session_id, error)
                # The vetted sentence rides a TYPE, so a relay can echo it
                # without having to guess from the text which messages are safe
                # to show. Anything unvetted stays a plain ConnectionError and
                # gets the generic sentence at the surface.
                if error.actionable:
                    raise ActionableConnectionError(error.actionable) from error
                raise ConnectionError(self._unavailable_reason()) from error
            # Re-checked AFTER the engage, which is the long await here (a
            # spawn plus up to ~2 s of construction). The TUI engages at mount
            # now, so `/resume` or `/new` typed in that first second disposes
            # this facade while the engage is in flight; binding anyway would
            # attach a live `attach` socket to a dead viewer — one nobody
            # closes, which holds the old runtime resident (residency term 3)
            # and never offers it back (review round 1, MAJOR-1). The runtime
            # that was spawned is left to the drain: with no viewer attached
            # and nothing written it exits in ~3 s and removes its directory.
            if self._disposed:
                return
            record, _owner = await asyncio.to_thread(
                find_owner_record, self._config_dir, self._session_id
            )
            if record is None:
                raise ConnectionError("could not start a runtime for this session")
            if self._disposed:
                return
            await self._bind_to(record)

    async def _bind_to(self, record: SessionRecord) -> None:
        """Attach this viewer to a live record and adopt its canonical state.

        The tail of :meth:`connect`, reused so a cold viewer becoming attached
        takes the identical path a fresh attach does — including the history
        boundary, which is what stops the rows already on screen from painting
        a second time.
        """
        try:
            await self._dial(record)
            frontend = await self._await_frontend()
            if self._disposed:
                raise ConnectionError("viewer disposed while synchronizing")
            self._install_frontend(frontend.snapshot, publish=True)
            await self._load_frontend_history(frontend)
            if self._disposed:
                raise ConnectionError("viewer disposed while synchronizing")
            self._finish_sync()
            self._deliberate_stop = False
            self._stopped_announced = False
            self._owner_ready.set()
        except BaseException:
            # A failed/cancelled sync is not an attached viewer. Retrying must
            # not leak the half-open socket or inherit its queued epoch suffix.
            self._discard_rejected_client()
            if not self._recovering:
                self._owner_ready.set()
            raise

    def _discard_rejected_client(self) -> None:
        """Drop a client whose runtime dialled fine but whose state was refused.

        ``_dial`` installs ``self._client`` the moment the socket authenticates,
        BEFORE the canonical sync is awaited and checked — so when the sync is
        rejected (``_install_frontend``'s identity guard, a malformed frame, the
        15 s sync timeout) the facade was left half-bound: ``is_cold`` said
        False because a connected client existed, every RPC still reached the
        owner, yet no state had been installed and none ever would be. That is
        the exact shape #573 produced on a switched-to fork: ``/model`` landed
        on the owner's journal while the band never repainted and the context
        segment stayed blank, so the switch read as "nothing happened".

        Closing the client makes ``is_cold`` honest again — the next
        ``_ensure_bound`` retries the whole engage and the TUI's own failure
        path can say why — and it releases the runtime's attach slot. Without
        the release each retry of ``_recover_owner`` opened another connection
        on top of the last, and the runtime's LRU cap evicted them in a burst
        (272 evictions logged in the minutes after one fork booted).

        ``close()`` also cancels the pump, which is the one thing that must not
        run on: with no state installed, the owner's next ``frontend_update``
        would land on a store at the wrong epoch and fail as a sequence gap.
        Cleared directly rather than through ``_on_disconnected`` because the
        socket did not fail — the state did — and the recovery loop that hook
        starts would redial the same runtime and be refused the same way.
        """
        client, self._client = self._client, None
        self._frontend_future = None
        self._runtime_pid = None
        # The rejected dial's buffered suffix belongs to its refused epoch.
        # connect, initial binding and recovery all share this discard boundary.
        self._pending_frontend_updates = None
        if client is None:
            return
        # ``abandon`` rather than ``close``: this closure is ours, not the
        # owner's, so it must not read as owner loss — a bind that was refused
        # would otherwise start a recovery that redials the same runtime and
        # is refused again.
        try:
            client.abandon()
        except Exception:  # noqa: BLE001 - teardown of a connection being abandoned
            logger.debug("closing a rejected owner connection failed", exc_info=True)

    async def _dial(self, record: SessionRecord) -> None:
        self._owner_record = record
        # Freeze relay delivery until the canonical sync is installed ahead of
        # raw event frames that follow it on the same socket.
        self._ready_for_events = False
        self._owner_ready.clear()
        self._pending_frontend_updates = []
        self._runtime_pid = record.pid
        loop = asyncio.get_running_loop()
        self._frontend_future = loop.create_future()
        pending_sync = self._frontend_future

        def on_disconnected(reason: str) -> None:
            # A connection that dies while we are still waiting for the sync
            # must fail the wait NOW rather than let it run out the 15 s
            # timeout. The oversized-frame case is exactly this: the client
            # knows within milliseconds that the frame is unreadable, but the
            # user still sat through a silent quarter-minute and then got a
            # degraded session with no explanation (UX round 1, U2; design
            # round 1, D5 is the same finding from the other side). The
            # reason string is carried into the error so the copy the pump
            # produced actually reaches a surface instead of only a log line.
            if not pending_sync.done():
                pending_sync.set_exception(ConnectionError(reason))
            # Closing a failed binding schedules the old pump's final callback.
            # It must not mark a retried/newer binding as recovering.
            if self._client is client:
                self._on_disconnected(reason)

        # The runtime's build, captured from the record BEFORE the socket is
        # opened: a runtime's version cannot change while it lives, so one
        # read at dial is complete. ``""`` means the owner predates the field,
        # which by construction makes it older than this terminal. The TUI
        # compares these with its own build and names the skew (see
        # ``app.py::_check_build_skew``); nothing here decides anything, so a
        # missing stamp degrades to "unknown", never to a refused attach.
        self.owner_version = getattr(record, "version", "") or ""
        self.owner_source_ref = getattr(record, "source_ref", "") or ""
        # The runtime's working directory, kept so a viewer that was ATTACHED
        # (``connect``, the `lop --resume` path) can engage a successor after
        # its owner retires for a refresh. Only ``cold()`` used to set ``_cwd``;
        # an attached viewer had none, so ``_ensure_bound`` would have spawned
        # the successor in the wrong directory. Never overwrites a cwd the
        # viewer was constructed with — that one is the user's choice.
        if not self._cwd:
            self._cwd = str(getattr(record, "cwd", "") or "")
        client = AttachClient(
            lambda _projection: None,
            on_disconnected,
            events=True,
            on_event=lambda data: (self._on_wire_event(data) if self._client is client else None),
            frontend_state=True,
            surface=self._surface,
            # THE full-TUI viewer is the client that renders action-carrying
            # receipts: ``_render_authoritative_slash`` submits their
            # ``request`` as a user turn. Declaring them is what tells the
            # runtime NOT to admit the request itself, which would run the
            # command twice. A viewer that omitted this (every build before
            # the field) is exactly the case the runtime completes for.
            slash_consumers=list(SLASH_ACTION_RECEIPTS),
            display_window=self._display_window_requested,
            on_frontend_sync=lambda data: (
                self._on_frontend_sync(data) if self._client is client else None
            ),
            on_frontend_update=lambda data: (
                self._on_frontend_update(data) if self._client is client else None
            ),
        )
        try:
            await client.connect(record, self._session_id)
        except BaseException:
            # A cancel (the app cancelling its engage worker at a swap) or a
            # failure inside `connect` leaves a half-open socket that nothing
            # else references; closing it here rather than leaving it to GC
            # is the same discipline `_deliver` keeps (review round 2,
            # MINOR-1).
            client.close()
            raise
        if self._disposed:
            # The facade was disposed while the socket was connecting. Holding
            # the client would leave an `attach` connection nobody owns on the
            # runtime, which pins it resident. Close it here, where the socket
            # was opened; `dispose` has already run and will not run again.
            client.close()
            raise ConnectionError("viewer disposed while attaching")
        self._client = client
        if self._surface == "desktop":
            from local_operator.session.runtime.types import DESKTOP_WATCH_LEASE_S

            # Reconnecting the proxy must not resurrect a renderer's expired
            # visibility/notification lease. Only another host heartbeat may.
            live = time.monotonic() - self._desktop_seen < DESKTOP_WATCH_LEASE_S
            try:
                await client.desktop_watch(
                    visible=live and self._desktop_visible,
                    can_notify=live and self._desktop_can_notify,
                )
            except BaseException:
                client.close()
                self._client = None
                raise

    async def _await_frontend(self) -> FrontendSync:
        future = self._frontend_future
        if future is None:
            raise ConnectionError("owner did not start frontend synchronization")
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=15.0)
        except TimeoutError as exc:
            raise ConnectionError("owner did not send frontend synchronization") from exc

    async def _load_frontend_history(self, frontend: FrontendSync) -> None:
        """Install the durable cut before live replay or command readiness."""
        window = frontend.display_history if self._display_window_requested else None
        previous = self._display_history
        if window is None or window.status != "ok":
            # Legacy owners and oversized prose keep the honest full replay.
            self._display_history = None
            self._history_hydrated = True
            self._durable_seed_ids.clear()
            self._durable_seed_tool_ids.clear()
            await self._load_history(frontend.live_cursor, strict_cut=window is not None)
            if previous is not None:
                self._buffered_events.insert(
                    0, HistoryDeltaEvent(messages=list(self._history), reset=True)
                )
            return
        self._validate_display_window(window, frontend.epoch, frontend.live_cursor)
        rows = list(window.messages)
        page = window
        reset = self._hydrated_once and (
            previous is None
            or previous.owner_epoch != window.owner_epoch
            or previous.history_generation != window.history_generation
        )
        if previous is not None and self._hydrated_once and not reset:
            # Reconnect must include ALL rows since the last durable frontier,
            # not just a recent tail. Appends preserve signed snapshot positions.
            while page.start > previous.total_message_count and page.before_token:
                page = await self._fetch_history_page(
                    page.before_token, frontend.epoch, frontend.live_cursor
                )
                if page.status != "ok":
                    raise ConnectionError("history changed during reconnect; retry attachment")
                rows[:0] = page.messages
        self._display_history = window.model_copy(
            update={
                "messages": rows,
                "start": page.start,
                "before_token": page.before_token,
                "has_more": page.has_more,
            }
        )
        self._history = rows
        self._live_history.clear()
        self._history_hydrated = page.start == 0 and len(rows) == window.total_message_count
        self._history_ids = {m.id for m in rows}
        self._durable_seed_ids = set(window.durable_seed_ids)
        self._durable_seed_tool_ids = set(window.durable_seed_tool_ids)
        if reset:
            self._message_events.clear()
            self._buffered_events.insert(0, HistoryDeltaEvent(messages=rows, reset=True))
        elif self._hydrated_once and previous is not None:
            self._replay_durable_suffix(rows[max(0, previous.total_message_count - page.start) :])
        # Loaded rows suppress duplicate relay, but are not all painted: the
        # TUI mounts only its viewport and pages older rows later.
        self._live_message_phase.clear()
        if not self._hydrated_once:
            self._filter_known_messages()
        self._hydrated_once = True

    def _validate_display_window(
        self, window: DisplayHistoryWindow, epoch: str, cursor: str | None
    ) -> None:
        if (
            window.conversation_id != self.session_id
            or window.owner_epoch != epoch
            or window.through_id != cursor
        ):
            raise ConnectionError("display history does not match canonical sync")
        if window.start < 0 or window.start + len(window.messages) > window.total_message_count:
            raise ConnectionError("invalid display history range")

    async def _fetch_history_page(
        self, before: str, epoch: str, cursor: str | None, anchor: str = ""
    ) -> DisplayHistoryWindow:
        client = self._client
        if client is None:
            raise ConnectionError("history owner is disconnected")
        page = DisplayHistoryWindow.model_validate(await client.history_page(before, anchor))
        if page.status != "reset":
            self._validate_display_window(page, epoch, cursor)
        return page

    async def history_page(self, before: str, *, anchor: str = "") -> DisplayHistoryWindow:
        """Read a signed page; reset is explicit and never silently mixed in."""
        window = self._display_history
        if window is None:
            raise RuntimeError("this session uses full-history replay")
        return await self._fetch_history_page(before, window.owner_epoch, window.through_id, anchor)

    def pending_display_tool_ids(self) -> set[str]:
        """Unanswered calls in the pending gate's current serialized user turn.

        A gate can precede tool_execution_start, so no invented start event is
        needed. The latest call group after the latest user boundary is the
        only eligible group; old interrupted turns must remain interrupted.
        """
        if self.pending_gate is None:
            return set()
        answered: set[str] = set()
        for message in reversed(self.display_history_window()):
            role = getattr(message, "role", "")
            if role == "user":
                break
            if role == "tool":
                answered.add(str(getattr(message, "tool_call_id", "")))
            calls = getattr(message, "tool_calls", None)
            if role == "assistant" and calls:
                return {call.id for call in calls} - answered
        return set()

    @property
    def display_history_revision(self) -> int:
        return self._display_revision

    @property
    def display_history_current(self) -> bool:
        return not self._display_invalidated

    def _invalidate_display_history(self) -> None:
        if not self._display_window_supported:
            return
        self._display_invalidated = True
        self._display_revision += 1
        task = self._display_refresh_task
        if task is None or task.done():
            task = asyncio.create_task(self._refresh_display_history())
            self._display_refresh_task = task

            def finished(done: asyncio.Task[None]) -> None:
                if not done.cancelled() and done.exception() is not None:
                    # Keep the invalidation fence closed. Selection awaits this
                    # task and reports the failure instead of painting stale rows.
                    logger.warning("canonical display refresh failed: %s", done.exception())

            task.add_done_callback(finished)

    async def ensure_display_current(self) -> None:
        task = self._display_refresh_task
        if task is not None:
            await asyncio.shield(task)
        if self._display_invalidated:
            await self._refresh_display_history()

    async def _refresh_display_history(self) -> None:
        """Reattach only this viewer; never restart, stop, or prompt the owner."""
        async with self._display_refresh_lock:
            record = self._owner_record
            if record is None or self._client is None or not self._client.connected:
                record, _ = await asyncio.to_thread(
                    find_owner_record, self._config_dir, self._session_id
                )
            if record is None:
                raise ConnectionError("history owner is unavailable")
            await self._bind_to(record)

    @property
    def history_before_token(self) -> str | None:
        window = self._display_history
        return window.before_token if window is not None and not self._history_hydrated else None

    async def load_older_display_page(self) -> list[Any]:
        window = self._display_history
        if window is None or not self.history_before_token:
            return []
        page = await self.history_page(self.history_before_token)
        if self._display_history is not window:
            raise RuntimeError("history changed while paging; retry")
        if page.status == "reset":
            await self._refresh_display_history()
            return []
        if page.status == "full_required":
            old_ids = set(self._history_ids)
            return [m for m in await self.materialize_history() if m.id not in old_ids]
        if page.start + len(page.messages) != window.start:
            raise ConnectionError("history page is not contiguous with the loaded window")
        self._history[:0] = page.messages
        self._history_ids.update(m.id for m in page.messages)
        self._display_history = window.model_copy(
            update={
                "start": page.start,
                "before_token": page.before_token,
                "has_more": page.has_more,
                "messages": list(self._history),
            }
        )
        self._history_hydrated = page.start == 0
        return list(page.messages)

    async def ensure_display_anchor(self, anchor: str) -> bool:
        window = self._display_history
        if window is None or not window.snapshot_token or not anchor:
            return True
        page = await self.history_page(window.snapshot_token, anchor=anchor)
        if page.status == "reset":
            await self._refresh_display_history()
            fresh = self._display_history
            if fresh is None or not fresh.snapshot_token:
                return False
            page = await self.history_page(fresh.snapshot_token, anchor=anchor)
            if page.status == "reset":
                return False
        if page.status == "full_required":
            await self.materialize_history()
            return True
        # Keep a contiguous loaded interval: the existing renderer pages both
        # ways inside it and must never mistake a disjoint anchor/tail for a
        # complete trajectory. The signed seek determines the exact frontier.
        while self._display_history is not None and self._display_history.start > page.start:
            await self.load_older_display_page()
        return True

    async def materialize_history(self) -> list[Any]:
        """Explicit full replay, off the click path, at the captured sync cut."""
        if self._history_hydrated:
            return self.history()
        window = self._display_history
        if window is None:
            raise RuntimeError("no canonical display history is installed")
        rows: list[Any] = []
        token = window.snapshot_token
        while token:
            page = await self.history_page(token)
            if page.status == "reset":
                raise RuntimeError("history changed while materializing; reconnect")
            if page.status == "full_required":

                def replay() -> list[Any]:
                    transcript = Transcript(self._config_dir / "sessions" / self._session_id)
                    return (
                        transcript.build_llm_history(through_id=window.through_id)
                        if window.through_id
                        else []
                    )

                rows = await asyncio.to_thread(replay)
                break
            rows[:0] = page.messages
            token = page.before_token
        if self._display_history is not window:
            raise RuntimeError("history changed while materializing; retry")
        self._history = rows
        self._history_ids = {m.id for m in rows}
        self._history_hydrated = True
        return self.display_history_window()

    async def _load_history(
        self,
        live_cursor: str | None = None,
        *,
        drop_history_duplicates: bool = True,
        want_checkpoint: bool = False,
        strict_cut: bool = False,
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
        entries, history = await self._read_transcript(
            want_checkpoint=want_checkpoint, through_id=live_cursor, strict_cut=strict_cut
        )
        self._bind_history(
            entries, history, live_cursor, drop_history_duplicates=drop_history_duplicates
        )

    async def _read_transcript(
        self,
        *,
        want_checkpoint: bool = False,
        through_id: str | None = None,
        strict_cut: bool = False,
    ) -> tuple[list[Any], list[Any]]:
        """Parse the durable transcript off-loop, once per sync.

        The single threaded read shared by initial connect AND reconnect:
        review round 3 (MAJOR-2) found the reconnect path re-running this
        exact parse synchronously on the event loop — a 60 MB transcript
        blocked it for ~90 ms, past the 50 ms no-stall bar #300 established
        for the connect path. Both callers now consume ONE threaded result
        (gap projection and ``_history`` reconciliation), so the file is
        parsed once per sync and never on the loop.

        ``want_checkpoint`` is the COLD path's opt-in to also extracting the
        durable frontend checkpoint from this same parse
        (``_restore_cold_details``), and it is opt-in rather than
        unconditional for a memory reason: the parsed ``Transcript`` pins
        every entry, measured at 1.23x file size (~127 MB on the reference
        session). Stashing it on the instance for every caller leaked that for
        the life of a WARM attach, which neither needs it nor ever cleared it
        (review round 1, C2). The checkpoint is a small dict, so extracting it
        INSIDE the worker lets the transcript die with the thread — nothing
        long-lived holds a reference on any path.
        """

        def _replay() -> tuple[list[Any], list[Any]]:
            transcript = Transcript(self._config_dir / "sessions" / self._session_id)
            if want_checkpoint:
                # Read here, on the worker, while the object is alive and
                # already parsed: a second ``Transcript(...)`` on the loop
                # would re-read and re-parse the whole file (0.68 s on the
                # largest observed session).
                self._cold_checkpoint = transcript.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
            cut = through_id
            if cut is not None and not transcript.has_entry(cut) and not strict_cut:
                # Older owners used best-effort cursors; preserve that fallback
                # only for legacy full replay, never the negotiated window cut.
                cut = None
            return transcript.entries(), transcript.build_llm_history(through_id=cut)

        return await asyncio.to_thread(_replay)

    def _bind_history(
        self,
        entries: list[Any],
        history: list[Any],
        live_cursor: str | None,
        *,
        drop_history_duplicates: bool,
    ) -> None:
        """Adopt canonical replay already selected at the journal cut.

        Filtering materialized IDs here used to lose synthetic compaction
        markers and could apply prunes newer than the cut. The parser now
        selects journal entries before running the shared replay semantics.
        """
        self._history = history
        self._live_history.clear()
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
                # Already painted is not the same as durable. Rebuild source
                # display storage from the canonical seed even when no event
                # should be re-emitted into an already-painted live surface.
                self._remember_live(event)
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
        self._maybe_start_gate()

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
            if (
                message_id in self._message_events
                or message_id in self._history_ids
                or message_id in self._durable_seed_ids
            ):
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
        """Record painted identity separately from complete source display data."""
        self._remember_live(event)
        message = getattr(event, "message", None)
        message_id = str(getattr(message, "id", "") or "")
        if message_id and (
            isinstance(event, MessageEndEvent)
            or (
                isinstance(event, MessageStartEvent)
                and bool(getattr(message, "text", "") or getattr(message, "tool_calls", None))
            )
        ):
            self._message_events.add(message_id)

    def _remember_live(self, event: AgentEvent[Any]) -> None:
        if isinstance(event, ToolExecutionEndEvent):
            if event.tool_call_id in self._durable_seed_tool_ids:
                return
            # Tool results arrive as execution events rather than message_end.
            # Keep a LIVE pairing dependency until its real durable message ID
            # arrives at the next sync; never claim this synthetic ID as durable.
            result = Message.tool_result(event.result)
            result.id = f"live-tool:{event.tool_call_id}"
            self._live_history[result.id] = result
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
            if message_id in self._history_ids or message_id in self._durable_seed_ids:
                return
            # Paint dedupe is not presentation storage. A source can leave the
            # screen while this row is in the live seed but not in its durable
            # attach window. Retain the complete row for the next prepared view.
            if isinstance(message, Message) and message.role == "tool":
                self._live_history.pop(f"live-tool:{message.tool_call_id}", None)
            self._live_history[message_id] = message

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
        if self._pending_frontend_updates is not None:
            self._pending_frontend_updates.append(update)
            return
        if self._frontend_store is None:
            raise ConnectionError("frontend update arrived before synchronization")
        state = self._frontend_store.apply_update(update)
        self._apply_frontend_facades(state)

    def _install_frontend(self, state: FrontendSessionState, *, publish: bool = False) -> None:
        if state.session_id != self._session_id:
            raise ConnectionError("frontend state belongs to another session")
        if self._frontend_store is None or self._frontend_store.state.epoch != state.epoch:
            if self._gate_task is not None:
                self._gate_task.cancel()
                self._gate_task = None
            self._gate_key = None
            self._gate_answered_key = None
            self._keep_gate_reply = False
        if self._frontend_store is None:
            self._frontend_store = FrontendStateStore(state)
        elif publish:
            self._frontend_store.replace_and_notify(state)
        else:
            self._frontend_store.replace(state)
        self._apply_frontend_facades(state)
        pending, self._pending_frontend_updates = self._pending_frontend_updates, None
        for update in pending or ():
            self._on_frontend_update(update.model_dump(mode="json"))

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
        if isinstance(startup, Mapping):
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
        self._deliver(event)

    def _deliver(self, event: AgentEvent[Any]) -> None:
        """Hand ``event`` to every subscribed handler now, buffering nothing.

        With NO subscriber the event is DROPPED, not parked — that is the
        contract, not an omission. Callers reach for this instead of
        ``_emit_or_buffer`` precisely when a deferred delivery would land on a
        LATER runtime's turn (see ``_end_turn_locally``), and parking the event
        for a future subscriber recreates exactly that hazard one seam over.
        """
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
        self._keep_gate_reply = False
        if pending is not None:
            self._maybe_start_gate(pending)

    def _maybe_start_gate(self, pending: PendingRequest | None = None) -> None:
        if self._disposed or not self._ready_for_events:
            return
        if pending is None:
            pending = _pending_request(self.pending_gate)
        if pending is None or self._gate_task is not None:
            return
        if self._gate_identity(pending) == self._gate_answered_key:
            return
        background = (
            self._gates_detached and self._background_approval and pending.kind == "approval"
        )
        if self._gates_detached and not background:
            return
        if pending.kind == "approval" and (self._approval_handler is not None or background):
            self._gate_task = asyncio.create_task(self._run_approval(pending))
        elif pending.kind == "ask" and self._ask_handler is not None:
            self._gate_task = asyncio.create_task(self._run_ask(pending))

    def _gate_reply_is_current(self, pending: PendingRequest, client: Any) -> bool:
        # Cancelling a bridge requests cooperation; even a handler that swallows
        # cancellation must not answer after another view/gate replaced it.
        return (
            (
                not self._gates_detached
                or self._keep_gate_reply
                or (self._background_approval and pending.kind == "approval")
            )
            and not self._disposed
            and self._gate_task is asyncio.current_task()
            and self._client is client
            and self._gate_key == self._gate_identity(pending)
        )

    async def _run_approval(self, pending: PendingRequest) -> None:
        try:
            handler = self._approval_handler
            client = self._client
            if client is None:
                return
            if self._gates_detached and self._background_approval:
                approved = True
            elif handler is not None:
                approved = await call_approval_gate(handler, pending.title, pending.detail)
            else:
                return
            approved = await call_approval_gate(handler, pending.title, pending.detail)
            if not self._gate_reply_is_current(pending, client):
                return
            await client.approval_answer(pending.request_id, approved)
            self._gate_answered_key = self._gate_identity(pending)
        except (asyncio.CancelledError, RuntimeError, ConnectionError):
            # Cancellation means another front end settled it. RuntimeError is
            # the owner's stale-request answer to the losing race. Both are an
            # ordinary first-valid-answer-wins outcome; the projection removes
            # the card.
            #
            # ConnectionError is the STOP path: settling a parked gate wakes
            # this task, which then tries to post its answer to an owner that
            # is gone. That is the expected end of a normal /stop, so letting
            # it escape only reached asyncio's default handler as a
            # "Task exception was never retrieved" traceback in the log
            # (round-6 NIT-3).
            pass
        finally:
            if (
                self._gate_key == self._gate_identity(pending)
                and self._gate_task is asyncio.current_task()
            ):
                self._gate_task = None

    async def _run_ask(self, pending: PendingRequest) -> None:
        try:
            handler = self._ask_handler
            client = self._client
            if handler is None or client is None:
                return
            options = [
                AskOption(
                    label=(
                        option.get("label", "") if isinstance(option, Mapping) else option.label
                    ),
                    description=(
                        option.get("description", "")
                        if isinstance(option, Mapping)
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
                self._gate_answered_key = self._gate_identity(pending)
        except (asyncio.CancelledError, RuntimeError, ConnectionError):
            # Same three outcomes as the approval gate above, including the
            # stop path's dead-owner post (round-6 NIT-3).
            pass
        finally:
            if (
                self._gate_key == self._gate_identity(pending)
                and self._gate_task is asyncio.current_task()
            ):
                self._gate_task = None

    # -- owner loss ---------------------------------------------------------

    def _on_disconnected(self, _reason: str) -> None:
        self._runtime_pid = None
        if self._disposed or self._recovering:
            return
        # A disconnect that follows OUR stop request (or arrives after the
        # owner already unpublishes) is the deliberate-stop landing: the
        # session ended on purpose, so there is no owner to recover and no
        # transcript lease to win. Stay a viewer showing the cold session —
        # the same shape bare /stop leaves an owner in. The record scan in
        # `_recover_owner` would otherwise rediscover nothing and take over.
        # The owner announced the stop on the wire before closing (the
        # ``stopping`` frame the client turns into this reason). That covers
        # the cases the local flag cannot: another TUI's /stop all, or a shell
        # `lop stop`, hitting a session THIS viewer merely watches — including
        # a session with no wakes, which leaves no on-disk marker to consult.
        if _reason == RETIRING_REASON:
            # A planned refresh, not owner death and not a stop: the runtime
            # left so the next engage runs the build now on disk. Nothing to
            # recover — the successor does not exist yet, and chasing the
            # record for 8 s would end in "runtime exited" for a housekeeping
            # event — and nothing was interrupted (the runtime retires only
            # when idle, so there is no turn to end). Go cold NOW and tell
            # the app so it can re-engage eagerly.
            self._go_cold(refresh=True)
            return
        if _reason == STOPPED_REASON:
            self._deliberate_stop = True
        if self._deliberate_stop:
            self._owner_ready.set()  # prompts route to the stopped notice
            # A stop ENDS the turn, exactly as a death does. Without this the
            # facade reports is_streaming forever — nothing else can clear it,
            # because every other writer of that flag is fed by the owner
            # whose socket just closed — so the spinner never stops and the
            # next message routes into the steer branch, is dropped on the
            # floor, and is receipted as "sends when this step finishes" for a
            # step that ended (round-4 MAJOR-3/D4-1). The honest refusal lives
            # on the prompt path, and this is what lets a message reach it.
            self._end_turn_locally()
            self._notify_stopped()
            return
        self._recovering = True
        self._owner_ready.clear()
        self._end_turn_locally()
        self._recovery_task = asyncio.create_task(self._recover_owner())

    def _end_turn_locally(self, *, direct: bool = False) -> None:
        """End an in-flight turn the owner can no longer end itself.

        All three terminal outcomes need it and none can get it from the
        owner: a killed owner factually aborted the turn, a stopped one ended
        the whole session under it, and a viewer going cold has no runtime
        left to hear from. Marked through the normal event path so no
        card/banner or attach vocabulary appears — the transcript reads as an
        ordinary aborted turn, which is what it is.

        ``direct`` bypasses the sync buffer and hands the end straight to the
        subscribed handlers (dropping it when there are none). The go-cold
        caller needs this: a failed successor ``_dial`` leaves
        ``_ready_for_events`` False, so a BUFFERED end would sit until the
        NEXT bind's ``_finish_sync`` drained it — behind that bind's seeded
        ``AgentStartEvent`` — and would tear down the new turn it never
        belonged to. Delivered now it ends the one it does.

        UNSTAMPED (``generation=0``), never ``self._generation`` — review
        round 1, BLOCKER-1. That field is not this viewer's own turn counter:
        ``_apply_frontend_facades`` overwrites it from whatever snapshot last
        arrived, and a SUCCESSOR is a fresh runtime whose counter restarts
        (``Session.__init__`` sets 0, so its first turn is 1) while the app's
        ``EventController`` has adopted the PREVIOUS owner's generation — 6,
        or any long session's turn count. Stamping the synthesised end with
        the successor's number therefore made ``_handle_agent_end`` drop it as
        ``gen < current`` (``tui/events.py``), so the end reached the handler
        and died one layer below it: no ``TurnEnded``, and — with edit 1's
        hold in place and no wall-clock timeout by design — a band and tab
        title asserting ``working`` for the rest of the process's life. ``0``
        is the field's default and the established "unstamped: belongs to
        whatever turn is open" encoding that the controller's ``if gen:``
        branch reads, which is exactly the semantics a locally synthesised end
        wants. It applies to the buffered path for the same reason: neither
        delivery has standing to speak for the successor's numbering.
        """
        if not self._streaming:
            return
        end = AgentEndEvent(aborted=True, generation=0, error=None)
        # THE STATE CHANGE IS THE CONTRACT; ONLY THE NOTIFICATION IS
        # BEST-EFFORT (review round 2, MAJOR-2). `_deliver` calls handlers
        # synchronously with no guard of its own, and
        # `EventController._handle_agent_end` does real work on this path —
        # flush, usage pricing, cost summation. When one of those raises, a
        # trailing assignment never runs and the facade reports `is_streaming`
        # True on a session that is now cold with no runtime left to clear it.
        # The caller's `try/except` cannot cover that: it catches the exception
        # OUTSIDE this method, by which point the clear has already been
        # skipped. That strands `_retire_turn_band`'s hold — the band and tab
        # title assert `working` for the rest of the process's life, with no
        # wall-clock timeout by design — and mis-routes the next message into
        # the steer branch, which is the round-4 MAJOR-3/D4-1 failure the
        # deliberate-stop comment above records.
        try:
            if direct:
                self._deliver(end)
            else:
                self._emit_or_buffer(end)
        finally:
            self._streaming = False

    async def _session_was_stopped(self) -> bool:
        """True when the disconnect's cause is a DELIBERATE stop, not owner death.

        Two shapes, one meaning — the session ended on purpose, so there is
        nothing to recover:

        1. This follower issued the stop itself (``_deliberate_stop``, set in
           ``request_stop`` before the op is sent).
        2. Someone ELSE stopped the session (another TUI's ``/stop all``, a
           shell ``lop stop``) while this follower watched: the stop stamps
           ``stopped_at`` on the wake-index entry (a durable, transcript-
           derived marker — survives the owner's exit, readable before any
           reconnect), and the owner never rediscovers. Both conditions
           together are the deliberate-stop wire shape: a dead owner leaves
           the marker absent, a stopped one leaves it set.
        """
        if self._deliberate_stop:
            return True
        from local_operator.wakes import store as wake_store

        entry = await asyncio.to_thread(wake_store.read_entry, self._config_dir, self._session_id)
        if entry is None or not entry.get("stopped_at"):
            return False
        # The marker says stopped; confirm nobody re-opened it in the
        # meantime (an open clears ``stopped_at``). If an owner is live and
        # reachable, this is a re-open — recover normally.
        record, _ = await asyncio.to_thread(find_owner_record, self._config_dir, self._session_id)
        return record is None

    def _unavailable_reason(self) -> str:
        """Why this facade cannot reach its owner right now, in the user's terms.

        A DELIBERATE stop and a dropped connection are opposite facts and
        must not share one sentence: "reconnecting" tells the user to wait
        for something that is never coming back, on the one path where the
        honest answer ("it was stopped; /resume reopens it") is already
        written for the owner's own screen.
        """
        if self._deliberate_stop:
            return "this session was stopped"
        return "session owner is reconnecting"

    def _go_cold(self, *, refresh: bool = False) -> None:
        """Unbind from a runtime that is gone, keeping the conversation.

        The viewer stays exactly as it is on screen; only its binding drops.
        ``_owner_ready`` is SET rather than left clear because a cold viewer is
        ready — the next prompt engages a runtime through ``_ensure_bound``
        instead of waiting for one that is never coming back.

        ``refresh`` is the planned-retirement variant (``RETIRING_REASON``):
        the runtime was idle by contract when it left, so there is no turn to
        end — ``_end_turn_locally`` is skipped, which is what keeps a refresh
        from ever painting ``interrupted`` — and the REFRESH callback fires
        instead of the went-cold one, so the app re-engages rather than
        reporting "runtime exited". If the runtime lied and a turn WAS live,
        ``_end_turn_locally`` would have produced a false abort anyway; the
        successor's snapshot re-seeds whatever is actually running.

        A turn still in flight is ENDED through the event path, not merely
        flagged off (#642, UX U11). Clearing ``_streaming`` alone told the
        facade the turn was over while the app — which holds its working line,
        band and title open until an ``AgentEndEvent`` reaches it — was never
        told anything, so a viewer that went cold mid-turn held a spinner
        forever with no toast and no notice. On the common path this is a
        no-op: ``_on_disconnected`` already ended the turn before
        ``_recover_owner`` started. The case it exists for is a SUCCESSOR
        dying mid-reattach — ``_on_disconnected`` returns early while
        ``_recovering``, and ``_apply_frontend_facades`` has just re-marked
        the turn live from the successor's snapshot — which leaves
        ``_streaming`` True with nothing else on the way to clear it.
        Delivered ``direct`` for the reason on ``_end_turn_locally``: the
        failed dial left the sync buffer closed, and a buffered end would land
        on the NEXT runtime's turn instead of this one.
        """
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 — teardown of a dead socket
                logger.debug("closing the lost owner connection failed", exc_info=True)
        # Guarded like the two teardown steps that bracket it (the client
        # close above, the went-cold callback below): a handler that raises —
        # `EventController._handle_agent_end` flushes, prices usage and sums
        # cost — must not propagate out of teardown and skip `_owner_ready`,
        # which would leave the prompt path waiting on an event nothing else
        # will set (review round 1, MINOR-1).
        if refresh:
            if self._streaming:
                # Should be unreachable (the runtime retires only when idle);
                # logged rather than asserted because the honest repair is
                # the successor's snapshot, not a false abort here.
                logger.warning("runtime retired for a refresh while a turn looked live")
            # A viewer that ATTACHED (``connect``) rather than started cold
            # keeps the legacy "recover by taking over" contract for owner
            # DEATH — but a refresh is not a death, and taking the lease into
            # this process would make the terminal the runtime for a session
            # that retired precisely so a fresh runtime could run it. From
            # here on this facade is a viewer: it engages a successor
            # (``_ensure_bound`` is gated on this flag) and, should THAT one
            # die, goes cold rather than taking over. The `lop --resume` TUI
            # is exactly this case (design-runtime-autorefresh §1.1).
            self._can_go_cold = True
        else:
            try:
                self._end_turn_locally(direct=True)
            except Exception:  # noqa: BLE001 — a viewer notice must not break teardown
                logger.debug("ending the in-flight turn on go-cold failed", exc_info=True)
        self._owner_ready.set()
        callback = self._refresh_callback if refresh else self._went_cold_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:  # noqa: BLE001 — a viewer notice must not break teardown
            logger.debug("%s callback failed", "refresh" if refresh else "went-cold", exc_info=True)

    def set_refresh_callback(self, callback: Callable[[], Any] | None) -> None:
        """Told when the runtime retired itself for a newer build.

        The viewer re-engages at once (``OperatorApp._on_runtime_refreshed``);
        the conversation is untouched and no notice is painted — the band's
        ``starting…`` state covers the ~1 s the re-engage takes.
        """
        self._refresh_callback = callback

    def owner_idle(self) -> bool:
        """Whether the bound runtime is doing nothing a refresh would lose.

        Read off the canonical snapshot rather than asked over the wire, so
        the build-skew seam can decide synchronously whether to REQUEST a
        refresh (idle) or paint the busy notice (not idle). The runtime is
        the authority — ``refresh_if_idle`` re-asks ``may_refresh`` — and
        this is only the viewer's best reading: streaming, a running job, or
        a parked gate each mean "busy". A cold viewer is not idle; it has no
        owner to refresh.
        """
        if self.is_cold or self._streaming:
            return False
        store = self._frontend_store
        if store is None:
            return False
        state = store.state
        if getattr(state, "streaming", False) or getattr(state, "pending_gate", None) is not None:
            return False
        for job in getattr(state, "jobs", ()) or ():
            if str(getattr(job, "status", "")) == "running":
                return False
        return True

    async def request_refresh(self) -> str:
        """Ask the bound runtime to retire now if idle and stale; see
        ``AttachClient.request_refresh``. Never raises: every failure means
        "the runtime stays", which the reaper's own check repairs within
        ``BUILD_CHECK_S + BUILD_SETTLE_S + stagger``."""
        client = self._client
        if client is None or not client.connected:
            return "kept: no runtime attached"
        ask = getattr(client, "request_refresh", None)
        if not callable(ask):
            return "kept: client cannot ask for a refresh"
        try:
            return str(await cast(Callable[[], Awaitable[str]], ask)())
        except Exception as exc:  # noqa: BLE001 — the reaper is the fallback
            # An owner too old to know the op answers the unknown-op error;
            # that is the honest "kept" for a runtime predating the refresh.
            logger.debug("refresh request failed", exc_info=True)
            return f"kept: {exc}"

    def set_went_cold_callback(self, callback: Callable[[], Any] | None) -> None:
        """Told when the runtime went away and no successor arrived.

        The viewer paints "runtime exited" on its band; the conversation is
        still readable and the next message starts a fresh runtime.
        """
        self._went_cold_callback = callback

    def _notify_stopped(self) -> None:
        """Tell the app once that this viewer's session ended deliberately.

        Fired exactly once per facade: the two recognition points (the
        owner's announcement on the wire, and the wake-marker inference)
        both route here, and either may run first.
        """
        if self._stopped_announced:
            return
        self._stopped_announced = True
        callback = self._stopped_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:  # noqa: BLE001 — a viewer notice must not break teardown
            logger.debug("stopped-session callback failed", exc_info=True)

    async def _recover_owner(self) -> None:
        if self._deliberate_stop:
            # The disconnect came from the stop this follower issued (or that
            # landed while it watched): the session is cold, not orphaned.
            # Nothing to recover — the transcript stays on screen and
            # /resume (or a peer's /resume) is the way back. A takeover here
            # would win the lease, republish a live record for a session the
            # user just stopped, and let a later `lop stop --all` SIGTERM
            # this terminal for a record it never made.
            self._owner_ready.set()  # prompts route to the stopped notice
            return
        delay = 0.1
        # Under the viewer model, owner loss has a THIRD outcome beside
        # "reattached" and "took over": the runtime exited and no successor is
        # coming, which is the ordinary end of a run-to-completion runtime and
        # not a failure at all. After this long without a record the viewer
        # stops chasing one and goes cold — the transcript stays on screen and
        # the next message engages a fresh runtime. Without a bound the loop
        # would redial forever against a session nobody is running.
        cold_deadline = time.monotonic() + COLD_FALLBACK_S
        try:
            while not self._disposed:
                if self._can_go_cold and time.monotonic() >= cold_deadline:
                    logger.info(
                        "no runtime for %s after %.0fs; the viewer is going cold",
                        self._session_id,
                        COLD_FALLBACK_S,
                    )
                    self._go_cold()
                    return
                # A stop by someone else while we watched: the transcript's
                # ``stopped_at`` marker plus no live owner is the deliberate
                # shape. Read it once at the top of each pass — cheap (one
                # small file, threaded) and it is what keeps the takeover
                # from resurrecting a session a kill switch just ended.
                if not self._deliberate_stop and await self._session_was_stopped():
                    self._deliberate_stop = True
                    self._owner_ready.set()  # prompts route to the stopped notice
                    self._notify_stopped()
                    return
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
                    except (ConnectionError, OSError, TimeoutError):
                        # ``_dial`` closes the client it built on every raise
                        # path and installs ``self._client`` only as its last
                        # statement, so a failed redial leaks no socket. What it
                        # DOES leave is the identity it stamped on entry:
                        # ``_runtime_pid = record.pid`` (and a pending
                        # ``_frontend_future``) for a runtime this viewer never
                        # attached to. That outlives the pass — ``_go_cold``
                        # does not clear it either — so a viewer that never
                        # bound reports ``runtime_pid`` as a live pid where the
                        # property promises None while cold, and
                        # ``take_unannounced_cleanup`` reads that pid to decide
                        # whether THIS viewer owns the cleanup notice: a stale
                        # match claims another runtime's notice and blanks the
                        # terminal that should have shown it (review round 1,
                        # F3). Discarding here keeps the whole loop on one
                        # rule — a pass that did not bind leaves no trace of
                        # the runtime it tried.
                        self._discard_rejected_client()
                        await asyncio.sleep(delay)
                        delay = min(delay * 1.7, 0.5)
                        continue
                    try:
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
                        if self._display_window_requested and frontend.display_history is not None:
                            await self._load_frontend_history(frontend)
                        else:
                            entries, history = (
                                await self._read_transcript(through_id=frontend.live_cursor)
                                if frontend.live_cursor is not None
                                else await self._read_transcript()
                            )
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
                        # The runtime answered but its state was refused (or
                        # the sync never came). The dialled client must not
                        # survive into the next pass: it would sit connected
                        # on the runtime, make ``is_cold`` lie, and be joined
                        # by another on every retry. See
                        # ``_discard_rejected_client``.
                        self._discard_rejected_client()
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

    async def load_job_trajectory(self, job_id: str) -> bool:
        """Fetch one child's retained event window from the owner, in pages.

        Called when a reader OPENS a subagent page. The attach snapshot carries
        no trajectories (a busy session's would exceed the socket's 1 MiB line
        limit and made the session unattachable), so the rows are pulled here
        and cached on the jobs facade.

        ``watch_job`` is issued FIRST and deliberately: subscribing before the
        read means events emitted during the fetch are relayed rather than
        lost, and the worst case is a row delivered twice, which the page
        already dedupes by ``TRAJECTORY_SEQ_KEY``. Returns False when the owner
        cannot serve trajectories (an older runtime, or the socket dropped) so
        the page can say so instead of rendering the child as empty.
        """
        client = self._client
        store = self._frontend_store
        if client is None or store is None:
            return False
        epoch = store.state.epoch
        identity = next((job for job in store.state.jobs if job.id == job_id), None)
        if identity is None:
            return False
        try:
            await client.watch_job(job_id)
        except (ConnectionError, RuntimeError):
            # An owner too old for the op cannot serve the fetch either; treat
            # the whole capability as absent.
            return False
        rows: list[dict[str, Any]] = []
        offset = 0
        base_seq: int | None = None
        details: Mapping[str, Any] = {}
        try:
            while True:
                payload = await client.job_trajectory(job_id, offset=offset)
                if not isinstance(payload, Mapping):
                    return False
                if offset == 0:
                    details = payload
                page = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
                page_base = payload.get("base_seq")
                page_base = page_base if isinstance(page_base, int) else None
                if offset and page_base != base_seq:
                    # The owner evicted from the front while we paged, so the
                    # offsets already read name different events now. Start over
                    # rather than splicing two halves of different windows.
                    rows, offset, base_seq = [], 0, None
                    continue
                base_seq = page_base
                rows.extend(page)
                total = payload.get("total")
                offset += len(page)
                if not page or not isinstance(total, int) or offset >= total:
                    break
        except (ConnectionError, RuntimeError):
            return False
        current = next((job for job in store.state.jobs if job.id == job_id), None)
        if (
            self._client is not client
            or self._frontend_store is not store
            or store.state.epoch != epoch
            or current is None
            or current.session_id != identity.session_id
        ):
            return False  # detached/reconnected/resumed while the request was in flight
        # Into the canonical state, where the live append stream will extend it
        # from here; see ``FrontendStateStore.seed_job_trajectory``.
        if not store.seed_job_trajectory(job_id, rows):
            return False
        detail_sequence = details.get("detail_sequence")
        if (
            details.get("detail_job_id") == job_id
            and details.get("detail_epoch") == epoch
            and isinstance(detail_sequence, int)
            and "todos" in details
        ):
            todos = details["todos"]
            if todos is None or isinstance(todos, list):
                store.seed_job_todos(
                    job_id,
                    todos,
                    epoch=epoch,
                    sequence=detail_sequence,
                    session_id=details.get("detail_session_id"),
                )
        self._apply_frontend_facades(store.state)
        return True

    async def unload_job_trajectory(self, job_id: str) -> None:
        """Stop watching one child's appends (its page closed).

        The cached rows are kept: reopening the same page is common and the
        next fetch refreshes them anyway. Only the owner-side subscription is
        released, which is what bounds the delta stream.
        """
        client = self._client
        if client is None:
            return
        try:
            await client.unwatch_job(job_id)
        except (ConnectionError, RuntimeError):
            pass

    def set_takeover_callback(self, callback: Callable[[Any], Any]) -> None:
        self._takeover_callback = callback

    def set_stopped_callback(self, callback: Callable[[], Any]) -> None:
        """Install the app's handler for "the session I am watching ended".

        The sibling of :meth:`set_takeover_callback`, for the opposite
        outcome. Takeover says "the owner died, you are the owner now";
        this says "the owner ENDED this session on purpose, stay a viewer of
        something cold". The app needs the distinction to say the true thing
        on screen: without it a viewer paints nothing at the moment the stop
        lands and then answers every later message with the owner-death
        wording, promising a reconnection that will never come (round-3
        D3-1/Q3-2/U3-1).
        """
        self._stopped_callback = callback

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
    def runtime_pid(self) -> int | None:
        """Pid of the runtime this viewer is attached to; ``None`` while cold."""
        return self._runtime_pid

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

    @property
    def pending_gate(self) -> Any:
        """The pending gate without the full-state clone ``frontend_state`` pays.

        For per-frame readiness checks only; see the store's own property.
        """
        if self._frontend_store is None:
            raise RuntimeError("frontend state has not synchronized")
        return self._frontend_store.pending_gate

    @property
    def epoch(self) -> str:
        """The owner epoch without the full-state clone.

        Paired with :attr:`pending_gate` because gate IDENTITY is epoch plus
        gate, and reading the epoch through ``frontend_state`` would put the
        clone back on the same per-frame path.
        """
        return self._read_state_field("epoch")

    def subscribe_frontend(self, handler):  # type: ignore[no-untyped-def]
        if self._frontend_store is None:
            raise RuntimeError("frontend state has not synchronized")
        return self._frontend_store.subscribe(handler)

    def _read_state_field(self, name: str) -> Any:
        """One canonical field without the whole-state clone.

        These accessors are called per FRAME by the band and panels, and each
        `frontend_state` read deep-copies every job and usage row (measured:
        ~30 ms of a 135 ms cold sidebar frame). The store enforces which fields
        are safe to share; anything else still goes through `frontend_state`.
        """
        if self._frontend_store is None:
            raise RuntimeError("frontend state has not synchronized")
        return self._frontend_store.read_field(name)

    @property
    def model_label(self) -> str:
        return self.frontend_state.model_label

    @property
    def model(self) -> ModelSpec:
        # Through the COPYING path deliberately: a spec is a non-frozen model,
        # and handing out the store's own instance let a caller rewrite
        # canonical state in place (review round 2, Q6/F4).
        model = self.frontend_state.selected_model
        if model is None:
            raise RuntimeError("owner has no selected model spec")
        return model

    @property
    def effective_model(self) -> ModelSpec:
        # One snapshot, not two reads: the fallback must not be able to pair a
        # spec with a newer state's selection. Copying path, per `model`.
        state = self.frontend_state
        model = state.effective_model or state.selected_model
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

    async def record_shell(self, command: str, result: ToolResult) -> None:
        from local_operator.session.shell_record import shell_record_messages

        await self._ensure_bound()
        client = self._client
        if client is None:
            raise ConnectionError("shell receipt owner is disconnected")
        await client.record_shell(command, result.model_dump(mode="json"))
        # The owner may queue persistence behind a running turn. These are
        # accepted LIVE display rows, not a fabricated durable cursor or ACK.
        for message in shell_record_messages(command, result):
            self._live_history[message.id] = message

    def history(self) -> list[Any]:
        if not self._history_hydrated:
            raise RuntimeError(
                "display-window history is not hydrated; await materialize_history()"
            )
        return self.display_history_window()

    def display_history_window(self) -> list[Any]:
        """Loaded durable rows plus canonical live rows, in display order."""
        durable_results = {
            m.tool_call_id for m in self._history if isinstance(m, Message) and m.role == "tool"
        }
        return [self._live_history.get(m.id, m) for m in self._history] + [
            m
            for key, m in self._live_history.items()
            if key not in self._history_ids
            and not (
                isinstance(m, Message) and m.role == "tool" and m.tool_call_id in durable_results
            )
        ]

    @property
    def history_message_count(self) -> int:
        durable = (
            self._display_history.total_message_count
            if self._display_history
            else len(self._history)
        )
        return durable + len(self.display_history_window()) - len(self._history)

    @property
    def history_theme_turn_count(self) -> int:
        if self._display_history is not None:
            return self._display_history.theme_turn_count + sum(
                key not in self._history_ids and getattr(m, "role", "") in ("user", "assistant")
                for key, m in self._live_history.items()
            )
        return sum(
            getattr(m, "role", "") in ("user", "assistant") for m in self.display_history_window()
        )

    @property
    def history_opener_text(self) -> str:
        if self._display_history is not None:
            return self._display_history.opener_text
        return next((m.text for m in self._history if getattr(m, "role", "") == "user"), "")

    def history_last_message(self) -> Any:
        rows = self.display_history_window()
        return rows[-1] if rows else None

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
            raise ConnectionError(self._unavailable_reason())
        # The authoritative request currently returns a settled answer. Feed it
        # through the normal delta callback once so the existing aside widget
        # uses the same rendering path without inventing remote-only UI state.
        answer = await client.complete_aside(
            [turn.model_dump(mode="json") for turn in turns if hasattr(turn, "model_dump")]
        )
        if answer and on_delta is not None:
            on_delta(answer)
        return answer

    @property
    def supports_completion_ack(self) -> bool:
        return bool(self._client and self._client.supports_completion_ack)

    async def refresh_attention(self) -> dict[str, Any]:
        return dict(self.frontend_state.attention)

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        # Capture this binding; a takeover cannot redirect an old render callback.
        client = self._client
        if client is None or not client.connected or self._recovering:
            raise ConnectionError("session is reconnecting")
        await client.acknowledge_attention(token)
        return dict(self.frontend_state.attention)

    async def fork_snapshot(self, message: str = "") -> dict[str, Any]:
        """The owner serializes the copy; a viewer never raw-copies a live store."""
        await self._ensure_bound()
        client = self._client
        if client is None or self._recovering or not client.connected:
            raise ConnectionError("session is reconnecting; retry /fork when it is ready")
        self._snapshot_clients[client] = self._snapshot_clients.get(client, 0) + 1
        try:
            return await client.fork_snapshot(message)
        except RuntimeError as error:
            if "unknown op" in str(error):
                raise RuntimeError("this owner cannot fork; update it and retry /fork") from error
            raise
        finally:
            remaining = self._snapshot_clients[client] - 1
            if remaining:
                self._snapshot_clients[client] = remaining
            else:
                del self._snapshot_clients[client]
                if self._disposed or self._client is not client:
                    client.close()

    async def detach_viewer_gates(self) -> None:
        """Withdraw this UI's waiters without answering the owner's questions.

        A fork switch must stop the answer bridge BEFORE the app clears its
        approval widgets. Clearing first resolves those widgets as denied, which
        would silently reject an original session's pending tool on departure.
        The next attach recreates the bridge from the unchanged owner state.
        """
        # A sibling frontend may settle Q1 while detach awaits cancellation.
        # Suppress the ensuing Q2 bridge as well, until this viewer is disposed.
        self._background_approval = False
        self._keep_gate_reply = False
        task = self.suspend_viewer_gates()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    @property
    def has_pending_gate_reply(self) -> bool:
        return bool(
            self._keep_gate_reply and self._gate_task is not None and not self._gate_task.done()
        )

    def suspend_viewer_gates(
        self, *, auto_approve: bool = False, keep_answer: bool = False
    ) -> asyncio.Task[Any] | None:
        """Suspend presentation, never invent an answer during navigation.

        An answer the user already committed must finish reaching its owner.
        A visited source's explicit allow-all grant can keep approving that
        source in the background; speculative viewers always pass False.
        """
        self._gates_detached = True
        self._background_approval = auto_approve
        task = self._gate_task
        if task is not None and (keep_answer or self._keep_gate_reply):
            self._keep_gate_reply = True
            return task
        self._keep_gate_reply = False
        self._gate_task = None
        if task is not None:
            task.cancel()
        self._maybe_start_gate()
        return task

    def resume_viewer_gates(self) -> None:
        """Recreate bridges only after the source is the visible input target."""
        if self._gates_detached:
            self._gates_detached = False
            self._gate_handled_key = None
        self._maybe_start_gate()

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
            raise ConnectionError(self._unavailable_reason())
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
        # An initial attachment is a bounded startup wait, not owner recovery.
        # Join its lock before mutating so a newer selection cannot be painted
        # and then overwritten by the older initial snapshot. Recovery keeps
        # its existing explicit refusal instead of queueing commands indefinitely.
        if (
            self._can_go_cold
            and self._bind_lock.locked()
            and not self._recovering
            and not self._deliberate_stop
        ):
            await self._ensure_bound()
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
            return CompactionOutcome(False, "unavailable", self._unavailable_reason())
        try:
            detail = await client.slash("compact", "")
        except ConnectionError:
            # The disconnect can land mid-request (review round 4, NIT-1): the
            # transport's own ``not attached`` must never surface as a
            # compaction receipt, so race the same rewrite the routed-slash
            # seam performs above.
            return CompactionOutcome(False, "unavailable", self._unavailable_reason())
        return CompactionOutcome(True, detail=detail)

    # -- driving turns ------------------------------------------------------

    async def prompt(
        self,
        text: str,
        images: Sequence[ImageContent] | None = None,
        *,
        message_id: str | None = None,
    ) -> None:
        """Send a prompt to the owner, optionally under a caller-supplied id.

        ``message_id`` becomes the ``ContinuationCommand`` id, which the owner
        adopts as the ``Message`` id and announces back on the user
        ``MessageStartEvent``. A follower TUI needs that round trip for the same
        reason the owner path does: it registers the id it painted a row for and
        matches the announcement against it, so a DISTINCT message with
        colliding words still paints (#228). Without the keyword the TUI's seam
        probe found nothing to hand an id to, registered the entry id-less, and
        an attached follower kept the swallow this class of fix removes.

        The steering twin has always carried identity this way
        (``_send_steer_when_ready`` sends ``command_id=message.id``); this is
        the prompt path catching up with its own sibling. Minted here when the
        caller supplies nothing, which is the historical behaviour.
        """
        # The cold-to-attached seam: a viewer that has been LOOKING at a
        # session starts working in it here, which is the first moment a
        # runtime is actually owed. A no-op once attached.
        await self._ensure_bound()
        await self._owner_ready.wait()
        target = self._takeover_target
        if target is not None:
            # A takeover means a real in-process Session now owns the
            # conversation. Forward the id only when that target can take one:
            # the seam is optional on SessionProtocol, and a target without it
            # mints its own — the same probe the TUI makes, for the same reason.
            if message_id and "message_id" in inspect.signature(target.prompt).parameters:
                await target.prompt(text, images, message_id=message_id)
            else:
                await target.prompt(text, images)
            return
        client = self._client
        if client is None or not client.connected:
            raise ConnectionError(self._unavailable_reason())
        images_wire = [_image_to_wire(image) for image in (images or [])]
        command = (
            ContinuationCommand(
                command_id=message_id,
                session_id=self._session_id,
                text=text,
                images=images_wire,
            )
            if message_id
            else ContinuationCommand.create(self._session_id, text, images_wire)
        )
        await client.send_command(command, streaming=self._streaming)

    async def seed_history(self, messages: list[Message]) -> None:
        if self.history_message_count:
            return
        self._history = list(messages)

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.steer_message(Message.user(text, images))

    def steer_message(self, message: Message) -> None:
        asyncio.create_task(self._send_steer_when_ready(message))

    async def _send_steer_when_ready(self, message: Message) -> None:
        await self._ensure_bound()
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

    async def request_stop(self) -> str:
        """Stop the session this follower is watching — deliberately.

        Marks the intent BEFORE the op is sent: the owner's graceful stop
        closes this very socket, and the disconnect handler must read that
        EOF as the stop landing, not as owner death to recover from.

        ...and CLEARS it again if the request failed, which is the other half
        of that bargain. Both failure shapes are reachable and both tell the
        user the stop did not happen — no client attached, and an owner too
        old to know the op answering unknown-op — so latching the flag on
        them would silently disable owner-death recovery for the rest of the
        session: the user keeps working in a viewer that will never take over
        when its owner is genuinely killed hours later (round-3 MAJOR-1).
        Only a stop that was actually ACCEPTED may suppress recovery.

        The wire variant — another process stopped the owner while this
        follower watched — arrives instead as the owner's ``stopping``
        announcement, which ``_on_disconnected`` reads.
        """
        self._deliberate_stop = True
        try:
            if self._client is None:
                raise ConnectionError("not attached")
            return await self._client.request_stop()
        except BaseException:
            self._deliberate_stop = False
            raise

    async def credential_op(self, action: str, key: str = "", value: str = "") -> dict[str, Any]:
        """Run one ``/credential`` verb on the owner's store.

        The viewer hosts the masked paste (the user is sitting HERE) and the
        owner holds the value (the agent's bash commands run THERE), so this is
        the seam between the two halves. A disconnected viewer answers
        ``unavailable`` rather than raising: the caller turns that into a
        notice naming what happened, which is the whole point of the fix — a
        capability that cannot run must SAY so instead of reporting a boot
        state that will never resolve.
        """
        client = self._client
        if client is None or self._recovering or not client.connected:
            return {"ok": False, "reason": "disconnected"}
        try:
            answer = await client.credential(action, key, value)
        except Exception:  # noqa: BLE001 — a lost owner is a notice, not a crash
            logger.debug("credential op failed", exc_info=True)
            return {"ok": False, "reason": "disconnected"}
        return answer if isinstance(answer, dict) else {"ok": False, "reason": "unavailable"}

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
        # Copying path: `Usage` is accumulated in place elsewhere in the
        # harness, so a shared instance is one `+=` from corrupting state.
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

    async def retire_if_unused(self) -> str:
        """Offer this viewer's runtime back if the session was never used.

        Called when a viewer LEAVES a session it engaged eagerly — the TUI is
        quitting, or `/resume` is moving to a different conversation. Without
        it, eager engagement would leak one idle runtime per terminal opened
        and closed without a message.

        This method only ASKS. Whether the runtime actually goes is decided by
        the runtime itself, which alone can see the things that make stopping
        unsafe — a wake that just fired, a peer's message arriving, a second
        terminal attached to the same session. See
        ``RuntimeServer._retire_if_pristine``.

        Never raises. Every failure means "the runtime stays up", which is the
        same outcome as before this existed: the residency drain reaps it once
        nobody is attached. A shutdown path is the wrong place to surface an
        error nobody can act on.
        """
        client = self._client
        if client is None or not client.connected:
            return "no runtime attached"
        if self._snapshot_clients:
            # The connection dispatches requests serially. Waiting for a retire
            # reply behind a held copy blocks navigation until BOTH RPCs time
            # out, losing the very result its socket lease protects. A copy is
            # ongoing work, never evidence of an unused runtime; keep it alive.
            return "fork snapshot is still in progress"
        ask = getattr(client, "retire_if_pristine", None)
        if not callable(ask):
            return "client cannot ask for retirement"
        try:
            return str(await cast(Callable[[], Awaitable[str]], ask)())
        except Exception as exc:  # noqa: BLE001 — the drain is the fallback
            logger.debug("retire-if-pristine request failed", exc_info=True)
            return f"request failed: {exc}"

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
            if self._client not in self._snapshot_clients:
                self._client.close()
            # A pending snapshot owns the final close, including failure and
            # cancellation. No new socket, create retry, or owner restart occurs.
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

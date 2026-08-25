"""Wire types for the mobile control plane.

This module is the CONTRACT between the three parties of the mobile stack —
the registrant shim inside every interactive ``lop`` process
(:mod:`local_operator.mobile.registrant`), the daemon
(:mod:`local_operator.mobile.daemon`), and the phone web UI — and it is
deliberately stdlib-only (dataclasses, no pydantic): the registrant ships on
the CLI startup path where a pydantic import would cost every ``lop``
invocation real milliseconds, and the frames are small enough that
``dataclasses.asdict`` round-tripping is all the validation the loopback
channel needs.

Two wire formats live here:

1. **Control frames** — the newline-delimited JSON spoken on the registrant's
   loopback socket. Every frame is ``{"op": ..., ...}``; requests carry a
   caller-chosen ``req`` id the matching ``ack``/``error`` echoes so one
   socket multiplexes concurrent answers (an approval prompt and a model
   switch can be in flight together).
2. **Web payloads** — the REST/SSE JSON the daemon serves the phone. These
   are the daemon's *projection* of a session, not the harness's own event
   taxonomy: the phone never sees raw ``AgentEvent`` objects, it sees the
   same folded transcript/todo/subagent state the TUI renders, because the
   fold is where the TUI's semantics (one-line tool calls, todo marks,
   subagent roster) are defined. Keeping the fold server-side means the phone
   client stays a renderer and the TUI's semantics have exactly one
   implementation.

Both formats version together: bumping ``PROTOCOL_VERSION`` is a breaking
change for registrant and daemon alike, which is fine — they ship in the same
binary and a stale registrant is re-registered on its next heartbeat.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ContinuationCommand:
    """Producer-owned prompt retained until its transcript row is durable.

    Process discovery and request ids are transport details. This identity is
    the conversation-level receipt that survives reconnects and host changes.
    """

    command_id: str
    session_id: str
    text: str
    images: list[dict[str, str]] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.time)

    @staticmethod
    def create(
        session_id: str, text: str, images: list[dict[str, str]] | None = None
    ) -> "ContinuationCommand":
        return ContinuationCommand(
            command_id=str(uuid.uuid4()),
            session_id=session_id,
            text=text,
            images=list(images or []),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ContinuationCommand":
        """Validate an untrusted continuation payload without coercing types.

        This constructor sits on both HTTP and control-socket boundaries. Silent
        ``str(...)`` coercion turns missing, object, and list fields into model
        input and lets malformed UUIDs escape as route-level 500s, so invalid
        producer data is rejected before any owner is spawned or transcript is
        opened.
        """
        if not isinstance(data, dict):
            raise ValueError("command payload must be an object")
        command_id = data.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("command_id must be a UUID string")
        try:
            uuid.UUID(command_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("command_id must be a valid UUID") from exc
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        images = data.get("images", [])
        if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
            raise ValueError("images must be a list of objects")
        submitted_at = data.get("submitted_at", time.time())
        if not isinstance(submitted_at, (int, float)) or isinstance(submitted_at, bool):
            raise ValueError("submitted_at must be a number")
        return ContinuationCommand(
            command_id=command_id,
            session_id=session_id,
            text=text,
            images=[dict(item) for item in images],
            submitted_at=float(submitted_at),
        )


def validate_control_frame(frame: dict[str, Any]) -> None:
    """Reject malformed authenticated mutations before dispatch side effects."""
    if not isinstance(frame, dict):
        raise ValueError("control frame must be an object")
    op = frame.get("op")
    if not isinstance(op, str) or not op:
        raise ValueError("op must be a non-empty string")
    if op in ("prompt", "steer"):
        text = frame.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        images = frame.get("images", [])
        if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
            raise ValueError("images must be a list of objects")
        # Protocol-v3 producers send identity on both paths. Older authenticated
        # loopback clients omitted it, so absence remains compatible; a supplied
        # id is always validated before reaching transcript or steering state.
        if "command_id" in frame:
            ContinuationCommand.from_json(
                {
                    "command_id": frame.get("command_id"),
                    "session_id": frame.get("session_id", "control-session"),
                    "text": text,
                    "images": images,
                }
            )
    elif op == "approval_answer":
        if not isinstance(frame.get("request_id"), str) or not frame["request_id"]:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(frame.get("approved"), bool):
            raise ValueError("approved must be a boolean")
        if "remember" in frame and not isinstance(frame.get("remember"), bool):
            raise ValueError("remember must be a boolean")
    elif op == "ask_answer":
        if not isinstance(frame.get("request_id"), str) or not frame["request_id"]:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(frame.get("value"), str):
            raise ValueError("value must be a string")
    elif op == "recall_steer":
        # v4: a follower unsending a queued steer names the message identity it
        # queued (the ContinuationCommand id that became the Message id). The
        # owner matches by id in its steering queue; a drained or unknown id is
        # an ordinary "no longer queued" error, never a crash.
        if not isinstance(frame.get("command_id"), str) or not frame["command_id"]:
            raise ValueError("command_id must be a non-empty string")


#: Bumped on any breaking change to control frames or web payloads. The
#: registrant and daemon always ship together; the phone UI learns the
#: version in its bootstrap payload and can warn on a stale cached bundle.
#:
#: v2 (attach + reaping) added the ``watch``/``unwatch`` ops, the auth
#: frame's optional ``client`` field, and multi-connection registrants. The
#: bump is load-bearing for ATTACH specifically: an old (v1) registrant
#: treats any authenticated dial as THE daemon and evicts the real one, so
#: an attach client must refuse to dial a record whose ``protocol`` is < 2
#: rather than silently breaking the owner's phone bridge. The record's
#: version field is the only pre-dial gate — the socket itself speaks the
#: same frame shapes either side of the bump.
#:
#: v4 (full-TUI attach) is ADDITIVE: an attach client's auth frame may carry
#: ``"events": true`` to subscribe to the owner's raw ``AgentEvent`` relay
#: (``event`` frames) plus the one-shot ``attach_sync`` live-turn seed, and
#: the ``recall_steer`` op lets a follower unsend a queued steer. A v3 attach
#: client that omits the flag gets exactly the v3 behaviour (projection
#: frames only), and daemon connections never see the new frames, so the
#: phone path is byte-identical across the bump.
PROTOCOL_VERSION = 4

#: Which side of the owner relationship a control connection speaks for.
#: ``daemon`` (the default when the auth frame omits ``client``) may rebind
#: the owner's conversation; ``attach`` is a follower terminal that may
#: watch and steer but never rebind. Absent-means-daemon keeps an OLD
#: daemon dialing a NEW registrant on the same class it always had.
ClientKind = Literal["daemon", "attach"]

#: How many concurrent attach (follower terminal) connections one registrant
#: accepts before evicting the least-recently-seen one. Connection close is
#: detected anyway (the reader loop drops the registry entry); the cap is
#: defense against leaked-but-open sockets — half-open TCP with no FIN —
#: which liveness detection cannot see.
ATTACH_MAX_CLIENTS = 4


# ---------------------------------------------------------------------------
# Discovery record
# ---------------------------------------------------------------------------

#: Directory (under the config root) holding one record per live session.
RUN_DIRNAME = "run/mobile"

#: How often a registrant rewrites its record's ``heartbeat_at``. The daemon
#: treats a record as wedged (not merely quiet) after ``HEARTBEAT_TIMEOUT_S``.
HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_TIMEOUT_S = 45.0


@dataclass
class SessionRecord:
    """The discovery record one ``lop`` process publishes for one session.

    Lives at ``~/.local-operator/run/mobile/<pid>.json`` — keyed by pid
    because a process hosts exactly one interactive session at a time, so the
    pid is the natural uniqueness token and ``kill -9`` leaves exactly one
    stale file to reap.

    ``control_key`` is the whole authorization story of the control socket:
    the record is mode 0600 under a 0700 directory, so anything that can read
    the key is already the owning account. The daemon never transmits it
    further — the phone never learns it.
    """

    pid: int
    kind: Literal["tui", "exec", "daemon"]
    session_id: str
    conversation_name: str
    cwd: str
    model_label: str
    control_port: int
    control_key: str
    protocol: int = PROTOCOL_VERSION
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "SessionRecord":
        # Tolerate unknown keys (a NEWER binary's record read by an older
        # daemon mid-upgrade): forward-compat here is what lets a restart
        # rolling-upgrade the daemon without the phone losing sessions.
        known = {f for f in SessionRecord.__dataclass_fields__}
        return SessionRecord(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Control frames (daemon <-> registrant socket)
# ---------------------------------------------------------------------------

# Requests the daemon may send. Kept as Literal aliases rather than enums so
# frames stay plain dicts — json.loads output needs no decoding step.
ControlOp = Literal[
    "prompt",  # {command_id, text, images?} — durable idempotent user turn
    "steer",  # {text} — inject mid-turn
    "abort",  # {} — the stop button; never kills the session
    "set_model",  # {provider, model_id} — the model sheet's choice
    "set_effort",  # {effort} — one rung from the model's ladder
    "slash",  # {command, args} — execute a TUI slash command
    "new_conversation",  # {} — the TUI's /new
    "resume_session",  # {session_id} — rebind the host to another transcript
    "approval_answer",  # {request_id, approved, remember}
    "ask_answer",  # {request_id, value}
    "snapshot",  # {} — ask for a fresh welcome-equivalent projection
    "ping",  # {} — liveness probe; answered with {"op": "ack", ...}
    # v2 (attach + reaping): phone SSE subscriber transitions, daemon ->
    # registrant. "watch" = a phone just started following this session;
    # "unwatch" = the last phone left. The child's self-reaper counts these
    # to know whether a front end still holds the session. Additive on the
    # wire: an OLD registrant answers `error: unknown op`, which the daemon
    # tolerates (fire-and-forget), so a mixed-version machine keeps working.
    "watch",  # {} — a phone SSE subscriber appeared for this session
    "unwatch",  # {} — the last phone SSE subscriber left
    # v4 (full-TUI attach): unsend one queued steering message by identity.
    # Follower Esc-recall parity — the TUI matches queue contents by the
    # Message id it queued, and the owner recalls exactly that entry.
    "recall_steer",  # {command_id}
]

# Events the registrant streams to the daemon.
EventOp = Literal[
    "welcome",  # full projection, first frame after auth
    "projection",  # full projection repaint (the only push form — no deltas)
    "ack",  # {req, detail} — a request landed
    "error",  # {req, message} — a request was rejected/failed
    # v4, event-subscribed attach clients ONLY (never the daemon): the owner
    # session's raw AgentEvent stream, serialized with model_dump(mode="json").
    # Fidelity by construction — the follower renders the same events the
    # owner's own EventController consumes, so nothing is inverse-folded.
    "event",  # {data: <AgentEvent dump>}
    # v4: one-shot live-turn seed pushed right after the welcome projection to
    # event clients, so a mid-turn join can rebuild the in-flight bubble and
    # running tool cards (see LiveTurnSeed).
    "attach_sync",  # {data: <LiveTurnSeed>}
]


# ---------------------------------------------------------------------------
# Web payloads (daemon -> phone projection)
# ---------------------------------------------------------------------------

#: Transcript entries are the folded render model, mirroring the TUI's own
#: rows: user/assistant text, one line per tool call, notices. ``details``
#: carries the expand-on-tap payload (args, output, diff) so a collapsed row
#: is one line and an expanded one needs no round trip.
EntryKind = Literal[
    "user",
    "assistant",
    "tool",
    "notice",
    "steer",
    "compaction",
    "parent_message",
    "subagent_message",
]

ToolState = Literal["composing", "running", "done", "failed", "interrupted"]

SubagentStatus = Literal["running", "completed", "failed", "cancelled", "parked"]

TodoStatus = Literal["pending", "done", "blocked", "dropped"]


@dataclass
class TranscriptEntry:
    """One renderable transcript row, pre-folded for the phone."""

    id: str
    kind: EntryKind
    text: str = ""
    # tool rows
    tool_call_id: str = ""
    tool_name: str = ""
    tool_state: ToolState = "done"
    summary: str = ""  # the one-line args summary (compacted path etc.)
    intent: str = ""  # the model's own narration, when it gave one
    diff_added: int = 0
    diff_removed: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)  # expand payload
    # Image attachments on a user turn, as lightweight REFERENCES not bytes:
    # each is ``{"index": int, "mime_type": str}``. The bytes are fetched
    # lazily from ``/api/sessions/{pid}/image?entry=<id>&i=<index>`` (which
    # reads them from the on-disk transcript), NEVER inlined here — a
    # projection repaint fires on every streaming token, and a few hundred KB
    # of base64 re-sent per token would swamp the SSE. ``id`` is the message
    # id the endpoint resolves against.
    images: list[dict[str, Any]] = field(default_factory=list)
    # assistant rows stream: ``final`` flips true on message_end
    final: bool = True

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TodoItem:
    text: str
    status: TodoStatus = "pending"
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


#: The name the tool store gives a flat/implicit list — one phase called
#: ``"Todos"`` (``builtin._IMPLICIT_PHASE``). Duplicated here rather than
#: imported so the wire types stay free of a tools dependency; the coupling is
#: that a projection with EXACTLY this one phase renders headerless, mirroring
#: the TUI's flat-list back-compat rule. Keep in sync with the builtin.
IMPLICIT_TODO_PHASE = "Todos"


@dataclass
class TodoPhase:
    """One named group of todos — the TUI's phase model on the wire.

    A single implicit ``"Todos"`` phase (``IMPLICIT_TODO_PHASE``) is how a
    legacy flat list is carried, and the front-end drops the header for that
    lone-phase case so it renders identically to the pre-phase flat list."""

    name: str
    items: list[TodoItem] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "items": [item.to_json() for item in self.items]}


@dataclass
class SubagentRow:
    """One row of the subagent roster — the TUI panel's shape."""

    job_id: str
    label: str
    agent: str = "task"
    status: SubagentStatus = "running"
    progress: str = ""  # latest step line while running
    elapsed_s: float = 0.0
    model_label: str = ""
    result_text: str = ""  # settled outcome, one line
    error_text: str = ""
    parent_job_id: str | None = None
    session_id: str | None = None
    prompt: str = ""
    launch_message_id: str = ""
    effort: str = ""
    ancestors: list[str] = field(default_factory=list)
    ancestor_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    peer_ids: list[str] = field(default_factory=list)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    todos: list[TodoPhase] = field(default_factory=list)
    activity: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AskOptionWire:
    """One selectable answer on an ``ask`` question, as it crosses the wire.

    The phone needs the SAME information the terminal picker shows: the label
    AND the one-line consequence under it (``AskOption.description``), because
    ``ask`` exists for consequential either/or decisions and dropping the
    description makes the remote user answer a materially thinner question
    (UX round 1, U3). A dataclass rather than a bare string so ``asdict``
    serializes it to ``{"label", "description"}`` — JSON-serializable, unlike
    the pydantic ``AskOption`` the harness uses (which ``asdict`` would leave
    as an object ``json.dumps`` cannot encode; that crash is what the label-only
    projection originally worked around)."""

    label: str
    description: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PendingRequest:
    """An approval gate or ask dialog waiting on the user — the phone's
    highest-priority render (branding.md §7: a question for the user is the
    most prominent thing on screen)."""

    request_id: str
    kind: Literal["approval", "ask"]
    title: str
    detail: str = ""
    # Ask pickers; empty means a free-text/secret paste field. Objects, not
    # bare labels, so the phone can render each option's consequence line the
    # same way the terminal does (U3).
    options: list[AskOptionWire] = field(default_factory=list)
    # True when this ask requests a credential (``AskQuestion.secret``): the
    # phone must render a MASKED paste field with a "not stored in transcript"
    # affordance rather than a plain text box (D1/U2). The secret VALUE still
    # never rides the projection — only this flag does.
    secret: bool = False
    # Position of the CURRENT question within a multi-question ask, so the phone
    # can show "Question 1 of 2" the way the terminal header does and the user
    # knows more questions follow (U1). ``question_total`` is 1 for the common
    # single-question ask.
    question_index: int = 0
    question_total: int = 1

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def ask_pending_request(
    request_id: str,
    question: Any,
    *,
    question_index: int = 0,
    question_total: int = 1,
) -> PendingRequest:
    """Build the phone's ask card from an ``AskQuestion`` (harness type).

    The single seam both projection sites use — the TUI bridge
    (:mod:`.tui_handle`) and the daemon-owned gate (:mod:`.owned`) — so the two
    surfaces cannot drift in what they carry to the phone (was UX nit-1: one
    site used a ``str(option)`` fallback, the other ``""``). Reading through
    ``getattr`` keeps this decoupled from the pydantic model and lets tests pass
    a duck-typed stand-in.

    Carries the option consequence lines (U3), the ``secret`` flag so the card
    can mask the paste field (D1/U2) — never the secret value, which lives only
    in the picker's future — and the question position for the "N of M" header
    (U1)."""
    options = [
        AskOptionWire(
            label=str(getattr(option, "label", "")),
            description=str(getattr(option, "description", "") or ""),
        )
        for option in (getattr(question, "options", []) or [])
    ]
    return PendingRequest(
        request_id=request_id,
        kind="ask",
        title=str(getattr(question, "question", "") or "the agent is asking"),
        detail="",
        options=options,
        secret=bool(getattr(question, "secret", False)),
        question_index=question_index,
        question_total=question_total,
    )


@dataclass
class SessionProjection:
    """The full snapshot the phone renders from — the ONLY push form.

    Pushes are repaints, not deltas (the omp mobile lesson): no delta
    protocol means no drift, and correctness comes free with caps. The
    transcript is capped at the tail the phone actually renders; history
    beyond it is fetched on demand when the user scrolls up.
    """

    session_id: str
    pid: int
    kind: str = "tui"
    conversation_name: str = ""
    cwd: str = ""
    model_label: str = ""
    model_selector: str = ""  # provider/model_id — the model sheet's value
    effort: str = ""  # current rung; "" when the model has no ladder
    effort_ladder: list[str] = field(default_factory=list)
    streaming: bool = False
    # What the turn is doing RIGHT NOW, TUI-working-line style: "thinking",
    # "responding", or the running tool's intent ("auditing merged MRs").
    # Folded from live events; empty when idle. The phone's working line
    # reads this and never invents a label.
    activity: str = ""
    # Monotonic-ish seconds since the activity began, for the clock next to
    # the label. Server-computed so every phone paints the same age.
    activity_started_s: float = 0.0
    # Why streaming last stopped: "completed" (turn finished) or "aborted"
    # (the user/agent stopped it) — the phone's "interrupted — tap to resume"
    # affordance reads THIS, never an inference from streaming flipping,
    # because a finished turn also flips it. Empty until the first turn ends.
    stop_reason: str = ""
    queued_count: int = 0  # user messages waiting for the turn boundary
    ended: bool = False  # process gone; history still resumable
    degraded: bool = False  # record fresh but socket unreachable
    transcript: list[TranscriptEntry] = field(default_factory=list)
    #: Todos grouped into phases (``builtin`` stores them phased). A single
    #: implicit ``"Todos"`` phase carries a flat list and renders headerless.
    todos: list[TodoPhase] = field(default_factory=list)
    subagents: list[SubagentRow] = field(default_factory=list)
    #: The FRONT waiting request; the phone renders it as the pinned card.
    pending: PendingRequest | None = None
    #: How many requests are waiting in total (>= 1 while ``pending`` is set).
    #: A parallel tool batch can open several approvals at once; the phone
    #: shows "1 of N" so the user knows more cards follow this one.
    pending_count: int = 0
    usage: dict[str, int] = field(default_factory=dict)  # input/output tokens
    version: int = 0  # projection epoch; the phone drops stale repaints

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["pending"] = self.pending.to_json() if self.pending else None
        return data


#: Transcript cap for a projection push — the tail the phone renders without
#: scrolling. History fetches page backwards beyond it. Matches omp mobile's
#: finding that a phone renders a tail, not a log.
PROJECTION_TRANSCRIPT_LIMIT = 80


@dataclass
class LiveTurnSeed:
    """The in-flight turn, for an event client joining mid-turn (v4).

    Durable history covers everything up to the last persisted boundary and
    the event relay covers everything from "now" — this seed is the gap
    between them. Messages persist at turn boundaries, so a follower joining
    mid-turn would otherwise show no streaming bubble and no running tool
    cards until the turn ended. The owner's ``LiveTurnTracker`` maintains it
    from the same event stream the relay forwards; it is bounded state (one
    accumulated assistant message plus the open tool calls of one batch).

    ``open_tools`` carries the serialized ``tool_call_compose`` /
    ``tool_execution_start`` events (``model_dump(mode="json")``) for calls
    started but not yet ended, in emission order, so a joining client can
    replay them through its normal event path and rebuild the running cards
    exactly as a continuously-connected client painted them.
    """

    streaming: bool = False
    generation: int = 0
    #: Accumulated in-flight assistant text (MessageUpdateEvent carries the
    #: accumulated message, so the tracker keeps only the latest).
    assistant_text: str = ""
    #: Whether an assistant message is currently open (message_start seen,
    #: message_end not). Distinct from ``assistant_text`` being empty: a
    #: bubble can be open with no tokens yet.
    assistant_open: bool = False
    #: The open assistant message's id, so a joiner can dedupe the synthetic
    #: seed bubble against the same message's own later ``message_end`` (the
    #: relay delivers the real end; the seed only pre-paints the middle).
    assistant_message_id: str = ""
    open_tools: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "LiveTurnSeed":
        # Same tolerant rebuild as the other wire dataclasses: unknown keys
        # from a newer owner are dropped, missing keys default (rolling
        # upgrade mid-push must not kill the join).
        known = {f for f in LiveTurnSeed.__dataclass_fields__}
        return LiveTurnSeed(**{k: v for k, v in (data or {}).items() if k in known})


def _projection_from_json(data: dict[str, Any], record: SessionRecord) -> SessionProjection:
    """Rebuild a projection from a wire payload.

    The registrant already serialized dataclasses; this tolerates missing
    keys (a rolling upgrade mid-push) by constructing through the dataclass
    with defaults. Lives in the wire-types module (not the daemon) because
    BOTH consumers of the socket rebuild projections from the same frames —
    the daemon for the phone, the attach client for a follower terminal —
    and a copy in each is exactly how two renderers drift.

    ``record`` supplies the pid: the fold stamps 0 (the registrant does not
    know its own pid until the record is published), and the discovery record
    is the source of truth.
    """
    from dataclasses import fields

    def build(cls: type, items: list[dict[str, Any]]) -> list[Any]:
        known = {f.name for f in fields(cls)}
        return [cls(**{k: v for k, v in item.items() if k in known}) for item in items]

    known = {f.name for f in fields(SessionProjection)}
    base = {
        k: v
        for k, v in data.items()
        if k in known and k not in ("transcript", "todos", "subagents", "pending")
    }
    projection = SessionProjection(**base)
    projection.pid = record.pid
    projection.transcript = build(TranscriptEntry, data.get("transcript", []))
    # Todos arrive PHASED; rebuild the two nested dataclass levels, tolerating
    # missing keys the same way ``build`` does for a rolling upgrade mid-push.
    projection.todos = [
        TodoPhase(
            name=str(phase.get("name", "")),
            items=build(TodoItem, phase.get("items", []) or []),
        )
        for phase in data.get("todos", []) or []
    ]
    projection.subagents = build(SubagentRow, data.get("subagents", []))
    pending = data.get("pending")
    if isinstance(pending, dict):
        known_pending = {f.name for f in fields(PendingRequest)}
        pending_kwargs = {k: v for k, v in pending.items() if k in known_pending}
        # ``options`` crosses the wire as a list of {label, description} dicts;
        # rebuild the dataclass so downstream code (and to_json round-trips)
        # see AskOptionWire, not bare dicts.
        raw_options = pending_kwargs.get("options") or []
        pending_kwargs["options"] = [
            (
                AskOptionWire(
                    label=str(opt.get("label", "")),
                    description=str(opt.get("description", "")),
                )
                if isinstance(opt, dict)
                else AskOptionWire(label=str(opt))
            )
            for opt in raw_options
        ]
        projection.pending = PendingRequest(**pending_kwargs)
    else:
        projection.pending = None
    return projection


# ---------------------------------------------------------------------------
# Slash command surface (exported to the phone)
# ---------------------------------------------------------------------------


@dataclass
class SlashCommandInfo:
    """One slash command as the phone's sheet needs it — the subset of the
    TUI's ``SlashCommand`` registry that makes sense off-terminal. Commands
    that only mutate TUI chrome (``/clear``) or quit the app (``/exit``) are
    excluded at export time, not here."""

    name: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    arguments: str = "none"  # none | optional | required — ArgumentMode names, lowercased

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

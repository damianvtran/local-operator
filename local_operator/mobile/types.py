"""Wire types for the mobile control plane.

This module is the CONTRACT between the three parties of the mobile stack —
the session runtime inside every interactive ``lop`` process
(:mod:`local_operator.session.runtime.server`), the daemon
(:mod:`local_operator.mobile.daemon`), and the phone web UI — and it is
deliberately stdlib-only (dataclasses, no pydantic): the runtime ships on
the CLI startup path where a pydantic import would cost every ``lop``
invocation real milliseconds, and the frames are small enough that
``dataclasses.asdict`` round-tripping is all the validation the loopback
channel needs.

The discovery record and the control-socket constants are NOT defined here
any more — they are session-runtime concepts and live in
:mod:`local_operator.session.runtime.types`, re-exported below.

Two wire formats live here:

1. **Control frames** — the newline-delimited JSON spoken on the runtime's
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
change for runtime and daemon alike, which is fine — they ship in the same
binary and a stale runtime is re-registered on its next heartbeat.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Discovery + control-plane primitives now live in the session runtime package:
# a record, a heartbeat, a client kind and an attach cap describe one session
# reachable over a control socket, and the phone is one client of that, not
# its owner. They are re-exported here (and NOT redefined) so the whole mobile
# stack — daemon, web layer, attach client, peer send — keeps importing them
# from the path it always has. See local_operator/session/runtime/types.py for
# why that package is neutral and why RUN_DIRNAME keeps its mobile-era name.
from local_operator.session.runtime.types import (  # noqa: F401  (re-exported)
    ATTACH_MAX_CLIENTS,
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_TIMEOUT_S,
    PROTOCOL_VERSION,
    RUN_DIRNAME,
    SLASH_ACTION_RECEIPTS,
    ClientKind,
    SessionRecord,
)


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
    elif op == "cancel":
        # An unknown mode is REFUSED rather than defaulted. The two modes differ
        # in whether a running tool is cut in half, so a typo ("gracefull",
        # "soft") silently resolving to either one is a wrong-semantics bug on
        # the exact op whose reason for existing is that the distinction
        # matters. Absent is fine and means graceful — the safe default that a
        # caller who has not thought about it should get.
        mode = frame.get("mode", "graceful")
        if mode not in ("graceful", "immediate"):
            raise ValueError("mode must be 'graceful' or 'immediate'")
    elif op == "complete_aside":
        turns = frame.get("turns")
        if not isinstance(turns, list) or not all(isinstance(item, dict) for item in turns):
            raise ValueError("turns must be a list of message objects")
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
    elif op in ("slash", "slash_result"):
        if not isinstance(frame.get("command"), str) or not frame["command"]:
            raise ValueError("command must be a non-empty string")
        if not isinstance(frame.get("args"), str):
            raise ValueError("args must be a string")
    elif op == "credential":
        # The one op that carries a SECRET (``value``, store only). Typed here
        # so a non-string value is refused rather than coerced by ``str()`` at
        # the dispatch and stored as its repr (review round 1, N2). The value
        # is never inspected beyond its type.
        if not isinstance(frame.get("action"), str) or not frame["action"]:
            raise ValueError("action must be a non-empty string")
        if not isinstance(frame.get("key", ""), str):
            raise ValueError("key must be a string")
        if not isinstance(frame.get("value", ""), str):
            raise ValueError("value must be a string")
    elif op == "adopt_aside":
        messages = frame.get("messages")
        if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
            raise ValueError("messages must be a list of message objects")
    elif op == "recall_steer":
        # v4: a follower unsending a queued steer names the message identity it
        # queued (the ContinuationCommand id that became the Message id). The
        # owner matches by id in its steering queue; a drained or unknown id is
        # an ordinary "no longer queued" error, never a crash.
        if not isinstance(frame.get("command_id"), str) or not frame["command_id"]:
            raise ValueError("command_id must be a non-empty string")
    elif op == "peer_message":
        # Cross-session hand-off from another local lop process (`lop send`).
        # Only the body is load-bearing; the sender identity is advisory (it
        # feeds the cross-session indicator) so an older/leaner sender that
        # omits it still delivers, just less labelled. Do NOT bump
        # PROTOCOL_VERSION for this op: it is purely additive, and an OLD
        # registrant that predates it answers "unknown op" gracefully (see the
        # PROTOCOL_VERSION note above) — the bump is only load-bearing when an
        # old client must refuse a new registrant, which is the opposite risk.
        text = frame.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        mode = frame.get("mode", "mailbox")
        if mode not in ("mailbox", "steer"):
            raise ValueError("mode must be 'mailbox' or 'steer'")
        if "wake" in frame and not isinstance(frame.get("wake"), bool):
            raise ValueError("wake must be a boolean")
        sender = frame.get("sender", {})
        if not isinstance(sender, dict):
            raise ValueError("sender must be an object")


# ---------------------------------------------------------------------------
# Control frames (daemon <-> runtime socket)
# ---------------------------------------------------------------------------

# Requests the daemon may send. Kept as Literal aliases rather than enums so
# frames stay plain dicts — json.loads output needs no decoding step.
ControlOp = Literal[
    "prompt",  # {command_id, text, images?} — durable idempotent user turn
    "steer",  # {command_id, text, images?} — idempotent mid-turn injection
    "abort",  # {} — the stop button; never kills the session
    # The supervised-agent counterpart of ``abort``, and the reason both exist.
    # ``abort`` fires the turn's AbortSignal immediately, cancelling the running
    # tool task — right for a human at a keyboard, who wants it to stop NOW and
    # can see and repair whatever was left half-done. A machine supervisor has
    # neither of those: its agent's tools push commits, open merge requests and
    # write rows, and a cut mid-``git push`` leaves damage nobody is watching
    # for. ``cancel`` defaults to the boundary-respecting mode, which lands
    # after the in-flight tool batch has produced its results and before the
    # next model request is spent (``Session.request_graceful_cancel``).
    # ``mode: "immediate"`` is the explicit opt-in to ``abort`` semantics, so a
    # caller only cuts a tool in half by asking for it in those words.
    "cancel",  # {mode?: graceful|immediate}
    "set_model",  # {provider, model_id} — the model sheet's choice
    "set_effort",  # {effort} — one rung from the model's ladder
    "slash",  # {command, args} — execute a TUI slash command
    "complete_aside",  # {turns} — authoritative off-record provider request
    "new_conversation",  # {} — the TUI's /new
    "resume_session",  # {session_id} — rebind the runtime to another transcript
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
    # Peer-to-peer session messaging (`lop send`): a short-lived sender process
    # from ANOTHER local lop session hands a message to this one. Additive; no
    # PROTOCOL_VERSION bump (see validate_control_frame + the version note).
    "peer_message",  # {text, mode: mailbox|steer, wake?, sender?}
    # The graceful rung of the kill switch (`lop stop` / `/stop`): deny parked
    # gates, abort the turn, dispose the session, release the lease, unpublish
    # the record, exit. Additive like peer_message — an old runtime answers
    # unknown-op and the stop ladder proceeds to identity-confirmed SIGTERM,
    # which old runtimes already honour, so no version bump and no split-brain.
    "stop",  # {}
    # A viewer that engaged a runtime at mount and is leaving without having
    # used it offers the runtime back. The RUNTIME decides: it stops only if
    # nothing durable ever happened in the session and no other attach
    # client is connected, and answers "kept: …" otherwise. Additive like
    # `stop` — an old runtime answers unknown-op, the viewer logs it, and the
    # residency drain reaps the runtime seconds later as it always did.
    "retire_if_pristine",  # {} — ack detail is "retired" or "kept: <why>"
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
    "frontend_sync",  # v5 attach-only atomic FrontendSessionState snapshot
    "frontend_update",  # v5 attach-only ordered state replacement
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
    # An inbound message from another local lop session (`lop send`). Rendered
    # as a distinct cross-session card, never as the user's own turn.
    "peer_message",
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
    # Settled streaming is not the same as complete representation: transport
    # caps can replace this row with a prefix while preserving its message ID.
    text_complete: bool = True

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
    attention: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["pending"] = self.pending.to_json() if self.pending else None
        return data


#: Transcript cap for a projection push — the tail the phone renders without
#: scrolling. History fetches page backwards beyond it. Matches omp mobile's
#: finding that a phone renders a tail, not a log.
PROJECTION_TRANSCRIPT_LIMIT = 80


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
        result = []
        for item in items:
            values = {k: v for k, v in item.items() if k in known}
            if cls is TranscriptEntry:
                # Older owners did not say whether the real row ending survived.
                # Unknown completeness cannot authorize a completion receipt.
                values["text_complete"] = item.get("text_complete") is True
            result.append(cls(**values))
        return result

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

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
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

#: Bumped on any breaking change to control frames or web payloads. The
#: registrant and daemon always ship together; the phone UI learns the
#: version in its bootstrap payload and can warn on a stale cached bundle.
PROTOCOL_VERSION = 1


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
    "prompt",  # {text, images?} — a full user turn (or queue while streaming)
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
]

# Events the registrant streams to the daemon.
EventOp = Literal[
    "welcome",  # full projection, first frame after auth
    "projection",  # full projection repaint (the only push form — no deltas)
    "ack",  # {req, detail} — a request landed
    "error",  # {req, message} — a request was rejected/failed
]


# ---------------------------------------------------------------------------
# Web payloads (daemon -> phone projection)
# ---------------------------------------------------------------------------

#: Transcript entries are the folded render model, mirroring the TUI's own
#: rows: user/assistant text, one line per tool call, notices. ``details``
#: carries the expand-on-tap payload (args, output, diff) so a collapsed row
#: is one line and an expanded one needs no round trip.
EntryKind = Literal["user", "assistant", "tool", "notice", "steer", "compaction"]

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
    options: list[str] = field(default_factory=list)  # ask pickers; empty = free text

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


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
    todos: list[TodoItem] = field(default_factory=list)
    subagents: list[SubagentRow] = field(default_factory=list)
    pending: PendingRequest | None = None
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

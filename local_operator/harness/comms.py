"""Parent↔subagent messaging and lifecycle control — the ``hub`` tool's engine.

A ``task`` subagent used to be write-only: the parent could start one
(:func:`~local_operator.harness.subagent.run_subagent`), block on it
(``wait``) and list it (``jobs``), and that was the whole conversation. There
was no way to tell a running child something, ask it whether it was stuck,
change its course, stop it from the model side, or pick a stopped one back
up. This module is that missing half.

Everything here is built on primitives the session already had, because a
child IS a :class:`~local_operator.session.session.Session`:

- **notes and questions** ride ``Session.queue_aside`` — the lazy aside
  channel whose whole point is out-of-band material injected at a tool-batch
  or yield boundary without interrupting a batch mid-flight. Asides are
  thunks, so a question whose asker gave up (timeout, child settled) can
  withdraw itself with :class:`~local_operator.harness.types.StaleAside`
  instead of landing in the child's context minutes late;
- **steering** rides ``Session.steer`` — a real user message in the child's
  transcript, which is the right shape for "stop doing that, do this
  instead" and the wrong shape for "are you stuck?";
- **stopping** rides ``AsyncJobManager.cancel``, which already aborts the
  child through the runner's abort bridge;
- **resuming** rides transcript replay. ``Transcript(dir)`` rehydrates from
  disk and ``Session`` seeds its ``LoopContext`` from
  ``transcript.build_llm_history()``, so relaunching a child against the
  session directory of a stopped one genuinely continues that agent — same
  history, same tool results — rather than starting a stranger with a
  summary. It is the same mechanism as the CLI's ``--resume``.

The asymmetry is deliberate. A parent addresses children by job id (or by
unique label); a child addresses exactly one peer, its parent, because
local-operator children are one level deep and have no siblings to talk to.

Reply resolution has two paths on purpose. The child is told to answer with
its ``hub`` tool, which resolves the waiting future exactly; but a model that
ignores the tool and just answers in prose would strand the parent for the
whole timeout, so a text-only assistant message (no tool calls — it is not
still working) also resolves the question. Whichever lands first wins.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

from local_operator.harness.types import (
    AgentEvent,
    AsideResult,
    CustomMessage,
    Message,
    MessageEndEvent,
    ModelSpec,
    StaleAside,
)
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.session.transcript import TRANSCRIPT_FILENAME, TranscriptEntry

if TYPE_CHECKING:
    from local_operator.session.session import Session

logger = logging.getLogger(__name__)

#: ``CustomMessage.custom_type`` of a hub message in either direction. The
#: session's transcript→LLM converter renders it as a user message carrying
#: ``details["text"]`` (see ``_default_convert_to_llm``); persisted like any
#: other message entry, so a resumed child still sees what it was told.
HUB_MESSAGE_TYPE = "hub_message"

#: Additive host-only rows used by the subagent viewer. Child-facing hub
#: messages remain ordinary message entries because replay must still deliver
#: them to the model; these lifecycle facts never enter LLM history and exist
#: only to correlate the parent's action with a later child reply.
HUB_COMMUNICATION_CUSTOM_TYPE = "hub_communication"

#: Opening and closing tags of the model-facing envelope
#: :meth:`SubagentComms._format_to_child` wraps parent→child text in. A hub
#: steer persists as a plain role=user Message carrying this envelope, so
#: human-facing surfaces match on the tag to keep the XML away from the reader
#: (see :func:`extract_parent_message`). BOTH halves are constants and both the
#: builder and its inverse use them, which is what stops the pair drifting
#: apart — a literal on either side would silently break extraction the day the
#: tag changed.
PARENT_MESSAGE_TAG = "<parent-message>"
PARENT_MESSAGE_CLOSE_TAG = "</parent-message>"

#: The instruction line the envelope carries, per communication kind. Keyed by
#: the same ``kind`` vocabulary the journaled communication fact uses
#: (``details["kind"]``), so a surface can label an extracted envelope exactly
#: as it labels the fact.
#:
#: WHY a table rather than three inline strings: it is the SHARED secret
#: between the builder and :func:`extract_parent_message`. Extraction requires
#: an exact match against one of these lines, which is what keeps a human who
#: quotes the envelope shape in their own message ("why does my log show
#: ``<parent-message>``?") from having their words re-rendered as a parent
#: redirection. Matching on tag shape alone cannot tell the two apart.
TO_CHILD_INSTRUCTIONS: dict[str, str] = {
    "steer": (
        "This changes your instructions. Apply it from now on, and drop work it " "makes pointless."
    ),
    "ask": (
        "Answer it now with the `hub` tool — a short, direct reply — then carry on "
        "with what you were doing. Do not restructure your work around the question."
    ),
    "note": (
        "This is a note, not a question. No reply is needed unless it changes what "
        "you should do."
    ),
}

#: Instruction line -> kind, for extraction. Built from the same table so the
#: two directions cannot disagree.
_INSTRUCTION_KINDS: dict[str, str] = {
    instruction: kind for kind, instruction in TO_CHILD_INSTRUCTIONS.items()
}


@dataclass(frozen=True)
class ParentMessage:
    """A parsed ``<parent-message>`` envelope: what the parent said, and which
    kind of communication said it.

    ``kind`` is one of ``TO_CHILD_INSTRUCTIONS``' keys (``steer``/``ask``/
    ``note``), matching the vocabulary of the journaled communication fact, so
    a surface labels an extracted envelope the same way it labels the fact
    rather than assuming every envelope is a redirection.
    """

    kind: str
    body: str


def extract_parent_message(text: str) -> ParentMessage | None:
    """Parse a model-facing ``<parent-message>`` envelope into its kind and
    the human-facing body the parent authored.

    Returns None when ``text`` is not an envelope THIS code built. The shape
    parsed is exactly what :meth:`SubagentComms._format_to_child` emits:
    ``<parent-message>\\n{instruction}\\n\\n{body}\\n</parent-message>``.

    WHY this exists as a shared helper: a hub steer is persisted as a plain
    role=user Message whose text is the envelope built for the MODEL, while
    the human-facing fact is a separate custom row. Human-facing surfaces
    (the TUI subagent view, the mobile projection) must show the body the
    parent authored and never the XML wrapper — including for transcripts
    persisted before steers carried their communication fact's id, where
    body-text correlation is the only match left. Both surfaces import this
    one parser so the builder and its inverse cannot drift apart.

    CONSTRAINT — the instruction line must match ``TO_CHILD_INSTRUCTIONS``
    exactly. Keying on the tag shape alone would rewrite a HUMAN's own message
    as a parent communication the moment they quoted the envelope (asking
    about this very wrapper is a realistic thing to do), silently
    misattributing their words and stripping their text. The preamble is the
    part a quoter has no reason to reproduce verbatim, so requiring it is what
    makes the rewrite safe. These three strings have been byte-stable for the
    life of the envelope; if one is ever reworded, the OLD text must stay in
    this table or every transcript written before the change stops extracting.
    """
    stripped = text.strip()
    if not stripped.startswith(PARENT_MESSAGE_TAG):
        return None
    end = stripped.rfind(PARENT_MESSAGE_CLOSE_TAG)
    if end < 0:
        return None
    inner = stripped[len(PARENT_MESSAGE_TAG) : end]
    # The builder writes a newline, the instruction line, a blank line, then the
    # body. Split on the FIRST blank line: anything else did not come from
    # _format_to_child and is left alone (returning None) rather than guessed at.
    if inner.startswith("\n"):
        inner = inner[1:]
    separator = inner.find("\n\n")
    if separator < 0:
        return None
    kind = _INSTRUCTION_KINDS.get(inner[:separator].strip())
    if kind is None:
        return None
    return ParentMessage(kind=kind, body=inner[separator + 2 :].strip())


#: How many child records to keep. A record is ~4 short strings and outlives
#: its job row on purpose (job rows are swept 5 minutes after settling, and
#: resuming a child an hour later is a legitimate thing to want). The cap
#: exists only so a session that spawns thousands of children over a day
#: cannot grow without bound; eviction is oldest-settled-first.
MAX_RECORDS = 256

DeliveryOutcome = Literal["injected", "queued", "cancelled", "paused", "failed"]

#: Default number of transcript steps ``peek`` returns when the caller does not
#: ask for a range. Five is "what is it doing right now" — the question the op
#: exists for — rather than "replay its reasoning", which is what the whole
#: transcript is for and what makes a parent's context explode.
PEEK_DEFAULT_STEPS = 5

#: Hard ceiling on steps per peek, whatever the caller asks for. A parent that
#: requests 500 steps is not making an informed choice about its own context
#: budget; it is treating peek as a transcript dump. The char cap below is the
#: real bound, but refusing absurd counts up front keeps the failure legible.
PEEK_MAX_STEPS = 50

#: Per-step character budget. Tool RESULTS are the bulk of any transcript (a
#: single ``bash`` result can be 8 KiB on its own), and the parent peeking
#: wants to know WHICH tool ran with WHAT outcome, not to re-read its output.
#: Anything longer is elided in the middle, which keeps both the head (the
#: command, the start of an answer) and the tail (the verdict, the error).
PEEK_STEP_CHARS = 600


@dataclass(frozen=True)
class Delivery:
    """One recipient's receipt for one message.

    ``injected`` — handed to a live child's injection queue (it lands at that
    child's next tool-batch or yield boundary, not instantly).
    ``queued`` — the child does not exist yet (its job is parked behind the
    manager's capacity gate); buffered and flushed when it starts.
    ``cancelled`` — the child was stopped (``op="cancel"`` only).
    ``failed`` — unknown id, or the child has already settled.
    """

    job_id: str
    label: str
    outcome: DeliveryOutcome
    error: str | None = None


@dataclass(frozen=True)
class Reply:
    """The outcome of one question put to one child."""

    job_id: str
    label: str
    text: str | None = None
    error: str | None = None
    timed_out: bool = False


@dataclass(frozen=True)
class PeekStep:
    """One rendered step of a child's transcript.

    A "step" is one transcript message, numbered from 1 at the start of the
    child's run so the numbers are STABLE: step 12 means the same thing on
    every peek, which is what lets a parent range over "what happened since
    last time" without the window sliding under it.
    """

    #: 1-based position in the child's transcript.
    index: int
    #: ``user`` | ``assistant`` | ``tool`` | ``hub`` | ``system``.
    kind: str
    #: One-line summary: the tool name and its intent, or the speaker.
    heading: str
    #: The body, already clipped to :data:`PEEK_STEP_CHARS`.
    body: str


@dataclass(frozen=True)
class PeekWindow:
    """The result of one ``peek``: a bounded slice of a child's transcript."""

    job_id: str
    label: str
    status: str
    #: Total steps in the child's transcript right now — the denominator that
    #: tells a parent whether it is looking at the end or the middle.
    total: int
    steps: list[PeekStep] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ChildInfo:
    """One row of the parent's subagent roster (``hub op='list'``).

    Exists because the two surfaces that could already answer "what children
    do I have?" both answer a narrower question. ``resolve("all")`` returns
    only RUNNING children, and the ``jobs`` tool lists job rows, which the
    manager sweeps a few minutes after they settle. A child that failed or was
    stopped therefore became invisible well before it stopped being
    resumable \u2014 the parent held a record with a transcript directory and no
    way to enumerate it, so the one case an operator most wants to act on (a
    stuck or crashed subagent) was the case it could not see.
    """

    job_id: str
    label: str
    #: ``running`` | ``queued`` | ``starting`` | ``pausing`` | ``paused`` |
    #: ``completed`` | ``failed`` | ``cancelled`` | ``gone``. Derived in
    #: :meth:`SubagentComms.roster`; ``gone`` covers a record whose job row was
    #: swept without a recorded outcome, and ``pausing`` a pause that has been
    #: asked for but has not yet landed (defensive — see :meth:`_describe`).
    status: str
    #: Whether ``hub op='resume'`` can pick this child back up right now. The
    #: single fact the caller acts on, computed once here rather than
    #: re-derived by every reader from ``status`` plus transcript existence.
    resumable: bool
    #: Seconds since the child settled, or since launch while it is live.
    age_s: float | None = None
    #: Why ``resumable`` is False, when it is False for an interesting reason.
    detail: str | None = None
    #: Terminal payloads resolved with the same lifecycle precedence as
    #: ``status``. These survive the job-manager sweep so reconnecting readers
    #: do not have to merge an ephemeral row with durable comms state again.
    result_text: str | None = None
    error_text: str | None = None
    #: The child's TRANSCRIPT directory name — the id ``--resume`` takes, which
    #: is not the ``job_id`` above. Carried because this roster is now the only
    #: surface that can show it: children were dropped from the ``/resume``
    #: picker (they are the machine's own runs, not the user's conversations),
    #: and the job row that carries ``job_id`` is swept minutes after a child
    #: settles. Without it, an operator investigating a subagent that crashed
    #: an hour ago had no in-product path to its transcript at all — exactly
    #: the case this class's docstring says it exists to cover.
    session_id: str | None = None


@dataclass(frozen=True)
class SubagentNode:
    """Immutable presentation identity for one node in the shared lineage."""

    job_id: str
    label: str
    parent_job_id: str | None
    session_id: str | None
    session_dir: Path | None
    prompt: str = ""
    effective_prompt: str = ""
    launch_message_id: str = ""
    agent_role: str = ""
    effort: str = ""
    #: Every deterministic launch-row identity this lineage owns mapped to the
    #: concise delegated instruction authored for it, INCLUDING attempts that
    #: #314 collapsed into this record. The viewer replaces each durable
    #: ``subagent-launch:<id>`` user row with its concise prompt; without the
    #: superseded attempts here it could only reconcile the newest launch and
    #: would leak every earlier attempt's full role/team/system preamble.
    launch_prompts: dict[str, str] = field(default_factory=dict)


class ChildSession(Protocol):
    """The three things this module needs from a live child.

    Narrower than :class:`~local_operator.session.session.Session` on
    purpose: it is the whole coupling between the parent's channel and a
    child, stated in one place. Notes and questions go through
    ``queue_aside``, course changes through ``steer_message``, and the reply
    watcher rides ``subscribe``. Anything a future op needs from a child
    belongs here first, where the cost of the coupling is visible.
    """

    def queue_aside(self, thunk: Callable[[], AsideResult]) -> None: ...

    def steer_message(self, message: Message) -> None: ...

    def subscribe(self, handler: Callable[[AgentEvent], Any]) -> Callable[[], None]: ...


@dataclass
class _ChildRecord:
    """What the parent keeps about one child, live or long settled."""

    job_id: str
    label: str
    # The shared registry stays flat; this edge supplies lineage without moving
    # execution ownership out of each session's own job manager.
    parent_job_id: str | None = None
    prompt: str = ""
    effective_prompt: str = ""
    launch_message_id: str = ""
    agent_role: str = ""
    effort: str = ""
    #: Whether this child was built under an MCP activation denial. Persisted
    #: HERE because the flag lives on the child Session as in-memory state and
    #: a resume constructs a NEW session: ``agent_role`` alone cannot re-derive
    #: it, since a denial is inherited from the lineage rather than declared by
    #: the child's own role (the leaking case is precisely a plain ``task``
    #: child of a restricted ``manager`` — its role says "unrestricted"). And
    #: :meth:`SubagentComms.resume` rebuilds against ``self._session``, the
    #: comms-owning ROOT rather than the child's real parent, so the parent
    #: read in ``_build_child_session`` finds an unrestricted session and a
    #: correctly-denied grandchild came back able to activate the writes it had
    #: been refused (review round 2, R5).
    #:
    #: Carried by :meth:`SubagentComms.snapshot`/:meth:`restore` like
    #: ``agent_role``, so it outlives the PROCESS and not merely this map.
    #: Holding it only in memory left it with the same lifetime as the Session
    #: attribute it was added to outlast, and a child that settled hours ago is
    #: normally resumed after a restart (review round 3, R6).
    restricted: bool = False
    #: The child's transcript directory. Set at attach; the whole basis of
    #: resume, and the reason a record outlives the job row.
    session_dir: Path | None = None
    child: ChildSession | None = None
    unsubscribe: Callable[[], None] | None = None
    #: Notes addressed to this child before its session existed.
    pending: list[CustomMessage] = field(default_factory=list)
    #: Futures for BUFFERED questions, keyed by the id of the message they were
    #: created for. Identity, not type: :meth:`SubagentComms.attach` must arm
    #: each flushed question with the future belonging to THAT message. Binding
    #: at flush time by asking "is this a question?" and reaching for
    #: ``record.ask`` returned one question's answer to a different caller —
    #: the exact hazard ``_thunk``'s identity check exists to prevent, defeated
    #: because the identity was resolved too late (review round 2, R5).
    pending_asks: dict[str, "asyncio.Future[str]"] = field(default_factory=dict)
    #: The parent's unanswered question, if any.
    ask: asyncio.Future[str] | None = None
    #: Set when that question actually reached the child's context, so a
    #: text-only assistant message can be read as its answer. Never armed
    #: before injection: the child may have been mid-sentence about something
    #: else when the question was asked.
    armed: bool = False
    #: Stable id of the question currently armed in the child. The Future
    #: identifies a waiter inside one process, but only the communication id can
    #: correlate a durable reply after restart or transcript paging.
    ask_message_id: str | None = None
    settled: bool = False
    #: When ``detach`` released the child; the eviction order. ``None`` on a
    #: record whose child never started, which is why those evict FIRST — a
    #: record with no transcript is the one worth least.
    settled_at: float | None = None
    #: Set by :meth:`SubagentComms.attach` the moment ``child`` becomes live.
    #: Exists so :meth:`SubagentComms.ask` can WAIT for a child that has been
    #: registered but whose runner the loop has not entered yet, instead of
    #: refusing the question outright — see :meth:`SubagentComms._await_child`.
    #: Created lazily on first use because it must be bound to the running
    #: loop, and ``record_launch`` can be reached from a synchronous caller.
    attached: "asyncio.Event | None" = None
    #: Set by :meth:`SubagentComms.pause` and cleared by
    #: :meth:`SubagentComms.resume`. A pause IS a cancel underneath (see
    #: :meth:`SubagentComms.pause`), so this flag is the only thing that
    #: distinguishes "stopped because the parent wants it back later" from
    #: "stopped for good" — without it the roster would show a deliberately
    #: parked child as plain ``cancelled`` and nothing would suggest resuming
    #: it.
    paused: bool = False
    #: The child's terminal job status, captured by
    #: :meth:`SubagentComms.record_outcome` at the moment the runner settles.
    #:
    #: Recorded here rather than read from the job row on demand because the
    #: job manager SWEEPS settled rows after its retention window (5 minutes
    #: by default) while this record deliberately outlives them — that is the
    #: whole reason a child stays resumable for an hour. Without this field a
    #: parent listing its children after the sweep could not tell a child that
    #: finished cleanly from one that crashed, which is exactly the question
    #: "which of my subagents failed?" needs answered.
    outcome: str | None = None
    #: The child's terminal payload. Kept with the outcome because status and
    #: content are one durable fact after the ephemeral job row is swept.
    result_text: str | None = None
    error_text: str | None = None
    #: Attempt handles superseded by this record. Persisted so a parent can
    #: keep using an id it mentioned before either process or child resumed.
    attempt_aliases: list[str] = field(default_factory=list)
    #: Concise instruction for each SUPERSEDED attempt, keyed by that attempt's
    #: ``subagent-launch:<id>`` identity. Captured when #314 collapses a prior
    #: attempt into this record so the viewer can render every historical
    #: launch row as its concise prompt, not just the current one.
    prior_launch_prompts: dict[str, str] = field(default_factory=dict)


class SubagentComms:
    """The parent session's live handle on the children it launched.

    One instance per top-level session, shared with every child it spawns
    (the child's tool context carries the PARENT's instance — that is what
    makes ``hub`` from inside a child reach the agent that delegated to it).
    """

    def __init__(self, session: "Session") -> None:
        self._session = session
        self._records: dict[str, _ChildRecord] = {}
        self._detail_listeners: set[Callable[[str], None]] = set()
        self._aliases: dict[str, str] = {}

    def subscribe_detail_changes(self, listener: Callable[[str], None]) -> Callable[[], None]:
        """Observe child transcript mutations through the shared root registry."""
        self._detail_listeners.add(listener)

        def unsubscribe() -> None:
            self._detail_listeners.discard(listener)

        return unsubscribe

    def notify_detail_persisted(self, job_id: str) -> None:
        """Publish only after a child has durably appended its new history."""
        self._notify_detail_change(job_id)

    def _notify_detail_change(self, job_id: str) -> None:
        for listener in tuple(self._detail_listeners):
            try:
                listener(job_id)
            except Exception:  # noqa: BLE001 - projection listeners are additive
                logger.debug("subagent detail listener failed", exc_info=True)

    # -- launch bookkeeping ---------------------------------------------------

    def record_launch(
        self,
        job_id: str,
        label: str,
        *,
        parent_job_id: str | None = None,
        prompt: str = "",
        effective_prompt: str = "",
        launch_message_id: str = "",
        agent_role: str = "",
        effort: str = "",
    ) -> None:
        """Note a child that has been registered but may not have started.

        Called by :func:`~local_operator.harness.subagent.run_subagent` the
        moment the job id exists, so a parent can address a child that is
        still parked behind the capacity gate.

        MERGES into an existing record instead of replacing it, and the
        difference is load-bearing under an eager task factory. Textual
        installs ``asyncio.eager_task_factory`` on the TUI's loop
        (``textual/app.py``), which makes ``ensure_future`` execute a new
        coroutine synchronously up to its first true suspension — so the
        subagent runner registered inside ``jobs_manager.register`` can build
        its child session and call :meth:`attach` BEFORE ``register`` returns
        to ``run_subagent``, which only then calls this method. Replacing the
        record here discarded that attach: the fresh record had no ``child``,
        no ``session_dir`` and no reply watcher, so every later ``send``/
        ``steer``/``ask`` buffered into ``pending`` on a record whose flush
        (attach) had already happened and would never happen again. Live
        effect, observed 2026-08-19: a healthy reviewer subagent worked for
        41 minutes while two ``hub ask`` status checks silently never reached
        it, it was cancelled as wedged, and the roster reported every settled
        child as "never started, so it has no transcript" — which also made
        ``resume`` refuse them.
        """
        existing = self._records.get(job_id)
        if existing is not None:
            # attach() ran first (eager task factory) and its record carries
            # the live child, the flushed asides and the reply watcher; the
            # only fact this call adds is the label run_subagent chose.
            existing.label = label
            existing.parent_job_id = parent_job_id
            existing.prompt = prompt
            existing.effective_prompt = effective_prompt
            existing.launch_message_id = launch_message_id
            existing.agent_role = agent_role
            existing.effort = effort
            return
        self._records[job_id] = _ChildRecord(
            job_id=job_id,
            label=label,
            parent_job_id=parent_job_id,
            prompt=prompt,
            effective_prompt=effective_prompt,
            launch_message_id=launch_message_id,
            agent_role=agent_role,
            effort=effort,
        )
        self._evict_overflow()

    def attach(self, job_id: str, child: ChildSession, session_dir: Path) -> None:
        """Bind the live child session to its record and flush buffered notes."""
        record = self._records.get(job_id)
        if record is None:
            record = _ChildRecord(job_id=job_id, label=job_id)
            self._records[job_id] = record
        # The directory, not the human label, identifies a resumed child. Only
        # a terminal predecessor can be replaced: two live children may share a
        # fixture directory, and collapsing those would hide autonomous work.
        # Fold before exposing the continuation so every roster read observes
        # either the old attempt or the new one, never both.
        prior = next(
            (
                item
                for item in self._records.values()
                if item.job_id != job_id
                and item.session_dir == session_dir
                and (item.child is None or item.settled)
            ),
            None,
        )
        if prior is not None:
            record.label = prior.label
            record.parent_job_id = prior.parent_job_id
            record.agent_role = prior.agent_role
            record.effort = prior.effort
            # Carried across the fold like the role, and for the same reason:
            # this record REPLACES a settled attempt, so dropping the denial
            # here would let the second resume of a restricted child come back
            # wider than the first. The live stamp below still wins when it is
            # truthy, so a fold can only ever preserve a denial, never clear
            # one.
            record.restricted = record.restricted or prior.restricted
            record.attempt_aliases = list(
                dict.fromkeys([*prior.attempt_aliases, prior.job_id, *record.attempt_aliases])
            )
            # Keep every collapsed attempt's concise prompt keyed by its own
            # deterministic launch identity. The prior's transcript rows (its
            # original launch turn and any it in turn folded) still live in the
            # shared session directory this continuation replays, so the viewer
            # needs each one's authored prompt to avoid re-exposing that
            # attempt's full effective-prompt preamble (review round 4 R4-1).
            merged_prior = dict(prior.prior_launch_prompts)
            if prior.launch_message_id and prior.prompt:
                merged_prior[prior.launch_message_id] = prior.prompt
            merged_prior.update(record.prior_launch_prompts)
            record.prior_launch_prompts = merged_prior
            self._records.pop(prior.job_id, None)
        for alias in record.attempt_aliases:
            self._aliases[alias] = job_id
        record.child = child
        record.session_dir = session_dir
        # Imported here, not at module scope: ``subagent`` imports this module
        # for its comms types, so a top-level import closes the cycle. Same
        # reason ``resume`` imports ``run_subagent`` locally.
        from local_operator.harness.subagent import MCP_DENIED_ATTR

        # Read off the live child, which ``_build_child_session`` has already
        # stamped, rather than recomputed from the role: the denial is a
        # property of the LINEAGE (see the field's own note), and this is the
        # one moment the built session and its durable record are both in hand.
        record.restricted = record.restricted or bool(getattr(child, MCP_DENIED_ATTR, False))
        jobs = self._jobs()
        bind = getattr(jobs, "bind_logical_identity", None) if jobs is not None else None
        if callable(bind):
            bind(job_id, str(session_dir))
        record.settled = False
        record.unsubscribe = child.subscribe(self._make_reply_watcher(record))
        for message in record.pending:
            # Armed BY IDENTITY: the future recorded for THIS message, never
            # whatever ``record.ask`` happens to hold now. Binding by type
            # ("is this a question?") let a stale buffered question — one whose
            # asker had already timed out, and which nothing withdrew — arm the
            # NEXT asker's future and hand that caller the answer to a question
            # it never asked, with no error and no timeout to signal it (review
            # round 2, R5). A question with no live future is a note: it still
            # reaches the child, which is right (it was asked), but it can no
            # longer resolve anyone's wait.
            awaiting = record.pending_asks.pop(message.id, None)
            if awaiting is not None and awaiting.done():
                awaiting = None
            child.queue_aside(self._thunk(record, message, awaiting=awaiting))
        record.pending.clear()
        record.pending_asks.clear()
        # Last: everything a waiter in ``_await_child`` needs must be in place
        # before it is allowed to proceed, or it would queue its question onto
        # a record whose buffered notes had not been flushed yet.
        if record.attached is not None:
            record.attached.set()

    def detach(self, job_id: str) -> None:
        """Release the live child. An unanswered question fails here rather
        than burning its caller's full timeout: the child is gone, no answer
        is coming."""
        record = self._record(job_id)
        if record is None:
            return
        record.settled = True
        record.settled_at = time.time()
        record.child = None
        record.armed = False
        # Wake anyone parked in ``_await_child``: the child is gone, so the
        # attach they are waiting for is never coming. They re-check
        # ``record.child`` after the wait and report the settled reason, so
        # setting the event here fails them fast instead of at their timeout.
        if record.attached is not None:
            record.attached.set()
        if record.unsubscribe is not None:
            try:
                record.unsubscribe()
            except Exception:  # a broken unsubscribe must not fail teardown
                logger.warning("subagent comms unsubscribe failed", exc_info=True)
            record.unsubscribe = None
        self._fail_ask(record, "the subagent finished before answering")

    def is_child(self, job_id: str | None) -> bool:
        """Whether ``job_id`` names a child of this session.

        The role test the ``hub`` tool builder uses: a session whose tool
        context carries a job id this instance knows is a CHILD, and gets the
        child-shaped tool.
        """
        return job_id is not None and self._record(job_id) is not None

    # -- lineage --------------------------------------------------------------

    def nodes(self) -> list[SubagentNode]:
        """Return every known descendant in stable launch order.

        Nested launches share this registry but do not emit their lifecycle
        events through the root session. Projection consumers therefore need
        the registry's complete roster rather than trying to infer lineage from
        the root event stream, which can only ever describe direct children.
        """
        return [node for job_id in self._records if (node := self.node(job_id)) is not None]

    def node(self, job_id: str) -> SubagentNode | None:
        record = self._record(job_id)
        if record is None:
            return None
        session_id = record.session_dir.name if record.session_dir is not None else None
        child_session_id = getattr(record.child, "session_id", None)
        if child_session_id:
            session_id = str(child_session_id)
        # The current launch plus every collapsed predecessor, so the viewer
        # reconciles ALL durable launch rows this lineage owns to their concise
        # prompts rather than only the newest attempt (review round 4 R4-1).
        launch_prompts = dict(record.prior_launch_prompts)
        if record.launch_message_id and record.prompt:
            launch_prompts[record.launch_message_id] = record.prompt
        return SubagentNode(
            job_id=record.job_id,
            label=record.label,
            parent_job_id=record.parent_job_id,
            session_id=session_id,
            session_dir=record.session_dir,
            prompt=record.prompt,
            effective_prompt=record.effective_prompt,
            launch_message_id=record.launch_message_id,
            agent_role=record.agent_role,
            effort=record.effort,
            launch_prompts=launch_prompts,
        )

    def job(self, job_id: str) -> Any | None:
        """Find a node's job without centralizing its execution manager."""
        sessions: list[Any] = [self._session]
        sessions.extend(
            record.child for record in self._records.values() if record.child is not None
        )
        for session in sessions:
            manager = getattr(session, "jobs", None)
            try:
                job = manager.get(job_id) if manager is not None else None
            except Exception:
                job = None
            if job is not None:
                return job
        return None

    def parent(self, job_id: str) -> SubagentNode | None:
        node = self.node(job_id)
        return self.node(node.parent_job_id) if node is not None and node.parent_job_id else None

    def children(self, job_id: str | None) -> list[SubagentNode]:
        rows: list[SubagentNode] = []
        for record in self._records.values():
            if record.parent_job_id != job_id:
                continue
            node = self.node(record.job_id)
            if node is not None:
                rows.append(node)
        return rows

    def peers(self, job_id: str) -> list[SubagentNode]:
        node = self.node(job_id)
        if node is None:
            return []
        return [peer for peer in self.children(node.parent_job_id) if peer.job_id != job_id]

    def ancestors(self, job_id: str) -> list[SubagentNode]:
        """Root-to-parent lineage, cycle-safe for malformed legacy snapshots."""
        rows: list[SubagentNode] = []
        seen = {job_id}
        current = self.parent(job_id)
        while current is not None and current.job_id not in seen:
            seen.add(current.job_id)
            rows.append(current)
            current = self.parent(current.job_id)
        rows.reverse()
        return rows

    # -- addressing -----------------------------------------------------------

    def live_ids(self) -> list[str]:
        """Job ids of children that are running right now."""
        return [job_id for job_id, record in self._records.items() if self._is_running(record)]

    def resolve(self, target: str) -> tuple[list[str], str | None]:
        """Resolve one address to job ids: ``(ids, error)``.

        Accepts a job id, ``"all"`` (every running child) or a label. Labels
        are how the model naturally refers to a child it just launched, and
        resolving them here means it does not have to keep a private id
        table; an ambiguous label is an error listing the candidates rather
        than a silent pick.
        """
        target = target.strip()
        if not target:
            return [], "empty target"
        if target == "all":
            live = self.live_ids()
            return (live, None) if live else ([], "no running subagents")
        if target in self._records:
            return [target], None
        matches = [job_id for job_id, record in self._records.items() if record.label == target]
        if len(matches) > 1:
            # A resumed child reuses its label, so the stopped record and its
            # continuation both match. Only one of them can be running, and a
            # live child is unambiguously what an address means; without this
            # the first resume would permanently revoke the label the model
            # was told to address children by. An ambiguity among SETTLED
            # records is real and still reported.
            live = [job_id for job_id in matches if self._is_running(self._records[job_id])]
            if len(live) == 1:
                return live, None
            return [], f"label {target!r} is ambiguous: {', '.join(matches)}"
        if len(matches) == 1:
            return matches, None
        return [], f"unknown subagent {target!r}"

    def record_outcome(
        self,
        job_id: str,
        status: str,
        error_text: str | None = None,
        result_text: str | None = None,
    ) -> tuple[str, str | None, str | None] | None:
        """Remember how a child settled, before its job row is swept.

        Returns the resolved ``(status, error, result)`` so a caller interrupted
        during end-event fan-out can emit the already-winning terminal fact
        instead of inventing a contradictory cancellation event.

        Called from the subagent runner's settle paths. The job manager drops
        settled rows after its retention window while records here outlive
        them by design, so this is the only durable answer to "did that child
        finish or crash?" once the row is gone.

        It deliberately does NOT clear :attr:`_ChildRecord.paused`. A pause is
        implemented as a cancel, so pausing a child makes its runner settle
        ``cancelled`` and call straight into here; clearing the flag would
        erase the parent's intent microseconds after it was recorded, and the
        roster would show a deliberately parked child as an ordinary
        cancellation. Only :meth:`resume` ends a pause.
        """
        record = self._record(job_id)
        if record is None:
            return None

        # Cancellation is an observation that the runner task was interrupted,
        # not evidence that work which already returned a result or raised an
        # error did not settle. End-event fan-out is awaited after those facts
        # are recorded, so a cancel can arrive in that window. Terminal facts
        # therefore only move upward in certainty: cancelled < failed <
        # completed. The winning fact owns its payload; repeated writes of the
        # same fact may fill a payload that an earlier persistence pass lacked.
        precedence = {"cancelled": 0, "failed": 1, "completed": 2}
        current = record.outcome
        if current in precedence and precedence.get(status, -1) < precedence[current]:
            # Membership narrows this for human readers; pyright needs it explicit.
            assert current is not None
            return current, record.error_text, record.result_text
        if current == status:
            assert current is not None
            if result_text is not None:
                record.result_text = result_text
            if error_text is not None:
                record.error_text = error_text
            return current, record.error_text, record.result_text
        record.outcome = status
        record.result_text = result_text
        record.error_text = error_text
        return status, error_text, result_text

    def roster(self) -> list[ChildInfo]:
        """Every child this session launched, live or long settled.

        Ordered newest-launch-last (insertion order), which is how the model
        refers to them conversationally: "the last one I started".
        """
        now = time.time()
        rows: list[ChildInfo] = []
        for record in self._records.values():
            rows.append(self._describe(record, now))
        return rows

    # -- persistence ----------------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        """The durable half of every record, for persistence to the transcript.

        Only the fields that survive a process exit and matter to a resumed
        session: the identity (``job_id``/``label``), the transcript directory
        that makes resume possible, and the settled outcome so the roster can
        say how a child ended after its job row is long gone. The live handles
        (``child``, ``unsubscribe``, ``ask``, ``pending``) are deliberately
        omitted — they belong to a running loop and cannot cross a restart.

        ``session_dir`` is stored as a string (``Path`` is not JSON-native) and
        a record with none (a child that never started, so has no transcript)
        is skipped entirely: it is not resumable and carries nothing a resumed
        session could act on.
        """
        rows: list[dict[str, Any]] = []
        for record in self._records.values():
            if record.session_dir is None:
                continue
            rows.append(
                {
                    "job_id": record.job_id,
                    "label": record.label,
                    "parent_job_id": record.parent_job_id,
                    "prompt": record.prompt,
                    "effective_prompt": record.effective_prompt,
                    "launch_message_id": record.launch_message_id,
                    "agent_role": record.agent_role,
                    "effort": record.effort,
                    # Rides with the role because it is the half the role
                    # CANNOT express (see the field). Losing it here would
                    # leave the flag alive only for the process, which is the
                    # lifetime the in-memory marker already had and precisely
                    # what persisting it was meant to fix: a child that settled
                    # hours ago is normally resumed AFTER a restart, so this row
                    # is the only thing standing between a restricted lineage
                    # and a resumed grandchild that can activate the writes it
                    # was refused (review round 3, R6).
                    "restricted": record.restricted,
                    "session_dir": str(record.session_dir),
                    "outcome": record.outcome,
                    "result_text": record.result_text,
                    "error_text": record.error_text,
                    "paused": record.paused,
                    "settled_at": record.settled_at,
                    "attempt_aliases": list(record.attempt_aliases),
                    # Concise prompt for each collapsed attempt, so a resumed
                    # session that reopens this record still renders every
                    # historical launch row as its short instruction.
                    "prior_launch_prompts": dict(record.prior_launch_prompts),
                }
            )
        return rows

    def restore(self, rows: list[dict[str, Any]]) -> None:
        """Rebuild settled records from a persisted snapshot at resume.

        This is the resume basis: ``hub op='resume'`` needs the
        ``job_id \u2192 session_dir`` mapping to relaunch a child against its old
        transcript, and the roster needs the recorded outcome to say how each
        child ended. Both die with the process, so without rehydrating them a
        resumed session cannot see \u2014 let alone continue \u2014 the children the
        previous one launched.

        Every restored record is SETTLED: its live child is gone, so it carries
        no ``child`` handle and is marked ``settled`` with the persisted
        ``settled_at`` (so eviction order and the roster's age column stay
        meaningful). A row whose ``job_id`` is already present is skipped \u2014 a
        live child of this session must never be clobbered by a stale snapshot
        of its predecessor.
        """
        # Newest wins for legacy snapshots that retained one record per resume
        # attempt. Older ids become aliases of that winner rather than vanished
        # handles, which preserves parent messages and old transcript references.
        winners: list[dict[str, Any]] = []
        by_dir: dict[str, dict[str, Any]] = {}
        for row in reversed(rows):
            raw_dir = row.get("session_dir")
            logical_id = str(raw_dir) if raw_dir else ""
            winner = by_dir.get(logical_id) if logical_id else None
            if winner is not None:
                old_id = str(row.get("job_id") or "")
                aliases = [
                    *row.get("attempt_aliases", []),
                    old_id,
                    *winner.get("attempt_aliases", []),
                ]
                winner["attempt_aliases"] = list(dict.fromkeys(alias for alias in aliases if alias))
                # Fold the collapsed attempt's own launch prompt (and any it had
                # already collapsed) under the winner, so a legacy snapshot with
                # one record per resume still yields a concise prompt for every
                # historical launch row rather than the newest alone.
                folded = dict(row.get("prior_launch_prompts") or {})
                launch_id = str(row.get("launch_message_id") or "")
                launch_prompt = str(row.get("prompt") or "")
                if launch_id and launch_prompt:
                    folded[launch_id] = launch_prompt
                folded.update(winner.get("prior_launch_prompts") or {})
                winner["prior_launch_prompts"] = folded
                continue
            copied = dict(row)
            winners.append(copied)
            if logical_id:
                by_dir[logical_id] = copied
        for row in reversed(winners):
            job_id = str(row.get("job_id") or "")
            if not job_id or job_id in self._records:
                continue
            raw_dir = row.get("session_dir")
            session_dir = Path(str(raw_dir)) if raw_dir else None
            record = _ChildRecord(
                job_id=job_id,
                label=str(row.get("label") or job_id),
                parent_job_id=(str(row["parent_job_id"]) if row.get("parent_job_id") else None),
                prompt=str(row.get("prompt") or ""),
                effective_prompt=str(row.get("effective_prompt") or ""),
                launch_message_id=str(row.get("launch_message_id") or ""),
                agent_role=str(row.get("agent_role") or ""),
                effort=str(row.get("effort") or ""),
                # Missing defaults to False, which is right for a sidecar
                # written before this field existed: it reproduces today's
                # behaviour for old rows and cannot invent a denial that was
                # never recorded. A denial can only ever be ADDED afterwards,
                # by the attach stamp or the live computation.
                restricted=bool(row.get("restricted")),
                session_dir=session_dir,
                settled=True,
                settled_at=row.get("settled_at"),
                paused=bool(row.get("paused")),
                outcome=(str(row["outcome"]) if row.get("outcome") is not None else None),
                result_text=(
                    str(row["result_text"]) if row.get("result_text") is not None else None
                ),
                error_text=(str(row["error_text"]) if row.get("error_text") is not None else None),
                attempt_aliases=[str(alias) for alias in row.get("attempt_aliases", []) if alias],
                prior_launch_prompts={
                    str(key): str(value)
                    for key, value in (row.get("prior_launch_prompts") or {}).items()
                    if key and value
                },
            )
            self._records[job_id] = record
            for alias in record.attempt_aliases:
                if alias != job_id:
                    self._aliases[alias] = job_id
        self._evict_overflow()

    def _record(self, job_id: str) -> _ChildRecord | None:
        """Resolve either a current attempt id or any durable predecessor."""
        return self._records.get(self._aliases.get(job_id, job_id))

    async def peek(
        self,
        job_id: str,
        *,
        start: int | None = None,
        end: int | None = None,
        steps: int | None = None,
    ) -> PeekWindow:
        """Read a bounded slice of a child's transcript — ``hub op='peek'``.

        The child's transcript is written incrementally (each tool batch lands
        on disk at the batch boundary), so a RUNNING child can be read the same
        way a settled one is: build a fresh ``Transcript`` off its session
        directory and render a window of steps. This is the observation path
        the parent's other surfaces do not cover: ``hub op='list'`` says a
        child is running, ``wait`` blocks until it is not, and ``ask`` spends
        the child's attention on a question — while peek answers "what is it
        doing right now" without touching the child at all.

        Ranges are 1-based inclusive step numbers, stable across calls, so a
        parent can page forward (``start=<last seen + 1>``). ``steps`` is the
        shorthand for "the last N", which is the common ask. All three are
        clamped: the caller's context budget is the whole reason this op
        exists, so an out-of-range request is an error about the range, not a
        transcript dump.
        """
        record = self._record(job_id)
        if record is None:
            return PeekWindow(job_id, job_id, "gone", 0, error=f"unknown subagent {job_id!r}")
        info = self._describe(record, time.time())
        if record.session_dir is None:
            return PeekWindow(
                job_id,
                record.label,
                info.status,
                0,
                error="the subagent has not started yet, so it has no transcript to read",
            )
        transcript_file = record.session_dir / TRANSCRIPT_FILENAME
        if not transcript_file.exists():
            return PeekWindow(
                job_id,
                record.label,
                info.status,
                0,
                error="the subagent's transcript is gone from disk",
            )

        # A FRESH reader, not the child's live Transcript object: the child
        # owns an in-memory entry list that a parent must not share (and a
        # settled child has no live object at all). ``Transcript`` loads the
        # file at construction and drops malformed lines individually, so a
        # half-written final line of a RUNNING child degrades to "not there
        # yet" rather than an error.
        #
        # Off the event loop: construction parses the WHOLE file (rendering
        # is O(window) but parsing is O(total)), and a long-lived review
        # child's transcript is megabytes. The codebase's own precedent for
        # this cost class (``compact_file``) keeps it off the shared loop.
        from local_operator.session.transcript import Transcript

        session_dir = record.session_dir

        def _read() -> list[TranscriptEntry]:
            # Construction is the expensive part (whole-file parse); it runs
            # in the worker, not on the shared loop.
            return Transcript(session_dir).entries()

        entries = await asyncio.to_thread(_read)
        rendered = _render_transcript_steps(entries)
        total = len(rendered)

        if total == 0:
            return PeekWindow(job_id, record.label, info.status, 0)

        lo, hi, error = _resolve_peek_range(total, start=start, end=end, steps=steps)
        if error is not None:
            return PeekWindow(job_id, record.label, info.status, total, error=error)
        return PeekWindow(
            job_id,
            record.label,
            info.status,
            total,
            steps=rendered[lo - 1 : hi],
        )

    def _describe(self, record: _ChildRecord, now: float) -> ChildInfo:
        """Collapse a record plus its (possibly swept) job row into one row.

        Precedence is deliberate. ``paused`` outranks everything because a
        pause is implemented AS a cancel, so the row would otherwise read
        ``cancelled`` and hide the parent's own intent.

        A RECORDED outcome then outranks the job row, which is not the obvious
        ordering. The runner records its outcome from inside its own settle
        path, while ``AsyncJobManager`` only stamps the row's status once that
        coroutine has returned - so between the two there is a real window in
        which the record knows the child finished and the row still says
        ``running``. Reading the row first reports a finished child as running
        for the width of that window, which is exactly when a parent polling
        the roster is looking. Nothing reuses a job id (a resume gets a fresh
        one, and ``attach`` only ever runs before a settle), so a record
        carrying an outcome is settled for good and that outcome is always the
        newer fact.
        """
        jobs = self._jobs()
        job = jobs.get(record.job_id) if jobs is not None else None
        age: float | None = None
        detail: str | None = None
        result_text: str | None = None
        error_text: str | None = None

        if record.paused:
            # DEFENSIVE, not a window anyone can currently observe.
            # ``pause`` sets this flag and then awaits ``jobs.cancel``, which
            # stamps ``job.status = "cancelled"`` before its first suspension
            # point — so no concurrent ``list`` gets to run in between, and a
            # poll of a real parent/child session never saw this state. Do not
            # read the branch as evidence that it can.
            #
            # It is kept because it costs one comparison and makes the
            # roster/``resume`` invariant hold STRUCTURALLY rather than by luck
            # about where an ``await`` happens to sit in another module. The
            # settle-window guard below is the one that fires in practice; this
            # is the same rule applied to the pause path so that adding an
            # await inside ``cancel`` can never silently reopen the divergence.
            if self._is_running(record):
                return ChildInfo(
                    job_id=record.job_id,
                    label=record.label,
                    status="pausing",
                    resumable=False,
                    age_s=None,
                    detail="pause is still landing; it becomes resumable in a moment",
                )
            status = "paused"
        elif record.outcome is not None:
            # ``record_outcome`` lands inside the runner before the manager can
            # stamp its still-running row. Once terminal, a job id is never
            # reused, so accepting that stale live status would resurrect work.
            # A terminal live row may carry the richer final payload; otherwise
            # the durable record is the post-sweep/reconnect source of truth.
            status = record.outcome
            if job is not None and job.status != "running":
                result_text = getattr(job, "result_text", None)
                error_text = getattr(job, "error_text", None)
            else:
                result_text = record.result_text
                error_text = record.error_text
        elif job is not None and job.status == "running":
            status = "queued" if getattr(job, "queued", False) else "running"
            if record.child is None and status == "running":
                # Registered and admitted, but the runner coroutine has not
                # been entered yet. Reported distinctly because "starting" and
                # "running" call for different advice: the first resolves on
                # the next loop yield, the second may need a nudge.
                status = "starting"
        elif job is not None:
            status = job.status
            result_text = getattr(job, "result_text", None)
            error_text = getattr(job, "error_text", None)
        elif record.settled:
            status = "cancelled" if record.session_dir is not None else "gone"
        else:
            status = "gone"

        if status in ("running", "queued", "starting"):
            started = getattr(job, "start_time", None) if job is not None else None
            age = (now - started) if started else None
        elif record.settled_at is not None:
            age = now - record.settled_at

        # Enumerated rather than defaulted to True: a status that reaches here
        # without being listed is one nobody has reasoned about, and the safe
        # answer for an unknown state is "not resumable" (the parent is told to
        # wait) rather than an invitation to resume something unexamined. The
        # old default meant any status added to the branch above was born
        # silently resumable.
        # ``gone`` belongs here: it means the job row was swept without a
        # recorded outcome, which says nothing about the transcript. ``resume``
        # asks only whether the record has a readable transcript and no live
        # twin, so omitting ``gone`` made the roster refuse a resume that would
        # in fact have succeeded — the same disagreement as F1, in the safe
        # direction. The later branches still veto it when there is genuinely
        # nothing to resume.
        # ``interrupted`` is the resume feature's own status: a child that was
        # running when the process exited, rehydrated from the persisted roster.
        # Its transcript survived on disk, so it is exactly the case ``resume``
        # was built for — the later transcript-existence check still vetoes it
        # if the directory is gone.
        resumable = status in (
            "completed",
            "failed",
            "cancelled",
            "paused",
            "gone",
            "interrupted",
        )
        detail = detail if resumable else "not resumable in this state"
        if status in ("running", "queued", "starting"):
            resumable, detail = False, "still running; cancel or pause it first"
        elif self._is_running(record):
            # The status is settled but the JOB ROW still says running, which is
            # the same window the precedence above exists for: the runner calls
            # ``record_outcome`` from inside its settle path, then still awaits
            # ``emit(SubagentEndEvent)`` — the parent's whole handler fan-out —
            # before returning, and only then does the manager stamp the row.
            #
            # ``resumable`` has to ask the job row here because ``resume()``
            # asks it (via ``_is_running``) and the two must never disagree:
            # deriving this from the status alone advertised "failed —
            # resumable", and the resume the parent then issued was refused with
            # "still running". A row promising a resume that then refuses is
            # worse than an honest refusal, and this window is measured in
            # hundreds of milliseconds, not nanoseconds — exactly when a parent
            # polling across the settle boundary looks.
            resumable, detail = False, "still settling; it becomes resumable in a moment"
        elif record.session_dir is None:
            resumable, detail = False, "never started, so it has no transcript"
        elif not (record.session_dir / TRANSCRIPT_FILENAME).exists():
            resumable, detail = False, "transcript is gone from disk"
        else:
            live = self._live_twin(record)
            if live is not None:
                resumable, detail = False, f"already resumed as job {live.job_id}"
            elif status == "failed" and record.error_text:
                detail = f"failed: {record.error_text}"

        return ChildInfo(
            job_id=record.job_id,
            label=record.label,
            status=status,
            resumable=resumable,
            age_s=age,
            detail=detail,
            result_text=result_text,
            error_text=error_text,
            session_id=record.session_dir.name if record.session_dir is not None else None,
        )

    def _live_twin(self, record: _ChildRecord) -> _ChildRecord | None:
        """Another running record already continuing this transcript, if any.

        Two children on one session directory destroy each other's history
        (see :meth:`resume`), so both ``resume`` and the roster's
        ``resumable`` flag have to ask this question and must agree on the
        answer.
        """
        return next(
            (
                other
                for other in self._records.values()
                if other.job_id != record.job_id
                and other.session_dir is not None
                and other.session_dir == record.session_dir
                and self._is_running(other)
            ),
            None,
        )

    def label_of(self, job_id: str) -> str:
        record = self._record(job_id)
        return record.label if record is not None else job_id

    def session_dir_of(self, job_id: str) -> Path | None:
        record = self._record(job_id)
        return record.session_dir if record is not None else None

    # -- parent -> child ------------------------------------------------------

    def send(self, job_id: str, text: str) -> Delivery:
        """Deliver a note to a child. No answer is expected or waited for."""
        return self._deliver(job_id, self._to_child_message(text, expects_reply=False))

    def steer(self, job_id: str, text: str) -> Delivery:
        """Change what a child is doing.

        A real user message rather than an aside: a course change is part of
        the child's instructions from then on, and the transcript should read
        that way — including on resume, where a note framed as an aside and a
        redirection framed as an order are very different things to replay.
        """
        record = self._record(job_id)
        if record is None:
            return Delivery(job_id, job_id, "failed", f"unknown subagent {job_id!r}")
        if record.child is None:
            if record.settled or not self._is_running(record):
                return Delivery(job_id, record.label, "failed", self._gone_reason(record))
            # Not started yet — parked behind the capacity gate, or still
            # building its session. Either way a course change has to reach a
            # child that has not read its prompt yet, so buffer it as a note.
            # It arrives before any work is done, which is the point.
            record.pending.append(self._to_child_message(text, expects_reply=False, steer=True))
            return Delivery(job_id, record.label, "queued")
        message = self._to_child_message(text, expects_reply=False, steer=True)
        self._journal_communication(
            record,
            direction="to_child",
            body=text,
            communication_id=message.id,
            kind="steer",
        )
        # ``steer_message`` (not ``steer``) so the persisted row carries the
        # SAME id as the journaled communication fact: human-facing surfaces
        # (the TUI subagent view, the mobile projection) correlate the two by
        # id and render the fact instead of the model-facing
        # ``<parent-message>`` XML envelope. ``steer`` would mint a fresh
        # Message id the correlation could never match — which is exactly how
        # the envelope leaked beside the fact. ``steer_message`` is the
        # identity-preserving seam: it queues the caller-built Message
        # verbatim. On a follower-owned child the id rides the wire as the
        # ContinuationCommand id, which the owner's handle hands back to
        # ``Session.steer`` as ``message_id``, so the correlation survives
        # that path too.
        record.child.steer_message(Message.user(str(message.details["text"]), id=message.id))
        return Delivery(job_id, record.label, "injected")

    async def _await_child(self, record: _ChildRecord, timeout_s: float) -> bool:
        """Wait (briefly) for a registered child's session to become live.

        WHY: ``run_subagent`` registers the job and ``AsyncJobManager.register``
        calls ``ensure_future``, which SCHEDULES the runner without entering
        it. A parent that launches a child and asks it something in its very
        next tool call therefore finds ``record.child is None`` — not because
        anything is wrong, but because the loop has not reached the runner yet.
        The old behaviour reported "subagent has not started yet … retry in a
        moment" and returned immediately, which is a refusal the model has no
        good answer to: it cannot yield the loop except by making another call,
        so it either polls in a loop or gives up on checking in at all. Both
        were observed live.

        Awaiting the attach event yields the loop, which is exactly what lets
        the runner start — so the wait is self-fulfilling rather than a
        gamble. A PARKED job is deliberately excluded by the caller: it may sit
        behind the capacity gate for minutes, and burning the asker's whole
        timeout on it would be worse than saying so.

        Returns True when the child is live, False on timeout (the caller then
        reports the honest not-started reason).
        """
        if record.child is not None:
            return True
        if record.attached is None:
            record.attached = asyncio.Event()
        try:
            await asyncio.wait_for(record.attached.wait(), timeout_s)
        except asyncio.TimeoutError:
            return False
        return record.child is not None

    #: How much of an ``ask`` timeout may be spent waiting for a
    #: scheduled-but-not-yet-entered child to come up, before the question is
    #: refused. A fraction rather than a constant so a caller that asked for a
    #: short timeout still gets a timely answer, and a patient caller stays
    #: patient. Capped by :data:`ATTACH_WAIT_MAX_S` because a child that needs
    #: longer than that is not merely "scheduled" — something is wrong, and
    #: saying so beats consuming the caller's whole budget in silence.
    ATTACH_WAIT_FRACTION = 0.5
    ATTACH_WAIT_MAX_S = 30.0

    async def ask(self, job_id: str, text: str, timeout_ms: int) -> Reply:
        """Put a question to a running child and wait for its answer.

        A child that is registered but not yet entered by the loop is WAITED
        for (see :meth:`_await_child`) rather than refused: "it starts when
        this session next yields" is not an actionable answer for the one
        caller who cannot yield without making another call.
        """
        record = self._record(job_id)
        if record is None:
            return Reply(job_id, job_id, error=f"unknown subagent {job_id!r}")
        # ONE guard for BOTH paths, and it has to be here rather than beside
        # the attached path's queue_aside: the buffered path used to skip it
        # and overwrite a live ``record.ask``, orphaning a caller that was
        # still waiting — nothing then held its future, so it could receive
        # neither an answer nor a failure and burned its whole budget (up to
        # the 600 s schema maximum). Refusing early is also cheaper: it
        # happens before the attach grace, so a doomed second question does
        # not first spend 30 s waiting to be refused (review round 2, R6).
        if record.ask is not None and not record.ask.done():
            return Reply(
                job_id, record.label, error="a question is already pending for this subagent"
            )
        # ``timeout_ms`` is the caller's WHOLE budget, so any time spent
        # waiting for the child to come up is deducted from the answer wait
        # below rather than added to it. Charging the grace on top let a
        # 1000 ms request block for 1452 ms while reporting it had waited
        # 1000 ms, and scaled: the 600 s schema maximum could block 900 s.
        deadline = asyncio.get_running_loop().time() + timeout_ms / 1000.0
        if record.child is None:
            if record.settled or not self._is_running(record):
                return Reply(job_id, record.label, error=self._gone_reason(record))
            jobs = self._jobs()
            job = jobs.get(job_id) if jobs is not None else None
            parked = bool(job is not None and getattr(job, "queued", False))
            # Parked behind the capacity gate: no amount of yielding starts it,
            # so report rather than burn the caller's timeout.
            grace = min(self.ATTACH_WAIT_MAX_S, (timeout_ms / 1000.0) * self.ATTACH_WAIT_FRACTION)
            if parked or not await self._await_child(record, grace):
                # A child that DIED while we waited is gone, not "not started":
                # telling a caller to retry in a moment is the polling loop this
                # wait exists to remove.
                if record.settled or not self._is_running(record):
                    return Reply(job_id, record.label, error=self._gone_reason(record))
                if not parked and self._has_begun(record):
                    # The grace expired but the RUNNER IS RUNNING — it simply
                    # has not attached (a slow child build, a resumed child
                    # replaying a large transcript, or an event loop the parent
                    # is monopolising). Refusing here reports "has not started
                    # yet ... retry in a moment" about a child that has been
                    # working for half an hour, and that message is acted on:
                    # one healthy reviewer subagent was cancelled on the
                    # strength of it after a 30 s grace expired (the wait is
                    # capped by ATTACH_WAIT_MAX_S, so a long timeout does not
                    # make it more patient).
                    #
                    # Buffer instead. ``attach`` flushes pending messages and
                    # arms each question with the future created for THAT
                    # message, so the question lands at the child's first
                    # injection boundary and the answer comes back here.
                    #
                    # RE-CHECK after the grace, the way the attached path does.
                    # The guard at the top of ``ask`` ran before this await, and
                    # this path only claims ``record.ask`` after it — so without
                    # this, two concurrent asks both pass the guard and the
                    # second orphans the first (review round 4, R8; reproduced
                    # as ``REFUSED COUNT: 0``). Unreachable today because the
                    # ``hub`` tool is ``concurrency="exclusive"`` and the loop
                    # batches it alone, but that is a property of a tool
                    # declaration elsewhere, not an invariant of this layer.
                    if record.ask is not None and not record.ask.done():
                        return Reply(
                            job_id,
                            record.label,
                            error="a question is already pending for this subagent",
                        )
                    future = asyncio.get_running_loop().create_future()
                    record.ask = future
                    record.armed = False
                    message = self._to_child_message(text, expects_reply=True)
                    record.pending.append(message)
                    # Bound to the MESSAGE, not just to ``record.ask``: a
                    # buffered question that times out leaves its message in
                    # ``pending``, and arming that stale message at flush time
                    # with whatever ``record.ask`` then held returned the old
                    # question's answer to the new asker (R5).
                    record.pending_asks[message.id] = future
                    remaining = deadline - asyncio.get_running_loop().time()
                    try:
                        if remaining <= 0:
                            return Reply(job_id, record.label, timed_out=True)
                        answer = await asyncio.wait_for(future, remaining)
                    except asyncio.TimeoutError:
                        return Reply(job_id, record.label, timed_out=True)
                    except Exception as exc:
                        return Reply(job_id, record.label, error=str(exc))
                    finally:
                        # WITHDRAW the message as well as clearing the future.
                        # Leaving it queued let a question nobody is waiting for
                        # reach the child and consume the next asker's answer.
                        self._withdraw_pending(record, message)
                        self._clear_ask(record, future)
                    return Reply(job_id, record.label, text=answer)
                return Reply(job_id, record.label, error=self._not_started_reason(record))
        if record.child is None:  # pragma: no cover - narrowing for the checker
            return Reply(job_id, record.label, error=self._not_started_reason(record))
        if record.ask is not None and not record.ask.done():
            return Reply(
                job_id, record.label, error="a question is already pending for this subagent"
            )

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        record.ask = future
        record.armed = False
        message = self._to_child_message(text, expects_reply=True)
        record.child.queue_aside(self._thunk(record, message, awaiting=future))
        try:
            # Whatever is LEFT of the caller's budget after the attach wait,
            # floored just above zero so a fully-spent budget still gets one
            # scheduling pass at an answer instead of raising immediately.
            remaining = max(0.001, deadline - asyncio.get_running_loop().time())
            answer = await asyncio.wait_for(future, remaining)
        except asyncio.TimeoutError:
            # The question stays in the child's context — it was asked, and a
            # late answer is still useful to the child's own reasoning — but
            # this caller stops waiting for it.
            return Reply(job_id, record.label, timed_out=True)
        except Exception as exc:  # discarded aside / child gone
            return Reply(job_id, record.label, error=str(exc))
        finally:
            # ``finally``, not one call per exit: CancelledError is a
            # BaseException, so an ``except Exception`` cleanup would leave
            # ``record.ask`` pointing at a dead future when the parent's own
            # turn is aborted mid-question.
            self._clear_ask(record, future)
        return Reply(job_id, record.label, text=answer)

    async def cancel(self, job_id: str) -> Delivery:
        """Stop a child. Idempotent from the caller's point of view: a second
        cancel reports the state rather than pretending to act."""
        record = self._record(job_id)
        label = record.label if record is not None else job_id
        jobs = self._jobs()
        if jobs is None:
            return Delivery(job_id, label, "failed", "no job manager attached to this session")
        job = jobs.get(job_id)
        if job is None and (record is None or not record.paused):
            return Delivery(job_id, label, "failed", f"unknown job {job_id!r}")
        if job is None or job.status != "running":
            # A PAUSED child is already stopped, so there is no job to abort —
            # but "cancel" still has a meaning here that nothing else expresses:
            # the parent has decided it will not be coming back for it. Without
            # this, a pause was a one-way door (only ``resume`` cleared the
            # flag), so a parent that paused a child and then changed its mind
            # had no way to say so and the roster went on advertising a pause it
            # had abandoned. Dropping the flag settles it as an ordinary
            # cancellation; the transcript is untouched, so a later resume is
            # still possible for anyone who kept the id.
            if record is not None and record.paused:
                record.paused = False
                if record.outcome is None:
                    record.outcome = "cancelled"
                return Delivery(job_id, label, "cancelled")
            state = job.status if job is not None else "gone"
            return Delivery(job_id, label, "failed", f"job is already {state}")
        await jobs.cancel(job_id)
        return Delivery(job_id, label, "cancelled")

    async def pause(self, job_id: str) -> Delivery:
        """Stop a child now, keeping it explicitly resumable.

        Mechanically a cancel: the job signal is aborted, the runner's
        teardown runs ``_persist_inflight`` so the turn the abort pre-empted
        is saved, and the transcript directory stays on disk. What ``pause``
        adds is INTENT. ``cancel`` and ``pause`` leave a child in the same
        physical state, and a parent coming back to a roster minutes later
        cannot otherwise tell "I stopped this to free a slot, pick it up
        again" from "I stopped this because it was wrong".

        A true in-memory suspend was considered and rejected: freezing the
        child at a tool-batch boundary would hold a job-capacity slot and the
        session's memory for the whole pause, and a child inside a long tool
        call would not reach a boundary for minutes \u2014 so the op would be
        slowest exactly when it is most wanted, on a wedged child. Checkpoint
        and replay costs a transcript rehydration on resume and frees
        everything meanwhile.
        """
        record = self._record(job_id)
        if record is None:
            return Delivery(job_id, job_id, "failed", f"unknown subagent {job_id!r}")
        if record.session_dir is None:
            # Nothing has been written yet, so there would be no transcript to
            # come back to; a "pause" that cannot resume is a cancel wearing
            # the wrong name, and the caller should say what it means.
            return Delivery(
                job_id,
                record.label,
                "failed",
                "subagent has not started yet, so there is no transcript to pause; "
                "use op='cancel' to drop it",
            )
        jobs = self._jobs()
        if jobs is None:
            return Delivery(
                job_id, record.label, "failed", "no job manager attached to this session"
            )
        job = jobs.get(job_id)
        if job is None or job.status != "running":
            state = job.status if job is not None else "gone"
            return Delivery(
                job_id,
                record.label,
                "failed",
                f"job is already {state}; use hub op='resume' to pick it back up",
            )
        # Set BEFORE the cancel: cancelling settles the runner, whose teardown
        # calls record_outcome synchronously on this loop. Setting the flag
        # afterwards would let the roster observe a moment where a deliberate
        # pause looked like a plain cancellation.
        record.paused = True
        await jobs.cancel(job_id)
        return Delivery(job_id, record.label, "paused")

    def resume(self, job_id: str, message: str) -> tuple[str | None, str | None]:
        """Relaunch a stopped child against its own transcript: ``(new job id,
        error)``.

        The new child reads the old one's session directory, so it replays
        every message and tool result the stopped run produced and continues
        from there. That is why this refuses when the directory is unknown
        instead of quietly spawning a fresh agent: a "resume" that starts a
        stranger is worse than no resume at all, because the parent believes
        the context survived.
        """
        from local_operator.harness.subagent import run_subagent

        record = self._record(job_id)
        if record is None:
            return None, f"unknown subagent {job_id!r}"
        if record.session_dir is None:
            return None, (
                f"subagent {record.label} never started, so it has no transcript to resume; "
                "launch a new one with 'task'"
            )
        if self._is_running(record):
            return None, (f"subagent {record.label} is still running; cancel or pause it first")
        live = self._live_twin(record)
        if live is not None:
            # Two children on one transcript directory is not merely confusing:
            # each holds its own in-memory ``Transcript._entries`` and
            # ``compact_file`` rewrites the whole file from one of them with
            # ``os.replace``, so the other's history is destroyed. Reachable in
            # normal operation — ``_maybe_compact`` calls it every turn.
            return None, (
                f"{record.label} is already resumed as job {live.job_id}; two agents sharing "
                "one transcript would overwrite each other's history"
            )
        if not (record.session_dir / TRANSCRIPT_FILENAME).exists():
            return None, f"transcript for {record.label} is gone from disk; launch a new one"
        jobs = self._jobs()
        if jobs is None:
            return None, "no job manager attached to this session"
        # The role and tier are carried FORWARD, not re-defaulted. Everything
        # that makes a child what it is hangs off ``agent``: ``_resolve_role``
        # picks the preamble (a role's instructions, a specialist's own
        # ``system_prompt.md``, or ``SCOUT_PREAMBLE``), and the profile it
        # returns decides the tool allowlist. That allowlist is a CAPABILITY
        # BOUNDARY rather than advice (see ``_build_child_session``): resumed
        # without it, a reviewer regains ``edit``/``write`` and can "helpfully"
        # fix the very code it was asked to review, and a scout loses its
        # read-only promise. It also re-stamps ``origin.json`` with the
        # agent, so a defaulted resume overwrites the real role on disk.
        #
        # ``restricted`` rides alongside for the half the role CANNOT express.
        # A denial is inherited from the lineage, so the leaking child is a
        # plain ``task`` one whose own role says "unrestricted" — and this
        # rebuild passes ``parent_session=self._session``, the comms-owning
        # ROOT rather than the child's real parent, so the parent read in
        # ``_build_child_session`` cannot recover it either. Without carrying
        # the flag, a grandchild of a restricted ``manager`` that was correctly
        # refused ``delete_issue`` while live came back from a resume able to
        # activate it (review round 2, R5).
        # ``run_subagent`` defaults ``agent`` to ``"task"``, so omitting these
        # silently downgraded every resumed child to a generic no-role one.
        agent = record.agent_role or "task"
        effort = record.effort or None
        # A resume is a SECOND LAUNCH, so it must re-resolve the tier into a
        # model the way the first one did (``run_subagent`` explains why the
        # tier does not survive that resolution and rides separately). Passing
        # ``effort`` alone would return a child launched at ``hi`` on the
        # parent's model while the panel still displayed ``hi``.
        #
        # Guarded rather than called outright: ``_session`` is a full
        # ``Session`` in production, but this class is also driven by the
        # reduced hosts and test doubles that supply only the queue/steer/
        # subscribe surface (see ``ChildSession``). A parent that cannot price
        # a tier must still be able to resume its child on the parent's model,
        # so a missing resolver degrades instead of raising.
        model_spec: ModelSpec | None = None
        resolve = getattr(self._session, "_resolve_subagent_model", None)
        if callable(resolve):
            resolved = resolve(agent, effort)
            if isinstance(resolved, ModelSpec):
                model_spec = resolved
        new_job_id = run_subagent(
            label=record.label,
            prompt=message,
            parent_session=self._session,
            jobs_manager=jobs,
            model_spec=model_spec,
            resume_dir=record.session_dir,
            agent=agent,
            effort=effort,
            restricted=record.restricted,
        )
        # The pause is over the moment its continuation exists. Left set, the
        # old record would keep advertising ``paused`` in the roster forever
        # beside the running child that replaced it, and a reader would be
        # invited to "resume" a run that is already going.
        record.paused = False
        return new_job_id, None

    # -- child -> parent ------------------------------------------------------

    def reply_to_parent(self, job_id: str, text: str) -> str:
        """Deliver a child's message to its parent; returns what happened.

        Answers the parent's pending question when the child has actually
        SEEN it, and otherwise lands as a note in the parent's context — a
        child that notices it is blocked should be able to say so unprompted.

        ``armed`` is the whole of that distinction and this path must respect
        it exactly as the prose watcher does. A question is armed only when
        its aside reached the child's context, so an unarmed question is one
        the child cannot be answering: it is either still a queued thunk, or
        one that timed out while a newer question waits. Resolving on
        ``ask is not None`` alone accepted a child's UNPROMPTED "I am blocked"
        as the answer to a question it had never read — and did double damage,
        because the real question then materialized as a ``StaleAside`` and
        was never asked, while the note the child actually sent was consumed
        rather than delivered. Unarmed messages therefore fall through to the
        note path below; nothing is lost either way.
        """
        record = self._record(job_id)
        label = record.label if record is not None else job_id
        if record is not None and record.armed and record.ask is not None and not record.ask.done():
            self._journal_communication(
                record,
                direction="to_parent",
                body=text,
                reply_to=record.ask_message_id,
            )
            record.ask.set_result(text)
            record.armed = False
            record.ask_message_id = None
            return "answered the parent's question"
        if record is not None:
            self._journal_communication(record, direction="to_parent", body=text)
        message = CustomMessage(
            custom_type=HUB_MESSAGE_TYPE,
            attribution="user",
            details={
                "direction": "to_parent",
                "job_id": job_id,
                "label": label,
                "body": text,
                "text": (
                    f"<subagent-message label={label!r} job={job_id!r}>\n"
                    f"{text}\n"
                    "</subagent-message>"
                ),
            },
        )
        self._session.queue_aside(lambda: message)
        return "delivered to the parent (it will read this at its next step)"

    # -- internals ------------------------------------------------------------

    def _jobs(self) -> Any:
        return getattr(self._session, "jobs", None)

    def _is_running(self, record: _ChildRecord) -> bool:
        jobs = self._jobs()
        if jobs is None:
            return False
        job = jobs.get(record.job_id)
        return job is not None and job.status == "running"

    @staticmethod
    def _expects_reply(message: CustomMessage) -> bool:
        """Whether a buffered message is a QUESTION (someone is blocked on it).

        Read off the message the sender built rather than tracked separately,
        so :meth:`attach`'s flush cannot drift from what was queued.
        """
        details = getattr(message, "details", None) or {}
        return bool(details.get("expects_reply"))

    def _has_begun(self, record: _ChildRecord) -> bool:
        """Whether the child's RUNNER has actually entered.

        ``AsyncJob.started_at`` is stamped as ``_run_job``'s first statement,
        before the child session is built, so it is true for the whole window
        in which a child is working but not yet attached — exactly the window
        ``ask`` must not describe as "not started". Neither ``status`` nor
        ``queued`` can answer this: ``status`` reads ``running`` from
        registration, before a line has run.
        """
        jobs = self._jobs()
        job = jobs.get(record.job_id) if jobs is not None else None
        return job is not None and getattr(job, "started_at", None) is not None

    def _not_started_reason(self, record: _ChildRecord) -> str:
        """Why a running job has no live child yet — and they are not the same.

        Two states reach here and the caller acts on them differently. A job
        the manager PARKED is waiting for a slot and may not start for
        minutes, so the honest advice is to stop waiting on it. An ADMITTED
        job is only waiting on this event loop: ``register`` calls
        ``ensure_future``, which schedules the runner without entering it, so
        a parent that registers a child and then does non-yielding work leaves
        it scheduled-but-unstarted for the whole stretch. One loop yield — the
        caller's next await, including its next tool call — is enough to start
        it, so the honest advice there is to retry.

        Neither string promises a session is being BUILT. In the state this
        branch was written for, the runner coroutine has not been entered at
        all, so nothing is under construction yet; "scheduled" covers that and
        the build that follows it, and both resolve on the same advice.

        Both used to report the capacity gate. That is a false statement about
        an admitted job, and it is the one an operator acts on: it says
        "nothing is happening" about a child that is already spending tokens,
        so a stuck-looking run gets cancelled instead of waited out.

        Neither string names ``record.label``: the only render site
        (``builtin.py``'s hub result) already prefixes ``{label} ({job_id}): ``,
        and the card truncates the reason as content — so a stuttered label
        spends the exact cells that would otherwise have carried the state.
        """
        jobs = self._jobs()
        job = jobs.get(record.job_id) if jobs is not None else None
        if job is not None and getattr(job, "queued", False):
            return (
                "subagent has not started yet (parked behind the job capacity gate); "
                "it starts when a slot frees, so do not wait on it"
            )
        if job is not None and getattr(job, "started_at", None) is not None:
            # A THIRD state, and the one that cost real work: the runner is
            # well underway but its session is not attached to this record.
            # ``_await_child``'s grace is capped at ATTACH_WAIT_MAX_S, so a
            # patient caller still lands here after 30 s and used to be told
            # the child "has not started yet ... retry in a moment" about an
            # agent half an hour into its run. An operator acted on that and
            # cancelled it. ``started_at`` is the authoritative answer (see
            # :meth:`_has_begun`), so say what is actually true.
            started = time.time() - float(job.started_at)
            return (
                f"subagent started {started:.0f}s ago but has not attached to this parent, "
                "so it cannot be questioned yet; it is RUNNING — use 'jobs'/'wait' for its "
                "result rather than cancelling it"
            )
        return (
            "subagent has not started yet (it is scheduled, and starts when this "
            "session next yields); retry in a moment"
        )

    def _gone_reason(self, record: _ChildRecord) -> str:
        # ``record.settled`` (set by detach, as the child's runner tears it
        # down) leads the job row, which the manager stamps a moment later.
        # Reading the row first would tell the caller the subagent "is
        # running" in the same breath as telling it to go fetch the result.
        jobs = self._jobs()
        job = jobs.get(record.job_id) if jobs is not None else None
        if record.settled and (job is None or job.status == "running"):
            state = "finishing"
        else:
            state = job.status if job is not None else "gone"
        return (
            f"subagent {record.label} is {state}; use 'wait'/'jobs' for its result, "
            "or hub op='resume' to pick it back up"
        )

    def _deliver(self, job_id: str, message: CustomMessage) -> Delivery:
        record = self._record(job_id)
        if record is None:
            return Delivery(job_id, job_id, "failed", f"unknown subagent {job_id!r}")
        if record.child is None:
            if record.settled or not self._is_running(record):
                return Delivery(job_id, record.label, "failed", self._gone_reason(record))
            record.pending.append(message)
            return Delivery(job_id, record.label, "queued")
        record.child.queue_aside(self._thunk(record, message))
        return Delivery(job_id, record.label, "injected")

    def _thunk(
        self,
        record: _ChildRecord,
        message: CustomMessage,
        awaiting: "asyncio.Future[str] | None" = None,
    ) -> Callable[[], Any]:
        """Wrap one message as an aside thunk evaluated at the injection
        boundary.

        A question withdraws itself if nobody is waiting for THIS one any
        more: the asker timed out or the child settled between queueing and
        the next boundary. The identity check against ``awaiting`` rather
        than "is any question pending" is load-bearing — a question that
        timed out while the child sat in a long tool call is still queued,
        and without it that stale question would be injected on the next
        boundary and would arm the NEXT question's future, so the child's
        answer to the old question would come back as the answer to the new
        one. Notes carry no future and always land.
        """

        def thunk() -> Any:
            if awaiting is not None:
                if record.ask is not awaiting or awaiting.done():
                    return StaleAside(message)
                record.armed = True
                record.ask_message_id = message.id
            self._journal_communication(
                record,
                direction="to_child",
                body=str(message.details.get("body") or ""),
                communication_id=message.id,
                kind=(
                    "steer"
                    if bool(message.details.get("steer"))
                    else "ask" if bool(message.details.get("expects_reply")) else "send"
                ),
            )
            return message

        if awaiting is not None:
            message.on_discard = lambda: self._fail_ask(
                record, "the question was withdrawn unasked", only=awaiting
            )
        return thunk

    def _make_reply_watcher(self, record: _ChildRecord) -> Callable[[AgentEvent], Any]:
        """Resolve a pending question from the child's own prose.

        The child is told to answer with its ``hub`` tool, and that is the
        precise path. This is the safety net for a model that answers in
        plain text instead: an assistant message carrying NO tool calls is
        the child talking rather than working, so it is the answer. A message
        that requests tools is not — that is mid-thought narration, and
        treating it as the reply would hand the parent "let me check the
        logs" as its answer.
        """

        async def watcher(event: AgentEvent) -> None:
            # Nested child events never flow through the root Session, but this
            # shared registry sees every attached child. Notify detail consumers
            # before ask-specific filtering so mobile history stays fresh.
            self._notify_detail_change(record.job_id)
            if not record.armed or record.ask is None or record.ask.done():
                return
            if not isinstance(event, MessageEndEvent):
                return
            message = event.message
            if not isinstance(message, Message) or message.role != "assistant":
                return
            if message.tool_calls:
                return
            text = message.text.strip()
            if not text:
                return
            self._journal_communication(
                record,
                direction="to_parent",
                body=text,
                reply_to=record.ask_message_id,
            )
            record.armed = False
            record.ask_message_id = None
            record.ask.set_result(text)

        return watcher

    def _to_child_message(
        self, text: str, *, expects_reply: bool, steer: bool = False
    ) -> CustomMessage:
        return CustomMessage(
            custom_type=HUB_MESSAGE_TYPE,
            attribution="user",
            details={
                "direction": "to_child",
                "body": text,
                "expects_reply": expects_reply,
                "steer": steer,
                "text": self._format_to_child(text, expects_reply=expects_reply, steer=steer),
            },
        )

    @staticmethod
    def _format_to_child(text: str, *, expects_reply: bool, steer: bool) -> str:
        # Both the tags and the instruction line come from the module-level
        # constants that `extract_parent_message` parses against: the inverse
        # requires an exact preamble match, so an instruction reworded only
        # here would stop every new envelope extracting.
        kind = "steer" if steer else ("ask" if expects_reply else "note")
        instruction = TO_CHILD_INSTRUCTIONS[kind]
        return f"{PARENT_MESSAGE_TAG}\n{instruction}\n\n{text}\n{PARENT_MESSAGE_CLOSE_TAG}"

    def _journal_communication(
        self,
        record: _ChildRecord,
        *,
        direction: str,
        body: str,
        communication_id: str | None = None,
        reply_to: str | None = None,
        kind: str | None = None,
    ) -> None:
        """Append a human-facing communication fact to the child's transcript.

        This is intentionally additive beside the replay-visible hub message.
        Reusing that row cannot represent replies consumed by ``ask`` (they
        never enter either model context), while changing it would alter resume
        semantics. Fire-and-forget matches aside persistence; transcript append
        itself is synchronous before its coroutine first yields.
        """
        child = record.child
        transcript = getattr(child, "_transcript", None) if child is not None else None
        if transcript is None:
            return
        details = {
            "direction": direction,
            "job_id": record.job_id,
            "label": record.label,
            "body": body,
            "communication_id": communication_id,
            "reply_to": reply_to,
            "kind": kind,
        }
        self._session._spawn_background(  # type: ignore[attr-defined]
            transcript.append_custom(HUB_COMMUNICATION_CUSTOM_TYPE, details)
        )

    def _fail_ask(
        self,
        record: _ChildRecord,
        reason: str,
        *,
        only: "asyncio.Future[str] | None" = None,
    ) -> None:
        """Fail the pending question. ``only`` restricts that to one specific
        future, so a withdrawn stale question cannot kill the question that
        replaced it."""
        pending = record.ask
        if pending is None or pending.done():
            return
        if only is not None and pending is not only:
            return
        pending.set_exception(RuntimeError(reason))
        record.armed = False
        record.ask_message_id = None

    @staticmethod
    def _clear_ask(record: _ChildRecord, future: asyncio.Future[str]) -> None:
        if record.ask is future:
            record.ask = None
        record.armed = False
        record.ask_message_id = None

    @staticmethod
    def _withdraw_pending(record: _ChildRecord, message: CustomMessage) -> None:
        """Drop a BUFFERED question nobody is waiting for any more.

        The buffered path's counterpart to ``_thunk``'s ``StaleAside``
        withdrawal. A question that was never wrapped in a thunk has no
        ``on_discard`` to fire, so a timed-out ask used to leave its message
        sitting in ``pending`` indefinitely — and the next flush injected it,
        let the child answer it, and consumed an answer meant for someone
        else. Idempotent: the message may already be gone if ``attach``
        flushed between the timeout and this call.
        """
        record.pending_asks.pop(message.id, None)
        for index, queued in enumerate(record.pending):
            if queued is message:
                del record.pending[index]
                return

    def _evict_overflow(self) -> None:
        """Drop the least useful records once the map is over the cap.

        Evictable is "no live child and no running job", NOT "detached":
        a job cancelled while still parked behind the capacity gate never
        reaches ``attach``/``detach``, so keying on ``settled`` alone left
        those records permanently unevictable and MAX_RECORDS was not a bound
        at all. Order is oldest-settled first, with never-started records
        (``settled_at is None``) going first because they have no transcript
        and so nothing to resume.
        """
        overflow = len(self._records) - MAX_RECORDS
        if overflow <= 0:
            return
        evictable = [
            record
            for record in self._records.values()
            if record.child is None and not self._is_running(record)
        ]
        evictable.sort(key=lambda record: (record.settled_at is not None, record.settled_at or 0.0))
        for record in evictable[:overflow]:
            del self._records[record.job_id]


# ---------------------------------------------------------------------------
# peek rendering — a bounded, readable view of a child's transcript
# ---------------------------------------------------------------------------


def _resolve_peek_range(
    total: int,
    *,
    start: int | None,
    end: int | None,
    steps: int | None,
) -> tuple[int, int, str | None]:
    """Turn the caller's range arguments into a clamped ``(lo, hi)`` window.

    Step numbers are 1-based inclusive positions in the child's transcript,
    stable across calls. The defaults answer the common ask — "the last few
    steps" — while every explicit bound is checked so a sloppy range becomes a
    legible error instead of an empty or runaway window.
    """
    if steps is not None:
        if steps < 1:
            return 0, 0, f"steps must be >= 1, got {steps}"
        if steps > PEEK_MAX_STEPS:
            return (
                0,
                0,
                (
                    f"steps={steps} is too many to peek at once (max {PEEK_MAX_STEPS}); "
                    "range over the transcript in pages instead"
                ),
            )
        return max(1, total - steps + 1), total, None

    if start is not None and start < 1:
        return 0, 0, f"start must be >= 1, got {start}"
    if end is not None and end < 1:
        return 0, 0, f"end must be >= 1, got {end}"
    if start is not None and start > total:
        return 0, 0, (f"nothing at step {start}: the transcript has {total} step(s) right now")
    if end is not None and start is not None and end < start:
        return 0, 0, f"end must be >= start, got {start}-{end}"

    if start is None and end is None:
        return max(1, total - PEEK_DEFAULT_STEPS + 1), total, None
    if end is not None and start is None:  # end only: a window ending there
        # Clamp BEFORE deriving lo: an end past the transcript (a public
        # caller can pass one; the hub tool's parser cannot) would otherwise
        # yield lo > hi and an empty window instead of the last steps.
        end = min(end, total)
        return max(1, end - PEEK_DEFAULT_STEPS + 1), end, None
    if start is not None and end is None:  # start only: default-sized window
        return start, min(start + PEEK_DEFAULT_STEPS - 1, total), None
    if start is not None and end is not None:
        lo, hi = start, min(end, total)
    else:  # unreachable; keeps the checker honest
        lo, hi = 1, total
    # The cap applies to EXPLICIT ranges as well as steps=: the docstring
    # promises a hard ceiling "whatever the caller asks for", and a
    # range='1-1000' against a long transcript would otherwise inject ~600k
    # characters through the one op that exists to bound context. Keep the
    # HEAD of the requested window — the caller asked to start there, and the
    # renderer's continuation hint pages them forward from the clamp point
    # (review round 1, major).
    return lo, min(hi, lo + PEEK_MAX_STEPS - 1), None


def _clip(text: str, limit: int = PEEK_STEP_CHARS) -> str:
    """Hold one step's body inside the budget, keeping head AND tail.

    The head says what was asked or run; the tail says how it ended. A plain
    head-truncation would leave every long ``bash`` result reading "exit
    code: 0" with the interesting middle gone, and a head-only view of a
    review verdict would show the preamble and hide the verdict.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n[… {len(text) - limit} chars elided …]\n" + text[-tail:]


def _blocks_text(blocks: Any) -> str:
    """The readable text of a message's content blocks; images become a marker.

    Image blocks appear in three shapes depending on where the media lives:
    inline (``data``), externalized to the attachment store (``attachment``
    digest — the shape a fresh reader of a large-image transcript sees), and
    the typed ``ImageContent`` object. All three render as a marker; peek
    never resolves the bytes.
    """
    parts: list[str] = []
    for block in blocks or []:
        if isinstance(block, dict):
            if "text" in block:
                parts.append(str(block["text"]))
            elif "data" in block or "attachment" in block or block.get("type") == "image":
                parts.append("[image]")
        elif hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(part for part in parts if part)


def _call_line(call: Any) -> str:
    """One tool call as a single readable line: name, intent, key arguments."""
    if isinstance(call, dict):
        name = str(call.get("name", "?"))
        args = call.get("arguments") or {}
    else:
        name = call.name
        args = call.arguments or {}
    if not isinstance(args, dict):
        args = {}
    intent = str(args.get("i") or "").strip()
    head = f"{name}" + (f" — {intent}" if intent else "")
    # The ONE argument that says what the call touches; enough to follow the
    # child's trail without re-reading its arguments.
    for key in ("path", "command", "pattern", "url", "query", "label", "prompt"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            one = " ".join(value.split())
            return f"{head}: {key}={one[:120]}"
    return head


def _render_transcript_steps(entries: list[Any]) -> list[PeekStep]:
    """Render transcript entries as numbered peek steps.

    Only LLM-visible rows become steps — compaction markers and host
    bookkeeping (wake schedules, checkpoints) are invisible to the child's
    reasoning and would only spend the parent's context saying nothing. A
    hub message renders by its direction so a parent reading its own child
    sees the conversation, not raw JSON.
    """
    steps: list[PeekStep] = []
    index = 0
    for entry in entries:
        payload = entry.payload or {}
        if entry.type == "message":
            index += 1
            steps.append(_render_message_step(index, payload))
        # compaction + custom bookkeeping: deliberately not a step.
    return steps


def _render_message_step(index: int, payload: dict[str, Any]) -> PeekStep:
    if payload.get("kind") == "custom":
        return _render_custom_step(index, payload)
    role = str(payload.get("role", "user"))
    if role == "assistant":
        calls = payload.get("tool_calls") or []
        heading = "assistant" + (
            f" calls {', '.join(str(c.get('name')) for c in calls)}" if calls else ""
        )
        body = _blocks_text(payload.get("content"))
        if calls and not body:
            body = "\n".join(_call_line(c) for c in calls)
        elif calls:
            body = body + "\n" + "\n".join(_call_line(c) for c in calls)
        return PeekStep(index, "assistant", heading, _clip(body))
    if role == "tool":
        name = str(payload.get("tool_name") or "tool")
        err = " (error)" if payload.get("is_error") else ""
        return PeekStep(
            index, "tool", f"{name} result{err}", _clip(_blocks_text(payload.get("content")))
        )
    return PeekStep(index, "user", "user", _clip(_blocks_text(payload.get("content"))))


def _render_custom_step(index: int, payload: dict[str, Any]) -> PeekStep:
    details = payload.get("details") or {}
    custom_type = str(payload.get("custom_type", "custom"))
    if custom_type == HUB_MESSAGE_TYPE:
        direction = details.get("direction")
        body = str(details.get("body") or details.get("text") or "")
        if direction == "to_child":
            sense = "question" if details.get("expects_reply") else "note"
            return PeekStep(index, "hub", f"parent → child ({sense})", _clip(body))
        if direction == "to_parent":
            return PeekStep(index, "hub", "child → parent", _clip(body))
        return PeekStep(index, "hub", "hub message", _clip(body))
    if custom_type == "compaction_summary":
        return PeekStep(index, "system", "compaction", _clip(str(details.get("summary", ""))))
    if custom_type == PEER_MESSAGE_MESSAGE_TYPE:
        # A parent peeking a child that received a `lop send` message sees who
        # reached in, so the peek view stays honest about cross-session traffic.
        sender = details.get("sender") or {}
        who = str(sender.get("conversation_name") or sender.get("pid") or "another session")
        body = str(details.get("body") or details.get("text") or "")
        return PeekStep(index, "system", f"peer ← {who}", _clip(body))
    return PeekStep(index, "system", custom_type, _clip(str(details.get("text", ""))))

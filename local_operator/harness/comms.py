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
    StaleAside,
)
from local_operator.session.transcript import TRANSCRIPT_FILENAME

if TYPE_CHECKING:
    from local_operator.session.session import Session

logger = logging.getLogger(__name__)

#: ``CustomMessage.custom_type`` of a hub message in either direction. The
#: session's transcript→LLM converter renders it as a user message carrying
#: ``details["text"]`` (see ``_default_convert_to_llm``); persisted like any
#: other message entry, so a resumed child still sees what it was told.
HUB_MESSAGE_TYPE = "hub_message"

#: How many child records to keep. A record is ~4 short strings and outlives
#: its job row on purpose (job rows are swept 5 minutes after settling, and
#: resuming a child an hour later is a legitimate thing to want). The cap
#: exists only so a session that spawns thousands of children over a day
#: cannot grow without bound; eviction is oldest-settled-first.
MAX_RECORDS = 256

DeliveryOutcome = Literal["injected", "queued", "cancelled", "failed"]


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


class ChildSession(Protocol):
    """The three things this module needs from a live child.

    Narrower than :class:`~local_operator.session.session.Session` on
    purpose: it is the whole coupling between the parent's channel and a
    child, stated in one place. Notes and questions go through
    ``queue_aside``, course changes through ``steer``, and the reply watcher
    rides ``subscribe``. Anything a future op needs from a child belongs here
    first, where the cost of the coupling is visible.
    """

    def queue_aside(self, thunk: Callable[[], AsideResult]) -> None: ...

    def steer(self, text: str) -> None: ...

    def subscribe(self, handler: Callable[[AgentEvent], Any]) -> Callable[[], None]: ...


@dataclass
class _ChildRecord:
    """What the parent keeps about one child, live or long settled."""

    job_id: str
    label: str
    #: The child's transcript directory. Set at attach; the whole basis of
    #: resume, and the reason a record outlives the job row.
    session_dir: Path | None = None
    child: ChildSession | None = None
    unsubscribe: Callable[[], None] | None = None
    #: Notes addressed to this child before its session existed.
    pending: list[CustomMessage] = field(default_factory=list)
    #: The parent's unanswered question, if any.
    ask: asyncio.Future[str] | None = None
    #: Set when that question actually reached the child's context, so a
    #: text-only assistant message can be read as its answer. Never armed
    #: before injection: the child may have been mid-sentence about something
    #: else when the question was asked.
    armed: bool = False
    settled: bool = False
    #: When ``detach`` released the child; the eviction order. ``None`` on a
    #: record whose child never started, which is why those evict FIRST — a
    #: record with no transcript is the one worth least.
    settled_at: float | None = None


class SubagentComms:
    """The parent session's live handle on the children it launched.

    One instance per top-level session, shared with every child it spawns
    (the child's tool context carries the PARENT's instance — that is what
    makes ``hub`` from inside a child reach the agent that delegated to it).
    """

    def __init__(self, session: "Session") -> None:
        self._session = session
        self._records: dict[str, _ChildRecord] = {}

    # -- launch bookkeeping ---------------------------------------------------

    def record_launch(self, job_id: str, label: str) -> None:
        """Note a child that has been registered but may not have started.

        Called by :func:`~local_operator.harness.subagent.run_subagent` the
        moment the job id exists, so a parent can address a child that is
        still parked behind the capacity gate.
        """
        self._records[job_id] = _ChildRecord(job_id=job_id, label=label)
        self._evict_overflow()

    def attach(self, job_id: str, child: ChildSession, session_dir: Path) -> None:
        """Bind the live child session to its record and flush buffered notes."""
        record = self._records.get(job_id)
        if record is None:
            record = _ChildRecord(job_id=job_id, label=job_id)
            self._records[job_id] = record
        record.child = child
        record.session_dir = session_dir
        record.settled = False
        record.unsubscribe = child.subscribe(self._make_reply_watcher(record))
        for message in record.pending:
            child.queue_aside(self._thunk(record, message))
        record.pending.clear()

    def detach(self, job_id: str) -> None:
        """Release the live child. An unanswered question fails here rather
        than burning its caller's full timeout: the child is gone, no answer
        is coming."""
        record = self._records.get(job_id)
        if record is None:
            return
        record.settled = True
        record.settled_at = time.time()
        record.child = None
        record.armed = False
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
        return job_id is not None and job_id in self._records

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

    def label_of(self, job_id: str) -> str:
        record = self._records.get(job_id)
        return record.label if record is not None else job_id

    def session_dir_of(self, job_id: str) -> Path | None:
        record = self._records.get(job_id)
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
        record = self._records.get(job_id)
        if record is None:
            return Delivery(job_id, job_id, "failed", f"unknown subagent {job_id!r}")
        if record.child is None:
            if record.settled or not self._is_running(record):
                return Delivery(job_id, record.label, "failed", self._gone_reason(record))
            # Parked behind the capacity gate: a course change has to reach a
            # child that has not read its prompt yet, so buffer it as a note.
            # It arrives before any work is done, which is the point.
            record.pending.append(self._to_child_message(text, expects_reply=False, steer=True))
            return Delivery(job_id, record.label, "queued")
        record.child.steer(self._format_to_child(text, expects_reply=False, steer=True))
        return Delivery(job_id, record.label, "injected")

    async def ask(self, job_id: str, text: str, timeout_ms: int) -> Reply:
        """Put a question to a running child and wait for its answer."""
        record = self._records.get(job_id)
        if record is None:
            return Reply(job_id, job_id, error=f"unknown subagent {job_id!r}")
        if record.child is None:
            reason = (
                self._gone_reason(record)
                if record.settled or not self._is_running(record)
                else "subagent has not started yet (parked behind the job capacity gate)"
            )
            return Reply(job_id, record.label, error=reason)
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
            answer = await asyncio.wait_for(future, timeout_ms / 1000.0)
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
        record = self._records.get(job_id)
        label = record.label if record is not None else job_id
        jobs = self._jobs()
        if jobs is None:
            return Delivery(job_id, label, "failed", "no job manager attached to this session")
        job = jobs.get(job_id)
        if job is None:
            return Delivery(job_id, label, "failed", f"unknown job {job_id!r}")
        if job.status != "running":
            return Delivery(job_id, label, "failed", f"job is already {job.status}")
        await jobs.cancel(job_id)
        return Delivery(job_id, label, "cancelled")

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

        record = self._records.get(job_id)
        if record is None:
            return None, f"unknown subagent {job_id!r}"
        if record.session_dir is None:
            return None, (
                f"subagent {record.label} never started, so it has no transcript to resume; "
                "launch a new one with 'task'"
            )
        if self._is_running(record):
            return None, f"subagent {record.label} is still running; cancel it first"
        live = next(
            (
                other
                for other in self._records.values()
                if other.job_id != record.job_id
                and other.session_dir == record.session_dir
                and self._is_running(other)
            ),
            None,
        )
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
        new_job_id = run_subagent(
            label=record.label,
            prompt=message,
            parent_session=self._session,
            jobs_manager=jobs,
            resume_dir=record.session_dir,
        )
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
        record = self._records.get(job_id)
        label = record.label if record is not None else job_id
        if record is not None and record.armed and record.ask is not None and not record.ask.done():
            record.ask.set_result(text)
            record.armed = False
            return "answered the parent's question"
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
        record = self._records.get(job_id)
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
            record.armed = False
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
                "text": self._format_to_child(text, expects_reply=expects_reply, steer=steer),
            },
        )

    @staticmethod
    def _format_to_child(text: str, *, expects_reply: bool, steer: bool) -> str:
        if steer:
            instruction = (
                "This changes your instructions. Apply it from now on, and drop work it "
                "makes pointless."
            )
        elif expects_reply:
            instruction = (
                "Answer it now with the `hub` tool — a short, direct reply — then carry on "
                "with what you were doing. Do not restructure your work around the question."
            )
        else:
            instruction = (
                "This is a note, not a question. No reply is needed unless it changes what "
                "you should do."
            )
        return f"<parent-message>\n{instruction}\n\n{text}\n</parent-message>"

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

    @staticmethod
    def _clear_ask(record: _ChildRecord, future: asyncio.Future[str]) -> None:
        if record.ask is future:
            record.ask = None
        record.armed = False

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

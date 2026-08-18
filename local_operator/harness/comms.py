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

DeliveryOutcome = Literal["injected", "queued", "cancelled", "paused", "failed"]


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
    #: The child's TRANSCRIPT directory name — the id ``--resume`` takes, which
    #: is not the ``job_id`` above. Carried because this roster is now the only
    #: surface that can show it: children were dropped from the ``/resume``
    #: picker (they are the machine's own runs, not the user's conversations),
    #: and the job row that carries ``job_id`` is swept minutes after a child
    #: settles. Without it, an operator investigating a subagent that crashed
    #: an hour ago had no in-product path to its transcript at all — exactly
    #: the case this class's docstring says it exists to cover.
    session_id: str | None = None


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
    #: The child's error text when ``outcome == "failed"``. Kept so the roster
    #: can say WHY a child failed after its job row (which held ``error_text``)
    #: has been swept.
    error_text: str | None = None


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
        # Last: everything a waiter in ``_await_child`` needs must be in place
        # before it is allowed to proceed, or it would queue its question onto
        # a record whose buffered notes had not been flushed yet.
        if record.attached is not None:
            record.attached.set()

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

    def record_outcome(self, job_id: str, status: str, error_text: str | None = None) -> None:
        """Remember how a child settled, before its job row is swept.

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
        record = self._records.get(job_id)
        if record is None:
            return
        record.outcome = status
        record.error_text = error_text

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
            status = record.outcome
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
        resumable = status in ("completed", "failed", "cancelled", "paused", "gone")
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
            # Not started yet — parked behind the capacity gate, or still
            # building its session. Either way a course change has to reach a
            # child that has not read its prompt yet, so buffer it as a note.
            # It arrives before any work is done, which is the point.
            record.pending.append(self._to_child_message(text, expects_reply=False, steer=True))
            return Delivery(job_id, record.label, "queued")
        record.child.steer(self._format_to_child(text, expects_reply=False, steer=True))
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
        record = self._records.get(job_id)
        if record is None:
            return Reply(job_id, job_id, error=f"unknown subagent {job_id!r}")
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
                reason = (
                    self._gone_reason(record)
                    if record.settled or not self._is_running(record)
                    else self._not_started_reason(record)
                )
                return Reply(job_id, record.label, error=reason)
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
        record = self._records.get(job_id)
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
        record = self._records.get(job_id)
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

        record = self._records.get(job_id)
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
        new_job_id = run_subagent(
            label=record.label,
            prompt=message,
            parent_session=self._session,
            jobs_manager=jobs,
            resume_dir=record.session_dir,
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

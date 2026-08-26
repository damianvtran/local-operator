"""Async background job manager.

Async job manager (jobs + the owner-scoped delivery sink; the adaptive poll
ladder and delivery-retry backoff live host-side in this rewrite).

Semantics worth preserving exactly:

- **Queued jobs hold no execution slot.** ``at_capacity`` and ``register``
  both count only ``status == running and not queued`` jobs, so a large
  parked batch cannot starve registration.
- **Owned completions route exclusively through the owner's registered
  sink.** If the owner has no live sink the delivery is DEAD-LETTERED
  (dropped with a warning; the row keeps ``result_text`` until retention
  eviction) — it is never routed to the generic fallback, because that would
  leak one agent's result into another agent's session. Only genuinely
  unowned jobs use the fallback.
- ``cancel(job_id, owner_id)`` treats an owner mismatch as not-found, so a
  subagent teardown cannot cancel its parent's jobs.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from local_operator.harness.types import AbortSignal, Usage

logger = logging.getLogger(__name__)

DEFAULT_MAX_RUNNING_JOBS = 15
DEFAULT_RETENTION_MS = 5 * 60_000

#: Cap on a job's retained live-output tail (``AsyncJob.output_tail``). Sized
#: to hold a meaningful working window of a chatty job (a terraform plan, a
#: training loop's recent epochs) while staying far below the per-call result
#: budget, so even a full drain of the tail cannot dominate a peek's cost.
#: Bytes past this drop from the FRONT; ``output_seq`` still counts them, which
#: is what lets a peek report the loss instead of hiding it.
OUTPUT_TAIL_CHARS = 64_000

#: ``result_text`` stamped by :meth:`AsyncJobManager.cancel` on a job whose
#: runner was never entered, so the one fact the cancellation destroys — that
#: the job never ran — survives on the row every settled surface already reads.
#: Named rather than inlined because two renderers match on it: the panel row
#: prints it as the outcome, and the full-page view's title spends it instead of
#: the bare word ``cancelled`` beside a duration that was spent waiting.
CANCELLED_BEFORE_START = "cancelled before it started"

#: ``interrupted`` is a RESTORE-only status: it names a ``task`` row that was
#: ``running`` when the process exited, rehydrated from the persisted roster on
#: the next resume. It never arises from a live transition — a job the manager
#: itself settles becomes completed/failed/cancelled — so every LIVE reader
#: (capacity math, retention sweep, delivery) treats it as terminal. It exists
#: only so a resumed session can SHOW the child that was cut off mid-run and,
#: when its transcript survived on disk, offer to resume it (see
#: ``SubagentComms.roster``/``resume``), instead of the row vanishing with the
#: process.
JobStatus = Literal["running", "completed", "failed", "cancelled", "interrupted"]
JobType = Literal["bash", "task"]

# run(job_id, signal, report_progress) -> awaitable text result
JobRunFn = Callable[[str, AbortSignal, Callable[[str], None]], Awaitable[str | None]]
DeliverySink = Callable[[str, str, "AsyncJob | None"], Awaitable[None] | None]

#: Custom-message type used by a session to deliver a settled job's result
#: back into the conversation as a re-entering message (rendered as a user
#: message). Lives here because the job manager owns the lifecycle.
JOB_RESULT_MESSAGE_TYPE = "job_result"
ProgressFn = Callable[[str], None]


def _usage_components(usage: Usage | None, model_label: str | None) -> list[Usage]:
    """Detach priceable calls from one direct usage aggregate."""
    if usage is None:
        return []
    provider, _, model_id = (model_label or "").partition("/")
    source = usage.cost_components or [usage]
    components: list[Usage] = []
    for item in source:
        component = item.model_copy(deep=True)
        component.cost_components = []
        component.provider = component.provider or provider or None
        component.model_id = component.model_id or model_id or None
        components.append(component)
    return components


def _merge_accounting_component(
    grouped: dict[tuple[str | None, str | None, bool], Usage], component: Usage
) -> None:
    """Bound a subtree summary without erasing receipt-vs-estimate provenance."""
    has_receipt = component.usd_cost is not None
    key = (component.provider, component.model_id, has_receipt)
    total = grouped.get(key)
    if total is None:
        total = component.model_copy(deep=True)
        total.cost_components = []
        grouped[key] = total
        return
    total.input_tokens += component.input_tokens
    total.output_tokens += component.output_tokens
    total.cache_read_tokens += component.cache_read_tokens
    total.cache_write_tokens += component.cache_write_tokens
    total.reasoning_tokens += component.reasoning_tokens
    if component.context_tokens is not None:
        total.context_tokens = component.context_tokens
    if has_receipt:
        # The key prevents a receipt-backed call from absorbing estimated calls.
        total.usd_cost = (total.usd_cost or 0.0) + (component.usd_cost or 0.0)


class AsyncJob(BaseModel):
    """One registered background job. ``agent_id`` names the subagent the job
    RUNS (when applicable); ``owner_id`` names who registered it."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str
    type: JobType
    status: JobStatus = "running"
    start_time: float
    # Epoch seconds when the job's runner actually BEGAN, or ``None`` if it
    # never did. Distinct from ``start_time``, which is stamped at
    # REGISTRATION: ``register`` calls ``ensure_future``, which schedules the
    # runner without entering it, so between the two a job is admitted, counted
    # against capacity, and has not run a line. ``queued`` does not answer this
    # — it means "parked behind the gate", and the admitted-but-not-yet-entered
    # state is not parked — so anything that needs to know whether work
    # happened must read this and not that.
    started_at: float | None = None
    # Epoch seconds when the job SETTLED (completed/failed/cancelled).
    # Retention sweeps against this, not start_time, so a long-running job
    # stays observable for the full window after it finishes.
    settled_at: float | None = None
    label: str
    result_text: str | None = None
    error_text: str | None = None
    latest_details: dict[str, Any] | None = None
    # -- live output tail (``peek``) -----------------------------------------
    # Rolling tail of what a RUNNING job has emitted so far, so a caller can
    # observe progress without waiting for the job to settle. ``result_text``
    # cannot serve this: it is written once, at settle, which is exactly the
    # moment observation stops being useful for a job that runs for an hour.
    #
    # Two fields, because a peek must be able to report incrementally. The
    # buffer is a BOUNDED tail (oldest bytes drop past ``OUTPUT_TAIL_CHARS``),
    # while ``output_seq`` counts every char ever appended and never rewinds.
    # A reader that remembers the ``output_seq`` it last saw can therefore ask
    # "what is new since N?" and be told truthfully — including the case where
    # output scrolled past the tail between two peeks, which a length-only
    # cursor silently misreports as "nothing new".
    output_tail: str = ""
    output_seq: int = 0
    owner_id: str | None = None
    # Set when a caller (the `wait` tool) has already returned this job's
    # result to the model: auto-delivery must then stay quiet, or the same
    # result would reach the conversation twice.
    consumed: bool = False
    agent_id: str | None = None
    queued: bool = False
    # Serialized child events of a ``task`` job (``model_dump(mode="json")``
    # of each relayed AgentEvent), kept in memory for click-through trajectory
    # rendering. Bounded by the writer (``subagent.TRAJECTORY_CAP``) — the
    # runner appends one dict per child event and drops the oldest past the
    # cap, so this field can never grow a live session out of memory.
    # ``None`` (not []) on jobs without a trajectory: a host probing
    # ``getattr(job, "trajectory", None)`` must be able to tell "no child
    # events recorded" apart from "this job type has none".
    trajectory: list[dict[str, Any]] | None = None
    # The instruction a ``task`` job was launched with, verbatim. ``label`` is
    # a short summary the launcher wrote for a status line, so it cannot stand
    # in for this: a reader looking at a child's transcript otherwise sees an
    # agent working with no statement of what it was asked to do. Recorded
    # HERE because the prompt is not recoverable anywhere else a host can
    # reach — ``Session.prompt`` feeds it straight into the turn pipeline
    # without emitting an event, so it never reaches ``trajectory``.
    #
    # Unbounded on purpose, unlike ``trajectory``: this is one string the
    # launcher already holds for the life of the job, and it does not grow.
    # ``None`` (not "") on job types that have no prompt, same convention as
    # ``trajectory`` — a probing host must be able to tell "not recorded"
    # apart from "launched with an empty instruction".
    prompt: str | None = None
    # -- child accounting (``task`` jobs) ------------------------------------
    # Both default ``None``, never 0/"": a reader must be able to tell "not
    # recorded" from "recorded as nothing". Written by the subagent runner and
    # its relay (``harness/subagent.py``), read by the TUI's subagent panel and
    # by parent-side cost aggregation.
    #
    # The CHILD session's ``provider/model_id``, captured once when the child
    # is built. Read off the child, never off the parent: ``run_subagent``
    # takes a ``model_spec`` override, and a child running on a different
    # model is exactly the fact this records.
    model_label: str | None = None
    # Cumulative provider-reported usage for the child, summed over each
    # assistant ``message_end`` — not just the final one, because a tool-using
    # child spends most of its tokens in the earlier model calls of the same
    # run. ``context_tokens`` is point-in-time (the LAST reported value) and
    # is therefore replaced rather than summed.
    usage: Usage | None = None
    # Settled descendants, collapsed by serving identity and accounting mode.
    # ``usage`` remains SELF-only so row surfaces never mistake subtree tokens
    # for this child's context. Components are copied here at the ownership
    # boundary because the child manager is disposable and its local job ids are
    # not globally unique. Grouping receipt-backed calls separately from table-
    # priced calls preserves exact provider bills without letting one receipt
    # suppress estimates for its siblings.
    descendant_usage: list[Usage] = Field(default_factory=list)
    # A live edge exists only while the child can still mutate its own ledger.
    # It is runtime-only and deliberately cleared after the final settlement
    # snapshot, otherwise one retained observability row pins the disposed child
    # Session through its manager's bound completion callback.
    child_jobs: Any | None = Field(default=None, exclude=True)
    # The CHILD model's context window, so a reader can render usage as a
    # PERCENTAGE of what the child actually has. Captured at launch beside
    # ``model_label``, off the spec the child was already built with — never
    # resolved on a render path. The panel used to call ``resolve_model_info``
    # while painting at 12.5 fps; once registry rows gained
    # ``limits_from_listing`` a memo miss became a provider discovery fetch,
    # measured at 45 ms warm-disk, 222 ms cold and up to the 10 s discovery
    # timeout against a slow host — ten seconds of a TUI not reading the
    # keyboard, which no try/except can catch. It cannot change under a
    # running child, so once is right. ``0`` = not recorded.
    context_window: int = 0
    # Set on a row rehydrated from the persisted roster at resume (see
    # ``AsyncJobManager.restore``). A restored row is a HISTORICAL record, not a
    # live job: it has no asyncio task, no abort signal, and no runner, so it
    # can never be cancelled, delivered, or promoted. The retention sweep skips
    # it (it has no ``settled_at`` clock to age against and the resumed session
    # must keep showing it), and capacity math already ignores it because its
    # status is terminal. Kept as an explicit flag rather than inferred from a
    # missing task because a reader must be able to tell "this session started
    # it" from "a previous session did" without consulting the task table.
    restored: bool = False
    # The subagent ROLE this job runs ("task", "scout", ...) and the effort
    # TIER it was launched with ("lo"/"med"/"hi"), stamped at REGISTRATION
    # beside ``prompt`` — a queued job that never starts must still be able to
    # say what kind of child it is and at what effort, so neither can wait for
    # the runner. Both default ``None`` on the same convention as the fields
    # above: a reader must be able to tell "not recorded" (a job type with no
    # role/tier) from a recorded value. ``effort`` is recorded independently of
    # ``model_label`` on purpose: the model is resolved from the tier upstream
    # (``Session._resolve_subagent_model``) and a child on a DIFFERENT model
    # than the parent still ran at a known tier — which is exactly the case the
    # status band needs so it can name the level beside a model whose own
    # ladder it cannot see.
    agent_role: str | None = None
    effort: str | None = None


class AsyncJobManager:
    """Registers and tracks background jobs with owner-scoped delivery.

    A completion delivers exactly once: through the owner's sink when one is
    registered, otherwise through the ``on_job_complete`` fallback for jobs
    without an owner. Owned jobs with no live sink are dead-lettered.
    """

    def __init__(
        self,
        *,
        max_running: int = DEFAULT_MAX_RUNNING_JOBS,
        retention_ms: int = DEFAULT_RETENTION_MS,
        on_job_complete: (
            Callable[[str, str, "AsyncJob | None"], Awaitable[None] | None] | None
        ) = None,
        on_roster_change: Callable[[], None] | None = None,
        on_job_change: Callable[[], None] | None = None,
    ) -> None:
        self._max_running = max_running
        self._retention_ms = retention_ms
        self._on_job_complete = on_job_complete
        # Fired (best-effort, synchronously) whenever the SET of ``task`` rows
        # or one of their statuses changes, so the session can re-snapshot the
        # roster to disk without the manager knowing what a transcript is. Kept
        # as a bare callback rather than a persist coroutine because the manager
        # runs it on the hot path of every registration and settle: it signals
        # "something changed", and the owner decides how (and how cheaply) to
        # persist. A raising callback must never break job bookkeeping, so the
        # single call site guards it.
        self._on_roster_change = on_roster_change
        self._on_job_change = on_job_change
        self._jobs: dict[str, AsyncJob] = {}
        # Terminal rows hand their subtree into this bounded accumulator before
        # retention can remove them. The owning parent runner later copies this
        # snapshot onto its AsyncJob; polling never participates in durability.
        self._settled_accounting: dict[tuple[str | None, str | None, bool], Usage] = {}
        # The status band reads this ledger every second. Cache the bounded
        # aggregate and propagate invalidations through live manager edges so
        # unchanged reads never recurse through the subagent tree.
        self._accounting_revision = 0
        self._accounting_cache_revision = -1
        self._accounting_cache: tuple[Usage, ...] = ()
        self._accounting_listeners: set[Callable[[set[int]], None]] = set()
        self._child_accounting_unsubscribes: dict[str, Callable[[], None]] = {}
        self._signals: dict[str, AbortSignal] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._sinks: dict[str, DeliverySink] = {}
        self._queued_runners: dict[str, JobRunFn] = {}
        # Teardown for a job whose runner has NOT been entered yet. See
        # ``register(on_cancel=...)``: a runner's own ``finally`` cannot clean
        # up a coroutine that never ran, so ownership of an already-spawned
        # resource lives here until the runner takes it over.
        self._pending_cleanups: dict[str, Callable[[], None]] = {}
        # One event per job, set exactly once when the job settles. This is
        # what lets ``wait`` sleep until there is news instead of re-checking
        # a status field on a timer: a 50 ms poll loop wakes 6000 times over a
        # five-minute wait, and every one of those wakeups runs on the SAME
        # event loop that serves the parent turn, every sibling subagent, and
        # the TUI repaint. Created lazily (most jobs are never waited on) and
        # dropped by the retention sweep with the job row.
        self._settled_events: dict[str, asyncio.Event] = {}

    # -- queries ------------------------------------------------------------

    def get(self, job_id: str, *, owner_id: str | None = None) -> AsyncJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if owner_id is not None and job.owner_id != owner_id:
            return None  # scoping: mismatch is not-found
        return job

    def list(self, *, owner_id: str | None = None) -> list[AsyncJob]:
        jobs = list(self._jobs.values())
        if owner_id is not None:
            jobs = [job for job in jobs if job.owner_id == owner_id]
        return sorted(jobs, key=lambda job: job.start_time)

    def at_capacity(self) -> bool:
        running = sum(
            1 for job in self._jobs.values() if job.status == "running" and not job.queued
        )
        return running >= self._max_running

    def accounting_components(self) -> list[Usage]:
        """A detached, provenance-preserving snapshot of this whole ledger.

        Job ids are manager-local, so accounting never flattens rows by id.
        Components instead collapse by the facts that control pricing: serving
        provider/model and whether an authoritative receipt exists. This keeps
        the summary proportional to distinct billing routes rather than child
        fan-out or model-call count, while retaining every token and dollar.
        """
        if self._accounting_cache_revision != self._accounting_revision:
            rebuilt = self._collect_accounting_components(set())
            self._accounting_cache = tuple(component.model_copy(deep=True) for component in rebuilt)
            self._accounting_cache_revision = self._accounting_revision
        return [component.model_copy(deep=True) for component in self._accounting_cache]

    def _collect_accounting_components(self, seen: set[int]) -> list[Usage]:
        identity = id(self)
        if identity in seen:
            return []
        seen.add(identity)
        grouped = {
            key: component.model_copy(deep=True)
            for key, component in self._settled_accounting.items()
        }
        for job in self._jobs.values():
            # Terminal rows are already in ``_settled_accounting``. Counting
            # retained rows again would make totals depend on retention length.
            if job.type != "task" or job.status != "running":
                continue
            components = [*_usage_components(job.usage, job.model_label), *job.descendant_usage]
            child_manager = job.child_jobs
            if isinstance(child_manager, AsyncJobManager):
                components.extend(child_manager._collect_accounting_components(seen))
            for component in components:
                _merge_accounting_component(grouped, component)
        return [component.model_copy(deep=True) for component in grouped.values()]

    def _invalidate_accounting(self, seen: set[int] | None = None) -> None:
        """Invalidate this aggregate and notify parents once, tolerating cycles."""
        visited = seen if seen is not None else set()
        identity = id(self)
        if identity in visited:
            return
        visited.add(identity)
        self._accounting_revision += 1
        for listener in tuple(self._accounting_listeners):
            listener(visited)

    def subscribe_accounting(self, listener: Callable[[set[int]], None]) -> Callable[[], None]:
        self._accounting_listeners.add(listener)

        def unsubscribe() -> None:
            self._accounting_listeners.discard(listener)

        return unsubscribe

    def attach_child_manager(self, job_id: str, child: "AsyncJobManager") -> None:
        """Attach the live accounting lease and propagate child mutations."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        unsubscribe = self._child_accounting_unsubscribes.pop(job_id, None)
        if unsubscribe is not None:
            unsubscribe()
        job.child_jobs = child
        self._child_accounting_unsubscribes[job_id] = child.subscribe_accounting(
            self._invalidate_accounting
        )
        self._invalidate_accounting()

    def detach_child_manager(self, job_id: str, descendant_usage: list[Usage]) -> None:
        """Replace a live child edge with its final detached durable ledger."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        unsubscribe = self._child_accounting_unsubscribes.pop(job_id, None)
        if unsubscribe is not None:
            unsubscribe()
        job.descendant_usage = [item.model_copy(deep=True) for item in descendant_usage]
        job.child_jobs = None
        self._invalidate_accounting()

    def note_usage_changed(self) -> None:
        """Invalidate after an in-place Usage mutation owned by a child relay."""
        self._invalidate_accounting()

    def _notify_roster_change(self) -> None:
        """Signal a task-roster mutation to persistence and observers."""
        self._notify_job_change(task_roster=True)

    def _notify_job_change(self, *, task_roster: bool) -> None:
        """Publish every job mutation while persisting only resumable tasks."""
        callbacks = ([self._on_roster_change] if task_roster else []) + [self._on_job_change]
        for callback in callbacks:
            if callback is None:
                continue
            try:
                callback()
            except Exception:  # noqa: BLE001 - observation must not break jobs
                logger.warning("job-change listener raised", exc_info=True)

    def restore(self, rows: list["AsyncJob"]) -> None:
        """Rehydrate task rows from a persisted roster at resume.

        The manager's ``_jobs`` table lives only in memory, so a resumed
        session opens with an empty subagent panel even though the children ran
        and their transcripts survive on disk. This re-seeds the table from the
        snapshot the session persisted (see ``Session._persist_subagent_roster``)
        so the panel, the ``jobs`` tool, and ``hub op='list'`` show the children
        the previous process launched.

        A row that was actually RUNNING when the process died is downgraded to
        ``interrupted`` here: its asyncio task is gone, so it is not live, and
        presenting it as ``running`` would spin a status the manager can never
        settle. A row that was still ``queued`` (parked behind the capacity
        gate, ``status == "running"`` with ``queued == True``) never ran and has
        no transcript, so it is NOT interrupted — it is simply gone; it is
        dropped rather than restored, matching the comms side, which already
        skips it in ``snapshot()`` because its record has no ``session_dir``.
        Every restored row is flagged ``restored`` and carries no runtime
        handles (an ``AsyncJob`` serializes none — the abort signal and asyncio
        task live in the manager's own ``_signals``/``_tasks`` maps, which the
        snapshot never touched — so there is nothing to clear). Rows are NOT
        re-run; resuming one is an explicit ``hub op='resume'`` that starts a
        fresh job against the old transcript.

        Idempotent-ish: a row whose id is already present (a live job of this
        session) is left untouched, so a mis-timed double restore cannot
        clobber a running child.

        Deliberately does NOT fire ``on_roster_change``: rehydrating the table
        is not a roster *change* the owner needs to persist — the snapshot being
        read is already on disk — and notifying here would re-append a
        byte-identical snapshot on every resume (and, if a host ever constructs
        a Session off-loop, raise a spurious warning from the persist spawn).
        """
        for row in rows:
            if row.id in self._jobs:
                continue
            if row.status == "running":
                if row.queued:
                    # Parked and never started, so it has no transcript to show
                    # or resume; a ``⇥ interrupted`` row for it would invite a
                    # resume that finds nothing. Drop it entirely.
                    continue
                # No task backs it any more; a live-looking row would spin a
                # spinner forever and invite a cancel that finds nothing.
                row.status = "interrupted"
            row.restored = True
            self._jobs[row.id] = row
            if row.status != "running":
                self._record_settled_accounting(row)
        self._invalidate_accounting()

    # -- registration -------------------------------------------------------

    def register(
        self,
        type: JobType,
        label: str,
        run: JobRunFn,
        *,
        owner_id: str | None = None,
        agent_id: str | None = None,
        queued: bool = False,
        on_cancel: Callable[[], None] | None = None,
    ) -> str:
        """Register and start a background job. Returns the job id.

        ``queued`` jobs are parked behind a caller-managed gate: they hold no
        execution slot and are not started here — the run coroutine must call
        the job (the manager merely tracks it) via ``start_queued`` once its
        gate opens.

        ``on_cancel`` is teardown for a resource the CALLER already created
        before registering — a spawned process, an open kernel. It exists
        because ``register`` only ``ensure_future``s the runner: the coroutine
        is scheduled, not entered, so a ``cancel``/``dispose`` landing in the
        same event-loop turn never executes the runner body and never reaches
        its ``finally``. The resource would then outlive the job row that was
        supposed to own it, reparented to init with nothing holding a
        reference. It fires ONLY while the runner has not started; the first
        statement of ``_run_job`` hands ownership over, after which the
        runner's own cleanup is authoritative and this is dropped.
        """
        # Capacity is checked BEFORE any row is inserted: a rejected register
        # must never leave a phantom job occupying or shadowing a slot.
        if not queued and self.at_capacity():
            raise RuntimeError(f"at most {self._max_running} background jobs may run concurrently")
        job_id = uuid.uuid4().hex[:12]
        job = AsyncJob(
            id=job_id,
            type=type,
            label=label,
            start_time=time.time(),
            owner_id=owner_id,
            agent_id=agent_id,
            queued=queued,
        )
        signal = AbortSignal()
        self._jobs[job_id] = job
        self._signals[job_id] = signal
        if on_cancel is not None:
            self._pending_cleanups[job_id] = on_cancel

        if not queued:
            progress = self._progress_fn(job_id)
            coro = run(job_id, signal, progress)
            task = asyncio.ensure_future(self._run_job(job, coro))
            self._tasks[job_id] = task
        else:
            # Parked behind a caller-managed gate; holds no execution slot and
            # keeps its runner for start_queued().
            self._queued_runners[job_id] = run
        # A new row is a roster change the owner may want to persist (task rows
        # only carry a resumable transcript, but the listener filters that).
        if type == "task":
            self._invalidate_accounting()
        self._notify_job_change(task_roster=type == "task")
        return job_id

    def start_queued(self, job_id: str) -> bool:
        """Promote a parked ``queued`` job to running (gate opened)."""
        runner = self._queued_runners.get(job_id)
        job = self._jobs.get(job_id)
        if runner is None or job is None or job.status != "running" or not job.queued:
            return False
        # register() checks capacity on admission; this second entry path must
        # too, or promoting a parked job at a full manager yields 16 running.
        if self.at_capacity():
            return False
        job.queued = False
        del self._queued_runners[job_id]
        signal = self._signals[job_id]
        progress = self._progress_fn(job_id)
        task = asyncio.ensure_future(self._run_job(job, runner(job_id, signal, progress)))
        self._tasks[job_id] = task
        # The queued -> running transition is a status-affecting task-row change
        # like register/settle/cancel, so it notifies too: without this the flag
        # move only reaches disk at the next roster event, and a snapshot taken
        # in between would restore the row as if it were still parked.
        if job.type == "task":
            self._invalidate_accounting()
        self._notify_job_change(task_roster=job.type == "task")
        return True

    # -- delivery -----------------------------------------------------------

    def mark_consumed(self, job_id: str) -> None:
        """Flag a job's result as already handed to the model (see the
        ``consumed`` field): auto-delivery checks it and stays quiet."""
        job = self._jobs.get(job_id)
        if job is not None:
            job.consumed = True

    def register_delivery_sink(self, owner_id: str, sink: DeliverySink) -> Callable[[], None]:
        """Route completions owned by ``owner_id`` to ``sink``. Returns an
        unregister callable."""
        self._sinks[owner_id] = sink

        def _unregister() -> None:
            if self._sinks.get(owner_id) is sink:
                del self._sinks[owner_id]

        return _unregister

    # -- cancellation -------------------------------------------------------

    async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool:
        """Cancel a job. An owner mismatch is treated as not-found so a
        subagent teardown cannot cancel its parent's jobs."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if owner_id is not None and job.owner_id != owner_id:
            return False
        if job.status != "running":
            return False
        job.status = "cancelled"
        # A job cancelled while still parked stops being queued: ``queued`` is
        # "waiting for a slot", and a cancelled job is waiting for nothing. Left
        # set, every reader that branches on it reported the job as still
        # pending — the panel painted ``⏳ queued`` on a row whose status was
        # ``cancelled``, which reads as work that is about to start. Its runner
        # is dropped for the same reason: ``start_queued`` refuses a non-running
        # job, so the entry could never run again and only kept the closure
        # (prompt, parent session, model spec) alive for the life of the manager.
        #
        # Cancelling also erases the record that the job never ran, and every
        # surface then presents its WAITING time as work time: the panel row and
        # the page title both measure ``settled_at - start_time``, which for a
        # job that never began is how long it sat, printed in the column where
        # every other row's number is time a child spent working. So the fact
        # moves to the carrier the settled-row paths already read.
        #
        # Keyed on ``started_at``, NOT on ``queued``. ``queued`` means "parked
        # at the capacity gate", which is only one of three ways a job reaches
        # here without its runner ever being entered: a job admitted with
        # ``queued=False`` is merely SCHEDULED (``register`` calls
        # ``ensure_future``, which does not enter the coroutine), and a job
        # promoted by ``start_queued`` has ``queued`` cleared before its runner
        # runs. Both did no work, and the first is the state this fix was
        # written for — the ledger that motivated it recorded
        # ``at_capacity: False`` with nothing parked, so keying on ``queued``
        # would have stamped none of the rows in that incident.
        #
        # A genuinely running job is left alone for a different reason: it may
        # be mid-``_run_job`` and owns ``result_text``. ``started_at is None``
        # excludes it by construction.
        if job.started_at is None and job.result_text is None:
            job.result_text = CANCELLED_BEFORE_START
        job.queued = False
        self._queued_runners.pop(job_id, None)
        signal = self._signals.get(job_id)
        if signal is not None:
            signal.abort("cancelled")
        task = self._tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Two sources of CancelledError here. Awaiting a task we
                # cancelled that never got to run (cancelled before it
                # started) raises it as the NORMAL outcome — the job row is
                # already marked cancelled, so swallow that. But if the
                # CALLER of cancel() was itself cancelled mid-await, this
                # task's own cancelling count is nonzero and the error MUST
                # propagate, never be swallowed into a clean return (HC-16).
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    raise
            except Exception:
                logger.warning("job %s task raised on cancel", job_id, exc_info=True)
        # Teardown for a runner that was never entered. Awaiting the cancelled
        # task above does NOT run the coroutine body, so a resource the caller
        # spawned before registering (a process group, a kernel) still has no
        # owner at this point; without this it survives the job row that was
        # meant to own it. Popped first so it can only ever fire once, and
        # AFTER the task await so a runner that did start has already claimed
        # ownership and removed it — the two paths cannot both kill.
        cleanup = self._pending_cleanups.pop(job_id, None)
        if cleanup is not None:
            try:
                cleanup()
            except Exception:  # noqa: BLE001 — teardown must not fail a cancel
                logger.warning("job %s pre-start cleanup raised", job_id, exc_info=True)
        self._settle(job)
        self._sweep_due()
        if job.type == "task":
            self._invalidate_accounting()
        self._notify_job_change(task_roster=job.type == "task")
        return True

    # -- lifecycle ----------------------------------------------------------

    async def dispose(self) -> None:
        """Cancel every running job and drop runtime state.

        Job ROWS are kept: they are observability records whose eviction is
        owned by the retention sweep (settled_at based), not by teardown.
        Callers may still ``get``/``list`` after dispose; nothing runs.
        """
        job_ids = [job_id for job_id, job in self._jobs.items() if job.status == "running"]
        for job_id in job_ids:
            await self.cancel(job_id)
        self._signals.clear()
        self._tasks.clear()
        self._sinks.clear()
        self._queued_runners.clear()
        # Any cleanup still pending belongs to a job that was not ``running``
        # (so ``cancel`` above skipped it) yet never entered its runner. Firing
        # them here is what makes dispose a complete teardown rather than one
        # that leaks whatever those jobs had already spawned.
        for job_id, cleanup in list(self._pending_cleanups.items()):
            del self._pending_cleanups[job_id]
            try:
                cleanup()
            except Exception:  # noqa: BLE001 — teardown must not fail a dispose
                logger.warning("job %s pre-start cleanup raised", job_id, exc_info=True)
        # Wake anything still parked on a job before dropping the events: a
        # waiter that outlives dispose must observe "settled" (cancel() set
        # each event on its way through) rather than sleeping to its deadline
        # against a manager that will never run again.
        for event in self._settled_events.values():
            event.set()

    # -- live output ---------------------------------------------------------

    def append_output(self, job_id: str, text: str) -> None:
        """Append live output to a job's rolling tail.

        Called from a job's own runner as bytes arrive. Unknown ids are
        ignored rather than raising: a runner draining a pipe must not be able
        to kill the job it is reporting for because the row was already swept
        by retention.
        """
        job = self._jobs.get(job_id)
        if job is None or not text:
            return
        job.output_seq += len(text)
        combined = job.output_tail + text
        # Drop from the FRONT past the cap: for a job that is still running the
        # recent end is the informative one (the current step, the error that
        # just landed), while the opening lines have usually already been read.
        job.output_tail = combined[-OUTPUT_TAIL_CHARS:]
        self._notify_job_change(task_roster=job.type == "task")

    def read_output(self, job_id: str, since: int = 0) -> tuple[str, int, bool] | None:
        """``(text, seq, gap)`` for a peek, or ``None`` when the job is unknown.

        ``since`` is an ``output_seq`` value from a previous peek. The return
        is only what was appended after it, so polling a quiet job costs
        almost nothing and a caller's context grows by what is genuinely new
        rather than re-receiving the whole tail every time.

        ``gap`` reports that output between ``since`` and the returned text
        was evicted from the bounded tail before this peek read it — the
        caller has an incomplete record and is told so rather than being
        handed a contiguous-looking excerpt that silently skips a step.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        seq = job.output_seq
        if since >= seq:
            return "", seq, False
        available = len(job.output_tail)
        # How many chars back from the head the caller asked to resume.
        wanted = seq - max(since, 0)
        if wanted <= available:
            return job.output_tail[-wanted:], seq, False
        return job.output_tail, seq, True

    # -- internals ----------------------------------------------------------

    def _progress_fn(self, job_id: str) -> ProgressFn:
        def report(details: str) -> None:
            job = self._jobs.get(job_id)
            if job is not None:
                job.latest_details = {"progress": details}
                self._notify_job_change(task_roster=job.type == "task")

        return report

    async def _run_job(self, job: AsyncJob, coro: Awaitable[str | None]) -> None:
        # FIRST statement, before any await: this is the fact "the runner
        # actually began", and everything that asks whether a job did work
        # reads it. Stamping it later would leave a window where the coroutine
        # is running and the row still says it never started.
        job.started_at = time.time()
        # The runner is now entered, so its own teardown (its ``finally``, its
        # CancelledError handler) is authoritative for the resources it owns.
        # Dropping the pre-start cleanup here is what keeps a cancel from
        # killing the same process twice through two different owners.
        self._pending_cleanups.pop(job.id, None)
        try:
            result = await coro
            if job.status == "cancelled":
                return
            job.status = "completed"
            job.result_text = result if result is not None else ""
        except asyncio.CancelledError:
            job.status = "cancelled"
            self._settle(job)
            self._tasks.pop(job.id, None)
            self._sweep_due()
            self._notify_job_change(task_roster=job.type == "task")
            return
        except Exception as exc:
            job.status = "failed"
            job.error_text = str(exc)
            logger.warning("background job %s failed", job.id, exc_info=True)
        self._settle(job)
        self._tasks.pop(job.id, None)
        # After the settle stamp, before delivery: a listener re-reading the
        # roster now sees the terminal status, and persisting here (rather than
        # only after delivery) means a crash between settle and delivery still
        # leaves the outcome on disk for the next resume.
        if job.type == "task":
            self._invalidate_accounting()
        self._notify_job_change(task_roster=job.type == "task")
        try:
            await self._deliver(job)
        except Exception:
            # A raising sink must not become an unobserved task exception nor
            # skip the retention sweep (the task handle was already popped).
            logger.warning("delivery sink raised for job %s", job.id, exc_info=True)
        finally:
            self._sweep_due()
            # The slot this job just freed belongs to the longest-waiting
            # parked job. Without this, a ``queued`` job sat forever as
            # ``running + queued`` (occupying neither a slot nor a clear
            # status): the task tool answered "waiting", the wait tool timed
            # out on "still running", and the band painted it as done. Order
            # is FIFO so the oldest request runs first.
            self._promote_oldest_queued()

    def _promote_oldest_queued(self) -> None:
        """Start the longest-waiting ``queued`` job now that a slot is free.

        Called after a running job settles (its slot is free). Idempotent and
        safe to call from anywhere: promotes at most one job, and only when
        the manager is under capacity — so a burst of completions promotes a
        matching burst of parked jobs without ever exceeding
        ``self._max_running``.
        """
        for job_id in self.queued_ids():
            if self.at_capacity():
                return
            self.start_queued(job_id)

    async def _deliver(self, job: AsyncJob) -> None:
        # ``_run_job`` always stores a string on completion, so the ``or ""``
        # is belt-and-braces: it keeps the sink's ``str`` contract total even
        # if a row is ever built or replayed without one.
        text = (job.result_text or "") if job.status == "completed" else (job.error_text or "")
        if job.owner_id is not None:
            sink = self._sinks.get(job.owner_id)
            if sink is None:
                # Dead-letter: never route an owned job to the fallback sink,
                # that would leak one agent's result into another's session.
                logger.warning(
                    "dead-lettering job %s: no live sink for owner %s", job.id, job.owner_id
                )
                return
            await self._maybe_await(sink(job.id, text, job))
            return
        if self._on_job_complete is not None:
            await self._maybe_await(self._on_job_complete(job.id, text, job))

    def settled_event(self, job_id: str) -> asyncio.Event:
        """The event set when ``job_id`` settles, created on first request.

        Pre-set for a job that has ALREADY settled (including one this manager
        has never heard of, which cannot ever settle): a waiter that arrives
        after the transition must not block forever on news that already
        happened. This is the race the poll loop hid by re-reading status.
        """

        job = self._jobs.get(job_id)
        if job is None:
            # No row: nothing will ever settle this id, so hand back a pre-set
            # event WITHOUT storing it. Storing one would strand an entry that
            # no cleanup path can reach — ``_sweep_due`` only pops ids it finds
            # in ``self._jobs``, and this id is by definition not one of them.
            settled = asyncio.Event()
            settled.set()
            return settled
        event = self._settled_events.get(job_id)
        if event is None:
            event = asyncio.Event()
            self._settled_events[job_id] = event
        if job.status != "running":
            event.set()
        return event

    def _settle(self, job: AsyncJob) -> None:
        """Record the settle timestamp on the status transition (idempotent).

        Also wakes every waiter. This is the ONE place a job becomes settled,
        which is why the notification belongs here rather than beside each of
        the three call sites: a future fourth transition would otherwise leave
        its waiters asleep until their deadline.
        """
        if job.settled_at is None:
            job.settled_at = time.time()
            # Same transition as the settle stamp: after this line retention may
            # erase the row immediately without erasing its financial history.
            self._record_settled_accounting(job)
        event = self._settled_events.get(job.id)
        if event is not None:
            event.set()

    def _record_settled_accounting(self, job: AsyncJob) -> None:
        """Transfer one terminal subtree exactly once into manager ownership."""
        components = [*_usage_components(job.usage, job.model_label), *job.descendant_usage]
        child_manager = job.child_jobs
        if isinstance(child_manager, AsyncJobManager):
            components.extend(child_manager.accounting_components())
        for component in components:
            _merge_accounting_component(self._settled_accounting, component)
        unsubscribe = self._child_accounting_unsubscribes.pop(job.id, None)
        if unsubscribe is not None:
            unsubscribe()
        self._invalidate_accounting()

    def _sweep_due(self) -> None:
        """Drop settled jobs older than the retention window.

        Retention runs from ``settled_at`` (settle time), never
        ``start_time``: a job that ran longer than the window must still be
        observable for the full retention period after it settles.
        """
        cutoff = time.time() - self._retention_ms / 1000.0
        for job_id in [
            job_id
            for job_id, job in self._jobs.items()
            # ``restored`` rows are exempt: they carry the PREVIOUS session's
            # settle stamp, which is almost always already past the retention
            # window, so an unguarded sweep would evict every rehydrated child
            # on the first pass after resume — the exact rows this feature
            # exists to keep visible. They leave only when the session ends.
            if not job.restored
            and job.status != "running"
            and job.settled_at is not None
            and job.settled_at < cutoff
        ]:
            del self._jobs[job_id]
            self._signals.pop(job_id, None)
            self._tasks.pop(job_id, None)
            self._settled_events.pop(job_id, None)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    # -- capacity bookkeeping ------------------------------------------------

    def queued_ids(self) -> list[str]:
        """Ids of jobs parked by ``register(..., queued=True)`` that have not
        been promoted yet, oldest first.

        A queued row holds no execution slot; promotion is automatic — the
        manager starts the longest-waiting parked job via
        ``_promote_oldest_queued`` whenever a running job settles and frees a
        slot (see ``_run_job``'s finalizer). This accessor is the bookkeeping
        half of that contract: the caller asks "who is waiting" exactly at a
        settle, and an empty list means nothing is owed.
        """
        waiting = [
            job
            for job in self._jobs.values()
            if job.status == "running" and job.queued and job.id in self._queued_runners
        ]
        return [job.id for job in sorted(waiting, key=lambda job: job.start_time)]

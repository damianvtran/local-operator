"""Async background job manager.

Port of the load-bearing parts of omp ``async/job-manager.ts`` (jobs + the
owner-scoped delivery sink; the adaptive poll ladder and delivery-retry
backoff live host-side in this rewrite).

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

from pydantic import BaseModel, ConfigDict

from local_operator.harness.types import AbortSignal

logger = logging.getLogger(__name__)

DEFAULT_MAX_RUNNING_JOBS = 15
DEFAULT_RETENTION_MS = 5 * 60_000

JobStatus = Literal["running", "completed", "failed", "cancelled"]
JobType = Literal["bash", "task"]

# run(job_id, signal, report_progress) -> awaitable text result
JobRunFn = Callable[[str, AbortSignal, Callable[[str], None]], Awaitable[str | None]]
DeliverySink = Callable[[str, str, "AsyncJob | None"], Awaitable[None] | None]
ProgressFn = Callable[[str], None]


class AsyncJob(BaseModel):
    """One registered background job. ``agent_id`` names the subagent the job
    RUNS (when applicable); ``owner_id`` names who registered it."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str
    type: JobType
    status: JobStatus = "running"
    start_time: float
    label: str
    result_text: str | None = None
    error_text: str | None = None
    latest_details: dict[str, Any] | None = None
    owner_id: str | None = None
    agent_id: str | None = None
    queued: bool = False


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
        on_job_complete: Callable[[str, str, "AsyncJob | None"], Awaitable[None] | None] | None = None,
    ) -> None:
        self._max_running = max_running
        self._retention_ms = retention_ms
        self._on_job_complete = on_job_complete
        self._jobs: dict[str, AsyncJob] = {}
        self._signals: dict[str, AbortSignal] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._sinks: dict[str, DeliverySink] = {}
        self._queued_runners: dict[str, JobRunFn] = {}

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
    ) -> str:
        """Register and start a background job. Returns the job id.

        ``queued`` jobs are parked behind a caller-managed gate: they hold no
        execution slot and are not started here — the run coroutine must call
        the job (the manager merely tracks it) via ``start_queued`` once its
        gate opens.
        """
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

        if not queued:
            if self.at_capacity():
                raise RuntimeError(
                    f"at most {self._max_running} background jobs may run concurrently"
                )
            progress = self._progress_fn(job_id)
            coro = run(job_id, signal, progress)
            task = asyncio.ensure_future(self._run_job(job, coro))
            self._tasks[job_id] = task
        else:
            # Parked behind a caller-managed gate; holds no execution slot and
            # keeps its runner for start_queued().
            self._queued_runners[job_id] = run
        return job_id

    def start_queued(self, job_id: str) -> bool:
        """Promote a parked ``queued`` job to running (gate opened)."""
        runner = self._queued_runners.get(job_id)
        job = self._jobs.get(job_id)
        if runner is None or job is None or job.status != "running" or not job.queued:
            return False
        job.queued = False
        del self._queued_runners[job_id]
        signal = self._signals[job_id]
        progress = self._progress_fn(job_id)
        task = asyncio.ensure_future(self._run_job(job, runner(job_id, signal, progress)))
        self._tasks[job_id] = task
        return True

    # -- delivery -----------------------------------------------------------

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
        signal = self._signals.get(job_id)
        if signal is not None:
            signal.abort("cancelled")
        task = self._tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._sweep_due()
        return True

    # -- lifecycle ----------------------------------------------------------

    async def dispose(self) -> None:
        """Cancel every running job and clear the registry."""
        job_ids = [job_id for job_id, job in self._jobs.items() if job.status == "running"]
        for job_id in job_ids:
            await self.cancel(job_id)
        self._jobs.clear()
        self._signals.clear()
        self._tasks.clear()
        self._sinks.clear()
        self._queued_runners.clear()

    # -- internals ----------------------------------------------------------

    def _progress_fn(self, job_id: str) -> ProgressFn:
        def report(details: str) -> None:
            job = self._jobs.get(job_id)
            if job is not None:
                job.latest_details = {"progress": details}

        return report

    async def _run_job(self, job: AsyncJob, coro: Awaitable[str | None]) -> None:
        try:
            result = await coro
            if job.status == "cancelled":
                return
            job.status = "completed"
            job.result_text = result if result is not None else ""
        except asyncio.CancelledError:
            job.status = "cancelled"
            self._sweep_due()
            return
        except Exception as exc:
            job.status = "failed"
            job.error_text = str(exc)
            logger.warning("background job %s failed", job.id, exc_info=True)
        self._tasks.pop(job.id, None)
        await self._deliver(job)
        self._sweep_due()

    async def _deliver(self, job: AsyncJob) -> None:
        text = job.result_text if job.status == "completed" else (job.error_text or "")
        if job.owner_id is not None:
            sink = self._sinks.get(job.owner_id)
            if sink is None:
                # Dead-letter: never route an owned job to the fallback sink,
                # that would leak one agent's result into another's session.
                logger.warning("dead-lettering job %s: no live sink for owner %s", job.id, job.owner_id)
                return
            await self._maybe_await(sink(job.id, text, job))
            return
        if self._on_job_complete is not None:
            await self._maybe_await(self._on_job_complete(job.id, text, job))

    def _sweep_due(self) -> None:
        """Drop settled jobs older than the retention window."""
        cutoff = time.time() - self._retention_ms / 1000.0
        for job_id in [
            job_id
            for job_id, job in self._jobs.items()
            if job.status != "running" and job.start_time < cutoff
        ]:
            del self._jobs[job_id]
            self._signals.pop(job_id, None)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

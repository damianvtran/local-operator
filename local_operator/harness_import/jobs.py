"""One process-local import-job registry shared by every surface.

The desktop routes used to own their own ``_JOBS`` map and the agent tool would
have needed a second one. Two registries would disagree about what is running:
the service's file lock refuses a concurrent import, but a second surface would
still advertise a job id whose progress nobody updates. This module is the one
owner of job state; the routes and the tool are thin adapters over it.

Jobs are deliberately bounded and ephemeral. They survive no restart, which is
safe because the import manifest makes a re-run a no-op or a resume. The service
itself still holds the cross-process lock; this registry only prevents a second
job from being started in THIS process.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from typing import Any

from local_operator.harness_import.types import (
    ImportProgress,
    ImportRequest,
    ImportResult,
)

MAX_JOBS = 20


class ImportJobBusy(RuntimeError):
    """A job is already running in this process."""


class ImportJob:
    """One accepted import: its live progress, result, task and owning service.

    ``service`` is the instance whose cancellation event the running task
    observes. A caller must never pass a freshly constructed service to
    ``cancel``: ``HarnessImportService.cancel`` flips an event on ``self``, so a
    throwaway instance reports success while the real import keeps running.
    """

    __slots__ = ("job_id", "session_key", "progress", "result", "service", "request", "task")

    def __init__(self, job_id: str, progress: ImportProgress, *, session_key: str = ""):
        self.job_id = job_id
        self.session_key = session_key
        self.progress = progress
        self.result: ImportResult | None = None
        self.service: Any = None
        #: The accepted request, retained so a page that reattaches to a job
        #: (``/import status``) can paint its selected session rows without
        #: re-reading the source or guessing from progress alone.
        self.request: ImportRequest | None = None
        self.task: asyncio.Task[Any] | None = None


class ImportJobRegistry:
    """Bounded newest-first registry; one running job per process."""

    def __init__(self, max_jobs: int = MAX_JOBS):
        self.max_jobs = max_jobs
        self.jobs: OrderedDict[str, ImportJob] = OrderedDict()

    def running(self) -> ImportJob | None:
        for job in self.jobs.values():
            if job.task is not None and not job.task.done():
                return job
        return None

    def get(self, job_id: str, *, session_key: str | None = None) -> ImportJob | None:
        job = self.jobs.get(job_id)
        if job is not None and session_key is not None and job.session_key != session_key:
            return None
        return job

    def latest(self, *, session_key: str | None = None) -> ImportJob | None:
        for job in reversed(self.jobs.values()):
            if session_key is None or job.session_key == session_key:
                return job
        return None

    def start(
        self,
        service: Any,
        request: ImportRequest,
        *,
        job_id: str | None = None,
        session_key: str = "",
    ) -> ImportJob:
        """Accept a job and run it on the current event loop."""
        if self.running() is not None:
            raise ImportJobBusy("an import job is already running")
        progress = ImportProgress(
            job_id=job_id or f"imp-{uuid.uuid4().hex[:12]}",
            state="queued",
            phase="queued",
            started_at=time.time(),
            updated_at=time.time(),
        )
        job = ImportJob(progress.job_id, progress, session_key=session_key)
        job.service = service
        job.request = request
        # A cancel can arrive between accepting the job and the event loop
        # running its first step; clear only here, before ``create_task``.
        service.clear_cancel()

        def on_progress(state: ImportProgress) -> None:
            job.progress = state

        async def run() -> None:
            result = await service.import_sessions(request, progress=on_progress)
            job.result = result
            if result.state == "done":
                job.progress.state = "done"
            elif result.state == "cancelled":
                job.progress.state = "cancelled"
            else:
                job.progress.state = "failed"
                job.progress.error = result.error

        job.task = asyncio.create_task(run())
        self._remember(job)
        return job

    def _remember(self, job: ImportJob) -> None:
        self.jobs[job.job_id] = job
        while len(self.jobs) > self.max_jobs:
            self.jobs.popitem(last=False)

    def payload(self, job: ImportJob) -> dict[str, Any]:
        payload = job.progress.to_dict()
        # The registry key is the public identity; the service mints its own
        # result id internally, and a client polling by the id it was given must
        # see that same id in the payload.
        payload["job_id"] = job.job_id
        if job.result is not None:
            payload["result"] = job.result.to_dict()
        return payload

    def cancel(
        self, job_id: str | None = None, *, session_key: str | None = None
    ) -> dict[str, Any]:
        """Cancel the job's OWN service; never a caller-supplied instance."""
        job = (
            self.get(job_id, session_key=session_key) if job_id else self.running_owned(session_key)
        )
        if job is None:
            raise KeyError(job_id or "<running>")
        if job.task is None or job.task.done():
            return {"job_id": job.job_id, "cancelled": False, "settled": True}
        return {
            "job_id": job.job_id,
            "cancelled": bool(job.service and job.service.cancel()),
            "settled": False,
        }

    def running_owned(self, session_key: str | None) -> ImportJob | None:
        running = self.running()
        if running is None:
            return None
        if session_key is not None and running.session_key != session_key:
            return None
        return running


#: The one process-local registry every surface uses.
JOBS = ImportJobRegistry()

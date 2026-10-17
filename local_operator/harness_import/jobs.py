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
    """One accepted import: its live progress, result and asyncio task."""

    __slots__ = ("job_id", "progress", "result", "task")

    def __init__(self, job_id: str, progress: ImportProgress):
        self.job_id = job_id
        self.progress = progress
        self.result: ImportResult | None = None
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

    def get(self, job_id: str) -> ImportJob | None:
        return self.jobs.get(job_id)

    def latest(self) -> ImportJob | None:
        return next(reversed(self.jobs.values()), None)

    def start(
        self, service: Any, request: ImportRequest, *, job_id: str | None = None
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
        job = ImportJob(progress.job_id, progress)

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

    def cancel(self, service: Any, job_id: str | None = None) -> dict[str, Any]:
        """Request cancellation; returns the public status payload."""
        job = self.get(job_id) if job_id else self.running()
        if job is None:
            raise KeyError(job_id or "<running>")
        if job.task is None or job.task.done():
            return {"job_id": job.job_id, "cancelled": False, "settled": True}
        return {
            "job_id": job.job_id,
            "cancelled": service.cancel(),
            "settled": False,
        }


#: The one process-local registry every surface uses.
JOBS = ImportJobRegistry()

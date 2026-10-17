"""Desktop import surface: sources, sessions, preview and bounded import jobs.

Thin routes over :class:`local_operator.harness_import.service.HarnessImportService`.
They parse a request, call the service, and report progress — no foreign-store
parsing and no transcript writing lives here, so the TUI/agent/CLI surfaces
cannot drift from this one.

Jobs are in-memory and bounded to the newest 20. They survive no restart, and
that is safe because the manifest makes a re-run a no-op or a resume. Only one
import may run per process; the service's file lock refuses a second job rather
than interleaving two writers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from local_operator.harness_import.jobs import JOBS, ImportJob, ImportJobBusy
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.types import ImportRequest
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(
    prefix="/v1/desktop/imports",
    tags=["Desktop imports"],
    dependencies=[Depends(require_desktop)],
)

#: Kept as a module alias because tests and older callers clear the map.
_JOBS = JOBS.jobs


class ImportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[str] | None = None
    session_ids: dict[str, list[str]] = Field(default_factory=dict)
    all: bool = False
    include_subagents: bool = False
    include_cleared_segments: bool = False
    include_memory: bool = True
    include_instructions: bool = True
    memory_status: str = "active"
    max_sessions: int | None = Field(default=None, ge=1, le=5000)

    def request(self) -> ImportRequest:
        status = self.memory_status if self.memory_status in ("active", "candidate") else "active"
        return ImportRequest(
            sources=self.sources,
            session_ids=self.session_ids,
            all=self.all,
            include_subagents=self.include_subagents,
            include_cleared_segments=self.include_cleared_segments,
            include_memory=self.include_memory,
            include_instructions=self.include_instructions,
            memory_status=status,  # type: ignore[arg-type]
            max_sessions=self.max_sessions,
        )


def _job_or_404(job_id: str) -> ImportJob:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="import job not found")
    return job


def _job_payload(job: ImportJob) -> dict[str, Any]:
    return JOBS.payload(job)


def service_for(request: Request) -> HarnessImportService:
    existing = getattr(request.app.state, "harness_import_service", None)
    if existing is not None:
        return existing
    manager = get_config_manager(request)
    service = HarnessImportService(manager.config_dir)
    request.app.state.harness_import_service = service
    return service


@router.get("/sources")
async def import_sources(request: Request) -> CRUDResponse[Any]:
    service = service_for(request)
    sources = await service.sources()
    return CRUDResponse(
        status=200,
        message="Import sources.",
        result={"sources": [info.to_dict() for info in sources]},
    )


@router.get("/sessions")
async def import_sessions(
    request: Request,
    source: str = Query(min_length=1, max_length=16),
    include_subagents: bool = False,
) -> CRUDResponse[Any]:
    service = service_for(request)
    if source not in service.adapters:
        raise HTTPException(status_code=404, detail=f"unknown import source: {source}")
    sessions = await service.list_sessions(source, include_subagents=include_subagents)
    return CRUDResponse(
        status=200,
        message="Import sessions.",
        result={"sessions": [session.to_dict() for session in sessions]},
    )


@router.post("/preview")
async def import_preview(body: ImportBody, request: Request) -> CRUDResponse[Any]:
    service = service_for(request)
    preview = await service.preview(body.request())
    return CRUDResponse(status=200, message="Import preview.", result=preview.to_dict())


@router.post("/jobs")
async def import_start(body: ImportBody, request: Request) -> CRUDResponse[Any]:
    service = service_for(request)
    try:
        job = JOBS.start(service, body.request())
    except ImportJobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CRUDResponse(
        status=200,
        message="Import job accepted.",
        result={"job_id": job.job_id},
    )


@router.get("/jobs/{job_id}")
async def import_status(job_id: str) -> CRUDResponse[Any]:
    return CRUDResponse(status=200, message="Import job.", result=_job_payload(_job_or_404(job_id)))


@router.delete("/jobs/{job_id}")
async def import_cancel(job_id: str, request: Request) -> CRUDResponse[Any]:
    _job_or_404(job_id)
    cancelled = JOBS.cancel(service_for(request), job_id)
    if cancelled.get("settled"):
        return CRUDResponse(
            status=200,
            message="Import job already settled.",
            result=_job_payload(_job_or_404(job_id)),
        )
    return CRUDResponse(
        status=200,
        message="Import cancellation requested.",
        result={**cancelled, "state": _job_or_404(job_id).progress.state},
    )

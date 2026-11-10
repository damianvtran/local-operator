"""Agents proposal review through the authenticated human desktop surface.

Like desktop_profiles, the bearer holder IS the actor; signatures and mesh
membership are irrelevant. Idempotency receipts prevent replaying a decision.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from local_operator import profile_proposals as P
from local_operator.server.desktop import require_desktop
from local_operator.server.routes.desktop_profiles import registries
from local_operator.server.routes.desktop_sessions import (
    Input,
    RequestID,
    errors,
    receipts,
    reply,
)
from local_operator.server.utils.desktop_profiles import profile_detail

router = APIRouter(tags=["Desktop profile proposals"], dependencies=[Depends(require_desktop)])


class Accept(Input):
    request_id: RequestID
    after: dict[str, Any] | None = None


class Reject(Input):
    request_id: RequestID
    note: str | None = Field(default=None, max_length=2000)


def root(request: Request) -> Path:
    return Path(request.app.state.config_manager.config_dir)


def refusal(exc: P.ProposalError, request: Request, proposal_id: str) -> NoReturn:
    status = (
        404
        if exc.code == "profile_proposal_not_found"
        else 409 if exc.code in {"profile_proposal_stale", "profile_proposal_not_pending"} else 422
    )
    detail: dict[str, Any] = {"code": exc.code, "message": str(exc), "fields": list(exc.fields)}
    if exc.code == "profile_proposal_stale":
        record = P.get(root(request), proposal_id)
        if record:
            detail["current"] = P._snapshot(root(request), record["kind"], record["target_id"])
    raise HTTPException(status, detail)


@router.get("/v1/desktop/profile-proposals")
async def proposals(request: Request, state: str = "pending"):
    async with errors(request):
        rows = await asyncio.to_thread(P.list_proposals, root(request), state=state)
        return reply({"proposals": [P.summary(r) for r in rows], "counts": P.counts(rows)})


@router.get("/v1/desktop/profile-proposals/{proposal_id}")
async def proposal(proposal_id: str, request: Request):
    async with errors(request):
        try:
            record = await asyncio.to_thread(P.get, root(request), proposal_id)
            if record is None:
                raise P.ProposalError("profile_proposal_not_found", "Proposal not found.")
            return reply(P.detail(root(request), record))
        except P.ProposalError as exc:
            refusal(exc, request, proposal_id)


@router.post("/v1/desktop/profile-proposals/{proposal_id}/accept")
async def accept(proposal_id: str, body: Accept, request: Request):
    def mutate() -> dict[str, Any]:
        try:
            result = P.apply(root(request), proposal_id, actor="desktop", edited_after=body.after)
            record = result["proposal"]
            agents, teams = registries(request)
            profile = (
                profile_detail(agents, record["name"])
                if record["kind"] == "agent"
                else teams.get_team(record["target_id"]).model_dump(mode="json")
            )
            return {
                "proposal": P.detail(root(request), record),
                "profile": profile,
                "backup_id": result["backup_id"],
            }
        except P.ProposalError as exc:
            refusal(exc, request, proposal_id)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "profile-proposal-accept:" + proposal_id + ":" + body.request_id,
                body.model_dump(),
                lambda: asyncio.to_thread(mutate),
                retry_safe=True,
            )
        )


@router.post("/v1/desktop/profile-proposals/{proposal_id}/reject")
async def reject(proposal_id: str, body: Reject, request: Request):
    def mutate() -> dict[str, Any]:
        try:
            record = P.reject(root(request), proposal_id, actor="desktop", note=body.note)
            return {"proposal": P.detail(root(request), record)}
        except P.ProposalError as exc:
            refusal(exc, request, proposal_id)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "profile-proposal-reject:" + proposal_id + ":" + body.request_id,
                body.model_dump(),
                lambda: asyncio.to_thread(mutate),
                retry_safe=True,
            )
        )

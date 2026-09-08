"""Session-independent authoring adapters over the tool and slash authorities."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from local_operator.agent_profiles import NameTakenError, install_seed
from local_operator.agents import AgentRegistry
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import (
    Input,
    RequestID,
    errors,
    receipts,
    reply,
)
from local_operator.server.utils.desktop_profiles import (
    profile_catalogue,
    profile_detail,
    team_catalogue,
)
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tools.agent_tool import AgentParams, write_profile

router = APIRouter(tags=["Desktop profiles"], dependencies=[Depends(require_desktop)])


class NamedMutation(Input):
    request_id: RequestID
    name: str = Field(min_length=1, max_length=128)


class ProfileEdit(Input):
    request_id: RequestID
    kind: Literal["role", "specialist"] | None = None
    description: str | None = Field(default=None, max_length=8000)
    instructions: str | None = Field(default=None, max_length=8000)
    tools: list[str] | None = None
    effort: str | None = None
    delegate: bool | None = None


class ProfileCreate(Input):
    request_id: RequestID
    name: str = Field(min_length=1, max_length=128)
    kind: Literal["role", "specialist"] = "role"
    description: str = Field(max_length=8000)
    instructions: str = Field(min_length=1, max_length=8000)
    tools: list[str] | None = None
    effort: str | None = None
    delegate: bool | None = None


class TeamEdit(Input):
    request_id: RequestID
    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = None
    manager: str | None = None
    members: list[TeamMember] | None = None
    instructions: str | None = Field(default=None, max_length=8000)
    project: str | None = Field(default=None, max_length=8000)


class TeamCreate(Input):
    request_id: RequestID
    name: str = Field(min_length=1, max_length=64)
    description: str | None = None
    manager: str | None = None
    members: list[TeamMember] | None = None
    instructions: str | None = Field(default=None, max_length=8000)
    project: str | None = Field(default=None, max_length=8000)


def registries(request: Request) -> tuple[AgentRegistry, TeamRegistry]:
    # Fresh metadata per operation sees authoring from another process without
    # waiting for the legacy chat API's polling cache. These are adapters over
    # the same directories, not an alternate registry or migration.
    root = request.app.state.config_manager.config_dir
    return AgentRegistry(root), TeamRegistry(root)


@router.get("/v1/desktop/profiles", response_model=CRUDResponse)
async def profiles(request: Request):
    async with errors():
        return reply(
            {"profiles": await asyncio.to_thread(lambda: profile_catalogue(registries(request)[0]))}
        )


@router.get("/v1/desktop/profiles/{name}", response_model=CRUDResponse)
async def profile(name: str, request: Request):
    async with errors():
        return reply(await asyncio.to_thread(lambda: profile_detail(registries(request)[0], name)))


@router.post("/v1/desktop/profiles/install", response_model=CRUDResponse)
async def install(body: NamedMutation, request: Request):
    def mutate() -> dict[str, Any]:
        agents, _ = registries(request)
        agents.require_complete_metadata()
        try:
            installed = install_seed(body.name, registry=agents)
        except NameTakenError:
            raise HTTPException(
                409,
                "That name belongs to another agent. "
                "Choose a different name to extend the packaged profile.",
            ) from None
        if installed is None:
            raise HTTPException(404, "Packaged profile not found")
        return profile_detail(agents, installed[0].name)

    async with errors():
        return reply(
            await receipts(request).run(
                "profile-install:" + body.request_id,
                body.model_dump(),
                lambda: asyncio.to_thread(mutate),
                retry_safe=True,
            )
        )


async def save_profile(
    name: str, body: ProfileEdit | ProfileCreate, request: Request, *, creating: bool
):
    # Validate effort through the same live configuration validator as the tool;
    # its structured result, never the tool's prose, drives the HTTP response.
    payload = body.model_dump(exclude={"request_id", "name"}, exclude_unset=True)
    params = AgentParams(op="create" if creating else "update", name=name, **payload)

    def mutate() -> dict[str, Any]:
        agents, _ = registries(request)
        agents.require_complete_metadata()
        resolved, _kind = write_profile(agents, params, creating=creating)
        return profile_detail(agents, resolved)

    return reply(
        await receipts(request).run(
            "profile-write:" + body.request_id,
            {"name": name, "creating": creating, **body.model_dump()},
            lambda: asyncio.to_thread(mutate),
        )
    )


@router.post("/v1/desktop/profiles", response_model=CRUDResponse)
async def create_profile(body: ProfileCreate, request: Request):
    async with errors():
        return await save_profile(body.name, body, request, creating=True)


@router.patch("/v1/desktop/profiles/{name}", response_model=CRUDResponse)
async def update_profile(name: str, body: ProfileEdit, request: Request):
    async with errors():
        return await save_profile(name, body, request, creating=False)


@router.get("/v1/desktop/teams", response_model=CRUDResponse)
async def teams(request: Request):
    async with errors():
        return reply(
            {"teams": await asyncio.to_thread(lambda: team_catalogue(registries(request)[1]))}
        )


@router.get("/v1/desktop/teams/{name}", response_model=CRUDResponse)
async def team(name: str, request: Request):
    def read() -> dict[str, Any]:
        found = registries(request)[1].get_team_by_name(name)
        if found is None:
            raise KeyError(name)
        return found.model_dump(mode="json")

    async with errors():
        return reply(await asyncio.to_thread(read))


async def save_team(name: str, body: TeamEdit | TeamCreate, request: Request, *, creating: bool):
    fields = TeamEditFields(**body.model_dump(exclude={"request_id"}, exclude_unset=True))

    def mutate() -> dict[str, Any]:
        registry = registries(request)[1]
        if creating:
            result = registry.create_team(fields)
        else:
            current = registry.get_team_by_name(name)
            if current is None:
                raise KeyError(name)
            # Partial update hydrates untouched briefs itself. A metadata DTO
            # sent to save_team would silently replace lazy briefs with blanks.
            result = registry.update_team(current.id, fields)
        return result.model_dump(mode="json")

    return reply(
        await receipts(request).run(
            "team-write:" + body.request_id,
            {"target": name, "creating": creating, **body.model_dump()},
            lambda: asyncio.to_thread(mutate),
        )
    )


@router.post("/v1/desktop/teams", response_model=CRUDResponse)
async def create_team(body: TeamCreate, request: Request):
    async with errors():
        return await save_team(body.name, body, request, creating=True)


@router.patch("/v1/desktop/teams/{name}", response_model=CRUDResponse)
async def update_team(name: str, body: TeamEdit, request: Request):
    async with errors():
        return await save_team(name, body, request, creating=False)

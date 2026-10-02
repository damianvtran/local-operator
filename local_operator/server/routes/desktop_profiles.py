"""Session-independent authoring adapters over the tool and slash authorities."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from local_operator.agent_profiles import NameTakenError, install_seed
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.env import EnvConfig, get_env_config
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_config_manager,
    get_provider_auth_store,
)
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

router = APIRouter(tags=["Desktop profiles"], dependencies=[Depends(require_desktop)])


class NamedMutation(Input):
    request_id: RequestID
    name: str = Field(min_length=1, max_length=128)


class ProfileSync(Input):
    """A sync request: one profile by ``name``, or everything.

    ``all``/``name`` are the design's ``{name?|all}``; passing both is refused
    (422) rather than silently resolved, because each reading of that request
    does materially different work — "sync reviewer" and "sync everything"
    are not two phrasings of one action.
    """

    request_id: RequestID
    name: str | None = Field(default=None, min_length=1, max_length=128)
    all: bool = False
    #: DEPRECATED (design B5.5). It used to mean "overwrite local edits" for the hub
    #: arm; that is now ``replace`` and must be CONFIRMED, because it is the one
    #: operation that discards a side. It still forces the SEED arm, a different
    #: family with its own well-understood meaning.
    force: bool = False
    confirm_replace: bool = False


class ProfileEdit(Input):
    request_id: RequestID
    kind: Literal["role", "specialist"] | None = None
    description: str | None = Field(default=None, max_length=8000)
    instructions: str | None = Field(default=None, max_length=8000)
    tools: list[str] | None = None
    effort: str | None = None
    delegate: bool | None = None
    #: The proactive-class switch's editing path (design §8.1.3): the same
    #: field the ``agent`` tool exposes, so the desktop Class control (UI-3)
    #: writes through the ordinary profile update route. Additive and
    #: optional: an older client simply never sends it, and an omitted field
    #: never clears a class (the merge rule ``write_profile`` documents).
    action_class: Literal["reactive", "proactive"] | None = None
    #: Display metadata (see ``local_operator.agents.AgentData.label``): the
    #: label every listing paints while ``name`` stays the addressing key.
    #: Deliberately no ``max_length``: the registry validates the stored shape
    #: (whitespace collapsed, 80 characters, no controls) and a route cap
    #: stricter than the registry would refuse values the other write paths
    #: accept. A label of "" resets to the derived default; ``None``/absent
    #: leaves it alone.
    label: str | None = None


class ProfileCreate(Input):
    request_id: RequestID
    name: str = Field(min_length=1, max_length=128)
    kind: Literal["role", "specialist"] = "role"
    description: str = Field(max_length=8000)
    instructions: str = Field(min_length=1, max_length=8000)
    tools: list[str] | None = None
    effort: str | None = None
    delegate: bool | None = None
    #: See ProfileEdit. Creation defaults to reactive downstream when absent
    #: (R37); naming ``proactive`` here is the explicit spare-use act.
    action_class: Literal["reactive", "proactive"] | None = None
    #: See ProfileEdit. Creation just carries it through; an absent label means
    #: the derived default.
    label: str | None = None


class TeamEdit(Input):
    request_id: RequestID
    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = None
    manager: str | None = None
    members: list[TeamMember] | None = None
    #: Display metadata (see ``local_operator.teams.Team``): a LOCAL-only label
    #: every listing shows label-first, and extra addressing keys. Deliberately
    #: no ``max_length``: the registry validates the stored shape (whitespace
    #: collapsed, 80 characters, no controls) and a route cap stricter than the
    #: registry would refuse values the other write paths accept. A label of ""
    #: resets to the derived default; ``None``/absent leaves it alone.
    label: str | None = None
    aliases: list[str] | None = None
    instructions: str | None = Field(default=None, max_length=8000)
    project: str | None = Field(default=None, max_length=8000)


class TeamCreate(Input):
    request_id: RequestID
    name: str = Field(min_length=1, max_length=64)
    description: str | None = None
    manager: str | None = None
    members: list[TeamMember] | None = None
    #: See TeamEdit. Creation just carries them through; an absent label means
    #: the derived default, an absent alias list means none.
    label: str | None = None
    aliases: list[str] | None = None
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
    async with errors(request):
        return reply(
            {"profiles": await asyncio.to_thread(lambda: profile_catalogue(registries(request)[0]))}
        )


@router.get("/v1/desktop/profiles/{name}", response_model=CRUDResponse)
async def profile(name: str, request: Request):
    async with errors(request):
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
        # Pass through whether this call WROTE anything. The install-all
        # shortcut loops this op once per built-in, so without the flag its
        # summary cannot distinguish a fresh install from an idempotent no-op
        # and reports a count that is simply wrong (contract §5.6).
        return profile_detail(agents, installed[0].name, already_installed=installed[1])

    async with errors(request):
        return reply(
            await receipts(request).run(
                "profile-install:" + body.request_id,
                body.model_dump(),
                lambda: asyncio.to_thread(mutate),
                retry_safe=True,
            )
        )


@router.post("/v1/desktop/profiles/sync", response_model=CRUDResponse)
async def sync_profiles(
    body: ProfileSync,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    env_config: EnvConfig = Depends(get_env_config),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Pull the latest for installed starter profiles and hub-pulled agents.

    One request can do two very different kinds of work: an unedited starter
    updates its local text in place, and a hub-pulled row re-fetches its
    marketplace listing — so the response carries every verdict with the echo
    of any replaced instructions (the same recoverability ``reset`` gives),
    never just a count. The hub arm needs a credential: without one those rows
    report ``unavailable`` and the local seed updates still run (design §9.2),
    which is why credential resolution here is best-effort rather than a 401.
    """

    if body.name and body.all:
        raise HTTPException(422, "Pass either name or all, not both")
    names = [body.name] if body.name else None

    # Imported per call, not at module scope: this pulls the sync machinery's
    # graph (agents + radient client), which is dead weight on every `lop serve`
    # boot for a route that only runs when someone asks for an update.
    from local_operator.agent_profiles import sync_installed_seeds
    from local_operator.agent_sync import SyncReport, sync_payload
    from local_operator.hub_sync import service as svc

    async def mutate() -> dict[str, Any]:
        agents, _ = registries(request)
        agents.require_complete_metadata()
        # The person-scoped resolver with the route's shared AuthStore, the pattern
        # the Radient routes already use so a credential refresh persists in the one
        # place the login owns. Best-effort rather than a 401: without a login the
        # hub rows report `unavailable` and the local seed updates still run.
        for_tenant, credential = await svc.build_clients(config_manager, provider_auth_store)
        ctx = svc.HubSyncContext(
            config_dir=config_manager.config_dir,
            config_manager=config_manager,
            client_for_tenant=for_tenant,
            credential=credential,  # type: ignore[arg-type]
            agent_registry=agents,
        )

        def run() -> dict[str, Any]:
            # `force` on the hub arm now means `replace`, which throws away the
            # user's copy, so a bare boolean from an old client must not reach it
            # unconfirmed. Decided BEFORE any work (the seed arm below also honours
            # `force`, and a refused request must have changed nothing), and only
            # when the hub arm would actually be reached: an old seed-arm-only
            # client that sends `force` for the starters keeps working and simply
            # gets the hub arm as a plain merge.
            replace = "remote" if body.force and body.confirm_replace else None
            if body.force and not body.confirm_replace and _has_hub_rows(agents, names):
                raise HTTPException(
                    422, "--force replaces your copy; use replace with confirm_replace"
                )
            seeds = SyncReport(
                entries=tuple(sync_installed_seeds(agents, names=names, force=body.force))
            )
            try:
                hub = svc.apply_items(ctx, kind="agent", names=names, replace=replace)
            except svc.HubBusy as busy:
                raise HTTPException(409, str(busy)) from None
            payload = sync_payload(seeds)
            payload["hub"] = hub.to_json()
            return payload

        return await asyncio.to_thread(run)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "profile-sync:" + body.request_id,
                body.model_dump(),
                mutate,
                retry_safe=True,
            )
        )


def _has_hub_rows(agents: Any, names: list[str] | None) -> bool:
    """Whether the hub arm has any linked agent to act on for this request.

    Decides whether an unconfirmed ``force`` is a seed-only call (harmless, kept
    working for old clients) or one that would reach a hub-pulled row.
    """

    from local_operator.agents import hub_origin

    wanted = {n.strip().casefold() for n in names} if names else None
    return any(
        hub_origin(a) is not None and (wanted is None or str(a.name).strip().casefold() in wanted)
        for a in agents.list_agents()
    )


async def save_profile(
    name: str, body: ProfileEdit | ProfileCreate, request: Request, *, creating: bool
):
    # Validate effort through the same live configuration validator as the tool;
    # its structured result, never the tool's prose, drives the HTTP response.
    # Imported per write, not at module scope: `tools.agent_tool` pulls
    # `tools.builtin`, and this router is imported by `server.app`, so the tool
    # layer was ~160 ms of every `lop serve` boot for a route that only runs when
    # someone saves a profile (backend load report B-F10).
    from local_operator.tools.agent_tool import AgentParams, write_profile

    payload = body.model_dump(exclude={"request_id", "name"}, exclude_unset=True)
    params = AgentParams(op="create" if creating else "update", name=name, **payload)

    def mutate() -> dict[str, Any]:
        agents, _ = registries(request)
        agents.require_complete_metadata()
        # Three values since the class shipped; the third is only used by the
        # tool's receipt, so the route discards it (its response carries the
        # profile detail, class included).
        resolved, _kind, _action_class = write_profile(agents, params, creating=creating)
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
    async with errors(request):
        return await save_profile(body.name, body, request, creating=True)


@router.patch("/v1/desktop/profiles/{name}", response_model=CRUDResponse)
async def update_profile(name: str, body: ProfileEdit, request: Request):
    async with errors(request):
        return await save_profile(name, body, request, creating=False)


@router.get("/v1/desktop/teams", response_model=CRUDResponse)
async def teams(request: Request):
    async with errors(request):
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

    async with errors(request):
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
    async with errors(request):
        return await save_team(body.name, body, request, creating=True)


@router.patch("/v1/desktop/teams/{name}", response_model=CRUDResponse)
async def update_team(name: str, body: TeamEdit, request: Request):
    async with errors(request):
        return await save_team(name, body, request, creating=False)

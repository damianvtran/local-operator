"""Protected desktop reads from runtime, provider and analytics authorities."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.auth import get_desktop_auth
from local_operator.server.routes.desktop_sessions import errors, host, reply
from local_operator.server.utils.desktop_auth import DesktopAuth
from local_operator.server.utils.desktop_commands import command_catalogue
from local_operator.slash_commands import slash_command_for

router = APIRouter(tags=["Desktop catalogues"], dependencies=[Depends(require_desktop)])


class CommandMetadata(BaseModel):
    name: str
    description: str
    aliases: list[str]
    arguments: Literal["none", "optional", "required"]
    echo: bool
    consumes_prompt: bool
    destination: str
    execution: Literal["owner", "native"]


class Commands(BaseModel):
    commands: list[CommandMetadata]


class Catalogue(BaseModel):
    models: list[dict[str, Any]]
    source: Literal["initial", "live"]
    errors: dict[str, str] = Field(default_factory=dict)
    #: Whether the credential store could be read. When False, every row's
    #: `connected` is a listing default rather than a statement about auth, and
    #: a caller must not group or badge on it.
    credentials_known: bool = True


class UsageReports(BaseModel):
    reports: list[dict[str, Any]]
    source: Literal["cached", "live"]
    fetched_at: int


class Report(BaseModel):
    data: dict[str, Any]


class Entities(BaseModel):
    command: str
    entities: list[dict[str, Any]]
    current: Any = None


@router.get("/v1/desktop/commands", response_model=CRUDResponse[Commands])
async def commands():
    return reply({"commands": command_catalogue()})


@router.get("/v1/desktop/models", response_model=CRUDResponse[Catalogue])
async def models(live: bool = False, auth: DesktopAuth = Depends(get_desktop_auth)):
    controller = auth.controller()
    try:
        failures: dict[str, str] = {}
        if live:
            entries, raw_failures = await controller.live_catalogue()
            # Provider exceptions can carry response bodies or credential URLs.
            failures = {key: "Model listing unavailable" for key in raw_failures}
        else:
            # NOT `asyncio.to_thread`. `initial_catalogue` is synchronous and
            # I/O-free by contract (it exists to paint on the keystroke that
            # opens the picker; measured 0.21 ms median, 0.77 ms max), so the
            # hop bought nothing -- and it cost correctness: the AuthStore's
            # sqlite connection is created on the event-loop thread, so reading
            # it from a worker raised `ProgrammingError`, which
            # `usable_providers()` reported as "store unreadable" and the
            # catalogue turned into "everything is connected" on a machine with
            # no credentials (D18). Keep this call on the loop thread.
            entries = controller.initial_catalogue()
        # `CatalogueEntry.connected` is True both when a provider IS usable and
        # when the credential store could not be read at all -- the deliberate
        # "show everything rather than claim you own no models" degradation. For
        # LISTING that is right; for LABELLING it is not, and it put every model
        # under a "Connected" heading on a fixture with no credentials (D5).
        # Carrying the uncertainty separately lets the picker keep listing
        # everything while only claiming what is known.
        credentials_known = controller.usable_providers() is not None
        return reply(
            {
                "models": [dataclasses.asdict(row) | {"selector": row.selector} for row in entries],
                "source": "live" if live else "initial",
                "errors": failures,
                "credentials_known": credentials_known,
            }
        )
    finally:
        controller.close()


@router.get("/v1/desktop/usage", response_model=CRUDResponse[UsageReports])
async def usage(
    provider: str | None = Query(default=None, max_length=64),
    live: bool = False,
    refresh: bool = False,
    auth: DesktopAuth = Depends(get_desktop_auth),
):
    controller = auth.controller()
    try:
        if provider and controller.provider(provider) is None:
            raise HTTPException(422, "Unknown provider")
        if refresh and not live:
            raise HTTPException(422, "Refresh requires live usage")
        reports = (
            await controller.fetch_usage([provider] if provider else None, force_refresh=refresh)
            if live
            else controller.cached_usage_reports(provider)
        )
        rows = []
        now = int(time.time() * 1000)
        for report in reports:
            row = dataclasses.asdict(report)
            # Arbitrary provider notes are not a safe public error vocabulary.
            row["notes"] = None
            row["age_ms"] = max(0, now - report.fetched_at)
            row["state"] = (
                "reauth_required"
                if report.credential_invalid
                else (
                    "unavailable"
                    if report.usage_unavailable or not report.limits
                    else "partial" if report.consecutive_failures else "available"
                )
            )
            rows.append(row)
        return reply({"reports": rows, "source": "live" if live else "cached", "fetched_at": now})
    finally:
        controller.close()


@router.get("/v1/desktop/analytics", response_model=CRUDResponse[Report])
async def analytics(
    request: Request,
    since_ms: int | None = Query(default=None, ge=0),
    until_ms: int | None = Query(default=None, ge=0),
    session_id: str | None = Query(default=None, pattern=r"^[a-f0-9]{12}$"),
    days: int = Query(default=30, ge=1, le=366),
):
    from local_operator.analytics.store import AnalyticsStore

    if since_ms is not None and until_ms is not None and since_ms > until_ms:
        raise HTTPException(422, "The start must precede the end")

    def read_report():
        store = AnalyticsStore(request.app.state.config_manager.config_dir / "analytics.db")
        try:
            aggregate = store.aggregate(since_ms=since_ms, until_ms=until_ms, session_id=session_id)
            return {
                "aggregate": dataclasses.asdict(aggregate),
                "daily": [dataclasses.asdict(row) for row in store.daily_series(days)],
                "daily_scope": "all_sessions",
            }
        finally:
            store.close()

    return reply({"data": await asyncio.to_thread(read_report)})


@router.get("/v1/desktop/skills", response_model=CRUDResponse[Report])
async def skills(
    request: Request,
    session_id: str = Query(pattern=r"^[a-f0-9]{12}$"),
    name: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_.-]{1,128}$"),
):
    from local_operator.skills import default_skill_roots, discover_skills
    from local_operator.skills.api import resolve_skill_url

    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        cwd = bridge.remote.frontend_state.cwd
        discovered, warnings = await asyncio.to_thread(
            discover_skills, default_skill_roots(Path(cwd))
        )
        detail = None
        if name is not None:
            by_name = {item.name: item for item in discovered}
            if name not in by_name:
                raise HTTPException(404, "Skill not found")
            detail = await asyncio.to_thread(resolve_skill_url, "skill://" + name, by_name)
        # Details use the runtime's closed internal-URL resolver, not arbitrary paths.
        return reply(
            {
                "data": {
                    "skills": [
                        {"name": item.name, "description": item.description} for item in discovered
                    ],
                    "scope": "discoverable",
                    "detail": detail,
                    "warning_count": len(warnings),
                }
            }
        )


@router.get("/v1/desktop/sessions/{session_id}/failovers", response_model=CRUDResponse[Report])
async def failovers(session_id: str, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        state = bridge.remote.frontend_state
        from local_operator.settings_io import read_chains

        chains = read_chains(request.app.state.config_manager)
        return reply(
            {
                "data": {
                    "selected": state.selected_model,
                    "effective": state.effective_model,
                    "chains": chains,
                    "scope": "configured_defaults",
                    "live_model_source": "owner",
                }
            }
        )


@router.get(
    "/v1/desktop/sessions/{session_id}/command-entities", response_model=CRUDResponse[Entities]
)
async def entities(
    session_id: str,
    command: str,
    request: Request,
    auth: DesktopAuth = Depends(get_desktop_auth),
    name: str | None = Query(default=None, max_length=128),
):
    spec = slash_command_for("/" + command.removeprefix("/"))
    if spec is None:
        raise HTTPException(422, "Unknown command")
    async with errors(), host(request).session(session_id) as bridge:
        remote = bridge.remote
        assert remote is not None
        assert bridge.remote is not None
        state = bridge.remote.frontend_state
        rows: list[dict[str, Any]] = []
        current: Any = None
        if spec.name == "model":
            controller = auth.controller()
            try:
                # On the loop thread for the same reason as `/v1/desktop/models`
                # above: the store's connection belongs to this thread, and a
                # worker turns an unreadable-store degradation into a false
                # "connected" for every model in the `/model` picker (D18).
                rows = [
                    dataclasses.asdict(row) | {"value": row.selector}
                    for row in controller.initial_catalogue()
                ]
                current = state.selected_model
            finally:
                controller.close()
        elif spec.name == "effort":
            # Owner-resolved capabilities can differ from the static model-id
            # registry (aggregator listings and explicit model overrides).
            rows = [{"value": value} for value in remote.model.reasoning_efforts]
            current = remote.model.reasoning_effort
        elif spec.name == "approvals":
            rows = [{"value": value} for value in ("auto", "ask")]
        elif spec.name in {"team", "agent"}:
            registry = remote.team_registry if spec.name == "team" else remote.agent_registry
            if registry is None:
                raise HTTPException(503, "The profile registry is unavailable")
            if spec.name == "team":
                from local_operator.org_chart import resolve_org

                rows = [
                    item.model_dump(mode="json") | {"value": item.name}
                    for item in registry.list_teams()
                ]
                if name:
                    current = dataclasses.asdict(
                        resolve_org(name, teams=registry, agents=remote.agent_registry)
                    )
            else:
                from local_operator.agent_profiles import (
                    is_role,
                    is_specialist,
                    list_seeds,
                    resolve_profile_or_specialist,
                )

                names = {
                    item.name
                    for item in registry.list_agents()
                    if is_role(item) or is_specialist(item)
                } | set(list_seeds())
                for candidate in sorted(names):
                    kind, profile, instructions, resolved_name = resolve_profile_or_specialist(
                        candidate, registry=registry
                    )
                    rows.append(
                        {
                            "value": candidate,
                            "name": resolved_name,
                            "kind": kind,
                            "description": profile.description if profile else "",
                            "profile": (
                                dataclasses.asdict(profile)
                                if name == candidate and profile
                                else None
                            ),
                            "instructions": instructions if name == candidate else None,
                        }
                    )
        return reply({"command": spec.name, "entities": rows, "current": current})

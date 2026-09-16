"""Protected desktop reads from runtime, provider and analytics authorities."""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
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
    #: Whether text after this command's word is an ARGUMENT the command owns on
    #: the DESKTOP (see ``SlashCommand.prefixes_text``). ADDITIVE with a default,
    #: so a response produced before this field existed still validates — the
    #: renderer ignores the key or falls back to its own derivation.
    prefixes_text: bool = False
    #: The SHAPE the trailing text must have for the desktop to use it as this
    #: command's argument — ``none`` (never), ``word`` (one selector token),
    #: ``provider`` (one token naming a provider), ``subcommand`` (``<sub>
    #: [name]``, the MCP shape) or ``any`` (a handler/form field takes any text).
    #: See ``ArgumentShape``. Additive with a default like ``prefixes_text``.
    argument_shape: Literal["none", "word", "provider", "subcommand", "any"] = "none"
    #: The vocabulary ``argument_shape``'s first token must come from; empty
    #: means any word. Carried so a renderer reproduces the endpoint's answer
    #: (``/login openai`` is a command, ``/login zzz`` is a message) without a
    #: second copy of the provider or subcommand list.
    argument_words: list[str] = Field(default_factory=list)
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


#: A session id in a PATH segment. Declared rather than validated inside the
#: handler for the reason ``AttachmentDigest`` is: the ledger query takes the id
#: as a bound parameter, and pinning the shape at the route means no later edit
#: inside a handler can route around it. 12 hex characters is the canonical
#: session id (the same pattern ``/analytics`` and ``/skills`` use for the query
#: form of this identifier).
#:
#: ``Path`` here is FastAPI's route-parameter marker, as it is in the sibling
#: ``desktop_sessions.py``: this module also builds filesystem paths
#: (``pathlib.Path(cwd)`` for skill discovery), so the filesystem class is
#: imported by module name rather than by name — flake8's F811 is the guard that
#: noticed when they collided.
SessionID = Annotated[str, Path(pattern=r"^[a-f0-9]{12}$")]


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
            # ``session_names``/``session_parents`` are SIDE ATTRIBUTES on the
            # aggregate (``store.py`` attaches them with ``setattr``), so
            # ``asdict`` drops them: it walks declared fields only. They are
            # read here rather than folded into the per-session rows because the
            # TUI's tree rollup is a PRESENTATION choice — the payload stays a
            # raw per-session feed whose column sums to the headline total, and a
            # client that wants the tree is handed the edges to re-partition
            # itself. Absent (an older ledger without ``parent_session_id``, or a
            # store that predates the attributes) is ``{}``, which is a true
            # "no edges known" rather than a failure.
            #
            # ``session_names`` is filtered to NAMED sessions. The store's map is
            # COMPLETE over ``by_session`` with ``""`` wherever no name is
            # recorded, and an empty string is not the same fact as an unknown
            # one — it defeats the documented fallback rather than triggering it.
            # A client reads this as ``names?.[id] ?? id``, and ``"" ?? id`` is
            # ``""``, so a session with no title would paint a blank cell while
            # being styled as though it were showing an id. Absence is what
            # makes ``?? id`` fire, so absence is what we emit.
            return {
                "aggregate": dataclasses.asdict(aggregate),
                "daily": [dataclasses.asdict(row) for row in store.daily_series(days)],
                "daily_scope": "all_sessions",
                "session_names": {
                    sid: name
                    for sid, name in (getattr(aggregate, "session_names", None) or {}).items()
                    if name
                },
                "session_parents": dict(getattr(aggregate, "session_parents", None) or {}),
            }
        finally:
            store.close()

    return reply({"data": await asyncio.to_thread(read_report)})


#: The `info.get` fields `LiveState()` leaves at a dataclass DEFAULT and this
#: read therefore never measured — the session-attached half of the snapshot.
#: Each is a `0`/`False`/`[]` that is indistinguishable from a reading, and the
#: contract this route implements names that indistinguishability as its
#: riskiest assumption: a host whose backend never attached a session would
#: otherwise paint "MCP 0 connected" and "no subagents" as facts about the
#: machine, on the one screen whose job is to be believed.
#:
#: Nulled HERE rather than in `collect_snapshot` because it is a statement about
#: this SURFACE: the desktop's host view has no session by construction, while
#: the TUI's `/info` renders these same fields from the live session it is
#: attached to, where `0` is a real reading. `collect_snapshot`'s shared output
#: is deliberately left alone.
#:
#: The host half is NOT touched: `agents.profiles`/`teams`, the session-registry
#: tallies, `env.guides`/`credential_keys` and the terminal/browser/mobile probes
#: are all real readings of this machine. "Terminal probe" there means
#: `env.term`/`colorterm`/`multiplexer`/`is_tty` — `os.environ` and `isatty`
#: reads this very call performs.
#:
#: `env.theme` is the one field that is neither nulled nor a reading, and it is
#: named here so the two spellings cannot drift apart again. The theme is an APP
#: fact: `collect_live` takes it as a parameter and the TUI passes
#: `theme.current_theme()` (`tui/app.py`), and it is the TERMINAL's `tui.theme`
#: scope, which `docs/DESKTOP_API.md` keeps distinct from the desktop's own
#: theme. This route has no such app, so it ships the dataclass default `""`,
#: which is `env.theme`'s documented UNKNOWN spelling (`EnvInfo.theme`): the name
#: of a registered theme is never empty — `set_theme` raises on an unknown one —
#: and the client's `omitEmpty` drops empty rows (verified in review round 2),
#: so nothing blank is painted. Nulling it
#: instead would need `theme` nullable in §5.1's `theme: string` plus a client
#: change, to say what that row already says.
_UNMEASURED_ON_THE_HOST_VIEW: dict[str, tuple[str, ...]] = {
    "agents": (
        "tree",
        "running",
        "queued",
        "settled",
        "max_running",
        "at_capacity",
        "max_depth",
        "deeper",
        "roster_unread",
        "cross_session_known",
    ),
    "env": (
        "mcp_configured",
        "mcp_connected",
        "mcp_failed",
        "mcp_settling",
        "mcp_failures",
        "approval_mode",
        "skills",
    ),
}


def _unmeasure_live_half(snapshot: dict[str, Any]) -> dict[str, Any]:
    """``None`` — the unknown spelling — for every field no session measured."""
    for block, fields in _UNMEASURED_ON_THE_HOST_VIEW.items():
        for field in fields:
            snapshot[block][field] = None
    return snapshot


@router.get("/v1/desktop/info", response_model=CRUDResponse[Report])
async def info():
    """The host read behind the desktop ``/info`` panel.

    No parameters, because ``/info`` has exactly one answer per host — the same
    reason the slash command takes no argument at all.

    ``collect_snapshot`` BLOCKS (the macOS session probe measured ~880 ms,
    because it shells ``top -l1`` for the whole system), so it runs on a worker
    thread through ``asyncio.to_thread`` exactly as ``/analytics`` does; on the
    loop it would stall every other request for the duration.

    ``LiveState()`` is deliberately EMPTY — no session is attached and the
    session bridge is not touched. The live half of the snapshot (the subagent
    tree, job counts, MCP probes, approval mode) is a fact about a SESSION, and
    the desktop already holds those live in ``canonical.frontend`` for the
    conversation on screen; filling them here would give one live fact two
    sources of truth. The host half (install, process, session registry, env)
    is what this panel renders, and `_unmeasure_live_half` nulls the fields that
    belong to the other half so they cannot read as measurements nobody took.
    """
    from local_operator.info.collect import LiveState, collect_snapshot

    snapshot = await asyncio.to_thread(collect_snapshot, LiveState())
    return reply({"data": _unmeasure_live_half(dataclasses.asdict(snapshot))})


@router.get("/v1/desktop/sessions/{session_id}/report", response_model=CRUDResponse[Report])
async def session_report(
    session_id: SessionID,
    request: Request,
    recent_limit: int = Query(default=12),
):
    """One exact session's ledger report, from one WAL snapshot.

    ``AnalyticsStore.session_report`` opens a single explicit read transaction,
    so every figure on the panel comes from the same committed state even while
    the recorder writes behind it. It uses arithmetic and bounds the caller does
    not restate here — in particular the 0..50 bound on ``recent_limit`` is the
    store's own clamp (``store.py``), applied there so the HTTP surface and the
    terminal report cannot disagree about how many rows "the last 50" means.

    ``recent_limit`` is therefore deliberately UNBOUNDED at this declaration: a
    caller asking for 500 gets the store's 50, which is the documented "hard cap
    in the API" rather than a 422 whose error text would have to repeat the
    number the store owns.

    JSON encoding is part of the contract, not an implementation detail.
    ``by_model`` is keyed by a ``(provider, model_id)`` TUPLE and
    ``by_purpose_outcome`` by a ``(purpose, outcome)`` tuple; a JSON object
    cannot carry tuple keys, so ``asdict`` of either is not serialisable and both
    are rebuilt as ARRAYS of objects here.

    ``by_purpose`` is keyed by a plain string and would therefore serialise as an
    object, and that validity is exactly why it is converted anyway: one response
    carrying three sibling group-bys in two different encodings makes the client
    keep one shape per breakdown and makes a reader remember which is which. The
    key type is an accident of the data, not a decision about the wire, so all
    three are ``[{...key fields, "aggregate": {...}}]``.
    """
    from local_operator.analytics.store import AnalyticsStore

    def read_report() -> dict[str, Any]:
        store = AnalyticsStore(request.app.state.config_manager.config_dir / "analytics.db")
        try:
            report = store.session_report(session_id, recent_limit=recent_limit)
        finally:
            store.close()
        # ALL THREE group-bys are blanked BEFORE the dump rather than
        # overwritten after it: ``asdict`` would otherwise build every one of
        # them only for the rebuild below to discard it. ``by_model`` and
        # ``by_purpose_outcome`` are keyed by tuples and could not be serialised
        # at all; ``by_purpose`` is keyed by a string and WOULD serialise — it
        # is blanked because one response must not ship two encodings for one
        # idea (the docstring above states the whole rule).
        payload = dataclasses.asdict(
            dataclasses.replace(report, by_model={}, by_purpose={}, by_purpose_outcome={})
        )
        payload["by_model"] = [
            {
                "provider": provider,
                "model_id": model_id,
                "aggregate": dataclasses.asdict(aggregate),
            }
            for (provider, model_id), aggregate in report.by_model.items()
        ]
        payload["by_purpose"] = [
            {"purpose": purpose, "aggregate": dataclasses.asdict(aggregate)}
            for purpose, aggregate in report.by_purpose.items()
        ]
        payload["by_purpose_outcome"] = [
            {"purpose": purpose, "outcome": outcome, "calls": calls}
            for (purpose, outcome), calls in report.by_purpose_outcome.items()
        ]
        return payload

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
            discover_skills, default_skill_roots(pathlib.Path(cwd))
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
            # Runtime-resolved capabilities can differ from the static model-id
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
                from local_operator.server.utils.desktop_profiles import (
                    profile_catalogue,
                    profile_detail,
                )

                rows = [dict(item, value=item["name"]) for item in profile_catalogue(registry)]
                if name:
                    current = profile_detail(registry, name)
        return reply({"command": spec.name, "entities": rows, "current": current})

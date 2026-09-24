"""HTTP boundary + real settings/auth stores, without third-party credentials."""

import asyncio
import dataclasses
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from local_operator import settings_io
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.jobs import JobManager
from local_operator.providers import registry
from local_operator.scheduler_service import SchedulerService
from local_operator.server.app import desktop_validation_error, managed_desktop_boundary
from local_operator.server.routes import (
    agents,
    auth,
    capabilities,
    config,
    credentials,
    jobs,
    schedules,
    settings,
)

TOKEN = "desktop-contract-test-token"
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(capabilities.router)
    app.include_router(auth.router)
    app.include_router(settings.router)
    app.include_router(config.router)
    app.include_router(credentials.router)
    # The agent and job routers are mounted so the legacy-gate tests assert on
    # the GATE rather than on a missing route: without them every gated path
    # 404s, and the unmanaged-mode assertions ("status is not 401/403") pass
    # whether or not the boundary works.
    app.include_router(agents.router)
    app.include_router(jobs.router)
    # Mounted for the same reason as agents/jobs: without it every schedules
    # path 404s and the gate assertions below pass whether or not the boundary
    # exists. A schedule is delayed EXECUTION, so this router's gating is the
    # one that matters most.
    app.include_router(schedules.router)
    app.middleware("http")(managed_desktop_boundary)
    app.exception_handler(RequestValidationError)(desktop_validation_error)
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.agent_registry = AgentRegistry(tmp_path)
    app.state.job_manager = JobManager()
    # Never started: these tests assert on the HTTP boundary, and a running
    # APScheduler would fire `_trigger_agent_task` for real. The routes only
    # need `add_or_update_job`/`remove_job` to be callable.
    app.state.scheduler_service = MagicMock(spec=SchedulerService)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


async def wait_for_state(client: AsyncClient, operation_id: str, *states: str) -> dict[str, Any]:
    # The host publishes state on event-loop turns. Do not turn this into a
    # sleep calibrated to one developer's machine; the bound only catches hangs.
    for _ in range(1000):
        data = (await client.get(f"/v1/auth/operations/{operation_id}")).json()["result"]
        if data["state"] in states:
            return data
        await asyncio.sleep(0)
    pytest.fail(f"Login did not reach {states}")


async def test_desktop_token_origin_and_unconfigured_fail_closed(desktop, monkeypatch):
    client, _ = desktop
    assert (await client.get("/v1/settings", headers={"Authorization": ""})).status_code == 401
    assert (
        await client.get("/v1/settings", headers={"Origin": "https://evil.example"})
    ).status_code == 403
    assert (await client.get("/v1/settings", headers={"Origin": "null"})).status_code == 403
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
    assert (
        await client.get("/v1/settings", headers={"Origin": "http://localhost:5187"})
    ).status_code == 200
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    assert (await client.get("/v1/settings")).status_code == 503
    public = (await client.get("/v1/capabilities")).json()["result"]
    assert public["desktop_available"] is False
    assert TOKEN not in str(public)


async def test_managed_legacy_controls_require_token_but_unmanaged_remains_compatible(
    desktop, monkeypatch
):
    client, _ = desktop
    for path in ("/v1/config", "/v1/credentials", "/v1/config/system-prompt"):
        assert (await client.get(path, headers={"Authorization": ""})).status_code == 401
        assert (await client.patch(path, json={}, headers={"Authorization": ""})).status_code == 401
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    assert (await client.get("/v1/config", headers={"Authorization": ""})).status_code == 200
    assert (
        await client.patch(
            "/v1/credentials",
            json={"key": "EXAMPLE_KEY", "value": "example"},
            headers={"Authorization": ""},
        )
    ).status_code == 200


def test_managed_gate_covers_every_legacy_route() -> None:
    """Every ``/v1/agents*`` and ``/v1/jobs*`` route is gated, or an explicit exception.

    THE regression guard for this boundary. Hand-maintained string matching
    missed three routes in review round 1 and five more in round 2 -- including
    an unauthenticated cross-origin ``PATCH`` that renamed an agent and
    persisted it -- because each new route had to be remembered into the gate.

    This walks the ROUTER instead, so a route added tomorrow is covered the day
    it exists: either the gate matches it, or its ``METHOD /template`` key sits
    in ``_LEGACY_GATE_EXCEPTIONS`` with a stated reason and a reviewer had to
    write that reason down. Do not silence a failure here by adding a key
    without one.
    """
    from local_operator.server.app import (
        _LEGACY_GATE_EXCEPTIONS,
        _legacy_desktop_gated,
        legacy_gate_routes,
    )

    routes = legacy_gate_routes()
    # Guards the walker itself: an `_IncludedRouter` whose nested routes are not
    # followed yields nothing, and every assertion below would pass vacuously.
    assert len(routes) >= 25, f"router walk found only {len(routes)} routes; it is not descending"

    sample = {"{agent_id}": "11111111-2222-3333-4444-555555555555", "{job_id}": "job-1"}
    ungated: list[str] = []
    for path, methods in routes:
        concrete = path
        for token, value in sample.items():
            concrete = concrete.replace(token, value)
        concrete = concrete.replace("{variable_key}", "some-key")
        for method in methods:
            key = f"{method} {path}"
            if _legacy_desktop_gated(concrete, method):
                assert key not in _LEGACY_GATE_EXCEPTIONS, (
                    f"{key} is gated but also listed as an exception; "
                    "remove the stale entry so the list stays meaningful."
                )
            elif key not in _LEGACY_GATE_EXCEPTIONS:
                ungated.append(key)

    assert not ungated, (
        "these legacy routes answer without the desktop bearer in managed mode:\n  "
        + "\n  ".join(sorted(ungated))
        + "\nGate them, or add an entry to `_LEGACY_GATE_EXCEPTIONS` stating why "
        "the route is safe to leave open."
    )

    # Every exception must name a route that still exists, so the list cannot
    # rot into a set of keys that quietly match nothing.
    live = {f"{method} {path}" for path, methods in routes for method in methods}
    stale = set(_LEGACY_GATE_EXCEPTIONS) - live
    assert not stale, f"exception list names routes that no longer exist: {sorted(stale)}"
    assert all(reason.strip() for reason in _LEGACY_GATE_EXCEPTIONS.values())


def _apiroutes(routes: Iterable[Any]) -> Iterator[APIRoute]:
    """Every :class:`APIRoute` the app publishes, nested routers included.

    The same descent as ``app._iter_routes``, but yielding the route OBJECT:
    the question here is what dependencies a route carries, which the
    ``(path, methods)`` pair cannot answer.
    """
    for route in routes:
        nested = getattr(route, "routes", None)
        if nested is None:
            original = getattr(route, "original_router", None)
            nested = getattr(original, "routes", None) if original is not None else None
        if nested:
            yield from _apiroutes(nested)
        if isinstance(route, APIRoute):
            yield route


def _dependency_names(route: APIRoute) -> set[str]:
    """The ``__qualname__`` of every ``Depends`` a route resolves, however nested.

    Router-level dependencies (``APIRouter(dependencies=[...])``) are not copied
    onto ``route.dependencies``; they land in ``route.dependant``. The whole
    tree is walked so the assertion does not depend on which of the two
    spellings a router happened to use.
    """
    found: set[str] = set()
    stack = list(getattr(route.dependant, "dependencies", []) or [])
    while stack:
        dependency = stack.pop()
        call = getattr(dependency, "call", None)
        if call is not None:
            found.add(getattr(call, "__qualname__", repr(call)))
        stack.extend(getattr(dependency, "dependencies", []) or [])
    return found


def _control_surface_routes() -> list[tuple[str, frozenset[str]]]:
    """Every route the tenant's CONTROL-SURFACE routers publish, templates included.

    The same walk the gate uses for the prefix families, applied to the routers
    themselves. That is the point of it: a route can be published by one of these
    modules and still sit outside every gated prefix -- the hyphen in
    ``/v1/agent-name-availability`` is not a segment boundary, so the prefix test
    never saw it -- and the prefix-scoped test above cannot fail for a route it
    never enumerates.
    """
    from local_operator.server.app import _iter_routes
    from local_operator.server.routes import agents as agents_module
    from local_operator.server.routes import jobs as jobs_module
    from local_operator.server.routes import schedules as schedules_module

    routes: list[tuple[str, frozenset[str]]] = []
    for router in (agents_module.router, jobs_module.router, schedules_module.router):
        routes.extend(_iter_routes(router.routes))
    return routes


def test_managed_gate_covers_every_control_surface_route() -> None:
    """Every route the agents/jobs/schedules ROUTERS publish is gated, or excused.

    The prefix families are covered by the test above; this one covers the same
    modules' routes that fall OUTSIDE those prefixes, which is how
    ``/v1/agent-name-availability`` shipped ungated while the file's
    deny-by-default comment said a new route could not. A route published by one
    of these three modules is the tenant's data, its agent inventory or an egress
    this machine makes on a caller's behalf, so the gate must see it or a reviewer
    must write down why it does not.
    """
    from local_operator.server.app import _LEGACY_GATE_EXCEPTIONS, _legacy_desktop_gated

    routes = _control_surface_routes()
    # Guards the walk itself: an empty router list would pass vacuously.
    assert len(routes) >= 15, f"the search found only {len(routes)} routes; the walk is wrong"

    sample = {
        "{agent_id}": "11111111-2222-3333-4444-555555555555",
        "{job_id}": "job-1",
        "{schedule_id}": "schedule-1",
        "{variable_key}": "some-key",
    }
    ungated: list[str] = []
    for path, methods in routes:
        concrete = path
        for token, value in sample.items():
            concrete = concrete.replace(token, value)
        for method in methods:
            key = f"{method} {path}"
            if not _legacy_desktop_gated(concrete, method) and key not in _LEGACY_GATE_EXCEPTIONS:
                ungated.append(key)

    assert not ungated, (
        "these routes of the control-surface routers answer without the desktop "
        "bearer in managed mode:\n  "
        + "\n  ".join(sorted(ungated))
        + "\nGate them (add the path to `_LEGACY_CONTROL_PATHS` when it is a flat "
        "singleton), or add an entry to `_LEGACY_GATE_EXCEPTIONS` stating why the "
        "route is safe to leave open."
    )


def test_managed_gate_control_paths_are_live_and_gated_on_every_method() -> None:
    """Every ``_LEGACY_CONTROL_PATHS`` entry names a live route and is gated.

    The exception list has a staleness check; the singleton set had none, and an
    entry that rots (a route renamed away, a typo) would leave the boundary
    quietly wider than the comment claims. Gating is asserted for every method,
    because that set is method-agnostic on purpose: a control path does not become
    safe by being reached with a verb its route happens not to register.
    """
    from local_operator.server.app import (
        _LEGACY_CONTROL_PATHS,
        _iter_routes,
        _legacy_desktop_gated,
    )
    from local_operator.server.app import app as application

    live = {path for path, _methods in _iter_routes(application.routes)}
    stale = sorted(path for path in _LEGACY_CONTROL_PATHS if path not in live)

    assert not stale, f"control-path entries name routes that do not exist: {stale}"
    for path in sorted(_LEGACY_CONTROL_PATHS):
        for method in ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"):
            assert _legacy_desktop_gated(path, method), f"{method} {path} is not gated"


async def test_agent_name_availability_requires_the_desktop_bearer(desktop) -> None:
    """The availability route is behind the boundary, not beside it.

    Read-only, credential-free and hub-side, but it is EGRESS this machine makes
    when a page asks it to, on an app whose CORS policy allows every origin, so an
    unauthenticated caller must not be able to drive it. The request is refused by
    the middleware, before any hub client is built -- which is why this test needs
    no stub: nothing outbound happens.
    """
    client, _ = desktop

    gated = await client.get(
        "/v1/agent-name-availability?name=coder", headers={"Authorization": ""}
    )

    assert gated.status_code == 401
    assert gated.json()["detail"] == "Desktop authorization is required."


def test_managed_gate_covers_every_desktop_route() -> None:
    """Every ``/v1/desktop/*`` route carries ``require_desktop``, but one.

    The desktop plane's analogue of ``test_managed_gate_covers_every_legacy_route``
    above, and it exists for the same reason: ``/v1/desktop/claim`` is the
    first desktop route NOT behind ``require_desktop``, so the plane's "a route
    is gated the moment it exists" property now rests on the routing table
    rather than on nobody forgetting. A future ``/v1/desktop/x`` added without
    the dependency is reachable while the plane is unclaimed — the exact bug
    class this PR's thesis exists to prevent.

    ``/v1/desktop/claim`` is the named exception, and gating it would be a
    deadlock rather than a hardening: an app that cannot claim an unclaimed
    daemon cannot attach to it at all. Its admission rules live in
    ``test_desktop_claim.py`` (the key, plus the page-cannot-claim rule).

    The count assertion guards the walker itself: a descent that stopped at the
    ``_IncludedRouter`` wrappers would yield nothing and pass vacuously.
    """
    from local_operator.server.app import app

    seen = 0
    ungated: list[str] = []
    for route in _apiroutes(app.routes):
        if not route.path.startswith("/v1/desktop/"):
            continue
        seen += 1
        if route.path == "/v1/desktop/claim":
            continue
        if "require_desktop" not in _dependency_names(route):
            ungated.append(f"{sorted(route.methods or ())} {route.path}")

    assert seen >= 40, f"the route walk found only {seen} desktop routes; it is not descending"
    assert ungated == [], (
        "these desktop routes answer without `require_desktop`:\n  "
        + "\n  ".join(ungated)
        + "\nAdd the dependency, or add the route to this test's exception "
        "(the claim) with a reason."
    )


async def test_the_five_routes_round_two_found_are_gated(desktop, monkeypatch):
    """The exact requests review round 2 reproduced against a live backend.

    Named individually rather than folded into the router walk above because a
    reviewer should be able to read this file and see the reported bypasses
    closed, without re-deriving them from the routing table.
    """
    client, _ = desktop
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
    agent_id = "11111111-2222-3333-4444-555555555555"
    reads = (
        f"/v1/agents/{agent_id}/history",
        f"/v1/agents/{agent_id}/execution-variables",
        f"/v1/agents/{agent_id}/system-prompt",
        f"/v1/agents/{agent_id}/export",
        f"/v1/agents/{agent_id}/download",
        "/v1/jobs/some-job-id",
    )
    for path in reads:
        assert (await client.get(path, headers={"Authorization": ""})).status_code == 401, path
        assert (
            await client.get(path, headers={"Origin": "https://evil.example"})
        ).status_code == 403, path
        # Starlette answers HEAD from the GET route; the gate must cover it too.
        assert (await client.head(path, headers={"Authorization": ""})).status_code == 401, path

    # The WRITE. Unauthenticated and cross-origin, this renamed an agent and the
    # rename persisted.
    for headers in ({"Authorization": ""}, {"Origin": "https://evil.example"}):
        response = await client.patch(
            f"/v1/agents/{agent_id}",
            json={"name": "PWNED-by-unauthenticated-caller"},
            headers=headers,
        )
        assert response.status_code in (401, 403), response.status_code
        assert (await client.delete(f"/v1/agents/{agent_id}", headers=headers)).status_code in (
            401,
            403,
        )


async def test_unmanaged_mode_keeps_every_legacy_route_open(desktop, monkeypatch):
    """No desktop token: the CLI/script contract is unchanged on ALL of them.

    The gate widened from 4 paths to the whole agent/job surface, including
    mutating methods. That must remain invisible to a standalone legacy server,
    which is the posture every CLI client and script runs against.
    """
    client, _ = desktop
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    agent_id = "11111111-2222-3333-4444-555555555555"
    for path, method in (
        ("/v1/agents", "get"),
        ("/v1/jobs", "get"),
        (f"/v1/agents/{agent_id}", "get"),
        (f"/v1/agents/{agent_id}/history", "get"),
        (f"/v1/agents/{agent_id}/export", "get"),
        (f"/v1/agents/{agent_id}/system-prompt", "get"),
        ("/v1/jobs/some-job-id", "get"),
    ):
        response = await getattr(client, method)(path, headers={"Authorization": ""})
        assert response.status_code not in (401, 403), f"{method} {path} -> {response.status_code}"
    # Mutating methods too: these now sit on gated paths, and gating a PATH must
    # not have changed what an unmanaged server accepts.
    assert (
        await client.patch(
            f"/v1/agents/{agent_id}", json={"name": "cli-rename"}, headers={"Authorization": ""}
        )
    ).status_code not in (401, 403)
    assert (
        await client.post("/v1/agents", json={"name": "cli-created"}, headers={"Authorization": ""})
    ).status_code not in (401, 403)


async def test_legacy_reads_and_import_are_gated_in_managed_mode(desktop, monkeypatch):
    """Agent inventory, conversations, jobs and ZIP import sit behind the boundary.

    These routes disclose the same tenant's data as the control plane (names,
    working-directory paths, job history, conversation content) and were
    answering any origin without a bearer while the desktop app held the
    backend open on a predictable loopback port.
    """
    client, _ = desktop
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
    agent_id = "11111111-2222-3333-4444-555555555555"
    reads = (
        "/v1/agents",
        "/v1/jobs",
        f"/v1/agents/{agent_id}",
        f"/v1/agents/{agent_id}/conversation",
    )
    for path in reads:
        assert (await client.get(path, headers={"Authorization": ""})).status_code == 401, path
        assert (
            await client.get(path, headers={"Origin": "https://evil.example"})
        ).status_code == 403, path
    # The ungated WRITE from the same family.
    assert (
        await client.post("/v1/agents/import", headers={"Authorization": ""})
    ).status_code == 401
    assert (
        await client.post("/v1/agents/import", headers={"Origin": "https://evil.example"})
    ).status_code == 403
    # Unmanaged (no desktop token) must stay wire-compatible for CLI clients.
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    for path in reads:
        assert (await client.get(path, headers={"Authorization": ""})).status_code != 401, path


async def test_absent_origin_is_refused_only_for_browser_shaped_requests(desktop, monkeypatch):
    """An absent Origin used to skip the allowlist entirely.

    It cannot become an unconditional requirement: Electron main and the dev
    proxy both fetch server-side and legitimately send none. ``Sec-Fetch-Site``
    is attached by the browser and cannot be forged or removed by page script,
    so it is what separates the two.
    """
    client, _ = desktop
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
    # Native caller: no Origin, no fetch metadata. Must still work.
    assert (await client.get("/v1/settings")).status_code == 200
    # Browser-shaped: fetch metadata present, Origin withheld to dodge the check.
    for value in ("cross-site", "same-site", "none"):
        assert (
            await client.get("/v1/settings", headers={"Sec-Fetch-Site": value})
        ).status_code == 403, value
    # An allowed origin still passes with the metadata attached.
    assert (
        await client.get(
            "/v1/settings",
            headers={"Origin": "http://localhost:5187", "Sec-Fetch-Site": "same-origin"},
        )
    ).status_code == 200


async def test_the_cors_echo_follows_the_admitted_set(tmp_path: Path, monkeypatch):
    """The CORS echo is scoped to the ALLOWLIST IN FORCE, not to the posture.

    The app is mounted with ``allow_origins=["*"]`` and
    ``allow_credentials=True``, which makes Starlette ECHO the caller's origin
    -- turning every open route into something a drive-by page can read with
    ``fetch()`` while the desktop app holds the backend on a known loopback
    port. Missing auth was only half of QA's Q2; this is the half that made it
    browser-exploitable rather than curl-only.

    Three states, because the rule is scoped to the ADMITTED SET: unmanaged and
    allowlist-less keep the historical wildcard echo (which the packaged app's
    ``file://`` renderer reads ``/health`` with -- suppressing it there made a
    live daemon report as down); an allowlist admits exactly its own origins
    and strips the grant from every other.
    """
    from fastapi.middleware.cors import CORSMiddleware

    from local_operator.server import desktop as desktop_module
    from local_operator.server.app import desktop_origin_cors

    app = FastAPI()

    @app.get("/health")
    async def _health():
        return {"ok": True}

    # Registration ORDER is the load-bearing part: Starlette runs the most
    # recently added middleware outermost, so only a middleware added AFTER the
    # CORS one can observe (and remove) the header it wrote.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(desktop_origin_cors)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        # NO token, no claim: a standalone server keeps its historical wildcard
        # CORS, so existing embedders are untouched. The latch is cleared
        # explicitly because it is module-global and a claim made by another
        # test in this worker would otherwise leave this daemon governed.
        monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
        monkeypatch.setattr(desktop_module, "_CLAIMED", None)
        hostile = await client.get("/health", headers={"Origin": "http://evil.example"})
        assert hostile.headers.get("access-control-allow-origin") == "http://evil.example"

        # A token WITHOUT a list: the app-managed default, and the state the
        # PACKAGED app runs in. Its renderer is loaded with
        # ``mainWindow.loadFile(...)``, so it runs at ``file://``, every request
        # it makes carries the opaque origin ``"null"``, and it reads
        # ``/health`` DIRECTLY as its "server offline" signal. Suppressing the
        # echo here -- which #1093 did, by scoping to the posture instead of to
        # the admitted set -- stripped the grant from that reply and made the
        # app report a live daemon as down. So with nothing admitted the echo
        # STANDS, and the control half (``require_desktop`` on the gated
        # families) is what protects the plane in this state.
        monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
        for origin in ("http://evil.example", "null"):
            hostile = await client.get("/health", headers={"Origin": origin})
            assert hostile.status_code == 200
            assert hostile.headers.get("access-control-allow-origin") == origin, origin
            # Credentials must go with the grant, or the pair reads as a wildcard.
            assert hostile.headers.get("access-control-allow-credentials") == "true", origin

        # A configured list still admits exactly its own origin, and nothing else.
        monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
        hostile = await client.get("/health", headers={"Origin": "http://evil.example"})
        assert hostile.status_code == 200
        assert "access-control-allow-origin" not in hostile.headers
        assert "access-control-allow-credentials" not in hostile.headers

        allowed = await client.get("/health", headers={"Origin": "http://localhost:5187"})
        assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5187"


async def test_settings_census_typed_writes_reset_and_secret_exclusion(desktop):
    client, app = desktop
    manager = app.state.config_manager
    manager.set_config_value("private_secret", "never-serialize-me")
    data = (await client.get("/v1/settings")).json()["result"]
    assert {row["key"] for row in data["settings"]} == {s.key for s in settings_io.SETTINGS}
    assert "never-serialize-me" not in str(data)
    # Both sentinels are deliberately UNIQUE strings rather than ordinary
    # words. The query sentinel was `hidden`, which stopped proving anything
    # the moment an unrelated setting shipped a choice named `hidden`
    # (`display.dock`): the assertion then failed on a legitimate row while a
    # real leak of the query value would still have been reported the same
    # way. A sentinel has to be a string that can only have come from the
    # value under test.
    manager.set_config_value(
        "web_search",
        {"searxng_endpoint": "https://user:private-inline@example.org/?key=private-query"},
    )
    protected = await client.get("/v1/settings")
    assert protected.headers["cache-control"] == "no-store"
    assert "private-inline" not in protected.text
    assert "private-query" not in protected.text
    assert next(
        row
        for row in protected.json()["result"]["settings"]
        if row["key"] == "web_search.searxng_endpoint"
    )["redacted"]
    assert (
        await client.patch(
            "/v1/settings/web_search.searxng_endpoint",
            json={"value": "https://user:private@example.org"},
        )
    ).status_code == 422
    assert (
        await client.patch("/v1/settings/private_secret", json={"value": "changed"})
    ).status_code == 404
    key = "providers.anthropic.cache_ttl_1h_min_context_tokens"
    assert (await client.patch(f"/v1/settings/{key}", json={"value": 1.5})).status_code == 422
    assert (await client.patch(f"/v1/settings/{key}", json={"value": True})).status_code == 422
    response = await client.patch(f"/v1/settings/{key}", json={"value": 42})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["value"] == 42
    fresh = ConfigManager(manager.config_dir)
    setting = settings_io.resolve_key(key)
    assert setting is not None
    assert settings_io.read_setting(fresh, setting) == 42
    assert (await client.post(f"/v1/settings/{key}/reset")).json()["result"]["is_default"]
    dotted = next(
        s for s in settings_io.SETTINGS if s.is_flat_dotted and s.kind is settings_io.Kind.BOOL
    )
    assert (
        await client.patch(f"/v1/settings/{dotted.key}", json={"value": not dotted.default})
    ).status_code == 200
    fresh.reload()
    assert fresh.config.values[dotted.key] is not dotted.default
    assert fresh.config.values["private_secret"] == "never-serialize-me"


async def test_settings_projection_serves_the_registry_authored_annotations(desktop):
    """The three fields the registry authors and only the wire can carry.

    A `warning` is a consequence written for one key, `placeholder` is an example
    for a field whose shape its label cannot show, and `gated_by` names the key
    whose value decides whether another key may be edited at all. A desktop
    renderer cannot infer any of them (`kind` says how a value is edited and
    `is_default` is a value comparison), so the projection is the only way the
    surface can state a consequence, show an example, or render a gated child as
    disabled instead of saveable.

    Two properties are asserted, and the second is the one a later edit is most
    likely to break: every row carries all three KEYS (so a client may read them
    unconditionally) and their VALUES are the registry's own (so the route
    cannot grow a second opinion about a key's consequence).
    """
    client, _app = desktop
    rows = {
        row["key"]: row for row in (await client.get("/v1/settings")).json()["result"]["settings"]
    }
    assert rows, "the projection returned no rows"
    for key, row in rows.items():
        for field in ("warning", "placeholder", "gated_by"):
            assert field in row, f"{key} is missing {field}"
    for setting in settings_io.SETTINGS:
        # Named rather than indexed: a key the projection dropped would otherwise
        # fail this loop as a bare `KeyError`, which names nothing about which key
        # the registry and the wire disagree about.
        assert setting.key in rows, f"{setting.key} is not in the projection"
        row = rows[setting.key]
        assert row["warning"] == setting.warning, setting.key
        assert row["placeholder"] == setting.placeholder, setting.key
        assert row["gated_by"] == setting.gated_by, setting.key
    # The assertions above are only evidence that the fields are PLUMBED while at
    # least one key of each kind exists in the registry, so a registry that lost
    # them all would not quietly turn this into a test of empty strings.
    values = list(rows.values())
    assert any(row["warning"] for row in values), "no registered key carries a warning"
    assert any(row["placeholder"] for row in values), "no registered key carries a placeholder"
    assert any(row["gated_by"] for row in values), "no registered key is gated"


async def test_settings_cascade_preserves_concurrent_siblings(desktop):
    client, app = desktop
    manager = app.state.config_manager
    settings_io.write_chains(manager, {"primary": ["openai/gpt-5"]})
    base = settings_io.read_chains(manager)
    other = ConfigManager(manager.config_dir)
    settings_io.write_chains(other, {**base, "other": ["anthropic/claude-sonnet-4"]}, base=base)
    response = await client.patch(
        "/v1/settings/retry.fallbackChains",
        json={
            "value": {"primary": ["openai/gpt-5", "openrouter/openai/gpt-5"]},
            "base": base,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["value"]["other"] == ["anthropic/claude-sonnet-4"]
    assert (
        await client.patch("/v1/settings/retry.fallbackChains", json={"value": {}})
    ).status_code == 422


async def test_unreadable_settings_are_not_replaced_by_a_get(desktop):
    client, app = desktop
    path = app.state.config_manager.config_file
    path.write_text("\tinvalid: yaml\n")
    before = path.read_bytes()
    assert (await client.get("/v1/settings")).status_code == 409
    assert path.read_bytes() == before


async def test_provider_census_alias_storage_and_redacted_keys(desktop):
    client, app = desktop
    rows = (await client.get("/v1/auth/providers")).json()["result"]["providers"]
    assert {row["id"] for row in rows} == {
        p.id
        for p in registry.PROVIDER_REGISTRY
        if registry.credential_provider_id(p.id) == p.id and p.wire != "mock"
    }
    assert {method["id"] for row in rows for method in row["auth_methods"]} == {
        p.id for p in registry.PROVIDER_REGISTRY if p.login is not None
    }
    assert next(row for row in rows if row["id"] == "radient")["login_kind"] == "browser"
    assert next(row for row in rows if row["id"] == "openrouter")["login_kind"] == "api_key"
    assert (
        next(
            method
            for row in rows
            for method in row["auth_methods"]
            if method["id"] == "alibaba-token-plan-oauth"
        )["kind"]
        == "device"
    )
    secret = "contract-secret-never-return"
    response = await client.put("/v1/auth/providers/xai-oauth/key", json={"value": secret})
    assert response.status_code == 200, response.text
    assert secret not in response.text
    stored = app.state.desktop_auth.store.list_credentials("xai")
    assert len(stored) == 1
    assert stored[0].data["key"] == secret
    assert secret not in (await client.get("/v1/auth/providers")).text
    bad = await client.put("/v1/auth/providers/xai/key", json={"value": {"secret": secret}})
    assert bad.status_code == 422
    assert secret not in bad.text


async def test_actual_registry_key_login_input_cancel_and_persistence(desktop):
    client, app = desktop
    started = await client.post("/v1/auth/login", json={"provider": "openrouter"})
    assert started.status_code == 200, started.text
    operation_id = started.json()["result"]["id"]
    awaiting = await wait_for_state(client, operation_id, "input_required")
    assert awaiting["input_required"]
    # A second start no longer answers 409: it SUPERSEDES the active operation
    # (see `test_a_new_sign_in_supersedes_a_leftover_one`), so that case is not
    # exercised here -- it would cancel the operation this test goes on to finish.
    secret = "registry-login-secret"
    response = await client.post(
        f"/v1/auth/operations/{operation_id}/input",
        json={"value": secret, "prompt_id": awaiting["prompt_id"]},
    )
    assert response.status_code == 200
    done = await wait_for_state(client, operation_id, "succeeded")
    assert secret not in str(done)
    assert done["auth_url"] is None
    assert app.state.desktop_auth.store.list_credentials("openrouter")[0].data["key"] == secret
    assert (
        await client.post(
            f"/v1/auth/operations/{operation_id}/input",
            json={"value": secret, "prompt_id": awaiting["prompt_id"]},
        )
    ).status_code == 409
    again = (await client.post("/v1/auth/login", json={"provider": "openrouter"})).json()["result"][
        "id"
    ]
    await wait_for_state(client, again, "input_required")
    assert (await client.delete(f"/v1/auth/operations/{again}")).json()["result"][
        "state"
    ] == "cancelled"
    assert (await client.delete("/v1/auth/providers/openrouter/credentials")).status_code == 200
    assert not app.state.desktop_auth.store.list_credentials("openrouter")


async def test_oauth_failure_is_redacted_and_browser_opener_is_per_flow(desktop, monkeypatch):
    client, _ = desktop
    observed = []

    # ``**_kwargs``: the controller passes ``signal=`` to every login callable
    # so a host can cancel a pending flow. Without it this double raises
    # TypeError before recording anything, and the assertion below reads as
    # "the browser opener was never injected".
    async def login(callbacks, *, open_browser, **_kwargs):
        observed.append(open_browser)
        await asyncio.sleep(0)
        raise RuntimeError("raw-provider-access-token-do-not-return")

    definition = registry.get_provider_definition("radient")
    assert definition is not None
    monkeypatch.setitem(registry._BY_ID, "radient", dataclasses.replace(definition, login=login))
    operation_id = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()[
        "result"
    ]["id"]
    done = await wait_for_state(client, operation_id, "failed")
    assert observed
    assert "raw-provider-access-token" not in str(done)


async def test_round_three_schedules_are_gated_and_no_row_is_created(desktop, monkeypatch):
    """The schedules surface: refused, and nothing persisted as a side effect.

    Round 3 reproduced an unauthenticated cross-origin ``POST`` to
    ``/v1/agents/{id}/schedules`` returning 201, with an authenticated ``GET
    /v1/schedules`` reading the attacker's prompt back with ``is_active`` true.
    That is worse than the round-2 agent rename: ``create_schedule_for_agent``
    calls ``add_or_update_job``, so the stored text is later executed by the
    user's own agent with its tools and credentials. The whole family is now
    gated, including the ``PATCH`` that reaches the same execution by rewriting
    an existing job's prompt without creating anything.

    The status code alone is NOT the assertion. A route that returns 403 after
    writing the row is still exploited, so this reads the surface back through
    the AUTHENTICATED client and asserts the absence of the attacker's text.
    """
    client, _ = desktop
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
    agent_id = (await client.post("/v1/agents", json={"name": "r3-victim"})).json()["result"]["id"]
    seeded = (
        await client.post(
            f"/v1/agents/{agent_id}/schedules",
            json={"prompt": "LEGIT-SEED", "interval": 1, "unit": "days", "is_active": True},
        )
    ).json()["result"]["id"]

    payload = {
        "prompt": "EXFIL-BY-UNAUTH-CALLER",
        "interval": 1,
        "unit": "days",
        "is_active": True,
    }
    for headers in ({"Authorization": ""}, {"Origin": "https://evil.example"}):
        # The blocker: the only create on the whole schedules surface.
        assert (
            await client.post(f"/v1/agents/{agent_id}/schedules", json=payload, headers=headers)
        ).status_code in (401, 403), headers
        # The major: rewriting an executing job's prompt reaches the same
        # autonomous execution without creating anything.
        assert (
            await client.patch(
                f"/v1/schedules/{seeded}",
                json={"prompt": "EXFIL-BY-UNAUTH-PATCH"},
                headers=headers,
            )
        ).status_code in (401, 403), headers
        for path in (
            f"/v1/agents/{agent_id}/schedules",
            "/v1/schedules",
            f"/v1/schedules/{seeded}",
        ):
            assert (await client.get(path, headers=headers)).status_code in (401, 403), path
            # Starlette answers HEAD from the GET route.
            assert (await client.head(path, headers=headers)).status_code in (401, 403), path
        assert (await client.delete(f"/v1/schedules/{seeded}", headers=headers)).status_code in (
            401,
            403,
        ), headers

    # The side effect, read back through the authenticated client: no row was
    # created, and the seeded row still carries its original prompt.
    rows = (await client.get("/v1/schedules", params={"per_page": 100})).json()["result"][
        "schedules"
    ]
    prompts = [row["prompt"] for row in rows]
    assert not [p for p in prompts if "EXFIL-BY-UNAUTH" in p], prompts
    assert prompts == ["LEGIT-SEED"], prompts
    # The DELETE was refused too, so the legitimate schedule is still there.
    assert (await client.get(f"/v1/schedules/{seeded}")).status_code == 200


async def test_unmanaged_mode_keeps_the_schedules_surface_open(desktop, monkeypatch):
    """Gating the schedules family must be invisible without a desktop token.

    Same contract as ``test_unmanaged_mode_keeps_every_legacy_route_open``: the
    CLI and every script client run against a standalone server, and a PATH
    becoming gated must not change what that server accepts.
    """
    client, _ = desktop
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    bare = {"Authorization": ""}
    agent_id = (await client.post("/v1/agents", json={"name": "cli-victim"}, headers=bare)).json()[
        "result"
    ]["id"]
    created = await client.post(
        f"/v1/agents/{agent_id}/schedules",
        json={"prompt": "cli-scheduled", "interval": 1, "unit": "days", "is_active": True},
        headers=bare,
    )
    assert created.status_code not in (401, 403), created.status_code
    schedule_id = created.json()["result"]["id"]
    for path, method in (
        ("/v1/schedules", "get"),
        (f"/v1/agents/{agent_id}/schedules", "get"),
        (f"/v1/schedules/{schedule_id}", "get"),
    ):
        response = await getattr(client, method)(path, headers=bare)
        assert response.status_code not in (401, 403), f"{method} {path} -> {response.status_code}"
    assert (
        await client.patch(
            f"/v1/schedules/{schedule_id}", json={"prompt": "cli-edited"}, headers=bare
        )
    ).status_code not in (401, 403)
    assert (await client.delete(f"/v1/schedules/{schedule_id}", headers=bare)).status_code not in (
        401,
        403,
    )


# ---------------------------------------------------------------------------
# Suggested models, first-run defaults and key validation on the desktop routes
# ---------------------------------------------------------------------------


async def test_provider_census_carries_the_suggested_model(desktop):
    """The renderer shows "Suggested: ..." from THIS field; it carries no table."""
    client, _ = desktop
    rows = {
        row["id"]: row
        for row in (await client.get("/v1/auth/providers")).json()["result"]["providers"]
    }
    assert rows["anthropic"]["suggested_model"] == {
        "id": "claude-opus-5-5",
        "name": "Claude Opus 5.5",
    }
    assert rows["deepseek"]["suggested_model"]["id"] == "deepseek-flash"
    assert rows["ollama"]["suggested_model"] is None
    assert rows["typesafe"]["suggested_model"] is None
    # Per method, because the route decides the spelling (Kimi).
    kimi = {m["method_id"]: m["suggested_model"]["id"] for m in rows["kimi"]["auth_methods"]}
    assert kimi == {"kimi": "k3", "kimi:api-key": "kimi-k3"}


async def test_key_save_on_an_empty_config_sets_the_suggested_default(desktop):
    client, app = desktop
    response = await client.put(
        "/v1/auth/providers/deepseek/key", json={"value": "sk-first-run-secret"}
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["valid"] is None  # the conftest stubs the live check
    assert result["defaults_applied"] == {
        "hosting": "deepseek",
        "model": "deepseek-flash",
        "model_name": "DeepSeek Flash",
        "receipt": "Set default hosting to 'deepseek', model to 'deepseek-flash'.",
    }
    assert "sk-first-run-secret" not in response.text
    # The side effect, read from disk through a fresh manager.
    on_disk = ConfigManager(app.state.config_manager.config_dir)
    assert on_disk.get_config_value("hosting") == "deepseek"
    assert on_disk.get_config_value("model_name") == "deepseek-flash"


async def test_key_save_leaves_an_existing_working_choice_alone(desktop):
    client, app = desktop
    app.state.config_manager.set_config_value("hosting", "anthropic")
    app.state.config_manager.set_config_value("model_name", "claude-sonnet-5")
    response = await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-second"})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["defaults_applied"] is None
    on_disk = ConfigManager(app.state.config_manager.config_dir)
    assert on_disk.get_config_value("hosting") == "anthropic"
    assert on_disk.get_config_value("model_name") == "claude-sonnet-5"


async def test_key_save_fills_an_empty_model_for_the_configured_provider(desktop):
    client, app = desktop
    app.state.config_manager.set_config_value("hosting", "deepseek")
    app.state.config_manager.set_config_value("model_name", "")
    response = await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-fill"})
    applied = response.json()["result"]["defaults_applied"]
    assert applied["hosting"] == "deepseek" and applied["model"] == "deepseek-flash"
    assert ConfigManager(app.state.config_manager.config_dir).get_config_value("model_name") == (
        "deepseek-flash"
    )


async def test_key_save_reads_config_written_elsewhere_since_boot(desktop):
    """The server's manager is re-read: a hosting chosen by the TUI after this
    server booted must not be treated as empty (and then overwritten)."""
    client, app = desktop
    ConfigManager(app.state.config_manager.config_dir).set_config_value("hosting", "anthropic")
    response = await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-elsewhere"})
    assert response.json()["result"]["defaults_applied"] is None
    assert ConfigManager(app.state.config_manager.config_dir).get_config_value("hosting") == (
        "anthropic"
    )


async def test_a_rejected_key_is_refused_and_not_stored(desktop, monkeypatch):
    from local_operator.providers import key_check

    async def rejected(_provider, _key, **_kwargs):
        return key_check.KeyCheck(False, "DeepSeek rejected this API key. Check it and try again.")

    monkeypatch.setattr(key_check, "check_api_key", rejected)
    client, app = desktop
    response = await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-bad-key"})
    assert response.status_code == 422
    assert response.json()["detail"] == "DeepSeek rejected this API key. Check it and try again."
    assert "sk-bad-key" not in response.text
    assert not app.state.desktop_auth.store.list_credentials("deepseek")
    # No defaults either: nothing usable was connected.
    assert not ConfigManager(app.state.config_manager.config_dir).get_config_value("hosting")


async def test_a_verified_key_reports_valid(desktop, monkeypatch):
    from local_operator.providers import key_check

    async def accepted(_provider, _key, **_kwargs):
        return key_check.KeyCheck(True, None)

    monkeypatch.setattr(key_check, "check_api_key", accepted)
    client, app = desktop
    response = await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-good"})
    result = response.json()["result"]
    assert (result["valid"], result["reason"]) == (True, None)
    assert app.state.desktop_auth.store.list_credentials("deepseek")


def _stub_login(monkeypatch, provider: str, login):
    definition = registry.get_provider_definition(provider)
    assert definition is not None
    monkeypatch.setitem(registry._BY_ID, provider, dataclasses.replace(definition, login=login))


async def test_the_first_login_reply_carries_the_auth_url(desktop, monkeypatch):
    """The reported bug: the browser never opened until "reopen", because the
    first reply was snapshotted before the flow ran a step (auth_url null)."""
    client, _ = desktop
    release = asyncio.Event()

    async def login(callbacks, *, signal=None, **_kwargs):
        await asyncio.sleep(0.01)  # a real flow binds a port and builds PKCE first
        from local_operator.providers.oauth.callback_server import report_flow_details

        await report_flow_details(
            callbacks, launch_url="http://localhost:54549/launch", expires_in=300
        )
        callbacks.on_auth_url("https://radienthq.com/authorize?x=1", instructions=None)
        await release.wait()
        return {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}

    _stub_login(monkeypatch, "radient", login)
    started = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()["result"]
    assert started["state"] == "waiting"
    assert started["auth_url"] == "https://radienthq.com/authorize?x=1"
    assert started["launch_url"] == "http://localhost:54549/launch"
    # The flow's own deadline, not the host's 900 s cap.
    assert 290 <= started["expires_in"] <= 300
    assert started["input_optional"] is False
    release.set()
    done = await wait_for_state(client, started["id"], "succeeded")
    assert done["auth_url"] is None and done["launch_url"] is None


async def test_a_device_flow_publishes_its_user_code(desktop, monkeypatch):
    client, _ = desktop

    async def login(callbacks, *, signal=None, **_kwargs):
        from local_operator.providers.oauth.callback_server import report_flow_details

        await report_flow_details(callbacks, user_code="WXYZ-1234", expires_in=600)
        callbacks.on_auth_url("https://auth.x.ai/device", instructions="Enter code: WXYZ-1234")
        await asyncio.Event().wait()

    _stub_login(monkeypatch, "xai-oauth", login)
    started = (await client.post("/v1/auth/login", json={"provider": "xai-oauth"})).json()["result"]
    assert started["user_code"] == "WXYZ-1234"
    # Kept for compatibility with renderers that print it.
    assert started["instructions"] == "Enter code: WXYZ-1234"
    assert 590 <= started["expires_in"] <= 600


async def test_the_flow_timeout_reads_expired_not_failed(desktop, monkeypatch):
    client, _ = desktop

    async def login(callbacks, *, signal=None, **_kwargs):
        from local_operator.providers.oauth.callback_server import LoginTimeoutError

        raise LoginTimeoutError()

    _stub_login(monkeypatch, "radient", login)
    operation_id = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()[
        "result"
    ]["id"]
    done = await wait_for_state(client, operation_id, "expired", "failed")
    assert done["state"] == "expired"
    assert "expired" in done["message"].lower()


async def test_a_new_sign_in_supersedes_a_leftover_one(desktop, monkeypatch):
    """A 409 here blocked exactly the retry that was the user's way out."""
    client, _ = desktop

    async def login(callbacks, *, signal=None, **_kwargs):
        callbacks.on_auth_url("https://radienthq.com/authorize", instructions=None)
        await asyncio.Event().wait()

    _stub_login(monkeypatch, "radient", login)
    first = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()["result"]
    second = await client.post("/v1/auth/login", json={"provider": "radient"})
    assert second.status_code == 200, second.text
    assert second.json()["result"]["id"] != first["id"]
    old = (await client.get(f"/v1/auth/operations/{first['id']}")).json()["result"]
    assert old["state"] == "cancelled"
    assert old["message"] == "Replaced by a new sign-in."
    # A DIFFERENT provider supersedes too; only one flow is ever live.
    third = await client.post("/v1/auth/login", json={"provider": "openrouter"})
    assert third.status_code == 200
    replaced = (await client.get(f"/v1/auth/operations/{second.json()['result']['id']}")).json()[
        "result"
    ]
    assert replaced["state"] == "cancelled"


async def test_an_optional_paste_keeps_the_browser_primary(desktop, monkeypatch):
    """Anthropic: the paste box is a fallback; the copy must lead with the browser."""
    client, _ = desktop

    async def login(callbacks, *, signal=None, **_kwargs):
        callbacks.on_auth_url("https://claude.ai/oauth/authorize", instructions=None)
        await callbacks.on_manual_code_input()
        await asyncio.Event().wait()

    _stub_login(monkeypatch, "anthropic", login)
    started = (await client.post("/v1/auth/login", json={"provider": "anthropic"})).json()["result"]
    assert started["input_optional"] is True
    waiting = await wait_for_state(client, started["id"], "waiting")
    for _ in range(200):
        waiting = (await client.get(f"/v1/auth/operations/{started['id']}")).json()["result"]
        if waiting["input_required"]:
            break
        await asyncio.sleep(0)
    assert waiting["input_required"] is True
    assert waiting["state"] == "waiting"
    assert waiting["message"].startswith("Finish signing in in your browser")
    assert "Paste the key" not in waiting["message"]


async def test_oauth_success_applies_and_reports_the_suggested_default(desktop, monkeypatch):
    client, app = desktop

    async def login(callbacks, *, signal=None, **_kwargs):
        callbacks.on_auth_url("https://claude.ai/oauth/authorize", instructions=None)
        return {"type": "oauth", "access": "at", "refresh": "rt", "expires": 1}

    _stub_login(monkeypatch, "anthropic", login)
    started = (await client.post("/v1/auth/login", json={"provider": "anthropic"})).json()["result"]
    done = await wait_for_state(client, started["id"], "succeeded")
    assert done["defaults_applied"] == {
        "hosting": "anthropic",
        "model": "claude-opus-5-5",
        "model_name": "Claude Opus 5.5",
        "receipt": "Set default hosting to 'anthropic', model to 'claude-opus-5-5'.",
    }
    on_disk = ConfigManager(app.state.config_manager.config_dir)
    assert on_disk.get_config_value("hosting") == "anthropic"
    assert on_disk.get_config_value("model_name") == "claude-opus-5-5"


async def test_concurrent_starts_leave_exactly_one_live_flow(desktop, monkeypatch):
    """Review round 1, #2: superseding awaits the old flow's teardown, and two
    starts interleaving across that await both created a flow. The teardown here
    takes real time, like a loopback server closing, which is what opened the
    window."""
    client, app = desktop
    live = 0
    peak = 0

    async def login(callbacks, *, signal=None, **_kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            callbacks.on_auth_url("https://radienthq.com/authorize", instructions=None)
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.05)
            live -= 1

    _stub_login(monkeypatch, "radient", login)
    # One start first, so both concurrent ones have an active op to supersede --
    # the interleaving the reviewer measured.
    await client.post("/v1/auth/login", json={"provider": "radient"})
    replies = await asyncio.gather(
        client.post("/v1/auth/login", json={"provider": "radient"}),
        client.post("/v1/auth/login", json={"provider": "radient"}),
        client.post("/v1/auth/login", json={"provider": "radient"}),
    )
    assert all(reply.status_code == 200 for reply in replies)
    running = [
        op for op in app.state.desktop_auth.operations.values() if op.task and not op.task.done()
    ]
    assert len(running) == 1
    assert live == 1
    assert peak == 1, "two flows were live at once"


async def _real_callback_flow(callbacks, *, signal=None, timeout=300.0, open_browser=None):
    """A REAL ``OAuthCallbackFlow`` (loopback server, paste race and all) whose
    only fake is the token exchange -- so the paste prompt is cancelled exactly
    as production cancels it, without being awaited."""
    from local_operator.providers.oauth.callback_server import (
        CallbackFlowOptions,
        OAuthCallbackFlow,
    )

    class _Flow(OAuthCallbackFlow):
        async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
            return f"https://claude.ai/oauth/authorize?state={state}&redirect_uri={redirect_uri}"

        async def exchange_token(self, code, state, redirect_uri):
            return {"type": "oauth", "access": "at", "refresh": "rt", "expires": 1}

    # Port 0: an OS-assigned port, so this never fights the real 54545.
    flow = _Flow(
        CallbackFlowOptions(preferred_port=0, timeout_seconds=timeout),
        callbacks,
        open_browser=open_browser or (lambda _url: None),
        signal=signal,
    )
    return await flow.run()


@pytest.mark.parametrize(
    ("ending", "terminal"),
    [("timeout", "expired"), ("denied", "failed"), ("callback", "succeeded")],
)
async def test_an_optional_paste_sign_in_ends_in_its_terminal_state(
    desktop, monkeypatch, ending, terminal
):
    """QA round 1, Q1: when the flow ended, it cancelled the paste prompt's task
    without awaiting it, and that task's ``finally`` then reset the op to
    ``waiting`` -- so an expired or failed Anthropic/Z.AI sign-in read as live
    forever. Each ending must stay the terminal state it reached.

    The ``callback`` arm needs one more step to be able to fail (review round 2,
    NIT 2): in production the defaults write runs through ``asyncio.to_thread``,
    whose yield lets the prompt's ``finally`` run BEFORE ``succeeded`` is written,
    so the old unconditional reset was accidentally harmless there. Making that
    write non-yielding puts the late ``finally`` after the terminal state, which
    is the ordering the fix must survive -- any scheduling change that removes
    the yield would otherwise reintroduce the bug with this test still green."""
    client, _ = desktop
    if ending == "callback":
        from local_operator.server.utils import desktop_auth

        async def inline(fn, /, *args, **kwargs):
            return fn(*args, **kwargs)

        monkeypatch.setattr(desktop_auth.asyncio, "to_thread", inline)

    async def login(callbacks, *, signal=None, **_kwargs):
        timeout = 0.2 if ending == "timeout" else 30.0
        return await _real_callback_flow(callbacks, signal=signal, timeout=timeout)

    _stub_login(monkeypatch, "anthropic", login)
    started = (await client.post("/v1/auth/login", json={"provider": "anthropic"})).json()["result"]
    assert started["input_optional"] is True
    # The prompt is open: this is the state whose `finally` used to clobber.
    snapshot: dict[str, Any] = {}
    for _ in range(500):
        snapshot = (await client.get(f"/v1/auth/operations/{started['id']}")).json()["result"]
        if snapshot["input_required"]:
            break
        await asyncio.sleep(0)
    assert snapshot["input_required"] is True
    if ending != "timeout":
        from urllib.parse import parse_qs, urlsplit

        query = parse_qs(urlsplit(started["auth_url"]).query)
        redirect, state = query["redirect_uri"][0], query["state"][0]
        params = (
            {"error": "access_denied", "state": state}
            if ending == "denied"
            else {"code": "c0de", "state": state}
        )
        async with AsyncClient() as loopback:
            await loopback.get(redirect, params=params)
    done = None
    for _ in range(400):
        done = (await client.get(f"/v1/auth/operations/{started['id']}")).json()["result"]
        if done["state"] in ("expired", "failed", "succeeded", "cancelled"):
            break
        await asyncio.sleep(0.01)
    assert done is not None and done["state"] == terminal, done
    # Let every cancelled prompt task run its `finally`, then look again: the
    # regression was a LATE overwrite, so the first terminal read proves nothing.
    for _ in range(20):
        await asyncio.sleep(0)
    after = (await client.get(f"/v1/auth/operations/{started['id']}")).json()["result"]
    assert after["state"] == terminal
    assert after["input_required"] is False


async def test_the_desktop_never_opens_a_system_browser_for_any_provider(desktop, monkeypatch):
    """QA round 1, Q3: the registry forwarded the desktop's no-op opener to
    Anthropic and OpenAI only, so Z.AI and Radient fell back to
    ``webbrowser.open`` and the BACKEND opened a second tab beside the
    renderer's. Every callback provider, through the real registry thunk and the
    real flow constructor; only ``run`` is replaced, so no network is used."""
    import webbrowser

    from local_operator.providers.oauth import callback_server

    system_opens: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: system_opens.append(url))

    async def run(self):
        # What the real `run` does with the opener, minus the network.
        self._open_browser("https://provider.invalid/authorize")
        return {"type": "oauth", "access": "at", "refresh": "rt", "expires": 1}

    monkeypatch.setattr(callback_server.OAuthCallbackFlow, "run", run)
    client, _ = desktop
    browser_providers = [
        p.id for p in registry.PROVIDER_REGISTRY if p.login is not None and p.callback_port
    ]
    assert {"anthropic", "openai", "zai-oauth", "radient"} <= set(browser_providers)
    for provider in browser_providers:
        started = (await client.post("/v1/auth/login", json={"provider": provider})).json()[
            "result"
        ]
        done = await wait_for_state(client, started["id"], "succeeded", "failed")
        assert done["state"] == "succeeded", (provider, done)
    assert system_opens == []


async def test_every_browser_login_forwards_the_callers_opener():
    """The same seam from the other side: the opener a host passes is the one
    the flow calls, for every callback provider (a name list dropped two)."""
    from local_operator.providers.oauth import callback_server
    from local_operator.providers.oauth.callback_server import LoginCallbacks

    original = callback_server.OAuthCallbackFlow.run

    async def run(self):
        self._open_browser("https://provider.invalid/authorize")
        return {}

    callback_server.OAuthCallbackFlow.run = run  # type: ignore[method-assign]
    try:
        for provider in registry.PROVIDER_REGISTRY:
            if provider.login is None or not provider.callback_port:
                continue
            opened: list[str] = []
            await provider.login(LoginCallbacks(), signal=None, open_browser=opened.append)
            assert opened == ["https://provider.invalid/authorize"], provider.id
    finally:
        callback_server.OAuthCallbackFlow.run = original  # type: ignore[method-assign]


async def test_a_pasted_key_in_a_login_operation_is_checked_like_a_saved_one(desktop, monkeypatch):
    """QA round 1, Q2: ``POST /v1/auth/login`` + ``/input`` stored a fake key
    unchecked. It now runs the same check as ``PUT .../key``; a rejection is a
    422 that leaves the prompt open for a corrected paste."""
    from local_operator.providers import key_check

    verdicts = {"sk-bad": key_check.KeyCheck(False, "OpenRouter rejected this API key.")}
    checked: list[str] = []

    async def check(provider, key, **_kwargs):
        checked.append(provider)
        return verdicts.get(key, key_check.KeyCheck(True, None))

    monkeypatch.setattr(key_check, "check_api_key", check)
    client, app = desktop
    operation_id = (await client.post("/v1/auth/login", json={"provider": "openrouter"})).json()[
        "result"
    ]["id"]
    awaiting = await wait_for_state(client, operation_id, "input_required")
    refused = await client.post(
        f"/v1/auth/operations/{operation_id}/input",
        json={"value": "sk-bad", "prompt_id": awaiting["prompt_id"]},
    )
    assert refused.status_code == 422
    assert refused.json()["detail"] == "OpenRouter rejected this API key."
    assert "sk-bad" not in refused.text
    assert not app.state.desktop_auth.store.list_credentials("openrouter")
    still = (await client.get(f"/v1/auth/operations/{operation_id}")).json()["result"]
    assert still["state"] == "input_required" and still["prompt_id"] == awaiting["prompt_id"]
    accepted = await client.post(
        f"/v1/auth/operations/{operation_id}/input",
        json={"value": "sk-good", "prompt_id": awaiting["prompt_id"]},
    )
    assert accepted.status_code == 200
    await wait_for_state(client, operation_id, "succeeded")
    assert app.state.desktop_auth.store.list_credentials("openrouter")[0].data["key"] == "sk-good"
    assert checked == ["openrouter", "openrouter"]


async def test_an_oauth_paste_is_not_sent_to_the_key_check(desktop, monkeypatch):
    """The check is for KEYS: an authorization code pasted into Anthropic's
    fallback box is not an API key and must never be sent to a provider as one."""
    from local_operator.providers import key_check

    checked: list[str] = []

    async def check(provider, key, **_kwargs):
        checked.append(provider)
        return key_check.KeyCheck(False, "should not run")

    monkeypatch.setattr(key_check, "check_api_key", check)

    async def login(callbacks, *, signal=None, **_kwargs):
        callbacks.on_auth_url("https://claude.ai/oauth/authorize", instructions=None)
        pasted = await callbacks.on_manual_code_input()
        assert pasted == "code#state"
        return {"type": "oauth", "access": "at", "refresh": "rt", "expires": 1}

    _stub_login(monkeypatch, "anthropic", login)
    client, _ = desktop
    started = (await client.post("/v1/auth/login", json={"provider": "anthropic"})).json()["result"]
    snapshot: dict[str, Any] = {}
    for _ in range(200):
        snapshot = (await client.get(f"/v1/auth/operations/{started['id']}")).json()["result"]
        if snapshot["input_required"]:
            break
        await asyncio.sleep(0)
    reply = await client.post(
        f"/v1/auth/operations/{started['id']}/input",
        json={"value": "code#state", "prompt_id": snapshot["prompt_id"]},
    )
    assert reply.status_code == 200
    await wait_for_state(client, started["id"], "succeeded")
    assert checked == []


#: Every login whose paste prompt reads something that is NOT an API key, and
#: why. The walk below fails for any paste-accepting provider that is neither
#: checked nor named here, so a new provider cannot slip in unchecked.
_PASTE_NOT_A_KEY = {
    "anthropic": "the fallback box reads an OAuth authorization code (code#state)",
    "zai-oauth": "the fallback box reads an OAuth authorization code or redirect URL",
}


def test_every_paste_prompt_is_classified_for_the_key_check():
    """Review round 2, MAJOR 1: the key check was keyed on the login FLAVOUR, so
    ``alibaba-token-plan-oauth`` -- a device login whose prompt reads the
    ``sk-sp-`` key -- was never checked. Walk every provider a host offers a
    prompt for and require an explicit answer: checked as a key, or exempt with
    a reason."""
    offered = [p for p in registry.PROVIDER_REGISTRY if p.login and p.accepts_paste_prompt]
    required = {p.id for p in offered if p.paste_prompt_required}
    assert "alibaba-token-plan-oauth" in required
    unclassified = [
        p.id for p in offered if not p.paste_is_api_key and p.id not in _PASTE_NOT_A_KEY
    ]
    assert unclassified == [], f"paste prompts with no key-check decision: {unclassified}"
    # An exemption is only honest while it is true.
    for provider_id in _PASTE_NOT_A_KEY:
        definition = registry.get_provider_definition(provider_id)
        assert definition is not None and not definition.paste_is_api_key, provider_id
    # Every REQUIRED prompt in the registry today reads a key (the flavour gate
    # checked all but one of them); a required non-key prompt must be a
    # deliberate addition to the exemptions, never a silent default.
    assert {p.id for p in offered if p.paste_is_api_key} >= required


@pytest.mark.parametrize(
    "provider_id",
    sorted(p.id for p in registry.PROVIDER_REGISTRY if p.login and p.paste_is_api_key),
)
async def test_a_rejected_key_pasted_into_any_key_prompt_is_refused(
    desktop, monkeypatch, provider_id
):
    """The route-level half of the walk, through each provider's REAL login: a
    definite rejection answers 422, stores nothing, and leaves the same prompt
    open. ``alibaba-token-plan-oauth`` is the row the flavour gate let through;
    its login reads the key before any network step, so no request is made."""
    from local_operator.providers import key_check

    checked: list[str] = []

    async def check(provider, key, **_kwargs):
        checked.append(provider)
        return key_check.KeyCheck(False, "rejected by the test")

    monkeypatch.setattr(key_check, "check_api_key", check)
    client, app = desktop
    started = (await client.post("/v1/auth/login", json={"provider": provider_id})).json()
    operation_id = started["result"]["id"]
    awaiting = await wait_for_state(client, operation_id, "input_required")
    refused = await client.post(
        f"/v1/auth/operations/{operation_id}/input",
        json={"value": "sk-sp-fake", "prompt_id": awaiting["prompt_id"]},
    )
    assert refused.status_code == 422, refused.text
    assert checked == [provider_id]
    storage = registry.credential_provider_id(provider_id)
    assert not app.state.desktop_auth.store.list_credentials(storage)
    still = (await client.get(f"/v1/auth/operations/{operation_id}")).json()["result"]
    assert still["state"] == "input_required"
    assert still["prompt_id"] == awaiting["prompt_id"]
    await client.delete(f"/v1/auth/operations/{operation_id}")


def _port_is_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


async def test_a_burst_of_starts_leaves_no_fixed_callback_port_bound(desktop, monkeypatch):
    """QA round 2, Q1: a burst of simultaneous starts could leave a superseded
    flow's listener on the fixed callback port with no live operation, and every
    later sign-in then advertised an ephemeral redirect port. The REAL Radient
    flow (its fixed port moved to a free one so this never touches 54549), four
    starts at once, several rounds; after each round the one live flow must own
    the fixed port, and once everything is cancelled nothing may hold it."""
    import socket
    from urllib.parse import parse_qs, urlsplit

    from local_operator.providers.oauth import radient

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        fixed = int(probe.getsockname()[1])
    monkeypatch.setattr(radient, "CALLBACK_PORT", fixed)
    client, app = desktop

    async def start_after(turns: int):
        for _ in range(turns):
            await asyncio.sleep(0)
        return await client.post("/v1/auth/login", json={"provider": "radient"})

    # Staggered by 0..N loop turns: a supersede must eventually land on every
    # step of the old flow's bind, including the one yield that used to orphan
    # its listener. A plain simultaneous burst hits it only by scheduling luck.
    for _round in range(24):
        replies = await asyncio.gather(
            start_after(0), start_after(_round), start_after(_round + 1), start_after(_round * 2)
        )
        assert all(reply.status_code == 200 for reply in replies)
        host = app.state.desktop_auth
        running = [op for op in host.operations.values() if op.task and not op.task.done()]
        assert len(running) == 1
        live = await wait_for_state(client, running[0].id, "waiting")
        redirect = parse_qs(urlsplit(live["auth_url"]).query)["redirect_uri"][0]
        assert urlsplit(redirect).port == fixed, "the live flow lost the fixed port"
        for op in list(host.operations.values()):
            await host.cancel(op)
        assert _port_is_free(fixed), f"round {_round}: the fixed port is still bound"


async def test_a_hung_teardown_does_not_block_later_starts(desktop, monkeypatch):
    """Review round 2, MINOR 2: ``start`` holds its lock across the cancel of the
    op it supersedes, so a teardown that never returns used to wedge EVERY later
    start. The first flow here ignores its cancellation until released; a second
    and a third start must still complete, each within the teardown bound."""
    from local_operator.server.utils import desktop_auth

    monkeypatch.setattr(desktop_auth, "CANCEL_TEARDOWN_TIMEOUT_S", 0.2)
    release = asyncio.Event()
    calls = 0

    async def login(callbacks, *, signal=None, **_kwargs):
        nonlocal calls
        calls += 1
        hangs = calls == 1
        callbacks.on_auth_url("https://radienthq.com/authorize", instructions=None)
        try:
            await asyncio.Event().wait()
        finally:
            if hangs:
                # A teardown that does not return: absorb the cancellation and
                # keep waiting, as a stuck provider flow would.
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue

    _stub_login(monkeypatch, "radient", login)
    client, app = desktop
    first = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()["result"]
    try:
        second, third = await asyncio.wait_for(
            asyncio.gather(
                client.post("/v1/auth/login", json={"provider": "radient"}),
                client.post("/v1/auth/login", json={"provider": "radient"}),
            ),
            timeout=5.0,
        )
        assert second.status_code == third.status_code == 200
        host = app.state.desktop_auth
        stale = host.operations[first["id"]]
        assert stale.state == "cancelled"
        assert stale.task is not None and not stale.task.done(), "the stuck flow should linger"
        assert stale.task in host._lingering
        # Exactly one of the two newer starts is live (whichever took the lock
        # last); the stuck one is the only other unfinished task.
        newer = {second.json()["result"]["id"], third.json()["result"]["id"]}
        running = {
            op.id
            for op in host.operations.values()
            if op.task and not op.task.done() and op.id != first["id"]
        }
        assert len(running) == 1 and running <= newer
        # And the abandoned flow is not charged to later starts: a fourth start
        # supersedes only the live one. Asserted structurally, not on a clock
        # (review round 3, MINOR 3; AGENTS.md "Wait on the event"): the stuck
        # task was cancelled exactly ONCE, by the second start. A re-cancel --
        # which is what re-charges the teardown bound -- would make it two.
        assert stale.task.cancelling() == 1
        fourth = await client.post("/v1/auth/login", json={"provider": "radient"})
        assert fourth.status_code == 200
        assert stale.task.cancelling() == 1, "a later start re-cancelled the lingering flow"
        # And neither may a repeated DELETE on it (review round 3, MINOR 2).
        deleted = await client.delete(f"/v1/auth/operations/{first['id']}")
        assert deleted.status_code == 200
        assert stale.task.cancelling() == 1, "DELETE re-cancelled the lingering flow"
    finally:
        release.set()
        await asyncio.sleep(0)


@pytest.mark.parametrize("first", ["start", "close"])
async def test_close_racing_a_start_leaves_no_flow_running(desktop, monkeypatch, first):
    """Review round 2, MINOR 3: ``close`` did not take the start lock, so a start
    that created its op between ``close``'s iteration and its ``clear()`` left a
    flow running past ``store.close()``. Now a start either lands before close
    (and is cancelled by it) or after (and is refused) -- never in between.

    ``first="start"`` is the ordering the LOCK guards: the start is mid-way
    through superseding the existing op (awaiting its teardown) when close
    begins, so it has already passed the closed check. ``first="close"`` is the
    ordering the closed flag guards: a start queued behind close."""
    started: list[asyncio.Task[Any]] = []

    async def login(callbacks, *, signal=None, **_kwargs):
        started.append(asyncio.current_task())  # type: ignore[arg-type]
        callbacks.on_auth_url("https://radienthq.com/authorize", instructions=None)
        try:
            await asyncio.Event().wait()
        finally:
            # A teardown that yields, so close's cancel is a real await -- the
            # window a racing start used to fall into.
            await asyncio.sleep(0.02)

    _stub_login(monkeypatch, "radient", login)
    client, app = desktop
    await client.post("/v1/auth/login", json={"provider": "radient"})
    host = app.state.desktop_auth
    if first == "start":
        racing = asyncio.create_task(host.start("radient"))
        await asyncio.sleep(0)  # into the supersede's teardown await
        closing = asyncio.create_task(host.close())
    else:
        closing = asyncio.create_task(host.close())
        racing = asyncio.create_task(host.start("radient"))
    await closing
    outcome = (await asyncio.gather(racing, return_exceptions=True))[0]
    for _ in range(5):
        await asyncio.sleep(0)
    assert all(task.done() for task in started), "a flow outlived close()"
    assert host.operations == {}
    if first == "close":
        from local_operator.server.utils.desktop_auth import SignInUnavailableError

        assert isinstance(outcome, SignInUnavailableError), outcome
    else:
        # It won the lock, so it created its op -- and close then cancelled it.
        assert not isinstance(outcome, BaseException), outcome
        assert outcome.state == "cancelled", outcome
    # Nothing left for the fixture's own close to find.
    app.state.desktop_auth = None


async def test_a_start_after_close_answers_503(desktop):
    """Review round 3, NIT 2: a start refused because the server is shutting down
    is not an invalid request, so it is not 422. 503 is also what the renderer's
    ``isServerUnreachable`` reads as "the backend is not answering"."""
    client, app = desktop
    await client.get("/v1/auth/providers")  # builds the host
    host = app.state.desktop_auth
    await host.close()
    refused = await client.post("/v1/auth/login", json={"provider": "radient"})
    assert refused.status_code == 503, refused.text
    assert "shuts down" in refused.json()["detail"]
    # An unknown provider is still the caller's mistake: 422, unchanged.
    app.state.desktop_auth = None
    unknown = await client.post("/v1/auth/login", json={"provider": "no-such-provider"})
    assert unknown.status_code == 422, unknown.text


@pytest.mark.parametrize("returns_from", ["provider_login", "controller"])
async def test_an_abandoned_flow_that_finishes_late_writes_nothing(
    desktop, monkeypatch, returns_from
):
    """Review round 3, MINOR 2: a flow that absorbs its cancellation and then
    RETURNS a credential after the teardown bound used to flip its op from
    ``cancelled`` to ``succeeded``, store the credential and rewrite
    ``config.yml`` defaults for the provider the user had walked away from. Once
    cancelled, every one of those writes is a no-op: state, auth URL, stored
    credentials and the config file are all exactly as the cancel left them.

    ``provider_login`` returns through the real ``ProviderController.login``,
    whose aborted-signal check refuses the credential write. ``controller``
    replaces that method with one that ignores the signal and returns normally,
    so the HOST's own guards (terminal state, login defaults) are pinned on their
    own rather than only behind the controller's."""
    from local_operator.providers.controller import ProviderController
    from local_operator.server.utils import desktop_auth

    monkeypatch.setattr(desktop_auth, "CANCEL_TEARDOWN_TIMEOUT_S", 0.05)
    cancelled = asyncio.Event()
    release = asyncio.Event()
    applied: list[str] = []
    real_apply = desktop_auth.apply_desktop_login_defaults

    def spy_apply(manager, provider_id, *, oauth):
        applied.append(provider_id)
        return real_apply(manager, provider_id, oauth=oauth)

    monkeypatch.setattr(desktop_auth, "apply_desktop_login_defaults", spy_apply)

    async def login(callbacks, *, signal=None, **_kwargs):
        callbacks.on_auth_url("https://radienthq.com/authorize", instructions=None)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Swallow the cancel and carry on, as a stuck provider flow would.
            cancelled.set()
        await release.wait()
        # A late progress callback, then a credential.
        callbacks.on_auth_url("https://radienthq.com/authorize?late=1", instructions=None)
        return {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}

    if returns_from == "controller":

        async def controller_login(self, provider_id, **kwargs):
            definition = registry.get_provider_definition(provider_id)
            assert definition is not None
            await login(self._login_callbacks(definition), **kwargs)
            return "Logged in."

        monkeypatch.setattr(ProviderController, "login", controller_login)
    else:
        _stub_login(monkeypatch, "radient", login)
    client, app = desktop
    config_file = Path(app.state.config_manager.config_dir) / "config.yml"
    config_before = config_file.read_bytes() if config_file.exists() else None
    started = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()["result"]
    host = app.state.desktop_auth
    op = host.operations[started["id"]]

    deleted = await client.delete(f"/v1/auth/operations/{op.id}")
    assert deleted.json()["result"]["state"] == "cancelled"
    assert cancelled.is_set() and op.task is not None and not op.task.done()
    assert op.task in host._lingering

    release.set()
    await asyncio.wait_for(op.task, timeout=5.0)

    after = (await client.get(f"/v1/auth/operations/{op.id}")).json()["result"]
    assert after["state"] == "cancelled", after
    assert after["auth_url"] is None, "a late callback repainted the abandoned op"
    assert applied == [], "login defaults were applied for an abandoned sign-in"
    assert not host.store.list_credentials("radient"), "an abandoned sign-in stored a credential"
    assert (config_file.read_bytes() if config_file.exists() else None) == config_before


async def test_a_second_cancel_of_a_lingering_flow_returns_at_once(desktop, monkeypatch):
    """Review round 3, MINOR 2: ``cancel`` on an op already abandoned used to
    cancel the stuck task again and wait out the full bound a second time. It now
    returns without touching the task -- asserted on the task's own cancel count,
    not on a clock."""
    from local_operator.server.utils import desktop_auth

    monkeypatch.setattr(desktop_auth, "CANCEL_TEARDOWN_TIMEOUT_S", 0.05)
    release = asyncio.Event()

    async def login(callbacks, *, signal=None, **_kwargs):
        callbacks.on_auth_url("https://radienthq.com/authorize", instructions=None)
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass

    _stub_login(monkeypatch, "radient", login)
    client, app = desktop
    started = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()["result"]
    host = app.state.desktop_auth
    op = host.operations[started["id"]]
    try:
        await host.cancel(op)
        assert op.task is not None and not op.task.done()
        assert op.task.cancelling() == 1
        waits: list[Any] = []
        real_wait = asyncio.wait

        async def counting_wait(*args, **kwargs):
            waits.append(args)
            return await real_wait(*args, **kwargs)

        monkeypatch.setattr(desktop_auth.asyncio, "wait", counting_wait)
        await host.cancel(op)
        assert waits == [], "a second cancel waited on the teardown bound again"
        assert op.task.cancelling() == 1, "a second cancel re-cancelled the task"
        assert op.state == "cancelled"
    finally:
        release.set()
        if op.task is not None:
            await asyncio.wait_for(op.task, timeout=5.0)


def test_each_event_loop_keeps_its_own_start_lock():
    """Review round 3, MINOR 1: the lock used to be ONE slot, swapped whenever the
    running loop changed. With two loops live at once, a call from loop B
    replaced it while a task on loop A held the old one, so the next start on A
    got a fresh, unheld lock and ran alongside the holder. Reproduced exactly:
    A1 holds the lock on loop A, loop B (another thread) asks for its lock, then
    A2 on loop A must wait for A1."""
    import threading

    from local_operator.server.utils.desktop_auth import DesktopAuth

    host = DesktopAuth(MagicMock(), None, None)
    loop_b_locked = threading.Event()
    b_lock: list[asyncio.Lock] = []

    def on_loop_b() -> None:
        async def take() -> None:
            lock = host._lock()
            b_lock.append(lock)
            async with lock:
                loop_b_locked.set()

        asyncio.run(take())

    async def on_loop_a() -> dict[str, bool]:
        a1_holds = asyncio.Event()
        a1_release = asyncio.Event()
        a2_entered = asyncio.Event()

        async def a1() -> None:
            async with host._lock():
                a1_holds.set()
                await a1_release.wait()

        async def a2() -> None:
            async with host._lock():
                a2_entered.set()

        first = asyncio.create_task(a1())
        await a1_holds.wait()
        held = host._lock()
        # Loop B asks for a lock while A1 still holds loop A's.
        other = threading.Thread(target=on_loop_b)
        other.start()
        await asyncio.to_thread(other.join)
        assert loop_b_locked.is_set()
        second = asyncio.create_task(a2())
        for _ in range(5):
            await asyncio.sleep(0)
        result = {
            "same_lock_on_a": host._lock() is held,
            "b_got_its_own": b_lock[0] is not held,
            "a2_entered_while_a1_held": a2_entered.is_set(),
        }
        a1_release.set()
        await asyncio.gather(first, second)
        result["a2_entered_after"] = a2_entered.is_set()
        return result

    assert asyncio.run(on_loop_a()) == {
        "same_lock_on_a": True,
        "b_got_its_own": True,
        "a2_entered_while_a1_held": False,
        "a2_entered_after": True,
    }


def test_every_callback_login_accepts_the_desktop_opener():
    """Review round 2, NIT 1: the forwarding rule, structurally. ``_lazy_login``
    forwards ``open_browser`` only to a login whose signature names it, and the
    desktop passes a no-op opener because its renderer opens the URL -- so a
    callback-port provider whose login does not accept it silently falls back
    to ``webbrowser.open``. Checked on the real login functions' signatures, so
    no flow class or ``run`` patch can hide a miss."""
    import importlib
    import inspect

    missing: list[str] = []
    for provider in registry.PROVIDER_REGISTRY:
        if provider.login is None or provider.callback_port is None:
            continue
        # A ``_lazy_login`` thunk names its target in its closure; resolve it,
        # because the thunk's own signature always has ``open_browser``. A login
        # that is not a thunk is checked as itself.
        cells = dict(zip(provider.login.__code__.co_freevars, provider.login.__closure__ or ()))
        target: Any = provider.login
        if "module" in cells and "attr" in cells:
            module = importlib.import_module(cells["module"].cell_contents)
            target = getattr(module, cells["attr"].cell_contents)
        if "open_browser" not in inspect.signature(target).parameters:
            missing.append(provider.id)
    assert missing == []

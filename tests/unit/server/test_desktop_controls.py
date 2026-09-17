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
from local_operator.credentials import CredentialManager
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
    app.state.credential_manager = CredentialManager(tmp_path)
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
    assert (await client.post("/v1/auth/login", json={"provider": "radient"})).status_code == 409
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

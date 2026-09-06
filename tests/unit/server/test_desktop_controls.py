"""HTTP boundary + real settings/auth stores, without third-party credentials."""

import asyncio
import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
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


async def test_arbitrary_origins_are_not_echoed_into_the_cors_grant(tmp_path: Path, monkeypatch):
    """A hostile origin must get no ``Access-Control-Allow-Origin`` at all.

    The app is mounted with ``allow_origins=["*"]`` and
    ``allow_credentials=True``, which makes Starlette ECHO the caller's origin
    -- turning every open route into something a drive-by page can read with
    ``fetch()`` while the desktop app holds the backend on a known loopback
    port. Missing auth was only half of QA's Q2; this is the half that made it
    browser-exploitable rather than curl-only.
    """
    from fastapi.middleware.cors import CORSMiddleware

    from local_operator.server.app import desktop_origin_cors

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
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
        # No allowlist configured: a standalone server keeps its historical
        # wildcard CORS, so existing embedders are untouched.
        hostile = await client.get("/health", headers={"Origin": "http://evil.example"})
        assert hostile.headers.get("access-control-allow-origin") == "http://evil.example"

        monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
        hostile = await client.get("/health", headers={"Origin": "http://evil.example"})
        assert hostile.status_code == 200
        assert "access-control-allow-origin" not in hostile.headers
        # Credentials must go with the grant, or the pair reads as a wildcard one.
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

    async def login(callbacks, *, open_browser):
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

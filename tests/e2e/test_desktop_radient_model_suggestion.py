"""Model suggestions through the REAL desktop pull routes, against a fake hub.

What this proves (the §5 e2e cells for S3): the desktop download and team-pull
routes carry a hub ``model_suggestion`` through the REAL registries and the
REAL availability resolver, with only the hub itself faked over loopback:

1. an available suggestion → the imported row carries the pair, payload notice
   null;
2. an unknown provider → the import still SUCCEEDS, the row keeps its empty
   fields (the user's default at launch) and the payload carries the reason —
   non-blocking by construction.

What it deliberately does not: prove anything about the real Radient account
behind a real credential. The upstream is a labelled fake serving the ZIP and
team-document shapes the hub serves; the live half remains QA's.
"""

import asyncio
import io
import secrets
import socket
import time
import zipfile
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import Response

from local_operator.server.app import app
from tests.e2e.test_desktop_controls import until

pytestmark = pytest.mark.e2e


@asynccontextmanager
async def serve(application):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(application, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await until(lambda: server.started)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 30)
        listener.close()


@asynccontextmanager
async def serve_threaded(application):
    """The fake HUB, on its OWN event loop in its own thread.

    Load-bearing rather than tidy: the desktop's hub call in the download
    route is synchronous ``requests`` on the request's event-loop thread, so
    a fake served on the TEST's loop is starved by the very request it must
    answer -- measured here as a 150 s+ hang with the main thread parked in
    ``http.client._read_status``. A thread-isolated loop cannot be starved by
    the loop it answers.
    """
    import threading

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(application, log_level="error"))

    def run() -> None:
        asyncio.run(server.serve(sockets=[listener]))

    thread = threading.Thread(target=run, daemon=True, name="fake-hub-loop")
    thread.start()
    try:
        await until(lambda: server.started)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(30)
        listener.close()


def _agent_zip(suggestion: dict[str, str]) -> bytes:
    """The archive the hub serves for a pull, carrying ``model_suggestion``."""
    metadata = {
        "id": "hub-agent-77",
        "name": "Suggestion Carrier",
        "created_date": "2024-01-01T00:00:00Z",
        "version": "0.2.16",
        "model_suggestion": suggestion,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("agent.yml", yaml.dump(metadata))
    return buffer.getvalue()


def _seed_store() -> None:
    """The state a completed sign-in leaves: an openrouter key and a Radient
    OAuth row (the person-scoped credential org calls spend)."""
    store = app.state.desktop_auth.store
    store.upsert_credential(
        "openrouter", {"key": "sk-fixture", "source": "login", "type": "api_key"}
    )
    store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": secrets.token_hex(16),
            "refresh": secrets.token_hex(16),
            "expires": int(time.time() * 1000) + 3_600_000,
        },
    )


async def _prepared_client(desktop_url: str, token: str) -> httpx.AsyncClient:
    client = httpx.AsyncClient(base_url=desktop_url, timeout=30)
    client.headers["Authorization"] = "Bearer " + token
    # Materialises (and initialises) the desktop auth store, then writes the
    # credential state the sign-in routes would have written.
    response = await client.get("/v1/auth/status")
    assert response.status_code == 200, response.text
    _seed_store()
    return client


@pytest.mark.asyncio
async def test_desktop_pull_applies_an_available_suggestion(headless_tui_env, monkeypatch):
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    zips = {
        "hub-agent-1": _agent_zip({"hosting": "openrouter", "model": "anthropic/claude-opus-5.5"})
    }
    fake = FastAPI()

    @fake.get("/v1/agents/{agent_id}/download")
    async def download(agent_id: str):
        return Response(content=zips[agent_id], media_type="application/zip")

    async with serve_threaded(fake) as upstream_url:
        monkeypatch.setenv("RADIENT_API_BASE_URL", upstream_url + "/v1")
        async with serve(app) as desktop_url:
            client = await _prepared_client(desktop_url, token)
            try:
                response = await client.get("/v1/agents/hub-agent-1/download")
            finally:
                await client.aclose()

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["model_notice"] is None

    from local_operator.agents import AgentRegistry
    from local_operator.paths import config_dir

    agent = AgentRegistry(config_dir()).get_agent(result["id"])
    assert agent is not None
    # The suggestion was CONSUMED into the row's own fields, so every surface
    # (list, exec, this read) sees it like any user-set choice.
    assert (agent.hosting, agent.model) == ("openrouter", "anthropic/claude-opus-5.5")


@pytest.mark.asyncio
async def test_desktop_pull_fails_over_an_unknown_provider_with_a_notice(
    headless_tui_env, monkeypatch
):
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    zips = {"hub-agent-1": _agent_zip({"hosting": "no-such-provider", "model": "m"})}
    fake = FastAPI()

    @fake.get("/v1/agents/{agent_id}/download")
    async def download(agent_id: str):
        return Response(content=zips[agent_id], media_type="application/zip")

    async with serve_threaded(fake) as upstream_url:
        monkeypatch.setenv("RADIENT_API_BASE_URL", upstream_url + "/v1")
        async with serve(app) as desktop_url:
            client = await _prepared_client(desktop_url, token)
            try:
                response = await client.get("/v1/agents/hub-agent-1/download")
            finally:
                await client.aclose()

    # The import SUCCEEDED (200, a row exists) — the notice is the non-blocking
    # half, and the row keeps the empty fields that resolve to the user's
    # configured default at launch.
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["model_notice"] == {
        "reason": "unknown_provider",
        "requested": {"hosting": "no-such-provider", "model": "m"},
    }

    from local_operator.agents import AgentRegistry
    from local_operator.paths import config_dir

    agent = AgentRegistry(config_dir()).get_agent(result["id"])
    assert agent is not None
    assert (agent.hosting, agent.model) == ("", "")


_TEAM_DOCUMENT = {
    "id": "hub-team-77",
    "tenant_id": "org-fixture",
    "name": "carrier-crew",
    "description": "Ships it.",
    "manager": "manager",
    "members": [],
    "instructions": "You ship.",
    "project": "rad-1",
    "version": "1.0.0",
}


@pytest.mark.asyncio
async def test_desktop_team_pull_stores_an_available_suggestion(headless_tui_env, monkeypatch):
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.setenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", "1")
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    document = dict(
        _TEAM_DOCUMENT, model_suggestion={"hosting": "openrouter", "model": "vendor/model"}
    )
    fake = FastAPI()

    @fake.get("/v1/teams/{team_id}")
    async def get_team(team_id: str, request: Request):
        # The hub's API-wide envelope; ``get_team`` unwraps it (the fixture
        # mirrors the client's own reader, not the raw document).
        return {"msg": "Team retrieved successfully", "result": document}

    async with serve_threaded(fake) as upstream_url:
        monkeypatch.setenv("RADIENT_API_BASE_URL", upstream_url + "/v1")
        async with serve(app) as desktop_url:
            client = await _prepared_client(desktop_url, token)
            try:
                response = await client.get("/v1/teams/pull/hub-team-77")
            finally:
                await client.aclose()

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["model_notice"] is None

    from local_operator.paths import config_dir
    from local_operator.teams import TeamRegistry

    stored = TeamRegistry(config_dir()).get_team_by_name("carrier-crew")
    assert stored is not None
    assert stored.model_suggestion is not None
    assert (stored.model_suggestion.hosting, stored.model_suggestion.model) == (
        "openrouter",
        "vendor/model",
    )


@pytest.mark.asyncio
async def test_desktop_team_pull_omits_an_unavailable_suggestion_with_a_notice(
    headless_tui_env, monkeypatch
):
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.setenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", "1")
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    document = dict(
        _TEAM_DOCUMENT,
        name="skipping-crew",
        model_suggestion={"hosting": "no-such-provider", "model": "m"},
    )
    fake = FastAPI()

    @fake.get("/v1/teams/{team_id}")
    async def get_team(team_id: str, request: Request):
        # The hub's API-wide envelope; ``get_team`` unwraps it (the fixture
        # mirrors the client's own reader, not the raw document).
        return {"msg": "Team retrieved successfully", "result": document}

    async with serve_threaded(fake) as upstream_url:
        monkeypatch.setenv("RADIENT_API_BASE_URL", upstream_url + "/v1")
        async with serve(app) as desktop_url:
            client = await _prepared_client(desktop_url, token)
            try:
                response = await client.get("/v1/teams/pull/hub-team-77")
            finally:
                await client.aclose()

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["model_notice"] == {
        "reason": "unknown_provider",
        "requested": {"hosting": "no-such-provider", "model": "m"},
    }

    from local_operator.paths import config_dir
    from local_operator.teams import TeamRegistry

    stored = TeamRegistry(config_dir()).get_team_by_name("skipping-crew")
    assert stored is not None
    assert stored.model_suggestion is None

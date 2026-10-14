"""§5.6: the agent-messages settings SECTION is dynamically absent.

The static registry is not the gate (§5.4): the two registry drainers must
withhold both the section header and every row belonging to it unless the
feature's state says the user should see it — absent in state A (no eligible
organization) and G (dark launch), present from B on. The state is the
process-local availability record (``team_bridge.availability``), written by
the status discipline's boundary reads; this file pins the DESKTOP drainer
(``GET /v1/settings``), and ``tests/unit/tui/test_settings_section_visibility.py``
pins the TUI one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import settings as settings_routes
from local_operator.team_bridge import availability

pytestmark = pytest.mark.asyncio

TOKEN = "test-token"


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    availability.clear()
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(settings_routes.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, root
    availability.clear()


async def _team_bridge_view(client: AsyncClient) -> tuple[list[str], list[str]]:
    response = await client.get("/v1/settings")
    assert response.status_code == 200
    result = response.json()["result"]
    sections = [row["name"] for row in result["sections"] if row["name"] == "team_bridge"]
    rows = [row["key"] for row in result["settings"] if row["key"].startswith("team_bridge.")]
    return sections, rows


async def test_never_observed_reads_as_absent(api):
    client, _root = api
    sections, rows = await _team_bridge_view(client)
    assert sections == []
    assert rows == []


async def test_state_a_and_g_read_as_absent(api):
    """A: ready but no eligible tenant and no endpoint; G: dark launch."""
    client, root = api

    availability.record(root, visible=False, state="ready", checked_at=1.0)
    assert await _team_bridge_view(client) == ([], [])

    availability.record(root, visible=False, state="unavailable", checked_at=2.0)
    assert await _team_bridge_view(client) == ([], [])


async def test_an_eligible_observation_brings_the_section_and_its_rows_back(api):
    client, root = api
    availability.record(root, visible=True, state="ready", checked_at=1.0)

    sections, rows = await _team_bridge_view(client)

    assert sections == ["team_bridge"]
    assert rows == ["team_bridge.enabled", "team_bridge.session"]

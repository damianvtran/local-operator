"""Actual desktop project router/store, isolated in-process ASGI; no UI or socket.

The wire spelling `%252f` must reach the route as literal `%2f`, not `/`.
The UI relay's separately authored test exercises its request construction;
neither test alone is an end-to-end desktop-to-Uvicorn proof.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_projects


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["release%2fready", "literal%2e%2e%5c%2f", "release?"])
async def test_percent_like_milestone_create_remove(name: str, tmp_path: Path, monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("CMUX_") or key.startswith("LOP_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-project-token")
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_projects.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer synthetic-project-token"},
    ) as client:
        created = await client.post("/v1/desktop/projects", json={"name": "demo-project"})
        assert created.status_code == 200
        project_id = created.json()["result"]["id"]
        added = await client.post(
            f"/v1/desktop/projects/{project_id}/milestones", json={"name": name}
        )
        assert added.status_code == 200
        assert [row["name"] for row in added.json()["result"]["milestones"]] == [name]
        encoded = quote(name, safe="")
        removed = await client.delete(f"/v1/desktop/projects/{project_id}/milestones/{encoded}")
        assert removed.status_code == 200
        assert removed.json()["result"]["milestones"] == []

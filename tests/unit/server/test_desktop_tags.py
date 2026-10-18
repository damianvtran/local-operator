"""Real ASGI routes and catalogue projection against an isolated durable store."""

import json
import os
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "test-tags-token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer test-tags-token"},
    ) as client:
        yield client, tmp_path
    await app.state.desktop_sessions.close()


def seed(root: Path, sid: str, delegated: bool = False) -> Path:
    path = root / "sessions" / sid
    path.mkdir(parents=True)
    (path / "created_at.json").write_text("1700000000")
    (path / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    if delegated:
        mark_session_origin(path, ORIGIN_SUBAGENT)
    return path


@pytest.mark.asyncio
async def test_registry_assignments_and_catalogue(api) -> None:
    client, root = api
    assert (await client.get("/v1/desktop/tags")).json()["result"] == {"tags": []}
    response = await client.put(
        "/v1/desktop/tags", json={"tags": [{"name": "Review"}, {"name": "Fix"}]}
    )
    assert response.status_code == 200, response.text
    a, b = response.json()["result"]["tags"]
    assert a["order"] == 0 and a["conversations"] == 0
    sid, untagged, delegated = "aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa3"
    paths = [seed(root, sid), seed(root, untagged), seed(root, delegated, True)]
    body = {"tags": [a["id"], b["id"]], "primary": b["id"]}
    response = await client.put(f"/v1/desktop/sessions/{sid}/tags", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["primary"] == b["id"]
    store = root / "tags" / "session" / f"{sid}.json"
    before = store.stat().st_mtime_ns
    assert (await client.put(f"/v1/desktop/sessions/{sid}/tags", json=body)).status_code == 200
    assert store.stat().st_mtime_ns == before
    response = await client.put(
        f"/v1/desktop/sessions/{delegated}/tags", json={"tags": [a["id"]], "primary": None}
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["primary"] == a["id"]
    counts = (await client.get("/v1/desktop/tags")).json()["result"]["tags"]
    assert [tag["conversations"] for tag in counts] == [2, 1]
    response = await client.get("/v1/desktop/sessions")
    assert response.status_code == 200, response.text
    rows = {row["id"]: row for row in response.json()["result"]["sessions"]}
    assert rows[sid]["tags"] == [{"id": a["id"], "name": "Review"}, {"id": b["id"], "name": "Fix"}]
    assert rows[sid]["primary"] == b["id"]
    assert rows[untagged]["tags"] == [] and rows[untagged]["primary"] is None
    response = await client.put(
        "/v1/desktop/tags", json={"tags": [{"id": a["id"], "name": "Renamed"}]}
    )
    assert response.status_code == 200, response.text
    rows = {
        row["id"]: row
        for row in (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"]
    }
    assert rows[sid]["tags"] == [{"id": a["id"], "name": "Renamed"}]
    assert rows[sid]["primary"] == a["id"]
    assert all(path.is_dir() for path in paths)
    cleared = await client.put(
        f"/v1/desktop/sessions/{sid}/tags", json={"tags": [], "primary": None}
    )
    assert cleared.json()["result"] == {"session_id": sid, "tags": [], "primary": None}
    rows = (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"]
    assert all("tags" in row and "primary" in row for row in rows)
    assert (await client.get("/v1/capabilities")).json()["result"]["features"]["session_tags"] == 1


@pytest.mark.asyncio
async def test_invalid_requests_are_refused_without_writes(api) -> None:
    client, root = api
    sid = "abcdef123456"
    seed(root, sid, True)
    for body in (
        {"tags": [{"name": "Untagged"}]},
        {"tags": [{"name": "a"}, {"name": "A"}]},
        {"tags": [{"id": "unknown", "name": "A"}]},
        {"tags": [{"name": str(i)} for i in range(33)]},
    ):
        assert (await client.put("/v1/desktop/tags", json=body)).status_code == 422
    for body in (
        {"tags": ["unknown"], "primary": None},
        {"tags": [], "primary": "unknown"},
        {"tags": []},
    ):
        assert (await client.put(f"/v1/desktop/sessions/{sid}/tags", json=body)).status_code == 422
    assert (
        await client.put(
            "/v1/desktop/sessions/000000000000/tags", json={"tags": [], "primary": None}
        )
    ).status_code == 404
    assert (
        await client.put("/v1/desktop/sessions/not-an-id/tags", json={"tags": [], "primary": None})
    ).status_code == 404
    assert not (root / "tags").exists()
    assert (
        await client.put("/v1/desktop/projects/project/tags", json={"tags": [], "primary": None})
    ).status_code == 404


@pytest.mark.asyncio
async def test_assignment_response_uses_one_registry_snapshot(api, monkeypatch) -> None:
    """Deletion between writing IDs and resolving names cannot leak an orphan primary."""
    from local_operator.session import session_tags

    client, root = api
    sid = "aaaaaaaaaaa1"
    seed(root, sid)
    category = (await client.put("/v1/desktop/tags", json={"tags": [{"name": "Review"}]})).json()[
        "result"
    ]["tags"][0]
    original = session_tags.read_categories
    reads = 0

    def registry_snapshot(root):
        nonlocal reads
        reads += 1
        # Validation and the store's write-result read see the old registry;
        # the HTTP name-resolution read sees a concurrent category deletion.
        return original(root) if reads < 3 else []

    monkeypatch.setattr(session_tags, "read_categories", registry_snapshot)
    response = await client.put(
        f"/v1/desktop/sessions/{sid}/tags",
        json={"tags": [category["id"]], "primary": category["id"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"session_id": sid, "tags": [], "primary": None}

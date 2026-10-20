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
    assert rows[sid]["tags_revision"] == "1"
    assert rows[untagged]["tags_revision"] == "0"
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
    assert cleared.json()["result"] == {
        "session_id": sid,
        "tags": [],
        "primary": None,
        "tags_revision": "2",
    }
    rows = (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"]
    assert all("tags" in row and "primary" in row for row in rows)
    features = (await client.get("/v1/capabilities")).json()["result"]["features"]
    assert features["session_tags"] == 1
    # V2-3's optional fork half is its own key: a client must be able to ask
    # before sending `auto_tag`, because `Fork` forbids unknown body keys.
    assert features["session_tags_auto_tag"] == 1


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
        # Store validation sees the old registry; the HTTP name-resolution
        # snapshot sees a concurrent category deletion. Assignment state is
        # returned directly by the write so its token cannot race a re-read.
        return original(root) if reads < 2 else []

    monkeypatch.setattr(session_tags, "read_categories", registry_snapshot)
    response = await client.put(
        f"/v1/desktop/sessions/{sid}/tags",
        json={"tags": [category["id"]], "primary": category["id"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {
        "session_id": sid,
        "tags": [],
        "primary": None,
        "tags_revision": "1",
    }


@pytest.mark.asyncio
async def test_name_only_put_replay_preserves_ids_assignments_and_counts(api) -> None:
    client, root = api
    # Byte-identical replay models a response lost before the client learns IDs.
    body = '{"tags":[{"name":"Review"},{"name":"Fix"}]}'
    headers = {"Content-Type": "application/json"}
    response = await client.put("/v1/desktop/tags", content=body, headers=headers)
    assert response.status_code == 200, response.text
    original = response.json()["result"]["tags"]
    ids = [tag["id"] for tag in original]
    sids = ["aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa3"]
    for sid in sids:
        seed(root, sid)
        assigned = await client.put(
            f"/v1/desktop/sessions/{sid}/tags", json={"tags": ids, "primary": ids[1]}
        )
        assert assigned.status_code == 200, assigned.text
    response = await client.put("/v1/desktop/tags", content=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["tags"] == [{**tag, "conversations": 3} for tag in original]
    rows = (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"]
    for row in rows:
        assert [tag["id"] for tag in row["tags"]] == ids
        assert row["primary"] == ids[1]


@pytest.mark.asyncio
async def test_assignment_put_preserves_oldest_tag_order(api) -> None:
    client, root = api
    sid = "aaaaaaaaaaa1"
    seed(root, sid)
    registry = await client.put("/v1/desktop/tags", json={"tags": [{"name": "A"}, {"name": "B"}]})
    a, b = registry.json()["result"]["tags"]
    url = f"/v1/desktop/sessions/{sid}/tags"
    assert (await client.put(url, json={"tags": [a["id"]], "primary": a["id"]})).status_code == 200
    for _ in range(2):
        response = await client.put(url, json={"tags": [b["id"], a["id"]], "primary": b["id"]})
        assert response.status_code == 200, response.text
        assert [tag["id"] for tag in response.json()["result"]["tags"]] == [a["id"], b["id"]]
    await client.put("/v1/desktop/tags", json={"tags": [{"id": a["id"], "name": "A"}]})
    row = (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"][0]
    assert row["primary"] == a["id"]


def test_route_schema_publishes_store_caps() -> None:
    from local_operator.server.routes.desktop_sessions import TagAssignment, TagRegistry

    assert TagRegistry.model_json_schema()["properties"]["tags"]["maxItems"] == 32
    assert TagAssignment.model_json_schema()["properties"]["tags"]["maxItems"] == 8


@pytest.mark.asyncio
async def test_buffered_rename_and_old_name_creation_replays_without_regrouping(api) -> None:
    client, root = api
    first = await client.put("/v1/desktop/tags", json={"tags": [{"name": "Review"}]})
    old_id = first.json()["result"]["tags"][0]["id"]
    sid = "aaaaaaaaaaa1"
    seed(root, sid)
    assignment = await client.put(
        f"/v1/desktop/sessions/{sid}/tags", json={"tags": [old_id], "primary": old_id}
    )
    assert assignment.status_code == 200, assignment.text
    body = {"tags": [{"id": old_id, "name": "Renamed"}, {"name": "Review"}]}
    response = await client.put("/v1/desktop/tags", json=body)
    assert response.status_code == 200, response.text
    renamed, fresh = response.json()["result"]["tags"]
    assert renamed["id"] == old_id and renamed["conversations"] == 1
    assert fresh["id"] != old_id and fresh["conversations"] == 0
    replay = await client.put("/v1/desktop/tags", json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json()["result"] == response.json()["result"]
    rows = (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"]
    row = next(row for row in rows if row["id"] == sid)
    assert row["primary"] == old_id
    assert row["tags"] == [{"id": old_id, "name": "Renamed"}]


@pytest.mark.asyncio
async def test_conditional_assignment_conflict_and_aba(api) -> None:
    client, root = api
    sid = "aaaaaaaaaaa1"
    seed(root, sid)
    tag = (await client.put("/v1/desktop/tags", json={"tags": [{"name": "A"}]})).json()["result"][
        "tags"
    ][0]
    ref = {"id": tag["id"], "name": tag["name"]}
    url = f"/v1/desktop/sessions/{sid}/tags"
    body = {"tags": [tag["id"]], "primary": tag["id"]}
    response = await client.put(url, json={**body, "expected_revision": "0"})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["tags_revision"] == "1"
    store = root / "tags" / "session" / f"{sid}.json"
    before = (store.read_bytes(), store.stat().st_mtime_ns)
    response = await client.put(url, json={"tags": [], "primary": None, "expected_revision": 0})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "tags_conflict"
    assert response.json()["detail"]["result"] == {
        "session_id": sid,
        "tags": [ref],
        "primary": tag["id"],
        "tags_revision": "1",
    }
    assert (store.read_bytes(), store.stat().st_mtime_ns) == before
    response = await client.put(url, json={"tags": [], "primary": None, "expected_revision": 1})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["tags_revision"] == "2"
    before = (store.read_bytes(), store.stat().st_mtime_ns)
    # The initial empty assignment has returned, but zero is still stale (ABA).
    response = await client.put(url, json={**body, "expected_revision": "0"})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["result"] == {
        "session_id": sid,
        "tags": [],
        "primary": None,
        "tags_revision": "2",
    }
    assert (store.read_bytes(), store.stat().st_mtime_ns) == before
    for invalid in (
        {"tags": ["unknown"], "primary": None},
        {"tags": [], "primary": tag["id"]},
    ):
        response = await client.put(url, json={**invalid, "expected_revision": "2"})
        assert response.status_code == 422, response.text
        assert (store.read_bytes(), store.stat().st_mtime_ns) == before
    # Legacy clients retain unconditional desired-state writes.
    response = await client.put(url, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["tags_revision"] == "3"


@pytest.mark.asyncio
async def test_remote_assignment_still_refused_before_revision_check(api, monkeypatch) -> None:
    client, root = api
    sid = "aaaaaaaaaaa1"
    seed(root, sid)
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_mesh.remote_owner",
        lambda root_arg, session_id: ("synthetic-peer", "test-box"),
    )
    response = await client.put(
        f"/v1/desktop/sessions/{sid}/tags",
        json={"tags": [], "primary": None, "expected_revision": "stale"},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "session_is_remote"
    assert not (root / "tags" / "session" / f"{sid}.json").exists()


@pytest.mark.asyncio
async def test_remote_row_model_does_not_invent_local_revision(api) -> None:
    from local_operator.server.models.desktop_sessions import SessionRow

    client, root = api
    sid = "aaaaaaaaaaa1"
    seed(root, sid)
    row = (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"][0]
    assert row.pop("tags_revision") == "0"
    row["locality"] = "remote"
    assert "tags_revision" not in SessionRow.model_validate(row).model_dump(mode="json")

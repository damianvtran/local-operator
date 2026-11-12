"""Temporary chats over the desktop control plane (contract §2.3, frozen wire).

Real routes, real ``errors()`` ladder, real response models and a real store on
a temporary filesystem: create with ``temporary``, the ``peer`` refusal, the
toggle, list exclusion/inclusion, the row and snapshot field, the tombstone 404,
the pin and project refusals, and the capability key.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import temporary
from local_operator.session.cleanup import mark_store

MINE = "a" * 12
OTHER = "b" * 12
TOKEN = "temporary-chats-test-token"


def _session(root: Path, session_id: str) -> Path:
    path = root / "sessions" / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "created_at.json").write_text("1700000000")
    (path / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    return path


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    mark_store(tmp_path / "sessions")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path.resolve()
    await app.state.desktop_sessions.close()


def _rid() -> str:
    return str(uuid.uuid4())


def _rows(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    return response.json()["result"]["sessions"]


async def _toggle(client: AsyncClient, session_id: str, on: bool):
    return await client.post(f"/v1/desktop/sessions/{session_id}/temporary", json={"temporary": on})


@pytest.mark.asyncio
async def test_the_capability_is_advertised(api) -> None:
    client, _root = api
    features = (await client.get("/v1/capabilities")).json()["result"]["features"]
    assert features["temporary_chats"] == 1


@pytest.mark.asyncio
async def test_create_with_temporary_marks_the_chat(api) -> None:
    client, root = api
    response = await client.post(
        "/v1/desktop/sessions", json={"cwd": str(root), "temporary": True, "request_id": _rid()}
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["result"]["session_id"]
    assert temporary.is_marked(root, session_id)
    plain = await client.post("/v1/desktop/sessions", json={"cwd": str(root), "request_id": _rid()})
    assert not temporary.is_marked(root, plain.json()["result"]["session_id"])


@pytest.mark.asyncio
async def test_temporary_with_peer_is_a_422_with_its_code(api) -> None:
    client, root = api
    response = await client.post(
        "/v1/desktop/sessions",
        json={"cwd": "", "peer": "a" * 16, "temporary": True, "request_id": _rid()},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "temporary_peer_unsupported"


@pytest.mark.asyncio
async def test_a_non_bool_temporary_is_refused(api) -> None:
    client, root = api
    response = await client.post(
        "/v1/desktop/sessions", json={"cwd": str(root), "temporary": 1, "request_id": _rid()}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_the_list_excludes_temporary_chats_unless_asked(api) -> None:
    client, root = api
    _session(root, MINE)
    _session(root, OTHER)
    listed = _rows(await client.get("/v1/desktop/sessions"))
    assert {row["id"]: row["temporary"] for row in listed} == {MINE: None, OTHER: None}

    toggled = await _toggle(client, OTHER, True)
    assert toggled.status_code == 200, toggled.text
    state = toggled.json()["result"]
    assert state["session_id"] == OTHER
    assert state["temporary"]["idle_minutes"] == 30 and state["temporary"]["hold"] is None
    assert state["temporary"]["deletes_at"] is not None

    assert {row["id"] for row in _rows(await client.get("/v1/desktop/sessions"))} == {MINE}
    both = _rows(await client.get("/v1/desktop/sessions?include_temporary=true"))
    by_id = {row["id"]: row["temporary"] for row in both}
    assert by_id[MINE] is None
    assert set(by_id[OTHER]) == {"marked_at", "idle_minutes", "deletes_at", "hold"}


@pytest.mark.asyncio
async def test_keep_is_idempotent_and_returns_null(api) -> None:
    client, root = api
    _session(root, MINE)
    assert (await _toggle(client, MINE, True)).status_code == 200
    for _ in range(2):
        kept = await _toggle(client, MINE, False)
        assert kept.status_code == 200
        assert kept.json()["result"] == {"session_id": MINE, "temporary": None}
    assert {row["id"] for row in _rows(await client.get("/v1/desktop/sessions"))} == {MINE}


@pytest.mark.asyncio
async def test_search_never_returns_a_temporary_chat(api) -> None:
    client, root = api
    _session(root, MINE)
    assert (await _toggle(client, MINE, True)).status_code == 200
    response = await client.get("/v1/desktop/sessions/search?q=" + MINE)
    assert response.status_code == 200, response.text
    assert all(hit["id"] != MINE for hit in response.json()["result"]["sessions"])


@pytest.mark.asyncio
async def test_pinning_a_temporary_chat_is_a_409(api) -> None:
    client, root = api
    _session(root, MINE)
    assert (await _toggle(client, MINE, True)).status_code == 200
    refused = await client.post(f"/v1/desktop/sessions/{MINE}/pin", json={"pinned": True})
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "session_temporary"
    # Unpinning is never refused.
    assert (
        await client.post(f"/v1/desktop/sessions/{MINE}/pin", json={"pinned": False})
    ).status_code == 200


@pytest.mark.asyncio
async def test_making_a_pinned_chat_temporary_is_a_409(api) -> None:
    client, root = api
    _session(root, MINE)
    assert (
        await client.post(f"/v1/desktop/sessions/{MINE}/pin", json={"pinned": True})
    ).status_code == 200
    refused = await _toggle(client, MINE, True)
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "session_pinned"
    assert not temporary.is_marked(root, MINE)


@pytest.mark.asyncio
async def test_making_a_project_chat_temporary_is_a_409(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = api
    _session(root, MINE)
    monkeypatch.setattr(temporary, "_linked_to_project", lambda *_a: True)
    refused = await _toggle(client, MINE, True)
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "session_in_project"


@pytest.mark.asyncio
async def test_a_delegated_run_cannot_be_made_temporary(api) -> None:
    client, root = api
    path = _session(root, MINE)
    (path / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    refused = await _toggle(client, MINE, True)
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "session_not_user"


@pytest.mark.asyncio
async def test_a_deleted_temporary_chat_answers_its_own_404(api) -> None:
    client, root = api
    now = time.time()
    (root / "sessions" / temporary.TOMBSTONES_NAME).write_text(
        json.dumps([{"id": MINE, "deleted_at": now}])
    )
    for method, url, body in (
        ("get", f"/v1/desktop/sessions/{MINE}", None),
        ("post", f"/v1/desktop/sessions/{MINE}/temporary", {"temporary": False}),
        ("post", f"/v1/desktop/sessions/{MINE}/pin", {"pinned": True}),
        ("post", f"/v1/desktop/sessions/{MINE}/archive", {"archived": True}),
    ):
        response = await getattr(client, method)(url, **({"json": body} if body else {}))
        assert response.status_code == 404, (url, response.text)
        detail = response.json()["detail"]
        assert detail["code"] == "session_temporary_deleted", url
        assert detail["deleted_at"] == now
    # An unknown id with no tombstone keeps the generic answer.
    response = await client.post(
        f"/v1/desktop/sessions/{OTHER}/temporary", json={"temporary": True}
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] != "session_temporary_deleted"


@pytest.mark.asyncio
async def test_the_snapshot_carries_the_field(api) -> None:
    client, root = api
    created = await client.post(
        "/v1/desktop/sessions", json={"cwd": str(root), "temporary": True, "request_id": _rid()}
    )
    session_id = created.json()["result"]["session_id"]
    response = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert response.status_code == 200, response.text
    payload = response.json()["result"]["payload"]
    assert payload["temporary"]["idle_minutes"] == 30
    assert (await _toggle(client, session_id, False)).status_code == 200
    payload = (await client.get(f"/v1/desktop/sessions/{session_id}")).json()["result"]["payload"]
    assert payload["temporary"] is None

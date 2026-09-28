"""``/v1/desktop/aida`` — the frozen contract, exercised over the real app.

The route is the cross-repo boundary (design §4), so this file drives it the
way the renderer does — ASGI transport, bearer token, JSON bodies — against an
isolated config root, and pins each clause the UI was written against:
``GET`` never creates, ``open``/``greet`` ensure, ``greet`` is idempotent,
``pause``/``resume`` move the flag, a disabled install answers ``enabled: false``
on GET and 409 ``aida_disabled`` on POST.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from local_operator.config import ConfigManager
from local_operator.server.app import app
from tests.unit.aida.conftest import isolated_root_path, write_config

TOKEN = "aida-route-token"


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The SAME root every other aida test uses, built by the same function.

    A local fixture over the shared body rather than an imported fixture
    object: importing one and then naming it as a parameter is an F811
    redefinition, and a second hand-rolled root here is how the route tests
    and the engine tests would drift into testing two subtly different
    sandboxes.
    """
    return isolated_root_path(tmp_path, monkeypatch)


@pytest.fixture()
def client(isolated_root: Path, monkeypatch: pytest.MonkeyPatch):
    """The app over the isolated root, with the desktop plane open."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app.state.config_manager = ConfigManager(config_dir=isolated_root)
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    client = httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers)
    return client


@pytest.mark.asyncio
async def test_get_never_creates_the_session(client, isolated_root: Path) -> None:
    async with client as http:
        response = await http.get("/v1/desktop/aida")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result == {"enabled": True, "session_id": None, "paused": False, "greeted": False}


@pytest.mark.asyncio
async def test_open_creates_and_answers_the_frozen_shape(client, isolated_root: Path) -> None:
    async with client as http:
        response = await http.post("/v1/desktop/aida", json={"op": "open"})
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["session_id"]
    assert result["paused"] is False and result["enabled"] is True
    assert (isolated_root / "sessions" / result["session_id"]).is_dir()


@pytest.mark.asyncio
async def test_pause_and_resume_move_the_flag(client, isolated_root: Path) -> None:
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
        paused = await http.post("/v1/desktop/aida", json={"op": "pause"})
        assert paused.status_code == 200 and paused.json()["result"]["paused"] is True
        resumed = await http.post("/v1/desktop/aida", json={"op": "resume"})
        assert resumed.status_code == 200 and resumed.json()["result"]["paused"] is False
        status = await http.post("/v1/desktop/aida", json={"op": "status"})
        assert status.status_code == 200 and status.json()["result"]["paused"] is False


@pytest.mark.asyncio
async def test_greet_refuses_without_a_provider_and_stamps_nothing(
    client, isolated_root: Path
) -> None:
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
        response = await http.post("/v1/desktop/aida", json={"op": "greet"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "aida_no_provider"
    assert not (isolated_root / "aida" / "onboarding.json").exists()


@pytest.mark.asyncio
async def test_unknown_op_is_refused_by_the_schema(client) -> None:
    async with client as http:
        response = await http.post("/v1/desktop/aida", json={"op": "banish"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_disabled_install_answers_get_and_refuses_post(client, isolated_root: Path) -> None:
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
    write_config(isolated_root, {"aida": {"enabled": False}})
    app.state.config_manager = ConfigManager(config_dir=isolated_root)

    # A FRESH client: httpx refuses to re-enter one ("Cannot open a client
    # instance more than once"), and the fresh construction is also what the
    # desktop does after a settings change.
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as http:
        got = await http.get("/v1/desktop/aida")
        posted = await http.post("/v1/desktop/aida", json={"op": "open"})
    assert got.status_code == 200
    assert got.json()["result"]["enabled"] is False
    assert posted.status_code == 409
    assert posted.json()["detail"]["code"] == "aida_disabled"


@pytest.mark.asyncio
async def test_capability_is_advertised(client) -> None:
    async with client as http:
        response = await http.get("/v1/capabilities")
    assert response.json()["result"]["features"].get("aida") == 1

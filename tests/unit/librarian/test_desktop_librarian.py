"""``/v1/desktop/librarian`` — the frozen contract, exercised over the real app.

Drives the route the way the renderer does — ASGI transport, bearer token,
JSON bodies — against an isolated config root, and pins each clause the
Library header was written against: ``GET`` never creates and carries the
nested ``last_run`` plus its flat mirrors, ``open`` ensures, ``run`` arms a
pass now, ``pause``/``resume`` move the flag, a disabled install answers
``enabled: false`` on GET and 409 ``librarian_disabled`` on POST, and a held
store lock refuses with 409 ``librarian_store_busy``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from local_operator.config import ConfigManager
from local_operator.server.app import app
from tests.unit.aida.conftest import write_config
from tests.unit.librarian.conftest import isolated_root_path

TOKEN = "librarian-route-token"


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # A local fixture over the shared body rather than an imported fixture
    # object (F811), the same pattern the Aida route tests use.
    return isolated_root_path(tmp_path, monkeypatch)


@pytest.fixture()
def client(isolated_root: Path):
    """The app over the isolated root, with the desktop plane open."""
    app.state.config_manager = ConfigManager(config_dir=isolated_root)
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}", "Host": "127.0.0.1:1111"}
    return httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers)


@pytest.fixture(autouse=True)
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.mark.asyncio
async def test_get_never_creates_and_carries_the_frozen_shape(client, isolated_root: Path) -> None:
    async with client as http:
        response = await http.get("/v1/desktop/librarian")
    assert response.status_code == 200
    assert response.json()["result"] == {
        "enabled": True,
        "session_id": None,
        "paused": False,
        "name": "Librarian",
        "last_run": None,
        "last_run_at": None,
        "library": {"total": 0, "unorganized": 0, "state": "ready"},
        "unorganized": 0,
        "held": False,
    }
    assert not (isolated_root / "librarian").exists()
    assert not (isolated_root / "sessions").exists()


@pytest.mark.asyncio
async def test_open_ensures_her_session(client, isolated_root: Path) -> None:
    async with client as http:
        opened = await http.post("/v1/desktop/librarian", json={"op": "open"})
    assert opened.status_code == 200
    result = opened.json()["result"]
    assert result["session_id"]
    assert (isolated_root / "sessions" / result["session_id"]).is_dir()


@pytest.mark.asyncio
async def test_status_is_a_read_not_a_create(client, isolated_root: Path) -> None:
    async with client as http:
        response = await http.post("/v1/desktop/librarian", json={"op": "status"})
    assert response.status_code == 200
    assert response.json()["result"]["session_id"] is None


@pytest.mark.asyncio
async def test_run_arms_a_pass_due_now(client, isolated_root: Path) -> None:
    async with client as http:
        opened = await http.post("/v1/desktop/librarian", json={"op": "open"})
        sid = opened.json()["result"]["session_id"]
        ran = await http.post("/v1/desktop/librarian", json={"op": "run"})
    assert ran.status_code == 200
    entry = json.loads((isolated_root / "wakes" / f"{sid}.json").read_text())
    assert [row["id"] for row in entry["schedules"] if row["id"].startswith("librarian-extra-")]


@pytest.mark.asyncio
async def test_pause_then_resume_moves_the_key(client, isolated_root: Path) -> None:
    async with client as http:
        await http.post("/v1/desktop/librarian", json={"op": "open"})
        paused = await http.post("/v1/desktop/librarian", json={"op": "pause"})
        assert paused.status_code == 200 and paused.json()["result"]["paused"] is True
        resumed = await http.post("/v1/desktop/librarian", json={"op": "resume"})
        assert resumed.status_code == 200 and resumed.json()["result"]["paused"] is False


@pytest.mark.asyncio
async def test_a_disabled_install_reads_and_refuses_with_codes(client, isolated_root: Path) -> None:
    write_config(isolated_root, {"librarian": {"enabled": False}})
    async with client as http:
        read = await http.get("/v1/desktop/librarian")
        opened = await http.post("/v1/desktop/librarian", json={"op": "open"})
    assert read.status_code == 200 and read.json()["result"]["enabled"] is False
    assert opened.status_code == 409
    detail = opened.json()["detail"]
    assert detail["code"] == "librarian_disabled"


@pytest.mark.asyncio
async def test_a_held_store_lock_refuses_pause_with_a_code(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.librarian import proactive
    from local_operator.wakes.lock import WakeLockBusy

    async def busy(*_args: object, **_kwargs: object):
        raise WakeLockBusy("Something else on this device is updating the Librarian.")

    monkeypatch.setattr(proactive, "pause", busy)
    async with client as http:
        await http.post("/v1/desktop/librarian", json={"op": "open"})
        refused = await http.post("/v1/desktop/librarian", json={"op": "pause"})
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "librarian_store_busy"


@pytest.mark.asyncio
async def test_get_reports_the_nested_receipt_and_its_flat_mirrors(
    client, isolated_root: Path
) -> None:
    """The desktop-alignment freeze: nested ``last_run`` is authoritative, the
    two flat fields are mirrors of its ``at`` and of ``library.unorganized``."""
    from local_operator.librarian import state as librarian_state
    from local_operator.library.identity import asset_id
    from local_operator.library.metadata import MetadataStore

    at_ms = 1_790_000_000_000
    librarian_state.update_state(isolated_root, session_id="abc123456789", last_run_at=at_ms)
    store = MetadataStore(isolated_root)
    ident = asset_id("p:/tmp/a-report.md")
    store.ensure_asset(ident, "p:/tmp/a-report.md")
    store.update_fields(
        ident,
        {"name": "Weekly report", "description": "The Monday numbers."},
        actor="librarian",
        at=at_ms / 1000.0 + 1,
    )
    async with client as http:
        response = await http.get("/v1/desktop/librarian")
    result = response.json()["result"]
    assert result["last_run"] == {"at": at_ms, "named": 1, "tagged": 0, "failed": 0}
    assert result["last_run_at"] == float(at_ms)
    assert result["unorganized"] == result["library"]["unorganized"]


@pytest.mark.asyncio
async def test_an_unknown_op_is_a_422(client) -> None:
    async with client as http:
        response = await http.post("/v1/desktop/librarian", json={"op": "organize"})
    assert response.status_code == 422


def test_the_capability_map_advertises_her() -> None:
    from local_operator.server.features import feature_flags

    assert feature_flags()["librarian"] == 1

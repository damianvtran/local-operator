"""The phone relay's half of agent pictures: rows carry identity, bytes are served.

The phone reaches only this relay, never the core server, so both halves of the
contract (``local_operator.agent_avatar``) must exist here: ``agent_id`` +
``avatar_sha256`` on ``/api/sessions`` rows, and ``GET /v1/agents/{id}/avatar``
behind relay auth with the core route's ETag/304/404 semantics.
"""

from __future__ import annotations

import hashlib
import io

import pytest
from starlette.testclient import TestClient

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.mobile.daemon import MobileDaemon, SessionTable, build_app
from local_operator.resume import write_session_attachment

Image = pytest.importorskip("PIL.Image")


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (40, 40), (90, 10, 200)).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A config root with one agent ("coder", with a picture) and two sessions."""
    from tests.unit.session.test_catalog_read_failures import _store

    cfg = tmp_path / "config"
    _store(cfg, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    registry = AgentRegistry(config_dir=cfg, refresh_interval=0.0)
    agent = registry.create_agent(AgentEditFields(name="coder"))
    meta = registry.set_agent_avatar(agent.id, _png())
    write_session_attachment(cfg / "sessions" / "aaaaaaaaaaaa", team="", agent="coder", goal="")
    return cfg, agent, meta


@pytest.mark.asyncio
async def test_relay_rows_carry_agent_id_and_avatar_sha(store) -> None:
    _cfg, agent, meta = store
    rows = {row["session_id"]: row for row in await SessionTable().summaries()}
    assert rows["aaaaaaaaaaaa"]["agent_id"] == agent.id
    assert rows["aaaaaaaaaaaa"]["avatar_sha256"] == meta.sha256
    # Unbound: BOTH keys, null -- the phone's merge reads absence as no claim.
    assert rows["bbbbbbbbbbbb"]["agent_id"] is None
    assert rows["bbbbbbbbbbbb"]["avatar_sha256"] is None


def test_relay_avatar_route_auth_etag_304_404(store) -> None:
    cfg, agent, meta = store
    client = TestClient(
        build_app(MobileDaemon(port=0, password="fixture-password")), follow_redirects=False
    )
    url = f"/v1/agents/{agent.id}/avatar"
    # Unauthenticated: a 401, never the login redirect (an <img> cannot log in).
    assert client.get(url).status_code == 401
    assert client.post("/login", data={"password": "fixture-password"}).status_code == 303

    got = client.get(url)
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.headers["etag"] == f'"{meta.sha256}"'
    assert got.headers["cache-control"] == "private, no-cache"
    assert hashlib.sha256(got.content).hexdigest() == meta.sha256

    cached = client.get(url, headers={"if-none-match": got.headers["etag"]})
    assert cached.status_code == 304 and cached.content == b""
    assert cached.headers["etag"] == got.headers["etag"]

    assert client.get("/v1/agents/does-not-exist/avatar").status_code == 404
    AgentRegistry(config_dir=cfg, refresh_interval=0.0).clear_agent_avatar(agent.id)
    gone = client.get(url)
    assert gone.status_code == 404 and gone.json()["code"] == "not_found"


def _login(cfg) -> TestClient:
    client = TestClient(
        build_app(MobileDaemon(port=0, password="fixture-password")), follow_redirects=False
    )
    assert client.post("/login", data={"password": "fixture-password"}).status_code == 303
    return client


def test_relay_route_answers_404_not_500_for_a_nul_in_the_id(store) -> None:
    """QA Q1 / review R6: ``a%00b`` raised a bare ValueError out of ``read_text``."""
    _cfg, agent, _meta = store
    client = _login(_cfg)
    for bad in ("a%00b", "..%00", f"{agent.id}%00"):
        got = client.get(f"/v1/agents/{bad}/avatar")
        assert got.status_code == 404, (bad, got.status_code)
        assert got.json()["code"] == "not_found"


def test_relay_route_refuses_a_symlinked_picture_and_tampered_bytes(store, tmp_path) -> None:
    """Review R2 / QA Q2 on the phone route: nothing outside the agent dir is served."""
    cfg, agent, meta = store
    client = _login(cfg)
    url = f"/v1/agents/{agent.id}/avatar"
    path = cfg / "agents" / agent.id / "avatar.png"
    good = path.read_bytes()
    assert client.get(url).status_code == 200

    path.write_bytes(good + b"appended")
    assert client.get(url, headers={"if-none-match": '"stale"'}).status_code == 404
    path.write_bytes(good)

    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"TOPSECRET-FILE-CONTENT\n")
    path.unlink()
    path.symlink_to(outside)
    got = client.get(url, headers={"if-none-match": '"stale"'})
    assert got.status_code == 404 and b"TOPSECRET" not in got.content

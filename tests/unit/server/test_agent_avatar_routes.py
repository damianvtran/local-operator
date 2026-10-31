"""``/v1/agents/{id}/avatar`` and the ``avatar`` block on agent listings.

Contract: ``local_operator.agent_avatar``. The client builds
``/v1/agents/{id}/avatar?v=<sha256>`` from listing rows, so the listing must
carry the same sha the GET serves as its ETag.
"""

from __future__ import annotations

import hashlib
import io

import pytest

from local_operator.agents import AgentEditFields
from local_operator.server.app import app

Image = pytest.importorskip("PIL.Image")


def _png(color=(10, 20, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (100, 60), color).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def agent_id(test_app_client) -> str:
    return app.state.agent_registry.create_agent(AgentEditFields(name="routebot")).id


@pytest.mark.asyncio
async def test_put_get_304_delete_404(test_app_client, agent_id):
    url = f"/v1/agents/{agent_id}/avatar"
    assert (await test_app_client.get(url)).status_code == 404

    put = await test_app_client.put(url, content=_png(), headers={"content-type": "image/png"})
    assert put.status_code == 200, put.text
    meta = put.json()["result"]["avatar"]
    assert meta["mime"] == "image/png" and len(meta["sha256"]) == 64

    got = await test_app_client.get(url)
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.headers["etag"] == f'"{meta["sha256"]}"'
    # The managed boundary stamps no-store over the route's no-cache for the
    # whole gated /v1/agents family (see the route block's SET vs SERVED note).
    assert got.headers["cache-control"] == "no-store"
    assert hashlib.sha256(got.content).hexdigest() == meta["sha256"]
    assert len(got.content) == meta["bytes"]

    cached = await test_app_client.get(url, headers={"if-none-match": got.headers["etag"]})
    assert cached.status_code == 304 and cached.content == b""
    assert cached.headers["etag"] == got.headers["etag"]
    stale = await test_app_client.get(url, headers={"if-none-match": '"0000"'})
    assert stale.status_code == 200

    listed = await test_app_client.get("/v1/agents", params={"per_page": 100})
    row = next(a for a in listed.json()["result"]["agents"] if a["id"] == agent_id)
    assert row["avatar"]["sha256"] == meta["sha256"]
    single = await test_app_client.get(f"/v1/agents/{agent_id}")
    assert single.json()["result"]["avatar"]["sha256"] == meta["sha256"]

    removed = await test_app_client.delete(url)
    assert removed.status_code == 200 and removed.json()["result"]["removed"] is True
    again = await test_app_client.delete(url)
    assert again.status_code == 200 and again.json()["result"]["removed"] is False
    gone = await test_app_client.get(url)
    assert gone.status_code == 404
    assert gone.json()["detail"]["code"] == "not_found"
    single = await test_app_client.get(f"/v1/agents/{agent_id}")
    assert single.json()["result"]["avatar"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b"plain text, not an image", 415, "unsupported_type"),
        # A real PNG header (sniffs as PNG with dimensions) cut off before the
        # pixel data: refused at decode, not at the type gate.
        (_png()[:40], 422, "undecodable"),
        (b"", 400, "empty_body"),
    ],
)
async def test_put_refusals_carry_reason_codes(test_app_client, agent_id, body, status, code):
    response = await test_app_client.put(f"/v1/agents/{agent_id}/avatar", content=body)
    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["code"] == code and detail["text"]


@pytest.mark.asyncio
async def test_put_over_cap_is_too_large(test_app_client, agent_id, monkeypatch):
    from local_operator import agent_avatar

    monkeypatch.setattr(agent_avatar, "INPUT_MAX_BYTES", 10)
    response = await test_app_client.put(f"/v1/agents/{agent_id}/avatar", content=_png())
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"


@pytest.mark.asyncio
async def test_unknown_agent_is_404(test_app_client):
    for method in ("get", "delete"):
        response = await getattr(test_app_client, method)("/v1/agents/nope/avatar")
        assert response.status_code == 404
    response = await test_app_client.put("/v1/agents/nope/avatar", content=_png())
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_feature_flag_advertised(test_app_client):
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["agent_avatar"] == 1

"""``GET /v1/desktop/sessions/{id}/notification-avatar``: an agent's own banner image.

The desktop main process asks this for a single-session OS banner and passes
the PNG as the notification icon; any non-200 means "post today's banner".
These tests pin the frozen desktop<->core contract: the composite is the bound
agent's picture with the LO badge (and NO agent marker), the query is closed,
every miss is one ``not_found`` refusal, and the ETag moves with the picture,
the size and the theme.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_sessions
from local_operator.team_bridge import notification_avatar as na

#: Real session ids are 12 lowercase hex (``SESSION_ID``); synthetic ones here.
BOUND = "a0a0a0a0a0a1"
UNBOUND = "a0a0a0a0a0a2"
NO_PICTURE = "a0a0a0a0a0a3"
RED = (220, 20, 30)


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch):
    """The desktop sessions router over this test's own isolated config root."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "test-desktop-token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer test-desktop-token"},
    ) as client:
        yield client, tmp_path
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def _session(root: Path, session_id: str, agent: str = "") -> None:
    from local_operator.resume import write_session_attachment
    from local_operator.session.creation import ensure_session_created_at

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    ensure_session_created_at(directory, 1_000.0)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    if agent:
        write_session_attachment(directory, team="", agent=agent, goal="")


def _png(colour: tuple[int, int, int], size: tuple[int, int] = (300, 200)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def seeded(api):
    """One agent WITH a picture (bound to ``BOUND``), one without, one unbound session."""
    from local_operator.agents import AgentRegistry
    from tests.unit.test_agent_avatar import _edit_fields

    client, root = api
    registry = AgentRegistry(config_dir=root, refresh_interval=0.0)
    pictured = registry.create_agent(_edit_fields(name="painter"))
    registry.create_agent(_edit_fields(name="plain"))
    registry.set_agent_avatar(pictured.id, _png(RED))
    _session(root, BOUND, agent="painter")
    _session(root, UNBOUND)
    _session(root, NO_PICTURE, agent="plain")
    return client, root, registry, pictured


def _url(session_id: str, query: str = "size=256&theme=light") -> str:
    return f"/v1/desktop/sessions/{session_id}/notification-avatar?{query}"


def _decode(body: bytes) -> Any:
    image = Image.open(io.BytesIO(body))
    assert image.format == "PNG"
    return image.convert("RGBA")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [64, 256])
@pytest.mark.parametrize("theme", ["light", "dark"])
async def test_the_composite_is_the_agents_picture_with_the_lo_badge(seeded, size, theme) -> None:
    client, _root, _registry, _agent = seeded
    answer = await client.get(_url(BOUND, f"size={size}&theme={theme}"))
    assert answer.status_code == 200, answer.text
    assert answer.headers["content-type"] == "image/png"
    assert answer.headers["x-content-type-options"] == "nosniff"
    image = _decode(answer.content)
    assert image.size == (size, size)

    def px(fx: float, fy: float) -> tuple[int, int, int, int]:
        return image.getpixel((int(size * fx), int(size * fy)))

    # The face is the agent's own picture, circle-cropped: red in the middle,
    # transparent outside the circle.
    assert px(0.4, 0.4)[:3] == pytest.approx(RED, abs=6)
    assert px(0.02, 0.02)[3] == 0
    # The LO badge sits bottom-right on the theme's disc: opaque, not the face.
    fill, ink = na._BADGE_STYLE[theme]
    badge = px(0.86, 0.95)  # inside the disc, below the mark glyph
    assert badge[3] == 255
    assert badge[:3] == pytest.approx(fill[:3], abs=12)
    # NO agent marker: the top-right of the face is still the picture.
    assert px(0.80, 0.20)[:3] == pytest.approx(RED, abs=6)


@pytest.mark.asyncio
async def test_the_body_is_the_shared_renderers_output(seeded) -> None:
    client, root, _registry, agent = seeded
    from local_operator import agent_avatar

    data, _mime, _sha = agent_avatar.read_stored(root, agent.id)
    answer = await client.get(_url(BOUND, "size=64&theme=dark"))
    assert answer.status_code == 200
    assert answer.content == na.compose_agent_picture(data, theme="dark", size=64)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_id",
    [
        "b1b1b1b1b1b1",  # well-formed, no such session
        "not-a-session",  # outside the id grammar: never a path
        UNBOUND,  # no bound agent
        NO_PICTURE,  # bound agent has no picture
    ],
)
async def test_every_miss_is_one_not_found_refusal(seeded, session_id) -> None:
    client, *_ = seeded
    answer = await client.get(_url(session_id))
    assert answer.status_code == 404, answer.text
    assert answer.json()["detail"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_a_picture_that_no_longer_matches_its_digest_is_not_found(seeded) -> None:
    """Tampered bytes are never composited under the recorded sha's ETag."""
    client, root, _registry, agent = seeded
    (root / "agents" / agent.id / "avatar.png").write_bytes(_png((0, 0, 255)))
    answer = await client.get(_url(BOUND))
    assert answer.status_code == 404
    assert answer.json()["detail"]["code"] == "not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "size=256",  # missing theme
        "theme=light",  # missing size
        "",  # nothing
        "size=256&theme=light&v=1",  # unknown name
        "size=256&size=64&theme=light",  # duplicate
        "size=256&theme=light&theme=dark",  # duplicate
        "size=128&theme=light",  # outside the closed size set
        "size=abc&theme=light",
        "size=256&theme=blue",  # outside the closed theme set
    ],
)
async def test_the_query_is_closed(seeded, query) -> None:
    client, *_ = seeded
    answer = await client.get(_url(BOUND, query))
    assert answer.status_code == 422, (query, answer.text)


@pytest.mark.asyncio
async def test_the_closed_query_is_checked_before_the_session(seeded) -> None:
    """A bad query on an unknown session is still the 422: nothing is read first."""
    client, *_ = seeded
    answer = await client.get(_url("b1b1b1b1b1b1", "size=1&theme=light"))
    assert answer.status_code == 422


@pytest.mark.asyncio
async def test_the_etag_moves_with_picture_size_and_theme(seeded) -> None:
    client, _root, registry, agent = seeded
    first = await client.get(_url(BOUND))
    tag = first.headers["etag"]
    assert tag.startswith('"') and tag.endswith('"')
    others = {
        (await client.get(_url(BOUND, "size=64&theme=light"))).headers["etag"],
        (await client.get(_url(BOUND, "size=256&theme=dark"))).headers["etag"],
    }
    assert tag not in others and len(others) == 2

    # Revalidation with the current tag is a 304 and renders nothing.
    same = await client.get(_url(BOUND), headers={"If-None-Match": tag})
    assert same.status_code == 304
    assert same.headers["etag"] == tag
    assert same.content == b""

    # A new picture moves the tag, and the old tag no longer revalidates.
    meta = registry.set_agent_avatar(agent.id, _png((10, 200, 40)))
    moved = await client.get(_url(BOUND), headers={"If-None-Match": tag})
    assert moved.status_code == 200
    assert moved.headers["etag"] != tag
    assert meta.sha256 in moved.headers["etag"]
    assert _decode(moved.content).getpixel((100, 100))[:3] == pytest.approx((10, 200, 40), abs=6)


@pytest.mark.asyncio
async def test_renderer_overload_is_a_transient_503(seeded, monkeypatch) -> None:
    client, *_ = seeded

    async def busy(*_a: Any, **_k: Any) -> bytes:
        raise na.RendererBusy("renderer busy")

    monkeypatch.setattr(na, "render_agent_picture", busy)
    answer = await client.get(_url(BOUND))
    assert answer.status_code == 503
    assert answer.json()["detail"]["code"] == "renderer_busy"


@pytest.mark.asyncio
async def test_the_route_needs_desktop_auth(seeded) -> None:
    client, *_ = seeded
    answer = await client.get(_url(BOUND), headers={"Authorization": "Bearer wrong"})
    assert answer.status_code in (401, 403)


def test_the_capability_is_advertised_to_the_desktop_only() -> None:
    from local_operator.server.features import DESKTOP_ONLY_FEATURES, feature_flags

    assert feature_flags()["desktop_session_notification_avatar"] == 1
    assert "desktop_session_notification_avatar" in DESKTOP_ONLY_FEATURES


def test_compose_agent_picture_refuses_bytes_outside_the_decode_limits() -> None:
    with pytest.raises(na.AvatarImageRejected):
        na.compose_agent_picture(b"not an image", theme="light", size=64)
    with pytest.raises(ValueError):
        na.compose_agent_picture(_png(RED), theme="light", size=128)

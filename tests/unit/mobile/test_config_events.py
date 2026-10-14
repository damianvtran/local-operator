"""The relay's config channel: ``GET /api/config/events`` (wave-5 slice D).

What these pin: the cookie gate; the connect-time BASELINE frame (the current
resolved language with ``changed_keys: []`` — the desktop feed's no-replay
discipline, rendered as an SSE reality); a REAL change producing one frame
through the SHARED payload builder (one contract with the desktop feed — see
``tests/unit/test_i18n_config_frame_contract.py``); and the fan-out's release
when the last subscriber leaves.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request

from local_operator.config_watch import _reset_for_tests, existing_watcher
from local_operator.i18n.frames import config_frame_payload
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, build_app


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "config"
    home.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home))
    monkeypatch.delenv("LOP_LANG", raising=False)
    return home


def _request(authed: bool = True) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if authed:
        headers.append((b"cookie", f"{COOKIE_NAME}={sign_cookie('pw')}".encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/config/events",
            "path_params": {},
            "query_string": b"",
            "headers": headers,
            "scheme": "http",
            "server": ("fixture", 80),
        }
    )


def _endpoint(app: Any):
    return next(r.endpoint for r in app.routes if r.path == "/api/config/events")


def _parse(frame: str) -> dict[str, Any]:
    lines = frame.splitlines()
    assert lines[0] == "event: config", frame
    assert lines[1].startswith("data: "), frame
    return json.loads(lines[1][len("data: ") :])


@pytest.mark.asyncio
async def test_the_route_is_gated(config_home: Path) -> None:
    app = build_app(MobileDaemon(password="pw"))
    response = await _endpoint(app)(_request(authed=False))
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_the_first_frame_is_the_baseline(config_home: Path) -> None:
    """The connect frame: CURRENT resolved value, empty ``changed_keys`` — no
    historical replay, the desktop feed's discipline on an SSE leg."""
    (config_home / "config.yml").write_text("values:\n  language: en\n", encoding="utf-8")
    daemon = MobileDaemon(password="pw")
    app = build_app(daemon)
    response = await _endpoint(app)(_request())
    iterator = response.body_iterator
    try:
        payload = _parse(await anext(iterator))
        assert payload == config_frame_payload((), "en")
    finally:
        await iterator.aclose()
    assert daemon._config_unsubscribe is None  # last-out released the fan-out


@pytest.mark.asyncio
async def test_a_never_pulled_response_attaches_nothing(config_home: Path) -> None:
    """Attach happens on FIRST PULL (CI round 1, MINOR-1): a response that
    never starts — a client that vanished between the call and the first
    read — must leave neither a queue nor the fan-out behind. A route-time
    attach used to leak one subscriber per never-started stream."""
    (config_home / "config.yml").write_text("values:\n  language: en\n", encoding="utf-8")
    daemon = MobileDaemon(password="pw")
    app = build_app(daemon)
    response = await _endpoint(app)(_request())
    assert daemon.config_subscribers == set()
    assert daemon._config_unsubscribe is None
    await response.body_iterator.aclose()
    assert daemon.config_subscribers == set()


@pytest.mark.asyncio
async def test_a_real_change_publishes_through_the_shared_builder(config_home: Path) -> None:
    """A change detected against the FILE (a hand edit + the watcher's poll)
    produces one frame via the shared payload builder, naming the changed key
    and the freshly resolved tag. ``fr`` now resolves to ``fr`` — wave 1 ships
    the locale, so the frame's tag follows it (the shipped filter and the
    announcement contract are unchanged)."""
    (config_home / "config.yml").write_text("values:\n  language: en\n", encoding="utf-8")
    daemon = MobileDaemon(password="pw")
    app = build_app(daemon)
    response = await _endpoint(app)(_request())
    iterator = response.body_iterator
    try:
        assert _parse(await anext(iterator)) == config_frame_payload((), "en")
        (config_home / "config.yml").write_text("values:\n  language: fr\n", encoding="utf-8")
        watcher = existing_watcher(config_home)
        assert watcher is not None
        watcher.poll_now()

        frame = await asyncio.wait_for(anext(iterator), timeout=5)
        assert _parse(frame) == config_frame_payload(("language",), "fr")
    finally:
        await iterator.aclose()


@pytest.mark.asyncio
async def test_the_fan_out_releases_only_on_the_last_subscriber(config_home: Path) -> None:
    (config_home / "config.yml").write_text("values:\n  language: en\n", encoding="utf-8")
    daemon = MobileDaemon(password="pw")
    app = build_app(daemon)
    endpoint = _endpoint(app)
    first = await endpoint(_request())
    second = await endpoint(_request())
    try:
        await anext(first.body_iterator)
        await anext(second.body_iterator)
        assert daemon._config_unsubscribe is not None
        await first.body_iterator.aclose()
        assert daemon._config_unsubscribe is not None  # one subscriber still holds it
        await second.body_iterator.aclose()
        assert daemon._config_unsubscribe is None
    finally:
        await first.body_iterator.aclose()
        await second.body_iterator.aclose()

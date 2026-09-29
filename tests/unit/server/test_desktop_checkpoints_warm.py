"""``sessions.checkpoints.warm`` over the real router: the daemon's half (D2/D9).

The route → adapter → owner contract is what this file pins. Nothing here can
serve a real session, so the bridge's ``remote`` is the one faked part — the
adapter's own reachability, payload and parse logic runs for real, the route's
real validation and error ladder run, and the bytes the owner receives are
asserted exactly, so the JSON ``checkpoint_naming`` parses cannot silently
drift from the JSON the route sends.

The cold and peer answers are asserted here rather than left to the session
suites because they are the ROUTE's contract: a rail gesture is decoration,
and neither case may become an error the user has to read.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessionBridge

pytestmark = pytest.mark.asyncio

TOKEN = "desktop-checkpoints-warm-route-token"


@pytest.fixture(autouse=True)
def _desktop_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the desktop plane the way its own app does (test_desktop_pins' shape)."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


class FakeRemote:
    """The two facts the adapter reads: reachability, and the routed slash."""

    def __init__(self, outcome: Any = None, *, reachable: bool = True) -> None:
        self._outcome = outcome if outcome is not None else {"kind": "block", "data": {}}
        self._reachable = reachable
        self.slashes: list[tuple[str, str]] = []

    @property
    def owner_reachable(self) -> bool:
        return self._reachable

    async def route_shared_slash(self, command: str, args: str) -> Any:
        self.slashes.append((command, args))
        return self._outcome


class FakeBridge:
    """A bridge whose ``checkpoints_warm`` is the REAL adapter method.

    Duck-typed deliberately: the adapter touches ``remote_row`` and ``remote``
    and nothing else, and calling the real bound-unbound method here is what
    makes the payload/parse assertions below real.
    """

    def __init__(self, remote: FakeRemote | None = None, *, remote_row: Any = None) -> None:
        self.remote = remote
        self.remote_row = remote_row

    async def checkpoints_warm(
        self, *, ids: list[str] | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        adapter = DesktopSessionBridge.checkpoints_warm
        return await adapter(cast(DesktopSessionBridge, self), ids=ids, limit=limit)


class FakePool:
    """``DesktopSessions``-shaped: the route's only door to a session."""

    def __init__(self, bridges: dict[str, FakeBridge]) -> None:
        self.bridges = bridges

    @contextlib.asynccontextmanager
    async def session(
        self, session_id: str, *, read: bool = False, allow_draft: bool = False
    ) -> AsyncIterator[FakeBridge]:
        del read, allow_draft
        bridge = self.bridges.get(session_id)
        if bridge is None:
            raise KeyError("Unknown session")
        yield bridge


def _app(tmp_path: Path, bridges: dict[str, FakeBridge]) -> FastAPI:
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path / "cfg")
    app.state.desktop_sessions = FakePool(bridges)
    app.include_router(desktop_sessions.router)
    return app


@contextlib.asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client


def _result(response) -> dict[str, Any]:
    body = response.json()
    assert body["status"] == 200, body
    return body["result"]


async def test_a_warm_round_trips_the_receipt_and_records_the_payload(tmp_path: Path) -> None:
    remote = FakeRemote({"kind": "block", "data": {"accepted": ["a4", "a2"], "pending": ["a4"]}})
    app = _app(tmp_path, {"s1": FakeBridge(remote)})
    async with _client(app) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/checkpoints/warm", json={"ids": ["a4", "a2"]}
        )
    assert response.status_code == 200
    assert _result(response) == {"accepted": ["a4", "a2"], "pending": ["a4"]}
    # The owner's wire, byte-for-byte: the module on the other side parses this.
    assert remote.slashes == [
        ("checkpoints_warm", json.dumps({"ids": ["a4", "a2"], "limit": None}))
    ]


async def test_no_ids_serialise_as_null_and_a_limit_passes_through(tmp_path: Path) -> None:
    remote = FakeRemote({"kind": "block", "data": {"accepted": ["a9"], "pending": ["a9"]}})
    app = _app(tmp_path, {"s1": FakeBridge(remote)})
    async with _client(app) as client:
        response = await client.post("/v1/desktop/sessions/s1/checkpoints/warm", json={"limit": 3})
    assert response.status_code == 200
    assert _result(response) == {"accepted": ["a9"], "pending": ["a9"]}
    assert remote.slashes == [("checkpoints_warm", json.dumps({"ids": None, "limit": 3}))]


@pytest.mark.parametrize(
    "body",
    [
        {"ids": [f"id-{n}" for n in range(17)]},  # over the module's 16-id cap
        {"ids": ["a1"], "limit": 17},  # over the 16 limit cap
        {"limit": 0},  # a bound of zero names nothing
    ],
)
async def test_bounds_are_a_422_not_a_silent_truncation(tmp_path: Path, body) -> None:
    app = _app(tmp_path, {"s1": FakeBridge(FakeRemote())})
    async with _client(app) as client:
        response = await client.post("/v1/desktop/sessions/s1/checkpoints/warm", json=body)
    assert response.status_code == 422, response.text


async def test_an_unknown_session_is_a_404(tmp_path: Path) -> None:
    app = _app(tmp_path, {})
    async with _client(app) as client:
        response = await client.post(
            "/v1/desktop/sessions/nope/checkpoints/warm", json={"ids": ["a1"]}
        )
    assert response.status_code == 404, response.text


async def test_a_cold_conversation_answers_empty_and_dials_nothing(tmp_path: Path) -> None:
    """No owner to run an errand: the empty receipt, never a 5xx.

    The rail's fallback text is the design's own degrade (D2/R5); starting a
    runtime for decoration would be the bug this shape avoids.
    """
    remote = FakeRemote(reachable=False)
    app = _app(tmp_path, {"s1": FakeBridge(remote)})
    async with _client(app) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/checkpoints/warm", json={"ids": ["a1"]}
        )
    assert response.status_code == 200
    assert _result(response) == {"accepted": [], "pending": []}
    assert remote.slashes == []


async def test_a_bridge_with_no_remote_at_all_answers_empty(tmp_path: Path) -> None:
    app = _app(tmp_path, {"s1": FakeBridge(None)})
    async with _client(app) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/checkpoints/warm", json={"ids": ["a1"]}
        )
    assert response.status_code == 200
    assert _result(response) == {"accepted": [], "pending": []}


async def test_a_peer_conversation_answers_empty(tmp_path: Path) -> None:
    """v1 names on the device that holds the journal only (D4)."""
    remote = FakeRemote()
    app = _app(tmp_path, {"s1": FakeBridge(remote, remote_row=SimpleNamespace())})
    async with _client(app) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/checkpoints/warm", json={"ids": ["a1"]}
        )
    assert response.status_code == 200
    assert _result(response) == {"accepted": [], "pending": []}
    assert remote.slashes == []


async def test_an_owner_that_cannot_run_errands_is_a_soft_empty(tmp_path: Path) -> None:
    """The no-provider handshake: an error receipt the route must not 5xx.

    The serving side answers ``naming_unavailable`` when the session has no
    callable errand seam; decoration failing is not an error the renderer
    should paint, so the route degrades to the same empty receipt as cold.
    """
    remote = FakeRemote({"kind": "error", "data": {"code": "naming_unavailable"}})
    app = _app(tmp_path, {"s1": FakeBridge(remote)})
    async with _client(app) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/checkpoints/warm", json={"ids": ["a1"]}
        )
    assert response.status_code == 200
    assert _result(response) == {"accepted": [], "pending": []}


async def test_an_unusable_owner_reply_is_a_soft_empty(tmp_path: Path) -> None:
    remote = FakeRemote("not a mapping at all")
    app = _app(tmp_path, {"s1": FakeBridge(remote)})
    async with _client(app) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/checkpoints/warm", json={"ids": ["a1"]}
        )
    assert response.status_code == 200
    assert _result(response) == {"accepted": [], "pending": []}

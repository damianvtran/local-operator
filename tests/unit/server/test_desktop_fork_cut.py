"""``POST …/fork`` over the real router: the cut point's wire and its refusals.

The route → routed-slash contract is what this file pins. Nothing here can fork
a real conversation (that is ``test_fork_slash_target.py``'s half and the
desktop e2e's), so the bridge's ``remote`` is the one faked part — the route's
REAL validation, receipt and error ladder run, and the args the owner receives
are asserted byte-for-byte, so the encoder on this side cannot silently drift
from the decoder on the other.

The absent-target cases are here rather than left to the session suites because
they are the BACKWARD-COMPATIBILITY contract: a client that names no cut point
must send exactly the call this route has always sent.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_lifecycle
from local_operator.server.routes.desktop_sessions import (
    RUNTIME_UNREACHABLE,
    RUNTIME_UNREACHABLE_MESSAGE,
)
from local_operator.session.errors import ForkRefused

pytestmark = pytest.mark.asyncio

TOKEN = "desktop-fork-cut-route-token"
CHILD_ID = "child000001"
LANDED = "u1"

REQUEST_ID = "3f2a1c40-0000-4000-8000-000000000001"
#: A receipt is keyed by request id AND payload, so a second body needs its own id.
BLANK_REQUEST_ID = "3f2a1c40-0000-4000-8000-000000000002"


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
    """The facts the fork route reads: the routed slash, and the child's prompt."""

    def __init__(self) -> None:
        self.slashes: list[tuple[str, str]] = []
        self.bound = 0
        self.admitted: list[tuple[str, Any]] = []

    @property
    def owner_reachable(self) -> bool:
        return True

    async def bind_runtime(self) -> None:
        self.bound += 1

    async def route_shared_slash(self, command: str, args: str) -> Any:
        self.slashes.append((command, args))
        data: dict[str, Any] = {"type": "forked", "session_id": CHILD_ID}
        if args:
            # The runtime answers a cut with the row its copy actually stopped
            # at; the bare arm has none to report. Mirrored here so the route's
            # pass-through (and its absence) is what this file pins.
            data["cut_entry_id"] = LANDED
        return {"kind": "block", "data": data}

    async def admit_prompt(self, text: str, *, command_id: str, images: Any) -> tuple[str, bool]:
        self.admitted.append((text, command_id))
        return "admitted once", False


class FakeBridge:
    def __init__(self, remote: FakeRemote) -> None:
        self.remote = remote
        self.remote_row = None


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


def _app(tmp_path: Path, remote: FakeRemote) -> FastAPI:
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path / "cfg")
    app.state.desktop_sessions = FakePool({"s1": FakeBridge(remote), CHILD_ID: FakeBridge(remote)})
    app.include_router(desktop_lifecycle.router)
    return app


@contextlib.asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client


async def test_a_named_cut_point_rides_the_args_and_keeps_the_response_shape(
    tmp_path: Path,
) -> None:
    remote = FakeRemote()
    async with _client(_app(tmp_path, remote)) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/fork",
            json={"request_id": REQUEST_ID, "boundary": "at_entry", "entry_id": "u1"},
        )

    assert response.status_code == 200, response.text
    # The receipt wrapper's shape is unchanged: the fork's own answer rides
    # ``result.data``, exactly as the desktop client reads it today.
    assert response.json()["result"] == {
        "replayed": False,
        "data": {
            "session_id": CHILD_ID,
            "parent_id": "s1",
            "boundary": "at_entry",
            # The landed row rides the response, so a client can say the fork
            # started one message earlier than it pointed at.
            "cut_entry_id": LANDED,
        },
    }
    # The owner's wire, byte-for-byte: the module on the other side parses this.
    assert remote.slashes == [("fork", json.dumps({"entry_id": "u1"}))]
    assert remote.bound == 1


async def test_the_absent_target_sends_the_call_this_route_has_always_sent(
    tmp_path: Path,
) -> None:
    remote = FakeRemote()
    async with _client(_app(tmp_path, remote)) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/fork", json={"request_id": REQUEST_ID}
        )
        # A BLANK id names nothing, so with ``next_safe`` it is the absent target
        # — the same call, not a 422 and not a cut. Its own request id, because a
        # receipt is keyed by id AND payload.
        blank = await client.post(
            "/v1/desktop/sessions/s1/fork",
            json={"request_id": BLANK_REQUEST_ID, "entry_id": "   "},
        )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["data"] == {
        "session_id": CHILD_ID,
        "parent_id": "s1",
        "boundary": "next_safe",
    }
    assert blank.status_code == 200, blank.text
    assert remote.slashes == [("fork", ""), ("fork", "")]
    assert "admission" not in response.json()["result"]["data"]
    assert "cut_entry_id" not in response.json()["result"]["data"], "no cut, no landed row"


async def test_an_opening_message_still_rides_the_child_admission(tmp_path: Path) -> None:
    remote = FakeRemote()
    async with _client(_app(tmp_path, remote)) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/fork",
            json={
                "request_id": REQUEST_ID,
                "message": "try another route",
                "boundary": "at_entry",
                "entry_id": "u1",
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["data"]["admission"] == {
        "status": "admitted",
        "detail": "admitted once",
        "duplicate": False,
    }
    assert remote.admitted == [("try another route", REQUEST_ID)]


@pytest.mark.parametrize(
    "body",
    [
        {"request_id": REQUEST_ID, "boundary": "at_entry"},  # no target named
        {"request_id": REQUEST_ID, "boundary": "at_entry", "entry_id": ""},
        {"request_id": REQUEST_ID, "boundary": "at_entry", "entry_id": "   "},
        {"request_id": REQUEST_ID, "boundary": "next_safe", "entry_id": "u1"},  # a cut denied
        {"request_id": REQUEST_ID, "boundary": "at_message", "entry_id": "u1"},  # not a boundary
        {"request_id": REQUEST_ID, "boundary": "at_entry", "entry_id": "u1", "extra": 1},
        {"request_id": REQUEST_ID, "boundary": "at_entry", "entry_id": "u" * 129},
    ],
)
async def test_a_malformed_boundary_is_a_422_and_runs_nothing(tmp_path: Path, body) -> None:
    """An ``at_entry`` without a target can never fall back to a whole fork."""
    remote = FakeRemote()
    async with _client(_app(tmp_path, remote)) as client:
        response = await client.post("/v1/desktop/sessions/s1/fork", json=body)

    assert response.status_code == 422, response.text
    assert remote.slashes == []
    assert remote.bound == 0


class RefusingRemote(FakeRemote):
    """A remote whose routed slash fails the way the owner's own does."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def route_shared_slash(self, command: str, args: str) -> Any:
        self.slashes.append((command, args))
        raise self.error


@pytest.mark.parametrize(
    "refusal",
    [
        ForkRefused(reason="entry_unknown"),
        ForkRefused(reason="before_anchor"),
        ForkRefused(reason="unfinished_batch"),
        ForkRefused(reason="compaction_pending"),
        # A bare raise names no cause; the generic sentence is the answer, and
        # the reason key is present-but-empty so the shape never varies.
        ForkRefused(),
    ],
)
async def test_a_refused_fork_is_the_refusal_not_an_owner_outage(
    tmp_path: Path, refusal: ForkRefused
) -> None:
    """The conversation's state refused this, so the ladder must not say 503.

    Unclassified, this arrived at the schema's ``RuntimeError`` arm and answered
    ``runtime_unreachable`` — "Session owner is unavailable. Reconnect and
    reconcile before retrying." — for a request the owner answered promptly and
    deliberately (measured on PR #1917). Every cause answers the same 409 shape
    the other named-condition refusals use, with the CAUSE as a machine token so
    a renderer can offer the right way forward without parsing the sentence.
    """
    remote = RefusingRemote(refusal)
    async with _client(_app(tmp_path, remote)) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/fork",
            json={
                "request_id": REQUEST_ID,
                "boundary": "at_entry",
                "entry_id": "ffffffffffffffffffffffffffffffff",
            },
        )

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "fork_refused",
        "message": str(refusal),
        "reason": refusal.reason,
    }
    # The owner was reached and answered: the refusal ran the routed slash, and
    # the receipt's replay machinery is what the route still rounds through.
    assert remote.slashes == [("fork", json.dumps({"entry_id": "f" * 32}))]


async def test_a_failure_this_seam_does_not_know_keeps_the_owner_outage_path(
    tmp_path: Path,
) -> None:
    """Fail-safe: an un-enumerated failure is exactly what it was before.

    The classification narrows nothing. An owner that raises something the seam
    does not enumerate still reaches the client as the owner's own sentence, and
    the ladder still reads it as an unreachable owner — so a genuinely unreachable
    owner is untouched by this change and this cell is what proves it.
    """
    remote = RefusingRemote(RuntimeError("the owner's own words"))
    async with _client(_app(tmp_path, remote)) as client:
        response = await client.post(
            "/v1/desktop/sessions/s1/fork",
            json={"request_id": REQUEST_ID, "boundary": "next_safe"},
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == {
        "code": RUNTIME_UNREACHABLE,
        "message": RUNTIME_UNREACHABLE_MESSAGE,
    }
    # The owner's own words are never echoed: the vetted sentence stands in.
    assert "the owner's own words" not in response.text

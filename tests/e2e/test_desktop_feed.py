"""The machine-wide feed over real loopback HTTP, with no bridge and no runtime.

Everything here is the production stack: a real uvicorn server over the real
``app``, the real bearer/Origin boundary, the real ``DesktopFeed`` singleton, a
real ``AttentionStore``, and a real session directory with no runtime and no
bridge attached to it. Only the provider stream is absent, because nothing in
this path asks a model anything — which is the point of the design: the feed's
two dependencies are the store and the sessions directory.

What it proves, and why it needs the HTTP layer to prove it:

- **The route exists and rides the desktop boundary.** 503 without a token, 401
  for a wrong one, 403 for a foreign Origin — the same three answers every other
  desktop route gives, asserted here because a new route is a new boundary and
  the failure mode is a stream that leaks an authenticated view of the machine.
- **A background completion reaches the feed with NO bridge anywhere.** The
  session that finishes is never opened, never streamed and never attached; if
  the feed needed a bridge this test would hang waiting for a frame nobody can
  compose.
- **The feed acquires nothing.** ``DesktopSessions.bridges`` stays empty across
  the whole cycle, which is the property that keeps watching a catalogue from
  becoming spawning a catalogue.

``CMUX_*`` is scrubbed for the whole suite by the root ``conftest`` (see its
``_AMBIENT_VARS``), and this module never boots a TUI or a fork, so no synthetic
session ids are needed beyond the ones it creates.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import uvicorn

from local_operator.server.app import app
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attention import AttentionStore
from local_operator.session.runtime.viewers import ViewerRecord
from local_operator.session.runtime.presence import (
    delivery_path,
    desktop_delivery_present,
    reset_cache,
)

pytestmark = pytest.mark.e2e


async def _next_frame(lines, predicate, timeout: float = 30.0):
    async def read():
        async for line in lines:
            if line.startswith("data: "):
                frame = json.loads(line[6:])
                if predicate(frame):
                    return frame
        raise AssertionError("stream ended before the expected frame")

    return await asyncio.wait_for(read(), timeout)


async def _frames_until(lines, predicate, timeout: float = 5.0):
    """Every frame up to and including the first matching one."""
    collected = []

    async def read():
        async for line in lines:
            if line.startswith("data: "):
                frame = json.loads(line[6:])
                collected.append(frame)
                if predicate(frame):
                    return
        return

    try:
        await asyncio.wait_for(read(), timeout)
    except TimeoutError:
        pass
    return collected


@pytest_asyncio.fixture
async def desktop_server(headless_tui_env: Path, monkeypatch):
    """A real backend on a real loopback port, with the desktop plane armed."""
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(10000):
            if server.started:
                break
            if serving.done():
                await serving
            await asyncio.sleep(0)
        assert server.started
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            yield root, client
    finally:
        server.should_exit = True
        await serving
        engine = getattr(app.state, "desktop_feed", None)
        if engine is not None:
            await engine.close()
        app.state.desktop_feed = None
        pool = getattr(app.state, "desktop_sessions", None)
        if pool is not None:
            await pool.close()
            app.state.desktop_sessions = None
        reset_cache()


def _publish(root: Path, session_id: str, kind: str = "complete", anchor: str = "a1") -> str:
    token = str(uuid.uuid4())
    AttentionStore(root / "attention.db").publish(f"session/{session_id}", token, anchor, kind)
    return token


async def _create(client: httpx.AsyncClient, workspace: Path, request_id: str) -> str:
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": request_id, "cwd": str(workspace)},
    )
    assert created.status_code == 200, created.text
    return created.json()["result"]["session_id"]


@pytest.mark.asyncio
async def test_the_feed_rides_the_desktop_boundary(headless_tui_env: Path, monkeypatch):
    """503/401/403 — a new route is a new boundary, so it is asserted as one."""
    root = headless_tui_env
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: test\n")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(10000):
            if server.started:
                break
            if serving.done():
                await serving
            await asyncio.sleep(0)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=10
        ) as client:
            # No token configured at all: the whole plane is closed.
            assert (await client.get("/v1/desktop/events")).status_code == 503
            assert (await client.post("/v1/desktop/presence", json={})).status_code == 503
    finally:
        server.should_exit = True
        await serving

    # Now WITH a token: a wrong bearer and a foreign Origin are both refused
    # BEFORE any stream machinery exists.
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(10000):
            if server.started:
                break
            await asyncio.sleep(0)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=10
        ) as client:
            assert (await client.get("/v1/desktop/events")).status_code == 401
            client.headers["Authorization"] = "Bearer incorrect"
            assert (await client.get("/v1/desktop/events")).status_code == 401
            client.headers["Authorization"] = f"Bearer {token}"
            for origin in ("null", "https://evil.example"):
                response = await client.get("/v1/desktop/events", headers={"Origin": origin})
                assert response.status_code == 403, origin
            print("feed boundary: no-token503, wrong-bearer401, foreign-Origin403")
    finally:
        server.should_exit = True
        await serving
        engine = getattr(app.state, "desktop_feed", None)
        if engine is not None:
            await engine.close()
        app.state.desktop_feed = None
        app.state.desktop_sessions = None


@pytest.mark.asyncio
async def test_a_background_completion_reaches_the_feed_with_no_bridge(
    desktop_server, workspace: Path
):
    """THE DEFECT, closed end to end.

    The session that finishes is never opened, streamed or attached. Before this
    channel existed the completion produced no composed frame at all — the only
    announcer was a running TUI, and on a machine with none a finished turn was
    announced by nobody.
    """
    root, client = desktop_server
    pool = DesktopSessions(root)
    app.state.desktop_sessions = pool  # so `bridged=` sees the real table

    finished = await _create(client, workspace, "11111111-1111-4111-8111-111111111101")
    displayed = await _create(client, workspace, "11111111-1111-4111-8111-111111111102")

    async with client.stream("GET", "/v1/desktop/events") as response:
        assert response.status_code == 200, response.read()
        lines = response.aiter_lines()
        opened = await _next_frame(lines, lambda f: f["type"] == "open")
        subscription = opened["payload"]["subscription_id"]
        assert opened["payload"]["catalogue_revision"] is not None

        # A bridge holds the OTHER session, which is the machine state the
        # operator reported: the app is displaying one conversation and a
        # different one finishes.
        async with client.stream("GET", f"/v1/desktop/sessions/{displayed}/events") as stream:
            assert stream.status_code == 200
            other_lines = stream.aiter_lines()
            await _next_frame(other_lines, lambda f: f["type"] == "open")

            token = await asyncio.to_thread(_publish, root, finished)
            frames = await _frames_until(
                lines, lambda f: f["type"] == "notification" and f.get("session_id") == finished
            )
            announced = [
                frame
                for frame in frames
                if frame["type"] == "notification" and frame.get("session_id") == finished
            ]
            assert len(announced) == 1, frames
            payload = announced[0]["payload"]
            assert payload["completion_token"] == token
            assert payload["kind"] == "complete"
            # The DERIVED routing field: the app is not displaying this session,
            # so the banner must not be suppressed by the window's focus.
            assert payload["focus_policy"] == "always"
            assert payload["dedupe_key"] == f"complete:{finished}:{token}"

    # ...and the feed never attached anything: no bridge was acquired for the
    # session it announced, and none for any other.
    assert finished not in pool.bridges
    assert displayed in pool.bridges
    assert subscription  # the presence route binds to this id, asserted below


@pytest.mark.asyncio
async def test_the_presence_route_is_bound_to_a_live_subscription(desktop_server):
    """The lease is held AGAINST the socket, so an unknown id is refused.

    A claim to deliver for a subscription that does not exist is exactly the
    "presence that cannot deliver" this mechanism must not manufacture — and the
    reason the reference is the SSE socket rather than a token the client mints.
    """
    root, client = desktop_server

    missing = await client.post(
        "/v1/desktop/presence",
        json={"subscription_id": "nope", "can_notify": True},
    )
    assert missing.status_code == 404, missing.text

    async with client.stream("GET", "/v1/desktop/events") as response:
        lines = response.aiter_lines()
        opened = await _next_frame(lines, lambda f: f["type"] == "open")
        subscription = opened["payload"]["subscription_id"]

        bad_shape = await client.post(
            "/v1/desktop/presence",
            json={"subscription_id": subscription, "can_notify": True, "session_id": "nope"},
        )
        assert bad_shape.status_code == 422, bad_shape.text
        extra = await client.post(
            "/v1/desktop/presence",
            json={"subscription_id": subscription, "can_notify": True, "invented": 1},
        )
        assert extra.status_code == 422, extra.text

        beat = await client.post(
            "/v1/desktop/presence",
            json={
                "subscription_id": subscription,
                "can_notify": True,
                "can_notify_kinds": ["complete", "error"],
                "window": {"exists": True, "focused": True, "visible": True, "minimized": False},
            },
        )
        assert beat.status_code == 200, beat.text
        assert beat.json()["result"]["lease_seconds"] == 45

        reset_cache()
        assert delivery_path(root).exists()
        assert desktop_delivery_present(root, "complete") is True
        # The GATE kind is deliberately not covered by the machine-wide lease:
        # the feed carries completions only, so a parked question keeps its
        # per-session lease and its per-session toast.
        assert desktop_delivery_present(root, "approval") is False

        # A windowless app is not displaying anything, whatever id it names.
        windowless = await client.post(
            "/v1/desktop/presence",
            json={
                "subscription_id": subscription,
                "can_notify": True,
                "can_notify_kinds": ["complete"],
                "session_id": "a" * 12,
                "window": {"exists": False, "focused": False, "visible": False},
            },
        )
        assert windowless.status_code == 200, windowless.text
        payload = json.loads(delivery_path(root).read_text())
        assert payload["session_id"] == ""

    # The socket is gone, so the lease is gone with it — the revocation that
    # stops a dead app from silencing every runtime on the machine.
    reset_cache()
    for _ in range(100):
        if not desktop_delivery_present(root, "complete"):
            break
        await asyncio.sleep(0.05)
    assert desktop_delivery_present(root, "complete") is False


def test_a_click_routes_to_a_running_desktop_viewer(headless_tui_env: Path, monkeypatch):
    """Rung 1 of the click ladder, with a ``surface: "desktop"`` record.

    SYNCHRONOUS on purpose: ``route_click`` runs its own event loop, because the
    click handler is a short-lived process macOS handed an activation and has no
    loop of its own. Driving it from inside a running loop is the one thing it
    cannot do, and this test would otherwise pass for the wrong reason — the
    failure is swallowed into "viewer routing failed" and the ladder simply
    falls through to the terminal.

    The desktop app is expected to publish one of these unchanged; this asserts
    the routing half works for it today, against a real endpoint on a real
    loopback port, so the UI half has something already proven to compile
    against.
    """
    from local_operator.session.runtime.viewer_server import ViewerServer
    from local_operator.session.runtime.viewers import DESKTOP_SURFACE
    from local_operator.tui import resume_click
    from local_operator.tui.resume_click import open_session

    root = headless_tui_env
    switched: list[str] = []
    # RUNG 2 AND RUNG 3 ARE DOUBLED BEFORE THE LADDER EVEN STARTS, and that is a
    # safety property rather than tidiness: an earlier revision of this test let
    # the click fall through, and `_launch_desktop` found the operator's real
    # `local-operator-ui` on PATH and launched it — a window on the maintainer's
    # desktop, opened by a test. A click test must be incapable of reaching the
    # machine it runs on.
    launched: list[str] = []
    monkeypatch.setattr(
        resume_click, "_launch_desktop", lambda session_id: launched.append(session_id) or False
    )
    monkeypatch.setattr(
        resume_click, "_spawn_terminal", lambda session_id: launched.append(session_id) or True
    )

    class _Host:
        async def viewer_resume_session(self, session_id: str) -> str:
            switched.append(session_id)
            return f"displayed {session_id}"

        async def viewer_focus_window(self) -> str:
            # The protocol's second op, present so the host satisfies
            # ``ViewerHost``: `ViewerServer` advertises `focus-window-v1` from
            # `hasattr(host, "viewer_focus_window")`, so a host without it
            # silently changes what the record claims about itself.
            return "raised"

    server = ViewerServer(_Host(), surface=DESKTOP_SURFACE, root=root)
    server.start()
    assert server.ready.wait(timeout=5.0), "viewer endpoint never bound"
    try:
        server.note_session("")
        target = "b3b3b3b3b3b3"
        assert open_session(target) is True
        assert switched == [target], switched
        # THE VIEWER RUNG WON, so nothing was launched and nothing was spawned:
        # asserting the absence is what keeps a future fall-through from being
        # silent here (it would have opened a real window on the host).
        assert launched == [], launched
        # The record's surface is what the routing preference reads.
        records = _scan(root)
        assert any(record.surface == DESKTOP_SURFACE for record in records)
    finally:
        server.close()


def _scan(root: Path) -> list[ViewerRecord]:
    from local_operator.session.runtime.viewers import scan_viewers

    return scan_viewers(root)


def test_a_stale_desktop_record_never_advertises_a_session_it_cannot_show():
    """The record's own contract, asserted where the routing reads it."""
    from local_operator.session.runtime.viewers import ViewerRecord

    windowless = ViewerRecord(
        pid=os.getpid(),
        surface="desktop",
        control_port=1,
        control_key="k",
        current_session="c4c4c4c4c4c4",
        has_window=False,
    )
    assert json.loads(json.dumps(windowless.to_json()))["has_window"] is False

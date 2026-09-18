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
import contextlib
import json
import os
import secrets
import socket
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import uvicorn

from local_operator.server.app import app
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attention import AttentionStore
from local_operator.session.creation import CREATED_AT_NAME
from local_operator.session.runtime import registry
from local_operator.session.runtime.presence import (
    delivery_dir,
    delivery_path,
    desktop_delivery_present,
    reset_cache,
)
from local_operator.session.runtime.types import SessionRecord
from local_operator.session.runtime.viewers import ViewerRecord

pytestmark = pytest.mark.e2e


def _published_record(root: Path) -> dict[str, Any]:
    """The ONE presence record this server published, as a dict (review R6).

    A publisher owns a record per PROCESS now, in ``run/desktop/delivery/``,
    and the legacy ``delivery.json`` is read-only — nothing writes it any more.
    So the question these tests used to ask of a fixed path — "what did the
    route just publish?" — has to be asked of the directory.

    Asserting EXACTLY ONE record is the point rather than a convenience: if a
    single server process ever published two, its own reader would union them
    and a stale window could outvote the live one.
    """
    records = sorted(delivery_dir(root).glob("*.json"))
    assert len(records) == 1, [path.name for path in records]
    return json.loads(records[0].read_text())


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


def _publish_record(root: Path, session_id: str, **fields: Any) -> Path:
    """The runtime's OWN record write — the only trace a gate edge leaves.

    Written through ``registry.publish`` rather than by hand so the staged write
    + rename really happens: that rename is what moves ``run/mobile``'s mtime,
    which is the signal the feed's doorbell is built on. This process's pid
    stands in for the runtime's, which is all ``classify`` reads.
    """
    return registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="tui",
            session_id=session_id,
            conversation_name="a conversation",
            cwd=str(root),
            model_label="mock",
            control_port=1,
            control_key="0" * 64,
            **fields,
        ),
        root,
    )


@pytest.mark.asyncio
async def test_an_answered_gate_reaches_the_feed_and_the_lists_stamp_agrees(
    desktop_server, workspace: Path
):
    """THE REPORTED SYMPTOM, end to end, with both writers compared.

    The row's status could only ever be refreshed by re-reading the whole list, so
    an answered gate on a row the user was not looking at took up to 30 s to
    appear. This drives the real uvicorn app, the real SSE stream and the real
    discovery-record write a runtime makes, and asserts the two halves of the
    contract: the frame arrives within the doorbell's own clock, and
    ``GET /v1/desktop/sessions`` carries the same counter the last frame did —
    which is what lets a client discard a list computed before a frame it has
    already applied.
    """
    root, client = desktop_server
    session_id = await _create(client, workspace, "33333333-3333-4333-8333-333333333301")

    async with client.stream("GET", "/v1/desktop/events") as response:
        assert response.status_code == 200, response.read()
        lines = response.aiter_lines()
        await _next_frame(lines, lambda f: f["type"] == "open")

        # The gate is PARKED: the runtime rewrites its discovery record, and that
        # write is the entire trace the edge leaves anywhere. No bridge holds this
        # session and nothing is watching it.
        parked_path = await asyncio.to_thread(_publish_record, root, session_id, pending="approval")
        assert len(list(parked_path.parent.glob("*.json"))) == 1, "a second record exists"
        parked = await _next_frame(lines, lambda f: f["type"] == "session_status")
        assert parked["session_id"] == session_id
        assert parked["epoch"]
        assert parked["payload"] == {"code": "approval", "label": "Approval needed", "revision": 1}

        # ...and the answer reaches the row, which is the half the operator
        # reported as 5-10 s late.
        await asyncio.to_thread(_publish_record, root, session_id)
        resumed = await _next_frame(lines, lambda f: f["type"] == "session_status")
        assert resumed["session_id"] == session_id
        assert resumed["payload"]["code"] != parked["payload"]["code"]
        assert resumed["payload"]["revision"] > parked["payload"]["revision"]

    listed = await client.get("/v1/desktop/sessions", params={"limit": 50})
    assert listed.status_code == 200, listed.text
    rows = listed.json()["result"]["sessions"]
    row = next((entry for entry in rows if entry["id"] == session_id), None)
    assert row is not None, rows
    assert row["status_epoch"] == resumed["epoch"]
    assert row["status_revision"] == resumed["payload"]["revision"]
    # The list derives the pair itself (``load_catalog``), so agreeing here is
    # the parity claim over HTTP rather than inside one process's caches.
    assert (row["status"]["code"], row["status"]["label"]) == (
        resumed["payload"]["code"],
        resumed["payload"]["label"],
    )


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
        # R6: the machine-wide file is NO LONGER WRITTEN. Its absence is half the
        # fix — while every publisher wrote that one path, the last writer decided
        # the machine's answer and the first process to exit revoked a live
        # sibling's lease.
        assert not delivery_path(root).exists()
        assert _published_record(root)["pid"] == os.getpid()
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
        payload = _published_record(root)
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
    against. Both directions are asserted: a running desktop IS the destination
    while the launch is allowed, and is NOT one once it is refused (review round
    2, R13) — the second half is the reason this test can no longer inherit the
    suite-wide refusal, because the hole it used to travel through is closed and
    an assertion that leans on it would be asserting the defect.
    """
    from local_operator.session.runtime.viewer_server import ViewerServer
    from local_operator.session.runtime.viewers import DESKTOP_SURFACE
    from local_operator.tui import resume_click
    from local_operator.tui.resume_click import open_session

    # The visible opt-out, the way the TUI's own ladder tests take theirs: this
    # test exercises rung 1, which only EXISTS while the launch is allowed.
    monkeypatch.delenv(resume_click.DESKTOP_LAUNCH_REFUSED_ENV, raising=False)

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

        # AND THE REFUSAL TAKES THE APP OUT OF THE LADDER (review round 2,
        # R13). Measured against the SAME real endpoint rather than over
        # synthetic records, because this server would answer the click if
        # anything asked it: with the refusal set the click must fall through
        # to the (doubled) spawn and this host must never see a resume.
        monkeypatch.setenv(resume_click.DESKTOP_LAUNCH_REFUSED_ENV, "1")
        switched.clear()
        launched.clear()
        assert open_session(target) is True
        assert switched == [], "a refused desktop viewer took the click"
        assert launched == [target]
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


@pytest.mark.asyncio
async def test_the_window_state_decides_the_banner_over_real_http(desktop_server):
    """QA round 1's presence matrix, re-run at the layer it was measured on.

    QA reported ``presence-unfocused-same`` and ``presence-hidden-same`` as FAIL:
    the app's window was showing the SAME conversation the completion was about,
    but was no longer focused (or no longer visible), and no banner was raised
    where one is owed. The rule was right and the READ was stale — the decision
    came off the 2 s presence cache while the probe flipped the window state
    inside that window, so the answer described focus the user had already given
    up.

    That reads as a small timing artifact and is not: this decision is TERMINAL.
    A suppressed completion is never re-decided, and the runtime's own rung 4
    defers whenever a desktop is reachable — so for that turn no surface raised
    anything at all. Hence the uncached read, and hence this test, which is the
    whole matrix through the real route, the real feed and a real completion
    written by another process.

    The ONE suppressing state is a focused window showing THIS conversation.
    ``reset_cache`` is deliberately NOT called between cells, because the stale
    answer is what is under test.
    """
    root, client = desktop_server
    displayed = await _create(client, root, str(uuid.uuid4()))
    other = await _create(client, root, str(uuid.uuid4()))
    hidden = {"exists": True, "focused": False, "visible": False, "minimized": False}
    cells = [
        # name, window, completion for the DISPLAYED session?, is a banner owed?
        ("focused-same", {"focused": True, "visible": True}, True, False),
        ("focused-other", {"focused": True, "visible": True}, False, True),
        ("unfocused-same", {"focused": False, "visible": True}, True, True),
        ("unfocused-other", {"focused": False, "visible": True}, False, True),
        ("hidden-same", {}, True, True),
        ("hidden-other", {}, False, True),
        ("no-window-same", {"exists": False}, True, True),
        ("no-window-other", {"exists": False}, False, True),
    ]

    async with client.stream("GET", "/v1/desktop/events") as response:
        lines = response.aiter_lines()
        opened = await _next_frame(lines, lambda f: f["type"] == "open")
        subscription = opened["payload"]["subscription_id"]
        # A READER TASK rather than a bounded read per cell. Cancelling an
        # `aiter_lines()` iteration closes the response it is reading, which on
        # the suppressing cell would tear the subscription down and turn the
        # NEXT cell's presence post into a 404 — a harness failure that reads
        # exactly like the product bug under test.
        streamed: list[dict[str, Any]] = []

        async def pump() -> None:
            async for line in lines:
                if line.startswith("data: "):
                    streamed.append(json.loads(line[6:]))

        reader = asyncio.create_task(pump())

        def announced_for(token: str) -> list[dict[str, Any]]:
            return [
                frame
                for frame in streamed
                if frame["type"] == "notification"
                and frame["payload"].get("completion_token") == token
            ]

        async def settle() -> None:
            # ONE SECOND, and the number is the test. The feed polls at 100 ms, so
            # this is ten poll intervals — a banner that is coming has arrived,
            # asserted by the check below rather than assumed. It is ALSO well
            # inside ``PRESENCE_CACHE_TTL_S`` (2 s), which is what reproduces
            # QA's probe: it changed the window state and read the answer within
            # that window, which is the whole reason two of its rows failed. A
            # longer wait here would expire the cache, the stale read would
            # repair itself, and this test would pass against the unfixed code.
            await asyncio.sleep(1.0)

        try:
            for name, window, same, expected in cells:
                streamed.clear()
                beat = await client.post(
                    "/v1/desktop/presence",
                    json={
                        "subscription_id": subscription,
                        "can_notify": True,
                        "can_notify_kinds": ["complete", "error"],
                        "session_id": displayed,
                        "window": {**hidden, **window},
                    },
                )
                assert beat.status_code == 200, f"{name}: {beat.text}"
                target = displayed if same else other
                token = await asyncio.to_thread(_publish, root, target)
                await settle()
                # The attention frame for this completion is published in the
                # same tick as the banner and BEFORE it, so its arrival is what
                # says the tick ran at all.
                assert any(
                    frame["type"] == "attention" and frame.get("session_id") == target
                    for frame in streamed
                ), f"{name}: the tick never ran"
                announced = announced_for(token)
                assert bool(announced) is expected, f"{name}: {streamed}"
                if expected:
                    assert announced[0]["session_id"] == target
                    assert announced[0]["payload"]["focus_policy"] == "always"
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader


@pytest.mark.asyncio
async def test_a_completion_inside_active_reorders_the_list_and_is_announced(
    desktop_server, workspace: Path
):
    """THE REPORTED SYMPTOM, end to end: a session that is ALREADY Active finishes.

    ``CatalogEntry.active`` is section membership; the order the sidebar renders is
    ``CatalogEntry.rank``. A working row (tier 4) that finishes becomes an unread
    completion (tier 1) with ``active`` True -> True, so the feed's section
    comparison published nothing at all — measured on the real backend as ZERO
    catalogue frames in 15 s, and still zero across ~100 ticks with the probes
    accelerated 20x, while the backend's own list read had already moved the row —
    and the client kept the last list's order until some OTHER row's section move
    refetched it (5-10 s on this machine, which is what the operator reported).

    Everything here is the production stack over loopback HTTP: the real app, the
    real SSE stream, the real discovery-record write, a real completion in the
    attention store. The claim is two-sided, and the second half is what makes the
    first worth anything: the frame arrives on the doorbell's own clock, AND the
    list read it triggers leads with the completed row.
    """
    root, client = desktop_server
    elder = await _create(client, workspace, "55555555-5555-4555-8555-555555555501")
    completer = await _create(client, workspace, "55555555-5555-4555-8555-555555555502")

    # PIN THE BIRTHS. The ORDER KEY's third term is the session's canonical birth,
    # read from ``created_at.json`` by ``session_created_at``, and two POSTs land
    # inside the same second — without this the ordering assertion would be about
    # the id tiebreak rather than about the resort under test.
    for session_id, at in ((elder, 1_700_000_000.0), (completer, 1_700_000_600.0)):
        (root / "sessions" / session_id / CREATED_AT_NAME).write_text(
            json.dumps(at), encoding="utf-8"
        )

    async with client.stream("GET", "/v1/desktop/events") as response:
        assert response.status_code == 200, response.read()
        lines = response.aiter_lines()
        await _next_frame(lines, lambda f: f["type"] == "open")

        # The machine state the operator described: an older unread completion
        # (tier 1, Active) and a session WORKING (tier 4, Active), so both are in
        # "Active chats" with the working one second.
        await asyncio.to_thread(_publish, root, elder)
        await asyncio.to_thread(_publish_record, root, completer, busy=True)
        # The working edge is DRAINED before the completion is driven: otherwise
        # the completion would be that row's FIRST edge, which invalidates for a
        # different reason and would leave this test proving nothing.
        await _next_frame(
            lines,
            lambda f: f["type"] == "session_status"
            and f["session_id"] == completer
            and f["payload"]["code"] == "busy",
        )
        # The working edge is ALSO a section move — the row is cold until a runtime
        # reports it — so it publishes an invalidation of its own. Drained here, so
        # the frame asserted below cannot be this one.
        working_edge = await _next_frame(lines, lambda f: f["type"] == "catalogue")
        assert working_edge["payload"]["revision"] >= 1

        before = (await client.get("/v1/desktop/sessions", params={"limit": 50})).json()["result"]
        before_ids = [row["id"] for row in before["sessions"]]
        assert before_ids.index(elder) < before_ids.index(completer), before_ids
        assert all(row["active"] for row in before["sessions"]), before["sessions"]

        # IT FINISHES: the record goes quiet and the completion lands. The section
        # stays "Active chats"; only the row's position inside it changes.
        await asyncio.to_thread(_publish_record, root, completer)
        await asyncio.to_thread(_publish, root, completer, "complete")
        # The 2 s is a hang backstop, not the assertion: the frame itself is what
        # is waited on, and the doorbell's clock is 100 ms.
        frames = await _frames_until(lines, lambda f: f["type"] == "catalogue", timeout=2.0)
        catalogues = [frame for frame in frames if frame["type"] == "catalogue"]
        assert len(catalogues) == 1, frames
        kinds = [frame["type"] for frame in frames]
        # AFTER the status frame, so the client paints the checkmark and then
        # re-reads a list that already agrees with it.
        assert kinds.index("catalogue") > kinds.index("session_status"), kinds

    # ...and the read that frame triggers is the one that leads with the completed
    # row. Both rows are still Active: this is the resort, not a section move.
    listed = await client.get("/v1/desktop/sessions", params={"limit": 50})
    assert listed.status_code == 200, listed.text
    after = listed.json()["result"]["sessions"]
    assert after[0]["id"] == completer, after
    assert (after[0]["active"], after[1]["active"]) == (True, True), after
    assert (after[0]["status"]["code"], after[0]["status"]["label"]) == (
        "complete",
        "Unseen completion",
    ), after[0]


@pytest.mark.asyncio
async def test_a_pin_written_over_the_route_rings_the_catalogue_doorbell(
    desktop_server, workspace: Path
):
    """THE CROSS-SURFACE DOORBELL, over the real stack.

    A pin made in the TUI has to reach the desktop app with no manual refresh, and
    the only vehicle for that is the ``catalogue`` frame this feed publishes — the
    sidebar re-runs its catalogue fetch once per frame. So the pin file is in the
    probe's invalidation token, and this is the test that says the token actually
    moves: a unit test of the key string cannot, because the failure it guards
    against (a probe that never notices, leaving the app on its 30 s safety poll)
    lives entirely in the timing between a write and a frame.

    Three claims, each needing the HTTP layer:

    1. a pin WRITE publishes exactly one catalogue frame, within the probe interval;
    2. a QUIET window publishes none — otherwise the feed would refetch every
       sidebar on the machine once a second for nothing;
    3. a client connecting AFTER the write is not REPLAYED it, because the frame is
       an invalidation and the newcomer's ``open`` snapshot already carries the
       counter it would have announced.
    """
    root, client = desktop_server
    session_id = await _create(client, workspace, "44444444-4444-4444-8444-444444444401")

    async with client.stream("GET", "/v1/desktop/events") as response:
        assert response.status_code == 200, response.read()
        lines = response.aiter_lines()
        opened = await _next_frame(lines, lambda f: f["type"] == "open")
        assert opened["payload"]["catalogue_revision"] is not None

        # A READER TASK, not a bounded read per window: cancelling an
        # `aiter_lines()` iteration closes the response it is reading, so a
        # windowed read tears the subscription down and every later window reads
        # nothing at all — a harness failure that reads exactly like the product
        # bug under test. The same idiom the presence matrix below uses.
        streamed: list[dict[str, Any]] = []

        async def pump() -> None:
            async for line in lines:
                if line.startswith("data: "):
                    streamed.append(json.loads(line[6:]))

        reader = asyncio.create_task(pump())

        def catalogues(since: int) -> list[dict[str, Any]]:
            return [frame for frame in streamed[since:] if frame["type"] == "catalogue"]

        try:
            # The feed's own first probe publishes one catalogue frame as the
            # token is established, so it is DRAINED rather than asserted away:
            # the windows below have to be attributable to the pin.
            await asyncio.sleep(2.5)

            # (2) A quiet window publishes NOTHING catalogue-shaped.
            quiet_from = len(streamed)
            await asyncio.sleep(2.5)
            assert catalogues(quiet_from) == [], streamed[quiet_from:]

            # (1) The write, and the frame it owes within the probe interval plus
            # slack (a little over 2 x CATALOGUE_PROBE_INTERVAL_S).
            pinned = await client.post(
                f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True}
            )
            assert pinned.status_code == 200, pinned.text
            assert (root / "sidebar-pins.json").read_text() == json.dumps([session_id])

            wrote_from = len(streamed)
            await asyncio.sleep(2.5)
            published = catalogues(wrote_from)
            assert len(published) == 1, streamed[wrote_from:]
            revision = published[0]["payload"]["revision"]

            # ...and nothing follows it while the pin file sits still.
            after_from = len(streamed)
            await asyncio.sleep(2.5)
            assert catalogues(after_from) == [], streamed[after_from:]
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader

    # (3) A LATER subscriber is not replayed the invalidation: its own `open`
    # snapshot is the answer, and a replayed frame would make every reconnecting
    # client refetch a catalogue nothing has changed.
    async with client.stream("GET", "/v1/desktop/events") as response:
        assert response.status_code == 200, response.read()
        lines = response.aiter_lines()
        streamed_later: list[dict[str, Any]] = []

        async def pump_later() -> None:
            async for line in lines:
                if line.startswith("data: "):
                    streamed_later.append(json.loads(line[6:]))

        later_reader = asyncio.create_task(pump_later())
        try:
            # Read the open frame out of the PUMP's list rather than off `lines`:
            # two concurrent iterations of one async generator is a RuntimeError,
            # and the `open` snapshot is the only frame this leg wants to see.
            for _ in range(200):
                if streamed_later:
                    break
                await asyncio.sleep(0.05)
            assert streamed_later and streamed_later[0]["type"] == "open", streamed_later
            await asyncio.sleep(2.5)
        finally:
            later_reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await later_reader

    assert streamed_later[0]["payload"]["catalogue_revision"] == revision
    assert [frame for frame in streamed_later if frame["type"] == "catalogue"] == []

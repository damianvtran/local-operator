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
import math
import os
import secrets
import socket
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio
import uvicorn

from local_operator.agents import AgentRegistry
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
from local_operator.tools.agent_tool import AgentParams, write_profile
from tests.notification_opt_in import notification_path_opt_in

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def notification_path_on(headless_tui_env: Path) -> Iterator[None]:
    """This suite's subject IS the ``notification`` frame, so it opts in.

    BOTH gates suppress it otherwise, and here the SECOND one is load-bearing:
    the fixture creates its sessions on the test hosting (the deterministic
    wire), so the feed declines them as test sessions and the frame the
    assertions look for never appears. That is what CI caught on the merge — the
    kill-switch opt-in this suite never had is only half of it.

    It TAKES ``headless_tui_env`` because pytest builds an autouse fixture
    before a same-scope one that does not depend on it, so a bare fixture would
    clear the switch and be re-armed by the e2e conftest's fixture a moment
    later. See ``tests/notification_opt_in`` for what the escape can and cannot
    do.
    """
    with notification_path_opt_in():
        yield


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


async def _await_frames(
    streamed: list[dict[str, Any]],
    frame_type: str,
    *,
    since: int,
    what: str,
    guard: float = 5.0,
) -> list[dict[str, Any]]:
    """Wait until at least one ``frame_type`` frame lands after ``since``.

    WHY A WAIT AND NOT A WINDOW (flake sweep, 2026-10-17). The doorbell cells
    used to sample ``len(streamed)`` AFTER the write's acknowledgement, sleep a
    fixed window, and require the frame to fall inside it. Nothing enforces
    that ordering: the feed's probe publishes the frame the instant the write
    is visible on disk, on its own 1 s clock (``CATALOGUE_PROBE_INTERVAL_S``),
    so a delivery that beats the post-ack sample sits in ``streamed[:since]``
    and the window reads ``[]``. That is exactly how
    ``test_an_authored_profile_and_team_ring_the_authoring_doorbell`` failed
    twice in one evening (runs 39062450274 / 39075893260 — ``AssertionError:
    []`` at the team leg, green on re-execution). Locally instrumented, delivery
    follows a write by 0.0-0.4 s (the probe's 1 s clock sets the phase), so the
    ack path and the delivery chain are the same order of magnitude — the
    ordering is a genuine race, not a theoretical one. Callers therefore take
    ``since`` BEFORE the write and wait on the fact their assertion reads; the
    guard (5 s, five probe intervals) is a hang backstop, never evidence — a
    silent bell still fails loudly here, naming what it waited for and the
    frames it did see.
    """
    deadline = time.monotonic() + guard
    while True:
        frames = [frame for frame in streamed[since:] if frame["type"] == frame_type]
        if frames:
            return frames
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{what}: no {frame_type!r} frame within {guard:g}s "
                f"(the guard fired); frames since the baseline: {streamed[since:]!r}"
            )
        await asyncio.sleep(0.05)


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


def _builtin_session_ids(root: Path) -> list[str]:
    """The assistant sessions the boot hook may create on this root.

    Aida and the Librarian are USER sessions by design (R3/R7), so a fresh
    config lists their rows and auto-pins beside the test's own. The feed and
    pin tests below assert exact sets; filter both built-ins by their state
    files rather than hard-coding only Aida and failing when her twin lands.
    """
    ids: list[str] = []
    for name in ("aida", "librarian"):
        path = root / name / "state.json"
        if not path.exists():
            continue
        session_id = json.loads(path.read_text()).get("session_id")
        if session_id:
            ids.append(session_id)
    return ids


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
        # BOTH waits are SCOPED to this session, matching the sibling completion
        # test's predicate: the feed is machine-wide, and the app's boot hook
        # ensures Aida's session on every server start, so her row's first edges
        # ride this stream too. Under CI load one of them landed between the two
        # publishes and the unscoped wait returned HER frame (tui-e2e ubuntu,
        # run 36466096472: '4c85c86834bc' != the created id).
        parked = await _next_frame(
            lines,
            lambda f: f["type"] == "session_status" and f["session_id"] == session_id,
        )
        assert parked["session_id"] == session_id
        assert parked["epoch"]
        assert parked["payload"] == {"code": "approval", "label": "Approval needed", "revision": 1}

        # ...and the answer reaches the row, which is the half the operator
        # reported as 5-10 s late.
        await asyncio.to_thread(_publish_record, root, session_id)
        resumed = await _next_frame(
            lines,
            lambda f: f["type"] == "session_status" and f["session_id"] == session_id,
        )
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


class _PresenceCacheClock:
    """Logical time for ONLY this root's native presence-cache lookup.

    Rebinding this module's time namespace must not mutate Python's shared
    time module, asyncio's clock, or the presence lease's wall clock. The
    native lookup does not pass its root to monotonic(), so inspect ONLY that
    exact caller's root; every other caller/root keeps the real clock.
    """

    def __init__(self, root: Path, native: Any) -> None:
        self.root = root
        self.native_code = native.__code__
        # A WHOLE number of seconds, never the raw ``time.monotonic()``. The
        # matrix measures a cache entry's age as ``now - ((now + TTL) - TTL)``,
        # which is exactly 0 only if every addition is exact. With a fractional
        # uptime the round trip loses a ulp and the age comes back as -1.4e-14
        # (reproduced at uptime 252.56993219293187), failing ``0 <= age`` on a
        # healthy cache. An integer base plus the TTL, +1 and TTL/16 steps used below
        # are all dyadic, so the arithmetic stays exact at any magnitude.
        # ``ceil`` (not ``int``) keeps the base at or above the real clock, so
        # a cache entry stamped by the real clock before this patch still
        # expires after the first logical advance.
        self.now = float(math.ceil(time.monotonic()))

    def monotonic(self) -> float:
        frame = sys._getframe(1)
        if frame.f_code is self.native_code:
            from local_operator.session.runtime.presence import config_dir

            if (frame.f_locals.get("root") or config_dir()) == self.root:
                return self.now
        return time.monotonic()

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class _BannerProcessingWitness:
    """Observe real decisions and drain the persistent reader through their cursor."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from local_operator.server.utils.desktop_feed import DesktopFeed

        self.decisions: dict[str, asyncio.Event] = {}
        self.through: dict[str, tuple[str, int]] = {}
        self.frames: list[dict[str, Any]] = []
        self.changed = asyncio.Event()
        self.phase: str | None = None
        self.observations: dict[str, list[dict[str, Any]]] = {}
        self.tasks: dict[str, asyncio.Task[Any] | None] = {}
        original = DesktopFeed._emit_notifications

        async def observe(feed, fresh, states):
            # Other roots retain the original early-return/argument contract;
            # diagnostic token reads must not precede the producer's own gates.
            if feed.root != root:
                await original(feed, fresh, states)
                return
            # Capture call identity before the real await; diagnostics must not
            # mistake a post-return value for the batch actually delegated.
            before_root = feed.root
            before_tokens = tuple(row["token"] for row in fresh)
            observed: list[dict[str, Any]] = []
            if before_root == root:
                for token in before_tokens:
                    record: dict[str, Any] = {
                        "stage": "entered",
                        "root_match_before": True,
                        "task_is_feed_poller_at_entry": asyncio.current_task() is feed._task,
                    }
                    self.observations.setdefault(token, []).append(record)
                    self.tasks[token] = asyncio.current_task()
                    observed.append(record)
            try:
                await original(feed, fresh, states)
            except BaseException as error:
                for record in observed:
                    record.update(stage="raised", error_type=type(error).__name__)
                raise
            else:
                for record in observed:
                    record.update(
                        stage="returned",
                        root_match_after=feed.root == root,
                        tokens_unchanged=before_tokens == tuple(row["token"] for row in fresh),
                    )
            if feed.root == root:
                # Attention was queued BEFORE this awaited operation. Only its
                # return means a suppressing verdict is final, and draining the
                # actual SSE reader through this cursor proves no banner hid
                # behind an earlier attention frame in the client's queue.
                for row in fresh:
                    token = row["token"]
                    self.through[token] = (feed.epoch, feed.sequence)
                    self.decisions.setdefault(token, asyncio.Event()).set()

        monkeypatch.setattr(DesktopFeed, "_emit_notifications", observe)

    def received(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)
        self.changed.set()

    async def wait(self, token: str) -> None:
        # Reuse the file's existing bounded frame-wait budget; no elapsed-time
        # retry or longer sleep can make an expired cache appear discriminating.
        stage = "decision"
        try:
            async with asyncio.timeout(5):
                await self.decisions.setdefault(token, asyncio.Event()).wait()
                stage = "stream_cursor"
                epoch, sequence = self.through[token]
                while not any(
                    frame.get("epoch") == epoch and frame.get("seq", -1) >= sequence
                    for frame in self.frames
                ):
                    self.changed.clear()
                    await self.changed.wait()
        except TimeoutError as error:
            # Snapshot only THIS token/task at the failed await, not a thread
            # census or a claim that no upstream work ran. Never signal success
            # from an error/cancellation path. Preserve the original guard.
            task = self.tasks.get(token)
            error.add_note(
                json.dumps(
                    {
                        "banner_wait_stage": stage,
                        "matrix_cell": self.phase,
                        "cursor_recorded_at_note": token in self.through,
                        "received_frame_count_at_note": len(self.frames),
                        "token_observations": self.observations.get(token, []),
                        "decision_is_set_at_note": self.decisions[token].is_set(),
                        "observed_task_present": task is not None,
                        "observed_task_done_at_note": task.done() if task is not None else None,
                        "observed_task_cancelled_at_note": (
                            task.cancelled() if task is not None else None
                        ),
                    },
                    sort_keys=True,
                )
            )
            raise


# These controls exercise the real witness with a SYNTHETIC emitter boundary.
# They do not construct a feed/server or prove the production hosting path.
# The synthetic carrier is cast at the call boundary: it presents exactly the
# attributes `_emit_notifications` and the witness touch. Casting keeps the real
# signature honest rather than widening it or adding a blanket ignore.
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["return", "raise"])
async def test_banner_witness_records_owned_emitter_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, request: pytest.FixtureRequest
) -> None:
    from types import SimpleNamespace

    from local_operator.server.utils.desktop_feed import DesktopFeed

    root = tmp_path / "owned"
    feed = SimpleNamespace(
        root=root, epoch="helper-epoch", sequence=7, _task=asyncio.current_task()
    )
    rows = [{"token": str(uuid.uuid4())}]
    states = {}
    calls = []
    failure = ValueError("synthetic emitter failure")
    error_identity: bool | None = None
    cursor_wait_completed = False

    async def emitter(actual_feed, actual_rows, actual_states):
        calls.append((actual_feed, actual_rows, actual_states))
        if outcome == "raise":
            raise failure

    monkeypatch.setattr(DesktopFeed, "_emit_notifications", emitter)
    witness = _BannerProcessingWitness(root, monkeypatch)
    if outcome == "raise":
        with pytest.raises(ValueError) as raised:
            await DesktopFeed._emit_notifications(cast("DesktopFeed", feed), rows, states)
        assert raised.value is failure
        error_identity = raised.value is failure
        assert witness.through == {}
        assert witness.decisions == {}
        assert witness.observations[rows[0]["token"]][0]["stage"] == "raised"
        assert witness.observations[rows[0]["token"]][0]["error_type"] == "ValueError"
    else:
        await DesktopFeed._emit_notifications(cast("DesktopFeed", feed), rows, states)
        assert witness.through == {rows[0]["token"]: (feed.epoch, feed.sequence)}
        assert witness.decisions[rows[0]["token"]].is_set()
        assert witness.observations[rows[0]["token"]][0] == {
            "stage": "returned",
            "root_match_before": True,
            "task_is_feed_poller_at_entry": True,
            "root_match_after": True,
            "tokens_unchanged": True,
        }
        # Exercise the actual cursor wait, without substituting its algorithm.
        witness.received({"epoch": feed.epoch, "seq": feed.sequence})
        await witness.wait(rows[0]["token"])
        cursor_wait_completed = True
    assert len(calls) == 1
    assert calls[0][0] is feed and calls[0][1] is rows and calls[0][2] is states
    assert witness.tasks[rows[0]["token"]] is asyncio.current_task()
    # Export assertion-backed helper facts: successful stdout is not JUnit evidence.
    record = witness.observations[rows[0]["token"]][0]
    request.node.user_properties.append(
        (
            "banner_witness_helper",
            json.dumps(
                {
                    "boundary": "synthetic_emitter",
                    "outcome": outcome,
                    "calls": len(calls),
                    "delegation_identity": calls[0][0] is feed
                    and calls[0][1] is rows
                    and calls[0][2] is states,
                    "stage": record["stage"],
                    "error_class": record.get("error_type"),
                    "error_identity": error_identity,
                    "cursor_count": len(witness.through),
                    "decision_count": len(witness.decisions),
                    "cursor_wait_completed": cursor_wait_completed,
                    "entry_task_is_current": witness.tasks[rows[0]["token"]]
                    is asyncio.current_task(),
                },
                sort_keys=True,
            ),
        )
    )


@pytest.mark.asyncio
async def test_banner_witness_delegates_foreign_rows_without_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    from types import SimpleNamespace

    from local_operator.server.utils.desktop_feed import DesktopFeed

    feed = SimpleNamespace(root=tmp_path / "foreign")
    rows = [{}]  # Legal for an emitter which returns before inspecting its rows.
    states = {}
    calls = []

    async def emitter(actual_feed, actual_rows, actual_states):
        calls.append((actual_feed, actual_rows, actual_states))

    monkeypatch.setattr(DesktopFeed, "_emit_notifications", emitter)
    witness = _BannerProcessingWitness(tmp_path / "owned", monkeypatch)
    await DesktopFeed._emit_notifications(cast("DesktopFeed", feed), rows, states)
    assert len(calls) == 1
    assert calls[0][0] is feed and calls[0][1] is rows and calls[0][2] is states
    assert witness.observations == {} and witness.tasks == {}
    assert witness.through == {} and witness.decisions == {}
    assert rows == [{}]
    request.node.user_properties.append(
        (
            "banner_witness_helper",
            json.dumps(
                {
                    "boundary": "synthetic_emitter",
                    "outcome": "foreign_passthrough",
                    "calls": len(calls),
                    "delegation_identity": calls[0][0] is feed
                    and calls[0][1] is rows
                    and calls[0][2] is states,
                    "rows_unchanged": rows == [{}],
                    "owned_collection_counts": {
                        "observations": len(witness.observations),
                        "tasks": len(witness.tasks),
                        "cursors": len(witness.through),
                        "decisions": len(witness.decisions),
                    },
                },
                sort_keys=True,
            ),
        )
    )


@pytest.mark.asyncio
async def test_banner_witness_timeout_note_precedes_successful_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    from types import SimpleNamespace

    from local_operator.server.utils.desktop_feed import DesktopFeed

    root = tmp_path / "owned"
    feed = SimpleNamespace(root=root, epoch="helper-epoch", sequence=7, _task=None)
    rows = [{"token": str(uuid.uuid4())}]
    states = {}
    entered = asyncio.Event()
    release = asyncio.Event()
    body_finally = asyncio.Event()
    calls = []

    async def emitter(actual_feed, actual_rows, actual_states):
        calls.append((actual_feed, actual_rows, actual_states))
        try:
            entered.set()
            await release.wait()
        finally:
            body_finally.set()

    monkeypatch.setattr(DesktopFeed, "_emit_notifications", emitter)
    witness = _BannerProcessingWitness(root, monkeypatch)
    witness.phase = "synthetic-held-emitter"
    invocation = asyncio.create_task(
        DesktopFeed._emit_notifications(cast("DesktopFeed", feed), rows, states)
    )
    feed._task = invocation
    try:
        async with asyncio.timeout(5):
            await entered.wait()
        assert witness.observations[rows[0]["token"]][0]["stage"] == "entered"
        # Do not patch the guard or clock: this exercises the original 5s wait.
        with pytest.raises(TimeoutError) as timed_out:
            await witness.wait(rows[0]["token"])
        captured_note = timed_out.value.__notes__[-1]
        note = json.loads(captured_note)
        assert note == {
            "banner_wait_stage": "decision",
            "matrix_cell": "synthetic-held-emitter",
            "cursor_recorded_at_note": False,
            "received_frame_count_at_note": 0,
            "token_observations": [
                {
                    "stage": "entered",
                    "root_match_before": True,
                    "task_is_feed_poller_at_entry": True,
                }
            ],
            "decision_is_set_at_note": False,
            "observed_task_present": True,
            "observed_task_done_at_note": False,
            "observed_task_cancelled_at_note": False,
        }
        assert witness.through == {} and not witness.decisions[rows[0]["token"]].is_set()
        assert not invocation.done() and not body_finally.is_set()
    finally:
        release.set()
        # Own this short async invocation, not the production long-lived poller.
        # Settle it before monkeypatch undo; no worker/thread reaping assertion.
        await asyncio.shield(invocation)
    assert invocation.done() and not invocation.cancelled()
    assert body_finally.is_set() and invocation.result() is None
    assert len(calls) == 1
    assert calls[0][0] is feed and calls[0][1] is rows and calls[0][2] is states
    assert witness.observations[rows[0]["token"]][0]["stage"] == "returned"
    assert witness.through == {rows[0]["token"]: (feed.epoch, feed.sequence)}
    assert witness.decisions[rows[0]["token"]].is_set()
    witness.received({"epoch": feed.epoch, "seq": feed.sequence})
    await witness.wait(rows[0]["token"])
    # Retain the original timeout string separately from settled async-task facts.
    # This is the fake invocation's body, not an OS worker/thread cleanup receipt.
    assert json.loads(captured_note) == note
    request.node.user_properties.append(
        (
            "banner_witness_helper",
            json.dumps(
                {
                    "boundary": "synthetic_emitter",
                    "outcome": "held_timeout_then_return",
                    "calls": len(calls),
                    "delegation_identity": calls[0][0] is feed
                    and calls[0][1] is rows
                    and calls[0][2] is states,
                    "captured_timeout_note": captured_note,
                    "captured_note_unchanged_after_settlement": json.loads(captured_note) == note,
                    "post_settlement": {
                        "same_invocation_done": invocation.done(),
                        "same_invocation_cancelled": invocation.cancelled(),
                        "same_invocation_result_is_none": invocation.result() is None,
                        "fake_body_finally_seen": body_finally.is_set(),
                        "observed_stage": witness.observations[rows[0]["token"]][0]["stage"],
                        "cursor": list(witness.through[rows[0]["token"]]),
                        "decision_set": witness.decisions[rows[0]["token"]].is_set(),
                        "cursor_wait_completed": True,
                    },
                },
                sort_keys=True,
            ),
        )
    )


def test_the_presence_cache_clock_is_not_the_server_or_another_roots_clock(tmp_path, monkeypatch):
    from local_operator.session.runtime import presence as presence_module

    native = presence_module.desktop_presence
    original_monotonic = time.monotonic
    clock = _PresenceCacheClock(tmp_path, native)
    clock.now += 100
    monkeypatch.setattr(presence_module, "time", clock)
    native(tmp_path, cached=True)
    assert (
        presence_module._CACHE[str(tmp_path)][0] == clock.now + presence_module.PRESENCE_CACHE_TTL_S
    )
    other = tmp_path / "other-root"
    before = time.monotonic()
    native(other, cached=True)
    after = time.monotonic()
    expiry = presence_module._CACHE[str(other)][0]
    assert before <= expiry - presence_module.PRESENCE_CACHE_TTL_S <= after
    assert time.monotonic is original_monotonic
    assert clock.time is time.time


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["held-poll", "held-decision", "native-stale-cache"])
async def test_banner_witness_waits_for_processing_and_keeps_stale_cache_sensitive(
    desktop_server, monkeypatch, mode
):
    from local_operator.server.utils import desktop_feed as feed_module
    from local_operator.session.runtime import presence as presence_module

    root, _ = desktop_server
    entered = asyncio.Event()
    release = asyncio.Event()
    waiting = asyncio.Event()
    attention = asyncio.Event()
    state = {"beats": 0, "waits": 0, "returned": False}
    tick = feed_module.DesktopFeed._tick
    notifications = feed_module.DesktopFeed._emit_notifications
    post = httpx.AsyncClient.post
    wait = _BannerProcessingWitness.wait
    received = _BannerProcessingWitness.received
    native_presence = presence_module.desktop_presence

    async def gated_tick(feed):
        if mode == "held-poll" and state["beats"] == 2 and not release.is_set():
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
        return await tick(feed)

    async def gated_notifications(feed, fresh, states):
        if mode == "held-decision" and fresh and not release.is_set():
            state["token"] = fresh[0]["token"]
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
        return await notifications(feed, fresh, states)

    async def observed_post(client, url, *args, **kwargs):
        response = await post(client, url, *args, **kwargs)
        if str(url).endswith("/presence"):
            state["beats"] += 1
            if mode == "held-poll" and state["beats"] == 2:
                await asyncio.wait_for(entered.wait(), 5)
        return response

    async def observed_wait(witness, token):
        state["waits"] += 1
        wanted = 2 if mode == "held-poll" else 1
        if state["waits"] == wanted:
            state["witness"] = witness
            state["token"] = token
            waiting.set()
            await wait(witness, token)
            state["returned"] = True
            return
        return await wait(witness, token)

    def observed_frame(witness, frame):
        received(witness, frame)
        if frame["type"] == "attention" and frame["payload"].get("completion_token") == state.get(
            "token"
        ):
            attention.set()

    async def controller():
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(waiting.wait(), 5)
        witness = state["witness"]
        token = state["token"]
        if mode == "held-decision":
            # Attention was actually delivered, yet the suppressing decision
            # is still pending. An attention-only witness would falsely pass.
            if not any(
                frame["type"] == "attention" and frame["payload"].get("completion_token") == token
                for frame in witness.frames
            ):
                await asyncio.wait_for(attention.wait(), 5)
        assert not state["returned"] and not witness.decisions[token].is_set()
        if mode == "held-poll":
            # Execute the ORIGINAL one-second readiness assertion once, on the
            # actual reader and durable publisher input. Its failure signals
            # release; this is not an arbitrary >1s sleep or a new deadline.
            store = AttentionStore(root / "attention.db")
            rows = store.published_since(0)
            row = next(row for row in rows if row["token"] == token)
            target = row["conversation"].split("/", 1)[1]
            legacy_checked = asyncio.get_running_loop().create_future()

            def original_assertion():
                try:
                    with pytest.raises(AssertionError, match="focused-other: the tick never ran"):
                        assert any(
                            frame["type"] == "attention" and frame.get("session_id") == target
                            for frame in witness.frames
                        ), "focused-other: the tick never ran"
                    legacy_checked.set_result(None)
                except BaseException as error:
                    legacy_checked.set_exception(error)

            timer = asyncio.get_running_loop().call_later(1.0, original_assertion)
            try:
                await asyncio.wait_for(legacy_checked, 5)
                assert not state["returned"] and not witness.decisions[token].is_set()
                print(f"Original one-second assertion rejected durable token {token}; release now")
            finally:
                timer.cancel()
        print(f"Banner control {mode}: real token {token} pending at witness")
        release.set()

    monkeypatch.setattr(feed_module.DesktopFeed, "_tick", gated_tick)
    monkeypatch.setattr(feed_module.DesktopFeed, "_emit_notifications", gated_notifications)
    monkeypatch.setattr(httpx.AsyncClient, "post", observed_post)
    monkeypatch.setattr(_BannerProcessingWitness, "wait", observed_wait)
    monkeypatch.setattr(_BannerProcessingWitness, "received", observed_frame)
    task = None
    if mode == "native-stale-cache":
        # Reintroduce ONLY the original cached-read defect. Native cache,
        # delivery, focus policy and notification emission all remain real.
        monkeypatch.setattr(
            feed_module,
            "desktop_presence",
            lambda root, *, cached=False: native_presence(root, cached=True),
        )
    else:
        task = asyncio.create_task(controller())
    try:
        if mode == "native-stale-cache":
            with pytest.raises(AssertionError, match="unfocused-same: banner verdict"):
                # Isolate the cache counterexample from focused-other's unrelated
                # positive composer. Its delivery remains covered by the full
                # healthy matrix and held-poll control; neither belongs in this
                # negative arm's prerequisites.
                await _assert_window_banner_matrix(
                    desktop_server,
                    monkeypatch,
                    cell_names=("focused-same", "unfocused-same"),
                )
            print("Native unexpired cached presence suppresses the owed banner: negative caught")
        else:
            # Controls exercise only their direct cells; the public test above
            # runs the unchanged full eight-window matrix once.
            await _assert_window_banner_matrix(
                desktop_server, monkeypatch, cell_limit=2 if mode == "held-poll" else 1
            )
            assert task is not None
            await task
            assert state["returned"]
    finally:
        release.set()
        if task is not None:
            task.cancel()
            results = await asyncio.gather(task, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    raise result


@pytest.mark.asyncio
async def test_the_window_state_decides_the_banner_over_real_http(desktop_server, monkeypatch):
    await _assert_window_banner_matrix(desktop_server, monkeypatch)


@pytest.mark.asyncio
async def test_the_stale_cache_counterexample_pair_is_healthy_without_mutation(
    desktop_server, monkeypatch
):
    await _assert_window_banner_matrix(
        desktop_server, monkeypatch, cell_names=("focused-same", "unfocused-same")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cell_names",
    [(), ("unknown",), ("focused-same", "focused-same"), ("unfocused-same", "focused-same")],
)
async def test_banner_cell_selection_cannot_pass_empty_or_reorder_the_control(
    tmp_path, monkeypatch, cell_names
):
    # Invalid setup must fail distinctly, never with the negative arm's matched
    # 'unfocused-same: banner verdict' assertion or an empty successful matrix.
    with pytest.raises(ValueError, match="invalid banner cell selection"):
        await _assert_window_banner_matrix((tmp_path, None), monkeypatch, cell_names=cell_names)


async def _assert_window_banner_matrix(
    desktop_server, monkeypatch, *, cell_limit=None, cell_names=None
):
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
    written on another worker thread/SQLite connection in this same process.

    The ONE suppressing state is a focused window showing THIS conversation.
    ``reset_cache`` is deliberately NOT called between cells, because the stale
    answer is what is under test.
    """
    from local_operator.session.runtime import presence as presence_module

    root, client = desktop_server
    native_presence = presence_module.desktop_presence
    clock = _PresenceCacheClock(root, native_presence)
    monkeypatch.setattr(presence_module, "time", clock)
    witness = _BannerProcessingWitness(root, monkeypatch)
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

    if cell_names is not None:
        if (
            cell_limit is not None
            or not isinstance(cell_names, tuple)
            or not cell_names
            or not all(isinstance(name, str) for name in cell_names)
        ):
            raise ValueError("invalid banner cell selection")
        selected = [cell for cell in cells if cell[0] in cell_names]
        if tuple(cell[0] for cell in selected) != cell_names:
            raise ValueError("invalid banner cell selection: unknown, duplicate or reordered names")
    else:
        if cell_limit is not None and (
            type(cell_limit) is not int or not 1 <= cell_limit <= len(cells)
        ):
            raise ValueError("invalid banner cell selection: limit out of range")
        selected = cells[:cell_limit]
    assert selected  # No healthy arm may succeed without executing a cell.
    displayed = await _create(client, root, str(uuid.uuid4()))
    other = await _create(client, root, str(uuid.uuid4()))

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
                    frame = json.loads(line[6:])
                    streamed.append(frame)
                    witness.received(frame)

        reader = asyncio.create_task(pump())

        def announced_for(token: str) -> list[dict[str, Any]]:
            return [
                frame
                for frame in streamed
                if frame["type"] == "notification"
                and frame["payload"].get("completion_token") == token
            ]

        try:
            for name, window, same, expected in selected:
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
                if name == "focused-same":
                    # ONE coherent native cache seed, never cleared/rewarmed as
                    # the window changes. Logical cache time, not scheduler wall
                    # time, keeps the original stale-focus bug observable.
                    # Expire a bootstrap no-window answer ONCE by ordinary
                    # cache time advancement, before the matrix's coherent seed.
                    clock.now += presence_module.PRESENCE_CACHE_TTL_S + 1
                    initial = native_presence(root, cached=True)
                    assert initial.attended and initial.session_id == displayed
                else:
                    clock.now += presence_module.PRESENCE_CACHE_TTL_S / 16
                expiry, _ = presence_module._CACHE[str(root)]
                target = displayed if same else other
                witness.phase = name
                token = await asyncio.to_thread(_publish, root, target)
                await witness.wait(token)
                age = clock.now - (expiry - presence_module.PRESENCE_CACHE_TTL_S)
                assert 0 <= age < presence_module.PRESENCE_CACHE_TTL_S and clock.now < expiry
                print(f"Banner witness: {name}, token {token}, logical cache age {age:.3f}s")
                # The attention frame for this completion is published in the
                # same tick as the banner and BEFORE it, so its arrival is what
                # says the tick ran at all.
                assert any(
                    frame["type"] == "attention"
                    and frame.get("session_id") == target
                    and frame["payload"].get("completion_token") == token
                    for frame in streamed
                ), f"{name}: the tick never ran"
                announced = announced_for(token)
                assert bool(announced) is expected, f"{name}: banner verdict {streamed}"
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
        # HER BUILT-IN ROW IS IN EVERY LIST. The server boot hook ensures
        # Aida's session (a USER session by design — R3/R7), so a fresh
        # config root lists three rows here, not two. This test's subject is
        # the two rows IT created, so her row — cold, and none of this
        # test's business — is filtered out; her presence and pinning are
        # asserted by their own tests.
        builtin_ids = _builtin_session_ids(root)
        before_rows = [row for row in before["sessions"] if row["id"] not in builtin_ids]
        before_ids = [row["id"] for row in before_rows]
        assert before_ids.index(elder) < before_ids.index(completer), before_ids
        assert all(row["active"] for row in before_rows), before_rows

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
    after = [row for row in listed.json()["result"]["sessions"] if row["id"] not in builtin_ids]
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

    1. a pin WRITE publishes exactly one catalogue frame (awaited as a fact —
       see ``_await_frames`` for why a post-ack window cannot attribute it, and
       why the guard is a backstop rather than a bound);
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
            # The CONNECTION BOUNDARY establishes the catalogue token
            # (``_take_baseline`` primes it), and priming publishes NOTHING — the
            # feed announces changes after connect, not state, so nothing
            # catalogue-shaped arrives here in a healthy run. The sleep stays as
            # slack so the windows below are attributable to the first write
            # alone.
            await asyncio.sleep(2.5)

            # (2) A quiet window publishes NOTHING catalogue-shaped.
            quiet_from = len(streamed)
            await asyncio.sleep(2.5)
            assert catalogues(quiet_from) == [], streamed[quiet_from:]

            # (1) The pin write, and the frame it owes. The baseline is taken
            # BEFORE the write: the frame is owed the moment the write lands,
            # and `_await_frames` carries why a post-ack sample cannot
            # attribute it.
            wrote_from = len(streamed)
            pinned = await client.post(
                f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True}
            )
            assert pinned.status_code == 200, pinned.text
            # HER PIN IS THERE TOO (R27's auto-pin, written by the same boot
            # hook that created her): the claim under test is that THIS pin
            # write landed and rang the bell, so the file must hold both ids
            # and nothing else.
            builtin_ids = _builtin_session_ids(root)
            assert sorted(json.loads((root / "sidebar-pins.json").read_text())) == sorted(
                [session_id, *builtin_ids]
            )

            published = await _await_frames(
                streamed, "catalogue", since=wrote_from, what="the pin write"
            )
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


@pytest.mark.asyncio
async def test_an_authored_profile_and_team_ring_the_authoring_doorbell(
    desktop_server, workspace: Path
):
    """THE AUTHORING DOORBELL, over the real stack.

    The reported defect: an agent inside a session creates a team or a profile and
    the app's Teams/Agents lists keep showing the rows they were mounted with until
    a refresh or a tab switch. The only vehicle for the fix is this feed, so this
    is the test that says the token actually moves for BOTH registries — a unit
    test of the key string cannot, because the failure it guards against (a probe
    that never notices the write) lives entirely in the timing between a write and
    a frame.

    Both legs drive the real surfaces rather than a hand-written file: the profile
    through ``write_profile``, which is the call the ``agent`` tool makes from
    inside a session, and the team through ``POST /v1/desktop/teams``, which is the
    route the app (and its own agents) use.

    Four claims, each needing the HTTP layer:

    1. authoring a profile publishes exactly one ``authoring`` frame (awaited as
       a fact — see ``_await_frames`` for why a post-ack window cannot
       attribute it, and why the guard is a backstop rather than a bound);
    2. a team created over the route publishes exactly one more, with the counter
       advanced by one;
    3. a QUIET window after each publishes none — otherwise the feed would refetch
       every sidebar's profiles and teams once a second for nothing;
    4. a client connecting afterwards is not REPLAYED the invalidation: its ``open``
       snapshot carries the counter it would otherwise have been told about.
    """
    root, client = desktop_server

    async with client.stream("GET", "/v1/desktop/events") as response:
        assert response.status_code == 200, response.read()
        lines = response.aiter_lines()
        opened = await _next_frame(lines, lambda f: f["type"] == "open")
        assert opened["payload"]["authoring_revision"] is not None

        # A READER TASK, not a bounded read per window: cancelling an
        # `aiter_lines()` iteration closes the response it is reading, so a
        # windowed read tears the subscription down and every later window reads
        # nothing at all. The same idiom the pin-doorbell test above uses.
        streamed: list[dict[str, Any]] = []

        async def pump() -> None:
            async for line in lines:
                if line.startswith("data: "):
                    streamed.append(json.loads(line[6:]))

        reader = asyncio.create_task(pump())

        def authoring(since: int) -> list[dict[str, Any]]:
            return [frame for frame in streamed[since:] if frame["type"] == "authoring"]

        try:
            # The CONNECTION BOUNDARY establishes the authoring token
            # (``_take_baseline`` primes it), and priming publishes NOTHING — the
            # feed announces changes after connect, not state. A healthy run sees
            # an empty stream here until the first write (instrumented
            # 2026-10-17: no frame at all in the first five seconds); the sleep
            # stays as slack so the windows below are attributable to the writes
            # alone.
            await asyncio.sleep(2.5)

            quiet_from = len(streamed)
            await asyncio.sleep(2.5)
            assert authoring(quiet_from) == [], streamed[quiet_from:]

            # (1) A role authored the way the `agent` tool authors one. The
            # baseline is taken BEFORE the write: the frame is owed the moment
            # the write lands, and `_await_frames` carries why a post-ack sample
            # cannot attribute it.
            wrote_from = len(streamed)
            registry = AgentRegistry(root)
            name, kind, _action_class = await asyncio.to_thread(
                write_profile,
                registry,
                AgentParams(
                    op="create",
                    name="e2e-doorbell-role",
                    instructions="Answer the question in one sentence.",
                ),
                creating=True,
                scope="global",
            )
            assert (name, kind) == ("e2e-doorbell-role", "role"), (name, kind)
            assert (root / "agents").is_dir(), "the write did not create a profile row"

            profiles = await _await_frames(
                streamed, "authoring", since=wrote_from, what="the profile write"
            )
            assert len(profiles) == 1, streamed[wrote_from:]
            revisions = [frame["payload"]["revision"] for frame in profiles]

            # ...and nothing follows it while both registries sit still. This is
            # the half that catches a token keyed on something a turn moves: with
            # the projection missing, a write of this shape keeps publishing.
            after_from = len(streamed)
            await asyncio.sleep(2.5)
            assert authoring(after_from) == [], streamed[after_from:]

            # (2) A team created over the route the app uses. Baseline before
            # the write, for the profile leg's reason.
            team_from = len(streamed)
            created = await client.post(
                "/v1/desktop/teams",
                json={"request_id": str(uuid.uuid4()), "name": "e2e-doorbell-team"},
            )
            assert created.status_code == 200, created.text

            teams = await _await_frames(
                streamed, "authoring", since=team_from, what="the team write"
            )
            assert len(teams) == 1, streamed[team_from:]
            revisions.append(teams[0]["payload"]["revision"])
            # The feed is not a session, and a fabricated id would make the
            # client's `observe(sessionId, frame)` look like it had one.
            assert "session_id" not in teams[0], teams[0]

            # (3) ...and the team write is followed by a quiet window too.
            quiet_after_from = len(streamed)
            await asyncio.sleep(2.5)
            assert authoring(quiet_after_from) == [], streamed[quiet_after_from:]
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader

    # The counter is MONOTONE across both writes, which is what makes it usable as
    # a dependency value on the client: two invalidations cannot share a number.
    assert revisions == [revisions[0], revisions[0] + 1], revisions

    # (4) A LATER subscriber is not replayed the invalidation: its own `open`
    # snapshot is the answer, and a replayed frame would make every reconnecting
    # client refetch two lists nothing has changed.
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

    assert streamed_later[0]["payload"]["authoring_revision"] == revisions[-1]
    assert [frame for frame in streamed_later if frame["type"] == "authoring"] == []

"""The private request spends exactly one launch's authority and loop verdict."""

from __future__ import annotations

import asyncio
import os
import socket
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

from local_operator.server import desktop_idle_update as idle_update
from local_operator.server import reload


class Owner:
    def __init__(self) -> None:
        self.closed = False
        self.why = None

    def begin_reload(self) -> None:
        assert not self.closed
        self.closed = True

    def end_reload(self) -> None:
        self.closed = False

    def reload_blocker(self) -> str | None:
        assert self.closed
        return self.why


@pytest.fixture
def daemon():
    return SimpleNamespace(
        state=SimpleNamespace(
            instance_id="x" * 43,
            scheduler_service=Owner(),
            hub_sync=Owner(),
            job_manager=SimpleNamespace(jobs={}),
            **{reload.FENCE_ATTR: True, reload.REQUESTS_ATTR: 0, reload.UPDATE_STREAMS_ATTR: set()},
        )
    ), SimpleNamespace(started=True, should_exit=False)


@pytest.mark.parametrize(
    "blocker",
    ["scheduler", "hub", "http", "mesh", "tunnel", "sign-in", "mcp", "settings", "attest"],
)
def test_owned_work_refuses_without_consuming_and_reopens(daemon, blocker):
    app, server = daemon
    state = app.state
    pending = SimpleNamespace(done=lambda: False)
    if blocker == "scheduler":
        state.scheduler_service.why = "scheduler run"
    elif blocker == "hub":
        state.hub_sync.why = "active hub operation"
    elif blocker == "http":
        setattr(state, reload.REQUESTS_ATTR, 1)
    elif blocker == "mesh":
        state.mesh_onboard_tasks = {"example": pending}
    elif blocker == "tunnel":
        state.tunnel_setup_task = pending
    elif blocker == "sign-in":
        state.desktop_auth = SimpleNamespace(idle_update_busy=lambda: True)
    elif blocker == "mcp":
        state.desktop_mcp_host = SimpleNamespace(ops=SimpleNamespace(running={"probe"}))
    elif blocker == "settings":
        state.settings_applier = SimpleNamespace(idle_update_busy=lambda: True)
    else:
        state.mesh_pairing_attest = SimpleNamespace(in_flight=1)

    def must_not_consume():
        pytest.fail("busy daemon issued a committed receipt")

    assert idle_update._try_stop(app, server, None, must_not_consume) is not None
    assert not state.scheduler_service.closed
    assert not state.hub_sync.closed
    assert not getattr(state, reload.ADMISSION_ATTR)
    assert not server.should_exit


@pytest.mark.parametrize(
    ("attr", "state_check", "reason"),
    [
        ("librarian_boot_task", "done", "librarian-bootstrap"),
        ("bridge_connector_boot_task", "done", "connector-bootstrap"),
        ("org_hub_boot_thread", "is_alive", "org-hub-bootstrap"),
    ],
)
def test_pending_durable_boot_work_defers_only_the_private_stop(daemon, attr, state_check, reason):
    app, server = daemon
    held = True
    setattr(
        app.state,
        attr,
        SimpleNamespace(**{state_check: lambda: held == (state_check == "is_alive")}),
    )
    # The ordinary reload inventory is unchanged by these update-only fences.
    app.state.scheduler_service.begin_reload()
    app.state.hub_sync.begin_reload()
    assert reload.work_blocker(app) is None
    app.state.scheduler_service.end_reload()
    app.state.hub_sync.end_reload()

    consumed = False

    def acknowledge():
        nonlocal consumed
        consumed = True
        return True

    assert idle_update._try_stop(app, server, None, acknowledge) == reason
    assert not consumed and not server.should_exit
    assert not getattr(app.state, reload.ADMISSION_ATTR)
    held = False
    assert idle_update._try_stop(app, server, None, acknowledge) is None
    assert consumed and server.should_exit


def test_pending_reload_refuses_without_stealing_the_fence(daemon):
    app, server = daemon
    setattr(app.state, reload.ADMISSION_ATTR, True)
    assert idle_update._try_stop(app, server, None, lambda: True) == "transition"
    assert getattr(app.state, reload.ADMISSION_ATTR)
    assert not server.should_exit


@pytest.mark.parametrize("write_error", [False, True])
def test_unconsumed_commit_reopens_admissions(daemon, write_error):
    app, server = daemon

    def acknowledge():
        if write_error:
            raise BrokenPipeError
        return False

    if write_error:
        with pytest.raises(BrokenPipeError):
            idle_update._try_stop(app, server, None, acknowledge)
    else:
        assert idle_update._try_stop(app, server, None, acknowledge) == "not-consumed"
    assert not getattr(app.state, reload.ADMISSION_ATTR)
    assert not app.state.scheduler_service.closed
    assert not app.state.hub_sync.closed
    assert not server.should_exit


@pytest.mark.asyncio
async def test_channel_identity_and_duplicate_commit(daemon):
    app, server = daemon
    left, right = socket.socketpair()
    with left, right:
        left.setblocking(False)
        idle_update.bind(app, idle_update.Channel(left, os.getppid()), server)
        reader, writer = await asyncio.open_connection(sock=right)
        control = asyncio.create_task(idle_update.run(app, None))
        try:
            async with asyncio.timeout(2):
                instance = app.state.instance_id.encode()
                expected = b"desktop-idle-update-v1 %d %s\n" % (os.getpid(), instance)
                assert await reader.readline() == expected
                writer.write(b"stop previous-instance\n")
                assert await reader.readline() == b"refused identity\n"
                assert not server.should_exit

                app.state.scheduler_service.why = "busy"
                writer.write(b"stop " + instance + b"\n")
                assert await reader.readline() == b"deferred owned-work\n"
                assert not app.state.scheduler_service.closed

                app.state.scheduler_service.why = None
                writer.write((b"stop " + instance + b"\n") * 2)
                assert await reader.readline() == b"accepted " + instance + b"\n"
                # Unread input when closing a duplex pair can produce RST; either
                # EOF or reset means no reply or opportunity to consume twice.
                try:
                    assert await reader.readline() == b""
                except ConnectionResetError:
                    pass
                await control
                assert server.should_exit
                assert getattr(app.state, reload.ADMISSION_ATTR)
                assert app.state.scheduler_service.closed
        finally:
            control.cancel()
            writer.close()


@pytest.mark.asyncio
async def test_accepted_stop_exits_with_nonreading_tcp_feed(daemon, monkeypatch):
    app, _ = daemon
    parked = asyncio.Event()
    initial_event = b"data: ready\n\n"

    async def sse(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        if not scope["query_string"]:
            await send({"type": "http.response.body", "body": initial_event, "more_body": True})
            await parked.wait()
        else:
            # A small-window TCP peer that never reads eventually pauses writes.
            # The large test-only payload reaches that state without timing it.
            body = b"data:" + b"x" * (1 << 20) + b"\n\n"
            while True:
                await send({"type": "http.response.body", "body": body, "more_body": True})

    async def dispatch(request, call_next):
        return await call_next(request)

    from uvicorn.protocols.http.auto import AutoHTTPProtocol

    server = uvicorn.Server(
        uvicorn.Config(
            # Like the production app, an outer HTTP middleware queues the body
            # rather than sending it directly to the transport.
            BaseHTTPMiddleware(reload.ReloadAdmissionMiddleware(sse, owner=app), dispatch),
            http=idle_update.stream_protocol(AutoHTTPProtocol, app),
            lifespan="off",
            log_config=None,
        )
    )
    monkeypatch.setattr(server, "capture_signals", nullcontext)
    with socket.socket() as listener, socket.socket() as blocked:
        listener.bind(("127.0.0.1", 0))
        address = listener.getsockname()
        blocked.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        blocked.setblocking(False)
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        reader_transport = None
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    await asyncio.sleep(0)
                reader, reader_transport = await asyncio.open_connection(*address)
                reader_transport.write(
                    b"GET /v1/desktop/events HTTP/1.1\r\nHost: localhost\r\n\r\n"
                )
                assert (await reader.readuntil(initial_event + b"\r\n")).startswith(b"HTTP/1.1 200")

                loop = asyncio.get_running_loop()
                await loop.sock_connect(blocked, address)
                await loop.sock_sendall(
                    blocked,
                    b"GET /v1/desktop/sessions/abc/events?blocked HTTP/1.1\r\n"
                    b"Host: localhost\r\n\r\n",
                )
                while not any(conn.flow.write_paused for conn in server.server_state.connections):
                    await asyncio.sleep(0)
                assert len(getattr(app.state, reload.UPDATE_STREAMS_ATTR)) == 2

                assert idle_update._try_stop(app, server, None, lambda: True) is None
                # The reading peer gets valid chunked framing. The other TCP
                # socket remains open and has never read a byte when serve exits.
                assert await reader.read() == b"0\r\n\r\n"
                await asyncio.shield(serving)
                assert blocked.fileno() >= 0
        finally:
            if reader_transport is not None:
                reader_transport.close()
                await reader_transport.wait_closed()
            if not serving.done():
                blocked.close()
                server.should_exit = True
                await serving


@pytest.mark.asyncio
async def test_incomplete_command_at_eof_cannot_stop(daemon):
    app, server = daemon
    left, right = socket.socketpair()
    with left, right:
        left.setblocking(False)
        right.setblocking(False)
        idle_update.bind(app, idle_update.Channel(left, os.getppid()), server)
        control = asyncio.create_task(idle_update.run(app, None))
        loop = asyncio.get_running_loop()
        try:
            async with asyncio.timeout(2):
                assert (await loop.sock_recv(right, 1024)).startswith(b"desktop-idle-update-v1 ")
                await loop.sock_sendall(right, b"stop " + app.state.instance_id.encode())
                right.shutdown(socket.SHUT_WR)
                await control
                assert not server.should_exit
                assert not app.state.scheduler_service.closed
        finally:
            control.cancel()

"""The private request spends exactly one launch's authority and loop verdict."""

from __future__ import annotations

import asyncio
import os
import socket
from types import SimpleNamespace

import pytest

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

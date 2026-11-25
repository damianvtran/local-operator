"""Launch-private validate-and-stop for the desktop overnight updater.

There is deliberately no HTTP route, discovery record, credential on disk, or
shared bearer that can exercise this authority. The ability to write the other
end of an unnamed socketpair is the launch capability. The kernel's peer PID
must also be the direct parent; accepting a connected named socket would let a
local proxy turn the control into a listener. The daemon never connects back.

The app must bind the hello to both its original ChildProcess PID and the
instance_id of its directly connected, authenticated HTTP backend. A reconnect
or re-exec does not transfer ownership. See docs/desktop-idle-update.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from local_operator.server import reload

logger = logging.getLogger(__name__)
ATTACHED = "desktop_idle_update_control"
STOPPING = "desktop_idle_update_stopping"
# A v1 command can hold `stop ` plus one 256-bit URL-safe boot ID and LF.
# secrets.token_urlsafe(32) encodes 32 bytes in 43 unpadded characters.
MAX_LINE = len(b"stop ") + 43 + len(b"\n")


@dataclass
class Channel:
    socket: socket.socket
    parent: int

    def still_owned(self) -> bool:
        return self.parent > 1 and os.getppid() == self.parent


def adopt(fd: int) -> Channel | None:
    """Take the app's extra stdio slot before branding, shell probes or spawns."""
    if fd != 3 or sys.platform not in ("darwin", "linux"):
        return None
    try:
        channel = socket.socket(fileno=fd)
    except OSError:
        return None
    adopted = False
    try:
        if (
            channel.family != socket.AF_UNIX
            or channel.type != socket.SOCK_STREAM
            or channel.getsockname()
            or channel.getpeername()
        ):
            return None
        if sys.platform == "darwin":
            # Darwin SOL_LOCAL/LOCAL_PEERPID (sys/un.h), measured with libuv's
            # inherited Node stdio socketpair, not a guessed pathname/UID.
            peer = struct.unpack("i", channel.getsockopt(0, 2, struct.calcsize("i")))[0]
        else:
            peer, _, _ = struct.unpack(
                "iII",
                channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("iII")),
            )
        parent = os.getppid()
        if parent <= 1 or peer != parent:
            return None
        channel.set_inheritable(False)
        channel.setblocking(False)
        adopted = True
        return Channel(channel, parent)
    except OSError:
        return None
    finally:
        if not adopted:
            # Invalid declarations are not ours; don't consume someone else's fd.
            channel.detach()


def bind(app: Any, channel: Channel, server: Any) -> None:
    """Only the fresh, owned uvicorn Server may install the control lifecycle."""
    setattr(app.state, ATTACHED, (channel, server))
    setattr(app.state, reload.UPDATE_STREAMS_ATTR, set())


def stopping(app: Any) -> bool:
    return getattr(app.state, STOPPING, False) is True


def _extra_blocker(state: Any, settings_resume: asyncio.Task[int] | None) -> str | None:
    # Relay-backed onboarding and tunnel setup can outlive the HTTP response.
    # The relay service itself and remote/runtime turns are different processes;
    # they are neither sampled as daemon work nor stopped by this transaction.
    if any(not task.done() for task in getattr(state, "mesh_onboard_tasks", {}).values()):
        return "mesh-onboarding"
    tunnel = getattr(state, "tunnel_setup_task", None)
    if tunnel is not None and not tunnel.done():
        return "tunnel-setup"
    auth = getattr(state, "desktop_auth", None)
    if auth is not None and auth.idle_update_busy():
        return "desktop-sign-in"
    mcp = getattr(state, "desktop_mcp_host", None)
    if mcp is not None and mcp.ops.running:
        return "desktop-mcp"
    if settings_resume is not None and not settings_resume.done():
        return "settings-intent-resume"
    applier = getattr(state, "settings_applier", None)
    if applier is not None and applier.idle_update_busy():
        return "settings-intent"
    attest = getattr(state, "mesh_pairing_attest", None)
    if attest is not None and attest.in_flight:
        return "mesh-pairing"
    return None


def _try_stop(
    app: Any,
    server: Any,
    settings_resume: asyncio.Task[int] | None,
    acknowledge: Callable[[], bool],
) -> str | None:
    """One loop turn from taking the gates through handing exit to uvicorn.

    No wait for busy work and no client-owned reservation. A refused, failed or
    disconnected commit reopens all acquired gates on this same turn; a
    consumed commit keeps the gates through ordinary uvicorn shutdown.
    """
    state = app.state
    if not server.started or server.should_exit or any(
        getattr(state, name, False) for name in (reload.ADMISSION_ATTR, reload.PENDING_ATTR)
    ):
        return "transition"
    if reload.rolling_contract(app) != reload.ROLLING_CONTRACT:
        return "unsupported"
    scheduler = getattr(state, "scheduler_service", None)
    if scheduler is None or getattr(state, "job_manager", None) is None:
        return "unsupported"
    hub = state.hub_sync
    committed = False
    setattr(state, reload.ADMISSION_ATTR, True)
    try:
        hub.begin_reload()
        scheduler.begin_reload()
        busy = reload.work_blocker(app)
        if busy is not None:
            logger.debug("desktop idle update deferred: %s", busy)
            return "owned-work"
        reason = _extra_blocker(state, settings_resume)
        if reason is not None:
            return reason
        streams = tuple(getattr(state, reload.UPDATE_STREAMS_ATTR))
        # Do not cross a disconnect/failure after the final verdict and then
        # stop anyway. A small nonblocking write either queues the whole frame
        # or fails this attempt; a partial frame is not a valid receipt.
        if server.should_exit or not acknowledge():
            return "not-consumed"
        committed = True
        setattr(state, STOPPING, True)
        server.should_exit = True
        for stream in streams:
            stream.cancel()
        return None
    finally:
        if not committed:
            setattr(state, reload.ADMISSION_ATTR, False)
            try:
                scheduler.end_reload()
            finally:
                hub.end_reload()


async def run(app: Any, settings_resume: asyncio.Task[int] | None) -> None:
    channel, server = getattr(app.state, ATTACHED)
    sock = channel.socket
    loop = asyncio.get_running_loop()
    try:
        if not channel.still_owned() or reload.rolling_contract(app) != reload.ROLLING_CONTRACT:
            return
        identity = app.state.instance_id.encode("ascii")
        stop = b"stop " + identity
        if len(stop) + 1 != MAX_LINE:
            return
        await loop.sock_sendall(
            sock, b"desktop-idle-update-v1 %d %s\n" % (os.getpid(), identity)
        )
        buffer = b""
        while channel.still_owned():
            chunk = await loop.sock_recv(sock, MAX_LINE - len(buffer))
            if not chunk or not channel.still_owned():
                return
            buffer += chunk
            if b"\n" not in buffer:
                if len(buffer) == MAX_LINE:
                    return
                continue
            line, buffer = buffer.split(b"\n", 1)
            if line != stop:
                await loop.sock_sendall(sock, b"refused identity\n")
                continue
            receipt = b"accepted " + identity + b"\n"
            reason = _try_stop(
                app, server, settings_resume,
                lambda: channel.still_owned() and sock.send(receipt) == len(receipt),
            )
            if reason is None or reason == "not-consumed":
                return
            await loop.sock_sendall(sock, b"deferred " + reason.encode("ascii") + b"\n")
    except OSError:
        # EOF, an unwritable reply or an absent peer cannot authorize a stop.
        logger.debug("desktop idle update channel closed", exc_info=True)
    except Exception:  # noqa: BLE001 — provision is optional; unreadable means no exit
        logger.warning("desktop idle update control could not complete", exc_info=True)
    finally:
        sock.close()

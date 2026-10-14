"""Ready-successor handoff for a listening daemon that must not be killed.

The update that installs a new build runs while the old daemon is still serving
operator work. Killing it loses in-flight turns, so the new generation is brought
up as a FENCED successor: it constructs its acceptors but serves nothing, the old
daemon drains its local work and stops accepting on one non-awaiting step, and
only then does the successor accept. Accepted connections and their tasks are
never torn down — the transfer is of the LISTENING socket, not of live traffic.

Scope and constraints:

* Capable builds only. A stock install whose API lacks the rolling contract
  cannot construct the successor and refuses; the one-time bootstrap for such an
  install is a separate, operator-owned step, not something this module fakes.
* POSIX passes the listener by ``os.dup`` over ``SCM_RIGHTS`` on a same-machine
  control channel — the same descriptor mechanism ``session.runtime.standby``
  already uses. Windows has no integer-FD equivalent here and refuses.
* Nothing here calls ``Server.shutdown``/``should_exit``, closes accepted
  transports, or touches the lifespan: ``pause_accept`` closes ACCEPTORS only,
  exactly the shape uvicorn's own reload uses (``server.py:285-286``).
* The transfer record rides the existing install-root transaction as one extra
  observation on the durable operation — no second lock, no CAS service.
"""

from __future__ import annotations

import asyncio
import os
import socket
import struct
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

HANDOFF_CONTRACT = "ready-successor-v1"
#: Record states. ``ready`` is B announcing under the transaction; ``commit`` is
#: A's irreversible step; ``ack`` is B confirming it holds the listener; anything
#: else (abort/unknown) leaves BOTH roots and BOTH processes alive for recovery.
HANDOFF_STATES = ("ready", "commit", "ack", "aborted")
_CONTROL_HEADER = struct.Struct("!I")
_MAX_FRAME = 1 << 16


class HandoffError(RuntimeError):
    """An actionable refusal; never permission to force or to kill a daemon."""


class ListenerUnsupported(HandoffError):
    """This platform/build cannot transfer a listener; refuse, do not improvise."""


def _fd_socket_path_supported() -> bool:
    return hasattr(os, "dup") and hasattr(socket, "send_fds") and os.name != "nt"


class ListenerOwner:
    """Owns the master listening socket and the accept-only lifecycle.

    One per listening daemon. The acceptors are recreated from the retained
    master socket rather than rebound, so the bound port never closes and the
    listener inode is stable across a handoff — the successor verifies that inode
    before it accepts, which is what makes "the same port" a checkable fact
    instead of a hope.
    """

    def __init__(self, app: Any, servers: "list[Any]", listener: socket.socket) -> None:
        self.app = app
        # A DUP of the bound listener: closing an acceptor must not close the
        # master, and the original object stays owned by the caller's frame.
        self.socket = listener.dup() if hasattr(listener, "dup") else listener
        self.servers = servers
        self.accepting = True
        self._create: Callable[[], Awaitable[list[Any]]] | None = None

    @property
    def listener_ino(self) -> str:
        """A stable identity for THIS listener, so a successor can prove it."""

        info = os.fstat(self.socket.fileno())
        return f"{info.st_dev}:{info.st_ino}"

    def _bind_acceptor_factory(self, factory: Callable[[], Awaitable[list[Any]]]) -> None:
        self._create = factory

    async def start_fenced(self) -> None:
        """Build acceptors WITHOUT serving. Nothing is accepted until resume.

        ``start_serving=False`` is the whole point: a successor that accepted
        before the old daemon committed would serve two leaders for the same
        port. uvicorn 0.54's ``startup`` serves immediately, so the factory is
        supplied by the caller (which owns the protocol class) and the acceptors
        are created here, fenced.
        """

        if self._create is None:
            raise HandoffError("fenced start requires an acceptor factory from the caller")
        self.servers = await self._create()
        self.accepting = False

    def pause_accept(self) -> None:
        """Stop accepting NEW connections; leave accepted ones running.

        ``Server.close`` on an acceptor closes only the listening half — accepted
        transports hold their own descriptors and are untouched. This is the
        uvicorn reload shape, not ``Server.shutdown``.
        """

        for server in self.servers:
            server.close()
        self.accepting = False

    async def resume_accept(self) -> None:
        """Rebuild acceptors from the retained master, then serve.

        A closed acceptor is terminal — ``Server.close()`` nulls its ``_sockets``,
        so ``start_serving`` cannot revive it. Resume therefore recreates acceptors
        from ``self.socket`` (which was never closed, because the acceptor owns a
        dup) and starts them. Never rebinds: the bound port must not move.
        """

        if self._create is None:
            raise HandoffError("resume requires an acceptor factory from the caller")
        self.servers = await self._create()
        for server in self.servers:
            await server.start_serving()
        self.accepting = True

    def pass_control(self, channel: socket.socket, payload: bytes = b"") -> None:
        """Send the master listener descriptor over a same-machine channel.

        Only the LISTENER (and never accepted transports) crosses the boundary;
        the payload carries the transfer record the receiver verifies.
        """

        if not _fd_socket_path_supported():
            raise ListenerUnsupported(
                "listener handoff needs POSIX descriptor passing; refusing on this platform"
            )
        header = _CONTROL_HEADER.pack(len(payload))
        socket.send_fds(channel, [header + payload], [self.socket.fileno()])


def receive_listener(channel: socket.socket) -> "tuple[socket.socket, bytes]":
    """Read a transferred listener + record from a control channel.

    Reads the framed header with ``recv_fds`` so the descriptor arrives with the
    message that names it (the ``standby`` convention: a marker byte sent
    separately is indistinguishable from the frame's length prefix).
    """

    if not _fd_socket_path_supported():
        raise ListenerUnsupported(
            "listener receive needs POSIX descriptor passing; refusing on this platform"
        )
    header = b""
    fds: list[int] = []
    while len(header) < _CONTROL_HEADER.size:
        chunk, received, _flags, _addr = socket.recv_fds(
            channel, _CONTROL_HEADER.size - len(header), max(1, 4 - len(fds))
        )
        if not chunk:
            for fd in fds:
                os.close(fd)
            raise HandoffError("handoff control channel closed before the listener arrived")
        header += chunk
        fds.extend(received)
    if len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise HandoffError("handoff must carry exactly one listener descriptor")
    size = _CONTROL_HEADER.unpack(header)[0]
    if size > _MAX_FRAME:
        os.close(fds[0])
        raise HandoffError("handoff record frame is unreasonably large")
    payload = b""
    while len(payload) < size:
        chunk = channel.recv(size - len(payload))
        if not chunk:
            os.close(fds[0])
            raise HandoffError("handoff record truncated")
        payload += chunk
    return socket.socket(fileno=fds[0]), payload


async def _noop() -> None:  # pragma: no cover - reserved default factory shape
    return None


def _acceptor_factory(listener: socket.socket, protocol_factory: Callable[[], Any]):
    """Return an async factory that creates acceptors with ``start_serving=False``.

    The acceptor is built from a DUP of the master. ``loop.create_server(sock=)``
    takes ownership of the socket it is handed, so closing an acceptor (pause) would
    otherwise close the one master the handoff must still pass on. Each rebuild dups
    afresh, so a pause/resume cycle never exhausts or closes the retained master.
    """

    async def build() -> list[Any]:
        loop = asyncio.get_running_loop()
        owned = listener.dup()
        try:
            server = await loop.create_server(
                protocol_factory, sock=owned, start_serving=False
            )
        except BaseException:
            owned.close()
            raise
        return [server]

    return build


def _scheduler_occurrence_exclusions(record: "dict[str, Any] | None") -> "list[str]":
    """Occurrence IDs the successor must not re-dispatch.

    A wrote these at commit: scheduler ``/`` relay work that was mid-flight on A.
    B skips them until each records durable completion, so the swap never
    double-dispatches a scheduled job or a peer message.
    """

    if not record:
        return []
    inherited = record.get("handoff", {}).get("inherited")
    if not isinstance(inherited, list):
        return []
    return [item for item in inherited if isinstance(item, str)]


__all__ = [
    "HANDOFF_CONTRACT",
    "HANDOFF_STATES",
    "HandoffError",
    "ListenerOwner",
    "ListenerUnsupported",
    "receive_listener",
    "_acceptor_factory",
    "_scheduler_occurrence_exclusions",
]

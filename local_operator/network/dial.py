"""The ONE way the mesh package opens a local session-control socket.

WHY THIS MODULE EXISTS AT ALL. ``mesh-transport-identity.md`` §7.2 asks for one
place — ``Authorizer.dial_local`` — through which every local control socket in
this package is opened. The reason is not tidiness: a second dialer is a second
place that decides with what authority a frame reaches a runtime, and the
authority here is delicate. This dial always declares ``locality: "remote"``
(§7.4): the runtime's own locality gates are what stop a remote viewer from
being treated as a terminal on the runtime's machine, and the property is
enforced at the one process that can enforce it — the side that dials.

IT NEVER DIALS AS ``daemon``. §3.5: there is exactly one ``daemon`` slot per
runtime and it belongs to the local phone daemon; a relay claiming it would
evict the phone from the peer's own runtime. Every connection this module opens
is kind ``attach``, which is also what makes the relay one of the viewers the
attach cap bounds rather than a second privileged class.

WHY IT DOES NOT REUSE ``wire.FrameReader``. The runtime's control socket is
JSON-lines, and ``wire``'s line reader is bounded at 16 KiB
(``MAX_HANDSHAKE_LINE``) because its job is handshake frames. A session frame is
not: a welcome projection carries a transcript tail in ONE line, and the
existing control dialer documents the consequence — ``StreamReader.readline``
raises instead of returning such a line, so the frame can never be read and
everything behind it is lost (``session/runtime/control.py``). This reader
therefore keeps the same discipline as that one: read in chunks, frame here,
bound at ``MAX_SESSION_FRAME_BYTES`` (8 MiB, the same number), and DISCARD an
oversized line rather than raise — a frame that cannot be parsed must cost its
own frame, never the connection.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: The largest session frame this reader will assemble. The same number
#: ``session/runtime/control.py`` uses for the same socket, deliberately: two
#: readers of one socket disagreeing about "too big" is a frame one accepts and
#: the other loses.
MAX_SESSION_FRAME_BYTES = 1 << 23

_READ_CHUNK = 1 << 16

#: How long to wait for the runtime's unsolicited welcome after authenticating.
#: The same budget ``control.py`` uses for its identity check, because it is the
#: same wait: an authenticated dial always receives a projection first.
WELCOME_TIMEOUT_S = 10.0


class OwnerUnreachable(Exception):
    """No conversation was possible with a session's runtime.

    Its own type so a caller can tell "the owner is gone" (the ordinary
    ``degrade`` case) from "the owner answered with an error" (a refusal the
    peer must see). Nothing here ever reports one as the other.
    """


@dataclass
class LineReader:
    """JSON-lines frames from one socket, bounded but generously.

    Extracted so the relay's local control socket and its dials to a runtime
    frame frames IDENTICALLY: two readers on one socket is already impossible
    (the second would lose whatever the first buffered), and two readers of the
    same protocol with different limits is how a frame is one transport's
    business and the other's loss.
    """

    sock: socket.socket
    limit: int = MAX_SESSION_FRAME_BYTES
    #: Set when the socket reached EOF or failed rather than merely going
    #: quiet. A pump that only sees ``None`` cannot tell "nothing yet" from
    #: "owner gone", and those are different events to a viewer.
    eof: bool = False
    _buffer: bytearray = field(default_factory=bytearray)
    _skipping: bool = False

    def read_frame(self, timeout_s: float) -> dict[str, Any] | None:
        """The next frame, or ``None`` on timeout or a closed socket."""
        deadline = time.monotonic() + timeout_s
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                if self._skipping:
                    # This newline ends a discarded oversized line; whatever
                    # follows it is a real frame again.
                    self._skipping = False
                    continue
                try:
                    frame = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if isinstance(frame, dict):
                    return frame
                continue
            if len(self._buffer) > self.limit:
                # Bounded memory, and the frames after such a line still
                # survive in the buffer — see the module docstring.
                self._buffer.clear()
                self._skipping = True
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                self.sock.settimeout(remaining)
                chunk = self.sock.recv(_READ_CHUNK)
            except (socket.timeout, TimeoutError):
                return None
            except OSError:
                self.eof = True
                return None
            if not chunk:
                self.eof = True
                return None
            self._buffer.extend(chunk)


@dataclass
class OwnerDial:
    """One authenticated ``attach`` conversation with a session's runtime."""

    sock: socket.socket
    session_id: str
    reader: "LineReader | None" = None
    #: Set when the socket reached EOF rather than merely going quiet. A pump
    #: that only sees ``None`` cannot tell "nothing yet" from "owner gone",
    #: and those are different events to a viewer.
    eof: bool = False

    def _read(self) -> "LineReader":
        if self.reader is None:
            self.reader = LineReader(self.sock)
        return self.reader

    # -- framing ------------------------------------------------------------

    def recv(self, timeout_s: float) -> dict[str, Any] | None:
        """The next frame, or ``None`` on timeout or a closed socket."""
        frame = self._read().read_frame(timeout_s)
        self.eof = self._read().eof
        return frame

    def send(self, frame: dict[str, Any]) -> None:
        self.sock.sendall(
            (json.dumps(frame, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )

    def exchange(
        self,
        frame: dict[str, Any],
        *,
        timeout_s: float,
        on_frame: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        """Send one op and return ITS reply, skipping state frames on the way.

        A reply is matched on ``req``, never on arrival order: an authenticated
        attach connection also receives the welcome projection and, once
        subscribed, raw event frames — an ack handed to the wrong caller is a
        cross-wired answer that looks like success.
        """
        req = frame.get("req")
        self.send(frame)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            reply = self.recv(remaining)
            if reply is None:
                return None
            if reply.get("req") == req and reply.get("op") in ("ack", "error"):
                return reply
            if on_frame is not None:
                on_frame(reply)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


#: The attach auth fields a viewer declares per connection (§3.5: ``events``,
#: ``frontend_state``, ``display_window``, ``slash_consumers`` and ``surface``
#: are per-CONNECTION facts, which is exactly why the relay forwards one
#: upstream connection per viewer rather than multiplexing them).
AUTH_FIELDS: tuple[str, ...] = (
    "events",
    "frontend_state",
    "display_window",
    "slash_consumers",
    "surface",
)


def dial_owner(
    config_dir: Path,
    session_id: str,
    *,
    capabilities: list[str] | None = None,
    auth: dict[str, Any] | None = None,
    peer: dict[str, Any] | None = None,
    connect_timeout_s: float = 5.0,
    welcome_timeout_s: float = WELCOME_TIMEOUT_S,
    _record: Any = None,
) -> tuple[OwnerDial, dict[str, Any] | None]:
    """Authenticate against ``session_id``'s runtime on THIS device.

    Returns ``(dial, welcome_or_None)``. The welcome is the runtime's own
    identity answer — the session id it is serving RIGHT NOW — and the caller
    keeps it because a dial whose welcome names another session is a dial to a
    stranger (the same proof the stop ladder's ``_identity_by_record`` uses).

    Raises :class:`OwnerUnreachable` when there is no live runtime, the connect
    is refused, or nothing answers. There is deliberately no fallback that
    starts a runtime here: starting an owner is ``launch.engage_runtime``'s job
    (one implementation), and a dialer that spawned would be a second one.
    """
    if _record is None:
        from local_operator.mobile.attach_client import find_runtime_record

        record, _owner = find_runtime_record(Path(config_dir), session_id)
    else:
        record = _record
    if record is None:
        raise OwnerUnreachable(f"no runtime is running for session {session_id}")
    # NOTE, deliberately: a record whose ``session_id`` differs from the ask is
    # NOT refused here. ``find_runtime_record`` documents that rebind-race
    # fallback (the record is re-stamped every heartbeat) and hands the
    # arbitration to the welcome projection's identity check below — the same
    # check ``AttachClient.connect`` performs, so a stale match costs one
    # refused dial in one place rather than two disagreeing gates.
    try:
        sock = socket.create_connection(
            ("127.0.0.1", int(record.control_port)), timeout=connect_timeout_s
        )
    except OSError as exc:
        raise OwnerUnreachable(f"that session's runtime did not accept a connection: {exc}")

    auth_frame: dict[str, Any] = {
        "key": record.control_key,
        "client": "attach",
        # §2.2: locality is never taken from the frame that arrived. This
        # process dialled the runtime, and it dialled on behalf of another
        # device, so it — and only it — knows the truth.
        "locality": "remote",
    }
    if capabilities:
        auth_frame["capabilities"] = sorted(set(capabilities))
    if peer:
        # Who is on the other end of this dial, so the runtime's refusals can name
        # the machine to act on (§4.4). Additive: a runtime that does not know the
        # field drops it and keeps its own generic sentence.
        auth_frame["peer"] = dict(peer)
    for name in AUTH_FIELDS:
        if auth and name in auth and auth[name] is not None:
            auth_frame[name] = auth[name]
    dial = OwnerDial(sock=sock, session_id=session_id)
    try:
        dial.send(auth_frame)
    except OSError as exc:
        dial.close()
        raise OwnerUnreachable(f"the auth frame could not be sent: {exc}")
    # The runtime's first frame on an authenticated dial is its WELCOME
    # projection, and its ``data.session_id`` is read live off the handle the
    # process is serving RIGHT NOW — that is the whole identity proof, and it is
    # the same one the stop ladder uses (``control._confirmed_session_id``). An
    # absent id means a runtime too old to carry it, which is accepted exactly
    # as that function accepts it, never as a mismatch.
    welcome = dial.recv(welcome_timeout_s)
    if welcome is None:
        dial.close()
        raise OwnerUnreachable("that session's runtime is not answering its socket")
    raw_data = welcome.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    served = str(data.get("session_id") or "")
    if served and served != session_id:
        dial.close()
        raise OwnerUnreachable(
            f'the runtime on that port serves session "{served}", not "{session_id}"'
        )
    return dial, welcome

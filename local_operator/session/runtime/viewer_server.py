"""The viewer control endpoint: a socket that outlives every session swap.

WHAT THIS IS FOR. A notification click needs to tell an already-running TUI
"display session X and come to the front". The ops for that mostly exist
already — ``resume_session`` is a session-runtime op with a working handler
(``mobile/tui_handle.py``) — but in the state the operator actually runs in,
**there is no socket to send them to.**

Why not: ``RuntimeServer`` binds its listener and publishes its record in the
same step (``server.py::_serve``), and ``OperatorApp._mobile_adopted`` closes
the whole registrant the moment the app follows a ``RemoteSession``. Following a
remote session is the *normal* sidebar state, so a sidebar user's TUI is
listening on nothing. Keeping the RuntimeServer alive instead is not available:
its own comment records that a second registrant for one transcript corrupts
daemon routing, and this host runs a live phone daemon.

So this is a **transport**, not a protocol change. ``PROTOCOL_VERSION`` does not
move and no session frame changes shape; what is new is an endpoint scoped to
the *process* rather than to a *transcript*, which is a different object with a
different lifetime. That distinction is the whole design: a session endpoint
dies at every ``/resume``, and the thing a click must reach is precisely the
thing that survives one.

DELIBERATELY TWO OPS, AND NOT MORE. ``resume_session`` and ``focus_window``,
full stop. It is tempting to mirror RuntimeServer's surface "for symmetry" —
do not. The authorization story here is exactly one 0600 key under a 0700
directory, and that is sufficient *because* the surface is two ops that a user
could perform with two keystrokes at the same keyboard. Every op added is one
a later reader will assume was held to the same standard, so the narrowness is
load-bearing rather than an oversight to be tidied up.

LIFECYCLE IS THE RISK HERE, so it is stated plainly. This object owns a
listening socket across state changes that previously tore everything down.
Three properties are maintained and tested:

* ``start()`` is idempotent — a swap must never bind a second listener. The
  cautionary tale is the source leak at ``tui/app.py`` (25 open/close cycles
  leaked 50 sources); a leaked *listener* is worse than a leaked source because
  it also leaks a record advertising a port.
* ``close()`` is idempotent, safe from any thread, and unpublishes the record.
* A record whose process died is reaped by the reader (``scan_viewers`` checks
  pid liveness), so a ``kill -9`` costs one stale file and never a misroute.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, cast

from local_operator.session.runtime.viewers import (
    FOCUS_WINDOW_CAPABILITY,
    VIEWER_HEARTBEAT_INTERVAL_S,
    ViewerRecord,
    publish_viewer,
    unpublish_viewer,
)

logger = logging.getLogger(__name__)

#: How long a dial may take to present its key. Matches RuntimeServer's own
#: auth deadline: the client is on loopback and has nothing to compute.
_AUTH_TIMEOUT_S = 5.0

#: A viewer frame is a handful of fields. The cap exists so a wedged or hostile
#: writer cannot grow this process's memory; it is generous by two orders of
#: magnitude against any real frame.
_MAX_LINE_BYTES = 64 * 1024

#: Re-exported for the callers (and tests) that reach for it here, beside the
#: endpoint that advertises it. It is DEFINED in ``viewers`` because the client
#: reads it too, and a capability string spelled in two modules is a capability
#: that will eventually be spelled two ways.
__all__ = ["FOCUS_WINDOW_CAPABILITY", "ViewerHost", "ViewerServer"]


class ViewerHost(Protocol):
    """What the viewer endpoint needs from the application hosting it.

    Both methods are called ON the viewer's own loop (its own thread) and are
    expected to make whatever hop the host needs — for the TUI, Textual's
    ``call_from_thread``. Each returns a short human-readable receipt that
    becomes the ``ack`` detail, matching ``SessionHandle``'s convention so the
    two endpoints read the same way.
    """

    async def viewer_resume_session(self, session_id: str) -> str:
        """Display ``session_id`` in this viewer. Same act as the user pressing
        the sidebar row."""
        ...

    async def viewer_focus_window(self) -> str:
        """Bring this viewer's window to the front, best effort."""
        ...


class ViewerServer:
    """One per process. Construct, :meth:`start`, :meth:`close`.

    Threaded like ``RuntimeServer``: its own loop on its own thread, so a
    blocked application loop cannot stop a click from being answered, and a
    slow click cannot stall the application.
    """

    def __init__(self, host: ViewerHost, *, surface: str = "tui", root: Path | None = None) -> None:
        self._host = host
        self._root = root
        self._record = ViewerRecord(
            pid=os.getpid(),
            surface=surface,
            control_port=0,  # stamped when the listener binds
            control_key=secrets.token_hex(32),
            # Gated on the host actually implementing it, never assumed. A
            # headless or non-macOS host that cannot raise a window must not
            # advertise that it can.
            capabilities=(
                [FOCUS_WINDOW_CAPABILITY] if hasattr(host, "viewer_focus_window") else []
            ),
        )
        self._server: asyncio.AbstractServer | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._closed = threading.Event()
        #: Set once the listener is bound and the record published, so a caller
        #: that needs the record to exist (a test, or a diagnostic) can wait
        #: for it rather than sleeping.
        self.ready = threading.Event()

    # -- state the application pushes in -------------------------------------

    def note_session(self, session_id: str) -> None:
        """Record which session is on screen now. Called on every swap.

        Cheap by construction: one field and one staged write, with no
        enumeration of anything. The record deliberately holds only the CURRENT
        session rather than a list of displayable ones — see ``ViewerRecord``,
        where the reasoning is that an enumerated list is a cache of something
        that changes without this process knowing.

        **THIS RUNS ON TEXTUAL'S LOOP AND TOUCHES DISK.** ``_republish`` is
        ``mkstemp`` + ``write`` + ``chmod`` + ``os.replace``, measured at
        **0.808 ms per call** on this host. That is affordable only because of
        the cadence: the early return below makes a repeat swap free, and the
        other two callers are a focus gain and a 15 s heartbeat. A third caller
        on a poll or a per-frame hook would put a synchronous disk write on the
        event loop at UI cadence — the shape the sidebar-lag investigation
        traced its stalls back to — and must hop to a thread instead.
        """
        if self._record.current_session == session_id:
            return
        self._record.current_session = session_id
        self._republish()

    def note_focused(self, focused: bool) -> None:
        """Record that this window just gained OS focus.

        Only the gaining edge is stamped: ``focused_at`` answers "when was the
        user last HERE", which is the tiebreak when several viewers could take
        a session. A blur carries no routing information, so it writes nothing
        — and not writing is what keeps an alt-tab from producing a disk write
        per keystroke.
        """
        if not focused:
            return
        self._record.focused_at = time.time()
        self._republish()

    def _republish(self) -> None:
        if self._closed.is_set() or not self.ready.is_set():
            return
        try:
            publish_viewer(self._record, self._root)
        except OSError:
            # A full or read-only disk must not take down the TUI over a
            # discovery record. The next heartbeat retries.
            logger.debug("viewer record refresh failed", exc_info=True)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Bind, publish, and begin heartbeating on a dedicated thread.

        IDEMPOTENT, and that is load-bearing rather than defensive: this is
        started from the app's session-adoption path, which runs again on every
        ``/resume``. A second bind there would leak a listener and a record per
        swap — the leak class documented in ``tui/app.py``, made worse by also
        advertising the leaked port.
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="lop-viewer-endpoint", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception:  # noqa: BLE001 — a dead endpoint must not kill the TUI
            logger.warning("viewer endpoint loop died", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                loop.close()
            self.ready.set()  # unblock anyone waiting on a start that failed

    async def _serve(self) -> None:
        # Port 0: the OS picks and the record carries the number. Loopback
        # only — the key authorizes, but binding to localhost means an
        # off-host attacker never reaches the auth check at all.
        self._server = await asyncio.start_server(
            self._on_connection, host="127.0.0.1", port=0, limit=_MAX_LINE_BYTES
        )
        self._record.control_port = self._server.sockets[0].getsockname()[1]
        try:
            publish_viewer(self._record, self._root)
        except OSError:
            # Nothing can route to us, but the TUI is otherwise fine. Serve
            # anyway: the heartbeat retries the publish, and a transient ENOSPC
            # should not permanently cost the user click-through.
            logger.debug("viewer record publish failed", exc_info=True)
        self.ready.set()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            while not self._closed.is_set():
                await asyncio.sleep(0.2)
        finally:
            await self._shutdown()

    async def _heartbeat_loop(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(VIEWER_HEARTBEAT_INTERVAL_S)
            if self._closed.is_set():
                return
            self._republish()

    async def _shutdown(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        unpublish_viewer(self._record.pid, self._root)

    def close(self) -> None:
        """Stop serving and remove the record. Idempotent, safe from any thread.

        The record is removed HERE as well as in ``_shutdown`` because a loop
        that never started (a failed bind) has no ``_shutdown`` to run, and a
        record left behind advertises a port nothing is listening on — which
        costs a later click its bounded dial timeout before it falls back.
        """
        if self._closed.is_set():
            return
        self._closed.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # The serve loop polls `_closed` every 200 ms and runs `_shutdown`
            # itself. Join briefly so a clean exit really has released the
            # port; a slow teardown must not hold up the app's own exit, and
            # the daemon thread cannot outlive the process regardless.
            thread.join(timeout=2.0)
        unpublish_viewer(self._record.pid, self._root)

    # -- the wire -------------------------------------------------------------

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """One control conversation: authenticate, then serve ops until EOF.

        Auth is the first frame and anything wrong closes WITHOUT a reply, for
        the reason RuntimeServer states: a port that answers bad keys with
        errors is an oracle, however small.
        """
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=_AUTH_TIMEOUT_S)
            frame = json.loads(line.decode("utf-8", "replace"))
        except (TimeoutError, ValueError, UnicodeDecodeError, OSError):
            writer.close()
            return
        key = frame.get("key", "") if isinstance(frame, dict) else ""
        if not isinstance(key, str) or not hmac.compare_digest(key, self._record.control_key):
            logger.warning("viewer control: rejected a bad key")
            writer.close()
            return
        try:
            while not self._closed.is_set():
                try:
                    line = await reader.readline()
                except ValueError:
                    # An over-limit line: `start_server(..., limit=...)` makes
                    # `readline` raise `LimitOverrunError` (a `ValueError`)
                    # WITHOUT consuming the buffer, so the same read would raise
                    # forever. Uncaught it escaped `client_connected_cb`, skipped
                    # the reply and logged an asyncio traceback while the client
                    # waited out its ack timeout. `viewer_client._read_reply`
                    # hand-rolls its framing to dodge this exact defect on its
                    # side; closing the conversation is the server's equivalent —
                    # the peer is authenticated, so this is a bug in a peer
                    # rather than an attack, and it falls back correctly.
                    logger.debug("viewer control: oversized frame, closing the conversation")
                    return
                if not line:
                    return
                try:
                    request = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue  # noise on an authenticated loopback socket
                if not isinstance(request, dict):
                    continue
                await self._dispatch(request, writer)
        except (OSError, ConnectionError):
            return
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def _dispatch(self, frame: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        op = str(frame.get("op") or "")
        req = frame.get("req")
        try:
            detail = await self._apply(op, frame)
            reply: dict[str, Any] = {"op": "ack", "req": req, "detail": detail}
        except Exception as exc:  # noqa: BLE001 — the error IS the answer
            # An UNKNOWN op lands here too, and that is the interoperability
            # story: a newer client asking an older viewer for something it
            # does not have reads the error and degrades, exactly as an old
            # daemon does with an unknown session op.
            reply = {"op": "error", "req": req, "detail": str(exc)}
        try:
            writer.write(json.dumps(reply).encode() + b"\n")
            await writer.drain()
        except (OSError, ConnectionError):
            return

    async def _apply(self, op: str, frame: dict[str, Any]) -> str:
        if op == "resume_session":
            session_id = str(frame.get("session_id", ""))
            if not session_id:
                raise ValueError("resume_session needs a session_id")
            handler: Callable[[str], Awaitable[str]] = self._host.viewer_resume_session
            return await handler(session_id)
        if op == "focus_window":
            # getattr-probed rather than called directly: the capability is
            # optional, and a host that never implemented it must answer the
            # unknown-op error rather than raise AttributeError. Same shape the
            # session runtime uses for its own optional ops.
            focus = getattr(self._host, "viewer_focus_window", None)
            if not callable(focus):
                raise ValueError("this viewer cannot activate its window")
            focus_call = cast(Callable[[], Awaitable[str]], focus)
            return await focus_call()
        raise ValueError(f"unknown viewer op: {op or '(none)'}")

    # -- introspection for tests and diagnostics ------------------------------

    @property
    def record(self) -> ViewerRecord:
        return self._record

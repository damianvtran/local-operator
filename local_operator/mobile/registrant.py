"""The registrant: make a live session reachable by the mobile daemon.

Every interactive ``lop`` process (the TUI, and ``exec`` when it runs
attached) hosts one of these. It does three things:

1. **Publishes** the discovery record (see :mod:`.registry`) and rewrites it
   on a heartbeat — cheap enough that a machine with no daemon installed
   pays nothing but one small file write every 15 seconds.
2. **Listens** on a random loopback port for the daemon's control
   connection, authenticating it with the record's key (constant-time
   compare — the key is the whole credential).
3. **Bridges**: folds the session's event stream into the phone projection
   (:class:`.projection.ProjectionFold`) and pushes a repaint on change;
   applies the daemon's requests to the session through a host-provided
   :class:`SessionHandle`.

The handle indirection exists because the two registrant hosts drive their
session differently: the TUI must route mutations through Textual's message
pump and thread (``call_from_thread``), while an exec-mode host can call the
session directly. The registrant speaks to the handle, never to Textual.

Threading: the control socket server runs on its own thread with its own
event loop — the TUI's loop must never block on a phone, and the daemon's
requests (model switches, aborts) must land even while the TUI is mid-repaint.
All session mutations funnel through the handle, whose contract is "callable
from the registrant's loop, serialized by the implementor".
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import threading
from typing import Any, Callable, Protocol

from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.registry import RecordPublisher
from local_operator.mobile.types import (
    HEARTBEAT_INTERVAL_S,
    PendingRequest,
    SessionProjection,
    SessionRecord,
)

logger = logging.getLogger(__name__)

#: A prompt payload past 1 MB is a bug, not a prompt — the line limit the
#: control socket reader enforces.
_MAX_LINE_BYTES = 1 << 20


class SessionHandle(Protocol):
    """What the registrant needs from its host application.

    Every method is awaited on the REGISTRANT'S loop (its own thread); the
    implementor guarantees any hop the session needs (for the TUI: Textual's
    ``call_from_thread``; for an owned session: ``run_coroutine_threadsafe``
    back to the daemon loop). Methods return a short human-readable receipt
    that becomes the ``ack`` detail — the same line the TUI would print as a
    notice.
    """

    @property
    def session_projection_seed(self) -> SessionProjection:
        """The projection skeleton: identity fields the registrant folds onto."""
        ...

    def subscribe(self, on_projection: Callable[[], None]) -> Callable[[], None]:
        """Feed the fold from the host's event stream; call ``on_projection``
        (thread-safe) whenever the projection changed and should be pushed.
        Returns an unsubscribe callable."""
        ...

    async def prompt(self, text: str) -> str: ...
    async def steer(self, text: str) -> str: ...
    async def abort(self) -> str: ...
    async def set_model(self, provider: str, model_id: str) -> str: ...
    async def set_effort(self, effort: str) -> str: ...
    async def slash(self, command: str, args: str) -> str: ...
    async def new_conversation(self) -> str: ...
    async def resume_session(self, session_id: str) -> str: ...
    async def approval_answer(self, request_id: str, approved: bool, remember: bool) -> str: ...
    async def ask_answer(self, request_id: str, value: str) -> str: ...

    async def refresh(self) -> None:
        """Re-read session state into the projection (post-resume, rename,
        model change): the registrant pushes whatever changed."""
        ...


class Registrant:
    """One per interactive process. Construct, ``start()``, ``close()``."""

    def __init__(self, handle: SessionHandle, *, kind: str = "tui") -> None:
        self._handle = handle
        seed = handle.session_projection_seed
        seed.kind = kind
        self._fold = ProjectionFold(seed)
        self._record = SessionRecord(
            pid=os.getpid(),
            kind=kind,  # type: ignore[arg-type]
            session_id=seed.session_id,
            conversation_name=seed.conversation_name,
            cwd=seed.cwd,
            model_label=seed.model_label,
            control_port=0,  # stamped when the listener binds
            control_key=secrets.token_hex(32),
        )
        self._publisher: RecordPublisher | None = None
        self._server: asyncio.AbstractServer | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._closed = threading.Event()
        self._push_scheduled = False
        self._send_lock: asyncio.Lock | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Bind the listener, publish the record, start the heartbeat and the
        event feed — on a dedicated thread with its own loop. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="lop-mobile-registrant", daemon=True)
        self._thread.start()

    async def start_in_process(self) -> None:
        """The same startup as :meth:`start` but on the CALLER'S running
        loop — for hosts whose session already lives on that loop (the
        mobile child process), where a second loop would force every handle
        call through a cross-thread hop for no benefit."""
        if self._server is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._send_lock = asyncio.Lock()
        await self._serve()

    def close(self) -> None:
        """Unpublish and shut down. Safe from any thread, safe twice. In
        in-process mode prefer :meth:`aclose` — it awaits the cleanup instead
        of posting it onto a loop the caller may be about to tear down."""
        if self._closed.is_set():
            return
        self._closed.set()
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                logger.debug("registrant unsubscribe failed", exc_info=True)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._shutdown)
        if self._publisher is not None:
            self._publisher.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    async def aclose(self) -> None:
        """In-process mode shutdown, awaited on the owning loop: cancel the
        heartbeat, close the server and the daemon connection, unpublish.
        Posting this to a loop that is about to close (the child's amain
        returns right after) is how heartbeats outlive their process — so the
        child awaits it instead."""
        if self._closed.is_set():
            return
        self._closed.set()
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("registrant unsubscribe failed", exc_info=True)
        self._shutdown()
        if self._server is not None:
            await self._server.wait_closed()
        if self._publisher is not None:
            self._publisher.close()

    # -- the registrant's own loop -------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._send_lock = asyncio.Lock()
        try:
            loop.run_until_complete(self._serve())
        except Exception:  # noqa: BLE001 — a dead registrant must not kill the host
            logger.warning("mobile registrant loop died", exc_info=True)
        finally:
            loop.close()

    async def _serve(self) -> None:
        # Port 0: the OS picks; the record carries the number. Binding
        # loopback only is the security invariant of the whole design.
        self._server = await asyncio.start_server(
            self._on_connection, host="127.0.0.1", port=0, limit=_MAX_LINE_BYTES
        )
        port = self._server.sockets[0].getsockname()[1]
        self._record.control_port = port
        self._publisher = RecordPublisher(self._record)
        self._unsubscribe = self._handle.subscribe(self._schedule_push)
        heartbeat = asyncio.ensure_future(self._heartbeat_loop())
        self._heartbeat_task = heartbeat
        if self._thread is not None:
            # Thread mode owns the loop: park here until closed. In-process
            # mode returns so the caller's loop keeps running its own work —
            # the caller then owns cancelling the heartbeat (close() does).
            await self._closed_wait()
            heartbeat.cancel()
            self._server.close()
            await self._server.wait_closed()

    async def _closed_wait(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(0.2)

    def _shutdown(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        if self._server is not None:
            self._server.close()
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def _heartbeat_loop(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            if self._closed.is_set():
                return
            try:
                seed = self._handle.session_projection_seed
                if self._publisher is not None:
                    self._publisher.heartbeat(
                        session_id=seed.session_id,
                        conversation_name=seed.conversation_name,
                        model_label=seed.model_label,
                        cwd=seed.cwd,
                    )
            except Exception:  # noqa: BLE001 — a missed heartbeat is self-healing
                logger.debug("registrant heartbeat failed", exc_info=True)

    # -- connections -----------------------------------------------------------

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """One daemon connection. Auth is the first frame: ``{"key": ...}``
        within a short deadline, constant-time compared. Anything else closes
        without a reply — an open port that answers wrong keys with errors is
        an oracle, however small.

        One control connection at a time — the daemon is the only legitimate
        client, and a second one racing it would interleave repaints on the
        same socket. A new dial REPLACES the old, which is also the reconnect
        story: the daemon re-dials after any drop and the stale socket is
        evicted."""
        peer = writer.get_extra_info("peername")
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            frame = json.loads(line.decode("utf-8", "replace"))
        except (TimeoutError, ValueError, UnicodeDecodeError):
            writer.close()
            return
        key = frame.get("key", "")
        if not isinstance(key, str) or not hmac.compare_digest(key, self._record.control_key):
            logger.warning("mobile control: rejected bad key from %s", peer)
            writer.close()
            return

        # Authenticated: this becomes THE daemon connection. A prior one is
        # evicted — reconnect after a drop is the normal path here.
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
        self._writer = writer
        await self._push()  # the welcome: a full projection, unprompted
        try:
            while not self._closed.is_set():
                line = await reader.readline()
                if not line:
                    return  # daemon hung up
                try:
                    frame = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                await self._on_request(frame)
        except (ConnectionResetError, BrokenPipeError):
            return
        finally:
            if self._writer is writer:
                self._writer = None
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def _on_request(self, frame: dict[str, Any]) -> None:
        op = str(frame.get("op") or "")
        req = frame.get("req")
        try:
            detail = await self._dispatch(op, frame)
            await self._send({"op": "ack", "req": req, "detail": detail})
            # Mutations change the projection; push what the phone should see.
            await self._handle.refresh()
            await self._push()
        except Exception as exc:  # noqa: BLE001 — the error IS the reply
            await self._send({"op": "error", "req": req, "message": str(exc)[:400]})
            await self._push()

    async def _dispatch(self, op: str, frame: dict[str, Any]) -> str:
        h = self._handle
        if op == "ping":
            return "pong"
        if op == "snapshot":
            await self._push()
            return "snapshot sent"
        if op == "prompt":
            return await h.prompt(str(frame.get("text", "")))
        if op == "steer":
            return await h.steer(str(frame.get("text", "")))
        if op == "abort":
            return await h.abort()
        if op == "set_model":
            return await h.set_model(str(frame.get("provider", "")), str(frame.get("model_id", "")))
        if op == "set_effort":
            return await h.set_effort(str(frame.get("effort", "")))
        if op == "slash":
            return await h.slash(str(frame.get("command", "")), str(frame.get("args", "")))
        if op == "new_conversation":
            return await h.new_conversation()
        if op == "resume_session":
            return await h.resume_session(str(frame.get("session_id", "")))
        if op == "approval_answer":
            return await h.approval_answer(
                str(frame.get("request_id", "")),
                bool(frame.get("approved")),
                bool(frame.get("remember")),
            )
        if op == "ask_answer":
            return await h.ask_answer(str(frame.get("request_id", "")), str(frame.get("value", "")))
        raise ValueError(f"unknown op: {op!r}")

    # -- pushes ----------------------------------------------------------------

    def _schedule_push(self) -> None:
        """Called by the host on projection change, from ANY thread. Coalesce
        bursts (a streaming assistant row changes 30×/s) into one repaint per
        registrant-loop tick — pushes are snapshots, so intermediate states
        carry no information."""
        if self._loop is None or self._closed.is_set():
            return
        try:
            self._loop.call_soon_threadsafe(self._push_soon)
        except RuntimeError:  # loop closing
            pass

    def _push_soon(self) -> None:
        if self._push_scheduled:
            return
        self._push_scheduled = True
        asyncio.ensure_future(self._push_later())

    async def _push_later(self) -> None:
        # One sleep(0) lets the current event batch fold before the snapshot.
        await asyncio.sleep(0.05)
        self._push_scheduled = False
        await self._push()

    async def _push(self) -> None:
        await self._send({"op": "projection", "data": self._fold.projection.to_json()})

    async def _send(self, frame: dict[str, Any]) -> None:
        if self._writer is None or self._send_lock is None:
            return
        async with self._send_lock:
            try:
                self._writer.write(json.dumps(frame).encode() + b"\n")
                await self._writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                self._writer = None

    # -- host-facing helpers ----------------------------------------------------

    @property
    def fold(self) -> ProjectionFold:
        return self._fold

    def set_pending(self, pending: PendingRequest | None) -> None:
        self._fold.set_pending(pending)
        self._schedule_push()

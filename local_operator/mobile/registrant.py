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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from local_operator.harness.types import ImageContent

from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.registry import RecordPublisher
from local_operator.mobile.types import (
    ATTACH_MAX_CLIENTS,
    HEARTBEAT_INTERVAL_S,
    ClientKind,
    PendingRequest,
    SessionProjection,
    SessionRecord,
)

logger = logging.getLogger(__name__)


def image_blocks(images: list[dict[str, str]] | None) -> list["ImageContent"]:
    """Decode the wire's [{data_b64, mime_type}] into ImageContent blocks.

    Bad entries are dropped, not fatal: a paste that half-decoded should cost
    that one image, not the whole prompt. Empty input yields an empty list,
    which ``_submit_prompt`` treats exactly like no images. This is part of
    the mobile contract — both handles use it.
    """
    if not images:
        return []
    from local_operator.harness.types import ImageContent

    out: list[ImageContent] = []
    for item in images:
        if not isinstance(item, dict):
            logger.debug("mobile image dropped: not a dict (%r)", type(item).__name__)
            continue
        data = item.get("data_b64") or item.get("data") or ""
        mime = item.get("mime_type") or "image/png"
        if not data:
            logger.debug("mobile image dropped: no data_b64/data")
            continue
        out.append(ImageContent(data=data, mime_type=mime))
    return out


#: A prompt payload past 1 MB is a bug, not a prompt — the line limit the
#: control socket reader enforces.
_MAX_LINE_BYTES = 1 << 20
# A projection is replaceable state. If a peer cannot accept one within this
# bound, dropping that peer is safer than blocking authority-bearing ACKs for
# every healthy front end.
_SEND_TIMEOUT_S = 1.0


@dataclass
class _ClientConn:
    """One authenticated control connection in the registrant's registry.

    Multiplexing is already half-there (frames carry caller-chosen ``req``
    ids), so multi-front-end needs N concurrent connections on the ONE
    socket rather than a second protocol: the daemon plus up to
    ``ATTACH_MAX_CLIENTS`` attach terminals. ``last_seen`` is the LRU clock
    for attach eviction — stamped on every request so the least-recently-ACTIVE
    follower is the one dropped when the cap is hit.
    """

    writer: asyncio.StreamWriter
    kind: ClientKind
    last_seen: float = field(default_factory=time.monotonic)
    # Frames on one TCP stream must stay ordered, while unrelated streams must
    # never queue behind its backpressure.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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

    async def prompt(self, text: str, images: list[dict[str, str]] | None = None) -> str: ...
    async def steer(self, text: str, images: list[dict[str, str]] | None = None) -> str: ...
    async def abort(self) -> str: ...
    async def set_model(self, provider: str, model_id: str) -> str: ...
    async def set_effort(self, effort: str) -> str: ...
    async def slash(self, command: str, args: str) -> str: ...
    async def new_conversation(self) -> str: ...
    async def resume_session(self, session_id: str) -> str: ...
    async def approval_answer(self, request_id: str, approved: bool, remember: bool) -> str: ...
    async def ask_answer(  # noqa: E301
        self, request_id: str, value: str, question_index: int | None = None
    ) -> str: ...

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
        # N authenticated connections keyed by id(writer): one daemon (a new
        # daemon dial evicts the old — that IS its reconnect story) plus up to
        # ATTACH_MAX_CLIENTS attach clients. A single _writer could not carry
        # the phone bridge and a follower terminal at once.
        self._clients: dict[int, _ClientConn] = {}
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._closed = threading.Event()
        self._push_scheduled = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        # -- front-end accounting (the child reaper's inputs, §4) --------------
        # Phone SSE watchers, fed by the daemon's watch/unwatch pushes. Floored
        # at 0: a daemon restart redials without unwatching, and a counter that
        # went negative would read as "watchers" to an == 0 check forever.
        self.phone_watchers: int = 0
        # Latched True on the first watch/unwatch EVER received. Until then
        # watcher count is UNKNOWN (an old daemon never sends the ops), and
        # unknown must be treated as "present" — a new child under an old
        # daemon must not reap a session a phone is actively watching. The
        # latch never resets: once a watch-capable daemon has spoken, absence
        # of the op means absence of watchers.
        self.watch_supported: bool = False

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
        for conn in list(self._clients.values()):
            try:
                conn.writer.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()

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
        """One control connection. Auth is the first frame: ``{"key": ...}``
        within a short deadline, constant-time compared. Anything else closes
        without a reply — an open port that answers wrong keys with errors is
        an oracle, however small.

        Protocol v2 carries N connections: one ``daemon`` plus up to
        ``ATTACH_MAX_CLIENTS`` ``attach`` followers. A new daemon dial still
        REPLACES the old one (that is its reconnect story, preserved from the
        single-writer era); a further attach dial past the cap evicts the
        least-recently-seen attach client. Connection close is detected by
        the reader loop's ``finally``; the cap only guards leaked-but-open
        sockets liveness detection cannot see.
        """
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
        # Absent client field means daemon: an OLD daemon dialing a NEW
        # registrant must land on the class it always had, or every rolling
        # upgrade would demote the phone bridge to a follower.
        raw_kind = frame.get("client", "daemon")
        kind: ClientKind = "attach" if raw_kind == "attach" else "daemon"

        if kind == "daemon":
            # At most ONE daemon connection — a new dial evicts the old, which
            # is also the reconnect path after a daemon restart.
            for other in [
                c for c in self._clients.values() if c.kind == "daemon" and c.writer is not writer
            ]:
                self._drop_client(other)
        else:
            # Attach cap with LRU eviction: the least-recently-seen follower
            # goes. Sending on the evicted socket first (a goodbye) is not
            # worth the failure modes — its reader loop is still alive and
            # will observe the close as EOF, which is the attach screen's
            # owner-death signal minus a corpse.
            attaches = [c for c in self._clients.values() if c.kind == "attach"]
            if len(attaches) >= ATTACH_MAX_CLIENTS:
                victim = min(attaches, key=lambda c: c.last_seen)
                logger.info("mobile control: evicting attach client %s (cap)", peer)
                self._drop_client(victim)

        conn = _ClientConn(writer=writer, kind=kind)
        self._clients[id(writer)] = conn
        await self._push_to(conn)  # the welcome: a full projection, unprompted
        try:
            while not self._closed.is_set():
                line = await reader.readline()
                if not line:
                    return  # client hung up
                try:
                    frame = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                conn.last_seen = time.monotonic()
                await self._on_request(frame, conn)
        except (ConnectionResetError, BrokenPipeError):
            return
        finally:
            self._drop_client(conn)

    def _drop_client(self, conn: _ClientConn) -> None:
        """Remove one connection from the registry and close its socket.

        The ONLY removal path: reader-loop exit, shutdown, daemon eviction,
        and attach-cap eviction all funnel here so the registry can never
        retain an entry whose socket is closed (the reaper counts them)."""
        self._clients.pop(id(conn.writer), None)
        try:
            conn.writer.close()
        except Exception:  # noqa: BLE001
            pass

    def attach_clients(self) -> int:
        """How many attach (follower terminal) connections are live.

        The child reaper's front-end count: an attached TUI is a front end
        exactly like a phone, so it must hold the child in ACTIVE."""
        return sum(1 for c in self._clients.values() if c.kind == "attach")

    async def _on_request(self, frame: dict[str, Any], conn: _ClientConn) -> None:
        op = str(frame.get("op") or "")
        req = frame.get("req")
        try:
            # Attach clients are followers: rebinding the owner's conversation
            # from a follower terminal surprises the user AT THAT TERMINAL's
            # owner. The error frame is the reply — the attach screen surfaces
            # it like any other rejected op. The daemon keeps both ops (the
            # phone's resume button rides them).
            if conn.kind == "attach" and op in ("new_conversation", "resume_session"):
                raise ValueError(
                    "attached front ends cannot rebind the session; detach and /resume instead"
                )
            if op in ("watch", "unwatch"):
                # The reaper's phone-watcher signal (§2.8). watch_supported
                # latches on the FIRST op seen so a mixed-version child never
                # mistakes silence for zero watchers.
                self.watch_supported = True
                if op == "watch":
                    self.phone_watchers += 1
                else:
                    self.phone_watchers = max(0, self.phone_watchers - 1)
                detail = f"watchers: {self.phone_watchers}"
            else:
                detail = await self._dispatch(op, frame)
            await self._send_to(conn, {"op": "ack", "req": req, "detail": detail})
            # Mutations change the projection; push what every front end
            # should see.
            if op not in ("watch", "unwatch"):
                await self._handle.refresh()
                await self._push()
        except Exception as exc:  # noqa: BLE001 — the error IS the reply
            await self._send_to(conn, {"op": "error", "req": req, "message": str(exc)[:400]})
            await self._push()

    async def _dispatch(self, op: str, frame: dict[str, Any]) -> str:
        h = self._handle
        if op == "ping":
            return "pong"
        if op == "snapshot":
            await self._push()
            return "snapshot sent"
        if op == "prompt":
            images = frame.get("images")
            return await h.prompt(
                str(frame.get("text", "")),
                images=images if isinstance(images, list) else None,
            )
        if op == "steer":
            images = frame.get("images")
            return await h.steer(
                str(frame.get("text", "")),
                images=images if isinstance(images, list) else None,
            )
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
            # ``question_index`` is the question the phone was DISPLAYING when
            # the user tapped (U8 guard): the handle rejects the answer if the
            # picker has since advanced past it. Optional — an older client that
            # omits it falls back to answering the current question, the
            # pre-guard behaviour.
            raw_index = frame.get("question_index")
            question_index = int(raw_index) if isinstance(raw_index, (int, float)) else None
            return await h.ask_answer(
                str(frame.get("request_id", "")),
                str(frame.get("value", "")),
                question_index=question_index,
            )
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
        """Broadcast a projection repaint to every live connection.

        Projections are snapshots (no deltas), so every front end wants each
        one — the daemon fans it to the phone, each attach terminal renders
        it directly. Sending is per-connection: one dead socket must not
        block or corrupt the others' frames."""
        await self._broadcast({"op": "projection", "data": self._fold.projection.to_json()})

    async def _push_to(self, conn: _ClientConn) -> None:
        """The welcome form of a push: one full projection to one connection."""
        await self._send_to(conn, {"op": "projection", "data": self._fold.projection.to_json()})

    async def _broadcast(self, frame: dict[str, Any]) -> None:
        # Copy the registry: a send failure drops its own entry, and mutating
        # the dict mid-iteration is exactly the failure being handled.
        await asyncio.gather(*(self._send_to(conn, frame) for conn in list(self._clients.values())))

    async def _send_to(self, conn: _ClientConn, frame: dict[str, Any]) -> None:
        """One frame to one connection. A failed send drops ONLY that client
        from the registry (never retried — the reader loop will observe the
        close and its finally is a no-op second removal)."""
        async with conn.send_lock:
            try:
                conn.writer.write(json.dumps(frame).encode() + b"\n")
                await asyncio.wait_for(conn.writer.drain(), timeout=_SEND_TIMEOUT_S)
            except (TimeoutError, ConnectionResetError, BrokenPipeError, OSError):
                self._drop_client(conn)

    async def _send(self, frame: dict[str, Any]) -> None:
        """Broadcast alias kept for the pre-v2 call shape (tests, hosts that
        grabbed a reference before the bump)."""
        await self._broadcast(frame)

    # -- host-facing helpers ----------------------------------------------------

    @property
    def fold(self) -> ProjectionFold:
        return self._fold

    def set_pending(self, pending: PendingRequest | None) -> None:
        self._fold.set_pending(pending)
        self._schedule_push()

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
import copy
import hmac
import inspect
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, cast

if TYPE_CHECKING:
    from local_operator.harness.types import ImageContent

from local_operator.mobile.live_turn import LiveTurnTracker
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
# Raw events are lossless only while a follower keeps pace. One bounded FIFO per
# event client prevents a non-reader from retaining an unbounded stream before
# its active drain reaches the timeout; overflow drops that client so it can
# reconnect through history + attach_sync rather than observe a gapped stream.
_EVENT_QUEUE_MAX = 64


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
    # v4: this attach client asked for the raw AgentEvent relay in its auth
    # frame. Daemon connections never set it; a v3 attach client that omitted
    # the flag keeps projection-only behaviour.
    wants_events: bool = False
    # Flipped only AFTER the welcome projection and the attach_sync seed have
    # been queued on this connection's ordered stream. Event fan-out snapshots
    # recipients synchronously against the tracker fold, so this flag is what
    # guarantees a joining client never sees an event the seed already covers
    # (welcome → seed → live events, gapless and duplicate-free).
    events_ready: bool = False
    event_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_EVENT_QUEUE_MAX)
    )
    # Exactly one writer drains the queue, so delivery stays ordered without a
    # task per event. Held for shutdown and slow-client eviction.
    event_writer_task: asyncio.Task[None] | None = None


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

    async def prompt(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        command_id: str | None = None,
    ) -> str: ...
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

    # -- v4 optional capabilities (probed with getattr, never required) -------
    # subscribe_events(on_event) -> unsubscribe: feed the host session's raw
    #   AgentEvent stream, serialized (``model_dump(mode="json")``) on the
    #   host's own loop, to ``on_event`` (thread-safe). The registrant relays
    #   the dicts to event-subscribed attach clients. Optional so old hosts
    #   and reduced test handles keep working — without it, attach clients
    #   simply get v3 projection-only behaviour.
    # recall_steer(command_id) -> str: unsend the queued steering message the
    #   follower submitted under ``command_id``; raises when it already
    #   drained. Optional for the same reason.


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
        # v4 event relay: the live-turn tracker folds the SAME serialized
        # frames the relay fans out, on the registrant loop, so a joining
        # client's seed is exactly consistent with the frames already sent.
        self._live_turn = LiveTurnTracker()
        self._unsubscribe_events: Callable[[], None] | None = None
        # Strong references to the one event writer per subscribed client. A
        # bare create_task can be collected mid-flight, which drops frames.
        self._event_sends: set[asyncio.Task[None]] = set()
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
        # The delayed repaint must be owned like the heartbeat. A bare task can
        # still be sleeping when close tears down the registrant loop, producing
        # an orphan warning and proving teardown returned before its work ended.
        self._push_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        # Shutdown is represented by one owner-loop task so synchronous close,
        # awaited close, and the thread runner can converge without cancelling
        # loop-owned objects from whichever thread happened to request teardown.
        self._shutdown_task: asyncio.Task[None] | None = None
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
        """Unpublish and shut down. Safe from any thread, safe twice.

        Thread-hosted registrants are joined before returning. In-process hosts
        should prefer :meth:`aclose`; when ``close`` is called on their owning
        loop it schedules cleanup instead of deadlocking that loop waiting for
        itself.
        """
        self._request_close()
        loop = self._loop
        if self._thread is not None:
            # The thread runner observes the close latch and performs teardown
            # itself. Joining it is both simpler and race-free when close lands
            # while the thread is still publishing its loop reference.
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=2.0)
            return
        if loop is None or loop.is_closed():
            return
        if self._on_owner_loop():
            self._ensure_shutdown_task()
            return
        shutdown = self._shutdown_on_loop()
        try:
            asyncio.run_coroutine_threadsafe(shutdown, loop).result(timeout=2.0)
        except RuntimeError:
            # run_coroutine_threadsafe does not consume the coroutine when the
            # loop wins the close race, so close it to avoid a false leak warning.
            shutdown.close()
            logger.debug("registrant loop exited during shutdown", exc_info=True)
        except TimeoutError:
            logger.debug("registrant shutdown did not finish before timeout", exc_info=True)

    async def aclose(self) -> None:
        """Await complete teardown on the owning loop.

        Repeated calls still join the original cleanup task; merely observing
        the cross-thread closed flag is not proof that loop-owned work ended.
        """
        self._request_close()
        if not self._on_owner_loop():
            raise RuntimeError("Registrant.aclose() must run on its owning event loop")
        await self._shutdown_on_loop()

    def _request_close(self) -> None:
        """Latch closure and detach the host feed exactly once."""
        if self._closed.is_set():
            return
        self._closed.set()
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                logger.debug("registrant unsubscribe failed", exc_info=True)
            self._unsubscribe = None
        if self._unsubscribe_events is not None:
            try:
                self._unsubscribe_events()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                logger.debug("registrant event unsubscribe failed", exc_info=True)
            self._unsubscribe_events = None

    def _on_owner_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

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
        # v4: hosts that can serialize their event stream feed the relay.
        # Probed, not required — a handle without the capability leaves attach
        # clients on v3 projection-only behaviour, never broken.
        subscribe_events = getattr(self._handle, "subscribe_events", None)
        if callable(subscribe_events):
            try:
                subscribe = cast(
                    Callable[[Callable[[dict[str, Any]], None]], Callable[[], None]],
                    subscribe_events,
                )
                self._unsubscribe_events = subscribe(self._relay_event)
            except Exception:  # noqa: BLE001 — relay is additive, never a gate
                logger.debug("event relay subscribe failed", exc_info=True)
        heartbeat = asyncio.ensure_future(self._heartbeat_loop())
        self._heartbeat_task = heartbeat
        if self._thread is not None:
            # Thread mode owns the loop: park here until closed. In-process
            # mode returns so the caller's loop keeps running its own work —
            # the caller then owns cancelling the heartbeat (close() does).
            await self._closed_wait()
            await self._shutdown_on_loop()

    async def _closed_wait(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(0.2)

    def _ensure_shutdown_task(self) -> asyncio.Task[None]:
        """Create the one teardown task; called only on the registrant loop."""
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(self._shutdown_impl())
            self._shutdown_task = task
        return task

    async def _shutdown_on_loop(self) -> None:
        """Join idempotent teardown from a coroutine on the registrant loop."""
        await asyncio.shield(self._ensure_shutdown_task())

    async def _shutdown_impl(self) -> None:
        """Cancel and join every object owned by the registrant event loop."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        if self._push_task is not None:
            self._push_task.cancel()
        for task in list(self._event_sends):
            task.cancel()
        self._event_sends.clear()
        if self._server is not None:
            self._server.close()
        clients = list(self._clients.values())
        for conn in clients:
            self._drop_client(conn)
        await self._await_push_shutdown()
        heartbeat = self._heartbeat_task
        if heartbeat is not None:
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._heartbeat_task = None
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None
        if clients:
            await asyncio.gather(
                *(conn.writer.wait_closed() for conn in clients), return_exceptions=True
            )
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None

    async def _await_push_shutdown(self) -> None:
        """Join the coalesced repaint before its owning loop can disappear."""
        task = self._push_task
        if task is None:
            return
        await asyncio.gather(task, return_exceptions=True)
        if self._push_task is task:
            self._push_task = None
        self._push_scheduled = False

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
        # v4: only attach clients may subscribe to the raw event relay. The
        # daemon's projection path must stay byte-identical, so a daemon auth
        # carrying the flag (there is none today) is deliberately ignored.
        wants_events = kind == "attach" and bool(frame.get("events"))

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

        conn = _ClientConn(writer=writer, kind=kind, wants_events=wants_events)
        self._clients[id(writer)] = conn
        await self._push_to(conn)  # the welcome: a full projection, unprompted
        if conn.wants_events:
            # The live-turn seed, once, right after the welcome. The snapshot
            # and the ready flag are set in ONE synchronous block: any event
            # folded before this instant is inside the seed and was not sent
            # to this connection; any event folded after it is sent and is not
            # in the seed. That single ordering fact is the mid-turn-join
            # correctness argument — no gap, no duplicate. (The lock fast-path
            # in ``_send_to`` acquires without yielding on an uncontended
            # connection, so a relay callback scheduled after this block still
            # queues its frame BEHIND the seed on the ordered stream.)
            seed_frame = {"op": "attach_sync", "data": self._live_turn.seed().to_json()}
            conn.events_ready = True
            await self._send_to(conn, seed_frame)
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
        task = conn.event_writer_task
        conn.event_writer_task = None
        if task is not None:
            self._event_sends.discard(task)
            # A send timeout drops its own connection from inside this task.
            # Cancelling self here would interrupt the cleanup path at its next
            # await and leave the stream close only half-observed.
            if task is not asyncio.current_task():
                task.cancel()
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
        from local_operator.mobile.types import validate_control_frame

        validate_control_frame(frame)
        h = self._handle
        if op == "ping":
            return "pong"
        if op == "snapshot":
            await self._push()
            return "snapshot sent"
        if op == "prompt":
            images = frame.get("images")
            fields: dict[str, Any] = {"images": images}
            if "command_id" in inspect.signature(h.prompt).parameters:
                fields["command_id"] = frame.get("command_id")
            return await h.prompt(frame["text"], **fields)
        if op == "steer":
            fields = {"images": frame.get("images")}
            if "command_id" in inspect.signature(h.steer).parameters:
                fields["command_id"] = frame.get("command_id")
            return await h.steer(frame["text"], **fields)
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
                frame["request_id"],
                frame["approved"],
                frame.get("remember", False),
            )
        if op == "recall_steer":
            # v4: follower Esc-recall parity. Optional capability — an owner
            # host that predates it answers with the unknown-op error, which
            # the follower surfaces as "cannot recall here".
            recall = getattr(h, "recall_steer", None)
            if not callable(recall):
                raise ValueError("this owner cannot recall queued steering")
            typed_recall = cast(Callable[[str], Awaitable[str]], recall)
            return await typed_recall(str(frame.get("command_id", "")))
        if op == "ask_answer":
            # ``question_index`` is the question the phone was DISPLAYING when
            # the user tapped (U8 guard): the handle rejects the answer if the
            # picker has since advanced past it. Optional — an older client that
            # omits it falls back to answering the current question, the
            # pre-guard behaviour.
            raw_index = frame.get("question_index")
            question_index = int(raw_index) if isinstance(raw_index, (int, float)) else None
            return await h.ask_answer(
                frame["request_id"],
                frame["value"],
                question_index=question_index,
            )
        raise ValueError(f"unknown op: {op!r}")

    # -- v4 event relay --------------------------------------------------------

    def _relay_event(self, data: dict[str, Any]) -> None:
        """Thread-safe relay entry: events fire on the HOST's thread (the
        Textual loop for a TUI owner), and everything relay-ordered must run
        on the registrant loop. ``call_soon_threadsafe`` from one producer
        thread preserves emission order, which is the whole relay contract."""
        loop = self._loop
        if loop is None or self._closed.is_set():
            return
        try:
            loop.call_soon_threadsafe(self._relay_on_loop, data)
        except RuntimeError:  # loop closing
            pass

    def _relay_on_loop(self, data: dict[str, Any]) -> None:
        """Fan one serialized AgentEvent out to event-subscribed attach clients.

        Folding into the live-turn tracker and snapshotting the recipient set
        happen SYNCHRONOUSLY here — that, plus the seed block in
        ``_on_connection``, is what makes a mid-turn join gapless (see the
        comment there). Each connection owns one bounded FIFO writer: a slow
        follower cannot delay healthy peers or accumulate tasks indefinitely,
        while a healthy follower receives every frame in emission order.

        Daemon connections are never in the recipient set: the phone's
        projection path is byte-identical to v3 by construction.
        """
        if self._closed.is_set():
            return
        self._live_turn.fold(data)
        recipients = [
            conn
            for conn in self._clients.values()
            if conn.kind == "attach" and conn.wants_events and conn.events_ready
        ]
        if not recipients:
            return
        frame = {"op": "event", "data": data}
        for conn in recipients:
            try:
                conn.event_queue.put_nowait(frame)
            except asyncio.QueueFull:
                # A raw stream cannot recover from one dropped frame. Remove the
                # peer immediately; reconnect rebuilds from durable history and
                # a fresh bounded live-turn seed.
                self._drop_client(conn)
                continue
            if conn.event_writer_task is None:
                task = asyncio.create_task(self._drain_event_queue(conn))
                conn.event_writer_task = task
                self._event_sends.add(task)
                task.add_done_callback(self._event_sends.discard)

    async def _drain_event_queue(self, conn: _ClientConn) -> None:
        """Drain one follower's raw event FIFO until it is dropped."""
        try:
            while id(conn.writer) in self._clients and not self._closed.is_set():
                frame = await conn.event_queue.get()
                try:
                    await self._send_to(conn, frame)
                finally:
                    conn.event_queue.task_done()
                if id(conn.writer) not in self._clients:
                    return
        except asyncio.CancelledError:
            pass
        finally:
            if conn.event_writer_task is asyncio.current_task():
                conn.event_writer_task = None

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
        # A callback may already be queued when close flips the cross-thread
        # event. Recheck here so shutdown cannot create new work behind itself.
        if self._closed.is_set() or self._push_scheduled:
            return
        self._push_scheduled = True
        self._push_task = asyncio.create_task(self._push_later())

    async def _push_later(self) -> None:
        try:
            # One short delay lets the current event batch fold before snapshot.
            await asyncio.sleep(0.05)
            if not self._closed.is_set():
                await self._push()
        finally:
            self._push_scheduled = False

    async def _push(self) -> None:
        """Broadcast projection repaints, preserving daemon bytes exactly.

        Event clients may need a follower-only gate overlay (currently a TUI
        approval). Build the ordinary projection once for daemon and legacy
        attach clients; only event subscribers get the overlaid copy. This is
        the protocol-v4 promise that phone frames remain byte-identical.
        """
        ordinary = {"op": "projection", "data": self._fold.projection.to_json()}
        await asyncio.gather(
            *(
                self._send_to(conn, self._projection_frame(conn, ordinary))
                for conn in list(self._clients.values())
            )
        )

    def _projection_frame(self, conn: _ClientConn, ordinary: dict[str, Any]) -> dict[str, Any]:
        if not conn.wants_events:
            return ordinary
        pending = getattr(self._handle, "event_pending", None)
        if pending is None:
            return ordinary
        overlaid = copy.deepcopy(ordinary)
        overlaid["data"]["pending"] = pending.to_json()
        overlaid["data"]["pending_count"] = max(
            1, int(overlaid["data"].get("pending_count", 0) or 0)
        )
        return overlaid

    async def _push_to(self, conn: _ClientConn) -> None:
        """The welcome form of a push: one full projection to one connection."""
        ordinary = {"op": "projection", "data": self._fold.projection.to_json()}
        await self._send_to(conn, self._projection_frame(conn, ordinary))

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

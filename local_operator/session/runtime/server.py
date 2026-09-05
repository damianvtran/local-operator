"""The session runtime: make a live session reachable over a control socket.

Every interactive ``lop`` process (the TUI, and ``exec`` when it runs
attached) hosts one of these. It does three things:

1. **Publishes** the discovery record (see :mod:`.registry`) and rewrites it
   on a heartbeat — cheap enough that a machine with no daemon installed
   pays nothing but one small file write every 15 seconds.
2. **Listens** on a random loopback port for control connections,
   authenticating each with the record's key (constant-time
   compare — the key is the whole credential).
3. **Bridges**: folds the session's event stream into the phone projection
   (:class:`~local_operator.mobile.projection.ProjectionFold`) and pushes a
   repaint on change; applies clients' requests to the session through a
   host-provided :class:`SessionHandle`.

This was ``mobile/registrant.py`` and the class was ``Registrant``. Nothing
about it is phone-specific except the projection fold, which is an injected
:class:`ProjectionSink` collaborator built lazily on the first daemon dial
when none is supplied: the phone daemon, an attach terminal, and — later —
wakes and background automations are all VIEWERS of one session runtime.
``Registrant`` remains as an alias at the bottom of this module and at the old
import path, so no call site had to change in the move.

The handle indirection exists because the two runtime kinds drive their
session differently: the TUI must route mutations through Textual's message
pump and thread (``call_from_thread``), while an exec-mode host can call the
session directly. The runtime speaks to the handle, never to Textual.

Threading: the control socket server runs on its own thread with its own
event loop — the TUI's loop must never block on a phone, and clients'
requests (model switches, aborts) must land even while the TUI is mid-repaint.
All session mutations funnel through the handle, whose contract is "callable
from the runtime's loop, serialized by the implementor".
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import logging
import os
import secrets
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, cast

if TYPE_CHECKING:
    from local_operator.harness.types import ImageContent

from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.types import SessionProjection
from local_operator.session.frontend_state import FRONTEND_CAPABILITY
from local_operator.session.runtime.registry import RecordPublisher
from local_operator.session.runtime.types import (
    ATTACH_MAX_CLIENTS,
    HEARTBEAT_INTERVAL_S,
    ClientKind,
    ClientLocality,
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
# reconnect through durable history + canonical frontend_sync instead of drift.
_EVENT_QUEUE_MAX = 64

# Ops whose answer is structured data (a typed slash result, a cancel count)
# rather than a one-line receipt: they reply with a ``result`` frame so the
# invoker renders the outcome locally instead of the owner's transcript
# printing it.
#: How long ``announce_stop`` will wait for its frame to be handed to the
#: transport on the thread-hosted path. Bounds a courtesy write against a
#: stalled viewer: the caller is the TUI's event loop during a /stop, so this
#: is a frozen-UI budget, not a delivery guarantee. A viewer that misses the
#: frame degrades to the pre-announcement behaviour; the stop is unaffected.
_ANNOUNCE_WRITE_TIMEOUT_S = 0.25

_PAYLOAD_OPS = {"slash_result", "cancel_subagents", "job_trajectory", "credential"}


def _accepts_locality(fn: Any) -> bool:
    """Whether ``fn`` takes the ``locality`` keyword.

    Cached because it is asked on every routed slash command and
    ``inspect.signature`` is not cheap. Keyed by the underlying function so
    bound methods of the same class share one answer. A handle whose signature
    cannot be read (a C callable, an exotic mock) is treated as not accepting
    it: the caller then uses the narrow call, which every implementation has
    always supported.
    """
    target = getattr(fn, "__func__", fn)
    cached = _LOCALITY_SUPPORT.get(target)
    if cached is not None:
        return cached
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        answer = False
    else:
        answer = "locality" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    _LOCALITY_SUPPORT[target] = answer
    return answer


#: Memo for ``_accepts_locality``. Keyed weakly so a handle class that goes
#: away (a per-test double) does not pin its function objects in memory.
_LOCALITY_SUPPORT: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()


#: Rows one ``job_trajectory`` reply may carry. The whole retained window is
#: 500 events with no size bound per event, which is what overflows the frame
#: limit in the first place, so the viewer pages rather than asking for all of
#: it: 120 rows of ordinary tool traffic sit far inside ``_MAX_LINE_BYTES``
#: while keeping the round trips for a full window in single digits.
_TRAJECTORY_PAGE_MAX = 120


@dataclass
class _ClientConn:
    """One authenticated control connection in the runtime's registry.

    Multiplexing is already half-there (frames carry caller-chosen ``req``
    ids), so multi-front-end needs N concurrent connections on the ONE
    socket rather than a second protocol: the daemon plus up to
    ``ATTACH_MAX_CLIENTS`` attach terminals. ``last_seen`` is the LRU clock
    for attach eviction — stamped on every request so the least-recently-ACTIVE
    follower is the one dropped when the cap is hit.
    """

    writer: asyncio.StreamWriter
    kind: ClientKind
    #: Whether the human on the other end is at this machine. Declared in the
    #: auth frame; see ``ClientLocality``. Only ops that act on the USER's
    #: surroundings (an OAuth browser tab) read it.
    locality: ClientLocality = "local"
    last_seen: float = field(default_factory=time.monotonic)
    # Frames on one TCP stream must stay ordered, while unrelated streams must
    # never queue behind its backpressure.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # v4: this attach client asked for the raw AgentEvent relay in its auth
    # frame. Daemon connections never set it; a v3 attach client that omitted
    # the flag keeps projection-only behaviour.
    wants_events: bool = False
    # v5 canonical state is attach-only and independently negotiated so daemon
    # projection bytes never gain frontend frames.
    wants_frontend: bool = False
    frontend_ready: bool = False
    # Updates can be scheduled back to this loop while the owner-loop
    # subscription call is returning. Hold them until the sync frame is queued;
    # dropping them creates an immediate sequence hole at every busy join.
    frontend_pending: list[dict[str, Any]] = field(default_factory=list)
    # Flipped only AFTER welcome + canonical frontend_sync are queued. Raw
    # events begin behind that boundary on the same FIFO, so a joining client
    # cannot see transcript animation ahead of the snapshot that seeded it.
    events_ready: bool = False
    event_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_EVENT_QUEUE_MAX)
    )
    # Exactly one writer drains the queue, so delivery stays ordered without a
    # task per event. Held for shutdown and slow-client eviction.
    event_writer_task: asyncio.Task[None] | None = None
    frontend_unsubscribe: Callable[[], None] | None = None
    # Job ids whose trajectory deltas this connection wants (``watch_job``).
    # Empty by default and per-connection by necessity: the snapshot ships no
    # trajectories at all (they overflow ``_MAX_LINE_BYTES``), so a viewer
    # opts in only for the child page a reader actually opened. A second
    # viewer watching a different child must not widen this one's stream.
    watched_jobs: set[str] = field(default_factory=set)


class SessionHandle(Protocol):
    """What the runtime needs from its host application.

    Every method is awaited on the RUNTIME'S loop (its own thread); the
    implementor guarantees any hop the session needs (for the TUI: Textual's
    ``call_from_thread``; for an owned session: ``run_coroutine_threadsafe``
    back to the daemon loop). Methods return a short human-readable receipt
    that becomes the ``ack`` detail — the same line the TUI would print as a
    notice.
    """

    @property
    def session_projection_seed(self) -> SessionProjection:
        """The projection skeleton: identity fields the runtime folds onto."""
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
        model change): the runtime pushes whatever changed."""
        ...

    # -- the kill switch: graceful self-stop (optional, probed) --------------
    # request_stop() -> None: deny parked gates, abort the turn, dispose the
    # session and begin the runtime's own shutdown, so the ``stop`` control
    # op ends this runtime the way SIGTERM would. Probed with getattr like
    # every optional capability below so reduced handles (tests, older
    # bridges) keep satisfying the protocol: a handle without it answers the
    # ``stop`` op with the unknown-op error, and the caller's escalation
    # ladder (session/runtime/control.py) proceeds to identity-confirmed
    # SIGTERM — which the runtime's signal handler has always honoured.
    # Sync and non-raising by contract: called ON the runtime loop from the
    # dispatch, and a stop that faults here is still a stop.
    #
    # -- v4 optional capabilities (probed with getattr, never required) -------
    # subscribe_events(on_event) -> unsubscribe: feed the host session's raw
    #   AgentEvent stream, serialized (``model_dump(mode="json")``) on the
    #   host's own loop, to ``on_event`` (thread-safe). The runtime relays
    #   the dicts to event-subscribed attach clients. Optional so old hosts
    #   and reduced test handles keep working — without it, attach clients
    #   simply get v3 projection-only behaviour.
    # recall_steer(command_id) -> str: unsend the queued steering message the
    #   follower submitted under ``command_id``; raises when it already
    #   drained. Optional for the same reason.
    # receive_peer_message(text, *, mode="mailbox", wake=False, sender=None)
    #   -> str: deliver a message from another local lop session (`lop send`).
    #   Optional (getattr-probed in _dispatch) so reduced test handles and
    #   non-interactive exec hosts that never wired it keep working — a handle
    #   lacking it answers "this session cannot receive peer messages".
    # cancel_gracefully() -> str: stop the turn at the POST-TOOL boundary
    #   instead of cutting the running tool (Session.request_graceful_cancel).
    #   Serves the ``cancel`` op's default mode. Deliberately distinct from
    #   ``abort`` rather than a parameter on it: abort's contract is "stop now,
    #   the human will repair the mess", and a supervised agent has no human to
    #   repair a half-finished push. Optional and getattr-probed so hosts that
    #   cannot honour a boundary (a reduced handle, an older bridge) say so
    #   plainly instead of silently doing the destructive thing.


class ProjectionSink(Protocol):
    """What the runtime needs from a projection collaborator.

    The phone renders from a :class:`SessionProjection` snapshot that the
    runtime broadcasts on change; :class:`ProjectionFold` is the production
    implementation. The runtime only ever READS ``projection`` (to serialize
    a frame) and calls ``set_pending`` (the reduced-handle bridge for gate
    cards), so that is the whole contract — narrow enough that a test can
    hand in a stub and a future runtime with no phone can hand in nothing.
    """

    @property
    def projection(self) -> SessionProjection: ...

    def set_pending(self, pending: Any) -> None: ...


class RuntimeServer:
    """One per interactive process. Construct, ``start()``, ``close()``."""

    def __init__(
        self,
        handle: SessionHandle,
        *,
        kind: str = "tui",
        projection_sink: ProjectionSink | None = None,
    ) -> None:
        #: Live state mirrored into the discovery record. Held here rather
        #: than read off the record so the publish is one assignment and the
        #: fields have a defined value before the record exists.
        self._pending: str | None = None
        self._busy = False
        #: True until a terminal attaches. A freshly spawned runtime genuinely
        #: has no viewer, so this starts True rather than False — the old
        #: default had every new runtime claiming a terminal it had never had.
        self._detached = True
        self._handle = handle
        # Back-reference so the handle can publish record state it alone knows
        # about — today the parked-gate ``pending`` bit, which originates deep
        # inside the approval gate and has to reach the discovery record for
        # `lop sessions` and the picker to show it. Set defensively: reduced
        # handles in tests are plain objects and must not fail on an attribute
        # assignment they never asked for.
        try:
            handle._registrant = self  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — an unwritable handle simply cannot publish
            logger.debug("handle does not accept a registrant back-reference", exc_info=True)
        # With the back-reference in place the handle can answer "is anyone
        # watching?", which is what the model needs in its prompt so a
        # detached session does not ask a question nobody can answer.
        installer = getattr(handle, "_install_interactivity_probe", None)
        if callable(installer):
            try:
                installer()
            except Exception:  # noqa: BLE001 — a probe is never worth a runtime
                logger.debug("could not install the interactivity probe", exc_info=True)
        seed = handle.session_projection_seed
        seed.kind = kind
        # The projection fold is an OPTIONAL, injected collaborator. A caller
        # that already owns a fold may hand it in and the runtime uses it
        # as-is — only tests do today; every production constructor call
        # (the TUI at ``tui/app.py``, the owned-session process) passes none.
        # Given none, nothing is built until a client that consumes
        # projection semantics (the mobile daemon) actually dials, so a
        # runtime that only ever serves a follower terminal or fires a wake
        # constructs no fold at all. Welcomes and repaints before that moment
        # serialize the seed directly: the seed IS the object the handle's
        # own fold mutates, so the bytes on the wire are identical either
        # way. See ``_ensure_projection_sink``.
        self._projection_sink: ProjectionSink | None = projection_sink
        #: How many times this runtime built a fold on its own. Observable
        #: for tests and for the "did a headless runtime pay for a fold?"
        #: question; 0 after a lifetime with no daemon client is the claim.
        self.projection_sinks_built: int = 0
        self._record = SessionRecord(
            pid=os.getpid(),
            kind=kind,  # type: ignore[arg-type]
            session_id=seed.session_id,
            conversation_name=seed.conversation_name,
            cwd=seed.cwd,
            model_label=seed.model_label,
            control_port=0,  # stamped when the listener binds
            control_key=secrets.token_hex(32),
            capabilities=([FRONTEND_CAPABILITY] if hasattr(handle, "subscribe_frontend") else []),
            # A runtime is born with no terminal watching it. Stamped at
            # construction rather than left to the first transition, because
            # the window before a viewer attaches is exactly when a detached
            # runtime is most interesting to look at.
            detached=True,
        )
        self._publisher: RecordPublisher | None = None
        self._server: asyncio.AbstractServer | None = None
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
        # One warning per contiguous run of oversized frames, not one per
        # frame: a busy session repaints ~30x/s and a per-frame warning is the
        # log flood the cap exists to prevent. Reset when a frame fits again.
        self._frame_cap_warned = False
        # The delayed repaint must be owned like the heartbeat. A bare task can
        # still be sleeping when close tears down the runtime loop, producing
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
        #
        # READ THIS BEFORE YOU TOUCH THE COUNTER. It is SERVER-GLOBAL state
        # mutated from THREE per-connection contexts, and that mismatch — not
        # any one line — is what has produced four separate defects in this
        # predicate across four review rounds:
        #
        #   1. removal          `_drop_client` never released a dead daemon's
        #                       count, so a phantom viewer suppressed toasts
        #                       forever (R5).
        #   2. second removal   `_drop_client` is documented to run TWICE on
        #                       one connection; an unconditional release on the
        #                       late call wiped the REPLACEMENT's live count
        #                       (R7).
        #   3. request          a `watch`/`unwatch` buffered behind a parked op
        #                       still arrives after its connection is evicted,
        #                       because closing the writer does not stop the
        #                       `StreamReader` (R8) — and an `attach` client's
        #                       `watch` incremented a count only a `daemon`
        #                       drop could clear (R9).
        #
        # Every instance failed the same way: the count outlives, or is stolen
        # from, the connection it describes. Each is now defended separately,
        # which is why three guards say "is this connection still registered?"
        # in three places.
        #
        # THE DURABLE FIX IS STRUCTURAL, and deliberately not taken here: hold
        # the count on `_ClientConn` and fold over the live registry, exactly
        # as `watching_surfaces` already does for `attach` clients. A dropped
        # connection then removes its own contribution BY CONSTRUCTION — no
        # zeroing, no registry guards, and every one of these spellings becomes
        # unrepresentable rather than separately defended. It was scoped out of
        # this release as too large for a review round; do it before adding a
        # fourth mutation site, not after the fifth defect.
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
        # The thread name is a runtime-observable diagnostic (py-spy, thread
        # dumps, `ps -M`) and deliberately keeps its mobile-era spelling: this
        # move changes no behaviour, and renaming it would silently invalidate
        # anyone's saved grep. Rename it only alongside RUN_DIRNAME, if ever.
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

    def announce_stop(self) -> None:
        """Tell every attached viewer this session is ending DELIBERATELY.

        THE single emitter of the ``stopping`` frame, called by both triggers:
        the control-op path (a peer's ``lop stop``, another TUI's
        ``/stop all``) and the owner's own bare ``/stop``, which runs its
        teardown locally and never dispatches an op. Round 3 found that second
        route silent, so a follower watching an owner that stopped itself saw
        a plain EOF and took over the session the user had just ended — U2-4
        surviving on a different path.

        Must be called BEFORE the teardown that closes these sockets. Safe
        twice: a viewer reads the frame only as "the disconnect coming next
        is deliberate", so a duplicate is a no-op.

        There is deliberately no on-disk fallback. A wakeless session has no
        wake-index entry at all (``write_entry`` removes the file when a
        session has no schedules), so the wire is the only channel covering
        every session — which is why the marker approach was dropped.

        Safe from any thread, like :meth:`close`, and best-effort by
        contract: a viewer that never receives it degrades to the pre-round-2
        behaviour, which is strictly better than a stop failing because one
        socket was slow.
        """
        if self._closed.is_set():
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        frame = {"op": "stopping", "session_id": self._record.session_id}
        if self._on_owner_loop():
            # An in-process runtime shares the TUI's loop, so the owner's own
            # /stop arrives HERE, from a synchronous caller whose very next
            # statement tears the sockets down. Awaiting a drain is therefore
            # not available and scheduling a task is too late — the teardown
            # would win the race. ``write`` is synchronous (it buffers into
            # the transport), and a transport closed afterwards still flushes
            # what it holds, so writing inline is what actually gets the frame
            # to the viewer ahead of the EOF it must explain.
            self._write_now(frame)
            return
        # The THREAD-HOSTED path, which is the one production takes: the TUI
        # hosts its registrant with `.start()`, and the caller is a coroutine
        # on the TUI's own event loop. Awaiting a drain here froze that loop
        # for up to two seconds against a viewer whose receive window was full
        # (round-4 MINOR-3, the #401 class) — on a path whose whole contract
        # is that announcing must never make a stop slower.
        #
        # The same reasoning the inline branch rests on applies once the write
        # is on the right thread: ``write`` only buffers, and a transport
        # closed afterwards still flushes what it holds. So hand the write to
        # the runtime's loop and wait only for it to have BEEN WRITTEN, with a
        # bound far below any user-perceptible pause. Missing that bound costs
        # a viewer its explanation, never the stop.
        written = threading.Event()

        def _write_and_signal() -> None:
            try:
                self._write_now(frame)
            finally:
                written.set()

        try:
            loop.call_soon_threadsafe(_write_and_signal)
        except RuntimeError:
            # Loop already closing: nothing is listening that could care.
            return
        if not written.wait(timeout=_ANNOUNCE_WRITE_TIMEOUT_S):
            logger.debug("stop announcement did not reach viewers before the teardown")

    def _write_now(self, frame: dict[str, Any]) -> None:
        """Buffer one frame to every viewer without awaiting a drain.

        PRECONDITION: must run ON the runtime's event loop. Both callers
        satisfy it — the in-process branch is already there, and the
        thread-hosted branch hands this to the loop with
        ``call_soon_threadsafe`` — and it is what makes skipping
        ``conn.send_lock`` sound: no other coroutine can be mid-write at that
        instant, so a partially-written frame is impossible, and taking the
        lock would require awaiting, which the synchronous caller cannot do.
        Called from any other thread the lock-free write would be unsafe
        (round-4 NIT-2: the guarantee belongs to the call site, not the
        method, and saying so is what stops the next reuse from breaking it).
        """
        for conn in list(self._clients.values()):
            try:
                conn.writer.write(json.dumps(frame).encode() + b"\n")
            except Exception:  # noqa: BLE001 — announcing is best-effort
                logger.debug("stop announcement write failed", exc_info=True)

    def close(self) -> None:
        """Unpublish and shut down. Safe from any thread, safe twice.

        Thread-hosted runtimes are joined before returning. In-process hosts
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
            logger.debug("runtime loop exited during shutdown", exc_info=True)
        except TimeoutError:
            logger.debug("runtime shutdown did not finish before timeout", exc_info=True)

    async def aclose(self) -> None:
        """Await complete teardown on the owning loop.

        Repeated calls still join the original cleanup task; merely observing
        the cross-thread closed flag is not proof that loop-owned work ended.
        """
        self._request_close()
        if not self._on_owner_loop():
            raise RuntimeError("RuntimeServer.aclose() must run on its owning event loop")
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
                logger.debug("runtime unsubscribe failed", exc_info=True)
            self._unsubscribe = None
        if self._unsubscribe_events is not None:
            try:
                self._unsubscribe_events()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                logger.debug("runtime event unsubscribe failed", exc_info=True)
            self._unsubscribe_events = None

    def _on_owner_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    # -- the runtime's own loop -----------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception:  # noqa: BLE001 — a dead runtime must not kill the host
            logger.warning("session runtime loop died", exc_info=True)
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
        """Create the one teardown task; called only on the runtime loop."""
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(self._shutdown_impl())
            self._shutdown_task = task
        return task

    async def _shutdown_on_loop(self) -> None:
        """Join idempotent teardown from a coroutine on the runtime loop."""
        await asyncio.shield(self._ensure_shutdown_task())

    async def _shutdown_impl(self) -> None:
        """Cancel and join every object owned by the runtime event loop."""
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
                logger.debug("runtime heartbeat failed", exc_info=True)

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
        # runtime must land on the class it always had, or every rolling
        # upgrade would demote the phone bridge to a follower.
        raw_kind = frame.get("client", "daemon")
        kind: ClientKind = "attach" if raw_kind == "attach" else "daemon"
        # Absent means LOCAL, matching every client that exists today: the
        # listener is loopback-only, so anything that dialed is on this
        # machine. A relay forwarding a remote device's commands is the one
        # caller that must say ``"remote"``, and an old client that never
        # heard of the field keeps the behaviour it always had.
        locality: ClientLocality = "remote" if frame.get("locality") == "remote" else "local"
        # v4: only attach clients may subscribe to the raw event relay. The
        # daemon's projection path must stay byte-identical, so a daemon auth
        # carrying the flag (there is none today) is deliberately ignored.
        wants_events = kind == "attach" and bool(frame.get("events"))
        wants_frontend = kind == "attach" and bool(frame.get("frontend_state"))
        if wants_frontend and FRONTEND_CAPABILITY not in self._record.capabilities:
            writer.close()
            return

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

        conn = _ClientConn(
            writer=writer,
            kind=kind,
            locality=locality,
            wants_events=wants_events,
            wants_frontend=wants_frontend,
        )
        self._clients[id(writer)] = conn
        # A terminal arriving flips ``detached`` (round 1, U2: it was computed
        # only inside a pending transition, so every session claimed a viewer
        # it might never have had). Cheap and de-duplicated — see
        # ``_republish_detached``.
        if kind == "attach":
            self._republish_detached()
        if kind == "daemon":
            # The daemon is the one client that renders projections (attach
            # clients read the welcome for identity only), so its arrival is
            # when a lazily-built fold earns its keep.
            self._ensure_projection_sink()
        await self._push_to(conn)  # the welcome: a full projection, unprompted
        if conn.wants_frontend:
            subscribe_frontend = getattr(self._handle, "subscribe_frontend", None)
            if not callable(subscribe_frontend):
                self._drop_client(conn)
                return

            def on_update(update: Any) -> None:
                payload = (
                    update.model_dump(mode="json") if hasattr(update, "model_dump") else update
                )
                self._relay_frontend_to(conn, payload)

            from local_operator.session.frontend_state import (
                FrontendSubscription,
                oversized_frame_report,
                sync_wire_payload,
            )

            outcome = subscribe_frontend(on_update)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            subscription = cast(FrontendSubscription, outcome)
            sync = subscription.sync
            # Trajectories are stripped here and re-fetched per job through
            # ``job_trajectory``; see ``sync_wire_payload`` for why the frame
            # cannot carry them.
            sync_payload = sync_wire_payload(sync)
            conn.frontend_unsubscribe = subscription.unsubscribe
            # Registration and snapshot capture happened synchronously on the
            # authoritative loop. Mark ready only after queuing that snapshot;
            # later updates therefore cannot overtake it.
            sync_frame = {"op": "frontend_sync", "data": sync_payload}
            # An oversized sync is unreadable, not merely large: the client's
            # readline raises and its pump dies, so the viewer waits out its
            # full sync timeout and then degrades to a cold session with no
            # roster and no todos. That looked exactly like a slow owner for
            # one release. Say so loudly and name the field responsible, so
            # the next unbounded list is one log line to find rather than a
            # profiling session.
            oversize = oversized_frame_report(sync_frame, _MAX_LINE_BYTES)
            if oversize is not None:
                logger.error("session runtime: frontend_sync will not fit — %s", oversize)
            await self._send_to(conn, sync_frame)
            conn.frontend_ready = True
            for pending in conn.frontend_pending:
                self._enqueue_client_frame(conn, {"op": "frontend_update", "data": pending})
            conn.frontend_pending.clear()
        if conn.wants_events:
            # In-flight transcript state rides the same canonical sync as every
            # other full-TUI field. Raw events begin only after that frame.
            conn.events_ready = True
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
        # DID THIS CALL ACTUALLY REMOVE THE CONNECTION? `_drop_client` is
        # designed to run TWICE on one connection — `_send_to` drops a client
        # whose send failed, and that connection's reader loop later observes
        # the close and drops it again from its `finally`, which the docstring
        # there calls "a no-op second removal". Anything below that mutates
        # SERVER-GLOBAL state has to honour that contract, or the late second
        # call reaches across to whatever connection replaced this one.
        was_registered = self._clients.pop(id(conn.writer), None) is not None
        # The other half of the ``detached`` transition: the last terminal
        # leaving is precisely when the picker must start saying "nobody is
        # watching this". Published from the ONE removal path so no exit route
        # (reader-loop end, shutdown, eviction) can miss it.
        if conn.kind == "attach":
            self._republish_detached()
        # `phone_watchers` is the daemon CONNECTION's state kept in a
        # server-global counter, and only an `unwatch` op decrements it — an op
        # the daemon sends from an SSE generator's `finally`, in the process
        # that just died. So a daemon restart while a phone is watching left
        # the +1 behind forever: the new daemon's `_reconcile` replays `watch`
        # (a second +1), the phone's eventual close sends ONE `unwatch`, and
        # the residue reports a viewer nobody can see. Every parked approval on
        # that session then sends no toast and the model is told a human is
        # watching — round 3's B1 failure mode, restored by a third route
        # (round 5, R5).
        #
        # Zeroed rather than decremented: the count belongs to the connection
        # that reported it, a new daemon re-announces every session it watches,
        # and at most one daemon connection exists at a time.
        #
        # GUARDED ON `was_registered`, because at most one daemon connection
        # exists but its DROPS are not unique. An evicted daemon parked inside
        # `_on_request` unwinds only when its await returns — by then the
        # replacement has dialled, replayed `watch`, and owns the counter, so
        # an unconditional zero here wiped a LIVE watcher (round 6, R7). That
        # failed OPEN, unlike the leak it replaced: a phone being looked at
        # reported nobody watching, so every parked approval toasted a card
        # already on the user's screen and the model was told no one could
        # answer. Derived from the registry rather than asserted, the same way
        # the `attach` half above computes `detached`.
        if conn.kind == "daemon" and was_registered:
            self.phone_watchers = 0
        task = conn.event_writer_task
        conn.event_writer_task = None
        if task is not None:
            self._event_sends.discard(task)
            # A send timeout drops its own connection from inside this task.
            # Cancelling self here would interrupt the cleanup path at its next
            # await and leave the stream close only half-observed.
            if task is not asyncio.current_task():
                task.cancel()
        frontend_unsubscribe = conn.frontend_unsubscribe
        conn.frontend_unsubscribe = None
        if frontend_unsubscribe is not None:
            try:
                frontend_unsubscribe()
            except Exception:  # noqa: BLE001 — connection cleanup must finish
                logger.debug("frontend client unsubscribe failed", exc_info=True)
        try:
            conn.writer.close()
        except Exception:  # noqa: BLE001
            pass

    def set_record_pending(self, pending: str | None) -> None:
        """Record that this session is waiting for a PERSON (or no longer is).

        Named for the RECORD it writes, distinct from ``set_pending`` below,
        which carries a ``PendingRequest`` into the projection so a front end
        can paint the card. Two different consumers: that one is "show the
        user this question", this one is "tell the machine a person is owed".

        ``"approval"`` / ``"ask"`` / ``None``. Republished immediately rather
        than waiting for the 15 s heartbeat, because the whole value of the
        field is that a user hunting for "what is that 283 MB process doing"
        finds the answer at once.
        """
        if self._pending == pending:
            return
        self._pending = pending
        self._republish()

    def _republish_detached(self) -> None:
        """Refresh the record when the attached-terminal count crosses 0↔1.

        De-duplicated on the resulting BOOLEAN rather than on the count: a
        second terminal attaching to a session that already had one changes
        nothing a reader can see, and republishing for it would put a staged
        write on every connection churn.
        """
        detached = self.attach_clients() == 0
        if detached == self._detached:
            return
        self._detached = detached
        self._republish()
        if detached and self._pending:
            # A GATE WAS OPENED WHILE SOMEBODY WAS WATCHING, and they have now
            # closed the terminal. The routing decision was made once, at
            # announce time, and correctly sent no toast then — so without
            # this the question waits up to 24 h and the user is never told
            # (round 3, B2). "I approved something, shut the laptop, came back
            # to a session that had been waiting all day" is the ordinary
            # shape of it. Re-announcing on the transition is what turns the
            # one-shot decision into a live one.
            announce = getattr(self._handle, "reannounce_pending", None)
            if callable(announce):
                try:
                    announce()
                except Exception:  # noqa: BLE001 — a toast never breaks teardown
                    logger.debug("could not re-announce the parked gate", exc_info=True)

    def set_busy(self, busy: bool) -> None:
        """Record whether a turn is running, for the picker's liveness marker."""
        if self._busy == busy:
            return
        self._busy = busy
        self._republish()

    def _republish(self) -> None:
        """Refresh the discovery record with the current live state.

        Through ``RecordPublisher.heartbeat``, which is already the one way
        this process rewrites its record — a second publish path here would be
        a second thing that can disagree about the record's contents.

        Best-effort: publishing is one small staged write and rename, and a
        failure costs a marker rather than a session. Called on every
        transition rather than left to the 15 s heartbeat because the value of
        these fields is that they are current when somebody looks.
        """
        publisher = getattr(self, "_publisher", None)
        if publisher is None:
            return
        try:
            publisher.heartbeat(
                pending=self._pending,
                busy=self._busy,
                detached=self.attach_clients() == 0,
            )
        except Exception:  # noqa: BLE001 — a stale marker is not worth an exception
            logger.debug("could not republish the session record", exc_info=True)

    def attach_clients(self) -> int:
        """How many attach (follower terminal) connections are live.

        Term 3 of the runtime's residency predicate (``process._should_exit``):
        an interactive viewer holds the runtime warm; ``daemon`` clients never
        do. Also the attach-cap count."""
        return sum(1 for c in self._clients.values() if c.kind == "attach")

    def watching_surfaces(self) -> frozenset[str]:
        """Which KINDS of surface have a HUMAN watching this session right now.

        Notification routing needs the kind, not the count: a question goes to
        whatever is actually watching, and only falls out to the OS when
        nothing is.

        **A ``daemon`` connection is NOT somebody watching.** ``"daemon"`` is
        the default kind for an auth frame with no ``client`` field, which is
        exactly what the mobile daemon's ADOPTION dial sends — and that dial
        covers every session on the machine and is held open permanently
        (`mobile/daemon.py::_dial`). Counting it meant that on any machine
        running ``lop mobile`` no parked approval ever sent a notification,
        the gate held ~283 MB for 24 h, and the model was told a human was
        watching (round 3, B1). ``process.py::_viewer_attached`` reads this
        same table and counts only ``"attach"`` for exactly this reason.

        **A PHONE THAT IS BEING LOOKED AT REGISTERS THROUGH ``watch``.**
        ``phone_watchers`` is incremented by the ``watch`` control op, which
        the daemon pushes when a session's SSE subscriber count crosses 0↔N
        (`mobile/daemon.py::notify_watch_transition`) — i.e. exactly when a
        person opens or closes the session on their phone. That is the signal
        production already produces, and reading anything else is how round 3
        traded B1's false positive for a false negative: the fix introduced a
        parallel `note_viewer_active` mechanism that NOTHING called, so a user
        reading the session on their phone got a desktop toast for a card
        already on their screen, and the model was told nobody could answer
        (round 4, R1/Q1).

        ``watch_supported`` guards the mixed-version case for us: it latches
        on the first ``watch``/``unwatch`` ever seen, so a daemon too old to
        send the op leaves it False and this reports no phone rather than
        inventing one. That matches the reaper's reading of the same pair.

        Deliberately the live connection table rather than a cached flag:
        surfaces come and go constantly, and a stale answer here means a
        notification delivered to a surface that has gone away.
        """
        watching = {c.kind for c in self._clients.values() if c.kind == "attach"}
        if self.watch_supported and self.phone_watchers > 0:
            # Reported as ``viewer`` rather than ``daemon`` so a reader cannot
            # confuse "a relay is connected" (true of every session on a
            # machine running `lop mobile`) with "a person is looking".
            watching.add("viewer")
        return frozenset(watching)

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
            if op in ("watch_job", "unwatch_job"):
                # Trajectory subscription for ONE child page, per connection.
                # Handled here rather than in ``_dispatch`` because it mutates
                # this connection's own state and never touches the session:
                # the dispatcher deliberately has no ``conn``.
                job_id = str(frame.get("job_id") or "")
                if not job_id:
                    raise ValueError("job_id must be a non-empty string")
                if op == "watch_job":
                    conn.watched_jobs.add(job_id)
                else:
                    conn.watched_jobs.discard(job_id)
                detail = f"watching {len(conn.watched_jobs)} job(s)"
            elif op in ("watch", "unwatch"):
                # The reaper's phone-watcher signal (§2.8). watch_supported
                # latches on the FIRST op seen so a mixed-version child never
                # mistakes silence for zero watchers. Deliberately OUTSIDE the
                # registry guard below: it is a version signal, not a count,
                # and a frame proves the daemon speaks the op whenever it
                # arrived.
                self.watch_supported = True
                # ONLY A REGISTERED CONNECTION MAY MOVE THE COUNT. The reader
                # loop is strictly serial — `readline()` then `await
                # _on_request(...)` — so anything the daemon sent before it
                # died is still in the socket buffer while an op is parked.
                # `_drop_client` closes the WRITER, but the `StreamReader`
                # keeps yielding those buffered lines, so this runs on a
                # connection already evicted from the registry. A dying
                # daemon's `unwatch` (pushed from the SSE generator's
                # `finally`) then wiped the REPLACEMENT daemon's live count
                # (round 7, R8).
                #
                # Gated on `conn.kind` too: only the daemon's count is ever
                # cleared (`_drop_client` zeroes for `kind == "daemon"`), so an
                # attach client's `watch` would increment something nothing can
                # clear — a permanent phantom viewer (round 7, R9).
                if conn.kind != "daemon" or id(conn.writer) not in self._clients:
                    pass
                elif op == "watch":
                    self.phone_watchers += 1
                else:
                    self.phone_watchers = max(0, self.phone_watchers - 1)
                detail = f"watchers: {self.phone_watchers}"
            elif op in _PAYLOAD_OPS:
                # Structured-answer ops reply with a ``result`` frame whose
                # ``data`` the invoker renders locally (a slash command's typed
                # outcome, a cancel's authoritative count) rather than a
                # one-line receipt that would paint in the owner's transcript.
                data = await self._dispatch_payload(op, frame, conn.locality)
                await self._send_to(conn, {"op": "result", "req": req, "data": data})
                await self._handle.refresh()
                await self._push()
                return
            else:
                duplicate = self._already_admitted(op, frame)
                if duplicate:
                    # A retry of an errand this transcript already owns (a
                    # sender that crashed after the row was durable, a wake
                    # re-fired by a restarted supervisor). Acked, not executed:
                    # the caller's outcome is "delivered", which is true, and
                    # nothing is appended twice. See
                    # ``OwnedSessionHandle.has_admitted_command``.
                    detail = "already admitted"
                else:
                    detail = await self._dispatch(op, frame)
                await self._send_to(
                    conn,
                    {"op": "ack", "req": req, "detail": detail, "duplicate": duplicate},
                )
                if not duplicate:
                    await self._handle.refresh()
                    await self._push()
                return
            await self._send_to(conn, {"op": "ack", "req": req, "detail": detail})
            # Mutations change the projection; push what every front end
            # should see. ``stop`` is exempt: its whole job is to END the
            # session, so a post-ack refresh would re-read a host that is
            # mid-dispose (the TUI's handle raises "session is still
            # starting" the moment its session reference drops) — the ack is
            # the reply and the ladder's exit-wait is the confirmation.
            if op not in ("watch", "unwatch", "watch_job", "unwatch_job", "stop"):
                await self._handle.refresh()
                await self._push()
        except Exception as exc:  # noqa: BLE001 — the error IS the reply
            await self._send_to(conn, {"op": "error", "req": req, "message": str(exc)[:400]})
            await self._push()

    def _already_admitted(self, op: str, frame: dict[str, Any]) -> bool:
        """Is this a retry of a turn the transcript already carries?

        Only ``prompt`` carries a durable, append-only identity, so only it can
        be answered from the transcript. ``steer`` is deliberately excluded:
        its idempotency is the handle's own reservation map, and a steer is not
        an append-only user row to match against.

        Optional capability, probed — a reduced handle without it simply never
        reports a duplicate, which is the pre-idempotency behaviour.
        """
        if op != "prompt":
            return False
        command_id = str(frame.get("command_id") or "")
        if not command_id:
            return False
        checker = getattr(self._handle, "has_admitted_command", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(command_id))
        except Exception:  # noqa: BLE001 — never fail a turn over a dedupe probe
            logger.debug("admitted-command probe failed", exc_info=True)
            return False

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
        if op == "cancel":
            # Two rungs, one op, because the CHOICE is the feature. The default
            # is graceful: a supervisor cancelling a sentinel mid-``git push``
            # must not tear the push in half, and defaulting the other way makes
            # the dangerous behaviour the one you get by not thinking. Asking
            # for ``immediate`` is asking for ``abort``, so it routes there
            # rather than growing a second implementation of "stop now".
            #
            # Optional capability, getattr-probed like every other addition to
            # this dispatch: a reduced test handle or an older bridge answers
            # the unknown-op error, and a caller that needs a stop regardless
            # falls back to ``abort`` (or to the stop ladder). Additive on the
            # wire for the same reason ``peer_message`` and ``stop`` were, so
            # no PROTOCOL_VERSION bump.
            if str(frame.get("mode", "graceful")) == "immediate":
                return await h.abort()
            cancel = getattr(h, "cancel_gracefully", None)
            if not callable(cancel):
                raise ValueError("this session cannot cancel at a tool boundary")
            typed_cancel = cast(Callable[[], Awaitable[str]], cancel)
            return await typed_cancel()
        if op == "set_model":
            return await h.set_model(str(frame.get("provider", "")), str(frame.get("model_id", "")))
        if op == "set_effort":
            return await h.set_effort(str(frame.get("effort", "")))
        if op == "complete_aside":
            complete_aside = getattr(h, "complete_aside", None)
            if not callable(complete_aside):
                raise ValueError("this owner cannot run off-record requests")
            result = complete_aside(list(frame.get("turns") or []))
            if not inspect.isawaitable(result):
                raise ValueError("owner complete_aside operation must be awaitable")
            return await result
        if op == "slash":
            images = frame.get("images")
            if images:
                slash_images = getattr(h, "slash_images", None)
                if not callable(slash_images):
                    raise ValueError("this owner cannot route slash-command images")
                typed_slash_images = cast(
                    Callable[[str, str, list[dict[str, str]]], Awaitable[str]],
                    slash_images,
                )
                return await typed_slash_images(
                    str(frame.get("command", "")),
                    str(frame.get("args", "")),
                    images,
                )
            # Old daemon/reduced handles predate attachment-bearing slash ops;
            # preserve their two-argument call shape when no pixels ride the frame.
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
        if op == "adopt_aside":
            adopt = getattr(h, "adopt_aside", None)
            if not callable(adopt):
                raise ValueError("this owner cannot adopt an aside")
            result = adopt(list(frame.get("messages") or []))
            if not inspect.isawaitable(result):
                raise ValueError("owner adopt_aside operation must be awaitable")
            return await result
        if op == "peer_message":
            # Cross-session `lop send` delivery. Optional capability — an owner
            # host that predates peer messaging (or a non-interactive exec host
            # that never wired it) answers with a clear error, which the sender
            # surfaces as "this session cannot receive peer messages". Same
            # getattr guard as recall_steer above.
            receive = getattr(h, "receive_peer_message", None)
            if not callable(receive):
                raise ValueError("this session cannot receive peer messages")
            typed_receive = cast(Callable[..., Awaitable[str]], receive)
            return await typed_receive(
                frame["text"],
                mode=str(frame.get("mode", "mailbox")),
                wake=bool(frame.get("wake", False)),
                sender=frame.get("sender") or {},
            )
        if op == "stop":
            # PR 3 (the kill switch): the graceful rung of the stop ladder
            # (session/runtime/control.py). The plan is deny parked gates →
            # abort/dispose the session → release the lease → unpublish the
            # record → exit, executed by the host's ``request_stop`` hook
            # (OwnedSessionHandle.request_stop owns the ordering); all this
            # dispatch does is trigger it and ack, so the ack reaching the
            # caller means "the stop is underway", not "it finished" — the
            # ladder's timeout decides what a slow exit costs.
            #
            # Optional capability, probed: a host that predates the hook (a
            # reduced test handle, an old TUI bridge) answers with an error,
            # which the ladder treats as a scheduled miss and proceeds to
            # identity-confirmed SIGTERM. Additive on the wire — no
            # PROTOCOL_VERSION bump, for the same reason ``peer_message``
            # needed none: an old runtime's unknown-op error is exactly the
            # answer the ladder is built to continue from.
            request_stop = getattr(h, "request_stop", None)
            if not callable(request_stop):
                raise ValueError("this owner cannot stop itself gracefully")
            # Tell every attached viewer the disconnect they are about to see
            # is DELIBERATE, before the session goes away. Without this a
            # follower cannot distinguish a stop from owner death and its
            # recovery takes over the session a user just ended — republishing
            # a live record for a cold session (U2-4). Announced BEFORE the
            # hook runs because the hook's own teardown closes these sockets.
            # An old viewer ignores the unknown frame, so this stays additive.
            await self._broadcast({"op": "stopping", "session_id": self._record.session_id})
            result = request_stop()
            if inspect.isawaitable(result):
                result = await result
            # The host's own line when it gives one (a TUI owner names the
            # session and the reopen command), else the bare progress word.
            return str(result) if isinstance(result, str) and result else "stopping"
        raise ValueError(f"unknown op: {op!r}")

    async def _dispatch_payload(
        self, op: str, frame: dict[str, Any], locality: ClientLocality = "local"
    ) -> Any:
        """Structured-answer ops: the return value becomes the ``result`` data."""
        h = self._handle
        if op == "slash_result":
            run = getattr(h, "run_slash_authoritative", None)
            if not callable(run):
                raise ValueError("this owner cannot run typed slash results")
            args: list[Any] = [
                str(frame.get("command", "")),
                str(frame.get("args", "")),
                list(frame.get("images") or []),
            ]
            # ``locality`` is passed only to handles that accept it. A handle
            # is an injected collaborator — the TUI's, the runtime's, and test
            # doubles all implement this — so widening the call unconditionally
            # would break every implementation that has not been updated. The
            # probe keeps the parameter OPTIONAL in the protocol rather than
            # forcing a lockstep change, which is the same capability-probe
            # stance the ops above take with ``getattr``.
            if _accepts_locality(run):
                result = run(*args, locality=locality)
            else:
                result = run(*args)
            if inspect.isawaitable(result):
                result = await result
            return result
        if op == "credential":
            # Validated HERE because the payload path does not run
            # ``validate_control_frame`` the way ``_dispatch`` does (a
            # pre-existing gap for every payload op). Only this op is wired,
            # deliberately: it is the one that carries a secret, and widening
            # the validator over ``slash_result`` is a behaviour change for
            # every routed slash that belongs in its own review.
            from local_operator.mobile.types import validate_control_frame

            validate_control_frame(frame)
            # A DEDICATED op rather than a `slash_result` with the secret in
            # its `args` string. The value must never sit in a general-purpose
            # field that other code paths echo, log, or put in a transcript —
            # `args` is the same field that carries `/goal` text. Here the
            # secret has exactly one named home and one consumer.
            #
            # The store lives on the owner because that is where the agent's
            # bash commands run and read it from the environment; a follower
            # storing it locally would tell the model about a key that no tool
            # on the executing side can use.
            #
            # Optional capability, getattr-probed like `cancel` above: a
            # reduced handle or an older runtime answers the unknown-op error
            # rather than silently accepting a secret it will not store.
            credential = getattr(h, "credential_op", None)
            if not callable(credential):
                raise ValueError("this owner cannot hold session credentials")
            action = str(frame.get("action", ""))
            if action == "store" and locality == "remote":
                # Same locality rule the `/mcp` grant verbs apply
                # (``mcp/grants.py::REMOTE_GRANT_NOTICE``): a secret pasted on
                # a RELAYED client — a phone, in a future relay topology —
                # would be typed on a device the desktop's environment was
                # never meant to trust, and then injected into every bash
                # command here. Loopback attach clients are ``local`` by
                # construction, so no user today is refused; the gate exists
                # so the relay, when it lands, does not inherit a write it
                # never opted into (review round 1, R4). The read verbs stay
                # open: they return key NAMES only, never a value.
                return {"ok": False, "reason": "remote-client"}
            result = credential(
                action,
                str(frame.get("key", "")),
                str(frame.get("value", "")),
            )
            if inspect.isawaitable(result):
                result = await result
            return result
        if op == "cancel_subagents":
            cancel = getattr(h, "cancel_subagents_count", None)
            if not callable(cancel):
                raise ValueError("this owner cannot cancel subagents")
            result = cancel()
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, int) else 0
        if op == "job_trajectory":
            # The other half of the frame-size fix: the attach snapshot omits
            # trajectories, so a viewer opening a child page pulls that one
            # job's window here, in pages. Optional capability — an older
            # runtime answers unknown-op and the viewer degrades to "trajectory
            # unavailable" rather than rendering the child as empty.
            fetch = getattr(h, "job_trajectory", None)
            if not callable(fetch):
                raise ValueError("this owner cannot serve job trajectories")
            job_id = str(frame.get("job_id") or "")
            if not job_id:
                raise ValueError("job_id must be a non-empty string")
            offset = max(0, int(frame.get("offset") or 0))
            requested = int(frame.get("limit") or _TRAJECTORY_PAGE_MAX)
            limit = max(1, min(requested, _TRAJECTORY_PAGE_MAX))
            result = fetch(job_id, offset, limit)
            if inspect.isawaitable(result):
                result = await result
            return result
        raise ValueError(f"unknown op: {op!r}")

    # -- v5 frontend state relay ----------------------------------------------

    def _relay_frontend_to(self, conn: _ClientConn, data: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or self._closed.is_set():
            return
        try:
            loop.call_soon_threadsafe(self._relay_frontend_to_on_loop, conn, data)
        except RuntimeError:
            pass

    def _relay_frontend_to_on_loop(self, conn: _ClientConn, data: dict[str, Any]) -> None:
        if id(conn.writer) not in self._clients:
            return
        from local_operator.session.frontend_state import filter_update_trajectories

        # Per-connection, and applied on THIS loop rather than at the producer:
        # one canonical update fans out to every client, each of which has its
        # own open child page (or none).
        data = filter_update_trajectories(data, conn.watched_jobs.__contains__)
        if not conn.frontend_ready:
            if len(conn.frontend_pending) >= _EVENT_QUEUE_MAX:
                # A join that cannot install its boundary before this many
                # canonical edges is already stale. Drop and let it reconnect
                # to one fresh snapshot rather than retain an unbounded suffix.
                self._drop_client(conn)
                return
            conn.frontend_pending.append(data)
            return
        self._enqueue_client_frame(conn, {"op": "frontend_update", "data": data})

    def _enqueue_client_frame(self, conn: _ClientConn, frame: dict[str, Any]) -> None:
        """Queue one state/event frame on the connection's sole ordered FIFO."""
        try:
            conn.event_queue.put_nowait(frame)
        except asyncio.QueueFull:
            # A full FIFO is not always a slow CLIENT: a provider can hand the
            # session a whole turn of token deltas in one loop tick, which
            # overflows any bound before the drain writes a byte. Those frames
            # are losslessly coalescible (each ``message_update`` carries the
            # full accumulated message and its append-only ``delta``), so
            # compact first and drop only a client that stays full — the
            # genuinely slow reader the bound exists for.
            compacted = self._compact_event_queue(conn)
            if compacted:
                try:
                    conn.event_queue.put_nowait(frame)
                    compacted = True
                except asyncio.QueueFull:
                    compacted = False
            if not compacted:
                self._drop_client(conn)
                return
        if conn.event_writer_task is None:
            task = asyncio.create_task(self._drain_event_queue(conn))
            conn.event_writer_task = task
            self._event_sends.add(task)
            task.add_done_callback(self._event_sends.discard)

    # -- v4 event relay --------------------------------------------------------

    def _relay_event(self, data: dict[str, Any]) -> None:
        """Thread-safe relay entry: events fire on the HOST's thread (the
        Textual loop for a TUI owner), and everything relay-ordered must run
        on the runtime loop. ``call_soon_threadsafe`` from one producer
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
        recipients = [
            conn
            for conn in self._clients.values()
            if conn.kind == "attach" and conn.wants_events and conn.events_ready
        ]
        if not recipients:
            return
        frame = {"op": "event", "data": data}
        for conn in recipients:
            # Raw events and canonical deltas share this FIFO. Their producer
            # callbacks are scheduled in session order, so a follower cannot
            # observe transcript animation ahead of the state edge that caused it.
            self._enqueue_client_frame(conn, frame)

    def _compact_event_queue(self, conn: _ClientConn) -> bool:
        """Merge runs of same-message ``message_update`` frames in place.

        Lossless by construction: the later event's ``message`` already
        contains the earlier one's text, and concatenating ``delta`` preserves
        the append contract UIs rely on. Returns whether any room was freed.
        Runs synchronously on the runtime loop, so the drain task cannot
        observe a half-compacted queue.
        """
        frames: list[dict[str, Any]] = []
        while True:
            try:
                frames.append(conn.event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            conn.event_queue.task_done()
        compacted: list[dict[str, Any]] = []
        for frame in frames:
            previous = compacted[-1] if compacted else None
            if (
                previous is not None
                and frame.get("op") == "event"
                and previous.get("op") == "event"
            ):
                data = frame.get("data") or {}
                prior = previous.get("data") or {}
                if (
                    data.get("type") == "message_update"
                    and prior.get("type") == "message_update"
                    and (data.get("message") or {}).get("id")
                    == (prior.get("message") or {}).get("id")
                ):
                    merged = dict(data)
                    merged["delta"] = str(prior.get("delta", "")) + str(data.get("delta", ""))
                    compacted[-1] = {"op": "event", "data": merged}
                    continue
            compacted.append(frame)
        for frame in compacted:
            conn.event_queue.put_nowait(frame)
        return len(compacted) < len(frames)

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
        runtime-loop tick — pushes are snapshots, so intermediate states
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
        ordinary = self._projection_payload()
        await asyncio.gather(
            *(
                self._send_to(conn, self._projection_frame(conn, ordinary))
                for conn in list(self._clients.values())
            )
        )

    def _projection_payload(self) -> dict[str, Any]:
        """The broadcast frame, capped to the soft size limit.

        The daemon's control reader drops any line past 1 MB, so an oversized
        projection is a silently lost repaint — and a flood of them starves
        the daemon loop for every other session. ``cap_projection_frame``
        degrades optional payload tiers (subagent text previews, transcript
        expand details, then the transcript tail) until the frame fits, so a
        busy deep-roster session degrades gracefully instead of wedging the
        whole relay. The projection itself is never mutated.
        """
        from local_operator.mobile.projection import cap_projection_frame

        sink = self._projection_sink
        projection = sink.projection if sink is not None else self._handle.session_projection_seed
        data, degraded = cap_projection_frame(projection)
        if degraded and not self._frame_cap_warned:
            self._frame_cap_warned = True
            logger.warning(
                "session runtime: projection frame for session %s exceeded the "
                "soft cap; degrading optional payload tiers to fit",
                self._record.session_id,
            )
        elif not degraded:
            self._frame_cap_warned = False
        return {"op": "projection", "data": data}

    def _projection_frame(self, conn: _ClientConn, ordinary: dict[str, Any]) -> dict[str, Any]:
        # Projection is exclusively the mobile renderer. Full terminal clients
        # authenticate with this welcome but consume no semantic overlays from it.
        return ordinary

    async def _push_to(self, conn: _ClientConn) -> None:
        """The welcome form of a push: one full projection to one connection."""
        await self._send_to(conn, self._projection_frame(conn, self._projection_payload()))

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

    def _ensure_projection_sink(self) -> ProjectionSink:
        """Return the sink, building the default :class:`ProjectionFold` on
        first need. Idempotent; the counter records only builds this runtime
        performed itself, never an injected sink."""
        sink = self._projection_sink
        if sink is None:
            sink = ProjectionFold(self._handle.session_projection_seed)
            self._projection_sink = sink
            self.projection_sinks_built += 1
            logger.info(
                "session runtime: built projection fold for session %s",
                self._record.session_id,
            )
        return sink

    @property
    def record(self) -> SessionRecord:
        """The discovery record this runtime publishes.

        Read-only by intent — the runtime owns every mutation (the port is
        stamped when the listener binds, the heartbeat rewrites the liveness
        fields), and a caller that reassigned it would leave the publisher
        writing a record nobody else holds.

        Exposed because a host that starts a runtime and must then TELL
        someone where it is (``exec --control`` prints the endpoint to stderr)
        otherwise has to re-read the file the runtime just wrote, or reach for
        ``_record``. Deliberately not the control key by a separate accessor:
        the key rides the record, and the record's 0600 file is the whole
        authorization model, so nothing should be encouraged to copy it out.
        """
        return self._record

    @property
    def projection_sink(self) -> ProjectionSink | None:
        """The sink in use, or ``None`` while no consumer has needed one."""
        return self._projection_sink

    @property
    def fold(self) -> ProjectionFold:
        """The default fold, built on demand. Handles that reach for this want
        the full :class:`ProjectionFold` surface (subagent details, todos);
        an injected sink that is not one is a programming error here."""
        sink = self._ensure_projection_sink()
        if not isinstance(sink, ProjectionFold):
            raise TypeError("an injected projection sink is not a ProjectionFold")
        return sink

    def set_pending(self, pending: Any | None) -> None:
        """Set mobile pending state and canonical gate state for reduced hosts.

        Production TUI handles publish directly when their real widget mounts;
        this bridge keeps owned/reduced handles on the same contract.
        """
        self._ensure_projection_sink().set_pending(pending)
        frontend = getattr(self._handle, "_frontend", None)
        mutate = getattr(frontend, "mutate", None)
        if callable(mutate):
            mutate(pending_gate=pending.to_json() if pending is not None else None)
        self._schedule_push()


#: The pre-move name. ``Registrant`` was mobile-era vocabulary for the same
#: object; it stays bound here (and re-exported from
#: ``local_operator.mobile.registrant``) so the rename costs no call site and
#: an out-of-tree caller keeps working. New code should use ``RuntimeServer``.
Registrant = RuntimeServer

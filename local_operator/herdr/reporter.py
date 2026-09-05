"""The reporter: one pane, one worker thread, one strictly ordered stream.

What is reported is the :class:`HerdrState` of this pane's session. How it is
reported is the documented Herdr CLI (``herdr pane report-agent`` /
``release-agent``), chosen over the raw socket for the same reason
``multiplexer/cmux.py`` shells out rather than speaking JSON-RPC itself: the
CLI is the contract Herdr documents for custom hooks, and it absorbs protocol
changes the socket would expose. ``HERDR_SOCKET_PATH`` is noted in
``terminals.py`` for a future backend and is not read here.

THE ORDERING PROBLEM, AND WHY ``--seq`` IS ASSIGNED ON THE CALLER'S THREAD
--------------------------------------------------------------------------
Herdr keeps, per pane and per ``--source``, the highest ``--seq`` it has seen
and silently ignores any report carrying a lower one ("accepted by the API
but ignored by pane state"). That is the mechanism this module leans on to
make out-of-order delivery harmless — but it only works if the number
reflects the order the STATE CHANGED in, not the order the subprocesses
happened to finish in. So the sequence number is taken under a lock at the
moment :meth:`HerdrReporter.report` is called, before the call is queued, and
the single worker thread then executes the queue in that order. Two workers,
or a seq assigned inside the worker, would let a ``working`` overtake the
``idle`` that followed it and leave the panel spinning on a finished turn.

WHY THE SEQUENCE IS NOT A COUNTER FROM ONE
------------------------------------------
Measured against Herdr 0.8.2: the per-source high-water mark PERSISTS across
``release-agent`` and across processes. A process that releases at seq 12 is
followed, in the same pane, by a ``/reload`` re-exec or a quit-and-relaunch
whose first report is seq 1 — and every report that process ever makes is
ignored, forever, with no error anywhere. The sequence is therefore anchored
to the wall clock in microseconds, which is monotonic across processes in the
same pane by construction, and ``max(previous + 1, clock)`` keeps it strictly
increasing within a process even if the clock steps backwards. The clock is
injectable, so tests observe the plain ``1, 2, 3`` the contract is easiest to
read in.

WHY A WORKER THREAD AND NOT A DETACHED ``Popen`` PER CALL
---------------------------------------------------------
A detached spawn per transition would be simpler, but it is unordered: two
spawns a millisecond apart race to the socket. The worker gives ordering
for free, bounds the process to one Herdr subprocess at a time, and — like
``SessionBroadcast`` — keeps the event loop off every subprocess wait. It is
a daemon thread so a wedged ``herdr`` cannot hold the interpreter open; the
exit drain below is what makes that safe for the release.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import weakref
from typing import Callable, Literal, Mapping, Sequence

from local_operator.terminals import HERDR_BIN_ENV, HERDR_PANE_ENV, is_herdr

logger = logging.getLogger(__name__)

EnvMap = Mapping[str, str]

#: Herdr's lifecycle vocabulary for a pane's agent. ``unknown`` is in the
#: type because Herdr accepts it, but this reporter never emits it: it means
#: "present but unclassifiable", and every state this app is in classifies.
#: A turn that ends in an error is ``idle`` — the user's turn again.
HerdrState = Literal["idle", "working", "blocked", "unknown"]

#: The ``--source`` every call carries. ``custom:`` is the namespace Herdr
#: documents for hooks that are not official integrations; the charset is
#: ASCII letters, digits and ``:._-``, at most 80 characters.
HERDR_SOURCE = "custom:local-operator"

#: The ``--agent`` label, shown in the Agents panel row. Must match
#: ``[a-z][a-z0-9_-]{0,31}``.
HERDR_AGENT = "local-operator"

#: Environment kill switch, mirroring ``LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME``
#: and ``LOCAL_OPERATOR_NO_TERMINAL_TITLE``. For a recording, a CI job, or a
#: session opened in a pane whose Agents row belongs to something else.
_ENV_DISABLE = "LOCAL_OPERATOR_NO_HERDR"

#: How long one ``herdr`` call may take before it is abandoned. The TUI never
#: waits on it (the worker does), but a wedged socket must not leak a process
#: per transition either. The same figure as ``multiplexer.cmux.CALL_TIMEOUT_S``.
CALL_TIMEOUT_S = 5.0

#: Worst-case delay a user can experience at interpreter exit because of the
#: release. One bounded join per process, shared by every reporter, never on
#: the event loop — ``atexit`` runs on the main thread after ``on_unmount``
#: has returned. A healthy ``herdr`` releases in one subprocess spawn; the
#: bound is only reached against a wedged socket, and a release that outwaits
#: it leaves a row Herdr will reconcile itself when the pane process is gone.
EXIT_DRAIN_TIMEOUT_S = 2.0

#: ``(subcommand, argv)``. ``argv`` is the complete command line, binary
#: first; ``subcommand`` is repeated so a recording fake can assert on it
#: without parsing. Raises on failure — the worker is the one place that
#: catches, so an injected fake that raises proves failure isolation.
Invoker = Callable[[str, Sequence[str]], None]

#: Sequence-number source. Epoch MICROseconds rather than milliseconds so a
#: process that transitions faster than the clock ticks still has headroom
#: before ``max(previous + 1, clock)`` has to run ahead of it — and still
#: ~1.8e15, four orders of magnitude inside Herdr's ``u64``.
Clock = Callable[[], int]


def _default_clock() -> int:
    return time.time_ns() // 1_000


def _source(env: EnvMap | None) -> EnvMap:
    return os.environ if env is None else env


def herdr_reporting_enabled(env: EnvMap | None = None) -> bool:
    """Whether this process may report at all (the kill switch is unset).

    An environment gate only, with no config-flag counterpart — see the
    package docstring for why this matches the multiplexer's switch rather
    than the terminal title's two-tier one.
    """
    return not (_source(env).get(_ENV_DISABLE) or "").strip()


def herdr_binary(env: EnvMap | None = None) -> str | None:
    """The ``herdr`` CLI to call, or None when there is nothing to call.

    ``HERDR_BIN_PATH`` first: Herdr exports the path of the binary that spawned
    the pane, and that binary's protocol version is by definition the running
    server's. A ``herdr`` on PATH is the fallback for a pane whose environment
    was scrubbed (a ``env -i`` wrapper, a container that forwards only the
    ids) and is the documented alternative. Neither resolving is the common
    case outside Herdr and is not an error.

    The exported path is checked for executability rather than trusted: the
    markers are inherited across an ssh hop into a host where that path does
    not exist, and a missing binary should read as "no Herdr here" rather
    than as a spawn failure per transition.
    """
    source = _source(env)
    exported = (source.get(HERDR_BIN_ENV) or "").strip()
    if exported and os.path.isfile(exported) and os.access(exported, os.X_OK):
        return exported
    return shutil.which("herdr")


def state_from_title(title_state: str) -> HerdrState:
    """Translate the terminal title's run state into Herdr's vocabulary.

    The title is the ONE derivation of external state this app keeps (see
    ``StatusLine._title_state``), so this is a translation and not a second
    derivation. ``attention`` — a tool approval or an ``ask`` waiting on the
    user — is Herdr's ``blocked`` exactly. ``failed`` is the title's mark for
    "the last turn errored", which for the panel's purposes is the user's turn
    again: ``idle``, never ``unknown``.
    """
    if title_state == "attention":
        return "blocked"
    if title_state == "working":
        return "working"
    return "idle"


def _run_cli(subcommand: str, argv: Sequence[str]) -> None:
    """The production invoker: one ``herdr`` subprocess, bounded, no shell.

    Raises on anything the worker should log — a spawn failure, a timeout, a
    non-zero exit — so that the fake invokers in tests and this one share the
    same failure shape.
    """
    completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
        list(argv),
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_S,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"herdr {subcommand} exited {completed.returncode}: {completed.stderr[:200]}"
        )


class HerdrReporter:
    """Reports one pane's lifecycle state, in order, off the event loop.

    Construct through :func:`start_reporter`, which decides whether there is
    anything to report to; a directly constructed reporter is always active
    (tests build them that way, with a fake invoker).

    Thread-safety: :meth:`report`, :meth:`release` and :meth:`set_session_id`
    may be called from any thread. The event loop calls them in production and
    never blocks — each one takes a lock held for microseconds, appends to a
    queue and returns.
    """

    def __init__(
        self,
        *,
        pane_id: str,
        binary: str,
        session_id: str | None = None,
        invoker: Invoker | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._pane_id = pane_id
        self._binary = binary
        self._session_id = (session_id or "").strip() or None
        self._invoker: Invoker = invoker or _run_cli
        self._clock: Clock = clock or _default_clock
        # Guards `_last`, `_seq` and `_session_id`: the seq must be minted in
        # the same critical section that decides the report is not a
        # duplicate, or two callers could both pass the de-dupe and enqueue
        # the same state twice in either order.
        self._lock = threading.Lock()
        self._last: HerdrState | None = None
        self._seq = 0
        #: Every call ever enqueued, in seq order. `None` is the worker's
        #: stop sentinel and is enqueued exactly once, by `release`.
        self._queue: queue.SimpleQueue[tuple[str, tuple[str, ...]] | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        # Latched by `release` before the release call is queued, and what
        # makes `release` exactly-once and every later `report` a no-op. An
        # Event rather than a bool under `_lock` so `released` is readable
        # from any thread without contending with a caller mid-enqueue.
        #
        # Reports ALREADY queued ahead of the release are still delivered, in
        # order. Skipping them would make the emitted stream depend on how
        # close to quit the last transition happened, and against a wedged
        # binary it saves nothing the bounded exit drain does not already cap.
        self._released = threading.Event()

    # -- introspection (tests, diagnostics) --------------------------------

    @property
    def pane_id(self) -> str:
        return self._pane_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def last_state(self) -> HerdrState | None:
        """The state most recently ENQUEUED (not necessarily delivered)."""
        return self._last

    @property
    def released(self) -> bool:
        return self._released.is_set()

    # -- lifecycle -----------------------------------------------------------

    def set_session_id(self, session_id: str | None) -> None:
        """Change the ``--agent-session-id`` metadata, e.g. on a ``/new`` swap.

        Clears the de-dupe so the NEXT report goes out even if the state is
        unchanged: the pane's row is the same, the process is the same, only
        the session behind it moved, and Herdr learns that from the next
        report rather than from a release-and-re-report that would flash the
        row empty.
        """
        with self._lock:
            cleaned = (session_id or "").strip() or None
            if cleaned == self._session_id:
                return
            self._session_id = cleaned
            self._last = None

    def report(self, state: HerdrState) -> None:
        """Queue a ``report-agent`` for ``state``, unless it is a duplicate.

        Cheap enough for the 12.5 Hz spinner tick that reaches it through
        ``StatusLine.refresh``: one comparison under a lock in the common case
        and nothing else. Never raises.
        """
        if self._released.is_set():
            return
        with self._lock:
            if state == self._last:
                return
            self._last = state
            seq = self._next_seq_locked()
            session_id = self._session_id
        argv = self._argv(
            "report-agent",
            "--state",
            state,
            "--seq",
            str(seq),
            *(("--agent-session-id", session_id) if session_id else ()),
        )
        self._enqueue(("report-agent", argv))

    def release(self) -> None:
        """Queue the ``release-agent``, exactly once, and stop the worker after it.

        Idempotent and safe from any thread — the event loop on ``on_unmount``,
        and the exit drain from ``atexit``. Returns immediately: the call runs
        on the worker, and the exit drain is what guarantees it lands before
        the interpreter is gone.
        """
        # `Event.set` is not test-and-set, so the once-only decision needs the
        # lock; the latch is still an Event so the worker can read it without
        # contending for that lock mid-subprocess.
        with self._lock:
            if self._released.is_set():
                return
            self._released.set()
            seq = self._next_seq_locked()
        argv = self._argv("release-agent", "--seq", str(seq))
        self._enqueue(("release-agent", argv))
        self._queue.put(None)

    def join(self, timeout: float = EXIT_DRAIN_TIMEOUT_S) -> None:
        """Wait for the worker to drain, bounded. Tests and the exit drain only.

        Never called from the event loop: a call parked in a subprocess
        timeout would stall the TUI for exactly as long as this waits.
        """
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    # -- internals -----------------------------------------------------------

    def _next_seq_locked(self) -> int:
        # See the module docstring for why this is not `+= 1` alone.
        self._seq = max(self._seq + 1, self._clock())
        return self._seq

    def _argv(self, subcommand: str, *rest: str) -> tuple[str, ...]:
        return (
            self._binary,
            "pane",
            subcommand,
            self._pane_id,
            "--source",
            HERDR_SOURCE,
            "--agent",
            HERDR_AGENT,
            *rest,
        )

    def _enqueue(self, item: tuple[str, tuple[str, ...]]) -> None:
        self._queue.put(item)
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        # Started on the first call rather than in the constructor so a
        # reporter that never reports (built and discarded) costs no thread.
        # The exit drain is registered at the same moment, and BEFORE the
        # thread exists, so an interpreter exit racing the start still finds
        # this reporter in the registry.
        with self._lock:
            if self._thread is not None:
                return
            _LIVE_REPORTERS.add(self)
            _register_exit_drain()
            thread = threading.Thread(target=self._run, name="lop-herdr-report", daemon=True)
            self._thread = thread
        thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            subcommand, argv = item
            try:
                self._invoker(subcommand, argv)
            except Exception:  # noqa: BLE001 — best-effort by contract
                logger.debug("herdr %s failed", subcommand, exc_info=True)


def start_reporter(
    session_id: str | None = None,
    *,
    env: EnvMap | None = None,
    invoker: Invoker | None = None,
    clock: Clock | None = None,
) -> HerdrReporter | None:
    """A reporter for this pane, or None when there is nothing to report to.

    None is the common case and not an error: not inside Herdr, the kill
    switch, or no resolvable ``herdr`` binary. The caller is expected to hand
    the result to ``StatusLine.set_herdr_reporter``, which emits the initial
    state — this function itself sends nothing, so the first report carries
    whatever the band's state actually is rather than an assumed ``idle``.

    Never raises: this runs at session adoption, where an exception would cost
    the user their session for the sake of a sidebar row.
    """
    try:
        source = _source(env)
        if not herdr_reporting_enabled(source):
            return None
        if not is_herdr(source):
            return None
        binary = herdr_binary(source)
        if binary is None:
            logger.debug("inside Herdr but no herdr binary is resolvable; not reporting")
            return None
        pane_id = (source.get(HERDR_PANE_ENV) or "").strip()
        reporter = HerdrReporter(
            pane_id=pane_id,
            binary=binary,
            session_id=session_id,
            invoker=invoker,
            clock=clock,
        )
        logger.debug("reporting lifecycle state to Herdr pane %s", pane_id)
        return reporter
    except Exception:  # noqa: BLE001 — must never break session startup
        logger.debug("herdr reporter failed to start", exc_info=True)
        return None


def release_reporter(reporter: HerdrReporter | None) -> None:
    """Release the pane's row on a clean exit. Safe with None, never raises.

    The release itself runs on the reporter's worker, and the exit drain
    guarantees it lands before the interpreter is gone — see
    :func:`_register_exit_drain` for why that is ``atexit`` and not a join
    here. A join here would be on the Textual event loop.
    """
    if reporter is None:
        return
    try:
        reporter.release()
    except Exception:  # noqa: BLE001 — best-effort by contract
        logger.debug("herdr release failed", exc_info=True)


# ---------------------------------------------------------------------------
# Exit drain
# ---------------------------------------------------------------------------

_EXIT_DRAIN_LOCK = threading.Lock()
_exit_drain_registered = False

#: Every reporter that has started its worker in this process. Weak, so a
#: reporter dropped before exit is not kept alive to be drained for nothing.
#: Several reporters per process happens only in tests; production has one
#: per pane, and a pane is a process.
_LIVE_REPORTERS: weakref.WeakSet["HerdrReporter"] = weakref.WeakSet()


def _register_exit_drain() -> None:
    """Make sure the release survives interpreter exit — and is issued at all.

    WHY THIS EXISTS
    ---------------
    The worker is a daemon thread, and daemon threads are killed at
    interpreter exit without running what is left of their target. Without
    this, ``release()`` on quit would return in microseconds having only
    QUEUED the call; the interpreter then exits and the pane's row keeps
    saying ``idle`` for a process that no longer exists. And an exit that
    never reached ``on_unmount`` at all (an exception unwinding out of Textual)
    would never even queue it — so this drain also ISSUES the release for any
    reporter still unreleased, which is the "abrupt exit still releases" half
    of the contract.

    WHY ``atexit`` AND NOT A JOIN IN ``release``
    --------------------------------------------
    A join in ``release`` would sit on the Textual event loop (``on_unmount``
    is a coroutine) for as long as a wedged ``herdr`` takes to time out —
    the exact freeze ``SessionBroadcast.stop`` documents removing.
    ``atexit`` runs on the main thread after the loop is gone.

    EVERY exec-shaped exit skips ``atexit``: a hard crash, ``os._exit``, and
    the POSIX re-exec behind ``/reload`` and ``/update`` (``reexec.py``
    replaces the image with ``os.execvpe``). That is harmless here for the
    same reason it is for the multiplexer: the re-exec has already queued its
    release in ``on_unmount`` before ``cli.py`` calls ``replace_self``, and
    whether or not that subprocess lands before or after the successor's
    first report, the clock-anchored ``--seq`` makes Herdr keep the newer
    one. A crash leaves the row for Herdr's own process-exit reconciliation.

    Registered once per process, not once per reporter.
    """
    global _exit_drain_registered
    with _EXIT_DRAIN_LOCK:
        if _exit_drain_registered:
            return
        _exit_drain_registered = True
    atexit.register(_drain_at_exit)


def _drain_at_exit() -> None:
    """Release every live reporter and join its worker, bounded. Never raises."""
    reporters = tuple(_LIVE_REPORTERS)
    for reporter in reporters:
        try:
            reporter.release()
        except Exception:  # noqa: BLE001 — an exit path must never raise
            logger.debug("exit release failed", exc_info=True)
    deadline = time.monotonic() + EXIT_DRAIN_TIMEOUT_S
    for reporter in reporters:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            reporter.join(timeout=remaining)
        except Exception:  # noqa: BLE001 — an exit path must never raise
            logger.debug("exit release drain failed", exc_info=True)

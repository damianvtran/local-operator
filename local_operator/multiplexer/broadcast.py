"""When a binding is published, when it is withdrawn, and what it may contain.

This module owns the SAFETY rules the package docstring states, so that no
backend and no call site has to re-derive them:

* only a user's own session is ever published (never a subagent's child);
* the published command is restore-and-idle and can carry nothing else;
* every publication is best-effort, off the caller's thread, and silent;
* a clean exit withdraws the binding, and that withdrawal is FINAL.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import time
import weakref
from pathlib import Path

from local_operator.multiplexer.cmux import REASSERT_INTERVAL_S
from local_operator.multiplexer.registry import active_backend
from local_operator.multiplexer.types import (
    EnvMap,
    MultiplexerBackend,
    SessionBinding,
    env_or_process,
)

logger = logging.getLogger(__name__)

#: How often the loop re-checks whether a not-yet-resumable session has become
#: resumable. This cadence runs ONLY while :func:`is_resumable_session` is
#: False, and in that state a pass really is a single ``stat``: ``_publish_once``
#: returns before it reaches the backend, so no subprocess is spawned. That is
#: what makes a five-second poll affordable. Keying it on resumability rather
#: than on publish success is deliberate — a session whose publish keeps failing
#: (socket refusing, surface closed) would otherwise stay on this cadence
#: forever and spawn one ``cmux rpc`` process every five seconds, per pane, for
#: the life of the session.
#: This is the delay between a cold session's first turn landing on disk and
#: its pane advertising a resume command, so it is deliberately short.
_PENDING_POLL_S = 5.0

#: Default grace for :meth:`SessionBroadcast.join`, which a SWAP (the
#: successor broadcast waiting out its predecessor's withdrawal) and the
#: tests both use. Long enough for a backend call already inside a subprocess
#: to finish and see the retire latch. ``stop`` itself never waits this out —
#: see :meth:`SessionBroadcast.stop`.
CALL_JOIN_TIMEOUT_S = 6.0

#: Bound on how long the successor broadcast waits for the OUTGOING
#: session's withdrawal before publishing anyway. This is the swap ordering
#: made safe: without a wait, a ``/new``/``/resume`` publishes the new binding
#: while the old one's retire is still in a subprocess, and whichever lands
#: last wins the pane — the pane then advertises nothing, which is the exact
#: failure this package exists to prevent.
#
#: It is a bound and not a join because the wait runs on the SUCCESSOR's own
#: timer thread, off the event loop, and must not become a hang there either:
#: a wedged multiplexer socket that ignores ``CALL_TIMEOUT_S``-bounded
#: subprocesses entirely (or a retire queued behind a publish already parked
#: in one) can outwait any fixed number, and the successor has to publish
#: eventually. What the timeout trades away is the wezterm corner case below,
#: and a publish that wins the pane late is recoverable by the re-assert — an
#: unreachable socket is refusing the successor's writes too.
SWAP_DRAIN_TIMEOUT_S = CALL_JOIN_TIMEOUT_S

#: Environment kill switch, mirroring ``LOCAL_OPERATOR_NO_TERMINAL_TITLE`` in
#: ``tui/terminal_title.py``. Wanted by anything that must not rewrite a pane's
#: resume binding: a recording, a CI job, or a session opened purely to read
#: someone else's transcript in a pane that already holds a real one.
_ENV_DISABLE = "LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME"

#: The flag that reopens a session, and the ONLY one a published command
#: carries. Named here so the restore-and-idle rule has one spelling: cmux and
#: the marker backends all build their command from :func:`resume_argv`.
RESUME_FLAG = "--resume"


def multiplexer_resume_enabled(env: EnvMap | None = None) -> bool:
    """Whether this process may publish anything at all.

    An environment gate only, with no config flag counterpart — unlike the
    terminal title, this writes nothing a user sees, so the reason to turn it
    off is always situational (this run, this pane) rather than a standing
    preference.
    """
    return not (env_or_process(env).get(_ENV_DISABLE) or "").strip()


def resume_executable() -> str:
    """Absolute path to the launcher that should reopen this session.

    ``sys.argv[0]`` and NOT ``sys.executable``: a ``lop`` process is really
    ``…/uv/tools/local-operator/bin/python3 …/.local/bin/lop``, so the
    interpreter path would restore a bare Python REPL. ``argv[0]`` is the
    ``lop`` script the user actually launched, which is also the path their
    cmux vault registration matches on.

    Resolved to an absolute path because a restore may run with a different
    PATH or cwd than the launch did (a crash restore runs from whatever shell
    the multiplexer spawns), and a relative ``argv[0]`` would then resolve to
    nothing.
    """
    launcher = sys.argv[0] if sys.argv else ""
    if launcher:
        resolved = Path(launcher).expanduser()
        try:
            if resolved.exists():
                return str(resolved.resolve())
        except OSError:
            pass
    # Nothing usable on argv[0] (an embedder, a frozen host): name the console
    # script and let the restoring shell's PATH find it. Better than a path
    # that is known not to exist.
    return "lop"


def resume_argv(session_id: str, executable: str | None = None) -> tuple[str, ...]:
    """The restore-and-idle launch line for ``session_id``.

    THIS IS A SAFETY BOUNDARY, not a convenience. The returned argv replays a
    transcript and then waits for the user; it carries no prompt, no
    ``--exec``, and nothing that continues an interrupted turn. A restore
    happens unattended and to every pane at once — typically after a crash
    nobody chose — so the worst case of a spurious restore has to be an idle
    session rather than a dozen agents resuming tool execution with no one
    watching. Every backend builds its command from here so no call site can
    opt out of that.
    """
    return (executable or resume_executable(), RESUME_FLAG, session_id)


def build_binding(
    session_id: str,
    *,
    cwd: str | None = None,
) -> SessionBinding | None:
    """Describe this session for publication, or None when it must not be.

    Returns None — rather than raising — for every "do not publish" case, so
    the caller has exactly one branch to write.
    """
    session_id = (session_id or "").strip()
    if not session_id:
        return None
    executable = resume_executable()
    return SessionBinding(
        session_id=session_id,
        executable=executable,
        argv=resume_argv(session_id, executable),
        cwd=cwd if cwd is not None else os.getcwd(),
    )


def is_user_owned_session(session_id: str) -> bool:
    """Whether this session is the USER's own, rather than a subagent's.

    A PERMANENT property, decided once: a child session is an ephemeral
    directory with the exact shape of a real conversation, and it runs in the
    SAME pane as its parent — so publishing one would overwrite the pane's
    binding and a crash restore would reopen a delegated code review in place
    of the user's own work. ``origin.json`` is the marker and
    :func:`~local_operator.resume.is_user_session` is the one reader of it (the
    same gate the ``/resume`` picker uses). cmux's omp integration carries the
    same guard, ``isNestedArtifactSession``, for the same reason.

    Kept separate from :func:`is_resumable_session` because the two refusals
    have different lifetimes, and conflating them is what would either spin a
    pointless timer for every subagent or never publish a cold session at all.
    """
    from local_operator.paths import config_dir
    from local_operator.resume import is_user_session

    try:
        return is_user_session(config_dir() / "sessions" / session_id)
    except OSError:
        logger.debug("session origin check failed", exc_info=True)
        return False


def is_resumable_session(session_id: str) -> bool:
    """Whether ``--resume`` would actually reopen this session yet.

    A TRANSIENT property, which is why it is re-checked before every publish
    rather than once at startup. ``--resume`` refuses an id whose transcript is
    not on disk — deliberately, so a typo cannot open an empty session that
    merely looks resumed — and the transcript file does not exist until the
    first turn is persisted. A session therefore starts life unpublishable and
    becomes publishable a turn later.

    Checking this only once, at startup, is the subtle version of the bug this
    whole feature exists to fix: every COLD session (which is most of them)
    would silently never publish, and the omission would surface only after a
    crash, when the user has already lost the work.
    """
    from local_operator.paths import config_dir
    from local_operator.resume import TRANSCRIPT_NAME

    try:
        return (config_dir() / "sessions" / session_id / TRANSCRIPT_NAME).is_file()
    except OSError:
        logger.debug("session transcript check failed", exc_info=True)
        return False


class SessionBroadcast:
    """Keeps one pane's binding published for as long as the session lives.

    Owns a daemon timer thread rather than an asyncio task on purpose: every
    call here ends in a subprocess, and the TUI's event loop must never be the
    thing waiting on a multiplexer socket. A daemon thread also cannot hold
    the process open if some exit path forgets to stop it.

    That rule covers the WITHDRAWAL as well as the publish: :meth:`stop` runs
    on the event loop (a ``/new`` swap, and quit) and hands the retire to a
    worker rather than making the call itself, so a wedged socket cannot
    freeze the TUI at exactly the moment this feature exists to survive.

    The re-assert exists because cmux caches its live-agent index for 60s and
    retires bindings judged against a stale snapshot — see
    :data:`local_operator.multiplexer.cmux.REASSERT_INTERVAL_S` for why that
    retirement is permanent and why the interval must exceed the TTL.
    """

    def __init__(
        self,
        binding: SessionBinding,
        backend: MultiplexerBackend,
        *,
        env: EnvMap | None = None,
        interval_s: float = REASSERT_INTERVAL_S,
        predecessor: SessionBroadcast | None = None,
    ) -> None:
        self._binding = binding
        self._backend = backend
        self._env = env_or_process(env)
        self._interval_s = interval_s
        # The broadcast this one REPLACES in the same pane, if any. Not read
        # after `start()` returns: the successor's timer thread drains it once
        # (below) and then the reference is dropped, so a chain of swaps can
        # never accumulate handles. Ordering the swap through the successor
        # rather than the app keeps the event loop out of it entirely — the
        # app calls `stop()` (prompt, returns immediately) and `start()`, and
        # the sequencing happens on a thread neither of them is waiting on.
        self._predecessor = predecessor
        self._drained = threading.Event()
        # Signals the timer to wake and exit. Also the mechanism that makes
        # `stop()` prompt rather than waiting out a 90s sleep.
        self._stopped = threading.Event()
        # Signals the timer to wake and exit. Also the mechanism that makes
        # `stop()` prompt rather than waiting out a 90s sleep.
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        # Survives `stop()` clearing `_thread`, so `join()` can still settle
        # it — and join has a production caller now: a SWAP's successor drains
        # its predecessor on the successor's own timer thread (see `_run`).
        self._timer_thread: threading.Thread | None = None
        # Serialises the publish/retire pair. Without it a re-assert already in
        # flight on the timer thread could land AFTER the clean-exit clear and
        # resurrect the binding the user just quit out of — the retirement bug
        # inverted, and invisible until their next new shell replayed a dead
        # session.
        #
        # HELD ACROSS A SUBPROCESS CALL, so nothing on the event loop may ever
        # wait on it. `stop()` deliberately touches only the lock-free latches
        # below; that is what keeps a wedged socket off the calling thread.
        self._call_lock = threading.Lock()
        # An Event, not a bool under `_call_lock`: `stop()` has to set the
        # latch without ever contending with a retire that is currently sitting
        # in a five-second subprocess timeout. Event set/is_set is atomic, so
        # the latch is readable and writable from any thread for free.
        self._retired = threading.Event()
        # Guards only the retire-dispatch bookkeeping. Held for microseconds
        # and NEVER across a backend call, so stop() cannot block on it.
        self._dispatch_lock = threading.Lock()
        self._retire_thread: threading.Thread | None = None
        #: Whether the session's transcript has appeared yet. This and NOT
        #: publish success is what selects the cadence: once a session can be
        #: resumed the loop drops to the re-assert interval whether or not the
        #: backend is answering, because retrying a failing publish every five
        #: seconds forever is subprocess churn rather than resilience.
        self._resumable = False

    def start(self) -> None:
        """Publish now, then keep re-asserting until :meth:`stop`.

        On a swap, called with ``predecessor`` set. The first publish then
        happens only after the predecessor's withdrawal has been drained, so
        the two cannot race for the pane — and the drain runs on THIS
        broadcast's timer thread, which is exactly the thread the event loop
        is not waiting on.
        """
        if self._thread is not None:
            return
        thread = threading.Thread(
            target=self._run,
            name="lop-multiplexer-resume",
            daemon=True,
        )
        self._thread = thread
        self._timer_thread = thread
        thread.start()

    def _publish_once(self) -> None:
        # Checked BEFORE the lock, not only under it. The retire worker holds
        # `_call_lock` for the whole of a backend call, so a re-assert that
        # arrives during a withdrawal would otherwise sit in the queue for a
        # subprocess timeout only to discover it must do nothing. The latch is
        # set synchronously by `stop`, so this early exit is not a race: it
        # sees every retirement that has been decided.
        if self._retired.is_set():
            return
        with self._call_lock:
            # Re-read under the lock: `stop` may have latched between the
            # check above and acquiring it. The clean-exit clear is FINAL.
            if self._retired.is_set():
                return
            # Re-checked on EVERY pass, not once at startup: a cold session has
            # no transcript until its first turn persists, so publishing a
            # binding now would advertise a command `--resume` still refuses.
            # The timer is what turns that into a delay rather than a silent
            # never (see `is_resumable_session`).
            if not is_resumable_session(self._binding.session_id):
                return
            # Recorded BEFORE the publish attempt and independently of its
            # result: the cadence question is "is there still something to wait
            # for?", and once the transcript exists the answer is no even if
            # the backend refuses every call.
            self._resumable = True
            try:
                self._backend.publish(self._binding, self._env)
            except Exception:  # noqa: BLE001 — best-effort by contract
                logger.debug("multiplexer publish failed", exc_info=True)

    def _run(self) -> None:
        # A swap must not let the outgoing session's withdrawal land after
        # the incoming session's publish: both name the same pane, so the
        # loser's write deletes the winner's. Waiting here is what keeps the
        # event loop out of it — this is the successor's own timer thread, a
        # thread nothing user-facing blocks on. The bound is
        # `SWAP_DRAIN_TIMEOUT_S` rather than a plain join because the
        # predecessor's retire may itself be queued behind a publish parked
        # in a `CALL_TIMEOUT_S` subprocess, and the successor must publish
        # eventually rather than hang behind a socket that never answers.
        #
        # `_drain_retire` and NOT `join`: `join` applies its timeout PER
        # WORKER over the retire worker and the timer, so a `join(timeout=6)`
        # here really bounded the successor's first publish at 12s — twice the
        # number this comment and `SWAP_DRAIN_TIMEOUT_S` both name (measured
        # 12.02s). Worse, it compounded down a swap chain: draining the
        # predecessor's TIMER means waiting out that timer's own drain of ITS
        # predecessor, so A→B→C paid B's drain of A as well. Only the
        # withdrawal carries swap ordering — the predecessor's timer is
        # `_stopped` and its `_publish_once` refuses under the `_retired`
        # latch, so it can no longer touch the pane and joining it buys
        # nothing. This is the same narrowing `_drain_retire` already
        # documents for the exit path.
        predecessor = self._predecessor
        if predecessor is not None:
            self._predecessor = None
            try:
                predecessor._drain_retire(  # noqa: SLF001 - same package
                    timeout=SWAP_DRAIN_TIMEOUT_S
                )
            except Exception:  # noqa: BLE001 — a swap must never fail on this
                logger.debug("predecessor withdrawal drain failed", exc_info=True)
        self._drained.set()
        # Published immediately so a RESUMED session's binding exists from the
        # first moment; a cold session no-ops here and lands on the poll below.
        self._publish_once()
        elapsed = 0.0
        while not self._stopped.wait(_PENDING_POLL_S):
            elapsed += _PENDING_POLL_S
            # Two cadences, one timer, switched on RESUMABILITY and not on
            # publish success. While the session has no transcript the pass is
            # a single stat that never reaches the backend, so five seconds is
            # cheap and makes a cold session's binding appear seconds after its
            # first turn. Once it is resumable only the re-assert matters, and
            # that must stay slower than cmux's 60s index TTL. Gating on
            # success instead would pin a session whose backend is refusing to
            # the fast cadence permanently, turning one unreachable socket into
            # a subprocess every five seconds in every pane.
            if self._resumable and elapsed < self._interval_s:
                continue
            elapsed = 0.0
            # Deliberately silent: this runs for the life of a session, and a
            # log line per pass would bury the debug log of anyone debugging
            # something else. Failures inside `_publish_once` are logged at
            # debug individually, which is enough to see a broken socket.
            self._publish_once()

    def stop(self, *, retire: bool = True) -> None:
        """Stop re-asserting; on a clean exit, withdraw the binding.

        ``retire=False`` leaves the binding in place, which is what a crash or
        a re-exec wants: the binding surviving is the entire feature. The
        default is to withdraw, because a session the user deliberately quit
        must not be replayed into their next shell.

        NON-BLOCKING, and that is a hard requirement rather than an
        optimisation. Both callers run on the Textual event loop — a ``/new``
        or ``/resume`` swap (``_adopt_session``) and quit (``on_unmount``) —
        so doing the retire here would park the whole TUI inside
        ``subprocess.run(timeout=5.0)`` whenever the multiplexer socket is
        unresponsive. That is precisely the post-crash window this feature
        exists for, so the freeze would land exactly when it hurts most
        (measured at 9.8s against a wedged backend). The retire therefore goes
        to a short-lived worker and this method returns immediately.

        Correctness does not depend on that worker running, let alone
        finishing: ``_retired`` is latched HERE, synchronously, before this
        returns, and ``_publish_once`` re-reads it as its first action. A
        re-assert can never resurrect a binding the user quit out of, whether
        or not the withdrawal itself succeeds. The withdrawal is best-effort
        like every other call in this package; the latch is the guarantee.

        Idempotent, and safe to call from any thread — including from an exit
        path that runs while a re-assert is mid-flight.
        """
        self._stopped.set()
        # Cleared so `start()` could run again, but kept on `_timer_thread` so
        # `join()` can still settle it in tests.
        self._thread = None
        if retire:
            # Latched before the worker is dispatched, so a re-assert already
            # waiting on `_call_lock` sees the retirement and does not
            # republish even if the retire call itself never lands.
            self._retired.set()
            self._dispatch_retire()
        # Deliberately NOT joined. The timer is a daemon thread that cannot
        # hold the process open, it wakes on `_stopped` immediately, and the
        # `_retired` latch already prevents it from doing anything meaningful
        # after this point. Joining here would reintroduce the event-loop
        # stall the dispatch above exists to avoid — a publish in flight can
        # sit in a subprocess timeout for seconds. The two callers that DO
        # need the withdrawal sequenced get it off the loop: a swap's
        # successor drains this broadcast on its own timer thread, and quit
        # is covered by the exit drain registered with the retire dispatch.

    def _dispatch_retire(self) -> None:
        """Run the backend retire off the caller's thread, at most once.
        A dedicated short-lived thread rather than the timer thread: the timer
        may be parked inside a publish subprocess for seconds, and the
        withdrawal should not queue behind it. Daemon, so a wedged socket
        cannot hold the process open — with the exit drain below making that
        safe: without it, interpreter exit killed the worker before the
        withdrawal ran, and a session the user deliberately quit stayed
        advertised in the pane until their next shell replayed it.
        """
        with self._dispatch_lock:
            if self._retire_thread is not None:
                return
            # Recorded before the thread exists so the exit drain can see a
            # dispatch that is still starting up.
            _RETIRE_REGISTRY.add(self)

            def _retire() -> None:
                # Takes `_call_lock` so the retire cannot interleave with a
                # publish mid-flight; because this is a worker, waiting on
                # that lock costs nothing the user can feel.
                with self._call_lock:
                    try:
                        self._backend.retire(self._binding, self._env)
                    except Exception:  # noqa: BLE001 — best-effort by contract
                        logger.debug("multiplexer retire failed", exc_info=True)

            worker = threading.Thread(
                target=_retire,
                name="lop-multiplexer-retire",
                daemon=True,
            )
            self._retire_thread = worker
            _register_exit_drain()
            worker.start()

    def join(self, timeout: float = CALL_JOIN_TIMEOUT_S) -> None:
        """Wait for the timer and any retire worker to finish.
        Called by tests, by teardown diagnostics, and by a SWAP: the successor
        broadcast's timer thread joins its predecessor before publishing, so
        the two bindings cannot race for the pane. The event loop is never a
        caller — both of its call sites (`_adopt_session`, `on_unmount`) use
        `stop()`/`retire_session()`, which never join.
        """
        with self._dispatch_lock:
            retire_worker = self._retire_thread
        for worker in (retire_worker, self._timer_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=timeout)

    def _drain_retire(self, *, timeout: float) -> None:
        """Join only the retire worker, bounded. Exit-path half of the drain.

        Deliberately narrower than :meth:`join`: the timer thread has nothing
        left to do once `_stopped` is set (and its `_publish_once` refuses
        under the `_retired` latch), so joining it at exit buys nothing. Only
        the withdrawal carries state the user can still be harmed by.
        """
        with self._dispatch_lock:
            worker = self._retire_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)


def broadcast_session(
    session_id: str,
    *,
    cwd: str | None = None,
    env: EnvMap | None = None,
    predecessor: SessionBroadcast | None = None,
) -> SessionBroadcast | None:
    """Start publishing ``session_id`` for this pane, if anything should be.
    Returns the handle to stop later, or None when there is nothing to do —
    no multiplexer, the kill switch is set, a subagent's session, or a
    session with no transcript yet. None is the common case and is not an
    error.

    ``predecessor`` is the broadcast this one replaces in the same pane, on a
    ``/new``/``/resume`` swap. The successor waits for its withdrawal before
    publishing (on the successor's own thread — never the event loop), which
    is what stops the outgoing clear from deleting the incoming binding on
    the marker backends that cannot scope a clear themselves. When this
    returns None the caller has already stopped the predecessor itself, so
    the withdrawal still happens; it is simply not sequenced against a
    successor that does not exist.

    Never raises. This is called from session startup, where an exception
    would cost the user their session for the sake of a bookkeeping write.
    """
    try:
        source = env_or_process(env)
        if not multiplexer_resume_enabled(source):
            return None
        backend = active_backend(source)
        if backend is None:
            return None
        # Only the PERMANENT refusal is decided here. Whether the transcript
        # exists yet is transient and belongs to each publish attempt, because
        # a session that is not resumable at startup becomes resumable one turn
        # later and must publish then.
        if not is_user_owned_session(session_id):
            return None
        binding = build_binding(session_id, cwd=cwd)
        if binding is None:
            return None
        broadcast = SessionBroadcast(binding, backend, env=source, predecessor=predecessor)
        broadcast.start()
        logger.debug("publishing resume binding to %s for %s", backend.name, session_id)
        return broadcast
    except Exception:  # noqa: BLE001 — must never break session startup
        logger.debug("multiplexer broadcast failed to start", exc_info=True)
        return None


#: Worst-case delay a user can experience at interpreter exit because of a
#: resume-binding withdrawal. One bounded join per process, shared by every
#: broadcast (see :func:`_register_exit_drain`), never on the event loop —
#: `atexit` runs on the main thread after `on_unmount` has already returned.
#:
#: The number is a bound, not an expectation: a healthy backend retires in
#: one subprocess spawn (single-digit milliseconds), so the join returns in
#: that. The bound only ever gets used when the multiplexer socket is wedged
#: mid-restart — the state where the alternative (a synchronous retire on the
#: event loop) froze the whole TUI for ~9.8s. A quit that outwaits it leaves
#: a stale marker, which is the package's explicitly allowed failure for the
#: crash case; the crash case cannot reach this code at all.
EXIT_DRAIN_TIMEOUT_S = 2.0

#: Registered once per process; guarded by `_EXIT_DRAIN_LOCK`.
_EXIT_DRAIN_LOCK = threading.Lock()
_exit_drain_registered = False

#: Every broadcast with a retire dispatched in this process. Weak, so a
#: broadcast that is garbage-collected before exit (a swap chain) is not kept
#: alive — and its already-finished worker is not joined for nothing.
_RETIRE_REGISTRY: weakref.WeakSet["SessionBroadcast"] = weakref.WeakSet()


def _register_exit_drain() -> None:
    """Make sure a dispatched withdrawal survives interpreter exit.

    WHY THIS EXISTS
    ---------------
    The retire worker is a daemon thread, and daemon threads are killed at
    interpreter exit without running their remaining code. Before this
    drain, `stop(retire=True)` on quit returned in microseconds having only
    STARTED the worker; the interpreter then exited and the withdrawal never
    ran, so the pane kept advertising a session the user had deliberately
    closed — and their next shell in that pane replayed it. A stale marker
    after a crash is the feature; a stale marker after a clean quit is the
    bug this function exists to close.

    WHY `atexit` AND NOT A JOIN IN `stop`
    -------------------------------------
    Joining in `stop` would put the wait on the Textual event loop (both real
    callers are coroutines), which is the ~9.8s freeze F1's fix removed.
    `atexit` handlers run on the main thread AFTER `on_unmount` has returned
    and the event loop is gone, so nothing user-facing is blocked by the
    join: the process is already on its way out.

    EVERY exec-shaped exit skips `atexit`, not just the Windows one. A hard
    crash does (no handlers run), `os._exit` does (it is the point of
    `os._exit`), and so does the POSIX re-exec: `reexec.py` replaces the
    process image with `os.execvpe`, and exec runs no `atexit` handler either.
    Naming Windows alone here would mislead the next reader working out which
    exits this drain actually covers.

    That is harmless for both shapes, for different reasons. A crash is the
    case where a surviving binding IS the feature. A re-exec has already
    dispatched its withdrawal before the exec happens — `_request_relaunch`
    exits through Textual, so `on_unmount` runs `_stop_multiplexer_broadcast`
    → `retire_session`, and `cli.py` only calls `replace_self` after
    `asyncio.run` has returned — and a relaunch wants the binding withdrawn
    anyway, because the successor process republishes its own.

    The handler is registered once for the process rather than once per
    broadcast: several panes' worth of broadcasts live in one process only in
    tests, but a chain of swaps produces one live broadcast plus a series of
    retired ones, and each would otherwise register its own handler.
    """
    global _exit_drain_registered
    with _EXIT_DRAIN_LOCK:
        if _exit_drain_registered:
            return
        _exit_drain_registered = True

    atexit.register(_drain_retires_at_exit)


def _drain_retires_at_exit() -> None:
    """Join every live retire worker, bounded. Never raises.

    Idempotent and cheap to call early: the registry of broadcasts with a
    dispatched retire is consulted rather than guessed at, so a process that
    never dispatched one pays nothing.
    """
    deadline = time.monotonic() + EXIT_DRAIN_TIMEOUT_S
    for broadcast in tuple(_RETIRE_REGISTRY):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            broadcast._drain_retire(timeout=remaining)  # noqa: SLF001 - same package
        except Exception:  # noqa: BLE001 — an exit path must never raise
            logger.debug("exit retire drain failed", exc_info=True)


def retire_session(broadcast: SessionBroadcast | None) -> None:
    """Withdraw a binding on a clean exit. Safe with None, never raises.

    The withdrawal itself runs on the broadcast's own retire worker, and the
    exit drain guarantees it lands before the interpreter is gone — see
    :func:`_register_exit_drain` for why that is `atexit` and not a join
    here. A join here would be on the Textual event loop.
    """
    if broadcast is None:
        return
    try:
        broadcast.stop(retire=True)
    except Exception:  # noqa: BLE001 — best-effort by contract
        logger.debug("multiplexer retire failed", exc_info=True)

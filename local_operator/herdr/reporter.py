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
happened to finish in. So the sequence number is minted **and the call
enqueued in the same critical section**, under ``_lock``, at the moment
:meth:`HerdrReporter.report` is called; the single worker thread then
executes the queue in that order. Two workers, or a seq assigned inside the
worker, would let a ``working`` overtake the ``idle`` that followed it and
leave the panel spinning on a finished turn.

MINTING AND ENQUEUEING MUST BE ONE ATOMIC STEP, not two. An earlier version
took the seq under the lock and then queued outside it, which is a weaker
promise than it looks: two callers could mint 1, 2 and enqueue 2, 1, so the
delivery order was not the mint order and this paragraph was false of the
code — measured at 80/200 trials under ``sys.setswitchinterval(1e-6)``.
Worse, a ``report`` that lost that race to a concurrent ``release`` was
delivered AFTER the ``release-agent`` carrying a HIGHER seq, which Herdr's
high-water mark cannot discard, leaving the pane's row alive for an exited
process (review round 1, A1/A2). ``release`` puts its call and the worker's
stop sentinel inside that same section, so nothing can be queued behind it,
and ``report`` re-tests the released latch inside the lock so a report that
loses the race is DROPPED rather than re-sent.

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

WHY ONE LOST DELIVERY USED TO FREEZE THE ROW FOREVER (RETRY)
-------------------------------------------------------------
Each transition enqueued exactly ONE subprocess, and the worker caught every
failure — non-zero exit, spawn error, the 5 s timeout — logged it at DEBUG
and DROPPED the item. That looks like the right "best effort" shape until it
is combined with the de-dupe in :meth:`HerdrReporter.report`, which updates
``_last`` at ENQUEUE time. The 12.5 Hz spinner tick keeps calling ``report``
with the same state for the whole of a turn, and every one of those calls is
de-duped away — so the state whose SINGLE delivery failed can never be sent
again. The observed symptom, reproduced deterministically with a wrapper
binary failing one ``--state working`` call: the terminal title spins for a
ten-minute turn while the Agents row still says ``idle``. Two sinks, one
derivation, and only one of them lossy.

So delivery is retried on the worker thread, in place, up to
``len(RETRY_BACKOFF_S)`` times with 0.5 s / 2.0 s / 8.0 s between attempts.
The geometry matters more than the numbers: the first retry is inside a
human's "did that register?" window, and the last lands ~10.5 s out, which
covers a Herdr server being restarted under a running pane without the
worker sitting on a wedged socket for a minute. The item stays at the HEAD
of the queue while it is retried, which preserves the module's ordering
invariant (mint order == delivery order) rather than merely not breaking it;
a retry that lands after a newer report would be harmless anyway, because
Herdr's high-water mark discards the lower seq. On exhaustion the item is
dropped and logged at WARNING, not DEBUG: a row that has stopped tracking a
live session is a user-visible defect, and DEBUG is where the original one
hid for a whole release.

That WARNING covers the MID-LIFE case — a pane still running when its retries
run out, which is the one where a stale row persists while somebody is there
to notice it. It is NOT reachable for a ``release-agent`` exhausting at exit,
and the prose used to promise otherwise (review round 2, M5). The arithmetic:
attempts land at ~0 s, ~0.5 s, ~2.5 s and ~10.5 s, so exhaustion is ~10.5 s
out — but the exit drain gives up after :data:`EXIT_DRAIN_TIMEOUT_S` (2 s),
and the interpreter then kills the daemon worker where it sits, mid-backoff,
before even the third attempt. Nothing survives to log the WARNING.

Loss on the exit path is therefore BOUNDED rather than reported: the drain
caps what it costs the user, and a row left behind falls to Herdr's own
reconciliation once the pane process disappears — the same fallback a hard
crash relies on (see :func:`_register_exit_drain`).

A report abandons its backoff the moment ``release`` latches — the release
queued behind it is the delivery that matters, and the row is about to be
gone. The ``release-agent`` itself does NOT abandon its backoff: nothing is
queued behind it, and it is the one call whose loss leaves a row describing
an exited process. Both are bounded at exit by the drain, which joins a
daemon thread for :data:`EXIT_DRAIN_TIMEOUT_S` and then lets the interpreter
go.

WHY A HEARTBEAT AND NOT JUST RETRY
-----------------------------------
Retry fixes a delivery that fails. It cannot fix a delivery that SUCCEEDED
and was then forgotten, and Herdr does not persist agent rows across a server
restart. A pane whose session is mid-turn emits no further transitions — the
spinner tick is de-duped — so a row lost to ``herdr`` restarting comes back
only at the next state change, which for a long turn is many minutes away,
and for an idle session is never.

The fix is a re-assertion heartbeat: a daemon thread that wakes every
``resync_interval_s`` (default 30 s) and re-sends the CURRENT state with a
fresh seq. It is not a second source of truth — it does not decide anything,
it asks the injected state provider (the band, through
``StatusLine.set_herdr_reporter``) for the same derivation ``report`` uses,
which keeps the single-derivation property this module is built on. 30 s is
chosen against what the row is for: a sidebar a human glances at, where
half a minute of staleness after an event as rare as a server restart is
invisible.

The cost is one subprocess per pane per 30 s, and it is NEW work rather than
a rounding error on existing work: comparing it against the 12.5 Hz tick
(review round 2, M3) was wrong, because that tick is de-duped and spawns
NOTHING for the whole of a turn. In spawns, which is the unit that costs
anything: an idle pane goes from 0 to 2 per minute — 2880 a day, per pane —
and a busy pane adds those same 2 a minute on top of its transitions. That
is the price of a row that comes back after a server restart, paid whether
or not one ever happens.

It cannot break ordering: the heartbeat mints its seq and enqueues in the
same ``_lock`` critical section every other caller uses (:meth:`_enqueue_report`
is that section, factored out so there is exactly one of it), so mint order is
still delivery order. Nor can it outrank a newer transition, which takes a
guard rather than the lock alone — see "THE STALE-READ RACE" below. It cannot
resurrect a released row for the same reason ``report`` cannot: it re-tests
the released latch INSIDE the lock, so a heartbeat racing a ``release`` is
dropped rather than delivered behind it with a higher seq. And it stops
promptly rather than up to an interval late, because the sleep is
``_released.wait(interval)`` and not ``time.sleep``.

The heartbeat deliberately does NOT de-dupe. Same state plus a new seq is
exactly the message that recreates a lost row, and suppressing it as a
duplicate would reintroduce the bug the retry above exists to fix.

THE STALE-READ RACE, AND WHY THE PROVIDER IS NOT CALLED UNDER THE LOCK
----------------------------------------------------------------------
The provider is read OUTSIDE ``_lock``, so the read and the mint are not one
atomic step, and a transition can land between them. Left alone that is a
real defect: the heartbeat samples ``working``, the turn ends, ``report``
mints ``idle`` at seq N, the heartbeat then mints its already-stale
``working`` at N+1 — and Herdr's high-water mark, which exists to discard
lower seqs, keeps the HIGHER stale one. Reproduced as a row reading
``working`` against a session that was ``idle`` (review round 2, M2).

The obvious fix — hold ``_lock`` across the provider call — is worse than the
bug. It puts caller-supplied code inside the one critical section ``report``
and ``release`` both need, so a provider that blocks stalls the whole
reporter: measured at a 29.9 s ``release()`` against a provider that slept,
which also blows the :data:`EXIT_DRAIN_TIMEOUT_S` bound the exit drain
promises. A contract saying "must be cheap" is not a bound; not calling user
code under the lock is.

So the heartbeat instead detects the race rather than preventing it. Every
mint increments ``_mints`` inside the critical section that mints seqs. The
heartbeat snapshots that counter BEFORE reading the provider and passes the
snapshot to :meth:`_enqueue_report`, which re-checks it under the lock and
drops the tick if it moved. Why that is sufficient: a stale read is harmful
only if a transition landed between the snapshot and the enqueue; every such
transition mints, so it bumps the counter, so the guard sees it and the
heartbeat defers to it — and deferring loses nothing, because that newer
transition carries the truth the heartbeat was trying to assert. A transition
landing AFTER the heartbeat's mint needs no guard at all: it takes a higher
seq and wins at Herdr on its own. Either way the newest word wins, with the
provider still outside the lock.
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
from typing import Callable, Literal, Mapping, Sequence, cast

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

#: Backoff between delivery attempts for ONE queued call, in seconds; its
#: length is therefore the retry budget (3 retries, 4 attempts total). Slept
#: on the worker thread, never the event loop. See the module docstring for
#: why the shape is 0.5 / 2 / 8 and why exhaustion logs at WARNING.
RETRY_BACKOFF_S: tuple[float, ...] = (0.5, 2.0, 8.0)

#: How often the heartbeat re-asserts the current state. Injectable per
#: reporter so tests do not wait on it. See "WHY A HEARTBEAT" above.
RESYNC_INTERVAL_S = 30.0

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

#: Reads the CURRENT state, for the heartbeat to re-assert. Called off the
#: event loop and NEVER with ``_lock`` held, so a slow or wedged
#: implementation delays only its own heartbeat tick — not ``report``, not
#: ``release``, not the exit drain. It should still be cheap and not block
#: (the band's implementation is two attribute reads), but the module no
#: longer depends on that for its bounds. See "WHY A HEARTBEAT" for the
#: mint counter that makes a stale read harmless without the lock.
StateProvider = Callable[[], HerdrState]


def _default_clock() -> int:
    return time.time_ns() // 1_000


def _argv_flag(argv: Sequence[str], name: str) -> str | None:
    """The value following ``name`` in an argv, for log lines only.

    Reading it back out of the argv rather than carrying it alongside keeps
    the queue item the ``(subcommand, argv)`` pair the whole module (and
    every fake invoker in the tests) is written against.
    """
    items = list(argv)
    try:
        return items[items.index(name) + 1]
    except (ValueError, IndexError):
        return None


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
        resync_interval_s: float = RESYNC_INTERVAL_S,
        retry_backoff_s: Sequence[float] | None = None,
    ) -> None:
        self._pane_id = pane_id
        self._binary = binary
        self._session_id = (session_id or "").strip() or None
        self._invoker: Invoker = invoker or _run_cli
        self._clock: Clock = clock or _default_clock
        self._resync_interval_s = resync_interval_s
        self._retry_backoff_s: tuple[float, ...] = tuple(
            RETRY_BACKOFF_S if retry_backoff_s is None else retry_backoff_s
        )
        # Guards `_last`, `_seq` and `_session_id`: the seq must be minted in
        # the same critical section that decides the report is not a
        # duplicate, or two callers could both pass the de-dupe and enqueue
        # the same state twice in either order.
        self._lock = threading.Lock()
        self._last: HerdrState | None = None
        self._seq = 0
        #: How many seqs have been minted, under `_lock`. Not a sequence
        #: number and never sent: purely the generation counter the heartbeat
        #: compares against to tell whether a transition overtook the state it
        #: just read (review round 2, M2).
        self._mints = 0
        #: Every call ever enqueued, in seq order. `None` is the worker's
        #: stop sentinel and is enqueued exactly once, by `release`.
        self._queue: queue.SimpleQueue[tuple[str, tuple[str, ...]] | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        #: A worker created under `_lock` by `_enqueue` and started outside it
        #: by `_start_worker`. Exists so the mint-and-enqueue critical section
        #: contains nothing that can block (review round 1, A2).
        self._pending_start: threading.Thread | None = None
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
        #: The state of the last `report-agent` the worker actually DELIVERED,
        #: as opposed to `_last`, which is what was last enqueued. Written on
        #: the worker thread only; a plain attribute because a single
        #: reference assignment needs no lock and no reader is making a
        #: decision on it. Cleared back to None on a delivered `release-agent`
        #: — the row is gone at that point, so "what Herdr is holding" is
        #: nothing, and a stale `idle` here would be the wrong answer for any
        #: diagnostic that reads it.
        self._delivered: HerdrState | None = None
        #: Set by `set_state_provider`; the heartbeat is inert without one, so
        #: a reporter nobody wired stays exactly as cheap as it was before.
        self._state_provider: StateProvider | None = None
        self._resync_thread: threading.Thread | None = None

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
    def delivered_state(self) -> HerdrState | None:
        """The state of the last ``report-agent`` that actually landed.

        None before the first delivery, and again after a delivered
        ``release-agent``. Differs from :attr:`last_state` exactly when a
        delivery is in flight, being retried, or was dropped on exhaustion.
        """
        return self._delivered

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
        self._enqueue_report(state, dedupe=True)

    def set_state_provider(self, provider: StateProvider | None) -> None:
        """Wire the heartbeat to the caller's derivation of the current state.

        Attaching one starts the resync thread; it is the only thing that
        does, so a reporter nobody wires never pays for a thread. Idempotent
        in the sense that matters — the thread is started at most once, and a
        later provider simply replaces the one it reads.

        The provider is called ON THE RESYNC THREAD, never on the event loop,
        so it must be cheap and must not block. It must also not raise; one
        that does is logged and the tick skipped, because a heartbeat that
        killed its own thread would silently take the resync with it.
        """
        # Detaching always takes effect; attaching to a RELEASED reporter does
        # not store anything. The heartbeat is over at that point, so the only
        # thing keeping the provider would be this attribute — and its closure
        # holds the StatusLine, which would then outlive the row it described
        # for the rest of the process.
        if provider is None:
            self._state_provider = None
            return
        if self._released.is_set():
            return
        self._state_provider = provider
        with self._lock:
            if self._resync_thread is not None or self._released.is_set():
                return
            thread = threading.Thread(
                target=self._resync_loop, name="lop-herdr-resync", daemon=True
            )
            self._resync_thread = thread
        thread.start()

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
            # Under the lock for the same reason as `report`, and with the
            # stop sentinel in the same critical section: the release must be
            # the LAST item in the queue, which is only guaranteed if no
            # `report` can slip between the two puts.
            self._enqueue(("release-agent", argv))
            self._queue.put(None)
        self._start_worker()

    def join(self, timeout: float = EXIT_DRAIN_TIMEOUT_S) -> None:
        """Wait for this reporter's threads to finish, bounded by ``timeout`` TOTAL.

        Never called from the event loop: a call parked in a subprocess
        timeout would stall the TUI for exactly as long as this waits.

        ``timeout`` is the budget for the WHOLE call, not per thread. Passing
        it to each :meth:`threading.Thread.join` in turn is the obvious
        version and it is wrong: with both the worker and the resync thread
        blocked, one reporter consumed 2x the figure — ``join(timeout=1.0)``
        measured at 2.02 s — which let a single reporter overrun the shared
        ``remaining`` that :func:`_drain_at_exit` computes against
        :data:`EXIT_DRAIN_TIMEOUT_S`, so the documented worst-case exit delay
        was the bound times the number of threads (review round 2, M1).
        """
        deadline = time.monotonic() + timeout
        for thread in (self._thread, self._resync_thread):
            if thread is None or thread is threading.current_thread():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            thread.join(timeout=remaining)

    # -- internals -----------------------------------------------------------

    def _next_seq_locked(self) -> int:
        # See the module docstring for why this is not `+= 1` alone.
        self._seq = max(self._seq + 1, self._clock())
        # Every mint bumps this, which is what lets the heartbeat notice that a
        # transition landed while it was reading the provider (M2). Counted
        # rather than compared against `_last`, because a transition that
        # returns to the state the heartbeat sampled still means the sample is
        # no longer the newest word on it.
        self._mints += 1
        return self._seq

    def _enqueue_report(
        self, state: HerdrState, *, dedupe: bool, only_if_mints: int | None = None
    ) -> None:
        """Mint a seq for ``state`` and queue its ``report-agent``. Lock NOT held.

        The ONE mint-and-enqueue critical section, shared by :meth:`report`
        and by the heartbeat so the lock invariant lives in a single place.
        ``dedupe`` is the only difference between the two callers: the
        spinner tick must be suppressed, the heartbeat must not be (see the
        module docstring).

        ``only_if_mints`` is the heartbeat's staleness guard: the value of
        :attr:`_mints` the caller observed BEFORE it read the state it is now
        enqueueing. If another mint has happened since, that state was read
        before a transition this reporter has already queued, so enqueueing it
        would re-assert a value the session has moved off — the call is
        dropped instead (review round 2, M2). ``None`` means "no guard", which
        is every caller that already knows its own state.
        """
        # Re-tested INSIDE the lock below, not only here. This early exit is a
        # cheap filter for the common post-release case; it is not the
        # decision, because a `report` that passes it and then blocks on
        # `_lock` — held by a concurrent `release` — would otherwise mint a
        # HIGHER seq than the release and resurrect the row for a process that
        # has already exited. Herdr's high-water mark cannot discard a higher
        # seq, so that row would say `working` forever (review round 1, A1).
        if self._released.is_set():
            return
        with self._lock:
            if self._released.is_set():
                return
            if only_if_mints is not None and self._mints != only_if_mints:
                return
            if dedupe and state == self._last:
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
            # Enqueued UNDER the lock, so the queue order is the mint order.
            # Building the argv outside it and putting afterwards let two
            # callers mint 1,2 and enqueue 2,1 — measured at 80/200 trials
            # under `sys.setswitchinterval(1e-6)` (review round 1, A2). The
            # lock is held for a list build and a `SimpleQueue.put`, both
            # non-blocking, so this costs the caller nothing it can feel.
            self._enqueue(("report-agent", argv))
        # Outside the lock: see `_start_worker`.
        self._start_worker()

    def _resync_loop(self) -> None:
        """Re-assert the current state every interval, until released.

        ``_released.wait`` rather than ``time.sleep`` so a quit does not have
        to outlast a 30 s nap: it returns True the instant the latch is set,
        which is both the sleep and the stop condition.
        """
        while not self._released.wait(self._resync_interval_s):
            provider = self._state_provider
            if provider is None:
                continue
            # Snapshot BEFORE the read, so any transition that mints while the
            # provider runs is visible to the guard in `_enqueue_report` (M2).
            mints = self._mints
            try:
                state = provider()
            except Exception:  # noqa: BLE001 — a bad provider must not kill the thread
                logger.debug("herdr state provider failed", exc_info=True)
                continue
            # Called OUTSIDE `_lock` on purpose: see the module docstring.
            self._enqueue_report(state, dedupe=False, only_if_mints=mints)

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
        """Queue one call. THE CALLER MUST HOLD ``_lock``.

        That requirement is the ordering guarantee, not an implementation
        detail: the queue order has to be the order the seqs were minted in,
        and the only way to promise that is to mint and put in one critical
        section (review round 1, A2).

        ``_lock`` is a plain :class:`threading.Lock` and is therefore NOT
        reentrant, which is why the worker start is split out below rather
        than called from here — an ``_ensure_worker`` that took the lock again
        would deadlock every caller.
        """
        self._queue.put(item)
        # Registered while the lock is held and BEFORE the thread exists, so
        # an interpreter exit racing the first enqueue still finds this
        # reporter in the registry and drains it.
        if self._thread is None:
            _LIVE_REPORTERS.add(self)
            _register_exit_drain()
            # Created here but started by `_start_worker` outside the lock:
            # a reporter that never reports (built and discarded) costs no
            # thread, and nothing user-facing waits on the start.
            self._thread = threading.Thread(target=self._run, name="lop-herdr-report", daemon=True)
            self._pending_start = self._thread

    def _start_worker(self) -> None:
        """Start the worker created by :meth:`_enqueue`, if any. Lock NOT held.

        Split from the enqueue so the ordering-critical section stays free of
        anything that could block, and so a plain (non-reentrant) lock is
        enough to hold the whole mint-and-put together.
        """
        with self._lock:
            pending = self._pending_start
            self._pending_start = None
        if pending is not None:
            pending.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            subcommand, argv = item
            self._deliver(subcommand, argv)

    def _deliver(self, subcommand: str, argv: tuple[str, ...]) -> None:
        """Invoke one queued call, retrying in place. Worker thread only.

        Retrying HERE, rather than re-queueing the item at the back, is what
        keeps delivery order equal to mint order: the item under retry holds
        the head of the queue and nothing behind it can overtake it.
        """
        released_aborts = subcommand != "release-agent"
        # `Exception`, not `BaseException`: the `except` below catches exactly
        # that, so this is the widest thing that can ever land here.
        last_error: Exception | None = None
        for attempt in range(len(self._retry_backoff_s) + 1):
            try:
                self._invoker(subcommand, argv)
            except Exception as error:  # noqa: BLE001 — best-effort by contract
                logger.debug("herdr %s attempt %d failed", subcommand, attempt + 1, exc_info=True)
                last_error = error
            else:
                # Delivered. `release-agent` clears the tracker rather than
                # leaving a state behind for a row that no longer exists.
                if subcommand == "report-agent":
                    state = _argv_flag(argv, "--state")
                    self._delivered = cast(HerdrState, state) if state is not None else None
                else:
                    self._delivered = None
                return
            if attempt == len(self._retry_backoff_s):
                break
            delay = self._retry_backoff_s[attempt]
            # A report gives up its backoff to the release queued behind it;
            # a release has nothing behind it and must not give up. See the
            # module docstring.
            if released_aborts:
                if self._released.wait(delay):
                    return
            elif delay > 0:
                # Clamped because `time.sleep` RAISES on a negative, which on
                # this thread would kill the worker and take every queued
                # call with it; `retry_backoff_s` is injectable, so a test
                # (or a future caller) can hand in a negative.
                time.sleep(max(0.0, delay))
        # WARNING, not DEBUG: the de-dupe means this state will not be sent
        # again by a transition, so the row is now stale until the heartbeat
        # or the next real change catches it. That is user-visible.
        logger.warning(
            "herdr %s dropped for pane %s after %d attempts (state=%s seq=%s): %s",
            subcommand,
            self._pane_id,
            len(self._retry_backoff_s) + 1,
            _argv_flag(argv, "--state"),
            _argv_flag(argv, "--seq"),
            last_error,
        )


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
        # `resync_interval_s` and `retry_backoff_s` are deliberately NOT
        # forwarded: they are test-only knobs for making the heartbeat and the
        # backoff affordable in a suite, and production takes the module
        # defaults (RESYNC_INTERVAL_S, RETRY_BACKOFF_S) so the timings a user
        # experiences are the ones the module docstring argues for. Adding them
        # here would make those figures a per-call-site choice.
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

#: Every reporter that has queued a call in this process, so the exit drain
#: can release and join each one.
#:
#: A ``WeakSet`` for hygiene rather than for reclamation, and the difference
#: is worth stating because the comment here used to claim the latter: a
#: reporter whose worker is RUNNING is never collected while it is in the set,
#: because the worker thread's own ``self._run`` bound method holds a strong
#: reference to it (measured: 50 built-and-dropped reporters → 50 still
#: retained and 50 threads parked on ``queue.get`` after two ``gc.collect``
#: passes — review round 1, A3). What the weak set does buy is that once a
#: reporter IS released and its worker has returned, nothing here keeps it
#: alive. Production has one reporter per process — a pane is a process, and
#: a session swap re-labels the existing reporter rather than building
#: another — so the retention is bounded at one; only a test that builds many
#: accumulates threads, which is why they release explicitly.
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

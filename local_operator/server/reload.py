"""Move a running ``serve`` daemon onto the build ``current`` names, in place.

WHY THIS IS NOT AN EXIT, AND WHY IT IS NOT A RESTART EITHER
-----------------------------------------------------------
:mod:`local_operator.server.retire` establishes, with measurements behind it,
that a changed build stamp is not a licence for a daemon to leave: this process
owns ``SchedulerService`` work, and the desktop app holds an unbounded relay
(``GET /v1/desktop/sessions/{id}/events``) for as long as a conversation is on
screen, so the drain an exit waits for is never empty while anybody is using the
machine. A daemon started outside the app has no supervisor to bring it back
either, which is why that module's production path announces the new build and
then keeps serving the old one indefinitely.

The generation layout (``update.py``'s ``generations/<id>``) removes the reason
to leave at all. A build lives in a tree that is written once and never
rewritten, so a process can hand its **listening socket** to the new build's
interpreter and come back as the same pid, with the same cwd and the same
environment. Nothing has to prove that a *successor* is ready, because there is
no successor: this process is the successor, and the listening socket it keeps
is the same kernel object, so the port is never unbound and a client connecting
across the change is served rather than refused.

That is what makes this the honest answer to "the app updated the install and
the server is still serving the old build": the daemon moves ITSELF, so a server
the app did not start is brought along without the app having to own, kill or
respawn a process that is not its child. Measured on the reporting host while it
was serving eight live runtimes, the alternative — a kill and a relaunch — is
what a reload exists to avoid.

WHAT A RELOAD COSTS, AND WHAT IT DOES NOT
-----------------------------------------
It does NOT touch runtimes. A runtime is spawned with ``start_new_session=True``
(``session/runtime/launch.py``), holds its own transcript lease, and is not this
process's child in any sense an ``execve`` here can reach; a session engaged
after the reload is constructed on the build the pointer names, which is the
fleet-convergence rule ``launch._spawn_interpreter`` already implements.

It DOES cut the in-process, in-memory attachments, which is the whole reason the
drain below exists: a desktop relay stream, a watch lease, the SSE frames of a
turn streaming right now, and — measured on the end-to-end rig rather than
reasoned about — every ESTABLISHED connection, whose descriptor belongs to this
image and is not carried across the exec. Only the LISTENER is inherited. So a
keep-alive connection or an open relay is dropped and must be re-opened; the
port itself never has a gap, which is the property that makes this better than a
stop-and-start (measure it with the rig: a client dialling throughout the change
is served, while the one that was already connected is cut).

Those cuts are *recoverable by reconnect* — the app re-opens its relay and
re-reads history, and the turn itself is running inside the runtime, which never
stopped. Two things are NOT recoverable, and the original version of this
paragraph named only one of them (review round 1, R1-6):

* **A runtime being SPAWNED**, whose ~1.2 s handshake would be cut in the middle.
  This is what the drain below waits for.
* **Daemon-owned `SchedulerService` work.** ``SchedulerService._run_tasks`` runs
  inside THIS process, and a reload does not run the lifespan's shutdown, so a
  scheduled run in flight is cut exactly as a SIGKILL would cut it. No probe
  reports it (``server/retire``'s own module docstring says the same about its
  drain predicate) and the drain deliberately does not gate on it: waiting for
  "no scheduled work anywhere" would be a reload that never happens, and this is
  a strictly better position than the status quo it replaces, which had no route
  off a stale build at all. It is recorded as a limit rather than smoothed over.

FAIL-CLOSED, ALWAYS
-------------------
Every failure below keeps the daemon serving the build it loaded. A reload whose
pointer cannot be read, whose interpreter cannot be resolved, whose drain does
not empty inside the budget, or which is asked to move onto the build it is
already running, logs why and leaves the daemon alone: the operator still has a
working backend, and ``lop services status`` still reports the drift.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.interpreter import SAFE_PATH_FLAG

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

logger = logging.getLogger("local_operator.server.reload")

#: ``app.state`` attribute carrying the listening socket's fd for a reload.
#:
#: Set by ``cli.serve_command`` on the SAME app object it hands uvicorn, which
#: is the only process a reload can happen in: the socket lives in
#: ``serve_command``'s frame, and the lifespan runs inside the loop that frame is
#: about to enter, so ``app.state`` is the one channel that reaches both without
#: a second global (the reason ``registry.announce_address`` uses it too).
LISTENER_FD_ATTR = "serve_listener_fd"

#: ``app.state`` flag: a reload has been REQUESTED and not yet performed.
#:
#: Deliberately not the record's ``retiring_*`` pair, which is an announcement
#: about the install on disk that any reader may observe. This flag is this
#: process's private intent, set by the signal handler and cleared only when the
#: reload it asked for is either done (which ends the process) or refused.
PENDING_ATTR = "serve_reload_pending"

#: The signal that asks a daemon to move itself onto ``current``.
#:
#: ``SIGUSR1`` because it is the POSIX name for exactly this, and because its
#: DEFAULT action is to terminate — so a caller must never send it blind. The
#: guard against that is the record's ``reloadable`` field: the capability is
#: published BY the daemon that has it, so a caller that waits for ``reloadable``
#: can only ever signal a process that installed this handler.
RELOAD_SIGNAL = signal.SIGUSR1

#: How long a requested reload may wait for the narrow drain before giving up.
#:
#: Sized against the two things it actually waits for: the desktop pool's spawn
#: handshake, measured at ~1.2 s in ``desktop_sessions.in_flight_reason``'s own
#: note, and an ordinary request handler, which is tens of milliseconds. Ten
#: seconds is several times that worst case rather than a guess, and an expiry
#: means something is genuinely stuck — a reason to keep serving, not a reason to
#: cut a client's work off.
DRAIN_BUDGET_S = 10.0

#: The poll period inside the drain. Not zero: a reload that spun the event loop
#: while it waited would stop the very handlers it is waiting for.
DRAIN_POLL_S = 0.1

#: How long the target build's CLI is given to prove it can start.
#:
#: Generous on purpose — a cold import of a CLI with a large dependency closure on
#: a loaded machine is seconds rather than milliseconds — but it has to FIT INSIDE
#: the caller's patience, because this runs before the exec and a caller that gave
#: up first would report a failure for a reload that then succeeded (review round
#: 2, MINOR-2: the first version was 30 s, which with the 10 s drain overran
#: ``services.RELOAD_WAIT_S``). Thirty was never a measurement; ten is still an
#: order of magnitude over the measured cold import, and an expiry is treated as
#: "the check could not run" — see :func:`_smoke`.
SMOKE_TIMEOUT_S = 10.0


class ReloadRefusal(Exception):
    """A reload that must not happen, with the sentence a caller can print."""


@dataclass(frozen=True)
class ReloadPlan:
    """What the replacement process will be.

    A value rather than a set of arguments, because it is the thing an operator
    and a test both want to see: which tree is being left, which is being entered,
    and what the daemon will be launched as.
    """

    #: The interpreter of the build ``current`` names — the replacement image.
    interpreter: Path
    #: The install root that interpreter belongs to, for the log line.
    target_root: Path
    #: ``sys.prefix`` of this process: the tree being left.
    loaded_root: Path
    #: The listening socket to hand across ``execve``.
    listener_fd: int
    #: The address to report. Cosmetic on the reload path (the socket is
    #: inherited, never re-bound) but it is what the record and the banner name.
    host: str
    port: int


def bind_listener_fd(app: "FastAPI", fd: int) -> None:
    """Publish the bound listener's fd to the lifespan. Called by ``serve_command``.

    A function rather than an attribute assignment at the call site so the
    attribute name lives in one place: ``cli.serve_command`` writes it and the
    lifespan reads it, and a rename that updated only one would end in a daemon
    whose reload silently never fires.
    """
    setattr(app.state, LISTENER_FD_ATTR, int(fd))


def listener_fd(app: Any) -> int | None:
    """The fd a reload should hand to the new build, or ``None``.

    ``None`` is a real answer rather than an error: a ``--reload`` child's port
    belongs to uvicorn's supervisor, and a boot that was never announced (a bare
    uvicorn, a nested import) has no socket of its own to preserve. Both cases
    refuse the reload instead of guessing.
    """
    value = getattr(getattr(app, "state", None), LISTENER_FD_ATTR, None)
    return int(value) if isinstance(value, int) else None


def is_reloadable(app: Any) -> bool:
    """The capability the record publishes, decided in ONE place.

    The record field and the armed watch must agree: a record claiming
    ``reloadable`` for a daemon that never installed the handler is a caller
    signalling a process whose default disposition for ``SIGUSR1`` is death.
    """
    return listener_fd(app) is not None and not _is_reload_child(app)


def _is_reload_child(app: Any) -> bool:
    from local_operator.server import registry as serve_registry

    return serve_registry.is_reload_child(app)


class ReloadWatch:
    """The pending flag, the signal that sets it, and the task that acts on it.

    ONE object rather than two globals, because the flag and the task have to
    agree about which daemon they belong to: the suite drives several apps in one
    process, and a signal that reached another app's loop would reload a daemon
    nobody asked to move.
    """

    def __init__(self, app: "FastAPI", *, stop: asyncio.Event | None = None) -> None:
        self.app = app
        self.stop = stop
        self.requested = asyncio.Event()
        setattr(app.state, PENDING_ATTR, False)

    def request(self) -> None:
        """Note that a reload was asked for. SAFE IN A SIGNAL HANDLER.

        It only sets an Event, which is what ``loop.add_signal_handler`` is for:
        the callback runs on the loop between iterations, so nothing here takes a
        lock, touches a socket, or allocates a task.

        INFO, and the daemon's console logging runs at WARNING by default, so this
        line may not appear where an operator is looking. That is deliberate and
        it is the same trade ``retire`` records for its own notice: the request is
        made by a command that reports the outcome itself (``lop services
        restart``), and a daemon that logged every request at WARNING would make a
        routine update look like a fault.
        """
        logger.info("serve reload requested; the daemon will move onto the current build")
        setattr(self.app.state, PENDING_ATTR, True)
        self.requested.set()

    @property
    def pending(self) -> bool:
        return bool(getattr(getattr(self.app, "state", None), PENDING_ATTR, False))

    async def wait(self) -> None:
        """Block until a reload is requested or the daemon is stopping."""
        if self.stop is None:
            await self.requested.wait()
            return
        requested = asyncio.ensure_future(self.requested.wait())
        stopped = asyncio.ensure_future(self.stop.wait())
        try:
            await asyncio.wait({requested, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (requested, stopped):
                task.cancel()

    async def run(self) -> None:
        """The lifespan's task: wait for a request, drain, then replace this image.

        Returns only when the daemon is stopping or the reload was REFUSED — a
        successful reload never returns, because ``execve`` has replaced this
        process. That asymmetry is why the refusal paths are loud: they are the
        only ones anything downstream will ever see.
        """
        while True:
            await self.wait()
            if self.stop is not None and self.stop.is_set():
                return
            self.requested.clear()
            try:
                await self.perform()
            except ReloadRefusal as refusal:
                logger.warning("serve reload refused: %s", refusal)
            except Exception:  # noqa: BLE001 — a reload must never take the daemon down
                logger.warning(
                    "serve reload failed before the exec; still serving the loaded build",
                    exc_info=True,
                )
            # Reached only when the reload did NOT replace this process, so the
            # flag is cleared either way and the loop waits for the next request.
            setattr(self.app.state, PENDING_ATTR, False)

    async def perform(self) -> None:
        """Drain, announce, then ``execve`` onto the current build.

        Never returns on success: the ``execve`` at the end of :func:`_exec`
        either replaces this image or raises.
        """
        plan = self.plan()
        await self.drain()
        # OFF THE EVENT LOOP (review round 2, MINOR-3). The check spawns a process
        # and waits for it, measured at 0.20-0.35 s cold; run inline it would stop
        # this daemon answering HTTP for that whole time, which is the opposite of
        # what a reload is for. `to_thread` rather than a second loop: the body is
        # a single bounded subprocess call, so there is nothing to serialise.
        await asyncio.to_thread(_smoke, plan.interpreter)
        # THE DRAIN DOES NOT COVER THAT AWAIT, SO IT IS RE-ASKED (review round 3,
        # R3-1). Moving the smoke off the loop made the pre-exec phase longer than
        # the wait that guards it, and the term the wait exists for — a runtime
        # being SPAWNED, whose ~1.2 s handshake is the one cut this module calls
        # unrecoverable — can begin inside that window. A first drain that returned
        # at t=0 says nothing about t=0.3, so the predicate is asked again rather
        # than assumed to still hold: the budget restarts, and a spawn that arrived
        # in between is waited for on the same terms as one that was already there.
        # It refuses rather than proceeds if the second budget expires, which is
        # the fail-closed direction: staying on the loaded build costs a skew, and
        # cutting a handshake costs a runtime.
        await self.drain()
        logger.info(
            "serve reload: pid %d is leaving %s for %s (interpreter %s, listener fd %d)",
            os.getpid(),
            plan.loaded_root,
            plan.target_root,
            plan.interpreter,
            plan.listener_fd,
        )
        _exec(plan)

    def plan(self) -> ReloadPlan:
        """Resolve what the replacement will be, or refuse with the reason.

        NO WORK IS DONE HERE, deliberately: this is the half a test can drive on
        a machine with no install (it is a pure read of the pointer, this
        process's own prefix, and two pieces of app state), and keeping it
        separate from the drain means a refusal is decided before anything is
        interrupted.
        """
        from local_operator import update
        from local_operator.server import registry as serve_registry

        fd = listener_fd(self.app)
        if fd is None:
            raise ReloadRefusal(
                "this daemon has no listening socket of its own to carry across an "
                "exec (a --reload child's port belongs to its supervisor)"
            )
        announced = serve_registry.advertised_address(self.app)
        if announced is None:
            raise ReloadRefusal("this daemon was never told which address it is serving on")
        interpreter = update.current_interpreter()
        if interpreter is None:
            raise ReloadRefusal(
                "the install pointer resolves to nothing, so there is no build to move onto; "
                "run `lop update` or `lop-update` first"
            )
        target_root = update.current_install_root()
        loaded_root = Path(update.process_install_root())
        if target_root is not None and _same_tree(loaded_root, target_root):
            raise ReloadRefusal(
                f"this daemon is already running the build the pointer names ({loaded_root})"
            )
        host, port = announced
        return ReloadPlan(
            interpreter=Path(interpreter),
            target_root=target_root or Path(interpreter).parent.parent,
            loaded_root=loaded_root,
            listener_fd=fd,
            host=host,
            port=int(port),
        )

    async def drain(self) -> None:
        """Wait for the narrow set of things a reload must not cut. Bounded.

        The predicate is the desktop pool's ``reload_blocker``, which is
        deliberately NARROWER than ``in_flight_reason``: that one counts the
        standing relay and the watch lease, both of which are recoverable by
        reconnect and both of which are held for as long as the app is open — so
        gating a reload on it would mean a reload that never happens on exactly
        the machine it exists for.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + DRAIN_BUDGET_S
        while True:
            reason = self.blocker()
            if reason is None:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ReloadRefusal(
                    f"{reason} and it did not finish within {DRAIN_BUDGET_S:.0f}s; "
                    "still serving the loaded build, and the next request will retry"
                )
            await asyncio.sleep(min(DRAIN_POLL_S, remaining))

    def blocker(self) -> str | None:
        """Why the reload should wait, or ``None``. An unreadable probe is ``None``.

        "Absent" and "unreadable" both mean this process has no reason to wait:
        a pool that was never built has no spawn in flight, and a probe that
        raises is not evidence about work — the same rule ``retire.in_flight``
        states for its own probes, in the direction that keeps the daemon alive.
        """
        pool = getattr(getattr(self.app, "state", None), "desktop_sessions", None)
        check = getattr(pool, "reload_blocker", None)
        if not callable(check):
            return None
        try:
            return check()
        except Exception:  # noqa: BLE001 — an unreadable probe may not decide this
            logger.warning("serve reload: the drain probe could not be read", exc_info=True)
            return None


def _same_tree(left: Path, right: Path) -> bool:
    """Are these two paths the same directory, compared through symlinks?

    ``current`` is a symlink, and a daemon's ``sys.prefix`` is the concrete
    generation path, so a string comparison would call an identical tree
    different and reload a daemon onto the build it is already running — the
    exact case this refusal exists to prevent.
    """
    try:
        return left.resolve() == right.resolve()
    except OSError:  # pragma: no cover - an unresolvable path is not a match
        return False


def _exec(plan: ReloadPlan) -> None:
    """Replace this process image. Separated so a test can observe the plan.

    ``os.execve`` and not ``os.execvpe``: the environment is passed explicitly
    because it is the whole of what this process knows that its successor needs
    — the desktop claim, the config root, the log directory, the auth posture —
    and ``execvpe`` would search ``PATH`` for an image this code has already
    resolved to an exact path.

    ``argv[0]`` IS THE INTERPRETER'S OWN PATH, and that is a deliberate trade
    against this repository's process-naming ladder. A branded ``argv[0]`` is
    only safe alongside a branded image (``procname.spawn_identity``), and the
    branded hardlink is planted per venv — the NEW generation's tree may not have
    one yet, which ``update._post_upgrade_invocation`` records for the same
    reason. CPython on Linux derives ``sys.executable`` from ``argv[0]``, so a
    labelled ``argv[0]`` here would hand the replacement an EMPTY
    ``sys.executable`` and break every ``subprocess.run([sys.executable, …])``
    in it. Correctness of the daemon wins over the name it appears under, and
    the role is still readable from the command line beside it and from the
    record the replacement republishes.

    ``-P`` (``SAFE_PATH_FLAG``) rides along for the reason it does on every other
    product spawn: the successor must import from its own generation rather than
    from whatever directory this one happened to be started in.

    TWO THINGS HAPPEN HERE THAT ARE NOT ABOUT THE EXEC ITSELF, and both exist for
    the window between this call and the successor's first instruction — a window
    the record CANNOT cover, because the record it will publish does not exist
    yet (review round 1, R1-1 and NIT-2).
    """
    fd = plan.listener_fd
    # The fd must survive ``execve``. Python 3's PEP 446 makes every fd
    # non-inheritable by default, so without this the replacement starts with no
    # listener and the port is simply gone — the failure mode this whole module
    # exists to avoid, and it is silent from the outside.
    os.set_inheritable(fd, True)
    # AND THE SIGNAL MUST BE MADE HARMLESS, which is not belt-and-braces. For the
    # successor's whole boot the LAST published record is still live and
    # heartbeat-fresh, still advertising ``reloadable: true``, while the process
    # it names has not yet reached ``add_signal_handler`` — so SIGUSR1 is still
    # at its DEFAULT disposition of terminate. A second ``lop services restart``,
    # or the app's own updater firing twice (measured 24 s apart on the reporting
    # host), signals inside that window and kills the daemon it was moving.
    # Reproduced end to end in review: 1.45 s of stale-but-live record, SIGUSR1
    # at t=0.8 s, pid gone, port dead, nothing logged.
    #
    # ``SIG_IGN`` SURVIVES ``execve`` (POSIX inherits ignored dispositions across
    # it, unlike handlers), so the successor starts deaf and
    # ``add_signal_handler`` then installs the real handler. That closes the
    # window by construction rather than by timing.
    previous = signal.getsignal(RELOAD_SIGNAL)
    signal.signal(RELOAD_SIGNAL, signal.SIG_IGN)
    argv = [
        str(plan.interpreter),
        SAFE_PATH_FLAG,
        "-m",
        "local_operator.cli",
        "serve",
        "--host",
        plan.host,
        "--port",
        str(plan.port),
        "--listener-fd",
        str(fd),
    ]
    try:
        os.execve(str(plan.interpreter), argv, dict(os.environ))
    finally:
        # Reached only when the exec FAILED — a successful one never returns.
        # Both restorations run even if the first raises (review round 2, NIT-3):
        # a daemon left deaf is worse off than one left with a writable
        # descriptor, because the first can never be asked to reload again.
        # The descriptor goes back exactly as the reload found it, because this
        # process is still serving: an fd left inheritable would be handed to
        # every child it later spawns, including the runtime spawns that are its
        # whole purpose.
        try:
            signal.signal(RELOAD_SIGNAL, previous)
        finally:
            os.set_inheritable(fd, False)


def _smoke(interpreter: Path) -> None:
    """Prove the target build can even start its CLI, or refuse. Bounded.

    THE ONE THING THAT MAKES "FAIL-CLOSED, ALWAYS" TRUE ON THE FAR SIDE OF THE
    EXEC (review round 1, R1-5). Every other refusal in this module happens
    BEFORE anything is interrupted, which is what makes a refusal cheap: the
    daemon keeps its pid, its socket and its record. An exec into a build that
    cannot import its own CLI is the one failure with no such floor — a dead
    daemon, a dead port, and no process left to log it. This is the cheapest
    check that distinguishes "a build is there" from "a build works".

    A CHECK THAT COULD NOT RUN IS NOT A BAD BUILD, which is the same rule
    ``server/retire`` applies to its own probes and the app's seal probe applies
    to ``codesign``: a spawn failure, a timeout or a vanished interpreter is
    logged and the reload PROCEEDS, because a machine where the probe cannot run
    is a machine where refusing would strand the daemon on the old build for that
    reason alone — and the pointer was already trusted enough to resolve. A check
    that RAN and exited non-zero is a refusal.
    """
    import subprocess

    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(interpreter), SAFE_PATH_FLAG, "-c", "import local_operator.cli"],
            capture_output=True,
            text=True,
            timeout=SMOKE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # noqa: PERF203 — one probe, once
        logger.warning(
            "serve reload: the build at %s could not be smoke-tested; proceeding on the "
            "pointer's word alone",
            interpreter,
            exc_info=True,
        )
        return
    if completed.returncode == 0:
        return
    output = (completed.stderr or completed.stdout or "").strip().splitlines()
    detail = output[-1][:200] if output else "no output"
    raise ReloadRefusal(
        f"the build at {interpreter} cannot import its own CLI "
        f"(exit {completed.returncode}: {detail}); still serving the loaded build"
    )


def observe_reload(task: asyncio.Task[None]) -> None:
    """Report a reload task that DIED, rather than letting it die silently.

    The same reasoning as ``retire.observe_poll``, and it matters more here: a
    reload task that died leaves a daemon that still ADVERTISES the capability in
    its record, so a caller would go on signalling a process whose handler now
    sets an Event nothing is waiting on — the request accepted, the daemon
    unmoved, and no line anywhere explaining either.

    A cancelled task is the ordinary teardown and is not reported: the lifespan
    cancels this task as part of ``stop``, which is a successful ending.
    """
    if task.cancelled():
        return
    failure = task.exception()
    if failure is not None:
        logger.error(
            "serve reload task died; this daemon will keep serving its loaded build "
            "and its record still advertises the reload capability",
            exc_info=failure,
        )


def install(app: "FastAPI", stop: asyncio.Event | None = None) -> ReloadWatch | None:
    """Arm the reload for this daemon, or return ``None`` when it cannot be armed.

    Called from the lifespan, inside the running loop, because that is the only
    context in which ``add_signal_handler`` works — and it is the right one
    anyway: the handler must set an asyncio Event on THIS loop rather than a flag
    some other thread reads.

    ``None`` is refusal, not failure, and the daemon keeps serving: a ``--reload``
    child (whose port is its supervisor's), a boot that was never announced, and
    a platform without ``asyncio`` signal handlers (Windows raises
    ``NotImplementedError``) all land here. Each is the pre-existing behaviour,
    which is what makes ``None`` safe to return without a fallback.
    """
    if not is_reloadable(app):
        logger.debug("serve reload not armed: this boot has no listener of its own")
        return None
    watch = ReloadWatch(app, stop=stop)
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(RELOAD_SIGNAL, watch.request)
    except (NotImplementedError, RuntimeError, ValueError):
        logger.debug("serve reload not armed: this platform has no asyncio signal handlers")
        return None
    return watch

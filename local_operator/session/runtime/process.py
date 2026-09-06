"""A detached session process: ``python -m local_operator.session.runtime.process``.

The daemon spawns one of these per phone-started session instead of hosting
the session in-process, for one reason: **lifetime**. The daemon is
supervised state — launchd restarts it on crash and on ``lop mobile
restart`` — and a session living inside it would die with every restart,
taking an in-flight turn with it. A child with its own pid has terminal
session lifetime: the daemon going away costs the phone its view, never the
session its work.

The child builds a session with the CLI's composition root, wraps it in the
owned-session handle (approval/ask gates resolved from the phone), registers
it through the normal record + control socket path, and idles until a signal
arrives or the residency predicate (:func:`_should_exit`) holds for one
sustained drain. Environment variables are the
spawn contract (``LOP_MOBILE_CHILD_CWD``, ``_PROVIDER``, ``_MODEL``) — argv
would be ps-readable.

**Residency (design §6.1).** The runtime is a unit of WORK, not of state; it
runs its trajectory to completion and exits when idle, so a closed terminal
costs nothing and a wake fires in a fresh process later. It stays resident
while any of three things holds — see :func:`_should_exit` for each term and
the reasoning behind it.

**Self-refresh (design-runtime-autorefresh §3.2).** Independently of the
quiet exit, an idle runtime whose install on disk has moved under it
(``lop-update`` ran) ANNOUNCES ``retiring`` to its viewers and exits, so the
next engage runs the new build. An attached viewer does not hold this — it
re-engages a successor itself — which is what keeps a five-hour-stale
runtime from staying resident because someone was looking at it. See
:func:`_should_refresh` and :func:`_refresh_for`.

This was ``mobile/child.py``. Only the phone spawns one today, but nothing in
it is phone-specific: it is the generic "a session running with no interface
owner" process, which is what later work needs for wakes and background
automations. The ``LOP_MOBILE_CHILD_*`` environment names keep their spelling
for the same reason ``RUN_DIRNAME`` does — they are a cross-process contract,
and during an upgrade a daemon of one version spawns a child of another.
``local_operator.mobile.child`` still resolves and still runs this ``main``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import random
import signal
import sys
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

if TYPE_CHECKING:
    from local_operator.update import BuildStamp

logger = logging.getLogger(__name__)

#: Fine polling makes the 3-second drain predictable while remaining cheap for
#: one event loop. Viewer TRAFFIC is not an input to this loop (a chatty
#: viewer does not reset the drain); viewer PRESENCE is, through term 3 of
#: the predicate.
REAP_CHECK_S = 0.25

#: Idle runtimes are disposable once their durable session state is
#: quiescent. This is a drain for newly arriving work, not a reconnect grace.
DEFAULT_GRACE_S = 3.0

#: A runtime whose own scheduler will fire a wake within this window stays
#: resident instead of exiting and paying a ~1.2 s cold start (plus the
#: supervisor's tick latency) to come back for it. Chosen to exceed
#: ``MIN_WAKE_INTERVAL_MS`` (60 s, harness.wake) by a margin: a session with
#: the tightest allowed recurrence then never thrashes exit → spawn → exit
#: once a minute, because the next fire is always inside the window. Anything
#: due further out is cheaper to leave to a cold spawn than to hold ~283 MB
#: for. Not env-tunable on purpose — it pairs with a constant in the wake
#: layer, and a knob would let the two drift apart.
WARM_WINDOW_S = 90.0

#: How often an idle runtime re-reads the install on disk. Cheap (one
#: dist-info lookup + one 60-byte file), but there is no reason to do it on
#: every 250 ms reaper tick: a runtime that is already idle can afford to
#: notice an update within a few seconds, and a busy one never checks.
BUILD_CHECK_S = 5.0

#: A freshly written install is not a stable one: ``lop-update`` runs
#: ``uv tool install --force`` (which rewrites site-packages over several
#: seconds) and THEN writes ``.lop-source``. Retiring against a half-written
#: tree would spawn a successor that imports a mix of two builds. Require
#: the marker's mtime to be at least this old before acting on it.
#: Env-overridable ONLY so the e2e stage can flip a fake marker and observe
#: the retirement within its budget; production never sets the variable.
BUILD_SETTLE_S = 10.0

#: After the refresh predicate first holds, sleep a uniform random slice of
#: this before re-checking and retiring. This host runs ~16 resident
#: runtimes; ``lop-update`` would otherwise have all of them notice on the
#: same tick and their viewers spawn sixteen successors within a second.
#: Spread over 20 s the eager re-engages average ≤1 spawn/s. Same env
#: override rule as the settle: test-only.
BUILD_STAGGER_S = 20.0


def _grace_seconds() -> float:
    raw = os.environ.get("LOP_SESSION_GRACE_S", "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_GRACE_S
    return value if value > 0 else DEFAULT_GRACE_S


def _positive_seconds(raw: str, default: float) -> float:
    """``raw`` as a positive float, else ``default``.

    Same shape as ``_grace_seconds``: the refresh timings are constants in
    production and only the e2e stage shortens them (``LOP_BUILD_SETTLE_S``,
    ``LOP_BUILD_STAGGER_S``), so a malformed or non-positive value falls back
    to the constant rather than disabling the protection it names. ``0`` is
    deliberately NOT accepted for the settle: a zero settle is the torn-tree
    race this constant exists to prevent, and a test that wants "fast" can
    say ``0.1``. The two readers below spell their variable names out as
    literals so the ambient-environment audit (``test_ambient_env_isolation``)
    can see them.
    """
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _build_settle_seconds() -> float:
    return _positive_seconds(os.environ.get("LOP_BUILD_SETTLE_S", ""), BUILD_SETTLE_S)


def _build_stagger_seconds() -> float:
    return _positive_seconds(os.environ.get("LOP_BUILD_STAGGER_S", ""), BUILD_STAGGER_S)


def _build_prefix() -> str | None:
    """Where to read the install stamp from: ``sys.prefix`` in production.

    ``LOP_BUILD_PREFIX`` exists ONLY so the e2e stage can point a real
    ``process.py`` at a temp directory carrying a fake ``.lop-source`` and
    flip it under the runtime. Nothing outside ``tests/e2e`` sets it, and a
    production runtime that inherited it by accident would merely compare
    against a marker that never changes — it can never retire early.
    """
    return os.environ.get("LOP_BUILD_PREFIX") or None


def _build_changed(boot: "BuildStamp | None") -> "BuildStamp | None":
    """The build now on disk, if it differs from ``boot`` AND has settled.

    ``None`` means "nothing to do": same stamp, an unreadable stamp, a boot
    stamp that was never captured (a reduced test server), or a marker still
    inside the settle window (see ``BUILD_SETTLE_S``). Editable checkouts have
    no ``.lop-source`` and a constant version, so they never trip this — by
    design, matching ``design-build-skew.md`` §6.5: a developer's worktree
    runtime must not retire because they touched a file.
    """
    if boot is None:
        return None
    from local_operator import update as update_mod

    prefix = _build_prefix()
    try:
        on_disk = update_mod.installed_build(prefix)
    except Exception:  # noqa: BLE001 — an unreadable stamp is "no change"
        logger.debug("build stamp unreadable; no refresh", exc_info=True)
        return None
    if on_disk == boot:
        return None
    age = update_mod.build_marker_age_s(prefix)
    if age is None or age < _build_settle_seconds():
        # Younger than the settle, or unknowable: the install may still be
        # mid-write. Try again next check; the marker only gets older.
        return None
    return on_disk


def _should_refresh(handle: object, boot: "BuildStamp | None") -> "BuildStamp | None":
    """Retire so the next engage spawns from the build now on disk?

    Returns the NEW stamp when yes, ``None`` when no. Not a fourth term of
    :func:`_should_exit`: that predicate answers "may I exit *quietly*", and a
    refresh must ANNOUNCE (the ``retiring`` frame) so a viewer re-engages
    rather than reading the exit as owner death. Only when the runtime is
    doing NOTHING it would lose — ``OwnedSessionHandle.may_refresh`` is the
    one predicate for that, shared with the viewer-driven ``refresh_if_idle``
    op so both sides agree what idle means. An attached viewer does NOT hold
    (the operator's rule; the viewer re-engages on its own), and neither does
    pristineness — a pristine stale runtime is the cheapest refresh there is.
    A handle without the probe (an older host, a reduced test handle) never
    refreshes: unknown state is not an invitation to exit.
    """
    may_refresh = getattr(handle, "may_refresh", None)
    if not callable(may_refresh):
        return None
    try:
        if may_refresh():
            return None
    except Exception:  # noqa: BLE001 — uncertainty keeps the runtime
        logger.debug("refresh predicate failed; keeping runtime", exc_info=True)
        return None
    return _build_changed(boot)


def _wake_within_window(handle: object, *, now_ms: int | None = None) -> bool:
    """Term 2 of the predicate: does the runtime's OWN scheduler have a wake
    due within ``WARM_WINDOW_S``? Read through the handle (an optional
    capability, probed) so reduced test handles and older handle
    implementations that never grew the accessor behave as "no wakes" rather
    than crash the reaper."""
    accessor = getattr(handle, "next_wake_due_at", None)
    if not callable(accessor):
        return False
    try:
        due_at = accessor()
    except Exception:  # noqa: BLE001 — a broken accessor must not pin the runtime
        logger.debug("next_wake_due_at failed; treating as no wake", exc_info=True)
        return False
    if not isinstance(due_at, int) or isinstance(due_at, bool):
        return False  # None, or a shape this reaper does not understand
    now = int(time.time() * 1000) if now_ms is None else now_ms
    return due_at - now <= WARM_WINDOW_S * 1000


def _viewer_attached(runtime: object) -> bool:
    """Term 3 of the predicate: is an INTERACTIVE viewer connected?

    Only ``ClientKind == "attach"`` counts — a TUI following this session, or
    the phone's interactive attach while the user has the session open.
    ``"daemon"`` clients (the mobile daemon's adoption dial, ``lop send``,
    ``lop stop``, the future supervisor) deliberately do not: the daemon
    adopts EVERY session on the machine, so if its connection held runtimes
    warm nothing would ever exit. ``RuntimeServer.attach_clients()`` already
    computes exactly this count for the attach cap; it is probed rather than
    required so the reduced handles in tests keep working.
    """
    count = getattr(runtime, "attach_clients", None)
    if not callable(count):
        return False
    try:
        live = count()
    except Exception:  # noqa: BLE001 — uncertainty here must not pin the runtime
        logger.debug("attach_clients failed; treating as no viewer", exc_info=True)
        return False
    return isinstance(live, int) and live > 0


def _should_exit(handle: object, runtime: object) -> bool:
    """The residency predicate (design §6.1): exit when ALL three hold.

    1. ``handle.is_busy()`` is False — no turn, compaction, subagents, jobs,
       queued prompts, or gate parked on a user's answer. Work is
       authoritative: nothing below can end a turn early.
    2. No wake is due within :data:`WARM_WINDOW_S` — a runtime about to fire
       its own wake is cheaper kept than re-spawned (see the constant).
    3. No interactive viewer is attached — a user looking at the session is
       about to type, and holding the process warm turns "every message after
       a 3 s pause costs a cold start" into "the first message of a
       conversation costs one".

    Reconciling term 3 with the older rule "watchers and replicas observe
    work; they do not own it": both are still true, and they are about
    different things. OWNERSHIP of the work is the turn's — a viewer leaving
    does not abort a turn (term 1 is checked first and alone decides that),
    and a daemon-class client never holds anything. Term 3 is about
    READINESS: an attached interactive viewer is the one signal that the
    next message is imminent, so residency follows it. The phone's SSE
    watcher count (``phone_watchers``) stays out of the predicate — the
    daemon's connection is not the user's attention, and the phone's
    interactive attach dials as ``"attach"`` when it wants warmth.
    """
    is_busy = getattr(handle, "is_busy", None)
    if is_busy is not None and is_busy():
        return False
    if _wake_within_window(handle):
        return False
    if _viewer_attached(runtime):
        return False
    return True


async def _clean_exit(handle: object, runtime: object) -> None:
    """Dispose the quiescent session, then unpublish its owner record.

    The reaper reaches this only after ordinary gate timeouts and all resumed
    work have drained, so injecting a shutdown denial here would violate the
    same no-interruption invariant that selected this state.
    """
    try:
        await handle.dispose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — dispose is best-effort at exit
        logger.warning("child session dispose failed", exc_info=True)
    try:
        await runtime.aclose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("child runtime aclose failed", exc_info=True)


async def _reaper(handle: object, runtime: object, stop: asyncio.Event) -> None:
    """Exit the disposable session runtime after one uninterrupted idle drain.

    The drain is re-checked every ``REAP_CHECK_S`` against the full predicate,
    so any term flipping back — work arriving, a viewer attaching, a wake
    entering the warm window — cancels it and the clock restarts from the
    next fully-idle tick. A wake that fires during the drain starts a turn,
    which flips ``is_busy()``: that is how "due within the drain" fires once,
    in-process, with no supervisor involvement.
    """
    grace_s = _grace_seconds()
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    next_build_check = time.monotonic() + BUILD_CHECK_S

    async def refresh_check() -> bool:
        """The refresh branch on its own slower cadence. True once the
        runtime has retired (the caller returns). Runs on BOTH loop levels
        below: outside the drain — where an attached viewer, which holds the
        quiet exit, must not hold this — and inside it, because a grace of
        minutes (``LOP_SESSION_GRACE_S``) would otherwise starve the check
        for an unwatched runtime that is exactly the one nobody else will
        ever refresh."""
        nonlocal next_build_check
        if time.monotonic() < next_build_check:
            return False
        next_build_check = time.monotonic() + BUILD_CHECK_S
        newer = _should_refresh(handle, boot)
        if newer is None:
            return False
        return await _refresh_for(newer, handle, runtime, stop)

    while not stop.is_set():
        await asyncio.sleep(REAP_CHECK_S)
        if stop.is_set():
            continue
        if await refresh_check():
            return
        if stop.is_set() or not _should_exit(handle, runtime):
            continue
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            await asyncio.sleep(REAP_CHECK_S)
            if await refresh_check():
                return
            if stop.is_set() or not _should_exit(handle, runtime):
                break  # a predicate term flipped back (or shutdown began)
        else:
            logger.info(
                "session runtime: idle for %.1fs (no work, no viewer, no wake within %.0fs); "
                "exiting cleanly",
                grace_s,
                WARM_WINDOW_S,
            )
            await _clean_exit(handle, runtime)
            stop.set()  # amain's wait() returns; exit code stays 0
            return


async def _refresh_for(
    newer: "BuildStamp", handle: object, runtime: object, stop: asyncio.Event
) -> bool:
    """Announce and retire so the next engage runs ``newer``. True if exited.

    Three checks of the predicate, and each is load-bearing:

    1. Before the stagger (the caller's) — the cheap gate.
    2. After the stagger — work may have arrived while sixteen siblings
       spread their exits over ``BUILD_STAGGER_S``; a runtime that picked up
       a turn meanwhile keeps it and tries again next check.
    3. After the announce — ``announce_retiring`` is an await (it drains each
       viewer's writer), and a ``peer_message`` or ``prompt`` can open a turn
       in that gap. Same shape as ``RuntimeServer._retire_if_pristine``'s
       re-check after its ``stopping`` broadcast. A turn that starts between
       THIS re-check and ``_clean_exit`` is aborted by the dispose exactly as
       a ``stop`` op racing a turn is; the message is persisted and the
       sender's next engage runs the new build.

    ``retiring`` is announced AFTER the stagger, immediately before exit, so
    a viewer never waits on a runtime that is merely "about to" leave.
    """
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    delay = random.uniform(0, _build_stagger_seconds())  # noqa: S311 — jitter, not security
    logger.info(
        "session runtime: build on disk is %s but this process loaded %s; idle, retiring in "
        "%.1fs so the next engage runs the new build",
        newer.label(),
        boot.label() if boot is not None else "<unknown>",
        delay,
    )
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
        return False  # a stop landed during the stagger; its path owns the exit
    except asyncio.TimeoutError:
        pass
    if _should_refresh(handle, boot) is None:
        logger.info("session runtime: work arrived during the refresh stagger; keeping")
        return False
    announce = getattr(runtime, "announce_retiring", None)
    if callable(announce):
        try:
            await cast(Callable[..., Awaitable[None]], announce)("stale-build", to=newer.label())
        except Exception:  # noqa: BLE001 — a viewer that misses this goes cold the slow way
            logger.debug("retiring announcement failed", exc_info=True)
    if stop.is_set():
        return False
    if _should_refresh(handle, boot) is None:
        # Refusing AFTER announcing is safe for the same reason it is for
        # ``stopping``: ``retiring`` only latches the disconnect REASON in an
        # attach client, and does nothing unless the socket then closes.
        logger.info("session runtime: work arrived while retiring was announced; keeping")
        return False
    logger.info("session runtime: retiring for %s", newer.label())
    await _clean_exit(handle, runtime)
    stop.set()
    return True


async def _drain_inbox_into(handle: object) -> int:
    """Deliver every message spooled while this session was cold. Count sent.

    Called from :func:`amain` after the session exists and before the control
    socket listens — see the call site for why that ordering is the delivery
    guarantee rather than an implementation detail.

    Delivery uses the record-only branch (``mode="mailbox"``, ``wake=False``):
    these arrived as QUIET notes, and a spool that opened a turn per message on
    the next open would turn "read this when you next run" into "start work
    now", which is the opposite of what the sender asked for.

    Best-effort per message: one malformed or rejected row must not stop the
    rest, and none of it may prevent the runtime from starting.
    """
    from local_operator.session.runtime.inbox import drain_inbox

    session = getattr(handle, "_session", None)
    directory = getattr(getattr(session, "transcript", None), "directory", None)
    if directory is None:
        return 0
    try:
        lines = await asyncio.to_thread(drain_inbox, directory)
    except Exception:  # noqa: BLE001 — a bad spool must not block the runtime
        logger.warning("inbox drain failed", exc_info=True)
        return 0
    probed = getattr(handle, "receive_peer_message", None)
    if not lines or not callable(probed):
        return 0
    receive = cast(Callable[..., Awaitable[str]], probed)
    delivered = 0
    for line in lines:
        try:
            await receive(line.text, mode="mailbox", wake=False, sender=line.sender)
            delivered += 1
        except Exception:  # noqa: BLE001 — one bad row is not the others' problem
            logger.warning("spooled message could not be delivered", exc_info=True)
    if delivered:
        logger.info("delivered %d spooled message(s) at open", delivered)
    return delivered


async def amain() -> int:
    # Deferred for startup cost, not to break a cycle: importing the owned
    # handle pulls the composition root, and `python -m` on this module must
    # not pay for it before the log file is configured in main().
    from local_operator.session.runtime.owned import (
        OwnedSessionHandle,
        spawn_owned_session,
    )
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session_lease import SessionLeaseHeldError

    cwd = os.environ.get("LOP_MOBILE_CHILD_CWD") or os.path.expanduser("~")
    provider = os.environ.get("LOP_MOBILE_CHILD_PROVIDER") or None
    model_id = os.environ.get("LOP_MOBILE_CHILD_MODEL") or None
    resume = os.environ.get("LOP_MOBILE_CHILD_RESUME") or None
    if resume:
        # A runtime ADOPTS the id it was given rather than requiring a
        # directory to already exist. The viewer mints the session id before
        # anything is on disk (it is a name, not a directory, until there is
        # work), so the first engage of a brand-new session arrives here with
        # nothing to resume — and the strict `--resume` path would refuse it.
        # See ``session_factory._transcript_dir_and_agent_id``.
        os.environ["LOP_RUNTIME_ADOPT_SESSION"] = "1"

    loop = asyncio.get_running_loop()
    try:
        handle: OwnedSessionHandle = await spawn_owned_session(
            loop, cwd=cwd, provider=provider, model_id=model_id, resume=resume
        )
    except SessionLeaseHeldError as exc:
        # LOSING THE LEASE IS NOT AN ERROR. Under ``engage_runtime`` every
        # contender is allowed to spawn a candidate and the lease decides which
        # one lives (session/runtime/launch.py) — so a loser is a race working
        # exactly as designed, and it exits 0. Returning non-zero here made an
        # ordinary ten-way engage look like nine crashes in the logs, and would
        # make a supervisor's KeepAlive treat normal arbitration as a failure
        # loop.
        logger.info(
            "runtime lost the lease for %s to pid %s; exiting",
            resume or "<new>",
            exc.pid,
        )
        return 0
    except Exception as error:
        logger.exception("session runtime child: session construction failed")
        # ALSO to stderr, which the spawning parent captures. `main()` points
        # logging at the daemon's own file, so the traceback above is written
        # where only a person reading logs later can find it -- the parent saw
        # nothing at all and could only report a generic timeout after burning
        # its whole deadline (QA Q1). One line on stderr is what lets an engage
        # fail fast and name the actual cause.
        print(
            f"{type(error).__module__}.{type(error).__qualname__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    # THE ORDERING IS THE GUARANTEE (design §11.4). Messages spooled while the
    # session was cold are delivered here, BEFORE the control socket begins
    # listening, so they cannot be interleaved with an errand a client sends
    # over that socket — there is no socket yet. Draining after
    # ``start_in_process`` would race the engaging caller's own prompt and
    # deliver a note written minutes ago after one written just now.
    await _drain_inbox_into(handle)

    # The wake scheduler is armed HERE, after the inbox drain and before the
    # socket listens. A runtime the supervisor starts for an overdue wake has
    # no errand that delivers a prompt — the WakeErrand carries nothing by
    # design — so the wake's turn comes from the session's own catch-up path,
    # which only runs once the scheduler is pumped. Round 2 (U4/Q9) found the
    # cold runtime never called ``async_init``: the runtime started, idled,
    # and exited, and a one-shot wake was consumed without ever running.
    #
    # After the drain so spooled quiet notes land before the wake's turn
    # starts; before the socket so a client's prompt cannot race the catch-up.
    # ``async_init`` is idempotent and degrades to ``ensure_future`` for
    # background work, so a host that also calls it later changes nothing.
    session = getattr(handle, "_session", None)
    init = getattr(session, "async_init", None)
    if callable(init):
        try:
            result = init()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 — an unarmable scheduler is not a dead runtime
            logger.warning("wake scheduler did not arm at boot", exc_info=True)

    runtime = RuntimeServer(handle, kind="daemon")
    await runtime.start_in_process()

    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    if os.environ.get("LOP_RUNTIME_DEBUG_STACKS") == "1":
        # SIGUSR1 prints every asyncio task's stack to the child log. The
        # child has no terminal and no attached debugger, and a wedged turn
        # (round 2, U6) is exactly the state whose cause is "which await is
        # the turn parked in" — invisible to py-spy without root and to the
        # main-thread faulthandler dump, which shows the loop idling under a
        # parked task. Opt-in so a normal runtime pays nothing.
        def _dump_task_stacks() -> None:
            session = getattr(handle, "_session", None)
            try:
                subagents = session.running_subagents() if session is not None else 0
            except Exception as exc:  # noqa: BLE001 — the dump must not die
                subagents = f"RAISES {type(exc).__name__}: {exc}"
            logger.info(
                "state: streaming=%s compacting=%s lock=%s queue=%s drain_done=%s "
                "subagents=%s is_busy=%s",
                getattr(session, "_is_streaming", "?"),
                getattr(session, "_compacting", "?"),
                getattr(getattr(session, "_turn_lock", None), "locked", lambda: "?")(),
                len(getattr(handle, "_prompt_queue", [])),
                getattr(getattr(handle, "_prompt_drain_task", None), "done", lambda: "?")(),
                subagents,
                handle.is_busy(),
            )
            sig = getattr(session, "_signal", None)
            logger.info(
                "signal: present=%s aborted=%s abort_requested=%s",
                sig is not None,
                getattr(sig, "aborted", None),
                getattr(session, "_abort_requested", "?"),
            )
            for task in asyncio.all_tasks(loop):
                if task.done():
                    continue
                # The parked await is at the BOTTOM of the coroutine chain:
                # each awaited coroutine's frame hangs off the outer one's
                # cr_await, not its f_back, so format_stack alone prints only
                # the outermost frame. Walk the chain to see where the turn
                # is actually parked.
                lines: list[str] = []
                obj: Any = task.get_coro()
                while obj is not None:
                    frame = getattr(obj, "cr_frame", None) or getattr(obj, "gi_frame", None)
                    if frame is None:
                        break
                    code = frame.f_code
                    lines.append(f"  {code.co_filename}:{frame.f_lineno} in {code.co_name}")
                    obj = getattr(obj, "cr_await", None) or getattr(obj, "gi_yieldfrom", None)
                logger.info("task %r await-chain:\n%s", task.get_name(), "\n".join(lines))

        loop.add_signal_handler(signal.SIGUSR1, _dump_task_stacks)
    # The socket ``stop`` op (the kill switch's graceful rung) and SIGTERM
    # converge on the same event, so the deny → dispose → aclose ordering
    # below runs once, identically, for both triggers.
    handle.on_stop_requested = stop.set
    # The self-reaper: a phone session nobody watches and nothing runs is a
    # live process doing nothing, and before this it idled FOREVER. Runs
    # beside the signal wait; whichever fires first wins.
    reaper = asyncio.ensure_future(_reaper(handle, runtime, stop))
    reaper_ran_clean_exit = False
    await stop.wait()
    if not reaper.done():
        reaper.cancel()
    elif reaper.exception() is None:
        # The reaper completed (not was cancelled): it already ran the clean
        # ordering. A signal-initiated stop still owes it.
        reaper_ran_clean_exit = True
    if not reaper_ran_clean_exit:
        try:
            handle._deny_pending_gates()
        except Exception:  # noqa: BLE001 — shutdown must proceed
            logger.debug("child gate deny failed", exc_info=True)
        try:
            await handle.dispose()
        except Exception:  # noqa: BLE001
            logger.warning("child session dispose failed", exc_info=True)
    await runtime.aclose()

    # Under `LOP_RUNTIME_DEFER_MATERIALISE` the transcript and roster sidecar
    # never create the session directory, but the LEASE cannot be deferred —
    # it arbitrates "at most one runtime per session" and lives inside the
    # directory — so a speculatively warmed runtime that was never given real
    # work leaves a lease-only `sessions/<id>/` behind. This exit path USED
    # TO `rmdir` that directory (#622, `_remove_unwritten_session_dir`). It
    # no longer does, and must not: the operator's logs show it firing on a
    # real store, and the rule after that incident is that no exit hook, no
    # startup hook and no sweep removes a session directory on its own
    # judgement — a lease-only directory is exactly what the user-enabled
    # `session.cleanup.remove_empty` policy exists for, and it is off by
    # default. `tests/unit/session/test_no_session_deletion.py` forbids
    # reintroducing an `rmdir` here.
    return 0


def main() -> int:
    # A child has no terminal and no inherited log stream — without this its
    # warnings (a failed prompt, a dead provider) vanish, which is how a
    # silently-dropped turn went undiagnosed. The daemon's own log file is
    # the natural place: `lop mobile logs` covers both.
    from local_operator.paths import log_dir

    log_dir().mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_dir() / "mobile.log"),
    )
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

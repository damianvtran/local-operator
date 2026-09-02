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
import logging
import os
import signal
import sys
import time

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


def _grace_seconds() -> float:
    raw = os.environ.get("LOP_SESSION_GRACE_S", "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_GRACE_S
    return value if value > 0 else DEFAULT_GRACE_S


def _wake_within_window(handle: object, *, now_ms: int | None = None) -> bool:
    """Term 2 of the predicate: does the runtime's OWN scheduler have a wake
    due within ``WARM_WINDOW_S``? Read through the handle (an optional
    capability, probed) so reduced test handles and older hosts that never
    grew the accessor behave as "no wakes" rather than crash the reaper."""
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
    while not stop.is_set():
        await asyncio.sleep(REAP_CHECK_S)
        if stop.is_set() or not _should_exit(handle, runtime):
            continue
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            await asyncio.sleep(REAP_CHECK_S)
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


async def amain() -> int:
    # Deferred for startup cost, not to break a cycle: importing the owned
    # handle pulls the composition root, and `python -m` on this module must
    # not pay for it before the log file is configured in main().
    from local_operator.session.runtime.owned import (
        OwnedSessionHandle,
        spawn_owned_session,
    )
    from local_operator.session.runtime.server import RuntimeServer

    cwd = os.environ.get("LOP_MOBILE_CHILD_CWD") or os.path.expanduser("~")
    provider = os.environ.get("LOP_MOBILE_CHILD_PROVIDER") or None
    model_id = os.environ.get("LOP_MOBILE_CHILD_MODEL") or None
    resume = os.environ.get("LOP_MOBILE_CHILD_RESUME") or None

    loop = asyncio.get_running_loop()
    try:
        handle: OwnedSessionHandle = await spawn_owned_session(
            loop, cwd=cwd, provider=provider, model_id=model_id, resume=resume
        )
    except Exception:
        logger.exception("session runtime child: session construction failed")
        return 2

    runtime = RuntimeServer(handle, kind="daemon")
    await runtime.start_in_process()

    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
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

"""A daemon-spawned session process: ``python -m local_operator.mobile.child``.

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
arrives or its work and pending gates drain long enough for self-reaping (see
:func:`_reaper`). Environment variables are the
spawn contract (``LOP_MOBILE_CHILD_CWD``, ``_PROVIDER``, ``_MODEL``) — argv
would be ps-readable.
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
#: one event loop. Viewer traffic is deliberately not an input to this loop.
REAP_CHECK_S = 0.25

#: Idle execution hosts are disposable once their durable session state is
#: quiescent. This is a drain for newly arriving work, not a reconnect grace.
DEFAULT_GRACE_S = 3.0


def _grace_seconds() -> float:
    raw = os.environ.get("LOP_SESSION_GRACE_S", "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_GRACE_S
    return value if value > 0 else DEFAULT_GRACE_S


def _should_exit(handle: object, registrant: object) -> bool:
    """True when the execution host is quiescent, regardless of viewers."""
    del registrant  # Watchers and replicas observe work; they do not own it.
    is_busy = getattr(handle, "is_busy", None)
    return not (is_busy is not None and is_busy())


async def _clean_exit(handle: object, registrant: object) -> None:
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
        await registrant.aclose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("child registrant aclose failed", exc_info=True)


async def _reaper(handle: object, registrant: object, stop: asyncio.Event) -> None:
    """Exit the disposable execution host after one uninterrupted idle drain.

    Autonomous work and ordinary pending-gate timeouts are authoritative. Any
    new activity resets the drain; viewer count intentionally does not.
    """
    grace_s = _grace_seconds()
    while not stop.is_set():
        await asyncio.sleep(REAP_CHECK_S)
        if stop.is_set() or not _should_exit(handle, registrant):
            continue
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            await asyncio.sleep(REAP_CHECK_S)
            if stop.is_set() or not _should_exit(handle, registrant):
                break  # work arrived (or shutdown began)
        else:
            logger.info("mobile child: idle for %.1fs; exiting cleanly", grace_s)
            await _clean_exit(handle, registrant)
            stop.set()  # amain's wait() returns; exit code stays 0
            return


async def amain() -> int:
    from local_operator.mobile.owned import OwnedSessionHandle, spawn_owned_session
    from local_operator.mobile.registrant import Registrant

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
        logger.exception("mobile child: session construction failed")
        return 2

    registrant = Registrant(handle, kind="daemon")
    await registrant.start_in_process()

    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    # The self-reaper: a phone session nobody watches and nothing runs is a
    # live process doing nothing, and before this it idled FOREVER. Runs
    # beside the signal wait; whichever fires first wins.
    reaper = asyncio.ensure_future(_reaper(handle, registrant, stop))
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
    await registrant.aclose()
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

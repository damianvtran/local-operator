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
it through the normal record + control socket path, and idles until its work
ends, a signal arrives, or — since the attach/reaping design — no front end
holds it any longer (see :func:`_reaper`). Environment variables are the
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

#: How often the reaper re-evaluates "is anyone still holding this session?".
#: Short enough that a front end leaving is noticed within the grace budget's
#: own resolution, long enough that five-second polls cost nothing.
REAP_CHECK_S = 5.0

#: Default grace before a held-by-nobody session exits, overridable with
#: ``LOP_SESSION_GRACE_S``. Long enough that a phone screen-lock / tunnel
#: blip (SSE drop + reconnect) never kills a session the user is coming back
#: to; short enough that unattended sessions cannot accumulate as live
#: processes — the buildup this reaper exists to prevent.
DEFAULT_GRACE_S = 120.0


def _grace_seconds() -> float:
    raw = os.environ.get("LOP_SESSION_GRACE_S", "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_GRACE_S
    return value if value > 0 else DEFAULT_GRACE_S


def _should_exit(handle: object, registrant: object) -> bool:
    """True iff NO front end holds the session and no work is running.

    The mixed-version guard is load-bearing: until the registrant has SEEN a
    watch/unwatch op (``watch_supported``), phone watchers are UNKNOWN — an
    old daemon never sends the ops — and unknown must count as present, or a
    new child under an old daemon would reap a session a phone is actively
    watching. The latch never resets, so once a watch-capable daemon has
    spoken, a true zero is trusted forever after.
    """
    watchers_known_empty = (
        bool(getattr(registrant, "watch_supported", False))
        and getattr(registrant, "phone_watchers", 1) == 0
    )
    if not watchers_known_empty:
        return False
    if getattr(registrant, "attach_clients", lambda: 0)() > 0:
        return False
    is_busy = getattr(handle, "is_busy", None)
    if is_busy is not None and is_busy():
        return False
    return True


async def _clean_exit(handle: object, registrant: object) -> None:
    """Deny gates, dispose the session (releases the claim), unpublish.

    Ordering mirrors OperatorApp.on_unmount: the gates are refused BEFORE
    dispose because dispose awaits teardown and a turn parked on an
    unanswered card would never reach it; the record is unpublished LAST so
    a scanner never sees a gone record while the claim is still held.
    """
    deny = getattr(handle, "_deny_pending_gates", None)
    if deny is not None:
        try:
            deny()
        except Exception:  # noqa: BLE001 — shutdown must proceed
            logger.debug("child gate deny failed", exc_info=True)
    try:
        await handle.dispose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — dispose is best-effort at exit
        logger.warning("child session dispose failed", exc_info=True)
    try:
        await registrant.aclose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("child registrant aclose failed", exc_info=True)


async def _reaper(handle: object, registrant: object, stop: asyncio.Event) -> None:
    """Exit the child cleanly once no front end has held it for a grace period.

    The child is the only party with the authoritative liveness facts (its
    turn, its subagents, its connections), so the child reaps ITSELF — the
    daemon never kills children. The grace timer starts at the earliest
    moment all hold conditions are false: a turn mid-flight when the last
    front end leaves simply defers the start (``is_busy`` gates it), so the
    grace outlives the turn by construction. Any holder returning mid-grace
    cancels the countdown.
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
                break  # a front end came back (or shutdown began)
        else:
            logger.info(
                "mobile child: no front end for %.0fs — exiting cleanly", grace_s
            )
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

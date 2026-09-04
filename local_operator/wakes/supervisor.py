"""The wake supervisor: the process that exists so a wake can fire at all.

A wake scheduled in a session whose terminal is then closed had nowhere to
fire from — the schedule was durable, but nothing was running to notice it
came due. This is the small always-on process that closes that gap: it reads
:mod:`local_operator.wakes.store`, sleeps until the earliest due time across
every session, and starts a runtime for whichever session's wake is due.

**It does not deliver the wake, and that is the design, not an omission.**
The obvious shape — a ``wake_fire`` control op naming the occurrence to
deliver — fires every wake TWICE, because a session already delivers its own
overdue wakes on load: ``WakeScheduler.load`` re-arms anything whose
``next_due_at`` has passed to ``now + LOAD_GRACE_MS`` and records it for the
resume catch-up (``harness/wake.py``). So the mere EXISTENCE of a runtime is
what fires the wake, and an op on top of that would append the occurrence a
second time.

This keeps one writer of schedule state (``Session._persist_wake_schedules``),
which is the property that matters: the supervisor never advances, retires or
persists a schedule, so it can never disagree with the session about what has
fired. Its whole job is "make a runtime exist for this session, now".

**The no-live-record rule.** A session with a live discovery record is
already running and fires its own wakes through its own scheduler. The
supervisor therefore SKIPS it entirely — engaging there would be redundant at
best, and at worst a second opinion about a schedule the live session is
actively advancing. The rule is checked at fire time rather than at scan
time, because a session can come up during the sleep.

**Self-retirement.** An empty index means nothing left to supervise, so the
process exits 0 and its LaunchAgent (``KeepAlive: {SuccessfulExit: False}``)
leaves it down. That is why the exit code matters: a crash restarts, a
finished job stays finished. The next persist reinstalls it.

Stdlib-only and import-light, like the rest of ``wakes/``: this runs as its
own supervised process and must not drag the harness in at import.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Never sleep longer than this, however far away the next wake is. A long
#: sleep is not a correctness problem (the due time is recomputed on every
#: pass) but it is an OBSERVABILITY one: a supervisor asleep for nine hours
#: has not noticed a schedule cancelled eight hours ago, so its own liveness
#: and the index's state drift apart. Waking hourly to re-read is cheap.
MAX_SLEEP_S = 3600.0

#: Never sleep less than this. A schedule due in the past, or a clock that
#: jumped backwards, must not turn the loop into a spin — the floor is what
#: bounds a pathological index to one pass per second rather than thousands.
MIN_SLEEP_S = 1.0

#: How late a wake may be before the supervisor stops treating it as due and
#: leaves it to the session's own catch-up. Sessions handle arbitrarily-old
#: overdue wakes at load (that is what the resume catch-up IS), so there is no
#: value in the supervisor racing to start a runtime for one that has been due
#: for a week — the user will get the catch-up when they open it.
STALE_AFTER_S = 7 * 24 * 3600.0


def _due_sessions(index: dict[str, dict[str, Any]], now_ms: int) -> list[tuple[str, str, int]]:
    """``(session_id, cwd, due_ms)`` for every session with a wake due now.

    Reads the index defensively — it is written by other processes and a
    hand-edited or half-written entry must cost one session's wake, never the
    whole sweep.
    """
    from local_operator.wakes.store import next_due_at

    due: list[tuple[str, str, int]] = []
    for session_id, entry in index.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("stopped_at"):
            # Dormant: the session was deliberately stopped, and PR 3's
            # contract is that its wakes stay armed but do not fire until the
            # user reopens it. Firing here would resurrect a session the kill
            # switch ended.
            continue
        earliest = next_due_at(entry)
        if earliest is None or earliest > now_ms:
            continue
        if (now_ms - earliest) / 1000.0 > STALE_AFTER_S:
            continue
        cwd = entry.get("cwd")
        due.append(
            (session_id, cwd if isinstance(cwd, str) and cwd else os.path.expanduser("~"), earliest)
        )
    # Oldest first: if several are due at once, the one that has waited
    # longest gets its runtime first.
    due.sort(key=lambda row: row[2])
    return due


def _next_wake_ms(index: dict[str, dict[str, Any]]) -> int | None:
    """Earliest future due time across the whole index, or None."""
    from local_operator.wakes.store import next_due_at

    earliest: int | None = None
    for entry in index.values():
        if not isinstance(entry, dict) or entry.get("stopped_at"):
            continue
        due = next_due_at(entry)
        if due is None:
            continue
        if earliest is None or due < earliest:
            earliest = due
    return earliest


async def _has_live_runtime(config_dir: Path, session_id: str) -> bool:
    """Whether a runtime is already hosting ``session_id``.

    The no-live-record rule's implementation. Off the loop: it walks the
    record directory.
    """
    from local_operator.mobile.attach_client import find_owner_record

    record, _owner = await asyncio.to_thread(find_owner_record, config_dir, session_id)
    return record is not None


async def fire_due_wakes(config_dir: Path, *, now_ms: int | None = None) -> int:
    """Start a runtime for every cold session whose wake is due. Returns count.

    One pass. Split out from :func:`serve` so the whole decision — which
    sessions are due, which are skipped for being live, what engaging does —
    is testable without a loop or a clock.
    """
    from local_operator.session.runtime.launch import WakeErrand, engage_runtime
    from local_operator.wakes.store import read_index

    moment = now_ms if now_ms is not None else int(time.time() * 1000)
    index = await asyncio.to_thread(read_index, config_dir)
    fired = 0
    for session_id, cwd, due_ms in _due_sessions(index, moment):
        if await _has_live_runtime(config_dir, session_id):
            # The no-live-record rule: that session fires its own wakes.
            logger.debug("wake for %s is due but it is already running", session_id)
            continue
        try:
            # A derived id, not a random one: a supervisor that crashes after
            # engaging and retries must not start a second runtime for the
            # same occurrence.
            await engage_runtime(
                session_id,
                cwd,
                WakeErrand(
                    schedule_id="",
                    occurrence_ms=due_ms,
                    command_id=f"wake-{session_id}-{due_ms}",
                ),
                config_dir=config_dir,
            )
        except (TimeoutError, ConnectionError, OSError) as exc:
            # One session's wake failing must not stop the others'. The
            # schedule is untouched, so the next pass tries again.
            logger.warning("could not start a runtime for %s: %s", session_id, exc)
            continue
        logger.info("started a runtime for %s (wake due %d)", session_id, due_ms)
        fired += 1
    return fired


async def serve(config_dir: Path, *, once: bool = False) -> int:
    """Run the supervisor loop. Returns the process exit code.

    Exits **0** when the index empties, which is what lets the LaunchAgent's
    ``KeepAlive: {SuccessfulExit: False}`` leave a finished supervisor down
    instead of restarting it forever. A crash exits non-zero and is restarted.
    """
    from local_operator.wakes.store import read_index

    while True:
        index = await asyncio.to_thread(read_index, config_dir)
        if not index:
            logger.info("wake index is empty; the supervisor is retiring")
            return 0

        await fire_due_wakes(config_dir)
        if once:
            return 0

        # Recomputed from the index AFTER firing, so a schedule the fired
        # runtime advanced is already reflected rather than re-read stale.
        index = await asyncio.to_thread(read_index, config_dir)
        if not index:
            logger.info("wake index is empty; the supervisor is retiring")
            return 0
        upcoming = _next_wake_ms(index)
        now_ms = int(time.time() * 1000)
        if upcoming is None:
            delay = MAX_SLEEP_S
        else:
            delay = max(MIN_SLEEP_S, min(MAX_SLEEP_S, (upcoming - now_ms) / 1000.0))
        logger.debug("sleeping %.1fs until the next wake", delay)
        await asyncio.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    """``python -m local_operator.wakes.supervisor`` — the LaunchAgent entry."""
    parser = argparse.ArgumentParser(
        prog="local-operator-wake-supervisor",
        description="Fire scheduled wakes for sessions that are not running.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single pass and exit (diagnostics; the agent runs the loop)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from local_operator.paths import config_dir

    try:
        return asyncio.run(serve(config_dir(), once=args.once))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

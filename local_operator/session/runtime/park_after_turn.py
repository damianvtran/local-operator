"""Deferred self-park: park THIS session once the turn that asked for it ends.

WHY A DETACHED HELPER PROCESS. ``sessions op=park`` with no address is the
session volunteering its memory back (field gap 2026-11-12: idle sessions under
swap pressure could not park themselves, so the operator's agent had to stop
them from outside and their wakes went dormant). The tool call runs INSIDE the
runtime it wants to end, mid-turn, so it can do neither half itself:

* ending the runtime now would cut the very turn that asked — the owner's
  ``stop_if_idle`` latch (``ServingSessionHandle.begin_retire``) refuses while
  the turn runs, which is exactly the protection we want to keep;
* the ``parked_at`` stamp is written AFTER the runtime has exited
  (``control.stop_session``), and an in-process task would die with it.

So the tool spawns this module as a detached process (its own session/process
group, so it survives the runtime's exit and is never reaped with it). It runs
the ordinary :func:`control.park_session` against the caller's pid in a loop:
``busy`` (the turn is still finishing) is retried every :data:`POLL_S`; the
first idle moment parks the session through the same idle-only ladder every
other park uses, and the helper exits. It gives up — leaving the session up,
never signalling it — on any other refusal (a wake due inside the hold window,
an old owner, a silent socket), when the runtime is already gone, or after
:data:`DEADLINE_S` of continuous work. A new turn started after the park
request does NOT cancel it: the helper parks at the first idle moment after
that turn as well, provided it comes within :data:`DEADLINE_S`; only work that
outlasts the deadline keeps the session up (review round 1, R1-8).

One log line per outcome goes to ``<log_dir>/park-after-turn.log``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from local_operator.session.runtime.park_pressure import plural_noun

logger = logging.getLogger("local_operator.session.runtime.park_after_turn")

#: Re-try cadence while the owner reports ``busy``. A turn's tail (the final
#: assistant message, transcript persist) is typically sub-second to a few
#: seconds; 2 s keeps the park prompt without hammering the control socket.
POLL_S = 2.0

#: Give up after this long without an idle moment. Long enough for a slow final
#: message under load; short enough that a park the operator overtook (by
#: starting new work in the session) does not fire half an hour later.
DEADLINE_S = 600.0


def spawn(pid: int, session_id: str) -> int:
    """Start the helper for ``pid``/``session_id`` detached; returns its pid.

    ``start_new_session=True`` puts it in its own process group, so the
    runtime's exit (and any group-wide teardown of it) does not take the helper
    down before it has stamped the park. Inherits the environment, so
    ``LOCAL_OPERATOR_CONFIG_DIR`` and the interpreter match the caller's.
    """
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, our own interpreter
        [
            sys.executable,
            "-m",
            "local_operator.session.runtime.park_after_turn",
            "--pid",
            str(pid),
            "--session",
            session_id,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid


def _find_record(root: Path, pid: int, session_id: str):  # type: ignore[no-untyped-def]
    """The live discovery record for exactly this pid AND session, or None.

    Both must match: a recycled pid now hosting another session, or this
    session reopened elsewhere, is not the runtime that asked to be parked.
    """
    from local_operator.session.runtime import registry

    for record, _state in registry.scan(root, reap=False):
        if record.pid == pid and record.session_id == session_id:
            return record
    return None


async def run(
    pid: int,
    session_id: str,
    *,
    root: Path | None = None,
    poll_s: float = POLL_S,
    deadline_s: float = DEADLINE_S,
) -> str:
    """Park ``session_id`` (on ``pid``) at its first idle moment.

    Returns the outcome token it logged: ``parked``, ``gone``, ``refused:<reason>``
    or ``deadline``.
    """
    from local_operator.paths import config_dir
    from local_operator.session.runtime import control

    target_root = root if root is not None else config_dir()
    give_up_at = time.monotonic() + deadline_s
    while True:
        record = _find_record(target_root, pid, session_id)
        if record is None:
            logger.info(
                "self-park %s (pid %s): runtime already gone; nothing to park", session_id, pid
            )
            return "gone"
        outcome = await control.park_session(
            record, _root=target_root, _command="sessions tool park (self)"
        )
        if outcome.parked:
            logger.info(
                "self-park %s (pid %s): parked — %s and %s stay armed",
                session_id,
                pid,
                plural_noun(outcome.wakes_armed, "wake"),
                plural_noun(outcome.monitors_armed, "monitor"),
            )
            return "parked"
        if outcome.method == "refused" and outcome.reason == control.REASON_BUSY:
            if time.monotonic() >= give_up_at:
                logger.info(
                    "self-park %s (pid %s): still working after %.0fs; left running",
                    session_id,
                    pid,
                    deadline_s,
                )
                return "deadline"
            await asyncio.sleep(poll_s)
            continue
        token = outcome.reason or outcome.method
        logger.info(
            "self-park %s (pid %s): not parked (%s): %s", session_id, pid, token, outcome.line
        )
        return f"refused:{token}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="park_after_turn", description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    try:
        from local_operator.paths import log_dir

        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(directory / "park-after-turn.log"),
            level=logging.INFO,
            format="%(asctime)s %(process)d %(message)s",
        )
    except OSError:
        logging.basicConfig(level=logging.INFO)
    outcome = asyncio.run(run(args.pid, args.session))
    return 0 if outcome in ("parked", "gone") else 1


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess
    os.umask(0o077)
    raise SystemExit(main())

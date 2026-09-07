"""Launching a child process this app does not wait for.

One helper, extracted from ``tui/notify`` when the fork spawn backends needed
the identical shape. It is stated once because the three properties below are
easy to get subtly wrong and each omission fails in a way that points somewhere
else entirely:

- ``start_new_session=True`` puts the child in its own session and process
  group, so it survives this process exiting and — more importantly — a Ctrl-C
  in this terminal does not deliver SIGINT to it.
- stdio fully redirected to ``DEVNULL``. A child that inherits this process's
  stdout writes bytes straight into the middle of a painted Textual frame, and
  one that inherits stdin competes with the input loop for keystrokes.
- never waited on and never polled. The child is a side effect (a toast, a new
  terminal window); its exit status tells this app nothing it can act on, and
  waiting for a hung emulator or a stalled D-Bus activation would hold the
  event loop.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


def spawn_detached(
    argv: Sequence[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    label: str | None = None,
) -> bool:
    """Fire-and-forget ``argv``. Returns True when the child was STARTED.

    Never raises, by contract: every caller is on a best-effort path where the
    only correct response to a failure is to carry on. The return value
    distinguishes "could not even start it" (a missing binary, a bad cwd) from
    success, which is what lets a fork tell the user its window did not open —
    but it deliberately says NOTHING about whether the child then succeeded,
    because that would require the wait this function exists not to do.

    ``cwd`` matters for the spawn backends: a new terminal window must open in
    the session's own working directory, not wherever this process happens to
    be, or the restored conversation points at the wrong project.

    ``label`` names the child in the OS process listing when ``argv[0]`` is a
    Python interpreter. The image is switched to the branded hardlink via
    ``executable=`` (what Activity Monitor reads) and ``argv[0]`` becomes the
    label (what ``ps``/``top`` read) — the two independent name axes documented
    in :mod:`local_operator.procname`. Opt-in and best-effort: with no branded
    image available the argv is spawned exactly as passed, so a caller can set
    it unconditionally.
    """
    launch = list(argv)
    executable: str | None = None
    if label and launch:
        from local_operator import procname

        link = procname.ensure_branded_interpreter()
        # Only rewrite when argv[0] really is the interpreter we would be
        # replacing. A spawn of some OTHER binary (a terminal emulator, `open`)
        # must keep its own image, or `executable=` would run Python under that
        # program's arguments.
        if link is not None and os.path.realpath(launch[0]) == os.path.realpath(sys.executable):
            executable = str(link)
            launch[0] = procname.branded_argv0(label)

    try:
        subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            launch,
            executable=executable,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        # Best-effort by design: a missing binary or a spawn failure must never
        # surface as an error in a session, because the user asked for a task
        # and not for a toast (or a window).
        logger.debug("detached spawn failed: %s", list(argv)[:1], exc_info=True)
        return False

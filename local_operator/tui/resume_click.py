"""Open a session in the user's terminal — the notification's click action.

``python -m local_operator.tui.resume_click <session-id>``

A desktop notification is only sent when NOTHING is watching the session, so
by definition there is no emulator around the sending process to inherit. The
click therefore has to OPEN a terminal, which is exactly what the fork
machinery already does: :func:`local_operator.spawn.registry.active_backend`
picks Ghostty / kitty / WezTerm / Apple Terminal, and
:func:`local_operator.multiplexer.broadcast.resume_argv` builds the
restore-and-idle command line behind its safety boundary (no prompt, no
``--exec``, nothing that continues an interrupted turn unattended).

Kept as a module rather than inlined in the notifier's shell command for two
reasons. The click command is embedded in an ``NSTask`` shell string, so the
less quoting it carries the fewer ways it can break on a session id or a path
with a space. And a backend choice made at CLICK time is better than one made
when the notification was posted — the user may have opened a terminal in
between, which is the common case for "I came back to my desk".

Best-effort, like everything on this path: an unpickable backend falls back to
launching the resume argv directly, and every failure is silent. The user's
recourse is the same either way — `lop --resume <id>` in their own terminal.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def open_session(session_id: str) -> bool:
    """Open ``session_id`` in a terminal. True if something was launched."""
    from local_operator.multiplexer.broadcast import resume_argv, resume_executable
    from local_operator.spawn.registry import active_backend
    from local_operator.spawn.types import ForkLaunch, env_or_process

    executable = resume_executable()
    argv = tuple(resume_argv(session_id, executable))
    env = env_or_process(None)

    # `cwd` is the process's own: a notification click carries no project
    # context, and `--resume` restores the session's real working directory
    # from its transcript anyway.
    import os

    launch = ForkLaunch(
        session_id=session_id,
        executable=executable,
        argv=argv,
        cwd=os.getcwd(),
        title="local-operator",
    )

    backend = None
    try:
        backend = active_backend(env)
    except Exception:  # noqa: BLE001 — a backend bug must not eat the click
        logger.debug("could not select a terminal backend", exc_info=True)

    if backend is not None:
        try:
            if backend.spawn(launch, env):
                return True
        except Exception:  # noqa: BLE001 — fall through to the bare launch
            logger.debug("terminal backend refused the launch", exc_info=True)

    # No emulator we know: run the resume line directly. On a desktop this
    # usually does nothing visible, but it is strictly better than dropping
    # the user's click, and it keeps this path honest about its fallback.
    try:
        from local_operator.proc import spawn_detached

        return bool(spawn_detached(list(argv)))
    except Exception:  # noqa: BLE001
        logger.debug("resume launch failed", exc_info=True)
        return False


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return 2
    return 0 if open_session(args[0]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

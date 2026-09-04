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


def _session_cwd(session_id: str) -> str:
    """Where the session was working, best effort.

    Read from the discovery record (live session) and otherwise from the wake
    index, which keeps a ``cwd`` for cold sessions. Falls back to the user's
    home: an unknown project directory is a mild annoyance, whereas defaulting
    to this process's cwd puts the user in a runtime's disposable worktree.
    """
    import os

    try:
        from local_operator.paths import config_dir
        from local_operator.session.runtime import registry

        for record, _state in registry.scan(config_dir()):
            if record.session_id == session_id and record.cwd:
                return str(record.cwd)
    except Exception:  # noqa: BLE001 — a missing record is an ordinary answer
        logger.debug("could not read the session record for cwd", exc_info=True)

    try:
        from local_operator.paths import config_dir
        from local_operator.wakes import store as wake_store

        entry = wake_store.read_entry(config_dir(), session_id) or {}
        cwd = entry.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    except Exception:  # noqa: BLE001
        logger.debug("could not read the wake entry for cwd", exc_info=True)

    return os.path.expanduser("~")


def open_session(session_id: str) -> bool:
    """Open ``session_id`` in a terminal. True if something was launched."""
    import shutil

    from local_operator.multiplexer.broadcast import resume_argv, resume_executable
    from local_operator.spawn.registry import active_backend
    from local_operator.spawn.types import ForkLaunch, env_or_process

    # PATH first, `resume_executable()` second. This module runs as `lop
    # resume-click`, so `argv[0]` is usually right — but it is also reachable
    # as `python -m`, where `resume_executable()` returns the interpreter and
    # the terminal would open a REPL instead of the session. Resolving `lop`
    # on PATH is what the user would type themselves.
    executable = shutil.which("lop") or resume_executable()
    argv = tuple(resume_argv(session_id, executable))
    env = env_or_process(None)

    # THE SESSION'S OWN DIRECTORY, not this process's. The click is handled by
    # whatever process the notification activated — inheriting its cwd landed
    # the user's new terminal in the runtime's worktree, which on a shared
    # machine is a disposable checkout that may not even exist any more. The
    # session's cwd is recorded when it is published, so read it back and fall
    # back to the user's home rather than to an arbitrary directory.
    launch = ForkLaunch(
        session_id=session_id,
        executable=executable,
        argv=argv,
        cwd=_session_cwd(session_id),
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

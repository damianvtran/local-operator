"""Take the user to a session — the notification's click action.

``python -m local_operator.tui.resume_click <session-id>``

**THE PREMISE THIS MODULE WAS BUILT ON EXPIRED, AND THE STALE VERSION OF THIS
DOCSTRING IS HOW A BUG SHIPPED.** It used to open with: "a desktop notification
is only sent when NOTHING is watching the session, so by definition there is no
emulator around the sending process to inherit. The click therefore has to OPEN
a terminal." That was true when written — ``detached_notify`` had one caller,
``session/runtime/owned.py::_announce_pending``, which fires only after
``_watching_surfaces()`` comes back empty.

Background-completion notifications (#724) added a second caller: the observer
leg in ``tui/app.py::_deliver_background_completion`` posts a toast **from a
live, running TUI** for a background session that TUI can already display
through its own sidebar. The handler then did exactly what the docstring said
and spawned a terminal — so clicking a notification opened a second window
running a second process for a session that was one keystroke away. The
operator reported it, and an orphaned Ghostty from this exact path
(``ghostty --initial-command=... lop --resume <id>``) was still resident when
this was rewritten.

So the premise is no longer assumed; it is **checked**. The ladder:

1. **A live viewer can display it** — tell that process to switch and to bring
   its window forward. No new process, no second copy of the session.
2. **Nothing suitable is running** — the original premise genuinely holds, and
   the spawn below is exactly what it always was. That path is why this module
   exists and it is deliberately unchanged.

The spawn still uses the fork machinery wholesale:
:func:`local_operator.spawn.registry.active_backend` picks Ghostty / kitty /
WezTerm / Apple Terminal, and
:func:`local_operator.multiplexer.broadcast.resume_argv` builds the
restore-and-idle command line behind its safety boundary (no prompt, no
``--exec``, nothing that continues an interrupted turn unattended).

**Switching a viewer is a DIFFERENT ACT from spawning a resume, and both stay
honest.** The spawn path restores a transcript into a new process behind
``resume_argv``'s boundary. The viewer path starts no process at all — it asks
a window the user already has to display something it could already display, by
the same route as pressing the sidebar row. It is strictly the safer of the
two, and it never touches the attention store: a click must not mark anything
read.

**SOLICITED FOCUS LIVES HERE AND ONLY HERE.** A background agent must never
take the user's focus. A user clicking a notification is the paradigm case of
asking for it. That distinction is encoded structurally rather than with a
runtime flag: this module is reached only from the notifier's click action, so
the activation call is reachable only from a human gesture, and a grep proves
it. Do not call the focus helper from a poller, a refresh, or a notification
poster.

Kept as a module rather than inlined in the notifier's shell command for two
reasons. The click command is embedded in an ``NSTask`` shell string, so the
less quoting it carries the fewer ways it can break on a session id or a path
with a space. And a backend choice made at CLICK time is better than one made
when the notification was posted — the user may have opened a terminal in
between, which is the common case for "I came back to my desk".

Best-effort, like everything on this path: an unreachable viewer falls through
to the spawn, an unpickable backend falls back to launching the resume argv
directly, and every failure is silent. The user's recourse is the same either
way — `lop --resume <id>` in their own terminal.
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
    """Take the user to ``session_id``. True if anything was achieved.

    Tries the in-place route first (a running viewer switches and comes
    forward), and falls back to spawning a terminal when nothing suitable is
    running. The fallback is the behaviour that shipped before viewers existed
    and is deliberately byte-identical to it.
    """
    if _route_to_viewer(session_id):
        return True
    return _spawn_terminal(session_id)


def _route_to_viewer(session_id: str) -> bool:
    """Ask an already-running viewer to display the session. True if one did.

    Every failure mode — no viewer, a wedged one, a refused socket, a viewer
    that died between the scan and the dial — returns False and falls through
    to the spawn, which is the path that works when nothing is running. The
    dial is bounded end to end (``viewer_client``), so a wedged viewer costs a
    second or two rather than the click.

    The import is local and stays that way: this module is reached from a
    detached click process where startup cost is the user's latency, and
    nothing above needs the viewer stack.
    """
    try:
        from local_operator.session.runtime.viewer_client import route_click

        outcome = route_click(session_id)
    except Exception:  # noqa: BLE001 — routing must never eat the click
        logger.debug("viewer routing failed; falling back to a spawn", exc_info=True)
        return False
    if outcome.switched:
        logger.debug("click routed to a live viewer: %s", outcome.detail)
        return True
    return False


def _spawn_terminal(session_id: str) -> bool:
    """Open ``session_id`` in a NEW terminal. True if something was launched.

    THE ORIGINAL PATH, unchanged. Reached only when no viewer can take the
    click, which is the condition the module's first docstring assumed always
    held. It is the only route on a machine with no TUI running, so its argv
    and its fallbacks are preserved exactly.
    """
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

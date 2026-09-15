"""Take the user to a session — the notification's click action.

``python -m local_operator.tui.resume_click <session-id>``

**THE PREMISE THIS MODULE WAS BUILT ON EXPIRED, AND THE STALE VERSION OF THIS
DOCSTRING IS HOW A BUG SHIPPED.** It used to open with: "a desktop notification
is only sent when NOTHING is watching the session, so by definition there is no
emulator around the sending process to inherit. The click therefore has to OPEN
a terminal." That was true when written — ``detached_notify`` had one caller,
``session/runtime/serving.py::_announce_pending``, which fires only after
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
2. **A desktop app is installed but not running** — launch it WITH the session
   id (`--open-session <id>`), so it opens straight into the conversation
   rather than into the catalogue. This is the rung that makes a `nothing is
   running` banner's click land in the app instead of spawning a terminal, which
   is the surface the user actually wants for a conversation they are being
   told finished.
3. **Nothing suitable is running** — the original premise genuinely holds, and
   the spawn below is exactly what it always was. That path is why this module
   exists and it is deliberately unchanged.

RUNG 2 IS ORDERED AFTER THE VIEWER RUNG and not before it, because a RUNNING
app is a viewer and is reached by rung 1 — including the macOS case where it
has no window, which its record reports and `needs_switch` honours. Rung 2 is
strictly about a process that does not exist yet.

**AND ANY TEST THAT DRIVES ``open_session`` MUST DOUBLE RUNG 2.** It is not a
hypothetical hazard: rung 2 looks for ``local-operator-ui`` on ``PATH``, which on
the machine this was written on FOUND IT, so a unit test of the terminal rung
launched the operator's real desktop app and left it running. Two tests in
``tests/unit/tui/test_notify.py`` predate the rung and now stub it out
(``_no_desktop_app``), and the e2e click test doubles both lower rungs before
the ladder runs. Do the same rather than trusting that nothing is installed.

**DISCOVERY HAS TO BE LOUD, which is why rung 2 does not use
``spawn_detached``.** ``spawn_detached`` reports only whether a child was
STARTED, and a launcher that is not installed starts fine and exits 1 — so a
candidate that cannot work would look like success, no later candidate would be
tried, and the click would land nowhere. Rung 2 therefore waits (bounded) for
the exit status, which is the signal that distinguishes "launched" from "not
here".

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
import os
import sys

logger = logging.getLogger(__name__)

#: The ``desktop.launch_command`` default: empty means "discover the app".
#:
#: Stated as a module constant rather than inline so the settings registry's
#: restated default can be guarded against THIS value by name
#: (``tests/unit/test_settings_io.py::_consumer_defaults``) — the registry
#: cannot import this module without adding an import edge from the CLI's
#: settings layer into the TUI.
DESKTOP_LAUNCH_COMMAND_DEFAULT = ""

#: Where the session id is substituted in a configured ``desktop.launch_command``.
LAUNCH_SESSION_PLACEHOLDER = "{session}"

#: The desktop app's argv contract for "open this conversation". Shared with the
#: app's own command-line parsing; the two halves are separate repositories, so
#: the spelling is pinned here and asserted on the wire in the e2e suite.
OPEN_SESSION_FLAG = "--open-session"

#: The npm channel's bin name, resolved on PATH. The packaged macOS bundle is
#: not on PATH, so it needs an identifier instead: the ``appId`` the Electron
#: build is configured with.
DESKTOP_BIN_NAME = "local-operator-ui"
DESKTOP_BUNDLE_ID = "com.local-operator"

#: How long rung 2 waits for a launcher's exit status. Generous against every
#: real launcher (both shipped ones exit in well under a second) and small
#: enough that a wedged one cannot eat the click: the module's whole budget is
#: the ~8.5 s the viewer dial may already have spent.
_LAUNCH_PROBE_TIMEOUT_S = 2.0


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

    FOUR rungs, and the ORDER is the operator's stated requirement rather than a
    measurable preference (review round 1, R9): the requested destination is the
    DESKTOP UI, and the terminal is the fallback.

    1. **A running DESKTOP viewer** — including a windowless one, whose
       ``resume_session`` recreates the window and then navigates, which is
       exactly the "app is alive but its window is closed" click.
    2. **An installed-but-not-running desktop app**, launched into the
       conversation.
    3. **A running TUI viewer**, switched in place.
    4. **A terminal**, byte-identical to the behaviour that shipped before
       viewers existed.

    RUNGS 2 AND 3 USED TO BE THE OTHER WAY ROUND, and that is the defect: a TUI
    that happened to be open swallowed every click before discovery ever ran, so
    a user who asked for the app got their terminal instead — decided by nothing
    but incidental focus history. The UI is tried first, and only a UI that is
    unavailable OR fails within its bounded attempt diverts into a terminal.

    The refusal in :data:`DESKTOP_LAUNCH_REFUSED_ENV` takes the app out of the
    LADDER, not just out of the launch: "never launch the desktop" has to mean
    the app is not a destination, or the setting would only skip the launch and
    then have a routing rung pick the same running app up again. So rung 3 is
    narrowed to the TUI surfaces when the refusal is set (review round 2, R13) —
    without which a refused ladder still landed in a running desktop, measured
    as ``open_session -> True`` via the desktop viewer.
    """
    app_available = not _desktop_launch_refused()
    # Local import, like the viewer stack below it: this module is reached from a
    # detached click process where startup cost is the user's latency, and the
    # surface names are the only thing needed from it here. Imported whichever
    # way the refusal goes, because rung 3 needs a name from it in both.
    from local_operator.session.runtime.viewers import DESKTOP_SURFACE, TUI_SURFACE

    if app_available and _route_to_viewer(session_id, surface=DESKTOP_SURFACE):
        return True
    if app_available and _launch_desktop(session_id):
        return True
    # RUNG 3 IS THE ROUTING RUNG, so it is the one that has to honour the
    # refusal: asking it for "anything at all" let ``choose_viewer``'s own
    # desktop preference select a running app the user told us not to use. When
    # the launch is allowed this stays nil, so the fallback rung keeps answering
    # "is anything left?" exactly as it did.
    if _route_to_viewer(session_id, surface=None if app_available else TUI_SURFACE):
        return True
    return _spawn_terminal(session_id)


def _desktop_launch_refused() -> bool:
    """Whether the user (or the suite) forbade the desktop app entirely.

    Read in ONE place so no rung can disagree about it, and so "refused" can be
    answered WITHOUT doing any discovery work — the refusal is checked before the
    scan, not after it. ``_launch_desktop`` keeps its own
    check for the same reason it always had one: it is the rung that could
    actually start the app, and it must not depend on a caller having asked
    first.
    """
    return bool(os.environ.get(DESKTOP_LAUNCH_REFUSED_ENV))


def _configured_launch_command() -> list[str]:
    """``desktop.launch_command`` as argv, or ``[]`` when unset.

    Read through the settings layer at CLICK time. Every rung on this path is
    best-effort, so a settings failure degrades to discovery rather than
    killing the click — and the read is deliberately the same registry the
    settings page writes, so an edit is visible here without a restart.
    """
    try:
        from local_operator.tui.settings import settings_get

        raw = settings_get("desktop.launch_command", "")
    except Exception:  # noqa: BLE001 — discovery is the fallback, not a failure
        logger.debug("could not read desktop.launch_command", exc_info=True)
        return []
    if not isinstance(raw, str) or not raw.strip():
        return []
    import shlex

    try:
        # `posix=False` on Windows: its command lines are not POSIX-quoted, and
        # shlex would strip the backslashes out of every path there.
        parts = shlex.split(raw, posix=sys.platform != "win32")
    except ValueError:
        logger.debug("desktop.launch_command is not a valid command line")
        return []
    return parts


#: A refusal that no rung can override: with this set, a click never launches the
#: desktop app and falls through to the terminal instead.
#:
#: TWO CALLERS, and the second is why it exists. It is a plain user preference —
#: some people want a banner click to open a terminal, and the setting's default
#: is discovery, so without this there is no way to say "never" — but it is
#: ALSO the suite's central gate. Rung 2 looks for `local-operator-ui` on PATH,
#: which on a developer's machine FINDS IT, so a test that drives `open_session`
#: and forgets to double this rung starts the operator's real app and leaves it
#: running. `tests/conftest.py` sets this for every test, exactly as it sets
#: `LOCAL_OPERATOR_NO_NOTIFICATIONS` for the OS-toast path, and a test that
#: deliberately exercises the launch ladder clears it (see
#: `tests/unit/tui/test_resume_click.py`) — the visible, deliberate opt-in.
DESKTOP_LAUNCH_REFUSED_ENV = "LOCAL_OPERATOR_NO_DESKTOP_LAUNCH"


def _launch_desktop(session_id: str) -> bool:
    """Rung 2: launch the desktop app with the session id.

    DISCOVERY ORDER, and each step is cheaper than the one before it: a
    configured command (the user's own answer, used verbatim), then the npm bin
    on PATH (a real existence check via ``which``), then the packaged macOS
    bundle by id. The ``pnpm dev`` case is deliberately absent — a repository
    checkout is not reliably discoverable as an installed app, and inventing a
    third launcher for it would be a second thing to keep in step for a
    developer-only case. It falls through to the terminal, which is exactly
    what it did before.

    Every candidate is TRIED in order and abandoned only on a non-zero exit, so
    an uninstalled bundle costs one failed ``open`` rather than a dead click.

    REFUSED ENTIRELY under :data:`DESKTOP_LAUNCH_REFUSED_ENV`, checked first so
    the refusal costs nothing and cannot be reached by any candidate.
    """
    import os
    import shutil

    if os.environ.get(DESKTOP_LAUNCH_REFUSED_ENV):
        return False

    attempts: list[list[str]] = []
    configured = _configured_launch_command()
    if configured:
        attempts.append(
            [part.replace(LAUNCH_SESSION_PLACEHOLDER, session_id) for part in configured]
        )
    else:
        binary = shutil.which(DESKTOP_BIN_NAME)
        if binary:
            attempts.append([binary, OPEN_SESSION_FLAG, session_id])
        if sys.platform == "darwin":
            # `--args` because `open` forwards nothing to the app otherwise, and
            # the bundle id rather than a path because the user may have moved
            # the app anywhere LaunchServices can find it.
            attempts.append(
                ["open", "-b", DESKTOP_BUNDLE_ID, "--args", OPEN_SESSION_FLAG, session_id]
            )
    if not attempts:
        return False
    env = dict(os.environ)
    for argv in attempts:
        if _launch_once(argv, env):
            logger.debug("click launched the desktop app: %s", argv[0])
            return True
    return False


def _launch_once(argv: list[str], env: dict[str, str]) -> bool:
    """Start ``argv`` detached and report whether it really launched.

    A bounded wait for the EXIT STATUS, which is the only thing that separates
    "the app started" from "this launcher is not installed": ``open -b`` exits
    1 for an unknown bundle id and the npm bin exits 1 for a missing app
    directory, and both of those start a process successfully. A TIMEOUT COUNTS
    AS SUCCESS, because a launcher still alive after two seconds is one that is
    running — never kill it; that would be the click launching the app and then
    immediately shutting it down.

    Detached from this process's session, because this process is a notification
    handler that macOS will reap: a child that shared its process group would
    take the app down with it on some platforms.
    """
    import subprocess

    try:
        process = subprocess.Popen(  # noqa: S603 — argv is constructed here, never from input
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    except (OSError, ValueError):
        logger.debug("launcher %r could not be started", argv[:1], exc_info=True)
        return False
    try:
        return process.wait(timeout=_LAUNCH_PROBE_TIMEOUT_S) == 0
    except subprocess.TimeoutExpired:
        return True


def _route_to_viewer(session_id: str, *, surface: str | None = None) -> bool:
    """Ask an already-running viewer to display the session. True if one did.

    Every failure mode — no viewer, a wedged one, a refused socket, a viewer
    that died between the scan and the dial, **and a viewer that took the
    request but could not display the session** — returns False and falls
    through to the next rung. That last one is why the viewer's ack resolves
    against the boot OUTCOME rather than the dispatch: a "yes" for a session
    that failed to open would suppress the fallback AND cost the user the
    conversation they were reading, leaving them on an error splash with no
    window and no way back.

    ``surface`` narrows the question to one surface (review round 1, R9), which
    is how :func:`open_session` asks "is the desktop app running?" separately
    from "is anything at all running?" without inferring either from the other.

    The dial is bounded end to end (``viewer_client``), so a wedged viewer costs
    a second or two rather than the click.

    The import is local and stays that way: this module is reached from a
    detached click process where startup cost is the user's latency, and
    nothing above needs the viewer stack.
    """
    try:
        from local_operator.session.runtime.viewer_client import route_click

        outcome = route_click(session_id, surface=surface)
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

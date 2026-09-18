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

So the premise is no longer assumed; it is **checked**. The ladder, in the
order the operator asked for — the UI first, a terminal last:

1. **A running DESKTOP viewer** — tell that process to switch and to bring its
   window forward. No new process, no second copy of the session, including the
   macOS case where the app is up with no window: its record says so and
   `needs_switch` honours it.
2. **A desktop app is installed but not running** — launch it WITH the session
   id (`--open-session <id>`), so it opens straight into the conversation
   rather than into the catalogue. This is the rung that makes a `nothing is
   running` banner's click land in the app instead of spawning a terminal, which
   is the surface the user actually wants for a conversation they are being
   told finished.
3. **A running TUI viewer** — switch that window in place. It starts no process
   either, and it is what stops a click from opening a SECOND terminal for a
   session that is one keystroke away in a terminal the user already has open.
4. **A terminal** — the original premise genuinely holds, and the spawn below
   runs the argv it always ran. That path is why this module exists. What it no
   longer does is claim a landing it did not make: it opens a window, or it
   reports failure.

THREE RUNGS BECAME FOUR when the ordering was corrected (review round 1, R9):
asking "is anything running?" before discovery let a TUI that happened to be
open swallow every click, so a user who asked for the app got their terminal
instead. The running TUI became a destination of its own at 3 and the spawn
moved to 4. This list went on saying three for considerably longer than the
code did, which is the `docs/DESKTOP_API.md` staleness (R12) happening inside a
docstring in the same file — so it is stated against the code and not from
memory of it.

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
to the spawn and nothing here raises. The user's recourse is the same on every
branch — `lop --resume <id>` in their own terminal.

**A CLICK THAT CANNOT LAND SAYS SO OUT OF BAND, not only on stderr.** The
receipt the caller prints on the failure branch is right for a hand-run, but a
real click is handled by a DETACHED helper whose three streams are ``/dev/null``
(``spawn_detached`` opens them so; the notifier's ``NSTask`` inherits them), so
that stream has no reader and three reachable failures — ssh, a non-darwin host,
a hand-edited ``desktop.launch_command`` typo — were clicks that did nothing and
said nothing, which is the very defect this ladder exists to remove (UX round 2,
U10). Hence :func:`_notify_click_failed` on that branch.

**AND THE LAST RUNG REPORTS A LANDING, NOT A SPAWN.** It used to end in
``spawn_detached(["lop", "--resume", <id>])``, a process with no terminal
attached: nothing appeared, ``open_session`` still answered True, and the
caller therefore printed nothing — a click indistinguishable from a slow one,
on the one rung defined by having nothing else to try (UX round 1, U5). See
:func:`_spawn_terminal`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    # The spawn package itself is imported INSIDE the functions that need it:
    # this module is reached from a detached click process where startup cost
    # is the user's latency, and nothing here constructs a backend at import
    # time. Only the annotations need the names.
    from local_operator.spawn.types import EnvMap, SpawnBackend

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
    4. **A terminal** — the rung that did ship before viewers existed, and the
       last one: it runs the argv it always ran and reports whether a window
       opened (see :func:`_spawn_terminal`). It is NOT byte-identical to what
       shipped — that rung ended in a detached ``lop --resume`` that opened
       nothing and still answered True (UX round 1, U5).

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
    landed = _spawn_terminal(session_id)
    if not landed:
        # THE CLICK'S OWN REPORT, out of band. Nothing above can land, so the
        # only place left to say so is the channel that reached the user in the
        # first place — see `_notify_click_failed` for why the receipt alone is
        # not enough on a real click.
        _notify_click_failed(session_id)
    return landed


def _notify_click_failed(session_id: str) -> None:
    """Tell the user, out of band, that a click could not take them anywhere.

    THE RECEIPT HAS NO READER ON A CLICK (UX round 2, U10; agent review round 1,
    M4). `cli.resume_click` prints ``could not open a terminal for session <id>
    — run: lop --resume <id>`` to STRDERR, which is correct for the hand-run it
    is reachable by — but the click chain hands the CLI ``/dev/null`` for stdin,
    stdout AND stderr: `spawn_detached` opens the notifier that way and the
    notifier's own ``NSTask``/``sh -c`` inherits it, so the line reaches nobody.
    Driven end to end (``lsof`` on the CLI: ``0r/1w/2w CHR /dev/null``) on all
    three reachable failures — ssh, non-darwin, and a typo'd
    ``desktop.launch_command`` written into ``config.yml`` by hand, whose click
    time WARNING lands on the same stream. From the chair each was a click that
    did nothing and said nothing, which is the symptom this whole ladder was
    raised for.

    BEST-EFFORT BY CONTRACT, exactly like every other call on this path:
    ``detached_notify`` never blocks and never raises, it is a no-op when
    notifications are disabled, and a failure here costs the toast and nothing
    else — the receipt is still printed for whoever is reading a terminal.

    THE COPY IS THE RECEIPT'S, so the two reports of one failure agree. The
    session id is NOT passed as ``session_id``: that argument is what makes a
    macOS toast CLICKABLE, and the click it would post is this same ladder —
    which has just failed. A toast that invites a retry loop is worse than one
    that names the command the user can run themselves.
    """
    try:
        from local_operator.tui.notify import APP_NAME, detached_notify

        detached_notify(
            APP_NAME,
            f"could not open a terminal for session {session_id} — "
            f"run: lop --resume {session_id}",
        )
    except Exception:  # noqa: BLE001 — a toast must never outrank the receipt
        logger.debug("could not post the failed-click toast", exc_info=True)


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

    Read through the SETTINGS PAGE's own reader at CLICK time, so an edit lands
    on the next click without a restart, which is the requirement this read
    exists for.

    THAT READER IS ``settings_io`` AND NOT ``tui.settings.settings_get``, and
    the difference is a shipped defect rather than a style preference (QA round
    2, Q1). ``settings_get`` is the DISPLAY fast path: its ``_load`` returns
    only ``settings_io.display_defaults()`` — the flat-dotted ``display.*``
    flags — so a NESTED key can never be in its cache and the read came back as
    its default every time. Measured on the real ladder, the settings page
    showed the user's launcher while this returned ``""``: discovery then ran a
    different program than the one configured, and against a machine with
    nothing discoverable the click fell all the way to rung 4.

    The registry's ``path`` is the half that was right. Flat-dotted keys are
    reserved for ``display.*`` precisely BECAUSE ``tui.settings`` reads them
    (see ``AGENTS.md``, "Adding a configuration key"), and
    ``desktop.launch_command`` is not a display flag — so the fix is here, not
    in the key: widening the display fast path to serve a nested, non-display
    key would put a click-time setting on the paint-path cache. The read goes to
    the module every writer already funnels through instead
    (``settings_io.write_setting`` serves ``/settings``, ``PATCH /v1/settings``
    and ``lop config edit``), which is what makes the two halves agree by
    construction rather than by coincidence.

    Every rung on this path is best-effort, so a settings failure degrades to
    discovery rather than killing the click.
    """
    try:
        from local_operator import settings_io
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        setting = settings_io.resolve_key("desktop.launch_command")
        if setting is None:  # pragma: no cover - the row is registered
            return []
        raw = settings_io.read_setting(ConfigManager(config_dir()), setting)
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
        # A WARNING, not a debug: the user's own configured launcher is being
        # ignored for every click from now on and is worth finding in a log
        # (UX round 1, U4). The write-time validator in `settings_io` rejects
        # the value when it is typed into `/settings`; this covers the copy
        # that reached `config.yml` by hand, or before that validator shipped.
        logger.warning(
            "desktop.launch_command is not a valid command line; "
            "notification clicks fall through to a terminal"
        )
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

    DISCOVERY ORDER: the npm bin on PATH first (a real existence check via
    ``which``), then the packaged macOS bundle by id. A configured
    ``desktop.launch_command`` REPLACES those two rather than leading them: an
    explicit answer is not a first try to be silently second-guessed, so the
    two orders are alternatives and not one chain. The ``pnpm dev`` case is
    deliberately absent — a repository checkout is not reliably discoverable
    as an installed app, and inventing a third launcher for it would be a
    second thing to keep in step for a developer-only case. It falls through
    to the terminal, which is exactly what it did before.

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
    # Same reason as the terminal rung's env: a GUI the user's click opened is
    # not a command an agent's tool call ran, and everything THIS process later
    # spawns would otherwise inherit that claim for the app's whole life (the
    # app is long-lived: it outlives the session that opened it).
    from local_operator.agent_shell import without_agent_shell_marker

    env = without_agent_shell_marker(os.environ)
    for argv in attempts:
        if _launch_once(argv, env):
            logger.debug("click launched the desktop app: %s", argv[0])
            return True
    if configured:
        # A CONFIGURED LAUNCHER THAT CANNOT RUN IS NOT A QUIET FALLBACK (UX
        # round 1, U4). Discovery failing is ordinary — that is what the rungs
        # below are for. But a user who set `desktop.launch_command` has stated
        # where a click should land, and a typo in it sends every click to a
        # terminal instead, indefinitely, visible nowhere. The write-time
        # validator rejects the typo at the settings page; this is the log
        # line for the value that got in another way, and it names the token
        # that could not be run.
        logger.warning(
            "desktop.launch_command (%s) could not be launched; "
            "notification clicks fall through to a terminal",
            attempts[0][0],
        )
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


def _last_resort_backend(env: EnvMap, detected: SpawnBackend | None) -> SpawnBackend | None:
    """The backend that can open a window WITHOUT being inside one, or None.

    DETECTION IS THE WRONG QUESTION ON THE LAST RUNG, and this function is the
    whole of that argument (UX round 1, U5). Rung 4 runs in a process that by
    construction has no terminal around it: a notification click, handled by a
    detached helper whose environment carries none of the markers
    :func:`~local_operator.spawn.registry.active_backend` keys on. Every other
    backend answers "is a terminal DISCOVERABLE from here?", which on such a
    host is a question with no useful answer.

    On darwin that is not "no terminal can be opened":
    :class:`~local_operator.spawn.apple.TerminalAppBackend` is an ``osascript``
    doing ``tell application "Terminal"``, which LAUNCHES Terminal.app — it
    never needs this process to be inside it, and Terminal.app ships with
    macOS. So it is offered as the guaranteed last candidate, AFTER any
    detected backend so the user's own terminal still wins when it is
    recognised.

    ``None`` on every other platform, where nothing here can promise a visible
    window; rung 4 then reports failure and the caller prints the receipt.

    OVER SSH IT IS ALSO ``None``, and that is honesty rather than a gap.
    ``osascript`` starts and accepts the script, but the ``tell application``
    inside it fails for want of a window server — so the spawn would report a
    landing that never happened, which is precisely the defect this function
    exists to fix. ``spawn.fallback`` draws that line from the same fact
    (``terminals.is_ssh``) and puts it in ITS OWN receipt (``no window server
    over ssh``); the wording is not shared, because this path does not print
    that receipt — ``cli.resume_click`` prints its generic "could not open a
    terminal" for every failing branch, and the rung-4 failure toast repeats
    that sentence verbatim (:func:`_notify_click_failed`, review round 1, N1).
    """
    if sys.platform != "darwin":
        return None
    from local_operator import terminals
    from local_operator.spawn.apple import TerminalAppBackend

    if isinstance(detected, TerminalAppBackend):
        # Already the candidate the registry picked: trying it twice would
        # report the same refusal twice and delay the failure.
        return None
    if terminals.is_ssh(env):
        return None
    return TerminalAppBackend()


def _spawn_terminal(session_id: str) -> bool:
    """Open ``session_id`` in a NEW terminal. True if a WINDOW was opened.

    THE ORIGINAL PATH: reached only when no viewer can take the click, which is
    the condition the module's first docstring assumed always held. It is the
    only route on a machine with no TUI running, so its argv and its cwd are the
    ones that always shipped, and the backend it tries FIRST is still the one
    the registry detects.

    WHAT IT NO LONGER DOES IS CLAIM A LANDING IT DID NOT MAKE (UX round 1, U5).
    It used to fall through to ``spawn_detached(["lop", "--resume", <id>])``:
    a process started DETACHED, never waited on, with DEVNULL for all three
    streams — so no terminal was attached to it, nothing became visible, and
    the True it returned suppressed the receipt the caller prints on failure.
    From the user's side that is a click that did nothing and said nothing, on
    the rung whose entire definition is "there is nothing else".

    So the rung now tries terminal backends ONLY — the detected one, then the
    darwin launcher that needs no detection (:func:`_last_resort_backend`) —
    and returns False when none of them opened a window. False is a real
    answer on this path: ``cli.resume_click`` turns it into the
    ``lop --resume <id>`` receipt, and :func:`_notify_click_failed` raises the
    same sentence as a toast for the click that never sees stderr.
    """
    import shutil

    from local_operator.agent_shell import without_agent_shell_marker
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
    # The click is the USER's gesture and the window it opens is theirs, so the
    # child must not inherit the agent-shell marker: `cli.main` would refuse its
    # `lop --resume` and the terminal would open on a refusal. See
    # `agent_shell.without_agent_shell_marker` (review round 1, F1).
    env = without_agent_shell_marker(env_or_process(None))

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

    candidates: list[SpawnBackend] = []
    if backend is not None:
        candidates.append(backend)
    last_resort = _last_resort_backend(env, backend)
    if last_resort is not None:
        candidates.append(last_resort)

    # A backend is abandoned only where it opened NOTHING — a missing binary, a
    # refused socket, an error from its own spawn — and that is the PREMISE the
    # fall-through rests on, not a claim the interface guarantees. It holds for
    # the AppleScript and `spawn_detached` backends, which is why the darwin last
    # resort can be reached safely; it does NOT hold for `CmuxBackend`, whose
    # `_spawn_surface` answers False both for a failed send AND for a
    # surface-creating placement it could not read an id out of — so on that one
    # path a second candidate can follow a window that already exists (review
    # round 1, M2). The distinction is not lost where it is made:
    # `spawn/cmux.py:176` is the failed CREATE and `:178-179` is the zero-exit
    # create whose stdout carried no id, and `_spawn_surface`'s own `bool`
    # return is what collapses the two by the time this loop sees them. So the
    # refusal stays narrow for a stated reason rather than an unavailable one:
    # acting on it needs a tri-state (or a `placed` flag) out of `CmuxBackend`,
    # i.e. a change to that backend, and this rung's whole value is that a click
    # which placed nothing still lands somewhere (agent review round 2, M1).
    for candidate in candidates:
        try:
            if candidate.spawn(launch, env):
                return True
        except Exception:  # noqa: BLE001 — try the next candidate, then report
            logger.debug("terminal backend refused the launch", exc_info=True)

    logger.debug("no terminal backend could open a window for %s", session_id)
    return False


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return 2
    return 0 if open_session(args[0]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

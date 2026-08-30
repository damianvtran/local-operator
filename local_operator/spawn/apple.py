"""Terminal.app and iTerm2: open the fork by AppleScript.

Both macOS applications are driven the same way — ``osascript -`` with the
script on stdin — so they share a module. They stay TWO backends because their
scripts differ and because the registry must be able to detect them
independently.

WHY THE SCRIPT COMES IN ON STDIN
--------------------------------
``osascript -e`` would require splicing the launch line into AppleScript source
as a quoted string, and the launch line contains a filesystem path this app does
not control. Reading the script from stdin and passing the command as an ``argv``
argument to its ``on run`` handler keeps user-influenced text out of the source
entirely — the same technique, for the same reason, as ``clipboard._read_macos``.

WHY THE LAUNCH LINE IS STILL A SHELL STRING
-------------------------------------------
Both applications' automation vocabulary takes a COMMAND to type into a new
window (``do script`` / ``write text``); neither accepts an argv vector. So the
argv is joined with ``shlex.join``, which is what makes a launcher path with a
space in it survive being re-tokenised by the shell those applications spawn.
"""

from __future__ import annotations

import shlex

from local_operator import terminals
from local_operator.spawn.types import EnvMap, ForkLaunch

#: Terminal.app: ``do script`` with no ``in`` target opens a NEW window and runs
#: the command in it. ``cd`` is part of the command because Terminal.app has no
#: working-directory parameter on this verb.
#:
#: NO ``activate``. It is an extra line beyond what ``do script`` needs to open
#: the window, and all it adds is raising Terminal.app over whatever the user is
#: typing in — a fork is something they asked for while working somewhere else.
#: iTerm2's script below does not activate either, so the two behave alike. A
#: new OS window may still take focus on its own; that is the platform, and it
#: is one more reason cmux placement is the better experience where cmux exists.
TERMINAL_SCRIPT = """on run argv
\tset launchCommand to item 1 of argv
\ttell application "Terminal"
\t\tdo script launchCommand
\tend tell
end run
"""

#: iTerm2: create a window from the default profile, then type the command into
#: its session. Two steps because ``create window with default profile`` does
#: not take a command.
ITERM_SCRIPT = """on run argv
\tset launchCommand to item 1 of argv
\ttell application "iTerm2"
\t\tset newWindow to (create window with default profile)
\t\ttell current session of newWindow
\t\t\twrite text launchCommand
\t\tend tell
\tend tell
end run
"""


def launch_command(launch: ForkLaunch) -> str:
    """The shell line these applications type into a new window.

    ``cd <cwd> && <argv>``: neither verb used here takes a working directory, so
    it is part of the command. The cwd is quoted by ``shlex.quote`` for the same
    reason the argv is joined by ``shlex.join`` — a project path with a space in
    it is ordinary, and an unquoted one would ``cd`` somewhere else entirely.
    """
    return f"cd {shlex.quote(launch.cwd)} && {shlex.join(launch.argv)}"


def osascript_argv(launch: ForkLaunch) -> list[str]:
    """``osascript - <command>``; the script itself arrives on stdin."""
    return ["osascript", "-", launch_command(launch)]


class _AppleScriptBackend:
    """Shared spawn half; the subclasses differ only in script and detection."""

    name = "applescript"
    opened_place = "a new window"
    script = ""

    def spawn(self, launch: ForkLaunch, env: EnvMap) -> bool:
        # This is the one backend that does NOT go through ``proc.spawn_detached``,
        # because the script has to reach the child on stdin and that helper
        # deliberately points stdin at DEVNULL. Every other property it
        # guarantees is reproduced here: ``start_new_session``, stdout/stderr to
        # DEVNULL, and no wait — ``communicate`` is never called, so a hung
        # osascript cannot hold this process.
        import subprocess

        try:
            process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                osascript_argv(launch),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            return False
        try:
            if process.stdin is not None:
                process.stdin.write(self.script.encode("utf-8"))
                process.stdin.close()
        except Exception:
            # The child started but would not take the script: it will exit on
            # its own (osascript reading EOF from a closed pipe), and the caller
            # falls through to the receipt.
            return False
        return True


class TerminalAppBackend(_AppleScriptBackend):
    """Opens a fork in a new macOS Terminal.app window."""

    name = "apple-terminal"
    opened_place = "a new Terminal window"
    script = TERMINAL_SCRIPT

    def detect(self, env: EnvMap) -> bool:
        return terminals.is_apple_terminal(env)


class ITerm2Backend(_AppleScriptBackend):
    """Opens a fork in a new iTerm2 window."""

    name = "iterm2"
    opened_place = "a new iTerm2 window"
    script = ITERM_SCRIPT

    def detect(self, env: EnvMap) -> bool:
        return terminals.is_iterm(env)

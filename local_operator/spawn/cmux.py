"""cmux: open the fork as its own workspace, or as a surface in this one.

Two placements, chosen by ``fork.cmux_placement``:

- **workspace** (default) — ``cmux new-workspace``, a sidebar row of its own.
  One call carries the cwd AND the launch line (``--command`` sends text+Enter
  after creation), which is why it is the default: there is no window in which
  the workspace exists but is not yet running the session.
- **surface** — ``cmux new-surface``, a tab in the current workspace. Needs TWO
  calls, because ``new-surface`` takes ``--working-directory`` but has no
  ``--command`` flag (verified against the installed CLI's ``--help``): the
  launch line goes separately through ``cmux send``. That gap is the reason this
  is not the default.

FOCUS, EXPLICITLY FALSE, ON EVERY CALL
--------------------------------------
cmux's socket gate only permits focus, window raise and workspace switching when
a command carries an explicitly truthy ``focus``. A fork is something the user
asked for while working somewhere else, so it must appear WITHOUT stealing the
window they are typing in. Both calls therefore pass ``--focus false`` rather
than relying on cmux's default, so the intent is visible at the call site and a
future default change cannot silently start raising windows.

NO SECOND CMUX CLIENT
---------------------
The binary resolution, the RPC helper and the surface target come from
``multiplexer.cmux``'s public wrappers. That module documents why PATH beats
``CMUX_BUNDLED_CLI_PATH`` and why a resolvable BINARY rather than the ``CMUX_*``
markers is the real gate (every one of those variables survives an ssh hop into
a host with no cmux).
"""

from __future__ import annotations

import re
import shlex

from local_operator.multiplexer.cmux import cmux_binary, surface_target
from local_operator.proc import spawn_detached
from local_operator.spawn.types import CALL_TIMEOUT_S, EnvMap, ForkLaunch

#: ``fork.cmux_placement`` values this backend understands. Anything else falls
#: back to the workspace form rather than failing the fork: an unknown placement
#: is a config typo, and refusing to open a window over one would be a worse
#: answer than opening the default one.
PLACEMENT_WORKSPACE = "workspace"
PLACEMENT_SURFACE = "surface"

#: How ``cmux new-surface`` reports what it made. Measured against the installed
#: CLI, which answers on ONE line:
#:
#:     OK surface:142 pane:35 workspace:35
#:
#: The whole line is not a surface id, so the id has to be picked out of it —
#: sending the raw line to ``cmux send --surface`` targets nothing and the new
#: tab sits empty, which presents as a fork that opened and then hung.
_SURFACE_ID_RE = re.compile(r"\bsurface:\d+\b")


def workspace_argv(binary: str, launch: ForkLaunch) -> list[str]:
    """The ``cmux new-workspace`` argv for ``launch``. Pure, so it is testable.

    ``--command`` receives a SHELL STRING (cmux types it and presses Enter), so
    the argv is joined with ``shlex.join`` and not ``" ".join``: a launcher path
    containing a space (``/Applications/My Tools/lop``) would otherwise arrive as
    two arguments and start nothing.
    """
    return [
        binary,
        "new-workspace",
        "--name",
        launch.title,
        "--cwd",
        launch.cwd,
        "--command",
        shlex.join(launch.argv),
        "--focus",
        "false",
    ]


def surface_argv(binary: str, launch: ForkLaunch) -> list[str]:
    """The ``cmux new-surface`` argv — creation only; the command follows."""
    return [
        binary,
        "new-surface",
        "--type",
        "terminal",
        "--working-directory",
        launch.cwd,
        "--focus",
        "false",
    ]


def send_argv(binary: str, surface_id: str, launch: ForkLaunch) -> list[str]:
    """The ``cmux send`` argv that starts the session in a created surface.

    The trailing ``\\n`` is what presses Enter; without it the launch line sits
    typed-but-unrun in the new tab, which looks exactly like a hung fork.
    """
    return [
        binary,
        "send",
        "--surface",
        surface_id,
        f"{shlex.join(launch.argv)}\n",
    ]


class CmuxBackend:
    """Opens a fork in a new cmux workspace (or surface in this workspace)."""

    name = "cmux"

    def __init__(self, placement: str = PLACEMENT_WORKSPACE) -> None:
        self.placement = placement

    @property
    def opened_place(self) -> str:
        """What this placement actually creates, in the user's words.

        A workspace is a SIDEBAR ROW in the window the user already has, and a
        surface is a tab in the workspace they are already in — neither is a
        window, and both are deliberately unfocused, so the receipt is the only
        thing that tells the user where to look.
        """
        if self.placement == PLACEMENT_SURFACE:
            return "a new surface in this workspace"
        return "a new cmux workspace"

    def detect(self, env: EnvMap) -> bool:
        """True when this process is a cmux surface AND a cmux CLI exists.

        The binary is the gate, not the markers alone — see the module
        docstring. Detected FIRST in the registry because cmux embeds ghostty
        and exports ``GHOSTTY_RESOURCES_DIR``, so a ghostty-first order would
        open a stray OS window out of a cmux surface.
        """
        if surface_target(env) is None:
            return False
        return cmux_binary() is not None

    def spawn(self, launch: ForkLaunch, env: EnvMap) -> bool:
        binary = cmux_binary()
        if binary is None:
            return False
        if self.placement == PLACEMENT_SURFACE:
            return self._spawn_surface(binary, launch)
        return spawn_detached(workspace_argv(binary, launch))

    def _spawn_surface(self, binary: str, launch: ForkLaunch) -> bool:
        """Create a surface, then send the launch line into it.

        The creating call is run and WAITED FOR, unlike every other spawn in
        this package, because the second call needs the id of the surface the
        first one made and there is nothing else to learn it from. Bounded by
        CALL_TIMEOUT_S, and the whole fork runs on a worker thread, so the TUI
        is never blocked on it.
        """
        import subprocess

        try:
            completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
                surface_argv(binary, launch),
                capture_output=True,
                text=True,
                timeout=CALL_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if completed.returncode != 0:
            return False
        match = _SURFACE_ID_RE.search(completed.stdout or "")
        if match is None:
            return False
        return spawn_detached(send_argv(binary, match.group(0), launch))

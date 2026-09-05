"""ghostty: open the fork in a new ghostty window.

THE macOS FORM IS NOT THE LINUX FORM, AND THE DIFFERENCE IS NOT COSMETIC.
Ghostty's own ``--help`` states it outright: "On macOS, launching the terminal
emulator from the CLI is not supported and only actions are supported. Use
``open -na Ghostty.app`` instead." A coder who reaches for ``ghostty -e`` on a
Mac gets a confusing partial failure rather than a clean error, so the platform
branch is explicit here and is the whole reason this module has two argv
builders.
"""

from __future__ import annotations

import shlex
import shutil
import sys

from local_operator import terminals
from local_operator.proc import spawn_detached
from local_operator.spawn.types import EnvMap, ForkLaunch

#: The macOS bundle name. ``open -na`` starts a NEW instance ("n") of the
#: application ("a") rather than handing the URL to a running one, which is what
#: makes this open a new window instead of focusing the existing session.
GHOSTTY_APP = "Ghostty.app"


def macos_argv(launch: ForkLaunch) -> list[str]:
    """``open -na Ghostty.app --args …`` — the only supported macOS launch."""
    return [
        "open",
        "-na",
        GHOSTTY_APP,
        "--args",
        f"--working-directory={launch.cwd}",
        # The CLI shortcut also activates macOS command forwarding: through
        # `open --args` Ghostty 1.3.1 can execute it twice (#572). The config
        # key applies only to the first surface. It takes shell text, so quote
        # every argument rather than letting spaces or metacharacters execute.
        f"--initial-command={shlex.join(launch.argv)}",
    ]


def linux_argv(binary: str, launch: ForkLaunch) -> list[str]:
    """``ghostty --working-directory=… -e …`` — the CLI form, Linux only."""
    return [binary, f"--working-directory={launch.cwd}", "-e", *launch.argv]


class GhosttyBackend:
    """Opens a fork in a new ghostty window."""

    name = "ghostty"
    opened_place = "a new Ghostty window"

    def detect(self, env: EnvMap) -> bool:
        """True inside ghostty — but the registry asks cmux FIRST.

        cmux embeds ghostty and exports its markers, so this predicate is true
        inside a cmux surface too. Ordering is what disambiguates them; see
        ``spawn/registry.py``.
        """
        return terminals.is_ghostty(env)

    def spawn(self, launch: ForkLaunch, env: EnvMap) -> bool:
        if sys.platform == "darwin":
            # `open` is in the base system; no resolution needed, and its
            # absence would mean a far more broken machine than this can help.
            return spawn_detached(macos_argv(launch))
        binary = shutil.which("ghostty")
        if binary is None:
            # Markers present but no binary: an ssh hop out of a ghostty window,
            # or a container that inherited the environment. Not this backend's
            # to open, and returning False lets the registry's fallback receipt
            # give the user something that works.
            return False
        return spawn_detached(linux_argv(binary, launch))

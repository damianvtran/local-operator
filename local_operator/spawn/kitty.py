"""kitty: open the fork in a new kitty OS window.

Two forms, and the SAFER one is the default rather than the richer one:

- ``kitty @ launch --type=os-window`` reuses the running kitty and puts the fork
  in a new window of it. It requires ``allow_remote_control`` to be enabled,
  which is off by default in kitty, so it is attempted and checked rather than
  assumed.
- ``kitty --directory <cwd> <argv>`` starts a new kitty instance. It always
  works, which is why it is the fallback.
"""

from __future__ import annotations

import shutil
import subprocess

from local_operator import terminals
from local_operator.proc import spawn_detached
from local_operator.spawn.types import CALL_TIMEOUT_S, EnvMap, ForkLaunch


def remote_argv(binary: str, launch: ForkLaunch) -> list[str]:
    """``kitty @ launch --type=os-window --cwd <cwd> -- <argv>``."""
    return [
        binary,
        "@",
        "launch",
        "--type=os-window",
        "--cwd",
        launch.cwd,
        "--",
        *launch.argv,
    ]


def direct_argv(binary: str, launch: ForkLaunch) -> list[str]:
    """``kitty --directory <cwd> <argv>`` — a fresh instance, always available."""
    return [binary, "--directory", launch.cwd, *launch.argv]


class KittyBackend:
    """Opens a fork in a new kitty OS window."""

    name = "kitty"

    def detect(self, env: EnvMap) -> bool:
        return terminals.is_kitty(env)

    def spawn(self, launch: ForkLaunch, env: EnvMap) -> bool:
        binary = shutil.which("kitty")
        if binary is None:
            return False
        # Checked rather than fired-and-forgotten: remote control is OFF by
        # default in kitty, so this call's failure is the ordinary case and is
        # the signal to start a fresh instance instead.
        try:
            completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
                remote_argv(binary, launch),
                capture_output=True,
                text=True,
                timeout=CALL_TIMEOUT_S,
            )
            if completed.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
        return spawn_detached(direct_argv(binary, launch))

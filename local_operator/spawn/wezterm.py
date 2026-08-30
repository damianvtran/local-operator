"""WezTerm: open the fork in a new WezTerm window.

Two forms, tried in order, because they need different things to be true:

- ``wezterm cli spawn --new-window`` talks to a running MUX SERVER. It is the
  right call inside an existing WezTerm session (the new window joins the same
  server) and it fails when no server is reachable.
- ``wezterm start`` launches a fresh WezTerm. It always works but starts a
  separate instance, which is why it is the fallback rather than the first try.
"""

from __future__ import annotations

import shutil
import subprocess

from local_operator import terminals
from local_operator.proc import spawn_detached
from local_operator.spawn.types import CALL_TIMEOUT_S, EnvMap, ForkLaunch


def cli_spawn_argv(binary: str, launch: ForkLaunch) -> list[str]:
    """``wezterm cli spawn --new-window --cwd <cwd> -- <argv>``."""
    return [binary, "cli", "spawn", "--new-window", "--cwd", launch.cwd, "--", *launch.argv]


def start_argv(binary: str, launch: ForkLaunch) -> list[str]:
    """``wezterm start --cwd <cwd> -- <argv>`` — no mux server needed."""
    return [binary, "start", "--cwd", launch.cwd, "--", *launch.argv]


class WezTermBackend:
    """Opens a fork in a new WezTerm window."""

    name = "wezterm"
    opened_place = "a new WezTerm window"

    def detect(self, env: EnvMap) -> bool:
        return terminals.is_wezterm(env)

    def spawn(self, launch: ForkLaunch, env: EnvMap) -> bool:
        binary = shutil.which("wezterm")
        if binary is None:
            return False
        # The mux form is RUN AND CHECKED rather than fired and forgotten,
        # because its failure is the signal to try the other form. Bounded by
        # the package's own CALL_TIMEOUT_S — a wezterm CLI waiting on an
        # unreachable mux server must not become the fork's latency.
        try:
            completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
                cli_spawn_argv(binary, launch),
                capture_output=True,
                text=True,
                timeout=CALL_TIMEOUT_S,
            )
            if completed.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
        return spawn_detached(start_argv(binary, launch))

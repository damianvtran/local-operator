"""Bring this process's terminal window forward, at the measured ceiling.

Called by a viewer answering a notification click — the user performed a
deliberate gesture whose entire meaning is "take me there", which is the one
circumstance in which taking focus is solicited rather than rude. Nothing else
in this codebase may call it; see ``tui/resume_click.py`` for why that rule is
structural rather than a flag.

## What is reachable, and what is a trap

Measured on this host (macOS 25.6.0, Ghostty 1.3.1) before any of this was
written:

| probe | result | time |
|---|---|---|
| ``open -a Ghostty`` | rc=0 | **0.11 s** |
| ``osascript ... to activate`` | rc=0 | 0.53 s |
| ``osascript ... get frontmost`` | ``false`` | 0.07 s |
| ``osascript ... count windows`` | **``0`` — while 2 windows existed** | 0.20 s |
| ``osascript ... get properties of front window`` | **error** | 0.18 s |

**NEVER TRAVERSE THE EMULATOR'S APPLESCRIPT OBJECT MODEL.** Not with a guard,
not with a timeout. Two independent measurements of Ghostty 1.3.1 disagree
about *how* it fails — one observed ``count windows`` hanging until killed at
30 s, the other a fast, confidently wrong ``0`` while System Events reported
two real windows — and the disagreement does not matter, because both are
disqualifying. A wrong answer is worse than a hang: a timeout at least fails
loudly, whereas ``0`` leads code to conclude there is no window and skip the
activation that would have worked.

So window *selection* is not available from the OS side and is not attempted
here. What is available is application-level activation, which goes through
AppKit, touches no Apple Event object graph, and is five times faster.

## The honest ceiling

With one terminal window this is exact: the viewer switched to the right
session and its application comes forward. With **two** terminal windows the
session switch is still exact — it was addressed to a specific process — but
``open -a`` fronts the *application*, and macOS raises whichever window was
frontmost most recently, which may not be the one that switched. The user then
sees the right session one Cmd-\\` away.

That is a real limitation, and it is still strictly better than the behaviour
it replaces: a brand-new window running a second process for a session already
on screen. Window-exact focus is reachable through the accessibility API
(``AXRaise`` works on these windows and the TUI already writes a session-named
title), but it requires a permission grant with a system prompt, so it is
deliberately left for a later, opt-in change rather than imposed on everyone
for a chrome feature.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from local_operator import terminals

logger = logging.getLogger(__name__)

#: Every OS call is bounded. `timeout(1)` is not on this host's PATH, so the
#: bound is Python's — ``subprocess.run(..., timeout=)``. Generous against the
#: 0.11 s measurement and still far below the point a user would call the
#: click broken.
_ACTIVATION_TIMEOUT_S = 5.0


def _bundle_for(env: terminals.EnvMap | None = None) -> str | None:
    """The macOS application bundle to activate, from environment markers only.

    Markers rather than a query, for the reason ``terminals`` states: a
    capability query needs stdin, and stdin belongs to Textual's input loop
    while the app is running.

    Returns ``None`` for a terminal with no bundle to raise (ssh, an unknown
    emulator), which is an ordinary answer — the switch has already happened
    and the user finds their own window.
    """
    if terminals.is_ssh(env):
        # A window on the far end of an ssh hop is not ours to raise, and the
        # markers are inherited across the hop, so this test must come first.
        return None
    if terminals.is_ghostty(env):
        return "Ghostty"
    if terminals.is_iterm(env):
        return "iTerm"
    if terminals.is_apple_terminal(env):
        return "Terminal"
    if terminals.is_wezterm(env):
        return "WezTerm"
    if terminals.is_kitty(env):
        return "kitty"
    return None


def activate_window(env: terminals.EnvMap | None = None) -> bool:
    """Bring this process's terminal application forward. True if a call ran.

    Best-effort by contract and non-raising: this is chrome on a notification
    click, and a window that fails to raise must never propagate an error into
    the caller — still less into the event loop of the TUI hosting it. Callers
    run it off the loop; see ``tui/app.py``'s viewer host.
    """
    if sys.platform != "darwin":
        # Linux/BSD window activation is a window-manager question with no
        # portable answer (wmctrl, xdotool, a Wayland compositor protocol, or
        # nothing at all). Reporting False here means the session switch still
        # happened and the user alt-tabs, which is honest; guessing at a WM
        # would be a second untested surface for no measured benefit.
        return False
    bundle = _bundle_for(env)
    if bundle is None:
        return False
    opener = shutil.which("open")
    if opener is None:
        # `open` is in the base system; its absence means a far more broken
        # machine than a notification click can help with.
        return False
    try:
        # `open -a <bundle>` WITHOUT `-n`. The `-n` in `spawn/ghostty.py` is
        # what makes that path open a NEW instance — precisely the behaviour
        # this module exists to avoid. Bare `-a` activates the running one.
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [opener, "-a", bundle],
            capture_output=True,
            timeout=_ACTIVATION_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        logger.debug("window activation did not complete", exc_info=True)
        return False
    return result.returncode == 0

"""One answer to "should this surface be animating right now?".

Four surfaces in this app run repaint timers that exist only to move something
— the working line's shimmer and spinner (30 fps), the subagent page's spinner,
the subagent panel's spinner and the status band's spinner (12.5 fps each), and
the welcome splash's mark pulse (12.5 fps). None of them carries information a
user can only get from the motion; every one of them is the app saying "this is
alive".

A terminal the user is not looking at gets nothing from any of that, and it is
not free. Measured on a single idle session: shimmer on costs 7.89% of a core
and 17.3 KB/s of terminal output, against 2.27% and 1.0 KB/s with animation off
— the animation is 92% of what an idle session writes. That output is not
cheap to consume either: the steady-state stream is 48% escape sequences, and
a JS terminal (cmux/Electron) held 51-60% of a core parsing it where a native
GPU one (Ghostty) held 0.6%. Multiply by the eighteen sessions this developer
keeps open and the idle cost of windows nobody is looking at is the single
largest thing lop asks of the machine.

So animation is gated on TERMINAL FOCUS, and this module is the one place that
knows the answer. It is a module-level flag rather than an app attribute
because the widgets that read it are built and mounted at different times and
several of them (``StatusLine``) are not ``Widget`` subclasses at all; a
surface created while the terminal is blurred has to come up throttled without
having to find the app first.

Two invariants this must never violate, both of them load-bearing:

* **A session that never learns about focus must never be throttled.** Some
  hosts and pty setups deliver no focus events at all. The flag therefore
  starts ``True`` and only an explicit blur can clear it, so "no information"
  and "focused" are the same state. The app additionally re-derives it from
  Textual's own ``app_focus`` reactive, which Textual itself sets back to
  ``True`` on any keypress or mouse click — so even a spurious blur with no
  matching focus event heals on the user's next keystroke.
* **Only the RATE of animation may drop; content may not.** Nothing here is
  allowed to suppress a repaint that carries new information. Every throttled
  timer keeps running at a reduced interval, and every surface repaints itself
  immediately when focus returns, so a refocused terminal shows current state
  rather than the frame it was paused on.
"""

from __future__ import annotations

#: Interval for a spinner on a blurred terminal. The app's animated cadence is
#: 0.08 s (:data:`~local_operator.tui.terminal_title.SPINNER_INTERVAL_S`); this
#: is 12.5x cheaper and still advances, so a glance at an unfocused window
#: still says "running" rather than "hung". It is not zero deliberately: a
#: stopped spinner and a finished job look identical, and this app uses motion
#: as its word for alive.
BLURRED_SPINNER_INTERVAL_S = 1.0

#: Starts focused. See the module docstring — "we were never told" must read as
#: focused, never as blurred, or a session on a host that sends no focus events
#: would animate at 1 fps forever.
_focused = True


def animation_focused() -> bool:
    """Whether the terminal is believed to have OS focus."""
    return _focused


def set_animation_focused(focused: bool) -> bool:
    """Record terminal focus. Returns whether this CHANGED the answer.

    The return value is what lets callers skip the fan-out: Textual re-asserts
    focus on every keypress, so an unconditional resync would stop and restart
    four timers on each character the user types.
    """
    global _focused
    if _focused == focused:
        return False
    _focused = focused
    return True


def reset_animation_focus() -> None:
    """Restore the default (focused). For tests, which share a process."""
    global _focused
    _focused = True


def motion_enabled() -> bool:
    """Whether a surface should animate at its full, designed cadence.

    Both gates in one call, because "hold still" is one decision and a surface
    asking only one of the two questions is a bug: the ``display.shimmer``
    setting / ``LOCAL_OPERATOR_NO_SHIMMER`` kill switch decides whether this
    app animates AT ALL (CI and the SVG snapshot harness turn it off so every
    frame is reproducible), and focus decides whether it is worth doing now.
    """
    from local_operator.tui.shimmer import shimmer_enabled

    return shimmer_enabled() and _focused

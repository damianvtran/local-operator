"""Terminal mode negotiation we opt OUT of, before Textual opts us in.

THE SYMPTOM. After any pane or window resize, every mouse position in the app
collapses toward the top-left corner: a click lands cells away from the glyph
under the pointer, and drags select the wrong region. It never recovers within
the session, because the state that causes it is one-way.

THE DIVISOR — AND THE ACTUAL BUG. Textual's parser divides incoming mouse
coordinates by the cell pixel size whenever ``mouse_pixels`` is set
(``_xterm_parser.py:95-102``): it computes ``pixel_width / width`` and
``pixel_height / height`` and scales x and y by those ratios. That arithmetic is
correct only if the coordinates on the wire are PIXELS. A Herdr-class terminal
keeps forwarding CELL coordinates even once mode 1016 has been requested, so a
position at cell (40, 44) arrives as (40, 44) and is divided by a measured
(8.00, 16.00) into (5, 2) — the collapse. THIS mismatch is the defect; the mode
negotiation below is only how we walk into it.

THE REPORTS ARE SOLICITED — TEXTUAL ASKS FOR THEM ITSELF. A clean Herdr VT is
spec-compliant: it answers ``CSI ?2048$p`` with ``;2`` (supported, NOT enabled)
and sends NOTHING on a real geometry change. Reports begin only once an app sets
mode 2048 — and on an unpatched boot the app that sets it is Textual, writing
``?2048h`` at byte ~123 of its own output in reply to that ``;2``. Measured on
herdr 0.9.0.

MODE 2048 IS STICKY PER-VT, WHICH IS WHY THE RESET IS AT THE TERMINAL. The mode
is terminal state, not process state: it outlives the process that set it and is
shared by every process on that tty. A VT dirtied by an earlier app — a previous
Textual run that exited without resetting, say — therefore feeds reports to a
process that never asked for them. That is why the reset is written to the
TERMINAL at boot rather than negotiated per session: we cannot assume we found
the VT clean, and nothing in Textual's own negotiation would clear inherited
state.

THE UNGATED ONE-WAY LATCH. ``mouse_pixels`` is set to True by the in-band
resize report handler at ``_xterm_parser.py:271-283``, on the mere ARRIVAL of a
``CSI 48;rows;cols;pxH;pxW t`` report, with nothing checked first and nothing
that ever sets it back. It does not care WHY the report came — solicited by
Textual moments earlier, or inherited from a sticky mode this process never
touched. That is why the environment guard alone is not enough: it can stop us
asking, but a report that arrives anyway still latches the divisor.

THE NEGOTIATION GATE. The other direction is gated. ``TEXTUAL_SMOOTH_SCROLL=0``
makes ``constants.SMOOTH_SCROLL`` False, and the mode-report branch at
``_xterm_parser.py:321`` then refuses to emit an ``InBandWindowResize`` token at
all, so Textual never concludes the mode is available.

THE RE-ENABLE BRANCH. That gate is why the mode reset alone is not enough
either. Textual's driver answers a ``;2`` reply — supported but reset, exactly
what a bare ``CSI ?2048l`` produces — by turning the mode back on: the
supported-but-not-enabled branch at ``linux_driver.py:470-483`` calls
``_enable_in_band_window_resize()`` (``:480``) and ``_enable_mouse_pixels()``
(``:482``), putting ``?2048h`` and ``?1016h`` on the wire. This is exactly what
an unpatched boot was measured doing — ``?2048h`` at byte ~123, ``?1016h`` at
~131, reports immediately after — so without the guard our reset would be undone
within milliseconds of being written, by us.

SO BOTH MECHANISMS ARE LOAD-BEARING AND NEITHER IS REDUNDANT. The reset clears
the mode AT THE TERMINAL, including a sticky one we inherited; the environment
guard stops TEXTUAL from turning it straight back on after seeing the reset.
Remove either one and the divisor latches again — the first on state left set in
the VT by an earlier app, the second on the driver's reply to its own query.

ORDERING. :func:`reset_in_band_resize` must reach the wire before the driver
issues its ``CSI ?2048$p`` query at ``linux_driver.py:299``, and
:func:`guard_pixel_mouse_latch` must run before ``textual.constants`` is
imported, because ``SMOOTH_SCROLL`` is a ``Final`` read once at import time.

RESIZE STAYS LIVE. Nothing here costs us resize handling. With the negotiation
suppressed the driver's ``_in_band_window_resize`` stays False, which is the
condition its SIGWINCH handler at ``linux_driver.py:246-250`` tests before
sending a size event — so the out-of-band signal path remains the one in use,
exactly as it is on any terminal where mode 2048 is not enabled.

THE ACCEPTED COST. ``App.supports_smooth_scrolling`` (``app.py:865``) is only
ever set True from an in-band resize message, so it stays False. Scrollbar drags
therefore take the animated path at ``scrollbar.py:395`` instead of the
immediate one. A slightly animated scrollbar drag is worth a mouse that points
where the user is pointing.

Deliberately stdlib-only, importing nothing from this package and — critically —
nothing from ``textual``: the same leaf discipline as ``terminals.py``, and here
it is also a correctness requirement, since importing this module must not be
what pulls ``textual.constants`` in ahead of the guard.

All references pinned to textual 8.2.8.
"""

from __future__ import annotations

import os
import sys
from typing import MutableMapping, TextIO

#: ``CSI ? 2048 l`` — reset in-band window resize notifications. Addressed at the
#: TERMINAL rather than at Textual's view of the mode because mode 2048 is
#: STICKY per-VT: a previous app can have left it set, and no amount of
#: not-negotiating on our side clears state we inherited.
DISABLE_IN_BAND_RESIZE = "\x1b[?2048l"

#: Environment kill switch, mirroring ``LOCAL_OPERATOR_NO_TERMINAL_TITLE``
#: (``terminal_title.py:54``). Wanted by anything capturing raw terminal output
#: where an unexpected escape is noise.
#:
#: It suppresses ONLY the reset, not the guard, so it is not the switch for a
#: user on a terminal that genuinely implements pixel coordinates: with the
#: guard still closing the negotiation, ``?1016h`` is never sent and pixel
#: coordinates never arrive. That user wants ``TEXTUAL_SMOOTH_SCROLL=1``, which
#: :func:`guard_pixel_mouse_latch` defers to as an explicit override; the driver
#: then re-enables the mode on the ``;2`` reply and the divisor is correct.
_ENV_DISABLE = "LOCAL_OPERATOR_NO_MODE_RESET"

#: Textual reads this once, as a ``Final``, at ``constants.py`` import time.
_SMOOTH_SCROLL_ENV = "TEXTUAL_SMOOTH_SCROLL"


def guard_pixel_mouse_latch(env: MutableMapping[str, str] | None = None) -> bool:
    """Ask Textual not to negotiate in-band resize; True when we set it.

    Returns False when the variable is already present, which is a deliberate
    user override we leave alone — someone who set ``TEXTUAL_SMOOTH_SCROLL``
    by hand has said what they want on their terminal.

    Must be called before ``textual.constants`` is imported to have any effect.
    """
    if env is None:
        env = os.environ
    if _SMOOTH_SCROLL_ENV in env:
        return False
    env.setdefault(_SMOOTH_SCROLL_ENV, "0")
    return True


def reset_in_band_resize(stream: TextIO | None = None) -> bool:
    """Write ``CSI ?2048l`` to the terminal; True when the bytes were written.

    Defaults to ``sys.__stderr__``, which is the same handle Textual's driver
    writes its own escapes to (``linux_driver.py:58``), so the reset is
    serialised onto the wire ahead of the driver's query rather than racing a
    different file object.

    Returns False without writing when there is no terminal to write to (no
    stream, no ``isatty``, not a tty) or when the kill switch is set. Write
    failures are swallowed: a redirected or already-closed stderr must never
    be what stops the app from booting.
    """
    if stream is None:
        stream = sys.__stderr__
    if stream is None or not hasattr(stream, "isatty"):
        return False
    if os.environ.get(_ENV_DISABLE):
        return False
    try:
        if not stream.isatty():
            return False
        stream.write(DISABLE_IN_BAND_RESIZE)
        stream.flush()
    except (OSError, ValueError):
        # Redirected, detached or closed stderr. Nothing to reset, and a
        # cosmetic escape is never worth failing a boot over.
        return False
    return True

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

AN INHERITED VALUE COUNTS ONLY IF TEXTUAL READS IT. ``constants.SMOOTH_SCROLL``
is ``_get_environ_int("TEXTUAL_SMOOTH_SCROLL", 1) == 1`` (``constants.py:30-52``
and ``:168``), and that helper returns its DEFAULT for anything ``int()``
rejects. An unparseable inherited value — empty, or ``true`` — therefore means
"smooth scrolling ON" to Textual, so deferring merely because the name is set
would hand the latch back to a value Textual never acted on. The guard parses
the value the same way Textual does and treats what Textual ignores as absent.

THE RE-ENABLE BRANCH. That gate is why the mode reset alone is not enough
either. Textual's driver answers a ``;2`` reply — supported but reset, exactly
what a bare ``CSI ?2048l`` produces — by turning the mode back on: the
supported-but-not-enabled branch at ``linux_driver.py:470-483`` calls
``_enable_in_band_window_resize()`` (``:480``) and ``_enable_mouse_pixels()``
(``:482``), putting ``?2048h`` and ``?1016h`` on the wire. This is exactly what
an unpatched boot was measured doing — ``?2048h`` at byte ~123, ``?1016h`` at
~131, reports immediately after — so without the guard our reset would be undone
within milliseconds of being written, by us.

A BOOT-TIME RESET CANNOT CATCH A CO-TENANT THAT DIRTIES THE MODE LATER. Mode
2048 is per-VT state shared by every process on that tty, and the report for the
resize that REVEALS a dirty mode arrives before any handler could act on it: the
parser latches ``mouse_pixels`` while parsing that very report.

THE PARSER GATE. :func:`install_pixel_mouse_gate` patches the CLASS method
``XTermParser.parse_mouse_code`` to clear ``mouse_pixels`` before delegating, so
that latch can never scale a coordinate we parse. It installs itself only when
the negotiation is CLOSED — :func:`pixel_mouse_negotiation_open` mirrors the
gate Textual puts on its own mode-report branch — because with the negotiation
closed we never put ``?1016h`` on the wire AND we take that mode back off the
terminal (see THE CO-TENANCY TRADE below), so no delivered report can be a
legitimate statement that our coordinates are pixels. The decision is read from
``constants.SMOOTH_SCROLL``/``IS_ITERM`` rather than from
:func:`guard_pixel_mouse_latch`'s return value, and that matters: the guard
defers to an inherited ``TEXTUAL_SMOOTH_SCROLL=0`` — which is the value the guard
itself writes, so every re-entrant boot and every user who followed our own
documentation would silently lose both halves of this fix. An explicit
``TEXTUAL_SMOOTH_SCROLL=1`` on a non-iTerm terminal still negotiates and keeps
upstream behaviour byte for byte, because there no report is illegitimate.
Provenance: a LOCAL workaround for upstream
behaviour, deliberately narrow (the one latch, never Textual's resize or
terminal-size handling), idempotent, and reversible so a test can uninstall it.
It lives in lop rather than upstream because the trade it implements is OURS: we
are the ones who chose to stop negotiating pixel coordinates (THE ACCEPTED COST,
below), so we are the ones who must not let a divisor that is only correct for
that negotiation keep running — or a pixel-scale stream keep arriving. DELETE IT
once upstream gates the latch on the mode actually having been requested — the
issue's own mitigation candidate 3, i.e. ``mouse_pixels`` set only where
``SMOOTH_SCROLL`` is on. The tripwire that a Textual bump shipping that fix
trips is ``test_pixel_mouse_latch.py::
test_in_band_report_latches_pixel_mouse_coordinates``: it asserts an UNGATED
parser still leaves ``mouse_pixels`` True after a report, so a release that gates
the latch fails it rather than passing silently.

THE MID-SESSION RE-CLEAN. :class:`InBandResizeReclaimer` re-asserts both resets
(``CSI ?2048l`` and ``CSI ?1016l``) through the app's driver writer on ``Resize``
and on focus-in — the issue's title, answered within one interaction instead of
at the next boot. It exists on the same condition as the gate (the app builds one
only where :func:`pixel_mouse_gate_installed` reports the gate in force), so a
user whose textual is still negotiating keeps their smooth scrolling and pixel
coordinates. The sink is ``driver.write`` and not ``sys.__stderr__`` for the
reason ``terminal_title.py`` and ``tui/images.py`` document: Textual serialises
every byte it paints through one writer thread, so a second writer interleaves
an escape into the middle of a frame. Volume: one 16-byte write per delivered
``Resize`` message and per focus gain, with no per-frame or per-widget
multiplier. Textual's ``App._on_resize`` (``app.py:4345-4356``) coalesces the
SCREEN-level re-arrange at 1/120 s and returns early on an unchanged size, but
that early return is inside its own handler, so the public ``on_resize`` still
runs for every delivered message — the ceiling is the terminal's SIGWINCH
delivery rate, the only remaining source in this configuration because the
in-band path is off. A clock, a flush timer and their tests to turn a
few-times-a-second 16-byte write into a smaller number of 16-byte writes is not a
trade worth making, so this deliberately does not coalesce.

SO EACH MECHANISM IS LOAD-BEARING AND NONE IS REDUNDANT. The reset clears the
modes AT THE TERMINAL — 2048 and, for the reason below, 1016 — including a sticky
pair we inherited; the environment guard
stops TEXTUAL from turning it straight back on after seeing the reset; the
parser gate keeps a report that arrives anyway from scaling our coordinates,
which no amount of not-asking can do; and the mid-session re-clean closes a mode
a co-tenant set AFTER boot, which the boot reset cannot reach. Remove any one and
the divisor latches again — the first on state left set in the VT by an earlier
app, the second on the driver's reply to its own query, the third on a dirty
mode that outlived our boot, the fourth on a co-tenant that dirtied it while we
were running.

ORDERING. :func:`reset_in_band_resize` — both of its sequences — must reach the
wire before the driver issues its ``CSI ?2048$p`` query at
``linux_driver.py:299``, and :func:`guard_pixel_mouse_latch` must run before
``textual.constants`` is imported, because ``SMOOTH_SCROLL`` is a ``Final`` read
once at import time.
:func:`install_pixel_mouse_gate` therefore runs AFTER the guard — it imports
``textual._xterm_parser`` (which imports ``textual.constants``), so it must not
be what freezes that constant ahead of the guard's write — and its own
precondition is read from those frozen constants, which is only meaningful after
they exist. It must run before the app starts reading input, since it patches the
class the driver's parser is an instance of.

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

THE CO-TENANCY TRADE, WHICH IS THE PREMISE THE GATE RESTS ON. Under the guard we
ask for NEITHER in-band resize reports NOR pixel-scale mouse reporting, so both
resets are asserted at boot and mid-session: ``?2048l`` (the mode whose report
reveals the state) and ``?1016l`` (the mode that makes a terminal send PIXELS).
This is the same class of action #979 already took for 2048, on the same tty and
for the same reason: a mode THIS PROCESS DID NOT SET, on state shared with every
process on the tty, that no negotiation of ours clears and that changes what the
numbers on the wire mean. The trade is therefore explicit rather than implied —
a co-tenant that wanted pixel mouse loses it — and it is what makes "our
coordinates are cells" true at the terminal instead of assumed.

The gate's divisor clearing is only correct where the coordinates really are
cells, and that is exactly what the 1016 reset establishes. The three
configurations a VT can be in, and why the pair of resets is what collapses them
to the one our code assumes:

- **2048 only** — cell-scale mouse plus in-band reports: the reported bug
  (#986), which the latch gate fixes.
- **2048 and 1016** — genuine PIXEL-scale mouse plus in-band reports, which is
  what an unpatched ``lop`` boot writes (THE RE-ENABLE BRANCH above) and what a
  co-tenant enabling pixel mouse sets. Here Textual's division was CORRECT, so a
  gate that only cleared the divisor inverts the error: cell (5, 2) arrives as
  the wire's (41, 33) — a measured (8.00, 16.00) cell — and is read as (40, 32);
  past the screen edge the event is swallowed whole, so the symptom there is
  dead hover. Clearing 1016 fixes this arm, because the terminal then reports
  cells.
- **1016 only** — pixel-scale mouse with no reports, so nothing latches the
  divisor and pixels are read as cells: a misread that predates this fix and
  that the same ``?1016l`` fixes.

A ``TEXTUAL_SMOOTH_SCROLL=1`` user keeps upstream behaviour untouched: nothing is
installed for them (the gate and the re-clean both key on the negotiation being
closed), so their pixel coordinates stay correct. The BOOT reset is
unconditional, exactly as #979's ``?2048l`` already was — and the driver
re-enables ``?1016h`` on its own negotiation reply within milliseconds for that
user (``linux_driver.py:480-482``, the same branch that wrote it in the first
place), so their pixel mouse is restored by Textual rather than lost to us.

Deliberately stdlib-only at module scope, importing nothing from this package and
— critically — nothing from ``textual``: the same leaf discipline as
``terminals.py``, and here it is also a correctness requirement, since importing
this module must not be what pulls ``textual.constants`` in ahead of the guard.
:func:`install_pixel_mouse_gate` and :func:`pixel_mouse_gate_installed` import
``textual._xterm_parser`` INSIDE the call for exactly that reason: patching a
third-party class is not worth breaking the invariant that this module is safe
to import at any point in the boot.

All references pinned to textual 8.2.8. Every bare ``linux_driver.py`` means
``textual/drivers/linux_driver.py`` — there is no top-level file of that name,
so the prefix is what makes a reference greppable — and ``_xterm_parser.py``,
``constants.py``, ``app.py``, ``scrollbar.py`` and ``messages.py`` are the
files of those names in the ``textual`` package root.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING, Any, Callable, MutableMapping, TextIO

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed
    # Deliberately NOT a runtime import: see the module docstring's leaf
    # discipline. These two names exist so the patched method and the reclaimer
    # can be annotated without pulling ``textual`` in at import time.
    from textual._xterm_parser import XTermParser as _XTermParser
    from textual.message import Message

#: ``CSI ? 2048 l`` — reset in-band window resize notifications. Addressed at the
#: TERMINAL rather than at Textual's view of the mode because mode 2048 is
#: STICKY per-VT: a previous app can have left it set, and no amount of
#: not-negotiating on our side clears state we inherited.
DISABLE_IN_BAND_RESIZE = "\x1b[?2048l"

#: ``CSI ? 1016 l`` — reset pixel-scale mouse reporting. Cleared alongside 2048
#: and for the same class of reason: it is per-VT state somebody else can set
#: (Textual's re-enable branch writes ``?2048h`` AND ``?1016h`` together,
#: ``linux_driver.py:480-482``), and while it is set a compliant VT sends
#: PIXEL coordinates where we parse cells. Under the guard we ask for neither
#: mode, so asserting both off is what makes "our coordinates are cells" true
#: at the terminal instead of assumed — see THE CO-TENANCY TRADE in the module
#: docstring for the arm this exists for.
DISABLE_PIXEL_MOUSE = "\x1b[?1016l"

#: Both resets as ONE write, in the mirror-image order of the driver's own
#: re-enable branch (``?2048h`` then ``?1016h``, ``linux_driver.py:480-482``):
#: one write because the pair has to reach the driver's serialised writer as one
#: unit (a frame painted between them would be sized by a mode we have already
#: given up on), and this order because the only sequence we are undoing is
#: upstream's — so ours reads as its inverse at a glance.
DISABLE_PIXEL_SCALE_MODES = DISABLE_IN_BAND_RESIZE + DISABLE_PIXEL_MOUSE

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


def _textual_honours(value: str) -> bool:
    """True when Textual's own parse gives ``value`` meaning.

    ``constants._get_environ_int`` returns its DEFAULT for a value ``int()``
    rejects, and its caller compares that default to 1, so an unparseable
    value reads to Textual as "smooth scrolling ON". Presence alone is
    therefore not a statement of intent, and this predicate is deliberately
    the same ``int()``: the guard and Textual must agree on which values are
    meaningful, or the guard defers to a value Textual discards.
    """
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def guard_pixel_mouse_latch(env: MutableMapping[str, str] | None = None) -> bool:
    """Ask Textual not to negotiate in-band resize; True when we set it.

    Returns False when the variable is already present AND holds a value
    Textual honours — a deliberate user override we leave alone, because
    someone who set ``TEXTUAL_SMOOTH_SCROLL`` to an integer has said what they
    want on their terminal (``1`` keeps smooth scrolling and pixel
    coordinates; any other integer turns the negotiation off).

    A present but unparseable value is NOT an override and is treated as
    absent: Textual ignores it and falls back to its default of 1, i.e. to
    smooth scrolling ON, so deferring to it would silently disable this fix.

    Must be called before ``textual.constants`` is imported to have any effect.
    """
    if env is None:
        env = os.environ
    inherited = env.get(_SMOOTH_SCROLL_ENV)
    if inherited is not None and _textual_honours(inherited):
        return False
    # Assignment, not setdefault: the deferral above is the only path that
    # leaves an inherited value alone, so a default here would be unreachable.
    env[_SMOOTH_SCROLL_ENV] = "0"
    return True


def reset_in_band_resize(stream: TextIO | None = None) -> bool:
    """Write ``CSI ?2048l`` and ``CSI ?1016l``; True when the bytes were written.

    Both modes, because under the guard we ask for neither: see
    :data:`DISABLE_PIXEL_SCALE_MODES` and THE CO-TENANCY TRADE in the module
    docstring. The name still says 2048 because that is the mode #979 closed
    here and the one this function's callers know it for.

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
        stream.write(DISABLE_PIXEL_SCALE_MODES)
        stream.flush()
    except (OSError, ValueError):
        # Redirected, detached or closed stderr. Nothing to reset, and a
        # cosmetic escape is never worth failing a boot over.
        return False
    return True


#: Attribute this module stamps onto the wrapper it installs, so
#: :func:`install_pixel_mouse_gate` can tell its own wrapper from upstream's
#: method without comparing against a saved reference (which would be a second
#: source of truth for "installed").
_GATE_MARKER = "_lop_pixel_mouse_latch_gate"


def pixel_mouse_negotiation_open() -> bool:
    """True when Textual itself may negotiate pixel-mouse coordinates.

    Mirrors, deliberately one-for-one, the gate Textual 8.2.8 puts on its own
    mode-report branch (``_xterm_parser.py:319-322``): that branch emits the
    ``InBandWindowResize`` token — and the driver's ``process_message`` then
    answers ``;2`` with ``?2048h`` + ``?1016h`` (``linux_driver.py:470-483``) —
    only when ``constants.SMOOTH_SCROLL`` is on AND the terminal is not iTerm
    (whose pixel handshake Textual handles its own way). With the branch shut,
    nothing we do asks the terminal to send an in-band report, so a report that
    arrives anyway is not a statement about OUR coordinates and dividing by its
    cell size is wrong.

    Read from the constants rather than from :func:`guard_pixel_mouse_latch`'s
    return value, and that is a correctness requirement rather than a
    preference: the guard defers to an inherited ``TEXTUAL_SMOOTH_SCROLL=0`` —
    a value the guard ITSELF wrote on an earlier boot in this process, and the
    value any user who followed our own documentation has exported. Keying on
    "did this call write the variable" would therefore drop both halves of this
    fix on exactly the configuration we tell people to use.

    Must be called after ``textual.constants`` is imported (both names are
    frozen at import time), which in production order is after the guard.
    """
    from textual import constants
    from textual._xterm_parser import IS_ITERM

    return bool(constants.SMOOTH_SCROLL and not IS_ITERM)


def install_pixel_mouse_gate() -> bool:
    """Stop the in-band-report latch from scaling coordinates; True if installed.

    Patches the CLASS method ``XTermParser.parse_mouse_code``, which is the only
    place the divisor is computed (``_xterm_parser.py:94-102``), to clear
    ``mouse_pixels`` before delegating to the original. The class and not an
    instance because the driver builds its parser when the app starts, after
    this runs; a per-instance patch would have to reach into the driver.

    Refuses — installing nothing, returning False — while
    :func:`pixel_mouse_negotiation_open` is True: a user whose textual still
    negotiates pixel mouse asked for pixel coordinates, and forcing the divisor
    off would ignore that, so upstream behaviour stays byte for byte. That
    precondition is checked HERE rather than at the call site so no caller can
    install the gate in a configuration where a delivered report is legitimate.

    Narrow on purpose: it touches the ONE one-way latch and nothing else.
    Textual's resize handling, ``terminal_size``/``terminal_pixel_size``
    bookkeeping and the ``Resize`` token are all left exactly as upstream has
    them, so the out-of-band SIGWINCH path the guard puts us on is unaffected.

    Idempotent — single-threaded by contract: the check-then-``setattr`` below
    takes no lock, and two concurrent installs could both wrap (uninstall would
    then peel one layer), which is fine because the only caller is the boot
    thread. Installing over its own wrapper returns False and does not wrap
    twice. Reversible by :func:`uninstall_pixel_mouse_gate`, which restores the
    original function object, so a test can leave the class as it found it.
    """
    from textual._xterm_parser import XTermParser

    if pixel_mouse_negotiation_open():
        return False

    current = XTermParser.parse_mouse_code
    if getattr(current, _GATE_MARKER, False):
        return False
    original = current

    # ``functools.wraps`` for the ``__wrapped__`` link uninstall follows, and
    # because a bare wrapper loses the original's name and docstring in any
    # traceback that goes through mouse parsing.
    @functools.wraps(original)
    def parse_mouse_code(self: _XTermParser, code: str) -> Message | None:
        # The latch is one-way and ungated upstream (a delivered report sets it
        # whatever we negotiated), so clearing it here is what keeps the divisor
        # off every coordinate this parser will ever see.
        if self.mouse_pixels:
            self.mouse_pixels = False
        return original(self, code)

    setattr(parse_mouse_code, _GATE_MARKER, True)
    # ``setattr`` on the class rather than an annotated assignment: this is a
    # third-party class we do not import for typing, and the wrapper is a plain
    # function, so it still binds as a method.
    setattr(XTermParser, "parse_mouse_code", parse_mouse_code)
    return True


def uninstall_pixel_mouse_gate() -> bool:
    """Undo :func:`install_pixel_mouse_gate`; False when nothing was installed.

    Restores the ORIGINAL function object rather than unwrapping by hand, so a
    test that installs and uninstalls leaves ``XTermParser`` byte-identical to
    how it found it — including after several install/uninstall cycles.
    """
    from textual._xterm_parser import XTermParser

    current = XTermParser.parse_mouse_code
    if not getattr(current, _GATE_MARKER, False):
        return False
    original = getattr(current, "__wrapped__", None)
    if original is None:  # pragma: no cover - the marker implies the link
        return False
    setattr(XTermParser, "parse_mouse_code", original)
    return True


def pixel_mouse_gate_installed() -> bool:
    """True when the parser class currently carries the gate.

    The app asks this rather than remembering the guard's return value, so the
    gate and the re-clean cannot disagree about which configuration is in
    force: there is one answer, and it is the state of the class.
    """
    from textual._xterm_parser import XTermParser

    return bool(getattr(XTermParser.parse_mouse_code, _GATE_MARKER, False))


class InBandResizeReclaimer:
    """Re-assert the pixel-scale resets mid-session through an injected sink.

    Both halves of :data:`DISABLE_PIXEL_SCALE_MODES` — ``?2048l`` for the report
    mode the resize reveals, ``?1016l`` for the pixel-mouse mode a co-tenant
    sets with it, which is what keeps the coordinates arriving as cells.

    The boot reset cannot reach a mode a co-tenant sets while we run, and the
    report that reveals it arrives before any handler could act, so this is the
    recency half: :meth:`reclaim` runs on ``Resize`` and on focus-in and closes
    the mode within one interaction.

    The sink is whatever the caller passes — in the app, ``driver.write``, which
    is ``App._driver.write`` and the same door ``terminal_title.py``,
    ``tui/images.py`` and ``tui/notify.py`` use, because Textual serialises
    everything it paints through one writer thread and a second writer
    interleaves an escape into the middle of a frame.

    Headless and no-driver gating lives with the caller (the app builds one of
    these only when it has a driver and is not headless), so this object stays a
    single-purpose, directly testable thing.
    """

    __slots__ = ("_env", "_write")

    def __init__(
        self,
        write: Callable[[str], Any],
        *,
        env: MutableMapping[str, str] | None = None,
    ) -> None:
        self._write = write
        #: Injection point for tests, mirroring :func:`reset_in_band_resize`'s
        #: stream parameter: production passes nothing and reads ``os.environ``
        #: at call time, which is what makes the kill switch work when it is set
        #: after boot.
        self._env = env

    def reclaim(self) -> bool:
        """Hand ``CSI ?2048l`` and ``CSI ?1016l`` to the sink; True when sent.

        The kill switch is the same ``LOCAL_OPERATOR_NO_MODE_RESET`` the boot
        reset honours, and it is read per call rather than cached so setting it
        mid-session suppresses the next write.

        A failed write is swallowed, like :func:`reset_in_band_resize`'s: a
        writer thread that has already died must not turn a resize into an
        exception, and a cosmetic escape is never worth an event.
        """
        env = os.environ if self._env is None else self._env
        if env.get(_ENV_DISABLE):
            return False
        try:
            self._write(DISABLE_PIXEL_SCALE_MODES)
        except (OSError, ValueError):
            return False
        return True

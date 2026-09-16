"""Keep a non-UTF-8 byte on stdin from killing the app, and read X10 mouse reports.

THE SYMPTOM. ``lop`` boots, paints, then exits on its own a few seconds later.
Nothing in the transcript explains it; the only trace is a decoder error in
``local-operator.log``:

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 4: invalid start byte

That is the app dying on TERMINAL INPUT, not on a resource it loaded.

THE DEFECT. Textual's driver decodes stdin with a STRICT incremental UTF-8
decoder, created once and reused for every read
(``textual/drivers/linux_driver.py:430`` — ``getincrementaldecoder("utf-8")()``
— and used at ``:447``). A byte that cannot start a UTF-8 sequence therefore
raises ``UnicodeDecodeError`` out of ``run_input_thread``, which LinuxDriver's
wrapper answers by calling ``self._app.panic(...)`` (``:405-410``); the app is
gone within a frame. The input path has no fallback: ANY stray byte is fatal,
and the decoder's own ``errors`` default is ``strict``.

WHERE THE BYTE COMES FROM — LEGACY X10 MOUSE REPORTS. Under DECSET 1000 without
SGR 1006, a terminal reports the mouse as ``CSI M`` followed by THREE RAW BYTES:
``ESC [ M  <button+32> <x+32> <y+32>``. Those bytes are the coordinate values
plus 32, so any coordinate past cell 95 is >= 0x80 and is not valid UTF-8. The
position in the operator's log is the giveaway: index 4 is the x byte (``ESC``,
``[``, ``M``, button, x), and x=96 is 0x80, x=215 is 0xf7 — the two values in
the two log lines.

WHY IT IS NEW IN 0.54.35. The bytes on the wire are the terminal's business; what
changed is ours. 0.54.34's boot wrote only ``CSI ?2048l``; 0.54.35 added
``CSI ?1016l`` to the same write (6749a8438, "clear DEC mode 1016 together with
the 2048 reset") and re-asserts it mid-session through
:class:`~local_operator.tui.terminal_modes.InBandResizeReclaimer`. Resetting
pixel-scale mouse reporting leaves a terminal that honours 1016 reporting the
mouse in the LEGACY binary encoding unless something asks for SGR. Stated
honestly because it is inference rather than measurement: the decode crash and
our new reset writes are both proven here, but which physical terminal behaviour
the reset selects could not be identified from the logs, so the wiring between
them is the best explanation of the timing rather than a reproduced fact.

WHAT THIS MODULE DOES — TWO HALVES, BOTH NEEDED.

1. **Nothing can raise.** :class:`NonFatalStdinDecoder` replaces the driver's
   strict decoder factory, so the input thread's decode step is incapable of
   raising whatever arrives: bytes it cannot interpret are replaced, never
   raised. Installed by :func:`install_nonfatal_stdin_decode` onto the
   module-level ``getincrementaldecoder`` name the driver looked up
   (``linux_driver.py:10``, ``:430``), which is the narrowest lever that reaches
   the reader — the same shape as :mod:`local_operator.tui.terminal_modes`'
   patch of ``XTermParser.parse_mouse_code``.

2. **The mouse keeps WORKING, not merely not-crashing.** A lossy decode first
   would destroy the report: ``errors="replace"`` turns 0x80 into U+FFFD and the
   coordinate is gone, so the app would survive and the pointer would be dead.
   The decoder therefore scans the RAW bytes for the X10 introducer, translates
   a complete report into the SGR form Textual's parser actually implements
   (:func:`x10_mouse_to_sgr`), and only then decodes what is left. The parser
   accepts ``CSI < b ; x ; y M`` (``_xterm_parser.py:64``) and — for the legacy
   form — matches the sequence and then finds no handler at all
   (``parse_mouse_code`` only understands SGR, ``:84``), so an untranslated
   report is silently dropped: alive and pointerless is the other half of the
   bug, and this is the half that fixes it.

WHAT THIS DELIBERATELY IS NOT. It is not a terminal-mode fix: the mode resets
live in :mod:`local_operator.tui.terminal_modes` and this module neither reads
nor writes a terminal. That is why it is installed UNCONDITIONALLY — not behind
``TEXTUAL_SMOOTH_SCROLL``, not behind ``LOCAL_OPERATOR_NO_MODE_RESET``, and not
only while the pixel-mouse negotiation is closed. Those switches decide what we
ASK the terminal for; this decides what we do with a byte that was already on
its way, and no user preference or kill switch should be able to make an
unfamiliar byte fatal. It also imports nothing from ``textual`` at module
scope, for the import-weight reason ``terminal_modes`` documents: ``cli``
imports the TUI package in interactive mode only, and a Textual import here
would freeze ``textual.constants`` before ``run_tui`` sets the guard.

WHY NOT CATCH AT THE DRIVER'S READ INSTEAD. Catching ``UnicodeDecodeError``
around the read is the obvious alternative and it is worse in three ways. The
read is up to 4096 bytes, so the whole chunk — keystrokes included — would be
thrown away rather than just the bytes that cannot be text. The incremental
decoder would be left holding a partial sequence with no way to resynchronise,
so the NEXT read would start mid-character and mis-decode valid text after the
bad byte. And it would need a patch inside ``run_input_thread`` itself, which
is a larger and more fragile surface than the factory the module already looks
up by name. Doing it at the decoder keeps everything after the report intact.

LINUX DRIVER ONLY, ON PURPOSE. ``textual.drivers.windows_driver`` has no byte
decoder at all (it reads the console through the win32 API), so this class of
failure does not exist there and there is nothing to patch — which is also why
:func:`install_nonfatal_stdin_decode` tolerates the module being unimportable
rather than requiring it.
"""

from __future__ import annotations

from codecs import getincrementaldecoder as _codecs_getincrementaldecoder
from typing import Any

#: ``ESC [ M`` — the introducer of a legacy X10 mouse report. Textual's own
#: parser matches this form too (``_re_mouse_event``, ``_xterm_parser.py:29``),
#: which is what makes 6 bytes the right frame here: the same three bytes it
#: would consume as a report.
X10_MOUSE_PREFIX = b"\x1b[M"

#: Introducer plus button, x and y — the whole report, always six bytes.
X10_MOUSE_LENGTH = 6


#: Attribute stamped on the installed factory so install/uninstall/installed
#: can agree by looking at the object rather than at a module-level flag that a
#: second install could disagree with.
_MARKER = "_lop_nonfatal_stdin_decode"


def _held_partial_introducer(buffer: bytes) -> int:
    """Bytes to hold back because they may start an X10 report: 2, else 0.

    A trailing ``ESC[`` may be the first two bytes of a report whose remaining
    four have not been read yet, and the parser can act on neither half alone,
    so holding it costs nothing. A trailing ``ESC[M`` never reaches here — the
    scan finds the introducer and the report branch holds the fragment — and a
    trailing LONE ``ESC`` is answered with 0 deliberately: it can start anything,
    so the parser's own ESC timeout is the right owner of it, and holding it
    would delay the ESC key until the user's next keystroke.
    """
    return 2 if buffer.endswith(X10_MOUSE_PREFIX[:2]) else 0


def x10_mouse_to_sgr(report: bytes) -> str | None:
    """Translate one complete X10 report into SGR; ``None`` when not a report.

    The protocol adds 32 to every value, so ``report[3:]`` minus 32 recovers the
    button code and the 1-BASED cell coordinates, which is exactly what SGR
    carries (Textual subtracts 1 again at ``_xterm_parser.py:88-89``). The
    subtraction is done on the RAW bytes and not on a decoded string: decoding
    first is what destroys these values.

    ``None`` — the report is dropped — for a byte below 32, which no encoder can
    emit for a value it added 32 to. That is the "anything we cannot interpret
    is discarded harmlessly" half of the contract, and it cannot stand in for
    the fatal path: the caller's job is to make dropping the only consequence.

    The final byte is always ``M``. Legacy X10 has no press/release marker of
    its own (a release is button code 3), so the code is carried verbatim and
    Textual's SGR interpretation applies to it, rather than this function
    inventing a distinction the wire did not carry. Clicks, drags, hovers and
    wheel events translate exactly; a legacy release reads as a move, which is
    what Textual does with the equivalent ``CSI < 3 ; x ; y M`` too.
    """
    button, x, y = report[3] - 32, report[4] - 32, report[5] - 32
    if button < 0 or x < 0 or y < 0:
        return None
    return f"\x1b[<{button};{x};{y}M"


class NonFatalStdinDecoder:
    """Codecs-shaped incremental decoder that cannot raise on terminal bytes.

    The interface the driver uses is ``decode(data: bytes, final: bool) -> str``
    (``linux_driver.py:447``); nothing else is promised and nothing else is
    implemented. ``__init__`` accepts and ignores the arguments
    ``codecs.getincrementaldecoder`` would pass through (``errors`` first), so a
    Textual release that starts passing them cannot turn a construction into a
    TypeError on the way to a boot.
    """

    __slots__ = ("_utf8", "_pending")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # ``errors="replace"`` is the safety property, and it is on the decoder
        # that reads the bytes we did NOT translate: an uninterpretable byte
        # becomes U+FFFD instead of an exception, and U+FFFD is a character the
        # parser, the transcript and the renderer all already handle.
        #
        # Incremental and instance-held, like the driver's own: a multi-byte
        # character split across two reads still arrives as one character.
        self._utf8 = _codecs_getincrementaldecoder("utf-8")(errors="replace")
        #: A partial X10 report at a read boundary, held RAW. Only ever 2-5
        #: bytes, and only while the tail of the buffer is the introducer or a
        #: prefix of a report. A LONE ``ESC`` is deliberately NOT held (see
        #: :meth:`_decode`): holding it would delay the ESC key until the user's
        #: next keystroke, and losing one report beats losing a keybinding.
        self._pending = b""

    def decode(self, data: bytes | bytearray | str, final: bool = False) -> str:
        """Decode ``data``; never raises, whatever the bytes are.

        The guard is a real one rather than a formality: this runs on the input
        thread, where the exception this module exists to remove was fatal, so a
        bug in the translation below must degrade to "those bytes are dropped"
        and not to "the app panics". A caller that hands us a ``str`` — not the
        driver's contract, but nobody else's decoder would refuse it — gets it
        back untouched.
        """
        try:
            return self._decode(data, final)
        except Exception:  # pragma: no cover - the fallback is the point
            if isinstance(data, (bytes, bytearray)):
                return bytes(data).decode("utf-8", errors="replace")
            return str(data)

    def _decode(self, data: bytes | bytearray | str, final: bool) -> str:
        if isinstance(data, str):
            return data
        buffer = self._pending + bytes(data)
        self._pending = b""
        out: list[str] = []
        while buffer:
            start = buffer.find(X10_MOUSE_PREFIX)
            if start < 0:
                hold = _held_partial_introducer(buffer)
                if hold and not final:
                    ordinary = buffer[: len(buffer) - hold]
                    self._pending = buffer[len(buffer) - hold :]
                else:
                    # At EOF a held introducer is never completed, and an
                    # unheld tail is just text: both go to the lenient decoder,
                    # and let the parser's own incomplete-sequence handling
                    # decide what a dangling ``ESC[`` means.
                    ordinary = buffer
                if ordinary:
                    out.append(self._utf8.decode(ordinary, final=final))
                break
            if start:
                out.append(self._utf8.decode(buffer[:start], final=False))
            report = buffer[start : start + X10_MOUSE_LENGTH]
            if len(report) < X10_MOUSE_LENGTH:
                # The reader's chunk boundary fell inside a report. Hold the
                # fragment undecoded and resume on the next read: this is the
                # only reason the raw-byte scan is stateful.
                if not final:
                    self._pending = report
                break
            translated = x10_mouse_to_sgr(report)
            if translated is not None:
                out.append(translated)
            buffer = buffer[start + X10_MOUSE_LENGTH :]
        if final:
            # EOF: flush a half-finished character, and let go of any held
            # fragment — nothing will ever complete it.
            self._pending = b""
            out.append(self._utf8.decode(b"", final=True))
        return "".join(out)

    def reset(self) -> None:
        """Forget buffered state, mirroring ``codecs``' incremental decoders."""
        self._utf8.reset()
        self._pending = b""


def decoder_factory(encoding: str, *args: Any, **kwargs: Any) -> Any:
    """Stand-in for ``codecs.getincrementaldecoder`` inside the driver module.

    Only UTF-8 is replaced — that is the encoding the driver asks for and the
    only one whose strictness is fatal here. Every other encoding is delegated
    to the real factory, so a Textual release that asks for something else keeps
    exactly the decoder it asked for.
    """
    if str(encoding).replace("-", "").lower() == "utf8":
        return NonFatalStdinDecoder
    return _codecs_getincrementaldecoder(encoding, *args, **kwargs)


def _driver_module() -> Any | None:
    """The module whose ``getincrementaldecoder`` name the input thread read.

    The import is function-local for the import-weight reason the module
    docstring gives: ``cli`` imports the TUI package in interactive mode only,
    and a module-scope Textual import here would freeze ``textual.constants``
    before ``run_tui`` sets the guard.

    ``None`` when the module cannot be imported at all, which is the WINDOWS
    case: ``linux_driver`` imports ``termios`` and ``tty``, neither of which
    exists there, and Textual drives Windows through ``windows_driver``'s win32
    console API instead — that driver has no byte decoder and no equivalent
    crash, so there is nothing to harden and nothing to be sorry about. It has
    to be handled rather than assumed because ``run_tui`` calls the install
    unconditionally, on every platform: an ImportError raised there would be a
    new boot failure in the fix for a boot failure.
    """
    try:
        import textual.drivers.linux_driver as linux_driver
    except ImportError:  # pragma: no cover - Windows, and a Textual that moves it
        return None
    return linux_driver


def install_nonfatal_stdin_decode() -> bool:
    """Point the driver's decoder factory at :func:`decoder_factory`.

    Returns True when this call installed it, False when it was already
    installed or when there is no such module to patch (:func:`_driver_module`).
    The second False is deliberately silent rather than fatal: a decoder we
    cannot find is a hardening we lost, not a reason to refuse a boot. The
    tripwire that catches the Linux case is a test, not this function:
    ``test_stdin_decode.py::test_the_driver_still_binds_the_decoder_at_module_level``
    fails on that release instead of letting the crash quietly return.

    Idempotent, and reversible by :func:`uninstall_nonfatal_stdin_decode`. Must
    run before the driver's input thread starts, which in the product is
    ``run_tui`` before the app is constructed — the same ordering constraint the
    other terminal patches carry, and for the same reason: the driver reads the
    factory when the thread begins.
    """
    module = _driver_module()
    if module is None:
        return False
    current = getattr(module, "getincrementaldecoder", None)
    if current is None or getattr(current, _MARKER, False):
        return False
    setattr(decoder_factory, _MARKER, True)
    module.getincrementaldecoder = decoder_factory
    return True


def uninstall_nonfatal_stdin_decode() -> bool:
    """Restore the stdlib factory the driver imported; False if none was ours.

    Restores the ORIGINAL object — ``codecs.getincrementaldecoder``, which is
    what ``from codecs import getincrementaldecoder`` bound — so a test that
    installs and uninstalls leaves the module byte-identical to how it found it.
    """
    module = _driver_module()
    if module is None:
        return False
    current = getattr(module, "getincrementaldecoder", None)
    if current is None or not getattr(current, _MARKER, False):
        return False
    module.getincrementaldecoder = _codecs_getincrementaldecoder
    return True


def nonfatal_stdin_decode_installed() -> bool:
    """True while the driver module carries our factory.

    Asked of the module rather than remembered by the caller, so the answer
    cannot drift from the patch actually in force — the reason
    :func:`terminal_modes.pixel_mouse_gate_installed` reads the class too.
    """
    module = _driver_module()
    if module is None:
        return False
    return bool(getattr(getattr(module, "getincrementaldecoder", None), _MARKER, False))

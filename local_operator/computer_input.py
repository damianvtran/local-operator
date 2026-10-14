"""Source generation for explicit X11 clipboard input, independent of benchmarks.

Only the guest imports GUI dependencies. The host can validate/compile without a
DISPLAY, xclip or pyautogui. This is deliberately not the composer's best-effort
clipboard helper: failure after ownership changes is an ambiguous mutation and
must escape to the caller's existing no-replay policy.
"""

from __future__ import annotations

import base64
import inspect
import textwrap
import unicodedata
from collections.abc import Sequence

MAX_PASTE_CHARACTERS = 100_000

#: Drag duration bounds. They mirror the protocol's DragAction floor/ceiling
#: (default 500 ms); the helper re-checks them so a caller that bypasses the
#: protocol cannot compile a zero, negative or unbounded hold into the guest.
#: A drift guard lives in the unit tests, exactly because a second unfettered
#: number here is how the paste/type bounds silently diverged before.
DRAG_MIN_DURATION_MS = 100
DRAG_MAX_DURATION_MS = 5_000
_DRAG_BUTTONS = ("left", "middle", "right")
#: Generous sanity bound on a native coordinate. The protocol validates
#: against the actual frame geometry; this only stops a nonsensical value
#: (negative, or absurd) from reaching pyautogui if a caller bypasses it.
MAX_DRAG_COORDINATE = 1_000_000


def validate_paste_text(text: str) -> None:
    """Reject unrepresentable/control payloads without exposing their contents."""
    if not 1 <= len(text) <= MAX_PASTE_CHARACTERS:
        raise ValueError("paste text must contain 1 to 100000 characters")
    if any(unicodedata.category(char) in ("Cc", "Cs") and char not in "\t\r\n" for char in text):
        raise ValueError("paste text contains unsupported controls or invalid Unicode")
    text.encode("utf-8", errors="strict")


def _paste_x11(encoded: str, keys: tuple[str, ...]) -> None:
    # This function is shipped as source into a guest that need not have this
    # package installed. Keep its imports and dependencies local and stdlib-only
    # until the explicit keyboard dispatch at the very end.
    import base64
    import importlib
    import os
    import selectors
    import shutil
    import subprocess
    import tempfile
    import time

    # The host intentionally has no GUI dependency; resolve it only in the
    # generated guest program, where the adapter already requires it.
    pyautogui = importlib.import_module("pyautogui")

    payload = base64.b64decode(encoded, validate=True)
    executable = shutil.which("xclip")
    if executable is None:
        raise RuntimeError("clipboard transport unavailable: xclip is required")
    deadline = time.monotonic() + 5.0
    owner = None
    success = False

    def read_new_payload() -> bytes:
        # communicate()/capture_output can accumulate an unbounded selection if
        # another application replaces CLIPBOARD during readiness. Limit BOTH
        # the bytes read and the time; never include those bytes in diagnostics.
        reader = subprocess.Popen(
            [executable, "-selection", "clipboard", "-out", "-target", "UTF8_STRING"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert reader.stdout is not None
            data = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(reader.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise RuntimeError("clipboard readiness timed out")
                    part = os.read(reader.stdout.fileno(), min(65536, len(payload) + 1 - len(data)))
                    if not part:
                        break
                    data.extend(part)
                    if len(data) > len(payload):
                        # The old owner may still be visible while xclip takes
                        # ownership. Discard at the bound and poll readiness;
                        # never retain a larger old selection or dispatch keys.
                        return b""
            if reader.wait(timeout=max(0.001, deadline - time.monotonic())) != 0:
                return b""
            return bytes(data)
        finally:
            if reader.poll() is None:
                reader.kill()
            reader.wait(timeout=1)
            if reader.stdout is not None:
                reader.stdout.close()

    try:
        # A regular, unnamed file avoids a full stdin pipe blocking a 400KB
        # UTF-8 payload before the timeout machinery gets a chance to run.
        with tempfile.TemporaryFile() as source:
            source.write(payload)
            source.seek(0)
            owner = subprocess.Popen(
                [
                    executable,
                    "-selection",
                    "clipboard",
                    "-in",
                    "-target",
                    "UTF8_STRING",
                    "-quiet",
                    "-loops",
                    "0",
                ],
                stdin=source,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        # -quiet keeps xclip in the foreground (no untracked fork). X11 holds
        # exactly one CLIPBOARD owner: this process exits when the next owner
        # replaces it. Do not use -loops 1: readiness itself is a selection
        # request, and consuming that quota would erase the paste's source.
        while True:
            if owner.poll() is not None:
                raise RuntimeError("clipboard owner exited before readiness")
            if time.monotonic() >= deadline:
                raise RuntimeError("clipboard readiness timed out")
            if read_new_payload() == payload:
                break
            time.sleep(0.01)
        # There is deliberately no focus/title heuristic, default chord, Enter,
        # clipboard restoration, native fallback, or replay. A successful chord
        # is not proof the receiving application inserted the requested text.
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        success = True
    except Exception:
        # Third-party errors can contain commands or clipboard contents. The
        # outer transport must see failure, but never an echo of the payload.
        raise RuntimeError("clipboard paste failed; input may have partially applied") from None
    finally:
        if not success and owner is not None:
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=1)


def paste_text_source(text: str, keys: Sequence[str]) -> str:
    """Compile one explicit clipboard replacement and keyboard dispatch.

    ``keys`` are backend key names supplied by the caller's validated key
    vocabulary. Python literals, never shell interpolation, carry them into the
    guest. CLIPBOARD remains the new text; PRIMARY is never selected or read.
    """
    validate_paste_text(text)
    if not keys or len(keys) > 8 or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("paste requires an explicit bounded key chord")
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    source = textwrap.dedent(inspect.getsource(_paste_x11))
    source += f"\n_paste_x11({encoded!r}, {tuple(keys)!r})"
    # Controllers commonly prepend imports with a semicolon, after which a
    # compound `def` statement is illegal. exec keeps this one simple statement.
    return f"exec({source!r})"


def _drag_x11(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    button: str,
    duration_ms: int,
) -> None:
    """One atomic press-move-release; shipped as source into the guest.

    This function is delivered as SOURCE, so it must stay import-cheap and
    stdlib-only until the explicit pyautogui resolution below: the host need
    not have a DISPLAY or pyautogui installed. All input mutation stays in
    THIS process — there is deliberately no detached mouse-down state that
    could survive the statement.
    """
    import importlib

    # The host intentionally has no GUI dependency; resolve it only in the
    # generated guest program, where the adapter already requires it.
    pyautogui = importlib.import_module("pyautogui")

    # `press_attempted` is marked BEFORE mouseDown: a press that injects and
    # then raises still leaves the button held, and this flag is the only
    # thing that makes the cleanup below attempt a release.
    press_attempted = False
    released = False
    failure = None
    release_failure = None
    original_failsafe = getattr(pyautogui, "FAILSAFE", None)
    try:
        pyautogui.moveTo(start_x, start_y)
        press_attempted = True
        pyautogui.mouseDown(button=button)
        # The drag's duration rides the PRESSED move: pyautogui eases the
        # motion over that many seconds while the button is held.
        pyautogui.moveTo(end_x, end_y, duration=duration_ms / 1000.0)
        pyautogui.mouseUp(button=button)
        released = True
    except Exception as exc:
        failure = exc
    finally:
        # These guarantees cover cooperative unwinding only (exceptions and
        # graceful cancellation): a client timeout or SIGKILL cannot run
        # finally, so a caller that cannot OBSERVE the release must retain an
        # unknown/provisional outcome and never replay a drag it could not
        # observe — the caller's no-replay policy stays the authority.
        if press_attempted and not released:
            try:
                # FAILSAFE is disabled ONLY around this release and restored in
                # the nested finally: a mouseUp at a screen corner must not
                # itself raise FailSafeException, and the original value must
                # survive even when the release fails. A naive finally:mouseUp
                # would either leave a held button or permanently disarm the
                # movement failsafe.
                try:
                    pyautogui.FAILSAFE = False
                    try:
                        pyautogui.mouseUp(button=button)
                        released = True
                    except Exception as exc:
                        release_failure = exc
                finally:
                    pyautogui.FAILSAFE = original_failsafe
            except Exception as exc:
                if release_failure is None:
                    release_failure = exc
    if release_failure is not None:
        # Truthful failure: never report success while the release is in
        # doubt. The original motion failure (if any) stays chained as cause.
        raise RuntimeError(
            "drag failed and the pressed button could not be released; "
            "the guest may still be holding a mouse button"
        ) from (failure or release_failure)
    if failure is not None:
        raise RuntimeError("drag failed after the press; the button was released") from failure


def validate_drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    button: str,
    duration_ms: int,
) -> None:
    """Reject unrepresentable drags without touching the guest."""
    for value in (start_x, start_y, end_x, end_y):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("drag coordinates must be integers")
        if not 0 <= value <= MAX_DRAG_COORDINATE:
            raise ValueError("drag coordinates must be inside the sanity bound")
    if button not in _DRAG_BUTTONS:
        raise ValueError("drag button must be one of left/middle/right")
    if not DRAG_MIN_DURATION_MS <= duration_ms <= DRAG_MAX_DURATION_MS:
        raise ValueError(f"drag duration must be {DRAG_MIN_DURATION_MS}..{DRAG_MAX_DURATION_MS} ms")


def drag_source(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    button: str,
    duration_ms: int,
) -> str:
    """Compile one atomic press-drag-release guest statement.

    Coordinates are NATIVE pixels already mapped by the adapter's
    ``FrameGeometry.model_to_native``; this function never rescales — a second
    geometry implementation would be exactly the divergent copy the adapter's
    comments forbid. Literals, never interpolation, carry the values into the
    guest. No host GUI dependency is imported and no raw Python tool is
    exposed to the model; the model only ever sees the typed action schema.
    """
    validate_drag(start_x, start_y, end_x, end_y, button, duration_ms)
    source = textwrap.dedent(inspect.getsource(_drag_x11))
    source += (
        f"\n_drag_x11({start_x!r}, {start_y!r}, {end_x!r}, {end_y!r}, {button!r}, {duration_ms!r})"
    )
    return f"exec({source!r})"


# The only program that ever reaches the guest's argv. It is a fixed constant:
# it names no action, carries no agent text, and is byte-identical for every
# statement we run, so nothing an agent writes can appear in it. ``decode()``
# takes no argument only because its default IS UTF-8, which is the encoding
# ``python_source_argv`` encodes with; the two must not drift apart.
_SOURCE_BOOTSTRAP = "import base64, sys; exec(base64.b64decode(''.join(sys.argv[1:])).decode())"

# Linux caps a single argv entry at MAX_ARG_STRLEN (32 pages = 128KB). Base64 is
# ASCII, so one character is one byte and this character bound IS the byte bound.
_MAX_ARGUMENT_CHARACTERS = 16_000


def python_source_argv(source: str) -> list[str]:
    """Carry one guest exec as base64, so our own argv never quotes the agent.

    *source* is base64-encoded rather than passed literally because argv is a
    PUBLIC channel: ``/proc/<pid>/cmdline`` is what ``pkill -f``, ``pgrep -f``
    and ``ps | grep`` match against, and the guest process running a statement is
    itself a running process. An agent cleaning up after itself with
    ``pkill -f "ffmpeg -y -f x11grab"`` — routine, legitimate, and its own
    recording to kill — used to match the very process typing that text and
    SIGTERM it, which is the observed ``exit -15`` that killed ep-0ce67ac2d3a1.
    Encoding removes the false match at its source; the alternative of screening
    the agent's text for process-matching commands is both incomplete and not
    ours to impose.

    Base64 also subsumes the byte-limit problem this function already solved. The
    clipboard accepts 100k Unicode characters (up to 400KB UTF-8, 533KB encoded),
    which no single argv entry can hold, so the encoded form is split across
    trailing arguments that the bootstrap rejoins. Splitting the *source*, never
    the actions, is what preserves the single-shot mutation boundary: the guest
    still performs exactly one exec, so a batch cannot half-commit.

    There is deliberately no short-source fast path emitting ``python -c
    <source>``. That shape is precisely the exposure, and a size threshold would
    reinstate it for every statement small enough to fit — which is all of them
    that matter here, since a ``pkill`` line is a few dozen characters.
    """
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return [
        "python",
        "-c",
        _SOURCE_BOOTSTRAP,
        *(
            encoded[offset : offset + _MAX_ARGUMENT_CHARACTERS]
            for offset in range(0, len(encoded), _MAX_ARGUMENT_CHARACTERS)
        ),
    ]

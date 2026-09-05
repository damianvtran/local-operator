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


def python_source_argv(source: str) -> list[str]:
    """Keep one guest exec without exceeding Linux's per-argument byte limit.

    The clipboard accepts 100k Unicode characters (up to 400KB UTF-8), so even
    one base64 payload exceeds MAX_ARG_STRLEN. Splitting *source*, not actions,
    preserves the single-shot mutation boundary. Small legacy commands retain
    their exact argv; each large-source chunk is at most 64KB in UTF-8.
    """
    if len(source.encode("utf-8")) <= 64_000:
        return ["python", "-c", source]
    return [
        "python",
        "-c",
        "import sys; exec(''.join(sys.argv[1:]))",
        *(source[offset : offset + 16_000] for offset in range(0, len(source), 16_000)),
    ]

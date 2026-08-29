"""Drive the REAL `local-operator` binary in a pty and send raw option+arrow bytes.

This is the floor of evidence for issue #370: not a test host, not a harness,
but the shipped entry point running under a pseudo-terminal, receiving the exact
bytes iTerm2/Terminal.app write for the Esc-prefixed option+arrow chord.
"""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import sys
import time
from pathlib import Path

WORKTREE = str(Path(__file__).resolve().parent.parent)
PYTHON = os.path.expanduser("~/local-operator/.venv/bin/python")
SAMPLE = "alpha beta gamma delta"
ANSI = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[=>]")


def read_for(fd: int, seconds: float) -> bytes:
    out = b""
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
    return out


def main() -> int:
    env = dict(os.environ)
    env.update(
        {
            "TERM": "xterm-256color",
            "PYTHONPATH": WORKTREE,
            "COLUMNS": "100",
            "LINES": "30",
        }
    )
    env.pop("NO_COLOR", None)

    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(WORKTREE)
        os.execve(PYTHON, [PYTHON, "-c", "from local_operator.cli import main; main()"], env)
        os._exit(1)

    try:
        import fcntl
        import struct
        import termios

        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))

        print("booting the real binary under a pty…")
        boot = read_for(fd, 18.0)
        print(f"  received {len(boot)} bytes of screen output")

        # Type the sample into the composer.
        os.write(fd, SAMPLE.encode())
        read_for(fd, 3.0)

        def screen_has(text: str, blob: bytes) -> bool:
            return text.encode() in ANSI.sub(b"", blob)

        after_type = read_for(fd, 1.0)
        print(f"  composer holds the sample: {screen_has('alpha', after_type + boot)}")

        # The Esc-prefixed option+left chord, twice — the encoding that used to
        # abort the turn. Then a printable character, which lands wherever the
        # caret actually is. If word movement worked, it lands before "gamma".
        for _ in range(2):
            os.write(fd, b"\x1b\x1b[D")
            time.sleep(0.25)
        read_for(fd, 1.0)
        os.write(fd, b"X")
        time.sleep(0.5)
        final = read_for(fd, 3.0)

        clean = ANSI.sub(b"", final).decode("utf-8", "replace")
        moved = "alpha beta Xgamma delta"
        one_char = "alpha beta gamXma delta"
        print()
        print("  looking for the marker in the rendered screen:")
        print(f"    word-wise movement  {moved!r}: {moved in clean}")
        print(f"    char-wise movement  {one_char!r}: {one_char in clean}")
        if moved in clean:
            print("  RESULT: two chords moved the caret TWO WORDS left. Correct.")
            return 0
        print("  RESULT: unexpected caret position")
        print(clean[-1500:])
        return 1
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())

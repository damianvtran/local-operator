"""Raw-mode PTY probe: enable bracketed paste, dump every byte stdin delivers.

Independent re-verification of the FINDINGS.md premise. Writes the captured
bytes to the path given as argv[1] so each terminal/clipboard/key combination
gets its own artifact.
"""

import os
import sys
import termios
import time
import tty

out = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
buf = b""
try:
    tty.setraw(fd)
    os.write(1, b"\x1b[?2004h")
    os.write(1, b"PROBE READY\r\n")
    os.set_blocking(fd, False)
    end = time.time() + seconds
    while time.time() < end:
        try:
            chunk = os.read(fd, 8192)
            if chunk:
                buf += chunk
        except BlockingIOError:
            pass
        time.sleep(0.02)
    os.write(1, b"\x1b[?2004l")
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
with open(out, "wb") as handle:
    handle.write(buf)
os.write(1, ("\r\nCAPTURED %d bytes: %r\r\n" % (len(buf), buf[:200])).encode())

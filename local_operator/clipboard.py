"""Reading an image (or a file URL) off the SYSTEM clipboard.

The composer needs this because a terminal cannot give it to us. Textual's
``Paste`` event carries text and nothing else — there is no binary channel in
the terminal protocol — so an image on the pasteboard reaches the app as an
EMPTY bracketed paste and disappears (issue #372). ``Cmd+V`` after a native
macOS screenshot (``Cmd+Shift+Ctrl+4``) was therefore a dead keystroke: the
pasteboard held 20 KB of PNG, the terminal handed over ``""``, and the composer
had no way to ask for the bytes. Reading the clipboard OURSELVES is the only
terminal-independent route to them, which is why this module exists at all.

The gap stayed invisible for a long time because it does not reproduce in the
one place the code was developed. **cmux** watches the pasteboard and writes an
image to ``$TMPDIR/clipboard-<stamp>-<hash>.png``, then bracket-pastes that
filename — so inside cmux the composer's path-only ingestion sees a real path
and works perfectly. Ghostty, Terminal.app, iTerm2 and every other emulator
paste text only, so outside cmux the same gesture produced nothing. A design
that is correct for one terminal's helper is not a clipboard implementation,
and the docstrings in ``tui/widgets/editor.py`` used to credit Ghostty with
cmux's behaviour.

**All four platforms are peers here.** There is one dispatch
(:func:`read_clipboard_image`) that picks a backend from ``sys.platform`` and
the session's environment; each backend is an independent function with the
same contract, and none of them is a fast path the others hang off. That
matters for more than tidiness: the failure this module fixes was itself the
result of a single-environment assumption baked into the ingest path.

Every backend obeys the same four rules, which is what makes them substitutable:

1. **Return ``None``, never raise.** A missing ``xclip``, a Wayland compositor
   with no clipboard, a locked-down PowerShell — all of them mean "no image on
   the clipboard", which is the same answer as an empty clipboard. This runs on
   a keystroke: the user pressed ``Cmd+V``, and an exception (or a stderr line
   about a missing binary) on every stray empty paste would be worse than the
   silence it replaced.
2. **Bounded by :data:`CLIPBOARD_TIMEOUT_S`.** Each backend shells out, and a
   wedged clipboard daemon (a hung ``wl-paste``, an X11 selection owner that
   never answers, a stalled AppleScript) would otherwise hold the process
   forever. Two seconds matches the cap the reference implementation uses for
   the same subprocess reads.
3. **Bounded by ``max_bytes`` BEFORE the payload is handed back.** The
   clipboard is an untrusted-size source in exactly the way a pasted file path
   is, so the same ceiling applies, and it is applied to the bytes as they are
   captured rather than after a decode.
4. **Silent on tooling absence.** ``shutil.which`` gates every backend, so a
   Linux box without ``xclip`` installed simply has no clipboard images.

**Never over SSH.** :func:`clipboard_reads_are_local` refuses every read when
the session looks remote. This is a confidentiality rule, not an accuracy one:
the process is on the server, so its clipboard is the SERVER's, and quietly
attaching the server's clipboard contents to a prompt the user typed from their
laptop would exfiltrate something they never chose to send. Refusing looks
identical to an empty clipboard, which is the honest outcome — the user's real
clipboard is genuinely unreachable from here.

The reads are blocking by design and callers put them on a thread
(``asyncio.to_thread``); this module deliberately holds no event-loop
machinery, so it stays testable as plain functions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from local_operator.media import sniff_image

#: The budget for ONE clipboard read operation, across every subprocess it
#: takes. A clipboard daemon that never answers is what this bounds: the read
#: runs on the keystroke that pasted, so an unbounded subprocess is a
#: permanently frozen composer rather than a slow one.
#:
#: A WHOLE-OPERATION deadline and not a per-process one, which is the
#: correction from review round 1 (F2). A per-`_run` cap looks equivalent and
#: is not: `_read_x11_image` tries four MIME types in sequence and the composer
#: adds a second file-URL read, so four hung `xclip` calls at 2 s each measured
#: **8.0 s** of dead composer against a docstring promising "one visible
#: pause". `_Deadline` below hands each `_run` only the time left, so the total
#: is what the constant says regardless of how many calls a backend makes.
#:
#: Two seconds is generous for a local IPC read (the macOS backend measures
#: ~200 ms for a 20 KB PNG including AppleScript startup, and ~0.6 s for an
#: 8 MB Retina screenshot) and short enough that a wedged daemon costs one
#: visible pause.
CLIPBOARD_TIMEOUT_S = 2.0

#: The INGEST ceiling: how many bytes may be pulled off the clipboard at all.
#:
#: Deliberately far above the composer's ``MAX_ATTACHMENT_BYTES`` (4 MB),
#: because the two bounds protect against different things and conflating them
#: broke the exact gesture this module exists for (review round 1, U1). The
#: ATTACHMENT budget governs what may reach a provider, and it is applied after
#: ``bound_image_for_model`` has resized the image. The INGEST budget only has
#: to stop a runaway read from a hostile or broken clipboard owner — the bytes
#: are transient and are about to be shrunk.
#:
#: Measured, with the real ``screencapture -c`` that ``Cmd+Shift+Ctrl+4``
#: invokes, on a 3456x2234 Retina display: the pasteboard PNG is 8.4-8.5 MB and
#: bounds down to 0.28 MB at 1568x1014, fourteen times under the attachment
#: cap. Handing the 4 MB attachment cap to the READ therefore discarded every
#: full-screen screenshot before the resize that makes it attachable could run,
#: and reported "no image on the clipboard" for a clipboard that plainly had
#: one. 64 MB reads that screenshot in ~0.6 s and still refuses a payload no
#: screen capture could produce.
MAX_CLIPBOARD_READ_BYTES = 64 * 1024 * 1024

#: Environment variables that mean "this process is on the far end of an SSH
#: connection". Any ONE of them is enough; they are set by different sshd
#: versions and configurations, and a session with only ``SSH_CLIENT`` set is
#: exactly as remote as one with all three.
SSH_ENV_VARS = ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT")

#: MIME types worth pulling off an X11/Wayland clipboard, in preference order.
#: PNG first because it is lossless and what a screenshot tool puts there;
#: JPEG/GIF/WebP follow so a copy out of a browser still attaches. The list is
#: intentionally the set ``media.SUPPORTED_IMAGE_MIME_TYPES`` can send — asking
#: a compositor for a type no provider accepts only converts a failed paste
#: into a failed request.
IMAGE_MIME_PREFERENCE = ("image/png", "image/jpeg", "image/gif", "image/webp")


@dataclass(frozen=True)
class ClipboardImage:
    """Image bytes read off the clipboard, with the MIME type they really are.

    The MIME type is the one :func:`_as_image` SNIFFED, not the one the backend
    asked for. Those differ in practice: ``xclip`` answers a request for
    ``image/png`` against a text-only clipboard by returning the text with a
    zero exit status, so a backend that trusted its own request would hand back
    ``ClipboardImage(b'some text', 'image/png')``. That was observed against
    real ``xclip`` under Xvfb, and no mocked test would have shown it.
    """

    data: bytes
    mime_type: str


def _as_image(data: bytes | None, max_bytes: int) -> ClipboardImage | None:
    """Accept ``data`` only if the BYTES are an image within ``max_bytes``.

    The single gate every backend returns through, and it exists because a
    clipboard tool's answer cannot be taken at face value. ``xclip -t image/png
    -o`` on a clipboard holding plain text exits 0 and prints the TEXT: the
    target request is advisory, and X11's selection owner is free to answer
    with whatever it has. Under Xvfb this produced a cheerful
    ``('image/png', 19)`` for the string ``just text, no image``, which would
    have travelled all the way to a provider as a corrupt image block.

    Sniffing the header answers it for every backend at once rather than
    special-casing the one that was caught doing it, and it also settles the
    size bound in the same place — the clipboard is an untrusted-size source,
    so the ceiling belongs where the bytes are accepted.

    """
    if not data or len(data) > max_bytes:
        return None
    info = sniff_image(data)
    # `sendable` and not merely "recognised": a HEIC on the pasteboard sniffs
    # fine and no provider accepts it, and the composer would rather report
    # "no image" than attach a block that earns a 400 mid-turn.
    if info is None or not info.sendable:
        return None
    return ClipboardImage(data, info.mime_type)


def clipboard_reads_are_local(env: Mapping[str, str] | None = None) -> bool:
    """Is the clipboard we would read the one the USER is looking at?

    False over SSH. The check is deliberately conservative — presence of any
    SSH variable disqualifies the read — because the cost of the two answers is
    wildly asymmetric. A false negative means a remote session cannot attach
    screenshots, which the user can work around by pasting a path. A false
    positive silently attaches the SERVER's clipboard to a prompt, which is a
    confidentiality failure the user has no way to notice: the marker says
    ``[Image #1, 800x600]`` either way.
    """
    source = os.environ if env is None else env
    return not any(source.get(name) for name in SSH_ENV_VARS)


class _Deadline:
    """The remaining budget for one clipboard operation, shared by its calls.

    Threaded through the backends so a multi-call read costs what
    :data:`CLIPBOARD_TIMEOUT_S` says in total, rather than that much per
    subprocess (review round 1, F2). Constructed once per public entry point
    and passed down; a backend never gets to decide its own budget.

    ``expired`` is checked BEFORE each spawn so an exhausted deadline costs no
    further processes at all, rather than three more that are each handed a
    zero timeout and killed.
    """

    def __init__(self, seconds: float) -> None:
        self._end = time.monotonic() + seconds

    @property
    def remaining(self) -> float:
        return self._end - time.monotonic()

    @property
    def expired(self) -> bool:
        # A read cannot usefully be given a zero or negative timeout, so the
        # floor is what "no time left" means rather than a bare `<= 0`.
        return self.remaining <= _MIN_SPAWN_BUDGET_S


#: Below this much time left, a further subprocess is not worth spawning: the
#: interpreters here (``osascript``, ``pwsh``) cost more than this just to
#: start, so a spawn under it can only end in a kill.
_MIN_SPAWN_BUDGET_S = 0.05


def _run(
    argv: list[str],
    deadline: _Deadline,
    *,
    stdin_text: str | None = None,
    max_bytes: int | None = None,
) -> bytes | None:
    """Run ``argv`` within ``deadline`` and return stdout, or ``None``.

    The single subprocess seam every backend goes through, so the timeout, the
    byte ceiling, the binary stdout and the never-raise contract are decided
    once instead of four times. ``stderr`` is swallowed rather than merged:
    ``xclip`` writes "Error: target image/png not available" to it for the
    ordinary case of a text-only clipboard, and that is not news the user needs
    on a keystroke.

    A non-zero exit is ``None`` and not an error for the same reason — every
    one of these tools reports "nothing of that type on the clipboard" as a
    failed exit.

    ``stdin_text`` feeds a script to an interpreter reading from stdin, which
    is how the AppleScript backends are invoked. It keeps the script out of the
    argument vector, where it would be visible in every ``ps`` listing.

    ``max_bytes`` STOPS THE READ rather than judging it afterwards, which is
    why this cannot use ``subprocess.run``: that reads the pipe to EOF before
    returning, so a ceiling applied to its result is a verdict on memory
    already spent. The pipe backends let the SELECTION OWNER choose the payload
    size, and round 1 (F3) measured 300 MB buffered and 750 MB of peak RSS
    against a 4 MB cap. Reading ``max_bytes + 1`` and refusing a longer stream
    is the same stat-first discipline the composer's path branch documents.

    The deadline is REQUIRED rather than defaulted, so a future backend cannot
    quietly opt out of the total bound by omitting it.
    """
    if deadline.expired:
        return None
    limit = None if max_bytes is None else max_bytes + 1
    try:
        with subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        ) as process:
            try:
                if stdin_text is not None and process.stdin is not None:
                    # Written before the bounded read so an interpreter waiting
                    # on its script is not deadlocked against a reader waiting
                    # on its output. These scripts are a few hundred bytes,
                    # comfortably inside the pipe buffer.
                    process.stdin.write(stdin_text.encode("utf-8"))
                    process.stdin.close()
                if process.stdout is None:
                    stdout = b""
                elif limit is None:
                    stdout = process.stdout.read()
                else:
                    stdout = process.stdout.read(limit)
                # The wait is what turns a bounded read into a bounded RUN: a
                # tool that answers instantly and then hangs would otherwise
                # escape the deadline entirely.
                returncode = process.wait(timeout=max(deadline.remaining, 0.0))
            finally:
                # `Popen.__exit__` closes the pipes but only reaps a process
                # that has already exited, so a timed-out child is killed here
                # rather than left orphaned holding the selection.
                if process.poll() is None:
                    process.kill()
    except (OSError, subprocess.SubprocessError, ValueError):
        # OSError covers the binary vanishing between `which` and `exec`;
        # SubprocessError covers the timeout; ValueError covers a pipe closed
        # under us. All of them mean "no clipboard image".
        return None
    if returncode != 0:
        return None
    if limit is not None and len(stdout) >= limit:
        # Longer than the ceiling allows. Dropped rather than truncated: a
        # truncated PNG still sniffs as one and would be attached as a corrupt
        # image block.
        return None
    return stdout


#: AppleScript is the macOS backend because it needs no third-party binary and
#: no Python extension. ``pngpaste`` would be a Homebrew dependency the user
#: does not have, and PyObjC is a large compiled wheel for one read — while
#: ``osascript`` is present on every macOS install and reaches the same
#: ``NSPasteboard`` API through ``use framework "AppKit"``.
#:
#: TIFF is handled explicitly because it is not a rare case: several macOS apps
#: (Preview's copy, some screenshot utilities) put ONLY ``public.tiff`` on the
#: pasteboard, and no provider accepts TIFF. ``NSBitmapImageRep`` re-encodes it
#: to PNG in-process, which is cheaper and more reliable than declining.
#: ``representationUsingType:4`` is ``NSBitmapImageFileTypePNG``; the numeric
#: form is used because the symbolic constant is not visible to AppleScript's
#: ObjC bridge.
_MACOS_IMAGE_SCRIPT = """\
use framework "AppKit"
use framework "Foundation"
use scripting additions

on run argv
\tset dest to item 1 of argv
\tset pb to current application's NSPasteboard's generalPasteboard()
\tset png to pb's dataForType:"public.png"
\tif png is missing value then
\t\tset tiff to pb's dataForType:"public.tiff"
\t\tif tiff is missing value then return "none"
\t\tset rep to current application's NSBitmapImageRep's imageRepWithData:tiff
\t\tif rep is missing value then return "none"
\t\tset props to current application's NSDictionary's dictionary()
\t\tset png to rep's representationUsingType:4 |properties|:props
\t\tif png is missing value then return "none"
\tend if
\tpng's writeToFile:dest atomically:true
\treturn "ok"
end run
"""

#: The Finder ``Cmd+C`` case. Finder puts only a ``public.file-url`` flavor on
#: the pasteboard — no plain text and no image bytes — so ``pbpaste`` returns
#: zero bytes and the image backend above finds nothing either. Reading the
#: URLs turns that gesture back into the path list the composer already knows
#: how to attach.
#:
#: Enumerated by INDEX rather than with ``repeat with u in``, which hands back
#: AppleScript's own coerced items instead of the ``NSURL`` objects and fails
#: with "doesn't understand the isFileURL message".
_MACOS_FILE_URL_SCRIPT = """\
use framework "AppKit"
use framework "Foundation"
use scripting additions

on run argv
\tset pb to current application's NSPasteboard's generalPasteboard()
\tset urls to pb's readObjectsForClasses:{current application's NSURL} options:(missing value)
\tif urls is missing value then return ""
\tset out to ""
\tset sep to (ASCII character 0)
\trepeat with i from 0 to ((urls's |count|() as integer) - 1)
\t\tset u to (urls's objectAtIndex:i)
\t\tif (u's isFileURL()) as boolean then set out to out & ((u's |path|()) as text) & sep
\tend repeat
\treturn out
end run
"""


def _read_macos_image(max_bytes: int, deadline: _Deadline) -> ClipboardImage | None:
    """macOS: the ``public.png`` flavor, or ``public.tiff`` re-encoded to PNG.

    Writes to a temp FILE rather than returning bytes on stdout. ``osascript``
    prints an AppleScript result through a text coercion, so raw image bytes on
    stdout are mangled by encoding before this process ever sees them — the
    same trap the Windows backend documents. A file is the only lossless
    channel out of ``osascript``, and it is deleted before this returns.
    """
    if not shutil.which("osascript"):
        # Not reachable on a stock macOS, but this module must never assume a
        # binary exists just because the platform usually ships it.
        return None
    with tempfile.TemporaryDirectory(prefix="lo-clip-") as tmp:
        dest = Path(tmp) / "clipboard.png"
        # `-` reads the script from stdin; everything after it is `argv` to the
        # script's `on run` handler, so the destination never has to be spliced
        # into the source text.
        stdout = _run(
            ["osascript", "-", str(dest)],
            deadline,
            stdin_text=_MACOS_IMAGE_SCRIPT,
        )
        if stdout is None:
            return None
        data = _read_bounded(dest, max_bytes)
    return _as_image(data, max_bytes)


def _read_bounded(path: Path, max_bytes: int) -> bytes | None:
    """Read ``path`` only if it is a regular file within ``max_bytes``.

    Stat before read, the same order the composer's path branch uses and for
    the same measured reason: checking the size after reading pays the cost the
    cap exists to prevent, and the payload here is whatever the clipboard held.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    if not stat.st_size or stat.st_size > max_bytes:
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _read_macos_file_urls(deadline: _Deadline) -> list[str]:
    """macOS: file paths from the pasteboard's ``public.file-url`` flavor.

    Split on ``\\x00`` rather than on newlines. A newline is legal in a macOS
    filename, and splitting on it turned one real path into two nonexistent
    ones — which the composer's all-or-nothing rule then correctly refused, so
    a Finder copy of such a file silently attached nothing (review round 1,
    F5). NUL cannot occur in a path, so it is the one unambiguous separator.
    """
    if not shutil.which("osascript"):
        return []
    stdout = _run(["osascript", "-"], deadline, stdin_text=_MACOS_FILE_URL_SCRIPT)
    if stdout is None:
        return []
    text = stdout.decode("utf-8", errors="replace")
    return [path for path in text.split("\x00") if path.strip()]


def _read_wayland_image(max_bytes: int, deadline: _Deadline) -> ClipboardImage | None:
    """Wayland: ``wl-paste --list-types`` then ``wl-paste --type <mime>``.

    The type list is queried first because ``wl-paste --type image/png`` on a
    clipboard that has no PNG both fails AND, on some compositors, blocks while
    the offer is negotiated. Asking what is on offer costs one cheap call and
    turns four speculative reads into at most one real one.

    Then it tries every offered type in preference order, the same as X11.
    Round 1 (F4) caught these two loops disagreeing: this one used to stop
    after the first offered type on the theory that a second candidate is "the
    same image again", which is a guess about the clipboard owner rather than a
    fact — a compositor may well offer a huge PNG and a small JPEG of
    different pictures. Both loops now do the same thing, and the total cost is
    bounded by the shared deadline instead of by loop length.
    """
    if not shutil.which("wl-paste"):
        return None
    listing = _run(["wl-paste", "--list-types"], deadline)
    if listing is None:
        return None
    offered = {line.strip() for line in listing.decode("utf-8", "replace").splitlines()}
    for mime in IMAGE_MIME_PREFERENCE:
        if mime not in offered:
            continue
        image = _as_image(
            _run(
                ["wl-paste", "--no-newline", "--type", mime],
                deadline,
                max_bytes=max_bytes,
            ),
            max_bytes,
        )
        if image is not None:
            return image
    return None


def _read_x11_image(max_bytes: int, deadline: _Deadline) -> ClipboardImage | None:
    """X11: ``xclip -selection clipboard -t <mime> -o``.

    Each type is attempted directly. X11 has ``TARGETS``, but querying it costs
    a round trip per read and ``xclip`` already exits non-zero within
    milliseconds for a type the selection owner does not offer, so the
    speculative reads are cheaper than the negotiation Wayland needs.

    The loop is what made F2's worst case the worst: four hung selection owners
    used to cost four full timeouts. The shared deadline now caps all four
    together, and `_run` refuses to spawn once it is exhausted.
    """
    if not shutil.which("xclip"):
        return None
    for mime in IMAGE_MIME_PREFERENCE:
        image = _as_image(
            _run(
                ["xclip", "-selection", "clipboard", "-t", mime, "-o"],
                deadline,
                max_bytes=max_bytes,
            ),
            max_bytes,
        )
        if image is not None:
            return image
    return None


#: PowerShell writes the clipboard image to a FILE, and the file path is the
#: whole point of the design. ``Get-Clipboard -Format Image`` yields a
#: ``System.Drawing.Bitmap`` object, not bytes, so something has to encode it;
#: and piping the encoded bytes to stdout corrupts them, because PowerShell's
#: stdout is a TEXT stream that applies an output encoding to whatever crosses
#: it. That corruption is silent and produces a payload that sniffs as PNG for
#: its first eight bytes and then fails to decode.
#:
#: ``[IO.File]::WriteAllBytes`` is used rather than ``Set-Content -Encoding
#: Byte`` because the two PowerShell generations disagree about that parameter:
#: Windows PowerShell 5.1 spells it ``-Encoding Byte``, and PowerShell 7+
#: removed that value in favour of ``-AsByteStream``. The .NET call is
#: identical on both, which is what makes one script serve both.
#:
#: ``System.Drawing`` is loaded explicitly: it is auto-loaded in 5.1 but not in
#: 7+, where the assembly must be requested by name.
#:
#: NOTE: unit-tested against mocked invocations only. There is no Windows host
#: in this project's development or CI environment, so this command has been
#: reviewed rather than executed.
#: Both objects are disposed in a ``finally`` rather than on the success path.
#: Round 1 (F6) caught the leak: ``WriteAllBytes`` throwing (a full disk, a
#: permission fault) skipped straight to the ``catch``, so the bitmap and the
#: stream were never released — and this is a native GDI+ handle, not managed
#: memory the collector will shortly reclaim.
_WINDOWS_SCRIPT = """\
$ErrorActionPreference = 'Stop'
$img = $null
$stream = $null
try {
  Add-Type -AssemblyName System.Windows.Forms, System.Drawing | Out-Null
  $img = [Windows.Forms.Clipboard]::GetImage()
  if ($null -eq $img) { exit 1 }
  $stream = New-Object System.IO.MemoryStream
  $img.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
  [IO.File]::WriteAllBytes($args[0], $stream.ToArray())
} catch {
  exit 1
} finally {
  if ($null -ne $stream) { $stream.Dispose() }
  if ($null -ne $img) { $img.Dispose() }
}
"""


def _windows_shell() -> str | None:
    """``pwsh`` if present, else Windows PowerShell, else nothing.

    PowerShell 7+ is preferred where it exists because it starts faster, which
    matters inside a two-second cap on a keystroke. ``powershell.exe`` is the
    fallback that is always present on a supported Windows.
    """
    return shutil.which("pwsh") or shutil.which("powershell")


def _read_windows_image(max_bytes: int, deadline: _Deadline) -> ClipboardImage | None:
    """Windows: PowerShell reads the clipboard bitmap and PNG-encodes it.

    Via a temp file for the encoding reason documented on
    :data:`_WINDOWS_SCRIPT`: binary on PowerShell's stdout is corrupted by the
    output encoding, and no combination of flags makes that stream safe for
    image bytes.
    """
    shell = _windows_shell()
    if shell is None:
        return None
    with tempfile.TemporaryDirectory(prefix="lo-clip-") as tmp:
        dest = Path(tmp) / "clipboard.png"
        script = Path(tmp) / "read_clipboard.ps1"
        try:
            script.write_text(_WINDOWS_SCRIPT, encoding="utf-8")
        except OSError:
            return None
        stdout = _run(
            [
                shell,
                "-NoProfile",
                "-NonInteractive",
                # STA is required: the Windows clipboard API is single-threaded
                # apartment only, and `Clipboard::GetImage` throws outright from
                # the MTA that `pwsh` uses by default.
                "-STA",
                # `-File`, not `-Command`: only `-File` binds trailing tokens to
                # the script's `$args`, and it also keeps the destination path
                # out of a string PowerShell would parse as source.
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                str(dest),
            ],
            deadline,
        )
        if stdout is None:
            return None
        data = _read_bounded(dest, max_bytes)
    return _as_image(data, max_bytes)


def read_clipboard_image(
    max_bytes: int,
    *,
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ClipboardImage | None:
    """The image on the system clipboard, or ``None``.

    ``None`` covers every "there is nothing to attach" case without
    distinguishing them, because no backend can reliably tell them apart: an
    empty clipboard, a text-only clipboard, a missing ``xclip``, a wedged
    daemon and an unreadable payload are one answer here. The two cases the
    CALLER can distinguish are deliberately not collapsed into this function —
    a remote session is knowable up front via
    :func:`clipboard_reads_are_local`, and an oversized image is knowable from
    the bytes it returns — because those are the two the user can act on
    (review round 1, D2/U2).

    ``max_bytes`` is the INGEST ceiling, not the attachment budget; see
    :data:`MAX_CLIPBOARD_READ_BYTES` for why passing the smaller of the two
    here defeats the resize that makes a screenshot attachable.

    ``platform`` and ``env`` are injectable so each backend is testable on any
    host — this module's whole failure mode is a platform assumption that only
    one developer's environment could disprove.

    Wayland is chosen over X11 by ``WAYLAND_DISPLAY`` rather than by
    distribution: a Wayland session commonly also runs XWayland, so ``DISPLAY``
    is set in both, and testing ``DISPLAY`` first would route a Wayland session
    to ``xclip`` and read XWayland's separate, usually empty selection.
    """
    if not clipboard_reads_are_local(env):
        return None
    # ONE deadline for the whole operation, created here rather than inside a
    # backend so a multi-call read cannot outlive the documented cap (F2).
    deadline = _Deadline(CLIPBOARD_TIMEOUT_S)
    system = sys.platform if platform is None else platform
    source = os.environ if env is None else env
    if system == "darwin":
        return _read_macos_image(max_bytes, deadline)
    if system == "win32":
        return _read_windows_image(max_bytes, deadline)
    if system.startswith("linux") or "bsd" in system:
        # BSDs run the same X11/Wayland stacks, so they take the same backends
        # rather than falling through to "no clipboard".
        if source.get("WAYLAND_DISPLAY"):
            return _read_wayland_image(max_bytes, deadline)
        if source.get("DISPLAY"):
            return _read_x11_image(max_bytes, deadline)
        # A headless Linux box (a container, a bare tty) has no clipboard at
        # all, and shelling out to discover that costs a subprocess per paste.
        return None
    return None


def read_clipboard_file_paths(
    *,
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """File paths on the clipboard, for the Finder ``Cmd+C`` case.

    macOS only, and deliberately so. This exists for one specific pasteboard
    shape — ``public.file-url`` with no text and no image flavor — which is
    what Finder's copy produces. Linux file managers put ``text/uri-list`` on
    the clipboard alongside plain text of the same paths, so the terminal
    bracket-pastes those paths and the composer's existing path branch already
    handles them; adding a second route there would be a way for one drop to be
    attached twice.
    """
    if not clipboard_reads_are_local(env):
        return []
    system = sys.platform if platform is None else platform
    if system != "darwin":
        return []
    return _read_macos_file_urls(_Deadline(CLIPBOARD_TIMEOUT_S))


@dataclass(frozen=True)
class ClipboardContents:
    """What one look at the clipboard found, and whether it was even allowed.

    A single result for a single gesture. The composer used to ask two separate
    questions (image, then file URLs), which gave the pair two independent
    deadlines and a 4 s worst case against a constant that says 2 (review round
    1, F2). Both are answered here under one budget.

    ``refused_remote`` is carried because it is the one state this module knows
    with certainty and the user can act on: over SSH the read never happens, so
    reporting "no image on the clipboard" would be describing a clipboard
    nobody looked at (review round 1, D2/U2). "An image was found but could not
    be attached" is the OTHER distinguishable case, and it deliberately lives
    with the caller, which is where the attachment budget and the resize are;
    this type only reports what was on the clipboard.

    Everything else stays collapsed: an empty clipboard, a text-only one, a
    missing ``xclip`` and a wedged daemon are one answer, because a message
    that guessed between them would be inventing a diagnosis.
    """

    image: ClipboardImage | None = None
    paths: tuple[str, ...] = ()
    #: The read never happened: this session is remote, so the clipboard would
    #: be the server's. Not a failure to find an image, and must not be
    #: reported as one.
    refused_remote: bool = False

    @property
    def found_nothing(self) -> bool:
        """Nothing to attach, and not because the read was refused."""
        return self.image is None and not self.paths and not self.refused_remote


def read_clipboard(
    max_bytes: int = MAX_CLIPBOARD_READ_BYTES,
    *,
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ClipboardContents:
    """One look at the clipboard for one paste, under one deadline.

    The entry point the composer uses. It exists so that the image read and the
    file-URL read share a single :data:`CLIPBOARD_TIMEOUT_S` budget rather than
    getting one each, and so the caller learns the two things it can act on
    without this module having to explain itself per backend.

    ``max_bytes`` defaults to the INGEST ceiling
    (:data:`MAX_CLIPBOARD_READ_BYTES`) rather than to any attachment budget:
    resizing happens downstream, and a ceiling applied before it discards
    images that would have been perfectly attachable (U1).
    """
    if not clipboard_reads_are_local(env):
        return ClipboardContents(refused_remote=True)

    deadline = _Deadline(CLIPBOARD_TIMEOUT_S)
    system = sys.platform if platform is None else platform
    source = os.environ if env is None else env

    image: ClipboardImage | None = None
    if system == "darwin":
        image = _read_macos_image(max_bytes, deadline)
    elif system == "win32":
        image = _read_windows_image(max_bytes, deadline)
    elif system.startswith("linux") or "bsd" in system:
        if source.get("WAYLAND_DISPLAY"):
            image = _read_wayland_image(max_bytes, deadline)
        elif source.get("DISPLAY"):
            image = _read_x11_image(max_bytes, deadline)
    if image is not None:
        return ClipboardContents(image=image)

    # Only macOS has a second shape to try; every other platform's file manager
    # puts the paths on the clipboard as text, which the terminal already
    # bracket-pastes into the composer's existing path branch.
    paths: tuple[str, ...] = ()
    if system == "darwin":
        paths = tuple(_read_macos_file_urls(deadline))
    return ClipboardContents(paths=paths)

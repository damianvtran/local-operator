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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from local_operator.media import sniff_image

#: Every clipboard read is capped here. A clipboard daemon that never answers
#: is the failure this bounds: the read runs on the keystroke that pasted, so
#: an unbounded subprocess is a permanently frozen composer rather than a slow
#: one. Two seconds is generous for a local IPC read (the macOS backend
#: measures ~200 ms for a 20 KB PNG, including AppleScript startup) and short
#: enough that a wedged daemon costs one visible pause and nothing more.
CLIPBOARD_TIMEOUT_S = 2.0

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


def _run(
    argv: list[str],
    *,
    timeout: float = CLIPBOARD_TIMEOUT_S,
    stdin_text: str | None = None,
) -> bytes | None:
    """Run ``argv`` and return its stdout bytes, or ``None`` for any failure.

    The single subprocess seam every backend goes through, so the timeout, the
    binary stdout, and the never-raise contract are decided once instead of
    four times. ``stderr`` is swallowed rather than merged: ``xclip`` writes
    "Error: target image/png not available" to it for the ordinary case of a
    text-only clipboard, and that is not news the user needs on a keystroke.

    A non-zero exit is ``None`` and not an error for the same reason — every
    one of these tools reports "nothing of that type on the clipboard" as a
    failed exit.

    ``stdin_text`` feeds a script to an interpreter reading from stdin, which
    is how the AppleScript backends are invoked. It keeps the script out of the
    argument vector, where it would be visible in every ``ps`` listing.
    """
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            input=None if stdin_text is None else stdin_text.encode("utf-8"),
            stdin=None if stdin_text is not None else subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # OSError covers the binary vanishing between `which` and `run`;
        # SubprocessError covers the timeout. Both mean "no clipboard image".
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


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
\trepeat with i from 0 to ((urls's |count|() as integer) - 1)
\t\tset u to (urls's objectAtIndex:i)
\t\tif (u's isFileURL()) as boolean then set out to out & ((u's |path|()) as text) & linefeed
\tend repeat
\treturn out
end run
"""


def _read_macos_image(max_bytes: int) -> ClipboardImage | None:
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


def _read_macos_file_urls() -> list[str]:
    """macOS: file paths from the pasteboard's ``public.file-url`` flavor."""
    if not shutil.which("osascript"):
        return []
    stdout = _run(["osascript", "-"], stdin_text=_MACOS_FILE_URL_SCRIPT)
    if stdout is None:
        return []
    text = stdout.decode("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _read_wayland_image(max_bytes: int) -> ClipboardImage | None:
    """Wayland: ``wl-paste --list-types`` then ``wl-paste --type <mime>``.

    The type list is queried first because ``wl-paste --type image/png`` on a
    clipboard that has no PNG both fails AND, on some compositors, blocks while
    the offer is negotiated. Asking what is on offer costs one cheap call and
    turns four speculative reads into at most one real one.
    """
    if not shutil.which("wl-paste"):
        return None
    listing = _run(["wl-paste", "--list-types"])
    if listing is None:
        return None
    offered = {line.strip() for line in listing.decode("utf-8", "replace").splitlines()}
    for mime in IMAGE_MIME_PREFERENCE:
        if mime not in offered:
            continue
        # A type that was offered but could not be read (or was oversized, or
        # did not sniff as what it claimed) is not a reason to try a worse
        # encoding of the same picture: the next candidate is the same image
        # again, past the same ceiling.
        return _as_image(_run(["wl-paste", "--no-newline", "--type", mime]), max_bytes)
    return None


def _read_x11_image(max_bytes: int) -> ClipboardImage | None:
    """X11: ``xclip -selection clipboard -t <mime> -o``.

    Each type is attempted directly. X11 has ``TARGETS``, but querying it costs
    a round trip per read and ``xclip`` already exits non-zero within
    milliseconds for a type the selection owner does not offer, so the
    speculative reads are cheaper than the negotiation Wayland needs.
    """
    if not shutil.which("xclip"):
        return None
    for mime in IMAGE_MIME_PREFERENCE:
        image = _as_image(_run(["xclip", "-selection", "clipboard", "-t", mime, "-o"]), max_bytes)
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
_WINDOWS_SCRIPT = """\
$ErrorActionPreference = 'Stop'
try {
  Add-Type -AssemblyName System.Windows.Forms, System.Drawing | Out-Null
  $img = [Windows.Forms.Clipboard]::GetImage()
  if ($null -eq $img) { exit 1 }
  $stream = New-Object System.IO.MemoryStream
  $img.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
  [IO.File]::WriteAllBytes($args[0], $stream.ToArray())
  $stream.Dispose()
  $img.Dispose()
} catch { exit 1 }
"""


def _windows_shell() -> str | None:
    """``pwsh`` if present, else Windows PowerShell, else nothing.

    PowerShell 7+ is preferred where it exists because it starts faster, which
    matters inside a two-second cap on a keystroke. ``powershell.exe`` is the
    fallback that is always present on a supported Windows.
    """
    return shutil.which("pwsh") or shutil.which("powershell")


def _read_windows_image(max_bytes: int) -> ClipboardImage | None:
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
            ]
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
    distinguishing them, because the caller cannot act differently on any of
    them: an empty clipboard, a text-only clipboard, a missing ``xclip``, a
    remote session, a wedged daemon and an oversized payload all mean the same
    thing to a composer deciding whether a paste produced an attachment.

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
    system = sys.platform if platform is None else platform
    source = os.environ if env is None else env
    if system == "darwin":
        return _read_macos_image(max_bytes)
    if system == "win32":
        return _read_windows_image(max_bytes)
    if system.startswith("linux") or "bsd" in system:
        # BSDs run the same X11/Wayland stacks, so they take the same backends
        # rather than falling through to "no clipboard".
        if source.get("WAYLAND_DISPLAY"):
            return _read_wayland_image(max_bytes)
        if source.get("DISPLAY"):
            return _read_x11_image(max_bytes)
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
    return _read_macos_file_urls()

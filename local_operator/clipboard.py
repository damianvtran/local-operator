"""Reading an image, a file URL, or text off the SYSTEM clipboard.

The composer needs this because a terminal cannot give it to us. Textual's
``Paste`` event carries text and nothing else — there is no binary channel in
the terminal protocol — so an image on the pasteboard can never reach the app
as bytes (issue #372). ``Cmd+V`` after a native macOS screenshot
(``Cmd+Shift+Ctrl+4``) was therefore a dead keystroke. Reading the clipboard
OURSELVES is the only terminal-independent route to those bytes, which is why
this module exists at all.

**MEASURE WHAT THE TERMINAL ACTUALLY SENDS.** An earlier revision of this
module said the pasteboard image reached the app "as an EMPTY bracketed
paste", and the composer hung its whole fix off that claim. It is false, and
nobody had checked. Captured with a raw-mode PTY probe on macOS 25.6 with a
PNG on the pasteboard and bracketed paste enabled (``ESC[?2004h``):

=============  ==================  =========  =================================
terminal       clipboard           keystroke  bytes delivered on stdin
=============  ==================  =========  =================================
Terminal.app   text                Cmd+V      25 (``ESC[200~…ESC[201~``)
Terminal.app   PNG screenshot      Cmd+V      **0** — and the terminal beeps
Ghostty        PNG screenshot      Cmd+V      **0** — and the terminal beeps
Terminal.app   PNG screenshot      Ctrl+V     1 (``\\x16``)
Ghostty        PNG screenshot      Ctrl+V     1 (``\\x16``)
=============  ==================  =========  =================================

With an image-only pasteboard the terminal sends NOTHING AT ALL. No bytes
means no ``Paste`` event, so a handler keyed on an empty paste is unreachable
on exactly the terminals it was written for. ``Cmd+V`` cannot be hooked from
inside a TUI: the terminal swallows it and there is no protocol by which we
can ask for pasteboard bytes. ``Ctrl+V`` does arrive, on both terminals, which
is why :meth:`~local_operator.tui.widgets.editor.Editor.action_system_paste`
is bound to it and is the reachable caller of this module.

The gap stayed invisible for a long time because it does not reproduce in the
one place the code was developed. **cmux** watches the pasteboard and writes an
image to ``$TMPDIR/clipboard-<stamp>-<hash>.png``, then bracket-pastes that
filename — so inside cmux the composer's path-only ingestion sees a real path
and works perfectly. Ghostty, Terminal.app, iTerm2 and every other emulator
paste text only, so outside cmux the same gesture produced nothing. A design
that is correct for one terminal's helper is not a clipboard implementation.

TEXT is read alongside the two attachable shapes because ``Ctrl+V`` is now a
SYSTEM paste, and a system paste that silently dropped the ordinary case would
be a worse key than the one it replaced (Textual's own ``ctrl+v`` pastes
``App.clipboard``, an internal buffer the system clipboard never fills). The
caller decides what to do with each shape; this module only reports what was
there.

**All four platforms are peers here.** There is one dispatch
(:func:`read_clipboard`) that picks a backend from ``sys.platform`` and
the session's environment; each backend is an independent function with the
same contract, and none of them is a fast path the others hang off. That
matters for more than tidiness: the failure this module fixes was itself the
result of a single-environment assumption baked into the ingest path. The text
shape obeys the same rule — a ``Ctrl+V`` that pasted text on macOS and dropped
it on Linux would put that assumption straight back.

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
import signal
import subprocess
import sys
import tempfile
import threading
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


#: Below this much time left, a further subprocess is not worth spawning: the
#: interpreters here (``osascript``, ``pwsh``) cost more than this just to
#: start, so a spawn under it can only end in a kill.
_MIN_SPAWN_BUDGET_S = 0.05

#: How long to wait for a KILLED child to be reaped. A signalled process is
#: gone almost immediately; this only exists so a pathological one cannot make
#: the cleanup itself unbounded, which would reintroduce the bug the kill is
#: there to fix.
_REAP_TIMEOUT_S = 0.5

#: POSIX gives each spawned tool its own process group so the whole tree can be
#: signalled at once. Windows has no equivalent here (and no forking clipboard
#: helper either — the PowerShell backend is one process), so it takes the
#: plain kill.
_SUPPORTS_PROCESS_GROUPS = os.name == "posix"


def _kill_tree(process: "subprocess.Popen[bytes]", pgid: int | None) -> None:
    """Kill the tool AND anything it forked, so nothing holds the pipe open.

    Signalling the process GROUP is the part that matters: a wedged tool is
    typically a shell or an interpreter with a child doing the actual blocking,
    and killing only the parent leaves that child alive holding the inherited
    stdout, which keeps the reader thread blocked forever (round 2, F2 — the
    same freeze, one level down).

    ``pgid`` is REMEMBERED FROM SPAWN rather than looked up here, and that is
    load-bearing rather than tidy. Looking it up with ``os.getpgid(pid)`` works
    only while the leader is alive; once the direct child has exited and been
    reaped the lookup raises ``ProcessLookupError``, so the naive form silently
    degrades to "no kill" in exactly the case that needs one — a descendant
    still holding the pipe (round 3, F4). ``start_new_session`` makes the child
    its own group leader, so the group id equals its pid and stays signallable
    after the leader is gone.

    Best-effort by design: the group may already be empty, or the platform may
    not support groups. Neither may raise into a keystroke handler.
    """
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (OSError, AttributeError):
            # Already reaped, or the group vanished between the check and the
            # signal. The direct kill below is the remaining option.
            pass
    try:
        process.kill()
    except OSError:
        pass


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

    **The read happens on a THREAD, and the deadline abandons it.** This is the
    correction from round 2 (F2/U6), and it is the whole reason the obvious
    shape does not work here. Reading the pipe inline and *then* calling
    ``wait(timeout=...)`` cannot bound anything: ``read()`` blocks until EOF,
    and a wedged tool holds its stdout open forever, so the line that enforces
    the deadline is never reached. That regression measured 15 s on X11 and
    12 s on macOS against a 2 s budget, with the child orphaned because the
    cleanup never ran either — strictly worse than the 8 s it was meant to fix,
    and invisible to any test whose fake stdout cannot block.

    A reader thread is what satisfies both bounds at once, which neither
    ``run(timeout=)`` nor ``communicate(timeout=)`` does: they enforce the
    deadline but buffer without limit, reopening F3. Here the join is the time
    bound and the ``read(limit)`` is the byte bound. The child is killed on
    expiry, which is also what unblocks the thread (the pipe hits EOF), so the
    thread is a daemon purely as a backstop and never accumulates.

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
            # Its OWN process group, so the kill below reaches the whole tree.
            # These tools fork: `osascript` and a shell-wrapped `xclip` both
            # leave a grandchild holding the inherited stdout, and signalling
            # only the direct child leaves that grandchild alive and the pipe
            # open — the read stays blocked and the bound is defeated by the
            # same mechanism it was written to fix. Measured: killing just the
            # child still took 30 s on a 1 s budget.
            start_new_session=_SUPPORTS_PROCESS_GROUPS,
        ) as process:
            # Remembered NOW, while the leader is certainly alive. After it
            # exits, `os.getpgid(pid)` raises and the group becomes unkillable
            # by lookup — see `_kill_tree` (round 3, F4).
            pgid = process.pid if _SUPPORTS_PROCESS_GROUPS else None
            captured: list[bytes] = []

            def drain() -> None:
                # Never raises into the caller: this runs on its own thread, so
                # an exception here would be printed by the interpreter and
                # otherwise lost. A failed read is an empty capture, which the
                # returncode check below already treats as "no image".
                try:
                    stream = process.stdout
                    if stream is not None:
                        captured.append(stream.read() if limit is None else stream.read(limit))
                except (OSError, ValueError):
                    pass

            reader = threading.Thread(target=drain, daemon=True)
            try:
                if stdin_text is not None and process.stdin is not None:
                    # Written before the reader starts so an interpreter waiting
                    # on its script is not deadlocked against a reader waiting
                    # on its output. These scripts are a few hundred bytes,
                    # comfortably inside the pipe buffer.
                    process.stdin.write(stdin_text.encode("utf-8"))
                    process.stdin.close()
                reader.start()
                reader.join(max(deadline.remaining, 0.0))
                if reader.is_alive():
                    # Still reading when the budget ran out: the tool is wedged.
                    # Killing the group is what ends the thread's read, and it
                    # is done in the `finally` below so the same path covers a
                    # timed-out wait.
                    return None
                returncode = process.wait(timeout=max(deadline.remaining, 0.0))
            finally:
                # THE CONDITION IS THE READER, NOT THE CHILD. Gating this on
                # `process.poll() is None` looks right and is the F4 bug: when
                # the direct child has already exited while a descendant still
                # holds the inherited stdout, `poll()` returns 0, the kill is
                # skipped, and the reader stays blocked on a pipe nothing will
                # close. `Popen.__exit__` then calls `stdout.close()`, which
                # needs the `BufferedReader` lock that blocked reader holds, so
                # the MAIN thread deadlocks forever — an unbounded freeze on
                # the very axis this bound exists for (round 3, F4: measured
                # never returning after 20 s on a 2 s budget).
                #
                # Killing the group whenever the reader is still alive covers
                # both shapes at once, and is harmless when the group is
                # already empty. `wait()` after the kill is what reaps the
                # child: without it the process becomes a zombie until this
                # process exits.
                if reader.is_alive() or process.poll() is None:
                    _kill_tree(process, pgid)
                    try:
                        process.wait(timeout=_REAP_TIMEOUT_S)
                    except subprocess.SubprocessError:
                        pass
                    # The kill closes the write end, so the abandoned read ends
                    # promptly. Joining before `Popen.__exit__` runs is what
                    # keeps `stdout.close()` off the reader's lock — bounded,
                    # because a reader that somehow outlives its own pipe must
                    # not become the new unbounded wait.
                    reader.join(_REAP_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, ValueError):
        # OSError covers the binary vanishing between `which` and `exec`;
        # SubprocessError covers the timeout; ValueError covers a pipe closed
        # under us. All of them mean "no clipboard image".
        return None
    if returncode != 0:
        return None
    stdout = captured[0] if captured else b""
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
#: BOTH pasteboard shapes are answered by ONE script, and the reason is
#: measured rather than aesthetic. ``osascript`` costs 2-4 s of WALL time per
#: spawn on a loaded machine against ~0.25 s of CPU — it is the AppleScript
#: runtime starting up, not the pasteboard, and merely reaching
#: ``generalPasteboard()`` with an empty script reproduces it (``pbpaste``, no
#: AppleScript involved, answers the same pasteboard in 0.05 s). So the spawn
#: count IS the latency, and asking the image question and the file-URL
#: question separately doubled the cost of every miss — which is the common
#: case, since a text clipboard reaches both (review round 2, U3/U7).
#:
#: The image wins when several are present: it is what the user copied in the
#: reported gesture. The file-URL branch is the Finder ``Cmd+C`` fallback, and
#: it is asked BEFORE text for a measured reason — a Finder copy puts
#: ``public.file-url`` AND the file's display name on the pasteboard together
#: (verified against a real Finder-shaped copy: ``public.file-url``,
#: ``NSFilenamesPboardType`` and an Apple URL flavor all present), so testing
#: text first would type a filename instead of attaching the copied file.
#:
#: TEXT is last and is the ordinary case. It is read in this same script rather
#: than by a second tool (``pbpaste``) so the single-spawn property survives:
#: the spawn IS the latency, and a text clipboard is the most common thing
#: ``Ctrl+V`` meets.
#:
#: Output is a one-line verdict on stdout (``image``, ``text``, or the
#: NUL-separated paths); the image bytes AND the text go to a FILE, because
#: ``osascript`` prints its result through a text coercion that mangles binary
#: — the same trap the Windows backend documents. Text takes the file route
#: too: the coercion normalises line endings, so a multi-line clipboard would
#: come back altered, and its content would otherwise be indistinguishable
#: from the one-word verdicts.
#:
#: TIFF is handled explicitly because it is not a rare case: several macOS apps
#: (Preview's copy, some screenshot utilities) put ONLY ``public.tiff`` on the
#: pasteboard, and no provider accepts TIFF. ``NSBitmapImageRep`` re-encodes it
#: to PNG in-process, which is cheaper and more reliable than declining.
#: ``representationUsingType:4`` is ``NSBitmapImageFileTypePNG``; the numeric
#: form is used because the symbolic constant is not visible to AppleScript's
#: ObjC bridge.
#:
#: The URLs are enumerated by INDEX rather than with ``repeat with u in``,
#: which hands back AppleScript's own coerced items instead of the ``NSURL``
#: objects and fails with "doesn't understand the isFileURL message". They are
#: NUL-separated because a newline is legal in a macOS filename and splitting
#: on one turned a real path into two nonexistent ones (round 1, F5).
_MACOS_CLIPBOARD_SCRIPT = """\
use framework "AppKit"
use framework "Foundation"
use scripting additions

on run argv
\tset dest to item 1 of argv
\tset pb to current application's NSPasteboard's generalPasteboard()
\tset png to pb's dataForType:"public.png"
\tif png is missing value then
\t\tset tiff to pb's dataForType:"public.tiff"
\t\tif tiff is not missing value then
\t\t\tset rep to current application's NSBitmapImageRep's imageRepWithData:tiff
\t\t\tif rep is not missing value then
\t\t\t\tset props to current application's NSDictionary's dictionary()
\t\t\t\tset png to rep's representationUsingType:4 |properties|:props
\t\t\tend if
\t\tend if
\tend if
\tif png is not missing value then
\t\tpng's writeToFile:dest atomically:true
\t\treturn "image"
\tend if
\tset urls to pb's readObjectsForClasses:{current application's NSURL} options:(missing value)
\tset out to ""
\tif urls is not missing value then
\t\tset sep to (ASCII character 0)
\t\trepeat with i from 0 to ((urls's |count|() as integer) - 1)
\t\t\tset u to (urls's objectAtIndex:i)
\t\t\tif (u's isFileURL()) as boolean then set out to out & ((u's |path|()) as text) & sep
\t\tend repeat
\tend if
\tif out is not "" then return out
\tset str to pb's stringForType:"public.utf8-plain-text"
\tif str is missing value then return ""
\tset payload to str's dataUsingEncoding:(current application's NSUTF8StringEncoding)
\tif payload is missing value then return ""
\tpayload's writeToFile:dest atomically:true
\treturn "text"
end run
"""


def _read_macos(max_bytes: int, deadline: _Deadline) -> ClipboardContents:
    """macOS: image bytes, file URLs, or text, from ONE ``osascript`` spawn.

    All three shapes in one call because the spawn is the cost — see
    :data:`_MACOS_CLIPBOARD_SCRIPT`. Asking them separately doubled the latency
    of every clipboard miss, which is the case a text or empty clipboard hits
    (round 2, U3/U7), and adding text as a second spawn would have undone that
    fix on the shape that is now the most common one.

    The image and the text are both collected from a temp FILE, which is the
    only lossless channel out of ``osascript``; it is deleted before this
    returns.
    """
    if not shutil.which("osascript"):
        # Not reachable on a stock macOS, but this module must never assume a
        # binary exists just because the platform usually ships it.
        return ClipboardContents()
    with tempfile.TemporaryDirectory(prefix="lo-clip-") as tmp:
        dest = Path(tmp) / "clipboard.png"
        # `-` reads the script from stdin; everything after it is `argv` to the
        # script's `on run` handler, so the destination never has to be spliced
        # into the source text.
        stdout = _run(
            ["osascript", "-", str(dest)],
            deadline,
            stdin_text=_MACOS_CLIPBOARD_SCRIPT,
        )
        if stdout is None:
            return ClipboardContents()
        verdict = stdout.decode("utf-8", errors="replace")
        if verdict.strip() == "image":
            image = _as_image(_read_bounded(dest, max_bytes), max_bytes)
            # An image that was found and then refused (oversized, unsendable)
            # does NOT fall through to the file-URL list: the pasteboard's
            # answer to "what did the user copy" was the image, and attaching
            # some unrelated file instead would be a different gesture.
            return ClipboardContents(image=image)
        if verdict.strip() == "text":
            # Bounded by the same ceiling as the image, and for the same reason:
            # the clipboard is an untrusted-size source, and this text is about
            # to be inserted into the composer's document. Over the bound reads
            # as an empty clipboard rather than as an error — the same collapse
            # every other "nothing usable here" case takes.
            raw = _read_bounded(dest, max_bytes)
            if raw is None:
                return ClipboardContents()
            return ClipboardContents(text=raw.decode("utf-8", errors="replace"))
    return ClipboardContents(paths=tuple(p for p in verdict.split("\x00") if p.strip()))


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


#: Text MIME types worth asking a Wayland compositor for, in preference order.
#: ``text/plain;charset=utf-8`` is what most toolkits actually offer and is
#: unambiguous about encoding; bare ``text/plain`` is the fallback, and the two
#: X11 selection names appear because compositors commonly re-advertise them
#: through XWayland. Only types the compositor LISTS are ever read \u2014 see
#: :func:`_read_wayland`.
TEXT_MIME_PREFERENCE = ("text/plain;charset=utf-8", "text/plain", "UTF8_STRING", "STRING")


def _read_wayland(max_bytes: int, deadline: _Deadline) -> tuple[ClipboardImage | None, str]:
    """Wayland: ONE ``--list-types`` listing, then at most one real read.

    Image types first, then text, and NEITHER is read speculatively: a type the
    compositor did not list is never asked for. That discipline predates the
    text shape and is kept for the reason it was introduced \u2014 ``wl-paste
    --type X`` on a clipboard with no X both fails AND, on some compositors,
    blocks while the offer is negotiated, so a speculative read is a stall
    rather than a cheap miss.

    One listing serves both questions, which is why this is a single function
    and not an image reader beside a text reader: asking twice would double the
    spawn count on the shape ``Ctrl+V`` meets most often, the same cost the
    macOS backend collapses into one script (round 2, U3/U7).
    """
    if not shutil.which("wl-paste"):
        return None, ""
    listing = _run(["wl-paste", "--list-types"], deadline)
    if listing is None:
        return None, ""
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
            return image, ""
    for mime in TEXT_MIME_PREFERENCE:
        if mime not in offered:
            continue
        # ``--no-newline`` matches the image reads and stops ``wl-paste``
        # appending a newline the user never copied, which in a prompt buffer
        # is a blank line they then have to delete.
        text = _decode_clipboard_text(
            _run(
                ["wl-paste", "--no-newline", "--type", mime],
                deadline,
                max_bytes=max_bytes,
            ),
            max_bytes,
        )
        if text:
            return None, text
    return None, ""


def _read_x11_text(max_bytes: int, deadline: _Deadline) -> str:
    """X11: the clipboard's plain text, or ``""``.

    ``-t UTF8_STRING`` rather than bare ``-o``: without a target ``xclip``
    picks one itself and can hand back a non-text flavor on a clipboard that
    offers several, which would reach the composer as mojibake.
    """
    if not shutil.which("xclip"):
        return ""
    return _decode_clipboard_text(
        _run(
            ["xclip", "-selection", "clipboard", "-t", "UTF8_STRING", "-o"],
            deadline,
            max_bytes=max_bytes,
        )
    )


def _decode_clipboard_text(data: bytes | None, max_bytes: int | None = None) -> str:
    """Clipboard bytes as text, or ``""`` — the one decode every backend uses.

    ``errors="replace"`` and never a raise: this runs on a keystroke, and a
    clipboard holding bytes that are not valid UTF-8 is a paste that should
    degrade, not an exception on the key that pasted. Shared so the platforms
    cannot disagree about what a clipboard string is.
    """
    if not data:
        return ""
    if max_bytes is not None and len(data) > max_bytes:
        return ""
    return data.decode("utf-8", errors="replace")


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


#: Windows text, read through the same PowerShell the image backend picks.
#:
#: ``-Raw`` is what makes this correct rather than merely working:
#: ``Get-Clipboard`` without it returns an ARRAY of lines, which PowerShell
#: then joins with the host's line separator on the way to stdout, so a
#: multi-line clipboard comes back re-terminated. ``-Raw`` hands over the
#: string as the clipboard holds it.
#:
#: Stdout is safe here where it is not for the image, because this payload IS
#: text: the output encoding that corrupts binary is exactly the right
#: treatment for a string, and ``[Console]::OutputEncoding`` is pinned to UTF-8
#: so the bytes this process decodes match what was copied.
#:
#: NOTE: unit-tested against mocked invocations only, like the image script —
#: there is no Windows host in this project's development or CI environment.
_WINDOWS_TEXT_SCRIPT = """\
$ErrorActionPreference = 'Stop'
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
  $text = Get-Clipboard -Raw
  if ($null -eq $text) { exit 1 }
  [Console]::Out.Write($text)
} catch {
  exit 1
}
"""


def _read_windows_text(max_bytes: int, deadline: _Deadline) -> str:
    """Windows: the clipboard's text, or ``""``.

    ``-STA`` for the same reason the image backend needs it: the Windows
    clipboard API is single-threaded-apartment only.
    """
    shell = _windows_shell()
    if shell is None:
        return ""
    with tempfile.TemporaryDirectory(prefix="lo-clip-") as tmp:
        script = Path(tmp) / "read_text.ps1"
        try:
            script.write_text(_WINDOWS_TEXT_SCRIPT, encoding="utf-8")
        except OSError:
            return ""
        return _decode_clipboard_text(
            _run(
                [
                    shell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-STA",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                deadline,
                max_bytes=max_bytes,
            ),
            max_bytes,
        )


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

    Everything else stays collapsed: an empty clipboard, a missing ``xclip``
    and a wedged daemon are one answer, because a message that guessed between
    them would be inventing a diagnosis. A TEXT-only clipboard is no longer in
    that collapse — it has its own field, because ``Ctrl+V`` has to insert it.

    The three shapes are MUTUALLY EXCLUSIVE by construction, in the order
    image, paths, text. That order is the user's intent, not a convenience: a
    Finder copy puts a file URL and its display name on the pasteboard at once,
    so a text field filled alongside paths would let one gesture both attach a
    file and type its name. The backends resolve the precedence; the caller
    reads whichever field is set.
    """

    image: ClipboardImage | None = None
    paths: tuple[str, ...] = ()
    #: Plain text on the clipboard, when there was no image and no file URL.
    #: Read for ``Ctrl+V``, which is a SYSTEM paste and therefore owes the user
    #: the ordinary text case as well as the attachable ones — Textual's own
    #: ``ctrl+v`` action pastes ``App.clipboard``, an internal buffer that a
    #: copy made in another application never touches.
    text: str = ""
    #: The read never happened: this session is remote, so the clipboard would
    #: be the server's. Not a failure to find an image, and must not be
    #: reported as one.
    refused_remote: bool = False


def read_clipboard(
    max_bytes: int = MAX_CLIPBOARD_READ_BYTES,
    *,
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ClipboardContents:
    """One look at the clipboard for one paste, under one deadline.

    The ONLY entry point, so the whole operation cannot cost more than
    :data:`CLIPBOARD_TIMEOUT_S` no matter how many shapes or subprocesses a
    platform needs. Separate per-shape functions were how round 1's F2 got a
    4 s worst case out of a 2 s constant, and re-exposing them would put that
    back one caller at a time.

    ``max_bytes`` defaults to the INGEST ceiling
    (:data:`MAX_CLIPBOARD_READ_BYTES`) rather than to any attachment budget:
    resizing happens downstream, and a ceiling applied before it discards
    images that would have been perfectly attachable (round 1, U1).

    ``platform`` and ``env`` are injectable so each backend is testable on any
    host — this module's whole failure mode was a platform assumption that only
    one developer's environment could disprove.

    Wayland is chosen over X11 by ``WAYLAND_DISPLAY`` rather than by
    distribution: a Wayland session commonly also runs XWayland, so ``DISPLAY``
    is set in both, and testing ``DISPLAY`` first would route a Wayland session
    to ``xclip`` and read XWayland's separate, usually empty selection.
    """
    if not clipboard_reads_are_local(env):
        return ClipboardContents(refused_remote=True)

    deadline = _Deadline(CLIPBOARD_TIMEOUT_S)
    system = sys.platform if platform is None else platform
    source = os.environ if env is None else env

    if system == "darwin":
        # One spawn answers both shapes; see `_MACOS_CLIPBOARD_SCRIPT` for why
        # the spawn count is the latency here.
        return _read_macos(max_bytes, deadline)

    image: ClipboardImage | None = None
    text = ""
    if system == "win32":
        image = _read_windows_image(max_bytes, deadline)
        if image is None:
            text = _read_windows_text(max_bytes, deadline)
    elif system.startswith("linux") or "bsd" in system:
        if source.get("WAYLAND_DISPLAY"):
            # One call for both shapes: the compositor's type listing answers
            # the image question and the text question together.
            image, text = _read_wayland(max_bytes, deadline)
        elif source.get("DISPLAY"):
            image = _read_x11_image(max_bytes, deadline)
            if image is None:
                text = _read_x11_text(max_bytes, deadline)
        # A headless Linux box (a container, a bare tty) has no clipboard at
        # all, and shelling out to discover that costs a subprocess per paste.
    # The text read is skipped when an image was found, so the common
    # screenshot gesture does not pay for a second subprocess, and so the two
    # fields keep the exclusivity `ClipboardContents` documents.
    #
    # No file-URL shape off macOS: every other platform's file manager puts the
    # paths on the clipboard as text too, which the terminal already
    # bracket-pastes into the composer's existing path branch — a second route
    # here would be a way for one copy to be attached twice. That text now
    # arrives through the `text` field on Ctrl+V as well, where the composer's
    # path branch parses it exactly as it parses a bracketed paste.
    return ClipboardContents(image=image, text=text)

# Evidence: clipboard image paste (issue #372)

Captured on macOS (Darwin 25.6.0, arm64) and in Debian bookworm under Docker
28.3.0, on 2026-08-28, against `dev-clipboard-paste`.

| File | What it shows |
|---|---|
| `macos-pasteboard.txt` | Five real pasteboard states driven through the real `OperatorApp` |
| `linux-xclip-wlpaste.txt` | The X11 and Wayland backends against real `xclip` and `wl-paste` |
| `composer-before.png` | `Cmd+V` on a PNG pasteboard at `origin/main`: nothing happens |
| `composer-after.png` | The same gesture on this branch: `[Image #1, 1200×700]` |
| `composer-no-image-notice.png` | The notice a paste raises when nothing was attachable |
| `baseline-preexisting.txt` | The failing tests reproduced on clean `origin/main` |
| `macos_evidence.sh` | Reproduces `macos-pasteboard.txt` |
| `linux_evidence.sh` | Reproduces `linux-xclip-wlpaste.txt` (runs inside the container) |

## What is proven

**The reported gesture works, end to end.** `macos-pasteboard.txt` case 1 puts a
real PNG on this Mac's pasteboard — `clipboard info` reports `«class PNGf»,
6291` while `pbpaste` returns **0 bytes**, which is precisely why the terminal
had nothing to send — and the composer ends up holding `[Image #1, 1200x700]`
with 6291 bytes of real PNG behind it, in 626 ms. The image also reaches the
model: submitting that draft sent one prompt carrying one image.

**TIFF-only pasteboards attach too.** Case 2 sets `public.tiff` with no PNG
flavor, the shape several macOS apps produce. `NSBitmapImageRep` re-encodes it
in-process and the composer attaches 16161 bytes of PNG. No provider accepts
TIFF, so declining would have been a visible failure in a case the user cannot
distinguish from case 1.

**Finder `Cmd+C` attaches.** Case 3's pasteboard is `«class furl», 27` and
nothing else — `pbpaste` again returns 0 bytes — and the composer attaches the
file it named, through the same path branch a drag-and-drop uses.

**The negative cases are silent and honest.** A text-only clipboard (case 4) and
an SSH session with an image still on the pasteboard (case 5) both attach
nothing, insert nothing, and raise `no image on the clipboard to attach`. The
SSH case is the one that matters: the clipboard was full and readable, and the
read was refused anyway, because over SSH it would be the *server's* clipboard.

**Linux is executed, not mocked.** `linux-xclip-wlpaste.txt` runs both backends
against real tooling in a container: `xclip` under Xvfb and `wl-paste` under a
headless sway compositor. Both return the 109415-byte PNG **byte-identical to
the source** (sha256 `4ddea2b7220dec89`). The negative cases run there too: an
oversized payload refused at `max_bytes=1024`, the SSH refusal, a headless box
with no `DISPLAY` spawning no subprocess at all, and both backends returning
`None` with their binaries off `PATH`.

**Wayland wins over XWayland.** The last Wayland case leaves the PNG on the X11
clipboard and text on the Wayland one, then reads with both `WAYLAND_DISPLAY`
and `DISPLAY` set. The result is `None`, which is only possible if `wl-paste`
was chosen — a dispatch testing `DISPLAY` first would have returned the PNG from
XWayland's separate selection.

**Real tooling found a bug that mocks could not.** `xclip -selection clipboard
-t image/png -o` on a **text-only** clipboard exits **zero** and prints the
text: the target request is advisory, and the selection owner answers with
whatever it has. The first implementation trusted its own request and would have
returned `ClipboardImage(b'just text, no image', 'image/png')` — a corrupt image
block on its way to a provider. The captured line

```
raw xclip -t image/png -o gives: just text, no image (exit 0)
backend result: None
```

is that bug and its fix: the payload is sniffed rather than trusted, so the MIME
type on the result describes the bytes. Regression-tested in
`tests/unit/test_clipboard.py`.

**Frames are settled.** Each capture saves two consecutive frames; `paste.svg`
and `settled.svg` are byte-identical in all three states, so nothing reflows
after paint.

## What is NOT proven

**Windows has not been executed.** There is no Windows host in this project's
environment. The PowerShell backend is unit-tested against mocked invocations,
and its command was reviewed for the two traps that make the obvious
implementation wrong — binary on PowerShell's text stdout is silently corrupted,
and `-Encoding Byte` does not exist in PowerShell 7+ — but no claim is made that
it has run on Windows.

## Pre-existing test failures

`baseline-preexisting.txt` is `tests/unit/tools/test_eval_tool.py` and
`tests/unit/server/test_server_models.py` run on a clean `origin/main` worktree
(`fa62d097`): **32 failed, 25 passed**. The same tests are the entire failure set
on this branch, so they are unrelated to this change.

`tests/unit/session/test_conversation_name_persistence.py` and
`tests/unit/tui/test_app_pilot.py` also failed once here under full-suite load
and were reproduced failing identically on the clean baseline under the same
load, then passed on both trees when run quiet. They are load-sensitive flakes,
not a regression.

## Reproducing

The frames and the macOS captures need this Mac's pasteboard, so they are
scripts rather than tests:

```sh
sh docs/evidence/clipboard-paste/macos_evidence.sh
```

Linux runs anywhere Docker does, and needs no local X11 or Wayland session:

```sh
docker run --rm \
  -v "$PWD/local_operator:/work/local_operator:ro" \
  -v "$PWD/docs/evidence/clipboard-paste/linux_evidence.sh:/work/run.sh:ro" \
  debian:bookworm-slim sh /work/run.sh
```

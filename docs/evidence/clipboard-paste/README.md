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
| `round1-remediation.txt` | Review round 1 fixes verified: U1, F1/D1, D2/U2, D3, F2 |
| `composer-whitespace-paste.png` | A pasted indent lands in the buffer again (D1) |
| `composer-notice-ssh.png` | The SSH notice, which no longer claims the clipboard is empty (D2/U2) |
| `macos_evidence.sh` | Reproduces `macos-pasteboard.txt` |
| `round1_evidence.sh` | Reproduces `round1-remediation.txt` |
| `round2-remediation.txt` | Review round 2 fixes verified: F2/U6, U3/U7, D8, D9, D10, D11 |
| `round2_evidence.sh` | Reproduces `round2-remediation.txt` |
| `round3-remediation.txt` | Review round 3 fixes verified: F4, D12, D13 |
| `round3_evidence.sh` | Reproduces `round3-remediation.txt` |
| `composer-notice-file.png` | The reworded file notice (D12) |
| `composer-notice-unattachable.png` | The reworded refusal notice (D10) |
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

## Review round 1 remediation

`round1-remediation.txt` is the re-verification after the three review rounds,
run the same way: real pasteboard state through the real `OperatorApp`.

**U1 (blocker) — a real Retina screenshot attaches.** The pasteboard PNG is
7.37 MB; reading it at the old 4 MB attachment cap still returns `None`, which
is the blocker reproduced. With the ingest ceiling the composer shows
`[Image #1, 1568x1014 ↓]` in 0.77 s, the attachment is 1.01 MB (under the cap
that governs what is sent), and submitting delivers one image to the model. The
fixture is deliberately high-entropy: a flat-colour image of the same
dimensions compresses under the old cliff by accident, which is exactly why the
first round of testing missed this.

**F1/D1 (blocker) — pasted whitespace is inserted again.** All seven payloads
(`'    '`, `'  '`, `' '`, `'\t'`, `'\n'`, `'\n\n'`, and the genuinely empty
one) now land in the buffer verbatim, and a whitespace paste raises no notice
at all — it succeeded on its own terms.

**D2/U2 (major) — the message states the outcome.** Four states, three
sentences: an empty or text-only clipboard gets "Couldn't attach an image from
the clipboard", an image that was found and refused gets "Try a smaller one, or
paste its file path", and SSH gets "Clipboard images aren't read over SSH".
Nothing asserts the clipboard is empty when the app did not look.

**D3 (major) — a held notice cannot be replayed after it is false.** The
designer's captured sequence, re-run: the MCP failure holds the slot, the empty
paste defers the notice, the screenshot attaches, and the deferred card is
`None` — so when the failure expires nothing stale surfaces.

**F2 (major) — the whole read is bounded.** With every `xclip` hanging and four
MIME types tried, total blocking time is 2.00 s against a 2.0 s cap (it was
8.0 s), and each spawn is handed the remaining budget rather than a fresh one.

## Review round 2 remediation

`round2-remediation.txt`, reproducible with `round2_evidence.sh`. Captured at
load average 143, so the timings are upper bounds.

**F2/U6 (blocker) — a real hung tool is bounded again.** Round 1's fix replaced
`subprocess.run(timeout=)` with a hand-rolled `Popen` whose blocking
`stdout.read()` ran *before* the deadline check, so a tool holding stdout open
never reached it: 15 s on X11, 12 s on macOS, child orphaned. The read now runs
on a thread the deadline abandons, and the child is killed by process GROUP —
without that, `sh -c` leaves a grandchild holding the pipe and the bound is
defeated one level down (measured 30 s on a 1 s budget while writing this).
All three hang shapes now return in 2.0 s with no orphan:

```
hung tool, bounded read    returned 2.02s (budget 2.0s) -> None  orphans: none
hung tool, unbounded read  returned 2.02s (budget 2.0s) -> None  orphans: none
answers then hangs         returned 2.00s (budget 2.0s) -> None  orphans: none
healthy tool               -> b'hello'
```

The regression tests for this spawn **real** processes. The round-1 fakes could
not fail on the bug: their `stdout` was a `BytesIO` that never blocks, and the
hang was injected by raising from `wait()` — the one line a real hang never
reaches.

**U3/U7 (major/minor) — the latency is gone.** The cost was `osascript`
*spawns*: 2-4 s of wall time each against ~0.25 s of CPU (`pbpaste` answers the
same pasteboard in 0.05 s, so it is the AppleScript runtime, not the
pasteboard), and the miss path paid two of them. One script now answers both
shapes, and a copied-whitespace paste does not consult the clipboard at all:

```
TEXT paste (control)               24 ms  -> 'hello world'
WHITESPACE paste (was 4203ms)     106 ms  -> '    '
TAB paste (was 3850ms)            157 ms  -> '\t'
EMPTY -> screenshot (was 4-9s)   1825 ms  -> '[Image #1, 1568x1014 ↓]'
```

**D8 (major) — the path route retires a held notice.** The notice tells the user
to paste a file path; they do, it works, and the card used to reappear to deny
it. The full sequence now ends with nothing stale surfacing.

**D9 (minor) — copied whitespace is no longer traded for an image.** Reproduced
first (an indent with a PNG on the pasteboard became `[Image #1, 1568x200]`),
then fixed: only the genuinely empty payload lets an image win.

**D10/D11 (minor) — message accuracy and toast rank.** "Paste its file path" is
gone from the failures it could not cure, a copied file that will not attach has
its own message, and the vague notice returns to the courtesy duration so it no
longer suppresses a copy receipt for 10 s.

## Review round 3 remediation

`round3-remediation.txt`, reproducible with `round3_evidence.sh`.

**F4 (major) — the fourth hang shape.** When the direct child exits while a
descendant still holds the inherited stdout, `poll()` returns 0, so a kill gated
on it was skipped; the reader stayed blocked and `Popen.__exit__` deadlocked the
MAIN thread on `stdout.close()`. Reproduced (never returned after 20 s on a 2 s
budget, faulthandler pinning both threads), then fixed by gating the kill on the
READER and remembering the pgid from spawn — `os.getpgid()` on a reaped leader
raises, so the lookup form degrades to no kill in exactly this case. Now returns
in 2.00 s with no survivors, and the regression test hangs the runner at
`cb0ef24a`.

**D12 (major) — one image, two routes, contradictory answers.** The clipboard
route bounds before the attachment cap (U1's fix); the path route stat-gated
against that cap, so Finder `Cmd+C` refused a screenshot `Cmd+V` attached and
blamed its format. Both the copy and the gate are fixed, so the same 6.7 MB PNG
now yields `[Image #1, 1568x543 ↓]` either way, and the four genuine failures
get a sentence that does not assert a cause.

**D13 (minor) — a showing card outliving its answer.** `withdraw` instead of
`drop_deferred`; the attach lands 45-57 ms after the notice, so there is no
mid-read to interrupt. The D3/D8 held-card sequence stays closed.

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

# Clipboard paste outside cmux: what #376 missed, and how this was verified

PR #376 set out to fix issue #372 (a clipboard image would not attach outside
cmux). It shipped a correct clipboard reader and hung it off a branch the
terminal never enters, so the bug survived on exactly the emulators it was
filed against. This directory holds the measurements that establish both the
old failure and the new fix.

## 1. The false premise, measured

#376 assumed that with an image-only pasteboard the terminal sends a bare
`ESC[200~ ESC[201~`, which Textual's `XTermParser` turns into `Paste(text='')`.
The parser half is true and irrelevant: **the terminal never sends those
bytes**.

Captured with `probe2.py`, a raw-mode PTY probe that enables bracketed paste
(`ESC[?2004h`) and dumps every byte stdin delivers. Driven through AppleScript
+ System Events on macOS 25.6 (arm64).

| terminal | clipboard | keystroke | bytes | artifact |
|---|---|---|---|---|
| Terminal.app | text `hello-control` | Cmd+V | 25 — `\x1b[200~hello-control\x1b[201~` | `bytes/terminal-text-cmdv.bin` |
| Terminal.app | PNG screenshot | Cmd+V | **0** | `bytes/terminal-image-cmdv.bin` |
| Terminal.app | PNG screenshot | Ctrl+V | 1 — `\x16` | `bytes/terminal-image-ctrlv.bin` |
| Ghostty | PNG screenshot | Cmd+V | **0** | `bytes/ghostty-image-cmdv.bin` |
| Ghostty | PNG screenshot | Ctrl+V | 1 — `\x16` | `bytes/ghostty-image-ctrlv.bin` |

Zero bytes means no `Paste` event, so `_on_paste` never runs and #376's entire
code path — the empty-paste branch, `_attach_clipboard_image`, the
`EditorPasteEmpty` toast — is unreachable outside cmux. The zero-byte case is
also audible: both terminals beep, which is the "nil sound" in the report.

`Ctrl+V` arrives on both terminals, so it is the only paste keystroke a TUI can
observe. That is why the fix is a key binding and not a paste handler.

## 2. The clipboard reader was never the problem

`local_operator/clipboard.py` was verified against this Mac's real pasteboard
and returned `image/png`, 1874491 bytes, in 0.26 s. It had no reachable caller
outside cmux. This PR adds the caller and a text shape; it does not rewrite the
reader.

## 3. The fix, in the real app

The real TUI was run from this worktree (`PYTHONPATH` pinned to it, so the code
under test is what executes) in both terminals, with a real 2400x1600
screenshot on the pasteboard, and `ctrl+v` pressed through System Events.

| artifact | what it shows |
|---|---|
| `terminal-before-ctrlv.png` | Terminal.app, composer empty, image on the pasteboard |
| `terminal-after-ctrlv-image.png` | `[Image #1, 1568x1045 ↓]` — attached and bounded |
| `terminal-after-ctrlv-text.png` | a text clipboard inserted as text, beside the marker |
| `terminal-after-ctrlv-empty.png` | empty clipboard: the notice fires, and does **not** advise `ctrl+v` to someone who just pressed it |
| `ghostty-after-ctrlv-image.png` | the same attach in Ghostty |

`shot-source.png` is the screenshot used as the pasteboard payload.

## 4. Rendered frames (AGENTS.md "Visual validation")

`frames/` holds before/after SVG stills captured from the **real**
`OperatorApp` (which declares `CSS_PATH`, so the stylesheet applies — the
lightweight test hosts do not, and would not show a style change at all).

`before-composer-empty.svg` is ONE shared before-frame, deliberately. Every
before-state of this change is the same frame — an empty composer, because
before the fix `ctrl+v` did nothing at all on every clipboard shape — and an
earlier revision shipped that same frame three times under three names, which
implied three distinct captured states (design round 1, D6).

| state | before | after |
|---|---|---|
| image on clipboard, `ctrl+v` | `before-composer-empty.svg` | `after-image.svg` — `[Image #1, 1568x200]` |
| empty clipboard | same | `after-empty.svg` — "Nothing on the clipboard to paste." |
| remote session | same | `after-remote.svg` — one line, logo unclipped (D4) |
| text on clipboard | same | `after-text.svg` |
| read timed out | same | `after-timeout.svg` — "Clipboard read timed out. Try ctrl+v again." (U3) |
| the ambient affordance | n/a | `after-tip-ctrlv.svg` — the splash tip that teaches the key (D1/U1) |

The before-frame was captured before any file was edited, per AGENTS.md.

## 6. Round 1 review fixes, verified in a real terminal

Re-driven in Terminal.app against the fixed code, same method as section 3.

| artifact | finding | result |
|---|---|---|
| `terminal-ctrlv-selection.png` | U2 + U3 | typed `keep WORD`, selected `WORD`, `ctrl+v` with `REPLACEMENT` on the clipboard → **`keep REPLACEMENT`** (was `keep REPLACEMENTWORD`). The same frame shows the **"Reading the clipboard…"** card acknowledging the slow read. |
| `terminal-ctrlv-selection-image.png` | U2, image shape | same gesture with a PNG on the pasteboard → **`keep [Image #1, 900x600]`**; the selection is replaced, not inserted into. |

`/help` was also driven live: the `ctrl+v` row now sits directly under the
command table, on one line at 120 columns (71 cells against the 76-cell box at
80 columns), out of the `lop config edit` toggle block it used to sit inside
(D2).

## 5. Live clipboard reads, all four shapes

Against the real macOS pasteboard, through the shipped `read_clipboard`:

```
IMAGE   -> mime image/png, 1874491 bytes, text '', in 0.26s
FILEURL -> paths ('…/shot-source.png',), text '', image None
TEXT    -> 'héllo — ünicode ✅\nline2'   (unicode and newlines exact)
EMPTY   -> ClipboardContents(image=None, paths=(), text='', refused_remote=False)
```

Exclusivity holds in every case: a Finder copy carries a file URL *and* the
file's display name, and the backend answers with the path, not the name.

## Reproducing

```sh
# byte capture (run inside the terminal under test)
python3 probe2.py /tmp/out.bin 10
# put a PNG on the pasteboard
osascript -e 'set the clipboard to (read (POSIX file "…/shot-source.png") as «class PNGf»)'
```


## 7. Round 2 review fixes

**U3 — the progress card now paints DURING the read.** Round 2 measured the
card arriving 2-3 ms before the paste at every read duration: the timer fired
on time (0.351 s) but the handler ran at `t_done`, because
`action_system_paste` is an awaited binding action holding the Editor's message
pump and `post_message` enqueues onto that same blocked queue. Posting to the
App's pump does not help either (measured: handler at 1.512 s against a 1.5 s
read). The event loop is fine throughout, so the card is now raised by a timer
callback that CALLS the app directly instead of enqueuing anything.

| artifact | what it shows |
|---|---|
| `terminal-card-during-read.png` | **1.6 s into a 3.5 s read**: "Reading the clipboard…" is on screen. This is the exact frame round 2 captured as blank. |
| `terminal-card-retired.png` | after the read: card gone, `[Image #1, 900x600]` attached |
| `terminal-text-card-retired.png` | **D7**: the text route, 5.5 s after the paste, with no stale card. It used to sit for the full 5 s. |

The first frame also shows the U1 placeholder (`ctrl+v pastes an image`), and
the third shows it retired after the key has been used once.

Measured directly, before and after:

```
BEFORE  read=2.0s  handler ran at 2.012s  action returned at 2.013s
AFTER   card visible at 0.4s, 0.7s and 1.0s of a 1.3s read
```

**U8 — the test now fails if the card never paints.** The old assertion was
`len(reading_notices) == 1`, which was true while the card was invisible to
every user. `test_the_reading_card_is_on_screen_while_the_read_is_still_running`
samples the real `Toast` in the real `OperatorApp` from inside the clipboard
read itself (the one place genuinely concurrent with the stall). Verified by
regressing the fix deliberately:

```
with the direct call      : PASSED
with the call deferred    : FAILED - "at 0.32s of a 0.6s read the card was ''"
```

Sampling from a timer does not work, and that is itself the finding: every
scheduled callback on either pump is queued behind the blocked handler.

**U1 — the affordance now reaches the mid-session user.** Verified the
designer's measurement first: with `LOCAL_OPERATOR_NO_SHIMMER=1`,
`_sync_tip_timer` creates no timer at all, `_tip_index` holds at 0 forever, and
the `ctrl+v` tip is unreachable on any launch for those users. The tip stays
(it is right for the splash) but is no longer the only surface: the composer
placeholder now reads `Message Local Operator…  ctrl+v pastes an image` until
the key has been used once. That is on screen in exactly the empty-composer
state a user is in when they reach for a paste, mid-session, where
`WelcomeView` is not displayed at all.

Adding a fourth `welcome.HINTS` row was tried and reverted: it costs the splash
a whole terminal row (the block goes 23 -> 24) and the height ladder spends
that row on the logo.

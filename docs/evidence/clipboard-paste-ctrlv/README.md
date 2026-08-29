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

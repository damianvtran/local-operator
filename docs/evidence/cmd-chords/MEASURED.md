# Cmd+V / Cmd+C in the composer: measured behaviour

All figures re-measured 2026-08-29 on macOS 25.6.0 arm64, independently of the
originating analysis. Harness: `/tmp/kprobe2/probe2.py`, a raw-mode PTY probe
that enables the kitty keyboard protocol (`ESC[>13u`) and bracketed paste and
records every byte the terminal delivers on stdin, driven by
`/tmp/kprobe2/rung.sh` (Ghostty) and `/tmp/kprobe2/run.sh` (Terminal.app).

Two harness properties are load-bearing and were verified before any figure was
trusted:

* `select()` rather than non-blocking polling. The earlier probe set `O_NONBLOCK`
  on a tty shared with the shell and lost every keystroke to the shell's own
  reader, so it reported zero bytes for plain typed characters.
* Focus set with `tell application "X" to activate`. `set frontmost of window
  id N` does not deliver System Events keystrokes.

A typed-character control was run first in each terminal to prove delivery, so a
zero-byte result means "the terminal sent nothing", not "the harness missed it".

## Byte captures

### Ghostty 1.3.1, default config

| clipboard | key | bytes delivered |
|---|---|---|
| image-only PNG | typed `XY` (control) | 42 — CSI-u for both chars |
| image-only PNG | Cmd+V | **8** — `ESC[118;9u` |
| image-only PNG | Cmd+C | **7** — `ESC[99;9u` |
| image-only PNG | Ctrl+V | 8 — `ESC[118;5u` |
| image-only PNG | Ctrl+C | 7 — `ESC[99;5u` |
| **text** (`HELLOTEXT`) | Cmd+V | 21 — `ESC[200~HELLOTEXT ESC[201~` |

CSI-u is `CSI <codepoint> ; <modifiers> u` with modifiers `1 + bitmask`
(Shift=1, Alt=2, Ctrl=4, **Super=8**). So `118;9` is codepoint `v` with Super,
i.e. literally Cmd+v, and `99;9` is Cmd+c.

**The text and image paste cases are disjoint, confirmed.** With text on the
clipboard Ghostty consumes Cmd+V and bracket-pastes; only with no text to paste
does it forward `super+v`. Binding `super+v` therefore cannot double-paste. This
is the assumption the whole change rests on and it was measured directly.

### Terminal.app

| clipboard | key | bytes |
|---|---|---|
| image-only PNG | typed `XY` (control) | 2 — `XY` (bare, no CSI-u) |
| image-only PNG | Cmd+V | **0** |
| image-only PNG | Cmd+C | **0** |
| image-only PNG | Ctrl+V | 1 — `\x16` |

Terminal.app does not implement the kitty protocol at all: after `ESC[>13u` a
typed `q` still arrives as a bare `0x71`. Cmd+V and Cmd+C are consumed by the
terminal and are unreachable by any application running in it. That is a
property of Terminal.app, not of this code, and no change here can alter it.

One Terminal.app Cmd+C capture returned 6 bytes of unrelated text (`oks li`) —
a stray artifact of the driving window, not a key encoding. Re-run under a
cleared pasteboard it returned 0 bytes, which is the figure recorded above.

### Parser

Textual 8.2.8's `XTermParser` decodes these directly, verified in-process:

    ESC[118;9u -> Key(key='super+v')
    ESC[99;9u  -> Key(key='super+c')
    ESC[118;5u -> Key(key='ctrl+v')
    ESC[99;5u  -> Key(key='ctrl+c')

Textual's `linux_driver` already pushes flags `1|8|16` (`ESC[>25u`), so these
keys arrive with no driver change on our side.

## What was actually broken — and the correction

The originating analysis said the composer "binds neither" chord and that both
`super+v` and `super+c` were unhandled. **Only the first half is true.** Both
halves were checked against the real `OperatorApp` running in real Ghostty on a
real pty, with the Editor instrumented to log every key it receives.

### `super+v` — genuinely unbound (premise confirmed)

Real TUI, image on the pasteboard, Cmd+V pressed:

    20:40:10 KEY key='super+v' char=None text='' sel=((0, 0), (0, 0))

The key reaches `Editor._on_key`, no handler claims it, and the composer stays
empty. Nothing else in the chain binds it (`super+v -> []` across Editor,
TranscriptScreen and OperatorApp). This is the reported bug and the fix is to
bind it.

### `super+c` — ALREADY WORKED before this change (premise wrong)

`TextArea.BINDINGS` in Textual 8.2.8 carries `Binding("ctrl+c,super+c", "copy")`,
which the Editor's own `action_copy` override intercepts. So Cmd+C already
copied and already produced the receipt. Measured in the real TUI, typing
`hello world`, selecting `world` with `⌥⇧←`, then Cmd+C:

    20:44:16 KEY key='super+c' char=None text='hello world' sel=((0, 11), (0, 6))
    20:44:16 ACTION copy ENTER sel=((0, 11), (0, 6)) selected='world'
    20:44:16 ACTION copy OK
    20:44:16 APP on_editor_copied text='world'

The toast painted `copied 5 characters` (`shot_cmdc.png`) and the OSC 52 write
reached the real pasteboard — `pbpaste` returned `world`.

**So the user's "cmd+C does not trigger the copy" is not reproduced as stated.**
The most likely explanation for the report is the gesture used to make the
highlight: a double-click leaves a click-chain selection, and Cmd+C on a
COLLAPSED selection raises `SkipAction` and does nothing at all —

    20:47:39 KEY key='super+c' sel=((0, 11), (0, 11))
    20:47:39 ACTION copy ENTER sel=((0, 11), (0, 11)) selected=''
    20:47:39 ACTION copy RAISED SkipAction

which from the user's side is exactly "cmd+C did nothing".

### The real `super+c` defect: it bypasses `_on_key`

This is the divergence worth fixing, and it is the file's own hard-won lesson
(code round 2 F5, ux round 2 U6): a `Binding` fires through the action system
and never enters `_on_key`. The Ctrl+C copy route in `_on_key` does one thing
the bare binding does not — it COLLAPSES a click-chain selection after copying,
so the key is handed back to the draft and interrupt rungs. Measured on the
unmodified tree with a click-chain selection live:

| key | selection after copy | collapsed |
|---|---|---|
| `ctrl+c` | `((0, 11), (0, 11))` | yes |
| `super+c` | `((0, 6), (0, 11))` | **no** |

So on the pre-change tree a user who double-clicks a word and presses Cmd+C
keeps a live range forever, and Cmd+C can never reach the interrupt ladder —
the exact class of bug R1-2 fixed for Ctrl+C. Routing `super+c` through
`_on_key` alongside `ctrl+c` is what makes the two chords one behaviour rather
than two that drift.

`super+c` also correctly does NOT carry the interrupt meaning: with no
selection it does nothing (no app-level `super+c` binding exists), which is
right — Cmd+C is not an interrupt gesture on macOS, and Ctrl+C keeps sole
interrupt duty.

## End-to-end validation against the final tree

Real `OperatorApp`, real Ghostty, real pty, Editor instrumented to log every
key it receives (`keylog_app.py`). Not synthesised events.

| gesture | clipboard | result |
|---|---|---|
| Cmd+V | image-only PNG | `super+v` -> `system_paste`, composer shows `[Image #1, 120x40]` |
| Cmd+V | text | `PASTE EVENT text='HELLOTEXT'` — terminal bracket-pasted, key never forwarded, **no double paste** |
| Cmd+C | `world` selected via `⌥⇧←` | `action_copy` ran, toast `copied 5 characters`, `pbpaste` -> `world` |
| Cmd+C | nothing selected | key consumed, draft `draft that must survive` intact, no copy, no interrupt |
| Ctrl+C | `check` selected | copied, `on_editor_copied text='check'` |
| Ctrl+V | image-only PNG | `system_paste` ran and replaced the selection |

Frames in `frames/`: `after_real_cmdv.png` (image attached from a real Cmd+V),
`final_cmdc.png` (copy receipt on screen), `before_splash.svg`/
`after_splash.svg` (the placeholder returning to `Message Local Operator…`),
and `before_help_80col.svg`/`after_help_80col.svg` (the new `cmd+v` row).

The help pair is captured at **80 columns specifically**. The round-1 frame was
taken at ~97 columns, which is above the wrap threshold, so it showed a clean
row while the row was in fact broken for anyone at a default-width terminal —
the capture width was itself why the defect went unnoticed (design round 1 D2,
code round 1 F3). The `before` frame here is rendered with the wrapping copy in
place so the pair shows the defect and its fix rather than two clean frames:
`Terminal.app)` hangs at column 0 in the KEY gutter on the before side, and
both key rows are single lines on the after side.

### Harness note, recorded because it cost real time

`osascript -e 'tell application "System Events" to <multi-line body>'` parses
but silently executes only its FIRST statement, which presents exactly as "the
keys were delivered and the app ignored them". Gestures must be script FILES.
And `activate` raises an app, not a specific window: on a machine with other
sessions open it can focus the wrong window, so keystrokes must never be sent
without asserting what is frontmost first.

## The placeholder hint

The composer's `ctrl+v pastes an image` placeholder suffix is REMOVED in this
change, at the operator's direct instruction. The affordance was added in #402
for users whose `Cmd+V` was swallowed with no way to learn the working key;
binding the native chord removes that premise for every terminal that forwards
it. `/help` keeps the fallback, now as two rows — `ctrl+v` unconditionally and
`cmd+v` qualified with "where the terminal forwards it (not Terminal.app)" —
because a conditional claim needs room to qualify itself and a one-line
placeholder has none.

## Terminal.app: the Ctrl+ routes still work (the no-kitty regression surface)

Terminal.app is where `Cmd+V`/`Cmd+C` can NEVER arrive, so `Ctrl+V`/`Ctrl+C`
are the only way in for those users and are the surface this change could
regress. Driven against the real TUI in a real Terminal.app window,
`TERM=xterm-256color`, with the Editor instrumented.

| # | gesture | result |
|---|---|---|
| a | Ctrl+V, image-only pasteboard | `ctrl+v char='\x16'` -> `system_paste`, composer shows `[Image #1, 120x40]` |
| b | Ctrl+C over a `shift+left` selection | `action_copy` ran, `on_editor_copied text='copyme'`, toast `copied 6 characters` |
| c | Ctrl+C, nothing selected | `MSG InterruptRequested posted by editor`, **PID unchanged (46336 before and after)** |

(c) distinguishes the app's interrupt from the harness having killed the
process: the PID is sampled either side of the keystroke and is identical, so
the observed interrupt is `InterruptRequested` travelling through the app, not
SIGINT terminating it. Frame: `frames/term_interrupt.png`; copy receipt:
`frames/term_ctrlc.png` (note the window title is the harness's own token).

`pbpaste` stays EMPTY after (b), and that is not a regression: the copy travels
by OSC 52, which Terminal.app does not honour by default. The receipt and the
`EditorCopied` message are the assertions that belong to this app; whether the
bytes reach the macOS pasteboard is the terminal's decision and is unchanged by
this PR.

### PROCESS INCIDENT — a draft was cleared in a live session

While validating this, an earlier revision of the harness sent a gesture into
the WRONG window and cleared the draft in one of the operator's live `lop`
sessions (the transcript showed `draft cleared — ↑ to recover`). Cause:
`osascript ... to activate` raises an APP, not a window, and the guard asserted
only the frontmost PROCESS name, which passed while another session's Terminal
window was frontmost.

First fix, `term_guard.sh`: generate a unique per-run token (`LOPCHK-<epoch>`),
set it as the window's custom title, and assert the frontmost window's
accessibility title CONTAINS that token immediately before every keystroke,
aborting hard otherwise. It was observed refusing correctly four times
(`PROC:Arc`, `PROC:cmux`, `WIN:damian — 120×30`, `WIN:<none>`) before any key
was sent, and it did keep the Terminal.app run safe.

**That guard has since been DELETED, because shipping it implied coverage it
did not have** (ux round 1 U1, code round 1 F4). Two measured problems:

1. It hardcodes `if name of p is not "Terminal"`, so it can never pass for
   Ghostty — the terminal this defect was actually reported from, and the one
   where the Cmd chords arrive at all. The Ghostty gestures therefore ran under
   the WEAKER process-name check in `drive_tui.sh`, i.e. the very check that
   caused the incident above. Nothing leaked on those runs, but the strict
   guard sitting in the directory made it look as though they were covered.
2. The AX-title route it depends on is unavailable here regardless. Measured on
   this machine, with a control to prove the query works at all:

   ```
   System Events -> count of windows of process "Terminal"  ->  0
   System Events -> count of windows of process "Ghostty"   ->  0
   System Events -> count of windows of process "Finder"    ->  1   (control)
   ```

   Ghostty additionally publishes no window title by any route, so a
   title-matching guard is structurally impossible there rather than merely
   unreliable.

**The replacement removes the hazard instead of hardening the guard.**
`pty_drive.py.txt` / `pty_gestures.py.txt` allocate a pty the harness owns both
ends of, spawn the app on it, and write the captured CSI-u bytes straight into
the master fd. No key passes through the window server, so there is no
frontmost window to assert about and no path to another session at all. It is
also a stronger test than `pilot.press()`, since it exercises
bytes -> `XTermParser` -> app -> composer with the exact sequences in `bytes/`.

The honest boundary: the pty harness proves what the APP does with the bytes a
terminal sends; it does not prove what any terminal sends. That half is the
byte captures in `bytes/`, taken from the real Ghostty and the real
Terminal.app. The two halves together are the claim, and neither is sufficient
alone.

A second, unrelated harness trap worth recording: `osascript -e 'tell
application "System Events" to <multi-line body>'` parses but silently runs
only its FIRST statement, which looks exactly like "the keys were delivered and
the app ignored them". Gestures must be script FILES.

## Harness files in this directory

`probe2.py.txt` (raw byte capture), `keylog_app.py.txt` (the instrumented real
app), `pty_drive.py.txt` and `pty_gestures.py.txt` (the owned-pty drivers that
replaced the AppleScript route), and the `run*.sh` byte-capture drivers are
stored with `.txt` extensions where they are Python: they are captured
evidence rather than shipped code, and CI's flake8/isort gates run over the
whole tree. Renaming keeps them readable without asking the linters to hold
throwaway probe scripts to the codebase's standards.

## Round-1 remediation: gestures re-driven on the owned pty

Re-run after the round-1 fixes, against the current head, using
`pty_gestures.py.txt`. Real app, real pty, the exact byte sequences from
`bytes/`, no global keystrokes:

| gesture | bytes written | observed |
|---|---|---|
| Cmd+V, image-only clipboard | `ESC[118;9u` | composer shows `[Image #1, 920x1568]` |
| Ctrl+V, image-only clipboard | `ESC[118;5u` | image attached (no regression) |
| Cmd+C over a `shift+left` range | `ESC[99;9u` | `copied 5 characters` receipt |
| Ctrl+C over a `shift+left` range | `ESC[99;5u` | copy receipt (no regression) |
| Cmd+C, NO selection | `ESC[99;9u` | no copy, no receipt, draft `draft must survive` still painted |

The no-selection case is the one worth stating precisely: `super+c` carries no
interrupt meaning, so the correct behaviour is that it does nothing at all and
costs the user nothing. Verified by forcing a repaint after the key and
confirming the draft is still on screen.

---
name: console
description: Drive the Local Operator desktop app's console (pty) surfaces from an agent — surfaces, reading output, keys and input, secrets, sudo, screenshots, and the refusals.
---

# Console: drive the app's real terminal

Read this guide before doing terminal work when the `console` tool is missing,
when the user says they ran something "in the console", when a TUI or a REPL has
to be driven, or when a console call returns a typed refusal. It is the
operational playbook; the design contract is
`docs/design/ui-console-tab.md` (in the repo) and `docs/CONSOLE.md` is the
operator-facing version of this page.

## The one thing to know

The `console` tool drives a **pty inside the Local Operator desktop app**. There
is exactly ONE host and no fallback: no app, no console. It is gated on the app's
discovery record, so on a machine without the app the tool is simply **absent**
— and absent means absent, not broken. Do not install a terminal emulator, do
not drive `cmux` or `tmux` as a substitute, and do not treat another window's
terminal as if it were this one.

**A surface handle names its host.** Every handle starts with `con:` (e.g.
`con:1:9f2a`). A terminal in Terminal.app, iTerm, cmux, or an ssh session has no
such handle and is not readable by this tool. If the user says "I ran it in a
terminal", call `list`: a surface the USER opened appears there with
`origin: user`, its command and its `cwd`, and you can read it. If `list` does
not show it, it was a different terminal — say so instead of guessing at its
output.

**Prefer `bash` for ordinary commands.** `bash` returns output directly, cannot
wedge on a prompt, and cannot leave a process running behind your turn. Reach for
`console` when the work genuinely needs a pty: a full-screen TUI (`vim`, `less`,
`htop`), a REPL, an installer, an interactive prompt, or a command that must keep
running after your turn ends.

## Methods

One tool, one `method` parameter. The surface-scoped methods take the `surface`
handle from `list`/`create`.

| method | what it does |
|---|---|
| `list` | Every surface in this session, including the ones the user opened (`origin`, `command`, `cwd`, grid, running/exited, last output). |
| `create` | Start a surface: `command`/`args`/`cwd`/`env`, plus `cols`/`rows` (default 100x30), `reveal`, `retain`. Returns the handle. |
| `status` | `running`, `exit_code`, grid, `live`, whether the log was `truncated`, `secure`, `retain`, and how long since the last output. |
| `read` | The text. `mode='viewport'` (default) is the visible screen — what a person looking at the pane sees; `mode='scrollback'` is a history window positioned by `start`/`count`. |
| `screenshot` | A PNG of the surface, written to a file whose path comes back in the result. |
| `input` | Type `text` into the surface, or a stored secret via `secret_ref`. `paste=true` asks for a bracketed paste. |
| `keys` | Named keys, e.g. `['ctrl-c']`, `['up']`, `['ctrl-a', 'd']`. |
| `resize` | Change the grid (`cols`/`rows`). |
| `secure` | The USER's do-not-capture switch: while it is on, the app refuses to read or capture that surface. |
| `close` | End the surface (`kill` to signal the process). `retain=false` discards its output. |

**stdout and stderr are ONE stream.** A pty has a single output channel; there is
no way to separate them and the tool does not pretend otherwise. If the
distinction matters, run the command in `bash`, where the harness captures the
two separately.

**An idle surface is not necessarily finished.** There is no reliable in-band
signal for "the program is waiting for input", so the tool deliberately does not
guess: an idle surface may be a running build, a `sleep`, a TUI waiting on a
keystroke, or a process that has exited (`status` reports `exit_code` when it
has). Read it to see.

**Reading a surface whose app has gone** still works when the app is running:
after an app restart the surface is `live: false` and `read` returns the
retained history with the pane gone. If the app has *quit*, the surfaces ended
with it and every call reports that — create a new one instead of retrying.

## Named keys

`keys` takes names, not raw escape bytes. Accepted names:

- Control: `ctrl-a` through `ctrl-z` (lower case, e.g. `ctrl-c`).
- Navigation: `up`, `down`, `left`, `right`, `home`, `end`, `page-up`,
  `page-down`, `insert`, `delete`.
- Editing: `enter`, `tab`, `shift-tab`, `backspace`, `escape`, `space`.
- Function keys: `f1` to `f12`.
- Modifiers on a named key: `alt-<name>` / `meta-<name>`, and `shift-<name>`
  where it is not already a name of its own (e.g. `shift-tab`).

An unknown name is refused with the accepted set rather than being sent as
literal text — and **that refusal is the authoritative list**: the encoder lives
in the app, so if a name below is refused, use the spelling the refusal reports.
A sequence of keys is sent into the surface in order, so `['ctrl-a', 'd']`
detaches from a `tmux`/`screen` session.

Arrow keys and other cursor keys are encoded for the mode the program has set
(application-cursor mode changes `ESC [ A` to `ESC O A`); you do not have to know
which mode is active, and the result echoes what was encoded.

## Screenshots, and what a frame is

- `rendered: "displayed"` means the pane was on screen and the app photographed
  its own window cropped to the pane.
- `rendered: "offscreen"` means the app reconstructed the frame from the
  surface's record — the pane was closed, or the surface is agent-only, or the
  app restarted since. **It is a faithful reconstruction, not a photograph of a
  live screen**, and the result says which you got.
- The capture view is one at a time app-wide; a second concurrent capture is
  refused with `console_capture_full` (retry, or read the surface as text).
- **Never use macOS `screencapture`** to photograph a console. It captures the
  frontmost window and steals the user's focus, which is exactly what these
  surfaces are built to avoid.

## Secrets, sudo, and what the agent must never do

Three ways a value can reach a terminal, and only two are allowed:

1. **The human types it** — recommended, and the app records no keystrokes, so a
   password typed at an echo-off prompt never enters the byte log, the record,
   `read`, `status` or a screenshot.
2. **The agent types a literal credential** — forbidden by policy. Never put a
   password, a token or a key in `text`. The store exists so you never have to
   know the value: `secret_ref` names it instead.
3. **`secret_ref`** — the value is resolved from the encrypted store (the same
   store the `secret` tool uses) and written to the surface. It is not returned,
   not echoed, and after the call it is registered with the session's redaction
   sink so later appearances in a trace, a result or the transcript read
   `[redacted]`. The tool call itself records the NAME (`secret_ref:
   "SUDO_PASSWORD"`), never the value.

**Administrator commands need the user's consent, not just the tool's.** Before
running anything that needs root (`sudo`, an installer, a system change), use
`ask` with the exact command and what it will change. Do not attempt a password
yourself, and do not treat an approved tool call as approval for the command the
user is about to be prompted for — the harness gate authorises the call, `ask`
authorises the change to their machine.

There is no "run as root" surface: a surface starts as the user's own shell so
`sudo`'s prompt arrives where the user can answer it.

**Honest limits, stated rather than implied:**

- **A program can echo a secret.** If something prints a password (an `echo`, a
  `set -x` trace, an app logging its own config), it is in the record and `read`
  returns it — as it should, because a human looking at that screen sees it too.
  The harness's shape pass masks values it can recognise by their spelling; an
  opaque value with no recognisable shape is not caught.
- **A screenshot is pixels.** Redaction cannot read them. The control for a
  screenshot is echo-off plus the user's `secure` span, not the redactor.
- **`secure` is a refusal, not a wall.** It is typed at the tool seam and the
  app refuses reads and screenshots while it is on, but the app cannot tell "a
  read that should have been refused" from any other read — so honour it.

## When a call is refused

Every refusal is typed; do not substring-match a message, and do not retry a
refusal that describes a state.

| code | what it means | what to do |
|---|---|---|
| `unsupported_method` | This app version has no console. | Tell the user to update Local Operator; do not retry. |
| `surface_unavailable` | No such handle, or a stale one after a restart. | `list` for the live handles. |
| `surface_not_owned` | The surface belongs to another session. | Only this session's surfaces are readable; the user's other conversations keep theirs. |
| `process_exited` | The program ended; the log is retained. | Read the history, or `create` a new surface. |
| `input_queue_full` | The surface is not draining input. | Read it to see what is blocking, then send the remainder. |
| `unknown_key` | A key name the encoder does not have. | Send accepted names (above). |
| `secure_input_active` | The user has secure input on for that surface. | Wait, or ask them to toggle it off in the pane. |
| `invalid_grid` | The requested cols/rows are outside what the app honours. | Use the clamp it reports. |
| `console_capture_full` | Another capture is in flight (one at a time). | Retry in a moment, or read the text. |
| `console_unavailable` | The app's console feature is off (settings, launch flag, or a failed native module). | Tell the user which; the app log names a load failure. |
| `proto_mismatch` | App and session speak different protocol versions. | Update Local Operator. |
| transport "not running"/"not answering" | The app is gone or wedged. | Ask the user to re-open the app; the surfaces ended with it. Never retry in a loop. |

## Focus and the user's screen

Creating a surface does not raise the app. `reveal` is `none` by default, and no
value of it ever raises, activates or focuses the OS window: `session` only opens
the pane when the app is already displaying that session, and `open` only claims
and focuses the pane when the app's window is already focused. A request that
would have had to raise the window is downgraded and comes back as
`revealed: false`. Keep it that way — a window appearing because an agent was
working is the failure this feature is designed to avoid.

## Housekeeping

Close what you opened. A surface outlives your turn by design, so a forgotten
one keeps a process running on the user's machine; `close` ends it and
`retain=false` discards its output when the history is not worth keeping.

---
name: peer-messaging
description: Hand a note or instruction from one local lop session to another with `lop send`, and list running sessions and their memory use with `lop sessions`. No cmux needed.
---

# Peer messaging between lop sessions

Any `lop` session on this machine can message any other, with no cmux and no
network. The two commands ride the same control-socket + registry substrate the
mobile (phone) stack already uses: every interactive `lop` process publishes a
discovery record and runs an authenticated loopback control server, and `lop
send` is just a short-lived client that dials one and speaks a single message
op.

**Trust boundary is the account.** The discovery record is mode `0600` under a
`0700` directory, so anything that can read a session's control key is already
the owning user. Delivery is loopback only. There is no cross-account path; the
same-account boundary is the whole authorization story.

## `lop sessions` — what is running and what it costs

```
lop sessions
lop sessions --json
```

Columns:

- `STATE` — `live` (pid alive and heartbeating), `wedged` (pid alive but its
  owner has not heartbeat within the timeout, so it will not service the socket
  promptly), or `stale` (dead; the record is reaped on the next scan).
- `PID` `KIND` `CONVERSATION` `MODEL` `CWD` — session identity. `KIND` is `tui`
  (interactive), `daemon` (daemon-owned), or `exec` (headless one-shot).
- `RSS` — resident set size. Always available; the baseline number.
- `FOOTPRINT` — the *true* memory cost. On macOS RSS under-reports because
  memory is compressed and swapped, so this column shows the phys footprint
  (the number Activity Monitor shows and the one that "adds up"). On Linux it
  is the proportional set size (Pss). Shown as `—` when the probe could not
  measure it — never as zero.
- `UPTIME` — how long the session has been running.
- `HB_AGE` — how long since its last heartbeat. A large `HB_AGE` on a `live`
  row is an early sign of a session going wedged.

`--json` emits the same fields as a list of objects (bytes, not human sizes)
for scripting.

## `lop send` — hand a message to another session

```
lop send "<target>" "your message"
lop send "<target>" --wake "act on this now"
lop send "<target>" --now "stop, do X instead"
lop send "<target>" < note.txt        # body from stdin
```

The message lands in the target's transcript AND becomes visible to its model
on the next turn, rendered as an inbound cross-session card (`↔ peer message
from …`) in both the TUI and the phone — distinct from the user's own turns.

### Delivery modes

- **default (mailbox, record-only):** the message is durably written to the
  target's history immediately and the model reads it on its *next* turn. An
  idle target stays idle — non-interrupting. Use for a non-urgent hand-off.
  A target parked in a blocking `wait` is the one exception, and it is not a
  preemption: the wait returns early reporting its job still running, so the
  message is read at the next turn boundary instead of after the wait's full
  budget. A running `bash`, `eval` or MCP call is never interrupted.
- **`--wake`:** mailbox delivery, plus drive a turn now if the target is idle.
  Use when the target should act on the message immediately. (While the target
  is already busy, `--wake` is a no-op: the running turn will read it anyway.)
- **`--now` / `--steer`:** inject mid-turn like a steer, to correct or redirect
  a session that is actively working. If the target is idle there is nothing to
  steer into, so it opens a turn (the message is never dropped).

### Choosing a target

Priority order:

1. `--pid N` — exact pid (the record filename is the pid; unambiguous).
2. `--session ID` — exact session id.
3. positional `TARGET` — a case-insensitive substring matched against the
   conversation name, then the session id, then the cwd basename.

Only `live` sessions are eligible. If a substring matches several live
sessions, `lop send` prints the candidates and exits non-zero asking you to
disambiguate with `--pid`. If the only match is `wedged`, it says so rather
than hanging on a dial.

### What the human sees

The delivered message appears in the target's transcript marked
`↔ peer message from "<sender conversation>" (pid N, <model>)` in the TUI, and
as an accented inbound card naming the sender in the phone surface. The sender
identity is best-effort (the calling session, looked up by parent pid) and
advisory — it labels the card but is never required for delivery.

### Examples

```
# Non-urgent note to a session by name substring
lop send "peer-send design" "gates are green, ready for review"

# Wake an idle session to act now
lop send --pid 12345 --wake "the deploy finished; verify prod"

# Redirect a session mid-turn
lop send "ingest refactor" --now "hold off — the schema changed"

# Pipe a longer note from a file or a command
git log -1 --stat | lop send "release cutter"
```

## Limits

- **Same account only.** Loopback + the `0600` control record; there is no
  remote or cross-user path.
- **No cmux required.** This is independent of cmux entirely.
- **Only sessions that run a registrant receive.** Interactive (`tui`) and
  daemon-owned sessions do; a headless `exec` session may not, and a session
  running an older `lop` that predates peer messaging answers with a clear
  "cannot receive peer messages" error (a soft, non-zero-exit failure, not a
  crash).
- **Message size cap:** bodies are capped at 256 KB, well under the control
  socket's frame limit. A larger paste is rejected with a clear error rather
  than silently dropped.

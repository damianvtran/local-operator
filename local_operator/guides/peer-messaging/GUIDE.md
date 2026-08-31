---
name: peer-messaging
description: Message another local lop session — the in-session `send` tool for agents, `lop send` for humans — and list running sessions with `lop sessions`. No cmux needed.
---

# Peer messaging between lop sessions

Any `lop` session on this machine can message any other, with no cmux and no
network. Both entry points ride the same control-socket + registry substrate the
mobile (phone) stack already uses: every interactive `lop` process publishes a
discovery record and runs an authenticated loopback control server, and a send
is just a short-lived client that dials one and speaks a single message op.

Two entry points, one wire:

- **The `send` tool** — the way an AGENT messages a peer from inside its own
  session (below). Wake defaults ON, so an idle peer responds right away.
- **`lop send`** — the shell command for HUMANS (further below). Mailbox by
  default; `--wake` opts in to waking.

They share the same target resolution and delivery semantics; the only
difference is the default and who runs them.

## The `send` tool — agent-facing, in-session

An agent messages a peer with the `send` tool, not by shelling out to `lop
send`: the tool call is its own auditable card, and its defaults are tuned for
agent-to-agent hand-offs.

Parameters:

- `target` — case-insensitive substring of the conversation name, session id,
  or cwd basename (`lop sessions` lists what is running).
- `pid` — exact pid (unambiguous; the disambiguation error lists these).
- `session` — exact session id.
- `message` — the body; it lands in the peer's transcript as an inbound
  cross-session card.
- `wake` (default **true**) — mailbox mode: wake an idle peer so it responds
  right away. `wake=False` is the quiet drop: the peer reads it on its next
  turn and stays idle.
- `now` — steer the peer mid-turn instead of using the mailbox; opens a turn
  if the peer is idle.

**Agent sends wake idle targets by default.** A peer parked on a scheduled
wake or an idle turn answers the moment the message lands — there is no
"it will notice next time it wakes" gap. Pass `wake=False` deliberately for a
non-urgent hand-off the peer may fold into whatever it does next.

The result echoes the receive side's own detail string, so the sender knows
exactly how the peer took it:

- `delivered and woke the session` — mailbox + wake, peer was idle.
- `delivered to the mailbox (will be read on the next turn)` — quiet drop, or
  the peer was already busy (its running turn reads it anyway).
- `delivered mid-turn (steered)` — `now=True` while the peer was working.
- `delivered (opened a turn)` — `now=True` while the peer was idle.

A session cannot send to itself (the tool refuses), and an ambiguous `target`
returns the candidate list asking for a `pid`.

## `lop sessions` — what is running and what it costs

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

## `lop send` — the shell command (humans)

The same wire, driven from a terminal. Mailbox by default — a human sending
from a shell usually wants the non-interrupting drop; `--wake` opts in to
waking an idle session.

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
identity is best-effort and advisory — it labels the card but is never required
for delivery. The CLI looks the session up by its PARENT pid (`lop send` is a
child of the TUI that spawned it); the `send` tool runs inside the session
process and looks itself up by pid. Both name the same sender.

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

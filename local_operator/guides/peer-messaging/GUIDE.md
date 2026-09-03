---
name: peer-messaging
description: Message another local lop session — agents use the `send` tool (never a shelled `lop send`), humans use `lop send` — and list sessions with `lop sessions`. No cmux needed.
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
- **`lop send`** — the shell command for HUMANS at a terminal (further below).
  Mailbox by default; `--wake` opts in to waking.

They share the same target resolution and delivery semantics; the only
difference is the default and who runs them.

> **Agents: use the `send` tool. Never `bash` a `lop send`.** Three reasons,
> and they are why the rule is worth remembering rather than looking up. The
> tool call is its own **auditable card** — a delivery buried in a shell trace
> is hard to review and easy to miss in an approval prompt. The tool **wakes by
> default**, where the CLI's mailbox default leaves an idle peer sitting on the
> note until something else wakes it. And the tool prompts at the **write
> tier**, like every other capability that can start autonomous work in another
> process. The shell command below is documented for the human at a terminal,
> not as an agent recipe.

**Trust boundary is the account.** The discovery record is mode `0600` under a
`0700` directory, so anything that can read a session's control key is already
the owning user. Delivery is loopback only. There is no cross-account path; the
same-account boundary is the whole authorization story.

## The `send` tool — the agent entry point

This is how an agent messages a peer. It runs inside the session process, so it
resolves the target, names the delivery in its own card, and returns the
receive side's own detail string.

Parameters:

- `target` — case-insensitive substring of the conversation name, session id,
  or cwd basename (`lop sessions` lists what is running).
- `pid` — exact pid (unambiguous; the disambiguation error lists these).
- `session` — exact session id.
- `message` — the body; it lands in the peer's transcript as an inbound
  cross-session card. Required, and non-empty.
- `wake` (default **true**) — mailbox mode: wake an idle peer so it responds
  right away. `wake=False` is the quiet drop: the peer reads it on its next
  turn and stays idle (the TUI card shows this mode as `quiet`). Ignored when
  `now=True`.
- `now` (default **false**) — steer the peer mid-turn instead of using the
  mailbox; opens a turn if the peer is idle.

Addressing precedence is `pid`, then `session`, then the `target` substring —
supply one. `pid` is the form to reach for when a substring could match more
than one session.

### Worked examples

A non-urgent hand-off the peer folds into whatever it does next — it stays
idle and reads this on its next turn:

```
send(target="peer-send design", message="gates are green, ready for review", wake=False)
→ delivered to the mailbox (will be read on the next turn)
```

Something the peer should act on now. This is the default, so `wake` needs no
spelling out:

```
send(target="release cutter", message="the deploy finished; verify prod")
→ delivered and woke the session
```

The same, addressed unambiguously by pid — the form the disambiguation error
hands you:

```
send(pid=64190, message="the deploy finished; verify prod")
→ delivered and woke the session
```

Redirect a peer that is actively working, before it goes further down a wrong
path:

```
send(target="ingest refactor", message="hold off — the schema changed", now=True)
→ delivered mid-turn (steered)
```

**Sends wake idle targets by default.** A peer parked on a scheduled wake or
an idle turn answers the moment the message lands — there is no "it will
notice next time it wakes" gap. Pass `wake=False` deliberately when the note
is not urgent.

### What the result tells you

The result echoes the receive side's own detail string, so the sender knows
exactly how the peer took it:

- `delivered and woke the session` — mailbox + wake, peer was idle.
- `delivered to the mailbox (will be read on the next turn)` — quiet drop, or
  the peer was already busy (its running turn reads it anyway).
- `delivered mid-turn (steered)` — `now=True` while the peer was working.
- `delivered (opened a turn)` — `now=True` while the peer was idle.

Refusals are answers too, and each names its own fix:

- An ambiguous `target` returns the candidate list — `2 sessions match; retry
  with pid=<n>:` followed by one `pid=<n>` line per session.
- No match returns `no live session matches '<target>'`.
- A session cannot send to itself; the tool refuses before it dials.
- A body over the 256 KB cap is rejected with the measured size, not truncated.
- If the ack is lost after the peer committed the message, the tool says the
  message **may or may not** have arrived rather than claiming failure — check
  with the peer instead of resending, or it lands twice.

## `lop sessions` — what is running and what it costs

```
lop sessions
lop sessions --json
```

The table prints `STATE PID KIND CONVERSATION MODEL RSS FOOTPRINT UPTIME
HB_AGE`:

- `STATE` — `live` (pid alive and heartbeating), `wedged` (pid alive but its
  owner has not heartbeat within the timeout, so it will not service the socket
  promptly), or `stale` (dead; the record is reaped on the next scan).
- `PID` `KIND` `CONVERSATION` `MODEL` — session identity. `KIND` is `tui`
  (interactive), `daemon` (daemon-owned), or `exec` (headless one-shot).
  `CONVERSATION` falls back to the session id when the session has no name yet.
- `RSS` — resident set size. Always available; the baseline number.
- `FOOTPRINT` — the *true* memory cost. On macOS RSS under-reports because
  memory is compressed and swapped, so this column shows the phys footprint
  (the number Activity Monitor shows and the one that "adds up"). On Linux it
  is the proportional set size (Pss). Shown as `—` when the probe could not
  measure it — never as zero.
- `UPTIME` — how long the session has been running.
- `HB_AGE` — how long since its last heartbeat. A large `HB_AGE` on a `live`
  row is an early sign of a session going wedged.

**The table does not show a session's `cwd`.** `cwd` and `session_id` are
`--json`-only fields — which matters because `target` matches against the cwd
basename, so `--json` is where you look to see what a substring will match:

```
lop sessions --json
```

It emits one object per session with `state`, `pid`, `kind`,
`conversation_name`, `session_id`, `model_label`, `cwd`, `rss_bytes`,
`footprint_bytes`, `uptime_s` and `heartbeat_age_s` (bytes and seconds, not
human-formatted sizes) for scripting.

## `lop send` — the HUMAN entry point

The same wire, driven from a terminal by a person. Mailbox by default — a human
sending from a shell usually wants the non-interrupting drop; `--wake` opts in
to waking an idle session.

**This section is not an agent recipe.** An agent inside a session uses the
`send` tool above; shelling out to this command buries the delivery in a shell
trace and takes the mailbox default, which leaves an idle peer sitting on the
message.

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

**Address the session exactly one way.** A `--pid`/`--session` selector already
names the recipient, so a positional alongside one is the MESSAGE, not a second
address:

```
lop send --pid 12345 "the deploy finished; verify prod"          # body
lop send --pid 12345 --wake "the deploy finished; verify prod"   # body
lop send "release cutter" "the deploy finished; verify prod"     # target + body
```

Passing a positional *target* **and** a selector names two different sessions,
so it is refused rather than resolved — the selector would otherwise win
silently and deliver to a session the command does not appear to name:

```
$ lop send "release cutter" "gates are green" --pid 12345
ambiguous recipient: 'release cutter' and --pid 12345 name different sessions.
Drop one — `lop send --pid 12345 'gates are green'` to address by pid, or
`lop send 'release cutter' 'gates are green'` to address by name
```

Nothing is delivered there and the exit status is 1. `--pid` and `--session`
are mutually exclusive too; argparse rejects that pair at parse time. When a
substring matches several sessions, `lop send` lists them and asks you to
**replace** the target with a `--pid` — appending the flag to the command you
just typed produces exactly the refused form above.

A blank selector (`--session ''`) is an error rather than a silent fallback to
substring matching; drop the flag if you meant to address by name.

**Pipe the body when it is long** or when it comes out of another command — not
because the argument form does not work. Both forms are fully supported:

```
echo "the deploy finished; verify prod" | lop send --pid 12345 --wake
git log -1 --stat | lop send "release cutter"
```

**Do not pipe a body and type one at the same time.** With a selector, a
positional and a piped body are two candidate messages, so `lop send` refuses
rather than picking a winner — silently discarding the payload you did not get
is worse than a retype:

```
$ git log -1 --stat | lop send "release cutter" --pid 12345
ambiguous body: 'release cutter' and the piped input both look like the message.
Drop one — `lop send --pid 12345` to send the piped input, or
`lop send --pid 12345 'release cutter'` with nothing piped to send 'release cutter'
```

Without a selector both positional slots are filled, so `lop send NAME "body"`
with a pipe is unambiguous and the typed body wins.

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
lop send "release cutter" --wake "the deploy finished; verify prod"

# Wake an idle session addressed by exact pid — with a selector, the single
# positional IS the body
lop send --pid 12345 --wake "the deploy finished; verify prod"

# Refused: a positional target AND a selector name two different sessions
lop send "release cutter" "gates are green" --pid 12345   # exit 1, nothing sent

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

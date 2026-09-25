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
- No match returns `no session matches '<target>' (searched live and stored
  sessions)` — a name only a closed session had still resolves, so a miss
  means neither the running fleet nor the store answered to it.
- An unengaged session (`/new`, nobody has sent a message in it yet) is refused
  rather than written to: *"… has not been engaged yet (no user message has
  been sent in it), so it cannot receive peer messages — its owner has to send
  a first message"*. Its owner's first message makes it a recipient again, and
  the refusal is the same sentence whichever layer answers — including the
  receiving session, so an older peer's send fails the same way.
- A session cannot send to itself; the tool refuses before it dials.
- A body over the 256 KB cap is rejected with the measured size, not truncated.
- If the ack is lost after the peer committed the message, the tool says the
  message **may or may not** have arrived rather than claiming failure — check
  with the peer instead of resending, or it lands twice.

## Switching another session's model

The same `send` tool switches a peer's model: pass `model="<provider>/<model-id>"`
instead of `message`. Humans use `lop model` (below). Both have `/model`'s
semantics in that session: the switch lands at its **next provider call**, so a
call already streaming finishes on the old model and every later call, including
the rest of a running turn, uses the new one.

```
send(target="experiment 1", model="deepseek/deepseek-flash")
switched to deepseek/deepseek-flash (was anthropic/claude-opus-5)
its next turn runs on it
→ experiment 1 (pid 64190)
```

The result is the target's own answer, outcome first, in short lines, with the
address last in `lop send`'s grammar. `lop model` prints the same lines.

```
lop model "experiment 1" deepseek/deepseek-flash
lop model --pid 64190 deepseek/deepseek-flash
lop model --session 7661a465019b deepseek/deepseek-flash
```

Rules:

- **Exactly one of `message` or `model`.** Passing both is refused (send the
  note in a second call); `now=True` does not apply to a switch, and `wake` is
  ignored.
- **The target validates, not the sender.** The pair is checked against the
  TARGET's config, catalogue and credentials: a known provider, a model its
  catalogue serves (an aggregator or local endpoint that cannot be listed
  offline is accepted), and a credential it can run on. A refused pair changes
  nothing.
- **Live, engaged sessions only.** A closed or stored session is not switched
  remotely: open it and use `/model`, or `lop --resume <id> --hosting <p>
  --model <m>`. An unengaged `/new` is refused as for a message. A live session
  whose new name has not reached its record yet is still found, through its id.
- **A pinned fallback is not "already on".** If a provider fallback is serving
  the model you ask for while the session's selection is another model, the
  switch still runs and makes it the selection, withdrawing the fallback — as
  `/model` does.
- **The model is a positional.** `lop model <target> <provider>/<model>`;
  `--model`/`--hosting` are refused there with that correction.
- **Effort is not carried.** A TUI target applies its own effort choice; a
  runtime-owned target uses the model's default level.
- **Children keep their model.** A child's model is decided when it starts or
  resumes. An inheriting child takes the parent's model at that moment; a
  pinned child re-resolves its tier. Switching the parent — `/model`, the
  phone, or another session — does not change children already running; pause
  and resume one to move it. The result counts them.
- **The card says who, never why.** `model` and `message` are exclusive, so
  follow a switch with a one-line `send` note when the owner should know the
  reason.
- The approval prompt reads `switch <target>'s model to <p/m> (changes that
  session's billing)`.

What the result tells you (the first line is the outcome; the send card's
collapsed row shows it as `switched`, `no change` or `pending`):

- `switched to <new> (was <old>)` / `its next turn runs on it` — the target was
  idle.
- `switched to <new> (was <old>)` / `mid-turn: the call in flight finishes on
  the old model` — the target was waiting on a provider call; every later call
  uses the new one. `the current step` instead of `the call in flight` when it
  was in a tool or an approval.
- `back on <model> (was on fallback <fallback>)` — you asked for the model the
  session had selected while a provider fallback was serving instead; the
  fallback was dropped.
- `<n> running subagent(s) stay(s) on the old model; new and resumed ones
  switch` — added when the target has running children.
- `already on <new>` / `nothing changed`.
- `pending: switch to <new> accepted` — a local-setup provider on a TUI target;
  it applies once that session's capacity check finishes, and may still fail.
- `switched to <new> (was <old>)` / `with an error after the switch: …` — the
  switch is in force, but a later step of it raised. The send row shows
  `switched (error)` with the warning glyph.
- `refused: <reason>; still on <old>` — nothing changed (non-zero exit for
  `lop model`). A switch that did not take is reported this way even when a
  fallback happens to be serving the requested model already.
- `older lop: it cannot switch models remotely; nothing changed — update it
  (lop update) or run /model in that session`.
- `could not reach that session; nothing changed (…)` — the socket never
  opened.
- `no answer — the switch may or may not have landed; check lop sessions
  before retrying` — the op was sent and no answer came back within 15 s.

The target's transcript records the switch twice: the usual `[model switch]`
notice, and a peer card whose header names the sender and whose body reads
`[remote model switch] now on <new> (was <old>)`. The new model leads because on
resume this card is the only trace of the switch. A `lop model` from a plain
terminal is the sender `terminal`, and the body ends `— from a terminal in
<dir>`; one run inside a lop session names that session. A pending local switch
writes `switch to <new> requested (on <old> until it applies)` instead. The card
is record-only; it does not start a turn.

## `lop sessions` — what is running and what it costs

```
lop sessions
lop sessions --json
```

The table prints `STATE PID KIND CONVERSATION MODEL RSS FOOTPRINT UPTIME
HB_AGE`:

- `STATE` — `live` (pid alive and heartbeating), `wedged` (pid alive but its
  runtime has not sent a heartbeat within the timeout, so it will not service
  the socket promptly), or `stale` (dead; the record is reaped on the next
  scan).
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

**Closed sessions are still reachable.** `lop sessions --all` adds `stored`
rows — sessions that are not running — shown with `—` for RSS/UPTIME, sorted
newest-first, capped by `--limit` (default 50). `send`'s `target` falls back
to stored-session names when no live session matches: `wake=True` engages a
runtime, a quiet mailbox drop spools to the inbox for the next open, and a
steer on a stored session behaves as wake. A stored session that never ran a
turn is not a recipient either — it is skipped, so a broadcast cannot spool a
note into a conversation nobody started, and an exact `--session` send to one
is refused with the reason above. A `NEEDS` row is a parked
question; idle-vs-busy is `--json`'s `busy`, not the table.

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

**A session that has not been engaged yet is not a recipient.** A fresh
`/new` window is already listed (`live`, with its record published) while its
owner is still composing the first prompt, and it is deliberately out of
reach until that first message is sent:

- an exact `--pid`/`--session` send is refused with the reason — *"… has not
  been engaged yet (no user message has been sent in it), so it cannot receive
  peer messages — its owner has to send a first message"* — and nothing is
  delivered or spooled. On a **stored** session (nothing running) the closing
  clause is worded for a window nobody can type into — *"it becomes a
  recipient once someone opens it and sends a first message"* — because a
  sender cannot open it;
- a substring/broadcast match skips such sessions, without falling through to
  a stored session that merely shares the name. A send that still reaches a
  recipient says how many were left out — `→ name (pid N): delivered to the
  mailbox (will be read on the next turn); 3 matches skipped (not engaged yet)`
  — so a partly-delivered broadcast is never reported as a clean success; when
  nothing else matched, the refusal says what it reached — the pid when a
  single session matched, otherwise the needle and the count — instead of the
  `no session matches` form;
- `/stop <target>` (and `lop stop <target>`) still resolves it — the kill
  switch names a session in order to stop it, so a composer window someone
  needs to end stays reachable.

The session becomes eligible the moment it runs its first turn: the owner's
first message flips the record's `started` bit, and the next send lands with no
special handling. The gate is enforced on the receiving side too, so a peer
running an older `lop` cannot deliver into an unengaged session either — the
delivery fails there with the same sentence rather than writing the row.

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
- **A session must be engaged before it receives.** One that has not run a
  real turn yet (a fresh `/new` with no message sent) is not a recipient: a
  send to it is refused with the reason above and a broadcast skips it, and it
  becomes eligible after its first turn. `/stop` is unaffected — it still
  reaches such a session.
- **Message size cap:** bodies are capped at 256 KB, well under the control
  socket's frame limit. A larger paste is rejected with a clear error rather
  than silently dropped.

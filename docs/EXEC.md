# Headless sessions (`lop exec`)

`lop exec --help` is the complete command reference. Exec runs an ordinary
persisted conversation without starting a terminal UI. A saved team is a real
attachment: the current session becomes its manager, with the saved roster,
manager instructions, collaboration and project briefs. Delegated work uses
that same team; the prompt is never rewritten into a simulated slash command.

```sh
lop exec 'Review the implementation' --team release --background
lop exec --tools read,mcp__vendor_screen 'Screen these names'
printf 'Summarize this report' | lop exec --profile reviewer
lop exec --goal 'Finish the acceptance checklist' --loop 3 --name 'Night audit'
lop exec --resume SESSION_ID --loop-goal 'All acceptance checks are verified'
lop exec --status JOB_ID
lop --resume SESSION_ID
```

## Startup options and precedence

| Option | Contract |
| --- | --- |
| `command` | Literal initial prompt. `-`, or omitted with piped stdin, reads stdin (surrounding whitespace is trimmed). An omitted prompt is allowed for a valid loop. Slash-looking text stays literal. |
| `--team NAME` | Resolve a saved team and attach it. Unknown names fail before model construction or a detached spawn, and are checked again by the worker. |
| `--profile NAME` | Attach a registered role, specialist or packaged starter, exactly like `/agent`. Unknown names fail preflight. A team owns the agent slot, so this flag cannot be combined with `--team`: with a team attached the attach is refused with the reason (`/team clear` detaches the team). |
| `--agent NAME`, `--agent-name NAME` | Existing legacy named-agent selection; creates a missing agent. Not the `/agent` role attachment. |
| `--agent-id ID` | Select an exact existing legacy agent; mutually exclusive with `--agent`. |
| `--goal TEXT` | Set the literal standing goal (`clear` is text, not a command). A goal alone does not start a model, and it is not sent as a message — see [Divergence from `/goal`](#divergence-from-goal). |
| `--clear-goal` | Explicitly clear a resumed standing goal; mutually exclusive with `--goal`. |
| `--loop N` | Run 1–25 continuation iterations **after** an optional initial prompt. Requires a new or resumed standing goal. |
| `--loop-goal TEXT` | Run continuation/judge iterations until achieved. No fixed iteration cap; repeated undecidable judge results fail safely. Mutually exclusive with `--loop`. |
| `--name TEXT` | Set the persisted conversation title. |
| `--tools NAMES` | Declare this run's whole reach: a comma-separated list of tools, and the only ones the session may reach — an excluded tool is unreachable by name, not merely unapproved, and delegated children inherit the bound. The declaration is **one-way** for the session's life (a second declaration may only tighten it; a host that needs a different set starts a session with it) and it is not persisted, so a later `--resume` without the flag is unrestricted. It also stands as the APPROVAL for the names it lists *where nobody can be asked* — a non-TTY run without `--control`; on a terminal, and under `--control`, every write/exec call is still put to the gate. Overrides an attached role's `tools:` allow-list, and inherits it when the flag is absent. A name this build does not have is unreachable, and reported at the end of the run. |
| `--effort LEVEL` | Set reasoning effort using the selected model's existing validation. Unsupported levels fail before a turn. |
| `--output-format FORMAT` | Enforce the format of the assistant's FINAL response for this run: `markdown`, `json`, `yaml` or `toml`. Bounded by contract: formats, optional schemas and a retry budget only — enforcement is validation plus bounded retry, never a re-encode, and the payload on stdout is the model's own validated bytes. Without this flag no enforcement happens and the run is byte-identical to before the flag existed. See [Final-response enforcement](#final-response-enforcement---output-format). |
| `--output-schema PATH` | A JSON file the decoded payload must satisfy: a JSON Schema for `json`/`yaml`/`toml`, or `{"required_sections": ["<section>", ...]}` for `markdown`. Read and validated before the run starts (a bad file fails as `exec failed: …`, exit 1, with no session directory behind it). Requires `--output-format`. |
| `--output-retries N` | Max retries after a rejected final response (0–5; default 2, so three attempts). Not persisted — each `--resume` invocation carries its own flags. Requires `--output-format`. |
| `--workstream` | Publish this run as a long-lived parallel WORKSTREAM the operator asked for: it is listed in the sidebar, `/resume` and the phone list, labelled with the session that opened it, and can be followed and steered. It changes **visibility only**, never the approval posture: every exec run already publishes its discovery record, so the row is followable and steerable without `--control`, and a `--tools` declaration still stands as the approval exactly as it does for an unflagged run. Add `--control` yourself when the run's gates should park for a supervisor. Only meaningful under an agent's shell — it is what selects the `agent-workstream` stamp below; at your own terminal nothing is stamped and the run is an ordinary session. Absent (the default, and what every caller that predates the flag gets) an agent-opened run is `agent-shell`: hidden everywhere and silent. |
| `--resume [ID]` | Reopen the same transcript; omit the ID to select the most recent session. A live headless runtime is refused rather than raced; `lop --resume ID` attaches the TUI to that runtime instead. |
| `--background` | Detach a worker. The launcher prints a bounded readiness receipt, not a claim that the work completed. |
| `--status JOB_ID` | Print the durable job record as JSON, including terminal outcome and canonical session ID. Does not run a model; cannot combine with a prompt or run options. |
| `--control` | Install supervisor approval/question gates. These may wait for an attached user. Runtime discovery and live TUI attachment work without this flag too. |
| `--yolo` | Explicit approval override; never implied by teams, loops, attachment or background execution. |
| `--json` | Emit one JSON object per agent event on stdout, carrying the session ID. Notices and receipts use stderr. |
| `--hosting`, `--model` | Existing provider/model selection; ordinary model-resolution precedence is unchanged. |
| `--run-in DIRECTORY` | Change the working directory before executing or spawning the worker. |
| `--debug` | Enable verbose CLI logging (launcher/foreground process). |
| `-h`, `--help` | Show every exec flag and startup examples without executing a task. |
| `--train` | Existing legacy agent-directory history mode (or an autosave agent when unnamed), not needed for normal session persistence. `--resume` wins over its directory selection. |

This is deliberately a bounded startup interface, not an
arbitrary slash-command interpreter: UI commands and configuration-mutating
commands are not startup flags.

Resume restores the conversation, team, profile, goal and title first. Explicit
startup options override only their own slots. A team and a profile cannot both
be in force: a team OWNS the agent slot (see `guide://teams`), so attaching one
makes that team's manager the session's speaker and refuses a `--profile` or
`/agent` attach until the team is detached. A sidecar written before this rule
that named both comes back as the team's manager. Stored loop progress remains
visible, but restarting or resuming never automatically replays iterations.
Pass a new loop option explicitly to start another loop.

## Final-response enforcement (`--output-format`)

`lop exec` can require the assistant's FINAL response to be a machine-readable
payload and fail the run (exit 1) when it is not. The check runs at the one
point a turn ends cleanly: a rejected response becomes one more model step
**inside the same turn** (the retry notice is an ordinary, persisted user
message, so a resumed transcript explains the continuation), and a run that
exhausts its retries fails loudly rather than printing something wrong.

```sh
lop exec 'Extract the invoice as JSON' --output-format json
lop exec 'Reply with YAML' --output-format yaml --output-retries 3
lop exec 'Draft release notes' --output-format markdown --output-schema sections.json
lop exec 'Parse the report' --output-format json --output-schema report.schema.json --background
```

**What is enforced, per format.** Tolerance is about *locating* the payload;
content is strict.

| format | decoder | strict about | tolerated / noted |
| --- | --- | --- | --- |
| `json` | `json.loads` | full strict JSON: no trailing commas, no single quotes, `NaN`/`Infinity` rejected | any JSON value (object/array/scalar) when the whole text or a fence is exact; a prose scan finds a bare `{`/`[`-anchored span |
| `yaml` | PyYAML safe loader | one document only; the decoded value must be a mapping or sequence (a bare scalar is prose, not a payload) | PyYAML semantics: JSON is a subset, `yes`/`on` are booleans, duplicate keys last-wins, dates decode to dates |
| `toml` | `tomllib` | TOML 1.0 as stdlib enforces it: duplicate keys rejected | trailing commas in arrays; the root is always a table, so the payload is always a mapping |
| `markdown` | none — the text IS the payload | non-empty; balanced fenced code blocks | everything else; markdown has no formal well-formedness |

**Where the payload is found (in this order, first decode that also satisfies
the schema wins):** a fence labelled with the target format; an
unlabelled/unknown-label fence; the whole text; and for `json` only, a bare
`{`- or `[`-anchored span in prose. A fence labelled with a *different*
recognised format is skipped — the label is an explicit claim, and the retry is
the remedy.

**Schemas.** For `json`/`yaml`/`toml`, `--output-schema` names a JSON Schema
file (Draft 2020-12; meta-schema checked before the run). For `markdown` it is
the one vocabulary markdown has: `{"required_sections": ["Summary", "Risks"]}`
— each named section must appear as an ATX heading, case-insensitively, in the
given order.

**Failure semantics.** A rejected attempt logs one dim stderr line
(`! output check failed (<format>, attempt <n>/<m>): <reason>`) and, when the
budget is spent, the run ends with exit 1 and the two stderr lines
`Error: final response did not satisfy the output contract (<format>) after N attempts: <reason>`
and `exec failed: the turn did not complete` (the existing headless failure
line, whose text comes from `exec_session`'s fallback unless a prior prompt
failure recorded a more specific reason). **stdout carries
nothing** for an exhausted turn — a wrong payload on stdout is worse than an
empty one for `… | jq` consumers. On success, stdout is the VALIDATED payload
span (the winning candidate, stripped): `lop exec --output-format json | jq .`
works even when the payload arrived inside a fence or surrounded by prose.

**Interactions.**

* `--json` gains one `output_validation` event per checked attempt
  (`ok`, `attempt`, `max_attempts`, `payload_text` on success); nothing else on
  the event stream changes, and an unenforced run emits none.
* `--loop` / `--loop-goal` enforce **every** turn; a turn that exhausts its
  retries fails the run and stops the loop.
* `--background` is the same request run elsewhere: the schema path is made
  absolute before the spawn, the worker re-validates the file, and a failed run
  reaches the ledger as `failed` (`lop exec --status JOB_ID`), with both stderr
  lines in the job log.
* `--resume` never persists enforcement: the flags of THIS invocation apply,
  and a later resume without them is unrestricted (like `--tools`).
* `--output-retries` counts retries **per turn** (default 2, so three attempts
  per turn); the range is 0–5.

## What may not start a session: an agent's shell

A `lop` invocation that descends from an agent's `bash` tool call
(`LOCAL_OPERATOR_AGENT_SHELL`, set by that tool on every command it runs) may not
open a session, with ONE allowance that follows the calling session's own tools.
The two entry points differ, and the asymmetry is deliberate:

* **`lop exec` is allowed when the calling session may delegate** — when its
  LIVE tool inventory holds `task`, which the `bash` tool reports to the command
  it spawns as `LOCAL_OPERATOR_AGENT_MAY_DELEGATE=1`. That is the case the
  operator asks for by asking for it: "spin up parallel sessions and fully
  delegate this work", or a fan-out of long-lived independent workstreams, is a
  shape a `task` child cannot be — a child is one prompt and ends. A session that
  does NOT hold `task` is refused exactly as before, because for that session
  `lop exec` is a way around a missing tool rather than a way to delegate.
  `task` in the inventory the session ACTUALLY has is the whole test — not the
  role's `delegate:` flag. What builds that inventory differs by path and the two
  can disagree: `--tools` narrows the list on both, and the SUBAGENT path
  additionally prunes `task` from a child whose role does not delegate
  (`harness.subagent._build_child_session`), while `exec --profile` derives its
  inventory from the role's `tools:` allow-list ALONE — the prune lives on the
  subagent path only. A `delegate: false` role that declares no `tools:` list
  (the packaged `coder` seed, for one) therefore keeps `task` in an `exec`-opened
  session, and that session IS admitted, correctly: it genuinely holds the tool,
  so the rule is satisfied. The asymmetry between the two paths predates this
  allowance. An absent marker reads as NO.
* **the interactive path stays refused for EVERY agent shell**, delegating or not
  (`lop`, `lop --resume ID`, `--tui`). An agent has no terminal, so what that
  path opens is a front end on the OPERATOR's screen — not the separate session
  the relaxation above is about.

`lop exec --status` starts nothing and is unaffected.

What a refused run prints:

```
exec failed: this `lop` invocation from inside an agent session cannot open one — the session it would start is a top-level conversation the operator never opened, listed in their session list and desktop sidebar as if they had, and running outside the job manager that lets this session see, steer, cancel and account for delegated work.
Delegated work is launched with the `task` tool. A session that does not hold `task` may not create subagents at all: do the work yourself, and say so with `hub` if the slice genuinely cannot be done alone — `hub` reaches the session that delegated to you and the brief travels in the message. Work that must happen later is not a child session's to arm — `wake` is pruned from every child — so a child routes it back to the session that delegated to it, while a session that holds `wake` arms it there itself.
```

The incident this answers (2026-09-18): a subagent owed a review round on a PR,
held no `task` tool to run it with, and reached for `lop exec --profile reviewer
--background`. Two sessions — `lo-1281-review` and `lo-1281-qa`, 7 ms apart —
appeared in the operator's sidebar as chats they had opened.

The other half of that incident is the role's allowance, and it is the half the
guard does not fix: whether a subagent may delegate at all is its ROLE's answer
(`delegate: yes`), and a subagent that holds `task` is expected to use it, at any
depth. A role that does not delegate — a `coder`, a `reviewer`, a `scout` — never
holds it, and is expected to do the work itself rather than route around the
rule. So a team brief that owes a review round to a slice gives that slice a role
that may delegate, or keeps the round with the session that delegates.

**When to use `lop exec` from inside a session.** Default to `task`. Reach for
`exec` when the USER has explicitly asked for separate top-level sessions —
independent workstreams that must outlive this turn, or a large fan-out of them —
and never as a way to get around a `task` tool this session does not hold. A
session without `task` does the work itself and reports a genuine blocker with
`hub`. `guide://agents` carries the same rule to the agent.

**The stamp, and why neither route is silent.** A session opened this way is
stamped `origin.json` = `agent-shell` (`ORIGIN_AGENT_SHELL`): it is a TOP-LEVEL
conversation by every other test the store applies — an ordinary session
directory with no origin — so without the stamp `is_user_session` reports that
the operator opened it, and the session list, the desktop sidebar and the
phone's history all offer it as their own work. The stamp is written for BOTH
routes (the escape hatch below, and an allowed delegating run), and only for a
session the run CREATES: a conversation it merely RESUMES is the operator's own
work and is left alone. It is not a lock — the session is on disk at
`<config>/sessions/<id>` and `lop sessions` lists the live run; the id is in the
run's own output, the OPERATOR reopens it with `lop --resume <id>`, and an AGENT
reaches it again with `lop exec --resume <id>`, because the bare `--resume` form
is the interactive path and that stays refused for every agent shell.

**The stamp has TWO values, and `--workstream` chooses the second.** Visibility
here is an opt-in allow-list (`resume.USER_ORIGINS`) and every listing in the
tree funnels through one scan and one predicate, so the value on disk is the
whole decision. `agent-shell` is the default and means *ephemeral*: hidden from
the sidebar, the `/resume` picker and the phone's history, and silent — the run's
own runtime raises no completion banner, because a session the operator's
listings hide must not put anything on his screen (the delegated run's
supervising session owns it, and the durable records keep it findable).
`--workstream` writes `agent-workstream` (`ORIGIN_AGENT_WORKSTREAM`), which is
registered as a USER origin: the run is listed everywhere a conversation of the
operator's own is, and it keeps its notifications.

A workstream marker also records WHO asked: `opened_by` = `{agent, label,
name, session}`, read at creation from the requesting session's directory (its
role and task label, when it is itself a delegated child) and published on the
desktop row as `{agent, label, session}` (`docs/DESKTOP_API.md`). That is the
other half of the 2026-09-18 incident: a machine-started row that read as the
operator's own conversation was the confusion, so a listed one carries its
owner. A member that cannot be read is `null` rather than guessed.

**Two answers the flag's semantics depend on.**

* An ephemeral-stamped session is NOT promotable later. `origin.json` is written
  once, at creation, and is immutable from then on — the same rule a subagent
  child and a fork obey — so a run started without `--workstream` stays hidden
  for its whole life. The remedy is to start it as a workstream, not to edit the
  marker: a hand-edited `origin.json` is a supported un-hide for the operator on
  their own store, but it is not a durable way to run one, and nothing in the
  product writes it after creation.
* `lop exec --resume <id>` of an agent-workstream session does NOT re-stamp
  (`created_here` is false) and therefore KEEPS its visibility — the row stays
  listed, with its opener, while the run continues. The operator's own
  `lop --resume <id>` of it works too, and the row stays listed either way: a
  resumed conversation is never re-marked in either direction, so this form can
  neither hide a listed session nor un-hide a hidden one.

**The escape, for tests and QA runs.** `LOCAL_OPERATOR_ALLOW_NESTED_SESSION=1`
waives the refusal for one invocation, on BOTH entry points. It exists because
testing evidence here comes from exercising the REAL CLI — including the TUI in
a pty — which a QA run of the front end itself cannot do through the guard. It is
deliberately NOT named in the refusal text (that text is model-facing and its job
is to route the reader to `task`/`hub`/`wake`; obscurity, not secrecy — this
file and `AGENTS.md` both name it), and script harnesses declare themselves with
`agent_shell.harness_child_env()` rather than copying the variable by hand. It is
not a silent equivalent either: a session it OPENS is stamped
`origin.json` = `agent-shell`, so it stays out of the `/resume` picker, the
desktop sidebar and the phone's list (a conversation it merely RESUMES is the
operator's own work and is left alone). The picker is a FILTERED VIEW, not the
store: that session is still on disk at `<config>/sessions/<id>` and
`lop --resume <id>` opens it, which is the route back for the run that forgot to
isolate (typed at the operator's own terminal — from inside an agent shell that
same form is refused, and the agent's route to a session it opened is
`lop exec --resume <id>`). The id is in the run's own output either way: `lop exec --background`
prints its receipt line, a foreground `lop exec` that reaches its runtime prints
`lop exec session: session_id=<id>`, and a pty-driven front end prints
`lop --resume <id>` when it exits. That front end creates the session directory
as it OPENS — the transcript lands on its first turn — so if the only thing to
hand is the store, the entry to resume is the newest one under
`<config>/sessions/` carrying a `transcript.jsonl`: an open-and-quit leaves a
directory holding only `origin.json`, which `--resume` refuses, and
`--resume @latest` reads the same user-filtered listing the picker does.
Isolating the run (`LOCAL_OPERATOR_CONFIG_DIR=<scratch>`) remains what keeps a
test off the operator's own store; the stamp is the seatbelt for the run that
forgets.

**What this does not cover**, stated so the rule is not read as a boundary:

* The marker is set by the `bash` tool alone. A subprocess spawned by the
  `eval` tool, or one started with `env -u LOCAL_OPERATOR_AGENT_SHELL`, does not
  carry it, so it is not refused. The same is true of the
  `LOCAL_OPERATOR_AGENT_MAY_DELEGATE` allowance: it is the session's answer about
  itself, reported by the tool that spawns the command and not a secret, so a
  shell that scrubs it looks like a session that may not delegate (which is the
  direction that FAILS CLOSED).
* `--workstream` is not a way around that guard: it selects the stamp on a run
  the guard already allowed, and the guard never consults it. A session that
  does not hold `task` is refused with or without the flag, and the interactive
  path stays refused for every agent shell.
* The attribution is a SNAPSHOT taken at creation, from the environment the
  `bash` tool signed. A requesting session that renames itself later does not
  re-write the row, and a run with no readable scratchpad path records `null`
  members rather than a guess.
* The guard lives at `cli.main`: a marked process that starts `lop serve`, or
  engages a runtime, mints sessions through the server/runtime composition root
  and is not refused there.
* A session's own front end opening a conversation for its user is deliberately
  exempt — the TUI restart, `/fork`'s new window and a notification click's
  terminal all drop the marker before they re-exec
  (`agent_shell.without_agent_shell_marker`), because those are the user's
gestures, not an agent's command. That helper drops BOTH markers: a restarted or
forked window is a SESSION, so it must not be handed its parent's delegation
allowance either.

## Divergence from `/goal`

The TUI's `/goal <text>` sets the objective **and** submits that text as an
ordinary user message, so a single Enter both records the goal and starts the
work. `--goal TEXT` deliberately does not: it only sets the standing goal.

The reason is that exec already has a message channel — the positional prompt,
`-`, or piped stdin. If `--goal` also submitted its text, then
`lop exec 'Do the thing' --goal 'Ship safely'` would send two messages for one
invocation, and there would be no way to set a goal without spending a turn.
So the goal is the objective, the prompt is the message, and a run with a goal
but no prompt and no loop is refused rather than silently starting a turn.

To get the `/goal` behaviour in one command, pass the same text both ways:

```sh
lop exec 'Finish the checklist' --goal 'Finish the checklist'
```

## Approvals and lifetime

Without `--control`, non-TTY approval requests remain denied under the existing
headless gate. Publishing a discovery record does not replace that gate with
an interactive one. With `--control`, work may park for a supervisor. Choose
`--yolo` only when you intend that override; it is not required for an
unattended run. Viewer attach/detach does not end the worker's work.

`--tools` is a reach bound and not an approval override, so it does not change
that: the declaration stands as the approval for the names it lists only where
nobody can answer — a non-TTY run without `--control`, which includes a
`--background` worker. On a terminal the per-call prompt remains, for every
declared write/exec call as much as any undeclared one: naming a tool says
which tools this run may reach, never that each command it is about to run has
been agreed to in advance.

A parked run stays `running` for as long as nobody answers it, so the status
names what it is waiting on: `--status` reports `"pending": "approval"`
alongside `"status": "running"`, and `lop sessions` shows the same thing in its
`NEEDS` column. That field is live state read from the run's runtime record,
not a durable ledger row, so it disappears once the gate is answered.

A foreground run exits 0 on success and nonzero on failure/cancellation. A
background launcher exits after readiness (or reports `starting` after a
bounded wait); its exit code is **not** the eventual execution result. Follow
`lop exec --status JOB_ID` for `starting`, `running`, `succeeded`, `failed`,
`cancelled` or `interrupted`, and inspect its log (the table below says which
stop is which). Worker termination disposes
the session before writing the terminal result. Abrupt death cannot truthfully
report success: status reconciliation compares the PID's process-start identity
and marks a proven-dead runtime interrupted. It never restarts a loop or cleans
up a successor's resources.

### Why a worker stopped: `stop_class` and the stop artifacts

Exit code 130 is shared by a stop somebody asked for and a SIGTERM nobody
explained, so the terminal ledger row says which it was instead of calling both
`cancelled` — an unexplained SIGTERM was recorded as a cancellation, which is
the conflation this vocabulary removes.

**One sentence on the words, because `interrupted` means two things.**
`cancelled` = someone asked for this; `interrupted` = it ended without anyone
asking, whether or not the cause can be named. Read `stop_class` for which.
That is deliberately NOT the session surface's "interrupted", which is its word
for a *deliberate* stop (`interrupted`/`user-stop`); on the ledger the deliberate
case is `cancelled`/`deliberate`. The two surfaces are one `grep` apart, so read
the pair, not the single word.

| terminal `status` | `stop_class` | when |
|---|---|---|
| `cancelled` | `deliberate` | the supervisor's `stop`/`cancel` control op, or a SIGTERM that paired to a covering **deliberate** stop marker (`lop stop`'s ladder) |
| `interrupted` | `attributed-involuntary` | a SIGTERM paired to a covering marker with `deliberate: false` (a prune/install/reclaim that named itself) — someone is named, nobody asked for the stop |
| `interrupted` | `unattributed-signal` | a stop signal recorded by the receipt with **no** covering marker, or exit 130/143 with no receipt at all (`stop.via` says which — `signal` or `exit-code`) |
| `interrupted` | `unattributed-death` | no terminal row at all and the owner is gone (reconciled by `--status`): SIGKILL, crash, OOM or power loss |
| `succeeded` / `failed` | *(absent)* | a normal exit carries no stop keys |

A stop row also carries a `stop` object (`via`, the `signal` entry, any covering
`marker`) and, when the marker belonged to a `stop --all` sweep, its `sweep_id`.
On SIGTERM arrival the worker appends a status-less **arrival row**
(`{"id", "signal": {...}}`) before it does anything else, so the signal is on
record even if the worker is SIGKILLed inside its 5 s settle. Scripts that
branched on `cancelled` to mean "terminated by SIGTERM" should read `stop_class`.

Two shapes of 130/143 that the table cannot cover from the row alone: a signal
that lands before the session exists (during provider discovery) has no
conversation directory to read a stop marker from, so it is recorded
`sanction: none` and classified `unattributed-signal` even if a ladder had staged
a marker for that run — the supervisor's own `stop` op is unaffected, because it
cancels the lifetime and the row says `via: control-stop`. And where a run
received several signals, the **latest** one decides the classification (the
others stay in the receipt's `signals` list with `count` giving the total): a
later signal a covering deliberate marker explains is a stop somebody asked for.

**Artifacts** (all best-effort; a failed write never changes a stop):

* `<session dir>/runtime-signal.json` — a *runtime's* receipt of a termination
  signal, written at arrival. v1 schema: `{"v":1,"kind":"runtime","session_id",
  "pid","started_at","signals":[{"name","number","at","in_flight","action":
  "drain"|"stop"|"repeat-absorbed","sender":{"state":"unavailable",...},
  "stop_marker":null|{...},"sanction":"none"|"marker"}],"count","receiver":
  {"uid","euid","gid","ppid","pgid","argv0"},"platform"}`. `sanction` is decided
  at write time: `marker` iff a `runtime-stop.json` for this exact run, staged no
  more than `PAIR_WINDOW_S` before the signal and never after, was on disk when it
  arrived; otherwise `none` — nobody staged a stop. `receiver.ppid` is the
  receiver's parent (lineage), not the sender. Never removed by a runtime; a new
  run (different pid/`started_at`) replaces it.
* `<config>/logs/stop-sweeps.jsonl` — one `begin` and one `end` row per
  `stop_all` call that acted on at least one target (`lop stop --all`, the TUI's
  stop-all); each marker the sweep stages carries the same `sweep_id`. Rotated
  once past 256 KiB. A single `lop stop` is not a sweep and writes nothing.

**Rendered times are local, with their offset** (`2026-09-30 20:00:03 -0400`).
The durable fact is the receipt's epoch `at`; the sentence exists so a person
reading a transcript days later, from another device, is not left guessing whose
clock `20:00:03` is on.

**Honest limits.** The *sender* is unavailable on macOS (the stdlib exposes no
siginfo and asyncio discards it), which is why the receipt says "unidentified
sender" rather than naming a party; an unsanctioned wave therefore yields one
receipt per victim and **no** sweep row — never a fabricated one. SIGKILL, a
crash and power loss leave nothing target-side. The TUI and `serve`/mobile
daemons are not covered by the receipt (their in-process sessions are ended by
their own owner, not by a runtime receiving a signal), and neither is a turn
ended by a party that never recorded a signal — for those the session surface's
own vocabulary still applies.

The job ID identifies the execution receipt; the session ID identifies the
conversation. They are distinct. Receipts include the log path and, once ready,
`lop --resume SESSION_ID`. The append-only job ledger lives under the same
configuration root's `logs/exec-jobs.jsonl`; the ordinary transcript and
attachment stay in the normal session storage. Runtime discovery records are
ephemeral; completed outcomes remain in the ledger after their record is gone.

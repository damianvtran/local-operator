---
name: sessions
description: "Use the `sessions` tool to list, inspect, spawn, resume, stop or peek at other local lop sessions; `lop sessions` and `lop exec --workstream` remain the CLI fallback."
---

# The `sessions` tool — other local sessions, tool-first

The `sessions` tool manages other local `lop` sessions from inside a session:
top-level sessions and stored conversations. It is the tool-first path for
session work — listing what is running, opening a parallel session, ending one,
or reading a transcript — with the CLI (`lop sessions`, `lop exec`) as the
fallback for a human's terminal, a script, or a host where the tool is absent.

It does not cover subagents (they are never published sessions; `hub` owns that
surface), and it does not deliver messages or steer: `send` owns delivery, and
steering mid-turn is `send` with `now=True`. `guide://peer-messaging` carries
the message protocol.

## Ops

| op | what it does | approval tier |
| --- | --- | --- |
| `list` | live sessions as lean rows; `include_stored=true` adds stored conversations, and `query` searches names and recent content | read |
| `info` | one session: state, directory, transcript path, origin, opener, sidebar visibility | read |
| `spawn` | open a parallel session — a listed workstream by default | write |
| `resume` | reopen a stored or stopped session headlessly | write |
| `stop` | end a running session gracefully (no force in this version) | exec |
| `peek` | a bounded transcript read — a window, a search, or a digest | read |
| `help` | the full per-op reference — accepted inputs with types and defaults, one example and the refusals per op | read |

Address a session with exactly one of `session` (exact id), `target`
(name/cwd substring), or `pid` — the same resolver `send` uses, so an
ambiguous target comes back with the candidate rows rather than a guess.

## Remote sessions (mesh)

`list` covers local AND remote rows by default — each remote row names the
device that holds it — and `scope` (`all`/`local`/`remote`) or `peer` (one
device) narrows the view. With no mesh, or a relay that is not answering,
`list` shows the rows it could read plus ONE note saying why no remote rows
could be; a peer that did not answer is named. `info`, `peek`, `spawn`,
`resume` and `stop` take `peer` (a device name or id) and act ON that device's
runtime: `spawn` mints the session there; `resume` warms it, or with `prompt`
drives a turn there and returns the owner's reply; `stop` ends it where it
lives — this tool sends no force (the CLI's `lop network sessions --stop <id>
--force` is the escalation). `peek` reads only the newest `steps` rows where
the session LIVES, live sessions only — a stored one refuses with the warm-up
named — and `query`/`regex`/`digest` stay local. Message delivery to a session
on another device is `send` with `peer`; `guide://peer-messaging` carries it.

## Visibility: `spawn` is listed by default

`spawn` opens the run as a **workstream** by default: the operator's sidebar,
`/resume` and the phone list show it, labelled with the session that opened it.
The default is deliberate — a parallel run should not go hidden because a flag
was forgotten — and every spawn/resume result names the visibility it produced.
`visibility="ephemeral"` is the explicit opt-out, for a throwaway run the
operator did NOT ask to see; ephemeral runs are hidden everywhere and silent.

Spawn only when the USER asked for the work to run separately. An independent
slice of the current job is still a `task` subagent — `guide://agents` carries
that decision.

`resume` never re-stamps: `origin.json` is written once at creation, so a
resumed session keeps its visibility. A hidden session stays hidden; the remedy
is a new workstream, not an edit.

`stop` runs the graceful ladder and releases the session lease. Restart is two
auditable calls (`stop` + `resume`); this version has no force option.

## Resume — the canonical flow

`sessions(op='resume', session='<id>', prompt='<what to do next>')` reopens a
stored or stopped conversation headlessly. The receipt says what was opened —
session, job and pid once published — and never claims more than the ledger
knows: a job that died before the session went live returns an error naming
the worker log and the CLI fallback below, and a job still booting says
"starting" and points at `lop exec --status <job>` rather than claiming a
reopen.

```
sessions(op='resume', session='a1b2c3d4e5f6', prompt='continue the audit')
→ reopened "audit" (session a1b2c3d4e5f6, job d83d63e37b63, pid 50601) — …

sessions(op='help')     # the full per-op reference, on demand
```

Validation is per op, and a refusal names the set the op DOES take — read it
instead of retrying blind:

```
sessions(op='resume', session='a1b2c3d4e5f6', prompt='go', timeout_ms=1)
→ `timeout_ms` is not a sessions parameter. `resume` takes:
  session|target|pid, prompt, background. Call op='help' for the full per-op
  reference.
```

The always-loaded description carries each op's accepted inputs; `op='help'`
prints the fuller reference (targets, defaults, refusals) any time, and
`read tool://sessions` serves the same text where that reader exists.

## Bounded peek

`peek` reads a transcript WITHOUT pulling the whole journal into context:
operator transcripts run to hundreds of megabytes, so every read is bounded and
every op's default output is budgeted (a `list` stays near 600 tokens; a
12-step `peek` near 1,800). When a body is over budget it is spilled — the
result's `details` carries a `spill` handle you can expand.

- `steps=N` — the last N steps (default 12, max 50).
- `head=N` — the first N steps, for how the session started.
- `before_id=...` / `around_id=...` — cursor windows using the entry ids an
  earlier peek returned; ids are stable across compaction, and the reply's
  footer carries the continuation hint plus `has_older`/`has_newer`.
- `query="..."` — walks the transcript BACKWARD from its end until a step
  matches (literal text; `regex=true` for a regex), then reads the window
  around the match. It reports `scanned_bytes`, and when the search budget is
  exhausted without a hit you get the honest miss and how to widen it.
- `digest=true` — a compact fold of the newest rows: counts per kind, the
  newest user ask and assistant line, the last tool calls, and live state.

One window per call: `steps`, `head`, `before_id` and `around_id` are mutually
exclusive; `regex` needs `query`; `digest` stands alone.

## What it does NOT do

- **No messaging or steering** — delivery, wake modes and model switches are
  `send`'s; there is no steer op here.
- **No subagent lifecycle** — subagents are `task`/`hub`'s surface.
- **No scheduling** — `wake`/`monitor` are untouched; a spawned session may
  arm its own.
- **No project linkage** — use the `project` tool.
- **No interactive launch** — the interactive path (`lop`, `lop --resume`)
  stays refused for every agent shell; resume here is headless.

## Fallback (CLI)

Where the tool is absent — a human's terminal, a script, a host without it —
the CLI remains:

```bash
lop sessions                                  # the live-session table
lop sessions --all --json                     # stored sessions too, machine-readable
lop exec --workstream --name <name> "<task>"  # the CLI fallback for a spawn
lop exec --resume <session-id> --background   # the CLI fallback for a resume
lop exec --status <job-id>                    # follow a background run
```

`--workstream` is what makes a CLI-spawned run listed; without it an
agent-opened run is hidden and silent, and the stamp is written once — it
cannot be promoted afterwards.

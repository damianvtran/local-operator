# Aida — the built-in chief of staff

Aida ships with local-operator; there is no setup step. She is one long-lived
conversation you reach with `/aida` (TUI or desktop composer), she introduces
herself on a fresh install's first run, and she wakes herself once a day to
review the state of your world and report what needs you. This document is the
operator-facing summary: what she writes to disk, how to pause her, and how to
switch her off entirely.

## Using her

- `/aida` — open her conversation (creating it on first use).
- `/aida <message>` — open her and send the message as your next turn.
- `/aida pause` — stop her proactive output for now (see Pausing below).
- `/aida resume` — allow it again; the cadence re-arms at the next cadence time.
- `/aida status` — what state she is in (enabled / paused / next cadence / budget).
- `/aida rename <name>` — rename HER everywhere (see Renaming her below);
  bare `rename` reports the current name.
- `/aida =pause ...` — the `=` prefix is the escape hatch for messages that
  start with a reserved word (same grammar as `/team =chart`).

She is an ordinary session otherwise: `/resume`, the sidebar and the phone all
list her like any conversation, and every runtime feature (tools, teams,
projects, subagents, wakes) works normally.

## Renaming her

She is yours to name. `/aida rename <name>` — or renaming her conversation
(the `/title` command, the picker's rename, the desktop) — stores `aida.name`,
and every surface reads that key LIVE: the `/aida` receipts, her sidebar and
picker rows (her conversation's title is rewritten to match), the desktop
payload's `name` field, and the first-run greeting. `aida.name` defaults to
`Aida`; an invalid value (empty after trimming, more than 80 characters,
control characters) is refused with the same validator `/settings` uses. The
config key is canonical: if it and her conversation's title ever disagree, the
title is rewritten to match it.

## First-run onboarding (R20–R26)

On a **fresh install** whose setup has just completed (a provider configured,
no human conversations yet), the first conversation is hers: she greets you,
introduces herself as your chief of staff, says briefly what she can do, and
asks a few details — your name, how you want to be addressed, what you work on,
an email if you want it on file. She then records what you agree to keep,
saying so first, with one command that writes ONLY into
`<config root>/system_prompt.md` (the "About the operator" section every future
session and subagent reads):

```sh
lop aida note "Name: …" 
```

The write is guarded at the source: the file is resolved from the config root
(never a path argument), a section marker keeps her notes apart from your own
instructions, nothing outside the root is ever written, and oversized or
malformed notes are refused with a sentence rather than half-applied. Editing
the section yourself is supported — deleting it is how you say "forget".
Existing installs are never re-routed: the routing predicate is "no human
conversations besides hers AND the greeting still owed".

**With no provider configured** — she still has a view. `/aida` opens it with
the provider cue as a system block, and sends are refused with the same cue
(`/login openai to get started — no provider configured (/provider lists all)`)
instead of a session-starting promise. Connecting a provider afterwards runs
the same first-run routing.

**Integration nudges (R25).** When a check-in's cadence message carries a
nudge-window line (the engine opens one at most every
`aida.onboarding.nudge_days`, default 14), she may suggest ONE missing
integration and offer to set it up; she can add an MCP server for you, and
where a login is interactive she hands you the command rather than attempting
it. The window is recorded in `aida/onboarding.json` by the engine, so the
bound is enforced, not merely advised.

## The proactive cadence

At her configured cadence, a wake fires in her conversation: she reviews sessions, projects, scheduled wakes and
usage signals, and reports only what you must act on. If there is nothing
actionable she stays quiet in substance (her instructions say so explicitly).
Her own runtime re-arms the one-shot; the cadence works with every terminal
closed because arming also installs the wake supervisor.

**Daily default or elapsed interval.** `aida.cadence.interval_hours` is a whole
number from **0 through 168**. Zero (or unset) preserves the daily local
`aida.cadence.at` time, default `08:30`; a positive value ignores `.at` and uses
elapsed hours. Set **4** for an intrinsic check-in every four hours. Whole hours
avoid accidentally scheduling provider-backed turns every few seconds; one week
is a bounded upper limit for a useful check-in, not a change to any other user's
default.

The first interval arm anchors a grid at enable time: first due = anchor +
interval. Restarts, incoming events, greetings and escalation extras do not
reset it. If the machine misses several ticks, the existing generic catch-up
owes **one check-in**, not a burst; after acceptance and one-shot retirement,
the next due is the first grid point strictly after now. DST/timezone changes
therefore do not change the elapsed interval. Pause/reactive mode suppresses
check-ins; resume retains the anchor and chooses the next future grid point,
without replaying ticks suppressed by the hold. Daily pause/resume is unchanged.

Switching interval/daily modes (or changing a positive interval) replaces a
future cadence with a newly anchored mode. An already-owed occurrence, including
one shifted into load grace or accepted into a busy session, remains owned by
its original arm until it is accepted and retired; the new mode starts after
that occurrence. Changing `.at` during interval mode has no timing effect.

**Durability and authority.** Interval rows carry a reserved arm origin,
`aida-cadence:v1:<interval-hours>:<anchor-ms>:<due-ms>`, in `request_id`. The generic
writer treats it as opaque retry identity: due is included so different
occurrences cannot falsely settle each other's arms. Origin and due commit
atomically in the authoritative `wake_schedules` transcript snapshot. Unknown
versions/malformed origins preserve present rows. Load grace can change the
scheduler's working due, but not this original owed instant.

The Aida state file mirrors `cadence_phase` after a row commits and prepares
`cadence_pending` before an arm attempt. If an initial arm fails/crashes, its
pending due remains owed. If a crash happens after the transcript commit but
before the mirror, the existing row wins and repairs the mirror. Only an
accepted **matching cadence origin** at the Session persist boundary can advance
the grid; row/index disappearance or wall time alone cannot consume a tick.
A failed persist retains the accepted receipt for retry. State writes are skipped
when unchanged. Independent `wN`, project-trigger, greeting and extra rows keep
their existing ownership and scheduling paths.

**Notifications.** Her check-ins reach you with NO surface attached: the
cadence row and the escalation extras declare `notify=True`, so a check-in that
found something actionable raises an OS banner through the completion ladder
(the desktop app, a TUI or a watching surface takes over when one is running),
and clicking the banner opens the conversation. Whether a banner is actually
raised is decided at settle time from her ACTUAL REPLY — a quiet day does not
notify, because a daily "nothing needs your attention" banner is how a chief of
staff gets muted, taking the actionable ones with it:

- a reply of exactly `Nothing needs your attention today.` is silent (case,
  padding, quotes, backticks and a trailing period are tolerated; the legacy
  `(no action needed)` spelling is still recognised; a paraphrase is not);
- a quiet-day TIP is a silent row by decision: it lands unread in her
  conversation and shows on any surface you open, but does not interrupt. The
  one-line switch is `TIP_REPLY_NOTIFIES` in `aida/proactive.py`;
- a check-in on an install whose greeting ledger is not yet `delivered`/
  `skipped` never banners — she does not ping someone she has not met;
- at most `MAX_BANNERS_PER_DAY` (3, a constant — no config key) banners are
  raised in a rolling 24 h, counted in `aida/onboarding.json` under `banners`;
- trigger check-ins stay quiet — they ask her to ACT (message the linked
  sessions), not to report, so the sentinel rule does not describe them;
- a turn that ends with `no_reply` publishes nothing: no banner, no unread
  mark;
- `display.notifications` and `LOCAL_OPERATOR_NO_NOTIFICATIONS` apply exactly
  as for every other banner, and per-run/automation homes (a `lop exec` run,
  agent-runtime-svc's per-run `HOME`) never raise one at all.

An ERRORED check-in turn always notifies, whatever the reply reads like.

**Escalation (bounded).** During a turn she can ask for an extra proactive
check-in by writing `<config>/aida/escalate.json`:

```json
{"wakes": [{"in": "4h", "message": "re-check the failing deploy"}, "at 14:00"]}
```

Each entry is either an object (`in`/`at` plus optional `message`) or a bare
time string. The engine arms each as an `aida-extra-N` one-shot, subject to:

- `aida.cadence.max_extra_per_day` (default 2; `0` disables escalation),
- `aida.cadence.min_gap_minutes` (default 90 — minimum spacing from another
  Aida wake, measured across every request armed in the same drain),
- the generic 16-schedules-per-session cap. NOTE: the 60 s wake floor is NOT
  among them — it bounds a recurring interval (`every_ms`) and a one-shot
  `{"in": "1s"}` arms about a second out; only the cap, the budget and the
  spacing floor apply to extras (review round 1, n1).

Requests beyond a bound are dropped with a note, so the bound is observable
rather than silent. WHERE the note lands depends on the writer: the in-session
reconcile journals it to her transcript (the requesting turn can read it
there); the external drain — a runtime-less `resume`, or boot recovery for a
tray left by a process that died mid-turn — has no transcript writer and logs
it instead (`logger.warning`, module `local_operator.aida.proactive`). The
level is WARNING, not INFO, deliberately (QA round 2, Q3): every note here is a
request that did not take effect or was handed to another writer, and `lop
serve` configures its console logging at WARNING by default — info-level lines
were exactly the lines a default-daemon operator could not see.

## Trigger check-ins (project staleness)

The cadence is not her only reason to wake. A generic **wake-trigger layer**
(`local_operator/wakes/triggers/`) evaluates named sources for "something has
gone stale", records a pending check-in, and lets the wake supervisor engage
her exactly as a scheduled wake would. The first source watches PROJECT
STALENESS: a project in `planning`/`active`/`qa`/`validation` whose last
progress line is older than `projects.stale_after_hours` (default 4) earns one
check-in — she messages the linked sessions (or the project's manager) for a
status update and a `project` progress refresh, makes ONE bounded resume
attempt for a session that looks dead or stalled, and surfaces a sessionless
project to you instead of spawning work. **She never does the update work
herself.**

Bounds, all configurable in `/settings` (section "Wake triggers"):

- `wakes.triggers.enabled` (default `true`) — the master switch,
- `wakes.triggers.max_per_day` (default 6) — a per-target rolling-24 h budget;
  `0` disables,
- `wakes.triggers.min_gap_minutes` (default 60) — minimum spacing between two
  trigger wakes to the same target,
- `wakes.triggers.project_staleness.enabled` (default `true`) — the source's
  own switch.

One wake per stale EPISODE: the identity is the project's
`(id, status, progress_updated_at)` fingerprint — the same latch the
completion-time check uses — so a record that stays stale earns nothing more,
and a new progress line (or a status move) is a fresh episode. Done, paused
and archived projects are never candidates, and a paused/disabled/reactive
Aida suppresses trigger wakes exactly as she suppresses her cadence. The
supervisor evaluates on its own ~5-minute throttle; the published settings
snapshot it reads is written on every settings edit and on her boot/reconcile.

## Pausing

`/aida pause` (or the desktop's Aida control) sets `aida.cadence.paused = true`:

- her `aida-*` wake rows are cancelled through whichever writer owns them,
- `held_at` is stamped on her wake-index entry, so the wake supervisor skips
  her entirely while paused (no runtime is even started) — when the entry
  survives the cancel; with only her rows the entry is pruned and the
  supervisor skips by its absence all the same,
- an unfired one-time greeting is un-stamped by the cancel, so the resume
  arms it again instead of losing it (the paused greet receipt promises
  exactly that),
- a session opened while paused does not arm her rows at all, and a row that
  somehow comes due while paused is dropped instead of delivered.

`/aida resume` clears the flag and re-arms at the next cadence time. While
paused she does not run the internal cadence and does not send proactive
output; ordinary conversation with her works normally.

## Switching her off (R17/R18)

Two switches, either one disables her completely — this is the supported switch
for harness-only or automation installs (no TUI, no desktop):

- config: `aida.enabled = false`
- environment: `LOCAL_OPERATOR_NO_AIDA=1`

Disabled means **zero footprint**: `ensure_session` returns before any path is
joined, so no session is created, no state file is written, no wake is armed
and no supervisor is installed. Re-enabling restores everything on the next
boot or `/aida`.

**Auto-activation (R21), the default beyond the switches.** A boot only
CREATES her when a human surface is present in that process: a terminal
(the TUI, a `lop serve` you started yourself), or a daemon the desktop app
spawned (`LOCAL_OPERATOR_DESKTOP_TOKEN`). A cloud/automation install —
agent-runtime-svc pipes, no desktop plane — is not auto-activated by default;
it pays no session, no cadence and no wake cost, and still gets her the moment
something opens her explicitly (`POST /v1/desktop/aida {op:"open"}`, `/aida`,
a desktop claim). Deployers who want no trace at all set one of the switches
above.

## What she writes to disk

Everything lives under `<config>/aida/` (default `~/.local-operator/aida/`),
plus the standard session directory:

| path | what it is |
|---|---|
| `aida/state.json` | which session id is hers, when she was last paused, today's escalation budget |
| `aida/onboarding.json` | the one-time greeting ledger (`greeted_at`) and the R25 nudge-window bookkeeping (`nudge_offered_at`, `nudge_offers`) |
| `aida/escalate.json` | her escalation in-tray (written by her, consumed by the engine) |
| `aida/ensure.lock` | the cross-process lock serialising all of the above |
| `sessions/<id>/` | her conversation — a normal session directory (transcript, attachment sidecar naming the `aida` role, its current title — `aida.name`) |
| `wakes/<id>.json` | the standard wake index entry for her session |
| `system_prompt.md` | the operator's custom instructions — written by her ONLY through `lop aida note`, inside the `About the operator` section |

None of these is a config key: they are runtime-managed bookkeeping, not
settings a user authors.

## Configuration keys

| key | default | meaning |
|---|---|---|
| `aida.enabled` | `true` | master switch (read at boot and by `/aida`) |
| `aida.name` | `"Aida"` | her display name on every surface (see Renaming her) |
| `aida.cadence.at` | `"08:30"` | daily cadence time, `HH:MM` local; ignored for positive intervals |
| `aida.cadence.interval_hours` | `0` | `0` = local daily mode; `1–168` = elapsed whole hours |
| `aida.cadence.paused` | `false` | the pause flag (written by `/aida pause|resume`) |
| `aida.cadence.max_extra_per_day` | `2` | escalation budget (`0` disables escalation) |
| `aida.cadence.min_gap_minutes` | `90` | minimum spacing between Aida wakes |
| `aida.onboarding.nudge_days` | `14` | integration-nudge window length, read by the cadence engine (R25) |
| `projects.stale_after_hours` | `4` | a project's progress older than this reads stale (badge, tool rows and the trigger all resolve through the same reader) |
| `wakes.triggers.enabled` | `true` | master switch for wake triggers |
| `wakes.triggers.max_per_day` | `6` | per-target rolling 24 h trigger budget (`0` disables) |
| `wakes.triggers.min_gap_minutes` | `60` | minimum spacing between trigger wakes to one target |
| `wakes.triggers.project_staleness.enabled` | `true` | the project-staleness source's own switch |

All of these are editable from `/settings` (sections "Aida", "Projects" and
"Wake triggers") and `lop config`.
Renames made through `/aida rename` or a conversation rename also write
`aida.name`, so the two gestures and the settings page cannot drift apart.

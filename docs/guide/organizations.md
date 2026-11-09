<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Agent organizations and cross-agent communication

## Agent Organizations

Three layers: subagents are the parallel workers, roles decide what each one
is allowed to touch, and a team is a saved roster you can point at a different
product.

**Subagents.** Ask for parallel work and the agent fans it out into concurrent
background workers, then keeps working while they run. The subagent dock shows
each worker's status, spend, and progress live, and you can open any of them to
read its transcript and plan (the reader's keys and limits are in
[docs/subagent-reader.md](../../docs/subagent-reader.md)).

<p align="center">
  <img src="../../static/ui-subagents.png" alt="The desktop app's run pane mid-fan-out, 2 running and 1 queued: 'Summarise the findings' (writer, writing reports/unpaid-march.md) and an audit of the March invoices against the ledger (reviewer, checked 448 of 452 ledger rows), each with elapsed time, context use and spend, with 'Publish the summary' queued behind them, above the shared to-do list; beside them, the conversation shows the reconcile request and its Read invoices/march.csv and reconcile.py receipts" width="720">
</p>

<p align="center"><i>Two workers mid-run in the run pane — one drafting, one checking the ledger — and a third queued, each with its role, elapsed time, context use, and spend, above the to-do list they share.</i></p>

**Roles are capability boundaries.** A subagent launched as `reviewer` carries
vetted review guidance *and loses the tools to edit code*. It can read and run
tests, but it has no way to alter what it reviews. A restricted role cannot
enable new MCP tools either, and the restriction is inherited by everything
it delegates to, at any depth. Packaged starters for `reviewer`, `coder`,
`architect`, `manager`, `designer`, `scout`, `ux-reviewer`, `tui-designer`, and
`copy-reviewer` ship in the package:
`task(agent=…)` and `/team` use them even on a fresh install, and
`agent install` copies one into your registry so you can edit it.
`lop agents list` shows what's installed, so a fresh install prints "No agents
found." You can also author your own **agent profiles**: reusable roles and
named specialists with their own instruction sets, matched to tasks by
semantic routing. When a profile gives bad guidance, you fix the profile once
instead of every prompt that uses it.

**Teams.** A saved roster (a manager plus members with counts) layered with
two briefs the individual agents never hard-code: a *collaboration* brief (how
this group works together, who blocks a release) and a *project* brief (what
product this instance owns). Swap the project brief and the same roster staffs
a different product. A roster slot can name another team (`team:<name>`), so a
team becomes an org of teams; `/team chart <name>` draws it as an org chart.
Nesting is live: `task(agent="team:<name>")` starts that sub-team's manager,
with its own roster and briefs. `lop teams list` is empty until you create one.

<p align="center">
  <img src="../../static/ui-teams.png" alt="The desktop app's Agents and teams page with the release-crew team open: its name and description, manager agent (architect), members (coder ×2, reviewer ×1), collaboration instructions, and project brief; the page's own list shows the release-crew and docs-pod teams." width="720">
</p>

<p align="center"><i>A team in the desktop app: the roster — manager plus members and counts — and its two briefs, how the group works together and what it ships.</i></p>

<p align="center">
  <img src="../../static/tui-teams.png" alt="The /team picker listing a saved team" width="720">
</p>

<p align="center">
  <img src="../../static/tui-team-command.png" alt="Sending a real request to a team: /team lopdev Can you implement a mobile relay functionality in lop using tailwind, shadcn" width="720">
</p>

Sending a request to a team is one line: `/team <name> <request>` makes the
current agent that roster's manager, which breaks the work down and puts the
right roles on it. Agents and teams can also be managed from the CLI:

```bash
lop agents create "My Agent"
lop agents list
lop teams list
lop teams show lopdev
```

## Cross-Agent Communication

Two shapes of conversation, one machine, no cloud in the middle.

**Down the tree.** Every worker a session spawns is addressable through `hub`:
`peek` reads the last few steps of its transcript without spending its
attention, `ask` poses a question and waits for the answer, `send` drops a
note, `steer` changes its course, `pause` stops it while keeping it resumable,
`cancel` ends it, and `resume` relaunches a stopped, paused, or settled child
against its own transcript, including after the parent process has restarted.

**Across sessions.** Two `lop` sessions you started yourself, in different
repos, with no parent between them and no shared context, can message each
other directly. One can tell another to hold off on a deploy, hand over a
finished branch, or claim a shared resource, without you relaying it between
terminals.

<p align="center">
  <img src="../../static/tui-peer-message.png" alt="A lop session receiving an inbound peer message card from another session named 'Audit custom fields on profiles E2E' (pid 50793), which announces it is taking the user-dashboard QA and prod deploy slot for MR !1356 and asks the receiver to object now if it has an in-flight QA validation; below it the receiving session's own send tool card replies 'No objection — go ahead', followed by its wait, bash, and hub peek receipts" width="720">
</p>

<p align="center"><i>Two independent sessions negotiating a shared deploy slot. One claims it and asks for objections; the other checks its own in-flight work and clears it. No human in the loop.</i></p>

`lop sessions` is the directory of every session on the machine: state,
pid, kind, conversation, model, memory footprint, uptime, and heartbeat age
(`--json` adds `cwd` and `session_id`). From a shell you use `lop send`; from
inside a session the agent uses its own `send` tool, which lands as an
auditable card in its transcript. Both share the same three delivery modes
and differ only in the default: the CLI drops to the mailbox unless you ask
otherwise, while an agent's `send` wakes an idle peer. `--wake` starts a
turn if the target is idle; `--now` injects mid-turn to redirect a session
that is actively working. A running `bash`, `eval`, or MCP call is never cut
short, and a session parked in `wait` returns early to read the message.

```bash
lop sessions
lop send "release cutter" "gates are green, ready for review"
lop send "release cutter" --wake "the deploy finished; verify prod"
lop send "ingest refactor" --now "hold off, the schema changed"
lop send --pid 12345 "gates are green, ready for review"   # address by exact pid
git log -1 --stat | lop send "release cutter"      # body from stdin
```

Every delivery leaves a receipt on both ends: the target sees an inbound
`↔ peer message from "<conversation>" (pid N, <model>)` card, and the sender
gets back how the message landed. Targeting is a case-insensitive substring
over conversation name, session id, and cwd basename, and it refuses to guess:
an ambiguous match lists the candidates and exits non-zero rather than
delivering to the wrong session. The trust boundary is your OS account: every
session publishes a `0600` discovery record under a `0700` directory and
answers on an authenticated loopback server, so there is no remote or
cross-user path. The full targeting and refusal rules, the receipt strings,
and the limits (256 KB bodies; headless `exec` sessions may not receive) are
in the packaged [peer-messaging guide](../../local_operator/guides/peer-messaging/GUIDE.md).

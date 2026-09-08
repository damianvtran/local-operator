# Headless sessions (`lop exec`)

`lop exec --help` is the complete command reference. Exec runs an ordinary
persisted conversation without starting a terminal UI. A saved team is a real
attachment: the current session becomes its manager, with the saved roster,
manager instructions, collaboration and project briefs. Delegated work uses
that same team; the prompt is never rewritten into a simulated slash command.

```sh
lop exec 'Review the implementation' --team release --background
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
| `--profile NAME` | Attach a registered role, specialist or packaged starter, exactly like `/agent`. Unknown names fail preflight. |
| `--agent NAME`, `--agent-name NAME` | Existing legacy named-agent selection; creates a missing agent. Not the `/agent` role attachment. |
| `--agent-id ID` | Select an exact existing legacy agent; mutually exclusive with `--agent`. |
| `--goal TEXT` | Set the literal standing goal (`clear` is text, not a command). A goal alone does not start a model, and it is not sent as a message — see [Divergence from `/goal`](#divergence-from-goal). |
| `--clear-goal` | Explicitly clear a resumed standing goal; mutually exclusive with `--goal`. |
| `--loop N` | Run 1–25 continuation iterations **after** an optional initial prompt. Requires a new or resumed standing goal. |
| `--loop-goal TEXT` | Run continuation/judge iterations until achieved. No fixed iteration cap; repeated undecidable judge results fail safely. Mutually exclusive with `--loop`. |
| `--name TEXT` | Set the persisted conversation title. |
| `--effort LEVEL` | Set reasoning effort using the selected model's existing validation. Unsupported levels fail before a turn. |
| `--resume [ID]` | Reopen the same transcript; omit the ID to select the most recent session. A live headless owner is refused rather than raced; `lop --resume ID` attaches the TUI to that owner instead. |
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
startup options override only their own slots. Team and profile can coexist;
a profile does not remove the team's roster. Stored loop progress remains
visible, but restarting or resuming never automatically replays iterations.
Pass a new loop option explicitly to start another loop.

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

A foreground run exits 0 on success and nonzero on failure/cancellation. A
background launcher exits after readiness (or reports `starting` after a
bounded wait); its exit code is **not** the eventual execution result. Follow
`lop exec --status JOB_ID` for `starting`, `running`, `succeeded`, `failed`,
`cancelled` or `interrupted`, and inspect its log. Worker termination disposes
the session before writing the terminal result. Abrupt death cannot truthfully
report success: status reconciliation compares the PID's process-start identity
and marks a proven-dead owner interrupted. It never restarts a loop or cleans
up a successor's resources.

The job ID identifies the execution receipt; the session ID identifies the
conversation. They are distinct. Receipts include the log path and, once ready,
`lop --resume SESSION_ID`. The append-only job ledger lives under the same
configuration root's `logs/exec_jobs.jsonl`; the ordinary transcript and
attachment stay in the normal session storage. Runtime discovery records are
ephemeral; completed outcomes remain in the ledger after their record is gone.

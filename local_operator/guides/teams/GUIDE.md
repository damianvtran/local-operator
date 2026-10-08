---
name: teams
description: "Create, update, and run Local Operator teams: a manager plus reusable agents, with layered briefs. Covers pull/push/update, including the public hub."
---

# Teams

A team is a named roster of reusable agents under one manager, plus two instruction layers that do not belong on any one agent.

Do not treat a team as a new kind of agent. Agents stay reusable; the team is the grouping.

## The three instruction layers

A member actually sees, outermost last:

1. **Base** — the agent's own `system_prompt.md` (or a packaged role seed). Write this once. A `coder` or a "User Dashboard Agent" can sit on many teams.
2. **Collaboration** — `teams/<id>/instructions.md`. How THIS group works together: review order, who blocks a release, how the manager delegates.
3. **Project** — `teams/<id>/project.md`. The product or domain this instance of the team is responsible for. Swap this file to reuse the same roster on another product.

Never copy a team's collaboration or project brief into an agent's base instructions. That is how a reusable coder becomes "the user-dashboard coder" and cannot staff anything else.

Roster members are the same roles/specialists `/agent` exposes: authoring an agent for a team also makes it individually invokable with `/agent <name> <message>`, so its name must follow the no-spaces rule either way.

## When the user asks to create a team

Work with them. Do not invent a roster silently. Ask, using the `ask` tool when a
choice is theirs. If the call returns a receipt rather than the answers, the host
queues asks: the roster comes from the answers, which arrive as a turn, never
from the receipt:

- the team **name** (letters, digits, dot, underscore, hyphen; no spaces — it is a `/team` argument)
- the **manager** and what they are responsible for (default: install the `manager` starter)
- each **member**: a packaged role (`coder`, `reviewer`, `architect`, `designer`, `scout`, `manager`, `ux-reviewer`, `tui-designer`, `copy-reviewer`) or a specialist the user wants authored, and how many of each
- **collaboration**: how they work together
- **project**: only if this instance owns a product or domain

Then:

1. `agent` `op='install'` for each packaged role that is not yet in the registry, or `op='create'` `kind='specialist'` (or `kind='role'`) for a new profile with a real instruction set.
2. `team` `op='create'` with `manager`, `members` as `role` or `role:count`, `instructions`, and `project`.
3. Tell the user they launch it with `/team <name> <request>`.

Example: "create a Feature Release Team with a manager, coder, designer, architect, and security reviewer" → install those starters (author a `security-reviewer` role if none exists), agree collaboration ("architect designs, coder implements, reviewer and designer sign off, manager reports"), create the team, and stop. Do not start the work until they send `/team Feature-Release …`.

## Names, labels and aliases

A team has exactly one addressable key: its **name** (letters, digits, dot,
underscore, hyphen; no spaces). Everything you type resolves by key —
`/team <name> <request>`, `team show <name>`, `--team <name>`, `team:<name>` —
and no display form ever changes it.

A team can also carry a display **label** (free text, spaces allowed) and extra
**aliases** (more TUI-safe keys resolving to the same team):

- Every surface paints ONE shared display form, so the `/team` listing, the
  picker, the settings pane, the status band, the org chart, the CLI and the
  `team` tool cannot disagree about how a team reads:
  - no label → the raw name;
  - the label is only the derived Title-Case default of the name and casefolds
    to it → the raw name (`lopdev` stays `lopdev`: title case nobody chose is
    noise);
  - a derived default whose spelling really differs → the label alone
    (`data-quality` → `Data Quality`);
  - a chosen label → `label (name)`, so the key you type stays visible
    (`Platform Reliability (ops)`), unless the label casefolds to the name
    (`OPS` for `ops`).
  It is LOCAL display metadata: it never rides the hub push document and
  hub-sync merges never touch it.
- A team with no label renders a derived Title-Case default
  (`data-quality` → `Data Quality`), which is persisted on the team's next write
  and re-derived when the name changes. Six common initialisms stay upper-case
  (`qa-tester` → `QA Tester`, `pergamon-ai` → `Pergamon AI`).
- Each alias is another key for the same team (`--alias`, or `aliases=` on the
  `team` tool). Up to 8, each obeying the name rule; no alias may collide
  (case-insensitively) with any team's name or another team's alias. Aliases
  complete in the `/team` picker like a name, and `teams show` / the `team`
  tool's `show` list them.
- Set them with the `team` tool (`label=` / `aliases=`), the CLI
  (`lop teams create feature-release --label "Feature Release" --alias fr`), or
  the desktop API. On `update`, `label=""` resets to the derived default
  rather than clearing it.

## Nested teams (orgs)

A member slot can reference ANOTHER team instead of an agent, turning a flat roster into an **org** — a team of teams. Prefix the member token with `team:`:

- `team:pod` — nest the team named `pod` as a sub-org.
- `team:pod:2` — two independent copies of the `pod` sub-org.

A bare token (no prefix) stays an agent, so the existing `coder` / `reviewer:2` grammar is untouched. A nested team carries its own collaboration and project briefs; the slot only points at it by name, so the same `pod` team can be nested under two different orgs without copying its briefs into each parent. Nesting is bounded (a reference deeper than the org-depth limit, or a cycle where A nests B nests A, is truncated rather than followed).

`team show <name>` badges a nested slot `(team)` so an org is distinguishable from a flat roster.

## Running a nested team

A nested team is launched like a role: `task(agent="team:<name>")` starts that
team's **manager** as a child, briefed with the sub-team's own roster and
collaboration/project briefs. The prefix is case-insensitive. A bare name is a
team launch only when the parent's roster has a `kind: team` slot of that name
and no agent slot sharing it — if a roster declares both a member and a
sub-team called `pod`, the bare `pod` stays an agent launch and the prefix is
what starts the team.

`team:pod:2` on a **roster** means the manager may run up to two independent
`team:pod` leads; it is advisory, exactly like `coder:2`. As a **launch**
argument a count is an error: one `task` call starts one child, so launch one
`team:pod` per copy you want.

Depth counts hops below the top session: the top is 0, its `task` children are
1, their children 2. In any tree that runs a team, a launch that would sit
deeper than `subagents.max_team_depth` (default 3, clamped to 1–8) is refused
with a `depth cap:` error naming the depth and the key. A cycle — a team that
nests a team above it, including itself — is refused with a `cycle:` error
naming the chain, and an unknown name with `unknown team`. These are errors,
never a silent generic child: a launch that cannot run says so instead of
pretending to delegate.

Every `team:` launch at any depth, and every launch at depth 2 or deeper inside
a team lineage, carries one line of chain of command: who it reports to, that
it must not push, merge, deploy, release, delete data or print secrets, and
that when the work needs one of those it stops that step and escalates through
`hub` with what it would have run and why. Depth-1 members of the top session's
own team carry the team brief without that line; the base safety rules cover
them.

That line is prompt-level, not enforcement: session-wide auto-approve reaches
delegated work, so a grandchild under `--yolo` has no human gate. The escalation
wording is what stands in for one — keep it in force when you edit a team's
instructions.

A lead that holds `task` (a sub-team's manager) gets the parent-shaped `hub`
over **its own** subtree: list, peek, send, ask, steer, pause, cancel and
resume its workers, and nothing else — an id outside its descendants is refused
as `not your subagent`. `to=["parent"]` (or a bare `message` with no `op`)
reports up to the agent that delegated to it. A non-delegating child keeps the
message-only `hub`.

Resuming a stopped child relaunches it under its live parent when that parent
is still running, so the continuation lands in the ledger that owns it. When
the parent is gone — a restart, or a sidecar whose parent row did not survive —
the resume falls back to this session rather than refusing, and the child keeps
the team, lineage and depth its own record carries: a pod worker resumed after
its lead died still comes back under the pod's brief, not the root team's.

## Running a team

`/team` lists teams. `/team <name> <request>` attaches the team to this session and sends the request as a real turn: **the team's manager becomes this session's speaker.**

A team OWNS this session's agent slot, so the two identities can never run at once:

- Attaching a team claims the slot for its manager, and replaces any profile adopted earlier in the session.
- While a team is attached, `/agent` is refused (and `--profile` alongside `--team` fails) with the reason and the way out; `/agent clear` is refused for the same reason.
- `/team clear` (or `/team none`) detaches the team and frees the slot again, so `/agent` works once more.
- Clients can read the resulting identity from the session's frontend state as `effective_identity: {speaker, team, role_of_speaker}` — the team's manager, or the attached profile, or neither.

`/team chart [name]` opens a scrollable, zoomable **org chart** of a team: the manager at the top, members beneath, nested teams expanded recursively. `chart` is a reserved first-argument subcommand under `/team` (the same shape `/mcp login|logout|reauth` uses), and `clear`/`none` (the detach verb above) is the other:

- `/team chart <name>` charts that team.
- `/team chart` (bare) charts the team currently attached to this session, or explains how to name one.
- `/team chart chart` charts a team literally named `chart` (the second token is the `[name]`).
- `/team =chart <request>` TALKS to a team named `chart` — a leading `=` on the first token means "literal team name, never a subcommand" (`=` cannot appear in a real team name, so it never collides).
- `/team clear <anything>` is a mistyped attach (only the bare verb detaches) and is reported as the unknown name it is.

Inside the chart: `+`/`-` change zoom tier (outline → standard → detailed), `f` fits to the viewport width (never collapsing past where the members are visible), `e` expands/collapses the whole canvas, `?` toggles a glyph legend (◆ manager, `?` unresolved, `↩` cycle, `⋯` depth-limit, `·N` members, `×N` copies). Arrows scroll a line; `shift+←/→` page horizontally and `PageUp/PageDown` vertically; `Home`/`End` jump to the top-left / bottom-right corner; `Esc` leaves. The chart is wide, so horizontal scroll (the `↔↕` footer hint) is the primary way to reach members off the right edge.

As the manager:

- You coordinate; you do not implement.
- Delegate with `task(agent='<role>')` using the roster. Each member already carries the team's collaboration and project briefs — give them the TASK, not a restatement of the team.
- Spin up the counts the roster names; do not invent extra copies.
- Verify from a primary source before reporting done.

## Tools

Use the `team` tool:

- `list` / `show` — what exists and what it says
- `create` / `update` — author or fix a team

Permanent removal uses the separate `team_delete` tool. It is deliberately
write-tier so the user sees and approves the destructive action; never route a
delete through the read-tier authoring tool.

Use the `agent` tool to author or install the members first. A team that names a role nobody has installed still launches; `task(agent='coder')` falls back to the packaged starter.

## CLI

```bash
local-operator teams list
local-operator teams create feature-release --manager manager --member coder --member reviewer:2
# an org: a member that is itself a team
local-operator teams create eng-org --manager director --member team:feature-release --member team:platform-pod:2
local-operator teams show feature-release
local-operator teams delete --name feature-release
local-operator teams sync --name feature-release   # check the hub and merge its updates
local-operator teams link feature-release <hub-team-id> --org <tenant>   # adopt a team pulled before tracking
local-operator teams push --public feature-release   # publish to the public hub
local-operator teams push --org <tenant> --id <hub-team-id> feature-release   # republish (overwrite) a hub team
local-operator teams pull <hub-team-id>   # pull by id; a public team also pulls by name
local-operator teams search <query>   # browse the public teams
```

Do not list the team registry at every session start. Discover it when the user asks about teams or when a task would benefit from one.

## Teams on the Agent Hub

Teams live on the Agent Hub, in an organization (`teams push --org`) or in the public catalogue (`teams push --public`; `teams pull` takes a hub id or a public team's name without `--org`; `teams search` browses the public teams). A pulled team remembers its hub id and the text it pulled; `teams sync` (and the background update check) merges hub changes three-way into description, manager, roster, collaboration brief and project brief. The local **name** never changes. The roster merges per slot (added / removed / count) and a slot you removed stays removed. Roles the roster names but you lack produce a `missing-role` warning, not a failure. Teams pulled before this feature are unlinked until `teams link`. Auto-update is `hub.auto_update.teams`; `--check`, `--prefer local|remote` and `--replace --yes` work as for agents (see the agents guide, "Agent Hub: pull, push, update", for the FAQ).

Every push previews first: the hub checks the document for personal references and shows what would be replaced before anything is published, then asks you to confirm the pinned result (`--preview-only` stops after the diff and exits 2; a scripted push needs `--yes`; declining exits 3). `teams push --id <hub-team-id>` republishes (overwrites) an existing hub row: the row is read first, so a wrong organization is refused before anything is spent. A public push leaves the local-only `project` field out of the published copy and says so before sending.

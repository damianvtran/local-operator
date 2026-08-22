---
name: teams
description: "Create, update, and run Local Operator teams: a manager plus reusable agents, with layered collaboration and project briefs."
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

Work with them. Do not invent a roster silently. Ask, using the `ask` tool when a choice is theirs:

- the team **name** (letters, digits, dot, underscore, hyphen; no spaces — it is a `/team` argument)
- the **manager** and what they are responsible for (default: install the `manager` starter)
- each **member**: a packaged role (`coder`, `reviewer`, `architect`, `designer`, `scout`) or a specialist the user wants authored, and how many of each
- **collaboration**: how they work together
- **project**: only if this instance owns a product or domain

Then:

1. `agent` `op='install'` for each packaged role that is not yet in the registry, or `op='create'` `kind='specialist'` (or `kind='role'`) for a new profile with a real instruction set.
2. `team` `op='create'` with `manager`, `members` as `role` or `role:count`, `instructions`, and `project`.
3. Tell the user they launch it with `/team <name> <request>`.

Example: "create a Feature Release Team with a manager, coder, designer, architect, and security reviewer" → install those starters (author a `security-reviewer` role if none exists), agree collaboration ("architect designs, coder implements, reviewer and designer sign off, manager reports"), create the team, and stop. Do not start the work until they send `/team Feature-Release …`.

## Nested teams (orgs)

A member slot can reference ANOTHER team instead of an agent, turning a flat roster into an **org** — a team of teams. Prefix the member token with `team:`:

- `team:pod` — nest the team named `pod` as a sub-org.
- `team:pod:2` — two independent copies of the `pod` sub-org.

A bare token (no prefix) stays an agent, so the existing `coder` / `reviewer:2` grammar is untouched. A nested team carries its own collaboration and project briefs; the slot only points at it by name, so the same `pod` team can be nested under two different orgs without copying its briefs into each parent. Nesting is bounded (a reference deeper than the org-depth limit, or a cycle where A nests B nests A, is truncated rather than followed).

`team show <name>` badges a nested slot `(team)` so an org is distinguishable from a flat roster.

> The chart renders the **declared** org — the structure the roster describes. The runtime that lets a manager delegate INTO a nested team's manager is a separate capability; a team-boundary node is tagged `(declared)` so the chart never implies a wiring that is not live yet.

## Running a team

`/team` lists teams. `/team <name> <request>` attaches the team to this session (the current agent becomes the manager of that roster) and sends the request as a real turn.

`/team chart [name]` opens a scrollable, zoomable **org chart** of a team: the manager at the top, members beneath, nested teams expanded recursively. `chart` is a reserved first-argument subcommand under `/team` (the same shape `/mcp login|logout|reauth` uses):

- `/team chart <name>` charts that team.
- `/team chart` (bare) charts the team currently attached to this session, or explains how to name one.
- `/team chart chart` charts a team literally named `chart` (the second token is the `[name]`).
- `/team =chart <request>` TALKS to a team named `chart` — a leading `=` on the first token means "literal team name, never a subcommand" (`=` cannot appear in a real team name, so it never collides).

Inside the chart: `+`/`-` change zoom tier (outline → standard → detailed), `f` fits the chart to the viewport width, `e` expands/collapses the whole canvas, arrows/PageUp/PageDown/Home/End scroll, and `Esc` leaves.

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
```

Do not list the team registry at every session start. Discover it when the user asks about teams or when a task would benefit from one.

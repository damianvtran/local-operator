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

## Running a team

`/team` lists teams. `/team <name> <request>` attaches the team to this session (the current agent becomes the manager of that roster) and sends the request as a real turn.

As the manager:

- You coordinate; you do not implement.
- Delegate with `task(agent='<role>')` using the roster. Each member already carries the team's collaboration and project briefs — give them the TASK, not a restatement of the team.
- Spin up the counts the roster names; do not invent extra copies.
- Verify from a primary source before reporting done.

## Tools

Use the `team` tool:

- `list` / `show` — what exists and what it says
- `create` / `update` — author or fix a team
- `delete` — remove one

Use the `agent` tool to author or install the members first. A team that names a role nobody has installed still launches; `task(agent='coder')` falls back to the packaged starter.

## CLI

```bash
local-operator teams list
local-operator teams create feature-release --manager manager --member coder --member reviewer:2
local-operator teams show feature-release
local-operator teams delete --name feature-release
```

Do not list the team registry at every session start. Discover it when the user asks about teams or when a task would benefit from one.

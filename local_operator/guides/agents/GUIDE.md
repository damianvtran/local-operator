---
name: agents
description: "Use Local Operator agent profiles, roles, and subagents: discover, create, select, interact, delegate, and choose the correct collaboration mode."
---

# Agent profiles and subagents

Local Operator has two different mechanisms. Do not treat them as interchangeable.

## Persistent registered agents

A registered agent is a durable profile under the Local Operator configuration root. It can carry a provider/model, sampling settings, description, tags, categories, working directory, and persisted state.

List profiles only when a task may benefit from an existing specialist or the user asks about agents:

```bash
local-operator agents list
local-operator agents list --page 2 --perpage 10
```

Do not list the registry at every session start. Local Operator semantically compares the first task with descriptions, tags, and categories locally; when a profile may fit, it exposes this guide as a short hint without injecting the registry contents. Listing is the explicit action that reveals names and metadata.

Create a profile:

```bash
local-operator agents create research
```

Start an interactive session with it, or send it one headless task:

```bash
local-operator --agent research
local-operator exec "Review this evidence and return the supported findings" --agent research
```

`--agent NAME` creates a missing profile automatically, but explicit `agents create` is clearer when setup is intentional. `exec --agent-id ID` selects an exact existing profile and fails on a missing ID.

A registered profile is not automatically a live peer in the current process. Interact by starting a session or an `exec` run with that profile. Do not imply that a message was delivered to another live agent unless a tool actually reports delivery.

## Ephemeral task subagents

Use the `task` tool when the current job contains an independent, well-bounded slice that can run concurrently or needs isolated context. A task subagent:

- inherits the parent model unless overridden, working directory, approval gate, and compaction budget
- receives only the prompt passed to `task`; include every requirement and expected output
- reports through the parent session's jobs/events
- is one level deep and cannot spawn grandchildren
- is ephemeral; it does not become a registered profile or keep durable specialist state

## Roles: `task(agent=...)`

`agent` names the ROLE a subagent runs as. A role supplies standing guidance and may restrict the child's tools, so the delegating prompt carries the TASK and the role carries how that kind of work is done well.

```
task(label="review-135", prompt="Review PR #135 ...", agent="reviewer")
```

Roles resolve from the operator's registry first, then from packaged starters (`reviewer`, `coder`, `architect`, `manager`, `designer`, `scout`). An unknown name is not an error: it launches an ordinary full child.

A role's tool allowlist is a capability boundary, not advice. A `reviewer` has no `edit`/`write` — it reads and runs tests but cannot alter what it reviews, which is what stops a reviewer from silently reviewing its own patch. Roles that do not coordinate also lose `task`/`wait`/`jobs`.

The boundary is drawn around CHANGE, not around reach. An allowlisted role keeps full read-only reach: `web_search` and `web_fetch` (retrieval with no local side effect), and the MCP tools its parent had already enabled. It cannot enable NEW MCP tools — `read mcp://<server>/<tool>` renders the reason instead of a schema — so a restricted child can never widen its surface past the parent that delegated to it, while `read mcp://` discovery still answers. A `scout` is read-only in the sense that it changes nothing; it is not offline, and a research role that could not search the web would be structurally unable to do its job.

Because a role's allowlist is persisted into its registry row when it is installed, a role installed under an older release names an older tool set. The read-only network tools are re-admitted as a FLOOR at child construction (`_with_network_floor` in `harness/subagent.py`), so an old row regains them without the harness rewriting the operator's profile behind their back. Nothing else is floored: write and execution denials come only from the allowlist.

Before a subagent's terminal handoff, it must close any browser surface it owns
with `browser action=close`, following the inherited system lifecycle rule and
its narrow exceptions. Runner disposal already calls the child's `dispose` in
a `finally` path and closes a leftover surface, but that is crash/cancellation
fallback: when writing a custom task runner, put child disposal in `finally`,
and do not use teardown as routine per-turn cleanup.

Use the `agent` tool to work with roles and specialists:

- `search` — which role fits a task, by meaning. Use it when you are about to delegate something specialized and are not sure a role exists.
- `list` / `show` — what exists, and what a role actually says.
- `install` — pull a packaged starter into the registry on first need.
- `create` / `update` — author a role (`kind='role'`, the default) or a specialist (`kind='specialist'`), or FIX one whose guidance produced a bad run. Write `description` as the trigger condition ("reviewing a merge request"), not as a job title, because that text is what `search` matches. `instructions` are the reusable BASE behaviour; a team layers collaboration and project briefs on top without rewriting them.

When a user asks for a named specialist — "create a User Dashboard Agent that knows our release practices" — that is `kind='specialist'` with a real instruction set, not a one-off prompt. Put it on a team roster later rather than baking the team into the agent.

### One profile, three surfaces

A registered role or specialist is modular: the SAME profile is (a) launchable as a subagent via `task(agent=...)`, (b) directly usable in the current session via `/agent <name> <message>` (bare `/agent` lists roles and specialists; `/agent <name>` alone adopts the profile's instructions for this session), and (c) placeable on any team roster. Author once, and it must work on all three surfaces — so satisfy the strictest surface's naming rule: no spaces in the name (it becomes a slash-command argument parsed at the first whitespace, the same constraint the teams guide states for team names). Ordinary conversational agents are not exposed on any of these surfaces.

When a delegated run goes wrong in a way the role should have prevented, update the role rather than only patching this one prompt. That is the mechanism by which review guidance improves instead of being re-derived every time.

## Teams

A team is a named roster of these agents under one manager, plus collaboration and project briefs that do not belong on any one agent. Read `guide://teams` before creating or modifying a team, and before answering a question about how teams work.

## Awaiting delegated work

`wait` returns the moment a job settles, so prefer ONE generous wait over repeated short ones — a short budget does not make the result arrive sooner, it just costs a model round trip to learn the work is still running. Pass a LIST of job ids to wake on the first of several to finish, which is how to await a fan-out without polling each child in turn.

Use `jobs` to inspect background work, and `hub` to question or steer a running child rather than cancelling and relaunching it.

## Selection rule

- Existing specialist with relevant durable instructions or state: list the registry, inspect the descriptions, then run that registered agent.
- Independent slice of the current task: spawn a task subagent, with a role when one fits.
- Work that shares mutable decisions or must happen in order: keep it in the current agent rather than paying coordination overhead.
- No clear specialization or concurrency benefit: do not delegate.

Never inject or enumerate every registered agent speculatively. The registry can be large and may contain private descriptions; discover it only in response to a relevant task.

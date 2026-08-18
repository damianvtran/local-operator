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

Use the `agent` tool to work with roles:

- `search` — which role fits a task, by meaning. Use it when you are about to delegate something specialized and are not sure a role exists.
- `list` / `show` — what exists, and what a role actually says.
- `install` — pull a packaged starter into the registry on first need.
- `create` / `update` — author a role, or FIX one whose guidance produced a bad run. Write `description` as the trigger condition ("reviewing a merge request"), not as a job title, because that text is what `search` matches.

When a delegated run goes wrong in a way the role should have prevented, update the role rather than only patching this one prompt. That is the mechanism by which review guidance improves instead of being re-derived every time.

## Awaiting delegated work

`wait` returns the moment a job settles, so prefer ONE generous wait over repeated short ones — a short budget does not make the result arrive sooner, it just costs a model round trip to learn the work is still running. Pass a LIST of job ids to wake on the first of several to finish, which is how to await a fan-out without polling each child in turn.

Use `jobs` to inspect background work, and `hub` to question or steer a running child rather than cancelling and relaunching it.

## Selection rule

- Existing specialist with relevant durable instructions or state: list the registry, inspect the descriptions, then run that registered agent.
- Independent slice of the current task: spawn a task subagent, with a role when one fits.
- Work that shares mutable decisions or must happen in order: keep it in the current agent rather than paying coordination overhead.
- No clear specialization or concurrency benefit: do not delegate.

Never inject or enumerate every registered agent speculatively. The registry can be large and may contain private descriptions; discover it only in response to a relevant task.

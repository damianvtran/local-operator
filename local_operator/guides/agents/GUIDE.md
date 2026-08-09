---
name: agents
description: "Use Local Operator agent profiles and subagents: discover, create, select, interact, delegate, and choose the correct collaboration mode."
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

Use `jobs` to inspect background work and `wait` only when no other useful work remains.

## Selection rule

- Existing specialist with relevant durable instructions or state: list the registry, inspect the descriptions, then run that registered agent.
- Independent slice of the current task: spawn a task subagent.
- Work that shares mutable decisions or must happen in order: keep it in the current agent rather than paying coordination overhead.
- No clear specialization or concurrency benefit: do not delegate.

Never inject or enumerate every registered agent speculatively. The registry can be large and may contain private descriptions; discover it only in response to a relevant task.

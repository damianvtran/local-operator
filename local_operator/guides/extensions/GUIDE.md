---
name: extensions
description: Create and extend Local Operator skills, references, plugins, custom tools, and MCP-based executable extensions.
---

# Skills and extensions

Choose the smallest extension surface that fits:

- A **skill** adds instructions and reference material. It does not execute code by itself.
- An **MCP server** is the supported plugin boundary for new executable tools without changing Local Operator.
- A **built-in tool** is a source contribution to Local Operator and is appropriate only for a universal harness capability.

Local Operator does not currently load arbitrary in-process Python plugin packages. Do not invent a plugin directory or import hook: use an MCP server for executable third-party functionality.

## Create a skill

Project skill, available in that project and descendants:

```text
.local-operator/skills/my-skill/
├── SKILL.md
└── references/
    └── details.md
```

Global skill, available in every workspace:

```text
~/.local-operator/skills/my-skill/SKILL.md
```

Local Operator walks from the current directory toward the filesystem root for `.local-operator/skills`, then checks the global root. Earlier, more-local roots win name collisions.

Choose the root by who owns the practice: put it in `~/.local-operator/skills/<name>/SKILL.md` when it should follow the operator into every workspace, and in `<repo>/.local-operator/skills/<name>/SKILL.md` when it belongs to that repository and should be committed and shared with the team. The directory name is the `skill://` name unless frontmatter `name` overrides it — keep them the same and the URL is never a surprise.

Minimal `SKILL.md`:

```markdown
---
name: my-skill
description: Configure and troubleshoot Acme deployment pipelines and release jobs.
---

# Acme deployments

Read `references/details.md` before changing a release job.
```

Frontmatter fields:

- `name`: stable `skill://<name>` identifier; defaults to the directory name
- `description`: required semantic routing signal; keep it concrete and include the task vocabulary users will use
- `enabled: false`: exclude the skill entirely
- `hide: true` or `disable-model-invocation: true`: keep direct `skill://` reads available but prevent semantic prompt listing

`description` is required: a skill without a non-blank one is not loaded at all, because it is the entire routing signal. A skill that fails to load now explains itself — read its `skill://` URL and the error names the cause (no `SKILL.md`, missing `description`, malformed frontmatter, `enabled: false`, a frontmatter `name` that disagrees with the directory, or an earlier root shadowing it) and the fix.

Keep `SKILL.md` procedural and small. Put large tables, examples, and narrow workflows under `references/`; the agent reads them only when the task reaches that branch. Dotfiles and paths escaping the skill directory are intentionally unreadable.

A skill you create is readable immediately. The first `read skill://<name>` that misses re-scans the skill roots, so the new skill resolves in this session and in subagents already running — no restart. What does **not** update until the next session is *semantic selection*: the skill will not appear in the `<skills>` listing or be auto-suggested, because that listing is frozen for prompt-cache stability. Read it by name, and tell subagents its name. Editing an existing skill's **body** has always taken effect on the next read; editing its `name`, `description` or `enabled` affects selection only, and waits for the next session.

Only the skill name and description enter the system context; `SKILL.md` enters after `read skill://my-skill`, and reference files enter only after their own reads.

## Create an executable plugin with MCP

Implement a standard MCP server over stdio or HTTP/SSE, give every tool a narrow schema and useful description, then register it:

```bash
local-operator mcp add my-plugin --command my-mcp-server --arg serve
# or
local-operator mcp add my-plugin --url https://example.com/mcp
```

Use project scope only when the extension belongs to that repository. Never commit secrets in MCP `env`; load them through the server's secure configuration mechanism.

## Contribute a built-in tool

Built-in tools follow the `createIf` registry in `local_operator/tools/registry.py`: a stable name maps to a factory that returns an `AgentTool` or `None` when the session lacks the capability. Add a typed parameter model, approval tier/description, focused behavior tests, and real-path verification. Built-ins add schema tokens to every eligible session, so prefer MCP unless the capability is universal.

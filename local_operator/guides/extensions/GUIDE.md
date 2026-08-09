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

Keep `SKILL.md` procedural and small. Put large tables, examples, and narrow workflows under `references/`; the agent reads them only when the task reaches that branch. Dotfiles and paths escaping the skill directory are intentionally unreadable.

Start a new session after adding or changing a skill. Discovery and semantic vectors are built at session creation, and the first task freezes the selected short listing for prompt-cache stability. Only the skill name and description enter the system context; `SKILL.md` enters after `read skill://my-skill`, and reference files enter only after their own reads.

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

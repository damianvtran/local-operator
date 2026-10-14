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

Local Operator walks from the current directory toward the filesystem root for `.local-operator/skills`, then checks the global root, then the skill roots other agent tools use (`~/.omp/agent/skills`, `~/.claude/skills`, `~/.codex/skills`, `~/.agents/skills`), and finally its packaged builtin catalog. Earlier, more-local roots win name collisions, so a copy in any earlier root overrides a same-named builtin. Builtins ship with each release — override or silence one by copying it out (`hide: true` on the copy silences it from semantic selection; an `enabled: false` copy is ignored), never by editing the packaged files. Full precedence table and the builtin list: `guide://skills`.

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

A skill you create is readable immediately. The first `read skill://<name>` that misses re-scans the skill roots, so the new skill resolves in this session and in subagents already running — no restart. What waits for the next session is the semantic `<skills>` listing, which is frozen for prompt-cache stability. The recommendation roster refreshes sooner — a skill authored mid-session can be a candidate on the very next message. Read it by name, and tell subagents its name. Editing an existing skill's **body** has always taken effect on the next read; editing its `name`, `description` or `enabled` affects selection only, and waits for the next session.

Only the skill name and description enter the system context; `SKILL.md` enters after `read skill://my-skill`, and reference files enter only after their own reads.

## Skills on the Agent Hub

A managed skill can be linked to an Agent Hub origin the same way an agent or team is. Linked skills install under `<config_dir>/skills/<alias>` and are discovered through the same root list above — the active profile's managed root is injected ahead of the home and ecosystem roots, so an overridden config directory resolves its own skills. The manifest is strict UTF-8 text — a `SKILL.md` plus optional `references/`, `scripts/` and `assets/` — with bounded size (64 files, 200 KiB per file, 1 MiB total); a binary or oversized file is refused by name, never silently dropped. Scripts are inert: a pull never executes them and never sets an executable mode.

## Subscriptions and the policy of a link

Every link to a hub item — agent, team or skill — carries **two independent policies**, both defaulting to `subscribe: true`, `auto_push: false`:

- **subscribe** governs unattended *inbound* updates (the pull side). Off, updates still appear as available and you apply them yourself.
- **auto_push** governs unattended *outbound* publication. It is **off by default**; enabling it requires the server to positively attest its atomic revision-CAS (plus, for a public origin, a deliberate egress consent). Until a deployed server supplies that attestation the inventory reports the origin as not eligible rather than faking it.

A manual pull or check always works regardless of either policy. List every link, healthy or not, and edit one policy:

```bash
local-operator hub links                 # every linked agent, team and skill with source, scope and policy
local-operator hub links --kind skill    # filter by kind
local-operator hub link set <link-id> --subscribe false    # unattended inbound off
local-operator hub link set <link-id> --auto-push true     # outbound on (needs eligibility)
```

`hub link set` writes through the coordinated config writer under a policy-revision check (a stale revision is refused rather than clobbering a concurrent edit); omit `--expected-revision` to read the current one first, and add `--confirm-public-egress` to consent to publishing authored changes to a public origin. One local artifact has ONE writable origin: linking the same skill to a second source is a separate local alias/copy, not an in-place relink. `hub links` lists an identity-less legacy link under `unresolved` (with reason `source-unresolved`) rather than inventing a source for it.

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

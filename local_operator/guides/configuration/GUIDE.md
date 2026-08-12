---
name: configuration
description: Set Local Operator custom instructions, system prompt, standing rules and AGENTS.md-style defaults; configure providers, models, credentials, and config file locations.
---

# Local Operator configuration

## Find the effective files first

The backend configuration root is:

1. `$LOCAL_OPERATOR_CONFIG_DIR` when set; otherwise
2. `~/.local-operator`

This rule is the same for a pip install, an editable source install, and the backend started by the downloaded desktop UI. The desktop app's private Python virtual environment changes where the package is installed, not the backend configuration root. The UI also has Electron-owned window/session settings under the operating system's application-data directory; those are not `config.yml` and do not control the backend model.

Use the CLI to avoid guessing:

```bash
local-operator config create
local-operator config open
local-operator config list
```

`config create` and `config open` print the resolved backend path. Main files under that root include:

- `config.yml`: provider, model, compaction, retries, variables, TUI, and retention settings
- `system_prompt.md`: the user's custom instructions, added to every session's system prompt
- `credentials.env` and the credential database: secrets managed by credential/login commands
- `mcp.json`: user-scoped MCP servers
- `agents/<id>/agent.yml`: persistent agent profiles
- `sessions/`: ephemeral transcripts

## Custom instructions (the system prompt)

Standing user preferences — commit conventions, review gates, where projects live, how to address the user — go in one file:

```
<config root>/system_prompt.md
```

That single file is what the desktop UI's Settings → **Instructions** box edits and what `GET`/`PATCH /v1/config/system-prompt` reads and writes, so the CLI, TUI, server, and desktop app all share one definition of "custom instructions". There is no `AGENTS.md` mechanism and no `custom_instructions` key in `config.yml`; writing either does nothing.

To install or update them, write the file directly:

```bash
mkdir -p ~/.local-operator
$EDITOR ~/.local-operator/system_prompt.md
```

Copying instructions in from another agent harness is a plain file copy — the format is just markdown, most commonly a bulleted list of standing rules:

```bash
cp ~/.some-other-agent/AGENTS.md ~/.local-operator/system_prompt.md
```

How the content is used:

- It is appended to the packaged persona inside the **first** system block, wrapped in a `<user_instructions>` tag, under a "User's custom instructions" heading. Framing is "the operator's default expectations"; an explicit instruction in the live conversation still wins.
- It is read **once, at session start**, and closed over for the whole session. That block is byte-stable for prompt caching, so re-reading it per turn would invalidate the cached prefix. **An edit takes effect in the next session, not the running one** — restart to pick it up.
- Subagents inherit it, for the same reason they inherit `/goal`: a machine-wide preference must not depend on the parent remembering to restate it in a task prompt.
- An agent profile's own `agents/<id>/system_prompt.md` is **appended to** the global file, not a replacement — a profile specializes behaviour without discarding machine-wide preferences. Set it with `agent_registry.set_agent_system_prompt` or the agent system-prompt endpoint.
- An unreadable file degrades to "no custom instructions" rather than failing the session, and undecodable bytes are replaced rather than discarding the file, so a bad edit never costs a startup or the rest of your preferences.

Keep the file free of secrets and of absolute home paths (prefer `~/`) when it may be shared. For task-specific knowledge that should load only when relevant, prefer a skill (`extensions` guide) over adding bulk here: this file is in context for every single turn.

## Set the default provider and model

```bash
local-operator config edit hosting openrouter
local-operator config edit model_name openai/gpt-4.1
```

Use a provider/model pair that the selected provider actually serves. One-session overrides do not rewrite the file:

```bash
local-operator --hosting anthropic --model claude-sonnet-4
local-operator --hosting openrouter --model openai/gpt-4.1 exec "summarize this repository"
```

Resolution precedence is agent profile, then CLI flags, then `config.yml`. An agent with its own `hosting` or `model` therefore overrides both the session flags and global defaults.

Manage secrets separately; never place API keys in `config.yml`:

```bash
local-operator login openai
local-operator login anthropic
local-operator credential update OPENROUTER_API_KEY
local-operator login-status
```

## Configuration sections

`values` in `config.yml` accepts these current groups:

- `hosting`, `model_name`: global model defaults
- `auto_save_conversation`: legacy conversation persistence switch
- `compaction`: `enabled`, `strategy`, `reserve_tokens`, `keep_recent_tokens`, `threshold_percent`, `threshold_tokens`, `max_threshold_tokens`, `auto_continue`, `mid_turn_enabled`
- `retry`: `enabled`, `maxRetries`, `baseDelayMs`, `modelFallback`, and `fallbackChains` (snake_case spellings are also accepted for retry fields)
- `variables`: non-secret named values exposed through the variable tools; environment values remain lower precedence
- `tui`: terminal UI settings such as `theme`
- `session_retention_max_sessions`, `session_retention_max_bytes`, `session_retention_max_age_days`: independent ephemeral-session ceilings; `0` disables that ceiling

`conversation_length`, `detail_length`, and `max_learnings_history` remain readable for compatibility but are deprecated and do not govern the current compaction engine.

`config edit` is best for scalar top-level values. Edit nested mappings with `config open`, preserve the top-level `version`, `metadata`, and `values` structure, then start a new session for model, retry, skill-index, or MCP startup changes.

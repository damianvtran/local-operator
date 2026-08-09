---
name: configuration
description: Configure Local Operator providers, default models, credentials, runtime settings, and file locations for pip and desktop installs.
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
- `credentials.env` and the credential database: secrets managed by credential/login commands
- `mcp.json`: user-scoped MCP servers
- `agents/<id>/agent.yml`: persistent agent profiles
- `sessions/`: ephemeral transcripts

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

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

To install or update them, write the file directly, resolving the root by the rule at the top of this guide so the commands stay correct under `LOCAL_OPERATOR_CONFIG_DIR`:

```bash
CONFIG_ROOT="${LOCAL_OPERATOR_CONFIG_DIR:-$HOME/.local-operator}"
mkdir -p "$CONFIG_ROOT"
$EDITOR "$CONFIG_ROOT/system_prompt.md"
```

Copying instructions in from another agent harness is a plain file copy — the format is just markdown, most commonly a bulleted list of standing rules:

```bash
CONFIG_ROOT="${LOCAL_OPERATOR_CONFIG_DIR:-$HOME/.local-operator}"
mkdir -p "$CONFIG_ROOT"
cp ~/.some-other-agent/AGENTS.md "$CONFIG_ROOT/system_prompt.md"
```

How the content is used:

- It is appended to the packaged persona inside the **first** system block, wrapped in a `<user_instructions>` tag, under a "User's custom instructions" heading. Framing is "the operator's default expectations"; an explicit instruction in the live conversation still wins.
- It is read **once, at session start**, and closed over for the whole session. That block is byte-stable for prompt caching, so re-reading it per turn would invalidate the cached prefix. **In the CLI and TUI an edit takes effect in the next session, not the running one** — restart to pick it up. The server and desktop app build a session per turn, so there an edit lands on the following message.
- Subagents inherit it, for the same reason they inherit `/goal`: a machine-wide preference must not depend on the parent remembering to restate it in a task prompt. A child re-reads the file when it starts, so a subagent launched after an edit can see newer instructions than its parent.
- An agent profile's own `agents/<id>/system_prompt.md` is **appended to** the global file, not a replacement — a profile specializes behaviour without discarding machine-wide preferences. Set it with `agent_registry.set_agent_system_prompt` or the agent system-prompt endpoint.
- An unreadable file degrades to "no custom instructions" rather than failing the session, and undecodable bytes are replaced rather than discarding the file, so a bad edit never costs a startup or the rest of your preferences.

The content is capped at 64,000 characters, past which it is truncated with an explicit marker and a logged warning, because it is re-sent as the cached prefix of every request. The global file and an agent profile's prompt are bounded separately so neither can crowd the other out: each is guaranteed at least 16,000 characters and may spend whatever the other leaves, so a file under that share costs the other source nothing.

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

## Configure quota-aware account and model fallback

Fallback order lives under `values.retry.fallbackChains`, and the routing engine
around it — multi-account rotation, the quota reserve, cooldowns back to the
primary — is documented in one place: read `guide://failover`. Two guides
asserting cooldown semantics is how they drift apart.

In the TUI, `/failovers` prints the live cascade and which provider is serving.


## Configuration sections

`values` in `config.yml` accepts these current groups:

- `hosting`, `model_name`: global model defaults
- `auto_save_conversation`: legacy conversation persistence switch
- `compaction`: `enabled`, `strategy`, `reserve_tokens`, `keep_recent_tokens`, `threshold_percent`, `threshold_tokens`, `auto_continue`, `mid_turn_enabled`. A pass fires when the context passes `min(threshold_percent * context_window, threshold_tokens)` — the smaller of the two, defaults `0.80` (80% of the window; `80` is accepted and means the same) and `600000` tokens. On a 1M-token model that resolves to 600k; on a 200k model to 160k. Lower either knob to compact earlier; the legacy `max_threshold_tokens` key is read as `threshold_tokens` with a rename warning.
  - Compaction advisor (**BETA, off by default**): `advisor_enabled` (`false`), `advisor_every_n_turns` (`20`), `advisor_floor_tokens` (`200000`), `advisor_trigger_tokens` (`300000`), `advisor_min_confidence` (`0.6`), `advisor_timeout_s` (`30.0`), `advisor_max_calls` (`200`), `advisor_cooldown_turns` (`60`). The size trigger above cuts on recency, which can land in the middle of a task the agent is still working on. With `advisor_enabled: true` the model is asked, off the turn's critical path, whether the context is at a natural task boundary worth compacting early — advice can only make a pass fire **earlier**, never later, and never below `advisor_floor_tokens`. The pass it triggers also runs in the background and applies at the next safe boundary, so an early pass does not stall the conversation; a pass at the ordinary ceiling and a manual `/compact` stay synchronous. The advisor is not consulted below `advisor_trigger_tokens` (no problem to solve yet), is asked at most every `advisor_every_n_turns` turns, is suppressed for `advisor_cooldown_turns` after a pass it caused, and switches itself off for the session if its passes stop reclaiming meaningful headroom. Note `advisor_max_calls: 0` means **no calls at all** (the advisor is off), not "unlimited" — it is a spend ceiling, so zero fails closed. Each call re-reads the conversation, which is affordable only because it rides the provider's prompt cache; leave the default threshold at `600000` rather than lowering it to imitate this, since the point is to keep the full ceiling available while compacting earlier when there is a good place to cut. Field-level descriptions live on `CompactionSettings` in `local_operator/compaction/thresholds.py`.
- `retry`: `enabled`, `maxRetries`, `baseDelayMs`, `modelFallback`, `usageAwareFallback`, `usageReservePercent`, `usageAwareAccountPick` (default `true`: a session's first pick among same-provider accounts prefers the one with the most cached remaining quota; `false` restores the pure per-session hash spread), and `fallbackChains` (snake_case spellings are also accepted for retry fields)
- `effort`: `auto` (default false) enables the zero-token local prompt-complexity classifier; `allowMax` lets high-complexity prompts select a model's maximum effort (default stops one rung below max)
- `variables`: non-secret named values exposed through the variable tools; environment values remain lower precedence
- `tui`: terminal UI settings such as `theme`
- `session_retention_max_sessions`, `session_retention_max_bytes`, `session_retention_max_age_days`: RETIRED. Session transcripts are never deleted automatically — only an explicit user action removes a session — so these ceilings no longer do anything at any value. A config still carrying a non-zero value logs a one-line warning at startup. The only automated cleanup under `sessions/` is the reaping of directories that contain nothing (no transcript), which lose nothing by removal.

`conversation_length`, `detail_length`, and `max_learnings_history` remain readable for compatibility but are deprecated and do not govern the current compaction engine.

`config edit` is best for scalar top-level values. Edit nested mappings with `config open`, preserve the top-level `version`, `metadata`, and `values` structure, then start a new session for model, retry, skill-index, or MCP startup changes.

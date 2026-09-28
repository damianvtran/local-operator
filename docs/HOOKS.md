# Forwarded Claude Code / Codex hooks

lop can run the hooks you already configured for Claude Code and Codex. It
does not have a hook format of its own. It reads the same files, sends the same
stdin payload, and honours the same output contract, so a script written for
either tool behaves the same in a lop session.

## Turning it on

Both sources are off by default. Enable them in `/settings` → "Forwarded
hooks", or in `config.yml`:

```yaml
hooks:
  forward_claude: true   # ~/.claude/settings.json, project .claude/settings{,.local}.json, enabled plugins
  forward_codex: false   # ~/.codex/hooks.json
```

Both keys are read on every tool call, so a change applies from the next call.
`disableAllHooks: true` in a Claude settings file is honoured too.

## What is supported

- **Events:** `PostToolUse` (the tool succeeded) and `PostToolUseFailure` (the
  tool returned an error).
- **Matchers:** Claude's rules. `*`, an empty string or no matcher matches
  everything. Letters, digits, `_`, `-`, spaces, `,` and `|` form an exact list.
  Anything else is an unanchored regex.
- **Tool names:** lop's tools are reported under Claude's names: `bash`→`Bash`,
  `write`→`Write`, `edit`→`Edit`, `read`→`Read`, `grep`→`Grep`, `glob`→`Glob`,
  `web_fetch`→`WebFetch`, `web_search`→`WebSearch`, `task`→`Task`. Other tools
  keep their own names. A Codex `apply_patch` matcher matches `Write` and `Edit`.
- **Payload:** `session_id`, `transcript_path`, `cwd`, `hook_event_name`,
  `tool_name`, `tool_input`, `tool_use_id`, `duration_ms`, plus
  `tool_response` (for PostToolUse) or `error` (for PostToolUseFailure). Inside
  a subagent it also carries `agent_id` and `agent_type`, as Claude Code does,
  so a hook that skips subagents keeps skipping them.
- **Output:** `hookSpecificOutput.additionalContext`, `decision: "block"` with
  a `reason`, and exit code 2 with stderr are all appended to the tool result
  the model sees next. That is where Claude Code places them for this event.
  Each string is capped at 10,000 characters.
- **Execution:** hooks run through `/bin/sh -c` in the session's cwd, with
  `CLAUDE_PROJECT_DIR` and `CLAUDE_PLUGIN_ROOT` set and expanded. Matching hooks
  run in parallel. Each hook gets its own process group. A timed-out hook's
  group is killed; the default timeout is 600 s and a per-hook `timeout` (in
  seconds) overrides it.
- **Failures never break a turn:** an unreadable file, a crash, a timeout or
  garbage output is logged, and the tool result goes back unchanged.

## Not supported yet

`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`, `Stop`,
`SubagentStart`/`SubagentStop` and the remaining events. The same applies to
`http`, `prompt`, `agent` and `mcp_tool` hook types, `updatedToolOutput`, and
`async` hooks.

## Known caveat

Status-reporting hooks such as Orca's `~/.orca/agent-hooks/claude-hook.sh`
also match `PostToolUse`. Under lop they receive a Claude-shaped payload from a
session that is not Claude Code, which can make those tools report lop activity
as a Claude session.

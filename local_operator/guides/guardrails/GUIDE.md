---
name: guardrails
description: "Guardrail rules: block, warn or review a tool call before it runs, so the agent cannot force-push or delete what you protect, plus the built-in dangerous-delete check."
---

# Guardrails

Two layers run at ONE checkpoint — `guardrails/check.py::check_tool_call`, called
from `harness/loop.py` after the approval gate and before the tool executes. A
refusal is returned as the call's own result and always starts with
`BLOCKED by guardrail`; that marker is final, so never reword or retry a refused
call — report it.

**Order, first refusal wins:**

1. **Built-in dangerous-delete** (bash only). CRITICAL is refused with no model
   call, HIGH goes to a forked judge, LOW runs with zero model calls.
2. **Your rules** — `<config_dir>/rules/*.md` (default `~/.local-operator/rules`),
   plus `guardrails.rules_dir` when configured.

Neither layer can be loosened. `guardrails.enabled` switches user rules only; the
built-in is not switchable on purpose, because a config file the model can write
must not lift the guard protecting that same file. `guardrails.critical_override`
turns CRITICAL into HIGH — a judge decision, never a pass. Rules can only ADD
restriction: nothing in a rule file disables or loosens a built-in.

## What the built-in ranks

- **CRITICAL** — the state root, `config.yml`, `secrets/`, the install layout,
  `$HOME`, `/`, and an ancestor of those. Refused outright.
- **HIGH** — another session's directory, a direct child of `$HOME`, a repo root,
  `git clean` of untracked work, and **every target the resolver could not pin
  down** (opaque substitution, empty expansion, unknown cwd). Goes to the judge.
- **LOW** — everything else: this session's scratchpad, deep paths, `/tmp`. Runs
  unexamined.

A target derived from a secret is refused even with an approved override
(`check.py`), because consent cannot be shown for a secret you should never see.

## What matching sees and does

- Bash: the command string. Every other tool: the JSON of all its arguments —
  which is why `tool:*` works anywhere.
- `block` — nothing runs; the result is `BLOCKED by guardrail <name> (block): …`
  followed by the body.
- `review` — the same forked judge as dangerous deletes: the body is the policy
  and `question` is what it asks. A denial refuses the call; a timeout fails
  closed (the call does not run).
- `warn` — the call runs, and its result gains `Guardrail <name>: <body>`.

## Write a rule

One file per rule in `<config_dir>/rules/`. Creating the file is the whole
registration — nothing else lists them, and nothing reloads the turn:

```markdown
---
name: no-force-push-main
scope: tool:bash
condition:
  - 'git\s+push\b.*(--force|-f)\b.*\bmain\b'
action: block
---
Never force-push main; open a PR instead.
```

| field | meaning |
| --- | --- |
| `name` | the rule's name in a refusal; defaults to the filename |
| `scope` | `tool:<name>`, a glob, or `tool:*` — v1 is tool scopes only |
| `condition` | one regex or a list; ANY match fires |
| `action` | `block`, `review` or `warn` |
| `question` | asked of the judge; `review` only |
| body (after `---`) | the guidance the agent and the refusal carry |

## Limits

- 200 rule files per directory; the rest are ignored.
- A broken file — invalid frontmatter, unknown action, non-`tool:` scope, a regex
  that does not compile — is logged and that rule (or just that pattern) is
  skipped. It never breaks the turn, which also means **a rule that fails to
  parse protects nothing and says nothing**.
- Parsed rules are cached per (path, mtime, size), so an edit is re-read once.
- A rule file cannot express a path resolver; conditions are text patterns.

## Settings

`guardrails.enabled` (true), `guardrails.critical_override` (false),
`guardrails.judge_timeout_s` (20.0, clamped 1–120), `guardrails.rules_dir` ("")
— an extra folder, additive: the default folder always loads.

## The ceiling — state it, do not paper over it

The checkpoint guards **model tool calls**, and it ranks **shell text**. It is not
an OS sandbox: in-process janitors (session cleanup, disk reclaim) never pass
through it, and a delete issued through another interpreter or a form the
resolver cannot pin down is not verified — it lands at HIGH and goes to the
judge. Measured on this build: `rm -rf ~/.local-operator` → CRITICAL,
`bash -c 'rm -rf ~/.local-operator'` → CRITICAL,
`python -c "shutil.rmtree('~/.local-operator')"` → HIGH (not resolved),
`rm -rf ~/local-operator` (a repo root) → HIGH.

## When the user asks for a rule

1. **Propose it first** — the exact path, frontmatter and body, and what it will
   and will not catch — and let the user see it before it exists.
2. **Write it to the real rules directory**, never a scratch or sample copy. A
   rule outside `<config_dir>/rules/` (or the configured `rules_dir`) does
   nothing.
3. **Verify it loads and fires** — `parse_rule` on its text, or `load_rules` on
   the directory — because a file that fails to parse is skipped silently, so an
   unverified rule looks like protection that is not there. Then say what you
   observed.
4. **Say the limit in the same breath**: which tool scope it covers, what it
   matches textually, and that another interpreter or an unpinned target can get
   past it.

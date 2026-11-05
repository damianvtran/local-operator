---
name: guardrails
description: "Guardrail rules: block, warn or review a tool call before it runs, so the agent cannot force-push or delete what you protect, plus the built-in dangerous-delete check."
---

# Guardrails

Two layers run at ONE checkpoint — `guardrails/check.py::check_tool_call`, from
`harness/loop.py` after the approval gate and before the tool executes. A refusal is
the call's own result and starts with `BLOCKED by guardrail`; never reword or retry
a refused call — report it.

**Order, first refusal wins:** (1) the **built-in dangerous-delete** (bash only):
CRITICAL is refused with no model call, HIGH goes to a forked judge, LOW runs with
zero model calls; then (2) **your rules** — `<config_dir>/rules/*.md` (default
`~/.local-operator/rules`), plus `guardrails.rules_dir` when configured.

Neither layer can be loosened. `guardrails.enabled` switches user rules only; the
built-in is not switchable, because a config file the model can write must not lift
the guard protecting that same file. `guardrails.critical_override` turns CRITICAL
into HIGH — a judge decision, never a pass. Rules only ADD restriction.

## What the built-in ranks

- **CRITICAL** — the state root, `config.yml`, `secrets/`, the install layout,
  `$HOME`, `/`, an ancestor of those, and a glob reaching such a name
  (`rm -rf ~/.local-op*`). Refused outright.
- **HIGH** — another session's directory, any other path in the state root, a direct
  child of `$HOME`, a repo root, `git clean` of untracked work, and **every target
  the resolver could not pin down** (opaque substitution, empty expansion, unknown
  cwd) — unless it lands on something already critical, as an empty expansion
  collapsing to `/` does. Goes to the judge.
- **LOW** — everything else: this session's scratchpad, deep paths, `/tmp`. Runs
  unexamined. A target derived from a secret is refused even with an approved
  override: consent cannot be shown for a secret you should never see.

## What matching sees and does

Matching sees the command string for bash and the JSON of all arguments for every
other tool — which is why `tool:*` works anywhere. `block` refuses with the marker
plus the body; `review` asks the judge (a denial refuses, a timeout fails closed);
`warn` runs the call and appends `Guardrail <name>: <body>`.

## Write a rule

One file per rule in `<config_dir>/rules/`; creating it is the whole registration and
the next tool call re-reads the directory, so no reload is needed.

```markdown
---
name: no-force-push-main
scope: tool:bash
condition:
  - 'git\s+push\b.*(--force|-f)\b.*\bmain\b'
  - 'git\s+push\b.*\bmain\b.*(--force|-f)\b'
action: block
---
Never force-push main; open a PR instead.
```

Both patterns are ONE rule: `condition` is a list and ANY entry matching fires —
which is what catching both flag positions takes. Neither catches a bare
`git push --force`: which branch it force-pushes is undecidable from the text, so
state that limit when you ship the rule.

| field | meaning |
| --- | --- |
| `name` | the rule's name in a refusal; defaults to the filename |
| `scope` | `tool:<name>`, a glob, or `tool:*` — v1 is tool scopes only |
| `condition` | one regex or a list; ANY match fires |
| `action` | `block`, `review` or `warn` |
| `question` | asked of the judge; `review` only |
| body (after `---`) | the guidance the agent and the refusal carry |

## Test it before it bites

The checkpoint is pure until the judge, so a call resolves without being run. From
`eval` (the session kernel already imports `local_operator`):

```
from local_operator.guardrails.preview import preview_tool_call
print(preview_tool_call("bash", {"command": "git push --dry-run"}).render())
```

It reports, in the guard's own order: the built-in's risk, targets and branch
(`pass`, `judge`, `refused outright`); every rule that matches; and for `block`/`warn`
the exact text the agent would get, from the same builders the live checkpoint uses.
A `review` comes back as the question and policy the judge would be *asked*, never
a verdict: no judge runs. Test a candidate before it exists with
`settings=GuardSettings(rules_dirs=(Path("/tmp/candidate"),))`, and pass
`ctx=CallContext(cwd=…, env=…, scratchpad=…, session_id=…)` to rank paths as your
session would.

**What it proves:** one call against the CURRENT rule files, under the built-in
layer's real precedence (a rule that "passes" a call the built-in refuses protects
nothing). **What it does not:** that the rule catches forms you did not think of; it
never runs the call, asks the judge, reads the conversation, or models another
session's config.

## Calibrate: score it before saving

Score every new or changed rule against at least **three genuine true positives and
three false positives**: a rule that blocks the user's ordinary work is one they
will ask you to switch off, and a switched-off rule is worse than none.

```
A (broad):    git\s+push\b
B (narrowed): git\s+push\b[^\n]*\s(--force|-f)(\s|$)
```

| call | A | B |
| --- | --- | --- |
| `git push --force origin main` | blocks ✓ | blocks ✓ |
| `git push origin main --force` | blocks ✓ | blocks ✓ |
| `git push -f main` | blocks ✓ | blocks ✓ |
| `git push --dry-run` | blocks ✗ **FP** | passes |
| `git push --tags` | blocks ✗ **FP** | passes |
| `git push origin HEAD:refs/heads/feature` | blocks ✗ **FP** | passes |

Narrowing is the move: every true positive still blocks, every false positive
passes, and all six are re-run through `preview_tool_call` after each edit. Drift
shows up in three places:

- **The log** — the `guardrail.verdict` line the checkpoint writes per non-LOW
  decision at INFO (`check.py::_journal`):
  `guardrail.verdict rule=%s risk=%s verdict=%s latency_s=%.3f targets=%s`, in
  `<config_dir>/logs/runtime.log`
  (`grep guardrail.verdict ~/.local-operator/logs/runtime.log`). It carries rule, risk,
  verdict, judge latency and targets — deliberately **not the command text**, so it is
  not a record of what was typed.
- **A user report** ("why was that blocked").
- **Your own recollection** — a call you had to reword to get past a rule.

## The calibration policy

| the change | what you do |
| --- | --- |
| Only **narrows**: tighter regex, an exclusion for an obviously safe shape (`rm -rf ./build` in the workspace, `git push --dry-run`), a clearer tool scope — **and** the true positive still blocks on re-test | Calibrate autonomously; say in one line what changed, why and the evidence. |
| **Broadens** scope, downgrades `block` to `review`/`warn`, removes or disables a rule, or would let through anything in the destructive/critical class | **Ask the user first**, with the evidence and the exact proposed diff — not a vague concern. |
| Lets one specific blocked call through | Never: that is the bypass the guard exists to stop. |
| The built-in dangerous-delete check | Never touch it; it is the shipped default, not your file. |
| A single awkward event | Never calibrate from one; score three and three. |

## Leave the reason in the file

A calibration edit leaves a short dated comment in the rule body — what fired, what
evidence, what changed — so the next reader sees why the pattern is shaped that way.
The body is also the text a refusal or warning shows the agent, so keep one line:

```
Never force-push main; open a PR instead.
<!-- 2026-11-05: broad `git\s+push\b` blocked `git push --dry-run` (user report).
Narrowed to the force flags; all three forced-push forms still block. -->
```

At most **one autonomous calibration per rule per session**, and re-test after
editing — the parse cache keys on path, mtime and size, so the next call re-reads.

## Limits

- 200 rule files per directory; the rest are ignored.
- A broken file (bad frontmatter, unknown action, non-`tool:` scope, a regex that does
  not compile) is logged and skipped, per rule or per pattern. **A rule that fails
  to parse protects nothing and says nothing.**
- A rule file cannot express a path resolver: naming `resolver:` (as the shipped
  `dangerous-delete.md` does) makes `parse_rule` drop the rule ENTIRELY.

## Settings

`guardrails.enabled` (true), `guardrails.critical_override` (false),
`guardrails.judge_timeout_s` (20.0, clamped 1–120), `guardrails.rules_dir` ("") — an
extra folder, additive: the default folder always loads.

`enabled` and `critical_override` are honoured only when this machine's settings set
them in the running process (`LOOSENING_KEYS` in `check.py`): a `config.yml` edit,
by hand or by the model, reads as the default and does nothing. Never edit the file
and report an override you did not get — propose the setting to the operator.

## The ceiling — state it, do not paper over it

The checkpoint guards **model tool calls** and ranks **shell text**. It is not an OS
sandbox: in-process janitors (session cleanup, disk reclaim) never pass through it,
and a delete through another interpreter or a form the resolver cannot pin down is
not verified — it lands at HIGH and goes to the judge. Measured on this build:
`rm -rf ~/.local-operator` and
`bash -c 'rm -rf ~/.local-operator'` → CRITICAL;
`python -c "shutil.rmtree('~/.local-operator')"` and `rm -rf ~/local-operator` (a
repo root) → HIGH.

## When the user asks for a rule

1. **Propose it first** — the exact path, frontmatter and body, what it will and will
   not catch — and let the user see it before it exists.
2. **Write it to the real rules directory**, never a scratch or sample copy: a rule
   outside `<config_dir>/rules/` (or `rules_dir`) does nothing. Never edit or delete a
   rule the user did not ask about to make your own fire — a file cannot loosen
   another, so that is the only way an agent can weaken protection.
3. **Verify it loads and fires** — `parse_rule` on its text and `preview_tool_call`
   on a real call — because a file that fails to parse is skipped silently, so an
   unverified rule looks like protection that is not there. Calibrate it (above)
   before saving.
4. **Say the limit in the same breath**: which tool scope it covers, what it matches,
   and that another interpreter or an unpinned target can get past it.

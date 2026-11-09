---
name: guardrails
description: "Guardrail rules: block, warn or review a tool call before it runs, so the agent cannot force-push or delete what you protect, plus the built-in dangerous-delete check."
---

# Guardrails

Two layers run at ONE checkpoint — `guardrails/check.py::check_tool_call`, from
`harness/loop.py` after the approval gate and before the tool executes. A refusal is
the call's own result and starts with `BLOCKED by guardrail`. It is final for THAT
action: the same effect on the same targets, by any spelling, tool or path — never
reword, retry or reroute it; report it. It does not taint later work: a NEW artifact
from your own fresh work, written to your own worktree's mandated location (a repo's
`i18n/receipts/`, say), is a different action and needs no approval. Copying or
importing the blocked material there by another route is still a reroute.

**Order, first refusal wins:** (1) the **built-in dangerous-delete** (bash only):
CRITICAL is refused with no model call, HIGH goes to a forked judge, LOW runs with
zero model calls; then (2) **your rules** — `<config_dir>/rules/*.md` (default
`~/.local-operator/rules`), plus `guardrails.rules_dir` when configured.

Matching sees a bash call's command string, and the JSON of every other tool's
arguments — which is why `tool:*` reaches them all.

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
  Also LOW: a HIGH path this session (or a subagent it launched) recorded
  creating, and `git worktree remove <path>` of a clean linked worktree.

## Cleaning up what you made

Deleting your own rig directories and worktrees works without asking, through
three routes the checkpoint can verify. Everything CRITICAL is still refused.

- **Creation ledger.** After every bash call the loop records the directories it
  created: `mkdir [-p]` (the topmost new component), `mktemp -d TEMPLATE`,
  `git worktree add <path>`, `git clone <url> [<dir>]`. Rows live in
  `<config_dir>/guardrails/created.sqlite`, keyed by session id + realpath +
  inode, written under the creator and every ancestor session, so a parent may
  delete what its subagents made. Overwriting or deleting the ledger is CRITICAL.
  A later delete of exactly that path, or anything inside it, ranks LOW —
  unless another session's ledger row sits at or under the target, or the
  target is or holds a git work tree or repository this session's ledger does
  not record itself (an owned `.worktrees/` does not own a sibling's worktree;
  the session's own `git worktree add`/`git clone` leaf is recorded). A glob
  ranks LOW only when every current match is recorded. A primary checkout stays
  HIGH unless this session cloned it. `$HOME`, the state root and the protected
  roots are never recorded.
- **Clean linked worktree.** `git worktree remove <exact path>` (no `--force`)
  is LOW when the path is a registered linked worktree with no tracked or
  untracked change, no ignored entry beyond regenerable output (`node_modules`,
  `.venv`, `__pycache__`, `build`, `dist` …) and no repository nested inside
  that output, and HEAD on a branch or in some
  ref. Anything else takes the ordinary ranking. Prefer this to
  `rm -rf .worktrees/<name>`. A worktree this session created is its own:
  through the ledger it ranks LOW even dirty or with `--force`, so commit or
  push what you want to keep before removing your own tree.
- **History evidence.** A HIGH delete goes to a judge forked from the session's
  own prefix. Before it runs, the checkpoint extracts real tool calls: this
  session's from the harness's memory, its descendants' (found by `origin.json`
  parent pointers, created after it) from their transcripts. A call that is not
  read-only vouches for a directory born inside its window when its command, or
  a script the same session wrote and the call ran, names it (its basename, or
  the literal prefix that produced it, such as `qa-r5-dg-` in
  `f"qa-r5-dg-{os.getpid()}"`), or its output prints the directory's full path.
  A git repository (a `.git` directory) is vouched for only as a clone the call
  itself made: the command must resolve to `git clone` with exactly that
  destination (`C=… && git clone … $C` counts; a name in the output never
  does), and the clone must be clean — no uncommitted, untracked or
  non-regenerable ignored change, no commit that no remote-tracking ref
  contains, no stash, no nested repository, no linked worktrees, no file
  hidden from `git status` by `--assume-unchanged` or (outside a sparse
  checkout) `--skip-worktree`, and no local or `file://` remote that is gone or
  no longer a git repository ("origin gone"). Tags count
  only when the origin lacks them: against a local origin the tag must name the
  same commit there; with a network origin (never contacted) a loose tag under
  `.git/refs/tags/` is local work and a packed one came with the clone — so a
  `git pack-refs` after tagging hides a local tag (an accepted limitation). A
  `--no-checkout` clone that was never checked out has no work tree to lose.
  The judge sees that state in its facts.
  Descendants come from the harness-written lineage registry
  (`<config_dir>/guardrails/lineage.jsonl`, appended when the harness spawns a
  subagent or opens an agent session). `origin.json` parent pointers are only a
  migration fallback for child directories born before the guard went live on
  this machine (the earliest installed `lop` generation carrying it; a dev run
  falls back to the registry file's birth): a directory planted later is born
  after the guard could refuse the plant. That is a narrowing, not a seal —
  macOS `utimes` can move a birth time backwards. Symlinked session directories,
  transcripts and `origin.json` files are never read.
  Writes into `<config_dir>/sessions/` (outside every session's scratchpad) or
  `<config_dir>/guardrails/` are CRITICAL. A scratchpad holds working files,
  not evidence: deleting, overwriting or appending (`>>`) in another session's
  scratchpad from bash ranks HIGH ("another session's files"), so the judge or
  a verified ask naming the path can lift it — unless the path resolves through
  a link into that session's transcript, lineage or ask store, is a hardlinked
  file, or an unknowable part (`$(...)`) could leave the pad; those stay
  CRITICAL. A path written with `..` is judged where the kernel lands it
  (`..` resolves after following links): inside your own scratchpad it stays
  LOW, inside another session's scratchpad HIGH, anywhere else in session
  state CRITICAL. Where it cannot be pinned at check time — a `..` below a directory that
  does not exist yet, as in `mkdir -p <pad>/n && rm -f <pad>/n/../x` — it is
  deliberately CRITICAL: write `rm -f <pad>/x` instead. `cd -P` (or `cd` after
  `set -P`/`set -o physical`) is followed to the directory the kernel resolves. The
  `write`/`edit` file tools are not ranked HIGH there: they refuse only the
  evidence files, so a plain file in another session's scratchpad is not
  refused by them. CRITICAL here means every redirection (`>`, `>>`, `>|`,
  `&>`, `>&`), the destination of a known writer (`cp`, `mv`, `rsync`, `ditto`,
  `install`, `ln`, `tar -x -C`, `unzip -d`, `git clone`, `git worktree add`,
  `git log/show/diff --output`, `mkdir`, `touch`, `truncate`, `dd of=`, `tee`,
  `sed -i`, `perl -i`, `sqlite3` without `-readonly`) and any `rm`/`rmdir`/
  `unlink` or `rsync --delete` there. Reading is not writing: sources of `cp`,
  `rsync`, `zip` and `tar -c` are fine, as are `ls`, `cat`, `head`, `tail`,
  `wc`, `stat`, `du`, `grep`, `jq`, `find` (no `-delete`/`-exec`) and
  `git status/log/show/diff/rev-parse/ls-files` without `--output`, `-c`,
  `--config-env`, `--ext-diff`, `--textconv`, `--git-dir`/`--work-tree`, a
  pager flag or a `GIT_*`/`PAGER` assignment. A path there handed to any other
  program (`python3 ana.py <transcript>`, `awk`, `sort`) ranks HIGH and goes
  to the context-free judge, which can let a read through. That HIGH is a
  backstop over the writer list: any command that is not strictly read-only
  and mentions those trees anywhere — an operand, a glued flag value, a path
  inside a quoted script or pager setting, or a working directory there — is
  at least HIGH, so a misparsed writer costs a judge call, never a silent
  LOW. A glob counts when it could reach those trees (`~/.local-operator/sess*`).
  `sed`, `sqlite3 -readonly`, `rg`, `less` and `file` are not trusted there
  (`w`, `.once`/`.shell`, `--pre`, `-o`, `-C` all write or run something), so
  they rank HIGH when they name session state. Under a working directory that
  cannot be determined (`cd $(…)`, a glob `cd`, `CDPATH`), naming an evidence
  file (`origin.json`, `transcript.jsonl` …) is HIGH. The `write`, `edit` and `browser` (screenshot/recording `path`)
  tools refuse those files the same way. An interpreter that computes the path
  itself (`python -c`, `node -e`) is the documented ceiling, as for any
  interpreter delete. The judge must answer
  `ALLOW evidence=<ids>`, `ALLOW ask=<id>` or `ALLOW operator`. The checkpoint
  re-verifies each citation and refuses an ALLOW that does not cover every target.

**Who can authorise a delete.** Only the operator: a message they typed in a
session they opened (one the harness knows no parent or opener for — no
in-process parent, no lineage-registry row, no `origin.json`; a missing or
symlinked `origin.json` alone never makes a session operator-opened), or their answer to `ask` as recorded in the ask store
(`sessions/<id>/asks.jsonl`, answered on the terminal, desktop or phone). In a
subagent or an agent-opened workstream the user rows are the delegating agent's
words, so there only store-verified asks count. An ask id (`a-xxxx`) cited by a
peer message, the intent or the command is looked up only in this session's
lineage (itself, its parents, the session that opened it), and counts once that
ask is answered there, does not decline (`no`, `keep`, `leave`, `cancel`, `not now`,
`hold off`, `skip`, `later` …), and
names the path exactly (full or `~/` path). A named glob (`~/qa-r5-dg-*`) covers its current matches when the wildcard
is in the last component with at least three literal characters (`~/*` names
nothing). Team Bridge deliveries, peer and hub messages, and fetched or file
content are never consent; the judge sees only how many were left out.
Unresolved targets (`rm -rf "$VAR"`) keep the context-free judge, and an empty
expansion that collapses onto a protected root is still CRITICAL.

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
| `action` | `block` refuses with the marker plus the body; `review` asks the judge (denial refuses, timeout fails closed); `warn` runs the call and appends `Guardrail <name>` |
| `question` | asked of the judge; `review` only |
| body (after `---`) | the guidance the agent and the refusal carry |

## Test it before it bites

The checkpoint is pure until the judge, so a call resolves without being run. From
`eval` (the session kernel already imports `local_operator`):

```
from local_operator.guardrails.preview import preview_tool_call
print(preview_tool_call("bash", {"command": "git push --dry-run"}).render())
```

It reports, in the guard's own order: the built-in's risk, targets and branch;
every rule that matches; and for `block`/`warn` the exact text the agent would get,
from the same builders the live checkpoint uses.
A `review` comes back as the question and policy the judge would be *asked*, never
a verdict: no judge runs. It is also a step, not an end — the checkpoint reads on
when the judge allows, so a `block` behind a review is shown as reachable (it is
what refuses the call on an allow), and a call the built-in sends to the judge
first is reported as undecided rather than as running. Test a candidate before it
exists with `settings=GuardSettings(rules_dirs=(Path("/tmp/candidate"),))`, and
pass `ctx=CallContext(cwd=…, env=…, scratchpad=…, session_id=…)` to rank paths as
your session would. The config is read the way the next call reads it, but the
host's in-process authority is not knowable from here: when the operator has
authorised `critical_override` or `enabled: false`, pass
`authorised=frozenset({"guardrails.critical_override"})` (or `settings=`) or the
preview will be stricter than the running process.

**What it proves:** one call against the rule files the preview loaded — it names
the directories — under the built-in layer's real precedence (a rule that
"passes" a call the built-in refuses protects nothing). **What it does not:**
that the rule catches forms you did not think of; it never runs the call, asks the
judge, reads the conversation, or models another session's config.

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

- **The log**: the `guardrail.verdict` line per non-LOW decision, logged at INFO
  by `check.py::_journal`:
  `guardrail.verdict rule=%s risk=%s verdict=%s latency_s=%.3f targets=%s`.
  Which file carries it depends on the shape that made the call: a session
  runtime writes `<config-dir>/logs/runtime.log` (a process without
  `LOCAL_OPERATOR_CONFIG_DIR` writes `~/Library/Logs/local-operator/runtime.log`;
  `paths.log_dirs()` lists both), a foreground TUI writes
  `<log-dir>/local-operator.log`, and `lop exec` or a team worker writes its own
  job log, `<config-dir>/logs/exec-<stamp>-<slug>-<job>.log`, because the
  worker's stderr IS that file. The record is INFO, so a destination running at
  WARNING never writes it: `lop serve` does at the default `LOG_LEVEL` (raise it
  to `INFO` to get the line), and the headless REPL pins WARNING and keeps no log
  file at all. An empty log is therefore not evidence the checkpoint said nothing:
  say the log did not tell you, ask the user what happened, or read the log of the
  process that made the call — never conclude the guard was silent. It carries
  rule, risk, verdict, latency and targets — never the command text, so it is not
  a record of what was typed.
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
The body is also what a refusal or warning shows the agent, HTML comment included, so
keep it to one line and keep the comment short:

```
Never force-push main; open a PR instead. <!-- 2026-11-05: narrowed, was broad. -->
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

1. **Propose it first** — path, frontmatter, body, and what it will and will not
   catch — before it exists.
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

# Rewrite verification evidence

Measured on branch `feat/harness-rewrite`, 2026-08-04, macOS arm64, Python 3.13.
Model for the agentic runs: `deepseek/deepseek-v4-flash-0731` via OpenRouter
(1M context, implicit prompt caching, $0.09/M in) unless noted.

Every number below comes from a real run through `local-operator exec`, not a
unit test. Reproduce with the commands quoted in each section.

## 1. Performance contracts (docs/REWRITE.md §A)

| Contract | Target | Measured | How |
|---|---|---|---|
| Fresh conversation start context | ≤ 30,000 tokens | **1,998** (15-skill corpus) | `scripts/bench_context_budget.py --skills-dir ~/.omp/agent/skills` |
| Structural prefix stability | ≥ 90% | **93.4%** | `scripts/bench_cache_rate.py` |
| Live cache rate | ≥ 90% | **94.2%** | 30-turn agentic run, §3 below |

Static skill listing for the same corpus would cost 2,818 tokens; semantic
selection picks 2 of 15 skills for a given query at 1,072 tokens, and the tool
schema payload is 926 tokens.

## 2. Exec-mode task: CRUD app from a blank workspace

```
local-operator --hosting openrouter --model anthropic/claude-sonnet-4 \
  --run-in /tmp/lo-e2e/crud --yolo exec \
  "Create a small task-tracker CRUD app in Python: tasks.py with
   add/list/complete/delete backed by a JSON file, plus test_tasks.py with
   pytest tests covering each operation. Then run the tests and report."
```

Result: 121s wall. The agent found no `pytest` on the system interpreter,
created its own venv, installed pytest, wrote `tasks.py` (5.4KB) and
`test_tasks.py` (11.3KB), and ran the suite.

Independently verified afterwards (not trusting the transcript):

```
$ ./venv/bin/python -m pytest -q
26 passed in 0.10s
$ python3 tasks.py add "verify me" && python3 tasks.py list
Added task #3: verify me
1  ○ Pending  Buy groceries
2  ✓ Done     Walk the dog
3  ○ Pending  verify me
```

## 3. Exec-mode task: browser game + adversarial self-test (token trajectory)

```
local-operator --hosting openrouter --model deepseek/deepseek-v4-flash-0731 \
  --run-in /tmp/lo-e2e/game --yolo exec --json \
  "Create a playable browser tic-tac-toe: index.html, style.css, script.js
   with a minimax AI. Also write ai_test.js, a plain Node script that plays
   the AI against every possible opening sequence and asserts it never loses.
   Run it with node and report."
```

368s wall, 30 turns, 35 tool executions, 929 events emitted. Independently
re-run afterwards:

```
$ node ai_test.js
Draws: 183 | AI losses: 0 | AI win rate: 67.8% | Draw rate: 32.2%
SUCCESS: AI never lost a game across every possible opening sequence.
```

Token trajectory (`context_tokens` per turn, from the event stream):

| turn | prompt | of which cached | output |
|---|---|---|---|
| 1 | 2,097 | 64 | 219 |
| 2 | 2,450 | 1,792 | 202 |
| 4 | 10,112 | 9,728 | 2,004 |
| 28 | 24,311 | 23,808 | 270 |
| 30 | 25,097 | 24,576 | 458 |

Totals: 520,899 prompt tokens of which 490,560 served from cache — a **94.2%
cache rate**, 5.8% billed at full input price. A 30-turn agentic task with 35
tool calls ended at 25k context, so compaction never had to fire.

Cost: **$0.0164** with caching vs $0.0517 without — a 68% saving on the same
trajectory.

## 4. Exec-mode task: skills (semantic selection + progressive disclosure)

Workspace seeded with 5 real skills under `.local-operator/skills/`
(minerva-credentials, minerva-observability, minerva-usage-metrics,
cmux-browser, scheduled-wakeups-omp), then:

```
local-operator ... exec "I need to check production log volume for a Minerva
service in Datadog. Read the relevant skill first, then tell me exactly which
skill you read, which reference file it points at for log queries, and the
first concrete step it says to take."
```

- Semantic selection surfaced `minerva-observability` from the 5-skill corpus.
- The agent called `read` with `skill://minerva-observability` — one tool call,
  no filesystem guessing.
- Context went 2,560 → 4,240 tokens across 2 turns: the SKILL.md body was
  pulled in, the `references/` files were **not**. The agent named
  `references/datadog.md` as the pointer for log queries without reading it,
  which is exactly the progressive-disclosure contract (§C).

## 5. Provider metadata (regression this run caught)

The listing mapper only carried price, so every OpenRouter/Radient model
resolved with `context_window=-1` and `supports_prompt_cache=False`. That
silently disabled compaction (thresholds derive from the window) and suppressed
cache_control emission. Post-fix:

| model | window | max out | cache | images |
|---|---|---|---|---|
| deepseek/deepseek-v4-flash-0731 | 1,048,576 | 65,536 | yes | no |
| anthropic/claude-sonnet-4 | 200,000 | 64,000 | yes | yes |
| google/gemini-2.5-pro-preview | 1,048,576 | 65,536 | yes | yes |

`claude-sonnet-4` advertises a 1M headline window but the routed provider
serves 200k; the mapper takes the smaller so a prompt sized to the window
cannot 400 on the provider that actually serves it.

`usage.context_tokens` was likewise declared but never populated — the
compaction trigger fell back to its local estimate and the TUI status line had
nothing to report. Provider semantics differ (OpenAI/Google include cached
blocks in the prompt count, Anthropic excludes them) and are normalized in the
wire clients.

## 6. Auto-compaction (live, threshold forced to 3,000)

`~/.local-operator/config.yml` with `compaction.threshold_tokens: 3000`,
`keep_recent_tokens: 1200`, then the five-file writing task:

```
ctx per turn: 2078 2268 2452 2638 2826 3026 3226 3421 3617 3855 4101 4301
              -> compaction ->  2904 3301 3904
```

Compaction fired at 4,301 tokens, dropped the context to 2,904, and the task
still finished (all five `utils*.py` plus `summary.md` written and verified on
disk). Event stream:

```
boundary order: agent_start, compaction_start, compaction_end, agent_end
agent_start: 1   agent_end: 1   compaction pairs: 1/1
```

That single-boundary shape is a **binding streaming contract**: this run
originally emitted `agent_end -> compaction -> agent_start -> ... -> agent_end`,
which tells any UI keying off `agent_end` that the task finished and then
restarted. The session now holds the loop's `agent_end` until compaction has
decided whether the run continues, and stamps the emitted end with the
generation of the `agent_start` that opened the run. Aborted or errored runs are
never held.

## 7. Post-remediation re-verification (2026-08-05)

After the engine review remediation (compaction blockers, cache layout,
abort/continuation contract, message breakpoints):

- Full unit suite: **1391 passed, 3 skipped, 0 failed** (encompasses the TUI,
  server, providers, skills, compaction, harness, scheduler).
- `scripts/bench_cache_rate.py --turns 4`: **93.5%** structural stability
  (contract ≥ 90%) — the frozen-per-session skills block moved to the volatile
  tail; per-turn churn measured 40% and was fixed by matching omp's
  session-start selection.
- `scripts/check_streaming_contract.py` PASSES on a fresh live `exec --json`
  run: one agent_start/agent_end pair per prompt with matching generation, tool
  pairing legality, delta-only message updates, compaction inside the boundary,
  usage carrying context_tokens.
- Live E2E re-run against OpenRouter (deepseek-v4-flash): task completed,
  artifacts on disk, context ≈2–4k tokens.

Remaining gate note: `pyright` strict typing reports ~99 source-level
diagnostics (parameterized generics, optional-narrowing across the new
harness/providers/compaction modules). These are typing-precision refinements
with no runtime/behavioural impact; black and flake8 are clean. A dedicated
typing pass is tracked as follow-up, not merged into this change.

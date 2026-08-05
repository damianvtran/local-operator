# Rewrite verification evidence

Measured on branch `feat/harness-rewrite`, 2026-08-04, macOS arm64, Python 3.13.
Model for the agentic runs: `deepseek/deepseek-v4-flash-0731` via OpenRouter
(1M context, implicit prompt caching, $0.09/M in) unless noted.

Every number below comes from a real run through `local-operator exec`, not a
unit test. Reproduce with the commands quoted in each section.

## 1. Performance contracts (docs/REWRITE.md §A)

| Contract | Target | Measured | How |
|---|---|---|---|
| Fresh conversation start context | ≤ 30,000 tokens | **1,998** (15-skill corpus) | `scripts/bench_context_budget.py` (15-skill corpus) |
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
cmux-browser, scheduled-wakeups), then:

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
  tail; per-turn churn measured 40% and was fixed by selecting skills at
  session start (matching the established behavior).
- `scripts/check_streaming_contract.py` PASSES on a fresh live `exec --json`
  run: one agent_start/agent_end pair per prompt with matching generation, tool
  pairing legality, delta-only message updates, compaction inside the boundary,
  usage carrying context_tokens.
- Live E2E re-run against OpenRouter (deepseek-v4-flash): task completed,
  artifacts on disk, context ~2-4k tokens.

The pyright backlog noted in the original version of this section has since
been cleared; see §8.

## 8. Second pass (2026-08-05) — deliveries re-verified independently

A verification agent re-checked the previous round's claims against the real
binaries rather than the unit mocks, and found three shipped features that did
not work. All are fixed and re-verified here.

### What was actually broken

- **The browser tool never worked.** Real `cmux --json new-surface` returns
  `surface_ref`; the parser only looked for `surface`/`surface_id`/`id`, so no
  handle was recorded and every `goto`/`screenshot` failed. `screenshot` also
  needs `--out <path>` — passed positionally, cmux ignores it, exits 0 and
  writes elsewhere, so we reported a file that did not exist. The unit tests
  passed because they mocked a payload cmux never emits.
- **`exec --json` stdout was not machine-readable.** `logging.basicConfig`
  wrote INFO records to stdout, the same channel as the event stream, and the
  `--run-in` banner printed there too. The contract checker skipped non-JSON
  lines, which is how it passed a corrupted stream.
- **The provider controller reached the TUI positionally**, binding into the
  wrong parameter, so the whole provider slash-command surface was inert while
  the app still started cleanly.

### Evidence after the fixes

| check | result |
|---|---|
| unit suite | **1515 passed, 3 skipped, 3 snapshots** |
| `pyright local_operator/` | **0 errors** (was 87 in-scope + 18 in the TUI) |
| flake8 / black | clean |
| `exec --json` stdout | 42/42 lines parse as JSON; operator notices on stderr |
| streaming contract | PASS, and confirmed to FAIL on a deliberately polluted capture |
| structural cache stability | **94.1%** (contract ≥ 90%) |
| start context (15-skill corpus) | **2 567 tokens** vs 30 000 budget |
| default install | **25 packages / 23 MB** (from 63 / 112 MB) |

Real-binary and real-provider runs:

- **Browser, through the agent.** The model opened `https://example.com`,
  captured a 1600x1200 / 67 821-byte PNG, and verified the file itself with
  `ls`. Surfaces were closed afterwards, leaving the operator's layout intact.
- **Full task E2E** (`deepseek-v4-flash` via OpenRouter): the agent wrote a
  memoized `fib.py`, wrote `test_fib.py`, ran it (3 tests OK), and its two
  `write` calls reported `added: 10` and `added: 23` — the counters the TUI
  renders as +N/-N.
- **`/goal` and `/loop`** driven in a live TUI: set, show and clear all work;
  `/loop 2` submitted exactly two turns and stopped with a "loop finished"
  notice. The goal reaches the system prompt's LAST block, block arity stays
  at 4, and the first three blocks stay byte-identical — so the objective is
  visible to the model without invalidating the cached prefix.

### Reimplementations checked against references

The dependency trim replaced three compiled libraries. Each was verified
independently of the agent that wrote it:

- `skills/vectors.py` (replacing faiss+numpy) reproduces a reference inner
  product to **8e-7** with **identical ranking**, round-trips its persisted
  matrix exactly, and rejects both truncated and garbage blobs.
- `compaction/png.py` (replacing Pillow) emits `IHDR`/`IDAT`/`IEND` with every
  CRC independently verified by `zlib.crc32`, correct IHDR fields, filter byte
  0 on every scanline, and decodes **pixel-identical** under both a
  hand-rolled decoder and Pillow.

### Skill selection was silently dropping matches

Verifying the faiss removal surfaced a separate defect: the correct skill
ranked FIRST but scored 0.21–0.29 against a 0.27 threshold, so short realistic
queries selected **nothing** and the agent proceeded without its playbook. The
threshold is now the max-margin midpoint 0.18:

| threshold | recall | false positives |
|---|---|---|
| 0.27 (was) | 56% | 0% |
| **0.18 (now)** | **100%** | **0%** |

Cost: +279 tokens of start context. The old value came from a calibration at
dim 512, where the noise floor was 0.31; at dim 4096 it is 0.07–0.08. The
tests now pin the margin rather than the constant and were confirmed to fail
on 0.27.

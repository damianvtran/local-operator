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
threshold is now 0.19, the midpoint of the score-gap INTERSECTION of two
independent corpora (so it is not tuned to either one):

| threshold | recall | false positives |
|---|---|---|
| 0.27 (was) | 56% | 0% |
| **0.19 (now)** | **100%** | **0%** |

Cost: +279 tokens of start context. The old value came from a calibration at
dim 512, where the noise floor was 0.31; at dim 4096 it is 0.07–0.08. The
tests now pin the margin rather than the constant and were confirmed to fail
on 0.27.


## 9. Third pass (2026-08-05) — round-2 review, and four false claims

A second independent reviewer checked the round-1 remediation against clean
builds and the real binaries rather than the working venv. The three round-1
blockers were confirmed genuinely fixed. Twenty-four further findings followed,
and the four worth recording here are the ones where a previous pass of this
document, or a test it cited, asserted something untrue.

### Claims that were false

| Claim | Reality |
|---|---|
| "threshold pinned with a MINIMUM margin" | Both assertions were zero-margin strict inequalities. `MIN_MARGIN` had exactly one AST reference — its own assignment — because an earlier rewrite of the test silently did not apply. A threshold 2.7e-5 above the best false match passed. |
| This file said the threshold is 0.18 | The same commit shipped 0.19. Three module docstrings also said 0.18. |
| "unknown-extra guard survives optimization" | The test inspected `LOAD_GLOBAL` argvals for `"AssertionError"`. An `assert` compiles to `LOAD_ASSERTION_ERROR` and never to that, so the test passed on precisely the implementation it claimed to reject. |
| "100% recall, 0% false positives" | True for two corpora and two query sets, and stated without that qualification. Against the same 15-skill corpus with an independently chosen query set, recall is ~71%. |

All four are corrected in place. The recall figure now carries its
corpus/query-set qualification and the reason: a hashed term-frequency embedder
matches vocabulary, not meaning, so `"where do i put my api keys"` against a
description reading `"credential loading and storage"` scores near zero however
relevant it is. `ApiEmbedder` is the accurate path; the offline class's ceiling
is lexical overlap. The labelled query set now ships as
`scripts/calibration_queries.json`, so the figure is reproducible by the exact
command the docstring gives:

```
$ .venv/bin/python scripts/calibrate_skill_threshold.py \
    --skills-dir ~/.omp/agent/skills --labelled scripts/calibration_queries.json
relevant  : min 0.2134  median 0.2813  max 0.5208  (n=18)
unrelated : max 0.1499  median 0.0802  (n=8)
gap       : (0.1499, 0.2134)  midpoint 0.1817
0.19 100% 0%  <- shipped
```

### A fix that measured well and was wrong

Recorded because the next person will have the same idea. Capping the embedded
description at 300 characters to reduce the length dilution L2-normalised TF
suffers looked like a clear win on a query set chosen while looking at the
failures — real-corpus recall 57% -> 71%, ranking 4/7 -> 6/7, lower noise
ceiling. On the **shipped labelled set** it was worse: recall 100% -> 83% and
the score gap vanished entirely. Reverted, and the rejected idea is recorded in
the module docstring with both measurements. Trimming the routing text removes
matching vocabulary as often as it removes noise; measure any such change
against the committed query file, not a fresh set chosen post hoc.

### Verification gaps in the CI added to close R10

- The stdout-purity gate had a **proven false negative**. `while read -r line`
  returns non-zero at EOF on an unterminated final line, so pollution with no
  trailing newline never entered the loop body and the gate reported clean —
  the exact shape of an unflushed error path, which is what it exists to catch.
- Widening `test` to a 3.12/3.13 matrix left `upload-artifact` on a fixed name;
  v4 rejects a duplicate name per run with 409. The matrix would have failed
  the job it was added to.
- A bare `pull_request:` trigger runs the live-LLM jobs on fork PRs where
  secrets are absent, giving external contributors a permanently red check.
  Both live jobs are now gated on the PR originating in this repository.

### Shipped-behaviour defects

| ID | Defect |
|---|---|
| S1 | `tool_name` is model-controlled and was rendered raw by BOTH renderers, so a name carrying an erase-display escape clears the operator's terminal from inside our frame without the tool running. Stripping moved to a stdlib-only `local_operator/ansi.py` so both share one implementation. |
| S12 | String-control payloads leaked as visible text when they contained an ESC that is not part of ST (a negated class cannot cross one), and charset designators like `ESC ( B` lost only the ESC. No live control byte escaped, but attacker-chosen text in the card's styling can forge status lines. |
| S2 | The `ModuleNotFoundError` branch reported every internal import failure in a six-import block as a missing `server` extra — strictly less diagnostic than the message it replaced. |
| S9 | The R3 fix silently stopped token stats honouring `request.model`, over-reporting CJK by 23% on every o200k model. |
| S23 | The 64 KB scan window dropped a legitimate oversized payload entirely at a sharp cliff. A quoted-key fallback recovers it; a 5 MB response parses in under a millisecond. |
| S24 | `goto` puts its URL in a positional argv slot, so `goto --help` exited 0 and printed help, which was reported as successful navigation. |

### Test-quality items

`tests/conftest.py` was **entirely commented out** — zero effective lines — so
nothing isolated the environment, which is why two environment-dependent tests
shipped in one week. It now points `HOME` at a scratch directory and clears
twenty provider/config variables. (Environment only: a first version that also
patched `Path.home` and `os.path.expanduser` broke 38 tests that set `HOME`
themselves.) Three calibration tests used `assert not accumulator`, which
passes when the loop never runs. `jsonl.py` calls its on-disk encoding a
migration-sensitive contract and had zero tests on a line this branch had just
changed; eleven now cover the LF terminator, UTF-8 without escaping, embedded
newlines, CRLF and missing-trailing-newline reads, and every malformed shape.

### End-to-end re-run on the remediated head

`exec --json --yolo`, fizzbuzz + pytest self-test, `deepseek-v4-flash-0731`:

| Measure | Result |
|---|---|
| exit code | 0 |
| stdout | pure JSON — every line parsed; 5 turns, 5 tool calls |
| event pairing | 1 `agent_start` / 1 `agent_end`; 5 balanced `turn_*`, `message_*`, `tool_execution_*` |
| stderr | 580 bytes, all HTTP request logs — no diagnostics on stdout |
| start context | **2,438** tokens (budget 30,000) |
| peak context | 3,280 tokens |
| cache rate | 72.4% overall; **86.4%** excluding the unavoidably cold first call |
| correctness | the produced `fizzbuzz(15)` verified against an independent expected list — exact match |

Usage is carried on `AgentMessage.usage` (nested under the event's `message`),
not on the envelope of `turn_end`/`agent_end`. Worth stating because a consumer
reading `event["usage"]` gets `None` and would reasonably conclude usage is
never reported.

## 10. TUI product feedback (2026-08-05) — and what building it exposed

Four items came from using the TUI: no padding above the first transcript line,
no command picker on `/`, no new-session view, and a status band carrying only
the model. Built as three concurrent slices (picker, splash, band) against a
fixed contract, then verified by hand.

### Delivered

| Item | Evidence |
|---|---|
| Top padding above the first line | `TranscriptView` is `padding: 1 1 0 1` — one row at the top only, so the first line of a turn is not flush against the terminal edge |
| Slash picker with soft match | `/` lists all 15 commands; `/cmpct` → `compact`, `/qit` → `quit`, `/lg` → `login` + `logout`. One row per suggestion at 20/40/80/200 cells, `… N more` overflow row, Up/Down wrap, Tab completes without submitting, Enter completes then submits, Esc dismisses, mouse click and hover both select |
| Centered new-session splash | Block-glyph lockup + letterspaced wordmark, version, model, cwd, credential warning, hint rows. Rendered by hand at 80x22, 40x22, 26x22, 13x22, 80x9, 80x5 and 80x2: no row ever exceeds the width, and the shed order is decoration → teaching → information with the credential warning surviving to the last row |
| Rich status band | `π openrouter/deepseek-v4-flash-0731 · /private/tmp    0.2%/1.0M · $0.0005 · 2s` — provider/model, effort, cwd, subagent count, context usage as percent-of-window, cost, active duration, conversation name. Ten-step drop ladder pinned as an ORDER rather than per-width thresholds |

Auto-naming already existed (`session/naming.py`, user-set titles never
displaced by a generated one); the band consumes it.

### A defect the live run exposed: aggregator model metadata was never resolved

Driving the band against a real provider showed `128k` for a model with a 1M
window, and `$—` where a cost belonged. Root cause: `model/registry.py` carries
a PLACEHOLDER entry for aggregators (`context_window = -1`, zero prices)
because OpenRouter alone routes hundreds of models. `configure_model` could
read the real numbers from `list_models()`, but only when handed a
`model_info_client` — and `session_factory` never handed it one. So every
OpenRouter and Radient session ran on the fallbacks.

Not cosmetic. The window feeds the compaction threshold:

| | before | after |
|---|---|---|
| resolved window | 128,000 (fallback) | **1,048,576** (real) |
| compaction fires at | ~102k | ~600k (the `min(0.8·window, 600k)` cap) |
| input / output price | 0.0 / 0.0 → `$—` | $0.09/M, $0.18/M → `$0.0005` |
| `supports_prompt_cache` | False (no `cache_control` emitted) | True |

A 1M-context session was summarising history at a tenth of the window it could
actually hold, on every long run, and models reached *through* OpenRouter that
need explicit cache breakpoints silently never got them.

Fixed with `model/catalogue.py`: a disk-cached catalogue (24h TTL, atomic
write, `~/.local-operator/cache`) plus one shared `resolve_model_info` so the
numbers a session runs on and the numbers a UI prices with cannot disagree —
`_cost_for` in the TUI had been calling the static registry directly, which is
why the band said "pricing unknown" for numbers the session had already
resolved.

Cost of the fix, measured: **418 ms** cold (one HTTP call), **26 ms** warm.
Every failure mode degrades instead of raising, because none of them is a
reason to refuse to start a session:

| Situation | Result |
|---|---|
| fresh cache | used, no network |
| stale cache + fetch fails | stale copy used — days-old numbers beat being wrong by 8x |
| no cache + fetch fails | static fallback, no raise |
| corrupt / half-written cache | treated as absent, refetched |
| model absent from the catalogue | static fallback |
| provider schema drift | static fallback |
| **keyless install** | static fallback |
| provider with a real registry entry (Anthropic) | catalogue never consulted — 0.0 ms, asserted |

The keyless case is worth calling out: `OpenRouterClient` raises
`RuntimeError` on an empty key *in its constructor*, and `RadientClient`
requires a positional `base_url`. The first version of this got both wrong and
turned a metadata optimisation into "the CLI will not start without a key". The
pre-existing default-names tests caught it; two tests now pin it directly.

### Verification

- 1,737 unit tests pass, 3 skipped. `flake8`, `black --check` and `pyright`
  clean across `local_operator/`.
- Live TUI against OpenRouter: splash on boot, picker on a live session, a real
  turn that called the `write` tool, `hello.txt` written with exactly
  `live-tui-ok`, and the band reporting real usage, real cost and real duration.
- Graceful failure observed for free when a bad session factory was passed:
  `✗ session failed to start: …` in the transcript, `π session error` in the
  band, and the picker still functional.

### One reported non-issue, recorded so it is not "fixed" later

A lone `▁` appears at the top right of a narrow frame. It is the vertical
scrollbar's sub-cell end cap: at 60x20 the transcript holds 16 rows in a
15-row viewport (the new top padding is real space), so a scrollbar is
correct. Its full cells are painted as coloured spaces, which a plain-text
dump cannot show — only the fractional cap has a glyph.

## 11. Review round on the TUI work (2026-08-05)

Two independent reviewers on the TUI commit: a design/UX round on the rendered
frames and a code round on the catalogue. 30 findings, no blockers, every one
verified by the reviewer rather than inferred. All 30 are fixed.

### The two that mattered most

**The catalogue fix never engaged for most users.** `_catalogue_source` read
the key from `os.environ` only, but both sanctioned credential flows bypass the
environment — `local-operator credential update` writes the `CredentialManager`
file and the TUI's `/login` writes the `AuthStore`. So the users who configured
credentials the app's own way got a client that refused to construct on an empty
key, silently kept the 128k fallback window and `$—` forever, and saw only a
`logger.debug` line — while the welcome view simultaneously reported them logged
in, because that check reads a different store. Every other key reader in the
repo goes through `CredentialManager`; this one was the outlier.

Fixed by resolving env → `CredentialManager`, and by sending a placeholder when
neither has a key: `GET https://openrouter.ai/api/v1/models` returns 200 and all
340 models with **no Authorization header at all** (verified), so a keyless or
OAuth-only install can still learn its real context window. The `AuthStore`
cascade is deliberately not consulted — its accessor is async and this runs in a
synchronous render path.

**A drift shape escaped the "never raises" contract.** The listing schemas set
`extra="allow"`, so a payload with non-scalar extras (`"context_length":
{"max": 1000}`) validates cleanly and then raises `TypeError` inside
`float()`/`int()` — and only `ValueError` was caught, so session start failed.
Worse, `cached_listing` writes before anything interprets the payload, so the
poisoned document was served as a *fresh* cache hit on every subsequent start:
the failure would repeat for a full day with no refetch. Now caught, and the
entry is purged so the next start recovers on its own.

### Four issues found by auditing my own code before the reviewer reached them

| Issue | Consequence |
|---|---|
| `_read_cache` clamped a negative age to zero | An entry written under a skewed clock looked permanently fresh — one NTP correction pinned the catalogue with no recovery but deleting the file |
| `_write_cache` used one temp name for all writers | Two sessions starting together interleaved into the same file, and the atomic rename then guaranteed the corruption arrived intact |
| A failed write stranded its temp file | One leaked file per failing start |
| `resolve_model_info` was a bare `lru_cache` | It outlives the disk TTL, so the server and scheduler workers would pin boot-time metadata for days while the disk cache refreshed underneath them |

The memo key now carries a TTL bucket, so an older bucket misses.

### The accent budget was the design round's central finding

Frame 4 carried **five** green items where the budget permits one, and the band
had it exactly inverted: the always-on brand glyph was accent while the
streaming spinner was `dim`, so the bottom row looked identical whether the
agent was working or idle — the one row an operator glances at for liveness.
Meanwhile `Keyword` painted every `def` in accent and `markdown.code` painted
inline code in the same green as the `+N` diff counts, so one hue meant both
"code literal" and "lines added" in one viewport.

Measured from the SVG CSS after the fixes — accent now appears **once per
frame**, on exactly what the budget allocates:

| Frame | accent `#38c96a` | success `#57c785` | signal `#6ea8d8` |
|---|---|---|---|
| picker open | `❯` (focused chevron), `/help` (selected name) | — | — |
| populated | `❯` | `+12` only | `parser.py` |

### Other design fixes worth naming

- **The band could jam its two groups one cell apart.** `_compose` padded with
  `max(1, …)` and the fit test only measured the composed row, so at ordinary
  widths (98, 116, 123) a filesystem path abutted a percentage with a gap
  *tighter than the separator used inside each group* — the left/right
  architecture dissolved into one run. The fit test now reserves a 4-cell seam,
  deliberately wider than the 3-cell intra-group ` · `.
- **The drop ladder protected a constant over a live number.** It shed context
  usage two rungs before `effort`, so a band could show `high` but not
  `49.6%/1M` — keeping the field nobody re-reads and dropping the one that says
  compaction is coming. Reordered to name → duration → subagents → shorten cwd →
  shorten model → effort → cost → context.
- **The last rung produced the band its own docstring called worse than
  nothing**: it dropped the model label, leaving a bare glyph on an empty strip.
  The label is truncated now — `kimi-k2-t…` still answers who is replying.
- **Enter could run an ambiguous fuzzy pick.** `/lo` highlights `loop` while
  `login` and `logout` also match, so a user reaching for login could start
  autonomous work in one keystroke, with their text rewritten and sent together.
  Enter now sends only when there is a single match or the name was typed in
  full; otherwise it completes and a second Enter confirms.
- **Short queries surfaced the matcher's whole tail** — `/u` offered
  `usage, quit, accounts, logout`. Fuzzy matching now starts at three
  characters, which is where real typos live (`/cmpct`, `/lgout` still work).
- **The scrollbar was a bright border.** At `dim` the thumb was a full-cell
  saturated column and the largest continuous fill in a narrow frame, abutting
  the tool cards so the filled slabs looked edged. Moved to `edge`; verified as
  14 rects at `#3b3527` with zero `dim` rects left.
- **Selection rested on one signal.** surface→raised measures 1.096:1, so the
  picker's "elevation plus accent" was really just accent, and hover gave a
  mouse user nothing. Added a `tint-select` ground — hue for state, elevation
  for row — the same move `tint-danger` already makes on a failed tool row.
- **The failed tool row's glyph left the scan column**, sitting ~25 cells left
  of every neighbour's `✓`. All four glyphs now measure at cell 91.

### Test-quality findings, including one my own fix exposed

Making `test_primary_column_aligns_descriptions` non-vacuous immediately proved
the point: one of its three probes was `"List MCP servers"`, and `mcp` is the
13th of 15 commands so it never rendered at all. The old assertion collected
positions only for probes that happened to appear, so `len(starts) == 1` passed
on the survivors. Two more of the same shape were fixed earlier in this branch.

`test_the_cache_write_is_atomic` was *named* for the concurrency guarantee but
wrote once, single-threaded — which is why the shared-temp-file race passed it
and had to be found by audit. It now drives three writer threads and three
readers with distinct payloads and asserts no reader ever observes a torn
document.

### Final state

| Gate | Result |
|---|---|
| unit tests | **1,766 passed**, 3 skipped (266 TUI, 1,328 core, 172 server) |
| `flake8` | clean |
| `black --check` | clean |
| `pyright` | 0 errors, 0 warnings across `local_operator/` |
| live TUI (OpenRouter) | splash, picker, real turn, `hello.txt` written, band showing `0.2%/1.0M · $0.0005 · 1s` |
| live exec `--json` | exit 0, pure-JSON stdout, 1 `agent_start`/1 `agent_end`, start context 2,407 tokens, **89.8%** warm cache rate, agent's own pytest passing and independently re-verified |
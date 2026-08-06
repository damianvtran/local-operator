# Performance & cost benchmarks

Measurable evidence for the harness's fixed cost and its per-task workload.
Run these with a real provider key; they hit the live API.

## Complex-task cost (`scripts/bench_task_cost.py`)

The predecessor of this section measured four prompts and trusted the agent's
own account of whether they worked. That is not a result. This benchmark
scaffolds four real codebases, runs the agent against them headlessly, and
then **checks the outcome from the benchmark process** — restoring each
contract test from a pristine copy the agent cannot reach, adding a hold-out
suite the agent never saw, and exercising the produced CLIs with inputs it
was never shown. A task the script cannot decide is reported as
`unverified`, never as a pass.

### The tasks, and why these

| task | what it is | why it is representative |
|---|---|---|
| `refactor` | A 3-module package with the SKU normalisation logic copy-pasted verbatim into each, behind a 7-test suite. Extract it once; keep the suite passing; do not touch the suite. | The commonest shape of real maintenance: no new behaviour to invent, several files to read before writing one, and a contract that must not move. |
| `debug` | A tiered-pricing module whose memo cache is keyed on `units` and silently drops `tier`, plus a failing test. Find and fix it. | Wrong only for certain call sequences and invisible in a single-case test. This is the reproduce/localise/fix loop, not a typo hunt. |
| `build` | Greenfield `iniround.py`: parse a specified INI dialect to JSON and emit it back, `parse(emit(parse(x))) == parse(x)`, exit 2 on a malformed line. | Precise interface contract including the parts implementations get wrong — an empty section, a value containing the delimiter, a specified non-zero exit code. Round-tripping is a property the script can check without trusting the agent's tests. |
| `longhaul` | Greenfield `sheetcalc.py`: tokeniser, precedence-aware expression parser, `SUM/AVG/MIN/MAX` over rectangular ranges, `#DIV/0!`, cycle detection. | Deliberately the longest trajectory in the set. It exists to answer whether the warm cache rate survives many turns of accumulating tool output. |

Every scaffold is written by the script, so the whole benchmark reproduces
from a clean checkout with no fixture data on disk.

### Method

Each task gets its own scratch directory and one
`local_operator.cli exec --json --yolo` run. Token accounting is read from
the `turn_end` events, using `context_tokens` as the prompt-side denominator
because OpenAI-compatible gateways report `prompt_tokens` *including* the
cached prefix while Anthropic reports it *excluding* it — `context_tokens` is
the field the provider adapters normalise, and it is the only denominator
that yields the same cache rate on both. Prices come from the live listing
via `local_operator.model.discovery`. **Warm cache rate excludes turn 1**,
which is the only turn that can legitimately miss.

Reproduce (both runs below were executed against a pinned `git worktree` at
commit `a305e1f` so that concurrent edits could not corrupt a paid
measurement):

```
OPENROUTER_API_KEY=... .venv/bin/python scripts/bench_task_cost.py \
    --out /tmp/lo-bench-cost --json-out /tmp/lo-bench-cost/results.json
```

### Measured

`deepseek/deepseek-v4-flash-0731` via OpenRouter, 2026-08-06, live prices
$0.09/M in, $0.18/M out, $0.018/M cache read:

| task | turns | tool calls | prompt tok | of which cached | output tok | warm cache | cost | wall | verified |
|---|---|---|---|---|---|---|---|---|---|
| `refactor` | 6 | 8 | 21 644 | 18 432 | 1 060 | 84.2% | $0.0008 | 20 s | PASS — 31 tests incl. hold-out; duplication gone (1 definition, was 3) |
| `debug` | 6 | 5 | 19 321 | 17 152 | 979 | 87.9% | $0.0007 | 21 s | PASS — 11 tests incl. hold-out |
| `build` | 7 | 6 | 27 192 | 21 056 | 1 953 | 76.7% | $0.0013 | 51 s | PASS — parse, round-trip and exit-2 probes |
| `longhaul` | 18 | 17 | 145 138 | 129 088 | 19 074 | 90.6% | $0.0072 | 264 s | PASS — 15/15 cells correct |
| **total** | **37** | **36** | **213 295** | **185 728** | **23 066** | **88.1%** | **$0.0100** | **355 s** | 4/4 |

The same four trajectories priced without caching come to $0.0233, so
caching saved 57.3%.

An agent is stochastic, so one sample is not a measurement. A second
independent run of the identical script:

| task | turns | warm cache | cost | wall | verified |
|---|---|---|---|---|---|
| `refactor` | 8 | 88.5% | $0.0010 | 21 s | PASS |
| `debug` | 6 | 86.9% | $0.0008 | 22 s | PASS |
| `build` | 6 | 86.1% | $0.0009 | 17 s | PASS |
| `longhaul` | 17 | 91.3% | $0.0110 | 602 s | PASS |
| **total** | **37** | **89.9%** | **$0.0136** | **662 s** | 4/4 |

Turn counts are stable to ±2. Cost is not: `longhaul` swung $0.0072 →
$0.0110 on 19k vs 45k output tokens for the same verified outcome. Wall time
swings harder still (264 s vs 602 s) and is provider queueing, not harness
work. Quote the token counts; treat a single wall-time figure as noise.

### Does the warm cache rate hold on a long task?

It holds, and it improves. `longhaul` — the longest trajectory in the set at
17–18 turns — posted the **highest** warm rate of the four in both runs
(91.3% and 90.6%), and its per-turn rate trends upward across the run,
finishing at its highest value:

```
turn  2    3    4    5    9   12   15   18
     90%  86%  53%  93%  92%  90%  93%  97%
```

The mechanism is visible in the raw numbers: of the 31 distinct
`cache_read_tokens` values these two runs produced, 30 are exact multiples of
256 (the sole exception is the 64-token floor discussed below), so a hit is
quantised to 256-token blocks. Each turn therefore misses the newly appended
tool result plus up to 255 tokens of granularity slack. `longhaul` turn 18 is
the arithmetic in full: prompt 13 221, cached 12 800, of the 421-token miss
211 are the tokens appended since turn 17 and 210 are block slack. That miss
is roughly fixed while the prompt keeps growing, so the rate rises. The ~90%
ceiling is the provider's block size, not prefix instability.

Two turns across the two runs read only 64 tokens (`build` turn 3,
`longhaul` turn 1). Those are shared-pool evictions, not broken prefixes: the
*next* turn recovered to 90%+ immediately, which is impossible if the prefix
had actually changed — a real break stays broken. This is also why the warm
rate excludes turn 1, whose hit rate depends on whether some earlier request
happened to warm DeepSeek's shared pool (it ranged 2.4% to 95.0% across the
eight task-runs and says nothing about the harness).

Short tasks score *lower*, not higher: `build` at 76.7% is the worst of the
eight. With only 6–7 turns there is no long stable prefix to amortise the
per-turn miss against.

### Where the cost actually goes

The usual claim is "the tool results, not the prompt". The data confirms it
for the prompt side and refutes it as an account of total cost.

The start context is not the cost driver. It is 2 424–2 744 tokens, and the
33–75% of prompt tokens it accounts for is entirely because it is re-sent
every turn — at cache-read prices that re-send costs about $0.00005 per turn.

What grows is the accumulated conversation, and that is overwhelmingly tool
traffic (bytes of the event stream, so the ratio is exact even though the
units are characters):

| task | tool results | tool args | assistant prose | tool share | prompt $ | output $ |
|---|---|---|---|---|---|---|
| `refactor` | 4 108 c | 2 180 c | 540 c | 92% | 76% | 24% |
| `debug` | 3 181 c | 721 c | 560 c | 87% | 74% | 26% |
| `build` | 2 855 c | 3 694 c | 972 c | 87% | 73% | 27% |
| `longhaul` | 9 192 c | 15 781 c | 1 115 c | 96% | 52% | 48% |

Three things follow, and the third is the one that matters:

1. Agent prose is 4–13% of what enters the context. Trimming the model's
   commentary is not a cost lever.
2. On short tasks the prompt side is 73–76% of the bill — but it is *cheap*
   in absolute terms precisely because 77–89% of it is cache reads at a fifth
   of the input price.
3. **Tool arguments outweigh tool results on the two build tasks** (15 781 c
   vs 9 192 c on `longhaul`). A `write` argument *is* the file body, and it
   is billed twice: once as output at $0.18/M — double the input rate, and
   never cacheable — and then as prompt on every subsequent turn. That is why
   output is 48% of `longhaul`'s cost in one run and 74% in the other, while
   never exceeding 27% on the short tasks. On long build-heavy work the
   dominant cost is the code the model *writes*, not the code it reads.

The practical consequence is that output volume, not context size, is the
lever on long build-heavy work. `longhaul` emitted 15 781 characters of file
bodies as tool arguments against `refactor`'s 2 180, and its output share of
cost is roughly double. Both tasks reached for `write` rather than `edit`, so
this data does not measure a diff-shaped tool against a whole-file one — but
it does say which side of the ledger such a change would land on, and it is
not the side a smaller system prompt touches.

## Base harness overhead

Constructing one session (imports + auth store + skills index + embedding
backend + tool inventory + transcript) adds **+49.3 MiB peak RSS** on top of
a bare interpreter (20.0 -> 70.3 MiB maxrss, measured). That is almost
entirely the enabled skills/embedding stack and the module graph; a session
with no skills skips the embedder. This is the fixed cost — everything else
is agent workload.

`scripts/bench_context_budget.py` measures the startup system-prompt size.
The figure depends on the skills corpus, so it is only meaningful quoted
alongside one:

| skills corpus | static listing | semantic worst case | tool schemas | total |
|---|---|---|---|---|
| 15 skills (`~/.omp/agent/skills`) | 2 859 | 1 392 (3 of 15 selected) | 1 175 | **2 567** |

That is 8.5% of the <=30k budget. Reproduce with:

```
.venv/bin/python scripts/bench_context_budget.py --skills-dir <corpus>
```

`scripts/bench_cache_rate.py` measures structural prefix stability: **94.1%**
(contract >= 90%). A live 30-turn trajectory measured a **94.2%** cache rate,
$0.0164 against $0.0517 uncached (68% saving). OpenRouter's shared pool does
not report cache statistics, so live figures require a direct provider key;
the structural number is the one this repo can reproduce unattended.

## Token / turn discipline

Values and skills are deliberately NOT baked into the context:

- `list_variables` / `read_variable` — variable VALUES never enter the
  system prompt; the agent lists names and reads one value on demand.
- Skills are selected semantically and **frozen per session**; the volatile
  skills block rides LAST in a fixed-arity block list so the conversation
  prefix stays byte-stable (the cache design).
- The env block is ~3 lines (platform, python, cwd), never a full dump.

## Install weight

The default install dropped from 63 packages / 112 MB to 25 / 23 MB, and seven
compiled wheels left the default path (faiss-cpu, numpy, pillow, pillow-heif,
cryptography, tiktoken+regex, websockets, psutil).

| install | packages | site-packages |
|---|---|---|
| default (`pip install .`) | 25 | 23 MB |
| everything (`.[all]`) | 56 | 76 MB |
| before this work | 63 | 112 MB |

Three compiled wheels remain in the default set, and it is worth being precise
about which, because "no build tools needed" depends on it:

| package | compiled | pure-Python fallback |
|---|---|---|
| `pydantic-core` | yes (Rust) | no — unavoidable, pydantic is the type foundation |
| `pyyaml` | yes (libyaml) | **no** — can hit an sdist build where no wheel is published |
| `charset-normalizer` (via requests) | yes | yes — degrades instead of building |

So the honest claim is that the *fragile* wheels are gone — the ones most
likely to lack a wheel for a freshly released CPython on Windows — not that
nothing compiles. `pyyaml` is the one remaining package that can force a
source build.

Optional features live behind extras (`server`, `mcp`, `images`,
`tokenizer`) and each degrades with an actionable message naming the extra
rather than an ImportError. `faiss-cpu`+`numpy` (34 MB of compiled code) were
replaced by an exhaustive inner-product scan, and Pillow by a ~100-line
grayscale PNG encoder; both are exercised by the test suite against
reference implementations.

## Browser use

The `browser` tool is advertised only when a CMUX browser is reachable, so
the model never sees a capability the host cannot honour. Verified end to end
against the real `cmux` binary — the agent opened `https://example.com`,
captured a 1600x1200 / 67 821-byte PNG, and confirmed the file itself with
`ls`:

```
browser {"action":"open","url":"https://example.com"}   -> surface:95
browser {"action":"screenshot","path":".../example.png"} -> saved
bash    ls -la .../example.png                           -> 67821 bytes
```

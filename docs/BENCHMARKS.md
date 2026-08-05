# Performance & cost benchmarks

Measurable evidence for the harness's fixed cost and its per-task workload.
Run these with a real provider key; they hit the live API.

## Complex-task cost (`scripts/bench_complex_tasks.py`)

Four non-trivial, multi-step tasks run through the real exec mode on
`deepseek/deepseek-v4-flash-0731` via OpenRouter (2026-08-05):

| task | wall | input tok | output tok | max context | cost |
|---|---|---|---|---|---|
| key-value store + tests, run | 24 s | 27 140 | 1 128 | 4 389 | $0.0026 |
| recursive-descent parser + tests, run | 86 s | 27 286 | 3 630 | 4 585 | $0.0031 |
| stdlib CRUD HTTP server + smoke test | 102 s | 81 908 | 7 104 | 5 986 | $0.0087 |
| shapes module + importer + grep verify | 32 s | 29 518 | 1 426 | 3 491 | $0.0029 |
| **total (4 tasks)** | **244 s** | **165 852** | **13 288** | — | **$0.0173** |

Notes:

- Pricing is fetched live from OpenRouter's `/models` (per-token scaled
  to per-1M: prompt $0.09 / completion $0.18).
- Tasks that stay under ~5k context do not trigger compaction; the httpd
  task ran many turns (81.9k cumulative input) yet the rolling context stayed
  under 6k — evidence the fixed context budget holds.
- `httpd` is the cost driver because it performs more tool turns, not because
  of harness overhead.

## Base harness overhead

Constructing one session (imports + auth store + skills index + embedding
backend + tool inventory + transcript) adds about **+50 MiB peak RSS** on top
of a bare interpreter (20 MiB -> 70 MiB). That is almost entirely the enabled
skills/embedding stack and the module graph; a session with no skills skips
the embedder. This is the fixed cost — everything else is agent workload.

`scripts/bench_context_budget.py` measures the startup system-prompt size:
**~2 000 tokens** against the <=30k budget.
`scripts/bench_cache_rate.py` measures structural prefix stability: **93.5%**
(contract >= 90%).

## Token / turn discipline

Values and skills are deliberately NOT baked into the context:

- `list_variables` / `read_variable` — variable VALUES never enter the
  system prompt; the agent lists names and reads one value on demand.
- Skills are selected semantically and **frozen per session**; the volatile
  skills block rides LAST in a fixed-arity block list so the conversation
  prefix stays byte-stable (the cache design).
- The env block is ~3 lines (platform, python, cwd), never a full dump.

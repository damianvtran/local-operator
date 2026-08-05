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

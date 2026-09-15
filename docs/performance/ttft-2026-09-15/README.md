# Time to first token, September 2026

The report was "the first message takes a while to start streaming, and it is
much worse in the desktop app than in the TUI". This is what was measured, what
was changed, and what the change is worth.

Everything below is a local observation on one macOS machine under load, not a
promise about provider latency. The provider is stubbed out entirely (see
*What the benchmark does and does not measure*), so every number here is
local-operator's own overhead — which is the only part this repository can move.

## The headline

| Path | What it is | Before | After | |
| --- | --- | ---: | ---: | ---: |
| TUI, first turn in a fresh process | `warm_session_imports` at boot, then one prompt | 464 / 541 ms | **56 / 36 ms** | 8.3x / 14.9x |
| Desktop, first message of a new session | POST `/messages` on a session with no runtime: cold engage, then the turn | 3670 / 5315 ms | **1044 / 1481 ms** | 3.5x / 3.6x |
| Desktop, second message | the same session with its runtime already up | 54 / 45 ms | **24 / 65 ms** | the per-turn path was never the problem |
| Cold engage + one complete turn | the same as row 2, measured to `agent_end` | 3176 / 2575 ms | **1017 / 1796 ms** | 3.1x / 1.4x |

Two independent rounds, interleaved (before, after, before, after) so the
machine's load — which swung between 48 and 282 during this work — could not
land on one arm. Raw results:
[round 1](desktop-before-r1.json), [round 1 after](desktop-after-r1.json),
[round 2](desktop-before-r2.json), [round 2 after](desktop-after-r2.json).

## The three costs

### 1. Every fresh process recompiled its own import graph (~1.1 s)

The desktop app spawns Python with **`PYTHONDONTWRITEBYTECODE=1`** and
`PYTHONPYCACHEPREFIX` set (`local-operator-ui`'s `python-bytecode-cache.ts`);
the daemon then hands that environment to every runtime child
(`session/runtime/launch.py` copies `os.environ`). Only the WRITE is refused —
the READ is not — but the cache was empty, so every child recompiled everything
from source. Measured inside one child's first turn: **501 modules and 749 ms of
`compile`**, and `local_operator/harness/types.py` alone was 633 ms of
`-X importtime` self time.

`local_operator/bytecode.py` populates that cache once, from a long-lived
process (the daemon's lifespan, and the TUI/CLI boot warm), using a subprocess
with the refusal dropped and the redirect kept. Later processes then read `.pyc`
like any other install.

Isolated on one runtime-child-shaped process:

| | time to first streamed token |
| --- | ---: |
| `PYTHONDONTWRITEBYTECODE=1`, cache cold | 1,277 ms |
| the same, cache populated once | **174 ms** |

The two safety properties, because this module writes files:

* It does nothing unless the prefix is set. The prefix is a *redirect*; without
  one, a write would land in `__pycache__` beside the source — which for an
  install inside a code-sealed `.app` is the unsealing the app's two variables
  exist to prevent. A prefix that points inside an `.app` is refused too,
  mirroring the UI's own `insideAppBundle` predicate.
* It writes only what CPython would accept (`_pyc_matches_source` reads the
  `.pyc` header, the same one `_validate_timestamp_pyc` reads), and the child is
  spawned with `-P` so it cannot import a checkout that merely happens to be the
  cwd.

### 2. `jedi`, imported on every session construction (~108-164 ms)

`tools/registry.py` imports `tools/lsp.py` eagerly to fill its factory table, so
the `lsp` extra's whole inference graph was executed for every session — and on
the desktop plane a session construction is a fresh runtime child, i.e. every
attach. The tool is now advertised through `find_spec` and the module is
resolved on first *use*. `tests/unit/test_import_graph.py` pins it.

### 3. The tokenizer, built inside the first request (~122 ms)

tiktoken's `cl100k_base` table is constructed on first use, and that first use
is `providers.context.measure_request` — which
`model.configure.SessionStreamFn.__call__` **awaits** before it opens the
provider request, so the whole cost is inside time-to-first-token. `122 ms` cold
against `0.004 ms` warm, once per process. It is now warmed at runtime-child
boot, at daemon startup, and in the TUI's existing off-loop boot warm.

## What the benchmark does and does not measure

`scripts/bench_ttft.py` drives three paths against the built-in `test` mock
provider, so nothing here is a provider's time to first byte:

* `tui` — a **fresh interpreter per measurement** running
  `warm_session_imports` and then two turns; the second turn is the steady-state
  number the same process reaches once warm. A child per measurement is the
  point: a TUI is one long-lived process, so its first turn is the only one that
  pays the cold caches, and looping in-process would report the settled number
  six times out of seven.
* `desktop-cold` — real uvicorn daemon, real HTTP + SSE, and the first message
  on a session with no runtime, so the POST carries the cold engage. One honest
  qualification: the harness opens the subscription and posts `/watch` before
  the measured message, and a visible watch lease arms a speculative warm of its
  own, so this can race a spawn the harness started. Both arms race identically.
* `desktop-warm` — the same session's next message, runtime already up.

It does **not** measure: the renderer (`local-operator-ui`), which is a separate
repository; a live provider's latency; or anything about total turn time beyond
the one `prime_ms` column.

Two harness settings matter and are forced rather than inherited: the app's spawn
environment (`PYTHONDONTWRITEBYTECODE=1` + a `PYTHONPYCACHEPREFIX` of the
harness's own), and `TIKTOKEN_CACHE_DIR`, pinned per invocation. Without the
latter the per-run `TMPDIR` made tiktoken re-download its 1.6 MB BPE file every
run — 413 ms of `SSLSocket.read` and 326 ms inside `load_tiktoken_bpe`, measured
in a profile — which is a property of the harness, not of local-operator.

Reproduce:

```sh
# baseline arm: a checkout of the base commit with its own venv
.venv/bin/python scripts/bench_ttft.py --runs 2 --pycache-prefix /tmp/before-pc \
    --json before.json
# candidate arm: the cache primed once, as the daemon does
.venv/bin/python scripts/bench_ttft.py --runs 2 --pycache-prefix /tmp/after-pc \
    --prime-bytecode --json after.json
```

## Deliberately not done

* **A pre-forked ("zygote") runtime pool.** The daemon could hold a warm
  interpreter and `fork()` children out of it, which would remove the exec and
  the whole import graph. It would also make every runtime child's build a
  property of the daemon's start time, and this repository has a documented
  history of `fork`/`close`/`flock` interactions on macOS. The bytecode cache
  above buys most of the same time with none of that risk.
* **Vendoring around `httpx._main`.** `import httpx` pulls `rich` and `typer`
  (~230 ms) purely to offer `httpx.main`, which local-operator never calls.
  Avoiding it means importing submodules of a third-party package directly,
  which is the kind of coupling that breaks on an upgrade.
* **Deferring the skill index / embeddings.** They are already cached on disk
  (`<config>/cache/*.skills.vec`) and cost ~250 ms only on the very first
  session after an install, which is not the per-attach cost.

## Follow-up worth a ticket

The desktop app sets `PYTHONDONTWRITEBYTECODE=1` on the daemon and everything it
spawns, to keep CPython from writing `.pyc` into its code-sealed `.app`. Its own
note says the cost is "paid on an app start"; it is actually paid **per runtime
child**, which is per attach. The `PYTHONPYCACHEPREFIX` it already sets is the
part that protects the seal, and it is set on every spawn; the refusal rides
along as a belt. Deciding whether the belt is worth ~1 s per attach is a
`local-operator-ui` question, and this repository now self-heals either way.

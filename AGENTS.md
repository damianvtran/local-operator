# Working on local-operator

Notes for agents (and humans) changing this codebase. This file covers the
things that are easy to get wrong here and expensive to discover later. For
what the rewrite set out to do see `docs/REWRITE.md`; for the evidence behind
each round see `docs/VERIFICATION.md`.

## Environment

```sh
cd ~/local-operator
.venv/bin/python -m pytest tests/unit -q          # ~2700 tests, ~3.5 min
```

TUI tests need a colour-capable terminal, so run them with the environment the
suite expects:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/unit/tui -q
```

`tests/e2e` is a separate stage that drives the **assembled** application —
boot, a real turn through a real tool, and `/resume` — and it is **deselected
from the default run** (`-m "not e2e"` in `addopts`). Run it explicitly:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/e2e -m e2e -n0 -q
```

It exists because the whole unit suite was green while the TUI was completely
frozen (#401: a blocking `flock` in the MCP OAuth refresh lock deadlocked the
event loop on `/resume`). Two things about it are load-bearing rather than
stylistic, and both are documented at length in `tests/e2e/watchdog.py`:

- **Its failure mode is a hang, not an assertion.** The deadlock parks two
  threads inside syscalls, so `asyncio.wait_for`, thread watchdogs and
  signal-based timeouts all fail to fire — verified, not assumed. Only
  `faulthandler.dump_traceback_later(exit=True)` survives it, because it runs
  in a C thread and needs no GIL. A tripped watchdog kills the process and
  writes every thread's stack to a file; `python -m tests.e2e.watchdog` prints
  it back.
- **It is why the stage is deselected and runs `-n0`.** Under xdist a fired
  watchdog would kill a worker carrying unrelated tests and report them as an
  infrastructure error rather than as the freeze they are.

It is fully headless (Textual's `run_test()` pilot, no window, no display, no
TTY) and uses no API key, so its CI job carries **no fork gate** — unlike
`cli-sanity`/`server-sanity`, whose live-LLM secrets force one. That is
deliberate: the resume-liveness assertion is the regression guard, so it has to
run on every PR including forks.

**It runs on a `[ubuntu-latest, macos-latest]` matrix, and the macOS leg is the
one that makes it a regression guard.** The deadlock is a macOS/BSD property —
`close()` blocks there behind a sibling `flock()`, and on Linux it returns in
microseconds. Measured, not assumed: the same probe reports `close_blocked=True`
on darwin and `False` on linux, and the resume test run against the pre-fix tree
(`80df237b^`) in `python:3.12-slim` reports `1 passed`. Linux still runs the
stage because boot, the write-turn artifacts and the transcript replay are
platform-neutral, but **do not drop the macOS leg** — without it this stage goes
green against the exact commit it exists to catch.

Gates, all of which must be clean before a PR. **Run them over the whole tree,
exactly as CI does** — these are the commands from `.github/workflows/ci.yml`:

```sh
.venv/bin/python -m flake8 .
uvx --from black==26.1.0 black --check .
uvx isort==5.13.2 --check .
.venv/bin/python -m pyright --pythonpath .venv/bin/python .
```

Narrowing the last two to `local_operator tests`, or passing `--profile black`
instead of letting isort read the repo's own config, checks something CI does
not: both combinations pass on a file that CI then rejects. An unsorted
function-local import reached CI exactly that way.

Editing `exclude` under `[tool.pyright]` in `pyproject.toml`? It **replaces**
pyright's built-in defaults rather than extending them, so always restate
`"**/node_modules"`, `"**/__pycache__"` and `"**/.*"` alongside whatever you
are adding. Dropping `**/.*` makes pyright follow the `.venv` symlink every
worktree has and type-check all of site-packages — 29466 files and a 30-minute
run instead of 566 files and about 15 seconds. CI never creates a `.venv`, so
it stays green while every local run of the gate becomes unusable.

The venv is uv-managed and has the package installed **editable**, so source
edits are live. After a pull that changes dependencies:

```sh
uv pip install -e ".[all,dev]" --python .venv/bin/python
```

## Releasing the stable `lop` runtime

Development and the global launcher deliberately use different installations:

- `uv run local-operator` and `.venv/bin/local-operator` execute the current
  checkout. Use them while developing and validating source changes.
- `lop` executes the non-editable uv tool installation under
  `~/.local/share/uv/tools/local-operator`. It must remain independent of the
  checkout so branch switches and uncommitted work cannot break the global TUI.

After a change is tested and committed to `main`, make it live with:

```sh
lop-update
```

`lop-update` archives the committed `main` ref, builds and installs that
snapshot, and records the exact source revision in
`~/.local/share/uv/tools/local-operator/.lop-source`. It never packages the
currently checked-out branch or uncommitted files. A specific committed ref can
be installed deliberately with `lop-update <git-ref>`.

Every agent asked to "update local-operator" or make a change available through
`lop` must treat runtime publication as a separate final step: merge the tested
change to `main`, run `lop-update`, verify the `.lop-source` marker, then smoke
test `lop` from outside the repository. Never repoint `lop` at the editable
`.venv`; doing so couples the stable command back to in-progress work.

## Versioning: choose the bump by materiality, not commit type

The version in `pyproject.toml` and the `vX.Y.Z` release tag are chosen by the
**user-facing materiality** of the change, **not** by its conventional-commit
type. A `feat:` commit is *not* automatically a minor. Using the commit type as
the version signal is how a run of bug-fix and reliability releases inflates the
minor number and drains its meaning — a minor should mark a step-function
improvement a user would notice and adopt, so that going from `0.N.x` to
`0.(N+1).0` still tells them something.

- **Patch (`0.N.x` → `0.N.(x+1)`) — the default; most releases are patches.**
  Bug fixes, performance and reliability improvements (backoff, retries,
  timeouts, tuning), refactors, internal cleanups, docs, and small
  self-contained features that do not change what the product can fundamentally
  do. A single small `feat:` commit is a patch. **When in doubt, patch.**

- **Minor (`0.N.x` → `0.(N+1).0`) — a material, step-function capability.**
  Reserve it for a new surface or subsystem a user would notice and adopt: the
  browser extension, the mobile relay, peer-to-peer session messaging. The test
  is simple — if you cannot name the step-function capability in the release
  title (`X.Y.0: <the new thing>`), it is a patch, not a minor. Several small
  features bundled together are still patches unless one of them clears this
  bar on its own.

- **Major (`X.y.z` → `(X+1).0.0`) — only on explicit request.** Bump the
  major version *only* when the developer explicitly asks for it, in the rare
  case where the new version is considered a distinct product from its
  predecessor. Never decide a major bump on your own judgement.

Because releases run frequently here, err toward patch: an under-called bump is
trivially corrected by the next release, while an over-called minor permanently
misreports how much changed.

## Visual validation: how to actually look at a UI change

This is a terminal UI. **A passing test is not evidence that a visual change
looks right**, and every spacing, layout, or animation change in this repo has
to be inspected as a rendered frame before it is claimed to work. The recipe
below is the one used for the usage-card spacing and the `/resume` picker; it
takes about a minute.

### 1. Render the screen to an SVG still

Two of these already exist and are worth reusing before writing a new one:
`scripts/ask_shot.py` and `scripts/approval_shot.py` capture the `ask` picker
and the tool-approval prompt over a seeded conversation, at any terminal size:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/ask_shot.py out.svg 100x30
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/approval_shot.py out.svg 100x30
```

Both seed real transcript blocks first, which is what makes "does this surface
still let me read the conversation?" an answerable question rather than a
screenshot of an empty app. `approval_shot.py` takes a third argument, `focus`,
which puts focus in the composer before the shot — the state that used to send
the prompt's answer keys into the prompt buffer.

**Note that both force the approval gate on** (`app._set_approve_all(False)`).
The app reads the developer's own `tool_approval_mode` from `~/.local-operator`,
so on a machine set to `auto` a naive capture shows a frame with no prompt in it
at all, and it looks like the surface is broken rather than skipped.

For anything else, Textual can export exactly what it painted. Drive the app
with `run_test`, put it in the state you care about, and save a frame:

```python
# /tmp/shot.py — env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/shot.py out.svg
import asyncio
import sys

sys.path.insert(0, "/path/to/local-operator")  # repo root, so `tests.` imports resolve

from tests.unit.tui.test_app_pilot import FakeSession, _factory
from local_operator.tui.app import OperatorApp

async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # ... put the app in the state under test: press keys, push a screen,
        # call a widget's show_*() directly ...
        await pilot.pause()
        app.save_screenshot(sys.argv[1])

asyncio.run(main())
```

**Use the real `OperatorApp`.** The lightweight hosts in the test files
(`_PanelHost` in `test_usage_panel.py`, `_PickerHost` in `test_session_picker.py`)
declare no `CSS_PATH`, so `local_operator.tcss` is **not applied** to them.
They are fine for asserting text content, and useless for judging padding,
colour, or placement — a still captured from one of them will not show a
stylesheet change at all.

### 2. Look at the image

An SVG is not something to eyeball as markup. Render it and view it — e.g.
open `file:///tmp/out.svg` in a browser tool and screenshot it, or open it in
any image viewer. The point is that a human or a vision-capable agent
**sees the frame**.

### 3. Always capture before AND after

**Capture `before.svg` FIRST, before you touch a file.** The before-frame is
the cheapest artifact in this recipe and it only stays cheap while the tree is
still clean — write the shot script, capture, then start editing:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/shot.py /tmp/before.svg
#   ... now make the change ...
env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/shot.py /tmp/after.svg
```

**Never `git stash` to get a before-frame.** Assume you are not alone in this
checkout: several agents routinely hold uncommitted work in it at the same
time, and a whole-tree operation is not a local undo — `stash` pockets every
peer's uncommitted work along with yours and hands it all back only if nothing
goes wrong in between. Nothing about the command tells you it did that. The
same applies to `git checkout -- <path>`, `restore`, `reset --hard` and
`clean`, and to any whole-file overwrite of a tracked file (`cp` over it, a
`>` redirect, an editor "revert"). Already edited and need a before-frame
anyway? Take it from a throwaway worktree, which cannot reach anyone's work
but yours:

```sh
git worktree add --detach /tmp/lo-before HEAD
ln -s ~/local-operator/.venv /tmp/lo-before/.venv
cd /tmp/lo-before && env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/shot.py /tmp/before.svg
git worktree remove --force /tmp/lo-before
```

Two stills side by side catch what a single "looks fine" never does. The
usage-card round found a **pre-existing** bug this way: the after-frame had a
scrollbar the before-frame did not, which turned out to be any tall overlay
pushing the screen's virtual height past its size — costing two cells of width
and reflowing the transcript behind the popup.

### 4. Check the numbers behind the frame, not just the pixels

Stills show you the symptom; the widget's own geometry tells you the cause.
Useful probes:

- `widget.size` (content box) vs `widget.styles.height` (border box — Textual
  sizes **border-box**, so a widget that pins its own height must add its
  padding back or it clips its own last rows).
- `app.screen.virtual_size` vs `app.screen.size` — if virtual exceeds actual,
  something is making the screen scrollable, and on this app that is always a
  bug (the transcript scrolls; the input is docked).
- `app.screen.show_vertical_scrollbar` — a scrollbar appearing is also a
  silent two-cell width loss.
- The `render_lines_for_test()` helpers on `UsagePanel` and
  `SessionPickerScreen` return the plain strings a user reads, which is the
  right thing to assert in a test.

### 5. Animation and multi-frame changes

For anything that animates or settles, capture **consecutive** frames
(`await pilot.pause()` between saves) and compare them. If the first painted
frame differs from the settled frame, the layout is reflowing after paint —
that is visible to the user as motion, whether or not anyone intended an
animation. Frames should be identical once settled.

The SVG goldens under `tests/unit/tui/__snapshots__` are a local design aid,
not CI: Textual's SVG output is not byte-stable across interpreters or OSes,
so they are opt-in (`LO_RUN_SNAPSHOTS=1`) and regenerated with
`--snapshot-update`. Do not add a golden as a substitute for looking at the
change.

## TUI conventions worth knowing before you edit a widget

- **Do not shadow Textual's API.** `Widget` already owns `query`, `visible`,
  `render`, and `_render`; a property or method with one of those names breaks
  focus, layout, or paint from inside your widget, and the traceback points
  somewhere else entirely (`'str' object is not callable`,
  `'Text' object has no attribute 'render_strips'`). Name list state
  `visible_rows`, filter state `filter_query`, and renderers `_card_text`.
- **Overlays float; they must not disturb the layout beneath them.** Cards on
  the `toast` layer are sized by the widget and positioned by an offset. Keep
  `overflow: hidden` on `Screen` so a tall overlay cannot introduce a
  scrollbar, and `event.stop()` in any mouse handler so one gesture does not
  move both the card and the transcript.
- **Wrapping vs clamping.** Arrow keys wrap (a discrete, deliberate press);
  wheel and page movement **clamp**. A scroll gesture that teleports to the
  other end of the list reads as the list resetting itself.

  **Documented exception — a long, sectioned, SCROLLED list clamps its arrows
  too.** The wrap rule is written for a short picker, where the whole list is
  on screen and coming round is a shortcut to a row the user can already see.
  It does not transfer to a page whose list is several times its viewport
  (`/settings`: 60-odd rows against 14 at 100x30). There the bottom is a
  destination the user travels to deliberately, and wrapping threw them out of
  the section they were working in and scrolled the viewport with them, which
  is what the report against v0.43.0 said. So `SettingsView.action_move`
  clamps, and every movement on that page clamps with it — a page where `down`
  clamps and `pagedown` wraps is worse than either rule applied uniformly.
  Tab or pane CYCLING is outside both rules: a small closed set of tabs that
  are all on screen (`←→` between the teams and agents panes) has no ends to
  clamp against and nothing that scrolls, so it keeps cycling.
- **Rows are load-bearing.** The welcome splash is content-sized and rests on
  the input card, so anything that changes its line count moves the whole
  block. Animated content must reserve its row even when it has nothing to
  show.
- **Comments explain the why.** This codebase documents constraints and the
  failure that motivated the code, not what the line does. Match that density;
  a comment that restates the code is noise, and a change with a non-obvious
  reason needs the reason recorded.

## Usage analytics (`local_operator/analytics/`)

Every provider call across every session contributes to one shared, on-disk
ledger (`<config_dir>/analytics.db`, SQLite/WAL/0600). `/analytics /usage`
opens an Esc-dismissable screen summarising token consumption: authoritative
provider counts (input/output/cache, and the thinking-vs-generation split of
output), an *estimated* dollar **cost** overlay, and an *estimated* breakdown of
input across the system prompt, custom instructions (agent/team profiles), tool
inventory, tool schemas, environment, knowledge, conversation, and tool
results. The frame is responsive — it grows to `max-width: 140` on a large
terminal (see the `.analytics-panel` CSS and `AnalyticsScreen._card_width`,
which must stay in step), and the per-provider/per-session tables shed the cache
column below `_WIDE_TABLE_MIN` to keep the cost column.

Things that will bite you if you forget them:

- **Recording is off the critical path and best-effort.** The wrapper is
  `SessionStreamFn._record_stream` in `model/configure.py`: it forwards the
  provider stream unchanged and records ONLY after the stream is fully
  consumed, so a turn's latency is untouched. The one on-loop cost is
  `analytics.model.snapshot_component_chars` (character-length reads;
  benchmarks under 0.4 ms even on a 340k-token context) — it must run on the
  loop because the transcript mutates the messages after the call. Tokenising,
  apportioning, and the SQLite write all happen on the recorder's background
  thread. Nothing in analytics may raise into a turn; every path is guarded.

- **One writer thread, one write connection.** `AnalyticsRecorder` funnels both
  call samples and session-name upserts through a single queue drained by one
  daemon thread. Do NOT add a second thread that writes to the store: two
  threads opening their first connection to a freshly-created SQLite file race
  in a way that leaves the writer unable to see its own commits (this cost a
  round of flaky tests). Reads open a fresh short-lived connection
  (`AnalyticsStore._read_connection`) so a report always sees the newest commit
  rather than a stale WAL snapshot.

- **Parallel-safe by the engine, not by us.** Several `lop` sessions write to
  the one file at once; WAL + `busy_timeout` + a bounded busy-retry make that
  atomic. The retry lives on the background thread, so accuracy under
  contention costs a session nothing.

- **The component split is an ESTIMATE and is labelled as one.** The provider
  bills one input total; the split is that total apportioned by character
  length (largest-remainder rounding, so it sums exactly). Authoritative counts
  and estimates must never be presented as the same kind of number — the screen
  says "estimated split of context tokens" for a reason.

- **Cost is computed at RECORD time, not aggregation time.** Dollar cost is
  priced in `SessionStreamFn._cost_micro` (`configure.py`) where the exact model
  is known, via the shared `cost_for_usage` — so analytics can never disagree
  with the status band. It is stored per call as `cost_micro` (USD × 1e6,
  INTEGER so the `SUM` is exact) plus a `cost_known` flag. A model with no
  published price records `cost_known=False` (rendered `$—`, never `$0.00`); a
  scope where some calls were unpriced is a LOWER BOUND (rendered `$12.30+`).
  Cost is an ESTIMATE (list price × tokens; it cannot see a plan, discount, or
  free tier) and is labelled as one, same discipline as the component split.

- **Adding a component OR a stored column is a schema migration.**
  `COMPONENT_KEYS` maps to one `c_<key>` column each in `store._SCHEMA`. A
  database from an older release is upgraded on open by `AnalyticsStore._migrate`
  (idempotent `ALTER TABLE ADD COLUMN`, since `CREATE TABLE IF NOT EXISTS` never
  alters an existing table) — this is how the `cost_*` columns reach a
  token-only ledger. Any new stored column needs a `_MIGRATION_COLUMNS` entry
  with a DEFAULT so old rows read as a sane value, never NULL.

## The tool-surface footprint ladder

Every core tool ships its schema on **every** API request. The tools array
rides in the same cache prefix as the system prompt (`tools/registry.py`
builds it in a stable order precisely so that prefix stays cache-stable), so
adding a core tool is a permanent per-call token tax on every session and every
subagent, paid whether or not the tool is ever called. The realized core surface
is on the order of a few thousand schema tokens — lean, and worth keeping that
way (the exact figure moves as `createIf`-gated tools drop out of a session that
cannot use them; `/context` reports the live number). Before adding a tool, take
the **highest (least-footprint) rung** that solves the problem:

1. **Extend an existing tool.** The capability is usually a variation of
   `bash`, `read`, `edit`, `grep`, or `eval`. A new parameter or mode on a tool
   that already exists costs no new schema. This is the default answer.
2. **A skill + `bash`.** Config, state, or infra work expressible as shell
   commands belongs in a `skill://` guide the agent runs through `bash`, not in
   a new tool. Zero model-schema footprint.
3. **A `createIf`-gated tool.** If the capability genuinely needs structured
   params AND only makes sense when a prerequisite is present, add it to
   `TOOL_BUILDERS` (`tools/registry.py`) as a factory that returns `None` when
   the prerequisite is absent — the way `build_wake_tool` returns `None` with no
   scheduler attached and `build_browser_tool` returns `None` with no browser
   surface. A gated tool costs zero schema in every session that cannot use it.
4. **An MCP server.** If it is tool-shaped but not core-fundamental, put it
   behind the MCP client: `mcp://` discovery is lazy, so its schema stays out of
   the prefix until the session enables it. Prefer this over a new core tool for
   anything integration-specific.
5. **A new core tool — last resort.** Only when the capability is fundamental,
   useful to nearly every session, and unreachable via the rungs above. The
   ungated core tools (`read`, `grep`, `eval`, `bash`) clear that bar; a niche
   or setup-specific capability does not. (`browser` reads as core but is
   actually rung 3 — `build_browser_tool` returns `None` without a browser
   surface — which is the ladder working as intended.)

**Gating answers reachability or opt-in, not host identity.** A `createIf`
factory may ask "is the dependency configured or reachable?" — it must not
encode "which host spawned me" in a way that strips a tool a reachable client
could otherwise use. And when a tool's cost is in doubt, measure it: add it,
run `/context`, and confirm the schema delta is justified by the capability.
Adding a second gating convention beside `createIf` is itself a footprint
regression — extend the table, do not invent a parallel mechanism.

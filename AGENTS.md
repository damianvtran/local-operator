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

**The suite caps its own parallelism.** `addopts` asks for `-n auto`, but the
root `conftest.py` implements xdist's `pytest_xdist_auto_num_workers` hook and
resolves that to a capped count (2-8) instead of the one-worker-per-core xdist
would otherwise pick. It has to sit at the rootdir: the hook runs before the
`tests/` conftest is loaded, so a copy under `tests/` is never called.

The reason is that this repo is worked through many concurrent worktrees. On a
14-core / 36 GB host, one-per-core meant 14 workers and a measured **3,661 MB
of xdist-worker RSS** against 1,579 MB for the capped run; three suites at once
drove load average to 98-128 and consumed 6.1 of 7.2 GB of swap. Fewer workers were also *faster* there — an interleaved A/B
on `tests/unit/server` measured `-n 4` at 5.3-5.9s against `-n 14` at
7.8-11.3s, because the suite waits on the event loop rather than on CPU. The
cap is the smaller of a CPU share and a memory budget, so a machine already
under pressure from sibling worktrees backs off on its own. The budget divides
by 600 MB per worker, which is a deliberate ~2.5x safety envelope rather than
the measured figure: cleanly measured workers sit at 226-262 MB, but a worker's
RSS depends on which tests it draws (worst observed ~1,090 MB) and some tests
fork their own subprocesses the budget must still cover.

The memory probe reads *current* pressure, which is subtler than it sounds:
`psutil` is deliberately not a dependency, so macOS is read from `vm_stat`
(`free + speculative + file-backed`) and Linux from `MemAvailable`. Two
plausible-looking alternatives are both wrong and are documented in
`conftest.py` so nobody re-adds them. Counting `Pages inactive` as free reports
phantom headroom on a swapping machine (8,137 MB while the host had 452 MB
free); `File-backed pages` is the subset `vm_stat` itself marks clean and
cheaply reclaimable, so no invented discount fraction is needed. And subtracting
`vm.swapusage` looks like a pressure signal but is **cumulative** — macOS never
decrements it, so a host that swapped hours ago stayed pinned to 2 workers while
the OS reported 78% free. Every term used here is instantaneous and recovers on
its own. Any probe failure degrades to the CPU-only cap; the hook never raises.

Override it per session, or bypass it entirely:

```sh
PYTEST_XDIST_AUTO_NUM_WORKERS=12 .venv/bin/python -m pytest tests/unit -q  # honoured unclamped
.venv/bin/python -m pytest tests/unit -n 12 -q   # explicit -n bypasses the hook
.venv/bin/python -m pytest tests/unit -n0 -q     # serialise for a debugger
```

**CI keeps every core.** The CPU share is skipped when `CI` is set: a hosted
runner is dedicated and single-purpose, so the contention the share protects
against does not exist there. Applying it anyway halved CI parallelism (a
4-vCPU runner resolved to 2 workers instead of 4), which is a regression paid on
every PR. The memory budget and the 2-8 clamp still apply on CI.

`--dist worksteal` is also in `addopts` — per-test
durations here vary by orders of magnitude, and the default `load` scheduler
pre-assigns chunks, leaving workers idle at the tail while one grinds through
the slow Textual pilot tests.

TUI tests need a colour-capable terminal, so run them with the environment the
suite expects:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/unit/tui -q
```

**A local failure CI does not have is usually your shell, and the fix belongs
in the fixture.** Tests that construct a third-party client inherit whatever
the developer exports. The OSWorld AWS tests scrub `AWS_PROFILE`,
`AWS_DEFAULT_REGION` and `AWS_REGION` in an autouse fixture for exactly this
reason — but botocore also reads `AWS_DEFAULT_PROFILE`, which the fixture
missed, so a developer with that exported gets 24 failures
(`ProfileNotFound: The config profile (sandbox) could not be found`) on a tree
that is green on CI. Confirm the diagnosis by unsetting the variable for one
run:

```sh
env -u AWS_DEFAULT_PROFILE .venv/bin/python -m pytest tests/unit/evaluation -q
```

Then add the missing name to the fixture's scrub list. Do **not** leave it as a
local workaround or a note in the PR: the next agent re-diagnoses it from
scratch, and a fixture whose docstring promises "no ambient configuration may
reach these tests" is simply wrong until it covers every variable the client
reads.

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

The `tui-e2e` job must not `need` the unit `test` job or `pip-audit`. It exists
because a green unit suite did not catch #401; gating it on `test` skipped the
freeze guard on every red unit run (observed on PR #426, which *fixed* a
deadlock while `tui-e2e` reported `skipping`). It needs only `lint` and
`type-check` — cheap syntax gates. A flake or a newly-published CVE must not
disarm the macOS resume-liveness assertion.

Gates, all of which must be clean before a PR. **Run them over the whole tree,
exactly as CI does** — these are the commands from `.github/workflows/ci.yml`:

```sh
.venv/bin/python -m flake8 .
uvx --from black==26.1.0 black --check .
uvx isort==5.13.2 --check .
.venv/bin/python -m pyright --pythonpath .venv/bin/python .
```

Do **not** invoke `.venv/bin/black`, `.venv/bin/flake8`, `.venv/bin/isort`, or
`.venv/bin/pyright` directly. Those console scripts carry a shebang baked in
at install time; after a parallel-agent worktree that owned the venv is
deleted they fail with `bad interpreter` and exit **126**. `rc=126` is
swallowed by a pipeline (`cmd | tail` reports `tail`'s 0), so a lint or
format gate that never actually ran looks green. `python -m` and `uvx`
cannot hit that path — that is why the commands above are spelled that way,
and why `make lint` / `make format` / `make type-check` go through them
rather than the console scripts.

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

### Every feature worktree owns its own venv. Never symlink one.

This repo is worked through many concurrent worktrees, and **each one gets a
real `.venv` installed editable from itself**:

```sh
git worktree add -b feat/my-change ~/workspace/repos/lo-my-change main
cd ~/workspace/repos/lo-my-change
uv venv --python 3.12 .venv
uv pip install -e ".[all,dev]" --python .venv/bin/python
```

It is tempting to symlink the parent's venv instead — it is one line and the
test suite passes either way. Do not. An editable install resolves imports
through a generated finder holding **one** hard-coded source root:

```python
MAPPING = {'local_operator': '/Users/you/workspace/repos/local-operator/local_operator'}
```

A symlinked venv therefore imports the tree it was *installed from*, not the
worktree you are standing in. The failure is silent and total:
`.venv/bin/local-operator` launches the TUI, the banner shows your branch's
version, the status bar shows your worktree's path, and every line of code
executing is `main`'s. An agent that "tested the feature interactively" this
way has tested nothing, and will report a working change as broken — or worse,
a broken change as working.

Verify which source a venv actually resolves before trusting an interactive
session, especially after any venv repair:

```sh
cd ~/workspace/repos/lo-my-change
.venv/bin/python -c "import local_operator; print(local_operator.__file__)"
# MUST print .../lo-my-change/local_operator/__init__.py
```

Wrong path? Rebuild the venv with the two `uv` commands above. `PYTHONPATH=$PWD`
is prepended ahead of the finder and will also work, but it is a per-invocation
plaster that the next command forgets — fix the venv instead.

**Why `cd` alone does not save you.** Python puts the *script's* directory on
`sys.path[0]`, not your current directory. For a console entry point that is
`.venv/bin`, so the worktree you are in never enters the path and the finder
wins. `python -c` and `python -m` do put the cwd first, which is exactly why a
quick `python -c "import local_operator"` can look correct while
`.venv/bin/local-operator` runs someone else's code — do not use the former to
clear the latter.

The one legitimate symlink is a **throwaway** worktree used to capture a
before-frame (see "Always capture before AND after"), where the parent's code
is what you want and the checkout is deleted minutes later. Scripts under
`scripts/` self-correct with `sys.path.insert(0, ...)` at the top, so they read
the tree they live in regardless. Anything you intend to *run as the product*
needs a real venv.

### A dead venv fails loudly; a wrong one does not

When the worktree that owned a shared venv is deleted, the console scripts in
it keep a shebang pointing at a `python` that no longer exists:

```
zsh: .venv/bin/local-operator: bad interpreter: .../lo-old-branch/.venv/bin/python: no such file
```

The fix is to reinstall editable **from the checkout you want to run**, not
merely to make the error go away:

```sh
cd ~/workspace/repos/local-operator   # or the worktree that should own it
uv pip install -e ".[all,dev]" --python .venv/bin/python
```

Repairing it from the *wrong* checkout converts a loud failure into a silent
one — the import now resolves, to the wrong tree. That has already cost a
session: a stale editable install was also collecting `tests/unit/tools/
test_eval_tool.py` as **24 failures** that three agents in a row wrote off as
"environmental". They were a broken venv, and they all pass once it points
somewhere real. Treat a standing block of "known environmental" failures as an
unverified claim, not as weather.

### Isolating a run: `LOCAL_OPERATOR_CONFIG_DIR` alone is not enough

`LOCAL_OPERATOR_CONFIG_DIR` redirects the **config** dir. It does *not* redirect
the **cache**: `model/catalogue.default_cache_dir()` derives its root from the
home directory independently (and `skills/index.py` and `update.py` hardcode the
same root, deliberately — one place to clear), so a run with only that variable
set reads and writes `~/.local-operator/cache` while believing it is isolated.

**Redirect `HOME` as well** — that is the reliable method, and it makes both
agree:

```sh
env HOME=/tmp/iso-run LOCAL_OPERATOR_CONFIG_DIR=/tmp/iso-run/.local-operator ...
```

This is worth spelling out because the failure is silent and it produces a
*plausible* wrong answer rather than an error. It has now cost two separate QA
rounds a false result in the same way: an "offline" cell resolved a provider
listing out of a catalogue cache that the same session's own earlier live calls
had written into the real home. Any cell whose point is a cold cache needs a
**fresh** `HOME` per cell, not merely a fresh config dir.

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

Four of these already exist and are worth reusing before writing a new one.
They capture the `ask` picker and the tool-approval prompt over a seeded
conversation, at any terminal size:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/ask_shot.py out.svg 100x30
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/approval_shot.py out.svg 100x30
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/ask_user_repro.py out.svg 150x40 [ROW] [reveal]
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/ask_long_shot.py out.svg 150x40 [ROW] [reveal]
```

- `ask_shot.py` — the picker carrying SHORT descriptions, which is the shape
  the card was originally measured against.
- `approval_shot.py` — the tool-approval gate, where each option's description
  is the consequence of authorising the call.
- `ask_user_repro.py` — the user's reported frame: three options with
  paragraph-long prose plus the free-text row, the reproduction for the
  description-truncation report and for `ctrl+e`.
- `ask_long_shot.py` — a LONG question as well as long descriptions, which is
  what makes the question's own wrapped lines compete with the option rows.

The last two take an optional `ROW` (arrow presses before the shot, so the
selected row's prose can be checked) and `reveal` (press `ctrl+e` first). Pass
both to check the property the reveal rests on: the card is the same height
whichever row the cursor is on, so two shots at different rows differ in their
text and in nothing else.

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

The symlink is safe **here and only here**: a shot script starts with
`sys.path.insert(0, <repo root>)`, so it reads the tree it lives in rather than
the one the venv was installed from, and this checkout is deleted immediately.
Never copy that line into a feature worktree — see "Every feature worktree
owns its own venv".

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

## Timing, flakes, and how to assert that something is fast

A test that spends a fixed budget waiting for asynchronous work, then asserts
as though the budget were a guarantee, is not a test — it is a bet on machine
load. On an idle dev box the budget is 20-50x more than the work needs, so it
looks solid; under `-n auto` contention on a 4-vCPU CI runner the work
stretches, the budget does not, and the suite goes red for reasons that have
nothing to do with the code under test. This repo has paid for that lesson
repeatedly (#122, #373, #461, #486, #496, #498, #499), so the rules below are
not style preferences.

### Wait on the event, never on the clock

**Do not** poll a wall-clock deadline and assert afterwards:

```python
await asyncio.sleep(2.0)              # "surely it finished by now"
assert store.detail is not None
```

**Do** wait on a publication from the code under test.
`tests/unit/harness/test_comms.py` has the pattern: `ChangeSignal` subscribes
to notifications the code already emits, and `wait_for` re-tests the predicate
after each one. The subscriptions are what make it work, so build and release
it in full:

```python
signal = ChangeSignal().watch_comms(comms).watch_session(parent)
try:
    await wait_for(lambda: comms.session_dir_of(job_id) is not None, signal=signal)
finally:
    signal.close()
```

A bare `ChangeSignal()` subscribes to nothing, so `wait_for` blocks the full
`DEADLOCK_GUARD_S` and then reports "it is wedged, not merely slow" — a
confident diagnosis of the wrong thing. Watch the sources you need.

There is no elapsed-time comparison anywhere on the success path: the wait
lasts exactly as long as the work does. The deadline exists so a genuine hang
fails the run instead of blocking forever. It is a backstop, not the assertion.

When there is nothing to subscribe to — an in-process coroutine the test
itself scheduled — bound by **loop turns** (`MAX_PUMP_TURNS`) rather than
seconds. A turn count survives contention that a wall-clock budget does not.

If you catch yourself tuning a `sleep` until CI goes green, you are
calibrating a bet, not fixing a test.

### Prefer a structural invariant to a numeric one

The strongest version of "this work happened off the event loop" is not a
timing bound at all — it is thread identity.
`test_store_maintenance_callbacks_run_off_the_event_loop_thread` wraps the real
maintenance callbacks, records `threading.get_ident()` inside each, and asserts
none of them equals the loop thread. That assertion cannot flake: it is a fact
about where the code ran, not how long it took, and it fails deterministically
the moment someone drops an `asyncio.to_thread`.

This is not a stylistic preference — it is where this repo's timing bounds keep
landing. `tests/unit/session/test_launch_subagent.py` carries the full account
under "WHY THIS IS A STRUCTURAL SPY AND NOT A TIME BOUND": a flat 1.0 s wall
bound failed, a calibrated bound failed, converting to `time.thread_time`
removed the *load* sensitivity but not the *core-speed* sensitivity, and the
conclusion was that **no portable numeric bound exists** for that site — the
only window left was between CI's healthy 1156 ms and this box's 1315 ms
regression, which is too narrow to sit a bound in. It was converted to a
structural spy. Read that docstring before you reach for a number.

Reach for a number only when no structural fact expresses the property.

### If you must measure, measure CPU, not wall time

Wall-clock gaps between probe wakes conflate two unrelated things: the event
loop genuinely blocking, and the OS simply not scheduling the process. On a
loaded runner the second dominates — measured here at 525-668 ms of "stall"
with the loop idle and nothing blocked.

Use `time.thread_time()`, which is per-thread and excludes time asleep or
waiting on the GIL, so a sample is large only when the loop thread really ran
without yielding. Two probes exist and they are **not interchangeable**:
`LoopCpuProbe` (`tests/unit/tools/test_loop_liveness.py`) records CPU only;
`StallRecorder` (`tests/unit/test_tui_responsiveness.py`) records CPU **and**
wall, and asserts both. If you need the blocking-sleep backstop described
below, use `StallRecorder` or add the wall assertion yourself — reusing
`LoopCpuProbe` silently drops it.

The distinction is measurable, not theoretical:

| scenario                   | wall gap    | CPU gap |
| -------------------------- | ----------- | ------- |
| 60 MB sync parse on loop   | 96 ms       | 96 ms   |
| pure `time.sleep` on loop  | 306 ms      | 0.1 ms  |
| loop idle, OS-starved      | 525-668 ms  | 0.0 ms  |

Note the middle row: the CPU clock is **blind to a pure blocking sleep**. If
that shape matters at your site, keep a wall-clock assertion alongside it, with
a ceiling set for catastrophe (seconds) rather than for precision.

### Calibrate ceilings from CI, never from your laptop

This is the mistake that cost three PRs. Local measurements of the same
loaded-bar sites read 36-38 ms and ~0 ms; the CI runners recorded **206, 264,
321, and 413 ms** of entirely legitimate loop CPU at those same sites. Every
ceiling calibrated from the local numbers flaked a green tree within a day.

A slower core burns more CPU-seconds on identical work, so a bound calibrated
on a dev box is not a bound on CI — see the `test_launch_subagent.py` account
above, where that spread ended in abandoning the bound entirely. If you have
established a number is genuinely the only option:

- Take the healthy maximum from **CI logs across several runs**, not from one
  local run and not from one CI run.
- Leave real headroom above it, and **write the dataset and the margin into the
  docstring** so the next person is not re-deriving it from scratch.
- A number copied from an older comment is not evidence. The 549 ms figure in
  this repo's history is a **wall-clock** measurement from a probe that no
  longer exists; citing it to justify a CPU ceiling compares two different
  quantities.

When a green tree trips the ceiling, **find out which it is before touching the
number**. Either the tree is not as green as it looks, or the ceiling is
miscalibrated — and "raise it" is only correct in the second case. Reproduce
the sample, check whether the same test fails on `main` and on unrelated
branches, and confirm the work under the probe is the legitimate kind. Widening
a bound because it went red is how a guard stops guarding.

### Prove the test can still fail

A guard that cannot go red is worse than no guard, because it is believed. When
you change how a test detects a regression, reintroduce the regression and
watch it fail:

```sh
# put the synchronous call back on the loop, or delete the to_thread
.venv/bin/python -m pytest tests/unit/test_tui_responsiveness.py::<test> -n0
# expect a failure with a message that names the real cause, then revert
```

This is not hypothetical. In `test_tui_responsiveness.py` the loaded bar was
raised to 500 ms to stop it flaking, which meant a 200-500 ms stall at the
boot and turn sites — whose only guard is that bar — would pass unnoticed.
That trade was made deliberately and on evidence: those sites recorded 321 ms
and 413 ms of *legitimate* loop CPU on CI, so a tighter bar failed green trees
instead. The strict 50 ms bar on the reconnect/connect sites was left alone
precisely so the 90-130 ms parse regression stayed catchable. When you widen a
ceiling, name what you just stopped catching — and record the measurements that
forced your hand.

Then check the other direction: run the file several times under load
(`for i in $(seq 1 5)` with a few CPU spinners) and confirm it stays green.

### When a test is already flaking

Reproduce and classify before touching a threshold. A single failure out of
~9900 on a different timing-sensitive test each run is a flake, not a
regression; the same test failing on several branches at once — check `main` —
is repo-wide and not yours. Re-run to confirm, then fix the measurement rather
than widening the bound. Do not merge a red head on the assumption that it is
"probably the known flake" without checking the log.

### Two things this section cannot do for you

**There is no `pytest-timeout` in this suite.** A test that waits forever hangs
its CI job until the workflow's `timeout-minutes` reclaims the runner (40 min
for `test`, and that ceiling exists because a job once held a slot for 3h38m).
So an unbounded wait is not merely slow, it is expensive for everyone queued
behind it — which is the other half of why `wait_for` carries
`DEADLOCK_GUARD_S`.

**A deadlock defeats every technique above.** If the failure mode is the event
loop freezing at the C level rather than running slowly, a Python-level probe
reports nothing — the coroutine that would record the sample never gets
scheduled either. Nor do the obvious fallbacks: `watchdog.py` documents, with a
measurement, that a `threading.Thread` watchdog needs a GIL the wedged thread
never releases, and `pytest-timeout`'s default `signal` method never fires
because a Python signal handler only runs between bytecodes. The pre-fix code
ignored a 20 s thread watchdog and had to be `kill -9`'d.

That class is covered by `tests/e2e/watchdog.py`, whose
`faulthandler.dump_traceback_later(exit=True)` is armed in C and fires from a
separate OS thread, so it survives a wedged interpreter and takes the process
down with a full thread dump. It runs as its own CI stage (`tui-e2e`, both
Linux and macOS) rather than in the unit run, since firing it under `-n auto`
would kill a worker carrying unrelated tests. If you are guarding against a
hang rather than a stall, that is the file to read.

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

  **Documented exception — a list that IS the whole page clamps its arrows
  too. Today that is `/settings` and nothing else.** The wrap rule is written
  for a picker: a short list overlaid on a screen the user is still looking at,
  where coming round is a shortcut to a row already visible. It does not
  transfer to a full-page mode whose list is several times its viewport
  (`/settings`: 60-odd rows against 14 at 100x30). There the bottom is a
  destination the user travels to deliberately, and wrapping threw them out of
  the section they were working in and scrolled the viewport with them, which
  is what the report against v0.43.0 said. So `SettingsView.action_move`
  clamps, and every movement on that page clamps with it — a page where `down`
  clamps and `pagedown` wraps is worse than either rule applied uniformly.

  Tab or pane CYCLING is outside both rules: a small closed set of tabs that
  are all on screen (`←→` between the teams and agents panes) has no ends to
  clamp against and nothing that scrolls, so it keeps cycling.

- **One gesture owns the viewport at a time.** A list inside a
  `ScrollableContainer` has TWO positions that can each move the view: the
  container's own scroll offset (wheel, scrollbar drag) and a cursor whose
  "scroll the selection into view" recomputes the offset from the selected row.
  Let both drive the same view from the same gesture and they overwrite each
  other — on `/settings` the wheel moved the cursor and then re-derived the
  viewport from it, so wheeling to the bottom and giving one more notch snapped
  the view back to the top (reported against v0.43.10).

  The split that holds: **the wheel and the scrollbar move the VIEWPORT and
  leave the cursor alone; keys that move the CURSOR scroll it into view.** The
  cursor is then allowed to go off screen, which is what every editor and list
  UI does and the only arrangement in which the scrollbar thumb is telling the
  truth.

  Two traps when handing the wheel to the container. Textual stops the wheel
  event on the container *while it can still scroll*, so a widget-level
  `on_mouse_scroll_*` handler runs **only at the ends of the travel and outside
  the container's region** — which is why this defect looked like "it bounces
  when I reach the bottom" and why the same notch behaved differently over the
  pane than over the list. And accumulate from `scroll_target_y` with
  `immediate=True`, not `scroll_relative`: the default defers to
  `call_after_refresh`, so notches arriving in one burst all read the same stale
  offset and a fast flick collapses to a couple of rows.

  Step the handler by the LIVE `app.scroll_sensitivity_y`, never by a constant
  copied from it. It is a per-instance attribute set in `App.__init__`, so a
  hardcoded step that matches today desynchronises the moment anything changes
  it — measured at 4.0, the container applied 4 rows per notch over the list and
  the widget handler 2 everywhere else, which is the position-dependence above
  reappearing.

  Finally, **the cursor being allowed off screen is a contract with the keys**:
  once the wheel can leave it behind, every key that ACTS on the cursor has to
  scroll it into view first, or it writes to a row the user cannot see and the
  frame does not change. Reveal-then-act, not an interlock that makes the first
  press a no-op — the press should still do what it says on the first try.

  The trigger is "the list is the whole page", NOT "the list scrolls", and the
  difference is load-bearing: `model_picker.move` windows a catalogue of
  hundreds of rows and still WRAPS, correctly, because it is an overlay on the
  conversation rather than a place the user has navigated to
  (`command_picker.move` likewise). Do not read this exception as licence to
  clamp them. `session_picker._move_to` already clamps, and it is a full
  surface too.
- **Rows are load-bearing.** The welcome splash is content-sized and rests on
  the input card, so anything that changes its line count moves the whole
  block. Animated content must reserve its row even when it has nothing to
  show.
- **Comments explain the why.** This codebase documents constraints and the
  failure that motivated the code, not what the line does. Match that density;
  a comment that restates the code is noise, and a change with a non-obvious
  reason needs the reason recorded.

## Adding a configuration key

**Every new configuration key must also be added to the `/settings` registry
(`local_operator/settings_io.py`).** A key that only exists in the code that
reads it is invisible: `/settings` is the surface users browse to discover what
is configurable, and `lop config edit`/`config list` resolve names out of that
same registry, so an unregistered key can only be set by someone who already
knows it exists and edits `config.yml` by hand.

Adding one means, in the same change:

1. A `Setting` in `SETTINGS`, in a declared `Section`. If the section is new,
   add it to `SECTIONS` too — and pick its `Scope` honestly. Scope is uniform
   within a section by construction, so a key that takes effect immediately does
   not belong in a section labelled "new sessions"; split the section instead.
2. **A real module-level default constant next to the code that reads the key**,
   mapped in `_consumer_defaults()` in `tests/unit/test_settings_io.py`.
3. `path=` spelled deliberately. Almost every key is a genuinely NESTED tuple
   (`("fork", "mode")`). The `display.*` flags are the exception — they are
   literal dotted keys at the top level because that is what `tui/settings.py`
   reads — and getting this backwards writes a key nothing reads while looking
   like success from every angle.

`test_every_default_matches_its_consumer` is the enforcement, and it fails **by
name** for any setting with no consumer entry. There is an allow-list
(`_NO_SINGLE_VALUE_CONSUMER`) for keys that genuinely have no single-value
consumer, guarded by its own staleness test — it is for free-text keys whose
"default" is "unset means inherit", not an escape hatch for a key you did not
feel like wiring up. Using it to silence the failure defeats the guard: the
whole point is that a registry default disagreeing with the code's default is a
painted lie that nothing else reports.

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

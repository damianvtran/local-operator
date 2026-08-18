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

Gates, all of which must be clean before a PR:

```sh
.venv/bin/python -m flake8 .
uvx --from black==26.1.0 black --check local_operator tests
uvx isort --check-only --profile black local_operator tests
.venv/bin/python -m pyright --pythonpath .venv/bin/python .
```

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
- **Rows are load-bearing.** The welcome splash is content-sized and rests on
  the input card, so anything that changes its line count moves the whole
  block. Animated content must reserve its row even when it has nothing to
  show.
- **Comments explain the why.** This codebase documents constraints and the
  failure that motivated the code, not what the line does. Match that density;
  a comment that restates the code is noise, and a change with a non-obvious
  reason needs the reason recorded.

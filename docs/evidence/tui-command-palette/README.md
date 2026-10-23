# `ctrl+p` command palette — rendered evidence

Frames exported from the **real `OperatorApp`** (not a stylised host: the
`OperatorApp` CSS and `local_operator.tcss` are applied), via
`scripts.visual_capture.save_capture` with `isolate_capture()` called before any
app import. Both sizes are the ones the slice's acceptance names: **100x30** and
**60x24**. `*.geometry.json` beside each still records the grid, native pixel
size, resolved font face and per-widget boxes that back the image.

## What each state is

| State | How it is produced |
|---|---|
| `after/discover` | `ctrl+p`, empty query — the app catalogue's discovery order |
| `after/populated` | `ctrl+p`, then `sett` — the `/settings` hit with its dim help line |
| `after/no-match` | `ctrl+p`, then `zzzzzzzz`, held past the framework's 0.5 s no-match countdown |
| `after/empty` | `ctrl+p` with **both** catalogues stubbed empty (no app rows, no system actions) — the framework's empty discovery state |
| `before/discover` | same keystroke on `origin/main` (`8426d1cac1`): placeholder `Search for commands…`, only the five framework system actions |

`before` is the "empty catalogue" the slice started from. There is no useful
`before` frame for the other two states — on `origin/main` a query of `sett` and
of `zzzzzzzz` both produce the same "No matches found" list, which is the
absence the slice exists to fix.

## How to reproduce

```sh
# from the worktree root; the script lives in the session scratchpad and is
# reproduced here because it is 40 lines and behaviour is what matters
cat > shot.py <<'PY'
import asyncio, sys, os
REPO, OUT, STATE, SIZE = sys.argv[1], sys.argv[2], sys.argv[3], tuple(int(x) for x in sys.argv[4].split("x"))
sys.path.insert(0, REPO)
import scripts.probe_isolation  # noqa: F401 — isolates HOME/config before app imports
from scripts.visual_capture import save_capture
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from local_operator.tui.app import OperatorApp

QUERIES = {"discover": "", "populated": "sett", "no-match": "zzzzzzzz", "empty": ""}

async def main():
    app = OperatorApp(lambda: _factory(FakeSession()))
    if STATE == "empty":
        app._palette_entries = lambda: []
        app.get_system_commands = lambda screen: []
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        if QUERIES[STATE]:
            await pilot.press(*QUERIES[STATE])
            await pilot.pause(0.8 if STATE == "no-match" else 0.2)
        save_capture(app, os.path.join(OUT, f"{STATE}-{SIZE[0]}x{SIZE[1]}.svg"))

asyncio.run(main())
PY
env -u NO_COLOR TERM=xterm-256color .venv/bin/python shot.py "$PWD" out after populated 100x30
rsvg-convert out/populated-100x30.svg -o out/populated-100x30.png
```

`rsvg-convert` (librsvg) is the rasteriser, as `docs/VISUAL_CAPTURE.md` requires;
`qlmanage` is never used from a rig.

## What the frames show (and what was checked by looking)

- **Order is fixed, system actions first.** `discover` lists Keys, Maximize,
  Quit, Screenshot, Theme, then `/help`, `/keys`, `/exit`, `/clear`, `/copy`… —
  the rows `origin/main` already offered, followed by the app catalogue. The
  same order is asserted in `test_discovery_keeps_the_framework_actions_then_the_app_catalogue`;
  this is why the slice registers one provider instead of four (`App.COMMANDS`
  is a set and every discovery hit scores 0).
- **The hit is two lines**: label (`/settings`) above the dim `command · …`
  help line. At 60x24 that line is still complete for this hit, and a longer
  help line wraps to a second line (the `Keys` row does) — the framework's
  unchanged rendering, not this slice's copy budget.
- **No-match** centres `No matches found` in the list region and leaves the
  input live; at 60x24 it is one line and does not push the input (viewed at
  both sizes).
- **Empty** shows the input alone — the list region collapses, no stray rule or
  spinner is left behind.
- The placeholder at 60x24 (`Search commands, settings, sessions, agents…`, 44
  cells) fits the field without truncation, which is the `maxCells: 48` budget
  the context sidecar records.

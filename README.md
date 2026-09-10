# Evidence — durable audit history behind the display window

Rendered frames for the fix that makes pre-compaction history reachable
(`fix(session): make pre-compaction history reachable for audit`). See the PR
for the measured numbers; this branch holds the pixels those numbers explain.

> **Why this is a separate branch, not part of the PR.** `AGENTS.md` §7
> ("Evidence goes on the PR, never into the repository") bans committing frames
> into the merged tree, and names `docs/evidence/<change>/` specifically —
> those directories were once swept out at ~60 MB of frames nothing loaded.
> This branch is **orphaned, never merged, and deleted when #899 merges**, so
> the frames are reviewable without ever entering `main`'s history. The
> reusable part — `scripts/audit_history_shot.py` — lives in the tree on the
> PR branch, which is exactly what §7 says belongs there.

## How these were captured

The **real** `OperatorApp` driven against a **real** `RuntimeServer` over the
actual control socket — not a lightweight test host, which declares no
`CSS_PATH` and would apply none of the stylesheet. Pages therefore arrive
through the same `history_page` RPC production uses.

```sh
# after/ frames — this branch
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/audit_history_shot.py <out-dir> all 100x34
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/audit_history_shot.py <out-dir> all 140x50

# before/ frames — a detached worktree at origin/main, same script, same fixture
env -u NO_COLOR TERM=xterm-256color PYTHONPATH=<base-worktree> \
    .venv/bin/python scripts/audit_history_shot.py <out-dir> <state> <geometry>
```

The script clears every `CMUX_*` variable before importing the app and isolates
`HOME`/config through `scripts.visual_capture.isolate_capture()`. Fixture is
synthetic (6 compactions x 400 rows, or 3 x 30 for `marker`) — no real session
data is in any frame.

## Why two geometries

A head-notice state is a function of the **viewport** as well as the history:
`_reconcile_head_notice` picks between its scrollable and not-scrollable copy
from `virtual_size` against `container_size`. A frame captured at a single size
can therefore show a state that does not reproduce at another, so every claimed
state is captured at both.

| geometry | `run_test(size=...)` | screen size | region |
|---|---|---|---|
| `100x34` | `(100, 34)` | `98 x 32` | `[0, 0, 100, 34]` |
| `140x50` | `(140, 50)` | `138 x 48` | `[0, 0, 140, 50]` |

Screen size is 2 cells smaller than the requested size in each axis because the
transcript sits inside the app's border. Full numbers, including font
provenance and per-widget boxes, are in the `.geometry.json` beside each frame
when regenerated (not committed — they are a capture byproduct).

**Result: every state below renders the same copy at both geometries.** No
claimed state is geometry-dependent.

## The frames

Each state was captured twice, before and after a further settle, and every
pair was **byte-identical** — so there is no reflow between first paint and
settled frame, and only one frame per state is committed. (Regenerate with the
script to get the `.settled.svg` twin and re-check with `cmp`.)

| frame | shows |
|---|---|
| `before/context.*` | Context rows still remain. Head notice: `older messages above — scroll up to load`. **Unchanged by this fix** — included so the reviewer can confirm the non-audit path did not move. |
| `before/audit.*` | **The reported defect.** The reader has drained the model's replay; the head row says `start of conversation` above `question 280`, with ~2,400 rows still on disk and unaddressable. |
| `before/exhausted.*` | Same claim, chain fully drained — identical to `before/audit`, because on `main` the chain has nowhere further to go. |
| `after/context.*` | Byte-identical copy to `before/context`: the context phase is untouched. |
| `after/audit.*` | Same point in the same conversation as `before/audit`. Head notice now reads `earlier history above — scroll up to load`, the compaction marker sits directly below it, and phase-5 pre-compaction content is on screen. |
| `after/exhausted.*` | `start of conversation` above **phase 0, item 0** — the journal's genuine first message row. The notice is now a true statement rather than a report of what the layer could reach. |
| `after/marker.*` | The compaction boundary mid-transcript, with rows on both sides: `In phase 2 we settled item 29` (pre-compaction) above the marker, live context below. This marker rendered as **nothing** before the fix — `project_settled_rows` had no branch for `compaction_summary` — so there is no `before/` twin to pair it with. |

## Head-notice strings, verbatim

Printed by the capture script at the moment each frame was taken.

| state | `main` (before) | this branch (after) |
|---|---|---|
| context rows remain | `older messages above — scroll up to load` | `older messages above — scroll up to load` |
| context drained, audit rows behind | `start of conversation` **(false)** | `earlier history above — scroll up to load` |
| chain fully drained | `start of conversation` | `start of conversation` **(now true)** |

`audit=` and `more=` in the script's output are `RemoteSession.history_is_audit`
and `bool(history_before_token)`, which are what the notice branches on.

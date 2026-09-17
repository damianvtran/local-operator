# `/goal --clear` and `/loop --stop|--clear` — picker autofill evidence

Frames for the flag forms and the argument-picker row that teaches them, captured
with `scripts.visual_capture.save_capture` through the real `OperatorApp` (the only
host that loads `local_operator.tcss`). Reproduce with:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/slash-goal-loop-clear/shot_slash_flag_picker.py OUT.svg 100x30 CASE
```

`CASE` is one of `goal-set` (default), `goal-none`, `goal-typed`, `loop-running`,
`loop-idle`. Each run writes the frame, its `.geometry.json`, and a SECOND frame
one pause later named `*-settled.svg`; the pair is byte-identical for all ten
captures here (`cmp` on each pair reports no difference), which is the
no-reflow check AGENTS.md §5 asks for. The settled duplicates are not committed —
the script regenerates them, and re-running `cmp PATH.svg PATH-settled.svg` is the
check.

The before frames come from a throwaway worktree at `d7b53582a` (this branch's
base) using this same script and fixture:

```sh
git worktree add --detach /tmp/lo-before HEAD && mkdir -p /tmp/lo-before/docs/evidence/slash-goal-loop-clear
cp docs/evidence/slash-goal-loop-clear/shot_slash_flag_picker.py /tmp/lo-before/docs/evidence/slash-goal-loop-clear/
cd /tmp/lo-before && env -u NO_COLOR TERM=xterm-256color \
    ~/local-operator-worktrees/goal-loop-clear/.venv/bin/python \
    docs/evidence/slash-goal-loop-clear/shot_slash_flag_picker.py /tmp/before-CASE.svg 100x30 CASE
```

## What to read off the frames

* **`goal-set`** — `before`: `/goal ` in the composer, picker CLOSED, nothing
  offered. `after`: one dim row under the composer, `--clear  Clear the standing
  goal`, plus the ghost completion of `--clear` at the caret. This is the whole
  requirement: the flag is now reachable without guessing the bare word `clear`.
* **`loop-running`** — the same pair for `/loop `, offering `--stop  Stop the
  running loop`.
* **`goal-none` / `loop-idle`** — `before` and `after` are identical: no goal ⇒
  no `--clear` row, no loop ⇒ no `--stop` row. The offer is gated on the live
  state, so the palette never advertises a no-op.
* **`goal-typed`** — `/goal ship it` offers nothing in either frame: free text in
  the argument region closes the list and submits the goal unchanged.

## Geometry behind the frames

`100x30` pilot. Printed by the script (`picker` = the command picker widget):

| case | frame | picker mode | rows | picker content vs pinned | transcript | screen virtual vs size | vscroll |
| --- | --- | --- | --- | --- | --- | --- | --- |
| goal-set | before | `command`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |
| goal-set | after | `argument`, open | 1 | 96x1 / 1 | 97x20 | 98x28 / 98x28 | no |
| goal-none | before | `command`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |
| goal-none | after | `argument`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |
| goal-typed | before | `command`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |
| goal-typed | after | `argument`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |
| loop-running | before | `command`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |
| loop-running | after | `argument`, open | 1 | 96x1 / 1 | 97x20 | 98x28 / 98x28 | no |
| loop-idle | before | `command`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |
| loop-idle | after | `argument`, closed | 0 | 0x0 / auto | 97x21 | 98x28 / 98x28 | no |

Two numbers matter and both are clean:

* **`virtual == size` and `vscroll: no` in every frame.** The row does not push
  the screen into scrolling, which on this app is always a bug (the transcript
  scrolls; the input is docked). A scrollbar would also cost two cells of width.
* **`picker content 96x1 == pinned 1`.** The widget pins its own height, so the
  one row is the whole border box — nothing is clipped off the bottom.

The transcript box is one row shorter (97x20) exactly where a row is shown, and
back to 97x21 where it is not: the picker is docked above the composer and takes
its row from the transcript, which is the same trade the command picker makes.
The composer (92x1) and the band are unmoved in every frame.

## Round 1 remediation (`round-1-remediation/`)

The five cases above were re-captured after the review round and are
**byte-identical** to the committed ones (`git status` reports no change to the
SVGs), which is the evidence for the fix's central claim: `alert=True` on the
two rows changes the GATE and not one pixel — the row is always `selected`, and
`command_picker._argument_row` skips the danger tint on the selected row by
design. The one defect that re-capture did fix was the script itself: it now
WAITS for session adoption before assigning the buffer, because the `--clear`
row's gate reads `self._session.goal`, which is empty until the app has adopted
its session (QA Q1: three runs gave `[True, True, False]`, and an earlier batch
wrote a row-less `goal-set` frame at 6716 B against the committed 7464 B).
Three runs per case now reproduce the published geometry exactly.

The frames here are the round-1 remediation delta, captured from
`cef6a4c4e` (before) and the remediation head (after) with the two scripts added
in this round:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/slash-goal-loop-clear/shot_slash_flag_flow.py DIR CASE
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/slash-goal-loop-clear/shot_slash_help_row.py OUT.svg 80x50
```

| frame | case | what it shows |
| --- | --- | --- |
| `after/goal-fill.svg` | `shot_slash_flag_flow.py DIR goal-fill` | ONE Enter on `/goal ` FILLS — buffer `/goal --clear`, `goal='land the OAuth refresh fix'`, no receipt. Before the fix the same keystroke cleared the goal (design D1 / UX U1 / code MAJOR-1). |
| `after/goal-run.svg` | `shot_slash_flag_flow.py DIR goal-run` | The SECOND Enter runs it: `goal=''` and the receipt reads `goal cleared: land the OAuth refresh fix` — the echo the design round asked for (D4 / U3), visible in the frame. |
| `after/picker-40x30.svg` | `shot_slash_flag_flow.py DIR narrow` | The row at a 40-column terminal (38 painted): `❯  --clear   Clear the standing…`. Before, the shared collapse rule dropped the label and the row was bare `❯  --clear` (UX U5). |
| `after/help-80x50.svg` | `shot_slash_help_row.py` | `/loop`'s new description on ONE painted line at 80 columns. |
| `before/help-80x50.svg` | the same script, from `cef6a4c4e` | `--stop or` wrapped with `--clear` orphaned in the command column (D2's phantom command) — the row this edit fixes. |
| `before/picker-40x30.svg` | `shot_slash_flag_flow.py DIR narrow`, from `cef6a4c4e` | the bare row at 38 painted columns (label dropped, U5's defect). |

There is deliberately no `before` frame for the fill/run pair: at `cef6a4c4e` the
one-Enter behaviour is the script's own assertion failing, and the base tree has
no row to fill at all. What the pair proves is the AFTER, plus the measured
observation the two `before` frames and the earlier rounds carry — `/goal ` +
Enter left `goal=''` with the notice `goal cleared`, and `/loop ` + Enter set
`_loop_cancelled=True` where the base refused with `a loop is already running`.

Measured rows (printed by the scripts, through the real compositor /
`render_rows`, never a width arithmetic of our own):

```
# /help, painted row for /loop, 80 columns
before: '  /loop               Loop toward a goal: /loop <goal>, /loop <n>, --stop or'  (76 cells)
        '  --clear'                                                                    (9 cells, orphan)
after:  '  /loop               Loop toward a goal: /loop <goal>, <n>; --stop cancels'   (75 cells, one line)

# the argument row, `render_rows(width)` for `/goal ` with a goal set
before: 100/80/60/44 -> '❯  --clear     Clear the standing goal' ; 40 and 38 -> '❯  --clear'
after:  100/80/60/44/40 -> '❯  --clear     Clear the standing goal' ; 38 -> '❯  --clear     Clear the standing g…'
```

`after/help-80x50.svg` is the only frame captured without a settled pair: the
`/help` table is static text with no arrival animation, unlike the picker row,
whose first painted frame is checked against its settled duplicate by the
sibling script (`cmp` on each pair reports no difference).

The PNGs of the earlier cases are separate rasterizations (`rsvg-convert X.svg
-o X.png`); the two scripts here write SVG plus `.geometry.json` only.

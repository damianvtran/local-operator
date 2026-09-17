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

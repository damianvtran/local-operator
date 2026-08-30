# The sibling full-page modes opened from the boot screen — rendered evidence

`/settings` was not alone in colliding with the boot layout. The **subagent
view** and the **org chart** mount through the identical three lines (hide the
transcript, mount before `#input-dock`, add the mode's class), so both shared
the same defect when opened from the splash: the boot layout stayed applied
underneath them and the page had to share the screen with a docked,
width-clamped, centred input card belonging to another layout.

These frames are the condition for fixing that in
`_sync_boot_layout_class` — generalising its condition to every full-page mode
rather than merely correcting its docstring to match a narrower implementation
(review round 1, F2). The rule this repo follows is that a fix extended to a
surface ships with that surface's own frames rather than riding along
unphotographed.

Capture either side:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/sibling_shot.py OUT.svg 100x30 subagent
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/sibling_shot.py OUT.svg 140x40 org
```

`before-*` frames are captured at `6ca77b34`, the head that fixed `/settings`
alone; `after-*` frames are this branch.

| file | what it shows |
| --- | --- |
| `before-subagent-100x30.svg` | the child page under a clamped 73-cell card that stops mid-screen, with 13 rows for the page and three rows of composition reserve in the dock's padding |
| `after-subagent-100x30.svg` | the dock spans the full 96 cells as a plain bar, the reserve is gone, and the page gets 17 rows — what the same page gets over a conversation |
| `before-subagent-140x40.svg` | the same collision at a large terminal: 24 rows, card clamped to 94 cells |
| `after-subagent-140x40.svg` | 27 rows, dock spanning 136 cells |
| `before-org-100x30.svg` | the chart under the inset, centred card — the shape the original report photographed, card at column 12 |
| `after-org-100x30.svg` | the card un-clamped to column 1 and full width; the chart itself is unchanged, which is the point |
| `before-org-140x40.svg` | 26 of 38 rows, card clamped to 94 cells at column 22, lifted off the bottom by four reserved rows |
| `after-org-140x40.svg` | 31 rows, dock flush and full width |

## The numbers behind the frames

Both dimensions, because the collision is not the same shape at every size —
at 100x30 the org chart loses no rows at all and the entire defect is the
card's clamp and offset.

```
                          boot   boot-card  shell.w  shell.x  dock.outer  view.h
before  subagent 100x30   True   True        73        1       12          13
after   subagent 100x30   False  False       96        1        9          17
before  subagent 140x40   True   True        94        1       11          24
after   subagent 140x40   False  False      136        1        9          27
before  org      100x30   True   True        73       12        5          21
after   org      100x30   False  False       96        1        5          21
before  org      140x40   True   True        94       22        9          26
after   org      140x40   False  False      136        1        5          31
```

`screen.virtual_size.height <= screen.size.height` on every frame, so nothing
overflows and the two-cell scrollbar tax never appears.

The geometry is also pinned by tests rather than left to the frames:
`test_the_page_takes_the_whole_view_when_opened_from_the_splash` in
`tests/unit/tui/test_subagent_view.py` and
`test_the_chart_takes_the_whole_view_when_opened_from_the_splash` in
`tests/unit/tui/test_team_chart.py` both compare the splash route against the
conversation route in **both** dimensions, at 100x30 and 140x40.

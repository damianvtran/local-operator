# `/settings` page — rendered evidence

Frames captured with `scripts/settings_shot.py`, which drives the real
`OperatorApp` (the only host that loads `local_operator.tcss`) against a
scratch config dir seeded with a handful of non-default values, so the
changed-vs-default styling is visible rather than a page of uniform defaults.

Reproduce any of these:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/settings_shot.py OUT.svg 100x30 overview
```

| file | what it shows |
| --- | --- |
| `overview-100x30.svg` | the page as it opens, two columns |
| `overview-140x40.svg` | the same at a large terminal |
| `overview-80x24.svg` | narrow: the read-only pane sheds and `←→ panes` leaves the footer with it |
| `enum-100x30.svg` | an enum row expanded, current member marked `●` |
| `enum-80x24.svg` | the same expansion at a split-pane width |
| `error-100x30.svg` | a rejected value with the editor still open and the reason inline |
| `error-80x24.svg` | the same rejection at a split-pane width |
| `cascade-100x30.svg` | the failover cascade, one chain open with its ordered hops |
| `cascade-80x24.svg` | the same cascade at a split-pane width |
| `confirm-100x30.svg` | `d` on a chain row asking before it deletes the whole chain |
| `confirm-80x24.svg` | the same ask at 80 columns on a short chain name |
| `confirm-long-100x30.svg` | the ask on a 26-character chain name, which fits in full because it is budgeted against the detail ROW (D12) |
| `confirm-long-140x40.svg` | the same at a large terminal, where the long key rung fits too (D12) |
| `confirm-long-80x24.svg` | the same at 80 columns, where the name genuinely sheds so `esc cancels` survives (D8/D13) |
| `teams-100x30.svg` | the read-only teams pane |
| `agents-100x30.svg` | the read-only agents pane |
| `agents-100x28.svg` | the pane one row below the band where the view's height steps down (D6) |
| `agents-100x26.svg` | providers folding to `… N more` while the signed-in one keeps its row (D11) |
| `agents-100x24.svg` | the roster shed to the caption, provider count still stated (D6/D11) |
| `agents-100x22.svg` | the section down to a header plus a count (D11) |
| `agents-100x20.svg` | the tightest pane, `providers  … 3 more` on one line (D11) |

The SHORT `agents` frames (`100x28` down to `100x20`) exist for the same reason,
from the other axis. The pane's height derivation tracks the real pane exactly
down to 29 terminal rows and then the view's height steps down in one jump, and
the round-1 evidence set stopped at 30 — so the pane painted eight lines into a
seven-row pane with `read-only` the line that fell off, one size band below
where the fix had been looked at (design round 2, D6). Height is a dimension
this page fails at in bands, so it is captured in bands.

The three `80x24` EXPANDED states are captured because the narrow width is
where the expansions are most likely to break and the set previously carried
only `overview` there. Design review round 1 had to regenerate them to reach a
finding (D3, the scope tags colliding with the gutter), which is the signal
that they belong in the committed set rather than in a reviewer's scratch
directory.

Each capture prints its own geometry (`screen.virtual_size` vs `screen.size`,
`show_vertical_scrollbar`, the body and pane sizes). On this app a scrollable
SCREEN is always a bug — the body scrolls and the dock is docked — and
`test_the_page_never_makes_the_screen_scrollable` asserts it at three sizes.

Three defects were found by looking at these frames rather than by a test:
rows wrapping by the scrollbar's two cells (widths were being read off child
widgets before layout had sized them, so every child size was still 0), and an
enum or chain expanded at the viewport's bottom edge showing an open `▾`
marker with its group entirely below the fold.

The band was extended down to 20 rows in round 3. The pane's shedding loop
deleted from index 1, which is the first PROVIDER row rather than the separator
its comment named, so between 20 and 26 rows it ate signed-in providers and at
20-24 painted a bold `providers` header with nothing under it — a frame that
reads as "none configured" while three providers are signed in, and the exact
opposite of the honest empty state the page paints deliberately (design round 3,
D11). The committed set stopped at 24, which is why it went unphotographed. The
rule the frames now show is that a provider which exists is always represented:
as its own row, or folded into a `… N more` count, never silently.

The `confirm-long-*` frames use a 26-character chain name for the same reason.
`default` is 7 characters and fits at every width, so the original `confirm`
frames demonstrated none of the ask's width behaviour — they hid both the
over-clipping at 100 and 140 columns (D12) and the shed the 80-column frame was
captioned for (D13).

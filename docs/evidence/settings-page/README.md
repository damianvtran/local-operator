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
| `confirm-80x24.svg` | the same ask at 80 columns, where the chain name sheds so `esc cancels` survives (D8) |
| `teams-100x30.svg` | the read-only teams pane |
| `agents-100x30.svg` | the read-only agents pane |
| `agents-100x28.svg` | the pane one row below the band where the view's height steps down (D6) |
| `agents-100x24.svg` | the shortest terminal the page is captured at, where the roster sheds to the caption (D6) |

The two SHORT `agents` frames (`100x28`, `100x24`) exist for the same reason,
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

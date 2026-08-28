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
| `error-100x30.svg` | a rejected value with the editor still open and the reason inline |
| `cascade-100x30.svg` | the failover cascade, one chain open with its ordered hops |
| `teams-100x30.svg` | the read-only teams pane |
| `agents-100x30.svg` | the read-only agents pane |

Each capture prints its own geometry (`screen.virtual_size` vs `screen.size`,
`show_vertical_scrollbar`, the body and pane sizes). On this app a scrollable
SCREEN is always a bug — the body scrolls and the dock is docked — and
`test_the_page_never_makes_the_screen_scrollable` asserts it at three sizes.

Three defects were found by looking at these frames rather than by a test:
rows wrapping by the scrollbar's two cells (widths were being read off child
widgets before layout had sized them, so every child size was still 0), and an
enum or chain expanded at the viewport's bottom edge showing an open `▾`
marker with its group entirely below the fold.

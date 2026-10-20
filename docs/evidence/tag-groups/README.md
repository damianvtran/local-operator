# TUI `/tags` frames — tag-grouping slice

Rendered from the **real styled `OperatorApp`** (production `local_operator.tcss`),
not a bare test host and not Storybook. Source: `b820be61c2`, the implementation
commit on `feat/tui-tag-command`.

## How these were captured

The rig is a scratch script, not a repo script: it seeds `SessionSidebar` with six
`SessionRow`s (one pinned, two active, three previous) and calls the repo's own
capture helper:

```sh
env -u NO_COLOR TERM=xterm-256color PYTHONPATH=. \
  /Users/damian/local-operator/.venv/bin/python scripts/run_bounded.py --timeout 240 -- \
  /Users/damian/local-operator/.venv/bin/python <scratch>/tag_groups_capture.py <out>
```

The helper is `scripts.visual_capture.save_capture`, which writes the SVG and a
`.geometry.json` beside it. Each SVG was rasterized with `rsvg-convert` and the
PNG was inspected. The spinner was pinned (`_timer` stopped, `_frame = 2`) so the
off/on pair differs only by grouping, and the app ran in an isolated HOME/config
root; no real session store was read or written.

## Frames

| File | State | Screen | Sidebar content | Scrollbars |
|---|---|---|---|---|
| `tags-off-wide-dark.png` | `/tags off` baseline | 118×30 | 34×30 | none |
| `tags-on-wide-dark.png` | `/tags on`, same fixture | 118×30 | 34×30 | none |
| `tags-on-narrow-dark.png` | `/tags on`, 60×20, scrolled to the end | 58×18 | 29×12 | none |
| `tags-on-wide-light.png` | `/tags on`, light palette | 118×30 | 34×30 | none |
| `tags-off-wide-dark-again.png` | `/tags on` then `/tags off` | 118×30 | 34×30 | none |

Observed in the wide dark frame (the pair is the acceptance):

- Pinned, Active and Previous remain the outer sections in their existing order.
- Each section is subdivided by **primary tag in registry order**, then
  **Untagged**; the empty `Gamma` category paints no heading.
- Every row is painted once, under exactly one bucket. (The secondary-tag case
  is pinned by `test_secondary_tags_never_duplicate_and_primary_wins`, not by
  this fixture.)
- `Untagged` is an explicit occupied bucket; it is not a catch-all for tagged
  rows.
- Rows keep the same left column and title budget in the off and on frames;
  only heading lines are added. The sidebar's content box is 34×30 in both.
- The narrow frame keeps the adaptive 29-cell content width, ellipsizes long
  titles and both tag headings, and shows the explicit `Untagged` bucket after
  scrolling. No screen or sidebar scrollbar appears.
- The light frame keeps the heading ink legible against the light surface
  (same `muted` semantic role as the existing section headings).

`tags-off-wide-dark.svg` and `tags-off-wide-dark-again.svg` are byte-identical
(`699f2132357b2a37346fd0edc191b7f9e35a8ed67831ea80e6c3f45a64ad8763`),
which is the "off restores the existing rendering, order and title budget"
claim at the paint level.

## What these do not prove

- They are a synthetic fixture, not the operator's live catalog.
- They do not exercise the real `/tag` picker interaction; that path is covered
  by `tests/unit/tui/test_tag_command.py` (typed exact name + Enter, twice for
  toggle, then a novel name for create-and-apply).
- They do not stand in for an independent QA run, and they are not desktop/UI
  evidence.

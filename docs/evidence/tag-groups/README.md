# TUI `/tag` and `/tags` frames

Rendered from the **real styled `OperatorApp`** (production `local_operator.tcss`),
not a bare test host and not Storybook. Source: `956681f13e`; the source files
were unchanged between the implementation commit `b820be61c2` and these captures
(only tests and the receipt's canonical spelling changed).

Every frame ships with its **source SVG and `.geometry.json`** beside the PNG.
The geometry files carry the screen, widget tree, content boxes and scrollbar
state; the PNGs are `rsvg-convert` rasterizations of the committed SVGs.

## How these were captured

Both rigs are scratch scripts, not repo scripts. They call the repo's own
`scripts.visual_capture.isolate_capture` (isolated HOME/config root, no real
store) and `scripts.visual_capture.save_capture`, so the real app and CSS are
what produced the frames.

```sh
# sidebar grouping: six SessionRows, spinner pinned
env -u NO_COLOR TERM=xterm-256color PYTHONPATH=. \
  /Users/damian/local-operator/.venv/bin/python scripts/run_bounded.py --timeout 240 -- \
  /Users/damian/local-operator/.venv/bin/python <scratch>/tag_groups_capture.py <out>

# /tag surface: real store seeded with four categories, two applied
env -u NO_COLOR TERM=xterm-256color PYTHONPATH=. \
  /Users/damian/local-operator/.venv/bin/python scripts/run_bounded.py --timeout 300 -- \
  /Users/damian/local-operator/.venv/bin/python <scratch>/tag_surface_capture.py <out>
```

## Sidebar grouping (`/tags`)

Fixture: six `SessionRow`s (one pinned, one active, four previous), registry
`Alpha`, `Beta`, `Gamma` (Gamma intentionally empty); Alpha/Beta primaries.

| File | State | Screen | Sidebar content | Scrollbars |
|---|---|---|---|---|
| `tags-off-wide-dark.{png,svg,geometry.json}` | `/tags off` baseline | 118×30 | 34×30 | none |
| `tags-on-wide-dark.{png,svg,geometry.json}` | `/tags on`, same fixture | 118×30 | 34×30 | none |
| `tags-on-narrow-dark.{png,svg,geometry.json}` | `/tags on`, 60×20, scrolled to the end | 58×18 | 29×12 | none |
| `tags-on-wide-light.{png,svg,geometry.json}` | `/tags on`, light palette | 118×30 | 34×30 | none |
| `tags-off-wide-dark-again.{png,svg,geometry.json}` | `/tags on` then `/tags off` | 118×30 | 34×30 | none |

Observed:

- Pinned, Active and Previous remain the outer sections in their existing order.
- Each section is subdivided by **primary tag in registry order**, then
  **Untagged**; the empty `Gamma` category paints no heading.
- Every row is painted once, under exactly one bucket. (The secondary-tag case
  is pinned by `test_secondary_tags_never_duplicate_and_primary_wins`.)
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

## `/tag` surface

Fixture: categories `Review`, `Ops`, `Deep Work`, `Deep Dive`; this conversation
is assigned `Deep Work` as **primary** and `Review` as secondary.

| File | State | What the frame shows |
|---|---|---|
| `tag-surface-no-arg.{png,svg,geometry.json}` | bare `/tag ` | Header “Tags for this conversation · 2/8 applied”; `[x] Review` `applied`; `[x] Deep Work` `Primary`; `[ ] Ops`, `[ ] Deep Dive` `not applied` |
| `tag-surface-partial.{png,svg,geometry.json}` | `/tag dee` | Filtered to `Create and apply “dee”` plus `[x] Deep Work Primary` and `[ ] Deep Dive not applied` |
| `tag-surface-create.{png,svg,geometry.json}` | `/tag New Label` | Exactly one create action: `Create and apply “New Label”` · `Creates a shared tag and applies it here.` |
| `tag-surface-created.{png,svg,geometry.json}` | after Enter | Receipt `Created and applied “New Label”.`, no second confirmation |
| `tag-surface-after-create.{png,svg,geometry.json}` | reopened `/tag ` | Count moved to `3/8`; `[x] New Label applied`; `Deep Work` remains `Primary` |

## Keyboard contract (real app)

`keyboard-contract.txt` is the captured stdout of a scratch probe driving the
real `OperatorApp`; the same assertions are pinned by
`tests/unit/tui/test_tag_command.py::test_tag_keyboard_arrows_enter_escape_and_inline_draft`.

```
open: True text: '/tag '
rows: ['Review', 'Ops']
arrow: 'Review' -> 'Ops'
store after arrow: {'tags': [], 'primary': None, 'revision': 0}
escape open: False text: '/tag '
store after escape: {'tags': [], 'primary': None, 'revision': 0}
enter assignment: {'tags': ['03461bd8a5'], 'primary': '03461bd8a5', 'revision': 1}
inline after enter text: 'keep this draft /tag ' open: True
inline after escape text: 'keep this draft /tag ' open: False
```

Arrows and Escape perform **zero writes** (revision stays 0); Enter commits
exactly one assignment; an inline `/tag` at the end of a draft accepts the
command word and opens the list without consuming the prose.

## What these do not prove

- They are synthetic fixtures, not the operator's live catalog.
- They do not stand in for an independent QA run, and they are not desktop/UI
  evidence.
- The `/tag` frames are the dark palette only; the sidebar pair covers light.

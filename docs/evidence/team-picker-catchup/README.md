# `/team` picker catch-up and highlight evidence

Captured with the real `OperatorApp` at 100×30 via Textual `run_test`, using
`env -u NO_COLOR TERM=xterm-256color`. `before-*.svg` comes from an unmodified
throwaway worktree at `origin/main` (`85b0737a`); `after-*.svg` comes from this
branch. PNGs are Quick Look renders that were viewed, and the montages stack the
five states for comparison.

## Frames

1. `catchup-empty`: `/team lop` while session construction is still blocked.
2. `catchup-filled`: same unchanged buffer after the delayed session and `lopdev`
   registry arrive. Before stays empty; after shows the `lopdev` row.
3. `typed-before`: hand-typed `/team lopdev fix it` before session adoption.
4. `typed-after`: same unchanged buffer after adoption. Before keeps `lopdev` as
   prose; after paints the exact name green without another keystroke.
5. `inline-multiline`: inline `/team` completion reassembled to
   `/team lopdev review\nthis`; after preserves the green name snapshot while the
   message remains prose.

## Geometry

Every capture reported `screen=98×28`, `virtual=98×28`, and
`show_vertical_scrollbar=False`. The catch-up row adds picker content without
changing virtual geometry or introducing a scrollbar/reflow.

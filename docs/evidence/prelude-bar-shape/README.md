# Prelude v6 — horizontal bar shape, before and after

**Subject:** the Highlights graphic the operator reported on 2026-10-19: "doesn't look incredibly
slick and seems to not be aligned left with the constraint/message width, and there's a weird bar
on the right side."

**Component:** the exact stored attachment `a395a5c68b1c72d18274bc5ca3d77a55` from session
`13c37091ac12` — three prices (`Input $0.15`, `Output $0.60`, `Cached input (prompt-cache read)
$0.015`), a horizontal `LO.bar` with `unit: "USD / 1M tokens"`. It was rendered through
`assemble_stored_component`, so the numeric assembly path is the real one.

**Before:** `origin/main` at `4da76988b5` (the merged #2668 head; prelude v5).
**After:** the same assembler on this branch (prelude v6).

## How the frames were made

The assembled document was mounted in a **480 px** iframe — the UI's `HEIGHT_HINT_MAX` — inside a
column of the stated width, under a paragraph at the same width. The host page does **not** post a
theme: that reproduces the operator's surface, where the frame fell back to `data-mode=dark` and
the pre-v6 stylesheet left the *light-theme* ink on `#121212`. (A host that does push a theme
overrides the tokens inline.) After the iframe's `load`, the harness logs the real
`documentElement.scrollWidth/Height`, the SVG rect, every bar rect and every text rect; those
numbers are in the geometry block under each screenshot and repeated below.

Frames were captured with the harness's browser tool at 1024×576 from a 2560×1440 window; the page
is `docs/evidence/prelude-bar-shape/render.py`-generated HTML served from a localhost HTTP server.

## Geometry (browser-measured, not a model)

| column | frame client | frame scroll (before) | frame scroll (after) | SVG (before) | SVG (after) |
|---|---|---|---|---|---|
| 360 | 360×480 | 360×480 | 360×480 | 356×160 | 356×142 |
| 620 | 620×480 | 620×480 | 620×480 | 616×277 | 616×142 |
| 900 | 900×480 | 900×480 | 900×480 | 896×403 | 896×145 |
| 1200 | 1200×480 | **1185×547** | **1200×480** | 1181×531 | 1196×193 |

At 1200 the pre-fix frame paints **both** scrollbars — `scrollWidth` is 15 px short of the client
width because the vertical scrollbar occupies it, and `scrollHeight` is 67 px over the 480 clamp.
That is the "weird bar on the right side". After the fix the document is shorter than the frame at
every width, so there is no scrollbar and the width is the full 1200.

| column | bars start (before) | bars start (after) | category label (before) | category label (after) |
|---|---|---|---|---|
| 360 | 122 | 152 | `Cached input (pr…` clipped at 114 | `Cached input (prompt-…` at 4..144 (tooltip has full text) |
| 620 | 122 | 209 | `Cached input (pr…` at 10..114 | full at 4..201 |
| 900 | 124 | 212 | `Cached input (pr…` at 10..116 | full at 4..204 |
| 1200 | 163 | 283 | `Cached input (pr…` at 13..152 | full at 5..272 |

The measured gutter is the longest category label plus the 8 px inset the label is anchored at;
the label column therefore starts at the same left edge as the paragraph (4-5 px inside the
iframe, which is its 2 px body padding plus the SVG's 2 px), not at a flat 120 px that the text
overflows. At 360 the clamp cannot hold the full 29-character label; it is ellipsised in the same
`_fit` path and the `<title>` keeps the full text.

## What each frame shows

* `before-360.png` — width-derived height; the fixed 120 px gutter truncates the long label; the
  four tick labels collide (`.2` and the unit tick merge); values are dark-on-dark.
* `after-360.png` — content-sized height (142), full label up to the 360 clamp (ellipsised), two
  ticks (`0`, `0.5 USD / 1M tokens`) that do not collide, values legible outside their bars.
* `before-620.png` — 277 px tall for three bars; labels clipped; values nearly invisible.
* `after-620.png` — 142 px tall; full label; bars start at 209 (the label column, not 12% of the
  width); values legible.
* `before-900.png` — 403 px tall, 86 px bars; labels clipped; values nearly invisible.
* `after-900.png` — 145 px tall, 24 px bars; full label; four clean ticks; values legible.
* `before-1200.png` — the reported scrollbar: `frameScroll 1185×547`, 531 px SVG, 113 px bars.
* `after-1200.png` — `frameScroll 1200×480`, 193 px SVG, no scrollbar.

**Not claimed:** the value label was already *outside* the bar in v5 (`max(a,z)+4`); the dark
pixels in the operator's screenshot were the fallback ink (`--lo-ink:#1f1f1f`,
`--lo-ink-muted:#4e4e4e`) on `#121212`, not a label on a series fill. v6 gives the dark fallback
its own light ink pair. The host-pushed theme path is unchanged.

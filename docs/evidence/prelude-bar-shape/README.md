# Prelude v7 — horizontal bar shape, review round 1 evidence

**Subject:** the operator's 2026-10-19 report — the three-price Highlights chart "doesn't look
incredibly slick", did not align with the message column, and had "a weird bar on the right side".

**Component:** the stored attachment `a395a5c68b1c72d18274bc5ca3d77a55` from session
`13c37091ac12` (Input 0.15, Output 0.6, Cached input 0.015 USD / 1M tokens), assembled with
`assemble_stored_component` exactly as the host does.

**Base:** `origin/main` at `4da76988b5` (prelude v5) for the three-width before set.
**Round-1 head:** `ce84e17e2d` (prelude v6) for the many-bar before set — the case the first
round did not cover.
**After:** prelude v7 (`b227c3ed80`) for the shape set, v8 (`fbd90306e0`) for the vertical scroll-lock set, and v9 (`f52360cb25`) for the tall-content scroller.

This directory lives on an evidence-only branch so the PR branch carries no evidence files.
It is not merged.

## Method

`render.py` assembles the stored component (or a synthetic n-bar chart with `--bars`) into the
real document, then wraps it in a host iframe fixed at **480 px** (the UI's `HEIGHT_HINT_MAX`)
with a 1 px border. The page prints the actual frame and SVG boxes as `EVIDENCE {…}`; the
screenshots are the browser's own frames at dpr 2. Charts are dark-fallback renderings
(`data-mode=dark`, no host push), the stricter ground for ink legibility.

Run: `.venv/bin/python render.py --out DIR --label after --widths 360,620,900,1200`
and `--bars 9|40 --widths 1200,1440` for the cap cases. Serve over localhost and screenshot.

## Final geometry (rendered CSS px)

| case | frame client | frame scroll | SVG | verdict |
|---|---|---|---|---|
| 3 bars @360 | 360×480 | 360×480 | 356×142 | fits |
| 3 bars @620 | 620×480 | 620×480 | 616×142 | fits |
| 3 bars @900 | 900×480 | 900×480 | 880×142 | fits, 1:1 |
| 3 bars @1200 | 1200×480 | 1200×480 | 880×142 | fits, 1:1 (v6: 1196×193) |
| 9 bars @1200 | 1200×480 | 1200×480 | 880×346 | fits (v6: scroll 1185×480 + 15 px bar) |
| 9 bars @1440 | 1440×480 | 1440×480 | 880×346 | fits (v6: scroll 1425×574) |
| 40 bars @1200 | 1200×480 | 1200×480 | 880×360 | fits (v6: 1185×505) |
| 40 bars @1440 | 1440×480 | 1440×480 | 880×360 | fits (v6: scroll 1425×597) |

The 880 px cap is the viewBox, now also the element's width/height attributes: above 880 CSS
only shrinks, never magnifies, so the 360 px viewBox cap stays 360 rendered px and 12 px type
stays 12 px.

## Frames

Three-width set (v5 → v7): `before-360.png`, `after-360.png`; `before-620.png`, `after-620.png`;
`before-900.png`, `after-900.png`; `before-1200.png`, `after-1200.png`.

Many-bar cap set (v6 → v7): `many9-before-1200.png`, `many9-after-1200.png`;
`many9-before-1440.png`, `many9-after-1440.png`; `many40-before-1200.png`,
`many40-after-1200.png`; `many40-before-1440.png`, `many40-after-1440.png`.

What they show: the v5/v6 before frames clip the long category at the left, stretch the chart
and its type with the column, paint a scrollbar at the right edge (and sometimes the bottom),
and leave the grid louder than the bars. The v7 after frames keep one compact bar band, fit the
label column to the longest label, end the two-tick axis on 0.6, put every value outside its
bar, and suppress value/last-category labels only when a 14 px label cannot fit the band.



## v9 tall content — 30-row table

`html{overflow:hidden}` alone clipped a 30-row table (876 px document) at the 480 px host
clamp with no affordance. v9 makes the body the scroller (`overflow:hidden auto;max-height:100vh`)
and reports `body.scrollHeight`; normal content never scrolls, oversized content scrolls to its
last row. The chart width is the viewport minus the body's 4 px padding, so a body scrollbar
cannot narrow a chart and recreate the 379/373 lock.

| case | v8 before | v9 after |
|---|---|---|
| 30 rows @810, seed 320 | body `810×876`, body scrollbar 0, `bodyTop 0`, row 30 bottom 868, not visible | body `795×480`, body scrollbar 15, `bodyTop 396.5`, row 30 bottom 471, visible |
| 30 rows @620 | — | body `605×480`, scrollbar 15, row 30 visible |
| 30 rows @1200 | — | body `1185×480`, scrollbar 15, row 30 visible |
| 4-bar @810, seed 373 | `810×373`, 15 px strip | `810×379`, scrollbar 0, reports `[379,379,379]` |
| 4-bar @620 / @1200 | — | `620×293` / `1200×412`, scrollbar 0 |

Frames: `table-before-810.png` (clipped at row 16, no scrollbar), `table-after-810.png`
(scrolled to row 30, body scrollbar), `table-after-620.png`, `table-after-1200.png`;
`vertical-after-810.png` (no strip). `render_table.py` builds the clamped host and scrolls
the body to the end; `render_vertical.py` logs the resize reports.

## v8 scroll-lock — vertical four-bar chart


The operator's "weird bar on the right" was a VERTICAL four-bar chart at an 810 px column.
The host seeds `height_hint` 320, but the prelude's first layout at 810 is 363 px; with the
16 px body padding the document is 379 px. A 373 px host box therefore added a 15 px
scrollbar, the chart narrowed and shrank 7 px, and `size()` reported 373 — a settled feedback
lock. `html{overflow:hidden}` removes the scrollbar; `size()` reports `body.offsetHeight`.

| case | before (v7) | after (v8) |
|---|---|---|
| 810, seed 373 | `810×373`, scrollbar 15 px, SVG `791×356` | `810×379`, scrollbar 0, SVG `806×363` |
| 620, seed 320 | — | `620×293`, scrollbar 0, SVG `616×277` |
| 1200, seed 320 | — | `1200×412`, scrollbar 0, SVG `880×396` |

Frames: `vertical-before-810.png` (strip at the right edge), `vertical-after-620.png`,
`vertical-after-810.png`, `vertical-after-1200.png` (no strip). `render_vertical.py` builds a
host that resizes the iframe from the prelude's `resize` post, starting at the seed height.

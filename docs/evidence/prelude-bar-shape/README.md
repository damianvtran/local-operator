# Prelude v7 — horizontal bar shape, review round 1 evidence

**Subject:** the operator's 2026-10-19 report — the three-price Highlights chart "doesn't look
incredibly slick", did not align with the message column, and had "a weird bar on the right side".

**Component:** the stored attachment `a395a5c68b1c72d18274bc5ca3d77a55` from session
`13c37091ac12` (Input 0.15, Output 0.6, Cached input 0.015 USD / 1M tokens), assembled with
`assemble_stored_component` exactly as the host does.

**Base:** `origin/main` at `4da76988b5` (prelude v5) for the three-width before set.
**Round-1 head:** `ce84e17e2d` (prelude v6) for the many-bar before set — the case the first
round did not cover.
**After:** this remediation, prelude v7 (`b227c3ed80`).

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

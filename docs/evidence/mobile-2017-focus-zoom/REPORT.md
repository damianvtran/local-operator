# #2017 reproduction + measured fix — numbers

All values measured on the REAL built SPA (served by the fixture daemon) out
of headless Chrome with CDP device metrics, at 390x844 and 360x780, before and
after the fix. Raw data: `before-focus-zoom-report.json`,
`after-focus-zoom-report.json`, `*-focus-behavior-report.json`.

## Wide-view preconditions (both runs)

| reading | before | after |
|---|---|---|
| meta after footer toggle | `width=512, viewport-fit=cover` | same (unchanged) |
| innerWidth / layout width | 512 | 512 |
| `visualViewport.scale` at load (390 / 360) | 0.7617 / 0.7031 | 0.7617 / 0.7031 (unchanged) |
| `--lo-fit-scale` on `<html>` | absent | `0.761` / `0.703` (floored: 390/512=0.76171875; 360/512=0.703125) |
| served `maximum-scale` / `user-scalable` | none | none (unchanged) |

## Field fonts (computed CSS px), before → after

| field | selector (anchor in `web/src`) | default off | wide 390 | wide 360 | expected iOS zoom before → after (wide 390) |
|---|---|---|---|---|---|
| Composer | `textarea[placeholder='Message…']` (composer.tsx:1357) | 16 → 16 | 16 → **21.025** | 16 → **22.7596** | 1.313 → 0.999 |
| Pending secret | `[data-testid="pending-card"] input` (pending-card.tsx:491) | 14 → 16 | 14 → 21.025 | 14 → 22.7596 | 1.500 → 0.999 |
| Model search | `input[placeholder='filter models']` (model-sheet.tsx:94) | 14 → 16 | 14 → 21.025 | 14 → 22.7596 | 1.500 → 0.999 |
| Session-list search | `input[placeholder='Search conversations…']` (session-list.tsx:1190) | 14 → 16 | 14 → 21.025 | 14 → 22.7596 | 1.500 → 0.999 |
| Past search | `input[placeholder='search names and conversations…']` (past-sessions.tsx:85) | 14 → 16 | 14 → 21.025 | 14 → 22.7596 | 1.500 → 0.999 |
| Pair code + name | pair.tsx:155, 168 | 14 → 16 | 14 → 21.025 | 14 → 22.7596 | 1.500 → 0.999 |
| Directory path | `input[placeholder='or type another path…']` (directory-sheet.tsx:205) | 12 → 16 | 12 → 21.025 | 12 → 22.7596 | 1.750 → 0.999 |
| Projects create fields | `#create-name/-description/-status/-tags` (projects-sheet.tsx:79-80) | 14 → 16 | 14 → 21.025 | 14 → 22.7596 | 1.500 → 0.999 |
| Slash filter | `input[placeholder='filter commands']` (composer.tsx:238) | 14 → 16 | 14 → 21.025 | 14 → 22.7596 | 1.500 → 0.999 |

- Zoom factors per the issue's WebKit rule: `target = clamp(16 / fontSize)`,
  zoom = target / current (current = 1 default; 0.7617 / 0.7031 wide).
  After: every field's target ≤ current (`16/21.025 = 0.76103 ≤ 0.7617`;
  `16/22.7596 = 0.70301 ≤ 0.703125`) — no zoom, with the three-decimal floor's
  margin.
- Physical size check: 21.025 x 0.76171875 = 16.01px; 22.7596 x 0.703125 =
  16.00px (≥ 16 physical px).
- Ask-card free-text input is not materialised in this fixture: its only
  free-text asks (`st-second`, `qa-declined`) sit on rows the app renders as
  terminal (`runtime_live=false, durable=false` — "this conversation no longer
  exists" branch, DOM dump in `before-st-second-card.png` + `probe_st_second.py`).
  Its class (`text-body`, ask-card.tsx:111) is covered by the same 16px floor
  and the wide rule as the measured siblings.

## Focus behaviour (why the zoom leg is not locally verifiable)

`probe_focus_zoom.py`, both runs: focusing composer (16px), pending secret
(14px), model search (14px), directory path (12px) leaves
`visualViewport.scale` unchanged in every case (1 → 1 default; 0.7617 → 0.7617
wide). Chromium does not implement iOS's `_zoomToFocusRect:`.

`xcrun simctl list runtimes` on this host: `unable to find utility "simctl"` —
CommandLineTools only, no Xcode, no simulator runtimes (recorded on PR #1913 —
the reason the original wide-view PR carried the same caveat). The device leg
is handed to QA.

## Served-asset scans (`served-assets-checks-*.txt`)

- Login page + SPA HTML: no `maximum-scale`, no `user-scalable`, no
  `text-size-adjust`.
- Served CSS: no `maximum-scale` / `user-scalable`; one
  `-webkit-text-size-adjust:100%` — the Tailwind v4 preflight normalizer
  (`node_modules/tailwindcss/preflight.css:31`), not an app directive.
- After: both new rules present in the served CSS
  (`font-size:max(16px,1em)` and
  `font-size:calc(16px / var(--lo-fit-scale))`).
- Served JS + app source: no `maximum-scale` / `user-scalable` /
  `text-size-adjust` anywhere.

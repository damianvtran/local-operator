# Mobile UX batch 1 — before/after numbers

Rig: the REAL built bundle over synthetic projections, isolated HOME, own port, headless Chrome, real touch events, 390×844 + 360×780 @ dpr2 (designer-audit fixture + capture, reused per the audit hand-off). Sweep fields below come from the same DOM sweep the audits used.

## D1 — list overflow (main scrollWidth / clientWidth; row line scrollWidth / clientWidth)

| width | before main | after main | before line | after line |
|---|---|---|---|---|
| 320 | 453/320 (overflow 133) | 320/320 | 441/296 | 296/296 |
| 360 | 453/360 (overflow 93) | 360/360 | 441/336 | 336/336 |
| 390 | 453/390 (overflow 63) | 390/390 | 441/366 | 366/366 |
| 430 | 453/430 (overflow 23) | 430/430 | 441/406 | 406/406 |

The long-model row's model span now truncates inside its line (before: scrollWidth 433 painted at 433px wide; after: clientWidth 358).

## D2 — sub-44px interactions per screen (same sweep, before → after)

| frame | before | after |
|---|---|---|
| `list-390x844` | 14 | 0 |
| `pin-midhold-390x844` | 14 | 0 |
| `slash-open-390x844` | 25 | 1 |
| `slash-sheet-390x844` | 53 | 1 |
| `theme-sheet-390x844` | 45 | 0 |
| `model-sheet-390x844` | 14 | 1 |
| `effort-sheet-390x844` | 9 | 1 |
| `chips-390x844` | 5 | 1 |
| `chips-desk-390x844` | 6 | 1 |
| `subagents-390x844` | 5 | 1 |
| `pair-390x844` | 1 | 0 |
| `past-390x844` | 5 | 0 |

The remaining `1` on session screens is the composer textarea (min-height state, grows with content) — deliberately unchanged (not in the audit's input enumeration; flagged for the design round).

## Changed controls (measured boxes, 390×844)

| control | before | after |
|---|---|---|
| session header back | 32×32 | 44×44 |
| session header ★ | 32×32 | 44×44 |
| session header approvals | 71×32 | 71×44 |
| composer model chip | 58×32 | 58×44 |
| composer effort chip | 29×32 | 44×44 |
| pair back | 32×32 | 44×44 |
| past resume | 71×36 | 71×44 |
| list search input | — | — |
| model-chip label, wrapping | 354×35 (2 lines) | 354×44 (1 line, ellipsis) |
| composer textarea (unchanged) | 184×24 | 184×24 |

## U1 — pin sheet across the finger's release

- before: dialogs after release = **0** (tail of the event log: pointerdown→BUTTON, pointerup→BUTTON, click→BUTTON)
- after: dialogs after release = **1** (tail of the event log: pointerdown→BUTTON, pointerup→BUTTON)
- after (action): tapped “Unpin from the top” → dialogs 0, desk ★ False (round-trip confirmed; re-pin restores it)

## U5/U8 — slash sheet state matrix

| step | before (dialog / focus / composer / filter) | after |
|---|---|---|
| slash-open | True / BUTTON|close sheet / `/de` / de | True / INPUT|filter commands / `/de` / de |
| slash-typeahead | True / BUTTON|close sheet / `/de` / de | True / INPUT|filter commands / `/de` / delete |
| slash-after-pick | True / BUTTON|close sheet / `/delete ` / delete | False / TEXTAREA|Message… / `/delete ` / None |
| slash-after-args | True / BUTTON|close sheet / `/delete ` / delete | False / TEXTAREA|Message… / `/delete x` / None |
| slash-escape-before | True / BUTTON|close sheet / `/del` / del | True / INPUT|filter commands / `/del` / del |
| slash-escape-after-type | True / BUTTON|close sheet / `/dele` / dele | False / TEXTAREA|Message… / `/dele` / None |
| slash-fresh | True / BUTTON|close sheet / `/` /  | True / INPUT|filter commands / `/` /  |

## D5/D6/D7

- D5 chips @390: model chip 354×35 → 354×44, label white-space nowrap, overflow hidden (before: plain wrap).
- D2 chips @390 (typical): model 58×32 → 58×44; effort 29×32 → 44×44.
- D6 subagents header @390: `▸subagents 1/5 running· 1 queued· 1 failed· answer first` → `▸subagents1/5 running· 1 queued· 1 failed· answer first`; label yields alone, `1/5 running` survives whole (whitespace-nowrap + shrink-0).
- D7: session-view header controls (back/★ 32×32, approvals 71×32) → 44×44 / 44×44 / 71×44 via one idiom; pair back 32×32 → 44×44.

## U10 — entry point

- footer controls before: ['new session', 'projects', 'choose theme']
- footer controls after: ['new session', 'past', 'projects', 'choose theme']
- tapping `past`: hash `#/past`, control 52×44

## Gates

- `pnpm build` ✅ (tsc -b + vite + check-bundle)
- `pnpm test` ✅ 38 files / 319 tests (8 new: U1 ×2, U5/U8 ×4, U10 ×1, D6 ×1)

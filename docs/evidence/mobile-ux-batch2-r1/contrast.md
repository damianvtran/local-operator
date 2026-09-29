# D1 — the tones are now asserted on `elevated`

`scripts/contrast-contract.mjs` asserted tones on canvas, surface and their own
wash only; the new pin-refusal and resume-refusal strips paint `text-danger` on
`bg-elevated`, so that pair was the one the gate could not see. The contract
now asserts all four tones on `elevated` (39 → 43 assertions per theme; 1209 →
**1333** assertions across 31 themes, `pnpm check-themes` green).

Measured (WCAG 2.x, the contract's own formula over `palette-source.mjs`):

| theme | danger | on elevated | on canvas | on surface | on dangerWash |
| --- | --- | --- | --- | --- | --- |
| monokai | `#FB6097` → `#FC81AC` | **3.76 → 4.63** | 5.11 → 6.28 | 4.65 → 5.71 | 4.64 → 5.70 |
| dracula | `#FF7171` → `#FF8383` | **4.11 → 4.62** | 5.32 → 5.99 | 4.76 → 5.35 | 4.62 → 5.19 |
| neon | `#FF00A0` → `#FF1AAA` | **4.43 → 4.60** | 5.35 → 5.56 | 5.00 → 5.19 | 4.96 → 5.15 |

The other 28 palettes already cleared 4.5 on the new pair and are unchanged
(worst case: catppuccinFrappe at 4.500 — unchanged, above the floor). Each
nudge is a lift along the theme's own danger hue ("smallest lift that clears"),
the same move the batch made for dracula's `elevated`; every other asserted
ground for these three improved at the same time.

Frames: `after-u15-cutoff-ended-390.png` / `after-d5-resume-refusal-390.png`
paint the pair (danger ink on the elevated strip) in the default dark theme;
the reviewer's monokai/dracula/neon frames live in the design round's
scratchpad alongside these numbers.

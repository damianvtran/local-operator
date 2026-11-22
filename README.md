# Evidence — relay jump-to-latest chip

Frames for `feat/relay-jump-to-latest`, captured on head `6b92e9f58` (the
visual code is unchanged from `3a68293f6`; only generated i18n data moved).

Rig: `scripts/mobile_overflow_capture.py`'s headless-Chrome driver, one
browser for the whole run, against an isolated loopback fixture serving the
real built bundle (`local_operator/mobile/web/dist`) with a 38-row
conversation and an SSE append endpoint. Never the operator's relay.

| file | state |
|---|---|
| `shots/before-arrival-<vp>.png` | reader scrolled up, no new content yet — no chip |
| `shots/after-chip-<vp>.png` | one row arrived over SSE while scrolled up — chip visible, scroll position unchanged |
| `shots/at-bottom-<vp>.png` | after pressing the chip — at the tail, chip gone |

`<vp>` is `390-light`, `390-dark` and `320-light` (390×844 and 320×640,
DPR 3, iPhone safe-area insets). `measurements.json` is the raw geometry:
`scrollTop` before/after the arrival (identical: no yank), the chip's rect
(right edge 378/308 = 12 px inset; bottom 8 px above the scroller), the
composer's top (the chip never covers it), and the short label at 320
(`Latest` with `aria-label="Jump to latest"`).

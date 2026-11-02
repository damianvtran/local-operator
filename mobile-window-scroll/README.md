# Mobile session view — the window is not a scroller (evidence)

Frames for PR `fix(mobile): stop the session screen's window from scrolling`.

## Rig

- `local_operator/mobile/web` built with `pnpm build` in a worktree, served by
  `scripts/mobile_overflow_fixture.py` (real bundle, synthetic session
  `operator-spend-long`, 60 transcript rows) on loopback.
- Headless Google Chrome over CDP through `scripts/mobile_overflow_capture.py`'s
  `Chrome`/`Page`: unique profile, `--remote-debugging-port=0`, `--headless=new`,
  viewport by `Emulation.setDeviceMetricsOverride` at DPR 2, one browser for the
  whole run, process group reaped by pid.
- Every scroll is a real `Input.dispatchTouchEvent` drag, never a
  `scrollTop` assignment.
- Identical rig for both sides; only `dist/` differs (rebuilt from the tree
  before and after the change).

## What Chrome cannot show

iOS Safari's elastic overscroll on a document with no scroll room, and its
separate visual viewport. Chrome has one viewport, so the two iOS-only
mismatches are injected into the real bundle instead of claimed as reproduced:

- `gap` — the document is taller than the viewport, as it is on iOS while the
  browser bars are showing (`html {height:100%}` = layout viewport,
  column = `100dvh` = smaller). 56 px at 390x844, 120 px at 360x780.
- `pan` — the column's own keyboard pin has written `top: 100px`, which is what
  `session-view.tsx` writes when iOS reports `visualViewport.offsetTop = 100`.

## Frames

| frame | case | drag starts on |
| --- | --- | --- |
| `gap-390x844-before.png` / `-after.png` | body 56 px taller than viewport | composer |
| `gap-360x780-before.png` / `-after.png` | body 120 px taller | composer |

In the BEFORE frames the header is gone and the bottom of the frame is blank
canvas: that blank is what the window "scrolled" to. In the AFTER frames the
header and composer are where they were before the drag.

No frame is published for the `pan` case: once the page has scrolled by the
pin's offset the column is back at y=0, so the symptomatic frame looks exactly
like a healthy screen. Its measurement is in the table below.

## Measured

`document.scrollingElement` after one real touch drag, DPR 2, Chrome headless:

| case | before: scrollTop, header y | after: scrollTop, header y |
| --- | --- | --- |
| 390x844 gap, drag on composer | 0 → **56**, header 0 → **-56** | 0 → 0, header 0 → 0 |
| 360x780 gap, drag on composer | 0 → **120**, header 0 → **-120** | 0 → 0, header 0 → 0 |
| 390x844 pan, drag on header | 0 → **100**, header 100 → **0** | 0 → 0, header 100 → 100 |
| 390x844 pan, document scrollHeight | **944** (client 844) | **844** |

Computed on the same runs:

| property | before | after |
| --- | --- | --- |
| `html` overflow / overscroll-behavior | `visible` / `auto` | `hidden` / `none` |
| `body` overflow / overscroll-behavior | `visible` / `auto` | `hidden` / `none` |
| `body` height / min-height | `844px` / `100%` | `844px` / none |
| transcript scroller overscroll-behavior-y | `auto` | `contain` |

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

## `#/past` — the one screen that still relied on the document scrolling

Same rig, `#/past` instead of a session: the real built bundle, headless Chrome
over CDP at DPR 2, one browser per run, viewport by
`Emulation.setDeviceMetricsOverride`, and every scroll a real
`Input.dispatchTouchEvent` drag. The 22 rows are served at the daemon's
`_search_sessions` seam — `#/past` calls `GET /api/sessions/search`, so the
route, the wire shape, the renderer and the CSS are all shipped code; only the
store's answer is synthetic, because the real store is empty under isolation.
22 is the count the finding and the QA round measured with.

| frame | case |
| --- | --- |
| `past-390x844-before-top.png` / `-bottom.png` | 22 rows, at 390x844 — initial, and after real drags |
| `past-390x844-after-top.png` / `-bottom.png` | the same, with the fix |
| `past-360x780-*.png` | the same pair at 360x780 |

`-top` and `-bottom` are the same image on the BEFORE side (byte-identical
sha256, `ae53c5fc…` at 360x780 and `ef3dc6e3…` at 390x844): the drags moved
nothing at all. In the BEFORE frames the list is cut at the fold with rows 15-22
painted below it, unreachable. In the AFTER frames the header and the search
field are where they were and row 22 with its `resume` button is fully in view.

| measurement | before, 390x844 | before, 360x780 | after, 390x844 | after, 360x780 |
| --- | --- | --- | --- | --- |
| root height / min-height / overflow | `1314px` / `100%` / `visible` | same | `844px` / `0px` / `hidden` | `780px` / `0px` / `hidden` |
| list clientHeight / scrollHeight | `1198` / `1198` (no scroller) | same | `728` / `1198` | `664` / `1198` |
| list `overflow-y` | `visible` | `visible` | `auto` | `auto` |
| list scrollTop, real drags | `0` → `0` | `0` → `0` | `0` → **`470`** (= max) | `0` → **`534`** (= max) |
| last row, y top-bottom | `1250`–`1302` | `1250`–`1302` | `780`–`832` | `716`–`768` |
| last row inside the viewport | **no** (458px past the fold) | **no** (522px) | **yes** | **yes** |
| `document.scrollingElement.scrollTop` | `0` → `0` | `0` → `0` | `0` → `0` | `0` → `0` |
| header y before/after drags | `0` / `0` | `0` / `0` | `0` / `0` | `0` / `0` |

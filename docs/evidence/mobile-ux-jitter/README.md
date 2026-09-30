# Session-list jitter — reproduction and fix evidence (project `lo-mobile-ux`, batch 3)

Rig: the REAL phone bundle (`local_operator/mobile/web/dist`) served by the real
mobile daemon over synthetic sessions (isolated HOME via `scripts.probe_isolation`,
`dial_registrants=False`, own loopback port, per-run password), driven by
headless Chrome over CDP with real touch events (`Input.dispatchTouchEvent`). An
in-page rAF sampler records every card's painted top, height and computed
transform — plus the element under a watched touch point — and a loopback
control plane reorders the list and drives SSE frame bursts on demand.

Every number below comes from these runs. Each JSON in `series/` is the raw
sample series (times in ms; per card `[top, height, transform]`; `frames` are
the client-observed SSE arrivals; tap runs also carry `at_release` — the element
under the finger when it lifted — and `press_ms`).

## Measured frame cadence

The daemon pushes a list frame per projection update. On this fixture, pumping
at 30 Hz for 60 s, the client observed **23.7 `sessions` frames/s** (1423
frames; p50 gap 33.1 ms, p95 125.2 ms, max 434.6 ms) → roughly **4-5 commits
inside one 180 ms settle window** during a stream. (`series/cadence.json`.)

## Before (`origin/main` @ `fc851a94e`) — the mirrored settle

`getBoundingClientRect().top` includes the settle's OWN in-flight transform, so a
commit inside the 180 ms window re-measures the mid-flight offset and writes it
back — `dy = prev − (target + t) = −t` — mirroring the card across its final
slot; every further commit flips the sign again.

- **single reorder** (control): one clean settle. (`before-single.json`)
- **burst** (one swap + commits at +33/+90/+150 ms): the swapped row jumps
  91.9 → 244.6 px (transform −50.9 → **+101.8 px**, i.e. ~102 px PAST its slot),
  then to −9.9 px (transform −152.7), 2 direction changes, 382 ms to rest —
  the operator's "rows go up and down". (`before-burst.json`)
- **pump** (30 Hz for 4 s after one swap): the mirrored writes AMPLIFY —
  |transform| grows per commit and saturates at the 2^25 px matrix clamp
  (±33,554,400 px), 39 direction changes over 4.16 s. The row is off-screen for
  the whole pump: `frames/before-2-pump-mid.png` shows the two swapped rows
  absent and the rows below shoved down. (`before-pump.json`)
- **tap during settle** (finger on row 2; the swap frame and two more commits
  land mid-touch; release ~150 ms into the settle): the row slides out from
  under the finger — `at_release.hit` is the OTHER row — and the tap opens
  **the wrong session**: `#/s/jit-foxtrot` instead of the pressed
  `#/s/jit-echo`. (`before-tap.json`)
- **empty-second-line corner**: a card whose second line is empty renders
  44 px (its `min-h-11` floor) vs 50.89 px with one; the fill shifts rows below
  by 6.9 px and the same mirrored settle fires on them. No real row in the
  fixture transitions empty→filled (live and durable rows carry
  `cwd`/`model_label`), which is why no line reservation was added — see the
  PR's "not addressed". (`before-corner.json`)

## After (`fix/mobile-ux-jitter`) — layout-space settle + frame hand-off

The settle now measures LAYOUT space (`offsetTop` chain — transform- and
scroll-immune; verified: a 173 px scroll moves a card's rect 397.2 → 224.2 while
its offset chain stays 397.0), compares layout targets (unchanged target ⇒ no
write — the core guard), and a settling card whose layout moves again continues
from its current paint. The rows also receive store frames at most once per
animation frame, and not at all while a pointer is down on the list.

- **single**: one clean settle, 0 direction changes, overshoot 0.3 px
  (sub-pixel rounding). (`after-single.json`)
- **burst**: 193.8 → 147.2 → 142.8, monotone, 0 direction changes, one settle;
  the three extra commits inside the window write nothing. (`after-burst.json`)
- **pump**: one settle at the start; then **113 further frames over ~5 s produce
  no motion at all** (every card tr = 0, tops constant to a tenth of a pixel).
  (`after-pump.json`)
- **tap**: the rows do not move during the touch (`at_release.hit` is still the
  pressed row) and the tap opens the right session, `#/s/jit-echo`; two
  consecutive runs agree. (`after-tap.json`)
- **corner**: the 6.9 px shift settles in one pass — no mirror. (`after-corner.json`)

## Frames

`frames/{before,after}-0-rest.png` — the control: identical before/after (same
fixture state, same order). `-2-pump-mid.png` — the sustained pump: before, the
two swapped rows are painted out of view and the rest are displaced; after, all
six rows sit in place. `-3-settled.png` — control after the pump: identical.
(Stills for the burst and tap properties are not useful: a single capture
cannot reliably catch a 380 ms oscillation — the series JSONs are the record.
The stills' run had the empty-second-line row filled; each before/after pair is
state-matched.)

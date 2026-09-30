# Session-list jitter — round-1 remediation evidence (project `lo-mobile-ux`, batch 3)

This round answers UX review round 1's **U32 (MAJOR)**: the frame hold deferred
moves a *new* frame would start, but a settle **already gliding** when a finger
landed was still a moving target under it. The fix extends the hold to motion:
a finger pauses the list, whatever started the move.

Same rig as round 0 (README one directory up): the REAL phone bundle served by
the real mobile daemon over synthetic sessions (isolated `HOME` via
`scripts.probe_isolation`, own loopback port, per-run password), driven by
headless Chrome over CDP with real touch events, an in-page rAF sampler
recording every card's painted top/height/computed transform per frame plus
**pointerdown/up/cancel timestamps and SSE frame arrivals on the page's own
clock**. Every verdict below is "did the tap open the row the finger was on
when it landed" (the sampler's `hit` under the watched point), which is
independent of the rig's dispatch latency.

## Two protocol notes, disclosed

- **The matrix runs and the stills run dial the settle transition** in the page
  (`--transition-duration-base`): 1500 ms for `series/*-midtouch-*`, 6000 ms
  (linear) for `series/*-frozen*`. Under fleet load a single CDP press costs
  hundreds of milliseconds, so a press cannot always land inside a 180 ms glide
  (and two CDP stills cannot both land inside one). The property is
  duration-independent — the same pointerdown → pin → release → resume path
  runs — and **both sides run the identical protocol**, so the before/after
  pair is one variable. UX round 1's own numbers, at the real 180 ms, are the
  independent reproduction (3 of 9 mid-settle presses opened the wrong session,
  the pressed row travelling 274.6–407.0 px under the press; 0 of 10 on a
  settled list).
- **A same-URL `Page.navigate` is a same-document navigation** (measured): a
  pin sheet from a previous attempt survives it and its scrim makes every hit
  test land on the container. Retry loops here use `Page.reload`.

## U32 before → after

| scenario | before (`93922cf4d`) | after (this head) |
| --- | --- | --- |
| `midtouch` 3 frame-synced presses (press~330–500 ms, no sheet) | **the row under the finger changed mid-press** (phase 0: finger on Charlie, the gliding Foxtrot crossed under it, the click opened **Foxtrot** — the wrong row; the other two phases landed outside the glide and passed) | **3/3 opened the row under the finger** (`hit@down == hit@up` in all three) |
| `frozen` — finger held >1 s into a glide | the row **kept gliding through the whole hold** (6.24 px drift over 1487 ms) and finished after the release | **frozen: 0.00 px across 27 samples / 1064 ms**, then one monotone resume glide after the cancel, completing at the slot (142.78 px, 0 direction changes) |

The `frames/` stills are the same story as pixels: `before-1-frozen-a` and
`before-2-frozen-b` catch the two crossing rows at **different** points (they
kept moving); `after-1-frozen-a` and `after-2-frozen-b` are the **identical
frozen pose** (the finger holds them mid-cross); `after-3-resume` (0.25 s into
the 6 s rig glide, ≈1 px of travel) shows the glide under way again — the
series carries its completion.

## Regression re-runs on the fixed head (same rig)

- `single`: one settle, 104 ms, **0 direction changes**, overshoot 0.0 px,
  max |tr| 51 px (one slot).
- `burst` (swap + extra frames in-window): **0 direction changes**, overshoot
  0.0 px, max |tr| 51 px.
- `pump` (30 Hz for ~5.3 s, 117 frames): one settle at the start, then
  **0.00 px steady-state motion**; max |tr| 51 px.
- `tap` (frame lands mid-touch): **2 frames in touch**, press 395 ms, no
  sheet, **opened the pressed row** (`#/s/jit-echo`).
- `corner` / `cadence`: unchanged paths (no new surface); round 0's figures
  still stand. `cadence.json` (round 0) measured 23.7 list-frames/s — the
  input rate this fixes against.

The `series/before-{single,burst,pump}.json` counterparts are round 0's own
captures (their scenario code is unchanged); `before-tap.json` from round 0
was taken with an earlier tap protocol and is left to the round-0 folder.

## Round-1 review items carried in this round

- **M4 (wording)**: round 0's pump note said "113 further frames"; the series
  carries 113 frames **total**, 104 of them after the settle. Corrected here.
- **M1/M2**: unit tests for the continuation branch and for the ≤1-apply-per-
  frame coalescing — both **verified to fail** against the respective
  regressions (continuation removed → `-40` written instead of `-70`; apply
  made immediate → the intermediate frame paints).
- **M3 (per-card layout flush)**: assessed, left as is — the effect runs only
  on commits where cards actually move (a sustained 30 Hz pump wrote nothing
  and moved nothing: 0.00 px over 117 frames), the pre-change loop had the same
  shape, and a two-pass collect-then-apply remains the documented improvement
  if device profiling ever shows a cost.
- **N1/N2**: the pin-hint comment now says what it reads (the painted rows'
  confirmed pins, ≤1 frame behind the store); `onListPointerDown` is a stable
  `useCallback`.
- **D1 (design)**: the second line now always carries at least one blank, so an
  empty second line cannot collapse a card 6.89 px below its neighbours —
  structural (a blank's line box follows the type scale), unit-tested.
- **U33/U34 (decision)**: the stale-for-the-hold window is **accepted and
  documented, with no cue** — applying anything mid-hold would move the very
  targets the hold exists to keep still (and a cue would advertise a state the
  app cannot act on while the finger is down); taps see the window <150 ms and
  a long-press ends under the sheet, so nothing wrong follows from it.

## Re-running

The rig lives in the round-0 scratch session (probe + fixture + analyzer);
`scripts/probe_isolation` first, `env -u XPC_FLAGS`, headless Chrome
`--use-mock-keychain`, never the operator's daemon/sessions.

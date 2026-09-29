# Mobile UX batch 2 — round-4 remediation evidence

Captured headless (Chrome over CDP, `--use-mock-keychain`, isolated HOME via
`scripts.probe_isolation`, own ephemeral port, real touch events only) against
the REAL built bundle of each side: `before-*` is the round-3 head `d103e788c`
(`dist/assets/index-Bb92uoPC.js`), `after-*` is this remediation's bundle
(`dist/assets/index-BE5l9Bjg.js`, the head's own build output). Same fixture as
round 3 plus two additions: a 140-row live session past the phone's 120-row
window (`r4-long`, U28's shape) and a fixture-side history page that can be
armed and delayed, so `loadOlder`'s prepend path can be driven at all (the
durable fold's row ids can never match a synthetic projection, and the fixture
has no history pages).

Raw probes: `measure-r4-before.json` / `measure-r4-after.json` (produced by
`run_r4.py`; frames beside them). Rows are tracked by id; "move" is the largest
displacement of any tracked row between the two samples.

## The finding this remediation fixes

The round-3 opt-out (`[overflow-anchor:none]`) left the one hand on `scrollTop`
answering only for the reserve's own edges. The round-4 reviews found the two
changes above the reader that hand was not answering for: the live window
**dropping its oldest row on each append at the cap** (U28 — native anchoring
had been covering that removal silently, so the opt-out regressed it to a
whole row per append) and the browser's own **clamp when the reserve clears at
the tail** (reviewer MAJOR 1 = U29: the clamp followed the shrink and the write
subtracted the same height again, leaving the newest row under the fold).

Mechanism now: the follow effect tracks EVERY rendered row and holds the row
the reader was nearest at the viewport's top edge across whatever moved above
them — the reserve's edges, window evictions, `show N more loaded`, a
prepended page — with the user's own scrolling divided out exactly. The
decision (and the two positions that are not "hold this row": the tail, where
the clamp has already followed, and the top with the reserve's edge, where the
slide is the reveal) is `lib/scroll-follow.ts`, pinned by
`lib/scroll-follow.test.ts`. The older-rows rAF restore is gone: prepends are
one case of the same mechanism now.

## Before → after, measured

**Evictions at the 120-row cap, mid-history, per live append** (abs px):

| width | before | after | scrollTop, after |
| --- | --- | --- | --- |
| 390 | 64 / 29 / 29 / 29 | **0 / 0 / 0 / 0** | 1706→1677→1648→1619 (one row per append) |
| 320 | 29 / 29 / 29 / 64 | **0 / 0 / 0 / 0** | 2003→1974→1945→1881 |

**An append landing mid-gesture** (40-step real drag, 6px/step, append at
step 20, samples at 19 and 40): finger movement 126px; measured row movement
155px before (29px = the eviction), **126px after (0px beyond the finger)**.

**Rung CLEAR at the tail, settled, NO gesture after the transition** (the
scene the round-3 claim missed):

| transition | before | after |
| --- | --- | --- |
| degraded clears | dist **34**, rows move 53.39 | dist **0**, rows move 0.39 |
| ended clears | dist **61**, rows move 60.0 | dist **0**, rows move 0.0 |

Both widths. The rung APPEARING at the tail (reader settled): before left the
ended rung at dist 34 / degraded at 1; after: 0 / 1 (the 1px is the
sub-pixel reading of a rung that is itself landing at the bottom).

**Regressions checked on the same scene set**

| scene | before | after |
| --- | --- | --- |
| mid-history rung appear/clear (U27) | 0.61 / 0.39 / 0.0 / 0.0 (390) | 0.39 / 0.39 / 0.0 / 0.0 |
| at-top reveal (scrollTop 0, first row clear) | 0 / 0 both widths | **0 / 0 both widths** |
| tail re-pin through appends | 0 / 0 | 0 / 0 |
| below-cap appends (short session) | 1 / 0 / 0 / 0 | **0 / 0 / 0 / 0** |
| older page lands mid-history (390) | held 0.0 (rAF restore) | held **0.0** (the one mechanism) |

320's mid-history degraded APPEAR read 1.61px in this capture (0.39 in the
round-3 runs); it is the harness's drag momentum, in the same class as the
0.61/0.39 scatter the earlier rounds recorded. The prepend leg at 320 did not
land inside the scene's window (`landed: false` in the probe) — a harness
timing limit, stated rather than papered over; the 390 leg landed with the
reader mid-history and held at 0.0.

## Frames

`frames/before-*` and `frames/after-*` pairs: `evict-base/-after` (390, 320),
`drag-append`, `tail-cleared` (390, 320), `top-ended` (390, 320),
`mid-ended` (390, 320), `prepend` (390, 320).

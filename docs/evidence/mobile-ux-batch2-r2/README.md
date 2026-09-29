# Mobile UX batch 2 — round-2 remediation evidence

Captured headless (Chrome over CDP, `--use-mock-keychain`, isolated HOME via
`scripts.probe_isolation`, own ephemeral port, real touch events) against the
REAL built bundle of each side: `before-*` is the round-1 head `b784bb379`
(`dist/assets/index--wxUjN5P.js`), `after-*` is this remediation's bundle
(`dist/assets/index-4P-JK8cG.js`). The fixture is the round-1 observer fixture
extended with three round-2 sessions (short ended + long live, both carrying
the spend/context glance, plus a vanished session for the real 404). Nothing
here touches the operator's daemon or sessions.

`measure-r2-before.json` / `measure-r2-after.json` are the raw probes;
`digest.json` is the before/after pair extracted from them.

## U23 = D7 — the first row, covered vs revealed

`hidden_px` = how many pixels of the first transcript row's top sit under the
rung at `scrollTop 0`. The short scenes do not scroll at all (`max 0`), which
is the unreachable case.

| scene (width) | hidden before | hidden after | spacer after | scroller top before → after |
| --- | --- | --- | --- | --- |
| short, no glance row (390) | **28.0** | **0** | 53px | 53 → 53 |
| short, no glance row (320) | **38.2** | **0** | 63.17px | 53 → 53 |
| short, with glance row (390) | 1.6 | 0 | 53px | 79.39 → 79.39 |
| short, with glance row (320) | 11.8 | 0 | 63.17px | 79.39 → 79.39 |
| long, at scrollTop 0 (390) | 18.6 | 0 | 53px | 79.39 → 79.39 |
| long, at scrollTop 0 (320) | 28.8 | 0 | 63.17px | 79.39 → 79.39 |

Frames: `{before,after}-r2-covered-short-{390,320}.png`,
`{before,after}-r2-covered-long-{390,320}.png`. The 320 short frame is the one
the rounds measured: before, `question number 0` is cut by the strip and
nothing can scroll it clear; after, it is fully readable below the strip.

## D9 — the glance row under the transient rungs

`status_covered` = the rung's box overlaps the spend/context row.

| state (390 and 320) | before | after |
| --- | --- | --- |
| degraded | covered (rung 53–79.39 over the row) | **not covered** (rung 79.39–105.78) |
| reconnect | covered (rung 53–79.39) | **not covered** (rung 79.39–105.78) |
| ended | covered | **not covered** |
| scroller top, every state | 79.39 | 79.39 (no movement) |

Frames: `{before,after}-r2-degraded-{390,320}.png`,
`{before,after}-r2-reconnect-{390,320}.png`.

## U24 — the acceptance line

`after-r2-u24-line-390.png`: after a resume whose POST succeeds but whose
session does not come back live, the strip says
`reopening — this can take a few seconds` (before: silence — the frame is
byte-identical to the pre-tap state). `after-r2-u24-later-390.png`: past the
20s cap it reads `still reopening — it has not come up yet`. The line lives
inside the measured overlay, so the reserve covers it too (strip box 53 →
79.39px, first row still clear).

## U25 / U26 — copy

* `resume reopens it in ~` → `resume reopens it in your home folder`; strip
  height unchanged at 390 (53px) and at 320 (63.17px).
* real 404 arm, before: `Could not resume: no such past session: r2-vanished`;
  after: `Could not resume: this session is no longer saved`.
  Frames: `{before,after}-r2-u26-404-390.png`.

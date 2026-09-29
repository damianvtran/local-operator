# Mobile UX batch 2 — round-3 remediation evidence

Captured headless (Chrome over CDP, `--use-mock-keychain`, isolated HOME via
`scripts.probe_isolation`, own ephemeral port, real touch events) against the
REAL built bundle of each side: `before-*` is the round-2 head `2b12ef18c`
(`dist/assets/index-4P-JK8cG.js`), `after-*` is this remediation's bundle
(`dist/assets/index-Bb92uoPC.js`). Same fixtures as round 2, plus a seed so the
long session can reach the resume route.

`measure-r3-*.json` are the raw probes (mid-history scene + the round-2 scene
set re-run on the new bundle, `measure-r2-r3.json`); `digest.json` is the
before/after pair extracted from them.

## The mid-history reader (reviewer MAJOR 1 = UX U27 = QA Q1)

A long transcript parked mid-history with real touch drags; the rows on screen
are tracked by id across each rung's APPEAR and CLEAR, and the number below is
the largest move of any tracked row (px; negative = up).

| transition (width) | before | after |
| --- | --- | --- |
| degraded appears / clears (390) | **±25.61** | **0.39** |
| degraded appears / clears (320) | **±25.61** | **0.39** |
| ended appears / clears (390) | **±53.0** | **0.0** |
| ended appears / clears (320) | **±62.83** | **0.0** |
| `reopening…` line grows (390) | **−25.61** | **0.61** |
| `reopening…` line grows (320) | **−26.61** | **0.39** |

The fix is one mechanism with one hand on `scrollTop`: the scroller carries
`[overflow-anchor:none]` (native scroll anchoring can no longer compensate the
same insertion), and the follow effect compensates by the reserve's own growth
— the spacer plus the flex gap it opens, measured off the DOM, so the
compensation is exact and cannot count rows appended at the tail in the same
commit. Frames:
`{before,after}-r3-mid-{degraded,ended}-{390,320}.png`,
`{before,after}-r3-mid-line-*.png`.

Tail (pinned reader) is unchanged: `scrollTop == max` in base, degraded and
cleared, before and after (`dist 0` in every state).

## D10 — the ended title at 320 (design round 3, NIT)

`this session has ended — its history is kept` → `session ended — history kept`.
The ended strip's height at 320 goes **63.17 → 53 px** (one text line, matching
390); the spacer follows to 53 and the first row's clearance is unchanged
(`hidden 0`). After side: `after-r3-r2-covered-short-320.png`; the round-2
head's frame is pinned in the round-2 evidence at `ec1321c53`
(`after-r2-covered-short-320.png`).

## Round-2 acceptance re-run on the new bundle (`measure-r2-r3.json`)

`hidden_px` at scrollTop 0 is 0 in every covered scene at 390 and 320 (spacer
53 everywhere, scroller top unmoved: 79.39 with the glance row, 53 without);
`status_covered` false; U24's two sentences and U26's mapped 404 render as
before. The only intended difference is the D10 strip height above.

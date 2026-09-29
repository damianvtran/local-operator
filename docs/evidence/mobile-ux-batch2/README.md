# Mobile UX batch 2 — evidence

Captured headless (Chrome over CDP, isolated HOME, own ports, synthetic
projections served by the audit's fixture; never the operator's daemon) against
the REAL built bundle of the branch head `3cb8fbb9b`. `state/before/*` are the
audit lanes' own frames (the pre-fix bundle) — credited, copied for pairing.

## state/

`matrix.txt` is the machine-checked pass log (12/12) for:

- `after/list-chips-390-dark.png` — the list: `Socket unreachable` carries the
  quiet `not answering` chip; `list-chips-ended-390-dark.png` scrolls the
  durable-only row into view: `Ended conversation` carries `ended`.
  (Before: `before/list-390-dark.png` — no chip exists in the vocabulary.)
- `after/ended-session-390-dark.png` — strip "this session has ended — its
  history is kept" + a 44px `resume` (the documented affordance).
  Before: `before/ended-390-dark.png` (identical to a live session).
- `after/degraded-session-390-dark.png` — "not answering — showing its last
  synced view". Before: `before/degraded-390-dark.png` (identical to live).
- `after/pin-refusal-390-dark.png` — `Could not save the pin: no saved messages
  yet — pin it after you send one` (the daemon's own 409 sentence, rendered in
  the session view; before: the ★ flipped back silently).
- `after/login-error-390-dark.png` — `Wrong password.` in `role="alert"`.
- `after/disconnect-session-390-dark.png` / `after/disconnect-list-390-dark.png`
  — the fixture is killed mid-watch: `reconnecting — showing the last synced
  view` (session, stale rows retained) and the list's `reconnecting…` chip.
  Before: `before/disconnect-390-dark.png` (12 s of silence).
- Titles (U4), sampled: list `(3) local operator`; session
  `Mobile UAT — audit the phone surfaces — local operator`; agent
  `viewport-audit — local operator`.

## overprint/ (#12, 320x568)

`measure-{before,after}.json` hold the geometry; at 320x568:

| | before | after |
| --- | --- | --- |
| tasks row vs card top | row bottom 172 over card top 129 → **−43px** | **+43px** |
| subagents row vs card top | row bottom 173 over card top 129 → **−44px** | adjacent (the `−1` in `measure-after.json` is rounding of two touching boxes) |
| column | | `scroll 568/568` — nothing clipped |

`after-320-row-zoom.png` shows the subagents row: the hint stands down below its
measured fit width (353px of viewport), so `· 1 failed` renders WHOLE — before,
the hint (left edge 235.1) overdrew the danger count's tail (right edge 263),
~28px. Full frames at 320/360/390 in `before-*.png` / `after-*.png`.

## contrast.md

D3/D4 before → after ratios (all six palettes now clear their floors) and the
new gate: `pnpm check-themes` (freshness + 1209 assertions across 31 themes)
runs in CI on every mobile-web change.

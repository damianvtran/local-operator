# PR #1018 — review round 1 remediation evidence

Head `2c43bb024` (remediation) vs `12f95f3f6` (reviewed head).

Both trees captured in the SAME mode and viewports: headless Chrome
(`--headless=new`), `Emulation.setDeviceMetricsOverride` at 390x844 and
360x780, `dpr=2`, `mobile=true`. Before-frames come from a throwaway worktree
detached at `12f95f3f6` with its own `pnpm build` — never a `git stash` or a
`checkout --` in a live checkout. Both fixtures run `dial_registrants=False`
under an isolated `HOME`, so the operator's daemon was never touched.

`before-*` is the reviewed head; `after-*` is the remediation. Same scenario
names on both sides, so each pair is directly comparable.

## The keyboard divergence (C1/U1) — the fix's whole point

`*-secret-keyboard-300.png` and `*-approval-keyboard-300.png`.

NOT captured by shrinking the viewport: a uniform `setDeviceMetricsOverride`
shrink moves the layout AND visual viewports together, so `dvh` shrinks with
the column and a broken `dvh` cap looks fine. That artefact is what hid this
defect from the UX round's first attempt. These frames pin the column to a
keyboard-reduced height exactly as `screens/session-view.tsx` does on a
`visualViewport` resize, leaving device metrics at full height so `dvh` is
UNCHANGED — which is what an overlay keyboard actually does on iOS.

| | before | after |
|---|---|---|
| 360x780, kb 300 | card 57→525 in a 480px column, cap 468px, `dvh` still 780 | card 96→384 in a 480px column, cap **288px** |
| 390x844, kb 300 | card 57→563 in a 544px column, cap 506px, `dvh` still 844 | card 122→448 in a 544px column, cap **326px** |

The cap now moves with the column; before it did not move at all.

## Reachability, driven by real touch input

`reachability-before.txt` / `reachability-after.txt`
(`scripts/mobile_reachability_check.py`, every scroll an
`Input.dispatchTouchEvent` drag, never a `scrollTop` assignment).

**before: 19/44 checks passed — after: 44/44.**

Every failure on the before side is one of the round's findings, measured:
approve/deny 0/44px on arrival at both viewports (U2), the card overrunning the
pinned column at every keyboard height (C1/U1), and the stacked approval's
decision 0/44px (D1).

## Frame index

| scenario | finding |
|---|---|
| `*-approval-arrival` | U2/Q1 — approve/deny visible ON ARRIVAL, no gesture |
| `*-approval-keyboard-300`, `*-secret-keyboard-300` | C1/U1 — the divergence case |
| `*-ask10-arrival`, `*-ask10-after-gestures` | reachability + U4 (fade cue, option count) |
| `*-stacked-approval-arrival` | D1 — panels held collapsed, nothing clipped |
| `*-stale-error` | U3 — error pinned, visible at the list's foot |
| `*-roster-collapsed-failures` | U5 — `· 3 failed` in the collapsed header |

`geometry-*.json` carries the numbers behind each frame (§4): column
clientH/scrollH, the card's rect and resolved cap, and each control's visible
height.

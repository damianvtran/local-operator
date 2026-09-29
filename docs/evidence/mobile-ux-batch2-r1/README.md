# Mobile UX batch 2 — round-1 remediation evidence

Captured headless (Chrome over CDP, `--use-mock-keychain`, isolated HOME via
`scripts.probe_isolation`, own ephemeral ports, real touch events) against the
REAL built bundle of each side: `before-*` is the PR head `39cc964e0`
(`dist/assets/index-DhLBqppW.js`), `after-*` is the remediation commit's bundle
(`dist/assets/index--wxUjN5P.js`). Nothing here touches the operator's daemon
or sessions; the live leg spawns real children in an isolated sandbox and reaps
them.

`measure-{before,after}.json` and `live-measure-{before,after}.json` hold the
machine-read numbers behind every claim below (every scene writes them from the
same page it photographed).

## U15 — one resume affordance on an ended cut-off session

`before-u15-cutoff-ended-390.png` — the strip's `resume` AND the composer's
full-width `turn cut off — tap to resume` (2 buttons; the composer's sends
`continue` to a dead runtime). `after-u15-cutoff-ended-390.png` — the strip's
`resume` only. Counts: before `["resume", "turn cut off — tap to resume"]`,
after `["resume"]` (`u15_dual_resume.resume_buttons`).

LIVE, both sides (`live-measure-*`): a real daemon (started like
`service.py`, scan loop live) spawns a child; the phone sends a message; the
child is SIGKILLed — leg A 0.35 s after the send tap (a real kill while the
turn is being worked; the sandbox has no provider key, so a turn ends
involuntarily inside ~1 s and the daemon never projected `streaming` long
enough to sample, which is why the kill's mid-turn-ness is stated as intent,
not as a sampled state), leg B after the turn's own involuntary end. Both
deaths reach the phone with the cut-off receipt, which is the shape U15 gates
on. Affordance counts: before 2, after 1 in BOTH legs
(`legA.affordance_count`, `legB.affordance_count`). `after-liveB-resumed-390.png`
— tapping the strip's resume respawns the session, the history is kept
(`history_kept: true`), no strip remains. `before/after-liveB-after-send-390.png`
— the composer's button REMAINS for a live session (the gate is on `ended`
only).

## U16 — a successful resume cannot strand the control

`before-u16-resume-norevive-390.png` — the POST succeeded, no live frame
arrived, and the button sat at `resuming…` disabled (`disabled: true`).
`after-u16-resume-norevive-390.png` — `resume`, enabled; a second press
re-POSTs (pinned in `session-view.health.test.tsx`).

## U17 — the pin refusal clears once its instruction is followed

`before-u17-pin-after-send-390.png` — the message was sent and the refusal
line is STILL on screen (`alert_after_send` = the same sentence).
`after-u17-pin-after-send-390.png` — cleared (`alert_after_send: null`).

## D5 — one refusal voice

`before/after-d5-resume-refusal-390.png` — `observer daemon cannot start
sessions` → `Could not resume: observer daemon cannot start sessions`
(`d5_refusal.refusal`). The past-sessions screen uses the same helper.

## U19 / D4 — the strips no longer move the column

`measure-*.u19_y_stability` (the transcript scroller's top edge, at its
settled scroll position):

| width | state | before top | after top |
| --- | --- | --- | --- |
| 390 | base / no strip | 53 | 53 |
| 390 | degraded (26px strip) | 79 (+26) | 53 (±0) |
| 390 | cleared | 53 | 53 |
| 390 | ended (button strip) | 106 (+53) | 53 (±0) |
| 390 | restored | 53 | 53 |
| 320 | degraded | 79 (+26) | 53 (±0) |
| 320 | ended | 106 (+53) | 53 (±0) |

`reconnect` (fixture SIGKILLed under the open view): `transcript_top` 53 → 79
before; 53 → 53 after, at 390 and 320. A finger drag STARTING on the strip row
scrolls the transcript under it after the change (`overlay_scroll.moved:
true`, scrollTop 456 → 83) — before, that 26px band was a dead zone for
scrolling (`moved: false`).

## D2 / D6 / review NIT 2 — one hint stand-down for the held-shut rows

The shared class is `max-[385px]:hidden` (compiled: `@media not all and
(min-width:385px)` — hidden strictly below 385; see `boundary-385.json`).

| viewport | before subagents | after subagents | before tasks | after tasks |
| --- | --- | --- | --- | --- |
| 390 | label 35px, hint shown | label 35px, hint shown | 57px shown | 57px shown |
| 385 | — | label 30px, hint shown | — | 57px shown |
| 384 | — | label 63px, hint hidden | — | 57px hidden |
| 360 | label 6px, hint shown | label 63px, hint hidden | hint shown | hint hidden |
| 320 | label 41px, hint hidden | label 41px, hint hidden | hint SHOWN | hint hidden |

(`before-hints-360.png` shows the squeezed `s`; `after-hints-360.png` /
`after-hints-320.png` show both rows agreeing.)

## U21 — tab title

`before-u21-title-untitled-390.png` → `document.title` = `session — local
operator` while the header reads `untitled`; after:
`untitled — local operator` (`u21_title.title`).

## U18 / U20

See `after-u15-cutoff-ended-390.png` for the one-line disclosure (`resume
reopens it in ~`) and `before/after-u20-ended-send-390.png` for the failed-send
sentence: `Couldn’t continue this conversation. Try again.` →
`This session has ended — tap resume to continue.` (the draft is retained in
the field in both).

## contrast (D1)

`contrast.md` carries the before/after ratios and the contract's new
assertion count.

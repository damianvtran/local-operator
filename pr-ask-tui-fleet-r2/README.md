# ask-tui fleet scope — round 2 evidence

Captured from `feat/all-asks-tui` @ the round-2 head, against the PR's base
commit `7d21201f3` for the before/after pair. Frames are SVGs with the
`save_capture` measurements beside them (`*.geometry.json`) and were rendered to
PNG with `rsvg-convert` and looked at.

**ONE PROCESS CWD, BOTH SIDES (round 1: D5/F11).** The head scripts ran from
`~/local-operator-worktrees/ask-tui` and the base scripts — from a throwaway
worktree pinned at `7d21201f3` — were invoked with the SAME cwd, so the status
band's cwd cell is identical in every pair and cannot be mistaken for a change.
Verified: `before/ask_shot.svg` and `before/approval_shot.svg` are BYTE-IDENTICAL
to their head twins (the ask-long-descriptions invariant), and the
`before/list-100x30-dark.svg` → `list-100x30-dark.svg` pair differs in exactly
ONE painted line — the list header.

## What each frame is for

| frame | the finding it answers |
|---|---|
| `list-100x30-dark` (+ `before/`) | D3/F11: the `d decline` hint is named again at 100×30; D6: the settled status word sits after the row's own `·` |
| `list-settled-100x30-{dark,light}` | D2/U7: an all-settled queue advertises only `esc collapse`; `All asks settled` (D4/F3) |
| `filter-all/filter-settled-100x30-{dark,light}` | D1: the `?` marker and the `answered` word are the derived `chip-*` inks on `overlay` (the light ramp's own failing pairs) |
| `list-delivering-100x30-{dark,light}` | F3/Q1/U4: a delivering row is PENDING — the clause says `1 answer delivering — the agent will be told`, never `1 settled` |
| `in-flight-100x30-dark` | F2/U5: a row mid-engage paints `…` and refuses a second gesture |
| `list-fleet-capped-100x30-dark` | F6: the fleet list states the backend tally (`20 outstanding`) and withholds the split |
| `list-fleet-100x30-dark`, `list-fleet-190x50-dark`, `list-fleet-100x30-light` | the fleet scope's subject, the per-row conversation handles and the narrow floor |
| `list-60x20-dark`, `list-100x30-nocolor` | D10: the 60-column rung and the colour-less ramp |
| `sidebar-marks-100x30-{dark,light}`, `sidebar-marks-80x24-dark` | U3: the note survives 80×24; D10: the light sidebar frame round 1 lacked |
| `sidebar-gate-100x30-dark` | D10: `!` keeping the cell over an ask mark, while the footer still counts it |
| `sidebar-fleet-100x30-dark` | the door exercised end to end (the ONE list on the fleet scope) |
| `picker-asks-100x30-{dark,light}` | U2: the count on the picker's chrome row, now a press target |

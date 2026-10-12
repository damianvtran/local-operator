# Team Bridge H1 — state frames, F posture, accessible ink, narrow wrap

The `/settings` half of the harness Team Bridge client (agent messages), captured
from the REAL `OperatorApp` with the stylesheet loaded, at the reviewed sizes and
in both effective themes. The availability posture is REPLAYED from the live
wire captures committed beside this file (`wire-captures.json` — the desktop
route's own `result` bodies from the local S11 stack), so the F sentences and
the visible/read-only flags shown here are the ones the wire published, not
hand-built states.

Round 2 (R9) collapses F's sentence to the closed set's NEUTRAL member for
every cause: the payload's `eligible:false` rows are byte-identical across a
plan lapse, a kill and a scope exclusion, and no v1 answer positively
evidence a lapse — so the frames below show "Agent messages are turned off"
for the no-flags, the kill and the unreadable-confirmation arms alike (the
lapse sentence stays reserved for a future positively-evidenced answer).
R10 restores the one-indent facts line: the pane frames show the populated
proactive row at its caller budget (45 cells, +0 over the reserve).

Reproduce (run from the repo root, with the repo's venv; needs `rsvg-convert`
for the PNGs):

```sh
env -u NO_COLOR TERM=xterm-256color \
    LOCAL_OPERATOR_SCRATCHPAD=$(mktemp -d) \
    .venv/bin/python docs/evidence/team-bridge-h1/capture-rig.py
# frames land in $LOCAL_OPERATOR_SCRATCHPAD/frames-r1/

env -u NO_COLOR TERM=xterm-256color \
    LOCAL_OPERATOR_SCRATCHPAD=$(mktemp -d) \
    .venv/bin/python docs/evidence/team-bridge-h1/proactive-rig.py
# pane frames land in $LOCAL_OPERATOR_SCRATCHPAD/proactive-r2/
```

`capture-rig.py`/`proactive-rig.py` are the exact capture scripts used for
these frames. They pin `LOCAL_OPERATOR_NO_SHIMMER=1` (still frames), and BOTH
strip every inherited `CMUX_*`/`LOP_*` variable BEFORE their first harness
import — an isolated HOME, `run_test` and synthetic sessions neutralise
filesystem and identity state, never env-carried live-host identifiers, and a
headless boot carrying an inherited `CMUX_WORKSPACE_ID` has renamed the
operator's real cmux workspaces before (the team incident the rule exists
for); `LOCAL_OPERATOR_*` — each rig's own scratch/config namespace — is
deliberately retained. They write a settled pair (two captures with a pause
between) plus a `.geometry.json` per frame; `frames-results.json` carries the
per-case numbers below, including the F `space` no-write check, and
`r2-proactive-results.json` the pane arithmetic. A regression
(`tests/unit/team_bridge/test_evidence_rigs.py`) executes each rig's stdlib
prologue in a subprocess with sentinel `CMUX_*`/`LOP_*` values and asserts
none survive to the import boundary.

The before-baseline is QA round-1's `verified-*` frames (dark-only B/F set, and
their observed defects): D1's faint 1.97:1 helper and the configured-CoS helper
loss, D3's eligible on-ramp in F, D4's 74→54-character clip at 60×20.

| file | state | theme | what it shows |
| --- | --- | --- | --- |
| `r1-B-80x24-dark-enabled.*` | B, default | dark | the H1 rows + "Off by default…" helper at the `muted` ink |
| `r1-B-80x24-light-enabled.*` | B, default | light | the same, on the genuine light palette |
| `r1-B-80x24-dark-cosconfigured.*` | B, CoS set | dark | the wake explanation SURVIVES a configured CoS (D1) |
| `r1-B-60x20-{dark,light}-enabled.*` | B, narrow | both | the helper WRAPS; "eligible organization" stays reachable (D4) |
| `r1-F_unknown-80x24-{dark,light}-enabled.*` | F, no positive evidence | both | the NEUTRAL closed-set sentence; view-only (no enter key); space writes nothing |
| `r1-F_unknown-60x20-dark-enabled.*` | F, narrow | dark | the same sentence at 60×20 |
| `r1-F_kill-80x24-dark-enabled.*` | F, ops kill (wire recaptured) | dark | byte-identical sentence to the no-evidence arm (R9) |
| `r1-absent-80x24-dark.*` | A/G | dark | no section header/rows (256 rows vs 259) |
| `r2-proactive-150x40-{dark,light}-{1,2}.*` | populated proactive pane | both | the agents pane's facts row at its caller budget; before-fix it cost +12 cells and clipped `model/name` (QA round 2) |

Geometry and ink, from `frames-results.json` (per-case `.geometry.json` beside
each frame; native 8×17 px cells):

- 80×24: body 76×8, virtual 75×259 with the section present; **256 rows** when
  absent. 60×20: body 56×4, virtual 55×259, `detail_two_line` true (the wrap).
- Effective themes are logged per frame (`theme_effective`), never inferred
  from filenames; the light set renders on the light palette (#efeae0 surface).
- Ink: the H1 explanation resolves `muted` — **7.93:1** dark (#b5afa2 on
  #1e1a14) and **6.58:1** light (#565147 on #efeae0), both above the 4.5:1
  normal-text floor, against the `faint` comparator at 1.81:1 / 1.90:1 that
  the design round's D1 flagged.
- F: ONE sentence in v1 — "Agent messages are turned off" — for the
  no-evidence, kill and unreadable-confirmation arms (R9; the frames carry it
  at 80×24 dark/light and 60×20); `read_only: true`,
  `space_enabled_before_after: [false, false]` — the space key writes nothing
  (view-only controls; pause/opt-out stay unlocked server-side).
- Proactive pane (R10): the facts line at helper room 80 renders 45 cells,
  **+0 over the reserve** (before the fix: 57 cells, +12, `model/name`
  clipped — QA round-2 `proactive` matrix); the narrow pane arm truncates
  inside budget with the `proactive` marker intact and painted `muted`; the
  settled pairs are byte-identical SVGs (no reflow after idle).

Wire receipts for the states above (the original 27 wire cases, plus the
round-2 F recapture: 5/5 live cases — no-flags F, one-dispatch trace with NO
`/messaging/state` read, kill F, and both route bodies F/read-only/neutral)
live in the implementer session's scratchpad (`wireproof-matrix.json`,
`wireproof-dark.json`, `f-recapture-matrix.json`, `f-recapture.log`) and are
summarized in this PR's QA remediation comments.

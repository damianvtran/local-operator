# Team Bridge H1 — state frames, F posture, accessible ink, narrow wrap

The `/settings` half of the harness Team Bridge client (agent messages), captured
from the REAL `OperatorApp` with the stylesheet loaded, at the reviewed sizes and
in both effective themes. The availability posture is REPLAYED from the live
wire captures committed beside this file (`wire-captures.json` — the desktop
route's own `result` bodies from the local S11 stack), so the F sentences and
the visible/read-only flags shown here are the ones the wire published, not
hand-built states.

Reproduce (run from the repo root, with the repo's venv; needs `rsvg-convert`
for the PNGs):

```sh
env -u NO_COLOR TERM=xterm-256color \
    LOCAL_OPERATOR_SCRATCHPAD=$(mktemp -d) \
    .venv/bin/python docs/evidence/team-bridge-h1/capture-rig.py
# frames land in $LOCAL_OPERATOR_SCRATCHPAD/frames-r1/
```

`capture-rig.py` is the exact capture script used for these frames. It pins
`LOCAL_OPERATOR_NO_SHIMMER=1` (still frames), strips inherited `CMUX_*`/`LOP_*`,
and writes a settled pair (two captures with a pause between) plus a
`.geometry.json` per frame; `frames-results.json` carries the per-case numbers
below, including the F `space` no-write check.

The before-baseline is QA round-1's `verified-*` frames (dark-only B/F set, and
their observed defects): D1's faint 1.97:1 helper and the configured-CoS helper
loss, D3's eligible on-ramp in F, D4's 74→54-character clip at 60×20.

| file | state | theme | what it shows |
| --- | --- | --- | --- |
| `r1-B-80x24-dark-enabled.*` | B, default | dark | the H1 rows + "Off by default…" helper at the `muted` ink |
| `r1-B-80x24-light-enabled.*` | B, default | light | the same, on the genuine light palette |
| `r1-B-80x24-dark-cosconfigured.*` | B, CoS set | dark | the wake explanation SURVIVES a configured CoS (D1) |
| `r1-B-60x20-{dark,light}-enabled.*` | B, narrow | both | the helper WRAPS; "eligible organization" stays reachable (D4) |
| `r1-F_lapse-80x24-{dark,light}-enabled.*` | F, plan lapse | both | "Plan ended — reading only"; view-only (no enter key); space writes nothing |
| `r1-F_kill-80x24-dark-enabled.*` | F, ops kill | dark | "Agent messages are turned off"; same posture |
| `r1-absent-80x24-dark.*` | A/G | dark | no section header/rows (256 rows vs 259) |

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
- F: `read_only: true`, `space_enabled_before_after: [false, false]` — the
  space key writes nothing in both F cases (view-only controls; pause/opt-out
  stay unlocked server-side).

Wire receipts for the states above (plus all 27 wire cases) live in the
implementer session's scratchpad (`wireproof-matrix.json`, `wireproof-dark.json`,
`wireproof.log`) and are summarized in this PR's QA remediation comment.

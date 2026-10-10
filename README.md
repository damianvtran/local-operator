# Video surfaces (TUI) — before/after frames (`feat/video-tui-surfaces`, head `30ba6b33a7b3444d5687457caf567be5a9af0bd7`)

Frames, SVG sources and the capture rig for the TUI slice of the video phase —
surfaces spec `video-surfaces-design.md` §1.1/§5 (R1/R4/R5), frozen contract
§4.2/§5.4. They live on this `evidence/` branch, never in the PR's tree
(AGENTS.md §7); review comments link each image by the commit-pinned
`raw.githubusercontent.com` URL for this branch.

* Head described: `30ba6b33a7b3444d5687457caf567be5a9af0bd7`
* Capture: the real `OperatorApp` under `local_operator.tcss` via Textual
  `run_test`, `scripts/visual_capture.save_capture`; states driven through the
  app's own ToolStarted/ToolUpdated/ToolEnded path. No live session, no
  browser, nothing on the operator's screen. `env -u NO_COLOR
  TERM=xterm-256color`; native 8x17px cells.
* Before-frames: the same rig, unchanged, against a detached worktree at
  `ea2df49b14` (base), reached with `PYTHONPATH=<base tree>` so the venv's
  editable finder cannot win; the rig asserts the tree it imported.
* `running-image-control`: the same running state for `generate_image`, whose
  row carries `⟨ctrl+c to interrupt⟩` — which the video row must not (R4).
* The band pair runs one video job beside one bash job: base `▣ 1 job`,
  head `▣ 2 jobs`.

## Frames

| file | what it shows |
| --- | --- |
| `after-running-100x30` / `before-running-100x30` | the collapsed row, running: hint absent on both, the variant slot shows only the state word |
| `after-running-image-control-100x30` | control: the image row carries the hint the video row must not |
| `after-running-body-100x30` / `before-running-body-100x30` | expanded: the live body (word + 10-cell canvas + log tail) against the base's generic state line only |
| `after-running-body-flat-100x30` | reduced/flat: the shimmer band parked |
| `after-queued-100x30` / `before-queued-100x30` | queued cell; body carries `2 requests ahead` |
| `after-cancelling-100x30` / `before-cancelling-100x30` | hold word `⋯ cancelling`; no hint either side |
| `after-settled-100x30` / `before-settled-100x30` | the receipt line; the kind-video attachment mounts nothing, nothing breaks |
| `after-settled-150x40` | the full receipt at width (no elision) |
| `after-band-100x30` / `before-band-100x30` | `▣ 2 jobs` vs `▣ 1 job` with one video job running |

Note: the base's queued call has no expandable body (nothing beyond the row),
so there is no `before-queued-body` frame — that absence is the delta.

## Numbers

* Grid 100×30 → native 800×510 px; 150×40 → 1200×680 px (8×17 cells, Menlo).
* Per-frame geometry: `*.geometry.json` beside each SVG.
* sha256 of every artifact: `capture/SHA256SUMS`.

## The rig

`capture/video_card_shot.py` — one script, one state matrix, parameterised by
`LOP_TREE` (and `LOP_SHIMMER=off` for the reduced/flat frame):

    cd <tree> && LOP_TREE=$PWD LOP_SHIMMER=on <venv>/bin/python video_card_shot.py running-body out.svg 100x30

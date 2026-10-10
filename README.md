# Video surfaces (TUI) — remediation-round frames (`feat/video-tui-surfaces`, head `bc89c52ba71633af7fed71389787ace11463fef5`)

Frames, SVG sources and the capture rig for the TUI slice of the video phase —
surfaces spec `video-surfaces-design.md` §1.1/§5 (R1/R4/R5), frozen contract
§2.4/§4.2/§5.4. They live on this `evidence/` branch, never in the PR's tree
(AGENTS.md §7); review comments link each image by the commit-pinned
`raw.githubusercontent.com` URL for this branch.

**Round 2** — remediation round 1 on PR #2196: the settled pair is recaptured
with the §2.4 submit receipt (design D1) and the held-delivery frame added
(D2). The settled frames on `evidence/video-tui-surfaces-r1` are superseded;
every other frame there is unchanged here.

* Head described: `bc89c52ba71633af7fed71389787ace11463fef5`
* Capture: the real `OperatorApp` under `local_operator.tcss` via Textual
  `run_test`, `scripts/visual_capture.save_capture`; states driven through the
  app's own ToolStarted/ToolUpdated/ToolEnded path (and, for `held`, the boot
  replay of a held `job_result` row — the same construction
  `tests/unit/tui/test_harness_injection_replay.py` uses). No live session, no
  browser, nothing on the operator's screen. `env -u NO_COLOR
  TERM=xterm-256color`; native 8x17px cells.
* Before-frames: the same rig, unchanged, against a detached worktree at
  `ea2df49b14` (base), reached with `PYTHONPATH=<base tree>` so the venv's
  editable finder cannot win; the rig asserts the tree it imported.
* `running-image-control`: the same running state for `generate_image`, whose
  row carries `⟨ctrl+c to interrupt⟩` — which the video row must not (R4).
* The band pair runs one video job beside one bash job: base `▣ 1 job`,
  head `▣ 2 jobs`.

## D1 — the settled pair now stages the submit receipt

The settled states render the §2.4 SUBMIT receipt — the text the card actually
settles with, because the call is detached and returns instantly:

    Video generation started — job `abc123` (mode i2v, 8 s, ~$1.60 est., Radient). You'll get a completion notice when it settles; cancel with `jobs op='cancel', job_id='abc123'`.

Measured from the SVGs: at 100×30 the line elides at ``…You'll get a…``
(matching the design comment's model exactly); at 150×40 the cut lands one
cell earlier than the comment's prediction — ``…cancel with `j…`` against the
modelled ``` …`jo… ``` (the built line's budget is 142 cells; a 143-cell
budget reproduces the comment's string). The completion sentence ("Video ready
— …") is not staged on the card at all any more; it appears where it belongs,
on the delivered `job_result` row, in `after-held-100x30`. No
`AttachmentContent` rides the submit payload (a kind-video attachment belongs
to the completion; "mounts nothing broken" stays with
`test_tool_result_image_blocks.py`).

The 150×40 clip is the generic per-line crop (`_append_output_body` →
`truncate_cells`); whether the receipt's cancel-hint tail deserves a
wrap/claim carve-out (the partial-disclosure precedent) or a shorter runtime
string (#2191) is assessed in the QA-remediation comment and left to the
design r2 delta — no production change this round.

## Frames

| file | what it shows |
| --- | --- |
| `after-running-100x30` / `before-running-100x30` | the collapsed row, running: hint absent on both, the variant slot shows only the state word |
| `after-running-image-control-100x30` | control: the image row carries the hint the video row must not |
| `after-running-body-100x30` / `before-running-body-100x30` | expanded: the live body (word + 10-cell canvas + log tail) against the base's generic state line only |
| `after-running-body-flat-100x30` | reduced/flat: the shimmer band parked |
| `after-queued-100x30` / `before-queued-100x30` | queued cell; body carries `2 requests ahead` |
| `after-cancelling-100x30` / `before-cancelling-100x30` | hold word `⋯ cancelling`; no hint either side |
| `after-settled-100x30` / `before-settled-100x30` | the settle with the §2.4 submit receipt + `✓ 0.4s` spine; base reads the same — the diff carries the settle unchanged (R5) |
| `after-settled-150x40` / `before-settled-150x40` | same at 150 cols; the elision lands inside the cancel hint (see above) |
| `after-held-100x30` / `before-held-100x30` | the held-delivery notice for a video completion (base-identical, R6): the completion text + `[session warning] held when it arrived — no turn ran for it at that point` |
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

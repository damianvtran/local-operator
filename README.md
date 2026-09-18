# PR #1193 — design review round 1 frames (`/links`, head `0299e3b0`)

Frames for the `### Design review — round 1` comment on
[damianvtran/local-operator#1193](https://github.com/damianvtran/local-operator/pull/1193).
They live on this `evidence/` branch, not in the PR's tree, per AGENTS.md §7
("Evidence goes on the PR, never into the repository"). The comment links each
frame by the commit-pinned `raw.githubusercontent.com` URL for this branch.

* Reviewer: independent designer subagent (see the comment's `Designer:` line).
* Head reviewed: `0299e3b00a9ccfad8db8f0e071cd6a7eb8c66b8b`.
* Before-frames: `133818fe6d09bfd8e6fcc3e270888ac02ad2beb8`, rendered from a
  throwaway worktree of that commit with the same script (`LO_ROOT=` that tree).
* Native capture: `scripts/visual_capture.save_capture` at 8x17 px cells / 13 px
  Menlo, so a 100x30 terminal is 800x510 px. Frames F2/F4/F5/F6 are 2x
  rasterisations of the same SVGs; nothing else about them is altered.

## The frames

| file | what it is |
| --- | --- |
| `F1-too-small-notice.png` | the too-small notice at 30x8: sibling `/copy` next to `/links` next to `/links` at 38x8 — D1 |
| `F2-faint-glosses.png` | 100x30 at 2x, and the card's floor at 38x9 — the meta row's ink — D2 |
| `F3-pointer.png` | pointer resting, pointer moved over row 2 (byte-identical), the same row clicked — D3 |
| `F4-round3-shape.png` | `[https://a.test/x](https://a.test/x)` and `[see https://a.test/x](https://b.test/y)` at head and at `133818fe6` — the round-3 fix, before/after |
| `F5-shapes-parens-ipv6.png` | paren depth 2, depth 4, IPv6 literal — whole URLs |
| `F6-shapes-several-long-narrow.png` | three links in one prose row; a 118-cell URL at 100x30; the same at 44x24 |
| `F7-card-100x30.png` | the ordinary frame |
| `F8-card-many.png` | 24 links: the 11-row window and the counter |
| `F9-receipts.png` | after `enter`: the success receipt and the no-browser receipt |
| `F10-states.png` | empty state, 44x24, and the card's floor at 38x9 |

## Captured digests (the numbers behind the frames)

`capture/digest-*.json` — per case: terminal size, screen virtual size and
scrollbar, transcript region/size/virtual/scrollbar, card region and
`styles.height`/`styles.width`/`padding`, `_row_budget()`, `_card_width()`,
`_content_size()`, body region, `_selected`, `_offset`, the target list, and
every painted row as `(text, ink colour, cells)` runs.

Key readings (100x30 ⇒ 98x28 screen content box, card centred):

* ordinary card `71x7` at `[14,11]`, body `67x5`, `height: auto`, `padding: 1 2`
  — border box = 7 = 2 padding + 2 title/rule + 1 row + 1 blank + 1 footer, so
  the card adds its own padding back and clips nothing. `virtual == size` on
  every screen measured; **no screen scrollbar in any frame** (the only
  `show_vertical_scrollbar` seen is the transcript's own, in the 24-link case,
  whose virtual is 49 rows against 21 actual).
* `_card_width()` is the widest of title / full footer / every visible row, so
  over 3 short links the card is 33 wide because the footer is.
* 24 links: card `43x18` at `[28,6]`, `_row_budget() == 11`, counter row
  `showing 1–11 of 24`, footer still on screen.
* 44x24 (42x22 content): card `42x7` at `[1,8]` — flush with the content box.
* 38x9 (36x7 content): card `36x7`, `_row_budget() == 1`, footer shed to
  `enter open · esc cancel`; 44x8 / 36x8 / 34x8 / 30x8 are below the floor
  (`is_drawable()` false) and paint the sibling notice instead.
* every case: the frame saved again after one `pilot.pause()` is byte-identical
  (no reflow after paint); the after-`enter` frames likewise.

## Reproducing

```sh
# from a worktree of the head, with its own venv
env -u NO_COLOR -u CMUX_WORKSPACE_ID -u CMUX_SURFACE_ID -u CMUX_TAB_ID \
  TERM=xterm-256color LO_ROOT=$PWD .venv/bin/python capture/capture.py OUTDIR
env -u NO_COLOR TERM=xterm-256color LO_ROOT=$PWD \
  .venv/bin/python capture/hover.py OUTDIR      # the pointer frames
# before-frames: the same capture.py with LO_ROOT=/tmp/lo-before-133818fe6
```

Both scripts isolate `HOME`/`LOCAL_OPERATOR_CONFIG_DIR` before importing app
modules (`scripts.visual_capture.isolate_capture`), patch
`local_operator.mcp.auth.open_browser_quietly` with a spy so no browser is ever
launched, and drive the real `OperatorApp` (production `local_operator.tcss`) —
not the CSS-less hosts in `tests/` — by typing `/links` into the real editor.

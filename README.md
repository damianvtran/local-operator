# PR #1193 — remediation round-4 evidence frames (`/links`, head `6bb58ed9`)

Frames and scripts for the `### Agent review remediation — round 4` and
`### Design review remediation — round 1` comments on
[damianvtran/local-operator#1193](https://github.com/damianvtran/local-operator/pull/1193).
They live on this `evidence/` branch, not in the PR's tree, per AGENTS.md §7
("Evidence goes on the PR, never into the repository"); the comments link each
image by the commit-pinned `raw.githubusercontent.com` URL for this branch.

* Author: the PR's implementer (coder lane) — see the comments' `Author:`/`Scope:` lines.
* Head described: `6bb58ed9382cd8d2b7af287769d265e9df7a76a2`.
* Before-frames: rendered from the same worktree with the two changed source
  files restored to `0299e3b0` (`git checkout --`), then put back — same host,
  same venv, same script, so a frame pair differs only by the delta.
* Capture: the real `OperatorApp` under the production `local_operator.tcss`,
  `/links` typed into the real editor; no browser is launched (the opener is a
  spy), and no live lop session, config dir or cmux workspace is touched.

## The frames

| file | what it is |
| --- | --- |
| `frames/E1-notice-30x8.jpg` | the too-small notice at 30x8: before (`terminal too small for` — the command named, ` · esc` clipped) above, after (`too small · esc`) below — design D1 |
| `frames/E2-meta-ink-100x30.jpg` | the card's meta layer at 100x30, cropped to the card: before (footer glosses, counter words and every row's sender at `faint` = 1.49:1 on `#302a20`) above, after (`dim` = 3.43:1, ` · ` separators still `faint`) below — design D2 |
| `frames/E3-hover.jpg` | the pointer over row 2: before (top — byte-identical to the resting frame, no highlight) / after resting (middle) / after moved (bottom, `raised` ground on that row, hand pointer) — design D3 |
| `capture/*.svg` | the native SVG sources those three images come from: `hover-before-resting.svg` / `hover-before-moved.svg` (byte-identical), `hover-after-resting.svg` / `hover-after-moved.svg`, `notice-30x8-{before,after}.svg`, `meta-many-{before,after}.svg` |

## The scripts (reproducible, not part of the PR)

| file | what it does |
| --- | --- |
| `capture/frames.py` | drives the app for the three frame pairs above (`--out DIR`) |
| `capture/hover.py`, `capture/capture.py` | the design round-1 harness, run unchanged against this head and against `0299e3b0` |
| `capture/drive.py` | the end-to-end extraction table: `/links` typed into the real editor, the card's painted rows and the opener's receipt per shape |
| `capture/sweep_fuzz.py` | the depth sweep (10 arms x depths 0/1/2/3/4/8) and the 40,000-text differential fuzz against a longest-valid-prefix oracle |
| `capture/fuzz_mutants.py` | the same fuzz against `0299e3b0`, `133818fe6` and a `]`-in-`_BODY_STOP` mutant, so the instrument is shown to fail |
| `capture/contrast.py` | the WCAG 2.1 ratios for the theme's tokens on `$lo-overlay`, from the theme's own colours |
| `capture/rich_spans.py` | what Rich's `Markdown` actually emits for a markdown link, an autolink and a bare URL (design D5's evidence) |

## Captured digests (the numbers behind the frames)

* Card ground `$lo-overlay` = `#302a20` (the same value `SessionPickerScreen` uses):
  `faint` #4a4539 = **1.49:1**, `dim` #837c6d = **3.43:1**, `muted` #b5afa2 = 6.51:1,
  `fg` #e9e5db = 11.30:1, `accent` #38c96a = 6.59:1.
* Frame hashes (sha256, first 16): `hover-resting` before `b23e8bf85dea65e9`,
  `hover-moved` before `b23e8bf85dea65e9` (**identical** — the design round's
  measurement, reproduced), `hover-moved` after `9062fd6bf2ac3d03` against
  `hover-resting` after `1372488f9696ca06` (**differs**).
* The design round's own `hover.py` on this head: `pointer_after_move` `default`
  → `pointer`, `frame_changed_on_hover` `false` → `true` (before/after).
* The design round's own `capture.py`, case `links30x8`: painted
  `terminal too small for` before, `too small · esc` after; `links38x8` and the
  sibling `/copy` are unchanged in both.

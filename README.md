# Design round 2 — PR #1193 (`/links`), head `6bb58ed9`

Evidence for `### Design review — round 2` on `damianvtran/local-operator#1193`.
Nothing here is in the PR's tree: this branch is the review's own artifact
(`AGENTS.md` §7).

**Head reviewed:** `6bb58ed9382cd8d2b7af287769d265e9df7a76a2` (`fix(tui): pair a
link body's brackets, and light the /links card's meta layer`), against the
round-1 head `0299e3b00a9ccfad8db8f0e071cd6a7eb8c66b8b`.

**Tree:** the existing worktree `~/lo-wt/open-links-r3`, verified clean and at
that exact SHA before use. Frames come from the real `OperatorApp` with the
production `local_operator.tcss` — never from the CSS-less test hosts — over a
seeded transcript, with `/links` typed into the real editor and submitted.

## Frames → findings

| frame | what it shows | finding |
| --- | --- | --- |
| `frames/R2-F1-meta-100x30.png` | the card at 100x30: title fg, rows `❯`-marked, senders and footer words all at `dim`, ` · ` separators at `faint` | D2 fixed |
| `frames/R2-F2-meta-many-100x30.png` | 24 links: the `showing 1–11 of 24` counter, the footer and every row's `agent` at `dim` | D2 fixed (counter + large directory) |
| `frames/R2-F3-hover-resting-vs-hovered.png` | left: resting (SHA `dc342fd2…`); right: pointer over row 1, `raised` band (SHA `223c061a…`) | D3 fixed |
| `frames/R2-F4-notice-band-30-38x8.png` | 30x8, 35x8, 36x8 → `too small · esc` whole; 37x8, 38x8 → `terminal too small for /links · esc` whole | D1 fixed |
| `frames/R2-F5-sibling-meta-ink.png` | `/links` footer, `/copy` footer, `/resume` meta rows — glyph ink peaks at `#837c6d` (`dim`) on all three | D2 matches siblings |
| `frames/R2-F6-light-ramp-100x30.png` | the same card on the shipped `light` (Operator Light) ramp | D2 on the paper ramp (round-1 gap closed) |
| `frames/R2-F7-bracket-shapes.png` | `[https://a.test/x]` / `[[https://a.test/y]]` (top) and `[https://en.wikipedia.org/wiki/Foo_(bar)]` (bottom) — no stray `]` on any row | the delta's other half |
| `frames/R2-F8-transcript-d5.png` | a bare URL painted in the same ink as the prose around it, `[the docs](…)` in `signal` | D5 still open |
| `frames/R2-F9-sender-column-94-wide.png` | the 94-cell card: `agent` two-thirds of a screen from its URL | D4 / D7 |

## Numbers behind the frames

* Screen content box `98x28` inside a `100x30` terminal (`Screen { padding: 1 }`);
  `app.screen.virtual_size == app.screen.size` and
  `show_vertical_scrollbar == False` in **all 13** cases — no screen-level
  scrolling, and the card never grows past its box.
* Card regions, all `height: auto`, `padding: 1 2`, so the body box is the
  region minus 4 cells of width and 2 of height: `meta` `[31,10,37,9]` body
  `33x7`; `many` `[28,6,43,18]` body `39x16`; `long` `[1,11,98,8]` body `94x6`;
  `narrow44` `[1,8,42,7]` body `38x5`. Body height always equals
  padding 1 + title 1 + rule 1 + rows + blank 1 + (counter 1) + footer 1 +
  padding 1 — nothing clipped.
* Row height is 1 cell everywhere; every row is `no_wrap=True,
  overflow="ellipsis"`, so a row never wraps.
* Contrast from the theme's own tokens (`theme_mod.BRAND_TOKENS`, WCAG 2.1
  relative luminance) against the card's own ground `$lo-overlay`:
  `dim #837c6d` on `#302a20` = **3.43:1** (was `faint #4a4539` = **1.49:1**),
  `muted` 6.51:1, `fg` 11.30:1, `accent` 6.59:1. On the paper ramp
  (`light`, ground `#d8d1c0`): `dim` **2.72:1**, `faint` 1.50:1 — see the
  round-2 note.
* Painted pixels, not just requests: card ground `(48,42,32)` = `#302a20` on
  the dark ramp and `(216,209,192)` = `#d8d1c0` on the paper ramp; the hovered
  row's ground `(39,34,25)` = `#272219` = `raised`, against `(48,42,32)`
  resting; `/links`, `/copy` and `/resume` meta rows all peak at `#837c6d`.
* Resting rows are **byte-identical** to the round-1 head's for `several`,
  `long` and `many` (compared against round 1's `capture/digest-head.json`):
  the delta moved ink, not a single cell.
* Every case was captured twice, one `pilot.pause()` apart: all 13 pairs are
  byte-identical (no reflow after paint), and the hover frame is byte-identical
  one pause after the pointer stops moving.

## Reproducing

```sh
WT=~/lo-wt/open-links-r3          # at 6bb58ed9, clean
OUT=/tmp/lo-design-1193-r2/out
env -u NO_COLOR -u CMUX_WORKSPACE_ID -u CMUX_SURFACE_ID TERM=xterm-256color \
  LO_ROOT=$WT $WT/.venv/bin/python capture.py $OUT      # frames + digest.json
env … $WT/.venv/bin/python hover.py $OUT                # hover-report.json
env … $WT/.venv/bin/python light.py $OUT                # the paper ramp
env … $WT/.venv/bin/python shapes.py $OUT               # the bracket shapes
```

`capture.py`, `hover.py`, `light.py` and `shapes.py` are in `capture/`; so are
`digest.json` (per-case geometry and per-row ink runs), `hover-report.json` and
`shapes.json`. `crop.py` and `box.py` render an SVG at `Nx` and cut a
terminal-cell box out of it, by coordinates or by finding the card's own ground.
All four drive the real app in an isolated `HOME`/config (`isolate_capture()`),
run one process at a time, and write only under `/tmp/lo-design-1193-r2`.

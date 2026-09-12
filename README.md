# Design review evidence — PR #994 (`/analytics` pagination lands in one frame)

Round 1, designer. Base `b85a610cc`, head `2eb17f4aee713a656165ba126fdbff9f34a0572a`.
Nothing here is part of the product tree; evidence only, per the repo's convention
(`evidence/pr-994` is the author's branch from the same PR).

Every still was produced by the repo's own recipe: `scripts.probe_isolation`
first (HOME/config isolated, `CMUX_*` dropped), the real `OperatorApp` with its
production stylesheet, `scripts.visual_capture.save_capture` (native 8x17 cells,
`.geometry.json` beside each SVG). `frames/scripts/` holds the capture scripts;
`svgrows.py` parses a captured SVG back into the rows that were **painted**, which
is what the numbers in the review are read from.

## Reproduction

```sh
# head tree, detached at the PR head; base tree, detached at the PR's base
git -C ~/local-operator worktree add --detach /tmp/design994      2eb17f4ae
git -C ~/local-operator worktree add --detach /tmp/designer994base b85a610cc
ln -s ~/local-operator/.venv /tmp/design994/.venv        # and for the base tree
cd /tmp/design994   && env -u NO_COLOR TERM=xterm-256color .venv/bin/python <script> /tmp/design994  ...
cd /tmp/designer994base && env -u NO_COLOR TERM=xterm-256color .venv/bin/python <script> /tmp/designer994base ...

seamshot.py <repo> <out-dir> <tag> key|container   # the before/after pair + end/home + resize
states.py  <repo> <out-dir>                        # one key per state, frames + row-exactness + hashes
verify_rows.py <repo>                              # painted row == composed line at scroll_y+i
landing.py <repo>                                  # every compositor frame one key writes, with ms
wheel.py <repo> <out.svg>                          # the wheel path, driven on both trees
```

## The frames

| file | what it is |
| --- | --- |
| `after-first-frame-pagedown.{svg,png}` | head, 120x45, the FIRST frame after one `pagedown` key: `scroll_y=29`, 29 rows, all 2958 body cells, thumb in the same frame |
| `before-first-frame-pagedown.{svg,png}` | base, same key, first frame: `scroll_y=5` of 29 — a complete but shifted page (the wave is 18 such frames, not a torn one) |
| `after-end-settled.{svg,png}` | head, `end`: content rows 31..59, the report's last line fully painted, thumb clamped at the track bottom |
| `after-home-identical-to-opening-frame.{svg,png}` | head, `home`: byte-identical to the opening frame |
| `after-resize-100x35.svg`, `after-resize-pagedown-100x35.{svg,png}` | head, resize to 100x35 then one `pagedown`: one complete 84x21 frame, no stale column from the 120-wide layout |
| `before/after-narrow-60x24-pagedown.svg` | head, 60x24: one complete 48x11 frame; the report's overrun is cropped, no horizontal scrollbar |
| `fits-before-pagedown.svg`, `fits-after-pagedown-byte-identical.svg` | head, a report that fits: the hint hides, there is no scrollbar, and `pagedown` paints nothing (byte-identical) |
| `record-{head,base}-120x45.json`, `states-{head,base}.json` | geometry, per-frame row/cell coverage, scroll offsets, md5s |
| `landing-frames-head.json` | every compositor frame one `pagedown` writes on the fixed tree (1), with render ms |
| `row-verify-head.json` | painted body row vs the body's composed line at `scroll_y + i` for every captured frame |

## Hashes computed in this round (md5)

| frame | md5 |
| --- | --- |
| head pagedown-first == head pagedown-settled == base pagedown-settled == author's `before-step-05.svg` == author's `after-step-01.svg` | `03aac32d31f220d8080d37cd83e431e2` |
| head start == head home == base start == base home | `f333950999f147816da2c9288751c495` |
| head end == base end | `be45e14d28b7ddc881384e0c9ed305c9` |
| head 100x35 resize == base 100x35 resize | `73d8c40ba2a6bce399a013e927ac169b` |
| head 60x24 pagedown == base 60x24 pagedown | `7442b8472f23…` |
| head 60x24 end == base 60x24 end | `8782b4ab1248…` |
| wheel x10 landed frame on head == on base | `18d01e0d694d207f4c75cbcfc78d4d5c` |
| base mid-easing first frame (`before-first-frame-pagedown.svg`) == author's `before-step-01.svg` | `222def9928d49bc7daa79638a2dd6e21` |

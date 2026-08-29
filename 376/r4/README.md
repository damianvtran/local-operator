# Design review round 4 — PR #376, head `70b3e878`

Rendered evidence for the round-4 design review. Captured from the **real
`OperatorApp`** under `run_test` (so `local_operator.tcss` applies), per
`AGENTS.md` "Visual validation". Every frame was rendered to PNG and looked at
before any claim in the review comment was made.

Capture: `capture-frames-r4.py`, run as

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python capture-frames-r4.py <outdir>
```

Geometry and buffer facts behind every frame: `measurements-r4.json`.
`frames/` holds the Textual SVG exports, `png/` the rasterised versions
(`qlmanage -t -s 1400`) that were actually viewed.

## D12 — the two-route contradiction

The same valid PNG, delivered three ways. Route A is clipboard image bytes
(`Cmd+V` on a screenshot), route B is a Finder `Cmd+C` file URL (`contents.paths`,
the route D12 was raised against), route C is a path in the paste text.

| frame | route |
| --- | --- |
| `d12-20.3MB-A-clipboard-bytes` / `-B-finder-fileurl` / `-C-path-text` | 20.29 MB PNG (21,275,898 B) |
| `d12-8.6MB-A-clipboard-bytes` / `-B-finder-fileurl` / `-C-path-text` | 8.60 MB PNG (9,018,045 B) |

All three PNGs per size are **byte-identical files** (sha256 of the rasterised
frame), marker `[Image #1, 1568x1467 ↓]` and `[Image #1, 1568x1352 ↓]`
respectively, attached payloads equal, no notice shown.

## D12 — the copy

Every failure that still reaches the `unreadable` branch, each rendered:
`d12-copy-non-image`, `-heic`, `-mixed`, `-missing`, `-over` (over the 64 MB
ingest ceiling). All five paint the same card, which is the point: the sentence
no longer asserts a cause it cannot know. Narrow widths at
`d12-copy-unreadable-60x24` and `-30x18`.

## D13 — a showing notice withdrawn by the paste that answers it

`d13-1-notice-showing` → `d13-2-after-attach-settled` → `d13-3-second-consecutive-frame`.
No card above a populated composer, and the second consecutive frame is
pixel-identical to the first (no reflow).

`d3-1-mcp-owns-slot` → `d3-2-notice-deferred` → `d3-3-attached` →
`d3-4-after-expiry` re-checks that the `withdraw` swap did not regress D3/D8:
the held card is still retired and never surfaces when the MCP card expires.

## Frame invariants

`inv-{100x30,60x24,30x18}-{first,settled}`: first paint after the attach lands
versus settled. Each pair is pixel-identical, and no frame overflows or raises a
scrollbar.

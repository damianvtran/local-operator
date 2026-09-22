# Terminal capture fidelity

A screenshot has **three independent sizes**: Textual's columns/rows, the
export's pixels per cell, and the viewer's display scale. State all three.
Never change live TUI CSS to compensate for a screenshot viewer's thumbnail.

## Native capture

All current `scripts/*shot.py`, `shot_login.py`, `ask_user_repro.py`,
`theme_preview.py` and `steer_receipt_probe.py` use `scripts.visual_capture`.
Their existing positional CLIs are unchanged. The subsequently integrated
`eager_boot_shot.py` is a real-provider bootstrap probe: explicit `--live` retains
its configured-provider/MCP purpose, while `--isolated` captures real
unconfigured boot safely and is the only mode included in the gallery. Live
provider binding is manual opt-in, not part of the offline coverage claim.
Historical PR evidence remains available through its original commit/PR links;
new generated evidence belongs on the PR, not in this repository.
The app's public `save_screenshot` / `export_screenshot` APIs are unchanged.

```sh
.venv/bin/python scripts/ask_shot.py /tmp/ask.svg 100x30
rsvg-convert /tmp/ask.svg -o /tmp/ask.png
.venv/bin/python scripts/pages_shot.py /tmp/welcome.svg welcome 158x44 radient
.venv/bin/python scripts/pages_shot.py /tmp/specimen.svg specimen 100x30 radient
.venv/bin/python scripts/visual_gallery.py --list
.venv/bin/python scripts/visual_gallery.py /tmp/gallery
# Fast, selected repeats use exact IDs from --list:
.venv/bin/python scripts/visual_gallery.py /tmp/gallery-repeat --case page-specimen
```

Use the worktree's own editable venv. No browser engine is installed or driven
by these tools. Native PNG conversion uses optional `rsvg-convert` (librsvg).
The gallery fails clearly if it is absent; `--svg-only` explicitly records that
raster validation was skipped. Images in the generated HTML index are navigation
thumbnails; click through to native PNGs before assessing typography. A manifest
`PASS` proves script/export/raster execution, **not visual approval**. Each
artifact starts `NOT_INSPECTED`; QA/design record what they actually viewed.

The committed `scripts/visual_inventory.json` freezes the source census and
page/state boundaries. `visual_gallery.py --list` is the executable matrix:
111 documented legacy variants plus the new isolated bootstrap probe, all
registered palettes, the missing page families, and representative sizes rather
than a Cartesian product. A script
may emit several frames in one invocation. Existing script nominal dimensions
remain authoritative. Empty todo/wake sidebars deliberately collapse; no fixture
forces them open to manufacture a page. Providers, jobs and analytics use
synthetic data in the **real OperatorApp** with production CSS, not CSS-less
unit-test hosts. They validate rendering, not live provider authentication.

Two additions to that matrix are worth naming here, because each was a state a
review round had to rebuild a rig to look at (design round 4, D26/D29):

- `sidebar_shot.py silent` and `silent-two` — a peer that answered nothing, so
  its whole section is the heading `⇄ <device> (unreachable)` with no rows under
  it. Neither is a fixture the row producer can express, because the state is
  precisely the ABSENCE of rows: the script calls `set_silent_peers` itself and
  refuses to write a frame whose headings are missing or painted below
  `⌥ Subagent Runs`. Capture at **100x45** — at 100x30 the frame stops before the
  subagent tier, and the tier is half of what the case asserts.
- `mesh_sidebar_shot.py` — the ONE capture built from a real mesh (two config
  roots, two identities, two relays on loopback, a live link) rather than from
  hand-stamped rows, which is what makes a row-producer regression fail here
  instead of shipping (`Q-R10-1`). It drives `sidebar_shot.py peers-focus`, so
  the census rasterizes it at that script's own nominal 800x510 (a 100x30 grid at
  the 8x17 preset). THAT IS NOT THE PIXEL SIZE OF THE COMMITTED PNG, and reading
  the two as one size is what design round 5's D33 caught — see "A fresh capture
  against a committed PNG" below, which is the only place the committed pixels
  are stated. It also refuses a frame whose model chip is still `connecting…`,
  the mid-connect transient one run in three produced (D30).

Both `silent` cases and the mesh case are in the committed inventory, so the
next round re-derives them with `visual_gallery.py --case …` instead of writing
another rig. Pairing in the mesh case is written into both stores rather than
negotiated (no pty, no second human), so it is evidence about the transport and
the rendering, never about the pairing ceremony.

### A fresh capture against a committed PNG: state the zoom, or the check lies

The census and the committed `static/*.png` artifacts are **the same frame at two
different scales**, and nothing used to say so (design round 5, D33).
`visual_gallery.py` rasterizes at native size — `rsvg-convert OUT.svg -o OUT.png`,
no zoom — so a 100x30 case writes 800x510 while the artifact it is compared
against is bigger:

| committed artifact | census grid | census raster | committed pixels | zoom |
| --- | --- | --- | --- | --- |
| `tui-mesh-sidebar.png` | 100x30 | 800x510 | 1440x918 | 1.8 |
| `tui-mesh-network.png` | 100x30 | 800x510 | 1440x918 | 1.8 |
| `tui-mesh-picker.png` | 110x34 | 880x578 | 1584x1041 | 1.8 (height rounded up) |

1440 px is also exactly twice the 720 px the sidebar's README `<img>` pins, so
that figure lands on one image pixel per device pixel on a 2x display — the motive
`credential_readme_shot.py` documents for its own two frames. The picker's 1584 px
is nothing so neat: it is 1.8x its own 880 px grid, and the README pins that figure
at 620.

So a comparison must normalize the scale FIRST. Rendering the fresh SVG at the
artifact's own zoom is the cheap half: `rsvg-convert -z 1.8 OUT.svg -o OUT.png`
then diff against the committed PNG, which for the picker is exact (verified
2026-09-22: AE=0). Skipping it does not produce a small difference — it produces
a difference that is entirely about scale. Measured on the picker's own pair, with
the same bytes at the same zoom on both sides: upscaling the 880x578 census raster
to the artifact's 1584x1041 and diffing reports **3.1% of pixels differing where
the honest answer is zero**. The height is a `ceil` (110x34 at 1.8 is 1040.4), so
the two images do not even share an integer ratio on both axes; a check that
assumes one is comparing like with like is measuring the rescale filter, not the
app.

`tests/unit/tui/test_visual_gallery.py` pins the table above against the actual
committed PNGs' pixel dimensions, so a re-shoot at another zoom fails there and in
the `README` at once instead of quietly invalidating every comparison made off
these frames.

A CENSUS THAT CANNOT FIRE IS WORSE THAN NO CENSUS, and this export makes that easy
to write (design round 5, D32). The check that pins the splash's update row was
first written as `"latest is v" in exported`, which is FALSE on a frame that paints
the row, for two independent reasons: `terminal_svg` gives every grapheme cluster
its own `<tspan>`, and the spaces it writes are U+00A0 — the row reassembles to
`'!\xa0latest\xa0is\xa0v0.62.2\xa0—\xa0/update\n'`. Both censuses now read rows
through `visual_capture.svg_text_runs_by_row`, which groups runs by baseline and
folds no-break space to a plain space, and
`tests/unit/tui/test_visual_capture.py` pins both halves on a real export so the
next census author inherits the parse instead of the trap.

ONE MORE MACHINE VALUE IS IN THESE FRAMES, and it is recorded rather than fixed:
the welcome splash prints its CWD, so the directory the probe is launched from is
part of the picture — `static/tui-mesh-picker.png` carries the worktree root
(`/Users/damian/local-operator-worktrees/mesh-network`), which is why the AE=0
above is byte-exact only from the repository root. Measured 2026-09-22 by
comparing the committed PNG against a capture from another directory, per row at
the artifact's zoom: grid rows 15-21 (the version, model and cwd rows, the blank
row and the keymap rows) each move left by 159 px — 419 to 260 — and the cwd row
is the widest of them, which is why the block re-centers as a whole; grid row 31,
the composer's status band, grows with it (the band prints the cwd's last
component: `⌂ before` where the committed frame reaches `⌂ mesh-network`, so the
band's right edge sits at x=588 instead of 1237). Everything between — the
picker's own rows — is byte-identical (rows 23, 26-29 share their exact x
extents). The page fixture already pins an isolated `~` cwd for the same reason
(see the reference sizes below); pinning this one would change a shipped README
figure, so it is written down here and left to a round that means to re-shoot it.

## What changed

Textual exports the real compositor through Rich. Rich's default SVG is a
presentation with 20px Fira Code, 12.2x24.4px cells, a webfont URL, a 9px/41px
content offset and decorative window chrome. librsvg does not implement its
`textLength` layout hint consistently; adjacent styled runs can join words.

The capture helper preserves compositor content and cell layout, removes only
the presentation chrome, and projects coordinates into an explicit native cell
grid. Grapheme clusters get independent origins: ASCII spaces remain real
cells, CJK takes Rich's measured cell width, and combining/ZWJ sequences remain
whole for font shaping. No font is stretched horizontally, and no network
webfont is requested. Unsupported upstream SVG structure fails loudly rather
than silently producing believable incorrect measurements.

The default is an **explicit reproducible preset**, not an emulator default:

- 8x17 pixels per terminal cell, 13px font; cell aspect 0.470588.
- `Menlo, DejaVu Sans Mono, monospace`: macOS system-standard Menlo first,
  a common Linux monospace fallback second. No Fira Code dependency.
- Native 100x30 is exactly 800x510 pixels, with no capture padding or chrome.
- The `.geometry.json` sidecar records grid/native dimensions, CSS source,
  screen and widget content/virtual geometry and scrollbar state.
- Font provenance queries fontconfig (the librsvg path), including regular,
  bold and italic face selection. When Pillow is available it also records
  measured `i`, `W`, `0`, `1` advances and ascent/descent. Missing query tools or
  fallback are explicitly labelled, never reported as the requested font.
  Browser font resolution may differ. Emoji/Nerd glyph fallback is not an exact
  terminal-font match; inspect the specimen and report missing glyphs honestly.

Custom measured terminal settings can be supplied without changing layout:

```sh
LOP_CAPTURE_CELL_WIDTH=8 LOP_CAPTURE_CELL_HEIGHT=17 \
LOP_CAPTURE_FONT_SIZE=13 LOP_CAPTURE_FONT_FAMILY='Menlo, monospace' \
.venv/bin/python scripts/ask_shot.py /tmp/calibrated.svg 100x30
```

Measure actual cell pitch at the terminal's current font/zoom first. Positive,
finite dimensions are required; font size must not exceed cell height. Font
fallback and glyph coverage remain rasterizer concerns, not evidence to resize
the app. Every offline sample establishes a temporary HOME **and** config root before
app imports so themes, caches, auth and approval policy do not leak from the
operator. The separately labelled `eager_boot_shot.py --live` intentionally uses
real configuration and must not be run automatically or without that explicit
operator choice. Settings samples use that same root for their explicit config fixtures.

## Comparing the supplied terminal references

The original condensed image used a 150x34 pilot, Rich SVG, `rsvg-convert -w
1400`, then approximately 474x226 display inside a tool result. Its cells are
therefore much smaller on screen than their native SVG sizes. This is primarily
viewport/display scale, not a product spacing failure.

The estimated reference grids are **158x44** (Terminal.app) and **208x54**
(Ghostty). These were inferred from composer bounds, not measured emulator
settings. At the 8x17 preset they export 1264x748 and 1664x918 pixels. A uniform
1024px-wide comparison produces approximately 1024x606 and 1024x565. Preserve
native images alongside these comparison previews:

```sh
rsvg-convert -w 1024 /tmp/welcome.svg -o /tmp/welcome-fit1024.png
```

The matching 100-cell composer widths/heights are approximately 648x69 and
492x52 in those previews, close to the supplied 650x69 and 490x52 estimates.
The row count, model label, font, zoom and original terminal chrome are not
known exactly, so vertical position and rasterization are not pixel-exact
claims. `radient` is the explicit cool reference-comparison palette
(background #090d13); the shipped `dark` palette intentionally uses warm brown.
The page fixture waits for boot, refreshes welcome info and uses `~` as its
isolated cwd so random temporary paths do not change centering.

## Before, after and consecutive frames

Capture before editing. The helper does not modify the app, so the same running
pilot can also call `app.save_screenshot` for a legacy comparison. Use
`save_capture(app, path)` for the faithful frame, render both, and actually look
at them. `pages_shot.py` emits `.first.svg` and a settled `.svg`; settings and
steer scripts retain their existing consecutive-frame variants. Compare widget
regions/virtual sizes between frames, not timestamps or animation text. A
first/settled comparison is only evidence of the captured interval, not proof
that every animation or network transition is stable.

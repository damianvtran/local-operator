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

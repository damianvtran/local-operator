# Visual fidelity evidence

Implementation base: `619ec60a52efc11e3257d5439ace31dd4bcec22f`.
Capture implementation checkpoint: `a8b03393`.
No files in the runtime package changed. The live TUI and inherited Textual
screenshot methods are unchanged; only developer capture tools adopted the new
export and isolation helper.

## Reproduction and measured correction

`ask-before.png` was captured **before editing** with the existing
`scripts/ask_shot.py` at 100x30, then `rsvg-convert` at native size. Native
1238x782 includes Rich's 9,41 content offset/window chrome and 12.2x24.4 cells.
The tool viewer reduced it to 1024x647. `ask-after.png` uses the same script/grid
through the helper, native800x510, 8x17cells, no window chrome. Its spaces remain
cells across styled runs: legacy `3of` / `enteranswer` become `3 of` /
`enter answer`. These images were rendered and viewed, not assessed from SVG
markup.

The system font is Menlo13px on this macOS host. Fontconfig resolved
`/System/Library/Fonts/Menlo.ttc`, regular index0, bold1, italic2.
Pillow/FreeType measured `i/W/0/1` advances8px for each style, ascent13/descent4.
`specimen.geometry.json` retains the requested/full resolved family chain and
measurements. Explicitly unavailable families fall back to a system monospace
when the generic chain is present; a resolved variable-width custom face is
flagged. Font rendering is not pixel-exact terminal emulation: in particular
emoji silhouettes and Nerd fallback may differ. The native specimen was viewed;
cluster follower positions are regression-tested separately.

Reference previews use explicitly selected `radient`, settled `test/model`,
normalized isolated HOME (`~`), and **estimated**, not measured, terminal grids:

| Preview | Grid | Native pixels | Uniform comparison | Composer estimate |
| --- | --- | --- | --- | --- |
| `terminal-reference-preview.png` |158x44|1264x748|1024x606|648x69 vs supplied650x69|
| `ghostty-reference-preview.png` |208x54|1664x918|1024x565|492x53 vs supplied490x52|

The original terminal fonts, zoom, viewport and chrome are unknown. The parent
manager alone had the original embedded images; the implementing agent viewed
these generated frames and used the supplied numeric context, not a claim of
having independently inspected the originals. Vertical offsets are retained,
not “corrected” by changing product layout.

## Complete finite gallery

Executed:

```sh
.venv/bin/python scripts/visual_gallery.py /tmp/lop-visual-gallery-v2
# 152 PASS, zero FAIL; 257 native SVG/PNG pairs plus geometry sidecars
```

152 invocations comprise111documented legacy variants across23scripts,
30missing-page states and11reference/responsive repeats. The one theme
invocation captures all54registered palettes. QA's separate112legacy matrix
includes an additional fork-before control; it is not this census count.
`manifest.json` is the compact durable command/artifact/geometry record; the
full local gallery is regenerable and is deliberately not committed.

Viewed all54palette thumbnails, all30page-state thumbnails, legacy/script
contact sheets, and full native reference welcome, specimen, narrow ask,
narrow settings and light transcript frames. Contact sheets check composition,
not character-level readability; representative native frames plus measured
coordinates and tests provide the typography evidence. `NOT_INSPECTED` is the
runner's initial state, not implied approval from a successful process exit.

`settling.json`:30/30new page first/settled pairs retain identical screen and
widget geometry/virtual sizes/scrollbar state. No outer screen scrollbar.
This covers the interval captured, not arbitrary network or animation timing.
Existing settings and steer scripts also emit their native paired variants.

**Known fixture gap (D1, non-gating):** `transcript-loading` is an unfinished
AssistantBlock but does not expose a visible loading cue. Do not count it as
visible loading evidence. `usage-loading` visibly says fetching; `aside-loading`
visibly says thinking. Those exercise the required visible loading transport.
The gap is deferred rather than expanding this capture-only change into live
thinking-indicator behaviour.

## Source provenance and final replay

The complete v2 gallery ran with the cluster-safe projection. During its early
cases the subsequent changes were SVG-extension validation, generic-monospace
validation, font measurement metadata/warnings, and a fixture job lifetime type
guard; no pixel-coordinate or visual content change. Accordingly it is **not**
claimed that every v2 subprocess imported byte-identical final source.

Nine representative invocations were then replayed from committed `a8b03393`:
specimen; resume-empty; usage-error; todo-overflow; both reference welcome grids;
80x24ask and settings; 100x30light transcript. All9PASS with native PNGs,
sidecars and consecutive frames where supported. The committed specimen and
reference previews come from that replay. Source hashes:

```
303acec63d875d8ccf6c3cfd836bcef123e4a87876dcb76706ab4135a6c73ec3 scripts/visual_capture.py
c455fe1b566acef44a228a20633182764685a2c2c5f4a4f251e72f2c570a16d6 scripts/pages_shot.py
96e763b1532ac18927aa71ebba0ca4fbf4d79fc365292b0ad39f918ec00462c7 scripts/visual_gallery.py
```

## Gates at the capture checkpoint

- Whole tree flake8:0.
- Whole tree pinned Black26.1.0:876files unchanged,0.
- Whole tree isort5.13.2:0.
- Whole tree pyright:0errors,0warnings.
- Focused capture/CLI tests:22passed (including CSS-less host and wrong-extension
  rejection before output writes).
- Assembled app e2e:9passed in16.09s.
- The initial whole-unit run inherited daemon-only `LOP_RUNTIME_ADOPT_SESSION=1`
  and `LOP_RUNTIME_DEFER_MATERIALISE=1`;18resume assertions failed. A standalone
  file repro produced18failed/93passed, still18failed with fresh HOME/config.
  Unsetting only those two flags produced111passed. The suite's new file-local
  autouse fixture scrubs them; explicitly setting both flags again now produces
  **111passed in5.37s**. This is test isolation, no product change.
- Whole unit summary and the final integration/version result are recorded in
  the PR, not guessed here while a long run is in progress.

All app captures used isolated HOME/config. No live sessions, browser engines,
provider credentials, or user settings were touched.

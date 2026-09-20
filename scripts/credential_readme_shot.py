"""Re-shoot the two README figures for the inline ``/credential`` gesture.

Run from the worktree root — the three profile variables are not optional:

    env -u NO_COLOR TERM=xterm-256color \
        LOP_CAPTURE_CELL_WIDTH=16 LOP_CAPTURE_CELL_HEIGHT=34 LOP_CAPTURE_FONT_SIZE=26 \
        .venv/bin/python \
        scripts/credential_readme_shot.py OUTDIR [COLSxROWS] [THEME]

Without the profile this dies after the ~60 s boot with ``90 cells at 8 px is
720 px, not 2x the README's 720 px pin``: ``BAND_CELLS`` below is 2x the
README's 720 px pin only at the 16 px cell of the 2x profile, and the
environment's defaults are the 8x17 / 13 px gallery preset. The check runs
after the boot rather than before it, so the omission is an expensive one to
discover.

Then rasterize and install (this script never writes into ``static/``):

    rsvg-convert OUTDIR/credential-chip.svg -o static/tui-credential-chip.png
    rsvg-convert OUTDIR/credential-masked-typing.svg \
        -o static/tui-credential-masked-typing.png

WHY THIS IS A SCRIPT AND NOT A ONE-OFF. These two figures are the README's
only pictures of the composer, so they are re-shot every time the composer's
chrome, the chip's ink, or the masked-typing notice changes — and the merged
pair was the operator's own capture, cropped but never re-shootable, which is
exactly how they ended up as the page's two softest figures (design round 1,
D1 on PR #1295: 1x sources whose glyph stems peak at ~70 % of the foreground
colour). Rendering them from the real ``OperatorApp`` makes the next re-shoot
one command.

WHAT IT IS: the two states the README's prose describes, driven through the
composer's own keys — the description typed first, ``/credential`` and a space
to arm, then either a bracketed paste (which mints the chip) or ten typed
characters (which stay masked). The chip frame is captured with the chip at
REST — the state a reader's own paste leaves behind — and never clicked: the
alt text calls it "a **collapsed** [Credential #1, 15 chars] chip" and no page
prose mentions clicking one, while a press inside the marker repaints it under
the ``-selected`` rule (``$lo-fg`` on ``$lo-tint-attach-hi``), a state the page
never introduces. The frame is therefore the paste's own result rather than a
state this script paints, and the mint check in ``shoot`` fails the run if no
chip reached the composer at all.

WHAT IT IS NOT: a general credential coverage script. The armed / held /
disarmed states, the destructive key sequences and the store assertions live
in ``scripts/credential_armed_shot.py``; this file exists only for the two
frames the README embeds, and it deliberately does not grow a second dialect
for driving the feature.

THEME: ``tron`` by default, which is also the palette of the merged pair this
replaces — measured, not assumed. Those two files carry Apple's "Display" ICC
profile, and read through their own profile their ground ``(14, 20, 31)`` is
sRGB ``(12, 20, 32)`` = tron's ``surface``, and their chip ground
``(23, 35, 55)`` is ``(19, 35, 57)`` = tron's ``tint-attach``. Reading those
same bytes as untagged sRGB is what makes them look like a different palette:
no registered palette carries ``#0e141f`` as ``surface``, because ``#0e141f``
is not an sRGB value at all — it is ``#0c1420`` in that profile's encoding.
This capture is untagged sRGB, so its bytes are tron's tokens directly and a
colour-managed page paints both at the same colour. Pass THEME to compare
palettes deliberately.

ISOLATION AND THE SECRET: every inherited ``CMUX_*`` variable is dropped and
``probe_isolation`` re-homes HOME and the config root before any application
import, so a headless run can neither rename the operator's cmux workspaces
nor read their config. The value pasted in the chip frame is a synthetic
15-character placeholder, and the script refuses to finish if its rendered
frame contains anything but the collapsed marker — the assertion that makes
"the frame shows a chip" a measurement rather than a claim.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = "http://www.w3.org/2000/svg"

# Drop every multiplexer id BEFORE any application import: an inherited live
# CMUX_WORKSPACE_ID has let a headless pilot rename the operator's workspaces.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual import events  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import CaptureProfile, save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

#: Synthetic, obviously-not-a-key, and exactly 15 characters so the chip
#: reports the length the README's alt text quotes ([Credential #1, 15 chars]).
#: A real key here would be published to the repository by this script.
SECRET = "sk-or-v1-" + "0" * 6

#: Typed into the masked frame. Its length is what the frame's bullet count
#: shows; the VALUE never reaches the screen, only one bullet per character.
TYPED_SECRET = "hunter2-42"

DESCRIPTION_KEY = "Here's my OpenRouter key, use it to test the integration: "
DESCRIPTION_PASSWORD = "Here's my password for e2e testing: "

THEME = "tron"

#: The crop covers the composer shell's three cell rows (one cell of padding,
#: the line, one cell of padding) and, in the masked frame, the notice row the
#: picker paints beneath it. Both are grid-aligned: a crop that lands mid-cell
#: clips the chip's own background box, which is how a "tight" crop turns into
#: a sliced glyph.
#:
#: 90 cells is not a taste decision. At the 16x34 px profile of a 2x re-shoot it
#: is exactly 1440 px, which is exactly 2x the 720 px the README pins — so the
#: page paints one image pixel per device pixel on a 2x display instead of
#: upscaling, which is the whole point of re-shooting these two figures. The
#: band's own content (the panel's padding cell, the chevron pair, the
#: gutter, the draft and the caret cell) measures 88 cells here, so the pin
#: decides the width and the content has to fit — asserted rather than assumed.
BAND_CELLS = 90

#: What the README pins both figures to. Kept here so the width above can be
#: checked against the one number it is derived from.
README_PIN_PX = 720


def composer_band(app, editor) -> tuple[int, int, int]:
    """``(column, row, rows)`` of the composer's own band, in cells.

    The band is the panel's padding row, the line (plus the picker's notice row
    when one is painted) and the panel's trailing padding row. Derived from the
    widgets rather than written down: the dock also owns the status band, so
    its own region is taller than the composer and would put the model line in
    the figure.
    """
    chevron = app.query_one("#prompt-chevron")
    picker = getattr(editor, "picker", None)
    last = editor.region.y
    if picker is not None and picker.display and picker.region.height > 0:
        last = max(last, picker.region.y + picker.region.height - 1)
    top = editor.region.y - 1
    return chevron.region.x - 1, top, last - top + 2


def row_band_top(svg: Path, row: int) -> float:
    """The painted top of screen row ``row``, read from the SVG's own clip.

    Rich paints each row inside a clip rect whose geometry IS the row's line
    box, and the projection scales that box without re-rounding it — so a crop
    taken at ``row * cell_height`` sits a couple of pixels above the band and
    leaves a sliver of the previous row's ground along the figure's top edge.
    Taking the origin from the clip rect keeps the crop on the boundary the
    renderer actually paints, and it fails loudly rather than guessing if the
    clip is ever absent.
    """
    root = ET.parse(svg).getroot()
    for clip in root.iter(f"{{{_NS}}}clipPath"):
        if not clip.get("id", "").endswith(f"-line-{row}"):
            continue
        rect = clip.find(f"{{{_NS}}}rect")
        if rect is None:
            break
        value = rect.get("y")
        if value is None:
            break
        return float(value)
    raise ValueError(f"{svg.name}: no clip rect for row {row}")


def crop_svg(
    source: Path, destination: Path, *, x: float, y: float, width: int, height: int
) -> dict[str, float]:
    """Write ``source`` cropped to a pixel rect, at the same 1:1 pixel scale.

    A viewBox crop rather than a rasterize-then-cut: the SVG already carries
    every glyph at the capture profile's pixel geometry, so narrowing the
    window renders the same pixels the full frame would have, with no second
    rasterization pass and no dependency on an image library. The origin may
    be fractional (see :func:`row_band_top`); the EXTENT may not, because a
    fractional extent is what would make librsvg resample the whole figure.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("width", str(width))
    root.set("height", str(height))
    root.set("viewBox", f"{x:g} {y:g} {width} {height}")
    tree.write(destination, encoding="unicode", xml_declaration=False)
    return {"x": x, "y": y, "width": width, "height": height}


async def type_text(pilot, text: str) -> None:
    for char in text:
        await pilot.press(char)
    for _ in range(3):
        await pilot.pause()


def painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def shoot(outdir: Path, size: tuple[int, int], theme: str) -> dict[str, dict[str, object]]:
    """Boot once per state, drive it, and write the full frame and its crop."""
    results: dict[str, dict[str, object]] = {}
    for name in ("credential-chip", "credential-masked-typing"):
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await _boot(pilot, app)
            app._apply_theme(theme)
            # The splash centres the composer in a narrow card; the README's
            # frames are of a STARTED conversation, where the composer runs the
            # full width of the dock. Dropping the splash is what makes the
            # chip frame's left margin the two cells the merged figure has
            # rather than the card's twenty.
            app._set_welcome_visible(False)
            editor = app._editor()
            editor.focus()
            await type_text(pilot, "The composer")  # settle the box before the draft
            editor.text = ""
            for _ in range(2):
                await pilot.pause()

            if name == "credential-chip":
                await type_text(pilot, DESCRIPTION_KEY)
                await type_text(pilot, "/credential ")
                app.post_message(events.Paste(SECRET))
                for _ in range(4):
                    await pilot.pause()
                # The chip is NOT clicked. This figure is the paste's own
                # result — the state a reader reaches, and the state the alt
                # text ("a **collapsed** … chip") and the caption describe. A
                # press inside the marker repaints it under the -selected rule
                # ($lo-fg on $lo-tint-attach-hi), which no page prose
                # introduces; the merged figure measured the resting tokens, so
                # selecting here was the capture drifting, not the subject
                # changing. The check below is what keeps "there is a chip in
                # this frame" a measurement rather than a claim.
                if not any(span[2] is False for span in editor._marker_cells(0)):
                    raise RuntimeError("the pasted secret did not mint a chip")
            else:
                await type_text(pilot, DESCRIPTION_PASSWORD)
                await type_text(pilot, "/credential ")
                await type_text(pilot, TYPED_SECRET)
                for _ in range(4):
                    await pilot.pause()

            frame = painted(app)
            if SECRET in frame:
                raise RuntimeError("the synthetic secret reached the screen as plaintext")
            if name == "credential-masked-typing" and TYPED_SECRET in frame:
                raise RuntimeError("the typed secret reached the screen unmasked")
            # README.md:928's alt text asserts this notice is on the frame, so a
            # frame without it would publish a figure whose caption is false.
            if name == "credential-masked-typing" and "masked as you type" not in frame:
                raise RuntimeError(
                    "the masked frame lost the 'masked as you type' notice README.md:928 asserts"
                )

            full = outdir / f"{name}.full.svg"
            save_capture(app, full)
            band_x, band_row, band_rows = composer_band(app, editor)
            profile = CaptureProfile.from_env()
            native_width = int(BAND_CELLS * profile.cell_width)
            if native_width != 2 * README_PIN_PX:
                raise ValueError(
                    f"{BAND_CELLS} cells at {profile.cell_width:g} px is {native_width} px, "
                    f"not 2x the README's {README_PIN_PX} px pin"
                )
            # Annotated rather than inferred: ``editor.text = ""`` above narrows the
            # attribute to the empty literal, and the analyzer then treats the
            # width check below as unreachable code.
            draft: str = editor.text
            widest = max((len(line) for line in draft.splitlines()), default=0)
            # chevron pair + gutter + the widest line + the caret's own cell.
            content_cells = 2 + 1 + widest + 1
            if content_cells > BAND_CELLS:
                raise ValueError(f"the draft needs {content_cells} cells; the band is {BAND_CELLS}")
            crop = crop_svg(
                full,
                outdir / f"{name}.svg",
                x=band_x * profile.cell_width,
                y=row_band_top(full, band_row),
                width=native_width,
                height=int(band_rows * profile.cell_height),
            )
            results[name] = {
                "draft": editor.text,
                "grid": list(app.size),
                "profile": {
                    "cell_width": profile.cell_width,
                    "cell_height": profile.cell_height,
                    "font_size": profile.font_size,
                    "font_family": profile.font_family,
                },
                "band": {"cell": band_x, "row": band_row, "rows": band_rows},
                "crop": crop,
                "full": str(full),
            }
            print(f"\n=== {name} ===")
            print(f"  draft:        {editor.text!r}")
            print(f"  editor:       {tuple(editor.region)} gutter={editor.gutter}")
            print(f"  band:         cell {band_x}, row {band_row}, {band_rows} rows")
            print(f"  crop:         {crop} (native {native_width}x{crop['height']})")
            print(f"  notice shown: {'masked as you type' in frame}")
            print(f"  grid:         {tuple(app.size)}")
            (outdir / f"{name}.json").write_text(json.dumps(results[name], indent=2) + "\n")
    return results


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/credential-readme")
    size = (120, 36)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    theme = sys.argv[3] if len(sys.argv) > 3 else THEME
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"profile: {CaptureProfile.from_env()}")
    asyncio.run(shoot(outdir, size, theme))
    print("\nrasterize at native size; -w would resample the figure:")
    for name, target in (
        ("credential-chip", "tui-credential-chip.png"),
        ("credential-masked-typing", "tui-credential-masked-typing.png"),
    ):
        print(f"  rsvg-convert {outdir / f'{name}.svg'} -o static/{target}")


if __name__ == "__main__":
    main()

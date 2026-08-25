#!/usr/bin/env python3
"""Derive the extension's toolbar icons from the repo's product mark.

Usage: python extension/scripts/generate-icons.py [--check]

The source PNGs (static/local-operator-icon-2-*) are design artifacts far
larger than any toolbar slot; the store requires exact 16/32/48/128 squares.
Checked-in outputs keep the extension buildable without Pillow present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "static" / "local-operator-icon-2-light-clear.png"
TARGETS = {
    size: REPO_ROOT / "extension" / "icons" / f"icon-{size}.png" for size in (16, 32, 48, 128)
}


#: Brand-safe backplate colour. The shipped mark is a black glyph on
#: transparency, which vanishes on dark toolbar/store chrome (finding D3). A
#: light rounded backplate keeps the glyph legible on BOTH light and dark
#: surfaces without inventing a new brand colour — it is the same off-white the
#: popup header sits on.
BACKPLATE = (247, 247, 249, 255)
#: Corner radius and inset as fractions of the icon edge, so the rounded plate
#: and glyph padding scale identically at 16/32/48/128.
RADIUS_FRACTION = 0.22
INSET_FRACTION = 0.16


def _rounded_plate(size: int):
    """An opaque rounded-square backplate at the target size."""
    from PIL import Image, ImageDraw

    plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    radius = max(2, round(size * RADIUS_FRACTION))
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    plate.paste(Image.new("RGBA", (size, size), BACKPLATE), (0, 0), mask)
    return plate


def render(*, write: bool = True) -> dict[int, bytes]:
    """Composite the square logo mark onto a rounded backplate at each size."""
    from PIL import Image

    with Image.open(SOURCE) as opened:
        rgba = opened.convert("RGBA")
        alpha_box = rgba.getchannel("A").getbbox()
        if alpha_box is None:
            raise ValueError(f"{SOURCE} has no visible pixels")
        left, top, right, bottom = alpha_box
        edge = max(right - left, bottom - top)
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        # The source is wide transparent marketing art; alpha bounds isolate
        # the mark without baking today's canvas proportions into the script.
        source = rgba.crop(
            (center_x - edge // 2, center_y - edge // 2, center_x + edge // 2, center_y + edge // 2)
        )
    rendered: dict[int, bytes] = {}
    TARGETS[16].parent.mkdir(parents=True, exist_ok=True)
    for size, path in TARGETS.items():
        import io

        plate = _rounded_plate(size)
        # Inset the glyph so it never touches the plate edge or the rounded
        # mask, keeping the figure clear at 16 px.
        inset = max(1, round(size * INSET_FRACTION))
        glyph_size = size - 2 * inset
        glyph = source.resize((glyph_size, glyph_size), Image.Resampling.LANCZOS)
        plate.paste(glyph, (inset, inset), glyph)
        buffer = io.BytesIO()
        plate.save(buffer, format="PNG", optimize=True)
        rendered[size] = buffer.getvalue()
        if write:
            path.write_bytes(rendered[size])
    return rendered


def check() -> int:
    """Regenerate to memory and compare against the checked-in bytes."""
    import hashlib

    rendered = render(write=False)
    for size, path in TARGETS.items():
        try:
            existing = path.read_bytes()
        except OSError:
            existing = b""
        if hashlib.sha256(existing).digest() != hashlib.sha256(rendered[size]).digest():
            print(f"{path} is stale; run extension/scripts/generate-icons.py", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    render()
    for path in TARGETS.values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

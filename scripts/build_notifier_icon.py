"""Generate the notifier app's ``.icns`` from the project's existing mark.

Run when ``static/local-operator-icon-2-*.png`` changes:

```sh
.venv/bin/python scripts/build_notifier_icon.py
```

The output (``local_operator/tui/notifier_app/icon.icns``) is COMMITTED, so a
user's machine never needs Pillow, ``iconutil`` or the source art to show the
right icon on a notification — the wheel ships the finished file. This script
exists so the icon can be regenerated from the one source of truth rather
than being an opaque binary nobody can rebuild.

Why a generated square rather than the PNG as-is: the shipped art is a
2048x750 wordmark canvas whose glyph occupies a 318x348 box in the middle, so
handing it to macOS directly produces a tiny mark adrift in a wide
transparent field. macOS also expects an app icon to be a filled shape at the
platform's corner radius; a bare transparent glyph renders as the generic
placeholder's cousin — technically present, visually wrong.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# macOS reads these exact names out of an iconset; anything else is ignored.
_SIZES = [
    (16, "icon_16x16.png"),
    (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"),
    (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"),
    (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"),
    (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_512x512@2x.png"),
]

#: The macOS "squircle" is ~22.37% of the side. Matching it keeps the icon
#: from looking like a foreign square among the system's own.
_RADIUS_RATIO = 0.2237

#: The glyph sits on a dark plate rather than transparency: a notification
#: banner is itself translucent, and a transparent icon reads as a hole.
_PLATE = (28, 32, 44, 255)


def build(repo_root: Path) -> Path:
    from PIL import Image, ImageDraw

    source = repo_root / "static" / "local-operator-icon-2-dark-clear.png"
    art = Image.open(source).convert("RGBA")
    glyph = art.crop(art.getbbox())  # trim the wordmark canvas to the mark

    out_dir = repo_root / "local_operator" / "tui" / "notifier_app"
    out_dir.mkdir(parents=True, exist_ok=True)
    icns = out_dir / "icon.icns"

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for size, name in _SIZES:
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            plate = Image.new("RGBA", (size, size), _PLATE)
            mask = Image.new("L", (size, size), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, size - 1, size - 1), radius=int(size * _RADIUS_RATIO), fill=255
            )
            canvas.paste(plate, (0, 0), mask)

            # The glyph keeps its aspect ratio inside a generous margin: the
            # mark is tall and thin, so fitting it to the full square would
            # crop it or stretch it out of shape.
            inner = int(size * 0.62)
            ratio = min(inner / glyph.width, inner / glyph.height)
            scaled = glyph.resize(
                (max(1, int(glyph.width * ratio)), max(1, int(glyph.height * ratio))),
                Image.LANCZOS,
            )
            canvas.paste(
                scaled,
                ((size - scaled.width) // 2, (size - scaled.height) // 2),
                scaled,
            )
            canvas.save(iconset / name)

        subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
            check=True,
        )
    return icns


if __name__ == "__main__":
    path = build(Path(__file__).resolve().parents[1])
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    sys.exit(0)

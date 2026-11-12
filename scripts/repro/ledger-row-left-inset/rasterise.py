"""Rasterise a Textual SVG capture without eating the spacing it proves.

`rsvg-convert` treats a run of whitespace in `xml:space="default"` the way the
XML spec says to — collapse it — and Rich's `<text>` nodes carry the row's
leading pad as literal spaces. The captures from `scripts/visual_capture.py`
already set `xml:space="preserve"` on every `<text>` node, but it is asserted
here rather than assumed: an SVG from any other path (or any future Rich
version) would otherwise rasterise with the indent collapsed, which is exactly
the spacing these frames exist to show. Cheap to check, silently wrong to skip.

Done by regex on the raw bytes, not via ElementTree: serialising `xml:space`
back out of stdlib ElementTree emits the attribute twice and librsvg then
refuses the file ("Attribute xml:space redefined"), which is a worse failure
than the one being guarded against.

Usage: rasterise.py IN.svg OUT.png [SCALE]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    svg = src.read_text(encoding="utf-8")
    stripped = re.sub(r'xml:space="preserve"', "", svg)
    if "<text" not in stripped:
        raise SystemExit(f"{src}: no <text> nodes")
    preserved = stripped.replace("<text", '<text xml:space="preserve"')
    prepared = src.with_suffix(".preserve.svg")
    prepared.write_text(preserved, encoding="utf-8")

    try:
        subprocess.run(
            ["rsvg-convert", "-z", str(scale), str(prepared), "-o", str(dst)],
            check=True,
        )
    finally:
        prepared.unlink()


main()

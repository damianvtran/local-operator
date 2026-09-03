"""Rasterise the committed SVG frames to PNG without collapsing their spacing.

The PNGs exist so the frames can be read inline on the PR, and the thing a
design reviewer looks at is spacing — so a rasteriser that eats it produces a
picture that invites bug reports about defects the terminal does not have
(`Usage1m ago`, `rrefresh`, `escclose` were all read off the first round's
PNGs; the SVGs were correct all along).

Textual writes each styled run as its own `<text>` node positioned by `x` plus
`textLength`, and encodes the gaps as leading `&#160;`. librsvg applies XML
whitespace collapsing to those runs unless the element opts out, which drops
the leading spaces and then stretches what remains to satisfy `textLength` —
every style boundary loses its gap. Adding `xml:space="preserve"` is the whole
fix; it is applied to a COPY so the committed SVGs stay byte-identical to what
Textual exported and remain the authoritative artifact.

    .venv/bin/python docs/evidence/usage-freshness/rasterise.py
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
WIDTH = "1200"


def main() -> int:
    if shutil.which("rsvg-convert") is None:
        print("rsvg-convert not found (brew install librsvg)", file=sys.stderr)
        return 1
    for svg in sorted(HERE.glob("*.svg")):
        source = svg.read_text()
        # Opt every text run out of whitespace collapsing; see the module docstring.
        patched = source.replace("<text ", '<text xml:space="preserve" ')
        with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as handle:
            handle.write(patched)
            temporary = handle.name
        try:
            subprocess.run(
                ["rsvg-convert", "-w", WIDTH, temporary, "-o", str(svg.with_suffix(".png"))],
                check=True,
            )
        finally:
            pathlib.Path(temporary).unlink(missing_ok=True)
        print(f"{svg.name} -> {svg.with_suffix('.png').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

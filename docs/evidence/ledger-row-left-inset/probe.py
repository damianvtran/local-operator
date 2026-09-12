"""Report, per painted line, where each glyph run starts — in CELL columns.

Rich's SVG puts one <tspan x=...> per grapheme cluster, so this reads the
exact cell the compositor was told to draw each glyph at. That is the
geometry behind the stills: the still shows the misalignment, these numbers
say which cell each field landed in.

Usage: probe.py FILE.svg [CELL_WIDTH]
"""

from __future__ import annotations

import sys
from xml.etree import ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"


def lines(path: str) -> list[tuple[float, list[tuple[float, str]]]]:
    root = ET.parse(path).getroot()
    rows: dict[float, list[tuple[float, str]]] = {}
    for node in root.iter(f"{NS}text"):
        y = float(node.get("y", "0"))
        for span in node.iter(f"{NS}tspan"):
            text = span.text or ""
            if not text.strip():
                continue
            rows.setdefault(y, []).append((float(span.get("x", "0")), text))
    out = []
    for y in sorted(rows):
        runs = sorted(rows[y])
        # Coalesce consecutive clusters of one field into a single run so the
        # report names the icon / name / summary rather than every letter.
        merged: list[list] = []
        for x, ch in runs:
            if merged and abs(merged[-1][1] - x) < 0.01:
                merged[-1][0] += ch
                merged[-1][1] = x + 8.0
            else:
                merged.append([ch, x + 8.0, x])
        out.append((y, [(m[2], m[0]) for m in merged]))
    return out


def main() -> None:
    path = sys.argv[1]
    cell = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    for y, runs in lines(path):
        parts = " | ".join(f"c{int(round(x / cell))}={txt!r}" for x, txt in runs)
        print(f"y={y:8.3f}  {parts}")


main()

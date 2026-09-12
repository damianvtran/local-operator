"""Parse a visual_capture SVG into the rows a reader actually sees.

Textual's exported SVG carries one ``clip-path="...#terminal-<id>-line-<N>"``
per terminal row and one ``<text>`` per run with the cell x in its ``x``
attribute (8px cells). Reconstructing the row from those is how this round
reads a frame: the reconstructed row is what was PAINTED, which is the only
thing a design review may judge.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

_CELL = re.compile(r"-line-(\d+)\)?$")
_NS = "{http://www.w3.org/2000/svg}"


def svg_rows(path: str, cell_width: float = 8.0) -> dict[int, str]:
    """Return ``{row_index: text}`` for every painted row of the frame."""
    root = ET.parse(path).getroot()
    runs: dict[int, list[tuple[float, str]]] = {}
    for node in root.iter(_NS + "text"):
        clip = node.get("clip-path", "")
        match = _CELL.search(clip)
        if not match:
            continue
        row = int(match.group(1))
        # A run's own x is the START of the run; tspans may reposition it.
        pieces: list[tuple[float, str]] = []
        tspans = list(node.iter(_NS + "tspan"))
        if tspans:
            for span in tspans:
                x = span.get("x")
                if x is None:
                    x = node.get("x", "0")
                pieces.append((float(x), "".join(span.itertext())))
        else:
            pieces.append((float(node.get("x", "0")), "".join(node.itertext())))
        runs.setdefault(row, []).extend(pieces)
    rows: dict[int, str] = {}
    for row, pieces in runs.items():
        pieces.sort(key=lambda item: item[0])
        text = ""
        for x, chunk in pieces:
            column = int(round(x / cell_width))
            if column > len(text):
                text += " " * (column - len(text))
            text += chunk
        rows[row] = text
    return rows


def body_rows(path: str, body_y: int, body_height: int) -> list[str]:
    """The body region's rows, top to bottom, exactly as painted."""
    rows = svg_rows(path)
    return [rows.get(body_y + offset, "") for offset in range(body_height)]

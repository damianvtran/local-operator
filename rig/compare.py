"""Cell-level diff of two visual_capture SVGs: what changed on screen, and in what ink."""
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"


def grid(path: str):
    s = re.sub(r"terminal-\d+", "terminal-T", Path(path).read_text())
    styles = dict(re.findall(r"\.(terminal-T-r\d+) \{ ([^}]*)\}", s))
    cells = {}
    for text in ET.fromstring(s).iter(f"{NS}text"):
        style = styles.get(text.get("class", ""), "")
        m = re.search(r"fill: (#[0-9a-fA-F]+)", style)
        fill = m.group(1) if m else "?"
        bold = "bold" in style
        y = round(float(text.get("y")))
        for span in text.iter(f"{NS}tspan"):
            ch = span.text or ""
            if ch:
                cells[(y, round(float(span.get("x"))))] = (ch, fill, bold)
    return cells


def compare(a_path: str, b_path: str) -> list:
    A, B = grid(a_path), grid(b_path)
    keys = sorted(set(A) | set(B))
    diffs = [k for k in keys if A.get(k) != B.get(k)]
    fa, fb = {v[1] for v in A.values()}, {v[1] for v in B.values()}
    print(f"=== {Path(a_path).name} -> {Path(b_path).name}: {len(A)}/{len(B)} cells, {len(diffs)} changed")
    print(f"    fills added {sorted(fb - fa)} / removed {sorted(fa - fb)}")
    for k in diffs:
        print(f"    row={k[0]} x={k[1]}: {A.get(k)} -> {B.get(k)}")
    return diffs


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])

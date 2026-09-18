"""Find the card's painted box in a frame by its own ground colour.

Usage: box.py FRAME.svg GROUND_HEX [ZOOM]  -> writes FRAME.card.png
"""
import subprocess, sys
from collections import Counter
from PIL import Image

CELL_W, CELL_H = 8, 17


def main() -> None:
    svg, ground_hex = sys.argv[1], sys.argv[2]
    zoom = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ground = tuple(int(ground_hex[i:i + 2], 16) for i in (1, 3, 5))
    subprocess.run(["rsvg-convert", "-o", "/tmp/_box.png", svg], check=True)
    im = Image.open("/tmp/_box.png").convert("RGB")
    cols, rows = im.width // CELL_W, im.height // CELL_H
    hit_cols, hit_rows = [], []
    for r in range(rows):
        for c in range(cols):
            # a cell belongs to the card if its centre pixel is the card ground
            if im.getpixel((c * CELL_W + CELL_W // 2, r * CELL_H + 7)) == ground:
                hit_cols.append(c)
                hit_rows.append(r)
    if not hit_cols:
        print("card not found")
        return
    x0, x1 = min(hit_cols), max(hit_cols)
    y0, y1 = min(hit_rows), max(hit_rows)
    print(f"card box: cols {x0}..{x1} ({x1 - x0 + 1}), rows {y0}..{y1} ({y1 - y0 + 1})")
    out = svg.replace(".svg", ".card.png")
    subprocess.run([
        "rsvg-convert", "-z", str(zoom), "-o", "/tmp/_boxz.png", svg,
    ], check=True)
    subprocess.run([
        "magick", "/tmp/_boxz.png", "-crop",
        f"{(x1 - x0 + 3) * CELL_W * zoom}x{(y1 - y0 + 3) * CELL_H * zoom}"
        f"+{(x0 - 1) * CELL_W * zoom}+{(y0 - 1) * CELL_H * zoom}",
        "+repage", out,
    ], check=True)
    print("wrote", out)


main()

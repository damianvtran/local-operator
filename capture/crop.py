"""Crop a terminal-cell box out of a rendered SVG frame, at Nx zoom."""
import subprocess, sys
from pathlib import Path

CELL_W, CELL_H = 8, 17  # the harness's native cells


def crop(svg: str, out: str, x: int, y: int, w: int, h: int, zoom: int = 3,
         pad: int = 1) -> None:
    x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    subprocess.run(["rsvg-convert", "-z", str(zoom), "-o", "/tmp/_z.png", svg], check=True)
    subprocess.run([
        "magick", "/tmp/_z.png", "-crop",
        f"{w * CELL_W * zoom}x{h * CELL_H * zoom}+{x * CELL_W * zoom}+{y * CELL_H * zoom}",
        "+repage", out,
    ], check=True)
    print(f"{out}: {Path(out).stat().st_size} bytes")


if __name__ == "__main__":
    svg, out, x, y, w, h = sys.argv[1:7]
    zoom = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    crop(svg, out, int(x), int(y), int(w), int(h), zoom)

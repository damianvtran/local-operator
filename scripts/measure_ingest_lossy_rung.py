"""Why the PNG-over-budget ladder rung is unreachable at the ingest bound.

The rung in :func:`local_operator.imaging.bound_image_for_model` fires only when
BOTH of these hold at once: the PNG encode is over :data:`IMAGE_MAX_BYTES`, AND
the PNG is still smaller than the JPEG. At the 1024px ingest bound it cannot
happen, and this script is the evidence.

It exists because the comment stating that fact has now carried THREE wrong
mechanisms across three review rounds (PR #603 F2, QA Q1, QA Q3). Every attempt
to summarise the numbers in prose drifted; the conclusion survived every attempt
to falsify it. So the generator is checked in and the comments point here rather
than restating figures nobody can re-derive.

    .venv/bin/python scripts/measure_ingest_lossy_rung.py

Crossover COLOUR COUNTS are fixture-specific (they move with the palette
construction and the seed). The REGIME ORDERING is not, and the ordering is the
whole argument: PNG is never both the smaller encode and over the budget.
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from local_operator.imaging import (  # noqa: E402
    IMAGE_INGEST_MAX_EDGE,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_BYTES,
)

MIB = 1024 * 1024


def _palette_noise(colours: int, edge: int, seed: int) -> Image.Image:
    """Square of uniform noise over ``colours`` distinct values.

    Palette noise is the adversarial case on purpose: it defeats PNG's
    predictors (so the PNG grows with the palette) while staying hostile to
    JPEG's DCT, which is the only shape that could plausibly put PNG over the
    budget while still beating JPEG.
    """
    rng = random.Random(seed)
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]
    while len(palette) < colours:
        palette.append((rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    palette = palette[:colours]

    image = Image.new("RGB", (edge, edge))
    pixels = image.load()
    assert pixels is not None
    draw = random.Random(seed)
    for y in range(edge):
        for x in range(edge):
            pixels[x, y] = palette[draw.randrange(len(palette))]
    return image


def main() -> int:
    edge = IMAGE_INGEST_MAX_EDGE
    print(
        f"area {edge}x{edge}, budget {IMAGE_MAX_BYTES / MIB:.2f} MiB, jpeg q{IMAGE_JPEG_QUALITY}\n"
    )
    header = f"{'colours':>8} {'PNG MiB':>9} {'JPEG MiB':>9} {'over':>6} {'PNGwins':>8}  regime"
    print(header)
    print("-" * len(header))

    reachable = []
    for seed in (1234, 99, 7):
        for colours in (2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192, 256):
            image = _palette_noise(colours, edge, seed)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png = len(buf.getvalue())
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY)
            jpeg = len(buf.getvalue())

            over = png > IMAGE_MAX_BYTES
            wins = png < jpeg
            if over and wins:
                regime = "RUNG REACHED"
                reachable.append((seed, colours))
            elif wins:
                regime = "A: PNG smaller, under budget"
            elif not over:
                regime = "B: JPEG smaller, under budget"
            else:
                regime = "C: JPEG smaller, PNG over budget"
            if seed == 1234:
                print(
                    f"{colours:>8} {png / MIB:>9.3f} {jpeg / MIB:>9.3f} "
                    f"{str(over):>6} {str(wins):>8}  {regime}"
                )

    print("\nsweep: 14 palette sizes x 3 seeds = 42 samples")
    print(f"rung reached: {reachable or 'never'}")
    print(
        "\nThe three regimes are walked in order A -> B -> C. The rung needs "
        "'PNG smaller' AND\n'PNG over budget', which would be a fourth regime "
        "between A and C; B sits there instead."
    )
    return 1 if reachable else 0


if __name__ == "__main__":
    raise SystemExit(main())

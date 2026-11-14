"""Server CPU spent over the 20 snapshot reads, from `ps -o cputime` before/after."""

from __future__ import annotations

import sys


def seconds(text: str) -> float:
    total = 0.0
    for part in text.strip().split(":"):
        total = total * 60 + float(part)
    return total


label = sys.argv[1]
before, after = seconds(sys.argv[2]), seconds(sys.argv[3])
delta = after - before
print(
    f"  {label} 20 snapshot reads: server CPU {sys.argv[2]} -> {sys.argv[3]} "
    f"= {delta:.2f} s total, {delta / 20 * 1000:.1f} ms/request"
)

"""D2's numbers, computed from the theme's real colours (WCAG 2.1 relative luminance)."""
import sys
sys.path.insert(0, "/Users/damian/lo-wt/open-links-r3")
from local_operator.tui import theme as theme_mod

def lum(hexstr):
    h = hexstr.lstrip("#")
    parts = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    lin = [p / 12.92 if p <= 0.04045 else ((p + 0.055) / 1.055) ** 2.4 for p in parts]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)

ground = theme_mod.semantic_color("overlay")
for token in ("faint", "dim", "muted", "fg", "accent"):
    c = theme_mod.semantic_color(token)
    print(f"{token:6} {c} on overlay {ground}: {ratio(c, ground):.2f}:1")
print()
print("card ground == $lo-overlay:", theme_mod.tcss_variable_map().get("lo-overlay"))

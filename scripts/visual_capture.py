"""Developer-only terminal captures; never alter the live app's renderer.

Rich's SVG is a presentation (20px Fira Code plus synthetic window chrome),
not a measurement of the terminal which produced its cell grid. Reproject its
public export into an explicit pixel grid, preserving Textual's layout and
colour output. See docs/VISUAL_CAPTURE.md for calibration and raster limits.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from rich.cells import cell_len

_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _NS)
_SANDBOX: tempfile.TemporaryDirectory[str] | None = None


def isolate_capture() -> None:
    """Call before app imports: config and caches independently consult HOME."""
    global _SANDBOX
    if _SANDBOX is not None:
        return
    _SANDBOX = tempfile.TemporaryDirectory(prefix="lop-visual-")
    os.environ["HOME"] = _SANDBOX.name
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(Path(_SANDBOX.name) / "config")
    os.environ.pop("NO_COLOR", None)
    os.environ["TERM"] = "xterm-256color"
    os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"


@dataclass(frozen=True)
class CaptureProfile:
    """Reproducible preset, not a claim about Terminal.app or Ghostty defaults."""

    cell_width: float = 8
    cell_height: float = 17
    font_size: float = 13
    font_family: str = "Menlo, DejaVu Sans Mono, monospace"

    def __post_init__(self) -> None:
        for value in (self.cell_width, self.cell_height, self.font_size):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("capture dimensions must be finite and positive")
        if self.font_size > self.cell_height:
            raise ValueError("font size must not exceed cell height")
        if not re.fullmatch(r"[\w ,.-]+", self.font_family):
            raise ValueError("font family must be a plain CSS font list")

    @classmethod
    def from_env(cls) -> CaptureProfile:
        return cls(
            cell_width=float(os.environ.get("LOP_CAPTURE_CELL_WIDTH", "8")),
            cell_height=float(os.environ.get("LOP_CAPTURE_CELL_HEIGHT", "17")),
            font_size=float(os.environ.get("LOP_CAPTURE_FONT_SIZE", "13")),
            font_family=os.environ.get("LOP_CAPTURE_FONT_FAMILY", cls.font_family),
        )


def terminal_svg(svg: str, columns: int, rows: int, profile: CaptureProfile) -> str:
    """Keep the compositor output; replace only Rich's presentation geometry.

    Fail loudly if the upstream SVG contract changes. In particular, silently
    keeping a new translation would create plausible but false evidence again.
    Explicit glyph x positions avoid depending on SVG textLength support (librsvg
    does not implement it); combining marks share their preceding cell position.
    """
    root = ET.fromstring(svg)
    style = root.find(f"{{{_NS}}}style")
    group = root.find(f"{{{_NS}}}g[@clip-path]")
    if style is None or group is None or group.get("transform") != "translate(9, 41)":
        raise ValueError("unsupported Rich SVG presentation; inspect export geometry")
    clip = root.find(f".//{{{_NS}}}clipPath/{{{_NS}}}rect")
    if clip is None or columns <= 0 or rows <= 0:
        raise ValueError("missing terminal clip or invalid terminal grid")
    old_width = (float(clip.attrib["width"]) + 1) / columns
    old_height = (float(clip.attrib["height"]) + 1) / rows
    if not math.isclose(old_width, 12.2) or not math.isclose(old_height, 24.4):
        raise ValueError("unsupported Rich SVG cell metrics")
    sx, sy = profile.cell_width / old_width, profile.cell_height / old_height
    width, height = columns * profile.cell_width, rows * profile.cell_height
    root.set("width", f"{width:g}")
    root.set("height", f"{height:g}")
    root.set("viewBox", f"0 0 {width:g} {height:g}")
    for child in list(root):
        if child.tag not in {f"{{{_NS}}}style", f"{{{_NS}}}defs"} and child is not group:
            root.remove(child)
    group.attrib.pop("transform")
    # Rich subtracts a pixel from its terminal clip for its window border. A
    # chrome-free capture needs the whole last cell, including its background.
    clip.set("width", str(columns * old_width))
    clip.set("height", str(rows * old_height))
    css = re.sub(r"@font-face\s*\{.*?\}", "", style.text or "", flags=re.S)
    css = re.sub(r"font-family:[^;]+;", f"font-family: {profile.font_family};", css)
    css = re.sub(r"font-size:[^;]+;", f"font-size: {profile.font_size}px;", css)
    css = re.sub(r"line-height:[^;]+;", f"line-height: {profile.cell_height}px;", css)
    style.text = css
    for element in root.iter():
        if element is root:
            continue
        for attr, scale in (("x", sx), ("y", sy), ("width", sx), ("height", sy)):
            if attr in element.attrib:
                element.set(attr, f"{float(element.attrib[attr]) * scale:g}")
        if element.tag == f"{{{_NS}}}text":
            start = float(element.get("x", "0"))
            positions: list[str] = []
            offset = 0
            for char in element.text or "":
                cells = cell_len(char)
                column = offset if cells else max(0, offset - 1)
                positions.append(f"{start + column * profile.cell_width:g}")
                offset += cells
            element.set("x", " ".join(positions))
            element.attrib.pop("textLength", None)
            element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return ET.tostring(root, encoding="unicode")


def save_capture(app: Any, filename: str | Path, *, profile: CaptureProfile | None = None) -> str:
    """Save a native-size SVG and the cell/box measurements needed to audit it."""
    if not app.CSS_PATH:
        raise ValueError("visual evidence requires a real app with its production CSS")
    profile = profile or CaptureProfile.from_env()
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns, rows = app.size
    path.write_text(terminal_svg(app.export_screenshot(), columns, rows, profile))
    widgets = []
    for widget in app.query("*"):
        if not widget.display or not widget.region:
            continue
        widgets.append(
            {
                "widget": widget.__class__.__name__,
                "id": widget.id,
                "region": list(widget.region),
                "content_region": list(widget.content_region),
                "size": list(widget.size),
                "virtual_size": list(widget.virtual_size),
                "scrollbar": [widget.show_horizontal_scrollbar, widget.show_vertical_scrollbar],
            }
        )
    path.with_suffix(".geometry.json").write_text(
        json.dumps(
            {
                "grid": [columns, rows],
                "native_pixels": [columns * profile.cell_width, rows * profile.cell_height],
                "profile": asdict(profile),
                "font_note": "Local fallback; rasterizer/font versions affect glyphs, not cells",
                "css_path": [str(p) for p in app.CSS_PATH],
                "widgets": widgets,
            },
            indent=2,
        )
        + "\n"
    )
    return str(path)

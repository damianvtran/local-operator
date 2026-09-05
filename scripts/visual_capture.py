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
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import regex
from rich.cells import cell_len

_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _NS)
_SANDBOX: tempfile.TemporaryDirectory[str] | None = None


def isolate_capture() -> None:
    """Call before app imports: config and caches independently consult HOME.

    Prefer ``import scripts.probe_isolation`` as the FIRST import of a
    script — it does this on import and refuses if any ``local_operator``
    module is already loaded, which is the failure this function cannot
    catch (an app imported above the call). Kept for the scripts that
    already call it; a script that imported ``probe_isolation`` first is
    already sandboxed and this is a no-op.
    """
    global _SANDBOX
    if _SANDBOX is not None or "scripts.probe_isolation" in sys.modules:
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
        if "monospace" not in [name.strip().casefold() for name in self.font_family.split(",")]:
            raise ValueError("font family must include a generic monospace fallback")

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
    Explicit cluster x positions avoid depending on SVG textLength support
    (librsvg does not implement it). Grapheme clusters stay whole so ZWJ emoji,
    variation selectors and combining accents can still be shaped by the font.
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
    # Snapshot before adding tspans: they already carry native coordinates and
    # must not be visited (and scaled a second time) by this projection pass.
    for element in list(root.iter()):
        if element is root:
            continue
        for attr, scale in (("x", sx), ("y", sy), ("width", sx), ("height", sy)):
            if attr in element.attrib:
                element.set(attr, f"{float(element.attrib[attr]) * scale:g}")
        if element.tag == f"{{{_NS}}}text":
            start = float(element.get("x", "0"))
            text = element.text or ""
            element.text = None
            offset = 0
            for cluster in regex.findall(r"\X", text):
                span = ET.SubElement(element, f"{{{_NS}}}tspan")
                span.set("x", f"{start + offset * profile.cell_width:g}")
                span.text = cluster
                offset += cell_len(cluster)
            element.attrib.pop("textLength", None)
            element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return ET.tostring(root, encoding="unicode")


@lru_cache(maxsize=16)
def font_provenance(profile: CaptureProfile) -> dict[str, Any]:
    """Measure fontconfig's local selection, used by the librsvg gallery path.

    A browser/other rasterizer may resolve differently. Do not call CSS evidence
    of a font being installed, and do not silently label a fallback as Menlo.
    """
    matcher = shutil.which("fc-match")
    result: dict[str, Any] = {"requested": profile.font_family, "scope": "fontconfig/librsvg"}
    if matcher is None:
        return {**result, "status": "unresolved: fc-match unavailable"}
    faces = []
    for style in ("Regular", "Bold", "Italic"):
        try:
            match = subprocess.run(
                [
                    matcher,
                    "-f",
                    "%{family}\\n%{style}\\n%{file}\\n%{index}\\n",
                    f"{profile.font_family}:style={style}",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.splitlines()
            family, resolved_style, filename, index = match[:4]
            face: dict[str, Any] = {
                "requested_style": style,
                "family": family,
                "style": resolved_style,
                "file": filename,
                "index": index,
            }
            try:
                from PIL import ImageFont

                font = ImageFont.truetype(filename, size=profile.font_size, index=int(index))
                face["ascii_advances"] = {c: font.getlength(c) for c in "iW01"}
                face["ascent_descent"] = list(font.getmetrics())
                advances = list(face["ascii_advances"].values())
                face["monospace_ascii"] = max(advances) - min(advances) < 0.01
                face["measurement"] = "Pillow/FreeType; rasterizer hinting may differ"
            except (ImportError, OSError, ValueError) as exc:
                face["measurement"] = f"unavailable: {type(exc).__name__}"
            faces.append(face)
        except (OSError, ValueError, subprocess.SubprocessError):
            return {**result, "status": "unresolved: fontconfig query failed", "faces": faces}
    first_requested = profile.font_family.split(",")[0].strip().casefold()
    result.update(
        status="resolved" if first_requested in faces[0]["family"].casefold() else "fallback",
        faces=faces,
    )
    if any(face.get("monospace_ascii") is False for face in faces):
        result["warning"] = "Resolved font has variable ASCII advances; not monospace fidelity"
    return result


def save_capture(app: Any, filename: str | Path, *, profile: CaptureProfile | None = None) -> str:
    """Save a native-size SVG and the cell/box measurements needed to audit it."""
    if not app.CSS_PATH:
        raise ValueError("visual evidence requires a real app with its production CSS")
    profile = profile or CaptureProfile.from_env()
    path = Path(filename)
    if path.suffix.lower() != ".svg":
        raise ValueError("capture destination must end in .svg; rasterize separately")
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
                "font": font_provenance(profile),
                "font_note": "Local fallback; rasterizer/font versions affect glyphs, not cells",
                "css_path": [
                    str(p)
                    for p in ([app.CSS_PATH] if isinstance(app.CSS_PATH, str) else app.CSS_PATH)
                ],
                "screen": {
                    "size": list(app.screen.size),
                    "virtual_size": list(app.screen.virtual_size),
                    "region": list(app.screen.region),
                    "scrollbar": [
                        app.screen.show_horizontal_scrollbar,
                        app.screen.show_vertical_scrollbar,
                    ],
                },
                "widgets": widgets,
            },
            indent=2,
        )
        + "\n"
    )
    return str(path)

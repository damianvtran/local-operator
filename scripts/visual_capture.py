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
from html import unescape
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
    # THE DESKTOP SWITCHES TOO, because most captures boot the REAL
    # ``OperatorApp`` (68 shot scripts come through here) and the app is a
    # notification surface: a capture that ends a turn can raise a genuine
    # macOS banner, titled with a fixture string, on a machine running dozens
    # of other sessions. ``probe_isolation`` always set these; this sandbox did
    # not, which is exactly the drift that had to be closed. Spelled as
    # literals for the same reason that module does — this function is called
    # BEFORE the app import it protects, so it may not import
    # ``local_operator.tui.notify`` to ask for the names; the pin in
    # ``tests/unit/test_notification_isolation.py`` is what keeps both
    # sandboxes in step with ``tui.notify.ENV_DISABLE``.
    os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
    os.environ["LOCAL_OPERATOR_NO_DESKTOP_LAUNCH"] = "1"


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


#: One ``<text>`` element of an export, with its baseline ``y`` and its body.
_TEXT_ELEMENT = re.compile(r'<text[^>]*\by="([\d.]+)"[^>]*>(.*?)</text>', re.S)
_MARKUP = re.compile(r"<[^>]+>")


def svg_text_runs_by_row(svg: str) -> list[list[str]]:
    """The text of an exported frame, grouped by baseline, oldest row first.

    Returns each row as its styled RUNS, because joining them is the caller's
    policy: a reader that must not miss a phrase split across two styled spans
    joins a row, while a reader asking which run holds a glyph must not. The
    grouping is the part every caller got wrong in its own copy.

    THREE THINGS ABOUT THIS EXPORT DEFEAT A NAIVE MATCH, and a census written
    without knowing them is a check that can never fire — worse than no census,
    because it looks like one. All three were measured on 2026-09-22 against this
    helper's own export of the welcome splash:

    * A ROW IS NOT A SUBSTRING OF THE FILE. ``terminal_svg`` hands every grapheme
      cluster its own ``<tspan x=…>`` origin, so the row reading
      ``! latest is v0.62.2 — /update`` is stored as ``>!</tspan><tspan
      x="248">l``, and ``"latest is v" in svg`` is FALSE on a frame that paints it.
    * ITS SPACES ARE U+00A0, NOT U+0020. The observed row joins to
      ``'!\\xa0latest\\xa0is\\xa0v0.62.2\\xa0—\\xa0/update\\n'``, so even after
      reassembling the row, ``"latest is v" in joined`` is still FALSE. Textual
      pads cells with no-break space and this export preserves it.
    * THE SAME FRAME HAS TWO FORMS, AND THIS HELPER IS HANDED BOTH. ``App.
      export_screenshot`` (what the shot scripts census) escapes that padding as
      the XML ENTITY ``&#160;`` — 201 of them in a 110x34 frame, and ZERO literal
      U+00A0 bytes — while ``terminal_svg`` round-trips through ElementTree, which
      DECODES the entity, so the artifact written to disk carries 200 literal
      U+00A0 and no entity. A parse that folds only the literal form reports the
      update row ABSENT on the export and PRESENT on the very same frame's
      artifact: measured on one frame, ``hits=0`` against ``hits=1``, and the
      census that believed the first wrote the artifact anyway (design round 6,
      D34). So the body is XML-unescaped first (which is what the string means — an
      escaped ``<`` in a frame is a character, not markup) and the no-break space is
      folded after it, whatever form it arrived in.

    The runs are therefore returned with no-break space folded to a plain space:
    what a reader of the frame sees is a space, and a census is a comparison
    against what is on the screen. A caller comparing a token with no spaces
    (``◆``, ``connecting…``) is unaffected by the fold.
    """
    rows: dict[float, list[str]] = {}
    for y, body in _TEXT_ELEMENT.findall(svg):
        text = unescape(_MARKUP.sub("", body)).replace("\u00a0", " ")
        rows.setdefault(float(y), []).append(text)
    return [rows[y] for y in sorted(rows)]


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


async def settle_status_line(pilot: Any, app: Any, *, tries: int = 200) -> None:
    """Pump until the bottom band carries a real model label, then return.

    WHY THIS EXISTS. The status band is pushed the resolved model label a few
    frames after boot, and until it lands the band paints the ``MODEL_PENDING``
    sentinel (``connecting…``). A capture taken on a fixed number of ``pause()``
    calls therefore races that push: the same script on the same tree painted
    ``connecting…`` in one run and ``test/model`` in the next, which put an
    unrelated pixel band in a before/after pair whose whole point is to differ
    in one thing. QA round 1 on PR #972 caught exactly that (Q2) in the
    committed peer frames.

    Waits on the band's own state rather than on a frame count, because the
    number of frames is what is not knowable — the label arrives from the
    session, not from the layout. Reading ``_status``/``_model_label`` is a
    reach into the band's private state, and that is deliberate: the public
    surface paints the sentinel, so any public read would have to parse the
    very text being waited on.

    The pending state is a NON-EMPTY sentinel, which is the trap this helper
    shipped with: testing ``_model_label`` for truthiness is already true while
    the band is pending, so the wait collapsed to one ``pause()`` — the very
    "hope" it exists to replace (review round 2, M1). Hence the explicit
    comparison, and hence the import of the sentinel instead of a re-typed
    string that could drift from the one the app pushes.

    NEVER RAISES, by design. A status line is not worth failing a capture over,
    so an app with no readable band is a no-op, and a band that never settles
    is bounded and then reported on stderr — the silent version of that path is
    what let a pending frame ship in the first place.
    """
    # Imported here rather than at module scope: this module is imported BEFORE
    # `isolate_capture()` runs, and importing a `local_operator` module pulls in
    # the app's package graph — which must not happen until HOME and the config
    # dir have been redirected, or the isolation applies too late to matter.
    from local_operator.tui.widgets.welcome import MODEL_PENDING

    band = getattr(app, "_status", None)
    label = getattr(band, "_model_label", None)
    if label is None:
        # No band, or a band this helper cannot read. There is no pending state
        # to wait out, so this is a no-op rather than 200 frames of waiting.
        return
    for _ in range(tries):
        if label and label != MODEL_PENDING:
            return
        await pilot.pause()
        label = getattr(band, "_model_label", "")
    print(
        f"warning: status band still reads {label!r} after {tries} frames; "
        "saving the capture with it unsettled",
        file=sys.stderr,
    )


def refuse_flag_shaped_argument(value: str, *, what: str) -> None:
    """Refuse a POSITIONAL that is really a mistyped flag, before it is used as a path.

    WHY THIS EXISTS. ``python scripts/network_shot.py --out frames/`` mkdirs
    ``--out`` — a directory whose name is a flag — inside whatever directory the
    script was run from, and the same shape in ``sidebar_shot.py`` writes
    ``--help.svg``. It happened twice in one review round, in a shared checkout,
    and was cleaned up by hand both times: a scratch by-product of a capturer is
    exactly the sort of thing that ends up in a commit nobody read. The author's
    intent is never ambiguous (nobody has a directory called ``--out``), so the
    loud refusal costs nothing and the silent directory costs a stray file.

    ``what`` names the argument for the message, because "refusing -x" alone
    does not tell a reader WHICH positional was wrong.
    """
    if value.startswith("-"):
        raise SystemExit(
            f"refusing {what} {value!r}: it starts with '-', so it is a mistyped flag "
            "rather than a path (no capture writes into a directory named after a "
            "flag). Pass the positional this script documents."
        )


def save_capture(app: Any, filename: str | Path, *, profile: CaptureProfile | None = None) -> str:
    """Save a native-size SVG and the cell/box measurements needed to audit it.

    THE MEASUREMENTS DESCRIBE THE SCREEN THE SVG SHOWS, and that has to be said
    because the first version did not do it: the widget walk used ``app.query``,
    which Textual's ``App._get_dom_base`` resolves to the **default** screen —
    "when querying from the app, we want to query the default screen" is its own
    docstring — while ``app.export_screenshot()`` composites the ACTIVE one. For a
    modal capture the two are different screens, so the SVG showed the panel and
    the geometry beside it described the transcript underneath, complete with
    ``screen.size = [98, 28]`` and no modal widget at all. Two of the mesh round's
    frames shipped that way, byte-identical to each other and to a screen nobody
    was looking at, which is strictly worse than shipping no geometry: it passes a
    glance (design round 1, D1).

    The active screen is named in the ``screen`` block, so a reader can tell WHICH
    screen a file describes without inferring it from the widget list, and the
    walk now starts from that same screen so the two can never disagree again. For
    every capture that pushes no modal (the other 67 scripts) ``app.screen`` IS the
    default screen, so this is a no-op there.
    """
    if not app.CSS_PATH:
        raise ValueError("visual evidence requires a real app with its production CSS")
    profile = profile or CaptureProfile.from_env()
    path = Path(filename)
    if path.suffix.lower() != ".svg":
        raise ValueError("capture destination must end in .svg; rasterize separately")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns, rows = app.size
    # Read ONCE, before anything is written: the screenshot, the widget walk and
    # the ``screen`` block must all describe the same screen, and a screen pushed
    # or popped mid-capture must not be able to split them.
    screen = app.screen
    path.write_text(terminal_svg(app.export_screenshot(), columns, rows, profile))
    widgets = []
    for widget in screen.query("*"):
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
                    "class": screen.__class__.__name__,
                    "size": list(screen.size),
                    "virtual_size": list(screen.virtual_size),
                    "region": list(screen.region),
                    "scrollbar": [
                        screen.show_horizontal_scrollbar,
                        screen.show_vertical_scrollbar,
                    ],
                },
                "widgets": widgets,
            },
            indent=2,
        )
        + "\n"
    )
    return str(path)

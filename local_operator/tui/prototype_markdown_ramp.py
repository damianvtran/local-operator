"""PROTOTYPE — throwaway. Markdown colour ramp variants for review.

Question this answers: the heading ramp is visually flat (h3/h4 both `muted`,
h5/h6 both `dim` — two greys across four levels) and replies read as plain and
hard to skim. What should the markdown surface actually look like?

Four RADICALLY different answers, rendered through the REAL binding table and
the REAL rich pipeline, so what you see is the bytes a terminal receives — not
a mockup. Flip between them with the bar at the bottom, or the arrow keys.

    make prototype-ramp     # or: .venv/bin/python -m local_operator.tui.prototype_markdown_ramp

Writes ONE self-contained HTML file and opens it. No persistence, no tests, no
abstractions — this is a question-answering artifact, and it gets deleted once
a variant wins. See `skill://prototype`.

NOT IMPORTED BY THE APP. Nothing in `local_operator` imports this module.
"""

from __future__ import annotations

import html
import io
import re
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from local_operator.tui import bindings as b
from local_operator.tui import markdown_theme as mt
from local_operator.tui import palettes  # noqa: F401  (registers the 54 palettes)
from local_operator.tui import theme as t

# --------------------------------------------------------------------------
# The sample. Deliberately the shape of a real assistant reply — nested
# headings, a fence, a table, quotes, mixed inline markup — because a ramp
# that only looks good on six bare headings is not the thing being judged.
# --------------------------------------------------------------------------
SAMPLE = """# Release notes

Prose with **strong emphasis**, *italic*, `inline_code`, and a
[link](https://example.com) so the inline ramp is visible against body text.

## What changed

The heading below is the level assistants actually emit most often.

### Implementation detail

Nested reasoning sits here, one level down from the section.

#### Edge case

##### Aside

###### Footnote

- first bullet, long enough to wrap onto a second rendered row for realism
- second bullet
  - nested bullet
1. ordered item
2. second ordered item

> A quoted log line — someone else's words inside the answer.

```python
def render(level: int) -> str:
    \"\"\"Docstring, a string literal, and a number: 42.\"\"\"
    return "#" * level  # comment
```

| theme | levels | status |
| --- | --- | --- |
| dark | 6 | ok |
| radient | 6 | fixed |

---

Closing paragraph after a rule.
"""

THEMES = [
    "rose-pine-dawn",
    "rose-pine-moon",
    "dark",
    "light",
    "nightfox",
    "catppuccin-mocha",
    "gruvbox-light",
    "everforest",
    "radient",
    "ayu-mirage",
    "tokyo-night",
    "solarized-light",
]


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    blurb: str
    #: element -> (token, bold, italic). Only overrides; anything absent keeps
    #: whatever the shipped table says.
    overrides: dict[str, tuple[str, bool, bool]]


# --------------------------------------------------------------------------
# The four answers. These disagree STRUCTURALLY about what carries hierarchy —
# that is the point. A: lightness. B: hue cycling. C: role-coded zones.
# D: typography with hue removed almost entirely.
# --------------------------------------------------------------------------
VARIANTS: list[Variant] = [
    Variant(
        "A",
        "Current (shipped in #614)",
        "Two greys across h3-h6. The thing being replaced — the control.",
        {},
    ),
    Variant(
        "B",
        "Descending hue ramp",
        "A different hue per level, ordered so contrast DESCENDS h1 to h6. "
        "Hue for scannability, lightness for rank. Avoids warning/danger.",
        {
            "markdown.h1": ("label", True, False),
            "markdown.h2": ("signal", True, False),
            "markdown.h3": ("string", True, False),
            "markdown.h4": ("success", False, False),
            "markdown.h5": ("muted", False, False),
            "markdown.h6": ("dim", False, False),
            "markdown.item.bullet": ("success", False, False),
            "markdown.item.number": ("success", False, False),
            "markdown.block_quote": ("label", False, True),
            "markdown.strong": ("fg", True, False),
        },
    ),
    Variant(
        "C",
        "Spectral cycle (nvim rose-pine style)",
        "Full hue cycle like nvim: every level its own hue, no lightness "
        "ordering. Maximum colour. Spends accent AND warning on prose.",
        {
            "markdown.h1": ("label", True, False),
            "markdown.h2": ("signal", True, False),
            "markdown.h3": ("accent", True, False),
            "markdown.h4": ("warning", True, False),
            "markdown.h5": ("success", True, False),
            "markdown.h6": ("muted", True, False),
            "markdown.item.bullet": ("signal", False, False),
            "markdown.item.number": ("signal", False, False),
            "markdown.block_quote": ("label", False, True),
            "markdown.code": ("string", False, False),
        },
    ),
    Variant(
        "D",
        "Two-hue + typographic rank",
        "Hue marks only the two levels that open sections; everything below "
        "is weight, marker length and lightness. Quietest, most restrained.",
        {
            "markdown.h1": ("label", True, False),
            "markdown.h2": ("signal", True, False),
            "markdown.h3": ("fg", True, False),
            "markdown.h4": ("muted", True, False),
            "markdown.h5": ("muted", False, False),
            "markdown.h6": ("dim", False, False),
            "markdown.item.bullet": ("signal", False, False),
            "markdown.item.number": ("signal", False, False),
        },
    ),
]

_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_OSC8 = re.compile(r"\x1b\]8;[^\x1b]*\x1b\\")


def _apply(variant: Variant) -> None:
    """Rewrite the live binding table in place for ``variant``."""
    table = []
    for binding in b._MARKDOWN_BINDINGS:
        override = variant.overrides.get(binding.element)
        if override:
            token, bold, _italic = override
            binding = replace(binding, token=token, bold=bold)
        table.append(binding)
    b._MARKDOWN_BINDINGS = tuple(table)
    for binding in table:
        b.BY_ELEMENT[binding.element] = binding


def _ansi_to_html(text: str) -> str:
    """Convert rich's truecolor ANSI into spans. Good enough for a prototype."""
    text = _OSC8.sub("", text)
    out: list[str] = []
    fg: str | None = None
    bold = italic = underline = False
    open_span = False

    def close() -> None:
        nonlocal open_span
        if open_span:
            out.append("</span>")
            open_span = False

    def opens() -> None:
        nonlocal open_span
        css = []
        if fg:
            css.append(f"color:{fg}")
        if bold:
            css.append("font-weight:700")
        if italic:
            css.append("font-style:italic")
        if underline:
            css.append("text-decoration:underline")
        out.append(f'<span style="{";".join(css)}">' if css else "<span>")
        open_span = True

    pos = 0
    for m in _SGR.finditer(text):
        if m.start() > pos:
            if not open_span:
                opens()
            out.append(html.escape(text[pos : m.start()]))
        codes = [c for c in m.group(1).split(";") if c] or ["0"]
        i = 0
        close()
        while i < len(codes):
            c = int(codes[i])
            if c == 0:
                fg, bold, italic, underline = None, False, False, False
            elif c == 1:
                bold = True
            elif c == 3:
                italic = True
            elif c == 4:
                underline = True
            elif c == 22:
                bold = False
            elif c == 23:
                italic = False
            elif c == 24:
                underline = False
            elif c == 38 and i + 4 < len(codes) and codes[i + 1] == "2":
                fg = "#%02x%02x%02x" % (
                    int(codes[i + 2]),
                    int(codes[i + 3]),
                    int(codes[i + 4]),
                )
                i += 4
            elif c == 39:
                fg = None
            i += 1
        pos = m.end()
    if pos < len(text):
        if not open_span:
            opens()
        out.append(html.escape(text[pos:]))
    close()
    return "".join(out)


def _render(theme_name: str, width: int = 76) -> str:
    t.set_theme(theme_name)
    buf = io.StringIO()
    console = Console(
        width=width, file=buf, force_terminal=True, color_system="truecolor"
    )
    console.push_theme(mt.brand_markdown_theme())
    console.print(Markdown(SAMPLE))
    return _ansi_to_html(buf.getvalue())


def build() -> Path:
    mt.install_markdown_theme()
    pristine = b._MARKDOWN_BINDINGS

    panels: list[str] = []
    for variant in VARIANTS:
        b._MARKDOWN_BINDINGS = pristine
        for binding in pristine:
            b.BY_ELEMENT[binding.element] = binding
        _apply(variant)

        frames = []
        for theme_name in THEMES:
            t.set_theme(theme_name)
            bg = t.semantic_color("bg")
            fg = t.semantic_color("fg")
            body = _render(theme_name)
            frames.append(
                f'<figure class="frame" data-theme="{html.escape(theme_name)}">'
                f"<figcaption>{html.escape(theme_name)}</figcaption>"
                f'<pre style="background:{bg};color:{fg}">{body}</pre>'
                f"</figure>"
            )
        panels.append(
            f'<section class="variant" id="v-{variant.key}" hidden>'
            f'<header class="vh"><h2>{variant.key} — {html.escape(variant.name)}</h2>'
            f"<p>{html.escape(variant.blurb)}</p></header>"
            f'<div class="grid">{"".join(frames)}</div>'
            f"</section>"
        )

    b._MARKDOWN_BINDINGS = pristine
    for binding in pristine:
        b.BY_ELEMENT[binding.element] = binding

    keys = [v.key for v in VARIANTS]
    labels = {v.key: f"{v.key} — {v.name}" for v in VARIANTS}
    theme_opts = "".join(
        f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in THEMES
    )

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>PROTOTYPE — markdown colour ramp</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; background:#0d0d10; color:#e8e8ea;
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
 .banner {{ background:#3a2c12; border-bottom:1px solid #6b5423; color:#f4d58d;
            padding:10px 20px; font-size:13px; }}
 .vh {{ padding:22px 20px 8px; }}
 .vh h2 {{ margin:0 0 4px; font-size:19px; }}
 .vh p  {{ margin:0; color:#a0a0a8; max-width:70ch; }}
 .grid {{ display:grid; gap:18px; padding:14px 20px 120px;
          grid-template-columns:repeat(auto-fit,minmax(660px,1fr)); }}
 .frame {{ margin:0; border:1px solid #26262c; border-radius:8px; overflow:hidden; }}
 figcaption {{ padding:6px 12px; background:#17171b; color:#8f8f99;
               font-size:12px; font-family:ui-monospace,Menlo,monospace; }}
 pre {{ margin:0; padding:16px; overflow-x:auto; font-size:12.5px; line-height:1.5;
        font-family:ui-monospace,Menlo,"SF Mono",monospace; }}
 .bar {{ position:fixed; left:50%; transform:translateX(-50%); bottom:22px;
         display:flex; align-items:center; gap:14px; padding:10px 14px;
         background:#f4f4f6; color:#101014; border-radius:999px;
         box-shadow:0 8px 34px rgba(0,0,0,.6); z-index:99; }}
 .bar button {{ border:0; background:#101014; color:#fff; width:32px; height:32px;
                border-radius:50%; cursor:pointer; font-size:15px; }}
 .bar .lab {{ min-width:290px; text-align:center; font-weight:650; font-size:13px; }}
 .bar select {{ border-radius:6px; border:1px solid #c3c3cb; padding:5px 7px;
                font-size:12px; background:#fff; color:#101014; }}
</style>
<div class="banner">
  <strong>PROTOTYPE — throwaway.</strong>
  Real ANSI from the real binding table, so this is what a terminal actually
  paints. Flip variants with <kbd>←</kbd>/<kbd>→</kbd> or the bar. Filter to one
  theme to compare like-for-like.
</div>
{"".join(panels)}
<div class="bar">
  <button id="prev" title="Previous variant">←</button>
  <div class="lab" id="lab"></div>
  <button id="next" title="Next variant">→</button>
  <select id="theme"><option value="__all">all themes</option>{theme_opts}</select>
</div>
<script>
const KEYS = {keys!r};
const LABELS = {labels!r};
const qs = new URLSearchParams(location.search);
let idx = Math.max(0, KEYS.indexOf(qs.get('variant') || KEYS[0]));
let theme = qs.get('theme') || '__all';

function paint() {{
  KEYS.forEach((k, i) => {{
    document.getElementById('v-' + k).hidden = (i !== idx);
  }});
  document.getElementById('lab').textContent = LABELS[KEYS[idx]];
  document.querySelectorAll('.frame').forEach(f => {{
    f.style.display = (theme === '__all' || f.dataset.theme === theme) ? '' : 'none';
  }});
  const u = new URLSearchParams();
  u.set('variant', KEYS[idx]);
  if (theme !== '__all') u.set('theme', theme);
  history.replaceState(null, '', '?' + u.toString());
  document.getElementById('theme').value = theme;
}}
const step = d => {{ idx = (idx + d + KEYS.length) % KEYS.length; paint(); }};
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('theme').onchange = e => {{ theme = e.target.value; paint(); }};
addEventListener('keydown', e => {{
  const el = document.activeElement, tag = el ? el.tagName : '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      || (el && el.isContentEditable)) return;
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
}});
paint();
</script>
"""
    out = Path("/tmp/lop-ramp-prototype.html")
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    path = build()
    print(f"prototype written: {path}")
    print(f"variants: {', '.join(v.key for v in VARIANTS)}")
    webbrowser.open(f"file://{path}")

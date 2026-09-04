"""PROTOTYPE — throwaway. The questions still open after the rmnvim investigation.

Renders each variant through the REAL binding table and rich pipeline, so what
you see is the bytes a terminal receives.

Open questions this answers:

  A  SHIPPED       h1-h5 bold, h6 plain. h4 `success` and h5 `accent` are both
                   greens -- dE76 5.5 apart in `light`, effectively identical.
  B  WEIGHT RANK   h1-h3 bold, h4-h6 plain: weight carries rank again, the
                   state before commit 31a6d530.
  C  5-HUE         Drop h5 to the neutral ramp so no two levels share a hue
                   family. h5 `muted` bold, h6 `muted` plain.
  D  DANGER@h5     Spend the one token measured free in 42/54 themes. Tests
                   whether "never use danger for structure" is a real
                   constraint or an inherited assumption.
  E  MARKERS       A + `display.heading_markers` on, for comparison.

    python /tmp/shots/proto2.py

Writes one self-contained HTML file. Deleted once a variant wins.
"""

from __future__ import annotations

import html
import io
import re
import sys
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, ".")

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.style import Style  # noqa: E402
from rich.text import Text  # noqa: E402

from local_operator.tui import bindings as b  # noqa: E402
from local_operator.tui import markdown_theme as mt  # noqa: E402
from local_operator.tui import palettes  # noqa: F401,E402
from local_operator.tui import theme as t  # noqa: E402

MD = """# Retry backoff

## What I found

Three call sites drop the `Retry-After` header, so a 429 is retried
**immediately** instead of backing off — see `client/session.py`.

### Why it only shows under load

The header is parsed but never threaded onto `RetryState`.

#### Reproduction

- run the load harness with two requests in one window

##### Caveat

###### Footnote
"""

CODE = '''class RetryPolicy:
    """Backoff policy honouring Retry-After."""

    def retry_after(self, response: Response) -> float:
        raw = response.headers.get("Retry-After")  # often missing
        if raw is None:
            return 1.0
        return max(0.0, float(raw))
'''

TOOLS = [
    ("⛁", "read", "client/session.py · 412 lines", "tool.row.name_read"),
    ("✎", "edit", "client/session.py · +18 −4", "tool.row.name_mutate"),
    ("⚡", "bash", "pytest tests/unit/test_retry.py -q", "tool.row.name_exec"),
    ("◆", "task", "reviewer · audit the retry diff", "tool.row.name_meta"),
]

THEMES = [
    "dark",
    "light",
    "rose-pine-dawn",
    "rose-pine-moon",
    "gruvbox-light",
    "catppuccin-mocha",
    "radient",
    "nightfox",
    "solarized-light",
    "everforest",
    "tokyo-night",
    "ayu-mirage",
]


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    blurb: str
    ramp: tuple[tuple[str, bool], ...]
    markers: bool = False


VARIANTS: list[Variant] = [
    Variant(
        "A",
        "Shipped now",
        "h1-h5 bold, h6 plain. h4 `success` and h5 `accent` are both greens — "
        "dE76 5.5 apart in `light`, which is effectively one colour.",
        (("signal", True), ("label", True), ("warning", True),
         ("success", True), ("accent", True), ("muted", False)),
    ),
    Variant(
        "B",
        "Weight carries rank",
        "h1-h3 bold, h4-h6 plain — the state before 31a6d530. Weight is a "
        "second ordering channel; the trade is that h4-h6 read lighter "
        "against prose.",
        (("signal", True), ("label", True), ("warning", True),
         ("success", False), ("accent", False), ("muted", False)),
    ),
    Variant(
        "C",
        "Five hues, neutral tail",
        "Drops h5 onto the neutral ramp so no two levels share a hue family. "
        "Fixes the green/green pair outright; costs the tail its colour.",
        (("signal", True), ("label", True), ("warning", True),
         ("success", True), ("muted", True), ("muted", False)),
    ),
    Variant(
        "D",
        "danger at h5",
        "Spends the one chromatic token measured free in 42/54 themes. Tests "
        "whether 'never spend danger on structure' is a real constraint or an "
        "inherited assumption — a red heading may simply read as an error.",
        (("signal", True), ("label", True), ("warning", True),
         ("success", True), ("danger", True), ("muted", False)),
    ),
    Variant(
        "E",
        "Shipped + markers on",
        "Variant A with `display.heading_markers: true`, so the levels are "
        "stated as well as tinted.",
        (("signal", True), ("label", True), ("warning", True),
         ("success", True), ("accent", True), ("muted", False)),
        markers=True,
    ),
]

_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_OSC8 = re.compile(r"\x1b\]8;[^\x1b]*\x1b\\")


def ansi_to_html(text: str) -> str:
    text = _OSC8.sub("", text)
    out: list[str] = []
    fg = bg = None
    bold = italic = underline = False
    live = False

    def close() -> None:
        nonlocal live
        if live:
            out.append("</span>")
            live = False

    def open_() -> None:
        nonlocal live
        css = []
        if fg:
            css.append(f"color:{fg}")
        if bg:
            css.append(f"background:{bg}")
        if bold:
            css.append("font-weight:700")
        if italic:
            css.append("font-style:italic")
        if underline:
            css.append("text-decoration:underline")
        out.append(f'<span style="{";".join(css)}">' if css else "<span>")
        live = True

    pos = 0
    for m in _SGR.finditer(text):
        if m.start() > pos:
            if not live:
                open_()
            out.append(html.escape(text[pos : m.start()]))
        codes = [c for c in m.group(1).split(";") if c] or ["0"]
        nl_next = text[m.end() : m.end() + 1] == "\n"
        keep_bg = bg if (nl_next and codes == ["0"]) else None
        close()
        i = 0
        while i < len(codes):
            c = int(codes[i])
            if c == 0:
                fg = bg = None
                bold = italic = underline = False
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
                fg = "#%02x%02x%02x" % tuple(int(codes[i + j]) for j in (2, 3, 4))
                i += 4
            elif c == 39:
                fg = None
            elif c == 48 and i + 4 < len(codes) and codes[i + 1] == "2":
                bg = "#%02x%02x%02x" % tuple(int(codes[i + j]) for j in (2, 3, 4))
                i += 4
            elif c == 49:
                bg = None
            i += 1
        if keep_bg is not None:
            bg = keep_bg
        pos = m.end()
    if pos < len(text):
        if not live:
            open_()
        out.append(html.escape(text[pos:]))
    close()
    return "".join(out)


def apply(v: Variant) -> None:
    table = []
    for binding in b._MARKDOWN_BINDINGS:
        if binding.element.startswith("markdown.h") and binding.element[10:].isdigit():
            token, bold = v.ramp[int(binding.element[10:]) - 1]
            binding = replace(binding, token=token, bold=bold)
        table.append(binding)
    b._MARKDOWN_BINDINGS = tuple(table)
    for binding in table:
        b.BY_ELEMENT[binding.element] = binding


def render(theme_name: str, width: int = 78) -> str:
    t.set_theme(theme_name)
    buf = io.StringIO()
    con = Console(width=width, file=buf, force_terminal=True, color_system="truecolor")
    con.push_theme(mt.brand_markdown_theme())
    con.print(Markdown(MD))
    con.print(Markdown(f"```python\n{CODE}```"))
    con.print(Text(""))
    for icon, name, summary, element in TOOLS:
        st = b.style(element)
        row = Text()
        row.append(f" {icon} ", style=st)
        row.append(f"{name:<8}", style=st)
        row.append(summary, style=Style(color=t.semantic_color("dim")))
        con.print(row)
    return ansi_to_html(buf.getvalue())


def build() -> Path:
    mt.install_markdown_theme()
    pristine = b._MARKDOWN_BINDINGS
    real = mt.settings_get
    panels = []

    for v in VARIANTS:
        b._MARKDOWN_BINDINGS = pristine
        for x in pristine:
            b.BY_ELEMENT[x.element] = x
        apply(v)
        mt.settings_get = (
            (lambda k, d=None: True if k == "display.heading_markers" else real(k, d))
            if v.markers
            else real
        )
        frames = []
        for n in THEMES:
            t.set_theme(n)
            frames.append(
                f'<figure class="frame" data-theme="{html.escape(n)}">'
                f"<figcaption>{html.escape(n)}</figcaption>"
                f'<pre style="background:{t.semantic_color("bg")};'
                f'color:{t.semantic_color("fg")}">{render(n)}</pre></figure>'
            )
        ramp = " · ".join(
            f"h{i + 1} {tok}{'' if bold else ' (plain)'}" for i, (tok, bold) in enumerate(v.ramp)
        )
        panels.append(
            f'<section class="variant" id="v-{v.key}" hidden>'
            f'<header class="vh"><h2>{v.key} — {html.escape(v.name)}</h2>'
            f"<p>{html.escape(v.blurb)}</p><p class='ramp'>{html.escape(ramp)}</p></header>"
            f'<div class="grid">{"".join(frames)}</div></section>'
        )

    mt.settings_get = real
    b._MARKDOWN_BINDINGS = pristine
    for x in pristine:
        b.BY_ELEMENT[x.element] = x

    keys = [v.key for v in VARIANTS]
    labels = {v.key: f"{v.key} — {v.name}" for v in VARIANTS}
    opts = "".join(f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in THEMES)

    doc = f"""<!doctype html><meta charset="utf-8"><title>PROTOTYPE — heading ramp</title>
<style>
 :root{{color-scheme:dark}}
 body{{margin:0;background:#0c0c0f;color:#e9e9ec;
   font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
 .banner{{background:#3a2c12;border-bottom:1px solid #6b5423;color:#f4d58d;
   padding:11px 20px;font-size:13px}}
 .banner b{{color:#ffe9b0}}
 .vh{{padding:20px 20px 8px}} .vh h2{{margin:0 0 5px;font-size:19px}}
 .vh p{{margin:0 0 4px;color:#a2a2ab;max-width:80ch}}
 .vh .ramp{{font:12px ui-monospace,Menlo,monospace;color:#7f8fa0}}
 .grid{{display:grid;gap:16px;padding:12px 20px 130px;
   grid-template-columns:repeat(auto-fit,minmax(680px,1fr))}}
 .frame{{margin:0;border:1px solid #26262e;border-radius:8px;overflow:hidden}}
 figcaption{{padding:6px 11px;background:#17171d;color:#8f8f99;font-size:12px;
   font-family:ui-monospace,Menlo,monospace}}
 pre{{margin:0;padding:15px;overflow-x:auto;font-size:12px;line-height:1.5;
   font-family:ui-monospace,Menlo,"SF Mono",monospace}}
 .bar{{position:fixed;left:50%;transform:translateX(-50%);bottom:22px;display:flex;
   align-items:center;gap:14px;padding:10px 14px;background:#f4f4f6;color:#101014;
   border-radius:999px;box-shadow:0 8px 34px rgba(0,0,0,.6);z-index:99}}
 .bar button{{border:0;background:#101014;color:#fff;width:32px;height:32px;
   border-radius:50%;cursor:pointer;font-size:15px}}
 .bar .lab{{min-width:250px;text-align:center;font-weight:650;font-size:13px}}
 .bar select{{border-radius:6px;border:1px solid #c3c3cb;padding:5px 7px;
   font-size:12px;background:#fff;color:#101014}}
</style>
<div class="banner">
 <b>PROTOTYPE — throwaway.</b> Real ANSI from the real binding table.
 <b>A</b> is what is on the PR now. Look at <b>light</b> in A: h4 and h5 are both
 greens 5.5 dE apart — the open defect. <kbd>←</kbd>/<kbd>→</kbd> to switch.
</div>
{"".join(panels)}
<div class="bar">
 <button id="prev">←</button><div class="lab" id="lab"></div><button id="next">→</button>
 <select id="theme"><option value="__all">all themes</option>{opts}</select>
</div>
<script>
const KEYS={keys!r}, LABELS={labels!r};
const qs=new URLSearchParams(location.search);
let idx=Math.max(0,KEYS.indexOf(qs.get('variant')||KEYS[0]));
let theme=qs.get('theme')||'__all';
function paint(){{
 KEYS.forEach((k,i)=>{{document.getElementById('v-'+k).hidden=(i!==idx);}});
 document.getElementById('lab').textContent=LABELS[KEYS[idx]];
 document.querySelectorAll('.frame').forEach(f=>{{
   f.style.display=(theme==='__all'||f.dataset.theme===theme)?'':'none';}});
 const u=new URLSearchParams(); u.set('variant',KEYS[idx]);
 if(theme!=='__all')u.set('theme',theme);
 history.replaceState(null,'','?'+u.toString());
 document.getElementById('theme').value=theme;
}}
const step=d=>{{idx=(idx+d+KEYS.length)%KEYS.length;paint();}};
document.getElementById('prev').onclick=()=>step(-1);
document.getElementById('next').onclick=()=>step(1);
document.getElementById('theme').onchange=e=>{{theme=e.target.value;paint();}};
addEventListener('keydown',e=>{{
 const el=document.activeElement,tag=el?el.tagName:'';
 if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT'||(el&&el.isContentEditable))return;
 if(e.key==='ArrowLeft')step(-1); if(e.key==='ArrowRight')step(1);
}});
paint();
</script>"""
    out = Path("/tmp/lop-heading-proto.html")
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"prototype: {p}")
    print(f"variants: {', '.join(v.key for v in VARIANTS)}")
    webbrowser.open(f"file://{p}")

"""PROTOTYPE — throwaway. Whole-TUI colour variants for review.

The question: **lop underuses each theme's palette.** Measured on the shipped
binding table:

- 61% of all 76 bindings are painted from the NEUTRAL grey ramp
  (`fg`/`muted`/`dim`/`faint`); only 39% spend any hue at all.
- The transcript — the surface you spend the most time reading — is only 34%
  hue.
- 12 of the 23 semantic tokens every theme is required to define are NEVER
  referenced by the table: `surface`, `raised`, `overlay`, `sunken`, `faint`,
  `edge`, `edge-hi`, and all five `tint-*`.
- Meanwhile the average theme offers a **median of 6** mutually distinct,
  AA-legible hues (>=4.5:1 on `bg`, dE>=15 apart); 47 of 54 offer >=5.

So the palettes are rich and the UI is beige. This prototype shows what
spending that headroom actually looks like, across the three surfaces that
make up a session: assistant markdown, code fences, and the tool ledger.

Rendered through the REAL binding table and the REAL rich pipeline, so what
you see is the bytes a terminal receives — not a mockup.

    make prototype-ramp

Writes ONE self-contained HTML file and opens it. No persistence, no tests,
no abstractions — this gets deleted once a variant wins. See
`skill://prototype`.

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
from rich.text import Text

from local_operator.tui import bindings as b
from local_operator.tui import markdown_theme as mt
from local_operator.tui import palettes  # noqa: F401  (registers the 54 palettes)
from local_operator.tui import theme as t

# --------------------------------------------------------------------------
# A realistic slice of a session: prose, a fence, and a tool ledger. A ramp
# that only looks good on six bare headings is not the thing being judged.
# --------------------------------------------------------------------------
SAMPLE_MD = """## What I found

The failure is in the retry path, not the transport. Three call sites drop the
`Retry-After` header, so a 429 is retried **immediately** instead of backing
off — see `client/session.py` and the [upstream issue](https://example.com).

### Why it only shows under load

- the header is parsed but never threaded onto `RetryState`
- a single-request test can never catch it
  - the bug needs two requests in the same window
1. reproduce with the load harness
2. assert on the sleep, not the response

> 429 Too Many Requests — Retry-After: 30

| path | calls | fixed |
| --- | --- | --- |
| session.py | 3 | yes |
| pool.py | 1 | no |
"""

SAMPLE_CODE = '''def retry_after(response: Response, default: float = 1.0) -> float:
    """Seconds to wait before retrying. Honours Retry-After when present."""
    raw = response.headers.get("Retry-After")  # often missing
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default
'''

#: (icon, name, summary, state) — a settled ledger plus one live row.
SAMPLE_TOOLS = [
    ("⛁", "read", "client/session.py · 412 lines", "read"),
    ("✎", "edit", "client/session.py · +18 −4", "mutate"),
    ("⚡", "bash", "pytest tests/unit/test_retry.py -q", "exec"),
    ("⛁", "read", "client/pool.py · 88 lines", "read"),
    ("◆", "task", "reviewer · audit the retry diff", "meta"),
    ("⚡", "bash", "ruff check client/", "fail"),
    ("⧗", "web_fetch", "fetching upstream issue…", "running"),
]


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    blurb: str
    #: element -> (token, bold). Only overrides; absent = shipped value.
    overrides: dict[str, tuple[str, bool]]
    #: Ground token for the code fence. `bg` (the shipped value) means the
    #: fence is invisible AS A BLOCK — it is the same paper as the prose
    #: around it. `raised`/`surface` give it an actual slab, which is the
    #: single largest unspent token in the palette: `raised` is visibly
    #: distinct from `bg` in 53 of 54 themes and still holds `fg` at 6.09:1.
    fence_ground: str = "bg"


# --------------------------------------------------------------------------
# Four structurally different theories of where colour belongs. They disagree
# about WHICH SURFACE earns hue, not merely about which hue.
# --------------------------------------------------------------------------
VARIANTS: list[Variant] = [
    Variant(
        "A",
        "Shipped today (control)",
        "61% neutral overall, 34% in the transcript. Code fences are almost "
        "entirely grey; the settled ledger is a wall of grey. The baseline.",
        {},
    ),
    Variant(
        "B",
        "Syntax-first",
        "Spend the budget where the eye actually parses structure: the CODE "
        "FENCE. Real syntax highlighting (keywords, functions, classes, "
        "builtins all get hue) plus a hue-ordered heading ramp. Tool ledger "
        "stays calm so code is the brightest thing on screen.",
        {
            # Headings: hue carries scannability, marker length carries rank.
            "markdown.h1": ("label", True),
            "markdown.h2": ("signal", True),
            "markdown.h3": ("string", True),
            "markdown.h4": ("success", False),
            "markdown.h5": ("muted", False),
            "markdown.h6": ("dim", False),
            # The fence stops being beige.
            "code.keyword": ("label", True),
            "code.name_function": ("signal", False),
            "code.name_class": ("warning", False),
            "code.name_builtin": ("accent", False),
            "code.operator": ("label", False),
            "code.string": ("success", False),
            "code.number": ("warning", False),
            "code.comment": ("dim", False),
            "markdown.item.bullet": ("signal", False),
            "markdown.item.number": ("signal", False),
        },
    ),
    Variant(
        "C",
        "Ledger-first",
        "Spend it on the TOOL ROW so a long session can be skimmed for what "
        "touched the machine. Every settled row's name and summary carry the "
        "tool-category hue instead of one uniform grey. Prose stays quiet.",
        {
            "markdown.h1": ("label", True),
            "markdown.h2": ("signal", True),
            "markdown.h3": ("muted", True),
            "markdown.h4": ("muted", False),
            # The ledger stops being one grey.
            "tool.row.name_settled": ("signal", True),
            "tool.row.summary_settled": ("muted", False),
            "tool.row.icon_settled": ("signal", False),
            "tool.row.chip_settled": ("label", False),
            "tool.status.duration": ("muted", False),
            "tool.search.snippet": ("muted", False),
            "tool.fetch.snippet": ("muted", False),
            "tool.args.value": ("string", False),
            "tool.diff.hunk": ("label", False),
            "code.keyword": ("label", False),
            "code.name_function": ("signal", False),
            "code.string": ("success", False),
        },
    ),
    Variant(
        "D",
        "Maximal — every surface",
        "Spend everywhere: hue-ordered headings, full syntax colour, a "
        "category-coded ledger, and coloured bullets, quotes and rules. The "
        "loudest option, and the honest upper bound on 'use more colour'.",
        {
            "markdown.h1": ("label", True),
            "markdown.h2": ("signal", True),
            "markdown.h3": ("string", True),
            "markdown.h4": ("success", False),
            "markdown.h5": ("muted", False),
            "markdown.h6": ("dim", False),
            "markdown.item.bullet": ("success", False),
            "markdown.item.number": ("success", False),
            "markdown.block_quote": ("warning", False),
            "markdown.hr": ("label", False),
            "markdown.strong": ("string", True),
            "markdown.code": ("signal", False),
            "code.keyword": ("label", True),
            "code.name_function": ("signal", False),
            "code.name_class": ("warning", False),
            "code.name_builtin": ("accent", False),
            "code.operator": ("label", False),
            "code.string": ("success", False),
            "code.number": ("warning", False),
            "tool.row.name_settled": ("signal", True),
            "tool.row.icon_settled": ("signal", False),
            "tool.row.chip_settled": ("label", False),
            "tool.args.value": ("string", False),
            "tool.diff.hunk": ("label", False),
        },
    ),
    Variant(
        "E",
        "Maximal + elevation (uses the unspent grounds)",
        "Everything in D, plus the 12 tokens the table never touches. The "
        "code fence gets a real `raised` slab instead of sitting on the same "
        "paper as the prose — the single biggest unspent token, visibly "
        "distinct from `bg` in 53/54 themes and still holding `fg` at "
        "6.09:1. Colour AND depth, not just colour.",
        {
            "markdown.h1": ("label", True),
            "markdown.h2": ("signal", True),
            "markdown.h3": ("string", True),
            "markdown.h4": ("success", False),
            "markdown.h5": ("muted", False),
            "markdown.h6": ("dim", False),
            "markdown.item.bullet": ("success", False),
            "markdown.item.number": ("success", False),
            "markdown.block_quote": ("warning", False),
            "markdown.hr": ("label", False),
            "markdown.strong": ("string", True),
            "markdown.code": ("signal", False),
            "code.keyword": ("label", True),
            "code.name_function": ("signal", False),
            "code.name_class": ("warning", False),
            "code.name_builtin": ("accent", False),
            "code.operator": ("label", False),
            "code.string": ("success", False),
            "code.number": ("warning", False),
            "tool.row.name_settled": ("signal", True),
            "tool.row.icon_settled": ("signal", False),
            "tool.row.chip_settled": ("label", False),
            "tool.args.value": ("string", False),
            "tool.diff.hunk": ("label", False),
        },
        fence_ground="raised",
    ),
]

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
    "tokyo-night",
    "solarized-light",
    "ayu-mirage",
]

_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_OSC8 = re.compile(r"\x1b\]8;[^\x1b]*\x1b\\")

#: Which token paints a settled row of each category, per variant intent.
_CATEGORY_TOKEN = {
    "read": "tool-read",
    "mutate": "tool-mutate",
    "exec": "tool-exec",
    "meta": "tool-meta",
    "fail": "danger",
    "running": "accent",
}


def _apply(variant: Variant) -> None:
    table = []
    for binding in b._MARKDOWN_BINDINGS:
        ov = variant.overrides.get(binding.element)
        table.append(replace(binding, token=ov[0], bold=ov[1]) if ov else binding)
    b._MARKDOWN_BINDINGS = tuple(table)
    for binding in table:
        b.BY_ELEMENT[binding.element] = binding

    code = []
    for binding in b._CODE_BINDINGS:
        ov = variant.overrides.get(binding.element)
        code.append(replace(binding, token=ov[0], bold=ov[1]) if ov else binding)
    b._CODE_BINDINGS = tuple(code)
    for binding in code:
        b.BY_ELEMENT[binding.element] = binding

    for element, (token, bold) in variant.overrides.items():
        if element.startswith("tool."):
            cur = b.BY_ELEMENT.get(element)
            if cur is not None:
                b.BY_ELEMENT[element] = replace(cur, token=token, bold=bold)

    # The fence's SLAB. `code.background` is a Role.GROUND binding, so it is
    # read through `ground_hex()` rather than `style()` — it paints the
    # surface instead of ink on one. Repointing it is how the fence stops
    # being the same paper as the prose around it.
    ground = b.BY_ELEMENT.get("code.background")
    if ground is not None:
        b.BY_ELEMENT["code.background"] = replace(
            ground, token=variant.fence_ground, ground=variant.fence_ground
        )
        b._CODE_BINDINGS = tuple(
            b.BY_ELEMENT["code.background"] if x.element == "code.background" else x
            for x in b._CODE_BINDINGS
        )


def _ansi_to_html(text: str) -> str:
    text = _OSC8.sub("", text)
    out: list[str] = []
    fg = None
    bg = None
    bold = italic = underline = False
    live = False

    def close():
        nonlocal live
        if live:
            out.append("</span>")
            live = False

    def open_():
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
            # 48;2;r;g;b — the BACKGROUND. Rich emits it on every cell of a
            # code fence (the slab), immediately after the foreground in the
            # same escape. Not consuming it here left `i` pointing into the
            # middle of the triplet, so the next loop read `2`/`39`/`34` as
            # fresh SGR codes — and `39` is "default foreground", which
            # silently CLEARED the colour rich had just set. That is why the
            # fence rendered bold-but-colourless while every other surface
            # was fine: only fences carry a background.
            elif c == 48 and i + 4 < len(codes) and codes[i + 1] == "2":
                bg = "#%02x%02x%02x" % tuple(int(codes[i + j]) for j in (2, 3, 4))
                i += 4
            elif c == 49:
                bg = None
            i += 1
        pos = m.end()
    if pos < len(text):
        if not live:
            open_()
        out.append(html.escape(text[pos:]))
    close()
    return "".join(out)


def _console(width: int) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    con = Console(width=width, file=buf, force_terminal=True, color_system="truecolor")
    con.push_theme(mt.brand_markdown_theme())
    return con, buf


def _render_session(theme_name: str, width: int = 78) -> str:
    """Prose + fence + tool ledger, as one continuous transcript."""
    t.set_theme(theme_name)
    con, buf = _console(width)
    con.print(Markdown(SAMPLE_MD))
    con.print(Markdown(f"```python\n{SAMPLE_CODE}```"))

    def sty(token: str, bold: bool = False):
        from rich.style import Style

        return Style(color=t.semantic_color(token), bold=bold)

    con.print(Text(""))
    for icon, name, summary, state in SAMPLE_TOOLS:
        running = state == "running"
        failed = state == "fail"
        if running:
            icon_tok, name_tok, sum_tok = "accent", "string", "muted"
        elif failed:
            icon_tok, name_tok, sum_tok = "danger", "danger", "muted"
        else:
            name_b = b.BY_ELEMENT.get("tool.row.name_settled")
            icon_b = b.BY_ELEMENT.get("tool.row.icon_settled")
            sum_b = b.BY_ELEMENT.get("tool.row.summary_settled")
            cat = _CATEGORY_TOKEN.get(state, "muted")
            name_tok = name_b.token if name_b else "muted"
            if name_tok in ("muted", "dim"):
                name_tok = cat if cat in t.SEMANTIC_TOKENS else name_tok
            icon_tok = icon_b.token if icon_b else "dim"
            sum_tok = sum_b.token if sum_b else "dim"
        row = Text()
        row.append(f" {icon} ", style=sty(icon_tok))
        row.append(f"{name:<10}", style=sty(name_tok, bold=state != "running"))
        row.append(summary, style=sty(sum_tok))
        if not running:
            row.append("   1.2s", style=sty("dim"))
        con.print(row)
    return _ansi_to_html(buf.getvalue())


def build() -> Path:
    mt.install_markdown_theme()
    md0, code0 = b._MARKDOWN_BINDINGS, b._CODE_BINDINGS
    tool0 = {e: x for e, x in b.BY_ELEMENT.items() if e.startswith("tool.")}

    panels = []
    for v in VARIANTS:
        b._MARKDOWN_BINDINGS, b._CODE_BINDINGS = md0, code0
        for x in list(md0) + list(code0):
            b.BY_ELEMENT[x.element] = x
        for e, x in tool0.items():
            b.BY_ELEMENT[e] = x
        _apply(v)

        frames = []
        for name in THEMES:
            t.set_theme(name)
            bg, fg = t.semantic_color("bg"), t.semantic_color("fg")
            frames.append(
                f'<figure class="frame" data-theme="{html.escape(name)}">'
                f"<figcaption>{html.escape(name)}</figcaption>"
                f'<pre style="background:{bg};color:{fg}">{_render_session(name)}</pre>'
                f"</figure>"
            )
        panels.append(
            f'<section class="variant" id="v-{v.key}" hidden>'
            f'<header class="vh"><h2>{v.key} — {html.escape(v.name)}</h2>'
            f"<p>{html.escape(v.blurb)}</p></header>"
            f'<div class="grid">{"".join(frames)}</div></section>'
        )

    b._MARKDOWN_BINDINGS, b._CODE_BINDINGS = md0, code0
    for x in list(md0) + list(code0):
        b.BY_ELEMENT[x.element] = x
    for e, x in tool0.items():
        b.BY_ELEMENT[e] = x

    keys = [v.key for v in VARIANTS]
    labels = {v.key: f"{v.key} — {v.name}" for v in VARIANTS}
    opts = "".join(f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in THEMES)

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>PROTOTYPE — lop colour use</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; background:#0d0d10; color:#e8e8ea;
        font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
 .banner {{ background:#3a2c12; border-bottom:1px solid #6b5423; color:#f4d58d;
            padding:11px 20px; font-size:13px; }}
 .banner b {{ color:#ffe9b0; }}
 .vh {{ padding:22px 20px 8px; }} .vh h2 {{ margin:0 0 5px; font-size:19px; }}
 .vh p {{ margin:0; color:#a2a2ab; max-width:78ch; }}
 .grid {{ display:grid; gap:18px; padding:14px 20px 130px;
          grid-template-columns:repeat(auto-fit,minmax(700px,1fr)); }}
 .frame {{ margin:0; border:1px solid #26262c; border-radius:8px; overflow:hidden; }}
 figcaption {{ padding:6px 12px; background:#17171b; color:#8f8f99; font-size:12px;
               font-family:ui-monospace,Menlo,monospace; }}
 pre {{ margin:0; padding:16px; overflow-x:auto; font-size:12.5px; line-height:1.5;
        font-family:ui-monospace,Menlo,"SF Mono",monospace; }}
 .bar {{ position:fixed; left:50%; transform:translateX(-50%); bottom:22px; display:flex;
         align-items:center; gap:14px; padding:10px 14px; background:#f4f4f6;
         color:#101014; border-radius:999px; box-shadow:0 8px 34px rgba(0,0,0,.6); z-index:99; }}
 .bar button {{ border:0; background:#101014; color:#fff; width:32px; height:32px;
                border-radius:50%; cursor:pointer; font-size:15px; }}
 .bar .lab {{ min-width:280px; text-align:center; font-weight:650; font-size:13px; }}
 .bar select {{ border-radius:6px; border:1px solid #c3c3cb; padding:5px 7px;
                font-size:12px; background:#fff; color:#101014; }}
</style>
<div class="banner">
 <b>PROTOTYPE — throwaway.</b> Real ANSI from the real binding table: prose,
 a code fence and a tool ledger, so colour is judged on a whole session rather
 than on headings alone. Today <b>61% of bindings are grey</b> while the median
 theme offers <b>6 legible hues</b>. <kbd>←</kbd>/<kbd>→</kbd> to switch.
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
function paint() {{
 KEYS.forEach((k,i)=>{{document.getElementById('v-'+k).hidden=(i!==idx);}});
 document.getElementById('lab').textContent=LABELS[KEYS[idx]];
 document.querySelectorAll('.frame').forEach(f=>{{
   f.style.display=(theme==='__all'||f.dataset.theme===theme)?'':'none';}});
 const u=new URLSearchParams(); u.set('variant',KEYS[idx]);
 if(theme!=='__all') u.set('theme',theme);
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
</script>
"""
    out = Path("/tmp/lop-colour-prototype.html")
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"prototype written: {p}")
    print(f"variants: {', '.join(v.key for v in VARIANTS)}")
    webbrowser.open(f"file://{p}")

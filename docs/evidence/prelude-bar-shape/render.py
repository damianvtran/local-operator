"""Build the before/after host pages for the prelude-bar-shape evidence.

The stored component is inlined, so this reproduces the frames without the operator's
attachment store. Run it from the checkout whose prelude you want to render:

    .venv/bin/python docs/evidence/prelude-bar-shape/render.py --out /tmp/after \
        --label after --widths 360,620,900

Then serve the output directory over localhost and screenshot each page; the page logs
`EVIDENCE {…}` with the real frame scroll size, SVG rect, bar rects and text rects.

`--document PATH` reuses an already-assembled document (that is how the "before" frames
were produced from `origin/main`'s prelude while the checkout carried the fix).
`--push none` is the operator's surface: no host theme message, so the frame takes its
own `prefers-color-scheme` fallback (dark on this machine). The page waits for the
400 ms fallback before measuring.

The 480 px iframe is the UI's HEIGHT_HINT_MAX. The paragraphs above the frame are the
column the graphic sits under, so alignment is judged against real text.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib

from local_operator.supplements.document import assemble_stored_component

BLOB = (
    '<data>{"d1":{"title":"answer","columns":["col1","USD / 1M tokens"],'
    '"rows":[["Input","$0.15"],["Output","$0.60"],'
    '["Cached input (prompt-cache read)","$0.015"]]}}</data>\n'
    '<div id="c"></div><script>LO.bar(document.getElementById("c"), "d1", '
    '{x: "col1", y: "USD / 1M tokens", horizontal: true, unit: "USD / 1M tokens"})</script>'
)

PARAGRAPH = (
    "it carries OpenRouter's free-variant rate limits. By contrast OpenRouter's "
    "minimax/minimax-m3 lists $0.30/$1.20, exactly MiniMax's own rate, so MiniMax's "
    "own pricing flows through there intact."
)

# The desktop UI's resolved dark tokens (THEME_VARIABLE_NAMES). Only used with --push.
DARK = {
    "--lo-canvas": "#121212",
    "--lo-surface": "#1a1a1a",
    "--lo-sunken": "#222222",
    "--lo-ink": "#e8e8e8",
    "--lo-ink-muted": "#b8b8b8",
    "--lo-ink-dim": "#8a8a8a",
    "--lo-hairline": "#333333",
    "--lo-border-control": "#888888",
    "--lo-accent": "#7c9cff",
    "--lo-accent-wash": "#1e2438",
    "--lo-success": "#4ade80",
    "--lo-warning": "#fbbf24",
    "--lo-danger": "#f87171",
    "--lo-info": "#38bdf8",
    "--font-sans": "system-ui,-apple-system,sans-serif",
    "--font-mono": "ui-monospace,monospace",
    "--s1": "#56B4E9",
    "--s2": "#E69F00",
    "--s3": "#2EC4A0",
    "--s4": "#E89AC7",
    "--s5": "#F0E442",
    "--s6": "#D9D9D9",
}


def page(width: int, document_html: str, push: bool) -> str:
    # A JSON string holding the document contains `</script>`; escape `<` so the outer
    # inline script cannot be terminated by the payload.
    payload = json.dumps(document_html).replace("<", "\\u003c")
    theme = (
        json.dumps(
            {
                "lo": "supplement-host",
                "t": "theme",
                "mode": "dark",
                "nonce": "evidence",
                "vars": DARK,
            }
        )
        if push
        else "null"
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>prelude evidence {width}px</title>
<style>
html,body{{margin:0;background:#121212;color:#e8e8e8;
  font:14px/1.5 system-ui,-apple-system,sans-serif}}
body{{padding:16px}}
.col{{width:{width}px}}
p{{margin:0 0 14px}}
iframe{{width:{width}px;height:480px;border:1px solid #333;display:block;background:#121212}}
#m{{margin-top:8px;font:11px/1.4 ui-monospace,monospace;color:#8a8a8a;white-space:pre}}
</style></head><body>
<div class="col">
<p>{html.escape(PARAGRAPH)}</p>
<iframe id="f"></iframe>
<div id="m">waiting for frame</div>
</div>
<script>
const doc = {payload};
const theme = {theme};
const f = document.getElementById("f");
f.onload = () => {{
  if (theme) f.contentWindow.postMessage(theme, "*");
  setTimeout(() => {{
    const d = f.contentDocument, svg = d.querySelector("svg");
    const col = document.querySelector(".col").getBoundingClientRect();
    const s = svg.getBoundingClientRect();
    const bars = [...d.querySelectorAll("rect")].map(r => {{
      const b = r.getBoundingClientRect(); return [Math.round(b.left), Math.round(b.top),
        Math.round(b.width), Math.round(b.height)];
    }});
    const labels = [...d.querySelectorAll("text")].map(t => {{
      const b = t.getBoundingClientRect();
      return [t.textContent, Math.round(b.left), Math.round(b.right), Math.round(b.top)];
    }});
    const out = {{
      width: {width},
      columnLeft: Math.round(col.left),
      frameClient: [f.clientWidth, f.clientHeight],
      frameScroll: [d.documentElement.scrollWidth, d.documentElement.scrollHeight],
      svg: [Math.round(s.left), Math.round(s.top), Math.round(s.width), Math.round(s.height)],
      bars, labels,
    }};
    document.getElementById("m").textContent = JSON.stringify(out);
    console.log("EVIDENCE " + JSON.stringify(out));
  }}, theme ? 600 : 800);
}};
f.srcdoc = doc;
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--widths", default="360,620,900")
    parser.add_argument("--document", type=pathlib.Path)
    parser.add_argument("--push", choices=("none", "dark"), default="none")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    document_html = (
        args.document.read_text(encoding="utf-8")
        if args.document
        else assemble_stored_component(BLOB)
    )
    (args.out / f"document-{args.label}.html").write_text(document_html, encoding="utf-8")
    for width in (int(part) for part in args.widths.split(",")):
        (args.out / f"{args.label}-{width}.html").write_text(
            page(width, document_html, args.push == "dark"), encoding="utf-8"
        )
    print(f"wrote {args.label} pages to {args.out}")


if __name__ == "__main__":
    main()

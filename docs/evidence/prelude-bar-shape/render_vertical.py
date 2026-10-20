"""Render the vertical 4-bar scroll-lock case with a host that sizes from the resize report."""

from __future__ import annotations

import argparse
import html
import json
import pathlib

from local_operator.supplements.document import assemble_stored_component

BLOB = (
    '<data>{"d1":{"title":"vertical","columns":["name","price"],'
    '"rows":[["alpha",4],["beta",7],["gamma",2],["delta",9]]}}</data>\n'
    '<div id="c"></div><script>LO.bar(document.getElementById("c"), "d1", '
    '{x: "name", y: "price"})</script>'
)
PARAGRAPH = (
    "it carries OpenRouter's free-variant rate limits. By contrast OpenRouter's "
    "minimax/minimax-m3 lists $0.30/$1.20, exactly MiniMax's own rate."
)


def page(width: int, document_html: str, seed: int) -> str:
    payload = json.dumps(document_html).replace("<", "\\u003c")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>vertical scroll lock {width}px seed {seed}</title>
<style>
html,body{{margin:0;background:#121212;color:#e8e8e8;font:14px/1.5 system-ui,sans-serif}}
body{{padding:16px}} .col{{width:{width}px}} p{{margin:0 0 14px}}
iframe{{width:{width}px;height:{seed}px;border:1px solid #333;display:block;background:#121212}}
#m{{margin-top:8px;font:11px/1.4 ui-monospace,monospace;color:#8a8a8a;white-space:pre-wrap;word-break:break-all}}
</style></head><body><div class="col"><p>{html.escape(PARAGRAPH)}</p>
<iframe id="f"></iframe><div id="m">waiting</div></div>
<script>
const doc = {payload};
const f = document.getElementById("f");
const reports = [];
window.addEventListener("message", (e) => {{
  const d = e.data;
  if (d && d.lo === "supplement" && d.t === "resize" && typeof d.h === "number" && d.h > 0) {{
    reports.push(d.h);
    f.style.height = Math.min(700, d.h) + "px";
  }}
}});
f.onload = () => {{
  setTimeout(() => {{
    const d = f.contentDocument, svg = d.querySelector("svg");
    const s = svg ? svg.getBoundingClientRect() : {{}};
    const out = {{
      width: {width}, seed: {seed}, reports,
      frameClient: [f.clientWidth, f.clientHeight],
      frameOffset: [f.offsetWidth, f.offsetHeight],
      frameScroll: [d.documentElement.scrollWidth, d.documentElement.scrollHeight],
      innerWidth: f.contentWindow.innerWidth,
      docClientWidth: d.documentElement.clientWidth,
      scrollbarPx: f.contentWindow.innerWidth - d.documentElement.clientWidth,
      svg: svg ? [Math.round(s.left), Math.round(s.top), Math.round(s.width), Math.round(s.height)] : null,
    }};
    document.getElementById("m").textContent = JSON.stringify(out);
    console.log("EVIDENCE " + JSON.stringify(out));
  }}, 1200);
}};
f.srcdoc = doc;
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases", default="620:320,810:373,1200:320")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    document_html = assemble_stored_component(BLOB)
    (args.out / f"document-{args.label}.html").write_text(document_html, encoding="utf-8")
    for case in args.cases.split(","):
        width, seed = (int(x) for x in case.split(":"))
        (args.out / f"{args.label}-{width}.html").write_text(
            page(width, document_html, seed), encoding="utf-8"
        )
    print(f"wrote {args.label} to {args.out}")


if __name__ == "__main__":
    main()

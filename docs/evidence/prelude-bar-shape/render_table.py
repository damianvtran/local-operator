"""Render a 30-row table through a host that clamps the iframe at 480 and scrolls to the end."""

from __future__ import annotations

import argparse
import html
import json
import pathlib

from local_operator.supplements.document import assemble_stored_component

ROWS = [[f"row {i:02d}", i] for i in range(1, 31)]
BLOB = (
    "<data>"
    + json.dumps(
        {"d1": {"title": "30 rows", "columns": ["name", "value"], "rows": ROWS}},
        separators=(",", ":"),
    )
    + "</data>\n"
    '<div id="c"></div><script>LO.table(document.getElementById("c"), "d1", {})</script>'
)
PARAGRAPH = (
    "it carries OpenRouter's free-variant rate limits. By contrast OpenRouter's "
    "minimax/minimax-m3 lists $0.30/$1.20, exactly MiniMax's own rate."
)
LIMIT = 480


def page(width: int, document_html: str, seed: int) -> str:
    payload = json.dumps(document_html).replace("<", "\\u003c")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>30-row table {width}px seed {seed}</title>
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
    f.style.height = Math.min({LIMIT}, d.h) + "px";
  }}
}});
f.onload = () => {{
  setTimeout(() => {{
    const d = f.contentDocument, body = d.body;
    const rows = d.querySelectorAll("tbody tr");
    const last = rows[rows.length - 1];
    const before = {{
      frameClient: [f.clientWidth, f.clientHeight],
      frameScroll: [d.documentElement.scrollWidth, d.documentElement.scrollHeight],
      scrollbarPx: f.contentWindow.innerWidth - d.documentElement.clientWidth,
      bodyClient: [body.clientWidth, body.clientHeight],
      bodyScroll: [body.scrollWidth, body.scrollHeight],
      bodyScrollbarPx: body.offsetWidth - body.clientWidth,
      bodyTop: body.scrollTop,
      lastBottom: last ? Math.round(last.getBoundingClientRect().bottom) : null,
    }};
    body.scrollTop = body.scrollHeight;
    setTimeout(() => {{
      const after = {{
        bodyTop: body.scrollTop,
        lastBottom: last ? Math.round(last.getBoundingClientRect().bottom) : null,
        lastVisible: last ? last.getBoundingClientRect().bottom <= f.clientHeight + 1 : false,
      }};
      const out = {{width: {width}, seed: {seed}, reports, before, after}};
      document.getElementById("m").textContent = JSON.stringify(out);
      console.log("EVIDENCE " + JSON.stringify(out));
    }}, 250);
  }}, 1200);
}};
f.srcdoc = doc;
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases", default="620:320,810:320,1200:320")
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

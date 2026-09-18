"""D5's evidence: does the transcript's renderer give a BARE url a span of its own?"""
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style
sys.path.insert(0, "/Users/damian/lo-wt/open-links-r3")
from local_operator.tui import bindings

console = Console(theme=bindings.markdown_theme(), width=200, no_color=False, force_terminal=False, highlight=False)
for label, text in (
    ("markdown link", "Read [the docs](https://a.test/x) now"),
    ("autolink", "Read <https://a.test/x> now"),
    ("bare url", "Read https://a.test/x now"),
):
    renderable = console.render_lines(Markdown(text), console.options) if False else None
    from rich.text import Text
    md = Markdown(text)
    segs = list(console.render(md, console.options))
    styles = [seg.style for seg in segs if seg.text.strip()]
    link_styles = [s for s in styles if s is not None and (s.link or s.color)]
    print(f"{label:14} segments={len(segs)} styled={len(link_styles)} "
          f"links={[s.link for s in link_styles]} colors={[s.color.name if s.color else None for s in link_styles]}")
print()
print("markdown.link      ->", bindings.style("markdown.link"))
print("markdown.link_url  ->", bindings.style("markdown.link_url"))

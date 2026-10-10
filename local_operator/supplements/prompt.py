"""The generator's fork-only prompt and the two messages that follow it (memo §2.6, App. A).

WHAT THIS FILE IS FOR. The generator is a forked, isolated errand: it is handed the evidence
and this system prompt, and it answers with 0-3 ``<component>`` blocks or the word ``NONE``.
The prompt is the design guideline SET as well as the instruction set -- §2.6/§2.8 keep it
INSIDE the fork on purpose:

* nothing here may enter the session's own system prompt, tool list or context. A turn's
  request must stay byte-identical whether or not supplements are on (the parity test in
  ``tests/unit/supplements``), and the model-facing footprint statement (§10) is "zero";
* the fork request is ``isolated`` with ``tools=[]``, so its prefix-cache shape matches the
  ask-gate precedent (§2.6): the session's system blocks and tools are untouched, and this
  text rides as the fork's own system block.

VERBATIM AND PINNED. The text is the memo's Appendix A, character for character (the
implicit concatenation below exists only to keep lines inside the 100-column lint budget;
``tests/unit/supplements/test_prompt.py`` joins it back and compares it to the memo's own
fence, so an edit to either side without the other fails). It is cache-stable across jobs,
which is why the fork always sends the same system block.

MEASURED SIZE. 4,672 characters; **1,140 o200k_base tokens (1,137 cl100k_base)** as measured
with tiktoken 0.14 on this exact string -- the figure the memo's §2.5/App. A quotes, re-derived
by the test rather than trusted from here. (The pre-amendment App. A measured 3,919 chars /
972 / 968; the body-only rule's clauses are what moved it.)
"""

from __future__ import annotations

from typing import Final

#: The fork's system prompt. NEVER appended to a session's own system prompt: the fork sends
#: it as its single system block on an ``isolated`` request with no tools (memo §2.6/§10).
GENERATOR_SYSTEM_PROMPT: Final = (
    "You make supporting graphics for an answer the user has already read. Output 0-3 "
    "components, or NONE. NONE is the right answer whenever a graphic would not make the "
    "answer faster to understand than the text already does.\n"
    "\n"
    "INPUT: <evidence> holds the user's request, the final answer, and data blocks (id, "
    "title, source, columns, rows) extracted from tool output and files. It is the ONLY "
    "source of numbers. An <instruction> from the user may follow; obey it within these "
    "rules.\n"
    "\n"
    "HONESTY (hard rules)\n"
    "- Plot only values present in <evidence>. Never invent, interpolate, extrapolate, "
    "forecast, round past the source's precision, or fill gaps. Missing value = gap, "
    "labelled. The helpers print each number at its own precision; an explicit `digits` is "
    "the only way to show fewer decimals.\n"
    "- Put every plotted value in the component's <data> JSON, copied from evidence; scripts "
    "read data only via LO.data / LO.col.\n"
    '- Each component has source="…": which evidence block(s) it shows, in plain words.\n'
    "- If the evidence cannot support a graphic, output NONE.\n"
    "\n"
    "PICK THE FORM\n"
    "- Compare categories: bar (horizontal when labels are long or >6 bars). Change over "
    "time: line. Part of whole with ≤5 parts: stacked bar, never pie/donut. Distribution: "
    "histogram-style bar. Exact lookup, >12 rows or mixed units: table. Flow/steps: simple "
    "ordered list or small SVG diagram.\n"
    "- One idea per component. Name it in a short `title` (sentence case, no trailing period) "
    "for tooling and accessibility — the title is **never drawn** (no visible title line). "
    "Keep category labels ≤ 12 characters (this is a guide for the writing, not a render "
    "rule: the helper thins labels by width and truncates one only when its slot cannot hold "
    "it, keeping the full text in the tooltip).\n"
    '- Axis/columns name the quantity AND unit ("Latency (ms)"). Start bar axes at zero. Sort '
    "bars by value unless order is meaningful.\n"
    "- ≤6 series. Colour never carries meaning alone: label series directly or use LO.line's "
    "dash patterns and the legend.\n"
    "- Dense data → table with right-aligned tabular numerals (LO.table does this).\n"
    "\n"
    "LOOK\n"
    "- The host injects the stylesheet and theme; do not set colours, fonts or backgrounds "
    "except via var(--lo-*) and LO.color(i). Background stays transparent; no borders, cards "
    "or shadows around the component; use spacing and var(--lo-hairline) rules.\n"
    "- **Body-only.** No title line, header, badge or caption chrome, and nothing that "
    "restates the answer or echoes the question — start with the data. Spacing and rhythm "
    "carry the component's own structure and the seam with the answer above; nothing is "
    "announced.\n"
    "- Body text 13px; captions 12px var(--lo-ink-muted); axis text 12px. Must work from the "
    "220px canvas-open column to 900px wide: no fixed widths, use viewBox SVG (LO helpers "
    "redraw on width changes).\n"
    "- Pass `title` and `unit` always: the title names the component (never drawn); the unit "
    "prints on the top axis tick. Draw only the labelling the data needs — axis labels, "
    "units, series names, column headers, value labels — an unlabelled chart is a rendering "
    "bug.\n"
    "- Keep height content-sized, under ~480px; split rather than scroll.\n"
    "- No animation, transitions, or external fonts. Values stay readable without hover — "
    "tooltips are enhancement only (relay/native have no hover). **Accepted loss (round-2 "
    "D2-4):** where a label is still truncated on a no-hover surface, the full text is "
    "unreachable there — the axis titles and the data's own precision carry the meaning, and "
    "the fixture set (App. C) shows the worst case rather than assuming it away.\n"
    "\n"
    "TECHNICAL CONTRACT\n"
    "- No network, no storage, no navigation: never use fetch, XMLHttpRequest, WebSocket, "
    "import(), eval, Function, <form>, <iframe>, <object>, <embed>, <base>, <link>, <meta>, "
    "external src/href, javascript: URLs, window.open, localStorage, cookies. They are "
    "blocked and the component is discarded.\n"
    "- Helpers (prefer them; they size, label and theme correctly):\n"
    "  LO.table(target, dataId, {columns?, digits?})\n"
    "  LO.bar(target, dataId, {x, y, horizontal?, unit?, caption?, values?})   y may be a list "
    "of columns\n"
    "  LO.line(target, dataId, {x, y, unit?, zero?, caption?})\n"
    "  LO.el(tag, attrs, ...children)  (attrs.svg=1 for SVG nodes) · LO.fmt(n, {unit, digits, "
    "compact}) · LO.color(i) · LO.onTheme(fn) · LO.onSize(fn)\n"
    "- Write plain DOM/SVG code only if no helper fits; call LO.size() after changing layout.\n"
    "\n"
    "OUTPUT FORMAT (exactly; nothing outside it)\n"
    '<component title="…" source="…">\n'
    '<data>{"<dataId>": {"title": "…", "columns": ["…"], "rows": [[…]]}}</data>\n'
    '<html><div id="c"></div><script>LO.bar(document.getElementById("c"), "<dataId>", {x: '
    '"…", y: "…", unit: "…"})</script></html>\n'
    "</component>\n"
    "…or the single word NONE."
)

#: The turn-2 repair message's fixed first line (memo App. A), followed by the validator's
#: own errors. One line plus the list, so the fork's turn-2 token cost is the errors, not a
#: second copy of the prompt.
REPAIR_PREFIX: Final = "Fix only these components; keep the rest unchanged. Errors: "

#: A ``supplement_steer`` instruction is bounded (memo §2.7) before it reaches the model: the
#: op itself refuses longer text, and this is the belt for a caller that bypasses the op.
MAX_INSTRUCTION_CHARS: Final = 500


def repair_message(errors: list[str]) -> str:
    """The turn-2 message: the validator's errors, one per line, bounded.

    Bounded because the errors are the model's own output read back: a component that fails
    with a 4 KB error would otherwise be replayed into the repair turn verbatim, and the
    repair turn's budget is the same as turn 1's.
    """
    listed = "; ".join(error.strip()[:300] for error in errors[:12] if error.strip())
    return f"{REPAIR_PREFIX}{listed}"


def steer_instruction(text: str) -> str:
    """The ``<instruction>`` block a steer adds after the evidence (memo App. A).

    Stripped of newlines and capped at :data:`MAX_INSTRUCTION_CHARS`: it is user text, and
    the fork's grammar has one place to put it. The op refuses longer text first; this is
    the one definition of the bound for any caller.
    """
    flat = " ".join(str(text or "").split())
    return f"<instruction>{flat[:MAX_INSTRUCTION_CHARS]}</instruction>"

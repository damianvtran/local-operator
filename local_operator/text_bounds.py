"""Head/tail elision of oversized text — one clipper, one marker.

Why this module exists
----------------------
Two different subsystems need to shorten a string and leave a visible scar where
the middle was removed: the TOOLS layer caps what a single tool run may put into
the prompt (``tools/builtin.py``), and the HARNESS caps what is re-sent per turn
(:mod:`local_operator.harness.replay_bound`). They must agree on the shape of
the cut — both ends snapped inward to a line boundary, one marker across the gap
— or the model meets two incompatible ways of being told "the middle is gone".

It cannot live in either one. The tools layer is barred from the harness (a
neutral runner imports the harness, never the tool registry), and the harness
cannot reach into ``tools/builtin.py`` for the same reason in reverse. So the
shared helper sits below both, next to ``ansi.py``, importing nothing.

Two semantics, deliberately two markers
---------------------------------------
The CLIPPER is shared; the MARKER TEXT is not. ``... [output truncated] ...``
means "the tool cut this output; the elided bytes are in a spill store you can
expand by handle", which is true only where a handle exists. A harness replay
bound cuts something the tool layer chose to keep, and its elided bytes live in
the durable transcript, which the model can neither see nor address — the
recovery route is to re-run the tool. Telling the model to expand a handle that
does not exist is worse than telling it nothing, so the two call sites keep
their own sentence and share everything else.
"""

from __future__ import annotations

__all__ = [
    "OUTPUT_TRUNCATION_MARKER",
    "clip_head_tail",
]

#: Marker written where the middle of a tool output was removed. Kept as a
#: public name because tests and the browser paths reference it; the text names
#: the recovery route rather than just announcing a loss.
OUTPUT_TRUNCATION_MARKER = "\n\n... [output truncated] ...\n\n"


def clip_head_tail(text: str, limit: int) -> tuple[str, str]:
    """``(head, tail)`` slices of ``text`` totalling at most ``limit`` chars.

    Both cuts snap INWARD to a line boundary, so neither end shows half a
    line. Half a line is not a cosmetic problem: a truncated ``File "x.py",
    line 12`` reads as a different path, and a model that acts on it edits the
    wrong file. When snapping would empty a side — one enormous line with no
    newline to snap to — the raw character slice is kept, because a fragment
    of the answer still beats none of it.
    """
    head_budget = limit // 2
    tail_budget = limit - head_budget

    head = text[:head_budget]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[: cut + 1]

    tail = text[len(text) - tail_budget :]
    cut = tail.find("\n")
    if 0 <= cut < len(tail) - 1:
        tail = tail[cut + 1 :]

    return head, tail

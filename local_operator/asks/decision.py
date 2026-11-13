"""Which verified decision a card answer makes — shared by every answer surface.

A question carrying a producer-stamped ``question_id`` (a Team Bridge request,
or a headless node's ``site:v1:`` browser approval) is decided ONLY through the
signed ``bridge_decide`` door; the plain ``ask_respond`` door refuses it. The
TUI and the desktop answer route both need to turn "the user picked Approve on
this card" into ``(ask_id, question_id, word)``, so the mapping lives here
rather than in either surface (it used to live in ``tui/app.py``, which the
server must not import).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def bridge_decision_target(
    row: Any, answers: Mapping[str, Sequence[str]]
) -> tuple[str, str, str] | None:
    """The ``(ask_id, question_id, word)`` a BRIDGE card answer decides.

    ``None`` for every ordinary ask: a decision exists only for a request whose
    question carries the producer's ``question_id`` AND whose picked value
    resolves to one of that question's own option ids. The word sent on the
    wire is the OPTION ID (``approve``/``deny``), never the drawn label — the
    door validates the word against the request's closed set, and a label is
    display copy that can be re-worded without touching the decision.

    Ambiguity is ``None`` rather than a guess: two picked request questions, or
    a value that maps to no option id, must fall back to the ordinary door and
    its refusal rather than decide something the human did not choose.
    """

    picked: list[tuple[str, str]] = []
    for question in row.questions:
        if not isinstance(question, Mapping):
            continue
        question_id = str(question.get("question_id") or "")
        if not question_id:
            continue
        selected = answers.get(str(question.get("id") or "")) or ()
        if len(selected) != 1:
            continue
        options = {
            str(option.get("label") or "").strip().lower(): str(option.get("id") or "")
            for option in (question.get("options") or ())
            if isinstance(option, Mapping)
        }
        value = str(selected[0]).strip()
        picked.append((question_id, options.get(value.lower()) or value.lower()))
    if len(picked) != 1:
        return None
    return (row.ask_id, picked[0][0], picked[0][1])

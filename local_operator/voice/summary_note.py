"""The turn note that asks the agent for a ``Spoken:`` paragraph.

While the voice slot is held, a prompt that was DICTATED carries one extra
paragraph telling the agent the user is listening and asking it to end its
reply with a short ``Spoken:`` paragraph that ``speakable`` then selects
(``docs/design/voice-spoken-summary.md``).

Model-facing prompt machinery, so it is English by design (AGENTS.md: model-
facing prompts are exempt from the keyed-copy rule), and ``Spoken:`` is a
protocol label that is never translated. The note rides the newest user
message rather than a system block, so the cached prompt prefix is untouched.

The wording is fixed and complete: ``harness.rows.user_row_text`` hides it from
the rendered row by matching these exact bytes (see :func:`note_spans`), so a
user who merely quotes part of it loses nothing from their row.
"""

from __future__ import annotations

#: One paragraph, parenthesised like the other harness notes that ride a turn.
VOICE_SUMMARY_NOTE = (
    "(The user dictated this message by voice and is listening to your reply. "
    "End your reply with a final paragraph that starts with `Spoken:`. Write at "
    "most two plain sentences for the ear: no code, no URLs, no markdown, no "
    "list. Say the point of the reply, in the language the user is speaking. "
    "Keep the word `Spoken:` exactly as written. Do this even when the reply "
    "is short.)"
)

_JOIN = "\n\n"


def with_note(text: str) -> str:
    """``text`` with the note appended as its own paragraph."""
    return f"{text}{_JOIN}{VOICE_SUMMARY_NOTE}"


def note_spans(text: str) -> list[tuple[int, int]]:
    """Every ``(start, end)`` span of a complete note in ``text``.

    The span starts at the separator :func:`with_note` writes when it is there,
    so the row does not keep a dangling blank line where the note stood.
    """
    spans: list[tuple[int, int]] = []
    start = text.find(VOICE_SUMMARY_NOTE)
    while start != -1:
        end = start + len(VOICE_SUMMARY_NOTE)
        begin = start - len(_JOIN) if text.startswith(_JOIN, start - len(_JOIN)) else start
        spans.append((max(begin, 0), end))
        start = text.find(VOICE_SUMMARY_NOTE, end)
    return spans

"""The off-the-record aside's model-facing instruction — one home, two seams.

WHY THIS MODULE EXISTS. ``ASIDE_PROMPT`` used to live in the TUI's aside widget
(``tui/widgets/aside_panel.py``) and was consumed only by ``tui/app.py``. A
prompt owned by a surface module is a prompt the OTHER surface forgets, and that
is exactly what happened: the desktop ``/asides`` route appended the user's raw
question and called the provider with no instruction at all, so on that path the
model was never told the request was off the record and never told it could not
call a tool. Those two facts are the whole reason the prompt exists, and both
were missing on the surface that shipped last — which is why tool calls appeared
there and not in the TUI.

So the prompt lives here, at SESSION level, beside :mod:`session.goal_loop`'s own
constants (the precedent for a session-owned prompt), and the REMOTE seam is what
applies it: :meth:`ServingSessionHandle.complete_aside` and
:meth:`TuiSessionHandle.complete_aside`, through :func:`wrap_aside_turns`.

:meth:`Session.complete_aside` deliberately does NOT wrap. Its in-process callers
own their own instruction — the TUI overlay supplies :data:`ASIDE_PROMPT` itself,
and the goal-loop judge supplies ``LOOP_JUDGE_PROMPT`` — so a wrap in the shared
primitive would hand the judge an aside instruction it must never receive (it is
judging the conversation, not stepping aside from it).
"""

from __future__ import annotations

from collections.abc import Sequence

from local_operator.harness.types import Message, TextContent

#: The prompt a side question is wrapped in. Three instructions, each earning
#: its line: OFF THE RECORD so the model does not treat the question as a new
#: task and start narrating a plan; TEXT ONLY because the request does carry the
#: live tool catalogue (it has to, to stay on the working turn's cache prefix)
#: and on Anthropic even ``tool_choice`` reads ``auto`` on the wire (see
#: ``Session.complete_aside``), so the prompt is the model-facing half of "calls
#: nothing" — a call it makes anyway is rejected and handed back to it as an
#: error, which is what ``Session.complete_aside``'s one bounded retry does; and
#: answer-from-context because the whole reason to ask here rather than in the
#: chat is that the agent already knows.
ASIDE_PROMPT = """<aside>
The user has stepped aside to ask you something about this session. This is OFF
THE RECORD: neither their question nor your answer joins the conversation, and
no work is being asked for. Answer from the context you already have, briefly
and directly, in prose. Answer in text only: tools are NOT AVAILABLE here, so a
tool or function call is rejected and returned to you as an error rather than
run — do not call one, do not propose a plan, and do not ask a follow-up
question. If your context does not answer it, say so plainly.
Question:
{question}
</aside>"""

#: Everything in :data:`ASIDE_PROMPT` before ``{question}`` — the fixed part a
#: wrapped turn always begins with. DERIVED from the template rather than
#: restated, so :func:`_carries_instruction` recognises a turn wrapped by this
#: module by construction and cannot drift when the prompt's wording changes.
_ASIDE_PREFIX = ASIDE_PROMPT.split("{question}", 1)[0]


def _carries_instruction(message: Message) -> bool:
    """Whether ``message`` is already a wrapped question (see :func:`_compose`)."""
    # Anchored at the start, ignoring the caller's own leading whitespace: the
    # wrapper is the FIRST thing in a wrapped turn, so a question that merely
    # mentions the marker somewhere in its body is still a question.
    return message.text.lstrip().startswith(_ASIDE_PREFIX.lstrip())


def wrap_aside_turns(turns: Sequence[Message]) -> list[Message]:
    """A copy of ``turns`` whose LAST user turn carries :data:`ASIDE_PROMPT`.

    The instruction is scaffolding for ONE REQUEST, never part of the exchange:
    the caller's list is left untouched, and the raw question is what an aside
    entry stores and returns — so the user never sees ``<aside>`` XML, and an
    adopted exchange contains the words they typed (``tui/app.py``'s
    ``_fork_aside_worker`` states the same rule for the fork path).

    Only the LAST turn is wrapped, because that is the new question. The earlier
    user/assistant pairs are the aside's own prior exchanges, and a wrapper there
    would tell the model to answer a question it has already answered.

    IDEMPOTENT, and that is the BELT under the ``aside_instruction`` flag the
    seams carry. The flag is the caller's declaration (see
    :meth:`ServingSessionHandle.complete_aside`); this is what makes a caller
    that pre-wrapped its own turns harmless even when it got the flag wrong —
    the failure mode is silent, because both parties believe they did the right
    thing and the model simply receives the instruction twice, nested. So the
    marker is checked as well as the flag, and the two cannot disagree without
    the turn list showing it. An already-instructed turn is returned unchanged
    rather than re-wrapped.

    Called at the REMOTE seam, once per aside — see the module docstring for why
    the shared primitive stays out of it, and why a client that supplies its own
    instruction must say so with the flag (or send a turn that already carries
    this one, which is the same statement and is honoured identically).
    """
    if not turns:
        return []
    last = turns[-1]
    if last.role != "user":
        # Nothing to instruct (a caller ending in an assistant turn is not
        # asking a new question). Returned unchanged rather than refused: this
        # seam must not be the thing that fails an aside the owner can still
        # answer.
        return list(turns)
    if _carries_instruction(last):
        return list(turns)
    # ``Message.user`` puts the text block first and attachments after it, so the
    # wrapped text replaces position 0 and any pixels ride along untouched.
    wrapped = last.model_copy(
        update={"content": [TextContent(text=_compose(last.text)), *last.content[1:]]}
    )
    return [*turns[:-1], wrapped]


def _compose(question: str) -> str:
    """The one place :data:`ASIDE_PROMPT` is formatted.

    Separate from the constant so the widget, the seam and the tests all render
    the same bytes; a second ``.format`` call site is a second place the template
    could drift from what the model is actually sent.
    """
    return ASIDE_PROMPT.format(question=question)

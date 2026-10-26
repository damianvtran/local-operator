"""The harness's own recovery/continuation prompts, and the predicate for one family.

WHY THIS MODULE EXISTS, AND WHY IT MUST STAY IMPORT-FREE. ``harness.rows`` decides
whether a user-role row is HARNESS CHROME rather than the operator's own words, and
both renderers ask it — the TUI's replay loop and the phone fold. To answer it needs
the exact strings the producers mint, and those lived beside the loop that writes
them (``harness.loop``) and the compaction that writes the compaction continuation
(``session.session``). Importing either to READ A STRING is what cost a cold desktop
open its first paint. Measured on this tree in a fresh interpreter, the first
``harness.rows.is_harness_chrome()`` call imported **231 ``local_operator`` modules**
(473 modules in all) and cost roughly a second of CPU on an idle host — 1.5-8.7 s
under fleet load — with ``session.session`` and ``tools.builtin`` among them, because
the constants sat inside the two heaviest modules in the application. Read from here,
the same call imports **5 ``local_operator`` modules** (about 82-86 in all) in tens to
a few hundred milliseconds, with neither of those two modules reachable at all. The
NAMESPACE COUNTS are the stable fact — a module graph, not a stopwatch — so they are
what this file is judged by; the millisecond ranges are context, and they move with
whatever else the host is doing.

So this file keeps the rule ``session/roster_sidecar.py`` keeps for the subagent
roster: STDLIB ONLY, no ``local_operator`` imports at all. A string a renderer has to
recognise does not get to drag the tool registry into every process that paints a row.

ONE SOURCE OF TRUTH, NOT A COPY. ``harness.loop`` re-exports every name below under
its historical spelling, and ``session.session`` re-exports the compaction continuation
as ``_CONTINUATION_PROMPT``; ``tests/unit/server/test_desktop_cold_enter_fastpath.py``
pins the identity, so a producer that drifts back to a private copy fails a test
instead of silently painting its own prompt as the operator's words.

WHAT DOES NOT LIVE HERE. The loop that INTERPOLATES these strings
(``_continuation_instruction``, which needs ``ToolCall``) and the recovery budgets
(``MAX_*``) stay in ``harness.loop``: this module answers "is this text the harness's
own words", nothing else.
"""

from __future__ import annotations

#: What the loop tells the model after dropping a tool call that arrived with
#: no name. The drop is invisible to the model otherwise -- there is no call to
#: attach a result to, and the transcript row cannot be replayed on any wire --
#: so this message is the only record of why the conversation continues. It
#: names the one repair the model can make: re-send the call with a name.
#: Harness chrome -- see ``harness.rows.harness_chrome_prompts``.
NAMELESS_CALL_RECOVERY_PROMPT = (
    "Harness recovery notice: a tool call in your last message arrived without "
    "a tool name. It was not executed, and its result is not in this "
    "conversation. Re-issue the call with the tool name included, or continue "
    "without it if it is no longer needed."
)


# This fixed prompt is also recognised on replay: the recovery is harness
# chrome, never an instruction attributed to the operator.
NO_PROGRESS_RECOVERY_PROMPT = (
    "Harness recovery notice: you have repeated the same tool action or short "
    "cycle with unchanged arguments and results, with no observable change. "
    "Re-read the current state and take a different approach; do not repeat "
    "the unchanged cycle. If progress requires an external change, report the "
    "blocker instead of continuing to poll without progress."
)


#: What the loop tells the model after a turn arrived with no text and no tool
#: calls on a clean stop. Like the nameless-call notice above it is the only
#: record of why the conversation continues: the empty message itself is
#: REMOVED from the wire history (an assistant turn with no content blocks is
#: illegal on the Anthropic wire and teaches a re-ask to repeat it), so without
#: this prompt the fact that the turn arrived empty would be invisible to the
#: model -- and, on a resumed session, to the reader. Harness chrome -- see
#: ``harness.rows.harness_chrome_prompts``.
WORDLESS_STOP_RECOVERY_PROMPT = (
    "Harness recovery notice: your last message arrived with no text and no "
    "tool calls, so there was nothing to act on and the turn would have ended "
    "silently. Produce your next tool call, or your final answer if the work "
    "is done."
)


#: What the loop tells the model after a turn was cut off at the output limit
#: with NOTHING produced, on a model the loop has no cheaper rung to retreat to
#: (the router's ``auto`` sentinel: the level is delegated to the server, so
#: ``_lower_effort`` declines — see its docstring). The empty message itself is
#: REMOVED from the wire history by that arm (an assistant turn with no content
#: blocks is illegal on the Anthropic wire, and replaying "I said nothing"
#: teaches the re-ask to say nothing), so without this prompt neither the model
#: nor a reader of a resumed session can see why the conversation continues.
#: Harness chrome -- see ``harness.rows.harness_chrome_prompts``.
LENGTH_STOP_RECOVERY_PROMPT = (
    "Harness recovery notice: your last message hit your output limit "
    "before anything visible was produced — no text and no tool calls arrived, "
    "so the turn would have ended here with nothing to show for it. Continue "
    "the work: produce your next tool call, or your final answer if the work is "
    "done, keeping each step well inside the output budget."
)


#: What the loop tells the model after the output limit cut its answer off
#: mid-prose. Second person, like its recovery siblings, and it names the one
#: instruction that matters: continue from the cut, do not restart. Persisted as
#: harness chrome (see ``harness.rows.harness_chrome_prompts``) so replay shows
#: the notice rather than attributing the harness's words to the user.
CONTINUE_TRUNCATED_ANSWER_PROMPT = (
    "Harness recovery notice: your last message was cut off mid-answer by your "
    "output limit — the rest of it was never sent. Continue from exactly where "
    "it stopped, without repeating what you already wrote, and finish the answer."
)


#: What the loop tells the model after the network cut its answer short.
#:
#: Phrased as an instruction about what the USER can see, because that is the
#: constraint the model cannot infer: the partial text is in the transcript and
#: has already been read, so restarting the answer would show it twice. It is a
#: user-role message rather than a bare trailing assistant turn on purpose —
#: see the call site: prefilling is rejected by current Claude models.
CONNECTIVITY_CONTINUATION_PROMPT = (
    "[system] Your previous response was cut off mid-sentence by a network "
    "interruption. The partial text above has already been shown to the user. "
    "Continue it seamlessly from exactly where it stopped — do not repeat any "
    "of it, do not restart, and do not apologise or mention the interruption."
)


#: What the loop tells the model when the interruption landed inside a TOOL CALL.
#:
#: The prompt above assumes the cut fell in a SENTENCE, and its instruction is
#: to continue that sentence seamlessly. A call still being dictated when the
#: socket died is truncated JSON: the continuation branch drops it — it cannot
#: be run, and cannot be replayed faithfully — so the model's own record of
#: having chosen that action is erased. Leaving it to the prose prompt then asks
#: the model to continue a sentence it never finished while the action it had
#: already decided on is silently lost, which is exactly what happened to the
#: aborted ``write`` call in the incident. This instruction carries the one fact
#: the drop erased, names the tool so a multi-call turn is unambiguous, and is
#: written to stand alone — a cut that produced no prose at all gets it on its
#: own — as well as to follow the prose prompt above.
CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT = (
    "[system] A tool call ({tools}) was aborted by the network interruption "
    "before it finished, so it never ran. If you still need that action, issue "
    "the call again from scratch."
)


#: The two fixed ends of `CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT`, split
#: around its ``{tools}`` hole. Declared from the template itself so the
#: recogniser below cannot fall out of step with the string it recognises.
_TOOL_CALL_INSTRUCTION_HEAD, _, _TOOL_CALL_INSTRUCTION_TAIL = (
    CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT.partition("{tools}")
)


def _is_continuation_tool_list(text: str) -> bool:
    """Whether ``text`` is a tools list ``_continuation_instruction`` would write.

    The producer writes ``", ".join(sorted({call.name for call in interrupted}))``,
    so the list is comma-and-space separated and every name is a single
    whitespace-free token. Checked rather than waved through because the
    predicate below is what keeps harness chrome out of the user's row: an
    operator message that merely QUOTES one of these instructions — a whole
    line, a multi-line paste, a wrapping that put a newline inside the tools
    list — must stay theirs. A tool name this grammar rejects costs only the
    pre-fix behaviour (the row paints), never a swallowed operator turn.
    """
    names = text.split(", ")
    return all(name and not any(char.isspace() for char in name) for name in names)


def is_connectivity_continuation_instruction(text: str) -> bool:
    """Whether ``text`` is EXACTLY one of the instructions this module mints.

    :func:`_continuation_instruction` returns three shapes, and the front ends
    have to recognise all three: the prose prompt alone, the tool-call prompt
    alone (a cut that produced no prose), and the two joined by a single space
    (the incident's shape — prose with a call still being dictated).

    Lives HERE, beside the strings it matches, rather than in the shared
    ``harness.rows`` decision that calls it, for two reasons. The composed shape
    cannot be enumerated: the tool-call half interpolates the aborted calls'
    names, so a table of exact strings would have to list every possible tools
    list. And the shape is a property of the producer, not of a row — the row
    decision is "does the harness mint this text", and this is how the harness
    answers that about its own words.

    Deliberately not a loose prefix test on ``[system] ``: that prefix is also
    minted by the approval and question gates (see
    ``rows._HARNESS_NOTICE_HEADS``), and an operator is free to type it. What is
    matched is the fixed sentence each half opens with, plus the exact tail the
    tool-call half closes with.
    """
    stripped = text.strip()
    if stripped == CONNECTIVITY_CONTINUATION_PROMPT:
        return True
    if stripped.startswith(CONNECTIVITY_CONTINUATION_PROMPT + " "):
        stripped = stripped[len(CONNECTIVITY_CONTINUATION_PROMPT) + 1 :]
    if not (
        stripped.startswith(_TOOL_CALL_INSTRUCTION_HEAD)
        and stripped.endswith(_TOOL_CALL_INSTRUCTION_TAIL)
    ):
        return False
    tools = stripped[
        len(_TOOL_CALL_INSTRUCTION_HEAD) : len(stripped) - len(_TOOL_CALL_INSTRUCTION_TAIL)
    ]
    return _is_continuation_tool_list(tools)


#: What the session re-asks after a compaction, so the turn resumes where the summary
#: stopped. Lives here because ``harness.rows`` needs to recognise it: the session's own
#: copy is a re-export (``session.session._CONTINUATION_PROMPT``), not a second literal.
CONTINUATION_AFTER_COMPACTION = (
    "The conversation context was just compacted into a summary. "
    "Continue the task from where it left off."
)

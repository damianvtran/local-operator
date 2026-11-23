"""Legacy clearance verdict grammar and standalone fork retry copy.

Queued asks no longer call this fork or interpret its verdict as an answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

#: The reason line's wire bound, applied at parse time (design §2.3: "truncated
#: to 200 chars"). One constant because the prompt, the parser and the tests all
#: quote the same number.
REASON_MAX_CHARS = 200

#: The fixed frame. Bullets, not prose (design §2.3): the ask block below it is
#: the model's own data, and the whole message stays ~15 fixed lines.
#:
#: The raise bullet carries the genuine-user list VERBATIM and states the
#: destructive/irreversible rule and the "any question" atomicity — those three
#: are the requirements table's row 5 and the settled choice that clear/resolve
#: are per-ASK, never per-question.
CLEARANCE_PROMPT = """<ask-clearance>
A forked check, OFF THE RECORD: nothing here reaches the user or the conversation.
The agent is about to raise this ask. Decide, for the ask as a whole:

The ask:
{ask_block}

Verdicts:
- clear — the recommended option is plainly the best choice here, for ALL questions
  (with no recommendation, the context plainly settles them). The agent proceeds.
- resolve — not clear from context, but not the user's call either: the agent should
  resolve it with ONE `task` subagent and proceed.
- raise — the decision is genuinely the user's: critical go/no-go decisions;
  opinionated architectural choices; missing access or credentials; a preference,
  name, or roster only they can state. A destructive or irreversible action is
  NEVER cleared. If ANY question is the user's, raise.

Answer EXACTLY two lines, nothing else:
VERDICT: clear|resolve|raise
REASON: <one line, <=200 chars>
</ask-clearance>"""

#: A verdict line, FULL-LINE matched, decoration tolerated (star/bold/quote/heading
#: markers and stray backticks around it — reasoning models decorate). The memo's
#: grammar is ``[*_>#\s]*VERDICT:\s*(clear|resolve|raise)[*_`\s]*``;
#: implemented with ONE documented widening — see the note below — the decoration
#: class also runs between the colon and the word, so ``**VERDICT:** clear``
#: (the memo's own §4 example) matches rather than falling to fail-open.
#: Case-insensitive; ``fullmatch`` rather than a search, because a sentence that
#: merely MENTIONS the word "verdict" mid-line is deliberation, not a verdict.
_VERDICT_LINE = re.compile(
    r"[*_>#\s]*VERDICT:[*_`\s]*(clear|resolve|raise)[*_`\s]*",
    re.IGNORECASE,
)

#: The reason line, matched (not full-line): everything after the label is the
#: reason, bounded below. A MISSING reason is NOT a parse failure (design §2.3):
#: it degrades the diagnostic, never the route. The same one-step widening as
#: the verdict line — decoration between the colon and the text — plus a
#: trailing-decoration strip, so `**REASON:** a thing**` reads `a thing`.
_REASON_LINE = re.compile(r"[*_>#\s]*REASON:[*_`\s]*(.*)", re.IGNORECASE)
_REASON_TAIL = re.compile(r"[*_`]+\s*$")

#: The correction paired against a rejected tool call when the clearance answer
#: is a bare call and nothing else (the aside's one bounded retry, design §2.1).
#: Clearance-specific rather than the aside's own sentence because that one
#: says "answer the user's question" and "prose" — neither is true of a gate
#: request, whose whole contract is the two-line verdict.
CLEARANCE_TOOL_CALL_REFUSAL = (
    "This was an off-the-record clearance check: tool calls are not available "
    "here, so this call was rejected and nothing ran. The call was refused "
    "rather than answered because this request needs TEXT ONLY. "
    "Answer EXACTLY two lines, nothing else:\n"
    "VERDICT: clear|resolve|raise\n"
    "REASON: <one line, <=200 chars>"
)


def _collapse(text: Any) -> str:
    """Whitespace-collapsed string form for the legacy question prompt.

    ``" ".join(split())`` rather than ``str.strip()``: the design's rule
    (design §2.5) is that a re-raise whose only change is whitespace is the
    SAME content, so interior runs collapse too.
    """
    return " ".join(str(text or "").split())


def format_question_block(questions: Sequence[Any]) -> str:
    """Render the ask's own questions/options for the gate message.

    Read from the VALIDATED question objects (``AskQuestion``), so the
    recommended option is the hoisted ``options[0]`` the user-facing surfaces
    would show — the fork must see the ask as the user would have. Numbered
    options keep the model's answer space addressable ("Q1", "option 2") if it
    deliberates, and ``(recommended)`` marks every recommended row (the hoisted
    ``recommended_rows``: position 0 for one pick, the first k for a
    multi-select's set) only when the model actually gave a recommendation (a
    question without one must not grow a marker).

    Secret questions never reach here (the gate's step 3 refuses the whole ask),
    so no masking concern rides this path.
    """
    blocks: list[str] = []
    for index, question in enumerate(questions, start=1):
        head = f"Q{index}: {_question_text(question)}"
        if bool(getattr(question, "multi", False)):
            head += " [multi-select]"
        lines = [head]
        # The set, not the int: `recommended` alone names only the first row.
        recommended = frozenset(getattr(question, "recommended_rows", ()) or ())
        for option_index, option in enumerate(getattr(question, "options", ()) or ()):
            label = _option_text(option, "label")
            description = _option_text(option, "description")
            line = f"  {option_index + 1}. {label}"
            if option_index in recommended:
                line += " (recommended)"
            if description:
                line += f" — {description}"
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _question_text(question: Any) -> str:
    """The question's own line, whitespace-collapsed (it is one wire row)."""
    return _collapse(getattr(question, "question", ""))


def _option_text(option: Any, field: str) -> str:
    """One option field, whitespace-collapsed like the question."""
    return _collapse(getattr(option, field, ""))


def build_clearance_prompt(questions: Sequence[Any]) -> str:
    """The one user-role message a clearance request appends (design §2.3).

    A function, not a second ``.format`` call site: the prompt template has one
    home and this is how every caller renders it, so the bytes the model is
    sent cannot drift between the session and the tests.
    """
    return CLEARANCE_PROMPT.format(ask_block=format_question_block(questions))


def parse_verdict(text: str | None) -> str | None:
    """The routing verdict this answer carries, or ``None`` (design §2.3).

    Full-line matches only, and the LAST one wins: a reasoning model that
    deliberates in the open ("it could be clear ... but actually VERDICT:
    raise") states its conclusion at the END, so an earlier match is its
    thinking rather than its answer. No match — including an empty, truncated
    or non-conforming answer — is ``None``, and the caller routes that to the
    unchanged enqueue (fail-open, design §2.2).
    """
    if not text:
        return None
    verdict: str | None = None
    for line in str(text).splitlines():
        match = _VERDICT_LINE.fullmatch(line.strip())
        if match:
            verdict = match.group(1).lower()
    return verdict


def parse_reason(text: str | None) -> str:
    """The bounded ``REASON:`` line, or ``""`` when the answer carries none.

    A missing reason must NOT force fail-open (design §2.3): the verdict is the
    only routing input, and the reason is a diagnostic embedded in the note.
    Like the verdict, the LAST match wins for the same deliberation reason.
    """
    if not text:
        return ""
    reason = ""
    for line in str(text).splitlines():
        match = _REASON_LINE.match(line.strip())
        if match:
            reason = _REASON_TAIL.sub("", match.group(1)).strip()
    return reason[:REASON_MAX_CHARS]

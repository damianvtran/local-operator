"""The ask gate's model-facing half: prompt, verdict grammar, fingerprints, notes.

WHY THIS MODULE EXISTS. The ask gate (``docs/design/ask-gate.md``) runs one
forked, off-the-record request before a queued ``ask`` reaches the operator,
asking a single question — *are the recommended options clear?* — and routes on
its strict two-line answer. Everything that decision touches that is PURE lives
here, one home per decision, exactly as :mod:`session.aside` owns the aside
prompt:

* the gate message (:data:`CLEARANCE_PROMPT`, :func:`build_clearance_prompt`) —
  the genuine-user list rides it VERBATIM, because that list is where "never
  miss a decision the operator would want to own" is enforced;
* the verdict grammar (:func:`parse_verdict`, :func:`parse_reason`) — the only
  routing input, so it is unit-pinned rather than trusted;
* the honor rule's digest (:func:`fingerprint`) — the content normalisation a
  re-raise is recognised by, in ONE place so the record and the consult cannot
  disagree;
* the two decision-point notes (:func:`clearance_note`) the model reads when an
  ask is diverted — no user-attribution anywhere ("the receipt is not consent"),
  the re-raise licence, and the ONE-subagent instruction.

The module is deliberately free of session, provider and host imports (stdlib
only): it is consumed by ``Session._gate_ask`` and by tests, and a module-scope
import of the session would make every unit of the grammar drag the tree.

NOTHING HERE TALKS TO A MODEL. The request itself is
``Session.complete_clearance``; this module only builds the one message it
carries and reads the one answer it returns (design §2.1/§2.3).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

#: The fingerprints a session remembers before evicting the oldest (design §2.5
#: LRU). An eviction costs one redundant check for a re-raised ask — never a
#: lost ask, because the skip direction is ENQUEUE — so the cap is generosity,
#: not a correctness boundary.
GATE_FINGERPRINT_CAP = 64

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

#: The two decision-point notes (design §2.4). Each MUST name that no question was
#: put to the user (no "the user said" register anywhere), carry the instruction,
#: and license the re-raise explicitly. The resolve note carries the
#: ONE-subagent instruction. Final copy may be tuned in a later PR for wording;
#: these three properties are the contract.
_CLEAR_NOTE = (
    "[Ask clearance] No question was put to the user. A forked check read the "
    "conversation and found your recommended option plainly the best path — "
    "proceed with it and state the assumption where it matters. The user was NOT "
    "asked and did not answer. If new information makes this genuinely their "
    "call, call `ask` again: a re-raise of this same question reaches them "
    "without another check."
)
_RESOLVE_NOTE = (
    "[Ask clearance] No question was put to the user. The answer is not clear "
    "from context and it is yours to resolve: spawn ONE `task` subagent with the "
    "relevant expertise (scout/research, architect, designer/UX — as the question "
    "needs), decide on its answer, and carry on. The user was NOT asked and did "
    "not answer. If after that the decision is still genuinely theirs, call "
    "`ask` again: a re-raise of this same question reaches them without another "
    "check."
)


def _collapse(text: Any) -> str:
    """Whitespace-collapsed string form — the fingerprint's normal form.

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
    deliberates, and ``(recommended)`` marks position 0 only when the model
    actually gave a recommendation (``recommended is not None`` after the
    validator's hoist; a question without one must not grow a marker).

    Secret questions never reach here (the gate's step 3 refuses the whole ask),
    so no masking concern rides this path.
    """
    blocks: list[str] = []
    for index, question in enumerate(questions, start=1):
        head = f"Q{index}: {_question_text(question)}"
        if bool(getattr(question, "multi", False)):
            head += " [multi-select]"
        lines = [head]
        recommended = getattr(question, "recommended", None)
        for option_index, option in enumerate(getattr(question, "options", ()) or ()):
            label = _option_text(option, "label")
            description = _option_text(option, "description")
            line = f"  {option_index + 1}. {label}"
            if recommended is not None and option_index == recommended:
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


def fingerprint(questions: Sequence[Any]) -> str:
    """The sha256 digest of an ask's USER-FACING content (design §2.5).

    Per question, in order: the whitespace-collapsed question text plus the
    option list as ``[label, description]`` pairs, canonical-JSON encoded
    before hashing. ``id``, ``recommended`` and ``multi`` are EXCLUDED on
    purpose — the user-facing content is question + options, and a re-raise
    with a changed recommendation or id is the SAME content, so it must hit
    the recorded fingerprint rather than run a second check. Option ORDER is
    significant (the model's ranking is content; the hoist already moved the
    recommendation to the top).

    Secret questions never gate (the caller refuses before fingerprinting) and
    are deliberately NOT special-cased here: a digest computed over their
    (empty) options would be near-constant, and the one place that must never
    forget the refusal is the callable, not the hash.
    """
    content: list[Any] = []
    for question in questions:
        options = [
            [_option_text(option, "label"), _option_text(option, "description")]
            for option in getattr(question, "options", ()) or ()
        ]
        content.append([_question_text(question), options])
    canonical = json.dumps(content, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def clearance_note(verdict: str, reason: str = "") -> str:
    """The decision-point note the tool result carries instead of a receipt.

    ``clear`` and ``resolve`` only — a ``raise`` produces NO note (it enqueues
    and the model gets today's receipt). The check's ``REASON`` is embedded
    after the instruction when the answer carried one, bounded by
    :func:`parse_reason`; it explains the verdict to the model and keeps the
    diversion auditable.
    """
    base = _RESOLVE_NOTE if verdict == "resolve" else _CLEAR_NOTE
    if reason:
        return f"{base}\n(Check's reason: {reason})"
    return base

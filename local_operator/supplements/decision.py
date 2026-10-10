"""The supplement decision: two typed questions on the existing classification grammar (memo §2.3).

NO VENDOR CHANGE. The classification layer already speaks everything this needs
(``classification/types.py``): a ``choice`` question answers one option id plus a probability
map, a ``noul`` question answers one probability. C0's fixtures pin both shapes. This module
only BUILDS the two questions and READS the two answers; the cascade (Radient -> TypeSafe ->
OpenRouter), the breaker, the credential memo and the keep-alive client all belong to the
session's one shared ``ClassificationService`` (``decide`` takes its guards from there).

* **Files** -- one ``choice`` question ``supplement_files`` over ``f1..fN`` (N <= 12) plus an
  explicit ``none``. "Which of these files" is a multi-select, and a decision model cannot
  emit a list, so the featured SET is derived from the probability map (see
  :func:`featured_ids`). Option text is the basename, the size, the tool and the call's own
  one-line intent -- NEVER a directory: the relative directory is the field that would carry
  a client's or project's name to a third-party vendor, and it does not help judge "is this
  the deliverable?" (memo round-1 S-R8). All of it is harness-derived text; the one
  model-controlled string, the file's name, is sanitised and clipped.
* **Graphics** -- one ``noul`` question ``supplement_graphics`` at :data:`GRAPHICS_THRESHOLD`.
  Precision over recall: a "yes" below 0.7 is a "no", and a "yes" the pre-filter's structured-
  data signal does not back is OVERRULED (the model has no rows, only shapes, so it cannot see
  whether the data exists; the harness can).

FAIL-OPEN, AND WHAT OPEN MEANS HERE. A ``None`` answer (disabled, circuit open, timeout, no
leg, schema rejection) degrades to the no-vendor heuristic, not to an error: tier-1 files (a
``write`` call the turn made) of a deliverable kind, by recency, capped. Graphics NEVER run
without a vendor -- a session-model fallback for that question is ~100x the decision cost and
exactly the "default-on spends money" surprise the welcome copy warns about (memo §2.3 Step 3).

THE LEDGER. The decision is a classification HTTP call, not a provider stream request, so it
does not pass through the request ledger; its spend is the INFO line
``ClassificationService.decide`` already logs. The names are still reserved in ``policy``.

NOT HERE: the generator, the validator, progress events, the ops -- later lanes.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

from local_operator.ansi import sanitize_prompt_line
from local_operator.supplements.candidates import MAX_OFFERED, Candidate
from local_operator.supplements.evidence import Evidence

logger = logging.getLogger(__name__)

FILES_QUESTION_ID: Final = "supplement_files"
GRAPHICS_QUESTION_ID: Final = "supplement_graphics"
NONE_OPTION: Final = "none"
#: ``SupplementDecision.vendor`` when no vendor answered the files question and the no-vendor
#: heuristic chose; ``"none"`` when no question was asked at all.
HEURISTIC_VENDOR: Final = "heuristic"

#: A non-top option joins the featured set only above this floor AND above half the top
#: probability (memo §2.3).
FILE_PROB_FLOOR: Final = 0.25
#: ``none`` winning with at least this probability means no files at all.
NONE_WINS_P: Final = 0.5
#: Graphics go ahead iff the answer is at least this AND the structured signal is true.
GRAPHICS_THRESHOLD: Final = 0.7

#: State bounds (memo §2.3 State), cut with the monitors' marker so the model knows a preview
#: is incomplete. Duplicated, not imported: ``monitors.classify`` is import-cheap but its
#: ``TRUNCATION_MARKER`` is the pinned spelling this module must match.
USER_TEXT_CHARS: Final = 1_500
ANSWER_TEXT_CHARS: Final = 3_000
_OPTION_NAME_CHARS: Final = 80
_OPTION_INTENT_CHARS: Final = 80

FILES_INSTRUCTIONS: Final = (
    "The assistant just finished answering a user. Pick the ONE file, if any, that is a "
    "DELIVERABLE the user asked for or would want pointed out: a report, document, export, "
    "spreadsheet, image or other finished product written for a reader to open. Edits to "
    "source code or configuration, logs, caches, intermediate or scratch files, and files "
    "that were only read are NOT deliverables. When unsure, answer none."
)
FILES_NONE_DESCRIPTION: Final = (
    "none of these is a finished deliverable (code edits, scratch, logs, nothing to open)"
)
GRAPHICS_INSTRUCTIONS: Final = (
    "The assistant just finished answering a user. Would a chart or figure help the user "
    "understand the numbers? Answer with a HIGH probability only when the answer or its tool "
    "output actually shows a table or series of three or more comparable measurements whose "
    "shape, ranking or trend is faster to see than to read. Answer LOW for a single value, a "
    "version or count, a short list, a prose-only answer, or data that was only talked about "
    "and never shown."
)
GRAPHICS_CRITERIA: Final[dict[str, str]] = {
    "true": "a chart would let the user see a ranking, trend or comparison at a glance",
    "false": "no figure helps: one or two values, prose, or no data actually shown",
}


@dataclass(frozen=True)
class Decision:
    """What the decision settled, in the form the row and the runner need."""

    featured: tuple[Candidate, ...] = ()
    more: tuple[Candidate, ...] = ()
    #: Display path -> probability, for the offered options only (the row's ``files_p``).
    files_p: dict[str, float] = field(default_factory=dict)
    graphics: bool = False
    graphics_p: float = 0.0
    #: The answering vendor, :data:`HEURISTIC_VENDOR`, or ``"none"`` (nothing was asked).
    vendor: str = "none"

    @property
    def empty(self) -> bool:
        return not self.featured and not self.graphics


def _bound(text: str, limit: int) -> str:
    # The monitors' marker, so one spelling of "cut here" reaches every classification state.
    marker = " …[truncated]"
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


def option_text(candidate: Candidate) -> str:
    """One option's description: ``name (kind, 4.1 KB) written by write: intent``. No directory."""
    name = sanitize_prompt_line(candidate.name, limit=_OPTION_NAME_CHARS) or "file"
    size = candidate.size_bytes
    shown = (
        f"{size} B"
        if size < 1024
        else f"{size / 1024:.1f} KB" if size < 1_048_576 else (f"{size / 1_048_576:.1f} MB")
    )
    text = f"{name} ({candidate.kind}, {shown}) {candidate.why}"
    intent = sanitize_prompt_line(candidate.intent, limit=_OPTION_INTENT_CHARS)
    return f"{text}: {intent}" if intent else text


def option_ids(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    return {f"f{index + 1}": item for index, item in enumerate(candidates[:MAX_OFFERED])}


def files_state(user_text: str, answer_text: str) -> str:
    return (
        "USER REQUEST:\n"
        + _bound(user_text.strip(), USER_TEXT_CHARS)
        + "\n\nFINAL ANSWER:\n"
        + _bound(answer_text.strip(), ANSWER_TEXT_CHARS)
    )


def graphics_state(user_text: str, answer_text: str, evidence: Evidence) -> str:
    """The graphics state: request, answer, and the evidence SHAPES -- never rows (memo §2.3)."""
    shapes = "\n".join(f"- {d.title}: {d.shape()}" for d in evidence.datasets) or (
        "- numbers inside the answer text"
    )
    return files_state(user_text, answer_text) + "\n\nDATA SHOWN THIS TURN:\n" + shapes


def files_question(options: dict[str, Candidate]) -> Any:
    from local_operator.classification.types import Question

    criteria = {key: option_text(item) for key, item in options.items()}
    criteria[NONE_OPTION] = FILES_NONE_DESCRIPTION
    return Question(
        id=FILES_QUESTION_ID, kind="choice", instructions=FILES_INSTRUCTIONS, criteria=criteria
    )


def graphics_question() -> Any:
    from local_operator.classification.types import Question

    return Question(
        id=GRAPHICS_QUESTION_ID,
        kind="noul",
        instructions=GRAPHICS_INSTRUCTIONS,
        criteria=dict(GRAPHICS_CRITERIA),
    )


def qualifying_ids(
    chosen: str, probabilities: dict[str, float], *, offered: Sequence[str]
) -> list[str]:
    """The featured option ids from one ``choice`` answer (memo §2.3 Featured set).

    The chosen id plus every option whose probability is >= :data:`FILE_PROB_FLOOR` and >= half
    the top probability, minus ``none``. ``none`` winning with >= :data:`NONE_WINS_P` is an
    empty set. Ordered chosen-first, then by probability, restricted to ids that were actually
    OFFERED -- a vendor cannot invent one. NOT capped: the caller splits the list into the
    featured set and the "N more" disclosure, because "more" means files the decision
    QUALIFIED but the cap held back -- never files it rejected (showing those would be spam).
    """
    valid = set(offered)
    if chosen == NONE_OPTION and probabilities.get(NONE_OPTION, 1.0) >= NONE_WINS_P:
        return []
    probabilities = {k: v for k, v in probabilities.items() if k in valid or k == NONE_OPTION}
    top = max(probabilities.values(), default=0.0)
    chosen_ok = chosen in valid
    if not probabilities:
        return [chosen] if chosen_ok else []
    cutoff = max(FILE_PROB_FLOOR, top / 2.0)
    qualifying = [key for key, p in probabilities.items() if key in valid and p >= cutoff]
    if chosen_ok and chosen not in qualifying:
        qualifying.append(chosen)
    qualifying.sort(key=lambda key: (key != chosen, -probabilities.get(key, 0.0), key))
    return qualifying


def apply_graphics(answer: Any, evidence: Evidence) -> tuple[float, bool]:
    """The graphics half's ONE rule: threshold AND evidence (memo §2.3).

    A pure function so the evidence overrule can be tested against an answer that ARRIVES
    anyway (a service whose answer was cached before the signal moved): the caller never asks
    without the signal, but the rule must not depend on the caller having asked correctly.
    """
    if answer is None or not isinstance(getattr(answer, "value", None), (int, float)):
        return 0.0, False
    probability = round(float(answer.value), 4)
    return probability, probability >= GRAPHICS_THRESHOLD and evidence.structured


def heuristic_files(candidates: Sequence[Candidate]) -> list[Candidate]:
    """The no-vendor fallback: tier-1 (written this turn) deliverable files, newest first."""
    chosen = [c for c in candidates if c.tier == 1 and c.deliverable]
    chosen.sort(key=lambda c: (-c.order, c.path))
    return chosen


async def _ask(service: Any, state: str, question: Any) -> Any:
    """One ``decide`` call that cannot raise (cancellation excepted): ``None`` = no answer."""
    decide = getattr(service, "decide", None)
    if decide is None:
        return None
    try:
        return await decide(state=state, question=question)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — fail-open: a broken vendor is "no answer"
        logger.debug("supplements: decide raised", exc_info=True)
        return None


async def decide(
    service: Any | None,
    *,
    user_text: str,
    answer_text: str,
    candidates: Sequence[Candidate],
    evidence: Evidence,
    want_files: bool,
    want_graphics: bool,
    max_featured: int,
) -> Decision:
    """Ask the two questions (concurrently) and fold the answers into a :class:`Decision`.

    ``want_graphics`` must already include "a generator exists" -- asking a paid question whose
    "yes" nothing can act on is waste, and C1a ships no generator. The structured-signal
    overrule is applied here, so a model "yes" over no evidence is a "no".
    """
    options = option_ids(candidates) if want_files else {}
    ask_files = bool(options)
    ask_graphics = want_graphics and evidence.structured
    vendor = "none"
    files_task = (
        _ask(service, files_state(user_text, answer_text), files_question(options))
        if ask_files and service is not None
        else None
    )
    graphics_task = (
        _ask(service, graphics_state(user_text, answer_text, evidence), graphics_question())
        if ask_graphics and service is not None
        else None
    )
    pending = [task for task in (files_task, graphics_task) if task is not None]
    results = list(await asyncio.gather(*pending)) if pending else []
    files_answer = results.pop(0) if files_task is not None else None
    graphics_answer = results.pop(0) if graphics_task is not None else None

    featured: list[Candidate] = []
    files_p: dict[str, float] = {}
    if ask_files:
        if files_answer is not None and isinstance(files_answer.value, str):
            vendor = str(getattr(service, "vendor_name", None) or "unknown")
            probabilities = {
                key: float(value) for key, value in dict(files_answer.probabilities).items()
            }
            files_p = {
                options[key].path: round(value, 4)
                for key, value in probabilities.items()
                if key in options
            }
            qualified = [
                options[key]
                for key in qualifying_ids(files_answer.value, probabilities, offered=list(options))
            ]
        else:
            vendor = HEURISTIC_VENDOR
            qualified = heuristic_files(candidates)
        cap = max(1, max_featured)
        featured, more = qualified[:cap], tuple(qualified[cap:])
    else:
        more = ()

    graphics_p, graphics = apply_graphics(graphics_answer, evidence)
    if graphics_answer is not None and vendor == "none":
        vendor = str(getattr(service, "vendor_name", None) or "unknown")
    return Decision(
        featured=tuple(featured),
        more=more,
        files_p=files_p,
        graphics=graphics,
        graphics_p=graphics_p,
        vendor=vendor,
    )

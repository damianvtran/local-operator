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

THE EGRESS BOUNDARY. Every string this module hands a vendor -- the two state texts and the
option text -- passes :func:`_egress` at the point it is composed (memo §4's one-statement
egress boundary; round-1 security S-R8): a literal pass over the turn's own path spellings
first, then the conservative shape pass. What is scrubbed, what survives and why is stated
on :func:`_egress`.

NOT HERE: the generator, the validator, progress events, the ops -- later lanes.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Sequence

from local_operator.ansi import sanitize_prompt_line
from local_operator.redaction_shapes import scrub_shapes
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


#: One path component on its way to a vendor: a run of word characters (``.``/``@``/``%``/
#: ``+``/``~``/``-`` included), never whitespace. A confident match STOPS at whitespace; a
#: whitespace run that is really inside a path is closed by the carried-run pass
#: (:func:`_carried`), never guessed at inside this token class.
_TOKEN: Final = r"[\w.@%+~-]+"

#: An absolute-path occurrence on its way to a vendor: a ``file://`` URI (any case, RFC 8089),
#: or a ``/``- or ``~/``-rooted POSIX path (``candidates._PROSE_PATH``'s own roots). The lead
#: refuses a match that is really a URL tail, a fraction or a relative path (``https://host/x``,
#: ``24/7`` and ``and/or`` all survive) while taking a ``:``/``-`` lead (``path:/x``, ``to-/x``,
#: a forward-slash drive ``C:/x``); the final component must carry a character, so a bare
#: separator cannot match. Windows backslash roots (``C:\...``, ``\\server\share\...``) are a
#: recorded non-scrub -- see :func:`_egress`.
_PATH_SPAN: Final = re.compile(r"(?<![\w./~])(?i:file://)?~?/(?:" + _TOKEN + r"/)*" + _TOKEN)

#: The lead a carried run may cross: any whitespace except the line break (so a run never
#: leaves its sentential fragment -- the state's sections are line-bounded), with one lone
#: ``/`` in front for the separator a component can leave behind (``/Users/damian/
#: Client Work/reports``).
_TAIL_LEAD: Final = re.compile(r"/?[^\S\n\r]+")

#: The same whitespace class, between the tokens of a carried run.
_INLINE_WS: Final = re.compile(r"[^\S\n\r]+")

#: One slash-attached component chain: a token plus any ``/token`` continuations, with a
#: lone trailing ``/`` (a component named as a directory: ``Corp/``) taken with it.
_CHAIN: Final = re.compile(_TOKEN + r"(?:/" + _TOKEN + r")*/?")

#: One token, matched at a position -- the carried-run scan's step.
_TOKEN_AT: Final = re.compile(_TOKEN)


def _base_of(path: str) -> str:
    """One path spelling -> its base name (a ``file://`` scheme, any case, rides with it)."""
    if path[:7].lower() == "file://":
        path = path[7:]
    return path.rstrip("/").rsplit("/", 1)[-1].rstrip(".")


def _carried(text: str, start: int) -> tuple[int, str] | None:
    """One whitespace run plus the slash-attached chain it reaches, ``None`` for prose.

    ``start`` sits just past a confident match. The run is CROSSED -- the span extended
    through the chain, whose final component becomes the base name -- only when it reaches
    a slash-attached component (a token immediately followed by ``/``). The boundary the
    tests pin:

    * plain tokens before that component are crossed with it: over-redaction is preferred
      to a tail leak, so ``Acme Corp Ltd/q3.pdf`` and ``Client Work/reports`` reduce whole,
      and a multi-run path (``Ltd/Sub Dir/q3.pdf``) crosses once per run;
    * the whitespace class is every separator except the line break, so tab- and
      NBSP-separated runs (a pasted path) cross exactly as spaces do;
    * a token that STARTS a rooted path (``/x``, ``~/x``) ends the scan: two paths in
      prose are never joined, and ``see /x and /y`` stays ``see x and y`` the same way
      ``see /x and then /y`` stays itself;
    * a run that never reaches a slash-attached component is prose and is not crossed at
      all (``and more`` survives); a ``word/word`` token inside the run is itself
      slash-attached, so ``see /x and/or y`` reads the ``and/`` as a continuation and
      becomes ``see or y`` -- the one continuation reading, pinned exactly by test.
    """
    lead = _TAIL_LEAD.match(text, start)
    if lead is None:
        return None
    pos = lead.end()
    while True:
        if text.startswith("/", pos) or text.startswith("~/", pos):
            return None
        token = _TOKEN_AT.match(text, pos)
        if token is None:
            return None
        if text.startswith("/", token.end()):
            chain = _CHAIN.match(text, pos)
            assert chain is not None  # the token just matched is the chain's first component
            return chain.end(), _base_of(chain.group(0))
        ws = _INLINE_WS.match(text, token.end())
        if ws is None:
            return None
        pos = ws.end()


def _span_end(text: str, match: re.Match[str]) -> tuple[int, str]:
    """A confident match as the span to reduce: its base name, carried runs extending it.

    Each carried run replaces the base with its chain's final component, so the whole span
    -- match start through the LAST crossed component -- drops to that final base name
    alone: "drop the fragment to its safe form", no directory component of an ambiguous
    run can ride, at the cost of crossing the plain words on the way to it.
    """
    end, base = match.end(), _base_of(match.group(0))
    while True:
        step = _carried(text, end)
        if step is None:
            return end, base
        end, base = step


def _reduce(text: str) -> str:
    """Reduce every rooted-path occurrence, each to the base name of its widest span."""
    out: list[str] = []
    pos = 0
    for match in _PATH_SPAN.finditer(text):
        if match.start() < pos:
            continue  # a carried run already took this position
        end, base = _span_end(text, match)
        out.append(text[pos : match.start()])
        out.append(base)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _egress(text: str, literals: Iterable[tuple[str, str]] = ()) -> str:
    """The decision payload's egress boundary: every string a vendor receives passes here.

    One call per composition point (the two state builders and the option builder below),
    because the decision is a paid third-party call that would otherwise carry the
    operator's final answer and directory layout off the machine on every eligible turn
    (memo §4's egress statement; round-1 security S-R8). Three passes, in this order:

    * THE LITERAL PASS (primary; round-4 R4-1/R4-2/R4-3). Every full path spelling the
      turn's OWN evidence holds (:func:`_egress_literals` over each candidate's
      ``absolute`` and display ``path``) is replaced by its base name -- exact string
      replacement, longest needle first -- so the payload can never carry a directory the
      turn itself just wrote, whatever characters (spaces, tabs, NBSP) sit inside the
      path. The absolute form is a SEARCH KEY here and nothing else; only the base name it
      maps to can appear in a payload. Spellings the evidence does not hold fall to the
      shape pass below.
    * THE SHAPE PASS (secondary, conservative). Any ``~/``- or ``/``-rooted absolute path
      still present reduces through :func:`_reduce`: a confident match (:data:`_PATH_SPAN`,
      which stops at whitespace) drops to its base name, and a whitespace run that reaches
      a slash-attached chain is crossed so the WHOLE span drops to that chain's final base
      name -- over-redaction preferred to a tail leak. ``Acme Corp Ltd/q3.pdf`` and
      ``Client Work/reports`` reduce whole; tab- and NBSP-separated runs cross the same
      way; ``see /x and /y`` never crosses. The crossing's exact boundary, the scan's
      stopping rules and the one recorded ``and/or`` continuation reading are on
      :func:`_carried`.
    * The minimised text then passes the project's one credential-shape table
      (:func:`local_operator.redaction_shapes.scrub_shapes`), the same pass classification
      §6 requires for outbound state: a credential the table recognises is masked wherever
      it stands -- the prose, an intent line, a base name -- and never dropped, so the mask,
      not the value, is what a vendor receives. The table's own negative carries through
      too: a vendor-prefix token carrying a dotted artifact tail (a ``glpat-`` token with a
      ``.md`` tail) is NOT masked -- dotted artifact names must survive -- so as a base name
      it arrives whole, and as a directory component it leaves with the directory.

    Deliberately NOT scrubbed, so a future fix does not "repair" them into over-masking:
    relative paths and bare directory names in prose (indistinguishable from ordinary text
    without taking ``and/or``, ``24/7`` and URLs with them); a slash glued to a word character
    is a URL tail or a relative path and stays whole (the lead refuses it -- ``example.com/x``,
    ``clients/acme-corp/q3.pdf``); Windows root spellings (``C:\\...``, ``\\\\server\\share\\...``)
    are recorded rather than silently missed -- the backslash is the escape character of every
    embedded command and code fragment, so the form needs its own false-positive review (the
    forward-slash drive form reduces today, through the relaxed ``:`` lead); e-mail
    addresses in prose --
    not credentials, and the shape corpus pins ``user@example.com`` as a must-survive
    negative (an e-mail sitting in a credential POSITION, a password value, is masked as
    that credential); and the harness-authored instruction and criteria constants.

    One RECORDED may-leak remains, stated as its must-survive/may-leak pair: a run that
    never reaches a slash-attached continuation -- a spaced component (``.../Acme Corp
    and more``), or one a punctuation mark stops early (``.../Acme, Corp/q3.pdf``) -- is
    indistinguishable from prose without eating arbitrary words (every ``and``/``then``/
    ``more`` is a plain token), so the confident match reduces to its own base name and
    the run's remaining tokens ride -- may-leak: the component tail (``Corp``);
    must-survive: the prose after it (``and more``). The pair is pinned by test rather
    than left as an inference.
    """
    for needle, replacement in sorted(literals, key=lambda pair: -len(pair[0])):
        if needle and needle != replacement:
            text = text.replace(needle, replacement)
    return scrub_shapes(_reduce(text))


def _egress_literals(candidates: Sequence[Candidate]) -> tuple[tuple[str, str], ...]:
    """The turn's own path spellings for :func:`_egress`: full path -> base name.

    Both spellings a :class:`Candidate` holds -- the resolved ``absolute`` and the display
    ``path`` -- because the answer may carry either; a spelling that IS the base name is
    dropped as a no-op. These are SEARCH KEYS: never journaled, never sent anywhere, and
    only the base names they map to can appear in a payload.
    """
    pairs: dict[str, str] = {}
    for candidate in candidates:
        for spelling in (candidate.absolute, candidate.path):
            if spelling and spelling != candidate.name:
                pairs.setdefault(spelling, candidate.name)
    return tuple(pairs.items())


def option_text(candidate: Candidate, *, literals: Iterable[tuple[str, str]] = ()) -> str:
    """One option's description: ``name (kind, 4.1 KB) written by write: intent``. No directory.

    The name and the intent line are the option's two model/user-controlled halves, so the
    composed text passes the egress scrub -- the turn's own path spellings first, then the
    shape pass: a recognised credential spelling or a path in the intent is masked or
    reduced, while a benign base name arrives byte-identical (the decision needs it to
    judge the option).
    """
    name = sanitize_prompt_line(candidate.name, limit=_OPTION_NAME_CHARS) or "file"
    size = candidate.size_bytes
    shown = (
        f"{size} B"
        if size < 1024
        else f"{size / 1024:.1f} KB" if size < 1_048_576 else (f"{size / 1_048_576:.1f} MB")
    )
    text = f"{name} ({candidate.kind}, {shown}) {candidate.why}"
    intent = sanitize_prompt_line(candidate.intent, limit=_OPTION_INTENT_CHARS)
    return _egress(f"{text}: {intent}" if intent else text, literals)


def option_ids(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    return {f"f{index + 1}": item for index, item in enumerate(candidates[:MAX_OFFERED])}


def _state_text(user_text: str, answer_text: str) -> str:
    """The bounded request/answer block both states share, BEFORE the egress scrub."""
    return (
        "USER REQUEST:\n"
        + _bound(user_text.strip(), USER_TEXT_CHARS)
        + "\n\nFINAL ANSWER:\n"
        + _bound(answer_text.strip(), ANSWER_TEXT_CHARS)
    )


def files_state(
    user_text: str, answer_text: str, *, literals: Iterable[tuple[str, str]] = ()
) -> str:
    """The files state: the bounded texts, scrubbed for egress as they are composed."""
    return _egress(_state_text(user_text, answer_text), literals)


def graphics_state(
    user_text: str,
    answer_text: str,
    evidence: Evidence,
    *,
    literals: Iterable[tuple[str, str]] = (),
) -> str:
    """The graphics state: request, answer, and the evidence SHAPES -- never rows (memo §2.3).

    The dataset titles ride in the shapes and are turn data too, so the whole composed
    string passes the same egress scrub.
    """
    shapes = "\n".join(f"- {d.title}: {d.shape()}" for d in evidence.datasets) or (
        "- numbers inside the answer text"
    )
    return _egress(
        _state_text(user_text, answer_text) + "\n\nDATA SHOWN THIS TURN:\n" + shapes, literals
    )


def files_question(
    options: dict[str, Candidate], *, literals: Iterable[tuple[str, str]] = ()
) -> Any:
    from local_operator.classification.types import Question

    criteria = {key: option_text(item, literals=literals) for key, item in options.items()}
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
    literals = _egress_literals(candidates)
    options = option_ids(candidates) if want_files else {}
    ask_files = bool(options)
    ask_graphics = want_graphics and evidence.structured
    vendor = "none"
    files_task = (
        _ask(
            service,
            files_state(user_text, answer_text, literals=literals),
            files_question(options, literals=literals),
        )
        if ask_files and service is not None
        else None
    )
    graphics_task = (
        _ask(
            service,
            graphics_state(user_text, answer_text, evidence, literals=literals),
            graphics_question(),
        )
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

"""Questions built from the candidate roster, and the answers mapped back onto it.

QUESTION DESIGN, AND WHY IT IS SHAPED THIS WAY
==============================================

Three questions, one per resource kind — a ``choice`` over that kind's
candidates plus an explicit ``none``. That is the whole set, and the shape is
the result of what the model can and cannot do:

* **One question per kind, not one per candidate.** A decision model answers a
  typed question; it cannot emit a list. Asking "which skill?" once per KIND is
  therefore the cheapest way to get a per-kind pick, and the alternative — a
  ``noul`` per candidate ("does this one apply?") — costs one question block
  each: 12 candidates × 3 kinds would be 36 question objects, i.e. 36
  instructions strings, where the choice form needs 3. Measured on the live
  alpha route (2026-09-18): one ``choice`` question with 2 options and one
  ``noul`` question together cost 380 input tokens, while test calls show the
  per-question instructions are the dominant fixed cost of a small request.

* **An explicit ``none`` option.** Without it the model must pick something, and
  a wrong recommendation costs a line of context plus, worse, a nudge toward a
  resource that does not fit. The renderer also says "ignore the rest" (§7), so
  ``none`` is belt-and-braces rather than the only guard — but it is the cheap
  one, and it is what makes "the model found nothing relevant" an answer we can
  tell apart from "we sent a bad roster".

* **No ``noul`` per kind.** §12 records this as an open question: "is one of
  these actually needed at all?" would cost ~30-40 input tokens per kind for a
  gate the renderer already applies in one line, and the layer is additive-only
  (§12) so there is no behavioural difference to buy. Skipped deliberately, with
  the tokens named, rather than dropped silently.

* **Option descriptions are the candidate's NAME plus its harness-owned
  description** (§6). The rubric has to be self-contained: a decision model
  judges against the option text alone, and the measured PEP-tier result the
  contract cites (31/31 with the rubric in the descriptions, 17/31 without)
  is the whole reason the description is repeated into the option instead of
  relying on the state's candidate lines. The text comes from the same
  ``candidate_line`` helper the state uses, so the two cannot drift.

MEASURED COST (live, OpenRouter alpha route, 2026-09-18)
=======================================================

Three questions over 5 options total with a ~430-char state: **519 input
tokens, 102 output tokens, $0.0000218** reported by the vendor. Deriving the
price from three calls (519/102 → 0.000021798; 380/60 → 0.00001596;
421/42 → 0.000017682) gives **$0.042 per Mtok of input and output costed at
zero** — the route bills input tokens only, which is also what §10 has the
Radient route mirror.

ESTIMATE for the default full roster (not measured, arithmetic shown so the
estimate can be checked): ``maxCandidates`` 12 per kind = 36 option lines plus
3 ``none`` options; at the §5 line cap of 120 chars that is ≤4.7k chars of
options, and the state cap is 6k chars — together ≲10.7k chars ≈ 2.7k input
tokens ≈ **$0.00012 per user message**, i.e. about 1.2 cents per hundred-turn
session. That is the number the PR's cost arithmetic should quote; the measured
figure above is the floor, not the typical case.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from local_operator.classification.context import (
    Candidate,
    ResourceKind,
    candidate_line,
    select_candidates,
    setting_int,
)
from local_operator.classification.types import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    Question,
    SkipReason,
)

#: ``values.classification.maxRecommendations`` — how many resources a single
#: message may be told about.
DEFAULT_MAX_RECOMMENDATIONS = 3


def max_recommendations(settings: Mapping[str, Any] | None) -> int:
    """``values.classification.maxRecommendations``, defensively read."""
    return setting_int(settings, "maxRecommendations", DEFAULT_MAX_RECOMMENDATIONS)


#: The option every choice question ends with. Its presence is what makes "no
#: recommendation" an answerable outcome rather than an unpicked one.
NONE_OPTION = "none"

#: The text the ``none`` option carries. Fixed, harness-owned prose (§6) — no
#: config text and no remote text may reach an option description.
NONE_OPTION_TEXT = "None of these fits this request"

#: Question ids, one per kind, fixed strings because the session's disabled-shape
#: set and the answer mapping both key off them. Stable across sessions on
#: purpose: a question id that varied per call would make "disable this shape"
#: a per-question accident instead of a per-shape decision.
QUESTION_ID_PREFIX = "recommend_"

#: Kinds in the order their questions are asked (skills first: they are the
#: most specific kind, and a model reading top-down spends its attention there).
QUESTION_KIND_ORDER: tuple[ResourceKind, ...] = ("skill", "guide", "mcp")

_OPTION_ID_SAFE = re.compile(r"[^A-Za-z0-9_.:-]+")


@dataclass(frozen=True)
class RecommendationRequest:
    """One classification pass: the user's message, optional context, the roster."""

    user_message: str
    context: str | None
    candidates: Sequence[Candidate]
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS


@dataclass(frozen=True)
class Recommendation:
    """The advisory outcome. Empty is a NORMAL result, and ``skipped`` says why.

    ``block`` is what a caller injects; ``""`` means "inject nothing", so a
    disabled, unavailable, timed-out or empty outcome changes the prompt not at
    all (§7's byte-identical degradation requirement).
    """

    resources: tuple[Candidate, ...] = ()
    block: str = ""
    vendor: str | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0
    skipped: SkipReason | None = None
    #: Set by a CALLER whose turn stopped waiting before the answer arrived, so
    #: this recommendation is being delivered by a LATER message than the one it was
    #: computed for. Not produced by anything in this package — the service answers
    #: inside its own deadline and does not know what the turn did with the
    #: wait — but the notice renderer needs it, because saying "for this message"
    #: about advice the model was asked for one message ago is a claim the line
    #: cannot support (design round 1, D2).
    late: bool = False


@dataclass(frozen=True)
class QuestionPlan:
    """The questions, plus the option-id → :class:`Candidate` map the answers need.

    The plan is what makes mapping an answer back a lookup instead of a parse:
    option ids are the only thing a decision model echoes, and they are
    sanitized candidate names (see :func:`option_id`), so the original
    ``Candidate`` — with its real ``resource_url`` — has to be carried
    alongside them or the recommendation would have to rebuild a URL from a
    normalized name.
    """

    questions: tuple[Question, ...]
    #: question id -> option id -> the candidate that option stands for.
    options: Mapping[str, Mapping[str, Candidate]] = field(default_factory=dict)
    #: question id -> the resource kind it asks about.
    kinds: Mapping[str, ResourceKind] = field(default_factory=dict)

    def kind_of(self, question_id: str) -> ResourceKind | None:
        """The resource kind a question id asks about, or ``None``."""
        return self.kinds.get(question_id)

    def chose(self, answer: Answer) -> Candidate | None:
        """The candidate an answer picked, or ``None`` for ``none``/unknown."""
        if answer.kind != "choice" or not isinstance(answer.value, str):
            return None
        options = self.options.get(answer.id)
        if not options:
            return None
        return options.get(answer.value)


def option_id(name: str, taken: set[str]) -> str:
    """A vendor-safe option id for a candidate name, unique within its question.

    Option ids travel as JSON object keys, so the id has to survive whatever a
    skill, guide or server is called — including names with spaces, slashes or
    unicode. Sanitizing is lossy, so the ``taken`` set is consulted: two
    different resources that normalize to the same id must not silently share an
    option, or one of them could never be recommended and the other would be
    recommended under the wrong name.
    """
    cleaned = _OPTION_ID_SAFE.sub("_", name).strip("_")
    candidate_id = cleaned or "option"
    if candidate_id != NONE_OPTION and candidate_id not in taken:
        taken.add(candidate_id)
        return candidate_id
    index = 2
    while f"{candidate_id}_{index}" in taken or f"{candidate_id}_{index}" == NONE_OPTION:
        index += 1
    unique = f"{candidate_id}_{index}"
    taken.add(unique)
    return unique


def build_questions(
    candidates: Sequence[Candidate],
    *,
    limit: int | None = None,
    skip_question_ids: frozenset[str] = frozenset(),
) -> QuestionPlan:
    """Build one choice question per kind that has candidates.

    ``skip_question_ids`` is how a session that hit a schema error stops asking
    a question shape the vendor rejected (see
    :class:`~local_operator.classification.service.ClassificationService`). A
    kind with no candidates — or whose question is skipped — gets no question at
    all rather than an empty choice, because an empty choice would ask the model
    to pick from nothing.
    """
    roster = select_candidates(candidates, limit)
    by_kind: dict[ResourceKind, list[Candidate]] = {}
    for candidate in roster:
        by_kind.setdefault(candidate.kind, []).append(candidate)

    questions: list[Question] = []
    options: dict[str, dict[str, Candidate]] = {}
    kinds: dict[str, ResourceKind] = {}
    for kind in QUESTION_KIND_ORDER:
        kind_candidates = by_kind.get(kind, [])
        if not kind_candidates:
            continue
        question_id = f"{QUESTION_ID_PREFIX}{kind}"
        if question_id in skip_question_ids:
            continue
        taken: set[str] = {NONE_OPTION}
        criteria: dict[str, str] = {}
        kind_options: dict[str, Candidate] = {}
        for candidate in kind_candidates:
            identifier = option_id(candidate.name, taken)
            # The option text is the candidate's name plus its harness-owned
            # description, and it is the SAME text the state's candidate line
            # carries: a model that sees two different descriptions for one
            # resource is being asked to reconcile a contradiction (§6).
            criteria[identifier] = candidate_line(candidate)
            kind_options[identifier] = candidate
        criteria[NONE_OPTION] = NONE_OPTION_TEXT
        kind_label = {"skill": "skill", "guide": "guide", "mcp": "MCP server"}[kind]
        questions.append(
            Question(
                id=question_id,
                kind="choice",
                instructions=(
                    f"Which {kind_label}, if any, would actually help with the request above? "
                    f"Choose '{NONE_OPTION}' when none of them fits, and do not pick one merely "
                    "because it exists."
                ),
                criteria=criteria,
            )
        )
        options[question_id] = kind_options
        kinds[question_id] = kind
    return QuestionPlan(questions=tuple(questions), options=options, kinds=kinds)


def collect_resources(
    response: DecisionResponse,
    plan: QuestionPlan,
    *,
    max_recommendations: int,
) -> tuple[Candidate, ...]:
    """Map answers back to candidates, best first, capped.

    Ordering is by the model's own confidence in the pick, then by question
    order (skills before guides before MCP servers) as a stable tie-break — a
    model that is 51% sure about a skill and 95% about a guide should spend the
    first line of the block on the guide. Resources the model explicitly
    declined (``none``) contribute nothing, and a duplicate pick across two
    kinds cannot happen because a candidate belongs to exactly one kind.
    """
    picks: list[tuple[float, int, Candidate]] = []
    for index, question in enumerate(plan.questions):
        answer = response.answers.get(question.id)
        if answer is None:
            continue
        chosen = plan.chose(answer)
        if chosen is None:
            continue
        confidence = (
            answer.confidence if answer.confidence is not None else _top_probability(answer)
        )
        picks.append((confidence, index, chosen))
    picks.sort(key=lambda item: (-item[0], item[1]))
    ordered = [candidate for _, _, candidate in picks]
    return tuple(ordered[: max(0, max_recommendations)])


def _top_probability(answer: Answer) -> float:
    """A confidence stand-in for answers that carry only a distribution.

    ``noul`` answers have no confidence field (measured: ``{"type":"noul",
    "noul":0.59}``), so the distribution's own maximum is the closest thing to
    one. It is used ONLY for ordering — nothing downstream reads it as a
    confidence, so a missing number cannot turn into a claim.
    """
    return max(answer.probabilities.values(), default=0.0)


def render_block(resources: Sequence[Candidate]) -> str:
    """The advisory block, verbatim per §7, or ``""`` when there is nothing to say.

    Phrasing rules are part of the contract and are load-bearing: advisory ("may
    help", "ignore the rest"), never imperative, never exclusive, never a claim
    that a resource is authoritative for the turn. A wrong recommendation must
    cost a line of context, not a wrong action (§6: this layer may not gate a
    capability).

    No trailing newline, matching ``skills/index.py:render_block`` — the caller
    owns the join, and a block that appends its own newline makes every call
    site's concatenation look subtly different.
    """
    if not resources:
        return ""
    lines = [
        "<resource_recommendations>",
        "These may help with this request — read the ones that actually fit, ignore the rest:",
    ]
    lines.extend(f"- {candidate.resource_url}" for candidate in resources)
    lines.append("</resource_recommendations>")
    return "\n".join(lines)


def build_decision_request(
    plan: QuestionPlan,
    state: dict[str, Any],
) -> DecisionRequest:
    """Wrap a built state and plan into the vendor call's request object.

    Kept as its own function so the state builder's output and the question
    builder's output meet in exactly one place — the alternative is every caller
    assembling a request inline, which is one more site that has to know both
    shapes.
    """
    return DecisionRequest(state=state, questions=plan.questions)


__all__ = [
    "DEFAULT_MAX_RECOMMENDATIONS",
    "NONE_OPTION",
    "NONE_OPTION_TEXT",
    "QUESTION_ID_PREFIX",
    "QUESTION_KIND_ORDER",
    "QuestionPlan",
    "Recommendation",
    "RecommendationRequest",
    "ResourceKind",
    "build_decision_request",
    "build_questions",
    "collect_resources",
    "max_recommendations",
    "option_id",
    "render_block",
]

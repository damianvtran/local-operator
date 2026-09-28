"""The classifier gate (§8): one typed question per changed monitor.

WHAT THIS MODULE OWNS
=====================

The gate's monitor-side half, and nothing else:

* the typed question (§8.3, verbatim — the instructions and criteria text is
  the measured request in the contract's §17; do not paraphrase it, the
  numbers and the two suppressed classes' descriptions were measured
  together);
* the bounded state (:func:`bounded_state`; ``classifyMaxChars``, with the
  truncation-marker floor its docstring states);
* the fork's mapping (§8.4: which choice ids suppress, and under which
  counters keys, :func:`suppressed_counter`);
* the adapter that turns the session's shared classification seam into the
  scheduler's ``classify`` callback (:func:`monitor_classify`).

The fork's EXECUTION — suppress + count, or deliver — lives in the scheduler
(``monitors/scheduler.py``), because the counters and the delivery are its
state. The call's GUARDS live in ``ClassificationService.decide``, because
the breaker, the client and the credential memo are that instance's state.

WHY EVERY CLASSIFICATION IMPORT HERE IS LAZY
============================================

This module is imported by ``monitors.scheduler``, which the SESSION imports
for every session — so a module-scope ``from local_operator.classification...
import`` would execute the classification package's ``__init__`` (the full
cascade, vendors, httpx) on the import graph of every session, including ones
that never arm a monitor, and the package's cold import is ~1.8 s of
cumulative time that the layer deliberately pays only when it is switched on
(``session_factory._attach_classification``). So :func:`materiality_question`
imports ``Question`` inside the function — reached only once a gate seam
actually answers — and the truncation marker is a local duplicate of
``classification.context.TRUNCATION_MARKER`` rather than an import, because
:func:`bounded_state` can run with the layer switched OFF (a ``deltaMaxChars``
raised above ``classifyMaxChars`` still clips, and a lazy import there would
pay the cold import on a running session's event loop). The parity test in
``tests/unit/monitors/test_classifier_gate.py`` fails if the two spellings
drift.

WHY THE QUESTION ID IS CONSTANT (AND NOT ``monitor_materiality:m1``)
====================================================================

§8.3 allows the id to carry the monitor id "for vendor-side logs". It does
not, because the service disables a question id for the whole session on a
schema rejection (``ClassificationService._disable_shape``): a per-monitor id
would turn one broken shape into one 422 per monitor, each disabling only
itself, where a single stable id disables the shape once — the stable-id rule
``recommend.py`` states for its own questions. Attribution never depended on
the id anyway: the state is one monitor's delta by construction, and the
per-monitor call is what makes the answer attributable (§8.1).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_operator.classification.types import Question

#: The question's id — one stable string for the whole gate. See the module
#: docstring for why it does not vary per monitor.
QUESTION_ID = "monitor_materiality"

#: The three classes of §8.3. The two below ``material`` are the suppress
#: classes; their spellings are the vendor-facing option ids.
MATERIAL = "material"
NON_MATERIAL_METADATA = "non-material-metadata"
IGNORABLE = "ignorable"

#: §8.3's instructions, verbatim. The question's whole rubric is here — a
#: decision model judges against the question text alone, so a paraphrase is
#: a different question with different measured costs.
QUESTION_INSTRUCTIONS = (
    "Decide whether this change is MATERIAL: a human asked to be told about it. "
    "MATERIAL = new information a person would want (a reply, a status flip, a new record). "
    "NON-MATERIAL METADATA = bookkeeping that changed without meaning (timestamps, ordering, "
    "volatile ids). IGNORABLE = noise that will never matter (whitespace, boilerplate)."
)

#: §8.3's criteria, verbatim — ``{option id: description}`` for a choice.
QUESTION_CRITERIA: dict[str, str] = {
    MATERIAL: "a person wanted to be told about this change",
    NON_MATERIAL_METADATA: "changed, but only metadata: timestamps, ordering, volatile ids",
    IGNORABLE: "noise that can never matter; whitespace, boilerplate, formatting",
}

#: The truncation marker, equal to ``classification.context.TRUNCATION_MARKER``
#: — duplicated rather than imported for the import-graph reason in the module
#: docstring. The parity test in ``tests/unit/monitors/test_classifier_gate.py``
#: fails if the two spellings drift.
TRUNCATION_MARKER = " …[truncated]"

#: §8.4: the suppress classes and the per-monitor counter each moves. The keys
#: are the counters-file keys of §10.3 (``suppressed.non_material_metadata`` /
#: ``suppressed.ignorable``); everything not in this map — ``material``, a
#: class we did not offer, no answer — fails OPEN to a delivery.
SUPPRESSED_COUNTERS: dict[str, str] = {
    NON_MATERIAL_METADATA: "non_material_metadata",
    IGNORABLE: "ignorable",
}

#: The scheduler's half of the seam: bounded state -> the materiality class
#: (a choice id, or ``None`` when there is no classifier). ``None`` means
#: "no classifier" and the caller fails OPEN (delivers).
MonitorClassify = Callable[[str], Awaitable[str | None]]


def materiality_question() -> Question:
    """The one typed question, built on demand (its text is constant).

    A plain function rather than a module constant because the constant would
    need ``Question`` at import time — see the module docstring for the
    import-graph constraint. The construction is four field assignments; the
    scheduler calls this once per changed monitor.
    """
    from local_operator.classification.types import Question

    return Question(
        id=QUESTION_ID,
        kind="choice",
        instructions=QUESTION_INSTRUCTIONS,
        criteria=dict(QUESTION_CRITERIA),
    )


def bounded_state(delta_text: str, max_chars: int) -> str:
    """The delta, cut to ``classifyMaxChars`` (§8.2's "bounded input").

    The delta is already bounded by ``deltaMaxChars`` (§7.3), so this is the
    belt to that braces: it bites when the two settings disagree (a raised
    ``deltaMaxChars`` with the classification bound left alone) or when a
    caller hands the gate text from elsewhere. A cut carries the layer's own
    truncation marker — the model should know the preview is incomplete —
    which lives in this module as a pinned duplicate (module docstring).

    The result is ``≤ max(max_chars, len(TRUNCATION_MARKER))``, NOT simply
    ``≤ max_chars``: a positive cap below the marker's own 14 characters
    returns the bare marker, the same pathological-cap policy the
    classification layer states for its state builder ("the marker alone is
    the smallest valid request this module can build, and it is still a valid
    one"). Reachable only by a hand-edited config (the settings reader refuses
    non-positive caps but honours any positive one), and pinned by
    ``test_bounded_state_cuts_and_marks``.
    """
    if max_chars <= 0 or len(delta_text) <= max_chars:
        return delta_text
    keep = max(0, max_chars - len(TRUNCATION_MARKER))
    return delta_text[:keep] + TRUNCATION_MARKER


def suppressed_counter(choice: str | None) -> str | None:
    """The counters key a choice suppresses under; ``None`` means DELIVER.

    THE FORK (§8.4): only the two non-material classes suppress. Everything
    else fails OPEN to a delivery — ``material``, no answer at all, and a
    class we never offered (a host's own seam can say anything; the vendor
    parser already refuses a choice id we did not offer). A wrong suppression
    costs a delayed signal, so the unsure direction is "deliver".
    """
    if choice in SUPPRESSED_COUNTERS:
        return SUPPRESSED_COUNTERS[choice]
    return None


def monitor_classify(resolve: Callable[[], Any | None]) -> MonitorClassify:
    """Wrap a per-call seam resolver into the scheduler's gate callback.

    ``resolve`` returns the session's shared ``ClassificationService`` (or
    anything with the same ``decide``), resolved PER CALL — the seam is an
    injectable attribute on the wiring's side (tests swap it after session
    construction, and the composition root resolves it the same way for the
    message path), so a captured service would pin whichever object happened
    to be there first. A ``None`` seam, or one without ``decide`` (a host's
    own classifier: the published seam is ``recommend_resources`` and nothing
    here may require more), returns ``None`` — the caller's fail-open signal.

    The seam's own guards decide what a call costs and when it is skipped
    entirely (``decide`` short-circuits disabled/circuit-open/disabled-shape
    states before any network work); this adapter only checks presence and
    unwraps the answer.
    """

    # NOTE: no ``question`` prebuilt here on purpose — building it would run
    # at session construction even when the seam is None, and it is the thing
    # that pulls ``local_operator.classification`` in (module docstring).
    async def classify(state: str) -> str | None:
        seam = resolve()
        decide = getattr(seam, "decide", None)
        if decide is None:
            return None
        answer = await decide(state=state, question=materiality_question())
        if answer is None or not isinstance(answer.value, str):
            return None
        return answer.value

    return classify


__all__ = [
    "IGNORABLE",
    "MATERIAL",
    "MonitorClassify",
    "NON_MATERIAL_METADATA",
    "QUESTION_CRITERIA",
    "QUESTION_ID",
    "QUESTION_INSTRUCTIONS",
    "SUPPRESSED_COUNTERS",
    "TRUNCATION_MARKER",
    "bounded_state",
    "materiality_question",
    "monitor_classify",
    "suppressed_counter",
]

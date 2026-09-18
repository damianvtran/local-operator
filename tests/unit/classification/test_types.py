"""The typed shapes: criteria validation, answer kinds, and the two error classes."""

from __future__ import annotations

import pytest

from local_operator.classification.types import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    DecisionSchemaError,
    DecisionVendorError,
    Question,
)


def test_a_choice_question_takes_a_mapping_of_option_ids_to_descriptions() -> None:
    question = Question(
        id="recommend_skill",
        kind="choice",
        instructions="which?",
        criteria={"a": "does a", "none": "nothing"},
    )
    assert question.criteria == {"a": "does a", "none": "nothing"}


def test_a_score_question_takes_an_array_and_is_normalised_to_a_tuple() -> None:
    # A LIST, deliberately: the contract's declared type is a tuple, but callers
    # hold lists and the normalisation in __post_init__ is what makes that safe.
    levels = ["trivial", "routine", "complex"]
    question = Question(
        id="effort",
        kind="score",
        instructions="how hard?",
        criteria=levels,  # type: ignore[arg-type]
    )
    assert question.criteria == ("trivial", "routine", "complex")
    # The caller keeps its list; mutating it after construction must not reach a
    # question already handed to a vendor.
    levels.append("impossible")
    assert question.criteria == ("trivial", "routine", "complex")


def test_a_score_question_refuses_a_mapping() -> None:
    with pytest.raises(ValueError, match="ARRAY"):
        Question(id="effort", kind="score", instructions="?", criteria={"a": "b"})


@pytest.mark.parametrize("kind", ["choice", "noul"])
def test_choice_and_noul_questions_refuse_an_array(kind: str) -> None:
    with pytest.raises(ValueError, match="mapping of option id"):
        Question(id="q", kind=kind, instructions="?", criteria=("a", "b"))  # type: ignore[arg-type]


def test_the_schema_error_is_not_a_vendor_error() -> None:
    """The cascade must not be able to swallow our own bug as weather.

    If ``DecisionSchemaError`` were a ``DecisionVendorError``, every
    ``except DecisionVendorError: continue`` would treat a malformed question as
    a vendor outage and send the same malformed request to the next leg — the
    exact behaviour §4 forbids. The class relationship IS that guarantee, so it
    is asserted rather than assumed.
    """
    assert not issubclass(DecisionSchemaError, DecisionVendorError)
    error = DecisionSchemaError("bad shape", shape="recommend_skill", status=422)
    assert error.shape == "recommend_skill"
    assert error.status == 422


def test_vendor_error_carries_kind_and_status_for_logs_and_the_breaker() -> None:
    error = DecisionVendorError("boom", kind="rate-limit", status=429)
    assert (error.kind, error.status) == ("rate-limit", 429)


def test_answer_defaults_are_empty_so_a_noul_answer_needs_no_distribution() -> None:
    answer = Answer(id="needs_skill", kind="noul", value=0.59)
    assert answer.probabilities == {}
    assert answer.confidence is None


def test_response_defaults_report_no_spend_rather_than_zero_spend() -> None:
    """An unreported figure is absent, not zero — the test the counts had to join.

    ``0`` is a legal value on this wire (Radient bills with output tokens zero),
    so a default of ``0`` made "nobody sent a usage block" indistinguishable from
    "the vendor reported zero", and the operator's cost line printed a fabricated
    ``tokens=0/0`` beside a real cost.
    """
    response = DecisionResponse(vendor="typesafe", model="jev-1.13.0", answers={})
    assert response.cost_usd is None
    assert (response.input_tokens, response.output_tokens) == (None, None)
    assert response.latency_s == 0.0


def test_a_request_is_a_state_and_a_tuple_of_questions() -> None:
    request = DecisionRequest(
        state={"request": "hi"},
        questions=(Question(id="q", kind="noul", instructions="?", criteria={"true": "y"}),),
    )
    assert request.state == {"request": "hi"}
    assert request.questions[0].id == "q"

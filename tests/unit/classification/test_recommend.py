"""Question construction, answer mapping and block rendering (§4, §6, §7)."""

from __future__ import annotations

import pytest

from local_operator.classification.context import Candidate
from local_operator.classification.recommend import (
    DEFAULT_MAX_RECOMMENDATIONS,
    NONE_OPTION,
    NONE_OPTION_TEXT,
    Recommendation,
    build_decision_request,
    build_questions,
    collect_resources,
    max_recommendations,
    option_id,
    render_block,
)
from local_operator.classification.types import Answer, DecisionResponse, Question
from tests.unit.classification.support import candidate


def criteria_of(question: Question) -> dict[str, str]:
    """The option map of a choice question, narrowed for the type checker.

    ``Question.criteria`` is a union (a map for choice/noul, an array for score)
    because that is what the vendor accepts; every question this module builds is
    a choice, so the assertions below can say so once instead of casting.
    """
    assert isinstance(question.criteria, dict)
    return question.criteria


def three_kinds() -> list[Candidate]:
    return [
        candidate("minerva-deploy", kind="skill", description="Deploy a Minerva service"),
        candidate("tunnel", kind="guide", description="Phone access over a tunnel"),
        candidate("hubspot", kind="mcp", description="CRM contacts and deals"),
    ]


# ---------------------------------------------------------------------------
# The question set
# ---------------------------------------------------------------------------


def test_one_choice_question_per_kind_with_candidates() -> None:
    plan = build_questions(three_kinds())
    assert [question.id for question in plan.questions] == [
        "recommend_skill",
        "recommend_guide",
        "recommend_mcp",
    ]
    assert {question.kind for question in plan.questions} == {"choice"}
    assert plan.kinds == {
        "recommend_skill": "skill",
        "recommend_guide": "guide",
        "recommend_mcp": "mcp",
    }


def test_a_kind_with_no_candidates_gets_no_question() -> None:
    plan = build_questions([candidate("tunnel", kind="guide")])
    assert [question.id for question in plan.questions] == ["recommend_guide"]


def test_every_question_ends_with_an_explicit_none() -> None:
    """Without it the model must pick something, and a wrong pick costs context."""
    plan = build_questions(three_kinds())
    for question in plan.questions:
        criteria = criteria_of(question)
        assert list(criteria)[-1] == NONE_OPTION
        assert criteria[NONE_OPTION] == NONE_OPTION_TEXT


def test_option_descriptions_are_the_name_plus_the_harness_owned_description() -> None:
    plan = build_questions(three_kinds())
    criteria = criteria_of(plan.questions[0])
    assert criteria["minerva-deploy"] == "minerva-deploy: Deploy a Minerva service"


def test_the_instructions_stay_advisory() -> None:
    plan = build_questions(three_kinds())
    instructions = plan.questions[0].instructions.lower()
    assert NONE_OPTION in instructions
    for word in ("must", "always", "required"):
        assert word not in instructions


def test_the_question_set_is_capped_per_kind() -> None:
    plan = build_questions([candidate(f"skill-{index}") for index in range(20)], limit=4)
    assert len(plan.questions[0].criteria) == 5  # four candidates plus `none`


def test_a_disabled_question_is_left_out_entirely() -> None:
    """A session that hit a schema error stops asking that shape, not all of them."""
    plan = build_questions(three_kinds(), skip_question_ids=frozenset({"recommend_skill"}))
    assert [question.id for question in plan.questions] == ["recommend_guide", "recommend_mcp"]
    assert "recommend_skill" not in plan.options


def test_option_ids_survive_names_that_are_not_identifiers() -> None:
    plan = build_questions([candidate("a skill/with spaces:and-colons")])
    option = next(iter(plan.options["recommend_skill"]))
    assert option == "a_skill_with_spaces:and-colons"
    assert plan.options["recommend_skill"][option].name == "a skill/with spaces:and-colons"


def test_two_names_that_normalise_the_same_get_distinct_option_ids() -> None:
    """A collision would make one resource unreachable and mis-attribute the other."""
    plan = build_questions([candidate("a b"), candidate("a/b")])
    options = plan.options["recommend_skill"]
    assert len(options) == 2
    assert {item.name for item in options.values()} == {"a b", "a/b"}
    assert len(set(options)) == 2


def test_a_candidate_literally_named_none_cannot_take_the_sentinel() -> None:
    plan = build_questions([candidate("none")])
    options = plan.options["recommend_skill"]
    criteria = criteria_of(plan.questions[0])
    # The sentinel keeps its id — "none" must always mean "nothing" — and the
    # resource named "none" is still offered, under a disambiguated id.
    assert criteria[NONE_OPTION] == NONE_OPTION_TEXT
    assert set(options) == {"none_2"}
    assert options["none_2"].name == "none"
    assert criteria["none_2"] == "none: does a thing"


def test_option_id_is_unique_within_a_question() -> None:
    taken = {"a"}
    assert option_id("a", taken) == "a_2"
    assert option_id("a", taken) == "a_3"
    assert option_id("!!!", taken) == "option"


def test_the_request_wraps_the_state_and_the_questions() -> None:
    plan = build_questions(three_kinds())
    request = build_decision_request(plan, {"request": "hello"})
    assert request.state == {"request": "hello"}
    assert request.questions == plan.questions


# ---------------------------------------------------------------------------
# Mapping answers back
# ---------------------------------------------------------------------------


def response_of(plan, **picks: tuple[str, float]) -> DecisionResponse:
    answers = {}
    for question in plan.questions:
        pick = picks.get(question.id)
        if pick is None:
            continue
        choice, confidence = pick
        answers[question.id] = Answer(
            id=question.id, kind="choice", value=choice, confidence=confidence
        )
    return DecisionResponse(vendor="stub", model="m", answers=answers)


def test_answers_map_back_to_the_original_candidates() -> None:
    plan = build_questions(three_kinds())
    response = response_of(plan, recommend_skill=("minerva-deploy", 1.0))
    resources = collect_resources(response, plan, max_recommendations=3)
    assert [item.resource_url for item in resources] == ["skill://minerva-deploy"]


def test_none_picks_contribute_nothing() -> None:
    plan = build_questions(three_kinds())
    response = response_of(plan, recommend_skill=(NONE_OPTION, 0.9), recommend_mcp=("hubspot", 0.5))
    resources = collect_resources(response, plan, max_recommendations=3)
    assert [item.name for item in resources] == ["hubspot"]


def test_picks_are_ordered_by_the_models_own_confidence() -> None:
    plan = build_questions(three_kinds())
    response = response_of(
        plan,
        recommend_skill=("minerva-deploy", 0.51),
        recommend_mcp=("hubspot", 0.95),
    )
    resources = collect_resources(response, plan, max_recommendations=3)
    assert [item.name for item in resources] == ["hubspot", "minerva-deploy"]


def test_an_answer_without_a_confidence_is_ordered_by_its_distribution() -> None:
    """``noul`` answers have no confidence field, so the distribution stands in."""
    plan = build_questions(three_kinds())
    response = DecisionResponse(
        vendor="stub",
        model="m",
        answers={
            "recommend_skill": Answer(
                id="recommend_skill",
                kind="choice",
                value="minerva-deploy",
                probabilities={"minerva-deploy": 0.7, "none": 0.3},
            ),
            "recommend_mcp": Answer(
                id="recommend_mcp",
                kind="choice",
                value="hubspot",
                probabilities={"hubspot": 0.4, "none": 0.6},
            ),
        },
    )
    resources = collect_resources(response, plan, max_recommendations=3)
    assert [item.name for item in resources] == ["minerva-deploy", "hubspot"]


def test_the_pick_count_is_capped() -> None:
    plan = build_questions(three_kinds())
    response = response_of(
        plan,
        recommend_skill=("minerva-deploy", 0.9),
        recommend_guide=("tunnel", 0.8),
        recommend_mcp=("hubspot", 0.7),
    )
    assert len(collect_resources(response, plan, max_recommendations=2)) == 2
    assert len(collect_resources(response, plan, max_recommendations=0)) == 0


def test_unknown_answers_and_non_choice_answers_are_ignored() -> None:
    plan = build_questions(three_kinds())
    response = DecisionResponse(
        vendor="stub",
        model="m",
        answers={
            "recommend_skill": Answer(id="recommend_skill", kind="choice", value="ghost"),
            "recommend_guide": Answer(id="recommend_guide", kind="noul", value=0.9),
            "not_a_question": Answer(id="not_a_question", kind="choice", value="x"),
        },
    )
    assert collect_resources(response, plan, max_recommendations=3) == ()


def test_an_unanswered_question_is_simply_absent() -> None:
    plan = build_questions(three_kinds())
    assert collect_resources(response_of(plan), plan, max_recommendations=3) == ()


# ---------------------------------------------------------------------------
# The block (§7)
# ---------------------------------------------------------------------------


def test_the_block_is_the_contracts_text_verbatim() -> None:
    resources = collect_resources(
        response_of(build_questions(three_kinds()), recommend_skill=("minerva-deploy", 1.0)),
        build_questions(three_kinds()),
        max_recommendations=3,
    )
    assert render_block(resources) == (
        "<resource_recommendations>\n"
        "These may help with this request — read the ones that actually fit, ignore the rest:\n"
        "- skill://minerva-deploy\n"
        "</resource_recommendations>"
    )


def test_an_empty_block_is_the_empty_string_so_the_prompt_is_unchanged() -> None:
    assert render_block(()) == ""


def test_the_block_is_additive_prose_and_never_imperative() -> None:
    block = render_block([candidate("tunnel", kind="guide")])
    assert "may help" in block
    assert "ignore the rest" in block
    assert "guide://tunnel" in block
    for word in ("must", "always", "only", "authoritative"):
        assert word not in block


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_the_recommendation_cap_reads_its_setting() -> None:
    assert max_recommendations(None) == DEFAULT_MAX_RECOMMENDATIONS == 3
    assert max_recommendations({"classification": {"maxRecommendations": 5}}) == 5


def test_a_recommendation_defaults_to_nothing_at_all() -> None:
    empty = Recommendation()
    assert empty.resources == ()
    assert empty.block == ""
    assert empty.vendor is None
    assert empty.cost_usd is None
    assert empty.skipped is None


def test_asking_a_question_of_a_plan_answers_the_kind() -> None:
    plan = build_questions(three_kinds())
    assert plan.kind_of("recommend_mcp") == "mcp"
    assert plan.kind_of("nope") is None


def test_a_single_option_block_keeps_the_contract_shape() -> None:
    """One resource, three lines: header, preamble, item."""
    block = render_block([candidate("x")])
    assert block.count("\n") == 3
    assert block.startswith("<resource_recommendations>")
    assert block.endswith("</resource_recommendations>")


@pytest.mark.parametrize("url", ["skill://x", "guide://x", "mcp://x"])
def test_the_block_lists_exactly_the_resource_urls(url: str) -> None:
    item = candidate("x", resource_url=url)
    assert f"- {url}" in render_block([item])


def test_questions_are_only_choice_so_a_score_question_is_never_built_here() -> None:
    """The kind exists in the vendor contract; the resource questions do not use it.

    Asserted so that a future "let's also score relevance" edit has to change a
    test and read why: the three questions are the token budget (§ the module
    docstring), and a fourth question would be a deliberate cost decision.
    """
    plan = build_questions(three_kinds())
    assert all(
        isinstance(question, Question) and question.kind == "choice" for question in plan.questions
    )

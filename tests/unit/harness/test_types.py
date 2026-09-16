"""Invariants on the harness event/message contract types.

These are the shapes every front end (TUI, server websockets, exec --json) and
the compaction layer program against, so a field that can contradict itself is
a UI defect waiting to happen rather than a style question.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_operator.harness.types import (
    DEFAULT_TURN_OUTPUT_TOKENS,
    AskOption,
    AskQuestion,
    ChatRequest,
    Message,
    ModelSpec,
    TextContent,
    ToolExecutionEndEvent,
    ToolResult,
    turn_output_budget,
)


def test_tool_end_error_flag_cannot_disagree_with_result():
    """``ToolExecutionEndEvent.is_error`` mirrors ``result.is_error``.

    UIs and the JSON exec stream read the event-level flag, so a producer that
    sets only the result's flag renders a failed tool as a success — the exact
    defect the TUI showed (a ``permission denied`` grep result drawn with the
    success glyph) before this invariant existed.
    """
    failed = ToolResult(
        tool_call_id="t1",
        tool_name="grep",
        content=[TextContent(text="permission denied")],
        is_error=True,
    )
    event = ToolExecutionEndEvent(tool_call_id="t1", tool_name="grep", result=failed)
    assert event.is_error is True
    assert event.model_dump()["is_error"] is True


def test_tool_end_clean_result_stays_clean():
    ok = ToolResult(tool_call_id="t2", tool_name="read", content=[TextContent(text="ok")])
    clean = ToolExecutionEndEvent(tool_call_id="t2", tool_name="read", result=ok)
    assert clean.is_error is False


def test_explicit_event_flag_is_never_downgraded():
    """The loop stamps aborted/synthetic results via the event-level flag, so a
    clean result must not clear it."""
    ok = ToolResult(tool_call_id="t3", tool_name="bash", content=[TextContent(text="ok")])
    forced = ToolExecutionEndEvent(tool_call_id="t3", tool_name="bash", result=ok, is_error=True)
    assert forced.is_error is True


# --- ask: the recommendation is normalised to the top ------------------------


def _ask(labels: list[str], recommended: int | None) -> AskQuestion:
    return AskQuestion(
        id="rollout",
        question="Which rollout?",
        options=[AskOption(label=label) for label in labels],
        recommended=recommended,
    )


def test_a_mid_list_recommendation_is_hoisted_to_the_top():
    """Normalised in the MODEL, so every surface gets it — including the mobile
    wire, which carries no ``recommended`` field and can express the
    recommendation only as position."""
    question = _ask(["a", "b", "c", "d"], 2)
    assert question.recommended == 0
    assert [option.label for option in question.options] == ["c", "a", "b", "d"]


def test_hoisting_preserves_the_relative_order_of_the_rest():
    """A rotation, not a swap: what the model did not recommend is still ranked,
    and a swap would promote whatever sat at index 0 over all of it."""
    question = _ask(["a", "b", "c", "d"], 3)
    assert [option.label for option in question.options] == ["d", "a", "b", "c"]


def test_a_recommendation_already_at_the_top_is_untouched():
    question = _ask(["a", "b", "c"], 0)
    assert question.recommended == 0
    assert [option.label for option in question.options] == ["a", "b", "c"]


def test_no_recommendation_leaves_the_authored_order_alone():
    """Without a recommendation there is nothing to promote, and reordering
    would silently discard the model's own ranking."""
    question = _ask(["a", "b", "c"], None)
    assert question.recommended is None
    assert [option.label for option in question.options] == ["a", "b", "c"]


def test_normalising_an_already_normalised_question_is_a_no_op():
    """Idempotent, because a question is re-validated on every round trip it
    makes (model_validate of a dumped question, a resumed session): a hoist that
    ran twice would rotate the list a second time and endorse a different row."""
    once = _ask(["a", "b", "c"], 1)
    twice = AskQuestion.model_validate(once.model_dump())
    thrice = AskQuestion.model_validate(twice.model_dump())
    assert [option.label for option in twice.options] == ["b", "a", "c"]
    assert [option.label for option in thrice.options] == ["b", "a", "c"]
    assert twice.recommended == 0 and thrice.recommended == 0


def test_an_out_of_range_recommendation_is_still_refused():
    """The bounds check runs BEFORE the hoist: reordering against an index that
    indexes nothing would turn a correctable error into a scrambled list."""
    with pytest.raises(ValidationError) as excinfo:
        _ask(["a", "b"], 5)
    assert "recommended must index options (0..1)" in str(excinfo.value)


def test_a_negative_recommendation_is_refused_rather_than_indexing_from_the_end():
    """``-1`` is a valid Python index and would silently promote the LAST
    option — endorsing something the model did not choose, which is the exact
    failure the bounds check exists to prevent. The message states the usable
    range so the model can correct it."""
    with pytest.raises(ValidationError) as excinfo:
        _ask(["a", "b", "c"], -1)
    assert "recommended must index options (0..2)" in str(excinfo.value)


def test_a_refused_recommendation_never_reorders_the_options():
    """The refusal path must leave the list alone: a question that came back
    both rejected AND scrambled would hand the model a correction to make
    against options it no longer recognises."""
    options = [AskOption(label=label) for label in ("a", "b", "c")]
    with pytest.raises(ValidationError):
        AskQuestion(id="rollout", question="Which rollout?", options=options, recommended=7)
    assert [option.label for option in options] == ["a", "b", "c"]


def test_the_hoist_survives_a_json_round_trip():
    """A question is re-parsed from JSON on transcript replay and on the way to
    a subagent, not only from a dumped dict — the order the user was shown has
    to be the order that comes back."""
    once = _ask(["a", "b", "c", "d"], 2)
    replayed = AskQuestion.model_validate_json(once.model_dump_json())
    assert [option.label for option in replayed.options] == ["c", "a", "b", "d"]
    assert replayed.recommended == 0


def test_hoisting_moves_the_whole_option_not_just_its_label():
    """Each option carries the consequence line the user decides on, so a hoist
    that moved labels alone would pair the promoted option with somebody else's
    description."""
    question = AskQuestion(
        id="rollout",
        question="Which rollout?",
        options=[
            AskOption(label="a", description="cheapest"),
            AskOption(label="b", description="keeps history"),
            AskOption(label="c", description="safest"),
        ],
        recommended=2,
    )
    assert (question.options[0].label, question.options[0].description) == ("c", "safest")
    assert [option.description for option in question.options] == [
        "safest",
        "cheapest",
        "keeps history",
    ]


def test_a_secret_question_still_refuses_a_recommendation_after_the_hoist_landed():
    """The secret branch returns before the hoist, so a credential paste can
    never be reordered or preselected into endorsing a value nobody can see."""
    with pytest.raises(ValidationError) as excinfo:
        AskQuestion(id="GITHUB_TOKEN", question="Paste it.", options=[], secret=True, recommended=0)
    assert "no options to recommend" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The generation bound a request carries (Step 1 of the runtime convergence)
# ---------------------------------------------------------------------------
#
# The defect these pin is a bound that was never OURS: a request that named no ask
# carried the provider's advertised capability verbatim, so on a 1M aggregate
# model every call asked for 943,718 output tokens and a single decision ran to
# 97,189 (95,098 of them reasoning). The bound therefore lives on the request
# CONTRACT rather than at one call site, so a new interface cannot reintroduce it
# by forgetting.


def _spec(max_output_tokens: int) -> ModelSpec:
    return ModelSpec(
        provider="openrouter",
        model_id="meta/muse-spark-1.3",
        context_window=1_048_576,
        max_output_tokens=max_output_tokens,
    )


def test_a_request_with_no_ask_carries_the_policy_ceiling() -> None:
    """The measured case: a 1M-window model advertising 943,718 output tokens.

    Built from the advertised figure -- which is 90% of that window -- the
    request asked for it on EVERY call, which is what let one response reason
    for 95,098 tokens before answering.
    """
    request = ChatRequest(model=_spec(943_718), messages=[Message.user("hi")])

    assert request.max_tokens == DEFAULT_TURN_OUTPUT_TOKENS
    assert request.max_tokens < 943_718


def test_the_policy_ceiling_leaves_every_measured_ordinary_turn_intact() -> None:
    """The number is chosen against the operator's own ledger, not taste.

    876,719 recorded calls: 430 ever emitted more than 16,384 output tokens, 300
    of them ordinary sessions; 2 ordinary calls exceeded 65,536, both
    ``claude-opus-5`` at exactly its own 128,000 published ceiling; none exceeded
    131,072. So the ceiling has to sit above 128,000 and below the
    capability-shaped asks, and this pins both halves of that: the largest
    published ceiling in ordinary use passes through untouched, and a
    capability-shaped one is cut 7-8x.
    """
    assert DEFAULT_TURN_OUTPUT_TOKENS > 128_000
    assert ChatRequest(model=_spec(128_000), messages=[]).max_tokens == 128_000
    assert ChatRequest(model=_spec(1_047_576), messages=[]).max_tokens == DEFAULT_TURN_OUTPUT_TOKENS


def test_a_smaller_published_ceiling_wins_over_the_policy() -> None:
    """Model-aware in the narrowing direction only: a provider limit below the
    policy is a real limit and is kept, while a larger advertisement is not."""
    assert ChatRequest(model=_spec(4_096), messages=[]).max_tokens == 4_096
    assert ChatRequest(model=_spec(64_000), messages=[]).max_tokens == 64_000
    assert ChatRequest(model=_spec(131_072), messages=[]).max_tokens == 131_072


def test_a_spec_with_no_published_ceiling_is_still_bounded() -> None:
    """``0`` is "no data", not "unlimited", so the policy fills it.

    This is NOT the case that produced the 97k-token response -- that call ran on
    a model advertising 943,718, and it carried that figure on the wire. Keep the
    two apart: a genuinely cap-less spec is a shape production does not reach
    (unknown models resolve to 8,192, local ones to 1,024), which is why this arm
    is pinned as a contract property rather than as a reproduced incident.
    """
    assert ChatRequest(model=_spec(0), messages=[]).max_tokens == DEFAULT_TURN_OUTPUT_TOKENS


def test_an_explicit_ask_is_never_overridden() -> None:
    """``Session.ERRAND_MAX_TOKENS`` (1024, titling) and the compaction
    summariser name their own budget, and an explicit ask above the policy is
    honoured too: the bound exists to fill a silence, not to cap a decision."""
    assert ChatRequest(model=_spec(943_718), messages=[], max_tokens=1_024).max_tokens == 1_024
    assert ChatRequest(model=_spec(943_718), messages=[], max_tokens=500_000).max_tokens == 500_000


def test_asking_for_no_cap_is_unrepresentable() -> None:
    """``0`` is rejected, and that is a correction rather than a tightening.

    It used to mean "ask the provider for no cap", but the four wire builders
    never agreed on what an absent cap is (the OpenAI-shaped and Google bodies
    omit the key; Anthropic's API REQUIRES one), and on a model that advertises a
    cap it did not mean "no cap" at all -- the clamp fell back to the advertised
    capability and put 943,718 back on the wire (QA round 1, Q4). A caller that
    wants the provider's own default gets it by naming nothing.
    """
    with pytest.raises(ValidationError):
        ChatRequest(model=_spec(943_718), messages=[], max_tokens=0)


def test_a_policy_bound_follows_a_model_swap_and_a_named_ask_does_not() -> None:
    """``with_model`` is the failover hop, and the bound has to survive it.

    ``model_copy`` cannot re-run the validator, so the old failover clone carried
    the primary's ask onto a fallback publishing a smaller ceiling (34 shipped
    rows publish under 20K) and kept a small model's ask on a large fallback
    (review M1 / QA Q5, both directions measured).
    """
    small = _spec(8_192)
    big = _spec(943_718)
    policy_bound = ChatRequest(model=big, messages=[])
    assert policy_bound.with_model(small).max_tokens == 8_192
    assert policy_bound.with_model(small).max_tokens_from_policy is True
    # The original request is untouched -- the clone is what moves.
    assert policy_bound.max_tokens == DEFAULT_TURN_OUTPUT_TOKENS
    # A NAMED ask is the caller's own decision and is carried through unchanged,
    # which is the behaviour QA item 3 measured and must keep measuring.
    named = ChatRequest(model=small, messages=[], max_tokens=50_000)
    assert named.max_tokens_from_policy is False
    assert named.with_model(big).max_tokens == 50_000
    # A spec that publishes no cap re-derives to the policy, not to zero -- and
    # the small-to-big direction the QA finding measured re-derives UP to the
    # policy rather than leaving the small model's ask on the large fallback.
    assert policy_bound.with_model(_spec(0)).max_tokens == DEFAULT_TURN_OUTPUT_TOKENS
    assert ChatRequest(model=small, messages=[]).with_model(big).max_tokens == (
        DEFAULT_TURN_OUTPUT_TOKENS
    )


def test_the_policy_ceiling_is_configurable() -> None:
    """The bound is a default a host can raise, not a constant baked into every
    caller: the policy takes the ceiling explicitly, and a host that needs a
    longer answer names one on the request (pinned above)."""
    assert turn_output_budget(_spec(943_718), ceiling=32_000) == 32_000
    assert turn_output_budget(_spec(943_718)) == DEFAULT_TURN_OUTPUT_TOKENS
    # ``None``/``0`` mean "the default", not "no bound" -- a non-positive
    # override must not silently uncap a turn.
    assert turn_output_budget(_spec(943_718), ceiling=None) == DEFAULT_TURN_OUTPUT_TOKENS
    assert turn_output_budget(_spec(943_718), ceiling=0) == DEFAULT_TURN_OUTPUT_TOKENS

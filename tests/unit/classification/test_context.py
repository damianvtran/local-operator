"""The context budget and its truncation ladder (§5), rung by rung."""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.classification.context import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_STATE_CHARS,
    TRUNCATION_MARKER,
    Candidate,
    build_state,
    candidate_line,
    candidates_digest,
    max_candidates,
    max_state_chars,
    select_candidates,
    serialized_size,
    setting_int,
    shortlist,
)
from local_operator.classification.recommend import build_questions
from tests.unit.classification.support import candidate


def roster(
    kind: str = "skill", count: int = 3, description: str = "does a thing"
) -> list[Candidate]:
    return [
        candidate(f"{kind}-{index}", kind=kind, description=description) for index in range(count)
    ]


def size_of(state: dict[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Settings readers (§8)
# ---------------------------------------------------------------------------


def test_the_section_is_accepted_either_way_round() -> None:
    assert max_state_chars({"classification": {"maxStateChars": 900}}) == 900
    assert max_state_chars({"maxStateChars": 900}) == 900


@pytest.mark.parametrize("raw", [None, 0, -5, True, "many", "0"], ids=str)
def test_a_nonsense_cap_reads_as_the_default(raw) -> None:
    """``True`` is an ``int`` in Python and ``0`` is a budget for nothing at all."""
    assert setting_int({"timeoutMs": raw}, "timeoutMs", 1500) == 1500


def test_a_quoted_number_is_read_rather_than_ignored() -> None:
    assert setting_int({"maxCandidates": " 7 "}, "maxCandidates", 12) == 7


def test_the_defaults_are_the_contract_defaults() -> None:
    assert DEFAULT_MAX_STATE_CHARS == 6000
    assert DEFAULT_MAX_CANDIDATES == 12
    assert max_state_chars(None) == 6000
    assert max_candidates({}) == 12


# ---------------------------------------------------------------------------
# Rung 0 and the roster cap
# ---------------------------------------------------------------------------


def test_a_small_state_keeps_its_context_and_untrimmed_lines() -> None:
    long_description = "x" * 300
    state = build_state(
        user_message="deploy core to qa",
        context="compaction summary line",
        candidates=roster(description=long_description),
        max_chars=6000,
    )
    assert state["request"] == "deploy core to qa"
    assert state["context"] == "compaction summary line"
    assert state["candidates"]["skills"][0] == f"skill-0: {long_description}"


def test_the_roster_is_capped_per_kind_not_overall() -> None:
    candidates = roster("skill", 20) + roster("guide", 20)
    kept = select_candidates(candidates, 2)
    assert [item.kind for item in kept] == ["skill", "skill", "guide", "guide"]
    assert [item.name for item in kept] == ["skill-0", "skill-1", "guide-0", "guide-1"]


def test_the_roster_keeps_the_callers_order() -> None:
    """Order is discovery rank, and it is the option order the model sees."""
    candidates = [candidate("b"), candidate("a")]
    assert [item.name for item in select_candidates(candidates, 5)] == ["b", "a"]


def test_an_empty_kind_is_omitted_rather_than_sent_as_an_empty_list() -> None:
    state = build_state(user_message="hi", context=None, candidates=roster("guide", 1))
    assert set(state["candidates"]) == {"guides"}


def test_a_state_with_no_candidates_carries_only_the_request() -> None:
    state = build_state(user_message="hi", context=None, candidates=[])
    assert state == {"request": "hi"}


# ---------------------------------------------------------------------------
# Rung 1: context goes first
# ---------------------------------------------------------------------------


def test_context_is_dropped_before_any_candidate_text_is_touched() -> None:
    long_context = "c" * 400
    candidates = roster("skill", 3, description="x" * 200)
    with_context = build_state(
        user_message="hi", context=long_context, candidates=candidates, max_chars=10_000
    )
    assert with_context["context"] == long_context

    # A cap that fits the roster but not the roster plus the context: the context
    # goes, and the descriptions come through untouched.
    without_context = {
        "request": "hi",
        "candidates": {"skills": [candidate_line(item) for item in candidates]},
    }
    state = build_state(
        user_message="hi",
        context=long_context,
        candidates=candidates,
        max_chars=size_of(without_context),
    )
    assert "context" not in state
    assert state["candidates"]["skills"] == [candidate_line(item) for item in candidates]


def test_an_empty_context_is_never_sent() -> None:
    state = build_state(user_message="hi", context="", candidates=roster("skill", 1))
    assert "context" not in state


# ---------------------------------------------------------------------------
# Rung 2: the two line trims
# ---------------------------------------------------------------------------


def test_candidate_lines_are_trimmed_to_120_then_to_60() -> None:
    assert len(candidate_line(candidate("x", description="d" * 400), 120)) == 120
    assert candidate_line(candidate("x", description="d" * 400), 120).endswith("…")
    assert len(candidate_line(candidate("x", description="d" * 400), 60)) == 60
    # A line that already fits is left alone, marker and all.
    assert candidate_line(candidate("x", description="short"), 60) == "x: short"


def test_the_ladder_prefers_the_120_char_trim_over_dropping_a_kind() -> None:
    candidates = roster("skill", 4, description="d" * 200)
    full = {
        "request": "hi",
        "candidates": {"skills": [candidate_line(item) for item in candidates]},
    }
    trimmed = [candidate_line(item, 120) for item in candidates]
    trimmed_state = {"request": "hi", "candidates": {"skills": trimmed}}
    assert size_of(trimmed_state) < size_of(full)

    # At the full size the descriptions come through whole...
    whole = build_state(
        user_message="hi", context=None, candidates=candidates, max_chars=size_of(full)
    )
    assert whole["candidates"]["skills"] == [candidate_line(item) for item in candidates]

    # ...and one character tighter, the trim happens and the roster SURVIVES —
    # a kind is never dropped while a trim could still have made room.
    state = build_state(
        user_message="hi", context=None, candidates=candidates, max_chars=size_of(trimmed_state)
    )
    assert state["candidates"]["skills"] == trimmed
    assert len(state["candidates"]["skills"][0]) == 120


def test_a_trim_shorter_than_the_marker_still_returns_the_marker() -> None:
    assert candidate_line(candidate("x", description="d" * 50), 1) == "…"


# ---------------------------------------------------------------------------
# Rung 3: kinds are dropped lowest-priority first
# ---------------------------------------------------------------------------


def test_mcp_goes_before_guides_and_guides_before_skills() -> None:
    candidates = (
        roster("skill", 6, description="d" * 100)
        + roster("guide", 6, description="d" * 100)
        + roster("mcp", 6, description="d" * 100)
    )

    def lines_of(kinds: tuple[str, ...]) -> dict[str, list[str]]:
        keys = {"skill": "skills", "guide": "guides", "mcp": "mcp_servers"}
        return {
            keys[kind]: [candidate_line(item, 60) for item in candidates if item.kind == kind]
            for kind in kinds
        }

    size_all_three = size_of({"request": "hi", "candidates": lines_of(("skill", "guide", "mcp"))})
    size_without_mcp = size_of({"request": "hi", "candidates": lines_of(("skill", "guide"))})
    size_skills_only = size_of({"request": "hi", "candidates": lines_of(("skill",))})
    assert size_skills_only < size_without_mcp < size_all_three

    # At the size that fits all three, nothing is dropped (the lines are trimmed
    # to 60 to get there, which is rung 2 doing its job before rung 3 is reached).
    everything = build_state(
        user_message="hi", context=None, candidates=candidates, max_chars=size_all_three
    )
    assert set(everything["candidates"]) == {"skills", "guides", "mcp_servers"}

    # One step tighter: MCP servers go first.
    no_mcp = build_state(
        user_message="hi", context=None, candidates=candidates, max_chars=size_without_mcp
    )
    assert set(no_mcp["candidates"]) == {"skills", "guides"}

    # Then guides.
    skills_only = build_state(
        user_message="hi", context=None, candidates=candidates, max_chars=size_skills_only
    )
    assert set(skills_only["candidates"]) == {"skills"}


# ---------------------------------------------------------------------------
# Rung 4: the request itself, truncated last
# ---------------------------------------------------------------------------


def test_a_message_larger_than_the_whole_budget_is_classified_on_its_head() -> None:
    message = "deploy core to qa " * 1000
    state = build_state(user_message=message, context=None, candidates=[], max_chars=500)

    assert serialized_size(state) <= 500
    assert state["request"].endswith(TRUNCATION_MARKER)
    head = state["request"][: -len(TRUNCATION_MARKER)]
    assert message.startswith(head)
    assert 0 < len(head) < len(message)
    # Still a valid request for the vendor: a JSON-serializable object.
    assert json.loads(json.dumps(state))["request"] == state["request"]


def test_the_truncated_head_is_sized_against_the_marker_and_the_scaffolding() -> None:
    """The marker and the JSON keys are counted before the head is measured."""
    message = "a" * 10_000
    state = build_state(user_message=message, context=None, candidates=[], max_chars=200)
    assert serialized_size(state) <= 200
    # A cap of 200 with ~30 chars of scaffolding and a 14-char marker leaves a
    # head of roughly 150 chars — not 200, and nowhere near 10,000.
    assert 100 < len(state["request"]) < 200


def test_json_escaping_is_measured_rather_than_assumed() -> None:
    """One input character can cost six once escaped, so the loop re-measures.

    A quote is the cheap case (``\\"``); a control character is the expensive one
    (``\\u0000``). If the head were sized arithmetically on character counts, this
    state would land over the cap.
    """
    message = "\x00" * 500
    state = build_state(user_message=message, context=None, candidates=[], max_chars=300)
    assert serialized_size(state) <= 300
    assert state["request"].endswith(TRUNCATION_MARKER)


def test_a_cap_smaller_than_the_scaffolding_still_produces_a_valid_request() -> None:
    state = build_state(user_message="a" * 5000, context=None, candidates=[], max_chars=1)
    assert state == {"request": TRUNCATION_MARKER}
    assert json.loads(json.dumps(state))


def test_candidates_are_dropped_before_the_request_is_truncated() -> None:
    """Rung 3 outranks rung 4: the roster goes before the user's words do."""
    message = "deploy core to qa"
    candidates = roster("skill", 8, description="d" * 120)
    # A cap big enough for the request alone but not for the roster too.
    state = build_state(user_message=message, context=None, candidates=candidates, max_chars=60)
    assert state["request"] == message
    assert "candidates" not in state


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def test_the_digest_changes_when_a_description_changes() -> None:
    """Option text IS the description, so a changed description is a new question."""
    before = candidates_digest([candidate("a", description="one")])
    after = candidates_digest([candidate("a", description="two")])
    assert before != after
    assert candidates_digest([candidate("a")]) == candidates_digest([candidate("a")])


def test_the_digest_notices_a_reordered_roster() -> None:
    first = candidates_digest([candidate("a"), candidate("b")])
    second = candidates_digest([candidate("b"), candidate("a")])
    assert first != second


def test_a_caller_supplied_context_reaches_the_state_and_the_ladder_applies_in_order() -> None:
    """The ladder, rung by rung, with a context present — §5's one caller-less field.

    Context is the field no caller fills today (see the module docstring of
    ``context.py``), so the package owes proof that it WORKS when one does:
    the caller's string must reach the serialized state, and it must be the first
    thing the ladder gives up — before any candidate text is trimmed. Each
    assertion below sits on a cap computed from a hand-built state at that rung,
    so a reordering of the ladder changes which assertion fails rather than
    quietly passing.
    """
    summary = "compaction summary line: user asked to deploy core to qa"
    # Descriptions longer than the 120-char trim, so each trim rung actually
    # changes the state: a roster of short lines makes rungs 2 and 3 identical
    # and the "first state that fits" test ambiguous.
    long_description = "d" * 200
    candidates = (
        roster("skill", 2, description=long_description)
        + roster("guide", 2, description=long_description)
        + roster("mcp", 2, description=long_description)
    )
    state_keys = {"skill": "skills", "guide": "guides", "mcp": "mcp_servers"}

    def payload(
        context_text: str | None, limit: int | None, kinds: tuple[str, ...]
    ) -> dict[str, Any]:
        built: dict[str, Any] = {"request": "hi"}
        if context_text is not None:
            built["context"] = context_text
        if kinds:
            built["candidates"] = {
                state_keys[kind]: [
                    candidate_line(item, limit) for item in candidates if item.kind == kind
                ]
                for kind in kinds
            }
        return built

    all_kinds = ("skill", "guide", "mcp")
    size_rung0 = size_of(payload(summary, None, all_kinds))
    size_rung1 = size_of(payload(None, None, all_kinds))  # context dropped
    size_rung2 = size_of(payload(None, 120, all_kinds))
    size_rung3 = size_of(payload(None, 60, all_kinds))
    size_rung4 = size_of(payload(None, 60, ("skill", "guide")))  # mcp dropped
    size_rung5 = size_of(payload(None, 60, ("skill",)))  # guides dropped
    # Strictly descending, so "the first state that fits" is unambiguous at every
    # cap used below.
    assert size_rung0 > size_rung1 > size_rung2 > size_rung3 > size_rung4 > size_rung5

    # Rung 0: room to spare — the caller's context is IN the state with the
    # roster whole.
    roomy = build_state(
        user_message="hi", context=summary, candidates=candidates, max_chars=size_rung0
    )
    assert roomy["context"] == summary
    assert roomy["candidates"]["skills"] == [
        candidate_line(item) for item in candidates if item.kind == "skill"
    ]

    # Rung 1: one character tighter, the CONTEXT goes and nothing else does.
    dropped_context = build_state(
        user_message="hi", context=summary, candidates=candidates, max_chars=size_rung1
    )
    assert "context" not in dropped_context
    assert dropped_context["candidates"]["skills"] == [
        candidate_line(item) for item in candidates if item.kind == "skill"
    ]

    # Rung 2: then the lines trim, 120 before 60.
    trimmed_120 = build_state(
        user_message="hi", context=summary, candidates=candidates, max_chars=size_rung2
    )
    assert "context" not in trimmed_120
    assert len(trimmed_120["candidates"]["skills"][0]) == 120
    trimmed_60 = build_state(
        user_message="hi", context=summary, candidates=candidates, max_chars=size_rung3
    )
    assert len(trimmed_60["candidates"]["skills"][0]) == 60

    # Rung 3: then kinds, lowest priority first — MCP, then guides, then skills.
    no_mcp = build_state(
        user_message="hi", context=summary, candidates=candidates, max_chars=size_rung4
    )
    assert set(no_mcp["candidates"]) == {"skills", "guides"}
    no_guides = build_state(
        user_message="hi", context=summary, candidates=candidates, max_chars=size_rung5
    )
    assert set(no_guides["candidates"]) == {"skills"}

    # Rung 4: last, and only last, the request itself — and the roster is gone by
    # then, because a truncated request with a roster still attached would have
    # given up the operator's words before the model's own rubric.
    long_message = "deploy core to qa " * 100
    truncated = build_state(
        user_message=long_message, context=summary, candidates=candidates, max_chars=200
    )
    assert truncated["request"].endswith(TRUNCATION_MARKER)
    assert "candidates" not in truncated
    assert "context" not in truncated


def test_a_context_only_state_needs_no_candidates_key() -> None:
    """A caller with a context and an empty roster still gets a valid, bounded state."""
    state = build_state(user_message="", context="summary", candidates=[], max_chars=100)
    assert state == {"request": "", "context": "summary"}
    assert serialized_size(state) <= 100


# ---------------------------------------------------------------------------
# Scaling to hundreds of skills: which rows travel, and what the request costs
# ---------------------------------------------------------------------------


def test_a_roster_that_fits_every_kind_is_returned_by_identity() -> None:
    """The warm path must not allocate: a small catalogue IS the roster object.

    The wiring asserts that two consecutive messages carry the same candidates
    object, so a version of ``shortlist`` that copied unconditionally would make
    every warm message a fresh tuple for no gain.
    """
    rows = tuple(roster(count=5))
    assert shortlist(rows, "deploy core to qa", DEFAULT_MAX_CANDIDATES) is rows


def test_a_roster_over_the_cap_keeps_the_relevant_rows() -> None:
    """The per-kind cap is a cost bound, so WHICH rows survive has to be a choice.

    Before ``shortlist`` the survivors were the first ``maxCandidates`` in
    discovery order — a permanent dozen for the life of the session, with the rest
    of the catalogue unreachable. Here 61 skills contend for 12 places and the one
    the message is about wins one of them.
    """
    rows = tuple(
        [candidate(f"fleet-{index:03d}", description="Fleet automation") for index in range(60)]
        + [
            candidate(
                "flavia-adverse-media",
                description="Adverse media screening for a named person",
            )
        ]
        + [candidate("guide-x", kind="guide", description="A packaged guide")]
    )

    picked = shortlist(rows, "run an adverse media screen for Flavia", DEFAULT_MAX_CANDIDATES)
    names = [row.name for row in picked]

    assert "flavia-adverse-media" in names
    assert len([name for name in names if name.startswith("fleet-")]) == 11
    # A kind under its own cap is untouched, so the guide keeps its place.
    assert "guide-x" in names
    # SURVIVORS KEEP THE CALLER'S ORDER: the named skill sits where the roster put
    # it (last of the skills), not first because it scored highest. The option order
    # is part of the request the vendor sees.
    assert names == [name for name in (row.name for row in rows) if name in set(names)]


def test_the_shortlist_is_deterministic() -> None:
    """Two identical messages must not churn the request, or the cache key churns."""
    rows = tuple(roster(count=40))
    assert shortlist(rows, "deploy core", DEFAULT_MAX_CANDIDATES) == shortlist(
        rows, "deploy core", DEFAULT_MAX_CANDIDATES
    )


def test_the_request_stays_a_small_fraction_of_the_models_input_window() -> None:
    """Hundreds of installed skills, and the request the vendor receives.

    Every candidate travels TWICE — a line in the state and an option description
    in its kind's question — so the two visible bounds are ``maxStateChars``
    (6 000 chars) and ``maxCandidates`` per kind. This test pins the whole request
    at the shipped defaults against the model's real window: **32k input tokens**
    (operator's figure, 2026-09-18; §5 of the design doc said 64k, which nothing
    had measured).

    The numbers, so the assertion can be checked by hand: 6 000 chars of state plus
    12 skills + 12 guides + 12 MCP options at the §5 line cap (120 chars) is
    ~10.3k chars ≈ 2.6k tokens, i.e. ~8% of the window — and the candidate cap, not
    the window, is what the operator would move.
    """
    rows = tuple(roster(count=500)) + tuple(
        candidate(f"guide-{index}", kind="guide", description="A packaged guide")
        for index in range(30)
    )
    message = "why can't this tenant run legal searches? " * 8

    state = build_state(user_message=message, context=None, candidates=rows)
    state_chars = serialized_size(state)
    assert state_chars <= DEFAULT_MAX_STATE_CHARS

    plan = build_questions(rows)
    option_chars = sum(
        len(f"{name}: {text}")
        for question in plan.questions
        for name, text in question.criteria.items()
    )
    # The questions themselves carry instructions; counted roughly, they are a few
    # hundred characters each and bounded by the three kinds.
    instruction_chars = sum(len(question.instructions) for question in plan.questions)
    estimated_tokens = (state_chars + option_chars + instruction_chars) // 4

    print(
        f"500 skills + 30 guides: state={state_chars} chars, "
        f"options={option_chars} chars, instructions={instruction_chars} chars, "
        f"~{estimated_tokens} tokens of a 32k window"
    )
    assert estimated_tokens <= 3_000


def test_a_raised_candidate_cap_scales_linearly_and_stays_inside_the_window() -> None:
    """The knob's safe range, measured: the window is not what binds at 40 a kind."""
    rows = tuple(roster(count=500))
    message = "deploy core to qa"

    state = build_state(user_message=message, context=None, candidates=rows, candidate_limit=40)
    plan = build_questions(rows, limit=40)
    option_chars = sum(
        len(f"{name}: {text}")
        for question in plan.questions
        for name, text in question.criteria.items()
    )
    estimated_tokens = (serialized_size(state) + option_chars) // 4

    print(f"maxCandidates=40: state+options ~{estimated_tokens} tokens")
    assert estimated_tokens <= 16_000

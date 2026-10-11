"""The generator errand's bounds, its repair turn and its model ladder (memo §2.5, §5.1).

WHAT THESE TESTS ARE FOR. The generator is the only part of this feature that spends money
and the only part that runs unattended, so the interesting failures are not "does it produce a
chart" but "does it stop". Each bound gets a test that trips it and asserts what the row would
carry: the bound's name, the blocks that already passed, and no turn beyond the one that was
allowed.

The fork's transport is INJECTED (``complete``), which is what makes the loop testable without
a session, a provider or a credential: the tests below drive the real turn loop, the real
validator and the real prompt builder, and only the provider call is a stub.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest

from local_operator.supplements import generator
from local_operator.supplements.evidence import Dataset

DATASETS = (
    Dataset(
        title="Latency by region",
        source="bench.csv",
        columns=("region", "ms"),
        rows=(("us-east", "120"), ("us-west", "98.5"), ("eu-west", "143"), ("ap-south", "211.25")),
        n_rows=4,
        numeric_columns=("ms",),
    ),
)

HONEST_BLOCK = (
    '<component title="Latency" source="bench.csv">\n'
    '<data>{"lat":{"title":"Latency (ms)","columns":["region","ms"],'
    '"rows":[["us-east",120],["us-west",98.5]]}}</data>\n'
    '<html><div id="c"></div><script>LO.bar(document.getElementById("c"),"lat",'
    '{x:"region",y:"ms",unit:"ms"})</script></html>\n'
    "</component>"
)

LYING_BLOCK = HONEST_BLOCK.replace("211.25", "7000").replace(
    '["us-east",120],["us-west",98.5]', '["us-east",120],["us-west",7000]'
)


def model_spec(provider: str = "anthropic", model_id: str = "claude-sonnet-5") -> Any:
    """A real ``ModelSpec``: ``ChatRequest`` validates it, so a duck-typed stand-in fails the
    request build rather than the assertion under test."""
    from local_operator.harness.types import ModelSpec

    return ModelSpec(provider=provider, model_id=model_id)


class Recorder:
    """The injected transport: a scripted answer per call, plus what it was asked."""

    def __init__(self, answers: list[str], *, usage_tokens: int = 10) -> None:
        self._answers = list(answers)
        self.requests: list[Any] = []
        self.usage_tokens = usage_tokens

    async def __call__(self, request: Any) -> tuple[str, Any]:
        self.requests.append(request)
        text = self._answers.pop(0) if self._answers else "NONE"
        return text, _Usage(self.usage_tokens)


class _Usage:
    def __init__(self, tokens: int) -> None:
        self.input_tokens = tokens
        self.output_tokens = tokens


async def _run(
    answers: list[str],
    *,
    max_turns: int = 2,
    max_cost_usd: float = 1.00,
    price: Any = None,
    max_components: int = generator.MAX_COMPONENTS,
    timeout_s: float | None = None,
    started: float | None = None,
    usage_tokens: int = 10,
) -> tuple[generator.GenerationOutcome, Recorder]:
    recorder = Recorder(answers, usage_tokens=usage_tokens)
    outcome = await generator.generate(
        complete=recorder,
        model=generator.DesignModel(model_spec(), "ladder", "anthropic/claude-sonnet-5"),
        datasets=DATASETS,
        user_text="how did the bench go?",
        answer_text="p50 by region: 120 ms east, 98.5 west.",
        max_turns=max_turns,
        max_output_tokens=6000,
        max_cost_usd=max_cost_usd,
        max_components=max_components,
        timeout_s=timeout_s,
        price=price,
        started=started,
    )
    return outcome, recorder


@pytest.mark.asyncio
async def test_one_good_turn_is_one_request_and_the_block_is_accepted() -> None:
    outcome, recorder = await _run([HONEST_BLOCK])
    assert len(recorder.requests) == 1
    assert outcome.turns == 1 and [c.title for c in outcome.components] == ["Latency"]
    assert outcome.error == "" and not outcome.rejected


@pytest.mark.asyncio
async def test_a_rejected_block_is_repaired_once_and_then_kept() -> None:
    """Turn 2 is a REPAIR: same block, the validator's errors, and it replaces the failure."""
    repaired = LYING_BLOCK.replace("7000", "143")
    outcome, recorder = await _run([LYING_BLOCK, repaired])
    assert len(recorder.requests) == 2, "exactly one repair turn"
    repair = recorder.requests[1].messages[0]
    repair_text = (
        " ".join(part.text for part in repair.content)
        if isinstance(repair.content, list)
        else str(repair.content)
    )
    assert "Fix only these components" in repair_text and "evidence" in repair_text
    assert outcome.turns == 2 and len(outcome.components) == 1 and outcome.error == ""


@pytest.mark.asyncio
async def test_a_block_that_fails_twice_is_dropped_and_recorded() -> None:
    outcome, recorder = await _run([LYING_BLOCK, LYING_BLOCK])
    assert len(recorder.requests) == 2
    assert not outcome.components and outcome.rejected, "a failed block must never be stored"
    assert outcome.rejected[0].errors


@pytest.mark.asyncio
async def test_the_turn_bound_stops_the_run_cleanly() -> None:
    outcome, recorder = await _run([LYING_BLOCK, LYING_BLOCK, LYING_BLOCK], max_turns=2)
    assert len(recorder.requests) == 2, "maxTurns=2 must not spend a third turn"
    assert outcome.turns == 2
    # R7: the bound leaves a record, and with nothing accepted the row reports it.
    assert any("bound:turns" in line for line in outcome.detail), outcome.detail
    assert outcome.error == "bound:turns"


@pytest.mark.asyncio
async def test_the_clock_checked_between_turns_keeps_the_blocks_that_passed() -> None:
    """R4: an expired wall clock is never spent on a repair turn; the good block survives."""
    outcome, recorder = await _run(
        [HONEST_BLOCK + "\n" + LYING_BLOCK],
        timeout_s=1.0,
        started=time.perf_counter() - 5.0,
    )
    assert len(recorder.requests) == 1, "the repair turn was paid for after the clock ran out"
    assert [c.title for c in outcome.components] == ["Latency"]
    assert any("bound:time" in line for line in outcome.detail), outcome.detail
    assert outcome.error == "", "blocks survived, so the row reports the bound in detail only"
    # Nothing accepted: the same expiry is a failure, and it says which bound.
    empty, _ = await _run([LYING_BLOCK], timeout_s=1.0, started=time.perf_counter() - 5.0)
    assert empty.error == "bound:time" and not empty.components


@pytest.mark.asyncio
async def test_a_turn_at_the_token_cap_leaves_a_record_without_failing_the_job() -> None:
    """R7: the per-turn token bound is a CONSUMPTION bound -- recorded, not a failure."""
    outcome, _ = await _run([HONEST_BLOCK], usage_tokens=6000)
    assert outcome.components, "a truncated-but-valid answer still passes its blocks"
    assert any("bound:tokens" in line for line in outcome.detail), outcome.detail
    assert outcome.error == ""


@pytest.mark.asyncio
async def test_the_repair_turn_is_not_bought_between_turns_when_the_cap_is_reached() -> None:
    """The cap is checked BETWEEN turns (memo §2.5), so turn 1 completes and turn 2 never runs."""
    outcome, recorder = await _run([LYING_BLOCK], max_cost_usd=0.01, price=lambda usage: 5.0)
    assert len(recorder.requests) == 1
    assert any("cap" in line for line in outcome.detail), outcome.detail


@pytest.mark.asyncio
async def test_a_cap_that_fires_keeps_the_blocks_that_already_passed() -> None:
    """Two blocks, one bad: turn 2 would repair the bad one, but the cap is already spent."""
    outcome, recorder = await _run(
        [HONEST_BLOCK + "\n" + LYING_BLOCK], max_cost_usd=0.0001, price=lambda usage: 1.0
    )
    assert len(recorder.requests) == 1
    assert [c.title for c in outcome.components] == ["Latency"], "the good block was dropped"
    # R7a: with blocks surviving, the bound NAME still reaches the journal via ``detail``.
    assert any("bound:cost" in line for line in outcome.detail), outcome.detail


@pytest.mark.asyncio
async def test_NONE_is_an_answer_not_a_failure() -> None:
    outcome, _ = await _run(["NONE"])
    assert outcome.none and not outcome.components and outcome.error == ""


@pytest.mark.asyncio
async def test_more_blocks_than_the_row_may_carry_are_trimmed_and_named() -> None:
    four = "\n".join(HONEST_BLOCK for _ in range(4))
    outcome, _ = await _run([four])
    assert len(outcome.components) == generator.MAX_COMPONENTS
    assert any("components bound" in line for line in outcome.detail)


@pytest.mark.asyncio
async def test_a_transport_failure_fails_the_job_open() -> None:
    class Boom:
        async def __call__(self, request: Any) -> tuple[str, Any]:
            raise RuntimeError("provider died")

    outcome = await generator.generate(
        complete=Boom(),
        model=generator.DesignModel(model_spec(), "ladder", "anthropic/claude-sonnet-5"),
        datasets=DATASETS,
        user_text="q",
        answer_text="a",
        max_turns=2,
        max_output_tokens=6000,
        max_cost_usd=1.0,
    )
    assert outcome.error.startswith("generator:") and not outcome.components


# --- the fork request and the evidence budget ---------------------------------------------


@pytest.mark.asyncio
async def test_the_fork_request_is_isolated_tools_free_and_priceable() -> None:
    recorder = Recorder([HONEST_BLOCK])
    await generator.generate(
        complete=recorder,
        model=generator.DesignModel(model_spec(), "ladder", "anthropic/claude-sonnet-5"),
        datasets=DATASETS,
        user_text="q",
        answer_text="a",
        max_turns=1,
        max_output_tokens=1234,
        max_cost_usd=1.0,
    )
    request = recorder.requests[0]
    assert request.isolated is True and request.replayable is False
    assert request.tools == [] and request.tool_choice == "none"
    assert request.purpose == "supplement_render"
    assert request.max_tokens == 1234
    assert request.system_blocks == [generator.SYSTEM_PROMPT]


def test_the_evidence_budget_is_counted_in_bytes_not_characters() -> None:
    """R6: the memo's caps are BYTE caps; a CJK block that fits 64 KB of characters must
    still respect the 64 KB of bytes (the char-counted budget admitted all six datasets)."""
    big = "宽" * 6600
    datasets = tuple(
        Dataset(
            title="宽字符数据",
            source="cjk.csv",
            columns=("标签",),
            rows=((f"{big}{index}",),),
            n_rows=1,
            numeric_columns=(),
        )
        for index in range(6)
    )
    payloads = [
        json.dumps(
            {
                "id": generator.dataset_id(index, dataset),
                "title": dataset.title,
                "source": dataset.source,
                "columns": list(dataset.columns),
                "rows": [list(row) for row in dataset.rows],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for index, dataset in enumerate(datasets)
    ]
    char_total = sum(len(payload) + 1 for payload in payloads)
    byte_total = sum(len(payload.encode("utf-8")) + 1 for payload in payloads)
    assert char_total <= generator.MAX_EVIDENCE_BYTES, "fixture must fit the CHAR budget"
    assert byte_total > generator.MAX_EVIDENCE_BYTES, "fixture must break the BYTE budget"

    block = generator.evidence_block(datasets, user_text="", answer_text="")
    included = re.findall(r"<dataset>(.*?)</dataset>", block, re.DOTALL)
    assert included, block[:200]
    assert len(included) < len(datasets), "the byte budget must drop the tail"
    included_bytes = sum(len(item.encode("utf-8")) + 1 for item in included)
    assert included_bytes <= generator.MAX_EVIDENCE_BYTES


def test_the_evidence_block_clips_the_user_message_and_the_answer() -> None:
    prompt = generator.evidence_block(DATASETS, user_text="u" * 10_000, answer_text="a" * 20_000)
    # ``_clip`` keeps ``limit - 1`` characters and an ellipsis, so the bound is asserted on
    # the clipping itself rather than on a run of exactly ``limit`` characters.
    assert "u" * (generator.MAX_USER_CHARS - 1) in prompt
    assert "u" * generator.MAX_USER_CHARS not in prompt
    assert "\u2026</request>" in prompt
    assert "a" * (generator.MAX_ANSWER_CHARS - 1) in prompt
    assert "a" * generator.MAX_ANSWER_CHARS not in prompt
    assert "<dataset" in prompt and "bench.csv" in prompt


def test_the_evidence_block_scrubs_credential_shapes() -> None:
    """Memo §2.5: datasets pass the same redaction boundary outbound state does."""
    # The AWS documentation's own dummy access-key id, assembled at runtime so this file carries
    # no credential-SHAPED literal of its own (scanners, and this repo's own hygiene rules).
    secret = "AKIA" + "IOSFODNN7EXAMPLE"
    prompt = generator.evidence_block(DATASETS, user_text="key", answer_text=secret)
    assert secret not in prompt


def test_a_dataset_over_the_size_budget_is_dropped_rather_than_sent_whole() -> None:
    """Memo §2.5's second bound: one dataset over 24 KB is NOT sent, and the block stays small.

    Dropping (rather than truncating mid-row) is the honest choice: a half row would be data
    the model could plot without the row it belonged to, and the evidence block is a budget,
    not a promise that nothing is ever left out.
    """
    fat = Dataset(
        title="Fat",
        source="big.csv",
        columns=("a",),
        rows=tuple((f"{index}",) for index in range(20_000)),
        n_rows=20_000,
        numeric_columns=("a",),
    )
    prompt = generator.evidence_block((fat,), user_text="", answer_text="")
    assert "<dataset>" not in prompt
    assert len(prompt) < generator.MAX_DATASET_BYTES
    small = generator.evidence_block(DATASETS, user_text="", answer_text="")
    assert "<dataset>" in small and len(small) < generator.MAX_EVIDENCE_BYTES


# --- the model ladder ---------------------------------------------------------------------


def test_the_ladder_picks_the_first_usable_design_model() -> None:
    chosen = generator.resolve_design_model(
        configured="auto",
        session_spec=model_spec(),
        usable={"openai", "google"},
        static_models=lambda provider: {"gpt-5.6-sol": object(), "gemini-3.8-flash": object()},
    )
    assert chosen.source == "ladder" and chosen.spec.provider == "openai"


def test_the_ladder_skips_a_provider_with_no_credential() -> None:
    chosen = generator.resolve_design_model(
        configured="auto",
        session_spec=model_spec(),
        usable={"google"},
        # The registry's real answer for both providers: the anthropic ids are known, the
        # google rung's own id is known too -- the CREDENTIAL is what decides here, and a
        # ladder that ignored it would pick anthropic.
        static_models=lambda provider: {
            "claude-sonnet-5": object(),
            "gemini-2.5-pro-preview-05-06": object(),
        },
    )
    assert chosen.spec.provider == "google"


def test_an_operator_pin_outranks_the_ladder() -> None:
    chosen = generator.resolve_design_model(
        configured="openai/gpt-5.6-sol",
        session_spec=model_spec(),
        usable={"openai"},
        static_models=lambda provider: {"gpt-5.6-sol": object()},
    )
    assert chosen.source == "config" and chosen.label == "openai/gpt-5.6-sol"


def test_session_forces_the_session_model() -> None:
    session = model_spec()
    chosen = generator.resolve_design_model(
        configured="session",
        session_spec=session,
        usable={"anthropic"},
        static_models=lambda provider: {},
    )
    assert chosen.spec is session and chosen.source == "session"


def test_a_ladder_no_one_can_reach_falls_back_to_the_tier_then_the_session() -> None:
    session = model_spec()
    tier = model_spec(model_id="gpt-5.6-sol")

    def unknown(provider: str) -> dict[str, Any]:
        # The registry knows these providers and none of the ladder's ids: every rung skips.
        return {"some-other-model": object()}

    chosen = generator.resolve_design_model(
        configured="auto",
        session_spec=session,
        usable={"anthropic", "openai", "google"},
        static_models=unknown,
        tier_spec=tier,
    )
    assert chosen.spec is tier and chosen.source == "tier"
    plain = generator.resolve_design_model(
        configured="auto",
        session_spec=session,
        usable=None,
        static_models=lambda provider: {},
    )
    # ``usable=None`` is "cannot tell", not "none": a provider with NO registry rows is
    # accepted (an aggregator that lists nothing statically still serves models), so the
    # ladder's first rung is tried rather than the session silently winning.
    assert plain.source == "ladder" and plain.spec.provider == "anthropic"
    empty = generator.resolve_design_model(
        configured="auto",
        session_spec=session,
        usable=set(),
        static_models=lambda provider: {},
    )
    assert empty.source == "session" and empty.spec is session


def test_the_memo_and_the_code_agree_on_the_prompt() -> None:
    """A cheap structural guard beside the character-for-character fence comparison."""
    assert "NONE" in generator.SYSTEM_PROMPT and "<component" in generator.SYSTEM_PROMPT
    assert json.dumps(generator.component_bounds()) == "[120, 480]"

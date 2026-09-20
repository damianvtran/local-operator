"""The reasoning phase, end to end through the app.

The block's own tests prove it renders; these prove the app WIRES it — that a
session event mounts it, that the answer's first delta retires it in the right
order, and that the model's private reasoning never lands in the assistant
block. The harness (app, boot, submit, frame readers) is the shipped one from
``test_steering_approval``, reused rather than re-created so these tests cannot
pass against a friendlier app than the one users run.
"""

from __future__ import annotations

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageStart,
    ReasoningDelta,
    TurnStarted,
)
from local_operator.tui.settings import settings_reload
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.reasoning import (
    REASONING_LABEL,
    REASONING_MIN_ROWS,
    REASONING_VISIBLE_ROWS,
    ReasoningBlock,
)
from local_operator.tui.widgets.transcript import UserBlock
from tests.unit.tui.test_steering_approval import (
    SteerableSession,
    _boot,
    _factory,
    _submit,
    _wait_for_row,
    rows,
)


@pytest.mark.asyncio
async def test_reasoning_paints_above_the_answer_and_retires_when_it_starts() -> None:
    """The operator's complaint, as a frame: thinking appears, then the answer.

    Four facts, and each one is a behaviour the wiring could get wrong on its
    own: a ``reasoning_delta`` MOUNTS a block (before this change nothing did),
    the block sits ABOVE the answer it belongs to, the answer's own text is
    untouched by the reasoning, and the phase is retired once the answer starts
    (so a later model call's fragments cannot land in the previous phase's row).
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "think about it")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(ReasoningDelta("weighing the options"))
        await _wait_for_row(pilot, app, "weighing the options")

        reasoning_blocks = app.query(ReasoningBlock)
        assert len(reasoning_blocks) == 1
        phase = reasoning_blocks.first()
        assert app._reasoning_block is phase, "the live phase must be the mounted block"

        app.post_message(AssistantDelta("The answer is 42."))
        await _wait_for_row(pilot, app, "The answer is 42.")

        # Retired, so the NEXT call's thinking opens its own block...
        assert app._reasoning_block is None
        # ...while the painted row stays, collapsed to its header: the phase is
        # over, and its rows belong to nobody now that the answer is here (UX
        # review round 1, U1 — one ~7-row block per model call, kept for the life
        # of the session, was the readability half of that finding).
        assert len(app.query(ReasoningBlock)) == 1
        assert any(REASONING_LABEL in row for row in rows(app))
        assert not any("weighing the options" in row for row in rows(app))

        # Ordering: thinking, then the answer below it. Read from the laid-out
        # regions rather than from append order, because that is what the user
        # sees and the two are only equal if the app got the mount order right.
        answer = app.query_one(AssistantBlock)
        assert phase.region.y < answer.region.y
        # And the reasoning never became the answer's text.
        assert "weighing the options" not in str(answer.renderable)


@pytest.mark.asyncio
async def test_a_squeezed_terminal_keeps_the_question_on_screen() -> None:
    """The row budget yields to the viewport instead of expelling the question.

    At 100x14 the transcript is ~5 rows tall, and the constant six-row bound was
    taller than the whole transcript it lived in: the user's own prompt was
    pushed to y=-3 and a scrollbar appeared, which is the one place this change
    regressed against the base (design round 1, D3).
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 14)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "work out the arrival time")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(ReasoningDelta(" ".join(f"word{index}" for index in range(400))))
        for _ in range(4):
            await pilot.pause()

        block = app.query_one(ReasoningBlock)
        assert block.size.height <= REASONING_MIN_ROWS + 1
        assert block.size.height < REASONING_VISIBLE_ROWS + 1
        # The question the block belongs to is what the budget protects.
        assert app.query_one(UserBlock).region.y >= 0


@pytest.mark.asyncio
async def test_the_reasoning_channel_can_be_switched_off(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch `display.reasoning = false` mounts nothing at all.

    Written through ``settings_io`` rather than by patching ``settings_get``, so
    the flat dotted key the reader actually looks up is the one exercised — a
    nested write would pass a patched test and fail a user (the pattern
    ``display.narration``'s own toggle test established). This is also the one
    setting under which the live transcript and the resumed one agree exactly,
    because reasoning is never durable.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    settings_reload()
    settings_io.write_setting(
        ConfigManager(tmp_path), settings_io.BY_KEY["display.reasoning"], False
    )
    try:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "think about it")
            app.post_message(TurnStarted())
            app.post_message(AssistantMessageStart())
            app.post_message(ReasoningDelta("weighing the options"))
            app.post_message(AssistantDelta("The answer is 42."))
            await _wait_for_row(pilot, app, "The answer is 42.")

            assert len(app.query(ReasoningBlock)) == 0
            assert not any("\u00b7 " + REASONING_LABEL in row for row in rows(app))
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_an_empty_reasoning_flush_mounts_nothing() -> None:
    """No fragment, no block: a header row above nothing is the hole to avoid.

    The controller's equality guard means an empty flush is rare, but a relayed
    or replayed frame can carry an empty delta, and mounting a block for it would
    leave a labelled row that then sits above the answer explaining nothing.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "think about it")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(ReasoningDelta(""))
        app.post_message(AssistantDelta("The answer is 42."))
        await _wait_for_row(pilot, app, "The answer is 42.")

        assert len(app.query(ReasoningBlock)) == 0
        assert not any("· " + REASONING_LABEL in row for row in rows(app))

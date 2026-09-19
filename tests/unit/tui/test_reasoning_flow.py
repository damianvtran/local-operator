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

from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageStart,
    ReasoningDelta,
    TurnStarted,
)
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.reasoning import ReasoningBlock
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
        # ...while the painted row stays put for the reader.
        assert len(app.query(ReasoningBlock)) == 1
        assert any("weighing the options" in row for row in rows(app))

        # Ordering: thinking, then the answer below it. Read from the laid-out
        # regions rather than from append order, because that is what the user
        # sees and the two are only equal if the app got the mount order right.
        answer = app.query_one(AssistantBlock)
        assert phase.region.y < answer.region.y
        # And the reasoning never became the answer's text.
        assert "weighing the options" not in str(answer.renderable)


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
        assert not any("thinking" in row for row in rows(app))

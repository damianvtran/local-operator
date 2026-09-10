"""End-to-end: `ask` reaches the model, and a human can answer it on screen.

Why this file exists
--------------------

#868 shipped for eleven releases with a green unit suite. ``build_ask_tool``
gated on an installed ask hook AND on ``has_ui``, and the detached runtime that
has been the default ``lop`` path since 0.45.0 constructs ``has_ui=False`` and
*then* installs a real ask gate. So the tool was withheld from every ordinary
interactive session while the system prompt instructed the model at length to
use it — and nothing noticed, because every test that asserted the gate either
built a ToolContext by hand or built a session with the flag on.

That is an assembled-application defect: each half was individually correct and
the wiring between them was not. So the coverage belongs at the stage that
drives the assembled application, and the assertions here are deliberately
about the ARRAY THE PROVIDER RECEIVES and the ANSWER THE MODEL READS BACK,
rather than about a builder's return value.

Two host paths, because they install the hook at different times and the fix
has to hold for both:

* the **TUI host** (``OperatorApp._adopt_session`` → ``set_ask_handler``),
  which is the path Ben's verification on the issue explicitly did not cover;
  and
* the **detached runtime** shape (``spawn_owned_session``: ``has_ui=False``
  plus a gate), which is the one the defect was actually reported against.

The provider is scripted, for the reason the module docstring of
``test_tui_e2e`` gives at length: this regression is not model-shaped, and a
stage whose failure signal is "the question was never answerable" must not have
live provider latency inside its bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.session.session import Session
from local_operator.tui.app import OperatorApp
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    drain,
    text_turn,
    tool_call_turn,
    wait_for_adoption,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

SCREEN = (120, 40)
BOUND_S = 60.0

QUESTION = AskQuestion(
    id="deploy",
    question="Deploy to production or roll back?",
    options=[
        AskOption(label="Deploy", description="ships the current head"),
        AskOption(label="Roll back", description="returns to the last tag"),
    ],
)


def _ask_args() -> dict[str, Any]:
    """The arguments a model sends when it calls ``ask``."""
    return {
        "questions": [
            {
                "id": QUESTION.id,
                "question": QUESTION.question,
                "options": [
                    {"label": o.label, "description": o.description} for o in QUESTION.options
                ],
            }
        ]
    }


@pytest.mark.asyncio
async def test_the_tui_host_advertises_ask_and_a_person_can_answer_it(
    headless_tui_env: Path,
) -> None:
    """The TUI path: adoption installs the hook, so ``ask`` must reach the model.

    The session here carries ``has_ui=False`` (the ``Session`` default, which
    ``build_session`` does not override) and that is FAITHFUL rather than a
    convenience: since #576 ``lop``'s own TUI boots as a viewer over the
    detached-runtime machinery, so a real interactive session reaches
    ``_adopt_session`` with the flag off and receives its ask handler there.
    That combination is the defect, and this is the host that ships it.

    Asserted on the tools array of the request the provider actually received.
    A session attribute would not do: the bug lived precisely in the gap
    between "the session has a hook" and "the tool reached the wire".
    """
    stream = ScriptedStream([text_turn("ready")])
    session = build_session(
        headless_tui_env / "sessions" / "ask-tui",
        stream,
        cwd=headless_tui_env,
    )

    async def factory() -> Session:
        return session

    app = OperatorApp(factory)
    try:
        with bounded(BOUND_S, "TUI boot and one turn with `ask` advertised"):
            async with app.run_test(size=SCREEN) as pilot:
                await wait_for_adoption(app, pilot)
                await drain(pilot)

                # The real host installed the real hook during adoption.
                assert session._ask_user is not None, (
                    "the TUI adopted the session without installing an ask handler; "
                    "this test can no longer prove anything about the gate"
                )

                await session.prompt("hello")

                advertised = [tool.name for tool in stream.requests[-1].tools]
                assert "ask" in advertised, (
                    "a TUI session with a human at the keyboard did not advertise "
                    f"`ask`; got {advertised}"
                )
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_the_detached_runtime_shape_advertises_ask_and_answers_it(
    tmp_path: Path,
) -> None:
    """The reported path, end to end: advertised, called, answered, read back.

    ``has_ui=False`` plus an installed handler is exactly what
    ``spawn_owned_session`` builds. The turn script makes the model CALL the
    tool rather than merely be offered it, and the answer is asserted on the
    tool result the next request carries — the round trip, not the gate alone.
    A tool that is advertised but unanswerable would pass an advertisement-only
    assertion and fail this one.
    """
    answered: list[list[AskQuestion]] = []

    async def answer_on_screen(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        """Stands in for the person: picks an option, as the picker does."""
        answered.append(questions)
        return {questions[0].id: ["Roll back"]}

    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Let me confirm which way to go.",
                tool_name="ask",
                tool_call_id="ask-1",
                arguments=_ask_args(),
            ),
            text_turn("Rolling back."),
        ]
    )
    session = build_session(tmp_path / "ask-detached", stream, cwd=tmp_path)
    # The shape the runtime builds: no frontend store, and the gate installed
    # afterwards. ``has_ui`` is a construction argument, so it is set directly;
    # the handler arrives late exactly as ``_install_gates`` delivers it.
    session._has_ui = False
    session.set_ask_handler(answer_on_screen)

    try:
        with bounded(BOUND_S, "a detached-shape session asking and being answered"):
            first = [tool.name for tool in (stream.requests[0].tools if stream.requests else [])]
            await session.prompt("decide the release")

            advertised = [tool.name for tool in stream.requests[0].tools]
            assert "ask" in advertised, (
                "a runtime that CAN mount a question did not advertise `ask`; "
                f"got {advertised} (first: {first})"
            )

            # It was really asked, with the prose intact rather than a placeholder.
            assert answered, "the model called `ask` but the host was never asked"
            assert answered[0][0].question == QUESTION.question

            # And the answer came BACK to the model, which is the half that
            # makes the tool worth advertising at all.
            follow_up = stream.requests[-1]
            transcript = "\n".join((message.text or "") for message in follow_up.messages)
            assert (
                "Roll back" in transcript
            ), "the user's answer never reached the model's next request"
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_host_with_no_one_to_answer_still_gets_no_ask(tmp_path: Path) -> None:
    """The other half of the gate, at the same level.

    Widening the gate from two clauses to one is only safe if the remaining
    clause still refuses every unattended host. A plain ``lop exec`` run, the
    scheduler and a subagent all reach the engine with no handler installed,
    and none of them may be offered a question they could only block on.
    """
    stream = ScriptedStream([text_turn("working")])
    session = build_session(tmp_path / "ask-unattended", stream, cwd=tmp_path)

    try:
        with bounded(BOUND_S, "an unattended session refusing `ask`"):
            await session.prompt("do the work")
            advertised = [tool.name for tool in stream.requests[-1].tools]
            assert (
                "ask" not in advertised
            ), f"a host with no ask handler was offered `ask`; got {advertised}"
    finally:
        await dispose_quietly(session)

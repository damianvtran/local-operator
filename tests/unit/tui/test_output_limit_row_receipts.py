"""The OUTPUT LIMIT row on the TUI: what a resume paints, and how many times.

WHY THIS FILE EXISTS
--------------------
Two findings, one screen.

*The pin* (review round 2, MINOR-1 == QA Q-R2-2). The row substitution this PR
added — ``output_limit_call_receipt(details) or result_text`` — was pinned
nowhere outside ``tests/unit/harness/test_loop.py``, which calls the helper
rather than a surface. Deleting six characters on ANY one of the three row
surfaces stayed green in CI, so the MAJOR defect round 1 raised (a resumed row
carrying model-directed prose as the operator's own receipt, review F2) could
come straight back. ``test_a_resumed_limit_row_carries_the_receipt`` is the TUI
half of that pin, asserted on the row a cold resume actually paints.

*One sentence, one site* (design round 1, D2). Round 1 measured the card body
and the turn notice two rows below it BYTE-IDENTICAL on the cut arm, and the
notice claiming a cut on the arm where nothing was cut (D1). The receipt is now
owned by the call's own row; the notice states the turn, from the arm marker the
turn's own results carry. Both facts are asserted here on a rendered frame,
because "painted once" is a claim about a frame and not about a helper's return
value.

The fold decisions live in ``harness/rows.py``; what this file pins is that the
TUI's replay ASKS them.
"""

from __future__ import annotations

import os

import pytest

from local_operator.harness.loop import (
    LENGTH_ENDED_CALL_RESULT_TEXT,
    TRUNCATED_RESULT_TEXT,
)
from local_operator.harness.rows import output_limit_call_receipt
from local_operator.harness.types import (
    OUTPUT_LIMIT_ARGUMENTS,
    OUTPUT_LIMIT_KEY,
    OUTPUT_LIMIT_TURN,
    Message,
    TextContent,
    ToolCall,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.tool_card import ToolCard
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text

#: The operator-visible lines, spelled out rather than imported: these are copy,
#: and a test that read the module's own constants would follow a reword, which
#: is exactly the change it exists to catch.
_CUT_RECEIPT = "tool call cut off at the output limit (nothing ran)"
_TURN_RECEIPT = "turn cut off at the output limit before this call ran"
_CUT_NOTICE = "turn cut off at the output limit mid tool call — nothing ran"
_TURN_NOTICE = "turn cut off at the output limit — nothing ran"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    # A headless pilot must never touch the operator's real multiplexer or
    # config (the team's rule, and the one a CMUX_WORKSPACE_ID violated once).
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _s: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _s: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _s: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _s: None)
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)


def _limit_result(call_id: str, arm: str, tool: str = "write") -> Message:
    """The synthetic result the loop pairs a never-run call with."""
    model_text = (
        TRUNCATED_RESULT_TEXT if arm == OUTPUT_LIMIT_ARGUMENTS else LENGTH_ENDED_CALL_RESULT_TEXT
    )
    return Message(
        role="tool",
        content=[TextContent(text=model_text)],
        tool_call_id=call_id,
        tool_name=tool,
        is_error=True,
        provider_payload={"details": {OUTPUT_LIMIT_KEY: arm, "__synthetic": True}},
    )


def _limit_history(*arms: str) -> list[Message]:
    """One length-stopped, prose-less turn calling each tool once, one per arm."""
    calls = [
        ToolCall(id=f"c{i}", name="write", arguments={"path": f"f{i}.txt"})
        for i in range(len(arms))
    ]
    assistant = Message(
        role="assistant",
        content=[],
        stop_reason="length",
        provider_payload={},
    )
    assistant.tool_calls = calls
    results = [_limit_result(f"c{i}", arm) for i, arm in enumerate(arms)]
    return [Message.user("go"), assistant, *results]


async def _resumed_screen(app: OperatorApp, pilot) -> str:
    """The frame a COLD RESUME paints, with the tool card opened.

    No manual ``_project_settled_rows`` call: the app's own boot replays the
    session's history through ``replay_tool_call``, which is the path this test
    is about, and projecting on top of it paints every non-card row twice (the
    card dedups by stable id, a user row and a notice do not). It must settle
    first — the boot replay lands a couple of ticks after the pilot starts.
    """
    for _ in range(400):
        await pilot.pause()
        if app._session is not None and app._transcript_view().blocks():
            break
    for _ in range(10):
        await pilot.pause()
    card = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)][0]
    # The expansion is where the card prints the row's own line whole (the
    # status run mirrors it CAPPED, which is its pre-existing contract).
    card._expanded = True
    card._refresh_row()
    for _ in range(5):
        await pilot.pause()
    return _transcript_text(app)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arm", "receipt", "notice"),
    [
        (OUTPUT_LIMIT_ARGUMENTS, _CUT_RECEIPT, _CUT_NOTICE),
        (OUTPUT_LIMIT_TURN, _TURN_RECEIPT, _TURN_NOTICE),
    ],
    ids=["cut-arguments", "complete-arguments"],
)
async def test_a_resumed_limit_row_carries_the_receipt(arm, receipt, notice) -> None:
    """The row a cold resume paints is the harness's receipt, not the model's.

    Reverting ``receipt or result_text`` in ``replay_tool_call`` turns this red:
    the row then carries ``TRUNCATED_RESULT_TEXT``/``LENGTH_ENDED_CALL_RESULT_TEXT``
    — 400-odd characters of imperative addressed to a model, on the operator's
    own screen (review round 1, F2; the pin is review round 2, MINOR-1).
    """
    session = FakeSession()
    history = _limit_history(arm)
    session._history = history
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _resumed_screen(app, pilot)
        (card,) = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)]
        details = (history[2].provider_payload or {}).get("details")

        # The row reads the harness's vocabulary, keyed on the marker.
        assert output_limit_call_receipt(details) == receipt
        assert card._error == receipt
        assert receipt in card._build_content(card._built_width).plain

        # The notice states the TURN, and it is the arm's line, not the row's.
        assert screen.count(receipt) == 1
        assert screen.count(notice) == 1
        assert receipt not in notice and notice not in receipt

        # No arm of the model-facing prose reaches any surface, and the D4 verb
        # ("ended", where this family says "cut off") is retired.
        assert TRUNCATED_RESULT_TEXT not in screen
        assert LENGTH_ENDED_CALL_RESULT_TEXT not in screen
        assert "turn ended at the output limit" not in screen
        assert "oversize" not in screen


@pytest.mark.parametrize(
    ("arms", "notice"),
    [
        ((OUTPUT_LIMIT_ARGUMENTS,), _CUT_NOTICE),
        ((OUTPUT_LIMIT_TURN,), _TURN_NOTICE),
        # MIXED: a turn can dictate one call to completion and die writing a
        # second. The notice names the cut, which really happened in this turn,
        # and each call's own row stays precise about itself.
        ((OUTPUT_LIMIT_TURN, OUTPUT_LIMIT_ARGUMENTS), _CUT_NOTICE),
    ],
    ids=["cut", "complete", "mixed"],
)
@pytest.mark.asyncio
async def test_the_length_notice_reads_the_arm_off_the_turns_own_results(arms, notice) -> None:
    """D1: the turn-level notice must not name a cause the row contradicts.

    Before this, a length-stopped turn whose every call arrived COMPLETE read
    "tool call cut off at the output limit (nothing ran)" two rows under a card
    saying "turn cut off at the output limit before this call ran" (design round
    1, D1; QA Q-R2-1; review round 2, MINOR-2). The fold now reads the arm off
    the turn's own results, and the arm-neutral line is what an unreadable
    record falls back to — asserted on the phone fold, which owns the legacy
    case, in ``tests/unit/mobile/test_fold_parity.py``.
    """
    session = FakeSession()
    history = _limit_history(*arms)
    session._history = history
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _resumed_screen(app, pilot)

        assert notice in screen
        for other in (_CUT_NOTICE, _TURN_NOTICE):
            if other != notice:
                assert other not in screen
        # And the sentence the notice carries is not a second copy of a row's.
        for receipt in (_CUT_RECEIPT, _TURN_RECEIPT):
            assert notice != receipt

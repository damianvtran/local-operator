"""``display.narration``: removing mid-turn narration once its tools have run.

One agentic turn is many model calls, and every call that ends in TOOL CALLS
streamed prose first. That prose is painted by the same block as the final
answer, so a settled transcript interleaves thinking with outcome and the
reader cannot tell which paragraph is the answer.

The flag drops the mid-turn prose at FINALIZE, leaving ``user -> tools ->
answer``. Three properties carry the feature and are pinned here:

* **The classification.** ``tool_calls`` is the WHOLE rule. ``stop_reason``
  deliberately does not corroborate it (``tui/narration.py`` records why: a
  ``toolUse`` whose calls fail to assemble once left the user's prompt followed
  by silence, MAJOR-1), but every TERMINAL stop reason is the ANSWER and must
  survive, because removing a refusal or a length-cut turn erases the outcome
  the user needs.
* **Default ON is byte-identical to today.** The toggle is opt-in, so the
  test that matters most is the one where the flag is left alone and the
  narration STAYS.
* **Live and replay agree.** The live path removes the block after mounting
  it; replay never mounts one. Those are different mechanisms reaching the
  same frame, which is exactly the shape that drifts — so the parity test
  compares the two block sequences directly rather than either in isolation.

The flag decides whether narration is THERE; the RAIL separately decides whether
it wears the answer's mark, and both are set from the same classification of the
same event. A narration block that survives the flag is un-railed, and the tests
below assert BOTH surfaces' rail state — including the pair from one streamed
turn, because a rule that railed everything, or nothing, satisfies either half
alone (PR #1229 railed progress prose exactly as it railed the outcome).

Assertions are on what is in ``view.blocks()``, never on ``remove_block``
having been CALLED: a spy passes even when removal is a no-op.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from textual.content import Content

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.narration import DEFAULT_NARRATION, is_intermediate_narration
from local_operator.tui.widgets.assistant import RAIL, AssistantBlock
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    GAP_CLASS,
    TranscriptView,
    UserBlock,
    WorkingBlock,
    needs_gap_above,
)

from .test_app_pilot import FakeSession, _factory

NARRATION = "Let me check the config first."
ANSWER = "The timeout is 30 seconds."


def _painted(block: AssistantBlock) -> list[str]:
    """The rows ``block`` is painting right now — the frame, not a re-render."""
    visual = block._render()
    assert isinstance(visual, Content)
    return visual.plain.split("\n")


@pytest.fixture()
def narration_hidden(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Turn the flag OFF through the real config file and reader.

    Written through ``settings_io`` rather than by patching ``settings_get``,
    so the flat-dotted key the reader actually looks up is the one exercised —
    a nested write would pass a patched test and fail a user.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_reload

    settings_reload()
    settings_io.write_setting(
        ConfigManager(tmp_path), settings_io.BY_KEY["display.narration"], False
    )
    yield tmp_path
    settings_reload()


def _blocks(app: OperatorApp) -> list[Any]:
    return app.query_one(TranscriptView).blocks()


def _assistant_blocks(app: OperatorApp) -> list[AssistantBlock]:
    return [b for b in _blocks(app) if isinstance(b, AssistantBlock)]


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _stream(
    pilot,
    app: OperatorApp,
    text: str,
    *,
    stop_reason: str | None,
    has_tool_calls: bool,
    delta: bool = True,
) -> None:
    """Paint one model call through the REAL event path.

    Events rather than a hand-mounted block: the mount, the finalize and the
    removal are all owned by these handlers, so constructing a block directly
    would test a transcript the app never builds.
    """
    app.post_message(AssistantMessageStart())
    await pilot.pause()
    if delta:
        app.post_message(AssistantDelta(text))
        await pilot.pause()
    app.post_message(
        AssistantMessageEnd(text, stop_reason=stop_reason, has_tool_calls=has_tool_calls)
    )
    await pilot.pause()
    await pilot.pause()


# --------------------------------------------------------------------------
# Classification — pure
# --------------------------------------------------------------------------


def test_a_message_with_tool_calls_is_narration() -> None:
    assert is_intermediate_narration(stop_reason="toolUse", has_tool_calls=True) is True


def test_tool_calls_alone_classify_as_narration() -> None:
    """A provider that reports calls without the stop reason still continues."""
    assert is_intermediate_narration(stop_reason=None, has_tool_calls=True) is True


def test_the_stop_reason_alone_is_not_enough_to_hide_the_prose() -> None:
    """``toolUse`` with NO calls must KEEP the prose (review MAJOR-1).

    Reachable, not hypothetical: ``providers/clients.py`` maps
    ``finish_reason`` to ``stop_reason`` before the calls are assembled and
    ``harness/loop.py`` assigns the two independently, so a provider reporting
    ``finish_reason=tool_calls`` whose arguments fail to assemble lands here.

    Accepting the reason alone removed the prose while the tool loop mounted
    nothing and ``assistant_stop_notice`` returned None for ``toolUse`` — the
    user's prompt followed by silence. Hiding narration is only justified when
    tool activity supersedes it; with no calls, nothing does.
    """
    assert is_intermediate_narration(stop_reason="toolUse", has_tool_calls=False) is False


def test_a_plain_answer_is_not_narration() -> None:
    assert is_intermediate_narration(stop_reason="stop", has_tool_calls=False) is False


@pytest.mark.parametrize("stop_reason", ["length", "refusal", "error", "aborted"])
def test_a_terminal_stop_reason_is_never_narration(stop_reason: str) -> None:
    """Each of these ENDS the turn: removing it would erase the outcome.

    A refusal the user never sees reads as the agent ignoring them, and a
    length-cut half sentence removed entirely reads as silence.
    """
    assert is_intermediate_narration(stop_reason=stop_reason, has_tool_calls=False) is False


# --------------------------------------------------------------------------
# Live path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_narration_is_removed_when_the_call_finalizes_into_tool_calls(
    narration_hidden,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        assert _assistant_blocks(app) == [], "the narration block is still mounted"


@pytest.mark.asyncio
async def test_the_final_answer_survives_after_narration_was_removed(
    narration_hidden,
) -> None:
    """The point of the feature: what is LEFT is the answer, and only it."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        await _stream(pilot, app, ANSWER, stop_reason="stop", has_tool_calls=False)
        remaining = _assistant_blocks(app)
        assert len(remaining) == 1
        assert remaining[0].text().strip() == ANSWER


@pytest.mark.asyncio
async def test_narration_stays_when_the_toggle_is_on() -> None:
    """THE byte-identical-to-today guarantee — no fixture, the shipped default.

    Nothing writes the key here, so the reader falls back to the registry
    default. A change that removed narration unconditionally would pass every
    test above and fail this one.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        remaining = _assistant_blocks(app)
        assert len(remaining) == 1
        assert remaining[0].text().strip() == NARRATION


@pytest.mark.asyncio
async def test_a_turn_that_promised_calls_but_made_none_still_says_something(
    narration_hidden,
) -> None:
    """MAJOR-1, at the FRAME: this turn must not render as silence.

    The pure classification test above pins the rule; this pins the
    consequence, which is the thing the user actually suffered. Nothing else
    rescues this shape — the tool loop has no calls to mount and
    ``assistant_stop_notice`` returns None for ``toolUse`` — so if the prose
    goes, the prompt is followed by nothing at all.

    Asserted as "the prose is on screen" rather than "a block exists": a turn
    that mounted an EMPTY block would satisfy the weaker claim and still show
    the user nothing.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        view.append_block(UserBlock("what is the gate timeout?"))
        await pilot.pause()
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=False)
        remaining = _assistant_blocks(app)
        assert len(remaining) == 1, "the only thing the turn produced was swept"
        assert remaining[0].text().strip() == NARRATION


@pytest.mark.asyncio
async def test_a_truncated_turn_is_never_removed(narration_hidden) -> None:
    """An abort is what the user is left READING; it must survive the sweep.

    Guards the empty-text branch: it marks truncated and finalizes, and no
    narration removal may reach it — the turn produced no authoritative text,
    so there is no "answer" coming to replace what is on screen.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        app.post_message(AssistantDelta("half a senten"))
        await pilot.pause()
        app.post_message(AssistantMessageEnd(""))
        await pilot.pause()
        await pilot.pause()
        remaining = _assistant_blocks(app)
        assert len(remaining) == 1
        assert remaining[0].is_truncated() is True


@pytest.mark.asyncio
async def test_a_tool_only_turn_removes_nothing(narration_hidden) -> None:
    """A call that streamed NO prose never mounted a block to remove.

    ``on_assistant_message_start`` defers the mount, so the removal path must
    cope with there being nothing at all — and must not reach past it to the
    user row above.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        user = UserBlock("check the config")
        view.append_block(user)
        await pilot.pause()
        before = list(view.blocks())
        await _stream(pilot, app, "", stop_reason="toolUse", has_tool_calls=True, delta=False)
        assert _assistant_blocks(app) == []
        assert list(view.blocks()) == before, "an unrelated block was disturbed"


@pytest.mark.asyncio
async def test_the_working_line_survives_a_narration_removal(narration_hidden) -> None:
    """The WorkingBlock is pinned and TRANSIENT; a removal must not take it.

    It is the only live thing on screen while the tools that follow the
    narration run, so losing it to the sweep would leave the user watching a
    still frame during the longest part of the turn.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        working = WorkingBlock()
        view.append_block(working)
        view.pin_tail(working)
        await pilot.pause()
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        assert _assistant_blocks(app) == []
        assert working in view.blocks(), "the working line was swept with the narration"
        assert working.is_mounted


@pytest.mark.asyncio
async def test_the_gap_is_recomputed_after_a_removal(narration_hidden) -> None:
    """Removing a block changes what its NEIGHBOUR sits under.

    The tool card was preceded by narration and is now preceded by the user
    prompt, so its gap must be the one ``needs_gap_above`` answers for THAT
    pair — a stale gap leaves a visible hole where the narration was.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        prompt = UserBlock("check the config")
        view.append_block(prompt)
        await pilot.pause()
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        card = ToolCard("call-1", "bash", {"command": "echo hi"})
        view.append_block(card)
        await pilot.pause()
        assert _assistant_blocks(app) == []
        assert card.has_class(GAP_CLASS) is needs_gap_above(prompt, card)


# --------------------------------------------------------------------------
# Live path — the rail marks the ANSWER
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_narration_keeps_its_prose_and_loses_the_rail() -> None:
    """The report this change answers, on the live path.

    Under the shipped defaults ``display.narration`` ON leaves the block
    mounted, so the rail is the ONLY thing that could tell a progress sentence
    from the answer — and PR #1229 painted it on both. Finalized into tool
    calls, the block paints no gutter and its prose starts at column 0.

    No fixture: this is the shipped default, so a change that de-railed nothing
    would pass every fixture-driven test above and fail here.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        blocks = _assistant_blocks(app)
        assert len(blocks) == 1, "narration must stay mounted under the default"
        assert blocks[0].text().strip() == NARRATION
        assert blocks[0].is_narration() is True
        rows = _painted(blocks[0])
        assert sum(1 for row in rows if row.strip()) >= 1, rows
        assert all(
            not row.startswith(RAIL) for row in rows
        ), f"a progress sentence still carries the answer's rail: {rows!r}"


@pytest.mark.asyncio
async def test_live_the_pair_is_what_discriminates_progress_from_the_answer() -> None:
    """Same turn, two prose blocks, opposite marks — the whole point.

    A rule that railed nothing, or one that railed everything, would satisfy
    either half of this alone. Both blocks are asserted, in order, from one
    streamed turn.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        await _stream(pilot, app, ANSWER, stop_reason="stop", has_tool_calls=False)
        blocks = _assistant_blocks(app)
        assert [block.text().strip() for block in blocks] == [NARRATION, ANSWER]
        progress, answer = blocks
        assert progress.is_narration() is True and answer.is_narration() is False
        assert all(not row.startswith(RAIL) for row in _painted(progress))
        answer_rows = _painted(answer)
        assert sum(1 for row in answer_rows if row.strip()) >= 1, answer_rows
        assert all(row.startswith(RAIL) for row in answer_rows), answer_rows


@pytest.mark.asyncio
async def test_a_truncated_progress_message_is_unrailed_too() -> None:
    """The empty-text branch classifies from ITS OWN two fields.

    A provider that aborts mid-sentence with calls following leaves a block
    that survived ``mark_truncated`` — and it is still PROGRESS: the turn goes
    on and the answer is still coming. Classifying it as an answer would put the
    rail back on the very narration this change removes, on the one path where
    the block is built by a different branch.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        app.post_message(AssistantDelta(NARRATION[:12]))
        await pilot.pause()
        app.post_message(AssistantMessageEnd("", stop_reason="toolUse", has_tool_calls=True))
        await pilot.pause()
        await pilot.pause()
        blocks = _assistant_blocks(app)
        assert len(blocks) == 1
        assert blocks[0].is_truncated() is True
        assert blocks[0].is_narration() is True
        assert all(not row.startswith(RAIL) for row in _painted(blocks[0]))


@pytest.mark.asyncio
async def test_a_truncated_ANSWER_keeps_the_rail() -> None:
    """The control for the branch above, and the reason it reads its own fields.

    No authoritative text and NO calls: nothing follows this message, so its
    prose is the only thing the turn produced and the rail has to stay — that is
    what marks it as all there is. Same branch, opposite outcome, decided by the
    event's own ``has_tool_calls``.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        app.post_message(AssistantDelta(NARRATION[:12]))
        await pilot.pause()
        app.post_message(AssistantMessageEnd("", stop_reason="aborted", has_tool_calls=False))
        await pilot.pause()
        await pilot.pause()
        blocks = _assistant_blocks(app)
        assert len(blocks) == 1
        assert blocks[0].is_truncated() is True
        assert blocks[0].is_narration() is False
        rows = _painted(blocks[0])
        assert all(row.startswith(RAIL) for row in rows), rows


# --------------------------------------------------------------------------
# Replay parity — the non-negotiable one
# --------------------------------------------------------------------------


def _settled_history() -> list[Any]:
    """The same turn the live events above describe, as settled history."""
    return [
        SimpleNamespace(
            role="user",
            id="u-1",
            text="check the config",
            tool_calls=None,
            content=[],
            custom_type=None,
        ),
        SimpleNamespace(
            role="assistant",
            id="a-1",
            text=NARRATION,
            tool_calls=[
                SimpleNamespace(id="call-1", name="bash", arguments={"command": "echo hi"})
            ],
            custom_type=None,
            stop_reason="toolUse",
            provider_payload=None,
        ),
        SimpleNamespace(
            role="tool",
            id="t-1",
            tool_call_id="call-1",
            text="exit code: 0\nhi",
            is_error=False,
            provider_payload=None,
            content=[],
            custom_type=None,
        ),
        SimpleNamespace(
            role="assistant",
            id="a-2",
            text=ANSWER,
            tool_calls=[],
            custom_type=None,
            stop_reason="stop",
            provider_payload=None,
        ),
    ]


def _shape(blocks: list[Any]) -> list[tuple[str, str]]:
    """Block TYPES and TEXTS — the comparable thing, not a screenshot."""
    shape: list[tuple[str, str]] = []
    for block in blocks:
        if isinstance(block, (AssistantBlock, UserBlock)):
            shape.append((type(block).__name__, block.text().strip()))
        elif isinstance(block, ToolCard):
            shape.append((type(block).__name__, ""))
    return shape


@pytest.mark.asyncio
async def test_replay_hides_the_same_narration_the_live_path_hides(
    narration_hidden,
) -> None:
    """Live removes after mounting; replay never mounts. Same frame, or a
    resumed session shows rows the live one deliberately dropped.

    Different MECHANISMS on purpose — which is precisely why the sequences are
    compared to each other rather than each to a hand-written expectation that
    could be wrong in the same direction twice.
    """
    replay_app = OperatorApp(lambda: _factory(FakeSession()))
    async with replay_app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, replay_app)
        replay_app._project_settled_rows(_settled_history())
        for _ in range(10):
            await pilot.pause()
        replay_shape = _shape(_blocks(replay_app))

    live_app = OperatorApp(lambda: _factory(FakeSession()))
    async with live_app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, live_app)
        view = live_app.query_one(TranscriptView)
        view.append_block(UserBlock("check the config"))
        await pilot.pause()
        await _stream(pilot, live_app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
        view.append_block(ToolCard("call-1", "bash", {"command": "echo hi"}))
        await pilot.pause()
        await _stream(pilot, live_app, ANSWER, stop_reason="stop", has_tool_calls=False)
        live_shape = _shape(_blocks(live_app))

    assert replay_shape == live_shape
    # And it is the INTERESTING sequence, not two empty lists agreeing.
    assert ("AssistantBlock", ANSWER) in replay_shape
    assert ("AssistantBlock", NARRATION) not in replay_shape


@pytest.mark.asyncio
async def test_replay_keeps_narration_when_the_toggle_is_on() -> None:
    """The replay half of the opt-in guarantee: default ON changes nothing."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._project_settled_rows(_settled_history())
        for _ in range(10):
            await pilot.pause()
        shape = _shape(_blocks(app))
        assert ("AssistantBlock", NARRATION) in shape
        assert ("AssistantBlock", ANSWER) in shape


@pytest.mark.asyncio
async def test_replay_unrails_the_narration_and_rails_the_answer() -> None:
    """A resumed session must reproduce the UN-RAILED progress sentence.

    This module's whole purpose is that the live and the replayed transcript
    cannot disagree, and the rail is now part of what they have to agree about:
    a replay that mounted every prose block railed would show the answer's mark
    on progress the moment a session was resumed — a frame that never existed
    when it was live.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._project_settled_rows(_settled_history())
        for _ in range(10):
            await pilot.pause()
        blocks = _assistant_blocks(app)
        assert [block.text().strip() for block in blocks] == [NARRATION, ANSWER]
        progress, answer = blocks
        assert progress.is_narration() is True and answer.is_narration() is False
        assert all(not row.startswith(RAIL) for row in _painted(progress))
        assert all(row.startswith(RAIL) for row in _painted(answer))


@pytest.mark.asyncio
async def test_the_two_paths_agree_about_which_block_earned_the_rail() -> None:
    """The rail state, compared ACROSS the two surfaces rather than asserted.

    ``test_replay_hides_the_same_narration_the_live_path_hides`` compares block
    types and texts; the mark is a third thing both surfaces decide, from the
    same classification, and this is the comparison that fails when one of them
    forgets to apply it — which is exactly how a replay comes to show a rail the
    live frame never had.
    """
    railed_by_surface: dict[str, dict[str, bool]] = {}
    for surface in ("replay", "live"):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            if surface == "replay":
                app._project_settled_rows(_settled_history())
                for _ in range(10):
                    await pilot.pause()
            else:
                view = app.query_one(TranscriptView)
                view.append_block(UserBlock("check the config"))
                await pilot.pause()
                await _stream(pilot, app, NARRATION, stop_reason="toolUse", has_tool_calls=True)
                view.append_block(ToolCard("call-1", "bash", {"command": "echo hi"}))
                await pilot.pause()
                await _stream(pilot, app, ANSWER, stop_reason="stop", has_tool_calls=False)
            railed_by_surface[surface] = {
                block.text().strip(): any(row.startswith(RAIL) for row in _painted(block))
                for block in _assistant_blocks(app)
            }

    assert railed_by_surface["live"] == railed_by_surface["replay"]
    # And the agreed answer is the interesting one, not two empty maps.
    assert railed_by_surface["live"] == {NARRATION: False, ANSWER: True}


def test_the_registry_default_is_the_module_constant() -> None:
    """Pins the two together by CONTENT, both directions.

    ``assert DEFAULT_NARRATION is True`` alone would pin nothing — it compares
    a constant to a literal and passes against whatever someone later writes.
    """
    assert settings_io.BY_KEY["display.narration"].default == DEFAULT_NARRATION
    assert DEFAULT_NARRATION is True

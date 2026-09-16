"""``display.narration``: removing mid-turn narration once its tools have run.

One agentic turn is many model calls, and every call that ends in TOOL CALLS
streamed prose first. That prose is painted by the same block as the final
answer, so a settled transcript interleaves thinking with outcome and the
reader cannot tell which paragraph is the answer.

The flag drops the mid-turn prose at FINALIZE, leaving ``user -> tools ->
answer``. Three properties carry the feature and are pinned here:

* **The classification.** Tool calls OR ``stop_reason == "toolUse"`` means
  narration; every other stop reason is FINAL and must survive, because
  removing a refusal or a length-cut turn erases the outcome the user needs.
* **Default ON is byte-identical to today.** The toggle is opt-in, so the
  test that matters most is the one where the flag is left alone and the
  narration STAYS.
* **Live and replay agree.** The live path removes the block after mounting
  it; replay never mounts one. Those are different mechanisms reaching the
  same frame, which is exactly the shape that drifts — so the parity test
  compares the two block sequences directly rather than either in isolation.

Assertions are on what is in ``view.blocks()``, never on ``remove_block``
having been CALLED: a spy passes even when removal is a no-op.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.narration import DEFAULT_NARRATION, is_intermediate_narration
from local_operator.tui.widgets.assistant import AssistantBlock
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


def test_the_stop_reason_alone_classifies_as_narration() -> None:
    """And one that reports the stop reason before the calls are parsed."""
    assert is_intermediate_narration(stop_reason="toolUse", has_tool_calls=False) is True


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


def test_the_registry_default_is_the_module_constant() -> None:
    """Pins the two together by CONTENT, both directions.

    ``assert DEFAULT_NARRATION is True`` alone would pin nothing — it compares
    a constant to a literal and passes against whatever someone later writes.
    """
    assert settings_io.BY_KEY["display.narration"].default == DEFAULT_NARRATION
    assert DEFAULT_NARRATION is True

"""A genuine OperatorApp/Pilot click, not a direct action-function call."""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest
from rich.text import Text

from local_operator.harness.types import TextContent
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.tool_card import ToolCard
from scripts.visual_capture import save_capture
from tests.unit.session.test_account_profile_consent import SEED_INSTRUCTIONS
from tests.unit.session.test_account_profile_consent import (  # noqa: F401
    stores as consent_stores,
)
from tests.unit.session.test_profile_actions import saved
from tests.unit.tui.test_app_pilot import Editor, _await_session, _factory

stores = consent_stores
_EVIDENCE_ROOT = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")

# Reused from PR #3652 (0285063b), the owner repair for this pre-existing CI
# failure; keep its settle helper and failure controls together, not a new wait.
# States a revert can settle in. "pending" is the in-flight latch and
# "available" is the pre-press state, so neither counts as settled.
_TERMINAL_PROFILE_ACTION_STATES = frozenset({"succeeded", "failed", "stale"})
# Generous next to a sqlite round trip, and cheap when the state has already
# landed: the loop returns on its first terminal read.
_PROFILE_ACTION_SETTLE_S = 15.0
# How long one poll yields for. Long enough for an executor hop to make
# progress, short enough that a settled revert is noticed promptly.
_PROFILE_ACTION_SETTLE_POLL_S = 0.01
_WORKER_DRAIN_S = 30.0


async def _await_profile_action_settled(pilot, card) -> str:
    """Pause until the card's profile-action state reaches a terminal value.

    A revert is not one loop turn of work. The press posts
    ``on_tool_card_profile_revert_requested``, which only posts a worker and
    returns; that worker awaits ``asyncio.to_thread(...activate...)`` (an
    executor round trip) and only then awaits ``_refresh_profile_action``, whose
    status read is what finally calls ``set_profile_action_state``. A single
    ``pilot.pause()`` yields to the message pump and does not wait for a thread
    (the hazard ``test_app_pilot._await_session`` documents), so one pause lands
    between those hops on a loaded runner and reads the "pending" latch — that
    was the CI failure on the revert path.

    The bound is wall-clock rather than a count of pauses, because what it waits
    on is an executor hop whose latency scales with machine load: a fixed turn
    count is the same bet that lost. A state that never settles still FAILS here
    (``pytest.fail``), and the caller keeps asserting *which* terminal state it
    must be, so a silent pass is not possible.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _PROFILE_ACTION_SETTLE_S
    while True:
        await pilot.pause(_PROFILE_ACTION_SETTLE_POLL_S)
        if card._profile_action_state in _TERMINAL_PROFILE_ACTION_STATES:
            return card._profile_action_state
        if loop.time() >= deadline:
            pytest.fail(
                "profile action never reached a terminal state within "
                f"{_PROFILE_ACTION_SETTLE_S}s; last state "
                f"{card._profile_action_state!r}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [110, 60])
@pytest.mark.parametrize("activation", ["mouse", "keyboard"])
async def test_real_operator_revert_click_and_composer_negative_control(
    stores, columns, tmp_path, activation
):
    session, action, _ = await saved(stores)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, 30)) as pilot:
        await _await_session(app, pilot)
        card = ToolCard("trusted-call", "agent", {"op": "update", "name": "reviewer"})
        card.mark_done("Updated agent 'reviewer'.", details={"profile_action": action})
        # A saved verification flag is NOT a live acknowledgment.
        assert card._profile_action_state == "unavailable"
        app._append_block(card)
        for _ in range(12):
            await pilot.pause()
            if card._profile_action_state == "available":
                break
        assert card._profile_action_state == "available"
        row = stores["account"].get_agent_by_name("reviewer")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("r", "R")
        await pilot.pause()
        assert stores["account"].get_agent_system_prompt(row.id) == "Demand tests."
        assert card._profile_action_state == "available"
        assert editor.text == "rR"  # genuine composer input, not a no-op
        # Clear the composer so keyboard input cannot later become a prompt.
        editor.text = ""
        directory = Path(_EVIDENCE_ROOT or tmp_path) / "profile-action-frames"
        directory.mkdir(parents=True, exist_ok=True)
        save_capture(app, directory / f"before-{columns}-{activation}.svg")
        expanded = card.expanded
        # Revert is its own second-row hit target. The click must not expand
        # the parent tool row, and its state latch defeats same-batch repeats.
        if activation == "mouse":
            await pilot.click(card, offset=(7, 1))
        else:
            card.focus()
            await pilot.pause()
            assert card.has_focus
            await pilot.press("r", "R")
        await _await_profile_action_settled(pilot, card)
        assert card.expanded == expanded
        assert card._profile_action_state == "succeeded"
        assert stores["account"].get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS
        save_capture(app, directory / f"after-{columns}-{activation}.svg")
        # A duplicate focused key cannot reexecute the old inverse after a
        # later legitimate change, even in the same batch of keyboard events.
        stores["account"].set_agent_system_prompt(row.id, "After revert, new edit.")
        card.focus()
        await pilot.press("r", "R")
        await pilot.pause()
        assert stores["account"].get_agent_system_prompt(row.id) == "After revert, new edit."
        # Drain before the loop closes: a revert worker still in flight here
        # surfaces as RuntimeError('Event loop is closed') from teardown and
        # would mask the assertions above rather than naming them.
        await asyncio.wait_for(app.workers.wait_for_complete(), _WORKER_DRAIN_S)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["no-live-proof", "dropped-response", "dropped-response-and-status"]
)
async def test_live_proof_and_transport_uncertainty_never_enable_a_blind_retry(
    stores, mode, monkeypatch
):
    from local_operator.profile_actions import ProfileActions

    session, action, _ = await saved(stores)
    live = session._profile_actions
    if mode == "no-live-proof":
        # Model-writable transcript/backup flags survive, host memory does not.
        session._profile_actions = ProfileActions(session._session_id, stores["account"].config_dir)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 30)) as pilot:
        await _await_session(app, pilot)
        card = ToolCard("trusted-call", "agent", {"op": "update", "name": "reviewer"})
        card.mark_done("Saved.", details={"profile_action": action, "revert_verified": True})
        app._append_block(card)
        await pilot.pause()
        row = stores["account"].get_agent_by_name("reviewer")
        if mode == "no-live-proof":
            assert card._profile_action_state == "unavailable"
            await pilot.click(card, offset=(7, 1))
            card.focus()
            await pilot.press("r", "R")
            await pilot.pause()
            assert stores["account"].get_agent_system_prompt(row.id) == "Demand tests."
            assert not card.expanded
        else:
            assert card._profile_action_state == "available"

            class LostReply:
                calls = 0
                status_lost = mode == "dropped-response-and-status"

                async def profile_revert(self, request_id, *, approved=True):
                    self.calls += 1
                    live.activate(request_id, approved=approved, remember=False)
                    raise TimeoutError("synthetic lost reply AFTER actual apply")

                async def profile_action_status(self, **query):
                    if self.status_lost:
                        raise TimeoutError("synthetic lost status response")
                    return live.status(**query)

            client = LostReply()
            with monkeypatch.context() as scoped:
                scoped.setattr(session, "_client", client, raising=False)
                await pilot.click(card, offset=(7, 1))
                await pilot.pause()
                if client.status_lost:
                    assert card._profile_action_state == "pending"
                    assert isinstance(card._content, Text)
                    assert "Check status" in card._content.plain
                    assert client.calls == 1
                    client.status_lost = False
                    await pilot.click(card, offset=(7, 1))  # read, never reapply
                    await pilot.pause()
                assert card._profile_action_state == "succeeded"
                assert stores["account"].get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS
                assert client.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [110, 60])
@pytest.mark.parametrize("profile", [True, False])
async def test_header_mouse_click_toggles_exactly_once(stores, columns, profile):
    session, action, _ = await saved(stores)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, 30)) as pilot:
        await _await_session(app, pilot)
        card = ToolCard(
            "trusted-call", "agent" if profile else "bash", {"op": "update", "name": "reviewer"}
        )
        card.mark_done(
            "A populated result body.", details={"profile_action": action} if profile else {}
        )
        app._append_block(card)
        await pilot.pause()
        assert not card.expanded
        await pilot.click(card, offset=(18, 0))
        await pilot.pause()
        assert card.expanded
        await pilot.click(card, offset=(18, 0))
        await pilot.pause()
        assert not card.expanded


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [110, 60])
@pytest.mark.parametrize("state", ["unavailable", "stale", "failed", "pending"])
async def test_disabled_action_states_render_safe_guidance(
    stores, columns, state, tmp_path, monkeypatch
):
    import json

    from local_operator.profile_actions import ProfileActions

    session, action, backup = await saved(stores)
    if state == "unavailable":
        session._profile_actions = ProfileActions(session._session_id, stores["account"].config_dir)
    elif state == "stale":
        row = stores["account"].get_agent_by_name("reviewer")
        stores["account"].set_agent_system_prompt(row.id, "Later edit, never undo this.")
    elif state == "failed":
        (stores["session_dir"] / "profile-backups" / (backup + ".json")).write_text("{}")
        assert (
            session._profile_actions.activate(action["action_id"], approved=True, remember=False)[
                "state"
            ]
            == "failed"
        )
    else:
        path = (
            stores["account"].config_dir
            / "profile-transactions"
            / ("revert_" + action["action_id"] + ".json")
        )
        raw = json.loads(path.read_text())
        raw["state"] = "applying"
        path.write_text(json.dumps(raw))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, 30)) as pilot:
        await _await_session(app, pilot)
        card = ToolCard("trusted-call", "agent", {"op": "update", "name": "reviewer"})
        card.mark_done("Saved profile receipt.", details={"profile_action": action})
        app._append_block(card)
        for _ in range(12):
            await pilot.pause()
            if card._profile_action_checked:
                break
        assert card._profile_action_state == state
        if state != "pending":
            assert isinstance(card._content, Text)
            assert "Inspect" in card._content.plain
        else:
            assert isinstance(card._content, Text)
            assert "Check status" in card._content.plain
        directory = Path(_EVIDENCE_ROOT or tmp_path) / "profile-action-r3-frames"
        directory.mkdir(parents=True, exist_ok=True)
        save_capture(app, directory / f"{state}-{columns}.svg")
        await pilot.click(card, offset=(7, 1))
        await pilot.pause()
        assert not card.expanded


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [110, 60])
async def test_actual_saved_receipt_body_is_readable_and_header_opens_once(
    stores, columns, tmp_path
):
    import json

    from local_operator.tui import bindings

    session, action, _, result = await saved(stores, receipt=True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, 30)) as pilot:
        await _await_session(app, pilot)
        card = ToolCard(result.tool_call_id, result.tool_name, {"op": "update", "name": "reviewer"})
        card.mark_done(
            "\n".join(item.text for item in result.content if isinstance(item, TextContent)),
            details=result.details,
        )
        app._append_block(card)
        for _ in range(12):
            await pilot.pause()
            if card._profile_action_checked:
                break
        await pilot.click(card, offset=(18, 0))
        await pilot.pause()
        assert card.expanded
        assert isinstance(card._content, Text)
        assert "instructions" in card._content.plain
        directory = Path(_EVIDENCE_ROOT or tmp_path) / "profile-action-r3-frames"
        directory.mkdir(parents=True, exist_ok=True)
        save_capture(app, directory / f"expanded-{columns}.svg")
        style = bindings.style("tool.row.chip_settled")
        assert style.color is not None
        foreground = style.color.get_truecolor()
        background = style.bgcolor or card.rich_style.bgcolor
        assert background is not None
        background = background.get_truecolor()

        def luminance(rgb):
            values = [v / 255 for v in rgb]
            values = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
            return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]

        a, b = sorted([luminance(foreground), luminance(background)])
        ratio = (b + 0.05) / (a + 0.05)
        (directory / f"contrast-{columns}.json").write_text(
            json.dumps(
                {
                    "binding": "tool.row.chip_settled",
                    "foreground": list(foreground),
                    "background": list(background),
                    "contrast_ratio": ratio,
                    "card_region": list(card.region),
                    "content_region": list(card.content_region),
                    "virtual_size": list(card.virtual_size),
                    "actual_size": list(card.size),
                    "vertical_scrollbar": card.show_vertical_scrollbar,
                },
                indent=2,
            )
        )
        assert ratio >= 4.5


# The settle helper above is the thing that decides whether the revert tests
# wait long enough, so its bound is worth a control of its own: a state that
# never arrives must FAIL on the bound (not pass, not hang), and a state that
# does arrive must return promptly. Both run without the app, because the
# property under test is the wait, and booting a Pilot to prove it would make
# the control as load-sensitive as the test it guards.


class _FakePilot:
    """Just the pump surface the helper uses; ``delay`` is ignored on purpose."""

    def __init__(self, on_pause=None) -> None:
        self._on_pause = on_pause
        self.pauses = 0

    async def pause(self, delay: float | None = None) -> None:
        self.pauses += 1
        if self._on_pause is not None:
            self._on_pause(self.pauses)


class _FakeCard:
    """Only the latch the helper reads."""

    def __init__(self, state: str = "pending") -> None:
        self._profile_action_state = state


@pytest.mark.asyncio
async def test_settle_helper_returns_once_the_latch_reaches_a_terminal_state(monkeypatch):
    """Positive control: without this, the bound test below could pass for a
    helper that simply always fails."""
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_PROFILE_ACTION_SETTLE_S", 5.0)
    card = _FakeCard("pending")

    def advance(pauses: int) -> None:
        if pauses >= 3:
            card._profile_action_state = "succeeded"

    pilot = _FakePilot(advance)
    assert await _await_profile_action_settled(pilot, card) == "succeeded"
    assert pilot.pauses == 3


@pytest.mark.asyncio
async def test_settle_helper_fails_on_its_bound_when_the_state_never_settles(monkeypatch):
    """Negative control: the status read this delays on never lands.

    The card parks at the "pending" latch the CI failure observed. The helper
    must exhaust its bound and FAIL, naming the state it was stuck in — the
    assertion in the revert test must never be able to pass on a revert that
    did not actually happen, and the wait must not turn a stuck state into a
    hang either.
    """
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_PROFILE_ACTION_SETTLE_S", 0.05)
    card = _FakeCard("pending")
    pilot = _FakePilot()
    started = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as raised:
        await _await_profile_action_settled(pilot, card)
    elapsed = time.monotonic() - started
    message = str(raised.value)
    assert "never reached a terminal state" in message
    assert "'pending'" in message  # names what it was stuck in
    assert card._profile_action_state == "pending"
    # Bounded, not hung: a stuck state costs the bound, then fails.
    assert elapsed < 10.0
    assert pilot.pauses > 1

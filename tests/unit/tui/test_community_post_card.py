"""The agent-post approval card never approves text the owner has not seen whole (R1-5).

Driven through the REAL ``OperatorApp`` and the gate it installs on the
session, the way a ``community_post`` tool call reaches it: the card keeps the
audience and "shown as your own post" rows above the text, ``y`` on a cut card
opens the whole post (``CommunityPostScreen``: pinned header, scrolling text)
instead of approving, and only once that screen has been scrolled to its end
does ``y`` allow. Rendered frames: ``scripts/community_shot.py post-card-*``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.community.agent_ops import post_card
from local_operator.harness.approval import COMMUNITY_POST_GATE
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.approval import ApprovalPrompt
from local_operator.tui.widgets.community_post_view import CommunityPostScreen
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_approvals_ux import GatedSession, _boot, _gate, _wait_prompt

LONG = "\n".join(f"Step {i}: " + "run the doctor and paste what it says. " * 4 for i in range(20))


@pytest.fixture(autouse=True)
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


def _card_text(text: str) -> str:
    return post_card(community="LOC", channel="#help", text=text, tainted=False)


async def _settle(pilot, rounds: int = 4) -> None:
    for _ in range(rounds):
        await pilot.pause()


@pytest.mark.asyncio
async def test_short_post_on_a_roomy_card_allows_in_one_key() -> None:
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)(COMMUNITY_POST_GATE, _card_text("hi all")))
        await _wait_prompt(pilot, app)
        card = app.query(ApprovalPrompt).first()
        assert not card._post_unread()
        await pilot.press("y")
        assert await asyncio.wait_for(pending, 3) is True


@pytest.mark.asyncio
async def test_cut_post_needs_reading_to_the_end_before_allow() -> None:
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)(COMMUNITY_POST_GATE, _card_text(LONG)))
        await _wait_prompt(pilot, app)
        card = app.query(ApprovalPrompt).first()
        lines = "\n".join(card.render_lines_for_test())
        assert "Audience" in lines and "Shown as your own post" in lines
        assert "Deny" in lines, "the answers stay on the card"
        # Design review D2: the header-only card says the post is hidden and
        # what y does, instead of a bare "…" and an "Allow" it will not give.
        assert "… v reads the post" in lines
        assert "Read the post, then allow" in lines

        await pilot.press("y")
        await _settle(pilot)
        assert not pending.done(), "y on an unread cut post must not approve"
        viewer = app.screen
        assert isinstance(viewer, CommunityPostScreen) and not viewer.read_to_end
        await pilot.press("y")  # a habitual second press does nothing here
        await _settle(pilot, 2)
        assert not pending.done()

        await pilot.press("escape")  # back without reading: still unread
        await _settle(pilot)
        assert not isinstance(app.screen, CommunityPostScreen)
        await pilot.press("y")
        await _settle(pilot)
        assert not pending.done() and isinstance(app.screen, CommunityPostScreen)

        await pilot.press("end")
        await _settle(pilot)
        assert app.screen.read_to_end  # type: ignore[attr-defined]
        await pilot.press("escape")
        await _settle(pilot)
        relabelled = "\n".join(card.render_lines_for_test())
        assert "Read the post, then allow" not in relabelled and "Allow" in relabelled
        await pilot.press("y")
        assert await asyncio.wait_for(pending, 3) is True


@pytest.mark.asyncio
async def test_deny_stays_one_key_on_an_unread_post() -> None:
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)(COMMUNITY_POST_GATE, _card_text(LONG)))
        await _wait_prompt(pilot, app)
        await pilot.press("n")
        assert await asyncio.wait_for(pending, 3) is False


@pytest.mark.asyncio
async def test_roomy_card_names_the_hidden_rows_and_the_key() -> None:
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)(COMMUNITY_POST_GATE, _card_text(LONG)))
        await _wait_prompt(pilot, app)
        card = app.query(ApprovalPrompt).first()
        assert card._post_unread()
        assert any("v reads it all" in row for row in card.render_lines_for_test())
        await pilot.press("v")
        await _settle(pilot)
        assert isinstance(app.screen, CommunityPostScreen)
        await pilot.press("escape")
        await _settle(pilot)
        await pilot.press("n")
        assert await asyncio.wait_for(pending, 3) is False

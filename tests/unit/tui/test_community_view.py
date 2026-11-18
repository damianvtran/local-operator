"""``/community`` and ``/orgs`` through the REAL ``OperatorApp`` (stylesheet applied).

Each test opens the page the way a person does — the command typed into the
session composer — over ``community_fakes.FakeBackend`` and asserts on the
widget state the frames describe (TUI frames T1, T2, T5, T9, Amendment A).
The rendered frames themselves are produced by ``scripts/community_shot.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.community.service import PostOutcome
from local_operator.tui import community_backend
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.community_view import CommunityView
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.community_fakes import HOSTILE_HELP, FakeBackend
from tests.unit.tui.test_app_pilot import FakeSession, _await_session, _factory


async def _open(app: Any, pilot: Any, command: str = "/community") -> CommunityView:
    await _await_session(app, pilot)
    editor = app.query_one(Editor)
    editor.load_text(command)
    editor.focus()
    await pilot.press("enter")
    for _ in range(4):
        await pilot.pause()
    return app.query_one(CommunityView)


async def _type(pilot: Any, text: str) -> None:
    for char in text:
        await pilot.press("space" if char == " " else char)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    monkeypatch.setenv("LOCAL_OPERATOR_COMMUNITY", "1")
    fake = FakeBackend()
    monkeypatch.setattr(community_backend, "LiveCommunityBackend", lambda _cfg: fake)
    return fake


@pytest.mark.asyncio
async def test_dark_community_is_an_ordinary_unknown_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_OPERATOR_COMMUNITY", raising=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _await_session(app, pilot)
        assert app._dispatchable_slash("/community") is False
        assert app._dispatchable_slash("/orgs") is False
        assert not app.query(CommunityView)


@pytest.mark.asyncio
async def test_home_lists_channels_initial_first_and_esc_closes_without_leaving(
    backend: FakeBackend,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        assert [c.name for c in view.channels][:3] == ["welcome-and-rules", "general", "help"]
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(CommunityView)
        assert backend.left == 0, "closing the page is not leaving"


@pytest.mark.asyncio
async def test_composer_checking_cancel_restores_text(backend: FakeBackend) -> None:
    backend.submit_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        view.channel_index = 2  # #help
        view._open_channel()
        await pilot.press("tab", "tab")
        await _type(pilot, "hello there")
        await pilot.press("enter")
        await pilot.pause()
        composer = view.composer()
        assert composer is not None and composer.status == "checking"
        assert view._input.value == "", "↵ clears the composer at once"
        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()
        assert composer.status == "cancelled"
        assert view._input.value == "hello there"
        assert app.query(CommunityView), "the first esc cancels; it never leaves"
        backend.submit_gate.set()


@pytest.mark.asyncio
async def test_refusal_is_sender_only_and_f2_explains(backend: FakeBackend) -> None:
    backend.next_outcome = PostOutcome(
        "refused", category="harassment_or_abuse", reference="MOD-7Q4K-2M9D"
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("tab", "tab")
        await _type(pilot, "draft")
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        assert "Harassment or abuse" in str(view._status.render())
        assert view._input.value == "draft"
        await pilot.press("f2")
        await pilot.pause()
        assert view.overlay == "why"
        assert "MOD-7Q4K-2M9D" in view._overlay_text.render().plain  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_leading_slash_is_never_posted(backend: FakeBackend) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _open(app, pilot)
        await pilot.press("tab", "tab")
        await _type(pilot, "/help")
        await pilot.press("enter")
        await pilot.pause()
        assert backend.submitted == []


@pytest.mark.asyncio
async def test_join_needs_the_explicit_agreement(backend: FakeBackend) -> None:
    backend.joined = False
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("j")
        await pilot.pause()
        assert view.overlay == "join"
        await pilot.press("tab", "enter")  # Join while unticked
        await pilot.pause()
        assert backend.joins == [] and view.join_focus == 0, "↵ on disabled Join goes back"
        await pilot.press("space", "tab", "enter")
        for _ in range(4):
            await pilot.pause()
        assert backend.joins == ["2026-11"]
        assert view.overlay is None and view.home.joined


@pytest.mark.asyncio
async def test_orgs_defaults_to_home_org_and_picking_a_private_org_keeps_bridge_flow(
    backend: FakeBackend,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot, "/orgs")
        for _ in range(3):
            await pilot.pause()
        every = [*view.orgs_private, *view.orgs_communities]
        assert every[view.orgs_index].name == "Tran Robotics"
        assert [e.kind for e in view.orgs_communities] == ["community"]
        await pilot.press("enter")
        await pilot.pause()
        assert backend.saved_selection == "t_tran"
        assert not app.query(CommunityView)


@pytest.mark.asyncio
async def test_leave_defaults_to_stay(backend: FakeBackend) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("s", "tab")
        await pilot.pause()
        assert view.overlay == "settings" and view.settings_tab == 2
        await pilot.press("enter")
        await pilot.pause()
        assert backend.left == 0 and view.overlay is None


def _painted(widget: Any) -> str:
    """Every segment the widget would hand the terminal, joined."""
    from textual.geometry import Region

    region = Region(0, 0, widget.size.width, widget.size.height)
    return "".join(segment.text for strip in widget.render_lines(region) for segment in strip)


@pytest.mark.asyncio
async def test_hostile_post_and_name_reach_the_terminal_without_control_sequences(
    backend: FakeBackend,
) -> None:
    # R1-3: OSC 52 (clipboard), ESC[2J (clear), OSC 0 (title) and SGR blink in a
    # stranger's post and display name, through the real app and page.
    backend.histories["c_help"] = list(HOSTILE_HELP)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        view.channel_index = next(i for i, c in enumerate(view.channels) if c.name == "help")
        view._open_channel()
        for _ in range(4):
            await pilot.pause()
        painted = _painted(view._pane_text)
        assert "Mallory" in painted and "hi  there" in painted
        for payload in ("\x1b", "\x9b", "PWNED", "cm0gLXJmIH4K"):
            assert payload not in painted, payload
        # The model the pane reads is the sanitised copy; the backend's is exact.
        assert "\x1b" not in (view.messages["c_help"] or [])[0].text
        assert "\x1b" in backend.histories["c_help"][0].text


@pytest.mark.asyncio
async def test_orgs_checks_the_viewed_org_apart_from_the_cursor(backend: FakeBackend) -> None:
    # R1-6 (D21): from the community page the community is the one viewed,
    # while the cursor still defaults to the home org (Amendment A).
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("g")
        for _ in range(4):
            await pilot.pause()
        rows = view._paint_orgs().plain.splitlines()
        assert any(row.startswith("  ✓ ") and "Local Operator Community" in row for row in rows)
        assert any(row.startswith("▸   ") and "Tran Robotics" in row for row in rows)


@pytest.mark.asyncio
async def test_narrow_header_drops_whole_key_hints(backend: FakeBackend) -> None:
    # R1-7: at 60 columns the name and the hints never run together.
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 20)) as pilot:
        view = await _open(app, pilot)
        header = _painted(view._header)
        assert "connected g" not in header and "? ke" not in header.replace("? keys", "")

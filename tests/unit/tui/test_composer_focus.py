"""The composer is where the keyboard lives: every dock cell hands focus back.

Two things are guarded here, and they are the two halves of one report — "when
you click on a tool call and click back to the text input it doesn't appear
focused in style".

The first is the dead frame. Textual routes a click to focus by walking UP from
the widget under the pointer (``Screen.get_focusable_widget_at``,
``screen.py:709-712``); ``#input-dock`` and ``#input-shell`` were plain
containers, so the walk ran off the top and returned ``None`` — and Textual does
NOT blur on that path, it only does that for ``NoWidget``. The click vanished
without a trace. Measured at 120x40 settled, dock and shell both
``Region(x=1, y=34, w=118, h=5)`` with the editor body at
``Region(x=4, y=35, w=114, h=1)``: clicking ``x = dock.region.x + 2`` on EVERY
dock row left focus on the ToolCard. Rows that focused the editor: none. The
chevron — the app's own "you are focused" affordance — was itself a dead cell,
so the user clicked the exact glyph that means "focused" and got nothing.

The second is :meth:`OperatorApp._focus_is_claimed`, the predicate the later
focus work is built on. It answers "does some surface have a legitimate claim on
the keyboard", and every state below is opened through its REAL path rather than
by setting the attribute — a predicate tested against hand-set attributes is a
test of the attributes.

THE COORDINATE RULE, and it is not optional: every click offset is derived from
``shell.region``/``editor.region`` AT TEST TIME. Boot and settled layouts differ
(boot dock ``Region(x=1, y=29, w=118, h=10)`` against settled
``Region(x=1, y=34, w=118, h=5)``), so a hardcoded coordinate is wrong in one of
them. And ``Pilot.click`` bounds-checks against ``screen.size.region``, not
``screen.region`` (``pilot.py:440-443``) — at 120x40 those are
``Region(0,0,118,38)`` and ``Region(0,0,120,40)``. The shell's own
``region.bottom`` is 39, so its bottom two rows raise ``OutOfBounds`` at every
size checked (120x40, 100x30, 80x24). There is deliberately no "bottom pad"
click site below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.widgets.text_area import Selection

from local_operator.tui.app import COMPOSER_FOCUSED_CLASS, OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.welcome import WelcomeView

from .conftest import caret_cells, composer_cells
from .test_app_pilot import FakeSession, _factory


class _Overlay(ModalScreen[None]):
    """Stands in for whatever modal route pushed a Screen over the composer."""

    def compose(self) -> ComposeResult:
        yield Static("on top")


def _app(**kwargs: Any) -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()), **kwargs)


async def _boot(pilot: Any, app: OperatorApp) -> None:
    """Pause until the session is attached — the app is not usable before that."""
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _settle_boot(pilot: Any, ticks: int = 24) -> None:
    """Pause until the BOOT composition stops moving.

    Copied in shape from ``test_boot_layout._settle``: the splash's poll timer
    lands the model label a fraction of a second in and the block can change
    height with it, so a fixed pause count races the geometry this file derives
    every click from.
    """
    welcome = pilot.app.query_one(WelcomeView)
    dock = pilot.app.query_one("#input-dock")
    previous: tuple[bool, Any] | None = None
    for _ in range(ticks):
        await pilot.pause()
        current = (dock.has_class("-boot-gap"), dock.styles.padding.bottom)
        if welcome._timer is None and current == previous:
            break
        previous = current
    await pilot.pause()


async def _settled_with_a_card(pilot: Any, app: OperatorApp) -> ToolCard:
    """The SETTLED layout with one tool card in the ledger, and the card focused.

    Appending a block dismisses the splash, which is what puts the dock in the
    full-width settled geometry the report describes. The card is the thing that
    took focus off the composer in the user's session.
    """
    await _boot(pilot, app)
    app.query_one(Editor).focus()
    await pilot.pause()
    app._append_block(ToolCard("t1", "bash", {"command": "ls"}))
    await pilot.pause()
    await pilot.pause()
    card = next(iter(app.query(ToolCard).results()))
    card.focus()
    await pilot.pause()
    assert app.focused is card, "premise: the card holds focus before the click"
    return card


def _clamped(app: OperatorApp, offset: tuple[int, int]) -> tuple[int, int]:
    """Keep a derived offset inside what ``Pilot.click`` will accept.

    ``screen.size.region`` is two cells smaller per axis than ``screen.region``,
    and the shell's last rows fall in that gap.
    """
    x, y = offset
    bounds = app.screen.size.region
    return (min(x, bounds.width - 1), min(y, bounds.height - 1))


@pytest.mark.asyncio
async def test_clicking_the_shell_padding_returns_focus_to_the_composer() -> None:
    """The reported defect, in one click: padding is part of the input.

    Fails on ``origin/main`` — the click is swallowed and focus stays on the
    card. This is the fail-first guard for the whole slice.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        card = await _settled_with_a_card(pilot, app)
        dock = app.query_one("#input-dock")
        shell = app.query_one("#input-shell")
        editor = app.query_one(Editor)

        await pilot.click(
            offset=_clamped(app, (shell.region.x + shell.region.width // 2, shell.region.y))
        )
        await pilot.pause()

        assert app.focused is editor, f"the click landed on nothing: {app.focused!r}"
        assert app.focused is not card
        assert dock.has_class(COMPOSER_FOCUSED_CLASS), "the chevron is still dark"


#: The four dock cells a user can actually reach with a mouse, named. The bottom
#: pad is absent on purpose: it is out of ``Pilot.click``'s bounds at every size
#: (see the module docstring), not because it is expected to be dead.
DEAD_CELLS = [
    ("top-pad", lambda r: (r.x + r.width // 2, r.y)),
    ("left-of-chevron", lambda r: (r.x, r.y + 1)),
    ("chevron", lambda r: (r.x + 1, r.y + 1)),
    ("band-row", lambda r: (r.x + 5, r.y + 2)),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,site", DEAD_CELLS, ids=[n for n, _ in DEAD_CELLS])
async def test_every_dead_cell_of_the_dock_returns_focus(name: str, site: Any) -> None:
    """All four reachable dock cells, not just the padding.

    The chevron case is the one the user reported: the affordance that says
    "focused" has to be clickable, and it was a dead cell.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _settled_with_a_card(pilot, app)
        dock = app.query_one("#input-dock")
        editor = app.query_one(Editor)

        # Re-derived here rather than hoisted: focusing the card can move the
        # layout, and a stale region is a hardcoded coordinate by another name.
        await pilot.click(offset=_clamped(app, site(app.query_one("#input-shell").region)))
        await pilot.pause()

        assert app.focused is editor, f"{name} is still dead: {app.focused!r}"
        assert dock.has_class(COMPOSER_FOCUSED_CLASS)


@pytest.mark.asyncio
async def test_a_dock_click_does_not_move_the_caret() -> None:
    """Focus comes back; the caret stays where the user left it.

    A click on padding means "put me back in the input", not "put the caret at
    this coordinate" — there is no document position a padding cell maps to, and
    guessing one would cost the user the place they were editing.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("hello world")
        editor.selection = Selection((0, 5), (0, 5))
        await pilot.pause()
        before = caret_cells(composer_cells(app))

        await _settled_with_a_card(pilot, app)
        shell = app.query_one("#input-shell")
        await pilot.click(
            offset=_clamped(app, (shell.region.x + shell.region.width // 2, shell.region.y))
        )
        await pilot.pause()

        assert app.focused is editor
        assert editor.text == "hello world", "the draft did not survive the click"
        assert editor.selection == Selection((0, 5), (0, 5))
        # And the frame agrees: the caret is drawn on the same cell it was.
        assert caret_cells(composer_cells(app)) == before


@pytest.mark.asyncio
async def test_a_dock_click_is_refused_while_the_composer_is_read_only() -> None:
    """The subagent page's whole argument is that the dock is not where you are.

    ``_set_composer_read_only`` drops ``can_focus`` so nothing lands a caret in
    a field that refuses every key; the handler guards on it, so the click is a
    no-op rather than a caret on an inert field.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        card = await _settled_with_a_card(pilot, app)
        editor = app.query_one(Editor)
        app._set_composer_read_only(True)
        await pilot.pause()
        assert not editor.can_focus, "premise: read-only drops can_focus"

        shell = app.query_one("#input-shell")
        await pilot.click(
            offset=_clamped(app, (shell.region.x + shell.region.width // 2, shell.region.y))
        )
        await pilot.pause()

        assert app.focused is card, f"the click stole focus onto an inert field: {app.focused!r}"
        assert not editor.can_focus


@pytest.mark.asyncio
async def test_clicking_the_editor_body_still_places_the_caret() -> None:
    """The forwarding handler must not steal the editor's own mouse work.

    ``Click`` bubbles, so this handler runs AFTER ``Editor._on_mouse_down``/
    ``_on_click`` (``editor.py:4391/4438``) for a body click. It finds
    ``has_focus`` already true and does nothing — which is why it must not call
    ``event.stop()`` either.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _settled_with_a_card(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("hello world")
        await pilot.pause()

        # `content_region`, not `region`: the TextArea reserves a gutter cell, so
        # document column N sits at `content_region.x + N` and deriving from
        # `region.x` lands one column left of where the assertion says.
        body = editor.content_region
        await pilot.click(offset=_clamped(app, (body.x + 3, body.y)))
        await pilot.pause()

        assert app.focused is editor
        assert editor.selection == Selection((0, 3), (0, 3)), "the click did not place the caret"


# -- `_focus_is_claimed` ----------------------------------------------------


async def _claim_nothing(pilot: Any, app: OperatorApp) -> None:
    """The negative: a resting app with the composer focused claims nothing."""
    app.query_one(Editor).focus()
    await pilot.pause()


async def _claim_approval(pilot: Any, app: OperatorApp) -> None:
    app.run_worker(app.request_tool_approval("bash", "rm -rf /"), thread=False)
    for _ in range(6):
        await pilot.pause()
    assert app._approval is not None, "premise: the approval card is up"


async def _claim_ask(pilot: Any, app: OperatorApp) -> None:
    from local_operator.harness.types import AskOption, AskQuestion

    question = AskQuestion(
        id="stale",
        question="What should happen to the stale rows?",
        options=[
            AskOption(label="Drop them", description="nothing reads the column"),
            AskOption(label="Backfill", description="slower, keeps history"),
        ],
    )
    # Run as a worker rather than a bare task: a local `create_task` handle goes
    # out of scope when this opener returns and the loop may collect the task
    # out from under the picker.
    app.run_worker(app.request_user_choice([question]), thread=False)
    for _ in range(6):
        await pilot.pause()
    assert app._ask_screen is not None, "premise: the ask picker is up"


async def _claim_aside(pilot: Any, app: OperatorApp) -> None:
    """Bare ``/btw`` through the real command dispatch — it opens an empty card."""
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text("/btw")
    await pilot.press("enter")
    for _ in range(4):
        await pilot.pause()
    assert app._aside_is_open(), "premise: the aside card is open"


async def _claim_subagent_view(pilot: Any, app: OperatorApp) -> None:
    from .test_band_panels import _fake_jobs, _Job

    # The page resolves the job out of the session's manager, so the ledger has
    # to hold one; `test_app_pilot`'s stub manager has no `get`. Opened through
    # `_open_subagent_view`, which is the callback the band row itself uses
    # (`test_band_panels.py:916`).
    jobs = _fake_jobs(_Job("sub-1", "audit the ingest path"))
    app._session.jobs = jobs  # type: ignore[union-attr]
    app._refresh_band()
    await pilot.pause()
    app._open_subagent_view("sub-1")
    await pilot.pause()
    assert app._subagent_view is not None


async def _claim_org_chart(pilot: Any, app: OperatorApp) -> None:
    app._run_slash_command("/team chart org")
    for _ in range(4):
        await pilot.pause()
    assert app._org_chart_view is not None


async def _claim_settings(pilot: Any, app: OperatorApp) -> None:
    app._open_settings_view()
    for _ in range(4):
        await pilot.pause()
    assert app._settings_view is not None


async def _claim_sidebar(pilot: Any, app: OperatorApp) -> None:
    await pilot.press("f9")
    await pilot.pause()
    assert app._session_sidebar.has_focus, "premise: f9 focused the sidebar"


async def _claim_pushed_screen(pilot: Any, app: OperatorApp) -> None:
    app.push_screen(_Overlay())
    for _ in range(3):
        await pilot.pause()
    assert len(app.screen_stack) > 1


async def _claim_read_only(pilot: Any, app: OperatorApp) -> None:
    app._set_composer_read_only(True)
    await pilot.pause()
    assert not app.query_one(Editor).can_focus


#: Every claimant in the design's Q3 table, plus the negative. Each is opened
#: through the route a user takes, never by assigning the attribute the
#: predicate reads — that would be a test of the assignment.
CLAIMANTS = [
    ("resting", _claim_nothing, False),
    ("approval", _claim_approval, True),
    ("ask-picker", _claim_ask, True),
    ("aside", _claim_aside, True),
    ("subagent-view", _claim_subagent_view, True),
    ("org-chart", _claim_org_chart, True),
    ("settings", _claim_settings, True),
    ("sidebar-focused", _claim_sidebar, True),
    ("pushed-screen", _claim_pushed_screen, True),
    ("read-only", _claim_read_only, True),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,opener,expected", CLAIMANTS, ids=[n for n, _, _ in CLAIMANTS]
)
async def test_focus_is_claimed_covers_every_overlay(
    name: str, opener: Any, expected: bool, tmp_path: Path
) -> None:
    session = FakeSession()
    # `/team chart` resolves against the registry the session carries; the other
    # cases ignore it.
    session.team_registry = _chart_registry()  # type: ignore[attr-defined]
    app = OperatorApp(lambda: _factory(session), provider_controller=_controller(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert app._focus_is_claimed() is False, "premise: nothing claims focus on a fresh app"

        await opener(pilot, app)

        assert app._focus_is_claimed() is expected, f"{name} answered wrong"


def _controller(tmp_path: Path) -> Any:
    """The REAL ProviderController over a throwaway store, as the login tests use."""
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    return ProviderController(AuthStore(tmp_path / "auth.db"))


def _chart_registry() -> Any:
    from local_operator.teams import TeamEditFields, TeamMember

    from .test_team_chart import _registry

    return _registry(
        TeamEditFields(name="org", manager="manager", members=[TeamMember(role="coder")])
    )


@pytest.mark.asyncio
async def test_focus_is_claimed_by_a_live_key_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The login paste prompt, driven through the real ``/login`` flow.

    Separated from the table above because it needs the real provider registry
    and a controller over a throwaway store — a stub standing in for either end
    would be a test of the stub, and this prompt is the one surface where
    stealing focus put an API key into the transcript in plain text.
    """
    from .test_login_key_prompt import _LoginSession, _run_login

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)
    session = _LoginSession()
    app = OperatorApp(lambda: _factory(session), provider_controller=_controller(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert app._focus_is_claimed() is False

        await _run_login(pilot, app, "alibaba")
        assert app._key_prompt is not None, "premise: the key prompt is up"

        assert app._focus_is_claimed() is True


@pytest.mark.asyncio
async def test_the_boot_layout_gutter_returns_focus() -> None:
    """The first frame a new user sees has a dead gutter, and it is the widest one.

    On boot the shell is clamped to a centred card (``local_operator.tcss:1078``)
    and the dock spans full width behind it, so every cell of the 18-column
    gutter either side hits ``#input-dock`` directly. Measured at 120x40 on boot:
    dock ``Region(x=1, y=29, w=118, h=10)``, shell ``Region(x=19, y=29, w=82,
    h=5)``.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _settle_boot(pilot)
        dock = app.query_one("#input-dock")
        shell = app.query_one("#input-shell")
        editor = app.query_one(Editor)
        assert shell.region.width < dock.region.width, "premise: this IS the boot layout"

        # Inside the dock and outside the card: a cell that exists only here.
        gutter = (dock.region.x + 2, shell.region.y + 2)
        assert not shell.region.contains(*gutter)
        app.query_one("#transcript").focus()
        await pilot.pause()

        await pilot.click(offset=_clamped(app, gutter))
        await pilot.pause()

        assert app.focused is editor, f"the boot gutter is dead: {app.focused!r}"
        assert dock.has_class(COMPOSER_FOCUSED_CLASS)

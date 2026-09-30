"""Click-to-focus and typing-home for the Sessions list (issue #1357 decision).

The decision — `### lopdev — design decision: click-to-focus`, recorded on the
issue — moved the anti-swallow property off ``FOCUS_ON_CLICK = False`` and onto
the widget: a pointer press gives the list the keyboard (T1-T3), and the first
printable character or paste delivered to it is handed to the composer, which
takes the keyboard back (T12/G1). Two measured defects this retires:

* **F3** — on main, text typed into an ``f9``-focused list was swallowed
  (``editor.text`` stayed empty). Retired for every route into the list's
  keyboard mode.
* **F4** — on main, a press on the attached session's own row stole the
  keyboard from a live approval via the unguarded ``self._editor().focus()`` in
  ``on_session_sidebar_selected``. The press now moves the keyboard nowhere
  while a hard claimant holds it (G7).

The guard surfaces are the same predicate the composer's own routes use
(``OperatorApp._focus_is_claimed`` via ``composer_focus``); the tests here
exercise the shapes a press can take, including the ones that must NOT move the
keyboard. Frames for the visual side live on the PR, not in the tree.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual import events

from local_operator.resume import SessionRow
from local_operator.session.catalog import CatalogEntry
from local_operator.tui.app import COMPOSER_FOCUSED_CLASS, OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.session_sidebar import PIN_CELL_WIDTH

from .test_app_pilot import FakeSession, _factory
from .test_session_sidebar import _chip_span, _click_footer_cell


def _app(**kwargs: Any) -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()), **kwargs)


async def _boot(pilot: Any, app: OperatorApp) -> None:
    """Pause until the session is attached — the app is not usable before that."""
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


def _catalog(*session_ids: str) -> list[Any]:
    """List entries newest-first, one per id, as the app's refresh builds them."""
    return [
        CatalogEntry(SessionRow(sid, 1_000.0 - index, f"Conversation {sid}", live_state="idle"))
        for index, sid in enumerate(session_ids)
    ]


async def _open_list(pilot: Any, app: OperatorApp, *session_ids: str) -> Any:
    """The list OPEN and populated, with the composer still holding the keyboard.

    The app's own catalog refresh is paused: it rebuilds ``entries`` from the
    session store on a timer, and the pilot's ``FakeSession`` does not put the
    rows these tests click on into it. Pausing the timer keeps the seeded list
    in place for the length of the gesture — the mechanism the design round's
    probe used too.
    """
    sidebar = app._session_sidebar
    app._set_sidebar_open(True)
    for _ in range(8):
        await pilot.pause()
    refresh = app._sidebar_timer
    if refresh is not None:
        refresh.pause()
    spinner = sidebar._timer
    if spinner is not None:
        spinner.pause()
    sidebar.current_id = session_ids[0]
    sidebar.cursor_id = session_ids[0]
    sidebar.set_entries(_catalog(*session_ids))
    for _ in range(4):
        await pilot.pause()
    assert sidebar.display, "premise: the list is on the frame"
    return sidebar


def _rows(app: OperatorApp, sidebar: Any) -> dict[str, int]:
    """``session id -> screen y`` for every painted row, from the widget itself."""
    origin = sidebar.region.y
    found: dict[str, int] = {}
    for y in range(sidebar.size.height):
        entry = sidebar._entry_at(y)
        if entry is not None:
            found.setdefault(entry.id, origin + y)
    return found


def _row_body_x(sidebar: Any) -> int:
    """A row's body — one cell past the pin cell, which pins instead (#1357 2a)."""
    return sidebar.region.x + PIN_CELL_WIDTH + 1


def _dock_pad(app: OperatorApp) -> tuple[int, int]:
    """A dead dock cell — column 2 of the shell's top row (the D2 route)."""
    shell = app.query_one("#input-shell")
    bounds = app.screen.size.region
    x = min(shell.region.x + 2, bounds.width - 1)
    y = min(shell.region.y, bounds.height - 1)
    return (x, y)


async def _a_multi_select(pilot: Any, app: OperatorApp) -> Any:
    """A live multi-select ask picker: the question routed keys cannot reach."""
    from local_operator.harness.types import AskOption, AskQuestion

    question = AskQuestion(
        id="rows",
        question="Which rows should be dropped?",
        options=[
            AskOption(label="Stale", description="nothing reads them"),
            AskOption(label="Orphaned", description="no parent row"),
        ],
        multi=True,
    )
    app.run_worker(app.request_user_choice([question]), thread=False)
    for _ in range(8):
        await pilot.pause()
    picker = app._ask_screen
    assert picker is not None, "premise: the picker is up"
    return picker


# -- the press moves the keyboard ------------------------------------------------


@pytest.mark.asyncio
async def test_a_row_press_gives_the_list_the_keyboard_and_moves_the_cursor() -> None:
    """T1: the press focuses the list, moves its cursor, and keeps both.

    The row pressed is the ATTACHED session's own — the branch F4 found, whose
    behaviour changes by design: it used to return the keyboard to the composer
    and now keeps it on the list, because every row press behaves alike.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert not sidebar.has_focus, "premise: opening the list did not move the keyboard"

        rows = _rows(app, sidebar)
        assert "sess" in rows, "premise: the attached row is painted"
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()

        assert sidebar.has_focus, "a row press no longer gives the list the keyboard"
        assert sidebar.cursor_id == "sess", "the press no longer moves the cursor"
        assert not editor.has_focus, "the no-op branch handed the composer the keys back"
        assert not app.query_one("#input-dock").has_class(COMPOSER_FOCUSED_CLASS)


@pytest.mark.asyncio
async def test_arrow_after_a_row_press_drives_the_list_not_the_composer_history() -> None:
    """F2, inverted: the arrow after a row press works on the list, not history.

    On main the same gesture replaced the draft with a recalled prompt while the
    list's caret never moved (``sidebar_cursor`` byte-identical) — the reported
    symptom. The assertions are the two halves of the inversion: the cursor
    moves and the composer's text is byte-identical.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "aaaa", "sess", "zzzz")
        editor = app.query_one(Editor)
        editor._history = ["ship the scroll fix", "add the subagent chip"]
        editor._history_index = None
        editor.text = "draft: tighten the poll loop"
        editor.focus()
        await pilot.pause()

        # The ranking paints these in id order, so the attached row sits in the
        # middle and an Up has somewhere to go. Asserted rather than assumed:
        # a ranking change should fail HERE with its own message.
        assert [entry.id for entry in sidebar.entries] == ["aaaa", "sess", "zzzz"]
        rows = _rows(app, sidebar)
        assert "sess" in rows, "premise: the attached row is painted"
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()
        assert sidebar.has_focus, "premise: the press focused the list"

        await pilot.press("up")
        for _ in range(3):
            await pilot.pause()

        assert sidebar.cursor_id == "aaaa", "the arrow after a press did not move the list"
        assert (
            editor.text == "draft: tighten the poll loop"
        ), "the arrow after a press drove the composer's history again"


@pytest.mark.asyncio
async def test_enter_after_a_row_press_opens_the_cursor_row_not_the_draft() -> None:
    """T6: on the list, Enter opens the cursor row; the draft is not sent."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-a")
        editor = app.query_one(Editor)
        editor.text = "draft"
        editor.focus()
        await pilot.pause()

        opened: list[str] = []
        real_select = app._sidebar_navigation.select

        def _record(session_id: str) -> Any:
            opened.append(session_id)
            return None

        app._sidebar_navigation.select = _record  # type: ignore[method-assign]
        try:
            rows = _rows(app, sidebar)
            await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
            for _ in range(4):
                await pilot.pause()
            assert opened == [], "premise: the attached row is a no-op navigation"

            await pilot.press("down")
            for _ in range(2):
                await pilot.pause()
            assert sidebar.cursor_id == "sess-a", "premise: the cursor is on the other row"

            await pilot.press("enter")
            for _ in range(2):
                await pilot.pause()
        finally:
            app._sidebar_navigation.select = real_select  # type: ignore[method-assign]

        assert opened == ["sess-a"], "Enter did not open the cursor row"
        assert editor.text == "draft", "Enter submitted the draft instead of opening the row"


@pytest.mark.asyncio
async def test_a_press_on_panel_chrome_takes_the_keyboard_and_acts_on_no_row() -> None:
    """T3: dead space and chrome give the list the keyboard; nothing else moves."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        opened: list[str] = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app._sidebar_navigation, "select", opened.append)
            await pilot.click(offset=(sidebar.region.x + 5, sidebar.region.bottom - 3))
            for _ in range(4):
                await pilot.pause()

        assert sidebar.has_focus, "a press on the panel's dead space did not focus the list"
        assert sidebar.cursor_id == "sess", "the dead space moved the cursor"
        assert opened == [], "the dead space acted on a row"
        assert editor.text == "", "the dead space typed into the composer"


@pytest.mark.asyncio
async def test_a_press_on_the_pin_cell_pins_and_never_opens_the_row(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2: the pin cell is a control, and it now takes the keyboard like any press.

    The pin-toggle and never-open halves are slice 2a's rule (kept); the new
    half is the keyboard. The store is isolated: the toggle is real, so it must
    not reach the operator's config directory.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # `read_pins` prunes against the session store, so the pinned id needs a
    # real `sessions/<id>` directory or the round trip reads empty — the same
    # fixture rule `test_app_pilot._seed_session_dirs` records.
    (tmp_path / "sessions" / "sess").mkdir(parents=True, exist_ok=True)
    from local_operator.tui.sidebar_pins import read_pins

    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        rows = _rows(app, sidebar)
        local_y = rows["sess"] - sidebar.region.y
        opened: list[str] = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app._sidebar_navigation, "select", opened.append)
            await pilot.click("#session-sidebar", offset=(1, local_y))
            for _ in range(40):
                await pilot.pause()
                if read_pins(tmp_path):
                    break

        assert read_pins(tmp_path) == ["sess"], "a pin-cell press no longer pins"
        assert opened == [], "a pin-cell press started a session switch"
        assert sidebar.cursor_id == "sess", "the pin cell moved the cursor"
        assert sidebar.has_focus, "the pin cell did not give the list the keyboard"


# -- typing-home: the list hands text to the composer ------------------------------


@pytest.mark.asyncio
async def test_typing_after_a_row_press_lands_in_the_composer() -> None:
    """G1/T12: the first printable character is typed, and the composer takes it."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()
        assert sidebar.has_focus, "premise: the list owns the keyboard"

        await pilot.press("d")
        for _ in range(2):
            await pilot.pause()

        assert editor.text == "d", "the printable key was swallowed by the list"
        assert editor.has_focus, "the composer did not take the keyboard back"
        assert not sidebar.has_focus


@pytest.mark.asyncio
async def test_f9_then_typing_lands_in_the_composer() -> None:
    """F3, retired: the deliberate route into the list's mode is also typing-home."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        await pilot.press("f9")
        for _ in range(2):
            await pilot.pause()
        assert sidebar.has_focus, "premise: f9 focused the list"

        await pilot.press("A")
        for _ in range(2):
            await pilot.pause()

        assert editor.text == "A", "text typed into an f9-focused list was swallowed (F3)"
        assert editor.has_focus


@pytest.mark.asyncio
async def test_paste_after_a_row_press_lands_in_the_composer() -> None:
    """G1 for pastes (T12): a bracketed paste is delivered, never dropped."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()
        assert sidebar.has_focus, "premise: the list owns the keyboard"

        app.post_message(events.Paste("pasted text"))
        for _ in range(3):
            await pilot.pause()

        assert editor.text == "pasted text", "the paste was dropped on the list"
        assert editor.has_focus, "the composer did not take the keyboard back"


@pytest.mark.asyncio
async def test_space_types_and_never_pins() -> None:
    """T14: no plain-key pin mnemonic — Space is a character first.

    A Space-pin would fire on the space in "hello world" the moment the list is
    click-focused, which is the modal trap the issue exists to remove; the
    clickable star cell, F10 and the focused footer's `f10 pin` cue are the
    remaining routes.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()

        await pilot.press("space")
        for _ in range(2):
            await pilot.pause()

        assert editor.text == " ", "Space did not type"
        assert sidebar._pins == (), "Space pinned the row"
        assert sidebar.cursor_id == "sess", "Space moved the cursor"


@pytest.mark.asyncio
async def test_ctrl_a_still_toggles_the_layer_and_never_reaches_the_composer() -> None:
    """G9: ctrl+a/ctrl+o stay list-scoped chords; typing-home must not eat them."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.text = "draft"
        editor.focus()
        await pilot.pause()

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()

        assert sidebar.show_subagents is False, "premise: the layer starts off"
        await pilot.press("ctrl+a")
        for _ in range(2):
            await pilot.pause()

        assert sidebar.show_subagents is True, "ctrl+a no longer toggles the layer"
        assert editor.text == "draft", "ctrl+a reached the composer's line-start"
        assert sidebar.has_focus, "the chord handed the composer the keys"


# -- the composer's own routes stay open -----------------------------------------


@pytest.mark.asyncio
async def test_the_composer_chrome_still_takes_the_keyboard_back() -> None:
    """T4/D2: from the list's keyboard, a press on the composer's chrome comes home."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()
        assert sidebar.has_focus, "premise: the list owns the keyboard"

        await pilot.click(offset=_dock_pad(app))
        for _ in range(3):
            await pilot.pause()

        assert editor.has_focus, "the composer's own chrome did not take the keyboard back"
        await pilot.press("x")
        await pilot.pause()
        assert editor.text == "x", "the key after the dock click never reached the composer"


# -- guards: a press moves the keyboard nowhere while a claim is live ------------


@pytest.mark.asyncio
async def test_a_press_with_a_live_approval_moves_no_keyboard() -> None:
    """G7/F4: the press acts on the row; a live approval keeps the keys it needs.

    F4 measured the opposite on main: the press on the attached row ran the
    unguarded ``self._editor().focus()`` and the approval lost the keyboard
    mid-question. Both halves of T13 are asserted: the keyboard moves nowhere,
    AND the press still acts on the row it lands on (the cursor moves and the
    switch it would start still starts) — the refusal is the keyboard's alone.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        app.run_worker(
            app.request_tool_approval("bash", "rm -rf /Users/me/project/data"), thread=False
        )
        for _ in range(8):
            await pilot.pause()
        prompt = app._approval
        assert prompt is not None, "premise: the approval card is up"
        assert app.focused is prompt, "premise: the approval holds the keyboard"
        assert app._focus_is_claimed() is True, "premise: the predicate says claimed"

        rows = _rows(app, sidebar)
        # The attached row first: the F4 defect's own gesture.
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()

        assert app.focused is prompt, "a list press stole the keyboard from a live approval (F4)"
        assert not sidebar.has_focus, "the press took the keyboard for the list"
        assert not editor.has_focus, "the no-op branch handed the approval's keys to the composer"

        # T13's other half, on the row that WOULD switch: the press still acts
        # on it (cursor + the switch it starts), and only the keyboard is held.
        started: list[str] = []
        real_select = app._sidebar_navigation.select

        def _record(session_id: str) -> Any:
            started.append(session_id)
            return None

        app._sidebar_navigation.select = _record  # type: ignore[method-assign]
        try:
            await pilot.click(offset=(_row_body_x(sidebar), rows["sess-b"]))
            for _ in range(4):
                await pilot.pause()
        finally:
            app._sidebar_navigation.select = real_select  # type: ignore[method-assign]

        assert sidebar.cursor_id == "sess-b", "the press no longer acts on the row"
        assert started == ["sess-b"], "the press no longer starts the switch it lands on"
        assert app.focused is prompt, "a press moved the keyboard off the live claim"
        assert not sidebar.has_focus and not editor.has_focus


@pytest.mark.asyncio
async def test_a_press_with_a_live_multi_select_moves_no_keyboard() -> None:
    """G7 for the ask picker: Space/Enter answers must not be taken by a press."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        picker = await _a_multi_select(pilot, app)
        assert not editor.has_focus, "premise: the picker holds the keyboard"

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()

        assert app.focused is picker, "a list press moved the keyboard off the picker"
        assert app._focus_is_claimed() is True


# -- the footer chip (reconciliation with #1841) ---------------------------------


@pytest.mark.asyncio
async def test_the_footer_chip_is_exempt_from_the_press_focus() -> None:
    """Composition with #1841: the chip runs its OWN action; one cell off does not.

    The chip slice's test pins that a chip press toggles the ⌥ layer and moves
    no keyboard; this pins the same press against THIS slice's focus walk and
    pairs it with the other half of the reconciliation, so the exemption
    cannot widen: a press on a footer cell that is not the chip is a normal
    press on the panel and focuses the list. The chip's cells come from its
    own painted footer (`_chip_span`, the chip slice's own helper), never from
    the widget's hit-test, so the pair still discriminates if the hit-test
    drifts.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        sidebar.set_subagent_total(4)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        app._refresh_sidebar = lambda: None  # type: ignore[method-assign]

        first, _end = _chip_span(sidebar)
        await _click_footer_cell(pilot, app, sidebar, first)
        for _ in range(3):
            await pilot.pause()
        assert sidebar.show_subagents is True, "the chip press did not flip the layer"
        assert app.focused is editor, "the chip press took the keyboard for the list"
        assert not sidebar.has_focus

        first, _end = _chip_span(sidebar)
        await _click_footer_cell(pilot, app, sidebar, first)
        for _ in range(3):
            await pilot.pause()
        assert sidebar.show_subagents is False, "a second chip press must flip it back"
        assert app.focused is editor, "the second chip press took the keyboard"

        first, _end = _chip_span(sidebar)
        await _click_footer_cell(pilot, app, sidebar, max(0, first - 1))
        for _ in range(3):
            await pilot.pause()
        assert sidebar.has_focus, "a press off the chip no longer focuses the panel"


# -- the drawer placement --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_drawer_press_closes_the_panel_and_returns_to_the_composer() -> None:
    """G8/T1n: the drawer closes and the composer holds the keyboard; never None."""
    app = _app()
    async with app.run_test(size=(70, 30)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert sidebar.display, "premise: the drawer is open"

        rows = _rows(app, sidebar)
        assert "sess" in rows, "premise: the attached row is painted"
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()

        assert not sidebar.display, "the drawer did not close on a valid selection"
        assert app.focused is editor, (
            "the closed drawer left the keyboard somewhere else: "
            f"{type(app.focused).__name__ if app.focused is not None else None}"
        )
        assert editor.has_focus, "the composer did not take the keyboard back"


# -- the focused footer rung ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (150, 40)])
async def test_a_pressed_panel_shows_the_focused_footer_rung(size: tuple[int, int]) -> None:
    """§5 intent 17: the focus-aware ladder flips with the keyboard.

    Rung-for-rung at both decided sizes: resting teaches the way IN (`f9
    focus`), the pressed panel teaches the way OUT and the pin (`esc return ·
    f10 pin`). The design round verified both render without clipping at a
    29-cell and a 43-cell content width; this pins the strings themselves so a
    ladder change cannot move them silently.
    """
    app = _app()
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        resting = sidebar.render().plain.splitlines()[-1]
        assert "f9 focus" in resting, f"premise: the resting ladder shows the way in: {resting!r}"

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()
        assert sidebar.has_focus, "premise: the press focused the list"

        focused = sidebar.render().plain.splitlines()[-1]
        assert focused.strip() == "esc return · f10 pin", focused


# -- Esc and F9 keep their shipped meanings --------------------------------------


@pytest.mark.asyncio
async def test_esc_after_a_row_press_closes_the_panel_and_returns_the_keyboard() -> None:
    """T7: on the list, Esc dismisses and the keyboard goes back where it was."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.text = "draft"
        editor.focus()
        await pilot.pause()
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        await pilot.pause()

        rows = _rows(app, sidebar)
        await pilot.click(offset=(_row_body_x(sidebar), rows["sess"]))
        for _ in range(4):
            await pilot.pause()
        assert sidebar.has_focus, "premise: the list owns the keyboard"

        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()

        assert not sidebar.display, "Esc did not dismiss the panel"
        assert app.focused is editor, "Esc did not return the keyboard to the composer"
        assert editor.text == "draft"


@pytest.mark.asyncio
async def test_f9_still_toggles_the_keyboard_without_closing_the_panel() -> None:
    """T9: f9 in, f9 out — the panel stays open either way."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        sidebar = await _open_list(pilot, app, "sess", "sess-b")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        await pilot.press("f9")
        for _ in range(2):
            await pilot.pause()
        assert sidebar.has_focus, "f9 did not focus the list"
        assert sidebar.display, "f9 closed the panel"

        await pilot.press("f9")
        for _ in range(2):
            await pilot.pause()
        assert not sidebar.has_focus, "f9 did not return the keyboard"
        assert sidebar.display, "f9 closed the panel on the way out"
        assert editor.has_focus

"""The projects create form (S6d parity P4): slug, tab order, refusal, landing.

Driven through the real ``OperatorApp`` with an isolated HOME/config — the same
harness the P1/P2/P3 suites use — because the parts worth asserting are the
crossings: the view's mode machine, the app's write, and the store's own
refusals coming back as lines IN the form (spec §7.7).

The four gestures that make this a form rather than a page of text are each
pinned here: `c` opens it, the title drives the key until the key is touched,
`ctrl+s` writes through the app, and `esc` leaves — immediately when clean,
behind one inline confirm when not.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.projects import ProjectRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.projects_form import (
    DESCRIPTION_SCAFFOLD,
    DISCARD_PROMPT,
    ProjectsFormPage,
    slug_for_title,
)
from local_operator.tui.widgets.subagent_view import HintButton
from tests.unit.tui.test_projects_view import (
    _boot,
    _factory,
    _open,
    _ProjectSession,
    _registry,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _page(view: Any) -> ProjectsFormPage:
    page = view._form_page
    assert isinstance(page, ProjectsFormPage)
    return page


async def _open_form(pilot: Any, view: Any) -> ProjectsFormPage:
    await pilot.press("c")
    await pilot.pause()
    await pilot.pause()
    return _page(view)


def _store(tmp_path: Path) -> ProjectRegistry:
    return ProjectRegistry(tmp_path)


# ---------------------------------------------------------------------------
# Pure slug rule
# ---------------------------------------------------------------------------


def test_the_slug_rule_is_the_spec_rule() -> None:
    assert slug_for_title("TUI parity") == "tui-parity"
    assert slug_for_title("  Spaces, and  punctuation!  ") == "spaces-and-punctuation"
    assert slug_for_title("Under_score.dot-dash") == "under_score.dot-dash"
    assert slug_for_title("---leading---") == "leading"
    assert slug_for_title("") == ""


# ---------------------------------------------------------------------------
# Opening the form
# ---------------------------------------------------------------------------


async def test_c_opens_the_create_form_with_the_scaffold_and_its_own_chrome(
    tmp_path: Path,
) -> None:
    """`c` is a MODE of the page: canvas hidden, form titled, footer replaced."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        assert view._mode == "form"
        assert page.display and not view._body.display and not view._detail_page.display
        # The description editor opens on the scaffold (starter text, not a value)
        # and the caret is on the Goals heading a writer fills in first.
        assert page._description.text == DESCRIPTION_SCAFFOLD
        assert page._status_row.value == "planning"
        rows = view.rendered_rows()
        assert any("projects · new project" in row for row in rows)
        assert any("the key is the reference handle" in row for row in rows)
        # The ladder is the spec's own 41-cell rung, at 100 columns.
        hints = " ".join(
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display
        )
        assert "tab" in hints and "ctrl+s" in hints and "esc" in hints


class _FakeTeams:
    """The smallest thing `_team_registry()` can hand back (list_teams only)."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def list_teams(self) -> list[Any]:
        return [SimpleNamespace(name=name) for name in self._names]


async def test_the_team_field_hint_lists_the_registered_teams(tmp_path: Path) -> None:
    """Spec §7.7's dim hint names the vocabulary the reader may actually write.

    Read from the SAME registry `/team`'s argument list uses, so the form cannot
    offer a team the team store does not have.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    session.team_registry = _FakeTeams(["core", "personal"])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _open_form(pilot, view)

        assert any("known: core, personal" in row for row in view.rendered_rows())


async def test_the_title_drives_the_key_until_the_key_is_touched(tmp_path: Path) -> None:
    """Spec §7.7's auto-slug, including the two edges: detach and re-follow."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press(*"TUI parity")
        await pilot.pause()
        assert page._key_input.value == "tui-parity"

        # Focusing the KEY selects what is in it (Textual's own rule), so the
        # first keystroke REPLACES the slug — a real edit, which detaches it…
        page._key_input.focus()
        await pilot.press("x")
        await pilot.pause()
        assert page._key_input.value == "x"
        # … and the title no longer moves it. (Focusing a field selects what is
        # in it, so the title is extended deliberately here.)
        page._title_input.focus()
        await pilot.press("end")
        await pilot.press(*"!!")
        await pilot.pause()
        assert page._title_input.value == "TUI parity!!"
        assert page._key_input.value == "x", "the key must stop following"

        # CLEARING it re-follows (spec §7.7) — and the slug trims the trailing
        # separators the `!!` produced, as the rule says.
        page._key_input.focus()
        await pilot.press("backspace")
        await pilot.pause()
        await pilot.pause()
        assert page._key_input.value == "tui-parity", page._key_input.value


async def test_tab_and_shift_tab_walk_the_fields(tmp_path: Path) -> None:
    """`tab`/`shift+tab` move between fields — shift+tab through the APP's key.

    `shift+tab` is an app-wide PRIORITY binding (`cycle_effort`), so this is
    really a test of the delegation: if the app did not ask the page, the focus
    would not move at all (measured on the settings page's capture trap).
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        assert app.focused is page._title_input
        await pilot.press("tab")
        assert app.focused is page._key_input
        await pilot.press("tab")
        assert app.focused is page._description
        await pilot.press("shift+tab")
        assert app.focused is page._key_input, "shift+tab must walk BACK a field"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


async def test_local_validation_refuses_in_the_form_and_focuses_the_field(
    tmp_path: Path,
) -> None:
    """A bad value is a line under its field, and the cursor lands on it."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press(*"Fine")
        await pilot.pause()
        page._estimate_input.focus()
        await pilot.press("0")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert view._mode == "form", "a refused save must keep the form open"
        assert app.focused is page._estimate_input
        errors = [row for row in view.rendered_rows() if "error:" in row]
        assert any("estimate" in row for row in errors), errors
        # Nothing was written.
        assert _store(tmp_path).get_project_by_name("fine") is None


async def test_a_store_refusal_lands_in_the_form_with_the_draft_intact(
    tmp_path: Path,
) -> None:
    """A taken name is the STORE's sentence, in the form, with the values kept."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "taken")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press(*"taken")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert view._mode == "form"
        assert page._title_input.value == "taken", "the draft must survive a refusal"
        rows = view.rendered_rows()
        assert any("already exists" in row for row in rows), rows


# ---------------------------------------------------------------------------
# A landed create
# ---------------------------------------------------------------------------


async def test_ctrl_s_creates_through_the_store_and_lands_on_the_new_row(
    tmp_path: Path,
) -> None:
    """The write goes through `registry.create_project`, then the canvas moves.

    The scaffold is NOT submitted: a project created by someone who never
    opened the description has no description, rather than a body made of
    headings nobody wrote.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press(*"Brand new")
        await pilot.pause()
        page._team_input.focus()
        await pilot.press(*"core")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

        created = _store(tmp_path).get_project_by_name("brand-new")
        assert created is not None
        assert created.title == "Brand new"
        assert created.team == "core"
        assert created.status == "planning"
        assert created.description == "", "the untouched scaffold is not a value"

        assert view._mode == "canvas", "a landed create leaves the form"
        assert view.current_project_id() == created.id, "the cursor follows the new row"
        assert view._notice == "created 'Brand new'"
        assert any("brand-new" in row for row in view.rendered_rows())


async def test_a_long_refused_create_keeps_the_sentence_in_the_form(
    tmp_path: Path,
) -> None:
    """A name the store rejects for shape is refused in-form too — not a crash.

    The key is what the store validates, and the form must not invent a second
    wording for a rule the store owns: the line is the store's own sentence.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        page._key_input.focus()
        await pilot.press(*"-broken")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert view._mode == "form"
        rows = view.rendered_rows()
        assert any("error:" in row for row in rows), rows
        assert len(_store(tmp_path).list_projects()) == 1


# ---------------------------------------------------------------------------
# Cancelling
# ---------------------------------------------------------------------------


async def test_esc_on_a_clean_form_closes_and_a_dirty_one_confirms(
    tmp_path: Path,
) -> None:
    """Spec §7.7: clean `esc` leaves; dirty `esc` asks, and `y` discards."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        # Clean: an untouched form (scaffold and all) is not an edit.
        assert page.dirty is False
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"

        page = await _open_form(pilot, view)
        await pilot.press(*"Half typed")
        await pilot.pause()
        assert page.dirty is True
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "form", "a dirty cancel must not close the form"
        assert page.confirming is True
        assert any(DISCARD_PROMPT in row for row in view.rendered_rows())
        # The confirm owns the keyboard: a printable key is an answer, not text.
        await pilot.press("y")
        await pilot.pause()
        assert view._mode == "canvas"
        assert _store(tmp_path).get_project_by_name("half-typed") is None


async def test_cancel_from_the_detail_returns_to_that_page(tmp_path: Path) -> None:
    """`detail --c--> form --esc--> detail` (spec §1): the entry is the return."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"
        opened = view._detail_project_id

        await pilot.press("c")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "form"

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail", "a cancel returns to the page it came from"
        assert view._detail_project_id == opened


# ---------------------------------------------------------------------------
# The mode's invariants
# ---------------------------------------------------------------------------


async def test_the_canvas_keys_are_inert_while_the_form_is_up(tmp_path: Path) -> None:
    """`↑`/`↓` are not an Input's own keys, so they reach the view — and stop.

    Without the gate the canvas would move under a form the reader is filling
    in, and the footer's own chrome would repaint against a canvas nobody can
    see.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta", "gamma")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        before = view._cursor
        page = await _open_form(pilot, view)

        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("pageup")
        await pilot.press("end")
        await pilot.pause()

        assert view._cursor == before
        assert view._mode == "form"
        assert page._title_input.value == "", "printable keys belong to the field"

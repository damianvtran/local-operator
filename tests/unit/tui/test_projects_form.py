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
from typing import Any, cast

import pytest

from local_operator.model.configure import build_model_spec
from local_operator.projects import ProjectRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.projects_form import (
    DESCRIPTION_HINT,
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
        # … the hint SAYS it detached (agent review round 1, R1-3: only the
        # write path set it, so it went on claiming the key still followed).
        assert any(
            "key hint: set by hand — clear it to follow the title again" in row
            for row in view.rendered_rows()
        )

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
        assert any(
            "key hint: follows the title until you edit it" in row for row in view.rendered_rows()
        )


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


async def test_a_bracket_carrying_refusal_is_data_not_markup(tmp_path: Path) -> None:
    """The sentence is a store message, and store messages carry brackets.

    Textual's ``Static`` parses markup by default, so an unescaped sentence
    would raise inside the handler that paints it (QA round 1, Q-3).
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        sentence = "project ['taken'] is [not] a valid [name]"
        page.show_refusal(sentence)
        await pilot.pause()

        rows = view.rendered_rows()
        assert any(sentence in row for row in rows), rows


async def test_the_controls_are_one_row_and_the_form_is_short(tmp_path: Path) -> None:
    """The measured geometry behind D3: stock Inputs are 3-row bordered boxes.

    The form measured 58 content rows with 9 visible at 80x24 before the rule;
    it is 43 now, with one-row controls and a 1-cell themed scrollbar (D2/D3).
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        assert page._title_input.styles.height is not None
        assert page._title_input.region.height == 1, page._title_input.region
        # No border: the stock Input paints a TALL one (which is half of why it
        # was 3 rows). Textual renders an ABSENT edge as the empty string —
        # measured `''` here against `'tall'` before the rule.
        assert page._title_input.styles.border.top[0] == "", page._title_input.styles.border
        assert page._fields_scroll.styles.scrollbar_size_vertical == 1
        assert page._fields_scroll.virtual_size.height <= 46, page._fields_scroll.virtual_size


async def test_the_discard_confirm_is_visible_wherever_the_form_is(tmp_path: Path) -> None:
    """D1/Q-1: the question is pinned OUTSIDE the scroll body.

    It used to be the last child of the scrolled page — measured at y=62 with
    the viewport at scroll 0 at 80x24, i.e. a question nobody could see.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press(*"half typed")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert page.confirming is True
        page_box = page.region
        confirm = page._confirm_row.region
        assert confirm.height == 1, confirm
        assert page_box.y <= confirm.y < page_box.y + page_box.height, (page_box, confirm)


async def test_the_tab_order_runs_start_before_target(tmp_path: Path) -> None:
    """N1: the dated pair reads in the direction a plan does."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        order = page.fields()
        assert order.index(page._start_input) < order.index(page._target_input)


async def test_the_hints_state_the_rules_a_reader_cannot_guess(tmp_path: Path) -> None:
    """D7/D8: the scaffold's saving rule, and how a `‹ value ›` row is stepped."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _open_form(pilot, view)

        rows = view.rendered_rows()
        assert any("nothing is saved" in row for row in rows), rows
        assert any("← → change" in row for row in rows), rows


class _EffortProjectSession(_ProjectSession):
    """The projects rig with a REAL ``ModelSpec``, so the effort chord is live.

    ``FakeSession.model`` is ``None`` and the app refuses the effort chord before
    it can act, which would make the Q-4 pin below read zero cycles for a fix
    that was never there. ``build_model_spec`` is the same offline derivation
    the shipped controller lands on.
    """

    def __init__(self) -> None:
        super().__init__()
        self._spec = build_model_spec("anthropic", "claude-opus-5")

    @property
    def model(self) -> Any:
        return self._spec

    @property
    def model_label(self) -> str:
        return f"{self._spec.provider}/{self._spec.model_id}"

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        self._spec = model


async def test_shift_tab_at_the_confirm_never_moves_a_billable_setting(
    tmp_path: Path,
) -> None:
    """Q-4: the page claims the chord for the WHOLE form mode.

    ``shift+tab`` is an app-wide priority binding (``cycle_effort``). Narrowing
    the claim to "not confirming" let the app fall through the delegation, so
    every press at the discard question moved the session's reasoning effort —
    a billable setting changed silently while a reader answered a question. The
    belt that must stop the journey to the next field is in
    :meth:`ProjectsView.form_focus_previous`, not in the claim.
    """
    session = _EffortProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        # The instrument is live: with no form up, the chord really cycles.
        def level() -> str | None:
            assert app._session is not None
            return cast("str | None", app._session.model.reasoning_effort)

        start = level()
        await pilot.press("shift+tab")
        await pilot.pause()
        cycled = level()
        assert cycled != start, "the rig must be able to cycle, or this proves nothing"

        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        # In the form, the chord walks the fields and touches no setting.
        await pilot.press("shift+tab")
        await pilot.pause()
        assert level() == cycled

        # `shift+tab` walked back to the last field (a cycle row), so the title
        # takes the focus again before anything is typed into it.
        page._title_input.focus()
        await pilot.pause()
        await pilot.press(*"dirty")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert page.confirming is True

        for _ in range(3):
            await pilot.press("shift+tab")
            await pilot.pause()

        assert level() == cycled, "the chord must not move a billable setting at the confirm"
        assert page.confirming is True, "the question must survive the chord"


async def test_the_selection_band_takes_the_composers_treatment(tmp_path: Path) -> None:
    """D9: stock `Input` selection measured 1.41:1 on the light ramp.

    The band is stock ``#9bc3e3`` with the widget's own light glyphs, so the
    text a reader had just selected was the least readable text on the page.
    The form takes the composer's treatment instead, and this pins the two to
    the SAME resolved colours, so a future ramp change cannot separate them.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/theme light")
        await pilot.pause()
        await pilot.pause()
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        composer = app.query_one(Editor).get_component_rich_style("text-area--selection")
        field = page._title_input.get_component_rich_style("input--selection")
        area = page._description.get_component_rich_style("text-area--selection")

        assert field.bgcolor == composer.bgcolor, (field.bgcolor, composer.bgcolor)
        assert field.color == composer.color, (field.color, composer.color)
        assert area.bgcolor == composer.bgcolor, (area.bgcolor, composer.bgcolor)
        assert area.color == composer.color, (area.color, composer.color)


async def test_the_question_gets_its_own_row(tmp_path: Path) -> None:
    """D10: the confirm must not read as the clipped field's value.

    The blank row above it is the sheet's one sanctioned spacing class rather
    than a margin of the form's own.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press(*"dirty")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert page._confirm_row.has_class("gap-above")
        body = page._fields_scroll.region
        confirm = page._confirm_row.region
        assert confirm.y == body.y + body.height + 1, (body, confirm)


async def test_the_hints_are_the_labels_voice(tmp_path: Path) -> None:
    """D11: a hint wears the label's quiet ink, and does not repeat a placeholder."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        muted = str(app.get_css_variables()["lo-muted"]).lower()
        hint = page._title_block._hint_widget.styles.color
        assert hint is not None and hint.hex.lower() == muted, (hint, muted)
        rows = view.rendered_rows()
        assert any("at most 8 tags" in row for row in rows), rows
        assert not any("comma or space separated · at most" in row for row in rows), rows


async def test_the_caret_takes_the_composers_treatment(tmp_path: Path) -> None:
    """D13: the caret was the theme's near-white on the light field's own fill.

    About 1:1 — the one mark a reader needs while typing was invisible. Same
    treatment as the composer's caret, pinned to its resolved component style.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/theme light")
        await pilot.pause()
        await pilot.pause()
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        composer = app.query_one(Editor).get_component_rich_style("text-area--cursor")
        field = page._title_input.get_component_rich_style("input--cursor")
        area = page._description.get_component_rich_style("text-area--cursor")
        assert field.color == composer.color, (field.color, composer.color)
        assert field.bgcolor == composer.bgcolor, (field.bgcolor, composer.bgcolor)
        assert area.color == composer.color, (area.color, composer.color)


async def test_the_canvas_advertises_the_create_key(tmp_path: Path) -> None:
    """U1: `c` was a binding with no hint — nothing said how to make a project.

    The ladder is the honest home for it (the empty-store SENTENCE is shared
    with the transcript's `/project` reply, where `c` means nothing), and the
    two widest sizes a reader actually works at must carry it.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    for size in ((100, 30), (150, 40)):
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await _boot(pilot, app)
            view = await _open(pilot, app)
            assert view._create_hint.display is True, size
            assert "create" in str(view._create_hint.rendered()), (
                size,
                view._create_hint.rendered(),
            )


async def test_the_description_caret_starts_under_the_first_heading(tmp_path: Path) -> None:
    """U2: it opened ON `## Goals`, so the first keystroke glued onto it."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        assert page._description.text == DESCRIPTION_SCAFFOLD
        assert page._description.cursor_location == (1, 0), page._description.cursor_location
        page._description.focus()
        await pilot.press(*"hello")
        await pilot.pause()
        assert page._description.text.startswith("## Summary\nhello"), page._description.text


async def test_a_store_refusal_lands_on_the_field_it_names(tmp_path: Path) -> None:
    """U3: it painted under the TITLE and stayed while the title was corrected."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press(*"alpha")
        await pilot.pause()
        page._team_input.focus()
        await pilot.press("ctrl+s")
        await pilot.pause()

        rows = view.rendered_rows()
        assert any(row.startswith("key error:") for row in rows), rows
        assert not any(row.startswith("title error:") for row in rows), rows
        assert app.focused is page._key_input

        # Correcting the offending value clears the line: the reader's fix IS
        # the answer to the sentence under the field.
        page._title_input.focus()
        await pilot.press("x")
        await pilot.pause()
        assert not any(row.startswith("key error:") for row in view.rendered_rows())


async def test_an_empty_submit_points_at_the_title(tmp_path: Path) -> None:
    """U4: the form is title-first, so the empty case belongs on the title."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        await pilot.press("ctrl+s")
        await pilot.pause()

        rows = view.rendered_rows()
        assert any(row.startswith("title error:") for row in rows), rows
        assert not any(row.startswith("key error:") for row in rows), rows
        assert app.focused is page._title_input


async def test_n_and_enter_answer_the_question_keep_editing(tmp_path: Path) -> None:
    """U5: both were swallowed in silence; `esc` stays the advertised answer."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    for key in ("n", "enter"):
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            view = await _open(pilot, app)
            page = await _open_form(pilot, view)
            await pilot.press(*"half")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert page.confirming is True
            await pilot.press(key)
            await pilot.pause()
            await pilot.pause()
            assert page.confirming is False, key
            assert view._mode == "form", key
            assert page._title_input.value == "half", key


async def test_the_scaffold_hint_goes_away_once_the_writer_types(tmp_path: Path) -> None:
    """U6: it described a state that no longer existed."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        page = await _open_form(pilot, view)

        assert page._description_block.hint == DESCRIPTION_HINT
        assert any("nothing is saved" in row for row in view.rendered_rows())
        page._description.focus()
        await pilot.press("h")
        await pilot.pause()
        assert page._description_block.hint == ""
        assert not any("nothing is saved" in row for row in view.rendered_rows())


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
        # The confirm OWNS the keyboard (agent review round 1, R1-1): tab and
        # shift+tab must not carry the cursor into a field, where the advertised
        # `y` would be typed instead of answered.
        await pilot.press("tab")
        await pilot.press("shift+tab")
        await pilot.pause()
        assert page.confirming is True, "a focus key must not end the question"
        assert app.focused is page, app.focused
        # `n` answers the question too — keep editing (UX round 1, U5) — so the
        # confirm is re-armed below before the advertised `y` discards.
        await pilot.press("n")
        await pilot.pause()
        await pilot.pause()
        assert page.confirming is False
        assert view._mode == "form" and page._title_input.value == "Half typed"
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert page.confirming is True
        # The advertised `y` is what discards.
        await pilot.press("y")
        await pilot.pause()
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

"""``/project`` on the TUI: verbs, receipts, argument rows and dispatch parity.

The receipts are pinned the way ``test_slash_echo`` pins the echo policy: one
outcome per form, asserted at the REAL ``OperatorApp`` dispatch, and then
compared — input for input — against what the routed runtime mirror
(``serving.py::_project_slash``) answers for the same store. That comparison is
the point: both front ends call ONE runner
(``slash_commands.run_project_slash_op``), and this file is what would fail if
a later edit gave either surface its own sentence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.projects import ProjectEdit, ProjectRegistry
from local_operator.slash_commands import (
    PROJECT_SUBCOMMAND_HELP,
    PROJECT_SUBCOMMANDS,
    project_empty_text,
    project_show_refusal_text,
    project_subcommand_rows,
    project_unavailable_text,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import PickerMode
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: A session id the store's grammar accepts (12 lowercase hex), so `new` and
#: `link` exercise the auto-link path rather than its no-id degradation.
SESSION_ID = "ab12cd34ef56"


class _ProjectSession(FakeSession):
    """A FakeSession whose id is a linkable one; everything else is inherited."""

    @property
    def session_id(self) -> str:
        return SESSION_ID


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the whole machine at a scratch config root.

    The page and the verb runner both read ``config_dir()/projects`` and the
    run records beside it; a test that used the developer's own root would read
    their projects and — on `new`/`delete` — write them.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _registry(tmp_path: Path, *names: str) -> ProjectRegistry:
    registry = ProjectRegistry(tmp_path)
    for name in names:
        registry.create_project(ProjectEdit(name=name))
    return registry


async def _boot(pilot: Any, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type(pilot: Any, text: str) -> None:
    """Type ``text`` through real key presses (the picker detects on keystrokes)."""
    for char in text:
        await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _rows(app: OperatorApp) -> list[tuple[str, str, str, bool]]:
    """``(name, description, detail, alert)`` for every argument row offered."""
    picker = app.query_one(Editor).picker
    assert picker.mode is PickerMode.ARGUMENT, "the picker is not in argument mode"
    return [
        (choice.name, choice.description, choice.detail, choice.alert) for choice in picker._choices
    ]


# -- the verbs ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_and_empty_receipt(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project list")
        await pilot.pause()
        assert _notices(app)[0].startswith("- alpha [active]")

    session = _ProjectSession()
    session.project_registry = _registry(tmp_path / "empty")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project")
        await pilot.pause()
        assert project_empty_text() in _notices(app)[0]


@pytest.mark.asyncio
async def test_new_creates_and_links_this_session(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project new payments-migration")
        await pilot.pause()
        assert _notices(app)[0] == (
            "created project 'payments-migration' [active] and linked this "
            f"session ({SESSION_ID})."
        )
        project = session.project_registry.get_project_by_name("payments-migration")
        assert project is not None and project.sessions == [SESSION_ID]

        # The duplicate refusal names the way in rather than dying silently.
        app._run_slash_command("/project new payments-migration")
        await pilot.pause()
        assert "already exists" in _notices(app)[-1]
        assert "/project show 'payments-migration'" in _notices(app)[-1]


@pytest.mark.asyncio
async def test_link_and_unlink_receipts_name_the_link_set(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project link alpha")
        await pilot.pause()
        assert _notices(app)[-1] == f"linked session {SESSION_ID} to 'alpha' (1 linked now)."
        app._run_slash_command("/project link alpha")
        await pilot.pause()
        assert _notices(app)[-1] == (
            f"session {SESSION_ID} was already linked to 'alpha' (1 linked)."
        )
        app._run_slash_command("/project unlink alpha")
        await pilot.pause()
        assert _notices(app)[-1] == f"unlinked session {SESSION_ID} from 'alpha' (0 linked now)."
        app._run_slash_command("/project unlink alpha")
        await pilot.pause()
        assert "is not linked to 'alpha'" in _notices(app)[-1]


@pytest.mark.asyncio
async def test_delete_needs_the_typed_yes(tmp_path: Path) -> None:
    """``_cmd_delete``'s two-step shape: rehearsal first, ``yes`` removes."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)

        app._run_slash_command("/project delete alpha")
        await pilot.pause()
        rehearsal = _notices(app)[-1]
        assert "Nothing was deleted" in rehearsal
        assert "/project delete alpha yes to confirm." in rehearsal
        assert session.project_registry.get_project_by_name("alpha") is not None

        app._run_slash_command("/project delete alpha yes")
        await pilot.pause()
        assert _notices(app)[-1] == "deleted project 'alpha'."
        assert session.project_registry.get_project_by_name("alpha") is None

        # Deleting an unknown/renamed name is a notice naming `list`, never a
        # silent no-op (design §5.3).
        app._run_slash_command("/project delete nope")
        await pilot.pause()
        assert _notices(app)[-1] == project_show_refusal_text("nope")


@pytest.mark.asyncio
async def test_show_opens_the_page_on_that_project(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project show beta")
        await pilot.pause()
        view = app._projects_view
        assert view is not None
        assert view.view_type == "list"
        assert view.tracked == 2
        # The cursor lands on the named project, not the first row.
        assert view._views[view.cursor]["project"]["name"] == "beta"
        # No user row and no prompt: the page is the receipt (the MODE rule).
        assert session.prompts == []


@pytest.mark.asyncio
async def test_show_unknown_name_names_the_listing(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project show nope")
        await pilot.pause()
        assert app._projects_view is None
        assert _notices(app)[-1] == project_show_refusal_text("nope")


@pytest.mark.asyncio
async def test_unknown_word_is_refused_with_the_vocabulary(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project frobnicate")
        await pilot.pause()
        notice = _notices(app)[-1]
        assert "unknown /project subcommand 'frobnicate'" in notice
        for word in PROJECT_SUBCOMMANDS:
            assert word in notice


@pytest.mark.asyncio
async def test_registry_less_session_names_the_surfaces() -> None:
    session = _ProjectSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project list")
        await pilot.pause()
        assert _notices(app)[-1] == project_unavailable_text()


# -- dispatch parity with the routed mirror ----------------------------------


def _serving_texts(registry: ProjectRegistry, inputs: list[str]) -> list[tuple[str, str]]:
    from types import SimpleNamespace

    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.runtime.serving import ServingSessionHandle

    handle = ServingSessionHandle.__new__(ServingSessionHandle)
    session = SimpleNamespace(project_registry=registry, session_id=SESSION_ID)
    results = []
    for text in inputs:
        result = handle._project_slash(session, text.lstrip("/project").strip(), SlashResult)
        results.append((result.text, result.style))
        if text.startswith("/project") and result.kind != "notice":
            raise AssertionError(f"non-notice result for {text}: {result}")
    return results


@pytest.mark.asyncio
async def test_the_tui_and_the_routed_mirror_answer_identically(tmp_path: Path) -> None:
    """One runner, two front ends: the same inputs must yield the same words.

    Driven through BOTH real dispatch paths against two stores seeded the same
    way — the TUI's notices (text + kind) and the runtime's ``SlashResult``
    (text + style). A drift in either handler fails here rather than in the
    field, where a phone and a terminal would disagree about one store.
    """
    inputs = [
        "/project",  # empty store on the first call
        "/project new beta",
        "/project new beta",
        "/project link beta",
        "/project unlink beta",
        "/project delete beta",
        "/project delete beta yes",
        "/project show nope",
        "/project frobnicate",
        "/project list",
    ]
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path / "tui")
    app = OperatorApp(lambda: _factory(session))
    tui: list[tuple[str, str]] = []
    recorded: list[tuple[str, str]] = []

    def _record(text: str, kind: str = "info") -> None:
        recorded.append((text, kind))

    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        # The handler's own notice callback, not the transcript: this is the
        # exact (text, kind) pair both surfaces decide, with no rendering in
        # between to blur a drift.
        original = app._notice
        app._notice = _record  # type: ignore[method-assign]
        try:
            for text in inputs:
                app._run_slash_command(text)
                await pilot.pause()
        finally:
            app._notice = original  # type: ignore[method-assign]
        tui = list(recorded)

    routed = _serving_texts(_registry(tmp_path / "routed"), inputs)
    assert [text for text, _kind in tui] == [text for text, _style in routed]
    assert [kind for _text, kind in tui] == [style for _text, style in routed]


def test_the_tui_show_page_and_the_mirror_receipt_read_one_composition(
    tmp_path: Path,
) -> None:
    """``show`` differs by PRESENTATION, not by answer: both read the same view.

    The TUI opens the page (no receipt); the routed path prints the composed
    view as text. This pins the mirror's receipt to the composition the page
    renders, field by field.
    """
    from local_operator.projects import build_project_view
    from local_operator.slash_commands import project_show_receipt

    registry = _registry(tmp_path, "alpha")
    project = registry.get_project_by_name("alpha")
    assert project is not None
    view = build_project_view(project, config_dir=tmp_path)
    receipt = project_show_receipt(view)
    assert "alpha [active]" in receipt
    assert "progress (none recorded):" in receipt
    assert "linked sessions: (none)" in receipt


# -- argument rows -----------------------------------------------------------


@pytest.mark.asyncio
async def test_first_slot_offers_the_six_words_with_their_help(tmp_path: Path) -> None:
    """The rows come from the vocabulary, help included, pinned equal here.

    A word the picker offers that the handler refuses (or the reverse) is the
    drift ``PROJECT_SUBCOMMANDS`` exists to prevent, and this is the test that
    reads both through the one table.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        await _type(pilot, "/project ")
        rows = _rows(app)
    assert [(name, description) for name, description, _detail, _alert in rows] == list(
        project_subcommand_rows()
    )
    assert [name for name, _d, _det, _a in rows] == list(PROJECT_SUBCOMMANDS)
    for word, _help_text in project_subcommand_rows():
        assert PROJECT_SUBCOMMAND_HELP[word]
    # The destructive verb carries the alert tint (the `/mcp remove` precedent).
    assert [alert for _n, _d, _det, alert in rows] == [
        word == "delete" for word in PROJECT_SUBCOMMANDS
    ]


@pytest.mark.asyncio
async def test_second_slot_offers_compound_name_rows(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        await _type(pilot, "/project show ")
        rows = _rows(app)
        assert [name for name, _d, _det, _a in rows] == ["show alpha", "show beta"]
        # The detail column carries the status — a live fact, not a filler.
        assert all(detail == "active" for _n, _d, detail, _a in rows)


@pytest.mark.asyncio
async def test_new_slot_offers_no_rows(tmp_path: Path) -> None:
    """`new` takes a name that does not exist yet: no rows, so the user types.

    Offering existing names here would be offering the conflict refusal.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        await _type(pilot, "/project new ")
        picker = app.query_one(Editor).picker
        assert picker.mode is not PickerMode.ARGUMENT or not picker._choices

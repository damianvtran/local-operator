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
from local_operator.tui.widgets.command_picker import ArgumentChoice, PickerMode
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

        # The duplicate refusal names the way in rather than dying silently —
        # and the embedded command RUNS verbatim (F6: the quoted form 404'd a
        # second time when copied), so it is exercised as typed.
        app._run_slash_command("/project new payments-migration")
        await pilot.pause()
        duplicate = _notices(app)[-1]
        assert "already exists" in duplicate
        assert "/project show payments-migration opens the existing row." in duplicate
        embedded = duplicate.split("— ", 1)[1].split(" opens the existing row.", 1)[0]
        assert embedded == "/project show payments-migration"
        app._run_slash_command(embedded)
        await pilot.pause()
        view = app._projects_view
        assert view is not None and view.tracked == 1
        assert view.cursor == 0


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


# -- remediation round 1: the typed-`yes` edge, unreadable stores, receipts ---


@pytest.mark.asyncio
async def test_delete_yes_handles_a_project_named_yes(tmp_path: Path) -> None:
    """F7: a lone ``yes`` is a NAME, and its rehearsal names the ``yes yes`` form.

    The old parse stripped the trailing ``yes`` as the confirmation, leaving an
    empty name, and answered ``name a project`` — which neither said the row
    existed nor named its spelling.
    """
    session = _ProjectSession()
    registry = _registry(tmp_path, "yes")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project delete yes")
        await pilot.pause()
        rehearsal = _notices(app)[-1]
        assert "Nothing was deleted" in rehearsal
        assert "/project delete yes yes to confirm." in rehearsal
        assert registry.get_project_by_name("yes") is not None  # nothing deleted

        app._run_slash_command("/project delete yes yes")
        await pilot.pause()
        assert "deleted project 'yes'" in _notices(app)[-1]
        assert registry.get_project_by_name("yes") is None


@pytest.mark.asyncio
async def test_delete_yes_with_an_unresolvable_prefix_refuses_by_name(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project delete ghost yes")
        await pilot.pause()
        assert _notices(app)[-1] == (
            "no project named 'ghost' — /project list shows every tracked project."
        )


@pytest.mark.asyncio
async def test_unreadable_store_refuses_without_leaking_a_path(tmp_path: Path) -> None:
    """QA Q4: ``chmod 000`` must not read as an empty store, and no path leaks."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="alpha"))
    import os

    os.chmod(projects_dir, 0o000)
    try:
        session = _ProjectSession()
        # A FRESH reader: the degraded state this test is about (an existing
        # registry keeps its 5 s cache, which is the store's own documented
        # behaviour, not the surfaces').
        session.project_registry = ProjectRegistry(tmp_path)
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(120, 32)) as pilot:
            await _boot(pilot, app)
            for command in ("/project list", "/project show alpha", "/project delete alpha"):
                app._run_slash_command(command)
                await pilot.pause()
                answer = _notices(app)[-1]
                assert "unreadable" in answer, command
                assert "no projects yet" not in answer
                assert "no project named" not in answer
                assert str(tmp_path) not in answer
            app._run_slash_command("/project new x")
            await pilot.pause()
            answer = _notices(app)[-1]
            assert "could not create the project" in answer
            assert "permission denied" in answer
            assert str(tmp_path) not in answer
    finally:
        os.chmod(projects_dir, 0o755)
    # Readable again: the store recovers without a restart.
    recovered = ProjectRegistry(tmp_path)
    assert recovered.load_error is None
    assert recovered.get_project_by_name("alpha") is not None


def test_show_receipt_states_tags_and_missing_links() -> None:
    """F5 + the related note: ``missing`` for a gone directory, ``tags:`` always."""
    from local_operator.slash_commands import project_show_receipt

    view: dict[str, Any] = {
        "project": {
            "name": "alpha",
            "status": "active",
            "description": "",
            "progress": "",
            "progress_updated_at": None,
            "tags": ["infra", "q3"],
            "milestones": [
                {"name": "beta", "target_date": "2025-01-01", "completed_at": None},
                {"name": 17},  # malformed: the reader prints it without a status
            ],
            "estimate": None,
            "estimate_unit": "points",
        },
        "progress_stale": False,
        "sessions": [{"session_id": "deadbeef0000", "exists": False, "title": "", "runtime": None}],
    }
    receipt = project_show_receipt(view)
    assert "tags: infra, q3" in receipt
    assert "- deadbeef0000 [missing]" in receipt
    assert "[stopped]" not in receipt
    assert "  - beta [overdue] target 2025-01-01" in receipt
    assert "  - 17 [unknown] target —" in receipt


@pytest.mark.asyncio
async def test_picker_offers_the_delete_confirmation_row(tmp_path: Path) -> None:
    """Finding 9: the picker carries the way into `delete`'s confirmation.

    Rows exist once per opening, so the confirm variant is offered up front and
    the typed name FILTERS it — exactly the row the user needs after typing a
    name, and never for any other name.
    """
    from local_operator.tui.widgets.command_picker import argument_suggestions

    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        await _type(pilot, "/project delete ")
        rows = _rows(app)
        assert [name for name, _d, _det, _a in rows] == [
            "delete alpha",
            "delete beta",
            "delete alpha yes",
            "delete beta yes",
        ]
        # The name rows come FIRST: the empty-query default is a name, never a
        # destructive command.
        assert rows[0][3] is False
        assert [alert for name, _d, _det, alert in rows if name.endswith(" yes")] == [
            True,
            True,
        ]
        # Typing the name filters to its own confirm row, and no other's.
        typed = [
            (name, choice)
            for name, choice in argument_suggestions(
                "delete al", [ArgumentChoice(n, d, detail=det, alert=a) for n, d, det, a in rows]
            )
        ]
        names = [name for name, _choice in typed]
        assert "delete alpha yes" in names
        assert "delete beta yes" not in names


@pytest.mark.asyncio
async def test_unreadable_store_recovers_on_the_live_surface(tmp_path: Path) -> None:
    """QA round 2, Q5: the refusal must not be sticky on a repaired store.

    Same app, same registry, no restart: ``chmod`` 000 answers the refusal;
    ``chmod`` back and the VERY NEXT verb reads the store. A permission repair
    moves the inode's ctime, not the directory mtime, and the refusal check
    runs before any read that would refresh the snapshot — so the pre-fix
    surface stayed unreadable for the life of the process (QA measured eight
    receipts over eleven seconds).
    """
    import os

    from local_operator.slash_commands import run_project_slash_op

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    seed = ProjectRegistry(tmp_path)
    seed.create_project(ProjectEdit(name="alpha"))
    os.chmod(projects_dir, 0o000)
    try:
        # The LIVE reader is built while the store is unreadable, so its flag
        # is genuinely set — the state QA reproduced (a surface that has
        # already answered the refusal). ONE registry for the whole run: a
        # fresh one after the repair would mask the bug (re-reading on
        # construction is what made round 1's test pass).
        registry = ProjectRegistry(tmp_path)
        assert registry.load_error is not None
        session = _ProjectSession()
        session.project_registry = registry
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(120, 32)) as pilot:
            await _boot(pilot, app)
            app._run_slash_command("/project list")
            await pilot.pause()
            assert "unreadable" in _notices(app)[-1]
            os.chmod(projects_dir, 0o755)
            # No sleep, no restart, no fresh registry.
            app._run_slash_command("/project list")
            await pilot.pause()
            assert "unreadable" not in _notices(app)[-1]
            assert "alpha" in _notices(app)[-1]
            app._run_slash_command("/project show alpha")
            await pilot.pause()
            await pilot.pause()
            assert app._projects_view is not None
            # The runner path (phone/mobile receipts) recovers on the same
            # registry object, for both the verb gate and `show`.
            text, _style = run_project_slash_op(
                "list", "", registry=registry, config_dir=tmp_path, session_id=None
            )
            assert "unreadable" not in text and "alpha" in text
            text, _style = run_project_slash_op(
                "show", "alpha", registry=registry, config_dir=tmp_path, session_id=None
            )
            assert "no project named" not in text
    finally:
        os.chmod(projects_dir, 0o755)


def test_listing_rows_show_the_title_first_with_the_key() -> None:
    from local_operator.projects import Project
    from local_operator.slash_commands import project_listing_rows

    titled = Project(id="a" * 32, name="payments-migration", title="Q4 Payments Migration")
    untitled = Project(id="b" * 32, name="plain-key")
    rows = project_listing_rows([titled, untitled])
    assert rows[0].startswith("- Q4 Payments Migration (payments-migration) [active]")
    assert rows[1].startswith("- plain-key [active]")  # fallback: the key is the label


def test_show_receipt_states_title_key_owner_team_and_history() -> None:
    from local_operator.slash_commands import project_show_receipt

    view: dict[str, Any] = {
        "project": {
            "name": "payments-migration",
            "title": "Q4 Payments Migration",
            "owner": "Damian",
            "team": "Platform",
            "status": "active",
            "description": "",
            "progress": "second",
            "progress_updated_at": None,
            "progress_reported_by": "",
            "tags": [],
            "updates": [
                {
                    "at": "2026-09-27T21:00:00Z",
                    "text": "first",
                    "by": "ab12cd34ef56",
                    "attachments": [
                        {
                            "name": "shot.png",
                            "kind": "image",
                            "path": "/tmp/x/shot.png",
                            "bytes": 2048,
                        }
                    ],
                },
                {"at": "2026-09-27T21:05:00Z", "text": "second", "by": "", "attachments": []},
            ],
            "milestones": [],
            "estimate": None,
            "estimate_unit": "points",
        },
        "progress_stale": False,
        "sessions": [],
    }
    receipt = project_show_receipt(view)
    assert receipt.startswith("Q4 Payments Migration [active]\nkey: payments-migration")
    assert "owner: Damian" in receipt
    assert "team: Platform" in receipt
    assert "history (2):" in receipt
    assert "  - 2026-09-27T21:05:00Z: second" in receipt  # no reporter → no ` by`
    assert "attachment: shot.png [image, 2.0 KB] /tmp/x/shot.png" in receipt

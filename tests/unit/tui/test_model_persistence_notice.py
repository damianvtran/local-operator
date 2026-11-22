"""Keep display localization separate from machine selectors and legacy adapters."""

from types import SimpleNamespace
from typing import cast

import pytest
from rich.cells import cell_len
from rich.text import Text

from local_operator.i18n.keys import tui_model_persistence as keys
from local_operator.i18n.text import active_locale, set_active_locale
from local_operator.slash_commands import PERSIST_HINT
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptView,
    _break_word,
    _wrap_notice_atoms,
    wrap_cells,
)
from tests.unit.tui.conftest import await_session_boot
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _notice(*, setup: str | None = None, missing: bool = False, label: str = "") -> str:
    # This adapter supplies only the state the production method reads. Real
    # picker opening, styled wrapping and pinned-tail dedup remain pilot tests.
    state = SimpleNamespace(
        _session=SimpleNamespace(model_label=label),
        _setup_state=setup,
        _model_missing_for=missing,
    )
    # The adapter deliberately models only this synchronous method's state;
    # casting the receiver does not claim that it is a runnable Textual app.
    return OperatorApp._persist_hint_notice(cast(OperatorApp, state))


@pytest.fixture(autouse=True)
def restore_locale():
    previous = active_locale()
    yield
    set_active_locale(previous)


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"])
@pytest.mark.parametrize(
    ("setup", "missing", "routes"),
    [
        (None, False, ("persist", "saved", "settings")),
        ("missing_model", True, ("persist", "settings")),
        ("no_hosting", False, ("settings",)),
        ("unknown_hosting", False, ("settings",)),
    ],
)
def test_notice_resolves_each_semantic_branch_at_current_locale(locale, setup, missing, routes):
    set_active_locale(locale)
    expected_routes = " · ".join(
        getattr(keys, route)().envelope(locale=locale)["text"] for route in routes
    )
    # Selector syntax and labels are opaque input, even when they look like
    # markup. Changing either here would silently break copy/paste workflows.
    label = "vendor/[model]/v1:exact"
    expected = keys.with_model(label=label, routes=expected_routes).envelope(locale=locale)["text"]
    assert _notice(setup=setup, missing=missing, label=label) == expected
    assert _notice(setup=setup, missing=missing) == expected_routes
    assert label in expected
    for route in routes:
        command = {"persist": "/model default", "saved": "/model saved", "settings": "/settings"}[
            route
        ]
        assert command in expected


def test_notice_follows_live_locale_without_rewriting_previous_text():
    set_active_locale("en")
    english = _notice(label="opaque/model")
    assert english == (
        f"model: opaque/model — {PERSIST_HINT} · "
        "switch to the boot default with /model saved · /settings edits the boot default too"
    )
    set_active_locale("fr")
    french = _notice(label="opaque/model")
    assert french != english
    assert "saves this for new sessions" not in french
    assert "edits the boot default too" not in french
    assert keys.persist().envelope(locale="en")["text"] == PERSIST_HINT
    set_active_locale("en")
    assert _notice(label="opaque/model") == english


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (80, 24)])
@pytest.mark.parametrize("locale", ["zh-CN", "en"])
async def test_opened_notice_keeps_commands_on_one_rendered_row(monkeypatch, size, locale):
    """Exercise the opening path, not a constructed widget or normalized copy."""
    monkeypatch.setenv("LOP_LANG", locale)

    class OpaqueSession(FakeSession):
        @property
        def model_label(self) -> str:
            return "vendor/[model]/v1:exact"

    session = OpaqueSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await await_session_boot(app)
        app.query_one(Editor).focus()
        await pilot.press("slash", "m", "o", "d", "e", "l", "space")
        await pilot.pause()
        notices = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, NoticeBlock)]
        assert len(notices) == 1
        assert notices[0]._wrap_atoms == (
            "vendor/[model]/v1:exact",
            "/model default",
            "/model saved",
            "/settings",
        )
        renderable = notices[0].renderable
        assert isinstance(renderable, Text)
        rendered = renderable.plain
        assert active_locale() == locale
        for command in ("/model default", "/model saved", "/settings"):
            assert command in rendered, rendered


@pytest.mark.parametrize("width", [-1, 0, 1, 2, 8, 14, 30])
@pytest.mark.parametrize("text", ["", "ordinary prose", "  a   b  ", "a\nb", "模型甲乙丙"])
def test_default_atom_path_is_legacy(text, width):
    assert _wrap_notice_atoms(text, width, ()) == wrap_cells(text, width)
    assert _wrap_notice_atoms(text, width, ("not present",)) == wrap_cells(text, width)
    if width <= 0:
        assert _wrap_notice_atoms(text, width, (text,)) == [text]


@pytest.mark.parametrize("command", ["/model default", "/model saved", "/settings"])
def test_atoms_adjacent_to_prose_add_no_characters(command):
    text = f"前{command}后{command}{command}尾"
    rows = _wrap_notice_atoms(text, len(command), (command,))
    assert "".join(rows) == text
    assert sum(command in row for row in rows) == 3


@pytest.mark.parametrize("width", [1, 2, 8, 13, 14, 15])
def test_oversized_atom_uses_existing_cell_fallback(width):
    atom = "/model default"
    rows = _wrap_notice_atoms(atom, width, (atom,))
    assert "".join(rows) == atom
    assert all(cell_len(row) <= width for row in rows)
    if width >= cell_len(atom):
        assert rows == [atom]
    else:
        expected: list[str] = []
        tail = _break_word(atom, width, expected, cell_len(atom))
        if tail:
            expected.append(tail)
        assert rows == expected


@pytest.mark.parametrize(
    "label", ["vendor/[model]/v1:exact", "vendor/model default:2026/[opaque]", "x" * 120]
)
def test_longest_opaque_literal_wins(label):
    rows = _wrap_notice_atoms(label, len(label), ("/model default", "/settings", label))
    assert rows == [label]
    narrow = _wrap_notice_atoms(label, 8, ("/model default", label))
    expected: list[str] = []
    tail = _break_word(label, 8, expected, cell_len(label))
    if tail:
        expected.append(tail)
    assert narrow == expected
    assert "".join(narrow) == label


def test_notice_authored_newlines_indent_and_copy_stay_intact():
    text = "\n  前/model default后\n\n    /settings尾\n"
    block = NoticeBlock(text, wrap_atoms=("/model default", "/settings"))
    rows = block._rows(18)
    assert rows == ["  前/model default", "  后", "", "    /settings尾"]
    assert block.text() == text

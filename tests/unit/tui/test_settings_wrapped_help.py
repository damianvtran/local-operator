"""The ``update.keep_generations`` help keeps its guarantee on screen (PR #4135).

QA rendered the Updates section at 80/100/120 cols in en and fr: the one-row
ladder cut the help before "… are always kept", the clause that makes keep=0
safe to type. The row now takes the same two-row WRAP rung the bridge rows
use (``settings_view._WRAPPED_HELP_KEYS``). The band stays two rows tall, so a
width where two rows cannot hold the sentence still gets the MARKED cut.

Driven through the real app and view, the way the bridge-row wrap tests in
``test_settings_bridge_f_state.py`` are.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.style import Style

from local_operator import settings_io
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.settings_view import SettingsView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

KEY = "update.keep_generations"


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from local_operator.tui.settings import settings_reload

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    settings_reload()
    yield tmp_path


def _select_key(view: SettingsView, key: str) -> None:
    view._repaint()  # rows rebuild on paint
    for index, row in enumerate(view._rows):
        if row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            return
    raise AssertionError(f"no {key} row — is the updates section visible?")


def _help() -> str:
    """The row's help as the view reads it: the registry's catalogue-backed copy.

    Never a literal — the copy is owned by i18n and was reworded during review,
    so every assertion below is stated against whatever the catalogue says.
    """
    return str(settings_io.BY_KEY[KEY].help)


async def _paint_at(
    size: tuple[int, int], *, value: int | None = None
) -> tuple[str, int, SettingsView]:
    """Paint the row at ``size``; ``value`` stores it first (off-default rows)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        if value is not None:
            settings_io.write_setting(view._manager, settings_io.BY_KEY[KEY], value)
        _select_key(view, KEY)
        return view._detail_text.plain, view._detail_width(), view


def _assert_marked_cut(plain: str, width: int, help_text: str) -> None:
    """One row, a visible ``…``, within the width, and GUARANTEE-FIRST.

    The cut keeps the help's own opening words, so whatever the copy leads with
    (the "always kept" guarantee, in the current wording) is what survives.
    """
    assert "\n" not in plain
    assert plain.endswith("…")
    assert cell_len(plain) <= width
    kept = plain[: -len("…")].rstrip()
    assert kept and help_text.startswith(kept), (kept, help_text)
    assert plain.split(" ", 1)[0] == help_text.split(" ", 1)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
async def test_the_keep_help_shows_whole_when_two_rows_hold_it(size: tuple[int, int]) -> None:
    help_text = _help()
    plain, width, view = await _paint_at(size)
    if cell_len(help_text) <= 2 * width:
        lines = plain.split("\n")
        assert " ".join(lines) == help_text, (plain, help_text)
        assert 1 <= len(lines) <= 2
        assert all(cell_len(line) <= width for line in lines)
        # A one-row fit needs no wrap; a two-row fit spends the band's top row.
        assert view._detail_two_line is (len(lines) == 2)
    else:
        # Copy longer than the band at this width: the marked cut, never a
        # third row (the band height is fixed).
        _assert_marked_cut(plain, width, help_text)
        assert view._detail_two_line is False


@pytest.mark.asyncio
async def test_off_default_and_too_narrow_cuts_the_help_never_shows_the_clause_alone() -> None:
    """Off its default at 56 cols the floor is the guarantee, not ``default: 2``.

    Replaces the default-value 56-col guard (review round 2, R2-3b), which
    passed on a3b90985 too: on default the old and new ladders share the same
    ``help_alone`` floor. Off default they differ — the generic clause ladder
    sheds the help for ``[clause, key]`` (``default: 2   update.keep_generations``
    fits 54 cells), dropping the "always kept" sentence entirely, while the
    listed-help ladder never goes below the help's own MARKED cut.
    """
    help_text = _help()
    plain, width, view = await _paint_at((56, 20), value=0)
    # The precondition that makes this the CUT case under any copy this row
    # has carried (189 and 138 cells, both > 2 x 54).
    assert cell_len(help_text) > 2 * width, (cell_len(help_text), width)
    assert KEY not in plain, "the clause+key rung must not replace the help"
    _assert_marked_cut(plain, width, help_text)
    assert view._detail_two_line is False
    assert view._detail.styles.padding.top == 1


@pytest.mark.asyncio
async def test_the_wrapped_keep_help_takes_dim_ink_not_faint() -> None:
    """Wrapped, the help is the band's only text: D5's ``dim``, never ``faint``.

    Review round 2, R2-2: the wrap rung painted the listed row's help in
    ``faint`` (1.97:1). Fails on 9924c5f0 (faint spans) and on a3b90985 (no
    two-row wrap at 80 cols).
    """
    help_text = _help()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select_key(view, KEY)
        width = view._detail_width()
        dim = theme_mod.semantic_color("dim")
        faint = theme_mod.semantic_color("faint")
        text = view._detail_text
    # Preconditions: the copy needs exactly the wrap at this width, and the
    # two inks differ in the active theme (else the test proves nothing).
    assert width < cell_len(help_text) <= 2 * width, (cell_len(help_text), width)
    assert dim != faint
    assert "\n" in text.plain and view._detail_two_line is True
    # Every painted span is the help (the band holds nothing else when it
    # wraps), so every span must carry the `dim` hex.
    expected = Style(color=dim).color
    inks = [
        (span.style if isinstance(span.style, Style) else Style.parse(span.style)).color
        for span in text.spans
    ]
    assert inks and all(ink == expected for ink in inks), (inks, expected)

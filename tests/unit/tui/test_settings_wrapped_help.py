"""The ``update.keep_generations`` help keeps its guarantee on screen (PR #4135).

QA rendered the Updates section at 80/100/120 cols in en and fr: the one-row
ladder cut the help before "… are always kept", the clause that makes keep=0
safe to type. The row now takes the same WRAP rung the bridge rows use
(``settings_view._WRAPPED_HELP_KEYS``). QA round 2, Q1: two rows still cut the
fr copy at 80/100 cols, so a listed row's band may grow to
``_DETAIL_WRAP_MAX_ROWS`` rows; past that the MARKED cut is the floor.

Driven through the real app and view, the way the bridge-row wrap tests in
``test_settings_bridge_f_state.py`` are.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.style import Style

from local_operator import settings_io
from local_operator.i18n import catalogues
from local_operator.i18n import text as i18n_text
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import settings_view as settings_view_mod
from local_operator.tui.widgets.settings_view import (
    _DETAIL_WRAP_MAX_ROWS,
    SettingsView,
    _wrap_cells,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory

KEY = "update.keep_generations"
MSG = "wire.settings.update_keep_generations.help"


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from local_operator.tui.settings import settings_reload

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_I18N", raising=False)  # the kill switch pins en
    settings_reload()
    try:
        yield tmp_path
    finally:
        # The locale tests move the PROCESS locale; never leak it.
        i18n_text.set_active_locale("en")


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
    size: tuple[int, int], *, value: int | None = None, locale: str | None = None
) -> tuple[str, int, SettingsView]:
    """Paint the row at ``size``; ``value`` stores it first (off-default rows).

    ``locale`` moves the process locale AFTER boot (the app resolves its own
    at mount), which re-materialises the registry help in place.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        if locale is not None:
            i18n_text.set_active_locale(locale)
            assert i18n_text.active_locale() == locale
        if value is not None:
            settings_io.write_setting(view._manager, settings_io.BY_KEY[KEY], value)
        _select_key(view, KEY)
        return view._detail_text.plain, view._detail_width(), view


def _catalogue_help(locale: str) -> str:
    """The keep help as the ``locale`` catalogue ships it — never a literal."""
    return str(catalogues.load_catalogue(locale, "wire.settings")[MSG])


def _longest_help_locale() -> str:
    """The locale whose keep help is widest in CELLS, across every catalogue."""
    widths = {
        locale: cell_len(_catalogue_help(locale))
        for locale in catalogues.locales()
        if "wire.settings" in catalogues.namespaces(locale)
        and MSG in catalogues.load_catalogue(locale, "wire.settings")
    }
    assert len(widths) >= 2, widths
    return max(widths, key=lambda locale: widths[locale])


def _assert_whole(plain: str, width: int, help_text: str) -> list[str]:
    """Every word of ``help_text`` on screen, in order, within the 4-row cap."""
    lines = plain.split("\n")
    assert "…" not in plain, (plain, help_text)
    assert " ".join(lines) == help_text, (plain, help_text)
    assert 1 <= len(lines) <= _DETAIL_WRAP_MAX_ROWS, lines
    assert all(cell_len(line) <= width for line in lines), lines
    return lines


#: A neighbour in the same section whose help is NOT listed, so selecting it
#: hands the band back to the stylesheet's two rows.
NEIGHBOUR = "update.auto_prune"


async def _layout_at(size: tuple[int, int], *, locale: str) -> dict[str, dict[str, Any]]:
    """Settle the layout on the keep row in ``locale``, then on its neighbour.

    Returns the geometry the band test compares: the rows the view BUDGETED
    for the band and the list (``_detail_rows``, ``_body_rows``) against the
    rows the LAYOUT gave them, in both states, plus the screen's scrollbar.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        i18n_text.set_active_locale(locale)

        async def settle(key: str) -> dict[str, Any]:
            _select_key(view, key)
            await pilot.pause()
            await pilot.pause()
            return {
                "plain": view._detail_text.plain,
                "width": view._detail_width(),
                "band": view._detail_rows,
                "laid_band": view._detail.outer_size.height,
                "laid_body": view._body.size.height,
                "derived_body": view._body_rows(),
                "scrollbar": bool(app.screen.show_vertical_scrollbar),
            }

        grown = await settle(KEY)
        plain = await settle(NEIGHBOUR)
        return {"grown": grown, "plain": plain}


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
    if _wrap_cells(help_text, width, max_lines=_DETAIL_WRAP_MAX_ROWS) is not None:
        lines = _assert_whole(plain, width, help_text)
        # A one-row fit needs no wrap; a multi-row fit spends the band's top row.
        assert view._detail_two_line is (len(lines) >= 2)
    else:
        # Copy longer than the capped band at this width: the marked cut,
        # never a fifth row.
        _assert_marked_cut(plain, width, help_text)
        assert view._detail_two_line is False


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30)])
async def test_the_fr_keep_help_shows_whole_past_two_rows(size: tuple[int, int]) -> None:
    """QA round 2, Q1: fr was cut to "…celles acti…" (80) / "…enregi…" (100).

    Fails on b39c7004, whose band held two rows: the 196-cell fr help needs
    three at both widths, so it fell to the one-row marked cut and lost the
    "Toujours conservées : …" guarantee.
    """
    help_text = _catalogue_help("fr")
    plain, width, view = await _paint_at(size, locale="fr")
    # The view paints the catalogue's copy, and it is the >2-row case.
    assert str(settings_io.BY_KEY[KEY].help) == help_text
    assert cell_len(help_text) > 2 * width, (cell_len(help_text), width)
    lines = _assert_whole(plain, width, help_text)
    assert len(lines) > 2
    assert view._detail_two_line is True
    assert view._detail_rows == len(lines)


@pytest.mark.asyncio
async def test_the_longest_locale_keep_help_fits_the_cap_or_cuts_guarantee_first() -> None:
    """The WIDEST catalogue copy, computed now, so a reworded locale is covered."""
    locale = _longest_help_locale()
    help_text = _catalogue_help(locale)
    plain, width, view = await _paint_at((80, 24), locale=locale)
    if _wrap_cells(help_text, width, max_lines=_DETAIL_WRAP_MAX_ROWS) is not None:
        lines = _assert_whole(plain, width, help_text)
        assert view._detail_rows == max(len(lines), 2)
    else:
        # Too long (or unsplittable) even for four rows: the one-row cut, and
        # it keeps that locale's OPENING words — its guarantee clause.
        _assert_marked_cut(plain, width, help_text)
        assert view._detail_rows == 2


@pytest.mark.asyncio
async def test_the_grown_band_is_reserved_so_nothing_overflows() -> None:
    """The rows the band grows by come out of the list, not off the screen.

    The layout's detail height must equal the rows the view budgeted, the list
    must be laid out exactly as tall as `_body_rows` derives (so the cursor
    parking and pane budget agree with the screen), and the screen grows no
    scrollbar.
    """
    help_text = _catalogue_help("fr")
    layout = await _layout_at((80, 24), locale="fr")
    grown, plain = layout["grown"], layout["plain"]
    lines = _assert_whole(grown["plain"], grown["width"], help_text)
    extra = len(lines) - 2
    assert extra > 0 and grown["band"] == len(lines)
    # The layout gave the band exactly the rows the view budgeted, both ways.
    assert grown["laid_band"] == grown["band"]
    assert plain["band"] == plain["laid_band"] == 2
    # The list paid for them row for row — on screen AND in the derived budget
    # the cursor parking and pane use — rather than the page overflowing.
    assert plain["laid_body"] - grown["laid_body"] == extra
    assert plain["derived_body"] - grown["derived_body"] == extra
    assert grown["scrollbar"] is False and plain["scrollbar"] is False


@pytest.mark.asyncio
async def test_en_keep_help_is_unchanged_at_120_cols() -> None:
    """en at 120 cols: the same two-row wrap as b39c7004, band not grown."""
    help_text = _catalogue_help("en")
    plain, width, view = await _paint_at((120, 30), locale="en")
    lines = _assert_whole(plain, width, help_text)
    assert len(lines) == 2
    assert view._detail_two_line is True
    assert view._detail_rows == 2
    assert view._detail.styles.padding.top == 0


@pytest.mark.asyncio
async def test_a_help_longer_than_the_row_cap_still_takes_the_marked_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past `_DETAIL_WRAP_MAX_ROWS` rows the floor is the one-row MARKED cut.

    The registry entry is frozen, so the copy is lengthened at the view's paint
    boundary (`settings_view._paint`) — only for this row's help.
    """
    original = str(settings_io.BY_KEY[KEY].help)
    long_help = " ".join([original] * 4)
    real_paint = settings_view_mod._paint

    def paint(text: str, **kwargs: object) -> str:
        if text == original:
            return long_help
        return real_paint(text, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(settings_view_mod, "_paint", paint)
    plain, width, view = await _paint_at((80, 24))
    assert _wrap_cells(long_help, width, max_lines=_DETAIL_WRAP_MAX_ROWS) is None
    _assert_marked_cut(plain, width, long_help)
    assert view._detail_two_line is False
    assert view._detail_rows == 2


@pytest.mark.asyncio
async def test_off_default_and_narrow_keeps_the_help_never_shows_the_clause_alone() -> None:
    """Off its default at 56 cols the floor is the guarantee, not ``default: 2``.

    Replaces the default-value 56-col guard (review round 2, R2-3b), which
    passed on a3b90985 too: on default the old and new ladders share the same
    ``help_alone`` floor. Off default they differ — the generic clause ladder
    sheds the help for ``[clause, key]`` (``default: 2   update.keep_generations``
    fits 54 cells), dropping the "always kept" sentence entirely, while the
    listed-help ladder never goes below the help's own MARKED cut.

    QA round 2, Q1 lets the band grow to four rows, so at 54 cells the en help
    (3 rows) now WRAPS whole instead of cutting; whichever floor applies, the
    clause+key rung never replaces it.
    """
    help_text = _help()
    plain, width, view = await _paint_at((56, 20), value=0)
    # Precondition: the help cannot share one row with anything at this width.
    assert cell_len(help_text) > 2 * width, (cell_len(help_text), width)
    assert KEY not in plain, "the clause+key rung must not replace the help"
    if _wrap_cells(help_text, width, max_lines=_DETAIL_WRAP_MAX_ROWS) is not None:
        # D1 (design round 1): the default note rides the last wrapped line.
        lines = _assert_whole(plain.removesuffix(" · default: 2"), width, help_text)
        assert view._detail_rows == len(lines)
        assert view._detail.styles.padding.top == 0
    else:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 30), (56, 20)])
async def test_off_default_wrapped_help_names_the_default_when_it_fits(
    size: tuple[int, int],
) -> None:
    """Design round 1, D1: wrapped off-default help still says ``default: 2``.

    The wrap rung rendered only the help, so the clause every other
    off-default row shows was dropped even with room on the last line. The
    help stays whole; the note joins its last line only when it fits.
    """
    help_text = _help()
    plain, width, view = await _paint_at(size, value=0)
    tail = " · default: 2"
    lines = _wrap_cells(help_text, width, max_lines=_DETAIL_WRAP_MAX_ROWS)
    # Preconditions: the help wraps, and the note fits beside its last line.
    assert lines is not None and len(lines) > 1, (lines, width)
    assert cell_len(lines[-1] + tail) <= width, (lines[-1], width)
    assert plain.endswith(tail), plain
    rendered = _assert_whole(plain.removesuffix(tail), width, help_text)
    assert len(rendered) == len(lines) == view._detail_rows
    assert all(cell_len(line) <= width for line in plain.split("\n"))

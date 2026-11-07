"""The keys card's footer hint behaves identically in every locale.

``KeysLegend._hint_row`` filters the scroll cue (a card that does not scroll
must not advertise scrolling) and sheds segments as the card narrows. It used
to decide both by comparing the painted segment to its ENGLISH string, which
matched only in English: in every other locale a non-scrolling card still
painted the cue, and nothing was dropped, so the row was ellipsis-cropped
mid-word ("… /ke…") where English kept ``↑↓ scroll · esc close``.

The expectations below are derived from the generated key functions through
the catalogue, never from literals, so they hold for any locale (and any
future copy edit) and a text-keyed implementation fails them.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text
from textual.geometry import Region

from local_operator.i18n.resolve import SUPPORTED_LOCALES
from local_operator.i18n.text import set_active_locale
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import keys_legend
from local_operator.tui.widgets.keys_legend import KeysLegend
from tests.unit.tui.test_app_pilot import FakeSession, _factory

WIDTHS = (30, 40, 60, 80)
SEP = " · "


def _hint(monkeypatch: pytest.MonkeyPatch, width: int, scrolled: bool) -> str:
    """The painted hint row at a content ``width`` — no app, no stylesheet.

    ``_hint_row`` reads only ``_content_width``, so a bare instance with that
    one measurement pinned is the whole real path (the pilot tests cover the
    stylesheet-sized card in English).
    """
    monkeypatch.setattr(KeysLegend, "_content_width", lambda self: width)
    return KeysLegend.__new__(KeysLegend)._hint_row(scrolled).plain


def _ladder(scrolled: bool) -> list[list[str]]:
    """The expected states, widest first.

    Shed `?` first (the only conditionally-true segment), then `/keys`, then —
    only when the card scrolls — the cue, which OUTLIVES the reopen route
    because it is the one in-card sign that more is below. ``esc close`` is
    never shed. Built from the ``_HINT_SEGMENTS`` order (scroll, close,
    reopen, typing) independently of the implementation's own bookkeeping.
    """
    scroll, close, reopen, typing = keys_legend._HINT_SEGMENTS
    head = [scroll] if scrolled else []
    states = [
        [*head, close, reopen, typing],
        [*head, close, reopen],
        [*head, close],
    ]
    return [*states, [close]] if scrolled else states


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
@pytest.mark.parametrize("scrolled", [True, False])
@pytest.mark.parametrize("width", [*WIDTHS, 120])
def test_the_hint_row_sheds_segments_in_english_order_in_every_locale(
    monkeypatch: pytest.MonkeyPatch, locale: str, scrolled: bool, width: int
) -> None:
    set_active_locale(locale)
    scroll, close = keys_legend._HINT_SEGMENTS[:2]
    states = _ladder(scrolled)
    expected = next((state for state in states if cell_len(SEP.join(state)) <= width), states[-1])

    painted = _hint(monkeypatch, width, scrolled)

    assert painted == SEP.join(expected), (locale, width, scrolled)
    # The discriminating negatives, stated separately so a failure names them:
    if not scrolled:
        assert scroll not in painted, "a card that does not scroll advertised scrolling"
    assert close in painted, "esc-close must never be shed"
    assert "…" not in painted, "the ladder sheds whole segments, never crops mid-word"


def test_english_keeps_its_documented_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Literal pins for the English contract the role-keyed ladder must not move."""
    assert (
        _hint(monkeypatch, 120, True) == "↑↓ scroll · esc close · /keys reopens · ? when not typing"
    )
    assert _hint(monkeypatch, 120, False) == "esc close · /keys reopens · ? when not typing"
    assert _hint(monkeypatch, 40, True) == "↑↓ scroll · esc close · /keys reopens"
    assert _hint(monkeypatch, 30, True) == "↑↓ scroll · esc close"
    assert _hint(monkeypatch, 30, False) == "esc close · /keys reopens"


# -- the card's height accounting, painted by the real app in every locale --------
#
# Design round 1, D1. The hint tests above pin the hint ROW; they cannot see a
# hint row the card never gets to paint. A translated section heading can be
# wider than the content box at the narrow contract (fr 26 / es 25 cells against
# 22 at 30 columns), and a heading left unwrapped soft-wraps into a row that the
# pinned `height = len(rows) + gutter` did not count -- clipping the pinned hint
# row, the card's only exit cue, while the card is visibly cut off. Only the
# painted card shows that, so these drive the real ``OperatorApp`` (production
# stylesheet) and read the strips Textual actually renders.

#: Screen columns around the narrow-card contract (30) where fr breaks below 34
#: and es below 33; 26 is below every locale's heading width.
NARROW_COLUMNS = tuple(range(26, 35))

#: The two heights the card renders at: 30 keeps the vertical gutter, 10 spends
#: it (``KEYS_SQUEEZE_ROWS``) -- the two arms of ``_fit``.
CARD_SCREEN_ROWS = (30, 10)


async def _painted_card(app, pilot, locale: str) -> tuple[list[str], list[Text], KeysLegend]:
    """Open the card in ``locale`` through the real ``/language`` + ``/keys`` path.

    Returns ``(content lines the strips paint, the rows the card composed, the
    widget)``. The padding rows are trimmed by the same rule ``_fit`` applies,
    so what is left is exactly what sits inside the pinned height.
    """
    app._apply_language(locale)
    await pilot.pause()
    app._run_slash_command("/keys")
    legend = app.query_one(KeysLegend)
    for _ in range(20):
        await pilot.pause()
        if legend.is_open:
            break
    assert legend.is_open, locale
    strips = legend.render_lines(Region(0, 0, legend.outer_size.width, legend.outer_size.height))
    lines = [strip.text.strip() for strip in strips]
    pad = 0 if legend.has_class("-squeezed") else 1
    return lines[pad : len(lines) - pad], legend._compose_rows(), legend


@pytest.mark.asyncio
@pytest.mark.parametrize("screen_rows", CARD_SCREEN_ROWS)
@pytest.mark.parametrize("columns", NARROW_COLUMNS)
async def test_the_painted_card_counts_every_wrapped_row_and_keeps_the_exit_cue(
    columns: int, screen_rows: int
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    cue_in_english = False
    problems: list[str] = []
    async with app.run_test(size=(columns, screen_rows)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        # English first: the baseline is MEASURED on the same screen, so the
        # expectation for the other locales is "no worse than English here"
        # rather than a literal that a copy edit could invalidate.
        for locale in ("en", *(code for code in SUPPORTED_LOCALES if code != "en")):
            painted, composed, legend = await _painted_card(app, pilot, locale)
            close = keys_legend._HINT_SEGMENTS[1]
            has_cue = close in painted[-1]
            if locale == "en":
                cue_in_english = has_cue
            if cue_in_english and not has_cue:
                problems.append(f"{locale}: no exit cue, last rows {painted[-2:]}")

            # The row count: composing N rows and painting more than N is the
            # under-count itself. Each composed row is re-wrapped independently
            # by Rich to the content width; the painted height must equal that
            # sum, and (since every row is pre-wrapped) the sum must equal N.
            console = Console(width=legend.size.width, force_terminal=True, color_system=None)
            wrapped = sum(len(row.wrap(console, legend.size.width)) for row in composed)
            if wrapped != len(composed) or len(painted) != wrapped:
                problems.append(
                    f"{locale}: composed {len(composed)} rows, "
                    f"wrap to {wrapped}, painted {len(painted)}"
                )
            legend.close()
            await pilot.pause()
    # One assertion over every locale so a failure names them all, not the first.
    assert not problems, (columns, screen_rows, problems)

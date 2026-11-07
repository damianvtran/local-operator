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

from local_operator.i18n.resolve import SUPPORTED_LOCALES
from local_operator.i18n.text import set_active_locale
from local_operator.tui.widgets import keys_legend
from local_operator.tui.widgets.keys_legend import KeysLegend

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

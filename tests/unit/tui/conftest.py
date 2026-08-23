"""Hermetic environment and shared harnesses for the TUI suite.

Snapshot frames were captured with colour enabled and shimmer off; a caller
that exports NO_COLOR (a common developer default) would otherwise fail all
three snapshots for reasons that have nothing to do with the code under
test. The pins live in fixtures — scoped and reverted — instead of module
import time, so collection order never leaks environment into other suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.transcript import TranscriptView

#: The real stylesheet, so styled tests exercise the shipped rules rather
#: than a convenient approximation of them.
TCSS_PATH = str(Path(theme_mod.__file__).parent / "local_operator.tcss")


@pytest.fixture(autouse=True)
def hermetic_tui_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    # The splash starts a one-shot PyPI probe on mount. Unit tests must not
    # pay a 5 s timeout (or a real GET) for news they are not asserting.
    # Patch the worker, not ``check_latest``: ``/update`` needs the real
    # function so its own mocks can drive newer/same/error.
    monkeypatch.setattr(
        "local_operator.tui.app.OperatorApp._check_for_update",
        lambda self: None,
    )
    # The caret used to be pinned here too: `TextArea.cursor_blink` was patched
    # off for the whole suite because a blinking caret made whether a captured
    # frame contained one a coin flip, and the boot snapshot failed against a
    # file it had just regenerated. The product now ships a solid caret (see
    # `Editor.__init__`), so the pin is gone: a fixture that forces the
    # behaviour under test would make the editor's own caret tests vacuous, and
    # the strobe it hid from this suite was exactly the one users were seeing.


class StyledTranscriptApp(App[None]):
    """A transcript under the REAL sheet and the real brand variables.

    Widget-level assertions answer "does this build the right content"; this
    app answers "does the shipped CSS then turn that content into the rows
    and colours we claim", which is a different question — and the one that
    catches height, spacing, and hover regressions that unit tests cannot
    see. Nothing else is mounted, so no other rule can interfere.
    """

    CSS_PATH = TCSS_PATH

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables

    def compose(self) -> ComposeResult:
        yield TranscriptView()


#: The composer's prompt marker. The pickers repeat the glyph — deliberately,
#: so a highlighted row sits in the prompt's own column — but they mount BELOW
#: the input row, so the FIRST match down the frame is always the composer's.
PROMPT_CHEVRON = "❯"


def composer_cells(app: App[None]) -> list[tuple[str, str | None, str | None]]:
    """(text, fg hex, bg hex) for every segment of the composer's row.

    Shared because the composer's focus state is not a widget attribute: a
    caret is a cell whose colours have been swapped, and the chevron's ink is
    a colour the stylesheet resolved. Both are only answerable from what the
    terminal was SENT, which is what ``render_strips`` returns.

    Located by the chevron rather than by the copy so the same reader works in
    every mode the composer has words for — ``Read-only — press esc to reply``
    and ``Aside — esc returns to the chat`` are composer rows too.

    The FIRST match is the composer's even with a picker open. Both pickers
    repeat the glyph, deliberately in the prompt's own column, but ``compose``
    yields them AFTER ``#input-row`` in ``#input-shell``'s normal flow, so they
    can only render below it. The ``/resume`` picker is a pushed Screen, so
    while it is up the composer is genuinely off the frame and the raise is the
    honest answer rather than a missed row.
    """
    for strip in app.screen._compositor.render_strips():
        if PROMPT_CHEVRON not in strip.text:
            continue
        cells = []
        for segment in strip._segments:
            style = segment.style
            fg = style.color.get_truecolor().hex.lower() if style and style.color else None
            bg = style.bgcolor.get_truecolor().hex.lower() if style and style.bgcolor else None
            cells.append((segment.text, fg, bg))
        return cells
    raise AssertionError("the composer row is not on the frame at all")


def caret_cells(cells: list[tuple[str, str | None, str | None]]) -> list[str]:
    """What the caret is sitting ON: cells drawn with its inverted ground.

    The TEXT is returned rather than a count because "is there a caret" and
    "is the caret eating a letter" are the two questions this app has got
    wrong, and only the second one needs the content.
    """
    caret_ground = theme_mod.semantic_color("fg").lower()
    return [text for text, _, bg in cells if bg == caret_ground]


def chevron_colour(cells: list[tuple[str, str | None, str | None]]) -> str | None:
    """The prompt marker's ink — `fg` while the composer has focus, else `dim`.

    NOT the accent (D5): green means a turn is live, and a marker that is lit
    in nearly every frame cannot also mean that. Focus is a brightness step in
    the same neutral ramp, which is a 3.86x luminance move against `dim` where
    the accent was 2.15x.
    """
    return next(fg for text, fg, _ in cells if PROMPT_CHEVRON in text)

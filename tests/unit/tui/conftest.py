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
from textual.widgets import TextArea

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
    # The editor's caret BLINKS on a wall-clock timer, so whether a captured frame
    # contains it depends on when the capture happened to land. That made the boot
    # snapshot fail intermittently against a file it had just regenerated — a
    # 50/50 coin flip dressed up as a regression. Pinned rather than tolerated:
    # this suite asserts layout and colour, and a caret phase is neither.
    monkeypatch.setattr(TextArea, "cursor_blink", False, raising=False)


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

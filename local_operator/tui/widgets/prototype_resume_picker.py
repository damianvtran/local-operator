"""PROTOTYPE — THROWAWAY. The /resume picker variant harness.

Three variants of the /resume picker, switchable via --variant and ctrl+t,
against the real read-only session store.

See ``~/workspace/PROPOSAL-resume-picker.md``. This module owns the app, the
variant switcher and the state bar; the variants themselves live in
``prototype_resume_variants`` and the disk reads in ``prototype_resume_data``.

The state bar is deliberately NOT part of any design under test: it reports
which variant is showing and HOW MANY ROWS IT DREW, because the defect this
prototype exists to attack is that today's picker draws 10 rows out of 140 at
every terminal height — without a live count every variant merely "looks bigger".
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static

import local_operator.tui
from local_operator.tui import theme as theme_mod
from local_operator.resume import SessionRow
from local_operator.tui.widgets.prototype_resume_data import PreviewData


class _StubScreen(Screen[None]):
    """Placeholder so the harness runs before the variants land."""

    def __init__(
        self,
        rows: list[SessionRow],
        now: float,
        digests: dict[str, str],
        data: PreviewData,
    ) -> None:
        super().__init__()
        self._rows = rows

    def compose(self) -> ComposeResult:
        names = "\n".join(f"  {r.name}  {r.id}" for r in self._rows[:20])
        yield Static(f"stub variant — {len(self._rows)} rows\n\n{names}")

    def visible_rows(self) -> tuple[int, int]:
        return min(20, len(self._rows)), len(self._rows)


try:
    # importlib, not a static `from ... import`: Slice 2 owns this module and it
    # does not exist until that branch lands, so the harness must run without it.
    # Seam type is dict[str, tuple[str, type[Screen]]]; the class is typed Any
    # because every variant Screen takes the seam's four-argument constructor,
    # which is not Screen.__init__'s signature.
    VARIANTS: dict[str, tuple[str, Any]] = importlib.import_module(
        "local_operator.tui.widgets.prototype_resume_variants"
    ).VARIANTS
except ImportError:  # Slice 2 not landed yet
    VARIANTS = {
        "A": ("stub", _StubScreen),
        "B": ("stub", _StubScreen),
        "C": ("stub", _StubScreen),
    }


class PrototypeResumeApp(App[None]):
    """Hosts one variant at a time under the production stylesheet."""

    # A `Path` here raises TypeError inside save_capture, which does
    # `[app.CSS_PATH] if isinstance(app.CSS_PATH, str) else app.CSS_PATH`.
    CSS_PATH = str(Path(local_operator.tui.__file__).parent / "local_operator.tcss")

    # Chords only: printable keys must reach the variant's filter.
    BINDINGS = [
        ("ctrl+t", "cycle_variant", "cycle variant"),
        ("escape", "quit", "quit"),
    ]

    def __init__(self, data: PreviewData, start_variant: str, query: str) -> None:
        super().__init__()
        self._data = data
        self._variant = start_variant
        self._query = query
        self._now = time.time()
        self._rows = data.rows()
        self._digests = data.digests()
        self._bar: Static | None = None

    def get_css_variables(self) -> dict[str, str]:
        # Without the theme token map every `$lo-*` in the production sheet is
        # undefined and the frame renders unstyled.
        base = super().get_css_variables()
        base.update(theme_mod.tcss_variable_map(theme_mod.current_theme()))
        return base

    def on_mount(self) -> None:
        self.show_variant(self._variant)

    def show_variant(self, key: str) -> None:
        self._variant = key
        _label, cls = VARIANTS[key]
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.push_screen(cls(self._rows, self._now, self._digests, self._data))
        self.call_after_refresh(self._mount_bar)

    def _mount_bar(self) -> None:
        # The bar is mounted INTO the variant's screen, not the app's default
        # one: a pushed Screen covers the whole app area, so an app-level dock
        # renders underneath the design and is never seen.
        bar = Static("", id="prototype-state-bar")
        bar.styles.dock = "bottom"
        bar.styles.height = 1
        bar.styles.background = "#5f00af"
        bar.styles.color = "#ffffff"
        self._bar = bar
        self.screen.mount(bar)
        self.call_after_refresh(self._refresh_bar)

    def action_cycle_variant(self) -> None:
        keys = list(VARIANTS)
        self.show_variant(keys[(keys.index(self._variant) + 1) % len(keys)])

    def _refresh_bar(self) -> None:
        if self._bar is None:
            return
        label = VARIANTS[self._variant][0]
        screen: Any = self.screen
        drawn, total = (0, len(self._rows))
        if callable(getattr(screen, "visible_rows", None)):
            drawn, total = screen.visible_rows()
        cols, rows = self.size
        self._bar.update(
            f" {self._variant} — {label} · ctrl+t cycles · "
            f"{drawn} rows shown / {total} · {cols}×{rows}"
        )

    def on_resize(self) -> None:
        self.call_after_refresh(self._refresh_bar)

    def on_key(self) -> None:
        self.call_after_refresh(self._refresh_bar)


def main(store: Path, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="PROTOTYPE /resume picker variants")
    parser.add_argument("--variant", choices=["A", "B", "C"], default="A")
    parser.add_argument("--query", default="", help="preseed the filter")
    args = parser.parse_args(argv)
    PrototypeResumeApp(PreviewData(store), args.variant, args.query).run()
    return 0

"""Count the per-keypress repaint work on ``/settings`` — the guard's engine.

WHY THIS EXISTS
===============

``docs/settings-keypress-profile.md`` measured that an arrow press on
``/settings`` re-rasterises the ENTIRE settings list (96 lines at 96 rows, 218
at 218) in order to repaint a 14-line viewport in which two lines changed, and
that the cost is linear in row count at ~30 µs/row/press. A ~60-row Hotkeys
section is queued behind that finding, so the waste has to be measured as a
COUNT rather than as a duration: AGENTS.md ("Prefer a structural invariant to a
numeric one", "Calibrate ceilings from CI, never from your laptop") is explicit
that no portable wall-clock bound survives the CI/laptop core-speed spread, and
the profiler declined to propose one for exactly that reason.

So this module counts four operations per press and nothing else. It is shared
by the regression test (``tests/unit/tui/test_settings_repaint_cost.py``) and by
ad-hoc runs of this file, so the guard and the evidence in the PR cannot drift
apart by being two different probes.

WHAT IS COUNTED, AND WHY EACH ONE
---------------------------------

``semantic_color``   theme lookups. Five ``Style`` objects were constructed per
                     ROW per paint, before the row's kind was even examined, so
                     a header row paid for the accent/faint styles it never
                     uses. Linear in row count.
``row_text``         calls to ``SettingsView._row_text``. One per row per paint
                     means every off-screen row is composed to repaint a
                     viewport.
``strip_lines``      LINES handed back by ``Visual.to_strips`` for the list
                     widget. This is the direct measurement of the dominant
                     cost: ``Static.update`` invalidates the whole render cache
                     and ``Widget._render_content`` rasterises ``self.size``,
                     i.e. all N lines. Counted at ``Visual.to_strips`` because
                     that is the ONE boundary both the pre-fix (whole-widget)
                     and post-fix (per-line) render paths cross, so the same
                     number means the same thing on both trees.
``build_rows``       calls to ``SettingsView._build_rows``, which re-derives the
                     whole row list on a pure cursor move.

EVERY PRESS SAMPLED MUST ACTUALLY MOVE THE CURSOR — see
``scripts/settings_press_driver``. ``/settings`` is AGENTS.md's documented
wrap-vs-clamp exception, and a clamped press repaints nothing, so a naive
``down``-press loop silently dilutes the sample with no-ops. That bug made the
first counterfactual in the profile read ~0% saving where the truth was 12-16%.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.settings_press_driver import Bouncer  # noqa: E402


@dataclass
class PressCounts:
    """Operations counted while ``presses`` guaranteed-moving presses ran."""

    presses: int
    rows: int
    viewport: int
    semantic_color: int
    row_text: int
    strip_lines: int
    build_rows: int

    def per_press(self, field: str) -> float:
        """A counted total divided by the presses that produced it."""
        return getattr(self, field) / self.presses if self.presses else 0.0

    def summary(self) -> str:
        return (
            f"rows={self.rows} viewport={self.viewport} presses={self.presses} | "
            f"semantic_color={self.per_press('semantic_color'):.1f}/press "
            f"row_text={self.per_press('row_text'):.1f}/press "
            f"strip_lines={self.per_press('strip_lines'):.1f}/press "
            f"build_rows={self.per_press('build_rows'):.1f}/press"
        )


class _Counter:
    """A call tally plus the ``wrap`` that produces a countable stand-in.

    ``wrap`` returns a plain FUNCTION, never this object. A callable instance
    assigned to a class attribute is not a descriptor, so ``view._row_text(...)``
    would hand the instance back unbound and the call would lose ``self`` —
    which is exactly the ``TypeError`` a first version of this probe raised.
    """

    def __init__(self) -> None:
        self.count = 0
        self.armed = False

    def wrap(self, target: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            result = target(*args, **kwargs)
            if self.armed:
                self.count += 1
            return result

        return counted


async def count_presses(
    pilot: Any,
    view: Any,
    presses: int,
    monkeypatch: Any,
) -> PressCounts:
    """Drive ``presses`` bounced presses and return what they cost, counted.

    ``monkeypatch`` is pytest's fixture (or anything exposing ``setattr``), so
    every patch is undone by the caller's teardown rather than by a ``finally``
    this function would have to get right. Counting is ARMED only for the
    sampled presses: opening the page and the pilot's first settle legitimately
    rasterise everything once, and folding that into the average would report a
    fixed start-up cost as a per-press cost.
    """
    from textual.visual import Visual

    from local_operator.tui import theme as theme_mod
    from local_operator.tui.widgets.settings_view import SettingsView

    list_widget = view._list

    semantic = _Counter()
    row_text = _Counter()
    build_rows = _Counter()
    strips = _Counter()

    inner_to_strips = Visual.to_strips.__func__  # type: ignore[attr-defined]

    # `cls` FIRST, then `widget`. `to_strips` is a classmethod, so a wrapper
    # written as `(widget, *args)` silently binds `widget` to the Visual CLASS
    # and every `widget is list_widget` test is False — which is a guard that
    # passes at 0.0 lines/press against code that rasterises 96 of them.
    def to_strips(cls: Any, widget: Any, *args: Any, **kwargs: Any) -> Any:
        # Attributed to the LIST widget only, and counted in LINES rather than
        # in calls: the title, rule, detail and pane rasterise on the same press
        # and would otherwise be charged to the cost this guard is about.
        result = inner_to_strips(cls, widget, *args, **kwargs)
        if strips.armed and widget is list_widget:
            strips.count += len(result)
        return result

    # `semantic_color` is patched on the MODULE the widget resolves through:
    # `settings_view` does `from ... import theme as theme_mod`, so it shares
    # the module object and one patch covers the list, detail, pane and chrome
    # painters alike — which is what makes the count a per-PRESS figure.
    monkeypatch.setattr(theme_mod, "semantic_color", semantic.wrap(theme_mod.semantic_color))
    monkeypatch.setattr(SettingsView, "_row_text", row_text.wrap(SettingsView._row_text))
    monkeypatch.setattr(SettingsView, "_build_rows", build_rows.wrap(SettingsView._build_rows))
    monkeypatch.setattr(Visual, "to_strips", classmethod(to_strips))

    bouncer = Bouncer(view)
    # One un-armed warm-up press so the sample never includes the first paint
    # after the patches went in.
    await pilot.press(bouncer.key())
    await pilot.pause()

    for counter in (semantic, row_text, build_rows, strips):
        counter.armed = True

    for _ in range(presses):
        await pilot.press(bouncer.key())
        await pilot.pause()

    for counter in (semantic, row_text, build_rows, strips):
        counter.armed = False

    return PressCounts(
        presses=presses,
        rows=len(view._rows),
        viewport=view._body.size.height,
        semantic_color=semantic.count,
        row_text=row_text.count,
        strip_lines=strips.count,
        build_rows=build_rows.count,
    )


async def open_settings(app: Any, pilot: Any) -> Any:
    """Open ``/settings`` on a booted app and hand back the view."""
    from local_operator.tui.widgets.settings_view import SettingsView

    app._open_settings_view()
    await pilot.pause()
    view = app.query_one(SettingsView)
    await pilot.pause()
    return view


def main() -> None:
    """Run the probe standalone and print the counts (PR evidence)."""
    import asyncio

    from scripts.visual_capture import isolate_capture

    isolate_capture()
    os.environ.setdefault("LOCAL_OPERATOR_CONFIG_DIR", os.environ["LOCAL_OPERATOR_CONFIG_DIR"])

    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    class _Patch:
        """The two-line `monkeypatch` this file needs outside pytest."""

        def __init__(self) -> None:
            self._undo: list[tuple[Any, str, Any]] = []

        def setattr(self, target: Any, name: str, value: Any) -> None:
            self._undo.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self) -> None:
            for target, name, old in reversed(self._undo):
                setattr(target, name, old)

    size = (100, 30)
    presses = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    async def run() -> None:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            view = await open_settings(app, pilot)
            patch = _Patch()
            try:
                counts = await count_presses(pilot, view, presses, patch)
            finally:
                patch.undo()
            print(counts.summary())

    asyncio.run(run())


if __name__ == "__main__":
    main()

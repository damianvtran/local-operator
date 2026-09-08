"""A keypress driver that never measures a CLAMPED (no-op) press.

WHY THIS EXISTS — a measurement bug this investigation hit and had to correct.

``/settings`` is the documented exception to the repo's wrap-vs-clamp rule
(AGENTS.md, "Wrapping vs clamping"): ``action_move`` CLAMPS at the ends of the
list. The page has 96 rows but only 79 SELECTABLE ones, so a probe that sends
100 ``down`` presses gets 79 real moves followed by 21 presses on which the
cursor does not move at all.

A clamped press is not a cheap press — it is a DIFFERENT press. Verified
directly: after clamping, ``_list_text`` is byte-identical before and after the
press, so a probe that elides updates on equal content elides EVERY update on
those presses, including the list. That made an elision counterfactual look
~3x faster than it is, purely because most of its samples were presses with
nothing to repaint.

So the driver below reverses direction one press before an end. Every sample is
a press that genuinely moves the cursor and genuinely changes two row lines,
which is the operation the operator's report is about. Direction reversal is
itself never sampled.

Use ``bounce_presses`` for pilot-driven runs and ``bounce_moves`` for direct
``action_move`` calls.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator


class Bouncer:
    """Yields +1/-1 steps that stay strictly inside the selectable range.

    ``view`` is a ``SettingsView``. The turn is decided from the CURRENT cursor
    position against the live selectable list, so it stays correct when the row
    count changes underneath it (an inflated registry, an expansion opening).
    """

    def __init__(self, view: Any, margin: int = 2) -> None:
        self._view = view
        # Turn `margin` rows before the true end, so no sampled press is the one
        # that lands exactly on the boundary either — that press moves, but the
        # NEXT one would not, and keeping a margin makes the sampled population
        # uniform rather than mostly-moving-with-occasional-stalls.
        self._margin = max(1, margin)
        self._delta = 1

    def step(self) -> int:
        indices = self._view._selectable()
        if not indices:
            return self._delta
        try:
            position = indices.index(self._view._selected)
        except ValueError:
            position = 0
        if position >= len(indices) - self._margin:
            self._delta = -1
        elif position <= self._margin - 1:
            self._delta = 1
        return self._delta

    def key(self) -> str:
        return "down" if self.step() == 1 else "up"


def bounce_moves(view: Any, count: int) -> Iterator[int]:
    """Yield ``count`` deltas, calling nothing — the caller times each move."""
    bouncer = Bouncer(view)
    for _ in range(count):
        yield bouncer.step()


async def bounce_presses(pilot: Any, view: Any, count: int) -> AsyncIterator[str]:
    """Yield the key to press for each of ``count`` guaranteed-moving presses."""
    bouncer = Bouncer(view)
    for _ in range(count):
        yield bouncer.key()

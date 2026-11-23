"""The LIVE hotkey snapshot for widgets that paint a chord (R1/Q1/Q4).

``App.set_keymap`` keeps the applied mapping on ``App._keymap``, and reading it
from a widget is the only way a painted hint can follow a remap. Two rules make
that read safe everywhere:

* **An unmounted widget has no app.** ``DOMNode.app`` resolves the active app
  from the asyncio context and raises ``NoActiveAppError`` when there is none —
  measured on ``TodoPanel`` composed by ``tests/unit/tui/test_band_panels.py``
  without an app, which is a test shape, not a defect in the panel. A hint is
  chrome: it degrades to the shipped default rather than failing the paint.
* **The mapping is read live, never cached.** A remap is applied in place; a
  widget that copied the map at mount would keep naming the old chord.

Consumers also implement ``keymap_hints_changed()``, which the app calls after
every ``set_keymap`` so paint-once caches (the dock header's equality guard,
the todo footer's ``Static``, the ask bar's width budget) are invalidated
in the same tick. Keep the method cheap and idempotent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.types import NoActiveAppError

from local_operator import keymap as _keymap


def live_keymap(widget: Any) -> Mapping[str, str] | None:
    """The app's applied ``id -> key`` map, or ``None`` when there is no app."""
    try:
        snapshot = getattr(widget.app, "_keymap", None)
    except NoActiveAppError:
        return None
    return snapshot if isinstance(snapshot, Mapping) else None


def live_key(widget: Any, action_id: str) -> str:
    """The key ``action_id`` answers to right now: the remap, else the default."""
    return _keymap.painted_key(action_id, live_keymap(widget))


def live_copy(widget: Any, text: str) -> str:
    """``text`` with every shipped app chord replaced by its live remap."""
    return _keymap.substitute_keys(text, live_keymap(widget))

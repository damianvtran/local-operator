"""Busy-send delivery mode — the read of ``delivery.busy_default``.

The setting is neutral (owned by no one surface) because the TUI and the
desktop app share one config.yml: either editor changes what the next send
in the other app does. ``steer`` is urgent (today's behaviour); ``queue``
never cancels a running tool or skips a batch call; ``both`` is ``queue``
until the send has waited long enough, then counts as urgent. The mode names
the ENGINE vocabulary (``delivery=steer|queue|both`` in ``session.py``) so
the setting, the wire and the queue cannot drift into three spellings of
one word.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

#: The engine's delivery vocabulary. ``steer`` is today's urgent behaviour.
BusyDeliveryMode = Literal["steer", "queue", "both"]

#: The shipped default (design D6): a busy send behaves exactly as it always
#: has until the operator changes the setting. Restated as a LITERAL by the
#: registry in ``settings_io`` (which stays off this module's import path);
#: the test in ``tests/unit/test_settings_io.py`` pins the two together.
DEFAULT_BUSY_DELIVERY: BusyDeliveryMode = "steer"

_MODES: tuple[str, ...] = ("steer", "queue", "both")


def busy_delivery_mode(values: Mapping[str, Any]) -> BusyDeliveryMode:
    """The configured mode from a config ``values`` mapping, ``steer`` if unset.

    A hand-edited ``busy_delivery: "sometimes"`` reads as the default rather
    than reaching the engine as an unknown value the wire would refuse.
    """
    section = values.get("delivery")
    value = section.get("busy_default") if isinstance(section, Mapping) else None
    return value if value in _MODES else DEFAULT_BUSY_DELIVERY

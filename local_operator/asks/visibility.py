"""The ONE reader of ``asks.lapsed_visible_for``: how long a lapsed ask stays shown.

WHY A SEPARATE MODULE (decision D8). ``asks/store.py`` must stay stdlib-only
(``tests/unit/test_import_graph.py`` pins it), and reading a setting means the
config watcher, which pulls in YAML and the config defaults. So the store's
readers take the window as a plain ``lapsed_visible_ms`` int, and every caller
gets that int from here: the live projection (``frontend_state.ask_wire`` and
the queue's timer), the phone relay's ``/api/asks`` and stopped-conversation
projection, the desktop ``/v1/desktop/asks`` route, and the TUI fleet list.

WHERE THE VALUE COMES FROM. ``config_watch.process_watcher(dir).values`` — the
process's last-good snapshot of ``config.yml`` — rather than a fresh parse per
read. The runtime, the phone daemon and the TUI each start that watcher on
their loop, so its snapshot follows another process's write within one poll.
A process with no started watcher (the desktop server) asks for a fresh poll
through ``poll_now=True``; that call fans out to listeners, so it is made ON
the loop thread only (the ``desktop_team_bridge`` status route's pattern),
never from a worker thread.

An unknown or hand-mangled value reads as the default (7 d, today's
behaviour) rather than raising into a surface: hiding a question because of a
typo would be worse than showing it for the shipped window.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from local_operator.asks import store

#: The key's nested path in ``config.yml`` (``settings_io`` registers the same).
LAPSED_VISIBLE_FOR_PATH = ("asks", "lapsed_visible_for")


def lapsed_visible_ms_from_values(values: Mapping[str, Any]) -> int:
    """The window in ms from a ``config.yml`` ``values`` mapping."""
    node: Any = values
    for step in LAPSED_VISIBLE_FOR_PATH:
        if not isinstance(node, Mapping) or step not in node:
            return store.DEFAULT_LAPSED_VISIBLE_MS
        node = node[step]
    seconds = store.LAPSED_VISIBLE_FOR_S.get(node) if isinstance(node, str) else None
    if seconds is None:
        return store.DEFAULT_LAPSED_VISIBLE_MS
    return seconds * 1000


def lapsed_visible_ms(config_dir: Path | str | None = None, *, poll_now: bool = False) -> int:
    """The configured ``asks.lapsed_visible_for`` window, in ms.

    ``config_dir`` defaults to ``paths.config_dir()``. ``poll_now`` re-stats
    the file first, for a process whose watcher is not started; call it with
    ``True`` only on the event-loop thread (module docstring). Never raises: a
    watcher that cannot be built reads as the default window.
    """
    try:
        from local_operator.config_watch import process_watcher

        watcher = process_watcher(Path(config_dir) if config_dir is not None else None)
        if poll_now:
            watcher.poll_now()
        return lapsed_visible_ms_from_values(watcher.values)
    except Exception:  # noqa: BLE001 — a display preference must never break a surface
        return store.DEFAULT_LAPSED_VISIBLE_MS


__all__ = [
    "LAPSED_VISIBLE_FOR_PATH",
    "lapsed_visible_ms",
    "lapsed_visible_ms_from_values",
]

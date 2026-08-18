"""TUI display settings — a thin read-only view of ``config_dir()/config.yml``.

Display flags (``display.*`` keys in ``values``) are read lazily and cached
per process; a missing or unreadable config never breaks the TUI — every
lookup falls back to its default. Writes stay in the CLI's ``config edit``
surface; the TUI only reads.
"""

from __future__ import annotations

from typing import Any

#: Display flags and their defaults. Tests may poke ``_cache`` directly.
_DEFAULTS: dict[str, Any] = {
    "display.shimmer": True,
    # Nerd Font glyphs on tool rows. Defaults ON because the audience runs
    # patched fonts, and OFF is one `config edit` (or the env kill switch in
    # `tui/glyphs.py`) away for a terminal that would draw them as boxes.
    "display.nerd_icons": True,
    # The OSC 0 window/tab title carrying the session name and run state
    # (`tui/terminal_title.py`). Defaults ON: a terminal without OSC 0 ignores
    # the sequence entirely, and the title is saved on start and restored on
    # exit, so the worst case for an unsupported terminal is no change at all.
    "display.terminal_title": True,
    # Desktop notifications for the two moments a user who is looking elsewhere
    # needs to know about: the parent agent finished, or it is waiting on them
    # (`tui/notify.py`). Defaults ON, and only ever fires while the terminal is
    # UNFOCUSED, so a user watching the session is never interrupted; the env
    # kill switch is `LOCAL_OPERATOR_NO_NOTIFICATIONS`.
    "display.notifications": True,
}

_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    """Read ``values`` once; any failure yields pure defaults."""
    values: dict[str, Any] = {}
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        manager = ConfigManager(config_dir())
        values = dict(manager.get_config().values)
    except Exception:
        values = {}
    return {key: values.get(key, default) for key, default in _DEFAULTS.items()}


def settings_get(key: str, default: Any = None) -> Any:
    """Return the display setting ``key`` (e.g. ``display.shimmer``)."""
    global _cache
    if _cache is None:
        _cache = _load()
    if key not in _cache:
        return default
    return _cache[key]


def settings_reload() -> None:
    """Drop the cache so the next lookup re-reads the config file."""
    global _cache
    _cache = None

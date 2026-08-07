"""TUI display settings — a thin read-only view of ``~/.local-operator/config.yml``.

Display flags (``display.*`` keys in ``values``) are read lazily and cached
per process; a missing or unreadable config never breaks the TUI — every
lookup falls back to its default. Writes stay in the CLI's ``config set``
surface; the TUI only reads.
"""

from __future__ import annotations

from typing import Any

#: Display flags and their defaults. Tests may poke ``_cache`` directly.
_DEFAULTS: dict[str, Any] = {
    "display.shimmer": True,
}

_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    """Read ``values`` once; any failure yields pure defaults."""
    values: dict[str, Any] = {}
    try:
        from pathlib import Path

        from local_operator.config import ConfigManager

        manager = ConfigManager(Path.home() / ".local-operator")
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

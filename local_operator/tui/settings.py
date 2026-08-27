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
    # Nerd Font glyphs on tool rows. Default is None = AUTO: unset means
    # `tui/glyphs.py` decides from the terminal-emulator env markers (a
    # bundled Nerd symbol fallback font is enumerable per emulator), so a
    # bare Terminal.app gets plain icons and ghostty/kitty/wezterm get the
    # expanded set with zero user setup. An EXPLICIT bool in config overrides
    # both ways: True forces glyphs on for a user who installed a patched
    # font in an otherwise-unknown terminal, False forces them off. The
    # None-vs-bool distinction IS the tri-state — `settings_get` returns None
    # only when the key is absent from `values`, which is what "auto" reads.
    "display.nerd_icons": None,
    # The OSC 0 window/tab title carrying the session name and run state
    # (`tui/terminal_title.py`). Defaults ON: a terminal without OSC 0 ignores
    # the sequence entirely, and the title is saved on start and restored on
    # exit, so the worst case for an unsupported terminal is no change at all.
    "display.terminal_title": True,
    # Inline images on the transcript (tool-result screenshots, pasted
    # attachments — `tui/images.py`). Defaults ON: the supported terminals
    # degrade to half-cell pixels or a one-line receipt on their own, so OFF
    # exists for the user who wants a text-only ledger, not for compatibility.
    # The env override is `LOCAL_OPERATOR_IMAGES` (kitty|halfcell|text|off).
    "display.images": True,
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

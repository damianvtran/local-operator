"""TUI display flags — the cached fast-path reader for ``config.yml``.

Display flags (``display.*`` keys in ``values``) are read lazily and cached
per process; a missing or unreadable config never breaks the TUI — every
lookup falls back to its default.

This module READS. Writes go through ``local_operator.settings_io``, which the
``/settings`` page and ``lop config edit`` both drive, and which calls
:func:`settings_reload` afterwards — so a write that skipped it would leave the
running TUI painting the old flag and the change would look lost until
relaunch.

There is now a SECOND invalidator: the TUI's config-watch listener calls
:func:`settings_reload` when another process changes a ``display.*`` key. That
distinction matters to :func:`_load`, not just as trivia — the first
invalidator only ever fires after a write this process just made, so the file
was well-formed by construction, while the second fires on bytes the user may
be part-way through hand-editing.

The keys here are LITERAL dotted top-level keys: ``values["display.shimmer"]``,
not ``values["display"]["shimmer"]``. See ``settings_io`` for why that
distinction is load-bearing.

Defaults are derived from the schema registry rather than restated, so the page
and this reader cannot disagree about what "unset" means. The import is
function-local: this module sits on the shimmer/glyph/image fast path and is
imported during TUI paint, so it must stay cheap to import — pulling the
registry in at module scope would put it on every startup.
"""

from __future__ import annotations

from typing import Any

#: Documentation of the flags this module serves, kept as prose beside the
#: derivation below. The VALUES come from ``settings_io`` (see
#: :func:`_defaults`); this dict is not consulted at runtime.
_DEFAULT_NOTES: dict[str, Any] = {
    "display.shimmer": True,
    # One padding row above and below a tool row and a user prompt
    # (`.comfortable-rows` in the stylesheet). Default ON was changed to OFF
    # by the maintainer.
    "display.comfortable_rows": False,
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


def _defaults() -> dict[str, Any]:
    """The display-flag defaults, from the schema registry.

    Falls back to :data:`_DEFAULT_NOTES` if the registry cannot be imported.
    Not defensiveness for its own sake: this module is on the paint path and a
    display flag failing to resolve would take down a frame, where reading a
    stale-but-correct default merely means the TUI looks the way it shipped.
    """
    try:
        from local_operator.settings_io import display_defaults

        return display_defaults()
    except Exception:  # pragma: no cover - the registry is a plain data module
        return dict(_DEFAULT_NOTES)


def _load() -> dict[str, Any]:
    """Read ``values`` once; any failure yields pure defaults.

    Prefers the config WATCHER's already-validated snapshot over constructing a
    ``ConfigManager`` (review round 4, B3). The manager's constructor runs
    ``_load_config``, which MOVES a malformed ``config.yml`` aside to
    ``config.yml.bad.<ts>`` and continues from defaults — and the ``except``
    below cannot catch that, because from Python's point of view the move-aside
    succeeded. Destroying the user's settings from a paint path is not a
    trade-off this module gets to make.

    The exposure is new and specific: this cache used to be invalidated only by
    ``settings_io._store``, a write THIS process had just performed, so the
    bytes were well-formed by construction. The config watcher now invalidates
    it on ANOTHER process's write, which is exactly the case where the file on
    disk is arbitrary — the user's next hand-edit may be mid-save, truncated or
    mis-indented, and the watcher deliberately holds such a file in silence.

    ``existing_watcher`` rather than ``process_watcher``, matching
    ``settings_io._notify_watcher``: a CLI process that never started a watcher
    must not build one from a paint path, and falls back to the manager — which
    is safe there precisely because no watcher means no cross-process
    invalidation to race with.
    """
    values: dict[str, Any] = {}
    try:
        from local_operator.config_watch import existing_watcher
        from local_operator.paths import config_dir

        directory = config_dir()
        watcher = existing_watcher(directory)
        if watcher is not None:
            values = dict(watcher.values)
        else:
            from local_operator.config import ConfigManager

            values = dict(ConfigManager(directory).get_config().values)
    except Exception:
        values = {}
    return {key: values.get(key, default) for key, default in _defaults().items()}


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

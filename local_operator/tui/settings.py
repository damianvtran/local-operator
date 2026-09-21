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
    # Mid-turn narration — the prose a model call streams before it finalizes
    # into tool calls. Defaults ON, which is the shipped behaviour: narration
    # is rendered by the same block as the final answer, so it stays.
    #
    # OFF removes that block at finalize, leaving `user -> tools -> answer`, so
    # the user can tell an answer from thinking. It is removal AT FINALIZE, not
    # never-show: while deltas arrive nothing knows whether this call ends in
    # tool calls or in the answer, so narration streams live and is dropped one
    # event later.
    #
    # This flag decides whether narration is THERE at all; `display.rail`
    # separately decides whether it wears the rail. Both were set from the same
    # classification of the event that ended the message, so a narration block
    # that survives this setting is un-railed (the rail marks the ANSWER).
    #
    # A mid-session flip applies FORWARD ONLY — already-mounted blocks are left
    # exactly as they are. Re-projecting the transcript to apply it backwards
    # means mounting many blocks in one synchronous pass, which leaves them
    # unarranged and paints a blank frame; a display toggle must never risk
    # blanking a transcript. The session history keeps the narration either
    # way, so `/resume` re-reads it under the current value.
    "display.narration": True,
    # The model's own private reasoning, streamed live while it thinks, removed
    # WHOLE when the phase ends. Default OFF: the flag buys a live view of the
    # model working rather than a record, and a live view that is on by default
    # spends rows on every model call of every turn for a reader who never asked
    # to watch — the operator reported the accumulated frames as pollution.
    #
    # OFF is also the value under which the live transcript and the RESUMED one
    # agree exactly, which is the argument the `/settings` help string no longer
    # has room for: reasoning is never durable, so there is nothing to re-show,
    # and a reader who opens `/resume` sees the same transcript they left. That
    # help string is deliberately ONE sentence of 87 characters because the field
    # paints 93 at 100 columns and elides the rest without wrapping (design
    # review round 1, D1) — do not grow it back into the argument.
    #
    # Collapsing is not what this flag turns off: a finished phase leaves
    # NOTHING behind, header row included (`retire` closes the block and the app
    # removes it). A mid-session flip is FORWARD ONLY, for the same reason
    # `display.narration`'s is: re-projecting mounted blocks in one synchronous
    # pass leaves them unarranged and paints a blank frame.
    "display.reasoning": False,
    # A rule down the left edge of the assistant's ANSWER, in the `label`
    # token. It ECHOES the user prompt's rule rather than matching it: same
    # column and same role, but deliberately a different glyph and a different
    # ink (a quarter block in `label` against the prompt's half block in
    # `signal`), because a rail that looked identical to the prompt's would
    # remove the distinction it exists to draw.
    #
    # ANSWER means the TERMINAL message of the turn — precisely the finalized
    # message `tui/narration.py::is_intermediate_narration` returns False for.
    # A mid-turn progress sentence is painted by the same block, and railing it
    # too is what made progress and outcome look identical (the report this
    # change answers): the prose stays, because dropping it is
    # `display.narration`'s separate job, and the mark goes.
    #
    # The rail is a SETTLED-state property. A message still streaming paints no
    # bar and folds to the lane's full width — rail-OFF geometry exactly — and
    # the bar appears in the same paint that commits the message
    # (`AssistantBlock.finalize_text`). Narration is never railed, streaming or
    # settled, because the rail marks the ANSWER rather than every assistant
    # block; both routes are read from the one gate (`_rail_cols`), so the fold,
    # the paint, the copy gutter and the selection slice cannot disagree about a
    # frame. The cost of the streaming state is that the settle re-folds the
    # message two cells narrower, so a long message re-wraps once — see
    # `_rail_cols` for why the alternative (reserving the cells blank) was
    # rejected.
    #
    # Default ON. The rail answers "where does the answer start and stop",
    # which only bites a reader who cannot already tell — so the people it
    # helps are exactly the people who would never go looking for the setting.
    # The asymmetry decides it: a user who dislikes the rail sees a line and
    # turns it off, while a user who needs it under a default of OFF never
    # discovers it exists. The cost of the wrong default is recoverable in one
    # direction and invisible in the other.
    #
    # OFF restores pre-rail rendering exactly for a lane of `MIN_BODY` or
    # wider, which is every ordinary terminal. Below that the pre-rail build
    # had no floor and this one clamps for containment, so the two differ by
    # design (see `AssistantBlock._body_width`). It is not merely an unpainted
    # gutter either way: the fold width, the copy gutter and the selection
    # slice are all read at the same rate as the paint, so the prose is not
    # left indented two cells by a rail that is not there. An UN-RAILED
    # narration block, and a message that has not SETTLED yet, are that same
    # state, reached from inside the block rather than from this setting
    # (`mark_narration`, and the flag `finalize_text` raises). A mid-session
    # flip DOES reach blocks already on screen — `display.*` runs `retheme`,
    # which re-enters `_apply_rows`, which is where the rail is painted and
    # where the flag is read.
    "display.rail": True,
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
    # The literal `###` before a heading. rich STRIPS these when it parses a
    # heading, so restoring them is a deliberate act, not a leak of raw source.
    #
    # Defaults OFF, and that is a decision the colour ramp earned: the marker
    # was introduced when h1-h6 resolved to two greys and level could not be
    # read from ink at all. With the ramp spending a hue per level, ink and
    # weight alone give six mutually distinct levels in all 54 themes, so the
    # marker is no longer load-bearing and would be permanent chrome on every
    # heading of every answer.
    #
    # It stays available because it is the only channel that survives a
    # colourless terminal, `NO_COLOR`, and colour-vision deficiency — for
    # those readers it is the difference between six levels and one.
    "display.heading_markers": False,
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
    # Whether a toast is TITLED with the conversation's own name. Defaults ON,
    # which is what every notification here has always done. It earns a flag of
    # its own because the observer path (a BACKGROUND session finishing while
    # you are looking at a different one) widens where a model-written session
    # name appears — including on a lock screen, where macOS repeats banner
    # titles. Governs EVERY notification leg, not only the observer's: the
    # attached session's own toasts and a detached runtime's gate toasts reach
    # the same lock screen, and a flag that covered only some of them would
    # make its own copy false (review round 1, M2). Off falls back to the brand
    # name, which still says a session finished without saying which work.
    "display.notification_session_name": True,
    # The dock subagent panel's density when a session STARTS: `full` (one
    # row per child), `summary` (one row of counts) or `hidden`. Defaults to
    # full, today's shape. It seeds the panel and nothing more — `ctrl+g`
    # cycles from it and never writes it back, because a setting that pinned
    # the density would make the key a no-op again, the exact defect #525
    # fixed. Read by `widgets/subagent_panel.py` on the first non-empty
    # roster and on a live edit (applied only if the user has not cycled it
    # this session). Unknown strings read as full.
    "display.dock": "full",
    "display.time_format": "12h",
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

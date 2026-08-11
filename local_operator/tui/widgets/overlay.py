"""Geometry shared by the floating overlay cards (``/usage``, ``/btw``).

Every card on the ``toast`` CSS layer has the same three problems, and they are
problems a stylesheet cannot solve:

- **Layers do not size.** A widget owns its whole region and Textual BLANKS all
  of it, so a stretched host on an overlay layer erases the transcript either
  side of the card. Hosts are therefore ``width: auto`` and the card is placed
  by a Python-set offset, which rules out ``align: center middle`` — that needs
  the stretched host.
- **The input panel is DOCKED.** Centring against the screen puts the lower half
  of a tall card on top of the prompt. The rows a card may use, and the rows it
  is centred within, are the ground ABOVE whatever is docked at the bottom.
- **Absolute and relative boxes must be reconciled once.** ``region`` is
  absolute while ``screen.size`` is the content box that ``Screen { padding: 1 }``
  insets; subtracting one from the other is how a centred card ends up a cell
  off-centre.

These were :class:`~local_operator.tui.widgets.usage_panel.UsagePanel`'s private
methods until the aside card needed the identical answers. They are functions
over a widget rather than a base class because the two cards share nothing else
— one scrolls a table, the other tails a conversation — and a base class would
have invited them to.
"""

from __future__ import annotations

from textual.app import NoScreen
from textual.screen import Screen
from textual.widget import Widget

#: Fallback dimensions for a widget that is not on a screen yet (tests that
#: build a card standalone). Big enough to render something legible, small
#: enough that nothing sized against it looks deliberately chosen.
FALLBACK_SCREEN = (80, 24)


def screen_size(widget: Widget) -> tuple[int, int]:
    """The screen's CONTENT box, floored so arithmetic downstream stays sane."""
    try:
        size = widget.screen.size
    except NoScreen:
        return FALLBACK_SCREEN
    return max(20, size.width), max(8, size.height)


def rows_above_dock(widget: Widget) -> int:
    """Rows the card may occupy: the ground above the docked input panel.

    NOT "the screen height less a constant". How many rows the prompt takes is
    a function of the editor's line count, the subagent/todo band and the boot
    layout — five to ten across the sizes the tests sweep — so a constant put
    the card on top of the prompt at every size once its content was long
    enough (D19).

    Read off whatever is DOCKED rather than off an id: the invariant is "the
    card covers no docked surface", and a rule naming ``#input-dock`` would go
    quietly back to overlapping if the dock were renamed. A host with nothing
    docked (the widget-only test app) gets the whole content box, which is the
    same answer by the same rule.
    """
    try:
        screen = widget.screen
    except NoScreen:
        return screen_size(widget)[1]
    content = screen.content_region
    ceiling = content.bottom
    for sibling in screen.children:
        if sibling.display and sibling.styles.dock == "bottom":
            ceiling = min(ceiling, sibling.region.y)
    return max(1, ceiling - content.y)


def composer_column(widget: Widget) -> tuple[int, int]:
    """``(x offset inside the content box, width)`` of the visible input panel.

    For a card that belongs WITH the composer rather than over it. Matching the
    composer's own left and right edges is what makes the two read as one
    stacked column instead of a floating dialog and an unrelated dock — and the
    aside's input IS that composer, so the geometry has to say so.

    Resolved off ``#input-shell`` (the widget carrying the fill, which the boot
    layout clamps to a centred card) rather than off the full-width positioner
    it sits in, so the card tracks the panel a user actually sees. Falls back to
    the docked child, then to the whole content box, so a rename or a host with
    no composer at all (the widget-only test app) degrades to the honest answer
    instead of raising.
    """
    try:
        screen = widget.screen
    except NoScreen:
        return 0, FALLBACK_SCREEN[0]
    content = screen.content_region
    panel = None
    try:
        panel = screen.query_one("#input-shell")
    except Exception:  # noqa: BLE001 — no composer on this host; fall through
        for sibling in screen.children:
            if sibling.display and sibling.styles.dock == "bottom":
                panel = sibling
                break
    if panel is None or not panel.region.width:
        return 0, max(1, content.width)
    # Clamped to the CONTENT BOX. Mid-resize the screen has already taken its
    # new size while the dock has not re-arranged, so the panel's region is a
    # frame behind — and a card painted to a stale 118 on a terminal that just
    # became 60 overhangs the frame with its prose clipped mid-word. The clamp
    # is wrong by at most the dock's own inset for the one frame before the
    # re-measure lands, which is invisible; the overhang was not.
    x = max(0, panel.region.x - content.x)
    return x, max(1, min(panel.region.width, content.width - x))


def stack_on_dock(widget: Widget, width: int, height: int, x: int, gap: int = 0) -> None:
    """Rest the card on the dock at column ``x``, ``gap`` rows above it.

    The placement for a card that is part of the input column. ``x`` comes from
    :func:`composer_column` rather than being centred, and the vertical anchor
    is the dock rather than the middle of the ground, so the card grows UPWARD
    as its content does — a centred card moves half a row per line gained and
    drags the text being read with it.
    """
    parent = widget.parent
    if parent is None or isinstance(parent, Screen):
        return
    parent.styles.offset = (x, max(0, rows_above_dock(widget) - height - gap))


def recentre(widget: Widget, width: int, height: int) -> None:
    """Offset the card's HOST so the card centres in the ground above the dock.

    The host, never the widget: a card mounted straight onto the screen would
    otherwise shift the whole app sideways, which is a louder bug than an
    off-centre card — so that case is left alone.

    Horizontally the symmetric screen inset cancels. Vertically there is no
    such cancellation, which is why the height term is
    :func:`rows_above_dock` and not the screen height.
    """
    parent = widget.parent
    if parent is None or isinstance(parent, Screen):
        return
    parent.styles.offset = (
        max(0, (screen_size(widget)[0] - width) // 2),
        max(0, (rows_above_dock(widget) - height) // 2),
    )


def anchor_bottom(widget: Widget, width: int, height: int, gap: int = 1) -> None:
    """Rest the card ``gap`` rows above the dock, horizontally centred.

    The alternative placement to :func:`recentre`, for a card whose content
    GROWS while it is being read. Centred, every new row moves the card half a
    row up the screen, so a streaming answer drags everything already on it
    upward under the reader's eyes. Anchored to the bottom, growth extends
    away from the composer and the newest text stays exactly where it was —
    directly above the input the user is typing the next question into, which
    is also the shape every chat surface already has.
    """
    parent = widget.parent
    if parent is None or isinstance(parent, Screen):
        return
    parent.styles.offset = (
        max(0, (screen_size(widget)[0] - width) // 2),
        max(0, rows_above_dock(widget) - height - gap),
    )

"""Keep the narrow session drawer above the dock on the first painted frame."""

from __future__ import annotations

from dataclasses import replace

from textual.containers import Horizontal
from textual.geometry import Size
from textual.layout import DockArrangeResult


class SessionWorkspace(Horizontal):
    def arrange(self, size: Size, optimal: bool = False) -> DockArrangeResult:
        arrangement = super().arrange(size, optimal)
        sidebar = next(
            (item for item in arrangement.placements if item.widget.id == "session-sidebar"), None
        )
        conversation = next(
            (item for item in arrangement.placements if item.widget.id == "session-conversation"),
            None,
        )
        if sidebar is None or conversation is None or not self.has_class("sidebar-overlay"):
            return arrangement
        # A region read from the preceding frame is too late: a restored draft
        # or gate can change dock height without a terminal resize. Ask the
        # conversation's ordinary cached arrangement for THIS frame instead.
        # The compositor will reuse that same result when it descends; there is
        # no second stylesheet, measurement loop, or post-paint correction.
        gutter = conversation.widget.styles.gutter
        inner = conversation.region.shrink(gutter)
        children = conversation.widget.arrange(inner.size, optimal)
        dock = next((item for item in children.placements if item.widget.id == "input-dock"), None)
        if dock is None:
            return arrangement
        ceiling = conversation.region.y + gutter.top + dock.region.y
        region = sidebar.region
        clipped = sidebar._replace(region=region._replace(height=max(0, ceiling - region.y)))
        # Do not mutate Textual's cached arrangement or its lazy spatial map.
        return replace(
            arrangement,
            placements=[clipped if item is sidebar else item for item in arrangement.placements],
            _spatial_map=None,
        )

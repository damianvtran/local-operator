"""Sidebar preferences and compatibility exports for the shared session catalog.

Keep the import surface stable for TUI extensions; discovery and attention now
belong to the session layer so desktop and terminal never classify independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from local_operator.session.catalog import (  # noqa: F401 -- compatibility API
    CATALOG_SCAN_LIMIT,
    SUBAGENT_LAYER_CAP,
    CatalogEntry,
    cached_session_rows,
    decorate_rows,
    load_catalog,
    rank_entries,
    session_directory_name,
    subagent_population,
)

DEFAULT_SIDEBAR_VISIBLE = False
DEFAULT_SIDEBAR_POSITION = "left"
DEFAULT_SIDEBAR_SHOW_SUBAGENTS = False
SidebarPosition = Literal["left", "right"]


@dataclass(frozen=True)
class SidebarSettings:
    visible: bool = DEFAULT_SIDEBAR_VISIBLE
    position: SidebarPosition = "left"
    #: Appended last so positional construction — `SidebarSettings(False,
    #: "left")`, which `scripts/sidebar_shot.py` uses — keeps working.
    show_subagents: bool = DEFAULT_SIDEBAR_SHOW_SUBAGENTS

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> SidebarSettings:
        section = values.get("tui")
        section = section if isinstance(section, Mapping) else {}
        visible = section.get("sidebar_visible", DEFAULT_SIDEBAR_VISIBLE)
        position = section.get("sidebar_position", DEFAULT_SIDEBAR_POSITION)
        show_subagents = section.get("sidebar_show_subagents", DEFAULT_SIDEBAR_SHOW_SUBAGENTS)
        return cls(
            visible=visible if isinstance(visible, bool) else DEFAULT_SIDEBAR_VISIBLE,
            position="right" if position == "right" else "left",
            show_subagents=(
                show_subagents
                if isinstance(show_subagents, bool)
                else DEFAULT_SIDEBAR_SHOW_SUBAGENTS
            ),
        )

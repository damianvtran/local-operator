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
    CatalogEntry,
    cached_session_rows,
    decorate_rows,
    load_catalog,
    rank_entries,
    session_directory_name,
)

DEFAULT_SIDEBAR_VISIBLE = False
DEFAULT_SIDEBAR_POSITION = "left"
SidebarPosition = Literal["left", "right"]


@dataclass(frozen=True)
class SidebarSettings:
    visible: bool = DEFAULT_SIDEBAR_VISIBLE
    position: SidebarPosition = "left"

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> SidebarSettings:
        section = values.get("tui")
        section = section if isinstance(section, Mapping) else {}
        visible = section.get("sidebar_visible", DEFAULT_SIDEBAR_VISIBLE)
        position = section.get("sidebar_position", DEFAULT_SIDEBAR_POSITION)
        return cls(
            visible=visible if isinstance(visible, bool) else DEFAULT_SIDEBAR_VISIBLE,
            position="right" if position == "right" else "left",
        )

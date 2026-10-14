"""The Hub links overlay widget: pinned chrome, a scrolling body, the key hints.

The presentation (sections, the reserved status slot, policy text) lives in
:mod:`local_operator.tui.hub_links` as pure functions and is rendered here. The
widget owns only what it must: the rows handed to it, the scroll offset, the
focused row, and whether a refresh is in flight.

It follows :class:`~local_operator.tui.widgets.usage_panel.UsagePanel` on
purpose — mounted once and hidden, opened by a keystroke, never taking a row
from the transcript. A poll that changes a link's state repaints BODY ROWS
inside the reserved slot; it never changes the row COUNT, so a focused target
cannot move under the user.
"""

from __future__ import annotations

from typing import Any, Mapping

from rich.text import Text
from textual.message import Message
from textual.widgets import Static

from local_operator.tui import hub_links as core
from local_operator.tui.widgets import overlay

#: Chrome rows: title, rule, a blank, the key-hint row.
CHROME_ROWS = 4
CARD_PADDING_ROWS = 2
MAX_BODY_ROWS = 16


class HubLinksPanel(Static):
    """The overlay itself. Opened by ``/hub``; ``r`` refreshes, Esc closes."""

    can_focus = True

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("r", "refresh", "Refresh"),
        ("up", "move(-1)", "Up"),
        ("down", "move(1)", "Down"),
        ("enter", "activate", "Open policy"),
    ]

    class Dismissed(Message):
        pass

    class RefreshRequested(Message):
        pass

    class PolicyRequested(Message):
        """The focused row's source-scoped policy menu was asked for."""

        def __init__(self, link_id: str | None, local_id: str, direction: str) -> None:
            super().__init__()
            self.link_id = link_id
            self.local_id = local_id
            self.direction = direction

    def __init__(self) -> None:
        super().__init__(id="hub-links-panel")
        self._rows: list[core.LinkRow] = []
        self._offset = 0
        self._selected = 0
        self._loading = False
        self._refreshing = False
        self._error = ""

    # -- state ---------------------------------------------------------------

    def start_fetch(self) -> None:
        """Open the panel and mark a fetch in flight (kept rows, not a skeleton)."""
        self._loading = True
        self._refreshing = False
        self._error = ""
        self.display = True
        self._repaint()

    def show_links(self, snapshot: Mapping[str, Any]) -> None:
        """Install a finished inventory; keep the reader's place across a refresh."""
        self._rows = core.rows_from_snapshot(snapshot)
        self._loading = False
        self._refreshing = False
        self._error = ""
        self.display = True
        self._repaint()

    def show_error(self, message: str) -> None:
        self._loading = False
        self._refreshing = False
        self._error = message
        self.display = True
        self._repaint()

    def close(self) -> None:
        self._loading = False
        self._refreshing = False
        self._offset = 0
        self.display = False

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    @property
    def selected_row(self) -> core.LinkRow | None:
        ordered = self._visible_rows()
        if 0 <= self._selected < len(ordered):
            return ordered[self._selected]
        return None

    # -- actions -------------------------------------------------------------

    def action_dismiss(self) -> None:
        self.close()
        self.post_message(self.Dismissed())

    def action_refresh(self) -> None:
        self.post_message(self.RefreshRequested())

    def action_move(self, delta: int) -> None:
        count = len(self._visible_rows())
        if count == 0:
            return
        self._selected = max(0, min(count - 1, self._selected + delta))
        self._repaint()

    def action_activate(self) -> None:
        """The reserved slot's Enter: open the focused link's source-scoped menu.

        A healthy mark opens the policy menu; an available mark applies; a
        retryable failure retries its exact source/direction. The APP decides
        which (it owns the service); this widget reports the intent.
        """
        row = self.selected_row
        if row is None:
            return
        direction = {
            "available": "pull",
            "failure": "pull" if row.pull_state == "failed" else "push",
        }.get(row.slot_state, "policy")
        self.post_message(self.PolicyRequested(row.link_id, row.local_id, direction))

    # -- paint ---------------------------------------------------------------

    def _visible_rows(self) -> list[core.LinkRow]:
        return list(self._rows)

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        width = max(24, overlay.screen_size(self)[0] - 8)
        if self._error:
            body: list[Text] = [Text(self._error, style="dim")]
        else:
            body = core.render_rows(
                self._rows, width, selected=self._selected, loading=self._loading
            )
        lines = [self._title()]
        lines.append(Text("─" * min(width, 60), style="dim"))
        lines.append(Text(""))
        lines.extend(body[:MAX_BODY_ROWS])
        lines.append(Text(""))
        lines.append(Text("r refresh · Esc close · Enter opens policy", style="dim"))
        out = Text()
        for index, row in enumerate(lines):
            if index:
                out.append("\n")
            out.append_text(row)
        self.styles.width = width
        self.styles.height = len(lines) + CARD_PADDING_ROWS
        self.update(out)

    def _title(self) -> Text:
        mark = " (refreshing…)" if self._refreshing else ""
        return Text(f"Hub links{mark}", style="bold")

    def on_mount(self) -> None:
        if self.display:
            self._repaint()

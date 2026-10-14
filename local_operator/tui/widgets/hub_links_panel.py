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

from local_operator.i18n.keys import tui_hub_links as _k
from local_operator.i18n.text import tr
from local_operator.tui import hub_links as core
from local_operator.tui.widgets import overlay

_t = tr("tui.hub_links")

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
        # Mounted once and kept (the `/usage` rule), so it must START hidden:
        # `display` defaults True, and a visible 0-row host in the toast layer
        # still nudged the transcript one cell left before any fetch. `close()`
        # restores exactly this state.
        self.display = False

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

    def _body_window(self, body_len: int) -> int:
        """The first body row to paint so the SELECTED row stays visible.

        The body carries a header per non-empty section, so the selected LINK's
        painted row is its index plus the headers above it. The window scrolls to
        keep that row inside the :data:`MAX_BODY_ROWS` cap (the architect's third
        bug: rows past the cap were unreachable).
        """

        if body_len <= MAX_BODY_ROWS:
            return 0
        painted = self._painted_index()
        start = max(0, painted - MAX_BODY_ROWS // 2)
        return min(start, max(0, body_len - MAX_BODY_ROWS))

    def _painted_index(self) -> int:
        """The body row the SELECTED link paints on (headers included).

        Mirrors ``core.render_rows``' own ordering — sections in
        :func:`core.section_order`, headers only for non-empty sections, rows
        sorted by name — so the window and the paint cannot disagree about where
        the selection sits.
        """

        by_kind: dict[str, list[core.LinkRow]] = {}
        for row in self._rows:
            by_kind.setdefault(row.kind, []).append(row)
        painted = 0
        for kind in core.section_order():
            group = by_kind.get(kind, [])
            if not group:
                continue
            painted += 1  # the section header
            for row in sorted(group, key=lambda r: r.name.casefold()):
                if row is self.selected_row:
                    return painted
                painted += 1
        return max(0, painted - 1)

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        width = max(24, overlay.screen_size(self)[0] - 8)
        if self._error:
            body: list[Text] = [Text(self._error, style="dim")]
        else:
            body = core.render_rows(
                self._rows,
                width,
                selected=self._selected,
                loading=self._loading,
                tail_style=self._tail_style(),
            )
        # WINDOW the body around the selection, so a link past the visible rows
        # is reachable: the old `body[:MAX_BODY_ROWS]` sliced from the top and
        # ignored `_offset`, so rows past the cap could never be shown or
        # activated (the architect's third bug). `_offset` is a BODY-row index;
        # it is clamped so the selection stays inside the window.
        window = self._body_window(len(body))
        visible = body[window : window + MAX_BODY_ROWS]
        lines = [self._title()]
        lines.append(Text("─" * min(width, 60), style="dim"))
        lines.append(Text(""))
        lines.extend(visible)
        lines.append(Text(""))
        lines.append(Text(_t(_k.policy_saving()), style="dim") if self._refreshing else Text(""))
        out = Text()
        for index, row in enumerate(lines):
            if index:
                out.append("\n")
            out.append_text(row)
        self.styles.width = width
        self.styles.height = len(lines) + CARD_PADDING_ROWS
        self.update(out)

    def _title(self) -> Text:
        title = _t(_k.detail_title())
        mark = f" ({_t(_k.policy_saving())})" if self._refreshing else ""
        return Text(f"{title}{mark}", style="bold")

    def _tail_style(self) -> Any:
        """The resolved ink for the source+policy tail: theme ``muted`` (D4).

        Falls back to ``dim`` if the theme is unavailable, so a stripped harness
        still renders rather than raising ``MissingStyle``.
        """

        from local_operator.tui import theme as theme_mod

        try:
            return theme_mod.semantic_color("muted")
        except Exception:  # noqa: BLE001 - a card must not fail to paint
            return "dim"

    def on_mount(self) -> None:
        if self.display:
            self._repaint()

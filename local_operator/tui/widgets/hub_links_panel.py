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
#: The card's horizontal padding, matching the CSS `padding: 1 2` (2 cells each
#: side). Rows are composed to ``panel_width - PANEL_PADDING_CELLS`` so the
#: composed width and the padded box are the SAME number (round-2 D8).
PANEL_PADDING_CELLS = 4


def _entry_lines(entry: Text) -> int:
    """How many physical lines one body entry paints (1, or 2 when wrapped)."""

    return len(entry.split("\n")) or 1


def _entry_lines_text(entry: Text) -> list[Text]:
    """The physical lines of one body entry."""

    return entry.split("\n") or [Text("")]


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
        # The two source-scoped policy toggles (round-2 D7). Independent, on the
        # FOCUSED link only: `s` flips Subscribe, `a` flips Auto-push. Letters
        # because the composer is not in the focus chain while the panel holds
        # it (the esc/q/r precedent), and they are not composer keys. Enter
        # stays the route for the eligibility-gated / consent-requiring cases
        # the toggle cannot complete alone.
        ("s", "toggle_subscribe", "Toggle Subscribe"),
        ("a", "toggle_auto_push", "Toggle Auto-push"),
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

    class PolicyToggleRequested(Message):
        """Toggle ONE source-scoped policy flag on the focused link (round-2 D7).

        Reports the intent with the current values and revision the app needs
        for the CAS; the APP performs the write (it owns the service), exactly
        as ``PolicyRequested`` splits. ``policy`` is ``subscribe`` or
        ``auto_push``; ``value`` is the DESIRED new value; ``revision`` is the
        policy revision the panel last saw (the CAS guard). ``confirm_egress``
        is the DELIBERATE public-egress acknowledgement, set only by the
        panel's own two-step consent below — never implied.
        """

        def __init__(
            self,
            link_id: str | None,
            local_id: str,
            policy: str,
            value: bool,
            revision: str,
            confirm_egress: bool = False,
        ) -> None:
            super().__init__()
            self.link_id = link_id
            self.local_id = local_id
            self.policy = policy
            self.value = value
            self.revision = revision
            self.confirm_egress = confirm_egress

    def __init__(self) -> None:
        super().__init__(id="hub-links-panel")
        self._rows: list[core.LinkRow] = []
        self._offset = 0
        self._selected = 0
        self._loading = False
        self._refreshing = False
        self._error = ""
        #: The link_id whose public auto-push enable is armed and awaiting the
        #: deliberate second press (round-2 D7). Reset on close.
        self._consent_link_id: str | None = None
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
        self._consent_link_id = None
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

    def action_toggle_subscribe(self) -> None:
        """`s` — flip the focused link's Subscribe (round-2 D7).

        Subscribe has no eligibility gate and no egress consent: only the CAS
        revision is required. The app performs the write through the SAME
        service path `lop hub link set` uses.
        """
        row = self.selected_row
        if row is None:
            return
        if row.link_id is None or not row.policy_revision:
            # A legacy/unresolved link has no revision to CAS against; the app
            # prints the resolve-first refusal (its own message owns that copy).
            self.post_message(self.PolicyRequested(row.link_id, row.local_id, "policy"))
            return
        desired = not bool(row.subscribe)
        self.post_message(
            self.PolicyToggleRequested(
                row.link_id, row.local_id, "subscribe", desired, row.policy_revision
            )
        )

    def action_toggle_auto_push(self) -> None:
        """`a` — flip the focused link's Auto-push (round-2 D7), consent-aware.

        Turning auto-push ON is the egress case: it is refused locally when the
        service says the origin is ineligible (the named reason is shown), and
        when the origin is PUBLIC it needs the deliberate consent. The panel
        owns the two-step consent (a second press confirms) so the write the app
        performs always carries an explicit ``confirm_egress`` — it never infers
        consent from the keypress alone.
        """
        row = self.selected_row
        if row is None:
            return
        if row.link_id is None or not row.policy_revision:
            self.post_message(self.PolicyRequested(row.link_id, row.local_id, "policy"))
            return
        if row.auto_push:
            # Turning it OFF never needs eligibility or consent.
            self.post_message(
                self.PolicyToggleRequested(
                    row.link_id, row.local_id, "auto_push", False, row.policy_revision
                )
            )
            return
        # Turning ON. Were we already awaiting consent for THIS row? Then this
        # press is the deliberate second step.
        if self._consent_link_id == row.link_id:
            self._consent_link_id = None
            self.post_message(
                self.PolicyToggleRequested(
                    row.link_id,
                    row.local_id,
                    "auto_push",
                    True,
                    row.policy_revision,
                    confirm_egress=True,
                )
            )
            return
        if not row.can_auto_push:
            # Ineligible — EXCEPT the public-consent case. For a public origin the
            # service reports ``unsafe-publication`` precisely because no egress
            # grant exists yet; that grant is what the consent flow below
            # CREATES, so treating it as a hard refusal would make public enable
            # impossible (the round-2 review's own reasoning). Every OTHER reason
            # is a genuine refusal reported by name through the app.
            if row.visibility == "public" and row.eligibility_reason == "unsafe-publication":
                self._consent_link_id = row.link_id
                self._repaint()
                return
            self.post_message(self.PolicyRequested(row.link_id, row.local_id, "auto-push"))
            return
        if row.visibility == "public":
            # Public egress: arm the two-step consent and repaint the prompt.
            self._consent_link_id = row.link_id
            self._repaint()
            return
        self.post_message(
            self.PolicyToggleRequested(
                row.link_id, row.local_id, "auto_push", True, row.policy_revision
            )
        )

    # -- paint ---------------------------------------------------------------

    def _visible_rows(self) -> list[core.LinkRow]:
        return list(self._rows)

    def _body_window(self, body: list[Text]) -> int:
        """The first body ENTRY to paint so the SELECTED entry stays visible.

        The window is bounded in PHYSICAL LINES (:data:`MAX_BODY_ROWS`), not in
        entries: a narrow row now wraps its policy pair onto a second line, so
        counting entries would let a wrapped list paint past the cap and leave
        the selection off screen. Entries around the selected one are admitted
        until the line budget is spent, biased toward showing what follows the
        selection (the architect's third bug: entries past the cap were
        unreachable; round-2 D8 added wrapped rows to the same accounting).
        """

        counts = [_entry_lines(entry) for entry in body]
        if sum(counts) <= MAX_BODY_ROWS:
            return 0
        selected = self._painted_index()
        if not (0 <= selected < len(counts)):
            return 0
        lo = hi = selected
        used = counts[selected]
        # Take entries after the selection first, then before, until the budget
        # is spent — the reader sees the neighbours that come next.
        while hi + 1 < len(counts) and used + counts[hi + 1] <= MAX_BODY_ROWS:
            hi += 1
            used += counts[hi]
        while lo - 1 >= 0 and used + counts[lo - 1] <= MAX_BODY_ROWS:
            lo -= 1
            used += counts[lo]
        return lo

    def _windowed_lines(self, body: list[Text], window: int) -> tuple[list[Text], bool]:
        """Paint physical lines from ``window`` until the budget is spent.

        Returns the lines and whether the window covered the WHOLE body (False
        means the more-rows indicator must report that links are off screen).
        """

        counts = [_entry_lines(entry) for entry in body]
        lines: list[Text] = []
        used = 0
        entry = window
        while entry < len(body):
            if lines and used + counts[entry] > MAX_BODY_ROWS:
                break
            lines.extend(_entry_lines_text(body[entry]))
            used += counts[entry]
            entry += 1
        return lines, entry >= len(body)

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

    def _content_width(self) -> int:
        """The width rows are COMPOSED to: the card width less its padding.

        The CSS sets ``padding: 1 2`` on this widget, so the padded content box
        is 4 cells narrower than ``styles.width``. Composing rows to the outer
        width (the round-2 bug) made every row 4 cells too wide and wrapped the
        policy tail; the rows and the box must be the SAME number.
        """

        return max(1, self.panel_width() - PANEL_PADDING_CELLS)

    def panel_width(self) -> int:
        """The OUTER card width, shared with the other toast-layer cards."""

        return max(24, overlay.screen_size(self)[0] - 8)

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        outer = self.panel_width()
        width = self._content_width()
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
        # WINDOW the body around the selection, in PHYSICAL LINES (a narrow row
        # wraps its policy pair onto a second line, round-2 D8), so an entry past
        # the cap is reachable and a wrapped list never paints past the budget.
        window = self._body_window(body)
        visible, whole = self._windowed_lines(body, window)
        # The FIRST and LAST LINK visible: entries before `window` may be section
        # headers, so count only LINK entries (not the headers render_rows
        # interleaves) to report a truthful range.
        first_link, last_link = self._visible_link_range(body, window, len(visible))
        lines = [self._title()]
        lines.append(Text("─" * min(width, 60), style="dim"))
        lines.append(Text(""))
        lines.extend(visible)
        lines.append(Text(""))
        lines.append(self._hint_row(whole, first_link, last_link))
        out = Text()
        for index, row in enumerate(lines):
            if index:
                out.append("\n")
            out.append_text(row)
        self.styles.width = outer
        self.styles.height = len(lines) + CARD_PADDING_ROWS
        self.update(out)

    def _visible_link_range(
        self, body: list[Text], window: int, visible_len: int
    ) -> tuple[int, int]:
        """The 1-based first/last LINK the window shows, for the indicator.

        ``body`` interleaves section HEADERS with link entries, so the painted
        window may open on a header; this counts only LINK entries to report a
        truthful ``n–m`` range. Headers are identified by their rendered BOLD
        style (render_rows' own marker), so a renamed section cannot break it.
        """

        def header(entry: Text) -> bool:
            # render_rows builds a section title as ``Text(name, style="bold")``
            # — a BASE style, not a span — while a link row's base style is
            # unset and its first span carries the slot ink. So the header test
            # is the entry's base style, not its spans.
            return str(entry.style) == "bold"

        before = sum(1 for entry in body[:window] if not header(entry))
        shown = sum(1 for entry in body[window : window + visible_len] if not header(entry))
        if not shown:
            return (0, 0)
        return (before + 1, before + shown)

    def _hint_row(self, whole: bool, first_link: int = 0, last_link: int = 0) -> Text:
        """The key-hint row, built from the catalogue (round-2 D6).

        The panel takes focus and binds esc/q/r/↑↓/enter/s/a, so it must TEACH
        them — the round-2 regression dropped this row (a catalogue move left the
        binding set with no hint). Composed strictly from catalogue keys, so a
        translated catalogue drives it; there is NO literal fallback (the i18n
        no-literal rule). ``↑↓`` appears only when the body actually scrolls.

        When the window does not cover the whole body (``whole`` False) it also
        reports the visible RANGE of the total link count (round-2 D11), the
        sessions list's ``n–m/total`` precedent, so a list longer than the cap
        never looks complete when it is not. Refreshing replaces the range with
        the catalogue's saving words, which is the transient the row already
        carried.
        """

        if self._refreshing:
            return Text(_t(_k.policy_saving()), style="dim")
        # The consent prompt supersedes the ordinary hints while armed, so the
        # second press's meaning is unambiguous (round-2 D7).
        if self._consent_link_id is not None:
            row = next((r for r in self._rows if r.link_id == self._consent_link_id), None)
            if row is not None:
                return Text(_t(_k.confirm_public(item=row.name)), style="dim")
        width = self._content_width()
        segments: list[str] = []
        if not whole:
            total = len(self._rows)
            segments.append(
                _t(
                    _k.hint_range(
                        first=str(first_link or 1),
                        last=str(last_link or min(total, MAX_BODY_ROWS)),
                        total=str(total),
                    )
                )
            )
        if len(self._visible_rows()) > MAX_BODY_ROWS:
            segments.append(_t(_k.hint_scroll()))
        segments.append(_t(_k.hint_subscribe()))
        segments.append(_t(_k.hint_auto_push()))
        segments.append(_t(_k.hint_activate()))
        segments.append(_t(_k.hint_refresh()))
        close = _t(_k.hint_close())
        segments.append(close)
        # Drop WHOLE segments to fit, the UsagePanel discipline. Priority is
        # explicit: the range (D11) and `esc close` are FACTS the reader cannot
        # recover elsewhere on the frame, so they go last; the key hints
        # (recoverable by pressing the keys) yield first, most obscure first.
        drop_order = [
            _t(_k.hint_activate()),
            _t(_k.hint_refresh()),
            _t(_k.hint_auto_push()),
            _t(_k.hint_subscribe()),
            _t(_k.hint_scroll()),
        ]
        for candidate in drop_order:
            if len(" · ".join(segments)) <= width:
                break
            segments = [seg for seg in segments if seg != candidate]
        return Text(" · ".join(segments), style="dim")

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

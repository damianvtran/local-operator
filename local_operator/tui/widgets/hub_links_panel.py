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

from rich.style import Style
from rich.text import Text
from textual import events
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
#: The confirm key for a public-egress publication (round-3 D15). A control key,
#: so typed-ahead printable text can never produce it; the ONLY key that
#: completes a publication.
CONFIRM_EGRESS_KEY = "ctrl+y"
#: How long after the panel takes focus a printable key is treated as typed-ahead
#: (swallowed) rather than intent (round-3 D15). Short: the burst a fast typist
#: produces arrives within ~200 ms of the open.
TYPED_AHEAD_WINDOW_S = 0.2


def _humanize_reason(reason: str) -> str:
    """Turn a service eligibility code into readable text for a human.

    The service returns codes like ``unsupported-precondition``; a raw code in a
    user-facing notice is exactly what round-3 D19 flags. Hyphens become spaces
    so the message reads as a sentence while still naming the fact.
    """

    return (reason or "ineligible").replace("-", " ")


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
        # The two source-scoped policy toggles (round-2 D7, round-3 D15). The
        # letters are the DIRECT toggles (the design's toggle affordance). No
        # `enter` binding: it was a stub that only reported `unsafe-publication`
        # or `unsupported-precondition` while the hint row advertised it
        # (round-3 D17), so it is dropped.
        ("s", "toggle_subscribe", "Toggle Subscribe"),
        ("a", "toggle_auto_push", "Toggle Auto-push"),
        # The PUBLIC-EGRESS CONFIRM (round-3 D15). A DISTINCT key, never the
        # arming key: two `a` presses must not complete a publication. `ctrl+y`
        # is unbound at app level and is a CONTROL key, so typed-ahead text (all
        # printable) can never produce it. It only acts while the consent prompt
        # is armed; see `on_key` for the disarm-on-any-other-key rule.
        ("ctrl+y", "confirm_egress", "Publish publicly"),
    ]

    class Dismissed(Message):
        pass

    class RefreshRequested(Message):
        pass

    class RefusalRaised(Message):
        """A policy action was REFUSED; the app paints it ON the card (round-3 D19).

        Carries the humanized message and its severity, so the app routes it to
        the surface the user is looking at rather than a transcript notice
        hidden behind the overlay.
        """

        def __init__(self, message: str, severity: str = "warning") -> None:
            super().__init__()
            self.text = message
            self.severity = severity

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
        #: deliberate confirmation (round-2 D7, round-3 D15). Cleared on close,
        #: on the confirm key, and on ANY other key (`on_key`), so it cannot
        #: survive a stray keystroke.
        self._consent_link_id: str | None = None
        #: A refusal/notice shown on the card's status line (round-3 D19), with
        #: its severity, kept until the next action or repaint clears it.
        self._notice = ""
        self._notice_severity = "info"
        #: Whether the panel has been DELIBERATELY engaged by a control key since
        #: it opened (round-3 D15). Typed-ahead text arrives as a burst of
        #: printable keys immediately after `/hub` takes focus; the panel must
        #: not treat that burst as policy-write commands. The first key the user
        #: PRESSES WITH INTENT — a navigation or action key, or a printable key
        #: after the burst settles — engages it; printable policy-write keys
        #: (`a`/`s`) before engagement are ignored.
        self._engaged: bool = False
        #: When (monotonic) the panel last took focus, for the typed-ahead window.
        self._opened_at: float = 0.0
        # Mounted once and kept (the `/usage` rule), so it must START hidden:
        # `display` defaults True, and a visible 0-row host in the toast layer
        # still nudged the transcript one cell left before any fetch. `close()`
        # restores exactly this state.
        self.display = False

    # -- state ---------------------------------------------------------------

    def start_fetch(self) -> None:
        """Open the panel and mark a fetch in flight (kept rows, not a skeleton)."""
        import time

        self._loading = True
        self._refreshing = False
        self._error = ""
        self._engaged = False
        self._opened_at = time.monotonic()
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

    def show_notice(self, message: str, severity: str = "info") -> None:
        """Show a REFUSAL/notice on the card itself (round-3 D19).

        Distinct from ``show_error`` (which replaces the body): a notice sits on
        the panel's own status line while the rows stay put, so the user reads it
        on the surface they are looking at rather than a transcript notice
        behind the overlay. ``severity`` picks the warning ink when appropriate.
        """

        self._notice = message
        self._notice_severity = severity
        self._repaint()

    def close(self) -> None:
        self._loading = False
        self._refreshing = False
        self._offset = 0
        self._consent_link_id = None
        self._engaged = False
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

    def on_key(self, event: "events.Key") -> None:
        """The typed-ahead guard + the disarm-on-any-other-key rule (round-3 D15).

        Runs BEFORE the key bindings (Textual's dispatch order), which is what
        lets it veto them. Two jobs:

        1. TYPED-AHEAD. `/hub` takes focus while the user may still be typing
           ahead; that literal text must never reach a policy-write key. Until
           the panel is ENGAGED (a navigation/action key, a control key, or a
           printable key pressed after the burst settles) a printable key is
           swallowed. The burst window is short and real: a fast typist's
           text arrives within ~200 ms of the open.
        2. DISARM. Any key OTHER than the named confirm cancels an armed consent
           prompt (a second `a`, `s`, `r`, arrows — none may complete a
           publication). `escape` cancels the PROMPT only and leaves the overlay
           open; every other key falls through to its normal binding.
        """
        import time

        key = event.key
        # 1. Typed-ahead: a printable key in the settle window, before
        # engagement, is swallowed entirely (no binding, no write).
        if not self._engaged and len(key) == 1 and key.isprintable():
            if time.monotonic() - self._opened_at <= TYPED_AHEAD_WINDOW_S:
                event.stop()
                return
            self._engaged = True  # the burst settled; this keystroke is intent
        # Anything that is not a printable character (a navigation/action/control
        # key) is deliberate intent.
        if len(key) != 1 or not key.isprintable():
            self._engaged = True
        # 2. Disarm an armed consent on any key but the confirm itself.
        if self._consent_link_id is not None and key != CONFIRM_EGRESS_KEY:
            self._consent_link_id = None
            if key == "escape":
                # Cancel the PROMPT, not the overlay (the designer's measured
                # gap: esc used to close the whole panel with no cancel-only
                # key). Consume it so the `dismiss` binding does not fire.
                self._repaint()
                event.stop()
                return
            self._repaint()

    def action_refresh(self) -> None:
        self.post_message(self.RefreshRequested())

    def action_move(self, delta: int) -> None:
        count = len(self._visible_rows())
        if count == 0:
            return
        self._selected = max(0, min(count - 1, self._selected + delta))
        self._repaint()

    def action_toggle_subscribe(self) -> None:
        """`s` — flip the focused link's Subscribe (round-2 D7).

        Subscribe has no eligibility gate and no egress consent: only the CAS
        revision is required. The app performs the write through the SAME
        service path `lop hub link set` uses.
        """
        self._notice = ""
        row = self.selected_row
        if row is None:
            return
        if row.link_id is None or not row.policy_revision:
            self.post_message(self.RefusalRaised(_t(_k.refusal_resolve_first()), "warning"))
            return
        desired = not bool(row.subscribe)
        self.post_message(
            self.PolicyToggleRequested(
                row.link_id, row.local_id, "subscribe", desired, row.policy_revision
            )
        )

    def action_toggle_auto_push(self) -> None:
        """`a` — flip the focused link's Auto-push (round-2 D7, round-3 D15).

        Turning auto-push ON is the egress case: a PRIVATE eligible origin
        enables directly; a PUBLIC origin arms the consent prompt, which only
        `ctrl+y` can complete (never a second `a`), and any other key disarms.
        An ineligible origin is REFUSED with its named reason. Nothing here
        infers consent from a keypress.
        """
        self._notice = ""
        row = self.selected_row
        if row is None:
            return
        if row.link_id is None or not row.policy_revision:
            self.post_message(self.RefusalRaised(_t(_k.refusal_resolve_first()), "warning"))
            return
        if row.auto_push:
            # Turning it OFF never needs eligibility or consent.
            self.post_message(
                self.PolicyToggleRequested(
                    row.link_id, row.local_id, "auto_push", False, row.policy_revision
                )
            )
            return
        if not row.can_auto_push:
            # Ineligible — EXCEPT the public-consent case, which is reached ONLY
            # after every earlier gate passed (capability, known baseline, clean
            # lineage: service.py's ordering). The public egress grant is what
            # the consent flow CREATES, so ``unsafe-publication`` here means
            # "consent missing", not "ineligible". Every OTHER reason is a
            # genuine refusal, reported with its humanized reason.
            if row.visibility == "public" and row.eligibility_reason == "unsafe-publication":
                self._consent_link_id = row.link_id
                self._repaint()
                return
            self.post_message(
                self.RefusalRaised(
                    _t(_k.refusal_ineligible(reason=_humanize_reason(row.eligibility_reason))),
                    "warning",
                )
            )
            return
        if row.visibility == "public":
            # Public egress: arm the consent prompt; only `ctrl+y` completes it.
            self._consent_link_id = row.link_id
            self._repaint()
            return
        self.post_message(
            self.PolicyToggleRequested(
                row.link_id, row.local_id, "auto_push", True, row.policy_revision
            )
        )

    def action_confirm_egress(self) -> None:
        """`ctrl+y` — the deliberate confirmation of an armed public egress.

        The ONLY key that completes a publication. Inert with nothing armed, so
        a stray `ctrl+y` is a no-op rather than a write.
        """
        if self._consent_link_id is None:
            return
        row = next((r for r in self._rows if r.link_id == self._consent_link_id), None)
        self._consent_link_id = None
        if row is None:
            self._repaint()
            return
        self.post_message(
            self.PolicyToggleRequested(
                row.link_id,
                row.local_id,
                "auto_push",
                True,
                row.policy_revision or "",
                confirm_egress=True,
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
        # N3 (round-3): never open a window on a link whose SECTION HEADER scrolled
        # off above it — a row with no section is an orphan. If the entry just
        # above the window is a header, ALWAYS include it (one extra chrome line
        # is a fair trade for a labelled row); the header is a single line and
        # the budget is a display cap, not a hard invariant.
        if lo > 0 and self._is_header(body[lo - 1]):
            lo -= 1
        return lo

    @staticmethod
    def _is_header(entry: Text) -> bool:
        """Whether a body entry is a section HEADER (render_rows shows them bold)."""

        return str(entry.style) == "bold"

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
        # The refusal/notice line (round-3 D19) rides above the hint/consent, in
        # the warning ink when it is a refusal, so the message is on the surface
        # the user is looking at.
        if self._notice:
            from local_operator.tui import theme as theme_mod

            ink = "warning" if self._notice_severity == "warning" else "muted"
            try:
                notice_style = Style(color=theme_mod.semantic_color(ink))
            except Exception:  # noqa: BLE001 - a card must not fail to paint
                notice_style = "dim"
            for chunk in Text(self._notice).wrap(self, width) or [Text("")]:
                chunk.stylize(notice_style)
                lines.append(chunk)
            lines.append(Text(""))
        # The CONSENT PROMPT is its OWN wrapped, height-accounted block (round-3
        # D16), not a segment squeezed into the one-row hint: it is a decision,
        # it must show in FULL at any width, and it needs priority ink (D20).
        # When it is not armed, the ordinary hint row is painted instead.
        consent = self._consent_lines(width)
        if consent:
            lines.extend(consent)
        else:
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

    def _consent_lines(self, width: int) -> list[Text]:
        """The armed public-egress consent: a wrapped, height-accounted block.

        Round-3 D16: the prompt was a single hint-row segment, so its wrapped
        second line was CLIPPED by the pinned card height — the reader never saw
        "published publicly". This composes it as its own block, wrapped to the
        content width, whose line count IS counted into the card height. It
        leads with a catalogue question and names BOTH keys (confirm `ctrl+y`,
        cancel `esc`), all from catalogue keys. Priority ink (D20): the warning
        colour, and a bold lead, so a decision is not the dimmest thing on the
        card.
        """

        if self._consent_link_id is None:
            return []
        row = next((r for r in self._rows if r.link_id == self._consent_link_id), None)
        if row is None:
            return []
        from local_operator.tui import theme as theme_mod

        warn = Style(color=theme_mod.semantic_color("warning"))
        text = _t(_k.confirm_public(item=row.name))
        lead = Text(
            _t(_k.confirm_lead()),
            style=Style(color=theme_mod.semantic_color("warning"), bold=True),
        )
        body = Text(text, style=warn)
        keys = Text(
            _t(_k.confirm_keys(confirm=_t(_k.hint_confirm()), cancel=_t(_k.hint_cancel()))),
            style=warn,
        )
        # Wrap lead/body/keys to the content width; every produced line is a
        # real painted line, so the caller's height sum stays honest.
        out: list[Text] = []
        for para in (lead, body, keys):
            for chunk in para.wrap(self, width) or [Text("")]:
                out.append(chunk)
        return out

    def _hint_row(self, whole: bool, first_link: int = 0, last_link: int = 0) -> Text:
        """The key-hint row, built from the catalogue (round-2 D6, round-3 D17).

        The panel takes focus and binds esc/q/r/↑↓/s/a (and ctrl+y while the
        consent is armed), so the row must TEACH exactly the keys that DO
        something. Round-3 D17: the `enter opens policy` hint advertised a stub
        with no live effect, so both the `enter` binding and this hint are gone —
        `s`/`a` ARE the policy actions. Composed strictly from catalogue keys;
        there is NO literal fallback (the i18n no-literal rule).

        ``↑↓ select`` appears whenever there is more than ONE link (round-3 D18:
        it is about choosing a row, not only about scrolling past the cap). When
        the window does not cover the whole body (``whole`` False) the row also
        reports the visible RANGE of the total link count (round-2 D11).
        """

        if self._refreshing:
            return Text(_t(_k.policy_saving()), style="dim")
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
        # D18: more than one link means the arrows SELECT; below the cap that is
        # still the way to move between rows, so it is not scroll-only.
        if len(self._visible_rows()) > 1:
            segments.append(_t(_k.hint_select()))
        segments.append(_t(_k.hint_subscribe()))
        segments.append(_t(_k.hint_auto_push()))
        segments.append(_t(_k.hint_refresh()))
        close = _t(_k.hint_close())
        segments.append(close)
        # Drop WHOLE segments to fit: the range (D11) and `esc close` are FACTS
        # recoverable nowhere else on the frame, so they go last; the key hints
        # yield first, most obscure first.
        drop_order = [
            _t(_k.hint_refresh()),
            _t(_k.hint_auto_push()),
            _t(_k.hint_subscribe()),
            _t(_k.hint_select()),
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

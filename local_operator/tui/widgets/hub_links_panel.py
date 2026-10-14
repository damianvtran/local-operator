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

from rich.cells import cell_len
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

#: Tab / shift+tab are consumed by the panel (round-5 D29): they would otherwise
#: move focus OFF the card, stranding the overlay open but unfocused so `esc`/`q`
#: no longer reach it and typed text lands in the composer UNDER the card.
FOCUS_ESCAPE_KEYS: frozenset[str] = frozenset({"tab", "shift+tab"})


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

    return list(entry.split("\n")) or [Text("")]


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
        #: The policy revision the prompt was ARMED against (round-4 D24). If the
        #: row's revision differs at confirm time, the user is confirming a value
        #: they never saw, so the prompt disarms and reports instead of writing.
        self._consent_revision: str | None = None
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
        self._notice = ""
        self._engaged = False
        self._opened_at = time.monotonic()
        # A re-fetch can change the revision under an armed prompt; drop it so a
        # confirm can never write a value the user did not see (round-4 D24).
        self._consent_link_id = None
        self._consent_revision = None
        self.display = True
        self._repaint()

    def show_links(self, snapshot: Mapping[str, Any]) -> None:
        """Install a finished inventory; keep the reader's place across a refresh.

        D27: rows are sorted into PAINT order ONCE here (section_order, then
        name casefold) — the SAME order ``core.render_rows`` paints. The SERVICE
        snapshot is sorted ``(kind, name)`` (agent, skill, team), while the card
        paints agent, team, skill; leaving ``_rows`` in service order made
        ``_selected``, the highlight, the window and the policy-write target
        disagree (the highlighted row was not the row `a`/`s` wrote). One order
        for every index is the fix.

        D38: the reader's place is remembered by the SELECTED LINK, not by the
        integer index. `_selected` points into a list that reorders when a link
        is inserted above — so an index-only carry silently moved the highlight
        to a different link, and the next `s`/`a` wrote THAT link. The anchor is
        re-resolved after every install; if it vanished the index clamps to a
        real neighbour, and an armed consent whose link is gone is DISARMED (it
        can never write a link the user is no longer looking at).
        """

        anchor = self._selection_anchor()
        self._rows = core.paint_order(core.rows_from_snapshot(snapshot))
        self._restore_selection(anchor)
        # D38: the armed consent is link_id-keyed; if its link is gone the prompt
        # must not survive invisibly (ctrl+y would otherwise write nothing visible).
        if self._consent_link_id is not None and not any(
            row.link_id == self._consent_link_id for row in self._rows
        ):
            self._consent_link_id = None
            self._consent_revision = None
        self._loading = False
        self._refreshing = False
        self._error = ""
        self.display = True
        self._repaint()

    def _selection_anchor(self) -> tuple[str | None, str, str] | None:
        """The selected row's IDENTITY (D38), stable across a reorder.

        ``(link_id, kind, name)``: a legacy row has no link_id, so kind+name
        stands in. ``None`` when nothing is selected.
        """

        row = self.selected_row
        if row is None:
            return None
        return (row.link_id, row.kind, row.name)

    def _restore_selection(self, anchor: tuple[str | None, str, str] | None) -> None:
        """Re-resolve ``_selected`` from an anchor, clamping when it is gone."""

        if anchor is None:
            self._selected = min(self._selected, max(0, len(self._rows) - 1))
            return
        link_id, kind, name = anchor
        for index, row in enumerate(self._rows):
            if link_id is not None and row.link_id == link_id:
                self._selected = index
                return
            if link_id is None and row.link_id is None and row.kind == kind and row.name == name:
                self._selected = index
                return
        # The selected link is gone: clamp to the nearest valid position.
        self._selected = min(self._selected, max(0, len(self._rows) - 1))

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
        self._notice = ""
        self._consent_link_id = None
        self._consent_revision = None
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
        # D29: consume tab/shift+tab so focus cannot leave the card while it is
        # open. Left unhandled they move focus to the TranscriptView, stranding
        # the overlay open-but-unfocused: `esc`/`q` no longer reach the panel and
        # typed letters go to the composer UNDER the card. Consuming them keeps
        # the card the focus owner for its whole lifetime.
        if key in FOCUS_ESCAPE_KEYS:
            event.stop()
            event.prevent_default()
            return
        # A key is TYPED TEXT iff it produced a printable CHARACTER. Textual NAMES
        # non-alphanumeric printables (`space`, `comma`, `full_stop`), so `key` is
        # multi-char for exactly the characters a burst is full of — classifying on
        # `len(key)` let a space/comma/period count as a "deliberate" control key
        # and ENGAGE the panel, so `h<space>a` wrote policy (round-4 D23). The
        # character is the reliable signal; `event.is_printable` is Textual's own.
        printable = event.is_printable
        # 1. Typed-ahead: a printable key in the settle window, before
        # engagement, is swallowed entirely (no binding, no write).
        if not self._engaged and printable:
            if time.monotonic() - self._opened_at <= TYPED_AHEAD_WINDOW_S:
                event.stop()
                return
            self._engaged = True  # the burst settled; this keystroke is intent
        elif not printable and key in {
            # The panel's own entries are (key, action, description) tuples;
            # Textual also accepts ``Binding`` objects, whose key is the attribute.
            entry[0] if isinstance(entry, tuple) else entry.key
            for entry in self.BINDINGS
        }:
            # D30: engage only on a key the panel BINDS (its BINDINGS is the one
            # source of truth). `enter`, ctrl chords and F-keys are not panel
            # commands, so they must not count as intent — otherwise a burst like
            # `h`,enter,`a` engages and `a` writes policy.
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
        # D25: a refusal names the link it was about, so moving off it clears the
        # notice — otherwise a stale refusal appears to describe the new row.
        self._notice = ""
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
                self._consent_revision = row.policy_revision
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
            self._consent_revision = row.policy_revision
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
        a stray `ctrl+y` is a no-op rather than a write. The revision the prompt
        was ARMED against is checked first: if the row moved (an external edit or
        a re-fetch), the user is confirming a value they never saw, so the
        prompt disarms and reports instead (round-4 D24).
        """
        if self._consent_link_id is None:
            return
        row = next((r for r in self._rows if r.link_id == self._consent_link_id), None)
        armed_revision = self._consent_revision
        self._consent_link_id = None
        self._consent_revision = None
        if row is None:
            self._repaint()
            return
        if armed_revision is not None and row.policy_revision != armed_revision:
            self._notice = _t(_k.refusal_revision_moved())
            self._notice_severity = "warning"
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

    def _body_budget(self, consent_lines: int) -> int:
        """Body rows available before the card would run off the screen (D22).

        The body was capped by the CONSTANT :data:`MAX_BODY_ROWS`, so at a 24-row
        terminal a card with several links extended past the bottom edge and its
        consent keys were clipped. The real budget is what the card may occupy
        (:func:`overlay.rows_above_dock`) less its own chrome, the consent block,
        and the padding — floored so a tiny screen still shows some body.

        D36 SCOPE OF THE CLAIM: this fits the card above the dock for screens of
        at least 24 rows and 60 columns. Below that floor the body keeps its
        one-row minimum and the consent DECISION stays FULLY painted (lead, body
        and both keys) — it deliberately covers the composer rather than clipping
        the decision; `test_card_never_overlaps_the_dock` covers the fitting
        range only.
        """

        available = overlay.rows_above_dock(self)
        chrome = CHROME_ROWS + CARD_PADDING_ROWS + consent_lines + 1  # +1 hint row
        return max(1, min(MAX_BODY_ROWS, available - chrome))

    def _plan_window(self, body: list[Text], budget: int) -> tuple[int, int, int | None]:
        """The inclusive entry range ``[lo, hi]`` to paint, plus a sticky header.

        Budgeted in PHYSICAL LINES; it ALWAYS contains the selection (D21) and
        NEVER exceeds ``budget`` (D28: round-3's N3 header pull-back added a line
        AFTER the budget was spent, so a header-adorned window could paint one
        row past the screen and cover the dock). The header reservation is
        therefore applied BEFORE the final bounds, not after.

        Returns ``(lo, hi, sticky)`` where ``sticky`` is the index of a section
        HEADER to paint as the FIRST line when the window opens mid-section (D35:
        team/skill rows used to appear with no section title). The sticky line is
        counted against the budget, so a full budget drops the last entry rather
        than overflow.
        """

        counts = [_entry_lines(entry) for entry in body]
        if not counts:
            return (0, -1, None)
        # ``_selected`` indexes LINK rows (what render_rows paints and what the
        # write uses); ``body`` interleaves section HEADERS, so map link-index ->
        # body-index by counting non-headers. With ``_rows`` in paint order the
        # mapping is monotonic, so the window, the highlight and the write target
        # are all the same index (D27).
        selected = self._body_index_for(self._selected, body)
        if selected is None:
            selected = 0
        lo = hi = selected
        used = counts[selected]
        # Reserve the selection's OWN header first (N3), so it rides in with the
        # row and cannot crowd the selection out of a spent budget.
        if lo > 0 and self._is_header(body[lo - 1]):
            lo -= 1
            used += counts[lo]
        # Admit following entries, then preceding ones, within the budget.
        while hi + 1 < len(counts) and used + counts[hi + 1] <= budget:
            hi += 1
            used += counts[hi]
        # D37: never END the window on a section header with its rows below the
        # cap — an empty-looking section. If the last admitted entry is a header
        # past the selection, drop it.
        while hi > selected and self._is_header(body[hi]):
            used -= counts[hi]
            hi -= 1
        while lo - 1 >= 0 and used + counts[lo - 1] <= budget:
            lo -= 1
            used += counts[lo]
            # Keep a preceding header with its first row.
            if lo - 1 >= 0 and self._is_header(body[lo - 1]) and used + counts[lo - 1] <= budget:
                lo -= 1
                used += counts[lo]
        # N3: if the entry just above the window is a header, include it IF the
        # budget allows (never overflow).
        if lo > 0 and self._is_header(body[lo - 1]) and used + counts[lo - 1] <= budget:
            lo -= 1
            used += counts[lo]
        # D35 STICKY HEADER: if the window still opens mid-section, remember the
        # section header to paint AS the first line (it is not in the range, so
        # the caller counts its line against the budget and drops the last entry
        # to pay for it).
        sticky: int | None = None
        if lo > 0 and not self._is_header(body[lo]):
            for i in range(lo - 1, -1, -1):
                if self._is_header(body[i]):
                    sticky = i
                    break
        # D28: pay for the sticky line INSIDE the budget by trimming a NON-SELECTED
        # entry's tail/head, so the painted total (sticky + entries) never exceeds
        # ``budget`` and the card cannot reach the dock. The selection is ALWAYS
        # preserved: prefer the far tail, then the head, and only drop the sticky
        # line if nothing else can go.
        if sticky is not None:
            while used + 1 > budget and (hi > selected or lo < selected):
                if hi > selected:
                    used -= counts[hi]
                    hi -= 1
                elif lo < selected:
                    used -= counts[lo]
                    lo += 1
            if used + 1 > budget:
                # Even the selection alone plus the header exceeds a tiny budget;
                # drop the sticky line rather than overflow. A row without its
                # section is worse than a covered dock only on a 1-row screen.
                sticky = None
        return (lo, hi, sticky)

    @staticmethod
    def _body_index_for(link_index: int, body: list[Text]) -> int | None:
        """Map a LINK index to its BODY entry index (skipping section headers)."""

        seen = 0
        for i, entry in enumerate(body):
            if HubLinksPanel._is_header(entry):
                continue
            if seen == link_index:
                return i
            seen += 1
        return None

    def _windowed_lines(
        self, body: list[Text], lo: int, hi: int, sticky: int | None = None
    ) -> list[Text]:
        """The physical lines of body entries ``[lo, hi]``, with a sticky title.

        When ``sticky`` is set (D35) the section header at that index is painted
        FIRST — the section title a mid-section window would otherwise omit.
        """

        lines: list[Text] = []
        if sticky is not None:
            lines.append(body[sticky])
        for entry in body[lo : hi + 1]:
            lines.extend(_entry_lines_text(entry))
        return lines

    @staticmethod
    def _is_header(entry: Text) -> bool:
        """Whether a body entry is a section HEADER (render_rows shows them bold)."""

        return str(entry.style) == "bold"

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
        # The consent block is composed FIRST: its line count is part of the
        # screen-derived body budget (D22), so it cannot be clipped.
        consent = self._consent_lines(width)
        budget = self._body_budget(len(consent))
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
        # WINDOW the body around the selection, in PHYSICAL LINES and within the
        # SCREEN-derived budget (round-2 D8, round-4 D21/D22, round-5 D28/D35), so
        # an entry past the cap is reachable, the selection is ALWAYS painted, a
        # mid-section window carries its section title, and the painted total
        # never exceeds the budget (so the card cannot reach the dock).
        lo, hi, sticky = self._plan_window(body, budget)
        visible = self._windowed_lines(body, lo, hi, sticky)
        whole = lo == 0 and (hi == len(body) - 1)
        first_link, last_link = self._visible_link_range(body, lo, hi)
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
            for chunk in Text(self._notice).wrap(self.app.console, width) or [Text("")]:
                chunk.stylize(notice_style)
                lines.append(chunk)
            lines.append(Text(""))
        # The CONSENT PROMPT is its OWN wrapped, height-accounted block (round-3
        # D16), not a segment squeezed into the one-row hint: it is a decision,
        # it must show in FULL at any width, and it needs priority ink (D20).
        # When it is not armed, the ordinary hint row is painted instead.
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

    def _visible_link_range(self, body: list[Text], lo: int, hi: int) -> tuple[int, int]:
        """The 1-based first/last LINK the window ``[lo, hi]`` shows, for the indicator.

        ``body`` interleaves section HEADERS with link entries, so the painted
        window may open on a header; this counts only LINK entries to report a
        truthful ``n–m`` range.
        """

        before = sum(1 for entry in body[:lo] if not self._is_header(entry))
        shown = sum(1 for entry in body[lo : hi + 1] if not self._is_header(entry))
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
        # D26/D32/D39: the escape route when a terminal swallows a control chord.
        # The `or:` command line and the `link id` line are shown TOGETHER or not
        # at all — a user must never see `<id>` with no way to obtain an id; when
        # only the command line fits, a `lop hub links` pointer stands in. Each
        # emitted only when it fits unwrapped, so they can never break the D22
        # height budget or wrap the keys line.
        paras = [lead, body, keys]
        alt = _t(_k.confirm_alt())
        id_line = _t(_k.confirm_id(id=(row.link_id or row.local_id)))
        if cell_len(alt) <= width:
            if cell_len(id_line) <= width:
                paras.append(Text(alt, style=warn))
                paras.append(Text(id_line, style=warn))
            else:
                hint = _t(_k.confirm_id_hint())
                if cell_len(hint) <= width:
                    paras.append(Text(alt, style=warn))
                    paras.append(Text(hint, style=warn))
        # Wrap each paragraph to the content width; every produced line is a
        # real painted line, so the caller's height sum stays honest.
        out: list[Text] = []
        for para in paras:
            for chunk in para.wrap(self.app.console, width) or [Text("")]:
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

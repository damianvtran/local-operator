"""Quick-send: who a project's message goes to, and the card that asks.

Two halves, and the split is the same one every other projects surface keeps:

- :func:`send_targets` is PURE (a view payload in, rows out). The target list is
  the part worth pinning — its ORDER is the feature ("the manager first, then
  the project's own sessions, live ones first"), and a pure list is the only
  place that order can be asserted without a terminal.
- :class:`SendTargetCard` is the surface: a focusable ``Container`` that
  FLOATS over the page — mounted on the overlay layer with a deterministic
  height and a Python-set offset — rather than a ``ModalScreen`` or a flow
  child of the page (design review round 1, D1/D2).

The card is deliberately not a modal, and that is a deviation from the parity
spec's §5.6 parenthetical ("overlays are the `/resume`-family card") recorded
here rather than in a commit message: this app's own recorded decision
(``widgets/ask_picker.py``) moved its cards out of ``ModalScreen`` because a
modal covers the surface the question is ABOUT — the tool output being asked
about there, the project whose session you are messaging here. A card that
hides the project row it was opened from makes the reader dismiss it to
re-read what they are answering about.

Floating is what makes the non-modal shape safe. As a flow child the card
competed with the canvas for rows and lost its own tail at 60x24 (and at
80x24 painted zero target rows while `enter` still picked one — D1); on the
overlay layer it is arranged independently, so opening it moves nothing under
it, and its height (chrome + the rows actually painted) is pinned by
`SendTargetCard.set_available` to the ground the page handed it, so no row is
silently clipped.

The keys are the app's picker grammar (↑↓ wrap, ``enter`` selects, ``esc``
closes, type to filter) because that is what every other card in this app
teaches, and a second grammar would be a second thing to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rich.text import Text
from textual.containers import Container
from textual.message import Message
from textual.widgets import Input, Static

#: The footer line the card paints when it has no rows at all, and the last row
#: of every list (spec §7.5.1). It is a ROW rather than a hint because it is
#: advice about what to do next, not a key this card implements — `s` belongs
#: to the page, and a card that swallowed it would be a lie about its own keys.
#: The card's empty state. It names the two things that DO work today —
#: ``esc`` closes the card, and the project detail is where a link is made —
#: rather than the spec's ``s starts a session``, which is P5b's binding and is
#: not on this slice: a card pointing at a dead key is the one failure this
#: footer exists to prevent (agent review round 1, U1). P5b rebinds it to the
#: real start-session flow.
NO_TARGET_FOOTER = "no targets — esc closes · link a session from the project detail"

#: How many of a session id's characters name it in a row. The row also carries
#: the conversation name when there is one, so this is only the fallback handle
#: — long enough to be unique in practice, short enough to leave room for it.
SHORT_ID_CELLS = 12

#: The card's fixed chrome, in rows: one padding row above and below, the
#: `send to` title, the rule under it, the filter, the note and the legend.
#: The card's height is exactly this plus the rows it paints — a geometry the
#: page checks against the ground it floats over (design review round 1,
#: D1/D5: as a `1fr` flow child the card silently lost its tail at 60x24).
SEND_CARD_CHROME_ROWS = 7

#: The most target rows painted at once; the window scrolls within this so the
#: selected row is always on screen ("no unpainted selection", D1).
SEND_CARD_ROW_CAP = 5

#: The card's widest column count; narrower terminals get the page's width.
SEND_CARD_MAX_WIDTH = 60


def compose_band(target: SendTarget) -> str:
    """The composer's recipient strip while the page is composing (spec §7.5.2).

    ONE function so the strip cannot be spelled two ways: the app paints it
    when compose opens, and the receipt paths repaint the same surface. It
    names the target and the ONE key that still reaches the page while the
    composer holds the caret (``esc``); it does NOT advertise ``m target`` —
    ``m`` types a letter into the message, and a hinted key that types your
    sentence is the worst version of the hinted-key-that-does-nothing defect
    (UX round 1, Q3/U5).
    """
    return f"send to: {target.label} · esc cancel"


def pending_line(target: SendTarget) -> str:
    """The in-flight statement, painted where the reader is looking (UX U2).

    Shown on submit, before the resolver answers; the receipt replaces it.
    Without it the draft vanished and the band kept inviting another send, so
    for the seconds a delivery takes the surface said nothing had happened.
    """
    return f"sending to {target.label}…"


def sent_line(target: SendTarget, state_word: str) -> str:
    """A receipt in the app's HUMAN vocabulary (agent review F4; UX U3).

    ``state_word`` is :attr:`DeliveryOutcome.state_word` — the same words the
    send tool's card paints (``delivered`` / ``wake unconfirmed`` /
    ``delivery unconfirmed``) — rather than the model-facing stdout sentence
    (``→ <id>: …`` with a raw message uuid) this surface used to echo.
    """
    return f"sent to {target.label} · {state_word}"


def refusal_line(target: SendTarget, reason: str) -> str:
    """A refusal that names the row the reader picked, never a lone hex id.

    The band showed ``target.label``; the refusal repeats it (F4/U3: the old
    sentence named only the session id, so the sentence the reader had to
    recover from named neither the row nor the person).
    """
    return f"could not deliver to {target.label}: {reason}"


def send_error_line(target: SendTarget, reason: str) -> str:
    """A send that never reached delivery: resolution or validation failed."""
    return f"could not send to {target.label}: {reason}"


@dataclass(frozen=True)
class SendTarget:
    """One row of the target picker.

    ``session_id`` is what the send resolves (never a display name — the
    resolver's substring vocabulary is exactly the wrong-recipient hazard
    ``peer_send`` documents, so the card hands back an EXACT id). ``kind``
    separates the manager row from a project's own sessions, which the card
    paints differently and the sender reports differently.
    """

    kind: str
    session_id: str
    label: str
    state: str = ""
    live: bool = False

    @property
    def row_text(self) -> str:
        """The row as one line: ``manager  · session ab12 [live]``."""
        short = self.session_id[:SHORT_ID_CELLS] if self.session_id else ""
        parts = [self.label]
        if short:
            parts.append(f"session {short}")
        if self.state:
            parts.append(f"[{self.state}]")
        return "  · ".join(parts)


def _session_state(row: dict[str, Any]) -> tuple[str, bool]:
    """``(state word, is_live)`` for one linked-session payload row.

    Read from the payload the app ACTUALLY builds, not from a convenient key:
    ``build_project_view`` puts liveness under ``runtime.state`` (the values
    ``scan_runtime_states`` produces — ``live``/``wedged``/``stale``), and
    ``exists is False`` is the stale link the detail page already calls
    ``missing``. An earlier revision read a bare ``row["live"]`` flag, which
    only ever existed in this module's own fixtures — so the picker's
    live-first order and its ``[live]`` ink were both untested against real
    output (agent review round 1, F2; QA Q1).
    """
    if row.get("exists") is False:
        return ("missing", False)
    runtime_value = row.get("runtime")
    runtime: dict[str, Any] = runtime_value if isinstance(runtime_value, dict) else {}
    state = str(runtime.get("state") or "stopped")
    # Only `live` is dialable-now: a `wedged` record still has a process but
    # refused the last ack, and it must not outrank a healthy peer in the
    # order the reader is choosing from.
    return (state, state == "live")


def send_targets(
    view: dict[str, Any],
    *,
    own_session: str | None,
    manager: SendTarget | None = None,
) -> list[SendTarget]:
    """The picker's rows, in the spec's order (§7.5.1).

    Manager first when the session runs under an engaged team, then the
    project's linked sessions LIVE-FIRST, and this session excluded throughout
    (you cannot message yourself — the send core refuses it, and a row that can
    only be refused is a dead row).
    """
    rows: list[SendTarget] = []
    if manager is not None and manager.session_id != own_session:
        rows.append(manager)
    linked = view.get("sessions")
    live: list[SendTarget] = []
    rest: list[SendTarget] = []
    for row in linked if isinstance(linked, list) else []:
        if not isinstance(row, dict):
            continue
        session_id = str(row.get("session_id") or "")
        if not session_id or session_id == own_session:
            continue
        state, is_live = _session_state(row)
        title = str(row.get("title") or row.get("conversation_name") or "").strip()
        label = f'"{title}"' if title else session_id[:SHORT_ID_CELLS]
        target = SendTarget(
            kind="session",
            session_id=session_id,
            label=label,
            state=state,
            live=is_live,
        )
        (live if is_live else rest).append(target)
    rows.extend(live)
    rows.extend(rest)
    return rows


def filter_targets(rows: Iterable[SendTarget], needle: str) -> list[SendTarget]:
    """Subsequence match over a row's text — the app's picker filter grammar.

    Subsequence rather than prefix, for the reason the shipped pickers record:
    a reader who remembers "review" and half the ids should not have to type
    the row's leading characters to reach it.
    """
    needle = needle.strip().casefold()
    if not needle:
        return list(rows)
    kept: list[SendTarget] = []
    for row in rows:
        haystack = row.row_text.casefold()
        position = 0
        for character in needle:
            position = haystack.find(character, position) + 1
            if position == 0:
                break
        else:
            kept.append(row)
    return kept


class SendTargetCard(Container):
    """The send picker: rows, a filter, and the grammar every other card teaches.

    Focusable, and it TAKES focus on mount: the keys are ``↑``/``↓``/``enter``
    and printable characters, all of which the composer would otherwise swallow
    as input (the ask card's recorded reason). Focus returns to the page when
    the card closes.

    It FLOATS: the sheet puts it on the overlay layer (``layer: toast``) and
    the page positions it with a Python-set offset against the canvas's
    content box, so opening the card reflows nothing under it (D1/D2). Its
    height is chrome plus the rows it is actually painting — the budget comes
    from the page through :meth:`set_available` — so every painted row is
    inside the card and the card is inside its ground.
    """

    can_focus = True

    BINDINGS = [
        ("up", "move(-1)", "Move"),
        ("down", "move(1)", "Move"),
        ("enter", "choose", "Choose"),
        ("escape", "close", "Close"),
    ]

    class Chosen(Message):
        """A target was picked — the page owns what happens next."""

        def __init__(self, card: "SendTargetCard", target: SendTarget) -> None:
            super().__init__()
            self.card = card
            self.target = target

    class Closed(Message):
        """The card was dismissed without a target."""

        def __init__(self, card: "SendTargetCard") -> None:
            super().__init__()
            self.card = card

    def __init__(
        self,
        rows: list[SendTarget],
        *,
        style_for: Any = None,
    ) -> None:
        super().__init__(classes="projects-send-card")
        self._all = list(rows)
        self._rows: list[SendTarget] = list(rows)
        self._index = 0
        self._style_for = style_for
        #: The painted window's first row; slides so ``_index`` is always
        #: painted (design review round 1, D1's "no unpainted selection").
        self._top = 0
        #: Rows the window is painting right now — ``min(rows, cap, room)``.
        self._visible = 0
        #: The row budget the PAGE hands over (the canvas's content height).
        #: Standalone cards (tests) get a small default so they still paint.
        self._available = SEND_CARD_CHROME_ROWS + SEND_CARD_ROW_CAP

    def compose(self):
        yield Static("send to", classes="projects-send-title")
        # The one rule every surface here uses under its title (design D2's
        # "one rule or box", in this sheet's separator vocabulary).
        yield Static("", id="projects-send-rule", classes="projects-send-rule")
        yield Input(placeholder="type to filter", id="projects-send-filter")
        yield Static("", id="projects-send-rows", classes="projects-send-rows")
        yield Static("", id="projects-send-note", classes="projects-send-note")
        yield Static(
            "type to filter · ↑↓ move · ↵ select · esc close",
            id="projects-send-legend",
            classes="projects-send-hints",
        )

    def on_mount(self) -> None:
        self._repaint()
        # The INPUT takes focus, not the card: the card's grammar is "type to
        # filter", and with the card itself focused every printable key died
        # on it (QA round 1, Q2 — typing did nothing until a Tab, and nothing
        # advertised the Tab). ↑/↓/enter/escape still reach the card from the
        # Input — it does not consume them (probed) — so the old dance is
        # gone and typing filters on the first keystroke.
        try:
            self.query_one("#projects-send-filter", Input).focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    def on_resize(self) -> None:
        # The rule is cut to the card's measured width, which only exists once
        # layout has run; this repaint is where the first cut happens.
        self._repaint()

    # -- rows ---------------------------------------------------------------
    @property
    def rows(self) -> list[SendTarget]:
        """The rows the filter currently admits."""
        return list(self._rows)

    @property
    def index(self) -> int:
        return self._index

    def selected(self) -> SendTarget | None:
        return self._rows[self._index] if self._rows else None

    def painted_rows(self) -> list[str]:
        return [row.row_text for row in self._rows]

    # -- the painted window (design D1: no unpainted selection) --------------
    def set_available(self, rows: int) -> None:
        """Hand the card its row budget — the page's placement call.

        ``rows`` is the height of the ground the card floats over. The card
        spends it chrome-first and paints only the target rows that fit,
        capped at :data:`SEND_CARD_ROW_CAP`; :meth:`_sync_window` slides the
        window so the selection is always painted. Idempotent, and safe to
        call before the card has composed (an unmounted card has no children
        to size — ``on_mount`` paints when they exist).
        """
        self._available = max(0, rows)
        try:
            self._repaint()
        except Exception:  # noqa: BLE001 — not composed yet; on_mount paints
            pass

    def window_rows(self) -> list[SendTarget]:
        """The slice of the rows the card is painting right now."""
        return self._rows[self._top : self._top + self._visible]

    def painted_range(self) -> range:
        """The row positions currently painted (empty when the ground is too
        short for even one — the card then refuses to SELECT, see
        :meth:`action_choose`)."""
        return range(self._top, self._top + self._visible)

    def _visible_count(self) -> int:
        room = self._available - SEND_CARD_CHROME_ROWS
        return max(0, min(len(self._rows), SEND_CARD_ROW_CAP, room))

    def _sync_window(self) -> None:
        """Set the painted count and slide the window so ``_index`` is in it."""
        self._visible = self._visible_count()
        if self._visible <= 0:
            self._top = 0
            return
        top = min(self._top, max(0, len(self._rows) - self._visible))
        if self._index < top:
            top = self._index
        elif self._index >= top + self._visible:
            top = self._index - self._visible + 1
        self._top = max(0, top)

    def _repaint(self) -> None:
        self._sync_window()
        body = self.query_one("#projects-send-rows", Static)
        note = self.query_one("#projects-send-note", Static)
        rule = self.query_one("#projects-send-rule", Static)
        # The rows block is exactly as tall as the window: the card's height is
        # chrome + painted rows and NOTHING else (the deterministic geometry
        # the page asserts against its ground).
        body.styles.height = self._visible
        if not self._rows:
            body.update(Text(""))
            # The empty state IS the footer line (spec §7.5.1): a card with no
            # targets says what to do about that, not "no results".
            note.update(Text(NO_TARGET_FOOTER, style=self._ink("muted")))
        else:
            lines: list[str] = []
            for position, row in enumerate(self.window_rows(), start=self._top):
                marker = "▸" if position == self._index else " "
                lines.append(f"{marker} {row.row_text}")
            body.update(Text("\n".join(lines), no_wrap=True))
            # A window with rows off its edge says so, so a scrolled list never
            # reads as the whole list.
            hidden = len(self._rows) - self._visible
            note.update(Text(f"+{hidden} more" if hidden > 0 else "", style=self._ink("dim")))
        # The rule is cut to the card's own measured width; before the first
        # layout there is no measurement and this paints nothing (the next
        # repaint cuts it — `on_resize` guarantees one).
        width = self.content_size.width
        rule.update(Text("─" * width if width > 0 else "", style=self._ink("dim")))

    def _ink(self, key: str) -> Any:
        """A resolved style, or ``None`` when the host supplied no resolver."""
        if self._style_for is None:
            return None
        try:
            return self._style_for(key)
        except Exception:  # noqa: BLE001 — a card must not fail a keypress
            return None

    # -- the grammar --------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """``enter`` from the filter — the card's common choose path (Q2).

        With the Input focused, Enter never reaches the card's own `enter`
        binding; the Input posts this instead, and both paths converge on
        :meth:`action_choose` so "choose" cannot mean two things.
        """
        event.stop()
        self.action_choose()

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._rows = filter_targets(self._all, event.value)
        self._index = 0
        self._repaint()

    def action_move(self, delta: int) -> None:
        """``↑``/``↓`` — WRAP, the shipped picker convention."""
        if not self._rows:
            return
        self._index = (self._index + delta) % len(self._rows)
        self._repaint()

    def action_choose(self) -> None:
        target = self.selected()
        # "No unpainted selection" (design D1): `enter` can only pick a row that
        # is on screen. The guard is real, not defensive — a ground too short
        # for even one row leaves the list unselectable rather than answering
        # "who am I sending to" with nobody on screen (the 80x24 blind enter).
        if target is None or self._index not in self.painted_range():
            return
        self.post_message(self.Chosen(self, target))

    def action_close(self) -> None:
        self.post_message(self.Closed(self))

    def on_click(self, event: Any) -> None:  # noqa: ANN001 — Textual event type
        """A click selects a row; a second click on it chooses (spec §10.4).

        Positions are WINDOW-relative: the rows block paints the window, so a
        click maps to ``_top + row`` and a click on the note or legend hits no
        row (the old index arithmetic treated them as rows — D1's class).
        """
        offset = getattr(event, "y", None)
        if offset is None:
            return
        body_y = self._row_block_top()
        row = offset - body_y
        if row < 0 or row >= self._visible:
            return
        event.stop()
        position = self._top + row
        if position == self._index:
            self.action_choose()
            return
        self._index = position
        self._repaint()

    def _row_block_top(self) -> int:
        """The first row of the rows block, in card-relative coordinates."""
        try:
            block = self.query_one("#projects-send-rows", Static)
            return block.region.y - self.region.y
        except Exception:  # noqa: BLE001 — before layout there is no region
            # padding row + title + rule + filter
            return 3

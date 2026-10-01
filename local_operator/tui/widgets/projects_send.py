"""Quick-send: who a project's message goes to, and the card that asks.

Two halves, and the split is the same one every other projects surface keeps:

- :func:`send_targets` is PURE (a view payload in, rows out). The target list is
  the part worth pinning — its ORDER is the feature ("the manager first, then
  the project's own sessions, live ones first"), and a pure list is the only
  place that order can be asserted without a terminal.
- :class:`SendTargetCard` is the surface: a focusable ``Container`` mounted in
  the page, NOT a ``ModalScreen``.

The card is deliberately not a modal, and that is a deviation from the parity
spec's §5.6 parenthetical ("overlays are the `/resume`-family card") recorded
here rather than in a commit message: this app's own recorded decision
(``widgets/ask_picker.py``) moved its cards out of ``ModalScreen`` because a
modal covers the surface the question is ABOUT — the tool output being asked
about there, the project whose session you are messaging here. A card that
hides the project row it was opened from makes the reader dismiss it to
re-read what they are answering about.

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
NO_TARGET_FOOTER = "no target? s starts a session"

#: How many of a session id's characters name it in a row. The row also carries
#: the conversation name when there is one, so this is only the fallback handle
#: — long enough to be unique in practice, short enough to leave room for it.
SHORT_ID_CELLS = 12


def compose_band(target: SendTarget) -> str:
    """The composer's recipient strip while the page is composing (spec §7.5.2).

    ONE function because the band is painted in two places — the editor's
    placeholder when compose opens, and again after a refusal — and a second
    spelling of the strip is how the two end up disagreeing about the target or
    the way out.
    """
    return f"send to: {target.label} · m target · esc cancel"


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

    The payload's own vocabulary, not a second one: ``exists`` false means the
    session directory is gone (the detail page's "missing" row), and ``live``
    is the app's live-session flag. A row that can be neither dialled nor
    woken still gets a row — dropping it silently would make a linked session
    invisible and the count wrong.
    """
    if row.get("exists") is False:
        return ("missing", False)
    if row.get("live"):
        return ("live", True)
    return ("stopped", False)


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

    def compose(self):
        yield Static("send to", classes="projects-send-title")
        yield Input(placeholder="type to filter", id="projects-send-filter")
        yield Static("", id="projects-send-rows", classes="projects-send-rows")
        yield Static("", id="projects-send-note", classes="projects-send-note")
        yield Static(
            "type to filter · ↑↓ move · ↵ select · esc close",
            classes="projects-send-hints",
        )

    def on_mount(self) -> None:
        self._repaint()
        try:
            self.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

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

    def _repaint(self) -> None:
        body = self.query_one("#projects-send-rows", Static)
        note = self.query_one("#projects-send-note", Static)
        if not self._rows:
            body.update(Text(""))
            # The empty state IS the footer line (spec §7.5.1): a card with no
            # targets says what to do about that, not "no results".
            note.update(Text(NO_TARGET_FOOTER, style=self._ink("muted")))
        else:
            lines: list[str] = []
            for position, row in enumerate(self._rows):
                marker = "▸" if position == self._index else " "
                lines.append(f"{marker} {row.row_text}")
            body.update(Text("\n".join(lines)))
            note.update(Text(NO_TARGET_FOOTER, style=self._ink("dim")))

    def _ink(self, key: str) -> Any:
        """A resolved style, or ``None`` when the host supplied no resolver."""
        if self._style_for is None:
            return None
        try:
            return self._style_for(key)
        except Exception:  # noqa: BLE001 — a card must not fail a keypress
            return None

    # -- the grammar --------------------------------------------------------
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
        if target is not None:
            self.post_message(self.Chosen(self, target))

    def action_close(self) -> None:
        self.post_message(self.Closed(self))

    def on_click(self, event: Any) -> None:  # noqa: ANN001 — Textual event type
        """A click selects a row; a second click on it chooses (spec §10.4)."""
        offset = getattr(event, "y", None)
        if offset is None:
            return
        # The rows block starts two rows down (title, filter); the marker column
        # is 1 cell, so a click anywhere on a row's line lands on that row.
        body_y = self._row_block_top()
        position = offset - body_y
        if position < 0 or position >= len(self._rows):
            return
        event.stop()
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
            return 2

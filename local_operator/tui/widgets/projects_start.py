"""Start-session: which team/agent a project's next session boots as, and the card that asks.

Two halves, the same split every other projects surface keeps:

- :func:`start_targets` is PURE (the two registry catalogues in, rows out). The
  row set and its ORDER are the feature — teams first in registry order, then
  the agents, then the honest ``plain session`` fallback — and a pure list is
  the only place that order can be asserted without a terminal.
- :class:`StartPickerCard` is the surface: a floating ``Container`` on the
  overlay layer, built to the quick-send card's contract
  (:mod:`local_operator.tui.widgets.projects_send`) rather than to a second one
  — same ground, same one-line cropped rows, same picker grammar (type to
  filter, ``↑↓`` wrap, ``enter`` selects, ``esc`` closes). It carries section
  headers because the spec's §7.6.1 asks for them, and it is the one place this
  card differs from the send picker's flat row list.

WHY THE SECTIONS ARE PAINTED LINES AND NOT CHILD WIDGETS. The send card's
geometry is deterministic — chrome + the rows it paints, handed a row budget by
the page — and the page asserts against it. A section header that lives in the
layout as its own widget would add rows the budget does not know about, so
headers ride the same painted ``Text`` block as the rows and the card's height
stays ``chrome + painted lines``.

WHO BUILDS THE ROWS. The app does, from the SAME two projections the desktop
pane's new-chat picker reads (``desktop_profiles.team_catalogue`` /
``profile_catalogue``): this module never touches a registry, exactly as
``projects_send`` never touches a session store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rich.style import Style
from rich.text import Text
from textual.containers import Container
from textual.message import Message
from textual.widgets import Input, Static

#: The row that keeps the feature usable on an empty registry (spec §7.6.1/D7).
#: It is a ROW and not a footer because it is a real choice — the session it
#: makes is the one a plain ``/new`` makes — and a card that could only offer
#: rows from registries nobody has would be dead on a fresh install.
PLAIN_LABEL = "plain session (no team)"

#: What a section says when its registry is empty. Distinct from the filter
#: note below: "nothing is registered" is a fact about the install, "nothing
#: matched" is a keystroke to take back (the send card's N2 distinction).
NO_TEAM_NOTE = "no teams registered"
NO_AGENT_NOTE = "no agents registered"

#: The note when the FILTER admitted nothing at all.
NO_MATCH_NOTE = "no matching sessions — backspace to widen"

#: The in-flight sentence, painted where the reader is looking while the create
#: runs off the loop (spec §7.6.2's pending state).
PENDING_NOTE = "starting session …"

#: Section labels, in paint order: teams, agents, then the no-team fallback.
TEAM_SECTION = "teams"
AGENT_SECTION = "agents"
PLAIN_SECTION = "plain"

#: The card's fixed chrome, in rows: one padding row above and below, the title,
#: the rule under it, the filter, the note and the legend — the send card's own
#: arithmetic (:data:`projects_send.SEND_CARD_CHROME_ROWS`), because the two
#: cards are the same 2+5 rows of chrome around their rows.
START_CARD_CHROME_ROWS = 7

#: The most ROW lines painted at once. Section headers are additional painted
#: lines and are budgeted separately (they are not rows — nothing selects them).
START_CARD_ROW_CAP = 5

#: How many cells of a team's manager/roster clause are worth painting before
#: the row's own label absorbs the crop; a team with a 40-role roster must not
#: push its name off the card. The tail is the part that distinguishes two
#: similarly-named teams only weakly, so it yields first.
_DETAIL_CAP = 28


@dataclass(frozen=True)
class StartTarget:
    """One selectable row of the start picker.

    ``kind`` is what the boot path needs (``team`` / ``agent`` / ``plain``):
    the row's section is a paint concern and the kind is the create body's
    ``target.kind``, and keeping them equal by construction is why one field
    carries both. ``name`` is the ADDRESSABLE name (the team's name / the
    profile's resolved name) — empty for the plain row, which asks the create
    core for no attachment at all.
    """

    kind: str
    name: str
    label: str
    detail: str = ""

    def row_text(self) -> str:
        """One line per row: ``label · detail`` — also the filter's haystack.

        Subsequence matching runs over this, so ``detail`` is part of what a
        reader can type at: the spec asks for name AND description
        (§7.6.1), and a description the filter cannot see is a description
        nobody can search by.
        """
        return f"{self.label} · {self.detail}" if self.detail else self.label


def _team_roles(row: dict[str, Any]) -> int:
    """Member copies on a team roster (``Team.member_count``), manager excluded.

    Read off the catalogue dict rather than the model: the catalogue is what
    the picker is handed, and ``member_count`` sums ``count`` for the reason it
    documents — a ``reviewer x2`` slot is two members.
    """
    members = row.get("members")
    total = 0
    for member in members if isinstance(members, list) else []:
        if isinstance(member, dict):
            try:
                total += int(member.get("count") or 1)
            except (TypeError, ValueError):
                total += 1
    return total


def _team_detail(row: dict[str, Any]) -> str:
    roles = _team_roles(row)
    manager = str(row.get("manager") or "manager").strip() or "manager"
    return f"{manager} · {roles} role" + ("" if roles == 1 else "s")


def start_targets(
    *,
    teams: Iterable[dict[str, Any]],
    agents: Iterable[dict[str, Any]],
) -> list[StartTarget]:
    """The picker's rows, in the spec's order (§7.6.1).

    Teams in the CATALOGUE's order (the registries sort by case-folded name and
    this must not become a second sort), then the agents, then the plain row
    last — the page's honest-tail convention (``no team`` sorts last on the
    canvases for the same reason).

    A row whose name is empty is dropped: the create core addresses a target by
    name, and a row that could only be refused is a dead row (``send_targets``'
    rule, one surface over).
    """
    rows: list[StartTarget] = []
    for row in teams:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        label = str(row.get("label") or "").strip() or name
        rows.append(StartTarget(kind="team", name=name, label=label, detail=_team_detail(row)))
    for row in agents:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        label = str(row.get("label") or "").strip() or name
        rows.append(
            StartTarget(
                kind="agent",
                name=name,
                label=label,
                detail=str(row.get("description") or "").strip(),
            )
        )
    rows.append(StartTarget(kind="plain", name="", label=PLAIN_LABEL))
    return rows


def filter_start_targets(rows: Iterable[StartTarget], needle: str) -> list[StartTarget]:
    """Subsequence match over a row's text — the app's picker filter grammar.

    Subsequence rather than prefix, for the reason the shipped pickers record:
    a reader who remembers "review" should not have to type the row's leading
    characters to reach it.

    EVERY row is filtered uniformly, including the plain row. D7 keeps the
    plain row in the UNFILTERED list (it is what makes the feature work on an
    empty registry); a filter is the reader's own narrowing, and a row that
    survived it by rule would be a row the card kept for a reason the reader
    cannot see. Filtering it uniformly is also what makes the no-match note
    reachable — otherwise a mistyped filter would silently answer itself with
    `plain session`.
    """
    needle = needle.strip().casefold()
    if not needle:
        return list(rows)
    kept: list[StartTarget] = []
    for row in rows:
        haystack = row.row_text().casefold()
        position = 0
        for character in needle:
            position = haystack.find(character, position) + 1
            if position == 0:
                break
        else:
            kept.append(row)
    return kept


@dataclass(frozen=True)
class PaintedLine:
    """One painted line of the rows block.

    ``row`` is the index into the card's filtered rows for a selectable line and
    ``-1`` for a section header or an emptiness note — the two things ``↑↓``
    must skip and ``enter`` must never choose.
    """

    text: str
    row: int = -1
    header: bool = False


class StartPickerCard(Container):
    """The start picker: sections, a filter, and the app's picker grammar.

    Focusable and it takes focus on mount — the keys are ``↑``/``↓``/``enter``
    and printable characters, all of which the composer would otherwise swallow
    (the ask card's recorded reason). It FLOATS on the overlay layer with a
    Python-set offset so opening it reflows nothing under it, and its height is
    chrome plus the lines it is actually painting (the page hands the budget
    over through :meth:`set_available`), so every painted line is inside the
    card and the card is inside its ground.
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

        def __init__(self, card: "StartPickerCard", target: StartTarget) -> None:
            super().__init__()
            self.card = card
            self.target = target

    class Closed(Message):
        """The card was dismissed without a target."""

        def __init__(self, card: "StartPickerCard") -> None:
            super().__init__()
            self.card = card

    def __init__(
        self,
        rows: list[StartTarget],
        *,
        style_for: Any = None,
    ) -> None:
        super().__init__(classes="projects-send-card projects-start-card")
        self._all = list(rows)
        self._rows: list[StartTarget] = list(rows)
        self._index = 0
        self._style_for = style_for
        #: The painted window's first ROW; slides so ``_index`` is always
        #: painted (the send card's D1 rule, kept line-for-line).
        self._top = 0
        #: Rows the window is painting right now.
        self._visible = 0
        #: Lines the window is painting right now — rows PLUS the section
        #: headers among them. This is the other half of the card's height.
        self._visible_lines = 0
        #: The row budget the PAGE hands over (the ground's height in rows).
        self._available = START_CARD_CHROME_ROWS + START_CARD_ROW_CAP
        #: The in-flight/refusal sentence, or ``""``. While it carries the
        #: PENDING sentence the card refuses to choose (spec §7.6.2: a second
        #: pick would mint a second session).
        self._note = ""
        self._pending = False
        #: Whether the empty-registry notes are painted this frame; decided by
        #: `_visible_count` (they are spent out of the budget's leftovers) and
        #: read by `painted_lines`, so the block and the geometry agree.
        self._with_notes = True
        # The window is settled ONCE here so a card that has not been placed yet
        # (a unit test, or the instant between construction and the page's
        # `set_available`) still paints its own rows rather than an empty block:
        # `_repaint` recomputes it against the real ground as soon as the card
        # has one.
        self._sync_window()

    def compose(self):
        yield Static("start session", classes="projects-start-title projects-send-title")
        # The one rule every surface here uses under its title (design D2's
        # "one rule or box", in this sheet's separator vocabulary).
        yield Static("", id="projects-start-rule", classes="projects-send-rule")
        yield Input(placeholder="type to filter", id="projects-start-filter")
        yield Static("", id="projects-start-rows", classes="projects-send-rows")
        yield Static("", id="projects-start-note", classes="projects-send-note")
        yield Static(
            "type to filter · ↑↓ move · ↵ select · esc close",
            id="projects-start-legend",
            classes="projects-send-hints",
        )

    def on_mount(self) -> None:
        self._repaint()
        # The INPUT takes focus, not the card: with the card itself focused
        # every printable key died on it (the send card's QA round 1, Q2).
        try:
            self.query_one("#projects-start-filter", Input).focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    def on_resize(self) -> None:
        # The rule is cut to the card's measured width, which only exists once
        # layout has run; this repaint is where the first cut happens.
        self._repaint()

    # -- state --------------------------------------------------------------
    def set_pending(self) -> None:
        """Enter the pending state: the create is running off the loop.

        Choose is refused while it holds — a second ``enter`` would ask for a
        second session — and the sentence says what is happening, because a
        card that looks frozen during a runtime spawn reads as a hang.
        """
        self._pending = True
        self._note = PENDING_NOTE
        self._repaint()

    def show_refusal(self, sentence: str) -> None:
        """Leave the pending state with the reason, nothing half-created.

        The card stays up and the rows return: every refusal this card can meet
        (no registry, daemon down, the session cap) is answered by picking a
        different row or closing, so the surface must stay usable.
        """
        self._pending = False
        self._note = str(sentence or "").strip()
        self._repaint()

    @property
    def pending(self) -> bool:
        return self._pending

    # -- rows ---------------------------------------------------------------
    @property
    def rows(self) -> list[StartTarget]:
        """The rows the filter currently admits."""
        return list(self._rows)

    @property
    def index(self) -> int:
        return self._index

    def selected(self) -> StartTarget | None:
        return self._rows[self._index] if self._rows else None

    def painted_rows(self) -> list[str]:
        return [row.row_text() for row in self._rows]

    # -- the painted block (design D1: no unpainted selection) ---------------
    def set_available(self, rows: int) -> None:
        """Hand the card its row budget — the page's placement call.

        ``rows`` is the height of the ground the card floats over. The card
        spends it chrome-first and paints only the rows that fit, capped at
        :data:`START_CARD_ROW_CAP`; :meth:`_sync_window` slides the window so
        the selection is always painted. Idempotent, and safe to call before
        the card has composed (an unmounted card has no children to size —
        ``on_mount`` paints when they exist).
        """
        self._available = max(0, rows)
        self._repaint()

    def window_rows(self) -> list[StartTarget]:
        """The slice of the rows the card is painting right now."""
        return self._rows[self._top : self._top + self._visible]

    def painted_range(self) -> range:
        """The row positions currently painted (empty when the ground is too
        short for even one — the card then refuses to SELECT)."""
        return range(self._top, self._top + self._visible)

    def _section_of(self, row: StartTarget) -> str:
        return {
            "team": TEAM_SECTION,
            "agent": AGENT_SECTION,
        }.get(row.kind, PLAIN_SECTION)

    def _header_for(self, section: str) -> str:
        """The label a section paints. ``""`` for the plain row, which is its own label."""
        if section == TEAM_SECTION:
            return TEAM_SECTION
        if section == AGENT_SECTION:
            return AGENT_SECTION
        return ""

    def _empty_note(self, section: str) -> str:
        """What a section says when its REGISTRY is empty — "" otherwise.

        Keyed off ``self._all`` and never off the window: a section whose rows
        are merely filtered out or scrolled away is not an empty registry, and
        answering "no teams registered" there would be a false statement about
        the install.
        """
        if section == TEAM_SECTION:
            return NO_TEAM_NOTE if not any(row.kind == "team" for row in self._all) else ""
        if section == AGENT_SECTION:
            return NO_AGENT_NOTE if not any(row.kind == "agent" for row in self._all) else ""
        return ""

    def _query(self) -> str:
        try:
            return self.query_one("#projects-start-filter", Input).value.strip()
        except Exception:  # noqa: BLE001 — not composed yet
            return ""

    def painted_lines(self) -> list[PaintedLine]:
        """The block the card paints: section headers + one line per row.

        Sections always paint in the spec's order (teams, agents, plain), and a
        header is emitted with the first row of its section IN THE WINDOW — so
        a block scrolled past the top of a section still labels what it shows,
        which is the whole point of the header.

        An empty REGISTRY paints its header plus its note, and only when the
        block has room for them: the note is a fact about the install, so a
        filter cannot invent it, but it is ADVICE and never a reason to paint
        no rows at all — `_with_notes` is decided by the budget in
        :meth:`_visible_count` and shared here so the two cannot disagree.
        While a filter IS typed a section with no surviving rows paints nothing
        — a header over nothing is a section that is not there.
        """
        return self._lines_for(self._visible, self._top, notes=self._with_notes)

    def _lines_for(self, count: int, top: int, *, notes: bool) -> list[PaintedLine]:
        """The painted lines for a window of ``count`` rows starting at ``top``.

        ONE builder for the block and for the geometry: :meth:`_visible_count`
        spends its budget through this, so the height the card reserves and the
        lines it paints cannot disagree (the send card's D1 property, one
        header worse). ``notes`` carries the empty-registry lines; it is off
        when they do not fit, so they can never starve the rows they annotate.
        """
        start = max(0, top)
        window = [
            (index, row) for index, row in enumerate(self._rows[start : start + count], start)
        ]
        lines: list[PaintedLine] = []
        for section in (TEAM_SECTION, AGENT_SECTION, PLAIN_SECTION):
            in_section = [entry for entry in window if self._section_of(entry[1]) == section]
            if in_section:
                header = self._header_for(section)
                if header:
                    lines.append(PaintedLine(header, header=True))
                for index, row in in_section:
                    lines.append(PaintedLine(row.row_text(), row=index))
                continue
            if not notes:
                continue
            note = self._empty_note(section)
            if note:
                header = self._header_for(section)
                if header:
                    lines.append(PaintedLine(header, header=True))
                lines.append(PaintedLine(note))
        return lines

    def _visible_count(self) -> tuple[int, int]:
        """``(rows, lines)`` the window can paint in the ground it was given.

        Chrome first, then whole ROWS until either the row cap or the budget is
        spent — measured by the SAME builder that paints them, so a header can
        never be left outside the card while its row is inside. The
        empty-registry notes are spent only out of what is LEFT after the rows:
        charging them to the budget first (the obvious reading, and the first
        version) let two notes starve the plain row off a short ground, which is
        the one row the feature cannot do without.
        """
        room = max(0, self._available - START_CARD_CHROME_ROWS)
        if room <= 0 or not self._rows:
            return 0, 0
        count = min(len(self._rows), START_CARD_ROW_CAP)
        while count > 0 and len(self._lines_for(count, self._top, notes=False)) > room:
            count -= 1
        if count <= 0:
            return 0, 0
        rows_only = self._lines_for(count, self._top, notes=False)
        notes = bool(self._notes_wanted())
        self._with_notes = notes
        if notes:
            full = self._lines_for(count, self._top, notes=True)
            if len(full) <= room:
                return count, len(full)
            self._with_notes = False
        return count, len(rows_only)

    def _notes_wanted(self) -> bool:
        """Whether empty-registry notes are even candidates this frame."""
        return not self._query()

    def _sync_window(self) -> None:
        """Set the painted count and slide the window so ``_index`` is in it."""
        self._visible, self._visible_lines = self._visible_count()
        if self._visible <= 0:
            self._top = 0
            self._visible_lines = 0
            # No room for a row means no room for the block at all: the notes
            # ride the same budget, so they go too (the body's height is pinned
            # to `_visible_lines` and a note painted past it would spill).
            self._with_notes = False
            return
        top = min(self._top, max(0, len(self._rows) - self._visible))
        if self._index < top:
            top = self._index
        elif self._index >= top + self._visible:
            top = self._index - self._visible + 1
        self._top = max(0, top)

    def rows_text(self) -> Text:
        """The painted block as one rich ``Text``.

        ONE display line per entry, cropped rather than wrapped: the block is
        sized in logical rows, so a label long enough to wrap would paint two
        lines for one row and push the tail out of the card while
        :meth:`painted_range` still advertised it (the send card's QA round 3,
        Q3-2 — the same defect, the same fix). The LABEL absorbs the crop: the
        detail clause is what yields first.
        """
        width = self.content_size.width
        lines: list[Text] = []
        for line in self.painted_lines():
            if line.header:
                lines.append(self._header_text(line.text))
                continue
            selected = line.row == self._index
            lines.append(self._row_text(line, selected, width))
        combined = Text(no_wrap=True)
        for index, line in enumerate(lines):
            if index:
                combined.append("\n")
            combined.append_text(line)
        return combined

    def _header_text(self, label: str) -> Text:
        """A section header: the page's rule-row vocabulary (muted, no row band)."""
        text = Text(no_wrap=True)
        text.append(label, style=self._ink("muted") or Style())
        return text

    def _row_text(self, line: PaintedLine, selected: bool, width: int) -> Text:
        row = self._rows[line.row]
        text = Text(no_wrap=True)
        text.append("▸" if selected else " ", style=self._ink("cursor") if selected else Style())
        text.append(" ")
        detail = (
            row.detail if len(row.detail) <= _DETAIL_CAP else f"{row.detail[:_DETAIL_CAP - 1]}…"
        )
        tail = f"  · {detail}" if detail else ""
        budget = width - 2 - len(tail)
        label = row.label
        if width > 0 and len(label) > budget:
            label = f"{label[: budget - 1]}…" if budget > 0 else ""
        text.append(label)
        if detail:
            text.append("  · ")
            text.append(detail)
        if width > 0 and len(text) < width:
            if selected:
                text.pad_right(width - len(text))
        if selected:
            text.stylize(self._ink("row_selected") or Style(), 0, len(text))
        return text

    def _repaint(self) -> None:
        # The window is settled FIRST and unconditionally: it is the card's own
        # state (which rows it is showing) and does not need a composed widget
        # to be correct. Only the PAINTED children need one.
        self._sync_window()
        try:
            body = self.query_one("#projects-start-rows", Static)
            note = self.query_one("#projects-start-note", Static)
            rule = self.query_one("#projects-start-rule", Static)
        except Exception:  # noqa: BLE001 — not composed yet; on_mount paints
            return
        # The rows block is exactly as tall as the block paints: the card's
        # height is chrome + painted lines and NOTHING else (the deterministic
        # geometry the page asserts against its ground).
        body.styles.height = self._visible_lines
        # The empty-registry NOTES still paint when the window itself is empty
        # (a ground too short for one row spills nothing, but the install's own
        # fact is not the window's to hide) — so the guard is on the painted
        # block, not on the row list.
        body.update(self.rows_text() if self._visible_lines else Text(""))
        if self._note:
            # The pending/refusal sentence OUTRANKS the filter note: it is the
            # answer to the last thing the reader did.
            note.update(Text(self._note, style=self._ink("muted")))
        elif not self._rows:
            # The filter itself is a sibling the same composer mounted, so it is
            # present whenever the rows block is.
            query = self.query_one("#projects-start-filter", Input).value.strip()
            note.update(Text(NO_MATCH_NOTE if query else "", style=self._ink("muted")))
        else:
            hidden = len(self._rows) - self._visible
            note.update(Text(f"+{hidden} more" if hidden > 0 else "", style=self._ink("muted")))
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
        """``enter`` from the filter — the card's common choose path."""
        event.stop()
        self.action_choose()

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._rows = filter_start_targets(self._all, event.value)
        self._index = 0
        self._top = 0
        self._repaint()

    def action_move(self, delta: int) -> None:
        """``↑``/``↓`` — WRAP, the shipped picker convention."""
        if not self._rows or self._pending:
            return
        self._index = (self._index + delta) % len(self._rows)
        self._repaint()

    def action_choose(self) -> None:
        target = self.selected()
        # "No unpainted selection" (the send card's D1 rule): `enter` can only
        # pick a row that is on screen, and never while a create is in flight.
        if self._pending or target is None or self._index not in self.painted_range():
            return
        self.post_message(self.Chosen(self, target))

    def action_close(self) -> None:
        # Closing during the pending state is allowed and honest: the create
        # runs off the loop and its receipt lands on the page (see the app's
        # handler) — a card that trapped the reader for a spawn's seconds would
        # be worse than the receipt being one row elsewhere.
        self.post_message(self.Closed(self))

    def on_click(self, event: Any) -> None:  # noqa: ANN001 — Textual event type
        """A click selects a row; a second click on it chooses (spec §10.4).

        Positions are WINDOW-relative and map through the PAINTED LINES, so a
        click on a section header or the note hits no row — the old index
        arithmetic treated every painted line as a row (the send card's D1
        class, one header worse).
        """
        if self._pending:
            return
        offset = getattr(event, "y", None)
        if offset is None:
            return
        body_y = self._row_block_top()
        line_index = offset - body_y
        lines = self.painted_lines()
        if line_index < 0 or line_index >= len(lines):
            return
        line = lines[line_index]
        if line.row < 0:
            return
        event.stop()
        if line.row == self._index:
            self.action_choose()
            return
        self._index = line.row
        self._repaint()

    def _row_block_top(self) -> int:
        """The first row of the rows block, in card-relative coordinates."""
        try:
            block = self.query_one("#projects-start-rows", Static)
            return block.region.y - self.region.y
        except Exception:  # noqa: BLE001 — before layout there is no region
            # padding row + title + rule + filter
            return 3

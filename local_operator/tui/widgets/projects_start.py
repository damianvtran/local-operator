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
#:
#: THE WORDS ARE THE PARITY SURFACE'S, not a registry noun (UX review round 1,
#: U8): the desktop's own picker says `Nothing in particular` with “A plain
#: session; assign a team or agent later.”, which answers “why would I pick
#: this?” where `plain session (no team)` read as a degenerate team.
PLAIN_LABEL = "Nothing in particular"

#: The plain row's one-line detail — the desktop's second sentence, so the row
#: teaches as well as names (UX U8).
PLAIN_DETAIL = "a plain session; assign a team or agent later"

#: What a section says when its registry is empty. Distinct from the filter
#: note below: "nothing is registered" is a fact about the install, "nothing
#: matched" is a keystroke to take back (the send card's N2 distinction).
NO_TEAM_NOTE = "no teams registered"
NO_AGENT_NOTE = "no agents registered"

#: The note when the FILTER admitted nothing at all. It names the OBJECTS the
#: card actually lists: the first draft said `sessions`, which is the send
#: card's noun and not this card's — there is no session list here, and an
#: install may have no sessions at all (design review round 1, D3).
NO_MATCH_NOTE = "no matching targets — backspace to widen"

#: The note when the ground is too short to paint even one row. The `+N more`
#: counter would otherwise imply rows just above the fold while `enter` refuses
#: in silence (UX review round 1, U6 — measured at 60×20).
NO_ROOM_NOTE = "no room for a row here — make the window taller"

#: The in-flight sentence, painted where the reader is looking while the create
#: runs off the loop (spec §7.6.2's pending state).
PENDING_NOTE = "starting session …"

#: The marker a REFUSAL wears. The refusal shares the note slot with the pending
#: sentence and the `+N more` counter, so it needs a signal that survives a
#: colourless terminal (UX review round 1, U3 — `warning` on the card's own
#: ground clears AA at 7.09:1 dark / 4.78:1 light, pinned by
#: `test_the_warning_ink_clears_aa_on_the_card_ground`, but ink alone left the
#: three states reading alike).
REFUSAL_MARK = "! "

#: Section labels, in paint order: teams, agents, then the no-team fallback —
#: and the fallback has a header of its OWN (UX review round 1, U5: without one
#: it was painted under whichever header preceded it and read as a member of
#: that section, or as a team on a fresh install). The label is the page's own
#: noun for the bucket it is not in.
TEAM_SECTION = "teams"
AGENT_SECTION = "agents"
PLAIN_SECTION = "plain"
PLAIN_HEADER = "no team"

#: The card's fixed chrome, in rows: the padding row above and below, the
#: title, the SUBJECT line under it, the rule, the filter, the note and the
#: legend. One more row than the send card's 7 because this card has to say
#: what it is about to do and to which project (UX review round 1, U2), and
#: that sentence belongs to the card rather than to a note slot the reader is
#: also using for a refusal.
START_CARD_CHROME_ROWS = 8

#: The most ROW lines painted at once. Section headers are additional painted
#: lines and are budgeted separately (they are not rows — nothing selects
#: them). The cap is the send card's 5 raised: that cap is right for a list of
#: a project's own sessions, while this card is the whole team/agent catalogue
#: and the row the feature exists for (`the plain row`) sorts LAST — at 150×40
#: the ground holds 25 rows and the card was showing 5 of 14 (design review
#: round 1, D5).
START_CARD_ROW_CAP = 8

#: The fewest cells a detail clause is worth painting. Below this the clause is
#: all ellipsis and the row is better off as its bare name.
_MIN_DETAIL_CELLS = 8


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
    """The picker's rows: the no-team row first, then the spec's two registries.

    Teams in the CATALOGUE's order (the registries sort by case-folded name and
    this must not become a second sort), then the agents. The plain row LEADS,
    against §7.6.1's "last" — the reasons are on the construct below, and they
    are the parity surface's own layout.

    A row whose name is empty is dropped: the create core addresses a target by
    name, and a row that could only be refused is a dead row (``send_targets``'
    rule, one surface over).
    """
    # THE PLAIN ROW COMES FIRST, and its own section label above it (UX review
    # round 1, U4/U5 + design D5). Two of those findings pull the same way and
    # the desktop parity surface settles it: `project-start-session.tsx` lists
    # `Nothing in particular` FIRST and opens with it selected. The spec's
    # §7.6.1 put it last, which cost three things at once — opening the card on
    # the first TEAM made two keystrokes attach a roster by accident (U4), the
    # row the feature exists for sat behind a 5-row window on every populated
    # install (D5), and with the selection moved to it the WINDOW slid to the
    # tail, so the card opened on `agents` with the teams entirely off-screen
    # (measured on this fix's own frame). First, it needs no scroll to be seen,
    # it is what an unaimed `enter` takes, and the two registries keep the
    # spec's own order below it. The row's own header keeps it from reading as
    # a member of whichever section it abuts.
    rows: list[StartTarget] = [
        StartTarget(kind="plain", name="", label=PLAIN_LABEL, detail=PLAIN_DETAIL)
    ]
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
    return rows


def match_spans(needle: str, haystack: str) -> list[tuple[int, int]]:
    """The cell ranges a subsequence match of ``needle`` covers in ``haystack``.

    ONE matcher for the filter and the paint: :func:`filter_start_targets`
    admits a row with it and the card paints the cells it returns, so a row can
    never be admitted for a reason the reader cannot be shown (design review
    round 1, D2). Case-folded, and the returned ranges are merged so a run of
    adjacent characters is one span rather than one per character.
    """
    wanted = needle.strip().casefold()
    if not wanted:
        return []
    hay = haystack.casefold()
    spans: list[tuple[int, int]] = []
    position = 0
    for character in wanted:
        found = hay.find(character, position)
        if found < 0:
            return []
        if spans and spans[-1][1] == found:
            spans[-1] = (spans[-1][0], found + 1)
        else:
            spans.append((found, found + 1))
        position = found + 1
    return spans


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

    The admission and the highlight share :func:`match_spans`; the ranges it
    returns are taken over the row's full text, and the painter re-derives them
    over what it actually paints.
    """
    if not needle.strip():
        return list(rows)
    return [row for row in rows if match_spans(needle, row.row_text())]


def subject_line(project: str) -> str:
    """What this card is about to do, in the card's own voice (UX U2).

    The desktop parity dialog says it in a sentence ("The session starts now,
    links itself to this project, and opens with a prompt you can edit before
    sending"); this card said none of it, so `s` then `enter` began a paid turn
    for a named project that the reader was never told about. Two deliberate
    differences from the desktop's wording, both because they are TRUE here:
    the kickoff turn is SENT rather than left as an editable draft
    (``PromptErrand`` is a user turn), and the reader is TAKEN to the session
    (the hand-off is the app's own ``/resume`` path).

    Order is the riskiest fact first, because the narrow terminals crop the
    tail of this line: what starts, what it attaches to, where you end up.
    """
    if not project:
        return "starts a turn · you go there"
    return f"starts a turn · links to {project} · you go there"


def visible_match(needle: str, painted: str) -> list[tuple[int, int]]:
    """The spans of the longest PREFIX of ``needle`` that ``painted`` carries.

    The mark is what says WHY a row survived the filter, so it has to be derived
    from what the reader can actually see: a crop that cuts the match leaves the
    re-derived subsequence with nothing to find, and the row paints unmarked
    (QA round 2, Q4 — 10 of 51 admitted rows at 144 cells, 25 of 51 at 92, 35 of
    51 at 66). Matching the longest painted prefix keeps the mark honest about a
    partial match: the cells that did survive are marked, the trailing ellipsis
    already says the rest is cropped, and the caller's crop is what puts the
    match's first cell on screen in the first place (:func:`_fit_detail`).

    It still goes through :func:`match_spans`, so the admission and the mark
    keep one matcher's rules — order, case-folding, merged runs — and the only
    difference between them is the text they are handed.
    """
    wanted = needle.strip().casefold()
    if not wanted:
        return []
    haystack = painted.casefold()
    position = 0
    kept = 0
    for character in wanted:
        found = haystack.find(character, position)
        if found < 0:
            break
        kept += 1
        position = found + 1
    return match_spans(wanted[:kept], painted) if kept else []


def _fit_detail(detail: str, room: int, match: tuple[int, int] | None) -> str:
    """``detail`` cropped to ``room`` cells with the admitted match PAINTED.

    A plain left crop was design round 1's D2 complaint read literally, and the
    slide it introduced looked only at the match's START — so a match that began
    just inside the crop and ended past it was painted in half and marked not at
    all (QA round 2, Q4: `trade-offs` on the Architect row painted
    `…with trade-o…`, and `severity` on the Reviewer row painted nothing at
    all). The rule reads the match's whole RUN now: a window that can hold it is
    slid to COVER it, and one that cannot still paints its FIRST matched cell,
    so a row the filter admitted never paints unmarked.

    ``match`` is that run in ``detail`` coordinates — ``(first, last)`` — or
    ``None`` for a row admitted on its label alone, where the label is painted
    whole and needs no slide.
    """
    if room <= 1:
        return ""
    if len(detail) <= room or match is None:
        return detail[: room - 1] + "…"
    first, last = match
    # A window that can hold the whole run pays for the ellipsis on each side of
    # it; one that cannot still has to carry the run's first cell.
    covering = last - first + 2 <= room
    body = room - 2 if covering else room - 1
    start = max(0, min(first - 2, len(detail) - body))
    if covering:
        start = max(start, min(last - body, len(detail) - body))
    head = "…" if start > 0 else ""
    tail = "…" if start + body < len(detail) else ""
    return (head + detail[start : start + body] + tail)[:room]


def fit_row(label: str, detail: str, width: int, *, query: str = "") -> tuple[str, str]:
    """The ``(label, tail)`` one row paints in ``width`` cells (marker excluded).

    Pure, so the crop rule is pinned without a terminal. THE DETAIL ABSORBS THE
    CROP, and that is the family's rule read the right way round for this card:
    the send card lets its LABEL absorb because its label is a conversation
    title and its tail is the short id and state chip that tell two rows apart
    — here the label is the NAME the reader typed and the create resolves, and
    the detail is descriptive prose, so the prose yields first (design review
    round 1, D1: a fixed 28-cell cap cropped it at every width, leaving 104 of
    142 cells empty at 150×40). The label is cropped only when even a bare name
    cannot fit, which is the last resort rather than the rule.

    The crop is told the filter's whole RUN over the detail, not just where the
    match starts: that is what lets it COVER the match rather than cut it in
    half (QA round 2, Q4).
    """
    if width <= 0:
        return "", ""
    if len(label) > width:
        return (f"{label[: width - 1]}…" if width > 1 else ""), ""
    separator = "  · "
    room = width - len(label) - len(separator)
    if not detail or room < _MIN_DETAIL_CELLS:
        return label, ""
    run: tuple[int, int] | None = None
    if query:
        spans = match_spans(query, detail)
        if spans:
            run = (spans[0][0], spans[-1][1])
    return label, separator + _fit_detail(detail, room, run)


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
        project: str = "",
        style_for: Any = None,
    ) -> None:
        # ONE class: the family's. `.projects-start-card` used to ride along and
        # had no rule anywhere in the sheet — an inert second name that the
        # tcss comment argues against (agent review round 1, F8).
        super().__init__(classes="projects-send-card")
        self._all = list(rows)
        self._rows: list[StartTarget] = list(rows)
        #: The project the new session is for: the subject line names it, and
        #: the create/link/kickoff all happen against it (UX U2).
        self._project = project
        self._style_for = style_for
        #: The plain row is the DEFAULT (UX review round 1, U4): `s` then
        #: `enter` is the gesture a reader makes to have a session, and row 0
        #: used to be the first TEAM — so two keystrokes committed a roster's
        #: worth of tokens by accident. The desktop parity dialog defaults the
        #: same way (`START_SESSION_PLAIN`).
        self._index = self._default_index()
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
        #: The filter's needle (see :meth:`_query`).
        self._query_text = ""
        #: What KIND of sentence ``_note`` carries — ``""``, ``"pending"`` or
        #: ``"refusal"``. The note slot does several jobs and they must not read
        #: alike (UX review round 1, U3).
        self._note_kind = ""
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
        # WHAT THIS IS ABOUT TO DO, AND FOR WHICH PROJECT (UX U2). The desktop
        # parity dialog says it in a sentence; this card said none of it, so `s`
        # then `enter` started a paid turn for a named project the reader was
        # never told about. The `id` is a semantic hook rather than a selector:
        # the line wears the family note's ink through `projects-send-note`, the
        # way `projects-start-title` wears `projects-send-title` (agent review
        # round 2, N5).
        yield Static(
            subject_line(self._project),
            id="projects-start-subject",
            classes="projects-start-subject projects-send-note",
        )
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
        self._note_kind = "pending"
        self._repaint()

    def show_refusal(self, sentence: str) -> None:
        """Leave the pending state with the reason, nothing half-created.

        The card stays up and the rows return: every refusal this card can meet
        (no registry, daemon down, the session cap) is answered by picking a
        different row or closing, so the surface must stay usable.
        """
        self._pending = False
        self._note = str(sentence or "").strip()
        self._note_kind = "refusal" if self._note else ""
        self._repaint()

    @property
    def pending(self) -> bool:
        return self._pending

    @property
    def subject(self) -> str:
        """The line under the title: what this card will do, and for which
        project (UX U2). Public so the surface tests assert the COPY rather
        than the widget."""
        return subject_line(self._project)

    # -- rows ---------------------------------------------------------------
    def _default_index(self) -> int:
        """The row the card opens on: the plain one, else the first (UX U4)."""
        for position, row in enumerate(self._rows):
            if row.kind == "plain":
                return position
        return 0

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
        """The label a section paints — the plain row included (UX review U5):
        without a header of its own the plain row was painted directly under
        whichever header preceded it, and on a fresh install it read as a TEAM.
        The label is the page's own noun for the bucket it is not in.
        """
        if section == TEAM_SECTION:
            return TEAM_SECTION
        if section == AGENT_SECTION:
            return AGENT_SECTION
        return PLAIN_HEADER

    def empty_registry_note(self) -> str:
        """The install's emptiness, as one sentence for the NOTE slot.

        The same fact the in-block section notes carry, and it is needed twice
        because they are the first thing the budget drops on a short ground:
        measured at the 60×24 fixture the block lost BOTH the `teams` header and
        “no teams registered”, so a reader could not tell an install with no
        teams from a card that simply does not show them (design review round 1,
        D4). Empty means the registries are both populated (or a filter is up).
        """
        if self._query():
            return ""
        missing: list[str] = []
        if not any(row.kind == "team" for row in self._all):
            missing.append(NO_TEAM_NOTE)
        if not any(row.kind == "agent" for row in self._all):
            missing.append(NO_AGENT_NOTE)
        return " · ".join(missing)

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
        """The filter's current needle.

        CARD STATE, not a widget read: the block, the match mark and the note
        are all derived from it, and the card's own rule is that its painted
        state is correct without a composed widget (``on_input_changed`` is the
        one writer, so the two cannot drift).
        """
        return self._query_text

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
        for section in (PLAIN_SECTION, TEAM_SECTION, AGENT_SECTION):
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
        Q3-2 — the same defect, the same fix). What absorbs the crop is
        :func:`fit_row`'s decision, made against the MEASURED width.
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
        """One painted row: ``▸ label  · detail``, cropped to the card's width.

        The crop is :func:`fit_row`'s (pure, so it is pinned without a
        terminal). When a FILTER is active the cells it matched wear the match
        style and the detail is cropped AROUND its match, so a row admitted for
        something deeper in its description shows why it survived (design review
        round 1, D2).
        """
        row = self._rows[line.row]
        marker = 2
        label, tail = fit_row(row.label, row.detail, max(0, width - marker), query=self._query())
        text = Text(no_wrap=True)
        text.append("▸" if selected else " ", style=self._ink("cursor") if selected else Style())
        text.append(" ")
        text.append(label)
        if tail:
            text.append(tail)
        if selected and width > 0 and len(text) < width:
            text.pad_right(width - len(text))
        if selected:
            text.stylize(self._ink("row_selected") or Style(), 0, len(text))
        # The mark is derived over what is PAINTED, and only as much of the
        # needle as the paint actually carries: a highlight on a cell the crop
        # removed would be a claim about something the reader cannot see, and
        # re-deriving the FULL needle over a cropped row finds nothing at all
        # (QA round 2, Q4). It is COMPOSED with the row's own style (`+` gives
        # the left side precedence, and the two set different attributes), so a
        # matched cell on the SELECTED row keeps the selection band under the
        # mark rather than punching a hole in it.
        query = self._query()
        if query:
            ink = self._ink("match") or Style()
            under = (self._ink("row_selected") or Style()) if selected else Style()
            for start, end in visible_match(query, text.plain):
                text.stylize(under + ink, start, end)
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
        body.update(self.rows_text() if self._visible_lines else Text(""))
        note.update(self._note_text())
        width = self.content_size.width
        rule.update(Text("─" * width if width > 0 else "", style=self._ink("dim")))

    def _note_text(self) -> Text:
        """The card's one-line note, in priority order and in distinct inks.

        FIVE states share this slot and they must not read alike (UX review
        round 1, U3/D4): the in-flight sentence, a refusal, the install's
        emptiness, the window's remainder, and nothing. The refusal takes the
        app's own refusal ink PLUS a marker (the ink is `warning`, which the
        palette pin measures at 7.09:1 dark / 4.78:1 light on this card's
        ground — `danger` is 3.9:1 in light and would fail the gate); the
        emptiness fact outranks the counter, because a reader who cannot see
        whether a section exists is worse off than one who cannot see how many
        rows are below the fold.
        """
        if self._note:
            if self._note_kind == "refusal":
                return Text(REFUSAL_MARK + self._note, style=self._ink("refusal"))
            return Text(self._note, style=self._ink("muted"))
        if not self._rows:
            return Text(NO_MATCH_NOTE if self._query() else "", style=self._ink("muted"))
        # The install's own emptiness outranks the window's arithmetic: a reader
        # who cannot tell whether a section EXISTS is worse off than one who
        # cannot see how many rows are below the fold, and the no-room sentence
        # is about a card they can already see is tiny (design review round 1,
        # D4 ahead of UX U6).
        empty = self.empty_registry_note()
        if empty and not self._with_notes:
            return Text(empty, style=self._ink("muted"))
        if self._visible <= 0:
            # Rows exist and NONE fits: `+N more` would promise rows just above
            # the fold while `enter` refuses in silence (UX review round 1, U6).
            return Text(NO_ROOM_NOTE, style=self._ink("muted"))
        hidden = len(self._rows) - self._visible
        return Text(f"+{hidden} more" if hidden > 0 else "", style=self._ink("muted"))

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
        self._query_text = event.value.strip()
        self._rows = filter_start_targets(self._all, event.value)
        self._index = self._default_index()
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
            # padding row + title + SUBJECT + rule + filter (agent review round
            # 2, N4: the subject line is new chrome, and a stale constant here is
            # what a later reader would trust).
            return 5

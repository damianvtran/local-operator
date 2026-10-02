"""The start-session picker (P5b): its rows, its card, and the flow over the page.

Three layers, split the way the rest of the projects surfaces split them:

* the PURE half — :func:`start_targets`, :func:`filter_start_targets`,
  :func:`match_spans`, :func:`fit_row` and :func:`subject_line` — is pinned
  without a terminal, because the rows' order, the crop rule and the card's
  copy are the feature and none of them needs a screen to be wrong;
* the card's geometry and grammar are driven over the real page (the app is the
  only host that loads the shipped stylesheet);
* the boot flow is driven through the app's own handler with the creation core
  and the runtime engagement REPLACED, because the real ones spawn a detached
  runtime and write a session directory — what this file can honestly assert is
  the ORDER, the cancellability and the refusal handling.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from rich.style import Style

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.projects_start import (
    AGENT_SECTION,
    NO_AGENT_NOTE,
    NO_MATCH_NOTE,
    NO_ROOM_NOTE,
    NO_TEAM_NOTE,
    PLAIN_DETAIL,
    PLAIN_HEADER,
    PLAIN_LABEL,
    REFUSAL_MARK,
    START_CARD_CHROME_ROWS,
    START_CARD_ROW_CAP,
    TEAM_SECTION,
    StartPickerCard,
    StartTarget,
    filter_start_targets,
    fit_row,
    match_spans,
    start_targets,
    subject_line,
    visible_match,
)
from local_operator.tui.widgets.projects_view import (
    ProjectsViewStartRequested,
    _style_resolver,
)
from local_operator.tui.widgets.subagent_view import HintButton
from tests.unit.tui.test_projects_view import (  # noqa: E402
    _boot,
    _factory,
    _open,
    _ProjectSession,
    _registry,
)

#: A real-looking mint: the store validates ids as 12 hex characters, so a
#: readable placeholder would be refused by the LINK rather than by the test.
STARTED_ID = "ab12cd34ef56"


def _team(
    name: str, *, manager: str = "manager", counts: tuple[int, ...] = (1, 1)
) -> dict[str, Any]:
    return {
        "name": name,
        "label": name.title(),
        "manager": manager,
        "members": [{"role": f"r{i}", "count": c, "kind": "agent"} for i, c in enumerate(counts)],
    }


def _agent(name: str, description: str = "") -> dict[str, Any]:
    return {"name": name, "label": name.title(), "description": description}


def _start_rows() -> list[StartTarget]:
    return start_targets(
        teams=[_team("core"), _team("lopdev")],
        agents=[_agent("coder", "implements one bounded slice end to end")],
    )


def _row_line(label: str, detail: str = "") -> str:
    """The painted text of a row, in the card's own ``label · detail`` grammar."""
    return f"{label} · {detail}" if detail else label


def _filter(card: StartPickerCard, needle: str) -> None:
    """Type ``needle`` into the card's filter, without a terminal.

    The three things ``on_input_changed`` moves, in the same order: the query
    the paint reads, the rows the filter admits, and the selection the card
    opens on. Duplicating them here is the cost of asserting the card's PAINT
    without mounting it — the alternative is a terminal for every note and
    mark, which the surrounding pure tests exist to avoid.
    """
    card._query_text = needle.strip()
    card._rows = filter_start_targets(card._all, needle)
    card._index = card._default_index()
    card._top = 0


# -- the pure half ----------------------------------------------------------


def test_the_plain_row_leads_and_the_registries_keep_their_order() -> None:
    """The plain row FIRST (U4/U5/D5, the parity surface's layout), then the
    spec's teams and agents in their own order."""
    rows = start_targets(
        teams=[_team("lopdev", counts=(1, 2)), _team("core")],
        agents=[_agent("coder", "implements"), _agent("reviewer", "reviews")],
    )
    assert [row.kind for row in rows] == ["plain", "team", "team", "agent", "agent"]
    assert [row.label for row in rows] == [PLAIN_LABEL, "Lopdev", "Core", "Coder", "Reviewer"]
    # The catalogue's order is preserved — this is NOT a second sort.
    assert [row.name for row in rows][1:3] == ["lopdev", "core"]
    # A team row names its manager and how many member copies it runs; the
    # manager is excluded from the count (`Team.member_count`).
    assert rows[1].detail == "manager · 3 roles"
    assert rows[2].detail == "manager · 2 roles"
    # An agent row carries the registry's one-line description.
    assert rows[3].detail == "implements"
    # The plain row leads, asks the create core for no attachment at all, and
    # its copy is the parity surface's (U8).
    assert rows[0].name == ""
    assert rows[0].detail == PLAIN_DETAIL


def test_an_empty_registry_still_offers_a_plain_session() -> None:
    """D7: the feature has to work on a fresh install with nothing registered."""
    rows = start_targets(teams=[], agents=[])
    assert rows == [StartTarget(kind="plain", name="", label=PLAIN_LABEL, detail=PLAIN_DETAIL)]
    card = StartPickerCard(rows)
    # Both sections still say what is missing rather than vanishing — and the
    # plain row has a header of its OWN since U5, so it cannot read as a team.
    painted = [line.text for line in card.painted_lines()]
    assert NO_TEAM_NOTE in painted and NO_AGENT_NOTE in painted
    assert _row_line(PLAIN_LABEL, PLAIN_DETAIL) in painted
    assert PLAIN_HEADER in painted
    assert painted.index(PLAIN_HEADER) < painted.index(_row_line(PLAIN_LABEL, PLAIN_DETAIL))


def test_rows_without_a_name_are_dropped() -> None:
    """A target is addressed BY NAME, so a nameless row could only be refused."""
    rows = start_targets(teams=[{"name": "  "}, _team("core")], agents=[{}])
    assert [row.name for row in rows] == ["", "core"]


def test_the_filter_is_a_subsequence_over_the_name_and_the_description() -> None:
    rows = start_targets(
        teams=[_team("lopdev")],
        agents=[_agent("reviewer", "adversarial reading of diffs")],
    )
    # Subsequence, not prefix: "rvw" reaches the reviewer.
    assert [row.name for row in filter_start_targets(rows, "rvw")] == ["reviewer"]
    # The DESCRIPTION is searched too — a description the filter cannot see is
    # one nobody can search by.
    assert [row.name for row in filter_start_targets(rows, "adversarial")] == ["reviewer"]
    assert filter_start_targets(rows, "") == rows


def test_the_filter_narrows_the_plain_row_too() -> None:
    """A filter the reader typed must not be answered by a row kept for them."""
    rows = start_targets(teams=[_team("lopdev")], agents=[])
    assert [row.kind for row in filter_start_targets(rows, "zzz")] == []
    assert [row.kind for row in filter_start_targets(rows, "plain")] == ["plain"]


def test_match_spans_are_the_cells_the_filter_looked_at() -> None:
    """One matcher behind the admission and the paint (design D2)."""
    assert match_spans("core", "Core  · manager · 2 roles") == [(0, 4)]
    # A run of adjacent characters is ONE span, not one per character.
    assert match_spans("abc", "xabcy") == [(1, 4)]
    # Scattered characters are separate spans, and the order is the reader's.
    assert match_spans("cr", "Core") == [(0, 1), (2, 3)]
    # Casefolded both ways, and a miss is no spans at all.
    assert match_spans("CORE", "core") == [(0, 4)]
    assert match_spans("zzz", "core") == []
    # An empty needle matches nothing rather than everything: the caller's
    # "no filter" case is a separate question.
    assert match_spans("   ", "core") == []


def test_the_detail_absorbs_the_crop_and_the_full_width_is_used() -> None:
    """Design D1: the label is the NAME, so the prose yields first.

    A fixed 28-cell cap cropped the detail at every width and left 104 of 142
    cells empty at 150×40; the rule is the row's MEASURED width.
    """
    label = "Copy Reviewer"
    detail = (
        "Review of written copy before it ships: user-visible product copy and prose "
        "for a general reader, on comprehension, tone, claim support and AI-isms"
    )
    wide_label, wide_tail = fit_row(label, detail, 140)
    assert wide_label == label
    # The row FILLS the width it was given: the only slack is the cell the
    # ellipsis stands in for, and the prose is what absorbed the crop.
    assert len(wide_label) + len(wide_tail) == 140
    assert wide_tail.endswith("…")
    # Twice the width, twice as much prose — the crop tracks the MEASURED width
    # rather than a fixed 28 cells, which is the defect this pins.
    narrow_label, narrow_tail = fit_row(label, detail, 100)
    assert len(narrow_label) == len(wide_label)
    assert len(wide_tail) - len(narrow_tail) == 40
    # A detail too small to say anything is dropped rather than rendered as a
    # stub of an ellipsis.
    assert fit_row(label, detail, len(label) + 4) == (label, "")
    # The NAME is cropped only when even a bare name cannot fit — the last
    # resort rather than the rule.
    assert fit_row("a-very-long-name", "x", 8) == ("a-very-…", "")


def test_a_match_past_the_crop_slides_the_window_to_it() -> None:
    """Design D2: a row admitted for something the paint threw away."""
    detail = "Read-only research: investigates a question across the workspace"
    # No filter: the clause starts at the left, as always.
    _label, tail = fit_row("Scout", detail, 30)
    assert tail.startswith("  · Read-only")
    # Filtered on a word deep in the clause: the window slides to it and the
    # LEADING ellipsis says the text was entered in the middle.
    _label, matched = fit_row("Scout", detail, 30, query="workspace")
    assert "workspace" in matched
    assert matched.split("· ", 1)[1].startswith("…")


def test_the_subject_line_names_the_project_and_the_consequence() -> None:
    """UX U2: the card says what it will do, and for which project."""
    line = subject_line("parity-spec")
    assert "parity-spec" in line
    assert "links to" in line and "starts a turn" in line and "you go there" in line
    # No project resolved: the sentence degrades without inventing a name.
    assert "parity-spec" not in subject_line("")


# -- the card ---------------------------------------------------------------


def test_the_card_paints_a_header_per_section_and_never_selects_one() -> None:
    card = StartPickerCard(
        start_targets(teams=[_team("core")], agents=[_agent("coder", "implements")])
    )
    lines = card.painted_lines()
    assert [line.text for line in lines] == [
        PLAIN_HEADER,
        _row_line(PLAIN_LABEL, PLAIN_DETAIL),
        TEAM_SECTION,
        "Core · manager · 2 roles",
        AGENT_SECTION,
        "Coder · implements",
    ]
    # Headers and section notes carry no row index; only real rows do.
    assert [line.header for line in lines] == [True, False, True, False, True, False]
    assert [line.row for line in lines] == [-1, 0, -1, 1, -1, 2]
    # The block is what the card's height is made of: chrome + painted lines.
    assert card._visible_lines == len(lines)


def test_the_card_opens_on_the_plain_row() -> None:
    """UX U4: `s` then `enter` must not commit a roster by accident.

    The first row used to be the alphabetically first TEAM, so two unaimed
    keystrokes attached a team's worth of tokens; the desktop parity dialog
    defaults to its plain option, and so does this card.
    """
    card = StartPickerCard(
        start_targets(teams=[_team("core"), _team("lopdev")], agents=[_agent("coder")])
    )
    selected = card.selected()
    assert selected is not None and selected.kind == "plain"
    assert card.index == 0
    # And the window starts there rather than sliding anywhere: the default is
    # reachable without a single `↓`, and the sections below it are on screen.
    assert 0 in set(card.painted_range())
    assert card._top == 0


def test_the_geometry_the_card_reserves_is_the_block_it_paints() -> None:
    """Chrome + painted lines is the card's whole height (P5a's D1 property)."""
    card = StartPickerCard(start_targets(teams=[_team(f"t{i}") for i in range(6)], agents=[]))
    # A ground this short has already spent the SUBJECT line (D12), so the
    # budget is the chrome the card is actually carrying — asserted through the
    # same accessor the painter uses rather than a constant the rule can drift
    # away from.
    card.set_available(START_CARD_CHROME_ROWS + 3)
    assert card._chrome_rows() == START_CARD_CHROME_ROWS - 1
    assert card._visible_lines <= card._available - card._chrome_rows()
    assert card._visible <= START_CARD_ROW_CAP
    # A ground too short for even one line leaves nothing selectable, and
    # `enter` then refuses rather than answering with an unpainted row.
    card.set_available(2)
    assert card.painted_range() == range(0, 0)


def test_a_short_ground_spends_the_subject_line_before_the_list() -> None:
    """Design review round 2, D12: one target where the ground held three.

    Measured on the real app: 60×24, 70×24 and 80×24 all resolve a FOUR-line
    rows block, and with the subject painted that block bought `no team`, the
    plain row, `teams` and a single team — one real target, where the same
    ground held three before the plain row led the card. The sentence yields
    below five lines and the card is one row shorter in exchange.
    """
    card = StartPickerCard(
        start_targets(
            teams=[_team("core"), _team("lopdev")],
            agents=[_agent("coder", "implements one bounded slice")],
        )
    )
    # The measured 24-row ground.
    card.set_available(12)
    assert card._chrome_rows() == START_CARD_CHROME_ROWS - 1
    assert len(card.painted_lines()) == 5
    assert card._visible == 3, [(line.text, line.row) for line in card.painted_lines()]
    # A comfortable ground keeps it: 100×30 hands the card 18 rows.
    card.set_available(18)
    assert card._chrome_rows() == START_CARD_CHROME_ROWS
    # 60×20 resolves no block at all either way, and the note says why (U6).
    card.set_available(8)
    assert card.painted_range() == range(0, 0)


def test_the_window_slides_to_keep_the_selection_painted() -> None:
    rows = start_targets(teams=[_team(f"t{i:02d}") for i in range(8)], agents=[])
    card = StartPickerCard(rows)
    card.set_available(START_CARD_CHROME_ROWS + 7)
    for _ in range(3):
        card.action_move(-1)
    painted = set(card.painted_range())
    assert card.index in painted
    assert card._top > 0


def test_a_create_in_flight_refuses_a_second_pick() -> None:
    card = StartPickerCard(start_targets(teams=[_team("core")], agents=[]))
    posted: list[object] = []
    card.post_message = posted.append  # type: ignore[method-assign]
    card.action_choose()
    assert [m for m in posted if isinstance(m, StartPickerCard.Chosen)]
    posted.clear()
    card.set_pending()
    assert card.pending
    card.action_choose()
    assert not [m for m in posted if isinstance(m, StartPickerCard.Chosen)]
    # A refusal LEAVES the pending state: every refusal this card can meet is
    # answered by picking another row or closing.
    card.show_refusal("could not start a session: no runtime")
    assert not card.pending
    card.action_choose()
    assert [m for m in posted if isinstance(m, StartPickerCard.Chosen)]


def test_a_refusal_does_not_look_like_a_progress_note() -> None:
    """UX U3: three sentences shared one slot and one ink.

    The card is built WITH the resolver the page hands it, and the resolver's
    `refusal` key is asserted to be the measured `warning` token — otherwise
    this test compares `None` to `None` and ties nothing to the palette pin
    (agent review round 2, N1).
    """
    from rich.color import Color

    from local_operator.tui import theme as theme_mod

    card = StartPickerCard(
        start_targets(teams=[_team("core")], agents=[]), style_for=_style_resolver()
    )
    card.set_pending()
    pending = card._note_text()
    card.show_refusal("could not start a session: no runtime")
    refusal = card._note_text()
    assert pending.plain != refusal.plain
    assert refusal.plain.startswith(REFUSAL_MARK)
    # The refusal takes the app's own refusal ink, and that ink IS the pinned
    # `warning` pair (7.09:1 dark / 4.78:1 light on this ground) — not the
    # muted ink the counter and the pending sentence wear.
    assert refusal.style == card._ink("refusal")
    refusal_ink = _style_resolver()("refusal")
    assert refusal_ink.color == Color.parse(theme_mod.semantic_color("warning"))
    assert pending.style == card._ink("muted")


def test_a_filter_that_matches_nothing_names_the_right_objects() -> None:
    """Design D3: the note said `sessions` in a card that has none."""
    card = StartPickerCard(start_targets(teams=[_team("core")], agents=[]))
    _filter(card, "zzz")
    assert card._rows == []
    assert all(line.row < 0 for line in card.painted_lines())
    assert card._note_text().plain == NO_MATCH_NOTE
    assert "session" not in NO_MATCH_NOTE
    assert NO_MATCH_NOTE not in (NO_TEAM_NOTE, NO_AGENT_NOTE)


def test_a_ground_with_no_room_says_so_instead_of_counting() -> None:
    """UX U6: `+N more` promised rows the card could not paint."""
    card = StartPickerCard(
        start_targets(teams=[_team(f"t{i}") for i in range(6)], agents=[_agent("coder")])
    )
    card.set_available(START_CARD_CHROME_ROWS)
    assert card._visible == 0
    assert card._note_text().plain == NO_ROOM_NOTE


def test_the_empty_registry_fact_survives_a_tight_ground() -> None:
    """Design D4: the section note is the first thing the budget drops."""
    card = StartPickerCard(start_targets(teams=[], agents=[]))
    card.set_available(START_CARD_CHROME_ROWS + 1)
    assert card._with_notes is False
    note = card._note_text().plain
    assert NO_TEAM_NOTE in note and NO_AGENT_NOTE in note
    # With room for them the fact lives in the BLOCK, in its own section, and
    # the note slot goes back to the counter.
    card.set_available(START_CARD_CHROME_ROWS + 7)
    assert card._with_notes is True
    assert card._note_text().plain == ""


def test_the_painted_row_marks_the_cells_the_filter_matched() -> None:
    """Design D2: an admission the reader cannot audit reads as noise."""
    card = StartPickerCard(
        start_targets(teams=[], agents=[_agent("reviewer", "adversarial reading of diffs")]),
        style_for=_style_resolver(),
    )
    _filter(card, "adversarial")
    card.set_available(START_CARD_CHROME_ROWS + 5)
    # The row's own painter, at a MEASURED width: an unmounted card has no
    # content box, and a zero-width row paints nothing to mark.
    line = next(line for line in card.painted_lines() if line.row >= 0)
    text = card._row_text(line, line.row == card.index, 60)
    marked = [span for span in text.spans if isinstance(span.style, Style) and span.style.underline]
    assert marked, "the matched cells carry no mark"
    marked_text = "".join(text.plain[span.start : span.end] for span in marked)
    assert "adversarial" in marked_text
    # On the SELECTED row the mark composes with the selection band instead of
    # punching a hole in it (the two set different attributes, and `+` keeps
    # both), so the cursor's row is not the one place a match is unreadable.
    selected_row = card._row_text(line, True, 60)
    for span in selected_row.spans:
        if isinstance(span.style, Style) and span.style.underline:
            assert span.style.bgcolor is not None, "the mark dropped the selection band"


def _real_catalogue_rows() -> list[StartTarget]:
    """The PACKAGED catalogue, read from the manifest this repo ships.

    Not a fixture of made-up rows: the widths that break a crop are the ones the
    real descriptions reach, so the sweep has to run over the real text.
    """
    from importlib.resources import files

    manifest = json.loads(
        (files("local_operator") / "agent_seeds" / "manifest.json").read_text(encoding="utf-8")
    )
    return start_targets(
        teams=[
            {
                "name": "lopdev",
                "label": "Lopdev",
                "manager": "manager",
                "members": [{"role": "coder", "count": 1}],
            }
        ],
        agents=[
            {
                "name": seed["name"],
                "label": seed["label"],
                "description": seed.get("description", ""),
            }
            for seed in manifest["seeds"]
        ],
    )


def test_the_crop_covers_the_match_rather_than_halving_it() -> None:
    """QA round 2, Q4's two repros, as the crop rule they pin.

    The slide used to read only where the match STARTED, so a match that began
    inside the crop and ended past it was painted in half and marked not at all:
    `trade-offs` painted `…with trade-o…` and `severity` painted nothing.
    """
    architect = (
        "Explores a codebase and produces a design or technical proposal with trade-offs; "
        "may draft documents but never modifies existing source."
    )
    label, tail = fit_row("Architect", architect, 144, query="trade-offs")
    assert "trade-offs" in tail, tail
    assert len(label) + len(tail) <= 144
    reviewer = (
        "Independent code review of a diff, MR, or PR: finds defects, classifies them by "
        "severity, and never edits the code it reviews."
    )
    label, tail = fit_row("Reviewer", reviewer, 92, query="severity")
    assert "severity" in tail, tail
    assert len(label) + len(tail) <= 92
    # A match too long for the window cannot be covered, so the rule keeps its
    # FIRST matched cell painted — which is what the mark is derived from.
    _label, tail = fit_row(
        "Scout",
        "Read-only research: investigates a question across the workspace and on the web",
        34,
        query="workspace",
    )
    assert visible_match("workspace", tail), f"the first matched cell is not painted: {tail!r}"


def test_every_admitted_row_paints_a_mark() -> None:
    """QA round 2, Q4: an admission the reader cannot see a reason for.

    The filter admits a row on a subsequence over its FULL text while the card
    paints a CROP of it, so the crop must cover the match and the mark must be
    derived from the cropped text: otherwise a row paints with no mark at all
    (measured before this fix: 10 of 51 admitted rows at 144 cells, 25 of 51 at
    92, 35 of 51 at 66). The property is swept over the real catalogue at those
    three widths, with QA's needles plus the shipped names.
    """
    needles = (
        "trade-offs",
        "severity",
        "reviewer",
        "defects",
        "core",
        "workspace",
        "research",
        "tui",
        "adversarial",
        "milestones",
        "sessions",
        "ai-isms",
        "aida",
    )
    rows = _real_catalogue_rows()
    admitted = 0
    for width in (144, 92, 66):
        for needle in needles:
            card = StartPickerCard(rows, style_for=_style_resolver())
            _filter(card, needle)
            for line in card.painted_lines():
                if line.row < 0:
                    continue
                text = card._row_text(line, False, width)
                admitted += 1
                assert len(text.plain) <= width, f"{needle!r} at {width}: {text.plain!r} overflows"
                marked = [
                    span
                    for span in text.spans
                    if isinstance(span.style, Style) and span.style.underline
                ]
                assert marked, f"{needle!r} at {width}: {text.plain!r} paints no mark"
    assert admitted > 100, f"the sweep stopped admitting rows ({admitted})"


def test_the_ellipsis_tracks_the_crop_in_both_directions() -> None:
    """Design review round 2 / UX U11: the marker is a fact about the CROP.

    It appeared on rows that were not cropped at all (`manager · 2 roles…`) and
    vanished from rows that were cut mid-word (`…one bounded slice of work end
    to en`), because the fit branch always marked and the assembled string was
    clamped after the fact.
    """
    from local_operator.tui.widgets.projects_start import _fit_detail

    # FIT: returned whole — no marker, and no cell spent on one.
    assert _fit_detail("manager · 1 role", 80, None) == "manager · 1 role"
    _label, tail = fit_row("Core", "manager · 1 role", 90)
    assert tail == "  · manager · 1 role"
    # CUT: the marker is there, and the row ends in it rather than mid-word.
    assert _fit_detail("x" * 100, 10, None) == "x" * 9 + "…"
    coder = (
        "Implements one bounded slice of work end to end with the full toolset, "
        "then reports what changed and how it was verified."
    )
    out = _fit_detail(coder, 41, (8, 114))
    assert len(out) == 41
    assert out.endswith("…"), out
    assert coder[8] in out, "the match's first cell is not painted"


def test_a_row_wears_one_mark_and_the_prose_is_what_earned_it() -> None:
    """Design review round 2, D11: scattered cells inside words, and on names.

    The mark used to be re-derived over the whole painted row, so `trade` marked
    `Archi t ec t` as well as the word that admitted the row — six isolated cells
    that interrupted word shapes and stopped discriminating between rows.
    """
    architect = (
        "Explores a codebase and produces a design or technical proposal with trade-offs; "
        "may draft documents but never modifies existing source."
    )
    card = StartPickerCard(
        start_targets(teams=[], agents=[_agent("architect", architect)]),
        style_for=_style_resolver(),
    )
    _filter(card, "trade")
    line = next(line for line in card.painted_lines() if line.row >= 0)
    text = card._row_text(line, False, 144)
    marked = [span for span in text.spans if isinstance(span.style, Style) and span.style.underline]
    assert len(marked) == 1, [text.plain[span.start : span.end] for span in marked]
    span = marked[0]
    assert text.plain[span.start : span.end].casefold() == "trade"
    # The NAME carries none of it: the label starts at index 2.
    assert span.start >= 2 + len("Architect")


def test_a_click_on_a_section_header_selects_nothing() -> None:
    """Agent review F6: `on_click` maps through PAINTED LINES, not row indices."""
    card = StartPickerCard(
        start_targets(teams=[_team("core")], agents=[_agent("coder", "implements")])
    )
    card.set_available(START_CARD_CHROME_ROWS + 6)
    lines = card.painted_lines()
    header_line = next(index for index, line in enumerate(lines) if line.header)
    before = card.index

    class _Event:
        def __init__(self, y: int) -> None:
            self.y = y

        def stop(self) -> None:
            pass

    card.on_click(_Event(card._row_block_top() + header_line))
    assert card.index == before, "a header click moved the selection"
    # The row UNDER that header selects: the mapping is window-relative and
    # accounts for the header line it just skipped.
    row_line = next(index for index, line in enumerate(lines) if line.row >= 0)
    card.on_click(_Event(card._row_block_top() + row_line))
    assert card.index == lines[row_line].row


# -- the page ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_s_opens_the_card_and_esc_puts_it_away(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        assert view._mode == "canvas"
        await pilot.press("s")
        await pilot.pause()
        assert view._mode == "start"
        card = view._start_card
        assert card is not None
        assert [row.kind for row in card.rows] == ["plain", "team", "team", "agent"]
        # The card carries the grammar (spec §3.3's picker ladder) and the
        # subject line names the project it is about (UX U2).
        legend = card.query_one("#projects-start-legend").render()
        assert "esc close" in str(legend)
        subject = card.query_one("#projects-start-subject").render()
        assert "alpha" in str(subject)
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"
        assert view._start_card is None


@pytest.mark.asyncio
async def test_s_from_the_detail_page_gets_the_full_width_card(tmp_path: Path) -> None:
    """UX U1 (BLOCKER): the placement read the HIDDEN canvas body.

    In detail mode the canvas is `display=False`, so its region is zero-width
    and the card rendered 20 cells wide with its rows wrapped over three lines
    at every terminal size. `s` is one of the two documented entries, so the
    ground must be the surface that is actually showing.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("d")
        await pilot.pause()
        assert view._mode == "detail"
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        card = view._start_card
        assert card is not None
        assert (
            card.region.width >= view.size.width - 8
        ), f"the detail entry rendered a {card.region.width}-cell card"
        assert card.region.width <= view.size.width
        # The rows are one line each: the defect wrapped every one of them.
        for line in card.rows_text().plain.splitlines():
            assert len(line) <= card.region.width
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "detail"


@pytest.mark.asyncio
async def test_the_page_hint_row_is_blank_while_the_card_holds_the_keys(
    tmp_path: Path,
) -> None:
    """R2-3/D10 one mode over: page keys the card consumes must not be advertised."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        painted = [
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display and hint.rendered().strip()
        ]
        assert painted == [], painted


@pytest.mark.asyncio
async def test_the_card_floats_over_the_canvas_without_moving_the_page(
    tmp_path: Path,
) -> None:
    """The P5a geometry contract, re-asserted for the family's second card."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        before = view.canvas_size
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        card = view._start_card
        assert card is not None
        assert card._available >= START_CARD_CHROME_ROWS
        # Against the chrome the card is ACTUALLY carrying: at this size that is
        # the yielded chrome (D12), and a constant here would silently disagree
        # with the rule the painter uses.
        assert card._visible_lines <= max(0, card._available - card._chrome_rows())
        # The card lives inside the page's own content box and the SCREEN never
        # grows a scrollbar (the mode's shipped invariant).
        assert view.canvas_size == before
        content = view.content_region
        assert card.region.y >= content.y
        assert card.region.y + card.region.height <= content.y + content.height
        assert app.screen.virtual_size == app.screen.size


@pytest.mark.asyncio
async def test_choosing_a_row_names_the_project_and_the_target(tmp_path: Path) -> None:
    session = _ProjectSession()
    registry = _registry(tmp_path, "alpha")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        posted: list[object] = []
        real_post = view.post_message

        def record(message: object) -> object:
            posted.append(message)
            return real_post(message)  # type: ignore[arg-type]

        view.post_message = record  # type: ignore[method-assign]
        card = view._start_card
        assert card is not None
        # The DEFAULT is the plain row (U4), so move to a team explicitly.
        card._index = 1
        card.action_choose()
        await pilot.pause()
        requests = [m for m in posted if isinstance(m, ProjectsViewStartRequested)]
        assert len(requests) == 1
        assert requests[0].target.kind == "team"
        assert requests[0].target.name == "core"
        expected = registry.get_project_by_name("alpha")
        assert expected is not None
        assert requests[0].project_id == expected.id


@pytest.mark.asyncio
async def test_s_does_nothing_with_nothing_selected(tmp_path: Path) -> None:
    """No project, nothing to link and no snapshot to quote: no card."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        # An empty PAGE — the state a store that lost its last project leaves.
        view.load(views=[], updated_at=0.0, own_session=None)
        await pilot.pause()
        assert view.current_project_id() is None
        await pilot.press("s")
        await pilot.pause()
        assert view._start_card is None


def _key_flags(view: Any, rungs: list[Any]) -> list[bool]:
    """Which rungs carry `s start`, in ladder order."""
    return [any(hint is view._start_hint for hint, _label, _lead in plan) for plan, _esc in rungs]


def test_the_start_key_never_displaces_a_shipped_hint() -> None:
    """Agent review F4: the ladder must not move a shipped row.

    The key rides a one-rung-wider VARIANT of a shipped row, never an extra
    element inside one: the ladder stops at the first rung that fits, so a
    variant in front of its own base row can only ADD the key. This sweeps every
    budget from 3 cells to 220 and asserts that whatever the base ladder (the
    same rungs minus the variants) would paint is still painted, and that no
    variant is unreachable — a rung behind a wider one is dead.

    THE ONE RECORDED EXCEPTION is asserted below rather than left invisible
    (agent review round 2, N2): the timeline's own rung gives up `d detail` where
    the shipped ladder kept it, because `+/-` acts on THIS view and `d` acts on
    the page. The base here is "this PR's rungs minus the variants", so it cannot
    see a trade made by a rung this PR itself added — which is exactly why the
    trade is named and pinned instead of being covered by the sweep.
    """
    from local_operator.tui.widgets.projects_view import ProjectsView

    for view_name in ("list", "board", "timeline"):
        view = ProjectsView()
        view._view = view_name
        start_hint = view._start_hint
        rungs = view._canvas_hint_rungs()
        head = [(view._measure_hints(plan, esc), plan, esc) for plan, esc in rungs]
        base = [
            (view._measure_hints(plan, esc), plan, esc)
            for plan, esc in rungs
            if not any(hint is start_hint for hint, _label, _lead in plan)
        ]
        # Every variant is strictly narrower than the rung before it, so none is
        # shadowed by a wider row.
        previous = None
        for width, plan, _esc in head:
            if previous is not None and any(hint is start_hint for hint, _label, _lead in plan):
                assert (
                    width < previous
                ), f"{view_name}: a variant at {width} is shadowed by {previous}"
            previous = width
        displaced: list[int] = []
        for budget in range(3, 221):
            chosen_head = next((plan for width, plan, _e in head if width <= budget), head[-1][1])
            chosen_base = next((plan for width, plan, _e in base if width <= budget), base[-1][1])
            labels_head = {label for _h, label, _l in chosen_head}
            for _hint, label, _lead in chosen_base:
                if label.strip() and label not in labels_head:
                    displaced.append(budget)
                    break
        assert not displaced, f"{view_name}: budgets losing a shipped hint: {displaced[:8]}"

        if view_name == "timeline":
            # The pinned trade: somewhere in the sweep the zoom rung replaces
            # `d detail`, and the row still carries `+/- zoom`.
            traded = [
                (width, plan)
                for width, plan, _e in head
                if any(label.strip() == "zoom" for _h, label, _l in plan)
                and not any(label.strip() == "detail" for _h, label, _l in plan)
            ]
            assert traded, "the timeline's zoom-over-detail rung is gone"
            assert any(
                any(label.strip() == "detail" for _h, label, _l in plan)
                for width, plan, _e in base
                if width >= max(w for w, _p in traded)
            ), "the trade is no longer a trade: the base row had it too"


def test_the_start_hint_is_advertised_monotonically() -> None:
    """QA round 2, Q3: a hint that vanishes as the terminal WIDENS is a defect.

    The ladder takes the first rung that fits, so a key-less rung in front of a
    narrower key-carrying one un-advertises the key exactly when the reader has
    more room — measured on the previous revision as absent at 160/150/140/130…
    and present at 145 and 120. Both ladders must therefore carry the key on a
    PREFIX of themselves, and the shipped rows below it are untouched.
    """
    from local_operator.tui.widgets.projects_view import ProjectsView

    for view_name in ("list", "board", "timeline"):
        view = ProjectsView()
        view._view = view_name
        flags = _key_flags(view, view._canvas_hint_rungs())
        seen_off = False
        for index, carried in enumerate(flags):
            if not carried:
                seen_off = True
            else:
                assert (
                    not seen_off
                ), f"{view_name}: the key returns at rung {index} after a rung without it"
    detail_view = ProjectsView()
    detail_flags = _key_flags(detail_view, detail_view._detail_hint_rungs())
    seen_off = False
    for index, carried in enumerate(detail_flags):
        if not carried:
            seen_off = True
        else:
            assert not seen_off, f"detail: the key returns at rung {index}"


def test_the_start_hint_carries_the_specs_own_label() -> None:
    """Agent review F3: the label is `s start`, and the arithmetic that said
    otherwise was measuring the wrong budget (`size.width - 2`, where
    `size.width` is terminal − 4)."""
    from local_operator.tui.widgets.projects_view import ProjectsView

    view = ProjectsView()
    labels = {label for plan, _esc in view._canvas_hint_rungs() for _hint, label, _lead in plan}
    assert " start" in labels
    assert " new" not in labels
    detail_labels = {
        label for plan, _esc in view._detail_hint_rungs() for _hint, label, _lead in plan
    }
    assert " start" in detail_labels


# -- the boot flow (the app's own handler) ----------------------------------


async def _settle(pilot: Any, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_app_creates_links_kicks_off_and_hands_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONE order, and the session id it mints is the one every step sees."""
    session = _ProjectSession()
    registry = _registry(tmp_path, "alpha")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    calls: list[tuple[str, Any]] = []

    async def fake_create(cwd: str, target: StartTarget) -> str:
        calls.append(("create", (cwd, target.kind, target.name)))
        return STARTED_ID

    async def fake_kickoff(session_id: str, cwd: str, project: Any) -> None:
        calls.append(("kickoff", (session_id, project.name)))

    def fake_resume(session_id: str, notice: Any) -> None:
        calls.append(("resume", session_id))

    monkeypatch.setattr(app, "_create_project_session", fake_create)
    monkeypatch.setattr(app, "_kick_off_project_session", fake_kickoff)
    monkeypatch.setattr(app, "_resume_session", fake_resume)

    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        card = view._start_card
        assert card is not None
        card._index = 0
        card.action_choose()
        await _settle(pilot)

    assert [name for name, _ in calls] == ["create", "kickoff", "resume"]
    # The page closed before the hand-off (the mode must be gone before the
    # conversation it hid comes back) — `_resume_session` itself is stubbed
    # above, so this is the page's own exit and not the reboot's.
    assert app._projects_view is None
    project = registry.get_project_by_name("alpha")
    assert project is not None
    # The auto-link is a WORKING link (the CoS exemption marks a FILING).
    assert project.sessions == [STARTED_ID]
    assert project.coordination_sessions == []


@pytest.mark.asyncio
async def test_esc_while_pending_cancels_the_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent review F5 / UX: `esc` on `starting session …` used to switch anyway.

    The create is durable and keeps going — what the reader cancelled is being
    TAKEN somewhere, so no kickoff turn is sent on their behalf either, and the
    receipt names the id and the way in.
    """
    session = _ProjectSession()
    registry = _registry(tmp_path, "alpha")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    calls: list[str] = []
    notes: list[tuple[str, str]] = []
    release = asyncio.Event()

    async def slow_create(cwd: str, target: StartTarget) -> str:
        await release.wait()
        calls.append("create")
        return STARTED_ID

    async def fake_kickoff(session_id: str, cwd: str, project: Any) -> None:
        calls.append("kickoff")

    def fake_resume(session_id: str, notice: Any) -> None:
        calls.append("resume")

    monkeypatch.setattr(app, "_create_project_session", slow_create)
    monkeypatch.setattr(app, "_kick_off_project_session", fake_kickoff)
    monkeypatch.setattr(app, "_resume_session", fake_resume)
    # The receipt is the APP's notice (the transcript's own row), not the
    # card's — recorded here so the wording can be asserted.
    monkeypatch.setattr(app, "_notice", lambda text, kind="info": notes.append((text, kind)))

    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        card = view._start_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        assert card.pending, "the create did not enter its pending state"
        await pilot.press("escape")
        await pilot.pause()
        assert view._start_card is None
        release.set()
        await _settle(pilot)

        assert calls == ["create"], f"a cancelled start still ran {calls}"
        assert app._projects_view is not None, "a cancelled start closed the page"
        receipt = " ".join(text for text, _kind in notes)
        assert STARTED_ID in receipt and "/resume" in receipt
        # ...AND ON THE PAGE THE READER IS LOOKING AT (UX review round 2, U12):
        # the transcript copy is the durable record, but the settled page used
        # to be byte-identical to the pre-start frame, so `s` again was the
        # reasonable next move. The page's own notice row carries it now.
        assert view._notice is not None
        assert STARTED_ID in view._notice and "/resume" in view._notice

    project = registry.get_project_by_name("alpha")
    assert project is not None and project.sessions == [STARTED_ID]


@pytest.mark.asyncio
async def test_a_refused_create_lands_in_the_card_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _ProjectSession()
    registry = _registry(tmp_path, "alpha")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    handoffs: list[str] = []
    monkeypatch.setattr(
        app,
        "_create_project_session",
        lambda cwd, target: _raise(ValueError("there is no team named 'ghost'")),
    )
    monkeypatch.setattr(app, "_resume_session", lambda sid, notice: handoffs.append(sid))

    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        card = view._start_card
        assert card is not None
        card.action_choose()
        await _settle(pilot)

        assert card.pending is False
        assert "no team named 'ghost'" in card._note
        # The page never switched and the store never changed.
        assert view._mode == "start"
        project = registry.get_project_by_name("alpha")
        assert project is not None and project.sessions == []

    assert handoffs == []
    assert not list((tmp_path / "sessions").glob("*")) if (tmp_path / "sessions").exists() else True


@pytest.mark.asyncio
async def test_an_unknown_target_gets_the_rows_own_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create core refuses an unknown name with a bare ``KeyError``."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))

    async def unknown(cwd: str, target: StartTarget) -> str:
        raise KeyError(target.name)

    monkeypatch.setattr(app, "_create_project_session", unknown)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(start_targets(teams=[_team("ghost")], agents=[]))
        await pilot.press("s")
        await pilot.pause()
        card = view._start_card
        assert card is not None
        # The card OPENS on the plain row (UX U4), so step to the team row the
        # case is about before choosing.
        card._index = 1
        card.action_choose()
        await _settle(pilot)
        assert card._note == "could not start a session: no team named 'ghost'"


@pytest.mark.asyncio
async def test_the_desktop_create_core_materialises_the_team_binding(tmp_path: Path) -> None:
    """THE REAL CALLABLE, not a stand-in: what `_create_project_session` runs.

    ``DesktopSessions.create`` is what the pane's new-chat reaches through
    ``POST /v1/desktop/sessions``; this executes it against a synthetic root
    and asserts the two facts the whole design rests on — a 12-character hex id
    is minted, and the team the row named is on disk as the session's
    attachment BEFORE any runtime exists for it. No runtime is started (create
    writes records, it does not engage), so this needs no provider and no
    network.
    """
    from local_operator.resume import read_session_attachment
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    root = tmp_path / "root"
    root.mkdir()
    TeamRegistry(root).create_team(
        TeamEditFields(name="lopdev", members=[TeamMember(role="coder")])
    )
    pool = DesktopSessions(root)
    session_id = await pool.create(str(root), target={"kind": "team", "name": "lopdev"})

    assert len(session_id) == 12 and all(c in "0123456789abcdef" for c in session_id)
    directory = root / "sessions" / session_id
    attachment = read_session_attachment(directory)
    assert attachment is not None
    assert attachment.team == "lopdev"
    # And the refusal the app turns into a sentence is the core's own.
    with pytest.raises(KeyError):
        await pool.create(str(root), target={"kind": "team", "name": "ghost"})


async def _raise(error: Exception) -> None:
    raise error

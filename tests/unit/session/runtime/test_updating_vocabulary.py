"""Every phase of an update window reaches a surface, and its copy comes from one place.

The sibling of ``test_leaving_vocabulary.py``, and it exists for the failure that
file was written about (design round 1, D1/D2/D3; QA round 1, Q-1): a state
published on the record and painted by nobody, because each reader FALLS BACK to
a sentence about a different state — there it told the operator the session was
finishing work it had given up on; here it would tell them the update is still
coming when it has already been abandoned.

THE REACHABILITY HALF IS THE POINT (agent review round 1, MINOR 3; design review
round 1, D1). The first version of this file walked three per-phase tables and
asserted each had an entry — while every call site hardcoded a single phase, so
two thirds of the text under test could not be painted by any frame, and the
failed phase rendered a BLANK fleet cell. A pin that walks a table nothing reads
is the same artefact as no pin. So each phase is now asserted twice:

1. the COPY exists and is non-empty for every phase (:func:`types.update_phrase`,
   :func:`types.update_short`, ``info_panel._UPDATING_SHORT``);
2. the CALL SITES read the phase rather than assuming one — ``cli._updating_cell``
   and ``_sessions_section`` both go through :func:`types.update_phase`, which is
   what makes all three reachable, and the behavioural cells below
   (``test_sessions_extraction.test_a_failed_window_is_named_in_the_fleet_table``,
   ``test_info_names_a_failed_window_as_plainly_as_it_names_a_drain``) drive the
   real renderers with a failed record so the claim is not source-reading alone.
"""

from __future__ import annotations

import inspect

import pytest

from local_operator import buildwatch, cli, incidents
from local_operator.info import collect as info_collect
from local_operator.info.model import SessionLine
from local_operator.session.runtime import types
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import info_panel
from local_operator.tui.widgets.info_panel import _UPDATING_SHORT, _UPDATING_SHORT_OTHER
from local_operator.update import BuildStamp

PHASES: tuple[str, ...] = types.PUBLISHED_UPDATE_PHASES


#: The widest pair a fleet cell has to print, built the way a runtime builds one —
#: through ``BuildStamp.label()``, the ONLY producer of the labels that reach this
#: field. Hand-typing a longer pair would size the column for a shape no runtime can
#: emit, which is the "legend at a width no caller can produce" trap AGENTS.md
#: records (the picker footer measured at 74 cells, its legend wanting 88).
#:
#: A DEV/DESCRIBE LABEL IS WIDER THAN THE COLUMN ON PURPOSE, and that is what
#: ``test_the_column_marks_a_cut_rather_than_lying`` pins: the width is sized for the
#: common case and the overflow is MARKED, rather than the width being sized for a
#: string that also has to be cut somewhere (design review round 1, D3).
WIDEST_PAIR = buildwatch.update_pair_text(
    BuildStamp(version="0.59.9", source_ref=""),
    BuildStamp(version="0.59.11", source_ref="ead71b673a9a24e6925338bcb51315e0ac5d44f"),
)
DEV_PAIR = buildwatch.update_pair_text(
    BuildStamp(version="0.59.9", source_ref=""),
    BuildStamp(version="0.59.11.dev3+g1a2b3c4", source_ref=""),
)


def test_the_phases_are_enumerable_and_have_no_duplicates() -> None:
    assert PHASES, "the tuple a consumer pin iterates cannot be empty"
    assert len(set(PHASES)) == len(PHASES), PHASES


@pytest.mark.parametrize("phase", PHASES)
def test_the_sentence_exists_for_every_phase(phase: str) -> None:
    sentence = types.update_phrase(phase, WIDEST_PAIR)
    assert sentence, f"{phase!r} has no sentence: the prose surfaces would paint nothing"
    assert (
        WIDEST_PAIR in sentence or phase == types.UPDATE_FAILED
    ), f"{phase!r} does not name the build it is about: {sentence!r}"


@pytest.mark.parametrize("phase", PHASES)
def test_the_fleet_cell_exists_for_every_phase(phase: str) -> None:
    assert types.update_short(phase, WIDEST_PAIR), (
        f"{phase!r} has no fleet cell: the row would render BLANK, which is the defect "
        f"design review round 1 (D1) measured against the real ``lop sessions``"
    )
    assert (
        types.update_short(phase, WIDEST_PAIR) != types.update_short(types.UPDATING, WIDEST_PAIR)
        or phase == types.UPDATING
    ), "two phases must not share one cell's words"


@pytest.mark.parametrize("phase", PHASES)
def test_the_narrow_frame_form_knows_every_phase(phase: str) -> None:
    short = _UPDATING_SHORT.get(phase)
    assert short is not None and short != _UPDATING_SHORT_OTHER, (
        f"{phase!r} has no shelf form: below the note floor the row would degrade to "
        f"{_UPDATING_SHORT_OTHER!r}"
    )


def test_the_fleet_column_is_sized_for_the_common_case_and_marks_the_rest() -> None:
    """The width fits a real label; a wider one is MARKED, never silently cut.

    Design review round 1 (D3) measured the silent cut on the real renderer: a dev
    label came back as a plausible-looking ``0.59.11``, and the unparseable-pair
    fallback was cut mid-word. Both now carry the column's ellipsis, which is
    ``_clamp_reason_cell``'s mark for the same reason it carries it in WHY.
    """
    common = types.update_short(types.UPDATING, WIDEST_PAIR)
    assert cli.UPDATING_COLUMN_WIDTH >= len(common), (
        f"the UPDATING column ({cli.UPDATING_COLUMN_WIDTH}) truncates the common cell "
        f"({len(common)}): {common!r}"
    )


def test_the_column_marks_a_cut_rather_than_lying() -> None:
    """A cell wider than the column is visibly cut, not plausibly wrong."""
    over = cli._updating_cell(DEV_PAIR, "")
    assert (
        len(types.update_short(types.UPDATING, DEV_PAIR)) > cli.UPDATING_COLUMN_WIDTH
    ), "premise: this fixture must overflow the column, or the cell proves nothing"
    cut = cli._clamp_reason_cell(over, cli.UPDATING_COLUMN_WIDTH)
    assert len(cut) <= cli.UPDATING_COLUMN_WIDTH, cut
    assert cut.endswith("…"), f"a cut build label must be marked: {cut!r}"
    assert cut != over


def test_the_unparseable_pair_names_no_build_rather_than_a_cut_one() -> None:
    """The fallback is the phase alone — short enough that it cannot be cut at all."""
    cell = types.update_short(types.UPDATING, "")
    assert cell == "updating", cell
    assert len(cell) <= cli.UPDATING_COLUMN_WIDTH


def test_the_phase_reader_is_the_only_precedence() -> None:
    """``update_phase`` decides which of the three facts a surface renders."""
    assert types.update_phase("p", "u", "f") == (types.UPDATING, "p"), "an open window wins"
    assert types.update_phase("", "u", "f") == (types.UPDATE_FAILED, "f"), "then a failure"
    assert types.update_phase("", "u", "") == (types.UPDATING_DONE, "u"), "then an applied one"
    assert types.update_phase("", "", "") == ("", ""), "an idle row is not a phase"


def test_the_call_sites_read_the_phase_instead_of_hardcoding_one() -> None:
    """The construct MINOR 3 measured: a hardcoded phase makes two thirds unreachable.

    Read off the source because the phase is a runtime value: a cell can only prove
    reachability for the phases it drives, and these are the seams that decide
    whether the OTHERS can be reached at all. The behavioural halves live in
    ``test_sessions_extraction.test_a_failed_window_is_named_in_the_fleet_table`` (the CLI) and
    ``test_info_names_a_failed_window_as_plainly_as_it_names_a_drain``.
    """
    for owner, needle in (
        (cli._updating_cell, "update_phase"),
        (info_panel._sessions_section, "update_phase"),
        (OperatorApp._on_runtime_draining, "update_phrase"),
    ):
        source = inspect.getsource(owner)
        assert needle in source, (
            f"{owner.__name__} does not call {needle}: a phase it renders would have to be "
            f"one this build cannot reach (agent review round 1, MINOR 3)"
        )


def test_the_notice_names_the_build_the_frame_carried() -> None:
    """D5: the copy the operator reads IS the copy under test.

    The app used to compose its own sentence — "updating to the newer build" — while
    ``types.update_phrase`` composed a different one, so the frame and the pin were
    two texts about one state, and the notice was the one surface that could have
    named WHICH build was arriving.
    """
    source = inspect.getsource(OperatorApp._on_runtime_draining)
    assert "update_phrase(UPDATING, updating)" in source, source
    assert "update_phrase" in inspect.getsource(
        types
    ), "the vocabulary must stay in types, which is what the notice renders"


def test_the_notices_are_not_one_sentence_wearing_three_names() -> None:
    """The pin's own negative control: it must be possible to FAIL."""
    assert types.update_short(types.UPDATING, WIDEST_PAIR) != types.update_short(
        types.UPDATE_FAILED, WIDEST_PAIR
    )
    assert types.update_phrase(types.UPDATING, WIDEST_PAIR) != types.update_phrase(
        types.UPDATE_FAILED, WIDEST_PAIR
    )
    assert _UPDATING_SHORT[types.UPDATING] != _UPDATING_SHORT[types.UPDATE_FAILED]


def test_no_notice_borrows_the_refusal_clause() -> None:
    """The window must never borrow the departure vocabulary.

    Its whole promise is the opposite of a departure's: the message is held and the
    runtime is coming back. A notice that said "leaving" would be read as the
    incident — "your message is back in the composer" — while the message is spooled.
    """
    from local_operator.tui.app import _DRAIN_NOTICES

    for phase in PHASES:
        sentence = types.update_phrase(phase, WIDEST_PAIR)
        assert sentence not in set(_DRAIN_NOTICES.values())
        assert "will not start a turn" not in sentence, sentence


def test_the_failure_has_a_renderable_incident_cause_of_its_own() -> None:
    """A failed update must be legible in the durable account a successor reads.

    The same requirement ``BUILD_DRAIN_OVERDUE_CAUSE`` had (QA round 1, Q-2): the
    record is gone moments after the process, so the token the runtime records has
    to be one this taxonomy can render, and it must not be ``runtime-retired``,
    which every ordinary handover leaves too.
    """
    cause = types.UPDATE_FAILED_CAUSE
    assert cause in incidents.CUT_OFF_CAUSES
    assert cause != "runtime-retired"
    assert (
        cause not in incidents.DELIBERATE_CUT_OFF_CAUSES
    ), "a bound ending an update is not something the user asked for"
    sentence = incidents.render_cut_off_reason(cause)
    assert "update" in sentence, sentence
    assert (
        types.bound_text(buildwatch.UPDATE_LOCK_S) in sentence
    ), f"the rendered sentence must name the bound it spent: {sentence!r}"


def test_the_fleet_row_carries_the_window_and_the_failure() -> None:
    """``lop sessions --json`` is how a rotation script learns a session is moving.

    The row is built from the record, so the field has to survive the extraction as
    well as the dataclass — a record field no row builder reads is invisible to every
    consumer the column was added for. BOTH fields: the failed one is what D1 found
    reaching no surface at all.
    """
    from dataclasses import replace

    line = SessionLine()
    assert hasattr(line, "updating"), "the fleet row has no place for the window"
    assert hasattr(line, "update_failed"), "the fleet row has no place for the failure"
    line = replace(line, updating=WIDEST_PAIR)

    source = inspect.getsource(info_collect.session_rows)
    assert '"updating"' in source and "line.updating" in source, source
    assert (
        '"update_failed"' in source and "line.update_failed" in source
    ), "the JSON row is the published contract; the failure must ride it too"


def test_info_names_a_failed_window_as_plainly_as_it_names_a_drain() -> None:
    """D1/D2 on the REAL row builder: two rows that used to be byte-identical.

    ``/info`` rendered a session with an abandoned update exactly as it rendered an
    ordinary idle one. The row builder is the seam — the frames are captured
    separately — so this drives it with both records and asserts they differ, and
    that the failure's own words are the ones the vocabulary pins.
    """
    from local_operator.info.model import InfoSnapshot, SessionsInfo
    from local_operator.session.runtime.types import LEAVING_FOR_BUILD

    idle = SessionLine(
        pid=1, kind="daemon", state="live", conversation_name="D: ordinary idle", session_id="d"
    )
    failed = SessionLine(
        pid=2,
        kind="daemon",
        state="live",
        conversation_name="C: update failed",
        session_id="c",
        update_failed=WIDEST_PAIR,
    )
    draining = SessionLine(
        pid=3,
        kind="daemon",
        state="live",
        conversation_name="B: drain",
        session_id="b",
        leaving=LEAVING_FOR_BUILD,
    )

    def rows(*lines: SessionLine) -> list[str]:
        body = info_panel._Body(width=110)
        info_panel._sessions_section(
            body,
            InfoSnapshot(sessions=SessionsInfo(lines=lines, total=len(lines), live=len(lines))),
        )
        return [row.plain for row in body.lines]

    idle_rows, failed_rows, drain_rows = rows(idle), rows(failed), rows(draining)
    assert failed_rows != idle_rows, (
        "a failed window must not render byte-identical to an idle session (D1): "
        f"{failed_rows!r}"
    )
    assert any("update failed" in row for row in failed_rows), failed_rows
    # The WIDE frame is where D2 measured the window missing: the fact belongs in the
    # metas, where the drain's phrase already is, not only on the narrow shelf form.
    assert any(
        "updating → 0.59.11@ead71b6" in row
        for row in rows(
            SessionLine(
                pid=4,
                kind="daemon",
                state="live",
                conversation_name="A: update window",
                session_id="a",
                updating=WIDEST_PAIR,
            )
        )
    ), "the open window must be named in the wide metas, not only on the shelf"
    assert any("leaving for the build" in row for row in drain_rows), drain_rows

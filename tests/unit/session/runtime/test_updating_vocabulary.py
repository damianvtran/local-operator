"""Every phase of an update window is taught to every surface that renders one.

The sibling of ``test_leaving_vocabulary.py``, and it exists for the failure that
file was written about (design round 1, D1/D2/D3; QA round 1, Q-1): a state
published on the record and painted by nobody, because each reader FALLS BACK to
a sentence about a different trigger. There the fallback told the operator the
session was finishing in-flight work at the instant it had given up on it; here
the fallback would tell them the session is leaving when the truth is that it is
COMING BACK and their message is already on the successor's spool.

The list is data — ``types.PUBLISHED_UPDATE_PHASES`` — and this file walks it, so
adding a phase without teaching it to all four surfaces fails in the suite rather
than on an operator's screen.

The surfaces are keyed on the PHASE TOKEN and not on the phrase, which is the one
structural difference from the leaving vocabulary: every updating sentence carries
a build pair, so a table keyed by the full sentence could not be written at all.
The sentence itself is composed in exactly one place (``types.update_phrase``) and
the fleet cell in exactly one other (``types.update_short``).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from local_operator import buildwatch, cli, incidents
from local_operator.info import collect as info_collect
from local_operator.info.model import SessionLine
from local_operator.session.runtime import types
from local_operator.tui.app import _UPDATING_NOTICES, UPDATING_NOTICE_OTHER
from local_operator.tui.widgets.info_panel import _UPDATING_SHORT, _UPDATING_SHORT_OTHER
from local_operator.update import BuildStamp

PHASES: tuple[str, ...] = types.PUBLISHED_UPDATE_PHASES


#: The widest pair a fleet cell has to print, built the way a runtime builds one —
#: through ``BuildStamp.label()``, which is the ONLY producer of the labels that reach
#: this field. Hand-typing a longer pair would size the column for a shape no runtime
#: can emit, which is the "legend at a width no caller can produce" trap AGENTS.md
#: records (the picker footer measured at 74 cells, its legend wanting 88).
WIDEST_PAIR = buildwatch.update_pair_text(
    BuildStamp(version="0.59.9", source_ref=""),
    BuildStamp(version="0.59.11", source_ref="ead71b673a9a24e6925338bcb51315e0ac5d44f"),
)


def test_the_phases_are_enumerable_and_have_no_duplicates() -> None:
    assert PHASES, "the tuple a consumer pin iterates cannot be empty"
    assert len(set(PHASES)) == len(PHASES), PHASES


def test_the_fleet_column_is_sized_by_the_widest_published_cell() -> None:
    """The column width IS the cell's width, and the pair is the widest input."""
    widest = max(len(types.update_short(phase, WIDEST_PAIR)) for phase in PHASES)
    assert cli.UPDATING_COLUMN_WIDTH >= widest, (
        f"the LEAVING-style fleet column ({cli.UPDATING_COLUMN_WIDTH}) truncates the widest "
        f"updating cell ({widest}): {types.update_short(types.UPDATING, WIDEST_PAIR)!r}"
    )


@pytest.mark.parametrize("phase", PHASES)
def test_the_viewer_notice_knows_every_phase(phase: str) -> None:
    notice = _UPDATING_NOTICES.get(phase)
    assert notice is not None, (
        f"{phase!r} has no notice in the app's table: whichever surface paints this "
        f"phase would fall back to UPDATING_NOTICE_OTHER ({UPDATING_NOTICE_OTHER!r}), "
        f"the sentence reserved for a phase this build cannot name at all"
    )


@pytest.mark.parametrize("phase", PHASES)
def test_the_narrow_frame_form_knows_every_phase(phase: str) -> None:
    short = _UPDATING_SHORT.get(phase)
    assert short is not None and short != _UPDATING_SHORT_OTHER, (
        f"{phase!r} has no shelf form: below the note floor the row would degrade to "
        f"{_UPDATING_SHORT_OTHER!r}"
    )


def test_the_notices_are_not_one_sentence_wearing_three_names() -> None:
    """The pin's own negative control: it must be possible to FAIL.

    A table mapping every phase to one notice would pass every assertion above —
    including for the FAILED phase, whose whole point is that it says something
    the other two may not (the runtime is still on the OLD build, and the update
    is reportable).
    """
    assert _UPDATING_NOTICES[types.UPDATING] != _UPDATING_NOTICES[types.UPDATE_FAILED]
    assert _UPDATING_NOTICES[types.UPDATING] != _UPDATING_NOTICES[types.UPDATING_DONE]
    assert _UPDATING_SHORT[types.UPDATING] != _UPDATING_SHORT[types.UPDATE_FAILED]


def test_the_error_notice_is_not_a_leaving_notice() -> None:
    """The updating window must never borrow the departure vocabulary.

    Its whole promise is the opposite of a departure's: the message is held and
    the runtime is coming back. A notice that said "leaving" would be read as the
    incident — "your message is back in the composer" — while the message is in
    fact spooled.
    """
    from local_operator.tui.app import _DRAIN_NOTICES

    for phase in PHASES:
        assert _UPDATING_NOTICES[phase] not in set(_DRAIN_NOTICES.values())
        assert (
            "will not start a turn" not in _UPDATING_NOTICES[phase]
        ), f"{phase!r} borrows the refusal's clause: {_UPDATING_NOTICES[phase]!r}"


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


def test_the_fleet_row_carries_the_window() -> None:
    """``lop sessions --json`` is how a rotation script learns a session is moving.

    The row is built from the record, so the field has to survive the
    extraction as well as the dataclass — a record field no row builder reads is
    invisible to every consumer the column was added for.
    """
    import inspect

    line = SessionLine()
    assert hasattr(line, "updating"), "the fleet row has no place for the window"
    line = replace(line, updating=buildwatch.update_pair_text("0.59.9", "0.59.11@ead71b6"))

    # The JSON row builder is the published contract, and ``lop sessions``
    # renders the column from the same row; walk the builder directly, because
    # a row a consumer cannot read is the failure this pins.
    source = inspect.getsource(info_collect.session_rows)
    assert '"updating"' in source, "the JSON row is the published contract; the window must ride it"
    assert "line.updating" in source, source

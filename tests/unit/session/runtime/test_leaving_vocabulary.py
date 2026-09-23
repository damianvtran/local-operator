"""Every published leaving phrase is taught to every surface that renders one.

Four tables turn a phrase into words a person reads, and each is keyed BY the
phrase: ``tui.app._DRAIN_NOTICES`` (the notice a viewer paints),
``tui.widgets.info_panel._LEAVING_SHORT`` (the narrow-frame form),
``session.errors._TRIGGER_FOR_LEAVING`` (which head a refusal takes) and
``cli.LEAVING_COLUMN_WIDTH`` (the width of the fleet column the phrase is printed
in). A phrase missing from one of them is not a missing entry — every one of those
readers FALLS BACK, and the fallbacks are sentences about a trigger the reader
does not have, so the failure is a wrong sentence rather than a blank.

That is what happened to the bounded handover when it first shipped (design round
1, D1/D2/D3; QA round 1, Q-1): the record and the frame carried the new phrase, and
the app painted "this session is finishing in-flight work first, so a new message
will not start a turn" — at the instant the runtime had given up on that work — a
sentence byte-identical to the one it paints for a trigger it cannot place at all.
The fleet row lost every word of the state below an 82-column terminal, and
``lop sessions`` cut the bound off the end of the column.

SO THE LIST IS DATA, in ``types.PUBLISHED_LEAVING_PHRASES``, and this file walks it.
Adding a phrase without teaching it to all four surfaces fails here, in the suite,
instead of on an operator's screen.
"""

from __future__ import annotations

import pytest

from local_operator import cli, incidents
from local_operator.session import errors
from local_operator.session.runtime import types
from local_operator.tui.app import _DRAIN_NOTICES, DRAIN_NOTICE_OTHER
from local_operator.tui.widgets.info_panel import _LEAVING_SHORT, _LEAVING_SHORT_OTHER

PHRASES: tuple[str, ...] = types.PUBLISHED_LEAVING_PHRASES


def test_the_vocabulary_is_enumerable_and_has_no_duplicates() -> None:
    assert PHRASES, "the tuple a consumer pin iterates cannot be empty"
    assert len(set(PHRASES)) == len(PHRASES), PHRASES


def test_the_leaving_column_is_sized_by_the_widest_published_phrase() -> None:
    """The column width IS the phrase's width — pinned AT ITS OWN SEAM.

    The assertion that the width EQUALS the widest published phrase lives in
    ``tests/unit/info/test_sessions_extraction.py``, the file that owns the ``lop
    sessions`` render, because that is where the cut would actually happen. What
    this file contributes is the LIST both pins walk:``PUBLISHED_LEAVING_PHRASES``.
    A phrase that overflows the column is then a suite failure here as well as
    there, rather than a cell quietly trimmed on an operator's screen.
    """
    assert cli.LEAVING_COLUMN_WIDTH >= max(len(phrase) for phrase in PHRASES)


@pytest.mark.parametrize("phrase", PHRASES)
def test_the_viewer_notice_knows_every_phrase(phrase: str) -> None:
    notice = _DRAIN_NOTICES.get(phrase)
    assert notice is not None, (
        f"{phrase!r} has no notice: the app would paint DRAIN_NOTICE_OTHER "
        f"({DRAIN_NOTICE_OTHER!r}), which is the sentence reserved for a trigger this "
        f"build cannot name at all"
    )


@pytest.mark.parametrize("phrase", PHRASES)
def test_the_narrow_frame_form_knows_every_phrase(phrase: str) -> None:
    short = _LEAVING_SHORT.get(phrase)
    assert short is not None and short != _LEAVING_SHORT_OTHER, (
        f"{phrase!r} has no shelf form: below the note floor the row would degrade to "
        f"{_LEAVING_SHORT_OTHER!r}, the one state this panel documents as having been "
        f"fixed for exactly that regression"
    )


@pytest.mark.parametrize("phrase", PHRASES)
def test_the_refusal_trigger_table_knows_every_phrase(phrase: str) -> None:
    trigger = errors._TRIGGER_FOR_LEAVING.get(phrase)
    assert trigger in (errors.RuntimeRetiring.SIGNAL, errors.RuntimeRetiring.BUILD), (
        f"{phrase!r} establishes no trigger: a refusal at that instant would take the "
        f"unnamed head, which is the vaguer sentence for the more serious departure"
    )


def test_the_ordinary_build_phrase_still_earns_the_ordinary_sentence() -> None:
    """The pin's own negative control: it must be possible to FAIL.

    Without this, a table that mapped every phrase to one notice would pass every
    assertion above — including for a phrase whose whole point is that it says
    something the others may not.
    """
    assert (
        _DRAIN_NOTICES[types.LEAVING_FOR_BUILD] != _DRAIN_NOTICES[types.LEAVING_FOR_BUILD_OVERDUE]
    )
    assert (
        _LEAVING_SHORT[types.LEAVING_FOR_BUILD] != _LEAVING_SHORT[types.LEAVING_FOR_BUILD_OVERDUE]
    )


def test_the_bounded_handover_has_a_cut_off_cause_of_its_own() -> None:
    """A forced handover must be legible in the record a successor reads.

    The durable row is the only account that outlives the process — ``lop
    sessions --json`` loses the record ~97 ms after the escalation — so the token
    the rung records has to be one this taxonomy can render, and it must not be
    ``runtime-retired``, which every ordinary build handover leaves too (QA round
    1, Q-2).
    """
    cause = types.BUILD_DRAIN_OVERDUE_CAUSE
    assert cause in incidents.CUT_OFF_CAUSES
    assert cause != "runtime-retired"
    assert (
        cause not in incidents.DELIBERATE_CUT_OFF_CAUSES
    ), "a bound cutting a turn is not something the user asked for"
    sentence = incidents.render_cut_off_reason(cause)
    assert "no movement" in sentence, sentence
    assert (
        types.bound_text(types.BUILD_DRAIN_PROGRESS_S) in sentence
    ), f"the rendered sentence must name the bound: {sentence!r}"

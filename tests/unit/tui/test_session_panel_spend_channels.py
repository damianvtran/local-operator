"""The /session "Spend by channel" table: published object only, aligned columns.

Round 1's design review drove this file's shape: the section is a TABLE with a
fixed right-aligned money column and its basis tag read per row (D1/D4), an
unknown total prints ``$—`` rather than a fabricated zero (D2), an untracked
session states its scope in plain words and carries the lower-bound mark (D3),
and a row with no recorded unit count prints no count at all (Q6).
"""

from __future__ import annotations

import re
from dataclasses import replace

from local_operator.session.channel_spend import (
    ChannelSpendRecord,
    ChildrenSnapshot,
    InferenceSnapshot,
    combine,
    fold_records,
)
from local_operator.session.frontend_state import FrontendSpendChannels
from local_operator.tui.widgets.session_panel import (
    SessionDiagnostics,
    _Body,
    _draw_spend_channels,
)
from tests.unit.tui.test_session_panel import runtime


def payload(
    *, tracked: bool = True, children: ChildrenSnapshot | None = None
) -> FrontendSpendChannels:
    records = [
        ChannelSpendRecord(
            record_id="image:a",
            rev=1,
            channel="image",
            provider="radient",
            model="gpt-image-2",
            units=1,
            unit="images",
            amount_micro=61000,
            billing_basis="billed",
            cost_source="server_reported",
            status="ok",
        ),
        ChannelSpendRecord(
            record_id="tts:a",
            channel="tts",
            provider="radient",
            model="elevenlabs",
            units=420,
            unit="chars",
            amount_micro=None,
            status="ok",
        ),
        ChannelSpendRecord(
            # A legacy row recovered without a unit count: QA round 1, Q6.
            record_id="image:legacy",
            channel="image",
            provider="openai-sub",
            model="gpt-image-2",
            units=0,
            unit="",
            amount_micro=53000,
            billing_basis="estimated",
            cost_source="catalogue",
            status="ok",
        ),
    ]
    snapshot = combine(
        fold_records(records).rows(),
        inference=InferenceSnapshot(
            micro=900000,
            calls=10,
            priced_calls=10,
            knowledge="exact",
            by_identity={
                "anthropic/claude-sonnet-5-5": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-5-5",
                    "micro": 900000,
                    "calls": 10,
                    "unpriced": 0,
                }
            },
        ),
        children=children or ChildrenSnapshot(),
        tracked=tracked,
        lost=False,
    )
    return FrontendSpendChannels.model_validate(snapshot)


def rendered(runtime_diag: SessionDiagnostics, width: int = 100) -> str:
    body = _Body(width=width)
    assert _draw_spend_channels(body, runtime_diag) is True
    return body.to_text().plain


def test_section_renders_the_published_table() -> None:
    text = rendered(replace(runtime(), spend_channels=payload()))
    assert "Spend by channel" in text
    # Inference is its own row with the model identity and its own basis tag:
    # the money is stated, its billing BASIS is the missing part.
    assert "inference · anthropic/claude-sonnet-5-5" in text
    assert "$0.900" in text and "basis not recorded · 10 calls" in text
    # The billed image carries its tag beside its money, and a plan/estimated
    # recovery row reads as estimated — no "sub-equiv" shorthand anywhere.
    assert "$0.061" in text and "billed · 1 image" in text
    legacy_line = next(line for line in text.splitlines() if "openai-sub" in line)
    assert "$0.053" in legacy_line and "estimated" in legacy_line, legacy_line
    # The unsized tts row prints $— and a reason, and never a fabricated zero.
    assert "tts · radient/elevenlabs" in text and "$—" in text
    assert "price not stated · 420 chars" in text
    # The Total carries the lower-bound mark and the app's own legend words.
    assert "$1.01+" in text, text  # mark attached, dim: 61000+53000+900000 micros
    assert "+ lower bound" in text
    # The basis footer reconciles: the buckets plus the count account for the
    # whole total, with no count sitting inside a dollar sentence.
    assert "By basis:" in text
    assert "billed $0.061" in text
    assert "basis not recorded $0.900" in text
    assert "1 call with no price recorded" in text
    # No internal vocabulary.
    assert "knowledge:" not in text
    assert "sub-equiv" not in text
    assert "not_tracked" not in text


def test_unknown_total_prints_the_unknown_cell_not_zero() -> None:
    """D2: an unstateable total is ``$—``; only a real zero may print $0.0000."""
    only_unsized = [
        ChannelSpendRecord(
            record_id="tts:x",
            channel="tts",
            provider="radient",
            model="elevenlabs",
            units=12,
            unit="chars",
            amount_micro=None,
            status="ok",
        )
    ]
    snapshot = combine(
        fold_records(only_unsized).rows(),
        inference=InferenceSnapshot(),
        children=ChildrenSnapshot(),
        tracked=True,
        lost=False,
    )
    text = rendered(
        replace(runtime(), spend_channels=FrontendSpendChannels.model_validate(snapshot))
    )
    assert "$0.0000" not in text, text
    assert "$—" in text


def test_untracked_session_states_its_scope_and_marks_the_total() -> None:
    """D3: the notice names the missing channels; the total is a lower bound."""
    text = rendered(replace(runtime(), spend_channels=payload(tracked=False)))
    assert "wasn't recorded for this conversation" in text
    assert "Channels not tracked" not in text
    assert "+ lower bound" in text, "an untracked total can never read as exact"
    assert "knowledge: exact" not in text


def test_row_without_a_unit_count_prints_no_count() -> None:
    """Q6: ``$0.053   0 images`` was a fabricated zero; the note is omitted."""
    text = rendered(replace(runtime(), spend_channels=payload()))
    assert not re.search(r"\b0 (images|chars|reads|searches|calls)\b", text), text
    line = next(line for line in text.splitlines() if "openai-sub" in line)
    assert "$0.053" in line and "images" not in line, line


def test_money_column_is_aligned_across_rows() -> None:
    """D4: every money cell ends at one column; the Total's included."""
    text = rendered(replace(runtime(), spend_channels=payload()))
    # ``strip``: the footer's first line is indented like the rows now (design
    # round 2, D2-1), so a bare prefix check would let it into the money-column
    # set it is not part of.
    lines = [
        line
        for line in text.splitlines()
        if "$" in line and not line.strip().startswith("By basis")
    ]
    ends = set()
    for line in lines:
        start = line.rfind("$")
        ends.add(start + len(line[start:].split(" ")[0]))
    assert len(ends) == 1, f"ragged money column: {sorted(ends)} in {lines}"


def test_the_capped_aggregate_row_is_named_and_marked_not_itemised() -> None:
    """D2-3: the wire cap's ``other channels (N)`` row may not render as ``other``.

    The row carries the grouped rows' money and the worst knowledge; a bare
    "other" hid the count, the money's provenance and the fact that anything
    was grouped. The label comes from the wire; the note says the rest.
    """
    snapshot = payload().model_dump(mode="json")
    snapshot["rows"] = list(snapshot["rows"]) + [
        {
            "channel": "other",
            "provider": "",
            "model": "",
            "label": "other channels (4)",
            "units": None,
            "unit": "",
            "amount_micro": 1234,
            "knowledge": "partial",
            "basis": [],
            "price_versions": [],
        }
    ]
    text = rendered(
        replace(runtime(), spend_channels=FrontendSpendChannels.model_validate(snapshot))
    )
    line = next(line for line in text.splitlines() if "other channels" in line)
    assert "not itemised" in line, line


def test_short_rungs_keep_the_disclosures_at_the_canonical_80() -> None:
    """D2-2: 12 cells of note budget must not erase the Total's legend or a tag.

    The budget at an 80-column TERMINAL is 12 cells (the card's content width
    lands in the 62–78 band, ``_spend_note_budget``), which fits neither
    "+ lower bound" (13) nor "basis not recorded" (19); before round 2 the
    ladder fell through to the unit count and the biggest rows said "10 calls"
    with no basis, and the mark had no legend at all at the width where it
    needs one most.
    """
    text = rendered(replace(runtime(), spend_channels=payload()), width=65)
    total = next(line for line in text.splitlines() if "Total" in line)
    assert "lower bound" in total, total
    inference = next(line for line in text.splitlines() if "inference ·" in line)
    assert inference.rstrip().endswith("unrecorded"), inference


def test_children_get_their_own_row_when_they_carry_money() -> None:
    """D7b: the children total is a row, in the user's words, not a footer."""
    text = rendered(
        replace(
            runtime(),
            spend_channels=payload(
                children=ChildrenSnapshot(total_micro=210000, knowledge="floor")
            ),
        )
    )
    assert "subagents · included in total" in text
    assert "$0.210" in text
    assert "floor knowledge" not in text


def test_section_sheds_without_a_published_object() -> None:
    body = _Body(width=100)
    assert _draw_spend_channels(body, runtime()) is False
    assert body.to_text().plain.strip() == "", "no published object means no section"


def test_notes_shed_whole_at_narrow_widths() -> None:
    """D6: at 80 cells the long basis tag sheds rather than cropping mid-word."""
    text = rendered(
        replace(
            runtime(),
            spend_channels=FrontendSpendChannels.model_validate(
                combine(
                    fold_records(
                        [
                            ChannelSpendRecord(
                                record_id="image:sub",
                                channel="image",
                                provider="openai-sub",
                                model="gpt-image-2-very-long-id",
                                units=1,
                                unit="images",
                                amount_micro=53000,
                                billing_basis="subscription_api_equivalent",
                                cost_source="catalogue",
                                status="ok",
                            )
                        ]
                    ).rows(),
                    inference=InferenceSnapshot(),
                    children=ChildrenSnapshot(),
                    tracked=True,
                    lost=False,
                )
            ),
        ),
        width=80,
    )
    for line in text.splitlines():
        assert not line.endswith("("), line
        # A crop is a line that cuts the parenthetical mid-word; the full
        # spelling is fine wherever it appears whole.
        assert "(API price" not in line or "(API price)" in line, line

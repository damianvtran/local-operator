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
from local_operator.tui.costs import UNKNOWN_COST_CELL
from local_operator.tui.widgets.session_panel import (
    SessionDiagnostics,
    _Body,
    _draw_spend_channels,
)
from tests.unit.tui.test_session_panel import runtime


def payload(
    *,
    tracked: bool = True,
    children: ChildrenSnapshot | None = None,
    pending: list[dict[str, object]] | None = None,
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
        pending=pending or (),
    )
    return FrontendSpendChannels.model_validate(snapshot)


def voice_only_payload() -> FrontendSpendChannels:
    """The voice-only shape: nothing folded, two voice charges pending.

    The exact shape the PR's real-path evidence produced — ``rows []``,
    ``total_micro 0``, ``knowledge unknown``, ``tracked`` (the session marker
    exists) — and the one design round 1 (D1) measured as showing NO section
    at all.
    """
    snapshot = combine(
        [],
        tracked=True,
        pending=[
            {"record_id": "tts:one", "channel": "tts", "ts_ms": 1},
            {"record_id": "stt:two", "channel": "stt", "ts_ms": 2},
        ],
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
    # The unsized tts row prints $— and states WHY (a 200 that carried no
    # price), and never a fabricated zero.
    assert "tts · radient/elevenlabs" in text and "$—" in text
    assert "server stated no price" in text
    assert "price not stated" not in text
    # The Total carries the lower-bound mark and the app's own legend words.
    assert "$1.01+" in text, text  # mark attached, dim: 61000+53000+900000 micros
    assert "+ lower bound" in text
    # The basis footer reconciles: the buckets plus the count account for the
    # whole total, with no count sitting inside a dollar sentence.
    assert "By basis:" in text
    assert "Total split: $0.061 billed" in text
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
        if "$" in line and not line.strip().startswith(("By basis", "Total split:"))
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


def test_pending_voice_spend_is_stated_beside_the_total() -> None:
    """D1 (design round 1): recorded-but-excluded voice money must SAY so.

    The total is unchanged (the same figure as every other surface, and the
    money genuinely is not in it), but the section now states the exclusion in
    plain words with the count — never the `$—` class, which means "price not
    stated" and would read as if the amount were unknown.
    """
    text = rendered(
        replace(
            runtime(),
            spend_channels=payload(
                pending=[
                    {"record_id": "tts:one", "channel": "tts", "ts_ms": 1},
                    {"record_id": "stt:two", "channel": "stt", "ts_ms": 2},
                ]
            ),
        )
    )
    assert "Voice spend for this conversation is recorded but not yet included" in text, text
    assert "2\u00a0requests." in text, text  # NBSP: count + noun stay one unit (D2-2)
    # The total keeps its own knowledge-derived figure and mark untouched.
    assert "$1.01+" in text and "+ lower bound" in text
    # No wire vocabulary, and no fabricated zero for the excluded money.
    assert "pending]" not in text and "pending_owner_append" not in text
    assert "$0.0000" not in text, "$0 must never stand in for the excluded money"


def test_a_voice_only_session_draws_the_state_instead_of_nothing() -> None:
    """D1: the voice-only shape (rows [], total 0) used to draw nothing at all.

    Charged money read as non-existent on the exact surface whose job is to
    account for spend. The section now draws — header and the pending note —
    and the Total prints `$—` (nothing is stateable about what the fold holds)
    rather than a fabricated ``$0.0000``.
    """
    text = rendered(replace(runtime(), spend_channels=voice_only_payload()))
    assert "Spend by channel" in text, text
    assert "Voice spend for this conversation is recorded but not yet included" in text, text
    assert "2\u00a0requests." in text
    assert UNKNOWN_COST_CELL in text, "an empty fold totals $—, never a fabricated zero"
    assert "$0.0000" not in text


# -- one decision model, one row, and a stated reason ------------------------


def classification_payload(*records: ChannelSpendRecord) -> FrontendSpendChannels:
    return FrontendSpendChannels.model_validate(
        combine(fold_records(list(records)).rows(), tracked=True)
    )


def decision_record(
    record_id: str,
    *,
    model: str,
    amount_micro: int | None,
    status: str = "ok",
) -> ChannelSpendRecord:
    return ChannelSpendRecord(
        record_id=record_id,
        channel="classification",
        provider="radient",
        model=model,
        units=1,
        unit="calls",
        amount_micro=amount_micro,
        billing_basis="billed",
        cost_source="server_reported" if amount_micro is not None else "none",
        status=status,
    )


def test_a_failed_classification_row_says_no_charge_not_price_not_stated() -> None:
    text = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record(
                    "classification:failed", model="jev-1.13.0", amount_micro=None, status="failed"
                )
            ),
        )
    )
    line = next(line for line in text.splitlines() if "classification" in line)
    assert "no charge" in line, line
    assert "request failed · not billed" in line, line
    assert "price not stated" not in line, line
    assert "$—" not in line, "the failure is a stated not-billed outcome, not an unknown"


def test_a_failed_only_row_keeps_its_reason_at_80_columns() -> None:
    """The 65-cell body of an 80x24 terminal has a 12-cell note budget.

    The failed-only ladder keeps a rung that fits it; a failed-only row is the
    one case where ``no charge`` and ``not billed`` describe the whole row.
    """
    text = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record(
                    "classification:failed", model="jev-1.13.0", amount_micro=None, status="failed"
                )
            ),
        ),
        width=65,
    )
    line = next(line for line in text.splitlines() if "classification" in line)
    assert "no charge" in line, line
    assert "not billed" in line, line


def test_two_historical_spellings_render_as_one_classification_row() -> None:
    text = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record("classification:ok", model="jev-1.13.0", amount_micro=11),
                decision_record(
                    "classification:failed", model="jev-1.13", amount_micro=None, status="failed"
                ),
            ),
        )
    )
    lines = [line for line in text.splitlines() if "classification" in line]
    assert len(lines) == 1, lines
    assert "jev-1.13" in lines[0] and "jev-1.13.0" not in lines[0], lines[0]
    assert "<$0.0001" in lines[0], lines[0]
    # A MIXED row must attribute the zero to the failed attempt, never call the
    # priced group unbilled (design round 1, D-1).
    assert "failed attempt added no charge" in lines[0], lines[0]
    assert "not billed" not in lines[0] and "unbilled" not in lines[0], lines[0]


def test_a_mixed_row_at_80_columns_stays_truthful() -> None:
    """D-1 at the real 80x24 body width: the 11-cell rung must fit.

    The unqualified ``not billed`` used to appear here beside a positive cost.
    """
    text = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record("classification:ok", model="jev-1.13.0", amount_micro=11),
                decision_record(
                    "classification:failed", model="jev-1.13", amount_micro=None, status="failed"
                ),
            ),
        ),
        width=65,
    )
    line = next(line for line in text.splitlines() if "classification" in line)
    assert "failed: +$0" in line, line
    assert "not billed" not in line and "unbilled" not in line, line
    assert "<$0.0001" in line, line


def test_a_mixed_row_uses_the_19_cell_rung_at_the_100_column_body() -> None:
    """At the canonical 83-cell body (24-cell budget) the increment is explicit."""
    text = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record("classification:ok", model="jev-1.13.0", amount_micro=11),
                decision_record(
                    "classification:failed", model="jev-1.13", amount_micro=None, status="failed"
                ),
            ),
        ),
        width=83,
    )
    line = next(line for line in text.splitlines() if "classification" in line)
    assert "failed attempt: +$0" in line, line
    assert "not billed" not in line and "unbilled" not in line, line


def test_a_cancelled_classification_row_keeps_the_unknown_cell_and_says_why() -> None:
    text = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record(
                    "classification:cancelled",
                    model="jev-1.13.0",
                    amount_micro=None,
                    status="cancelled",
                )
            ),
        )
    )
    line = next(line for line in text.splitlines() if "classification" in line)
    assert "$—" in line, line
    assert "response lost · may be billed" in line, line
    assert "price not stated" not in line, line


def test_cancelled_and_partial_reasons_survive_80_columns() -> None:
    """D-2: a 65-cell body has a 12-cell note budget; the reason must not vanish."""
    cancelled = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record(
                    "classification:cancelled",
                    model="jev-1.13.0",
                    amount_micro=None,
                    status="cancelled",
                )
            ),
        ),
        width=65,
    )
    cancelled_line = next(line for line in cancelled.splitlines() if "classification" in line)
    assert "$—" in cancelled_line and "bill unknown" in cancelled_line, cancelled_line

    partial = rendered(
        replace(
            runtime(),
            spend_channels=classification_payload(
                decision_record("classification:ok", model="jev-1.13", amount_micro=11),
                decision_record("classification:parked", model="jev-1.13", amount_micro=None),
            ),
        ),
        width=65,
    )
    partial_line = next(line for line in partial.splitlines() if "classification" in line)
    assert "some unknown" in partial_line, partial_line
    assert "+$0" not in partial_line, partial_line


def test_payment_split_qualifies_total_and_child_money():
    """The parent's and child's plan equivalents must never read as billed."""
    snapshot = combine(
        [],
        children=ChildrenSnapshot(
            total_micro=12410000,
            by_basis={"billed": 790000, "subscription_api_equivalent": 11620000},
        ),
        tracked=True,
    )
    text = rendered(
        replace(runtime(), spend_channels=FrontendSpendChannels.model_validate(snapshot)), width=100
    )
    assert "$0.790 billed" in text and "$11.62 included (API-equivalent)" in text
    assert "subagents · included in total" in text
    assert "basis not recorded $12.41" not in text
    narrow = rendered(
        replace(runtime(), spend_channels=FrontendSpendChannels.model_validate(snapshot)), width=65
    )
    assert "included (API-equivalent)" in narrow
    assert "mixed basis" in narrow  # the child row only, not a repeated Total qualifier
    assert "By basis:" not in text
    total_line = next(line for line in text.splitlines() if line.strip().startswith("Total "))
    assert "billed" not in total_line and "included" not in total_line

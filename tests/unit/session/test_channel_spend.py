"""Channel spend: the fold, the knowledge matrix and the published wire object.

Every test names the claim it is the evidence for. The load-bearing ones are
the KNOWLEDGE MATRIX (the rule that stops an unknown amount from being painted
as a number), the "None is never 0" rule, and the WIRE CONTRACT pin — the JSON
fixture ``tests/fixtures/spend_channels_v1.json`` is what the UI PR types
against, so a rename here must fail there rather than ship silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.channel_spend import (
    BASIS_BILLED,
    BASIS_ESTIMATED,
    BASIS_NOT_TRACKED,
    BASIS_SUBSCRIPTION,
    CHANNEL_SPEND_CUSTOM_TYPE,
    NOT_TRACKED_MICRO,
    ChannelSpend,
    ChannelSpendRecord,
    ChildrenSnapshot,
    InferenceSnapshot,
    combine,
    fold_records,
    map_image_cost_labels,
    new_record_id,
    normalise_basis,
    records_from_details,
)
from local_operator.session.frontend_state import FrontendSpendChannels

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "spend_channels_v1.json"


def record(
    record_id: str = "search:one",
    *,
    channel: str = "search",
    amount_micro: int | None = 8000,
    basis: str = BASIS_ESTIMATED,
    status: str = "ok",
    rev: int = 0,
    units: float = 1,
    unit: str = "searches",
    provider: str = "tavily",
    model: str = "",
    price_version: str = "",
    ts_ms: int = 0,
) -> ChannelSpendRecord:
    return ChannelSpendRecord(
        record_id=record_id,
        rev=rev,
        ts_ms=ts_ms,
        channel=channel,
        provider=provider,
        model=model,
        units=units,
        unit=unit,
        amount_micro=amount_micro,
        billing_basis=basis,
        status=status,
        price_version=price_version,
    )


def payload(
    records: list[ChannelSpendRecord],
    *,
    inference: InferenceSnapshot | None = None,
    children: ChildrenSnapshot | None = None,
    tracked: bool = True,
    lost: bool = False,
) -> dict[str, Any]:
    return combine(
        fold_records(records).rows(),
        inference=inference or InferenceSnapshot(),
        children=children or ChildrenSnapshot(),
        tracked=tracked,
        lost=lost,
    )


# -- the fold ---------------------------------------------------------------


def test_fold_keeps_the_highest_rev_and_is_idempotent() -> None:
    """A rev bump supersedes; replaying the SAME record changes nothing."""
    fold = ChannelSpend()
    assert fold.apply(record(amount_micro=8000)) is True
    assert fold.apply(record(amount_micro=8000)) is False, "same rev must not change the fold"
    assert fold.apply(record(amount_micro=53000, rev=1, basis=BASIS_BILLED)) is True
    held = fold.get("search:one")
    assert held is not None and held.amount_micro == 53000
    assert fold.apply(record(amount_micro=99, rev=0)) is False, "a stale rev must not win"
    held = fold.get("search:one")
    assert held is not None and held.amount_micro == 53000
    assert len(fold) == 1


def test_fold_dedups_a_forked_journal_by_record_id() -> None:
    """A fork that copied the parent's rows counts each record once."""
    fork = fold_records([record(), record(), record(record_id="image:x", channel="image")])
    assert len(fork) == 2
    assert fork.total_known_micro() == 16000


def test_from_details_rejects_non_finite_amounts() -> None:
    """MINOR 1: a corrupt ``NaN`` amount reads as NO RECORD, never raises.

    JSON parses ``NaN``/``Infinity`` to floats, and ``int()`` on them raises:
    one bad row would otherwise take the whole session open down with it.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert (
            ChannelSpendRecord.from_details({**record().to_details(), "amount_micro": bad}) is None
        )


def test_record_round_trip_and_malformed_degradation() -> None:
    original = record(price_version="client-search-table-2026-09")
    recalled = ChannelSpendRecord.from_details(original.to_details())
    assert recalled == original
    assert ChannelSpendRecord.from_details(None) is None
    assert ChannelSpendRecord.from_details({"version": 2}) is None
    assert ChannelSpendRecord.from_details({"version": 1}) is None, "no record_id"
    assert ChannelSpendRecord.from_details({"version": 1, "record_id": "x:y", "rev": "2"}) is None


def test_new_record_id_is_channel_prefixed_or_unique() -> None:
    assert new_record_id("image", "req_1") == "image:req_1"
    first, second = new_record_id("image"), new_record_id("image")
    assert first.startswith("image:") and first != second


def test_absent_amount_normalises_the_basis_to_not_tracked() -> None:
    """No amount means no basis: ``estimated`` beside None would be a claim."""
    rec = ChannelSpendRecord(record_id="x:1", channel="image", amount_micro=None)
    assert rec.billing_basis == BASIS_NOT_TRACKED
    kept = ChannelSpendRecord(
        record_id="x:2", channel="image", amount_micro=1, billing_basis=BASIS_ESTIMATED
    )
    assert kept.billing_basis == BASIS_ESTIMATED


# -- the knowledge matrix ---------------------------------------------------


@pytest.mark.parametrize(
    ("status", "amount", "expected"),
    [
        ("ok", 8000, "exact"),
        ("ok", None, "partial"),
        ("cancelled", None, "partial"),
        ("cancelled", 0, "exact"),
        ("failed", None, "exact"),
        ("failed", 0, "exact"),
    ],
)
def test_knowledge_matrix_status_times_amount(
    status: str, amount: int | None, expected: str
) -> None:
    """Every status × known/None combination, with an unrelated inference half.

    The inference half is deliberately exact so the channel arm decides: a
    cancelled-unsettled record and an unreported success both make the total
    ``partial``, while a failed job with no charge claim does not degrade it
    (the documented presumption), and a settled zero IS a stated figure.
    """
    result = payload(
        [record(amount_micro=amount, status=status)],
        inference=InferenceSnapshot(micro=1000, calls=1, priced_calls=1, knowledge="exact"),
    )
    assert result["knowledge"] == expected


def test_unknown_when_nothing_is_stateable() -> None:
    result = payload([record(amount_micro=None)])
    assert result["knowledge"] == "unknown"
    assert result["total_micro"] == 0


def test_empty_session_is_unknown_not_exact_zero() -> None:
    """Nothing stateable — the accumulator's own rule for an empty record."""
    result = payload([])
    assert result["knowledge"] == "unknown"
    assert result["total_micro"] == 0


def test_unpriced_inference_degrades_a_known_channel_total_to_partial() -> None:
    result = payload(
        [record(amount_micro=8000)],
        inference=InferenceSnapshot(micro=0, calls=3, priced_calls=0, knowledge="unknown"),
    )
    assert result["total_micro"] == 8000
    assert result["knowledge"] == "partial", "money exists that inference could not state"


def test_empty_inference_does_not_degrade_a_channel_total() -> None:
    """No calls is not the same fact as calls we could not price."""
    result = payload(
        [record(amount_micro=8000)],
        inference=InferenceSnapshot(micro=0, calls=0, priced_calls=0, knowledge="unknown"),
    )
    assert result["knowledge"] == "exact"


def test_lost_rows_mark_floor_but_partial_outranks_it() -> None:
    assert payload([record()], lost=True)["knowledge"] == "floor"
    assert (
        payload([record(), record(record_id="s:2", amount_micro=None)], lost=True)["knowledge"]
        == "partial"
    )


def test_tracked_flag_semantics() -> None:
    """Untracked + recovered rows = a lower bound; untracked + silence = the figure.

    The flag alone carrying the news was design round 1's D3; the FIX went too
    far the other way and marked every marker-less session, which the operator's
    cold/in-process continuity tests reject (review round 2, M-4): a pre-feature
    session with NO recovered channel rows has nothing to warn about, and
    degrading most of the store's sessions is noise, not honesty. Degradation
    needs EVIDENCE — recovered rows, or a lost-money row — so the tests below
    pin both shapes.
    """
    with_rows = payload([record()], tracked=False)
    assert with_rows["tracked"] is False and with_rows["knowledge"] == "partial"
    without_rows = payload(
        [],
        inference=InferenceSnapshot(micro=500, calls=1, priced_calls=1, knowledge="exact"),
        tracked=False,
    )
    assert without_rows["tracked"] is False
    assert without_rows["knowledge"] == "exact", "no evidence of missed spend = no mark"
    assert without_rows["rows"][0]["channel"] == "inference"
    with_lost = payload(
        [],
        inference=InferenceSnapshot(micro=500, calls=1, priced_calls=1, knowledge="exact"),
        tracked=False,
        lost=True,
    )
    assert with_lost["knowledge"] == "partial", "a lost-money row is evidence too"


def test_a_stated_amount_with_no_basis_reconciles_the_buckets() -> None:
    """m1 / QA Q10: money whose basis is not tracked still lands in a bucket.

    ``from_details`` defaults a missing basis to ``not_tracked`` while keeping
    the amount, and PR-3's server-object adapter can produce the same shape —
    previously such money was counted (``not_tracked_calls``) but sat in NO
    bucket, breaking the "buckets sum to the total by construction" rule.
    """
    mixed = payload(
        [
            record(record_id="image:a", channel="image", amount_micro=53000, basis=BASIS_BILLED),
            record(record_id="tts:b", channel="tts", amount_micro=5000, basis=BASIS_NOT_TRACKED),
            record(record_id="stt:c", channel="stt", amount_micro=None, basis=BASIS_NOT_TRACKED),
        ],
        inference=InferenceSnapshot(micro=8000, calls=2, priced_calls=2, knowledge="exact"),
    )
    buckets = mixed["by_basis"]
    assert buckets[NOT_TRACKED_MICRO] == 8000 + 5000, "inference remainder + the unbased amount"
    assert buckets["not_tracked_calls"] == 1, "only the row with NO amount is a count"
    total = sum(
        buckets[key]
        for key in (BASIS_BILLED, BASIS_SUBSCRIPTION, BASIS_ESTIMATED, NOT_TRACKED_MICRO)
    )
    assert total == mixed["total_micro"] == 66000


def test_a_corrupt_row_is_skipped_without_losing_its_neighbours() -> None:
    """m3: per-row hardening — one NaN row must not zero out the fold.

    Measured in review round 2: a NaN ``ts_ms`` beside a healthy 8,000 µ$ row
    made the WHOLE fold publish ``total 0``. The timestamp is checked for
    finiteness before ``int()`` (NaN passes ``< 0`` and then raises), units
    reject non-finite values (``Infinity`` is invalid JSON on the wire), and a
    row that still raises is skipped individually.
    """
    good = record(record_id="image:good", channel="image", amount_micro=8000, basis=BASIS_BILLED)
    nan_row = dict(good.to_details(), record_id="image:nan", ts_ms=float("nan"))
    inf_row = dict(good.to_details(), record_id="image:inf", ts_ms=float("inf"))
    rows = [good.to_details(), nan_row, inf_row]
    recovered = records_from_details(rows)
    assert [r.record_id for r in recovered] == ["image:good"], "the good row survives"
    assert (
        ChannelSpendRecord.from_details({"version": 1, "record_id": "r", "ts_ms": float("inf")})
        is None
    )
    assert (
        ChannelSpendRecord.from_details(
            {"version": 1, "record_id": "r", "ts_ms": 1, "units": float("inf")}
        )
        is None
    )
    # And the record model itself normalises a non-finite count that slipped
    # through some future constructor: no Infinity can reach a wire.
    assert (
        ChannelSpendRecord(
            record_id="x", channel="image", provider="", model="", ts_ms=1, units=float("inf")
        ).units
        == 0.0
    )


def test_children_are_counted_once_and_propagate_their_knowledge() -> None:
    result = payload(
        [],
        inference=InferenceSnapshot(micro=100, calls=1, priced_calls=1, knowledge="exact"),
        children=ChildrenSnapshot(total_micro=250, knowledge="partial"),
    )
    assert result["total_micro"] == 350
    assert result["children"] == {"total_micro": 250, "knowledge": "partial"}
    assert result["knowledge"] == "partial"


# -- rows and by_basis ------------------------------------------------------


def test_group_rows_aggregate_and_unknown_groups_carry_null_amount() -> None:
    records = [
        record(record_id="image:a", channel="image", amount_micro=53000, basis=BASIS_BILLED),
        record(record_id="image:b", channel="image", amount_micro=None, status="cancelled"),
    ]
    rows = {r["channel"]: r for r in payload(records)["rows"]}
    assert rows["image"]["amount_micro"] == 53000
    assert rows["image"]["units"] == 2.0
    assert rows["image"]["knowledge"] == "partial"
    assert rows["image"]["basis"] == [BASIS_BILLED, BASIS_NOT_TRACKED], "sorted"

    only_unknown = payload([record(record_id="image:c", channel="image", amount_micro=None)])
    assert only_unknown["rows"][0]["amount_micro"] is None, "unknown, never a fabricated 0"


def test_subscription_dollars_stay_separate_from_billed() -> None:
    result = payload(
        [
            record(
                record_id="image:sub",
                channel="image",
                amount_micro=53000,
                basis=BASIS_SUBSCRIPTION,
            ),
            record(record_id="image:paid", channel="image", amount_micro=1000, basis=BASIS_BILLED),
        ]
    )
    assert result["by_basis"][BASIS_SUBSCRIPTION] == 53000
    assert result["by_basis"][BASIS_BILLED] == 1000
    assert result["by_basis"]["not_tracked_calls"] == 0
    assert result["total_micro"] == 54000, "the total includes both, the buckets keep them apart"


def test_by_basis_counts_untracked_records_and_failed_unbilled_is_skipped() -> None:
    result = payload(
        [
            record(record_id="image:f", channel="image", amount_micro=None, status="failed"),
            record(record_id="image:c", channel="image", amount_micro=None, status="cancelled"),
        ]
    )
    assert result["by_basis"]["not_tracked_calls"] == 1


def test_inference_rows_by_identity_plus_unattributed_remainder() -> None:
    inference = InferenceSnapshot(
        micro=900500,
        calls=13,
        priced_calls=13,
        knowledge="exact",
        by_identity={
            "anthropic/claude": {
                "provider": "anthropic",
                "model_id": "claude",
                "micro": 900000,
                "calls": 12,
                "unpriced": 0,
            }
        },
    )
    rows = payload([], inference=inference)["rows"]
    assert len(rows) == 2
    assert rows[0]["model"] == "claude" and rows[0]["units"] == 12.0
    assert rows[1]["label"] == "unattributed"
    assert rows[1]["amount_micro"] == 500, "the remainder is money the rows must still show"


def test_inference_without_by_identity_states_the_total_and_says_not_tracked() -> None:
    inference = InferenceSnapshot(micro=700, calls=4, priced_calls=4, knowledge="exact")
    rows = payload([], inference=inference)["rows"]
    assert len(rows) == 1
    assert rows[0]["model"] == "" and rows[0]["label"] == ""
    assert rows[0]["amount_micro"] == 700
    assert rows[0]["basis"] == [BASIS_NOT_TRACKED]


def test_normalise_basis_maps_the_wave_two_spelling() -> None:
    assert normalise_basis("subscription-api-equivalent") == BASIS_SUBSCRIPTION
    assert normalise_basis("billed") == BASIS_BILLED
    assert normalise_basis("") is None


@pytest.mark.parametrize(
    ("kwargs", "expected_basis", "expected_source"),
    [
        (
            dict(route="radient", cost_source=None, billing_basis=None, has_amount=True),
            "estimated",
            "server_reported",
        ),
        (
            dict(route="radient", cost_source="reported", billing_basis="billed", has_amount=True),
            "billed",
            "server_reported",
        ),
        (
            dict(
                route="openai-sub", cost_source="subscription", billing_basis=None, has_amount=True
            ),
            BASIS_SUBSCRIPTION,
            "catalogue",
        ),
        (
            dict(route="openrouter", cost_source="reported", billing_basis=None, has_amount=True),
            "billed",
            "provider_reported",
        ),
        (
            dict(
                route="xai",
                cost_source="reported",
                billing_basis="subscription-api-equivalent",
                has_amount=True,
            ),
            BASIS_SUBSCRIPTION,
            "provider_reported",
        ),
        (
            dict(route="openai", cost_source="rate_table", billing_basis=None, has_amount=True),
            "estimated",
            "catalogue",
        ),
        (
            dict(route="radient", cost_source=None, billing_basis=None, has_amount=False),
            BASIS_NOT_TRACKED,
            "server_reported",
        ),
        (
            dict(route="openai", cost_source=None, billing_basis=None, has_amount=False),
            BASIS_NOT_TRACKED,
            "none",
        ),
    ],
)
def test_image_label_mapping(
    kwargs: dict[str, Any], expected_basis: str, expected_source: str
) -> None:
    basis, source, _price_version = map_image_cost_labels(
        route=kwargs["route"],
        cost_source=kwargs["cost_source"],
        billing_basis=kwargs["billing_basis"],
        cost_provenance=None,
        has_amount=kwargs["has_amount"],
    )
    assert (basis, source) == (expected_basis, expected_source)


# -- the frozen wire contract ----------------------------------------------


def test_wire_fixture_is_the_golden_payload_and_the_contract() -> None:
    """The fixture is what ``combine`` produces, and its KEYS are the contract.

    The UI PR types against this file. If a key is renamed or dropped, this test
    (and the UI's own contract test) fails on purpose — the alternative is a
    silent wire break discovered on a user's screen.
    """
    records = [
        ChannelSpendRecord(
            record_id="image:req_7f3a",
            rev=1,
            ts_ms=1760000000000,
            session_id="s1",
            channel="image",
            provider="radient",
            model="gpt-image-2",
            units=1,
            unit="images",
            amount_micro=53000,
            billing_basis="billed",
            cost_source="server_reported",
            price_version="Radient GET /tools/media/status cost_usd",
            status="ok",
            request_id="req_7f3a",
        ),
        ChannelSpendRecord(
            record_id="image:req_9b11",
            ts_ms=1760000001000,
            session_id="s1",
            channel="image",
            provider="openai-sub",
            model="gpt-image-2",
            units=1,
            unit="images",
            amount_micro=53000,
            billing_basis=BASIS_SUBSCRIPTION,
            cost_source="catalogue",
            price_version=(
                "OpenAI image-generation pricing (gpt-image-2, 1024x1024 medium, 2026-10-09)"
            ),
            status="ok",
            request_id="req_9b11",
        ),
        ChannelSpendRecord(
            record_id="image:req_c4d2",
            ts_ms=1760000002000,
            session_id="s1",
            channel="image",
            provider="radient",
            model="gpt-image-2",
            units=2,
            unit="images",
            amount_micro=None,
            status="cancelled",
            request_id="req_c4d2",
            detail="cancelled before settlement was observed",
        ),
        ChannelSpendRecord(
            record_id="search:1a2b",
            ts_ms=1760000003000,
            session_id="s1",
            channel="search",
            provider="tavily",
            units=1,
            unit="searches",
            amount_micro=8000,
            billing_basis=BASIS_ESTIMATED,
            cost_source="catalogue",
            price_version="client-search-table-2026-09",
        ),
        ChannelSpendRecord(
            record_id="read:3c4d",
            ts_ms=1760000004000,
            session_id="s1",
            channel="read",
            provider="deepseek:read",
            units=1,
            unit="reads",
            amount_micro=2000,
            billing_basis=BASIS_ESTIMATED,
            cost_source="catalogue",
            price_version="client-search-table-2026-09",
        ),
    ]
    generated = payload(
        records,
        inference=InferenceSnapshot(
            micro=900000,
            calls=12,
            priced_calls=12,
            knowledge="exact",
            by_identity={
                "anthropic/claude-sonnet-5-5": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-5-5",
                    "micro": 900000,
                    "calls": 12,
                    "unpriced": 0,
                }
            },
        ),
    )
    fixture = json.loads(FIXTURE.read_text())
    assert generated == fixture, "the fixture drifted from the code that produces it"

    assert set(fixture) == {
        "version",
        "tracked",
        "total_micro",
        "knowledge",
        "by_basis",
        "rows",
        "children",
    }
    assert set(fixture["rows"][0]) == {
        "channel",
        "provider",
        "model",
        "label",
        "units",
        "unit",
        "amount_micro",
        "knowledge",
        "basis",
        "price_versions",
    }
    assert set(fixture["children"]) == {"total_micro", "knowledge"}
    assert set(fixture["by_basis"]) == {
        "billed",
        "subscription_api_equivalent",
        "estimated",
        # Money whose BASIS is not tracked yet (this session's inference plus
        # the children bundle): the bucket that makes the parts sum to the
        # whole. Additive on v1 — an old producer omits it (UI round-2 ask).
        "not_tracked_micro",
        "not_tracked_calls",
    }

    # And the wire MODEL accepts it unchanged: the fixture and the published
    # field are the same contract, not two lookalikes.
    parsed = FrontendSpendChannels.model_validate(fixture)
    assert parsed.model_dump(mode="json") == fixture


def test_custom_type_literal_is_pinned() -> None:
    """The transcript type is a compatibility surface; a rename is a migration."""
    assert CHANNEL_SPEND_CUSTOM_TYPE == "session_channel_spend.v1"


# -- SessionSpend.by_identity (the additive inference split) -----------------


def test_by_identity_splits_accruals_and_survives_a_round_trip() -> None:
    from local_operator.session.spend import SessionSpend

    spend = SessionSpend()
    spend.accrue(1_000_000, {"provider": "anthropic", "model_id": "claude"})
    spend.accrue(500_000, {"provider": "anthropic", "model_id": "claude"})
    spend.accrue(None, {"provider": "deepseek", "model_id": "r1"})
    assert spend.by_identity["anthropic/claude"] == {
        "provider": "anthropic",
        "model_id": "claude",
        "micro": 1_500_000,
        "calls": 2,
        "unpriced": 0,
    }
    assert spend.by_identity["deepseek/r1"]["unpriced"] == 1
    assert spend.micro == 1_500_000 and spend.calls == 3 and spend.unpriced_calls == 1

    recalled = SessionSpend.from_details(spend.to_details())
    assert recalled is not None and recalled.by_identity == spend.by_identity


def test_by_identity_caps_named_entries_and_folds_the_rest_into_other() -> None:
    from local_operator.session.spend import IDENTITY_CAP, SessionSpend

    spend = SessionSpend()
    for index in range(IDENTITY_CAP + 3):
        spend.accrue(1000, {"provider": "p", "model_id": f"m{index}"})
    named = [key for key in spend.by_identity if key != "other"]
    assert len(named) == IDENTITY_CAP
    assert spend.by_identity["other"]["calls"] == 3
    assert sum(entry["micro"] for entry in spend.by_identity.values()) == spend.micro


def test_by_identity_moves_with_a_correction() -> None:
    from local_operator.session.spend import SessionSpend

    spend = SessionSpend()
    index = spend.accrue(1000, {"provider": "p", "model_id": "m"})
    assert spend.correct(index, 5000) == 4000
    assert spend.by_identity["p/m"]["micro"] == 5000
    index = spend.accrue(None, {"provider": "p", "model_id": "m"})
    assert spend.correct(index, 2000) == 2000
    assert spend.by_identity["p/m"] == {
        "provider": "p",
        "model_id": "m",
        "micro": 7000,
        "calls": 2,
        "unpriced": 0,
    }


def test_from_details_accepts_a_row_without_by_identity() -> None:
    """An old ``session_spend.v1`` row reads as no breakdown, not as an error."""
    from local_operator.session.spend import SessionSpend

    old = {
        "version": 1,
        "micro": 2000000,
        "calls": 4,
        "priced_calls": 4,
        "unpriced_calls": 0,
        "floor": False,
        "rebuilt": False,
        "writer": "1:1",
    }
    recalled = SessionSpend.from_details(old)
    assert recalled is not None
    assert recalled.micro == 2_000_000
    assert recalled.by_identity == {}

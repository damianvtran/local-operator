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

from local_operator.session import channel_spend as channel_spend_module
from local_operator.session.channel_spend import (
    BASIS_BILLED,
    BASIS_ESTIMATED,
    BASIS_META,
    BASIS_NOT_TRACKED,
    BASIS_SUBSCRIPTION,
    BASIS_VALUES,
    CHANNEL_LABELS,
    CHANNEL_ORDER,
    CHANNEL_SPEND_CUSTOM_TYPE,
    CHANNELS,
    NOT_TRACKED_MICRO,
    PENDING_OWNER_APPEND,
    PENDING_OWNER_APPEND_REASON,
    PENDING_SIDECAR_NAME,
    SPEND_BREAKDOWN_ENTRY_CAP,
    ChannelSpend,
    ChannelSpendRecord,
    ChildrenSnapshot,
    InferenceSnapshot,
    combine,
    fold_records,
    map_image_cost_labels,
    new_record_id,
    normalise_basis,
    read_pending_records,
    records_from_details,
    write_pending_record,
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
    child_records: list[ChannelSpendRecord] | None = None,
    tracked: bool = True,
    lost: bool = False,
    pending: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return combine(
        fold_records(records).rows(),
        inference=inference or InferenceSnapshot(),
        children=children or ChildrenSnapshot(),
        child_records=child_records or (),
        tracked=tracked,
        lost=lost,
        pending=pending or (),
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
    assert result["children"] == {
        "total_micro": 250,
        "knowledge": "partial",
        "by_basis": {
            "billed": 0,
            "subscription_api_equivalent": 0,
            "estimated": 0,
            "not_tracked_micro": 250,
        },
    }
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
#
# ``golden_payload`` is the fixture's ONE regeneration source: the golden
# test compares the file against it, so a deliberate wire change updates both
# in one place and an accidental one fails here.


def golden_records() -> list[ChannelSpendRecord]:
    """The fixture's five records — the one source the fixture regenerates from."""
    return [
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


def golden_inference() -> InferenceSnapshot:
    """The fixture's inference half: twelve priced calls on one identity."""
    return InferenceSnapshot(
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
    )


def golden_child_records() -> list[ChannelSpendRecord]:
    """The fixture's relayed child record — the design note's §4.3 subagents half.

    A child session's image call relayed into the parent: it rides
    ``child_records`` (never ``rows``), lands in BOTH partitions — the
    ``subagents`` entry and the ``image`` source entry — and keeps the golden
    object exercising the dedup + children fold, not assuming them.
    """
    return [
        ChannelSpendRecord(
            record_id="image:child1",
            ts_ms=1760000005000,
            session_id="child",
            channel="image",
            provider="radient",
            model="gpt-image-2",
            units=1,
            unit="images",
            amount_micro=20000,
            billing_basis=BASIS_BILLED,
            cost_source="server_reported",
            status="ok",
        )
    ]


def golden_payload() -> dict[str, Any]:
    """The golden fixture's exact payload, produced by ``combine``.

    The §4.3 scenario: the five records, a 500,000 µ$ children ledger and ONE
    relayed child image record (20,000 µ$), so both partitions reconcile to
    ``total_micro = 1,536,000`` and the fixture carries every additive shape
    the UI PR types against.

    Regenerate ``tests/fixtures/spend_channels_v1.json`` after a DELIBERATE
    wire change by dumping this object the way the fixture is stored — two
    space indent plus a trailing newline::

        .venv/bin/python - <<'PY'
        import json

        from tests.unit.session.test_channel_spend import golden_payload

        path = "tests/fixtures/spend_channels_v1.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(golden_payload(), handle, indent=2)
            handle.write("\n")
        PY
    """
    return payload(
        golden_records(),
        inference=golden_inference(),
        children=ChildrenSnapshot(total_micro=500000, knowledge="exact"),
        child_records=golden_child_records(),
    )


def test_wire_fixture_is_the_golden_payload_and_the_contract() -> None:
    """The fixture is what ``combine`` produces, and its KEYS are the contract.

    The scenario is the design note's §4.3: the five records, a 500,000 µ$
    children ledger and ONE relayed child image record (20,000 µ$), so both
    partitions reconcile to the same 1,536,000 µ$ total. The UI PR types
    against this file. If a key is renamed or dropped, this test (and the UI's
    own contract test) fails on purpose — the alternative is a silent wire
    break discovered on a user's screen.
    """
    generated = golden_payload()
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
        "breakdown",
        "vocabulary",
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
    # ADDITIVE unpriced reason: present ONLY on a row that carries the condition,
    # so a priced row above keeps the exact key set every older reader knows.
    assert [row["unpriced_reason"] for row in fixture["rows"] if "unpriced_reason" in row] == [
        "cancelled_outcome_unknown"
    ]
    assert set(fixture["children"]) == {"total_micro", "knowledge", "by_basis"}
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
    # The additive sections' keys are the contract too: an entry and a basis
    # meta each carry exactly the fields the UI types against, so a rename
    # fails loudly here (design §4.2/§5).
    assert set(fixture["breakdown"]) == {"by_agent", "by_source"}
    entry_keys = {"id", "label", "amount_micro", "knowledge", "unstated_records", "bases"}
    assert set(fixture["breakdown"]["by_agent"][0]) == entry_keys
    assert set(fixture["breakdown"]["by_source"][0]) == entry_keys
    assert set(fixture["vocabulary"]) == {"channels", "bases"}
    assert set(fixture["vocabulary"]["channels"]["image"]) == {"label"}
    assert set(fixture["vocabulary"]["bases"]["billed"]) == {"label", "source_class", "marker"}
    assert set(fixture["vocabulary"]["bases"]["subscription_api_equivalent"]) == {
        "label",
        "source_class",
        "marker",
        "note",
    }, "the subscription note is the only explanatory line on the wire"

    # The golden SHAPE — the design note's §4.3 scenario verbatim, so a shape
    # change fails HERE first. Both partitions reconcile to the one 1,536,000 µ$
    # total; the relayed child record rides BOTH the ``subagents`` entry and
    # its own ``image`` source entry; the children ledger's not-tracked basis
    # sits in the ``inference`` entry's bases.
    assert fixture["total_micro"] == 1536000
    assert fixture["children"] == {
        "total_micro": 520000,
        "knowledge": "exact",
        "by_basis": {
            "billed": 20000,
            "subscription_api_equivalent": 0,
            "estimated": 0,
            "not_tracked_micro": 500000,
        },
    }
    assert fixture["by_basis"]["not_tracked_micro"] == 1400000
    assert fixture["breakdown"]["by_agent"] == [
        {
            "id": "parent",
            "label": "Parent agent",
            "amount_micro": 1016000,
            "knowledge": "partial",
            "unstated_records": 1,
            "bases": ["billed", "estimated", "not_tracked", "subscription_api_equivalent"],
        },
        {
            "id": "subagents",
            "label": "Subagents",
            "amount_micro": 520000,
            "knowledge": "exact",
            "unstated_records": 0,
            "bases": ["billed", "not_tracked"],
        },
    ]
    assert fixture["breakdown"]["by_source"] == [
        {
            "id": "inference",
            "label": "Inference",
            "amount_micro": 1400000,
            "knowledge": "exact",
            "unstated_records": 0,
            "bases": ["not_tracked"],
        },
        {
            "id": "image",
            "label": "Image",
            "amount_micro": 126000,
            "knowledge": "partial",
            "unstated_records": 1,
            "bases": ["billed", "subscription_api_equivalent"],
        },
        {
            "id": "search",
            "label": "Search",
            "amount_micro": 8000,
            "knowledge": "exact",
            "unstated_records": 0,
            "bases": ["estimated"],
        },
        {
            "id": "read",
            "label": "Read",
            "amount_micro": 2000,
            "knowledge": "exact",
            "unstated_records": 0,
            "bases": ["estimated"],
        },
    ]
    assert fixture["vocabulary"] == {
        "channels": {
            "inference": {"label": "Inference"},
            "image": {"label": "Image"},
            "search": {"label": "Search"},
            "read": {"label": "Read"},
        },
        "bases": {
            "billed": {"label": "billed", "source_class": "paygo", "marker": "billed"},
            "subscription_api_equivalent": {
                "label": "included",
                "source_class": "subscription",
                "marker": "included · API-equivalent",
                "note": (
                    "Covered by your plan — the amount is the API list-price equivalent, "
                    "not money charged."
                ),
            },
            "estimated": {"label": "estimated", "source_class": "paygo", "marker": "estimated"},
            "not_tracked": {
                "label": "not tracked",
                "source_class": "unknown",
                "marker": "not tracked",
            },
        },
    }
    for partition in ("by_agent", "by_source"):
        assert (
            sum(entry["amount_micro"] or 0 for entry in fixture["breakdown"][partition])
            == fixture["total_micro"]
        )

    # And the wire MODEL accepts it unchanged: the fixture and the published
    # field are the same contract, not two lookalikes.
    parsed = FrontendSpendChannels.model_validate(fixture)
    assert parsed.model_dump(mode="json") == fixture


def test_both_partitions_reconcile_to_the_total() -> None:
    """Every published shape: Σ by_agent == Σ by_source == total_micro.

    The two partitions are views of ONE total (design §4.5), so each must
    reconcile on its own — a ``null``-amount entry (money we cannot size)
    contributes nothing rather than a zero, and no piece appears twice.
    """
    shapes = [
        golden_payload(),
        payload([]),
        payload([record(record_id="search:only")]),
        payload([], inference=InferenceSnapshot(micro=900000, calls=12, priced_calls=12)),
        payload([], children=ChildrenSnapshot(total_micro=300000, knowledge="floor")),
        payload(
            [
                record(
                    record_id="image:1",
                    channel="image",
                    amount_micro=53000,
                    basis=BASIS_BILLED,
                ),
                record(
                    record_id="image:2",
                    channel="image",
                    amount_micro=None,
                    status="cancelled",
                ),
                record(record_id="search:2"),
            ],
            inference=InferenceSnapshot(
                micro=900000, calls=12, priced_calls=11, knowledge="partial"
            ),
            children=ChildrenSnapshot(total_micro=300000, knowledge="floor"),
            lost=True,
        ),
    ]
    for shape in shapes:
        for partition in ("by_agent", "by_source"):
            stated = sum(
                entry["amount_micro"]
                for entry in shape["breakdown"][partition]
                if entry["amount_micro"] is not None
            )
            assert stated == shape["total_micro"], (
                f"{partition} does not reconcile: {stated} != {shape['total_micro']} "
                f"for {shape['breakdown'][partition]!r}"
            )


def test_a_new_channel_and_basis_flow_through_the_breakdown_without_schema_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Design §5.3: an unregistered channel/basis renders ECHOED, then curated.

    ``video`` + ``video_credits`` is exactly what the coming video-spend PR
    will emit before anyone touches this module: the by-source entry exists
    with the fallback label, and the vocabulary names the id and ECHOES the
    basis (``source_class: "unknown"``, no note — never a subscription claim).
    Registering the pair in the tables flips both to the curated words, in
    canonical order, with every row and amount untouched.
    """
    records = [
        record(record_id="search:1", amount_micro=8000),
        record(
            record_id="video:1",
            channel="video",
            basis="video_credits",
            amount_micro=21000,
            provider="veo",
            units=1,
            unit="videos",
        ),
    ]
    generated = payload(records)
    assert [entry["id"] for entry in generated["breakdown"]["by_source"]] == [
        "search",
        "video",
    ], "an unknown channel sorts after every registered one"
    video = generated["breakdown"]["by_source"][1]
    assert video["label"] == "Video", "the fallback label is id.capitalize()"
    assert video["amount_micro"] == 21000
    assert video["bases"] == ["video_credits"]
    assert generated["vocabulary"]["channels"]["video"] == {"label": "Video"}
    assert generated["vocabulary"]["bases"]["video_credits"] == {
        "label": "video_credits",
        "source_class": "unknown",
        "marker": "video_credits",
    }, "a foreign spelling is echoed — never a subscription claim"

    # Register the pair the way a later analytics PR would: label, canonical
    # order, basis meta. Nothing else in the tables moves.
    monkeypatch.setitem(channel_spend_module.CHANNEL_LABELS, "video", "Video generation")
    order: list[str] = list(channel_spend_module.CHANNEL_ORDER)
    order.insert(order.index("image") + 1, "video")
    monkeypatch.setattr(channel_spend_module, "CHANNEL_ORDER", tuple(order))
    monkeypatch.setitem(
        channel_spend_module.BASIS_META,
        "video_credits",
        {"label": "video credits", "source_class": "paygo", "marker": "video credits"},
    )

    curated = payload(records)
    assert [entry["id"] for entry in curated["breakdown"]["by_source"]] == [
        "video",
        "search",
    ], "registration moves the id into its canonical order"
    assert curated["breakdown"]["by_source"][0]["label"] == "Video generation"
    assert curated["vocabulary"]["channels"]["video"] == {"label": "Video generation"}
    assert curated["vocabulary"]["bases"]["video_credits"] == {
        "label": "video credits",
        "source_class": "paygo",
        "marker": "video credits",
    }
    # Nothing else moved: the same money and the same rows (compared as a set:
    # a row's own order follows the display order registration just changed).
    assert curated["total_micro"] == generated["total_micro"]
    assert {row["channel"]: row for row in curated["rows"]} == {
        row["channel"]: row for row in generated["rows"]
    }, "registration must not change any row's own numbers"


def test_the_vocabulary_registries_cover_every_registered_value() -> None:
    """Registration cannot drift: the tables cover the id sets they name.

    ``CHANNEL_LABELS`` must name every ``CHANNELS`` id, ``CHANNEL_ORDER`` must
    place it (published order IS render order), and ``BASIS_META`` must
    describe every ``BASIS_VALUES`` spelling with the three fields every basis
    carries. Adding a channel or basis without its display words fails HERE
    rather than shipping an object that cannot name it.
    """
    assert set(CHANNELS) <= set(CHANNEL_LABELS)
    assert set(CHANNELS) <= set(CHANNEL_ORDER)
    assert set(BASIS_VALUES) <= set(BASIS_META)
    for spelling, meta in BASIS_META.items():
        assert set(meta) <= {"label", "source_class", "marker", "note"}, spelling
        assert {"label", "source_class", "marker"} <= set(meta), spelling
        assert meta["source_class"] in {"paygo", "subscription", "unknown"}, spelling
        assert meta["marker"], spelling


def test_the_breakdown_entry_cap_groups_the_remainder() -> None:
    """At most ``SPEND_BREAKDOWN_ENTRY_CAP`` entries; the rest fold into ``other``.

    The same overflow discipline as the wire's row cap (design §4.6; review
    round 1, MINOR): the first ``cap - 1`` ranked entries keep their places
    and everything past that edge aggregates into ONE ``other`` entry — money
    summed, worst rung applied, counts preserved — so the partition never
    exceeds the cap, loses no money and still reconciles.
    """
    records = [
        record(
            record_id=f"chan{index:02d}:1",
            channel=f"chan{index:02d}",
            amount_micro=None if index == 13 else (index + 1) * 1000,
            status="cancelled" if index == 13 else "ok",
        )
        for index in range(14)
    ]
    generated = payload(records)
    entries = generated["breakdown"]["by_source"]
    assert len(entries) == SPEND_BREAKDOWN_ENTRY_CAP
    assert [entry["id"] for entry in entries[:-1]] == [
        f"chan{index:02d}" for index in range(SPEND_BREAKDOWN_ENTRY_CAP - 1)
    ]
    folded = entries[-1]
    assert folded["id"] == "other"
    assert folded["label"] == "Other"
    assert (
        folded["amount_micro"] == 12000 + 13000
    ), "chan13 was never sized; only its count survives"
    assert folded["unstated_records"] == 1
    assert folded["knowledge"] == "partial", "stated money is never rated unknown"
    assert sum(entry["amount_micro"] or 0 for entry in entries) == generated["total_micro"]

    # A session that ALREADY has an ``other`` entry folds the remainder into
    # it rather than emitting a second one — the count then sits below the
    # cap and the partition stays a partition.
    with_other = payload(
        [
            *records,
            record(record_id="other:1", channel="other", amount_micro=777),
        ]
    )
    other_entries = with_other["breakdown"]["by_source"]
    assert len(other_entries) == SPEND_BREAKDOWN_ENTRY_CAP - 1
    assert other_entries[0]["id"] == "other"
    assert other_entries[0]["amount_micro"] == 777 + 11000 + 12000 + 13000
    assert sum(entry["amount_micro"] or 0 for entry in other_entries) == with_other["total_micro"]


def test_entry_knowledge_never_reads_exacter_than_its_parts() -> None:
    """The rung matrix across both partitions: unknown > partial > floor > exact.

    Entry rungs come from combine's OWN booleans (design §4.5): an unsized
    record reads ``partial`` (or ``unknown`` when nothing is sized), the
    inference half contributes its rung, the children-ledger rung lands on the
    ``inference`` entry and the published children knowledge on ``subagents``.
    """
    generated = payload(
        [
            record(
                record_id="cls:1",
                channel="classification",
                amount_micro=None,
                status="ok",
            ),
            record(record_id="image:1", channel="image", amount_micro=53000, basis=BASIS_BILLED),
            record(
                record_id="image:2",
                channel="image",
                amount_micro=None,
                status="cancelled",
            ),
            record(record_id="search:1", amount_micro=8000),
        ],
        inference=InferenceSnapshot(micro=900000, calls=12, priced_calls=12, knowledge="floor"),
        children=ChildrenSnapshot(total_micro=300000, knowledge="floor"),
        child_records=[
            record(record_id="tts:kid", channel="tts", amount_micro=40000, basis=BASIS_BILLED),
        ],
    )
    entries = {entry["id"]: entry for entry in generated["breakdown"]["by_source"]}
    assert entries["classification"]["amount_micro"] is None
    assert entries["classification"]["knowledge"] == "unknown"
    assert entries["classification"]["unstated_records"] == 1
    assert entries["image"]["knowledge"] == "partial", "a stated + unsized group"
    assert entries["search"]["knowledge"] == "exact"
    assert entries["inference"]["knowledge"] == "floor", "the ledger rung rides inference"
    assert entries["tts"]["knowledge"] == "exact", "a relayed, stated child record"
    agents = {entry["id"]: entry for entry in generated["breakdown"]["by_agent"]}
    assert agents["parent"]["knowledge"] == "partial", "own records degrade the parent"
    assert agents["subagents"]["knowledge"] == "floor", "the published children knowledge"
    assert (
        sum(entry["amount_micro"] or 0 for entry in generated["breakdown"]["by_agent"])
        == generated["total_micro"]
    )

    # With no unsized own record, the parent still cannot read exacter than
    # the inference half it carries.
    floor_only = payload(
        [record(record_id="search:1")],
        inference=InferenceSnapshot(micro=900000, calls=12, priced_calls=12, knowledge="floor"),
    )
    assert floor_only["breakdown"]["by_agent"][0]["knowledge"] == "floor"


def test_the_inference_half_rung_follows_combines_activity_gate() -> None:
    """MAJOR (review round 1): an entry mirrors the session over the inference half.

    A never-accrued accumulator can carry money with NO calls (a remainder) or
    unpriced calls only. ``combine`` counts the half's ``unknown`` only when
    calls exist, and maps that case to ``partial`` — so the ``parent`` and
    ``inference`` entries must derive their rungs from the same two facts, or
    one reads vaguer than the session over the identical money (or exacter,
    the other way).
    """
    remainder = payload(
        [record(record_id="search:1")],
        inference=InferenceSnapshot(micro=400000, calls=0),
    )
    assert remainder["knowledge"] == "exact"
    agents = {entry["id"]: entry for entry in remainder["breakdown"]["by_agent"]}
    sources = {entry["id"]: entry for entry in remainder["breakdown"]["by_source"]}
    assert agents["parent"]["knowledge"] == "exact", "money without calls is not unpriced activity"
    assert sources["inference"]["knowledge"] == "exact"

    unpriced = payload(
        [record(record_id="search:1")],
        inference=InferenceSnapshot(micro=0, calls=2, priced_calls=0, knowledge="unknown"),
    )
    assert unpriced["knowledge"] == "partial"
    agents = {entry["id"]: entry for entry in unpriced["breakdown"]["by_agent"]}
    sources = {entry["id"]: entry for entry in unpriced["breakdown"]["by_source"]}
    assert (
        agents["parent"]["knowledge"] == "partial"
    ), "unpriced calls degrade to partial, never vaguer"
    assert sources["inference"]["knowledge"] == "partial"


def test_all_unsized_entries_read_null_never_zero() -> None:
    """MAJOR (review round 2): an entry whose contributors are all unstated is null.

    Nothing is stated anywhere, so every entry reads ``null`` + its unsized
    count — never a fabricated 0 — and the siblings AGREE with each other on
    the same payload (parent, the channel entry, and the subagents entry).
    """
    generated = payload(
        [
            record(
                record_id="image:1",
                channel="image",
                amount_micro=None,
                status="cancelled",
            )
        ],
        child_records=[
            record(
                record_id="image:kid",
                channel="image",
                amount_micro=None,
                status="cancelled",
            )
        ],
    )
    agents = {entry["id"]: entry for entry in generated["breakdown"]["by_agent"]}
    sources = {entry["id"]: entry for entry in generated["breakdown"]["by_source"]}
    assert agents["parent"]["amount_micro"] is None
    assert agents["parent"]["unstated_records"] == 1
    assert sources["image"]["amount_micro"] is None
    assert sources["image"]["unstated_records"] == 2, "the own and the relayed record"
    assert agents["subagents"]["amount_micro"] is None
    assert agents["subagents"]["unstated_records"] == 1


def test_a_degraded_children_bundle_degrades_the_inference_entry() -> None:
    """MAJOR (review round 2): an entry never reads ``unknown`` over stated money.

    A children ledger of 400,000 µ$ with a relayed child record that has NO
    stated amount: the ledger money is stated (the inference entry carries
    400,000) and the bundle itself degrades to ``partial`` for the unsized
    relay — the entry takes that worst rung rather than out-reading the
    bundle.
    """
    generated = payload(
        [],
        children=ChildrenSnapshot(total_micro=400000, knowledge="exact"),
        child_records=[
            record(
                record_id="image:kid",
                channel="image",
                amount_micro=None,
                status="cancelled",
            )
        ],
    )
    assert generated["children"]["knowledge"] == "partial", "the bundle's own degradation"
    entries = {entry["id"]: entry for entry in generated["breakdown"]["by_source"]}
    assert entries["inference"]["amount_micro"] == 400000
    assert entries["inference"]["knowledge"] == "partial"
    assert entries["inference"]["unstated_records"] == 0


def test_an_inference_named_relay_cannot_sink_the_entry_below_partial() -> None:
    """MAJOR (review round 3): stated money is never rated ``unknown``.

    The shape round 2 cited: a 400,000 µ$ exact ledger plus ONE relayed child
    record whose channel is ``inference`` and which has no stated amount. The
    record set alone rates ``unknown``, but the entry CARRIES stated money,
    and combine rates the same bundle ``partial`` at the session and
    ``subagents`` level — the entry must read ``partial`` too, never
    ``unknown``.
    """
    generated = payload(
        [],
        children=ChildrenSnapshot(total_micro=400000, knowledge="exact"),
        child_records=[
            record(
                record_id="inference:kid",
                channel="inference",
                amount_micro=None,
                status="cancelled",
            )
        ],
    )
    assert generated["knowledge"] == "partial", "the session over the same inputs"
    assert generated["children"]["knowledge"] == "partial"
    entries = {entry["id"]: entry for entry in generated["breakdown"]["by_source"]}
    assert entries["inference"]["amount_micro"] == 400000
    assert entries["inference"]["knowledge"] == "partial"
    assert entries["inference"]["unstated_records"] == 1
    agents = {entry["id"]: entry for entry in generated["breakdown"]["by_agent"]}
    assert agents["subagents"]["knowledge"] == "partial", "the sibling over the same money"


def test_inference_bases_come_from_pr3s_route_columns() -> None:
    """PR-3 integration: the inference half states its REAL per-route bases.

    The bases derive from the by-identity ``bases: {basis: micro}`` columns —
    the same data the rows and ``by_basis`` read — so a plan-funded call
    reads ``subscription_api_equivalent`` on both entries instead of the
    legacy blanket ``not_tracked``, and ``not_tracked`` appears only for
    money those columns do not attribute.
    """
    subscribed = payload(
        [],
        inference=InferenceSnapshot(
            micro=900000,
            calls=12,
            priced_calls=12,
            knowledge="exact",
            by_identity={
                "openai/chat": {
                    "provider": "openai",
                    "model_id": "gpt-5",
                    "micro": 900000,
                    "calls": 12,
                    "unpriced": 0,
                    "bases": {"subscription_api_equivalent": 900000},
                }
            },
        ),
    )
    agents = {entry["id"]: entry for entry in subscribed["breakdown"]["by_agent"]}
    sources = {entry["id"]: entry for entry in subscribed["breakdown"]["by_source"]}
    assert agents["parent"]["bases"] == ["subscription_api_equivalent"]
    assert sources["inference"]["bases"] == ["subscription_api_equivalent"]
    assert (
        "subscription_api_equivalent" in subscribed["vocabulary"]["bases"]
    ), "the curated display word rides the object — no UI edit"

    # Half-attributed: the unattributed remainder keeps the honest fallback,
    # BESIDE the real spelling — never instead of it.
    remainder = payload(
        [],
        inference=InferenceSnapshot(
            micro=900000,
            calls=13,
            priced_calls=12,
            knowledge="partial",
            by_identity={
                "openai/chat": {
                    "provider": "openai",
                    "model_id": "gpt-5",
                    "micro": 800000,
                    "calls": 12,
                    "unpriced": 0,
                    "bases": {"subscription_api_equivalent": 800000},
                }
            },
        ),
    )
    entries = {entry["id"]: entry for entry in remainder["breakdown"]["by_source"]}
    assert entries["inference"]["bases"] == [
        "not_tracked",
        "subscription_api_equivalent",
    ]


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


# -- PR-3: the §4.2 fallback's marker and the pending wire key -----------------


def test_pending_voice_records_add_the_key_only_when_present() -> None:
    """``pending`` is ADDITIVE and absent for every ordinary session.

    The golden fixture asserts the exact key set of an ordinary payload; this
    pins the other half: a session WITH unfoled voice records says so, counts
    them, and does not move ``total_micro`` (the same total on all surfaces).
    """
    assert "pending" not in payload([])
    with_pending = payload(
        [],
        pending=[
            {"record_id": "tts:a", "channel": "tts", "ts_ms": 1},
            {"record_id": "stt:b", "channel": "stt", "ts_ms": 2},
            {"record_id": "search:c", "channel": "search", "ts_ms": 3},
        ],
    )
    assert with_pending["pending"] == {
        "voice": PENDING_OWNER_APPEND,
        "calls": 2,
        "reason": PENDING_OWNER_APPEND_REASON,
    }
    assert with_pending["total_micro"] == 0, "pending money is NOT in the total"


def test_pending_sidecar_roundtrip_is_tolerant_and_never_raises(tmp_path: Path) -> None:
    """One appended line per voice record; junk lines are skipped, not fatal."""
    record = ChannelSpendRecord(
        record_id="tts:abc",
        channel="tts",
        provider="elevenlabs",
        units=25,
        unit="chars",
        amount_micro=1000,
        billing_basis="billed",
        cost_source="server_reported",
        ts_ms=1760000000000,
    )
    assert write_pending_record(tmp_path, record) is True
    sidecar = tmp_path / PENDING_SIDECAR_NAME
    with open(sidecar, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    entries = read_pending_records(tmp_path)
    assert len(entries) == 1
    assert entries[0] == {
        "record_id": "tts:abc",
        "channel": "tts",
        "ts_ms": 1760000000000,
        "reason": PENDING_OWNER_APPEND,
    }

    # A non-voice channel is never written, and a missing directory is skipped
    # rather than created (a foreign/never-materialised session id).
    image = ChannelSpendRecord(record_id="image:x", channel="image", amount_micro=1)
    assert write_pending_record(tmp_path, image) is False
    assert read_pending_records(tmp_path / "missing") == []
    assert write_pending_record(tmp_path / "never-made", record) is False


def test_pending_calls_counts_unique_charges_not_marker_lines() -> None:
    """Q6 (QA round 1): a retried emission writes a second identical marker.

    Analytics holds ONE row, one fold retires both markers, and the money is
    counted once — so the wire's count must read one request, not the number
    of lines a crash-and-retry happens to have left in the sidecar.
    """
    with_pending = payload(
        [],
        pending=[
            {"record_id": "tts:one", "channel": "tts", "ts_ms": 1},
            {"record_id": "tts:one", "channel": "tts", "ts_ms": 2},
        ],
    )
    assert with_pending["pending"]["calls"] == 1
    # A marker without a record_id cannot be retired by the fold, so it cannot
    # join the count either (it would never go away).
    orphan = payload([], pending=[{"channel": "tts", "ts_ms": 3}])
    assert "pending" not in orphan


# ---------------------------------------------------------------------------
# The ``audio`` channel (the sound-generation lane's ledger home)
# ---------------------------------------------------------------------------


def test_the_audio_channel_is_registered_with_its_label_and_place() -> None:
    """Registration in all three tables is the whole contract.

    A channel id present in ``CHANNELS`` but missing from ``CHANNEL_LABELS``
    or ``CHANNEL_ORDER`` fails the vocabulary-coverage test above; this pin
    goes further and names the DISPLAY words and the canonical position, so a
    silent relabel or a reorder is a red test rather than a panel that moved.
    """
    from local_operator.session.channel_spend import CHANNEL_LABELS, CHANNEL_ORDER

    assert "audio" in CHANNELS
    assert CHANNEL_LABELS["audio"] == "Generated audio"
    assert CHANNEL_ORDER.index("audio") == CHANNEL_ORDER.index("image") + 1
    assert CHANNEL_ORDER.index("audio") < CHANNEL_ORDER.index(
        "other"
    ), "the audio channel was inserted before ``other``, not appended after it"


def test_map_audio_cost_labels_matches_the_frozen_audio_contract() -> None:
    from local_operator.session.channel_spend import (
        COST_SOURCE_CATALOGUE,
        COST_SOURCE_NONE,
        COST_SOURCE_SERVER,
        map_audio_cost_labels,
    )

    # Radient: the server's figure, quote reads estimated, billed reads billed.
    assert map_audio_cost_labels(
        route="radient",
        cost_source="reported",
        billing_basis="estimated",
        price_version="Radient ... (basis: quote)",
        has_amount=True,
    ) == (BASIS_ESTIMATED, COST_SOURCE_SERVER, "Radient ... (basis: quote)")
    assert (
        map_audio_cost_labels(
            route="radient",
            cost_source="reported",
            billing_basis="billed",
            price_version="Radient ... (basis: billed)",
            has_amount=True,
        )[0]
        == BASIS_BILLED
    )
    # BYOK: a rate-table estimate, catalogue-sourced, never a charge.
    assert map_audio_cost_labels(
        route="elevenlabs",
        cost_source="rate_table",
        billing_basis="estimated",
        price_version="audio-rates@2026-10-17",
        has_amount=True,
    ) == (BASIS_ESTIMATED, COST_SOURCE_CATALOGUE, "audio-rates@2026-10-17")
    # No amount at all: not tracked, never a fabricated zero.
    assert map_audio_cost_labels(
        route="openai",
        cost_source=None,
        billing_basis=None,
        price_version=None,
        has_amount=False,
    ) == (BASIS_NOT_TRACKED, COST_SOURCE_NONE, "")
    # An unlabelled amount degrades to estimated (never read as a charge).
    assert (
        map_audio_cost_labels(
            route="elevenlabs",
            cost_source="something-new",
            billing_basis=None,
            price_version=None,
            has_amount=True,
        )[0]
        == BASIS_ESTIMATED
    )


def test_an_audio_row_folds_into_the_published_payload() -> None:
    """The fold accepts the id and the record keeps its estimate flag."""
    from local_operator.session.channel_spend import (
        CHANNEL_LABELS,
        new_record_id,
        now_ms,
        usd_to_micro,
    )

    record_row = ChannelSpendRecord(
        record_id=new_record_id("audio", "req-1"),
        ts_ms=now_ms(),
        channel="audio",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        units=1000.0,
        unit="chars",
        amount_micro=usd_to_micro(0.08),
        billing_basis=BASIS_ESTIMATED,
        cost_source="catalogue",
        price_version="audio-rates@2026-10-17",
        status="ok",
        request_id="req-1",
    )
    spend = fold_records([record_row])
    rows = {rec.channel: rec for rec in spend.rows()}
    assert "audio" in rows
    assert rows["audio"].amount_micro == 80_000
    assert rows["audio"].billing_basis == BASIS_ESTIMATED, "the estimate flag survives the fold"
    assert CHANNEL_LABELS["audio"] == "Generated audio"


# -- one decision model, one row: canonical ids and explicit unpriced reasons --


def decision_record(
    record_id: str,
    *,
    model: str,
    amount_micro: int | None,
    status: str = "ok",
) -> ChannelSpendRecord:
    return record(
        record_id=record_id,
        channel="classification",
        provider="radient",
        model=model,
        amount_micro=amount_micro,
        basis=BASIS_BILLED,
        status=status,
        units=1,
        unit="calls",
    )


def test_historical_spellings_of_one_decision_model_are_one_priced_row() -> None:
    """The defect: a priced ``jev-1.13.0`` row and a failed ``jev-1.13`` attempt.

    Before the canonical key they drew two rows for one model, and the failed
    one said only "price not stated". They are one row now: the priced amount
    plus the failed attempt, with an explicit not-billed reason.
    """
    rows = [
        row
        for row in payload(
            [
                decision_record("classification:ok", model="jev-1.13.0", amount_micro=11),
                decision_record(
                    "classification:failed", model="jev-1.13", amount_micro=None, status="failed"
                ),
            ]
        )["rows"]
        if row["channel"] == "classification"
    ]
    assert len(rows) == 1
    assert rows[0]["model"] == "jev-1.13"
    assert rows[0]["amount_micro"] == 11
    assert rows[0]["units"] == 2.0
    assert rows[0]["unpriced_reason"] == "failed_not_billed"


def test_a_dated_vendor_echo_groups_under_the_canonical_id() -> None:
    rows = [
        row
        for row in payload(
            [
                decision_record(
                    "classification:dated", model="typesafe/jev-1.13-20261001", amount_micro=11
                ),
                decision_record(
                    "classification:cancelled",
                    model="typesafe/jev-1.13",
                    amount_micro=None,
                    status="cancelled",
                ),
            ]
        )["rows"]
        if row["channel"] == "classification"
    ]
    assert len(rows) == 1
    assert rows[0]["model"] == "jev-1.13"
    assert rows[0]["unpriced_reason"] == "cancelled_outcome_unknown"


def test_a_failed_only_group_states_why_no_amount_is_present() -> None:
    (row,) = payload(
        [
            decision_record(
                "classification:failed", model="jev-1.13.0", amount_micro=None, status="failed"
            )
        ]
    )["rows"]
    assert row["amount_micro"] is None, "unknown, never a fabricated zero"
    assert row["knowledge"] == "exact", "the failed-unbilled presumption is not missing money"
    assert row["unpriced_reason"] == "failed_not_billed"


def test_a_cancelled_group_says_the_outcome_is_unknown() -> None:
    (row,) = payload(
        [
            decision_record(
                "classification:cancelled",
                model="jev-1.13.0",
                amount_micro=None,
                status="cancelled",
            )
        ]
    )["rows"]
    assert row["unpriced_reason"] == "cancelled_outcome_unknown"


def test_an_ok_row_without_an_amount_says_the_server_stated_no_price() -> None:
    (row,) = payload([decision_record("classification:ok", model="jev-1.13", amount_micro=None)])[
        "rows"
    ]
    assert row["unpriced_reason"] == "not_stated"


def test_money_mixed_with_an_unstated_ok_attempt_reads_partial() -> None:
    (row,) = payload(
        [
            decision_record("classification:ok", model="jev-1.13", amount_micro=11),
            decision_record("classification:parked", model="jev-1.13", amount_micro=None),
        ]
    )["rows"]
    assert row["amount_micro"] == 11
    assert row["unpriced_reason"] == "partial"

"""The Radient ``cost`` object adapter (contract draft, session 086f6187e0ba).

Every test here pins a rule the adapter exists to keep: absence is ``None``
never zero, a quote is never a charge, and a decimal string cannot drift on the
way to integer micro-USD. Field names come from the draft and may move — when
they do, THIS file and ``radient_cost.py`` are what change.
"""

from __future__ import annotations

from local_operator.clients.radient_cost import (
    RadientCost,
    basis_from_wire,
    channel_from_tool_type,
    cost_from_headers,
    cost_from_payload,
    micro_from_amount,
    unit_from_wire,
)


def test_absent_object_is_none_never_zero() -> None:
    """An old server (no ``cost``) yields None — the call is unknown, not free."""
    assert cost_from_payload({"result": "ok"}) is None
    assert cost_from_payload(None) is None
    assert cost_from_payload({"cost": "billed"}) is None
    assert cost_from_headers(None) is None
    assert cost_from_headers([]) is None
    assert cost_from_headers([("X-Other-Header", "1")]) is None


def test_v1_body_object_maps_quote_to_estimated_and_keeps_the_rest() -> None:
    cost = cost_from_payload(
        {
            "cost": {
                "v": 1,
                "amount_usd": "0.053",
                "basis": "quote",
                "units": 1,
                "unit": "image",
                "unit_price_usd": 0.053,
                "provider": "radient",
                "model": "gpt-image-2",
                "tool_type": "image_generation",
                "usage_record_id": "a1b2c3d4e5f60718293a4b5c",
                "price_version": "media-registry@2026-10-08",
            }
        }
    )
    assert cost is not None
    assert cost.amount_micro == 53_000
    assert cost.basis == "estimated", "a quote must never be shown as a charge"
    assert cost.channel == "image"
    assert cost.unit == "images"
    assert cost.units == 1.0
    assert cost.provider == "radient" and cost.model == "gpt-image-2"
    assert cost.usage_record_id == "a1b2c3d4e5f60718293a4b5c"
    assert cost.price_version == "media-registry@2026-10-08"
    assert cost.version == 1


def test_billed_and_estimated_pass_through_and_failed_zero_is_a_known_zero() -> None:
    billed = cost_from_payload({"cost": {"v": 1, "amount_usd": 0.61, "basis": "billed"}})
    assert billed is not None and billed.basis == "billed" and billed.amount_micro == 610_000
    estimated = cost_from_payload({"cost": {"v": 1, "amount_usd": 0.61, "basis": "estimated"}})
    assert estimated is not None and estimated.basis == "estimated"
    failed = cost_from_payload({"cost": {"v": 1, "amount_usd": 0, "basis": "billed"}})
    assert failed is not None, "an explicit zero is a KNOWN zero, not an absence"
    assert failed.amount_micro == 0


def test_speech_headers_parse_with_case_insensitivity() -> None:
    cost = cost_from_headers(
        [
            ("Content-Type", "audio/mpeg"),
            ("X-Radient-Cost-Version", "1"),
            ("X-Radient-Cost-Amount-Usd", "0.0042"),
            ("X-Radient-Cost-Basis", "billed"),
            ("X-Radient-Cost-Unit", "char"),
            ("X-Radient-Cost-Units", "123"),
            ("X-Radient-Cost-Usage-Record-Id", "deadbeefdeadbeefdeadbeef"),
        ]
    )
    assert cost is not None
    assert cost.amount_micro == 4_200
    assert cost.basis == "billed"
    assert cost.unit == "chars" and cost.units == 123.0
    assert cost.usage_record_id == "deadbeefdeadbeefdeadbeef"


def test_amount_conversion_never_drifts() -> None:
    """Decimal strings and floats land on the SAME integer micro-USD."""
    assert micro_from_amount("0.1") == 100_000
    assert micro_from_amount(0.1) == 100_000
    assert micro_from_amount(0.1 + 0.2) == 300_000  # 0.30000000000000004
    assert micro_from_amount(0) == 0
    assert micro_from_amount("bad") is None
    assert micro_from_amount(None) is None
    assert micro_from_amount(True) is None, "bool is not an amount"


def test_basis_unit_and_channel_mappings_are_total() -> None:
    assert basis_from_wire("quote") == "estimated"
    assert basis_from_wire("billed") == "billed"
    assert basis_from_wire("subscription-api-equivalent") == "subscription_api_equivalent"
    assert basis_from_wire("") == "not_tracked"
    assert basis_from_wire(None) == "not_tracked"
    assert unit_from_wire("second") == "seconds"
    assert unit_from_wire("search") == "searches"
    assert unit_from_wire("mystery") == "mystery"
    assert channel_from_tool_type("speech") == "tts"
    assert channel_from_tool_type("transcription") == "stt"
    assert channel_from_tool_type("web_search") == "search"
    assert channel_from_tool_type("brand_new_tool") == "other"


def test_spend_kwargs_and_record_conversion() -> None:
    cost = RadientCost(
        amount_micro=None,
        basis="not_tracked",
        units=420.0,
        unit="chars",
        tool_type="speech",
        channel="tts",
        provider="radient",
        model="elevenlabs",
    )
    kwargs = cost.spend_kwargs()
    assert kwargs["channel"] == "tts"
    assert kwargs["amount_micro"] is None, "units known, amount unknown"
    assert kwargs["cost_source"] == "none"
    assert kwargs["detail"] == "speech"
    record = cost.to_record(record_id="tts:req-1", session_id="s1")
    assert record.record_id == "tts:req-1"
    assert record.billing_basis == "not_tracked"
    assert record.amount_micro is None

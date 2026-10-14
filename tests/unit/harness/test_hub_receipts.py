"""Hub display projection preserves facts without classifying the message body."""

from local_operator.harness.comms import TO_CHILD_INSTRUCTIONS
from local_operator.harness.hub_receipts import (
    hub_receipt_fields,
    hub_receipt_marker,
    hub_receipt_stamped,
)


def test_marker_requires_the_namespace_mapping_and_a_real_body() -> None:
    """Positive provenance only: a malformed or body-less marker is not a
    receipt, so callers keep their ordinary user handling."""
    from types import SimpleNamespace

    good = {
        "direction": "to_child",
        "expects_reply": False,
        "steer": True,
        "body": "Focus on retries",
    }
    assert hub_receipt_marker(SimpleNamespace(provider_payload={"hub_receipt": good})) == good
    for payload in (
        None,
        {},
        {"hub_receipt": []},
        {"hub_receipt": {}},
        {"hub_receipt": {"body": ""}},
        {"hub_receipt": {"body": "   "}},
        {"hub_receipt": {"body": 3}},
        {"hub_receipt": "to_child"},
    ):
        assert hub_receipt_marker(SimpleNamespace(provider_payload=payload)) is None
    assert hub_receipt_marker(SimpleNamespace()) is None
    # The child fold reads the ENCODED payload dict, not a Message object.
    assert hub_receipt_marker({"provider_payload": {"hub_receipt": good}}) == good
    assert hub_receipt_marker({"provider_payload": None}) is None
    assert hub_receipt_marker({}) is None


def test_a_present_marker_is_a_stamp_even_when_it_fails_validation() -> None:
    """``hub_receipt_stamped`` separates "no modern provenance" — where the
    legacy wrapper compatibility may still parse a row from its body — from
    "stamped but invalid", where the row must not be re-guessed (review round
    3, R3-1 acceptance)."""
    from types import SimpleNamespace

    assert hub_receipt_stamped(SimpleNamespace()) is False
    assert hub_receipt_stamped(SimpleNamespace(provider_payload=None)) is False
    assert hub_receipt_stamped(SimpleNamespace(provider_payload={})) is False
    for marker in (
        {"body": "Focus on retries"},
        {"body": "Focus on retries", "direction": "sideways"},
        {"body": "Focus on retries", "direction": None},
        {"body": "Focus on retries", "direction": ["to_child"]},
        "to_child",
    ):
        stamped = SimpleNamespace(provider_payload={"hub_receipt": marker})
        assert hub_receipt_marker(stamped) is None
        assert hub_receipt_stamped(stamped) is True
    good = SimpleNamespace(
        provider_payload={"hub_receipt": {"direction": "to_child", "body": "Focus"}}
    )
    assert hub_receipt_stamped(good) is True
    assert hub_receipt_marker(good) is not None


def test_marker_requires_a_known_direction_and_normalises_flags() -> None:
    """A body alone is not provenance (review round 3, R3-1; QA Q-child-1).

    Missing, null, numeric, list and unknown directions keep the row's
    ordinary user handling; present flags normalise to booleans or UNKNOWN —
    never truthy-cast, never guessed — and non-string advisory fields drop."""
    from types import SimpleNamespace

    base = {
        "direction": "to_child",
        "expects_reply": True,
        "steer": False,
        "body": "Focus on retries",
    }
    assert hub_receipt_marker(SimpleNamespace(provider_payload={"hub_receipt": base})) == base

    for bad_direction in (None, 3, "unrecognized", [], ["to_child"]):
        marker = {"body": "raw text", "direction": bad_direction}
        assert (
            hub_receipt_marker(SimpleNamespace(provider_payload={"hub_receipt": marker})) is None
        ), bad_direction
    assert (
        hub_receipt_marker(
            SimpleNamespace(provider_payload={"hub_receipt": {"body": "no direction"}})
        )
        is None
    )

    unknown = hub_receipt_marker(
        SimpleNamespace(
            provider_payload={
                "hub_receipt": {
                    "direction": "to_parent",
                    "body": "reply",
                    "expects_reply": "yes",
                    "steer": 1,
                    "label": 7,
                }
            }
        )
    )
    assert unknown is not None
    assert unknown["expects_reply"] is None
    assert unknown["steer"] is None
    assert "label" not in unknown


def test_structured_body_and_optional_facts_are_retained_verbatim() -> None:
    body = "  first paragraph\n\nsecond paragraph  "
    details = {
        "body": body,
        "text": "model envelope",
        "direction": "to_parent",
        "label": "worker",
        "job_id": "job-1",
        "sender": {"session_id": "synthetic"},
    }
    projected, facts = hub_receipt_fields(details)
    assert projected == body
    assert facts == details
    assert "expects_reply" not in facts
    assert "steer" not in facts
    assert "session_id" not in facts


def test_child_quoting_parent_instructions_does_not_request_a_reply_or_steer() -> None:
    for instruction in TO_CHILD_INSTRUCTIONS.values():
        body = instruction + "\n\nquoted, not instructions to this reader"
        text = f"<subagent-message label='worker' job='job-1'>\n{body}\n</subagent-message>"
        projected, facts = hub_receipt_fields({"text": text})
        assert projected == body
        assert facts["direction"] == "to_parent"
        assert facts["label"] == "worker"
        assert facts["job_id"] == "job-1"
        assert "expects_reply" not in facts
        assert "steer" not in facts


def test_legacy_parent_envelope_has_exact_producer_provenance_and_full_body() -> None:
    body = "  first\n\nsecond  "
    for kind, instruction in TO_CHILD_INSTRUCTIONS.items():
        projected, facts = hub_receipt_fields(
            {"text": f"<parent-message>\n{instruction}\n\n{body}\n</parent-message>"}
        )
        assert projected == body
        assert facts["direction"] == "to_child"
        assert facts["expects_reply"] is (kind == "ask")
        assert facts["steer"] is (kind == "steer")


def test_sender_parser_does_not_read_advisory_fields_from_child_body() -> None:
    body = "label='not-the-sender' job='not-the-job'"
    _, facts = hub_receipt_fields({"text": f"<subagent-message>\n{body}\n</subagent-message>"})
    assert "label" not in facts
    assert "job_id" not in facts


def test_malformed_optional_fields_remain_unknown_without_invalidating_body() -> None:
    details = {
        "body": "full readable body",
        "text": "envelope",
        "direction": {},
        "label": [],
        "job_id": 12,
        "expects_reply": "true",
        "steer": None,
        "sender": {"session_id": {}, "pid": True},
    }
    body, facts = hub_receipt_fields(details)
    assert body == "full readable body"
    assert not any(
        key in facts for key in ("direction", "label", "job_id", "expects_reply", "steer")
    )
    assert facts["sender"] == {}
    assert (
        details["expects_reply"] == "true"
    ), "projection does not mutate model/persistence details"


def test_unknown_and_empty_historical_envelopes_remain_readable() -> None:
    envelope = "<subagent-message label='worker'>\n\n</subagent-message>"
    assert hub_receipt_fields({"text": envelope})[0] == envelope
    assert hub_receipt_fields({"text": "legacy raw text"})[0] == "legacy raw text"
    assert hub_receipt_fields({"body": "", "text": "model instruction"})[0] == ""

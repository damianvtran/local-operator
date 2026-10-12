"""Hub display projection preserves facts without classifying the message body."""

from local_operator.harness.comms import TO_CHILD_INSTRUCTIONS
from local_operator.harness.hub_receipts import hub_receipt_fields


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

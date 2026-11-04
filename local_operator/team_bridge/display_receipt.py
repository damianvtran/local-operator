"""Closed display-only carriage; deliberately separate from admission authority."""

from typing import Any


def validated_display_receipt(carriage: dict[str, Any], *, owner: Any) -> dict[str, Any]:
    """Return optional presentation facts, never an authority descriptor."""
    # A carriage can cross a mailbox/spool boundary after account switching.
    # Binding the display facts prevents old tenant labels riding a new owner;
    # it deliberately does not make this locator a grant to read any content.
    if owner is None:
        return {}
    receipt = carriage.get("bridge_receipt")
    source = carriage.get("bridge_receipt_source")
    if not isinstance(receipt, dict) or not isinstance(source, dict):
        return {}
    if set(receipt) - {"v", "sender", "conversation", "attachment_count"}:
        return {}
    if type(receipt.get("v")) is not int or receipt.get("v") != 1:
        return {}
    source_keys = {
        "account_id",
        "tenant_id",
        "endpoint_id",
        "conversation_id",
        "message_ids",
        "conversation_seq",
    }
    if set(source) != source_keys:
        return {}
    for key in ("account_id", "tenant_id", "endpoint_id", "conversation_id"):
        value = source[key]
        if not _label(value, 240) or (
            key != "conversation_id" and value != getattr(owner, key, None)
        ):
            return {}
    mids = source["message_ids"]
    if (
        not isinstance(mids, list)
        or not 1 <= len(mids) <= 200
        or any(not _label(mid, 240) for mid in mids)
    ):
        return {}
    if not _count(source["conversation_seq"]):
        return {}
    sender = receipt.get("sender", {})
    conversation = receipt.get("conversation", {})
    if not isinstance(sender, dict) or set(sender) - {"name", "kind", "organization"}:
        return {}
    if not isinstance(conversation, dict) or set(conversation) - {"kind", "label"}:
        return {}
    if any(not _label(sender[key], 240) for key in ("name", "organization") if key in sender):
        return {}
    if "kind" in sender and sender["kind"] not in ("human", "agent"):
        return {}
    if "label" in conversation and not _label(conversation["label"], 240):
        return {}
    if "kind" in conversation and conversation["kind"] not in ("dm", "group", "channel"):
        return {}
    if "attachment_count" in receipt and not _count(receipt["attachment_count"]):
        return {}
    # Copy the closed leaves so mutation of a sender's dict cannot mutate the
    # persisted receipt or the live event after it has passed this boundary.
    return {
        "bridge_receipt": {**receipt, "sender": dict(sender), "conversation": dict(conversation)},
        "bridge_receipt_source": {**source, "message_ids": list(mids)},
    }


def _label(value: Any, limit: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= limit
        and all(ord(c) >= 32 and not 127 <= ord(c) <= 159 for c in value)
    )


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1_000_000


def projected_display_receipt(details: dict[str, Any]) -> dict[str, Any]:
    """Schema-check ALREADY persisted receiver details for a display projection.

    Projection never reads a body or grants authority. The session's live fence
    validated the owner before persistence; replay only preserves these same
    bounded facts, never reconstructing a receipt from an advisory sender.
    """
    from types import SimpleNamespace

    source = details.get("bridge_receipt_source")
    if not isinstance(source, dict):
        return {}
    return validated_display_receipt(
        details,
        owner=SimpleNamespace(
            account_id=source.get("account_id"),
            tenant_id=source.get("tenant_id"),
            endpoint_id=source.get("endpoint_id"),
        ),
    )

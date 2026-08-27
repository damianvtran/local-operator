"""Wire-frame validation for the peer_message control op (`lop send`).

These guard the two traps the design calls out for the protocol layer: the
frame validator must accept a well-formed peer_message and reject each
malformed field, and PROTOCOL_VERSION must NOT be bumped (the op is purely
additive — an old registrant answers unknown-op gracefully, so bumping would
wrongly make old clients refuse new registrants).
"""

from __future__ import annotations

import pytest

from local_operator.mobile.types import (
    PROTOCOL_VERSION,
    EntryKind,
    validate_control_frame,
)


def test_valid_peer_message_frame_passes() -> None:
    validate_control_frame(
        {
            "op": "peer_message",
            "req": 1,
            "text": "hello there",
            "mode": "mailbox",
            "wake": True,
            "sender": {"pid": 123, "conversation_name": "peer"},
        }
    )
    # mode/wake/sender are all optional; the bare form validates too.
    validate_control_frame({"op": "peer_message", "text": "hi"})


def test_peer_message_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        validate_control_frame({"op": "peer_message", "text": "   "})
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        validate_control_frame({"op": "peer_message"})


def test_peer_message_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        validate_control_frame({"op": "peer_message", "text": "hi", "mode": "shout"})


def test_peer_message_rejects_non_bool_wake() -> None:
    with pytest.raises(ValueError, match="wake must be a boolean"):
        validate_control_frame({"op": "peer_message", "text": "hi", "wake": "yes"})


def test_peer_message_rejects_non_dict_sender() -> None:
    with pytest.raises(ValueError, match="sender must be an object"):
        validate_control_frame({"op": "peer_message", "text": "hi", "sender": ["nope"]})


def test_protocol_version_not_bumped_for_peer_messaging() -> None:
    # The peer_message op is additive; bumping the version for IT would break
    # the opposite compatibility direction (an old client refusing a new
    # registrant). The pin sits at 5 because the canonical frontend-state
    # contract (unified session state) is a genuine wire break that owns that
    # bump — peer messaging still must not move it. If this fails, someone
    # "helpfully" bumped it for an additive op — don't.
    assert PROTOCOL_VERSION == 5


def test_peer_message_is_a_known_entry_kind() -> None:
    # The phone renders peer messages as their own card; the EntryKind literal
    # must include it or the fold produces an invalid entry kind.
    assert "peer_message" in EntryKind.__args__  # type: ignore[attr-defined]

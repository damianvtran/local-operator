"""H4e-render: the bridge PROVENANCE carriage, end to end and client-local.

Contract §13.4: a bridge wake is a NEW provenance kind, "rendered as its own
card", and "must not overload the wake/peer types". Three mandatory parts carry
it — a sender-declared custom type on the peer delivery frame (validated at
every hop against a client-local allowlist), the render tuple entry, and the
TUI replay branch — plus explicit unknown-type handling over a silent drop.
This file drives the real seams for each, without sockets, sessions or models:

* the allowlist constant and the SESSION tripwire that refuses an unknown kind
  before a durable row exists;
* the carriage's omission discipline (``custom_type=None`` keeps the frame
  byte-identical — the ``message_id`` precedent);
* the probe helper every hop agrees on (a ``**kwargs`` target answers NO);
* the serving trust boundary refusing an unknown kind before forwarding;
* the LLM conversion: every family member surfaces, a FUTURE family member
  still surfaces through the fallback arm, and out-of-family bookkeeping still
  drops.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any, cast

import pytest

from local_operator.harness import render as render_mod
from local_operator.harness.message_types import (
    PEER_DELIVERABLE_CUSTOM_TYPES,
    PEER_FAMILY_INPUT_TYPES,
    PEER_MESSAGE_MESSAGE_TYPE,
    TEAM_BRIDGE_MESSAGE_TYPE,
)
from local_operator.harness.types import CustomMessage, TextContent
from local_operator.mobile import peer_client as peer_client_mod
from local_operator.session.runtime.server import receives_custom_type
from local_operator.session.session import Session

# ---------------------------------------------------------------------------
# The allowlist and the last-layer tripwire
# ---------------------------------------------------------------------------


def test_the_deliverable_allowlist_is_exactly_the_two_kinds() -> None:
    """Widening this set is a wire decision; the test names it as one."""

    assert PEER_DELIVERABLE_CUSTOM_TYPES == frozenset(
        {PEER_MESSAGE_MESSAGE_TYPE, TEAM_BRIDGE_MESSAGE_TYPE}
    )


def test_unknown_custom_type_is_refused_at_the_session_tripwire() -> None:
    """The LAST validation point: refuse, never stamp an unverifiable kind."""

    with pytest.raises(ValueError):
        Session._peer_custom_message(
            cast(Session, None), "hello", {}, custom_type="definitely_not_a_kind"
        )


def test_default_stamp_is_the_peer_kind_and_override_stamps_bridge() -> None:
    """Default stays byte-compatible; the bridge kind gets its own envelope."""

    plain = Session._peer_custom_message(
        cast(Session, None), "hi", {"pid": 7, "conversation_name": "c"}
    )
    assert plain.custom_type == PEER_MESSAGE_MESSAGE_TYPE
    assert "<peer-session-message" in plain.details["text"]

    bridge = Session._peer_custom_message(
        cast(Session, None),
        "org hi",
        {"account_id": "acc_org", "source": TEAM_BRIDGE_MESSAGE_TYPE},
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
    )
    assert bridge.custom_type == TEAM_BRIDGE_MESSAGE_TYPE
    # The dedicated wrap: the peer wrap's from_pid/conversation would be
    # misleading for an org member with no local pid.
    assert "<team-bridge-message" in bridge.details["text"]
    assert "from_account='acc_org'" in bridge.details["text"]
    assert bridge.details["body"] == "org hi"


# ---------------------------------------------------------------------------
# The probe helper (one spelling for every hop)
# ---------------------------------------------------------------------------


def test_receives_custom_type_probes_literal_parameters_only() -> None:
    class Takes:
        def receive_peer_message(self, text: str, *, custom_type: str | None = None) -> None: ...

    class Misses:
        def receive_peer_message(self, text: str) -> None: ...

    class Swallows:
        def receive_peer_message(self, text: str, **kwargs: object) -> None: ...

    class Absent:
        pass

    assert receives_custom_type(Takes()) is True
    # VAR_KEYWORD would silently drop the declaration (and the default would be
    # stamped instead) — NO, exactly like ``receives_message_id``.
    assert receives_custom_type(Misses()) is False
    assert receives_custom_type(Swallows()) is False
    assert receives_custom_type(Absent()) is False


# ---------------------------------------------------------------------------
# The carriage's omission discipline
# ---------------------------------------------------------------------------


def test_client_carriage_omits_the_key_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, dict[str, object]] = {}

    async def fake_op(record, op, fields, **kwargs):  # noqa: ANN001, ANN003
        captured["fields"] = dict(fields)
        return "ok"

    monkeypatch.setattr(peer_client_mod, "send_control_op", fake_op)
    asyncio.run(
        peer_client_mod.send_peer_message(
            cast(Any, object()), text="t", mode="mailbox", wake=False, sender={}
        )
    )
    assert "custom_type" not in captured["fields"]

    asyncio.run(
        peer_client_mod.send_peer_message(
            cast(Any, object()),
            text="t",
            mode="mailbox",
            wake=False,
            sender={},
            custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        )
    )
    assert captured["fields"]["custom_type"] == TEAM_BRIDGE_MESSAGE_TYPE


# ---------------------------------------------------------------------------
# The serving trust boundary
# ---------------------------------------------------------------------------


def test_serving_boundary_refuses_unknown_before_forwarding() -> None:
    """The handle is not the last line; an unknown kind never travels onward."""

    from local_operator.session.runtime.serving import ServingSessionHandle

    fake = types.SimpleNamespace(_check_loop_thread=lambda: None)
    bound = ServingSessionHandle.receive_peer_message.__get__(fake)
    with pytest.raises(ValueError):
        asyncio.run(bound("text", custom_type="evil_kind"))


# ---------------------------------------------------------------------------
# The LLM conversion (and the A1-class defence)
# ---------------------------------------------------------------------------


def _text_of(out: list[Any]) -> str:
    """The first TEXT part's body (the content union carries image/audio too)."""

    assert out, "message was dropped from model context"
    for part in out[0].content:
        if isinstance(part, TextContent):
            return part.text
    raise AssertionError("no text part in the converted message")


def _custom(custom_type: str, text: str) -> CustomMessage:
    return CustomMessage(
        custom_type=custom_type,
        attribution="user",
        details={"text": text},
        id=f"m-{custom_type}",
    )


def test_every_family_member_surfaces_to_the_model() -> None:
    """The BEHAVIOURAL pin for tuple ⊇ family: drop one and this goes red."""

    for custom_type in sorted(PEER_FAMILY_INPUT_TYPES):
        out = render_mod._default_convert_to_llm([_custom(custom_type, f"payload-{custom_type}")])
        assert f"payload-{custom_type}" in _text_of(out), f"{custom_type} was dropped"


def test_a_future_family_member_surfaces_through_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The A1-class defence: a forgotten tuple entry must not silently drop."""

    future = "a_future_family_type"
    monkeypatch.setattr(render_mod, "PEER_FAMILY_INPUT_TYPES", PEER_FAMILY_INPUT_TYPES | {future})
    out = render_mod._default_convert_to_llm([_custom(future, "future payload")])
    assert "future payload" in _text_of(out)


def test_out_of_family_bookkeeping_still_drops() -> None:
    """The fallback is family-gated: bookkeeping entries stay hidden."""

    out = render_mod._default_convert_to_llm([_custom("monitor_schedules", "not for the model")])
    assert out == []

"""Wire-frame validation for the peer_message control op (`lop send`).

These guard the two traps the design calls out for the protocol layer: the
frame validator must accept a well-formed peer_message and reject each
malformed field, and PROTOCOL_VERSION must NOT be bumped (the op is purely
additive — an old registrant answers unknown-op gracefully, so bumping would
wrongly make old clients refuse new registrants).

The file also guards the pending-ask wire frame: every field a viewer's rebuilt
``AskQuestion`` reads has to survive the projection, the JSON hop and the
inbound rebuild, because the detached runtime is the path an ask crosses on the
default topology.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.mobile.types import (
    PROTOCOL_VERSION,
    AskOptionWire,
    EntryKind,
    PendingRequest,
    SessionProjection,
    _projection_from_json,
    ask_pending_request,
    validate_control_frame,
)
from local_operator.session.remote import _ask_question_from_pending
from local_operator.session.runtime.types import SessionRecord


def _record() -> SessionRecord:
    return SessionRecord(
        pid=42,
        kind="tui",
        session_id="s1",
        conversation_name="ask-wire",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="secret",
    )


def _round_trip(
    question: AskQuestion,
) -> tuple[dict[str, Any], PendingRequest, AskQuestion]:
    """Project, serialize, rebuild through the REAL inbound path.

    Goes through ``_projection_from_json`` rather than a hand-rolled filter so
    the test exercises the ``known_pending`` field filter and the
    ``AskOptionWire`` rebuild the wire actually uses, then rebuilds the
    question exactly as ``_run_ask`` does.
    """
    pending = ask_pending_request(request_id=question.id, question=question)
    projection = SessionProjection(session_id="s1", pid=0, pending=pending)
    payload = projection.to_json()
    received = _projection_from_json(payload, _record())
    assert received.pending is not None
    return payload["pending"], received.pending, _ask_question_from_pending(received.pending)


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


def test_the_ask_wire_carries_every_field_the_picker_reads() -> None:
    """A field the picker reads must survive projection, JSON and rebuild.

    The regression this guards is the whole class, not one field: `recommended`
    and `persist` were declared on `AskQuestion` and simply never carried, so a
    detached session rebuilt a question that had silently lost them.
    """
    question = AskQuestion(
        id="wire-1",
        question="Which migration?",
        options=[
            AskOption(label="Gamma", description="third"),
            AskOption(label="Alpha", description="first"),
            AskOption(label="Beta", description="second"),
        ],
        recommended=2,
    )
    # The validator hoists Beta to index 0 and rewrites recommended to 0.
    assert [option.label for option in question.options] == ["Beta", "Gamma", "Alpha"]
    assert question.recommended == 0

    wire, pending, rebuilt = _round_trip(question)

    assert pending.recommended == 0
    assert wire["recommended"] == 0
    assert rebuilt.recommended == 0
    assert [option.label for option in pending.options] == ["Beta", "Gamma", "Alpha"]
    assert [option["label"] for option in wire["options"]] == ["Beta", "Gamma", "Alpha"]
    assert [option.label for option in rebuilt.options] == ["Beta", "Gamma", "Alpha"]
    # Each description travelled WITH its label, not by position against a
    # re-sorted list.
    assert [(option.label, option.description) for option in rebuilt.options] == [
        ("Beta", "second"),
        ("Gamma", "third"),
        ("Alpha", "first"),
    ]
    assert pending.persist is False
    assert rebuilt.persist is False
    assert pending.secret is False
    assert rebuilt.secret is False

    # A future field added to PendingRequest without a wire value fails here
    # rather than in a user's terminal.
    carried = set(wire)
    expected = {f.name for f in fields(PendingRequest)}
    assert carried == expected, (
        "a PendingRequest field is not reaching the wire (or vice versa); "
        "a new field needs a wire value AND a rebuild in _run_ask"
    )

    # And the harness side, which is where the defect actually lived: the
    # ledger of what the wire deliberately does NOT carry.
    ask_fields = set(AskQuestion.model_fields)
    # `id` is replaced by `request_id` (mapped back via _pending_question_ids);
    # `multi` is DEFERRED — the answer path is single-value end to end, so
    # carrying it would render a picker that discards every answer but the first.
    deliberately_dropped = {"id", "multi"}
    # `question` is the one deliberate RENAME: AskQuestion.question is carried
    # as PendingRequest.title.
    assert ask_fields - deliberately_dropped <= expected | {"question"}, (
        "AskQuestion grew a field the ask wire does not carry; carry it or add "
        "it to deliberately_dropped with the reason"
    )


def test_the_rebuild_does_not_re_hoist_an_already_hoisted_option_list() -> None:
    """`recommended` indexes the list AS CARRIED, so the rebuild must not rotate.

    A rebuild that re-ran the validator's hoist on an already-hoisted list would
    move the badge to the wrong row. Rotating by 0 is the identity today, so
    this passes on the current validator — the test pins it so a future change
    that made the hoist unconditional fails here instead of in a terminal.
    """
    question = AskQuestion(
        id="wire-2",
        question="Which migration?",
        options=[
            AskOption(label="Beta", description="second"),
            AskOption(label="Gamma", description="third"),
            AskOption(label="Alpha", description="first"),
        ],
        recommended=0,
    )

    _, _, rebuilt = _round_trip(question)

    # The ORDER LIST, not options[recommended].label, which would also pass
    # under a rotation that moved everything.
    assert [option.label for option in rebuilt.options] == ["Beta", "Gamma", "Alpha"]
    assert rebuilt.recommended == 0

    # Idempotence: project and rebuild the REBUILT question a second time.
    _, _, twice = _round_trip(rebuilt)
    assert [option.label for option in twice.options] == ["Beta", "Gamma", "Alpha"]
    assert twice.recommended == 0


def test_a_persisted_secret_ask_keeps_its_persist_intent_across_the_wire() -> None:
    """The rebuilt secret question is a faithful copy, flag included.

    This does NOT fix a live data-loss bug and must not claim to: the credential
    promotion reads `question.persist` off the ORIGINAL AskQuestion list held by
    the ask tool in the OWNER process (`builtin.py`'s
    `_report_secret_answers(params.questions, ...)`), not off the viewer's
    rebuilt object, and the picker reads `persist` nowhere. What this pins is
    fidelity of the rebuild: `persist` is a declared field of the question every
    viewer-side consumer sees, so a future viewer-side affordance ("this will be
    saved permanently" on the paste field) reads the truth instead of a default.

    The last two assertions are the load-bearing ones: they cover the version-
    skew guard in `_ask_question_from_pending`. `AskQuestion`'s validator raises
    on both contradictory combinations, `ValidationError` is a `ValueError`, and
    `_run_ask` does not catch `ValueError` — so an unguarded rebuild is not a
    missing badge but a question that never mounts.
    """
    question = AskQuestion(
        id="OPENAI_API_KEY",
        question="Paste your key",
        secret=True,
        persist=True,
    )

    wire, pending, rebuilt = _round_trip(question)

    assert pending.persist is True
    assert wire["persist"] is True
    assert rebuilt.persist is True
    assert rebuilt.secret is True
    assert rebuilt.options == []
    assert rebuilt.recommended is None
    # The secret VALUE has no field on this path and none may be added: only the
    # flag rides.
    assert "value" not in wire
    assert "answer" not in wire

    # A hostile or version-skewed payload: secret with a recommendation.
    hostile = PendingRequest(
        request_id="OPENAI_API_KEY",
        kind="ask",
        title="Paste your key",
        secret=True,
        recommended=0,
        persist=True,
    )
    guarded = _ask_question_from_pending(hostile)
    assert guarded.recommended is None
    assert guarded.secret is True
    assert guarded.persist is True

    # The mirror case: persist without a secret to persist.
    mirrored = PendingRequest(
        request_id="wire-3",
        kind="ask",
        title="Which migration?",
        options=[
            AskOptionWire(label="A", description="a"),
            AskOptionWire(label="B", description="b"),
        ],
        secret=False,
        persist=True,
    )
    assert _ask_question_from_pending(mirrored).persist is False


def test_an_old_ask_payload_without_the_new_keys_still_rebuilds() -> None:
    """NEW viewer / OLD payload is the dangerous skew direction.

    Inbound reconstruction filters to known field names and calls
    `PendingRequest(**pending_kwargs)`, so a payload written before these fields
    existed supplies neither key. The dataclass defaults are the entire
    mitigation — without them this raises TypeError and the ask card crashes.
    """
    old_payload = {
        "session_id": "s1",
        "pid": 0,
        "pending": {
            "request_id": "old-1",
            "kind": "ask",
            "title": "Which migration?",
            "detail": "",
            "options": [{"label": "Beta", "description": "second"}],
            "secret": False,
            "question_index": 0,
            "question_total": 1,
        },
    }

    received = _projection_from_json(old_payload, _record())

    assert received.pending is not None
    assert received.pending.recommended is None
    assert received.pending.persist is False

"""The four condensation facts on the phone's transcript wire.

WHAT THIS PINS, and why these four fields exist at all. The phone is porting the
desktop's condensation model (tool groups, settled-turn bars, the plan), and that
model reads four facts this wire never carried:

* ``stop_reason`` — the provider's phase declaration on an assistant row. Without
  it the lead-in frame (prose that settles a moment before its own tool row
  appears) reads as a settled answer, so the cycle condenses and un-condenses one
  frame later.
* ``calls_tools`` — the assistant row carries calls of its own. It cannot be
  derived client-side: live tool ids are ``tc-<call>`` while durable ids are
  ``<msg>:<call>``, and the fold legitimately drops some call rows.
* ``custom_type`` — the verbatim ``custom_type`` of the ``CustomMessage`` a notice
  was folded from: the receipt family the fold groups by, and the fact that
  decides whether a row opens a cycle.
* ``terminal`` — this notice ends the turn. An explicit flag, not a severity
  heuristic: ``compaction_refused`` is error-level and is not terminal.

The tests below are about the CONTRACT, not about the model:
1. both producer arms (the live fold and the history fold) put the same values on
   the same row for the same message — the drift risk, because a reconnect
   replaces a live row with a durable one and a row that changes meaning across
   that boundary is a fold that opens and closes on its own;
2. the fields survive the frame cap's ``details``-stripping tier, which is why
   they are top level rather than in ``details``;
3. ``to_json`` omits each at its default (a capped frame is 700 KB, and these are
   absent on most rows), and ``_projection_from_json`` reconstructs them, ignores
   unknown keys, and defaults an absent one — the two directions of skew the
   rolling upgrade actually produces.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentMessage,
    CustomMessage,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    TextContent,
)
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.projection import cap_projection_frame
from local_operator.mobile.types import (
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
    _projection_from_json,
)

HUB_MESSAGE_TYPE = "hub_message"
JOB_RESULT_TYPE = "job_result"
WAKE_PROMPT_TYPE = "wake_prompt"
MONITOR_PROMPT_TYPE = "monitor_prompt"
TEAM_BRIDGE_TYPE = "team_bridge_message"
SESSION_INCIDENT_TYPE = "session_incident"


def session_record() -> SessionRecord:
    """The discovery record the socket seam rebuilds a projection against."""
    return SessionRecord(
        pid=1,
        kind="tui",
        session_id="s1",
        conversation_name="wire",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="secret",
    )


def make_fold() -> ProjectionFold:
    return ProjectionFold(SessionProjection(session_id="s1", pid=1))


def assistant(
    text: str, *, stop_reason: str | None = None, calls: list[Any] | None = None
) -> Message:
    """An assistant message with the phase declaration the provider sends."""
    message = Message.assistant()
    message.id = "m-assistant"
    message.content = [TextContent(text=text)]
    if stop_reason is not None:
        message.stop_reason = stop_reason
    message.tool_calls = calls or []
    return message


def custom(custom_type: str, message_id: str, text: str = "payload") -> Any:
    return CustomMessage(id=message_id, custom_type=custom_type, details={"text": text})


def history_fold(messages: list[AgentMessage]) -> dict[str, TranscriptEntry]:
    """The HISTORY arm: the pure fold the seed and every history page use."""
    fold = make_fold()
    fold.fold_history(list(messages))
    return {entry.id: entry for entry in fold.projection.transcript}


def live_fold(messages: list[AgentMessage]) -> dict[str, TranscriptEntry]:
    """The LIVE arm, driven by the events a running session emits.

    The live fold never sees a durable ``role="tool"`` message, so the test builds
    the same three shapes by hand: a start (which paints the row), an end (which
    settles the prose and the phase), and the custom deliveries a session receives
    as notices.
    """
    fold = make_fold()
    for message in messages:
        if isinstance(message, Message) and message.role == "assistant":
            fold.fold_event(MessageStartEvent(message=message))
            fold.fold_event(MessageEndEvent(message=message))
        elif isinstance(message, CustomMessage):
            fold.note_nonpeer_delivery(
                message.details.get("text", ""),
                message_id=message.id,
                custom_type=message.custom_type,
            )
    fold.fold_event(AgentEndEvent(generation=1))
    return {entry.id: entry for entry in fold.projection.transcript}


# --------------------------- the two arms agree ---------------------------


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("toolUse", "toolUse"), ("stop", "stop"), (None, "")],
)
def test_the_assistant_phase_survives_both_arms(stop_reason: str | None, expected: str) -> None:
    message = assistant("Looking at it.", stop_reason=stop_reason)
    for rows in (history_fold([message]), live_fold([message])):
        row = rows[message.id]
        assert row.kind == "assistant"
        assert row.stop_reason == expected


def test_the_assistant_calls_fact_survives_both_arms() -> None:
    """A row that carried calls says so, and one that did not says nothing."""
    message = assistant("Working.", stop_reason="toolUse", calls=[])
    message_with_calls = assistant("Working.", stop_reason="toolUse", calls=[])
    # The live arm reads the same field the history arm does; the distinction the
    # model needs is "did this row carry calls", and an empty list is "no".
    for rows in (history_fold([message]), live_fold([message_with_calls])):
        assert rows[message.id].calls_tools is False


def test_a_custom_notice_carries_its_family_in_both_arms() -> None:
    """The receipt family is what a fold groups by, so it must not be arm-specific."""
    messages: list[AgentMessage] = [custom(JOB_RESULT_TYPE, "c-job")]
    durable = history_fold(messages)
    live = live_fold(messages)
    assert durable["c-job"].custom_type == JOB_RESULT_TYPE
    assert live["c-job"].custom_type == JOB_RESULT_TYPE


def test_the_stop_notice_is_terminal_in_the_history_arm() -> None:
    """A refused turn's ``:stop`` notice ends the turn.

    The live arm's terminal notice is the ``AgentEndEvent.error`` row, which has
    its own id (``err-…``); the history arm's is ``<message>:stop``. Both are
    terminal, which is the fact the model reads.
    """
    fold = make_fold()
    fold.fold_history([assistant("I cannot.", stop_reason="refusal")])
    stops = [row for row in fold.projection.transcript if row.kind == "notice"]
    assert stops, "a refusal must paint a notice"
    assert all(row.terminal for row in stops)


def test_a_session_incident_is_terminal_in_the_history_arm() -> None:
    """`session_incident` is the one custom family that ends a turn.

    Read through the generic notice fallback, which is why the flag is set from the
    custom type there rather than in an arm of its own.

    WHY ONLY THE HISTORY ARM IS ASSERTED (agent review round 1, C1-m4). These
    families reach the phone through the history refold, not through the live fold:
    production calls ``note_nonpeer_delivery`` only from a
    ``PeerMessageDeliveredEvent``, and that path admits ``peer_message`` and
    ``team_bridge_message`` alone. Driving an incident through it in a test would be
    asserting a path production refuses - so the live arm's coverage is the family
    it actually carries (the ``team_bridge_message`` case below), and the incident's
    terminality is pinned here, on the arm that produces it.
    """
    durable = history_fold([custom(SESSION_INCIDENT_TYPE, "c-incident")])
    assert durable["c-incident"].custom_type == SESSION_INCIDENT_TYPE
    assert durable["c-incident"].terminal is True


def test_a_bridge_delivery_carries_its_family_in_both_arms() -> None:
    """The one receipt family the LIVE path really delivers.

    ``team_bridge_message`` is admitted by the live delivery path and folded by the
    history path, so it is the family where "both arms agree" is a claim about
    production rather than about a fixture. Round 1 measured the two arms matching
    on this type (notice + custom_type on each), which is what this pins.
    """
    messages: list[AgentMessage] = [custom(TEAM_BRIDGE_TYPE, "c-bridge")]
    durable = history_fold(messages)
    live = live_fold(messages)
    assert durable["c-bridge"].custom_type == TEAM_BRIDGE_TYPE
    assert live["c-bridge"].custom_type == TEAM_BRIDGE_TYPE
    assert durable["c-bridge"].terminal is False
    assert live["c-bridge"].terminal is False


def test_a_job_result_and_a_monitor_prompt_keep_their_family_in_the_history_arm() -> None:
    """ "Every family the fold groups by" is a READER of `custom_type`, and the
    history arm is where those rows are built."""
    durable = history_fold(
        [custom(JOB_RESULT_TYPE, "c-job"), custom(MONITOR_PROMPT_TYPE, "c-monitor")]
    )
    assert durable["c-job"].custom_type == JOB_RESULT_TYPE
    assert durable["c-monitor"].custom_type == MONITOR_PROMPT_TYPE
    assert durable["c-job"].terminal is False
    assert durable["c-monitor"].terminal is False


def test_a_wake_notice_is_not_terminal_and_keeps_its_family() -> None:
    """A wake opens a cycle; it does not close the turn.

    HISTORY ARM ONLY - a wake reaches the phone through the refold, never through
    the live delivery path (see the incident test's note)."""
    fold = make_fold()
    fold.fold_history([custom(WAKE_PROMPT_TYPE, "c-wake", "Scheduled wake w-1\n\nDo the thing.")])
    rows = [row for row in fold.projection.transcript if row.id == "c-wake"]
    assert rows, "a wake must paint a row"
    assert rows[0].custom_type == WAKE_PROMPT_TYPE
    assert rows[0].terminal is False


# ------------------------- the wire, in both directions -------------------------


def test_to_json_omits_each_fact_at_its_default() -> None:
    entry = TranscriptEntry(id="r1", kind="assistant", text="hi")
    payload = entry.to_json()
    for key in ("stop_reason", "calls_tools", "custom_type", "terminal"):
        assert key not in payload, f"{key} must not ride a row that does not carry it"


def test_to_json_sends_each_fact_when_it_is_set() -> None:
    entry = TranscriptEntry(
        id="r1",
        kind="assistant",
        text="hi",
        stop_reason="toolUse",
        calls_tools=True,
        custom_type=JOB_RESULT_TYPE,
        terminal=True,
    )
    payload = entry.to_json()
    assert payload["stop_reason"] == "toolUse"
    assert payload["calls_tools"] is True
    assert payload["custom_type"] == JOB_RESULT_TYPE
    assert payload["terminal"] is True


def test_the_frame_cap_cannot_strip_the_facts() -> None:
    """The reason they are TOP LEVEL: tier 2 empties ``details``.

    A fact the client needs on every repaint cannot live in the one field the cap
    drops first, so this asserts the field survives the same treatment ``details``
    does not — by round-tripping through the wire payload the cap is applied to.
    """
    entry = TranscriptEntry(
        id="r1", kind="tool", text="", tool_name="bash", details={"args": {"x": 1}}
    )
    payload = entry.to_json()
    payload["details"] = {}
    rebuilt = _projection_from_json(
        {"session_id": "s1", "pid": 1, "transcript": [{**payload, "terminal": True}]},
        session_record(),
    )
    assert rebuilt.transcript[0].terminal is True


def test_a_capped_frame_row_carries_no_empty_fact() -> None:
    """SPARSE MUST HOLD ON THE FRAME PATH, not only on a single row.

    Found by CI, not by these tests: ``SessionProjection.to_json`` walked the rows
    with ``asdict``, which emits every dataclass field at its DEFAULT, so a plain
    row left the daemon carrying ``stop_reason: ""``, ``calls_tools: false``,
    ``custom_type: ""`` and ``terminal: false`` - four keys times up to 120 rows
    saying nothing. The offloop wire digest (``test_projection_offloop``) caught
    the frame moving; this test names the reason. The frame is the only push form,
    so sparse-on-the-wire has to hold HERE or it does not hold at all.
    """
    plain = Message.assistant()
    plain.id = "m-plain"
    plain.content = [TextContent(text="plain")]
    fold = make_fold()
    fold.fold_history([plain])
    frame, _ = cap_projection_frame(fold.projection)
    assert frame["transcript"], "the fixture must carry a row"
    for row in frame["transcript"]:
        for key in ("stop_reason", "calls_tools", "custom_type", "terminal"):
            assert key not in row, f"{key} rode a capped frame row that does not carry it"


def test_a_capped_frame_row_keeps_a_fact_it_does_carry() -> None:
    """And the sparseness is not achieved by dropping the facts everywhere."""
    fold = make_fold()
    fold.fold_history([assistant("a", stop_reason="toolUse")])
    frame, _ = cap_projection_frame(fold.projection)
    rows = [row for row in frame["transcript"] if row["id"] == "m-assistant"]
    assert rows, "the fixture must carry the assistant row"
    assert rows[0]["stop_reason"] == "toolUse"


def test_a_projection_round_trips_the_facts() -> None:
    projection = SessionProjection(
        session_id="s1",
        pid=1,
        transcript=[
            TranscriptEntry(
                id="a",
                kind="assistant",
                text="hi",
                stop_reason="stop",
                calls_tools=True,
            ),
            TranscriptEntry(
                id="n", kind="notice", text="x", custom_type=JOB_RESULT_TYPE, terminal=True
            ),
        ],
    )
    rebuilt = _projection_from_json(projection.to_json(), session_record())
    assert rebuilt.transcript[0].stop_reason == "stop"
    assert rebuilt.transcript[0].calls_tools is True
    assert rebuilt.transcript[1].custom_type == JOB_RESULT_TYPE
    assert rebuilt.transcript[1].terminal is True


def test_an_older_payload_reads_as_unknown() -> None:
    """A new viewer, an old daemon: absence is the desktop's "unknown" branch."""
    rebuilt = _projection_from_json(
        {
            "session_id": "s1",
            "pid": 1,
            "transcript": [
                {"id": "a", "kind": "assistant", "text": "hi"},
                {"id": "n", "kind": "notice", "text": "x"},
            ],
        },
        session_record(),
    )
    assert rebuilt.transcript[0].stop_reason == ""
    assert rebuilt.transcript[0].calls_tools is False
    assert rebuilt.transcript[1].custom_type == ""
    assert rebuilt.transcript[1].terminal is False


def test_a_newer_payload_is_not_refused_by_an_older_viewer() -> None:
    """The other direction: unknown keys are dropped, never an exception."""
    rebuilt = _projection_from_json(
        {
            "session_id": "s1",
            "pid": 1,
            "transcript": [
                {
                    "id": "a",
                    "kind": "assistant",
                    "text": "hi",
                    "stop_reason": "stop",
                    "a_future_field": {"nested": True},
                }
            ],
        },
        session_record(),
    )
    assert rebuilt.transcript[0].stop_reason == "stop"
    assert not hasattr(rebuilt.transcript[0], "a_future_field")

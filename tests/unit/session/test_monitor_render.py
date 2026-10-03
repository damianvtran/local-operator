"""A monitor delivery must reach the MODEL, not only the transcript.

``harness/render.py`` turns a transcript into the provider request through an
ALLOW-LIST of custom types; an unlisted type is dropped as bookkeeping. The
``monitor_prompt`` type was missing from it, so every delta (and every
lifecycle notice) was persisted, counted as a delivery and shown to the human,
while the provider request carried ``messages=[]`` and the agent replied
"Monitor fired" and re-ran the check itself. These tests pin the three places
that can regress independently: the live turn, the replay of a stored
transcript, and every kind of monitor message through the renderer.
"""

from __future__ import annotations

import pytest

from local_operator.harness.render import _default_convert_to_llm
from local_operator.harness.types import (
    CustomMessage,
    Message,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.monitors.delivery import MonitorDelivery
from local_operator.monitors.spec import MONITOR_PROMPT_MESSAGE_TYPE
from local_operator.session.transcript import Transcript
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for


def _delivery(text_marker: str = "DELTA-CANARY") -> MonitorDelivery:
    return MonitorDelivery(
        monitor_id="m1",
        name="watch",
        tool="bash",
        changes=1,
        checks=2,
        skipped=0,
        delta_text=f"+1/-0 changed lines\n+ {text_marker}",
        at_ms=1_756_000_000_000,
    )


@pytest.mark.asyncio
async def test_monitor_prompt_reaches_the_model_as_a_user_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    stream = ScriptedStream([[StreamTextDelta(delta="seen"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    try:
        await session._deliver_monitor(_delivery())
        await wait_for(lambda: bool(stream.requests))
        # The idle path opens a paid turn: that turn's provider request must
        # carry the delta. Before the fix it carried ``messages=[]``.
        delivered = stream.requests[0].messages
        assert delivered, "the monitor turn was opened with an empty message list"
        assert any("DELTA-CANARY" in m.text for m in delivered)
        assert all(m.role == "user" for m in delivered)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_monitor_prompt_survives_replay_through_build_llm_history(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    stream = ScriptedStream([[StreamTextDelta(delta="seen"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    try:
        await session._deliver_monitor(_delivery("REPLAY-CANARY"))
        await wait_for(lambda: bool(stream.requests))
        await wait_for(lambda: not session._is_streaming)
    finally:
        await session.dispose()

    # A resumed session replays the stored rows, which is how the historic
    # rows (written while the renderer dropped them) first reach a model.
    replayed = Transcript(session._transcript.directory).build_llm_history()
    converted = _default_convert_to_llm(replayed)
    user_texts = [m.text for m in converted if isinstance(m, Message) and m.role == "user"]
    assert any("REPLAY-CANARY" in t for t in user_texts)


@pytest.mark.parametrize("kind", ["delta", "disabled", "stalled", "restored"])
def test_every_monitor_notice_kind_renders(kind):
    """One custom type carries all four kinds, so one allow-list entry covers
    them; a regression that gives a kind its own type would fail here."""
    message = CustomMessage(
        custom_type=MONITOR_PROMPT_MESSAGE_TYPE,
        attribution="user",
        details={"monitor_id": "m1", "kind": kind, "text": f"(monitor) {kind}-CANARY"},
    )
    out = _default_convert_to_llm([message])
    assert len(out) == 1
    assert out[0].role == "user"
    assert f"{kind}-CANARY" in out[0].text


def test_the_schedule_bookkeeping_entry_stays_out_of_the_model_context():
    """``monitor_schedules`` is state read via ``latest_custom_entry``; only the
    delivery type is model-visible."""
    message = CustomMessage(
        custom_type="monitor_schedules", attribution="user", details={"monitors": []}
    )
    assert _default_convert_to_llm([message]) == []

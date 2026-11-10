"""The turn's wait on the attention-store write is bounded (turn-lock wedge).

Hosted gates of ``test_quiet_acceptance`` failed with ``agent_end_count=2,
is_streaming=False, turn_lock_locked=True`` and a task parked in
``Session._publish_attention_outcome`` -> ``asyncio.to_thread``: the store write
runs in a turn's ``finally`` with ``_turn_lock`` held, so a write that blocks
(SQLite contention is budgeted up to ~12 s) left the next owner waiting. These
tests inject a write that blocks on an event, which is the same state a wedged
store produces, and pin: the lock is released within the ceiling, the expiry is
handled like a deferral (ladder armed, loud log), the orphaned worker finishing
late is harmless, and a normal publish is untouched.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid

import pytest

import local_operator.session.session as session_module
from local_operator.harness.types import AgentEndEvent, StreamEndEvent, StreamTextDelta
from local_operator.paths import config_dir
from local_operator.session.attention import AttentionStore, conversation_identity, provisional_anchor
from tests.unit.session.test_quiet_acceptance import SETTLE_TIMEOUT_S, peer_input
from tests.unit.session.test_session import ScriptedStream, make_session

TEST_CEILING_S = 0.3


def _answer():
    return ScriptedStream([[StreamTextDelta(delta="an answer"), StreamEndEvent(stop_reason="stop")]])


def test_shipping_ceilings_leave_room_inside_the_settle_window():
    total = (
        session_module.ATTENTION_PUBLISH_OFFLOAD_CEILING_S
        + session_module.ATTENTION_BANNER_STAMP_CEILING_S
    )
    assert total < SETTLE_TIMEOUT_S / 2


@pytest.mark.asyncio
async def test_a_blocked_store_write_does_not_pin_the_turn_lock(tmp_path, monkeypatch, caplog):
    # raising=False: on code without the ceiling this still reaches the hang
    # assertion below instead of failing on a missing attribute.
    monkeypatch.setattr(
        session_module, "ATTENTION_PUBLISH_OFFLOAD_CEILING_S", TEST_CEILING_S, raising=False
    )
    release = threading.Event()
    entered = threading.Event()
    real_publish = AttentionStore.publish
    calls: list[str] = []

    def blocked_publish(self, conversation, token, *args, **kwargs):
        calls.append(token)
        entered.set()
        release.wait(30)  # the worker the turn cannot kill
        return real_publish(self, conversation, token, *args, **kwargs)

    monkeypatch.setattr(AttentionStore, "publish", blocked_publish)
    session = make_session(tmp_path, _answer())
    try:
        with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
            try:
                await asyncio.wait_for(session._prompt_messages([peer_input()]), timeout=5)
            except TimeoutError as exc:
                raise AssertionError(
                    "turn never finished: "
                    f"entered_publish={entered.is_set()}, "
                    f"turn_lock_locked={session._turn_lock.locked()}, "
                    f"is_streaming={session.is_streaming}"
                ) from exc
        assert entered.is_set()
        assert not session._turn_lock.locked()
        assert "deferred" in caplog.text and "orphaned" in caplog.text, caplog.text
        assert session._attention_republish_due, "expiry must arm the republish ladder"

        # The orphan finishing late is harmless: it lands the one row by token,
        # and a ladder republish of the same token adds nothing.
        release.set()
        store = AttentionStore(config_dir() / "attention.db")
        identity = conversation_identity(session._transcript.directory)
        for _ in range(200):
            if store.state(identity).get("completion_token"):
                break
            await asyncio.sleep(0.05)
        token = store.state(identity)["completion_token"]
        assert token == calls[0]
        before = store.revision()
        assert await session._republish_journalled_outcome() in (True, False)
        assert store.revision() == before
    finally:
        release.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_normal_publish_still_lands_and_keeps_order(tmp_path):
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        store = AttentionStore(config_dir() / "attention.db")
        identity = conversation_identity(session._transcript.directory)
        first = str(uuid.uuid4())
        session._attention_run_token = first
        session._attention_outcome = AgentEndEvent(messages=[], error="first failure")
        await session._publish_attention_outcome()
        second = str(uuid.uuid4())
        session._attention_run_token = second
        session._attention_outcome = AgentEndEvent(messages=[], error="second failure")
        await session._publish_attention_outcome()
        assert store.state(identity)["completion_token"] == second
        assert not session._attention_republish_due
        # Replaying the older token is idempotent and does not move the newest row.
        store.publish(identity, first, provisional_anchor(first), "error")
        assert store.state(identity)["completion_token"] == second
    finally:
        await session.dispose()

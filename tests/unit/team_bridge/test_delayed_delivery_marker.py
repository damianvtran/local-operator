"""The delayed-delivery card line (write-once fix PR-2, operator via Aida 2026-11-24).

Messages screening allowed sat for days in ``write_once_conflict`` holds. When a
supersession or an owner Deliver finally lets one out, the card says — once,
quietly — that the delivery itself is late, so the agent does not read a
days-old request as fresh. The sent time stays the hub's own (the card carries
none; ``channel_history`` rows are the hub's rows).
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from local_operator.team_bridge import screening
from local_operator.team_bridge.connector import conversation
from local_operator.team_bridge.connector.coalesce import merge_metadata
from tests.unit.team_bridge.test_hold_release import rig  # noqa: F401 — a fixture
from tests.unit.team_bridge.test_screening import _setup


def _metadata(message_id: str = "bcmsg_abcdef01", seq: int = 1, **extra):
    return conversation.ConversationCardMetadata(
        conversation_id="bcv_abcdefgh",
        tenant_id="tnt_a",
        conversation_label="room",
        sender_label="Ada",
        sender_kind="human",
        mention=False,
        message_id=message_id,
        conversation_seq=seq,
        **extra,
    )


def test_the_line_is_quiet_closed_and_only_for_a_real_delay() -> None:
    line = conversation.delayed_delivery_line
    assert line(0) is None and line(59) is None
    assert line(60) == (
        "delivered late — held 1 min by screening; its sent time in channel_history is "
        "the original"
    )
    assert "3 h 5 min" in str(line(3 * 3600 + 5 * 60))
    assert "2 h by" in str(line(2 * 3600))
    assert "9 days" in str(line(9 * 86400 + 7))
    # Closed words only: never a hold code or screener detail.
    for text in (line(60), line(9 * 86400)):
        assert text is not None and "write_once" not in text and "conflict" not in text


def test_the_card_carries_the_line_only_when_held() -> None:
    plain = conversation.build_conversation_card(_metadata())
    assert "delivered late" not in plain
    late = conversation.build_conversation_card(_metadata(held_for_s=2 * 86400 + 30))
    assert late.count("delivered late — held 2 days by screening") == 1
    # Everything else on the card is unchanged.
    assert late.replace(str(conversation.delayed_delivery_line(2 * 86400 + 30)) + "\n", "") == plain


def test_an_ordinary_card_record_is_byte_identical_and_a_late_one_round_trips() -> None:
    plain = _metadata()
    record = conversation._card_record(plain)
    assert "held_for_s" not in record  # an older build still reads it
    late = replace(plain, held_for_s=7200)
    late_record = conversation._card_record(late)
    assert '"held_for_s":7200' in late_record
    rebuilt = conversation._card_from_record(late_record, plain, [plain.message_id])
    assert rebuilt.held_for_s == 7200
    assert conversation._card_from_record(record, plain, [plain.message_id]).held_for_s == 0


def test_a_burst_keeps_the_longest_hold() -> None:
    merged = merge_metadata(
        [_metadata("bcmsg_a1", 1, held_for_s=0), _metadata("bcmsg_a2", 2, held_for_s=5400)]
    )
    assert merged.held_for_s == 5400


def test_the_prior_hold_is_read_from_either_key_spelling(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    assert conversation.prior_delay_hold(store, owner, "bcmsg_abcdef01") is None
    before = time.time()
    screening._persist_hold(store, owner, "bcmsg_abcdef01", "write_once_conflict")
    screening._persist_hold(store, owner, "bcmsg_other", "model_hold")
    found = conversation.prior_delay_hold(store, owner, "bcmsg_abcdef01")
    assert found is not None
    created, conflict = found
    assert conflict is True and before - 1 <= created <= time.time()
    other = conversation.prior_delay_hold(store, owner, "bcmsg_other")
    assert other is not None and other[1] is False

    class Broken:
        def screening_hold_rows(self, _owner):
            raise OSError("unreadable")

    # Best effort: an unreadable store means no marker, never a failed delivery.
    assert conversation.prior_delay_hold(Broken(), owner, "bcmsg_abcdef01") is None


def test_an_owner_deliver_after_hours_puts_the_line_on_the_real_wake(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:
    """The real conversation spine: a work-scope hold, aged three hours, then the
    owner's Deliver. The one wake carries the line; the hold's age is the hold
    row's own ``created_at`` (read before the commit deletes it)."""

    import sqlite3

    from local_operator.team_bridge.store import state_db_path
    from tests.unit.team_bridge import test_hold_release as thr

    config_dir, _store, owner = rig
    _server, wakes, run = thr._spine(monkeypatch, config_dir, owner, thr.Runner(thr.NEW_WORK))
    assert run()["action"] == "hold"
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("UPDATE screening_holds SET created_at = created_at - ?", (3 * 3600 + 30,))
    thr._decide(config_dir, owner, "deliver")
    assert run()["action"] == "delivered"
    assert len(wakes) == 1 and wakes[0].count("delivered late — held 3 h by screening") == 1


def _co_due_conflict_holds(tmp_path, monkeypatch, *, aged_s: float, commit_clears: bool = False):
    """Two messages in ONE conversation, both in a due ``write_once_conflict``
    hold aged ``aged_s``, re-driven by the REAL retry pass: the batch prescreen
    (``prescreen_conversation_refs`` -> real ``screen_batch_async`` ->
    ``finish_screening``) commits both receipts, deleting both holds, before
    either handler runs (#4533 round 1, M1).

    Today that commit deletes only the BARE-id hold, and the retry pass reads
    the connector's composite row, which survives until the handler. With
    ``commit_clears`` the commit also deletes every spelling (what a fix of
    the pre-existing two-spelling hold deletion would do); the handler must
    still find the hold age, which only the pre-batch capture can supply.
    """

    import sqlite3

    from local_operator.team_bridge.connector import loop as loop_mod
    from local_operator.team_bridge.store import state_db_path
    from tests.unit.team_bridge.test_batch_screening import (
        BatchRunner,
        _hub_rig,
        deliver_everything,
    )

    failing = {"on": True}
    if commit_clears:
        from local_operator.team_bridge.store import Store

        real_finish = Store.finish_screening

        def finish_and_clear(self, owner, message_id, **kwargs):
            real_finish(self, owner, message_id, **kwargs)
            if kwargs.get("receipt") is not None:
                with sqlite3.connect(state_db_path(self.config_dir)) as conn:
                    conn.execute(
                        "DELETE FROM screening_holds WHERE message_id LIKE ?",
                        (f"%/{message_id}/%",),
                    )

        monkeypatch.setattr(Store, "finish_screening", finish_and_clear)

    def reply(prompt: str) -> str:
        if failing["on"]:
            raise RuntimeError("provider down")
        return deliver_everything(prompt, default_model="")[0]

    runner = BatchRunner(reply)
    env = _hub_rig(tmp_path, monkeypatch, runner)
    # Each message's OWN metadata, before the burst merges them into one card.
    from local_operator.team_bridge.connector import coalesce

    real_merge = coalesce.merge_metadata
    env.held = {}

    def spy(items):
        env.held.update({item.message_id: item.held_for_s for item in items})
        return real_merge(items)

    monkeypatch.setattr(coalesce, "merge_metadata", spy)
    posted = [env.hub.post("bep_b", f"can you send that file {n}") for n in "12"]
    env.rig._iterate(env.owner, env.config_dir)
    assert env.wakes == [], "both held while the vendor is down"
    with sqlite3.connect(state_db_path(env.config_dir)) as conn:
        conn.execute(
            "UPDATE screening_holds SET hold_code = 'write_once_conflict', no_batch = 0, "
            "created_at = created_at - ?, retry_at = 0",
            (aged_s,),
        )
    failing["on"] = False
    calls_before = len(runner.calls)
    loop_mod._RETRY_ATTEMPTS.clear()
    env.clock.now += 1.0
    env.rig._iterate(env.owner, env.config_dir)
    batch_calls = [p for p in runner.calls[calls_before:] if "Message ids are" in p]
    assert (
        len(batch_calls) == 1 and len(runner.calls) - calls_before == 1
    ), "both re-driven through ONE batch call, no single screen"
    for message in posted:
        assert env.store.screening_hold(env.owner, message.id) is None
    return env, posted


@pytest.mark.parametrize("commit_clears", [False, True], ids=["today", "commit-clears-all"])
def test_a_batched_re_drive_keeps_the_line_for_both_co_due_holds(
    tmp_path, monkeypatch, commit_clears
) -> None:
    env, posted = _co_due_conflict_holds(
        tmp_path, monkeypatch, aged_s=3 * 3600 + 30, commit_clears=commit_clears
    )
    text = "\n".join(wake["card"] for wake in env.wakes)
    assert all(message.id in text for message in posted), "both delivered"
    # Each message carries its own hold age; the burst's one card shows the line.
    assert sorted(env.held) == sorted(message.id for message in posted)
    assert all(3 * 3600 <= held <= 3 * 3600 + 120 for held in env.held.values()), env.held
    assert len(env.wakes) == 1
    assert text.count("delivered late — held 3 h by screening") == 1


def test_a_batched_re_drive_of_a_short_hold_says_nothing(tmp_path, monkeypatch) -> None:
    env, posted = _co_due_conflict_holds(tmp_path, monkeypatch, aged_s=20)
    text = "\n".join(wake["card"] for wake in env.wakes)
    assert all(message.id in text for message in posted), "both delivered"
    assert env.held and all(held < 60 for held in env.held.values()), env.held
    assert "delivered late" not in text

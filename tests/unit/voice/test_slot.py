"""The machine-wide voice slot (design V3): one session at a time, switchable."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from local_operator.voice import slot
from local_operator.wakes.lock import WakeLockBusy


def test_an_empty_store_has_no_session_enabled(tmp_path: Path) -> None:
    assert slot.read_slot(tmp_path) is None
    assert slot.slot_token(tmp_path) is None


def test_enabling_writes_the_contract_shape(tmp_path: Path) -> None:
    change = slot.enable(tmp_path, "aaaaaaaaaaaa", instance_id="inst", now_ms=1234)
    assert change.replaced is None
    assert change.active == slot.VoiceSlot("aaaaaaaaaaaa", 1234, "inst")
    on_disk = json.loads(slot.active_path(tmp_path).read_text())
    assert on_disk == {"session_id": "aaaaaaaaaaaa", "enabled_at": 1234, "instance_id": "inst"}
    # The wire object drops the process id: a renderer has no use for it.
    assert change.active is not None
    assert change.active.public() == {"session_id": "aaaaaaaaaaaa", "enabled_at": 1234}


def test_enabling_b_replaces_a_atomically(tmp_path: Path) -> None:
    slot.enable(tmp_path, "aaaaaaaaaaaa", now_ms=1)
    change = slot.enable(tmp_path, "bbbbbbbbbbbb", now_ms=2)
    assert change.replaced == "aaaaaaaaaaaa"
    assert slot.read_slot(tmp_path) == slot.VoiceSlot("bbbbbbbbbbbb", 2, "")


def test_re_enabling_the_holder_replaces_nobody_and_keeps_its_stamp(tmp_path: Path) -> None:
    slot.enable(tmp_path, "aaaaaaaaaaaa", now_ms=1)
    change = slot.enable(tmp_path, "aaaaaaaaaaaa", now_ms=99)
    assert change.replaced is None
    assert change.active is not None and change.active.enabled_at == 1


def test_disable_clears_only_for_the_holder(tmp_path: Path) -> None:
    """A stale window's ``disable A`` must not switch off B."""
    slot.enable(tmp_path, "bbbbbbbbbbbb", now_ms=2)
    stale = slot.disable(tmp_path, "aaaaaaaaaaaa")
    assert stale.active is not None and stale.active.session_id == "bbbbbbbbbbbb"
    cleared = slot.disable(tmp_path, "bbbbbbbbbbbb")
    assert cleared.active is None
    assert slot.read_slot(tmp_path) is None


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"session_id": 5, "enabled_at": 1}', '{"session_id": "x"}'],
)
def test_a_corrupt_slot_reads_as_empty_and_is_overwritable(tmp_path: Path, content: str) -> None:
    path = slot.active_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(content)
    assert slot.read_slot(tmp_path) is None
    change = slot.enable(tmp_path, "aaaaaaaaaaaa", now_ms=1)
    assert change.replaced is None
    assert slot.read_slot(tmp_path) is not None


def test_each_write_moves_the_token(tmp_path: Path) -> None:
    slot.enable(tmp_path, "aaaaaaaaaaaa", now_ms=1)
    first = slot.slot_token(tmp_path)
    slot.enable(tmp_path, "bbbbbbbbbbbb", now_ms=1)
    assert slot.slot_token(tmp_path) not in (None, first)


def test_a_held_lock_refuses_rather_than_writing_unlocked(tmp_path: Path) -> None:
    holder = slot._lock(tmp_path, 1.0)
    holder.acquire()
    try:
        with pytest.raises(WakeLockBusy) as refused:
            slot.enable(tmp_path, "aaaaaaaaaaaa", timeout_s=0.05)
        assert str(refused.value) == slot.BUSY_SENTENCE
    finally:
        holder.release()
    assert slot.read_slot(tmp_path) is None


def test_concurrent_enables_leave_exactly_one_holder(tmp_path: Path) -> None:
    """Many writers at once: the slot ends with one session, never a torn file."""
    ids = [f"{index:012x}" for index in range(8)]
    barrier = threading.Barrier(len(ids))
    replaced: list[str | None] = []

    def run(session_id: str) -> None:
        barrier.wait()
        replaced.append(slot.enable(tmp_path, session_id).replaced)

    threads = [threading.Thread(target=run, args=(sid,)) for sid in ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    final = slot.read_slot(tmp_path)
    assert final is not None and final.session_id in ids
    # Every enable after the first displaced exactly one holder: a chain, so
    # seven of eight answers name a predecessor and exactly one names nobody.
    assert replaced.count(None) == 1

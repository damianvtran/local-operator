"""The supervisor's ``held_at`` skip: a paused Aida is never engaged.

``wakes.store.is_held`` is the shared predicate; this file pins the two
decisions that would otherwise resurrect a paused session — the due scan that
decides which sessions get a runtime, and the keepalive that decides whether
the supervised process stays up for one.
"""

from __future__ import annotations

import time

from local_operator.wakes import supervisor
from local_operator.wakes.store import is_held

NOW = int(time.time() * 1000)


def _entry(due_ms: int, **extra: int) -> dict:
    return {
        "session_id": "s" * 12,
        "cwd": "/tmp",
        "schedules": [{"id": "w1", "message": "x", "next_due_at": due_ms, "fired_count": 0}],
        **extra,
    }


def test_is_held_covers_both_markers() -> None:
    assert is_held({"stopped_at": 1}) is True
    assert is_held({"held_at": 1}) is True
    assert is_held({}) is False
    assert is_held(None) is False


def test_due_scan_skips_held_entries() -> None:
    index = {"a" * 12: _entry(NOW - 5_000, held_at=NOW - 1_000)}
    assert supervisor._due_sessions(index, NOW) == []


def test_due_scan_still_engages_unheld_entries() -> None:
    index = {"b" * 12: _entry(NOW - 5_000)}
    due = supervisor._due_sessions(index, NOW)
    assert [session_id for session_id, _cwd, _ms in due] == ["b" * 12]


def test_next_wake_ignores_held_entries() -> None:
    index = {
        "c" * 12: _entry(NOW + 60_000, held_at=NOW),
        "d" * 12: _entry(NOW + 120_000),
    }
    assert supervisor._next_wake_ms(index) == NOW + 120_000
    only_held = {"e" * 12: _entry(NOW + 60_000, held_at=NOW)}
    assert supervisor._next_wake_ms(only_held) is None


def test_retirement_reason_counts_held_as_dormant(tmp_path) -> None:
    from local_operator.wakes import store as wake_store

    sessions = tmp_path / "sessions"
    (sessions / ("f" * 12)).mkdir(parents=True)
    (sessions / ("f" * 12) / "transcript.jsonl").write_text("", encoding="utf-8")
    wake_store.write_entry(
        tmp_path,
        "f" * 12,
        cwd=str(sessions / ("f" * 12)),
        schedules=_entry(NOW + 60_000)["schedules"],
        preserve={"held_at": NOW},
    )
    reason = supervisor._retirement_reason(tmp_path)
    # "dormant" (a lever parked it), never "stale"/"ghost" — a pause is a
    # deliberate state and must not read as a problem to chase.
    assert "dormant" in reason
    assert "stale" not in reason and "ghost" not in reason

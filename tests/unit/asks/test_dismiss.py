"""``asks/dismiss.py`` — the COLD dismissal, the one ask op with no runtime.

Why this file exists (design ``docs/design/ask-nonblocking.md`` §2.2/§14, fork A).
The phone's dismiss used to reach a stopped conversation by ENGAGING it, which
boots a runtime, spends a turn and — before fork A — was the only thing that ever
delivered the deadline notice the dismissal then cancelled. The cold door writes
the same ``dismissed`` row a runtime writes, rewrites the derived index, and
DISARMS the deadline wake row so the supervisor cannot boot a runtime for a
dismissed ask. Four properties are worth pinning and none of them is visible from
a green queue suite:

* the refusal is the queue's own sentence, so a surface cannot tell which door
  answered it;
* the wake row goes away — the whole reason a dismissal can boot nothing;
* a session with a LIVE owner is refused rather than written behind (:data:`code`
  ``ask_owner_present``), because the owner republishes its in-memory wake list
  over an outside write and would never see the row;
* the happy path leaves the log and the index agreeing, and starts nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from local_operator.asks import render, store
from local_operator.asks.dismiss import dismiss_cold
from local_operator.harness.wake import WakeSchedule
from local_operator.wakes.store import read_entry as read_wake_entry

SESSION = "colddismiss01"
ASK_ID = "a-colddismiss"
NOW = 1_700_000_000_000


def _run(coro):
    return asyncio.run(coro)


def _session(root: Path, *, queued_at: int = NOW - 3_600_000, timeout_s: int = 120) -> Path:
    """A session directory with ONE queued ask whose deadline has passed."""
    directory = store.session_dir(root, SESSION)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    store.append_event(
        directory,
        {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_QUEUED,
            "ask_id": ASK_ID,
            "at": queued_at,
            "created_at": queued_at,
            "expires_at": queued_at + timeout_s * 1000,
            "timeout_s": timeout_s,
            "urgent": False,
            "tool_call_id": "",
            "questions": [
                {
                    "id": "q1",
                    "question": "ship it?",
                    "options": [{"label": "yes", "description": ""}],
                    "multi": False,
                    "secret": False,
                    "persist": False,
                }
            ],
        },
    )
    return directory


def _wake_row(wake_id: str, **extra: Any) -> dict[str, Any]:
    return WakeSchedule(
        id=wake_id,
        message=f"message {wake_id}",
        next_due_at=NOW + 60_000,
        created_at=NOW,
        **extra,
    ).model_dump()


def _seed_wakes(directory: Path, rows: list[dict[str, Any]]) -> None:
    """The transcript's own ``wake_schedules`` entry — the arm module's base."""
    (directory / "transcript.jsonl").write_text(
        json.dumps(
            {
                "id": "wake-entry-1",
                "ts": 2.0,
                "type": "custom",
                "payload": {"custom_type": "wake_schedules", "details": {"schedules": rows}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _wake_ids(directory: Path) -> list[str]:
    """The ids in the transcript's LATEST ``wake_schedules`` row."""
    latest: list[str] = []
    for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        payload = entry.get("payload") or {}
        if payload.get("custom_type") == "wake_schedules":
            latest = [str(row.get("id")) for row in payload["details"]["schedules"]]
    return latest


def _kinds(directory: Path) -> list[str]:
    return [str(event.get("kind")) for event in store.read_events(directory)]


def test_a_cold_dismiss_refuses_an_open_ask_with_the_queues_own_words(tmp_path: Path):
    """A dismissal of a still-open ask is refused in the SAME sentence the queue
    uses, so the phone's card reads identically whichever door answered it."""
    directory = _session(tmp_path, queued_at=NOW, timeout_s=3600)
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {
        "ok": False,
        "error": "only a timed-out ask can be dismissed; it is still open.",
    }
    assert _kinds(directory) == [store.EVENT_QUEUED]


def test_a_cold_dismiss_refuses_an_ask_this_session_never_queued(tmp_path: Path):
    """An unknown id is its own answer, not an "already declined" that would be a
    lie about an id this process has never seen (:func:`render.refusal_copy`)."""
    directory = _session(tmp_path)
    outcome = _run(dismiss_cold(tmp_path, SESSION, "a-nowhere", by="phone", now_ms=NOW))
    assert outcome == {"ok": False, "error": render.refusal_copy(None)}
    assert _kinds(directory) == [store.EVENT_QUEUED]


def test_a_cold_dismiss_drops_the_deadline_wake_row(tmp_path: Path):
    """THE WAKE ROW IS WHY THIS CAN BOOT NOTHING.

    ``ask-timeout-<id>`` is what makes a deadline survive a runtime that does not
    exist: the supervisor engages a runtime for it. A dismissed ask has nothing
    left to deliver by fire (fork A delivers the notice at the next real reopen),
    so the row must go with the dismissal — leaving it would boot the very
    runtime the op exists to avoid, for a notice the row's owner has put away.
    """
    directory = _session(tmp_path)
    _seed_wakes(
        directory,
        [
            _wake_row(store.timeout_row_id(ASK_ID), kind="ask_timeout", hidden=True),
            _wake_row("w-user"),
        ],
    )
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}
    assert _wake_ids(directory) == ["w-user"]
    assert read_wake_entry(tmp_path, SESSION) is not None


def test_a_cold_dismiss_refuses_behind_a_live_owner(tmp_path: Path):
    """A live runtime owns the log AND its wake list, and republishes that list
    wholesale on its next persist — so an outside write would be lost, or would
    deliver the notice without the turn it owes. The relay falls back to the dial
    on THIS code alone, and nothing is written before it answers."""
    directory = _session(tmp_path)
    (directory / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": False, "code": "ask_owner_present"}
    assert _kinds(directory) == [store.EVENT_QUEUED]


def test_a_cold_dismiss_records_the_event_and_the_index_and_starts_nothing(tmp_path: Path):
    """The happy path: the log is the truth, the index is rewritten from it, and
    the fold on the way out says the notice is still owed (the next reconcile
    delivers it). No runtime is started — nothing here can start one."""
    directory = _session(tmp_path)
    store.write_entry(
        tmp_path,
        SESSION,
        cwd="/tmp/colddismiss",
        asks=[store.pending_row(store.fold(store.read_events(directory), NOW)[0])],
    )

    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}

    events = store.read_events(directory)
    assert [event["kind"] for event in events] == [store.EVENT_QUEUED, store.EVENT_DISMISSED]
    dismissed = events[-1]
    assert dismissed["at"] == NOW
    assert dismissed["by"] == {"surface": "phone"}

    entry = store.read_entry(tmp_path, SESSION)
    assert entry is not None
    (row,) = entry["asks"]
    assert row["ask_id"] == ASK_ID
    assert row["status"] == store.STATUS_DISMISSED

    # The fold's own half of the contract: dismissed, and the notice is still owed
    # to the model (the queue's reconcile is what delivers it).
    (record,) = store.fold(events, NOW + 1)
    assert record["status"] == store.STATUS_DISMISSED
    assert record["notice_owed"] is True
    # Nothing claimed the session: this op boots nothing, and no runtime had it.
    from local_operator.resume import live_runtime_pid

    assert live_runtime_pid(tmp_path, SESSION) is None


def test_a_cold_dismiss_injects_nothing_of_its_own(tmp_path: Path):
    """The op writes ONE event and one index row: no TRANSCRIPT row is appended
    for it, so a caller that reads the transcript before and after sees no new
    entry — the notice (if owed) is the next reconcile's business, not this
    door's.

    Seeded against the REAL clock (``now_ms`` omitted) because that is the arm the
    relay and the TUI call: the default has to fold an ask whose deadline has
    passed exactly as an explicit ``now_ms`` does.
    """
    real = store.now_ms()
    directory = _session(tmp_path, queued_at=real - 3_600_000, timeout_s=120)
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone"))
    assert outcome == {"ok": True}
    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == before

"""``wakes/arm.py`` — the one writer for a session nobody owns.

The three defects these tests pin were all live in ``lop wake create`` before
the helper existed, and each was silent:

- the CLI built ``id=f"w{len(existing) + 1}"``, so a 17th wake was written as
  ``w17`` past a 16-schedule cap that only the in-session tool enforced;
- the same line reissued a LIVE id after a cancel (``[w1, w2, w3]`` minus
  ``w2`` is three long, so the next arm called itself ``w3`` again — the same
  handle as an existing row);
- and the index write passed no ``preserve``, so arming a wake on a session the
  user had stopped un-parked it (dropping ``stopped_at``) and wiped the
  ``last_fired_at``/``last_attempt_at`` stamps the lateness report reads.

Everything else here is the contract the helper has to keep to be the writer
the desktop route and the CLI both call: transcript first, index second, install
hook third, and a post-write check that nobody appended on top of us.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.wake import MAX_WAKE_SCHEDULES, WakeSchedule
from local_operator.wakes.arm import (
    STATUS_CONFLICT,
    STATUS_SESSION_NOT_FOUND,
    STATUS_WAKE_NOT_FOUND,
    WakeWriteError,
    arm_wake,
    cancel_wake,
    edit_wake,
)
from local_operator.wakes.store import entry_path, read_entry

MINUTE = 60_000


def _row(wake_id: str, *, due: int = 0, **extra: Any) -> dict[str, Any]:
    return WakeSchedule(
        id=wake_id,
        message=f"message {wake_id}",
        next_due_at=due or int(time.time() * 1000) + 60 * MINUTE,
        created_at=1_700_000_000_000,
        **extra,
    ).model_dump()


def _session(root: Path, session_id: str, rows: list[dict[str, Any]] | None = None) -> Path:
    """A real session directory and transcript, optionally carrying wakes.

    A file rather than an object: this helper's whole job is external writing,
    and the transcript is the only thing it is allowed to treat as truth.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": "m1",
                "ts": 1.0,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
            }
        )
    ]
    if rows is not None:
        lines.append(
            json.dumps(
                {
                    "id": "wake-entry-1",
                    "ts": 2.0,
                    "type": "custom",
                    "payload": {
                        "custom_type": "wake_schedules",
                        "details": {"schedules": rows},
                    },
                }
            )
        )
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def _index(root: Path, session_id: str, rows: list[dict[str, Any]], **extra: object) -> None:
    """The derived entry, written directly.

    Seeded rather than built through the module under test: these tests are
    about what an external write does to an entry that already exists (a
    stopped session's ``stopped_at``, a lateness stamp, a cwd), and a helper
    that produced it by the same path would be asserting the writer against
    itself.
    """
    entry_path(root, session_id).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema": 1,
        "session_id": session_id,
        "cwd": "/work/here",
        "updated_at": 1,
        "schedules": rows,
    }
    entry.update(extra)
    entry_path(root, session_id).write_text(json.dumps(entry), encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """An isolated store. ``arm_wake`` is given the root explicitly rather than
    reading ``config_dir()``, but the supervisor install hook does read it, so
    the env var points at the same place and no real home is touched."""
    config = tmp_path / "cfg"
    (config / "sessions").mkdir(parents=True)
    return config


@pytest.mark.asyncio
async def test_arming_a_cold_session_creates_the_transcript_and_the_index_entry(
    root: Path,
) -> None:
    """The two facts that make a wake REAL: a transcript for the supervisor's
    ghost guard to find, and an index entry for it to read."""
    directory = root / "sessions" / "cold01"
    directory.mkdir(parents=True)

    outcome = await arm_wake(root, "cold01", {"message": "check the build", "in": "30m"})

    assert outcome.wake_id == "w1"
    assert outcome.index_written
    assert (directory / "transcript.jsonl").exists()
    entry = read_entry(root, "cold01")
    assert entry is not None
    assert [row["message"] for row in entry["schedules"]] == ["check the build"]
    assert entry["schedules"][0]["next_due_at"] == outcome.next_due_at


@pytest.mark.asyncio
async def test_the_sixteenth_schedule_can_be_armed_and_the_seventeenth_cannot(root: Path) -> None:
    """The cap is enforced by the shared validator, so the CLI, the desktop
    route and the agent's tool all refuse the same 17th arm with one sentence.

    Before the helper, the CLI bypassed it entirely and wrote ``w17``.
    """
    _session(root, "cap01", [_row(f"w{i}") for i in range(1, MAX_WAKE_SCHEDULES + 1)])

    with pytest.raises(WakeWriteError) as refused:
        await arm_wake(root, "cap01", {"message": "one too many", "in": "30m"})

    assert str(MAX_WAKE_SCHEDULES) in str(refused.value)
    assert refused.value.status == STATUS_CONFLICT
    # Nothing was written: the transcript still holds exactly the 16 it started
    # with, and nothing called itself ``w17``.
    rows = _rows_from_transcript(root / "sessions" / "cap01")
    assert len(rows) == MAX_WAKE_SCHEDULES
    assert "w17" not in {row["id"] for row in rows}


@pytest.mark.asyncio
async def test_a_cancelled_id_is_reissued_and_never_duplicated(root: Path) -> None:
    """A wake id is a per-session handle, so the first FREE slot is the one to
    hand out. ``w{len + 1}`` reissued a live id here (``w3`` twice)."""
    directory = _session(root, "reuse01", [_row("w1"), _row("w2"), _row("w3")])

    await cancel_wake(root, "reuse01", "w2")
    outcome = await arm_wake(root, "reuse01", {"message": "back", "in": "30m"})

    assert outcome.wake_id == "w2"
    rows = _rows_from_transcript(directory)
    assert [row["id"] for row in rows] == ["w1", "w3", "w2"]
    assert len({row["id"] for row in rows}) == 3
    entry = read_entry(root, "reuse01")
    assert entry is not None
    assert len(entry["schedules"]) == 3


@pytest.mark.asyncio
async def test_arming_preserves_a_stopped_sessions_stop_and_its_lateness_stamps(
    root: Path,
) -> None:
    """``stopped_at`` is the user's kill switch; ``last_fired_at`` and
    ``last_attempt_at`` are the only record of how late a wake actually ran.

    Both live in the index and neither belongs to this writer, which is why the
    index write carries them through. The CLI dropped all three, so scheduling
    a wake on a stopped session quietly restarted it.
    """
    _session(root, "stopped01", [_row("w1")])
    _index(
        root,
        "stopped01",
        [_row("w1")],
        stopped_at=1234,
        last_fired_at=5678,
        last_attempt_at=91011,
    )

    await arm_wake(root, "stopped01", {"message": "another", "in": "30m"})

    after = read_entry(root, "stopped01")
    assert after is not None
    assert after["stopped_at"] == 1234
    assert after["last_fired_at"] == 5678
    assert after["last_attempt_at"] == 91011
    assert len(after["schedules"]) == 2


@pytest.mark.asyncio
async def test_the_existing_due_time_is_preserved_when_one_is_set(root: Path) -> None:
    """An arm must not move a wake that is already scheduled, and the cwd the
    supervisor will use to start a runtime belongs to the entry, not to this
    call."""
    row = _row("w1")
    _session(root, "cwd01", [row])
    _index(root, "cwd01", [row], cwd="/somewhere/else")
    before = read_entry(root, "cwd01")
    assert before is not None

    await arm_wake(root, "cwd01", {"message": "another", "in": "30m"})

    after = read_entry(root, "cwd01")
    assert after is not None
    assert after["cwd"] == "/somewhere/else"
    assert after["schedules"][0]["next_due_at"] == before["schedules"][0]["next_due_at"]


@pytest.mark.asyncio
async def test_cancelling_the_last_wake_removes_the_index_entry(root: Path) -> None:
    """An empty list means "remove the file" rather than "write an empty one",
    which is also what releases the cleanup reap guard the session held."""
    _session(root, "last01", [_row("w1")])

    outcome = await cancel_wake(root, "last01", "w1")

    assert outcome.schedules == []
    assert outcome.index_written
    assert outcome.index_path == ""
    assert not entry_path(root, "last01").exists()


@pytest.mark.asyncio
async def test_cancel_and_edit_refuse_an_unknown_id_without_writing(root: Path) -> None:
    directory = _session(root, "unknown01", [_row("w1")])
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")

    with pytest.raises(WakeWriteError) as cancel_refusal:
        await cancel_wake(root, "unknown01", "w9")
    with pytest.raises(WakeWriteError) as edit_refusal:
        await edit_wake(root, "unknown01", "w9", {"message": "nope"})

    assert cancel_refusal.value.status == STATUS_WAKE_NOT_FOUND
    assert edit_refusal.value.status == STATUS_WAKE_NOT_FOUND
    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_an_unknown_session_is_refused_rather_than_created(root: Path) -> None:
    """A wake keyed on a session with no transcript is one the supervisor would
    faithfully fire into nothing, so the arm refuses instead of creating one."""
    with pytest.raises(WakeWriteError) as refused:
        await arm_wake(root, "nosuchsession", {"message": "hi", "in": "30m"})

    assert refused.value.status == STATUS_SESSION_NOT_FOUND
    assert not (root / "sessions" / "nosuchsession").exists()


@pytest.mark.asyncio
async def test_an_unreadable_schedule_list_is_refused_rather_than_treated_as_empty(
    root: Path,
) -> None:
    """A corrupt snapshot must not become the new base. Appending over it would
    write "no wakes" back, cancelling reminders the user still has."""
    directory = _session(root, "corrupt01")
    (directory / "transcript.jsonl").write_text(
        json.dumps(
            {
                "id": "wake-entry-1",
                "ts": 2.0,
                "type": "custom",
                "payload": {
                    "custom_type": "wake_schedules",
                    "details": {"schedules": "not a list"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WakeWriteError) as refused:
        await arm_wake(root, "corrupt01", {"message": "hi", "in": "30m"})

    assert refused.value.status == STATUS_CONFLICT


@pytest.mark.asyncio
async def test_an_edit_keeps_the_id_and_the_due_time_when_only_the_message_moves(
    root: Path,
) -> None:
    """The row's identity and its anchor are what a reword must not disturb —
    including when the row is ALREADY OVERDUE, which is exactly the row a user
    most often wants to fix (the create path's past-time guard would refuse it)."""
    due = int(time.time() * 1000) - 3_600_000
    _session(root, "edit01", [_row("w2", due=due)])

    outcome = await edit_wake(root, "edit01", "w2", {"message": "reworded"})

    assert outcome.wake_id == "w2"
    assert outcome.next_due_at == due
    rows = _rows_from_transcript(root / "sessions" / "edit01")
    assert rows[0]["id"] == "w2"
    assert rows[0]["message"] == "reworded"
    assert rows[0]["next_due_at"] == due


@pytest.mark.asyncio
async def test_an_edit_moves_the_time_and_bounds_and_keeps_the_history(root: Path) -> None:
    """``in``/``every``/``limit`` on an edit mean what they mean on a create;
    ``fired_count`` and ``created_at`` ride along because this is the same row,
    not a replacement."""
    _session(root, "edit02", [_row("w1")])

    outcome = await edit_wake(
        root, "edit02", "w1", {"message": "nightly", "in": "10m", "every": "1h", "limit": 4}
    )

    row = _rows_from_transcript(root / "sessions" / "edit02")[0]
    assert row["id"] == "w1"
    assert row["every_ms"] == 3_600_000
    assert row["limit"] == 4
    assert row["created_at"] == 1_700_000_000_000
    assert outcome.next_due_at == row["next_due_at"]


@pytest.mark.asyncio
async def test_an_edit_that_drops_the_repeat_also_drops_its_bound(root: Path) -> None:
    """``every`` sent as null turns the row into a one-shot, and a bound with no
    repeat is refused by the shared validator rather than silently ignored."""
    _session(root, "edit03", [_row("w1", every_ms=MINUTE * 60, limit=3)])

    with pytest.raises(WakeWriteError) as refused:
        await edit_wake(root, "edit03", "w1", {"every": None})

    assert "bound a repeat" in str(refused.value)

    # Dropping the bound as well is accepted, and the row stops repeating.
    await edit_wake(root, "edit03", "w1", {"every": None, "limit": None})
    assert _rows_from_transcript(root / "sessions" / "edit03")[0]["every_ms"] is None


@pytest.mark.asyncio
async def test_a_second_writer_landing_on_top_of_the_append_is_retried(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes CAN append to one transcript, and last writer wins. When
    somebody does, the arm re-reads the new base and applies the change again
    instead of reporting a success that erased it."""
    from local_operator.session import transcript as transcript_module

    _session(root, "race01", [_row("w1")])
    real = transcript_module.Transcript.append_custom
    calls = {"n": 0}

    async def racing(self, custom_type, details):  # type: ignore[no-untyped-def]
        entry = await real(self, custom_type, details)
        if calls["n"] == 0:
            calls["n"] += 1
            # A competing writer appends its own snapshot immediately after ours.
            await real(self, custom_type, {"schedules": [_row("w9")]})
        return entry

    monkeypatch.setattr(transcript_module.Transcript, "append_custom", racing)

    outcome = await arm_wake(root, "race01", {"message": "second", "in": "30m"})

    # The retry re-read the COMPETING snapshot as its base, so the new row took
    # the first free slot in THAT list (``w9`` held ``w9``, so ``w1`` is free)
    # and the final snapshot holds both writers' work.
    assert outcome.wake_id == "w1"
    rows = _rows_from_transcript(root / "sessions" / "race01")
    assert [row["message"] for row in rows] == ["message w9", "second"]


@pytest.mark.asyncio
async def test_a_store_that_never_settles_refuses_with_a_conflict(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retried ONCE, then refused. A write that reports success over somebody
    else's snapshot is how a user's cancel silently disappears."""
    from local_operator.session import transcript as transcript_module

    _session(root, "race02", [_row("w1")])
    real = transcript_module.Transcript.append_custom

    async def always_overtaken(self, custom_type, details):  # type: ignore[no-untyped-def]
        entry = await real(self, custom_type, details)
        await real(self, custom_type, {"schedules": [_row("w9")]})
        return entry

    monkeypatch.setattr(transcript_module.Transcript, "append_custom", always_overtaken)

    with pytest.raises(WakeWriteError) as refused:
        await arm_wake(root, "race02", {"message": "second", "in": "30m"})

    assert refused.value.status == STATUS_CONFLICT
    assert "retry" in str(refused.value).lower()


def _rows_from_transcript(directory: Path) -> list[dict[str, Any]]:
    """The LATEST ``wake_schedules`` snapshot on disk, parsed from the file.

    Read here rather than through ``Session._load_wake_schedules`` so the
    assertion is about bytes: what this module wrote is what every other reader
    (the supervisor, the picker, the next open) will see.
    """
    latest: list[dict[str, Any]] | None = None
    for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("type") != "custom":
            continue
        payload = entry.get("payload") or {}
        if payload.get("custom_type") != "wake_schedules":
            continue
        latest = list((payload.get("details") or {}).get("schedules") or [])
    assert latest is not None, "no wake_schedules entry was written"
    return latest


# ---------------------------------------------------------------------------
# Writing from OUTSIDE, correctly: serialised, and never behind an owner
# ---------------------------------------------------------------------------
#
# Both of these were found by driving a real server, not by reading the code:
# five concurrent arms left SIX rows and five 200s on one cold session (one
# request's row written at two ids, and a 409 for a write that had landed), and
# an arm whose session gained a runtime mid-request answered
# `200 {index_written: true}` for a row the owner's next persist then deleted
# from the transcript and the index while the supervisor skipped the session.

import asyncio  # noqa: E402 — grouped with the tests that need it
import functools  # noqa: E402
import threading  # noqa: E402

from local_operator.wakes.lock import WakeWriteLock  # noqa: E402


def _arms_in_parallel(
    root: Path, session_id: str, count: int
) -> tuple[list[Any], list[BaseException]]:
    """``count`` arms of one session at once, each on its own thread and loop.

    Threads rather than coroutines on purpose: the writers that race in the
    field are separate PROCESSES (the CLI, the desktop server, a second app),
    and a lock that only serialised coroutines would leave that race intact.
    """
    outcomes: list[Any] = []
    failures: list[BaseException] = []
    start = threading.Barrier(count)

    def arm(index: int) -> None:
        try:
            start.wait(timeout=30)
            outcomes.append(
                asyncio.run(
                    arm_wake(root, session_id, {"message": f"w{index}", "in": f"{index + 1}0m"})
                )
            )
        except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
            failures.append(exc)

    threads = [threading.Thread(target=arm, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return outcomes, failures


@pytest.mark.asyncio
async def test_concurrent_arms_leave_one_row_each_and_no_conflict(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N arms of one cold session ⇒ N rows, N distinct ids, N successes.

    The window is WIDENED on purpose so this is a deterministic catch rather
    than a probabilistic one: ``_read_rows`` is the step that makes the race,
    and slowing it is the same magnifier a 103 MB transcript provided when this
    was found. Without the per-session lock every thread reads the same base and
    the last append wins, so the assertion below fails at 1 row instead of 5.
    """
    import local_operator.wakes.arm as arm_module

    _session(root, "concurrent01")
    original_read = arm_module._read_rows

    def slow_read(session_dir: Path) -> list[WakeSchedule]:
        time.sleep(0.02)
        return original_read(session_dir)

    monkeypatch.setattr(arm_module, "_read_rows", slow_read)

    outcomes, failures = _arms_in_parallel(root, "concurrent01", 5)

    assert failures == []
    assert len(outcomes) == 5
    rows = _rows_from_transcript(root / "sessions" / "concurrent01")
    assert len(rows) == 5, "one row per request, and no dropped update"
    assert len({row["id"] for row in rows}) == 5, "no id handed out twice"
    assert sorted(row["message"] for row in rows) == [f"w{i}" for i in range(5)]
    # Every id the callers were told about is on disk exactly once: a returned
    # id that no row carries is the "written twice, returned once" shape.
    assert sorted(outcome.wake_id for outcome in outcomes) == sorted(row["id"] for row in rows)

    entry = read_entry(root, "concurrent01")
    assert entry is not None
    assert len(entry["schedules"]) == 5


@pytest.mark.asyncio
async def test_a_live_owner_that_appears_at_the_append_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route's owner check is a moment old by the time the writer runs. When
    a runtime has appeared in between, the row would be deleted by that owner's
    next persist — so the writer refuses instead of answering 200."""
    import local_operator.wakes.supervisor as supervisor

    session_dir = _session(root, "owned01", [_row("w1")])

    async def live(config_dir: Path, session_id: str) -> bool:
        return True

    monkeypatch.setattr(supervisor, "_has_live_runtime", live)

    with pytest.raises(WakeWriteError) as refused:
        await arm_wake(root, "owned01", {"message": "must not land", "in": "30m"})

    assert refused.value.status == 503
    assert refused.value.code == "wake_owner_present"
    assert "Retry in a moment" in str(refused.value)
    # NOTHING was written: the live row the session already had is untouched,
    # which is the point — the refusal happens before the append.
    assert [row["id"] for row in _rows_from_transcript(session_dir)] == ["w1"]


@pytest.mark.asyncio
async def test_a_wedged_owner_is_refused_by_the_writer_too(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alive, heartbeat stale, lease held: the supervisor cannot engage it and
    its in-memory list is still loaded, so a write here is just as doomed as one
    behind a healthy owner — and says the same sentence the route says."""
    import local_operator.wakes.supervisor as supervisor

    session_dir = _session(root, "wedged01", [_row("w1")])
    monkeypatch.setattr(supervisor, "wedged_runtime", lambda *args, **kwargs: (4321, 99.5))

    with pytest.raises(WakeWriteError) as refused:
        await arm_wake(root, "wedged01", {"message": "must not land", "in": "30m"})

    assert refused.value.status == 503
    assert refused.value.code == "wake_owner_wedged"
    assert [row["id"] for row in _rows_from_transcript(session_dir)] == ["w1"]


@pytest.mark.asyncio
async def test_a_contended_lock_refuses_rather_than_writing_unlocked(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contention is refused, never degraded. Running unlocked is precisely the
    duplicate row the lock exists to prevent, so the caller gets a retryable
    503 — the opposite of the group reaper's best-effort lock, which may run
    unlocked because its worst case is the behaviour it already had."""
    import local_operator.wakes.arm as arm_module

    session_dir = _session(root, "busy01", [_row("w1")])
    holder = WakeWriteLock(session_dir)
    holder.acquire()
    try:
        monkeypatch.setattr(
            arm_module, "WakeWriteLock", functools.partial(WakeWriteLock, timeout_s=0.05)
        )
        with pytest.raises(WakeWriteError) as refused:
            await arm_wake(root, "busy01", {"message": "queued behind a peer", "in": "30m"})
    finally:
        holder.release()

    assert refused.value.status == 503
    assert refused.value.code == "wake_write_busy"
    assert "Retry in a moment" in str(refused.value)
    assert [row["id"] for row in _rows_from_transcript(session_dir)] == ["w1"]

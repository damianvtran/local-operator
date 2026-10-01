"""``monitors/arm.py`` — the external arm and cancel writers for a session
nobody owns.

The properties the desktop routes and the CLI's ``lop monitor cancel`` rely on
are the ones pinned here, and each is a silent failure if it drifts:

- the base is the TRANSCRIPT, never the index (a stale base resurrects a
  cancelled watch or writes a live one away);
- the append REPLACES the list and re-emits ``next_seq`` unchanged (dropping
  the high-water mark would let a later arm reissue a cancelled id);
- the index write carries ``stopped_at`` through (cancelling one monitor on a
  stopped session must not un-park the rest);
- the post-write verification and the owner guard refuse rather than report a
  cancel that a non-party writer may have overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.monitors.arm import (
    STATUS_CONFLICT,
    STATUS_MONITOR_NOT_FOUND,
    STATUS_OWNER_BUSY,
    STATUS_SESSION_NOT_FOUND,
    MonitorWriteError,
    arm_monitor,
    cancel_monitor,
)
from local_operator.monitors.store import entry_path, read_entry

MINUTE = 60_000


def _row(monitor_id: str, **extra: Any) -> dict[str, Any]:
    row = {
        "id": monitor_id,
        "name": f"watch {monitor_id}",
        "tool": "bash",
        "arguments": {"command": "ls"},
        "every_ms": MINUTE,
        "created_at": 1_700_000_000_000,
    }
    row.update(extra)
    return row


def _session(root: Path, session_id: str, rows: list[dict[str, Any]] | None = None) -> Path:
    """A real session directory and transcript, optionally carrying monitors.

    A file rather than an object: this writer's whole job is external writing,
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
                    "id": "monitor-entry-1",
                    "ts": 2.0,
                    "type": "custom",
                    "payload": {
                        "custom_type": "monitor_schedules",
                        "details": {"monitors": rows, "next_seq": 3},
                    },
                }
            )
        )
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def _index(root: Path, session_id: str, rows: list[dict[str, Any]], **extra: object) -> None:
    """The derived entry, written directly (never through the writer under test)."""
    entry_path(root, session_id).parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "schema": 1,
        "session_id": session_id,
        "cwd": "/work/here",
        "updated_at": 1,
        "monitors": rows,
    }
    entry.update(extra)
    entry_path(root, session_id).write_text(json.dumps(entry), encoding="utf-8")


def _latest(root: Path, session_id: str) -> dict[str, Any]:
    from local_operator.session.transcript import read_latest_custom_entry

    entry = read_latest_custom_entry(root / "sessions" / session_id, "monitor_schedules")
    assert entry is not None
    return dict(entry.payload.get("details") or {})


@pytest.fixture
def root(tmp_path: Path) -> Path:
    config = tmp_path / "cfg"
    (config / "sessions").mkdir(parents=True)
    return config


@pytest.mark.asyncio
async def test_cancelling_one_monitor_rewrites_the_transcript_and_the_index(root: Path) -> None:
    _session(root, "surf01", [_row("m1"), _row("m2")])
    _index(root, "surf01", [_row("m1"), _row("m2")])

    outcome = await cancel_monitor(root, "surf01", "m1")

    assert outcome.monitor_id == "m1"
    assert outcome.name == "watch m1"
    assert outcome.remaining == 1
    assert outcome.index_written
    details = _latest(root, "surf01")
    assert [row["id"] for row in details["monitors"]] == ["m2"]
    # The high-water mark is re-emitted unchanged; dropping it here would let a
    # later arm reissue the cancelled ``m1``.
    assert details["next_seq"] == 3
    entry = read_entry(root, "surf01")
    assert entry is not None
    assert [row["id"] for row in entry["monitors"]] == ["m2"]
    assert not entry_path(root, "surf01").with_name("surf01.json").read_text().count("m1")


@pytest.mark.asyncio
async def test_cancelling_the_last_monitor_removes_the_index_entry(root: Path) -> None:
    """An empty list means "remove the file" rather than "write an empty one",
    which is also what releases the cleanup reap guard the session held."""
    _session(root, "last01", [_row("m1")])
    _index(root, "last01", [_row("m1")])

    outcome = await cancel_monitor(root, "last01", "m1")

    assert outcome.remaining == 0
    assert outcome.index_written
    assert outcome.index_path == ""
    assert not entry_path(root, "last01").exists()


@pytest.mark.asyncio
async def test_a_stopped_sessions_park_survives_the_cancel(root: Path) -> None:
    """``stopped_at`` is the user's kill switch: cancelling one watch on a
    stopped session must not un-park the ones that remain."""
    _session(root, "parked01", [_row("m1"), _row("m2")])
    _index(root, "parked01", [_row("m1"), _row("m2")], stopped_at=1234)

    await cancel_monitor(root, "parked01", "m1")

    after = read_entry(root, "parked01")
    assert after is not None
    assert after["stopped_at"] == 1234
    assert [row["id"] for row in after["monitors"]] == ["m2"]


@pytest.mark.asyncio
async def test_the_transcript_is_the_base_not_the_index(root: Path) -> None:
    """A stale index must not shape the new list: it lags, and the append
    REPLACES the snapshot, so rebasing on it would write a live watch away."""
    _session(root, "stale01", [_row("m1"), _row("m2")])
    _index(root, "stale01", [_row("m1")])  # the index has not seen m2 yet

    await cancel_monitor(root, "stale01", "m1")

    details = _latest(root, "stale01")
    assert [row["id"] for row in details["monitors"]] == ["m2"]
    entry = read_entry(root, "stale01")
    assert entry is not None
    assert [row["id"] for row in entry["monitors"]] == ["m2"]


@pytest.mark.asyncio
async def test_an_unknown_id_is_refused_without_writing(root: Path) -> None:
    directory = _session(root, "unknown01", [_row("m1")])
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")

    with pytest.raises(MonitorWriteError) as refused:
        await cancel_monitor(root, "unknown01", "m9")

    assert refused.value.status == STATUS_MONITOR_NOT_FOUND
    assert "m9" in str(refused.value)
    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_arming_glob_with_an_unknown_argument_is_refused_without_a_write(root: Path) -> None:
    """The external half of the ``glob({path: ...})`` report: the arm is the
    loud failure (§6.1), so it must refuse a call the tick would refuse rather
    than write a monitor that dies five times and disables itself.
    """
    _session(root, "shape01", [])

    with pytest.raises(MonitorWriteError) as refused:
        await arm_monitor(
            root,
            "shape01",
            {"tool": "glob", "arguments": {"pattern": "*.py", "path": "/tmp"}, "every": "60s"},
        )

    assert refused.value.status == STATUS_CONFLICT
    assert 'unknown argument(s) "path"' in str(refused.value)
    # Nothing was written: no transcript entry, no index entry.
    assert _latest(root, "shape01")["monitors"] == []
    assert read_entry(root, "shape01") is None

    # And the SAME call with the argument the tool declares arms cleanly.
    outcome = await arm_monitor(
        root, "shape01", {"tool": "glob", "arguments": {"pattern": "*.py"}, "every": "60s"}
    )
    # ``m3`` because the fixture's transcript carries ``next_seq: 3`` (the
    # high-water mark), not because anything was issued by the refusal above.
    assert outcome.monitor_id == "m3"


@pytest.mark.asyncio
async def test_an_unknown_session_is_refused(root: Path) -> None:
    """A monitor keyed on a session with no transcript would be written into
    nothing a resume could ever read."""
    with pytest.raises(MonitorWriteError) as refused:
        await cancel_monitor(root, "nosuch", "m1")
    assert refused.value.status == STATUS_SESSION_NOT_FOUND


@pytest.mark.asyncio
async def test_the_cancelled_monitors_state_files_are_removed(root: Path) -> None:
    from local_operator.monitors import state as monitor_state

    _session(root, "state01", [_row("m1"), _row("m2")])
    _index(root, "state01", [_row("m1"), _row("m2")])
    counters = monitor_state.counters_path(root, "state01", "m1")
    counters.parent.mkdir(parents=True, exist_ok=True)
    counters.write_text('{"schema":1,"monitor_id":"m1"}', encoding="utf-8")
    snapshot = monitor_state.snapshot_path(root, "state01", "m1")
    snapshot.write_text('{"schema":1,"monitor_id":"m1","snapshot":"x"}', encoding="utf-8")
    keep = monitor_state.counters_path(root, "state01", "m2")
    keep.write_text('{"schema":1,"monitor_id":"m2"}', encoding="utf-8")

    await cancel_monitor(root, "state01", "m1")

    assert not counters.exists()
    assert not snapshot.exists()
    assert keep.exists()  # a sibling's derived files are untouched


@pytest.mark.asyncio
async def test_cancelling_the_last_monitor_reclaims_its_state_directory(root: Path) -> None:
    """§D5: the files went, the container stayed, and 16 of them accumulated in
    the live store. A sibling monitor keeps the directory (asserted above).
    """
    from local_operator.monitors import state as monitor_state

    _session(root, "solo01", [_row("m1")])
    _index(root, "solo01", [_row("m1")])
    monitor_state.write_counters(root, "solo01", "m1", {"schema": 1, "monitor_id": "m1"})

    await cancel_monitor(root, "solo01", "m1")

    assert not monitor_state.state_dir(root, "solo01").exists()


@pytest.mark.asyncio
async def test_a_verification_mismatch_settles_when_the_monitor_is_gone(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebase that reads back a list without the monitor reports success:
    absence is the state the request asked for, and the newest snapshot (our
    own first append, or a peer that saw it) proves it."""
    from local_operator.monitors import arm as arm_mod

    _session(root, "conflict01", [_row("m1"), _row("m2")])
    _index(root, "conflict01", [_row("m1"), _row("m2")])
    monkeypatch.setattr(arm_mod, "_latest_entry_id", lambda _directory: "someone-else")

    outcome = await cancel_monitor(root, "conflict01", "m1")

    assert outcome.remaining == 1
    details = _latest(root, "conflict01")
    assert [row["id"] for row in details["monitors"]] == ["m2"]


@pytest.mark.asyncio
async def test_a_verification_mismatch_with_the_monitor_still_present_refuses(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the rebase still sees the row and a non-party writer keeps landing
    on top, the honest answer is a conflict, not a success report: the
    cancel's effect cannot be asserted."""
    from local_operator.monitors import arm as arm_mod
    from local_operator.monitors.spec import MonitorSpec

    _session(root, "conflict02", [_row("m1"), _row("m2")])
    _index(root, "conflict02", [_row("m1"), _row("m2")])
    specs = [MonitorSpec.model_validate(_row("m1")), MonitorSpec.model_validate(_row("m2"))]
    calls = {"n": 0}

    def wrong(_directory: Path) -> str:
        calls["n"] += 1
        return "someone-else"

    monkeypatch.setattr(arm_mod, "_read_rows", lambda _directory: (list(specs), 3))
    monkeypatch.setattr(arm_mod, "_latest_entry_id", wrong)

    with pytest.raises(MonitorWriteError) as refused:
        await cancel_monitor(root, "conflict02", "m1")

    assert refused.value.status == STATUS_CONFLICT
    assert calls["n"] == 2, "the rebase retry must actually run before the refusal"


@pytest.mark.asyncio
async def test_the_post_append_owner_guard_undoes_the_append(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal must leave nothing durable behind; the append the guard
    refuses gets rolled back before the sentence is raised."""
    from local_operator.monitors import arm as arm_mod

    _session(root, "guard01", [_row("m1"), _row("m2")])
    _index(root, "guard01", [_row("m1"), _row("m2")])
    calls = {"n": 0}

    async def fake_guard(config_dir: Path, session_id: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise MonitorWriteError("owner appeared", status=STATUS_OWNER_BUSY, code="x")

    monkeypatch.setattr(arm_mod, "_refuse_if_owned", fake_guard)

    with pytest.raises(MonitorWriteError):
        await cancel_monitor(root, "guard01", "m1")

    assert calls["n"] == 2
    details = _latest(root, "guard01")
    assert [row["id"] for row in details["monitors"]] == ["m1", "m2"], "undo must restore"


@pytest.mark.asyncio
async def test_a_pre_append_refusal_leaves_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.monitors import arm as arm_mod

    directory = _session(root, "guard02", [_row("m1")])
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")

    async def fake_guard(config_dir: Path, session_id: str) -> None:
        raise MonitorWriteError("no owners allowed", status=STATUS_OWNER_BUSY, code="x")

    monkeypatch.setattr(arm_mod, "_refuse_if_owned", fake_guard)

    with pytest.raises(MonitorWriteError):
        await cancel_monitor(root, "guard02", "m1")

    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_the_owner_guard_refuses_behind_a_live_runtime(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Appending behind a live owner would be undone by that owner's next
    persist, so the writer refuses with a sentence that says who owns the
    conversation."""
    from local_operator.mobile import attach_client
    from local_operator.wakes import supervisor

    _session(root, "owned01", [_row("m1")])
    monkeypatch.setattr(supervisor, "wedged_runtime", lambda *a, **k: None)
    monkeypatch.setattr(attach_client, "find_runtime_record", lambda *a, **k: (None, 4242))
    monkeypatch.setattr(attach_client, "dialable_record_exists", lambda *a, **k: True)

    with pytest.raises(MonitorWriteError) as refused:
        await cancel_monitor(root, "owned01", "m1")

    assert refused.value.status == STATUS_OWNER_BUSY
    assert "open in a running session" in str(refused.value)
    details = _latest(root, "owned01")
    assert [row["id"] for row in details["monitors"]] == ["m1"], "nothing was written"


# ---------------------------------------------------------------------------
# arm_monitor — the desktop arm route's writer
# ---------------------------------------------------------------------------


def _freeze_first_check(monkeypatch: pytest.MonkeyPatch, delay_ms: int = 2_000) -> None:
    """Pin the first-check jitter so ``next_due_at`` assertions are exact.

    The window is `MonitorScheduler._first_check_delay_ms`'s (uniform 1-3 s),
    and pinning it keeps these tests from asserting a range they would
    otherwise have to restate.
    """
    from local_operator.monitors import arm as arm_module

    monkeypatch.setattr(arm_module.random, "uniform", lambda _a, _b: float(delay_ms))


def _append_count(root: Path, session_id: str) -> int:
    text = (root / "sessions" / session_id / "transcript.jsonl").read_text(encoding="utf-8")
    count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line).get("payload") or {}
        if payload.get("custom_type") == "monitor_schedules":
            count += 1
    return count


@pytest.mark.asyncio
async def test_arming_a_cold_session_writes_the_transcript_and_the_index(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm path for an existing conversation with nobody home: transcript
    first, index second, and the index row carries the receipt's due instant
    and the fresh counters the session would rebuild anyway."""
    _session(root, "new01")
    _freeze_first_check(monkeypatch)
    now = 1_700_000_000_000

    outcome = await arm_monitor(
        root,
        "new01",
        {"name": "watch it", "tool": "bash", "arguments": {"command": "ls"}, "every": "60s"},
        now_ms=now,
    )

    assert outcome.monitor_id == "m1"
    assert outcome.name == "watch it"
    assert outcome.remaining == 1
    assert outcome.next_due_at == now + 2_000
    assert outcome.index_written is True
    assert outcome.already_armed is False and outcome.reactivated is False
    details = _latest(root, "new01")
    assert [row["id"] for row in details["monitors"]] == ["m1"]
    assert details["next_seq"] == 2
    entry = read_entry(root, "new01")
    assert entry is not None
    assert entry["monitors"][0]["id"] == "m1"
    assert entry["monitors"][0]["next_due_at"] == now + 2_000
    assert entry["monitors"][0]["disabled"] is False


@pytest.mark.asyncio
async def test_an_identical_spec_is_answered_without_appending(root: Path) -> None:
    """The dedupe identity is also the retry's idempotency key: a second arm of
    the same call — whatever its name or interval — is the EXISTING row, and
    the transcript's write log does not move."""
    _session(root, "dup01")
    first = await arm_monitor(
        root, "dup01", {"tool": "bash", "arguments": {"command": "ls"}, "every": "60s"}
    )
    appends = _append_count(root, "dup01")

    second = await arm_monitor(
        root,
        "dup01",
        {"name": "renamed", "tool": "bash", "arguments": {"command": "ls"}, "every": "5m"},
    )

    assert second.monitor_id == first.monitor_id == "m1"
    assert second.already_armed is True
    assert second.reactivated is False
    # The existing row is answered AS IS: the second request's name and
    # interval describe the arm it asked for, not an edit of the row it hit
    # (a different interval is cancel + create, §11.4).
    assert second.name == first.name == "bash"
    assert _append_count(root, "dup01") == appends, "a duplicate must not append"
    details = _latest(root, "dup01")
    assert len(details["monitors"]) == 1


@pytest.mark.asyncio
async def test_a_disabled_spec_is_reactivated_rather_than_duplicated(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§11.3's "re-arm to reactivate": the failing watch keeps its id and its
    snapshot baseline, and only its counters move."""
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors.scheduler import fresh_counters

    _session(root, "rev01", [_row("m1")])
    dead = fresh_counters("m1", None)
    dead.update({"disabled": True, "disabled_reason": "5 failed", "consecutive_failures": 5})
    monitor_state.write_counters(root, "rev01", "m1", dead)
    _freeze_first_check(monkeypatch)
    now = 1_700_000_000_000

    outcome = await arm_monitor(
        root, "rev01", {"tool": "bash", "arguments": {"command": "ls"}, "every": "60s"}, now_ms=now
    )

    assert outcome.monitor_id == "m1"
    assert outcome.reactivated is True
    assert outcome.already_armed is False
    assert outcome.next_due_at == now + 2_000
    assert _append_count(root, "rev01") == 1, "the list did not move"
    reset = monitor_state.read_counters(root, "rev01", "m1")
    assert reset is not None
    assert reset["disabled"] is False
    assert reset["disabled_reason"] == ""
    assert reset["next_due_at"] == now + 2_000


@pytest.mark.asyncio
async def test_the_read_only_gate_refuses_a_write_target_and_judges_bash(root: Path) -> None:
    """The arm-time gate (`external_monitor_verdict`, §6.7): a write-tier tool
    and a destructive bash line fail LOUDLY before anything is written, while a
    provably read-only command arms."""
    directory = _session(root, "ro01")

    with pytest.raises(MonitorWriteError) as refused:
        await arm_monitor(
            root, "ro01", {"tool": "write", "arguments": {"path": "x"}, "every": "60s"}
        )
    assert refused.value.code == "monitor_refused"
    assert refused.value.status == STATUS_CONFLICT
    assert '"write"' in str(refused.value)

    with pytest.raises(MonitorWriteError) as destructive:
        await arm_monitor(
            root, "ro01", {"tool": "bash", "arguments": {"command": "rm -rf /"}, "every": "60s"}
        )
    assert destructive.value.code == "monitor_refused"
    assert "read-only" in str(destructive.value)

    log = (directory / "transcript.jsonl").read_text(encoding="utf-8")
    assert log.count("monitor_schedules") == 0

    outcome = await arm_monitor(
        root, "ro01", {"tool": "bash", "arguments": {"command": "ls -la"}, "every": "60s"}
    )
    assert outcome.monitor_id == "m1"


@pytest.mark.asyncio
async def test_the_cap_refuses_the_ninth_arm_with_its_own_sentence(root: Path) -> None:
    """§11.4's cap is a rejection naming the remedy, never a silent drop — and
    the transcript is untouched by the refused request."""
    rows = [_row(f"m{i}", arguments={"command": f"ls {i}"}) for i in range(1, 9)]
    _session(root, "cap01", rows)

    with pytest.raises(MonitorWriteError) as refused:
        await arm_monitor(
            root, "cap01", {"tool": "bash", "arguments": {"command": "ls fresh"}, "every": "60s"}
        )

    assert refused.value.status == STATUS_CONFLICT
    assert "monitor limit reached" in str(refused.value)
    details = _latest(root, "cap01")
    assert len(details["monitors"]) == 8


@pytest.mark.asyncio
async def test_an_unknown_session_is_refused_rather_than_invented(root: Path) -> None:
    with pytest.raises(MonitorWriteError) as refused:
        await arm_monitor(root, "nosuch", {"tool": "bash", "arguments": {"command": "ls"}})

    assert refused.value.status == STATUS_SESSION_NOT_FOUND
    assert refused.value.code == "session_not_found"


@pytest.mark.asyncio
async def test_the_owner_guard_refuses_an_answering_runtime_before_the_append(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm path's pre-append guard, pinned separately from cancel's: an arm
    behind a live owner would be deleted by that owner's next persist, so it is
    refused with the same sentence and leaves the transcript untouched."""
    from local_operator.mobile import attach_client
    from local_operator.wakes import supervisor

    directory = _session(root, "armown01", [_row("m1")])
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")
    monkeypatch.setattr(supervisor, "wedged_runtime", lambda *a, **k: None)
    monkeypatch.setattr(attach_client, "find_runtime_record", lambda *a, **k: (None, 4242))
    monkeypatch.setattr(attach_client, "dialable_record_exists", lambda *a, **k: True)

    with pytest.raises(MonitorWriteError) as refused:
        await arm_monitor(root, "armown01", {"tool": "bash", "arguments": {"command": "ls other"}})

    assert refused.value.status == STATUS_OWNER_BUSY
    assert "open in a running session" in str(refused.value)
    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_the_storm_guard_refuses_the_third_identical_name(root: Path) -> None:
    """§11.4's storm guard at the writer level: two rows named "watch" already
    stand, so a third same-named arm is refused with the scheduler's sentence
    verbatim and appends nothing — and it is the storm guard that catches it,
    not the cap, which needs eight."""
    directory = _session(
        root,
        "storm01",
        [_row("m1", name="watch"), _row("m2", name="watch", arguments={"command": "ls m2"})],
    )
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")

    with pytest.raises(MonitorWriteError) as refused:
        await arm_monitor(
            root,
            "storm01",
            {"name": "watch", "tool": "bash", "arguments": {"command": "ls other"}},
        )

    assert refused.value.status == STATUS_CONFLICT
    assert refused.value.code == "monitor_refused"
    assert str(refused.value) == (
        "three monitors named 'watch' is a storm — cancel one or use a distinct name."
    )
    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == before
    details = _latest(root, "storm01")
    assert [row["id"] for row in details["monitors"]] == ["m1", "m2"]

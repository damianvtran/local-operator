"""Relay session-load performance: incremental fold cache, summaries cache,
seen store, and frame-cap tiering.

These pin the behaviour the relay-perf change is built on, against the same
store primitives (``Transcript``) the daemon reads, so the cached paths stay
byte-identical to the full-reparse paths they replaced.
"""

from __future__ import annotations

import asyncio
import json
import string

import pytest
from starlette.testclient import TestClient

from local_operator.harness.types import Message
from local_operator.mobile.daemon import MobileDaemon, SessionTable, build_app
from local_operator.mobile.durable import CustomSnapshotCache, DurableFoldCache
from local_operator.mobile.projection import (
    FRAME_CAP_RESULT_CHARS,
    PROJECTION_FRAME_SOFT_CAP_BYTES,
    cap_projection_frame,
)
from local_operator.mobile.seen import MAX_SEEN_ENTRIES, SEEN_STORE_NAME, SeenStore
from local_operator.mobile.types import SessionProjection, SubagentRow
from local_operator.session.transcript import Transcript

# ---------------------------------------------------------------------------
# Incremental durable fold cache
# ---------------------------------------------------------------------------


def _write_turns(directory, n: int, start: int = 0) -> list[str]:
    """Append n user/assistant turns; return the message ids in order."""
    transcript = Transcript(directory)
    ids: list[str] = []
    for turn in range(start, start + n):
        user = Message.user(f"user {turn}", id=f"u-{turn:03d}")
        assistant = Message.assistant(f"answer {turn}", id=f"a-{turn:03d}")
        asyncio.run(transcript.append_message(user))
        asyncio.run(transcript.append_message(assistant))
        ids.extend([user.id, assistant.id])
    return ids


async def _write_turns_async(directory, n: int, start: int = 0) -> list[str]:
    """Async variant for tests already inside a running event loop."""
    transcript = Transcript(directory)
    ids: list[str] = []
    for turn in range(start, start + n):
        user = Message.user(f"user {turn}", id=f"u-{turn:03d}")
        assistant = Message.assistant(f"answer {turn}", id=f"a-{turn:03d}")
        await transcript.append_message(user)
        await transcript.append_message(assistant)
        ids.extend([user.id, assistant.id])
    return ids


def _fresh_render(directory) -> list[str]:
    """The full-reparse render ids — the contract the cache must match."""
    from local_operator.mobile.projection import fold_messages_to_entries

    transcript = Transcript(directory)
    return [e.id for e in fold_messages_to_entries(transcript.build_llm_history())]


def test_fold_cache_matches_full_reparse_on_append(tmp_path, monkeypatch) -> None:
    """An appended tail folds to the SAME render as re-parsing the whole file."""
    cfg = tmp_path / "config"
    directory = cfg / "sessions" / "s1"
    directory.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    _write_turns(directory, 5)
    cache = DurableFoldCache()
    state = cache.load(directory)
    first_render = [e.id for e in state.render]
    assert first_render == _fresh_render(directory)

    # Append more turns; the cache reads only the tail and must converge.
    _write_turns(directory, 3, start=5)
    state = cache.load(directory)
    assert [e.id for e in state.render] == _fresh_render(directory)
    assert len(state.render) > len(first_render)
    # entry_count is the cursor/file agreement invariant (A10): the tail read
    # must have consumed exactly the lines a full re-parse would see.
    file_lines = [
        line for line in (directory / "transcript.jsonl").read_text().splitlines() if line.strip()
    ]
    assert state.entry_count == len(file_lines)


def test_fold_cache_refolds_on_compaction_in_tail(tmp_path, monkeypatch) -> None:
    """A compaction entry in the appended tail rebuilds the cached history
    exactly as ``build_llm_history`` does — the kept window plus the marker."""
    cfg = tmp_path / "config"
    directory = cfg / "sessions" / "s1"
    directory.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    ids = _write_turns(directory, 6)
    cache = DurableFoldCache()
    cache.load(directory)

    # Compact keeping the last two turns, then append one more turn after.
    transcript = Transcript(directory)
    asyncio.run(transcript.append_compaction("summary here", ids[-4], tokens_before=100))
    asyncio.run(transcript.append_message(Message.user("after compact", id="u-post")))
    asyncio.run(transcript.append_message(Message.assistant("post answer", id="a-post")))

    state = cache.load(directory)
    from local_operator.mobile.projection import fold_messages_to_entries

    fresh = fold_messages_to_entries(Transcript(directory).build_llm_history())
    # The compaction-summary notice id is a fresh uuid on every fold
    # (pre-existing behaviour — it is not a stable identifier), so compare
    # everything but the notice id, and assert the notices match by kind/text.
    assert [e.id for e in state.render if e.kind != "notice"] == [
        e.id for e in fresh if e.kind != "notice"
    ]
    assert [(e.kind, e.text) for e in state.render if e.kind == "notice"] == [
        (e.kind, e.text) for e in fresh if e.kind == "notice"
    ]
    # The kept window starts at the compaction's first_kept entry, and the
    # post-compaction turn is present.
    render_ids = [e.id for e in state.render]
    assert "u-post" in render_ids
    assert ids[0] not in render_ids  # summarized partition is gone


def test_fold_cache_applies_prune_in_tail(tmp_path, monkeypatch) -> None:
    """A prune entry in the tail blanks its cached target, matching replay."""
    cfg = tmp_path / "config"
    directory = cfg / "sessions" / "s1"
    directory.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    ids = _write_turns(directory, 4)
    cache = DurableFoldCache()
    cache.load(directory)

    transcript = Transcript(directory)
    target = ids[3]  # an assistant answer
    asyncio.run(transcript.append_prune(target, "pruned for brevity"))

    state = cache.load(directory)
    cached_ids = [e.id for e in state.render]
    fresh_ids = _fresh_render(directory)
    assert cached_ids == fresh_ids


def test_fold_cache_rebuilds_on_rotation(tmp_path, monkeypatch) -> None:
    """``compact_file`` replaces the file (new inode): the cache must rebuild
    rather than read against a stale byte cursor."""
    cfg = tmp_path / "config"
    directory = cfg / "sessions" / "s1"
    directory.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    _write_turns(directory, 4)
    cache = DurableFoldCache()
    cache.load(directory)

    # Rewrite the file wholesale (different content AND inode): the append-only
    # cursor is now a lie and the cache must detect the rotation.
    (directory / "transcript.jsonl").unlink()
    _write_turns(directory, 2)

    state = cache.load(directory)
    assert [e.id for e in state.render] == _fresh_render(directory)


def test_fold_cache_lru_evicts_oldest(tmp_path, monkeypatch) -> None:
    """The cache is bounded; loading past the cap evicts the oldest entry."""
    cfg = tmp_path / "config"
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    cache = DurableFoldCache(max_entries=3)
    dirs = []
    for i in range(5):
        directory = cfg / "sessions" / f"s{i}"
        directory.mkdir(parents=True)
        _write_turns(directory, 1)
        dirs.append(directory)
        cache.load(directory)
    # Only the 3 newest survive; the oldest two were evicted (a re-load
    # re-creates them, which is the documented cost of eviction).
    assert len(cache._states) == 3
    assert str(dirs[0]) not in cache._states
    assert str(dirs[-1]) in cache._states


def test_custom_snapshot_cache_tracks_newest(tmp_path) -> None:
    """The snapshot cache answers newest-wins custom reads and re-scans only
    when the file changes."""
    directory = tmp_path / "child"
    directory.mkdir()
    transcript = Transcript(directory)
    asyncio.run(transcript.append_custom("todo_snapshot", {"items": [{"text": "a"}]}))

    cache = CustomSnapshotCache()
    first = cache.load(directory, "todo_snapshot")
    assert first == {"items": [{"text": "a"}]}

    # Unchanged file: same answer, no re-scan needed (still correct).
    assert cache.load(directory, "todo_snapshot") == first

    # Append a newer snapshot: newest wins.
    asyncio.run(transcript.append_custom("todo_snapshot", {"items": [{"text": "b"}]}))
    assert cache.load(directory, "todo_snapshot") == {"items": [{"text": "b"}]}


def test_tracked_custom_types_match_the_source_of_truth() -> None:
    """The literal custom types durable.py tracks must match the session's
    roster constant — they are restated to avoid importing the session module
    into the fold cache, so pin the agreement."""
    from local_operator.mobile.durable import _TRACKED_CUSTOM_TYPES
    from local_operator.session.session import SUBAGENT_ROSTER_CUSTOM_TYPE

    assert SUBAGENT_ROSTER_CUSTOM_TYPE in _TRACKED_CUSTOM_TYPES


# ---------------------------------------------------------------------------
# Summaries cache (off-loop + TTL + invalidation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summaries_caches_durable_rows_within_ttl(tmp_path, monkeypatch) -> None:
    """Repeated summaries() within the TTL pay ONE durable scan, not one per
    call — the repaint storm the fix exists to stop."""
    cfg = tmp_path / "config"
    (cfg / "sessions").mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    calls = {"n": 0}
    from local_operator import resume as resume_module

    real_rows = resume_module.recent_session_rows

    def counting_rows(config_dir, limit=None):
        calls["n"] += 1
        return real_rows(config_dir, limit)

    monkeypatch.setattr(resume_module, "recent_session_rows", counting_rows)

    table = SessionTable()
    await table.summaries()
    await table.summaries()
    await table.summaries()
    assert calls["n"] == 1  # TTL cache: one scan for three reads


@pytest.mark.asyncio
async def test_summaries_invalidation_forces_rescan(tmp_path, monkeypatch) -> None:
    """An explicit structural invalidation drops the cache so the next read
    rescans. This is the seam the register/death/wake/seen sites call; a bare
    ``notify_list_changed`` (a projection repaint) deliberately does NOT
    invalidate — see test_projection_repaints_do_not_rescan_the_store."""
    cfg = tmp_path / "config"
    (cfg / "sessions").mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    calls = {"n": 0}
    from local_operator import resume as resume_module

    real_rows = resume_module.recent_session_rows

    def counting_rows(config_dir, limit=None):
        calls["n"] += 1
        return real_rows(config_dir, limit)

    monkeypatch.setattr(resume_module, "recent_session_rows", counting_rows)

    table = SessionTable()
    await table.summaries()
    table.invalidate_summaries_cache()
    await table.summaries()
    assert calls["n"] == 2  # invalidation forced a second scan


@pytest.mark.asyncio
async def test_streaming_outranks_unseen_matching_the_render_ladder(tmp_path, monkeypatch) -> None:
    """The sort ladder must match the render ladder (A14, and D5's substance).

    The client renders NEEDS DECISION > WORKING > UNREAD > IDLE and suppresses
    the `new` mark on a streaming row, because "new" means COMPLETED unviewed
    activity. Ranking unseen ABOVE streaming in the daemon therefore hoisted a
    streaming+unseen row over newer rows while it rendered no mark to explain
    why it was there — a position the surface contradicts.

    Both rows are LIVE so the comparison lands inside one section and the
    section term cannot decide it, and the streaming row is the OLDER one so
    recency cannot either: only the ladder can. Exercises the real
    ``summaries()`` output rather than a reimplementation of the sort key — a
    test that re-derives the ordering it checks proves nothing.
    """
    import os
    import time

    from local_operator.mobile.daemon import SessionEntry
    from local_operator.mobile.types import SessionProjection, SessionRecord

    cfg = tmp_path / "config"
    unread_dir = cfg / "sessions" / "unread-live"
    streaming_dir = cfg / "sessions" / "streaming-live"
    unread_dir.mkdir(parents=True)
    streaming_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    await _write_turns_async(unread_dir, 1)
    await _write_turns_async(streaming_dir, 1)
    # The STREAMING row is the OLDER transcript: with unseen ranked first the
    # unread row wins, with streaming ranked first the live row does.
    os.utime(streaming_dir / "transcript.jsonl", (1_000_000, 1_000_000))
    os.utime(unread_dir / "transcript.jsonl", (2_000_000, 2_000_000))

    def live(pid: int, session_id: str, *, streaming: bool) -> SessionEntry:
        record = SessionRecord(
            pid=pid,
            kind="tui",
            session_id=session_id,
            conversation_name=session_id,
            cwd="/tmp",
            model_label="m",
            control_port=1,
            control_key="k",
        )
        record.started_at = time.time()
        entry = SessionEntry(record)
        entry.projection = SessionProjection(
            session_id=session_id, pid=pid, kind="tui", streaming=streaming
        )
        return entry

    table = SessionTable()
    table.entries[5150] = live(5150, "streaming-live", streaming=True)
    table.entries[5151] = live(5151, "unread-live", streaming=False)
    await table.summaries()
    # unread-live gains activity nobody has seen; the streaming row is read.
    os.utime(unread_dir / "transcript.jsonl", (2_500_000, 2_500_000))
    table.seen_store.mark_seen("streaming-live", now=3_000_000.0)
    table.invalidate_summaries_cache()

    rows = await table.summaries()
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["unread-live"]["unseen"] is True
    assert by_id["streaming-live"]["streaming"] is True
    assert by_id["streaming-live"]["unseen"] is False
    # Same section, so the ladder — not the section term — decides.
    assert by_id["unread-live"]["section"] == by_id["streaming-live"]["section"]
    ordered = [row["session_id"] for row in rows]
    assert ordered.index("streaming-live") < ordered.index(
        "unread-live"
    ), "a streaming row must outrank an unread one, matching the render ladder"


@pytest.mark.asyncio
async def test_projection_repaints_do_not_rescan_the_store(tmp_path, monkeypatch) -> None:
    """A streaming session's repaints must NOT re-run the durable scan (A3).

    Every projection push calls ``notify_list_changed`` (~30x/s). While that
    invalidated the cache, one streaming session re-imposed a 42-92 ms
    blocking scan per repaint — restoring the starvation this work removed and
    defeating the TTL in exactly the busy case it was built for.
    """
    cfg = tmp_path / "config"
    (cfg / "sessions").mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    calls = {"n": 0}
    from local_operator import resume as resume_module

    real_rows = resume_module.recent_session_rows

    def counting_rows(config_dir, limit=None):
        calls["n"] += 1
        return real_rows(config_dir, limit)

    monkeypatch.setattr(resume_module, "recent_session_rows", counting_rows)

    table = SessionTable()
    await table.summaries()
    calls["n"] = 0
    # One second of streaming: a repaint notification per push.
    for _ in range(30):
        table.notify_list_changed()
        await table.summaries()
    assert calls["n"] == 0, "projection repaints must not rescan the durable store"

    # A structural change still refreshes, so correctness is preserved.
    table.invalidate_summaries_cache()
    await table.summaries()
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_summaries_rows_carry_unseen_key(tmp_path, monkeypatch) -> None:
    """Every summary row (both sections) carries the ``unseen`` key."""
    cfg = tmp_path / "config"
    session_dir = cfg / "sessions" / "durable-1"
    session_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    await _write_turns_async(session_dir, 1)

    table = SessionTable()
    rows = await table.summaries()
    assert rows, "expected the durable session in the listing"
    for row in rows:
        assert "unseen" in row
        assert isinstance(row["unseen"], bool)


@pytest.mark.asyncio
async def test_unseen_rows_sort_above_newer_seen_rows(tmp_path, monkeypatch) -> None:
    """An OLD unread session outranks a NEWER read one inside its section.

    This is the headline of the feature: without ``unseen`` in the sort tuple
    the mark renders but the list stays in plain mtime order, so the unread
    session the user is meant to notice sits wherever recency puts it.
    """
    import os

    cfg = tmp_path / "config"
    old_dir = cfg / "sessions" / "old-unread"
    new_dir = cfg / "sessions" / "new-read"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    await _write_turns_async(old_dir, 1)
    await _write_turns_async(new_dir, 1)

    # The OLD session is the older transcript; the NEW one is newer, so plain
    # recency ordering would put "new-read" first.
    old_transcript = old_dir / "transcript.jsonl"
    new_transcript = new_dir / "transcript.jsonl"
    os.utime(old_transcript, (1_000_000, 1_000_000))
    os.utime(new_transcript, (2_000_000, 2_000_000))

    table = SessionTable()
    store = table.seen_store
    # Both observed at their current mtimes (baseline), then the older one
    # gains activity the phone has not seen while the newer one is marked read.
    await table.summaries()
    os.utime(old_transcript, (1_500_000, 1_500_000))
    store.mark_seen("new-read", now=3_000_000.0)
    table.invalidate_summaries_cache()

    rows = await table.summaries()
    ordered = [row["session_id"] for row in rows]
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["old-unread"]["unseen"] is True
    assert by_id["new-read"]["unseen"] is False
    # The unread row wins despite being the OLDER transcript.
    assert ordered.index("old-unread") < ordered.index("new-read")


@pytest.mark.asyncio
async def test_watching_a_session_over_sse_keeps_it_seen(tmp_path, monkeypatch) -> None:
    """Holding the projection SSE stream IS viewing (A4, spec §3).

    A user sitting in a session watching a turn finish must not come back to
    that session marked new. Before this, the only writer was the client's
    /seen POST on mount, so any activity after mount re-lit the mark.
    """
    import os
    import time

    cfg = tmp_path / "config"
    session_dir = cfg / "sessions" / "watched"
    session_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    await _write_turns_async(session_dir, 1)
    transcript = session_dir / "transcript.jsonl"

    table = SessionTable()
    opened_at = time.time()
    os.utime(transcript, (opened_at, opened_at))

    rows = await table.summaries()
    assert next(r["unseen"] for r in rows if r["session_id"] == "watched") is False

    # The phone holds the projection stream for this session.
    table.session_subscribers["watched"] = {asyncio.Queue()}
    table.invalidate_summaries_cache()
    await table.summaries()

    # A turn completes while the user is still watching.
    completed_at = opened_at + 30
    os.utime(transcript, (completed_at, completed_at))
    table.invalidate_summaries_cache()
    rows = await table.summaries()
    assert next(r["unseen"] for r in rows if r["session_id"] == "watched") is False

    # They navigate away; activity after that DOES light the mark.
    table.session_subscribers.pop("watched")
    later = time.time() + 120
    os.utime(transcript, (later, later))
    table.invalidate_summaries_cache()
    rows = await table.summaries()
    assert next(r["unseen"] for r in rows if r["session_id"] == "watched") is True


@pytest.mark.asyncio
async def test_live_row_activity_clock_ignores_the_heartbeat(tmp_path, monkeypatch) -> None:
    """A live session with no durable row must not re-light on its heartbeat (A5).

    ``heartbeat_at`` is rewritten every HEARTBEAT_INTERVAL_S whether or not
    anything happened, so using it as the activity clock brought a cleared
    mark back 15 s later, forever.
    """
    import time

    from local_operator.mobile.daemon import SessionEntry
    from local_operator.mobile.types import SessionRecord

    cfg = tmp_path / "config"
    (cfg / "sessions").mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    table = SessionTable()
    now = time.time()
    record = SessionRecord(
        pid=4242,
        kind="tui",
        session_id="live-only",
        conversation_name="x",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    record.started_at = now
    record.heartbeat_at = now
    table.entries[4242] = SessionEntry(record)
    table.seen_store.mark_seen("live-only", now=now)

    for bump in (0, 15, 30, 45):
        record.heartbeat_at = now + bump
        table.invalidate_summaries_cache()
        rows = await table.summaries()
        row = next(r for r in rows if r["session_id"] == "live-only")
        assert row["unseen"] is False, f"heartbeat +{bump}s re-lit a cleared mark"


# ---------------------------------------------------------------------------
# Seen store
# ---------------------------------------------------------------------------


def test_seen_store_baseline_prevents_flood_on_first_observation(tmp_path) -> None:
    """A never-seen session is unseen only if it gained activity SINCE first
    observation — the rule that keeps an upgrade from lighting up the store."""
    store = SeenStore(tmp_path / SEEN_STORE_NAME)
    # First observation at mtime 100 records the baseline.
    assert store.is_unseen("s1", 100.0) is False
    # Same activity: not unseen. Newer activity: unseen.
    assert store.is_unseen("s1", 100.0) is False
    assert store.is_unseen("s1", 150.0) is True


def test_seen_store_mark_seen_clears_then_relights(tmp_path) -> None:
    """mark_seen clears the verdict; newer activity re-lights it."""
    store = SeenStore(tmp_path / SEEN_STORE_NAME)
    store.is_unseen("s1", 100.0)  # baseline
    assert store.is_unseen("s1", 150.0) is True
    store.mark_seen("s1", now=200.0)
    assert store.is_unseen("s1", 150.0) is False  # older than last_seen
    assert store.is_unseen("s1", 250.0) is True  # newer than last_seen


def test_watch_debounce_is_per_session(tmp_path) -> None:
    """Each watched session gets its own persist clock (A15).

    A single instance-wide timestamp meant the first watched session to write
    suppressed the disk write for every other session for the whole interval,
    so a second session watched in that window kept its stamp only in memory
    and lost it on a crash. Two phones on two sessions is the ordinary case.
    """
    import json

    path = tmp_path / SEEN_STORE_NAME
    store = SeenStore(path)
    for index, session_id in enumerate(("a", "b", "c")):
        store.touch_watched(session_id, now=1000.0 + index)
    persisted = json.loads(path.read_text())["sessions"]
    assert sorted(persisted) == ["a", "b", "c"]

    # The debounce still does its job WITHIN one session: repeat touches
    # inside the interval must not rewrite the file.
    before = path.stat().st_mtime_ns
    for _ in range(300):
        store.touch_watched("a", now=1003.0)
    assert path.stat().st_mtime_ns == before


def test_seen_store_persists_last_seen_across_restart(tmp_path) -> None:
    """last_seen survives a store reload; baselines re-derive safely."""
    path = tmp_path / SEEN_STORE_NAME
    store = SeenStore(path)
    store.is_unseen("s1", 100.0)
    store.mark_seen("s1", now=200.0)

    reloaded = SeenStore(path)
    # The persisted last_seen wins: activity before it is seen, after is not.
    assert reloaded.is_unseen("s1", 150.0) is False
    assert reloaded.is_unseen("s1", 250.0) is True


def test_seen_store_file_is_0600_and_atomic(tmp_path) -> None:
    """The store file is owner-only readable."""
    import os
    import stat

    path = tmp_path / SEEN_STORE_NAME
    store = SeenStore(path)
    store.mark_seen("s1", now=200.0)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_seen_store_bounds_entries(tmp_path) -> None:
    """Past the cap the oldest stamps are dropped."""
    store = SeenStore(tmp_path / SEEN_STORE_NAME)
    for i in range(MAX_SEEN_ENTRIES + 10):
        store.mark_seen(f"s{i}", now=float(1000 + i))
    assert len(store._last_seen) == MAX_SEEN_ENTRIES
    # The oldest were dropped, the newest kept.
    assert store.last_seen(f"s{MAX_SEEN_ENTRIES + 9}") is not None
    assert store.last_seen("s0") is None


# ---------------------------------------------------------------------------
# Frame-cap tiering
# ---------------------------------------------------------------------------


def test_frame_cap_delivers_oversized_subagent_result_under_cap() -> None:
    """Regression: a projection carrying a 5 MB subagent result still yields a
    frame under the soft cap — the exact payload that wedged the relay."""
    projection = SessionProjection(session_id="s1", pid=1, kind="tui")
    projection.subagents.append(SubagentRow(job_id="j1", label="big", result_text="R" * 5_000_000))
    frame, degraded = cap_projection_frame(projection)
    assert degraded is True
    size = len(json.dumps(frame).encode("utf-8"))
    assert size <= PROJECTION_FRAME_SOFT_CAP_BYTES
    # Tier 1 trimmed the result preview to the minimal bound.
    assert len(frame["subagents"][0]["result_text"]) <= FRAME_CAP_RESULT_CHARS


def test_frame_cap_does_not_mutate_the_projection() -> None:
    """Degradation happens on the serialized dict; the fold's projection is
    untouched (it is republished on the next repaint)."""
    projection = SessionProjection(session_id="s1", pid=1, kind="tui")
    projection.subagents.append(SubagentRow(job_id="j1", label="big", result_text="R" * 5_000_000))
    cap_projection_frame(projection)
    assert len(projection.subagents[0].result_text) == 5_000_000


def test_frame_cap_small_frame_passes_through_unchanged() -> None:
    """A frame under the cap is not degraded and reports not-degraded."""
    projection = SessionProjection(session_id="s1", pid=1, kind="tui")
    projection.subagents.append(SubagentRow(job_id="j1", label="small", result_text="ok"))
    frame, degraded = cap_projection_frame(projection)
    assert degraded is False
    assert frame["subagents"][0]["result_text"] == "ok"


def test_frame_cap_bounds_a_single_undegradable_row() -> None:
    """A projection tiers 1-3 cannot shrink is STILL bounded (A2/A8).

    One pasted file in a single row exceeds the whole cap on its own: no
    subagent text to trim, no details to drop, and the tail floor keeps the
    row. Before the text tier this returned a 2,000,644-byte frame that the
    daemon's 1 MB control reader dropped whole — the silent-repaint-loss the
    cap exists to prevent.
    """
    from local_operator.mobile.types import TranscriptEntry

    projection = SessionProjection(session_id="s1", pid=1, kind="tui")
    projection.transcript.append(TranscriptEntry(id="u1", kind="user", text="X" * 2_000_000))
    frame, degraded = cap_projection_frame(projection)
    assert degraded is True
    size = len(json.dumps(frame).encode("utf-8"))
    assert size <= PROJECTION_FRAME_SOFT_CAP_BYTES
    # And comfortably under the daemon's hard control-socket limit.
    assert size < (1 << 20)


def test_frame_cap_bounds_many_individually_large_rows() -> None:
    """Many rows that each survive the tail floor are still bounded."""
    from local_operator.mobile.types import TranscriptEntry

    projection = SessionProjection(session_id="s1", pid=1, kind="tui")
    for i in range(40):
        projection.transcript.append(
            TranscriptEntry(id=f"a{i}", kind="assistant", text="Y" * 80_000)
        )
    frame, degraded = cap_projection_frame(projection)
    assert degraded is True
    assert len(json.dumps(frame).encode("utf-8")) < (1 << 20)


#: Alphabets the frame-cap fuzz draws from. The non-ASCII entries are
#: load-bearing, not decoration: ``json.dumps`` defaults to
#: ``ensure_ascii=True``, so one CJK character costs 6 wire bytes and one
#: astral emoji 12, while Python ``len()`` counts both as 1. An ASCII-only
#: fuzz therefore cannot generate the input that breaks a character-counting
#: size estimate — which is exactly how that defect shipped green twice.
_FUZZ_ALPHABETS = (
    string.ascii_letters,
    '"\\\n',  # escape-dense ASCII
    "\u4e2d\u6587\u5b57",  # CJK: 6 wire bytes per char
    "\u00e9\u00f1\u00e0",  # Latin-1 accents: 6 wire bytes per char
    "\u0434\u0430",  # Cyrillic: 6 wire bytes per char
    "\U0001f600\U0001f680",  # astral emoji: 12 wire bytes per char
)

#: Single-character fillers for the roster/pending fields, same rationale.
_FUZZ_FILLERS = ("Z", "\u4e2d", "\u00e9", "\U0001f600")


def test_frame_measurement_gate_never_lets_an_oversized_frame_through() -> None:
    """The measurement short-circuit must never pass an oversized frame (A12/A13).

    ``cap_projection_frame`` skips its measuring ``json.dumps`` when a
    projection is structurally too simple to approach the cap. If that gate
    ever says "skip" for a frame that is actually over, the payload returns as
    ``degraded=False``, neither the registrant warning nor the tier-4 warning
    fires, and the socket drops the repaint in silence.

    An earlier revision gated on a SUM of known text fields and did exactly
    that: it never walked ``subagents[].todos`` (kept on the wire by design),
    ``subagents[].transcript`` or ``projection.pending``, so 80 children x 25
    todos x 600 chars estimated 40,860 bytes against a real 1,331,102. This
    fuzz therefore populates precisely those three fields — it passes
    vacuously without them, which is how the defect shipped.
    """
    import random

    from local_operator.mobile.types import (
        AskOptionWire,
        PendingRequest,
        SubagentRow,
        TodoItem,
        TodoPhase,
        TranscriptEntry,
    )

    random.seed(11)
    for _ in range(120):
        projection = SessionProjection(session_id="s", pid=1, kind="tui")
        # Row count and per-row length reach the region where a CJK/emoji
        # frame clears the cap: at 6-12 wire bytes per character, ~40 rows of
        # ~3,000 chars is already past 700 KB while a character-counting
        # estimate still reads it as ~120 KB. Drawn as a whole-projection
        # profile rather than per row so a run actually lands there instead of
        # averaging out to a harmless mix.
        row_count = random.randint(0, 60)
        row_chars = random.choice((200, 800, 2000, 3000))
        for i in range(row_count):
            alphabet = random.choice(_FUZZ_ALPHABETS)
            projection.transcript.append(
                TranscriptEntry(
                    id=f"t{i}",
                    kind="assistant",
                    text="".join(random.choice(alphabet) for _ in range(row_chars)),
                )
            )
        # A roster or a pending card FORCES measurement, so a fuzz that always
        # generates one never exercises the gate's own arithmetic — the path
        # where a character-counting estimate silently vouched for a 1.17 MB
        # CJK frame. Half the runs are deliberately bare.
        subagent_count = 0 if random.random() < 0.5 else random.randint(1, 60)
        for j in range(subagent_count):
            projection.subagents.append(
                SubagentRow(
                    job_id=f"job-{j}" * 3,
                    label="L" * 20,
                    result_text=random.choice(_FUZZ_FILLERS) * random.randint(0, 1500),
                    ancestors=["anc" * 10] * random.randint(0, 8),
                    ancestor_ids=["id" * 12] * random.randint(0, 8),
                    child_ids=["c" * 12] * random.randint(0, 8),
                    peer_ids=["p" * 12] * random.randint(0, 8),
                    # The three fields the old estimate ignored, at the
                    # magnitudes the real wire carries.
                    todos=[
                        TodoPhase(
                            name="Todos",
                            items=[
                                TodoItem(text=random.choice(_FUZZ_FILLERS) * random.randint(0, 600))
                                for _ in range(random.randint(0, 25))
                            ],
                        )
                    ],
                    transcript=[
                        TranscriptEntry(
                            id=f"c{k}",
                            kind="assistant",
                            text=random.choice(_FUZZ_FILLERS) * 1500,
                        )
                        for k in range(random.randint(0, 6))
                    ],
                )
            )
        for _k in range(random.randint(0, 4)):
            projection.todos.append(
                TodoPhase(
                    name="P" * 20,
                    items=[TodoItem(text="t" * 80) for _ in range(random.randint(0, 20))],
                )
            )
        if subagent_count and random.random() < 0.4:
            projection.pending = PendingRequest(
                request_id="r",
                kind="ask",
                title=random.choice(_FUZZ_FILLERS) * random.randint(0, 2000),
                options=[
                    AskOptionWire(
                        label="o" * 40,
                        description=random.choice(_FUZZ_FILLERS) * random.randint(0, 20000),
                    )
                    for _ in range(random.randint(0, 20))
                ],
            )

        frame, degraded = cap_projection_frame(projection)
        real = len(json.dumps(frame).encode("utf-8"))
        if not degraded:
            # The gate vouched for this frame without measuring it, so the
            # claim it made must be true.
            assert (
                real <= PROJECTION_FRAME_SOFT_CAP_BYTES
            ), f"measurement gate passed a {real}-byte frame as undegraded"


def test_frame_cap_bounds_non_ascii_prose_with_no_roster() -> None:
    """CJK/emoji prose alone must not slip the measurement gate (A12).

    ``json.dumps`` defaults to ``ensure_ascii=True``, so one CJK character is
    6 wire bytes and one astral emoji is 12, while Python ``len()`` counts
    both as 1. A character-counting gate therefore vouched for frames of
    1.17 MB (80 rows x 2,400 CJK chars) and 1.14 MB (one pasted CJK document)
    with no subagents and no pending card — returned ``degraded=False``, so no
    tier ran, nothing was logged, and the daemon's 1 MB reader dropped the
    repaint in silence. Ordinary Chinese, Japanese, Russian or accented French
    prose sits on this curve; it was never exotic input.
    """
    from local_operator.mobile.types import TranscriptEntry

    for label, rows, chars, char in (
        ("cjk-many-rows", 80, 2400, "\u4e2d"),
        ("cjk-pasted-doc", 1, 190_000, "\u4e2d"),
        ("emoji-pasted-doc", 1, 190_000, "\U0001f600"),
        ("latin1-french", 100, 2000, "\u00e9"),
        ("cyrillic", 100, 2000, "\u0434"),
    ):
        projection = SessionProjection(session_id="s", pid=1, kind="tui")
        for i in range(rows):
            projection.transcript.append(
                TranscriptEntry(id=f"t{i}", kind="assistant", text=char * chars)
            )
        frame, degraded = cap_projection_frame(projection)
        size = len(json.dumps(frame).encode("utf-8"))
        assert degraded is True, f"{label}: oversized non-ASCII frame not degraded"
        assert size <= PROJECTION_FRAME_SOFT_CAP_BYTES, f"{label}: {size} bytes"
        assert size < (1 << 20), f"{label}: over the daemon's hard reader limit"

    # The all-ASCII control still takes the cheap path: the fix must not cost
    # the hot repaint its short-circuit.
    from local_operator.mobile.projection import _frame_skips_measurement

    control = SessionProjection(session_id="s", pid=1, kind="tui")
    for i in range(120):
        control.transcript.append(TranscriptEntry(id=f"t{i}", kind="assistant", text="a" * 1500))
    assert _frame_skips_measurement(control) is True
    _, degraded = cap_projection_frame(control)
    assert degraded is False


def test_wire_charge_never_under_counts_the_serializer() -> None:
    """``_wire_charge`` must be an upper bound on what the wire charges (A12).

    The gate is only safe while this holds: it decides whether to skip the
    real measurement, so an under-count is a silently dropped repaint.
    """
    from local_operator.mobile.projection import _wire_charge

    for text in (
        "plain ascii text",
        '"quotes" and \\backslashes\\',
        "\u4e2d\u6587\u5b57" * 100,
        "\u00e9\u00f1\u00e0" * 100,
        "\u0434\u0430" * 100,
        "\U0001f600\U0001f680" * 100,
        "mixed \u4e2d ascii \U0001f600 prose",
        "",
    ):
        # What the serializer actually charges for this field, minus its quotes.
        actual = len(json.dumps(text).encode("utf-8")) - 2
        assert _wire_charge(text) >= actual, f"under-counted {text[:20]!r}"


def test_frame_cap_bounds_a_deep_roster_carrying_todos() -> None:
    """A deep roster's todos are on the wire by design and must be bounded (A12).

    ``set_subagent_hydrated_details`` keeps ``row.todos`` while dropping the
    child transcript, and todo text is agent-authored and unbounded. 80
    children x 25 todos x 600 chars measured 1,331,102 bytes returned as
    ``degraded=False`` before this was fixed — past the daemon's 1 MB reader,
    and silent.
    """
    from local_operator.mobile.types import SubagentRow, TodoItem, TodoPhase

    projection = SessionProjection(session_id="s", pid=1, kind="tui")
    for i in range(80):
        projection.subagents.append(
            SubagentRow(
                job_id=f"j{i}",
                label=f"child {i}",
                status="running",
                todos=[
                    TodoPhase(
                        name="Todos",
                        items=[TodoItem(text="T" * 600) for _ in range(25)],
                    )
                ],
            )
        )
    frame, degraded = cap_projection_frame(projection)
    assert degraded is True
    assert len(json.dumps(frame).encode("utf-8")) < (1 << 20)


def test_frame_cap_bounds_a_pending_card_with_long_options() -> None:
    """A pending card's option prose is bounded, and the card still ships (A12).

    ``projection.pending`` was never counted at all, so an ask with long
    consequence lines measured 1,123,365 bytes and returned undegraded.
    """
    from local_operator.mobile.types import AskOptionWire, PendingRequest

    projection = SessionProjection(session_id="s", pid=1, kind="tui")
    projection.pending = PendingRequest(
        request_id="r",
        kind="ask",
        title="T" * 200,
        options=[AskOptionWire(label="o" * 60, description="D" * 40_000) for _ in range(28)],
    )
    frame, degraded = cap_projection_frame(projection)
    assert degraded is True
    assert len(json.dumps(frame).encode("utf-8")) < (1 << 20)
    # The card itself survives: it is the one thing on the phone that blocks a
    # turn, so only the prose under each option is trimmed.
    assert frame["pending"] is not None
    assert len(frame["pending"]["options"]) == 28


def test_frame_cap_tier2_drops_transcript_details() -> None:
    """When subagent trims are not enough, transcript expand details go next."""
    from local_operator.mobile.types import TranscriptEntry

    projection = SessionProjection(session_id="s1", pid=1, kind="tui")
    # Many tool rows with large expand payloads but small subagent text.
    for i in range(60):
        projection.transcript.append(
            TranscriptEntry(
                id=f"t{i}",
                kind="tool",
                tool_name="bash",
                details={"output": "O" * 20_000},
            )
        )
    frame, degraded = cap_projection_frame(projection)
    assert degraded is True
    assert len(json.dumps(frame).encode("utf-8")) <= PROJECTION_FRAME_SOFT_CAP_BYTES
    assert all(entry["details"] == {} for entry in frame["transcript"])


# ---------------------------------------------------------------------------
# /seen endpoint
# ---------------------------------------------------------------------------


def test_seen_endpoint_requires_auth_and_marks_seen(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "config"
    session_dir = cfg / "sessions" / "durable-1"
    session_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    _write_turns(session_dir, 1)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    # Unauthenticated: 401 on the API route.
    assert client.post("/api/sessions/durable-1/seen").status_code == 401

    client.post("/login", data={"password": "pw123"})
    response = client.post("/api/sessions/durable-1/seen")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    # The store recorded the seen stamp.
    assert daemon.seen_store.last_seen("durable-1") is not None


def test_seen_endpoint_unknown_session_is_404(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "config"
    (cfg / "sessions").mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    assert client.post("/api/sessions/nope/seen").status_code == 404


def test_seen_endpoint_clears_unseen_in_summaries(tmp_path, monkeypatch) -> None:
    """Marking a session seen flips its summary ``unseen`` to False."""
    cfg = tmp_path / "config"
    session_dir = cfg / "sessions" / "durable-1"
    session_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    _write_turns(session_dir, 1)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})

    def unseen_for(session_id: str) -> bool:
        rows = client.get("/api/sessions").json()["sessions"]
        return next(r["unseen"] for r in rows if r["session_id"] == session_id)

    # First observation records the baseline at the current mtime, so the
    # session is NOT unseen on first paint (no flood on upgrade).
    assert unseen_for("durable-1") is False
    client.post("/api/sessions/durable-1/seen")
    assert unseen_for("durable-1") is False

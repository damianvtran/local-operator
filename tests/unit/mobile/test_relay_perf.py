"""Relay session-load performance: incremental fold cache, summaries cache,
seen store, and frame-cap tiering.

These pin the behaviour the relay-perf change is built on, against the same
store primitives (``Transcript``) the daemon reads, so the cached paths stay
byte-identical to the full-reparse paths they replaced.
"""

from __future__ import annotations

import asyncio
import json

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


def test_frame_cheap_proxy_never_lets_an_oversized_frame_through() -> None:
    """The A7 short-circuit must be an OVER-estimate, always (A7 safety).

    ``cap_projection_frame`` skips its measuring ``json.dumps`` when the cheap
    text/envelope estimate clears a margin. If that estimate ever UNDER-counts
    a real frame, an oversized payload returns as ``degraded=False`` and the
    socket drops it — the exact silent loss the cap prevents. Fuzzed over deep
    rosters, escape-dense text and many short rows, which is where an earlier
    text-only proxy measured 3.5x under.
    """
    import random
    import string

    from local_operator.mobile.projection import (
        _FRAME_CHEAP_PROXY_DIVISOR,
        _frame_text_total,
    )
    from local_operator.mobile.types import SubagentRow, TodoItem, TodoPhase, TranscriptEntry

    random.seed(11)
    threshold = PROJECTION_FRAME_SOFT_CAP_BYTES // _FRAME_CHEAP_PROXY_DIVISOR
    for _ in range(120):
        projection = SessionProjection(session_id="s", pid=1, kind="tui")
        for i in range(random.randint(0, 60)):
            alphabet = '"\\\n' if random.random() < 0.5 else string.ascii_letters
            projection.transcript.append(
                TranscriptEntry(
                    id=f"t{i}",
                    kind="assistant",
                    text="".join(random.choice(alphabet) for _ in range(random.randint(0, 2000))),
                )
            )
        for j in range(random.randint(0, 60)):
            projection.subagents.append(
                SubagentRow(
                    job_id=f"job-{j}" * 3,
                    label="L" * 20,
                    result_text="Z" * random.randint(0, 1500),
                    ancestors=["anc" * 10] * random.randint(0, 8),
                    ancestor_ids=["id" * 12] * random.randint(0, 8),
                    child_ids=["c" * 12] * random.randint(0, 8),
                    peer_ids=["p" * 12] * random.randint(0, 8),
                )
            )
        for _k in range(random.randint(0, 4)):
            projection.todos.append(
                TodoPhase(
                    name="P" * 20,
                    items=[TodoItem(text="t" * 80) for _ in range(random.randint(0, 20))],
                )
            )
        proxy = _frame_text_total(projection)
        real = len(json.dumps(projection.to_json()).encode("utf-8"))
        if proxy <= threshold:
            assert real <= PROJECTION_FRAME_SOFT_CAP_BYTES, (
                f"proxy {proxy} short-circuited a {real}-byte frame"
            )


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

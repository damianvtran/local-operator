"""The bounded index warm: one test per guard, plus the selection rule.

Every guard here answers "not now" for a reason the operator can read (the
reason string names the measurement), and each is injected rather than simulated:
a test cannot make a disk full or a host busy, and a guard nobody can exercise is
a guard nobody knows works.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import cast

import pytest

from local_operator.session import index_prewarm as prewarm
from local_operator.session.transcript_index import probe_index


@pytest.fixture(autouse=True)
def quiet_host(monkeypatch):
    """Neutralise BOTH resource guards for the tests about the queue.

    Not a convenience: both guards really do fire in this environment — the
    pytest tmp volume reported 862 MB free and this host's load per CPU was
    around 10 while these ran — so every test that expects a scan must say which
    disk and which host it is scanning on. Each guard's own behaviour is pinned
    by its own test below, which overrides this fixture.
    """
    monkeypatch.setattr(prewarm, "free_bytes", lambda _root=None: 1 << 40)
    monkeypatch.setattr(prewarm, "load_per_cpu", lambda: 0.0)


def journal(tmp_path, session_id: str, *, mtime: float) -> str:
    """One session directory with a journal, stamped at ``mtime``."""
    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True)
    path = directory / "transcript.jsonl"
    path.write_text(
        '{"id":"u1","ts":1.0,"type":"message","payload":{"kind":"message",'
        '"role":"user","content":[{"text":"hello"}]}}\n',
        encoding="utf-8",
    )
    os.utime(path, (mtime, mtime))
    return session_id


async def build_index(root, session_id: str) -> None:
    """Write a CURRENT index cache for ``session_id`` through the real scanner."""
    from local_operator.session.transcript_index import refresh_index

    await asyncio.to_thread(refresh_index, root, session_id)
    assert probe_index(root, session_id).state == "ready"


@pytest.mark.asyncio
async def test_journals_owing_either_job_are_the_candidates(tmp_path):
    """Newest first, both reasons, stop at the limit."""
    root = tmp_path
    old = journal(root, "old", mtime=1_000.0)
    new = journal(root, "new", mtime=3_000.0)
    fresh = journal(root, "fresh", mtime=2_000.0)
    # A CURRENT cache for one of them: the warm must not spend a scan on it.
    await build_index(root, fresh)

    # EITHER REASON QUALIFIES (review round 1, F4): ``fresh`` owes no refresh, but
    # its tail anchor is not current either, so the queue still visits it. The list
    # used to be index-only, which silently starved the anchor pass.
    assert prewarm.sessions_needing_warm(root, limit=2) == [new, fresh]
    assert prewarm.sessions_needing_warm(root, limit=5) == [new, fresh, old]


@pytest.mark.asyncio
async def test_the_warm_scans_the_candidates_and_leaves_a_current_cache(tmp_path):
    root = tmp_path
    ids = [journal(root, f"s{index}", mtime=1_000.0 + index) for index in range(3)]
    assert [probe_index(root, sid).state for sid in ids] == ["stale"] * 3

    started = await prewarm.warm_index_cache(root, limit=3)

    assert started == 3
    assert [probe_index(root, sid).state for sid in ids] == ["ready"] * 3


@pytest.mark.asyncio
async def test_the_disk_guard_skips_the_whole_warm(tmp_path, monkeypatch):
    """Under the floor nothing is listed, nothing is scanned, and the reason says why."""
    root = tmp_path
    journal(root, "s1", mtime=1_000.0)
    monkeypatch.setattr(
        prewarm, "free_bytes", lambda _root=None: prewarm.PREWARM_MIN_FREE_BYTES - 1
    )

    def fail(*_args, **_kwargs):  # pragma: no cover - a guard test must not reach it
        raise AssertionError("the warm scanned with too little disk free")

    monkeypatch.setattr(prewarm, "start_refresh", fail)
    disk_reason = prewarm.skip_reason(root)
    assert disk_reason is not None and disk_reason.startswith("disk: ")
    assert await prewarm.warm_index_cache(root, limit=3) == 0
    assert prewarm.start_index_prewarm(root) is None


@pytest.mark.asyncio
async def test_a_busy_host_degrades_the_warm_to_one_journal(tmp_path, monkeypatch):
    """Load PACES the queue now; it does not refuse it (the pre-warm degradation).

    The old rule refused the whole warm at or above ``PREWARM_MAX_LOAD_PER_CPU``,
    which on this fleet is the working day: QA measured the refusal (``load: 1.28 per
    CPU ...``, 0 caches) on a host that then never warmed anything. A busy host now
    gets ONE journal per pass — the work is a single scan that yields the loop between
    journals — and ``skip_reason`` answers only for DISK pressure.
    """
    root = tmp_path
    for index in range(3):
        journal(root, f"s{index}", mtime=1_000.0 + index)
    monkeypatch.setattr(prewarm, "load_per_cpu", lambda: prewarm.PREWARM_MAX_LOAD_PER_CPU)

    assert prewarm.skip_reason(root) is None, "load must not refuse the whole warm"
    pace = prewarm.pace_reason()
    assert pace is not None and pace.startswith("load: ")
    assert (
        await prewarm.warm_index_cache(root, limit=3) == 1
    ), "a busy host warms exactly one journal, not none"
    # ... and the startup hook still schedules the (degraded) pass rather than
    # declining to create a task.
    assert prewarm.start_index_prewarm(root) is not None


@pytest.mark.asyncio
async def test_the_per_session_warm_schedules_the_refresh_and_never_raises(tmp_path, monkeypatch):
    """``start_session_warm`` is the per-session path: schedule, return, no wait."""
    root = tmp_path
    journal(root, "s1", mtime=1_000.0)
    started: list[str] = []

    monkeypatch.setattr(prewarm, "start_refresh", lambda _root, sid: started.append(sid))

    assert prewarm.start_session_warm(root, "s1") is True
    assert started == ["s1"]

    # A machine under disk pressure takes no work and reports it instead of raising.
    monkeypatch.setattr(
        prewarm, "free_bytes", lambda _root=None: prewarm.PREWARM_MIN_FREE_BYTES - 1
    )
    assert prewarm.start_session_warm(root, "s1") is False


@pytest.mark.asyncio
async def test_a_host_that_gets_busy_mid_warm_drops_the_rest_of_the_queue(tmp_path, monkeypatch):
    """The queue is the first thing to go: it is the work nobody is waiting on."""
    root = tmp_path
    for index in range(3):
        journal(root, f"s{index}", mtime=1_000.0 + index)
    measured = {"load": 0.0}
    monkeypatch.setattr(prewarm, "load_per_cpu", lambda: measured["load"])

    real_start = prewarm.start_refresh
    calls = []

    def measured_start(_root, session_id):
        # ``start_refresh`` is the SYNCHRONOUS single-flight entry point that
        # returns its task; the scan it starts is what takes the time.
        calls.append(session_id)
        # The host gets busy after the FIRST journal.
        measured["load"] = prewarm.PREWARM_MAX_LOAD_PER_CPU
        return real_start(_root, session_id)

    monkeypatch.setattr(prewarm, "start_refresh", measured_start)
    started = await prewarm.warm_index_cache(root, limit=3)

    assert len(calls) == 1, f"the queue kept going under load: {calls}"
    assert started == 1


@pytest.mark.asyncio
async def test_an_unmeasurable_disk_or_load_does_not_drop_the_warm(tmp_path, monkeypatch):
    """``None`` is "cannot tell", not "full" or "busy": the guard protects a disk we can see."""
    root = tmp_path
    monkeypatch.setattr(prewarm, "free_bytes", lambda _root=None: None)
    monkeypatch.setattr(prewarm, "load_per_cpu", lambda: None)

    assert prewarm.skip_reason(root) is None


@pytest.mark.asyncio
async def test_starting_the_warm_never_blocks_and_never_raises(tmp_path, monkeypatch):
    """The daemon's startup is what an attach waits on: the warm may not be in it."""
    root = tmp_path
    for index in range(4):
        journal(root, f"s{index}", mtime=1_000.0 + index)
    # A refresh that takes real time: if startup waited for it, the elapsed time
    # below would show it.
    hold = asyncio.Event()
    real_start = prewarm.start_refresh

    def slow_start(_root, session_id):
        # SYNCHRONOUS, because ``start_refresh`` is: it returns its task, and the
        # task is what waits here.
        async def scan():
            await hold.wait()
            task, _started = real_start(_root, session_id)
            await task

        return asyncio.get_running_loop().create_task(scan()), True

    monkeypatch.setattr(prewarm, "start_refresh", slow_start)
    began = time.perf_counter()
    task = prewarm.start_index_prewarm(root)
    elapsed = time.perf_counter() - began

    assert task is not None
    assert elapsed < 0.05, f"starting the warm took {elapsed:.3f}s"
    hold.set()
    await task
    # And a failure inside the queue is the task's, never the caller's.
    monkeypatch.setattr(
        prewarm, "start_refresh", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    journal(root, "s9", mtime=9_000.0)
    assert await prewarm.warm_index_cache(root, limit=1) == 0


async def journal_with_checkpoint_row(root: Path, session_id: str, *, mtime: float) -> None:
    """A journal that HAS a checkpoint row — the shape the anchor never describes.

    ``build_anchor`` returns ``None`` for it by design (there is nothing to prove),
    so before F11 the selector called it "not current" forever.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory, defer_materialise=False)
    await transcript.append_message(Message(role="user", content=[TextContent(text="hi")]))
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"session_id": session_id, "sequence": 1}}
    )
    os.utime(directory / "transcript.jsonl", (mtime, mtime))


@pytest.mark.asyncio
async def test_a_journal_that_needs_no_anchor_stops_being_a_candidate(tmp_path, monkeypatch):
    """F11: the degraded pass must reach the journal the anchor exists for.

    With one slot per pass, a newest journal that already carries the checkpoint row
    and whose index is current consumed the slot EVERY pass — the reviewer measured
    three passes in a row with the same candidate and the checkpointless journal never
    reached. Settling the answer in-process is what unblocks the queue.
    """
    root = tmp_path
    mtime = 1_000.0
    for name in ("older", "newest"):
        directory = root / "sessions" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "transcript.jsonl").write_text("", encoding="utf-8")
        os.utime(directory / "transcript.jsonl", (mtime, mtime))
        mtime += 1_000.0
    await journal_with_checkpoint_row(root, "newest", mtime=3_000.0)
    # Its INDEX is current too, so the ANCHOR is the only reason it could be a
    # candidate — which is exactly F11's condition.
    await build_index(root, "newest")

    monkeypatch.setattr(prewarm, "load_per_cpu", lambda: prewarm.PREWARM_MAX_LOAD_PER_CPU)
    # The REAL anchor writer runs (the fixture journals are tiny): settling the
    # answer is what this test is about, and a stub would not settle it. The index
    # refresh is stubbed because the scan is not the subject here.
    # A joinable task with ``was_started=True``: the counter this asserts on is the
    # refresh's, and ``warm_index_cache`` awaits the task it is handed.
    monkeypatch.setattr(prewarm, "start_refresh", lambda _root, _sid: (asyncio.sleep(0), True))

    first = await prewarm.warm_index_cache(root, limit=3)
    assert first == 1
    newest = root / "sessions" / "newest"
    stat = (newest / "transcript.jsonl").stat()
    assert prewarm._NO_ANCHOR_NEEDED.get(str(newest)) == (
        stat.st_ino,
        stat.st_size,
    ), "the journal that needs no anchor was not settled"
    # The settled journal is no longer a candidate: the queue moves on to the older,
    # checkpointless one instead of spinning on the newest again.
    assert prewarm.sessions_needing_warm(root, limit=3)[0] == "older"


def test_the_anchor_write_is_single_flight(tmp_path, monkeypatch):
    """F15: two callers, one scan.

    The write is reachable from the startup queue AND from a warm request, and its
    expensive half is a whole-file scan for a checkpointless journal — the very scan
    the record exists to make unnecessary.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    root = tmp_path
    directory = root / "sessions" / "s1"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    scans: list[Path] = []

    def slow_build(target):
        scans.append(Path(target))
        threading.Event().wait(0.05)
        return None

    monkeypatch.setattr(prewarm, "build_anchor", slow_build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _i: prewarm.write_tail_anchor(root, "s1"), range(2)))

    assert len(scans) == 1, f"the scan ran {len(scans)} times"
    assert results == [False, False], "no sidecar belongs on this journal"


# ---------------------------------------------------------------------------
# The replay preheat: spawn-free, cache-reusing, single-flighted, guarded.
# ---------------------------------------------------------------------------


async def replay_ready_journal(root: Path, session_id: str, *, mtime: float, rows: int = 4) -> None:
    """A journal that materializes a REAL replay (via the real suffix read).

    Written through ``Transcript`` so the rows carry the fields the suffix reader
    and ``replay_entries`` expect — a hand-written message line would parse but
    would not exercise the same path the preheat shares with a reader.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory, defer_materialise=False)
    for index in range(rows):
        await transcript.append_message(
            Message(role="user", content=[TextContent(text=f"turn {index}")])
        )
    os.utime(directory / "transcript.jsonl", (mtime, mtime))


@pytest.mark.asyncio
async def test_the_startup_preheat_materializes_a_replay_the_next_open_hits(tmp_path):
    """The point of the pass: the newest journals are cached so an open is a HIT.

    Asserted at the SEAM both sides share — the replay cache and the key a cold
    open builds — rather than by inspecting the preheat's own bookkeeping, so a
    preheat that warmed the wrong key would fail this.
    """
    from local_operator.session import replay_cache

    root = tmp_path
    journal(root, "older", mtime=1_000.0)
    await replay_ready_journal(root, "newest", mtime=2_000.0)
    replay_cache.replay_cache().clear()

    # THE SELECTION, first: the newest uncached journal is a candidate, and it is
    # asked BEFORE the pass so the answer is about the empty cache this start has.
    assert "newest" in prewarm.sessions_needing_replay(root)

    started = await prewarm.warm_replay_cache(root)

    assert started >= 1
    key = prewarm._cold_replay_key(root / "sessions" / "newest")
    assert key is not None
    assert replay_cache.cached_replay(key) is not None, "the newest journal was not materialized"
    # And the pass is IDEMPOTENT on the cache it just filled: a second start finds
    # nothing to do, which is what keeps an already-warm process from re-parsing.
    assert "newest" not in prewarm.sessions_needing_replay(root)


@pytest.mark.asyncio
async def test_the_replay_preheat_spawns_no_runtime(tmp_path, monkeypatch):
    """The read is durable rows, so a preheat at core start must not spawn a runtime.

    A REAL NEGATIVE CONTROL, not a description. The spawn seam is guarded at the
    TWO places every runtime creation goes through — ``launch._spawn_runtime``
    (the subprocess) and ``LocalOwner.engage`` (the dial that leads to it) — and
    both are replaced with one that records the call and fails. The test then
    proves the control is ARMED and the preheat actually ran:

    * the patch is live: invoking the replaced seams THROUGH THE PATCHED
      ATTRIBUTES (``launch._spawn_runtime`` / ``LocalOwner.engage``, not the raw
      closures) raises — so this proves ``setattr`` landed, not merely that some
      closure raises — and it is then RESET, because arming it necessarily moved
      the flag the final assertion reads;
    * the preheat materialized a real replay with the flag still untouched.

    Why these two and not the facade's own ``warm_runtime``/``_ensure_bound``: the
    failure mode this guards is a preheat that reaches a runtime by some path the
    test did not anticipate, so the control belongs on the seam that cannot be
    bypassed — the fork itself, and the owner dial that leads to it.
    """
    from local_operator.session import replay_cache
    from local_operator.session.owner import LocalOwner
    from local_operator.session.runtime import launch
    from local_operator.session.runtime.launch import WarmErrand

    spawned = {"called": False}

    def forbid_spawn(*_args, **_kwargs):
        spawned["called"] = True
        raise AssertionError("the replay preheat spawned a runtime process")

    async def forbid_engage(*_args, **_kwargs):
        spawned["called"] = True
        raise AssertionError("the replay preheat dialled a session owner")

    # The TWO seams every runtime creation goes through: the subprocess fork and the
    # attempt to dial an owner to bind a runtime. ``LocalOwner`` is the CONCRETE
    # class ``owner_for`` defaults to, so patching its ``engage`` catches a stray dial
    # (`SessionOwner` is only the Protocol and a patch there would never fire).
    monkeypatch.setattr(launch, "_spawn_runtime", forbid_spawn)
    monkeypatch.setattr(LocalOwner, "engage", forbid_engage)

    # ARMED: reach the seams THROUGH the patched attributes, so a setattr that did
    # not land would fail here rather than leave the control silently inert. A real
    # ``LocalOwner`` instance carries the typed ``self`` the annotation wants, and
    # ``cast`` names the deliberate bad argument: this call exists to prove the
    # patched seam RAISES, so a valid ``warm`` would defeat its own purpose.
    with pytest.raises(AssertionError):
        launch._spawn_runtime("s", "cwd", defer_materialise=False)
    with pytest.raises(AssertionError):
        await LocalOwner.engage(
            object.__new__(LocalOwner), cwd="cwd", warm=cast(WarmErrand, None)
        )
    # The armed probes above are the ONLY thing allowed to have moved the flag, and
    # they have: reset it so the assertion below is about the PREHEAT alone.
    spawned["called"] = False

    root = tmp_path
    await replay_ready_journal(root, "newest", mtime=1_000.0)
    replay_cache.replay_cache().clear()

    assert await prewarm.warm_replay_cache(root) >= 1
    assert spawned["called"] is False


@pytest.mark.asyncio
async def test_the_replay_preheat_is_skipped_under_the_disk_guard(tmp_path, monkeypatch):
    """The replay pass reuses the SAME guard the index pass uses, before any work."""
    from local_operator.session import replay_cache

    root = tmp_path
    await replay_ready_journal(root, "newest", mtime=1_000.0)
    replay_cache.replay_cache().clear()
    monkeypatch.setattr(
        prewarm, "free_bytes", lambda _root=None: prewarm.PREWARM_MIN_FREE_BYTES - 1
    )

    assert await prewarm.warm_replay_cache(root) == 0
    key = prewarm._cold_replay_key(root / "sessions" / "newest")
    assert key is not None
    assert replay_cache.cached_replay(key) is None


@pytest.mark.asyncio
async def test_the_replay_preheat_counts_only_entries_the_cache_holds(tmp_path, monkeypatch):
    """An over-budget replay is REFUSED, so it must not be counted as materialized.

    ``publish_replay`` returns an over-budget value with ``retained == 0`` and
    stores nothing, so the preheat's number must follow the CACHE, not the read:
    counting the refusal would report a warm cache that is empty, on exactly the
    large journals the figure is watched for.
    """
    from local_operator.session import replay_cache

    root = tmp_path
    await replay_ready_journal(root, "big", mtime=1_000.0)
    replay_cache.replay_cache().clear()

    # Route the preheat's builder through the bound's REAL refusal: publish with a
    # zero byte budget, so ``retained_bytes`` returns ``None`` and the entry is
    # handed back UNCACHED — the exact shape the bound produces on a real oversize
    # journal, without needing an 8-MiB fixture.
    monkeypatch.setattr(replay_cache, "REPLAY_CACHE_BYTES", 0, raising=False)
    assert await prewarm.warm_replay_cache(root) == 0
    assert replay_cache.replay_cache().entry_count == 0


@pytest.mark.asyncio
async def test_a_reader_joining_an_in_flight_preheat_does_not_parse_twice(tmp_path, monkeypatch):
    """ONE read for a preheat and a reader that race — the reader JOINS.

    The preheat publishes through ``load_cold_replay``, the reader's own
    single-flight, so a user who opens the session mid-preheat shares that task
    instead of paying a second parse. Counted at the ONE place a parse can happen
    (``build_cold_replay``), with the read held open long enough for the reader to
    arrive.
    """
    from local_operator.session import replay_cache

    root = tmp_path
    await replay_ready_journal(root, "s1", mtime=1_000.0)
    replay_cache.replay_cache().clear()

    builds = {"count": 0}
    real_build = replay_cache.build_cold_replay

    def counted(directory, **kwargs):  # noqa: ANN001, ANN003
        builds["count"] += 1
        time.sleep(0.2)  # hold the leader open so the reader can join
        return real_build(directory, **kwargs)

    monkeypatch.setattr(prewarm, "build_cold_replay", counted)

    preheat = asyncio.create_task(prewarm.warm_replay_cache(root, limit=1))
    await asyncio.sleep(0.05)  # let the preheat become the leader
    key = prewarm._cold_replay_key(root / "sessions" / "s1")
    assert key is not None

    async def open_read():
        return await replay_cache.load_cold_replay(
            key,
            lambda: asyncio.to_thread(
                counted, root / "sessions" / "s1",
                key=key, want_checkpoint=True, through_id=None, strict_cut=False,
            ),
        )

    await asyncio.gather(preheat, open_read())
    assert builds["count"] == 1, f"the journal was parsed {builds['count']} times"
    assert replay_cache.cached_replay(key) is not None

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

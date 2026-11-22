"""C5 of remote files parity: precache the media of active remote sessions.

docs/design/remote-files-parity.md §4 "Precache" and the C5 acceptance: the active
set (2 h window, the opened LRU, the cap of 16) on synthetic catalogue rows; a real
loopback pass that lands ONLY image/audio/video at most 32 MiB, respects the budget
and yields to an on-screen request; and the RAM check (10 x 16 MiB with an RSS
delta under 64 MiB).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import files
from local_operator.remote_files import cache as remote_cache
from local_operator.remote_files import fetch as remote_fetch
from local_operator.remote_files import precache
from local_operator.session.attachments import AttachmentStore
from tests.unit.network.test_mesh_file_pull import mesh  # noqa: F401 — fixture
from tests.unit.network.test_mobility import SESSION, devices, pair  # noqa: F401
from tests.unit.network.test_remote_files_fetch import viewer  # noqa: F401 — fixture
from tests.unit.session_files._fixtures import append_journal, assistant, call, tool

NOW = 1_800_000_000.0


def _row(sid: str, *, age_s: float, **overrides: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "mtime": NOW - age_s,
        "locality": "remote",
        "reachable": True,
        "archived": False,
        "owner_device": "d_owner",
        **overrides,
    }


def test_the_active_set_is_recent_reachable_unarchived_plus_opened_capped_at_16() -> None:
    rows = [_row(f"{i:012x}", age_s=60.0 * i) for i in range(30)]
    rows += [
        _row("a" * 12, age_s=60, archived=True),
        _row("b" * 12, age_s=60, reachable=False),
        _row("c" * 12, age_s=60, locality="local"),
        _row("d" * 12, age_s=10 * 3600),  # old, but opened here
        _row("e" * 12, age_s=10 * 3600),  # old and never opened
    ]
    chosen = [row["id"] for row in precache.active_set(rows, ["d" * 12], now=NOW)]
    assert chosen[0] == "d" * 12, "an opened session comes first whatever its age"
    assert len(chosen) == precache.MAX_SESSIONS == 16
    assert not {"a" * 12, "b" * 12, "c" * 12, "e" * 12} & set(chosen)
    assert chosen[1:] == [f"{i:012x}" for i in range(15)], "newest first"
    within = precache.active_set(rows[:3] + [_row("f" * 12, age_s=2 * 3600 + 1)], [], now=NOW)
    assert "f" * 12 not in [row["id"] for row in within], "the window is 2 h"


def test_a_session_is_re_listed_only_when_it_moved_and_at_most_once_a_minute(
    tmp_path: Path,
) -> None:
    worker = precache.Precacher(tmp_path)
    queued: list[str] = []
    worker._enqueue = lambda row: queued.append(row["id"])  # type: ignore[method-assign]
    row = _row("1" * 12, age_s=0)
    row["mtime"] = time.time()
    worker.observe([row])
    worker.passes[row["id"]] = (time.monotonic(), row["mtime"])
    worker.observe([row])  # unchanged mtime
    worker.observe([{**row, "mtime": row["mtime"] + 1}])  # moved, but within 60 s
    worker.passes[row["id"]] = (time.monotonic() - 61, row["mtime"])
    worker.observe([{**row, "mtime": row["mtime"] + 1}])  # moved, past 60 s
    assert queued == ["1" * 12, "1" * 12]
    (tmp_path / "config.yml").write_text("desktop:\n  remote_files_cache_bytes: 0\n")
    worker.observe([{**row, "mtime": row["mtime"] + 5}])
    assert len(queued) == 2, "a 0 budget turns precache off"


def _pad_media(server: Any, sizes: dict[str, int]) -> dict[str, bytes]:
    """Pad files written by the session (path rows), with their bytes."""
    directory = Path(server.root) / "sessions" / SESSION
    pad = directory / "scratchpad"
    pad.mkdir(parents=True, exist_ok=True)
    (directory / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(pad)}))
    written: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for index, (name, size) in enumerate(sizes.items()):
        payload = os.urandom(size)
        (pad / name).write_bytes(payload)
        written[name] = payload
        rows += [
            assistant(
                f"pa{index}", 10.0 + index, [call(f"pc{index}", "write", path=str(pad / name))]
            ),
            tool(
                f"pt{index}", 10.5 + index, f"pc{index}", "write", "ok", {"path": str(pad / name)}
            ),
        ]
    append_journal(directory, rows)
    return written


def _catalogue_row(server_a: Any) -> dict[str, Any]:
    return _row(SESSION, age_s=0, mtime=time.time(), owner_device=server_a.identity.device_id)


def _cached(server_b: Any, payload: bytes) -> bool:
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()[:32]
    return (remote_cache.cache_root(Path(server_b.root)) / f"{digest}.bin").exists()


@pytest.mark.asyncio
async def test_a_pass_lands_only_media_within_the_per_file_ceiling(viewer) -> None:  # noqa: F811
    _client, (server_a, server_b, *_rest) = viewer
    written = _pad_media(
        server_a,
        {
            "photo.png": 200_000,
            "clip.wav": 300_000,
            "movie.mp4": 1_000_000,
            "notes.txt": 50_000,
            "huge.mp4": precache.PER_FILE_MAX_BYTES + 1,
        },
    )
    worker = precache.Precacher(Path(server_b.root))
    landed = await worker.prepare(_catalogue_row(server_a))
    for name in ("photo.png", "clip.wav", "movie.mp4"):
        assert _cached(server_b, written[name]), name
    assert not _cached(server_b, written["notes.txt"]), "a document is not media"
    assert not _cached(server_b, written["huge.mp4"]), "over the per-file ceiling"
    assert landed == 1_500_000
    assert not list((Path(server_b.root) / "attachments").glob("*.bin"))


@pytest.mark.asyncio
async def test_a_pass_stops_at_the_budget_and_evicts_nothing_it_landed(
    viewer,  # noqa: F811
) -> None:
    _client, (server_a, server_b, *_rest) = viewer
    written = _pad_media(server_a, {f"v{i}.mp4": 400_000 for i in range(4)})
    (Path(server_b.root) / "config.yml").write_text(
        "desktop:\n  remote_files_cache_bytes: 1000000\n"
    )
    worker = precache.Precacher(Path(server_b.root))
    await worker.prepare(_catalogue_row(server_a))
    present = [name for name, payload in written.items() if _cached(server_b, payload)]
    assert len(present) == 2, present
    assert remote_cache.total_bytes(Path(server_b.root)) <= 1_000_000
    assert len(worker.landed) == 2


async def _on_screen_while_precaching(viewer_fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    client, (server_a, server_b, *_rest) = viewer_fixture
    written = _pad_media(server_a, {f"slow{i}.mp4": 300_000 for i in range(5)})
    worker = precache.Precacher(Path(server_b.root))
    row = _catalogue_row(server_a)
    candidates = await worker.candidates(row)
    assert len(candidates) == 5
    batches = iter((candidates[:1], candidates[1:]))

    async def independent_candidates(_row: dict[str, Any]) -> Any:
        # Two independent passes share the REAL per-owner precache gate. One
        # serial prepare cannot expose a regression that gives background work
        # both owner slots. Keep real, disjoint candidate identities/sources.
        return next(batches)

    monkeypatch.setattr(worker, "candidates", independent_candidates)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    competing = asyncio.Event()
    release = threading.Event()
    transfer = files._transfer

    def slow(asker: Any, fd: int, meta: dict[str, Any], start: int) -> None:
        if str(meta.get("name", "")).startswith("slow"):
            # Publication happens INSIDE the transfer thread: a pending fetch
            # could still be waiting for its owner slot and proves nothing.
            loop.call_soon_threadsafe(entered.set)
            release.wait()
        transfer(asker, fd, meta, start)

    monkeypatch.setattr(files, "_transfer", slow)
    semaphore = worker._semaphore(row["owner_device"])
    attempts = admissions = 0

    @asynccontextmanager
    async def observe_admission():
        nonlocal attempts, admissions
        attempts += 1
        if attempts == 2:
            competing.set()
        # Event.set does not yield: on resumption the observer sees either a
        # blocked acquire or an admitted second pass, never an untried gate.
        async with semaphore:
            admissions += 1
            yield

    monkeypatch.setattr(worker, "_semaphore", lambda _owner: observe_admission())
    owned: list[asyncio.Task[Any]] = []
    fetches_before = set(remote_fetch._TASKS.values())
    # Failure-only watchdog, based on the existing route wait; no elapsed-time
    # acceptance bound. Healthy HTTP must finish while release is still unset.
    watchdog = 2 * remote_fetch.REMOTE_FETCH_WAIT_S
    try:
        owned.append(asyncio.create_task(worker.prepare(row)))
        await asyncio.wait_for(entered.wait(), watchdog)
        owned.append(asyncio.create_task(worker.prepare(row)))
        await asyncio.wait_for(competing.wait(), watchdog)
        assert admissions == 1, "background work consumed the reserved owner slot"

        # Register only AFTER the slow transfer entered and both candidate
        # snapshots were fixed: cache hits/single-flight cannot satisfy HTTP.
        on_screen = os.urandom(150_000)
        ref = AttachmentStore(Path(server_a.root) / "attachments").put_bytes(on_screen, "image/png")
        assert ref is not None
        append_journal(
            Path(server_a.root) / "sessions" / SESSION,
            [{"id": "os1", "content": [{"attachment": ref.digest}]}],
        )
        assert not _cached(server_b, on_screen)
        request = asyncio.create_task(
            client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{ref.digest}")
        )
        owned.append(request)
        response = await asyncio.wait_for(asyncio.shield(request), watchdog)
        assert response.status_code == 200, "on-screen HTTP did not finish before release"
        assert response.content == on_screen
        assert not release.is_set()
    finally:
        # Cancelling prepare does NOT stop shielded fetches or to_thread. Unblock
        # first, then drain our passes/HTTP AND their pulls before fixture teardown.
        release.set()
        await asyncio.gather(*owned, return_exceptions=True)
        pulls = set(remote_fetch._TASKS.values()) - fetches_before
        await asyncio.gather(*pulls, return_exceptions=True)
    assert len(worker.landed) == 5
    assert all(_cached(server_b, payload) for payload in written.values())


@pytest.mark.asyncio
async def test_an_on_screen_request_completes_while_the_precache_queue_is_full(
    viewer, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    await _on_screen_while_precaching(viewer, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_gate", ["owner", "precache"])
async def test_the_capacity_oracle_rejects_broken_gates(
    viewer, monkeypatch: pytest.MonkeyPatch, broken_gate: str  # noqa: F811
) -> None:
    if broken_gate == "owner":
        monkeypatch.setattr(remote_fetch, "OWNER_CONCURRENCY", 1)
        failure = "on-screen HTTP did not finish before release"
    else:
        monkeypatch.setattr(precache, "PRECACHE_PER_OWNER", 2)
        failure = "background work consumed the reserved owner slot"
    with pytest.raises(AssertionError, match=failure):
        await _on_screen_while_precaching(viewer, monkeypatch)


def _rss_bytes() -> int:
    with open("/proc/self/statm", encoding="ascii") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="needs /proc for RSS")
@pytest.mark.asyncio
async def test_precaching_160_mib_keeps_the_daemon_rss_flat(viewer) -> None:  # noqa: F811
    """§4's RAM check: 10 x 16 MiB precached with an RSS delta under 64 MiB. Both relays
    run in THIS process, so the delta also covers the owner's serving side."""
    _client, (server_a, server_b, *_rest) = viewer
    _pad_media(server_a, {f"big{i}.mp4": 16 * 1024 * 1024 for i in range(10)})
    worker = precache.Precacher(Path(server_b.root))
    before = _rss_bytes()
    landed = await worker.prepare(_catalogue_row(server_a))
    after = _rss_bytes()
    assert landed == 160 * 1024 * 1024
    assert after - before < 64 * 1024 * 1024, f"RSS grew {(after - before) / 2**20:.1f} MiB"
